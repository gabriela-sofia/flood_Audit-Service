"""Funcoes da v5: ajuste MAP, escalas, selecao interna de regularizacao e metricas por comunidade.

O que cada braco reutiliza (registrado no protocolo):
  A0  nada aprendido (-HAND).
  A1  pesos e escala dos doadores; nenhum rotulo brasileiro.
  A2_fraco / A2_forte  so a ESCALA dos doadores (media/desvio dos atributos); coeficientes com prior 0
      (precisao 1 / 25). Nao e ausencia absoluta de informacao externa.
  A2_estrito  escala aprendida so nos B exemplos de adaptacao; prior 0, precisao 1. Sem informacao externa.
  A2_cv       escala dos doadores; precisao escolhida por validacao agrupada dentro dos B exemplos.
  A3          pesos dos doadores como prior (precisao 25) + escala dos doadores; mesmos B exemplos de A2.
  A3_fraco    idem com precisao 1 (mesma forca de A2_fraco).
  A3_cv       prior dos doadores; precisao por validacao interna.
Verossimilhanca nao balanceada na adaptacao (v2/v3); A1 balanceado nos doadores (v3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

import v2_aprendizagem as ap
from comum_tv import auc_mann_whitney

GRADE_PRECISAO = [0.25, 1.0, 5.0, 25.0, 100.0]


def ajustar(z, y, prior=None, precisao=1.0, balancear=False):
    y = np.asarray(y, float)
    k = z.shape[1]
    m = np.zeros(k) if prior is None else np.asarray(prior, float)
    if len(np.unique(y)) < 2:
        return np.r_[m, 0.0], False
    w = np.ones(len(y))
    if balancear:
        w = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))

    def f(t):
        s = z @ t[:-1] + t[-1]
        e = w * (expit(s) - y)
        return (np.sum(w * (np.logaddexp(0, s) - y * s)) + .5 * precisao * np.sum((t[:-1] - m) ** 2),
                np.r_[z.T @ e + precisao * (t[:-1] - m), e.sum()])
    r = minimize(f, np.r_[m, 0.0], jac=True, method="L-BFGS-B", options={"maxiter": 500})
    if not r.success:
        raise RuntimeError("sem convergencia: " + str(r.message))
    return r.x, True


def escore(t, z):
    return z @ t[:-1] + t[-1]


class EscalaLocal:
    """Escala estimada apenas nos exemplos de adaptacao (controle estritamente local)."""

    def __init__(self, x: pd.DataFrame):
        self.mu, self.sd = x.mean(), x.std(ddof=0).replace(0, 1).fillna(1)

    def __call__(self, x: pd.DataFrame):
        return ((x[self.mu.index] - self.mu) / self.sd).to_numpy()


def escolher_precisao(z, y, grupos, prior=None, k=3, semente=0):
    """Validacao agrupada (grupos inteiros por dobra) dentro dos B exemplos; devolve (precisao, estado)."""
    y = np.asarray(y, int)
    g = np.asarray(grupos)
    unicos = np.array(sorted(set(g)))
    rng = np.random.default_rng(semente)
    dobra_de = dict(zip(unicos, rng.permutation(len(unicos)) % k))
    dobras = np.array([dobra_de[x] for x in g])
    medias = {}
    for p in GRADE_PRECISAO:
        vals = []
        for d in range(k):
            tr, va = dobras != d, dobras == d
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            t, _ = ajustar(z[tr], y[tr], prior, p)
            vals.append(auc_mann_whitney(y[va], escore(t, z[va])))
        if len(vals) >= 2:
            medias[p] = float(np.mean(vals))
    if not medias:
        return (25.0 if prior is not None else 1.0), "sem_dobras_validas_valor_fixo"
    melhor = max(medias.values())
    return min(p for p, v in medias.items() if v == melhor), "validacao_interna"


def auc_por_grupo(y, s, grupos, minimo=5):
    """AUC por grupo com >= minimo de cada classe; devolve Series indexada pelo grupo."""
    df = pd.DataFrame({"y": np.asarray(y, int), "s": np.asarray(s, float), "g": np.asarray(grupos)})
    df["r"] = df.groupby("g").s.rank(method="average")
    df["ry"] = df.r * df.y
    a = df.groupby("g").agg(n=("y", "size"), n1=("y", "sum"), r1=("ry", "sum"))
    a["n0"] = a.n - a.n1
    a = a[(a.n1 >= minimo) & (a.n0 >= minimo)]
    return ((a.r1 - a.n1 * (a.n1 + 1) / 2) / (a.n1 * a.n0)).astype(float)


def alerta_por_municipio(s, municipios, fracao=.2):
    """Topo `fracao` do escore dentro de cada municipio do conjunto avaliado; nao usa rotulo."""
    r = pd.Series(np.asarray(s, float)).groupby(np.asarray(municipios)).rank(pct=True, ascending=False)
    return (r <= fracao).to_numpy()


def transformar(df, cols):
    return ap.transformar(df)[cols]
