"""Explica um ponto reservado por contribuicoes exatas do modelo linear.

O dominio multivariado usa a distancia de Mahalanobis e limiar empirico
do treino. Isso e diagnostico de extrapolacao, nao prova de transferencia.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def bootstrap_treino(x, y, grupos, ajustar, n=200, semente=20261001) -> dict:
    """Reajusta grupos inteiros; compara coeficientes na mesma escala.

    Replicas guardam sua propria padronizacao para consultas individuais.
    Este IC mede incerteza do ajuste, nao ruido de uma ocorrencia futura.
    """
    x, y, grupos = np.asarray(x, float), np.asarray(y, int), np.asarray(grupos)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(grupos):
        raise ValueError("treino e grupos desalinhados")
    unicos = np.unique(grupos)
    indices = {g: np.flatnonzero(grupos == g) for g in unicos}
    rng = np.random.default_rng(semente)
    escala = x.std(axis=0)
    replicas, coeficientes = [], []
    falhas = {"uma_classe": 0, "escala_invalida": 0, "ajuste": 0}
    if not len(unicos) or not np.isfinite(x).all() or (escala <= 0).any():
        return {"estado": "not_testable", "motivo": "treino ou escala invalida", "replicas": [],
                "ic95_coef_escala_original_do_treino": []}
    for _ in range(n):
        i = np.concatenate([indices[g] for g in rng.choice(unicos, len(unicos), replace=True)])
        if len(np.unique(y[i])) != 2:
            falhas["uma_classe"] += 1
            continue
        mu, sd = x[i].mean(axis=0), x[i].std(axis=0)
        if (sd <= 0).any():
            falhas["escala_invalida"] += 1
            continue
        try:
            m = ajustar((x[i] - mu) / sd, y[i])
            coef = np.asarray(m.coef_).ravel()
            intercepto = float(np.asarray(m.intercept_).ravel()[0])
            if not np.isfinite(coef).all() or not np.isfinite(intercepto):
                raise ValueError("ajuste nao finito")
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            falhas["ajuste"] += 1
            continue
        replicas.append({"mu": mu.tolist(), "sd": sd.tolist(), "coef": coef.tolist(),
                         "intercepto": intercepto})
        coeficientes.append((coef / sd * escala).tolist())
    suficiente = len(replicas) >= max(100, int(np.ceil(.8 * n)))
    return {"estado": "executado" if suficiente else "inconclusive", "solicitadas": n,
        "validas": len(replicas), "falhas": falhas, "unidade": "grupo_independente_do_treino",
        "ic95_coef_escala_original_do_treino": np.percentile(coeficientes, [2.5, 97.5], axis=0).T.tolist()
            if suficiente else [], "replicas": replicas}


def incerteza_ponto(bootstrap: dict, x: np.ndarray) -> dict:
    if bootstrap.get("estado") != "executado":
        return {"estado": "inconclusive", "motivo": "bootstrap de treino insuficiente"}
    valores = [float(expit(r["intercepto"] + ((x - np.asarray(r["mu"])) /
        np.asarray(r["sd"])) @ np.asarray(r["coef"]))) for r in bootstrap["replicas"]]
    return {"estado": "executado", "ic95_escore_amostral": np.percentile(valores, [2.5, 97.5]).tolist(),
        "replicas_validas": len(valores), "unidade": bootstrap["unidade"],
        "limite": "incerteza do ajuste; nao intervalo de ocorrencia futura nem probabilidade absoluta"}


def dominio(treino: np.ndarray, ponto: np.ndarray) -> dict:
    if treino.ndim != 2 or treino.shape[0] < 3:
        return {"estado": "not_testable", "motivo": "treino insuficiente"}
    centro = treino.mean(axis=0)
    cov = np.atleast_2d(np.cov(treino, rowvar=False))
    posto = np.linalg.matrix_rank(cov)
    if posto != treino.shape[1]:
        return {"estado": "not_testable", "motivo": "covariancia singular"}
    inversa = np.linalg.inv(cov)
    delta = treino - centro
    distancias = np.einsum("ij,jk,ik->i", delta, inversa, delta)
    distancia = float((ponto - centro) @ inversa @ (ponto - centro))
    limiar = float(np.quantile(distancias, 0.99))
    return {"estado": "dentro" if distancia <= limiar else "fora",
            "distancia_quadratica": distancia, "limiar_99_treino": limiar,
            "metodo": "Mahalanobis empirico; covariancia ajustada apenas no treino"}


def responder(modelo: dict, ponto: dict, treino: np.ndarray,
              compatibilidade: dict, suporte: str, protocolo: str) -> dict:
    features = modelo["features"]
    corpo = {"tipo": "resposta_individual_suscetibilidade", "protocolo": protocolo,
             "estado": "recusado", "estimativa": None,
             "nao_e": "nao e causa do evento, probabilidade absoluta ou previsao de data"}
    exigencias = ("cadeia", "mecanismo", "resolucao", "referencia_drenagem")
    if not all(compatibilidade.get(k) is True for k in exigencias):
        return {**corpo, "motivo": "compatibilidade nao demonstrada", "compatibilidade": compatibilidade}
    if suporte != "supported":
        return {**corpo, "motivo": "transferencia para o alvo nao sustentada"}
    try:
        x = np.array([ponto[f] for f in features], dtype=float)
    except (KeyError, TypeError, ValueError):
        return {**corpo, "motivo": "preditores ausentes ou invalidos"}
    if not np.isfinite(x).all():
        return {**corpo, "motivo": "preditores nao finitos"}
    mu, sd, coef = (np.asarray(modelo[k], dtype=float) for k in ("mu", "sd", "coef"))
    if any(a.shape != x.shape or not np.isfinite(a).all() for a in (mu, sd, coef)) or (sd <= 0).any():
        return {**corpo, "motivo": "parametros de padronizacao ou modelo invalidos"}
    dom = dominio(treino, x)
    if dom["estado"] != "dentro":
        return {**corpo, "motivo": "dominio multivariado insuficiente", "dominio": dom}
    z = (x - mu) / sd
    contribuicoes = z * coef
    intercepto = modelo["intercepto"]
    logito = float(intercepto + contribuicoes.sum())
    return {**corpo, "estado": "explicacao_do_modelo_disponivel", "dominio": dom,
        "estimativa": {"valor": float(expit(logito)), "semantica": "escore logistico amostral; nao probabilidade territorial"},
        "explicacao_individual": {"referencia": "media do treino", "intercepto": intercepto,
            "logito": logito, "contribuicoes_logito": dict(zip(features, contribuicoes.tolist())),
            "valores_medidos": dict(zip(features, x.tolist()))},
        "incerteza_individual": incerteza_ponto(modelo["bootstrap_treino"], x)
            if "bootstrap_treino" in modelo else {"estado": "untested",
                "motivo": "bootstrap preditivo deste modelo ainda nao executado"},
        "limites": ["contribuicao explica modelo, nao causalidade", "DEM historico nao demonstrado",
                    "IC da AUC nao e IC individual", "modelo experimental sem promocao"]}
