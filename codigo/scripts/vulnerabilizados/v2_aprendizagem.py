"""Biblioteca da v2: bracos A0-A4, conjuntos de retencao, confiabilidade e aquisicao. Python 3.12.

A0  -HAND, sem ajuste.
A1  doadores externos sul-americanos (GFD + componente EMSR720), sem rotulo do destino; A1i: pool internacional.
A2  so rotulos brasileiros de desenvolvimento, orcamento B, prior N(0, 1/LAMBDA0) (equivale a C=1).
A3  mesmo orcamento B, prior N(coef_A1, 1/LAMBDA_A3): o conhecimento externo entra como regularizacao
    informada (reaproveita pesos), nao como refit escondido.
A4  atualizacao de pesos a partir de A3 com dados novos, por descida de gradiente com passos fixos,
    com e sem replay (mesmo numero de passos e mesmo tamanho de lote): mede ganho e esquecimento.
Escalonamento fixo: media/desvio dos preditores no pool A1 (sem rotulo), igual para todos os bracos.
Pesos balanceados por classe em todos os ajustes; o escore e logito de decisao, nao probabilidade.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tv_transferencia as tt
from comum_tv import OUT as V1, auc_mann_whitney
from v2_00_partida import V2

NUCLEO = ["l_hand", "l_slope", "twi"]
COM_CONSTRUIDO = NUCLEO + ["construido"]
COM_CANAL = NUCLEO + ["construido", "l_canal"]
LAMBDA0 = 1.0
LAMBDA_A3 = 25.0          # prior equivalente a ~100 observacoes balanceadas (informacao ~0,25 por obs)
ORCAMENTOS = [0, 25, 50, 100, 200, 400, 800, 1600]
SEMENTES = list(range(20261101, 20261111))
N_RETENCAO = 5


def transformar(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["l_hand"] = np.log1p(df.hand_m.clip(lower=0))
    out["l_slope"] = np.log1p(df.slope_deg.clip(lower=0))
    out["twi"] = df.twi_dinf.clip(0, 40)
    if "construido" in df:
        out["construido"] = df.construido.astype(float)
    if "dist_canal_m" in df:
        out["l_canal"] = np.log1p(df.dist_canal_m.clip(lower=0))
    return out


def _h(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


# ------------------------------------------------------------------ doadores
def doadores(alvo_lon, alvo_lat) -> tuple[dict, dict]:
    """Pools externos retidos para o destino + conjuntos de retencao fixos por hash (fora do ajuste)."""
    from tv09_manaus import pools
    d = tt.carregar_doadores()
    wc = pd.read_parquet(tt.DS06, columns=["fonte", "evento_id", "classe", "lat", "lon", "worldcover_2021"])
    wc = wc[wc.fonte.isin(["gfd", "cems"]) & wc.classe.isin([0, 1])]
    d = d.merge(wc[["evento_id", "lat", "lon", "worldcover_2021"]].drop_duplicates(["evento_id", "lat", "lon"]),
                on=["evento_id", "lat", "lon"], how="left")
    d["construido"] = (d.worldcover_2021 == 50).astype(float)
    alvo = pd.DataFrame({"lon": alvo_lon, "lat": alvo_lat})
    ps, info = pools(d, alvo)
    out, ret = {}, {}
    for nome, p in ps.items():
        unidades = sorted(p.unidade_doadora.unique(), key=lambda u: _h(f"tv-v2-ret|{u}"))
        r = unidades[:N_RETENCAO]
        ret[nome] = p[p.unidade_doadora.isin(r)].reset_index(drop=True)
        out[nome] = p[~p.unidade_doadora.isin(r)].reset_index(drop=True)
        info[f"retencao_{nome}"] = [str(u) for u in r]
    return {"ajuste": out, "retencao": ret}, info


def amostra_doadores(p: pd.DataFrame, semente: int) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    idx = [rng.choice(q.index.to_numpy(), size=min(tt.TETO_POR_CLASSE, len(q)), replace=False)
           for _, q in p.groupby(["unidade_doadora", "classe"])]
    return p.loc[np.concatenate(idx)]


# ------------------------------------------------------------------ logistica com prior
class Escala:
    def __init__(self, x: pd.DataFrame):
        self.mu, self.sd = x.mean(), x.std(ddof=0).replace(0, 1)

    def __call__(self, x: pd.DataFrame) -> np.ndarray:
        return ((x[self.mu.index] - self.mu) / self.sd).to_numpy()


def _pesos(y):
    n1 = y.sum()
    n0 = len(y) - n1
    return np.where(y == 1, len(y) / (2 * max(n1, 1)), len(y) / (2 * max(n0, 1)))


def ajustar(z: np.ndarray, y: np.ndarray, media=None, precisao=LAMBDA0) -> np.ndarray:
    """MAP da logistica balanceada com prior gaussiano nos coeficientes (intercepto livre). Devolve [coef..., b]."""
    from scipy.optimize import minimize
    y = np.asarray(y, float)
    w = _pesos(y)
    k = z.shape[1]
    m0 = np.zeros(k) if media is None else np.asarray(media, float)

    def perda(t):
        b, c = t[:k], t[k]
        s = z @ b + c
        p = 1 / (1 + np.exp(-s))
        g = w * (p - y)
        return (np.sum(w * (np.logaddexp(0, s) - y * s)) + .5 * precisao * np.sum((b - m0) ** 2),
                np.r_[z.T @ g + precisao * (b - m0), g.sum()])
    if y.min() == y.max():   # sem as duas classes: so o prior decide a ordenacao
        return np.r_[m0, 0.0]
    r = minimize(perda, np.r_[m0, 0.0], jac=True, method="L-BFGS-B")
    return r.x


def atualizar(theta, z, y, passos=200, taxa=.05):
    """A4: descida de gradiente a partir de theta, passos fixos (orcamento computacional pareado)."""
    y = np.asarray(y, float)
    w = _pesos(y) / len(y)
    t = np.array(theta, float)
    k = z.shape[1]
    for _ in range(passos):
        s = z @ t[:k] + t[k]
        g = w * (1 / (1 + np.exp(-s)) - y)
        t[:k] -= taxa * (z.T @ g)
        t[k] -= taxa * g.sum()
    return t


def escore(theta, z):
    return z @ theta[:-1] + theta[-1]


# ------------------------------------------------------------------ dados brasileiros
def brasil(nome: str) -> pd.DataFrame:
    """Pontos rotulados de desenvolvimento (RMPA, Manaus) ou de Curitiba, com covariaveis v2."""
    if nome == "taquari":
        p = pd.read_parquet(V1 / "desenvolvimento_taquari/pontos.parquet").drop_duplicates("ponto_id")
        return p.assign(territorio="taquari", grupo="AOI:" + p.aoi_id, municipio=p.CD_MUN.fillna("-"),
                        classe_territorial=p.classe_territorial)[["ponto_id", "territorio", "y", "hand_m", "slope_deg", "twi_dinf",
                                                                   "lon", "lat", "grupo", "municipio", "classe_territorial"]]
    pasta = {"rmpa": V1 / "confirmacao_rmpa", "manaus": V1 / "confirmacao_manaus", "curitiba": V2 / "curitiba"}[nome]
    g = pd.read_parquet(pasta / "grade.parquet")
    c = pd.read_parquet(V2 / f"covariaveis/{nome}.parquet")
    r = pd.read_parquet(pasta / "referencia.parquet", columns=["ponto_id", "estado_primaria"])
    d = g[g.terreno_valido].merge(c, on="ponto_id").merge(r, on="ponto_id")
    d = d[d.estado_primaria.isin(["positivo", "negativo"]) & d.dist_canal_m.notna()].copy()
    d["y"] = d.estado_primaria.eq("positivo").astype(int)
    d["territorio"] = nome
    d["grupo"] = np.where(d.classe_territorial.eq("fcu"), "FCU:" + d.CD_FCU.astype(str), "SET:" + d.CD_SETOR.astype(str))
    d["municipio"] = d.CD_MUN.astype(str)
    return d.reset_index(drop=True)


def particao_municipio(cod: str, fracao=.4) -> str:
    return "adaptacao" if _h(f"tv-v2|MUN:{cod}") < fracao else "teste"


def orcamento(df: pd.DataFrame, b: int, semente: int, modo="aleatorio", prioridade=None) -> pd.DataFrame:
    """Seleciona B pontos para rotular sem olhar o rotulo: aleatorio, estratificado por decil de HAND ou por incerteza."""
    if b == 0:
        return df.iloc[0:0]
    rng = np.random.default_rng(semente)
    b = min(b, len(df))
    if modo == "aleatorio":
        return df.iloc[rng.choice(len(df), b, replace=False)]
    if modo == "estratificado":
        dec = pd.qcut(df.hand_m.rank(method="first"), 10, labels=False)
        por = [df.index[dec == k] for k in range(10)]
        escolha = np.concatenate([rng.choice(p, size=min(len(p), b // 10 + (k < b % 10)), replace=False) for k, p in enumerate(por)])
        return df.loc[escolha]
    if modo == "incerteza":
        ordem = np.argsort(-np.asarray(prioridade) + rng.uniform(0, 1e-9, len(df)))
        return df.iloc[ordem[:b]]
    raise ValueError(modo)


# ------------------------------------------------------------------ confiabilidade
def desordem(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Contribuicao de cada ponto para 1-AUC: positivo -> fracao de negativos com escore >= ; negativo -> fracao de positivos <=."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    out = np.full(len(y), np.nan)
    if y.min() == y.max():
        return out
    neg = np.sort(s[y == 0])
    pos = np.sort(s[y == 1])
    sp, sn = s[y == 1], s[y == 0]
    acima = len(neg) - np.searchsorted(neg, sp, side="right")
    empate = np.searchsorted(neg, sp, side="right") - np.searchsorted(neg, sp, side="left")
    out[y == 1] = (acima + .5 * empate) / len(neg)
    abaixo = np.searchsorted(pos, sn, side="left")
    empate = np.searchsorted(pos, sn, side="right") - np.searchsorted(pos, sn, side="left")
    out[y == 0] = (abaixo + .5 * empate) / len(pos)
    return out


def atributos_confiabilidade(df: pd.DataFrame, s_a1, s_a1i, s_a0, maha) -> pd.DataFrame:
    """So informacao disponivel na inferencia: dominio, desacordo entre modelos, terreno e cobertura."""
    q = pd.DataFrame(index=df.index)
    m = df.municipio
    q["pct_a1"] = pd.Series(s_a1, index=df.index).groupby(m).rank(pct=True)
    q["desacordo_hand"] = (q.pct_a1 - pd.Series(s_a0, index=df.index).groupby(m).rank(pct=True)).abs()
    q["desacordo_pools"] = (q.pct_a1 - pd.Series(s_a1i, index=df.index).groupby(m).rank(pct=True)).abs()
    q["dominio"] = maha
    t = transformar(df)
    for c in ["l_hand", "l_slope", "twi"]:
        q[c] = t[c]
    q["construido"] = df.construido.astype(float) if "construido" in df else 0.0
    return q


def mahalanobis(x_ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    mu = x_ref.mean(0)
    inv = np.linalg.pinv(np.cov(x_ref, rowvar=False))
    d = x - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))


def auc_aceitos(y, s, risco, cobertura, rng=None, reps=0):
    """AUC nos pontos aceitos ao abster os de maior risco; com reps>0 devolve a media de abstencoes aleatorias."""
    n = len(y)
    k = int(round(cobertura * n))
    if reps:
        vals = []
        for _ in range(reps):
            a = rng.choice(n, k, replace=False)
            v = auc_mann_whitney(y[a], s[a])
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None
    a = np.argsort(np.asarray(risco) + np.random.default_rng(0).uniform(0, 1e-9, n))[:k]
    return auc_mann_whitney(y[a], s[a])
