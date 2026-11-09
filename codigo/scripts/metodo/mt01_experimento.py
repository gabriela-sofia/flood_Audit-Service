"""MT01, passos 4-6: nucleo fisico compartilhado x resposta condicionada x compartilhamento parcial.

Pergunta: quais relacoes aprendidas em territorios observados ordenam a inundacao
observada em territorios retidos, sem nenhum rotulo do destino no ajuste; sob que
modificadores de contexto; e quanto rotulo local muda isso.

Unidade de retencao: componente do grafo de dependencia (`mt01_independencia`,
desenho primario: evento, lugar, onda de cheia no mesmo MAIN_BAS, sistema de
chuva). Cada destino retem o componente inteiro; o ajuste nunca ve evento, lugar,
sub-bacia ou onda dele.

Modelos (todos com intercepto por evento nao penalizado: a pergunta e ordenar
DENTRO do evento; no destino o intercepto e desconhecido e nao entra na ordem):

  A0_regra_hand         -HAND, sem ajuste
  A1_regra_composta     -HAND -elev_local +ln_SCA -declive (padronizados), sentidos fisicos
                        escritos antes, pesos iguais, sem ajuste
  G_geral               logistica L2, efeitos principais dos tres blocos
  C_condicionado        G + interacoes de cada efeito com modificadores de contexto
                        sem rotulo (escala do rio, relevo do territorio, bacia costeira)
  H_parcial             C + inclinacoes por territorio de treino (efeito aleatorio MAP);
                        no destino a inclinacao propria e zero (nao ha como estima-la)
  G_sem_<bloco>         ablacoes de G
  G_sem_mesma_fonte     G sem nenhuma linha da fonte do destino (atalho de instrumento)
  D_contexto/D_proximo/D_aleatorio  doadores com MESMO numero de eventos e teto por classe
  curva_U<k>            G com k territorios de treino sorteados (quantidade de observacao)
  poucos_rotulos        braco SEPARADO: n pontos rotulados de um evento do destino ajustam
                        so a inclinacao local (prior = H); avaliado nos OUTROS eventos

Selecao de penalidades so em dobras internas por territorio de treino. Nada e
promovido; svc02, escore servido, portoes e maturidade intocados.

Uso:
    python scripts/metodo/mt01_experimento.py pre_registrar
    python scripts/metodo/mt01_experimento.py executar [--so N]
    python scripts/metodo/mt01_experimento.py verificar
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for nome in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(nome, "2")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402
import mt01_populacao as mp  # noqa: E402

VERSAO = "mt01_experimento_v1"
SAIDA = mp.SAIDA
PRINCIPAIS = {"acumulacao": ["f_hand", "f_elev"], "conexao": ["f_sca", "f_declive"],
              "producao": ["f_construido", "f_areia"]}
FLAGS: list = []  # ensaio de 10/10: a flag "solo ausente" pegou +0,51 -- SoilGrids falta sobre agua/area
# construida, entao a ausencia marca o rotulo. Sem flags: imputacao pela mediana do treino.
CONTEXTO = ["z_rio", "z_relevo", "z_hand90", "z_costa"]
SENTIDO_REGRA = {"f_hand": -1, "f_elev": -1, "f_sca": 1, "f_declive": -1}
CONFIG = {
    "fontes": list(mp.FONTES_CONTRASTE), "papeis": ["positivo_episodico", "negativo_observado"],
    "desenho": "primario", "semente": 20261010,
    "teto_ajuste_por_evento_classe": 400, "teto_teste_por_evento_classe": 2000,
    "minimo_teste_evento": 20, "dobras_internas": 3,
    "grade_lam_principal": [0.0001, 0.001, 0.01, 0.1], "grade_lam_interacao": [0.01, 0.1, 1.0, 10.0, 100.0],
    "grade_lam_territorio": [0.01, 0.1, 1.0],
    "k_doadores": 6, "eventos_doadores_max": 20, "curva_territorios": [3, 6, 12], "sementes_sorteio": 3,
    "poucos_rotulos_n": [20, 100, 500], "margem_auc": 0.01, "bootstrap": 2000,
    "topo_inspecao": [0.1, 0.2],
    "sensibilidades": {
        "jrc": "so bracos A0/A1/G/C/H; tira pontos com occurrence JRC v1.4 >= 50 (contrato gfd02); NaN fica",
        "exutorio": "so bracos A0/A1/G/C/H; unidades pelo desenho conservador do mesmo exutorio (MAIN_BAS em qualquer data)",
    },
    "corte_jrc": 50,
}
JUSTIFICATIVA_INTERACOES = {
    "f_hand:z_rio": "rio maior -> cota de cheia maior -> HAND alcanca mais longe; espera-se efeito de HAND menos negativo",
    "f_hand:z_relevo": "vale encaixado confina a lamina; TC-CTX01 v2 nao sustentou interacao com declive: sem sinal esperado",
    "f_sca:z_relevo": "concentracao pesa mais em relevo forte (enxurrada); espera-se efeito de ln_SCA mais positivo",
    "f_hand:z_costa": "saida restrita por mare/remanso em bacia costeira; espera-se HAND baixo mais decisivo",
    "demais": "modificadores declarados de contexto, penalizados; nenhuma monotonia imposta",
}


# ------------------------------------------------------------- dados

def carregar(pop: Path = SAIDA / "populacao.parquet", unid: Path = SAIDA / "unidades.parquet",
             ctx: Path = SAIDA / "contexto_unidades.parquet", sensibilidade: str | None = None) -> pd.DataFrame:
    d = pd.read_parquet(pop)
    d = d[d.fonte.isin(CONFIG["fontes"]) & d.papel.isin(CONFIG["papeis"])]
    if sensibilidade == "jrc":
        o = pd.read_parquet(SAIDA / "ocorrencia_jrc.parquet", columns=["ponto_id", "occurrence"])
        d = d.merge(o, on="ponto_id", how="left")
        d = d[~(d.occurrence >= CONFIG["corte_jrc"])]
    d = d[d.hand_m.notna() & d.elevation_local_m.notna() & d.ln_sca.notna() & np.isfinite(d.ln_sca)
          & d.slope_deg.notna()]
    duas = d.groupby("evento_id").classe.nunique()
    d = d[d.evento_id.isin(duas[duas == 2].index)]
    u = pd.read_parquet(unid)
    desenho = "exutorio" if sensibilidade == "exutorio" else CONFIG["desenho"]
    u = u[u.desenho == desenho][["evento_id", "unidade"]]
    d = d.merge(u, on="evento_id", how="left")
    c = pd.read_parquet(ctx)
    c = c[["unidade_terreno", "declive_p50", "hand_p90"]]
    d = d.merge(c, on="unidade_terreno", how="left")
    return d.sort_values("ponto_id", kind="stable").reset_index(drop=True)


def caracteristicas(d: pd.DataFrame) -> pd.DataFrame:
    """Transformacoes fixas antes de olhar resultado; padronizacao fica para o ajuste."""
    x = pd.DataFrame(index=d.index)
    x["f_hand"] = np.log1p(d.hand_m.clip(lower=0))
    x["f_elev"] = np.log1p(d.elevation_local_m.clip(lower=0))
    x["f_sca"] = d.ln_sca.clip(-5, 25)
    x["f_declive"] = np.log1p(d.slope_deg.clip(lower=0))
    x["f_construido"] = d.construido_frac
    x["f_areia"] = d.areia_pct_0_30 / 100
    x["sem_construido"] = d.construido_frac.isna().astype(float)
    x["sem_areia"] = d.areia_pct_0_30.isna().astype(float)
    x["z_rio"] = d.ln_up_area_l12
    x["z_relevo"] = np.log1p(d.declive_p50)
    x["z_hand90"] = np.log1p(d.hand_p90.clip(lower=0))
    x["z_costa"] = (d.COAST_L12 == 1).astype(float)
    return x


def amostrar(d: pd.DataFrame, teto: int, semente: int) -> pd.DataFrame:
    """Ate `teto` linhas por evento e classe, sorteio com semente; nao estratifica por resultado de modelo."""
    rng = np.random.default_rng(semente)
    partes = []
    for _, g in d.groupby(["evento_id", "classe"], sort=True):
        partes.append(g if len(g) <= teto else g.iloc[np.sort(rng.choice(len(g), teto, replace=False))])
    return pd.concat(partes).sort_index()


# ------------------------------------------------------------- modelo

class Padronizador:
    def fit(self, x: pd.DataFrame):
        self.mu = x.mean()
        self.sd = x.std().replace(0, 1).fillna(1)
        self.med = x.median()
        return self

    def __call__(self, x: pd.DataFrame) -> pd.DataFrame:
        return ((x.fillna(self.med) - self.mu) / self.sd).fillna(0.0)


class Logistica:
    """Logistica com intercepto por evento (livre), efeitos principais, interacoes e inclinacao por territorio.

    Perda: media por evento de [0,5 media log-perda dos positivos + 0,5 dos negativos]
           + 0,5 (lam_p |b|^2 + lam_i |g|^2 + lam_t |U|^2).
    Cada evento pesa 1 e cada classe metade dele: nem o GFD com centenas de milhares de linhas,
    nem a classe majoritaria, dominam o ajuste.
    """

    def __init__(self, principais, interacoes=(), territorio=False, lam_p=.01, lam_i=1., lam_t=.1):
        self.principais, self.interacoes = list(principais), list(interacoes)
        self.territorio, self.lam_p, self.lam_i, self.lam_t = territorio, lam_p, lam_i, lam_t

    def _matrizes(self, x: pd.DataFrame):
        z = self.pad(x)
        m = z[self.principais].to_numpy(float)
        i = np.column_stack([z[a].to_numpy() * z[b].to_numpy() for a, b in self.interacoes]) \
            if self.interacoes else np.zeros((len(z), 0))
        return m, i

    def fit(self, x: pd.DataFrame, y: np.ndarray, evento: np.ndarray, terr: np.ndarray | None = None):
        cols = sorted(set(self.principais) | {c for p in self.interacoes for c in p})
        self.pad = Padronizador().fit(x[cols])
        y = np.asarray(y, float)
        m, i = self._matrizes(x)
        ev_ids, ev = np.unique(evento, return_inverse=True)
        n_ev = len(ev_ids)
        npos = np.bincount(ev, weights=y, minlength=n_ev)
        nneg = np.bincount(ev, weights=1 - y, minlength=n_ev)
        w = np.where(y == 1, .5 / np.maximum(npos[ev], 1), .5 / np.maximum(nneg[ev], 1)) / n_ev
        p, q = m.shape[1], i.shape[1]
        if self.territorio:
            t_ids, t = np.unique(terr, return_inverse=True)
            nt = len(t_ids)
        else:
            t, nt = None, 0
        tam = n_ev + p + q + nt * p

        def separar(v):
            return v[:n_ev], v[n_ev:n_ev + p], v[n_ev + p:n_ev + p + q], v[n_ev + p + q:].reshape(nt, p)

        def perda(v):
            a, b, g, U = separar(v)
            eta = a[ev] + m @ b + (i @ g if q else 0)
            if nt:
                eta = eta + np.einsum("ij,ij->i", U[t], m)
            lp = np.logaddexp(0, eta) - y * eta
            r = w * (expit(eta) - y)
            f = np.sum(w * lp) + .5 * (self.lam_p * b @ b + self.lam_i * g @ g + self.lam_t * np.sum(U * U))
            ga = np.bincount(ev, weights=r, minlength=n_ev)
            gb = m.T @ r + self.lam_p * b
            gg = i.T @ r + self.lam_i * g if q else np.zeros(0)
            if nt:
                gU = np.zeros((nt, p))
                np.add.at(gU, t, r[:, None] * m)
                gU += self.lam_t * U
            else:
                gU = np.zeros((0, p))
            return f, np.concatenate([ga, gb, gg, gU.ravel()])

        res = minimize(perda, np.zeros(tam), jac=True, method="L-BFGS-B",
                       options={"maxiter": 2000, "gtol": 1e-8})
        a, self.b, self.g, self.U = separar(res.x)
        self.convergiu, self.iteracoes = bool(res.success), int(res.nit)
        self.territorios = t_ids if nt else np.array([])
        return self

    def escore(self, x: pd.DataFrame, u: np.ndarray | None = None) -> np.ndarray:
        """Ordem dentro do evento. `u` = inclinacao local adaptada (braco de poucos rotulos)."""
        m, i = self._matrizes(x)
        eta = m @ self.b + (i @ self.g if len(self.g) else 0)
        return eta + (m @ u if u is not None else 0)

    def adaptar(self, x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Inclinacao local com prior N(0, 1/lam_t) em volta do nucleo transferido; b e g fixos."""
        m, _ = self._matrizes(x)
        y = np.asarray(y, float)
        base = self.escore(x)
        npos, nneg = max(y.sum(), 1), max((1 - y).sum(), 1)
        w = np.where(y == 1, .5 / npos, .5 / nneg)
        p = m.shape[1]

        def perda(v):
            a, u = v[0], v[1:]
            eta = base + a + m @ u
            r = w * (expit(eta) - y)
            f = np.sum(w * (np.logaddexp(0, eta) - y * eta)) + .5 * self.lam_t * u @ u
            return f, np.concatenate([[r.sum()], m.T @ r + self.lam_t * u])

        return minimize(perda, np.zeros(p + 1), jac=True, method="L-BFGS-B").x[1:]


def regra_composta(x_treino: pd.DataFrame, x: pd.DataFrame) -> np.ndarray:
    pad = Padronizador().fit(x_treino[list(SENTIDO_REGRA)])
    z = pad(x[list(SENTIDO_REGRA)])
    return sum(s * z[c].to_numpy() for c, s in SENTIDO_REGRA.items())


# ------------------------------------------------------------- medidas

def auc(y: np.ndarray, s: np.ndarray) -> float:
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap(y: np.ndarray, s: np.ndarray) -> float:
    o = np.argsort(-s, kind="stable")
    yy = y[o]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float(np.sum(prec * yy) / max(yy.sum(), 1))


def medidas_evento(y: np.ndarray, s: np.ndarray) -> dict:
    prev = float(y.mean())
    out = {"auc": auc(y, s), "ap": ap(y, s), "prevalencia": prev, "n": int(len(y)), "positivos": int(y.sum())}
    out["ap_sobre_prevalencia"] = out["ap"] / prev if prev > 0 else float("nan")
    for f in CONFIG["topo_inspecao"]:
        corte = np.quantile(s, 1 - f)
        out[f"captura_topo_{int(f * 100)}"] = float(((s >= corte) & (y == 1)).sum() / max(y.sum(), 1))
    return out


# ------------------------------------------------------------- execucao por destino

def interacoes():
    efeitos = sum(PRINCIPAIS.values(), [])
    return [(a, b) for a in efeitos for b in CONTEXTO]


def ajustar(nome: str, xs: pd.DataFrame, ds: pd.DataFrame, lam: dict):
    efeitos = sum(PRINCIPAIS.values(), []) + FLAGS
    if nome == "G":
        mod = Logistica(efeitos, lam_p=lam["p"])
    elif nome == "C":
        mod = Logistica(efeitos, interacoes(), lam_p=lam["p"], lam_i=lam["i"])
    elif nome == "H":
        mod = Logistica(efeitos, interacoes(), territorio=True, lam_p=lam["p"], lam_i=lam["i"], lam_t=lam["t"])
    elif nome.startswith("G_sem_"):
        bloco = nome.removeprefix("G_sem_")
        fora = set(PRINCIPAIS[bloco])
        mod = Logistica([c for c in efeitos if c not in fora], lam_p=lam["p"])
    else:
        raise ValueError(nome)
    return mod.fit(xs, ds.classe.to_numpy(float), ds.evento_id.to_numpy(), ds.unidade.to_numpy())


def auc_media_por_unidade(d: pd.DataFrame, s: np.ndarray) -> float:
    v = []
    for _, gu in d.assign(_s=s).groupby("unidade"):
        a = [auc(g.classe.to_numpy(), g._s.to_numpy()) for _, g in gu.groupby("evento_id")]
        a = [x for x in a if np.isfinite(x)]
        if a:
            v.append(np.mean(a))
    return float(np.mean(v)) if v else float("nan")


def selecionar(d: pd.DataFrame, x: pd.DataFrame, semente: int) -> dict:
    """Penalidades por dobras internas de TERRITORIOS de treino; o destino nao participa."""
    unidades = np.array(sorted(d.unidade.unique()))
    rng = np.random.default_rng(semente)
    dobra = dict(zip(unidades, rng.permutation(len(unidades)) % CONFIG["dobras_internas"]))
    fd = d.unidade.map(dobra).to_numpy()

    def nota(nome, lam):
        v = []
        for k in range(CONFIG["dobras_internas"]):
            tr, te = fd != k, fd == k
            if d[te].unidade.nunique() == 0 or d[tr].unidade.nunique() < 2:
                continue
            m = ajustar(nome, x[tr], d[tr], lam)
            v.append(auc_media_por_unidade(d[te], m.escore(x[te])))
        return float(np.nanmean(v))

    lam = {"p": .01, "i": 1., "t": .1}
    notas = {}
    melhor = max(CONFIG["grade_lam_principal"], key=lambda l: notas.setdefault(f"G:{l}", nota("G", {**lam, "p": l})))
    lam["p"] = melhor
    lam["i"] = max(CONFIG["grade_lam_interacao"],
                   key=lambda l: notas.setdefault(f"C:{l}", nota("C", {**lam, "i": l})))
    lam["t"] = max(CONFIG["grade_lam_territorio"],
                   key=lambda l: notas.setdefault(f"H:{l}", nota("H", {**lam, "t": l})))
    return {"lam": lam, "notas_internas": notas}


def descritor_territorio(d: pd.DataFrame) -> pd.DataFrame:
    """Contexto do territorio pelo raster inteiro (sem rotulo) e posicao mediana; para escolher doadores."""
    return d.groupby("unidade").agg(declive_p50=("declive_p50", "median"), hand_p90=("hand_p90", "median"),
                                    lat=("lat", "median"), lon=("lon", "median"))


def doadores(destino: str, treino: pd.DataFrame, desc: pd.DataFrame, modo: str, rng) -> list:
    cand = desc.loc[sorted(treino.unidade.unique())]
    k = min(CONFIG["k_doadores"], len(cand))
    if modo == "contexto":
        cols = ["declive_p50", "hand_p90"]
        z = np.log1p(cand[cols].clip(lower=0).fillna(cand[cols].median()))
        alvo = np.log1p(desc.loc[[destino], cols].clip(lower=0).fillna(cand[cols].median()))
        sd = z.std().replace(0, 1)
        dist = np.sqrt((((z - alvo.iloc[0]) / sd) ** 2).sum(1))
    elif modo == "proximo":
        from mt01_independencia import haversine_km
        a = desc.loc[destino]
        dist = pd.Series(haversine_km(a.lat, a.lon, cand.lat.to_numpy(), cand.lon.to_numpy()), index=cand.index)
    else:
        return list(rng.choice(cand.index.to_numpy(), k, replace=False))
    return list(dist.sort_values(kind="stable").index[:k])


def executar_destino(destino: str, d: pd.DataFrame, x: pd.DataFrame, desc: pd.DataFrame,
                     completo: bool = True) -> dict:
    sem = CONFIG["semente"]
    alvo = d.unidade == destino
    treino_cheio = d[~alvo]
    tr = amostrar(treino_cheio, CONFIG["teto_ajuste_por_evento_classe"], sem)
    te = amostrar(d[alvo], CONFIG["teto_teste_por_evento_classe"], sem)
    ev_ok = te.groupby("evento_id").filter(lambda g: len(g) >= CONFIG["minimo_teste_evento"]
                                           and g.classe.nunique() == 2).evento_id.unique()
    te = te[te.evento_id.isin(ev_ok)]
    if te.empty:
        return {"destino": destino, "estado": "sem_evento_avaliavel"}
    xtr, xte = x.loc[tr.index], x.loc[te.index]
    sel = selecionar(tr, xtr, sem)
    lam = sel["lam"]
    escores = {"A0_regra_hand": -te.hand_m.to_numpy(), "A1_regra_composta": regra_composta(xtr, xte)}
    ajustes = {}
    for nome in ("G", "C", "H", "G_sem_acumulacao", "G_sem_conexao", "G_sem_producao")[:6 if completo else 3]:
        m = ajustar(nome, xtr, tr, lam)
        ajustes[nome] = m
        escores[nome] = m.escore(xte)
    if not completo:
        return finalizar(destino, te, escores, ajustes, sel, tr, set(te.fonte), {}, [], xte)
    # atalho de instrumento: sem nenhuma linha da(s) fonte(s) do destino
    fontes_dest = set(te.fonte)
    sf = ~tr.fonte.isin(fontes_dest)
    if sf.any() and tr[sf].classe.nunique() == 2:
        escores["G_sem_mesma_fonte"] = ajustar("G", xtr[sf], tr[sf], lam).escore(xte)
    # doadores com suporte comparavel: mesmo numero de eventos, mesmo teto por classe
    rng = np.random.default_rng(sem + 1)
    conjuntos = {"D_contexto": doadores(destino, tr, desc, "contexto", rng),
                 "D_proximo": doadores(destino, tr, desc, "proximo", rng)}
    for s in range(CONFIG["sementes_sorteio"]):
        conjuntos[f"D_aleatorio_{s}"] = doadores(destino, tr, desc, "aleatorio", np.random.default_rng(sem + 10 + s))
    eventos_por = {k: sorted(tr[tr.unidade.isin(v)].evento_id.unique()) for k, v in conjuntos.items()}
    n_ev = min(CONFIG["eventos_doadores_max"], min(len(v) for v in eventos_por.values()))
    suporte = {}
    for k, evs in eventos_por.items():
        escolhidos = list(np.random.default_rng(sem + 2).choice(evs, n_ev, replace=False)) if len(evs) > n_ev else evs
        sub = tr[tr.evento_id.isin(escolhidos)]
        suporte[k] = {"territorios": conjuntos[k], "eventos": len(escolhidos), "positivos": int(sub.classe.sum()),
                      "negativos": int((1 - sub.classe).sum()), "fontes": sorted(sub.fonte.unique())}
        if sub.classe.nunique() == 2:
            escores[k] = ajustar("G", xtr.loc[sub.index], sub, lam).escore(xte)
    # curva: quantos territorios observados
    unidades_tr = np.array(sorted(tr.unidade.unique()))
    for k in CONFIG["curva_territorios"]:
        if k >= len(unidades_tr):
            continue
        for s in range(CONFIG["sementes_sorteio"]):
            esc = np.random.default_rng(sem + 100 * k + s).choice(unidades_tr, k, replace=False)
            sub = tr[tr.unidade.isin(esc)]
            if sub.classe.nunique() == 2:
                escores[f"curva_U{k}_{s}"] = ajustar("G", xtr.loc[sub.index], sub, lam).escore(xte)
    poucos = poucos_rotulos(destino, d[alvo], x, ajustes["H"], sem)
    return finalizar(destino, te, escores, ajustes, sel, tr, fontes_dest, suporte, poucos, xte)


def finalizar(destino, te, escores, ajustes, sel, tr, fontes_dest, suporte, poucos, xte) -> dict:
    lam = sel["lam"]
    medidas = []
    for nome, s in escores.items():
        for ev, g in te.assign(_s=s).groupby("evento_id"):
            r = medidas_evento(g.classe.to_numpy(), g._s.to_numpy())
            r.update({"destino": destino, "modelo": nome, "evento_id": ev, "fonte": g.fonte.iloc[0]})
            medidas.append(r)
        # processo no ponto (so CEMS): positivos de um processo contra os mesmos negativos observados
        cems = te.assign(_s=s)[te.fonte == "cems"]
        for (ev, proc), g in cems[cems.classe == 1].groupby(["evento_id", "processo_ponto"]):
            neg = cems[(cems.evento_id == ev) & (cems.classe == 0)]
            if len(g) >= 10 and len(neg) >= 10:
                yy = np.r_[np.ones(len(g)), np.zeros(len(neg))]
                medidas.append({"destino": destino, "modelo": nome, "evento_id": ev, "fonte": "cems",
                                "processo": proc, "auc": auc(yy, np.r_[g._s, neg._s]), "positivos": len(g)})
    coef = {n: {"b": dict(zip(m.principais, np.round(m.b, 6))),
                "g": dict(zip([f"{a}:{b}" for a, b in m.interacoes], np.round(m.g, 6))),
                "convergiu": m.convergiu} for n, m in ajustes.items() if n in ("G", "C", "H")}
    pred = te[["ponto_id", "evento_id", "classe", "fonte", "processo_ponto"]].copy()
    for nome in ("A0_regra_hand", "A1_regra_composta", "G", "C", "H"):
        pred[nome] = escores[nome]
    return {"destino": destino, "estado": "executado", "lam": lam, "_pred": pred, "notas_internas": sel["notas_internas"],
            "treino": {"eventos": int(tr.evento_id.nunique()), "territorios": int(tr.unidade.nunique()),
                       "linhas": int(len(tr)), "positivos": int(tr.classe.sum())},
            "teste": {"eventos": int(te.evento_id.nunique()), "linhas": int(len(te)), "positivos": int(te.classe.sum()),
                      "fontes": sorted(fontes_dest)},
            "suporte_doadores": suporte, "coeficientes": coef, "medidas": medidas, "poucos_rotulos": poucos,
            "comportamento": comportamento(ajustes["C"], xte, te)}


def poucos_rotulos(destino: str, dd: pd.DataFrame, x: pd.DataFrame, h: Logistica, sem: int) -> list:
    """Evento de adaptacao = o primeiro no tempo (regra sem rotulo); avaliacao nos demais eventos."""
    evs = dd.groupby("evento_id").agg(ini=("janela_inicio", "min"), duas=("classe", "nunique"))
    evs = evs[evs.duas == 2].sort_values(["ini"], kind="stable")
    if len(evs) < 2:
        return []
    adapt = evs.index[0]
    aval = amostrar(dd[dd.evento_id.isin(evs.index[1:])], CONFIG["teto_teste_por_evento_classe"], sem)
    fonte_ad = dd[dd.evento_id == adapt]
    out = []
    xa = x.loc[aval.index]
    base = h.escore(xa)
    for n in CONFIG["poucos_rotulos_n"]:
        por_classe = n // 2
        pos, neg = fonte_ad[fonte_ad.classe == 1], fonte_ad[fonte_ad.classe == 0]
        if len(pos) < por_classe or len(neg) < por_classe:
            continue
        rng = np.random.default_rng(sem + n)
        amo = pd.concat([pos.iloc[rng.choice(len(pos), por_classe, replace=False)],
                         neg.iloc[rng.choice(len(neg), por_classe, replace=False)]])
        u = h.adaptar(x.loc[amo.index], amo.classe.to_numpy(float))
        so_local = Logistica(h.principais, lam_p=1.0).fit(x.loc[amo.index], amo.classe.to_numpy(float),
                                                         np.zeros(len(amo)))
        for nome, s in (("H_zero", base), ("H_adaptado", h.escore(xa, u)), ("so_local", so_local.escore(xa))):
            for ev, g in aval.assign(_s=s).groupby("evento_id"):
                if g.classe.nunique() == 2 and len(g) >= CONFIG["minimo_teste_evento"]:
                    out.append({"destino": destino, "n": n, "modelo": nome, "evento_id": ev,
                                "adaptacao": adapt, "auc": auc(g.classe.to_numpy(), g._s.to_numpy())})
    return out


def comportamento(m: Logistica, x: pd.DataFrame, d: pd.DataFrame) -> dict:
    """Intervencao SINTETICA no modelo (nao no mundo): +1 desvio em cada efeito, por quartil de contexto."""
    out = {}
    base = m.escore(x)
    for c in sum(PRINCIPAIS.values(), []):
        x2 = x.copy()
        x2[c] = x2[c] + m.pad.sd[c]
        delta = m.escore(x2) - base
        out[c] = {"media": float(np.mean(delta))}
        for z in ("z_rio", "z_relevo"):
            q = pd.qcut(x[z].rank(method="first"), 4, labels=False) if x[z].notna().sum() > 8 else None
            if q is not None:
                out[c][z] = [float(np.mean(delta[q.to_numpy() == k])) for k in range(4)]
    return out


# ------------------------------------------------------------- protocolo

def fontes_do_protocolo() -> list[Path]:
    return [SAIDA / "populacao.parquet", SAIDA / "unidades.parquet", SAIDA / "contexto_unidades.parquet",
            Path(__file__), Path(mp.__file__), REPO / "scripts/metodo/mt01_independencia.py",
            REPO / "scripts/metodo/mt01_bacias_globais.py", REPO / "scripts/metodo/mt01_ocorrencia_jrc.py",
            REPO / "scripts/servico/proveniencia.py"]


def pre_registrar() -> dict:
    arq = SAIDA / "pre_registro.json"
    if arq.exists():
        raise FileExistsError("pre-registro ja existe; versao nova exige nova pasta")
    d = carregar()
    unidades = d.groupby("unidade").agg(eventos=("evento_id", "nunique"), fontes=("fonte", lambda s: "+".join(sorted(set(s)))),
                                        positivos=("classe", "sum"), linhas=("classe", "size"))
    pre = {
        "tipo": "mt01_pre_registro", "versao": VERSAO, "quando_utc": agora_utc(),
        "pergunta": ("Relacoes aprendidas em territorios observados ordenam a inundacao observada em territorios "
                     "retidos sem rotulo local? Sob que modificadores? Quanto rotulo local muda isso?"),
        "populacao": {"linhas": int(len(d)), "eventos": int(d.evento_id.nunique()),
                      "unidades": int(len(unidades)), "positivos": int(d.classe.sum()),
                      "por_fonte": d.groupby("fonte").evento_id.nunique().to_dict()},
        "unidades_de_retencao": unidades.reset_index().to_dict("records"),
        "config": CONFIG, "principais": PRINCIPAIS, "contexto": CONTEXTO, "sentido_regra": SENTIDO_REGRA,
        "interacoes": JUSTIFICATIVA_INTERACOES,
        "hipoteses": {
            "H1": "G supera A0 e A1 em AUC media por unidade (margem 0,01; IC bootstrap por unidade exclui 0)",
            "H2": ("C e H nao-inferiores a G (margem -0,01) e superiores se delta > 0,01 com IC > 0; G vencer "
                   "rejeita a parametrizacao, nao a transferencia"),
            "H3": ("ablacao de bloco com papel identificavel perde > 0,01; sinais de G estaveis entre destinos; "
                   "comportamento sintetico de C coerente com JUSTIFICATIVA_INTERACOES"),
            "H4": "G_sem_mesma_fonte perde menos de 0,01 frente a G (ganho nao depende do instrumento do destino)",
            "doadores": "D_contexto vs D_proximo vs D_aleatorio com mesmo numero de eventos (secundario)",
            "poucos_rotulos": "H_adaptado vs H_zero e so_local por n (braco separado, nunca somado a H1-H4)",
        },
        "metricas": {"primaria": "AUC dentro do evento, media igual dos eventos na unidade e das unidades",
                     "secundarias": ["AP/prevalencia", "captura no topo 10% e 20% do quadro amostrado",
                                     "pior unidade e decil inferior", "AUC por processo no ponto (CEMS)"],
                     "nao_reportadas": "calibracao/Brier: intercepto do destino desconhecido; probabilidade "
                                       "condicionada ao evento observado nao e probabilidade anual"},
        "controles_de_atalho": ["desempenho por fonte/instrumento do destino", "por familia de MDT do destino "
                                "(fabdem_average x outro_bilinear; dentro do CEMS as duas coexistem)",
                                "por resolucao nativa do rotulo", "G_sem_mesma_fonte",
                                "doadores com mesmo numero de eventos e teto por classe", "curva por numero de territorios",
                                "processo no ponto (CEMS): fluvial x enxurrada x costeira contra os mesmos negativos"],
        "decisao": ("candidata para confirmacao futura = H se nao-inferior a G e pior unidade >= G; senao C pelo "
                    "mesmo criterio; senao G. Nada e promovido."),
        "ja_examinado": ("todos os eventos ja foram lidos em ciclos anteriores (reservas-examinadas); este protocolo "
                         "e desenvolvimento prospectivo, nao confirmacao independente. EMSR868/871 fora."),
        "minimo_unidades_para_inferir": 8,
        "fontes": {str(p.relative_to(REPO)): sha256_arquivo(p) for p in fontes_do_protocolo()},
    }
    pre["hash_resumo"] = sha256_json({k: v for k, v in pre.items() if k != "quando_utc"})
    mp.escrever(arq, pre)
    return pre


def verificar_fontes() -> dict:
    pre = json.loads((SAIDA / "pre_registro.json").read_text(encoding="utf-8"))
    for nome, h in pre["fontes"].items():
        if sha256_arquivo(REPO / nome) != h:
            raise RuntimeError(f"entrada/codigo alterado depois do pre-registro: {nome}")
    return pre


def executar(so: int | None = None, sensibilidade: str | None = None) -> None:
    pre = verificar_fontes()
    d = carregar(sensibilidade=sensibilidade)
    x = caracteristicas(d)
    desc = descritor_territorio(d)
    pasta = SAIDA / (f"destinos_{sensibilidade}" if sensibilidade else "destinos")
    pasta.mkdir(exist_ok=True)
    unidades = ([u["unidade"] for u in pre["unidades_de_retencao"]] if sensibilidade != "exutorio"
                else sorted(d.unidade.unique()))
    for k, dest in enumerate(unidades[:so] if so else unidades):
        arq = pasta / f"{sha256_json(dest)[7:23]}.json"
        if arq.exists():
            continue
        t0 = time.time()
        r = executar_destino(dest, d, x, desc, completo=sensibilidade is None)
        pred = r.pop("_pred", None)
        if pred is not None:
            pred.to_parquet(arq.with_suffix(".parquet"), index=False)
            r["predicoes_sha256"] = sha256_arquivo(arq.with_suffix(".parquet"))
        r["segundos"] = round(time.time() - t0, 1)
        r["pre_registro"] = pre["hash_resumo"]
        mp.escrever(arq, r)
        print(k, dest, r["estado"], r.get("segundos"), flush=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap_ = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap_.add_argument("acao", choices=("pre_registrar", "executar"))
    ap_.add_argument("--so", type=int)
    ap_.add_argument("--sensibilidade", choices=sorted(CONFIG["sensibilidades"]))
    a = ap_.parse_args()
    if a.acao == "pre_registrar":
        p = pre_registrar()
        print(json.dumps(p["populacao"], ensure_ascii=False), p["hash_resumo"])
    else:
        executar(a.so, a.sensibilidade)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
