"""MT09: o que da explicação física é identificável como mecanismo, e sob quais pressupostos.

O grafo conceitual (`conjunta_grafo_causal`) declarou o efeito do relevo `not_identified`
porque a amostra existe com seleção S=1 que depende da inundação. Isso vale para
quantidades marginais e interceptos. Não vale para a razão de chances DENTRO do evento:
quando a amostragem depende só do rótulo (sorteio por classe dentro do evento, como em
todo o MT01-MT07), as inclinações da logística são consistentes para as razões de chances
da população (Prentice & Pyke, 1979); a seleção do evento e da AOI cai no intercepto livre
por evento. O que sobra como ameaça é confundimento DENTRO do evento, erro de rótulo
diferencial e falta de positividade. Este módulo ataca cada um com um desenho próprio,
em vez de declarar a pergunta aberta:

  C1 amostragem por classe   inclinações com 100% e com 25% dos negativos do evento
  C2 dentro do evento        logística por evento; agregação por componente; meta-análise
                             de efeitos aleatórios (DerSimonian-Laird) com intervalo de
                             predição para um componente NOVO (critério de invariância)
  C3 triangulação            mesmo efeito por família de instrumento (MODIS óptico, CEMS
                             analista, Sen1Floods11 S1 manual, UFO urbano 3 m): erros de
                             observação diferentes não produzem o mesmo sinal por acaso
  C4 vizinhança              pares positivo-negativo a <= 3 km no mesmo evento, logística
                             condicional exata de pares: remove todo confundidor que varia
                             em escala maior que o par (AOI, chuva regional, família de MDT,
                             exposição, uso regional do solo); 1 km como sensibilidade
  C5 dose-resposta           HAND em faixas físicas; forma monotônica esperada da
                             hidráulica de enchimento
  C6 modificação por cobertura  efeito por classe WorldCover (exploratório)
  C7 explicação do modelo    sinais dos pesos Q_conexao nos 43 retreinos MT06 contra o
                             mecanismo sustentado
  C8 forçamento              leitura dos resultados já encerrados (autopareamento NY/SP;
                             chuva e detecção SAR entre datas), sem nova estimativa

Regra de leitura (antes de qualquer estimativa), por variável:
  direcao       C2: IC do efeito médio exclui zero
  invariante    C2: intervalo de predição 95% exclui zero
  triangulada   C3: mesmo sinal com IC excluindo zero em >= 3 das 4 famílias e nenhuma
                família com IC excluindo zero no sinal oposto
  local         C4 a 3 km: IC exclui zero no mesmo sinal
  mecanismo_local_triangulado               direcao + triangulada + local
  associacao_triangulada_sem_contraste_local direcao + triangulada
  associacao_dependente_do_instrumento      direcao, sem triangulação
  nao_sustentado                            sem direção
  O sinal observado é comparado ao previsto pela hidráulica (registrado aqui antes).

Não é efeito de intervenção: posição no relevo não se manipula. A afirmação possível é
estrutural-condicional: comparando dois pontos do mesmo evento e da mesma vizinhança, o
mais baixo em relação à drenagem tem mais chance de estar inundado, e isso não depende do
instrumento que viu a água. Ameaças que este desenho não remove ficam declaradas.

Uso (pela trava):
    python scripts/manutencao/executar_serializado.py -- python scripts/metodo/mt09_identificacao_causal.py executar
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "2")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.neighbors import BallTree

import mt06_cobertura_observabilidade as e
import mt08_teto_compartilhado as t8

SAIDA = e.REPO / "local_runs/mecanismos-transferiveis/v9/identificacao_causal_v1"
MECANISMOS = ["f_hand", "f_dist_canal", "f_dist_dren", "f_declive", "f_sca"]
SINAL_HIDRAULICO = {"f_hand": -1, "f_dist_canal": -1, "f_dist_dren": -1, "f_declive": -1, "f_sca": +1}
FAMILIAS = {"gfd": "MODIS óptico 250 m", "cems": "CEMS analista S1/óptico", "sen1floods11": "S1/S2 rótulo manual 10 m",
            "ufo": "urbano 3 m"}
FAIXAS_HAND = [0, .5, 1, 2, 5, 10, np.inf]
COBERTURA = {10: "arvore", 20: "campo", 30: "campo", 40: "agricultura", 50: "construido", 60: "solo_nu",
             80: "agua_umida", 90: "agua_umida", 95: "agua_umida"}
CFG = {"semente_amostra": 20261010, "por_evento_classe": 2000, "lam": 1e-3, "fracao_negativos_c1": .25,
       "semente_c1": 20261018, "raio_km": 3., "raio_sensibilidade_km": 1., "minimo_pares": 20,
       "minimo_por_classe": 20, "bootstrap": 2000, "semente_bootstrap": 7, "nivel": .95,
       "minimo_componentes": 8, "minimo_familias_triangulacao": 3}
RAIO_TERRA_KM = 6371.0088


# ------------------------------------------------------------------ estimadores

def logistica(x: np.ndarray, y: np.ndarray, lam: float, intercepto: bool = True):
    """Logística com crista leve nas inclinações. Devolve (beta, ep) das inclinações."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.column_stack([np.ones(len(x)), x]) if intercepto else x
    pen = np.r_[0., np.full(x.shape[1], lam)] if intercepto else np.full(x.shape[1], lam)

    def f(b):
        eta = z @ b
        return (np.sum(np.logaddexp(0, eta) - y * eta) + .5 * np.sum(pen * b * b),
                z.T @ (expit(eta) - y) + pen * b)

    r = minimize(f, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-9})
    p = expit(z @ r.x)
    h = (z * (p * (1 - p))[:, None]).T @ z + np.diag(pen)
    cov = np.linalg.pinv(h)
    k = 1 if intercepto else 0
    return r.x[k:], np.sqrt(np.clip(np.diag(cov)[k:], 0, None)), bool(r.success)


def logistica_pares(delta: np.ndarray, lam: float):
    """Logística condicional exata para pares 1:1: P(caso é o caso | par) = expit(beta . (x_caso - x_controle))."""
    delta = np.asarray(delta, float)

    def f(b):
        eta = delta @ b
        return np.sum(np.logaddexp(0, -eta)) + .5 * lam * b @ b, -delta.T @ expit(-eta) + lam * b

    r = minimize(f, np.zeros(delta.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-9})
    p = expit(delta @ r.x)
    h = (delta * (p * (1 - p))[:, None]).T @ delta + lam * np.eye(delta.shape[1])
    return r.x, np.sqrt(np.clip(np.diag(np.linalg.pinv(h)), 0, None)), bool(r.success)


def pares_vizinhos(lat, lon, y, raio_km):
    """Para cada positivo, o negativo mais próximo dentro do raio (com reposição)."""
    lat, lon, y = np.asarray(lat, float), np.asarray(lon, float), np.asarray(y, int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return np.zeros((0, 2), int)
    tree = BallTree(np.radians(np.c_[lat[neg], lon[neg]]), metric="haversine")
    dist, idx = tree.query(np.radians(np.c_[lat[pos], lon[pos]]), k=1)
    ok = dist[:, 0] * RAIO_TERRA_KM <= raio_km
    return np.c_[pos[ok], neg[idx[ok, 0]]]


def meta_dl(est: np.ndarray, ep: np.ndarray, nivel: float = .95) -> dict:
    """DerSimonian-Laird; IC do efeito médio e intervalo de predição para um componente novo."""
    est, ep = np.asarray(est, float), np.asarray(ep, float)
    ok = np.isfinite(est) & np.isfinite(ep) & (ep > 0)
    est, ep = est[ok], ep[ok]
    k = len(est)
    if k < 3:
        return {"k": k, "estado": "not_testable"}
    w = 1 / ep ** 2
    mu_f = np.sum(w * est) / w.sum()
    q = np.sum(w * (est - mu_f) ** 2)
    tau2 = max(0., (q - (k - 1)) / (w.sum() - np.sum(w ** 2) / w.sum()))
    ws = 1 / (ep ** 2 + tau2)
    mu = np.sum(ws * est) / ws.sum()
    se = np.sqrt(1 / ws.sum())
    zc = stats.norm.ppf(.5 + nivel / 2)
    tc = stats.t.ppf(.5 + nivel / 2, k - 2)
    pi = [mu - tc * np.sqrt(tau2 + se ** 2), mu + tc * np.sqrt(tau2 + se ** 2)]
    return {"k": k, "media": float(mu), "ep": float(se), "ic": [float(mu - zc * se), float(mu + zc * se)],
            "tau2": float(tau2), "i2": float(max(0., (q - (k - 1)) / q)) if q > 0 else 0.,
            "intervalo_predicao": [float(v) for v in pi],
            "fracao_mesmo_sinal": float(np.mean(np.sign(est) == np.sign(mu))),
            "opostos_com_ic_fora_de_zero": int(np.sum((np.sign(est) != np.sign(mu)) & (np.abs(est) > zc * ep)))}


def agregar_componentes(tab: pd.DataFrame) -> pd.DataFrame:
    """Evento -> componente com peso igual dos eventos; ep do componente pela soma das variâncias."""
    g = tab.groupby(["destino", "variavel"])
    return pd.DataFrame({"beta": g.beta.mean(), "ep": g.ep.apply(lambda s: np.sqrt(np.sum(s ** 2)) / len(s)),
                         "eventos": g.size(), "fonte": g.fonte.agg(lambda s: s.mode().iloc[0])}).reset_index()


def bootstrap_media(v: np.ndarray) -> list:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    b = np.random.default_rng(CFG["semente_bootstrap"]).choice(v, (CFG["bootstrap"], len(v))).mean(1)
    q = (1 - CFG["nivel"]) / 2
    return [float(x) for x in np.quantile(b, [q, 1 - q])]


# ------------------------------------------------------------------ dados

def carregar():
    d = pd.read_parquet(e.SAIDA / "base.parquet")
    c = pd.read_parquet(t8.COORDENADAS)
    d = d.merge(c, on="ponto_id", how="left", validate="one_to_one")
    d = e.mt.e1.amostrar(d, CFG["por_evento_classe"], CFG["semente_amostra"])
    completo = d[MECANISMOS + ["lat", "lon"]].notna().all(axis=1)
    escala = d.loc[completo, MECANISMOS].std()
    z = d.loc[completo].copy()
    for v in MECANISMOS:
        z[v] = (z[v] - d.loc[completo, v].mean()) / escala[v]
    return z.reset_index(drop=True), {"linhas_amostradas": len(d), "linhas_completas": int(completo.sum()),
                                      "desvio_padrao_escala": escala.to_dict()}


def eventos_validos(d):
    for ev, g in d.groupby("evento_id", sort=True):
        n = g.classe.value_counts()
        if min(n.get(0, 0), n.get(1, 0)) >= CFG["minimo_por_classe"]:
            yield ev, g


# ------------------------------------------------------------------ desenhos

def c1_amostragem(d):
    rng = np.random.default_rng(CFG["semente_c1"])
    rows = []
    for ev, g in eventos_validos(d):
        b_all, _, _ = logistica(g[MECANISMOS].to_numpy(), g.classe.to_numpy(), CFG["lam"])
        neg = np.flatnonzero(g.classe.to_numpy() == 0)
        keep = np.r_[np.flatnonzero(g.classe.to_numpy() == 1),
                     rng.choice(neg, max(CFG["minimo_por_classe"], int(len(neg) * CFG["fracao_negativos_c1"])), replace=False)]
        h = g.iloc[np.sort(keep)]
        b_sub, ep_sub, _ = logistica(h[MECANISMOS].to_numpy(), h.classe.to_numpy(), CFG["lam"])
        for v, a, s, es in zip(MECANISMOS, b_all, b_sub, ep_sub):
            rows.append({"evento_id": ev, "variavel": v, "dif": float(s - a), "dif_em_ep": float((s - a) / es) if es > 0 else np.nan})
    t = pd.DataFrame(rows)
    return {v: {"eventos": int(len(g)), "dif_media": float(g.dif.mean()), "ic_dif_media": bootstrap_media(g.dif),
                "mediana_abs_dif_em_ep": float(g.dif_em_ep.abs().median())} for v, g in t.groupby("variavel")}


def c2_dentro_do_evento(d, colunas=MECANISMOS):
    rows = []
    for ev, g in eventos_validos(d):
        b, ep, ok = logistica(g[colunas].to_numpy(), g.classe.to_numpy(), CFG["lam"])
        for v, bi, ei in zip(colunas, b, ep):
            rows.append({"evento_id": ev, "destino": g.unidade.iloc[0], "fonte": g.fonte.iloc[0], "variavel": v,
                         "beta": float(bi), "ep": float(ei), "convergiu": ok, "n": len(g)})
    return pd.DataFrame(rows)


def c4_pares(d, raio_km):
    rows = []
    for ev, g in eventos_validos(d):
        par = pares_vizinhos(g.lat, g.lon, g.classe, raio_km)
        if len(par) < CFG["minimo_pares"]:
            continue
        x = g[MECANISMOS].to_numpy()
        b, ep, ok = logistica_pares(x[par[:, 0]] - x[par[:, 1]], CFG["lam"])
        for v, bi, ei in zip(MECANISMOS, b, ep):
            rows.append({"evento_id": ev, "destino": g.unidade.iloc[0], "fonte": g.fonte.iloc[0], "variavel": v,
                         "beta": float(bi), "ep": float(ei), "convergiu": ok, "pares": len(par),
                         "positivos_pareados": float(len(par) / (g.classe == 1).sum())})
    return pd.DataFrame(rows)


def c5_dose_resposta(d):
    """HAND em faixas físicas (m), demais mecanismos contínuos; faixa vazia no evento fica de fora do evento."""
    faixa = pd.cut(d.hand_m.clip(lower=0), FAIXAS_HAND, right=False, labels=False)
    nomes = [f"[{a:g};{b:g})" for a, b in zip(FAIXAS_HAND[:-1], FAIXAS_HAND[1:])]
    outros = [v for v in MECANISMOS if v != "f_hand"]
    rows = []
    for ev, g in eventos_validos(d.assign(_faixa=faixa)):
        presentes = [i for i in range(1, len(nomes)) if (g._faixa == i).sum() >= 5]
        if (g._faixa == 0).sum() < 5 or not presentes:
            continue
        x = np.column_stack([(g._faixa == i).to_numpy(float) for i in presentes] + [g[outros].to_numpy()])
        b, ep, _ = logistica(x, g.classe.to_numpy(), CFG["lam"])
        for i, bi, ei in zip(presentes, b, ep):
            rows.append({"evento_id": ev, "destino": g.unidade.iloc[0], "fonte": g.fonte.iloc[0],
                         "variavel": f"faixa_{i}", "beta": float(bi), "ep": float(ei)})
    u = agregar_componentes(pd.DataFrame(rows))
    out = {"faixas_m": nomes, "referencia": nomes[0], "por_faixa": {}}
    medias = []
    for i, nome in enumerate(nomes[1:], start=1):
        g = u[u.variavel == f"faixa_{i}"]
        medias.append(float(g.beta.mean()) if len(g) else np.nan)
        out["por_faixa"][nome] = {"componentes": int(len(g)), "log_razao_chances_media": medias[-1],
                                  "ic": bootstrap_media(g.beta.to_numpy()) if len(g) >= CFG["minimo_componentes"] else None,
                                  "meta": meta_dl(g.beta.to_numpy(), g.ep.to_numpy())}
    seq = np.r_[0., medias]
    seq = seq[np.isfinite(seq)]
    out["monotonica_nao_crescente"] = bool(np.all(np.diff(seq) <= 0))
    return out


def c6_cobertura(d):
    d = d.assign(cobertura=d.worldcover_2021.map(COBERTURA))
    out = {}
    for cls, g in d.dropna(subset=["cobertura"]).groupby("cobertura"):
        t = c2_dentro_do_evento(g)
        if t.empty:
            continue
        u = agregar_componentes(t)
        out[cls] = {v: meta_dl(h.beta.to_numpy(), h.ep.to_numpy()) for v, h in u.groupby("variavel")}
    return out


def c7_explicacao_do_modelo():
    sinais = {v: [] for v in MECANISMOS}
    for f in sorted((e.SAIDA / "destinos").glob("*.json")):
        w = e.ler(f)["pesos"]["fisica_cobertura"]["principais"]
        for v in MECANISMOS:
            sinais[v].append(np.sign(w[v]))
    return {v: {"retreinos": len(s), "positivos": int(np.sum(np.array(s) > 0)), "negativos": int(np.sum(np.array(s) < 0))}
            for v, s in sinais.items()}


def c8_forcamento():
    """Leitura dos resultados já encerrados; nenhuma nova estimativa de forçamento."""
    out = {}
    f1 = e.REPO / "local_runs/aprendizado-conjunto/v1/causal_forcamento.json"
    if f1.exists():
        j = e.ler(f1)["por_fonte"]
        out["autopareamento_centrais"] = {
            "fonte": str(f1.relative_to(e.REPO)), "sha256": e.sha256_arquivo(f1), "desfecho": "notificação, não inundação física",
            **{k: {"rr_por_10mm": j[k]["principal"]["rr_por_10mm"], "ic95": j[k]["principal"]["ic95_estratos"],
                   "controle_30d_depois_rr": j[k]["controle_negativo_chuva_30d_depois"]["rr_por_10mm"],
                   "controle_30d_depois_ic95": j[k]["controle_negativo_chuva_30d_depois"].get("ic95_estratos")}
               for k in ("nyc", "sao_paulo") if k in j}}
    f2 = e.REPO / "local_runs/chuva-deteccao-mesmo-lugar/v2_1_era5_estavel/resultado.json"
    if f2.exists():
        j = e.ler(f2)["analises"]["primario20"]
        out["chuva72_deteccao_sar"] = {
            "fonte": str(f2.relative_to(e.REPO)), "sha256": e.sha256_arquivo(f2), "desfecho": "detecção SAR, não inundação validada",
            "rr_por_10mm": j["chuva72h"]["rr_por10mm"]["p72"], "controle_futuro": j["controle_futuro"]["rr_por10mm"]["futuro30"],
            "componentes": j["chuva72h"]["componentes"]}
    return out


# ------------------------------------------------------------------ leitura

def classificar(c2, tri, local):
    out = {}
    for v in MECANISMOS:
        m = c2[v]
        direcao = m.get("estado") != "not_testable" and (m["ic"][0] > 0 or m["ic"][1] < 0)
        s = int(np.sign(m["media"])) if direcao else 0
        inv = direcao and (m["intervalo_predicao"][0] > 0 or m["intervalo_predicao"][1] < 0)
        fams = tri.get(v, {})
        a_favor = sum(1 for f in fams.values() if "ic" in f and (f["ic"][0] > 0 if s > 0 else f["ic"][1] < 0))
        contra = sum(1 for f in fams.values() if "ic" in f and (f["ic"][1] < 0 if s > 0 else f["ic"][0] > 0))
        triang = direcao and a_favor >= CFG["minimo_familias_triangulacao"] and contra == 0
        lm = local.get(v, {})
        loc = direcao and "ic" in lm and (lm["ic"][0] > 0 if s > 0 else lm["ic"][1] < 0)
        nivel = ("mecanismo_local_triangulado" if direcao and triang and loc else
                 "associacao_triangulada_sem_contraste_local" if direcao and triang else
                 "associacao_dependente_do_instrumento" if direcao else "nao_sustentado")
        out[v] = {"nivel": nivel, "sinal_observado": s, "sinal_hidraulico_previsto": SINAL_HIDRAULICO[v],
                  "concorda_com_hidraulica": bool(s == SINAL_HIDRAULICO[v]) if s else None,
                  "direcao": bool(direcao), "invariante": bool(inv), "triangulada": bool(triang),
                  "familias_a_favor": a_favor, "familias_contra": contra, "contraste_local_3km": bool(loc)}
    return out


def triangulacao(t):
    """Por família de instrumento: componentes quando há >= 3; senão eventos (declarado)."""
    out = {}
    for fam, g in t.groupby("fonte"):
        u = agregar_componentes(g)
        nivel_unidade = "componente" if u.destino.nunique() >= 3 else "evento"
        base = u if nivel_unidade == "componente" else g
        for v, h in base.groupby("variavel"):
            out.setdefault(v, {})[fam] = {**meta_dl(h.beta.to_numpy(), h.ep.to_numpy()), "unidade": nivel_unidade,
                                          "familia": FAMILIAS.get(fam, fam)}
    return out


def preparar():
    arq = SAIDA / "protocolo.json"
    codigos = [Path(__file__), Path(t8.__file__), Path(e.__file__), Path(e.mt.e1.__file__)]
    p = {"tipo": "MT09_protocolo_antes_das_estimativas", "config": CFG, "mecanismos": MECANISMOS,
         "sinal_hidraulico_previsto": SINAL_HIDRAULICO, "familias": FAMILIAS, "faixas_hand_m": [str(x) for x in FAIXAS_HAND],
         "cobertura": {str(k): v for k, v in COBERTURA.items()},
         "base_sha256": e.sha256_arquivo(e.SAIDA / "base.parquet"), "coordenadas_sha256": e.sha256_arquivo(t8.COORDENADAS),
         "codigo_sha256": {str(c.relative_to(e.REPO)): e.sha256_arquivo(c) for c in codigos},
         "pressupostos": ["amostragem dentro do evento depende só do rótulo (verificado no código de amostragem; C1 testa a consequência)",
                          "erro de rótulo não diferencial na escala do par, condicional ao instrumento (C3 e C6 sondam)",
                          "sem confundidor que varie dentro de 3 km e cause terreno medido e inundação (C4 remove os de escala maior)",
                          "positividade: pares só onde há positivo e negativo a <= 3 km"],
         "ameacas_nao_removidas": ["infraestrutura local (dique, galeria, bomba) correlacionada com o relevo dentro de 3 km",
                                   "erro do MDT correlacionado com cobertura na escala do pixel (DSM sob vegetação/prédio)",
                                   "componentes e eventos são os conhecidos; não é confirmação independente"],
         "regime": "desenvolvimento retrospectivo; identificação estrutural-condicional, não efeito de intervenção",
         "fumaca_antes": "um evento por família, só para erro de código; nenhuma estimativa impressa ou lida",
         "reservas_lidas": False, "modelo_promovido": False}
    if arq.exists():
        old = e.ler(arq)
        if {k: v for k, v in old.items() if k != "quando_utc"} != e.mt.rel.limpo(p):
            raise ValueError("protocolo mudou; versionar")
        return old
    p["quando_utc"] = e.agora_utc()
    e.escrever(arq, p)
    return p


def executar():
    p = preparar()
    t0 = time.monotonic()
    d, info = carregar()
    c2 = c2_dentro_do_evento(d)
    u2 = agregar_componentes(c2)
    c4 = {r: c4_pares(d, r) for r in (CFG["raio_km"], CFG["raio_sensibilidade_km"])}
    marginais = {}
    for v in MECANISMOS:
        u = agregar_componentes(c2_dentro_do_evento(d, [v]))
        marginais[v] = meta_dl(u.beta.to_numpy(), u.ep.to_numpy())
    res_c2 = {v: meta_dl(g.beta.to_numpy(), g.ep.to_numpy()) for v, g in u2.groupby("variavel")}
    res_c4 = {}
    for r, t in c4.items():
        u = agregar_componentes(t)
        res_c4[f"{r:g}km"] = {v: {**meta_dl(g.beta.to_numpy(), g.ep.to_numpy()), "eventos": int(t[t.variavel == v].evento_id.nunique())}
                              for v, g in u.groupby("variavel")}
    tri = triangulacao(c2)
    tri_local = triangulacao(c4[CFG["raio_km"]])
    SAIDA.mkdir(parents=True, exist_ok=True)
    c2.to_parquet(SAIDA / "c2_por_evento.parquet", index=False)
    pd.concat([t.assign(raio_km=r) for r, t in c4.items()]).to_parquet(SAIDA / "c4_pares_por_evento.parquet", index=False)
    r = {"tipo": "MT09_resultado", "quando_utc": e.agora_utc(), "protocolo_sha256": e.sha256_arquivo(SAIDA / "protocolo.json"),
         "dados": info, "eventos_c2": int(c2.evento_id.nunique()), "componentes_c2": int(c2.destino.nunique()),
         "C1_amostragem_por_classe": c1_amostragem(d), "C2_dentro_do_evento": res_c2, "C2_marginais": marginais,
         "C3_triangulacao": tri, "C3_triangulacao_pares_3km": tri_local, "C4_pares_vizinhos": res_c4,
         "C5_dose_resposta_hand": c5_dose_resposta(d), "C6_cobertura_exploratorio": c6_cobertura(d),
         "C7_sinais_do_modelo_Q_conexao": c7_explicacao_do_modelo(), "C8_forcamento_fontes": c8_forcamento(),
         "classificacao": classificar(res_c2, tri, res_c4[f"{CFG['raio_km']:g}km"]),
         "segundos": time.monotonic() - t0, "regime": p["regime"], "ameacas_nao_removidas": p["ameacas_nao_removidas"],
         "modelo_promovido": False}
    e.escrever(SAIDA / "resultado.json", r)
    return {"classificacao": r["classificacao"], "segundos": r["segundos"]}


def fumaca():
    """Executa todos os desenhos em um evento por família só para achar erro de código; não imprime estimativa."""
    d, _ = carregar()
    evs = d.groupby("fonte").evento_id.first().tolist()
    s = d[d.evento_id.isin(evs)]
    c2 = c2_dentro_do_evento(s)
    c4 = c4_pares(s, CFG["raio_km"])
    partes = [agregar_componentes(c2), triangulacao(c2), c1_amostragem(s), c5_dose_resposta(s), c6_cobertura(s),
              c7_explicacao_do_modelo(), c8_forcamento()]
    return {"eventos": len(evs), "linhas_c2": len(c2), "linhas_c4": len(c4), "partes_sem_erro": len(partes),
            "estimativas_impressas": False}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["preparar", "fumaca", "executar"])
    a = ap.parse_args()
    print(json.dumps({"preparar": preparar, "fumaca": fumaca, "executar": executar}[a.acao](), ensure_ascii=False, default=str))
