"""Treino e avaliacao de todos os bracos do protocolo v1 (encostas, Idai -> Petropolis).

Recusa rodar sem o protocolo congelado ou com amostras diferentes das congeladas.
Grava pesos de cada ajuste, previsoes individuais, metricas por braco/semente,
contrastes com bootstrap, retencao, ablacoes e diagnosticos.

Nada aqui olha o teste para escolher: lambda e hiperparametros do boosting saem
do desenvolvimento Idai; orcamentos, sementes e precisoes estao no protocolo.
"""
from __future__ import annotations

import copy
import json
import pickle
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from pa_comum import (ESTUDO, OUT, SEMENTE, SVC03_TIF, EPSG_PETROPOLIS, TERRENO, Etapa, gravar_json,
                      ler_json, rel, sha256)

REF = OUT / "referencias"
RES = OUT / "resultados"
PESOS = OUT / "pesos"
PROT = ler_json(OUT / "protocolo" / "protocolo_v1.json")
F = PROT["atributos"]
BLOCOS = {"declive": ["decl_m3", "decl_max5"], "curvatura": ["curv_plan_m3", "curv_perf_m3"],
          "fluxo": ["logsca_m3"], "relevo": ["relevo7", "tpi11"]}
LAMBDAS_DOADOR = [0.01, 0.1, 1.0, 10.0, 100.0]
GRADE_HGB = [{"max_depth": d, "learning_rate": lr} for d in (3, 6) for lr in (0.05, 0.1)]
ORCAMENTOS = PROT["orcamentos"]["B"] + ["todos"]
SEMENTES = list(range(PROT["orcamentos"]["sementes"]))
N_BOOT = 2000
LOCAL_HGB = {"learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 10}
ITER_LOCAL = 50


# ------------------------------------------------------------------ modelos
def ajustar_log(X, y, lam, beta0=None):
    """Logistica com prior gaussiano centrado em beta0 (precisao lam); intercepto livre."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    beta0 = np.zeros(p) if beta0 is None else np.asarray(beta0, float)
    npos, nneg = y.sum(), (1 - y).sum()
    w = np.where(y == 1, 1.0, npos / nneg)

    def f(theta):
        z = theta[0] + X @ theta[1:]
        perda = np.sum(w * (np.logaddexp(0, z) - y * z))
        d = theta[1:] - beta0
        sig = 1 / (1 + np.exp(-z))
        g0 = np.sum(w * (sig - y))
        g = X.T @ (w * (sig - y)) + lam * d
        return perda + 0.5 * lam * d @ d, np.concatenate([[g0], g])

    x0 = np.concatenate([[0.0], beta0])
    r = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-8})
    return {"intercepto": float(r.x[0]), "coef": r.x[1:].tolist(), "convergiu": bool(r.success),
            "iteracoes": int(r.nit), "objetivo": float(r.fun), "mensagem": str(r.message), "lambda": lam,
            "beta0": beta0.tolist()}


def perda_log(m, X, y, beta0=None):
    z = m["intercepto"] + X @ np.asarray(m["coef"])
    npos, nneg = y.sum(), (1 - y).sum()
    w = np.where(y == 1, 1.0, npos / nneg)
    return float(np.sum(w * (np.logaddexp(0, z) - y * z)) / w.sum())


def escore_log(m, X):
    return m["intercepto"] + np.asarray(X, float) @ np.asarray(m["coef"])


def pesos_bal(y):
    y = np.asarray(y)
    return np.where(y == 1, 1.0, y.sum() / (1 - y).sum())


# ------------------------------------------------------------------ metricas
def pct_abaixo(v_pos, v_bg, w_bg=None):
    ordem = np.argsort(v_bg, kind="mergesort")
    s = v_bg[ordem]
    w = np.ones_like(s) if w_bg is None else w_bg[ordem]
    cw = np.cumsum(w)
    tot = cw[-1]
    esq = np.searchsorted(s, v_pos, "left")
    dir_ = np.searchsorted(s, v_pos, "right")
    abaixo = np.where(esq > 0, cw[np.maximum(esq - 1, 0)], 0.0)
    ate = np.where(dir_ > 0, cw[np.maximum(dir_ - 1, 0)], 0.0)
    return (abaixo + 0.5 * (ate - abaixo)) / tot


def auc_obj(v_pos, cic_idx, n_cic, v_bg, w_bg=None, w_cic=None):
    pct = pct_abaixo(v_pos, v_bg, w_bg)
    por = np.bincount(cic_idx, pct, n_cic) / np.bincount(cic_idx, minlength=n_cic)
    if w_cic is None:
        return float(por.mean()), por
    return float(np.sum(w_cic * por) / w_cic.sum()), por


def metricas_t1(v, T):
    pos = T["pos"]
    a, por = auc_obj(v[pos], T["cic_idx"], T["n_cic"], v[~pos])
    out = {"AUC_obj": a, "AUC_cel": float(roc_auc_score(T["y"], v)),
           "AP_cel": float(average_precision_score(T["y"], v)), "prevalencia": float(T["y"].mean())}
    for q in (0.1, 0.2, 0.3):
        out[f"rec_cic_{int(q * 100)}"] = float(np.mean(por >= 1 - q))
    return out, por


# ------------------------------------------------------------------ dados
def carregar():
    refs = ler_json(REF / "referencias.json")
    for nome, h in PROT["amostras_sha256"].items():
        if sha256(REF / nome) != h or refs["hashes"][nome] != h:
            raise SystemExit(f"amostra {nome} difere da congelada no protocolo")
    d = {k: pd.read_parquet(REF / f) for k, f in [
        ("idai", "amostra_idai.parquet"), ("teste", "teste_petropolis_2022.parquet"),
        ("pool", "pool_atualizacao_local.parquet"), ("dom", "dominio_relato.parquet"),
        ("hist", "teste_historico_sitios.parquet"), ("mun", "fundo_municipio_diagnostico.parquet")]}
    return d


def preparar_t1(teste):
    y = teste.y.values.astype(int)
    pos = y == 1
    cic = teste.id_cicatriz.values[pos]
    unicos = np.unique(cic)
    idx = np.searchsorted(unicos, cic)
    return {"y": y, "pos": pos, "cic_idx": idx, "n_cic": len(unicos), "cic_ids": unicos,
            "bloco_bg": teste.bloco500.values[~pos]}


def bootstrap_t1(vetores: dict, T, n=N_BOOT, semente=SEMENTE):
    """AUC_obj por reamostragem conjunta de cicatrizes e blocos de fundo de 500 m.

    vetores: nome -> lista de vetores de escore (sementes). Devolve nome -> matriz (n, sementes).
    """
    rng = np.random.default_rng(semente + 7)
    pos = T["pos"]
    blocos, inv = np.unique(T["bloco_bg"], return_inverse=True)
    prep = {}
    for nome, lista in vetores.items():
        prep[nome] = []
        for v in lista:
            vb = v[~pos]
            ordem = np.argsort(vb, kind="mergesort")
            prep[nome].append((v[pos], vb[ordem], inv[ordem]))
    saida = {nome: np.zeros((n, len(lista))) for nome, lista in vetores.items()}
    for r in range(n):
        wc = np.bincount(rng.integers(0, T["n_cic"], T["n_cic"]), minlength=T["n_cic"]).astype(float)
        wb_bloco = np.bincount(rng.integers(0, len(blocos), len(blocos)), minlength=len(blocos)).astype(float)
        for nome, lista in prep.items():
            for j, (vp, vb_s, inv_s) in enumerate(lista):
                w = wb_bloco[inv_s]
                cw = np.cumsum(w)
                tot = cw[-1]
                esq = np.searchsorted(vb_s, vp, "left")
                dir_ = np.searchsorted(vb_s, vp, "right")
                abaixo = np.where(esq > 0, cw[np.maximum(esq - 1, 0)], 0.0)
                ate = np.where(dir_ > 0, cw[np.maximum(dir_ - 1, 0)], 0.0)
                pct = (abaixo + 0.5 * (ate - abaixo)) / tot
                por = np.bincount(T["cic_idx"], pct, T["n_cic"]) / np.bincount(T["cic_idx"], minlength=T["n_cic"])
                saida[nome][r, j] = np.sum(wc * por) / wc.sum()
    return saida


def rotulo(media, lo, hi, margem=PROT["regra_de_interpretacao"]["margem"]):
    if media >= margem and lo > 0:
        return "sustentado"
    if media <= -margem and hi < 0:
        return "contrario"
    return "inconclusivo"


def amostrar_agua(teste):
    import rasterio
    from pyproj import Transformer
    with rasterio.open(SVC03_TIF) as r:
        tr = Transformer.from_crs(EPSG_PETROPOLIS, r.crs, always_xy=True)
        xs, ys = tr.transform(teste.x.values, teste.yc.values)
        v = np.array([s[0] for s in r.sample(zip(xs, ys))], float)
        nd = r.nodata
    if nd is not None:
        v[v == nd] = np.nan
    return v


def consultas_em_teste(pool, teste, ajustes) -> dict:
    """Quantos sitios consultados caem em celula de teste ou a menos de 300 m do suporte de teste."""
    celulas_teste = set(zip(teste.l, teste.c))
    por_id = pool.set_index("id")
    consultados = {i for a in ajustes for i in a["ids_sitios"].split(",")}
    em_celula = sum((int(por_id.loc[i, "l"]), int(por_id.loc[i, "c"])) in celulas_teste for i in consultados)
    xmin, xmax, ymin, ymax = teste.x.min() - 300, teste.x.max() + 300, teste.yc.min() - 300, teste.yc.max() + 300
    na_caixa = int(((por_id.loc[list(consultados), "x"].between(xmin, xmax)) &
                    (por_id.loc[list(consultados), "yc"].between(ymin, ymax))).sum())
    return {"sitios_consultados_distintos": len(consultados), "em_celula_de_teste": int(em_celula),
            "na_caixa_envolvente_do_teste_mais_300m": na_caixa,
            "nota": "a exclusao real e a envoltoria convexa + 300 m (aplicada no pool); a caixa e mais larga e so diagnostica"}


# ------------------------------------------------------------------ principal
def main() -> int:
    e = Etapa("pa_05_treino")
    t_ini = time.time()
    for p in (RES, PESOS, PESOS / "hgb"):
        p.mkdir(parents=True, exist_ok=True)
    d = carregar()
    idai, teste, pool, dom, hist, mun = (d[k] for k in ("idai", "teste", "pool", "dom", "hist", "mun"))
    aj = idai[idai.papel.eq("ajuste")]
    dev = idai[idai.papel.eq("desenvolvimento")]
    res = idai[idai.papel.eq("reserva_origem")].reset_index(drop=True)
    mu, sd = aj[F].mean(), aj[F].std(ddof=0)
    Z = lambda df, cols=F: ((df[cols] - mu[cols]) / sd[cols]).values  # noqa: E731
    gravar_json(PESOS / "padronizacao.json", {"media": mu.to_dict(), "desvio": sd.to_dict(),
                                              "origem": "Idai papel=ajuste", "atributos": F})
    T = preparar_t1(teste)
    Xt, Xr = Z(teste), Z(res)
    yr = res.y.values
    dom_bg_hist = dom[~dom.perto_deslizamento]
    Xh = np.vstack([Z(hist), Z(dom_bg_hist)])
    yh = np.r_[np.ones(len(hist)), np.zeros(len(dom_bg_hist))]

    pred_t = {"id": teste.id.values, "y": T["y"], "id_cicatriz": teste.id_cicatriz.values,
              "bloco500": teste.bloco500.values}
    pred_r = {"id": res.id.values, "y": yr, "bloco": res.bloco.values}
    pred_h = {"id": np.r_[hist.id.values, dom_bg_hist.id.values], "y": yh}
    metricas, ajustes, ablac = [], [], []
    custo = {}

    def avaliar(nome, vt, vr=None, vh=None, **meta):
        m, _ = metricas_t1(vt, T)
        if vr is not None:
            m["AUC_idai_reserva"] = float(roc_auc_score(yr, vr))
        if vh is not None:
            m["AUC_T_hist"] = float(roc_auc_score(yh, vh))
        m.update({"braco": nome, **meta})
        metricas.append(m)
        return m

    # ---------------- A0
    j0 = F.index("decl_m3")
    pred_t["A0_regra"], pred_r["A0_regra"], pred_h["A0_regra"] = Xt[:, j0], Xr[:, j0], Xh[:, j0]
    avaliar("A0_regra", Xt[:, j0], Xr[:, j0], Xh[:, j0], B=0, semente=-1)

    # ---------------- A1 logistica: lambda no desenvolvimento Idai
    t0 = time.time()
    sel = []
    for lam in LAMBDAS_DOADOR:
        m = ajustar_log(Z(aj), aj.y.values, lam)
        sel.append({"lambda": lam, "AUC_dev": float(roc_auc_score(dev.y, escore_log(m, Z(dev)))),
                    "convergiu": m["convergiu"]})
    melhor = max(sel, key=lambda s: (round(s["AUC_dev"], 6), s["lambda"]))
    a1 = ajustar_log(Z(aj), aj.y.values, melhor["lambda"])
    a1.update({"selecao_lambda_dev_idai": sel, "atributos": F, "fit_id": "A1_log"})
    gravar_json(PESOS / "A1_log.json", a1)
    custo["A1_log_s"] = round(time.time() - t0, 2)
    for nome, X_, alvo in [("t", Xt, pred_t), ("r", Xr, pred_r), ("h", Xh, pred_h)]:
        alvo["A1_log"] = escore_log(a1, X_)
    avaliar("A1_log", pred_t["A1_log"], pred_r["A1_log"], pred_h["A1_log"], B=0, semente=-1)
    beta_a1 = np.asarray(a1["coef"])

    # ---------------- A1 boosting
    t0 = time.time()
    sel_h = []
    for g in GRADE_HGB:
        m = HistGradientBoostingClassifier(max_iter=300, min_samples_leaf=50, early_stopping=False,
                                           random_state=SEMENTE, **g)
        m.fit(Z(aj), aj.y.values, sample_weight=pesos_bal(aj.y.values))
        sel_h.append({**g, "AUC_dev": float(roc_auc_score(dev.y, m.decision_function(Z(dev))))})
    gh = max(sel_h, key=lambda s: s["AUC_dev"])
    a1h = HistGradientBoostingClassifier(max_iter=300, min_samples_leaf=50, early_stopping=False,
                                         random_state=SEMENTE, max_depth=gh["max_depth"],
                                         learning_rate=gh["learning_rate"])
    a1h.fit(Z(aj), aj.y.values, sample_weight=pesos_bal(aj.y.values))
    with open(PESOS / "A1_hgb.pkl", "wb") as f:
        pickle.dump(a1h, f)
    gravar_json(PESOS / "A1_hgb_selecao.json", {"grade": sel_h, "escolhido": gh, "iteracoes": 300,
                                                "min_samples_leaf": 50, "sha256_pkl": sha256(PESOS / "A1_hgb.pkl")})
    custo["A1_hgb_s"] = round(time.time() - t0, 2)
    for X_, alvo in [(Xt, pred_t), (Xr, pred_r), (Xh, pred_h)]:
        alvo["A1_hgb"] = a1h.decision_function(X_)
    avaliar("A1_hgb", pred_t["A1_hgb"], pred_r["A1_hgb"], pred_h["A1_hgb"], B=0, semente=-1)

    # ---------------- ablacoes descritivas (A1_log sem cada bloco)
    for bloco, cols in BLOCOS.items():
        resto = [c for c in F if c not in cols]
        m = ajustar_log(Z(aj, resto), aj.y.values, melhor["lambda"])
        m.update({"sem_bloco": bloco, "atributos": resto})
        gravar_json(PESOS / f"A1_log_sem_{bloco}.json", m)
        idx = [F.index(c) for c in resto]
        vt, vr, vh = (escore_log(m, X_[:, idx]) for X_ in (Xt, Xr, Xh))
        nome = f"A1_log_sem_{bloco}"
        pred_t[nome], pred_r[nome], pred_h[nome] = vt, vr, vh
        mm = avaliar(nome, vt, vr, vh, B=0, semente=-1)
        ablac.append({"bloco_removido": bloco, **{k: mm[k] for k in ("AUC_obj", "AUC_idai_reserva", "AUC_T_hist")}})

    # ---------------- ajustes locais
    dom_ok = dom[~dom.dentro_exclusao & ~dom.perto_deslizamento].reset_index(drop=True)
    t_local = time.time()
    for B in ORCAMENTOS:
        for s in SEMENTES:
            rng = np.random.default_rng(SEMENTE + 1000 * (999 if B == "todos" else B) + s)
            sit = pool if B == "todos" else pool.iloc[np.sort(rng.choice(len(pool), B, replace=False))]
            nb = 3 * len(sit)
            bg = dom_ok.iloc[np.sort(rng.choice(len(dom_ok), nb, replace=False))]
            X = np.vstack([Z(sit), Z(bg)])
            y = np.r_[np.ones(len(sit)), np.zeros(len(bg))]
            perm = rng.permutation(len(F))
            while np.all(perm == np.arange(len(F))):
                perm = rng.permutation(len(F))
            bracos_log = {"A2_log_l1": (1.0, None), "A2_log_l25": (25.0, None),
                          "A3_log_l1": (1.0, beta_a1), "A3_log_l25": (25.0, beta_a1),
                          "A3perm_log_l25": (25.0, beta_a1[perm])}
            base = {"B": B, "semente": s, "n_sitios": len(sit), "n_relatos": int(sit.n_relatos.sum()),
                    "n_fundo": nb, "consultas": len(sit), "ids_sitios": ",".join(sit.id),
                    "ids_fundo_sha256": __import__("hashlib").sha256(",".join(bg.id).encode()).hexdigest()}
            for nome, (lam, b0) in bracos_log.items():
                t0 = time.time()
                m = ajustar_log(X, y, lam, b0)
                fid = f"{nome}__B{B}__s{s}"
                vt, vr = escore_log(m, Xt), escore_log(m, Xr)
                pred_t[fid], pred_r[fid] = vt, vr
                antes = {"intercepto": 0.0, "coef": (np.zeros(len(F)) if b0 is None else b0).tolist()}
                if b0 is not None:
                    # estado anterior = prior com o intercepto refeito localmente (so desloca o nivel)
                    antes["intercepto"] = float(ajustar_log(X, y, 1e8, b0)["intercepto"])
                ajustes.append({**base, "fit_id": fid, "braco": nome, "lambda": lam,
                                "permutacao": perm.tolist() if nome.startswith("A3perm") else None,
                                "convergiu": m["convergiu"], "iteracoes": m["iteracoes"],
                                "intercepto": m["intercepto"], **{f"coef_{c}": v for c, v in zip(F, m["coef"])},
                                **({f"beta0_{c}": v for c, v in zip(F, b0)} if b0 is not None else {}),
                                "mudanca_norma_coef": float(np.linalg.norm(np.asarray(m["coef"]) - np.asarray(antes["coef"]))),
                                "perda_treino_antes": perda_log(antes, X, y) if b0 is not None else None,
                                "perda_treino_depois": perda_log(m, X, y),
                                "auc_treino_depois": float(roc_auc_score(y, escore_log(m, X))),
                                "auc_treino_antes": float(roc_auc_score(y, escore_log(antes, X))) if b0 is not None else None,
                                "parede_s": round(time.time() - t0, 3)})
                avaliar(nome, vt, vr, None, B=B, semente=s, fit_id=fid)
            # boosting local e adaptado: mesmo numero de iteracoes e parametros
            for nome in ("A2_hgb", "A3_hgb"):
                t0 = time.time()
                if nome == "A2_hgb":
                    m = HistGradientBoostingClassifier(max_iter=ITER_LOCAL, early_stopping=False,
                                                       random_state=SEMENTE + s, **LOCAL_HGB)
                else:
                    m = copy.deepcopy(a1h)
                    m.set_params(warm_start=True, max_iter=a1h.n_iter_ + ITER_LOCAL, **LOCAL_HGB)
                m.fit(X, y, sample_weight=pesos_bal(y))
                fid = f"{nome}__B{B}__s{s}"
                arq = PESOS / "hgb" / f"{fid}.pkl"
                with open(arq, "wb") as f:
                    pickle.dump(m, f)
                vt, vr = m.decision_function(Xt), m.decision_function(Xr)
                pred_t[fid], pred_r[fid] = vt, vr
                antes_auc = float(roc_auc_score(y, a1h.decision_function(X))) if nome == "A3_hgb" else None
                ajustes.append({**base, "fit_id": fid, "braco": nome, "lambda": None, "convergiu": True,
                                "iteracoes": int(m.n_iter_), "pkl": rel(arq), "pkl_sha256": sha256(arq),
                                "auc_treino_antes": antes_auc, "auc_treino_depois": float(roc_auc_score(y, m.decision_function(X))),
                                "parede_s": round(time.time() - t0, 3)})
                avaliar(nome, vt, vr, None, B=B, semente=s, fit_id=fid)
        print(f"[local] B={B} pronto ({time.time() - t_local:.0f}s)", flush=True)
    custo["ajustes_locais_s"] = round(time.time() - t_local, 1)

    # ---------------- D1 vies de fundo: municipio inteiro como fundo
    d1 = []
    for s in SEMENTES:
        rng = np.random.default_rng(SEMENTE + 555_000 + s)
        sit = pool.iloc[np.sort(rng.choice(len(pool), 300, replace=False))]
        bg = mun.iloc[np.sort(rng.choice(len(mun), 900, replace=False))]
        m = ajustar_log(np.vstack([Z(sit), Z(bg)]), np.r_[np.ones(300), np.zeros(900)], 1.0)
        vt = escore_log(m, Xt)
        mm, _ = metricas_t1(vt, T)
        d1.append({"semente": s, "AUC_obj": mm["AUC_obj"], **{f"coef_{c}": v for c, v in zip(F, m["coef"])}})
    # D2 agua
    agua = amostrar_agua(teste)
    ok = np.isfinite(agua)
    d2 = None
    if ok.all():
        mm, _ = metricas_t1(agua, T)
        d2 = {"cobertura": 1.0, **mm}
    else:
        Tm = preparar_t1(teste[ok].reset_index(drop=True))
        mm, _ = metricas_t1(agua[ok], Tm)
        d2 = {"cobertura": float(ok.mean()), **mm}
    pred_t["D2_agua_svc03"] = agua

    # ---------------- D4 dominio
    q = aj[F].quantile([0.01, 0.99])
    dentro = lambda df: ((df[F] >= q.loc[0.01]) & (df[F] <= q.loc[0.99]))  # noqa: E731
    d4 = {"P1_P99_doador": q.to_dict(),
          "frac_dentro_por_atributo_teste": dentro(teste).mean().to_dict(),
          "frac_todas_dentro_teste": float(dentro(teste).all(1).mean()),
          "frac_todas_dentro_pool": float(dentro(pool).all(1).mean()),
          "frac_todas_dentro_fundo_municipio": float(dentro(mun).all(1).mean()),
          "medianas": {"idai_ajuste": aj[F].median().to_dict(), "petropolis_teste": teste[F].median().to_dict(),
                       "pool_local": pool[F].median().to_dict(), "cicatrizes_2022": teste[teste.y == 1][F].median().to_dict(),
                       "fontes_idai": aj[aj.y == 1][F].median().to_dict()}}

    # ---------------- gravar previsoes e metricas
    pd.DataFrame(pred_t).to_parquet(RES / "pred_teste_2022.parquet", index=False)
    pd.DataFrame(pred_r).to_parquet(RES / "pred_idai_reserva.parquet", index=False)
    pd.DataFrame(pred_h).to_parquet(RES / "pred_teste_historico.parquet", index=False)
    met = pd.DataFrame(metricas)
    met.to_csv(RES / "metricas_bracos.csv", index=False)
    pd.DataFrame(ajustes).to_csv(RES / "ajustes_locais.csv", index=False)
    pd.DataFrame(ablac).to_csv(RES / "ablacoes.csv", index=False)
    pd.DataFrame(d1).to_csv(RES / "D1_vies_fundo.csv", index=False)

    # ---------------- contrastes com bootstrap
    t0 = time.time()
    vet = {"A0_regra": [pred_t["A0_regra"]], "A1_log": [pred_t["A1_log"]], "A1_hgb": [pred_t["A1_hgb"]]}
    locais = ["A2_log_l1", "A2_log_l25", "A3_log_l1", "A3_log_l25", "A3perm_log_l25", "A2_hgb", "A3_hgb"]
    for B in ORCAMENTOS:
        for nome in locais:
            vet[f"{nome}__B{B}"] = [pred_t[f"{nome}__B{B}__s{s}"] for s in SEMENTES]
    boot = bootstrap_t1(vet, T)
    pont = {k: np.mean([metricas_t1(v, T)[0]["AUC_obj"] for v in lista]) for k, lista in vet.items()}
    por_sem = {k: np.array([metricas_t1(v, T)[0]["AUC_obj"] for v in lista]) for k, lista in vet.items()}
    pares = []
    for B in ORCAMENTOS:
        pares += [
            (B, "C1_transferencia", "A1_log", "A0_regra"),
            (B, "C2_aprendizado_local", f"A3_log_l25__B{B}", "A1_log"),
            (B, "C3_valor_conhecimento_externo", f"A3_log_l25__B{B}", f"A2_log_l25__B{B}"),
            (B, "C4_controle_estrutura", f"A3_log_l25__B{B}", f"A3perm_log_l25__B{B}"),
            (B, "local_so_vs_regra", f"A2_log_l25__B{B}", "A0_regra"),
            (B, "C3_lambda1", f"A3_log_l1__B{B}", f"A2_log_l1__B{B}"),
            (B, "C2_lambda1", f"A3_log_l1__B{B}", "A1_log"),
            (B, "hgb_transferencia", "A1_hgb", "A0_regra"),
            (B, "hgb_aprendizado_local", f"A3_hgb__B{B}", "A1_hgb"),
            (B, "hgb_valor_externo", f"A3_hgb__B{B}", f"A2_hgb__B{B}"),
            (B, "nao_linear_vs_linear_adaptado", f"A3_hgb__B{B}", f"A3_log_l25__B{B}"),
        ]
    contr = []
    for B, nome, a, b in pares:
        if nome in ("C1_transferencia", "hgb_transferencia") and B != ORCAMENTOS[0]:
            continue
        diff_boot = boot[a].mean(1) - boot[b].mean(1)
        lo, hi = np.percentile(diff_boot, [2.5, 97.5])
        media = pont[a] - pont[b]
        da, db = por_sem[a], por_sem[b]
        frac = float(np.mean((da - db) > 0)) if len(da) == len(db) else float(np.mean(da > db.mean()))
        primario = (nome.startswith("C") and "lambda" not in nome and
                    (nome == "C1_transferencia" or B == PROT["contrastes_primarios"]["orcamento"]))
        contr.append({"B": (0 if nome.endswith("transferencia") else B), "contraste": nome, "a": a, "b": b,
                      "AUC_obj_a": pont[a], "AUC_obj_b": pont[b], "delta": media, "ic95_lo": lo, "ic95_hi": hi,
                      "frac_sementes_positivas": frac, "primario": primario, "rotulo_regra": rotulo(media, lo, hi)})
    cdf = pd.DataFrame(contr)
    cdf.to_csv(RES / "contrastes_T1.csv", index=False)
    resumo_boot = {k: {"media": float(v.mean(1).mean()), "ic95": np.percentile(v.mean(1), [2.5, 97.5]).tolist()}
                   for k, v in boot.items()}
    custo["bootstrap_s"] = round(time.time() - t0, 1)

    # ---------------- retencao
    ret = []
    for B in ORCAMENTOS:
        for nome, ref_ in [("A3_log_l25", "A1_log"), ("A3_log_l1", "A1_log"), ("A3perm_log_l25", "A1_log"),
                           ("A3_hgb", "A1_hgb"), ("A2_log_l25", "A1_log"), ("A2_hgb", "A1_hgb")]:
            v = met[(met.braco == nome) & (met.B == B)].AUC_idai_reserva
            r0 = met[(met.braco == ref_) & (met.semente == -1)].AUC_idai_reserva.iloc[0]
            ret.append({"B": B, "braco": nome, "referencia": ref_, "AUC_reserva_media": float(v.mean()),
                        "AUC_reserva_min": float(v.min()), "AUC_reserva_max": float(v.max()),
                        "AUC_referencia": float(r0), "delta": float(v.mean() - r0),
                        "custo_retencao": bool(v.mean() - r0 <= -0.02)})
    pd.DataFrame(ret).to_csv(RES / "retencao_idai.csv", index=False)

    resultado = {
        "protocolo": PROT["id"], "lambda_doador": melhor["lambda"], "hgb_doador": gh,
        "A1_log_coef": dict(zip(F, a1["coef"])), "A1_log_intercepto": a1["intercepto"],
        "D2_agua": d2, "D4_dominio": d4, "bootstrap_resumo": resumo_boot,
        "contrastes_primarios": cdf[cdf.primario].to_dict("records"),
        "n_ajustes_locais": len(ajustes), "nao_convergiu": int((~pd.DataFrame(ajustes).convergiu).sum()),
        "custo": {**custo, "total_s": round(time.time() - t_ini, 1)},
        "consultas_em_teste": consultas_em_teste(pool, teste, ajustes),
    }
    gravar_json(RES / "resultado_treino.json", resultado)
    e.saida(*RES.glob("*"), *PESOS.glob("*.json"), PESOS / "A1_hgb.pkl")
    e.fechar(custo=resultado["custo"], n_ajustes=len(ajustes))
    print(cdf[cdf.primario].to_string())
    print(json.dumps(resultado["custo"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
