"""Executa a emenda 1: D5 (receptor no teste historico) e E1 (cicatrizes 2022, CV por bairro).

Le a emenda congelada e recusa rodar sem ela. Grava pesos, previsoes fora da dobra,
metricas, contrastes com bootstrap dentro da dobra, retencao e resultados por comunidade.
"""
from __future__ import annotations

import copy
import pickle
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from pa_comum import CHARTER, EPSG_PETROPOLIS, ESTUDO, OUT, SEMENTE, TERRENO, Etapa, gravar_json, ler_json, rel, sha256
from pa_03_referencias import _raster, _rc, ufrj_gdf
from pa_05_treino import escore_log, rotulo

REF = OUT / "referencias"
RES = OUT / "resultados" / "emenda1"
PESOS = OUT / "pesos" / "emenda1"
EM = ler_json(OUT / "protocolo" / "emenda_1.json")
PROT = ler_json(OUT / "protocolo" / "protocolo_v1.json")
F = PROT["atributos"]
N_BOOT = 2000
RAIO_BUFFER = 150.0
LOCAL_HGB = {"learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 10}
D8 = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}


def ajustar_log_w(X, y, w, lam, beta0=None):
    X, y, w = np.asarray(X, float), np.asarray(y, float), np.asarray(w, float)
    beta0 = np.zeros(X.shape[1]) if beta0 is None else np.asarray(beta0, float)

    def f(t):
        z = t[0] + X @ t[1:]
        d = t[1:] - beta0
        sig = 1 / (1 + np.exp(-z))
        return (np.sum(w * (np.logaddexp(0, z) - y * z)) + 0.5 * lam * d @ d,
                np.concatenate([[np.sum(w * (sig - y))], X.T @ (w * (sig - y)) + lam * d]))
    r = minimize(f, np.concatenate([[0.0], beta0]), jac=True, method="L-BFGS-B",
                 options={"maxiter": 2000, "gtol": 1e-8})
    return {"intercepto": float(r.x[0]), "coef": r.x[1:].tolist(), "convergiu": bool(r.success),
            "iteracoes": int(r.nit), "lambda": lam}


def montante_max(valor, ponteiro, passos=7):
    """Maximo de `valor` entre a celula e as que drenam para ela em ate `passos` passos D8."""
    h, w = valor.shape
    li, co = np.indices((h, w))
    dest = np.full(valor.size, -1, np.int64)
    p = ponteiro.ravel()
    for cod, (dl, dc) in D8.items():
        sel = p == cod
        l2, c2 = li.ravel()[sel] + dl, co.ravel()[sel] + dc
        ok = (l2 >= 0) & (l2 < h) & (c2 >= 0) & (c2 < w)
        idx = np.nonzero(sel)[0][ok]
        dest[idx] = l2[ok] * w + c2[ok]
    v = np.where(np.isfinite(valor), valor, -np.inf).ravel()
    src = np.nonzero(dest >= 0)[0]
    atual = v.copy()
    for _ in range(passos):
        novo = atual.copy()
        np.maximum.at(novo, dest[src], atual[src])
        atual = novo
    out = atual.reshape(h, w)
    out[~np.isfinite(valor)] = np.nan
    return out


def d5(Zf, a1):
    import rasterio
    with rasterio.open(TERRENO / "petropolis/atributos_encosta.tif") as r:
        pilha = r.read().astype("float64")
        rr = r
        pnt = rasterio.open(TERRENO / "petropolis/d8_pointer_esri.tif").read(1)
        X = Zf(pilha.reshape(len(F), -1).T)
        s_a1 = (a1["intercepto"] + X @ np.asarray(a1["coef"])).reshape(pilha.shape[1:])
        s_a1[~np.isfinite(pilha).all(0)] = np.nan
        up_a1 = montante_max(s_a1, pnt)
        up_a0 = montante_max(np.where(np.isfinite(pilha).all(0), pilha[0], np.nan), pnt)
        hist = pd.read_parquet(REF / "teste_historico_sitios.parquet")
        dom = pd.read_parquet(REF / "dominio_relato.parquet")
        bg = dom[~dom.perto_deslizamento]
        uf = ufrj_gdf()
        uf = uf[uf.localizado & uf.tipo_de_ev.eq("deslizamento")].copy()
        uf["l"], uf["c"] = _rc(rr, uf.geometry.x, uf.geometry.y)
    uf["preciso"] = uf.precisao_n.isin(["ROOFTOP", "RANGE_INTERPOLATED"])
    prec = uf.groupby(["l", "c"]).preciso.any().rename("preciso").reset_index()
    hist = hist.merge(prec, on=["l", "c"], how="left")
    linhas = []
    for estrato, sub in [("todos", hist), ("preciso", hist[hist.preciso == True]),  # noqa: E712
                         ("aproximado", hist[hist.preciso == False])]:  # noqa: E712
        y = np.r_[np.ones(len(sub)), np.zeros(len(bg))]
        for nome, arr in [("A0_cel", np.where(np.isfinite(pilha).all(0), pilha[0], np.nan)), ("A1_cel", s_a1),
                          ("A0_montante7", up_a0), ("A1_montante7", up_a1)]:
            v = np.r_[arr[sub.l, sub.c], arr[bg.l, bg.c]]
            ok = np.isfinite(v)
            linhas.append({"estrato": estrato, "sitios": int(len(sub)), "escore": nome,
                           "AUC_T_hist": float(roc_auc_score(y[ok], v[ok]))})
    return pd.DataFrame(linhas)


def main() -> int:
    e = Etapa("pa_05b_emenda1")
    t_ini = time.time()
    for p in (RES, PESOS):
        p.mkdir(parents=True, exist_ok=True)
    pad = ler_json(OUT / "pesos/padronizacao.json")
    mu, sd = pd.Series(pad["media"]), pd.Series(pad["desvio"])
    Zdf = lambda df: ((df[F] - mu[F]) / sd[F]).values  # noqa: E731
    Zarr = lambda a: (a - mu[F].values) / sd[F].values  # noqa: E731
    a1 = ler_json(OUT / "pesos/A1_log.json")
    beta_a1 = np.asarray(a1["coef"])
    with open(OUT / "pesos/A1_hgb.pkl", "rb") as f:
        a1h = pickle.load(f)

    # ------------------------------------------------ D5
    t0 = time.time()
    d5df = d5(Zarr, a1)
    d5df.to_csv(RES / "D5_receptor_T_hist.csv", index=False)
    t_d5 = time.time() - t0

    # ------------------------------------------------ E1 dobras
    import geopandas as gpd
    teste = pd.read_parquet(REF / "teste_petropolis_2022.parquet")
    ch = gpd.read_file(CHARTER).to_crs(EPSG_PETROPOLIS).reset_index(drop=True)
    ch["id_feicao"] = np.arange(len(ch))
    nf = ch[~ch.tipoevento.eq("Inundação")]
    cent = np.c_[nf.geometry.centroid.x, nf.geometry.centroid.y]
    _, ii = cKDTree(cent).query(np.c_[teste.x, teste.yc])
    dobra = nf.bairro.values[ii]
    bairro_cic = dict(zip(ch.id_feicao, ch.bairro))
    pos = teste.y.values == 1
    dobra[pos] = [bairro_cic[i] for i in teste.id_cicatriz.values[pos]]
    teste["dobra"] = dobra
    dobras = sorted(np.unique(dobra))
    Xt = Zdf(teste)
    yr_df = pd.read_parquet(REF / "amostra_idai.parquet")
    res_idai = yr_df[yr_df.papel.eq("reserva_origem")]
    Xr, yr = Zdf(res_idai), res_idai.y.values

    oof = {"A0_regra": teste.decl_m3.values.astype(float), "A1_log": escore_log(a1, Xt), "A1_hgb": a1h.decision_function(Xt)}
    ajustes, ret = [], []
    orc = EM["E1_cicatrizes_cv_bairro"]["orcamentos_K"]
    sementes = range(EM["E1_cicatrizes_cv_bairro"]["sementes"])
    bracos = ["A2_log_l1", "A2_log_l25", "A3_log_l1", "A3_log_l25", "A3perm_log_l25", "A2_hgb", "A3_hgb"]
    for K in orc:
        for s in (sementes if K != "todas" else [0]):
            for nome in bracos:
                oof[f"{nome}__K{K}__s{s}"] = np.full(len(teste), np.nan)
    for k in dobras:
        em_k = teste.dobra.values == k
        arv = cKDTree(np.c_[teste.x[em_k], teste.yc[em_k]])
        dist, _ = arv.query(np.c_[teste.x, teste.yc])
        treino_bg = (~em_k) & (~pos) & (dist > RAIO_BUFFER)
        treino_pos = (~em_k) & pos
        cic_treino = np.unique(teste.id_cicatriz.values[treino_pos])
        for K in orc:
            for s in (sementes if K != "todas" else [0]):
                rng = np.random.default_rng(SEMENTE + 31 * dobras.index(k) + 1000 * (999 if K == "todas" else K) + s)
                escolha = cic_treino if K == "todas" else np.sort(rng.choice(cic_treino, min(K, len(cic_treino)), replace=False))
                mpos = treino_pos & np.isin(teste.id_cicatriz.values, escolha)
                ncel = pd.Series(teste.id_cicatriz.values[mpos]).value_counts()
                wpos = 1.0 / ncel.loc[teste.id_cicatriz.values[mpos]].values
                nk = len(escolha)
                wbg = np.full(treino_bg.sum(), nk / treino_bg.sum())
                X = np.vstack([Xt[mpos], Xt[treino_bg]])
                y = np.r_[np.ones(mpos.sum()), np.zeros(treino_bg.sum())]
                w = np.r_[wpos, wbg]
                perm = rng.permutation(len(F))
                while np.all(perm == np.arange(len(F))):
                    perm = rng.permutation(len(F))
                base = {"dobra_retida": k, "K": K, "semente": s, "cicatrizes_ajuste": nk,
                        "celulas_pos": int(mpos.sum()), "celulas_fundo": int(treino_bg.sum()),
                        "ids_cicatrizes": ",".join(map(str, escolha))}
                for nome, (lam, b0) in {"A2_log_l1": (1.0, None), "A2_log_l25": (25.0, None),
                                        "A3_log_l1": (1.0, beta_a1), "A3_log_l25": (25.0, beta_a1),
                                        "A3perm_log_l25": (25.0, beta_a1[perm])}.items():
                    m = ajustar_log_w(X, y, w, lam, b0)
                    fid = f"{nome}__K{K}__s{s}"
                    oof[fid][em_k] = escore_log(m, Xt[em_k])
                    vr = escore_log(m, Xr)
                    ajustes.append({**base, "fit_id": f"{fid}__dobra_{k}", "braco": nome, "lambda": lam,
                                    "convergiu": m["convergiu"], "intercepto": m["intercepto"],
                                    **{f"coef_{c}": v for c, v in zip(F, m["coef"])},
                                    "AUC_idai_reserva": float(roc_auc_score(yr, vr))})
                for nome in ("A2_hgb", "A3_hgb"):
                    if nome == "A2_hgb":
                        m = HistGradientBoostingClassifier(max_iter=50, early_stopping=False, random_state=SEMENTE + s, **LOCAL_HGB)
                    else:
                        m = copy.deepcopy(a1h)
                        m.set_params(warm_start=True, max_iter=a1h.n_iter_ + 50, **LOCAL_HGB)
                    m.fit(X, y, sample_weight=w)
                    fid = f"{nome}__K{K}__s{s}"
                    arq = PESOS / f"{fid}__dobra_{k.replace(' ', '_').replace('.', '')}.pkl"
                    with open(arq, "wb") as f:
                        pickle.dump(m, f)
                    oof[fid][em_k] = m.decision_function(Xt[em_k])
                    ajustes.append({**base, "fit_id": f"{fid}__dobra_{k}", "braco": nome, "convergiu": True,
                                    "pkl": rel(arq), "pkl_sha256": sha256(arq),
                                    "AUC_idai_reserva": float(roc_auc_score(yr, m.decision_function(Xr)))})
        print(f"[E1] dobra {k} pronta ({time.time() - t_ini:.0f}s)", flush=True)

    # modelo de implantacao (todas as 81 cicatrizes, todo o fundo) para a resposta territorial
    ncel = pd.Series(teste.id_cicatriz.values[pos]).value_counts()
    wpos = 1.0 / ncel.loc[teste.id_cicatriz.values[pos]].values
    nbg = (~pos).sum()
    impl = ajustar_log_w(np.vstack([Xt[pos], Xt[~pos]]), np.r_[np.ones(pos.sum()), np.zeros(nbg)],
                         np.r_[wpos, np.full(nbg, len(ncel) / nbg)], 25.0, beta_a1)
    impl.update({"fit_id": "E1_implantacao_A3_log_l25_81cicatrizes", "atributos": F,
                 "avaliacao": "sem teste proprio: sua estimativa fora da amostra e a E1 K=todas (CV por bairro)",
                 "AUC_idai_reserva": float(roc_auc_score(yr, escore_log(impl, Xr)))})
    gravar_json(PESOS / "E1_implantacao_A3_log_l25.json", impl)

    aj = pd.DataFrame(ajustes)
    aj.to_csv(RES / "ajustes_E1.csv", index=False)
    pred = teste[["id", "l", "c", "y", "id_cicatriz", "bloco500", "dobra"]].copy()
    for k_, v in oof.items():
        pred[k_] = v
    pred.to_parquet(RES / "pred_oof_E1.parquet", index=False)

    # ------------------------------------------------ metricas dentro da dobra
    cic_ids = np.unique(teste.id_cicatriz.values[pos])
    cic_idx_all = np.searchsorted(cic_ids, teste.id_cicatriz.values[pos])
    blocos, inv_bloco = np.unique(teste.bloco500.values, return_inverse=True)
    dob_pos = teste.dobra.values[pos]

    def por_cicatriz(v, wb=None):
        pct = np.empty(pos.sum())
        for k in dobras:
            mb = (~pos) & (teste.dobra.values == k)
            mp = dob_pos == k
            vb = v[mb]
            ordem = np.argsort(vb, kind="mergesort")
            sb = vb[ordem]
            ww = np.ones(len(sb)) if wb is None else wb[inv_bloco[mb]][ordem]
            cw = np.cumsum(ww)
            vp = v[pos][mp]
            esq, dir_ = np.searchsorted(sb, vp, "left"), np.searchsorted(sb, vp, "right")
            ab = np.where(esq > 0, cw[np.maximum(esq - 1, 0)], 0.0)
            at = np.where(dir_ > 0, cw[np.maximum(dir_ - 1, 0)], 0.0)
            pct[mp] = (ab + 0.5 * (at - ab)) / cw[-1]
        return np.bincount(cic_idx_all, pct, len(cic_ids)) / np.bincount(cic_idx_all, minlength=len(cic_ids))

    met = []
    for k_, v in oof.items():
        pc = por_cicatriz(v)
        met.append({"vetor": k_, "AUC_obj_oof": float(pc.mean()), "rec_cic_20": float(np.mean(pc >= 0.8)),
                    "AUC_cel_pooled_nao_comparavel": float(roc_auc_score(teste.y, v))})
    met = pd.DataFrame(met)
    met.to_csv(RES / "metricas_E1.csv", index=False)

    grupos = {"A0_regra": ["A0_regra"], "A1_log": ["A1_log"], "A1_hgb": ["A1_hgb"]}
    for K in orc:
        for nome in bracos:
            grupos[f"{nome}__K{K}"] = [k_ for k_ in oof if k_.startswith(f"{nome}__K{K}__")]
    rng = np.random.default_rng(SEMENTE + 99)
    boot = {g: np.zeros((N_BOOT, len(v))) for g, v in grupos.items()}
    t0 = time.time()
    for r in range(N_BOOT):
        wc = np.bincount(rng.integers(0, len(cic_ids), len(cic_ids)), minlength=len(cic_ids)).astype(float)
        wb = np.bincount(rng.integers(0, len(blocos), len(blocos)), minlength=len(blocos)).astype(float) + 1e-12
        for g, nomes in grupos.items():
            for j, n_ in enumerate(nomes):
                pc = por_cicatriz(oof[n_], wb)
                boot[g][r, j] = np.sum(wc * pc) / wc.sum()
    t_boot = time.time() - t0
    pont = {g: float(np.mean([met.set_index("vetor").loc[n_, "AUC_obj_oof"] for n_ in nomes])) for g, nomes in grupos.items()}
    contr = []
    for K in orc:
        for nome, a, b in [("E1_C2", f"A3_log_l25__K{K}", "A1_log"), ("E1_C3", f"A3_log_l25__K{K}", f"A2_log_l25__K{K}"),
                           ("E1_C4", f"A3_log_l25__K{K}", f"A3perm_log_l25__K{K}"),
                           ("E1_local_vs_regra", f"A2_log_l25__K{K}", "A0_regra"),
                           ("E1_C2_l1", f"A3_log_l1__K{K}", "A1_log"), ("E1_C3_l1", f"A3_log_l1__K{K}", f"A2_log_l1__K{K}"),
                           ("E1_hgb_local", f"A3_hgb__K{K}", "A1_hgb"), ("E1_hgb_externo", f"A3_hgb__K{K}", f"A2_hgb__K{K}"),
                           ("E1_A3_vs_regra", f"A3_log_l25__K{K}", "A0_regra")]:
            d = boot[a].mean(1) - boot[b].mean(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            delta = pont[a] - pont[b]
            contr.append({"K": K, "contraste": nome, "a": a, "b": b, "AUC_obj_oof_a": pont[a], "AUC_obj_oof_b": pont[b],
                          "delta": delta, "ic95_lo": lo, "ic95_hi": hi,
                          "primario": K == "todas" and nome in ("E1_C2", "E1_C3", "E1_C4", "E1_local_vs_regra"),
                          "rotulo_regra": rotulo(delta, lo, hi)})
    zs = [{"K": 0, "contraste": "E1_transferencia_dentro_dobra", "a": "A1_log", "b": "A0_regra",
           "AUC_obj_oof_a": pont["A1_log"], "AUC_obj_oof_b": pont["A0_regra"], "delta": pont["A1_log"] - pont["A0_regra"],
           "ic95_lo": float(np.percentile(boot["A1_log"][:, 0] - boot["A0_regra"][:, 0], 2.5)),
           "ic95_hi": float(np.percentile(boot["A1_log"][:, 0] - boot["A0_regra"][:, 0], 97.5)), "primario": False}]
    zs[0]["rotulo_regra"] = rotulo(zs[0]["delta"], zs[0]["ic95_lo"], zs[0]["ic95_hi"])
    cdf = pd.DataFrame(zs + contr)
    cdf.to_csv(RES / "contrastes_E1.csv", index=False)

    r0 = float(roc_auc_score(yr, escore_log(a1, Xr)))
    for K in orc:
        for nome in bracos:
            v = aj[(aj.braco == nome) & (aj.K.astype(str) == str(K))].AUC_idai_reserva
            ret.append({"K": K, "braco": nome, "AUC_reserva_media": float(v.mean()), "min": float(v.min()),
                        "max": float(v.max()), "AUC_A1_log": r0, "delta_vs_A1_log": float(v.mean() - r0)})
    pd.DataFrame(ret).to_csv(RES / "retencao_E1.csv", index=False)

    # ------------------------------------------------ por comunidade (zona de teste)
    import rasterio
    fcu = gpd.read_file(ESTUDO / "fcu_42_geometrias.geojson").to_crs(EPSG_PETROPOLIS)
    fcu["k"] = np.arange(1, len(fcu) + 1)
    with rasterio.open(TERRENO / "petropolis/atributos_encosta.tif") as r:
        fr = _raster(r, list(fcu.geometry), valores=list(fcu.k), all_touched=True)
    teste["k_fcu"] = fr[teste.l, teste.c]
    pv1 = pd.read_parquet(OUT / "resultados/pred_teste_2022.parquet")
    v1_a3 = pv1[[c for c in pv1.columns if c.startswith("A3_log_l25__B100__")]].mean(1).values
    vet_com = {"A0_regra": (oof["A0_regra"], False), "A1_log": (oof["A1_log"], False),
               "v1_A3_hist_B100": (v1_a3, False),
               "E1_A3_log_l25_oof": (oof["A3_log_l25__Ktodas__s0"], True),
               "E1_A2_log_l25_oof": (oof["A2_log_l25__Ktodas__s0"], True)}
    pcs = {}
    for nome, (v, dentro_dobra) in vet_com.items():
        if dentro_dobra:
            pcs[nome] = por_cicatriz(v)
        else:
            bgv = np.sort(v[~pos])
            pp = (np.searchsorted(bgv, v[pos], "left") + np.searchsorted(bgv, v[pos], "right")) / 2 / len(bgv)
            pcs[nome] = np.bincount(cic_idx_all, pp, len(cic_ids)) / np.bincount(cic_idx_all, minlength=len(cic_ids))
    linhas = []
    for _, f in fcu.iterrows():
        cel = (teste.k_fcu.values == f.k)
        if not cel.any():
            continue
        cic_f = np.unique(teste.id_cicatriz.values[cel & pos])
        lin = {"CD_FCU": f.CD_FCU, "NM_FCU": f.NM_FCU, "celulas_no_suporte_teste": int(cel.sum()),
               "cicatrizes_2022_tocando_fcu": int(len(cic_f)),
               "dobra": ",".join(sorted(set(teste.dobra.values[cel])))}
        for nome in vet_com:
            pc = pcs[nome][np.searchsorted(cic_ids, cic_f)] if len(cic_f) else np.array([])
            lin[f"{nome}_pct_medio_cicatrizes"] = float(pc.mean()) if len(pc) else None
            lin[f"{nome}_cicatrizes_recuperadas_q20"] = int((pc >= 0.8).sum()) if len(pc) else 0
        linhas.append(lin)
    pd.DataFrame(linhas).to_csv(RES / "por_comunidade_zona_teste.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"id_cicatriz": cic_ids, "bairro": [bairro_cic[i] for i in cic_ids], **pcs}).to_csv(
        RES / "percentil_por_cicatriz.csv", index=False, encoding="utf-8-sig")

    resumo = {"emenda": EM["id"], "dobras": {k: int(((teste.dobra == k) & pos).sum()) for k in dobras},
              "cicatrizes_por_dobra": {k: int(len(np.unique(teste.id_cicatriz.values[pos & (teste.dobra.values == k)]))) for k in dobras},
              "contrastes_primarios": cdf[cdf.primario].to_dict("records"),
              "transferencia_dentro_dobra": zs[0], "D5": d5df.to_dict("records"),
              "implantacao_AUC_idai_reserva": impl["AUC_idai_reserva"],
              "custo_s": {"D5": round(t_d5, 1), "bootstrap": round(t_boot, 1), "total": round(time.time() - t_ini, 1)},
              "nao_convergiu": int((aj.convergiu == False).sum())}  # noqa: E712
    gravar_json(RES / "resultado_E1.json", resumo)
    e.saida(*RES.glob("*"), PESOS / "E1_implantacao_A3_log_l25.json")
    e.fechar(custo=resumo["custo_s"])
    pd.set_option("display.width", 250)
    print(d5df.round(4).to_string())
    print(cdf.round(4).to_string())
    print(pd.DataFrame(ret).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
