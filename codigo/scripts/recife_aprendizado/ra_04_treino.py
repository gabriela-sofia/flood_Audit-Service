"""Treino e avaliacao de todos os bracos do protocolo v1 de Recife (ruas com relato de agua).

Recusa rodar sem o protocolo ou com referencias/atributos diferentes dos congelados.
Grava coeficientes antes/depois, IDs consultados em ordem, previsoes por rua, metricas,
contrastes com bootstrap por bairro e retencao nas 6 unidades limpas da v10.
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from ra_comum import OUT, ROOT, SEMENTE, Etapa, gravar_json, ler_json, rel, sha256

sys.path[:0] = [str(ROOT / "scripts/vulnerabilizados"), str(ROOT / "scripts/metodo")]
PROT = ler_json(OUT / "protocolo/protocolo_v1.json")
RES = OUT / "resultados"
PESOS = OUT / "pesos"
QF = PROT["blocos"]["Q30"]
DF = PROT["blocos"]["DREN"]
SF = PROT["blocos"]["SAIDA"]
PF = PROT["blocos"]["PE3D"]
ORC = [10, 25, "todas"]
SEM = list(range(10))
N_BOOT = 2000
HGB = {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 10, "early_stopping": False}


def ajustar(X, y, lam, beta0):
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta0 = np.asarray(beta0, float)
    w = np.where(y == 1, 1.0, y.sum() / (1 - y).sum())

    def f(t):
        z = t[0] + X @ t[1:]
        d = t[1:] - beta0
        s = 1 / (1 + np.exp(-z))
        return (np.sum(w * (np.logaddexp(0, z) - y * z)) + 0.5 * lam * d @ d,
                np.concatenate([[np.sum(w * (s - y))], X.T @ (w * (s - y)) + lam * d]))
    r = minimize(f, np.concatenate([[0.0], beta0]), jac=True, method="L-BFGS-B", options={"maxiter": 3000, "gtol": 1e-8})
    return {"intercepto": float(r.x[0]), "coef": r.x[1:].tolist(), "convergiu": bool(r.success), "iteracoes": int(r.nit)}


def escore(m, X):
    return m["intercepto"] + np.asarray(X, float) @ np.asarray(m["coef"])


def perda(m, X, y):
    z = escore(m, X)
    w = np.where(y == 1, 1.0, y.sum() / (1 - y).sum())
    return float(np.sum(w * (np.logaddexp(0, z) - y * z)) / w.sum())


def pct_bairro(v, y, bairro):
    """Percentil de cada positiva contra o fundo do proprio bairro (empates 1/2)."""
    out = np.full(len(v), np.nan)
    for b in np.unique(bairro[y == 1]):
        mb = bairro == b
        fundo = np.sort(v[mb & (y == 0)])
        idx = np.nonzero(mb & (y == 1))[0]
        if len(fundo) == 0:
            continue
        esq = np.searchsorted(fundo, v[idx], "left")
        dir_ = np.searchsorted(fundo, v[idx], "right")
        out[idx] = (esq + 0.5 * (dir_ - esq)) / len(fundo)
    return out


def metricas(v, y, bairro):
    p = pct_bairro(v, y, bairro)
    pp = p[y == 1]
    pp = pp[np.isfinite(pp)]
    return {"AUC_bairro": float(pp.mean()), "rec_q20_bairro": float(np.mean(pp >= 0.8)),
            "AUC_agrupada": float(roc_auc_score(y, v)), "AP": float(average_precision_score(y, v)),
            "prevalencia": float(y.mean()), "positivas": int(y.sum())}, p


def rotulo(m, lo, hi, margem=PROT["regra"]["margem"]):
    if m >= margem and lo > 0:
        return "sustentado"
    if m <= -margem and hi < 0:
        return "contrario"
    return "inconclusivo"


def boot_auc(pcts: dict, y, bairro, n=N_BOOT, semente=SEMENTE + 11):
    """pcts: nome -> matriz (sementes, n) de percentis. Reamostra bairros do teste com reposicao."""
    rng = np.random.default_rng(semente)
    bs = np.unique(bairro)
    mpos = y == 1
    soma = {k: np.stack([np.bincount(np.searchsorted(bs, bairro[mpos]), np.nan_to_num(m[s][mpos]), len(bs)) for s in range(m.shape[0])])
            for k, m in pcts.items()}
    npos = np.bincount(np.searchsorted(bs, bairro[mpos]), minlength=len(bs)).astype(float)
    out = {k: np.zeros(n) for k in pcts}
    for r in range(n):
        c = np.bincount(rng.integers(0, len(bs), len(bs)), minlength=len(bs)).astype(float)
        den = (c * npos).sum()
        for k in pcts:
            out[k][r] = ((soma[k] * c).sum(1) / den).mean()
    return out


def main() -> int:
    e = Etapa("ra_04_treino")
    t_ini = time.time()
    for p in (RES, PESOS, PESOS / "hgb"):
        p.mkdir(parents=True, exist_ok=True)
    if sha256(OUT / "atributos/atributos_objetos.parquet") != PROT["atributos_sha256"]:
        raise SystemExit("atributos diferentes dos congelados")
    t = pd.read_parquet(OUT / "atributos/atributos_objetos.parquet")
    nq = ler_json(OUT / "atributos/nucleo_q_congelado.json")
    beta_q = np.asarray(nq["b"])
    assert nq["principais"] == QF
    zq = ((t[QF].fillna(pd.Series(nq["med"])[QF]) - pd.Series(nq["mu"])[QF]) / pd.Series(nq["sd"])[QF]).fillna(0).values
    aj = t.papel.values == "atualizacao"
    te = ~aj
    t["sem_canal_500m"] = t.pe3d_desnivel_canal.isna() & t.tem_pe3d
    U = DF + SF
    med_u = t.loc[aj, U + PF].median()
    xu_raw = t[U + PF].fillna(med_u)
    mu_u, sd_u = xu_raw[aj].mean(), xu_raw[aj].std(ddof=0).replace(0, 1)
    zu = ((xu_raw - mu_u) / sd_u).values
    iD = [ (U + PF).index(c) for c in DF]
    iS = [(U + PF).index(c) for c in SF]
    iP = [(U + PF).index(c) for c in PF]
    gravar_json(PESOS / "padronizacao.json", {"Q30": {"mu": nq["mu"], "sd": nq["sd"], "med": nq["med"], "origem": "nucleo Q"},
                                              "urbanos": {"mu": mu_u.to_dict(), "sd": sd_u.to_dict(), "med": med_u.to_dict(),
                                                          "origem": "ruas de atualizacao"}})
    y = t.y.values.astype(int)
    bairro = t.CBAIRRCODI.astype(int).values
    pe = t.tem_pe3d.values.astype(bool)
    yt, bt = y[te], bairro[te]

    pred = {"id_obj": t.id_obj.values, "papel": t.papel.values, "y": y, "CBAIRRCODI": bairro, "tem_pe3d": pe}
    met, ajustes = [], []

    def avaliar(nome, v, B, s, mask=te, sub="cidade", **meta):
        m, p = metricas(v[mask], y[mask], bairro[mask])
        met.append({"braco": nome, "B": B, "semente": s, "subconjunto": sub, **m, **meta})
        return p

    pred["A0_regra"] = -t.f_hand.values
    pred["A1_Q"] = t.q_escore.values
    pred["A0_pe3d"] = np.where(pe, -t.pe3d_rel50.values, np.nan)
    pcts = {}
    for nome in ("A0_regra", "A1_Q"):
        pcts[nome] = avaliar(nome, pred[nome], 0, -1)[None, :]
        pcts[nome + "|pe3d"] = avaliar(nome, pred[nome], 0, -1, mask=te & pe, sub="pe3d")[None, :]
    pcts["A0_pe3d|pe3d"] = avaliar("A0_pe3d", pred["A0_pe3d"], 0, -1, mask=te & pe, sub="pe3d")[None, :]

    pos_aj = np.nonzero(aj & (y == 1))[0]
    hgb_feats = {"A2_hgb": np.c_[zq, zu[:, iD + iS]], "A4_hgb": np.c_[t.q_escore.values, zu[:, iD + iS]]}
    t_loc = time.time()
    for B in ORC:
        for s in (SEM if B != "todas" else [0]):
            rng = np.random.default_rng(SEMENTE + 1000 * (999 if B == "todas" else B) + s)
            esc = pos_aj if B == "todas" else rng.choice(pos_aj, B, replace=False)
            bairros_esc = np.unique(bairro[esc])
            idx = np.r_[esc, np.nonzero(aj & (y == 0) & np.isin(bairro, bairros_esc))[0]]
            yy = y[idx]
            perm = rng.permutation(len(QF))
            while np.all(perm == np.arange(len(QF))):
                perm = rng.permutation(len(QF))
            base = {"B": B, "semente": s, "consultas_em_ordem": ",".join(t.id_obj.values[esc]),
                    "A1_antes_das_consultas": ",".join(f"{v:.4f}" for v in t.q_escore.values[esc]),
                    "positivas": int(len(esc)), "fundo": int((yy == 0).sum()), "bairros": int(len(bairros_esc))}
            z0 = np.zeros
            specs = {
                "A2_q_l1": (zq, 1.0, z0(len(QF)), QF), "A2_q_l25": (zq, 25.0, z0(len(QF)), QF),
                "A3_q_l1": (zq, 1.0, beta_q, QF), "A3_q_l25": (zq, 25.0, beta_q, QF),
                "A3perm_l25": (zq, 25.0, beta_q[perm], QF),
                "A2u_l25": (np.c_[zq, zu[:, iD + iS]], 25.0, z0(len(QF) + len(U)), QF + U),
                "A4_l25": (np.c_[zq, zu[:, iD + iS]], 25.0, np.r_[beta_q, z0(len(U))], QF + U),
                "A4_sem_DREN": (np.c_[zq, zu[:, iS]], 25.0, np.r_[beta_q, z0(len(SF))], QF + SF),
                "A4_sem_SAIDA": (np.c_[zq, zu[:, iD]], 25.0, np.r_[beta_q, z0(len(DF))], QF + DF),
            }
            for nome, (X, lam, b0, nomes) in specs.items():
                t0 = time.time()
                m = ajustar(X[idx], yy, lam, b0)
                fid = f"{nome}__B{B}__s{s}"
                v = escore(m, X)
                pred[fid] = v
                antes = {"intercepto": float(ajustar(X[idx], yy, 1e9, b0)["intercepto"]), "coef": list(b0)}
                ajustes.append({**base, "fit_id": fid, "braco": nome, "lambda": lam, "atributos": "|".join(nomes),
                                "permutacao": perm.tolist() if nome.startswith("A3perm") else None,
                                "convergiu": m["convergiu"], "iteracoes": m["iteracoes"], "intercepto": m["intercepto"],
                                **{f"coef_{c}": v_ for c, v_ in zip(nomes, m["coef"])},
                                **{f"antes_{c}": v_ for c, v_ in zip(nomes, b0)},
                                "perda_antes": perda(antes, X[idx], yy), "perda_depois": perda(m, X[idx], yy),
                                "auc_treino_antes": float(roc_auc_score(yy, escore(antes, X[idx]))) if np.any(b0) else None,
                                "auc_treino_depois": float(roc_auc_score(yy, escore(m, X[idx]))),
                                "parede_s": round(time.time() - t0, 3)})
                pcts.setdefault(f"{nome}__B{B}", []).append(avaliar(nome, v, B, s, fit_id=fid))
            for nome, X in hgb_feats.items():
                m = HistGradientBoostingClassifier(random_state=SEMENTE + s, **HGB)
                m.fit(X[idx], yy, sample_weight=np.where(yy == 1, 1.0, yy.sum() / (1 - yy).sum()))
                fid = f"{nome}__B{B}__s{s}"
                arq = PESOS / "hgb" / f"{fid}.pkl"
                with open(arq, "wb") as f:
                    pickle.dump(m, f)
                v = m.decision_function(X)
                pred[fid] = v
                ajustes.append({**base, "fit_id": fid, "braco": nome, "pkl": rel(arq), "pkl_sha256": sha256(arq), "convergiu": True,
                                "auc_treino_depois": float(roc_auc_score(yy, m.decision_function(X[idx])))})
                pcts.setdefault(f"{nome}__B{B}", []).append(avaliar(nome, v, B, s, fit_id=fid))
        print(f"[local] B={B} ({time.time() - t_loc:.0f}s)", flush=True)

    # subconjunto PE3D (B=todas): ajuste so com ruas de atualizacao na cobertura
    idx = np.nonzero(aj & pe)[0]
    yy = y[idx]
    Xa = np.c_[zq, zu[:, iD + iS]]
    Xp = np.c_[zq, zu[:, iD + iS + iP]]
    specs_p = {"A4p_l25": (Xa, np.r_[beta_q, np.zeros(len(U))], QF + U),
               "A5_l25": (Xp, np.r_[beta_q, np.zeros(len(U) + len(PF))], QF + U + PF),
               "A2p_l25": (Xp, np.zeros(len(QF) + len(U) + len(PF)), QF + U + PF),
               "A5_so_PE3D_l25": (np.c_[zq, zu[:, iP]], np.r_[beta_q, np.zeros(len(PF))], QF + PF)}
    for nome, (X, b0, nomes) in specs_p.items():
        m = ajustar(X[idx], yy, 25.0, b0)
        fid = f"{nome}__Btodas__s0"
        v = np.where(pe, escore(m, X), np.nan)
        pred[fid] = v
        ajustes.append({"B": "todas_pe3d", "semente": 0, "fit_id": fid, "braco": nome, "lambda": 25.0, "atributos": "|".join(nomes),
                        "positivas": int(yy.sum()), "fundo": int((yy == 0).sum()), "convergiu": m["convergiu"],
                        "intercepto": m["intercepto"], **{f"coef_{c}": v_ for c, v_ in zip(nomes, m["coef"])},
                        "consultas_em_ordem": ",".join(t.id_obj.values[idx[yy == 1]])})
        pcts[f"{nome}|pe3d"] = avaliar(nome, v, "todas_pe3d", 0, mask=te & pe, sub="pe3d", fit_id=fid)[None, :]
    for nome in ("A3_q_l25", "A4_l25"):
        pcts[f"{nome}__Btodas|pe3d"] = avaliar(nome, pred[f"{nome}__Btodas__s0"], "todas", 0, mask=te & pe, sub="pe3d",
                                               fit_id=f"{nome}__Btodas__s0")[None, :]

    # ------------------------------------------------ gravar
    pdf = pd.DataFrame(pred)
    pdf.to_parquet(RES / "pred_ruas.parquet", index=False)
    mdf = pd.DataFrame(met)
    mdf.to_csv(RES / "metricas_bracos.csv", index=False)
    adf = pd.DataFrame(ajustes)
    adf.to_csv(RES / "ajustes.csv", index=False)

    # ------------------------------------------------ contrastes
    pc = {k: (np.stack(v) if isinstance(v, list) else v) for k, v in pcts.items()}
    cid = {k: v for k, v in pc.items() if "|pe3d" not in k}
    sub = {k: v for k, v in pc.items() if "|pe3d" in k}
    bcid = boot_auc(cid, yt, bt)
    ypt, bpt = y[te & pe], bairro[te & pe]
    bsub = boot_auc(sub, ypt, bpt)
    pont = {**{k: float(np.nanmean(v[:, yt == 1])) for k, v in cid.items()},
            **{k: float(np.nanmean(v[:, ypt == 1])) for k, v in sub.items()}}
    bb = {**bcid, **bsub}
    pares = []
    for B in ORC:
        pares += [("C2_aprendizado_local", B, f"A3_q_l25__B{B}", "A1_Q"),
                  ("C3_valor_conhecimento_externo", B, f"A3_q_l25__B{B}", f"A2_q_l25__B{B}"),
                  ("C4_controle", B, f"A3_q_l25__B{B}", f"A3perm_l25__B{B}"),
                  ("C5_contexto_urbano", B, f"A4_l25__B{B}", f"A3_q_l25__B{B}"),
                  ("C5_controle_local", B, f"A2u_l25__B{B}", f"A2_q_l25__B{B}"),
                  ("ablacao_sem_DREN", B, f"A4_l25__B{B}", f"A4_sem_DREN__B{B}"),
                  ("ablacao_sem_SAIDA", B, f"A4_l25__B{B}", f"A4_sem_SAIDA__B{B}"),
                  ("C3_lambda1", B, f"A3_q_l1__B{B}", f"A2_q_l1__B{B}"),
                  ("A4_vs_regra", B, f"A4_l25__B{B}", "A0_regra"),
                  ("hgb_vs_A4", B, f"A4_hgb__B{B}", f"A4_l25__B{B}"),
                  ("hgb_externo", B, f"A4_hgb__B{B}", f"A2_hgb__B{B}")]
    pares = [("C1_transferencia", 0, "A1_Q", "A0_regra")] + pares + [
        ("C6_PE3D", "todas_pe3d", "A5_l25|pe3d", "A4p_l25|pe3d"),
        ("C6_so_PE3D_sobre_Q", "todas_pe3d", "A5_so_PE3D_l25|pe3d", "A3_q_l25__Btodas|pe3d"),
        ("PE3D_regra_vs_HAND", 0, "A0_pe3d|pe3d", "A0_regra|pe3d"),
        ("PE3D_regra_vs_Q", 0, "A0_pe3d|pe3d", "A1_Q|pe3d"),
        ("A2p_vs_A5", "todas_pe3d", "A5_l25|pe3d", "A2p_l25|pe3d")]
    primarios = {"C1_transferencia", "C2_aprendizado_local", "C3_valor_conhecimento_externo", "C4_controle",
                 "C5_contexto_urbano", "C5_controle_local", "C6_PE3D"}
    cont = []
    for nome, B, a, b in pares:
        d = bb[a] - bb[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        delta = pont[a] - pont[b]
        cont.append({"contraste": nome, "B": B, "a": a, "b": b, "AUC_bairro_a": pont[a], "AUC_bairro_b": pont[b],
                     "delta": delta, "ic95_lo": lo, "ic95_hi": hi, "rotulo_regra": rotulo(delta, lo, hi),
                     "primario": nome in primarios and (B in ("todas", 0, "todas_pe3d"))})
    cdf = pd.DataFrame(cont)
    cdf.to_csv(RES / "contrastes.csv", index=False)

    # ------------------------------------------------ retencao na origem limpa
    t0 = time.time()
    ret = []
    try:
        from v8_04_campanha import auc_evento_unidade, externo
        tr, x, _ = externo()
        papel = tr.unidade.map(ler_json(ROOT / "local_runs/transferencia-vulnerabilizados/v8_validacao_integrada/grafo_particoes.json")["unidades"])
        r_ = papel.eq("retencao").to_numpy()
        trr = tr[r_].reset_index(drop=True)
        zx = ((x.loc[r_, QF].fillna(pd.Series(nq["med"])[QF]) - pd.Series(nq["mu"])[QF]) / pd.Series(nq["sd"])[QF]).fillna(0).values
        base_q = auc_evento_unidade(trr, zx @ beta_q)
        ret.append({"modelo": "Q_congelado_efeitos_principais", "B": 0, "AUC_retencao": base_q, "unidades": int(trr.unidade.nunique())})
        for _, a in adf[adf.braco.isin(["A3_q_l25", "A3_q_l1", "A3perm_l25", "A4_l25", "A2_q_l25"])].iterrows():
            b = np.asarray([a[f"coef_{c}"] for c in QF], float)
            ret.append({"modelo": a.braco, "B": a.B, "semente": a.semente, "AUC_retencao": auc_evento_unidade(trr, zx @ b),
                        "parcela": "so coeficientes Q30" if a.braco == "A4_l25" else "modelo inteiro"})
        rdf = pd.DataFrame(ret)
        rdf["delta_vs_Q"] = rdf.AUC_retencao - base_q
    except Exception as exc:                                         # noqa: BLE001
        rdf = pd.DataFrame([{"erro": repr(exc)}])
    rdf.to_csv(RES / "retencao_origem.csv", index=False)

    resultado = {"protocolo": PROT["id"], "contrastes_primarios": cdf[cdf.primario].to_dict("records"),
                 "n_ajustes": len(adf), "nao_convergiu": int((adf.convergiu == False).sum()),  # noqa: E712
                 "retencao_resumo": (rdf.groupby(["modelo", "B"]).AUC_retencao.mean().reset_index().to_dict("records")
                                     if "AUC_retencao" in rdf else rdf.to_dict("records")),
                 "consultas_em_teste": int(sum(1 for a in adf.consultas_em_ordem.dropna() for i in a.split(",")
                                               if i in set(t.id_obj.values[te]))),
                 "custo_s": {"ajustes_locais": round(time.time() - t_loc, 1), "retencao": round(time.time() - t0, 1),
                             "total": round(time.time() - t_ini, 1)}}
    gravar_json(RES / "resultado_treino.json", resultado)
    e.saida(*RES.glob("*"), PESOS / "padronizacao.json")
    e.fechar(custo=resultado["custo_s"], consultas_em_teste=resultado["consultas_em_teste"])
    pd.set_option("display.width", 250)
    print(cdf.round(4).to_string())
    print(rdf.groupby(["modelo", "B"]).AUC_retencao.mean().round(4).to_string() if "AUC_retencao" in rdf else rdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
