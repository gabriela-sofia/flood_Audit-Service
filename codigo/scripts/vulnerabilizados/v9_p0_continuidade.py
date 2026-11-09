"""v9 P0: adaptacao local que PRESERVA a funcao completa do Q (efeitos principais + interacoes + imputacao).

  base      q(x) = Q.escore(x)  (Q_sel_ext do p0_regularizacao: lam escolhido so no exterior; Q_hist para comparacao)
  correcao  s(x) = q(x) + c + theta . z(x), z = 7 atributos na escala externa; theta ~ N(0, 1/tau), c livre;
            pesos balanceados; correcao inicialmente nula
  paridade  B=0 e ajuste recusado -> s = q exatamente (verificado numericamente)
  braços    Delta_completa (7 atributos), Delta_baixa (hand, declive), Delta_gate (completa so se >= 5 positivos, >= 5
            negativos e >= 3 agrupamentos com positivo; regra fixada antes, motivada pelo resultado v8 de que atualizar com
            ~3 positivos piora), e o P_ext da v8 (so efeitos principais) como referencia da simplificacao.
  quantifica  efeito da simplificacao "7 efeitos principais" vs Q completo: Spearman, |dif| maxima e AUC.
Orcamentos v4 (3 politicas x B 25/100/400 x 6 sementes), coortes RMPA/Manaus/Curitiba, tau em {1, 25}.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_comum import COORTES, COORTES_B, V4, V9, agora, auc, congelar9, etapa, g9

D = V9 / "p0_continuidade"
TAUS = [1.0, 25.0]
GATE = {"positivos": 5, "negativos": 5, "grupos_com_positivo": 3}
BAIXA = ["f_hand", "f_declive"]


def ajustar_offset(z, y, q, tau):
    """min sum w*logloss(y, q + c + z@theta) + tau/2 |theta|^2 ; devolve (theta, c, aprendeu)."""
    y = np.asarray(y, float)
    k = z.shape[1]
    if len(np.unique(y)) < 2:
        return np.zeros(k), 0.0, False
    w = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))

    def f(t):
        s = q + z @ t[:-1] + t[-1]
        e = w * (expit(s) - y)
        return (np.sum(w * (np.logaddexp(0, s) - y * s)) + .5 * tau * np.sum(t[:-1] ** 2), np.r_[z.T @ e + tau * t[:-1], e.sum()])
    r = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B")
    return r.x[:-1], float(r.x[-1]), True


def main():
    from v8_04_campanha import FEAT, caracteristicas, coorte, z_externa
    import v5_aprendizagem as v5
    congelar9(D / "protocolo.json", {"versao": "v9-p0-cont-1", "taus": TAUS, "gate": GATE, "baixa_dimensao": BAIXA,
                                     "base": "Q_sel_ext (p0_regularizacao)", "paridade": "Delta=0 => s=q exatamente"})
    t0 = time.time()
    with (V9 / "p0_regularizacao/modelos.pkl").open("rb") as f:
        M = pickle.load(f)
    Q, QH = M["Q_sel_ext"], M["Q_hist"]
    mu, sd, med = Q.pad.mu[FEAT], Q.pad.sd[FEAT], Q.pad.med[FEAT]
    b_main = np.array([dict(zip(Q.principais, Q.b))[c] for c in FEAT])
    ids = pd.read_csv(V4 / "ids_orcamentos.csv")
    simpl, metr, ajus, paridade = [], [], [], []
    for c in COORTES:
        d = coorte(c)
        x = caracteristicas(d)
        q_all = Q.escore(x)
        z_all = z_externa(x, mu, sd, med)
        so_main = z_all @ b_main
        te = (d.particao.eq("teste") & d.estado_primaria.isin(["positivo", "negativo"])).to_numpy()
        y = d.estado_primaria.eq("positivo").astype(int).to_numpy()
        fcu = d.classe_territorial.eq("fcu").to_numpy()
        simpl.append({"coorte": c, "spearman_completo_vs_principais": float(pd.Series(q_all[te]).corr(pd.Series(so_main[te]), method="spearman")),
                      "max_abs_dif_padronizada": float(np.max(np.abs((q_all - q_all.mean()) - (so_main - so_main.mean()))[te])),
                      "auc_completo": auc(y[te], q_all[te]), "auc_principais": auc(y[te], so_main[te]),
                      "auc_completo_fcu": auc(y[te & fcu], q_all[te & fcu]), "auc_principais_fcu": auc(y[te & fcu], so_main[te & fcu])})
        if c not in COORTES_B:
            continue
        idx = dict(zip(d.ponto_id, d.index))
        acum = {}
        for (pol, B, sem), g in ids[ids.territorio.eq(c)].groupby(["politica", "B", "semente"]):
            ii = [idx[p] for p in g.ponto_id if p in idx and d.estado_primaria.iat[idx[p]] in ("positivo", "negativo")]
            ya, za, qa = y[ii], z_all[ii], q_all[ii]
            gp = d.grupo.to_numpy()[ii][ya == 1]
            suporte = int(ya.sum()) >= GATE["positivos"] and int((1 - ya).sum()) >= GATE["negativos"] and len(set(gp)) >= GATE["grupos_com_positivo"]
            for tau in TAUS:
                th, cc, ok = ajustar_offset(za, ya, qa, tau)
                s_full = q_all + z_all @ th + cc
                thb, cb, okb = ajustar_offset(za[:, [FEAT.index(f) for f in BAIXA]], ya, qa, tau)
                s_baixa = q_all + z_all[:, [FEAT.index(f) for f in BAIXA]] @ thb + cb
                s_gate = s_full if (ok and suporte) else q_all
                thp, okp = v5.ajustar(za, ya, b_main, tau, balancear=True)
                s_pext = v5.escore(thp, z_all)
                bracos = {f"Delta_completa|t{tau:g}": (s_full, ok), f"Delta_baixa|t{tau:g}": (s_baixa, okb),
                          f"Delta_gate|t{tau:g}": (s_gate, ok and suporte), f"P_ext_principais|t{tau:g}": (s_pext, okp)}
                if not ok:
                    paridade.append(float(np.max(np.abs(s_full - q_all))))
                for k, (s, aprendeu) in bracos.items():
                    ajus.append({"coorte": c, "politica": pol, "B": int(B), "semente": int(sem), "braco": k, "positivos": int(ya.sum()),
                                 "validos": len(ii), "consultados": len(g), "atualizou": bool(aprendeu), "suporte_gate": suporte})
                    for pop, m in [("todas", te), ("fcu", te & fcu)]:
                        metr.append({"coorte": c, "politica": pol, "B": int(B), "semente": int(sem), "braco": k, "populacao": pop, "auc": auc(y[m], s[m])})
                    acum.setdefault((pol, B, k), []).append(s[te].astype("float32"))
        for pop, m in [("todas", te), ("fcu", te & fcu)]:
            metr.append({"coorte": c, "politica": "-", "B": 0, "semente": 0, "braco": "Q_completo_B0", "populacao": pop, "auc": auc(y[m], q_all[m])})
            metr.append({"coorte": c, "politica": "-", "B": 0, "semente": 0, "braco": "Q_hist_B0", "populacao": pop, "auc": auc(y[m], QH.escore(x)[m])})
        pred = d.loc[te, ["ponto_id", "grupo", "classe_territorial"]].assign(y=y[te], Q_completo=q_all[te].astype("float32"))
        for (pol, B, k), ss in acum.items():
            pred[f"{k}|{pol}|B{B}"] = np.mean(ss, axis=0)
        g9(D / f"predicoes_{c}.parquet", pred.reset_index(drop=True))
        print(c, round(time.time() - t0), flush=True)
    m = pd.DataFrame(metr)
    g9(D / "metricas_semente.parquet", m)
    g9(D / "ajustes.parquet", pd.DataFrame(ajus))
    g9(D / "simplificacao_principais.csv", pd.DataFrame(simpl))
    curva = m.groupby(["coorte", "populacao", "politica", "B", "braco"]).auc.agg(["mean", "std", "count"]).reset_index()
    g9(D / "curvas.csv", curva)
    g9(D / "resultado.json", {"quando_utc": agora(), "segundos": time.time() - t0, "paridade_recusa_max_abs": max(paridade) if paridade else 0.0,
                              "recusas": len(paridade), "ajustes": len(ajus)})
    etapa("p0_continuidade", segundos=time.time() - t0, paridade=max(paridade) if paridade else 0.0)
    print(pd.DataFrame(simpl).round(4).to_string())


if __name__ == "__main__":
    main()
