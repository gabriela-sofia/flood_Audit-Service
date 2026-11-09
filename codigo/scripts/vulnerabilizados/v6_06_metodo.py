"""v6: evolucao do metodo em DESENVOLVIMENTO (RMPA ja examinada; Recife E2 ja aberto nesta v6 -> desenvolvimento).

Intervencoes (regras fixadas aqui, antes de rodar):
 M1 nao linear monotonico: HistGradientBoosting com restricoes de sinal fisico (HAND -, declive -, TWI +,
    construido livre, distancia ao canal -); max_depth 3, 200 iteracoes, taxa 0,05, folha minima 20 (capacidade
    pequena, sem busca). Bracos: GBM_ext (doadores internacionais), GBM_loc (RMPA uniforme B=400, 6 sementes),
    GBM_mix (doadores + mesmos 400 RMPA com peso 4). Comparados a A1/A2_estrito/A3 logisticos v5 (coeficientes salvos).
 M2 doadores por contexto: das 14 unidades internacionais de ajuste, as 7 mais proximas do descritor sem rotulo da
    adaptacao RMPA (p90 HAND, mediana do declive, fracao construida; distancia padronizada); contra todas e
    20 subconjuntos aleatorios de 7.
 M3 diversidade x quantidade (RETROSPECTIVO ASSISTIDO POR ROTULO, nao politica de coleta): orcamentos de 100 pontos
    FCU da adaptacao com exatamente 8 positivos e 92 negativos, espalhados por G comunidades (G = 8, 25, 50, 100,
    limitado pelo disponivel); 6 sementes; A3 e A2_forte.
 M4 limites com equidade: abstencao estratificada (mesma fracao de recusa em cada classe) pelo erro previsto v5-S4,
    contra ranking global e acaso, cobertura 70%, no teste RMPA e em Manaus.
Avaliacao: macro FCU nas 10 comunidades primarias da RMPA e agrupada FCU/todas; Recife E2 agrupada por classe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_aprendizagem as ap
import v5_aprendizagem as v5
from comum_tv import auc_mann_whitney, bloquear_rede
from v6_comum import V3A, V4, V5, V6, agora, congelar6, g6, sha_hex

D = V6 / "metodo"
COLS = ap.COM_CANAL
MONO = [-1, -1, 1, 0, -1]


def gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=.05, min_samples_leaf=20,
                                          monotonic_cst=MONO, random_state=0)


def avaliar(nome, s_rmpa, te, prim, s_rec=None, rec=None, extra=None):
    f = te.classe_territorial.eq("fcu").to_numpy()
    pg = v5.auc_por_grupo(te.y.to_numpy()[f], s_rmpa[f], te.grupo.to_numpy()[f], 5).reindex(prim)
    out = {"modelo": nome, "rmpa_fcu_macro10": float(pg.mean()), "rmpa_fcu_agrupada": auc_mann_whitney(te.y.to_numpy()[f], s_rmpa[f]),
           "rmpa_todas": auc_mann_whitney(te.y.to_numpy(), s_rmpa), **(extra or {})}
    if s_rec is not None:
        for cl in ["fcu", "baixa_renda", "comparador_urbano"]:
            m = rec.classe_territorial.eq(cl).to_numpy()
            out[f"recife_{cl}"] = auc_mann_whitney(rec.y.to_numpy()[m], s_rec[m])
        out["recife_todas"] = auc_mann_whitney(rec.y.to_numpy(), s_rec)
    return out, pg


def main():
    bloquear_rede()
    congelar6(D / "protocolo.json", {"versao": "v6-metodo-1", "regras": __doc__, "codigo_sha256": sha_hex(Path(__file__))})
    d = pd.read_parquet(V3A / "rmpa.parquet")
    d = d[d.dist_canal_m.notna() & d.construido.notna()].reset_index(drop=True)
    te = d[d.particao.eq("teste")].reset_index(drop=True)
    ad = d[d.particao.eq("adaptacao")].set_index("ponto_id")
    prim = json.loads((V5 / "s1_s2/analise/resultado.json").read_text(encoding="utf-8"))["comunidades_primarias"]
    R = V6 / "recife_avaliacao"
    rec = pd.read_parquet(R / "grade.parquet").merge(pd.read_parquet(R / "covariaveis.parquet"), on="ponto_id").merge(
        pd.read_parquet(R / "referencia.parquet"), on="ponto_id")
    rec = rec[rec.terreno_valido & rec.dist_canal_m.notna() & (rec.worldcover > 0) & rec.estado_E2_primario.isin(["positivo", "negativo"])].reset_index(drop=True)
    rec["y"] = rec.estado_E2_primario.eq("positivo").astype(int)
    doad = pd.read_parquet(V3A / "doadores.parquet")
    dt = doad[doad.pool.eq("internacional") & doad.dist_canal_m.notna() & doad.construido.notna()]
    tr = dt[dt.papel.eq("ajuste")]
    esc = ap.Escala(v5.transformar(tr, COLS))
    z = lambda q: esc(v5.transformar(q, COLS))
    th1, _ = v5.ajustar(z(tr), tr.classe.to_numpy(), balancear=True)
    zt, zr = z(te), z(rec)
    ids = pd.read_csv(V4 / "ids_orcamentos.csv")
    linhas, por_com = [], []

    def reg(nome, s1, s2, extra=None):
        o, pg = avaliar(nome, s1, te, prim, s2, rec, extra)
        linhas.append(o)
        por_com.append(pg.rename(nome))
    reg("A0_HAND", -te.hand_m.to_numpy(), -rec.hand_m.to_numpy())
    reg("A1_logistico", v5.escore(th1, zt), v5.escore(th1, zr))
    # M1
    w = np.where(tr.classe == 1, len(tr) / (2 * tr.classe.sum()), len(tr) / (2 * (len(tr) - tr.classe.sum())))
    g_ext = gbm().fit(z(tr), tr.classe, sample_weight=w)
    reg("M1_GBM_ext", g_ext.predict_proba(zt)[:, 1], g_ext.predict_proba(zr)[:, 1])
    acc = {k: [np.zeros(len(te)), np.zeros(len(rec))] for k in ["M1_GBM_loc", "M1_GBM_mix", "A3_logistico_uni400", "A2_forte_uni400"]}
    sementes = sorted(ids.semente.unique())
    for sem in sementes:
        o = ad.loc[ids[(ids.territorio == "rmpa") & (ids.politica == "urbano_uniforme") & (ids.semente == sem) & (ids.B == 400)].ponto_id]
        zo, yo = z(o), o.y.to_numpy()
        gl = gbm().fit(zo, yo)
        gm = gbm().fit(np.vstack([z(tr), zo]), np.r_[tr.classe.to_numpy(), yo], sample_weight=np.r_[w, np.full(len(yo), 4.0)])
        t3 = v5.ajustar(zo, yo, th1[:-1], 25.0)[0]
        t2 = v5.ajustar(zo, yo, None, 25.0)[0]
        for k, (a, b) in {"M1_GBM_loc": (gl.predict_proba(zt)[:, 1], gl.predict_proba(zr)[:, 1]),
                          "M1_GBM_mix": (gm.predict_proba(zt)[:, 1], gm.predict_proba(zr)[:, 1]),
                          "A3_logistico_uni400": (v5.escore(t3, zt), v5.escore(t3, zr)), "A2_forte_uni400": (v5.escore(t2, zt), v5.escore(t2, zr))}.items():
            acc[k][0] += a / len(sementes)
            acc[k][1] += b / len(sementes)
    for k, (a, b) in acc.items():
        reg(k, a, b, {"sementes": len(sementes)})
    print("M1", flush=True)
    # M2
    def desc(q):
        return pd.Series({"hand": np.log1p(q.hand_m.clip(lower=0).quantile(.9)), "slope": np.log1p(q.slope_deg.clip(lower=0).median()), "constr": q.construido.mean()})
    alvo = desc(ad.reset_index())
    du = tr.groupby("unidade_doadora").apply(desc, include_groups=False)
    sd = du.std(ddof=0).replace(0, 1)
    dist = (((du - alvo) / sd) ** 2).sum(axis=1).sort_values(kind="stable")
    sel = list(dist.index[:7])
    def a1_de(us):
        q = tr[tr.unidade_doadora.isin(us)]
        th, _ = v5.ajustar(z(q), q.classe.to_numpy(), balancear=True)
        return th
    th_ctx = a1_de(sel)
    reg("M2_A1_contexto7", v5.escore(th_ctx, zt), v5.escore(th_ctx, zr), {"doadores": ";".join(map(str, sel))})
    rng = np.random.default_rng(20261022)
    unidades = sorted(tr.unidade_doadora.unique())
    alea = []
    for i in range(20):
        th_r = a1_de(list(rng.choice(unidades, 7, replace=False)))
        o, pg = avaliar(f"M2_A1_aleatorio7_{i}", v5.escore(th_r, zt), te, prim, v5.escore(th_r, zr), rec)
        alea.append(o)
    al = pd.DataFrame(alea)
    linhas.append({"modelo": "M2_A1_aleatorio7_media20", **al.drop(columns="modelo").mean(numeric_only=True).to_dict(),
                   "dp_rmpa_fcu_macro10": float(al.rmpa_fcu_macro10.std()), "dp_recife_todas": float(al.recife_todas.std())})
    print("M2", flush=True)
    # M3
    inv = pd.read_csv(V4 / "inventario_grupos.csv")
    eleg = set(inv[(inv.territorio == "rmpa") & (inv.particao_v3 == "adaptacao") & inv.elegivel_nova_adaptacao].grupo)
    cand = d[d.particao.eq("adaptacao") & d.classe_territorial.eq("fcu") & d.grupo.isin(eleg)].reset_index(drop=True)
    gpos = cand[cand.y.eq(1)].grupo.unique()
    m3 = []
    for G in [8, 25, 50, 100]:
        for sem in sementes:
            r = np.random.default_rng(int(sem) + G)
            grupos = list(r.choice(sorted(cand.grupo.unique()), min(G, cand.grupo.nunique()), replace=False))
            gp = [x for x in grupos if x in set(gpos)]
            if len(gp) == 0:
                continue
            pos = cand[cand.y.eq(1) & cand.grupo.isin(gp)]
            neg = cand[cand.y.eq(0) & cand.grupo.isin(grupos)]
            if len(pos) < 8 or len(neg) < 92:
                continue
            pi = []
            ordem = pos.sample(frac=1, random_state=int(sem)).groupby("grupo").cumcount().sort_values(kind="stable").index[:8]
            pi = pos.loc[ordem]
            ni = neg.sample(92, random_state=int(sem))
            o = pd.concat([pi, ni])
            zo, yo = z(o), o.y.to_numpy()
            for br, th in [("A3", v5.ajustar(zo, yo, th1[:-1], 25.0)[0]), ("A2_forte", v5.ajustar(zo, yo, None, 25.0)[0])]:
                s = v5.escore(th, zt)
                f = te.classe_territorial.eq("fcu").to_numpy()
                pg = v5.auc_por_grupo(te.y.to_numpy()[f], s[f], te.grupo.to_numpy()[f], 5).reindex(prim)
                m3.append({"G_comunidades_sorteadas": G, "comunidades_no_orcamento": int(o.grupo.nunique()), "comunidades_com_positivo": int(pi.grupo.nunique()),
                           "semente": int(sem), "braco": br, "macro10": float(pg.mean()), "agrupada_fcu": auc_mann_whitney(te.y.to_numpy()[f], s[f])})
    m3 = pd.DataFrame(m3)
    print("M3", flush=True)
    # M4
    p4 = pd.read_csv(V5 / "s4/predicoes_competencia.csv")
    p4 = p4[p4.variante.eq("sem_observacao")]
    m4 = []
    for dest, q in p4.groupby("destino"):
        e = q.erro.to_numpy()
        cl = q.classe.to_numpy()
        k = int(round(.7 * len(e)))
        glob = np.argsort(q.previsto.to_numpy(), kind="stable")[:k]
        estr = []
        for c in np.unique(cl):
            idx = np.where(cl == c)[0]
            kk = int(round(.7 * len(idx)))
            estr.extend(idx[np.argsort(q.previsto.to_numpy()[idx], kind="stable")[:kk]])
        estr = np.array(estr)
        rr = np.random.default_rng(1)
        aleat = np.mean([e[rr.choice(len(e), len(estr), replace=False)].mean() for _ in range(2000)])
        cob = lambda ac: {c: float(np.isin(np.where(cl == c)[0], ac).mean()) for c in ["fcu", "baixa_renda", "comparador_urbano"] if (cl == c).any()}
        m4.append({"destino": dest, "unidades": len(e), "erro_todos": float(e.mean()), "erro_global70": float(e[glob].mean()), "cobertura_global": cob(glob),
                   "erro_estratificado70": float(e[estr].mean()), "cobertura_estratificada": cob(estr), "erro_aleatorio_mesma_cobertura": float(aleat)})
    t = pd.DataFrame(linhas)
    g6(D / "modelos.csv", t)
    g6(D / "por_comunidade.csv", pd.concat(por_com, axis=1).reset_index().rename(columns={"index": "grupo"}))
    g6(D / "diversidade.csv", m3)
    g6(D / "limites_equidade.csv", pd.DataFrame(m4))
    g6(D / "resultado.json", {"quando_utc": agora(), "natureza": "desenvolvimento; Recife E2 ja aberto -> desenvolvimento",
                              "modelos": t.to_dict(orient="records"),
                              "diversidade_resumo": m3.groupby(["G_comunidades_sorteadas", "braco"])[["comunidades_no_orcamento", "comunidades_com_positivo", "macro10", "agrupada_fcu"]].mean().reset_index().to_dict(orient="records"),
                              "limites_equidade": m4, "doadores_contexto": sel})
    print(t.round(3).to_string())
    print(m3.groupby(["G_comunidades_sorteadas", "braco"])[["comunidades_no_orcamento", "comunidades_com_positivo", "macro10", "agrupada_fcu"]].mean().round(3).to_string())
    print(pd.DataFrame(m4).round(3).to_string())


if __name__ == "__main__":
    main()
