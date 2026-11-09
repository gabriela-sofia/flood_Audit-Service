"""Emenda 1 (pos-teste, desenvolvimento): confundidor de comprimento e valor externo com blocos urbanos.

Motivo (visto DEPOIS do protocolo v1): muitas ruas positivas sao avenidas longas; o relato
e por endereco, entao rua longa tem mais oportunidade de relato, e a distancia MINIMA ao canal
cai mecanicamente com o comprimento. Tambem: o melhor braco foi o local com drenagem/saida, e o
contraste A4 - A2u nao estava no protocolo.

Diagnosticos congelados aqui, antes de calcular:
  E1a regra de comprimento: escore = log(comprimento) (AUC dentro do bairro no teste)
  E1b A4 - A2u (B=todas): o conhecimento externo acrescenta quando os blocos urbanos estao presentes?
  E1c A4 e A2u com log(comprimento) como atributo adicional (prior zero, lambda 25): o ganho urbano sobrevive?
  E1d saida pelo centroide (distancia do ponto medio da rua ao canal), em vez do minimo: A4c e A2uc
O v1 nao e alterado nem reinterpretado; estes resultados sao desenvolvimento.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ra_comum import MACRO, OUT, SEMENTE, Etapa, gravar_json, ler_json, sha256
from ra_04_treino import ajustar, boot_auc, escore, metricas, rotulo

RES = OUT / "resultados" / "emenda1"


def main() -> int:
    import geopandas as gpd
    from shapely.strtree import STRtree
    destino = OUT / "protocolo" / "emenda_1.json"
    if not destino.exists():
        gravar_json(destino, {"id": "recife-aprendizado-v1-emenda-1", "congelada_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "motivo": __doc__, "diagnosticos": ["E1a", "E1b", "E1c", "E1d"],
                              "regra": "a mesma do v1; resultados sao desenvolvimento pos-teste"})
    e = Etapa("ra_04b_emenda1")
    RES.mkdir(parents=True, exist_ok=True)
    prot = ler_json(OUT / "protocolo/protocolo_v1.json")
    QF, DF, SF = prot["blocos"]["Q30"], prot["blocos"]["DREN"], prot["blocos"]["SAIDA"]
    t = pd.read_parquet(OUT / "atributos/atributos_objetos.parquet")
    nq = ler_json(OUT / "atributos/nucleo_q_congelado.json")
    beta_q = np.asarray(nq["b"])
    obj = gpd.read_file(OUT / "referencias/objetos_rua_bairro.gpkg").set_index("id_obj").loc[t.id_obj]
    can = list(gpd.read_file(MACRO / "layer_02_canais_epsg31985.gpkg").geometry) + list(gpd.read_file(MACRO / "layer_03_rios_e_riachos_epsg31985.gpkg").geometry)
    arv = STRtree(can)
    meio = [g.interpolate(0.5, normalized=True) if g.geom_type == "LineString" else max(g.geoms, key=lambda x: x.length).interpolate(0.5, normalized=True)
            for g in obj.geometry]
    _, dc = arv.query_nearest(meio, return_distance=True, all_matches=False)
    t["saida_log_dist_canal_centroide"] = np.log1p(dc)
    t["log_comprimento"] = np.log(t.comprimento_m)
    aj = t.papel.values == "atualizacao"
    te = ~aj
    y = t.y.values.astype(int)
    bairro = t.CBAIRRCODI.astype(int).values
    zq = ((t[QF].fillna(pd.Series(nq["med"])[QF]) - pd.Series(nq["mu"])[QF]) / pd.Series(nq["sd"])[QF]).fillna(0).values

    def zloc(cols):
        x = t[cols].fillna(t.loc[aj, cols].median())
        return ((x - x[aj].mean()) / x[aj].std(ddof=0).replace(0, 1)).values

    U = DF + SF
    Uc = DF + ["saida_log_dist_canal_centroide"]
    esc = np.nonzero(aj & (y == 1))[0]
    idx = np.r_[esc, np.nonzero(aj & (y == 0) & np.isin(bairro, np.unique(bairro[esc])))[0]]
    specs = {
        "A2u": (np.c_[zq, zloc(U)], np.zeros(len(QF) + len(U))),
        "A4": (np.c_[zq, zloc(U)], np.r_[beta_q, np.zeros(len(U))]),
        "A2u_comp": (np.c_[zq, zloc(U + ["log_comprimento"])], np.zeros(len(QF) + len(U) + 1)),
        "A4_comp": (np.c_[zq, zloc(U + ["log_comprimento"])], np.r_[beta_q, np.zeros(len(U) + 1)]),
        "A3_comp": (np.c_[zq, zloc(["log_comprimento"])], np.r_[beta_q, 0.0]),
        "A4c": (np.c_[zq, zloc(Uc)], np.r_[beta_q, np.zeros(len(Uc))]),
        "A2uc": (np.c_[zq, zloc(Uc)], np.zeros(len(QF) + len(Uc))),
        "A4c_comp": (np.c_[zq, zloc(Uc + ["log_comprimento"])], np.r_[beta_q, np.zeros(len(Uc) + 1)]),
    }
    pred = {"id_obj": t.id_obj.values, "comprimento": t.comprimento_m.values, "regra_comprimento": t.log_comprimento.values,
            "A3_q": None}
    pv1 = pd.read_parquet(OUT / "resultados/pred_ruas.parquet")
    pred["A3_q"] = pv1["A3_q_l25__Btodas__s0"].values
    pred["A1_Q"] = pv1["A1_Q"].values
    coefs = []
    for nome, (X, b0) in specs.items():
        m = ajustar(X[idx], y[idx], 25.0, b0)
        pred[nome] = escore(m, X)
        coefs.append({"braco": nome, "intercepto": m["intercepto"], "coef": m["coef"], "convergiu": m["convergiu"]})
    met, pc = [], {}
    for nome, v in pred.items():
        if nome in ("id_obj", "comprimento"):
            continue
        mm, p = metricas(v[te], y[te], bairro[te])
        met.append({"vetor": nome, **mm})
        pc[nome] = p[None, :]
    bb = boot_auc(pc, y[te], bairro[te], semente=SEMENTE + 77)
    pont = {k: float(np.nanmean(v[:, y[te] == 1])) for k, v in pc.items()}
    pares = [("E1a_comprimento_vs_Q", "regra_comprimento", "A1_Q"), ("E1b_externo_com_urbanos", "A4", "A2u"),
             ("E1c_urbano_controlado_comprimento", "A4_comp", "A3_comp"), ("E1c_externo_controlado", "A4_comp", "A2u_comp"),
             ("E1d_urbano_centroide", "A4c", "A3_q"), ("E1d_urbano_centroide_comp", "A4c_comp", "A3_comp"),
             ("E1d_externo_centroide", "A4c", "A2uc"), ("E1_Q_controlado_comprimento", "A3_comp", "regra_comprimento")]
    cont = []
    for nome, a, b in pares:
        d = bb[a] - bb[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        cont.append({"contraste": nome, "a": a, "b": b, "AUC_a": pont[a], "AUC_b": pont[b], "delta": pont[a] - pont[b],
                     "ic95_lo": lo, "ic95_hi": hi, "rotulo_regra": rotulo(pont[a] - pont[b], lo, hi)})
    pd.DataFrame(met).to_csv(RES / "metricas_E1.csv", index=False)
    cdf = pd.DataFrame(cont)
    cdf.to_csv(RES / "contrastes_E1.csv", index=False)
    pd.DataFrame(pred).to_parquet(RES / "pred_E1.parquet", index=False)
    gravar_json(RES / "coeficientes_E1.json", coefs)
    desc = {"comprimento_mediano_positivas": float(t.loc[y == 1, "comprimento_m"].median()),
            "comprimento_mediano_fundo": float(t.loc[y == 0, "comprimento_m"].median()),
            "corr_spearman_comprimento_dist_min_canal": float(pd.Series(t.comprimento_m).corr(t.saida_dist_canal_m, method="spearman"))}
    gravar_json(RES / "resultado_E1.json", {"descritivo": desc, "contrastes": cont, "sha_emenda": sha256(destino)})
    e.saida(*RES.glob("*"), destino)
    e.fechar(**desc)
    pd.set_option("display.width", 220)
    print(desc)
    print(pd.DataFrame(met)[["vetor", "AUC_bairro", "AUC_agrupada", "AP"]].round(4).to_string())
    print(cdf.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
