"""Verificacao independente: metricas por outra implementacao, reinferencia dos pesos salvos,
vazamento e preservacao das fontes protegidas.

Nao importa as funcoes de metrica do treino: o AUC por objeto aqui sai de postos
(scipy.stats.rankdata, Mann-Whitney por cicatriz), nao de busca ordenada com pesos.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from pa_comum import CHARTER, EPSG_PETROPOLIS, OUT, SEMENTE, Etapa, gravar_json, hashes_protegidas, ler_json, sha256

RES = OUT / "resultados"
REF = OUT / "referencias"
# Tolerancias: a primeira versao (1e-9 / 1e-8) falhou com 1,2e-9 (AUC_obj, soma de fracoes) e
# 5e-7 (escores). Causa conferida: o treino padroniza em float32 e a verificacao fazia em float64.
# A reinferencia agora reproduz o float32; a diferenca float64 continua relatada.
TOL_AUC = 1e-6
TOL_ESCORE = 1e-6


def auc_obj_postos(v_pos, cic, v_bg):
    """Para cada cicatriz: U de Mann-Whitney (celulas da cicatriz x fundo) / (n_c * n_bg), media das celulas."""
    out = []
    for c in np.unique(cic):
        vp = v_pos[cic == c]
        r = rankdata(np.r_[vp, v_bg])
        # posto de cada positivo entre fundo + ele mesmo e os demais positivos da cicatriz
        rp = r[:len(vp)]
        rp_proprio = rankdata(vp)
        u = (rp - rp_proprio).sum()           # quantos do fundo ficam abaixo (empates 1/2)
        out.append(u / (len(vp) * len(v_bg)))
    return float(np.mean(out))


def main() -> int:
    e = Etapa("pa_08_verificacao")
    rng = np.random.default_rng(SEMENTE + 4242)
    prot = ler_json(OUT / "protocolo/protocolo_v1.json")
    F = prot["atributos"]
    rel_ = {"ok": True, "falhas": []}

    def checar(nome, cond, **det):
        rel_[nome] = {"ok": bool(cond), **det}
        if not cond:
            rel_["ok"] = False
            rel_["falhas"].append(nome)

    # 1. metricas T1 recalculadas por postos
    pt = pd.read_parquet(RES / "pred_teste_2022.parquet")
    met = pd.read_csv(RES / "metricas_bracos.csv")
    pos = pt.y.values == 1
    cic = pt.id_cicatriz.values[pos]
    colunas = [c for c in pt.columns if c not in ("id", "y", "id_cicatriz", "bloco500", "D2_agua_svc03")]
    dif_obj, dif_auc, dif_ap = [], [], []
    for c in colunas:
        v = pt[c].values
        a = auc_obj_postos(v[pos], cic, v[~pos])
        linha = met[(met.fit_id == c)] if "__" in c else met[(met.braco == c) & (met.semente == -1)]
        dif_obj.append(abs(a - linha.AUC_obj.iloc[0]))
        dif_auc.append(abs(roc_auc_score(pt.y, v) - linha.AUC_cel.iloc[0]))
        dif_ap.append(abs(average_precision_score(pt.y, v) - linha.AP_cel.iloc[0]))
    checar("metricas_T1_recalculadas", max(dif_obj) < TOL_AUC and max(dif_auc) < TOL_AUC and max(dif_ap) < TOL_AUC,
           vetores=len(colunas), max_dif_AUC_obj=max(dif_obj), max_dif_AUC_cel=max(dif_auc), max_dif_AP=max(dif_ap))

    # 2. reinferencia a partir dos pesos salvos
    teste = pd.read_parquet(REF / "teste_petropolis_2022.parquet")
    pad = ler_json(OUT / "pesos/padronizacao.json")
    X64 = ((teste[F] - pd.Series(pad["media"])[F]) / pd.Series(pad["desvio"])[F]).values
    X = ((teste[F] - pd.Series(pad["media"])[F].astype("float32")) / pd.Series(pad["desvio"])[F].astype("float32")).values
    a1 = ler_json(OUT / "pesos/A1_log.json")
    d_a1 = np.max(np.abs(a1["intercepto"] + X @ np.asarray(a1["coef"]) - pt.A1_log.values))
    with open(OUT / "pesos/A1_hgb.pkl", "rb") as f:
        h = pickle.load(f)
    d_h = np.max(np.abs(h.decision_function(X) - pt.A1_hgb.values))
    aj = pd.read_csv(RES / "ajustes_locais.csv")
    amostra = aj.iloc[rng.choice(len(aj), 40, replace=False)]
    difs = []
    for _, r in amostra.iterrows():
        if isinstance(r.get("pkl"), str):
            with open(OUT.parents[2] / r.pkl, "rb") as f:
                m = pickle.load(f)
            assert sha256(OUT.parents[2] / r.pkl) == r.pkl_sha256
            s = m.decision_function(X)
        else:
            s = r.intercepto + X @ r[[f"coef_{c}" for c in F]].values.astype(float)
        difs.append(float(np.max(np.abs(s - pt[r.fit_id].values))))
    d_a1_64 = np.max(np.abs(a1["intercepto"] + X64 @ np.asarray(a1["coef"]) - pt.A1_log.values))
    checar("reinferencia_pesos_salvos", d_a1 < TOL_ESCORE and d_h < TOL_ESCORE and max(difs) < TOL_ESCORE,
           A1_log=float(d_a1), A1_log_se_float64=float(d_a1_64), A1_hgb=float(d_h), ajustes_locais_amostrados=len(difs), max_dif_local=max(difs))

    # 3. vazamento: pool fora do suporte de teste + 300 m
    import geopandas as gpd
    from shapely.geometry import Point
    ch = gpd.read_file(CHARTER).to_crs(EPSG_PETROPOLIS)
    excl = ch.union_all().convex_hull.buffer(300)
    pool = pd.read_parquet(REF / "pool_atualizacao_local.parquet")
    dentro = sum(excl.contains(Point(x, y)) for x, y in zip(pool.x, pool.yc))
    consultados = {i for s in aj.ids_sitios for i in s.split(",")}
    checar("sem_vazamento_pool_teste", dentro == 0 and consultados <= set(pool.id),
           sitios_pool_dentro_da_exclusao=int(dentro), sitios_consultados=len(consultados))

    # 4. emenda 1: OOF recalculado e cicatrizes de ajuste fora da dobra retida
    po = pd.read_parquet(RES / "emenda1/pred_oof_E1.parquet")
    me = pd.read_csv(RES / "emenda1/metricas_E1.csv").set_index("vetor")
    pos_o = po.y.values == 1
    difs_e1 = []
    for c in [c for c in po.columns if c in me.index]:
        v = po[c].values
        por = []
        for k in po.dobra.unique():
            mb = (~pos_o) & (po.dobra.values == k)
            mp = pos_o & (po.dobra.values == k)
            for cc in np.unique(po.id_cicatriz.values[mp]):
                vp = v[mp & (po.id_cicatriz.values == cc)]
                r = rankdata(np.r_[vp, v[mb]])[:len(vp)] - rankdata(vp)
                por.append(r.sum() / (len(vp) * mb.sum()))
        difs_e1.append(abs(np.mean(por) - me.loc[c, "AUC_obj_oof"]))
    aje = pd.read_csv(RES / "emenda1/ajustes_E1.csv")
    bairro = po[pos_o].groupby("id_cicatriz").dobra.first().to_dict()
    vaz = sum(any(bairro[int(i)] == r.dobra_retida for i in str(r.ids_cicatrizes).split(",")) for _, r in aje.iterrows())
    checar("emenda1_oof_e_dobras", max(difs_e1) < TOL_AUC and vaz == 0, vetores=len(difs_e1),
           max_dif_AUC_obj_oof=float(max(difs_e1)), ajustes_com_cicatriz_da_dobra_retida=int(vaz))

    # 5. contagens do protocolo
    refs = ler_json(REF / "referencias.json")
    checar("amostras_iguais_ao_congelado", all(sha256(REF / n) == h for n, h in prot["amostras_sha256"].items()))
    checar("contagens_teste", refs["petropolis_teste_2022"]["cicatrizes"] == 81 and int(pt.y.sum()) == refs["petropolis_teste_2022"]["celulas_positivas"],
           cicatrizes=refs["petropolis_teste_2022"]["cicatrizes"])

    # 6. fontes protegidas
    ini = ler_json(OUT / "estado_inicial.json")["protegidas_sha256"]
    agora = hashes_protegidas()
    mud = sorted(k for k in ini if agora.get(k) != ini[k])
    novos = sorted(set(agora) - set(ini))
    checar("protegidas_intactas", not mud, alterados=mud, novos_arquivos_nas_pastas_protegidas=novos, n=len(ini))

    rel_["protocolo_sha256"] = sha256(OUT / "protocolo/protocolo_v1.json")
    rel_["emenda1_sha256"] = sha256(OUT / "protocolo/emenda_1.json")
    gravar_json(OUT / "verificacao_independente.json", rel_)
    e.saida(OUT / "verificacao_independente.json")
    e.fechar(ok=rel_["ok"], falhas=rel_["falhas"])
    print({k: (v["ok"] if isinstance(v, dict) and "ok" in v else v) for k, v in rel_.items()})
    return 0 if rel_["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
