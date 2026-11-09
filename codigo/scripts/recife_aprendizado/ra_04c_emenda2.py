"""Emenda 2 (pos-emenda 1, desenvolvimento): o ganho do PE3D sobrevive ao controle de comprimento?

Motivo: a emenda 1 mostrou que o comprimento da rua (oportunidade de relato) explica boa parte
do ganho dos blocos urbanos. No PE3D, `pe3d_depr_max` e um maximo ao longo da rua e cresce com o
comprimento. Congelado antes de calcular:
  E2a no subconjunto PE3D, A5 - A4p com log(comprimento) nos dois lados
  E2b PE3D sem atributos dependentes de comprimento (sem depr_max): A5n - A4p, ambos com log(comprimento)
  E2c Q + comprimento + PE3D-neutro contra Q + comprimento (sem drenagem/saida)
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ra_comum import OUT, SEMENTE, Etapa, gravar_json, ler_json, sha256
from ra_04_treino import ajustar, boot_auc, escore, metricas, rotulo

RES = OUT / "resultados" / "emenda2"


def main() -> int:
    destino = OUT / "protocolo" / "emenda_2.json"
    if not destino.exists():
        gravar_json(destino, {"id": "recife-aprendizado-v1-emenda-2", "congelada_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "motivo": __doc__, "diagnosticos": ["E2a", "E2b", "E2c"], "regra": "a mesma do v1; desenvolvimento"})
    e = Etapa("ra_04c_emenda2")
    RES.mkdir(parents=True, exist_ok=True)
    prot = ler_json(OUT / "protocolo/protocolo_v1.json")
    QF, DF, SF, PF = (prot["blocos"][k] for k in ("Q30", "DREN", "SAIDA", "PE3D"))
    PFn = [c for c in PF if c != "pe3d_depr_max"]
    t = pd.read_parquet(OUT / "atributos/atributos_objetos.parquet")
    t["log_comprimento"] = np.log(t.comprimento_m)
    nq = ler_json(OUT / "atributos/nucleo_q_congelado.json")
    bq = np.asarray(nq["b"])
    aj = t.papel.values == "atualizacao"
    pe = t.tem_pe3d.values.astype(bool)
    y = t.y.values.astype(int)
    bairro = t.CBAIRRCODI.astype(int).values
    zq = ((t[QF].fillna(pd.Series(nq["med"])[QF]) - pd.Series(nq["mu"])[QF]) / pd.Series(nq["sd"])[QF]).fillna(0).values

    def zloc(cols):
        x = t[cols].fillna(t.loc[aj, cols].median())
        return ((x - x[aj].mean()) / x[aj].std(ddof=0).replace(0, 1)).values

    L = ["log_comprimento"]
    specs = {
        "A4p_comp": (np.c_[zq, zloc(DF + SF + L)], np.r_[bq, np.zeros(len(DF + SF) + 1)]),
        "A5_comp": (np.c_[zq, zloc(DF + SF + PF + L)], np.r_[bq, np.zeros(len(DF + SF + PF) + 1)]),
        "A5n_comp": (np.c_[zq, zloc(DF + SF + PFn + L)], np.r_[bq, np.zeros(len(DF + SF + PFn) + 1)]),
        "A3_comp": (np.c_[zq, zloc(L)], np.r_[bq, 0.0]),
        "A3_comp_pe3dn": (np.c_[zq, zloc(PFn + L)], np.r_[bq, np.zeros(len(PFn) + 1)]),
    }
    idx = np.nonzero(aj & pe)[0]
    te = (~aj) & pe
    pred, pc, met = {"id_obj": t.id_obj.values}, {}, []
    for nome, (X, b0) in specs.items():
        m = ajustar(X[idx], y[idx], 25.0, b0)
        v = escore(m, X)
        pred[nome] = v
        mm, p = metricas(v[te], y[te], bairro[te])
        met.append({"vetor": nome, "positivas_ajuste": int(y[idx].sum()), **mm})
        pc[nome] = p[None, :]
    bb = boot_auc(pc, y[te], bairro[te], semente=SEMENTE + 99)
    pont = {k: float(np.nanmean(v[:, y[te] == 1])) for k, v in pc.items()}
    cont = []
    for nome, a, b in [("E2a_PE3D_controlado", "A5_comp", "A4p_comp"), ("E2b_PE3D_neutro_controlado", "A5n_comp", "A4p_comp"),
                       ("E2c_PE3D_neutro_sobre_Q_comprimento", "A3_comp_pe3dn", "A3_comp")]:
        d = bb[a] - bb[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        cont.append({"contraste": nome, "a": a, "b": b, "AUC_a": pont[a], "AUC_b": pont[b], "delta": pont[a] - pont[b],
                     "ic95_lo": lo, "ic95_hi": hi, "rotulo_regra": rotulo(pont[a] - pont[b], lo, hi)})
    pd.DataFrame(met).to_csv(RES / "metricas_E2.csv", index=False)
    pd.DataFrame(cont).to_csv(RES / "contrastes_E2.csv", index=False)
    pd.DataFrame(pred).to_parquet(RES / "pred_E2.parquet", index=False)
    gravar_json(RES / "resultado_E2.json", {"contrastes": cont, "teste_positivas": int(y[te].sum()), "sha_emenda": sha256(destino)})
    e.saida(*RES.glob("*"), destino)
    e.fechar(teste_positivas=int(y[te].sum()))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(met)[["vetor", "positivas_ajuste", "AUC_bairro", "AUC_agrupada", "positivas"]].round(4).to_string())
    print(pd.DataFrame(cont).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
