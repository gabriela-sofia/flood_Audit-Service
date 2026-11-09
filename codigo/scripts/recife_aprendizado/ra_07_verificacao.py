"""Verificacao independente: metricas por postos, reinferencia dos pesos, particao/vazamento, fontes protegidas.

Nao importa as funcoes de metrica do treino.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from ra_comum import OUT, ROOT, SEMENTE, Etapa, gravar_json, hashes_protegidas, ler_json, sha256

RES = OUT / "resultados"
TOL = 1e-6


def auc_bairro_postos(v, y, b):
    vals = []
    for bb in np.unique(b[y == 1]):
        m = b == bb
        vp, vf = v[m & (y == 1)], v[m & (y == 0)]
        if len(vf) == 0:
            continue
        r = rankdata(np.r_[vp, vf])[:len(vp)] - rankdata(vp)
        vals += list(r / len(vf))
    return float(np.mean(vals))


def main() -> int:
    e = Etapa("ra_07_verificacao")
    rng = np.random.default_rng(SEMENTE + 5)
    rel_ = {"ok": True, "falhas": []}

    def checar(nome, cond, **det):
        rel_[nome] = {"ok": bool(cond), **det}
        if not cond:
            rel_["ok"] = False
            rel_["falhas"].append(nome)

    p = pd.read_parquet(RES / "pred_ruas.parquet")
    met = pd.read_csv(RES / "metricas_bracos.csv")
    te = p.papel.values == "teste"
    pe = p.tem_pe3d.values.astype(bool)
    y, b = p.y.values, p.CBAIRRCODI.values
    dif = []
    for _, r in met.iterrows():
        col = r.fit_id if isinstance(r.get("fit_id"), str) else r.braco
        mask = te & pe if r.subconjunto == "pe3d" else te
        v = p[col].values
        dif.append(abs(auc_bairro_postos(v[mask], y[mask], b[mask]) - r.AUC_bairro))
        dif.append(abs(roc_auc_score(y[mask], v[mask]) - r.AUC_agrupada))
    checar("metricas_recalculadas", max(dif) < TOL, linhas=len(met), max_dif=float(max(dif)))

    # reinferencia: coeficientes logisticos salvos e pickles do boosting
    t = pd.read_parquet(OUT / "atributos/atributos_objetos.parquet")
    assert (t.id_obj.values == p.id_obj.values).all()
    nq = ler_json(OUT / "atributos/nucleo_q_congelado.json")
    pad = ler_json(OUT / "pesos/padronizacao.json")
    aj = pd.read_csv(RES / "ajustes.csv")

    def z(cols):
        out = []
        for c in cols:
            if c in nq["principais"]:
                out.append(((t[c].fillna(nq["med"][c]) - nq["mu"][c]) / nq["sd"][c]).fillna(0).values)
            else:
                out.append(((t[c].fillna(pad["urbanos"]["med"][c]) - pad["urbanos"]["mu"][c]) / pad["urbanos"]["sd"][c]).values)
        return np.column_stack(out)

    amostra = aj.iloc[rng.choice(len(aj), 40, replace=False)]
    difs = []
    for _, r in amostra.iterrows():
        if isinstance(r.get("pkl"), str):
            with open(ROOT / r.pkl, "rb") as f:
                m = pickle.load(f)
            assert sha256(ROOT / r.pkl) == r.pkl_sha256
            nomes = nq["principais"] + ["dren_entradas_100m", "dren_dist_rede_m", "dren_frac_ligadas_5m", "dren_frac_canaleta", "saida_log_dist_canal"]
            X = z(nomes) if r.braco == "A2_hgb" else np.c_[t.q_escore.values, z(nomes[7:])]
            s = m.decision_function(X)
        else:
            nomes = r.atributos.split("|")
            s = r.intercepto + z(nomes) @ np.asarray([r[f"coef_{c}"] for c in nomes], float)
        v = p[r.fit_id].values
        ok = np.isfinite(v)
        difs.append(float(np.max(np.abs(s[ok] - v[ok]))))
    checar("reinferencia_pesos", max(difs) < 1e-6, ajustes=len(difs), max_dif=max(difs))

    # A1 = nucleo Q congelado recomputado pelo pickle original
    import sys
    sys.path[:0] = [str(ROOT / "scripts/vulnerabilizados"), str(ROOT / "scripts/metodo")]
    with open(ROOT / nq["origem"], "rb") as f:
        q = pickle.load(f)["Qsep_sel_fis"]
    cols = list(q.pad.mu.index)
    dq = float(np.max(np.abs(q.escore(t[cols]) - p.A1_Q.values)))
    checar("A1_igual_ao_nucleo_congelado", dq < 1e-9 and sha256(ROOT / nq["origem"]) == nq["sha256_pkl"], max_dif=dq)

    # particao e vazamento
    bt = set(p.CBAIRRCODI[te]); ba = set(p.CBAIRRCODI[~te])
    ids_teste = set(p.id_obj[te])
    consultados = {i for s in aj.consultas_em_ordem.dropna() for i in s.split(",")}
    checar("sem_vazamento", not (bt & ba) and not (consultados & ids_teste),
           bairros_em_comum=len(bt & ba), consultas_no_teste=len(consultados & ids_teste), consultados=len(consultados))

    refs = ler_json(OUT / "referencias/referencias.json")
    prot = ler_json(OUT / "protocolo/protocolo_v1.json")
    checar("referencias_iguais_ao_congelado", all(sha256(OUT / "referencias" / n) == h for n, h in prot["referencias_sha256"].items()))
    checar("contagens", int(y.sum()) == refs["objetos_positivos"] and int(y[te].sum()) == 48)

    ini = ler_json(OUT / "estado_inicial.json")["protegidas_sha256"]
    agora = hashes_protegidas()
    mud = sorted(k for k in ini if agora.get(k) != ini[k])
    checar("protegidas_intactas", not mud, alterados=mud, n=len(ini), novos=sorted(set(agora) - set(ini)))
    rel_["protocolo_sha256"] = sha256(OUT / "protocolo/protocolo_v1.json")
    rel_["emendas_sha256"] = {n: sha256(OUT / "protocolo" / n) for n in ("emenda_1.json", "emenda_2.json")}
    gravar_json(OUT / "verificacao_independente.json", rel_)
    e.saida(OUT / "verificacao_independente.json")
    e.fechar(ok=rel_["ok"], falhas=rel_["falhas"])
    print({k: (v["ok"] if isinstance(v, dict) and "ok" in v else v) for k, v in rel_.items()})
    return 0 if rel_["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
