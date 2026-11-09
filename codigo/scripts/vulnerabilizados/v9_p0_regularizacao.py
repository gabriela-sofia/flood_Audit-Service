"""v9 P0: regularizacao do Q escolhida SO em componentes estrangeiros.

Problema (revisao pos-v8): o Q da v8 reajustou coeficientes fora do Brasil, mas herdou lam = {p: 0,001, i: 100} escolhido
no MT03 sobre 91.446 linhas com Brasil e EMSR720. Aqui, o mesmo procedimento do MT01/MT03 (busca sequencial: lam_p
na grade MT01 com lam_i inicial, depois lam_i; 3 dobras de UNIDADES; nota = AUC dentro do evento media por unidade)
roda apenas sobre:
  sel_ext  todas as unidades fora_BR      -> Q ajustado em todo fora_BR (comparavel ao Q_ext v8)
  sel_fis  so as unidades de ajuste_fisico do grafo v8 -> Q_sep (competencia/retencao nunca escolhem lam)
Referencias: hist (lam herdado com Brasil) e fixo (lam inicial do MT01, p=0,01, i=1, sem selecao).
Avaliacao: coortes brasileiras (particao teste), AUC agrupada todas/FCU e contrastes contra hist com bootstrap por
comunidade/setor (300). Desenvolvimento; nenhuma confirmacao.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_comum import COORTES, V8, V9, agora, auc, congelar9, etapa, g9, ler

D = V9 / "p0_regularizacao"
GP = [0.0001, 0.001, 0.01, 0.1]
GI = [0.01, 0.1, 1.0, 10.0, 100.0]


def selecionar(mt, d, x, semente=20261010):
    unid = np.array(sorted(d.unidade.unique()))
    dobra = dict(zip(unid, np.random.default_rng(semente).permutation(len(unid)) % 3))
    fd = d.unidade.map(dobra).to_numpy()

    def nota(lam):
        v = []
        for k in range(3):
            tr, te = fd != k, fd == k
            m = mt.e1.Logistica(mt.BRACOS["Q_conexao"], mt.INTERACOES["Q_conexao"], lam_p=lam["p"], lam_i=lam["i"])
            m.fit(x[tr], d.classe[tr].to_numpy(float), d.evento_id[tr].to_numpy())
            v.append(mt.e1.auc_media_por_unidade(d[te], m.escore(x[te])))
        return float(np.nanmean(v))
    lam, notas = {"p": .001, "i": 1.0}, {}
    lam["p"] = max(GP, key=lambda l: notas.setdefault(f"p={l}", nota({**lam, "p": l})))
    lam["i"] = max(GI, key=lambda l: notas.setdefault(f"i={l}", nota({**lam, "i": l})))
    return lam, notas


def ajustar(mt, d, x, lam):
    m = mt.e1.Logistica(mt.BRACOS["Q_conexao"], mt.INTERACOES["Q_conexao"], lam_p=lam["p"], lam_i=lam["i"])
    m.fit(x, d.classe.to_numpy(float), d.evento_id.to_numpy())
    if not m.convergiu:
        raise ValueError("nao convergiu")
    return m


def main():
    from comum_tv import auc_mann_whitney, bootstrap_auc, intervalo
    from v8_01_q_externo import exportar
    from v8_04_campanha import caracteristicas, coorte, externo
    prot = congelar9(D / "protocolo.json", {"versao": "v9-p0-reg-1", "grade_p": GP, "grade_i": GI, "dobras": 3, "semente_dobras": 20261010,
                                            "nota": "AUC dentro do evento media por unidade", "variantes": ["hist", "fixo", "sel_ext", "sel_fis"],
                                            "hist": {"p": .001, "i": 100.0}, "fixo": {"p": .01, "i": 1.0}})
    t0 = time.time()
    tr, x, mt = externo()
    papel = tr.unidade.map(ler(V8 / "grafo_particoes.json")["unidades"])
    fis = papel.eq("ajuste_fisico").to_numpy()
    lam_ext, notas_ext = selecionar(mt, tr, x)
    lam_fis, notas_fis = selecionar(mt, tr[fis].reset_index(drop=True), x[fis].reset_index(drop=True))
    lams = {"hist": prot["hist"], "fixo": prot["fixo"], "sel_ext": lam_ext}
    modelos = {f"Q_{k}": ajustar(mt, tr, x, v) for k, v in lams.items()}
    modelos["Qsep_hist"] = ajustar(mt, tr[fis], x[fis], prot["hist"])
    modelos["Qsep_sel_fis"] = ajustar(mt, tr[fis], x[fis], lam_fis)
    D.mkdir(parents=True, exist_ok=True)
    with (D / "modelos.pkl").open("wb") as f:
        pickle.dump(modelos, f)
    g9(D / "pesos.json", {k: exportar(m) for k, m in modelos.items()})
    linhas, contr = [], []
    for c in COORTES:
        d = coorte(c)
        xx = caracteristicas(d)
        te = (d.particao.eq("teste") & d.estado_primaria.isin(["positivo", "negativo"])).to_numpy()
        dt = d[te].reset_index(drop=True)
        q = dt[["ponto_id", "grupo", "classe_territorial"]].assign(y=dt.estado_primaria.eq("positivo").astype(int))
        for k, m in modelos.items():
            q[k] = m.escore(xx[te])
        g9(D / f"predicoes_{c}.parquet", q)
        for pop in ["todas", "fcu"]:
            qq = q if pop == "todas" else q[q.classe_territorial.eq(pop)]
            linhas.append({"coorte": c, "populacao": pop, "positivos": int(qq.y.sum()), **{k: auc(qq.y, qq[k]) for k in modelos}})
            if qq.y.sum() >= 20 and qq[qq.y.eq(1)].grupo.nunique() >= 10:
                cols = list(modelos)
                bs = bootstrap_auc(qq, "grupo", cols, 300, 20261022)
                for a, b in [("Q_sel_ext", "Q_hist"), ("Q_fixo", "Q_hist"), ("Qsep_sel_fis", "Qsep_hist")]:
                    iv = intervalo(bs[:, cols.index(a)] - bs[:, cols.index(b)], .95)
                    contr.append({"coorte": c, "populacao": pop, "a": a, "b": b, "diferenca": auc_mann_whitney(qq.y, qq[a]) - auc_mann_whitney(qq.y, qq[b]), "ic95": iv})
        print(c, flush=True)
    g9(D / "auc_coortes.csv", pd.DataFrame(linhas))
    g9(D / "contrastes.csv", pd.DataFrame(contr))
    g9(D / "resultado.json", {"quando_utc": agora(), "segundos": time.time() - t0, "lam_sel_ext": lam_ext, "lam_sel_fis": lam_fis,
                              "notas_sel_ext": notas_ext, "notas_sel_fis": notas_fis, "unidades_sel_ext": int(tr.unidade.nunique()),
                              "unidades_sel_fis": int(tr[fis].unidade.nunique()),
                              "coeficientes": {k: dict(zip(m.principais, map(float, m.b))) for k, m in modelos.items()}})
    etapa("p0_regularizacao", segundos=time.time() - t0, lam_ext=lam_ext, lam_fis=lam_fis)
    print(lam_ext, lam_fis, pd.DataFrame(linhas).round(4).to_string())


if __name__ == "__main__":
    main()
