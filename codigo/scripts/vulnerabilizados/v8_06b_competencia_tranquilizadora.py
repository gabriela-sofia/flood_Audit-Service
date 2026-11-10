"""v8: UMA modificacao pos-diagnostico da competencia (exploratoria, declarada).

Diagnostico do v8_06: o estimador de erro ensinado nas unidades externas (prevalencia ~45%) aprendeu que o erro esta no
alerta; nos destinos brasileiros (prevalencia 2-5%) recusou quase todos os positivos (RMPA 321/6.117; FCU 11/241).
Modificacao: o alerta ("maior suscetibilidade", topo 20%) NUNCA e recusado; a competencia so decide, entre os pontos
ditos "menor suscetibilidade", quais ficam "com suporte" e quais viram "evidencia insuficiente". Alvo do estimador:
P(y=1 | decisao baixa) -- a resposta incorretamente tranquilizadora. Mesmos atributos, mesmas unidades de competencia,
mesma avaliacao. Cobertura c = fracao de TODOS os pontos com resposta com suporte (alertas + baixos aceitos).
Controles na mesma cobertura: aleatorio entre os baixos, margem, DI, qualidade. Nenhum limiar buscado nos destinos.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_06_competencia import TOPO, atributos_comp, postos
from v8_comum import V8, agora, congelar8, etapa, g8, ler

D = V8 / "competencia_tranquilizadora"
COBERTURAS = [.5, .6, .7, .8, .9, 1.0]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    from v8_04_campanha import FEAT, caracteristicas, coorte, externo, z_externa
    congelar8(D / "protocolo.json", {"versao": "v8-competencia-2", "natureza": "uma modificacao pos-diagnostico, exploratoria",
                                     "regra": "alerta nunca recusado; competencia so entre decisoes baixas; alvo P(y=1|baixo)",
                                     "coberturas": COBERTURAS})
    part = ler(V8 / "grafo_particoes.json")["unidades"]
    with (V8 / "q_sep.pkl").open("rb") as f:
        q = pickle.load(f)
    mu, sd, med = q.pad.mu[FEAT], q.pad.sd[FEAT], q.pad.med[FEAT]
    tr, x, _ = externo()
    papel = tr.unidade.map(part)
    zf = z_externa(x[papel.eq("ajuste_fisico")], mu, sd, med)
    zf = zf[np.random.default_rng(20261022).choice(len(zf), min(20000, len(zf)), replace=False)]
    dmed = float(NearestNeighbors(n_neighbors=2).fit(zf).kneighbors(zf, 2)[0][:, 1].mean())
    ref = (NearestNeighbors(n_neighbors=1).fit(zf), dmed)

    def prep(xx, grank):
        r = postos(q.escore(xx), grank)
        return r, atributos_comp(z_externa(xx, mu, sd, med), xx, r, ref)
    cmp = papel.eq("competencia")
    rc, fc = prep(x[cmp], tr[cmp].evento_id.to_numpy())
    yc = tr[cmp].classe.to_numpy()
    baixo = rc <= 1 - TOPO
    w = 1 / tr[cmp].groupby("evento_id").evento_id.transform("size").to_numpy()
    est = LogisticRegression(C=1.0, max_iter=2000).fit(fc[baixo], yc[baixo], sample_weight=(w / w.mean())[baixo])
    alvos = [("retencao_externa", tr[papel.eq("retencao")].classe.to_numpy(), x[papel.eq("retencao")], tr[papel.eq("retencao")].evento_id.to_numpy(),
              np.array(["externo"] * int(papel.eq("retencao").sum())), None)]
    for c in ["rmpa", "manaus", "curitiba", "recife", "rmr"]:
        d = coorte(c)
        xx = caracteristicas(d)
        te = (d.particao.eq("teste") & d.estado_primaria.isin(["positivo", "negativo"])).to_numpy()
        alvos.append((c, d.estado_primaria.eq("positivo").astype(int).to_numpy()[te], xx[te], np.zeros(int(te.sum())),
                      d.classe_territorial.to_numpy()[te], d.grupo.to_numpy()[te]))
    linhas, comu = [], []
    for nome, y, xx, grank, classes, grupos in alvos:
        r, f = prep(xx, grank)
        alto = r > 1 - TOPO
        p = est.predict_proba(f[list(fc.columns)])[:, 1]
        riscos = {"aleatorio": np.random.default_rng(3).random(len(y)), "margem": -f.margem.to_numpy(), "di": f.di.to_numpy(),
                  "qualidade": f.falta_areia.to_numpy() + 1e-3 * f.plano.to_numpy(), "aprendido": p}
        n = len(y)
        for c in COBERTURAS:
            for met, risco in riscos.items():
                aceito = alto.copy()
                ib = np.where(~alto)[0]
                k = max(0, int(round(c * n)) - int(alto.sum()))
                aceito[ib[np.argsort(risco[ib], kind="stable")[:k]]] = True
                for cl in ["todas", "fcu", "baixa_renda", "comparador_urbano"]:
                    m = np.ones(n, bool) if cl == "todas" else (classes == cl)
                    if m.sum() == 0:
                        continue
                    a, yy, al = aceito[m], y[m], alto[m]
                    linhas.append({"destino": nome, "metodo": met, "cobertura_alvo": c, "classe": cl, "n": int(m.sum()), "cobertura": float(a.mean()),
                                   "positivos": int(yy.sum()), "positivos_em_alerta": int((al & (yy == 1)).sum()),
                                   "tranquilizadoras_erradas": int((a & ~al & (yy == 1)).sum()),
                                   "taxa_tranquilizadora_errada": float((a & ~al & (yy == 1)).sum() / max(1, (a & ~al).sum())),
                                   "positivos_insuficiente": int((~a & (yy == 1)).sum()), "falsos_alertas": int((al & (yy == 0)).sum()),
                                   "carga_revisao": int((~a).sum()), "revisoes_concluidas": 0})
                if met == "aprendido" and c == .7 and grupos is not None:
                    df = pd.DataFrame({"g": grupos, "cl": classes, "a": aceito, "al": alto, "y": y})
                    comu += [{"destino": nome, "grupo": g, "n": len(z), "positivos": int(z.y.sum()), "cobertura": float(z.a.mean()),
                              "tranquilizadoras_erradas": int((z.a & ~z.al & z.y.eq(1)).sum()), "positivos_em_alerta": int((z.al & z.y.eq(1)).sum()),
                              "positivos_insuficiente": int((~z.a & z.y.eq(1)).sum())} for g, z in df[df.cl.eq("fcu")].groupby("g") if z.y.sum() > 0]
        print(nome, flush=True)
    t = pd.DataFrame(linhas)
    g8(D / "cobertura_tranquilizadoras.csv", t)
    g8(D / "por_comunidade_c70.csv", pd.DataFrame(comu))
    g8(D / "resultado.json", {"quando_utc": agora(), "coeficientes": dict(zip(fc.columns, est.coef_[0].tolist())), "intercepto": float(est.intercept_[0]),
                              "prevalencia_baixos_competencia": float(yc[baixo].mean())})
    etapa("v8_06b_competencia_tranquilizadora", linhas=len(t))


if __name__ == "__main__":
    main()
