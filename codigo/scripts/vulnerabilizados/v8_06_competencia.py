"""v8: competencia e recusa -- tres estados, sem esconder quem e recusado.

Separacao por dependencia (grafo v8): Q_sep ajustado nas unidades de ajuste_fisico; o estimador de erro so nas 8 unidades
de competencia; avaliacao nas unidades de retencao externas (nunca ensinaram nada) e nas coortes brasileiras
(desenvolvimento ja examinado).
Regra de decisao (sem rotulo no destino): topo de 20% do escore Q_sep dentro do territorio/evento -> "maior
suscetibilidade"; o restante -> "menor suscetibilidade". Erro = alto com y=0 ou baixo com y=1.
Atributos de competencia (sem rotulo): margem = |posto - 0,8|; DI (Meyer & Pebesma 2021) = distancia ao vizinho mais
proximo do ajuste fisico / distancia media entre vizinhos no ajuste; falta_areia; log1p(declive); log1p(HAND).
Estimador: logistica regularizada (eventos com peso igual). Recusa = "evidencia insuficiente" para a fracao 1-c com maior
erro previsto. Controles na MESMA cobertura: aleatorio (esperado), so margem, so DI, so qualidade (falta_areia +
planura), aprendido, aprendido estratificado (mesma cobertura em cada classe social).
Medidas por cobertura c em {0,5..1,0}: erro dos aceitos, positivos aceitos, falsos negativos aceitos (positivo dito "menor
com suporte"), positivos recusados, negativos aceitos, carga de revisao (recusados), revisoes concluidas (0).
Calibracao: erro previsto x observado por decil nas unidades de retencao e nas coortes.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_comum import V8, agora, congelar8, etapa, g8, ler

D = V8 / "competencia"
COBERTURAS = [.5, .6, .7, .8, .9, 1.0]
TOPO = .2


def postos(s, grupos):
    return pd.Series(s).groupby(np.asarray(grupos)).rank(pct=True).to_numpy()


def atributos_comp(z, xraw, s_rank, ref_nn):
    from sklearn.neighbors import NearestNeighbors
    nn, dmed = ref_nn
    dist = nn.kneighbors(z, 1)[0][:, 0]
    return pd.DataFrame({"margem": np.abs(s_rank - (1 - TOPO)), "di": dist / dmed, "falta_areia": xraw.f_areia.isna().astype(float).to_numpy(),
                         "plano": -xraw.f_declive.fillna(xraw.f_declive.median()).to_numpy(), "hand": xraw.f_hand.fillna(0).to_numpy()})


def decidir(s_rank):
    return s_rank > 1 - TOPO


def avaliar(nome, y, alto, risco, classes, grupos, rng):
    err = (alto & (y == 0)) | (~alto & (y == 1))
    out, por_comu = [], []
    n = len(y)
    for estrat in ["global", "estratificado"]:
        for c in COBERTURAS:
            for metodo, r in risco.items():
                if estrat == "estratificado" and metodo != "aprendido":
                    continue
                if metodo == "aleatorio":
                    r = rng.random(n)
                aceito = np.zeros(n, bool)
                if estrat == "global":
                    aceito[np.argsort(r, kind="stable")[:int(round(c * n))]] = True
                else:
                    for cl in np.unique(classes):
                        ii = np.where(classes == cl)[0]
                        aceito[ii[np.argsort(r[ii], kind="stable")[:int(round(c * len(ii)))]]] = True
                for cl in ["todas", "fcu", "baixa_renda", "comparador_urbano"]:
                    m = np.ones(n, bool) if cl == "todas" else (classes == cl)
                    if m.sum() == 0:
                        continue
                    a, yy, ee, al = aceito[m], y[m], err[m], alto[m]
                    out.append({"destino": nome, "estrategia": estrat, "metodo": metodo, "cobertura_alvo": c, "classe": cl, "n": int(m.sum()),
                                "cobertura": float(a.mean()), "erro_todos": float(ee.mean()), "erro_aceitos": float(ee[a].mean()) if a.any() else np.nan,
                                "positivos": int(yy.sum()), "positivos_aceitos": int((a & (yy == 1)).sum()),
                                "falsos_negativos_aceitos": int((a & (yy == 1) & ~al).sum()), "falsos_positivos_aceitos": int((a & (yy == 0) & al).sum()),
                                "positivos_recusados": int((~a & (yy == 1)).sum()), "negativos_aceitos": int((a & (yy == 0)).sum()),
                                "carga_revisao": int((~a).sum()), "revisoes_concluidas": 0})
                if estrat == "global" and metodo == "aprendido" and c == .7 and grupos is not None:
                    df = pd.DataFrame({"g": grupos, "cl": classes, "a": aceito, "y": y, "e": err})
                    f = df[df.cl.eq("fcu")]
                    por_comu += [{"destino": nome, "grupo": g, "n": len(q), "positivos": int(q.y.sum()), "cobertura": float(q.a.mean()),
                                  "positivos_aceitos": int((q.a & q.y.eq(1)).sum()), "erro_aceitos": float(q.e[q.a].mean()) if q.a.any() else np.nan}
                                 for g, q in f.groupby("g") if q.y.sum() > 0]
    return out, por_comu


def calibracao(nome, prob, err):
    q = pd.qcut(pd.Series(prob).rank(method="first"), 10, labels=False)
    return [{"destino": nome, "decil": int(k), "previsto": float(prob[q == k].mean()), "observado": float(err[q == k].mean()), "n": int((q == k).sum())}
            for k in range(10)]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    from v8_04_campanha import FEAT, caracteristicas, coorte, externo, z_externa
    prot = congelar8(D / "protocolo.json", {"versao": "v8-competencia-1", "regra_decisao": f"topo {TOPO} do escore no territorio/evento",
                                            "atributos": ["margem", "di", "falta_areia", "plano", "hand"], "coberturas": COBERTURAS,
                                            "controles": ["aleatorio", "margem", "di", "qualidade", "aprendido", "aprendido estratificado"],
                                            "separacao": "fisico=ajuste_fisico; competencia=8 unidades; avaliacao=retencao externa + coortes BR"})
    part = ler(V8 / "grafo_particoes.json")["unidades"]
    with (V8 / "q_sep.pkl").open("rb") as f:
        q = pickle.load(f)
    mu, sd, med = q.pad.mu[FEAT], q.pad.sd[FEAT], q.pad.med[FEAT]
    tr, x, _ = externo()
    papel = tr.unidade.map(part)
    zf = z_externa(x[papel.eq("ajuste_fisico")], mu, sd, med)
    rng = np.random.default_rng(20261022)
    zf = zf[rng.choice(len(zf), min(20000, len(zf)), replace=False)]
    nn = NearestNeighbors(n_neighbors=2).fit(zf)
    dmed = float(nn.kneighbors(zf, 2)[0][:, 1].mean())
    ref = (NearestNeighbors(n_neighbors=1).fit(zf), dmed)

    def preparar(dd, xx, grupos_rank):
        s = q.escore(xx)
        r = postos(s, grupos_rank)
        z = z_externa(xx, mu, sd, med)
        return s, r, atributos_comp(z, xx, r, ref)
    cmp = papel.eq("competencia")
    tc, xc = tr[cmp], x[cmp]
    _, rc, fc = preparar(tc, xc, tc.evento_id.to_numpy())
    yc = tc.classe.to_numpy()
    ec = ((rc > 1 - TOPO) & (yc == 0)) | ((rc <= 1 - TOPO) & (yc == 1))
    w = 1 / tc.groupby("evento_id").evento_id.transform("size").to_numpy()
    est = LogisticRegression(C=1.0, max_iter=2000).fit(fc, ec, sample_weight=w / w.mean())
    with (D / "estimador_competencia.pkl").open("wb") as f:
        pickle.dump({"estimador": est, "atributos": list(fc.columns), "dmed": dmed}, f)
    linhas, comu, cal = [], [], []
    alvos = []
    ret = papel.eq("retencao")
    alvos.append(("retencao_externa", tr[ret], x[ret], tr[ret].evento_id.to_numpy(), None, np.array(["externo"] * int(ret.sum()))))
    for c in ["rmpa", "manaus", "curitiba", "recife", "rmr"]:
        d = coorte(c)
        xx = caracteristicas(d)
        te = (d.particao.eq("teste") & d.estado_primaria.isin(["positivo", "negativo"])).to_numpy()
        dd = d[te].assign(classe=lambda f: f.estado_primaria.eq("positivo").astype(int))
        alvos.append((c, dd, xx[te], np.zeros(int(te.sum())), dd.grupo.to_numpy(), dd.classe_territorial.to_numpy()))
    for nome, dd, xx, grank, grupos, classes in alvos:
        s, r, f = preparar(dd, xx, grank)
        y = dd.classe.to_numpy()
        alto = decidir(r)
        p = est.predict_proba(f[list(fc.columns)])[:, 1]
        risco = {"aleatorio": None, "margem": -f.margem.to_numpy(), "di": f.di.to_numpy(), "qualidade": f.falta_areia.to_numpy() + 1e-3 * f.plano.to_numpy(),
                 "aprendido": p}
        o, pc = avaliar(nome, y, alto, risco, classes, grupos, np.random.default_rng(1))
        linhas += o
        comu += pc
        cal += calibracao(nome, p, ((alto & (y == 0)) | (~alto & (y == 1))).astype(float))
        print(nome, len(y), flush=True)
    t = pd.DataFrame(linhas)
    g8(D / "erro_cobertura.csv", t)
    g8(D / "por_comunidade_c70.csv", pd.DataFrame(comu))
    g8(D / "calibracao.csv", pd.DataFrame(cal))
    g8(D / "resultado.json", {"quando_utc": agora(), "protocolo_quando": prot["quando_utc"], "coeficientes": dict(zip(fc.columns, est.coef_[0].tolist())),
                              "intercepto": float(est.intercept_[0]), "unidades_competencia": int(tc.unidade.nunique()),
                              "erro_base_competencia": float(ec.mean())})
    etapa("v8_06_competencia", linhas=len(t))


if __name__ == "__main__":
    main()
