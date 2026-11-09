"""v9 Frente F: aquisicao ativa simulada (desenvolvimento retrospectivo) sob orcamento real de consultas.

Pool = TODOS os pontos dos agrupamentos da particao de adaptacao (v3), com QUALQUER estado da referencia (v1/v2):
consultar um ponto desconhecido ou de agua permanente gasta orcamento e nao devolve rotulo utilizavel. Rotulos ficam
escondidos ate a consulta. Teste (particao v3 'teste') nunca entra no pool.
Modelo: s = Q_sel_ext(x) + Delta(x) (correcao offset, tau 25), atualizada a cada lote de 25 SO se houver suporte
(>= 5 positivos, >= 5 negativos, >= 3 agrupamentos com positivo); sem suporte, Delta = 0 (funcao do Q preservada).
Estrategias (fixadas antes): uniforme; margem (|posto - 0,8| minimo, posto no pool); alto_escore; diversidade (k-centro
no espaco padronizado dos 7 atributos, sem rotulo); mistura (40% uniforme, 30% diversidade, 30% margem); BADGE
(embedding de gradiente da logistica offset, semeadura k-means++); oraculo_positivos (LIMITE SUPERIOR nao realizavel:
ve o rotulo escondido). Pools: todas as classes e so FCU. Coortes RMPA, Manaus, Curitiba; 6 sementes; ate 400 consultas.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_comum import COORTES_B, TER, TV, V3A, V8, V9, agora, auc, congelar9, etapa, g9
from v9_p0_continuidade import GATE, ajustar_offset

D = V9 / "f_aquisicao"
ESTRATEGIAS = ["uniforme", "margem", "alto_escore", "diversidade", "mistura", "badge", "oraculo_positivos"]
LOTE, MAX_B, TAU = 25, 400, 25.0
GRADES = {"rmpa": TV / "v1/confirmacao_rmpa", "manaus": TV / "v1/confirmacao_manaus", "curitiba": TV / "v2/curitiba"}


def pool_completo(c):
    """Pontos de todos os estados nos agrupamentos de adaptacao, com os atributos do contrato Q."""
    from v8_02_ponte_fisica import amostrar_raster
    from v8_02b_cobertura_solo import amostrar
    from mt02_atributos import distancias_da_unidade
    from mt11_confirmatorio_transdutivo import declive_p50
    alvo = D / f"pool_{c}.parquet"
    if alvo.exists():
        return pd.read_parquet(alvo)
    v3 = pd.read_parquet(V3A / f"{c}.parquet", columns=["grupo", "particao"]).drop_duplicates("grupo")
    adapt = set(v3[v3.particao.eq("adaptacao")].grupo)
    g = pd.read_parquet(GRADES[c] / "grade.parquet")
    g = g.merge(pd.read_parquet(GRADES[c] / "referencia.parquet", columns=["ponto_id", "estado_primaria"]), on="ponto_id")
    g = g[g.classe_territorial.isin(["fcu", "baixa_renda", "comparador_urbano"]) & g.terreno_valido]
    g["grupo"] = np.where(g.classe_territorial.eq("fcu"), "FCU:" + g.CD_FCU.astype(str), "SET:" + g.CD_SETOR.astype(str))
    g = g[g.grupo.isin(adapt)].reset_index(drop=True)
    partes = []
    for t, q in g.groupby("tile"):
        pasta = TER / t
        lon, lat = q.lon.to_numpy(), q.lat.to_numpy()
        d1, d2, _ = distancias_da_unidade(pasta, lon, lat)
        partes.append(q.assign(dist_drenagem_m=d1, dist_canal10_m=d2, declive_p50=declive_p50(pasta),
                               ln_sca=q.twi_dinf + np.log(np.tan(np.radians(np.clip(q.slope_deg, .01, None))))))
    g = pd.concat(partes, ignore_index=True)
    cob = V8 / "ponte/cobertura_solo_30m"
    g["construido_v2"] = amostrar(next(cob.glob(f"{c}_ghsl*.tif")), g.lon.to_numpy(), g.lat.to_numpy())
    g["areia_v2"] = amostrar(cob / f"{c}_areia.tif", g.lon.to_numpy(), g.lat.to_numpy())
    cols = ["ponto_id", "grupo", "classe_territorial", "estado_primaria", "hand_m", "slope_deg", "ln_sca", "construido_v2", "areia_v2",
            "dist_drenagem_m", "dist_canal10_m", "declive_p50"]
    g9(alvo, g[cols])
    return g[cols]


def k_centro(Z, Zusados, k, rng):
    """k-centro guloso; distancias iniciais aos pontos ja consultados (diversidade ao longo dos lotes)."""
    escolhidos = []
    dmin = np.full(len(Z), np.inf)
    for i in range(0, len(Zusados), 32):
        dmin = np.minimum(dmin, ((Z[:, None, :] - Zusados[None, i:i + 32, :]) ** 2).sum(-1).min(1))
    for _ in range(k):
        i = int(rng.integers(len(Z))) if not np.isfinite(dmin).any() else int(np.argmax(np.where(np.isfinite(dmin), dmin, -1)))
        escolhidos.append(i)
        dmin = np.minimum(dmin, ((Z - Z[i]) ** 2).sum(1))
    return escolhidos


def kmeanspp(G, k, rng):
    idx = [int(rng.integers(len(G)))]
    d = ((G - G[idx[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        p = d / d.sum() if d.sum() > 0 else None
        i = int(rng.choice(len(G), p=p))
        idx.append(i)
        d = np.minimum(d, ((G - G[i]) ** 2).sum(1))
    return idx


def main():
    from v8_04_campanha import FEAT, caracteristicas, coorte, z_externa
    from scipy.special import expit
    congelar9(D / "protocolo.json", {"versao": "v9-f-1", "estrategias": ESTRATEGIAS, "lote": LOTE, "max_B": MAX_B, "tau": TAU, "gate": GATE,
                                     "mistura": {"uniforme": .4, "diversidade": .3, "margem": .3}, "sementes": [1, 2, 3, 4, 5, 6],
                                     "pools": ["todas", "fcu"], "oraculo": "limite superior nao realizavel"})
    t0 = time.time()
    with (V9 / "p0_regularizacao/modelos.pkl").open("rb") as f:
        Q = pickle.load(f)["Q_sel_ext"]
    mu, sd, med = Q.pad.mu[FEAT], Q.pad.sd[FEAT], Q.pad.med[FEAT]
    linhas = []
    for c in COORTES_B:
        pool_all = pool_completo(c)
        xp = caracteristicas(pool_all)
        qp_all, zp_all = Q.escore(xp), z_externa(xp, mu, sd, med)
        d = coorte(c)
        xt = caracteristicas(d)
        te = (d.particao.eq("teste") & d.estado_primaria.isin(["positivo", "negativo"])).to_numpy()
        qt, zt = Q.escore(xt)[te], z_externa(xt, mu, sd, med)[te]
        yt = d.estado_primaria.eq("positivo").to_numpy()[te]
        ft = d.classe_territorial.eq("fcu").to_numpy()[te]
        for nome_pool in ["todas", "fcu"]:
            m = np.ones(len(pool_all), bool) if nome_pool == "todas" else pool_all.classe_territorial.eq("fcu").to_numpy()
            P = pool_all[m].reset_index(drop=True)
            qp, zp = qp_all[m], zp_all[m]
            est = P.estado_primaria.to_numpy()
            yv = np.where(est == "positivo", 1, np.where(est == "negativo", 0, -1))
            grupos = P.grupo.to_numpy()
            for estrat in ESTRATEGIAS:
                for sem in range(1, 7):
                    rng = np.random.default_rng(sem * 1000 + ESTRATEGIAS.index(estrat))
                    usado = np.zeros(len(P), bool)
                    th, cc, atualizado = np.zeros(len(FEAT)), 0.0, False
                    for B in range(LOTE, MAX_B + 1, LOTE):
                        livres = np.where(~usado)[0]
                        s_pool = qp + (zp @ th + cc if atualizado else 0)
                        if estrat == "uniforme":
                            esc = rng.choice(livres, LOTE, replace=False)
                        elif estrat == "alto_escore":
                            esc = livres[np.argsort(-s_pool[livres], kind="stable")[:LOTE]]
                        elif estrat == "margem":
                            r = pd.Series(s_pool[livres]).rank(pct=True).to_numpy()
                            esc = livres[np.argsort(np.abs(r - .8), kind="stable")[:LOTE]]
                        elif estrat == "diversidade":
                            sub = livres if len(livres) <= 30000 else rng.choice(livres, 30000, replace=False)
                            esc = sub[k_centro(zp[sub], zp[usado], LOTE, rng)]
                        elif estrat == "mistura":
                            nu, nd = 10, 8
                            e1_ = rng.choice(livres, nu, replace=False)
                            resto = np.setdiff1d(livres, e1_)
                            sub = resto if len(resto) <= 30000 else rng.choice(resto, 30000, replace=False)
                            e2_ = sub[k_centro(zp[sub], zp[usado], nd, rng)]
                            resto = np.setdiff1d(resto, e2_)
                            r = pd.Series(s_pool[resto]).rank(pct=True).to_numpy()
                            e3_ = resto[np.argsort(np.abs(r - .8), kind="stable")[:LOTE - nu - nd]]
                            esc = np.r_[e1_, e2_, e3_]
                        elif estrat == "badge":
                            sub = livres if len(livres) <= 30000 else rng.choice(livres, 30000, replace=False)
                            p = expit(s_pool[sub])
                            G = (p - (p > .5))[:, None] * np.c_[zp[sub], np.ones(len(sub))]
                            esc = sub[kmeanspp(G, LOTE, rng)]
                        else:
                            pos = livres[yv[livres] == 1]
                            esc = np.r_[pos[:LOTE], rng.choice(np.setdiff1d(livres, pos), max(0, LOTE - len(pos[:LOTE])), replace=False)][:LOTE]
                        usado[np.asarray(esc, int)] = True
                        ok = usado & (yv >= 0)
                        ya = yv[ok]
                        gp = set(grupos[ok][ya == 1])
                        suporte = ya.sum() >= GATE["positivos"] and (1 - ya).sum() >= GATE["negativos"] and len(gp) >= GATE["grupos_com_positivo"]
                        if suporte:
                            th, cc, atualizado = ajustar_offset(zp[ok], ya, qp[ok], TAU)
                        st = qt + (zt @ th + cc if atualizado else 0)
                        linhas.append({"coorte": c, "pool": nome_pool, "estrategia": estrat, "semente": sem, "consultados": int(usado.sum()),
                                       "validos": int(ok.sum()), "desconhecidos_ou_permanentes": int((usado & (yv < 0)).sum()),
                                       "positivos": int(ya.sum()), "agrupamentos_com_positivo": len(gp), "agrupamentos_consultados": int(len(set(grupos[usado]))),
                                       "fcu_consultados": int((usado & P.classe_territorial.eq("fcu").to_numpy()).sum()), "atualizou": bool(atualizado),
                                       "auc_teste_todas": auc(yt, st), "auc_teste_fcu": auc(yt[ft], st[ft])})
                print(c, nome_pool, round(time.time() - t0), flush=True)
                g9(D / "curvas_semente.parquet", pd.DataFrame(linhas))
    t = pd.DataFrame(linhas)
    g9(D / "curvas_semente.parquet", t)
    r = t.groupby(["coorte", "pool", "estrategia", "consultados"])[["validos", "desconhecidos_ou_permanentes", "positivos", "agrupamentos_com_positivo",
                                                                    "atualizou", "auc_teste_todas", "auc_teste_fcu"]].mean().reset_index()
    g9(D / "curvas.csv", r)
    g9(D / "resultado.json", {"quando_utc": agora(), "segundos": time.time() - t0, "linhas": len(t),
                              "pool_tamanhos": {c: int(len(pd.read_parquet(D / f"pool_{c}.parquet"))) for c in COORTES_B}})
    etapa("f_aquisicao", segundos=time.time() - t0)


if __name__ == "__main__":
    main()
