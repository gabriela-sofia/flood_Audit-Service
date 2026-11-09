"""v10 Frente IV: competencia, calibracao e carga com cadeia limpa.

Base: Qsep_sel_fis (papel fisico). Regra de decisao: alerta = topo 20% do escore no territorio/evento (nunca recusado).
Modelos auxiliares LIMPOS (so ajuste_fisico, configuracao escolhida por dobras de unidades fisicas):
  GB monotonico (grade max_depth {2,3} x lr {0,03; 0,1}) e MLP residual (largura {64,128}); deep ensemble de 5 MLP.
Separacao por dependencia: unidades de competencia divididas por sha256 em A (4: ensinam o estimador de risco) e
B (4: calibram o limiar); avaliacao nas 6 unidades de retencao e nas coortes brasileiras.
Ordenadores para encaminhamento entre decisoes baixas: margem (controle obrigatorio), DI, discordancia (Q, GB, MLP),
ensemble (desvio de 5 MLP), aprendido (logistica em A), aleatorio.
Controle conformal de risco (Angelopoulos et al., CRC): perda L_i(lam) = 1[y_i = 1, decisao baixa, risco_i <= lam]
(positivo aceito como "menor suscetibilidade"), limitada em [0,1] e nao decrescente em lam; lam* = max{lam :
n/(n+1) R_n(lam) + 1/(n+1) <= alfa}, alfa = 0,02, calibrado em B. A garantia E[L] <= alfa supoe troca entre
calibracao e uso; entre unidades e coortes ela NAO e justificada (dependencia espacial, prevalencia e referencia
diferentes): a taxa empirica no destino e reportada como diagnostico, nao como garantia.
Custo por comunidade FCU sob orcamentos de encaminhamento (1%, 5%, 10% dos pontos): falsos alertas, positivos aceitos
como baixos, positivos encaminhados, cobertura, carga; revisoes concluidas = 0.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_comum import COORTES, V10, agora, auc, congelar10, etapa, g10, nucleo_limpo, papeis

D = V10 / "iv_competencia"
TOPO, ALFA = .2, .02
COBERTURAS = [.5, .6, .7, .8, .9, 1.0]
ORC = [.01, .05, .10]


def postos(s, grupos=None):
    if grupos is None:
        return pd.Series(s).rank(pct=True).to_numpy()
    return pd.Series(s).groupby(np.asarray(grupos)).rank(pct=True).to_numpy()


def lambda_crc(y, baixo, risco, alfa):
    """Maior lambda com n/(n+1)*R(lambda) + 1/(n+1) <= alfa, perda = positivo baixo aceito (risco <= lambda); -inf se nenhum."""
    n, lam = len(y), -np.inf
    for l_ in np.sort(np.unique(risco[baixo])):
        if n / (n + 1) * np.mean((y == 1) & baixo & (risco <= l_)) + 1 / (n + 1) <= alfa:
            lam = l_
    return lam


def main():
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    import mt01_experimento as e1
    from v8_04_campanha import FEAT, caracteristicas, coorte, externo, z_externa
    from v9_b_tabular import MONO, Prep, pesos, prever_torch, treinar_torch
    congelar10(D / "protocolo.json", {"versao": "v10-iv-1", "topo": TOPO, "alfa_crc": ALFA, "coberturas": COBERTURAS, "orcamentos": ORC,
                                      "perda_crc": "1[y=1, baixo, risco<=lam]", "divisao_competencia": "sha256 -> A (4) estimador, B (4) calibracao"})
    t0 = time.time()
    Q = nucleo_limpo()
    tr, x, _ = externo()
    pap = tr.unidade.map(papeis()).to_numpy()
    fis = pap == "ajuste_fisico"
    F, XF = tr[fis].reset_index(drop=True), x[fis].reset_index(drop=True)
    unid = np.array(sorted(F.unidade.unique()))
    dobra = dict(zip(unid, np.random.default_rng(20261024).permutation(len(unid)) % 3))
    fd = F.unidade.map(dobra).to_numpy()
    prep = Prep(XF)
    zF, qF, yF, wF = prep(XF), Q.escore(XF).astype("float32"), F.classe.to_numpy().astype("float32"), pesos(F).astype("float32")
    # selecao limpa das configuracoes auxiliares (dobras de unidades fisicas)
    def nota_gb(cfg):
        v = []
        for k in range(3):
            m = HistGradientBoostingClassifier(**cfg, max_iter=300, min_samples_leaf=100, monotonic_cst=MONO, random_state=0).fit(zF[fd != k], yF[fd != k], sample_weight=wF[fd != k])
            v.append(e1.auc_media_por_unidade(F[fd == k], m.predict_proba(zF[fd == k])[:, 1]))
        return float(np.mean(v))
    gcfgs = [{"max_depth": a, "learning_rate": b} for a in (2, 3) for b in (.03, .1)]
    gnot = {str(c): nota_gb(c) for c in gcfgs}
    gbest = max(gcfgs, key=lambda c: gnot[str(c)])
    gb = HistGradientBoostingClassifier(**gbest, max_iter=300, min_samples_leaf=100, monotonic_cst=MONO, random_state=0).fit(zF, yF, sample_weight=wF)

    def nota_mlp(cfg):
        v = []
        for k in range(3):
            r, _ = treinar_torch("MLP", cfg, zF[fd != k], yF[fd != k], wF[fd != k], qF[fd != k], 0, epocas=20)
            v.append(e1.auc_media_por_unidade(F[fd == k], prever_torch(r, zF[fd == k], qF[fd == k])))
        return float(np.mean(v))
    mcfgs = [{"largura": l, "profundidade": 2, "dropout": 0, "lr": 1e-3, "wd": 1e-4} for l in (64, 128)]
    mnot = {str(c): nota_mlp(c) for c in mcfgs}
    mbest = max(mcfgs, key=lambda c: mnot[str(c)])
    membros = [treinar_torch("MLP", mbest, zF, yF, wF, qF, s, epocas=20)[0] for s in range(11, 16)]
    D.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(membros):
        torch.save(r.state_dict(), D / f"ens_mlp_s{11 + i}.pt")
    print("auxiliares", gbest, mbest, round(time.time() - t0), flush=True)
    mu, sd, med = Q.pad.mu[FEAT], Q.pad.sd[FEAT], Q.pad.med[FEAT]
    zref = z_externa(XF, mu, sd, med)
    zref = zref[np.random.default_rng(1).choice(len(zref), min(20000, len(zref)), replace=False)]
    dmed = float(NearestNeighbors(n_neighbors=2).fit(zref).kneighbors(zref, 2)[0][:, 1].mean())
    nn1 = NearestNeighbors(n_neighbors=1).fit(zref)

    def atributos(xx, s, r):
        z = prep(xx)
        q = Q.escore(xx)
        ens = np.stack([prever_torch(m, z, q) for m in membros])
        disc = np.std(np.stack([postos(s), postos(gb.predict_proba(z)[:, 1]), postos(ens.mean(0))]), 0)
        return pd.DataFrame({"margem": -np.abs(r - (1 - TOPO)), "di": nn1.kneighbors(z_externa(xx, mu, sd, med), 1)[0][:, 0] / dmed, "discordancia": disc,
                             "ensemble": ens.std(0), "falta_areia": xx.f_areia.isna().astype(float).to_numpy(), "plano": -xx.f_declive.fillna(0).to_numpy()})
    comp = sorted(tr[pap == "competencia"].unidade.unique(), key=lambda u: hashlib.sha256(f"v10iv|{u}".encode()).hexdigest())
    A_u, B_u = comp[:4], comp[4:]
    cols = ["margem", "di", "discordancia", "ensemble", "falta_areia", "plano"]

    def bloco(us):
        m = tr.unidade.isin(us).to_numpy()
        t, xx = tr[m].reset_index(drop=True), x[m].reset_index(drop=True)
        s = Q.escore(xx)
        r = postos(s, t.evento_id)
        return t, xx, s, r, atributos(xx, s, r)
    tA, _, _, rA, fA = bloco(A_u)
    baixoA = rA <= 1 - TOPO
    est = LogisticRegression(max_iter=2000).fit(fA.loc[baixoA, cols], tA.classe.to_numpy()[baixoA])
    tB, _, _, rB, fB = bloco(B_u)
    yB, baixoB = tB.classe.to_numpy(), rB <= 1 - TOPO
    riscoB = est.predict_proba(fB[cols])[:, 1]
    lam = lambda_crc(yB, baixoB, riscoB, ALFA)
    alvos = [("retencao", *bloco(sorted(tr[pap == "retencao"].unidade.unique())))]
    linhas, orcs, comu, crc = [], [], [], []
    for c in COORTES:
        d = coorte(c)
        xx = caracteristicas(d)
        te = (d.particao.eq("teste") & d.estado_primaria.isin(["positivo", "negativo"])).to_numpy()
        t = d[te].reset_index(drop=True).assign(classe=lambda f: f.estado_primaria.eq("positivo").astype(int))
        xt = xx[te].reset_index(drop=True)
        s = Q.escore(xt)
        r = postos(s)
        alvos.append((c, t, xt, s, r, atributos(xt, s, r)))
    for nome, t, xt, s, r, f in alvos:
        y = t.classe.to_numpy()
        alto = r > 1 - TOPO
        risco = {"margem": f.margem.to_numpy(), "di": f.di.to_numpy(), "discordancia": f.discordancia.to_numpy(), "ensemble": f.ensemble.to_numpy(),
                 "aprendido": est.predict_proba(f[cols])[:, 1], "aleatorio": np.random.default_rng(5).random(len(y))}
        cl = t.classe_territorial.to_numpy() if "classe_territorial" in t else np.array(["externo"] * len(y))
        gr = t.grupo.to_numpy() if "grupo" in t else t.unidade.to_numpy()
        N = len(y)
        baixos = np.where(~alto)[0]
        for met, rk in risco.items():
            ordem = baixos[np.argsort(-rk[baixos], kind="stable")]
            for cv in COBERTURAS:
                enc = np.zeros(N, bool)
                enc[ordem[:max(0, N - int(round(cv * N)))]] = True
                for pop in ["todas", "fcu"]:
                    m = np.ones(N, bool) if pop == "todas" else cl == "fcu"
                    if not m.any():
                        continue
                    linhas.append({"destino": nome, "metodo": met, "cobertura_alvo": cv, "populacao": pop, "n": int(m.sum()), "positivos": int(y[m].sum()),
                                   "tranquilizadoras_erradas": int((~enc & ~alto & (y == 1))[m].sum()), "positivos_encaminhados": int((enc & (y == 1))[m].sum()),
                                   "falsos_alertas": int((alto & (y == 0))[m].sum()), "carga": int(enc[m].sum()), "cobertura": float((~enc[m]).mean()),
                                   "revisoes_concluidas": 0})
            for o in ORC:
                enc = np.zeros(N, bool)
                enc[ordem[:int(round(o * N))]] = True
                orcs.append({"destino": nome, "metodo": met, "orcamento": o, "carga": int(enc.sum()), "positivos_encaminhados": int((enc & (y == 1)).sum()),
                             "tranquilizadoras_erradas": int((~enc & ~alto & (y == 1)).sum())})
                if met in ("margem", "aprendido") and o == .05:
                    df = pd.DataFrame({"g": gr, "cl": cl, "y": y, "alto": alto, "enc": enc})
                    comu += [{"destino": nome, "metodo": met, "grupo": g, "pontos": len(z), "positivos": int(z.y.sum()), "falsos_alertas": int((z.alto & z.y.eq(0)).sum()),
                              "positivos_aceitos_baixos": int((~z.enc & ~z.alto & z.y.eq(1)).sum()), "positivos_encaminhados": int((z.enc & z.y.eq(1)).sum()),
                              "positivos_em_alerta": int((z.alto & z.y.eq(1)).sum()), "cobertura": float((~z.enc).mean()), "carga": int(z.enc.sum())}
                             for g, z in df[df.cl.eq("fcu")].groupby("g")]
        acc = ~alto & (risco["aprendido"] <= lam)
        crc.append({"destino": nome, "lambda": float(lam), "alfa": ALFA, "taxa_perda_empirica": float(np.mean((y == 1) & acc)),
                    "cobertura_total": float(np.mean(alto | acc)), "baixos_aceitos": int(acc.sum()), "prevalencia": float(y.mean())})
        print(nome, round(time.time() - t0), flush=True)
    g10(D / "cobertura.csv", pd.DataFrame(linhas))
    g10(D / "orcamentos.csv", pd.DataFrame(orcs))
    g10(D / "custo_por_comunidade_orc05.csv", pd.DataFrame(comu))
    g10(D / "crc.csv", pd.DataFrame(crc))
    g10(D / "resultado.json", {"quando_utc": agora(), "segundos": time.time() - t0, "gb_cfg": gbest, "gb_notas": gnot, "mlp_cfg": mbest, "mlp_notas": mnot,
                               "unidades_A": A_u, "unidades_B": B_u, "lambda_crc": float(lam), "calibracao_n": int(len(yB)),
                               "coef_aprendido": dict(zip(cols, est.coef_[0].tolist()))})
    etapa("iv_competencia", segundos=time.time() - t0)


if __name__ == "__main__":
    main()
