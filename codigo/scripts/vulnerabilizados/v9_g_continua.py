"""v9 Frente G: crescimento continuo que conserva capacidades (Q completo preservado; so a correcao Delta muda).

s_k(x) = Q_sel_ext(x) + c_k + theta_k . z(x)   (z = 7 atributos na escala externa)
Ordens: (Manaus, Curitiba, RMPA), (Curitiba, RMPA, Manaus), (RMPA, Manaus, Curitiba); orcamento urbano_uniforme B=400
(IDs v4) por etapa; 6 sementes. Estrategias (hiperparametros fixados ANTES, nenhum escolhido nas unidades de retencao):
  congelado        Delta = 0 (funcao Q)
  regularizada     prior = Delta anterior, tau 25, atualiza se houver duas classes
  regularizada_gate  idem, so com suporte (>= 5 pos, >= 5 neg, >= 3 agrupamentos com positivo); sem suporte conserva
  replay           dados novos + memoria de 200 exemplos antigos (balanceada), prior 0, tau 25, com gate
  der_pp           replay + distilacao dos logits guardados na memoria (DER++: alfa = 1 na MSE dos logits,
                   beta = 1 no rotulo da memoria), com gate
  conjunta         referencia NAO sequencial: todas as etapas juntas ao final
Medidas apos cada etapa: AUC todas/FCU nos testes das tres coortes e AUC dentro do evento media por unidade nas 6
unidades externas de retencao (grafo v8; nunca ensinaram nem escolheram nada aqui). Resumo: ganho no destino novo,
transferencia para tras (BWT) e esquecimento externo.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_comum import COORTES_B, V4, V8, V9, agora, auc, congelar9, etapa, g9, ler
from v9_p0_continuidade import GATE

D = V9 / "g_continua"
ORDENS = [("manaus", "curitiba", "rmpa"), ("curitiba", "rmpa", "manaus"), ("rmpa", "manaus", "curitiba")]
ESTRATEGIAS = ["congelado", "regularizada", "regularizada_gate", "replay", "der_pp"]
TAU, MEM, ALFA, BETA = 25.0, 200, 1.0, 1.0


def ajustar(z, y, q, prior, tau, zm=None, ym=None, qm=None, sm=None):
    """Logistica offset com prior gaussiano em theta; memoria opcional com rotulo (beta) e distilacao de logits (alfa)."""
    k = z.shape[1]
    w = np.where(y == 1, len(y) / (2 * max(1, y.sum())), len(y) / (2 * max(1, len(y) - y.sum())))

    def f(t):
        th, c = t[:-1], t[-1]
        s = q + z @ th + c
        e = w * (expit(s) - y)
        val = np.sum(w * (np.logaddexp(0, s) - y * s)) + .5 * tau * np.sum((th - prior) ** 2)
        g = np.r_[z.T @ e + tau * (th - prior), e.sum()]
        if zm is not None and len(zm):
            s2 = qm + zm @ th + c
            if ym is not None:
                e2 = BETA * (expit(s2) - ym)
                val += BETA * np.sum(np.logaddexp(0, s2) - ym * s2)
                g += np.r_[zm.T @ e2, e2.sum()]
            if sm is not None:
                r = s2 - sm
                val += ALFA * np.sum(r ** 2) / 2
                g += np.r_[ALFA * zm.T @ r, ALFA * r.sum()]
        return val, g
    r = minimize(f, np.r_[prior, 0.0], jac=True, method="L-BFGS-B")
    return r.x[:-1], float(r.x[-1])


def main():
    import mt01_experimento as e1
    from v8_04_campanha import FEAT, caracteristicas, coorte, externo, z_externa
    congelar9(D / "protocolo.json", {"versao": "v9-g-1", "ordens": ORDENS, "estrategias": ESTRATEGIAS + ["conjunta"], "tau": TAU, "memoria": MEM,
                                     "alfa_distilacao": ALFA, "beta_memoria": BETA, "gate": GATE, "orcamento": "urbano_uniforme B=400, 6 sementes",
                                     "retencao": "6 unidades externas do grafo v8; nao ajustam nada"})
    t0 = time.time()
    with (V9 / "p0_regularizacao/modelos.pkl").open("rb") as f:
        Q = pickle.load(f)["Q_sel_ext"]
    mu, sd, med = Q.pad.mu[FEAT], Q.pad.sd[FEAT], Q.pad.med[FEAT]
    tr, x, _ = externo()
    ret = tr.unidade.map(ler(V8 / "grafo_particoes.json")["unidades"]).eq("retencao").to_numpy()
    tr_ret, q_ret, z_ret = tr[ret], Q.escore(x[ret]), z_externa(x[ret], mu, sd, med)
    ids = pd.read_csv(V4 / "ids_orcamentos.csv")
    dados = {}
    for c in COORTES_B:
        d = coorte(c)
        xx = caracteristicas(d)
        dados[c] = {"d": d, "q": Q.escore(xx), "z": z_externa(xx, mu, sd, med), "y": d.estado_primaria.eq("positivo").astype(int).to_numpy(),
                    "te": (d.particao.eq("teste") & d.estado_primaria.isin(["positivo", "negativo"])).to_numpy(),
                    "fcu": d.classe_territorial.eq("fcu").to_numpy(), "idx": dict(zip(d.ponto_id, d.index))}

    def medir(th, cc, rotulo):
        out = []
        for c in COORTES_B:
            dd = dados[c]
            s = dd["q"] + dd["z"] @ th + cc
            for pop, m in [("todas", dd["te"]), ("fcu", dd["te"] & dd["fcu"])]:
                out.append({**rotulo, "avaliado": c, "populacao": pop, "auc": auc(dd["y"][m], s[m])})
        out.append({**rotulo, "avaliado": "retencao_externa", "populacao": "eventos", "auc": e1.auc_media_por_unidade(tr_ret, q_ret + z_ret @ th + cc)})
        return out
    linhas, ajustes = [], []
    zero = np.zeros(len(FEAT))
    for oi, ordem in enumerate(ORDENS):
        for sem in sorted(ids.semente.unique()):
            est = {e: (zero.copy(), 0.0) for e in ESTRATEGIAS}
            mem = {"z": np.zeros((0, len(FEAT))), "y": np.zeros(0), "q": np.zeros(0), "s": np.zeros(0)}
            acum = {"z": [], "y": [], "q": []}
            for e in ESTRATEGIAS:
                linhas += medir(*est[e], {"ordem": oi, "semente": int(sem), "estrategia": e, "etapa": 0, "apos": "inicio"})
            for k, c in enumerate(ordem, start=1):
                dd = dados[c]
                pid = ids[(ids.territorio == c) & (ids.politica == "urbano_uniforme") & (ids.B == 400) & (ids.semente == sem)].ponto_id
                ii = [dd["idx"][p] for p in pid if p in dd["idx"] and dd["d"].estado_primaria.iat[dd["idx"][p]] in ("positivo", "negativo")]
                za, ya, qa = dd["z"][ii], dd["y"][ii], dd["q"][ii]
                gp = set(dd["d"].grupo.to_numpy()[ii][ya == 1])
                duas = len(np.unique(ya)) == 2
                suporte = duas and ya.sum() >= GATE["positivos"] and (1 - ya).sum() >= GATE["negativos"] and len(gp) >= GATE["grupos_com_positivo"]
                for e in ESTRATEGIAS:
                    th0, c0 = est[e]
                    atualiza = {"congelado": False, "regularizada": duas, "regularizada_gate": suporte, "replay": suporte, "der_pp": suporte}[e]
                    if atualiza:
                        if e.startswith("regularizada"):
                            est[e] = ajustar(za, ya, qa, th0, TAU)
                        elif e == "replay":
                            est[e] = ajustar(za, ya, qa, zero, TAU, mem["z"], mem["y"], mem["q"], None)
                        else:
                            est[e] = ajustar(za, ya, qa, zero, TAU, mem["z"], mem["y"], mem["q"], mem["s"])
                    ajustes.append({"ordem": oi, "semente": int(sem), "etapa": k, "destino": c, "estrategia": e, "positivos": int(ya.sum()), "duas_classes": duas,
                                    "suporte": suporte, "atualizou": atualiza, "memoria": int(len(mem["y"])) if e in ("replay", "der_pp") else 0})
                    linhas += medir(*est[e], {"ordem": oi, "semente": int(sem), "estrategia": e, "etapa": k, "apos": c})
                # memoria: guarda exemplos e o logito do modelo der_pp NO MOMENTO (para distilar depois)
                th_d, c_d = est["der_pp"]
                for kk, v in (("z", za), ("y", ya), ("q", qa)):
                    acum[kk].append(v)
                Z, Yy, Qq = np.vstack(acum["z"]), np.concatenate(acum["y"]), np.concatenate(acum["q"])
                rng = np.random.default_rng(int(sem) * 10 + k)
                pos, neg = np.where(Yy == 1)[0], np.where(Yy == 0)[0]
                npos = min(len(pos), MEM // 2)
                esc = np.r_[rng.choice(pos, npos, replace=False) if npos else np.zeros(0, int), rng.choice(neg, min(len(neg), MEM - npos), replace=False)].astype(int)
                mem = {"z": Z[esc], "y": Yy[esc], "q": Qq[esc], "s": Qq[esc] + Z[esc] @ th_d + c_d}
            Z, Yy, Qq = np.vstack(acum["z"]), np.concatenate(acum["y"]), np.concatenate(acum["q"])
            linhas += medir(*ajustar(Z, Yy, Qq, zero, TAU), {"ordem": oi, "semente": int(sem), "estrategia": "conjunta", "etapa": 3, "apos": ordem[-1]})
        print(ordem, round(time.time() - t0), flush=True)
    m = pd.DataFrame(linhas)
    g9(D / "matriz_semente.csv", m)
    g9(D / "ajustes.csv", pd.DataFrame(ajustes))
    # resumo: por ordem/estrategia, media das sementes
    r = m.groupby(["ordem", "estrategia", "etapa", "avaliado", "populacao"]).auc.mean().reset_index()
    resumo = []
    for (oi, e), g in r.groupby(["ordem", "estrategia"]):
        ordem = ORDENS[oi]
        fin = g[g.etapa == g.etapa.max()]
        ini = r[(r.ordem == oi) & (r.estrategia == "congelado") & (r.etapa == 0)]
        lin = {"ordem": "->".join(ordem), "estrategia": e}
        for c in COORTES_B:
            kc = ordem.index(c) + 1
            aprendeu = g[(g.etapa == kc) & (g.avaliado == c) & (g.populacao == "todas")].auc
            base = ini[(ini.avaliado == c) & (ini.populacao == "todas")].auc
            final = fin[(fin.avaliado == c) & (fin.populacao == "todas")].auc
            if len(aprendeu) and len(base):
                lin[f"ganho_{c}"] = float(aprendeu.iloc[0] - base.iloc[0])
                lin[f"retido_{c}"] = float(final.iloc[0] - aprendeu.iloc[0]) if len(final) else None
        rr = fin[fin.avaliado == "retencao_externa"].auc
        lin["retencao_externa_final_menos_inicio"] = float(rr.iloc[0] - ini[ini.avaliado == "retencao_externa"].auc.iloc[0]) if len(rr) else None
        resumo.append(lin)
    g9(D / "resumo.csv", pd.DataFrame(resumo))
    g9(D / "resultado.json", {"quando_utc": agora(), "segundos": time.time() - t0, "linhas": len(m)})
    etapa("g_continua", segundos=time.time() - t0)
    print(pd.DataFrame(resumo).round(4).to_string())


if __name__ == "__main__":
    main()
