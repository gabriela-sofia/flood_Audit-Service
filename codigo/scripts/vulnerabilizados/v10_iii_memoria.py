"""v10 Frente III: crescimento continuo com memoria REALMENTE limitada e logits preservados na insercao.

Nucleo limpo Qsep_sel_fis; s(x) = Q(x) + c + theta.z(x). Filtro de suporte da v9 em todas as variantes que atualizam.
Selecao (so desenvolvimento, nunca na retencao): pseudo-sequencia de 3 etapas com as 8 unidades de COMPETENCIA (ordem
sha256), metade dos pontos de cada unidade para ensinar (orcamento 400 por etapa) e metade para medir; grade M em
{50, 100, 200} x alfa em {0,3; 1; 3}; criterio = AUC media final nas metades de medida (3 sementes).
Aprendizes nas coortes (3 ordens x 6 sementes, orcamento v4 urbano_uniforme B=400):
  congelado          Delta = 0
  v9_variante        REPRODUCAO da v9: memoria de 200 reconstruida do historico COMPLETO a cada etapa e logits
                     recalculados com o modelo corrente (acesso ilimitado ao historico)
  replay_limitado    reservatorio de M* exemplos (so rotulos); nunca ve o historico fora do buffer
  der_limitado       reservatorio de M*; logit gravado NO MOMENTO DA INSERCAO (modelo apos aprender a etapa) e nunca
                     recalculado; perda = rotulo novo + beta*rotulo da memoria + alfa*MSE dos logits guardados (alfa*)
  der_so_logits      igual, beta = 0 (so distilacao)
  conjunta           referencia nao sequencial com historico completo (inacessivel aos aprendizes limitados)
Medidas por etapa: AUC todas/FCU nas tres coortes e AUC nas 6 unidades de RETENCAO (fora do nucleo e da selecao).
Estatistica: diferencas pareadas por (ordem, semente) -- repeticoes computacionais, nao territorios.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_comum import COORTES_B, V4, V10, agora, auc, congelar10, etapa, g10, nucleo_limpo, papeis

D = V10 / "iii_memoria"
ORDENS = [("manaus", "curitiba", "rmpa"), ("curitiba", "rmpa", "manaus"), ("rmpa", "manaus", "curitiba")]
GRADE_M, GRADE_A = [50, 100, 200], [.3, 1.0, 3.0]
TAU = 25.0
GATE = {"positivos": 5, "negativos": 5, "grupos_com_positivo": 3}


class Reservatorio:
    def __init__(self, M, rng):
        self.M, self.rng, self.n = M, rng, 0
        self.z, self.y, self.q, self.s = [], [], [], []

    def inserir(self, z, y, q, s):
        for i in range(len(y)):
            self.n += 1
            if len(self.y) < self.M:
                self.z.append(z[i]); self.y.append(y[i]); self.q.append(q[i]); self.s.append(s[i])
            else:
                j = self.rng.integers(self.n)
                if j < self.M:
                    self.z[j], self.y[j], self.q[j], self.s[j] = z[i], y[i], q[i], s[i]

    def arrays(self):
        if not self.y:
            return None
        return np.array(self.z), np.array(self.y, float), np.array(self.q), np.array(self.s)


def suporte(y, grupos):
    if len(np.unique(y)) < 2:
        return False
    return y.sum() >= GATE["positivos"] and (1 - y).sum() >= GATE["negativos"] and len(set(grupos[y == 1])) >= GATE["grupos_com_positivo"]


def aprender(estr, mem, za, ya, qa, th0, alfa):
    from v9_g_continua import ajustar
    zero = np.zeros(za.shape[1])
    if estr == "replay_limitado":
        a = mem.arrays()
        return ajustar(za, ya, qa, zero, TAU, *(a[:3] if a else (None, None, None)), None)
    if estr in ("der_limitado", "der_so_logits"):
        a = mem.arrays()
        if a is None:
            return ajustar(za, ya, qa, zero, TAU)
        import v9_g_continua as g
        b_ant, a_ant = g.BETA, g.ALFA
        g.BETA, g.ALFA = (1.0 if estr == "der_limitado" else 0.0), alfa
        try:
            return ajustar(za, ya, qa, zero, TAU, a[0], a[1] if estr == "der_limitado" else None, a[2], a[3])
        finally:
            g.BETA, g.ALFA = b_ant, a_ant
    raise ValueError(estr)


def main():
    import mt01_experimento as e1
    from v8_04_campanha import FEAT, caracteristicas, coorte, externo, z_externa
    from v9_g_continua import ajustar
    prot = congelar10(D / "protocolo.json", {"versao": "v10-iii-1", "grade_M": GRADE_M, "grade_alfa": GRADE_A, "tau": TAU, "gate": GATE, "ordens": ORDENS,
                                             "selecao": "pseudo-sequencia nas unidades de competencia; nunca retencao"})
    t0 = time.time()
    Q = nucleo_limpo()
    mu, sd, med = Q.pad.mu[FEAT], Q.pad.sd[FEAT], Q.pad.med[FEAT]
    tr, x, _ = externo()
    pap = tr.unidade.map(papeis())
    zall, qall = z_externa(x, mu, sd, med), Q.escore(x)
    yall = tr.classe.to_numpy()
    # ---------------- selecao em desenvolvimento (unidades de competencia)
    comp = sorted(tr[pap.eq("competencia")].unidade.unique(), key=lambda u: hashlib.sha256(f"v10|{u}".encode()).hexdigest())
    etapas = [comp[0:3], comp[3:6], comp[6:8]]
    metade = np.array([int(hashlib.sha256(p.encode()).hexdigest()[:2], 16) % 2 for p in tr.ponto_id])
    sel = []
    for estr in ["replay_limitado", "der_limitado"]:
        for M in GRADE_M:
            for alfa in (GRADE_A if estr == "der_limitado" else [None]):
                v = []
                for sem in (1, 2, 3):
                    rng = np.random.default_rng(sem)
                    mem = Reservatorio(M, rng)
                    th, cc = np.zeros(len(FEAT)), 0.0
                    for us in etapas:
                        idx = np.where(tr.unidade.isin(us).to_numpy() & (metade == 0))[0]
                        idx = rng.choice(idx, min(400, len(idx)), replace=False)
                        th, cc = aprender(estr, mem, zall[idx], yall[idx], qall[idx], th, alfa)
                        mem.inserir(zall[idx], yall[idx], qall[idx], qall[idx] + zall[idx] @ th + cc)
                    medida = tr.unidade.isin(comp).to_numpy() & (metade == 1)
                    v.append(e1.auc_media_por_unidade(tr[medida], qall[medida] + zall[medida] @ th + cc))
                sel.append({"estrategia": estr, "M": M, "alfa": alfa, "auc_dev": float(np.mean(v))})
    s = pd.DataFrame(sel)
    g10(D / "selecao_dev.csv", s)
    best = {e: s[s.estrategia == e].sort_values("auc_dev", ascending=False, kind="stable").iloc[0] for e in ["replay_limitado", "der_limitado"]}
    M_rep, M_der, A_der = int(best["replay_limitado"].M), int(best["der_limitado"].M), float(best["der_limitado"].alfa)
    print("selecao", M_rep, M_der, A_der, round(time.time() - t0), flush=True)
    # ---------------- coortes
    ret = pap.eq("retencao").to_numpy()
    R, qR, zR = tr[ret], qall[ret], zall[ret]
    ids = pd.read_csv(V4 / "ids_orcamentos.csv")
    dados = {}
    for c in COORTES_B:
        d = coorte(c)
        xx = caracteristicas(d)
        dados[c] = {"d": d, "q": Q.escore(xx), "z": z_externa(xx, mu, sd, med), "y": d.estado_primaria.eq("positivo").astype(int).to_numpy(),
                    "te": (d.particao.eq("teste") & d.estado_primaria.isin(["positivo", "negativo"])).to_numpy(), "fcu": d.classe_territorial.eq("fcu").to_numpy(),
                    "idx": dict(zip(d.ponto_id, d.index)), "g": d.grupo.to_numpy()}

    def medir(th, cc, rot):
        out = []
        for c in COORTES_B:
            dd = dados[c]
            sc = dd["q"] + dd["z"] @ th + cc
            out += [{**rot, "avaliado": c, "populacao": "todas", "auc": auc(dd["y"][dd["te"]], sc[dd["te"]])},
                    {**rot, "avaliado": c, "populacao": "fcu", "auc": auc(dd["y"][dd["te"] & dd["fcu"]], sc[dd["te"] & dd["fcu"]])}]
        out.append({**rot, "avaliado": "retencao", "populacao": "eventos", "auc": e1.auc_media_por_unidade(R, qR + zR @ th + cc)})
        return out
    linhas, regs = [], []
    ESTR = ["congelado", "v9_variante", "replay_limitado", "der_limitado", "der_so_logits"]
    zero = np.zeros(len(FEAT))
    for oi, ordem in enumerate(ORDENS):
        for sem in sorted(ids.semente.unique()):
            est = {e: (zero.copy(), 0.0) for e in ESTR}
            mems = {"replay_limitado": Reservatorio(M_rep, np.random.default_rng(int(sem))), "der_limitado": Reservatorio(M_der, np.random.default_rng(int(sem) + 1)),
                    "der_so_logits": Reservatorio(M_der, np.random.default_rng(int(sem) + 2))}
            hist = {"z": [], "y": [], "q": []}
            for e in ESTR:
                linhas += medir(*est[e], {"ordem": oi, "semente": int(sem), "estrategia": e, "etapa": 0})
            for k, c in enumerate(ordem, start=1):
                dd = dados[c]
                pid = ids[(ids.territorio == c) & (ids.politica == "urbano_uniforme") & (ids.B == 400) & (ids.semente == sem)].ponto_id
                ii = [dd["idx"][p] for p in pid if p in dd["idx"] and dd["d"].estado_primaria.iat[dd["idx"][p]] in ("positivo", "negativo")]
                za, ya, qa, ga = dd["z"][ii], dd["y"][ii], dd["q"][ii], dd["g"][ii]
                sup = suporte(ya, ga)
                for e in ESTR:
                    th0, c0 = est[e]
                    if e != "congelado" and sup:
                        if e == "v9_variante":
                            if hist["y"]:
                                Z, Y, Qq = np.vstack(hist["z"]), np.concatenate(hist["y"]), np.concatenate(hist["q"])
                                rng = np.random.default_rng(int(sem) * 10 + k)
                                pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]
                                npos = min(len(pos), 100)
                                esc = np.r_[rng.choice(pos, npos, replace=False) if npos else np.zeros(0, int), rng.choice(neg, min(len(neg), 200 - npos), replace=False)].astype(int)
                                est[e] = ajustar(za, ya, qa, zero, TAU, Z[esc], Y[esc], Qq[esc], Qq[esc] + Z[esc] @ th0 + c0)
                            else:
                                est[e] = ajustar(za, ya, qa, zero, TAU)
                        else:
                            est[e] = aprender(e, mems[e], za, ya, qa, th0, A_der)
                    if e in mems:
                        th1, c1 = est[e]
                        mems[e].inserir(za, ya, qa, qa + za @ th1 + c1)
                    regs.append({"ordem": oi, "semente": int(sem), "etapa": k, "destino": c, "estrategia": e, "suporte": bool(sup), "positivos": int(ya.sum()),
                                 "memoria": len(mems[e].y) if e in mems else (min(200, sum(len(v) for v in hist["y"])) if e == "v9_variante" else 0)})
                    linhas += medir(*est[e], {"ordem": oi, "semente": int(sem), "estrategia": e, "etapa": k})
                for kk, v in (("z", za), ("y", ya), ("q", qa)):
                    hist[kk].append(v)
            Z, Y, Qq = np.vstack(hist["z"]), np.concatenate(hist["y"]), np.concatenate(hist["q"])
            linhas += medir(*ajustar(Z, Y, Qq, zero, TAU), {"ordem": oi, "semente": int(sem), "estrategia": "conjunta", "etapa": 3})
        print(ordem, round(time.time() - t0), flush=True)
    m = pd.DataFrame(linhas)
    g10(D / "matriz.csv", m)
    g10(D / "registros.csv", pd.DataFrame(regs))
    # resumo pareado por (ordem, semente)
    pv = m.pivot_table(index=["ordem", "semente", "avaliado", "populacao", "etapa"], columns="estrategia", values="auc").reset_index()
    fin = pv[pv.etapa == 3]
    res = []
    for alvo in ["retencao", "rmpa", "manaus", "curitiba"]:
        for pop in (["eventos"] if alvo == "retencao" else ["todas", "fcu"]):
            f = fin[(fin.avaliado == alvo) & (fin.populacao == pop)]
            for a, b in [("der_limitado", "v9_variante"), ("der_limitado", "replay_limitado"), ("replay_limitado", "congelado"),
                         ("der_limitado", "congelado"), ("v9_variante", "congelado"), ("der_so_logits", "der_limitado")]:
                dif = (f[a] - f[b]).dropna().to_numpy()
                bs = [np.random.default_rng(i).choice(dif, len(dif)).mean() for i in range(2000)] if len(dif) else [np.nan]
                res.append({"avaliado": alvo, "populacao": pop, "a": a, "b": b, "media": float(dif.mean()) if len(dif) else None, "n_pares": len(dif),
                            "ic95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "a_maior_em": int((dif > 0).sum())})
    g10(D / "pareados_final.csv", pd.DataFrame(res))
    g10(D / "resultado.json", {"quando_utc": agora(), "segundos": time.time() - t0, "M_replay": M_rep, "M_der": M_der, "alfa_der": A_der,
                               "protocolo_quando": prot["quando_utc"]})
    etapa("iii_memoria", segundos=time.time() - t0)
    print(pd.DataFrame(res).round(4).to_string())


if __name__ == "__main__":
    main()
