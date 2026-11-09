"""v9 Frente B: modelos tabulares mais fortes e correcoes residuais sobre o Q (zero-shot nas coortes brasileiras).

Dados de ajuste: amostra MT03-A fora_BR (36.331 linhas, 59 eventos, 43 unidades; CEMS/UFO/Sen1Floods11), pesos com
evento = 1 e classe = metade (como o Q). Atributos: os 7 do contrato Q + z_relevo. Nenhum rotulo brasileiro.
Familias e construcoes:
  GB_mono / GB_livre     HistGradientBoosting direto, com/sem monotonia fisica (hand-, sca+, declive-, canal-)
  GBR_residual           GradientBoosting com init = Q (correcao residual sobre o logito do Q)
  MLP_direto / MLP_res   rede densa (torch); a residual soma ao logito do Q com camada final iniciada em zero
  FTT_direto             FT-Transformer pequeno (tokens por atributo)
  ENS_tab                media de postos dos finalistas (ensemble)
Busca progressiva (halving sucessivo): todas as configuracoes em 1 dobra de unidades; metade melhor em 3 dobras;
2 finalistas por familia em 3 sementes. Nota de selecao = AUC dentro do evento media por unidade (validacao agrupada
EXTERNA). O destino nunca participa da selecao. TabPFN nao foi usado: os pesos exigem aceitar termos de licenca e o
acervo de ajuste (36 mil linhas) excede o regime publicado de ate 10 mil exemplos.
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_comum import COORTES, V9, agora, auc, congelar9, etapa, g9

D = V9 / "b_tabular"
COLS = ["f_hand", "f_sca", "f_declive", "f_construido", "f_areia", "f_dist_dren", "f_dist_canal", "z_relevo"]
MONO = [-1, 1, -1, 0, 0, 0, -1, 0]


def pesos(tr):
    n = tr.groupby(["evento_id", "classe"]).classe.transform("size")
    w = (.5 / n / tr.evento_id.nunique()).to_numpy()
    return w / w.mean()


class Prep:
    """Mediana + padronizacao ajustadas so no treino admitido."""
    def __init__(self, x):
        self.med, self.mu = x[COLS].median(), None
        z = x[COLS].fillna(self.med)
        self.mu, self.sd = z.mean(), z.std().replace(0, 1)

    def __call__(self, x):
        return ((x[COLS].fillna(self.med) - self.mu) / self.sd).to_numpy("float32")


def treinar_torch(tipo, cfg, z, y, w, q=None, semente=0, epocas=None, zval=None, yval=None, qval=None):
    import torch
    import torch.nn as nn
    torch.manual_seed(semente)
    torch.set_num_threads(4)
    k = z.shape[1]
    if tipo == "MLP":
        cam, larg = [], k
        for _ in range(cfg["profundidade"]):
            cam += [nn.Linear(larg, cfg["largura"]), nn.GELU(), nn.Dropout(cfg["dropout"])]
            larg = cfg["largura"]
        final = nn.Linear(larg, 1)
        rede = nn.Sequential(*cam, final)
    else:
        class FTT(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb_w = nn.Parameter(torch.randn(k, cfg["d"]) * .1)
                s.emb_b = nn.Parameter(torch.zeros(k, cfg["d"]))
                s.cls = nn.Parameter(torch.zeros(1, 1, cfg["d"]))
                camada = nn.TransformerEncoderLayer(cfg["d"], cfg["cabecas"], cfg["d"] * 2, cfg["dropout"], batch_first=True)
                s.enc = nn.TransformerEncoder(camada, cfg["blocos"])
                s.saida = nn.Linear(cfg["d"], 1)

            def forward(s, xx):
                t = xx.unsqueeze(-1) * s.emb_w + s.emb_b
                t = torch.cat([s.cls.expand(len(xx), -1, -1), t], 1)
                return s.saida(s.enc(t)[:, 0])
        rede = FTT()
        final = rede.saida
    if q is not None:
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
    opt = torch.optim.AdamW(rede.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    Z, Y, W = torch.tensor(z), torch.tensor(y, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)
    Qt = torch.tensor(q, dtype=torch.float32) if q is not None else torch.zeros(len(y))
    n, lote = len(y), 512
    melhor, paciencia, estado = -np.inf, 0, None
    hist = []
    for ep in range(epocas or 60):
        rede.train()
        perm = torch.randperm(n)
        for i in range(0, n, lote):
            b = perm[i:i + lote]
            logit = rede(Z[b]).squeeze(-1) + Qt[b]
            perda = (nn.functional.binary_cross_entropy_with_logits(logit, Y[b], reduction="none") * W[b]).mean()
            opt.zero_grad()
            perda.backward()
            opt.step()
        if zval is not None:
            s = prever_torch(rede, zval, qval)
            a = auc(yval, s)
            hist.append(a)
            if a > melhor + 1e-4:
                melhor, paciencia, estado = a, 0, {kk: v.clone() for kk, v in rede.state_dict().items()}
            else:
                paciencia += 1
                if paciencia >= 8:
                    break
    if estado is not None:
        rede.load_state_dict(estado)
    return rede, {"epocas": ep + 1, "melhor_val": melhor, "historico": hist}


def prever_torch(rede, z, q=None):
    import torch
    rede.eval()
    with torch.no_grad():
        s = rede(torch.tensor(z)).squeeze(-1).numpy()
    return s + (q if q is not None else 0)


class QInit:
    """`init` do GradientBoosting: a ultima coluna de X e o logito do Q; a previsao inicial e expit(Q).
    Declarado: as arvores tambem podem dividir pela coluna do Q (correcao residual com Q como atributo)."""
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        from scipy.special import expit
        p = expit(np.asarray(X[:, -1], float))
        return np.c_[1 - p, p]


def configuracoes():
    gb = [{"max_depth": d, "learning_rate": lr, "min_samples_leaf": f, "max_iter": 300} for d, lr, f in itertools.product([2, 3, 4, 6], [.03, .1], [20, 100])]
    mlp = [{"largura": l, "profundidade": p, "dropout": dr, "lr": 1e-3, "wd": wd} for l, p, dr, wd in itertools.product([32, 64, 128], [2, 3], [0, .2], [1e-4, 1e-2])]
    ftt = [{"d": d, "cabecas": 4, "blocos": b, "dropout": .1, "lr": lr, "wd": 1e-4} for d, b, lr in itertools.product([16, 32], [1, 2, 3], [3e-4, 1e-3])]
    gbr = [{"max_depth": d, "learning_rate": lr, "n_estimators": 200, "subsample": .8} for d, lr in itertools.product([2, 3], [.03, .1])]
    return {"GB_mono": gb, "GB_livre": gb, "MLP_direto": mlp, "MLP_res": mlp, "FTT_direto": ftt, "GBR_residual": gbr}


def ajustar_familia(fam, cfg, tr, x, Q, idx_tr, idx_val, semente, epocas=None):
    from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
    prep = Prep(x.iloc[idx_tr])
    ztr, zva = prep(x.iloc[idx_tr]), (prep(x.iloc[idx_val]) if idx_val is not None else None)
    y, w = tr.classe.to_numpy()[idx_tr], pesos(tr.iloc[idx_tr])
    yva = tr.classe.to_numpy()[idx_val] if idx_val is not None else None
    if fam.startswith("GB_"):
        m = HistGradientBoostingClassifier(**cfg, monotonic_cst=MONO if fam == "GB_mono" else None, random_state=semente).fit(ztr, y, sample_weight=w)
        return ("sk", m, prep), {}
    if fam == "GBR_residual":
        m = GradientBoostingClassifier(**cfg, init=QInit(), random_state=semente)
        m.fit(np.c_[ztr, Q.escore(x.iloc[idx_tr])], y, sample_weight=w)
        return ("gbr", m, prep), {}
    q = Q.escore(x.iloc[idx_tr]) if fam == "MLP_res" else None
    qv = Q.escore(x.iloc[idx_val]) if (fam == "MLP_res" and idx_val is not None) else None
    rede, info = treinar_torch("MLP" if fam.startswith("MLP") else "FTT", cfg, ztr, y, w, q, semente, epocas, zva, yva, qv)
    return ("torch", rede, prep, fam == "MLP_res"), info


def prever(mod, x, Q):
    tipo, m, prep = mod[0], mod[1], mod[2]
    z = prep(x)
    if tipo == "sk":
        p = m.predict_proba(z)[:, 1]
        return np.log(p + 1e-9) - np.log(1 - p + 1e-9)
    if tipo == "gbr":
        return m.decision_function(np.c_[z, Q.escore(x)])
    return prever_torch(m, z, Q.escore(x) if mod[3] else None)


def main():
    from v8_04_campanha import caracteristicas, coorte, externo
    import mt01_experimento as e1
    prot = congelar9(D / "protocolo.json", {"versao": "v9-b-1", "atributos": COLS, "monotonia": MONO, "familias": list(configuracoes()),
                                            "n_configs": {k: len(v) for k, v in configuracoes().items()}, "halving": "1 dobra -> metade em 3 dobras -> 2 finalistas x 3 sementes",
                                            "selecao": "AUC dentro do evento media por unidade, validacao agrupada por unidade fora_BR",
                                            "tabpfn": "nao usado: licenca exige aceite; regime publicado <= 10 mil exemplos"})
    t0 = time.time()
    tr, x, _ = externo()
    with (V9 / "p0_regularizacao/modelos.pkl").open("rb") as f:
        Q = pickle.load(f)["Q_sel_ext"]
    unid = np.array(sorted(tr.unidade.unique()))
    dobra = dict(zip(unid, np.random.default_rng(20261023).permutation(len(unid)) % 3))
    fd = tr.unidade.map(dobra).to_numpy()
    nota_q = np.mean([e1.auc_media_por_unidade(tr[fd == k], Q.escore(x[fd == k])) for k in range(3)])

    # Q_sel_ext foi ajustado em todo fora_BR: nota_q e otimista (viu as dobras); serve so de escala
    def nota(fam, cfg, dobras, sem=0):
        v = []
        for k in dobras:
            itr, iva = np.where(fd != k)[0], np.where(fd == k)[0]
            mod, info = ajustar_familia(fam, cfg, tr, x, Q, itr, iva, sem)
            v.append(e1.auc_media_por_unidade(tr.iloc[iva], prever(mod, x.iloc[iva], Q)))
        return float(np.mean(v))
    busca = []
    final = {}
    for fam, cfgs in configuracoes().items():
        e1_ = [(i, nota(fam, c, [0])) for i, c in enumerate(cfgs)]
        top = sorted(e1_, key=lambda t: -t[1])[:max(2, len(cfgs) // 2)]
        e2_ = [(i, nota(fam, cfgs[i], [0, 1, 2])) for i, _ in top]
        fin = sorted(e2_, key=lambda t: -t[1])[:2]
        e3_ = [(i, np.mean([nota(fam, cfgs[i], [0, 1, 2], s) for s in (1, 2, 3)])) for i, _ in fin]
        best = max(e3_, key=lambda t: t[1])
        for i, a in e1_:
            busca.append({"familia": fam, "config": json.dumps(cfgs[i]), "estagio1": a, "estagio2": dict(e2_).get(i), "estagio3_3sementes": dict(e3_).get(i)})
        final[fam] = {"config": cfgs[best[0]], "nota": best[1]}
        g9(D / "busca.csv", pd.DataFrame(busca))
        etapa("b_tabular_familia", familia=fam, nota=best[1], segundos=time.time() - t0)
        print(fam, best, round(time.time() - t0), flush=True)
    # ajuste final em todo fora_BR, 3 sementes; epocas = media das epocas de parada na validacao
    modelos = {}
    tudo = np.arange(len(tr))
    for fam, r in final.items():
        for s in (1, 2, 3):
            if fam.startswith(("MLP", "FTT")):
                itr, iva = np.where(fd != 0)[0], np.where(fd == 0)[0]
                _, info = ajustar_familia(fam, r["config"], tr, x, Q, itr, iva, s)
                ep = info["epocas"]
            else:
                ep = None
            modelos[f"{fam}|s{s}"], _ = ajustar_familia(fam, r["config"], tr, x, Q, tudo, None, s, epocas=ep)
    import torch
    for k, mod in modelos.items():
        if mod[0] == "torch":
            torch.save(mod[1].state_dict(), D / f"pesos_{k.replace('|', '_')}.pt")
    with (D / "modelos.pkl").open("wb") as f:
        pickle.dump({k: (m[0], m[1] if m[0] != "torch" else None, m[2], m[3] if len(m) > 3 else None) for k, m in modelos.items()}, f)
    linhas = []
    for c in COORTES:
        d = coorte(c)
        xx = caracteristicas(d)
        te = (d.particao.eq("teste") & d.estado_primaria.isin(["positivo", "negativo"])).to_numpy()
        dt, xt = d[te].reset_index(drop=True), xx[te].reset_index(drop=True)
        p = dt[["ponto_id", "grupo", "classe_territorial"]].assign(y=dt.estado_primaria.eq("positivo").astype(int))
        p["Q_sel_ext"] = Q.escore(xt)
        p["A0_hand"] = -dt.hand_m.to_numpy()
        p["menos_declive"] = -dt.slope_deg.to_numpy()
        for fam in final:
            ss = [prever(modelos[f"{fam}|s{s}"], xt, Q) for s in (1, 2, 3)]
            p[fam] = np.mean(ss, axis=0)
            for s, v in zip((1, 2, 3), ss):
                p[f"{fam}|s{s}"] = v
        p["ENS_tab"] = np.mean([p[f].rank(pct=True) for f in final], axis=0)
        g9(D / f"predicoes_{c}.parquet", p)
        for pop in ["todas", "fcu"]:
            q = p if pop == "todas" else p[p.classe_territorial.eq(pop)]
            linhas.append({"coorte": c, "populacao": pop, "positivos": int(q.y.sum()),
                           **{k: auc(q.y, q[k]) for k in ["A0_hand", "menos_declive", "Q_sel_ext", *final, "ENS_tab"]},
                           **{f"{fam}_dp_sementes": float(np.std([auc(q.y, q[f"{fam}|s{s}"]) for s in (1, 2, 3)])) for fam in final}})
        print(c, flush=True)
    g9(D / "auc_coortes.csv", pd.DataFrame(linhas))
    g9(D / "resultado.json", {"quando_utc": agora(), "segundos": time.time() - t0, "finalistas": final, "nota_Q_mesmas_dobras_otimista": float(nota_q),
                              "protocolo_quando": prot["quando_utc"]})
    etapa("b_tabular", segundos=time.time() - t0)
    print(pd.DataFrame(linhas).round(3).to_string())


if __name__ == "__main__":
    main()
