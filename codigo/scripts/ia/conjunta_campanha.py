"""Campanha do aprendizado conjunto v1: o compartilhamento muda a inferencia fisica fora do ajuste?

Linhagem `aprendizado_conjunto_v1` (16/10/2026). Desenvolvimento retrospectivo: os 43
componentes e os 10 eventos SAR ja foram examinados no projeto; nada aqui e reserva nova.

Retencao: componente de dependencia INTEIRO, nas duas tarefas ao mesmo tempo (os pontos
Sen1Floods11 sairam dos mesmos rotulos manuais dos chips; um componente no teste nao pode
ter chip nem ponto no treino). Oito dobras por componente + duas dobras por fonte (CEMS e
UFO: toda a fonte e todo componente que a toca). Retencao da fonte Sen1Floods11 deixa a
observacao sem nenhum chip de treino: compartilhamento `not_testable` ali, por desenho.

Nada e escolhido no destino: passos fixos (sem selecao de checkpoint), pesos das perdas,
penalidades e taxas fixados no protocolo; padronizacao nos pontos de treino da dobra.

Uso:
    python scripts/ia/conjunta_campanha.py protocolo
    python scripts/ia/conjunta_campanha.py executar [--fumaca] [--mesmo-na-bateria]
    python scripts/ia/conjunta_campanha.py final
    python scripts/ia/conjunta_campanha.py analisar
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_n, "4")

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
for pasta in ("ia", "metodo", "sar", "neural", "servico", "manutencao"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
import conjunta_base as cb  # noqa: E402
import conjunta_modelo as cm  # noqa: E402
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402
from proveniencia_de_peso import hash_completo_do_estado  # noqa: E402

CAMP = cb.SAIDA / "campanha"
FUMACA = cb.SAIDA / "fumaca"
TODOS = list(range(len(cb.TERRENO)))
SEM_ACUM = [i for i, k in enumerate(cb.TERRENO) if k != "f_hand"]
SO_HAND = [cb.TERRENO.index("f_hand")]
CONFIG = {
    "sementes": [7, 11, 23], "passos": 500, "batch": 16, "recorte": 64,
    "lr_fisica": 3e-3, "lr_observacao": 1e-3, "lambda_observacao": 1.0, "l2_fisica": 1e-4,
    "dim": 8, "oculto": 16, "base": 4, "threads": 4, "checkpoint_cada": 100, "diagnostico_cada": 50,
    "dobras": 8, "semente_dobras": 20261016, "teto_treino": 400, "teto_teste": 2000,
    "semente_amostra": 20261010, "minimo_teste_evento": 20,
    "lam_Q": {"p": 1e-4, "i": 100.0}, "margem_auc": 0.01, "margem_ap": 0.025, "bootstrap": 2000,
    "fontes_retidas": ["cems", "ufo"],
    # niveis de orcamento (destinos, bracos, sementes), executados nesta ordem; nada depende de resultado
    "niveis": [
        ["componentes", ["fisica_so", "separado", "parada", "conjunto"], [7, 11, 23]],
        ["componentes", ["fisica_so_sem_acumulacao", "fisica_so_sem_conexao"], [7, 11, 23]],
        ["componentes", ["conjunto_rompido", "conjunto_sem_sar", "conjunto_sem_acumulacao", "conjunto_sem_conexao"], [7, 11]],
        ["fontes", ["fisica_so", "separado", "parada", "conjunto"], [7, 11, 23]],
    ],
    "workers": 2,
}
BRACOS_NEURAIS = {
    "separado": {"modo": "separado", "terreno": TODOS},
    "parada": {"modo": "parada", "terreno": TODOS},
    "conjunto": {"modo": "conjunto", "terreno": TODOS},
    "conjunto_sem_sar": {"modo": "conjunto", "terreno": TODOS, "usa_sar": False},
    "conjunto_rompido": {"modo": "conjunto", "terreno": TODOS, "rompido": True},
    "conjunto_sem_acumulacao": {"modo": "conjunto", "terreno": SEM_ACUM},
    "conjunto_sem_conexao": {"modo": "conjunto", "terreno": SO_HAND},
}
BRACOS_FISICOS = {
    "fisica_so": {"modo": "separado", "terreno": TODOS, "so_fisica": True},
    "fisica_so_sem_acumulacao": {"modo": "separado", "terreno": SEM_ACUM, "so_fisica": True},
    "fisica_so_sem_conexao": {"modo": "separado", "terreno": SO_HAND, "so_fisica": True},
}
BRACOS = {**BRACOS_FISICOS, **BRACOS_NEURAIS}


def escrever(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".part")
    t.write_text(json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False, default=str), encoding="utf-8")
    t.replace(p)


def ler(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ===================================================================== dados

class Dados:
    def __init__(self, base: Path = cb.BASE):
        self.pts = pd.read_parquet(base / "pontos.parquet")
        self.chips = pd.read_parquet(base / "chips.parquet")
        self.ter = np.load(base / "chips_terreno.npy", mmap_mode="r")
        self.sar = np.load(base / "chips_sar.npy", mmap_mode="r")
        self.alvo = np.load(base / "chips_alvo.npy", mmap_mode="r")
        self.base = base

    def chips_ok(self):
        return self.chips[self.chips.papel == "supervisao_observacao"]


def dobras(d: Dados, cfg=CONFIG) -> dict:
    comps = sorted(d.pts.unidade.unique())
    com_sar = sorted(set(d.chips_ok().componente))
    rng = np.random.default_rng(cfg["semente_dobras"])
    ordem = list(rng.permutation(com_sar)) + list(rng.permutation([c for c in comps if c not in com_sar]))
    return {str(c): i % cfg["dobras"] for i, c in enumerate(ordem)}


def amostrar(d: pd.DataFrame, teto: int, semente: int) -> pd.DataFrame:
    """Copia de `mt01_experimento.amostrar` (mesma ordem de sorteio): mesmos pontos do MT05."""
    rng = np.random.default_rng(semente)
    partes = []
    for _, g in d.groupby(["evento_id", "classe"], sort=True):
        partes.append(g if len(g) <= teto else g.iloc[np.sort(rng.choice(len(g), teto, replace=False))])
    return pd.concat(partes).sort_index()


def destinos(d: Dados, cfg=CONFIG) -> list[str]:
    return [f"dobra_{k}" for k in range(cfg["dobras"])] + [f"fonte_{f}" for f in cfg["fontes_retidas"]]


def particao(d: Dados, destino: str, cfg=CONFIG) -> dict:
    p = d.pts
    if destino.startswith("dobra_"):
        k = int(destino.split("_")[1])
        f = dobras(d, cfg)
        te_un = sorted(u for u, v in f.items() if v == k)
        tr0, te0 = p[~p.unidade.isin(te_un)], p[p.unidade.isin(te_un)]
        te_parts = [amostrar(te0[te0.unidade == u], cfg["teto_teste"], cfg["semente_amostra"]) for u in te_un]
        te = pd.concat(te_parts)
    else:
        fonte = destino.split("_", 1)[1]
        te0 = p[p.fonte == fonte]
        te_un = sorted(te0.unidade.unique())
        tr0 = p[~p.unidade.isin(te_un)]
        te = amostrar(te0, cfg["teto_teste"], cfg["semente_amostra"])
    te = te.groupby("evento_id").filter(lambda g: len(g) >= cfg["minimo_teste_evento"] and g.classe.nunique() == 2)
    tr = amostrar(tr0, cfg["teto_treino"], cfg["semente_amostra"])
    tr_un = sorted(tr.unidade.unique())
    if set(tr_un) & set(te0.unidade.unique()) or set(tr.evento_id) & set(te.evento_id):
        raise ValueError("componente ou evento compartilhado entre treino e destino")
    ck = d.chips_ok()
    ch_tr = ck[ck.componente.isin(tr_un)]
    ch_te = ck[ck.componente.isin(te_un)]
    if set(ch_tr.evento_mt) & set(ch_te.evento_mt):
        raise ValueError("evento SAR dos dois lados")
    for _, a in ch_tr.iterrows():  # nenhum chip de treino toca chip ou ponto de destino
        x0, y0, x1, y1 = a.bounds
        for _, b in ch_te.iterrows():
            u0, v0, u1, v1 = b.bounds
            if min(x1, u1) > max(x0, u0) and min(y1, v1) > max(y0, v0):
                raise ValueError(f"chips sobrepostos entre treino e destino: {a.nome} {b.nome}")
        dentro = (te.lon >= x0) & (te.lon <= x1) & (te.lat >= y0) & (te.lat <= y1)
        if dentro.any():
            raise ValueError(f"ponto de destino dentro de chip de treino: {a.nome}")
    return {"destino": destino, "tr": tr, "te": te, "unidades_treino": tr_un, "unidades_teste": te_un,
            "chips_treino": ch_tr.indice.astype(int).tolist(), "chips_teste": ch_te.indice.astype(int).tolist()}


def tensores_fisicos(df: pd.DataFrame):
    t = torch.tensor(df[list(cb.TERRENO)].to_numpy(np.float32))
    c = torch.tensor(df[list(cb.COBERTURA)].to_numpy(np.float32))
    return t, c


def derangement(ids: list[int], eventos: list[str], semente: int) -> dict:
    """Troca o terreno entre chips do MESMO evento (fallback: entre eventos), sem ponto fixo."""
    rng = np.random.default_rng(semente)
    mapa = {}
    grupos = {}
    for i, e in zip(ids, eventos):
        grupos.setdefault(e, []).append(i)
    sobra = []
    for e, g in sorted(grupos.items()):
        if len(g) < 2:
            sobra += g
            continue
        p = list(rng.permutation(g))
        mapa.update({a: b for a, b in zip(p, p[1:] + p[:1])})
    if len(sobra) == 1:
        a = sobra[0]
        b = next(iter(mapa))
        mapa[a], mapa[b] = mapa[b], a
    elif sobra:
        mapa.update({a: b for a, b in zip(sobra, sobra[1:] + sobra[:1])})
    if any(a == b for a, b in mapa.items()) or sorted(mapa) != sorted(ids) or sorted(mapa.values()) != sorted(ids):
        raise ValueError("permutacao invalida")
    return mapa


class Recortes:
    def __init__(self, d: Dados, chips: list[int], mapa_terreno: dict | None, rng: np.random.Generator, cfg=CONFIG):
        self.d, self.chips, self.mapa, self.rng, self.t = d, chips, mapa_terreno, rng, cfg["recorte"]
        self.batch = cfg["batch"]

    def lote(self):
        xs, xt, ys = [], [], []
        for _ in range(self.batch):
            for _ in range(200):
                i = self.chips[int(self.rng.integers(len(self.chips)))]
                a, b = (int(self.rng.integers(512 - self.t + 1)) for _ in range(2))
                y = np.asarray(self.d.alvo[i, a:a + self.t, b:b + self.t], np.int64)
                if (y != -1).any():
                    break
            else:
                raise ValueError("nenhum recorte rotulado")
            j = self.mapa[i] if self.mapa else i
            s = np.asarray(self.d.sar[i, :, a:a + self.t, b:b + self.t], np.float32)
            t = np.asarray(self.d.ter[j, :, a:a + self.t, b:b + self.t], np.float32)
            if self.rng.random() < .5:
                s, t, y = s[:, :, ::-1], t[:, :, ::-1], y[:, ::-1]
            if self.rng.random() < .5:
                s, t, y = s[:, ::-1], t[:, ::-1], y[::-1]
            xs.append(s.copy()); xt.append(t.copy()); ys.append(y.copy())
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(xt)), torch.from_numpy(np.stack(ys))


# ===================================================================== treino

def construir(braco: str, semente: int, cfg=CONFIG) -> cm.ModeloConjunto:
    b = BRACOS[braco]
    return cm.ModeloConjunto(b["modo"], b["terreno"], len(cb.TERRENO), len(cb.COBERTURA), dim=cfg["dim"],
                             oculto=cfg["oculto"], base=cfg["base"], usa_sar=b.get("usa_sar", True), semente=semente)


def _grad(perda, params):
    g = torch.autograd.grad(perda, params, retain_graph=True, allow_unused=True)
    return torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(g, params)])


def _estado_rng(rng):
    return rng.bit_generator.state


def treinar(pasta: Path, braco: str, semente: int, part: dict, d: Dados, cfg=CONFIG, parar_em: int | None = None):
    """Treina com checkpoint completo (modelo, intercepto, otimizador, geradores) e retoma do ultimo."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    final = pasta / "final.pt"
    if final.exists() and (pasta / "treino.json").exists():
        e = torch.load(final, weights_only=False, map_location="cpu")
        m = construir(braco, semente, cfg)
        m.load_state_dict(e["modelo"])
        if hash_completo_do_estado(e["modelo"]) != ler(pasta / "treino.json")["hash_estado"]:
            raise ValueError("peso final divergente do registro")
        return m, ler(pasta / "treino.json")
    b = BRACOS[braco]
    so_fisica = b.get("so_fisica", False)
    tr = part["tr"]
    T, C = tensores_fisicos(tr)
    y = torch.tensor(tr.classe.to_numpy(np.float32))
    ev_ids, ev = np.unique(tr.evento_id.to_numpy(), return_inverse=True)
    ev = torch.tensor(ev, dtype=torch.long)
    m = construir(braco, semente, cfg)
    m.pad_terreno.ajustar(T)
    m.pad_cobertura.ajustar(C)
    intercepto = nn.Parameter(torch.zeros(len(ev_ids)))
    g = m.grupos()
    opt = torch.optim.Adam([{"params": g["fisica"] + [intercepto], "lr": cfg["lr_fisica"]},
                            {"params": g["encoder_obs"], "lr": cfg["lr_fisica"]},
                            {"params": g["observacao"], "lr": cfg["lr_observacao"]}])
    rng = np.random.default_rng([semente, 5150])
    mapa = None
    if b.get("rompido"):
        ch = d.chips.set_index("indice")
        mapa = derangement(part["chips_treino"], [ch.loc[i, "evento_mt"] for i in part["chips_treino"]], semente + 99)
    rec = None if so_fisica else Recortes(d, part["chips_treino"], mapa, rng, cfg)
    hist, diag, inicio, t0 = [], [], 0, time.monotonic()
    seg_previos = 0.
    ck = pasta / "checkpoint.pt"
    if ck.exists():
        e = torch.load(ck, weights_only=False, map_location="cpu")
        m.load_state_dict(e["modelo"])
        with torch.no_grad():
            intercepto.copy_(e["intercepto"])
        opt.load_state_dict(e["otimizador"])
        rng.bit_generator.state = e["rng_numpy"]
        torch.set_rng_state(e["rng_torch"])
        hist, diag, inicio, seg_previos = e["historico"], e["diagnostico"], e["passo"], e["segundos"]
    compart = set(m.parametros_compartilhados())
    enc_params = list(m.enc_fis.parameters())
    passos = cfg["passos"]
    for passo in range(inicio, passos):
        if parar_em is not None and passo >= parar_em:
            break
        m.train()
        opt.zero_grad(set_to_none=True)
        lf = cm.perda_fisica(m.fisica(T, C), y, ev, intercepto) + cm.l2_fisica(m, cfg["l2_fisica"])
        lo = torch.zeros(())
        if not so_fisica:
            xs, xt, yo = rec.lote()
            lo = nn.functional.cross_entropy(m.observacao(xs if m.usa_sar else None, xt), yo, ignore_index=-1)
        total = lf + cfg["lambda_observacao"] * lo
        if passo % cfg["diagnostico_cada"] == 0 or passo == passos - 1:
            gf = _grad(lf, enc_params)
            go = _grad(cfg["lambda_observacao"] * lo, enc_params) if lo.requires_grad else torch.zeros_like(gf)
            nf, no = float(gf.norm()), float(go.norm())
            diag.append({"passo": passo, "norma_grad_fisica_enc": nf, "norma_grad_obs_enc": no,
                         "cosseno": float(gf @ go / (nf * no)) if nf > 0 and no > 0 else None})
        total.backward()
        opt.step()
        hist.append({"passo": passo, "perda_fisica": float(lf.detach()), "perda_obs": float(lo.detach())})
        if (passo + 1) % cfg["checkpoint_cada"] == 0 and passo + 1 < passos:
            tmp = ck.with_suffix(".part")
            torch.save({"modelo": m.state_dict(), "intercepto": intercepto.detach().clone(),
                        "otimizador": opt.state_dict(), "rng_numpy": _estado_rng(rng),
                        "rng_torch": torch.get_rng_state(), "historico": hist, "diagnostico": diag,
                        "passo": passo + 1, "segundos": seg_previos + time.monotonic() - t0}, tmp)
            tmp.replace(ck)
    if len(hist) < passos:
        return m, None  # interrompido de proposito (teste de retomada)
    estado = {k: v.detach().clone() for k, v in m.state_dict().items()}
    torch.save({"modelo": estado, "intercepto": intercepto.detach().clone(), "eventos_treino": list(ev_ids),
                "otimizador": opt.state_dict()}, final)
    rec_fis = [n for n, _ in m.named_parameters() if n.startswith(("enc_fis.", "cab_fis."))]
    t = {"braco": braco, "semente": semente, "destino": part["destino"], "hash_estado": hash_completo_do_estado(estado),
         "passos": passos, "segundos": seg_previos + time.monotonic() - t0,
         "parametros_total": sum(p.numel() for p in m.parameters()),
         "parametros_encoder_fisico": sum(p.numel() for p in m.enc_fis.parameters()),
         "parametros_observacao": sum(p.numel() for p in m.obs.parameters()) if not so_fisica else 0,
         "parametros_encoder_obs": sum(p.numel() for p in m.enc_obs.parameters()) if m.enc_obs is not None else 0,
         "parametros_compartilhados_pelas_duas_perdas": sorted(compart),
         "parametros_so_fisica": sorted(set(rec_fis) - compart),
         "observacao_ativa": not so_fisica, "pontos_treino": len(tr), "eventos_treino": len(ev_ids),
         "chips_treino": len(part["chips_treino"]), "historico": hist, "diagnostico_gradiente": diag,
         "memoria_rss_mb": _rss_mb(), "retomado_de_checkpoint": inicio > 0}
    escrever(pasta / "treino.json", t)
    if ck.exists():
        ck.unlink()
    return m, t


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:  # noqa: BLE001
        return None


# ===================================================================== avaliacao

def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


@torch.no_grad()
def avaliar(pasta: Path, m: cm.ModeloConjunto, braco: str, part: dict, d: Dados):
    m.eval()
    te = part["te"]
    T, C = tensores_fisicos(te)
    escore = m.fisica(T, C).numpy().astype(np.float64)
    pred = te[["ponto_id", "evento_id", "unidade", "classe", "fonte", "processo_ponto"]].copy()
    pred["escore"] = escore
    pred.to_parquet(pasta / "predicoes_fisicas.parquet", index=False)
    r = {"braco": braco, "predicoes_sha256": sha256_arquivo(pasta / "predicoes_fisicas.parquet")}
    if BRACOS[braco].get("so_fisica"):
        escrever(pasta / "avaliacao.json", r)
        return r
    ch = d.chips.set_index("indice")
    probs, ev_hist, chips_r, densa = {}, {}, [], {}
    for i in part["chips_teste"]:
        s = torch.from_numpy(np.asarray(d.sar[i], np.float32))[None]
        t = torch.from_numpy(np.asarray(d.ter[i], np.float32))[None]
        p = m.observacao(s if m.usa_sar else None, t).softmax(1)[0, 1].numpy()
        y = np.asarray(d.alvo[i], np.int64)
        q = np.round(p * 255).astype(np.uint8)
        probs[ch.loc[i, "nome"]] = q
        e = ch.loc[i, "evento_mt"]
        h = ev_hist.setdefault(e, {"positivo": np.zeros(1000, np.int64), "negativo": np.zeros(1000, np.int64),
                                   "tp": 0, "fp": 0, "fn": 0, "tn": 0})
        idx = np.minimum((p * 1000).astype(int), 999)
        h["positivo"] += np.bincount(idx[y == 1], minlength=1000)
        h["negativo"] += np.bincount(idx[y == 0], minlength=1000)
        pr = p >= .5
        for k, v in (("tp", pr & (y == 1)), ("fp", pr & (y == 0)), ("fn", ~pr & (y == 1)), ("tn", ~pr & (y == 0))):
            h[k] += int(v.sum())
        # fisica densa no chip: mesmo encoder, cobertura ausente (vira a mediana do treino) -- declarado
        tt = t[0].reshape(len(cb.TERRENO), -1).T
        cc = torch.full((tt.shape[0], len(cb.COBERTURA)), float("nan"))
        sf = m.fisica(tt, cc).numpy().reshape(512, 512)
        lab = y != -1
        dd = densa.setdefault(e, {"s": [], "y": []})
        dd["s"].append(sf[lab]); dd["y"].append(y[lab])
        chips_r.append({"nome": ch.loc[i, "nome"], "evento": e, "positivos": int((y == 1).sum()),
                        "negativos": int((y == 0).sum()), "tp": int((pr & (y == 1)).sum()), "fp": int((pr & (y == 0)).sum())})
    np.savez_compressed(pasta / "obs_prob_uint8.npz", **probs)
    eventos = {}
    for e, h in ev_hist.items():
        pos, neg = h["positivo"], h["negativo"]
        ys = np.concatenate(densa[e]["y"]); ss = np.concatenate(densa[e]["s"])
        eventos[e] = {"ap": ap_histograma(pos, neg), "iou": h["tp"] / max(h["tp"] + h["fp"] + h["fn"], 1),
                      "taxa_falso_positivo": h["fp"] / max(h["fp"] + h["tn"], 1),
                      "taxa_falso_negativo": h["fn"] / max(h["fn"] + h["tp"], 1),
                      "prevalencia": int(pos.sum()) / max(int(pos.sum() + neg.sum()), 1),
                      "auc_fisica_densa": _auc(ys, ss) if 0 < ys.sum() < len(ys) else None,
                      "pixels": int(len(ys)), **{k: int(h[k]) for k in ("tp", "fp", "fn", "tn")},
                      "histograma": {"positivo": pos.tolist(), "negativo": neg.tolist()}}
    r.update({"observacao_por_evento": eventos, "chips": chips_r,
              "obs_sha256": sha256_arquivo(pasta / "obs_prob_uint8.npz")})
    escrever(pasta / "avaliacao.json", r)
    return r


def ap_histograma(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """AP por limiares decrescentes de bin (mesma convencao de `calibracao_decisao.ap`)."""
    P = pos.sum()
    if P == 0:
        return None
    tp = np.cumsum(pos[::-1]); fp = np.cumsum(neg[::-1])
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / P
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))


# ===================================================================== referencia Q

def referencia_q(pasta: Path, part: dict):
    """Q_base/Q_conexao do MT05 reajustadas NA DOBRA (mesma logistica, lam fixo do protocolo)."""
    arq = pasta / "predicoes_q.parquet"
    if arq.exists():
        return pd.read_parquet(arq)
    import mt01_experimento as e1
    import mt05_transferencia_sem_amostra as mt
    tr, te = part["tr"], part["te"]
    xtr, xte = tr[list(cb.TERRENO) + list(cb.COBERTURA)], te[list(cb.TERRENO) + list(cb.COBERTURA)]
    pred = te[["ponto_id", "evento_id", "unidade", "classe", "fonte", "processo_ponto"]].copy()
    pred["A0_regra_hand"] = -te.hand_m.to_numpy()
    info = {}
    for nome in ("Q_base", "Q_conexao"):
        mod = e1.Logistica(mt.BRACOS[nome], mt.INTERACOES.get(nome, ()), lam_p=CONFIG["lam_Q"]["p"], lam_i=CONFIG["lam_Q"]["i"])
        mod.fit(xtr, tr.classe.to_numpy(float), tr.evento_id.to_numpy())
        if not mod.convergiu:
            raise ValueError("Q nao convergiu: " + nome)
        pred[nome] = mod.escore(xte)
        info[nome] = {"principais": dict(zip(mt.BRACOS[nome], mod.b.tolist())), "iteracoes": mod.iteracoes}
    pasta.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(arq, index=False)
    escrever(pasta / "q.json", {"coeficientes": info, "lam": CONFIG["lam_Q"], "sha256": sha256_arquivo(arq)})
    return pred


# ===================================================================== protocolo e execucao

def protocolo(d: Dados, cfg=CONFIG) -> dict:
    f = dobras(d, cfg)
    codigo = [Path(__file__), Path(cm.__file__), Path(cb.__file__), REPO / "scripts/sar/modelo_unet.py"]
    return {
        "versao": cb.VERSAO, "tipo": "protocolo_congelado_antes_dos_resultados",
        "regime": "desenvolvimento retrospectivo; eventos e componentes ja examinados; nenhuma reserva aberta",
        "base_manifesto_sha256": sha256_arquivo(cb.BASE / "manifesto.json"),
        "codigo_sha256": {str(p.relative_to(REPO)): sha256_arquivo(p) for p in codigo},
        "config": cfg, "bracos": {k: {**v, "terreno": [cb.TERRENO[i] for i in v["terreno"]]} for k, v in BRACOS.items()},
        "dobras": f, "destinos": destinos(d, cfg),
        "populacao": {"fisica": "pontos MT03-A (mesma amostragem MT05: 400/2000 por evento-classe, semente 20261010)",
                      "observacao": "chips Sen1Floods11 com pixel rotulado e terreno ter-01; alvo agua nao permanente"},
        "hipoteses": {
            "H1_compartilhamento": "no braco conjunto os gradientes das duas tarefas atingem os mesmos parametros do encoder (registrado por norma/cosseno) e a fisica resultante difere da parada; e mecanismo, nao ganho",
            "H2_transferencia_fisica": "conjunto - parada > 0 em AUC fisica por componente retido (ordem dentro do evento)",
            "H3_explicacao_fiel": "atribuicao do modelo final (gradientes integrados) soma ao escore menos a referencia; fiel ao estimador, nao causal",
            "H4_causal": "nenhuma relacao causal e testada pela campanha; ver grafo e lacuna de identificacao",
        },
        "primarias": {
            "P1": {"contraste": "conjunto - parada", "metrica": "AUC fisica media por componente (recorte todos, dobras_*)",
                   "unidade": "componente (43)", "tipo": "superioridade", "margem": cfg["margem_auc"], "ic": "bootstrap de componentes 97,5% (Bonferroni k=2)"},
            "P2": {"contraste": "conjunto - separado", "metrica": "AP de observacao por evento SAR retido",
                   "unidade": "evento SAR (10)", "tipo": "nao_inferioridade", "margem": cfg["margem_ap"], "ic": "bootstrap de eventos 97,5%"},
        },
        "regra_de_decisao": {
            "superioridade": "sustentado se media >= margem e IC_inf > 0; nao_sustentado se IC_sup < margem; senao indecidivel",
            "nao_inferioridade": "nao_inferior se IC_inf > -margem; inferior se IC_sup < -margem; superior se media>=margem e IC_inf>0; senao indecidivel",
            "transferencia_negativa": "declarada se IC_sup < 0 em P1 ou em qualquer recorte",
            "sementes": ("media das sementes por unidade ANTES do bootstrap; sementes nao sao unidades; "
                         "contraste entre bracos com conjuntos de sementes diferentes usa so as sementes comuns"),
        },
        "orcamento": ("niveis em config.niveis: principais (fisica_so, separado, parada, conjunto) com 3 sementes nas 8 "
                      "dobras e nas 2 fontes; ablacoes fisicas com 3; controles neurais (rompido, sem_sar, sem_acumulacao, "
                      "sem_conexao) com 2 sementes so nas 8 dobras. Mesmos passos, lote, recorte e taxas em todos os bracos"),
        "secundarias_exploratorias": [
            "conjunto - conjunto_rompido (fisica): dependencia do alinhamento real",
            "conjunto - conjunto_sem_sar (fisica): papel da modalidade SAR no compartilhamento",
            "parada/conjunto/fisica_so - Q_conexao (fisica): contra a referencia MT05 reajustada na dobra",
            "ablacoes retreinadas: sem_acumulacao e sem_conexao nos regimes conjunto e fisica_so; interacao Delta_conj - Delta_sep",
            "observacao: parada - separado, conjunto_sem_sar - conjunto, conjunto_rompido - conjunto; IoU, falso positivo",
            "recortes processo:fluvial e processo:enxurrada; componentes com e sem SAR; dobras por fonte (cems, ufo)",
            "AUC fisica densa nos chips SAR retidos (cobertura ausente imputada)",
        ],
        "poder": "ver poder.json (escrito junto, a partir do DP das diferencas MT05; antes dos resultados)",
        "identidades_esperadas": ["fisica de separado == parada == fisica_so (bit a bit)",
                                  "Q na dobra ~ Q MT05 LOCO (descritivo; treino diferente)"],
        "custos": "segundos, parametros, memoria por execucao; capacidade: separado tem um encoder a mais (registrado)",
        "nao_e": "nao promove modelo, nao altera escore/maturidade/contratos do servico, nao abre reserva",
    }


def poder(cfg=CONFIG) -> dict:
    m = pd.read_parquet(REPO / "local_runs/mecanismos-transferiveis/v5/sem_amostra_v1/medidas.parquet")
    g = m[m.recorte == "todos"].groupby(["destino", "evento_id", "modelo"]).auc.mean().groupby(["destino", "modelo"]).mean().unstack()
    out = {}
    for a, b in (("Q_conexao", "Q_base"), ("Q_base", "A0_regra_hand")):
        dd = (g[a] - g[b]).dropna()
        sd = float(dd.std())
        n = len(dd)
        out[f"{a}-{b}"] = {"n": n, "dp_entre_componentes": sd,
                           "efeito_minimo_detectavel_80pct_alfa2,5pct": float((2.24 + 0.84) * sd / np.sqrt(n)),
                           "semi_largura_ic95": float(1.96 * sd / np.sqrt(n))}
    out["observacao_10_eventos"] = {"nota": "com 10 eventos o Wilcoxon exato chega a p=2/1024; a margem 0,025 de AP exige DP pequeno entre eventos",
                                    "dp_referencia_ablacao_sar_terreno": 0.0296,
                                    "semi_largura_ic95_esperada": float(2.26 * 0.0296 / np.sqrt(10))}
    out["leitura"] = "efeitos menores que o MDE ficam indecidiveis por desenho; isso e limite do numero de componentes, nao das sementes"
    return out


def congelar(d: Dados, saida: Path = CAMP, cfg=CONFIG) -> dict:
    p = protocolo(d, cfg)
    arq = saida / "protocolo.json"
    if arq.exists():
        antigo, novo = ler(arq), json.loads(json.dumps(p, default=str))
        if {k: v for k, v in antigo.items() if k != "codigo_sha256"} != {k: v for k, v in novo.items() if k != "codigo_sha256"}:
            raise ValueError("protocolo mudou depois de congelado; criar nova versao")
        if antigo["codigo_sha256"] != novo["codigo_sha256"]:
            # correcao de codigo sem efeito cientifico so com emenda escrita ANTES de rodar com ela
            em = ler(saida / "emendas.json") if (saida / "emendas.json").exists() else []
            if not any(e["codigo_sha256"] == novo["codigo_sha256"] for e in em):
                raise ValueError("codigo mudou depois do congelamento sem emenda registrada em emendas.json")
        return p
    escrever(arq, p)
    escrever(saida / "poder.json", poder(cfg))
    escrever(saida / "registro_inicio.json", {"quando_utc": agora_utc(), "protocolo_sha256": sha256_arquivo(arq),
                                             "resultados_lidos": False})
    return p


def na_tomada() -> bool:
    import executar_serializado as serial
    return serial.energia()["na_tomada"]


PISO_BATERIA_PCT = 25  # autorizado pela pesquisadora em 16/10/2026: na bateria, pausar abaixo de 25%


def pode_rodar(mesmo_na_bateria: bool) -> bool:
    import executar_serializado as serial
    e = serial.energia()
    if e["na_tomada"]:
        return True
    return mesmo_na_bateria and (e.get("carga_pct") or 0) >= PISO_BATERIA_PCT


def esperar_tomada(mesmo_na_bateria: bool, limite_min: float) -> bool:
    """Trabalho pesado na tomada (regra do projeto). Com bateria autorizada, roda ate o piso de
    carga e entao espera a tomada; a decisao e tomada antes de cada execucao (checkpoint a cada 100 passos)."""
    fim = time.time() + limite_min * 60
    avisou = False
    while not pode_rodar(mesmo_na_bateria):
        if time.time() >= fim:
            return False
        if not avisou:
            print(json.dumps({"esperando": "tomada", "quando_utc": agora_utc()}), flush=True)
            avisou = True
        time.sleep(60)
    return True


def plano(d: Dados, cfg=CONFIG) -> list[tuple[str, str, int]]:
    """Ordem declarada: bracos principais antes dos controles; nada depende de resultado."""
    comp = [x for x in destinos(d, cfg) if x.startswith("dobra_")]
    fontes = [x for x in destinos(d, cfg) if x.startswith("fonte_")]
    vistos, out = set(), []
    for dests, bracos, sementes in cfg["niveis"]:
        alvo = comp if dests == "componentes" else fontes
        for s in sementes:
            for x in alvo:
                for b in bracos:
                    if (x, b, s) not in vistos:
                        vistos.add((x, b, s))
                        out.append((x, b, s))
    return out


def preparar_destino(d: Dados, destino: str, saida: Path, cfg=CONFIG) -> dict:
    part = particao(d, destino, cfg)
    pd_ = saida / destino
    info = {k: part[k] for k in ("destino", "unidades_treino", "unidades_teste", "chips_treino", "chips_teste")} | {
        "pontos_treino": len(part["tr"]), "pontos_teste": len(part["te"]),
        "eventos_teste": sorted(part["te"].evento_id.unique()),
        "ids_teste_sha256": sha256_json(part["te"].ponto_id.tolist())}
    if (pd_ / "particao.json").exists():  # varios workers leem; so quem cria grava
        if ler(pd_ / "particao.json") != json.loads(json.dumps(info, default=str)):
            raise ValueError(f"particao de {destino} mudou entre execucoes")
    else:
        escrever(pd_ / "particao.json", info)
    referencia_q(pd_, part)
    return part


def executar(saida: Path = CAMP, cfg=CONFIG, mesmo_na_bateria=False, worker=0, workers=1,
             esperar_tomada_min=720., so_destinos=None):
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    d = Dados()
    congelar(d, saida, cfg)
    itens = [x for k, x in enumerate(plano(d, cfg)) if k % workers == worker]
    if so_destinos:
        itens = [x for x in itens if x[0] in so_destinos]
    partes, feitos = {}, 0
    for destino, braco, semente in itens:
        pasta = saida / destino / f"{braco}_s{semente}"
        if (pasta / "avaliacao.json").exists():
            continue
        if not esperar_tomada(mesmo_na_bateria, esperar_tomada_min):
            return {"estado": "pausado_bateria", "execucoes_nesta_chamada": feitos}
        if destino not in partes:
            partes = {destino: preparar_destino(d, destino, saida, cfg)}  # uma particao por vez na memoria
        t0 = time.monotonic()
        m, t = treinar(pasta, braco, semente, partes[destino], d, cfg)
        avaliar(pasta, m, braco, partes[destino], d)
        feitos += 1
        print(json.dumps({"worker": worker, "destino": destino, "braco": braco, "semente": semente,
                          "s": round(time.monotonic() - t0, 1), "quando_utc": agora_utc()}), flush=True)
    return {"estado": "concluido", "execucoes_nesta_chamada": feitos, "worker": worker}


def executar_paralelo(workers: int, mesmo_na_bateria=False, esperar_tomada_min=720.):
    """Um trabalho serializado com `workers` processos de threads fixas; retoma pelo que ja foi gravado."""
    import subprocess
    d = Dados()
    congelar(d, CAMP, CONFIG)
    for x in sorted({x[0] for x in plano(d)}):  # particoes e Q antes dos workers, sem corrida
        preparar_destino(d, x, CAMP)
    procs = []
    for w in range(workers):
        cmd = [sys.executable, "-X", "utf8", str(Path(__file__)), "executar", "--worker", str(w),
               "--workers", str(workers), "--esperar-tomada-min", str(esperar_tomada_min)]
        if mesmo_na_bateria:
            cmd.append("--mesmo-na-bateria")
        log = open(CAMP / f"worker_{w}.log", "a", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    codigos = []
    for p, log in procs:
        codigos.append(p.wait())
        log.close()
    return {"workers": workers, "codigos": codigos}


def fumaca():
    """Cadeia pequena: 1 dobra, 1 semente, todos os bracos, 30 passos. Nao e resultado."""
    cfg = {**CONFIG, "passos": 30, "checkpoint_cada": 10, "diagnostico_cada": 10,
           "niveis": [["componentes", list(BRACOS), [7]]]}
    return executar(FUMACA, cfg, mesmo_na_bateria=True, so_destinos=["dobra_0"])


# ===================================================================== modelo final persistente

FINAL = cb.SAIDA / "final"


def treinar_final(bracos=("conjunto", "parada"), semente=7, cfg=CONFIG):
    """Ajuste em TODOS os componentes: pesos persistentes para inferencia, sem avaliacao (nao ha retencao)."""
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    d = Dados()
    tr = amostrar(d.pts, cfg["teto_treino"], cfg["semente_amostra"])
    part = {"destino": "todos_os_componentes", "tr": tr, "chips_treino": d.chips_ok().indice.astype(int).tolist()}
    saida = {}
    for b in bracos:
        _, t = treinar(FINAL / b, b, semente, part, d, cfg)
        saida[b] = {"hash_estado": t["hash_estado"], "segundos": t["segundos"]}
    escrever(FINAL / "manifesto.json", {
        "versao": cb.VERSAO, "tipo": "pesos_persistentes_sem_avaliacao", "quando_utc": agora_utc(), "bracos": saida,
        "treino": {"pontos": len(tr), "componentes": int(tr.unidade.nunique()), "eventos": int(tr.evento_id.nunique()),
                   "chips": len(part["chips_treino"])},
        "nao_e": "nao e modelo avaliado: o desempenho pertence aos pesos das dobras; nao promovido; nao altera servico",
        "entradas_consulta_fisica": list(cb.TERRENO) + list(cb.COBERTURA),
        "entradas_proibidas_na_consulta_fisica": ["SAR", "mascara de inundacao", "rotulo", "elevacao relativa a amostra"]})
    return saida


@torch.no_grad()
def inferir_fisica(pasta: Path, terreno: np.ndarray, cobertura: np.ndarray, braco="conjunto", semente=7) -> np.ndarray:
    """Consulta fisica reproduzivel: carrega o estado salvo, confere o hash e devolve a ordem relativa."""
    e = torch.load(Path(pasta) / "final.pt", weights_only=False, map_location="cpu")
    t = ler(Path(pasta) / "treino.json")
    if hash_completo_do_estado(e["modelo"]) != t["hash_estado"]:
        raise ValueError("estado salvo diverge do registro")
    m = construir(braco, semente)
    m.load_state_dict(e["modelo"])
    m.eval()
    return m.fisica(torch.tensor(terreno, dtype=torch.float32), torch.tensor(cobertura, dtype=torch.float32)).numpy()


def explicar(braco="conjunto", n=4000, passos_ig=64, semente=7) -> dict:
    """Gradientes integrados do escore fisico (referencia = mediana do treino). Fiel ao estimador: a soma
    das atribuicoes reproduz escore - escore(referencia). Nao e efeito causal de atributo."""
    d = Dados()
    e = torch.load(FINAL / braco / "final.pt", weights_only=False, map_location="cpu")
    m = construir(braco, semente)
    m.load_state_dict(e["modelo"])
    m.eval()
    rng = np.random.default_rng(semente)
    amostra = d.pts.iloc[np.sort(rng.choice(len(d.pts), n, replace=False))]
    T, C = tensores_fisicos(amostra)
    T = torch.where(torch.isnan(T), m.pad_terreno.mediana, T)
    C = torch.where(torch.isnan(C), m.pad_cobertura.mediana, C)
    T0, C0 = m.pad_terreno.mediana.expand_as(T), m.pad_cobertura.mediana.expand_as(C)
    tot_t, tot_c = torch.zeros_like(T), torch.zeros_like(C)
    for k in range(1, passos_ig + 1):
        a = (k - .5) / passos_ig
        tt = (T0 + a * (T - T0)).requires_grad_(True)
        cv = (C0 + a * (C - C0)).requires_grad_(True)
        gt, gcv = torch.autograd.grad(m.fisica(tt, cv).sum(), [tt, cv])
        tot_t += gt
        tot_c += gcv
    at = (tot_t / passos_ig * (T - T0)).detach()
    ac = (tot_c / passos_ig * (C - C0)).detach()
    with torch.no_grad():
        delta = m.fisica(T, C) - m.fisica(T0, C0)
    erro = (at.sum(1) + ac.sum(1) - delta).abs()
    nomes = list(cb.TERRENO) + list(cb.COBERTURA)
    attr = torch.cat([at, ac], 1).numpy()
    amostra = amostra.assign(**{f"ig_{k}": attr[:, i] for i, k in enumerate(nomes)})
    proc = {}
    for chave, g in (("todos", amostra), ("fluvial", amostra[amostra.processo_ponto == "fluvial"]),
                     ("enxurrada", amostra[amostra.processo_ponto == "enxurrada"])):
        if len(g):
            proc[chave] = {"n": len(g), "media_abs": {k: float(g[f"ig_{k}"].abs().mean()) for k in nomes},
                           "media_com_sinal": {k: float(g[f"ig_{k}"].mean()) for k in nomes},
                           "correlacao_com_atributo": {k: float(np.corrcoef(g[k].fillna(g[k].median()), g[f"ig_{k}"])[0, 1])
                                                       if g[k].nunique() > 1 else None for k in nomes}}
    r = {"braco": braco, "pontos": n, "passos_ig": passos_ig, "referencia": "mediana dos pontos de treino",
         "erro_completude_max": float(erro.max()), "erro_completude_medio": float(erro.mean()), "por_processo": proc,
         "leitura": "atribuicao do estimador treinado; nao e efeito causal nem intervencao no terreno",
         "linear_pela_cabeca": "a cabeca fisica e linear em [h, cobertura]; a contribuicao de cada atributo de terreno passa pelo encoder nao linear"}
    escrever(FINAL / f"explicacao_{braco}.json", r)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["protocolo", "executar", "paralelo", "fumaca", "final", "explicar", "tudo"])
    ap.add_argument("--mesmo-na-bateria", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--esperar-tomada-min", type=float, default=720.)
    a = ap.parse_args()
    if a.acao == "protocolo":
        print(json.dumps(congelar(Dados())["dobras"], indent=1))
    elif a.acao == "fumaca":
        print(fumaca())
    elif a.acao == "paralelo":
        print(executar_paralelo(a.workers, a.mesmo_na_bateria, a.esperar_tomada_min))
    elif a.acao == "tudo":
        r = executar_paralelo(a.workers, a.mesmo_na_bateria, a.esperar_tomada_min)
        print(json.dumps(r), flush=True)
        if any(r["codigos"]) or len(list(CAMP.glob("*/*_s*/avaliacao.json"))) < len(plano(Dados())):
            sys.exit("campanha incompleta; modelo final nao treinado")
        if not esperar_tomada(a.mesmo_na_bateria, a.esperar_tomada_min):
            sys.exit("sem tomada para o modelo final")
        print(json.dumps(treinar_final()), flush=True)
        print(json.dumps({b: explicar(b)["erro_completude_max"] for b in ("conjunto", "parada")}), flush=True)
    elif a.acao == "final":
        print(treinar_final())
    elif a.acao == "explicar":
        print(json.dumps({b: explicar(b) for b in ("conjunto", "parada")}, indent=1)[:4000])
    else:
        print(executar(mesmo_na_bateria=a.mesmo_na_bateria, worker=a.worker, workers=a.workers,
                       esperar_tomada_min=a.esperar_tomada_min))
