"""Campanha do fechamento: observar estados de agua no tempo melhora a fisica transferivel?

Linhagem `fechamento_aprendizado_transferencia_v1` (17/10/2026). Desenvolvimento retrospectivo:
43 componentes e 26 componentes CEMS da pilha ja examinados; nenhuma reserva aberta.

Retencao igual a v1 (8 dobras por componente inteiro, nas duas tarefas) + fonte UFO. Retencao de
CEMS e `not_testable`: toda a supervisao temporal vem de AOIs CEMS.

Alvo temporal por dobra (instrumental, nivel 3): deteccao = VV <= limiar_dobra E VV <= P75 do proprio
pixel nas datas fora ANTERIORES - 3 dB (minimo 10; senao recusa). Limiar = `pilha.calibrar_limiar`
nos componentes de TREINO. Componente fornece supervisao se, com esse limiar, a deteccao em terra
JRC=0 nas datas fora for <= 0,10 e a agua sazonal JRC 10-49 ficar >= 0,05 acima (constantes da pilha,
congeladas; JRC e instrumento, nao rotulo de inundacao). Destino usa a mesma regra so para definir a
populacao com supervisao admissivel; seus dados nunca entram no ajuste.

Lambda da perda temporal: {0,3; 1; 3} escolhido por dobra em 3 dobras internas de componentes do
TREINO (braco conjunto, semente 7), pela AUC fisica interna. Passos, lote e taxas fixos.

Uso:
    python scripts/ia/fechamento_campanha.py protocolo
    python scripts/manutencao/executar_serializado.py -- python scripts/ia/fechamento_campanha.py paralelo --workers 2
    python scripts/ia/fechamento_campanha.py final
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_n, "3")

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
for s in ("ia", "sar", "servico", "metodo", "neural", "manutencao"):
    sys.path.insert(0, str(REPO / "scripts" / s))
import conjunta_base as cb  # noqa: E402
import conjunta_campanha as cc  # noqa: E402
import conjunta_modelo as cm  # noqa: E402
import fechamento_diagnostico as fd  # noqa: E402
import fechamento_modelo as fm  # noqa: E402
import fechamento_temporal_base as ftb  # noqa: E402
import pilha_s1_mesmo_lugar as ps  # noqa: E402
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402
from proveniencia_de_peso import hash_completo_do_estado  # noqa: E402

CAMP = fd.SAIDA / "campanha"
FINAL = fd.SAIDA / "final"
V1 = REPO / "local_runs" / "aprendizado-conjunto" / "v1" / "campanha"
TODOS = list(range(6))
SEM_HAND = [i for i, k in enumerate(cb.TERRENO) if k != "f_hand"]
SO_HAND = [cb.TERRENO.index("f_hand")]
CFG = {
    "sementes": [7, 11, 23], "passos": 500, "lr_fisica": 3e-3, "lr_temporal": 3e-3, "l2_fisica": 1e-4,
    "batch_temporal": 8192, "lambdas": [0.3, 1.0, 3.0], "lambda_padrao_empate": 1.0, "dobras_internas": 3,
    "semente_dobras_internas": 20261017, "delta_e1_db": 3.0, "qualidade_terra_max": 0.10, "qualidade_contraste_min": 0.05,
    "teto_escasso": 20, "diagnostico_cada": 50, "checkpoint_cada": 100, "threads": 3, "workers": 2,
    "margem_auc": 0.01, "nivel_primarias": 0.975, "bootstrap": 2000, "minimo_unidades": 8,
}
BRACOS = {
    "fisica_so": {"modo": "fisica_so"}, "parada": {"modo": "parada"}, "conjunto": {"modo": "conjunto"},
    "rompido": {"modo": "conjunto", "rompido": True},
    "conjunto_sem_hand": {"modo": "conjunto", "terreno": SEM_HAND},
    "conjunto_sem_conexao": {"modo": "conjunto", "terreno": SO_HAND},
    "fisica_so_sem_hand": {"modo": "fisica_so", "terreno": SEM_HAND},
    "fisica_so_sem_conexao": {"modo": "fisica_so", "terreno": SO_HAND},
    "conjunto_sem_forcamento": {"modo": "conjunto", "forcamento": False, "antecedente": False},
    "conjunto_sem_antecedente": {"modo": "conjunto", "antecedente": False},
    "fisica_so_escasso": {"modo": "fisica_so", "escasso": True},
    "parada_escasso": {"modo": "parada", "escasso": True},
    "conjunto_escasso": {"modo": "conjunto", "escasso": True},
}
PLANO = [  # (destinos, bracos), na ordem de execucao; todas as sementes
    ("dobras", ["fisica_so", "parada", "conjunto", "rompido"]),
    ("dobras", ["fisica_so_escasso", "parada_escasso", "conjunto_escasso"]),
    ("dobras", ["conjunto_sem_hand", "conjunto_sem_conexao", "fisica_so_sem_hand", "fisica_so_sem_conexao",
                "conjunto_sem_forcamento", "conjunto_sem_antecedente"]),
    ("fonte_ufo", ["fisica_so", "parada", "conjunto", "rompido"]),
]


def ler(p):
    return cc.ler(p)


def substituir(tmp: Path, alvo: Path, tentativas: int = 20) -> None:
    """os.replace com novas tentativas: no Windows, o espelho do Drive/antivirus segura o alvo por instantes
    (WinError 5/32 observado em 17/10/2026). Falha de verdade continua levantando."""
    for k in range(tentativas):
        try:
            tmp.replace(alvo)
            return
        except PermissionError:
            if k == tentativas - 1:
                raise
            time.sleep(min(0.25 * (k + 1), 3.0))


def escrever(p, o):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".part")
    t.write_text(json.dumps(o, ensure_ascii=False, indent=1, allow_nan=False, default=str), encoding="utf-8")
    substituir(t, p)


# ===================================================================== alvo temporal por dobra

_T = None


def quadro_temporal() -> pd.DataFrame:
    global _T
    if _T is None:
        cols = ["bloco", "componente", "pixel", "data", "janela", "vv_db", "p75_passado_db", "p72", "ante3_14",
                "jrc_occ", *cb.TERRENO, *fm.INSTRUMENTAIS]
        _T = pd.read_parquet(ftb.BASE / "lugar_instante.parquet", columns=cols)
        for c in ("bloco", "componente", "data", "janela"):
            _T[c] = _T[c].astype("category")
        for c in ("p72", "ante3_14"):
            _T[c] = _T[c].astype(np.float32)
    return _T


def limiar_da_dobra(comps_treino: list[str]) -> dict:
    cal = pd.read_parquet(ftb.BASE / "calibracao_jrc.parquet")
    cal = cal[cal.componente.isin(comps_treino)]
    por = {c: (g[g.tipo == "agua"].vv_db.to_numpy(), g[g.tipo == "terra"].vv_db.to_numpy()) for c, g in cal.groupby("componente")}
    r = ps.calibrar_limiar(por)
    if r["limiar_db"] is None:
        raise ValueError("sem calibracao no treino")
    return {"limiar_db": float(r["limiar_db"]), "ba": r["ba"], "componentes_calibracao": r["componentes_usados"]}


def deteccao(t: pd.DataFrame, limiar: float, referencia: str = "p75_passado_db", delta: float = CFG["delta_e1_db"]) -> np.ndarray:
    """1/0 instrumental; NaN sem leitura ou sem historia anterior suficiente (recusa explicita)."""
    vv, ref = t.vv_db.to_numpy(), t[referencia].to_numpy()
    d = ((vv <= limiar) & (vv <= ref - delta)).astype(np.float32)
    d[~np.isfinite(vv) | ~np.isfinite(ref)] = np.nan
    return d


def qualidade(t: pd.DataFrame, det: np.ndarray) -> dict:
    """Regra congelada so com JRC (sem rotulo fisico, sem chuva): componente fornece supervisao?"""
    fora = (t.janela == "fora").to_numpy() & np.isfinite(det)
    df = pd.DataFrame({"c": t.componente.to_numpy()[fora], "j": t.jrc_occ.to_numpy()[fora], "d": det[fora]})
    out = {}
    for c, g in df.groupby("c"):
        terra = g[g.j == 0].d.mean()
        saz = g[(g.j >= 10) & (g.j < 50)].d.mean()
        out[c] = {"terra": float(terra), "sazonal": float(saz) if np.isfinite(saz) else None,
                  "admitido": bool(np.isfinite(terra) and np.isfinite(saz) and terra <= CFG["qualidade_terra_max"]
                                   and saz - terra >= CFG["qualidade_contraste_min"])}
    return out


def dados_temporais(comps: list[str], limiar: float, adm: set | None) -> dict | None:
    t = quadro_temporal()
    t = t[t.componente.isin(comps)]
    if adm is not None:
        t = t[t.componente.isin(adm)]
    y = deteccao(t, limiar)
    ok = np.isfinite(y) & t.p72.notna().to_numpy() & t.ante3_14.notna().to_numpy()
    t, y = t[ok], y[ok]
    if t.empty:
        return None
    bd = pd.factorize(t.bloco.astype(str) + "|" + t.data.astype(str))[0]
    cnt = t.groupby(["componente", "bloco"], observed=True).bloco.transform("size").to_numpy(float)
    nb = t.groupby("componente", observed=True).bloco.transform("nunique").to_numpy(float)
    w = 1 / (cnt * nb)
    return {"ter": t[list(cb.TERRENO)].to_numpy(np.float32), "instr": t[list(fm.INSTRUMENTAIS)].to_numpy(np.float32),
            "forc": np.log1p(t[["p72", "ante3_14"]].to_numpy(np.float32)), "y": y.astype(np.float32), "bd": bd,
            "w": w / w.sum(), "bloco": t.bloco.to_numpy(), "pixel": t.pixel.to_numpy(), "data": t.data.to_numpy(),
            "componente": t.componente.to_numpy()}


def romper_lugar(td: dict, semente: int) -> dict:
    """Troca os atributos de lugar (terreno e instrumentais) entre pixels do MESMO bloco, iguais em todas as
    datas: a serie de deteccao de um pixel passa a ter o lugar de outro. Preserva aquisicoes, marginais e
    forcamento; rompe so a relacao alvo-lugar."""
    rng = np.random.default_rng([semente, 990])
    ter, ins = td["ter"].copy(), td["instr"].copy()
    df = pd.DataFrame({"b": td["bloco"], "p": td["pixel"]})
    for b, g in df.groupby("b"):
        px = np.unique(g.p.to_numpy())
        if len(px) < 2:
            continue
        perm = dict(zip(px, np.roll(rng.permutation(px), 1)))
        idx = g.index.to_numpy()
        fonte = {p: idx[g.p.to_numpy() == p][0] for p in px}
        alvo = np.array([fonte[perm[p]] for p in g.p.to_numpy()])
        ter[idx], ins[idx] = td["ter"][alvo], td["instr"][alvo]
    return {**td, "ter": ter, "instr": ins}


# ===================================================================== treino

def construir(braco: str, semente: int) -> fm.ModeloTemporal:
    b = BRACOS[braco]
    return fm.ModeloTemporal(b["modo"], b.get("terreno", TODOS), semente=semente,
                             forcamento=b.get("forcamento", True), antecedente=b.get("antecedente", True))


def treinar(pasta: Path, braco: str, semente: int, tr: pd.DataFrame, td: dict | None, lam: float, cfg=CFG,
            parar_em: int | None = None, salvar: bool = True):
    pasta = Path(pasta)
    final = pasta / "final.pt"
    if salvar and final.exists() and (pasta / "treino.json").exists():
        e = torch.load(final, weights_only=False, map_location="cpu")
        m = construir(braco, semente)
        m.load_state_dict(e["modelo"])
        t = ler(pasta / "treino.json")
        if hash_completo_do_estado(e["modelo"]) != t["hash_estado"]:
            raise ValueError("peso final divergente do registro")
        return m, t
    b = BRACOS[braco]
    temporal = b["modo"] != "fisica_so"
    if temporal and td is None:
        raise ValueError("braco temporal sem supervisao temporal no treino")
    if b.get("rompido"):
        td = romper_lugar(td, semente)
    T, C = cc.tensores_fisicos(tr)
    y = torch.tensor(tr.classe.to_numpy(np.float32))
    ev_ids, ev = np.unique(tr.evento_id.to_numpy(), return_inverse=True)
    ev = torch.tensor(ev, dtype=torch.long)
    m = construir(braco, semente)
    m.pad_terreno.ajustar(T)
    m.pad_cobertura.ajustar(C)
    if temporal:
        m.pad_instr.ajustar(torch.tensor(td["instr"]))
        m.pad_forc.ajustar(torch.tensor(td["forc"]))
        A = (torch.tensor(td["ter"]), torch.tensor(td["instr"]), torch.tensor(td["forc"]), torch.tensor(td["y"]),
             torch.tensor(td["bd"], dtype=torch.long))
        a_bd = nn.Parameter(torch.zeros(int(td["bd"].max()) + 1))
    else:
        a_bd = nn.Parameter(torch.zeros(1))
    intercepto = nn.Parameter(torch.zeros(len(ev_ids)))
    fis, tmp = m.grupos()
    opt = torch.optim.Adam([{"params": fis + [intercepto], "lr": cfg["lr_fisica"]},
                            {"params": tmp + [a_bd], "lr": cfg["lr_temporal"]}])
    rng = np.random.default_rng([semente, 5151])
    hist, diag, inicio, seg0, t0 = [], [], 0, 0., time.monotonic()
    ck = pasta / "checkpoint.pt"
    if salvar and ck.exists():
        e = torch.load(ck, weights_only=False, map_location="cpu")
        m.load_state_dict(e["modelo"])
        with torch.no_grad():
            intercepto.copy_(e["intercepto"]); a_bd.copy_(e["a_bd"])
        opt.load_state_dict(e["otimizador"])
        rng.bit_generator.state = e["rng_numpy"]
        torch.set_rng_state(e["rng_torch"])
        hist, diag, inicio, seg0 = e["historico"], e["diagnostico"], e["passo"], e["segundos"]
    enc = list(m.enc_fis.parameters())
    for passo in range(inicio, cfg["passos"]):
        if parar_em is not None and passo >= parar_em:
            break
        opt.zero_grad(set_to_none=True)
        lf = cm.perda_fisica(m.fisica(T, C), y, ev, intercepto) + .5 * cfg["l2_fisica"] * sum(
            (p * p).sum() for n, p in m.named_parameters() if n.startswith(("enc_fis.", "cab_fis.")) and p.ndim > 1)
        lt = torch.zeros(())
        if temporal:
            ids = torch.from_numpy(rng.choice(len(td["y"]), cfg["batch_temporal"], replace=True, p=td["w"]))
            logit = m.temporal(A[0][ids], A[1][ids], A[2][ids]) + a_bd[A[4][ids]]
            lt = nn.functional.binary_cross_entropy_with_logits(logit, A[3][ids])
        if passo % cfg["diagnostico_cada"] == 0 or passo == cfg["passos"] - 1:
            gf = cc._grad(lf, enc)
            go = cc._grad(lam * lt, enc) if lt.requires_grad else torch.zeros_like(gf)
            nf, no = float(gf.norm()), float(go.norm())
            diag.append({"passo": passo, "norma_grad_fisica_enc": nf, "norma_grad_temporal_enc": no,
                         "cosseno": float(gf @ go / (nf * no)) if nf > 0 and no > 0 else None})
        (lf + lam * lt).backward()
        opt.step()
        hist.append({"passo": passo, "perda_fisica": float(lf.detach()), "perda_temporal": float(lt.detach())})
        if salvar and (passo + 1) % cfg["checkpoint_cada"] == 0 and passo + 1 < cfg["passos"]:
            pasta.mkdir(parents=True, exist_ok=True)
            tmpf = ck.with_suffix(".part")
            torch.save({"modelo": m.state_dict(), "intercepto": intercepto.detach().clone(), "a_bd": a_bd.detach().clone(),
                        "otimizador": opt.state_dict(), "rng_numpy": rng.bit_generator.state, "rng_torch": torch.get_rng_state(),
                        "historico": hist, "diagnostico": diag, "passo": passo + 1,
                        "segundos": seg0 + time.monotonic() - t0}, tmpf)
            substituir(tmpf, ck)
    if len(hist) < cfg["passos"]:
        return m, None
    estado = {k: v.detach().clone() for k, v in m.state_dict().items()}
    t = {"braco": braco, "semente": semente, "lambda": lam, "hash_estado": hash_completo_do_estado(estado),
         "passos": cfg["passos"], "segundos": seg0 + time.monotonic() - t0,
         "parametros_total": sum(p.numel() for p in m.parameters()),
         "parametros_encoder": sum(p.numel() for p in m.enc_fis.parameters()),
         "parametros_compartilhados": m.parametros_compartilhados(), "pontos_fisicos_treino": len(tr),
         "eventos_treino": len(ev_ids), "linhas_temporais_treino": int(len(td["y"])) if temporal else 0,
         "aquisicoes_treino": int(td["bd"].max() + 1) if temporal else 0,
         "historico": hist, "diagnostico_gradiente": diag, "retomado": inicio > 0,
         "memoria_rss_mb": cc._rss_mb(), "versoes": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    if salvar:
        pasta.mkdir(parents=True, exist_ok=True)
        torch.save({"modelo": estado, "intercepto": intercepto.detach().clone(), "a_bd": a_bd.detach().clone(),
                    "otimizador": opt.state_dict()}, final)
        escrever(pasta / "treino.json", t)
        if ck.exists():
            ck.unlink()
    return m, t


# ===================================================================== avaliacao

def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


@torch.no_grad()
def avaliar(pasta: Path, m: fm.ModeloTemporal, braco: str, te: pd.DataFrame, td_te: dict | None) -> dict:
    import mt05_transferencia_sem_amostra as mt
    T, C = cc.tensores_fisicos(te)
    s = m.fisica(T, C).numpy().astype(np.float64)
    pred = te[["ponto_id", "evento_id", "unidade", "classe", "fonte", "processo_ponto"]].copy()
    pred["escore"] = s
    pred.to_parquet(pasta / "predicoes_fisicas.parquet", index=False)
    medidas = []
    for un, sub in pred.groupby("unidade"):
        medidas += mt.medir(sub.reset_index(drop=True), {"escore": sub.escore.to_numpy()}, un)
    r = {"braco": braco, "medidas": medidas, "predicoes_sha256": sha256_arquivo(pasta / "predicoes_fisicas.parquet")}
    if BRACOS[braco]["modo"] != "fisica_so" and td_te is not None:
        st = m.temporal(torch.tensor(td_te["ter"]), torch.tensor(td_te["instr"]), torch.tensor(td_te["forc"])).numpy()
        np.savez_compressed(pasta / "temporal_escore.npz", escore=st.astype(np.float32))
        df = pd.DataFrame({"c": td_te["componente"], "bd": td_te["bd"], "y": td_te["y"], "s": st})
        por = []
        for (c, bd), g in df.groupby(["c", "bd"]):
            if 0 < g.y.sum() < len(g):
                por.append({"componente": c, "auc": auc(g.y, g.s)})
        tp = pd.DataFrame(por)
        r["temporal_por_componente"] = tp.groupby("componente").auc.agg(["mean", "size"]).rename(
            columns={"mean": "auc_dentro_da_aquisicao", "size": "aquisicoes"}).reset_index().to_dict("records") if len(tp) else []
        r["temporal_sha256"] = sha256_arquivo(pasta / "temporal_escore.npz")
    escrever(pasta / "avaliacao.json", r)
    return r


# ===================================================================== particao e lambda

def preparar_destino(d: cc.Dados, destino: str) -> dict:
    part = cc.particao(d, "fonte_ufo" if destino == "fonte_ufo" else destino)
    tr_comps, te_comps = part["unidades_treino"], part["unidades_teste"]
    lim = limiar_da_dobra(tr_comps)
    t = quadro_temporal()
    q = qualidade(t, deteccao(t, lim["limiar_db"]))
    adm_tr = {c for c in tr_comps if q.get(c, {}).get("admitido")}
    adm_te = sorted(c for c in te_comps if q.get(c, {}).get("admitido"))
    info = {"destino": destino, "limiar": lim, "qualidade": q, "supervisao_treino": sorted(adm_tr),
            "populacao_supervisao_admissivel_destino": adm_te, "unidades_teste": te_comps,
            "ids_teste_sha256": sha256_json(part["te"].ponto_id.tolist())}
    return {**part, "info": info, "adm_tr": adm_tr, "limiar_db": lim["limiar_db"]}


def escolher_lambda(destino: str, part: dict, d: cc.Dados) -> dict:
    arq = CAMP / destino / "lambda.json"
    if arq.exists():
        return ler(arq)
    import mt05_transferencia_sem_amostra as mt
    tr = part["tr"]
    comps = np.array(sorted(tr.unidade.unique()))
    k = CFG["dobras_internas"]
    dobra = dict(zip(comps, np.random.default_rng(CFG["semente_dobras_internas"]).permutation(len(comps)) % k))
    notas = {}
    for lam in CFG["lambdas"]:
        aucs = []
        for j in range(k):
            itr = tr[tr.unidade.map(dobra) != j]
            ite = tr[tr.unidade.map(dobra) == j]
            ite = ite.groupby("evento_id").filter(lambda g: len(g) >= 20 and g.classe.nunique() == 2)
            td = dados_temporais(sorted(set(itr.unidade) & part["adm_tr"]), part["limiar_db"], None)
            if td is None or ite.empty:
                continue
            m, _ = treinar(CAMP / destino / "_interno", "conjunto", 7, itr, td, lam, salvar=False)
            with torch.no_grad():
                T, C = cc.tensores_fisicos(ite)
                s = m.fisica(T, C).numpy()
            med = []
            for un, sub in ite.assign(escore=s).groupby("unidade"):
                med += [x["auc"] for x in mt.medir(sub.reset_index(drop=True), {"e": sub.escore.to_numpy()}, un) if x["recorte"] == "todos"]
            aucs.append(float(np.mean(med)))
        notas[str(lam)] = float(np.mean(aucs)) if aucs else None
    validas = {float(k_): v for k_, v in notas.items() if v is not None}
    melhor = max(validas, key=lambda l: (round(validas[l], 4), l == CFG["lambda_padrao_empate"])) if validas else CFG["lambda_padrao_empate"]
    r = {"destino": destino, "notas_internas_auc_fisica": notas, "lambda": melhor,
         "regra": "maior AUC fisica media nas dobras internas do treino; empate em 4 casas -> 1,0"}
    escrever(arq, r)
    return r


# ===================================================================== protocolo e execucao

def destinos() -> list[str]:
    return [f"dobra_{k}" for k in range(8)] + ["fonte_ufo"]


def plano() -> list[tuple[str, str, int]]:
    out = []
    for dests, bracos in PLANO:
        alvo = [f"dobra_{k}" for k in range(8)] if dests == "dobras" else [dests]
        for s in CFG["sementes"]:
            for x in alvo:
                for b in bracos:
                    out.append((x, b, s))
    return out


def poder() -> dict:
    a = ler(V1 / "analise.json")
    dd = pd.Series(a["primarias"]["P1_conjunto-parada_fisica"]["por_unidade"])
    sd = float(dd.std())
    return {"fonte": "conjunto - parada da v1 (43 componentes)", "dp_entre_componentes": sd, "n": len(dd),
            "semi_largura_ic97_5": float(2.24 * sd / np.sqrt(len(dd))),
            "efeito_minimo_detectavel_80pct": float((2.24 + .84) * sd / np.sqrt(len(dd))),
            "leitura": "com esta variancia, um ganho de +0,01 seria detectavel; um nulo estreito e informativo"}


def protocolo() -> dict:
    codigo = [Path(__file__), Path(fm.__file__), Path(ftb.__file__), Path(cm.__file__), Path(cc.__file__)]
    return {
        "versao": "fechamento_aprendizado_transferencia_v1", "tipo": "protocolo_congelado_antes_dos_resultados",
        "regime": "desenvolvimento retrospectivo; nenhuma reserva aberta; confirmacao pendente",
        "base_temporal_sha256": sha256_arquivo(ftb.BASE / "manifesto.json"),
        "base_fisica_sha256": sha256_arquivo(cb.BASE / "manifesto.json"),
        "diagnostico_A_sha256": sha256_arquivo(fd.SAIDA / "diagnostico" / "diagnostico_A.json"),
        "codigo_sha256": {str(p.relative_to(REPO)): sha256_arquivo(p) for p in codigo},
        "config": CFG, "bracos": {k: {**v, "terreno": [cb.TERRENO[i] for i in v.get("terreno", TODOS)]} for k, v in BRACOS.items()},
        "plano": PLANO, "destinos": destinos(), "ajustes_previstos": len(plano()),
        "populacoes": {"fisica": "43 componentes historicos (MT05/v1), mesmos pontos de teste",
                       "supervisao_temporal": "componentes CEMS da pilha admitidos pela regra congelada, por dobra; registrada antes dos resultados fisicos"},
        "primarias": {
            "P1": {"contraste": "conjunto - parada", "metrica": "AUC fisica por componente (recorte todos), dobras", "unidades": 43},
            "P2": {"contraste": "conjunto - parada", "metrica": "AUC fisica por componente", "unidades": "componentes de destino com supervisao temporal admissivel"},
            "nivel": "IC97,5% por bootstrap de componentes (familia de 2)",
            "regra": ("historica MT01-v1: sustentado se media >= 0,01 E IC_inf > 0; nao_sustentado se IC_sup < 0,01; "
                      "senao indecidivel. Isso NAO exige IC_inf > 0,01; a versao estrita (IC_inf > 0,01) e relatada a parte"),
        },
        "secundarias_exploratorias": [
            "conjunto - rompido (fisica): necessidade de alinhamento alvo-lugar",
            "conjunto_escasso - parada_escasso (fisica, 20 por evento-classe): hipotese tecnica T2 do diagnostico",
            "ablacoes retreinadas: sem_hand, sem_conexao (conjunto e fisica_so), sem_forcamento, sem_antecedente",
            "tarefa temporal: AUC dentro da aquisicao por componente retido (conjunto, parada, rompido, sem_forcamento)",
            "fonte UFO retida; fonte CEMS not_testable", "comparacao com Q_base/Q_conexao reajustadas na dobra (v1)",
        ],
        "agregacao": "eventos -> componente; sementes medias antes do bootstrap; contrastes pareados nas sementes comuns",
        "limites_validacao_cruzada": "treinos de dobras diferentes se sobrepoem; o IC por componentes nao inclui essa dependencia",
        "poder": poder(), "nao_e": "nao promove, nao altera servico, nao le reserva; alvo temporal e instrumental",
    }


def congelar() -> dict:
    p = json.loads(json.dumps(protocolo(), default=str))
    arq = CAMP / "protocolo.json"
    if arq.exists():
        antigo = ler(arq)
        if {k: v for k, v in antigo.items() if k != "codigo_sha256"} != {k: v for k, v in p.items() if k != "codigo_sha256"}:
            raise ValueError("protocolo mudou depois de congelado; versionar")
        if antigo["codigo_sha256"] != p["codigo_sha256"]:
            em = ler(CAMP / "emendas.json") if (CAMP / "emendas.json").exists() else []
            if not any(e["codigo_sha256"] == p["codigo_sha256"] for e in em):
                raise ValueError("codigo mudou sem emenda registrada")
        return p
    escrever(arq, p)
    escrever(CAMP / "registro_inicio.json", {"quando_utc": agora_utc(), "protocolo_sha256": sha256_arquivo(arq),
                                             "resultados_lidos": False})
    return p


def executar(worker=0, workers=1, mesmo_na_bateria=False):
    torch.set_num_threads(CFG["threads"])
    torch.use_deterministic_algorithms(True)
    congelar()
    d = cc.Dados()
    itens = [x for k, x in enumerate(plano()) if k % workers == worker]
    cache = {}
    feitos = 0
    for destino, braco, semente in itens:
        pasta = CAMP / destino / f"{braco}_s{semente}"
        if (pasta / "avaliacao.json").exists():
            continue
        if not cc.esperar_tomada(mesmo_na_bateria, 720.):
            return {"estado": "pausado_bateria", "feitos": feitos}
        if destino not in cache:
            part = preparar_destino(d, destino)
            if (CAMP / destino / "particao.json").exists():
                if ler(CAMP / destino / "particao.json")["ids_teste_sha256"] != part["info"]["ids_teste_sha256"]:
                    raise ValueError("particao mudou")
            lam = escolher_lambda(destino, part, d)["lambda"]
            td_tr = dados_temporais(sorted(part["adm_tr"]), part["limiar_db"], None)
            td_te = dados_temporais(part["unidades_teste"], part["limiar_db"], None)
            cache = {destino: (part, lam, td_tr, td_te)}
        part, lam, td_tr, td_te = cache[destino]
        tr = cc.amostrar(part["tr"], CFG["teto_escasso"], 20261010) if BRACOS[braco].get("escasso") else part["tr"]
        t0 = time.monotonic()
        m, _ = treinar(pasta, braco, semente, tr, td_tr, lam)
        avaliar(pasta, m, braco, part["te"], td_te)
        feitos += 1
        print(json.dumps({"worker": worker, "destino": destino, "braco": braco, "semente": semente,
                          "s": round(time.monotonic() - t0, 1), "quando_utc": agora_utc()}), flush=True)
    return {"estado": "concluido", "feitos": feitos, "worker": worker}


def preparar_todos():
    congelar()
    d = cc.Dados()
    for destino in destinos():
        part = preparar_destino(d, destino)
        if not (CAMP / destino / "particao.json").exists():
            escrever(CAMP / destino / "particao.json", part["info"])
        escolher_lambda(destino, part, d)
        print(destino, ler(CAMP / destino / "lambda.json")["lambda"], len(part["adm_tr"]), flush=True)


def executar_paralelo(workers: int, mesmo_na_bateria=False):
    import subprocess
    torch.set_num_threads(CFG["threads"])
    torch.use_deterministic_algorithms(True)
    preparar_todos()
    global _T
    _T = None  # o pai so espera os workers: nao segura o quadro temporal (~2 GB) enquanto eles rodam
    import gc
    gc.collect()
    import workers_seguros as ws  # emenda 3: espera todos e propaga falha/pausa
    cmds = [[sys.executable, "-X", "utf8", str(Path(__file__)), "executar", "--worker", str(w), "--workers", str(workers)]
            + (["--mesmo-na-bateria"] if mesmo_na_bateria else []) for w in range(workers)]
    r = ws.rodar_workers(cmds, [CAMP / f"worker_{w}.log" for w in range(workers)])
    return {"workers": workers, **r}


def completude():
    import workers_seguros as ws
    esperados = [CAMP / x / f"{b}_s{s}" / "avaliacao.json" for x, b, s in plano()]
    return ws.manifesto_completude(esperados, ws.json_valido, CAMP / "completude.json")


def treinar_final(semente=7) -> dict:
    """Pesos em TODOS os componentes (sem avaliacao propria) para o pacote de confirmacao."""
    torch.set_num_threads(CFG["threads"])
    torch.use_deterministic_algorithms(True)
    d = cc.Dados()
    tr = cc.amostrar(d.pts, 400, 20261010)
    comps = sorted(tr.unidade.unique())
    lim = limiar_da_dobra(comps)
    q = qualidade(quadro_temporal(), deteccao(quadro_temporal(), lim["limiar_db"]))
    adm = sorted(c for c in comps if q.get(c, {}).get("admitido"))
    td = dados_temporais(adm, lim["limiar_db"], None)
    lams = [ler(CAMP / x / "lambda.json")["lambda"] for x in destinos() if (CAMP / x / "lambda.json").exists()]
    lam = float(pd.Series(lams).mode().iloc[0]) if lams else 1.0
    out = {}
    for b in ("conjunto", "parada", "fisica_so"):
        _, t = treinar(FINAL / b, b, semente, tr, td, lam)
        out[b] = {"hash_estado": t["hash_estado"], "segundos": t["segundos"]}
    escrever(FINAL / "manifesto.json", {"tipo": "pesos_finais_para_pacote_de_confirmacao", "quando_utc": agora_utc(),
                                        "bracos": out, "lambda": lam, "limiar": lim, "supervisao": adm,
                                        "nao_e": "nao avaliado; desempenho pertence as dobras; nao promovido"})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["protocolo", "preparar", "executar", "paralelo", "final"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mesmo-na-bateria", action="store_true")
    a = ap.parse_args()
    if a.acao == "protocolo":
        print(json.dumps({k: v for k, v in congelar().items() if k in ("ajustes_previstos", "poder")}, indent=1))
    elif a.acao == "preparar":
        preparar_todos()
    elif a.acao == "paralelo":
        r = executar_paralelo(a.workers, a.mesmo_na_bateria)
        print(r)
        sys.exit(r["codigo"])  # emenda 3: a CLI nao sai com 0 se um worker falhou ou pausou
    elif a.acao == "final":
        import workers_seguros as ws
        ws.exigir(0, completude(), "final")  # emenda 3: final so com todas as avaliacoes integras
        print(treinar_final())
    else:
        r = executar(a.worker, a.workers, a.mesmo_na_bateria)
        print(r)
        sys.exit(0 if r["estado"] == "concluido" else 3)
