"""MS1: o piso amostral de `elevation_local_m` infla o aprendizado e quebra a transferencia? Posicao topografica sem rotulo, uma e varias escalas, a substitui?

-------------------------------------------------------------------------
PERGUNTAS (fixadas em `pre_registro.json` ANTES de qualquer ajuste)
-------------------------------------------------------------------------

  C0  base de terreno sem piso  x  regra de HAND           (referencia)
  C1  base + elevation_local_m  x  base                    quanto o piso amostral acrescenta
  C2  base + pos_min_2km        x  base                    a posicao SEM rotulo acrescenta?
  C3  base + multiescala        x  base + pos_min_2km      varias escalas acrescentam a uma?

  base         logistica de hand_m, twi_dinf, slope_deg
  multiescala  pos_min_{1,2,4}km + dev_{1,2,4}km (De Reu et al. 2013)
  cortes       gfd_dentro (5 dobras por evento, a1), cems_dentro (dobras por
               evento, a1), gfd_para_cems e cems_para_gfd (teste sem tile de
               1 grau do treino, pela coordenada)
  populacao    carregar(TERRENO) e as seis colunas de posicao finitas; a perda
               por classe vai gravada
  leitura      generalizacao_por_reservas.comparar + veredito: Wilcoxon da AUC
               DENTRO do evento, relevancia 0,01, >= 8 eventos; nao muda depois
  primario     C2 em gfd_dentro e em cems_para_gfd

Regime exploratorio: eventos conhecidos do ds06. O que ja se sabia antes do
pre-registro vai declarado: a v2 da generalizacao mostrou CEMS -> GFD 0,573 <
HAND 0,738, com a logistica do CEMS dando peso -1,37 a elevation_local_m.

nao_e: nao e teste confirmatorio, nao muda o vetor servido nem a resposta para
lugar novo; um ganho aqui so autoriza propor o vetor para um teste prospectivo.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

import generalizacao_por_reservas as gen  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "ms1-posicao-topografica"
BASE = ("hand_m", "twi_dinf", "slope_deg")
POSICAO = ("pos_min_1km", "pos_min_2km", "pos_min_4km", "dev_1km", "dev_2km", "dev_4km")
MODELOS = {
    "M0_regra_hand": None,
    "base": BASE,
    "base_piso_amostral": BASE + ("elevation_local_m",),
    "base_pos2km": BASE + ("pos_min_2km",),
    "base_multiescala": BASE + POSICAO,
}
COMPARACOES = (
    ("C0", "base", "M0_regra_hand", "base de terreno sem piso supera a regra de HAND?"),
    ("C1", "base_piso_amostral", "base", "o piso amostral (rotulo-dependente) acrescenta?"),
    ("C2", "base_pos2km", "base", "a posicao topografica sem rotulo acrescenta?"),
    ("C3", "base_multiescala", "base_pos2km", "varias escalas acrescentam a uma?"),
)
CORTES = ("gfd_dentro", "cems_dentro", "gfd_para_cems", "cems_para_gfd")
PRIMARIO = {"comparacao": "C2", "cortes": ["gfd_dentro", "cems_para_gfd"]}
SEMENTE = 20261007
VISTO_ANTES = ("generalizacao_v2 (07/10): CEMS -> GFD logistica 0,573 < HAND 0,738; logistica do CEMS com "
               "peso -1,37 em elevation_local_m e TWI com sinal trocado; descritores de posicao inspecionados "
               "em um tile (GFD_S12W051) sem ajuste de modelo")


def pre_registro() -> dict:
    return json.loads(json.dumps({
        "modelos": {k: list(v) if v else "regra: -hand_m" for k, v in MODELOS.items()},
        "comparacoes": [list(c) for c in COMPARACOES], "cortes": CORTES, "primario": PRIMARIO,
        "populacao": "carregar(TERRENO) com as seis colunas de posicao finitas",
        "estimador": "logistica IRLS com intercepto, padronizada no treino, ate 60.000 linhas por ajuste",
        "dobras": {"gfd": gen.DOBRAS["gfd"], "cems": gen.DOBRAS["cems"], "atribuicao": "a1"},
        "entre_fontes": "teste sem tile de 1 grau do treino (verdade_observacional.chave_espacial)",
        "regra": "generalizacao_por_reservas.veredito (Wilcoxon dentro do evento, relevancia 0,01, >= 8 eventos)",
        "posicao_topografica": "scripts/suscetibilidade/posicao_topografica.py v1 (MDT 240 m, raios 1/2/4 km)",
        "visto_antes_do_pre_registro": VISTO_ANTES, "semente": SEMENTE}))


def gravar_pre_registro(destino: Path = SAIDA / "pre_registro.json") -> str:
    corpo = pre_registro()
    h = sha256_json(corpo)
    if destino.exists():
        antigo = json.loads(destino.read_text(encoding="utf-8"))
        if antigo["hash"] != h:
            raise RuntimeError(f"pre-registro mudou ({antigo['hash']} -> {h}): crie MS2")
        return h
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"quando_utc": agora_utc(), "hash": h, "criterios": corpo},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
    return h


def ajustar(tr: pd.DataFrame, vetor) -> callable:
    if vetor is None:
        return lambda te: -te["hand_m"].to_numpy(float)
    vetor = list(vetor)
    if len(tr) > gen.MAX_LINHAS_TREINO:
        tr = tr.sample(gen.MAX_LINHAS_TREINO, random_state=SEMENTE)
    X = tr[vetor].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    b = gen.logistica_com_intercepto((X - mu) / sd, tr.classe.to_numpy(float))
    if b is None:
        raise RuntimeError(f"logistica nao convergiu: {vetor}")
    f = lambda te: expit(b[0] + ((te[vetor].to_numpy(float) - mu) / sd) @ b[1:])  # noqa: E731
    f.coef = dict(zip(vetor, np.round(b[1:], 4).tolist()))
    return f


def medir(s: pd.DataFrame, pred: dict) -> dict:
    """gen.medir com os nomes deste experimento: Brier so para o que e probabilidade."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y = s.classe.to_numpy()
    out = {}
    for m, p in pred.items():
        ok = np.isfinite(p)
        pe = gen.auc_por_evento(s[ok], p[ok])
        out[m] = {"auc_agrupada_fora_da_dobra": round(float(roc_auc_score(y[ok], p[ok])), 4),
                  "average_precision": round(float(average_precision_score(y[ok], p[ok])), 4),
                  "auc_media_dentro_do_evento": round(float(np.mean(list(pe.values()))), 4) if pe else None,
                  "auc_mediana_dentro_do_evento": round(float(np.median(list(pe.values()))), 4) if pe else None,
                  "eventos_com_auc": len(pe),
                  "brier": None if m.startswith("M0_regra") else round(float(brier_score_loss(y[ok], p[ok])), 4)}
    return out


def prever_dentro(s0: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict, dict]:
    atrib = gen.dobras(sorted(s0.evento_id.unique()), k, "a1")
    s = s0.assign(dobra=s0.evento_id.map(atrib)).reset_index(drop=True)
    pred = {m: np.full(len(s), np.nan) for m in MODELOS}
    coef = {m: [] for m in MODELOS if MODELOS[m]}
    for f in range(k):
        teste_ev = sorted(e for e, x in atrib.items() if x == f)
        tr, te = gen.separar(s, teste_ev)
        pos = np.flatnonzero(s.dobra.to_numpy() == f)
        for m, v in MODELOS.items():
            fn = ajustar(tr, v)
            pred[m][pos] = fn(te)
            if v:
                coef[m].append(fn.coef)
    return s, pred, coef


def prever_entre(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    import verdade_observacional as vo
    te = te[~vo.chave_espacial(te).isin(set(vo.chave_espacial(tr)))].reset_index(drop=True)
    pred, coef = {}, {}
    for m, v in MODELOS.items():
        fn = ajustar(tr, v)
        pred[m] = fn(te)
        if v:
            coef[m] = fn.coef
    return te, pred, coef


def executar() -> dict:
    import posicao_topografica as pt
    h = gravar_pre_registro()
    d, versao, filtro = gen.carregar(gen.TERRENO)
    antes = d
    d = pt.juntar(d)
    d = d[np.isfinite(d[list(POSICAO)].to_numpy(float)).all(axis=1)].copy()
    perda = {f: gen.perda_por_classe(antes[antes.fonte == f], d[d.fonte == f]) for f in gen.FONTES}
    rng = np.random.default_rng(SEMENTE)
    g, c = d[d.fonte == "gfd"], d[d.fonte == "cems"]
    inicio = time.time()
    cortes = {}
    for nome, (s, pred, coef) in (
            ("gfd_dentro", prever_dentro(g, gen.DOBRAS["gfd"])),
            ("cems_dentro", prever_dentro(c, min(gen.DOBRAS["cems"], c.evento_id.nunique()))),
            ("gfd_para_cems", prever_entre(g, c)),
            ("cems_para_gfd", prever_entre(c, g))):
        cmp = {}
        for cid, a, b, q in COMPARACOES:
            r = gen.comparar(s, pred[a], pred[b], rng)
            cmp[cid] = {"pergunta": q, "a": a, "b": b, **r, "veredito": gen.veredito(r)}
        cortes[nome] = {"linhas": int(len(s)), "eventos": int(s.evento_id.nunique()),
                        "medidas": medir(s, pred), "comparacoes": cmp, "coeficientes": coef}
        print(nome, round(time.time() - inicio), "s", {k: v["veredito"] for k, v in cmp.items()}, flush=True)
    r = {"tipo": "ms1_posicao_topografica", "dataset": versao, "pre_registro": h, "regime": "exploratorio",
         "populacao": {"linhas_por_fonte": d.groupby("fonte").size().to_dict(),
                       "perda_por_classe_ao_exigir_posicao": perda},
         "cortes": cortes,
         "primario": {k: cortes[k]["comparacoes"][PRIMARIO["comparacao"]]["veredito"] for k in PRIMARIO["cortes"]},
         "nao_e": ("nao e confirmatorio nem muda o vetor servido; ganho aqui so autoriza propor o vetor para um "
                   "teste prospectivo")}
    r["hash_resumo"] = sha256_json(json.loads(json.dumps(r, default=float)))
    r["quando_utc"] = agora_utc()
    r["segundos"] = round(time.time() - inicio)
    (SAIDA / "resultado.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return r


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-registrar", action="store_true")
    a = ap.parse_args()
    if a.pre_registrar:
        print(gravar_pre_registro())
        return 0
    r = executar()
    print(json.dumps(r["primario"], ensure_ascii=False))
    print(f"GRAVADO={SAIDA / 'resultado.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
