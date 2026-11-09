"""MS4: HAND, logistica sem o piso amostral, posicao, vizinhanca simples, GBM e encoder -- mesmas linhas, mesma avaliacao.

-------------------------------------------------------------------------
POR QUE
-------------------------------------------------------------------------

Os bracos foram medidos em experimentos separados, com populacoes e vetores
diferentes (N1/N1b com hand+twi, MS1 com o piso amostral, generalizacao com
`elevation_local_m`). A16 mostrou que o piso depende do desenho amostral e nao
e calculavel numa consulta; a posicao v2 (`posicao_no_ponto`) e. Antes de
ampliar redes, o encoder precisa passar da vizinhanca simples NAS MESMAS
LINHAS, com o mesmo estimador e o custo declarado.

-------------------------------------------------------------------------
DESENHO (pre_registro.json antes de qualquer ajuste)
-------------------------------------------------------------------------

  linhas     amostra do N1 (GFD planicie fluvial, admitida para avaliacao
             espacial, 69 eventos) com todas as colunas dos bracos finitas;
             perda por classe gravada
  dobras     5 por evento (generalizacao_por_reservas.dobras, a1)
  estimador  logistica L2 (C=1) padronizada no treino -- o do N1/N1b; GBM com
             a configuracao da generalizacao
  bracos     A0 regra -HAND
             A1 base: hand, twi, slope
             A1e base + elevation_local_m (so descritivo: piso amostral)
             A2 base + pos_min_2km (v2)
             A3 base + posicao v2 nas tres escalas (pos_min e dev)
             A4 A2 + vizinhanca simples (25 estatisticas da janela de 3,84 km,
                padronizadas e PCA16 no treino -- a referencia do N1b)
             A5 A4 + embedding do encoder congelado (PCA16 no treino)
             A6 GBM sobre o vetor de A2
  comparacoes K1 A1xA0, K2 A2xA1, K3 A3xA2, K4 A4xA2, K5 A5xA4, K6 A6xA2
  leitura    generalizacao_por_reservas.veredito (Wilcoxon dentro do evento,
             relevancia 0,01, >= 8 eventos); sem primario: mapa de
             desenvolvimento, nao teste
  custo      segundos de ajuste por braco; custo de pre-treino do encoder lido
             do manifesto dele

Regime exploratorio: a amostra do N1 ja foi lida pelo N1, N1b, N2 e V1.

nao_e: nao e teste, nao escolhe modelo para servir; ganho do encoder aqui so
autoriza leva-lo a um teste em territorio novo com o mesmo orcamento.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

import generalizacao_por_reservas as gen  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

N1 = REPO / "local_runs" / "campanha-integrada-20261003" / "N1"
V1 = REPO / "local_runs" / "campanha-integrada-20261003" / "V1" / "estatisticas_multiescala.npz"
ESCALA_VIZINHANCA = "128"                    # 128 px x 30 m = 3,84 km, a janela do N1
SAIDA = REPO / "local_runs" / "ms4-bracos-de-desenvolvimento"
SEMENTE = 20261008
COMPONENTES = 16
BASE = ("hand_m", "twi_dinf", "slope_deg")
POS = ("pos_min_1km", "dev_1km", "pos_min_2km", "dev_2km", "pos_min_4km", "dev_4km")
BRACOS = {
    "A0_regra_hand": {"tipo": "regra"},
    "A1_base": {"tipo": "logistica", "vetor": BASE},
    "A1e_base_piso_amostral": {"tipo": "logistica", "vetor": BASE + ("elevation_local_m",)},
    "A2_base_pos2km": {"tipo": "logistica", "vetor": BASE + ("pos_min_2km",)},
    "A3_base_multiescala": {"tipo": "logistica", "vetor": BASE + POS},
    "A4_vizinhanca_simples": {"tipo": "logistica", "vetor": BASE + ("pos_min_2km",), "bloco": "vizinhanca"},
    "A5_encoder": {"tipo": "logistica", "vetor": BASE + ("pos_min_2km",), "bloco": "vizinhanca+embedding"},
    "A6_gbm": {"tipo": "gbm", "vetor": BASE + ("pos_min_2km",)},
}
COMPARACOES = (("K1", "A1_base", "A0_regra_hand"), ("K2", "A2_base_pos2km", "A1_base"),
               ("K3", "A3_base_multiescala", "A2_base_pos2km"), ("K4", "A4_vizinhanca_simples", "A2_base_pos2km"),
               ("K5", "A5_encoder", "A4_vizinhanca_simples"), ("K6", "A6_gbm", "A2_base_pos2km"))
VISTO_ANTES = ("N1/N1b (03/10): ganho do embedding era vizinhanca e so dentro do evento no GFD; MS1 (07/10): pos_min_2km "
               "v1 +0,030 no GFD; A16 (08/10): piso amostral depende da extensao amostral")


def pre_registro() -> dict:
    return json.loads(json.dumps({
        "bracos": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in BRACOS.items()},
        "comparacoes": [list(c) for c in COMPARACOES], "primario": None,
        "linhas": "amostra do N1 com todas as colunas finitas", "dobras": "5 por evento, a1",
        "estimador": "logistica L2 C=1 padronizada no treino; GBM = generalizacao_por_reservas.GBM",
        "vizinhanca": f"V1 escala {ESCALA_VIZINHANCA} px, 25 estatisticas, padronizadas e PCA{COMPONENTES} no treino",
        "embedding": f"N1/embeddings.npy, PCA{COMPONENTES} no treino",
        "posicao": "posicao_no_ponto v2 (30 m, janela inteira, mesma funcao da consulta)",
        "regra": "generalizacao_por_reservas.veredito", "visto_antes": VISTO_ANTES, "semente": SEMENTE}))


def gravar_pre_registro(destino: Path = SAIDA / "pre_registro.json") -> str:
    corpo = pre_registro()
    h = sha256_json(corpo)
    if destino.exists():
        antigo = json.loads(destino.read_text(encoding="utf-8"))
        if antigo["hash"] != h:
            raise RuntimeError(f"pre-registro mudou ({antigo['hash']} -> {h}): crie MS5")
        return h
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"quando_utc": agora_utc(), "hash": h, "criterios": corpo},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
    return h


def medir_com_regra(d, pred: dict, regras: set) -> dict:
    """ms1.medir so reconhece a regra pelo prefixo M0_regra; aqui a regra tem outro nome e nao tem Brier."""
    import ms1_posicao_topografica as ms1
    out = ms1.medir(d, {k: v for k, v in pred.items() if k not in regras})
    for k in regras:
        out[k] = ms1.medir(d, {"M0_regra_hand": pred[k]})["M0_regra_hand"]
    return out


def montar() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    import posicao_no_ponto as pp
    from ds06_dataset_de_treino import versao_atual
    a = pd.read_parquet(N1 / "amostra.parquet")
    E = np.load(N1 / "embeddings.npy")
    V = np.load(V1)[ESCALA_VIZINHANCA]
    assert len(a) == len(E) == len(V) and a.janela_valida.all(), "N1/V1 desalinhados"
    d6 = pd.read_parquet(versao_atual() / "pontos.parquet", columns=["ponto_id", "slope_deg", "elevation_local_m"])
    a = a.merge(d6, on="ponto_id", how="left", validate="one_to_one")
    a = pp.juntar(a)
    colunas = sorted(set(BASE) | set(POS) | {"elevation_local_m"})
    ok = (np.isfinite(a[colunas].to_numpy(float)).all(axis=1) & np.isfinite(E).all(axis=1)
          & np.isfinite(V).all(axis=1))
    perda = gen.perda_por_classe(a, a[ok])
    return a[ok].reset_index(drop=True), E[ok], V[ok], perda


def prever(d: pd.DataFrame, E: np.ndarray, V: np.ndarray, rng) -> tuple[dict, dict]:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    atrib = gen.dobras(sorted(d.evento_id.unique()), 5, "a1")
    dobra = d.evento_id.map(atrib).to_numpy()
    y = d.classe.to_numpy(int)
    pred = {k: np.full(len(d), np.nan) for k in BRACOS}
    custo = {k: 0.0 for k in BRACOS}
    for f in range(5):
        tr, te = dobra != f, dobra == f
        for nome, b in BRACOS.items():
            t0 = time.time()
            if b["tipo"] == "regra":
                pred[nome][te] = -d.hand_m.to_numpy(float)[te]
                continue
            X = d[list(b["vetor"])].to_numpy(float)
            blocos = []
            if "vizinhanca" in b.get("bloco", ""):
                Z = (V - V[tr].mean(0)) / (V[tr].std(0) + 1e-9)
                blocos.append(PCA(COMPONENTES, random_state=SEMENTE).fit(Z[tr]).transform(Z))
            if "embedding" in b.get("bloco", ""):
                blocos.append(PCA(COMPONENTES, random_state=SEMENTE).fit(E[tr]).transform(E))
            X = np.column_stack([X, *blocos]) if blocos else X
            if b["tipo"] == "gbm":
                m = HistGradientBoostingClassifier(random_state=SEMENTE, **gen.GBM).fit(X[tr], y[tr])
                pred[nome][te] = m.predict_proba(X[te])[:, 1]
            else:
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                m = LogisticRegression(C=1.0, max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
                pred[nome][te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
            custo[nome] += time.time() - t0
    return pred, {k: round(v, 2) for k, v in custo.items()}


def custo_do_encoder() -> dict:
    import ci_n1_encoder_por_ponto as n1
    rotulo, escolha = n1.escolher_checkpoint()
    man = json.loads((n1.ENC / f"encoder_{rotulo}_manifesto.json").read_text(encoding="utf-8"))
    chaves = [k for k in man if any(s in k for s in ("segundo", "tempo", "epoca", "parametro", "janelas", "amostras"))]
    return {"checkpoint": rotulo, **{k: man[k] for k in chaves}}


def executar() -> dict:
    h = gravar_pre_registro()
    d, E, V, perda = montar()
    rng = np.random.default_rng(SEMENTE)
    pred, custo = prever(d, E, V, rng)
    import ms1_posicao_topografica as ms1
    med = medir_com_regra(d, pred, {"A0_regra_hand"})
    comps = {}
    for cid, a, b in COMPARACOES:
        r = gen.comparar(d, pred[a], pred[b], rng)
        comps[cid] = {"a": a, "b": b, **r, "veredito": gen.veredito(r)}
    desc = gen.comparar(d, pred["A1e_base_piso_amostral"], pred["A1_base"], rng)
    try:
        enc = custo_do_encoder()
    except Exception as e:                                       # noqa: BLE001
        enc = {"erro": f"{type(e).__name__}: {e}"}
    r = {"tipo": "ms4_bracos_de_desenvolvimento", "pre_registro": h, "regime": "exploratorio",
         "linhas": int(len(d)), "eventos": int(d.evento_id.nunique()), "perda_por_classe": perda,
         "medidas": med, "comparacoes": comps,
         "descritivo_A1e_vs_A1": {**desc, "veredito": gen.veredito(desc)},
         "custo_de_ajuste_s": custo, "custo_do_encoder": enc,
         "nao_e": "nao e teste nem escolha de modelo servido"}
    r["hash_resumo"] = sha256_json(json.loads(json.dumps(r, default=float)))
    r["quando_utc"] = agora_utc()
    (SAIDA / "resultado.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    # predicoes fora da dobra: o MS5 (recusa decomposta) le daqui, sem reajustar
    d[["ponto_id", "evento_id", "classe", *BASE, *POS]].assign(
        dobra=d.evento_id.map(gen.dobras(sorted(d.evento_id.unique()), 5, "a1")),
        **{k: v for k, v in pred.items()}).to_parquet(SAIDA / "predicoes_fora_da_dobra.parquet", index=False)
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
    print(json.dumps({k: v["veredito"] for k, v in r["comparacoes"].items()}, ensure_ascii=False))
    print(f"GRAVADO={SAIDA / 'resultado.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
