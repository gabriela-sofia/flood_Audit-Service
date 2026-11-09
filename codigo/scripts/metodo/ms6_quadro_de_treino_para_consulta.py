"""MS6: para a pergunta de CONSULTA (territorio delimitado antes do evento), importa em que quadro se treina?

-------------------------------------------------------------------------
DE ONDE VEM
-------------------------------------------------------------------------

`verdade_observacional.POPULACOES_ALVO`: a vizinhanca da inundacao (GFD do
ds06) depende do desfecho para ser delimitada e nao e consultavel antes do
evento; o territorio consultavel (TC01) e uma grade sobre o tile de 1 grau,
escolhido sem olhar a mancha. MS2/MS3 mostraram que a extensao de onde vem o
negativo domina a transferencia entre quadros. Aqui o teste e SEMPRE o
territorio; muda so de onde vem o treino. AUC nao escolhe quadro: a pergunta
escolhe o teste, e o experimento mede o custo de treinar no quadro errado.

-------------------------------------------------------------------------
DESENHO (pre_registro.json antes de qualquer ajuste)
-------------------------------------------------------------------------

  dados     vizinhanca: ds06 GFD, carregar(TERRENO) + posicao v2 finita
            territorio: TC01 desenvolvimento (positivo e negativo observado,
            terreno e posicao v2 finitos)
  eventos   os 69 eventos GFD (os mesmos nos dois quadros); 5 dobras por evento
            (generalizacao_por_reservas.dobras, a1), a MESMA atribuicao
  teste     territorio dos eventos da dobra
  bracos    T0 regra -HAND
            T1 base (hand, twi, slope) treinada na vizinhanca
            T2 base treinada no territorio
            T3 base + pos_min_2km na vizinhanca
            T4 base + pos_min_2km no territorio
            T5 GBM sobre o vetor de T4, no territorio
  estimador logistica IRLS padronizada (ms1.ajustar, ate 60.000 linhas por
            ajuste, amostra simples); GBM = generalizacao_por_reservas.GBM
  comparacoes Q1 T2xT1, Q2 T4xT3, Q3 T4xT2, Q4 T2xT0, Q5 T5xT4
  leitura   generalizacao_por_reservas.veredito sobre a AUC dentro do evento
  descritivo os mesmos bracos lidos na vizinhanca (a outra pergunta);
            average precision (positivo raro no territorio)

Regime exploratorio: eventos conhecidos do ds06 nos dois quadros.

nao_e: nao e probabilidade de ocorrencia (o tile foi escolhido porque houve
evento) nem teste em territorio novo.
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
for pasta in ("metodo", "servico", "suscetibilidade", "geo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

import generalizacao_por_reservas as gen  # noqa: E402
import ms1_posicao_topografica as ms1  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "ms6-quadro-de-treino-para-consulta"
TC01 = REPO / "local_runs" / "tc01-territorio-consultavel" / "desenvolvimento" / "pontos.parquet"
SEMENTE = 20261008
BASE = ms1.BASE
VETOR_POS = BASE + ("pos_min_2km",)
BRACOS = {"T0_regra_hand": ("regra", None, None),
          "T1_base_vizinhanca": ("logistica", BASE, "vizinhanca"),
          "T2_base_territorio": ("logistica", BASE, "territorio"),
          "T3_pos_vizinhanca": ("logistica", VETOR_POS, "vizinhanca"),
          "T4_pos_territorio": ("logistica", VETOR_POS, "territorio"),
          "T5_gbm_territorio": ("gbm", VETOR_POS, "territorio")}
COMPARACOES = (("Q1", "T2_base_territorio", "T1_base_vizinhanca"), ("Q2", "T4_pos_territorio", "T3_pos_vizinhanca"),
               ("Q3", "T4_pos_territorio", "T2_base_territorio"), ("Q4", "T2_base_territorio", "T0_regra_hand"),
               ("Q5", "T5_gbm_territorio", "T4_pos_territorio"))
VISTO_ANTES = ("MS2/MS3 (07/10): extensao do negativo domina a transferencia; piloto TC01: AUC -HAND 0,79-0,89 em 3 "
               "eventos (sem modelo ajustado); manifesto TC01 desenvolvimento: contagens por estado")


def pre_registro() -> dict:
    return json.loads(json.dumps({
        "bracos": {k: [v[0], list(v[1]) if v[1] else None, v[2]] for k, v in BRACOS.items()},
        "comparacoes": [list(c) for c in COMPARACOES], "primario": None,
        "teste": "territorio (TC01 desenvolvimento) dos eventos da dobra", "dobras": "5 por evento, a1, mesma atribuicao",
        "estimador": "ms1.ajustar (IRLS, ate 60.000 linhas); GBM = generalizacao_por_reservas.GBM",
        "regra": "generalizacao_por_reservas.veredito", "descritivo": ["mesmos bracos na vizinhanca", "average precision"],
        "visto_antes": VISTO_ANTES, "semente": SEMENTE}))


def gravar_pre_registro(destino: Path = SAIDA / "pre_registro.json") -> str:
    corpo = pre_registro()
    h = sha256_json(corpo)
    if destino.exists():
        antigo = json.loads(destino.read_text(encoding="utf-8"))
        if antigo["hash"] != h:
            raise RuntimeError(f"pre-registro mudou ({antigo['hash']} -> {h}): crie MS7")
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


def carregar_quadros() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import posicao_no_ponto as pp
    d, _, _ = gen.carregar(gen.TERRENO)
    v = pp.juntar(d[d.fonte == "gfd"])
    v = v[np.isfinite(v[list(VETOR_POS)].to_numpy(float)).all(axis=1)].reset_index(drop=True)
    t = pd.read_parquet(TC01)
    antes = t[t.estado.isin(["positivo", "negativo_observado"])].assign(
        classe=lambda x: (x.estado == "positivo").astype(int))
    t = antes[np.isfinite(antes[list(VETOR_POS)].to_numpy(float)).all(axis=1)].reset_index(drop=True)
    return v, t, {"territorio": gen.perda_por_classe(antes, t)}


def ajustar(tipo: str, vetor, tr: pd.DataFrame):
    if tipo == "regra":
        return lambda te: -te.hand_m.to_numpy(float)
    if tipo == "logistica":
        return ms1.ajustar(tr, vetor)
    from sklearn.ensemble import HistGradientBoostingClassifier
    if len(tr) > gen.MAX_LINHAS_TREINO:
        tr = tr.sample(gen.MAX_LINHAS_TREINO, random_state=SEMENTE)
    m = HistGradientBoostingClassifier(random_state=SEMENTE, **gen.GBM).fit(tr[list(vetor)].to_numpy(float),
                                                                            tr.classe.to_numpy(int))
    return lambda te: m.predict_proba(te[list(vetor)].to_numpy(float))[:, 1]


def executar() -> dict:
    h = gravar_pre_registro()
    v, t, perda = carregar_quadros()
    eventos = sorted(set(v.evento_id) | set(t.evento_id))
    atrib = gen.dobras(eventos, 5, "a1")
    v["dobra"], t["dobra"] = v.evento_id.map(atrib), t.evento_id.map(atrib)
    quadros = {"vizinhanca": v, "territorio": t}
    pred = {q: {k: np.full(len(x), np.nan) for k in BRACOS} for q, x in quadros.items()}
    inicio = time.time()
    for f in range(5):
        for nome, (tipo, vetor, origem) in BRACOS.items():
            tr = quadros[origem][quadros[origem].dobra != f] if origem else None
            fn = ajustar(tipo, vetor, tr)
            for q, x in quadros.items():
                m = (x.dobra == f).to_numpy()
                pred[q][nome][m] = fn(x[m])
        print("dobra", f, round(time.time() - inicio), "s", flush=True)
    rng = np.random.default_rng(SEMENTE)
    comps = {}
    for cid, a, b in COMPARACOES:
        r = gen.comparar(t, pred["territorio"][a], pred["territorio"][b], rng)
        comps[cid] = {"a": a, "b": b, **r, "veredito": gen.veredito(r)}
    r = {"tipo": "ms6_quadro_de_treino_para_consulta", "pre_registro": h, "regime": "exploratorio",
         "linhas": {q: {"linhas": int(len(x)), "positivos": int(x.classe.sum()), "eventos": int(x.evento_id.nunique())}
                    for q, x in quadros.items()},
         "perda_por_classe": perda,
         "medidas_no_territorio": medir_com_regra(t, pred["territorio"], {"T0_regra_hand"}),
         "medidas_na_vizinhanca_descritivo": medir_com_regra(v, pred["vizinhanca"], {"T0_regra_hand"}),
         "comparacoes": comps, "segundos": round(time.time() - inicio),
         "nao_e": "nao e probabilidade de ocorrencia nem teste em territorio novo"}
    r["hash_resumo"] = sha256_json(json.loads(json.dumps(r, default=float)))
    r["quando_utc"] = agora_utc()
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
    print(json.dumps({k: v["veredito"] for k, v in r["comparacoes"].items()}, ensure_ascii=False))
    print(f"GRAVADO={SAIDA / 'resultado.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
