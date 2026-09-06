"""
DS-02 -- Conjunto de treino separado por MECANISMO, nao por pais.

POR QUE A SEPARACAO MUDOU:
o `ds01` agrupava por regiao. Isso parecia neutro e nao era. Ao re-extrair os
278 pontos de Recife descobriu-se que o coeficiente de HAND no v12 e -0,0001
com p = 0,978 -- ausente, nao fraco -- e que o contraste bruto esta invertido:
positivos a 8,07 m acima da drenagem contra 5,25 m dos negativos.

Nao e erro. Recife e enchente PLUVIAL URBANA: a agua vem da chuva excedendo a
capacidade de drenagem, nao do canal subindo. HAND mede a lamina necessaria
para o canal alcancar o ponto -- num evento pluvial, nao e o mecanismo. Ja o
conjunto externo e fluvial e de enxurrada em vale, onde HAND e exatamente o
mecanismo (coeficiente -3,30 na planicie, IC longe de zero).

Sao fenomenos diferentes com o mesmo nome. Um modelo unico tentaria conciliar
um subconjunto onde HAND nao importa com outro onde HAND e a variavel
principal, e nao descreveria nenhum dos dois.

A TABELA DE MECANISMO abaixo e declarada com a evidencia que a sustenta, e nao
inferida de desempenho. Classificar por resultado seria circular.

O CASO DO UFO -- por que 215 grupos ficam de fora:
a Urban Flood Observations declara, na propria documentacao, cobrir drivers
pluvial, fluvial E mare de tempestade em 14 eventos, sem separacao por chip. E
o maior numero de grupos do conjunto e nao e atribuivel a um mecanismo.
Inclui-lo num modelo fluvial contaminaria o subconjunto. Fica marcado como
MISTO_NAO_SEPARAVEL e disponivel como conjunto de robustez -- onde a pergunta
e justamente se o modelo aguenta mistura.

PROCEDENCIA DAS FEATURES: para a trilha fluvial usa-se a cadeia harmonizada
(WhiteboxTools 30 m, limiar de canal em 0,1123 km2). Fonte que ainda nao foi
harmonizada entra marcada com `cadeia=global` e NAO deve ser misturada com a
harmonizada no mesmo ajuste -- o script conta e avisa, mas nao impede.

NAO faz: nao treina, nao decide qual mecanismo e mais importante, nao
reclassifica fonte por resultado de modelo.

Uso:
    python scripts/treino/ds02_montar_por_mecanismo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROJETO = REPO.parent / "PROJETO"
RUNS = REPO / "local_runs"
OUT = RUNS / "ds-02-mecanismo"

DS01 = RUNS / "ds-01-multirregiao" / "dataset_multirregiao_v1.csv"
HARM_EXT = RUNS / "ter-02-comparacao" / "dataset_serra_harmonizado.csv"
HARM_BR = RUNS / "ter-03-brasil-harmonizado"

FLUVIAL = "FLUVIAL_ENXURRADA"
PLUVIAL = "PLUVIAL_URBANO"
MISTO = "MISTO_NAO_SEPARAVEL"

# Classificacao declarada. Ver ext_resolucao_e_mecanismo_decisao_v1.md secao 4.
MECANISMO = {
    "UK_noroeste": (FLUVIAL, "Recorded Flood Outlines da EA; transbordamento de curso d'agua"),
    "nivel1_sen1floods11_handlabeled": (FLUVIAL, "a base declara dominancia de cheia fluvial em area aberta"),
    "nivel1_ufo_urban_flood_observations": (MISTO, "a base declara drivers pluvial, fluvial e mare de tempestade sem separacao por chip"),
    "curitiba": (FLUVIAL, "fluvial urbano em planalto"),
    "recife": (PLUVIAL, "HAND com coef -0,0001 (p=0,978) e contraste invertido; chuva antecedente domina"),
}
# toda ativacao CEMS e fluvial/enxurrada por construcao (observedEventA de Flood/Storm)
PREFIXO_FLUVIAL = "analogo_EMSR"

FEATURES_TOPO = ["elevation_m", "slope_deg", "hand_m", "twi_dinf"]


def classificar(regiao: str) -> tuple[str, str]:
    if regiao in MECANISMO:
        return MECANISMO[regiao]
    if str(regiao).startswith(PREFIXO_FLUVIAL):
        return FLUVIAL, "ativacao CEMS de Flood/Storm com observedEventA"
    return "NAO_CLASSIFICADO", "sem regra declarada"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("===DS02_POR_MECANISMO===")

    if not DS01.exists():
        print(f"ABORTADO: {DS01} ausente.")
        return 1

    partes = []

    # ---- externo (ds01) ----
    d = pd.read_csv(DS01, low_memory=False)
    d = d[d.classe.isin([0, 1])].copy()
    d["fonte"] = d["regiao"]
    d["cadeia"] = "global"

    # substitui pelas features harmonizadas onde existirem
    if HARM_EXT.exists():
        h = pd.read_csv(HARM_EXT, low_memory=False)
        cols = {f: f"{f}_wbt" for f in FEATURES_TOPO if f"{f}_wbt" in h.columns}
        if cols and "ponto_id" in h.columns and "ponto_id" in d.columns:
            sub = h[["ponto_id"] + list(cols.values())].copy()
            d = d.merge(sub, on="ponto_id", how="left")
            tem = d[list(cols.values())].notna().all(axis=1)
            for f, c in cols.items():
                d.loc[tem, f] = d.loc[tem, c]
            d.loc[tem, "cadeia"] = "wbt30"
            d = d.drop(columns=list(cols.values()))
            print(f"EXTERNO: {int(tem.sum()):,} pontos com cadeia harmonizada, "
                  f"{int((~tem).sum()):,} ainda em cadeia global")
    partes.append(d[["ponto_id", "fonte", "classe", "grupo_cv", "tipo_negativo",
                     "cadeia", "data_evento"] + FEATURES_TOPO])

    # ---- brasileiras (ter03) ----
    mapa_br = {"elevation_m": "elevation_m_wbt30", "slope_deg": "slope_deg_wbt30",
               "hand_m": "hand_m_dinf_wbt30", "twi_dinf": "twi_dinf_wbt30"}
    for nome, grupo_col, id_col in (("recife", "point_id", "point_id"),
                                    ("curitiba", "observation_unit_key", "point_id")):
        p = HARM_BR / f"{nome}_harmonizado.csv"
        if not p.exists():
            print(f"AVISO: {p} ausente -- {nome} fica fora. Rode ter03.")
            continue
        b = pd.read_csv(p, low_memory=False)
        b = b[b.label.isin([0, 1])].copy()
        novo = pd.DataFrame({
            "ponto_id": b[id_col].astype(str),
            "fonte": nome,
            "classe": b.label.astype(int),
            "grupo_cv": b[grupo_col].astype(str),
            "tipo_negativo": b.get("negative_source_type"),
            "cadeia": "wbt30",
            "data_evento": b.get("event_date"),
        })
        for f, c in mapa_br.items():
            novo[f] = b[c] if c in b.columns else np.nan
        partes.append(novo)
        print(f"BRASIL: {nome} {len(novo):,} linhas, {novo.grupo_cv.nunique():,} grupos")

    df = pd.concat(partes, ignore_index=True)
    df[["mecanismo", "mecanismo_evidencia"]] = df["fonte"].apply(
        lambda r: pd.Series(classificar(r)))

    print("\n--- POR MECANISMO ---")
    r = (df.groupby("mecanismo")
           .agg(n=("classe", "size"), pos=("classe", "sum"),
                grupos=("grupo_cv", "nunique"), fontes=("fonte", "nunique"))
           .assign(neg=lambda x: x.n - x.pos))
    r["epv_4feat"] = (r.grupos / 4).round(1)
    print(r[["n", "pos", "neg", "grupos", "fontes", "epv_4feat"]].to_string())

    print("\n--- PROCEDENCIA DA CADEIA, dentro de cada mecanismo ---")
    print(df.groupby(["mecanismo", "cadeia"]).size().to_string())

    nc = df[df.mecanismo == "NAO_CLASSIFICADO"]
    if len(nc):
        print(f"\nATENCAO: {len(nc):,} pontos NAO_CLASSIFICADOS em "
              f"{nc.fonte.nunique()} fontes: {sorted(nc.fonte.unique())[:6]}")

    df.to_csv(OUT / "dataset_por_mecanismo.csv", index=False)
    for mec in (FLUVIAL, PLUVIAL, MISTO):
        s = df[df.mecanismo == mec]
        if len(s):
            s.to_csv(OUT / f"dataset_{mec.lower()}.csv", index=False)

    (OUT / "resumo.json").write_text(json.dumps({
        "tabela_mecanismo": {k: {"mecanismo": v[0], "evidencia": v[1]}
                             for k, v in MECANISMO.items()},
        "regra_prefixo": {PREFIXO_FLUVIAL: FLUVIAL},
        "por_mecanismo": json.loads(r.to_json(orient="index")),
        "cadeia": json.loads(df.groupby(["mecanismo", "cadeia"]).size()
                             .unstack(fill_value=0).to_json(orient="index")),
        "nota_ufo": ("215 grupos marcados MISTO_NAO_SEPARAVEL: a base declara "
                     "drivers pluvial, fluvial e mare sem separacao por chip"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
