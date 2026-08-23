"""
FT-UK-03 -- Amostragem pareada e extracao das features topograficas.

Produz a tabela de pontos do piloto ingles, com as quatro features
topo-hidrologicas do v12 que ja podem ser calculadas. As duas features de
chuva entram em etapa separada (FT-UK-04), quando a serie CHIRPS estiver no
disco -- de proposito, para que uma pendencia de download nao bloqueie o que
ja da para fazer.

DECISOES DE AMOSTRAGEM, e o porque de cada uma:

1. TETO DE PONTOS POR EVENTO (`MAX_POR_EVENTO`).
   A regra U1-U3 do documento de adjudicacao declara o EVENTO como unidade de
   informacao independente. Sem teto, um unico evento grande geraria milhares
   de pontos e dominaria o ajuste -- pseudorreplicacao pura, que ja comprometeu
   o A/B do DINO neste projeto. O teto e a medida direta contra isso.

2. PAREAMENTO POR COBERTURA DO SOLO (criterio N4).
   O FT-UK-02 mediu: 22,4% dos positivos sao area construida contra 11,8% dos
   candidatos a negativo -- razao 1,90. Amostrar negativo sem parear ensinaria
   ao modelo "construido = enchente". A amostragem negativa aqui reproduz a
   distribuicao de classes ESA WorldCover observada nos positivos.

3. GRUPO DE VALIDACAO CRUZADA (`grupo_cv`).
   Positivo recebe o `rec_grp_id` do evento. Negativo recebe um bloco espacial
   de 5 km. Assim o GroupKFold nao separa um evento entre treino e teste, e
   tambem nao coloca negativos vizinhos dos dois lados da particao. Relatar so
   o n de pontos seria relato incompleto (regra U3): o numero de eventos vai
   junto na saida.

O QUE ESTE SCRIPT NAO FAZ: nao treina, nao calcula metrica de desempenho, nao
promove nada a rotulo definitivo. A coluna `label` aqui e candidata, e o
arquivo de saida diz isso no nome e no cabecalho.

Uso:
    python scripts/externo/ft_uk03_amostrar_e_extrair.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_store import ler_bbox  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
STORE = RUNS / "geostore"
RAST = RUNS / "ft-uk-01-rasters"
MASC = RUNS / "ft-uk-02-mascaras"
OUT = RUNS / "ft-uk-03-amostra"

AOI_BNG = (350_000, 350_000, 400_000, 450_000)
MAX_POR_EVENTO = 40      # teto anti-pseudorreplicacao
BLOCO_NEG_M = 5_000      # bloco espacial que agrupa negativos na CV
RAZAO_NEG = 1.0          # negativos por positivo
SEMENTE = 20260807


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()
    print("===FT-UK-03===")

    with rasterio.open(RAST / "elevation_m.tif") as src:
        forma, transform, crs = (src.height, src.width), src.transform, src.crs
    res = transform.a

    def ler(p: Path) -> np.ndarray:
        with rasterio.open(p) as s:
            return s.read(1)

    elev = ler(RAST / "elevation_m.tif")
    slope = ler(RAST / "slope_deg.tif")
    twi = ler(RAST / "twi_dinf.tif")
    hand = ler(RAST / "hand_m.tif")
    classe = ler(MASC / "classe_adjudicacao.tif")
    wc = ler(MASC / "worldcover.tif")
    dist = ler(MASC / "dist_inundavel.tif")
    print(f"GRADE={forma} res={res}m")

    # ---------------- positivos, com identidade de evento ----------------
    eleg = ler_bbox(STORE / "outlines_elegiveis.parquet", AOI_BNG).reset_index(drop=True)
    eleg["_idx"] = np.arange(1, len(eleg) + 1)
    idx_rast = rasterize(
        ((g, i) for g, i in zip(eleg.geometry, eleg["_idx"])
         if g is not None and not g.is_empty),
        out_shape=forma, transform=transform, fill=0, dtype="int32",
        all_touched=True,
    )
    linhas, colunas = np.nonzero(idx_rast > 0)
    idx = idx_rast[linhas, colunas]
    pos = pd.DataFrame({"row": linhas, "col": colunas, "_idx": idx})
    atrib = eleg.drop(columns="geometry")[
        ["_idx", "rec_out_id", "rec_grp_id", "start_date", "tema", "flood_src", "flood_caus"]
    ]
    pos = pos.merge(atrib, on="_idx", how="left")
    print(f"CELULAS_POSITIVAS_BRUTAS={len(pos):,} eventos={pos.rec_grp_id.nunique()}")

    # teto por evento
    pos = (pos.groupby("rec_grp_id", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), MAX_POR_EVENTO), random_state=SEMENTE)))
    pos = pos.reset_index(drop=True)
    pos["label"] = 1
    pos["grupo_cv"] = "EV_" + pos["rec_grp_id"].astype(str)
    print(f"POSITIVOS_APOS_TETO={len(pos):,} (teto={MAX_POR_EVENTO}/evento) "
          f"eventos={pos.rec_grp_id.nunique()}")
    print("POS_TEMA=" + "; ".join(f"{k}:{v}" for k, v in pos.tema.value_counts().items()))

    # ---------------- negativos pareados por cobertura ----------------
    wc_pos = pos.assign(wc=wc[pos["row"], pos["col"]])["wc"]
    alvo = (wc_pos.value_counts(normalize=True) * len(pos) * RAZAO_NEG).round().astype(int)
    print("ALVO_PAREAMENTO=" + "; ".join(f"{int(k)}:{v}" for k, v in alvo.items()))

    lin_n, col_n = np.nonzero(classe == 3)
    wc_neg_todos = wc[lin_n, col_n]
    escolhidos_l, escolhidos_c = [], []
    for cls, quanto in alvo.items():
        disponiveis = np.nonzero(wc_neg_todos == cls)[0]
        if len(disponiveis) == 0:
            print(f"  AVISO classe {int(cls)} sem candidato negativo -- pareamento incompleto")
            continue
        take = min(quanto, len(disponiveis))
        sel = rng.choice(disponiveis, size=take, replace=False)
        escolhidos_l.append(lin_n[sel])
        escolhidos_c.append(col_n[sel])
        if take < quanto:
            print(f"  AVISO classe {int(cls)}: pedido {quanto}, disponivel {take}")
    ln = np.concatenate(escolhidos_l)
    cn = np.concatenate(escolhidos_c)
    neg = pd.DataFrame({"row": ln, "col": cn})
    neg["label"] = 0
    neg["rec_out_id"] = pd.NA
    neg["rec_grp_id"] = pd.NA
    neg["start_date"] = pd.NaT
    neg["tema"] = "negativo"
    neg["flood_src"] = pd.NA
    neg["flood_caus"] = pd.NA
    x_neg = transform.c + (cn + 0.5) * res
    y_neg = transform.f - (ln + 0.5) * res
    neg["grupo_cv"] = ("NEG_" + (x_neg // BLOCO_NEG_M).astype(int).astype(str)
                       + "_" + (y_neg // BLOCO_NEG_M).astype(int).astype(str))
    print(f"NEGATIVOS={len(neg):,} blocos_espaciais={neg.grupo_cv.nunique()}")

    # ---------------- juncao e extracao ----------------
    df = pd.concat([pos.drop(columns="_idx"), neg], ignore_index=True)
    r, c = df["row"].to_numpy(), df["col"].to_numpy()
    df["x_bng"] = transform.c + (c + 0.5) * res
    df["y_bng"] = transform.f - (r + 0.5) * res
    df["elevation_m"] = elev[r, c]
    df["slope_deg"] = slope[r, c]
    df["twi_dinf"] = twi[r, c]
    df["hand_m"] = hand[r, c]
    df["worldcover"] = wc[r, c]
    df["dist_inundavel_m"] = dist[r, c]

    antes = len(df)
    df = df.dropna(subset=["elevation_m", "slope_deg", "twi_dinf", "hand_m"])
    if len(df) < antes:
        print(f"DESCARTADOS_SEM_FEATURE={antes - len(df)}")

    # lat/lon para juncao futura com a chuva (que esta em WGS84)
    from pyproj import Transformer
    tr = Transformer.from_crs(27700, 4326, always_xy=True)
    df["lon"], df["lat"] = tr.transform(df["x_bng"].values, df["y_bng"].values)

    df["ponto_id"] = [f"UK_{i:06d}" for i in range(len(df))]
    df["fonte_label"] = np.where(df["label"] == 1,
                                 "EA_recorded_flood_outlines", "adjudicacao_N1_N4")

    colunas = ["ponto_id", "label", "tema", "grupo_cv", "rec_grp_id", "rec_out_id",
               "start_date", "flood_src", "flood_caus", "x_bng", "y_bng", "lon", "lat",
               "elevation_m", "slope_deg", "twi_dinf", "hand_m", "worldcover",
               "dist_inundavel_m", "fonte_label", "row", "col"]
    df = df[colunas]
    saida = OUT / "amostra_candidata_uk_v1.csv"
    df.to_csv(saida, index=False)

    # ---------------- verificacao ----------------
    print("---VERIFICACAO---")
    print(f"N_TOTAL={len(df):,} pos={int((df.label==1).sum()):,} neg={int((df.label==0).sum()):,}")
    print(f"N_EVENTOS_INDEPENDENTES={df.loc[df.label==1,'rec_grp_id'].nunique()}")
    print(f"N_GRUPOS_CV={df.grupo_cv.nunique()}")
    comp_pos = df[df.label == 1].worldcover.value_counts(normalize=True).mul(100).round(1)
    comp_neg = df[df.label == 0].worldcover.value_counts(normalize=True).mul(100).round(1)
    print("WC_POS=" + "; ".join(f"{int(k)}:{v}%" for k, v in comp_pos.items()))
    print("WC_NEG=" + "; ".join(f"{int(k)}:{v}%" for k, v in comp_neg.items()))
    print("---SEPARACAO_UNIVARIADA (mediana pos vs neg)---")
    for f in ("elevation_m", "slope_deg", "twi_dinf", "hand_m"):
        mp = df.loc[df.label == 1, f].median()
        mn = df.loc[df.label == 0, f].median()
        print(f"  {f}: pos={mp:.2f} neg={mn:.2f}")

    (OUT / "resumo.json").write_text(json.dumps({
        "n_total": int(len(df)),
        "n_positivos": int((df.label == 1).sum()),
        "n_negativos": int((df.label == 0).sum()),
        "n_eventos_independentes": int(df.loc[df.label == 1, "rec_grp_id"].nunique()),
        "n_grupos_cv": int(df.grupo_cv.nunique()),
        "max_por_evento": MAX_POR_EVENTO,
        "bloco_neg_m": BLOCO_NEG_M,
        "razao_neg": RAZAO_NEG,
        "semente": SEMENTE,
        "worldcover_pos_pct": {int(k): float(v) for k, v in comp_pos.items()},
        "worldcover_neg_pct": {int(k): float(v) for k, v in comp_neg.items()},
        "aviso": ("Amostra CANDIDATA. `label` nao e rotulo promovido. Faltam as "
                  "duas features de chuva (FT-UK-04). Nenhum modelo treinado."),
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SAIDA={saida.name} TEMPO={round(time.time()-t0,1)}s")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
