"""
EXT-UK-02 -- Selecao da area urbana do piloto, por evidencia.

Entrada : local_runs/ext-uk-01-aquisicao/recorded_flood_outlines.gpkg
          (31.672 poligonos, EPSG:4326, Environment Agency, OGL v3)

Objetivo: decidir QUAL recorte espacial da Inglaterra vira o piloto da linha
causal SUSC, usando contagem real de registros elegiveis por celula, e nao
escolha arbitraria de cidade. Nenhuma variavel e extraida aqui; nenhum modelo
e treinado; nenhum negativo e criado.

Criterio de elegibilidade (definido ANTES de olhar o resultado):
  E1. start_date presente
  E2. coastal_f = False  (exclui inundacao costeira -- mecanismo diferente)
  E3. tidal_f   = False  (exclui inundacao de mare -- mecanismo diferente)
  E4. ano >= 2000        (janela com cobertura de DEM/HAND/chuva global)

Dois recortes tematicos sao contados em separado, porque respondem a
perguntas cientificas diferentes:
  FLUVIAL  -- transbordamento de canal (analogo a Petropolis/Curitiba)
  PLUVIAL  -- drenagem urbana / esgoto / agua superficial (analogo a Recife)

A grade e de 10 km em EPSG:27700 (British National Grid), metrico. A unidade
de contagem primaria e `rec_grp_id` (grupo de registro = evento), nao o
poligono solto -- varios poligonos podem descrever o mesmo evento e conta-los
individualmente inflaria a densidade aparente.

Uso:
    python scripts/externo/ext_uk02_selecionar_area_piloto.py
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
INDIR = REPO / "local_runs" / "ext-uk-01-aquisicao"
OUTDIR = REPO / "local_runs" / "ext-uk-02-area-piloto"
GPKG = INDIR / "recorded_flood_outlines.gpkg"
CELL = 10_000  # metros, EPSG:27700
TOP = 15

# Classificacao tematica por `flood_src` (valores reais observados no inventario)
SRC_PLUVIAL = {"sewer", "drainage", "local drainage/surface water"}
SRC_FLUVIAL = {"main river", "ordinary watercourse", "ephemeral watercourse"}
CAUS_PLUVIAL = {"local drainage/surface water", "groundwater/high water table"}


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(GPKG)
    n_bruto = len(gdf)

    # --- elegibilidade ---
    gdf["start_date"] = pd.to_datetime(gdf["start_date"], errors="coerce", utc=True)
    gdf["ano"] = gdf["start_date"].dt.year

    def flag_false(col: str) -> pd.Series:
        return gdf[col].astype(str).str.lower().isin(["false", "0", "no"])

    elegivel = (
        gdf["start_date"].notna()
        & flag_false("coastal_f")
        & flag_false("tidal_f")
        & (gdf["ano"] >= 2000)
    )
    eg = gdf[elegivel].copy()

    # --- recorte tematico ---
    src = eg["flood_src"].astype(str).str.lower().str.strip()
    caus = eg["flood_caus"].astype(str).str.lower().str.strip()
    eg["tema"] = "outro"
    eg.loc[src.isin(SRC_FLUVIAL), "tema"] = "fluvial"
    eg.loc[src.isin(SRC_PLUVIAL) | caus.isin(CAUS_PLUVIAL), "tema"] = "pluvial"

    # --- grade metrica ---
    eg = eg.to_crs(27700)
    cen = eg.geometry.centroid
    eg["gx"] = (cen.x // CELL).astype(int)
    eg["gy"] = (cen.y // CELL).astype(int)
    eg["cell"] = eg["gx"].astype(str) + "_" + eg["gy"].astype(str)

    def resumo(sub: pd.DataFrame) -> pd.DataFrame:
        g = sub.groupby("cell").agg(
            n_outlines=("rec_out_id", "count"),
            n_eventos=("rec_grp_id", "nunique"),
            n_datas=("start_date", "nunique"),
            ano_min=("ano", "min"),
            ano_max=("ano", "max"),
            gx=("gx", "first"),
            gy=("gy", "first"),
        )
        return g.sort_values("n_eventos", ascending=False)

    tabelas = {
        "TODOS": resumo(eg),
        "FLUVIAL": resumo(eg[eg["tema"] == "fluvial"]),
        "PLUVIAL": resumo(eg[eg["tema"] == "pluvial"]),
    }

    for nome, tab in tabelas.items():
        tab.to_csv(OUTDIR / f"celulas_{nome.lower()}.csv")

    eg.drop(columns=["ano"]).to_file(OUTDIR / "elegiveis.gpkg", driver="GPKG")

    # --- retorno sintetizado ---
    print("===EXT-UK-02===")
    print(f"BRUTO={n_bruto} ELEGIVEIS={len(eg)}")
    print("TEMA=" + "; ".join(f"{k}:{v}" for k, v in eg["tema"].value_counts().items()))
    print(f"GRADE={CELL//1000}km EPSG=27700 CELULAS_OCUPADAS={eg['cell'].nunique()}")

    for nome, tab in tabelas.items():
        print(f"---TOP_{nome}---")
        for cell, r in tab.head(TOP).iterrows():
            # bbox aproximado da celula em BNG, util pra recorte posterior
            x0, y0 = int(r["gx"]) * CELL, int(r["gy"]) * CELL
            print(
                f"{cell} eventos={int(r['n_eventos'])} outlines={int(r['n_outlines'])} "
                f"datas={int(r['n_datas'])} anos={int(r['ano_min'])}-{int(r['ano_max'])} "
                f"bng=({x0},{y0})-({x0+CELL},{y0+CELL})"
            )

    resumo_json = {
        "bruto": n_bruto,
        "elegiveis": len(eg),
        "criterio": ["start_date presente", "coastal=False", "tidal=False", "ano>=2000"],
        "grade_m": CELL,
        "tema": eg["tema"].value_counts().to_dict(),
        "top": {k: v.head(TOP).reset_index().to_dict("records") for k, v in tabelas.items()},
    }
    (OUTDIR / "selecao.json").write_text(
        json.dumps(resumo_json, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
