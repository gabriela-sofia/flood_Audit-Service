"""
EXT-UK-03 -- Dimensionamento e localizacao da AOI do piloto.

Entrada : local_runs/ext-uk-02-area-piloto/elegiveis.gpkg
          (16.667 outlines elegiveis: com data, nao costeiro, nao de mare, >=2000)

Problema que este script resolve: o EXT-UK-02 mostrou que a celula de 10 km
mais densa tem apenas 36 eventos distintos. A densidade por celula pequena e
insuficiente, entao e preciso descobrir (a) qual escala de janela entrega
volume util e (b) ONDE ficam os agrupamentos -- hoje so temos coordenada de
grade britanica, e nomear cidade por deducao seria chute.

O que faz:
  1. Testa tres escalas de janela (10, 25, 50 km) e reporta quanto cada uma
     entrega no melhor recorte -- para a escolha de escala ser medida.
  2. Para as melhores janelas, resolve a LOCALIZACAO a partir do proprio dado:
     centroide em lat/lon (EPSG:4326) e os termos mais frequentes do campo
     `name`, que traz rio e local de origem.
  3. Separa por tema (fluvial / pluvial), porque a pergunta cientifica difere:
     pluvial e o analogo de Recife; fluvial e o analogo de Petropolis/Curitiba.

NAO faz: nenhuma extracao de variavel, nenhum treino, nenhuma criacao de
negativo. So descreve o dado ja adquirido.

Uso:
    python scripts/externo/ext_uk03_dimensionar_aoi.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
INDIR = REPO / "local_runs" / "ext-uk-02-area-piloto"
OUTDIR = REPO / "local_runs" / "ext-uk-03-aoi"
GPKG = INDIR / "elegiveis.gpkg"
ESCALAS = [10_000, 25_000, 50_000]
TOP = 8

RUIDO = {
    "aerial", "photography", "photo", "survey", "surveyed", "modelo", "modelled",
    "outline", "flood", "flooding", "event", "the", "of", "and", "at", "from",
    "unknown", "data", "record", "recorded", "extent", "map", "mapping", "na",
}


def tokens(nomes: pd.Series, n: int = 6) -> list[tuple[str, int]]:
    """Termos mais frequentes do campo `name`, limpos de ruido de metadado."""
    c: Counter = Counter()
    for nome in nomes.dropna().astype(str):
        # descarta datas, numeros e sufixos tipo Aerial(251)
        limpo = re.sub(r"\(.*?\)", " ", nome)
        limpo = re.sub(r"\d", " ", limpo)
        for parte in re.split(r"[_\-/,;]+", limpo):
            for palavra in parte.split():
                p = palavra.strip().lower()
                if len(p) >= 3 and p not in RUIDO and p.isalpha():
                    c[p] += 1
    return c.most_common(n)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(GPKG)
    if gdf.crs is None or gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs(27700)

    cen = gdf.geometry.centroid
    gdf["_x"], gdf["_y"] = cen.x, cen.y
    gdf["ano"] = pd.to_datetime(gdf["start_date"], errors="coerce", utc=True).dt.year

    # centroide em lat/lon para localizar de verdade
    cen_ll = gpd.GeoSeries(cen, crs=27700).to_crs(4326)
    gdf["_lon"], gdf["_lat"] = cen_ll.x, cen_ll.y

    resultado: dict = {"escalas": {}}
    print("===EXT-UK-03===")
    print(f"ELEGIVEIS={len(gdf)}")

    for esc in ESCALAS:
        gdf["_c"] = (
            (gdf["_x"] // esc).astype(int).astype(str)
            + "_"
            + (gdf["_y"] // esc).astype(int).astype(str)
        )
        agg = gdf.groupby("_c").agg(
            outlines=("rec_out_id", "count"),
            eventos=("rec_grp_id", "nunique"),
            datas=("start_date", "nunique"),
            ano_min=("ano", "min"),
            ano_max=("ano", "max"),
            lat=("_lat", "mean"),
            lon=("_lon", "mean"),
        )
        # composicao tematica por janela
        tema = gdf.pivot_table(
            index="_c", columns="tema", values="rec_out_id", aggfunc="count"
        ).fillna(0)
        agg = agg.join(tema, how="left").fillna(0)
        agg = agg.sort_values("eventos", ascending=False)
        agg.to_csv(OUTDIR / f"janelas_{esc//1000}km.csv")

        print(f"---ESCALA_{esc//1000}km--- janelas_ocupadas={len(agg)}")
        registros = []
        for cid, r in agg.head(TOP).iterrows():
            sub = gdf[gdf["_c"] == cid]
            tk = "/".join(t for t, _ in tokens(sub["name"], 4))
            plu = int(r.get("pluvial", 0))
            flu = int(r.get("fluvial", 0))
            print(
                f"  {cid} ev={int(r['eventos'])} out={int(r['outlines'])} "
                f"datas={int(r['datas'])} anos={int(r['ano_min'])}-{int(r['ano_max'])} "
                f"fluv={flu} pluv={plu} @({r['lat']:.3f},{r['lon']:.3f}) [{tk}]"
            )
            registros.append(
                {
                    "janela": cid, "eventos": int(r["eventos"]), "outlines": int(r["outlines"]),
                    "datas": int(r["datas"]), "ano_min": int(r["ano_min"]), "ano_max": int(r["ano_max"]),
                    "fluvial": flu, "pluvial": plu,
                    "lat": round(float(r["lat"]), 4), "lon": round(float(r["lon"]), 4),
                    "termos": tk,
                }
            )
        resultado["escalas"][f"{esc//1000}km"] = registros

    # --- melhor janela para o recorte PLUVIAL (analogo de Recife) ---
    esc = 50_000
    gdf["_c"] = (
        (gdf["_x"] // esc).astype(int).astype(str) + "_" + (gdf["_y"] // esc).astype(int).astype(str)
    )
    plu = gdf[gdf["tema"] == "pluvial"]
    agg_p = plu.groupby("_c").agg(
        outlines=("rec_out_id", "count"), eventos=("rec_grp_id", "nunique"),
        datas=("start_date", "nunique"), ano_min=("ano", "min"), ano_max=("ano", "max"),
        lat=("_lat", "mean"), lon=("_lon", "mean"),
    ).sort_values("eventos", ascending=False)
    agg_p.to_csv(OUTDIR / "janelas_50km_pluvial.csv")
    print("---PLUVIAL_50km---")
    plu_reg = []
    for cid, r in agg_p.head(TOP).iterrows():
        sub = plu[plu["_c"] == cid]
        tk = "/".join(t for t, _ in tokens(sub["name"], 4))
        print(
            f"  {cid} ev={int(r['eventos'])} out={int(r['outlines'])} "
            f"anos={int(r['ano_min'])}-{int(r['ano_max'])} "
            f"@({r['lat']:.3f},{r['lon']:.3f}) [{tk}]"
        )
        plu_reg.append({"janela": cid, "eventos": int(r["eventos"]), "outlines": int(r["outlines"]),
                        "lat": round(float(r["lat"]), 4), "lon": round(float(r["lon"]), 4), "termos": tk})
    resultado["pluvial_50km"] = plu_reg

    (OUTDIR / "dimensionamento.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
