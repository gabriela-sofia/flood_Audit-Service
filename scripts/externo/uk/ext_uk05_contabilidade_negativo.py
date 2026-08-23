"""
EXT-UK-05 -- Contabilidade do espaco disponivel para negativo na AOI.

Pergunta que este script responde, e so ela: depois de excluir toda area com
inundacao REGISTRADA e toda area modelada como inundavel, sobra espaco
suficiente na AOI para construir uma classe negativa? E de que tamanho?

Isso precisa ser respondido ANTES de qualquer decisao de amostragem, porque
se a resposta for "sobra pouco" ou "sobra so area rural", a AOI esta errada e
nao adianta seguir.

Metodo: grade regular de pontos sobre a AOI, classificada por sobreposicao.
A grade e usada em vez de algebra de poligonos porque (a) e a unidade em que
o dataset SUSC realmente trabalha -- ponto, nao area -- e (b) evita uniao de
17 mil poligonos, que seria cara e desnecessaria.

Classes produzidas (nenhuma delas e promovida a rotulo aqui):
  POS_CAND  -- dentro de Recorded Flood Outline elegivel  -> candidato a positivo
  EXCLUIDO  -- fora de RFO mas dentro de Flood Zone       -> zona cinzenta,
               fisicamente plausivel de inundar sem registro. E exatamente a
               fonte de contaminacao que a Environment Agency adverte quando
               escreve que ausencia de outline nao significa ausencia de
               inundacao.
  NEG_CAND  -- fora de RFO e fora de Flood Zone           -> candidato a negativo

Entradas:
  local_runs/ext-uk-02-area-piloto/elegiveis.gpkg           (16.667 outlines)
  local_runs/ext-uk-04-flood-zones/Flood_Zones_.../page_*   (17.199 poligonos)

NAO faz: nao promove nada a rotulo, nao extrai variavel, nao treina. So conta.

Uso:
    python scripts/externo/ext_uk05_contabilidade_negativo.py
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
RFO = RUNS / "ext-uk-02-area-piloto" / "elegiveis.gpkg"
FZ_DIR = RUNS / "ext-uk-04-flood-zones" / "Flood_Zones_2_3_Rivers_and_Sea"
OUTDIR = RUNS / "ext-uk-05-contabilidade"

AOI = (350_000, 350_000, 400_000, 450_000)  # EPSG:27700
PASSO = 200  # metros entre pontos da grade


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("===EXT-UK-05===")

    aoi_geom = box(*AOI)
    area_aoi_km2 = aoi_geom.area / 1e6
    print(f"AOI_BNG={AOI} area={area_aoi_km2:.0f}km2 passo={PASSO}m")

    # --- Recorded Flood Outlines elegiveis, recortados na AOI ---
    rfo = gpd.read_file(RFO)
    if rfo.crs is None or rfo.crs.to_epsg() != 27700:
        rfo = rfo.to_crs(27700)
    rfo = rfo[rfo.intersects(aoi_geom)].copy()
    print(f"RFO_NA_AOI={len(rfo)}")
    if "tema" in rfo.columns:
        print("RFO_TEMA=" + "; ".join(f"{k}:{v}" for k, v in rfo["tema"].value_counts().items()))

    # --- grade de pontos ---
    xs = np.arange(AOI[0] + PASSO / 2, AOI[2], PASSO)
    ys = np.arange(AOI[1] + PASSO / 2, AOI[3], PASSO)
    xx, yy = np.meshgrid(xs, ys)
    grade = gpd.GeoDataFrame(
        {"gid": np.arange(xx.size)},
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()),
        crs=27700,
    )
    print(f"GRADE_PONTOS={len(grade)}")

    # --- classificacao ---
    em_rfo = gpd.sjoin(grade, rfo[["geometry"]], how="inner", predicate="within")
    ids_rfo = set(em_rfo["gid"])
    print(f"PONTOS_EM_RFO={len(ids_rfo)}")

    # Flood Zones processadas pagina a pagina: carregar as 35 paginas de uma
    # vez estoura a memoria (296MB de GeoJSON viram varios GB em objeto Python).
    # A contagem so precisa do conjunto de gids atingidos, entao acumula-se o
    # conjunto e descarta-se cada pagina.
    ids_fz: set = set()
    fz_total = 0
    zonas: Counter = Counter()
    origens: Counter = Counter()
    for i, p in enumerate(sorted(glob.glob(str(FZ_DIR / "page_*.geojson"))), 1):
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        feats_pg = d.get("features", [])
        if not feats_pg:
            continue
        fz_pg = gpd.GeoDataFrame.from_features(feats_pg, crs=4326).to_crs(27700)
        fz_total += len(fz_pg)
        if "flood_zone" in fz_pg.columns:
            zonas.update(fz_pg["flood_zone"].astype(str))
        if "origin" in fz_pg.columns:
            origens.update(fz_pg["origin"].astype(str))
        j = gpd.sjoin(grade, fz_pg[["geometry"]], how="inner", predicate="within")
        ids_fz.update(j["gid"].tolist())
        del fz_pg, feats_pg, d, j
        if i % 10 == 0:
            print(f"  FZ pagina {i} acumulado_pontos={len(ids_fz)}")
    print(f"FZ_TOTAL={fz_total}")
    print("FZ_ZONA=" + "; ".join(f"{k}:{v}" for k, v in zonas.most_common()))
    print("FZ_ORIGEM=" + "; ".join(f"{k[:38]}:{v}" for k, v in origens.most_common(6)))
    print(f"PONTOS_EM_FZ={len(ids_fz)}")

    grade["classe"] = np.where(
        grade["gid"].isin(ids_rfo), "POS_CAND",
        np.where(grade["gid"].isin(ids_fz), "EXCLUIDO", "NEG_CAND"),
    )
    cont = Counter(grade["classe"])
    n = len(grade)
    for k in ("POS_CAND", "EXCLUIDO", "NEG_CAND"):
        v = cont.get(k, 0)
        print(f"{k}={v} ({100*v/n:.1f}%) ~{v*(PASSO**2)/1e6:.0f}km2")

    # --- quanto do negativo esta em zona de RFO por tema ---
    # (positivos por tema, para dimensionar o desenho amostral)
    if "tema" in rfo.columns:
        for tema in ("fluvial", "pluvial"):
            sub = rfo[rfo["tema"] == tema]
            if len(sub) == 0:
                continue
            j = gpd.sjoin(grade, sub[["geometry"]], how="inner", predicate="within")
            print(f"POS_CAND_{tema.upper()}={len(set(j['gid']))}")

    # CSV em vez de GPKG: escrever GPKG sobre pasta montada (Windows via mount)
    # falha no SQLite ("no such table: gpkg_contents"). CSV com x/y em EPSG:27700
    # e suficiente e reabre em qualquer ferramenta.
    saida = grade.copy()
    saida["x_bng"] = saida.geometry.x
    saida["y_bng"] = saida.geometry.y
    saida.drop(columns="geometry").to_csv(
        OUTDIR / "grade_classificada.csv", index=False
    )
    resumo = {
        "aoi_bng": AOI,
        "passo_m": PASSO,
        "area_aoi_km2": round(area_aoi_km2),
        "rfo_na_aoi": int(len(rfo)),
        "fz_total": int(fz_total),
        "grade_pontos": int(n),
        "classes": {k: int(v) for k, v in cont.items()},
        "criterio_negativo": (
            "fora de qualquer Recorded Flood Outline elegivel E fora de "
            "qualquer Flood Zone modelada (FZ2 ou FZ3)"
        ),
        "aviso": (
            "Nenhuma classe aqui e rotulo. POS_CAND e NEG_CAND sao candidatos; "
            "a promocao exige tarefa propria de adjudicacao com criterio escrito."
        ),
    }
    (OUTDIR / "contabilidade.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
