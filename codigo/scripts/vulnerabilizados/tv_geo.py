"""Etapas que exigem geopandas (ambiente revp-susc): atribuir setor/classe territorial a pontos.

O ajuste dos modelos roda no Python 3.12 do projeto: no revp-susc o L-BFGS do SciPy cai com
0xc06d007f (DLL ausente), inclusive sem GDAL carregado. As duas metades trocam parquet.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum_tv import OUT, RUNS, SETORES, gravar
from tv01_universo_escassez import universo_nacional


def ler_malha(uf: str, colunas=("CD_SETOR",)):
    import geopandas as gpd
    shp = [n for n in zipfile.ZipFile(SETORES[uf]).namelist() if n.endswith(".shp")][0]
    return gpd.read_file(f"/vsizip/{SETORES[uf].as_posix()}/{shp}", columns=list(colunas))


def atribuir_setores(df: pd.DataFrame, uf: str, nac: pd.DataFrame | None = None) -> pd.DataFrame:
    import geopandas as gpd
    st = ler_malha(uf).to_crs(4326)
    g = gpd.GeoDataFrame(df.reset_index(drop=True), geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
    g["_i"] = np.arange(len(g))
    j = gpd.sjoin(g, st, predicate="within", how="left").drop_duplicates("_i").sort_values("_i")
    j = pd.DataFrame(j.drop(columns=["geometry", "index_right", "_i"]))
    nac = universo_nacional() if nac is None else nac
    j = j.merge(nac[["CD_SETOR", "classe_territorial", "CD_FCU", "NM_FCU", "CD_MUN", "NM_MUN", "renda_resp"]],
                on="CD_SETOR", how="left")
    j["classe_territorial"] = j.classe_territorial.fillna("fora_de_setor")
    return j


def taquari():
    p = pd.read_csv(RUNS / "cems-emsr720/pontos_emsr720.csv")
    p = p[p.classe_obs.isin(["INUNDACAO", "NAO_INUNDADO_OBSERVADO"])].copy()
    p["y"] = p.classe_obs.eq("INUNDACAO").astype(int)
    p = p[np.isfinite(p[["hand_m", "slope_deg", "twi_dinf"]]).all(axis=1) & (p.hand_m > -1000)]
    gravar(OUT / "desenvolvimento_taquari/pontos.parquet", atribuir_setores(p, "RS"))


if __name__ == "__main__":
    {"taquari": taquari}[sys.argv[1]]()
