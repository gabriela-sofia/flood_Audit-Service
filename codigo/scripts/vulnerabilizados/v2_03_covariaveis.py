"""Covariaveis das camadas para as tres grades (Python 3.12): cobertura ESA WorldCover 2021 e distancia ao canal.

- worldcover: classe do pixel de 10 m que contem o ponto (COG publico v200/2021); construido = classe 50.
  Os doadores do ds-06 ja trazem `worldcover_2021` amostrado no ponto, com o mesmo produto.
- dist_canal_m: distancia euclidiana (m) ao pixel de canal de `streams_dinf.tif` do mesmo tile do terreno
  (limiar de 0,1123 km2 da cadeia). Os doadores nao tem essa coluna: so entra em modelos ajustados com
  rotulo brasileiro.
Nenhum rotulo e lido. Datas: WorldCover 2021; episodios 2021, 2023 e 2024 (defasagem declarada).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum_tv import OUT as V1, RUNS, agora, sha
from v2_00_partida import V2, g2

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
WC = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{t}_Map.tif"
GRADES = {"curitiba": V2 / "curitiba/grade.parquet", "rmpa": V1 / "confirmacao_rmpa/grade.parquet",
          "manaus": V1 / "confirmacao_manaus/grade.parquet"}


def tile_wc(lon, lat):
    la = int(np.floor(lat / 3) * 3)
    lo = int(np.floor(lon / 3) * 3)
    return f"{'S' if la < 0 else 'N'}{abs(la):02d}{'W' if lo < 0 else 'E'}{abs(lo):03d}"


def worldcover(g: pd.DataFrame) -> np.ndarray:
    import rasterio
    from rasterio.windows import from_bounds
    out = np.zeros(len(g), dtype=np.int16)
    nomes = np.array([tile_wc(a, b) for a, b in zip(g.lon, g.lat)])
    for t in np.unique(nomes):
        k = np.where(nomes == t)[0]
        q = g.iloc[k]
        with rasterio.open(WC.format(t=t)) as s:
            w = from_bounds(q.lon.min() - .001, q.lat.min() - .001, q.lon.max() + .001, q.lat.max() + .001, transform=s.transform)
            w = w.round_offsets().round_lengths()
            a = s.read(1, window=w)
            tr = s.window_transform(w)
        col = np.floor((q.lon.to_numpy() - tr.c) / tr.a).astype(int)
        lin = np.floor((q.lat.to_numpy() - tr.f) / tr.e).astype(int)
        ok = (lin >= 0) & (lin < a.shape[0]) & (col >= 0) & (col < a.shape[1])
        v = np.zeros(len(k), dtype=np.int16)
        v[ok] = a[lin[ok], col[ok]]
        out[k] = v
        print("worldcover", t, len(k), flush=True)
    return out


def dist_canal(g: pd.DataFrame) -> np.ndarray:
    import rasterio
    from rasterio.windows import Window, from_bounds
    from scipy.ndimage import distance_transform_edt
    out = np.full(len(g), np.nan)
    for tile, q in g.groupby("tile"):
        with rasterio.open(RUNS / "ter-01-cadeia-harmonizada" / tile / "streams_dinf.tif") as s:
            w = from_bounds(q.x.min() - 3000, q.y.min() - 3000, q.x.max() + 3000, q.y.max() + 3000, transform=s.transform)
            w = w.round_offsets().round_lengths().intersection(Window(0, 0, s.width, s.height))
            a = s.read(1, window=w)
            tr = s.window_transform(w)
            canal = (a > 0) & (a != s.nodata) if s.nodata is not None else a > 0
        d = distance_transform_edt(~canal) * abs(tr.a)
        col = np.floor((q.x.to_numpy() - tr.c) / tr.a).astype(int)
        lin = np.floor((q.y.to_numpy() - tr.f) / tr.e).astype(int)
        ok = (lin >= 0) & (lin < d.shape[0]) & (col >= 0) & (col < d.shape[1])
        v = np.full(len(q), np.nan)
        v[ok] = d[lin[ok], col[ok]]
        out[g.index.get_indexer(q.index)] = v
        print("canal", tile, len(q), flush=True)
    return out


def main():
    res = {}
    for nome, p in GRADES.items():
        g = pd.read_parquet(p, columns=["ponto_id", "tile", "x", "y", "lon", "lat"])
        c = g[["ponto_id"]].copy()
        c["worldcover"] = worldcover(g)
        c["construido"] = (c.worldcover == 50).astype(int)
        c["dist_canal_m"] = dist_canal(g)
        g2(V2 / f"covariaveis/{nome}.parquet", c)
        res[nome] = {"pontos": len(c), "construido": float(c.construido.mean()), "worldcover_ausente": float((c.worldcover == 0).mean()),
                     "dist_canal_mediana_m": float(np.nanmedian(c.dist_canal_m)), "grade_sha": sha(p),
                     "sha": sha(V2 / f"covariaveis/{nome}.parquet")}
    g2(V2 / "covariaveis/resultado.json", {"quando_utc": agora(), "fontes": {"worldcover": WC.replace("/vsicurl/", "")}, "por_grade": res})
    print(res)


if __name__ == "__main__":
    main()
