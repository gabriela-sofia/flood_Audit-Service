"""Referencia de avaliacao na RMPA: agua optica Sentinel-2 de 06/05/2024 (primaria) e extensao EMSN194 (triangulacao).

Ambiente revp-susc. Este codigo NAO le as predicoes: confere so que o registro delas existe e copia
o hash, para provar a ordem. Le a grade (pontos, tile, linha, coluna) e as imagens.
Regras: as do protocolo congelado `confirmacao_rmpa/protocolo.json` (referencia_primaria).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum_tv import OUT, RUNS, agora, gravar, ler, rotulo_referencia, sha

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
BASE = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/22/J/{t}/2024/{m}/{item}/{banda}.tif"
TILES = ["22JDM", "22JDN"]
DATAS = {"evento": ("S2B", "20240506", 5), "pre": ("S2A", "20240418", 4)}


def url(tile: str, quando: str, banda: str) -> str:
    sat, dia, mes = DATAS[quando]
    return "/vsicurl/" + BASE.format(t=tile[3:], m=mes, item=f"{sat}_{tile}_{dia}_0_L2A", banda=banda)


def ler_janela(u: str, bounds, res: float):
    """Le a janela em 20 m (media para bandas de 10 m), devolve array e transform do grid de 20 m do tile."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    with rasterio.open(u) as s:
        w = from_bounds(*bounds, transform=s.transform).round_offsets().round_lengths()
        w = w.intersection(rasterio.windows.Window(0, 0, s.width, s.height))
        f = int(round(res / s.res[0]))
        shape = (int(w.height) // f, int(w.width) // f)
        a = s.read(1, window=w, out_shape=shape, resampling=Resampling.average if f > 1 else Resampling.nearest)
        tr = s.window_transform(w) * rasterio.Affine.scale(f, f)
        return a, tr, s.crs.to_string()


def amostrar(a, tr, x, y, fill):
    col = np.floor((x - tr.c) / tr.a).astype(int)
    lin = np.floor((y - tr.f) / tr.e).astype(int)
    ok = (lin >= 0) & (lin < a.shape[0]) & (col >= 0) & (col < a.shape[1])
    out = np.full(len(x), fill, dtype=a.dtype if np.issubdtype(a.dtype, np.floating) else np.float64)
    out[ok] = a[lin[ok], col[ok]]
    return out


def indices(b_verde, b_swir, b_nir):
    v, s, n = (b.astype("float64") for b in (b_verde, b_swir, b_nir))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (v - s) / (v + s), (v - n) / (v + n)


def main():
    import rasterio
    t0 = agora()
    prot = ler(OUT / "confirmacao_rmpa/protocolo.json")
    reg = OUT / "confirmacao_rmpa/predicoes_registro.json"
    if not reg.exists():
        raise SystemExit("predicoes ainda nao registradas: a referencia so e construida depois delas")
    sha_pred = ler(reg)["sha_predicoes"]
    g = pd.read_parquet(OUT / "confirmacao_rmpa/grade.parquet", columns=["ponto_id", "tile", "linha", "coluna", "x", "y", "lon", "lat"])
    pad = 200
    bounds = (g.x.min() - pad, g.y.min() - pad, g.x.max() + pad, g.y.max() + pad)
    amostras = {}
    fontes = []
    for quando in DATAS:
        por_tile = []
        for tile in TILES:
            cols = {}
            for banda in ["SCL", "B03", "B11", "B08"]:
                u = url(tile, quando, banda)
                a, tr, crs = ler_janela(u, bounds, 20.0)
                if crs != "EPSG:32722":
                    raise ValueError(f"CRS inesperado {crs}")
                cols[banda] = amostrar(a, tr, g.x.to_numpy(), g.y.to_numpy(), 0)
                fontes.append({"url": u.replace("/vsicurl/", ""), "janela_20m": list(a.shape)})
                if quando == "evento" and banda == "SCL":
                    salvar = OUT / f"confirmacao_rmpa/referencia_rasters/scl_{tile}_{quando}.tif"
                    salvar.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(salvar, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1,
                                       dtype=a.dtype, crs=crs, transform=tr, compress="deflate") as o:
                        o.write(a, 1)
                print(quando, tile, banda, flush=True)
            por_tile.append(cols)
        # mosaico fixo: primeiro tile da lista com SCL valido (mesma passagem; sobreposicao identica)
        valido_scl = prot["referencia_primaria"]["valido_scl"]
        escolha = {k: np.zeros(len(g)) for k in ["SCL", "B03", "B11", "B08"]}
        feito = np.zeros(len(g), bool)
        for cols in por_tile:
            usar = ~feito & (cols["SCL"] != 0)
            for k in escolha:
                escolha[k][usar] = cols[k][usar]
            feito |= usar
        mndwi, ndwi = indices(escolha["B03"], escolha["B11"], escolha["B08"])
        amostras[quando] = {"valido": np.isin(escolha["SCL"], valido_scl), "scl": escolha["SCL"].astype(int),
                            "mndwi": mndwi, "ndwi": ndwi, "com_dado": escolha["SCL"] != 0}
    # agua permanente JRC: so onde a camada existe no tile de terreno
    jrc = np.full(len(g), np.nan)
    for tile, q in g.groupby("tile"):
        p = RUNS / "ter-01-cadeia-harmonizada" / tile / "agua_permanente_jrc.tif"
        if p.exists():
            with rasterio.open(p) as s:
                if s.crs.to_string() != "EPSG:32722":
                    raise ValueError("JRC fora do CRS da grade")
                v = amostrar(s.read(1).astype(float), s.transform, q.x.to_numpy(), q.y.to_numpy(), np.nan)
                v[v == s.nodata] = np.nan
                jrc[q.index] = v
    perm_jrc = np.nan_to_num(jrc, nan=0) == 1
    ev, pre = amostras["evento"], amostras["pre"]
    ref = g[["ponto_id"]].copy()
    ref["oportunidade_s2"] = ev["com_dado"]
    ref["valido_evento"] = ev["valido"]
    ref["scl_evento"] = ev["scl"]
    ref["mndwi_evento"] = ev["mndwi"]
    ref["ndwi_evento"] = ev["ndwi"]
    ref["valido_pre"] = pre["valido"]
    ref["mndwi_pre"] = pre["mndwi"]
    ref["jrc_disponivel"] = ~np.isnan(jrc)
    ref["jrc_permanente"] = perm_jrc
    for nome, ag_ev, ag_pre in [("primaria", ev["mndwi"] > 0, pre["mndwi"] > 0),
                                ("mndwi_0_2", ev["mndwi"] > 0.2, pre["mndwi"] > 0.2),
                                ("ndwi_b08", ev["ndwi"] > 0, pre["ndwi"] > 0)]:
        ref[f"estado_{nome}"] = rotulo_referencia(ag_ev, ag_pre, perm_jrc, ev["valido"], pre["valido"])
    # triangulacao EMSN194 (dependente do FABDEM)
    import geopandas as gpd
    pts = gpd.GeoDataFrame(g[["ponto_id"]], geometry=gpd.points_from_xy(g.lon, g.lat), crs=4326)
    aoi = gpd.read_file(RUNS / "emsn-emsn194/00_area_of_interest.geojson").to_crs(4326)
    ext = gpd.read_file(RUNS / "emsn-emsn194/11_extent_temporal_evolution_06_05_2024.geojson").to_crs(4326)
    dentro_aoi = gpd.sjoin(pts, aoi[["geometry"]], predicate="within", how="inner").ponto_id.unique()
    dentro_ext = gpd.sjoin(pts, ext[["geometry"]], predicate="within", how="inner").ponto_id.unique()
    ref["emsn_aoi"] = ref.ponto_id.isin(dentro_aoi)
    ref["emsn_extensao"] = ref.ponto_id.isin(dentro_ext)
    e = np.full(len(ref), "fora_aoi_emsn", dtype=object)
    prim = ref.estado_primaria.to_numpy()
    e[ref.emsn_aoi & ref.emsn_extensao] = "positivo"
    e[ref.emsn_aoi & ~ref.emsn_extensao & (prim == "negativo")] = "negativo"
    e[ref.emsn_aoi & ~ref.emsn_extensao & (prim != "negativo")] = "desconhecido"
    e[ref.emsn_aoi & ref.emsn_extensao & (prim == "permanente")] = "permanente"
    ref["estado_emsn194"] = e
    gravar(OUT / "confirmacao_rmpa/referencia.parquet", ref)
    resumo = {c: ref[c].value_counts().to_dict() for c in ref.columns if c.startswith("estado_")}
    gravar(OUT / "confirmacao_rmpa/referencia.json", {
        "inicio_utc": t0, "fim_utc": agora(), "sha_predicoes_ja_registradas": sha_pred,
        "leu_predicoes": False, "estados": resumo, "oportunidade_s2": float(ref.oportunidade_s2.mean()),
        "valido_evento": float(ref.valido_evento.mean()), "jrc_disponivel": float(ref.jrc_disponivel.mean()),
        "fontes_imagem": fontes, "emsn194": {"aoi": sha(RUNS / "emsn-emsn194/00_area_of_interest.geojson"),
                                             "extensao_06_05": sha(RUNS / "emsn-emsn194/11_extent_temporal_evolution_06_05_2024.geojson"),
                                             "fontes_declaradas_da_camada": "1,2,3,4,981(FABDEM),982,983"},
        "sha_referencia": sha(OUT / "confirmacao_rmpa/referencia.parquet"),
        "protocolo_sha256": sha(OUT / "confirmacao_rmpa/protocolo.json"),
    })
    print(json.dumps(resumo, ensure_ascii=False))


if __name__ == "__main__":
    main()
