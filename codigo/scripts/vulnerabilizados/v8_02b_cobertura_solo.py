"""v8: construido (GHSL) e areia (SoilGrids) do contrato Q nas coortes brasileiras.

Definicao identica a cob01.imagem: construido_frac = GHS_BUILT_S (epoca <= ano do evento; 2020 para as cinco coortes)
/ 10.000 m2; areia_pct_0_30 = media das camadas 0-5, 5-15, 15-30 cm de sand_mean / 10. Rota: UM raster por coorte
(caixa dos pontos + 0,01 grau) baixado do Earth Engine na MESMA grade da API de pontos (EPSG:4326, passo de 30 m
em graus, origem 0,0 -- crs_transform explicito), em blocos, com mascara explicita (unmask(-1) -> NaN) e amostrado
localmente pelo pixel que contem o ponto. A primeira tentativa (100/250 m nativos, pasta `cobertura_solo/`) fica
preservada: o NoData 0 do GeoTIFF apagava construido=0 e a grade diferente dava 62-68% de concordancia. Conferencia: 300 pontos por coorte (sorteio com semente, sem rotulo) pela
API de pontos do proprio cob01 (reduceRegions, escala 30). Divergencia fica registrada, nao corrigida.
"""
from __future__ import annotations

import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_02_ponte_fisica import FONTES
from v8_comum import V8, agora, etapa, g8, sha_hex

D = V8 / "ponte" / "cobertura_solo_30m"
PASSO = 30 / 111319.49079327357  # grau equivalente a 30 m no equador (escala 30 da API)
ANO = {"rmpa": 2024, "manaus": 2021, "curitiba": 2023, "recife": 2022, "rmr": 2022}


def baixar(ee, img, caixa, destino: Path, bloco=1500):
    """Mosaico em blocos de `bloco` pixels na grade global (origem 0,0, passo PASSO)."""
    import rasterio
    from rasterio.transform import Affine
    c0, c1 = int(np.floor(caixa[0] / PASSO)), int(np.ceil(caixa[2] / PASSO))
    r0, r1 = int(np.floor(-caixa[3] / PASSO)), int(np.ceil(-caixa[1] / PASSO))
    mos = np.full((r1 - r0, c1 - c0), -1.0, dtype="float32")
    for rr in range(r0, r1, bloco):
        for cc in range(c0, c1, bloco):
            h, w = min(bloco, r1 - rr), min(bloco, c1 - cc)
            url = img.unmask(-1).toFloat().getDownloadURL({"crs": "EPSG:4326", "crs_transform": [PASSO, 0, cc * PASSO, 0, -PASSO, -rr * PASSO],
                                                           "dimensions": f"{w}x{h}", "format": "GEO_TIFF"})
            for tent in range(4):
                try:
                    with urllib.request.urlopen(url, timeout=300) as r:
                        dados = r.read()
                    break
                except Exception:  # noqa: BLE001 - modo restrito do EE: tenta de novo
                    if tent == 3:
                        raise
                    time.sleep(10)
            if dados[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(dados)) as z:
                    dados = z.read([n for n in z.namelist() if n.endswith(".tif")][0])
            with rasterio.MemoryFile(dados) as mf, mf.open() as s:
                mos[rr - r0:rr - r0 + h, cc - c0:cc - c0 + w] = s.read(1)[:h, :w]
    with rasterio.open(destino, "w", driver="GTiff", height=mos.shape[0], width=mos.shape[1], count=1, dtype="float32",
                       crs="EPSG:4326", transform=Affine(PASSO, 0, c0 * PASSO, 0, -PASSO, -r0 * PASSO), nodata=-1, compress="deflate") as s:
        s.write(mos, 1)


def amostrar(arq: Path, lon, lat):
    import rasterio
    from rasterio.transform import rowcol
    with rasterio.open(arq) as s:
        a = s.read(1).astype("float64")
        a[a < 0] = np.nan
        lin, col = (np.asarray(v) for v in rowcol(s.transform, lon, lat))
    ok = (lin >= 0) & (lin < a.shape[0]) & (col >= 0) & (col < a.shape[1])
    out = np.full(len(lon), np.nan)
    out[ok] = a[lin[ok], col[ok]]
    return out


def main():
    import ee
    import cob01_cobertura_e_solo as cob
    from mt11_confirmatorio_transdutivo import _amostrar_tabela
    ee.Initialize()
    D.mkdir(parents=True, exist_ok=True)
    for coorte, fonte in FONTES.items():
        alvo = D / f"{coorte}.parquet"
        if alvo.exists():
            continue
        t0 = time.time()
        g = pd.read_parquet(fonte, columns=["ponto_id", "lon", "lat"])
        ep = cob.anos_das_camadas(ANO[coorte])["ghsl"]
        caixa = [g.lon.min() - .01, g.lat.min() - .01, g.lon.max() + .01, g.lat.max() + .01]
        ghsl = ee.ImageCollection(cob.GHSL).filter(ee.Filter.eq("system:index", str(ep))).first().select("built_surface").divide(cob.AREA_CELULA_GHSL_M2)
        asset = cob.SOILGRIDS["areia"]
        areia = ee.Image(asset).select([f"sand_{c}_mean" for c in cob.CAMADAS_SOLO]).reduce(ee.Reducer.mean()).divide(10)
        arq_c, arq_a = D / f"{coorte}_ghsl{ep}.tif", D / f"{coorte}_areia.tif"
        if not arq_c.exists():
            baixar(ee, ghsl, caixa, arq_c)
        if not arq_a.exists():
            baixar(ee, areia, caixa, arq_a)
        out = g[["ponto_id"]].assign(construido_v2=amostrar(arq_c, g.lon.to_numpy(), g.lat.to_numpy()),
                                     areia_v2=amostrar(arq_a, g.lon.to_numpy(), g.lat.to_numpy()), ghsl_epoca=ep)
        qa = g.sample(300, random_state=20261022)
        t = _amostrar_tabela(ee, cob.imagem(ee, ANO[coorte]), qa[["ponto_id", "lat", "lon"]], 30).set_index("ponto_id").reindex(qa.ponto_id)
        m = out.set_index("ponto_id").loc[qa.ponto_id]
        conf = {}
        for a, b, tol in [("construido_v2", "construido_frac", .02), ("areia_v2", "areia_pct_0_30", .5)]:
            x, y = m[a].to_numpy(float), t[b].to_numpy(float)
            f = np.isfinite(x) & np.isfinite(y)
            conf[a] = {"n": int(f.sum()), "concordancia_tol": float((np.abs(x - y)[f] <= tol).mean()) if f.any() else None,
                       "tolerancia": tol, "spearman": float(pd.Series(x[f]).corr(pd.Series(y[f]), method="spearman")) if f.sum() > 2 else None,
                       "nan_raster": int((~np.isfinite(x)).sum()), "nan_api": int((~np.isfinite(y)).sum())}
        g8(alvo, out)
        g8(alvo.with_suffix(".json"), {"quando_utc": agora(), "linhas": len(out), "ghsl_epoca": ep, "segundos": time.time() - t0,
                                       "rasters_sha256": {arq_c.name: sha_hex(arq_c), arq_a.name: sha_hex(arq_a)},
                                       "finitos": {c: int(out[c].notna().sum()) for c in ["construido_v2", "areia_v2"]},
                                       "conferencia_api_pontos": conf})
        etapa("v8_02b_cobertura_solo", coorte=coorte, conferencia=conf)
        print(coorte, conf, flush=True)


if __name__ == "__main__":
    main()
