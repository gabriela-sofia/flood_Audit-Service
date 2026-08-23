"""SUSC-20F -- amostragem de terreno sob demanda (elevacao, declividade, HAND-Dinf,
TWI-Dinf) para QUALQUER ponto dentro da cobertura real do DTM PE3D merged usado para
treinar o v12 (SUSC-20B, `improvement2_hand_twi_dinf_v7_report.md`).

Isto NAO recalcula HAND/TWI do zero (isso exigiria WhiteboxTools + o DTM PE3D bruto,
que requer login/captcha manual em pe3d.pe.gov.br, ja documentado como bloqueado em
`susc_devro17_bloqueio_mdt_pe3d`). Em vez disso, AMOSTRA os 4 rasters ja merged e
processados (dem_filled.tif, slope_deg_wbt.tif, hand_dinf.tif, twi_dinf.tif --
EPSG:31985, 10m, bounds ~21km x 28km cobrindo essencialmente toda a regiao de Recife
ja definida no produto) exatamente como a extracao original fez pros 278 pontos --
mesma metodologia, mesmo dado, sem refit e sem download novo algum.

Cobertura real: qualquer lat/lon dentro do bbox WGS84 aproximado
(-35.033, -8.168, -34.843, -7.916) tem HAND/TWI/declividade/elevacao REAIS
disponiveis aqui. Fora disso -> None (fail-closed, nao interpola nem inventa).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import rasterio
from pyproj import Transformer

RASTER_FILES = {
    "elevation_m": "dem_filled.tif",
    "slope_deg": "slope_deg_wbt.tif",
    "hand_m_dinf": "hand_dinf.tif",
    "twi_dinf": "twi_dinf.tif",
}


def _raster_dir() -> Path | None:
    raw = os.environ.get("REVP_SUSC20F_TERRAIN_RASTER_DIR", "")
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_dir() else None


_transformer_to_31985 = Transformer.from_crs("EPSG:4326", "EPSG:31985", always_xy=True)


def sample_terrain_features(lat: float, lon: float) -> Optional[dict]:
    """Retorna {elevation_m, slope_deg, hand_m_dinf, twi_dinf} reais amostrados dos
    rasters, ou None se o ponto cai fora da cobertura raster ou em celula nodata, ou
    se REVP_SUSC20F_TERRAIN_RASTER_DIR nao estiver configurado (fail-closed)."""
    raster_dir = _raster_dir()
    if raster_dir is None:
        return None

    x, y = _transformer_to_31985.transform(lon, lat)

    out: dict[str, float] = {}
    for feat_name, fname in RASTER_FILES.items():
        path = raster_dir / fname
        if not path.exists():
            return None
        with rasterio.open(path) as ds:
            row, col = ds.index(x, y)
            if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
                return None
            val = next(ds.sample([(x, y)]))[0]
            if ds.nodata is not None and val == ds.nodata:
                return None
            out[feat_name] = float(val)
    return out


def coverage_bbox_wgs84() -> tuple[float, float, float, float]:
    """Bbox real (lon_min, lat_min, lon_max, lat_max) da cobertura DTM, reprojetado
    de EPSG:31985 (calculado uma vez a partir de dem_filled.tif nesta sessao)."""
    return (-35.032950343243996, -8.167561668035802, -34.84255396733535, -7.9156195118262955)
