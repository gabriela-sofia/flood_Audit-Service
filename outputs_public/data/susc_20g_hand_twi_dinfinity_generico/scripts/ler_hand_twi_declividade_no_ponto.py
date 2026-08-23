"""Leitura observacional de HAND/TWI/declividade no pixel de um ponto lat/lon.

Leitura de coerência física, **não** extração de variavel: não escreve dataset, não normaliza,
não compara com ponto negativo (não existe negativo adjudicado nestas regiões), não alimenta
modelo. Serve para olhar o valor cru no pixel de um candidato já adjudicado e dizer se ele é
fisicamente plausível.

Uso:
    python ler_hand_twi_declividade_no_ponto.py --raster-dir <dir_com_hand_twi_slope>
                                           --lat -22.51625 --lon -43.18828
                                           [--rotulo "Petropolis/Valparaiso"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

RASTERS = {"hand_m": "hand_dinf.tif", "twi": "twi_dinf.tif", "slope_deg": "slope_deg_wbt.tif"}


def read_at_point(raster_dir: Path | str, lat: float, lon: float, rotulo: str = "") -> dict:
    raster_dir = Path(raster_dir)
    out: dict = {"rotulo": rotulo, "lat": lat, "lon": lon, "raster_dir": str(raster_dir)}

    for key, fname in RASTERS.items():
        path = raster_dir / fname
        if not path.exists():
            out[key] = None
            out[f"{key}_status"] = f"RASTER_AUSENTE: {fname}"
            continue
        with rasterio.open(path) as src:
            x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
            b = src.bounds
            if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
                out[key] = None
                out[f"{key}_status"] = "FORA_DA_EXTENSAO_DO_RASTER"
                continue
            row, col = src.index(x, y)
            value = float(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
            nodata = src.nodata
            invalid = (not np.isfinite(value)) or value <= -32767 or (
                nodata is not None and value == nodata
            )
            out[key] = None if invalid else value
            out[f"{key}_status"] = "NODATA_NO_PIXEL" if invalid else "OK"
            out.setdefault("crs", str(src.crs))
            out.setdefault("tamanho_pixel_m", round(abs(src.transform.a), 4))
            out.setdefault("pixel_row_col", [int(row), int(col)])
            out.setdefault("easting_northing", [round(x, 2), round(y, 2)])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raster-dir", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--rotulo", default="")
    p.add_argument("--json-out")
    args = p.parse_args(argv)

    res = read_at_point(args.raster_dir, args.lat, args.lon, args.rotulo)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
