"""Baixa apenas a janela de uma bbox a partir de um DEM aberto publicado como COG.

Motivo de existir: os DEMs globais são publicados em tiles de 1°×1° (~50 MB cada). Ler o COG
remoto por `/vsicurl` e recortar a bbox na leitura traz só os blocos necessários — mínimo de
dado necessário, sem baixar tile inteiro.

Sem credencial, sem token: só funciona com bucket/HTTP público. Se a fonte exigir login, o
script falha alto em vez de tentar contornar.

Uso:
    python baixar_dem_aberto_bbox.py --url <cog_url> --bbox lon_min,lat_min,lon_max,lat_max
                                  --out <saida.tif> [--dst-crs EPSG:31984] [--dst-res 30.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds

DEFAULT_NODATA = -9999.0


def fetch_dem_bbox(
    url: str,
    bbox: tuple[float, float, float, float],
    out_path: Path | str,
    dst_crs: str | None = None,
    dst_res: float | None = None,
    nodata: float = DEFAULT_NODATA,
) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bbox

    with rasterio.open(url if url.startswith("/vsi") else f"/vsicurl/{url}") as src:
        info = {
            "source_url": url,
            "source_crs": str(src.crs),
            "source_shape": [src.height, src.width],
            "source_res": list(src.res),
            "source_bounds": list(src.bounds),
            "source_nodata": None if src.nodata is None else float(src.nodata),
            "bbox_requested": list(bbox),
        }
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        data = src.read(1, window=window).astype("float32")
        transform = src.window_transform(window)
        src_crs = src.crs
        src_nodata = src.nodata

    info["window_shape"] = list(data.shape)
    info["bytes_read_estimate"] = int(data.nbytes)

    if dst_crs is None:
        out_arr, out_transform, out_crs = data, transform, src_crs
    else:
        out_crs = CRS.from_string(dst_crs)
        west, south, east, north = array_bounds(*data.shape, transform)
        dst_transform, width, height = calculate_default_transform(
            src_crs, out_crs, data.shape[1], data.shape[0],
            left=west, bottom=south, right=east, top=north,
            resolution=dst_res if dst_res else None,
        )
        out_arr = np.full((int(height), int(width)), nodata, dtype="float32")
        reproject(
            source=data, destination=out_arr,
            src_transform=transform, src_crs=src_crs, src_nodata=src_nodata,
            dst_transform=dst_transform, dst_crs=out_crs, dst_nodata=nodata,
            resampling=Resampling.bilinear,
        )
        out_transform = dst_transform

    with rasterio.open(
        out_path, "w", driver="GTiff", height=out_arr.shape[0], width=out_arr.shape[1],
        count=1, dtype="float32", crs=out_crs, transform=out_transform, nodata=nodata,
        compress="deflate", tiled=True,
    ) as dst:
        dst.write(out_arr, 1)

    valid = out_arr > -1000.0
    info.update({
        "output": str(out_path),
        "output_crs": str(out_crs),
        "output_shape": list(out_arr.shape),
        "output_res_m": [abs(out_transform.a), abs(out_transform.e)],
        "n_celulas_validas": int(valid.sum()),
        "elev_min": float(out_arr[valid].min()) if valid.any() else None,
        "elev_max": float(out_arr[valid].max()) if valid.any() else None,
        "elev_mean": float(out_arr[valid].mean()) if valid.any() else None,
    })
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--bbox", required=True, help="lon_min,lat_min,lon_max,lat_max")
    p.add_argument("--out", required=True)
    p.add_argument("--dst-crs", default=None)
    p.add_argument("--dst-res", type=float, default=None)
    p.add_argument("--json-out")
    args = p.parse_args(argv)

    bbox = tuple(float(v) for v in args.bbox.split(","))
    if len(bbox) != 4:
        raise SystemExit("--bbox precisa de 4 valores: lon_min,lat_min,lon_max,lat_max")
    info = fetch_dem_bbox(args.url, bbox, args.out, args.dst_crs, args.dst_res)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
