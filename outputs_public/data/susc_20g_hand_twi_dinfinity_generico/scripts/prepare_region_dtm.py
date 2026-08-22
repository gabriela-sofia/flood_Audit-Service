"""Prepara o MDT/MDE de uma região como GeoTIFF pronto para o pipeline D-infinity.

Entrada: qualquer raster legível pelo GDAL — inclusive grade Esri ArcInfo Binary Grid (`AIG`),
que é o formato entregue pelo SGB/CPRM para Curitiba e Petrópolis.

O que faz (e só isso):
  - lê a grade nativa;
  - opcionalmente reamostra para uma resolução alvo (`average` ao reduzir, `bilinear` ao
    ampliar — ampliar é registrado como aviso no manifesto, porque não cria informação);
  - opcionalmente reetiqueta o CRS quando a fonte traz apenas um Transverse Mercator genérico
    sem código EPSG (o WKT original é preservado no manifesto);
  - escreve GeoTIFF float32 com nodata explícito.

Não recorta, não preenche buraco, não suaviza: qualquer diferença de conteúdo em relação à
fonte vem só da reamostragem pedida.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

DEFAULT_NODATA = -9999.0


def prepare_dtm(
    grid_path: Path | str,
    out_path: Path | str,
    target_res: float | None = None,
    assign_crs: str | None = None,
    nodata: float = DEFAULT_NODATA,
) -> dict:
    grid_path, out_path = Path(grid_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(grid_path) as src:
        src_wkt = src.crs.to_wkt() if src.crs else None
        native_res = (abs(src.transform.a), abs(src.transform.e))
        info = {
            "source": str(grid_path),
            "source_driver": src.driver,
            "source_crs_wkt": src_wkt,
            "source_shape": [src.height, src.width],
            "source_res_m": list(native_res),
            "source_nodata": None if src.nodata is None else float(src.nodata),
            "source_bounds": list(src.bounds),
        }
        src_nodata = src.nodata if src.nodata is not None else np.finfo("float32").min
        out_crs = assign_crs or src.crs

        if target_res is None or abs(target_res - native_res[0]) < 1e-9:
            data = src.read(1).astype("float32")
            transform = src.transform
            height, width = src.height, src.width
            info["resampling"] = "none"
        else:
            scale = native_res[0] / target_res
            width = int(round(src.width * scale))
            height = int(round(src.height * scale))
            transform = from_origin(src.bounds.left, src.bounds.top, target_res, target_res)
            method = Resampling.average if target_res > native_res[0] else Resampling.bilinear
            data = np.full((height, width), nodata, dtype="float32")
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=transform,
                dst_crs=src.crs,
                dst_nodata=nodata,
                resampling=method,
            )
            info["resampling"] = method.name
            if target_res < native_res[0]:
                info["resampling_warning"] = (
                    "resolução alvo mais fina que a nativa: interpolação não cria informação"
                )

    data = np.where(np.isfinite(data) & (data > -1e30), data, nodata).astype("float32")
    with rasterio.open(
        out_path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs=out_crs, transform=transform, nodata=nodata,
        compress="deflate", tiled=True,
    ) as dst:
        dst.write(data, 1)

    valid = data > -1000.0
    info.update(
        {
            "output": str(out_path),
            "output_crs": str(out_crs),
            "output_res_m": [target_res or native_res[0], target_res or native_res[1]],
            "output_shape": [height, width],
            "output_nodata": nodata,
            "n_total_cells": int(data.size),
            "n_valid_cells": int(valid.sum()),
            "elev_min_m": float(data[valid].min()) if valid.any() else None,
            "elev_max_m": float(data[valid].max()) if valid.any() else None,
            "elev_mean_m": float(data[valid].mean()) if valid.any() else None,
            "crs_reassigned": bool(assign_crs),
        }
    )
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", required=True, help="raster/grade de entrada (dir AIG ou GeoTIFF)")
    p.add_argument("--out", required=True, help="GeoTIFF de saída")
    p.add_argument("--target-res", type=float, default=None, help="resolução alvo em metros")
    p.add_argument("--assign-crs", default=None, help="ex.: EPSG:31982")
    p.add_argument("--nodata", type=float, default=DEFAULT_NODATA)
    p.add_argument("--json-out")
    args = p.parse_args(argv)

    info = prepare_dtm(args.grid, args.out, args.target_res, args.assign_crs, args.nodata)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
