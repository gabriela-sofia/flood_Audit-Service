"""HAND e TWI por D-infinity (Tarboton 1997) via WhiteboxTools — unidade genérica.

Reconstrói, como script parametrizável e testável, a sequência de chamadas que gerou
`hand_dinf.tif`/`twi_dinf.tif` do modelo v12 de Recife e que até aqui só existia descrita em
prosa em `improvement2_hand_twi_dinf_report.md` (PROJETO/local_runs/recife_modelo_v7_otimizado).

Sequência (idêntica à documentada no relatório do v7/v12):
  1. fill_depressions_wang_and_liu (Wang & Liu 2006, fix_flats=True)
  2. d_inf_flow_accumulation  out_type="cells"  -> acumulação para extração de drenagem
  3. d_inf_flow_accumulation  out_type="sca"    -> área de contribuição específica para o TWI
  4. slope (graus) sobre o MDT preenchido
  5. limiar de drenagem = percentil P da distribuição de acumulação em "cells"
     sobre as células válidas do MDT de entrada (P=98 no v12 de Recife)
  6. extract_streams(acumulação_cells, limiar) -> rede de drenagem
  7. elevation_above_stream(MDT preenchido, drenagem) -> HAND
  8. wetness_index(sca, slope)                        -> TWI

Nenhum caminho é fixo no código: o MDT de entrada e o diretório de saída são parâmetros.

Uso:
    python calcular_hand_twi_dinfinity.py --dtm <entrada.tif> --outdir <dir_saida>
                                         [--stream-percentile 98.0] [--valid-min -1000.0]
                                         [--regiao-rotulo recife]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio

STREAM_PERCENTILE_V12 = 98.0
VALID_MIN_V12 = -1000.0

OUTPUT_NAMES = {
    "dem_filled": "dem_filled.tif",
    "accum_cells": "dinf_accum_cells.tif",
    "sca": "dinf_sca.tif",
    "slope_deg": "slope_deg_wbt.tif",
    "streams": "streams_dinf.tif",
    "hand": "hand_dinf.tif",
    "twi": "twi_dinf.tif",
}


@dataclass
class DinfResult:
    """Caminhos gerados e números reais da execução (vão para o manifesto)."""

    region_label: str
    mdt_entrada: str
    outdir: str
    crs: str
    shape: list
    pixel_size: list
    n_total_cells: int
    n_celulas_validas: int
    drenagem_percentil: float
    drenagem_limiar_celulas: float
    valid_min: float
    versao_wbt: str
    elapsed_s: float
    outputs: dict


def _wbt(work_dir: Path):
    import whitebox

    wbt = whitebox.WhiteboxTools()
    wbt.set_working_dir(str(work_dir.resolve()))
    wbt.set_verbose_mode(False)
    return wbt


def _check(code: int, tool: str) -> None:
    if code != 0:
        raise RuntimeError(f"WhiteboxTools falhou em '{tool}' (exit code {code})")


def _valid_mask(path: Path, valid_min: float) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
    mask = np.isfinite(arr) & (arr > valid_min)
    if nodata is not None:
        mask &= arr != nodata
    return mask


def compute_hand_twi_dinf(
    dtm_path: Path | str,
    outdir: Path | str,
    drenagem_percentil: float = STREAM_PERCENTILE_V12,
    valid_min: float = VALID_MIN_V12,
    region_label: str = "unlabeled",
) -> DinfResult:
    """Roda o encadeamento D-infinity completo sobre `dtm_path`, escrevendo em `outdir`."""
    dtm_path = Path(dtm_path).resolve()
    outdir = Path(outdir).resolve()
    if not dtm_path.exists():
        raise FileNotFoundError(f"MDT de entrada não encontrado: {dtm_path}")
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    wbt = _wbt(outdir)
    paths = {k: outdir / v for k, v in OUTPUT_NAMES.items()}

    _check(
        wbt.fill_depressions_wang_and_liu(str(dtm_path), str(paths["dem_filled"]), fix_flats=True),
        "fill_depressions_wang_and_liu",
    )
    _check(
        wbt.d_inf_flow_accumulation(
            str(paths["dem_filled"]), str(paths["accum_cells"]), out_type="cells"
        ),
        "d_inf_flow_accumulation(cells)",
    )
    _check(
        wbt.d_inf_flow_accumulation(str(paths["dem_filled"]), str(paths["sca"]), out_type="sca"),
        "d_inf_flow_accumulation(sca)",
    )
    _check(
        wbt.slope(str(paths["dem_filled"]), str(paths["slope_deg"]), units="degrees"),
        "slope",
    )

    mask = _valid_mask(dtm_path, valid_min)
    with rasterio.open(paths["accum_cells"]) as src:
        accum = src.read(1)
    threshold = float(np.percentile(accum[mask], drenagem_percentil))
    del accum

    _check(
        wbt.extract_streams(str(paths["accum_cells"]), str(paths["streams"]), threshold),
        "extract_streams",
    )
    _check(
        wbt.elevation_above_stream(
            str(paths["dem_filled"]), str(paths["streams"]), str(paths["hand"])
        ),
        "elevation_above_stream",
    )
    _check(
        wbt.wetness_index(str(paths["sca"]), str(paths["slope_deg"]), str(paths["twi"])),
        "wetness_index",
    )

    with rasterio.open(dtm_path) as src:
        crs = str(src.crs)
        shape = [src.height, src.width]
        pixel_size = [abs(src.transform.a), abs(src.transform.e)]

    return DinfResult(
        region_label=region_label,
        mdt_entrada=str(dtm_path),
        outdir=str(outdir),
        crs=crs,
        shape=shape,
        pixel_size=pixel_size,
        n_total_cells=int(mask.size),
        n_celulas_validas=int(mask.sum()),
        drenagem_percentil=drenagem_percentil,
        drenagem_limiar_celulas=threshold,
        valid_min=valid_min,
        versao_wbt=str(wbt.version()).splitlines()[0].strip(),
        elapsed_s=round(time.time() - t0, 1),
        outputs={k: str(v) for k, v in paths.items()},
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtm", required=True, help="GeoTIFF de entrada (MDT/MDE projetado, unidade metro)")
    p.add_argument("--outdir", required=True, help="Diretório de saída dos rasters")
    p.add_argument("--stream-percentile", type=float, default=STREAM_PERCENTILE_V12)
    p.add_argument("--valid-min", type=float, default=VALID_MIN_V12)
    p.add_argument("--regiao-rotulo", default="unlabeled")
    args = p.parse_args(argv)

    res = compute_hand_twi_dinf(
        args.dtm,
        args.outdir,
        drenagem_percentil=args.drenagem_percentil,
        valid_min=args.valid_min,
        region_label=args.region_label,
    )
    manifest = Path(args.outdir) / "run_manifest.json"
    payload = asdict(res)
    payload["python"] = platform.python_version()
    payload["platform"] = platform.platform()
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"manifesto: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
