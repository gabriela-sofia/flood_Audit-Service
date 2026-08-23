"""Comparação pixel a pixel entre dois rasters de mesma grade.

Usado pela validação de regressão do script D-infinity contra os rasters já existentes do
v12 de Recife. Não faz reamostragem nem reprojeção de propósito: se a grade não bate, é erro,
não algo a corrigir silenciosamente.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def _read(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        return arr, src.nodata, src.transform, src.crs


def _finite_mask(arr: np.ndarray, nodata) -> np.ndarray:
    mask = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        mask &= arr != nodata
    # WhiteboxTools usa -32768 como nodata padrão mesmo quando a tag não é escrita
    mask &= arr > -32767.0
    return mask


def comparar_rasters(path_a: Path | str, path_b: Path | str, rotulo: str = "") -> dict:
    """Estatísticas de diferença entre `path_a` (gerado) e `path_b` (referência)."""
    a, a_nd, a_tr, a_crs = _read(Path(path_a))
    b, b_nd, b_tr, b_crs = _read(Path(path_b))

    grid_match = a.shape == b.shape and a_tr.almost_equals(b_tr, precision=1e-6)
    out = {
        "rotulo": rotulo,
        "path_generated": str(path_a),
        "path_reference": str(path_b),
        "shape_generated": list(a.shape),
        "shape_reference": list(b.shape),
        "crs_generated": str(a_crs),
        "crs_reference": str(b_crs),
        "grid_match": bool(grid_match),
    }
    if not grid_match:
        out["error"] = "grades diferentes (shape ou transform); comparação pixel a pixel impossível"
        return out

    mask = _finite_mask(a, a_nd) & _finite_mask(b, b_nd)
    n = int(mask.sum())
    out["n_valid_generated"] = int(_finite_mask(a, a_nd).sum())
    out["n_valid_reference"] = int(_finite_mask(b, b_nd).sum())
    out["n_compared"] = n
    if n < 2:
        out["error"] = "menos de 2 pixels válidos em comum"
        return out

    av, bv = a[mask], b[mask]
    diff = av - bv
    absdiff = np.abs(diff)
    out.update(
        {
            "pearson_r": float(np.corrcoef(av, bv)[0, 1]),
            "mean_abs_diff": float(absdiff.mean()),
            "max_abs_diff": float(absdiff.max()),
            "mean_signed_diff": float(diff.mean()),
            "p95_abs_diff": float(np.percentile(absdiff, 95)),
            "p99_abs_diff": float(np.percentile(absdiff, 99)),
            "frac_abs_diff_lt_1e-3": float((absdiff < 1e-3).mean()),
            "mean_generated": float(av.mean()),
            "mean_reference": float(bv.mean()),
            "std_generated": float(av.std()),
            "std_reference": float(bv.std()),
        }
    )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generated", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--rotulo", default="")
    p.add_argument("--json-out")
    args = p.parse_args(argv)

    res = comparar_rasters(args.generated, args.reference, args.rotulo)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
