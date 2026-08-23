"""Adjudicação física dos candidatos SIAC 156 geocodificados via HAND/TWI/declividade.

Mesma régua observacional já usada e registrada em
`susc_20g_hand_twi_dinfinity_generico/registries/v20g_observational_reading_adjudicated_points.csv`
pros pontos de Juvevê (Curitiba, enchente confirmada -- HAND baixo, TWI acima da mediana, perto
da drenagem) e Valparaíso (Petrópolis, REBAIXADO -- HAND alto, declividade de encosta, TWI abaixo
da mediana). Reaplica a mesma leitura (não recalcula raster, não recria pipeline D-infinity --
os rasters de Curitiba já existem em `local_runs/susc_20g_hand_twi_dinfinity_generico/curitiba/`,
gerados e validados no SUSC-20G).

Leitura de coerência física, não é variavel nem rótulo: não normaliza, não alimenta modelo, não
compara com ponto negativo. Adiciona ao critério já usado (HAND/TWI/declividade/percentil
regional) a distância à drenagem extraída (`streams_dinf.tif`), calculada por transformada de
distância (euclidiana em pixels métricos), igual ao valor já reportado pra Juvevê (135 m) e
Valparaíso (255 m) -- aqui calculada programaticamente em vez de manual.

Critério de leitura (documentado, não é threshold de modelo):
  - assinatura_enchente_plausivel: HAND <= percentil 50 regional E declividade <= mediana
    regional E TWI >= mediana regional -- mesma leitura qualitativa usada em Juvevê.
  - Não decide adjudicação final sozinho: cada ponto deve ser lido individualmente no relatório,
    igual ao que foi feito pra Valparaíso (que teve sinal físico contra, mas não foi rejeitado
    de imediato -- ressalvas documentadas).

Uso:
    python adjudicar_candidatos_hand_twi.py --geocoded candidatos_geocodificados.csv \
        --raster-dir <dir_com_hand_twi_slope_streams> --out adjudicados.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy import ndimage

RASTERS = {"hand_m": "hand_dinf.tif", "twi": "twi_dinf.tif", "slope_deg": "slope_deg_wbt.tif"}
STREAMS_RASTER = "streams_dinf.tif"


def _load_regional_stats(raster_dir: Path) -> dict:
    """Carrega os rasters inteiros uma única vez pra computar percentil regional e mediana --
    mesmo raciocínio do SUSC-20G (percentil calculado sobre todas as células válidas da região,
    não um recorte local)."""
    stats = {}
    arrays = {}
    for key, fname in RASTERS.items():
        path = raster_dir / fname
        with rasterio.open(path) as src:
            arr = src.read(1)
            nodata = src.nodata
            valid = np.isfinite(arr)
            if nodata is not None:
                valid &= arr != nodata
            valid &= arr > -1000.0
            arrays[key] = (arr, valid, src)
            stats[f"{key}_median_regional"] = float(np.median(arr[valid])) if valid.any() else None
    return arrays, stats


def _distance_to_stream_raster(raster_dir: Path) -> tuple[np.ndarray, rasterio.io.DatasetReader]:
    """Constrói (uma vez) um raster de distância euclidiana métrica até a célula de drenagem
    extraída mais próxima -- usa scipy.ndimage.distance_transform_edt com `sampling` no tamanho
    real do pixel, resultado em metros."""
    path = raster_dir / STREAMS_RASTER
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
        is_stream = np.isfinite(arr) & (arr > 0)
        if nodata is not None:
            is_stream &= arr != nodata
        pixel_size = (abs(src.transform.e), abs(src.transform.a))  # (row, col) em metros
        dist = ndimage.distance_transform_edt(~is_stream, sampling=pixel_size)
        return dist, src


def adjudicate_point(
    lat: float,
    lon: float,
    arrays: dict,
    stats: dict,
    dist_raster: np.ndarray,
    dist_src,
) -> dict:
    out: dict = {"lat": lat, "lon": lon}
    row_col = None
    for key, (arr, valid, src) in arrays.items():
        x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
        b = src.bounds
        if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
            out[key] = None
            out[f"{key}_status"] = "FORA_DA_EXTENSAO_DO_RASTER"
            continue
        row, col = src.index(x, y)
        value = float(arr[row, col])
        nodata = src.nodata
        invalid = (not np.isfinite(value)) or value <= -32767 or (nodata is not None and value == nodata)
        out[key] = None if invalid else round(value, 4)
        out[f"{key}_status"] = "NODATA_NO_PIXEL" if invalid else "OK"
        out.setdefault("crs", str(src.crs))
        out.setdefault("tamanho_pixel_m", round(abs(src.transform.a), 4))
        row_col = (row, col)

    # percentil regional (mesma definição do SUSC-20G: posição na distribuição de todas as
    # células válidas da região, não um raio local)
    for key in ("hand_m", "twi"):
        arr, valid, _ = arrays[key]
        value = out.get(key)
        if value is None:
            out[f"{key}_regional_percentile"] = None
            continue
        pct = float((arr[valid] <= value).sum()) / float(valid.sum()) * 100.0
        out[f"{key}_regional_percentile"] = round(pct, 1)

    out["slope_deg_median_regional"] = round(stats["slope_deg_median_regional"], 4) if stats["slope_deg_median_regional"] is not None else None
    out["twi_median_regional"] = round(stats["twi_median_regional"], 4) if stats["twi_median_regional"] is not None else None

    if row_col is not None:
        row, col = row_col
        if 0 <= row < dist_raster.shape[0] and 0 <= col < dist_raster.shape[1]:
            out["dist_drenagem_m"] = round(float(dist_raster[row, col]), 1)
        else:
            out["dist_drenagem_m"] = None
    else:
        out["dist_drenagem_m"] = None

    # leitura qualitativa -- não é threshold de modelo, é a mesma leitura textual já usada em
    # Juvevê/Valparaíso, formalizada em regra pra não depender de julgamento manual por ponto
    hand_ok = out.get("hand_m") is not None and out["hand_m"] <= np.median(
        arrays["hand_m"][0][arrays["hand_m"][1]]
    )
    slope_ok = out.get("slope_deg") is not None and out["slope_deg"] <= out["slope_deg_median_regional"]
    twi_ok = out.get("twi") is not None and out["twi"] >= out["twi_median_regional"]
    out["assinatura_enchente_plausivel"] = bool(hand_ok and slope_ok and twi_ok)

    return out


def adjudicate_candidates(geocoded_rows: list[dict], raster_dir: Path | str) -> list[dict]:
    raster_dir = Path(raster_dir)
    arrays, stats = _load_regional_stats(raster_dir)
    dist_raster, dist_src = _distance_to_stream_raster(raster_dir)

    out_rows = []
    point_cache: dict[tuple[str, str], dict] = {}
    for row in geocoded_rows:
        if not row.get("lat") or not row.get("lon"):
            merged = dict(row)
            merged.update({"status_adjudicacao": "SEM_COORDENADA_GEOCODIFICADA"})
            out_rows.append(merged)
            continue
        lat, lon = float(row["lat"]), float(row["lon"])
        cache_key = (row["lat"], row["lon"])
        if cache_key not in point_cache:
            point_cache[cache_key] = adjudicate_point(lat, lon, arrays, stats, dist_raster, dist_src)
        physical = point_cache[cache_key]
        merged = dict(row)
        merged.update(physical)
        merged["status_adjudicacao"] = "LIDO"
        out_rows.append(merged)
    return out_rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geocoded", required=True)
    p.add_argument("--raster-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out")
    args = p.parse_args(argv)

    with open(args.geocoded, encoding="utf-8", newline="") as f:
        geocoded_rows = list(csv.DictReader(f))

    out_rows = adjudicate_candidates(geocoded_rows, args.raster_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    n_plausivel = sum(1 for r in out_rows if r.get("assinatura_enchente_plausivel") is True)
    n_lido = sum(1 for r in out_rows if r.get("status_adjudicacao") == "LIDO")
    summary = {"total": len(out_rows), "lidos": n_lido, "assinatura_enchente_plausivel": n_plausivel}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
