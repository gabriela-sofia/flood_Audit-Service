"""SUSC-20F -- chuva sob demanda via Open-Meteo ERA5-Land archive API (publica, sem
autenticacao). Mesma formula exata do `fetch_rain_leadA_positives.py` (SUSC-20B):
14 dias de lookback terminando no dia anterior a `end_date`, rain_max_24h = maximo
diario da janela, rain_decay_index_api = indice de precipitacao antecedente com
decaimento exponencial k=0.85.

Depois aplica a ortogonalizacao JA TREINADA (beta/intercept da fonte
`open_meteo_era5_land_archive_api`, `v12_orthogonalization_stats.json`, SUSC-20C) --
nao reajusta nada, so aplica a formula linear ja fitada aos 97 pontos dessa fonte.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import requests

DECAY_K = 0.85
LOOKBACK_DAYS = 14

# beta/intercept ja treinados para a fonte open_meteo_era5_land_archive_api
# (v12_orthogonalization_stats.json, SUSC-20C) -- reusados, nao refeitos.
ORTHO_BETA = 0.392
ORTHO_INTERCEPT = 2.3942


def fetch_rain_window(lat: float, lon: float, end_date: date) -> Optional[dict]:
    """Retorna {rain_max_24h, rain_decay_index_api, n_days_found} pra janela de 14
    dias terminando no dia anterior a end_date, ou None se a API falhar ou nenhum
    dia real for encontrado (fail-closed, nunca inventa chuva)."""
    start = end_date - timedelta(days=LOOKBACK_DAYS)
    last = end_date - timedelta(days=1)
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}&start_date={start.isoformat()}&end_date={last.isoformat()}"
        "&daily=precipitation_sum&timezone=America%2FRecife"
    )
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        j = r.json()
    except Exception:
        return None

    daily = j.get("daily", {})
    times = daily.get("time", [])
    vals = daily.get("precipitation_sum", [])
    series = {t: v for t, v in zip(times, vals)}
    ordered = [series.get((end_date - timedelta(days=k)).isoformat()) for k in range(LOOKBACK_DAYS, 0, -1)]
    ordered = [np.nan if v is None else float(v) for v in ordered]
    arr = np.array(ordered, dtype=float)
    n_found = int(np.sum(~np.isnan(arr)))
    if n_found == 0:
        return None

    rain_max = float(np.nanmax(arr))
    filled = np.where(np.isnan(arr), 0.0, arr)
    api = 0.0
    for idx, p in enumerate(filled):
        day_offset = LOOKBACK_DAYS - idx
        api += p * (DECAY_K ** (day_offset - 1))

    return {"rain_max_24h_chirps": round(rain_max, 2),
            "rain_decay_index_api_chirps": round(float(api), 3),
            "n_days_found": n_found}


def rain_features_for_query(lat: float, lon: float, end_date: date) -> Optional[dict]:
    """Feature completo pronto pro motor: rain_decay_index_api_chirps +
    rain_peak_residual_orthogonalized (usando os coeficientes ja treinados da fonte
    open_meteo_era5_land_archive_api)."""
    base = fetch_rain_window(lat, lon, end_date)
    if base is None:
        return None
    residual = base["rain_max_24h_chirps"] - (ORTHO_INTERCEPT + ORTHO_BETA * base["rain_decay_index_api_chirps"])
    return {
        "rain_decay_index_api_chirps": base["rain_decay_index_api_chirps"],
        "rain_peak_residual_orthogonalized": round(float(residual), 4),
        "rain_max_24h_chirps": base["rain_max_24h_chirps"],
        "n_days_found": base["n_days_found"],
        "rain_data_source": "open_meteo_era5_land_archive_api",
    }
