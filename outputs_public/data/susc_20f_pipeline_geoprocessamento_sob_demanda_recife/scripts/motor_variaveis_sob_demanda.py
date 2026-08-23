"""SUSC-20F -- combina terreno (amostragem real de raster) + chuva (Open-Meteo ao
vivo) num vetor de 6 features compativel com FEATURE_COLS do motor SUSC-20D, para
QUALQUER lat/lon dentro da cobertura real (nao so os 269 pontos conhecidos).

Fail-closed em cada etapa: se terreno OU chuva falharem, retorna None (nunca
preenche com placeholder/media/interpolacao).
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from chuva_sob_demanda import rain_features_for_query
from terreno_sob_demanda import coverage_bbox_wgs84, sample_terrain_features

FEATURE_COLS = ["elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf",
                "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]


def compute_features_on_demand(lat: float, lon: float, end_date: date) -> Optional[dict]:
    """Retorna {variavel: valor} pros 6 FEATURE_COLS, mais metadados de proveniencia,
    ou None se terreno OU chuva nao puderem ser calculados com dado real (nunca
    inventa)."""
    terrain = sample_terrain_features(lat, lon)
    if terrain is None:
        return None
    rain = rain_features_for_query(lat, lon, end_date)
    if rain is None:
        return None

    features = {
        "elevation_m": terrain["elevation_m"],
        "slope_deg": terrain["slope_deg"],
        "hand_m_dinf": terrain["hand_m_dinf"],
        "twi_dinf": terrain["twi_dinf"],
        "rain_peak_residual_orthogonalized": rain["rain_peak_residual_orthogonalized"],
        "rain_decay_index_api_chirps": rain["rain_decay_index_api_chirps"],
    }
    provenance = {
        "terrain_source": "PE3D DTM merged D-infinity (dem_filled/slope_deg_wbt/hand_dinf/twi_dinf.tif, SUSC-20B)",
        "elevation_slope_caveat": "elevation_m/slope_deg amostrados do raster mais recente disponivel, "
                                   "nao bit-identico ao raster original de treino (ver "
                                   "validacao_terreno_sob_demanda.md) -- diferenca media historica "
                                   "~2.7m / ~5.6 graus; hand_m_dinf/twi_dinf sao match exato",
        "rain_source": rain["rain_data_source"],
        "rain_n_days_found": rain["n_days_found"],
        "rain_window_end_date": end_date.isoformat(),
    }
    return {"features": features, "provenance": provenance}


def is_within_terrain_coverage(lat: float, lon: float) -> bool:
    lon_min, lat_min, lon_max, lat_max = coverage_bbox_wgs84()
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max
