"""Testes reais (sintéticos, sem rede) pra engenharia de features de Curitiba -- susc_20l.

Cobre o que dá pra verificar offline: a fórmula de chuva tem que ser literalmente a mesma
de `baixar_chuva_positivos_lead_a.py` (SUSC-20B), a ortogonalização tem que produzir resíduo
de fato ortogonal, e o dataset publicado tem que ser consistente com os registries de
entrada (nada inventado, nada perdido).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts"
REGISTRIES = ROOT / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/registries"
sys.path.insert(0, str(SCRIPTS))
import montar_variaveis_curitiba_v20l as b20l  # noqa: E402

DATASET = REGISTRIES / "v20l_dataset_curitiba_variaveis_v1.csv"


# --- fórmula de chuva: equivalência literal ao script de Recife -----------------------------

def _recife_reference_formula(daily_values: list[float]) -> tuple[float, float]:
    """Cópia literal do trecho de cálculo de baixar_chuva_positivos_lead_a.py (SUSC-20B),
    para servir de oráculo independente."""
    arr = np.array(daily_values, dtype=float)
    rain_max = float(np.nanmax(arr))
    filled = np.where(np.isnan(arr), 0.0, arr)
    api = 0.0
    for idx, p in enumerate(filled):
        day_offset = 14 - idx
        api += p * (0.85 ** (day_offset - 1))
    return round(rain_max, 2), round(float(api), 3)


def _series_from(values, event: date) -> dict:
    from datetime import timedelta
    return {(event - timedelta(days=k)).isoformat(): v
            for k, v in zip(range(14, 0, -1), values)}


def test_rain_window_matches_recife_formula():
    event = date(2024, 3, 15)
    values = [0.0, 2.5, 0.0, 11.3, 0.0, 0.0, 4.1, 0.0, 33.8, 1.2, 0.0, 0.0, 7.9, 0.4]
    got = b20l.rain_window_features(_series_from(values, event), event)
    exp_max, exp_api = _recife_reference_formula(values)
    assert got["rain_max_24h_chirps"] == exp_max
    assert got["rain_decay_index_api_chirps"] == exp_api
    assert got["rain_n_days_found"] == 14
    assert got["rain_status"] == "OK"


def test_rain_window_excludes_event_day():
    """O dia do evento nunca entra na janela -- é a trava anti-vazamento herdada do v12."""
    event = date(2024, 3, 15)
    series = _series_from([0.0] * 14, event)
    series[event.isoformat()] = 999.0
    got = b20l.rain_window_features(series, event)
    assert got["rain_max_24h_chirps"] == 0.0


def test_rain_decay_weights_recent_days_more():
    event = date(2024, 3, 15)
    recente = b20l.rain_window_features(_series_from([0.0] * 13 + [10.0], event), event)
    antigo = b20l.rain_window_features(_series_from([10.0] + [0.0] * 13, event), event)
    assert recente["rain_decay_index_api_chirps"] > antigo["rain_decay_index_api_chirps"]


def test_rain_partial_window_is_flagged_not_imputed():
    event = date(2024, 3, 15)
    series = _series_from([1.0] * 14, event)
    for k in list(series)[:5]:
        del series[k]
    got = b20l.rain_window_features(series, event)
    assert got["rain_n_days_found"] == 9
    assert got["rain_status"] == "JANELA_PARCIAL"


def test_rain_empty_window_fails_closed():
    got = b20l.rain_window_features({}, date(2024, 3, 15))
    assert np.isnan(got["rain_max_24h_chirps"])
    assert got["rain_status"] == "SEM_DIA_REAL_NA_JANELA"


# --- ortogonalização -----------------------------------------------------------------------

def test_orthogonalization_removes_correlation():
    rng = np.random.default_rng(7)
    api = rng.uniform(0, 60, 300)
    df = pd.DataFrame({
        "rain_decay_index_api_chirps": api,
        "rain_max_24h_chirps": 3.0 + 0.5 * api + rng.normal(0, 4, 300),
        "rain_data_source": "fonte_unica",
    })
    out, rep = b20l.orthogonalize_per_source(df)
    assert abs(rep["fonte_unica"]["pearson_r_before_orthogonalization"]) > 0.5
    assert abs(rep["fonte_unica"]["pearson_r_after_orthogonalization"]) < 1e-6
    assert abs(float(out["rain_peak_residual_orthogonalized"].mean())) < 1e-9


def test_orthogonalization_is_fit_per_source_not_pooled():
    rng = np.random.default_rng(11)
    a = rng.uniform(0, 20, 150)
    b = rng.uniform(50, 90, 150)
    df = pd.DataFrame({
        "rain_decay_index_api_chirps": np.concatenate([a, b]),
        "rain_max_24h_chirps": np.concatenate([1.0 + 0.2 * a, 40.0 + 1.5 * b]),
        "rain_data_source": ["A"] * 150 + ["B"] * 150,
    })
    _, rep = b20l.orthogonalize_per_source(df)
    assert set(rep) == {"A", "B"}
    assert rep["A"]["beta"] != rep["B"]["beta"]


def test_orthogonalization_skips_rows_without_rain():
    df = pd.DataFrame({
        "rain_decay_index_api_chirps": [1.0, 2.0, 3.0, np.nan],
        "rain_max_24h_chirps": [2.0, 4.0, 6.0, 10.0],
        "rain_data_source": ["A"] * 4,
    })
    out, rep = b20l.orthogonalize_per_source(df)
    assert rep["A"]["n_used"] == 3
    assert pd.isna(out.loc[3, "rain_peak_residual_orthogonalized"])


# --- integridade do dataset publicado ------------------------------------------------------

@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    if not DATASET.exists():
        pytest.skip(f"dataset ainda nao gerado: {DATASET.name}")
    return pd.read_csv(DATASET)


def test_dataset_preserves_every_input_row(dataset):
    pos = pd.read_csv(REGISTRIES / "v20k2_dataset_positivos_curitiba_siac156_v1.csv")
    neg = pd.read_csv(REGISTRIES / "v20k3_dataset_negativos_curitiba_siac156_v1.csv")
    assert len(dataset) == len(pos) + len(neg)
    assert int((dataset.rotulo == 1).sum()) == len(pos)
    assert int((dataset.rotulo == 0).sum()) == len(neg)


def test_dataset_preserves_provenance_columns(dataset):
    pos = pd.read_csv(REGISTRIES / "v20k2_dataset_positivos_curitiba_siac156_v1.csv")
    neg = pd.read_csv(REGISTRIES / "v20k3_dataset_negativos_curitiba_siac156_v1.csv")
    assert set(pos.columns) | set(neg.columns) <= set(dataset.columns)


def test_dataset_has_target_schema(dataset):
    alvo = ["elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf",
            "rain_max_24h_chirps", "rain_decay_index_api_chirps",
            "mapbiomas_class_2023", "rain_data_source",
            "rain_peak_residual_orthogonalized"]
    assert set(alvo) <= set(dataset.columns)


def test_no_feature_derived_from_label(dataset):
    """Nenhuma variavel pode ser função do rotulo, de escore ou de threshold."""
    proibidos = ("escore", "threshold", "susceptib", "risk_index", "prob_")
    for col in dataset.columns:
        assert not any(p in col.lower() for p in proibidos), col


def test_every_missing_feature_has_explicit_status(dataset):
    for feat in ["elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf"]:
        faltando = dataset[feat].isna()
        assert (dataset.loc[faltando, f"{feat}_status"] != "OK").all()
        assert (dataset.loc[~faltando, f"{feat}_status"] == "OK").all()


def test_terrain_values_are_physically_plausible_for_curitiba(dataset):
    ok = dataset[dataset.elevation_m_status == "OK"]
    assert ok.elevation_m.between(850, 1100).all()   # planalto curitibano
    assert dataset.slope_deg.dropna().between(0, 90).all()
    assert (dataset.hand_m_dinf.dropna() >= 0).all()


def test_duplicate_observation_units_share_identical_features(dataset):
    """Linhas que são a mesma ocorrência (mesmo lat/lon/data) têm que ter variavel idêntica --
    se divergirem, a extração não é determinística."""
    feats = ["elevation_m", "slope_deg", "twi_dinf", "rain_max_24h_chirps",
             "rain_decay_index_api_chirps", "mapbiomas_class_2023"]
    n_distintos = dataset.groupby("chave_unidade_observacao")[feats].nunique().max()
    assert (n_distintos <= 1).all()


def test_rain_source_is_single_and_declared(dataset):
    assert set(dataset.rain_data_source.unique()) == {"open_meteo_era5_land_archive_api"}
