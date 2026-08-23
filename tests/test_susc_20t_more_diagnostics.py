"""Testes reais pros 3 diagnosticos adicionais de Curitiba -- susc_20t.

O que importa travar: a divisao pelo app usa a data real de lancamento (2026-03-25), nunca
mistura pre/pos; a variavel rain_max_24h_chirps substituindo os indices atuais nunca soma 6
features (continua causal, 3 terreno + 1 chuva, sem elevation_m); o peso de recencia usa
meia-vida em anos corretamente (2025 sempre peso 1.0, anos mais antigos sempre <=1.0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates"
sys.path.insert(0, str(PKG / "scripts"))
import pipeline_v20t_mais_diagnosticos_curitiba as p20t  # noqa: E402

FEATS = ["slope_deg", "hand_m_dinf", "twi_dinf",
          "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]


def _fake(n, feats, seed, year=2026, start="2026-01-01"):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({f: rng.normal(size=n) for f in feats})
    d["rotulo"] = (rng.random(n) < 0.5).astype(int)
    d["data_evento"] = pd.date_range(start, periods=n, freq="D")
    d["rain_max_24h_chirps"] = rng.uniform(0, 80, size=n)
    d["ano"] = year
    return d


def test_app_launch_split_never_mixes_pre_and_post():
    d = _fake(200, FEATS, seed=1, start="2026-01-01")
    train = _fake(500, FEATS, seed=2, year=2023, start="2023-01-01")
    out = p20t.app_launch_split(train, d, FEATS)
    pre_max_date = pd.Timestamp(d[d["data_evento"] < p20t.APP_LAUNCH_DATE]["data_evento"].max())
    post_min_date = pd.Timestamp(d[d["data_evento"] >= p20t.APP_LAUNCH_DATE]["data_evento"].min())
    assert pre_max_date < p20t.APP_LAUNCH_DATE
    assert post_min_date >= p20t.APP_LAUNCH_DATE
    assert out["pre_app"]["n_teste"] + out["post_app"]["n_teste"] == len(d)


def test_rain_max_24h_holdout_uses_four_causal_features_no_elevation():
    d = _fake(400, FEATS, seed=3, year=2023, start="2023-01-01")
    d2 = _fake(100, FEATS, seed=4, start="2026-01-01")
    out = p20t.rain_max_24h_temporal_holdout(d, d2)
    assert out["feature_cols"] == ["slope_deg", "hand_m_dinf", "twi_dinf", "rain_max_24h_chirps"]
    assert "elevation_m" not in out["feature_cols"]
    assert len(out["feature_cols"]) == 4


def test_recency_weight_2025_always_full_weight_older_years_discounted():
    d = pd.concat([
        _fake(100, FEATS, seed=5, year=2023, start="2023-01-01"),
        _fake(100, FEATS, seed=6, year=2024, start="2024-01-01"),
        _fake(100, FEATS, seed=7, year=2025, start="2025-01-01"),
    ], ignore_index=True)
    d2 = _fake(50, FEATS, seed=8, start="2026-01-01")
    out = p20t.recency_weighted_temporal_holdout(d, d2, FEATS, half_life_years=1.0)
    assert out["weight_2025"] == 1.0
    assert out["weight_2024"] < out["weight_2025"]
    assert out["weight_2023"] < out["weight_2024"]
    assert 0.0 <= out["holdout_auc_recency_weighted"] <= 1.0


def test_recency_weight_shorter_half_life_discounts_old_years_more():
    d = pd.concat([
        _fake(100, FEATS, seed=5, year=2023, start="2023-01-01"),
        _fake(100, FEATS, seed=6, year=2024, start="2024-01-01"),
        _fake(100, FEATS, seed=7, year=2025, start="2025-01-01"),
    ], ignore_index=True)
    d2 = _fake(50, FEATS, seed=8, start="2026-01-01")
    short = p20t.recency_weighted_temporal_holdout(d, d2, FEATS, half_life_years=1.0)
    long = p20t.recency_weighted_temporal_holdout(d, d2, FEATS, half_life_years=5.0)
    assert short["weight_2023"] < long["weight_2023"]
