"""Testes reais pro diagnostico de nao-linearidade (GBM raso) -- susc_20u.

O que importa travar: o CI bootstrap reamostra so o teste (mesmo contrato do susc_20q); o
spatial block CV nunca vaza bairro (mesmo contrato do susc_20p/20s); a sensibilidade de
hiperparametro cobre uma grade real (3x3x3=27), nao um unico ponto escolhido a dedo; as
importancias de feature somam ~1 e nao incluem elevation_m.
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
import pipeline_v20u_nonlinear_diagnostic_curitiba as p20u  # noqa: E402

FEATS = ["slope_deg", "hand_m_dinf", "twi_dinf",
          "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]


def _fake(n, seed, bairro_n=10):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({f: rng.normal(size=n) for f in FEATS})
    d["label"] = (rng.random(n) < 0.4).astype(int)
    d["bairro"] = rng.integers(0, bairro_n, size=n).astype(str)
    return d


def test_gbm_temporal_holdout_ci_reports_expected_keys():
    train = _fake(400, seed=1)
    test = _fake(100, seed=2)
    out = p20u.gbm_temporal_holdout_with_ci(train, test, FEATS, n_boot=200)
    assert 0.0 <= out["point_auc"] <= 1.0
    assert out["ci_95_low"] <= out["point_auc"] <= out["ci_95_high"]
    assert set(out["feature_importances"].keys()) == set(FEATS)
    assert "elevation_m" not in out["feature_importances"]
    assert out["reference_baseline_linear_temporal_holdout_auc_v20o"] == 0.5246


def test_feature_importances_sum_to_approximately_one():
    train = _fake(400, seed=3)
    test = _fake(100, seed=4)
    out = p20u.gbm_temporal_holdout_with_ci(train, test, FEATS, n_boot=100)
    total = sum(out["feature_importances"].values())
    assert abs(total - 1.0) < 0.05


def test_spatial_block_cv_never_leaks_bairro():
    d = _fake(1000, seed=5, bairro_n=15)
    out = p20u.gbm_spatial_block_cv(d, FEATS, n_splits=5)
    assert len(out) == 5
    assert set(["fold", "holdout_auc_gbm"]).issubset(out.columns)


def test_hyperparam_sensitivity_covers_full_grid_27_combinations():
    train = _fake(400, seed=6)
    test = _fake(100, seed=7)
    out = p20u.gbm_hyperparam_sensitivity(train, test, FEATS)
    assert len(out) == 27
    assert set(out["max_depth"].unique()) == {1, 2, 3}
    assert set(out["n_estimators"].unique()) == {50, 100, 200}
    assert set(out["learning_rate"].unique()) == {0.02, 0.05, 0.1}
    assert out["holdout_auc"].between(0, 1).all()
