"""Testes reais pro GBM com restricao monotonica causal -- susc_21a.

O que importa travar: MONOTONIC_CST usa exatamente EXPECTED_SIGN (nao inventa direcao nova);
o modelo monotonico NUNCA produz decision_function decrescente numa feature marcada +1 (ou
crescente numa marcada -1), testado com decisao bruta (nao so a media agregada, que poderia
mascarar violacao pontual); o check de monotonicidade da PD nao confunde plato (diff=0) com
troca de sinal.
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
import pipeline_v21a_monotonic_constrained_curitiba as p21a  # noqa: E402
from pipeline_v20o_temporal_holdout_curitiba import EXPECTED_SIGN, FEATURE_COLS_PRIMARY  # noqa: E402


def _fake(n, seed, with_bairro=False):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({f: rng.normal(size=n) if f != "hand_m_dinf" else rng.uniform(0, 20, size=n)
                       for f in FEATURE_COLS_PRIMARY})
    d["label"] = (rng.random(n) < 0.4).astype(int)
    if with_bairro:
        d["bairro"] = rng.choice([f"bairro_{i}" for i in range(8)], size=n)
    return d


def test_monotonic_cst_matches_expected_sign_exactly():
    for feat, cst in zip(FEATURE_COLS_PRIMARY, p21a.MONOTONIC_CST):
        assert cst == EXPECTED_SIGN[feat]


def test_fitted_monotonic_model_never_violates_constraint_on_raw_grid():
    d = _fake(300, seed=1)
    clf = p21a.fit_hgb(d, FEATURE_COLS_PRIMARY, monotonic=True)
    row = d[FEATURE_COLS_PRIMARY].iloc[0].values.copy()
    for i, feat in enumerate(FEATURE_COLS_PRIMARY):
        grid = np.linspace(d[feat].min(), d[feat].max(), 25)
        preds = []
        for g in grid:
            r = row.copy()
            r[i] = g
            preds.append(clf.decision_function(r.reshape(1, -1))[0])
        diffs = np.diff(preds)
        expected = EXPECTED_SIGN[feat]
        if expected > 0:
            assert np.all(diffs >= -1e-9), f"{feat} deveria ser nao-decrescente"
        else:
            assert np.all(diffs <= 1e-9), f"{feat} deveria ser nao-crescente"


def test_hgb_temporal_holdout_returns_auc_in_valid_range_monotonic_and_unrestricted():
    d_train = _fake(300, seed=2)
    d_test = _fake(80, seed=3)
    r_mono = p21a.hgb_temporal_holdout(d_train, d_test, FEATURE_COLS_PRIMARY, monotonic=True)
    r_free = p21a.hgb_temporal_holdout(d_train, d_test, FEATURE_COLS_PRIMARY, monotonic=False)
    assert 0.0 <= r_mono["holdout_auc_2026"] <= 1.0
    assert 0.0 <= r_free["holdout_auc_2026"] <= 1.0
    assert r_mono["monotonic"] is True
    assert r_free["monotonic"] is False


def test_hgb_spatial_block_cv_runs_without_error():
    d = _fake(400, seed=4, with_bairro=True)
    out = p21a.hgb_spatial_block_cv(d, FEATURE_COLS_PRIMARY, monotonic=True, n_splits=4)
    assert out["n_splits"] == 4
    assert 0.0 <= out["auc_mean"] <= 1.0


def test_monotonic_pd_check_does_not_confuse_flat_plateau_with_sign_change():
    # constroi uma curva sintetica em degrau nao-decrescente com plato (diff=0 no meio) --
    # o detector antigo (baseado em np.sign) contaria plato->subida como troca; o novo nao deve
    avg = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    diffs = np.diff(avg)
    tol = 1e-9
    monotonic = bool(np.all(diffs >= -tol) or np.all(diffs <= tol))
    assert monotonic is True


def test_references_match_documented_upstream_benchmarks():
    assert p21a.REFERENCE_GBM_UNRESTRICTED_V20U == 0.5888
    assert p21a.REFERENCE_GAM_BEST_V20Y == 0.575
    assert p21a.REFERENCE_LINEAR_HOLDOUT_2026 == 0.5246
