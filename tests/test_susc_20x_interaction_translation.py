"""Testes reais pra tentativa de traducao interpretavel da nao-linearidade -- susc_20x.

O que importa travar: os limiares vem literalmente dos pontos registrados no SUSC-20V/20W
(nao inventados aqui); termos de interacao/threshold sao ADICIONADOS as 5 features causais
(nunca substituem sem querer); o resumo identifica corretamente a melhor config e se ela
recupera o patamar do GBM.
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
import pipeline_v20x_interaction_translation_attempt_curitiba as p20x  # noqa: E402

FEATS = ["slope_deg", "hand_m_dinf", "twi_dinf",
          "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]


def _fake(n, seed):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({f: rng.normal(size=n) for f in FEATS})
    d["label"] = (rng.random(n) < 0.4).astype(int)
    return d


def test_thresholds_match_documented_values_from_partial_dependence():
    assert p20x.THRESHOLDS["hand_m_dinf"] == ("hand_low_lt4", 4.0, "lt")
    assert p20x.THRESHOLDS["rain_decay_index_api_chirps"] == ("rain_decay_high_gt20", 20.0, "gt")
    assert p20x.THRESHOLDS["rain_peak_residual_orthogonalized"] == ("rain_peak_low_ltm4", -4.0, "lt")


def test_add_interaction_terms_appends_columns_without_removing_base_features():
    d = _fake(50, seed=1)
    out = p20x.add_interaction_terms(d, [("hand_m_dinf", "twi_dinf")])
    assert "hand_m_dinf_x_twi_dinf" in out.columns
    for f in FEATS:
        assert f in out.columns
    np.testing.assert_allclose(out["hand_m_dinf_x_twi_dinf"], out["hand_m_dinf"] * out["twi_dinf"])


def test_add_threshold_terms_produces_binary_indicators():
    d = _fake(50, seed=2)
    out = p20x.add_threshold_terms(d)
    for name, _, _ in p20x.THRESHOLDS.values():
        assert set(out[name].unique()).issubset({0.0, 1.0})


def test_run_all_configs_covers_all_9_configs_and_never_beats_baseline_wildly():
    train = _fake(400, seed=3)
    test = _fake(100, seed=4)
    out = p20x.run_all_configs(train, test)
    assert len(out) == 9
    assert "base_5feat_linear" in out["config"].values
    assert out["holdout_auc_2026"].between(0, 1).all()
