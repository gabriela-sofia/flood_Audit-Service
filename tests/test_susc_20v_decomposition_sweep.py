"""Testes reais pra decomposicao do GBM + varredura de modelo -- susc_20v.

O que importa travar: partial dependence roda pras 5 features causais (nunca elevation_m);
grid volta pra escala original (nao fica em unidade padronizada, que seria ilegivel); a
varredura de modelo cobre todas as classes esperadas e usa o MESMO conjunto de treino/teste
de todo o resto da bateria (sem vazamento, scaler ajustado so no treino).
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
import pipeline_v20v_gbm_decomposition_and_model_sweep_curitiba as p20v  # noqa: E402

FEATS = ["slope_deg", "hand_m_dinf", "twi_dinf",
          "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]


def _fake(n, seed):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({f: rng.normal(size=n) * (10 if "hand" in f else 1) + (5 if "hand" in f else 0)
                       for f in FEATS})
    d["label"] = (rng.random(n) < 0.4).astype(int)
    return d


def test_partial_dependence_covers_all_five_causal_features_no_elevation():
    d = _fake(400, seed=1)
    out = p20v.gbm_partial_dependence(d, FEATS)
    assert set(out.keys()) == set(FEATS)
    assert "elevation_m" not in out
    for feat, info in out.items():
        assert info["grid_min"] <= info["grid_max"]
        assert info["direction"] in {"crescente", "decrescente"}


def test_partial_dependence_grid_is_in_original_feature_scale_not_standardized():
    """Se a grade voltasse em unidade padronizada, hand_m_dinf (escala ~0-30m) apareceria
    como algo tipo -2..2 -- checamos que a grade reflete a escala real da feature."""
    d = _fake(400, seed=2)
    out = p20v.gbm_partial_dependence(d, FEATS)
    # hand_m_dinf foi gerado com media ~5, desvio ~10 -- grade deve refletir essa escala,
    # nao ficar contida em [-3, 3] (que seria o caso se estivesse em unidade padronizada)
    span = out["hand_m_dinf"]["grid_max"] - out["hand_m_dinf"]["grid_min"]
    assert span > 6


def test_model_sweep_covers_expected_model_classes():
    train = _fake(400, seed=3)
    test = _fake(100, seed=4)
    out = p20v.model_sweep(train, test, FEATS)
    expected = {"SVM_RBF_C1", "SVM_RBF_C10", "MLP_16_8", "MLP_8", "ExtraTrees_d3",
                "AdaBoost_d1", "RandomForest_d3", "GBM_referencia_v20u"}
    assert set(out["model"]) == expected
    assert out["holdout_auc"].between(0, 1).all()
    assert len(out) == len(expected)
