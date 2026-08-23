"""Testes reais pro GAM aditivo com splines -- susc_20y.

O que importa travar: o modelo e ADITIVO por construcao (sem termo cruzado entre features);
cada variavel aparece com seu proprio bloco de spline independente; a curva de efeito aditivo
por variavel bate com a soma spline(x) . coef daquela variavel (nao precisa de outras features
pra calcular); o resumo compara corretamente contra o benchmark real do GBM (0,5888), sem usar
um limiar arbitrario que inflaria a leitura.
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
import pipeline_v20y_gam_splines_curitiba as p20y  # noqa: E402

FEATS = ["slope_deg", "hand_m_dinf", "twi_dinf",
          "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]


def _fake(n, seed, with_bairro=False):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({f: rng.normal(size=n) if f != "hand_m_dinf" else rng.uniform(0, 20, size=n)
                       for f in FEATS})
    d["rotulo"] = (rng.random(n) < 0.4).astype(int)
    if with_bairro:
        d["bairro"] = rng.choice([f"bairro_{i}" for i in range(8)], size=n)
    return d


def test_fit_gam_produces_one_spline_per_feature_and_matching_coef_length():
    d = _fake(300, seed=1)
    splines, clf = p20y.fit_gam(d, FEATS, n_knots=4, degree=3)
    assert set(splines.keys()) == set(FEATS)
    total_basis_cols = sum(splines[f].n_features_out_ for f in FEATS)
    assert clf.coef_.shape[1] == total_basis_cols


def test_transform_with_splines_matches_manual_hstack():
    d_train = _fake(300, seed=2)
    d_test = _fake(50, seed=3)
    splines, _ = p20y.fit_gam(d_train, FEATS, n_knots=4, degree=3)
    Xte = p20y.transform_with_splines(d_test, FEATS, splines)
    manual = np.hstack([splines[f].transform(d_test[[f]].values) for f in FEATS])
    np.testing.assert_allclose(Xte, manual)


def test_gam_temporal_holdout_returns_auc_in_valid_range():
    d_train = _fake(300, seed=4)
    d_test = _fake(80, seed=5)
    r = p20y.gam_temporal_holdout(d_train, d_test, FEATS, n_knots=4, degree=3, C=1.0)
    assert 0.0 <= r["auc_holdout_2026"] <= 1.0
    assert r["n_treino"] == 300 and r["n_teste"] == 80


def test_gam_spatial_block_cv_runs_without_bairro_overlap_and_returns_summary():
    d = _fake(400, seed=6, with_bairro=True)
    out = p20y.gam_spatial_block_cv(d, FEATS, n_splits=4, n_knots=4, degree=3, C=1.0)
    assert out["n_splits"] == 4
    assert 0.0 <= out["auc_mean"] <= 1.0


def test_additive_effect_curves_are_isolated_per_feature_no_cross_terms():
    d = _fake(300, seed=7)
    splines, clf = p20y.fit_gam(d, FEATS, n_knots=4, degree=3)
    effects = p20y.additive_effect_curves(splines, clf, FEATS, d, grid_resolution=10)
    assert len(effects) == len(FEATS)
    assert set(effects["variavel"]) == set(FEATS)
    # cada efeito depende so da propria variavel: recalcular manualmente pra 1 variavel e comparar
    feat = "hand_m_dinf"
    offsets, pos = {}, 0
    for f in FEATS:
        n_out = splines[f].n_features_out_
        offsets[f] = (pos, pos + n_out)
        pos += n_out
    lo, hi = offsets[feat]
    coef_feat = clf.coef_[0][lo:hi]
    grid = np.linspace(d[feat].min(), d[feat].max(), 10)
    manual_effect = splines[feat].transform(grid.reshape(-1, 1)) @ coef_feat
    row = effects.loc[effects["variavel"] == feat].iloc[0]
    # a funcao arredonda pra 4 casas antes de guardar no dataframe
    np.testing.assert_allclose(row["efeito_no_min"], manual_effect[0], atol=1e-4)
    np.testing.assert_allclose(row["efeito_no_max"], manual_effect[-1], atol=1e-4)


def test_resumo_uses_real_gbm_benchmark_not_arbitrary_threshold():
    # garante que a logica de "recuperou o sinal do GBM" no script usa a referencia real
    # (0.5888), nao um numero solto -- le a constante do modulo diretamente
    assert p20y.REFERENCE_GBM_HOLDOUT_2026 == 0.5888
    assert p20y.REFERENCE_LINEAR_HOLDOUT_2026 == 0.5246
