"""Testes reais pra interacao tensor-spline (te()-like) -- susc_20z.

O que importa travar: a base tensor tem exatamente n_out_f1 * n_out_f2 colunas (produto
tensorial, nao soma); adicionar 0 pares de interacao reproduz o GAM aditivo puro (mesmo
n_variaveis do modelo so-aditivo); resumo compara contra os benchmarks reais (GAM aditivo do
SUSC-20Y e GBM do SUSC-20U), nao contra numero solto.
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
import pipeline_v20z_gam_interacao_tensor_curitiba as p20z  # noqa: E402

FEATS = ["slope_deg", "hand_m_dinf", "twi_dinf",
          "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]


def _fake(n, seed):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({f: rng.normal(size=n) if f != "hand_m_dinf" else rng.uniform(0, 20, size=n)
                       for f in FEATS})
    d["rotulo"] = (rng.random(n) < 0.4).astype(int)
    return d


def test_tensor_basis_has_product_not_sum_of_dimensions():
    d = _fake(200, seed=1)
    sp1, sp2 = p20z.fit_tensor_spline_pair(d, "hand_m_dinf", "twi_dinf", n_knots=4, degree=2)
    basis = p20z.tensor_basis(d, "hand_m_dinf", "twi_dinf", sp1, sp2)
    k1 = sp1.transform(d[["hand_m_dinf"]].values).shape[1]
    k2 = sp2.transform(d[["twi_dinf"]].values).shape[1]
    assert basis.shape == (200, k1 * k2)


def test_zero_interaction_pairs_matches_pure_additive_gam_feature_count():
    d_train = _fake(300, seed=2)
    d_test = _fake(80, seed=3)
    r_no_tensor = p20z.gam_tensor_holdout_auc(d_train, d_test, FEATS, [])
    from pipeline_v20y_gam_splines_curitiba import fit_gam
    splines, clf = fit_gam(d_train, FEATS, n_knots=p20z.DEFAULT_N_KNOTS, degree=p20z.DEFAULT_DEGREE)
    assert r_no_tensor["n_variaveis_total"] == clf.coef_.shape[1]


def test_holdout_auc_in_valid_range_with_and_without_tensor():
    d_train = _fake(300, seed=4)
    d_test = _fake(80, seed=5)
    r0 = p20z.gam_tensor_holdout_auc(d_train, d_test, FEATS, [])
    r1 = p20z.gam_tensor_holdout_auc(d_train, d_test, FEATS, [("hand_m_dinf", "twi_dinf")])
    assert 0.0 <= r0["auc_holdout_2026"] <= 1.0
    assert 0.0 <= r1["auc_holdout_2026"] <= 1.0
    assert r1["n_variaveis_total"] > r0["n_variaveis_total"]


def test_run_all_pair_configs_covers_5_configs():
    d_train = _fake(300, seed=6)
    d_test = _fake(80, seed=7)
    df = p20z.run_all_pair_configs(d_train, d_test)
    assert len(df) == 5
    assert "additive_only_no_tensor" in df["config"].values
    assert "+all_3_tensor_interactions" in df["config"].values


def test_references_match_documented_upstream_benchmarks():
    assert p20z.REFERENCE_GAM_ADDITIVE_BEST == 0.575
    assert p20z.REFERENCE_GBM_HOLDOUT_2026 == 0.5888
    assert p20z.REFERENCE_LINEAR_HOLDOUT_2026 == 0.5246
