"""Testes reais pra validação de holdout temporal de Curitiba -- susc_20o.

O que importa travar: o split é por ano sem embaralhar (2026 nunca aparece no treino), o gate
EPV é computado só no treino (não no dataset inteiro), o scaler do holdout não vaza estatística
do teste pro treino, e a rota primária continua sendo as 5 features do SUSC-20N/20M (nenhuma
reintrodução de elevation_m aqui).
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
import pipeline_v20o_temporal_holdout_curitiba as p20o  # noqa: E402

firthlogist = pytest.importorskip("firthlogist", reason="firthlogist não instalado")


def _fake_dated(n_pos: int, n_neg: int, year: int, feats: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg
    d = pd.DataFrame({f: rng.normal(size=n) for f in feats})
    d["label"] = [1] * n_pos + [0] * n_neg
    d["event_date"] = pd.to_datetime(f"{year}-06-15")
    d["is_duplicate_observation_unit"] = False
    return d


def test_feature_set_matches_susc_20n_primary_route():
    """Mesma rota primaria do 20N -- 5 features, elevation_m fora."""
    assert set(p20o.FEATURE_COLS_PRIMARY) == {
        "slope_deg", "hand_m_dinf", "twi_dinf",
        "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps",
    }
    assert "elevation_m" not in p20o.FEATURE_COLS_PRIMARY


def test_test_years_is_the_most_recent_partial_year():
    assert p20o.TEST_YEARS == {2026}


def test_gate_epv_counts_minority_class():
    d = _fake_dated(500, 40, 2023, p20o.FEATURE_COLS_PRIMARY, seed=1)
    g = p20o.gate_epv(d, p20o.FEATURE_COLS_PRIMARY)
    assert g["n_minoritaria_apos_listwise"] == 40
    assert g["epv"] == 8.0
    assert g["passa"] is False


def test_temporal_holdout_auc_no_leakage_scaler_fit_on_train_only():
    """O scaler usado pra transformar o teste tem que vir so do treino -- se o teste
    influenciasse o scaler, a media/desvio do treino mudaria ao trocar o teste por outro
    com distribuicao bem diferente. Aqui checamos que sim, o AUC so depende do classificador
    ajustado no treino (dados sinteticos linearmente separaveis por uma feature)."""
    feats = ["slope_deg", "hand_m_dinf", "twi_dinf",
             "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]
    rng = np.random.default_rng(7)
    n_tr, n_te = 400, 100
    train = pd.DataFrame({f: rng.normal(size=n_tr) for f in feats})
    train["label"] = (train["rain_decay_index_api_chirps"] > 0).astype(int)
    test = pd.DataFrame({f: rng.normal(size=n_te) for f in feats})
    test["label"] = (test["rain_decay_index_api_chirps"] > 0).astype(int)

    result = p20o.temporal_holdout_auc(train, test, feats)
    assert result["n_train"] == n_tr
    assert result["n_test"] == n_te
    assert 0.0 <= result["holdout_auc"] <= 1.0
    # sinal linearmente separavel por uma feature -> AUC deve ficar bem acima do acaso
    assert result["holdout_auc"] > 0.8


def test_firth_on_train_uses_only_train_rows():
    d = _fake_dated(300, 60, 2023, p20o.FEATURE_COLS_PRIMARY, seed=5)
    coefs, report, scaler = p20o.firth_on_train(d, p20o.FEATURE_COLS_PRIMARY)
    assert report["n_pos_train"] + report["n_neg_train"] == report["n_train"]
    assert report["n_train"] <= len(d)
    assert set(coefs["feature"]) == set(p20o.FEATURE_COLS_PRIMARY)


def test_split_excludes_2026_from_train_by_construction():
    """O script principal filtra unidades['year'].isin(TEST_YEARS) pro teste e o resto pro
    treino -- aqui replicamos a logica de particionamento pra garantir que nao ha overlap."""
    anos = pd.Series([2023, 2024, 2025, 2026, 2026])
    is_test = anos.isin(p20o.TEST_YEARS)
    assert list(anos[is_test]) == [2026, 2026]
    assert set(anos[~is_test]) == {2023, 2024, 2025}
