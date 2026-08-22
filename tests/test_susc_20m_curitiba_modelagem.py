"""Testes reais pra modelagem/validação de Curitiba -- susc_20m.

O que importa travar aqui não é o valor dos coeficientes (isso é resultado, não contrato),
e sim: o gate EPV conta a classe certa, os sinais esperados não foram copiados errado, o
screening univariado não inventa direção, e o ambiente reproduz o v12 de Recife (sem isso,
nenhum número de Curitiba é comparável com o de Recife).
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
import pipeline_v20m_curitiba_primary as p20m  # noqa: E402

firthlogist = pytest.importorskip("firthlogist", reason="firthlogist não instalado")


# --- contrato de features e sinais ---------------------------------------------------------

def test_elevation_is_the_only_feature_dropped_from_recife_set():
    assert set(p20m.FEATURE_COLS_RECIFE_6) - set(p20m.FEATURE_COLS_PRIMARY) == {"elevation_m"}
    assert len(p20m.FEATURE_COLS_PRIMARY) == 5


def test_every_feature_has_an_expected_sign():
    for feat in p20m.FEATURE_COLS_RECIFE_6:
        assert p20m.EXPECTED_SIGN[feat] in (-1, +1)


def test_expected_signs_match_recife_pipeline():
    """Sinais herdados de SUSC-20C -- nenhum foi alterado pra Curitiba."""
    recife = {"elevation_m": -1, "slope_deg": -1, "hand_m_dinf": -1, "twi_dinf": +1,
              "rain_peak_residual_orthogonalized": +1, "rain_decay_index_api_chirps": +1}
    assert p20m.EXPECTED_SIGN == recife


def test_no_feature_is_derived_from_label_or_score():
    proibidos = ("label", "score", "threshold", "susceptib", "prob_")
    for feat in p20m.FEATURE_COLS_RECIFE_6:
        assert not any(p in feat.lower() for p in proibidos)


# --- gate EPV -------------------------------------------------------------------------------

def _fake(n_pos: int, n_neg: int, feats: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = n_pos + n_neg
    d = pd.DataFrame({f: rng.normal(size=n) for f in feats})
    d["label"] = [1] * n_pos + [0] * n_neg
    return d


def test_epv_counts_minority_class_not_majority():
    d = _fake(1000, 40, p20m.FEATURE_COLS_PRIMARY)
    g = p20m.gate_epv(d, p20m.FEATURE_COLS_PRIMARY)
    assert g["n_minoritaria_apos_listwise"] == 40
    assert g["epv"] == 8.0            # 40/5, jamais 1000/5
    assert g["passa"] is False


def test_epv_counts_after_listwise_deletion():
    d = _fake(50, 25, p20m.FEATURE_COLS_PRIMARY)
    d.loc[d.index[-5:], "hand_m_dinf"] = np.nan   # 5 negativos ficam incompletos
    g = p20m.gate_epv(d, p20m.FEATURE_COLS_PRIMARY)
    assert g["n_minoritaria_apos_listwise"] == 20


def test_epv_floor_is_twenty():
    assert p20m.EPV_FLOOR == 20.0
    assert p20m.gate_epv(_fake(500, 100, p20m.FEATURE_COLS_PRIMARY), p20m.FEATURE_COLS_PRIMARY)["passa"] is True
    assert p20m.gate_epv(_fake(500, 99, p20m.FEATURE_COLS_PRIMARY), p20m.FEATURE_COLS_PRIMARY)["passa"] is False


def test_primary_block_refuses_to_run_below_epv_floor(tmp_path):
    d = _fake(500, 10, p20m.FEATURE_COLS_PRIMARY)
    with pytest.raises(SystemExit):
        p20m.run_block("teste", d, p20m.FEATURE_COLS_PRIMARY, primario=True, out=tmp_path)


# --- screening univariado -------------------------------------------------------------------

def test_univariate_reports_observed_direction_faithfully():
    d = pd.DataFrame({"twi_dinf": [1.0] * 30 + [9.0] * 30, "label": [0] * 30 + [1] * 30})
    r = p20m.univariate_screen(d, ["twi_dinf"]).iloc[0]
    assert r["direction_observed"] == "pos>neg"
    assert r["expected_sign"] == 1
    assert bool(r["significant_p05"]) is True


def test_univariate_flags_direction_against_expectation():
    """Sinal contrário ao esperado tem que aparecer como tal, não ser silenciado."""
    d = pd.DataFrame({"twi_dinf": [9.0] * 30 + [1.0] * 30, "label": [0] * 30 + [1] * 30})
    r = p20m.univariate_screen(d, ["twi_dinf"]).iloc[0]
    assert r["direction_observed"] == "pos<neg"
    assert r["expected_sign"] == 1


def test_univariate_skips_feature_without_both_classes():
    d = pd.DataFrame({"twi_dinf": [1.0, 2.0, 3.0], "label": [1, 1, 1]})
    assert len(p20m.univariate_screen(d, ["twi_dinf"])) == 0


# --- ambiente ------------------------------------------------------------------------------

def test_environment_reproduces_recife_v12():
    qa = p20m.qa_reproducao_recife_v12()
    if qa.get("status") == "PULADO_ARQUIVO_AUSENTE":
        pytest.skip("dataset/resultados do v12 de Recife ausentes")
    assert qa["status"] == "REPRODUZIDO"
    assert max(qa["max_abs_diff"].values()) == 0.0


def test_firth_indexes_features_not_intercept():
    """firthlogist devolve pvals_/ci_ com o intercepto NO FIM -- indexar por posição de
    feature (como o pipeline de Recife faz) tem que continuar correto."""
    rng = np.random.default_rng(5)
    n = 300
    X = rng.normal(size=(n, 2))
    d = pd.DataFrame({"a": X[:, 0], "b": X[:, 1]})
    d["label"] = (3.0 * X[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)
    coefs, rep = p20m.firth_multivariate(d, ["a", "b"])
    assert len(coefs) == 2
    linha_a = coefs[coefs.feature == "a"].iloc[0]
    assert linha_a["p_value"] < 0.01                                   # 'a' domina
    assert linha_a["ci_low_95"] <= linha_a["coef_standardized"] <= linha_a["ci_high_95"]
    assert rep["n_used"] == n
