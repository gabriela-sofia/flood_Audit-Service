"""Testes reais pra validacao por blocos espaciais (bairro) de Curitiba -- susc_20p.

O que importa travar: nenhum bairro aparece simultaneamente em treino e teste do mesmo fold
(isso e a definicao de "bloco espacial" -- se vazar, o teste perde sentido), a rota primaria
continua sendo as 5 features causais do SUSC-20N/20M/20O (nenhuma reintroducao de elevation_m),
o gate EPV e computado por fold so no treino do fold, e o resumo agrega AUC por fold (nao
mistura fold com holdout temporal).
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
import pipeline_v20p_spatial_block_cv_curitiba as p20p  # noqa: E402

firthlogist = pytest.importorskip("firthlogist", reason="firthlogist não instalado")


def _fake_bairros(feats: list[str], n_bairros: int, per_bairro: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bairros):
        for _ in range(per_bairro):
            row = {f: rng.normal() for f in feats}
            row["bairro"] = f"BAIRRO_{b}"
            row["label"] = int(rng.random() < 0.5)
            rows.append(row)
    return pd.DataFrame(rows)


def test_feature_set_matches_primary_route_no_elevation():
    assert set(p20p.FEATURE_COLS_PRIMARY) == {
        "slope_deg", "hand_m_dinf", "twi_dinf",
        "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps",
    }
    assert "elevation_m" not in p20p.FEATURE_COLS_PRIMARY


def test_no_bairro_appears_in_both_train_and_test_of_same_fold():
    d = _fake_bairros(p20p.FEATURE_COLS_PRIMARY, n_bairros=25, per_bairro=12, seed=3)
    folds = p20p.spatial_block_folds(d, n_splits=5)
    assert len(folds) == 5
    for f in folds:
        assert f["bairro_overlap_train_test"] == []
        bairros_train = set(f["df_train"]["bairro"].unique())
        assert bairros_train.isdisjoint(set(f["bairros_test"]))


def test_every_bairro_is_held_out_exactly_once_across_folds():
    d = _fake_bairros(p20p.FEATURE_COLS_PRIMARY, n_bairros=25, per_bairro=12, seed=3)
    folds = p20p.spatial_block_folds(d, n_splits=5)
    all_test_bairros = []
    for f in folds:
        all_test_bairros.extend(f["bairros_test"])
    assert sorted(all_test_bairros) == sorted(f"BAIRRO_{b}" for b in range(25))
    assert len(all_test_bairros) == len(set(all_test_bairros))


def test_run_spatial_block_cv_reports_per_fold_epv_and_auc():
    d = _fake_bairros(p20p.FEATURE_COLS_PRIMARY, n_bairros=30, per_bairro=40, seed=11)
    out = p20p.run_spatial_block_cv(d, p20p.FEATURE_COLS_PRIMARY, n_splits=5)
    df_folds = out["fold_results"]
    summary = out["summary"]
    assert len(df_folds) == 5
    assert set(["epv_treino", "epv_passa", "holdout_auc", "n_test"]).issubset(df_folds.columns)
    assert summary["n_splits"] == 5
    assert 0.0 <= summary["auc_mean"] <= 1.0
    assert summary["auc_min"] <= summary["auc_mean"] <= summary["auc_max"]
    # referencias fixas usadas pra comparacao no relatorio -- nao podem mudar por acidente
    assert summary["reference_loo_auc_shuffled_v20n"] == 0.6459
    assert summary["reference_temporal_holdout_auc_v20o"] == 0.5246


def test_run_spatial_block_cv_raises_on_group_leakage(monkeypatch):
    """Se alguem trocar GroupKFold por algo que deixe um bairro vazar entre treino e teste, o
    assert dentro de run_spatial_block_cv tem que travar a corrida -- simulamos isso
    injetando um fold corrompido via monkeypatch, exercitando o caminho real do codigo."""
    d = _fake_bairros(p20p.FEATURE_COLS_PRIMARY, n_bairros=10, per_bairro=20, seed=2)
    real_folds = p20p.spatial_block_folds(d, n_splits=5)
    real_folds[0]["bairro_overlap_train_test"] = ["BAIRRO_0"]

    def _fake_folds(unidades, n_splits=p20p.N_SPLITS):
        return real_folds

    monkeypatch.setattr(p20p, "spatial_block_folds", _fake_folds)
    with pytest.raises(AssertionError, match="bairro em treino e teste"):
        p20p.run_spatial_block_cv(d, p20p.FEATURE_COLS_PRIMARY, n_splits=5)
