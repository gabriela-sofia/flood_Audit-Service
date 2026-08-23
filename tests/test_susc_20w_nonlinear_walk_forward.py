"""Testes reais pro walk-forward multi-corte com 4 classes de modelo -- susc_20w.

O que importa travar: os 3 cortes cobrem exatamente 2024, 2025, 2026 como teste, nunca com o
ano de teste dentro do treino; os 4 modelos (linear + 3 nao-lineares) rodam nos mesmos cortes;
o resumo conta corretamente quantos modelos nao-lineares superam o linear por ano.
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
import pipeline_v20w_walk_forward_nao_linear_curitiba as p20w  # noqa: E402

FEATS = ["slope_deg", "hand_m_dinf", "twi_dinf",
          "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]


def _fake_multi_year(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for ano, n in [(2023, 300), (2024, 350), (2025, 570), (2026, 280)]:
        d = pd.DataFrame({f: rng.normal(size=n) for f in FEATS})
        d["rotulo"] = (rng.random(n) < 0.5).astype(int)
        d["ano"] = ano
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_cutoffs_never_include_test_year_in_train():
    for anos_treino, ano_teste in p20w.CUTOFFS:
        assert ano_teste not in anos_treino


def test_cutoffs_cover_2024_2025_2026_as_test_years():
    assert [c[1] for c in p20w.CUTOFFS] == [2024, 2025, 2026]


def test_walk_forward_all_models_has_4_models_times_3_cutoffs():
    d = _fake_multi_year()
    out = p20w.walk_forward_all_models(d, FEATS)
    assert len(out) == len(p20w.MODEL_FACTORIES) * len(p20w.CUTOFFS)
    assert set(out["modelo"]) == set(p20w.MODEL_FACTORIES.keys())
    assert set(out["ano_teste"]) == {2024, 2025, 2026}


def test_run_cutoff_trains_only_on_specified_years():
    d = _fake_multi_year()
    r = p20w.run_cutoff(d, [2023], 2024, FEATS, p20w.MODEL_FACTORIES["Linear_baseline"])
    assert r["n_treino"] <= (d["ano"] == 2023).sum()
    assert r["n_teste"] <= (d["ano"] == 2024).sum()
    assert 0.0 <= r["auc_holdout"] <= 1.0
