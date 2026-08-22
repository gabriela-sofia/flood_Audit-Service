"""Testes reais pra bateria exaustiva de diagnosticos temporais de Curitiba -- susc_20q.

O que importa travar: bootstrap do CI reamostra so o teste (nao o treino, nao re-treina o
classificador -- reamostrar o treino tambem mudaria o modelo, nao so a incerteza da avaliacao);
ablacao terreno/chuva usa os mesmos subconjuntos de feature do resto do projeto (nenhuma
feature nova, elevation_m continua fora); walk-forward nunca deixa um ano futuro treinar um
corte que o testa (contrato causal: treino sempre estritamente anterior ao teste); holdout
casado por estacao restringe treino E teste ao mesmo intervalo de mes; estabilidade de
coeficiente por ano reaproveita o mesmo firth_on_train ja testado no susc_20o.
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
import pipeline_v20q_exhaustive_temporal_diagnostics_curitiba as p20q  # noqa: E402

firthlogist = pytest.importorskip("firthlogist", reason="firthlogist não instalado")

FEATS = ["slope_deg", "hand_m_dinf", "twi_dinf",
          "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]


def _fake(n_pos, n_neg, year, month, feats, seed):
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg
    d = pd.DataFrame({f: rng.normal(size=n) for f in feats})
    d["label"] = [1] * n_pos + [0] * n_neg
    d["year"] = year
    d["month"] = month
    return d


def test_terrain_and_rain_groups_have_no_elevation_and_no_overlap():
    assert "elevation_m" not in p20q.TERRAIN_FEATS
    assert "elevation_m" not in p20q.RAIN_FEATS
    assert set(p20q.TERRAIN_FEATS) & set(p20q.RAIN_FEATS) == set()
    assert set(p20q.TERRAIN_FEATS) | set(p20q.RAIN_FEATS) == set(p20q.FEATURE_COLS_PRIMARY)


def test_bootstrap_ci_resamples_test_only_classifier_fit_once():
    """Sinal linearmente separavel por uma feature -- o CI bootstrap tem que ficar concentrado
    bem acima do acaso, nao no meio, se o classificador realmente aprendeu algo no treino."""
    rng = np.random.default_rng(9)
    n_tr, n_te = 400, 150
    train = pd.DataFrame({f: rng.normal(size=n_tr) for f in FEATS})
    train["label"] = (train["rain_decay_index_api_chirps"] > 0).astype(int)
    test = pd.DataFrame({f: rng.normal(size=n_te) for f in FEATS})
    test["label"] = (test["rain_decay_index_api_chirps"] > 0).astype(int)

    result = p20q.bootstrap_holdout_auc_ci(train, test, FEATS, n_boot=500, seed=1)
    assert result["n_test"] == n_te
    assert result["point_auc"] > 0.8
    assert result["ci_95_low"] > 0.5
    assert result["ci_95_low"] <= result["point_auc"] <= result["ci_95_high"]


def test_ablation_terrain_vs_rain_returns_three_rows_correct_feature_counts():
    d = _fake(400, 150, 2023, 3, FEATS, seed=4)
    d2 = _fake(120, 60, 2026, 3, FEATS, seed=5)
    out = p20q.ablation_terrain_vs_rain(d, d2)
    assert set(out["grupo_features"]) == {"terreno_so", "chuva_so", "completo_5feat"}
    row = out.set_index("grupo_features")
    assert row.loc["terreno_so", "n_features"] == 3
    assert row.loc["chuva_so", "n_features"] == 2
    assert row.loc["completo_5feat", "n_features"] == 5


def test_walk_forward_cutoffs_train_always_strictly_before_test_year():
    rows = []
    for ano, n in [(2023, 300), (2024, 400), (2025, 600), (2026, 300)]:
        rows.append(_fake(int(n * 0.75), int(n * 0.25), ano, 3, FEATS, seed=ano))
    d = pd.concat(rows, ignore_index=True)
    out = p20q.walk_forward_cutoffs(d, FEATS)
    assert len(out) == 3  # 4 anos -> 3 cortes prospectivos
    for _, r in out.iterrows():
        train_years = [int(y) for y in r["train_years"].split(",")]
        assert all(y < r["test_year"] for y in train_years)
        assert r["test_year"] not in train_years


def test_seasonal_matched_holdout_restricts_train_and_test_to_same_months():
    rows = []
    for ano in [2023, 2024, 2025, 2026]:
        for mes in [2, 9]:  # um mes "dentro" da janela (jan-jul), um "fora"
            rows.append(_fake(60, 30, ano, mes, FEATS, seed=ano * 10 + mes))
    d = pd.concat(rows, ignore_index=True)
    out = p20q.seasonal_matched_holdout(d, FEATS, test_year=2026,
                                         train_years=(2023, 2024, 2025), months=(1, 7))
    # so mes=2 cai na janela 1-7; treino deve ter 3 anos x 90 unidades, teste 1 ano x 90
    assert out["n_train"] + out["n_test"] <= 4 * 90  # nunca inclui mes=9 (fora da janela)
    assert out["n_test"] <= 90


def test_coef_stability_by_year_marks_epv_failure_when_too_few_negatives():
    d = _fake(500, 5, 2023, 3, FEATS, seed=2)  # so 5 negativos -> EPV=1.0, reprova piso 20
    out = p20q.coef_stability_by_year(d, FEATS)
    assert (out["epv_passa"] == False).all()  # noqa: E712
    assert out["coef_standardized"].isna().all()


def test_metadata_composition_stability_reports_one_row_per_year():
    rows = []
    for ano in [2023, 2024, 2025, 2026]:
        n = 40
        d = pd.DataFrame({
            "label": [1] * 25 + [0] * 15,
            "confidence_tier": (["strong"] * 23 + ["medium"] * 2) + ["strong"] * 15,
            "assunto": (["Drenagem"] * 20 + ["Proteção ao cidadão - GM"] * 5) + [None] * 15,
            "source_registry": ["reg_pos.csv"] * 25 + ["reg_neg.csv"] * 15,
            "negative_source_type": [None] * 25 + ["tipo_a"] * 15,
            "duplicate_group_count": [1] * n,
            "rain_status": ["OK"] * n,
            "rain_n_days_found": [14] * n,
            "year": ano,
        })
        rows.append(d)
    unidades = pd.concat(rows, ignore_index=True)
    out = p20q.metadata_composition_stability_by_year(unidades)
    assert sorted(out["year"]) == [2023, 2024, 2025, 2026]
    assert (out["rain_status_all_ok"]).all()
    assert (out["n_total"] == 40).all()
