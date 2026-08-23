"""Testes reais pra correlacao ONI/ENOS vs. colapso de AUC -- susc_20r.

O que importa travar: 2026 nunca tem valor ONI inventado (a fonte nao publica -- se alguem
adicionar um numero pra 2026 no futuro, tem que vir com fonte, nao ser estimado aqui); a media
jan-jul usa exatamente os mesmos 5 meses (JFM,FMA,MAM,AMJ,MJJ) pra todo ano, sem mistura de
janela; a leitura nao afirma suporte a hipotese ENOS quando os dados reais nao sustentam isso.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates"
sys.path.insert(0, str(PKG / "scripts"))
import pipeline_v20r_correlacao_oni_enso_curitiba as p20r  # noqa: E402


def test_2026_has_no_fabricated_oni_value():
    """2026 nao pode ter numero de ONI -- a fonte consultada nao publica isso ainda."""
    assert p20r.ONI_JAN_JUL[2026] == {}
    assert p20r.mean_oni_jan_jul(2026) is None


def test_mean_uses_same_five_months_every_year():
    meses_esperados = {"JFM", "FMA", "MAM", "AMJ", "MJJ"}
    for ano in [2023, 2024, 2025]:
        assert set(p20r.ONI_JAN_JUL[ano].keys()) == meses_esperados


def test_mean_oni_jan_jul_matches_manual_average():
    manual = round((-0.29 + -0.02 + 0.27 + 0.57 + 0.84) / 5, 4)
    assert p20r.mean_oni_jan_jul(2023) == manual


def test_comparison_table_has_one_row_per_year_2023_to_2026():
    rows = p20r.build_comparison_table()
    assert [r["ano"] for r in rows] == [2023, 2024, 2025, 2026]
    row_2026 = next(r for r in rows if r["ano"] == 2026)
    assert row_2026["oni_publicado"] is False
    assert row_2026["oni_jan_jul_mean"] is None


def test_walk_forward_auc_reference_matches_susc_20q_committed_values():
    """Esses numeros vem do susc_20q ja commitado -- se mudarem aqui sem mudar la, alguem
    editou um dos dois sem atualizar o outro."""
    assert p20r.WALK_FORWARD_TEST_AUC[2024] == 0.6282
    assert p20r.WALK_FORWARD_TEST_AUC[2025] == 0.6652
    assert p20r.WALK_FORWARD_TEST_AUC[2026] == 0.5246
