"""Testes reais (sintéticos, não network) pro detector SAR Via C — susc_20j."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "outputs_public/data/susc_20j_sentinel1_sar_water_candidates/scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import detectar_candidatos_agua_sar as m  # noqa: E402


def test_to_db_converts_linear_power():
    linear = np.array([0.001, 0.01, 0.1, 1.0])
    db = m.to_db(linear)
    expected = 10 * np.log10(linear)
    assert np.allclose(db, expected)


def test_to_db_passthrough_when_already_db():
    already_db = np.array([-25.0, -15.0, -5.0, -1.0])
    out = m.to_db(already_db)
    assert np.allclose(out, already_db)


def test_physical_gate_flags_new_dark_pixel_not_old_dark_pixel():
    vv_before = np.array([[-10.0, -20.0]])  # segundo pixel ja era agua antes
    vv_after = np.array([[-20.0, -20.0]])   # ambos escuros depois
    gate = m.physical_water_gate_sar(vv_after, vv_before, vv_water_max_db=-17.0)
    assert gate[0, 0] == True  # noqa: E712 — novo (nao era agua antes)
    assert gate[0, 1] == False  # noqa: E712 — ja era agua antes, nao conta como novo


def test_detect_finds_cluster_with_real_drop_and_rejects_small_drop():
    shape = (20, 20)
    vv_before = np.full(shape, -10.0)
    vv_after = np.full(shape, -10.0)
    # cluster real: queda de 8dB (dentro da faixa publicada 3-8dB), fica abaixo do limiar
    vv_after[5:10, 5:10] = -18.0
    # pixel isolado com queda pequena (1dB) nao deve entrar
    vv_after[15, 15] = -11.0

    before = {"VV": vv_before}
    after = {"VV": vv_after}
    res = m.detectar_candidatos_agua_sar(before, after, m.SarDetectionConfig(min_cluster_px=5))
    assert len(res.clusters) == 1
    assert res.clusters[0]["n_pixels"] == 25


def test_vh_agreement_required_rejects_vv_only_signal():
    shape = (10, 10)
    vv_before = np.full(shape, -10.0)
    vv_after = np.full(shape, -10.0)
    vv_after[2:5, 2:5] = -20.0  # VV cai bastante

    vh_before = np.full(shape, -15.0)
    vh_after = np.full(shape, -15.0)  # VH nao muda — nao corrobora

    before = {"VV": vv_before, "VH": vh_before}
    after = {"VV": vv_after, "VH": vh_after}

    cfg_loose = m.SarDetectionConfig(min_cluster_px=5, require_vh_agreement=False)
    res_loose = m.detectar_candidatos_agua_sar(before, after, cfg_loose)
    assert len(res_loose.clusters) == 1  # sem exigir VH, aceita

    cfg_strict = m.SarDetectionConfig(min_cluster_px=5, require_vh_agreement=True)
    res_strict = m.detectar_candidatos_agua_sar(before, after, cfg_strict)
    assert len(res_strict.clusters) == 0  # exigindo VH, rejeita (VH nao corroborou)


def test_extra_mask_restricts_to_corridor():
    shape = (10, 10)
    vv_before = np.full(shape, -10.0)
    vv_after = np.full(shape, -10.0)
    vv_after[1:4, 1:4] = -20.0  # fora do corredor
    vv_after[6:9, 6:9] = -20.0  # dentro do corredor

    mask = np.zeros(shape, dtype=bool)
    mask[5:, 5:] = True

    before = {"VV": vv_before}
    after = {"VV": vv_after}
    res = m.detectar_candidatos_agua_sar(before, after, m.SarDetectionConfig(min_cluster_px=5), extra_mask=mask)
    assert len(res.clusters) == 1
    assert 5 <= res.clusters[0]["centroide_linha"] <= 9


def test_missing_vv_raises():
    with pytest.raises(ValueError, match="VV ausente"):
        m.detectar_candidatos_agua_sar({"VH": np.zeros((3, 3))}, {"VV": np.zeros((3, 3)), "VH": np.zeros((3, 3))})


def test_large_real_scale_performance_no_hang():
    """Regressao direta do bug corrigido no Via B (loop nao vetorizado travava com muitos clusters)."""
    import time
    rng = np.random.default_rng(0)
    shape = (800, 800)
    vv_before = np.full(shape, -10.0) + rng.normal(0, 0.5, shape)
    vv_after = vv_before.copy()
    # varios clusters pequenos espalhados, simulando cena real ruidosa
    for _ in range(500):
        r, c = rng.integers(0, 790), rng.integers(0, 790)
        vv_after[r:r+3, c:c+3] -= 9.0
    before = {"VV": vv_before}
    after = {"VV": vv_after}
    t0 = time.time()
    res = m.detectar_candidatos_agua_sar(before, after, m.SarDetectionConfig(min_cluster_px=3))
    elapsed = time.time() - t0
    assert elapsed < 15.0, f"deteccao demorou {elapsed:.1f}s — regressao de performance"
