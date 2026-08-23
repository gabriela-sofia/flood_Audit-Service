"""Testes do detector Via B (Sentinel-2 → candidatos a lâmina d'água).

Arrays sintéticos mínimos, construídos para isolar cada regra: o filtro físico de reflectância
absoluta, o voto de cada índice, a regra de consenso 2-de-3, o tamanho mínimo de cluster e o
comportamento fail-closed quando B12 falta. Não busca cena nova — não há Sentinel-2 novo em
disco nesta etapa.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "outputs_public" / "data" / "susc_20h_sentinel2_water_candidates"
    / "scripts" / "detectar_candidatos_agua.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("detectar_candidatos_agua", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["detectar_candidatos_agua"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def det():
    return _load()


SHAPE = (30, 30)


def _scene(b03: float, b08: float, b11: float, b12: float) -> dict:
    return {
        "B03": np.full(SHAPE, b03),
        "B08": np.full(SHAPE, b08),
        "B11": np.full(SHAPE, b11),
        "B12": np.full(SHAPE, b12),
    }


def _paint(scene: dict, sl, **values) -> None:
    for band, value in values.items():
        scene[band][sl] = value


@pytest.fixture
def base_scenes():
    """Cena seca antes e depois; blocos individuais são pintados por cada teste."""
    return _scene(0.10, 0.30, 0.25, 0.20), _scene(0.10, 0.30, 0.25, 0.20)


# --------------------------------------------------------------------------- #
# Índices — fórmulas
# --------------------------------------------------------------------------- #

def test_formulas_dos_indices(det):
    b03, b08, b11, b12 = (np.array([[0.10]]), np.array([[0.30]]),
                          np.array([[0.25]]), np.array([[0.20]]))
    assert det.ndwi(b03, b08)[0, 0] == pytest.approx((0.10 - 0.30) / 0.40)
    assert det.mndwi(b03, b11)[0, 0] == pytest.approx((0.10 - 0.25) / 0.35)
    # Feyisa et al. 2014: 4*(G - SWIR1) - (0.25*NIR + 2.75*SWIR2)
    assert det.awei_nsh(b03, b08, b11, b12)[0, 0] == pytest.approx(
        4 * (0.10 - 0.25) - (0.25 * 0.30 + 2.75 * 0.20)
    )


def test_divisao_por_zero_vira_nan_nao_excecao(det):
    z = np.zeros((2, 2))
    assert np.isnan(det.ndwi(z, z)).all()


# --------------------------------------------------------------------------- #
# Filtro físico obrigatório
# --------------------------------------------------------------------------- #

def test_filtro_fisico_exige_escuro_depois_e_nao_antes(det):
    escuro, claro = np.full((2, 2), 0.05), np.full((2, 2), 0.30)
    assert det.physical_water_gate(escuro, escuro, claro, claro).all()
    # já era escuro antes -> não é mudança, não é candidato
    assert not det.physical_water_gate(escuro, escuro, escuro, escuro).any()
    # continua claro depois -> não é água
    assert not det.physical_water_gate(claro, claro, claro, claro).any()
    # escuro só em NIR, SWIR ainda claro -> não passa (precisa dos dois)
    assert not det.physical_water_gate(escuro, claro, claro, claro).any()


# --------------------------------------------------------------------------- #
# Regra de consenso
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "deltas,esperado",
    [
        ((0.5, 0.5, 0.5), True),    # 3 de 3
        ((0.5, 0.5, 0.0), True),    # 2 de 3
        ((0.5, 0.0, 0.0), False),   # 1 de 3 — insuficiente
        ((0.0, 0.0, 0.0), False),   # 0 de 3
    ],
)
def test_consenso_exige_dois_de_tres(det, deltas, esperado):
    changes = {
        name: (np.full((1, 1), d), 0.1)
        for name, d in zip(("ndwi", "mndwi", "awei_nsh"), deltas)
    }
    mask, votes = det.consensus_mask(changes, required=2)
    assert bool(mask[0, 0]) is esperado
    assert int(votes[0, 0]) == sum(d > 0.1 for d in deltas)


def test_consenso_sem_indice_nenhum_falha_alto(det):
    with pytest.raises(ValueError):
        det.consensus_mask({}, required=2)


# --------------------------------------------------------------------------- #
# Cadeia completa
# --------------------------------------------------------------------------- #

def test_agua_plausivel_vira_candidato(det, base_scenes):
    before, after = base_scenes
    bloco = (slice(5, 15), slice(5, 15))
    _paint(after, bloco, B03=0.10, B08=0.04, B11=0.03, B12=0.02)
    res = det.detectar_candidatos_agua(before, after, det.DetectionConfig(min_cluster_px=20))
    assert len(res.clusters) == 1
    assert res.clusters[0]["n_pixels"] == 100
    assert res.clusters[0]["base_consenso"] == "NDWI_MNDWI_AWEI"
    assert res.clusters[0]["adjudicado"] == "false"


def test_nuvem_brilhante_nao_vira_candidato(det, base_scenes):
    """Nuvem sobe nas três bandas — reprovada pelo filtro físico."""
    before, after = base_scenes
    _paint(after, (slice(5, 15), slice(5, 15)), B03=0.60, B08=0.55, B11=0.50, B12=0.45)
    res = det.detectar_candidatos_agua(before, after, det.DetectionConfig(min_cluster_px=20))
    assert res.clusters == []
    assert res.n_pixels_physical_gate == 0


def test_escuro_com_um_indice_so_nao_passa_no_consenso(det, base_scenes):
    """Passa no filtro físico, mas só o NDWI cruza o limiar de mudança."""
    before, after = base_scenes
    bloco = (slice(5, 15), slice(5, 15))
    _paint(after, bloco, B03=0.10, B08=0.05, B11=0.14, B12=0.20)
    cfg = det.DetectionConfig(min_cluster_px=20, mndwi_delta=0.5, awei_delta=5.0)
    res = det.detectar_candidatos_agua(before, after, cfg)
    assert res.n_pixels_physical_gate == 100
    assert res.n_pixels_consensus == 0
    assert res.clusters == []


def test_cluster_menor_que_o_minimo_e_descartado(det, base_scenes):
    before, after = base_scenes
    _paint(after, (slice(5, 8), slice(5, 8)), B03=0.10, B08=0.04, B11=0.03, B12=0.02)
    res = det.detectar_candidatos_agua(before, after, det.DetectionConfig(min_cluster_px=20))
    assert res.n_clusters_before_size_filter == 1
    assert res.clusters == []


def test_dois_blocos_separados_viram_dois_clusters_ordenados(det, base_scenes):
    before, after = base_scenes
    _paint(after, (slice(2, 12), slice(2, 12)), B03=0.10, B08=0.04, B11=0.03, B12=0.02)
    _paint(after, (slice(20, 25), slice(20, 25)), B03=0.10, B08=0.04, B11=0.03, B12=0.02)
    res = det.detectar_candidatos_agua(before, after, det.DetectionConfig(min_cluster_px=20))
    assert [c["n_pixels"] for c in res.clusters] == [100, 25]


# --------------------------------------------------------------------------- #
# Fail-closed sem B12
# --------------------------------------------------------------------------- #

def test_sem_b12_falha_fechado(det, base_scenes):
    before, after = base_scenes
    for cena in (before, after):
        cena.pop("B12")
    _paint(after, (slice(5, 15), slice(5, 15)), B03=0.10, B08=0.04, B11=0.03)
    with pytest.raises(ValueError, match="B12"):
        det.detectar_candidatos_agua(before, after, det.DetectionConfig(min_cluster_px=20))


def test_fallback_sem_b12_marca_o_cluster_como_mais_fraco(det, base_scenes):
    before, after = base_scenes
    for cena in (before, after):
        cena.pop("B12")
    _paint(after, (slice(5, 15), slice(5, 15)), B03=0.10, B08=0.04, B11=0.03)
    cfg = det.DetectionConfig(min_cluster_px=20, allow_two_index_fallback=True)
    res = det.detectar_candidatos_agua(before, after, cfg)
    assert len(res.clusters) == 1
    assert res.clusters[0]["base_consenso"] == "NDWI_MNDWI_ONLY_WEAKER"
    assert res.indices_available == ["ndwi", "mndwi"]


def test_banda_obrigatoria_ausente_falha_alto(det, base_scenes):
    before, after = base_scenes
    after.pop("B11")
    with pytest.raises(ValueError, match="B11"):
        det.detectar_candidatos_agua(before, after, det.DetectionConfig())


# --------------------------------------------------------------------------- #
# Entrada em disco
# --------------------------------------------------------------------------- #

def test_resolve_bandas_por_nome_de_arquivo(det, tmp_path):
    for nome in ("T23KMQ_20220324T131241_B03_raw.tif", "algo_B08.tif",
                 "x_B11_analytical.tif", "y_B12.tif", "mascara_nuvem.tif"):
        (tmp_path / nome).write_bytes(b"")
    achados = det.resolve_band_files(tmp_path)
    assert set(achados) == {"B03", "B08", "B11", "B12"}
    assert achados["B03"].name.endswith("B03_raw.tif")


def test_grades_diferentes_entre_bandas_falham_alto(det, tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    def escreve(nome, n):
        with rasterio.open(
            tmp_path / nome, "w", driver="GTiff", height=n, width=n, count=1,
            dtype="float32", crs="EPSG:31983", transform=from_origin(0, 0, 10, 10),
        ) as dst:
            dst.write(np.zeros((n, n), dtype="float32"), 1)

    escreve("a_B03.tif", 10)
    escreve("a_B08.tif", 10)
    escreve("a_B11.tif", 12)
    with pytest.raises(ValueError, match="grade"):
        det.load_bands(tmp_path)
