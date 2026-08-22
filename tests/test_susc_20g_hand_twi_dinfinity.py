"""Testes do estágio SUSC-20G — HAND/TWI D-infinity como unidade genérica.

Dois níveis:

1. Invariantes do encadeamento sobre um MDT sintético (rápido, sem dependência externa):
   grade preservada, HAND >= 0, HAND == 0 na drenagem, limiar = percentil pedido.
2. Regressão real contra Recife: roda o script sobre o MESMO MDT merged de 10 m que gerou o
   v12 e compara pixel a pixel com `hand_dinf.tif`/`twi_dinf.tif` já existentes. É este teste
   que autoriza aplicar o script em qualquer região nova. Se os rasters de referência não
   estiverem montados nesta máquina, o teste é pulado com motivo explícito — nunca passa por
   omissão.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "outputs_public" / "data" / "susc_20g_hand_twi_dinfinity_generico" / "scripts"

# PROJETO é o repositório privado irmão de REV-P; `REVP_PROJETO_ROOT` permite apontar para
# outro local sem editar o teste.
PROJETO_ROOT = Path(os.environ.get("REVP_PROJETO_ROOT", ROOT.parent / "PROJETO"))
RECIFE_V12_DINF_DIR = PROJETO_ROOT / "local_runs" / "recife_modelo_v7_otimizado" / "scratch_dinf"
RECIFE_DTM = RECIFE_V12_DINF_DIR / "recife_dem_10m_raw.tif"

# Critério de aceite definido para esta rodada: sem 0,99 de correlação o script não é
# considerado equivalente ao que produziu o v12, e não pode ser aplicado a região nova.
MIN_PEARSON_R = 0.99


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dinf():
    return _load("compute_hand_twi_dinfinity")


@pytest.fixture(scope="module")
def cmp_mod():
    return _load("compare_rasters")


@pytest.fixture(scope="module")
def prep():
    return _load("prepare_region_dtm")


pytest.importorskip("whitebox", reason="WhiteboxTools (pacote python `whitebox`) não instalado")


def _write_dem(path: Path, arr: np.ndarray, res: float = 10.0, nodata: float = -9999.0):
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype="float32", crs="EPSG:31985", transform=from_origin(0.0, 0.0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype("float32"), 1)


def _synthetic_valley(rows: int = 160, cols: int = 160) -> np.ndarray:
    """Vale em V drenando para o sul, com ruído mínimo para não gerar platôs patológicos."""
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    z = 200.0 - 0.5 * r + 0.2 * np.abs(c - cols // 2)
    z += np.random.default_rng(0).normal(0.0, 1e-3, size=z.shape)
    return z


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory, dinf):
    d = tmp_path_factory.mktemp("dinf_sintetico")
    dem = d / "dem.tif"
    _write_dem(dem, _synthetic_valley())
    result = dinf.compute_hand_twi_dinf(dem, d / "out", region_label="sintetico")
    return result, dem


def test_todos_os_rasters_do_encadeamento_sao_gerados(synthetic_run, dinf):
    result, _ = synthetic_run
    assert set(result.outputs) == set(dinf.OUTPUT_NAMES)
    for key, path in result.outputs.items():
        assert Path(path).exists(), f"raster ausente: {key}"


def test_grade_de_saida_identica_a_de_entrada(synthetic_run):
    result, dem = synthetic_run
    with rasterio.open(dem) as src:
        shape, transform = (src.height, src.width), src.transform
    for key in ("hand", "twi", "dem_filled"):
        with rasterio.open(result.outputs[key]) as out:
            assert (out.height, out.width) == shape
            assert out.transform.almost_equals(transform, precision=1e-9)


def test_limiar_de_drenagem_e_o_percentil_pedido(synthetic_run):
    result, dem = synthetic_run
    with rasterio.open(dem) as src:
        valid = src.read(1) > result.valid_min
    with rasterio.open(result.outputs["accum_cells"]) as src:
        accum = src.read(1)
    esperado = float(np.percentile(accum[valid], result.stream_percentile))
    assert result.stream_threshold_cells == pytest.approx(esperado, rel=1e-12)


def test_hand_e_nao_negativo_e_zera_na_drenagem(synthetic_run):
    result, _ = synthetic_run
    with rasterio.open(result.outputs["hand"]) as src:
        hand, hand_nd = src.read(1).astype("float64"), src.nodata
    with rasterio.open(result.outputs["streams"]) as src:
        streams = src.read(1)
    valid = np.isfinite(hand) & (hand > -32767)
    if hand_nd is not None:
        valid &= hand != hand_nd
    assert valid.sum() > 0
    assert hand[valid].min() >= 0.0
    on_stream = valid & (streams > 0)
    assert on_stream.sum() > 0, "nenhuma célula de drenagem extraída no MDT sintético"
    assert np.allclose(hand[on_stream], 0.0, atol=1e-6)


def test_percentil_maior_produz_drenagem_menor(dinf, tmp_path):
    dem = tmp_path / "dem.tif"
    _write_dem(dem, _synthetic_valley())
    baixo = dinf.compute_hand_twi_dinf(dem, tmp_path / "p90", stream_percentile=90.0)
    alto = dinf.compute_hand_twi_dinf(dem, tmp_path / "p99", stream_percentile=99.0)
    assert alto.stream_threshold_cells > baixo.stream_threshold_cells

    def n_streams(res):
        with rasterio.open(res.outputs["streams"]) as src:
            return int((src.read(1) > 0).sum())

    assert n_streams(alto) < n_streams(baixo)


def test_mdt_inexistente_falha_alto(dinf, tmp_path):
    with pytest.raises(FileNotFoundError):
        dinf.compute_hand_twi_dinf(tmp_path / "nao_existe.tif", tmp_path / "out")


def test_comparador_detecta_grade_diferente(cmp_mod, tmp_path):
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write_dem(a, np.ones((10, 10)))
    _write_dem(b, np.ones((12, 12)))
    res = cmp_mod.compare_rasters(a, b, "grade_diferente")
    assert res["grid_match"] is False
    assert "error" in res


def test_comparador_reconhece_raster_identico(cmp_mod, tmp_path):
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    arr = _synthetic_valley(40, 40)
    _write_dem(a, arr)
    _write_dem(b, arr)
    res = cmp_mod.compare_rasters(a, b, "identico")
    assert res["grid_match"] is True
    assert res["max_abs_diff"] == 0.0
    assert res["pearson_r"] == pytest.approx(1.0, abs=1e-12)


def test_preparo_de_mdt_reamostra_e_preserva_extensao(prep, tmp_path):
    fino = tmp_path / "fino.tif"
    _write_dem(fino, _synthetic_valley(80, 80), res=2.5)
    info = prep.prepare_dtm(fino, tmp_path / "grosso.tif", target_res=10.0, assign_crs="EPSG:31982")
    assert info["resampling"] == "average"
    assert info["output_shape"] == [20, 20]
    assert info["output_crs"] == "EPSG:31982"
    with rasterio.open(fino) as a, rasterio.open(tmp_path / "grosso.tif") as b:
        assert b.bounds.left == pytest.approx(a.bounds.left)
        assert b.bounds.top == pytest.approx(a.bounds.top)
        assert b.nodata == -9999.0


# --------------------------------------------------------------------------- #
# Regressão real contra o v12 de Recife
# --------------------------------------------------------------------------- #

_recife_faltando = [
    str(p) for p in (RECIFE_DTM, RECIFE_V12_DINF_DIR / "hand_dinf.tif",
                     RECIFE_V12_DINF_DIR / "twi_dinf.tif") if not p.exists()
]


@pytest.fixture(scope="module")
def recife_result(dinf, tmp_path_factory):
    if _recife_faltando:
        pytest.skip(
            f"rasters de referência do v12 de Recife não montados nesta máquina: {_recife_faltando}"
        )
    outdir = tmp_path_factory.mktemp("recife_regressao")
    return dinf.compute_hand_twi_dinf(RECIFE_DTM, outdir, region_label="recife_regressao_pytest")


def test_regressao_recife_reproduz_parametros_do_v12(recife_result):
    # o limiar de drenagem documentado no relatório do v12 é 1122.74 células (percentil 98)
    assert recife_result.stream_threshold_cells == pytest.approx(1122.74, abs=0.01)
    assert recife_result.n_valid_cells == 3579890


@pytest.mark.parametrize("raster", ["hand_dinf", "twi_dinf"])
def test_regressao_recife_v12(cmp_mod, recife_result, raster):
    result = recife_result
    key = "hand" if raster == "hand_dinf" else "twi"
    stats = cmp_mod.compare_rasters(
        result.outputs[key], RECIFE_V12_DINF_DIR / f"{raster}.tif", raster
    )
    assert stats["grid_match"] is True
    assert stats["n_compared"] > 3_000_000
    assert stats["pearson_r"] >= MIN_PEARSON_R, (
        f"{raster}: correlação {stats['pearson_r']:.6f} < {MIN_PEARSON_R} — "
        "script NÃO equivalente ao que gerou o v12"
    )
    assert stats["max_abs_diff"] == pytest.approx(0.0, abs=1e-6), (
        f"{raster}: diferença máxima {stats['max_abs_diff']}"
    )
