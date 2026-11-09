"""Adapter Sen1/GEOID: alvo comum, desconhecido nunca negativo, suporte e canais. Dados artificiais."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/sar"))
import fontes_sar as fs


def tif(p, a, dtype):
    p.parent.mkdir(parents=True, exist_ok=True)
    a = a if a.ndim == 3 else a[None]
    with rasterio.open(p, "w", driver="GTiff", height=a.shape[1], width=a.shape[2], count=a.shape[0],
                       dtype=dtype, crs="EPSG:32630", transform=from_origin(0, 640, 10, 10)) as f:
        f.write(a.astype(dtype))
    return p


def sha(p):
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def chip(tmp_path):
    fs.limpar_cache()
    fs._HASH_OK.clear()
    y = np.zeros((64, 64), np.uint8)
    y[:8] = 2          # inundada
    y[8:16] = 1        # permanente
    y[16:20] = 255     # desconhecido
    val = np.ones((64, 64), np.uint8)
    val[20:24] = 0     # nao valido
    pos = np.full((2, 64, 64), .05, np.float32)
    pos[:, :8] = .002  # agua escura no posterior
    pos[:, 24:26] = 0  # SAR nao positivo: invisivel
    pre = np.full((2, 64, 64), .05, np.float32)
    pre[:, 26:28] = np.nan  # anterior sem leitura: so afeta o suporte do par
    cam = {"post": tif(tmp_path / "post.tif", pos, "float32"), "pre": tif(tmp_path / "pre.tif", pre, "float32"),
           "label": tif(tmp_path / "label.tif", y, "uint8"), "validity": tif(tmp_path / "val.tif", val, "uint8")}
    cam = {k: str(v) for k, v in cam.items()}
    return {"nome": "E1-1-1:r0:c0", "evento": "E1", "fonte": "geoid", "janela": [0, 0, 64, 64],
            "caminhos": cam, "hashes_arquivos_pai": {k: sha(v) for k, v in cam.items()}, "pre_pos_primario": True,
            "grade": {"forma": [64, 64]}}


def test_alvo_comum_e_desconhecido(chip):
    y = fs.alvo(chip, "pos")
    assert (y[:8] == 1).all() and (y[8:16] == -1).all() and (y[16:20] == -1).all()
    assert (y[20:24] == -1).all() and (y[24:26] == -1).all() and (y[28:] == 0).all()
    assert (y[26:28] == 0).all()                      # pre ausente nao afeta o modo so-pos
    assert (fs.alvo(chip, "pre_pos")[26:28] == -1).all()  # mas sai do suporte do par


def test_controle_pos_par_tem_mesmo_suporte_do_par(chip):
    a, va = fs.entrada(chip, "pre_pos")
    b, vb = fs.entrada(chip, "pos_par")
    assert (va == vb).all()
    assert np.allclose(b[2:], b[:2]) and not np.allclose(a[2:][:, va], a[:2][:, va])
    assert (a[:, ~va] == 0).all() and (b[:, ~vb] == 0).all()


def test_canais_pos_primeiro_e_normalizacao_fixa(chip):
    x, _, _, _ = fs.carregar(chip)
    assert np.isclose(x[0, 0, 0], 10*np.log10(.002), atol=1e-4)     # canal 0 = VV posterior
    assert np.isclose(x[2, 0, 0], 10*np.log10(.05), atol=1e-4)      # canal 2 = VV anterior
    z, _ = fs.entrada(chip, "pos")
    assert z.min() >= -1 and z.max() <= 1
    assert np.allclose(fs.normalizar_fixo(np.full((4, 1, 1), -40.)), -1)


def test_pre_pos_exige_par_admitido(chip):
    chip["pre_pos_primario"] = False
    with pytest.raises(ValueError, match="pre/pos real"):
        fs.entrada(chip, "pre_pos")


def test_arquivo_alterado_e_recusado(chip):
    Path(chip["caminhos"]["label"]).write_bytes(Path(chip["caminhos"]["label"]).read_bytes() + b"x")
    fs.limpar_cache()
    fs._HASH_OK.clear()
    with pytest.raises(ValueError, match="alterado"):
        fs.carregar(chip)


def test_contagem_soma(chip):
    c = fs.contagem(chip, "pos")
    assert c["positivo"] + c["negativo"] + c["ignorado"] == c["total"]
    assert c["positivo"] == 8*64 and c["permanente_ignorado"] == 8*64
