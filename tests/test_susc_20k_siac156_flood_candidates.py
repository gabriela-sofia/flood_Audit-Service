"""Testes reais (sintéticos, não network) pro pipeline SIAC 156 — susc_20k.

Sem chamada de rede (Nominatim) e sem raster real: a lógica de classificação/filtro/percentil é
testada isolada, com dados sintéticos pequenos. A rodada real (33 pontos geocodificados, 0
falhas) fica documentada no relatório do SUSC-20K, não repetida aqui.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import extrair_reclamacoes_enchente_siac156 as extract_mod  # noqa: E402
import geocodificar_nominatim as geocode_mod  # noqa: E402
import adjudicar_candidatos_hand_twi as adjudicate_mod  # noqa: E402


# --- extrair_reclamacoes_enchente_siac156 -----------------------------------------------------

def test_normalize_strips_accents_and_case():
    assert extract_mod.normalize("Cajuru") == "CAJURU"
    assert extract_mod.normalize("Aníbal Maria Difrancia") == "ANIBAL MARIA DIFRANCIA"
    assert extract_mod.normalize(None) == ""


@pytest.mark.parametrize(
    "assunto,subdivisao,expected",
    [
        ("Drenagem", "Alagamentos", True),
        ("Proteção ao cidadão - GM", "Enchentes, inundações ou alagamentos", True),
        ("Trânsito", "Fiscalização de estacionamento irregular", False),
        ("Iluminação pública - Via pública", "Manutenção de luminárias", False),
        (None, None, False),
    ],
)
def test_is_flood_related(assunto, subdivisao, expected):
    assert extract_mod.is_flood_related(assunto, subdivisao) == expected


def _write_siac_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["Tipo", "Orgao", "DataCriacao", "Assunto", "Subdivisao", "Situacao", "Logradouro", "Bairro", "Regional", "DataResposta", "Origem"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_extract_candidates_filters_by_date_and_keyword_and_dedupes(tmp_path):
    csv_path = tmp_path / "siac.csv"
    _write_siac_csv(
        csv_path,
        [
            {"Tipo": "Solicitação", "Orgao": "X", "DataCriacao": "03/02/2025", "Assunto": "Drenagem", "Subdivisao": "Alagamentos", "Situacao": "Aberto", "Logradouro": "Rua A", "Bairro": "XAXIM", "Regional": "", "DataResposta": "", "Origem": "Mobile"},
            {"Tipo": "Solicitação", "Orgao": "X", "DataCriacao": "03/02/2025", "Assunto": "Drenagem", "Subdivisao": "Alagamentos", "Situacao": "Aberto", "Logradouro": "Rua A", "Bairro": "XAXIM", "Regional": "", "DataResposta": "", "Origem": "Mobile"},  # duplicata exata
            {"Tipo": "Solicitação", "Orgao": "X", "DataCriacao": "03/02/2025", "Assunto": "Trânsito", "Subdivisao": "Fiscalização", "Situacao": "Aberto", "Logradouro": "Rua B", "Bairro": "CENTRO", "Regional": "", "DataResposta": "", "Origem": "Mobile"},  # nao eh flood
            {"Tipo": "Solicitação", "Orgao": "X", "DataCriacao": "10/02/2025", "Assunto": "Drenagem", "Subdivisao": "Alagamentos", "Situacao": "Aberto", "Logradouro": "Rua C", "Bairro": "XAXIM", "Regional": "", "DataResposta": "", "Origem": "Mobile"},  # fora da janela
        ],
    )
    events = [extract_mod.EventWindow(janela_id="EV1", data_evento_ddmmaaaa="03/02/2025", expected_bairros={"XAXIM"})]
    rows = extract_mod.extract_candidates([csv_path], events)
    assert len(rows) == 1  # so a linha flood-related, dentro da data, deduplicada
    assert rows[0]["janela_id"] == "EV1"
    assert rows[0]["bairro_esperado"] is True


def test_extract_candidates_marks_bairro_esperado_false_when_not_in_list(tmp_path):
    csv_path = tmp_path / "siac.csv"
    _write_siac_csv(
        csv_path,
        [{"Tipo": "Solicitação", "Orgao": "X", "DataCriacao": "03/02/2025", "Assunto": "Drenagem", "Subdivisao": "Alagamentos", "Situacao": "Aberto", "Logradouro": "Rua Z", "Bairro": "CENTRO", "Regional": "", "DataResposta": "", "Origem": "Mobile"}],
    )
    events = [extract_mod.EventWindow(janela_id="EV1", data_evento_ddmmaaaa="03/02/2025", expected_bairros={"XAXIM"})]
    rows = extract_mod.extract_candidates([csv_path], events)
    assert len(rows) == 1
    assert rows[0]["bairro_esperado"] is False


def test_load_events_parses_bairro_list(tmp_path):
    events_path = tmp_path / "eventos.csv"
    events_path.write_text("janela_id,data_evento_ddmmaaaa,bairros\nEV1,03/02/2025,Xaxim;Portao\n", encoding="utf-8")
    events = extract_mod.load_events(events_path)
    assert len(events) == 1
    assert events[0].expected_bairros == {"XAXIM", "PORTAO"}


# --- geocodificar_nominatim (só a lógica pura, sem chamada de rede) -----------------------------

def test_classify_result_prefers_street_level_over_place_level():
    bbox = (-49.45, -25.65, -49.10, -25.30)
    results = [
        {"lat": "-25.40", "lon": "-49.25", "class": "place", "type": "suburb", "display_name": "Bairro X", "osm_id": 1},
        {"lat": "-25.41", "lon": "-49.26", "class": "highway", "type": "residential", "display_name": "Rua Y", "osm_id": 2},
    ]
    res = geocode_mod.classify_result(results, bbox)
    assert res is not None
    assert res.nivel_confianca == "strong"
    assert res.osm_id == "2"


def test_classify_result_falls_back_to_medium_without_street_level():
    bbox = (-49.45, -25.65, -49.10, -25.30)
    results = [{"lat": "-25.40", "lon": "-49.25", "class": "place", "type": "suburb", "display_name": "Bairro X", "osm_id": 1}]
    res = geocode_mod.classify_result(results, bbox)
    assert res.nivel_confianca == "medium"


def test_classify_result_returns_none_when_outside_bbox():
    bbox = (-49.45, -25.65, -49.10, -25.30)
    results = [{"lat": "-8.0", "lon": "-35.0", "class": "highway", "type": "residential", "display_name": "Rua Recife", "osm_id": 1}]
    res = geocode_mod.classify_result(results, bbox)
    assert res is None


def test_geocode_uses_cache_without_network_call(monkeypatch):
    bbox = (-49.45, -25.65, -49.10, -25.30)
    cache = {
        "Rua A, XAXIM, Curitiba, PR, Brasil": {
            "nivel_confianca": "strong", "lat": -25.5, "lon": -49.26,
            "nominatim_nome_exibicao": "Rua A", "osm_class": "highway", "osm_type": "residential",
            "osm_id": "123", "reason": None,
        }
    }

    def _boom(*args, **kwargs):
        raise AssertionError("não deveria chamar a rede quando já está em cache")

    monkeypatch.setattr(geocode_mod, "_call_nominatim", _boom)
    res = geocode_mod.geocode_one("Rua A", "XAXIM", "Curitiba, PR, Brasil", bbox, opener=None, cache=cache)
    assert res.nivel_confianca == "strong"
    assert res.lat == -25.5


# --- adjudicar_candidatos_hand_twi (rasters sintéticos pequenos) --------------------------

def _write_synthetic_raster(path: Path, data: np.ndarray, crs="EPSG:31982", origin=(680000.0, 7178000.0), pixel=10.0, nodata=-9999.0):
    transform = from_origin(origin[0], origin[1], pixel, pixel)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
        dtype="float32", crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)


def test_adjudicate_point_flags_plausible_flood_signature(tmp_path):
    shape = (20, 20)
    hand = np.full(shape, 20.0)
    hand[10, 10] = 1.0  # ponto de HAND baixo
    twi = np.full(shape, 4.0)
    twi[10, 10] = 12.0  # TWI alto
    slope = np.full(shape, 10.0)
    slope[10, 10] = 1.0  # declividade baixa
    streams = np.zeros(shape)
    streams[10, 12] = 1.0  # drenagem a 2 pixels (20 m) de distância

    _write_synthetic_raster(tmp_path / "hand_dinf.tif", hand)
    _write_synthetic_raster(tmp_path / "twi_dinf.tif", twi)
    _write_synthetic_raster(tmp_path / "slope_deg_wbt.tif", slope)
    _write_synthetic_raster(tmp_path / "streams_dinf.tif", streams)

    arrays, stats = adjudicate_mod._load_regional_stats(tmp_path)
    dist_raster, dist_src = adjudicate_mod._distance_to_stream_raster(tmp_path)

    # centro do pixel [10,10] em coordenadas do CRS sintético
    with rasterio.open(tmp_path / "hand_dinf.tif") as src:
        x, y = src.xy(10, 10)
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(x, y)

    out = adjudicate_mod.adjudicate_point(lat, lon, arrays, stats, dist_raster, dist_src)
    assert out["hand_m"] == pytest.approx(1.0, abs=0.01)
    assert out["twi"] == pytest.approx(12.0, abs=0.01)
    assert out["slope_deg"] == pytest.approx(1.0, abs=0.01)
    assert out["dist_drenagem_m"] == pytest.approx(20.0, abs=0.5)
    assert out["assinatura_enchente_plausivel"] is True


def test_adjudicate_point_rejects_upland_signature(tmp_path):
    shape = (20, 20)
    # metade da grade em valores de vale (HAND/slope baixos, TWI alto), metade em valores de
    # encosta -- o ponto testado [10,10] fica do lado de encosta, pra dar mediana regional
    # não-trivial (raster uniforme deixaria a comparação com a mediana sempre verdadeira)
    hand = np.where(np.indices(shape)[0] < 10, 2.0, 40.0)
    twi = np.where(np.indices(shape)[0] < 10, 12.0, 3.0)
    slope = np.where(np.indices(shape)[0] < 10, 1.0, 30.0)
    streams = np.zeros(shape)
    streams[0, 0] = 1.0

    _write_synthetic_raster(tmp_path / "hand_dinf.tif", hand)
    _write_synthetic_raster(tmp_path / "twi_dinf.tif", twi)
    _write_synthetic_raster(tmp_path / "slope_deg_wbt.tif", slope)
    _write_synthetic_raster(tmp_path / "streams_dinf.tif", streams)

    arrays, stats = adjudicate_mod._load_regional_stats(tmp_path)
    dist_raster, dist_src = adjudicate_mod._distance_to_stream_raster(tmp_path)

    with rasterio.open(tmp_path / "hand_dinf.tif") as src:
        x, y = src.xy(10, 10)
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(x, y)

    out = adjudicate_mod.adjudicate_point(lat, lon, arrays, stats, dist_raster, dist_src)
    assert out["assinatura_enchente_plausivel"] is False


def test_adjudicate_point_outside_raster_extent_returns_status(tmp_path):
    shape = (20, 20)
    for name, val in [("hand_dinf.tif", 5.0), ("twi_dinf.tif", 5.0), ("slope_deg_wbt.tif", 5.0), ("streams_dinf.tif", 0.0)]:
        _write_synthetic_raster(tmp_path / name, np.full(shape, val))

    arrays, stats = adjudicate_mod._load_regional_stats(tmp_path)
    dist_raster, dist_src = adjudicate_mod._distance_to_stream_raster(tmp_path)

    out = adjudicate_mod.adjudicate_point(-8.0, -35.0, arrays, stats, dist_raster, dist_src)  # Recife, longe de Curitiba
    assert out["hand_m"] is None
    assert out["hand_m_status"] == "FORA_DA_EXTENSAO_DO_RASTER"
