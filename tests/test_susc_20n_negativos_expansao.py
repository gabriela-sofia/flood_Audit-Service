"""Testes reais (sintéticos, sem rede) pro builder de dataset negativo -- susc_20n.

Trava as 3 regras que decidem quem entra: aceite strong/medium (nunca `failed`), remoção por
colisão <30 m com positivo, e união com a geração anterior preservando `ponto_id` verbatim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates"
sys.path.insert(0, str(PKG / "scripts"))
import montar_dataset_negativos_final as neg  # noqa: E402


def _cand(logradouro="RUA X", bairro="CENTRO", data="01/03/2024", tier="strong",
          lat=-25.43, lon=-49.27, assunto="Coleta", subdivisao="Entulhos") -> dict:
    return {"ano_fonte": data[-4:], "data_criacao": data, "assunto": assunto,
            "subdivisao": subdivisao, "situacao": "Concluído", "logradouro": logradouro,
            "bairro": bairro, "regional": "R", "origem": "Telefone", "rank_key": "1",
            "consulta_geocodificacao": "q", "nivel_confianca": tier, "lat": lat, "lon": lon,
            "nominatim_nome_exibicao": "d", "osm_class": "", "osm_type": "secondary",
            "osm_id": "1", "motivo_geocodificacao": ""}


# --- haversine -------------------------------------------------------------------------------

def test_haversine_matches_known_distance():
    # 0.001 grau de latitude ~ 111,2 m
    d = neg.haversine_m(-25.430, -49.270, -25.431, -49.270)
    assert 110 < d < 113


def test_haversine_is_symmetric_and_zero_at_same_point():
    assert neg.haversine_m(-25.43, -49.27, -25.43, -49.27) == 0.0
    a = neg.haversine_m(-25.43, -49.27, -25.44, -49.28)
    b = neg.haversine_m(-25.44, -49.28, -25.43, -49.27)
    assert a == pytest.approx(b)


# --- colisão ---------------------------------------------------------------------------------

def test_collision_radius_is_thirty_meters():
    assert neg.COLLISION_RADIUS_M == 30.0


def test_point_within_thirty_meters_of_positive_is_dropped():
    pos = [(-25.4300, -49.2700)]
    rows, resumo = neg.build([_cand(lat=-25.43015, lon=-49.2700)], pos, None)  # ~17 m
    assert resumo["n_descartados_colisao_30m"] == 1
    assert rows == []


def test_point_beyond_thirty_meters_is_kept():
    pos = [(-25.4300, -49.2700)]
    rows, resumo = neg.build([_cand(lat=-25.4306, lon=-49.2700)], pos, None)  # ~67 m
    assert resumo["n_descartados_colisao_30m"] == 0
    assert len(rows) == 1


def test_collision_checked_against_every_positive_not_just_first():
    pos = [(-25.0, -49.0), (-24.0, -48.0), (-25.4300, -49.2700)]
    _, resumo = neg.build([_cand(lat=-25.43010, lon=-49.2700)], pos, None)
    assert resumo["n_descartados_colisao_30m"] == 1


# --- régua de confiança ----------------------------------------------------------------------

def test_failed_geocode_never_enters():
    rows, resumo = neg.build([_cand(tier="failed")], [], None)
    assert rows == []
    assert resumo["n_descartados_failed"] == 1


def test_strong_and_medium_both_accepted():
    rows, _ = neg.build([_cand(logradouro="A", tier="strong"),
                         _cand(logradouro="B", tier="medium")], [], None)
    assert len(rows) == 2


# --- união com geração anterior ---------------------------------------------------------------

def _existente(ponto_id="CUR_SIAC156_NEG_antigo1", logradouro="RUA X", bairro="CENTRO",
               data="01/03/2024") -> dict:
    return {"regiao": "Curitiba", "rotulo": "0", "ponto_id": ponto_id, "lat": "-25.43",
            "lon": "-49.27", "data_evento": data, "ano_fonte": data[-4:], "assunto": "Coleta",
            "subdivisao": "Entulhos", "logradouro": logradouro, "bairro": bairro,
            "nivel_confianca": "strong", "nominatim_nome_exibicao": "d", "nominatim_osm_id": "1",
            "tipo_fonte_negativo": neg.NEGATIVE_SOURCE_TYPE, "fenomeno_ocorrencia": "x",
            "papel_fonte_dataset": "siac156_curitiba_negative_v1", "uso_permitido": "a",
            "uso_proibido": "p"}


def test_existing_rows_are_carried_verbatim():
    e = _existente()
    rows, resumo = neg.build([], [], [e])
    assert rows[0] == e
    assert resumo["n_existentes_carregados"] == 1


def test_candidate_duplicating_an_existing_record_is_not_readded():
    e = _existente(logradouro="RUA X", bairro="CENTRO", data="01/03/2024")
    rows, resumo = neg.build([_cand(logradouro="RUA X", bairro="CENTRO", data="01/03/2024")], [], [e])
    assert len(rows) == 1
    assert rows[0]["ponto_id"] == "CUR_SIAC156_NEG_antigo1"   # id antigo preservado
    assert resumo["n_descartados_duplicata_de_existente"] == 1


def test_new_candidate_is_appended_to_existing():
    rows, resumo = neg.build([_cand(logradouro="RUA NOVA", lat=-25.50, lon=-49.30)], [], [_existente()])
    assert len(rows) == 2
    assert resumo["n_novos_aceitos"] == 1


def test_duplicate_candidates_within_same_batch_collapse():
    c = _cand(logradouro="RUA Y")
    rows, _ = neg.build([c, dict(c)], [], None)
    assert len(rows) == 1


# --- proveniência ------------------------------------------------------------------------------

def test_new_rows_carry_label_zero_and_source_type():
    rows, _ = neg.build([_cand()], [], None)
    assert rows[0]["rotulo"] == 0
    assert rows[0]["tipo_fonte_negativo"] == neg.NEGATIVE_SOURCE_TYPE
    assert rows[0]["uso_proibido"] == neg.PROHIBITED_USE


def test_event_date_is_the_records_own_date_never_synthetic():
    rows, _ = neg.build([_cand(data="17/09/2025")], [], None)
    assert rows[0]["data_evento"] == "17/09/2025"


def test_point_id_is_deterministic_and_namespaced():
    a = neg.negative_point_id("RUA X", "CENTRO", "01/03/2024")
    b = neg.negative_point_id("RUA X", "CENTRO", "01/03/2024")
    c = neg.negative_point_id("RUA X", "CENTRO", "02/03/2024")
    assert a == b and a != c
    assert a.startswith("CUR_SIAC156_NEG_")


# --- integridade dos artefatos publicados -----------------------------------------------------

REG = PKG / "registries"
NEG_V2 = REG / "v20n_dataset_negativos_curitiba_siac156_v2.csv"
FEAT_V2 = REG / "v20n_dataset_curitiba_variaveis_v2.csv"
FEAT_V1 = REG / "v20l_dataset_curitiba_variaveis_v1.csv"

pd = pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def negativos_v2():
    if not NEG_V2.exists():
        pytest.skip("v20n_dataset_negativos ainda nao gerado")
    return pd.read_csv(NEG_V2)


@pytest.fixture(scope="module")
def features_v2():
    if not FEAT_V2.exists():
        pytest.skip("v20n_dataset_curitiba_variaveis_v2 ainda nao gerado")
    return pd.read_csv(FEAT_V2)


def test_v2_is_superset_of_v1_negatives(negativos_v2):
    v1 = pd.read_csv(REG / "v20k3_dataset_negativos_curitiba_siac156_v1.csv")
    assert set(v1.ponto_id) <= set(negativos_v2.ponto_id)
    assert len(negativos_v2) > len(v1)


def test_v2_introduces_no_new_duplicate_record_key(negativos_v2):
    """As linhas do v1 entram verbatim, inclusive as 16 chaves ja repetidas la. Esta rodada
    nao pode ACRESCENTAR repeticao -- os pontos novos tem que ser todos de chave inedita."""
    v1 = pd.read_csv(REG / "v20k3_dataset_negativos_curitiba_siac156_v1.csv")
    novos = negativos_v2[~negativos_v2.ponto_id.isin(v1.ponto_id)]
    K = ["logradouro", "bairro", "data_evento"]
    assert not novos.duplicated(K).any()
    chaves_v1 = set(map(tuple, v1[K].astype(str).values))
    assert not set(map(tuple, novos[K].astype(str).values)) & chaves_v1


def test_v2_negatives_are_all_label_zero(negativos_v2):
    assert set(negativos_v2.rotulo.unique()) == {0}


def test_no_negative_collides_with_a_positive(negativos_v2):
    pos = pd.read_csv(REG / "v20k2_dataset_positivos_curitiba_siac156_v1.csv")
    p = list(zip(pos.lat.values, pos.lon.values))
    for r in negativos_v2.itertuples():
        assert not neg.collides_with_positive(r.lat, r.lon, p), r.ponto_id


def test_expanded_features_preserve_original_rows_bit_for_bit(features_v2):
    """Os 1357 pontos da rodada anterior nao podem ter mudado de valor bruto."""
    if not FEAT_V1.exists():
        pytest.skip("v20l ausente")
    v1 = pd.read_csv(FEAT_V1)
    key = ["ponto_id", "data_evento", "assunto", "subdivisao"]
    brutas = ["elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf",
              "rain_max_24h_chirps", "rain_decay_index_api_chirps", "mapbiomas_class_2023"]
    m = v1[key + brutas].merge(features_v2[key + brutas], on=key, suffixes=("_v1", "_v2"))
    assert len(m) >= len(v1)
    for c in brutas:
        iguais = (m[f"{c}_v1"] == m[f"{c}_v2"]) | (m[f"{c}_v1"].isna() & m[f"{c}_v2"].isna())
        assert iguais.all(), c


def test_expanded_rain_windows_are_complete(features_v2):
    """A rodada anterior deixou 3 janelas truncadas por cache sem span; nao pode voltar."""
    assert set(features_v2.rain_status.unique()) == {"OK"}
    assert set(features_v2.rain_n_days_found.unique()) == {14}


def test_expanded_dataset_passes_epv_floor_with_six_features(features_v2):
    seis = ["elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf",
            "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]
    u = features_v2[~features_v2.e_unidade_duplicada].dropna(subset=seis)
    assert int((u.rotulo == 0).sum()) / 6 >= 20.0
