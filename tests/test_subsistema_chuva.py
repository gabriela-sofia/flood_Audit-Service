"""Regressoes do contrato, provider, cache e cenarios de chuva."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "servico"))
sys.path.insert(0, str(ROOT / "scripts" / "treino"))

from chuva.contrato_chuva import (  # noqa: E402
    DECAY_K, LOOKBACK_DAYS, METHOD_VERSION, SCENARIOS, contract,
    derive_rain_features, parse_event_dates, quality_control,
    requested_window, scenario_evidence, scenario_values,
)
from chuva.provedores import OpenMeteoEra5LandProvider, RainRequest  # noqa: E402


def test_decay_index_da_peso_maximo_ao_valor_mais_recente():
    got = derive_rain_features([1.0, 2.0, 3.0], expected_days=None)
    expected = 1.0 * DECAY_K**2 + 2.0 * DECAY_K + 3.0
    assert got["rain_decay_index"] == pytest.approx(expected, abs=1e-4)

    antigo = derive_rain_features([10.0, 0.0, 0.0], expected_days=None)
    recente = derive_rain_features([0.0, 0.0, 10.0], expected_days=None)
    assert recente["rain_decay_index"] > antigo["rain_decay_index"]


def test_janela_exclui_o_dia_do_evento_e_tem_14_dias():
    inicio, fim = requested_window("2026-08-30")
    assert (inicio, fim) == ("2026-08-16", "2026-08-29")
    assert LOOKBACK_DAYS == 14


def test_parser_misto_nao_coage_recife_para_nat():
    parsed = parse_event_dates(pd.Series([
        "2025-01-23", "2022-05-28 00:00:00", None,
    ]))
    assert parsed.notna().tolist() == [True, True, False]
    assert parsed.dt.strftime("%Y-%m-%d").iloc[:2].tolist() == [
        "2025-01-23", "2022-05-28"]


def test_parser_misto_normaliza_fusos_sem_falhar():
    parsed = parse_event_dates(pd.Series([
        "2025-01-23", "2025-01-23T00:00:00Z",
        "2025-01-23T03:00:00-03:00",
    ]))
    assert parsed.notna().all()
    assert str(parsed.dt.tz) == "UTC"
    assert parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist() == [
        "2025-01-23T00:00:00Z",
        "2025-01-23T00:00:00Z",
        "2025-01-23T06:00:00Z",
    ]


@pytest.mark.parametrize("series", [
    [], [0.0] * 13, [0.0] * 13 + [-1.0], [0.0] * 13 + [float("nan")],
])
def test_derivacao_falha_fechado_para_serie_invalida(series):
    with pytest.raises(ValueError):
        derive_rain_features(series)


def test_quality_control_detecta_duplicata_missing_e_periodo():
    qc = quality_control(["2026-01-01", "2026-01-01"], [1.0, None],
                         expected_start="2025-12-31", expected_end="2026-01-02")
    assert not qc["ok"]
    assert {"duplicate_dates", "missing_precipitation",
            "unexpected_start_date", "unexpected_end_date"} <= set(qc["issues"])


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def get(self, *_args, **_kwargs):
        self.calls += 1
        return _Response(self.payload)


def test_provider_separa_aquisicao_derivacao_e_replay(tmp_path):
    dates = pd.date_range("2026-01-01", periods=14).strftime("%Y-%m-%d").tolist()
    payload = {"daily": {"time": dates,
                         "precipitation_sum": list(range(14))}}
    session = _Session(payload)
    provider = OpenMeteoEra5LandProvider(tmp_path, session=session,
                                         backoff_seconds=0)
    request = RainRequest(-8.05, -34.9, dates[0], dates[-1])
    acquired = provider.acquire(request, online=True)
    derived = provider.derive(acquired)
    assert session.calls == 1
    assert derived["rain_max_24h"] == 13.0
    assert derived["provider"] == "open_meteo"
    assert derived["product"] == "era5_land"
    assert derived["quality_control"]["ok"]

    replay = OpenMeteoEra5LandProvider(tmp_path).acquire(request, online=False)
    assert replay.cache_mode == "replay"
    assert replay.raw_sha256 == acquired.raw_sha256
    record = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert record["raw_payload"] == payload
    assert record["method_version"] == METHOD_VERSION


def test_cache_miss_offline_nao_toca_a_rede(tmp_path):
    provider = OpenMeteoEra5LandProvider(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not access network"):
        provider.acquire(RainRequest(-8, -35, "2026-01-01", "2026-01-14"),
                         online=False)


def test_cenario_e_unico_e_extremo_declara_extrapolacao():
    import resolver_camadas
    import svc03_grade_suscetibilidade as grade

    expected = {name: scenario_values(name) for name in SCENARIOS}
    assert resolver_camadas.CENARIOS_CHUVA == expected
    assert grade.CENARIOS_CHUVA == expected
    ev = scenario_evidence("maximo_observado", {
        "rain_max_24h": [3.3, 37.72],
        "rain_decay_index": [5.9962, 68.8708],
    })
    assert ev["extrapolation"] is True
    assert set(ev["extrapolation_variables"]) == {
        "rain_max_24h", "rain_decay_index"}
    assert ev["by_variable"]["rain_max_24h"]["distance_to_domain"] == 62.48


def test_contrato_declara_forcing_missing_e_politica_de_periodo():
    c = contract()
    assert c["semantic_role"] == "EVENT_FORCING"
    assert c["missing_policy"] == "fail_closed"
    assert "scenario-based" in c["period_policy"]
    assert c["variables"]["rain_decay_index"]["decay_parameter"] == DECAY_K


def test_auditoria_de_fonte_recusa_valor_sem_procedencia():
    from aud_chuva01_fontes_incompativeis import auditar

    d = pd.DataFrame({
        "classe": [0, 1], "fonte_chuva": ["ausente", "open_meteo_era5_land"],
        "rain_max_24h": [10.0, 20.0], "rain_decay_index": [15.0, 25.0],
        "data_evento": ["2026-01-01", "2026-01-01 00:00:00"],
    })
    result = auditar("synthetic", d)
    assert result["veredito"] == "VALOR_SEM_PROCEDENCIA"
    assert result["rain_values_without_provenance"] == 1


def test_uk_nao_herda_data_de_saida_derivada_entre_execucoes(
        tmp_path, monkeypatch):
    """A visao derivada nao pode virar fonte estrutural na segunda passagem."""
    import ds01_montar_multirregiao as ds01

    base_dir = tmp_path / "ft-uk-03-amostra"
    rain_dir = tmp_path / "ft-uk-04-chuva"
    base_dir.mkdir()
    rain_dir.mkdir()
    base = pd.DataFrame({
        "ponto_id": ["UK_POS", "UK_NEG"], "label": [1, 0],
        "tema": ["x", "x"], "grupo_cv": ["EV_1", "NEG_1"],
        "fonte_label": ["uk", "uk"],
        "lat": [51.0, 51.1], "lon": [-1.0, -1.1],
        "start_date": ["2025-01-01T00:00:00Z", None],
        "elevation_m": [10.0, 11.0], "slope_deg": [1.0, 2.0],
        "hand_m": [0.5, 0.6], "twi_dinf": [4.0, 5.0],
        "worldcover": [40, 40],
    })
    base.to_csv(base_dir / "amostra_candidata_uk_v1.csv", index=False)
    derived = base.copy()
    # Simula a saida historica que preenchia indevidamente a data negativa.
    derived.loc[1, "start_date"] = "2025-01-01T00:00:00Z"
    derived["rain_max_24h"] = [12.0, 12.0]
    derived["rain_decay_index"] = [20.0, 20.0]
    derived["fonte_chuva"] = "open_meteo_era5_land"
    derived.to_csv(rain_dir / "amostra_candidata_uk_v2_com_chuva.csv",
                   index=False)
    monkeypatch.setattr(ds01, "RUNS", tmp_path)

    first = ds01.carregar_uk()
    assert pd.isna(first.loc[first.ponto_id == "UK_NEG", "data_evento"]).all()

    # Segunda passagem perde a chuva negativa, mas nao muda a estrutura.
    derived.loc[1, ["rain_max_24h", "rain_decay_index"]] = float("nan")
    derived.loc[1, "fonte_chuva"] = "ausente"
    derived.to_csv(rain_dir / "amostra_candidata_uk_v2_com_chuva.csv",
                   index=False)
    second = ds01.carregar_uk()
    pd.testing.assert_series_equal(first["data_evento"], second["data_evento"])
    assert second.loc[second.ponto_id == "UK_NEG", "fonte_chuva"].item() == "ausente"
