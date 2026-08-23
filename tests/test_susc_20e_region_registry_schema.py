"""Formalizacao do gate #8 (revp_proxima_linhagem_programacao_pos_api.md
etapa 2): registro_regioes.py precisa recusar uma regiao 'available' sem
versao_modelo, e o JSON Schema versionado no repo precisa refletir o
modelo Pydantic que efetivamente valida o registro."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

API_SCRIPTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs_public" / "data" / "susc_20e_api_contrato_inferencia_recife" / "scripts"
)
sys.path.insert(0, str(API_SCRIPTS_DIR))

import registro_regioes as rr  # noqa: E402


def test_available_without_model_version_raises():
    with pytest.raises(ValidationError):
        rr.RegionInfo(
            name="fixture_region",
            bbox_wgs84=(0.0, 0.0, 1.0, 1.0),
            versao_modelo=None,
            region_maturity="available",
            status_note="",
        )


@pytest.mark.parametrize("maturity", ["limited_evidence", "insufficient"])
def test_non_available_without_model_version_is_allowed(maturity):
    info = rr.RegionInfo(
        name="fixture_region",
        bbox_wgs84=(0.0, 0.0, 1.0, 1.0),
        versao_modelo=None,
        region_maturity=maturity,
        status_note="",
    )
    assert info.versao_modelo is None


def test_live_registry_satisfies_the_gate():
    for name, info in rr.REGIONS.items():
        if info.region_maturity == "available":
            assert info.versao_modelo is not None, (
                f"regiao '{name}' esta available sem versao_modelo"
            )


def test_only_recife_is_available_today():
    maturity_by_region = {name: info.region_maturity for name, info in rr.REGIONS.items()}
    assert maturity_by_region["recife"] == "available"
    assert maturity_by_region["curitiba"] != "available"
    assert maturity_by_region["petropolis"] != "available"


def test_registry_as_dict_round_trips():
    payload = rr.registry_as_dict()
    assert payload["schema_version"] == rr.SCHEMA_VERSION
    assert payload["regions"]["recife"]["versao_modelo"] is not None
    assert payload["regions"]["petropolis"]["versao_modelo"] is None


def test_schema_file_in_repo_matches_current_model(tmp_path):
    committed_schema_path = API_SCRIPTS_DIR.parent / "schemas" / "esquema_registro_regioes_v1.json"
    assert committed_schema_path.exists(), "schema versionado nao foi gerado/commitado"
    committed_schema = json.loads(committed_schema_path.read_text(encoding="utf-8"))

    regenerated_path = rr.generate_schema_file(tmp_path / "esquema_registro_regioes_v1.json")
    regenerated_schema = json.loads(regenerated_path.read_text(encoding="utf-8"))

    assert committed_schema == regenerated_schema, (
        "schema commitado ficou fora de sincronia com o modelo Pydantic -- "
        "rode `python registro_regioes.py` e commit o resultado"
    )
    assert committed_schema["schema_version"] == rr.SCHEMA_VERSION
