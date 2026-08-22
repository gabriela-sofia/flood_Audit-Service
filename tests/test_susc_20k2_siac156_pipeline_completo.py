"""Testes reais (sintéticos, não network) pro pipeline completo de mineração multi-ano do SIAC
156 (Curitiba) -- réplica do método SEDEC/Recife. Sem rede, sem raster; usa os módulos já
testados em test_susc_20k_siac156_flood_candidates.py mais os dois módulos novos desta rodada:
`qa_dedup_siac156.py` e `build_final_dataset.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import qa_dedup_siac156 as qa_mod  # noqa: E402
import build_final_dataset as build_mod  # noqa: E402


# --- qa_dedup_siac156 -----------------------------------------------------------------------

def test_normalize_matches_extract_module_behavior():
    assert qa_mod.normalize("Rebouças") == "REBOUCAS"
    assert qa_mod.normalize(None) == ""


@pytest.mark.parametrize(
    "logradouro,expected",
    [
        ("", True),
        ("Rua", True),
        ("R", True),
        ("123", True),
        ("Rua Nunes Machado", False),
        ("Avenida Sete de Setembro", False),
    ],
)
def test_is_generic_or_empty_street(logradouro, expected):
    assert qa_mod.is_generic_or_empty_street(logradouro) == expected


def test_qa_decision_reject_when_no_address_and_no_neighborhood():
    decision, issues = qa_mod.qa_decision(bairro_reconhecido=False, rua_generica=True, tem_logradouro=False, tem_bairro=False)
    assert decision == "reject"
    assert "sem_logradouro_e_sem_bairro" in issues


def test_qa_decision_ready_when_everything_clean():
    decision, issues = qa_mod.qa_decision(bairro_reconhecido=True, rua_generica=False, tem_logradouro=True, tem_bairro=True)
    assert decision == "ready_for_geocoding"
    assert issues == []


def test_qa_decision_needs_cleanup_when_bairro_not_recognized():
    decision, issues = qa_mod.qa_decision(bairro_reconhecido=False, rua_generica=False, tem_logradouro=True, tem_bairro=True)
    assert decision == "needs_manual_cleanup"
    assert "bairro_fora_da_lista_de_referencia" in issues


def test_run_qa_flags_duplicate_group_count():
    rows = [
        {"Logradouro": "Rua A", "Bairro": "XAXIM", "DataCriacao": "03/02/2025", "Assunto": "Drenagem", "Subdivisao": "Alagamentos", "source_year": "2025", "Situacao": "Aberto", "Regional": "", "Origem": "Mobile"},
        {"Logradouro": "rua a", "Bairro": "xaxim", "DataCriacao": "03/02/2025", "Assunto": "Drenagem", "Subdivisao": "Alagamentos", "source_year": "2025", "Situacao": "Aberto", "Regional": "", "Origem": "Mobile"},  # mesma chave normalizada
        {"Logradouro": "Rua B", "Bairro": "XAXIM", "DataCriacao": "03/02/2025", "Assunto": "Drenagem", "Subdivisao": "Alagamentos", "source_year": "2025", "Situacao": "Aberto", "Regional": "", "Origem": "Mobile"},
    ]
    out = qa_mod.run_qa(rows, bairros_referencia={"XAXIM"})
    assert out[0]["duplicate_group_count"] == 2
    assert out[1]["duplicate_group_count"] == 2
    assert out[2]["duplicate_group_count"] == 1


def test_run_qa_produces_stable_qa_record_id():
    rows = [{"Logradouro": "Rua A", "Bairro": "XAXIM", "DataCriacao": "03/02/2025", "Assunto": "Drenagem", "Subdivisao": "Alagamentos", "source_year": "2025", "Situacao": "Aberto", "Regional": "", "Origem": "Mobile"}]
    out1 = qa_mod.run_qa(rows, bairros_referencia={"XAXIM"})
    out2 = qa_mod.run_qa(rows, bairros_referencia={"XAXIM"})
    assert out1[0]["qa_record_id"] == out2[0]["qa_record_id"]


# --- build_final_dataset --------------------------------------------------------------------

def test_build_dataset_excludes_failed_tier():
    rows = [
        {"qa_record_id": "a1", "confidence_tier": "strong", "lat": "-25.4", "lon": "-49.2", "data_criacao": "03/02/2025", "source_year": "2025", "assunto": "Drenagem", "subdivisao": "Alagamentos", "logradouro": "Rua A", "bairro": "XAXIM", "nominatim_display_name": "Rua A", "osm_id": "1", "duplicate_group_count": "1"},
        {"qa_record_id": "a2", "confidence_tier": "failed", "lat": "", "lon": "", "data_criacao": "03/02/2025", "source_year": "2025", "assunto": "Drenagem", "subdivisao": "Alagamentos", "logradouro": "Rua B", "bairro": "XAXIM", "nominatim_display_name": "", "osm_id": "", "duplicate_group_count": "1"},
        {"qa_record_id": "a3", "confidence_tier": "medium", "lat": "-25.5", "lon": "-49.3", "data_criacao": "03/02/2025", "source_year": "2025", "assunto": "Drenagem", "subdivisao": "Alagamentos", "logradouro": "Rua C", "bairro": "XAXIM", "nominatim_display_name": "Xaxim", "osm_id": "2", "duplicate_group_count": "1"},
    ]
    out = build_mod.build_dataset(rows)
    assert len(out) == 2
    tiers = {r["confidence_tier"] for r in out}
    assert tiers == {"strong", "medium"}
    assert all(r["label"] == 1 and r["region"] == "Curitiba" for r in out)


def test_build_dataset_point_id_is_traceable_to_qa_record():
    rows = [{"qa_record_id": "abc123", "confidence_tier": "strong", "lat": "-25.4", "lon": "-49.2", "data_criacao": "03/02/2025", "source_year": "2025", "assunto": "Drenagem", "subdivisao": "Alagamentos", "logradouro": "Rua A", "bairro": "XAXIM", "nominatim_display_name": "Rua A", "osm_id": "1", "duplicate_group_count": "1"}]
    out = build_mod.build_dataset(rows)
    assert out[0]["point_id"] == "CUR_SIAC156_abc123"
    assert "landslide excluded" in out[0]["occurrence_phenomenon"]
