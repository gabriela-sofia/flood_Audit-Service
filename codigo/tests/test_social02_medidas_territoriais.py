"""Medidas com unidades/denominadores; nenhum proxy vira pessoa ou risco."""
import sys
from pathlib import Path

import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/social"))
import social02_medidas_territoriais as s


def unidades():
    return [{"id": "A", "geometria": box(0, 0, 1, 1), "populacao": 100, "domicilios_ocupados": 50},
            {"id": "B", "geometria": box(1, 0, 2, 1), "populacao": 200, "domicilios_ocupados": 80}]


def test_exposicao_parcial_nao_multiplica_populacao_por_area():
    r = s.limites_exposicao(unidades(), box(0, 0, 1.5, 1), "recorte_de_estudo")
    assert r["limites_localizacao"]["populacao"] == {"minimo": 100, "maximo": 300, "unidades_sem_total": 0}
    assert r["estimativa_por_area"] is None


def test_contato_borda_nao_conta_como_setor_intersectado():
    r = s.limites_exposicao(unidades(), box(0, 0, 1, 1), "agua_observada")
    assert r["limites_localizacao"]["populacao"]["maximo"] == 100
    assert len(r["unidades_intersectadas"]) == 1


def test_populacao_ausente_nao_vira_zero_ou_limite_global():
    u = unidades(); u[1]["populacao"] = None
    r = s.limites_exposicao(u, box(0, 0, 2, 1), "cenario_suscetibilidade")
    assert r["limites_localizacao"]["populacao"]["maximo"] is None
    assert r["limites_localizacao"]["populacao"]["unidades_sem_total"] == 1


def test_suscetibilidade_nao_vira_agua_observada():
    r = s.limites_exposicao(unidades(), box(0, 0, 2, 1), "cenario_suscetibilidade")
    assert r["tipo_recorte"] == "cenario_suscetibilidade"
    with pytest.raises(ValueError): s.limites_exposicao(unidades(), box(0, 0, 2, 1), "risco_individual")


def test_unidades_duplicadas_recusadas():
    with pytest.raises(ValueError, match="duplicadas"):
        s.limites_exposicao(unidades()*2, box(0, 0, 2, 1), "agua_observada")


@pytest.mark.parametrize("v", ["...", ".", "X", "", None])
def test_numero_ausente_preservado(v):
    assert s.numero(v) is None


@pytest.fixture
def ledger():
    return [{"unidade_id": "A", "evento_canonico": "E1", "cena_id": "S1", "sensor": "optico", "regime": "urbano",
             "estrato_social": "estrato_predefinido", "evidencia_oportunidade": "qa_mask_hash",
             "area_visada_m2": 100, "area_detectavel_m2": 20, "area_referencia_admitida_m2": 10,
             "instrumento_admitido": "sim"}]


def test_cobertura_condicionada_a_oportunidade(ledger):
    r = s.cobertura_por_oportunidade(ledger)
    assert r["medidas"][0]["fracao_detectavel"] == .2
    assert r["medidas"][0]["cobertura_cientifica_condicional"] == .5
    assert not r["desigualdade_confirmada"]


def test_sem_oportunidade_nao_e_baixa_cobertura(ledger):
    ledger[0].update(area_detectavel_m2=0, area_referencia_admitida_m2=0)
    assert s.cobertura_por_oportunidade(ledger)["medidas"][0]["cobertura_cientifica_condicional"] is None


@pytest.mark.parametrize("alteracao", [{"area_detectavel_m2": None}, {"area_referencia_admitida_m2": 30},
                                       {"instrumento_admitido": "nao"}, {"evidencia_oportunidade": ""}])
def test_oportunidade_invalida_nao_produz_medida(ledger, alteracao):
    ledger[0].update(alteracao)
    with pytest.raises(ValueError): s.cobertura_por_oportunidade(ledger)


def test_cena_duplicada_nao_infla_observacoes(ledger):
    with pytest.raises(ValueError, match="duplicada"): s.cobertura_por_oportunidade(ledger*2)


def test_fontes_e_regimes_nao_sao_colapsados(ledger):
    ledger += [dict(ledger[0], cena_id="S2", sensor="sar", regime="fluvial")]
    assert len(s.cobertura_por_oportunidade(ledger)["medidas"]) == 2


def test_sem_ledger_mantem_not_testable():
    assert s.cobertura_por_oportunidade([])["estado"] == "not_testable"


def test_artefato_nao_e_sobrescrito(tmp_path):
    p = tmp_path / "r.json"
    s.gravar(p, {"medida": None})
    with pytest.raises(ValueError): s.gravar(p, {"medida": 0})
