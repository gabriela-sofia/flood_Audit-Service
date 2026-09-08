from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "social"))

from ibge_curitiba import (  # noqa: E402
    construir_unidades_sociais,
    flag_fcu,
    parse_num,
    percentile_scores,
)

ARTEFATO_REAL = (ROOT / "local_runs" / "camada-social-territorial"
                 / "ibge-curitiba" / "manifesto_ibge_curitiba_v1.json")


def test_parse_num_entende_formato_ibge():
    assert parse_num("1,25") == 1.25
    assert parse_num("2453.03") == 2453.03
    assert parse_num(".") is None
    assert parse_num("") is None


def test_percentil_invertido_da_renda_da_vulnerabilidade_maior_para_menor_renda():
    scores = percentile_scores({"rico": 5000.0, "pobre": 1000.0}, invert=True)
    assert scores["pobre"] == 1.0
    assert scores["rico"] == 0.0


def test_flag_fcu_so_quando_codigo_existe():
    assert flag_fcu({"CD_FCU": "41069020001"}) is True
    assert flag_fcu({"CD_FCU": "."}) is False
    assert flag_fcu({"CD_FCU": None}) is False


def test_construir_unidades_preserva_dominios_e_nao_cria_rotulo():
    malha = [
        {
            "CD_SETOR": "410690205010001", "NM_MUN": "Curitiba",
            "NM_BAIRRO": "A", "AREA_KM2": 1.0, "CD_FCU": ".",
            "bbox": {"lat_min": -25.1, "lon_min": -49.1,
                     "lat_max": -25.0, "lon_max": -49.0},
        },
        {
            "CD_SETOR": "410690205010002", "NM_MUN": "Curitiba",
            "NM_BAIRRO": "B", "AREA_KM2": 1.0, "CD_FCU": "41069020001",
            "bbox": {"lat_min": -25.2, "lon_min": -49.2,
                     "lat_max": -25.1, "lon_max": -49.1},
        },
    ]
    basico = [
        {"CD_SETOR": "410690205010001", "v0001": "100", "v0005": "2,0",
         "v0006": "0,01"},
        {"CD_SETOR": "410690205010002", "v0001": "200", "v0005": "4,0",
         "v0006": "0,04"},
    ]
    renda = [
        {"CD_SETOR": "410690205010001", "V06006": "5000"},
        {"CD_SETOR": "410690205010002", "V06006": "1000"},
    ]
    unidades, tabela, resumo = construir_unidades_sociais(malha, basico, renda)
    assert len(unidades) == 2
    assert resumo["setores_em_fcu"] == 1
    mais_vulneravel = next(u for u in unidades
                           if u.unidade_id.endswith("410690205010002"))
    assert mais_vulneravel.indicadores["condicao_socioeconomica"] == 1.0
    assert mais_vulneravel.indicadores["moradia_infraestrutura"] == 1.0
    assert "classe" not in mais_vulneravel.indicadores
    assert any(r["em_tipologia_territorial_ibge"] is True for r in tabela)
    assert any(r["tipologia_territorial_fonte"] for r in tabela)


def test_artefato_real_curitiba_tem_proveniencia_e_cobertura():
    if not ARTEFATO_REAL.exists():
        import pytest
        pytest.skip("piloto IBGE Curitiba ausente; rode scripts/social/ibge_curitiba.py")
    import json

    d = json.loads(ARTEFATO_REAL.read_text(encoding="utf-8"))
    resumo = d["resumo"]
    assert resumo["setores_com_unidade_social"] >= 3000
    assert resumo["setores_em_fcu"] > 0
    assert resumo["cobertura_por_dominio"]["exposicao_populacional"] == (
        resumo["setores_com_unidade_social"])
    assert resumo["cobertura_por_dominio"]["condicao_socioeconomica"] > 3000
    for fonte in d["manifesto_fontes"]:
        assert fonte["existe"] is True
        assert fonte["sha256"] and len(fonte["sha256"]) == 64


def test_piloto_real_curitiba_prova_nao_colapso():
    if not ARTEFATO_REAL.exists():
        import pytest
        pytest.skip("piloto IBGE Curitiba ausente; rode scripts/social/ibge_curitiba.py")
    import json

    d = json.loads(ARTEFATO_REAL.read_text(encoding="utf-8"))
    alto, baixo = d["piloto_real"]
    assert alto["eixos"]["suscetibilidade_fisica"]["escore_recebido"] == (
        baixo["eixos"]["suscetibilidade_fisica"]["escore_recebido"])
    assert alto["eixos"]["vulnerabilidade_exposicao"]["classe_relativa"] == "alta"
    assert baixo["eixos"]["vulnerabilidade_exposicao"]["classe_relativa"] == "baixa"
    assert alto["prioridade_publica"]["classe"] == "prioridade_publica_forte"
    assert baixo["prioridade_publica"]["classe"] == "prioridade_tecnica"
    assert alto["invariantes"]["nao_altera_escore_fisico"] is True
