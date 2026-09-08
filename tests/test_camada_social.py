from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "social"))

from camada_social import (  # noqa: E402
    DOMINIOS_SOCIAIS,
    VERSAO_CONTRATO_SOCIAL,
    avaliar_aoi_social,
    casos_experimentais,
    executar_experimentos,
    inventario_fontes,
    unidades_demo,
)


def test_inventario_social_e_multidimensional_e_nao_fisico():
    fontes = inventario_fontes()
    assert len(fontes) >= 4
    dominios = ";".join(f["dominios"] for f in fontes)
    for esperado in ("condicao_socioeconomica", "moradia_infraestrutura",
                     "lacuna_institucional"):
        assert esperado in dominios
    bruto = json.dumps(fontes, ensure_ascii=False).lower()
    assert "causa de enchente" not in " ".join(
        f["uso_admitido"].lower() for f in fontes)
    assert "causa de enchente" in bruto
    assert "rotulo do modelo fisico" in bruto


def test_camadas_sociais_nao_alteram_escore_fisico():
    unidades = unidades_demo()
    fisica = {"status": "ok", "escore": 0.82, "maturidade": "mvp_local"}
    alta = avaliar_aoi_social(
        {"lat_min": -8.075, "lon_min": -34.925,
         "lat_max": -8.050, "lon_max": -34.900},
        unidades,
        fisica,
    )
    baixa = avaliar_aoi_social(
        {"lat_min": -8.115, "lon_min": -34.885,
         "lat_max": -8.095, "lon_max": -34.860},
        unidades,
        fisica,
    )
    assert alta["eixos"]["suscetibilidade_fisica"]["escore_recebido"] == 0.82
    assert baixa["eixos"]["suscetibilidade_fisica"]["escore_recebido"] == 0.82
    assert alta["prioridade_publica"]["classe"] != baixa["prioridade_publica"]["classe"]
    assert alta["invariantes"]["nao_altera_escore_fisico"] is True


def test_payload_fisico_de_entrada_nao_e_mutado():
    fisica = {"status": "ok", "escore": 0.51, "maturidade": "mvp_local"}
    antes = dict(fisica)
    avaliar_aoi_social(
        {"lat_min": -8.075, "lon_min": -34.925,
         "lat_max": -8.050, "lon_max": -34.900},
        unidades_demo(),
        fisica,
    )
    assert fisica == antes


def test_fisica_recusada_em_area_vulneravel_nao_vira_baixo_risco():
    r = avaliar_aoi_social(
        {"lat_min": -8.075, "lon_min": -34.925,
         "lat_max": -8.050, "lon_max": -34.900},
        unidades_demo(),
        {"status": "insufficient_data", "escore": None,
         "gate_que_nao_fechou": "dominio_coberto"},
    )
    fisica = r["eixos"]["suscetibilidade_fisica"]
    assert fisica["classe_relativa"] == "nao_identificavel"
    assert r["prioridade_publica"]["classe"] == "prioridade_de_vistoria_e_coleta"
    assert r["eixos"]["lacuna_de_conhecimento_publico"]["nao_e"] == "baixo risco"


def test_ausencia_de_cobertura_social_nao_e_baixa_vulnerabilidade():
    r = avaliar_aoi_social(
        {"lat_min": -9.500, "lon_min": -36.000,
         "lat_max": -9.450, "lon_max": -35.950},
        unidades_demo(),
        {"status": "ok", "escore": 0.82, "maturidade": "mvp_local"},
    )
    social = r["eixos"]["vulnerabilidade_exposicao"]
    lacuna = r["eixos"]["lacuna_de_conhecimento_publico"]
    assert r["status"] == "insufficient_social_data"
    assert social["classe_relativa"] == "nao_identificavel"
    # A fisica desta AOI e alta (0,82). A prioridade tem de carregar as DUAS
    # coisas -- ha suscetibilidade e nao se sabe quem esta exposto. Enquanto
    # esta assercao dizia `prioridade_de_dado`, ela documentava o colapso:
    # a mesma classe saia para uma AOI seca sem dado, e a area que combina
    # suscetibilidade alta com territorio nao inventariado sumia da fila.
    assert r["prioridade_publica"]["classe"] == "prioridade_tecnica_e_de_dado"
    assert "cobertura_geografica_social_insuficiente" in lacuna["lacunas"]
    assert lacuna["nivel_ausencia_social"] == "fonte_existe_e_nao_alcanca_a_aoi"


def test_a_mesma_ausencia_social_nao_produz_a_mesma_prioridade():
    """A hierarquia da ausencia so existe se ela mudar alguma decisao.

    Duas AOIs igualmente sem cobertura social, uma com suscetibilidade fisica
    alta e outra baixa: se sairem iguais, a ausencia apagou a fisica.
    """
    fora = {"lat_min": -9.500, "lon_min": -36.000,
            "lat_max": -9.450, "lon_max": -35.950}
    alta = avaliar_aoi_social(fora, unidades_demo(),
                              {"status": "ok", "escore": 0.82,
                               "maturidade": "mvp_local"})
    baixa = avaliar_aoi_social(fora, unidades_demo(),
                               {"status": "ok", "escore": 0.10,
                                "maturidade": "mvp_local"})

    assert alta["eixos"]["vulnerabilidade_exposicao"]["classe_relativa"] == \
        baixa["eixos"]["vulnerabilidade_exposicao"]["classe_relativa"]
    assert alta["prioridade_publica"]["classe"] != \
        baixa["prioridade_publica"]["classe"]


def test_ausencia_por_falta_de_fonte_e_distinta_de_fonte_que_nao_alcanca():
    """O achado que a tese precisa nomear.

    Um territorio sem NENHUMA fonte social carregada nao esta na mesma
    situacao de um que tem fonte na regiao e fica fora do recorte dela. A
    primeira ausencia e do inventario publico, e ela mesma e um resultado.
    """
    aoi = {"lat_min": -9.500, "lon_min": -36.000,
           "lat_max": -9.450, "lon_max": -35.950}
    fisica = {"status": "ok", "escore": 0.82, "maturidade": "mvp_local"}

    sem_fonte = avaliar_aoi_social(aoi, [], fisica)
    fonte_longe = avaliar_aoi_social(aoi, unidades_demo(), fisica)

    assert sem_fonte["eixos"]["lacuna_de_conhecimento_publico"][
        "nivel_ausencia_social"] == "fonte_inexistente_para_o_territorio"
    assert fonte_longe["eixos"]["lacuna_de_conhecimento_publico"][
        "nivel_ausencia_social"] == "fonte_existe_e_nao_alcanca_a_aoi"
    # E nenhuma das duas pode virar leitura de seguranca.
    for r in (sem_fonte, fonte_longe):
        assert r["eixos"]["lacuna_de_conhecimento_publico"]["nao_e"] == \
            "baixo risco"
        assert r["invariantes"][
            "ausencia_social_nao_e_baixa_vulnerabilidade"] is True


def test_experimento_nao_colapso_passa():
    resultado = executar_experimentos()
    assert resultado["contrato"] == VERSAO_CONTRATO_SOCIAL
    assert resultado["experimento_nao_colapso"]["status"] == "PASS"
    assert resultado["experimento_nao_colapso"][
        "mesmo_escore_fisico_com_prioridades_diferentes"] is True
    assert resultado["experimento_nao_colapso"][
        "fisica_recusada_nao_vira_baixo_risco"] is True


def test_todos_os_casos_carregam_invariantes_metodologicos():
    resultado = executar_experimentos()
    assert {c["caso"] for c in resultado["casos"]} == {
        c["caso"] for c in casos_experimentais()
    }
    for caso in resultado["casos"]:
        inv = caso["invariantes"]
        assert inv["nao_treina_modelo"] is True
        assert inv["nao_cria_rotulo"] is True
        assert inv["vulnerabilidade_nao_e_causa_hidrologica"] is True
        assert set(caso["eixos"]["vulnerabilidade_exposicao"]
                   ["indicadores_agregados"]) <= set(DOMINIOS_SOCIAIS)


def test_saida_gravada_tem_formato_esperado(tmp_path):
    from camada_social import gravar_resultados

    resultado = executar_experimentos()
    gravar_resultados(resultado, tmp_path)
    esperados = {
        "experimentos_sociais_v0.json",
        "inventario_fontes_sociais_v0.csv",
        "casos_sociais_v0.csv",
        "matriz_prioridade_social_v0.csv",
        "relatorio_camadas_sociais_v0.md",
        "commands.txt",
    }
    assert esperados <= {p.name for p in tmp_path.iterdir()}
    relatorio = (tmp_path / "relatorio_camadas_sociais_v0.md").read_text(
        encoding="utf-8")
    assert "nao valida vulnerabilidade social local" in relatorio


def test_cli_aoi_com_unidades_json(tmp_path):
    import subprocess

    unidades = {
        "unidades_sociais": [
            {
                "unidade_id": "u1",
                "nome": "unidade teste",
                "geometria": {
                    "lat_min": -8.1, "lon_min": -34.9,
                    "lat_max": -8.0, "lon_max": -34.8,
                },
                "indicadores": {
                    "exposicao_populacional": 0.9,
                    "condicao_socioeconomica": 0.9,
                    "moradia_infraestrutura": 0.9,
                },
                "fonte": "teste",
            }
        ]
    }
    aoi = {"lat_min": -8.09, "lon_min": -34.89,
           "lat_max": -8.01, "lon_max": -34.81}
    fisica = {"status": "ok", "escore": 0.82, "maturidade": "mvp_local"}
    unidades_p = tmp_path / "unidades.json"
    aoi_p = tmp_path / "aoi.json"
    fisica_p = tmp_path / "fisica.json"
    unidades_p.write_text(json.dumps(unidades), encoding="utf-8")
    aoi_p.write_text(json.dumps(aoi), encoding="utf-8")
    fisica_p.write_text(json.dumps(fisica), encoding="utf-8")

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "social" / "camada_social.py"),
         "--unidades-json", str(unidades_p),
         "--aoi-json", str(aoi_p),
         "--fisica-json", str(fisica_p),
         "--output-dir", str(tmp_path / "out")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "prioridade=prioridade_publica_forte" in r.stdout
    resposta = json.loads((tmp_path / "out" / "resposta_social_aoi.json")
                          .read_text(encoding="utf-8"))
    assert resposta["invariantes"]["nao_altera_escore_fisico"] is True
