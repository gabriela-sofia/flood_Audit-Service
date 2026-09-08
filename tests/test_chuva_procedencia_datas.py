"""A auditoria de procedencia so vale se ela souber acusar o caso que existe.

O achado que motiva estes testes e que 93% da base tem data carimbada em lote
-- cena de satelite ou ativacao de emergencia --, e que a coluna que declarava
isso (`precisao_data`) e descartada antes do modelo. Um teste que apenas
confirmasse os numeros do relatorio nao protegeria nada: o que precisa ficar
travado e a REGRA que os produz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from chuva.procedencia_das_datas import (  # noqa: E402
    MM_MINIMO_PARA_LEITURA_PLUVIAL, PONTOS_POR_DATA_LOTE,
    PONTOS_POR_DATA_OCORRENCIA, agregado_da_base, contraste_positivo_negativo,
    cobertura_da_precisao, granularidade_da_data,
    precisao_versus_granularidade,
)


def registro(procedencia, *, n, classes, mediana, granularidade="por_ocorrencia"):
    return {"procedencia": procedencia, "n": n, "classes": classes,
            "mediana_mm_no_dia": mediana, "granularidade": granularidade}


class TestGranularidadeDaData:
    def test_uma_data_por_ocorrencia_e_granularidade_fina(self):
        assert granularidade_da_data(132, 97)["granularidade"] == "por_ocorrencia"

    def test_milhares_de_pontos_numa_data_e_carimbo_de_lote(self):
        # sen1floods11: 30.431 pontos em 11 datas de cena Sentinel-1
        r = granularidade_da_data(30431, 11)
        assert r["granularidade"] == "carimbo_de_lote"
        assert r["pontos_por_data"] == pytest.approx(2766.5, abs=0.1)

    def test_faixa_entre_os_cortes_nao_e_classificada_nos_extremos(self):
        # o Reino Unido cai aqui: 34 pontos por data, nem lote nem ocorrencia
        assert granularidade_da_data(3738, 110)["granularidade"] == "intermediaria"

    def test_sem_data_nenhuma_nao_inventa_classificacao(self):
        r = granularidade_da_data(500, 0)
        assert r["granularidade"] == "indeterminada"
        assert r["pontos_por_data"] is None

    def test_os_cortes_nao_se_cruzam(self):
        assert PONTOS_POR_DATA_OCORRENCIA < PONTOS_POR_DATA_LOTE


class TestAgregadoDaBase:
    def test_soma_apenas_o_que_foi_classificado_como_lote(self):
        r = agregado_da_base({
            "a": {"p1": registro("p1", n=90, classes=[1], mediana=5.0,
                                 granularidade="carimbo_de_lote")},
            "b": {"p2": registro("p2", n=10, classes=[1], mediana=5.0)},
        })
        assert r["pontos"] == 100
        assert r["pontos_com_data_de_lote"] == 90
        assert r["pct_com_data_de_lote"] == 90.0
        assert r["procedencias_de_lote"] == ["p1"]

    def test_base_toda_por_ocorrencia_nao_acusa_lote(self):
        r = agregado_da_base({"a": {"p": registro("p", n=10, classes=[1],
                                                  mediana=5.0)}})
        assert r["pontos_com_data_de_lote"] == 0
        assert r["procedencias_de_lote"] == []

    def test_base_vazia_nao_divide_por_zero(self):
        assert agregado_da_base({})["pct_com_data_de_lote"] is None


class TestContraste:
    def test_acusa_positivos_e_negativos_no_mesmo_dia_de_chuva(self):
        # Recife: 2,30 mm nos positivos contra 2,25 nos negativos
        r = contraste_positivo_negativo({"avaliado": True, "por_fonte": {
            "recife": {
                "v4": registro("v4", n=132, classes=[1], mediana=2.30),
                "neg": registro("neg", n=124, classes=[0], mediana=2.25),
            }}})
        assert r["por_fonte"]["recife"]["positivos_indistinguiveis_dos_negativos"]

    def test_nao_acusa_onde_a_chuva_separa_as_classes(self):
        # Curitiba: 6,80 contra 1,10, razao 6,2
        r = contraste_positivo_negativo({"avaliado": True, "por_fonte": {
            "curitiba": {
                "sim": registro("sim", n=1238, classes=[1], mediana=6.80),
                "nao": registro("nao", n=442, classes=[0], mediana=1.10),
            }}})
        c = r["por_fonte"]["curitiba"]
        assert not c["positivos_indistinguiveis_dos_negativos"]
        assert c["razao"] == pytest.approx(6.18, abs=0.01)

    def test_compara_a_procedencia_dominante_e_nao_a_media(self):
        # A media ponderada de Recife daria 8,49 mm e esconderia o achado: os
        # oito pontos do decreto a 112 mm puxam os 132 pontos a 2,30.
        r = contraste_positivo_negativo({"avaliado": True, "por_fonte": {
            "recife": {
                "decreto": registro("decreto", n=8, classes=[1], mediana=112.0),
                "v4": registro("v4", n=132, classes=[1], mediana=2.30),
                "neg": registro("neg", n=124, classes=[0], mediana=2.25),
            }}})
        c = r["por_fonte"]["recife"]
        assert c["procedencia_dominante_positiva"] == "v4"
        assert c["positivos_indistinguiveis_dos_negativos"]

    def test_fonte_sem_negativo_nao_entra_no_contraste(self):
        r = contraste_positivo_negativo({"avaliado": True, "por_fonte": {
            "uk": {"ea": registro("ea", n=3738, classes=[1], mediana=21.4)}}})
        assert "uk" not in r["por_fonte"]

    def test_procedencia_com_as_duas_classes_nao_e_lida_como_pura(self):
        # cems e ufo tem classes [0, 1] na mesma procedencia: nao ha
        # procedencia positiva nem negativa para contrastar.
        r = contraste_positivo_negativo({"avaliado": True, "por_fonte": {
            "cems": {"ems": registro("ems", n=23915, classes=[0, 1],
                                     mediana=10.7)}}})
        assert "cems" not in r["por_fonte"]

    def test_sem_gatilho_o_contraste_nao_e_avaliado(self):
        assert contraste_positivo_negativo({"avaliado": False})["avaliado"] is False


class TestCoberturaDaPrecisao:
    def test_acusa_ausencia_da_coluna_de_procedencia_da_data(self):
        # A regressao que este teste guarda: se a coluna voltar a sumir do
        # esquema, `data_evento` volta a ser valor sem procedencia.
        r = cobertura_da_precisao(pd.DataFrame({"data_evento": ["2022-05-28"]}))
        assert r["coluna_presente_na_tabela_unica"] is False
        assert r["declarada_no_esquema_alvo"] is False
        assert "regra de ouro" in r["consequencia"]

    def test_conta_os_valores_quando_a_coluna_chega(self):
        r = cobertura_da_precisao(pd.DataFrame(
            {"precisao_data": ["chip_stac", "chip_stac", "ativacao_cems"]}))
        assert r["coluna_presente_na_tabela_unica"] is True
        assert r["valores"] == {"chip_stac": 2, "ativacao_cems": 1}


class TestPrecisaoVersusGranularidade:
    def test_precisao_otima_com_carimbo_de_lote_e_acusada(self):
        # sen1floods11: a data e da cena que o analista rotulou -- melhor
        # precisao possivel -- e ainda assim vale para 2.766 pontos.
        d = pd.DataFrame({"fonte": ["s1f"] * 3,
                          "precisao_data": ["chip_stac"] * 3})
        analise = {"avaliado": True, "por_fonte": {"s1f": {
            "p": registro("p", n=3, classes=[1], mediana=3.6,
                          granularidade="carimbo_de_lote")}}}
        r = precisao_versus_granularidade(d, analise)
        assert r["por_fonte_e_precisao"]["s1f/chip_stac"][
            "declara_bem_e_carimba_lote"] is True

    def test_precisao_aproximada_com_lote_nao_e_discordancia(self):
        # ativacao_cems ja declara que e aproximacao: lote era o esperado
        d = pd.DataFrame({"fonte": ["cems"], "precisao_data": ["ativacao_cems"]})
        analise = {"avaliado": True, "por_fonte": {"cems": {
            "p": registro("p", n=1, classes=[1], mediana=10.7,
                          granularidade="carimbo_de_lote")}}}
        r = precisao_versus_granularidade(d, analise)
        assert not r["por_fonte_e_precisao"]["cems/ativacao_cems"][
            "declara_bem_e_carimba_lote"]

    def test_sem_a_coluna_nao_avalia(self):
        r = precisao_versus_granularidade(pd.DataFrame({"fonte": ["x"]}),
                                          {"avaliado": True, "por_fonte": {}})
        assert r["avaliado"] is False


def test_limiar_pluvial_nao_e_limiar_de_decisao_do_produto():
    # 5 mm e o piso abaixo do qual a diferenca entre classes deixa de ser
    # atribuivel a chuva, e nao um corte servido ao usuario.
    assert MM_MINIMO_PARA_LEITURA_PLUVIAL == 5.0
