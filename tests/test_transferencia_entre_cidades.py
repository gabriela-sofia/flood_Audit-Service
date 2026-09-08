"""O teste da segunda cidade tem de continuar dizendo o que ele diz.

POR QUE ESTE TESTE EXISTE:

Este e o unico teste de transferencia que o projeto executou, e ele REPROVA parte do
que o modelo usa: `twi_dinf`, a interacao `twi_dinf` x chuva e `elevation_rel_m` saem
com sinal invertido em Chicago em relacao a Nova York.

Resultado que contraria o projeto e o primeiro a sumir de um pacote publico -- nao por
ma-fe, e porque ninguem escreve com entusiasmo a secao que reprova o proprio trabalho.
Os testes abaixo tornam o sumico uma falha de suite.

Eles nao verificam se o resultado e "bom". Verificam que ele esta inteiro, que a
comparacao entre as duas cidades e internamente coerente, e que o documento publico
continua declarando a reprovacao.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))

CHICAGO = RAIZ / "modelo/execucoes/chicago-teste-da-lei/teste_da_lei_chicago.json"
DEFEITO = (RAIZ / "modelo/execucoes/chicago-teste-da-lei"
           / "teste_da_lei_chicago_defeito_centroide.json")
NYC = (RAIZ / "modelo/execucoes/nyc-terreno-sob-gatilho"
       / "terreno_sob_gatilho_1km.json")


@pytest.fixture(scope="module")
def chicago() -> dict:
    return json.loads(CHICAGO.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nyc() -> dict:
    return json.loads(NYC.read_text(encoding="utf-8"))


TERMOS = ["hand_m", "twi_dinf", "slope_deg", "elevation_rel_m",
          "mrms_qpe_24h_mm", "twi_dinf_x_chuva", "hand_m_x_chuva"]

# Os tres termos cuja transferencia falhou. Se algum sair desta lista, ou o dado
# mudou (e o pacote inteiro precisa ser relido) ou alguem editou o artefato.
INVERTERAM = ["twi_dinf", "twi_dinf_x_chuva", "elevation_rel_m"]
REPLICARAM = ["slope_deg", "mrms_qpe_24h_mm"]


# =========================================================================
# 1. o estimador e o mesmo nas duas cidades
# =========================================================================

def test_o_estimador_de_chicago_e_importado_do_de_nova_york(chicago):
    """Estimador reescrito tornaria a diferenca indistinguivel de bug."""
    assert chicago["estimador_importado_de"] == \
        "scripts/chuva/terreno_sob_gatilho_1km.py"
    assert (RAIZ / chicago["estimador_importado_de"]).exists()


def test_o_script_de_chicago_importa_de_fato_e_nao_so_declara():
    """A declaracao no JSON so vale se o codigo fizer o que ela diz."""
    fonte = (RAIZ / "scripts/chuva/terreno_sob_gatilho_chicago.py"
             ).read_text(encoding="utf-8")
    assert "from chuva.terreno_sob_gatilho_1km import" in fonte
    for funcao in ("ajustar", "centrar", "com_interacoes",
                   "estratos_utilizaveis", "ic_por_estrato"):
        assert funcao in fonte, funcao


def test_as_duas_cidades_usam_o_mesmo_produto_de_chuva(chicago, nyc):
    assert "mrms_qpe_24h_mm" in chicago["efeitos"]
    assert "mrms_qpe_24h_mm" in nyc["efeitos"]


def test_o_criterio_de_sinal_foi_fixado_antes(nyc):
    exigido = nyc["desenho"]["criterios_fixados_antes"]["sinal_exigido"]
    assert exigido["twi_dinf"] == 1 and exigido["hand_m"] == -1
    assert exigido["slope_deg"] == -1 and exigido["elevation_rel_m"] == -1


# =========================================================================
# 2. a comparacao e internamente coerente
# =========================================================================

@pytest.mark.parametrize("termo", TERMOS)
def test_o_coeficiente_de_chicago_na_comparacao_e_o_mesmo_do_ajuste(chicago, termo):
    assert chicago["comparacao_com_nyc"][termo]["chicago"]["coef"] == \
        chicago["efeitos"][termo]["coef"]


@pytest.mark.parametrize("termo", TERMOS)
def test_o_coeficiente_de_nova_york_na_comparacao_e_o_mesmo_do_artefato_de_la(
        chicago, nyc, termo):
    """Comparar com um numero que nao esta no artefato de Nova York seria comparar
    com memoria, nao com evidencia."""
    assert chicago["comparacao_com_nyc"][termo]["nyc"]["coef"] == \
        nyc["efeitos"][termo]["coef"]


@pytest.mark.parametrize("termo", TERMOS)
def test_mesmo_sinal_confere_com_os_coeficientes(chicago, termo):
    c = chicago["comparacao_com_nyc"][termo]
    esperado = (c["nyc"]["coef"] > 0) == (c["chicago"]["coef"] > 0)
    assert c["mesmo_sinal"] is esperado, termo


@pytest.mark.parametrize("termo", TERMOS)
def test_sobreposicao_de_intervalos_confere_com_os_intervalos(chicago, termo):
    c = chicago["comparacao_com_nyc"][termo]
    a, b = c["nyc"]["ic95"]
    x, y = c["chicago"]["ic95"]
    esperado = not (b < x or y < a)
    assert c["intervalos_se_sobrepoem"] is esperado, termo


@pytest.mark.parametrize("termo", TERMOS)
def test_aprendido_exige_sinal_certo_e_intervalo_que_exclui_zero(chicago, termo):
    e = chicago["efeitos"][termo]
    assert e["aprendido"] is bool(e["ic95_exclui_zero"] and e["sinal_bate"]), termo


@pytest.mark.parametrize("termo", TERMOS)
def test_o_intervalo_contem_o_ponto(chicago, termo):
    e = chicago["efeitos"][termo]
    a, b = e["ic95"]
    assert a <= e["coef"] <= b, termo


@pytest.mark.parametrize("termo", TERMOS)
def test_ic95_exclui_zero_confere_com_o_intervalo(chicago, termo):
    a, b = chicago["efeitos"][termo]["ic95"]
    assert chicago["efeitos"][termo]["ic95_exclui_zero"] is (a > 0 or b < 0), termo


# =========================================================================
# 3. o veredito continua sendo o que e
# =========================================================================

@pytest.mark.parametrize("termo", INVERTERAM)
def test_os_termos_que_inverteram_continuam_invertidos(chicago, termo):
    c = chicago["comparacao_com_nyc"][termo]
    assert c["mesmo_sinal"] is False, termo
    assert c["intervalos_se_sobrepoem"] is False, termo
    assert chicago["efeitos"][termo]["ic95_exclui_zero"] is True, (
        f"{termo} inverteu, mas com IC cruzando zero nao seria contradicao medida")


@pytest.mark.parametrize("termo", REPLICARAM)
def test_os_termos_que_replicaram_continuam_replicando(chicago, termo):
    c = chicago["comparacao_com_nyc"][termo]
    assert c["mesmo_sinal"] is True, termo
    assert c["aprendido_nas_duas"] is True, termo


def test_chicago_declara_que_nao_pode_responder_sobre_o_hand(chicago):
    """Nulo por falta de contraste nao e nulo. O artefato tem de dizer isso."""
    h = chicago["hand_sem_contraste"]
    assert h["sem_contraste"] is True
    assert h["p95_do_hand_m"] < h["limiar_declarado_m"]
    assert "falta de variacao" in h["consequencia"]


def test_chicago_declara_que_nao_altera_modelo_servido(chicago):
    assert "nao ajusta modelo servido" in chicago["nao_e"]
    assert "nao promove variavel" in chicago["nao_e"]


def test_nova_york_declara_que_o_produto_de_chuva_nao_cobre_o_brasil(nyc):
    assert "nao transfere sozinho para regiao brasileira" in nyc["nao_e"]


# =========================================================================
# 4. o resultado defeituoso fica, e fica marcado
# =========================================================================

def test_a_versao_defeituosa_continua_versionada_e_e_diferente(chicago):
    """Ela existe para mostrar o que muda quando o desenho de amostragem muda."""
    assert DEFEITO.exists()
    d = json.loads(DEFEITO.read_text(encoding="utf-8"))
    assert d["amostra"]["pontos"] != chicago["amostra"]["pontos"]
    assert "defeito" in DEFEITO.name


def test_o_defeito_de_amostragem_esta_explicado_no_manifesto_da_aquisicao():
    m = json.loads((RAIZ / "modelo/execucoes/chicago-teste-da-lei/insumos"
                    / "aquisicao_chamados_manifesto.json").read_text(encoding="utf-8"))
    fundo = m["fundo"]
    assert fundo["dias_cobertos"] == fundo["dias_pedidos"], (
        "o defeito era justamente fundo faltando em quase todos os dias")
    assert fundo["dias_perdidos"] == []
    assert "78 dias de 2.438" in fundo["por_que_por_dia"]
    assert fundo["vies_que_permanece"], "vies residual tem de continuar declarado"


# =========================================================================
# 5. o documento publico continua contando isso
# =========================================================================

@pytest.mark.parametrize("documento", ["README.md", "docs/EVIDENCIA.md",
                                       "docs/METODO.md", "docs/LIMITACOES.md",
                                       "modelo/MODELO.md"])
def test_o_documento_publico_menciona_a_segunda_cidade(documento):
    texto = (RAIZ / documento).read_text(encoding="utf-8")
    assert "Chicago" in texto, documento


def test_a_evidencia_declara_a_reprovacao_em_palavras():
    texto = (RAIZ / "docs/EVIDENCIA.md").read_text(encoding="utf-8")
    assert "Reprovada" in texto or "reprova" in texto
    assert "elevation_rel_m" in texto


def test_o_metodo_avisa_que_a_variavel_de_transferencia_inverte():
    """`elevation_rel_m` e apresentada como a variavel que permite comparar
    regioes. Quem le isso precisa ler, na mesma pagina, que o sinal dela nao se
    manteve na unica segunda cidade testada."""
    texto = (RAIZ / "docs/METODO.md").read_text(encoding="utf-8")
    trecho = texto[texto.find("elevation_rel_m"):]
    trecho = trecho[:trecho.find("| `slope_deg`")]
    assert "inverte" in trecho.lower(), (
        "a linha de elevation_rel_m no METODO nao diz que o sinal inverteu")
