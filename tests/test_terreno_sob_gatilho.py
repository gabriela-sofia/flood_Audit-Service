"""O desenho que mede terreno sob forçamento, travado em teste.

Estes dois scripts produzem AFIRMACAO -- "o TWI e o mecanismo", "o efeito cresce
com a dose" -- e ate agora nao tinham teste. A centragem por estrato e a peca
que sustenta as duas: se ela nao remover o efeito fixo do estrato, o que sobra
nao e comparacao dentro do dia, e o resultado inteiro muda de significado sem
mudar de aparencia.

Os testes usam dados sinteticos onde a resposta certa e conhecida por
construcao, e nao dependem de nenhum artefato em `local_runs/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from chuva import gatilho_terreno_nyc as g11km  # noqa: E402
from chuva import terreno_sob_gatilho_1km as g1km  # noqa: E402


# ------------------------------------------------ centragem por estrato

def test_a_centragem_remove_o_efeito_fixo_do_estrato():
    """Dois estratos com niveis diferentes e o MESMO padrao interno.

    Depois de centrar, os dois tem de ficar identicos: e isso que faz a
    comparacao ser dentro do estrato, e nao entre estratos.
    """
    s = pd.DataFrame({
        "dia": ["A", "A", "A", "B", "B", "B"],
        "hand_m": [10.0, 12.0, 14.0, 110.0, 112.0, 114.0],
    })
    z = g1km.centrar(s, ["hand_m"])
    assert z[:3].ravel() == pytest.approx(z[3:].ravel())
    assert z.ravel() == pytest.approx([-2.0, 0.0, 2.0, -2.0, 0.0, 2.0])


def test_variavel_constante_no_estrato_perde_toda_a_variancia():
    """A chuva a 11 km e constante na celula-dia: e por isso que ela nao entra.

    Se este teste falhar, a leitura de identificabilidade do projeto inteiro
    deixa de valer -- ele e o que garante que 'constante por construcao' e
    verdade e nao suposicao.
    """
    s = pd.DataFrame({
        "dia": ["A", "A", "B", "B"],
        "chuva": [5.0, 5.0, 40.0, 40.0],
        "terreno": [1.0, 9.0, 2.0, 8.0],
    })
    v = g1km.variacao_interna(s, ["chuva", "terreno"])
    assert v["chuva"] == pytest.approx(0.0)
    assert v["terreno"] > 0.5


# ------------------------------------------------ selecao de estratos

def test_estrato_de_uma_classe_so_nao_informa_e_sai():
    """Estrato sem as duas classes nao contribui para a verossimilhanca."""
    s = pd.DataFrame({
        "dia": ["A"] * 10 + ["B"] * 10,
        "classe": [0] * 10 + [0] * 5 + [1] * 5,
        "hand_m": np.arange(20, dtype=float),
    })
    fica = g1km.estratos_utilizaveis(s)
    assert set(fica.dia) == {"B"}


def test_estrato_pequeno_demais_sai_mesmo_com_as_duas_classes():
    s = pd.DataFrame({
        "dia": ["A", "A", "B", "B", "B", "B", "B", "B", "B", "B"],
        "classe": [0, 1] + [0, 1] * 4,
        "hand_m": np.arange(10, dtype=float),
    })
    fica = g1km.estratos_utilizaveis(s)
    assert set(fica.dia) == {"B"}


def test_o_estrato_de_11km_e_celula_por_dia_e_nao_so_o_dia():
    """A diferenca entre os dois desenhos e o estrato, e ela e o ponto todo."""
    assert g11km.ESTRATO == ["cl_lat", "cl_lon", "dia"]
    assert g1km.ESTRATO == ["dia"]


# ------------------------------------------------ contrato de sinal

def test_toda_variavel_do_ajuste_tem_sinal_exigido_declarado():
    """Variavel sem sinal declarado entraria no escore sem contrato fisico.

    E o portao que o projeto aplica desde o SVC-01: o sinal e fixado ANTES de
    rodar, e resultado com sinal invertido e publicado, nao escondido.
    """
    for f in g11km.TERRENO:
        assert f in g11km.SINAL_EXIGIDO, f
    esperadas = set(g1km.TERRENO) | {g1km.CHUVA} | {
        f"{f}_x_chuva" for f in g1km.INTERACOES}
    faltando = esperadas - set(g1km.SINAL_EXIGIDO)
    assert not faltando, f"sem sinal declarado: {sorted(faltando)}"


def test_os_sinais_das_interacoes_seguem_a_fisica_das_variaveis_base():
    """TWI x chuva positivo, HAND x chuva negativo: mais chuva separa mais."""
    assert g1km.SINAL_EXIGIDO["twi_dinf_x_chuva"] == +1
    assert g1km.SINAL_EXIGIDO["hand_m_x_chuva"] == -1
    assert g1km.SINAL_EXIGIDO["twi_dinf"] == +1
    assert g1km.SINAL_EXIGIDO["hand_m"] == -1


# ------------------------------------------------ interacoes

def test_a_interacao_e_o_produto_das_variaveis_JA_padronizadas():
    """Sem padronizar antes, metros x milimetros dominaria o ajuste pela escala."""
    s = pd.DataFrame({
        "dia": ["A"] * 6,
        "classe": [0, 1, 0, 1, 0, 1],
        "hand_m": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "twi_dinf": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "slope_deg": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "elevation_rel_m": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        g1km.CHUVA: [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
    })
    z, nomes = g1km.com_interacoes(s)
    assert "twi_dinf_x_chuva" in nomes and "hand_m_x_chuva" in nomes
    # cada base fica padronizada
    for f in g1km.TERRENO + [g1km.CHUVA]:
        assert z[f].mean() == pytest.approx(0.0, abs=1e-9)
        assert z[f].std() == pytest.approx(1.0, abs=1e-9)
    # e a interacao e literalmente o produto delas
    assert z["twi_dinf_x_chuva"].to_numpy() == pytest.approx(
        (z["twi_dinf"] * z[g1km.CHUVA]).to_numpy())


def test_variavel_constante_nao_explode_a_padronizacao():
    """Desvio zero viraria divisao por zero e NaN silencioso no ajuste."""
    s = pd.DataFrame({
        "dia": ["A"] * 4,
        "classe": [0, 1, 0, 1],
        "hand_m": [5.0, 5.0, 5.0, 5.0],
        "twi_dinf": [1.0, 2.0, 3.0, 4.0],
        "slope_deg": [1.0, 2.0, 3.0, 4.0],
        "elevation_rel_m": [1.0, 2.0, 3.0, 4.0],
        g1km.CHUVA: [1.0, 2.0, 3.0, 4.0],
    })
    z, _ = g1km.com_interacoes(s)
    assert np.isfinite(z["hand_m"].to_numpy()).all()


# ------------------------------------------------ ajuste

def test_o_ajuste_recupera_o_sinal_que_foi_plantado():
    """Prova de que a maquinaria mede o que diz medir.

    Dados construidos para que HAND alto empurre a classe para BAIXO. O
    coeficiente tem de sair negativo.
    """
    rng = np.random.default_rng(20260902)
    n = 4000
    hand = rng.normal(size=n)
    eta = -1.5 * hand
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    beta = g1km.ajustar(hand.reshape(-1, 1), y)
    assert beta[0] < 0
    assert beta[0] == pytest.approx(-1.5, abs=0.25)


def test_o_ajuste_nao_usa_intercepto_porque_o_estrato_ja_saiu():
    """Intercepto depois da centragem re-introduziria o nivel do estrato."""
    import inspect

    fonte = inspect.getsource(g1km.ajustar)
    assert "fit_intercept=False" in fonte
    assert "fit_intercept=False" in inspect.getsource(g11km.ajustar)
