"""Frente A, etapa A7: a calibracao do limiar de recusa.

Recusar melhora quase qualquer metrica -- basta recusar os chips dificeis e o
IoU dos que sobram sobe. O que estes testes protegem e o criterio: o limiar sai
do orcamento de cobertura, declarado antes, e o IoU e consequencia relatada, e
nunca o motivo da escolha.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "sar"))
sys.path.insert(0, str(ROOT / "scripts" / "servico"))

pytest.importorskip("torch")
pytest.importorskip("rasterio")


CURVA = [
    {"limiar": 0.0, "custo_de_cobertura": 0.0, "chips_avaliados": 100,
     "iou_micro_sem_agua_permanente": 0.30},
    {"limiar": 0.2, "custo_de_cobertura": 0.08, "chips_avaliados": 92,
     "iou_micro_sem_agua_permanente": 0.34},
    {"limiar": 0.4, "custo_de_cobertura": 0.40, "chips_avaliados": 60,
     "iou_micro_sem_agua_permanente": 0.55},
]


def test_o_iou_nao_escolhe_o_limiar():
    """O ponto de IoU 0,55 e o mais tentador, e custa 40% de cobertura."""
    from calibrar_recusa import escolher

    e = escolher(CURVA, custo_maximo=0.1)
    assert e["limiar"] == 0.2
    assert e["limiar"] != 0.4, "o limiar de maior IoU nao pode vencer por isso"


def test_o_ganho_e_relatado_e_nao_usado():
    from calibrar_recusa import escolher

    e = escolher(CURVA, custo_maximo=0.1)
    assert e["ganho_de_iou_sem_agua_permanente"] == pytest.approx(0.04)
    assert "consequencia relatada" in e["por_que"]
    assert "nao separa" in e["leitura_se_ganho_nulo"]


def test_orcamento_impossivel_devolve_nenhum_limiar():
    """Sem limiar dentro do orcamento, a resposta e None -- e nao o menos pior."""
    from calibrar_recusa import escolher

    e = escolher([CURVA[2]], custo_maximo=0.05)
    assert e["limiar"] is None
    assert "confianca separavel" in e["por_que"]


def test_calibrar_no_teste_e_bloqueado():
    """Calibrar no teste transforma o teste em conjunto de ajuste."""
    from calibrar_recusa import calibrar

    with pytest.raises(ValueError, match="conjunto de ajuste"):
        calibrar("fumaca", "teste")


def test_calibrar_no_holdout_e_bloqueado():
    """Gastar holdout em calibracao e a forma mais comum de perder um holdout."""
    from calibrar_recusa import calibrar

    with pytest.raises(ValueError, match="holdout"):
        calibrar("fumaca", "bolivia")


def test_a_recusa_nao_significa_ausencia_de_agua():
    """O contrato ja tem nome para isso, e o modulo tem de usar o mesmo."""
    import calibrar_recusa

    assert "nao_identificavel" in calibrar_recusa.__doc__


# ------------------------------------------------- a medida de confianca


def test_a_media_do_chip_e_dominada_pelo_fundo():
    """O defeito que a calibracao real revelou, preservado como teste.

    Num chip tipico ~87% dos pixels sao fundo obvio, que a rede acerta com
    probabilidade quase zero. A media sobre o chip inteiro sai altissima mesmo
    quando a rede esta em duvida EXATAMENTE onde se compromete -- e na rodada
    de validacao todos os limiares de 0,05 a 0,50 recusaram os mesmos tres
    chips, com ganho de IoU igual a zero.
    """
    import numpy as np
    from avaliar_rede import MEDIA_DO_CHIP, MARGEM_NA_PREDICAO, confianca

    prob = np.full((100, 100), 0.01)
    prob[:13, :] = 0.55                     # agua predita, com duvida real
    valido = np.ones((100, 100), dtype=bool)
    pred = prob >= 0.5

    dominada = confianca(prob, valido, pred, MEDIA_DO_CHIP)
    honesta = confianca(prob, valido, pred, MARGEM_NA_PREDICAO)
    assert dominada > 0.8, "o fundo infla a media"
    assert honesta < 0.2, "onde ela se compromete, a rede esta em duvida"


def test_chip_sem_predicao_de_agua_nao_e_recusa():
    """`None` nao e recusa: a rede TEM posicao ali -- ela diz que nao ha agua.

    Transformar isso em `nao_identificavel` confundiria "nao sei" com "nao ha",
    que e a distincao que a frente inteira existe para manter.
    """
    import numpy as np
    from avaliar_rede import MARGEM_NA_PREDICAO, confianca

    prob = np.full((10, 10), 0.01)
    assert confianca(prob, np.ones((10, 10), dtype=bool),
                     np.zeros((10, 10), dtype=bool),
                     MARGEM_NA_PREDICAO) is None


def test_modo_de_confianca_desconhecido_falha_alto():
    import numpy as np
    from avaliar_rede import confianca

    with pytest.raises(ValueError, match="modo de confianca"):
        confianca(np.zeros((4, 4)), np.ones((4, 4), dtype=bool),
                  np.zeros((4, 4), dtype=bool), "modo_inventado")
