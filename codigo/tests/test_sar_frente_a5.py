"""Frente A, etapa A5: a rede medida contra o baseline, na mesma regua.

O risco desta etapa nao e a rede ir mal -- e a comparacao ser injusta sem que
ninguem note. Tres formas de vencer o Otsu sem melhorar nada: medir em recorte
enquanto ele foi medido em chip inteiro, usar outra mascara, ou reimplementar a
metrica. Os testes abaixo fecham as tres.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "sar"))
sys.path.insert(0, str(ROOT / "scripts" / "servico"))

pytest.importorskip("torch")
pytest.importorskip("rasterio")

from dados_sen1floods11 import DADOS  # noqa: E402

TEM_DADO = DADOS.exists()
sem_dado = pytest.mark.skipif(not TEM_DADO,
                              reason="Sen1Floods11 ausente em local_runs")
CHECKPOINT_FUMACA = (ROOT / "local_runs" / "sar-inundacao"
                     / "unet_fumaca_manifesto.json")
sem_checkpoint = pytest.mark.skipif(
    not CHECKPOINT_FUMACA.exists(),
    reason="checkpoint de fumaca ausente; rode scripts/sar/treino.py")


def test_a_metrica_e_a_mesma_funcao_do_baseline():
    """Nao 'equivalente': a MESMA.

    Duas copias de uma regra de metrica divergem, e a divergencia apareceria
    como 'a rede ganhou' quando o que mudou foi quem contava.
    """
    import avaliar_rede
    import baseline_otsu

    assert avaliar_rede.contagens is baseline_otsu.contagens
    assert avaliar_rede.metricas is baseline_otsu.metricas


def test_checkpoint_ausente_falha_alto():
    """Um numero orfao e pior que nenhum numero."""
    from avaliar_rede import carregar

    with pytest.raises(FileNotFoundError):
        carregar("checkpoint_que_nao_existe_12345")


@sem_dado
@sem_checkpoint
def test_avaliacao_usa_o_chip_inteiro():
    """O treino usa recorte de 256; a avaliacao nao pode.

    O baseline foi medido em 512x512, e comparar numeros medidos em unidades
    diferentes e a forma mais silenciosa de vencer uma comparacao.
    """
    from avaliar_rede import carregar, prever

    modelo, manifesto = carregar("fumaca")
    from dados_sen1floods11 import nomes_do_split

    nome = nomes_do_split("validacao")[0]
    pred, prob = prever(modelo, nome, manifesto["normalizacao"])
    assert pred.shape == (512, 512)
    assert prob.shape == (512, 512)
    assert prob.min() >= 0.0 and prob.max() <= 1.0


@sem_dado
@sem_checkpoint
def test_pixel_sem_leitura_fica_fora_da_metrica_da_rede():
    """A rede nao teve entrada ali: nao se credita nem se pune."""
    from avaliar_rede import avaliar

    r = avaliar("fumaca", "validacao")
    por_chip = r["por_chip"]
    assert por_chip, "nenhum chip avaliado"
    for c in por_chip:
        # nenhum chip pode reportar mais pixels validos que o total
        assert c["pixels_validos"] <= 512 * 512
        assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == c["pixels_validos"]


@sem_dado
@sem_checkpoint
def test_a_recusa_tira_o_chip_da_metrica_e_o_declara():
    """A7 em forma de opcao: predicao morna nao vira 'sem agua'.

    Com limiar 1.0 nenhuma margem passa, entao todo chip EM QUE A REDE SE
    COMPROMETE e recusado, e a recusa aparece nomeada, nunca como silencio.

    O chip sem nenhum pixel predito como agua e o caso oposto e continua
    entrando na metrica: ali a rede TEM posicao -- ela diz que nao ha agua --,
    e transformar isso em recusa confundiria "nao sei" com "nao ha".
    """
    from avaliar_rede import MARGEM_NA_PREDICAO, avaliar

    r = avaliar("fumaca", "validacao", limiar_confianca=1.0,
                modo_confianca=MARGEM_NA_PREDICAO)["resumo"]
    assert r["chips_recusados"] > 0
    assert r["recusados"] and "certeza" in r["recusados"][0]
    assert r["recusados"][0]["modo"] == MARGEM_NA_PREDICAO
    # e os que sobraram sao exatamente os que nao predizem agua nenhuma
    sobreviventes = avaliar("fumaca", "validacao", limiar_confianca=1.0,
                            modo_confianca=MARGEM_NA_PREDICAO)["por_chip"]
    assert all(c["agua_predita"] == 0 for c in sobreviventes)


def test_so_uma_das_medidas_pode_se_abster():
    """A diferenca de fundo entre as duas medidas, verificada na funcao.

    `media_do_chip` SEMPRE tem opiniao, porque olha o chip inteiro e o chip
    inteiro sempre existe. `margem_na_predicao` se abstem quando a rede nao
    predisse agua nenhuma -- e a abstencao ali nao e recusa: e a rede dizendo
    que nao ha agua.

    Verificado direto na funcao em vez de por uma avaliacao completa: o
    comportamento e da medida, e uma corrida de 90 chips demoraria minutos
    para provar a mesma coisa.
    """
    from avaliar_rede import MARGEM_NA_PREDICAO, MEDIA_DO_CHIP, confianca

    prob = np.full((16, 16), 0.02)              # nenhuma predicao de agua
    valido = np.ones((16, 16), dtype=bool)
    pred = prob >= 0.5

    assert confianca(prob, valido, pred, MEDIA_DO_CHIP) is not None
    assert confianca(prob, valido, pred, MARGEM_NA_PREDICAO) is None
    # e quando nada e valido, nenhuma das duas inventa um numero
    vazio = np.zeros((16, 16), dtype=bool)
    assert confianca(prob, vazio, pred, MEDIA_DO_CHIP) is None
    assert confianca(prob, vazio, pred, MARGEM_NA_PREDICAO) is None


@sem_dado
@sem_checkpoint
def test_sem_limiar_nenhum_chip_e_recusado():
    from avaliar_rede import avaliar

    r = avaliar("fumaca", "validacao")["resumo"]
    assert r["chips_recusados"] == 0
    assert r["limiar_confianca"] is None


@sem_dado
@sem_checkpoint
def test_comparacao_publica_a_diferenca_com_o_sinal_que_der():
    """Perder para o Otsu e resultado publicavel, e tem de sair no relatorio."""
    from avaliar_rede import avaliar, comparar_com_baseline

    r = avaliar("fumaca", "teste")["resumo"]
    c = comparar_com_baseline(r, "teste")
    if not c["disponivel"]:
        pytest.skip("baseline do split de teste ausente; rode a etapa A2")
    for bloco in ("micro", "micro_sem_agua_permanente",
                  "macro", "macro_sem_agua_permanente"):
        assert bloco in c
        for m in ("iou", "f1", "precisao", "recall"):
            assert set(c[bloco][m]) == {"rede", "otsu", "diferenca"}
    # o checkpoint de fumaca NAO deve ganhar do baseline; se ganhasse, a
    # suspeita correta seria de bug na comparacao, e nao de sorte.
    assert c["micro"]["iou"]["diferenca"] < 0


@sem_dado
@sem_checkpoint
def test_o_resumo_declara_o_que_nao_pode_concluir():
    from avaliar_rede import avaliar

    r = avaliar("fumaca", "validacao")["resumo"]
    # As tres fronteiras que o numero desta etapa NAO atravessa.
    for fronteira in ("Recife", "Curitiba", "Petropolis",
                      "nao desloca escore"):
        assert fronteira in r["nao_e"], fronteira
    # E o numero nunca sai orfao do peso que o produziu.
    assert r["hash_estado"] and r["seed"] is not None
    assert r["parametros"] > 0 and r["epocas_treinadas"] > 0
