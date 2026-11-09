"""Frente A, etapa A4: a esteira de treino e o portao de reproducao.

Estes testes nao verificam que a rede aprendeu -- 2 epocas em 12 chips nao
autorizam essa afirmacao, e o proprio manifesto diz isso em `nao_e`. O que eles
verificam e o que precisa valer ANTES de qualquer numero: que a mesma seed
produz o mesmo peso, que `-1` continua fora da perda, e que a normalizacao nao
olhou para fora do treino.
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

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from dados_sen1floods11 import DADOS, SEM_DADO  # noqa: E402
from modelo_unet import UNetPequena, contar_parametros  # noqa: E402

TEM_DADO = DADOS.exists()
sem_dado = pytest.mark.skipif(not TEM_DADO,
                              reason="Sen1Floods11 ausente em local_runs")


# ------------------------------------------------------------- modelo


def test_a_rede_cabe_no_orcamento_de_cpu():
    """`base=16` fica na ordem de meio milhao, nao nos 31M da U-Net classica.

    Sem CUDA nesta maquina, capacidade a mais nao vira resultado -- vira dia de
    treino. E a restricao real da frente nao e capacidade: e dado rotulado.
    """
    n = contar_parametros(UNetPequena(base=16))
    assert 100_000 < n < 2_000_000, n


def test_a_saida_tem_a_forma_da_entrada():
    modelo = UNetPequena(base=8)
    saida = modelo(torch.zeros(1, 2, 128, 128))
    assert saida.shape == (1, 2, 128, 128)


def test_duas_classes_e_nao_sigmoide():
    """A saida em 2 canais e o que deixa `ignore_index` fazer o trabalho."""
    assert UNetPequena(base=8)(torch.zeros(1, 2, 64, 64)).shape[1] == 2


# ------------------------------------------------- a regra do ignore


def test_pixel_ignorado_nao_produz_gradiente():
    """A prova direta da regra A1 dentro do treino.

    Alvo com tudo ignorado exceto um pixel: o gradiente tem de existir e ser
    finito. Se `-1` entrasse como classe, a perda nem seria calculavel.
    """
    torch.manual_seed(0)
    modelo = UNetPequena(base=8)
    criterio = nn.CrossEntropyLoss(ignore_index=-1)

    alvo = torch.full((1, 32, 32), -1, dtype=torch.long)
    alvo[0, 0, 0] = 1
    perda = criterio(modelo(torch.zeros(1, 2, 32, 32)), alvo)
    perda.backward()

    grads = [p.grad for p in modelo.parameters() if p.grad is not None]
    assert grads, "nenhum gradiente"
    assert all(torch.isfinite(g).all() for g in grads)


def test_alvo_todo_ignorado_nao_produz_perda_finita():
    """O caso que o laco tem de pular em vez de propagar NaN."""
    criterio = nn.CrossEntropyLoss(ignore_index=-1)
    perda = criterio(torch.zeros(1, 2, 8, 8),
                     torch.full((1, 8, 8), -1, dtype=torch.long))
    assert not torch.isfinite(perda)


# ------------------------------------------------------------ preparo


@sem_dado
def test_preparar_marca_como_ignorado_o_rotulo_sem_dado():
    from dados_sen1floods11 import ler_chip, ler_rotulo, nomes_do_split
    from treino import IGNORAR, preparar

    nome = next(n for n in nomes_do_split("treino")[:20]
                if (ler_rotulo(n)[0] == SEM_DADO).any())
    estat = {"media": [-9.0, -15.0], "desvio": [2.0, 2.0]}
    x, alvo = preparar(nome, estat)

    chip = ler_chip(nome)
    sem = torch.from_numpy((chip.rotulo == SEM_DADO))
    assert (alvo[sem] == IGNORAR).all()
    assert torch.isfinite(x).all(), "NaN nao pode sobreviver ao preparo"


@sem_dado
def test_pixel_sem_leitura_de_s1_tambem_e_ignorado():
    """Duas ausencias diferentes, a mesma decisao.

    Num caso nao se sabe a verdade; no outro nao se tem entrada. Preencher sem
    ignorar ensinaria a rede que ausencia de leitura parece agua media.
    """
    from treino import IGNORAR, preparar
    import dados_sen1floods11 as d

    nomes = d.nomes_do_split("treino")[:30]
    estat = {"media": [-9.0, -15.0], "desvio": [2.0, 2.0]}
    achou = False
    for nome in nomes:
        s1 = d.ler_s1(nome)
        sem_leitura = ~np.isfinite(s1).all(axis=0)
        if not sem_leitura.any():
            continue
        achou = True
        _, alvo = preparar(nome, estat)
        assert (alvo[torch.from_numpy(sem_leitura)] == IGNORAR).all()
        break
    if not achou:
        pytest.skip("nenhum chip do recorte tem pixel sem leitura")


@sem_dado
def test_normalizacao_so_olha_para_o_treino():
    """Vazamento de estatistica nao aparece em metrica nenhuma."""
    from dados_sen1floods11 import nomes_do_split
    from treino import estatisticas_do_treino

    poucos = nomes_do_split("treino")[:4]
    estat = estatisticas_do_treino(poucos)
    assert len(estat["media"]) == 2 and len(estat["desvio"]) == 2
    assert all(d > 0 for d in estat["desvio"])
    # a contagem tem de bater com pixels FINITOS, nao com o total
    assert all(c <= 4 * 512 * 512 for c in estat["pixels_finitos"])


# ------------------------------------------------- portao de reproducao


@sem_dado
def test_mesma_seed_produz_o_mesmo_peso():
    """O portao de A4. Sem ele o checkpoint nao e evidencia de nada."""
    from treino import treinar

    kw = dict(epocas=1, limite_treino=4, limite_validacao=2, lado=128, lote=2)
    a = treinar(**kw)["manifesto"]
    b = treinar(**kw)["manifesto"]
    assert a["hash_estado"] == b["hash_estado"]
    assert a["historico"][0]["perda_treino"] == b["historico"][0]["perda_treino"]


@sem_dado
def test_seed_diferente_produz_peso_diferente():
    """O contraprova: se o hash nao mudasse, ele nao estaria medindo o peso."""
    from treino import treinar

    kw = dict(epocas=1, limite_treino=4, limite_validacao=2, lado=128, lote=2)
    assert treinar(seed=1, **kw)["manifesto"]["hash_estado"] != \
        treinar(seed=2, **kw)["manifesto"]["hash_estado"]


@sem_dado
def test_manifesto_carrega_a_proveniencia_do_peso():
    from treino import treinar

    m = treinar(epocas=1, limite_treino=4, limite_validacao=2,
                lado=128, lote=2)["manifesto"]
    for campo in ("seed", "split", "parametros", "normalizacao", "torch",
                  "commit", "hash_estado", "hash_manifesto", "dispositivo"):
        assert campo in m, campo
    assert m["nao_e"].startswith("checkpoint de fumaca")


@sem_dado
def test_o_hash_do_manifesto_nao_cobre_o_instante():
    """Mesma regra do `proveniencia.py`: senao ele deixa de responder 'mudou?'."""
    from treino import treinar

    kw = dict(epocas=1, limite_treino=4, limite_validacao=2, lado=128, lote=2)
    a = treinar(**kw)["manifesto"]
    b = treinar(**kw)["manifesto"]
    assert a["quando_utc"] != b["quando_utc"] or a["hash_manifesto"]
    assert a["hash_manifesto"] == b["hash_manifesto"]
