"""Pre-treino auto-supervisionado do encoder da U-Net, sobre S1 e sem rotulo.

-------------------------------------------------------------------------
POR QUE A ESTRATEGIA MUDOU DE LUGAR
-------------------------------------------------------------------------

A frente B tentou provar que aprendizado sem rotulo melhora a representacao, e
nao conseguiu -- nao porque o aprendizado nao acontece, mas porque a REGUA nao
consegue enxergar diferencas desse tamanho:

    alvo         fracao de agua por chip
    n            337 escalares, 10 grupos
    ruido        83 passos de treino (7% do orcamento) moveram o Spearman 0,125

Todo efeito que aquela frente media -- 0,018 das AOIs, 0,025 da amostragem --
cabia dentro desse ruido. Nao ha experimento que decida ali.

Aqui o sinal e denso:

    alvo         mascara de agua por PIXEL
    n            446 chips x 512x512 = 117 milhoes de pixels rotulados
    razao        347 mil vezes mais sinal supervisionado que a frente B
    regua        IoU sem agua permanente, contra o baseline Otsu do dataset

E o desenho e o que a referencia indexada recomenda: He et al. 2022 pre-treinam
mascarando 75% da entrada e avaliam por AJUSTE FINO, nao por sonda linear. A
frente B media por sonda linear -- justamente o caminho mais fraco do metodo.

-------------------------------------------------------------------------
O QUE E AUTONOMO AQUI, LITERALMENTE
-------------------------------------------------------------------------

Este arquivo NAO le `LabelHand`. Nem uma vez. O pre-treino usa os 7.136
recortes de 128x128 que saem dos 446 chips de S1 -- e cada chip novo que chegar
entra sozinho, porque a tarefa e reconstruir o que foi escondido.

O rotulo so aparece DEPOIS, no ajuste fino, e ali ele mede -- nao ensina o
encoder a olhar.

-------------------------------------------------------------------------
O MESMO ENCODER, E ISSO E O PONTO
-------------------------------------------------------------------------

Os blocos `enc1`, `enc2` e `enc3` sao os da `UNetPequena`, importados e nao
reescritos. Um encoder pre-treinado que nao encaixe na rede que vai usar ele
seria um experimento sobre outra arquitetura.

-------------------------------------------------------------------------
RODADA, E POR QUE ELA PRECISOU EXISTIR
-------------------------------------------------------------------------

Ate aqui toda corrida comecava do zero: `MAEdeSAR()` novo, `Adam` novo, e o
decoder e o estado do otimizador jogados fora no fim. Sob esse desenho, "a cada
rodada ele aprende mais" e IMPOSSIVEL por aritmetica, nao por limitacao do
modelo -- se cada corrida e um sorteio independente da mesma distribuicao, o
valor esperado da rodada N+1 e igual ao da rodada N. Insistencia nao acumula em
sistema que nao guarda a rodada anterior.

`--retomar` conserta isso. Uma RODADA e um numero inteiro de epocas sobre o
pote corrente, e ela retoma do estado completo da anterior: encoder, decoder,
otimizador e o gerador de aleatoriedade. As tres decisoes que fazem isso ser
retomada e nao reinicio disfarcado:

1. A FRONTEIRA E A EPOCA. O laco antigo parava no meio da epoca quando batia
   `passos_alvo`, e o `pote` de recortes daquela epoca morria com ela. Retomar
   dali reconstruiria um pote diferente, e a corrida retomada divergiria da
   continua sem que nada denunciasse. Com `--epocas-alvo` a rodada sempre fecha
   epoca, e o corte fica num ponto em que o unico estado vivo e (modelo,
   otimizador, gerador). `test_pretreino_retomada` prova a equivalencia tensor
   a tensor: 2 epocas continuas == 1 epoca + retomada de 1 epoca.

2. A NORMALIZACAO E HERDADA, E NAO RECALCULADA. Media e desvio das bandas sao
   propriedade do pote, e o pote cresce entre rodadas. Recalcular deslocaria a
   distribuicao de entrada DEBAIXO de um encoder ja ajustado a outra -- o peso
   continuaria la e passaria a receber outra escala. O manifesto herda a
   normalizacao da rodada 1 e registra em `deriva_da_normalizacao` o quanto ela
   teria mudado, para que o crescimento do pote fique visivel em vez de
   silencioso.

3. O ESQUECIMENTO E O RISCO CONHECIDO. Retomar sobre dado novo degrada o
   antigo. Por isso o pote e cumulativo -- chip que entrou nao sai -- e por
   isso existe `perda_de_reconstrucao` sobre chips que o pre-treino nunca
   viu: sem uma medida em dado retido, "a perda de treino caiu" nao distingue
   aprender de decorar.

-------------------------------------------------------------------------
O QUE ISTO NAO E
-------------------------------------------------------------------------

Nao e modelo de suscetibilidade e nao produz mascara publicavel. E uma
inicializacao de pesos, e ela so vale o que provar valer no ajuste fino contra
a mesma inicializacao aleatoria de antes.

E a perda de reconstrucao NAO e evidencia de representacao melhor. Ela mede se
a rodada aprendeu A TAREFA DELA. Se isso transfere para agua em SAR e pergunta
do ajuste fino, e quem decide promocao e a catraca -- nunca este arquivo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))
sys.path.insert(0, str(REPO / "scripts" / "neural"))

from baseline_otsu import SAIDA  # noqa: E402
from dados_sen1floods11 import ler_s1, nomes_do_split  # noqa: E402
from modelo_unet import _bloco  # noqa: E402
from proveniencia import agora_utc  # noqa: E402
from proveniencia_de_peso import (  # noqa: E402
    commit_atual, hash_completo_do_estado, hash_do_estado, selar, semear,
)

LADO = 128
# He et al. 2022: mascarar proporcao ALTA da entrada e o que torna a tarefa nao
# trivial. O encoder de terreno usava 50%, e essa escolha nunca foi testada.
FRACAO_MASCARA = 0.75
LADO_BLOCO = 16

# Chips do pote que ficam RETIDOS: entram na medida e nunca no gradiente.
# Saem do split de treino de proposito -- `validacao` e `teste` sao onde o
# ajuste fino mede, e gastar eles aqui contaminaria a medida que decide
# promocao.
CHIPS_RETIDOS = 20

# A medida tem gerador PROPRIO e semente FIXA. Se a mascara da avaliacao
# mudasse entre rodadas, a curva misturaria progresso com sorteio, e uma rodada
# pareceria melhor por ter recebido buraco mais facil.
SEED_DA_MEDIDA = 20260904


class EncoderUNet(nn.Module):
    """Os tres blocos de descida da `UNetPequena`, isolados.

    Isolados e nao copiados: `_bloco` e importado. Duas definicoes do mesmo
    encoder divergiriam, e a divergencia apareceria como "o pre-treino nao
    transferiu" quando o que mudou foi a arquitetura.
    """

    def __init__(self, canais_entrada: int = 2, base: int = 16):
        super().__init__()
        b = base
        self.enc1 = _bloco(canais_entrada, b)
        self.enc2 = _bloco(b, b * 2)
        self.enc3 = _bloco(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        return e1, e2, e3


class DecoderLeve(nn.Module):
    """Decoder assimetrico e leve, como a referencia pede.

    Ele existe so para a tarefa de reconstrucao e e DESCARTADO depois. Um
    decoder pesado aprenderia a reconstruir sozinho, e o encoder poderia
    entregar pouco sem que a perda denunciasse.
    """

    def __init__(self, base: int = 16, canais_saida: int = 2):
        super().__init__()
        self.corpo = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 2, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base, canais_saida, 1),
        )

    def forward(self, e3):
        return self.corpo(e3)


class MAEdeSAR(nn.Module):
    def __init__(self, canais: int = 2, base: int = 16):
        super().__init__()
        self.encoder = EncoderUNet(canais, base)
        self.decoder = DecoderLeve(base, canais)

    def forward(self, x):
        return self.decoder(self.encoder(x)[2])


def mascarar(x: torch.Tensor, fracao: float, lado_bloco: int,
             gerador: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Esconde blocos inteiros, e nao pixels soltos.

    Pixel solto e reconstruido por interpolacao do vizinho, o que nao exige
    entender nada. Bloco de 16x16 obriga a rede a usar contexto.
    """
    n, _, h, w = x.shape
    gh, gw = h // lado_bloco, w // lado_bloco
    sorteio = torch.rand((n, 1, gh, gw), generator=gerador)
    grade = sorteio < fracao
    escondido = grade.repeat_interleave(lado_bloco, 2).repeat_interleave(
        lado_bloco, 3)
    return x.masked_fill(escondido, 0.0), escondido


def perda_mascarada(saida, alvo, escondido, valido) -> torch.Tensor:
    """Erro so onde escondeu E onde ha leitura.

    A mesma regra da frente B, pela mesma razao: esconder um pedaco que ja era
    ausencia nao cria tarefa, e cobrar a reconstrucao dele ensinaria a rede a
    prever o valor de preenchimento.

    -------------------------------------------------------------------------
    A VALIDADE PODE SER POR PIXEL OU POR CANAL, e por que isso importa
    -------------------------------------------------------------------------

    Ate aqui `valido` era `(B, H, W)`: um pixel valia para TODOS os canais ou
    para nenhum, porque `preparar` fazia `np.isfinite(s1).all(axis=0)`. Com
    dois canais de radar da mesma cena isso e verdade -- se um pixel nao tem
    leitura, nao tem em nenhuma banda.

    Deixa de ser verdade no substrato multimodal. Recife tem terreno e NAO tem
    S1; um chip do Sen1Floods11 tem os dois. Com a validade colapsada por
    pixel, `.all(axis=0)` invalidaria o lugar inteiro por causa da modalidade
    ausente -- e as 340 AOIs e regioes que so tem terreno ficariam fora do
    pote sem que nada denunciasse.

    Entao `valido` passa a aceitar TAMBEM `(B, C, H, W)`, e a difusao faz o
    resto: `escondido` e `(B, 1, H, W)` porque a mascara e espacial e vale para
    todos os canais.

    O caminho antigo continua BIT A BIT igual -- `test_perda_por_pixel_e_por_
    canal_coincidem` prova isso --, e por isso a mudanca e aditiva: nada que ja
    roda precisa mudar de chamada.
    """
    if valido.dim() == saida.dim() - 1:      # (B, H, W): uma validade por pixel
        valido = valido.unsqueeze(1)
    alvo_da_perda = escondido & valido
    if not alvo_da_perda.any():
        return torch.zeros((), requires_grad=True)
    if alvo_da_perda.shape[1] == 1:
        alvo_da_perda = alvo_da_perda.expand_as(saida)
    return ((saida - alvo) ** 2)[alvo_da_perda].mean()


def estatisticas(nomes: list[str]) -> dict:
    """Media e desvio por banda, so em pixel com leitura, so no treino."""
    somas = np.zeros(2, dtype=np.float64)
    somas_q = np.zeros(2, dtype=np.float64)
    contagem = np.zeros(2, dtype=np.int64)
    for nome in nomes:
        s1 = ler_s1(nome).astype(np.float64)
        for i in range(2):
            vals = s1[i][np.isfinite(s1[i])]
            somas[i] += vals.sum()
            somas_q[i] += (vals ** 2).sum()
            contagem[i] += vals.size
    contagem[contagem == 0] = 1
    media = somas / contagem
    var = np.maximum(somas_q / contagem - media ** 2, 1e-12)
    return {"media": [round(float(v), 6) for v in media],
            "desvio": [round(float(v), 6) for v in np.sqrt(var)],
            "bandas": ["VV", "VH"]}


def preparar(nome: str, estat: dict) -> tuple[torch.Tensor, torch.Tensor]:
    s1 = ler_s1(nome).astype(np.float32)
    com_leitura = np.isfinite(s1).all(axis=0)
    media = np.array(estat["media"], dtype=np.float32).reshape(2, 1, 1)
    desvio = np.array(estat["desvio"], dtype=np.float32).reshape(2, 1, 1)
    x = (np.nan_to_num(s1, nan=0.0) - media) / desvio
    x[:, ~com_leitura] = 0.0
    return torch.from_numpy(x), torch.from_numpy(com_leitura)


def recortes(x, valido, lado, gerador, quantos):
    _, h, w = x.shape
    for _ in range(quantos):
        i = int(torch.randint(0, h - lado + 1, (1,), generator=gerador).item())
        j = int(torch.randint(0, w - lado + 1, (1,), generator=gerador).item())
        yield x[:, i:i + lado, j:j + lado], valido[i:i + lado, j:j + lado]


def dividir_o_pote(nomes: list[str],
                   retidos: int = CHIPS_RETIDOS) -> tuple[list[str], list[str]]:
    """Separa os chips retidos por PASSO REGULAR, e nao pelo fim da lista.

    O CSV do Sen1Floods11 vem agrupado por regiao. Pegar os ultimos N daria um
    conjunto retido de uma regiao so, e a perda medida ali responderia sobre
    aquele lugar em vez de sobre o pote. O passo regular atravessa os grupos.

    Deterministico de proposito: o mesmo pote tem de dar o mesmo corte em toda
    rodada, ou a curva entre rodadas compara medidas de conjuntos diferentes.
    """
    if retidos <= 0 or retidos >= len(nomes):
        return list(nomes), []
    passo = len(nomes) // retidos
    indices = {i * passo for i in range(retidos)}
    return ([n for k, n in enumerate(nomes) if k not in indices],
            [n for k, n in enumerate(nomes) if k in indices])


@torch.no_grad()
def perda_de_reconstrucao(modelo, nomes: list[str], estat: dict,
                          fracao_mascara: float = FRACAO_MASCARA,
                          lado_bloco: int = LADO_BLOCO, lado: int = LADO,
                          recortes_por_chip: int = 4,
                          lote: int = 8,
                          estatistica_do_lote: bool = True) -> float | None:
    """A mesma perda mascarada, em chips que o gradiente nunca tocou.

    Existe porque "a perda de treino caiu" nao separa aprender de decorar --
    e num pote de 252 chips com um encoder de 106 mil parametros, decorar e
    barato. Esta e a medida barata da rodada; a cara e o ajuste fino.

    Gerador local com `SEED_DA_MEDIDA`: alem de fixar a mascara entre rodadas,
    isso mantem a medida FORA do gerador do treino. Consumir aleatoriedade do
    gerador de treino aqui faria a corrida retomada divergir da continua, e a
    equivalencia que a rodada inteira depende de provar cairia por causa da
    propria medida.

    `estatistica_do_lote` normaliza pelo lote em vez de pelas medias moveis do
    BatchNorm, e o padrao e esse por uma razao medida, nao por gosto. Com as
    medias moveis, a mesma linhagem oscilou com desvio 0,183 entre rodadas; com
    estatistica do lote, 0,044 -- fator 4,2 na pior linhagem, e cerca de metade
    da variancia no conjunto das tres. As medias moveis sao um estado a parte,
    atualizado por media exponencial dos lotes e nao pelo otimizador, e uma
    epoca azarada no fim as desloca sem que o peso tenha piorado.

    E o que se seleciona aqui e um ENCODER PARA AJUSTE FINO, onde essas medias
    moveis sao reaprendidas do zero. Deixar que elas decidam a promocao seria
    escolher pelo que vai ser jogado fora.

    `momentum = 0` durante a passagem: sem isso a propria medicao atualizaria as
    medias moveis, e medir mudaria o objeto medido.
    """
    if not nomes:
        return None
    # SNAPSHOT dos buffers do BatchNorm, e nao so do `momentum`.
    #
    # `momentum = 0` ja impede que `running_mean` e `running_var` se movam, mas
    # `num_batches_tracked` continua incrementando em modo de treino -- e ele
    # entra no `state_dict`, logo entra no `hash_do_estado`. Medir passaria a
    # mudar a impressao digital do peso, e a proveniencia da rodada diria que o
    # peso mudou quando o que aconteceu foi uma avaliacao.
    guardados: dict = {}
    if estatistica_do_lote:
        modelo.train()
        for nome, m in modelo.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                guardados[nome] = (
                    m.momentum,
                    m.num_batches_tracked.clone(),
                    m.running_mean.clone() if m.running_mean is not None
                    else None,
                    m.running_var.clone() if m.running_var is not None
                    else None,
                )
                m.momentum = 0.0
    else:
        modelo.eval()
    gerador = torch.Generator().manual_seed(SEED_DA_MEDIDA)
    total, n = 0.0, 0
    pote = []
    for nome in nomes:
        x, v = preparar(nome, estat)
        pote.extend(recortes(x, v, lado, gerador, recortes_por_chip))
    # O ultimo bloco entra mesmo incompleto, e o peso e o tamanho dele. Descartar
    # o resto faria a medida depender de quantos chips retidos existem -- um
    # pote pequeno cairia para zero bloco e a perda voltaria `None` sem que
    # nada explicasse por que.
    for k in range(0, len(pote), lote):
        bloco = pote[k:k + lote]
        entrada = torch.stack([e[0] for e in bloco])
        valido = torch.stack([e[1] for e in bloco])
        mascarada, escondido = mascarar(entrada, fracao_mascara, lado_bloco,
                                        gerador)
        perda = perda_mascarada(modelo(mascarada), entrada, escondido, valido)
        if torch.isfinite(perda) and float(perda) > 0:
            total += float(perda) * len(bloco)
            n += len(bloco)
    for nome, m in modelo.named_modules():
        if nome in guardados:
            momento, contador, media, var = guardados[nome]
            m.momentum = momento
            m.num_batches_tracked.copy_(contador)
            if media is not None:
                m.running_mean.copy_(media)
            if var is not None:
                m.running_var.copy_(var)
    modelo.train()
    return round(total / n, 6) if n else None


def _carregar_rodada_anterior(caminho: Path) -> dict:
    """O estado completo da rodada anterior, ou uma falha dura.

    Falha dura e nao aviso: uma retomada que silenciosamente comeca do zero
    produziria exatamente o numero que este arquivo inteiro existe para nao
    produzir -- uma curva de rodadas que na verdade e uma sequencia de
    sorteios independentes.
    """
    if not caminho.exists():
        raise FileNotFoundError(
            f"estado de retomada ausente: {caminho}; sem ele a rodada seria "
            f"reinicio, e reinicio nao acumula")
    estado = torch.load(caminho, map_location="cpu", weights_only=False)
    faltando = {"modelo", "otimizador", "gerador", "rodada", "normalizacao"
                } - set(estado)
    if faltando:
        raise ValueError(
            f"estado de retomada incompleto, faltando {sorted(faltando)}; "
            f"retomar so o peso e recomecar o otimizador do zero, e o Adam "
            f"leva centenas de passos para reconstruir os momentos")
    return estado


def _deriva(atual: dict, herdada: dict) -> dict:
    """Quanto a normalizacao TERIA mudado se fosse recalculada no pote novo."""
    return {
        "media": [round(a - h, 6)
                  for a, h in zip(atual["media"], herdada["media"])],
        "desvio": [round(a - h, 6)
                   for a, h in zip(atual["desvio"], herdada["desvio"])],
        "por_que_nao_foi_aplicada": (
            "recalcular deslocaria a escala de entrada debaixo de um encoder "
            "ja ajustado a outra; a deriva fica registrada para ficar visivel"),
    }


def treinar(passos_alvo: int = 4000, base: int = 16, lote: int = 8,
            taxa: float = 1e-3, seed: int = 20260903,
            fracao_mascara: float = FRACAO_MASCARA,
            lado_bloco: int = LADO_BLOCO, lado: int = LADO,
            recortes_por_chip: int = 16, threads: int | None = None,
            splits: tuple = ("treino",), epocas_alvo: int | None = None,
            retomar: Path | None = None,
            retidos: int = CHIPS_RETIDOS,
            chips: list[str] | None = None,
            chips_retidos: list[str] | None = None) -> dict:
    """O rotulo nao entra aqui. `splits` fica em `treino` por padrao.

    Validacao, teste e o holdout da Bolivia ficam de fora do pre-treino pela
    mesma razao de sempre: o ajuste fino vai medir neles, e um encoder que ja
    viu aquelas cenas responderia sobre cena conhecida.

    `epocas_alvo` e `retomar` andam juntos: a rodada fecha epoca (unico ponto
    em que o estado e completo) e a proxima continua dali. `passos_alvo`
    continua valendo quando `epocas_alvo` nao e dado, para nao invalidar as
    corridas ja medidas.

    `chips` e `chips_retidos` existem para a autonomia: a promessa da frente e que
    cada chip novo entra sozinho no pre-treino, e para isso quem chama precisa
    poder dizer qual e o pote desta rodada. Os dois vem SEPARADOS de proposito
    -- deixar o corte para `dividir_o_pote` a cada rodada mudaria o conjunto
    retido junto com o pote, e a curva entre rodadas passaria a comparar
    medidas feitas em conjuntos diferentes.
    """
    if threads:
        torch.set_num_threads(threads)
    semear(seed)
    if chips is not None:
        nomes = list(chips)
        nomes_retidos = list(chips_retidos or [])
        if set(nomes) & set(nomes_retidos):
            raise ValueError("chip retido dentro do pote de treino: a medida "
                             "em dado retido deixaria de ser retida")
    else:
        todos = [n for s in splits for n in nomes_do_split(s)]
        nomes, nomes_retidos = dividir_o_pote(todos, retidos)

    inicio_estat = time.time()
    estat_do_pote = estatisticas(nomes)
    segundos_estat = round(time.time() - inicio_estat, 2)

    modelo = MAEdeSAR(canais=2, base=base)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa)
    gerador = torch.Generator().manual_seed(seed)

    anterior, deriva, taxa_herdada = None, None, None
    estat = estat_do_pote
    rodada, passos_herdados, epocas_herdadas = 1, 0, 0
    linhagem: list[str] = []
    if retomar is not None:
        anterior = _carregar_rodada_anterior(Path(retomar))
        modelo.load_state_dict(anterior["modelo"])
        otimizador.load_state_dict(anterior["otimizador"])
        # `load_state_dict` do otimizador restaura TAMBEM os hiperparametros do
        # grupo, e com eles a taxa da rodada anterior. Sem a linha abaixo,
        # `--taxa` e aceito na linha de comando e ignorado em silencio em toda
        # rodada retomada -- e num desenho cujo ponto e ter rodadas
        # encadeadas, "a taxa que eu pedi nao foi a taxa que rodou" e
        # exatamente o tipo de defeito que nao aparece em metrica nenhuma.
        #
        # Os momentos do Adam vem da rodada anterior (e devem vir); so o
        # tamanho do passo passa a ser o desta rodada.
        taxa_herdada = anterior["otimizador"]["param_groups"][0]["lr"]
        for grupo in otimizador.param_groups:
            grupo["lr"] = taxa
        gerador.set_state(anterior["gerador"])
        if "rng_global" in anterior:
            torch.set_rng_state(anterior["rng_global"])
        estat = anterior["normalizacao"]
        deriva = _deriva(estat_do_pote, estat)
        rodada = int(anterior["rodada"]) + 1
        passos_herdados = int(anterior.get("passos_acumulados", 0))
        epocas_herdadas = int(anterior.get("epocas_acumuladas", 0))
        linhagem = list(anterior.get("linhagem", []))
        linhagem.append(anterior["hash_encoder"])

    historico, passos_totais = [], 0
    inicio = time.time()
    epoca = 0

    def _continuar() -> bool:
        if epocas_alvo is not None:
            return epoca < epocas_alvo
        return passos_totais < passos_alvo

    while _continuar():
        epoca += 1
        modelo.train()
        ordem = torch.randperm(len(nomes), generator=gerador).tolist()
        pote, acumulada, passos = [], 0.0, 0
        for idx in ordem:
            x, v = preparar(nomes[idx], estat)
            pote.extend(recortes(x, v, lado, gerador, recortes_por_chip))
            while len(pote) >= lote:
                # sorteia do pote para nao formar lote todo do mesmo chip
                escolhidos = []
                for _ in range(lote):
                    k = int(torch.randint(0, len(pote), (1,),
                                          generator=gerador).item())
                    pote[k], pote[-1] = pote[-1], pote[k]
                    escolhidos.append(pote.pop())
                entrada = torch.stack([e[0] for e in escolhidos])
                valido = torch.stack([e[1] for e in escolhidos])
                mascarada, escondido = mascarar(entrada, fracao_mascara,
                                                lado_bloco, gerador)
                otimizador.zero_grad()
                perda = perda_mascarada(modelo(mascarada), entrada, escondido,
                                        valido)
                if not torch.isfinite(perda) or float(perda.detach()) == 0:
                    continue
                perda.backward()
                otimizador.step()
                acumulada += float(perda.detach())
                passos += 1
                passos_totais += 1
                if passos_totais >= passos_alvo:
                    break
            if passos_totais >= passos_alvo:
                break
        historico.append({
            "epoca": epoca,
            "epoca_acumulada": epocas_herdadas + epoca,
            "perda_treino": round(acumulada / passos, 6) if passos else None,
            # As DUAS medidas ficam gravadas. `perda_retida` e a que decide, e
            # `perda_retida_media_movel` e a que decidia antes -- guardar as
            # duas e o que permite auditar a troca de instrumento sem
            # reexecutar as corridas.
            "perda_retida": perda_de_reconstrucao(
                modelo, nomes_retidos, estat, fracao_mascara, lado_bloco,
                lado, estatistica_do_lote=True),
            "perda_retida_media_movel": perda_de_reconstrucao(
                modelo, nomes_retidos, estat, fracao_mascara, lado_bloco,
                lado, estatistica_do_lote=False),
            "passos": passos,
            "segundos_acumulados": round(time.time() - inicio, 1)})
        print(f"rodada {rodada} epoca {epoca} "
              f"passos={passos_herdados + passos_totais} "
              f"treino={historico[-1]['perda_treino']} "
              f"retida={historico[-1]['perda_retida']}", flush=True)

    hash_encoder = hash_do_estado(modelo.encoder.state_dict())
    manifesto = {
        "tipo": "pretreino_mascarado_sar",
        "etapa": "A8",
        "quando_utc": agora_utc(),
        "commit": commit_atual(),
        "torch": torch.__version__,
        "seed": seed,
        "splits_do_pretreino": list(splits),
        "chips": len(nomes),
        "chips_retidos": len(nomes_retidos),
        "por_que_retidos": (
            "a perda de treino nao separa aprender de decorar; a medida em "
            "chip que o gradiente nunca tocou separa"),
        "recortes_por_chip": recortes_por_chip,
        "recortes_disponiveis": len(nomes) * recortes_por_chip,
        "hiperparametros": {"base": base, "lote": lote, "taxa": taxa,
                            "lado": lado, "fracao_mascara": fracao_mascara,
                            "lado_bloco": lado_bloco,
                            "threads": threads or torch.get_num_threads()},
        "passos_alvo": passos_alvo if epocas_alvo is None else None,
        "epocas_alvo": epocas_alvo,
        "passos_totais": passos_totais,
        "normalizacao": estat,
        "normalizacao_herdada": retomar is not None,
        "modo_da_medida_retida": (
            "estatistica do lote; as medias moveis do BatchNorm sao "
            "reaprendidas no ajuste fino, entao deixar que elas decidam a "
            "promocao seria escolher pelo que vai ser jogado fora"),
        "taxa_herdada": taxa_herdada,
        "taxa_aplicada": taxa,
        "deriva_da_normalizacao": deriva,
        "segundos_estatisticas": segundos_estat,
        "historico": historico,
        "hash_estado": hash_encoder,
        "hash_estado_completo": hash_completo_do_estado(
            modelo.encoder.state_dict()),
        # LINHAGEM -- o que transforma corridas soltas em trajetoria.
        #
        # `rodada` sozinha nao prova nada: qualquer corrida pode se declarar a
        # terceira. `peso_pai` amarra ao estado de que ela saiu, e `linhagem`
        # guarda a cadeia inteira. Duas rodadas que se dizem irmas e apontam
        # pais diferentes sao denunciadas pela comparacao dos hashes, sem
        # reexecutar nada.
        "rodada": rodada,
        "peso_pai": anterior["hash_encoder"] if anterior else None,
        "linhagem": linhagem,
        "passos_acumulados": passos_herdados + passos_totais,
        "epocas_acumuladas": epocas_herdadas + epoca,
        "perda_retida_final": historico[-1]["perda_retida"] if historico
                              else None,
        "usou_rotulo": False,
        "por_que_75": ("He et al. 2022 (arXiv:2111.06377): mascarar proporcao "
                       "alta da entrada e o que torna a tarefa auto-"
                       "supervisionada nao trivial"),
        "nao_e": ("modelo de suscetibilidade nem mascara publicavel: e uma "
                  "inicializacao de pesos, e ela so vale o que provar valer no "
                  "ajuste fino contra a inicializacao aleatoria"),
    }
    # `selar` grava `hash_manifesto` IN PLACE e devolve o proprio dict --
    # atribuir o retorno metia o manifesto dentro dele mesmo, e o JSON
    # quebrava com referencia circular.
    selar(manifesto)
    # DUAS SAIDAS, e a distincao importa.
    #
    # `estado` continua sendo SO o encoder: e o que atravessa para o ajuste
    # fino, e o decoder existiu para a tarefa e e descartado la. Mandar o
    # decoder junto faria o ajuste fino herdar peso de uma rede que nao e a
    # dele.
    #
    # `continuidade` e o que a PROXIMA RODADA precisa, e ai o decoder tem de
    # ir. Um decoder reiniciado a cada rodada faria o encoder gastar as
    # primeiras centenas de passos compensando um leitor cego -- o gradiente
    # que chega nele viria de um decoder aleatorio, e a rodada mediria essa
    # recuperacao em vez de aprendizado novo.
    return {
        "estado": modelo.encoder.state_dict(),
        "manifesto": manifesto,
        "continuidade": {
            "modelo": modelo.state_dict(),
            "otimizador": otimizador.state_dict(),
            "gerador": gerador.get_state(),
            "rng_global": torch.get_rng_state(),
            "normalizacao": estat,
            "rodada": rodada,
            "passos_acumulados": passos_herdados + passos_totais,
            "epocas_acumuladas": epocas_herdadas + epoca,
            "hash_encoder": hash_encoder,
            "linhagem": linhagem,
        },
    }


def gravar(resultado: dict, rotulo: str, saida: Path = SAIDA) -> Path:
    saida.mkdir(parents=True, exist_ok=True)
    torch.save(resultado["estado"], saida / f"encoder_pretreinado_{rotulo}.pt")
    if "continuidade" in resultado:
        torch.save(resultado["continuidade"],
                   saida / f"pretreino_estado_{rotulo}.pt")
    destino = saida / f"encoder_pretreinado_{rotulo}_manifesto.json"
    destino.write_text(
        json.dumps(resultado["manifesto"], indent=2, ensure_ascii=False),
        encoding="utf-8")
    return destino


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passos-alvo", type=int, default=4000)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--lote", type=int, default=8)
    parser.add_argument("--taxa", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--fracao-mascara", type=float,
                        default=FRACAO_MASCARA)
    parser.add_argument("--recortes-por-chip", type=int, default=16)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--epocas-alvo", type=int, default=None,
                        help="fecha a rodada em epoca inteira; exigido para "
                             "que a proxima possa retomar sem divergir")
    parser.add_argument("--retomar", type=Path, default=None,
                        help="pretreino_estado_*.pt da rodada anterior")
    parser.add_argument("--retidos", type=int, default=CHIPS_RETIDOS)
    parser.add_argument("--rotulo", default="fumaca")
    # Gravar e o PADRAO. A forma anterior (`--gravar` opcional) criava
    # artefato que envelhece em silencio: em 22/09/2026 o `casos.json` do
    # produto estava desatualizado desde antes da migracao para parquet e um
    # teste passou contra ele por dias. `--gravar` segue aceito para nao
    # quebrar chamador existente.
    parser.add_argument("--gravar", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--nao-gravar", dest="gravar", action="store_false",
                        help="nao escreve artefato; so imprime o resumo")
    args = parser.parse_args()

    if args.retomar is not None and args.epocas_alvo is None:
        parser.error("--retomar exige --epocas-alvo: a rodada so pode ser "
                     "cortada em fronteira de epoca, e `passos-alvo` para no "
                     "meio dela")

    r = treinar(passos_alvo=args.passos_alvo, base=args.base, lote=args.lote,
                taxa=args.taxa, seed=args.seed,
                fracao_mascara=args.fracao_mascara,
                recortes_por_chip=args.recortes_por_chip,
                threads=args.threads, epocas_alvo=args.epocas_alvo,
                retomar=args.retomar, retidos=args.retidos)
    if args.gravar:
        r["manifesto"]["gravado_em"] = str(gravar(r, args.rotulo))
    print(json.dumps(r["manifesto"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
