"""B2 -- o encoder de terreno, e por que ele e mascarado.

-------------------------------------------------------------------------
AUTOENCODER SIMPLES APRENDE A COPIAR
-------------------------------------------------------------------------

Um autoencoder que ve a entrada inteira e tem de devolve-la resolve a tarefa
decorando o caminho mais curto entre entrada e saida. Com gargalo apertado ele
aprende alguma compressao -- mas compressao de textura, nao de estrutura, e o
resultado seria um vetor que descreve "quao rugoso e o recorte".

Isso ja existe, custa 25 numeros e chama-se baseline descritivo.

O mascaramento troca a pergunta. Escondendo blocos inteiros, reconstruir exige
saber PARA ONDE a forma continua: se um pedaco do fundo de vale esta oculto, o
unico jeito de recompor a altitude ali e ter aprendido que vale desce, que
encosta liga topo a canal, que acumulacao cresce rio abaixo.

E essa e exatamente a informacao que a rota DINOv2 nao tinha para oferecer: um
encoder optico congelado sabe o que a CENA parece, e agua nao se acumula onde
a cena parece molhada -- acumula onde a forma do terreno a leva.

-------------------------------------------------------------------------
O EMBEDDING E MEDIA E DESVIO, e nao so media
-------------------------------------------------------------------------

O vetor sai de um pooling global sobre o mapa latente, com as duas estatisticas
concatenadas. So a media descreveria o recorte medio e apagaria heterogeneidade
-- e um chip com metade varzea e metade encosta e diferente de um chip
inteiramente intermediario, ainda que as medias coincidam.

-------------------------------------------------------------------------
TAMANHO
-------------------------------------------------------------------------

`base=16` de novo, pela mesma restricao declarada da frente A: nao ha CUDA
nesta maquina. E de novo a restricao real nao e capacidade -- sao 341 chips de
treino e validacao, e um encoder grande decoraria os 341.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _bloco(entrada: int, saida: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(entrada, saida, 3, padding=1, bias=False),
        nn.BatchNorm2d(saida),
        nn.ReLU(inplace=True),
        nn.Conv2d(saida, saida, 3, padding=1, bias=False),
        nn.BatchNorm2d(saida),
        nn.ReLU(inplace=True),
    )


class EncoderDeTerreno(nn.Module):
    """Encoder convolucional de tres niveis sobre as cinco camadas."""

    def __init__(self, canais: int = 5, base: int = 16) -> None:
        super().__init__()
        b = base
        self.n1 = _bloco(canais, b)
        self.n2 = _bloco(b, b * 2)
        self.n3 = _bloco(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)
        self.dimensao_latente = b * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        h = self.n2(self.pool(h))
        return self.n3(self.pool(h))


class DecoderDeTerreno(nn.Module):
    """Reconstroi a pilha na resolucao de entrada."""

    def __init__(self, canais: int = 5, base: int = 16) -> None:
        super().__init__()
        b = base
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d2 = _bloco(b * 2, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d1 = _bloco(b, b)
        self.saida = nn.Conv2d(b, canais, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.d2(self.up2(z))
        h = self.d1(self.up1(h))
        return self.saida(h)


class AutoencoderMascarado(nn.Module):
    """Encoder + decoder. O mascaramento acontece na ENTRADA, no laco de treino.

    Manter o mascaramento fora do modulo e deliberado: o que a rede recebe e
    uma pilha com buracos, e o que ela devolve e uma pilha inteira. Quem sabe
    onde estavam os buracos e a perda, e so ela precisa saber.
    """

    def __init__(self, canais: int = 5, base: int = 16) -> None:
        super().__init__()
        self.encoder = EncoderDeTerreno(canais, base)
        self.decoder = DecoderDeTerreno(canais, base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Media e desvio do mapa latente, concatenados.

        Aceita qualquer tamanho de entrada: o pooling e global, e os chips do
        `ter-01` nao tem todos a mesma forma.
        """
        z = self.encoder(x)
        media = z.mean(dim=(2, 3))
        desvio = z.std(dim=(2, 3))
        return torch.cat([media, desvio], dim=1)

    @property
    def dimensao_embedding(self) -> int:
        return self.encoder.dimensao_latente * 2


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def mascarar(x: torch.Tensor, fracao: float, lado_bloco: int,
             gerador: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Esconde blocos quadrados. Devolve `(entrada_mascarada, onde_escondeu)`.

    BLOCOS, e nao pixels soltos: mascarar pixels isolados e uma tarefa que a
    interpolacao local resolve, e interpolar nao exige entender a forma. O
    bloco obriga a rede a olhar o contexto largo.

    O valor escondido vira zero, que apos a normalizacao por camada e a media
    do treino -- o ponto mais neutro disponivel. Mesma decisao da frente A com
    o pixel sem leitura de SAR.
    """
    b, _, h, w = x.shape
    mascara = torch.zeros((b, 1, h, w), dtype=torch.bool)
    blocos_h = max(h // lado_bloco, 1)
    blocos_w = max(w // lado_bloco, 1)
    total = blocos_h * blocos_w
    n_esconder = max(int(round(total * fracao)), 1)
    for i in range(b):
        escolhidos = torch.randperm(total, generator=gerador)[:n_esconder]
        for idx in escolhidos.tolist():
            bi, bj = divmod(idx, blocos_w)
            mascara[i, 0,
                    bi * lado_bloco:(bi + 1) * lado_bloco,
                    bj * lado_bloco:(bj + 1) * lado_bloco] = True
    return x.masked_fill(mascara, 0.0), mascara
