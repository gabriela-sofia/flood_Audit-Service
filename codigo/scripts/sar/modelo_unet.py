"""A4 -- a rede da frente A: U-Net pequena para agua em SAR.

-------------------------------------------------------------------------
PEQUENA POR RESTRICAO DECLARADA, e nao por preguica
-------------------------------------------------------------------------

O ambiente e `torch+cpu`, sem CUDA -- a maquina tem grafico Intel integrado, e
CUDA e exclusivo da NVIDIA. Isso nao e um detalhe de conveniencia: ele define o
orcamento. Uma U-Net classica (31M parametros) treinaria por dias aqui.

`base=16` da uma rede na casa de 1-2M parametros, que cabe no orcamento e ainda
tem capacidade de sobra para 252 chips. A restricao real da frente A nao e
capacidade do modelo -- e quantidade de dado rotulado.

-------------------------------------------------------------------------
DUAS CLASSES, e nao uma saida sigmoide
-------------------------------------------------------------------------

A saida tem 2 canais (nao-agua, agua) e a perda e `CrossEntropyLoss` com
`ignore_index`. Poderia ser sigmoide com mascara manual, e seria equivalente na
matematica -- mas nao na chance de errar: `ignore_index` e a forma que o
proprio framework oferece para dizer "este pixel nao existe para a perda", e
usar o mecanismo padrao e melhor que reimplementar a mascara em cada laco.

O `-1` do `LabelHand` entra ali diretamente. Nenhum passo intermediario
transforma `-1` em `0`, que e a regra da etapa A1.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _bloco(entrada: int, saida: int) -> nn.Sequential:
    """Conv-BN-ReLU duas vezes. `bias=False` porque o BatchNorm ja desloca."""
    return nn.Sequential(
        nn.Conv2d(entrada, saida, 3, padding=1, bias=False),
        nn.BatchNorm2d(saida),
        nn.ReLU(inplace=True),
        nn.Conv2d(saida, saida, 3, padding=1, bias=False),
        nn.BatchNorm2d(saida),
        nn.ReLU(inplace=True),
    )


class UNetPequena(nn.Module):
    """U-Net de tres niveis para entrada VV/VH.

    Tres niveis e nao quatro: com 512x512 de entrada, o quarto nivel chega a
    32x32 e o campo receptivo ja cobre folgado a escala de uma lamina d'agua
    urbana. Profundidade a mais custaria tempo de CPU sem responder pergunta
    nova.
    """

    def __init__(self, canais_entrada: int = 2, classes: int = 2,
                 base: int = 16) -> None:
        super().__init__()
        b = base
        self.enc1 = _bloco(canais_entrada, b)
        self.enc2 = _bloco(b, b * 2)
        self.enc3 = _bloco(b * 2, b * 4)
        self.fundo = _bloco(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = _bloco(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = _bloco(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = _bloco(b * 2, b)
        self.saida = nn.Conv2d(b, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        f = self.fundo(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(f), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.saida(d1)


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
