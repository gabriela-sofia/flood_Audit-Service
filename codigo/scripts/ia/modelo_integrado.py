"""Uma arquitetura, um estado de pesos: relações físicas e percepção SAR-terreno.

As duas saídas têm alvos diferentes. Não usar radar pós-evento para calcular
suscetibilidade antes do evento. A rede de percepção já aprende conjuntamente
SAR e terreno; o ajuste físico é herdado do MT03, não de pseudo-rótulos da rede.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
for pasta in ("sar", "neural", "servico"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from modelo_unet import UNetPequena

FEATURES = ("f_hand", "f_elev", "f_sca", "f_declive")
VERSAO = "revp_ia_integrada_v1"


class IAIntegrada(nn.Module):
    """Estado único, duas tarefas explícitas, explicação física aditiva exata."""

    def __init__(self, base=4):
        super().__init__()
        self.percepcao = UNetPequena(canais_entrada=7, base=base)
        self.relacoes = nn.Linear(len(FEATURES), 1, bias=False, dtype=torch.float64)
        for nome, valor in (("media", 0.), ("desvio", 1.), ("mediana", 0.)):
            self.register_buffer(nome, torch.full((len(FEATURES),), valor, dtype=torch.float64))

    def forward(self, *, atributos=None, cena=None):
        if atributos is None and cena is None:
            raise ValueError("informe atributos físicos ou cena SAR-terreno")
        saida = {}
        if atributos is not None:
            if atributos.ndim != 2 or atributos.shape[1] != len(FEATURES):
                raise ValueError("vetor físico deve ter quatro atributos MT03")
            x = atributos.to(dtype=torch.float64)
            if torch.isinf(x).any():
                raise ValueError("infinito não é ausência de atributo")
            faltantes = torch.isnan(x)
            z = (torch.where(faltantes, self.mediana, x) - self.media) / self.desvio
            c = z * self.relacoes.weight[0]
            saida.update(ordem_relativa=c.sum(1), contribuicoes=c, atributos_ausentes=faltantes)
        if cena is not None:
            if cena.ndim != 4 or cena.shape[1] != 7 or min(cena.shape[2:]) < 8:
                raise ValueError("cena exige sete canais SAR-terreno e dimensões >= 8")
            if any(s % 8 for s in cena.shape[2:]) or not torch.isfinite(cena).all():
                raise ValueError("cena deve ser finita e múltipla de oito")
            saida["logits_agua_observada"] = self.percepcao(cena.float())
        return saida
