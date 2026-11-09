"""Candidata temporal enxuta do fechamento: encoder fisico compartilhado + ramo de observacao temporal.

Linhagem `fechamento_aprendizado_transferencia_v1` (17/10/2026).

  fisica    igual a v1: MLP 6->16->8 sobre atributos de terreno sem rotulo, cabeca linear sobre
            [h, cobertura]. Recebe SO terreno e cobertura: nenhum SAR, forcamento, mascara ou destino.
  temporal  logit P(deteccao SAR no pixel p, passagem t) =
              a(bloco,data)                       intercepto da aquisicao (so no treino; absorve estacao,
                                                  umidade geral, calibracao e todo efeito comum do forcamento)
            + w.h_p                               nivel do lugar
            + (u.h_p) * f72_t + (v.h_p) * fant_t  resposta do lugar ao forcamento e ao antecedente
            + c.i_p + (k.i_p) * f72_t             ramo instrumental: arvore, construido, angulo
                                                  (observabilidade e efeito do solo umido no retroespalhamento)
            O alvo e instrumental (deteccao SAR); o termo h x forcamento e associacao, nao estado fisico.
Modos: `parada` (h.detach() no ramo temporal) e `conjunto`. `fisica_so` treina so a fisica.
Inicializacao por modulo com gerador proprio: fisica de `parada` == `fisica_so` bit a bit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ia"))
import conjunta_modelo as cm  # noqa: E402

INSTRUMENTAIS = ("arvore", "construido", "angulo")
MODOS = ("fisica_so", "parada", "conjunto")


class ModeloTemporal(nn.Module):
    def __init__(self, modo: str, terreno: list[int], n_terreno: int = 6, n_cobertura: int = 2, dim: int = 8,
                 oculto: int = 16, semente: int = 7, forcamento: bool = True, antecedente: bool = True):
        super().__init__()
        if modo not in MODOS:
            raise ValueError(f"modo desconhecido: {modo}")
        self.modo, self.usa_forcamento, self.usa_antecedente = modo, forcamento, antecedente
        self.register_buffer("idx_terreno", torch.tensor(terreno, dtype=torch.long))
        self.pad_terreno = cm.Padronizacao(n_terreno)
        self.pad_cobertura = cm.Padronizacao(n_cobertura)
        self.pad_instr = cm.Padronizacao(len(INSTRUMENTAIS))
        self.pad_forc = cm.Padronizacao(2)
        n = len(terreno)
        self.enc_fis = cm._semeado(semente, 1, lambda: cm.EncoderFisico(n, oculto, dim))
        self.cab_fis = cm._semeado(semente, 2, lambda: nn.Linear(dim + n_cobertura, 1))
        self.cab_lugar = cm._semeado(semente, 4, lambda: nn.Linear(dim, 1, bias=False))
        self.cab_f72 = cm._semeado(semente, 5, lambda: nn.Linear(dim, 1, bias=False))
        self.cab_ante = cm._semeado(semente, 6, lambda: nn.Linear(dim, 1, bias=False))
        self.instr = cm._semeado(semente, 7, lambda: nn.Linear(len(INSTRUMENTAIS), 1, bias=False))
        self.instr_f72 = cm._semeado(semente, 8, lambda: nn.Linear(len(INSTRUMENTAIS), 1, bias=False))

    def representacao(self, terreno):
        return self.enc_fis(self.pad_terreno(terreno.float())[..., self.idx_terreno])

    def fisica(self, terreno, cobertura):
        """Escore estatico de ordem dentro do evento: so terreno e cobertura."""
        h = self.representacao(terreno)
        return self.cab_fis(torch.cat([h, self.pad_cobertura(cobertura.float())], -1)).squeeze(-1)

    def temporal(self, terreno, instrumental, forcamento):
        """Logit da deteccao SAR sem o intercepto da aquisicao (ordem dentro da passagem)."""
        h = self.representacao(terreno)
        if self.modo == "parada":
            h = h.detach()
        f = self.pad_forc(forcamento.float())
        i = self.pad_instr(instrumental.float())
        out = self.cab_lugar(h).squeeze(-1) + self.instr(i).squeeze(-1)
        if self.usa_forcamento:
            out = out + self.cab_f72(h).squeeze(-1) * f[:, 0] + self.instr_f72(i).squeeze(-1) * f[:, 0]
        if self.usa_antecedente:
            out = out + self.cab_ante(h).squeeze(-1) * f[:, 1]
        return out

    def grupos(self):
        fis = list(self.enc_fis.parameters()) + list(self.cab_fis.parameters())
        tmp = [p for m in (self.cab_lugar, self.cab_f72, self.cab_ante, self.instr, self.instr_f72) for p in m.parameters()]
        return fis, tmp

    def parametros_compartilhados(self):
        return [f"enc_fis.{n}" for n, _ in self.enc_fis.named_parameters()] if self.modo == "conjunto" else []
