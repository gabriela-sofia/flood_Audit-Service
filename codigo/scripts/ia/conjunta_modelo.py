"""Modelo conjunto v1: um encoder fisico, duas cabecas, modos que isolam o compartilhamento.

Linhagem `aprendizado_conjunto_v1` (16/10/2026).

  encoder fisico  MLP por ponto/pixel sobre atributos de terreno calculaveis sem rotulo
                  (`conjunta_base.TERRENO`), padronizados com estatisticas do TREINO.
  cabeca fisica   linear sobre [h, cobertura]; ordem relativa DENTRO do evento. Nao recebe
                  SAR, mascara ou nada posterior ao evento: a assinatura de `fisica` nao aceita.
  cabeca de obs.  U-Net pequena sobre [SAR posterior (2 canais, ramo proprio), mapa h].

Modos (o que muda e SO o caminho do gradiente/dos parametros):
  separado   dois encoders com a mesma inicializacao; a observacao usa o seu. Integracao
             sem transferencia: numericamente igual a ajustar cada tarefa sozinha.
  parada     um encoder; a observacao recebe h.detach(). O encoder so aprende da fisica,
             a observacao usa a representacao fisica sem devolver gradiente.
  conjunto   um encoder; os gradientes das duas tarefas atualizam os mesmos parametros.

Inicializacao por modulo com geradores proprios: com o mesmo `semente`, o encoder e a
cabeca fisica nascem iguais em todos os modos. Por isso `parada` e `separado` devem dar a
MESMA fisica bit a bit -- e e isso que torna `conjunto - parada` o efeito do gradiente da
observacao, e nada mais.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "sar"))
from modelo_unet import UNetPequena  # noqa: E402

MODOS = ("separado", "parada", "conjunto")


def _semeado(semente: int, modulo: int, fabrica):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(semente * 1000 + modulo)
        return fabrica()


class EncoderFisico(nn.Module):
    """MLP pontual; em mapa vira convolucao 1x1. Sem vizinhanca: a conexao entra pelas distancias."""

    def __init__(self, n_entrada: int, oculto: int = 16, dim: int = 8):
        super().__init__()
        self.rede = nn.Sequential(nn.Linear(n_entrada, oculto), nn.Tanh(), nn.Linear(oculto, dim), nn.Tanh())

    def forward(self, x):
        return self.rede(x)

    def mapa(self, x):
        return self.rede(x.movedim(1, -1)).movedim(-1, 1)


class Padronizacao(nn.Module):
    """A regra do `mt01_experimento.Padronizador`: media/desvio/mediana dos pontos de treino
    ignorando ausentes; ausente vira a mediana; desvio nulo vira 1. Ausente sem mediana vira 0."""

    def __init__(self, n: int):
        super().__init__()
        for nome, v in (("media", 0.), ("desvio", 1.), ("mediana", 0.)):
            self.register_buffer(nome, torch.full((n,), v))

    @torch.no_grad()
    def ajustar(self, x: torch.Tensor):
        import pandas as pd
        df = pd.DataFrame(x.double().numpy())
        mu, sd, med = df.mean(), df.std().replace(0, 1).fillna(1), df.median()
        self.media.copy_(torch.tensor(mu.fillna(0).to_numpy()))
        self.desvio.copy_(torch.tensor(sd.to_numpy()))
        self.mediana.copy_(torch.tensor(med.fillna(mu.fillna(0)).to_numpy()))
        return self

    def forward(self, x, eixo=-1):
        shape = [1] * x.ndim
        shape[eixo] = -1
        med, mu, sd = (t.view(shape) for t in (self.mediana, self.media, self.desvio))
        return (torch.where(torch.isnan(x), med, x) - mu) / sd


class ModeloConjunto(nn.Module):
    def __init__(self, modo: str, terreno: list[int], n_terreno_total: int, n_cobertura: int = 2,
                 dim: int = 8, oculto: int = 16, base: int = 4, usa_sar: bool = True, semente: int = 7):
        super().__init__()
        if modo not in MODOS:
            raise ValueError(f"modo desconhecido: {modo}")
        self.modo, self.usa_sar = modo, usa_sar
        self.register_buffer("idx_terreno", torch.tensor(terreno, dtype=torch.long))
        self.pad_terreno = Padronizacao(n_terreno_total)
        self.pad_cobertura = Padronizacao(n_cobertura)
        n = len(terreno)
        self.enc_fis = _semeado(semente, 1, lambda: EncoderFisico(n, oculto, dim))
        self.cab_fis = _semeado(semente, 2, lambda: nn.Linear(dim + n_cobertura, 1))
        self.enc_obs = _semeado(semente, 1, lambda: EncoderFisico(n, oculto, dim)) if modo == "separado" else None
        self.obs = _semeado(semente, 4, lambda: UNetPequena(canais_entrada=(2 if usa_sar else 0) + dim, base=base))

    # ---------------------------------------------------------------- fisica
    def representacao(self, terreno: torch.Tensor) -> torch.Tensor:
        z = self.pad_terreno(terreno.float())[..., self.idx_terreno]
        return self.enc_fis(z)

    def fisica(self, terreno: torch.Tensor, cobertura: torch.Tensor) -> torch.Tensor:
        """Escore de ordem dentro do evento. Entradas: so atributos pre-evento [N, T] e [N, C]."""
        h = self.representacao(terreno)
        c = self.pad_cobertura(cobertura.float())
        return self.cab_fis(torch.cat([h, c], -1)).squeeze(-1)

    # ---------------------------------------------------------------- observacao
    def mapa_h(self, terreno_mapa: torch.Tensor) -> torch.Tensor:
        z = self.pad_terreno(terreno_mapa.float(), eixo=1)[:, self.idx_terreno]
        enc = self.enc_obs if self.modo == "separado" else self.enc_fis
        h = enc.mapa(z)
        return h.detach() if self.modo == "parada" else h

    def observacao(self, sar: torch.Tensor | None, terreno_mapa: torch.Tensor) -> torch.Tensor:
        h = self.mapa_h(terreno_mapa)
        if self.usa_sar:
            if sar is None:
                raise ValueError("braco com SAR exige SAR")
            h = torch.cat([sar.float(), h], 1)
        return self.obs(h)

    # ---------------------------------------------------------------- registro
    def parametros_compartilhados(self) -> list[str]:
        """Nomes dos parametros que recebem gradiente das DUAS tarefas neste modo."""
        if self.modo != "conjunto":
            return []
        return [f"enc_fis.{n}" for n, _ in self.enc_fis.named_parameters()]

    def grupos(self) -> dict:
        fis = list(self.enc_fis.parameters()) + list(self.cab_fis.parameters())
        enc_o = list(self.enc_obs.parameters()) if self.enc_obs is not None else []
        return {"fisica": fis, "encoder_obs": enc_o, "observacao": list(self.obs.parameters())}


def perda_fisica(escore, y, evento, intercepto):
    """Logistica da MT01: intercepto livre por evento; cada evento pesa 1, cada classe metade."""
    eta = escore + intercepto[evento]
    n_ev = intercepto.numel()
    npos = torch.bincount(evento, weights=y, minlength=n_ev)
    nneg = torch.bincount(evento, weights=1 - y, minlength=n_ev)
    w = torch.where(y == 1, .5 / npos.clamp(min=1)[evento], .5 / nneg.clamp(min=1)[evento]) / n_ev
    return (w * nn.functional.binary_cross_entropy_with_logits(eta, y, reduction="none")).sum()


def l2_fisica(modelo: ModeloConjunto, lam: float):
    pesos = [p for n, p in modelo.named_parameters() if n.startswith(("enc_fis.", "cab_fis.", "enc_obs.")) and p.ndim > 1]
    return .5 * lam * sum((p * p).sum() for p in pesos)
