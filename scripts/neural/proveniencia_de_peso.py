"""O que a frente A deixou de reutilizavel: proveniencia de peso.

-------------------------------------------------------------------------
POR QUE ISTO VIROU MODULO
-------------------------------------------------------------------------

A frente A precisou responder "este peso mudou?" sem reexecutar o treino, e a
resposta certa levou duas tentativas. A primeira hasheava o manifesto inteiro,
e dois treinos identicos com a mesma seed davam manifestos diferentes -- a
diferenca era a maquina estar mais ou menos ocupada, porque `segundos` estavam
dentro do hash.

Extrair isso agora, com dois usuarios em vista, evita a terceira copia. Nao e
abstracao antecipada: e a regra que ja errou uma vez, escrita num lugar so.

-------------------------------------------------------------------------
A REGRA
-------------------------------------------------------------------------

O hash cobre O QUE O PESO E, nunca QUANDO ou QUANTO DEMOROU. Instante de
registro e tempo de parede mudam a cada rodada, e um hash que muda sempre para
de responder a unica pergunta que existe para responder.

Mesma decisao do `proveniencia.py` sobre `quando_utc`; aqui ela so ganha os
campos de duracao, que a frente A descobriu do jeito difícil.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from proveniencia import sha256_json, sha256_texto  # noqa: E402

# Campos que descrevem QUANDO e QUANTO DEMOROU, e nao O QUE E.
CAMPOS_NAO_CONTEUDO = ("quando_utc", "segundos_estatisticas", "hash_manifesto",
                       "segundos_totais", "gravado_em")
CAMPOS_NAO_CONTEUDO_HISTORICO = ("segundos_acumulados", "segundos")


def commit_atual(repo: Path = REPO) -> str:
    """O commit que produziu o peso. `desconhecido` e melhor que ausente."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "desconhecido"


def semear(seed: int) -> None:
    """Determinismo declarado. Sem isso o checkpoint nao e evidencia de nada."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)


def hash_do_estado(estado: dict) -> str:
    """Impressao digital do peso, estavel entre execucoes.

    Usa nome, forma e os primeiros valores de cada tensor em vez do tensor
    inteiro: e o suficiente para distinguir dois treinos e barato o bastante
    para rodar em todo teste. Nao e criptografico sobre o peso completo, e a
    docstring diz isso em vez de deixar a duvida.
    """
    import torch

    return sha256_texto("|".join(
        f"{k}:{torch.as_tensor(v).flatten()[:8].tolist()}:{tuple(v.shape)}"
        for k, v in sorted(estado.items())))


def conteudo_para_hash(manifesto: dict) -> dict:
    """O manifesto sem o que muda a cada rodada."""
    limpo = {k: v for k, v in manifesto.items() if k not in CAMPOS_NAO_CONTEUDO}
    if "historico" in manifesto:
        limpo["historico"] = [
            {k: v for k, v in passo.items()
             if k not in CAMPOS_NAO_CONTEUDO_HISTORICO}
            for passo in manifesto["historico"]
        ]
    return limpo


def selar(manifesto: dict) -> dict:
    """Acrescenta `hash_manifesto` ao manifesto, in place, e devolve ele."""
    manifesto["hash_manifesto"] = sha256_json(conteudo_para_hash(manifesto))
    return manifesto
