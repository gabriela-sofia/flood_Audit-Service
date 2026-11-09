"""Linhagem v6 (evidencia robusta pos-v5): caminhos e escrita imutavel. Nao escreve em v1-v5 nem em
`v6_diagnostico_pos_v5` (auditoria de outra sessao, lida como fonte)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum_tv import gravar, ler  # noqa: E402
from v5_comum import REPO, TV, V3A, V4, V5, agora, sha_hex  # noqa: E402,F401

V6 = TV / "v6_evidencia_robusta"
AUDITORIA_EXTERNA = TV / "v6_diagnostico_pos_v5" / "auditoria_resumos.json"


def g6(p, valor):
    return gravar(p, valor, raiz=V6)


def congelar6(p, valor: dict) -> dict:
    p = Path(p)
    if p.exists():
        existente = ler(p)
        for k, v in valor.items():
            if json.loads(json.dumps(existente.get(k), default=str)) != json.loads(json.dumps(v, default=str)):
                raise ValueError(f"protocolo congelado diverge em {k}: use adendo datado")
        return existente
    g6(p, {**valor, "quando_utc": agora()})
    return ler(p)
RUNS_TV = TV
