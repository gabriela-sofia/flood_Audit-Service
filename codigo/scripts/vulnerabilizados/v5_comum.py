"""Linhagem v5 (execucao centrada em comunidades): caminhos, escrita imutavel e conferencia de entradas.

Nao escreve em v1-v4. Consome o preparo v4 (IDs aninhados) e as coortes/doadores v3, conferindo hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(k, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum_tv import gravar, ler  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TV = REPO / "local_runs" / "transferencia-vulnerabilizados"
V3A = TV / "v3" / "aprendizagem"
V4 = TV / "v4_preparo_centrado"
V5 = TV / "v5_execucao_comunidades"


def agora():
    return datetime.now(timezone.utc).isoformat()


def sha_hex(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def g5(p, valor):
    return gravar(p, valor, raiz=V5)


def congelar5(p, valor: dict) -> dict:
    p = Path(p)
    if p.exists():
        existente = ler(p)
        for k, v in valor.items():
            if json.loads(json.dumps(existente.get(k), default=str)) != json.loads(json.dumps(v, default=str)):
                raise ValueError(f"protocolo congelado diverge em {k}: use adendo datado")
        return existente
    g5(p, {**valor, "quando_utc": agora()})
    return ler(p)


def conferir_v4() -> dict:
    """Confere o manifesto do preparo v4 e as fontes v3 que ele registrou. Falha se algo mudou."""
    man = ler(V4 / "manifesto.json")
    res = {"artefatos": {}, "fontes": {}}
    for nome, h in man["artefatos"].items():
        p = REPO / nome.replace("\\", "/")
        res["artefatos"][Path(nome.replace("\\", "/")).name] = sha_hex(p) == h.replace("sha256:", "")
    for nome, h in man["fontes"].items():
        p = REPO / nome.replace("\\", "/")
        res["fontes"][Path(nome.replace("\\", "/")).name] = sha_hex(p) == h.replace("sha256:", "")
    if not all(res["artefatos"].values()) or not all(res["fontes"].values()):
        raise RuntimeError(f"preparo v4 ou fonte v3 divergente: {res}")
    return res
