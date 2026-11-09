"""Linhagem v8 (validacao integrada: ponte fisica MT01/MT11 -> coortes brasileiras, poucos rotulos, limites).

Escreve so em `v8_validacao_integrada`. Le v1-v7, MT01/MT03/MT05 e a auditoria como fontes, conferindo hashes.
Threads numericas limitadas antes de importar bibliotecas.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(k, "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum_tv import gravar, ler  # noqa: E402
from v5_comum import REPO, TV, V3A, V4, V5, agora, sha_hex  # noqa: E402,F401

V6 = TV / "v6_evidencia_robusta"
V7 = TV / "v7_aproveitamento_acervo"
V8 = TV / "v8_validacao_integrada"
MT = REPO / "local_runs" / "mecanismos-transferiveis"
AUD = REPO / "local_runs" / "auditoria-utilidade-20261021" / "v1"
TER = REPO / "local_runs" / "ter-01-cadeia-harmonizada"
for p in ("metodo", "geo", "servico", "suscetibilidade", "terreno"):
    sys.path.insert(0, str(REPO / "scripts" / p))

COORTES = {  # coortes brasileiras de desenvolvimento ja examinadas (v1-v7)
    "rmpa": {"tabela": V3A / "rmpa.parquet", "evento": "2024-05-06", "episodio": "RS_2024_05", "processo": "fluvial"},
    "manaus": {"tabela": V3A / "manaus.parquet", "evento": "2021-06-21", "episodio": "AM_2021_06", "processo": "fluvial"},
    "curitiba": {"tabela": V3A / "curitiba.parquet", "evento": "2023-11-05", "episodio": "PR_2023_11", "processo": "fluvial_pluvial"},
}


def g8(p, valor):
    return gravar(p, valor, raiz=V8)


def congelar8(p, valor: dict) -> dict:
    p = Path(p)
    if p.exists():
        existente = ler(p)
        for k, v in valor.items():
            if json.loads(json.dumps(existente.get(k), default=str)) != json.loads(json.dumps(v, default=str)):
                raise ValueError(f"protocolo congelado diverge em {k}: use adendo datado")
        return existente
    g8(p, {**valor, "quando_utc": agora()})
    return ler(p)


def etapa(nome: str, **info):
    """Registro de execucao (append) com horario, para retomadas e tempos."""
    V8.mkdir(parents=True, exist_ok=True)
    with (V8 / "execucao_etapas.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"quando_utc": agora(), "etapa": nome, **info}, ensure_ascii=False, default=str) + "\n")
