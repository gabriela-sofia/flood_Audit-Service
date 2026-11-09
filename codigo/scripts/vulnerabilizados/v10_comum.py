"""Linhagem v10 (ciclo contextual pos-v9: cadeia limpa, doadores no Q, ciclo aquisicao-suporte-atualizacao, lacuna): caminhos, escrita, protocolo congelado, registro de etapas e manifesto.

Escreve so em `v10_ciclo_contextual`. Le v1-v9, MT e auditoria como fontes.
Correcao herdada da revisao v8: o manifesto e fechado DEPOIS de registrar a etapa de fechamento, para que o log
de etapas tenha o mesmo hash no manifesto e no disco.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(k, "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum_tv import gravar, ler  # noqa: E402
from v8_comum import AUD, MT, REPO, TER, TV, V3A, V4, V6, V7, V8, agora, sha_hex  # noqa: E402,F401

V9 = TV / "v9_evolucao_abrangente"
V10 = TV / "v10_ciclo_contextual"
COORTES = ["rmpa", "manaus", "curitiba", "recife", "rmr"]
COORTES_B = ["rmpa", "manaus", "curitiba"]


def g10(p, valor):
    return gravar(p, valor, raiz=V10)


def congelar10(p, valor: dict) -> dict:
    p = Path(p)
    if p.exists():
        existente = ler(p)
        for k, v in valor.items():
            if json.loads(json.dumps(existente.get(k), default=str)) != json.loads(json.dumps(v, default=str)):
                raise ValueError(f"protocolo congelado diverge em {k}: use adendo datado")
        return existente
    g10(p, {**valor, "quando_utc": agora()})
    return ler(p)


def etapa(nome: str, **info):
    V10.mkdir(parents=True, exist_ok=True)
    with (V10 / "execucao_etapas.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"quando_utc": agora(), "etapa": nome, **info}, ensure_ascii=False, default=str) + "\n")


def fechar_manifesto(**info) -> dict:
    """Registra a etapa de fechamento e SO DEPOIS calcula os hashes (inclusive do log)."""
    etapa("fechar_manifesto", **info)
    arquivos = {p.relative_to(V10).as_posix(): sha_hex(p) for p in sorted(V10.rglob("*")) if p.is_file() and p.name != "manifesto.json"}
    m = {"quando_utc": agora(), "n": len(arquivos), "bytes": int(sum((V10 / k).stat().st_size for k in arquivos)), "arquivos": arquivos,
         "ordem": "etapa registrada antes do calculo dos hashes"}
    g10(V10 / "manifesto.json", m)
    return m


def auc(y, s):
    from comum_tv import auc_mann_whitney
    return auc_mann_whitney(y, s)


def nucleo_limpo():
    """Qsep_sel_fis (v9 P0): lam escolhido e coeficientes ajustados SO nas 29 unidades de ajuste_fisico."""
    import pickle
    with (V9 / "p0_regularizacao/modelos.pkl").open("rb") as f:
        return pickle.load(f)["Qsep_sel_fis"]


def papeis():
    return ler(V8 / "grafo_particoes.json")["unidades"]
