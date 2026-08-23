"""REV-P v1py — tabelas da fila visual para o TCC.

Gera as tabelas prontas para o texto: elegibilidade do ativo visual e fila de revisão
ampliada. Não cria rótulo, alvo nem referência de campo.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1pu_v1pz_elegibilidade_comum import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, require_no_abs_paths, write_csv, write_doc, write_schema,
)
from revp_v1pg_v1pm_representacao_comum import read_csv

IN_AUDIT = _p("REVP_V1PY_IN_AUDIT", DATASETS / "dino_visual_asset_eligibility_audit_v1pu.csv")
IN_QUEUE = _p("REVP_V1PY_IN_QUEUE", DATASETS / "dino_review_only_execution_queue_expanded_v1pw.csv")
OUT_T_ELIG = _p("REVP_V1PY_OUT_T_ELIG", DATASETS / "dino_tcc_table_visual_asset_eligibility_v1py.csv")
OUT_T_QUEUE = _p("REVP_V1PY_OUT_T_QUEUE", DATASETS / "dino_tcc_table_review_queue_v1py.csv")
SCH_ELIG = _p("REVP_V1PY_SCH_ELIG", SCHEMAS / "dino_tcc_table_visual_asset_eligibility_v1py_schema.csv")
SCH_QUEUE = _p("REVP_V1PY_SCH_QUEUE", SCHEMAS / "dino_tcc_table_review_queue_v1py_schema.csv")
DOC = _p("REVP_V1PY_DOC", DOCS / "revp_v1py_tabelas_fila_visual.md")

T_ELIG_FIELDS = [
    "visual_asset_id", "inferred_patch_id", "inferred_region",
    "asset_visual_type", "eligibility_status", "confidence", "dino_allowed_use",
]
T_QUEUE_FIELDS = [
    "queue_id", "patch_id", "regiao", "visual_type",
    "queue_priority", "queue_reason", "linkage_confidence", "dino_allowed_use",
]

TCC_TEXT = (
    "A fila de execução DINO foi construída como camada de representação visual review-only. "
    "A elegibilidade de um patch para extração de embedding não equivale à confirmação "
    "temporal Sentinel nem à validação de evento observado; ela apenas indica que existe "
    "um artefato visual adequado para gerar representação vetorial sem rótulo."
)


def _project(rows: list[dict[str, str]], fields: list[str]) -> list[dict[str, Any]]:
    return [{f: r.get(f, "") for f in fields} for r in rows]


def run() -> None:
    audit = read_csv(IN_AUDIT)
    queue = read_csv(IN_QUEUE)
    t_elig = _project(audit, T_ELIG_FIELDS)
    t_queue = _project(queue, T_QUEUE_FIELDS)

    for rotulo, rows in (("v1py_elig", t_elig), ("v1py_queue", t_queue)):
        require_no_abs_paths(rows, rotulo)
        assert_no_forbidden_true(rows, rotulo)

    write_csv(OUT_T_ELIG, t_elig, T_ELIG_FIELDS)
    write_csv(OUT_T_QUEUE, t_queue, T_QUEUE_FIELDS)
    write_schema(SCH_ELIG, T_ELIG_FIELDS, "v1py_tcc_visual_asset_eligibility")
    write_schema(SCH_QUEUE, T_QUEUE_FIELDS, "v1py_tcc_review_queue")
    write_doc(DOC, "v1py — DINO Visual Queue TCC Update", [
        "## Objetivo",
        "Tabelas TCC-ready para elegibilidade visual e fila expandida DINO review-only.",
        "## Interpretação metodológica (texto para o TCC)",
        TCC_TEXT,
        "## Guardrails",
        "DINO é representação visual review-only. Elegibilidade ≠ confirmação temporal "
        "≠ validação de evento. Nenhum rotulo, target ou treino criado.",
        f"## Resultado",
        f"Linhas eligibilidade: {len(t_elig)}. Linhas fila: {len(t_queue)}.",
    ])
    print(f"[v1py] elig={len(t_elig)} queue={len(t_queue)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1py dino visual queue tcc update").parse_args()
    run()
