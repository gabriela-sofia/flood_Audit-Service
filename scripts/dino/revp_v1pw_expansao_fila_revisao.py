"""REV-P v1pw — expansão da fila DINO somente-revisão.

Monta a fila ampliada a partir do registro de vínculo do v1pv, com prioridade declarada:
primeiro os patches do Protocolo C (v1oz), depois as referências de TIF Sentinel, depois
o restante que passou na elegibilidade. Não executa embedding aqui, e não cria rótulo,
alvo ou referência de campo.
"""
from __future__ import annotations

import argparse
import os
from typing import Any

from revp_v1pu_v1pz_elegibilidade_comum import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, normalize_region, path_hash,
    read_v1oz_queue, require_no_abs_paths, write_csv, write_doc, write_schema,
)
from revp_v1pg_v1pm_representacao_comum import read_csv

IN_LINKAGE = _p("REVP_V1PW_IN_LINKAGE", DATASETS / "dino_patch_visual_linkage_registry_v1pv.csv")
OUT_QUEUE = _p("REVP_V1PW_OUT_QUEUE", DATASETS / "dino_review_only_execution_queue_expanded_v1pw.csv")
OUT_SUM = _p("REVP_V1PW_OUT_SUM", DATASETS / "dino_review_only_execution_queue_expanded_summary_v1pw.csv")
SCH_QUEUE = _p("REVP_V1PW_SCH_QUEUE", SCHEMAS / "dino_review_only_execution_queue_expanded_v1pw_schema.csv")
SCH_SUM = _p("REVP_V1PW_SCH_SUM", SCHEMAS / "dino_review_only_execution_queue_expanded_summary_v1pw_schema.csv")
DOC = _p("REVP_V1PW_DOC", DOCS / "revp_v1pw_expansao_fila_revisao.md")

MAX_QUEUE = int(os.environ.get("REVP_DINO_MAX_QUEUE", "100"))
INCLUDE_MANUAL = os.environ.get("REVP_DINO_INCLUDE_MANUAL_CHECK", "false").lower() == "true"

QUEUE_FIELDS = [
    "queue_id", "visual_asset_id", "patch_id", "alias", "regiao",
    "visual_type", "queue_priority", "queue_reason", "linkage_confidence",
    "manual_check_required", "dino_allowed_use", "can_create_label",
    "can_train_model", "target_created", "execution_status", "blocked_reason", "notas",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def build_queue(linkage: list[dict[str, str]]) -> list[dict[str, Any]]:
    oz_patches = {r.get("patch_id", "").strip().upper() for r in read_v1oz_queue()}
    queue: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(r: dict[str, str], priority: int, reason: str) -> None:
        if len(queue) >= MAX_QUEUE:
            return
        k = r.get("visual_asset_id", "") or path_hash(r.get("patch_id", "") + r.get("visual_type", ""))
        if k in seen:
            return
        seen.add(k)
        queue.append({
            "queue_id": f"V1PW_Q_{len(queue)+1:05d}",
            "visual_asset_id": r.get("visual_asset_id", ""),
            "patch_id": r.get("patch_id", ""),
            "alias": r.get("alias", ""),
            "regiao": normalize_region(r.get("regiao", "")),
            "visual_type": r.get("visual_type", ""),
            "queue_priority": str(priority),
            "queue_reason": reason,
            "linkage_confidence": r.get("linkage_confidence", ""),
            "manual_check_required": r.get("requires_manual_check", "false"),
            "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION",
            "can_create_label": "false",
            "can_train_model": "false",
            "target_created": "false",
            "execution_status": "PENDING",
            "blocked_reason": "",
            "notas": "",
        })

    eligible = [r for r in linkage if r.get("eligible_for_dino_review") == "true"]
    manual = [r for r in linkage if r.get("requires_manual_check") == "true"]

    # Priority 1: Protocol C DINO review queue patches
    for r in eligible:
        if r.get("patch_id", "").strip().upper() in oz_patches:
            _add(r, 1, "protocol_c_dino_review_queue")

    # Priority 2: Sentinel TIF references (highest quality visual source)
    for r in eligible:
        if r.get("visual_type") in ("SENTINEL_TIF_REFERENCE",):
            _add(r, 2, "sentinel_tif_reference")

    # Priority 3: Patch previews and renders
    for r in eligible:
        if r.get("visual_type") in ("SENTINEL_PATCH_PREVIEW", "SENTINEL_TECHNICAL_RENDER"):
            _add(r, 3, "sentinel_preview_or_render")

    # Priority 4: Remaining eligible
    for r in eligible:
        _add(r, 4, "eligible_review_only")

    # Priority 5: Manual check candidates (opt-in)
    if INCLUDE_MANUAL:
        for r in manual:
            _add(r, 5, "manual_check_candidate")

    return queue


def run() -> None:
    linkage = read_csv(IN_LINKAGE)
    rows = build_queue(linkage)
    require_no_abs_paths(rows, "v1pw_queue")
    assert_no_forbidden_true(rows, "v1pw_queue")
    p1 = sum(1 for r in rows if r["queue_priority"] == "1")
    p2 = sum(1 for r in rows if r["queue_priority"] == "2")
    summary = [
        {"stat_key": "queue_total", "stat_value": str(len(rows))},
        {"stat_key": "protocol_c_priority_items", "stat_value": str(p1)},
        {"stat_key": "sentinel_tif_items", "stat_value": str(p2)},
        {"stat_key": "max_queue_limit", "stat_value": str(MAX_QUEUE)},
        {"stat_key": "include_manual_check", "stat_value": str(INCLUDE_MANUAL).lower()},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
    ]
    write_csv(OUT_QUEUE, rows, QUEUE_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_QUEUE, QUEUE_FIELDS, "v1pw_dino_review_only_queue_expanded")
    write_schema(SCH_SUM, SUM_FIELDS, "v1pw_dino_review_only_queue_expanded_summary")
    write_doc(DOC, "v1pw — DINO Review-Only Queue Expansion", [
        "## Objetivo",
        "Gerar fila DINO expandida a partir de v1pv, priorizando Protocol C patches "
        "e referências Sentinel TIF. Não executa embedding. Não cria rotulo.",
        "## Scene date",
        "A elegibilidade para fila DINO NÃO requer scene_date confirmada. A extração "
        "de embedding é uma representação visual review-only independente de adjudicação "
        "temporal.",
        "## Guardrails",
        "can_create_label, can_train_model e target_created sempre false.",
        f"## Resultado",
        f"Itens na fila: {len(rows)}. Protocol C priority: {p1}. "
        f"Sentinel TIF priority: {p2}.",
    ])
    print(f"[v1pw] queue={len(rows)} p1={p1} p2={p2}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pw dino review-only queue expansion").parse_args()
    run()
