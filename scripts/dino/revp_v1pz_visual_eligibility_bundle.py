"""REV-P v1pz — pacote de fechamento da elegibilidade visual.

Consolida v1pu a v1py em manifesto, resumo científico e documento final. O estado sai
DINO_VISUAL_QUEUE_READY_REVIEW_ONLY quando a fila tem pelo menos uma linha; fila vazia
sai FAIL_CLOSED, porque fila vazia não é fila pronta.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1pu_v1pz_visual_eligibility_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, require_no_abs_paths, write_csv, write_doc, write_schema,
)
from revp_v1pg_v1pm_dino_representation_common import read_csv

OUT_MANIFEST = _p("REVP_V1PZ_OUT_MANIFEST", DATASETS / "dino_visual_eligibility_bundle_manifest_v1pz.csv")
OUT_SUM = _p("REVP_V1PZ_OUT_SUM", DATASETS / "dino_visual_eligibility_scientific_summary_v1pz.csv")
SCH_MAN = _p("REVP_V1PZ_SCH_MAN", SCHEMAS / "dino_visual_eligibility_bundle_manifest_v1pz_schema.csv")
SCH_SUM = _p("REVP_V1PZ_SCH_SUM", SCHEMAS / "dino_visual_eligibility_scientific_summary_v1pz_schema.csv")
DOC = _p("REVP_V1PZ_DOC", DOCS / "revp_v1pz_visual_eligibility_bundle.md")

MANIFEST_FIELDS = ["artifact_id", "stage", "filename", "rows", "header_present", "role"]
SUM_FIELDS = ["summary_id", "metric", "value", "interpretation", "methodological_status", "writing_use"]

ARTIFACTS = [
    ("v1pu", "dino_visual_asset_eligibility_audit_v1pu.csv", "eligibility_audit"),
    ("v1pu", "dino_visual_asset_eligibility_summary_v1pu.csv", "eligibility_summary"),
    ("v1pv", "dino_patch_visual_linkage_registry_v1pv.csv", "linkage_registry"),
    ("v1pv", "dino_patch_visual_linkage_summary_v1pv.csv", "linkage_summary"),
    ("v1pw", "dino_review_only_execution_queue_expanded_v1pw.csv", "expanded_queue"),
    ("v1pw", "dino_review_only_execution_queue_expanded_summary_v1pw.csv", "expanded_queue_summary"),
    ("v1px", "dino_queue_leakage_audit_v1px.csv", "leakage_audit"),
    ("v1px", "dino_queue_leakage_summary_v1px.csv", "leakage_summary"),
    ("v1py", "dino_tcc_table_visual_asset_eligibility_v1py.csv", "tcc_eligibility_table"),
    ("v1py", "dino_tcc_table_review_queue_v1py.csv", "tcc_queue_table"),
]


def _stat(fname: str, key: str) -> str:
    for r in read_csv(DATASETS / fname):
        if r.get("stat_key") == key:
            return r.get("stat_value", "0")
    return "0"


def _count(fname: str) -> str:
    p = DATASETS / fname
    if not p.exists():
        return "MISSING"
    return str(len(read_csv(p)))


def _header(fname: str) -> bool:
    from revp_v1pg_v1pm_dino_representation_common import read_csv_header
    return bool(read_csv_header(DATASETS / fname))


def build_manifest() -> list[dict[str, Any]]:
    return [{
        "artifact_id": f"V1PZ_ART_{i:03d}",
        "stage": stage, "filename": fname,
        "rows": _count(fname), "header_present": str(_header(fname)).lower(), "role": role,
    } for i, (stage, fname, role) in enumerate(ARTIFACTS, 1)]


def build_summary() -> tuple[list[dict[str, Any]], str]:
    audited = _stat("dino_visual_asset_eligibility_summary_v1pu.csv", "visual_assets_audited")
    eligible = _stat("dino_visual_asset_eligibility_summary_v1pu.csv", "dino_eligible_review_only")
    manual = _stat("dino_visual_asset_eligibility_summary_v1pu.csv", "manual_check_candidates")
    blocked = _stat("dino_visual_asset_eligibility_summary_v1pu.csv", "blocked_assets")
    linked = _stat("dino_patch_visual_linkage_summary_v1pv.csv", "linked_patches_eligible")
    queue = _stat("dino_review_only_execution_queue_expanded_summary_v1pw.csv", "queue_total")
    leakage = _stat("dino_queue_leakage_summary_v1px.csv", "leakage_status")

    final = "DINO_VISUAL_QUEUE_READY_REVIEW_ONLY" if int(queue or "0") > 0 else "DINO_VISUAL_QUEUE_EMPTY_FAIL_CLOSED"

    def s(i: int, m: str, v: str, interp: str, ms: str = "RESULTADO_FINAL",
          use: str = "resultado_negativo_auditavel") -> dict[str, Any]:
        return {"summary_id": f"V1PZ_S{i:03d}", "metric": m, "value": v,
                "interpretation": interp, "methodological_status": ms, "writing_use": use}

    rows = [
        s(1, "visual_assets_audited", audited, "Assets visuais de manifests auditados", "AUDITAVEL", "metodologia_auditoria"),
        s(2, "eligible_review_only_assets", eligible, "Assets elegíveis para DINO review-only (sem scene_date)"),
        s(3, "manual_check_candidates", manual, "Candidatos que requerem verificação manual"),
        s(4, "blocked_assets", blocked, "Assets bloqueados (fixture, não-patch, sem ID)"),
        s(5, "linked_patches_eligible", linked, "Patches únicos com linkage elegível", "AUDITAVEL", "metodologia_auditoria"),
        s(6, "expanded_queue_rows", queue, "Itens na fila de execução expandida"),
        s(7, "leakage_audit_status", leakage, "Status do audit de guardrails (leakage)"),
        s(8, "labels_created", "0", "Labels operacionais criadas — 0 por design"),
        s(9, "targets_created", "0", "Targets de treinamento criados — 0 por design"),
        s(10, "final_status", final, "Status final da camada de elegibilidade visual DINO", "RESULTADO_FINAL", "conclusao_auditavel"),
    ]
    return rows, final


def run() -> None:
    manifest = build_manifest()
    summary, final = build_summary()
    for label, rows in (("v1pz_manifest", manifest), ("v1pz_summary", summary)):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)
    write_csv(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_MAN, MANIFEST_FIELDS, "v1pz_visual_eligibility_bundle_manifest")
    write_schema(SCH_SUM, SUM_FIELDS, "v1pz_visual_eligibility_scientific_summary")
    queue_n = _stat("dino_review_only_execution_queue_expanded_summary_v1pw.csv", "queue_total")
    write_doc(DOC, "v1pz — Visual Eligibility Bundle", [
        "## Objetivo",
        "Consolidar v1pu-v1py em manifest, summary científico e doc final.",
        "## Princípio metodológico",
        "Elegibilidade visual para DINO não requer scene_date confirmada nem temporal "
        "unlock. A representação vetorial é review-only e não equivale a validação de "
        "evento, ground truth ou rótulo supervisionado.",
        f"## Status final",
        f"**{final}**. Fila expandida: {queue_n} itens.",
    ])
    print(f"[v1pz] final={final} queue={queue_n}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pz visual eligibility bundle").parse_args()
    run()
