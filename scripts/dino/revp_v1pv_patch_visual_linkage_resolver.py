"""REV-P v1pv — resolve o vínculo entre ativo visual e patch.

Liga cada referência de ativo visual que saiu da auditoria do v1pu ao seu
patch_id/alias/região. Não exige scene_date nem destravamento temporal, trabalha só com
metadado, e não cria rótulo, alvo ou referência de campo.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1pu_v1pz_visual_eligibility_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, classify_visual_type,
    normalize_region, path_hash, require_no_abs_paths,
    write_csv, write_doc, write_schema,
)
from revp_v1pg_v1pm_dino_representation_common import read_csv

IN_AUDIT = _p("REVP_V1PV_IN_AUDIT", DATASETS / "dino_visual_asset_eligibility_audit_v1pu.csv")
OUT_REGISTRY = _p("REVP_V1PV_OUT_REGISTRY", DATASETS / "dino_patch_visual_linkage_registry_v1pv.csv")
OUT_SUM = _p("REVP_V1PV_OUT_SUM", DATASETS / "dino_patch_visual_linkage_summary_v1pv.csv")
SCH_REG = _p("REVP_V1PV_SCH_REG", SCHEMAS / "dino_patch_visual_linkage_registry_v1pv_schema.csv")
SCH_SUM = _p("REVP_V1PV_SCH_SUM", SCHEMAS / "dino_patch_visual_linkage_summary_v1pv_schema.csv")
DOC = _p("REVP_V1PV_DOC", DOCS / "revp_v1pv_patch_visual_linkage_resolver.md")

REG_FIELDS = [
    "linkage_id", "visual_asset_id", "patch_id", "alias", "region",
    "linkage_basis", "linkage_confidence", "visual_type", "eligible_for_dino_review",
    "requires_manual_check", "dino_allowed_use", "can_create_label",
    "can_train_model", "target_created", "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def build_registry(audit: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(audit, 1):
        pid = r.get("inferred_patch_id", "UNKNOWN_PATCH").strip().upper()
        alias = r.get("inferred_alias", pid)
        region = normalize_region(r.get("inferred_region", ""))
        elig = r.get("eligibility_status", "")
        conf = r.get("confidence", "LOW")
        vtype = classify_visual_type(r.get("relative_path", ""), r.get("asset_visual_type", ""))
        notes_src = r.get("notes", "")
        basis = "v1fu_manifest" if "source=v1fu" in notes_src else (
            "v1fm_designation" if "source=v1fm" in notes_src else "v1pn_inventory"
        )
        eligible = elig == "DINO_ELIGIBLE_REVIEW_ONLY"
        manual = elig == "DINO_REVIEW_CANDIDATE_NEEDS_MANUAL_CHECK"
        allowed = "REVIEW_ONLY_REPRESENTATION" if eligible or manual else "BLOCKED_INVALID_VECTOR"
        rows.append({
            "linkage_id": f"V1PV_LNK_{i:05d}",
            "visual_asset_id": r.get("visual_asset_id", ""),
            "patch_id": pid,
            "alias": alias,
            "region": region,
            "linkage_basis": basis,
            "linkage_confidence": conf,
            "visual_type": r.get("asset_visual_type", vtype),
            "eligible_for_dino_review": str(eligible).lower(),
            "requires_manual_check": str(manual).lower(),
            "dino_allowed_use": allowed,
            "can_create_label": "false",
            "can_train_model": "false",
            "target_created": "false",
            "blocked_reason": r.get("blocked_reason", ""),
            "notes": "",
        })
    return rows


def run() -> None:
    audit = read_csv(IN_AUDIT)
    rows = build_registry(audit)
    require_no_abs_paths(rows, "v1pv_registry")
    assert_no_forbidden_true(rows, "v1pv_registry")
    eligible = sum(1 for r in rows if r["eligible_for_dino_review"] == "true")
    manual = sum(1 for r in rows if r["requires_manual_check"] == "true")
    regions: dict[str, int] = {}
    for r in rows:
        if r["eligible_for_dino_review"] == "true":
            regions[r["region"]] = regions.get(r["region"], 0) + 1
    summary = [
        {"stat_key": "linkage_rows_total", "stat_value": str(len(rows))},
        {"stat_key": "eligible_for_dino_review", "stat_value": str(eligible)},
        {"stat_key": "requires_manual_check", "stat_value": str(manual)},
        {"stat_key": "linked_patches_eligible", "stat_value": str(len({r["patch_id"] for r in rows if r["eligible_for_dino_review"] == "true"}))},
        {"stat_key": "regions_eligible", "stat_value": str(regions)},
        {"stat_key": "labels_created", "stat_value": "0"},
    ]
    write_csv(OUT_REGISTRY, rows, REG_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_REG, REG_FIELDS, "v1pv_patch_visual_linkage_registry")
    write_schema(SCH_SUM, SUM_FIELDS, "v1pv_patch_visual_linkage_summary")
    write_doc(DOC, "v1pv — Patch Visual Linkage Resolver", [
        "## Objetivo",
        "Ligar assets visuais a patch_id/alias/region via manifests. "
        "Não exige scene_date nem temporal unlock.",
        "## Guardrails",
        "can_create_label, can_train_model e target_created sempre false.",
        f"## Resultado",
        f"Linkages: {len(rows)}. Elegíveis review-only: {eligible}. "
        f"Manual check: {manual}.",
    ])
    print(f"[v1pv] rows={len(rows)} eligible={eligible}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pv patch visual linkage resolver").parse_args()
    run()
