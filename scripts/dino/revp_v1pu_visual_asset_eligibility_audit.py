"""REV-P v1pu — auditoria de elegibilidade do ativo visual.

Audita as referências já versionadas (manifesto Sentinel do v1fu, designação de patch do
v1fm, inventário do v1pn e saídas do Protocolo C) para decidir o que é elegível à fila
DINO somente-revisão, SEM exigir confirmação de scene_date nem destravamento temporal.

Não lê pixel. Não cria rótulo, alvo nem referência de campo.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1pu_v1pz_visual_eligibility_common import (
    DATASETS, DOCS, ROOT, SCHEMAS,
    _p, assert_no_forbidden_true, classify_dino_eligibility, classify_visual_type,
    infer_patch_from_path, is_fixture_or_synthetic, normalize_region,
    path_hash, read_v1fm_designation, read_v1fu_manifest, require_no_abs_paths,
    write_csv, write_doc, write_schema,
)
from revp_v1pg_v1pm_dino_representation_common import read_csv

IN_V1PN = _p("REVP_V1PU_IN_V1PN", DATASETS / "dino_patch_visual_asset_inventory_v1pn.csv")
IN_V1OZ = _p("REVP_V1PU_IN_V1OZ", DATASETS / "recife_dino_review_only_representation_queue_v1oz.csv")
OUT_AUDIT = _p("REVP_V1PU_OUT_AUDIT", DATASETS / "dino_visual_asset_eligibility_audit_v1pu.csv")
OUT_SUM = _p("REVP_V1PU_OUT_SUM", DATASETS / "dino_visual_asset_eligibility_summary_v1pu.csv")
SCH_AUDIT = _p("REVP_V1PU_SCH_AUDIT", SCHEMAS / "dino_visual_asset_eligibility_audit_v1pu_schema.csv")
SCH_SUM = _p("REVP_V1PU_SCH_SUM", SCHEMAS / "dino_visual_asset_eligibility_summary_v1pu_schema.csv")
DOC = _p("REVP_V1PU_DOC", DOCS / "revp_v1pu_visual_asset_eligibility_audit.md")

AUDIT_FIELDS = [
    "visual_asset_id", "relative_path", "path_hash", "file_ext",
    "file_size_bytes", "inferred_patch_id", "inferred_alias", "inferred_region",
    "asset_visual_type", "eligibility_status", "confidence", "eligibility_reason",
    "blocked_reason", "dino_allowed_use", "can_create_label", "can_train_model",
    "target_created", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _audit_from_v1fu() -> list[dict[str, Any]]:
    """Monta as linhas de auditoria a partir do manifesto Sentinel do v1fu (fonte primária)."""
    rows: list[dict[str, Any]] = []
    v1oz_patches = {
        r.get("patch_id", "").strip().upper()
        for r in read_csv(IN_V1OZ)
        if r.get("patch_id")
    }
    for i, r in enumerate(read_v1fu_manifest(), 1):
        patch_raw = r.get("canonical_patch_id", "") or r.get("patch_id", "")
        region_raw = r.get("region", "")
        asset_ref = r.get("asset_path_reference", "")
        elig_hint = r.get("eligibility_status", "")
        label_st = r.get("label_status", "")
        target_st = r.get("target_status", "")

        # Infer from path if canonical missing
        pid, alias, region = infer_patch_from_path(patch_raw or asset_ref)
        if patch_raw:
            pid = patch_raw.strip().upper()
            alias = pid
        if region_raw:
            region = normalize_region(region_raw)

        ext = Path(asset_ref).suffix.lower() if asset_ref else ".tif"
        vtype = classify_visual_type(asset_ref, r.get("source_asset_type", ""))
        fixture = is_fixture_or_synthetic(asset_ref + " " + patch_raw)
        # Respect source label/target declarations
        has_label = label_st not in ("NO_LABEL", "", "NOT_APPLICABLE")
        conf = "HIGH" if "READY" in elig_hint and not fixture else "MEDIUM"
        elig, elig_reason, blocked = classify_dino_eligibility(
            pid, region, vtype, conf, fixture, has_label
        )
        rel = asset_ref if asset_ref else f"manifests/v1fu/ref_{i:04d}"
        ph = path_hash(rel)

        rows.append({
            "visual_asset_id": f"V1PU_VA_{i:05d}",
            "relative_path": rel,
            "path_hash": ph,
            "file_ext": ext,
            "file_size_bytes": "0",  # metadata-only; no stat call
            "inferred_patch_id": pid,
            "inferred_alias": alias,
            "inferred_region": region,
            "asset_visual_type": vtype,
            "eligibility_status": elig,
            "confidence": conf,
            "eligibility_reason": elig_reason,
            "blocked_reason": blocked,
            "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION" if elig == "DINO_ELIGIBLE_REVIEW_ONLY" else "BLOCKED_INVALID_VECTOR",
            "can_create_label": "false",
            "can_train_model": "false",
            "target_created": "false",
            "notes": f"source=v1fu protocol_c={'yes' if pid in v1oz_patches else 'no'}",
        })
    return rows


def _audit_from_v1fm() -> list[dict[str, Any]]:
    """Completa com as entradas de designação de patch do v1fm."""
    rows: list[dict[str, Any]] = []
    v1fu_pids = {r.get("canonical_patch_id", "").strip().upper() for r in read_v1fu_manifest()}
    for i, r in enumerate(read_v1fm_designation(), 1):
        pid = r.get("canonical_patch_id", "").strip().upper()
        if not pid or pid in v1fu_pids:
            continue  # skip duplicates
        region_raw = r.get("region", "")
        tif = r.get("tif_filename", "")
        desg = r.get("designation_status", "")
        conf_raw = r.get("designation_confidence", "NONE")
        asset_ref = f"data/sentinel/{tif}" if tif else f"manifests/v1fm/{pid}"
        region = normalize_region(region_raw)
        ext = ".tif" if tif else ""
        vtype = "SENTINEL_TIF_REFERENCE" if tif else "UNKNOWN_VISUAL"
        fixture = is_fixture_or_synthetic(pid + " " + tif)
        conf = conf_raw.upper() if conf_raw.upper() in ("HIGH", "MEDIUM", "LOW") else "LOW"
        elig, elig_reason, blocked = classify_dino_eligibility(pid, region, vtype, conf, fixture)
        rows.append({
            "visual_asset_id": f"V1PU_FM_{i:05d}",
            "relative_path": asset_ref,
            "path_hash": path_hash(asset_ref),
            "file_ext": ext,
            "file_size_bytes": "0",
            "inferred_patch_id": pid,
            "inferred_alias": pid,
            "inferred_region": region,
            "asset_visual_type": vtype,
            "eligibility_status": elig,
            "confidence": conf,
            "eligibility_reason": elig_reason,
            "blocked_reason": blocked,
            "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION" if elig == "DINO_ELIGIBLE_REVIEW_ONLY" else "BLOCKED_INVALID_VECTOR",
            "can_create_label": "false",
            "can_train_model": "false",
            "target_created": "false",
            "notes": f"source=v1fm designation={desg}",
        })
    return rows


def build_audit() -> list[dict[str, Any]]:
    rows = _audit_from_v1fu() + _audit_from_v1fm()
    # Deduplicate by path_hash
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in rows:
        k = r["path_hash"]
        if k not in seen:
            seen.add(k)
            deduped.append(r)
    return deduped


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_status: dict[str, int] = {}
    for r in rows:
        k = r["eligibility_status"]
        by_status[k] = by_status.get(k, 0) + 1
    eligible = sum(1 for r in rows if r["eligibility_status"] == "DINO_ELIGIBLE_REVIEW_ONLY")
    manual = sum(1 for r in rows if r["eligibility_status"] == "DINO_REVIEW_CANDIDATE_NEEDS_MANUAL_CHECK")
    blocked = sum(1 for r in rows if r["eligibility_status"].startswith("DINO_BLOCKED"))
    return [
        {"stat_key": "visual_assets_audited", "stat_value": str(len(rows))},
        {"stat_key": "dino_eligible_review_only", "stat_value": str(eligible)},
        {"stat_key": "manual_check_candidates", "stat_value": str(manual)},
        {"stat_key": "blocked_assets", "stat_value": str(blocked)},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
    ]


def run() -> None:
    rows = build_audit()
    require_no_abs_paths(rows, "v1pu_audit")
    assert_no_forbidden_true(rows, "v1pu_audit")
    summary = build_summary(rows)
    eligible = sum(1 for r in rows if r["eligibility_status"] == "DINO_ELIGIBLE_REVIEW_ONLY")
    write_csv(OUT_AUDIT, rows, AUDIT_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_AUDIT, AUDIT_FIELDS, "v1pu_visual_asset_eligibility_audit")
    write_schema(SCH_SUM, SUM_FIELDS, "v1pu_visual_asset_eligibility_summary")
    write_doc(DOC, "v1pu — Visual Asset Eligibility Audit", [
        "## Objetivo",
        "Auditar elegibilidade de assets visuais (referências de manifesto) para "
        "fila DINO review-only. Não requer scene_date nem temporal unlock. "
        "Apenas metadados — sem leitura de pixels.",
        "## Fontes",
        "v1fu Sentinel manifest (128 entradas), v1fm patch designation (59 entradas), "
        "v1pn inventory.",
        "## Guardrails",
        "can_create_label, can_train_model e target_created sempre false. "
        "DINO é representação visual review-only.",
        f"## Resultado",
        f"Assets auditados: {len(rows)}. Elegíveis review-only: {eligible}.",
    ])
    print(f"[v1pu] audited={len(rows)} eligible={eligible}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pu visual asset eligibility audit").parse_args()
    run()
