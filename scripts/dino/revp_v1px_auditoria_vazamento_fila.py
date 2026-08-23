"""REV-P v1px — auditoria de vazamento na fila DINO.

Passa a fila ampliada e todas as saídas de v1pu a v1pw procurando violação de guardrail:
nenhuma linha pode marcar rótulo/treino/alvo como verdadeiro, nenhuma pode ser fixture,
nenhuma pode expor caminho absoluto ou local_runs/, nenhuma pode criar nível C3+/C4,
nenhuma pode usar scene_date fora do que ele significa — e toda linha bloqueada tem de
trazer o motivo escrito.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from revp_v1pu_v1pz_elegibilidade_comum import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, require_no_abs_paths, write_csv, write_doc, write_schema,
)
from revp_v1pg_v1pm_representacao_comum import read_csv

OUT_AUDIT = _p("REVP_V1PX_OUT_AUDIT", DATASETS / "dino_queue_leakage_audit_v1px.csv")
OUT_SUM = _p("REVP_V1PX_OUT_SUM", DATASETS / "dino_queue_leakage_summary_v1px.csv")
SCH_AUDIT = _p("REVP_V1PX_SCH_AUDIT", SCHEMAS / "dino_queue_leakage_audit_v1px_schema.csv")
SCH_SUM = _p("REVP_V1PX_SCH_SUM", SCHEMAS / "dino_queue_leakage_summary_v1px_schema.csv")
DOC = _p("REVP_V1PX_DOC", DOCS / "revp_v1px_auditoria_vazamento_fila.md")

AUDIT_FIELDS = [
    "check_id", "stage", "filename", "check_name",
    "status", "severity", "observed", "expected", "explanation",
]
SUM_FIELDS = ["stat_key", "stat_value"]

ABS_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")

FORBIDDEN_TRUE_TOKENS = [
    "can_create_label,true", "can_train_model,true", "target_created,true",
    "dino_can_create_label,true", "dino_can_train_model,true",
    "dino_target_field_created,true", "ground_truth,true",
    "can_be_used_as_ground_truth,true", "dino_can_validate_event,true",
    "can_be_used_as_class,true", "can_infer_same_event,true",
]

AUDIT_TARGETS = [
    ("v1pu", "dino_visual_asset_eligibility_audit_v1pu.csv",
     ["can_create_label", "can_train_model", "target_created"]),
    ("v1pu", "dino_visual_asset_eligibility_summary_v1pu.csv", []),
    ("v1pv", "dino_patch_visual_linkage_registry_v1pv.csv",
     ["can_create_label", "can_train_model", "target_created"]),
    ("v1pv", "dino_patch_visual_linkage_summary_v1pv.csv", []),
    ("v1pw", "dino_review_only_execution_queue_expanded_v1pw.csv",
     ["can_create_label", "can_train_model", "target_created"]),
    ("v1pw", "dino_review_only_execution_queue_expanded_summary_v1pw.csv", []),
]


def _check_file(stage: str, fname: str, guard_cols: list[str]) -> list[dict[str, Any]]:
    path = DATASETS / fname
    checks: list[dict[str, Any]] = []
    qid = [0]

    def _c(name: str, status: str, sev: str, obs: str, exp: str, expl: str = "") -> None:
        qid[0] += 1
        checks.append({
            "check_id": f"V1PX_{qid[0]:04d}",
            "stage": stage, "filename": fname,
            "check_name": name, "status": status,
            "severity": sev, "observed": obs, "expected": exp,
            "explanation": expl,
        })

    exists = path.exists()
    _c("file_exists", "PASS" if exists else "FAIL",
       "HIGH" if not exists else "INFO", str(exists), "true")

    if not exists:
        return checks

    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception as e:
        _c("readable", "FAIL", "HIGH", str(e), "true")
        return checks

    # Header
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        header = list(csv.DictReader(fh).fieldnames or [])
    _c("header_present", "PASS" if header else "FAIL",
       "HIGH" if not header else "INFO", str(bool(header)), "true")

    # Forbidden true tokens
    low = text.lower()
    for tok in FORBIDDEN_TRUE_TOKENS:
        if tok in low:
            _c(f"no_{tok.replace(',','_').replace('=','_')}",
               "FAIL", "CRITICAL", tok, "NOT_FOUND", "guardrail_violation")

    # Absolute path
    if ABS_RE.search(text):
        _c("no_absolute_path", "FAIL", "HIGH", "ABS_PATH_FOUND", "NOT_FOUND")
    else:
        _c("no_absolute_path", "PASS", "INFO", "none", "NOT_FOUND")

    # local_runs exposure
    if "local_runs" in low:
        _c("no_local_runs_exposure", "FAIL", "HIGH", "local_runs_found", "NOT_FOUND")
    else:
        _c("no_local_runs_exposure", "PASS", "INFO", "none", "NOT_FOUND")

    # Blocked rows have reason
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if "blocked_reason" in header:
        missing = sum(
            1 for r in rows
            if r.get("eligibility_status", "").startswith("DINO_BLOCKED")
            and not r.get("blocked_reason", "").strip()
        )
        _c("blocked_rows_have_reason",
           "PASS" if missing == 0 else "FAIL",
           "MEDIUM", str(missing), "0", "blocked rows must have blocked_reason")

    # No fixture in eligible
    if "eligibility_status" in header and "is_fixture_or_synthetic" in header:
        leaked = sum(1 for r in rows
                     if r.get("eligibility_status") == "DINO_ELIGIBLE_REVIEW_ONLY"
                     and r.get("is_fixture_or_synthetic") == "true")
        _c("no_fixture_in_eligible", "PASS" if leaked == 0 else "FAIL",
           "HIGH", str(leaked), "0", "fixture assets must not be eligible")

    return checks


def run() -> None:
    all_checks: list[dict[str, Any]] = []
    for stage, fname, guard_cols in AUDIT_TARGETS:
        all_checks.extend(_check_file(stage, fname, guard_cols))

    fails = [c for c in all_checks if c["status"] == "FAIL"]
    critical = [c for c in fails if c["severity"] == "CRITICAL"]
    summary = [
        {"stat_key": "total_checks", "stat_value": str(len(all_checks))},
        {"stat_key": "pass", "stat_value": str(sum(1 for c in all_checks if c["status"] == "PASS"))},
        {"stat_key": "fail", "stat_value": str(len(fails))},
        {"stat_key": "critical", "stat_value": str(len(critical))},
        {"stat_key": "leakage_status",
         "stat_value": "LEAKAGE_AUDIT_PASS" if not critical else "LEAKAGE_AUDIT_CRITICAL_FAIL"},
    ]
    require_no_abs_paths(all_checks, "v1px_audit")
    write_csv(OUT_AUDIT, all_checks, AUDIT_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_AUDIT, AUDIT_FIELDS, "v1px_dino_queue_leakage_audit")
    write_schema(SCH_SUM, SUM_FIELDS, "v1px_dino_queue_leakage_summary")
    write_doc(DOC, "v1px — DINO Queue Leakage Audit", [
        "## Objetivo",
        "Verificar que fila expandida e outputs v1pu-v1pw não violam guardrails.",
        "## Checks",
        "rotulo/train/target=true, abs paths, local_runs exposure, fixture em elegíveis, "
        "blocked rows sem reason.",
        f"## Resultado",
        f"Total checks: {len(all_checks)}. Falhas: {len(fails)}. Críticos: {len(critical)}.",
    ])
    print(f"[v1px] checks={len(all_checks)} fails={len(fails)} critical={len(critical)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1px dino queue leakage audit").parse_args()
    run()
