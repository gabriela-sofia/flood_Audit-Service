"""Utilitários compartilhados da camada de elegibilidade visual (v1pu a v1pz).

Monta a fila de execução DINO somente-revisão a partir do metadado que já está
versionado nos manifestos — nunca a partir de leitura de pixel, nunca exigindo
scene_date, e nunca criando rótulo, alvo ou referência de campo.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any

from revp_v1pg_v1pm_dino_representation_common import (
    DATASETS, DOCS, SCHEMAS, ROOT,
    _p, assert_no_forbidden_true, is_fixture_or_synthetic,
    normalize_region, path_hash, require_no_abs_paths,
    sanitized_rel_path, sha256_short, write_csv, write_doc, write_schema,
)

__all__ = [
    "DATASETS", "DOCS", "SCHEMAS", "ROOT",
    "_p", "assert_no_forbidden_true", "is_fixture_or_synthetic",
    "normalize_region", "path_hash", "require_no_abs_paths",
    "sanitized_rel_path", "sha256_short", "write_csv", "write_doc", "write_schema",
    "VISUAL_ASSET_TYPES", "DINO_ELIGIBILITY_STATUSES", "FORBIDDEN_FIELDS",
    "PATCH_RE_CANONICAL", "PATCH_RE_RAW", "REGION_FROM_PATH_RE",
    "infer_patch_from_path", "classify_visual_type", "classify_dino_eligibility",
    "read_v1fu_manifest", "read_v1fm_designation", "read_v1oz_queue",
]

VISUAL_ASSET_TYPES = frozenset({
    "SENTINEL_PATCH_PREVIEW",
    "SENTINEL_TECHNICAL_RENDER",
    "SENTINEL_TIF_REFERENCE",
    "PATCH_CONTACT_SHEET",
    "GIS_CONTEXT_ONLY",
    "FIGURE_PANEL",
    "NON_PATCH_IMAGE",
    "UNKNOWN_VISUAL",
})

DINO_ELIGIBILITY_STATUSES = frozenset({
    "DINO_ELIGIBLE_REVIEW_ONLY",
    "DINO_REVIEW_CANDIDATE_NEEDS_MANUAL_CHECK",
    "DINO_BLOCKED_FIXTURE",
    "DINO_BLOCKED_NON_PATCH_IMAGE",
    "DINO_BLOCKED_REGION_MISMATCH",
    "DINO_BLOCKED_NO_PATCH_ID",
    "DINO_BLOCKED_LOW_CONFIDENCE",
})

FORBIDDEN_FIELDS = (
    "can_create_label", "can_train_model", "target_created",
    "dino_can_create_label", "dino_can_train_model", "dino_target_field_created",
    "can_be_used_as_class", "can_infer_same_event", "dino_can_validate_event",
)

# Canonical IDs: CUR_00038, REC_01, PET_00249, etc.
PATCH_RE_CANONICAL = re.compile(
    r"\b((?:CUR|REC|PET|RECIFE|CURITIBA|PETROPOLIS)_\d{1,6})\b", re.IGNORECASE
)
# Raw IDs: patch_curitiba_00038, patch_recife_01, curitiba_00249
PATCH_RE_RAW = re.compile(
    r"(?:patch_)?(curitiba|recife|petropolis|petrópolis)_(\d{3,6})", re.IGNORECASE
)
REGION_FROM_PATH_RE = re.compile(
    r"(curitiba|recife|petropolis|petr[oó]polis)", re.IGNORECASE
)

_REGION_PREFIX = {"CUR": "CURITIBA", "REC": "RECIFE", "PET": "PET"}
_REGION_RAW = {"curitiba": "CURITIBA", "recife": "RECIFE",
               "petropolis": "PET", "petrópolis": "PET"}


def infer_patch_from_path(path_str: str) -> tuple[str, str, str]:
    """Devolve (patch_id canônico, alias, região) a partir do caminho. Vazio se não der."""
    s = path_str.replace("\\", "/")
    # Try canonical first
    m = PATCH_RE_CANONICAL.search(s)
    if m:
        pid = m.group(1).upper()
        prefix = pid.split("_")[0]
        region = _REGION_PREFIX.get(prefix, normalize_region(prefix))
        return (pid, pid, region)
    # Try raw
    m = PATCH_RE_RAW.search(s)
    if m:
        reg_raw, num = m.group(1).lower(), m.group(2)
        prefix = {"curitiba": "CUR", "recife": "REC",
                  "petropolis": "PET", "petrópolis": "PET"}.get(reg_raw, reg_raw[:3].upper())
        pid = f"{prefix}_{int(num):05d}"
        region = _REGION_RAW.get(reg_raw, reg_raw.upper())
        return (pid, f"{prefix.lower()}_{num}", region)
    # Region hint only
    m = REGION_FROM_PATH_RE.search(s)
    if m:
        region = normalize_region(m.group(1))
        return ("UNKNOWN_PATCH", s.split("/")[-1], region)
    return ("UNKNOWN_PATCH", s.split("/")[-1], "UNKNOWN")


def classify_visual_type(path_str: str, asset_type_hint: str = "") -> str:
    low = (path_str + " " + asset_type_hint).lower()
    if any(x in low for x in (".tif", ".tiff", "sentinel_tif", "raster")):
        return "SENTINEL_TIF_REFERENCE"
    if any(x in low for x in ("preview", "thumbnail", "rgb_preview", "composite")):
        return "SENTINEL_PATCH_PREVIEW"
    if any(x in low for x in ("render", "technical", "visual_output", "rgb_render")):
        return "SENTINEL_TECHNICAL_RENDER"
    if any(x in low for x in ("contact_sheet", "mosaic", "tile_grid")):
        return "PATCH_CONTACT_SHEET"
    if any(x in low for x in ("gis", "context", "basemap", "background")):
        return "GIS_CONTEXT_ONLY"
    if any(x in low for x in ("fig", "figure", "panel", "plot", "graph", "chart")):
        return "FIGURE_PANEL"
    if any(x in low for x in ("patch_curitiba", "patch_recife", "patch_petropolis")):
        return "SENTINEL_TIF_REFERENCE"
    return "UNKNOWN_VISUAL"


def classify_dino_eligibility(
    patch_id: str,
    region: str,
    visual_type: str,
    confidence: str,
    is_fixture: bool,
    has_label: bool = False,
) -> tuple[str, str, str]:
    """Devolve (eligibility_status, eligibility_reason, blocked_reason)."""
    if is_fixture:
        return ("DINO_BLOCKED_FIXTURE", "", "fixture_or_synthetic")
    if has_label:
        return ("DINO_BLOCKED_FIXTURE", "", "label_detected_in_source")
    if patch_id in ("UNKNOWN_PATCH", ""):
        if confidence in ("LOW", "NONE", ""):
            return ("DINO_BLOCKED_NO_PATCH_ID", "", "no_patch_id_inferred")
    if visual_type in ("FIGURE_PANEL", "GIS_CONTEXT_ONLY"):
        return ("DINO_BLOCKED_NON_PATCH_IMAGE", "", f"non_patch_image_type={visual_type}")
    if visual_type == "UNKNOWN_VISUAL" and patch_id == "UNKNOWN_PATCH":
        return ("DINO_BLOCKED_LOW_CONFIDENCE", "", "unknown_type_and_no_patch_id")
    if confidence in ("MEDIUM", "LOW") and visual_type == "UNKNOWN_VISUAL":
        return ("DINO_REVIEW_CANDIDATE_NEEDS_MANUAL_CHECK", "needs_manual_type_verification", "")
    return ("DINO_ELIGIBLE_REVIEW_ONLY", "sentinel_patch_reference_review_only", "")


# ---------------------------------------------------------------------------
# Manifest readers
# ---------------------------------------------------------------------------

V1FU_MANIFEST = ROOT / "manifests" / "dino_inputs" / \
    "revp_v1fu_dino_sentinel_input_manifest" / "dino_sentinel_input_manifest_v1fu.csv"
V1FM_DESIGNATION = ROOT / "manifests" / "patch_grounding" / \
    "revp_v1fm_explicit_patch_tif_designation" / "patch_designation_table_v1fm.csv"
V1OZ_QUEUE = DATASETS / "recife_dino_review_only_representation_queue_v1oz.csv"


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def read_v1fu_manifest() -> list[dict[str, str]]:
    return _read(V1FU_MANIFEST)


def read_v1fm_designation() -> list[dict[str, str]]:
    return _read(V1FM_DESIGNATION)


def read_v1oz_queue() -> list[dict[str, str]]:
    return _read(V1OZ_QUEUE)
