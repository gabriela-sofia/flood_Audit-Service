"""Utilitários compartilhados do harness de execução DINO (v1pn a v1pt).

Harness controlado, auditável e de falha fechada para gerar embedding DINOv2 real. Não
cria rótulo, alvo de treino nem referência de campo, e não baixa modelo algum a menos
que REVP_DINO_ALLOW_DOWNLOAD=true esteja explicitamente ligado — o padrão é não sair
para a rede.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from revp_v1pg_v1pm_dino_representation_common import (
    DATASETS, DOCS, EXPECTED_DINO_DIM, ROOT, SCHEMAS,
    _f, _p, assert_no_forbidden_true, is_fixture_or_synthetic,
    normalize_region, path_hash, require_no_abs_paths,
    sanitized_rel_path, sha256_short, validate_vector, vector_stats,
    write_csv, write_doc, write_schema,
)

__all__ = [
    "DATASETS", "DOCS", "EXPECTED_DINO_DIM", "ROOT", "SCHEMAS",
    "_f", "_p", "assert_no_forbidden_true", "is_fixture_or_synthetic",
    "normalize_region", "path_hash", "require_no_abs_paths",
    "sanitized_rel_path", "sha256_short", "validate_vector", "vector_stats",
    "write_csv", "write_doc", "write_schema",
    "IMAGE_EXTENSIONS", "ALLOWED_EXECUTION_STATUSES",
    "FORBIDDEN_EXECUTION_TRUE_FIELDS", "LOCAL_ONLY_DIR",
    "probe_backend", "can_execute_embedding",
    "normalize_patch_from_name", "is_local_only_path", "mask_local_path",
    "build_vector_row_fields", "load_smoke_embeddings",
]

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"})

ALLOWED_EXECUTION_STATUSES = frozenset({
    "IMAGE_CANDIDATE_REVIEW_ONLY",
    "IMAGE_BLOCKED_FIXTURE",
    "IMAGE_BLOCKED_UNREADABLE",
    "MODEL_AVAILABLE_LOCAL",
    "MODEL_UNAVAILABLE_FAIL_CLOSED",
    "EMBEDDING_EXECUTED_REVIEW_ONLY",
    "EMBEDDING_SKIPPED_MODEL_UNAVAILABLE",
    "EMBEDDING_SKIPPED_DRY_RUN",
    "EMBEDDING_BLOCKED_INVALID_DIM",
    "EMBEDDING_BLOCKED_RUNTIME_ERROR",
    "EMBEDDING_BLOCKED_NO_MODEL",
})

FORBIDDEN_EXECUTION_TRUE_FIELDS = (
    "can_create_label", "can_train_model", "target_created",
    "dino_can_create_label", "dino_can_train_model", "dino_target_field_created",
    "can_be_used_as_class", "can_infer_same_event",
)

LOCAL_ONLY_DIR = "local_runs"

_REGION_NAME_RE = re.compile(
    r"(recife|rec\b|petropolis|petr[oó]polis|pet\b|curitiba|cwb)", re.IGNORECASE
)
_PATCH_ID_RE = re.compile(r"([A-Z]{2,5}_\d{4,6})", re.IGNORECASE)
_YEAR_RE = re.compile(r"(20\d{2})")


def normalize_patch_from_name(name: str) -> tuple[str, str, str]:
    """Deduz (patch_id, alias, região) por heurística sobre o nome do arquivo."""
    stem = Path(name).stem
    region = ""
    m = _REGION_NAME_RE.search(stem)
    if m:
        region = normalize_region(m.group(1))
    pid_m = _PATCH_ID_RE.search(stem)
    patch = pid_m.group(1).upper() if pid_m else ""
    alias = stem if not patch else patch
    return (patch or "UNKNOWN_PATCH", alias, region or "UNKNOWN")


def is_local_only_path(rel: str) -> bool:
    return rel.startswith(LOCAL_ONLY_DIR + "/") or rel.startswith(LOCAL_ONLY_DIR + "\\")


def mask_local_path(rel: str) -> str:
    return f"local_only:{path_hash(rel)}"


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _try_import(name: str) -> tuple[bool, str]:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "unknown")
        return (True, str(ver))
    except ImportError:
        return (False, "not_installed")


def probe_backend() -> dict[str, Any]:
    """Detecta quais backends Python estão disponíveis. Não baixa nada."""
    numpy_ok, numpy_ver = _try_import("numpy")
    pil_ok, pil_ver = (False, "not_installed")
    try:
        from PIL import Image  # noqa: F401
        import PIL
        pil_ok, pil_ver = True, getattr(PIL, "__version__", "unknown")
    except ImportError:
        pass
    torch_ok, torch_ver = _try_import("torch")
    transformers_ok, transformers_ver = _try_import("transformers")
    timm_ok, timm_ver = _try_import("timm")

    model_path = os.environ.get("REVP_DINO_MODEL_PATH", "")
    model_name = os.environ.get("REVP_DINO_MODEL_NAME", "")
    allow_download = os.environ.get("REVP_DINO_ALLOW_DOWNLOAD", "false").lower() == "true"

    model_path_exists = bool(model_path) and Path(model_path).exists()
    model_name_set = bool(model_name)

    model_available = model_path_exists or (model_name_set and allow_download)
    deps_ok = numpy_ok and pil_ok and (torch_ok or timm_ok) and (transformers_ok or timm_ok)

    can_execute = deps_ok and model_available

    if can_execute:
        final_status = "DINO_BACKEND_READY_LOCAL_MODEL"
    elif not deps_ok:
        final_status = "DINO_BACKEND_MISSING_DEPENDENCIES_FAIL_CLOSED"
    else:
        final_status = "DINO_BACKEND_MODEL_UNAVAILABLE_FAIL_CLOSED"

    return {
        "numpy": (numpy_ok, numpy_ver),
        "pil": (pil_ok, pil_ver),
        "torch": (torch_ok, torch_ver),
        "transformers": (transformers_ok, transformers_ver),
        "timm": (timm_ok, timm_ver),
        "model_path": model_path,
        "model_path_exists": model_path_exists,
        "model_name": model_name,
        "allow_download": allow_download,
        "can_execute": can_execute,
        "final_status": final_status,
    }


def can_execute_embedding() -> tuple[bool, str]:
    info = probe_backend()
    return (info["can_execute"], info["final_status"])


# ---------------------------------------------------------------------------
# Vector row builder (for v1pr output)
# ---------------------------------------------------------------------------

def build_vector_row_fields() -> list[str]:
    return [
        "embedding_id", "patch_id", "alias", "region", "source_run_id",
        "visual_asset_id", "vector_dim", "vector_sha256_16", "vector_norm_l2",
        "vector_mean", "vector_std", "has_nan", "has_inf", "is_zero_vector",
        "embedding_status", "dino_allowed_use", "dino_can_create_label",
        "dino_can_train_model", "dino_target_field_created", "blocked_reason", "notes",
    ]


def make_vector_row(
    idx: int, patch_id: str, alias: str, region: str,
    run_id: str, asset_id: str, vec: list[float] | None,
) -> dict[str, Any]:
    status, blocked = validate_vector(vec)
    st = vector_stats(vec) if vec is not None else {"dim": 0, "norm": float("nan"), "mean": float("nan"), "std": float("nan"), "has_nan": False, "has_inf": False, "is_zero": False}
    vsha = sha256_short(",".join(f"{x:.6g}" for x in vec)) if vec else ""
    allowed = "REVIEW_ONLY_REPRESENTATION" if status == "VALID_REVIEW_ONLY" else "BLOCKED_INVALID_VECTOR"
    return {
        "embedding_id": f"V1PR_EMB_{idx:05d}",
        "patch_id": patch_id, "alias": alias, "region": region,
        "source_run_id": run_id, "visual_asset_id": asset_id,
        "vector_dim": str(st["dim"]),
        "vector_sha256_16": vsha,
        "vector_norm_l2": _f(st["norm"]),
        "vector_mean": _f(st["mean"]),
        "vector_std": _f(st["std"]),
        "has_nan": str(st["has_nan"]).lower(),
        "has_inf": str(st["has_inf"]).lower(),
        "is_zero_vector": str(st["is_zero"]).lower(),
        "embedding_status": status,
        "dino_allowed_use": allowed,
        "dino_can_create_label": "false",
        "dino_can_train_model": "false",
        "dino_target_field_created": "false",
        "blocked_reason": blocked,
        "notes": "",
    }


# ---------------------------------------------------------------------------
# Load smoke embeddings (v1pr → v1ps)
# ---------------------------------------------------------------------------

def load_smoke_embeddings(registry_path: Path) -> list[dict[str, Any]]:
    """Devolve os embeddings válidos e não nulos de 768D do feature store do v1pr."""
    from revp_v1pg_v1pm_dino_representation_common import read_csv as _read_csv
    from revp_v1pg_v1pm_dino_representation_common import parse_embedding_from_row
    rows = _read_csv(registry_path)
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("embedding_status") != "VALID_REVIEW_ONLY":
            continue
        # no v1pr o embedding vem no CSV de resultados, dentro do campo embedding
        # We reconstruct from vector stats; for actual vectors load from results.
        out.append(r)
    return out
