"""Utilitários compartilhados do pipeline de smoke embedding (v1qg a v1qm).

Transforma a fila visual somente-revisão do v1qa num pipeline de smoke que roda de
verdade e produz embedding DINOv2 real de 768 dimensões — estritamente de falha fechada,
offline e somente-revisão.

O limite metodológico que este módulo nunca cruza:
  * gerar embedding NÃO confirma evento;
  * gerar embedding NÃO confirma scene_date;
  * similaridade, PCA e agrupamento NÃO são classe;
  * nenhum vetor vira rótulo, alvo ou referência de campo.
O DINO é representação visual auto-supervisionada usada para priorizar revisão humana,
nunca para validar operacionalmente.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from revp_v1pg_v1pm_dino_representation_common import (
    DATASETS, DOCS, EXPECTED_DINO_DIM, ROOT, SCHEMAS,
    _f, _p, assert_no_forbidden_true, cosine_similarity, euclidean_distance,
    is_fixture_or_synthetic, is_fixture_patch, kmeans_simple, normalize_region,
    parse_embedding_from_row, path_hash, pca_2d, read_csv, read_csv_header,
    read_json_safe, require_no_abs_paths, sanitized_rel_path, sha256_short,
    validate_vector, vector_stats, write_csv, write_doc, write_schema,
)

__all__ = [
    "DATASETS", "DOCS", "EXPECTED_DINO_DIM", "ROOT", "SCHEMAS",
    "_f", "_p", "assert_no_forbidden_true", "cosine_similarity",
    "euclidean_distance", "is_fixture_or_synthetic", "is_fixture_patch",
    "kmeans_simple", "normalize_region", "parse_embedding_from_row",
    "path_hash", "pca_2d", "read_csv", "read_csv_header", "read_json_safe",
    "require_no_abs_paths", "sanitized_rel_path", "sha256_short",
    "validate_vector", "vector_stats", "write_csv", "write_doc", "write_schema",
    "REVIEW_ONLY_TEXT", "SMOKE_FORBIDDEN_FIELDS",
    "env_true", "env_str", "env_int", "write_json", "mask_local",
    "file_sha256_short", "asset_candidate_roots", "resolve_local_asset",
    "safe_model_probe", "expected_embedding_dim", "embedding_columns",
    "write_vector_csv_row", "vector_to_columns", "read_embedding_rows",
    "pca_2d_review", "exploratory_clusters",
    "read_queue_v1qa", "read_backend_v1pp", "read_manifest",
    "normalize_identity", "guardrail_ok",
]

# Mandatory scientific boundary phrase (embedded verbatim in v1qm doc).
REVIEW_ONLY_TEXT = (
    "Os embeddings DINO smoke representam descritores visuais auto-supervisionados "
    "de patches Sentinel selecionados para revisão. Eles não constituem rótulo, "
    "ground truth, confirmação temporal, validação de evento observado ou evidência "
    "operacional de inundação/deslizamento."
)

# Field tokens that must never carry the value "true" in any v1qg-v1qm output.
SMOKE_FORBIDDEN_FIELDS = (
    "can_create_label", "can_train_model", "target_created", "ground_truth",
    "cluster_is_label", "similarity_validates_event", "pca_validates_event",
    "dino_can_create_label", "dino_can_train_model", "dino_target_field_created",
)

LOCAL_RUNS_PREFIX = ("local_runs/", "local_runs\\")


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def env_true(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# JSON / path masking / file hashing
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")


def mask_local(rel: str) -> str:
    """Mascara caminho sob local_runs/; qualquer outro sai como caminho relativo intacto."""
    if any(str(rel).startswith(p) for p in LOCAL_RUNS_PREFIX):
        return f"local_only:{path_hash(rel)}"
    return rel


def file_sha256_short(path: Path, n: int = 16) -> str:
    """sha256 of a local file (short). Empty string if missing/unreadable."""
    import hashlib
    try:
        if not path.exists() or not path.is_file():
            return ""
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:n]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Local asset resolution
# ---------------------------------------------------------------------------

def asset_candidate_roots() -> list[Path]:
    """Raízes candidatas vindas do ambiente, em ordem de prioridade. Variável ausente ⇒ pulada."""
    roots: list[Path] = []
    for name in ("REVP_SENTINEL_LOCAL_ROOT", "REVP_DINO_VISUAL_ROOT",
                 "REVP_DINO_ASSET_ROOT", "REVP_DINO_SOURCE_ROOT"):
        val = os.environ.get(name, "")
        if val:
            roots.append(Path(val))
    return roots


def resolve_local_asset(relative_path: str,
                        candidate_roots: list[Path] | None = None) -> Path | None:
    """Resolve um caminho relativo de ativo contra as raízes candidatas; a primeira que casar vence.

    Nunca devolve caminho fora de uma raiz candidata. Devolve None quando nada resolve para
    um arquivo existente — quem chama trata isso como falha fechada.
    """
    rel = (relative_path or "").strip()
    if not rel:
        return None
    roots = candidate_roots if candidate_roots is not None else asset_candidate_roots()
    for root in roots:
        try:
            cand = (root / rel)
            if cand.exists() and cand.is_file():
                return cand
        except Exception:
            continue
    # Allow an absolute-resolved relative path only when it exists on disk.
    try:
        direct = Path(rel)
        if direct.is_absolute() and direct.exists() and direct.is_file():
            return direct
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Model probe (offline, never loads weights for inference)
# ---------------------------------------------------------------------------

def expected_embedding_dim(model_name_or_config: Any) -> int:
    """Dimensão esperada do vetor. Lê hidden_size do config.json quando ele existe."""
    default = EXPECTED_DINO_DIM
    if isinstance(model_name_or_config, dict):
        try:
            return int(model_name_or_config.get("hidden_size", default))
        except (TypeError, ValueError):
            return default
    if isinstance(model_name_or_config, (str, Path)):
        p = Path(model_name_or_config)
        cfg = p / "config.json" if p.is_dir() else p
        data = read_json_safe(cfg) if cfg.suffix == ".json" or cfg.exists() else None
        if isinstance(data, dict):
            try:
                return int(data.get("hidden_size", default))
            except (TypeError, ValueError):
                return default
    return default


def _detect_arch(cfg: dict[str, Any]) -> tuple[bool, bool, str]:
    """Devolve (is_dinov2, is_dinov2_with_registers, model_type)."""
    blob = json.dumps(cfg, default=str).lower() if isinstance(cfg, dict) else ""
    model_type = str(cfg.get("model_type", "")).lower() if isinstance(cfg, dict) else ""
    is_v2 = "dinov2" in blob or "dinov2" in model_type
    with_registers = ("register" in blob) or ("dinov2_with_registers" in model_type) \
        or (isinstance(cfg, dict) and int(cfg.get("num_register_tokens", 0) or 0) > 0)
    return (is_v2, bool(with_registers), model_type or ("dinov2" if is_v2 else "unknown"))


def safe_model_probe(model_path: str | None) -> dict[str, Any]:
    """Audita um diretório de modelo local, offline. Não baixa nada. Não deduz nada.

    Carrega só a configuração (via AutoConfig do transformers quando disponível, senão pelo
    config.json) para ler hidden_size. Dos pesos, confere apenas a presença.
    """
    allow_download = env_true("REVP_DINO_ALLOW_DOWNLOAD", False)
    offline = env_str("HF_HUB_OFFLINE", "") == "1"

    info: dict[str, Any] = {
        "model_path": model_path or "",
        "model_path_exists": False,
        "config_exists": False,
        "weights_exists": False,
        "processor_exists": False,
        "transformers_available": False,
        "torch_available": False,
        "offline_mode": offline,
        "allow_download": allow_download,
        "expected_dim": EXPECTED_DINO_DIM,
        "detected_dim": 0,
        "is_dinov2": False,
        "is_dinov2_with_registers": False,
        "model_type": "unknown",
        "config_loadable": False,
        "image_processor_available": False,
        "auto_model_available": False,
    }

    try:
        import transformers  # noqa: F401
        info["transformers_available"] = True
        try:
            from transformers import AutoImageProcessor  # noqa: F401
            info["image_processor_available"] = True
        except Exception:
            pass
        try:
            from transformers import AutoModel  # noqa: F401
            info["auto_model_available"] = True
        except Exception:
            pass
    except Exception:
        pass
    try:
        import torch  # noqa: F401
        info["torch_available"] = True
    except Exception:
        pass

    if not model_path:
        return info
    p = Path(model_path)
    info["model_path_exists"] = p.exists() and p.is_dir()
    if not info["model_path_exists"]:
        return info

    cfg_path = p / "config.json"
    info["config_exists"] = cfg_path.exists()
    weight_globs = ("*.safetensors", "*.bin", "*.pt", "*.pth")
    info["weights_exists"] = any(any(p.glob(g)) for g in weight_globs)
    proc_names = ("preprocessor_config.json", "processor_config.json",
                  "image_processor_config.json")
    info["processor_exists"] = any((p / n).exists() for n in proc_names)

    cfg = read_json_safe(cfg_path) if info["config_exists"] else None
    if isinstance(cfg, dict):
        info["detected_dim"] = expected_embedding_dim(cfg)
        v2, regs, mtype = _detect_arch(cfg)
        info["is_dinov2"], info["is_dinov2_with_registers"], info["model_type"] = v2, regs, mtype

    # Try a config-only load (no weights, offline) to confirm transformers can parse it.
    if info["transformers_available"] and info["config_exists"]:
        try:
            from transformers import AutoConfig
            ac = AutoConfig.from_pretrained(str(p), local_files_only=not allow_download)
            info["config_loadable"] = True
            hs = getattr(ac, "hidden_size", None)
            if isinstance(hs, int) and hs > 0:
                info["detected_dim"] = hs
        except Exception:
            info["config_loadable"] = False

    return info


# ---------------------------------------------------------------------------
# 768D vector columns
# ---------------------------------------------------------------------------

def embedding_columns(dim: int = EXPECTED_DINO_DIM) -> list[str]:
    return [f"embedding_{i:03d}" for i in range(dim)]


def vector_to_columns(vec: list[float], dim: int = EXPECTED_DINO_DIM) -> dict[str, str]:
    cols = embedding_columns(dim)
    out: dict[str, str] = {}
    for i, name in enumerate(cols):
        out[name] = f"{vec[i]:.6g}" if i < len(vec) else ""
    return out


def write_vector_csv_row(base: dict[str, Any], vec: list[float],
                         dim: int = EXPECTED_DINO_DIM) -> dict[str, Any]:
    """Junta o metadado às colunas embedding_000..embedding_{dim-1}."""
    row = dict(base)
    row.update(vector_to_columns(vec, dim))
    return row


def read_embedding_rows(path: Path) -> list[dict[str, Any]]:
    """Lê o CSV do feature store e extrai o vetor de embedding de cada linha.

    Devolve dicionários no formato {meta: linha original, vector: list[float]|None}.
    """
    rows = read_csv(path)
    out: list[dict[str, Any]] = []
    for r in rows:
        vec = parse_embedding_from_row(r)
        out.append({"meta": r, "vector": vec})
    return out


# ---------------------------------------------------------------------------
# PCA / clustering (review-only, never a class)
# ---------------------------------------------------------------------------

def pca_2d_review(vectors: list[list[float]]) -> tuple[list[tuple[float, float]],
                                                       tuple[float, float], str]:
    """PCA to 2D. sklearn when available, numpy fallback. Fail-closed if n<2."""
    n = len(vectors)
    if n < 2:
        return ([], (0.0, 0.0), "PCA_FAIL_CLOSED_N_LT_2")
    try:
        from sklearn.decomposition import PCA  # type: ignore
        import numpy as np
        x = np.asarray(vectors, dtype=float)
        k = min(2, x.shape[1], x.shape[0])
        model = PCA(n_components=k, svd_solver="full", random_state=0)
        proj = model.fit_transform(x)
        evr = list(model.explained_variance_ratio_)
        ex = float(evr[0]) if len(evr) > 0 else 0.0
        ey = float(evr[1]) if len(evr) > 1 else 0.0
        coords = [(float(r[0]), float(r[1]) if proj.shape[1] > 1 else 0.0) for r in proj]
        return (coords, (ex, ey), "PCA_SKLEARN")
    except Exception:
        coords, evr = pca_2d(vectors)
        return (coords, evr, "PCA_NUMPY_FALLBACK")


def exploratory_clusters(vectors: list[list[float]], k: int = 3) -> list[int]:
    """Agrupamento exploratório determinístico. NUNCA é classe. Vazio quando n<4."""
    if len(vectors) < 4:
        return []
    return kmeans_simple(vectors, min(k, len(vectors)))


# ---------------------------------------------------------------------------
# Queue / backend / manifest readers
# ---------------------------------------------------------------------------

_V1QA_QUEUE = _p("REVP_DINO_SMOKE_QUEUE",
                 DATASETS / "dino_execution_queue_from_visual_expansion_v1qa.csv")
_V1PP_SUMMARY = _p("REVP_DINO_SMOKE_BACKEND",
                   DATASETS / "dino_backend_model_probe_summary_v1pp.csv")


def read_queue_v1qa(path: Path | None = None) -> list[dict[str, str]]:
    return read_csv(path or _V1QA_QUEUE)


def read_backend_v1pp(path: Path | None = None) -> dict[str, str]:
    rows = read_csv(path or _V1PP_SUMMARY)
    return {r.get("stat_key", ""): r.get("stat_value", "") for r in rows}


def read_manifest(rel_default: str, env: str) -> list[dict[str, str]]:
    return read_csv(_p(env, DATASETS / rel_default))


# ---------------------------------------------------------------------------
# Identity normalization & guardrails
# ---------------------------------------------------------------------------

def normalize_identity(row: dict[str, str]) -> tuple[str, str, str]:
    """Devolve (patch_id, alias, região) normalizados a partir de uma linha da fila."""
    patch = (row.get("patch_id", "") or "UNKNOWN_PATCH").strip().upper()
    alias = (row.get("alias", "") or patch).strip()
    region = normalize_region(row.get("region", ""))
    return (patch, alias, region)


def guardrail_ok(row: dict[str, Any]) -> tuple[bool, str]:
    """Devolve (ok, blocked_reason). ok=False ⇒ violação de guardrail ou de rótulo."""
    for f in ("can_create_label", "can_train_model", "target_created",
              "ground_truth", "cluster_is_label", "similarity_validates_event",
              "pca_validates_event"):
        if str(row.get(f, "false")).strip().lower() == "true":
            return (False, f"guardrail_{f}_true")
    return (True, "")
