"""Utilitários compartilhados da camada de representação DINO (v1pg a v1pm).

Este bloco monta uma camada de representação auditável e somente-revisão sobre os
embeddings DINOv2, integrada ao Protocolo C. O embedding é tratado pelo que ele é: uma
representação visual auto-supervisionada do patch Sentinel — nunca um rótulo
supervisionado, nunca alvo de treino, nunca referência de campo, nunca validador de
evento.

Tudo que sai daqui é exploratório, contextual ou bloqueado. Nenhuma função deste módulo
cria rótulo operacional, alvo de treino ou referência de campo.
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

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = ROOT / "docs" / "metodologia_cientifica"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_DINO_DIM = 768

ALLOWED_DINO_USE = frozenset([
    "REVIEW_ONLY_REPRESENTATION",
    "EXPLORATORY_SIMILARITY_ONLY",
    "EXPLORATORY_REPRESENTATION_ONLY",
    "VISUAL_CONTEXT_ONLY",
    "BLOCKED_NO_EMBEDDING",
    "BLOCKED_INVALID_VECTOR",
    "BLOCKED_FIXTURE_OR_SYNTHETIC",
])

# Exact "field,true" tokens that must never appear in any output CSV.
FORBIDDEN_TRUE_TOKENS = (
    "ground_truth,true",
    "can_train_model,true",
    "can_create_operational_label,true",
    "can_be_used_as_ground_truth,true",
    "can_promote_to_label,true",
    "dino_can_create_label,true",
    "dino_can_train_model,true",
    "dino_target_field_created,true",
)

# Field names that must never be the literal string "true" in any output row.
FORBIDDEN_TRUE_FIELDS = (
    "ground_truth",
    "can_train_model",
    "can_create_operational_label",
    "can_be_used_as_ground_truth",
    "can_promote_to_label",
    "can_create_label",
    "can_be_used_as_class",
    "can_infer_same_event",
    "dino_can_create_label",
    "dino_can_train_model",
    "dino_can_validate_event",
    "dino_target_field_created",
)

# Terms used by v1pg artifact discovery.
ARTIFACT_TERMS = (
    "dino", "embedding", "768", "pca", "neighbors", "neighbor",
    "similarity", "cluster", "patch_id", "alias",
)

ABS_PATH_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
FIXTURE_TERMS = ("fixture", "synthetic", "test_only", "dummy", "mock", "sample_random")
FIXTURE_PATCH_RE = re.compile(r"^(REC|PET|CWB)_0{3}\d{2}$")

REGION_ALIASES: dict[str, str] = {
    "recife": "RECIFE", "rec": "RECIFE",
    "petropolis": "PET", "petrópolis": "PET", "pet": "PET",
    "curitiba": "CURITIBA", "cwb": "CURITIBA",
}


# ---------------------------------------------------------------------------
# CSV / JSON IO
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Grava o CSV escrevendo o cabeçalho mesmo quando ``rows`` está vazio.

    Arquivo vazio com cabeçalho é resultado legível; arquivo sem cabeçalho é ambíguo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            return list(csv.DictReader(fh).fieldnames or [])
    except Exception:
        return []


def read_json_safe(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except Exception:
        return None


def write_schema(path: Path, fields: list[str], prefix: str) -> None:
    write_csv(
        path,
        [{"field": f, "description": f"{prefix}: {f}."} for f in fields],
        ["field", "description"],
    )


def write_doc(path: Path, title: str, paragraphs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# " + title + "\n\n" + "\n\n".join(paragraphs).strip() + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Paths / hashing
# ---------------------------------------------------------------------------

def sanitized_rel_path(path: Path, base: Path = ROOT) -> str:
    """Devolve caminho relativo POSIX — nunca um caminho absoluto/privado do Windows."""
    try:
        rel = path.resolve().relative_to(base.resolve())
        return rel.as_posix()
    except Exception:
        # Fall back to the file name only — never leak an absolute path.
        return path.name


def sha256_short(data: Any, n: int = 16) -> str:
    if isinstance(data, bytes):
        raw = data
    else:
        raw = str(data).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:n]


def path_hash(rel_path: str, n: int = 16) -> str:
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:n]


def normalize_region(raw: str) -> str:
    key = (raw or "").strip().lower()
    return REGION_ALIASES.get(key, (raw or "").strip().upper() or "UNKNOWN")


# ---------------------------------------------------------------------------
# Embedding parsing — supports many on-disk shapes
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_DIM_COL_RE = re.compile(r"^(?:dim_|f|embedding_|emb_|v|feat_)(\d+)$")


def _to_float_list(seq: Any) -> list[float] | None:
    try:
        out = [float(x) for x in seq]
    except (TypeError, ValueError):
        return None
    return out or None


def parse_embedding_from_text(text: str) -> list[float] | None:
    """Parse a JSON list or a bracket/comma/space separated numeric string."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    # JSON list form.
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return _to_float_list(parsed)
        except Exception:
            pass
    # Loose numeric extraction (string list, space/comma separated).
    nums = _NUM_RE.findall(s)
    if len(nums) >= 2:
        return _to_float_list(nums)
    return None


def parse_embedding_from_row(row: dict[str, Any]) -> list[float] | None:
    """Extrai o vetor de embedding de uma linha CSV/JSON, em qualquer formato aceito.

    Formatos aceitos:
      * coluna ``embedding`` / ``vector`` / ``features`` com uma lista JSON ou em texto;
      * colunas indexadas ``dim_0..dim_767`` / ``f0..f767`` /
        ``embedding_0..embedding_767`` (e alguns apelidos);
      * uma linha plana com exatamente EXPECTED_DINO_DIM colunas numéricas.
    """
    if not isinstance(row, dict):
        return None

    for key in ("embedding", "vector", "features", "feature_vector", "dino_embedding"):
        for k in row:
            if k and k.lower() == key:
                vec = parse_embedding_from_text(row[k])
                if vec is not None:
                    return vec

    # Indexed dimension columns.
    indexed: list[tuple[int, float]] = []
    for k, v in row.items():
        if not k:
            continue
        m = _DIM_COL_RE.match(k.strip().lower())
        if not m:
            continue
        try:
            indexed.append((int(m.group(1)), float(v)))
        except (TypeError, ValueError):
            return None
    if len(indexed) >= 2:
        indexed.sort(key=lambda t: t[0])
        return [val for _, val in indexed]

    # Flat numeric row of exactly the expected dimensionality.
    numeric: list[float] = []
    for v in row.values():
        try:
            numeric.append(float(v))
        except (TypeError, ValueError):
            numeric = []
            break
    if len(numeric) == EXPECTED_DINO_DIM:
        return numeric
    return None


# ---------------------------------------------------------------------------
# Vector validation & metrics
# ---------------------------------------------------------------------------

def vector_stats(vec: list[float]) -> dict[str, Any]:
    n = len(vec)
    has_nan = any(math.isnan(x) for x in vec)
    has_inf = any(math.isinf(x) for x in vec)
    norm = math.sqrt(sum(x * x for x in vec)) if not (has_nan or has_inf) else float("nan")
    mean = sum(vec) / n if n and not (has_nan or has_inf) else float("nan")
    if n and not (has_nan or has_inf):
        var = sum((x - mean) ** 2 for x in vec) / n
        std = math.sqrt(var)
    else:
        std = float("nan")
    is_zero = (not has_nan and not has_inf and norm == 0.0)
    return {
        "dim": n, "has_nan": has_nan, "has_inf": has_inf,
        "norm": norm, "mean": mean, "std": std, "is_zero": is_zero,
    }


def validate_vector(vec: list[float] | None, expected_dim: int = EXPECTED_DINO_DIM) -> tuple[str, str]:
    """Devolve (embedding_status, blocked_reason). Motivo vazio ⇒ vetor válido."""
    if vec is None:
        return ("BLOCKED_NO_EMBEDDING", "no_embedding_vector_found")
    st = vector_stats(vec)
    if st["dim"] != expected_dim:
        return ("BLOCKED_INVALID_DIMENSION", f"dim={st['dim']}_expected={expected_dim}")
    if st["has_nan"]:
        return ("BLOCKED_INVALID_VECTOR", "vector_contains_nan")
    if st["has_inf"]:
        return ("BLOCKED_INVALID_VECTOR", "vector_contains_inf")
    if st["is_zero"]:
        return ("BLOCKED_INVALID_VECTOR", "zero_vector")
    return ("VALID_REVIEW_ONLY", "")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return dot / (na * nb)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# PCA (numpy with pure-python fallback)
# ---------------------------------------------------------------------------

def pca_2d(vectors: list[list[float]]) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """Projeta os vetores em 2D. Devolve (coordenadas, (evr_x, evr_y))."""
    n = len(vectors)
    if n == 0:
        return ([], (0.0, 0.0))
    try:
        import numpy as np  # local import keeps the dependency optional
        x = np.asarray(vectors, dtype=float)
        x = x - x.mean(axis=0, keepdims=True)
        # Economy SVD is numerically stable for tall/wide matrices alike.
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        comp = vt[:2] if vt.shape[0] >= 2 else np.vstack([vt, np.zeros((2 - vt.shape[0], vt.shape[1]))])
        proj = x @ comp.T
        total = float((s ** 2).sum()) or 1.0
        evr = (s ** 2) / total
        ex = float(evr[0]) if evr.shape[0] > 0 else 0.0
        ey = float(evr[1]) if evr.shape[0] > 1 else 0.0
        coords = [(float(r[0]), float(r[1] if proj.shape[1] > 1 else 0.0)) for r in proj]
        return (coords, (ex, ey))
    except Exception:
        return _pca_2d_pure(vectors)


def _pca_2d_pure(vectors: list[list[float]]) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    n = len(vectors)
    d = len(vectors[0])
    mean = [sum(col) / n for col in zip(*vectors)]
    centered = [[v[j] - mean[j] for j in range(d)] for v in vectors]

    def cov_matvec(w: list[float]) -> list[float]:
        # (X^T X) w without materializing the full covariance matrix.
        out = [0.0] * d
        for row in centered:
            s = sum(row[j] * w[j] for j in range(d))
            for j in range(d):
                out[j] += row[j] * s
        return out

    def power_iter(deflate: list[tuple[list[float], float]]) -> tuple[list[float], float]:
        w = [1.0 / math.sqrt(d)] * d
        lam = 0.0
        for _ in range(60):
            v = cov_matvec(w)
            for vec, l in deflate:
                proj = sum(vec[j] * w[j] for j in range(d))
                for j in range(d):
                    v[j] -= l * proj * vec[j]
            nrm = math.sqrt(sum(c * c for c in v))
            if nrm == 0.0:
                break
            w = [c / nrm for c in v]
            lam = nrm
        return (w, lam)

    pc1, l1 = power_iter([])
    pc2, l2 = power_iter([(pc1, l1)])
    total = 0.0
    for row in centered:
        total += sum(c * c for c in row)
    total = total or 1.0
    coords = [
        (sum(row[j] * pc1[j] for j in range(d)), sum(row[j] * pc2[j] for j in range(d)))
        for row in centered
    ]
    return (coords, (l1 / total, l2 / total))


# ---------------------------------------------------------------------------
# Deterministic k-means (numpy with pure-python fallback)
# ---------------------------------------------------------------------------

def kmeans_simple(vectors: list[list[float]], k: int, iters: int = 25) -> list[int]:
    """k-means determinístico: semeia em pontos igualmente espaçados, sem aleatoriedade.

    Determinismo aqui não é preciosismo — é o que permite reproduzir o agrupamento entre
    execuções e entre máquinas.
    """
    n = len(vectors)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    try:
        import numpy as np
        x = np.asarray(vectors, dtype=float)
        idx = [round(i * (n - 1) / max(k - 1, 1)) for i in range(k)]
        centers = x[idx].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new = d.argmin(axis=1)
            if np.array_equal(new, labels):
                labels = new
                break
            labels = new
            for c in range(k):
                pts = x[labels == c]
                if len(pts):
                    centers[c] = pts.mean(axis=0)
        return [int(v) for v in labels]
    except Exception:
        return _kmeans_pure(vectors, k, iters)


def _kmeans_pure(vectors: list[list[float]], k: int, iters: int) -> list[int]:
    n = len(vectors)
    d = len(vectors[0])
    idx = [round(i * (n - 1) / max(k - 1, 1)) for i in range(k)]
    centers = [list(vectors[i]) for i in idx]
    labels = [0] * n
    for _ in range(iters):
        changed = False
        for i, v in enumerate(vectors):
            best, bestd = 0, float("inf")
            for c, cen in enumerate(centers):
                dist = sum((v[j] - cen[j]) ** 2 for j in range(d))
                if dist < bestd:
                    best, bestd = c, dist
            if labels[i] != best:
                changed = True
            labels[i] = best
        for c in range(k):
            members = [vectors[i] for i in range(n) if labels[i] == c]
            if members:
                centers[c] = [sum(col) / len(members) for col in zip(*members)]
        if not changed:
            break
    return labels


# ---------------------------------------------------------------------------
# Fixture / synthetic detection
# ---------------------------------------------------------------------------

def is_fixture_or_synthetic(text: str) -> bool:
    low = (text or "").lower()
    return any(t in low for t in FIXTURE_TERMS)


def is_fixture_patch(patch_id: str) -> bool:
    return bool(FIXTURE_PATCH_RE.match((patch_id or "").strip().upper()))


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def assert_no_forbidden_true(rows: list[dict[str, Any]], rotulo: str) -> None:
    for i, row in enumerate(rows):
        for field in FORBIDDEN_TRUE_FIELDS:
            if str(row.get(field, "false")).strip().lower() == "true":
                raise ValueError(
                    f"GUARDRAIL VIOLATION in {rotulo} row {i}: {field}=true is forbidden"
                )


def scan_text_for_forbidden(text: str) -> list[str]:
    found = []
    low = (text or "").lower()
    for tok in FORBIDDEN_TRUE_TOKENS:
        if tok in low:
            found.append(tok)
    return found


def require_no_abs_paths(rows: list[dict[str, Any]], rotulo: str) -> None:
    for i, row in enumerate(rows):
        for k, v in row.items():
            if ABS_PATH_RE.search(str(v)):
                raise ValueError(f"Absolute path in {rotulo} row {i} field '{k}': {v!r}")


# ---------------------------------------------------------------------------
# Shared analysis loader — re-parses VALID vectors for v1pj/v1pk/v1pl
# ---------------------------------------------------------------------------

def load_valid_embeddings(root: Path, discovery_path: Path, registry_path: Path) -> list[dict[str, Any]]:
    """Reprocessa os vetores VÁLIDOS e não duplicados para a análise seguinte.

    Casa o vetor reprocessado da fonte com o registro do v1ph pelo sha256 do vetor,
    guardando só as linhas que o registro marca ``VALID_REVIEW_ONLY`` e que não são
    duplicata. Devolve dicionários com: embedding_id, patch_id, alias, regiao, vector.
    """
    registry = read_csv(registry_path)
    valid_sha: dict[str, dict[str, str]] = {}
    for r in registry:
        if r.get("embedding_status") == "VALID_REVIEW_ONLY" and r.get("is_duplicate_vector") != "true":
            valid_sha.setdefault(r.get("vector_sha256_16", ""), r)
    if not valid_sha:
        return []

    discovery = read_csv(discovery_path)
    out: list[dict[str, Any]] = []
    used: set[str] = set()
    for art in discovery:
        if art.get("likely_embedding_source") != "true" or art.get("allowed_for_dino_registry") != "true":
            continue
        path = root / art.get("relative_path", "")
        if not path.exists() or path.suffix.lower() != ".csv":
            continue
        for srow in read_csv(path):
            vec = parse_embedding_from_row(srow)
            if vec is None:
                continue
            sha = sha256_short(",".join(f"{x:.6g}" for x in vec))
            if sha not in valid_sha or sha in used:
                continue
            used.add(sha)
            reg = valid_sha[sha]
            out.append({
                "embedding_id": reg.get("embedding_id", ""),
                "patch_id": reg.get("patch_id", "UNKNOWN_PATCH"),
                "alias": reg.get("alias", ""),
                "regiao": reg.get("regiao", "UNKNOWN"),
                "vector": vec,
            })
    return out


# ---------------------------------------------------------------------------
# Env-overridable path helper
# ---------------------------------------------------------------------------

def _p(env: str, default: Path) -> Path:
    return Path(os.environ[env]) if env in os.environ else default


def source_root() -> Path:
    """Raiz usada para resolver o caminho do artefato de origem (sobrescrevível em teste)."""
    return Path(os.environ["REVP_DINO_SOURCE_ROOT"]) if "REVP_DINO_SOURCE_ROOT" in os.environ else ROOT


def _f(value: float, nd: int = 6) -> str:
    """Format a float for CSV; NaN/inf rendered as explicit tokens."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf"
    return f"{value:.{nd}f}"
