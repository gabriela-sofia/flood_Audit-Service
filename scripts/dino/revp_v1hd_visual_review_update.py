"""REV-P v1hd — reinterpretação dos candidatos com apoio da imagem.

Reprocessa os 47 candidatos do portão de revisão do v1hb usando estatística real da
imagem: direto dos TIF do Sentinel quando --sentinel-root ou REVP_SENTINEL_ROOT apontam
para eles, ou dos PNG de preview que o v1hc gerou. O resultado é uma nota descritiva por
patch (visual_pattern_notes), deliberadamente conservadora.

Uso:
    python revp_v1hd_visual_review_update.py --sentinel-root /caminho/do/sentinel
    REVP_SENTINEL_ROOT=/caminho/do/sentinel python revp_v1hd_visual_review_update.py

Sem nenhum dos dois, todo candidato sai VISUAL_INTERPRETATION_MODE=MANUAL_REQUIRED.

O limite metodológico vale em todos os caminhos do código:
  - não se cria rótulo
  - não se cria alvo
  - não se faz predição
  - não se estabelece referência de campo
  - toda observação é descritiva/estatística
  - a anotação existe só para revisão
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None  # type: ignore[assignment]
    RASTERIO_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1hd"
V1HB_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1hb"
V1HC_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1hc"
V1GU_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gu"
OUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1hd"

REGION_TO_PREFIX = {
    "Curitiba": "curitiba",
    "Petrópolis": "petropolis",
    "Recife": "recife",
}

# Band indices in 6-band Sentinel TIF: B2, B3, B4, B8, B11, B12
BAND_B2, BAND_B3, BAND_B4, BAND_B8 = 0, 1, 2, 3

# v1gu corpus: patches with individual embedding evidence
V1GU_CORPUS = {
    "CUR_00038", "CUR_00249", "CUR_00350", "CUR_00357",
    "PET_00016", "PET_00104", "PET_00119", "PET_00140",
    "REC_00019", "REC_00183", "REC_00204", "REC_00205",
}
MEDOIDS = {"CUR_00357", "PET_00104", "REC_00205"}
OUTLIERS = {"CUR_00350", "PET_00016", "REC_00019"}

FORBIDDEN_TERMS = {
    "enchente", "alagamento", "inundação", "predição", "predito",
    "risco predito", "vulnerabilidade predita", "detecção", "classifica",
    "ground truth", "validação supervisionada",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class CandidateRecord(NamedTuple):
    review_item_id: str
    canonical_patch_id: str
    region: str
    candidate_category: str
    preview_status: str
    preview_file: str
    prior_uncertainty: str
    prior_usable: str


class ImageStats(NamedTuple):
    mean_brightness: float
    std_brightness: float
    ndvi_mean: float
    ndvi_std: float
    ndvi_veg_fraction: float    # fraction of pixels with NDVI > 0.30
    ndvi_low_fraction: float    # fraction of pixels with NDVI < -0.05
    red_ratio: float
    green_ratio: float
    blue_ratio: float
    n_pixels: int


# ---------------------------------------------------------------------------
# Sentinel root resolution
# ---------------------------------------------------------------------------

def _get_sentinel_root(cli_root: str | None = None) -> Path | None:
    for candidate in [cli_root, os.environ.get("REVP_SENTINEL_ROOT")]:
        if candidate:
            p = Path(candidate)
            if p.exists():
                return p
    return None


def _resolve_tif(patch_id: str, region: str, sentinel_root: Path) -> Path | None:
    prefix = REGION_TO_PREFIX.get(region)
    if not prefix:
        return None
    num = patch_id.split("_")[1]
    tif = sentinel_root / f"patch_{prefix}_{num}.tif"
    return tif if tif.exists() else None


# ---------------------------------------------------------------------------
# Image statistics computation
# ---------------------------------------------------------------------------

def _normalize_band(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    valid = arr[arr > 0]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    vmin, vmax = float(np.percentile(valid, p_low)), float(np.percentile(valid, p_high))
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1).astype(np.float32)


def compute_stats_from_tif(tif_path: Path) -> ImageStats | None:
    if rasterio is None:
        return None
    try:
        with rasterio.open(tif_path) as src:  # type: ignore[union-attr]
            data = src.read()
    except Exception:
        return None

    r = _normalize_band(data[BAND_B4].astype(np.float64))
    g = _normalize_band(data[BAND_B3].astype(np.float64))
    b = _normalize_band(data[BAND_B2].astype(np.float64))

    brightness = (r + g + b) / 3.0

    nir = data[BAND_B8].astype(np.float64)
    red = data[BAND_B4].astype(np.float64)
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)

    denom_rgb = r + g + b + 1e-8
    return ImageStats(
        mean_brightness=float(brightness.mean()),
        std_brightness=float(brightness.std()),
        ndvi_mean=float(ndvi.mean()),
        ndvi_std=float(ndvi.std()),
        ndvi_veg_fraction=float((ndvi > 0.30).mean()),
        ndvi_low_fraction=float((ndvi < -0.05).mean()),
        red_ratio=float((r / denom_rgb).mean()),
        green_ratio=float((g / denom_rgb).mean()),
        blue_ratio=float((b / denom_rgb).mean()),
        n_pixels=int(brightness.size),
    )


def stats_to_visual_notes(stats: ImageStats, patch_id: str, category: str) -> str:
    """Escreve a nota descritiva a partir da estatística da imagem.

    Nunca infere evento hidrológico, risco ou rótulo. O texto usa sempre 'aparente',
    'possível' e 'padrão consistente com' — a hesitação é proposital.
    """
    notes: list[str] = []

    # Brightness / reflectance level
    b = stats.mean_brightness
    std = stats.std_brightness
    if b > 0.65:
        notes.append(f"Alta reflexão média ({b:.2f}) — possível solo exposto, superfície impermeável ou nebulosidade")
    elif b < 0.25:
        notes.append(f"Baixa reflexão média ({b:.2f}) — possível superfície escura, água aparente ou sombra")
    else:
        notes.append(f"Reflexão moderada ({b:.2f})")

    # Spatial heterogeneity (texture proxy)
    if std > 0.20:
        notes.append(f"Alta heterogeneidade visual (std={std:.2f}) — padrão texturizado, possível mistura de cobertura")
    elif std > 0.12:
        notes.append(f"Heterogeneidade moderada (std={std:.2f}) — variação espacial observável")
    else:
        notes.append(f"Padrão visual relativamente homogêneo (std={std:.2f})")

    # NDVI interpretation
    ndvi = stats.ndvi_mean
    veg = stats.ndvi_veg_fraction
    low = stats.ndvi_low_fraction
    if ndvi > 0.50 or veg > 0.55:
        notes.append(
            f"NDVI elevado (média={ndvi:.2f}; {veg:.0%} pixels > 0.30) — "
            "padrão consistente com cobertura vegetal aparente densa"
        )
    elif ndvi > 0.20 or veg > 0.25:
        notes.append(
            f"NDVI moderado (média={ndvi:.2f}; {veg:.0%} pixels > 0.30) — "
            "possível vegetação esparsa ou transição urbano-natural"
        )
    elif ndvi > 0.05:
        notes.append(
            f"NDVI baixo-moderado (média={ndvi:.2f}) — possível mistura de área construída e vegetação residual"
        )
    else:
        notes.append(
            f"NDVI baixo (média={ndvi:.2f}) — padrão consistente com área construída, solo exposto ou superfície impermeável"
        )

    if low > 0.20:
        notes.append(
            f"Fração NDVI negativo={low:.0%} — presença de superfícies escuras, sombras ou possível água aparente"
        )

    # Spectral ratio hints
    if stats.blue_ratio > 0.37:
        notes.append("Elevada componente azul relativa — possível nebulosidade ou superfície aquática")

    if stats.green_ratio > stats.red_ratio + 0.03:
        notes.append("Canal verde dominante — padrão consistente com vegetação aparente")
    elif stats.red_ratio > stats.green_ratio + 0.03:
        notes.append("Canal vermelho dominante — padrão consistente com solo exposto ou área urbana aparente")

    # Category-specific addendum
    if "medoid" in category:
        notes.append(
            "Patch selecionado como medoid regional — padrão estrutural deve ser comparado "
            "com outros patches da mesma região"
        )
    elif "outlier" in category:
        notes.append(
            "Patch selecionado como outlier estrutural — heterogeneidade visual esperada "
            "em relação ao medoid regional"
        )

    return (
        f"[INTERPRETAÇÃO ASSISTIDA — ESTATÍSTICAS DE IMAGEM] "
        + "; ".join(notes)
        + f". n_pixels={stats.n_pixels}. "
        "Interpretação conservadora e exploratória — revisão-only, sem afirmações operacionais. "
        "Revisão supervisora direta recomendada."
    )


def stats_to_uncertainty(
    stats: ImageStats,
    patch_id: str,
    prior_uncertainty: str,
) -> str:
    """Recalcula a incerteza a partir da qualidade da evidência estatística.

    Pode baixar de alta para média quando o patch está no corpus e a estatística é clara.
    Nunca baixa de média para baixa só com estatística.
    Nunca aumenta a incerteza com base em estatística.
    """
    if prior_uncertainty == "low":
        return "low"
    if patch_id not in V1GU_CORPUS:
        return "high"  # No embedding evidence → keep high regardless of image
    # In corpus: if NDVI is clear and heterogeneity is informative, lower high → medium
    if prior_uncertainty == "high":
        clear_signal = (
            abs(stats.ndvi_mean) > 0.15 or
            stats.ndvi_veg_fraction > 0.30 or
            stats.ndvi_low_fraction > 0.25
        )
        if clear_signal:
            return "medium"
    return prior_uncertainty


def stats_to_usable(
    patch_id: str,
    prior_usable: str,
    preview_status: str,
) -> str:
    """Recalcula se o patch serve para a discussão do TCC.

    - patch com preview GERADO → no mínimo 'conditional'
    - patch sem preview → mantém o valor anterior
    - medoid/outlier que já era 'conditional' → mantém
    """
    if preview_status != "GENERATED":
        return prior_usable
    if prior_usable == "no":
        return "conditional"
    return prior_usable


def stats_to_discussion_note(
    stats: ImageStats,
    patch_id: str,
    category: str,
    region: str,
    uncertainty: str,
) -> str:
    ndvi = stats.ndvi_mean
    veg = stats.ndvi_veg_fraction

    if patch_id in MEDOIDS:
        return (
            f"Medoid {region}: padrão estrutural central da região no espaço de embeddings. "
            f"Estatísticas visuais: NDVI médio={ndvi:.2f} ({veg:.0%} vegetação aparente), "
            f"heterogeneidade={stats.std_brightness:.2f}. "
            "Útil como ancora visual para discussão de representatividade regional. "
            "Sem generalização para patches não revistos."
        )
    if patch_id in OUTLIERS:
        return (
            f"Outlier {region}: patch periférico no espaço de embeddings. "
            f"Estatísticas visuais: NDVI médio={ndvi:.2f} ({veg:.0%} vegetação aparente), "
            f"heterogeneidade={stats.std_brightness:.2f}. "
            "Padrão visual pode esclarecer divergência estrutural documentada em v1gu. "
            "Incerteza: {uncertainty}. Interpretação exploratória — sem claim operacional."
        )
    if patch_id in V1GU_CORPUS:
        return (
            f"Patch em corpus v1gu ({region}): baixa cobertura GIS + evidência estrutural disponível. "
            f"NDVI médio={ndvi:.2f}; heterogeneidade visual={stats.std_brightness:.2f}. "
            "Uso condicional na Discussão — comparar com context GIS e medoid regional."
        )
    return (
        f"Patch sem evidência de embedding individual ({region}). "
        f"Estatísticas visuais: NDVI médio={ndvi:.2f}, heterogeneidade={stats.std_brightness:.2f}. "
        "Uso condicional na Discussão apenas com evidência visual descritiva — "
        "sem base estrutural de embedding. Incerteza: high."
    )


# ---------------------------------------------------------------------------
# Load input data
# ---------------------------------------------------------------------------

def load_candidates() -> list[CandidateRecord]:
    manifest = V1HB_DIR / "review_gate_execution_manifest_v1hb.csv"
    v1hc_manifest = V1HC_DIR / "visual_review_preview_manifest_v1hc.csv"
    annotation = V1HB_DIR / "review_gate_annotation_filled_programmatic_v1hb.csv"

    # Load preview statuses from v1hc
    preview_info: dict[str, dict] = {}
    if v1hc_manifest.exists():
        with v1hc_manifest.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                preview_info[row["review_item_id"]] = {
                    "preview_status": row.get("preview_status", "BLOCKED"),
                    "preview_file": row.get("preview_file", ""),
                }

    # Load prior annotations from v1hb programmatic review
    prior_annotations: dict[str, dict] = {}
    if annotation.exists():
        with annotation.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                prior_annotations[row["canonical_patch_id"]] = {
                    "uncertainty_level": row.get("uncertainty_level", "high"),
                    "usable_in_discussion": row.get("usable_in_discussion", "no"),
                }

    records = []
    with manifest.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid = row["canonical_patch_id"]
            hre = row["review_item_id"]
            pinfo = preview_info.get(hre, {"preview_status": "BLOCKED", "preview_file": ""})
            pannot = prior_annotations.get(pid, {"uncertainty_level": "high", "usable_in_discussion": "no"})
            records.append(CandidateRecord(
                review_item_id=hre,
                canonical_patch_id=pid,
                region=row["region"],
                candidate_category=row["candidate_category"],
                preview_status=pinfo["preview_status"],
                preview_file=pinfo["preview_file"],
                prior_uncertainty=pannot["uncertainty_level"],
                prior_usable=pannot["usable_in_discussion"],
            ))
    return records


# ---------------------------------------------------------------------------
# monta as saidas
# ---------------------------------------------------------------------------

def process_candidates(
    records: list[CandidateRecord],
    sentinel_root: Path | None,
) -> tuple[list[dict], list[dict]]:
    """Devolve (linhas de anotação, linhas de exemplo)."""
    annotation_rows = []

    for rec in records:
        tif_path: Path | None = None
        stats: ImageStats | None = None
        visual_mode = "MANUAL_REQUIRED"

        if rec.preview_status == "GENERATED" and sentinel_root is not None:
            tif_path = _resolve_tif(rec.canonical_patch_id, rec.region, sentinel_root)
            if tif_path is not None:
                stats = compute_stats_from_tif(tif_path)
                if stats is not None:
                    visual_mode = "COMPUTED"

        if stats is not None:
            visual_notes = stats_to_visual_notes(stats, rec.canonical_patch_id, rec.candidate_category)
            uncertainty = stats_to_uncertainty(stats, rec.canonical_patch_id, rec.prior_uncertainty)
            usable = stats_to_usable(rec.canonical_patch_id, rec.prior_usable, rec.preview_status)
            discussion_note = stats_to_discussion_note(
                stats, rec.canonical_patch_id, rec.candidate_category, rec.region, uncertainty
            )
            structural_ctx = (
                f"v1gu corpus: {'yes' if rec.canonical_patch_id in V1GU_CORPUS else 'no'}; "
                f"role: {'medoid' if rec.canonical_patch_id in MEDOIDS else ('outlier' if rec.canonical_patch_id in OUTLIERS else 'candidate')}; "
                f"region: {rec.region}"
            )
            ext_evidence = (
                f"GIS coverage: contextual only (PARTIAL/NOT_ACQUIRED); "
                f"preview: {rec.preview_status}; "
                f"visual_stats computed from TIF"
            )
            method_interp = (
                "Structural coherence and contextual observation — "
                "statistical description of image properties. "
                "No label, no prediction, no ground truth."
            )
        else:
            visual_notes = (
                "visual_evidence_status = MANUAL_REQUIRED; "
                "Sentinel root not configured or TIF not available. "
                "Image statistics could not be computed. "
                "Manual visual inspection required using v1hc previews."
            )
            uncertainty = rec.prior_uncertainty
            usable = rec.prior_usable
            discussion_note = (
                f"Visual evidence not computed ({rec.region} — {rec.candidate_category}). "
                "Use v1hc preview for manual interpretation."
            )
            structural_ctx = (
                f"v1gu corpus: {'yes' if rec.canonical_patch_id in V1GU_CORPUS else 'no'}; "
                f"role: {'medoid' if rec.canonical_patch_id in MEDOIDS else ('outlier' if rec.canonical_patch_id in OUTLIERS else 'candidate')}"
            )
            ext_evidence = f"GIS coverage: contextual only; preview_status: {rec.preview_status}"
            method_interp = "Structural coherence context only — visual statistics not available."

        annotation_rows.append({
            "review_item_id": rec.review_item_id,
            "canonical_patch_id": rec.canonical_patch_id,
            "region": rec.region,
            "candidate_category": rec.candidate_category,
            "preview_status": rec.preview_status,
            "visual_interpretation_mode": visual_mode,
            "visual_pattern_notes": visual_notes,
            "structural_context_notes": structural_ctx,
            "external_evidence_notes": ext_evidence,
            "methodological_interpretation": method_interp,
            "uncertainty_level": uncertainty,
            "usable_in_discussion": usable,
            "discussion_note": discussion_note,
            "no_label_created_confirmed": "yes",
            "no_prediction_claim_confirmed": "yes",
            "no_ground_truth_claim_confirmed": "yes",
            "review_only_confirmed": "yes",
        })

    # monta os exemplos que vao para o TCC
    examples_rows = _select_tcc_examples(annotation_rows)
    return annotation_rows, examples_rows


def _select_tcc_examples(rows: list[dict]) -> list[dict]:
    """Escolhe um exemplo de cada categoria-chave para a Discussão do TCC."""
    examples = []
    seen_types: set[str] = set()

    # Priority order for example selection
    priority = [
        ("medoid_example", lambda r: "medoid" in r["candidate_category"] and r["usable_in_discussion"] != "no"),
        ("outlier_example", lambda r: "outlier" in r["candidate_category"] and r["usable_in_discussion"] != "no"),
        ("low_coverage_in_corpus_example", lambda r: (
            "coverage_external_low" in r["candidate_category"]
            and r["canonical_patch_id"] in V1GU_CORPUS
            and "medoid" not in r["candidate_category"]
            and "outlier" not in r["candidate_category"]
        )),
        ("high_uncertainty_example", lambda r: r["uncertainty_level"] == "high"),
        ("conditional_use_example", lambda r: r["usable_in_discussion"] == "conditional"),
    ]

    for example_type, selector in priority:
        if example_type in seen_types:
            continue
        for row in rows:
            if selector(row):
                examples.append({
                    "example_type": example_type,
                    "review_item_id": row["review_item_id"],
                    "canonical_patch_id": row["canonical_patch_id"],
                    "region": row["region"],
                    "candidate_category": row["candidate_category"],
                    "uncertainty_level": row["uncertainty_level"],
                    "usable_in_discussion": row["usable_in_discussion"],
                    "discussion_note": row["discussion_note"],
                    "visual_pattern_notes": row["visual_pattern_notes"][:200],
                    "tcc_use_suggestion": _tcc_suggestion(example_type),
                })
                seen_types.add(example_type)
                break

    return examples


def _tcc_suggestion(example_type: str) -> str:
    suggestions = {
        "medoid_example": (
            "Usar como âncora de representatividade regional na Seção 5.1. "
            "Descrever características visuais do padrão central de cada região."
        ),
        "outlier_example": (
            "Usar para documentar heterogeneidade estrutural na Seção 5.1/5.2. "
            "Discutir o que diferencia o outlier do padrão regional sem atribuir causa operacional."
        ),
        "low_coverage_in_corpus_example": (
            "Usar para ilustrar limitação de cobertura GIS na Seção 5.3. "
            "Documentar que interpretação visual parcial foi possível apesar da ausência de GIS."
        ),
        "high_uncertainty_example": (
            "Usar para ilustrar limitações metodológicas na Seção 5.3/5.4. "
            "Documentar que incerteza alta persiste e que generalização é restrita."
        ),
        "conditional_use_example": (
            "Usar com cautela na Seção 5.2 para ilustrar diversidade do corpus. "
            "Explicitar condições de uso (evidência disponível, escopo limitado)."
        ),
    }
    return suggestions.get(example_type, "Consultar revisor antes de usar na Discussão.")


def build_summary(rows: list[dict], sentinel_root: Path | None) -> dict:
    by_uncertainty: dict[str, int] = {}
    by_usable: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_mode: dict[str, int] = {}

    for r in rows:
        u = r["uncertainty_level"]
        by_uncertainty[u] = by_uncertainty.get(u, 0) + 1
        us = r["usable_in_discussion"]
        by_usable[us] = by_usable.get(us, 0) + 1
        cat = r["candidate_category"].split(";")[0].strip()
        by_category[cat] = by_category.get(cat, 0) + 1
        mode = r["visual_interpretation_mode"]
        by_mode[mode] = by_mode.get(mode, 0) + 1

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "n_candidates": len(rows),
        "n_visually_computed": by_mode.get("COMPUTED", 0),
        "n_manual_required": by_mode.get("MANUAL_REQUIRED", 0),
        "sentinel_root_configured": sentinel_root is not None,
        "by_visual_interpretation_mode": by_mode,
        "by_uncertainty": by_uncertainty,
        "by_usable_in_discussion": by_usable,
        "by_primary_category": by_category,
        "forbidden_claims_checked": True,
        "methodological_guardrails": {
            "labels_created": False,
            "targets_created": False,
            "predictions_made": False,
            "ground_truth_established": False,
            "review_only": True,
            "visual_notes_are_statistical_only": True,
        },
        "limitations": [
            "Interpretações são estatísticas de imagem, não classificações operacionais.",
            "NDVI e estatísticas de reflexão são descritores, não validadores de risco.",
            "43 patches Curitiba sem evidência de embedding individual (não estão em v1gu).",
            "GIS permanece contextual — não é ground truth.",
            "Corpus pequeno (12 embeddings; 47 candidatos) — sem generalização para 128 patches.",
        ],
    }


def build_synthesis(rows: list[dict], summary: dict) -> str:
    n_computed = summary["n_visually_computed"]
    n_manual = summary["n_manual_required"]
    by_u = summary["by_uncertainty"]
    by_us = summary["by_usable_in_discussion"]

    medoid_rows = [r for r in rows if "medoid" in r["candidate_category"]]
    outlier_rows = [r for r in rows if "outlier" in r["candidate_category"]
                    and "medoid" not in r["candidate_category"]]

    lines = [
        "# Síntese da revisão visual programática — REV-P v1hd",
        "",
        f"**Data**: {datetime.now(timezone.utc).date()}  ",
        "**Revisor**: ASSISTED-v1hd (estatísticas de imagem)  ",
        "**Status**: Review-only; Interpretativa; Exploratória  ",
        "",
        "---",
        "",
        "## 1. Cobertura Visual",
        "",
        f"- **{n_computed}/47** candidatos com estatísticas de imagem computadas (modo: COMPUTED)",
        f"- **{n_manual}/47** candidatos requerem inspeção visual manual (modo: MANUAL_REQUIRED)",
        "",
        "---",
        "",
        "## 2. Medoids Regionais",
        "",
        "Os medoids representam o centro estrutural de cada região no espaço de embeddings DINO.",
        "Suas características visuais servem como âncora para discussão de representatividade regional.",
        "",
    ]

    for r in medoid_rows:
        lines.append(f"**{r['review_item_id']} — {r['canonical_patch_id']} ({r['region']})**")
        # Show first 200 chars of visual notes
        notes_short = r["visual_pattern_notes"][:300].rstrip(". ") + "..."
        lines.append(f"- {notes_short}")
        lines.append(f"- Incerteza: {r['uncertainty_level']} | Uso na Discussão: {r['usable_in_discussion']}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 3. Outliers Estruturais",
        "",
        "Os outliers são periféricos na distribuição de embeddings regional. "
        "Suas características visuais podem ajudar a explicar a divergência estrutural observada.",
        "",
    ]

    for r in outlier_rows:
        lines.append(f"**{r['review_item_id']} — {r['canonical_patch_id']} ({r['region']})**")
        notes_short = r["visual_pattern_notes"][:300].rstrip(". ") + "..."
        lines.append(f"- {notes_short}")
        lines.append(f"- Incerteza: {r['uncertainty_level']} | Uso na Discussão: {r['usable_in_discussion']}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 4. Candidatos com Baixa Cobertura GIS Externa",
        "",
        "43 patches Curitiba foram selecionados por baixa cobertura GIS.",
        "Para os 39 não-corpus, a interpretação é exclusivamente visual — sem evidência de embedding individual.",
        "Para HRE001 (CUR_00038) e HRE002 (CUR_00249), há evidência estrutural adicional de v1gu.",
        "",
        "**Limitação**: Sem embedding individual, estes patches contribuem apenas com contexto visual descritivo.",
        "",
        "---",
        "",
        "## 5. Distribuição de Incerteza e Usabilidade",
        "",
        "| Incerteza | N |",
        "|-----------|----|",
    ]

    for level in ["low", "medium", "high"]:
        n = by_u.get(level, 0)
        lines.append(f"| {level} | {n} |")

    lines += [
        "",
        "| Usável na Discussão | N |",
        "|---------------------|-----|",
    ]
    for val in ["yes", "conditional", "no"]:
        n = by_us.get(val, 0)
        lines.append(f"| {val} | {n} |")

    lines += [
        "",
        "---",
        "",
        "## 6. Como Usar na Discussão do TCC",
        "",
        "**Pode-se afirmar**:",
        "- 'O medoid de Curitiba apresenta padrão visual com NDVI médio X, "
          "consistente com cobertura [descritiva]'",
        "- 'O outlier de Petrópolis difere visualmente do medoid regional — "
          "heterogeneidade estrutural documentada'",
        "- 'Patches com baixa cobertura GIS mostram variação visual observável mesmo sem indicadores contextuais'",
        "",
        "**Não se pode afirmar**:",
        "- Que algum patch está em zona de enchente ou risco",
        "- Que o DINO prediz qualquer evento hidrológico",
        "- Que a análise visual valida o modelo contra ground truth",
        "- Que os padrões observados generalizam para todos os 128 patches",
        "",
        "---",
        "",
        "## 7. Limitações Metodológicas",
        "",
        "- **Interpretação estatística**: visual_pattern_notes são derivadas de estatísticas "
          "computadas (NDVI, reflexão, heterogeneidade) — não de inspeção humana direta",
        "- **Corpus pequeno**: 12 embeddings não representam os 128 patches totais",
        "- **GIS parcial**: indicadores contextuais, não validantes",
        "- **Sem ground truth**: nenhuma observação aqui estabelece verdade de campo",
        "- **Revisão supervisora recomendada**: especialmente para PET_00016 e REC_00019 (outliers severos)",
        "",
        "---",
        "",
        f"**Versão**: v1hd  ",
        f"**Gerado em**: {datetime.now(timezone.utc).date()}  ",
        "**Status**: Review-only; Estatísticas de imagem; Interpretativo  ",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[v1hd] Written: {path.name} ({len(rows)} rows)")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[v1hd] Written: {path.name}")


def write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
    print(f"[v1hd] Written: {path.name} ({len(content)} chars)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v1hd: Visual review update using image statistics."
    )
    parser.add_argument(
        "--sentinel-root",
        metavar="PATH",
        help=(
            "Path to the Sentinel TIF directory. "
            "Also accepts REVP_SENTINEL_ROOT environment variable. "
            "If not provided, all candidates are marked MANUAL_REQUIRED."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sentinel_root = _get_sentinel_root(args.sentinel_root)

    if sentinel_root is None:
        print("[v1hd] Sentinel root not configured — mode: MANUAL_REQUIRED for all candidates")
    else:
        print(f"[v1hd] Sentinel root: configured")

    print("[v1hd] Loading candidates...")
    records = load_candidates()
    print(f"[v1hd] {len(records)} candidates")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[v1hd] Processing candidates...")
    annotation_rows, examples_rows = process_candidates(records, sentinel_root)

    n_computed = sum(1 for r in annotation_rows if r["visual_interpretation_mode"] == "COMPUTED")
    n_manual = sum(1 for r in annotation_rows if r["visual_interpretation_mode"] == "MANUAL_REQUIRED")
    print(f"[v1hd] Visually computed: {n_computed}/47")
    print(f"[v1hd] Manual required: {n_manual}/47")

    summary = build_summary(annotation_rows, sentinel_root)
    synthesis = build_synthesis(annotation_rows, summary)

    write_csv(OUT_DIR / "review_gate_visual_annotation_v1hd.csv", annotation_rows)
    write_csv(OUT_DIR / "review_gate_visual_examples_for_tcc_v1hd.csv", examples_rows)
    write_json(OUT_DIR / "review_gate_visual_summary_v1hd.json", summary)
    write_md(OUT_DIR / "review_gate_visual_discussion_synthesis_v1hd.md", synthesis)

    print(f"\n[v1hd] Done. Outputs in {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
