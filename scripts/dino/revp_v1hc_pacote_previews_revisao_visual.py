"""REV-P v1hc — pacote de previews visuais para a revisão humana.

Gera preview visual (RGB + NDVI) dos 47 candidatos do portão de revisão que o v1hb
selecionou, lendo os TIF originais do Sentinel apontados por --sentinel-root ou pela
variável REVP_SENTINEL_ROOT. Sai um preview por patch mais uma folha de contato por
categoria, para alguém olhar e decidir.

Nada aqui vira rótulo, classe, referência de campo ou afirmação operacional: o preview
serve à revisão humana, não ao modelo. Todo output cai em local_runs/, que não é
versionado, e o caminho do TIF de origem nunca entra em arquivo versionável — ele é
privado da máquina que rodou.

Uso:
    python revp_v1hc_pacote_previews_revisao_visual.py --sentinel-root /caminho/do/sentinel

    Ou pela variável de ambiente:
        REVP_SENTINEL_ROOT=/caminho/do/sentinel python revp_v1hc_...py

    Sem nenhum dos dois, todo candidato sai BLOCKED_SENTINEL_ROOT_NOT_CONFIGURED —
    falha fechada, não silenciosa.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None  # type: ignore[assignment]
    RASTERIO_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1hc"
V1HB_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1hb"
OUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1hc"
FIGURES_DIR = OUT_DIR / "figures"

# Band indices (0-based) in the 6-band Sentinel TIF: B2, B3, B4, B8, B11, B12
BAND_B2 = 0   # Blue
BAND_B3 = 1   # Green
BAND_B4 = 2   # Red
BAND_B8 = 3   # NIR (for NDVI)

REGION_TO_PREFIX = {
    "Curitiba": "curitiba",
    "Petrópolis": "petropolis",
    "Recife": "recife",
}

FORBIDDEN_REVIEW_TERMS = {
    "prediction", "predictive", "detect", "detection", "classify", "classification",
    "risk", "vulnerability", "ground truth", "ground-truth", "rotulo",
    "accuracy", "performance", "train", "supervised", "target", "causal",
}


class PatchRecord(NamedTuple):
    review_item_id: str
    canonical_patch_id: str
    regiao: str
    candidate_category: str
    uncertainty_level: str
    usable_in_discussion: str


def load_manifest() -> list[PatchRecord]:
    manifest_path = V1HB_DIR / "review_gate_execution_manifest_v1hb.csv"
    annotation_path = V1HB_DIR / "review_gate_annotation_filled_programmatic_v1hb.csv"

    uncertainty_map: dict[str, str] = {}
    usable_map: dict[str, str] = {}
    if annotation_path.exists():
        with annotation_path.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                pid = row["canonical_patch_id"]
                uncertainty_map[pid] = row.get("uncertainty_level", "high")
                usable_map[pid] = row.get("usable_in_discussion", "no")

    records = []
    with manifest_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid = row["canonical_patch_id"]
            records.append(PatchRecord(
                review_item_id=row["review_item_id"],
                canonical_patch_id=pid,
                regiao=row["regiao"],
                candidate_category=row["candidate_category"],
                uncertainty_level=uncertainty_map.get(pid, "high"),
                usable_in_discussion=usable_map.get(pid, "no"),
            ))
    return records


def resolve_tif_path(patch_id: str, regiao: str, sentinel_root: Path) -> Path | None:
    prefix = REGION_TO_PREFIX.get(regiao)
    if prefix is None:
        return None
    num = patch_id.split("_")[1]
    tif_path = sentinel_root / f"patch_{prefix}_{num}.tif"
    return tif_path if tif_path.exists() else None


def normalize_band(band: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    vmin = float(np.percentile(valid, p_low))
    vmax = float(np.percentile(valid, p_high))
    if vmax <= vmin:
        return np.zeros_like(band, dtype=np.float32)
    clipped = np.clip(band, vmin, vmax)
    return ((clipped - vmin) / (vmax - vmin)).astype(np.float32)


def load_rgb_ndvi(tif_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if rasterio is None:
        raise ImportError("rasterio is required")
    with rasterio.open(tif_path) as src:  # type: ignore[union-attr]
        data = src.read()

    r = normalize_band(data[BAND_B4].astype(np.float32))
    g = normalize_band(data[BAND_B3].astype(np.float32))
    b = normalize_band(data[BAND_B2].astype(np.float32))
    rgb = np.stack([r, g, b], axis=-1)

    nir = data[BAND_B8].astype(np.float64)
    red = data[BAND_B4].astype(np.float64)
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0).astype(np.float32)

    return rgb, ndvi


def _category_short(category: str) -> str:
    if "medoid" in category:
        return "MEDOID"
    if "outlier" in category:
        return "OUTLIER"
    return "COV-LOW"


def generate_patch_preview(
    record: PatchRecord,
    tif_path: Path,
    figures_dir: Path,
) -> Path | None:
    try:
        rgb, ndvi = load_rgb_ndvi(tif_path)
    except Exception:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.8))
    fig.patch.set_facecolor("#1a1a2e")

    ax_rgb, ax_ndvi = axes

    ax_rgb.imshow(rgb)
    ax_rgb.set_title(
        f"{record.review_item_id} · {record.canonical_patch_id}\n"
        f"{record.regiao} · {_category_short(record.candidate_category)}",
        fontsize=6, color="white", pad=3,
    )
    ax_rgb.axis("off")

    im = ax_ndvi.imshow(ndvi, cmap="RdYlGn", vmin=-0.3, vmax=0.8)
    ax_ndvi.set_title("NDVI", fontsize=6, color="white", pad=3)
    ax_ndvi.axis("off")
    plt.colorbar(im, ax=ax_ndvi, fraction=0.046, pad=0.04).ax.tick_params(
        labelsize=5, colors="white"
    )

    fig.text(
        0.5, 0.01,
        "Review-only | No rotulo | No prediction",
        ha="center", va="bottom", fontsize=4.5, color="#888888",
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"preview_{record.review_item_id}_{record.canonical_patch_id}_v1hc.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def build_contact_sheet(
    records: list[PatchRecord],
    preview_map: dict[str, Path],
    output_path: Path,
    title: str,
    ncols: int = 6,
) -> bool:
    available = [r for r in records if r.canonical_patch_id in preview_map]
    if not available:
        return False

    nrows = max(1, (len(available) + ncols - 1) // ncols)
    fig_w = ncols * 2.2
    fig_h = nrows * 2.0 + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d0d1a")
    fig.suptitle(
        title,
        fontsize=9, color="white", y=0.99, va="top",
    )

    for idx, record in enumerate(available):
        row = idx // ncols
        col = idx % ncols
        rect = (col / ncols + 0.005, (nrows - row - 1) / nrows + 0.02,
                1 / ncols - 0.012, 1 / nrows - 0.06)
        ax = fig.add_axes(rect)
        try:
            img = plt.imread(str(preview_map[record.canonical_patch_id]))
            ax.imshow(img)
        except Exception:
            ax.set_facecolor("#333355")
            ax.text(0.5, 0.5, "ERR", ha="center", va="center",
                    color="red", fontsize=6, transform=ax.transAxes)
        rotulo = f"{record.review_item_id}\n{record.canonical_patch_id}"
        ax.set_title(rotulo, fontsize=4.5, color="#cccccc", pad=1)
        ax.axis("off")

    # Fill unused slots
    n_used = len(available)
    n_slots = nrows * ncols
    for idx in range(n_used, n_slots):
        row = idx // ncols
        col = idx % ncols
        rect = (col / ncols + 0.005, (nrows - row - 1) / nrows + 0.02,
                1 / ncols - 0.012, 1 / nrows - 0.06)
        ax = fig.add_axes(rect)
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _get_sentinel_root(cli_root: str | None = None) -> tuple[Path | None, str]:
    """Descobre a raiz dos dados Sentinel sem fixar caminho privado no código.

    Ordem de prioridade:
    1. argumento de linha de comando ``--sentinel-root``
    2. variável de ambiente ``REVP_SENTINEL_ROOT``
    3. nenhum dos dois → todo candidato sai BLOCKED_SENTINEL_ROOT_NOT_CONFIGURED
    """
    candidates = [cli_root, os.environ.get("REVP_SENTINEL_ROOT")]
    for c in candidates:
        if c:
            p = Path(c)
            if p.exists():
                return p, "OK"
            return None, f"SENTINEL_ROOT_PATH_NOT_FOUND: {p}"
    return None, "BLOCKED_SENTINEL_ROOT_NOT_CONFIGURED"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v1hc: Generate Sentinel visual previews for review candidates."
    )
    parser.add_argument(
        "--sentinel-root",
        metavar="PATH",
        help=(
            "Path to the directory containing Sentinel TIF files "
            "(patch_curitiba_XXXXX.tif, etc.). "
            "Can also be set via REVP_SENTINEL_ROOT environment variable. "
            "If not provided, all candidates are marked BLOCKED."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not RASTERIO_AVAILABLE:
        print("[v1hc] ERROR: rasterio not available. Install with: pip install rasterio")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sentinel_root, root_status = _get_sentinel_root(args.sentinel_root)
    status_display = root_status if sentinel_root is None else "OK"
    print(f"[v1hc] Sentinel root status: {status_display}")

    print(f"[v1hc] Loading v1hb manifest...")
    records = load_manifest()
    print(f"[v1hc] {len(records)} candidates")

    preview_map: dict[str, Path] = {}
    manifest_rows: list[dict] = []
    readiness_rows: list[dict] = []
    patch_index_rows: list[dict] = []

    for rec in records:
        tif_path: Path | None = None
        tif_status = "ROOT_NOT_AVAILABLE"
        preview_status = "BLOCKED"
        preview_file = ""
        limitations = ""
        readiness = "BLOCKED"

        if sentinel_root is not None:
            tif_path = resolve_tif_path(rec.canonical_patch_id, rec.regiao, sentinel_root)
            if tif_path is not None:
                tif_status = "FOUND"
            else:
                tif_status = "NOT_FOUND"

        if tif_status == "FOUND" and tif_path is not None:
            out_png = generate_patch_preview(rec, tif_path, FIGURES_DIR)
            if out_png is not None:
                preview_map[rec.canonical_patch_id] = out_png
                preview_status = "GENERATED"
                preview_file = out_png.name
                readiness = "READY_FOR_VISUAL_REVIEW"
                limitations = "Visual inspection required before drawing interpretive conclusions."
            else:
                preview_status = "FAILED"
                readiness = "PREVIEW_FAILED"
                limitations = "Preview generation failed — inspect TIF manually."
        elif tif_status == "NOT_FOUND":
            preview_status = "TIF_NOT_FOUND"
            readiness = "TIF_NOT_FOUND"
            limitations = "TIF file not found in sentinel root."
        else:
            limitations = "Sentinel root directory not available on this machine."

        cat_short = _category_short(rec.candidate_category)
        scope = (
            f"{cat_short} structural candidate — review-only visual observation; "
            "no rotulo, no prediction, no ground truth."
        )

        manifest_rows.append({
            "review_item_id": rec.review_item_id,
            "canonical_patch_id": rec.canonical_patch_id,
            "regiao": rec.regiao,
            "candidate_category": rec.candidate_category,
            "tif_found": tif_status == "FOUND",
            "tif_path_status": tif_status,
            "preview_status": preview_status,
            "preview_file": preview_file,
            "visual_review_scope": scope,
            "limitations_note": limitations,
        })

        readiness_rows.append({
            "review_item_id": rec.review_item_id,
            "canonical_patch_id": rec.canonical_patch_id,
            "regiao": rec.regiao,
            "candidate_category": rec.candidate_category,
            "readiness_status": readiness,
        })

        patch_index_rows.append({
            "review_item_id": rec.review_item_id,
            "canonical_patch_id": rec.canonical_patch_id,
            "regiao": rec.regiao,
            "candidate_category": rec.candidate_category,
            "uncertainty_level": rec.uncertainty_level,
            "usable_in_discussion": rec.usable_in_discussion,
            "preview_file": preview_file,
            "preview_status": preview_status,
        })

    n_generated = sum(1 for r in manifest_rows if r["preview_status"] == "GENERATED")
    n_not_found = sum(1 for r in manifest_rows if r["tif_path_status"] == "NOT_FOUND")
    n_blocked = sum(1 for r in manifest_rows if r["preview_status"] == "BLOCKED")
    n_failed = sum(1 for r in manifest_rows if r["preview_status"] == "FAILED")

    print(f"[v1hc] Previews generated: {n_generated}/47")
    print(f"[v1hc] TIF not found: {n_not_found}")
    print(f"[v1hc] Blocked: {n_blocked}")
    print(f"[v1hc] Failed: {n_failed}")

    # Contact sheets
    print("[v1hc] Building contact sheets...")

    medoids = [r for r in records if "medoid" in r.candidate_category]
    outliers = [r for r in records if "outlier" in r.candidate_category]
    low_coverage = [r for r in records if "coverage_external_low" in r.candidate_category]

    cs_results: dict[str, bool] = {}

    cs_medoids_path = FIGURES_DIR / "contact_sheet_medoids_v1hc.png"
    cs_results["contact_sheet_medoids_v1hc.png"] = build_contact_sheet(
        medoids, preview_map, cs_medoids_path,
        "REV-P v1hc — Medoids Regionais (Review-only)",
        ncols=3,
    )

    cs_outliers_path = FIGURES_DIR / "contact_sheet_outliers_v1hc.png"
    cs_results["contact_sheet_outliers_v1hc.png"] = build_contact_sheet(
        outliers, preview_map, cs_outliers_path,
        "REV-P v1hc — Outliers Estruturais (Review-only)",
        ncols=3,
    )

    cs_low_path = FIGURES_DIR / "contact_sheet_low_external_coverage_v1hc.png"
    cs_results["contact_sheet_low_external_coverage_v1hc.png"] = build_contact_sheet(
        low_coverage, preview_map, cs_low_path,
        "REV-P v1hc — Baixa Cobertura GIS Externa (Review-only)",
        ncols=7,
    )

    cs_all_path = FIGURES_DIR / "contact_sheet_all_review_candidates_v1hc.png"
    cs_results["contact_sheet_all_review_candidates_v1hc.png"] = build_contact_sheet(
        records, preview_map, cs_all_path,
        "REV-P v1hc — Todos os Candidatos de Revisão (Review-only)",
        ncols=7,
    )

    for name, ok in cs_results.items():
        status = "OK" if ok else "SKIPPED (no previews)"
        print(f"[v1hc]   {name}: {status}")

    # grava as saidas
    write_csv(OUT_DIR / "visual_review_preview_manifest_v1hc.csv", manifest_rows)
    write_csv(OUT_DIR / "visual_review_readiness_v1hc.csv", readiness_rows)
    write_csv(OUT_DIR / "visual_review_patch_index_v1hc.csv", patch_index_rows)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "n_candidates": len(records),
        "n_tif_found": sum(1 for r in manifest_rows if r["tif_path_status"] == "FOUND"),
        "n_tif_not_found": n_not_found,
        "n_previews_generated": n_generated,
        "n_preview_failed": n_failed,
        "n_blocked": n_blocked,
        "contact_sheets_generated": {k: v for k, v in cs_results.items()},
        "preview_directory": str(FIGURES_DIR.relative_to(ROOT)),
        "sentinel_root_available": sentinel_root is not None,
        "sentinel_root_status": root_status,
        "methodological_guardrails": {
            "labels_created": False,
            "predictions_made": False,
            "ground_truth_established": False,
            "review_only": True,
            "private_paths_in_outputs": False,
        },
        "limitations": [
            "Previews are visual inspection aids — no automatic interpretation.",
            "RGB composite uses B4/B3/B2 with 2-98 percentile normalization per band.",
            "NDVI uses (B8-B4)/(B8+B4) — contextual index only.",
            "All interpretation requires reviewer judgment.",
        ],
    }
    write_json(OUT_DIR / "visual_review_preview_summary_v1hc.json", summary)

    print(f"\n[v1hc] Done. Outputs in {OUT_DIR.relative_to(ROOT)}")
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[v1hc] Written: {path.name} ({len(rows)} rows)")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[v1hc] Written: {path.name}")


if __name__ == "__main__":
    sys.exit(main())
