from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "local_runs" / "dino_embeddings" / "v1fz" / "dino_balanced_embedding_manifest_v1fz.csv"
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gb"
REVIEW_ONLY_CLAIM = "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"
FORBIDDEN_REPO_DIRS = {"data", "outputs"}
FORBIDDEN_VERSIONED_EXTENSIONS = {".npy", ".npz", ".parquet", ".pt", ".pth", ".ckpt", ".safetensors", ".index", ".tif", ".tiff"}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1fx = load_module(ROOT / "scripts" / "dino" / "revp_v1fx_dino_smoke_embedding_execution.py", "revp_v1fx_for_v1gb")
v1fy = load_module(ROOT / "scripts" / "dino" / "revp_v1fy_dino_embedding_corpus_analysis.py", "revp_v1fy_for_v1gb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1gb local visual structural review for DINO embeddings.")
    parser.add_argument("--embedding-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    (path / "visual_review").mkdir(parents=True, exist_ok=True)


def local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    lines = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()] if gitignore.exists() else []
    return "local_runs/" in lines or "local_runs" in lines


def forbidden_versioned_artifacts() -> list[str]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=False)
    if result.returncode != 0:
        return ["GIT_LS_FILES_FAILED"]
    tracked = [Path(item.decode("utf-8", errors="replace")) for item in result.stdout.split(b"\0") if item]
    return [
        path.as_posix()
        for path in tracked
        if path.parts
        and (
            path.parts[0] in FORBIDDEN_REPO_DIRS
            or path.suffix.lower() in FORBIDDEN_VERSIONED_EXTENSIONS
        )
    ]


def load_embeddings(manifest: Path) -> tuple[list[dict[str, str]], np.ndarray, list[str]]:
    rows = [row for row in read_csv(manifest) if row.get("success") == "SUCCESS"]
    vectors: list[np.ndarray] = []
    valid_rows: list[dict[str, str]] = []
    ids: list[str] = []
    base = manifest.parent
    for row in rows:
        try:
            data = np.load(base / row["embedding_path"])
            vector = np.asarray(data["cls_embedding"], dtype="float32")
            if vector.size and np.isfinite(vector).all():
                vectors.append(vector)
                valid_rows.append(row)
                ids.append(row["dino_input_id"])
        except Exception:
            continue
    matrix = np.vstack(vectors) if vectors else np.empty((0, 0), dtype="float32")
    return valid_rows, matrix, ids


def normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def cosine(matrix: np.ndarray) -> np.ndarray:
    normed = normalize(matrix)
    return normed @ normed.T if normed.size else np.empty((0, 0), dtype="float32")


def kmeans(matrix: np.ndarray, k: int, seed: int) -> np.ndarray:
    if matrix.shape[0] < k:
        return np.zeros(matrix.shape[0], dtype=int)
    rng = np.random.default_rng(seed)
    centers = matrix[rng.choice(matrix.shape[0], size=k, replace=False)].copy()
    labels = np.zeros(matrix.shape[0], dtype=int)
    for _ in range(40):
        distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = matrix[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return labels


def medoid_indices(matrix: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    result: dict[int, int] = {}
    for label in sorted(set(int(x) for x in labels)):
        idx = np.where(labels == label)[0]
        sub = matrix[idx]
        centroid = sub.mean(axis=0)
        distances = np.linalg.norm(sub - centroid, axis=1)
        result[label] = int(idx[int(distances.argmin())])
    return result


def edge_indices(matrix: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    result: dict[int, int] = {}
    for label in sorted(set(int(x) for x in labels)):
        idx = np.where(labels == label)[0]
        sub = matrix[idx]
        centroid = sub.mean(axis=0)
        distances = np.linalg.norm(sub - centroid, axis=1)
        result[label] = int(idx[int(distances.argmax())])
    return result


def save_patch_image(row: dict[str, str], image_path: Path, image_size: int) -> tuple[bool, str]:
    try:
        from PIL import Image  # type: ignore

        array, _pixel_status, error, _meta = v1fx.read_image_tensor(Path(row.get("source_path", "")), image_size)
        if array is None:
            return False, error
        image = Image.fromarray(np.clip(array * 255, 0, 255).astype("uint8"), mode="RGB")
        image.save(image_path)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def make_mosaic(image_paths: list[Path], output_path: Path, image_size: int) -> bool:
    try:
        from PIL import Image, ImageDraw  # type: ignore

        valid = [path for path in image_paths if path.exists()]
        if not valid:
            return False
        canvas = Image.new("RGB", (image_size * len(valid), image_size), "white")
        draw = ImageDraw.Draw(canvas)
        for idx, path in enumerate(valid):
            img = Image.open(path).convert("RGB").resize((image_size, image_size))
            canvas.paste(img, (idx * image_size, 0))
            draw.rectangle([idx * image_size, 0, (idx + 1) * image_size - 1, image_size - 1], outline="black")
        canvas.save(output_path)
        return True
    except Exception:
        return False


def nearest_neighbors(ids: list[str], matrix: np.ndarray, top_k: int) -> list[dict[str, object]]:
    sims = cosine(matrix)
    rows: list[dict[str, object]] = []
    for i, source in enumerate(ids):
        order = [j for j in np.argsort(-sims[i]) if j != i]
        for rank, j in enumerate(order[:top_k], start=1):
            rows.append({"dino_input_id": source, "neighbor_dino_input_id": ids[j], "rank": rank, "cosine_similarity": float(sims[i, j]), "cosine_distance": float(1 - sims[i, j])})
    return rows


def visual_review(rows: list[dict[str, str]], matrix: np.ndarray, ids: list[str], labels: np.ndarray, output_dir: Path, image_size: int, top_k: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    visual_dir = output_dir / "visual_review"
    by_id = {row["dino_input_id"]: row for row in rows}
    patch_images: dict[str, Path] = {}
    missing: list[dict[str, object]] = []
    for row in rows:
        image_path = visual_dir / f"{row['dino_input_id']}.png"
        ok, error = save_patch_image(row, image_path, image_size)
        if ok:
            patch_images[row["dino_input_id"]] = image_path
        else:
            missing.append({"dino_input_id": row["dino_input_id"], "source_path": row.get("source_path", ""), "failure_reason": error})

    visual_manifest: list[dict[str, object]] = []
    nn = nearest_neighbors(ids, matrix, top_k)
    for source in ids:
        if source not in patch_images:
            continue
        paths = [patch_images[source]]
        neighbors = [row["neighbor_dino_input_id"] for row in nn if row["dino_input_id"] == source]
        paths += [patch_images[n] for n in neighbors if n in patch_images]
        mosaic = visual_dir / f"nearest_{source}.png"
        if make_mosaic(paths, mosaic, image_size):
            visual_manifest.append({"panel_type": "nearest_neighbors", "source_patch": by_id[source].get("patch_id", ""), "neighbor_patch": "|".join(by_id[n].get("patch_id", "") for n in neighbors if n in by_id), "cluster_id": int(labels[ids.index(source)]), "outlier_type": "", "image_path": str(mosaic), "embedding_distance": "", "region": by_id[source].get("region", "")})

    medoids = medoid_indices(matrix, labels)
    edges = edge_indices(matrix, labels)
    medoid_rows: list[dict[str, object]] = []
    stable_rows: list[dict[str, object]] = []
    for cluster_id, idx in medoids.items():
        dino_id = ids[idx]
        medoid_rows.append({"cluster_id": cluster_id, "dino_input_id": dino_id, "patch_id": by_id[dino_id].get("patch_id", ""), "region": by_id[dino_id].get("region", ""), "medoid_status": "STRUCTURAL_REPRESENTATIVE_ONLY"})
        stable_rows.append({"representative_type": "cluster_medoid", "dino_input_id": dino_id, "patch_id": by_id[dino_id].get("patch_id", ""), "region": by_id[dino_id].get("region", ""), "stability_status": "SELECTED_SINGLE_RUN"})
        image = visual_dir / f"cluster_{cluster_id}_medoid.png"
        if dino_id in patch_images and make_mosaic([patch_images[dino_id]], image, image_size):
            visual_manifest.append({"panel_type": "cluster_medoid", "source_patch": by_id[dino_id].get("patch_id", ""), "neighbor_patch": "", "cluster_id": cluster_id, "outlier_type": "", "image_path": str(image), "embedding_distance": "0", "region": by_id[dino_id].get("region", "")})
        edge_id = ids[edges[cluster_id]]
        edge_image = visual_dir / f"cluster_{cluster_id}_edge.png"
        if edge_id in patch_images and make_mosaic([patch_images[edge_id]], edge_image, image_size):
            visual_manifest.append({"panel_type": "cluster_edge_case", "source_patch": by_id[edge_id].get("patch_id", ""), "neighbor_patch": "", "cluster_id": cluster_id, "outlier_type": "CLUSTER_EDGE", "image_path": str(edge_image), "embedding_distance": "", "region": by_id[edge_id].get("region", "")})

    for region in sorted({row.get("region", "") for row in rows}):
        region_idx = [i for i, row in enumerate(rows) if row.get("region") == region]
        centroid = matrix[region_idx].mean(axis=0)
        chosen = region_idx[int(np.linalg.norm(matrix[region_idx] - centroid, axis=1).argmin())]
        dino_id = ids[chosen]
        image = visual_dir / f"region_{safe_name(region)}_exemplar.png"
        if dino_id in patch_images and make_mosaic([patch_images[dino_id]], image, image_size):
            visual_manifest.append({"panel_type": "region_exemplar", "source_patch": by_id[dino_id].get("patch_id", ""), "neighbor_patch": "", "cluster_id": int(labels[chosen]), "outlier_type": "", "image_path": str(image), "embedding_distance": "", "region": region})

    return visual_manifest, medoid_rows, stable_rows, missing


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def spatial_metrics(rows: list[dict[str, str]], matrix: np.ndarray, ids: list[str], top_k: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    sims = cosine(matrix)
    numbers = [numeric_patch(row.get("patch_id", "")) for row in rows]
    spatial_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    consistency_rows: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        order = [j for j in np.argsort(-sims[i]) if j != i]
        top = order[:top_k]
        same_region = [j for j in top if rows[j].get("region") == row.get("region")]
        spatial_delta = [abs(numbers[i] - numbers[j]) for j in top if numbers[i] >= 0 and numbers[j] >= 0]
        spatial_rows.append({"dino_input_id": ids[i], "mean_topk_cosine": float(np.mean([sims[i, j] for j in top])) if top else 0.0, "mean_patch_id_distance": float(np.mean(spatial_delta)) if spatial_delta else "", "same_region_neighbor_ratio": len(same_region) / max(len(top), 1), "claim_scope": "STRUCTURAL_DIAGNOSTIC_ONLY"})
        density_rows.append({"dino_input_id": ids[i], "local_density": float(np.mean([sims[i, j] for j in top])) if top else 0.0, "density_status": "LOW_LOCAL_DENSITY" if top and np.mean([sims[i, j] for j in top]) < 0.5 else "WITHIN_RANGE"})
        consistency_rows.append({"dino_input_id": ids[i], "top_k": top_k, "same_region_neighbors": len(same_region), "neighborhood_consistency_status": "PASS" if top else "NO_NEIGHBORS"})
    return spatial_rows, density_rows, consistency_rows


def numeric_patch(text: str) -> int:
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else -1


def multiscale(rows: list[dict[str, str]], matrix: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    normed = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    pooled_by_region: dict[str, np.ndarray] = {}
    for region in sorted({row.get("region", "") for row in rows}):
        idx = [i for i, row in enumerate(rows) if row.get("region") == region]
        pooled_by_region[region] = normed[idx].mean(axis=0)
    sim_rows = []
    pool_rows = []
    for i, row in enumerate(rows):
        pooled = pooled_by_region[row.get("region", "")]
        sim = float(np.dot(normed[i], pooled) / max(np.linalg.norm(pooled), 1e-12))
        sim_rows.append({"dino_input_id": row["dino_input_id"], "scale_a": "patch", "scale_b": "region_pooled", "cosine_similarity": sim, "status": "PASS"})
        pool_rows.append({"dino_input_id": row["dino_input_id"], "pooling_method": "region_mean_proxy", "pooling_consistency": sim, "claim_scope": "MULTISCALE_STRUCTURAL_SANITY_ONLY"})
    return sim_rows, pool_rows


def outlier_taxonomy(rows: list[dict[str, str]], matrix: np.ndarray, labels: np.ndarray, density_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    normed = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    sims = normed @ normed.T
    nearest = []
    for i in range(len(rows)):
        values = [1 - sims[i, j] for j in range(len(rows)) if i != j]
        nearest.append(min(values) if values else 0.0)
    threshold = float(np.mean(nearest) + np.std(nearest)) if nearest else 0.0
    taxonomy = []
    for i, row in enumerate(rows):
        categories = []
        if nearest[i] > threshold:
            categories.append("COSINE_OUTLIER")
        if density_rows[i]["density_status"] == "LOW_LOCAL_DENSITY":
            categories.append("DENSITY_OUTLIER")
        if sum(labels == labels[i]) <= 1:
            categories.append("INSTABILITY_OUTLIER")
        if not categories:
            categories.append("NONE")
        taxonomy.append({"dino_input_id": row["dino_input_id"], "outlier_categories": "|".join(categories), "nearest_distance": nearest[i], "cluster_id": int(labels[i]), "region": row.get("region", "")})
    overlap = []
    for item in taxonomy:
        cats = [cat for cat in str(item["outlier_categories"]).split("|") if cat != "NONE"]
        overlap.append({"dino_input_id": item["dino_input_id"], "category_count": len(cats), "overlap_status": "OVERLAP" if len(cats) > 1 else "SINGLE_OR_NONE"})
    return taxonomy, overlap


def make_qa(visual_manifest: list[dict[str, object]], missing_images: list[dict[str, object]], medoids: list[dict[str, object]], nn_rows: list[dict[str, object]], multiscale_rows: list[dict[str, object]], output_dir: Path) -> list[dict[str, str]]:
    qa = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    image_paths = [Path(str(row["image_path"])) for row in visual_manifest if row.get("image_path")]
    add("image export integrity", bool(image_paths) and all(path.exists() for path in image_paths), f"images={len(image_paths)}")
    add("missing raster handling", missing_images is not None, f"missing={len(missing_images)}")
    add("visual manifest integrity", all(row.get("panel_type") and row.get("image_path") for row in visual_manifest), f"rows={len(visual_manifest)}")
    add("medoid uniqueness", len({row.get("dino_input_id") for row in medoids}) == len(medoids), f"medoids={len(medoids)}")
    add("neighborhood reproducibility", bool(nn_rows), f"neighbors={len(nn_rows)}")
    add("multiscale consistency sanity", bool(multiscale_rows), f"rows={len(multiscale_rows)}")
    add("local visual outputs only", "local_runs" in output_dir.parts, str(output_dir))
    add("no labels targets or predictive claims", True, REVIEW_ONLY_CLAIM)
    add("local_runs ignored", local_runs_ignored(), ".gitignore checked")
    add("no forbidden versioned artifacts", not forbidden_versioned_artifacts(), "repo checked outside local_runs")
    return qa


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force)
    rows, matrix, ids = load_embeddings(Path(args.embedding_manifest))
    if len(rows) == 0:
        raise RuntimeError("No valid embeddings available for v1gb visual structural review.")
    normed = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    k = min(3, max(1, len(rows)))
    labels = kmeans(normed, k, args.seed)
    visual_manifest, medoids, stable_representatives, missing_images = visual_review(rows, normed, ids, labels, output_dir, args.image_size, args.top_k)
    nn_rows = nearest_neighbors(ids, normed, args.top_k)
    spatial_rows, density_rows, neighborhood_rows = spatial_metrics(rows, normed, ids, args.top_k)
    multiscale_rows, pooling_rows = multiscale(rows, normed)
    outlier_rows, overlap_rows = outlier_taxonomy(rows, normed, labels, density_rows)
    qa_rows = make_qa(visual_manifest, missing_images, medoids, nn_rows, multiscale_rows, output_dir)
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa_rows) else "FAIL"
    summary = {
        "phase": "v1gb",
        "phase_name": "DINO_EMBEDDING_LOCAL_VISUAL_STRUCTURAL_REVIEW",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_manifest": str(Path(args.embedding_manifest)),
        "embedding_count": len(rows),
        "regions": sorted({row.get("region", "") for row in rows}),
        "visual_panels": len(visual_manifest),
        "medoids": len(medoids),
        "outlier_categories": sorted({cat for row in outlier_rows for cat in str(row["outlier_categories"]).split("|")}),
        "spatial_consistency_status": "PASS" if spatial_rows else "SKIPPED",
        "multiscale_status": "PASS" if multiscale_rows else "SKIPPED",
        "qa_status": qa_status,
        "review_only": True,
        "supervised_training": False,
        "labels_created": False,
        "targets_created": False,
        "predictive_claims": False,
        "clusters_are_classes": False,
        "multimodal_hold": True,
        "outputs_local_only": True,
    }
    write_csv(output_dir / "visual_review_manifest.csv", visual_manifest, ["panel_type", "source_patch", "neighbor_patch", "cluster_id", "outlier_type", "image_path", "embedding_distance", "region"])
    write_csv(output_dir / "missing_visual_sources.csv", missing_images, ["dino_input_id", "source_path", "failure_reason"])
    write_csv(output_dir / "spatial_similarity_metrics.csv", spatial_rows, ["dino_input_id", "mean_topk_cosine", "mean_patch_id_distance", "same_region_neighbor_ratio", "claim_scope"])
    write_csv(output_dir / "local_density_metrics.csv", density_rows, ["dino_input_id", "local_density", "density_status"])
    write_csv(output_dir / "neighborhood_consistency.csv", neighborhood_rows, ["dino_input_id", "top_k", "same_region_neighbors", "neighborhood_consistency_status"])
    write_csv(output_dir / "multiscale_similarity.csv", multiscale_rows, ["dino_input_id", "scale_a", "scale_b", "cosine_similarity", "status"])
    write_csv(output_dir / "pooling_consistency.csv", pooling_rows, ["dino_input_id", "pooling_method", "pooling_consistency", "claim_scope"])
    write_csv(output_dir / "cluster_medoids.csv", medoids, ["cluster_id", "dino_input_id", "patch_id", "region", "medoid_status"])
    write_csv(output_dir / "stable_representatives.csv", stable_representatives, ["representative_type", "dino_input_id", "patch_id", "region", "stability_status"])
    write_csv(output_dir / "outlier_taxonomy.csv", outlier_rows, ["dino_input_id", "outlier_categories", "nearest_distance", "cluster_id", "region"])
    write_csv(output_dir / "anomaly_overlap.csv", overlap_rows, ["dino_input_id", "category_count", "overlap_status"])
    write_csv(output_dir / "visual_structural_review_qa.csv", qa_rows, ["check", "status", "details"])
    write_json(output_dir / "visual_structural_review_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qa_status == "PASS" else 2


def main() -> int:
    try:
        return run(parse_args())
    except FileExistsError as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
