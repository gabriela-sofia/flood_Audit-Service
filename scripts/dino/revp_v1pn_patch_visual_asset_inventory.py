"""REV-P v1pn — inventário do ativo visual por patch (só metadado).

Varre os diretórios conhecidos do repositório atrás de imagem elegível a embedding DINO.
Lê só metadado — caminho, nome, tamanho, extensão — e nunca abre o pixel. Caminho sob
local_runs/ sai mascarado, e fixture ou imagem sintética é filtrada fora, para o
inventário não misturar dado real com dado fabricado.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1pn_v1pt_dino_execution_common import (
    DATASETS, DOCS, IMAGE_EXTENSIONS, ROOT, SCHEMAS,
    _p, assert_no_forbidden_true, is_fixture_or_synthetic,
    is_local_only_path, mask_local_path, normalize_patch_from_name,
    normalize_region, path_hash, require_no_abs_paths,
    sanitized_rel_path, write_csv, write_doc, write_schema,
)

OUT_INV = _p("REVP_V1PN_OUT_INV", DATASETS / "dino_patch_visual_asset_inventory_v1pn.csv")
OUT_SUM = _p("REVP_V1PN_OUT_SUM", DATASETS / "dino_patch_visual_asset_inventory_summary_v1pn.csv")
SCH_INV = _p("REVP_V1PN_SCH_INV", SCHEMAS / "dino_patch_visual_asset_inventory_v1pn_schema.csv")
SCH_SUM = _p("REVP_V1PN_SCH_SUM", SCHEMAS / "dino_patch_visual_asset_inventory_summary_v1pn_schema.csv")
DOC = _p("REVP_V1PN_DOC", DOCS / "revp_v1pn_patch_visual_asset_inventory.md")

SCAN_DIRS: tuple[str, ...] = ("datasets", "figures", "docs", "local_runs")
SCAN_GLOB_DIRS = ("figures_*",)

INV_FIELDS = [
    "visual_asset_id", "relative_path", "path_hash", "file_ext",
    "file_size_bytes", "patch_id", "alias", "region",
    "asset_role_hint", "is_local_only", "is_fixture_or_synthetic",
    "eligible_for_embedding_queue", "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _role_hint(name: str) -> str:
    low = name.lower()
    if "preview" in low or "thumbnail" in low:
        return "PREVIEW"
    if "patch" in low:
        return "PATCH"
    if "sentinel" in low or "s2" in low:
        return "SENTINEL_SCENE"
    return "UNKNOWN"


def _scan_dir(d: Path, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not d.exists():
        return rows
    idx = 0
    for path in sorted(d.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if "__pycache__" in path.parts:
            continue
        idx += 1
        rel = sanitized_rel_path(path, root)
        local = is_local_only_path(rel)
        display_path = mask_local_path(rel) if local else rel
        ph = path_hash(rel)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        name = path.name
        patch, alias, region = normalize_patch_from_name(name)
        fixture = is_fixture_or_synthetic(name + " " + rel)
        blocked = ""
        eligible = "true"
        if fixture:
            eligible, blocked = "false", "fixture_or_synthetic"
        elif local:
            eligible, blocked = "false", "local_only_not_committed"
        rows.append({
            "visual_asset_id": f"V1PN_IMG_{ph[:8]}",
            "relative_path": display_path,
            "path_hash": ph,
            "file_ext": path.suffix.lower(),
            "file_size_bytes": str(size),
            "patch_id": patch,
            "alias": alias,
            "region": region,
            "asset_role_hint": _role_hint(name),
            "is_local_only": str(local).lower(),
            "is_fixture_or_synthetic": str(fixture).lower(),
            "eligible_for_embedding_queue": eligible,
            "blocked_reason": blocked,
            "notes": "",
        })
    return rows


def discover(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    dirs = list(SCAN_DIRS)
    for pattern in SCAN_GLOB_DIRS:
        for d in sorted(root.glob(pattern)):
            if d.is_dir():
                dirs.append(str(d.relative_to(root)))
    for d_rel in dirs:
        d = root / d_rel
        for row in _scan_dir(d, root):
            k = row["path_hash"]
            if k not in seen:
                seen.add(k)
                rows.append(row)
    return rows


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    eligible = sum(1 for r in rows if r["eligible_for_embedding_queue"] == "true")
    local = sum(1 for r in rows if r["is_local_only"] == "true")
    fixture = sum(1 for r in rows if r["is_fixture_or_synthetic"] == "true")
    exts: dict[str, int] = {}
    for r in rows:
        exts[r["file_ext"]] = exts.get(r["file_ext"], 0) + 1
    return [
        {"stat_key": "visual_assets_found", "stat_value": str(len(rows))},
        {"stat_key": "eligible_for_queue", "stat_value": str(eligible)},
        {"stat_key": "local_only_masked", "stat_value": str(local)},
        {"stat_key": "fixture_or_synthetic", "stat_value": str(fixture)},
        {"stat_key": "extensions", "stat_value": str(exts)},
    ]


def run() -> None:
    rows = discover(ROOT)
    require_no_abs_paths(rows, "v1pn")
    assert_no_forbidden_true(rows, "v1pn")
    summary = build_summary(rows)
    write_csv(OUT_INV, rows, INV_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_INV, INV_FIELDS, "v1pn_patch_visual_asset_inventory")
    write_schema(SCH_SUM, SUM_FIELDS, "v1pn_patch_visual_asset_inventory_summary")
    eligible = sum(1 for r in rows if r["eligible_for_embedding_queue"] == "true")
    write_doc(DOC, "v1pn — Patch Visual Asset Inventory", [
        "## Objetivo",
        "Inventariar imagens elegíveis para geração de embedding DINO. "
        "Somente metadados — sem abertura de pixel.",
        "## Guardrails",
        "Caminhos local_runs/ mascarados. Fixtures bloqueadas. "
        "Nenhum label, target ou ground truth criado.",
        f"## Resultado",
        f"Imagens encontradas: {len(rows)}. Elegíveis para fila: {eligible}.",
    ])
    print(f"[v1pn] assets={len(rows)} eligible={eligible}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pn patch visual asset inventory").parse_args()
    run()
