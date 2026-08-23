"""REV-P v1qv — mapa de atenção do DINOv2 por rollout (real, somente-revisão).

Extrai o peso de auto-atenção REAL (do token CLS para os tokens de patch) do checkpoint
local do DINOv2-with-registers sobre preview RGB real do Sentinel, tira a média entre
cabeças e camadas pelo rollout de atenção, e renderiza um PNG com o mapa de calor
sobreposto. É puro auxílio de interpretabilidade — "para onde o DINO olha neste patch?".
Não confirma evento, não cria rótulo e não realimenta treino nenhum. Vale a mesma
família de portões de falha fechada do v1qg/v1qi/v1qj: exige REVP_DINO_DRY_RUN=false,
REVP_DINO_PIXEL_READ_ALLOWED=true e um diretório de modelo local offline. O padrão é
dry-run.

Como o rollout funciona (Abnar & Zuidema, 2020): em cada camada tira-se a média das
cabeças de atenção, soma-se a identidade para representar a conexão residual,
normaliza-se por linha, e multiplicam-se as matrizes de camada da primeira à última. A
linha do CLS na matriz resultante, restrita às colunas de token de patch (fora os tokens
de registro e o próprio CLS), é o mapa de atenção.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1qg_v1qm_smoke_embedding_comum import (
    DATASETS, DOCS, ROOT, SCHEMAS,
    _p, assert_no_forbidden_true, env_int, env_str, env_true, normalize_region,
    path_hash, read_csv, require_no_abs_paths, resolve_local_asset,
    safe_model_probe, write_csv, write_doc, write_schema,
)

IN_SEL = _p("REVP_V1QV_IN_SEL", DATASETS / "dino_smoke_sample_selection_linked_v1qu.csv")
OUT_DIR = _p("REVP_V1QV_OUT_DIR", ROOT / "patches" / "attention_maps")
OUT_MAN = _p("REVP_V1QV_OUT_MAN", DATASETS / "dino_attention_rollout_manifest_v1qv.csv")
OUT_SUM = _p("REVP_V1QV_OUT_SUM", DATASETS / "dino_attention_rollout_summary_v1qv.csv")
SCH_MAN = _p("REVP_V1QV_SCH_MAN", SCHEMAS / "dino_attention_rollout_manifest_v1qv_schema.csv")
DOC = _p("REVP_V1QV_DOC", DOCS / "revp_v1qv_mapa_atencao_rollout.md")

MAN_FIELDS = [
    "rollout_id", "smoke_id", "patch_id", "regiao", "relative_path",
    "output_png_relative_path", "output_png_path_hash", "num_layers",
    "num_heads", "num_register_tokens", "attention_use", "status",
    "blocked_reason", "notas",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _gate_status() -> tuple[str, dict[str, Any]]:
    dry_run = env_true("REVP_DINO_DRY_RUN", True)
    pixel_allowed = env_true("REVP_DINO_PIXEL_READ_ALLOWED", False)
    allow_dl = env_true("REVP_DINO_ALLOW_DOWNLOAD", False)
    model_path = env_str("REVP_DINO_MODEL_PATH", "")
    probe = safe_model_probe(model_path or None)
    model_ready = probe["model_path_exists"] and probe["config_exists"] \
        and probe["weights_exists"] and not allow_dl and probe["transformers_available"]
    ctx = {"dry_run": dry_run, "pixel_allowed": pixel_allowed, "model_path": model_path,
           "allow_dl": allow_dl, "model_ready": model_ready}
    if dry_run:
        return ("DRY_RUN", ctx)
    if not model_ready:
        return ("MODEL_MISSING", ctx)
    if not pixel_allowed:
        return ("PIXEL_BLOCKED", ctx)
    return ("EXECUTE", ctx)


def _rollout(attentions: list[Any]) -> Any:
    """Rollout de atenção ao longo das camadas (Abnar & Zuidema, 2020).

    ``attentions``: lista de tensores, um por camada, no formato (1, cabeças, N, N).
    Devolve a matriz (N, N) já acumulada.
    """
    import torch
    result = None
    for attn in attentions:
        a = attn[0].mean(dim=0)  # average heads -> (N, N)
        n = a.shape[-1]
        a = a + torch.eye(n)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else a @ result
    return result


def _render_overlay(img: Any, cls_to_patch: Any, grid: int, out_path: Path) -> None:
    import numpy as np
    from PIL import Image

    heat = cls_to_patch.reshape(grid, grid).detach().cpu().numpy()
    heat = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-9)
    heat_img = Image.fromarray((heat * 255).astype("uint8")).resize(img.size, Image.BILINEAR)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heat_rgba = matplotlib.colormaps["inferno"](np.asarray(heat_img) / 255.0)
    heat_rgb = (heat_rgba[:, :, :3] * 255).astype("uint8")
    base = np.asarray(img.convert("RGB")).astype("float32")
    overlay = (0.45 * heat_rgb.astype("float32") + 0.55 * base).clip(0, 255).astype("uint8")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(base.astype("uint8")); axes[0].set_title("RGB preview", fontsize=9); axes[0].axis("off")
    axes[1].imshow(heat_img, cmap="inferno"); axes[1].set_title("CLS attention rollout", fontsize=9); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("overlay", fontsize=9); axes[2].axis("off")
    fig.suptitle("DINOv2-with-registers — review-only attention rollout (not an event confirmation)", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run() -> None:
    gate, ctx = _gate_status()
    rows = read_csv(IN_SEL)
    max_n = env_int("REVP_DINO_ATTN_MAX", 3)
    manifest: list[dict[str, Any]] = []

    if gate != "EXECUTE":
        status_map = {
            "DRY_RUN": "ATTENTION_ROLLOUT_DRY_RUN_ONLY",
            "MODEL_MISSING": "ATTENTION_ROLLOUT_MODEL_MISSING_FAIL_CLOSED",
            "PIXEL_BLOCKED": "ATTENTION_ROLLOUT_PIXEL_READ_BLOCKED_FAIL_CLOSED",
        }
        final = status_map.get(gate, "ATTENTION_ROLLOUT_MODEL_MISSING_FAIL_CLOSED")
        for i, r in enumerate(rows[:max_n], 1):
            manifest.append({
                "rollout_id": f"V1QV_ROLL_{i:05d}", "smoke_id": r.get("smoke_id", ""),
                "patch_id": r.get("patch_id", ""), "regiao": normalize_region(r.get("regiao", "")),
                "relative_path": r.get("relative_path", ""), "output_png_relative_path": "",
                "output_png_path_hash": "", "num_layers": "", "num_heads": "",
                "num_register_tokens": "", "attention_use": "BLOCKED_NO_ATTENTION",
                "status": final, "blocked_reason": gate.lower(), "notas": "",
            })
        _write(manifest, final, gate)
        print(f"[v1qv] gate={gate} status={final} rendered=0")
        return

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    model_path = ctx["model_path"]
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    modelo = AutoModel.from_pretrained(model_path, local_files_only=True, attn_implementation="eager")
    modelo.eval()
    n_registers = int(getattr(modelo.config, "num_register_tokens", 0) or 0)
    n_heads = int(getattr(modelo.config, "num_attention_heads", 0) or 0)
    n_layers = int(getattr(modelo.config, "num_hidden_layers", 0) or 0)

    rendered = 0
    for i, r in enumerate(rows[:max_n], 1):
        smoke_id = r.get("smoke_id", "")
        patch = (r.get("patch_id", "") or "UNKNOWN").upper()
        regiao = normalize_region(r.get("regiao", ""))
        rel = r.get("relative_path", "")
        img_path = resolve_local_asset(rel) if rel else None
        row: dict[str, Any] = {
            "rollout_id": f"V1QV_ROLL_{i:05d}", "smoke_id": smoke_id, "patch_id": patch,
            "regiao": regiao, "relative_path": rel, "output_png_relative_path": "",
            "output_png_path_hash": "", "num_layers": str(n_layers), "num_heads": str(n_heads),
            "num_register_tokens": str(n_registers), "attention_use": "REVIEW_ONLY_ATTENTION_MAP",
            "status": "", "blocked_reason": "", "notas": "",
        }
        if img_path is None:
            row["status"] = "ATTENTION_ROLLOUT_ASSET_MISSING_FAIL_CLOSED"
            row["blocked_reason"] = "asset_not_resolved"
            manifest.append(row)
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                out = modelo(**inputs, output_attentions=True)
            attn = list(out.attentions)
            if not attn:
                raise RuntimeError("no_attention_returned")
            rolled = _rollout(attn)  # (N, N)
            n_special = 1 + n_registers  # CLS + registers, in that token order
            cls_to_patch = rolled[0, n_special:]
            grid = int(round(cls_to_patch.shape[0] ** 0.5))
            out_rel = f"{regiao.lower() if regiao != 'PET' else 'petropolis'}/{patch.lower()}__attention_rollout.png"
            out_path = OUT_DIR / out_rel
            _render_overlay(img, cls_to_patch, grid, out_path)
            row["output_png_relative_path"] = f"patches/attention_maps/{out_rel}"
            row["output_png_path_hash"] = path_hash(out_rel)
            row["status"] = "ATTENTION_ROLLOUT_READY_REVIEW_ONLY"
            rendered += 1
        except Exception as exc:  # noqa: BLE001 — fail-closed per row, continue others
            row["status"] = "ATTENTION_ROLLOUT_EXECUTION_FAILED_FAIL_CLOSED"
            row["blocked_reason"] = type(exc).__name__
        manifest.append(row)

    final = "ATTENTION_ROLLOUT_READY_REVIEW_ONLY" if rendered > 0 else "ATTENTION_ROLLOUT_EXECUTION_FAILED_FAIL_CLOSED"
    _write(manifest, final, gate, n_layers, n_heads, n_registers, rendered)
    print(f"[v1qv] gate={gate} status={final} rendered={rendered}/{len(rows[:max_n])}")


def _write(manifest: list[dict[str, Any]], final: str, gate: str,
           n_layers: int = 0, n_heads: int = 0, n_registers: int = 0, rendered: int = 0) -> None:
    require_no_abs_paths(manifest, "v1qv_manifest")
    assert_no_forbidden_true(manifest, "v1qv_manifest")
    summary = [
        {"stat_key": "execution_gate", "stat_value": gate},
        {"stat_key": "rollouts_rendered", "stat_value": str(rendered)},
        {"stat_key": "num_layers", "stat_value": str(n_layers)},
        {"stat_key": "num_heads", "stat_value": str(n_heads)},
        {"stat_key": "num_register_tokens", "stat_value": str(n_registers)},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "final_status", "stat_value": final},
    ]
    require_no_abs_paths(summary, "v1qv_summary")
    write_csv(OUT_MAN, manifest, MAN_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_MAN, MAN_FIELDS, "v1qv_dino_attention_rollout_manifest")
    write_doc(DOC, "v1qv — DINOv2 Attention Rollout Visualizer (review-only)", [
        "## Objetivo",
        "Extrair pesos de atenção REAIS (CLS -> tokens de patch) do DINOv2-with-registers "
        "local e renderizar rollout (Abnar & Zuidema, 2020) como heatmap. Auxílio de "
        "interpretabilidade apenas — nunca confirma evento, nunca cria rótulo.",
        "## Guardrails",
        "Fail-closed: requer REVP_DINO_DRY_RUN=false, REVP_DINO_PIXEL_READ_ALLOWED=true e "
        "modelo local offline. Default é dry-run.",
        f"## Resultado",
        f"**{final}**. Renderizados: {rendered}.",
    ])


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qv dino attention rollout visualizer").parse_args()
    run()
