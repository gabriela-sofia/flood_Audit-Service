"""SUSC-20D -- roda o motor de inferencia (susc_20d_motor_escore.py) nos 269
pontos completos do v12 e audita a saida contra os rotulos reais.

Isto NAO treina nada novo: e uma auditoria de que o motor, alimentado com os
coeficientes ja treinados e o procedimento de bootstrap ja documentado,
reproduz uma saida coerente com o que o SUSC-20C ja publicou. Nao e
validacao preditiva nova (isso ja existe: LOO-AUC=0.6781 held-out); e uma
checagem de integridade ponta-a-ponta do motor em si.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from susc_20d_motor_escore import (
    FEATURE_COLS, IN_PUBLISHED_BOOT, IN_PUBLISHED_COEFS, LIMITATIONS, MODEL_VERSION,
    REGION_SUPPORTED, REPORTS, RESULTS, SEED, bootstrap_projection_draws, fit_firth,
    load_dino_evidence_index, load_training_frame, read_csv, score_point,
    find_dino_evidence,
)


def validate_point_estimate(X: np.ndarray, y: np.ndarray, scaler: StandardScaler) -> dict:
    Xs = scaler.transform(X)
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    res = fit_firth(Xd, y)
    published = {r["variavel"]: float(r["coef_padronizado"]) for r in read_csv(IN_PUBLISHED_COEFS)}
    max_abs_diff = 0.0
    for i, feat in enumerate(FEATURE_COLS, start=1):
        diff = abs(res["beta"][i] - published[feat])
        max_abs_diff = max(max_abs_diff, diff)
    return {"beta": res["beta"], "max_abs_diff_vs_published": max_abs_diff}


def validate_bootstrap(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    draws = bootstrap_projection_draws(X, y, n_boot=1000, seed=SEED)
    published = read_csv(IN_PUBLISHED_BOOT)
    max_abs_diff = 0.0
    for i, feat in enumerate(FEATURE_COLS, start=1):
        pub = next(r for r in published if r["variavel"] == feat)
        lo, hi = np.percentile(draws[:, i], [2.5, 97.5])
        diff = max(abs(lo - float(pub["boot_ci_low_2.5pct"])), abs(hi - float(pub["boot_ci_high_97.5pct"])))
        max_abs_diff = max(max_abs_diff, diff)
    return draws, max_abs_diff


def run() -> None:
    kept, X, y, scaler = load_training_frame()
    n = len(kept)
    print(f"[susc_20d] n_used={n} n_pos={int(y.sum())} n_neg={int(n - y.sum())}")

    point_val = validate_point_estimate(X, y, scaler)
    print(f"[susc_20d] point-estimate validation vs published SUSC-20C coefs: "
          f"max_abs_diff={point_val['max_abs_diff_vs_published']:.6f}")
    assert point_val["max_abs_diff_vs_published"] < 1e-3, "Firth point estimate does not match SUSC-20C"

    boot_draws, boot_max_diff = validate_bootstrap(X, y)
    print(f"[susc_20d] bootstrap validation vs published SUSC-20C 2.5/97.5pct: "
          f"max_abs_diff={boot_max_diff:.4f} (n_boot_success={len(boot_draws)})")

    bboxes, emb_by_patch = load_dino_evidence_index()
    dino_available = bool(bboxes and emb_by_patch)
    print(f"[susc_20d] DINO evidence index: {'available' if dino_available else 'not configured (optional, skipped)'}")

    rows_out = []
    scores_for_auc = []
    for j in kept:
        out = score_point(j, scaler, point_val["beta"], boot_draws)
        dino_evidence = None
        if dino_available:
            try:
                lat, lon = float(j["lat"]), float(j["lon"])
                dino_evidence = find_dino_evidence(lat, lon, bboxes, emb_by_patch)
            except (ValueError, TypeError):
                dino_evidence = None
        rows_out.append({
            "ponto_id": j["ponto_id"], "rotulo_real": j["rotulo"],
            "escore": out["escore"], "ic95_lo": out["confidence_interval"][0],
            "ic95_hi": out["confidence_interval"][1],
            "variavel_principal": out["features_used"][0]["variavel"],
            "contribuicao_variavel_principal": round(out["features_used"][0]["contribution"], 4),
            "dino_evidence_patch_id": (dino_evidence or {}).get("patch_id", ""),
            "versao_modelo": MODEL_VERSION, "regiao": REGION_SUPPORTED,
        })
        scores_for_auc.append(out["escore"])

    in_sample_auc = float(roc_auc_score(y, scores_for_auc))
    documented_loo_auc = 0.6781
    ci_widths = [r["ic95_hi"] - r["ic95_lo"] for r in rows_out]

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "susc_20d_auditoria_escore_v1.csv").open("w", encoding="utf-8", newline="") as fh:
        import csv as csv_mod
        w = csv_mod.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    summary = {
        "n_audited": n, "n_pos": int(y.sum()), "n_neg": int(n - y.sum()),
        "firth_point_estimate_max_abs_diff_vs_published_susc20c": round(point_val["max_abs_diff_vs_published"], 6),
        "bootstrap_max_abs_diff_vs_published_susc20c_percentiles": round(float(boot_max_diff), 4),
        "in_sample_auc_this_engine": round(in_sample_auc, 4),
        "documented_loo_auc_susc20c_for_reference": documented_loo_auc,
        "in_sample_vs_loo_note": "in-sample AUC e otimista por construcao (mesmo dado usado pra treinar e avaliar); "
                                 "nao substitui o LOO-AUC ja documentado, serve so pra auditar coerencia do motor",
        "mean_ci_width": round(float(np.mean(ci_widths)), 4),
        "median_ci_width": round(float(np.median(ci_widths)), 4),
        "dino_evidence_index_available": dino_available,
        "n_points_with_dino_evidence_attached": sum(1 for r in rows_out if r["dino_evidence_patch_id"]),
        "labels_created": 0, "used_as_training_feature_dino": False,
        "versao_modelo": MODEL_VERSION, "region_supported": [REGION_SUPPORTED],
        "limitations": LIMITATIONS,
        "status": "SUSC_20D_MVP_AUDIT_READY_REVIEW_ONLY",
    }
    with (RESULTS / "susc_20d_resumo_auditoria.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = f"""# SUSC-20D -- Motor de inferencia local (MVP, Recife)

## Objetivo

Executa a Fase 3 do PLANO_ACAO_produto_v1.md: prova que o motor cientifico roda
ponta-a-ponta (escore + intervalo + features + evidencia + limitacoes) antes de
gastar esforco em API/backend/interface.

## Validacao do motor antes do audit (sem isso, nao ha prova)

- Coeficientes Firth ponto-estimado: diferenca maxima vs SUSC-20C publicado =
  {point_val['max_abs_diff_vs_published']:.6f} (tolerancia 1e-3 -- passou).
- Bootstrap preditivo (N=1000, seed={SEED}): diferenca maxima nos percentis
  2.5/97.5% vs SUSC-20C publicado = {boot_max_diff:.4f}.
- Evidencia DINO: {'indice carregado (' + str(sum(1 for r in rows_out if r['dino_evidence_patch_id'])) + ' pontos com patch DINO real anexado)' if dino_available else 'nao configurada nesta execucao (opcional -- variavel REVP_SUSC20D_SENTINEL_DIR nao definida)'}.

## Auditoria (n={n}, {int(y.sum())} pos / {int(n - y.sum())} neg)

AUC in-sample deste motor: **{in_sample_auc:.4f}**. LOO-AUC ja documentado em
SUSC-20C: **{documented_loo_auc:.4f}**. Nao sao o mesmo numero por desenho --
in-sample e otimista (mesmo dado usado pra treinar e avaliar); esta auditoria
nao substitui o LOO-AUC, so confirma que o motor (coeficientes fixos + escore
por ponto) produz uma saida internamente coerente com o rotulo real.

CI (bootstrap preditivo) medio: largura media={np.mean(ci_widths):.4f},
mediana={np.median(ci_widths):.4f}.

## Limitacoes explicitas (sempre presentes na saida do motor)

{chr(10).join('- ' + l for l in LIMITATIONS)}

## Arquivos

- `results/susc_20d_auditoria_escore_v1.csv` -- escore + CI + variavel dominante + evidencia DINO por ponto
- `results/susc_20d_resumo_auditoria.json` -- resumo da auditoria
"""
    (REPORTS / "RELATORIO_susc_20d_motor_inferencia_mvp.md").write_text(report, encoding="utf-8")

    print(f"[susc_20d] in_sample_auc={in_sample_auc:.4f} documented_loo_auc={documented_loo_auc} "
          f"mean_ci_width={np.mean(ci_widths):.4f}")


if __name__ == "__main__":
    run()
