"""SUSC-20P -- validacao por blocos espaciais (bairro) de Curitiba, rota primaria (5 features).

Motivacao (literatura + achado proprio): o SUSC-20O mostrou que o AUC cai de 0,6459 (LOO-CV
embaralhado, SUSC-20N) pra 0,5246 (holdout temporal 2026). A literatura de suscetibilidade a
enchente aponta DUAS fontes possiveis pra esse tipo de colapso, e elas pedem diagnosticos
diferentes:

  1. Vazamento espacial: autocorrelacao entre unidades do mesmo bairro infla o CV embaralhado
     porque duas queixas do mesmo bairro (ou proximas) caem uma no treino e outra no teste,
     entao o modelo "decora" identidade de vizinhanca em vez de aprender fisica de terreno.
     Estudos que comparam random split vs. spatial block CV relatam AUC 5-15% mais alto sob
     split aleatorio (ScienceDirect, "Next generation data-driven flood susceptibility
     modelling with spatial machine learning").
  2. Deriva temporal/administrativa: o padrao de queixas do SIAC 156 muda ano a ano (ex.: 2025
     com proporcao de positivos desigual, 2026 e ano parcial sem toda a sazonalidade) --
     mecanismo distinto de vazamento espacial.

Este script testa a hipotese (1) isoladamente: bloco = bairro (nao interpola coordenada, usa a
unidade administrativa ja catalogada), nenhum bairro aparece simultaneamente em treino e teste
de um mesmo fold (GroupKFold, sklearn). Mesma rota causal primaria do SUSC-20N/20M/20O (5
features, elevation_m fora -- decisao causal documentada em pipeline_v20m_curitiba_primario.py).
Se o AUC tambem cair sob blocos espaciais, ha vazamento espacial. Se ficar proximo do LOO-CV
embaralhado (~0,65) e so cair no holdout temporal, o problema e deriva temporal/administrativa,
nao vazamento espacial.

Reaproveita gate_epv, firth_on_train, univariate_screen_train e temporal_holdout_auc (generica,
so espera train/test df) de pipeline_v20o_holdout_temporal_curitiba -- nao duplica logica.

Uso:
    python pipeline_v20p_cv_blocos_espaciais_curitiba.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_v20o_holdout_temporal_curitiba import (  # noqa: E402
    DATASET,
    EPV_FLOOR,
    FEATURE_COLS_PRIMARY,
    gate_epv,
    temporal_holdout_auc,
)

RESULTS = Path(__file__).resolve().parents[4] / (
    "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results")
N_SPLITS = 5


def spatial_block_folds(unidades: pd.DataFrame, n_splits: int = N_SPLITS) -> list[dict]:
    """GroupKFold por bairro -- nenhum bairro aparece em treino e teste do mesmo fold."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = unidades["bairro"].values
    folds = []
    for i, (tr_idx, te_idx) in enumerate(gkf.split(unidades, groups=groups)):
        df_train = unidades.iloc[tr_idx].reset_index(drop=True)
        df_test = unidades.iloc[te_idx].reset_index(drop=True)
        bairros_train = set(df_train["bairro"].unique())
        bairros_test = set(df_test["bairro"].unique())
        overlap = bairros_train & bairros_test
        folds.append({
            "fold": i,
            "df_train": df_train,
            "df_test": df_test,
            "bairros_test": sorted(bairros_test),
            "n_bairros_test": len(bairros_test),
            "bairro_overlap_train_test": sorted(overlap),
        })
    return folds


def run_spatial_block_cv(unidades: pd.DataFrame, feature_cols: list[str],
                          n_splits: int = N_SPLITS) -> dict:
    folds = spatial_block_folds(unidades, n_splits)
    fold_rows = []
    for f in folds:
        assert not f["bairro_overlap_train_test"], (
            f"fold {f['fold']} tem bairro em treino e teste: {f['bairro_overlap_train_test']}")
        epv = gate_epv(f["df_train"], feature_cols)
        result = temporal_holdout_auc(f["df_train"], f["df_test"], feature_cols)
        fold_rows.append({
            "fold": f["fold"],
            "n_bairros_test": f["n_bairros_test"],
            "epv_treino": epv["epv"],
            "epv_passa": epv["passa"],
            **result,
        })
    df_folds = pd.DataFrame(fold_rows)
    aucs = df_folds["auc_holdout"].values
    summary = {
        "n_splits": n_splits,
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs, ddof=1)), 4),
        "auc_min": round(float(np.min(aucs)), 4),
        "auc_max": round(float(np.max(aucs)), 4),
        "all_epv_pass": bool(df_folds["epv_passa"].all()),
        "reference_loo_auc_shuffled_v20n": 0.6459,
        "reference_temporal_holdout_auc_v20o": 0.5246,
    }
    return {"fold_results": df_folds, "summary": summary}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(DATASET))
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--tag", default="v20p")
    p.add_argument("--n-splits", type=int, default=N_SPLITS)
    args = p.parse_args(argv)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    linhas = pd.read_csv(args.dataset)
    unidades = linhas[~linhas["e_unidade_duplicada"]].reset_index(drop=True)
    print(f"unidades: {len(unidades)} | bairros unicos: {unidades['bairro'].nunique()}")

    epv_full = gate_epv(unidades, FEATURE_COLS_PRIMARY)
    print(f"gate EPV (dataset completo, referencia): {epv_full['epv']} (piso {EPV_FLOOR})")

    print(f"[1] spatial block CV -- GroupKFold(n_splits={args.n_splits}) por bairro")
    out_run = run_spatial_block_cv(unidades, FEATURE_COLS_PRIMARY, args.n_splits)
    df_folds = out_run["fold_results"]
    summary = out_run["summary"]
    print(df_folds.to_string(index=False))
    print(json.dumps(summary, indent=2))

    df_folds.to_csv(out / f"{args.tag}_cv_blocos_espaciais_folds.csv", index=False)
    reports = {"feature_cols_primary": FEATURE_COLS_PRIMARY, "n_unidades": int(len(unidades)),
               "n_bairros": int(unidades["bairro"].nunique()), "summary": summary,
               "fold_results": df_folds.to_dict(orient="records")}
    (out / f"{args.tag}_relatorios_consolidados.json").write_text(
        json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
