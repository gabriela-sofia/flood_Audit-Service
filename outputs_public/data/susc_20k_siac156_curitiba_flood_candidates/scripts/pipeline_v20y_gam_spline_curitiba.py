"""SUSC-20Y -- GAM aditivo com B-splines por feature, Curitiba.

O SUSC-20X mostrou que interacao produto e limiar binario (termos simples num GLM) nao
recuperam o sinal nao-linear do GBM (AUC holdout 2026: GBM 0,5888 vs melhor tentativa
interpretavel 0,5295). A limitacao levantada no relatorio do SUSC-20X aponta pra proxima
tentativa natural: GAM (generalized additive model) com spline por feature, que capturaria a
FORMA real da partial dependence -- nao so um produto ou um corte -- mantendo a propriedade que
importa pra interpretabilidade causal: o modelo continua ADITIVO (sem interacao entre
features), entao o efeito de cada feature isolada e recuperavel diretamente (soma da base de
spline daquela feature vezes seu proprio coeficiente), sem precisar de partial_dependence pra
"congelar" as outras -- ja vem congelado pela propria forma aditiva do modelo.

Isso e estritamente mais forte que o SUSC-20X: em vez de 1 termo de interacao ou 1 limiar por
par, cada feature ganha uma curva de efeito flexivel (spline cubica, poucos nos), mas sem
NENHUMA interacao entre pares de features -- ou seja, se o achado do SUSC-20U/20V (GBM) depende
de interacao real entre features (nao so de forma nao-linear em cada uma isoladamente), o GAM
aditivo tambem vai falhar em recuperar o patamar do GBM. Isso e o proprio teste: se o GAM
recuperar o sinal, a nao-linearidade e por-feature (boa noticia, interpretavel). Se nao
recuperar, reforca a leitura do SUSC-20X de que ha interacao de ordem mais alta.

Nenhuma variavel nova, nenhum dado novo -- mesmas 5 features causais primarias (SUSC-20N/20O).

Uso:
    python pipeline_v20y_gam_spline_curitiba.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_v20o_temporal_holdout_curitiba import DATASET, EXPECTED_SIGN, FEATURE_COLS_PRIMARY  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results"

REFERENCE_LINEAR_HOLDOUT_2026 = 0.5246
REFERENCE_GBM_HOLDOUT_2026 = 0.5888
DEFAULT_N_KNOTS = 5
DEFAULT_DEGREE = 3
DEFAULT_C = 1.0


def fit_gam(df_train: pd.DataFrame, feature_cols: list[str], n_knots: int = DEFAULT_N_KNOTS,
            degree: int = DEFAULT_DEGREE, C: float = DEFAULT_C):
    """Ajusta 1 SplineTransformer por feature (independentes) + LogisticRegression aditiva
    sobre a base concatenada. Aditivo por construcao: nao ha termo cruzado entre features."""
    splines: dict[str, SplineTransformer] = {}
    parts = []
    for feat in feature_cols:
        sp = SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile", include_bias=False)
        Xb = sp.fit_transform(df_train[[feat]].values)
        splines[feat] = sp
        parts.append(Xb)
    Xtr = np.hstack(parts)
    ytr = df_train["label"].astype(int).values
    clf = LogisticRegression(penalty="l2", C=C, max_iter=3000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return splines, clf


def transform_with_splines(df: pd.DataFrame, feature_cols: list[str],
                            splines: dict[str, SplineTransformer]) -> np.ndarray:
    parts = [splines[f].transform(df[[f]].values) for f in feature_cols]
    return np.hstack(parts)


def gam_temporal_holdout(df_train: pd.DataFrame, df_test: pd.DataFrame, feature_cols: list[str],
                          n_knots: int = DEFAULT_N_KNOTS, degree: int = DEFAULT_DEGREE,
                          C: float = DEFAULT_C) -> dict:
    dtr = df_train.dropna(subset=feature_cols)
    dte = df_test.dropna(subset=feature_cols)
    splines, clf = fit_gam(dtr, feature_cols, n_knots, degree, C)
    Xte = transform_with_splines(dte, feature_cols, splines)
    yte = dte["label"].astype(int).values
    score = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, score)
    return {"n_train": int(len(dtr)), "n_test": int(len(dte)),
            "n_knots": n_knots, "degree": degree, "C": C,
            "holdout_auc_2026": round(float(auc), 4)}


def gam_spatial_block_cv(unidades: pd.DataFrame, feature_cols: list[str], n_splits: int = 5,
                          n_knots: int = DEFAULT_N_KNOTS, degree: int = DEFAULT_DEGREE,
                          C: float = DEFAULT_C) -> dict:
    gkf = GroupKFold(n_splits=n_splits)
    groups = unidades["bairro"].values
    aucs = []
    for tr_idx, te_idx in gkf.split(unidades, groups=groups):
        df_train = unidades.iloc[tr_idx].reset_index(drop=True)
        df_test = unidades.iloc[te_idx].reset_index(drop=True)
        r = gam_temporal_holdout(df_train, df_test, feature_cols, n_knots, degree, C)
        aucs.append(r["holdout_auc_2026"])
    aucs = np.array(aucs)
    return {"n_splits": n_splits, "auc_mean": round(float(aucs.mean()), 4),
            "auc_std": round(float(aucs.std(ddof=1)), 4),
            "auc_min": round(float(aucs.min()), 4), "auc_max": round(float(aucs.max()), 4)}


def gam_hyperparam_sensitivity(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for n_knots in (3, 4, 5, 6, 8):
        for degree in (2, 3):
            for C in (0.1, 1.0, 10.0):
                r = gam_temporal_holdout(df_train, df_test, feature_cols, n_knots, degree, C)
                rows.append(r)
    return pd.DataFrame(rows)


def additive_effect_curves(splines: dict[str, SplineTransformer], clf: LogisticRegression,
                            feature_cols: list[str], df_ref: pd.DataFrame,
                            grid_resolution: int = 30) -> pd.DataFrame:
    """Curva de efeito aditivo isolado por feature: spline(x) . coef_dessa_feature.
    Nao precisa congelar as outras features (o modelo ja e aditivo por construcao)."""
    offsets = []
    pos = 0
    for feat in feature_cols:
        n_out = splines[feat].n_features_out_
        offsets.append((feat, pos, pos + n_out))
        pos += n_out
    coef = clf.coef_[0]

    rows = []
    for feat, lo, hi in offsets:
        coef_feat = coef[lo:hi]
        x_min, x_max = df_ref[feat].min(), df_ref[feat].max()
        grid = np.linspace(x_min, x_max, grid_resolution)
        basis = splines[feat].transform(grid.reshape(-1, 1))
        effect = basis @ coef_feat
        diffs = np.diff(effect)
        n_sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
        direction = "crescente" if effect[-1] > effect[0] else "decrescente"
        rows.append({
            "feature": feat, "grid_min": round(float(x_min), 4), "grid_max": round(float(x_max), 4),
            "effect_at_min": round(float(effect[0]), 4), "effect_at_max": round(float(effect[-1]), 4),
            "effect_range": round(float(effect.max() - effect.min()), 4),
            "monotonic": bool(n_sign_changes == 0), "n_sign_changes_in_slope": n_sign_changes,
            "direction": direction, "expected_sign": EXPECTED_SIGN.get(feat),
            "direction_matches_expected": bool(
                (EXPECTED_SIGN.get(feat, 0) > 0 and direction == "crescente") or
                (EXPECTED_SIGN.get(feat, 0) < 0 and direction == "decrescente")),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(DATASET))
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--tag", default="v20y")
    args = p.parse_args(argv)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    linhas = pd.read_csv(args.dataset)
    linhas["event_date"] = pd.to_datetime(linhas["event_date"])
    unidades = linhas[~linhas["is_duplicate_observation_unit"]].reset_index(drop=True)
    unidades["year"] = unidades["event_date"].dt.year
    df_train = unidades[unidades["year"].isin([2023, 2024, 2025])].reset_index(drop=True)
    df_test = unidades[unidades["year"] == 2026].reset_index(drop=True)

    print(f"[1] holdout temporal com GAM (n_knots={DEFAULT_N_KNOTS}, degree={DEFAULT_DEGREE}, "
          f"C={DEFAULT_C})")
    holdout = gam_temporal_holdout(df_train, df_test, FEATURE_COLS_PRIMARY)
    print(json.dumps(holdout, indent=2))

    print("[2] spatial block CV (bairro) com GAM, config default")
    spatial = gam_spatial_block_cv(unidades, FEATURE_COLS_PRIMARY)
    print(json.dumps(spatial, indent=2))

    print("[3] sensibilidade a hiperparametro (n_knots x degree x C, 30 combos)")
    grid = gam_hyperparam_sensitivity(df_train, df_test, FEATURE_COLS_PRIMARY)
    grid.to_csv(out / f"{args.tag}_hyperparam_grid.csv", index=False)
    grid_summary = {
        "n_combos": int(len(grid)), "auc_min": round(float(grid["holdout_auc_2026"].min()), 4),
        "auc_median": round(float(grid["holdout_auc_2026"].median()), 4),
        "auc_max": round(float(grid["holdout_auc_2026"].max()), 4),
        "pct_above_gbm": round(float((grid["holdout_auc_2026"] >= REFERENCE_GBM_HOLDOUT_2026).mean() * 100), 1),
        "pct_above_linear_baseline": round(float((grid["holdout_auc_2026"] > REFERENCE_LINEAR_HOLDOUT_2026).mean() * 100), 1),
    }
    print(json.dumps(grid_summary, indent=2))

    print("[4] curvas de efeito aditivo por feature (config default, ajustado no treino completo)")
    splines, clf = fit_gam(df_train.dropna(subset=FEATURE_COLS_PRIMARY), FEATURE_COLS_PRIMARY)
    effects = additive_effect_curves(splines, clf, FEATURE_COLS_PRIMARY,
                                      df_train.dropna(subset=FEATURE_COLS_PRIMARY))
    effects.to_csv(out / f"{args.tag}_additive_effect_curves.csv", index=False)
    print(effects.to_string(index=False))

    resumo = {
        "reference_linear_baseline": REFERENCE_LINEAR_HOLDOUT_2026,
        "reference_gbm_v20u": REFERENCE_GBM_HOLDOUT_2026,
        "gam_holdout_auc_2026_default_config": holdout["holdout_auc_2026"],
        "gam_spatial_block_cv_auc_mean": spatial["auc_mean"],
        "gam_best_holdout_auc_grid": grid_summary["auc_max"],
        "recuperou_o_sinal_do_gbm": bool(grid_summary["auc_max"] >= REFERENCE_GBM_HOLDOUT_2026),
        "melhora_parcial_sobre_linear": bool(grid_summary["auc_max"] > REFERENCE_LINEAR_HOLDOUT_2026),
        "gap_para_gbm": round(REFERENCE_GBM_HOLDOUT_2026 - grid_summary["auc_max"], 4),
        "n_features_direction_matches_expected": int(effects["direction_matches_expected"].sum()),
        "n_features_monotonic": int(effects["monotonic"].sum()),
    }
    print(json.dumps(resumo, indent=2, ensure_ascii=False))

    reports = {"holdout_default": holdout, "spatial_block_cv_default": spatial,
               "hyperparam_grid_summary": grid_summary, "additive_effects": effects.to_dict(orient="records"),
               "resumo": resumo}
    (out / f"{args.tag}_all_reports.json").write_text(
        json.dumps(reports, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
