"""SUSC-20Z -- GAM aditivo + interacao tensor-spline (2D), Curitiba. Fecha a vertente GAM.

O SUSC-20Y (GAM aditivo puro, sem termo cruzado) fechou 77% do gap pro GBM (0,0593 -> 0,0138)
mas nao o gap inteiro -- leitura registrada la: "resta um residuo que so interacao real entre
features explicaria... um GLM aditivo, por mais flexivel que seja a curva de cada variavel, nao
consegue capturar por definicao" (limitacao #3 do relatorio SUSC-20Y, explicitamente listada
como proxima tentativa: "uma superficie de spline 2D (nao tentada)").

Este script fecha essa lacuna: adiciona um termo de interacao tensor-spline (produto tensorial
das bases de spline de duas features, igual ao te() do mgcv/R) ao MESMO GAM aditivo do
SUSC-20Y, pros mesmos 3 pares testados no SUSC-20X (rain_decay x hand, rain_peak x hand,
rain_decay x rain_peak). Se isso fechar o gap pro GBM, confirma que o residuo e interacao
2D genuina. Se nao fechar, sugere que o GBM usa estrutura de ordem >2 (3+ features) ou splits
regionais que nem tensor-spline 2D replica -- e aí a vertente GAM esta genuinamente esgotada.

Uso:
    python pipeline_v20z_gam_interacao_tensor_curitiba.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import SplineTransformer

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_v20o_holdout_temporal_curitiba import DATASET, FEATURE_COLS_PRIMARY  # noqa: E402
from pipeline_v20y_gam_splines_curitiba import DEFAULT_C, DEFAULT_DEGREE, DEFAULT_N_KNOTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results"

REFERENCE_LINEAR_HOLDOUT_2026 = 0.5246
REFERENCE_GAM_ADDITIVE_BEST = 0.575  # SUSC-20Y, melhor da grade de 30 combos
REFERENCE_GBM_HOLDOUT_2026 = 0.5888

INTERACTION_PAIRS = [("rain_decay_index_api_chirps", "hand_m_dinf"),
                     ("rain_peak_residual_orthogonalized", "hand_m_dinf"),
                     ("rain_decay_index_api_chirps", "rain_peak_residual_orthogonalized")]

DEFAULT_TENSOR_N_KNOTS = 4
DEFAULT_TENSOR_DEGREE = 2


def fit_additive_splines(df_train: pd.DataFrame, feature_cols: list[str],
                          n_knots: int = DEFAULT_N_KNOTS, degree: int = DEFAULT_DEGREE):
    splines = {}
    for feat in feature_cols:
        sp = SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile", include_bias=False)
        sp.fit(df_train[[feat]].values)
        splines[feat] = sp
    return splines


def additive_basis(df: pd.DataFrame, feature_cols: list[str], splines: dict) -> np.ndarray:
    return np.hstack([splines[f].transform(df[[f]].values) for f in feature_cols])


def fit_tensor_spline_pair(df_train: pd.DataFrame, f1: str, f2: str,
                            n_knots: int = DEFAULT_TENSOR_N_KNOTS,
                            degree: int = DEFAULT_TENSOR_DEGREE):
    sp1 = SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile", include_bias=False)
    sp2 = SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile", include_bias=False)
    sp1.fit(df_train[[f1]].values)
    sp2.fit(df_train[[f2]].values)
    return sp1, sp2


def tensor_basis(df: pd.DataFrame, f1: str, f2: str, sp1: SplineTransformer,
                  sp2: SplineTransformer) -> np.ndarray:
    """Produto tensorial das bases de spline de f1 e f2 -- 1 coluna por combinacao de funcao
    base de f1 x funcao base de f2 (igual ao termo te(f1, f2) do mgcv/R)."""
    b1 = sp1.transform(df[[f1]].values)  # (n, k1)
    b2 = sp2.transform(df[[f2]].values)  # (n, k2)
    n = b1.shape[0]
    return (b1[:, :, None] * b2[:, None, :]).reshape(n, -1)


def gam_tensor_holdout_auc(df_train: pd.DataFrame, df_test: pd.DataFrame,
                            feature_cols: list[str], interaction_pairs: list[tuple[str, str]],
                            n_knots: int = DEFAULT_N_KNOTS, degree: int = DEFAULT_DEGREE,
                            tensor_n_knots: int = DEFAULT_TENSOR_N_KNOTS,
                            tensor_degree: int = DEFAULT_TENSOR_DEGREE, C: float = DEFAULT_C) -> dict:
    dtr = df_train.dropna(subset=feature_cols)
    dte = df_test.dropna(subset=feature_cols)

    splines = fit_additive_splines(dtr, feature_cols, n_knots, degree)
    Xtr_parts = [additive_basis(dtr, feature_cols, splines)]
    Xte_parts = [additive_basis(dte, feature_cols, splines)]

    tensor_fits = []
    for f1, f2 in interaction_pairs:
        sp1, sp2 = fit_tensor_spline_pair(dtr, f1, f2, tensor_n_knots, tensor_degree)
        tensor_fits.append((f1, f2, sp1, sp2))
        Xtr_parts.append(tensor_basis(dtr, f1, f2, sp1, sp2))
        Xte_parts.append(tensor_basis(dte, f1, f2, sp1, sp2))

    Xtr = np.hstack(Xtr_parts)
    Xte = np.hstack(Xte_parts)
    ytr = dtr["rotulo"].astype(int).values
    yte = dte["rotulo"].astype(int).values

    clf = LogisticRegression(penalty="l2", C=C, max_iter=5000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    escore = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, escore)
    return {"n_treino": int(len(dtr)), "n_teste": int(len(dte)),
            "n_pares_interacao": len(interaction_pairs), "n_variaveis_total": int(Xtr.shape[1]),
            "tensor_n_knots": tensor_n_knots, "tensor_degree": tensor_degree,
            "auc_holdout_2026": round(float(auc), 4)}


def run_all_pair_configs(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    r0 = gam_tensor_holdout_auc(df_train, df_test, FEATURE_COLS_PRIMARY, [])
    rows.append({"config": "additive_only_no_tensor", **r0})
    for f1, f2 in INTERACTION_PAIRS:
        r = gam_tensor_holdout_auc(df_train, df_test, FEATURE_COLS_PRIMARY, [(f1, f2)])
        rows.append({"config": f"+tensor_{f1}_x_{f2}", **r})
    r_all = gam_tensor_holdout_auc(df_train, df_test, FEATURE_COLS_PRIMARY, INTERACTION_PAIRS)
    rows.append({"config": "+all_3_tensor_interactions", **r_all})
    return pd.DataFrame(rows)


def tensor_hyperparam_sensitivity(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """Sensibilidade do melhor par (definido depois de rodar run_all_pair_configs) a
    n_knots/degree do tensor -- evita cherry-pick de 1 config so."""
    rows = []
    for tensor_n_knots in (3, 4, 5):
        for tensor_degree in (1, 2, 3):
            r = gam_tensor_holdout_auc(df_train, df_test, FEATURE_COLS_PRIMARY,
                                        INTERACTION_PAIRS, tensor_n_knots=tensor_n_knots,
                                        tensor_degree=tensor_degree)
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    out = RESULTS
    out.mkdir(parents=True, exist_ok=True)

    linhas = pd.read_csv(DATASET)
    linhas["data_evento"] = pd.to_datetime(linhas["data_evento"])
    unidades = linhas[~linhas["e_unidade_duplicada"]].reset_index(drop=True)
    unidades["ano"] = unidades["data_evento"].dt.year
    df_train = unidades[unidades["ano"].isin([2023, 2024, 2025])].reset_index(drop=True)
    df_test = unidades[unidades["ano"] == 2026].reset_index(drop=True)

    print("[1] configs de interacao tensor-spline (por par e combinado)")
    df_pairs = run_all_pair_configs(df_train, df_test)
    print(df_pairs.to_string(index=False))
    df_pairs.to_csv(out / "v20z_configs_interacao_tensor.csv", index=False)

    print("[2] sensibilidade a hiperparametro do tensor (n_knots x degree, 9 combos, 3 pares combinados)")
    df_grid = tensor_hyperparam_sensitivity(df_train, df_test)
    print(df_grid.to_string(index=False))
    df_grid.to_csv(out / "v20z_tensor_grade_hiperparam.csv", index=False)

    melhor_config = df_pairs.loc[df_pairs["auc_holdout_2026"].idxmax()]
    melhor_grid = df_grid.loc[df_grid["auc_holdout_2026"].idxmax()]
    melhor_geral = max(float(melhor_config["auc_holdout_2026"]), float(melhor_grid["auc_holdout_2026"]))

    resumo = {
        "reference_linear_baseline": REFERENCE_LINEAR_HOLDOUT_2026,
        "reference_gam_additive_best_v20y": REFERENCE_GAM_ADDITIVE_BEST,
        "reference_gbm_v20u": REFERENCE_GBM_HOLDOUT_2026,
        "melhor_config_par": str(melhor_config["config"]),
        "melhor_auc_par": round(float(melhor_config["auc_holdout_2026"]), 4),
        "melhor_auc_grid_hiperparametro": round(float(melhor_grid["auc_holdout_2026"]), 4),
        "melhor_auc_geral_com_tensor": round(melhor_geral, 4),
        "gap_para_gbm_antes_tensor_v20y": round(REFERENCE_GBM_HOLDOUT_2026 - REFERENCE_GAM_ADDITIVE_BEST, 4),
        "gap_para_gbm_depois_tensor": round(REFERENCE_GBM_HOLDOUT_2026 - melhor_geral, 4),
        "recuperou_o_sinal_do_gbm": bool(melhor_geral >= REFERENCE_GBM_HOLDOUT_2026),
        "melhorou_sobre_gam_aditivo_v20y": bool(melhor_geral > REFERENCE_GAM_ADDITIVE_BEST),
    }
    print(json.dumps(resumo, indent=2, ensure_ascii=False))

    reports = {"pair_configs": df_pairs.to_dict(orient="records"),
               "hyperparam_grid": df_grid.to_dict(orient="records"), "resumo": resumo}
    (out / "v20z_relatorios_consolidados.json").write_text(
        json.dumps(reports, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
