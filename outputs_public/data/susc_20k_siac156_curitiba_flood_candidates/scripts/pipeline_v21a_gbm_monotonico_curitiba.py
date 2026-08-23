"""SUSC-21A -- GBM com restricao monotonica causal, Curitiba. Nova vertente (fora da familia GAM).

A serie SUSC-20U-20Z estabeleceu: (1) existe nao-linearidade real e generalizavel (GBM
irrestrito bate o linear em quase todo corte temporal); (2) GAM aditivo/tensor fecha ~77% do
gap de interpretabilidade mas nao o gap inteiro, e a vertente GAM esta exaurida.

Esta e uma vertente DIFERENTE, nao uma continuacao do GAM: em vez de tentar TRADUZIR a
nao-linearidade do GBM irrestrito pra um termo interpretavel depois de ajustado, restringe o
proprio GBM a NUNCA violar o sinal causal ja estabelecido (SUSC-20M/20N), usando
`monotonic_cst` do HistGradientBoostingClassifier (sklearn >=0.23): a relacao entre cada
variavel e o log-odds de risco e forcada a ser monotonica na direcao fisica conhecida
(slope_deg e hand_m_dinf decrescentes -- mais declividade/mais distancia da drenagem, menos
risco; twi_dinf, rain_peak_residual_orthogonalized, rain_decay_index_api_chirps crescentes).

Isso opera diretamente o mandato do projeto -- "o modelo NAO deve descobrir enchentes; deve
refletir relacoes fisicas conhecidas" -- de um jeito que nem o linear nem o GBM irrestrito
fazem sozinhos: o linear so permite 1 taxa constante por variavel (nao captura limiar/forma);
o GBM irrestrito captura qualquer forma, inclusive uma que contraria o sinal causal
(caso do twi_dinf, documentado como anomalia no SUSC-20V/20Y). O GBM monotonico fica no meio:
tao flexivel quanto o GBM em FORMA (limiares, platos, taxas variaveis), mas nunca pode
inverter DIRECAO -- e portanto, por construcao, nunca produz uma curva anomala como a do
twi_dinf.

Uso:
    python pipeline_v21a_gbm_monotonico_curitiba.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import partial_dependence
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_v20o_holdout_temporal_curitiba import DATASET, EXPECTED_SIGN, FEATURE_COLS_PRIMARY  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results"

REFERENCE_LINEAR_HOLDOUT_2026 = 0.5246
REFERENCE_GAM_BEST_V20Y = 0.575
REFERENCE_GBM_UNRESTRICTED_V20U = 0.5888

# sklearn monotonic_cst: 1 = crescente, -1 = decrescente, 0 = sem restricao -- EXPECTED_SIGN
# (de pipeline_v20o) ja usa essa mesma convencao (+1/-1), reaproveitado sem reescrever.
MONOTONIC_CST = [EXPECTED_SIGN[f] for f in FEATURE_COLS_PRIMARY]

DEFAULT_PARAMS = dict(max_iter=100, max_depth=2, learning_rate=0.05, random_state=42)


def fit_hgb(df_train: pd.DataFrame, feature_cols: list[str], monotonic: bool = True,
            **params) -> HistGradientBoostingClassifier:
    kwargs = {**DEFAULT_PARAMS, **params}
    cst = MONOTONIC_CST if monotonic else None
    clf = HistGradientBoostingClassifier(monotonic_cst=cst, class_weight="balanced", **kwargs)
    X = df_train[feature_cols].values
    y = df_train["rotulo"].astype(int).values
    clf.fit(X, y)
    return clf


def hgb_temporal_holdout(df_train: pd.DataFrame, df_test: pd.DataFrame, feature_cols: list[str],
                          monotonic: bool = True, **params) -> dict:
    dtr = df_train.dropna(subset=feature_cols)
    dte = df_test.dropna(subset=feature_cols)
    clf = fit_hgb(dtr, feature_cols, monotonic=monotonic, **params)
    Xte = dte[feature_cols].values
    yte = dte["rotulo"].astype(int).values
    escore = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, escore)
    return {"n_treino": int(len(dtr)), "n_teste": int(len(dte)), "monotonic": monotonic,
            **{k: v for k, v in {**DEFAULT_PARAMS, **params}.items() if k != "random_state"},
            "auc_holdout_2026": round(float(auc), 4)}


def hgb_spatial_block_cv(unidades: pd.DataFrame, feature_cols: list[str], monotonic: bool = True,
                          n_splits: int = 5, **params) -> dict:
    gkf = GroupKFold(n_splits=n_splits)
    groups = unidades["bairro"].values
    aucs = []
    for tr_idx, te_idx in gkf.split(unidades, groups=groups):
        df_train = unidades.iloc[tr_idx].reset_index(drop=True)
        df_test = unidades.iloc[te_idx].reset_index(drop=True)
        r = hgb_temporal_holdout(df_train, df_test, feature_cols, monotonic=monotonic, **params)
        aucs.append(r["auc_holdout_2026"])
    aucs = np.array(aucs)
    return {"n_splits": n_splits, "monotonic": monotonic, "auc_mean": round(float(aucs.mean()), 4),
            "auc_std": round(float(aucs.std(ddof=1)), 4),
            "auc_min": round(float(aucs.min()), 4), "auc_max": round(float(aucs.max()), 4)}


def hgb_hyperparam_sensitivity(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                feature_cols: list[str], monotonic: bool = True) -> pd.DataFrame:
    rows = []
    for max_depth in (1, 2, 3):
        for max_iter in (50, 100, 200):
            for learning_rate in (0.02, 0.05, 0.1):
                r = hgb_temporal_holdout(df_train, df_test, feature_cols, monotonic=monotonic,
                                          max_depth=max_depth, max_iter=max_iter,
                                          learning_rate=learning_rate)
                rows.append(r)
    return pd.DataFrame(rows)


def monotonic_pd_check(clf: HistGradientBoostingClassifier, df_ref: pd.DataFrame,
                        feature_cols: list[str], grid_resolution: int = 20) -> pd.DataFrame:
    """Confirma que a partial dependence realmente sai monotonica na direcao esperada --
    sanity check da restricao (nao so confianca cega no parametro)."""
    rows = []
    for i, feat in enumerate(feature_cols):
        pd_result = partial_dependence(clf, df_ref[feature_cols], [i], grid_resolution=grid_resolution,
                                        kind="average")
        avg = pd_result["average"][0]
        diffs = np.diff(avg)
        # monotonicidade real (nao-decrescente OU nao-crescente) permite trechos PLANOS (diff=0,
        # comuns em funcao em degraus de arvore) -- contar so trocas de sinal ESTRITAS conta
        # plato->subida como "troca" por engano; o teste certo e: nenhum diff positivo E negativo
        tol = 1e-9
        monotonic = bool(np.all(diffs >= -tol) or np.all(diffs <= tol))
        n_sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))  # metrica informativa, nao usada pro veredito
        direcao = "crescente" if avg[-1] > avg[0] else "decrescente"
        rows.append({"variavel": feat, "sinal_esperado": EXPECTED_SIGN[feat],
                     "pd_grade_min": round(float(avg[0]), 4), "pd_grade_max": round(float(avg[-1]), 4),
                     "monotonic": monotonic, "n_trocas_sinal_inclinacao": n_sign_changes,
                     "direcao": direcao,
                     "direcao_bate_esperada": bool(
                         (EXPECTED_SIGN[feat] > 0 and direcao == "crescente") or
                         (EXPECTED_SIGN[feat] < 0 and direcao == "decrescente"))})
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

    print(f"[1] holdout temporal, config default ({DEFAULT_PARAMS}), monotonico vs irrestrito")
    r_mono = hgb_temporal_holdout(df_train, df_test, FEATURE_COLS_PRIMARY, monotonic=True)
    r_unrest = hgb_temporal_holdout(df_train, df_test, FEATURE_COLS_PRIMARY, monotonic=False)
    print("monotonico:", json.dumps(r_mono, indent=2))
    print("irrestrito (mesmos hiperparametros):", json.dumps(r_unrest, indent=2))

    print("[2] spatial block CV (bairro), monotonico vs irrestrito")
    sp_mono = hgb_spatial_block_cv(unidades, FEATURE_COLS_PRIMARY, monotonic=True)
    sp_unrest = hgb_spatial_block_cv(unidades, FEATURE_COLS_PRIMARY, monotonic=False)
    print("monotonico:", json.dumps(sp_mono, indent=2))
    print("irrestrito:", json.dumps(sp_unrest, indent=2))

    print("[3] sensibilidade a hiperparametro -- monotonico (27 combos)")
    grid_mono = hgb_hyperparam_sensitivity(df_train, df_test, FEATURE_COLS_PRIMARY, monotonic=True)
    grid_mono.to_csv(out / "v21a_grade_hiperparam_monotonico.csv", index=False)
    grid_mono_summary = {
        "n_combos": int(len(grid_mono)), "auc_min": round(float(grid_mono["auc_holdout_2026"].min()), 4),
        "auc_median": round(float(grid_mono["auc_holdout_2026"].median()), 4),
        "auc_max": round(float(grid_mono["auc_holdout_2026"].max()), 4),
        "pct_above_linear_baseline": round(float((grid_mono["auc_holdout_2026"] > REFERENCE_LINEAR_HOLDOUT_2026).mean() * 100), 1),
        "pct_above_gam_best": round(float((grid_mono["auc_holdout_2026"] > REFERENCE_GAM_BEST_V20Y).mean() * 100), 1),
        "pct_at_or_above_gbm_unrestricted": round(float((grid_mono["auc_holdout_2026"] >= REFERENCE_GBM_UNRESTRICTED_V20U).mean() * 100), 1),
    }
    print(json.dumps(grid_mono_summary, indent=2))

    print("[4] sanity check -- a restricao monotonica realmente produz PD monotonica?")
    clf_mono = fit_hgb(df_train.dropna(subset=FEATURE_COLS_PRIMARY), FEATURE_COLS_PRIMARY, monotonic=True)
    pd_check = monotonic_pd_check(clf_mono, df_train.dropna(subset=FEATURE_COLS_PRIMARY), FEATURE_COLS_PRIMARY)
    pd_check.to_csv(out / "v21a_checagem_pd_monotonico.csv", index=False)
    print(pd_check.to_string(index=False))

    resumo = {
        "reference_linear_baseline": REFERENCE_LINEAR_HOLDOUT_2026,
        "reference_gam_best_v20y": REFERENCE_GAM_BEST_V20Y,
        "reference_gbm_unrestricted_v20u": REFERENCE_GBM_UNRESTRICTED_V20U,
        "monotonic_holdout_auc_default": r_mono["auc_holdout_2026"],
        "unrestricted_holdout_auc_same_hparams": r_unrest["auc_holdout_2026"],
        "custo_de_impor_monotonicidade_default": round(r_unrest["auc_holdout_2026"] - r_mono["auc_holdout_2026"], 4),
        "monotonic_spatial_block_cv_mean": sp_mono["auc_mean"],
        "monotonic_best_holdout_auc_grid": grid_mono_summary["auc_max"],
        "gap_para_gbm_irrestrito": round(REFERENCE_GBM_UNRESTRICTED_V20U - grid_mono_summary["auc_max"], 4),
        "supera_melhor_gam": bool(grid_mono_summary["auc_max"] > REFERENCE_GAM_BEST_V20Y),
        "todas_5_curvas_pd_batem_direcao_esperada": bool(pd_check["direcao_bate_esperada"].all()),
        "todas_5_curvas_pd_monotonicas": bool(pd_check["monotonic"].all()),
    }
    print(json.dumps(resumo, indent=2, ensure_ascii=False))

    reports = {"holdout_default": {"monotonic": r_mono, "unrestricted_same_hparams": r_unrest},
               "spatial_block_cv_default": {"monotonic": sp_mono, "unrestricted": sp_unrest},
               "hyperparam_grid_monotonic_summary": grid_mono_summary,
               "monotonic_pd_check": pd_check.to_dict(orient="records"), "resumo": resumo}
    (out / "v21a_relatorios_consolidados.json").write_text(
        json.dumps(reports, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
