"""SUSC-20U -- diagnostico de nao-linearidade (GBM raso) sobre as MESMAS 5 features causais.

Todas as vertentes lineares testadas ate aqui (Firth, logistic simples, PU bagging, peso de
recencia) ficam presas perto de AUC~0,52 no holdout temporal de 2026. Este diagnostico
pergunta uma coisa distinta: sera que a relacao entre as features fisico-causais e o rotulo e
NAO-LINEAR (limiares, interacoes), e um modelo linear simplesmente nao consegue captura-la --
independente de qual seja a causa da mudanca ano-a-ano?

**Isto e um diagnostico, nao uma proposta de rota primaria.** Gradient boosting raso
(max_depth=2, sem busca de hiperparametro) sobre as MESMAS 5 features causais de sempre
(nenhuma variavel nova, nenhuma orbital/proxy). Nao respeita o mesmo piso de EPV que rege a
rota Firth -- uma arvore rasa ja tem mais parametros efetivos que 5 coeficientes lineares, e
isso e uma limitacao real, nao escondida.

Uso:
    python pipeline_v20u_diagnostico_nao_linear_curitiba.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_v20o_holdout_temporal_curitiba import DATASET, FEATURE_COLS_PRIMARY  # noqa: E402

RESULTS = Path(__file__).resolve().parents[4] / (
    "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results")

GBM_PARAMS = dict(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42)


def gbm_temporal_holdout_with_ci(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                  feature_cols: list[str], n_boot: int = 2000,
                                  seed: int = 20260802) -> dict:
    dtr = df_train.dropna(subset=feature_cols)
    dte = df_test.dropna(subset=feature_cols).reset_index(drop=True)
    Xtr, ytr = dtr[feature_cols].values, dtr["rotulo"].astype(int).values
    Xte, yte = dte[feature_cols].values, dte["rotulo"].astype(int).values
    scaler = StandardScaler().fit(Xtr)
    clf = GradientBoostingClassifier(**GBM_PARAMS)
    clf.fit(scaler.transform(Xtr), ytr)
    escore = clf.predict_proba(scaler.transform(Xte))[:, 1]
    point = roc_auc_score(yte, escore)

    rng = np.random.default_rng(seed)
    n = len(dte)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = yte[idx], escore[idx]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(roc_auc_score(yb, sb))
    ic95_lo, ic95_hi = np.percentile(boots, [2.5, 97.5])
    importances = dict(zip(feature_cols, [round(float(x), 4) for x in clf.feature_importances_]))
    return {"n_treino": int(len(dtr)), "n_teste": int(len(dte)), "point_auc": round(float(point), 4),
            "ci_95_low": round(float(ic95_lo), 4), "ci_95_high": round(float(ic95_hi), 4),
            "ci_excludes_0_5_chance": bool(ic95_lo > 0.5),
            "feature_importances": importances,
            "reference_baseline_linear_temporal_holdout_auc_v20o": 0.5246}


def gbm_spatial_block_cv(unidades: pd.DataFrame, feature_cols: list[str],
                          n_splits: int = 5) -> pd.DataFrame:
    dnd = unidades.dropna(subset=feature_cols).reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for i, (tr_idx, te_idx) in enumerate(gkf.split(dnd, groups=dnd["bairro"].values)):
        dtr, dte = dnd.iloc[tr_idx], dnd.iloc[te_idx]
        assert not (set(dtr["bairro"]) & set(dte["bairro"])), f"fold {i} vaza bairro"
        Xtr, ytr = dtr[feature_cols].values, dtr["rotulo"].astype(int).values
        Xte, yte = dte[feature_cols].values, dte["rotulo"].astype(int).values
        scaler = StandardScaler().fit(Xtr)
        clf = GradientBoostingClassifier(**GBM_PARAMS)
        clf.fit(scaler.transform(Xtr), ytr)
        escore = clf.predict_proba(scaler.transform(Xte))[:, 1]
        rows.append({"fold": i, "n_treino": int(len(dtr)), "n_teste": int(len(dte)),
                     "auc_holdout_gbm": round(float(roc_auc_score(yte, escore)), 4)})
    return pd.DataFrame(rows)


def gbm_hyperparam_sensitivity(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                feature_cols: list[str]) -> pd.DataFrame:
    """Confere se o resultado sobrevive a variacoes razoaveis de hiperparametro (nao foi
    escolhido por busca -- checa se e fragil a mudanca pequena)."""
    dtr = df_train.dropna(subset=feature_cols)
    dte = df_test.dropna(subset=feature_cols)
    Xtr, ytr = dtr[feature_cols].values, dtr["rotulo"].astype(int).values
    Xte, yte = dte[feature_cols].values, dte["rotulo"].astype(int).values
    scaler = StandardScaler().fit(Xtr)
    Xtrs, Xtes = scaler.transform(Xtr), scaler.transform(Xte)
    rows = []
    for depth in [1, 2, 3]:
        for n_est in [50, 100, 200]:
            for lr in [0.02, 0.05, 0.1]:
                clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=depth,
                                                  learning_rate=lr, random_state=42)
                clf.fit(Xtrs, ytr)
                escore = clf.predict_proba(Xtes)[:, 1]
                auc = roc_auc_score(yte, escore)
                rows.append({"max_depth": depth, "n_estimators": n_est, "learning_rate": lr,
                             "auc_holdout": round(float(auc), 4)})
    return pd.DataFrame(rows)


def main() -> int:
    out = RESULTS
    out.mkdir(parents=True, exist_ok=True)
    linhas = pd.read_csv(DATASET)
    linhas["data_evento"] = pd.to_datetime(linhas["data_evento"])
    unidades = linhas[~linhas["e_unidade_duplicada"]].reset_index(drop=True)
    unidades["ano"] = unidades["data_evento"].dt.year

    df_train = unidades[unidades["ano"].isin([2023, 2024, 2025])].reset_index(drop=True)
    df_2026 = unidades[unidades["ano"] == 2026].reset_index(drop=True)

    reports = {}

    print("[1] GBM raso -- holdout temporal 2026, com bootstrap CI")
    d1 = gbm_temporal_holdout_with_ci(df_train, df_2026, FEATURE_COLS_PRIMARY)
    print(json.dumps(d1, indent=2))
    reports["1_gbm_temporal_holdout_ci"] = d1

    print("\n[2] GBM raso -- spatial block CV (bairro)")
    d2 = gbm_spatial_block_cv(unidades, FEATURE_COLS_PRIMARY)
    print(d2.to_string(index=False))
    d2.to_csv(out / "v20u_cv_blocos_espaciais_gbm.csv", index=False)
    mean_auc = round(float(d2["auc_holdout_gbm"].mean()), 4)
    reports["2_gbm_spatial_block_cv"] = {"fold_results": d2.to_dict(orient="records"),
                                          "auc_mean": mean_auc,
                                          "reference_baseline_linear_spatial_block_cv_v20p": 0.6442}
    print(f"AUC medio spatial block CV (GBM) = {mean_auc} (referencia linear = 0.6442)")

    print("\n[3] sensibilidade a hiperparametro (27 combinacoes)")
    d3 = gbm_hyperparam_sensitivity(df_train, df_2026, FEATURE_COLS_PRIMARY)
    print(d3.to_string(index=False))
    d3.to_csv(out / "v20u_sensibilidade_hiperparam_gbm.csv", index=False)
    reports["3_hyperparam_sensitivity"] = {
        "auc_min": round(float(d3["auc_holdout"].min()), 4),
        "auc_max": round(float(d3["auc_holdout"].max()), 4),
        "auc_median": round(float(d3["auc_holdout"].median()), 4),
        "frac_configs_above_0_55": round(float((d3["auc_holdout"] > 0.55).mean()), 4),
    }
    print(json.dumps(reports["3_hyperparam_sensitivity"], indent=2))

    (out / "v20u_relatorios_consolidados.json").write_text(
        json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
