"""SUSC-20V -- decomposicao do GBM (partial dependence) + varredura de mais classes de modelo.

Pedido explicito (2026-08-02): testar tudo, inclusive fora das regras atuais de
interpretabilidade, pra ver se ALGUMA classe de modelo recupera sinal prospectivo real em
2026. Este script faz duas coisas:

1. **Decompoe** o achado do SUSC-20U (GBM raso, AUC holdout=0,5888, IC exclui acaso) via
   partial dependence (sklearn, sem instalar nada novo -- shap/xgboost nao instalaram no
   sandbox por timeout de rede, documentado como limitacao, nao contornado por workaround
   arriscado) -- pra entender QUAL relacao (limiar, direcao) o modelo nao-linear capturou por
   variavel, e se isso e fisicamente interpretavel mesmo sem ser um coeficiente causal formal.
2. **Varre mais classes de modelo** nas MESMAS 5 features causais, sem filtro de
   interpretabilidade (SVM-RBF, MLP raso, ExtraTrees, AdaBoost) -- todas ja disponiveis no
   sklearn, nenhuma instalacao nova. Objetivo: mapear se o sinal do SUSC-20U e um achado do
   GBM especificamente ou de nao-linearidade em geral.

Regra que NAO foi relaxada, mesmo com o pedido de testar tudo: nenhuma variavel nova
derivada do rotulo, nenhum escore/proxy definido a partir da propria ocorrencia de queixa --
isso invalidaria a validacao (vazamento), nao e uma questao de estilo/interpretabilidade, e
foi mantido como piso de validade cientifica, nao de preferencia.

Uso:
    python pipeline_v20v_decomposicao_gbm_varredura_curitiba.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, RandomForestClassifier)
from sklearn.inspection import partial_dependence
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_v20o_holdout_temporal_curitiba import DATASET, FEATURE_COLS_PRIMARY  # noqa: E402
from pipeline_v20u_diagnostico_nao_linear_curitiba import GBM_PARAMS  # noqa: E402

RESULTS = Path(__file__).resolve().parents[4] / (
    "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results")


def gbm_partial_dependence(df_train: pd.DataFrame, feature_cols: list[str]) -> dict:
    dtr = df_train.dropna(subset=feature_cols)
    X, y = dtr[feature_cols].values, dtr["rotulo"].astype(int).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = GradientBoostingClassifier(**GBM_PARAMS)
    clf.fit(Xs, y)

    out = {}
    for i, feat in enumerate(feature_cols):
        pd_result = partial_dependence(clf, Xs, [i], kind="average", grid_resolution=15)
        grid_std = pd_result["grid_values"][0]
        # de volta pra escala original da variavel (desfaz o StandardScaler so nessa dimensao)
        grid_orig = grid_std * scaler.scale_[i] + scaler.mean_[i]
        avg = pd_result["average"][0]
        # tendencia global: sobe, desce, ou nao-monotonico (checa se o sinal da diferenca
        # muda de sinal ao longo do grid)
        diffs = np.diff(avg)
        sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
        out[feat] = {
            "grade_min": round(float(grid_orig.min()), 4), "grade_max": round(float(grid_orig.max()), 4),
            "pd_grade_min": round(float(avg[0]), 4), "pd_grade_max": round(float(avg[-1]), 4),
            "pd_range": round(float(avg.max() - avg.min()), 4),
            "monotonic": bool(sign_changes == 0),
            "direcao": "crescente" if avg[-1] > avg[0] else "decrescente",
            "n_trocas_sinal_inclinacao": sign_changes,
        }
    return out


def model_sweep(df_train: pd.DataFrame, df_test: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    dtr = df_train.dropna(subset=feature_cols)
    dte = df_test.dropna(subset=feature_cols)
    Xtr, ytr = dtr[feature_cols].values, dtr["rotulo"].astype(int).values
    Xte, yte = dte[feature_cols].values, dte["rotulo"].astype(int).values
    scaler = StandardScaler().fit(Xtr)
    Xtrs, Xtes = scaler.transform(Xtr), scaler.transform(Xte)

    models = {
        "SVM_RBF_C1": SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=42),
        "SVM_RBF_C10": SVC(kernel="rbf", C=10.0, class_weight="balanced", probability=True, random_state=42),
        "MLP_16_8": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42),
        "MLP_8": MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=42),
        "ExtraTrees_d3": ExtraTreesClassifier(n_estimators=200, max_depth=3, class_weight="balanced", random_state=42),
        "AdaBoost_d1": AdaBoostClassifier(n_estimators=100, random_state=42),
        "RandomForest_d3": RandomForestClassifier(n_estimators=200, max_depth=3, class_weight="balanced", random_state=42),
        "GBM_referencia_v20u": GradientBoostingClassifier(**GBM_PARAMS),
    }
    rows = []
    for name, clf in models.items():
        clf.fit(Xtrs, ytr)
        escore = clf.predict_proba(Xtes)[:, 1]
        auc = roc_auc_score(yte, escore)
        rows.append({"modelo": name, "n_treino": int(len(dtr)), "n_teste": int(len(dte)),
                     "auc_holdout": round(float(auc), 4)})
    df = pd.DataFrame(rows).sort_values("auc_holdout", ascending=False)
    return df


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

    print("[1] partial dependence do GBM (SUSC-20U) por variavel")
    d1 = gbm_partial_dependence(df_train, FEATURE_COLS_PRIMARY)
    print(json.dumps(d1, indent=2, ensure_ascii=False))
    reports["1_gbm_partial_dependence"] = d1

    print("\n[2] varredura de classes de modelo (holdout temporal 2026)")
    d2 = model_sweep(df_train, df_2026, FEATURE_COLS_PRIMARY)
    print(d2.to_string(index=False))
    d2.to_csv(out / "v20v_model_sweep_holdout_2026.csv", index=False)
    reports["2_model_sweep"] = d2.to_dict(orient="records")
    reports["reference_baseline_linear_v20o"] = 0.5246
    reports["reference_gbm_v20u"] = 0.5888
    reports["nota_dependencias_nao_instaladas"] = (
        "shap e xgboost nao instalaram no sandbox (timeout de rede em pip install) -- "
        "decomposicao feita via sklearn.inspection.partial_dependence (ja disponivel), "
        "sem workaround de rede arriscado.")

    (out / "v20v_relatorios_consolidados.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
