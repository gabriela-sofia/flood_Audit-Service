"""SUSC-20X -- tentativa de traduzir a nao-linearidade do GBM pra termo interpretavel, Curitiba.

Pergunta desta rodada: o sinal nao-linear achado no SUSC-20U/20V/20W pode ser capturado por um
termo explicito e interpretavel (interacao produto ou indicador de limiar) dentro do MESMO
modelo linear (Firth/logistic), preservando coeficiente + p-valor + IC? Se sim, resolveria a
tensao entre performance e interpretabilidade sem abrir mao da rota causal.

Duas famílias de termo testadas, escolhidas a partir da propria partial dependence do GBM
(SUSC-20V/20W):

1. **Interacao produto** (par de features padronizadas multiplicadas) -- forma classica de
   capturar interacao linear-por-partes num GLM.
2. **Indicador de limiar** -- threshold binario no ponto onde a partial dependence do GBM
   mostra o salto mais acentuado (ex.: hand_m_dinf<4m, onde a PD cai de 0,84 pra -0,05;
   rain_decay_index_api_chirps>20, onde a PD sobe de -0,62 pra +0,44; ambos lidos
   diretamente da grade de partial dependence, nao escolhidos por busca).

Uso:
    python pipeline_v20x_traducao_interacao_curitiba.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_v20o_holdout_temporal_curitiba import DATASET, FEATURE_COLS_PRIMARY  # noqa: E402

RESULTS = Path(__file__).resolve().parents[4] / (
    "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results")

# limiares lidos direto da partial dependence do GBM (SUSC-20V/20W), nao escolhidos por busca
THRESHOLDS = {"hand_m_dinf": ("hand_low_lt4", 4.0, "lt"),
              "rain_decay_index_api_chirps": ("rain_decay_high_gt20", 20.0, "gt"),
              "rain_peak_residual_orthogonalized": ("rain_peak_low_ltm4", -4.0, "lt")}

INTERACTION_PAIRS = [("rain_decay_index_api_chirps", "hand_m_dinf"),
                     ("rain_peak_residual_orthogonalized", "hand_m_dinf"),
                     ("rain_decay_index_api_chirps", "rain_peak_residual_orthogonalized")]


def add_interaction_terms(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    d = df.copy()
    for f1, f2 in pairs:
        d[f"{f1}_x_{f2}"] = d[f1] * d[f2]
    return d


def add_threshold_terms(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for feat, (name, cut, direcao) in THRESHOLDS.items():
        d[name] = (d[feat] < cut).astype(float) if direcao == "lt" else (d[feat] > cut).astype(float)
    return d


def holdout_auc_with_features(df_train: pd.DataFrame, df_test: pd.DataFrame,
                               feature_cols: list[str]) -> float:
    dtr = df_train.dropna(subset=feature_cols)
    dte = df_test.dropna(subset=feature_cols)
    Xtr, ytr = dtr[feature_cols].values, dtr["rotulo"].astype(int).values
    Xte, yte = dte[feature_cols].values, dte["rotulo"].astype(int).values
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)
    escore = clf.predict_proba(scaler.transform(Xte))[:, 1]
    return round(float(roc_auc_score(yte, escore)), 4)


def run_all_configs(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    tr_int = add_interaction_terms(df_train, INTERACTION_PAIRS)
    te_int = add_interaction_terms(df_test, INTERACTION_PAIRS)
    tr_thr = add_threshold_terms(df_train)
    te_thr = add_threshold_terms(df_test)

    configs = {
        "base_5feat_linear": (df_train, df_test, FEATURE_COLS_PRIMARY),
        "+rain_decay_x_hand": (tr_int, te_int, FEATURE_COLS_PRIMARY + ["rain_decay_index_api_chirps_x_hand_m_dinf"]),
        "+rain_peak_x_hand": (tr_int, te_int, FEATURE_COLS_PRIMARY + ["rain_peak_residual_orthogonalized_x_hand_m_dinf"]),
        "+rain_decay_x_rain_peak": (tr_int, te_int, FEATURE_COLS_PRIMARY + ["rain_decay_index_api_chirps_x_rain_peak_residual_orthogonalized"]),
        "+all_3_interactions": (tr_int, te_int, FEATURE_COLS_PRIMARY +
                                 [f"{f1}_x_{f2}" for f1, f2 in INTERACTION_PAIRS]),
        "+hand_threshold": (tr_thr, te_thr, FEATURE_COLS_PRIMARY + ["hand_low_lt4"]),
        "+rain_decay_threshold": (tr_thr, te_thr, FEATURE_COLS_PRIMARY + ["rain_decay_high_gt20"]),
        "+rain_peak_threshold": (tr_thr, te_thr, FEATURE_COLS_PRIMARY + ["rain_peak_low_ltm4"]),
        "+all_3_thresholds": (tr_thr, te_thr, FEATURE_COLS_PRIMARY +
                               [v[0] for v in THRESHOLDS.values()]),
    }
    rows = []
    for name, (tr, te, feats) in configs.items():
        auc = holdout_auc_with_features(tr, te, feats)
        rows.append({"config": name, "n_variaveis": len(feats), "auc_holdout_2026": auc})
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

    df = run_all_configs(df_train, df_2026)
    print(df.to_string(index=False))
    df.to_csv(out / "v20x_tentativas_traducao_interacao.csv", index=False)

    resumo = {
        "reference_baseline_linear_5feat": 0.5246,
        "reference_gbm_v20u": 0.5888,
        "melhor_config_interpretavel": df.loc[df["auc_holdout_2026"].idxmax(), "config"],
        "melhor_auc_interpretavel": float(df["auc_holdout_2026"].max()),
        "recuperou_o_sinal_do_gbm": bool(df["auc_holdout_2026"].max() >= 0.57),
    }
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    (out / "v20x_relatorios_consolidados.json").write_text(
        json.dumps({"configs": df.to_dict(orient="records"), "resumo": resumo},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
