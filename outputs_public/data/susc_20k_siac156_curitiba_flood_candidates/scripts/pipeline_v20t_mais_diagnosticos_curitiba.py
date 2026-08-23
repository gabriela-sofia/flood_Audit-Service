"""SUSC-20T -- mais 3 vertentes testadas (achado real de noticia + 2 tecnicas de literatura).

Continuacao do pedido explicito de esgotar toda alternativa. Tres testes, todos sobre dado ja
existente (nenhuma aquisicao nova):

1. **Divisao pelo lancamento do CuritibaApp (2026-03-25)** -- achado de noticia real: em
   25/03/2026 a Prefeitura lancou um app unificado com IA que absorve o 156 gradualmente
   (bemparana.com.br, curitiba.pr.gov.br). Como isso cai DENTRO da janela de teste jan-jul/2026,
   testamos se a quebra de AUC se concentra no periodo pos-lancamento.
2. **`rain_max_24h_chirps` como variavel alternativa/adicional** -- coluna ja presente no
   dataset (nunca usada como variavel de modelo), testada como candidata mais simples que os
   indices de pico/decaimento atuais.
3. **Treino com peso de recencia (decaimento exponencial)** -- literatura de concept drift
   (weighted-negative-log-likelihood com meia-vida) recomenda ponderar exemplos recentes mais
   que antigos em vez de pool flat -- testado como alternativa ao pooling atual (2023-2025
   com peso igual).

Uso:
    python pipeline_v20t_mais_diagnosticos_curitiba.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_v20o_holdout_temporal_curitiba import (  # noqa: E402
    DATASET, FEATURE_COLS_PRIMARY, temporal_holdout_auc, gate_epv)

RESULTS = Path(__file__).resolve().parents[4] / (
    "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results")

APP_LAUNCH_DATE = pd.Timestamp("2026-03-25")


# ---------------------------------------------------------------- diagnostico 1
def app_launch_split(df_train: pd.DataFrame, df_2026: pd.DataFrame,
                      feature_cols: list[str]) -> dict:
    pre = df_2026[df_2026["data_evento"] < APP_LAUNCH_DATE].reset_index(drop=True)
    post = df_2026[df_2026["data_evento"] >= APP_LAUNCH_DATE].reset_index(drop=True)
    r_pre = temporal_holdout_auc(df_train, pre, feature_cols)
    r_post = temporal_holdout_auc(df_train, post, feature_cols)
    frac_pos_pre = round(float((pre["rotulo"] == 1).mean()), 4) if len(pre) else None
    frac_pos_post = round(float((post["rotulo"] == 1).mean()), 4) if len(post) else None
    return {"app_launch_date": str(APP_LAUNCH_DATE.date()), "pre_app": r_pre,
            "post_app": r_post, "frac_positivo_pre_app": frac_pos_pre,
            "frac_positivo_post_app": frac_pos_post,
            "colapso_concentrado_pos_app": bool(
                r_pre["auc_holdout"] >= 0.55 and r_post["auc_holdout"] < 0.55)}


# ---------------------------------------------------------------- diagnostico 2
def rain_max_24h_correlation_by_year(unidades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ano in sorted(unidades["ano"].unique()):
        d = unidades[unidades["ano"] == ano].dropna(subset=["rain_max_24h_chirps"])
        corr = d["rain_max_24h_chirps"].corr(d["rotulo"].astype(float))
        rows.append({"ano": int(ano), "n": int(len(d)),
                     "corr_rain_max_24h_vs_label": round(float(corr), 4)})
    return pd.DataFrame(rows)


def rain_max_24h_temporal_holdout(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Substitui os 2 features de chuva atuais por rain_max_24h_chirps (variavel bruta ja no
    dataset, nunca usada como variavel de modelo) -- terreno + essa 1 variavel de chuva."""
    feats = ["slope_deg", "hand_m_dinf", "twi_dinf", "rain_max_24h_chirps"]
    return {"feature_cols": feats, **temporal_holdout_auc(df_train, df_test, feats)}


# ---------------------------------------------------------------- diagnostico 3
def recency_weighted_temporal_holdout(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                       feature_cols: list[str], half_life_years: float) -> dict:
    dtr = df_train.dropna(subset=feature_cols).copy()
    dte = df_test.dropna(subset=feature_cols)
    ref_year = dtr["ano"].max()  # ano mais recente do treino (2025)
    age_years = (ref_year - dtr["ano"]).astype(float)
    weights = 0.5 ** (age_years / half_life_years)
    Xtr, ytr = dtr[feature_cols].values, dtr["rotulo"].astype(int).values
    Xte, yte = dte[feature_cols].values, dte["rotulo"].astype(int).values
    scaler = StandardScaler().fit(Xtr)
    Xtrs, Xtes = scaler.transform(Xtr), scaler.transform(Xte)
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(Xtrs, ytr, sample_weight=weights.values)
    escore = clf.predict_proba(Xtes)[:, 1]
    auc = roc_auc_score(yte, escore)
    return {"half_life_years": half_life_years, "n_treino": int(len(dtr)), "n_teste": int(len(dte)),
            "weight_2023": round(float(0.5 ** ((ref_year - 2023) / half_life_years)), 4),
            "weight_2024": round(float(0.5 ** ((ref_year - 2024) / half_life_years)), 4),
            "weight_2025": round(float(0.5 ** ((ref_year - 2025) / half_life_years)), 4),
            "holdout_auc_recency_weighted": round(float(auc), 4)}


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

    print("[1] divisao pelo lancamento do CuritibaApp (2026-03-25)")
    d1 = app_launch_split(df_train, df_2026, FEATURE_COLS_PRIMARY)
    print(json.dumps(d1, indent=2, ensure_ascii=False))
    reports["1_app_launch_split"] = d1

    print("\n[2a] correlacao rain_max_24h_chirps x rotulo por ano")
    d2a = rain_max_24h_correlation_by_year(unidades)
    print(d2a.to_string(index=False))
    print("\n[2b] holdout temporal com rain_max_24h_chirps no lugar dos indices atuais")
    d2b = rain_max_24h_temporal_holdout(df_train, df_2026)
    print(json.dumps(d2b, indent=2, ensure_ascii=False))
    reports["2a_rain_max_24h_corr_by_year"] = d2a.to_dict(orient="records")
    reports["2b_rain_max_24h_temporal_holdout"] = d2b

    print("\n[3] treino com peso de recencia (decaimento exponencial), meia-vida 1 e 2 anos")
    d3 = [recency_weighted_temporal_holdout(df_train, df_2026, FEATURE_COLS_PRIMARY, hl)
          for hl in [1.0, 2.0]]
    for r in d3:
        print(json.dumps(r, indent=2))
    reports["3_recency_weighted_temporal_holdout"] = d3
    reports["reference_baseline_temporal_holdout_auc_v20o"] = 0.5246

    (out / "v20t_relatorios_consolidados.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
