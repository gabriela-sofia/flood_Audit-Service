"""SUSC-20O -- validacao temporal holdout de Curitiba, rota primaria (5 features, SUSC-20N).

LOO-CV e k-fold repetido (SUSC-20M/20N) embaralham unidades no tempo: um ponto de 2023 pode
cair no fold de teste enquanto um de 2026 treina o modelo, o que superestima capacidade
preditiva se houver qualquer deriva temporal (mudanca de composicao do SIAC 156, expansao de
bairro coberto, etc.). Este script testa a alternativa honesta: treina em anos passados, avalia
em ano nunca visto.

Split: TREINO = 2023-2025 (989 unidades: 318 neg / 671 pos), TESTE = 2026-parcial (282
unidades: 108 neg / 174 pos, jan-jul). Sem embaralhamento, sem re-sorteio -- e o corte
cronologico unico que os dados permitem.

Mesma rota primaria do SUSC-20N (5 features, sem elevation_m -- decisao causal, nao
estatistica, documentada em pipeline_v20m_curitiba_primary.py). EPV checado no TREINO antes de
ajustar (classe minoritaria = negativo no treino). Scaler ajustado SO no treino (sem vazamento
pro teste). AUC preditivo = 1 unico corte temporal, nao k-fold -- reportado com cautela (n
menor, sem repeticao possivel por so haver 1 ano de holdout).

Uso:
    python pipeline_v20o_temporal_holdout_curitiba.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[4]
DATASET = (ROOT / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/"
                  "registries/v20n_dataset_curitiba_features_v2.csv")
RESULTS = ROOT / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results"

SEED = 20260801
EPV_FLOOR = 20.0
TEST_YEARS = {2026}

FEATURE_COLS_PRIMARY = ["slope_deg", "hand_m_dinf", "twi_dinf",
                        "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]
EXPECTED_SIGN = {"slope_deg": -1, "hand_m_dinf": -1, "twi_dinf": +1,
                 "rain_peak_residual_orthogonalized": +1, "rain_decay_index_api_chirps": +1}


def _firth_cls():
    from firthlogist import FirthLogisticRegression
    from sklearn.utils.validation import validate_data

    if not hasattr(FirthLogisticRegression, "_validate_data"):
        FirthLogisticRegression._validate_data = (
            lambda self, X, y=None, **kw: validate_data(self, X, y, **kw))
    return FirthLogisticRegression


def gate_epv(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    d = df.dropna(subset=feature_cols)
    n_min = int((d["label"] == 0).sum())
    epv = n_min / len(feature_cols)
    return {"classe_minoritaria": "negativo", "n_minoritaria_apos_listwise": n_min,
            "n_features": len(feature_cols), "epv": round(epv, 2), "piso": EPV_FLOOR,
            "passa": bool(epv >= EPV_FLOOR)}


def firth_on_train(df_train: pd.DataFrame, feature_cols: list[str]):
    FirthLogisticRegression = _firth_cls()
    d = df_train.dropna(subset=feature_cols).copy()
    X = d[feature_cols].values
    y = d["label"].astype(int).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = FirthLogisticRegression(fit_intercept=True)
    model.fit(Xs, y)
    rows = []
    for i, feat in enumerate(feature_cols):
        rows.append({"feature": feat, "coef_standardized": round(float(model.coef_[i]), 4),
                     "ci_low_95": round(float(model.ci_[i][0]), 4),
                     "ci_high_95": round(float(model.ci_[i][1]), 4),
                     "p_value": round(float(model.pvals_[i]), 4),
                     "expected_sign": EXPECTED_SIGN.get(feat),
                     "sign_matches_expected": bool(np.sign(model.coef_[i]) == EXPECTED_SIGN.get(feat, 0)),
                     "ci_crosses_zero": bool(model.ci_[i][0] <= 0 <= model.ci_[i][1])})
    report = {"n_train": int(len(d)), "n_pos_train": int((y == 1).sum()), "n_neg_train": int((y == 0).sum())}
    return pd.DataFrame(rows), report, scaler


def temporal_holdout_auc(df_train: pd.DataFrame, df_test: pd.DataFrame, feature_cols: list[str]):
    dtr = df_train.dropna(subset=feature_cols)
    dte = df_test.dropna(subset=feature_cols)
    Xtr, ytr = dtr[feature_cols].values, dtr["label"].astype(int).values
    Xte, yte = dte[feature_cols].values, dte["label"].astype(int).values
    scaler = StandardScaler().fit(Xtr)
    Xtrs, Xtes = scaler.transform(Xtr), scaler.transform(Xte)
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(Xtrs, ytr)
    score = clf.predict_proba(Xtes)[:, 1]
    auc = roc_auc_score(yte, score)
    return {"n_train": int(len(dtr)), "n_test": int(len(dte)),
            "n_pos_test": int((yte == 1).sum()), "n_neg_test": int((yte == 0).sum()),
            "holdout_auc": round(float(auc), 4)}


def univariate_screen_train(df_train: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feat in feature_cols:
        d = df_train.dropna(subset=[feat])
        pos = d.loc[d["label"] == 1, feat].values
        neg = d.loc[d["label"] == 0, feat].values
        u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        rows.append({"feature": feat, "n_pos": len(pos), "n_neg": len(neg), "p_value": round(float(p), 4),
                     "significant_p05": bool(p < 0.05)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(DATASET))
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--tag", default="v20o")
    args = p.parse_args(argv)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    linhas = pd.read_csv(args.dataset)
    linhas["event_date"] = pd.to_datetime(linhas["event_date"])
    unidades = linhas[~linhas["is_duplicate_observation_unit"]].copy()
    unidades["year"] = unidades["event_date"].dt.year

    df_test = unidades[unidades["year"].isin(TEST_YEARS)].reset_index(drop=True)
    df_train = unidades[~unidades["year"].isin(TEST_YEARS)].reset_index(drop=True)
    print(f"treino: {len(df_train)} unidades ({sorted(df_train['year'].unique())}) | "
          f"teste: {len(df_test)} unidades ({sorted(df_test['year'].unique())})")

    epv = gate_epv(df_train, FEATURE_COLS_PRIMARY)
    print(f"gate EPV (treino): {epv['epv']} (piso {EPV_FLOOR}) -> {'PASSA' if epv['passa'] else 'REPROVA'}")
    if not epv["passa"]:
        raise SystemExit("EPV reprovado no treino -- holdout nao roda")

    print("[1] univariado (treino apenas)")
    univ = univariate_screen_train(df_train, FEATURE_COLS_PRIMARY)
    univ.to_csv(out / f"{args.tag}_univariate_train_only.csv", index=False)
    print(univ.to_string(index=False))

    print("[2] Firth multivariado (ajustado so no treino)")
    coefs, firth_report, _ = firth_on_train(df_train, FEATURE_COLS_PRIMARY)
    coefs.to_csv(out / f"{args.tag}_firth_train_only_coefs.csv", index=False)
    print(json.dumps(firth_report, indent=2))
    print(coefs.to_string(index=False))

    print("[3] AUC preditivo -- holdout temporal (treino ajusta, teste nunca visto)")
    holdout = temporal_holdout_auc(df_train, df_test, FEATURE_COLS_PRIMARY)
    print(json.dumps(holdout, indent=2))

    reports = {"split": {"treino_anos": sorted(int(y) for y in df_train["year"].unique()),
                          "teste_anos": sorted(int(y) for y in df_test["year"].unique())},
               "gate_epv_treino": epv, "firth_report_treino": firth_report, "holdout_auc": holdout}
    (out / f"{args.tag}_all_reports.json").write_text(
        json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
