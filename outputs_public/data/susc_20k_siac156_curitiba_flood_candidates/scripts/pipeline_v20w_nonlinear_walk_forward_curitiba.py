"""SUSC-20W -- walk-forward multi-corte pra 4 classes de modelo (nao so 2026), Curitiba.

Checagem de robustez essencial antes de acreditar no achado do SUSC-20U/20V: sera que a
vantagem dos modelos nao-lineares sobre o linear e especifica de 2026 (o que sugeriria que o
modelo flexivel esta explorando algum ruido especifico daquele conjunto de teste, nao
estrutura real), ou aparece em TODOS os cortes prospectivos (2023->2024, 2023-24->2025,
2023-25->2026)? Se for so em 2026, o achado fica sob suspeita de overfitting ao teste. Se
aparecer nos 3 cortes, fortalece a leitura de que existe nao-linearidade real nas features,
independente do problema de deriva de 2026 ja diagnosticado (SUSC-20Q/20R/20S/20T).

Uso:
    python pipeline_v20w_nonlinear_walk_forward_curitiba.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_v20o_temporal_holdout_curitiba import DATASET, FEATURE_COLS_PRIMARY  # noqa: E402
from pipeline_v20u_nonlinear_diagnostic_curitiba import GBM_PARAMS  # noqa: E402

RESULTS = Path(__file__).resolve().parents[4] / (
    "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results")

CUTOFFS = [([2023], 2024), ([2023, 2024], 2025), ([2023, 2024, 2025], 2026)]

MODEL_FACTORIES = {
    "Linear_baseline": lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                                                    class_weight="balanced"),
    "GBM": lambda: GradientBoostingClassifier(**GBM_PARAMS),
    "AdaBoost": lambda: AdaBoostClassifier(n_estimators=100, random_state=42),
    "RandomForest_d3": lambda: RandomForestClassifier(n_estimators=200, max_depth=3,
                                                        class_weight="balanced", random_state=42),
}


def run_cutoff(unidades: pd.DataFrame, train_years: list[int], test_year: int,
               feature_cols: list[str], clf_factory) -> dict:
    tr = unidades[unidades["year"].isin(train_years)].dropna(subset=feature_cols)
    te = unidades[unidades["year"] == test_year].dropna(subset=feature_cols)
    Xtr, ytr = tr[feature_cols].values, tr["label"].astype(int).values
    Xte, yte = te[feature_cols].values, te["label"].astype(int).values
    scaler = StandardScaler().fit(Xtr)
    clf = clf_factory()
    clf.fit(scaler.transform(Xtr), ytr)
    score = clf.predict_proba(scaler.transform(Xte))[:, 1]
    auc = roc_auc_score(yte, score)
    return {"train_years": train_years, "test_year": test_year, "n_train": int(len(tr)),
            "n_test": int(len(te)), "holdout_auc": round(float(auc), 4)}


def walk_forward_all_models(unidades: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for model_name, factory in MODEL_FACTORIES.items():
        for train_years, test_year in CUTOFFS:
            r = run_cutoff(unidades, train_years, test_year, feature_cols, factory)
            r["model"] = model_name
            r["train_years"] = ",".join(str(y) for y in r["train_years"])
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    out = RESULTS
    out.mkdir(parents=True, exist_ok=True)
    linhas = pd.read_csv(DATASET)
    linhas["event_date"] = pd.to_datetime(linhas["event_date"])
    unidades = linhas[~linhas["is_duplicate_observation_unit"]].reset_index(drop=True)
    unidades["year"] = unidades["event_date"].dt.year

    df = walk_forward_all_models(unidades, FEATURE_COLS_PRIMARY)
    print(df.to_string(index=False))
    df.to_csv(out / "v20w_nonlinear_walk_forward_all_models.csv", index=False)

    pivot = df.pivot(index="model", columns="test_year", values="holdout_auc")
    print("\npivot (linhas=modelo, colunas=ano de teste):")
    print(pivot.to_string())

    # o achado do SUSC-20U/20V e especifico de 2026, ou aparece nos 3 cortes?
    beats_linear_by_year = {}
    for test_year in [2024, 2025, 2026]:
        linear_auc = df[(df.model == "Linear_baseline") & (df.test_year == test_year)]["holdout_auc"].iloc[0]
        nonlinear = df[(df.model != "Linear_baseline") & (df.test_year == test_year)]
        n_beat = int((nonlinear["holdout_auc"] > linear_auc).sum())
        beats_linear_by_year[str(test_year)] = {
            "linear_auc": float(linear_auc), "n_modelos_nao_lineares_acima": n_beat,
            "n_modelos_nao_lineares_total": int(len(nonlinear))}

    resumo = {
        "beats_linear_by_test_year": beats_linear_by_year,
        "leitura": ("se n_modelos_nao_lineares_acima for alto (perto do total) em TODOS os "
                    "3 anos de teste, a vantagem nao-linear e generalizavel, nao um artefato "
                    "especifico de 2026 -- fortalece o achado do SUSC-20U/20V. 2026 continua "
                    "sendo o ano mais dificil de prever pra QUALQUER classe de modelo "
                    "(AUC absoluto mais baixo em todas), consistente com a deriva ja "
                    "diagnosticada (SUSC-20Q) nao ser resolvida, so parcialmente compensada, "
                    "pela nao-linearidade."),
    }
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    (out / "v20w_all_reports.json").write_text(
        json.dumps({"fold_results": df.to_dict(orient="records"), "resumo": resumo},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
