"""SUSC-20Q -- bateria exaustiva de diagnosticos do colapso de AUC prospectivo em Curitiba.

O SUSC-20O achou AUC caindo de 0,6459 (LOO-CV embaralhado) pra 0,5246 (holdout temporal 2026).
O SUSC-20P descartou vazamento espacial (spatial block CV = 0,6442, igual ao embaralhado).
Por pedido explicito (2026-08-02): testar exaustivamente toda vertente que a literatura oferece
pra diagnosticar a causa remanescente -- uma tarefa de cada vez dentro deste script, cada
diagnostico independente e reportado separadamente, nenhum vira decisao de metodologia sem
revisao humana.

Cinco diagnosticos, todos sobre dado ja existente (nenhuma aquisicao nova):

1. Bootstrap CI no AUC do holdout temporal -- a queda de 0,65->0,52 e real ou parcialmente
   ruido de amostra pequena (n_teste~280)?
2. Ablacao terreno-so vs. chuva-so sob holdout temporal -- qual grupo de feature carrega o
   colapso? (terreno = causal/estatico; chuva = pode carregar deriva administrativa)
3. Walk-forward multi-corte -- o colapso e especifico de 2026 ou aparece em qualquer corte
   prospectivo? (treino 2023->teste 2024; treino 2023-24->teste 2025; treino 2023-25->teste
   2026 = SUSC-20O)
4. Holdout casado por estacao -- treino Jan-Jul de 2023-2025, teste Jan-Jul 2026 (controla
   sazonalidade isolando de deriva administrativa entre anos)
5. Estabilidade de coeficiente por ano -- Firth ajustado em cada ano isoladamente, sinal e
   significancia comparados
6. Estabilidade de composicao/metadado por ano -- confidence_tier, source_registry, categoria
   'assunto', negative_source_type, duplicate_group_count e rain_status comparados ano a ano,
   pra checar se 2026 tem um processo de geracao de dado (nao fisico) visivelmente diferente
   dos anos que generalizaram bem (2024, 2025 no diagnostico 3)

Reaproveita gate_epv/firth_on_train/temporal_holdout_auc de pipeline_v20o (nao duplica logica).

Uso:
    python pipeline_v20q_exhaustive_temporal_diagnostics_curitiba.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_v20o_temporal_holdout_curitiba import (  # noqa: E402
    DATASET,
    EPV_FLOOR,
    FEATURE_COLS_PRIMARY,
    firth_on_train,
    gate_epv,
    temporal_holdout_auc,
)

RESULTS = Path(__file__).resolve().parents[4] / (
    "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results")

TERRAIN_FEATS = ["slope_deg", "hand_m_dinf", "twi_dinf"]
RAIN_FEATS = ["rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]
N_BOOT = 2000
BOOT_SEED = 20260802


# ---------------------------------------------------------------- diagnostico 1
def bootstrap_holdout_auc_ci(df_train: pd.DataFrame, df_test: pd.DataFrame,
                              feature_cols: list[str], n_boot: int = N_BOOT,
                              seed: int = BOOT_SEED) -> dict:
    """Ajusta 1x no treino, bootstrap-resample do teste (com reposicao) pra CI do AUC."""
    dtr = df_train.dropna(subset=feature_cols)
    dte = df_test.dropna(subset=feature_cols).reset_index(drop=True)
    Xtr, ytr = dtr[feature_cols].values, dtr["label"].astype(int).values
    Xte, yte = dte[feature_cols].values, dte["label"].astype(int).values
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)
    score_all = clf.predict_proba(scaler.transform(Xte))[:, 1]
    point_auc = roc_auc_score(yte, score_all)

    rng = np.random.default_rng(seed)
    n = len(dte)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = yte[idx], score_all[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yb, sb))
    boot_aucs = np.array(boot_aucs)
    ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
    return {
        "n_test": int(n), "point_auc": round(float(point_auc), 4),
        "n_boot_valid": int(len(boot_aucs)),
        "ci_95_low": round(float(ci_low), 4), "ci_95_high": round(float(ci_high), 4),
        "ci_excludes_0_5_chance": bool(ci_low > 0.5),
        "ci_excludes_reference_shuffled_0_6459": bool(ci_high < 0.6459),
        "reference_loo_auc_shuffled_v20n": 0.6459,
    }


# ---------------------------------------------------------------- diagnostico 2
def ablation_terrain_vs_rain(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for nome, feats in [("terreno_so", TERRAIN_FEATS), ("chuva_so", RAIN_FEATS),
                         ("completo_5feat", FEATURE_COLS_PRIMARY)]:
        r = temporal_holdout_auc(df_train, df_test, feats)
        r["grupo_features"] = nome
        r["n_features"] = len(feats)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- diagnostico 3
def walk_forward_cutoffs(unidades: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    anos = sorted(unidades["year"].unique())
    rows = []
    for i in range(1, len(anos)):
        train_years = anos[:i]
        test_year = anos[i]
        df_train = unidades[unidades["year"].isin(train_years)]
        df_test = unidades[unidades["year"] == test_year]
        epv = gate_epv(df_train, feature_cols)
        result = temporal_holdout_auc(df_train, df_test, feature_cols)
        rows.append({"train_years": ",".join(str(y) for y in train_years),
                      "test_year": int(test_year), "epv_treino": epv["epv"],
                      "epv_passa": epv["passa"], **result})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- diagnostico 4
def seasonal_matched_holdout(unidades: pd.DataFrame, feature_cols: list[str],
                              test_year: int = 2026, train_years=(2023, 2024, 2025),
                              months=(1, 7)) -> dict:
    """Treino = mesma janela Jan-Jul dos anos anteriores; teste = Jan-Jul do ano-alvo.
    Controla sazonalidade -- se o AUC melhorar bastante vs. o holdout temporal padrao (treino
    ano-cheio, teste parcial), sazonalidade nao-comparavel era parte real da causa."""
    m_lo, m_hi = months
    janela = unidades[unidades["month"].between(m_lo, m_hi)]
    df_train = janela[janela["year"].isin(train_years)]
    df_test = janela[janela["year"] == test_year]
    epv = gate_epv(df_train, feature_cols)
    result = temporal_holdout_auc(df_train, df_test, feature_cols)
    return {"desenho": f"treino jan-jul {train_years}, teste jan-jul {test_year}",
            "epv_treino": epv["epv"], "epv_passa": epv["passa"], **result}


# ---------------------------------------------------------------- diagnostico 5
def coef_stability_by_year(unidades: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for ano in sorted(unidades["year"].unique()):
        d = unidades[unidades["year"] == ano]
        epv = gate_epv(d, feature_cols)
        if not epv["passa"]:
            for feat in feature_cols:
                rows.append({"year": int(ano), "feature": feat, "epv": epv["epv"],
                             "epv_passa": False, "coef_standardized": None, "p_value": None,
                             "sign": None})
            continue
        coefs, report, _ = firth_on_train(d, feature_cols)
        for _, r in coefs.iterrows():
            rows.append({"year": int(ano), "feature": r["feature"], "epv": epv["epv"],
                         "epv_passa": True, "coef_standardized": r["coef_standardized"],
                         "p_value": r["p_value"],
                         "sign": "+" if r["coef_standardized"] > 0 else "-"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- diagnostico 6
def metadata_composition_stability_by_year(unidades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ano in sorted(unidades["year"].unique()):
        d = unidades[unidades["year"] == ano]
        pos = d[d["label"] == 1]
        neg = d[d["label"] == 0]
        rows.append({
            "year": int(ano),
            "n_total": int(len(d)),
            "frac_confidence_strong_pos": (round(float((pos["confidence_tier"] == "strong").mean()), 4)
                                            if len(pos) else None),
            "frac_assunto_drenagem_pos": (round(float((pos["assunto"] == "Drenagem").mean()), 4)
                                           if len(pos) else None),
            "n_source_registry_unique": int(d["source_registry"].nunique()),
            "n_negative_source_type_unique": int(neg["negative_source_type"].nunique()) if len(neg) else 0,
            "duplicate_group_count_mean": round(float(d["duplicate_group_count"].mean()), 4),
            "rain_status_all_ok": bool((d["rain_status"] == "OK").all()),
            "rain_n_days_found_mean": round(float(d["rain_n_days_found"].mean()), 2),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(DATASET))
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--tag", default="v20q")
    args = p.parse_args(argv)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    linhas = pd.read_csv(args.dataset)
    linhas["event_date"] = pd.to_datetime(linhas["event_date"])
    unidades = linhas[~linhas["is_duplicate_observation_unit"]].reset_index(drop=True)
    unidades["year"] = unidades["event_date"].dt.year
    unidades["month"] = unidades["event_date"].dt.month

    df_train_2325 = unidades[unidades["year"].isin([2023, 2024, 2025])].reset_index(drop=True)
    df_test_2026 = unidades[unidades["year"] == 2026].reset_index(drop=True)

    reports = {}

    print("[1] bootstrap CI -- holdout temporal 2026")
    d1 = bootstrap_holdout_auc_ci(df_train_2325, df_test_2026, FEATURE_COLS_PRIMARY)
    print(json.dumps(d1, indent=2))
    reports["1_bootstrap_ci"] = d1

    print("\n[2] ablacao terreno-so vs chuva-so, holdout temporal 2026")
    d2 = ablation_terrain_vs_rain(df_train_2325, df_test_2026)
    print(d2.to_string(index=False))
    d2.to_csv(out / f"{args.tag}_ablacao_terreno_vs_chuva.csv", index=False)
    reports["2_ablacao_terreno_vs_chuva"] = d2.to_dict(orient="records")

    print("\n[3] walk-forward multi-corte")
    d3 = walk_forward_cutoffs(unidades, FEATURE_COLS_PRIMARY)
    print(d3.to_string(index=False))
    d3.to_csv(out / f"{args.tag}_walk_forward_cutoffs.csv", index=False)
    reports["3_walk_forward"] = d3.to_dict(orient="records")

    print("\n[4] holdout casado por estacao (jan-jul 2023-25 -> jan-jul 2026)")
    d4 = seasonal_matched_holdout(unidades, FEATURE_COLS_PRIMARY)
    print(json.dumps(d4, indent=2))
    reports["4_seasonal_matched_holdout"] = d4

    print("\n[5] estabilidade de coeficiente Firth por ano")
    d5 = coef_stability_by_year(unidades, FEATURE_COLS_PRIMARY)
    print(d5.to_string(index=False))
    d5.to_csv(out / f"{args.tag}_coef_stability_by_year.csv", index=False)
    reports["5_coef_stability_by_year"] = d5.to_dict(orient="records")

    print("\n[6] estabilidade de composicao/metadado por ano")
    d6 = metadata_composition_stability_by_year(unidades)
    print(d6.to_string(index=False))
    d6.to_csv(out / f"{args.tag}_metadata_composition_stability_by_year.csv", index=False)
    reports["6_metadata_composition_stability"] = d6.to_dict(orient="records")

    (out / f"{args.tag}_all_reports.json").write_text(
        json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
