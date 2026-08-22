"""SUSC-20S -- piloto de redesenho de amostragem negativa via PU bagging, Curitiba.

Segunda vertente da revisao de literatura de 2026-08-02 (a primeira, ONI/ENOS, foi testada no
SUSC-20R e REFUTADA pelo dado real). Ataca o confundimento estrutural ja documentado desde o
SUSC-20K/20N: "negativo" em Curitiba e uma queixa nao-hidrologica numa data arbitraria, nao uma
confirmacao de ausencia de enchente naquele local/data -- literatura chama isso de
"case-control sampling with contaminated controls" (PBLC, Land 2022, DOI 10.3390/land11111971;
mesma linguagem usada nas nossas proprias limitacoes documentadas).

Metodo escolhido: **PU bagging** (Mordelet & Vert, "A bagging SVM to learn from positive and
unlabeled examples", arXiv:1010.0772) -- NAO o modelo bayesiano espacial de sub-reporte
(Agostini/Pierson/Garg, AAAI-24). Justificativa da escolha, registrada explicitamente porque e
uma decisao de metodologia, nao um detalhe: o modelo bayesiano espacial suaviza risco por
correlacao espacial entre queixas, o que criaria uma superficie continua derivada da propria
densidade de queixa -- proximo demais de um "score/proxy derivado do label" (proibido pelas
regras fixas do projeto: "nunca usar scores, thresholds, proxies ou variaveis derivadas do
label como features", "o modelo NAO deve descobrir enchentes"). PU bagging nao inventa nenhuma
variavel nova nem suaviza no espaco -- so muda como o classificador trata o rotulo "negativo"
(de "confirmado ausente" pra "nao confirmado", tratado como nao-confiavel em vez de verdade),
usando as MESMAS 5 features fisico-causais de sempre. Mais alinhado as regras fixas do projeto.

Receita (Mordelet & Vert): positivos ficam fixos em toda bag (rotulo confiavel); a cada bag,
reamostra com reposicao do pool "nao-rotulado" (nosso atual "negativo") um subconjunto do
mesmo tamanho dos positivos; ajusta um classificador; agrega a predicao media dos pontos
nao-rotulados que ficaram de fora daquela bag (out-of-bag) ao longo de todas as bags.

Escopo desta rodada -- **piloto, nao substitui a rota primaria**: só Curitiba, reaproveita as
MESMAS baterias de validacao ja commitadas (holdout temporal SUSC-20O, spatial block CV
SUSC-20P) trocando so o classificador, pra comparar lado a lado se PU bagging melhora
generalizacao prospectiva especificamente (o problema real diagnosticado). Recife nao e tocado
nesta rodada -- mudanca de metodologia usada em produção em outra regiao exige decisao humana
separada.

Uso:
    python pipeline_v20s_pu_bagging_pilot_curitiba.py
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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_v20o_temporal_holdout_curitiba import DATASET, FEATURE_COLS_PRIMARY  # noqa: E402

RESULTS = Path(__file__).resolve().parents[4] / (
    "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results")

N_BAGS = 300
BAG_SEED = 20260802


def pu_bagging_fit_predict(Xtr_std: np.ndarray, ytr: np.ndarray, Xte_std: np.ndarray,
                            n_bags: int = N_BAGS, seed: int = BAG_SEED) -> np.ndarray:
    """Ajusta n_bags classificadores (positivos fixos + reamostra de nao-rotulados do mesmo
    tamanho dos positivos), retorna a predicao MEDIA no conjunto de teste (nunca usado pra
    ajustar nada -- so recebe a media das n_bags predicoes)."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(ytr == 1)[0]
    unl_idx = np.where(ytr == 0)[0]
    n_pos = len(pos_idx)
    scores = np.zeros((n_bags, Xte_std.shape[0]))
    for b in range(n_bags):
        bag_unl_idx = rng.choice(unl_idx, size=n_pos, replace=True)
        bag_idx = np.concatenate([pos_idx, bag_unl_idx])
        Xb, yb = Xtr_std[bag_idx], ytr[bag_idx]
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
        clf.fit(Xb, yb)
        scores[b] = clf.predict_proba(Xte_std)[:, 1]
    return scores.mean(axis=0)


def pu_bagging_temporal_holdout(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                 feature_cols: list[str], n_bags: int = N_BAGS,
                                 seed: int = BAG_SEED) -> dict:
    dtr = df_train.dropna(subset=feature_cols)
    dte = df_test.dropna(subset=feature_cols)
    Xtr, ytr = dtr[feature_cols].values, dtr["label"].astype(int).values
    Xte, yte = dte[feature_cols].values, dte["label"].astype(int).values
    scaler = StandardScaler().fit(Xtr)
    Xtrs, Xtes = scaler.transform(Xtr), scaler.transform(Xte)
    score = pu_bagging_fit_predict(Xtrs, ytr, Xtes, n_bags=n_bags, seed=seed)
    auc = roc_auc_score(yte, score)
    return {"n_train": int(len(dtr)), "n_test": int(len(dte)),
            "n_pos_test": int((yte == 1).sum()), "n_neg_test": int((yte == 0).sum()),
            "n_bags": n_bags, "holdout_auc_pu_bagging": round(float(auc), 4)}


def pu_bagging_spatial_block_cv(unidades: pd.DataFrame, feature_cols: list[str],
                                 n_splits: int = 5, n_bags: int = N_BAGS,
                                 seed: int = BAG_SEED) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=n_splits)
    groups = unidades["bairro"].values
    rows = []
    for i, (tr_idx, te_idx) in enumerate(gkf.split(unidades, groups=groups)):
        df_train = unidades.iloc[tr_idx].reset_index(drop=True)
        df_test = unidades.iloc[te_idx].reset_index(drop=True)
        bairros_train = set(df_train["bairro"].unique())
        bairros_test = set(df_test["bairro"].unique())
        assert not (bairros_train & bairros_test), f"fold {i} vaza bairro entre treino/teste"
        r = pu_bagging_temporal_holdout(df_train, df_test, feature_cols, n_bags=n_bags,
                                         seed=seed + i)
        r["fold"] = i
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(DATASET))
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--tag", default="v20s")
    p.add_argument("--n-bags", type=int, default=N_BAGS)
    args = p.parse_args(argv)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    linhas = pd.read_csv(args.dataset)
    linhas["event_date"] = pd.to_datetime(linhas["event_date"])
    unidades = linhas[~linhas["is_duplicate_observation_unit"]].reset_index(drop=True)
    unidades["year"] = unidades["event_date"].dt.year

    df_train_2325 = unidades[unidades["year"].isin([2023, 2024, 2025])].reset_index(drop=True)
    df_test_2026 = unidades[unidades["year"] == 2026].reset_index(drop=True)

    reports = {}

    print(f"[1] PU bagging -- holdout temporal 2026 (n_bags={args.n_bags})")
    d1 = pu_bagging_temporal_holdout(df_train_2325, df_test_2026, FEATURE_COLS_PRIMARY,
                                      n_bags=args.n_bags)
    d1["reference_baseline_temporal_holdout_auc_v20o"] = 0.5246
    print(json.dumps(d1, indent=2))
    reports["1_pu_bagging_temporal_holdout"] = d1

    print(f"\n[2] PU bagging -- spatial block CV (n_bags={args.n_bags})")
    d2 = pu_bagging_spatial_block_cv(unidades, FEATURE_COLS_PRIMARY, n_bags=args.n_bags)
    print(d2.to_string(index=False))
    d2.to_csv(out / f"{args.tag}_pu_bagging_spatial_block_cv.csv", index=False)
    mean_auc = round(float(d2["holdout_auc_pu_bagging"].mean()), 4)
    print(f"AUC medio spatial block CV (PU bagging) = {mean_auc} "
          f"(referencia baseline SUSC-20P = 0.6442)")
    reports["2_pu_bagging_spatial_block_cv"] = {
        "fold_results": d2.to_dict(orient="records"), "auc_mean": mean_auc,
        "reference_baseline_spatial_block_cv_auc_v20p": 0.6442}

    (out / f"{args.tag}_all_reports.json").write_text(
        json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
