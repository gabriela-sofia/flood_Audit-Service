"""SUSC-20M -- modelagem e validacao estatistica de Curitiba (SIAC 156), espelhando o
`pipeline_v12_primario.py` de Recife (SUSC-20C): screening univariado (Mann-Whitney),
Firth penalized logistic regression multivariada, bootstrap estratificado N=1000 (CIs e
taxa de sign-flip) e AUC preditivo (LOO-CV + 5-fold repetido 50x). Mesma sequencia, mesmas
funcoes, mesmos parametros.

Duas divergencias em relacao a Recife, ambas decididas e documentadas antes de rodar:

1. UNIDADE DE ANALISE = (lat, lon, data_evento), nao linha do registry. Os registries
   v20k2/v20k3 repetem a mesma ocorrencia quando o SIAC 156 a categorizou duas vezes
   (ex.: "Drenagem/Alagamentos" + "Protecao ao cidadao - GM/Enchentes"). Essas linhas tem
   variavel identica bit a bit -- conta-las como observacoes independentes e
   pseudo-replicacao. 1357 linhas -> 1148 unidades (1045 pos / 103 neg).

2. 5 FEATURES, nao 6 -- `elevation_m` cortada. Razao fisica, nao conveniencia estatistica:
   Curitiba esta num planalto (todos os 1357 pontos entre 850 e 1100 m); a elevacao absoluta
   ali carrega identidade de sub-bacia, nao exposicao local a alagamento. HAND ja e a versao
   causalmente correta da mesma ideia ("altura acima do dreno mais proximo"), e mede o que
   `elevation_m` so aproxima. O v12 de Recife corrobora: la `elevation_m` saiu com sinal
   INVERTIDO em relacao ao esperado (coef +0.2662 onde a fisica pede negativo, p=0.374),
   problema ja diagnosticado como confundimento de amostragem em
   `task2_elevacao_crs_investigacao_v9.md`.
   Consequencia no gate EPV (piso 20, revp_revisao_literatura_alinhamento_metodos_v1.md):
   classe minoritaria = negativo, n=103 -> EPV = 103/6 = 17,17 (REPROVA) e 103/5 = 20,60
   (PASSA). O gate foi checado ANTES do modelo multivariado.

   ATUALIZACAO SUSC-20N (N de negativo ampliado, n=423): a EPV com 6 features passa
   folgada (70,5). Isso NAO reabre a decisao acima -- o corte de `elevation_m` continua
   sendo primario porque o argumento e causal (confundimento com identidade de
   sub-bacia/bairro), nao estatistico, e a EPV nunca foi a razao de fundo, so a
   confirmacao formal. Reintroduzir `elevation_m` com N maior (sensibilidade S1, secao
   abaixo) da coeficiente com sinal "correto" (-0.1686, p=0.036, IC fora de zero) mas
   contribuicao preditiva NULA (LOO-AUC 0.6453 vs 0.6459 sem ela, delta -0.0006) -- ou
   seja, nao ha ganho que justifique reabrir uma decisao causal por causa de um numero
   estatistico. Sinal "correto" tambem nao refuta confundimento: bairro mais alto pode
   so registrar menos queixa por composicao socioeconomica, nao por exposicao fisica
   real. Fica como sensibilidade, nao vira primario, ate haver argumento causal novo
   (decisao humana, nao automatica -- regra fixa do projeto).

Alem do modelo primario, roda duas sensibilidades explicitamente rotuladas como
NAO-PRIMARIAS: (S1) as 6 features originais -- reprovava a EPV em n=103 (SUSC-20M),
passa folgada em n=423 (SUSC-20N) mas segue fora do primario pelo motivo causal acima;
(S2) as linhas com pseudo-replicacao (1357 no SUSC-20M, 1680 no SUSC-20N), so para medir
o efeito da unidade de analise.

Uso:
    python pipeline_v20m_curitiba_primario.py
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
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[4]
DATASET = (ROOT / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/"
                  "registries/v20l_dataset_curitiba_variaveis_v1.csv")
RESULTS = ROOT / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results"

SEED = 20260731
EPV_FLOOR = 20.0

FEATURE_COLS_PRIMARY = ["slope_deg", "hand_m_dinf", "twi_dinf",
                        "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]
FEATURE_COLS_RECIFE_6 = ["elevation_m"] + FEATURE_COLS_PRIMARY

# sinais herdados de SUSC-20C. Revisados um a um para Curitiba, nenhum alterado:
#   slope_deg   -1  vertente mais inclinada escoa, nao empoça
#   hand_m_dinf -1  mais alto acima do dreno, menos alagamento
#   twi_dinf    +1  maior indice de umidade topografica, mais acumulo
#   rain_peak_residual_orthogonalized +1  pico de chuva acima do esperado dado o antecedente
#   rain_decay_index_api_chirps       +1  solo mais saturado pelo antecedente
EXPECTED_SIGN = {"elevation_m": -1, "slope_deg": -1, "hand_m_dinf": -1, "twi_dinf": +1,
                 "rain_peak_residual_orthogonalized": +1, "rain_decay_index_api_chirps": +1}


def _firth_cls():
    """FirthLogisticRegression com o shim para a API privada `_validate_data`, removida do
    scikit-learn em 1.6. A equivalencia foi verificada reproduzindo os coeficientes,
    p-valores e ICs publicados do v12 de Recife bit a bit (diferenca maxima 0.0) --
    ver `qa_reproducao_recife_v12` neste mesmo script."""
    from firthlogist import FirthLogisticRegression
    from sklearn.utils.validation import validate_data

    if not hasattr(FirthLogisticRegression, "_validate_data"):
        FirthLogisticRegression._validate_data = (
            lambda self, X, y=None, **kw: validate_data(self, X, y, **kw))
    return FirthLogisticRegression


# ---------------------------------------------------------------------------
# etapas (identicas a pipeline_v12_primario.py)
# ---------------------------------------------------------------------------
def univariate_screen(df, feature_cols):
    rows = []
    for feat in feature_cols:
        d = df.dropna(subset=[feat])
        pos = d.loc[d["rotulo"] == 1, feat].values
        neg = d.loc[d["rotulo"] == 0, feat].values
        if len(pos) < 2 or len(neg) < 2:
            continue
        u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        r_rb = 1 - (2 * u) / (len(pos) * len(neg))
        rows.append({"variavel": feat, "n_pos": len(pos), "n_neg": len(neg),
                     "media_pos": round(float(np.mean(pos)), 4), "media_neg": round(float(np.mean(neg)), 4),
                     "mannwhitney_U": round(float(u), 2), "p_valor": round(float(p), 4),
                     "rank_biserial_r": round(float(r_rb), 4),
                     "sinal_esperado": EXPECTED_SIGN.get(feat),
                     "direcao_observada": "pos>neg" if np.mean(pos) > np.mean(neg) else "pos<neg",
                     "significante_p05": bool(p < 0.05)})
    return pd.DataFrame(rows)


def firth_multivariate(df, feature_cols):
    FirthLogisticRegression = _firth_cls()
    d = df.dropna(subset=feature_cols).copy()
    X = d[feature_cols].values
    y = d["rotulo"].astype(int).values
    Xs = StandardScaler().fit_transform(X)
    modelo = FirthLogisticRegression(fit_intercept=True)
    modelo.fit(Xs, y)
    rows = []
    for i, feat in enumerate(feature_cols):
        rows.append({"variavel": feat, "coef_padronizado": round(float(modelo.coef_[i]), 4),
                     "ic95_lo": round(float(modelo.ci_[i][0]), 4),
                     "ic95_hi": round(float(modelo.ci_[i][1]), 4),
                     "p_valor": round(float(modelo.pvals_[i]), 4),
                     "sinal_esperado": EXPECTED_SIGN.get(feat),
                     "sinal_bate_esperado": bool(np.sign(modelo.coef_[i]) == EXPECTED_SIGN.get(feat, 0)),
                     "ic95_cruza_zero": bool(modelo.ci_[i][0] <= 0 <= modelo.ci_[i][1])})
    report = {"n_used": int(len(d)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
              "events_per_predictor_minority_class": round((y == 0).sum() / len(feature_cols), 2),
              "loglik": float(modelo.loglik_)}
    return pd.DataFrame(rows), report


def bootstrap_firth_coefs(df, feature_cols, n_boot=1000, seed=SEED):
    FirthLogisticRegression = _firth_cls()
    d = df.dropna(subset=feature_cols).reset_index(drop=True)
    X = d[feature_cols].values
    y = d["rotulo"].astype(int).values
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    boot_coefs = {f: [] for f in feature_cols}
    n_failed = 0
    for b in range(n_boot):
        bi = np.concatenate([rng.choice(pos_idx, size=len(pos_idx), replace=True),
                             rng.choice(neg_idx, size=len(neg_idx), replace=True)])
        Xb, yb = X[bi], y[bi]
        try:
            Xbs = StandardScaler().fit_transform(Xb)
            m = FirthLogisticRegression(fit_intercept=True, skip_ci=True, skip_pvals=True)
            m.fit(Xbs, yb)
            for i, f in enumerate(feature_cols):
                boot_coefs[f].append(float(m.coef_[i]))
        except Exception:
            n_failed += 1
        if (b + 1) % 200 == 0:
            print(f"      bootstrap {b + 1}/{n_boot}")
    rows = []
    for f in feature_cols:
        arr = np.array(boot_coefs[f])
        point_sign = np.sign(arr.mean())
        flip_pct = 100.0 * float(np.mean(np.sign(arr) != point_sign))
        ci_lo, ci_hi = np.percentile(arr, [2.5, 97.5])
        rows.append({"variavel": f, "n_boot_sucesso": len(arr), "coef_medio_boot": round(float(arr.mean()), 4),
                     "boot_ci_low_2.5pct": round(float(ci_lo), 4), "boot_ci_high_97.5pct": round(float(ci_hi), 4),
                     "ic95_cruza_zero": bool(ci_lo <= 0 <= ci_hi), "pct_troca_de_sinal": round(flip_pct, 1)})
    return pd.DataFrame(rows), {"n_boot_requested": n_boot, "n_boot_failed": n_failed, "seed": seed}


def predictive_auc(df, feature_cols, k=5, n_repeats=50, seed=SEED):
    d = df.dropna(subset=feature_cols).reset_index(drop=True)
    X = d[feature_cols].values
    y = d["rotulo"].astype(int).values
    y_true, y_score = [], []
    for tr_idx, te_idx in LeaveOneOut().split(X):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_idx])
        Xte = scaler.transform(X[te_idx])
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y[tr_idx])
        y_score.append(clf.predict_proba(Xte)[:, 1][0])
        y_true.append(y[te_idx][0])
    loo_auc = roc_auc_score(y_true, y_score)
    rng = np.random.default_rng(seed)
    reps_auc = []
    for _ in range(n_repeats):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(rng.integers(0, 1_000_000)))
        yt, ys = [], []
        for tr_idx, te_idx in skf.split(X, y):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr_idx])
            Xte = scaler.transform(X[te_idx])
            clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
            clf.fit(Xtr, y[tr_idx])
            ys.extend(clf.predict_proba(Xte)[:, 1])
            yt.extend(y[te_idx])
        reps_auc.append(roc_auc_score(yt, ys))
    reps_auc = np.array(reps_auc)
    return {"n_used": int(len(d)), "loo_auc": round(float(loo_auc), 4), "skf_k": k, "skf_n_repeats": n_repeats,
            "skf_auc_mean": round(float(reps_auc.mean()), 4), "skf_auc_std": round(float(reps_auc.std()), 4),
            "skf_auc_min": round(float(reps_auc.min()), 4), "skf_auc_max": round(float(reps_auc.max()), 4)}


# ---------------------------------------------------------------------------
def qa_reproducao_recife_v12() -> dict:
    """Prova que o ambiente (numpy 2.x / sklearn 1.8 / shim de _validate_data) devolve o
    MESMO resultado publicado do v12 de Recife. Se isto divergir, nenhum numero de Curitiba
    abaixo e comparavel com o de Recife."""
    rec = ROOT / "outputs_public/data/susc_20a_aquisicao_eventos_reais_recife/dataset/dataset_eventos_variaveis_v12_final.csv"
    pub = ROOT / "outputs_public/data/susc_20c_modelagem_validacao_estatistica_rigorosa_recife/results/primaria_v12_firth_coefs_multivariado.csv"
    if not (rec.exists() and pub.exists()):
        return {"status": "PULADO_ARQUIVO_AUSENTE"}
    got, _ = firth_multivariate(pd.read_csv(rec), FEATURE_COLS_RECIFE_6)
    esperado = pd.read_csv(pub)
    cmp = esperado.merge(got, on="variavel", suffixes=("_publicado", "_reproduzido"))
    diffs = {c: float(np.abs(cmp[f"{c}_publicado"] - cmp[f"{c}_reproduzido"]).max())
             for c in ["coef_padronizado", "p_valor", "ic95_lo", "ic95_hi"]}
    return {"status": "REPRODUZIDO" if max(diffs.values()) == 0.0 else "DIVERGENTE",
            "n_variaveis": len(cmp), "max_abs_diff": diffs}


def gate_epv(df, feature_cols) -> dict:
    d = df.dropna(subset=feature_cols)
    n_min = int((d["rotulo"] == 0).sum())
    epv = n_min / len(feature_cols)
    return {"classe_minoritaria": "negativo", "n_minoritaria_apos_listwise": n_min,
            "n_variaveis": len(feature_cols), "epv": round(epv, 2), "piso": EPV_FLOOR,
            "passa": bool(epv >= EPV_FLOOR)}


def run_block(nome: str, df: pd.DataFrame, feats: list[str], primario: bool, out: Path,
              tag: str = "v20m") -> dict:
    print(f"\n=== {nome} | n={len(df)} | {len(feats)} features ===")
    epv = gate_epv(df, feats)
    print(f"  gate EPV: {epv['epv']} (piso {EPV_FLOOR}) -> {'PASSA' if epv['passa'] else 'REPROVA'}")
    if primario and not epv["passa"]:
        raise SystemExit("gate EPV reprovado no bloco primario -- modelo multivariado nao roda")

    print("  [1] univariado Mann-Whitney")
    univ = univariate_screen(df, feats)
    univ.to_csv(out / f"{tag}_{nome}_mannwhitney_univariado.csv", index=False)
    print(univ.to_string(index=False))

    print("  [2] Firth multivariado")
    coefs, rep = firth_multivariate(df, feats)
    coefs.to_csv(out / f"{tag}_{nome}_firth_coefs_multivariado.csv", index=False)
    print(json.dumps(rep, indent=2))
    print(coefs.to_string(index=False))

    print("  [3] bootstrap estratificado 1000")
    boot, boot_rep = bootstrap_firth_coefs(df, feats)
    boot.to_csv(out / f"{tag}_{nome}_coefs_bootstrap.csv", index=False)
    print(boot.to_string(index=False))

    print("  [4] AUC preditivo (LOO + 5-fold repetido 50x)")
    auc = predictive_auc(df, feats)
    print(json.dumps(auc, indent=2))

    return {"features": feats, "gate_epv": epv, "firth_report": rep,
            "boot_report": boot_rep, "auc_report": auc}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(DATASET))
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--skip-sensibilidades", action="store_true")
    # o conjunto primario e argumento, nao edicao de codigo: '5feat' preserva a reproducao
    # exata do SUSC-20M; '6feat' so e legitimo se o gate EPV passar com 6 (SUSC-20N).
    p.add_argument("--primario", choices=["5feat", "6feat"], default="5feat")
    p.add_argument("--tag", default="v20m", help="prefixo dos arquivos de resultado")
    args = p.parse_args(argv)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[QA] reproducao do v12 de Recife neste ambiente")
    qa = qa_reproducao_recife_v12()
    print(json.dumps(qa, indent=2))
    if qa.get("status") == "DIVERGENTE":
        raise SystemExit("ambiente nao reproduz o v12 de Recife -- nenhum numero comparavel")

    linhas = pd.read_csv(args.dataset)
    unidades = linhas[~linhas["e_unidade_duplicada"]].reset_index(drop=True)
    print(f"\ndataset: {len(linhas)} linhas -> {len(unidades)} unidades de observacao "
          f"({int((unidades.rotulo == 1).sum())} pos / {int((unidades.rotulo == 0).sum())} neg)")

    reports = {"qa_reproducao_recife_v12": qa,
               "unidade_de_analise": "(lat, lon, data_evento) -- linhas pseudo-replicadas excluidas",
               "n_linhas_registry": int(len(linhas)), "n_unidades": int(len(unidades)),
               "seed": SEED}

    seis = args.primario == "6feat"
    feats_primario = FEATURE_COLS_RECIFE_6 if seis else FEATURE_COLS_PRIMARY
    feats_sens = FEATURE_COLS_PRIMARY if seis else FEATURE_COLS_RECIFE_6
    nome_primario = f"primario_{'6' if seis else '5'}feat"
    nome_sens = f"sens_s1_{'5' if seis else '6'}feat"

    reports[f"primario_{len(feats_primario)}_features"] = run_block(
        nome_primario, unidades, feats_primario, True, out, args.tag)

    if not args.skip_sensibilidades:
        reports[f"sensibilidade_s1_{len(feats_sens)}_features"] = run_block(
            nome_sens, unidades, feats_sens, False, out, args.tag)
        reports["sensibilidade_s2_linhas_com_pseudo_replicacao"] = run_block(
            "sens_s2_linhas", linhas, feats_primario, False, out, args.tag)

    (out / f"{args.tag}_relatorios_consolidados.json").write_text(
        json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"\nresultados -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
