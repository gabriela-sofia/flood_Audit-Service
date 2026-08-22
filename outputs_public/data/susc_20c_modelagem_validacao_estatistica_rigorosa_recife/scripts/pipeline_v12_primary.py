# SUSC-20C — pipeline final de modelagem/validacao estatistica rigorosa (v12 primary):
# screening univariado (Mann-Whitney), Firth penalized logistic regression multivariada,
# bootstrap estratificado N=1000 (CIs e taxa de sign-flip), e AUC preditivo (LOO-CV +
# 5-fold repetido 50x). Mesma metodologia exata usada desde v5.
# Original: local_runs/recife_modelo_v12_extracao_final/pipeline_v12_primary.py
# Nota de sanitizacao (REV-P): path absoluto local da sessao de execucao substituido por
# placeholder relativo.
import json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

LOCAL_RUNS_ROOT = "<local_runs_root>"  # era um path absoluto privado da sessao de execucao
V12 = f"{LOCAL_RUNS_ROOT}/recife_modelo_v12_extracao_final"
SEED = 20260723

FEATURE_COLS_PRIMARY = ["elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf",
                         "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]
EXPECTED_SIGN = {"elevation_m": -1, "slope_deg": -1, "hand_m_dinf": -1, "twi_dinf": +1,
                  "rain_peak_residual_orthogonalized": +1, "rain_decay_index_api_chirps": +1}

primary = pd.read_csv(f"{V12}/dataset_v12_final.csv")
print(f"primary n={len(primary)} ({(primary.label==1).sum()} pos / {(primary.label==0).sum()} neg)")

def univariate_screen(df, feature_cols):
    rows = []
    for feat in feature_cols:
        d = df.dropna(subset=[feat])
        pos = d.loc[d["label"] == 1, feat].values
        neg = d.loc[d["label"] == 0, feat].values
        if len(pos) < 2 or len(neg) < 2:
            continue
        u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        r_rb = 1 - (2 * u) / (len(pos) * len(neg))
        rows.append({"feature": feat, "n_pos": len(pos), "n_neg": len(neg),
                      "mean_pos": round(float(np.mean(pos)), 4), "mean_neg": round(float(np.mean(neg)), 4),
                      "mannwhitney_U": round(float(u), 2), "p_value": round(float(p), 4),
                      "rank_biserial_r": round(float(r_rb), 4),
                      "expected_sign": EXPECTED_SIGN.get(feat),
                      "direction_observed": "pos>neg" if np.mean(pos) > np.mean(neg) else "pos<neg",
                      "significant_p05": bool(p < 0.05)})
    return pd.DataFrame(rows)

def firth_multivariate(df, feature_cols):
    from firthlogist import FirthLogisticRegression
    d = df.dropna(subset=feature_cols).copy()
    X = d[feature_cols].values
    y = d["label"].astype(int).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
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
    coef_df = pd.DataFrame(rows)
    report = {"n_used": int(len(d)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
              "events_per_predictor_minority_class": round((y == 0).sum() / len(feature_cols), 2),
              "loglik": float(model.loglik_)}
    return coef_df, report

def bootstrap_firth_coefs(df, feature_cols, n_boot=1000, seed=SEED):
    from firthlogist import FirthLogisticRegression
    d = df.dropna(subset=feature_cols).reset_index(drop=True)
    X = d[feature_cols].values
    y = d["label"].astype(int).values
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    boot_coefs = {f: [] for f in feature_cols}
    n_failed = 0
    for b in range(n_boot):
        bi = np.concatenate([rng.choice(pos_idx, size=len(pos_idx), replace=True),
                              rng.choice(neg_idx, size=len(neg_idx), replace=True)])
        Xb, yb = X[bi], y[bi]
        try:
            scaler = StandardScaler()
            Xbs = scaler.fit_transform(Xb)
            m = FirthLogisticRegression(fit_intercept=True, skip_ci=True, skip_pvals=True)
            m.fit(Xbs, yb)
            for i, f in enumerate(feature_cols):
                boot_coefs[f].append(float(m.coef_[i]))
        except Exception:
            n_failed += 1
    rows = []
    for f in feature_cols:
        arr = np.array(boot_coefs[f])
        point_sign = np.sign(arr.mean())
        flip_pct = 100.0 * float(np.mean(np.sign(arr) != point_sign))
        ci_lo, ci_hi = np.percentile(arr, [2.5, 97.5])
        rows.append({"feature": f, "n_boot_success": len(arr), "boot_mean_coef": round(float(arr.mean()), 4),
                      "boot_ci_low_2.5pct": round(float(ci_lo), 4), "boot_ci_high_97.5pct": round(float(ci_hi), 4),
                      "ci_crosses_zero": bool(ci_lo <= 0 <= ci_hi), "pct_sign_flips": round(flip_pct, 1)})
    return pd.DataFrame(rows), {"n_boot_requested": n_boot, "n_boot_failed": n_failed, "seed": seed}

def predictive_auc(df, feature_cols, k=5, n_repeats=50, seed=SEED):
    d = df.dropna(subset=feature_cols).reset_index(drop=True)
    X = d[feature_cols].values; y = d["label"].astype(int).values
    loo = LeaveOneOut()
    y_true, y_score = [], []
    for tr_idx, te_idx in loo.split(X):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_idx]); Xte = scaler.transform(X[te_idx])
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y[tr_idx])
        y_score.append(clf.predict_proba(Xte)[:, 1][0]); y_true.append(y[te_idx][0])
    loo_auc = roc_auc_score(y_true, y_score)
    rng = np.random.default_rng(seed)
    reps_auc = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(rng.integers(0, 1_000_000)))
        yt, ys = [], []
        for tr_idx, te_idx in skf.split(X, y):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr_idx]); Xte = scaler.transform(X[te_idx])
            clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
            clf.fit(Xtr, y[tr_idx])
            ys.extend(clf.predict_proba(Xte)[:, 1]); yt.extend(y[te_idx])
        reps_auc.append(roc_auc_score(yt, ys))
    reps_auc = np.array(reps_auc)
    return {"n_used": int(len(d)), "loo_auc": round(float(loo_auc), 4), "skf_k": k, "skf_n_repeats": n_repeats,
            "skf_auc_mean": round(float(reps_auc.mean()), 4), "skf_auc_std": round(float(reps_auc.std()), 4),
            "skf_auc_min": round(float(reps_auc.min()), 4), "skf_auc_max": round(float(reps_auc.max()), 4)}

print("[1] PRIMARY univariate")
univ_df = univariate_screen(primary, FEATURE_COLS_PRIMARY)
univ_df.to_csv(f"{V12}/primaria_v12_univariate_mannwhitney.csv", index=False)
print(univ_df.to_string(index=False))

print("[2] PRIMARY Firth multivariate")
firth_df, firth_report = firth_multivariate(primary, FEATURE_COLS_PRIMARY)
firth_df.to_csv(f"{V12}/primaria_v12_firth_multivariate_coefs.csv", index=False)
print(json.dumps(firth_report, indent=2)); print(firth_df.to_string(index=False))

print("[3] PRIMARY bootstrap (1000)")
boot_df, boot_report = bootstrap_firth_coefs(primary, FEATURE_COLS_PRIMARY)
boot_df.to_csv(f"{V12}/primaria_v12_bootstrap_coefs.csv", index=False)
print(json.dumps(boot_report, indent=2)); print(boot_df.to_string(index=False))

print("[4] PRIMARY predictive AUC (LOO + repeated k-fold)")
auc_report = predictive_auc(primary, FEATURE_COLS_PRIMARY)
with open(f"{V12}/primaria_v12_predictive_auc.json", "w") as f:
    json.dump(auc_report, f, indent=2)
print(json.dumps(auc_report, indent=2))

all_reports = {"firth_report": firth_report, "boot_report": boot_report, "auc_report": auc_report}
with open(f"{V12}/all_reports_v12_primary.json", "w") as f:
    json.dump(all_reports, f, indent=2, default=str)
print("DONE.")
