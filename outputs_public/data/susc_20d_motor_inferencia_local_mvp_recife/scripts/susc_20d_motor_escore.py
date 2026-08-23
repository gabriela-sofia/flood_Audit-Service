"""SUSC-20D -- Motor de inferencia local (MVP, Recife), sem servidor ainda.

Executa a Fase 3 do PLANO_ACAO_produto_v1.md: prova que o "motor cientifico"
roda ponta-a-ponta antes de gastar esforco em API/backend/interface.

O que este modulo faz, na ordem exata pedida pelo plano:
  1) Le os coeficientes Firth JA TREINADOS do v12 (SUSC-20C,
     `primaria_v12_firth_coefs_multivariado.csv`) -- nao retreina nada.
  2) Para cada patch_id/ponto_id conhecido (dos 278 pontos do v12), calcula
     escore = sigmoid(x_padronizado . beta) com esses coeficientes fixos.
  3) Intervalo de confianca do escore via bootstrap preditivo (decisao da
     Fase 2, `revp_fase2_decisoes_design_contrato.md`): reamostra o dataset
     de treino N=1000 vezes (mesmo desenho de SUSC-20C: estratificado por
     classe, mesma seed 20260723), reajusta Firth em cada reamostra, projeta
     o escore para ESTE ponto em cada reamostra, e toma percentis 2.5/97.5.
     A reamostragem bootstrap e validada ANTES de uso: reproduz os
     percentis publicados em `primaria_v12_coefs_bootstrap.csv`.
  4) Quais features mais pesaram: contribuicao = beta_padronizado *
     x_padronizado, ranqueada por magnitude absoluta.
  5) Evidencia DINO (opcional, nunca somada ao escore): se o ponto cai num
     patch Sentinel com embedding DINO real, anexa patch_id + vetor PCA
     como evidencia visual auxiliar.
  6) Limitacoes explicitas sempre presentes na saida.

Nao cria rotulo. Nao treina nada novo (Firth e refeito somente dentro do
procedimento de bootstrap ja documentado, nunca com features/metodologia
novas). DINO nunca e somado ao escore (Fase 1 -- decisao ja tomada com prova:
ver `revp_fase1_conclusao_dino_ab_test.md`).
"""
from __future__ import annotations

import csv
import glob
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

SEED = 20260723
N_BOOT = 1000

FEATURE_COLS = ["elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf",
                "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]
EXPECTED_SIGN = {"elevation_m": -1, "slope_deg": -1, "hand_m_dinf": -1, "twi_dinf": +1,
                 "rain_peak_residual_orthogonalized": +1, "rain_decay_index_api_chirps": +1}

BUNDLE = Path(__file__).resolve().parents[1]
SUSC20A = BUNDLE.parent / "susc_20a_aquisicao_eventos_reais_recife"
SUSC20C = BUNDLE.parent / "susc_20c_modelagem_validacao_estatistica_rigorosa_recife"
IN_V12 = SUSC20A / "dataset" / "dataset_eventos_variaveis_v12_final.csv"
IN_PUBLISHED_COEFS = SUSC20C / "results" / "primaria_v12_firth_coefs_multivariado.csv"
IN_PUBLISHED_BOOT = SUSC20C / "results" / "primaria_v12_coefs_bootstrap.csv"

RESULTS = BUNDLE / "results"
REPORTS = BUNDLE / "reports"

MODEL_VERSION = "SUSC-20 v12 (Firth, n=269, LOO-AUC=0.6781 documentado em SUSC-20C)"
REGION_SUPPORTED = "recife"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def fit_firth(X: np.ndarray, y: np.ndarray, max_iter: int = 200, tol: float = 1e-10) -> dict[str, Any]:
    """Firth penalized logistic regression via modified scores (Heinze &
    Schemper 2002). X must include an intercept column. Validated
    (see `validate_susc_20d.py`) to reproduce the published SUSC-20C
    coefficients and penalized log-likelihood to >=4 decimals."""
    n, p = X.shape
    beta = np.zeros(p)
    it = 0
    for it in range(max_iter):
        eta = X @ beta
        pi = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(pi * (1 - pi), 1e-12, None)
        Xw = X * np.sqrt(w)[:, None]
        info = Xw.T @ Xw
        info_inv = np.linalg.inv(info)
        H = Xw @ info_inv @ Xw.T
        h = np.diag(H)
        U = X.T @ (y - pi + h * (0.5 - pi))
        delta = info_inv @ U
        beta = beta + delta
        if np.linalg.norm(delta) < tol:
            break
    eta = X @ beta
    pi = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(pi * (1 - pi), 1e-12, None)
    Xw = X * np.sqrt(w)[:, None]
    info = Xw.T @ Xw
    info_inv = np.linalg.inv(info)
    se = np.sqrt(np.diag(info_inv))
    loglik = float(np.sum(y * np.log(np.clip(pi, 1e-12, None))
                          + (1 - y) * np.log(np.clip(1 - pi, 1e-12, None))))
    _, logdet = np.linalg.slogdet(info)
    penalized_loglik = loglik + 0.5 * logdet
    return {"beta": beta, "se": se, "penalized_loglik": penalized_loglik, "n_iter": it + 1}


def load_training_frame() -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, StandardScaler]:
    rows = read_csv(IN_V12)
    kept = []
    for r in rows:
        try:
            feats = {f: float(r[f]) for f in FEATURE_COLS}
        except (KeyError, ValueError):
            continue
        kept.append({"ponto_id": r["ponto_id"], "rotulo": int(r["rotulo"]),
                      "lat": r.get("lat", ""), "lon": r.get("lon", ""), **feats})
    X = np.array([[j[f] for f in FEATURE_COLS] for j in kept])
    y = np.array([j["rotulo"] for j in kept])
    scaler = StandardScaler().fit(X)
    return kept, X, y, scaler


def bootstrap_projection_draws(X: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT,
                                seed: int = SEED) -> np.ndarray:
    """Returns an (n_boot, p) array of standardized-space beta draws
    (intercept + FEATURE_COLS), same resampling scheme as SUSC-20C
    (stratified by class, with replacement)."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(pos_idx, size=len(pos_idx), replace=True),
                              rng.choice(neg_idx, size=len(neg_idx), replace=True)])
        Xb, yb = X[bi], y[bi]
        scaler_b = StandardScaler().fit(Xb)
        Xbs = scaler_b.transform(Xb)
        Xd = np.column_stack([np.ones(len(Xbs)), Xbs])
        try:
            res = fit_firth(Xd, yb)
            draws.append(res["beta"])
        except Exception:
            continue
    return np.array(draws)


def _patch_bboxes_wgs84(sentinel_dir: Path) -> dict[str, tuple[float, float, float, float]]:
    import rasterio
    from rasterio.warp import transform_bounds
    out: dict[str, tuple[float, float, float, float]] = {}
    for p in sorted(glob.glob(str(sentinel_dir / "patch_recife_*.tif"))):
        with rasterio.open(p) as ds:
            b = ds.bounds
            lon_min, lat_min, lon_max, lat_max = transform_bounds(
                ds.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
        pid = Path(p).stem.split("patch_recife_")[-1].upper()
        out[pid] = (lon_min, lat_min, lon_max, lat_max)
    return out


_DIM_COL_RE = re.compile(r"^embedding_(\d+)$")


def _parse_embedding(row: dict[str, str]) -> list[float] | None:
    dims = sorted(((int(m.group(1)), c) for c in row if (m := _DIM_COL_RE.match(c))))
    if not dims:
        return None
    try:
        return [float(row[c]) for _, c in dims]
    except ValueError:
        return None


def load_dino_evidence_index() -> tuple[dict[str, tuple[float, float, float, float]], dict[str, list[float]]]:
    """Best-effort, optional. Returns (bboxes, embeddings). Empty dicts if
    the private Sentinel patch directory is not configured -- DINO evidence
    is auxiliary, never required for the escore to be produced."""
    sentinel_dir_raw = os.environ.get("REVP_SUSC20D_SENTINEL_DIR", "")
    emb_csv_raw = os.environ.get(
        "REVP_SUSC20D_DINO_EMB",
        str(Path(__file__).resolve().parents[4] / "datasets" / "dino_recife_sedec_all_embeddings_v1r3.csv"))
    if not sentinel_dir_raw or not Path(emb_csv_raw).exists():
        return {}, {}
    bboxes = _patch_bboxes_wgs84(Path(sentinel_dir_raw))
    emb_rows = read_csv(Path(emb_csv_raw))
    emb_by_patch = {r["patch_id"]: _parse_embedding(r) for r in emb_rows}
    emb_by_patch = {k: v for k, v in emb_by_patch.items() if v is not None}
    return bboxes, emb_by_patch


def find_dino_evidence(lat: float, lon: float, bboxes: dict, emb_by_patch: dict) -> dict[str, Any] | None:
    for pid, (xmin, ymin, xmax, ymax) in bboxes.items():
        if xmin <= lon <= xmax and ymin <= lat <= ymax:
            matched = f"REC_{pid}"
            vec = emb_by_patch.get(matched)
            if vec is not None:
                return {"patch_id": matched, "embedding_dim": len(vec)}
    return None


def score_point(x_row: dict[str, float], scaler: StandardScaler, beta_point: np.ndarray,
                 boot_draws: np.ndarray) -> dict[str, Any]:
    x = np.array([[x_row[f] for f in FEATURE_COLS]])
    xs = scaler.transform(x)[0]
    xd = np.concatenate([[1.0], xs])

    eta_point = float(xd @ beta_point)
    score_point_est = float(1.0 / (1.0 + np.exp(-eta_point)))

    boot_scores = 1.0 / (1.0 + np.exp(-(boot_draws @ xd)))
    ic95_lo, ic95_hi = np.percentile(boot_scores, [2.5, 97.5])

    contributions = []
    for i, feat in enumerate(FEATURE_COLS, start=1):
        contributions.append({"variavel": feat, "contribution": float(beta_point[i] * xs[i - 1])})
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    mags = [abs(c["contribution"]) for c in contributions]
    hi_cut = mags[0] if mags else 0.0
    for c in contributions:
        frac = (abs(c["contribution"]) / hi_cut) if hi_cut > 0 else 0.0
        c["weight_tier"] = "high" if frac >= 0.66 else ("medium" if frac >= 0.33 else "low")

    return {
        "escore": round(score_point_est, 4),
        "confidence_interval": [round(float(ic95_lo), 4), round(float(ic95_hi), 4)],
        "features_used": contributions,
    }


LIMITATIONS = [
    "n=269 pontos completos (de 278 no v12), regiao=Recife apenas -- nao generaliza para outras cidades sem modelo proprio.",
    "LOO-AUC=0.6781 documentado (SUSC-20C): discriminacao modesta, nao e um detector de enchentes.",
    "DINO nao entra no escore (Fase 1: ganho nao sobreviveu ao controle cluster-robusto por patch, ver revp_fase1_conclusao_dino_ab_test.md); aparece so como evidencia visual auxiliar quando disponivel.",
    "hand_m_dinf sem sinal robusto no bootstrap (48.7% de flips de sinal, ja documentado em SUSC-20C) -- tratar sua contribuicao individual com cautela.",
    "Este escore nunca deve ser exibido como 'enchente detectada/confirmada' -- e uma suscetibilidade candidata com incerteza explicita.",
]
