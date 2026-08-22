# IMPROVEMENT 3 — Rainfall Pair Orthogonalization

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-23

## Method

Instead of dropping either `rain_max_24h_chirps` (trigger/peak intensity) or
`rain_decay_index_api_chirps` (antecedent-saturation/retention, API with k=0.85 decay), both
conceptually distinct, residualized the peak feature against the API feature: ordinary linear
regression `rain_max_24h_chirps ~ rain_decay_index_api_chirps`, keeping the residual as
`rain_peak_residual_orthogonalized` — a standard technique ("surprise peak intensity beyond
baseline saturation") that removes the correlated component while retaining both concepts'
distinct information.

## Results

| Dataset | n used | r before | β (slope) | r after (by construction) |
|---|---:|---:|---:|---:|
| Primary (real-vs-real) | 172 | **0.7218** | 0.8809 | **0.0000** |
| Secondary (pseudo-absence-augmented) | 317 | **0.7439** | 0.9693 | **0.0000** |

Matches the task brief's stated prior correlation range (r=0.72–0.76). Post-orthogonalization
correlation is exactly 0 by construction (residual of an OLS fit is always uncorrelated with
the regressor).

## Effect on coefficients/signs

**Primary Firth model** (`primaria_v7_firth_multivariate_coefs.csv`):
- `rain_decay_index_api_chirps` (retained, unchanged): coef=+0.7849, **p=0.0045** (highly
  significant — a real improvement over v5's same feature, which had coef=+0.6889, p=0.1324,
  not significant). Bootstrap: CI [0.414, 1.492], **0% sign-flips** over 1000 resamples — the
  single most robust feature in either v5 or v7's primary model.
- `rain_peak_residual_orthogonalized` (new): coef=**−0.1261**, expected sign +1, **wrong
  sign**, p=0.6122 (not significant), bootstrap sign-flip rate 32.9%.

**Honest reading**: orthogonalizing did not validate the "surprise peak intensity" hypothesis
— once the shared saturation/retention signal is removed, what's left of the peak-intensity
concept shows no reliable relationship with flood-positive status in this sample (indeed a
non-significant negative point estimate). What it *did* do is make the retained API/decay
feature's own signal clearer and more statistically robust (removing the collinear peak
component apparently reduced noise/variance inflation on that coefficient) — the rainfall
construct as a whole became more defensible, but through the antecedent-saturation channel,
not the "surprise peak" channel the orthogonalization was designed to isolate.

**Secondary analysis** (`secundaria_v7_coeficientes_coerencia.csv`): both
`rain_decay_index_api_chirps` (coef=+0.8217) and `rain_peak_residual_orthogonalized`
(coef=+0.1299) come out correctly signed and coherent in the secondary logistic regression —
different from the primary result, likely because the much larger, more heterogeneous
secondary sample (n=317) lets the small residual peak-intensity signal show its (correctly
signed, this time) direction, even though it remains a minor contributor next to the
dominant API feature.

## Files
`build_v7_final_datasets.py` (orthogonalize function), `improvement3_orthogonalization_stats.json`.
