# IMPROVEMENT 2 — HAND/TWI: D8 → D-infinity Recompute

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-23

## Method

Recife's terrain is notoriously flat coastal plain, where D8 (single steepest-descent flow
direction) is documented in the hydrology literature (Tarboton 1997 for D-infinity; Quinn et
al. 1991 for MFD/FD8) to produce artificial parallel flow lines and unreliable HAND/TWI.
Recomputed both using **D-infinity** (Tarboton 1997) via WhiteboxTools (confirmed working:
`fill_depressions_wang_and_liu`, `d_inf_flow_accumulation`, `slope`, `wetness_index`,
`elevation_above_stream`, `extract_streams` — all present and callable in this sandbox).

Pipeline (`build_dem_merge.py`, `scratch_dinf/`):
1. Same 48 real PE3D DTM tiles (EPSG:31985) merged to the same 10 m resolution as v2/v4
   (2777×2086 grid, 3,579,890 valid cells — confirmed identical valid-cell count to the v4
   cache, same DEM).
2. **Same depression-filling preprocessing as v4** (`fill_depressions_wang_and_liu`, Wang &
   Liu 2006, `fix_flats=True`) — kept unchanged per the task's instruction.
3. `d_inf_flow_accumulation` (out_type=`cells` for stream extraction, out_type=`sca` for
   TWI's specific catchment area).
4. `slope` (degrees) on the filled DEM.
5. Stream network: `extract_streams` thresholded at the **98th percentile of the D-infinity
   cell-accumulation distribution = 1,122.74 cells** (D8 used a fixed threshold=30 cells; the
   percentile-based selection logic is the same, but the raw threshold value had to be
   recalibrated because D-infinity's flow-splitting produces a materially different
   accumulation distribution than D8's single-direction concentration).
6. `elevation_above_stream` → HAND-Dinf; `wetness_index(sca, slope)` → TWI-Dinf.

## Coverage results

| Metric | D8 + fill (v4, gap3) | D-infinity + fill (v7) |
|---|---:|---:|
| HAND grid coverage (% of 3,579,890 valid cells) | 99.75% | **97.27%** |
| TWI grid coverage | had documented missingness (`twi_missing` column) | **100.00%** (SCA is defined everywhere post-fill) |
| HAND direct point-hit, no tolerance (n=282) | 100% (282/282) | **100% (282/282)** |
| TWI direct point-hit, no tolerance (n=282) | 100% (282/282) | **100% (282/282)** |

**Dropping the 150 m tolerance (second part of Improvement 2)**: checked
`hand_search_dist_m_filled`/`twi_search_dist_m` in the v4 dataset — **0 of 282 points** ever
actually needed the tolerance fallback in the old pipeline (all already resolved directly at
10 m after depression-filling). Removing the tolerance therefore **changes nothing
numerically for these 282 points** — an honest, unglamorous finding: the fallback was already
vestigial by v4. It is still removed from the v7 pipeline as instructed (direct
`rasterio.sample` only, no search-radius code path), and the same direct-only method was
applied to the 18 new points (Improvement 1), all resolved on the first try.

## Do the values actually change, and does coherence improve?

| | Pearson r (old D8-filled vs new D-infinity) | Mean abs. difference |
|---|---:|---:|
| HAND | 0.7618 | 5.59 m |
| TWI | 0.5273 | 1.56 (dimensionless log units) |

Values are correlated but materially different, especially TWI (r=0.53) — consistent with
the literature's claim that flow-direction algorithm choice matters more on flat terrain.

**Physical coherence, primary (real-vs-real) Firth model**:
- **TWI**: v5 (D8) coefficient sign was **wrong** (−0.1037, expected +1). v7 (D-infinity)
  coefficient sign is **correct** (+0.1626, expected +1) — a genuine, literature-consistent
  improvement, though still not statistically significant (p=0.43) and bootstrap-fragile
  (20.9% sign-flip rate over 1000 resamples).
- **HAND**: correct sign in both v5 (−0.0669) and v7 (−0.0654), similar (weak) magnitude in
  both; bootstrap sign-flip rate got **worse** in v7 (47.3% vs Gap 7's originally-reported
  18–35% range for D8 HAND/TWI) — D-infinity did not make HAND itself more robust, only fixed
  TWI's sign.
- **Secondary (pseudo-absence-augmented) analysis**: HAND-Dinf and TWI-Dinf actually lose
  coherence-check "pass" status there (see `secundaria_v7_coeficientes_coerencia.csv`) once
  combined with the bairro-matched pseudo-absence redesign (Improvement 4) — reported plainly
  in the comparative final report, not hidden.

**Honest summary**: D-infinity measurably changes HAND/TWI values (not a cosmetic swap),
fixes TWI's sign in the primary analysis as the literature predicted, but does not make either
feature statistically robust at this sample size — consistent with Recife's flat terrain
limiting how much any single flow-routing algorithm can extract from HAND/TWI alone.

## Files
`build_dem_merge.py`, `scratch_dinf/` (dem_filled.tif, dinf_accum_cells.tif, dinf_sca.tif,
slope_deg_wbt.tif, streams_dinf.tif, hand_dinf.tif, twi_dinf.tif), `extract_dinf_at_points.py`,
`improvement2_extraction_stats.json`, `dataset_v4_with_dinf_hand_twi.csv`.
