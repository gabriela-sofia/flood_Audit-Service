"""Valida terrain_features_on_demand.py reamostrando os 278 pontos do v12 e
comparando com os valores ja armazenados em dataset_v12_final.csv (que vieram da
MESMA extracao original). Isto prova que a amostragem sob demanda reproduz
exatamente o que treinou o modelo -- nao um pipeline paralelo divergente.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from terrain_features_on_demand import sample_terrain_features

IN_V12 = Path(__file__).resolve().parents[2] / "susc_20a_aquisicao_eventos_reais_recife" / "dataset" / "dataset_eventos_features_v12_final.csv"
OUT = Path(__file__).resolve().parents[1] / "results"

FEATS = ["elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf"]


def run() -> None:
    with IN_V12.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    diffs = {f: [] for f in FEATS}
    n_in_coverage = 0
    n_out_of_coverage = 0
    mismatches = []

    for r in rows:
        try:
            lat, lon = float(r["lat"]), float(r["lon"])
            stored = {f: float(r[f]) for f in FEATS}
        except (KeyError, ValueError):
            continue
        sampled = sample_terrain_features(lat, lon)
        if sampled is None:
            n_out_of_coverage += 1
            continue
        n_in_coverage += 1
        for f in FEATS:
            d = abs(sampled[f] - stored[f])
            diffs[f].append(d)
            if d > 0.5:
                mismatches.append({"point_id": r["point_id"], "feature": f,
                                    "stored": stored[f], "sampled": sampled[f], "diff": d})

    summary = {"n_total_v12_points": len(rows), "n_in_coverage": n_in_coverage,
               "n_out_of_coverage": n_out_of_coverage, "n_mismatches_over_0.5": len(mismatches)}
    for f in FEATS:
        arr = np.array(diffs[f])
        if len(arr):
            summary[f"{f}_max_abs_diff"] = round(float(arr.max()), 6)
            summary[f"{f}_mean_abs_diff"] = round(float(arr.mean()), 6)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "validate_terrain_on_demand_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "mismatches_sample": mismatches[:20]}, fh, indent=2)

    print(json.dumps(summary, indent=2))
    if mismatches:
        print(f"WARNING: {len(mismatches)} mismatches > 0.5 -- see mismatches_sample below")
        for m in mismatches[:10]:
            print(" ", m)


if __name__ == "__main__":
    run()
