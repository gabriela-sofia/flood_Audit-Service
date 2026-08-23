"""Valida chuva_sob_demanda.py re-buscando a chuva real (Open-Meteo archive)
pros pontos do v12 cuja fonte original ja era open_meteo_era5_land_archive_api, e
comparando com os valores ja armazenados em dataset_v12_final.csv.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from chuva_sob_demanda import rain_features_for_query

IN_V12 = Path(__file__).resolve().parents[2] / "susc_20a_aquisicao_eventos_reais_recife" / "dataset" / "dataset_eventos_variaveis_v12_final.csv"
OUT = Path(__file__).resolve().parents[1] / "results"


def run(sample_n: int = 30) -> None:
    with IN_V12.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    target_rows = [r for r in rows if r.get("rain_data_source") == "open_meteo_era5_land_archive_api"
                   and r.get("data_evento")]
    print(f"n_open_meteo_source_rows={len(target_rows)}, sampling {min(sample_n, len(target_rows))}")

    diffs = {"rain_decay_index_api_chirps": [], "rain_peak_residual_orthogonalized": []}
    n_ok = 0
    n_fail = 0
    mismatches = []

    for r in target_rows[:sample_n]:
        lat, lon = float(r["lat"]), float(r["lon"])
        data_evento = datetime.strptime(r["data_evento"][:10], "%Y-%m-%d").date()
        try:
            stored_decay = float(r["rain_decay_index_api_chirps"])
            stored_resid = float(r["rain_peak_residual_orthogonalized"])
        except (KeyError, ValueError):
            continue
        out = rain_features_for_query(lat, lon, data_evento)
        time.sleep(0.3)
        if out is None:
            n_fail += 1
            continue
        n_ok += 1
        d1 = abs(out["rain_decay_index_api_chirps"] - stored_decay)
        d2 = abs(out["rain_peak_residual_orthogonalized"] - stored_resid)
        diffs["rain_decay_index_api_chirps"].append(d1)
        diffs["rain_peak_residual_orthogonalized"].append(d2)
        if d1 > 0.5 or d2 > 0.5:
            mismatches.append({"ponto_id": r["ponto_id"], "data_evento": r["data_evento"],
                                "stored_decay": stored_decay, "sampled_decay": out["rain_decay_index_api_chirps"],
                                "stored_resid": stored_resid, "sampled_resid": out["rain_peak_residual_orthogonalized"]})

    summary = {"n_sampled": n_ok + n_fail, "n_ok": n_ok, "n_fail_api": n_fail,
               "n_mismatches_over_0.5": len(mismatches)}
    for k, arr in diffs.items():
        a = np.array(arr)
        if len(a):
            summary[f"{k}_max_abs_diff"] = round(float(a.max()), 6)
            summary[f"{k}_mean_abs_diff"] = round(float(a.mean()), 6)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "validate_rain_on_demand_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "mismatches": mismatches}, fh, indent=2)

    print(json.dumps(summary, indent=2))
    for m in mismatches[:10]:
        print(" ", m)


if __name__ == "__main__":
    run()
