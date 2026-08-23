# SUSC-20B — script final de consolidacao do dataset v12 (evento + features fisico-hidrologicas)
# Original: local_runs/recife_modelo_v12_extracao_final/montar_dataset_v12.py
# Nota de sanitizacao (REV-P): o path absoluto local da sessao de execucao foi substituido
# por um placeholder relativo. Os arquivos _scratch_*/dataset_v9_final.csv/sar_backscatter_*
# referenciados abaixo sao artefatos intermediarios de local_runs (nao versionados no REV-P
# por politica interna). Script mantido como registro do metodo real executado.
import json
import numpy as np
import pandas as pd

LOCAL_RUNS_ROOT = "<local_runs_root>"  # era um path absoluto privado da sessao de execucao
V9 = f"{LOCAL_RUNS_ROOT}/recife_modelo_v9_final"
V12 = f"{LOCAL_RUNS_ROOT}/recife_modelo_v12_extracao_final"

v9 = pd.read_csv(f"{V9}/dataset_v9_final.csv")
new_pos_a = pd.read_csv(f"{V12}/_scratch_new_positives_leadA_full.csv")
new_pos_c = pd.read_csv(f"{V12}/_scratch_new_positive_leadC_full.csv")

common_cols = ["ponto_id", "regiao", "rotulo", "lat", "lon", "data_evento", "bairro",
               "tipo_fonte_negativo", "elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf",
               "rain_max_24h_chirps", "rain_decay_index_api_chirps", "mapbiomas_class_2023",
               "rain_data_source"]
new_pos_a_slice = new_pos_a[common_cols].copy()
new_pos_a_slice["papel_fonte_dataset"] = "LEADA_V12_DIARIO_OFICIAL_DECRETO_35669_28MAI2022_BAIRRO_LEVEL"
new_pos_c_slice = new_pos_c[common_cols].copy()
new_pos_c_slice["papel_fonte_dataset"] = "LEADC_V12_GLOBAL_FLOOD_DATABASE_DFO3291_EVENT_EXTENT_CENTROID"

v9_slice = v9[common_cols + ["papel_fonte_dataset"]].copy()

combined = pd.concat([v9_slice, new_pos_a_slice, new_pos_c_slice], ignore_index=True, sort=False)
print("combined n =", len(combined), f"({(combined.rotulo==1).sum()} pos / {(combined.rotulo==0).sum()} neg)")

def orthogonalize_single(d):
    x = d["rain_decay_index_api_chirps"].values.reshape(-1, 1)
    y = d["rain_max_24h_chirps"].values
    xm = x.mean()
    xc = x - xm
    beta = float((xc.flatten() @ y) / (xc.flatten() @ xc.flatten()))
    intercept = float(y.mean() - beta * xm)
    pred = intercept + beta * d["rain_decay_index_api_chirps"].values
    resid = y - pred
    return resid, beta, intercept

def orthogonalize_per_source(df):
    df = df.copy()
    df["rain_peak_residual_orthogonalized"] = np.nan
    reports = {}
    for src, sub in df.dropna(subset=["rain_max_24h_chirps", "rain_decay_index_api_chirps"]).groupby("rain_data_source"):
        resid, beta, intercept = orthogonalize_single(sub)
        df.loc[sub.index, "rain_peak_residual_orthogonalized"] = resid
        corr_before = float(sub["rain_max_24h_chirps"].corr(sub["rain_decay_index_api_chirps"]))
        corr_after = float(pd.Series(resid, index=sub.index).corr(sub["rain_decay_index_api_chirps"]))
        reports[src] = {"n_used": int(len(sub)), "beta": round(beta, 4), "intercept": round(intercept, 4),
                         "pearson_r_before_orthogonalization": round(corr_before, 4),
                         "pearson_r_after_orthogonalization": round(corr_after, 6)}
    return df, reports

combined, ortho_report_per_source = orthogonalize_per_source(combined)

# SAR informational (only present for v9 rows; new positives get NaN -- honest, not computed)
sar = pd.read_csv(f"{V9}/sar_backscatter_pairs_v9_final.csv")
sar_slice = sar[["ponto_id", "delta_vv_db", "delta_vh_db", "cena_ambigua",
                 "dias_antes", "dias_depois"]].rename(
    columns={"delta_vv_db": "sar_delta_vv_db_INFORMATIONAL_NOT_MODELED",
             "delta_vh_db": "sar_delta_vh_db_INFORMATIONAL_NOT_MODELED"})
combined = combined.merge(sar_slice, on="ponto_id", how="left")

combined.to_csv(f"{V12}/dataset_v12_final.csv", index=False)
with open(f"{V12}/v12_estatisticas_ortogonalizacao.json", "w") as f:
    json.dump({"per_source_deconfounded": ortho_report_per_source}, f, indent=2)
print("orthogonalization:", json.dumps(ortho_report_per_source, indent=2))
print("wrote dataset_v12_final.csv, shape", combined.shape)
