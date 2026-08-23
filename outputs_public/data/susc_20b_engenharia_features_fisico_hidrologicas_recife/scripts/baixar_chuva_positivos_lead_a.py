# SUSC-20B — aquisicao de chuva real (Open-Meteo ERA5-Land archive) para os 8 novos
# positivos da Lead A (Diario Oficial, decreto 35.669/2022-05-28).
# Original: local_runs/recife_modelo_v12_extracao_final/baixar_chuva_positivos_lead_a.py
# Nota de sanitizacao (REV-P): path absoluto local da sessao de execucao substituido por
# placeholder relativo.
import numpy as np, pandas as pd, requests, time

LOCAL_RUNS_ROOT = "<local_runs_root>"  # era um path absoluto privado da sessao de execucao
V12 = f"{LOCAL_RUNS_ROOT}/recife_modelo_v12_extracao_final"
DECAY_K = 0.85
LOOKBACK = 14

df = pd.read_csv(f"{V12}/_scratch_new_positives_leadA_with_static_features.csv")
df["data_evento"] = pd.to_datetime(df["data_evento"])

def fetch_one(lat, lon, data_evento):
    start = (data_evento - pd.Timedelta(days=LOOKBACK)).date()
    end = (data_evento - pd.Timedelta(days=1)).date()
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
           "&daily=precipitation_sum&timezone=America%2FRecife")
    r = requests.get(url, timeout=20)
    j = r.json()
    daily = j.get("daily", {})
    times = daily.get("time", [])
    vals = daily.get("precipitation_sum", [])
    series = {t: v for t, v in zip(times, vals)}
    ordered = [series.get(str((data_evento - pd.Timedelta(days=k)).date())) for k in range(LOOKBACK, 0, -1)]
    ordered = [np.nan if v is None else float(v) for v in ordered]
    arr = np.array(ordered, dtype=float)
    n_found = int(np.sum(~np.isnan(arr)))
    if n_found == 0:
        return None, None, 0
    rain_max = float(np.nanmax(arr))
    filled = np.where(np.isnan(arr), 0.0, arr)
    api = 0.0
    for idx, p in enumerate(filled):
        day_offset = LOOKBACK - idx
        api += p * (DECAY_K ** (day_offset - 1))
    return round(rain_max, 2), round(float(api), 3), n_found

rows = []
for r in df.itertuples():
    rm, api, n = fetch_one(r.lat, r.lon, r.data_evento)
    rows.append({"ponto_id": r.ponto_id, "rain_max_24h_chirps": rm,
                 "rain_decay_index_api_chirps": api, "n_days_found": n})
    time.sleep(0.3)

rain_df = pd.DataFrame(rows)
print(rain_df)
merged = df.merge(rain_df, on="ponto_id", how="left")
merged.to_csv(f"{V12}/_scratch_new_positives_leadA_full.csv", index=False)
print("saved")
