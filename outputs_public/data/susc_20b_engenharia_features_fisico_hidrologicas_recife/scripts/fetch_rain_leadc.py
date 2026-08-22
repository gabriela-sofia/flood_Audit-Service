# SUSC-20B — aquisicao de chuva real (Open-Meteo ERA5-Land archive) para o novo positivo
# da Lead C (Global Flood Database, evento DFO_3291/2008, Brasilia Teimosa).
# Original: local_runs/recife_modelo_v12_extracao_final/fetch_rain_leadc.py
# Nota de sanitizacao (REV-P): path absoluto local da sessao de execucao substituido por
# placeholder relativo.
import numpy as np, pandas as pd, requests

LOCAL_RUNS_ROOT = "<local_runs_root>"  # era um path absoluto privado da sessao de execucao
V12 = f"{LOCAL_RUNS_ROOT}/recife_modelo_v12_extracao_final"
DECAY_K = 0.85
LOOKBACK = 14

df = pd.read_csv(f"{V12}/_scratch_new_positive_leadC_with_static_features.csv")
df["event_date"] = pd.to_datetime(df["event_date"])

def fetch_one(lat, lon, event_date):
    start = (event_date - pd.Timedelta(days=LOOKBACK)).date()
    end = (event_date - pd.Timedelta(days=1)).date()
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
           "&daily=precipitation_sum&timezone=America%2FRecife")
    r = requests.get(url, timeout=20)
    j = r.json()
    daily = j.get("daily", {})
    times = daily.get("time", [])
    vals = daily.get("precipitation_sum", [])
    series = {t: v for t, v in zip(times, vals)}
    ordered = [series.get(str((event_date - pd.Timedelta(days=k)).date())) for k in range(LOOKBACK, 0, -1)]
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

r = df.iloc[0]
rm, api, n = fetch_one(r.lat, r.lon, r.event_date)
print("rain_max_24h:", rm, "api:", api, "n_days_found:", n)
df["rain_max_24h_chirps"] = rm
df["rain_decay_index_api_chirps"] = api
df.to_csv(f"{V12}/_scratch_new_positive_leadC_full.csv", index=False)
