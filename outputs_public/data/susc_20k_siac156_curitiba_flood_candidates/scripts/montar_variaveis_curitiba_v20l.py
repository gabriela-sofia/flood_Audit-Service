"""SUSC-20L -- engenharia de features fisico-hidrologicas para os pontos SIAC 156 de
Curitiba (positivos v20k2 + negativos v20k3), espelhando o metodo ja validado do
dataset v12 de Recife (SUSC-20B).

Schema-alvo (identico ao `dataset_eventos_variaveis_v12_final.csv`):
    elevation_m, slope_deg, hand_m_dinf, twi_dinf,
    rain_max_24h_chirps, rain_decay_index_api_chirps,
    mapbiomas_class_2023, rain_data_source,
    rain_peak_residual_orthogonalized

Equivalencias de fonte por etapa (o que e identico a Recife e o que e o analogo local):

  terreno   -- AMOSTRAGEM de raster ja processado, nao recalculo. Recife amostrou os
               rasters D-infinity derivados do DTM PE3D 1m (EPSG:31985); Curitiba
               amostra os rasters D-infinity de SUSC-20G derivados do MDT SGB/CPRM 10m
               (EPSG:31982). Mesmo pipeline WhiteboxTools 2.4.0, mesmas regras de
               NODATA do `ler_hand_twi_declividade_no_ponto.py` (SUSC-20G).
  chuva     -- Open-Meteo ERA5-Land archive API, MESMA formula de
               `baixar_chuva_positivos_lead_a.py` (SUSC-20B): janela de 14 dias terminando
               no dia ANTERIOR ao evento (nunca inclui o dia do evento),
               rain_max_24h = maximo diario da janela, indice API com decaimento
               k=0.85. Os nomes de coluna com sufixo `_chirps` sao legado do v12 de
               Recife (la 181 pontos vieram de CHIRPS e 97 de Open-Meteo); a coluna
               `rain_data_source` e quem diz a fonte real. Curitiba tem fonte unica:
               `open_meteo_era5_land_archive_api`.
               Divergencia declarada: timezone `America/Sao_Paulo` em vez de
               `America/Recife` (fuso local correto de cada regiao; ambos UTC-3 sem
               horario de verao no periodo coberto, entao os cortes diarios coincidem).
  mapbiomas -- MapBiomas Colecao 9, banda `classification_2023`, escala 30 m, mesmo
               asset publico ja usado em SUSC-19B/19F.
  ortogon.  -- `orthogonalize_per_source`, logica identica a `montar_dataset_v12.py`,
               refitada nos pontos de Curitiba (nao herda beta/intercept de Recife --
               sao regioes e regimes de chuva distintos).

Fail-closed em toda etapa: valor ausente vira NaN + coluna `*_status` explicita.
Nada e imputado, interpolado ou preenchido com media.

Cache: as respostas da API de chuva e do Earth Engine sao gravadas em `--cache-dir`
(dentro de `local_runs/`, git-ignored) para retomada sem repetir chamada de rede.

Uso:
    python montar_variaveis_curitiba_v20l.py                 # roda tudo
    python montar_variaveis_curitiba_v20l.py --skip-rain     # so terreno + mapbiomas
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[4]
REGISTRIES = ROOT / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/registries"
POS_CSV = REGISTRIES / "v20k2_dataset_positivos_curitiba_siac156_v1.csv"
NEG_CSV = REGISTRIES / "v20k3_dataset_negativos_curitiba_siac156_v1.csv"
OUT_CSV = REGISTRIES / "v20l_dataset_curitiba_variaveis_v1.csv"

DEFAULT_RASTER_DIR = ROOT / "local_runs/susc_20g_hand_twi_dinfinity_generico/curitiba"
DEFAULT_CACHE_DIR = ROOT / "local_runs/susc_20l_curitiba_features"

# variavel -> arquivo raster (elevacao vem do MDT bruto, nao do dem_filled hidro-condicionado,
# espelhando a extracao original do v12 que amostrou os tiles PE3D crus)
RASTER_FILES = {
    "elevation_m": "curitiba_dtm_10m.tif",
    "slope_deg": "slope_deg_wbt.tif",
    "hand_m_dinf": "hand_dinf.tif",
    "twi_dinf": "twi_dinf.tif",
}

# --- chuva (identico a SUSC-20B) ---
DECAY_K = 0.85
LOOKBACK_DAYS = 14
RAIN_TIMEZONE = "America/Sao_Paulo"
RAIN_SOURCE_LABEL = "open_meteo_era5_land_archive_api"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# --- mapbiomas (identico a SUSC-19B/19F) ---
MAPBIOMAS_COLLECTION = (
    "projects/mapbiomas-public/assets/brazil/lulc/collection9/"
    "mapbiomas_collection90_integration_v1"
)
MAPBIOMAS_BAND = "classification_2023"
MAPBIOMAS_SCALE = 30


# ---------------------------------------------------------------------------
# entrada
# ---------------------------------------------------------------------------
def load_points(pos_csv: Path = POS_CSV, neg_csv: Path = NEG_CSV) -> pd.DataFrame:
    pos = pd.read_csv(pos_csv, encoding="utf-8")
    neg = pd.read_csv(neg_csv, encoding="utf-8")
    pos["registro_fonte"] = pos_csv.name
    neg["registro_fonte"] = neg_csv.name
    df = pd.concat([pos, neg], ignore_index=True, sort=False)
    df["data_evento"] = pd.to_datetime(df["data_evento"], format="%d/%m/%Y")
    # unidade de observacao fisico-hidrologica: um local + uma data. Linhas que repetem
    # (ponto_id, data_evento) sao a MESMA ocorrencia categorizada duas vezes no SIAC 156
    # (ex.: "Drenagem/Alagamentos" + "Protecao ao cidadao - GM/Enchentes"), ver
    # susc_20k2_k3_replica_completa_metodo_recife_report.md secao 4. Marcadas aqui,
    # nao removidas -- a decisao de unidade de analise e da etapa de modelagem.
    key = df["lat"].round(7).astype(str) + "|" + df["lon"].round(7).astype(str) + "|" + df["data_evento"].dt.strftime("%Y-%m-%d")
    df["chave_unidade_observacao"] = [hashlib.sha1(k.encode()).hexdigest()[:16] for k in key]
    df["e_unidade_duplicada"] = df.duplicated("chave_unidade_observacao", keep="first")
    return df


# ---------------------------------------------------------------------------
# 1. terreno
# ---------------------------------------------------------------------------
def _invalid(value: float, nodata) -> bool:
    return (not np.isfinite(value)) or value <= -32767 or (nodata is not None and value == nodata)


def extract_terrain(df: pd.DataFrame, raster_dir: Path) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    coords = list(zip(df["lon"].values, df["lat"].values))

    for feat, fname in RASTER_FILES.items():
        path = raster_dir / fname
        if not path.exists():
            raise SystemExit(f"raster ausente: {path}")
        values, statuses = [], []
        with rasterio.open(path) as src:
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            b = src.bounds
            nodata = src.nodata
            xs, ys = tr.transform(*zip(*coords))
            for x, y in zip(xs, ys):
                if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
                    values.append(np.nan)
                    statuses.append("FORA_DA_EXTENSAO_DO_RASTER")
                    continue
                v = float(next(src.sample([(x, y)]))[0])
                if _invalid(v, nodata):
                    values.append(np.nan)
                    statuses.append("NODATA_NO_PIXEL")
                else:
                    values.append(v)
                    statuses.append("OK")
        out[feat] = values
        out[f"{feat}_status"] = statuses
        print(f"  {feat:14s} OK={statuses.count('OK'):5d}  NODATA={statuses.count('NODATA_NO_PIXEL'):4d}  "
              f"FORA={statuses.count('FORA_DA_EXTENSAO_DO_RASTER'):4d}")
    return out


# ---------------------------------------------------------------------------
# 2. chuva
# ---------------------------------------------------------------------------
def _archive_request(lats, lons, start: date, end: date) -> list[dict]:
    """Requisicao (possivelmente multi-ponto) ao archive. Retorna uma entrada por
    coordenada, na ordem pedida."""
    url = (f"{ARCHIVE_URL}?latitude={','.join(f'{v}' for v in lats)}"
           f"&longitude={','.join(f'{v}' for v in lons)}"
           f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
           f"&daily=precipitation_sum&timezone={RAIN_TIMEZONE.replace('/', '%2F')}")
    # o archive cobra por localizacao, nao por requisicao: um lote de 100 pontos consome
    # 100 do limite por minuto. 429 nao e erro de dado -- espera e repete.
    r = requests.get(url, timeout=120)
    for attempt in range(5):
        if r.status_code != 429:
            break
        wait = 60 * (attempt + 1)
        print(f"      429 (limite de taxa) -- aguardando {wait}s e repetindo")
        time.sleep(wait)
        r = requests.get(url, timeout=120)
    r.raise_for_status()
    j = r.json()
    blocks = j if isinstance(j, list) else [j]
    out = []
    for b in blocks:
        daily = b.get("daily", {})
        out.append({
            "grid_lat": b.get("latitude"),
            "grid_lon": b.get("longitude"),
            "grid_elevation_m": b.get("elevation"),
            "series": {t: v for t, v in zip(daily.get("time", []), daily.get("precipitation_sum", []))},
        })
    return out


def fetch_daily_series(lat: float, lon: float, start: date, end: date) -> dict:
    return _archive_request([lat], [lon], start, end)[0]


def rain_window_features(series: dict, data_evento: date) -> dict:
    """Formula literal de baixar_chuva_positivos_lead_a.py (SUSC-20B)."""
    ordered = [series.get((data_evento - timedelta(days=k)).isoformat()) for k in range(LOOKBACK_DAYS, 0, -1)]
    arr = np.array([np.nan if v is None else float(v) for v in ordered], dtype=float)
    n_found = int(np.sum(~np.isnan(arr)))
    if n_found == 0:
        return {"rain_max_24h_chirps": np.nan, "rain_decay_index_api_chirps": np.nan,
                "rain_n_days_found": 0, "rain_status": "SEM_DIA_REAL_NA_JANELA"}
    rain_max = float(np.nanmax(arr))
    filled = np.where(np.isnan(arr), 0.0, arr)
    api = 0.0
    for idx, p in enumerate(filled):
        day_offset = LOOKBACK_DAYS - idx
        api += p * (DECAY_K ** (day_offset - 1))
    return {"rain_max_24h_chirps": round(rain_max, 2),
            "rain_decay_index_api_chirps": round(float(api), 3),
            "rain_n_days_found": n_found,
            "rain_status": "OK" if n_found == LOOKBACK_DAYS else "JANELA_PARCIAL"}


def extract_rain(df: pd.DataFrame, cache_dir: Path, batch: int = 100, sleep_s: float = 0.5) -> pd.DataFrame:
    """Duas passadas, para nao repetir a mesma serie centenas de vezes:

    A) descoberta de celula -- requisicao curta (2 dias) em lote pras 848 coordenadas
       unicas; a API devolve o centro da celula de grade que efetivamente atende cada
       coordenada. Coordenadas que caem na mesma celula compartilham exatamente a mesma
       serie de precipitacao.
    B) serie completa -- uma requisicao por CELULA distinta, cobrindo todo o span de
       datas necessario, feita no centro da celula.

    Duas propriedades tornam isso identico a pedir a janela isolada ponto a ponto (como
    o script de Recife faz): (i) a soma diaria de uma data nao depende do intervalo
    requisitado; (ii) o centro da celula resolve para ela mesma (verificado por assert).
    A cadeia inteira ainda e checada de ponta a ponta por `qa_rain_window_equivalence`,
    que refaz a requisicao isolada nas coordenadas ORIGINAIS de uma amostra.
    """
    cell_path = cache_dir / "rain_cell_map.json"
    series_path = cache_dir / "rain_cell_series.json"
    cell_map = json.loads(cell_path.read_text()) if cell_path.exists() else {}
    series_cache = json.loads(series_path.read_text()) if series_path.exists() else {}

    uniq = df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    span_start = df["data_evento"].min().date() - timedelta(days=LOOKBACK_DAYS)
    span_end = df["data_evento"].max().date() - timedelta(days=1)
    print(f"  coordenadas unicas={len(uniq)}  span={span_start}..{span_end}")

    # --- passada A: descoberta de celula ---
    pending = [(r.lat, r.lon) for r in uniq.itertuples() if f"{r.lat:.7f},{r.lon:.7f}" not in cell_map]
    for start in range(0, len(pending), batch):
        block = pending[start:start + batch]
        res = _archive_request([c[0] for c in block], [c[1] for c in block], span_start, span_start + timedelta(days=1))
        for (lat, lon), b in zip(block, res):
            cell_map[f"{lat:.7f},{lon:.7f}"] = {"grid_lat": b["grid_lat"], "grid_lon": b["grid_lon"],
                                                "grid_elevation_m": b["grid_elevation_m"]}
        cell_path.write_text(json.dumps(cell_map))
        print(f"    [A] celulas descobertas {min(start + batch, len(pending))}/{len(pending)}")
        time.sleep(sleep_s)

    cells = sorted({(v["grid_lat"], v["grid_lon"]) for v in cell_map.values()})
    print(f"  [A] {len(uniq)} coordenadas -> {len(cells)} celulas de grade distintas")

    # --- passada B: serie completa por celula ---
    for glat, glon in cells:
        ck = f"{glat},{glon}"
        cached = series_cache.get(ck)
        if cached:
            # o cache guarda a serie, nao o span pedido: uma rodada posterior com pontos mais
            # antigos (ou mais recentes) exige refetch, senao a janela sai truncada em silencio.
            dias = sorted(cached)
            if dias and dias[0] <= span_start.isoformat() and dias[-1] >= span_end.isoformat():
                continue
            print(f"    [B] cache da celula ({glat},{glon}) cobre {dias[0]}..{dias[-1]}, "
                  f"span exigido {span_start}..{span_end} -- refetch")
        b = fetch_daily_series(glat, glon, span_start, span_end)
        assert (b["grid_lat"], b["grid_lon"]) == (glat, glon), (
            f"centro de celula nao resolveu para si mesmo: pedido ({glat},{glon}), "
            f"devolvido ({b['grid_lat']},{b['grid_lon']})")
        series_cache[ck] = b["series"]
        series_path.write_text(json.dumps(series_cache))
        print(f"    [B] serie {len(series_cache)}/{len(cells)} celula ({glat},{glon}) dias={len(b['series'])}")
        time.sleep(sleep_s)

    rows = []
    for r in df.itertuples():
        cell = cell_map.get(f"{r.lat:.7f},{r.lon:.7f}")
        series = series_cache.get(f"{cell['grid_lat']},{cell['grid_lon']}") if cell else None
        if series is None:
            rows.append({"rain_max_24h_chirps": np.nan, "rain_decay_index_api_chirps": np.nan,
                         "rain_n_days_found": 0, "rain_status": "FALHA_API",
                         "rain_grid_lat": np.nan, "rain_grid_lon": np.nan})
            continue
        f = rain_window_features(series, r.data_evento.date())
        f["rain_grid_lat"] = cell["grid_lat"]
        f["rain_grid_lon"] = cell["grid_lon"]
        rows.append(f)
    out = pd.DataFrame(rows, index=df.index)
    out["rain_data_source"] = RAIN_SOURCE_LABEL
    return out


def qa_rain_window_equivalence(df: pd.DataFrame, feats: pd.DataFrame, cache_dir: Path,
                               n_sample: int = 20, seed: int = 20260731) -> dict:
    """Refaz a requisicao ISOLADA de 14 dias (exatamente como o script de Recife faz)
    para uma amostra e confere que bate com o valor recortado do span longo."""
    rng = np.random.default_rng(seed)
    ok_idx = feats.index[feats["rain_status"].isin(["OK", "JANELA_PARCIAL"])]
    sample = rng.choice(ok_idx, size=min(n_sample, len(ok_idx)), replace=False)
    mismatches = []
    for i in sample:
        r = df.loc[i]
        ev = r["data_evento"].date()
        s = fetch_daily_series(r["lat"], r["lon"], ev - timedelta(days=LOOKBACK_DAYS), ev - timedelta(days=1))
        direct = rain_window_features(s["series"], ev)
        for col in ("rain_max_24h_chirps", "rain_decay_index_api_chirps"):
            a, b = direct[col], feats.loc[i, col]
            if not (pd.isna(a) and pd.isna(b)) and abs(float(a) - float(b)) > 1e-6:
                mismatches.append({"idx": int(i), "col": col, "direct": a, "from_span": float(b)})
        time.sleep(0.25)
    report = {"n_sampled": int(len(sample)), "n_mismatches": len(mismatches), "mismatches": mismatches[:10]}
    (cache_dir / "qa_rain_window_equivalence.json").write_text(json.dumps(report, indent=2))
    print(f"  QA equivalencia janela: {len(sample)} amostrados, {len(mismatches)} divergencias")
    return report


# ---------------------------------------------------------------------------
# 3. mapbiomas
# ---------------------------------------------------------------------------
def extract_mapbiomas(df: pd.DataFrame, cache_dir: Path, chunk: int = 400) -> pd.DataFrame:
    import ee

    cache_path = cache_dir / "mapbiomas_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    uniq = df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    todo = [(r.lat, r.lon) for r in uniq.itertuples() if f"{r.lat:.7f},{r.lon:.7f}" not in cache]
    print(f"  coordenadas unicas={len(uniq)}  a consultar={len(todo)}")

    if todo:
        ee.Initialize()
        img = ee.Image(MAPBIOMAS_COLLECTION).select(MAPBIOMAS_BAND)
        for start in range(0, len(todo), chunk):
            block = todo[start:start + chunk]
            fc = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([lon, lat]), {"k": f"{lat:.7f},{lon:.7f}"})
                for lat, lon in block
            ])
            res = img.reduceRegions(fc, ee.Reducer.first(), MAPBIOMAS_SCALE).getInfo()
            for f in res["features"]:
                cache[f["properties"]["k"]] = f["properties"].get("first")
            cache_path.write_text(json.dumps(cache))
            print(f"    {min(start + chunk, len(todo))}/{len(todo)}")

    vals, status = [], []
    for r in df.itertuples():
        v = cache.get(f"{r.lat:.7f},{r.lon:.7f}")
        if v is None:
            vals.append(np.nan)
            status.append("NODATA_NO_PIXEL")
        else:
            vals.append(int(v))
            status.append("OK")
    out = pd.DataFrame({"mapbiomas_class_2023": vals, "mapbiomas_class_2023_status": status}, index=df.index)
    out["fonte_cobertura_solo"] = "mapbiomas_collection9_gee"
    print(f"  OK={status.count('OK')}  NODATA={status.count('NODATA_NO_PIXEL')}")
    return out


# ---------------------------------------------------------------------------
# 4. ortogonalizacao (logica identica a montar_dataset_v12.py, SUSC-20B)
# ---------------------------------------------------------------------------
def orthogonalize_single(d: pd.DataFrame):
    x = d["rain_decay_index_api_chirps"].values.reshape(-1, 1)
    y = d["rain_max_24h_chirps"].values
    xm = x.mean()
    xc = x - xm
    beta = float((xc.flatten() @ y) / (xc.flatten() @ xc.flatten()))
    intercept = float(y.mean() - beta * xm)
    pred = intercept + beta * d["rain_decay_index_api_chirps"].values
    return y - pred, beta, intercept


def orthogonalize_per_source(df: pd.DataFrame):
    df = df.copy()
    df["rain_peak_residual_orthogonalized"] = np.nan
    reports = {}
    subset = df.dropna(subset=["rain_max_24h_chirps", "rain_decay_index_api_chirps"])
    for src, sub in subset.groupby("rain_data_source"):
        resid, beta, intercept = orthogonalize_single(sub)
        df.loc[sub.index, "rain_peak_residual_orthogonalized"] = resid
        corr_before = float(sub["rain_max_24h_chirps"].corr(sub["rain_decay_index_api_chirps"]))
        corr_after = float(pd.Series(resid, index=sub.index).corr(sub["rain_decay_index_api_chirps"]))
        reports[src] = {"n_used": int(len(sub)), "beta": round(beta, 4), "intercept": round(intercept, 4),
                        "pearson_r_before_orthogonalization": round(corr_before, 4),
                        "pearson_r_after_orthogonalization": round(corr_after, 6)}
    return df, reports


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raster-dir", default=str(DEFAULT_RASTER_DIR))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--pos-csv", default=str(POS_CSV))
    p.add_argument("--neg-csv", default=str(NEG_CSV))
    p.add_argument("--out", default=str(OUT_CSV))
    p.add_argument("--skip-rain", action="store_true")
    p.add_argument("--skip-mapbiomas", action="store_true")
    p.add_argument("--skip-qa-rain", action="store_true")
    args = p.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = load_points(Path(args.pos_csv), Path(args.neg_csv))
    print(f"[0] pontos carregados: {len(df)} linhas "
          f"({int((df.rotulo == 1).sum())} pos / {int((df.rotulo == 0).sum())} neg); "
          f"unidades de observacao unicas (lat,lon,data): {df['chave_unidade_observacao'].nunique()}")

    print("[1] terreno (amostragem de raster SUSC-20G)")
    df = pd.concat([df, extract_terrain(df, Path(args.raster_dir))], axis=1)

    qa_rain = None
    if args.skip_rain:
        print("[2] chuva PULADA (--skip-rain)")
    else:
        print("[2] chuva (Open-Meteo ERA5-Land archive)")
        rain = extract_rain(df, cache_dir)
        df = pd.concat([df, rain], axis=1)
        if not args.skip_qa_rain:
            qa_rain = qa_rain_window_equivalence(df, rain, cache_dir)

    if args.skip_mapbiomas:
        print("[3] mapbiomas PULADO (--skip-mapbiomas)")
    else:
        print("[3] mapbiomas (Colecao 9, classification_2023, 30m)")
        df = pd.concat([df, extract_mapbiomas(df, cache_dir)], axis=1)

    ortho_report = {}
    if "rain_max_24h_chirps" in df.columns:
        print("[4] ortogonalizacao de chuva por fonte")
        df, ortho_report = orthogonalize_per_source(df)
        print(json.dumps(ortho_report, indent=2))

    df["data_evento"] = df["data_evento"].dt.strftime("%Y-%m-%d")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[5] escrito {args.out}  shape={df.shape}")

    prov = {
        "stage": "SUSC-20L",
        "n_rows": int(len(df)),
        "n_positives": int((df.rotulo == 1).sum()),
        "n_negatives": int((df.rotulo == 0).sum()),
        "n_observation_units": int(df["chave_unidade_observacao"].nunique()),
        "n_observation_units_positive": int(df.loc[df.rotulo == 1, "chave_unidade_observacao"].nunique()),
        "n_observation_units_negative": int(df.loc[df.rotulo == 0, "chave_unidade_observacao"].nunique()),
        "terrain": {
            "raster_dir_role": "SUSC-20G curitiba D-infinity (MDT SGB/CPRM 10m, EPSG:31982)",
            "files": RASTER_FILES,
            "status_counts": {f: df[f"{f}_status"].value_counts().to_dict() for f in RASTER_FILES},
        },
        "rain": {
            "source": RAIN_SOURCE_LABEL,
            "api": ARCHIVE_URL,
            "timezone": RAIN_TIMEZONE,
            "lookback_days": LOOKBACK_DAYS,
            "decay_k": DECAY_K,
            "window_rule": "14 dias terminando no dia anterior ao data_evento (dia do evento excluido)",
            "status_counts": df["rain_status"].value_counts().to_dict() if "rain_status" in df else None,
            "n_days_found_counts": df["rain_n_days_found"].value_counts().sort_index().to_dict()
            if "rain_n_days_found" in df else None,
            "qa_window_equivalence": qa_rain,
        },
        "mapbiomas": {
            "collection": MAPBIOMAS_COLLECTION, "band": MAPBIOMAS_BAND, "scale_m": MAPBIOMAS_SCALE,
            "class_counts": df["mapbiomas_class_2023"].value_counts(dropna=False).to_dict()
            if "mapbiomas_class_2023" in df else None,
        },
        "orthogonalization_per_source": ortho_report,
    }
    prov_path = cache_dir / "v20l_provenance_and_qa.json"
    prov_path.write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    print(f"[6] proveniencia/QA -> {prov_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
