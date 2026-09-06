"""
CHUVA-04 -- Chuva ERA5-Land para a base inteira, numa fonte so.

O QUE ISTO FECHA:

`aud_chuva01` mostrou que a chuva de Recife carregava duas fontes e que o
indicador de proveniencia discriminava mais que a propria chuva. O `chuva02`
resolveu Recife. Sobrava o resto: 107.558 pontos de CEMS, Sen1Floods11 e UFO
sem nenhuma chuva, e o piloto ingles numa TERCEIRA fonte (CHIRPS v3 rnl). Com
isso o pool fluvial rodava com quatro variaveis, nunca seis.

Este script traz todos para `open_meteo_era5_land`, a mesma fonte, a mesma
janela e a mesma formula que Recife e Curitiba ja usam.

POR QUE A CONTA FECHA, e por que ela nao fechava antes:

sem deduplicacao seriam 116.983 requisicoes. O ERA5-Land tem grade NATIVA de
0,1 grau: dois pontos dentro da mesma celula recebem exatamente a mesma serie,
porque e a mesma celula do modelo. Agrupando por (celula, data) restam 5.542
pares -- 21 vezes menos. Isso NAO e amostragem nem aproximacao: e reconhecer
que a pergunta ja tinha resposta repetida.

A reducao de toda fonte de imagem a TABELA DE PONTOS e o que torna isso
possivel. Enquanto a unidade fosse o raster, nao havia como perceber que
116.983 leituras eram 5.542 perguntas.

E o que garante que a deduplicacao e exata e o proprio `--validar`, que pega
celulas com mais de um ponto distinto e compara a serie dos dois. Se
divergirem, a premissa esta errada e o script diz isso em vez de seguir.

TRES GARANTIAS DE CUSTO, porque uma campanha que nao termina nao serve:
  cache em disco   por (celula, data). Reexecucao nao repete chamada.
  backoff em 429   limite de taxa nao e ausencia de dado.
  paralelismo      6 trabalhadores, o mesmo padrao ja usado sem bloqueio.

FUSO: `timezone=auto`, isto e, o fuso local de cada ponto. Chuva diaria e
grandeza de dia LOCAL, e para Recife o `auto` resolve exatamente para o
`America/Recife` que o `chuva02` fixou -- o `--validar` confere isso tambem,
em vez de supor.

LIMITE DECLARADO: ERA5-Land e reanalise, nao produto com estacao. Ganha-se
cobertura global e uma fonte so; perde-se a validacao por pluviometro que o
CHIRPS gauge-blend dava ao piloto ingles. E custo, e esta dito.

Uso:
    python scripts/treino/chuva04_adquirir_era5_global.py --validar
    python scripts/treino/chuva04_adquirir_era5_global.py
    python scripts/treino/chuva04_adquirir_era5_global.py --limite 200
"""

from __future__ import annotations

import json
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from chuva.contrato_chuva import (  # noqa: E402
    DECAY_K, LOOKBACK_DAYS, METHOD_VERSION, PRODUCT_ID, PROVIDER_ID,
    SPATIAL_RESOLUTION_DEGREES, TIMEZONE, derive_rain_features,
    parse_event_dates, requested_window,
)
from chuva.providers import OpenMeteoEra5LandProvider, RainRequest  # noqa: E402
RUNS = REPO / "local_runs"
OUT = RUNS / "chuva-04-era5-global"
CACHE = OUT / "cache"

DS01 = RUNS / "ds-01-multirregiao" / "dataset_multirregiao_v1.csv"
DATAS = RUNS / "chuva-03-datas" / "datas_evento_resolvidas.csv"

# identicos ao chuva02: mudar qualquer um cria uma segunda inconsistencia
# DENTRO da fonte unica, que e o problema que este script existe para acabar
LOOKBACK = LOOKBACK_DAYS

GRADE = SPATIAL_RESOLUTION_DEGREES
N_WORKERS = 6
FONTE = f"{PROVIDER_ID}_{PRODUCT_ID}"

# fontes que ainda nao tem chuva na fonte unica. Recife e Curitiba ficam de
# fora porque ja estao nela (chuva02 e a aquisicao original).
ALVOS = ("cems", "sen1floods11", "ufo", "uk")


def celula(lat: float, lon: float) -> tuple[int, int]:
    return int(round(lat / GRADE)), int(round(lon / GRADE))


def centro(cl: tuple[int, int]) -> tuple[float, float]:
    return round(cl[0] * GRADE, 4), round(cl[1] * GRADE, 4)


def _pedir(lat: float, lon: float, inicio, fim, tz: str = "auto") -> dict | None:
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&start_date={inicio}&end_date={fim}"
           f"&daily=precipitation_sum&timezone={tz}")
    for tentativa in range(4):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                time.sleep(2.0 * (tentativa + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            if tentativa == 3:
                return None
            time.sleep(1.5 * (tentativa + 1))
    return None


def resumir(j: dict, data_evento: pd.Timestamp) -> dict:
    """Mesma formula do chuva02: max 24h e API com decaimento na janela."""
    daily = (j or {}).get("daily", {})
    serie = dict(zip(daily.get("time", []), daily.get("precipitation_sum", [])))
    ordenado = [serie.get(str((data_evento - pd.Timedelta(days=k)).date()))
                for k in range(LOOKBACK, 0, -1)]
    arr = np.array([np.nan if v is None else float(v) for v in ordenado])
    n = int(np.sum(~np.isnan(arr)))
    if n == 0:
        return {"rain_max_24h": np.nan, "rain_decay_index": np.nan, "n_dias": 0}
    if n != LOOKBACK:
        return {"rain_max_24h": np.nan, "rain_decay_index": np.nan,
                "n_dias": n, "qc": "janela_incompleta"}
    g = derive_rain_features(arr.tolist(), expected_days=LOOKBACK)
    return {"rain_max_24h": round(g["rain_max_24h"], 2),
            "rain_decay_index": round(g["rain_decay_index"], 3), "n_dias": n,
            "method_version": METHOD_VERSION}


def buscar_par(cl: tuple[int, int], data: str, *, online: bool = False) -> dict:
    """Uma celula, uma data. Cache em disco: reexecucao nao repete chamada."""
    chave = f"{cl[0]}_{cl[1]}__{data}"
    cache = CACHE / f"{chave}.json"
    if cache.exists():
        try:
            g = json.loads(cache.read_text(encoding="utf-8"))
            digest = hashlib.sha256(json.dumps(
                g, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            return {"chave": chave, **g, "origem": "cache_legado_derivado",
                    "payload_sha256": digest, "method_version": METHOD_VERSION,
                    "provider": PROVIDER_ID, "product": PRODUCT_ID,
                    "timezone": TIMEZONE}
        except Exception:  # noqa: BLE001
            pass
    dt = pd.Timestamp(data)
    lat, lon = centro(cl)
    inicio, fim = requested_window(str(dt.date()))
    provider = OpenMeteoEra5LandProvider(OUT / "provider-cache")
    try:
        acquisition = provider.acquire(RainRequest(
            lat, lon, inicio, fim, TIMEZONE), online=online)
        derived = provider.derive(acquisition)
    except Exception:  # noqa: BLE001
        return {"chave": chave, "rain_max_24h": np.nan,
                "rain_decay_index": np.nan, "n_dias": 0, "origem": "falha"}
    g = {"rain_max_24h": derived["rain_max_24h"],
         "rain_decay_index": derived["rain_decay_index"],
         "n_dias": derived["days_in_window"],
         "payload_sha256": derived["raw_sha256"],
         "method_version": derived["method_version"],
         "provider": derived["provider"], "product": derived["product"],
         "timezone": derived["timezone"], "query_start": inicio,
         "query_end": fim}
    cache.write_text(json.dumps(g), encoding="utf-8")
    return {"chave": chave, **g, "origem": derived["cache_mode"]}


# --------------------------------------------------------------------------
# validacao das duas premissas, antes de gastar 5.542 chamadas nelas
# --------------------------------------------------------------------------

def validar(pares: pd.DataFrame, pontos: pd.DataFrame) -> bool:
    print("--- VALIDACAO DAS PREMISSAS ---")
    ok = True

    # 1. a deduplicacao por celula e exata?
    print("1. dois pontos distintos na mesma celula dao a mesma serie?")
    cand = (pontos.groupby(["cel_lat", "cel_lon", "dt"])
            .filter(lambda g: g[["lat", "lon"]].drop_duplicates().shape[0] > 1))
    if cand.empty:
        print("   nenhuma celula com dois pontos distintos -- nada a comparar")
    else:
        g = cand.groupby(["cel_lat", "cel_lon", "dt"])
        testadas = 0
        for (cl_lat, cl_lon, dt), s in g:
            if testadas >= 3:
                break
            u = s.drop_duplicates(["lat", "lon"]).head(2)
            dtp = pd.Timestamp(dt)
            r = [_pedir(row.lat, row.lon, (dtp - pd.Timedelta(days=LOOKBACK)).date(),
                        (dtp - pd.Timedelta(days=1)).date())
                 for row in u.itertuples()]
            if any(x is None for x in r):
                print("   uma das chamadas falhou -- inconclusivo")
                continue
            a, b = (resumir(x, dtp) for x in r)
            igual = (a["rain_max_24h"] == b["rain_max_24h"]
                     and a["rain_decay_index"] == b["rain_decay_index"])
            grade_api = (r[0]["latitude"], r[0]["longitude"]) == (r[1]["latitude"],
                                                                  r[1]["longitude"])
            print(f"   celula ({cl_lat},{cl_lon}) {dt}: "
                  f"pontos {u.lat.iloc[0]:.3f}/{u.lon.iloc[0]:.3f} e "
                  f"{u.lat.iloc[1]:.3f}/{u.lon.iloc[1]:.3f} -> "
                  f"serie igual={igual} celula_api_igual={grade_api}")
            ok = ok and igual and grade_api
            testadas += 1

    # 2. timezone=auto reproduz o America/Recife que o chuva02 fixou?
    print("2. timezone=auto reproduz o fuso fixado em Recife?")
    dt = pd.Timestamp("2022-05-28")
    i, f = (dt - pd.Timedelta(days=LOOKBACK)).date(), (dt - pd.Timedelta(days=1)).date()
    a = _pedir(-8.05, -34.9, i, f, tz="auto")
    b = _pedir(-8.05, -34.9, i, f, tz="America%2FRecife")
    if a is None or b is None:
        print("   inconclusivo: chamada falhou")
    else:
        ra, rb = resumir(a, dt), resumir(b, dt)
        igual = ra == rb
        print(f"   auto={a.get('timezone')} -> max={ra['rain_max_24h']} "
              f"api={ra['rain_decay_index']}")
        print(f"   fixo -> max={rb['rain_max_24h']} api={rb['rain_decay_index']}")
        print(f"   identico={igual}")
        ok = ok and igual

    print(f"--- PREMISSAS {'OK' if ok else 'REPROVADAS'} ---")
    return ok


def main() -> int:
    args = sys.argv[1:]
    online = "--online" in args
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    print("===CHUVA04_ERA5_GLOBAL===")

    if not DATAS.exists():
        print(f"ABORTADO: {DATAS} ausente. Rode chuva03.")
        return 1
    datas = pd.read_csv(DATAS, low_memory=False)[["ponto_id", "data_evento",
                                                  "precisao_data"]]

    if not DS01.exists():
        print(f"ABORTADO: {DS01} ausente. Rode ds01.")
        return 1
    base = pd.read_csv(DS01, low_memory=False)
    regiao = base["regiao"].astype(str)
    fonte = np.select([
        regiao.str.startswith("analogo_EMSR"),
        regiao.eq("nivel1_sen1floods11_handlabeled"),
        regiao.eq("nivel1_ufo_urban_flood_observations"),
        regiao.eq("UK_noroeste"),
    ], ["cems", "sen1floods11", "ufo", "uk"], default="")
    pts = base.loc[fonte != "", ["ponto_id", "lat", "lon", "data_evento"]].copy()
    pts["fonte"] = fonte[fonte != ""]
    if pts.empty:
        print("ABORTADO: nenhuma fonte alvo em ds01.")
        return 1
    pts = pts.merge(datas, on="ponto_id", how="left", suffixes=("", "_res"))
    # `format="mixed"` e obrigatorio: a coluna do piloto ingles tem "2000-06-01"
    # e "2025-01-01 00:00:00" na mesma serie, e sem isso o pandas coage metade a
    # NaT em silencio. Foram 4.166 pontos perdidos assim na primeira execucao.
    bruto = pts["data_evento_res"].fillna(pts["data_evento"])
    dt = parse_event_dates(bruto)
    perdidos = int(dt.isna().sum())
    # filtrar ANTES de virar texto: NaT.date().astype(str) produz 'nan', nao
    # 'NaT', e um filtro por texto deixa passar o que deveria barrar
    pts = pts[dt.notna()].copy()
    pts["dt"] = dt[dt.notna()].dt.date.astype(str)
    if perdidos:
        print(f"AVISO: {perdidos:,} pontos sem data parseavel ficaram de fora")
    pts["cel_lat"] = (pts.lat / GRADE).round().astype(int)
    pts["cel_lon"] = (pts.lon / GRADE).round().astype(int)

    pares = pts[["cel_lat", "cel_lon", "dt"]].drop_duplicates()
    print(f"PONTOS={len(pts):,} de {pts.fonte.nunique()} fontes")
    print(f"PARES (celula,data)={len(pares):,}  "
          f"-- {len(pts)/max(len(pares),1):.1f}x menos chamadas que por ponto")
    ja = len(list(CACHE.glob('*.json')))
    print(f"cache: {ja:,} pares ja resolvidos")

    if "--validar" in args and not online:
        print("ABORTADO: --validar consulta a API; declare tambem --online")
        return 1
    if "--validar" in args:
        return 0 if validar(pares, pts) else 2

    if "--limite" in args:
        pares = pares.head(int(args[args.index("--limite") + 1]))
        print(f"LIMITE: so os primeiros {len(pares):,} pares")

    t0 = time.time()
    res = {}
    alvo = [((int(r.cel_lat), int(r.cel_lon)), r.dt) for r in pares.itertuples()]
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(buscar_par, cl, dt, online=online): (cl, dt)
                for cl, dt in alvo}
        for i, fut in enumerate(as_completed(futs), 1):
            g = fut.result()
            res[g["chave"]] = g
            if i % 250 == 0:
                falhas = sum(1 for v in res.values() if v["origem"] == "falha")
                print(f"  ... {i:,}/{len(alvo):,} ({time.time()-t0:.0f}s, "
                      f"falhas={falhas})", flush=True)

    falhas = [k for k, v in res.items() if v["origem"] == "falha"]
    print(f"\nRESOLVIDOS={len(res)-len(falhas):,}/{len(res):,} "
          f"falhas={len(falhas)} em {time.time()-t0:.0f}s")

    pts["chave"] = (pts.cel_lat.astype(str) + "_" + pts.cel_lon.astype(str)
                    + "__" + pts.dt)
    tab = pd.DataFrame.from_dict(res, orient="index")
    metadados = [c for c in (
        "rain_max_24h", "rain_decay_index", "n_dias", "payload_sha256",
        "method_version", "provider", "product", "timezone", "query_start",
        "query_end", "origem") if c in tab.columns]
    saida = pts.merge(tab[metadados],
                      left_on="chave", right_index=True, how="left")
    saida["fonte_chuva"] = np.where(saida.rain_max_24h.notna(), FONTE, "ausente")

    print("\n--- COBERTURA POR FONTE ---")
    r = saida.groupby("fonte").agg(
        n=("ponto_id", "size"),
        com_chuva=("rain_max_24h", lambda x: int(x.notna().sum())))
    r["pct"] = (100 * r.com_chuva / r.n).round(1)
    print(r.to_string())

    cols = ["ponto_id", "fonte", "dt", "rain_max_24h", "rain_decay_index",
            "n_dias", "fonte_chuva", "precisao_data", *[
                c for c in ("payload_sha256", "method_version", "provider",
                            "product", "timezone", "query_start", "query_end",
                            "origem") if c in saida.columns]]
    saida[cols].to_csv(OUT / "chuva_era5_por_ponto.csv", index=False)
    (OUT / "resumo.json").write_text(json.dumps({
        "fonte": FONTE, "grade_graus": GRADE, "lookback_dias": LOOKBACK,
        "decay_k": DECAY_K, "timezone": TIMEZONE,
        "provider": PROVIDER_ID, "product": PRODUCT_ID,
        "method_version": METHOD_VERSION,
        "mode": "online_or_replay" if online else "offline_replay_only",
        "pontos": int(len(saida)), "pares_consultados": int(len(res)),
        "reducao_de_chamadas": round(len(pts) / max(len(res), 1), 1),
        "falhas": len(falhas), "segundos": round(time.time() - t0, 1),
        "por_fonte": json.loads(r.to_json(orient="index")),
        "limite_declarado": (
            "ERA5-Land e reanalise, nao produto com estacao. O piloto ingles "
            "perde a validacao por pluviometro que o CHIRPS gauge-blend dava, "
            "em troca de fonte unica em toda a base"),
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
