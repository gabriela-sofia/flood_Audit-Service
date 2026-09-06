"""
CHUVA-02 -- Padroniza os 278 pontos de Recife numa unica fonte de chuva.

DE ONDE VEIO ISTO:
`aud_chuva01_fontes_incompativeis.py` mediu, nao supos: a coluna de chuva de
Recife carrega duas fontes na mesma coluna, 181 pontos em CHIRPS v2.0 e 97 em
Open-Meteo/ERA5-Land, e o indicador de fonte tem AUC de rank (0,826) maior que
a propria chuva (0,738) contra o rotulo. Como a chuva antecedente e o
preditor principal da trilha pluvial (coeficiente +0,9896, p < 1e-4, contra
HAND em -0,0001, p = 0,978), parte do que aquele coeficiente mede nao e chuva:
e proveniencia. O proprio audit script diz explicitamente o que NAO faz:
"reamostrar CHIRPS para os 97 pontos de ERA5-Land e aquisicao de dado, nao
auditoria". Este script e essa aquisicao -- na direcao inversa.

POR QUE A DIRECAO E OPEN-METEO PARA TODOS, E NAO CHIRPS PARA TODOS:
as duas direcoes tem custo, de natureza diferente.
  - CHIRPS para os 97: o servidor da CHC (data.chc.ucsb.edu) tem protecao
    documentada contra scraping -- `aq_chirps3_v3.py` registra uma tentativa
    de 163 minutos com quase zero sucesso antes de existir cache por dia. Os
    97 pontos cobrem 85 datas distintas entre 2015 e 2022 (nao sao 2 eventos
    concentrados), o que exigiria uma campanha dia-a-dia sem garantia de
    terminar.
  - Open-Meteo para os 181: a mesma API ja foi usada com sucesso 3 vezes
    neste projeto (v8, v9 bairro-matched, v12 lead A/C) sem historico de
    bloqueio, com uma requisicao por PONTO (nao por dia), entao 181 pontos e
    181 requisicoes -- minutos, nao horas.
Decisao tomada com a Gabriela em 2026-08-16: Open-Meteo/ERA5-Land para os 278.

O QUE ISTO NAO RESOLVE, declarado: ERA5-Land e reanalise (modelo), nao produto
com estacao real como o CHIRPS gauge-blend. Perde-se a validacao por
pluviometro que a maioria dos pontos tinha. Isso e custo, nao e escondido.

METODO, identico ao ja usado nas fontes Open-Meteo existentes (mesma janela,
mesma formula, para nao introduzir uma SEGUNDA inconsistencia dentro da fonte
unica): janela de 14 dias antes do evento, indice de decaimento com fator
0,85/dia, maximo de chuva em 24h dentro da janela.

O QUE ISTO NAO FAZ: nao toca PROJETO (so le point_id/lat/lon/event_date de
`dataset_v12_final.csv`, read-only); nao decide sobre o v12 em si; nao reroda
ds04/ds05/mod_mec02 -- isso e o proximo passo, separado.

Uso:
    python scripts/treino/chuva02_padronizar_fonte_unica_recife.py
"""

from __future__ import annotations

import json
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
    DECAY_K, LOOKBACK_DAYS, METHOD_VERSION, PRODUCT_ID, PROVIDER_ID, TIMEZONE,
    parse_event_dates, requested_window,
)
from chuva.providers import OpenMeteoEra5LandProvider, RainRequest  # noqa: E402
PROJETO = REPO.parent / "PROJETO"
RUNS = REPO / "local_runs"

V12 = PROJETO / "local_runs" / "recife_modelo_v12_extracao_final" / "dataset_v12_final.csv"
HARM = RUNS / "ter-03-brasil-harmonizado" / "recife_harmonizado.csv"
OUT = RUNS / "chuva-02-fonte-unica-recife"

FONTE_ALVO = "open_meteo_era5_land_archive_api"
FONTE_ORIGEM_A_TROCAR = "chirps_v2.0_gauge_satellite_blend"

LOOKBACK = LOOKBACK_DAYS
# concorrencia moderada -- o mesmo padrao ja usado com sucesso em
# fetch_rain_v9_bairro_matched.py (PROJETO), 6 workers, sem historico de
# bloqueio nesta API neste projeto
N_WORKERS = 4


def fetch_um(lat: float, lon: float, event_date: pd.Timestamp,
             *, online: bool = False) -> dict:
    if pd.isna(event_date):
        return {"rain_max_24h": np.nan, "rain_decay_index_api": np.nan,
                "n_dias_encontrados": 0, "erro": "event_date ausente na origem"}
    start, end = requested_window(str(event_date.date()))
    provider = OpenMeteoEra5LandProvider(OUT / "provider-cache")
    try:
        acquisition = provider.acquire(RainRequest(
            float(lat), float(lon), start, end, TIMEZONE), online=online)
        derived = provider.derive(acquisition)
    except Exception as exc:  # noqa: BLE001
        return {"rain_max_24h": np.nan, "rain_decay_index_api": np.nan,
                "n_dias_encontrados": 0, "erro": str(exc)}
    return {"rain_max_24h": round(derived["rain_max_24h"], 2),
            "rain_decay_index_api": round(derived["rain_decay_index"], 3),
            "n_dias_encontrados": derived["days_in_window"], "erro": None,
            "payload_sha256": derived["raw_sha256"],
            "method_version": derived["method_version"],
            "cache_mode": derived["cache_mode"]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("===CHUVA02_PADRONIZAR_FONTE_UNICA_RECIFE===")

    if not V12.exists():
        print(f"ABORTADO: {V12} ausente.")
        return 1
    if not HARM.exists():
        print(f"ABORTADO: {HARM} ausente. Rode ter03_reextrair_brasil.py.")
        return 1

    online = "--online" in sys.argv[1:]
    v12 = pd.read_csv(V12, low_memory=False)
    v12["event_date"] = parse_event_dates(v12["event_date"])
    alvo = v12[v12["rain_data_source"] == FONTE_ORIGEM_A_TROCAR].copy()
    print(f"PONTOS_V12={len(v12)} A_REAMOSTRAR={len(alvo)} "
          f"(hoje em '{FONTE_ORIGEM_A_TROCAR}')")

    t0 = time.time()
    legado = OUT / "reamostragem_open_meteo_181pts.csv"
    if not online and legado.exists():
        novo = pd.read_csv(legado, low_memory=False)
        novo["cache_mode"] = "legacy_derived_replay"
        novo["method_version"] = METHOD_VERSION
        print(f"REPLAY_OFFLINE={legado} n={len(novo)}")
    else:
        linhas = []
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(fetch_um, r.lat, r.lon, r.event_date,
                              online=online): r.point_id for r in alvo.itertuples()}
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                res["point_id"] = futs[fut]
                linhas.append(res)
                if i % 30 == 0:
                    print(f"  ... {i}/{len(alvo)}  ({time.time()-t0:.0f}s)", flush=True)
        novo = pd.DataFrame(linhas)
    falhas = novo[novo["rain_max_24h"].isna()]
    print(f"REAMOSTRADOS={len(novo)}  FALHAS={len(falhas)}  "
          f"tempo={(time.time()-t0):.0f}s")
    if len(falhas):
        print("  pontos sem dado (ficam NaN, nao estimados):",
              falhas["point_id"].tolist())

    # ---- aplica no harmonizado: sobrescreve so os 181, mantem os 97 como estao ----
    h = pd.read_csv(HARM, low_memory=False)
    antes = h["rain_data_source"].value_counts().to_dict()

    m = h["point_id"].isin(novo["point_id"])
    mapa_max = dict(zip(novo["point_id"], novo["rain_max_24h"]))
    mapa_api = dict(zip(novo["point_id"], novo["rain_decay_index_api"]))
    h.loc[m, "rain_max_24h_chirps"] = h.loc[m, "point_id"].map(mapa_max)
    h.loc[m, "rain_decay_index_api_chirps"] = h.loc[m, "point_id"].map(mapa_api)
    h.loc[m, "rain_data_source"] = FONTE_ALVO

    depois = h["rain_data_source"].value_counts().to_dict()
    print(f"\nFONTE_CHUVA antes={antes}")
    print(f"FONTE_CHUVA depois={depois}")

    ainda_sem = h[h["rain_max_24h_chirps"].isna()]
    print(f"AINDA_SEM_CHUVA={len(ainda_sem)} de {len(h)} "
          f"(NaN declarado, nao preenchido)")

    harmonizado_com_chuva = OUT / "recife_harmonizado_com_chuva.csv"
    h.to_csv(harmonizado_com_chuva, index=False)
    novo.to_csv(OUT / "reamostragem_open_meteo_181pts.csv", index=False)
    (OUT / "resumo.json").write_text(json.dumps({
        "fonte_alvo": FONTE_ALVO,
        "pontos_reamostrados": int(len(novo)),
        "falhas": int(len(falhas)),
        "point_id_falhas": falhas["point_id"].tolist(),
        "fonte_chuva_antes": antes,
        "fonte_chuva_depois": depois,
        "ainda_sem_chuva": int(len(ainda_sem)),
        "metodo": {"lookback_dias": LOOKBACK, "decay_k": DECAY_K,
                   "formula": "rain_max_24h = max(precip diario na janela); "
                              "rain_decay_index_api = soma(precip_i * "
                              "decay_k**(dias_antes_do_evento_i - 1))"},
        "entrada_harmonizada": str(HARM),
        "arquivo_derivado": str(harmonizado_com_chuva),
        "provider": PROVIDER_ID, "product": PRODUCT_ID,
        "timezone": TIMEZONE, "method_version": METHOD_VERSION,
        "modo": "online" if online else "replay_offline",
        "ordem_pipeline": "ter03 -> chuva02 -> ds04; chuva02 nunca sobrescreve ter03",
        "nao_toca": "PROJETO e read-only; so leu point_id/lat/lon/event_date",
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nGRAVADO={harmonizado_com_chuva} (derivado; entrada preservada)")
    print(f"GRAVADO={OUT}/reamostragem_open_meteo_181pts.csv")
    print("===END===")
    return 0 if len(falhas) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
