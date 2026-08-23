"""
AQ-RS-01 -- Camada de features para a quarta regiao do projeto: Rio Grande do Sul.

O RS entra como regiao do REV-P por uma razao que nenhuma das outras tres
oferece: e a unica com NEGATIVO FORMAL POR OBSERVACAO. O Copernicus EMS
EMSR720 declara a area analisada (257,26 km2, 100% coberta por imagem),
a inundacao detectada dentro dela (36,44 km2) e, por diferenca, os
216,55 km2 que foram olhados e nao estavam inundados -- com o deslizamento
(4,27 km2) removido das duas classes.

Recife, Curitiba e Petropolis nao tem equivalente: a consulta ao portal CEMS
filtrada por Brasil (2026-08-07) devolveu apenas quatro ativacoes no pais, e
nenhuma delas cobre essas tres cidades.

AOI (derivada do proprio EMSR720, nao escolhida a mao):
    WGS84   lon -51,9233 a -51,5980   lat -29,3254 a -29,0290
    UTM22S  410.351 / 6.755.630  a  441.820 / 6.788.646
    extensao 31,5 x 33,0 km -- vale do Rio dos Sinos

VANTAGEM OPERACIONAL: a AOI cabe num unico tile de 1 grau. A aquisicao inteira
e uma fracao da inglesa.

DIFERENCAS EM RELACAO A INGLATERRA que precisam ficar registradas:
  - CRS metrico: EPSG:31982 (SIRGAS 2000 / UTM 22S), nao EPSG:27700.
  - CHIRPS: o RS esta dentro de 50S-50N, entao aqui o v2.0 do dataset
    brasileiro TAMBEM funcionaria. Mesmo assim usamos v3 para manter a
    comparabilidade com o piloto ingles -- e a diferenca de versao entre
    regioes fica declarada em vez de escondida.
  - Cobertura do solo: alem do ESA WorldCover, existe o EMSN132 (Copernicus
    Exposure Mapping / GHSL) cobrindo o Brasil, que pode substituir ou
    complementar na condicao N4.

NAO faz: nao amostra ponto, nao extrai feature, nao treina. So adquire.

Uso:
    python scripts/externo/aq_rs01_features.py --sondar
    python scripts/externo/aq_rs01_features.py --baixar
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "local_runs" / "aq-rs-features"

AOI_LL = (-51.9233, -29.3254, -51.5980, -29.0290)   # lon0, lat0, lon1, lat1
AOI_UTM = (410_351, 6_755_630, 441_820, 6_788_646)
CRS_METRICO = 31982

# Janela de chuva: o evento do EMSR720 e de 2024-05-02, mas a cheia se
# desenvolve desde o fim de abril. Cobrimos 2024-04-10 a 2024-05-20, o que ja
# inclui os 14 dias anteriores exigidos pelo indice de decaimento.
DATA_INI = dt.date(2024, 4, 10)
DATA_FIM = dt.date(2024, 5, 20)

WORKERS = 6
TIMEOUT = 300
BBOX_CHUVA = (-52.1, -29.5, -51.4, -28.8)   # com folga


def mb(n) -> float:
    return round((n or 0) / 1024 ** 2, 1)


def sondar(url: str) -> dict:
    try:
        r = requests.head(url, timeout=60, allow_redirects=True)
        if r.status_code >= 400 or "Content-Length" not in r.headers:
            r = requests.get(url, timeout=60, stream=True,
                             headers={"Range": "bytes=0-1"})
        tam = r.headers.get("Content-Length")
        cr = r.headers.get("Content-Range", "")
        if cr and "/" in cr:
            tam = cr.split("/")[-1]
        return {"ok": r.status_code < 400, "status": r.status_code,
                "bytes": int(tam) if tam and str(tam).isdigit() else None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": type(exc).__name__, "bytes": None}


def baixar_um(par: tuple[str, Path]) -> tuple[str, bool, int]:
    url, destino = par
    if destino.exists() and destino.stat().st_size > 0:
        return url, True, destino.stat().st_size
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".part")
    for tentativa in range(4):
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                with tmp.open("wb") as fh:
                    for ch in r.iter_content(chunk_size=1 << 20):
                        fh.write(ch)
            tmp.replace(destino)
            return url, True, destino.stat().st_size
        except Exception:  # noqa: BLE001
            time.sleep(2 ** tentativa)
    return url, False, 0


def nome_copernicus(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00"


def tiles_1grau():
    return [(la, lo)
            for la in range(math.floor(AOI_LL[1]), math.floor(AOI_LL[3]) + 1)
            for lo in range(math.floor(AOI_LL[0]), math.floor(AOI_LL[2]) + 1)]


def alvos() -> dict[str, list[tuple[str, Path]]]:
    d = {"dem": [], "hand": [], "worldcover": [], "gsw": []}
    for la, lo in tiles_1grau():
        t = f"Copernicus_DSM_COG_10_{nome_copernicus(la, lo)}_DEM"
        d["dem"].append((f"https://copernicus-dem-30m.s3.amazonaws.com/{t}/{t}.tif",
                         OUT / "dem_glo30" / f"{t}.tif"))
        h = f"Copernicus_DSM_COG_10_{nome_copernicus(la, lo)}_HAND"
        d["hand"].append((f"https://glo-30-hand.s3.amazonaws.com/v1/2021/{h}.tif",
                          OUT / "hand_glo30" / f"{h}.tif"))
    # WorldCover: tiles de 3 graus ancorados em multiplos de 3
    la3 = int(math.floor(AOI_LL[1] / 3) * 3)
    lo3 = int(math.floor(AOI_LL[0] / 3) * 3)
    tw = f"S{abs(la3):02d}W{abs(lo3):03d}"
    d["worldcover"].append((
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
        f"ESA_WorldCover_10m_2021_v200_{tw}_Map.tif",
        OUT / "worldcover" / f"{tw}.tif"))
    # JRC Global Surface Water: tiles de 10 graus, nomeados pelo canto NO
    lo10 = int(math.floor(AOI_LL[0] / 10) * 10)
    la10 = int(math.ceil(AOI_LL[3] / 10) * 10)
    d["gsw"].append((
        "https://storage.googleapis.com/global-surface-water/downloads2021/"
        f"occurrence/occurrence_{abs(lo10)}W_{abs(la10)}Sv1_4_2021.tif",
        OUT / "gsw" / f"occurrence_{abs(lo10)}W_{abs(la10)}S.tif"))
    return d


def url_chirps(d: dt.date, flavor: str) -> str:
    return ("https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/"
            f"{flavor}/cogs/{d.year}/chirps-v3.0.{flavor}.{d.year}."
            f"{d.month:02d}.{d.day:02d}.cog")


def dias() -> list[dt.date]:
    n = (DATA_FIM - DATA_INI).days + 1
    return [DATA_INI + dt.timedelta(days=i) for i in range(n)]


def modo_sondar() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("===AQ_RS01_SONDAGEM===")
    print(f"AOI_LL={AOI_LL}")
    print(f"AOI_UTM={AOI_UTM} crs={CRS_METRICO}")
    print(f"TILES_1GRAU={tiles_1grau()}")
    rel = {}
    for chave, itens in alvos().items():
        s = sondar(itens[0][0])
        est = (s["bytes"] or 0) * len(itens)
        print(f"[{chave}] n={len(itens)} ok={s['ok']} status={s['status']} "
              f"amostra={mb(s['bytes'])}MB total_est={mb(est)}MB")
        if not s["ok"]:
            print(f"   URL={itens[0][0]}")
        rel[chave] = {"n": len(itens), "sonda": s, "estimado": est}
    ds = dias()
    print(f"[chuva] dias={len(ds)} ({ds[0]} a {ds[-1]})")
    for flavor in ("rnl", "sat"):
        s = sondar(url_chirps(ds[len(ds) // 2], flavor))
        print(f"   {flavor}: ok={s['ok']} {mb(s['bytes'])}MB (leitura por janela, "
              f"nao baixa o global)")
    print(f"ESTIMATIVA_TOTAL={mb(sum(v['estimado'] for v in rel.values()))}MB")
    (OUT / "sondagem.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("===END===")
    return 0


def modo_baixar() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tarefas = [t for itens in alvos().values() for t in itens]
    print("===AQ_RS01_DOWNLOAD===")
    for chave, itens in alvos().items():
        print(f"[{chave}] {len(itens)} arquivos")
    print(f"TOTAL_TAREFAS={len(tarefas)} workers={WORKERS}")

    ok, falhas, bytes_tot = 0, [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futuros = [ex.submit(baixar_um, t) for t in tarefas]
        for fut in as_completed(futuros):
            url, sucesso, tam = fut.result()
            if sucesso:
                ok += 1
                bytes_tot += tam
            else:
                falhas.append(url)
    print(f"RASTERS ok={ok}/{len(tarefas)} {mb(bytes_tot)}MB "
          f"tempo={(time.time()-t0)/60:.1f}min")
    for u in falhas:
        print(f"  FALHOU {u}")

    # --- chuva por janela em COG ---
    print("---CHUVA---")
    try:
        import numpy as np
        import rioxarray
        import xarray as xr
    except ImportError as exc:
        print(f"PULADO: {exc}")
        print("===END===")
        return 0

    for flavor in ("rnl", "sat"):
        destino = OUT / "chirps" / f"chirps3_{flavor}_rs_2024.nc"
        if destino.exists() and destino.stat().st_size > 0:
            print(f"[{flavor}] ja existe")
            continue
        partes, falhou = [], 0
        for d in dias():
            try:
                da = rioxarray.open_rasterio(url_chirps(d, flavor), chunks=None,
                                             masked=True, lock=False)
                da = da.rio.clip_box(minx=BBOX_CHUVA[0], miny=BBOX_CHUVA[1],
                                     maxx=BBOX_CHUVA[2], maxy=BBOX_CHUVA[3]).load()
                da = da.where(da != -9999.0)
                ds_dia = da.to_dataset(name="precip").squeeze("band", drop=True)
                partes.append(ds_dia.expand_dims(time=[np.datetime64(d)]))
            except Exception:  # noqa: BLE001
                falhou += 1
            time.sleep(0.3)
        if partes:
            destino.parent.mkdir(parents=True, exist_ok=True)
            xr.concat(partes, dim="time").to_netcdf(destino)
            print(f"[{flavor}] dias_ok={len(partes)} falhas={falhou} "
                  f"-> {destino.name}")

    (OUT / "PROVENIENCIA.md").write_text(
        "# Proveniencia - features da AOI do Rio Grande do Sul\n"
        f"- Data de acesso: {time.strftime('%Y-%m-%d %H:%M')}\n"
        f"- AOI derivada do Copernicus EMS EMSR720 (nao escolhida a mao)\n"
        f"- WGS84: {AOI_LL}\n"
        f"- UTM22S (EPSG:{CRS_METRICO}): {AOI_UTM}\n"
        f"- Janela de chuva: {DATA_INI} a {DATA_FIM}\n"
        "- Copernicus DEM GLO-30 / Global 30m HAND: Copernicus, uso livre\n"
        "- ESA WorldCover 10m v200 (2021): CC BY 4.0\n"
        "- JRC Global Surface Water: uso livre com atribuicao\n"
        "- CHIRPS v3 daily (rnl e sat): Climate Hazards Center, UCSB\n"
        "- NOTA: o RS esta dentro de 50S-50N, entao o CHIRPS v2.0 usado no v12\n"
        "  brasileiro tambem cobriria esta AOI. Usamos v3 mesmo assim, para\n"
        "  manter comparabilidade com o piloto ingles. Diferenca declarada.\n"
        "- Status: AQUISICAO. Nenhuma feature extraida, nenhum rotulo promovido.\n",
        encoding="utf-8")
    print("===END===")
    return 0


def main() -> int:
    if "--sondar" in sys.argv:
        return modo_sondar()
    if "--baixar" in sys.argv:
        return modo_baixar()
    print("uso: --sondar | --baixar")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
