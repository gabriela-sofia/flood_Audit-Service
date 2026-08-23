"""
AQUISICAO MESTRE -- tudo o que a frente externa ainda precisa, num script so.

MOTIVO: os downloads ate aqui foram feitos um a um, em serie, com conexao
unica, e alguns falharam em silencio (o inventario do CEMS leu 10 de 246
ativacoes sem erro visivel). Isso custou horas. Este script existe para que
a aquisicao aconteca UMA vez, com paralelismo, retomada e verificacao.

DOIS MODOS, e a ordem importa:

  --sondar   testa cada endpoint, mede tamanho, NAO baixa volume.
             Leva minutos. Serve para descobrir caminho errado ANTES de
             gastar horas nele -- foi exatamente isso que faltou no CEMS.

  --baixar   baixa tudo que a sondagem confirmou, com N conexoes em paralelo.

O que NAO esta aqui, de proposito: Sen1Floods11, UFO, CEMS e GFD, que ja tem
scripts proprios (`n1a`..`n1d`) e ja estao em execucao ou concluidos. Este
cobre a camada de FEATURES, que ainda nao foi tocada.

Nenhuma etapa extrai variavel, promove rotulo ou treina. So adquire.

Uso:
    python scripts/externo/aq_mestre.py --sondar
    python scripts/externo/aq_mestre.py --baixar
    python scripts/externo/aq_mestre.py --baixar --so dem,hand,worldcover
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
STORE = RUNS / "geostore"
OUT = RUNS / "aq-features"
AOI_BNG = (350_000, 350_000, 400_000, 450_000)
AOI_LL = (-3.05, 52.80, -1.70, 54.20)   # lon0, lat0, lon1, lat1
WORKERS = 8
TIMEOUT = 300
DIAS_ANTES = 14   # janela do indice de decaimento da chuva


# ---------------------------------------------------------------- utilitarios

def mb(n) -> float:
    return round((n or 0) / 1024 ** 2, 1)


def sondar(url: str) -> dict:
    """HEAD (com fallback para GET de 1 byte) -- descobre se existe e o tamanho."""
    try:
        r = requests.head(url, timeout=60, allow_redirects=True)
        if r.status_code >= 400 or "Content-Length" not in r.headers:
            r = requests.get(url, timeout=60, stream=True,
                             headers={"Range": "bytes=0-1"})
        tam = r.headers.get("Content-Length") or r.headers.get("content-length")
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


# ------------------------------------------------------------------- catalogo

def tiles_1grau() -> list[tuple[int, int]]:
    lon0, lat0, lon1, lat1 = AOI_LL
    import math
    return [
        (la, lo)
        for la in range(math.floor(lat0), math.floor(lat1) + 1)
        for lo in range(math.floor(lon0), math.floor(lon1) + 1)
    ]


def nome_copernicus(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00"


def alvo_dem() -> list[tuple[str, Path]]:
    """Copernicus DEM GLO-30, bucket publico AWS. Base de elevation e slope."""
    itens = []
    for la, lo in tiles_1grau():
        t = f"Copernicus_DSM_COG_10_{nome_copernicus(la, lo)}_DEM"
        itens.append((
            f"https://copernicus-dem-30m.s3.amazonaws.com/{t}/{t}.tif",
            OUT / "dem_glo30" / f"{t}.tif",
        ))
    return itens


def alvo_hand() -> list[tuple[str, Path]]:
    """
    Global 30m HAND (derivado do GLO-30). Substitui o calculo proprio de HAND.
    O caminho do bucket e incerto -- por isso a sondagem testa variantes.
    """
    itens = []
    for la, lo in tiles_1grau():
        base = f"Copernicus_DSM_COG_10_{nome_copernicus(la, lo)}_HAND"
        itens.append((
            f"https://glo-30-hand.s3.amazonaws.com/v1/2021/{base}.tif",
            OUT / "hand_glo30" / f"{base}.tif",
        ))
    return itens


def alvo_worldcover() -> list[tuple[str, Path]]:
    """ESA WorldCover 10m -- condicao N4 (pareamento por contexto construido)."""
    import math
    lon0, lat0, lon1, lat1 = AOI_LL
    itens = []
    for la in range(int(math.floor(lat0 / 3) * 3), int(math.floor(lat1 / 3) * 3) + 1, 3):
        for lo in range(int(math.floor(lon0 / 3) * 3), int(math.floor(lon1 / 3) * 3) + 1, 3):
            ns = "N" if la >= 0 else "S"
            ew = "E" if lo >= 0 else "W"
            t = f"{ns}{abs(la):02d}{ew}{abs(lo):03d}"
            itens.append((
                "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                f"ESA_WorldCover_10m_2021_v200_{t}_Map.tif",
                OUT / "worldcover" / f"{t}.tif",
            ))
    return itens


def alvo_agua_permanente() -> list[tuple[str, Path]]:
    """
    JRC Global Surface Water (occurrence). Exigido pela regra R3 do contrato de
    reducao a pontos: agua permanente NAO e inundacao e precisa ser removida
    antes de qualquer rotulo. Tiles de 10 graus.
    """
    return [(
        "https://storage.googleapis.com/global-surface-water/downloads2021/"
        "occurrence/occurrence_10W_60Nv1_4_2021.tif",
        OUT / "gsw" / "occurrence_10W_60N.tif",
    )]


def datas_necessarias() -> list[dt.date]:
    """Datas de evento na AOI mais a janela anterior, lidas do geostore."""
    import geopandas as gpd
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from armazenamento_geo import ler_bbox

    g = ler_bbox(STORE / "outlines_elegiveis.parquet", AOI_BNG)
    d = pd.to_datetime(g["start_date"], errors="coerce", utc=True).dt.date.dropna()
    precisa: set = set()
    for x in sorted(set(d)):
        for k in range(DIAS_ANTES + 1):
            precisa.add(x - dt.timedelta(days=k))
    return sorted(precisa)


# CHUVA: nao e mais tratada aqui.
#
# A sondagem de 2026-08-07 mostrou que nenhum palpite de URL funcionava. O
# template correto foi obtido da implementacao de referencia dhis2/dhis2eo:
#   .../v3.0/daily/final/{flavor}/cogs/{ano}/chirps-v3.0.{flavor}.{a}.{m}.{d}.cog
# e os arquivos sao COG, o que permite ler SO a janela da AOI em vez de baixar
# o global. Isso derruba o volume de ~40 GB para alguns MB e muda completamente
# o metodo de aquisicao -- por isso foi para um script proprio,
# `aq_chirps3_aoi.py`, em vez de virar mais um alvo de download em massa aqui.
PADROES_CHIRPS: list[str] = []

CATALOGO = {
    "dem": (alvo_dem, "Copernicus DEM GLO-30 -- elevation, slope", "Copernicus, livre"),
    "hand": (alvo_hand, "Global 30m HAND -- variavel hand_m", "Copernicus/ASF, livre"),
    "worldcover": (alvo_worldcover, "ESA WorldCover 10m -- condicao N4", "CC BY 4.0"),
    "gsw": (alvo_agua_permanente, "JRC Global Surface Water -- regra R3", "livre com atribuicao"),
}


# ------------------------------------------------------------------ execucao

def modo_sondar() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rel = {}
    print("===AQ_SONDAGEM===")
    print(f"AOI={AOI_LL} tiles_1grau={len(tiles_1grau())}")

    for chave, (fn, desc, lic) in CATALOGO.items():
        itens = fn()
        amostra = itens[0]
        s = sondar(amostra[0])
        est = (s["bytes"] or 0) * len(itens)
        print(f"[{chave}] {desc}")
        print(f"   n={len(itens)} amostra_ok={s['ok']} status={s['status']} "
              f"tam_amostra={mb(s['bytes'])}MB estimado_total={mb(est)}MB")
        if not s["ok"]:
            print(f"   URL_FALHOU={amostra[0]}")
        rel[chave] = {"n": len(itens), "sonda": s, "estimado_bytes": est,
                      "licenca": lic, "url_amostra": amostra[0]}

    # chuva: descobrir o padrao de URL antes de comprometer horas
    print("[chuva] CHIRPS v3 diario -- descoberta de padrao")
    try:
        datas = datas_necessarias()
        print(f"   datas_necessarias={len(datas)} "
              f"({datas[0]} a {datas[-1]}, evento + {DIAS_ANTES}d antes)")
    except Exception as exc:  # noqa: BLE001
        datas = [dt.date(2015, 12, 5)]
        print(f"   AVISO nao consegui ler o geostore ({type(exc).__name__}); "
              f"sondando com data de teste")
    teste = datas[len(datas) // 2]
    padrao_ok = None
    for p in PADROES_CHIRPS:
        url = p.format(a=teste.year, m=teste.month, d=teste.day)
        s = sondar(url)
        print(f"   {'OK ' if s['ok'] else 'nao'} {mb(s['bytes'])}MB  {url[:110]}")
        if s["ok"] and padrao_ok is None:
            padrao_ok = p
            rel["chuva"] = {"padrao": p, "tam_arquivo": s["bytes"],
                            "n_datas": len(datas),
                            "estimado_bytes": (s["bytes"] or 0) * len(datas)}
    if padrao_ok:
        e = rel["chuva"]["estimado_bytes"]
        print(f"   PADRAO_ESCOLHIDO -> estimado_total={mb(e)}MB para {len(datas)} dias")
    else:
        print("   NENHUM_PADRAO_FUNCIONOU -- alternativa e ERA5-Land via cdsapi "
              "(subsetting por bbox, arquivos muito menores) ou CHIRPS via GEE")
        rel["chuva"] = {"padrao": None}

    # credencial do CDS, caso a rota vire ERA5
    cdsrc = Path.home() / ".cdsapirc"
    print(f"CDSAPIRC_PRESENTE={cdsrc.exists()}")
    rel["cdsapirc"] = cdsrc.exists()

    total = sum(v.get("estimado_bytes", 0) for v in rel.values() if isinstance(v, dict))
    print(f"ESTIMATIVA_TOTAL={mb(total)}MB")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sondagem.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("===END===")
    return 0


def modo_baixar(somente: set[str] | None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sond = {}
    p_sond = OUT / "sondagem.json"
    if p_sond.exists():
        sond = json.loads(p_sond.read_text(encoding="utf-8"))

    tarefas: list[tuple[str, Path]] = []
    print("===AQ_DOWNLOAD===")
    for chave, (fn, desc, _lic) in CATALOGO.items():
        if somente and chave not in somente:
            continue
        itens = fn()
        tarefas += itens
        print(f"[{chave}] {len(itens)} arquivos -- {desc}")

    padrao = (sond.get("chuva") or {}).get("padrao")
    if padrao and (not somente or "chuva" in somente):
        datas = datas_necessarias()
        for x in datas:
            url = padrao.format(a=x.year, m=x.month, d=x.day)
            tarefas.append((url, OUT / "chirps" / Path(url).name))
        print(f"[chuva] {len(datas)} arquivos -- CHIRPS v3 diario")
    elif not padrao:
        print("[chuva] PULADO -- rode --sondar antes, ou nenhum padrao funcionou")

    print(f"TOTAL_TAREFAS={len(tarefas)} workers={WORKERS}")
    ok, falhou, bytes_tot = 0, [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futuros = {ex.submit(baixar_um, t): t for t in tarefas}
        for i, fut in enumerate(as_completed(futuros), 1):
            url, sucesso, tam = fut.result()
            if sucesso:
                ok += 1
                bytes_tot += tam
            else:
                falhou.append(url)
            if i % 50 == 0 or i == len(tarefas):
                dt_s = time.time() - t0
                taxa = mb(bytes_tot) / (dt_s / 60) if dt_s > 0 else 0
                print(f"  {i}/{len(tarefas)} ok={ok} falhas={len(falhou)} "
                      f"{mb(bytes_tot)}MB {taxa:.0f}MB/min", flush=True)

    print(f"CONCLUIDO ok={ok} falhas={len(falhou)} total={mb(bytes_tot)}MB "
          f"tempo={(time.time()-t0)/60:.1f}min")
    for u in falhou[:15]:
        print(f"  FALHOU {u}")
    (OUT / "download.json").write_text(
        json.dumps({"ok": ok, "falhas": falhou, "bytes": bytes_tot},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "PROVENIENCIA.md").write_text(
        "# Proveniencia - camada de features (AQ-MASTER)\n"
        f"- Data de acesso: {time.strftime('%Y-%m-%d %H:%M')}\n"
        f"- AOI: BNG {AOI_BNG} / WGS84 {AOI_LL}\n"
        "- Copernicus DEM GLO-30: Copernicus, uso livre com atribuicao\n"
        "- Global 30m HAND: derivado do GLO-30, uso livre\n"
        "- ESA WorldCover 10m v200 (2021): CC BY 4.0\n"
        "- JRC Global Surface Water: uso livre com atribuicao\n"
        "- CHIRPS v3: Climate Hazards Center, UC Santa Barbara\n"
        "- Natureza: TODAS sao camadas de FEATURE ou de filtro. Nenhuma e\n"
        "  rotulo de inundacao e nenhuma pode ser usada como evidencia de\n"
        "  evento ocorrido.\n"
        "- Status: AQUISICAO. Nenhuma variavel extraida, nenhum rotulo\n"
        "  promovido, nenhum modelo treinado.\n",
        encoding="utf-8")
    print("===END===")
    return 0 if not falhou else 2


def main() -> int:
    args = sys.argv[1:]
    somente = None
    if "--so" in args:
        somente = set(args[args.index("--so") + 1].split(","))
    if "--sondar" in args:
        return modo_sondar()
    if "--baixar" in args:
        return modo_baixar(somente)
    print("uso: --sondar | --baixar [--so dem,hand,worldcover,gsw,chuva]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
