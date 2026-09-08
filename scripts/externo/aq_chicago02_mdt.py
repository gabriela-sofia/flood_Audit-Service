"""MDT de Chicago a partir do 3DEP da USGS, ja em metros e recortado.

POR QUE NAO ENTREGAR O TILE CRU AO PIPELINE DE TERRENO:

`aq_chamados04` calcula o passo de reamostragem como

    passo = 30 m / (resolucao_nativa * fator_xy_para_m)

Isso pressupoe raster em unidade LINEAR. O 3DEP vem em NAD83 geografico, com XY
em GRAUS: para fazer a formula fechar seria preciso inventar um fator de
graus-para-metros, que muda com a latitude e seria uma constante errada em toda
parte menos numa linha. Nova York usa pes e Sao Paulo usa metros -- os dois
lineares -- e por isso a formula servia.

Entao a reprojecao acontece AQUI, na aquisicao: o que chega ao pipeline ja esta
em UTM metrico, e `fator_xy_para_m` vale 1,0 como nas outras cidades. A cadeia
canonica -- WhiteboxTools, D-infinity, 30 m, limiar de canal 0,1123 km2 --
continua sendo a mesma, e essa e a condicao para que o TWI de Chicago signifique
o mesmo que o TWI de Nova York.

-------------------------------------------------------------------------
A MARGEM, E O QUE ELA NAO CONSERTA
-------------------------------------------------------------------------

Area contribuinte e acumulada morro acima. Recortar exatamente na cidade cortaria
a bacia e daria acumulacao menor na borda -- e o TWI depende dela. Por isso o
recorte leva uma margem declarada.

A margem reduz o efeito e NAO o elimina: a bacia real pode comecar fora dela. E a
mesma limitacao que Nova York e Sao Paulo ja carregam, porque os dois tambem
foram derivados de um MDT municipal. Fica declarada no manifesto em vez de
silenciada.

Uso:
    python scripts/externo/aq_chicago02_mdt.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import rasterio.crs
import rasterio.transform
import requests
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "local_runs" / "mdt-cidades" / "chicago"
BRUTO = OUT / "bruto"
CACHE = OUT / "tile"

# 3DEP 1/3 arco-segundo (~10 m), bare-earth. O tile de 1 grau que cobre Chicago.
TILE = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/"
        "current/n42w088/USGS_13_n42w088.tif")
AGENTE = "REV-P/1.0 (pesquisa academica de suscetibilidade a inundacao)"

# Caixa dos chamados de Chicago, com margem para nao cortar a bacia na borda.
CAIXA_CIDADE = (-87.95, 41.62, -87.50, 42.05)     # lon_min, lat_min, lon_max, lat_max
MARGEM_GRAUS = 0.12                                # ~10 km
EPSG_METRICO = 32616                               # UTM 16N, o fuso de Chicago
RESOLUCAO_ALVO_M = 10.0                            # nativa do 3DEP; o pipeline vai a 30


def baixar_tile() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    destino = CACHE / TILE.rsplit("/", 1)[-1]
    if destino.exists() and destino.stat().st_size > 1_000_000:
        print(f"  tile ja em disco: {destino.name} "
              f"({destino.stat().st_size / 1048576:.1f} MB)")
        return destino
    print(f"  baixando {TILE}")
    with requests.get(TILE, stream=True, timeout=600,
                      headers={"User-Agent": AGENTE}) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        lido = 0
        with destino.open("wb") as f:
            for bloco in r.iter_content(chunk_size=4 << 20):
                f.write(bloco)
                lido += len(bloco)
                if total:
                    print(f"    {lido / 1048576:7.1f} / {total / 1048576:.1f} MB",
                          end="\r", flush=True)
    print()
    return destino


def recortar_e_reprojetar(tile: Path) -> dict:
    lon_min, lat_min, lon_max, lat_max = CAIXA_CIDADE
    caixa = (lon_min - MARGEM_GRAUS, lat_min - MARGEM_GRAUS,
             lon_max + MARGEM_GRAUS, lat_max + MARGEM_GRAUS)

    with rasterio.open(tile) as src:
        janela = from_bounds(*caixa, transform=src.transform)
        dados = src.read(1, window=janela)
        transform_origem = src.window_transform(janela)
        crs_origem = src.crs
        nodata = src.nodata
        res_nativa = abs(src.res[0])

    if nodata is not None:
        dados = np.where(dados == nodata, np.nan, dados)
    dados = dados.astype("float32")

    alvo_crs = rasterio.crs.CRS.from_epsg(EPSG_METRICO)
    transform_alvo, larg, alt = calculate_default_transform(
        crs_origem, alvo_crs, dados.shape[1], dados.shape[0],
        *rasterio.transform.array_bounds(dados.shape[0], dados.shape[1],
                                         transform_origem),
        resolution=RESOLUCAO_ALVO_M)
    saida = np.full((alt, larg), np.nan, dtype="float32")
    reproject(source=dados, destination=saida,
              src_transform=transform_origem, src_crs=crs_origem,
              dst_transform=transform_alvo, dst_crs=alvo_crs,
              src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.bilinear)

    BRUTO.mkdir(parents=True, exist_ok=True)
    destino = BRUTO / "chicago_3dep_utm16n_10m.tif"
    perfil = {"driver": "GTiff", "height": alt, "width": larg, "count": 1,
              "dtype": "float32", "crs": alvo_crs, "transform": transform_alvo,
              "nodata": np.nan, "compress": "deflate", "tiled": True}
    with rasterio.open(destino, "w", **perfil) as dst:
        dst.write(saida, 1)

    validos = np.isfinite(saida)
    return {
        "arquivo": str(destino.relative_to(REPO)),
        "sha256": hashlib.sha256(destino.read_bytes()).hexdigest(),
        "bytes": destino.stat().st_size,
        "forma": [int(alt), int(larg)],
        "resolucao_m": RESOLUCAO_ALVO_M,
        "crs": f"EPSG:{EPSG_METRICO}",
        "resolucao_nativa_graus": round(res_nativa, 8),
        "caixa_com_margem": list(caixa),
        "celulas_validas": int(validos.sum()),
        "pct_validas": round(100.0 * float(validos.mean()), 2),
        "elevacao_min_m": round(float(np.nanmin(saida)), 2),
        "elevacao_max_m": round(float(np.nanmax(saida)), 2),
        "elevacao_mediana_m": round(float(np.nanmedian(saida)), 2),
    }


def main() -> int:
    argparse.ArgumentParser().parse_args()
    print("=== MDT DE CHICAGO (USGS 3DEP) ===")
    tile = baixar_tile()
    print("  recortando e reprojetando para UTM 16N a 10 m")
    info = recortar_e_reprojetar(tile)
    for k in ("forma", "celulas_validas", "pct_validas", "elevacao_min_m",
              "elevacao_max_m", "elevacao_mediana_m"):
        print(f"  {k:20s} {info[k]}")

    manifesto = {
        "tipo": "mdt_cidade_chicago",
        "gerado_por": "scripts/externo/aq_chicago02_mdt.py",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cidade": "chicago",
        "origem": "USGS 3DEP 1/3 arco-segundo (~10 m), bare-earth",
        "url": TILE,
        "reprojecao_na_aquisicao": (
            "o 3DEP vem em NAD83 geografico, com XY em graus; o pipeline de "
            "terreno calcula o passo de reamostragem supondo unidade linear. "
            "Reprojetar aqui evita inventar um fator graus-para-metros, que "
            "muda com a latitude"),
        "margem_graus": MARGEM_GRAUS,
        "limitacao_da_margem": (
            "area contribuinte e acumulada morro acima; a margem reduz o corte "
            "da bacia na borda mas nao o elimina, e a bacia real pode comecar "
            "fora dela. Mesma limitacao que Nova York e Sao Paulo carregam, por "
            "tambem virem de MDT municipal"),
        **info,
        "nao_e": (
            "nao e modelo de superficie: e bare-earth, que e a exigencia do "
            "projeto em cidade densa. E nao e terreno derivado -- HAND, TWI e "
            "declive saem da cadeia canonica em `aq_chamados04`"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifesto_mdt.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/externo/aq_chicago02_mdt.py\n", encoding="utf-8")
    print(f"GRAVADO={destino_txt}" if (destino_txt := info["arquivo"]) else "")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
