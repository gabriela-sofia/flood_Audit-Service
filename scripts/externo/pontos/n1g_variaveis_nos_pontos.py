"""
N1G -- Anexa features topograficas aos 71.140 pontos do Nivel 1.

O PROBLEMA QUE RESOLVE:
Sen1Floods11 e UFO produziram 71.140 pontos, dos quais 30.500 sao
NAO_INUNDADO_OBSERVADO -- ou seja, 89% de todo o negativo por observacao do
projeto. Eles entraram no `ds01` com TODAS as colunas de variavel vazias, por
decisao registrada, e portanto estao inutilizaveis para qualquer modelo.
Sao o maior ativo parado do projeto.

POR QUE SO ELEVACAO, HAND E COBERTURA -- e nao declividade e TWI:
declividade e TWI exigem uma GRADE local reprojetada para metros; nao se
calculam ponto a ponto. Fazer isso para 661 chips espalhados por 6 continentes
significaria 661 reprojecoes e 661 execucoes de acumulacao de fluxo. O custo
nao se justifica para o uso pretendido destes pontos, que e AUDITAR o criterio
de negativo por concordancia espacial -- nao treinar o modelo do artigo.

Elevacao, HAND e cobertura do solo sao amostrados POR PONTO, direto do tile
global, sem reprojecao. Isso e valido porque nenhum deles e derivada
direcional: sao valores de celula, nao gradientes.

As colunas `slope_deg` e `twi_dinf` saem explicitamente como NaN, e o motivo
fica no `resumo.json`. Nao inventar valor e parte do contrato.

ESTRATEGIA DE ACESSO: os pontos sao agrupados por tile de 1 grau, o tile e
baixado uma vez (indo para o mesmo `cache-global` que o `cems02` usa) e todos
os pontos daquele tile sao amostrados de uma vez. Sem isso, 71 mil pontos
dariam 71 mil aberturas de raster.

NAO faz: nao promove negativo, nao treina, nao calcula derivada direcional.

Uso:
    python scripts/externo/n1g_variaveis_nos_pontos.py
    python scripts/externo/n1g_variaveis_nos_pontos.py --limite-tiles 20
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
CACHE = RUNS / "cache-global"
ENTRADA = RUNS / "n1f-pontos" / "nivel1_pontos_v1.csv"
OUT = RUNS / "n1g-pontos-com-features"


def baixar(url: str, destino: Path) -> bool:
    import requests

    if destino.exists() and destino.stat().st_size > 0:
        return True
    destino.parent.mkdir(parents=True, exist_ok=True)
    for tentativa in range(3):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                if r.status_code == 404:
                    return False          # tile inexistente (oceano) -- normal
                r.raise_for_status()
                tmp = destino.with_suffix(destino.suffix + ".part")
                with tmp.open("wb") as fh:
                    for ch in r.iter_content(chunk_size=1 << 20):
                        fh.write(ch)
                tmp.replace(destino)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2 ** tentativa)
    return False


def nome_base(la: int, lo: int) -> str:
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    return f"{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00"


def amostrar_tile(tif: Path, pts: pd.DataFrame) -> np.ndarray:
    """Amostra o raster nos pontos (lon, lat) usando `sample` do rasterio."""
    import rasterio

    with rasterio.open(tif) as src:
        coords = list(zip(pts["lon"].to_numpy(), pts["lat"].to_numpy()))
        vals = np.array([v[0] for v in src.sample(coords)], dtype="float64")
        nodata = src.nodata
    if nodata is not None:
        vals = np.where(vals == nodata, np.nan, vals)
    return vals


def main() -> int:
    args = sys.argv[1:]
    limite = (int(args[args.index("--limite-tiles") + 1])
              if "--limite-tiles" in args else 10_000)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===N1G_FEATURES_NIVEL1===")

    df = pd.read_csv(ENTRADA)
    df = df[df["lat"].notna() & df["lon"].notna()].reset_index(drop=True)
    print(f"PONTOS={len(df):,}")

    # agrupamento por tile de 1 grau (DEM/HAND) e de 3 graus (WorldCover)
    df["_la"] = np.floor(df["lat"]).astype(int)
    df["_lo"] = np.floor(df["lon"]).astype(int)
    df["_la3"] = (np.floor(df["lat"] / 3) * 3).astype(int)
    df["_lo3"] = (np.floor(df["lon"] / 3) * 3).astype(int)

    tiles_1g = sorted(set(zip(df["_la"], df["_lo"])))
    tiles_3g = sorted(set(zip(df["_la3"], df["_lo3"])))
    print(f"TILES_1GRAU={len(tiles_1g)} TILES_3GRAUS={len(tiles_3g)}")

    for col in ("elevation_m", "hand_m", "worldcover"):
        df[col] = np.nan

    # ---------- DEM e HAND, tile de 1 grau ----------
    ok_dem = ok_hand = faltou = 0
    for n, (la, lo) in enumerate(tiles_1g[:limite], 1):
        sel = (df["_la"] == la) & (df["_lo"] == lo)
        base = nome_base(la, lo)
        for chave, sufixo, url_base, coluna in (
            ("dem", "DEM", "https://copernicus-dem-30m.s3.amazonaws.com", "elevation_m"),
            ("hand", "HAND", "https://glo-30-hand.s3.amazonaws.com/v1/2021", "hand_m"),
        ):
            t = f"Copernicus_DSM_COG_10_{base}_{sufixo}"
            url = (f"{url_base}/{t}/{t}.tif" if chave == "dem"
                   else f"{url_base}/{t}.tif")
            destino = CACHE / chave / f"{t}.tif"
            if not baixar(url, destino):
                faltou += 1
                continue
            try:
                df.loc[sel, coluna] = amostrar_tile(destino, df.loc[sel])
                if chave == "dem":
                    ok_dem += 1
                else:
                    ok_hand += 1
            except Exception as exc:  # noqa: BLE001
                print(f"   AVISO {t}: {type(exc).__name__}")
        if n % 20 == 0:
            print(f"  tiles 1g {n}/{min(len(tiles_1g), limite)} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)

    # ---------- WorldCover, tile de 3 graus ----------
    ok_wc = 0
    for la, lo in tiles_3g[:limite]:
        sel = (df["_la3"] == la) & (df["_lo3"] == lo)
        ns = "N" if la >= 0 else "S"
        ew = "E" if lo >= 0 else "W"
        t = f"{ns}{abs(la):02d}{ew}{abs(lo):03d}"
        url = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
               f"ESA_WorldCover_10m_2021_v200_{t}_Map.tif")
        destino = CACHE / "worldcover" / f"{t}.tif"
        if not baixar(url, destino):
            continue
        try:
            df.loc[sel, "worldcover"] = amostrar_tile(destino, df.loc[sel])
            ok_wc += 1
        except Exception:  # noqa: BLE001
            pass

    # declividade e TWI ficam explicitamente ausentes -- ver cabecalho
    df["slope_deg"] = np.nan
    df["twi_dinf"] = np.nan

    df = df.drop(columns=["_la", "_lo", "_la3", "_lo3"])
    saida = OUT / "nivel1_pontos_com_features_v1.csv"
    df.to_csv(saida, index=False)

    print("---COBERTURA---")
    for c in ("elevation_m", "hand_m", "worldcover"):
        n_ok = int(df[c].notna().sum())
        print(f"  {c:14s} {n_ok:,}/{len(df):,} ({100*n_ok/len(df):.1f}%)")
    print(f"TILES ok_dem={ok_dem} ok_hand={ok_hand} ok_wc={ok_wc} sem_tile={faltou}")

    print("---MEDIANAS POR CLASSE---")
    for c, sub in df.groupby("classe_obs"):
        e = sub["elevation_m"].median()
        h = sub["hand_m"].median()
        print(f"  {c:24s} n={len(sub):6,} elev={e:8.1f} hand={h:8.2f}")

    (OUT / "resumo.json").write_text(json.dumps({
        "n_pontos": int(len(df)),
        "tiles_1grau": len(tiles_1g), "tiles_3graus": len(tiles_3g),
        "cobertura": {c: int(df[c].notna().sum())
                      for c in ("elevation_m", "hand_m", "worldcover")},
        "ausentes_por_decisao": ["slope_deg", "twi_dinf"],
        "motivo_ausencia": (
            "Declividade e TWI sao derivadas direcionais: exigem grade local "
            "reprojetada para metros, nao se calculam ponto a ponto. Para 661 "
            "chips em 6 continentes o custo nao se justifica no uso pretendido "
            "destes pontos, que e auditar o criterio de negativo, nao treinar "
            "o modelo do artigo."),
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SAIDA={saida.name} TEMPO={(time.time()-t0)/60:.1f}min")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
