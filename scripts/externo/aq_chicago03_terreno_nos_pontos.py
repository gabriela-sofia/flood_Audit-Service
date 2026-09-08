"""Amostra o terreno de Chicago nos pontos dos chamados.

A cadeia canonica ja produziu HAND, TWI e declive a 30 m para a cidade
(`aq_chamados04 --cidade chicago`). Falta ler o valor de cada raster no ponto de
cada chamado, que e o que o desenho consome.

DUAS DECISOES QUE MUDAM O RESULTADO:

  ELEVACAO RELATIVA. O vetor servido usa `elevation_rel_m`, a elevacao contra o
  P1 da propria regiao, e nao a absoluta. A absoluta punha 0,0% dos pontos de
  Curitiba no dominio de treino contra 94,5% da relativa. Aqui a linha-base e
  calculada DA PROPRIA CIDADE, pelo percentil 1 das elevacoes amostradas, que e
  a mesma definicao do `ds04`.

  AUSENCIA NAO VIRA ZERO. Ponto fora do raster, ou em celula sem dado, recebe
  NaN e sai da analise. Zero de HAND significa "no nivel do rio", que e o valor
  mais alagavel possivel -- preencher ausencia com zero inventaria o pior caso.

O QUE NAO FAZ: nao promove fonte, nao entra na tabela unica, nao vira variavel de
modelo servido.

Uso:
    python scripts/externo/aq_chicago03_terreno_nos_pontos.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import rowcol

REPO = Path(__file__).resolve().parents[2]
CHAMADOS = REPO / "local_runs" / "aq-chicago-01"
MDT = REPO / "local_runs" / "mdt-cidades" / "chicago"
OUT = REPO / "local_runs" / "aq-chicago-03"

RASTERS = {
    "elevation_m": MDT / "chicago_mdt_30m.tif",
    "hand_m": MDT / "chicago_mdt_30m_hand.tif",
    "twi_dinf": MDT / "chicago_mdt_30m_twi.tif",
    "slope_deg": MDT / "chicago_mdt_30m_slope.tif",
}
PERCENTIL_BASELINE = 1.0


def carregar_pontos() -> pd.DataFrame:
    partes = []
    for arquivo, classe in (("chicago311_positivos.csv", 1),
                            ("chicago311_fundo.csv", 0)):
        d = pd.read_csv(CHAMADOS / arquivo)
        d["classe"] = classe
        partes.append(d)
    d = pd.concat(partes, ignore_index=True).rename(
        columns={"sr_number": "ponto_id"})
    d["lat"] = pd.to_numeric(d.latitude, errors="coerce")
    d["lon"] = pd.to_numeric(d.longitude, errors="coerce")
    d["dia"] = d.created_date.astype(str).str[:10]
    d = d[d.lat.notna() & d.lon.notna()
          & d.dia.str.match(r"\d{4}-\d{2}-\d{2}")]
    return d[["ponto_id", "sr_type", "dia", "lat", "lon", "classe"]].copy()


def amostrar(d: pd.DataFrame) -> pd.DataFrame:
    with rasterio.open(RASTERS["elevation_m"]) as ref:
        crs_raster = ref.crs
    t = Transformer.from_crs(CRS.from_epsg(4326), crs_raster,
                             always_xy=True).transform
    x, y = t(d.lon.to_numpy(), d.lat.to_numpy())

    for nome, caminho in RASTERS.items():
        with rasterio.open(caminho) as src:
            z = src.read(1).astype("float32")
            if src.nodata is not None:
                z = np.where(z == src.nodata, np.nan, z)
            linhas, colunas = rowcol(src.transform, x, y)
            linhas = np.asarray(linhas)
            colunas = np.asarray(colunas)
            dentro = ((linhas >= 0) & (linhas < z.shape[0])
                      & (colunas >= 0) & (colunas < z.shape[1]))
            v = np.full(len(d), np.nan, dtype="float32")
            v[dentro] = z[linhas[dentro], colunas[dentro]]
        d[nome] = v
    return d


def main() -> int:
    argparse.ArgumentParser().parse_args()
    faltando = [str(p) for p in RASTERS.values() if not p.exists()]
    if faltando:
        print("rasters ausentes; rode antes "
              "`aq_chamados04_derivar_terreno.py --cidade chicago`:")
        for f in faltando:
            print("  ", f)
        return 2

    print("=== TERRENO NOS PONTOS DE CHICAGO ===")
    d = carregar_pontos()
    print(f"pontos: {len(d):,} ({int(d.classe.sum()):,} positivos)")
    d = amostrar(d)

    base = float(np.nanpercentile(d.elevation_m, PERCENTIL_BASELINE))
    d["elevation_rel_m"] = d.elevation_m - base
    print(f"linha-base de elevacao (P{PERCENTIL_BASELINE:.0f} da cidade): "
          f"{base:.3f} m")

    colunas = list(RASTERS) + ["elevation_rel_m"]
    completo = d[colunas].notna().all(axis=1)
    print(f"com as {len(colunas)} variaveis: {int(completo.sum()):,} "
          f"({100 * completo.mean():.1f}%)")
    for c in colunas:
        v = d.loc[completo, c]
        print(f"  {c:18s} mediana {v.median():9.3f}  p5 {v.quantile(.05):9.3f}  "
              f"p95 {v.quantile(.95):9.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    destino = OUT / "chicago_pontos_com_terreno.csv"
    d[completo].to_csv(destino, index=False)

    manifesto = {
        "tipo": "terreno_nos_pontos_chicago",
        "gerado_por": "scripts/externo/aq_chicago03_terreno_nos_pontos.py",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cidade": "chicago",
        "cadeia": "WhiteboxTools D-infinity 30 m, limiar de canal 0,1123 km2",
        "rasters": {k: str(v.relative_to(REPO)) for k, v in RASTERS.items()},
        "pontos_lidos": int(len(d)),
        "pontos_completos": int(completo.sum()),
        "pct_completos": round(100.0 * float(completo.mean()), 2),
        "positivos_completos": int(d.loc[completo, "classe"].sum()),
        "linha_base_elevacao_m": round(base, 4),
        "percentil_da_linha_base": PERCENTIL_BASELINE,
        "por_que_relativa": (
            "o vetor servido usa elevacao contra o P1 da propria regiao; a "
            "absoluta punha 0,0% dos pontos de Curitiba no dominio de treino "
            "contra 94,5% da relativa"),
        "ausencia_nao_vira_zero": (
            "ponto fora do raster ou em celula sem dado sai da analise; zero de "
            "HAND significa 'no nivel do rio', o valor mais alagavel possivel"),
        "saida": str(destino.relative_to(REPO)),
        "nao_e": (
            "nao e promocao de fonte e nao entra na tabela unica: e a leitura "
            "do terreno nos pontos de uma cidade que existe para testar se o "
            "efeito medido em Nova York se repete"),
    }
    (OUT / "manifesto.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/externo/aq_chicago03_terreno_nos_pontos.py\n",
        encoding="utf-8")
    print(f"GRAVADO={destino}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
