"""
EXT-UK-06 -- ESA WorldCover 10 m na AOI: a camada que falta para o criterio N4.

POR QUE ESTA CAMADA E OBRIGATORIA, e nao um extra:
a contabilidade do EXT-UK-05 mostrou que 94,4% da AOI e candidata a negativo
(117.940 pontos de 125.000). Amostrar negativo desse conjunto sem controle
produziria uma classe negativa dominada por campo aberto, enquanto os
positivos estao concentrados em fundo de vale urbanizado. O classificador
aprenderia a separar URBANO de RURAL -- o que ele faria com AUC alto e sem
nenhum valor cientifico.

A condicao N4 do documento de adjudicacao (docs/metodologia_cientifica/
ext_uk_adjudicacao_negativo_v1.md) exige pareamento por contexto construido.
Sem uma camada de cobertura do solo, N4 nao roda e o negativo nao pode ser
promovido.

Escolha do produto: ESA WorldCover 10 m v200 (2021).
  - 10 m, compativel com a escala de quarteirao;
  - global e gratuito (CC BY 4.0), o que preserva a transferibilidade do
    metodo para o Brasil sem trocar de fonte;
  - combina Sentinel-1 e Sentinel-2, entao nao depende de ceu limpo;
  - substitui o MapBiomas do v12 fora do Brasil (MapBiomas e Brasil-only).

Os tiles sao de 3x3 graus, nomeados pelo canto sudoeste. Este script calcula
quais cobrem a AOI e baixa apenas esses.

Fonte  : ESA WorldCover / bucket publico AWS (esa-worldcover)
Licenca: CC BY 4.0
AOI    : BNG E 350-400 km / N 350-450 km (EXT-UK-03)

NAO faz: nao extrai feature, nao classifica, nao promove rotulo.

Uso:
    python scripts/externo/ext_uk06_worldcover_aoi.py
"""

from __future__ import annotations

import math
from pathlib import Path

from pyproj import Transformer

from n1_common import BASE_OUT, baixar, escrever_proveniencia, gb, salvar_json

OUTDIR = BASE_OUT / "ext-uk-06-worldcover"
AOI_BNG = (350_000, 350_000, 400_000, 450_000)
BASE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)


def nome_tile(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("===EXT-UK-06===")

    tr = Transformer.from_crs(27700, 4326, always_xy=True)
    cantos = [
        tr.transform(AOI_BNG[0], AOI_BNG[1]),
        tr.transform(AOI_BNG[2], AOI_BNG[1]),
        tr.transform(AOI_BNG[0], AOI_BNG[3]),
        tr.transform(AOI_BNG[2], AOI_BNG[3]),
    ]
    lons = [c[0] for c in cantos]
    lats = [c[1] for c in cantos]
    print(f"AOI_WGS84 lon=({min(lons):.3f},{max(lons):.3f}) lat=({min(lats):.3f},{max(lats):.3f})")

    # tiles de 3 graus, ancorados em multiplos de 3
    lat0 = int(math.floor(min(lats) / 3) * 3)
    lat1 = int(math.floor(max(lats) / 3) * 3)
    lon0 = int(math.floor(min(lons) / 3) * 3)
    lon1 = int(math.floor(max(lons) / 3) * 3)

    tiles = [
        nome_tile(la, lo)
        for la in range(lat0, lat1 + 1, 3)
        for lo in range(lon0, lon1 + 1, 3)
    ]
    print("TILES=" + "; ".join(tiles))

    resultados = []
    for t in tiles:
        url = BASE_URL.format(tile=t)
        destino = OUTDIR / f"{t}.tif"
        ok = baixar(url, destino)
        tam = destino.stat().st_size if destino.exists() else 0
        resultados.append({"tile": t, "url": url, "ok": ok, "bytes": tam})
        print(f"  {t} ok={ok} {gb(tam)}GB")

    total = sum(r["bytes"] for r in resultados)
    n_ok = sum(1 for r in resultados if r["ok"] and r["bytes"] > 0)
    salvar_json(OUTDIR / "tiles.json", resultados)

    escrever_proveniencia(
        OUTDIR,
        titulo="ESA WorldCover 10 m v200 (2021) - recorte da AOI EXT-UK",
        fonte="ESA WorldCover / bucket publico AWS esa-worldcover",
        identificador="v200/2021/map, tiles " + ",".join(tiles),
        licenca="CC BY 4.0",
        natureza="OBSERVADO (classificacao Sentinel-1 + Sentinel-2), 10 m, "
                 "referencia temporal 2021 -- NAO e serie temporal.",
        uso_pretendido="Condicao N4 da adjudicacao do negativo: pareamento de "
                       "candidatos a negativo com positivos por contexto "
                       "construido. Substituto global do MapBiomas do v12.",
        uso_proibido="Usar como evidencia de inundacao; tratar a classe agua "
                     "como cheia (WorldCover 2021 e um retrato anual, nao "
                     "captura evento); assumir que representa o uso do solo "
                     "na data de eventos de 2000-2020 sem ressalva escrita.",
        extra=f"- Tiles baixados: {n_ok}/{len(tiles)} ({gb(total)}GB)\n"
              "- LIMITACAO TEMPORAL: referencia 2021 aplicada a eventos de\n"
              "  2000-2025. O v12 tem o mesmo problema (MapBiomas 2023 sobre\n"
              "  eventos de 2014+). Deve ser declarado como limitacao, nao\n"
              "  corrigido silenciosamente.\n",
    )

    print(f"BAIXADOS={n_ok}/{len(tiles)} total={gb(total)}GB")
    print("===END===")
    return 0 if n_ok == len(tiles) else 2


if __name__ == "__main__":
    raise SystemExit(main())
