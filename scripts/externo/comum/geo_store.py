"""
Camada de armazenamento geoespacial do REV-P: GeoParquet com poda por bbox.

MOTIVO DE EXISTIR (problema medido, nao preferencia de engenharia):
as aquisicoes da frente externa produziram paginas GeoJSON grandes -- 820 MB
em `ext-uk-01-aquisicao`, 296 MB em `ext-uk-04-flood-zones`. Carregar isso de
uma vez estourou a memoria de um ambiente de 3,9 GB durante o EXT-UK-05, e na
maquina de 15,5 GB obriga a ler a Inglaterra inteira so para olhar uma AOI de
5.000 km2. Alem disso, escrever GPKG sobre pasta montada falhou no SQLite
("no such table: gpkg_contents").

GeoParquet resolve os tres de uma vez:
  - colunar e comprimido: o mesmo dado ocupa uma fracao do espaco;
  - le so as colunas pedidas;
  - com colunas de bbox e linhas ORDENADAS POR LOCALIDADE ESPACIAL, o leitor
    descarta grupos de linhas inteiros que nao intersectam a consulta -- e a
    poda espacial de fato, sem precisar de banco.

A ordenacao e a peca essencial e costuma ser esquecida: sem ela, cada grupo de
linhas contem geometrias espalhadas pelo pais todo, o bbox do grupo cobre tudo
e nada e podado. Por isso `converter_*` ordena por celula de 10 km antes de
gravar.

NAO faz: nenhuma alteracao de conteudo. Conversao e reversivel e as paginas
originais permanecem no disco como evidencia de aquisicao.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

CRS_ALVO = 27700          # British National Grid, metrico
CELULA_ORDEM = 10_000     # m, granularidade da ordenacao espacial
LINHAS_POR_GRUPO = 5_000  # tamanho do row group; menor = poda mais fina


def _preparar(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reprojeta, anexa colunas de bbox e ordena por localidade espacial."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame sem CRS -- recusado de proposito")
    if gdf.crs.to_epsg() != CRS_ALVO:
        gdf = gdf.to_crs(CRS_ALVO)

    b = gdf.geometry.bounds  # minx, miny, maxx, maxy
    gdf = gdf.assign(
        bbox_xmin=b["minx"], bbox_ymin=b["miny"],
        bbox_xmax=b["maxx"], bbox_ymax=b["maxy"],
    )
    cx = (b["minx"] + b["maxx"]) / 2
    cy = (b["miny"] + b["maxy"]) / 2
    gdf = gdf.assign(
        _cx=(cx // CELULA_ORDEM).astype("int32"),
        _cy=(cy // CELULA_ORDEM).astype("int32"),
    )
    return gdf.sort_values(["_cy", "_cx"]).drop(columns=["_cx", "_cy"])


def converter_paginas(
    dir_paginas: Path, saida: Path, padrao: str = "page_*.geojson"
) -> dict:
    """Converte paginas GeoJSON em um GeoParquet unico, lendo uma por vez."""
    caminhos = sorted(glob.glob(str(Path(dir_paginas) / padrao)))
    if not caminhos:
        raise FileNotFoundError(f"nenhuma pagina em {dir_paginas}")

    partes: list[gpd.GeoDataFrame] = []
    n_bruto = 0
    for p in caminhos:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        feats = d.get("features", []) or []
        if not feats:
            continue
        n_bruto += len(feats)
        partes.append(gpd.GeoDataFrame.from_features(feats, crs=4326))
        del d, feats

    gdf = gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=4326)
    del partes
    gdf = _preparar(gdf)

    saida.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(saida, index=False, row_group_size=LINHAS_POR_GRUPO)
    return {
        "origem": str(dir_paginas),
        "paginas": len(caminhos),
        "features_lidas": n_bruto,
        "features_gravadas": int(len(gdf)),
        "crs": CRS_ALVO,
        "saida": str(saida),
        "bytes_saida": saida.stat().st_size,
    }


def converter_paginas_particionado(
    dir_paginas: Path, dir_saida: Path, padrao: str = "page_*.geojson"
) -> dict:
    """
    Converte pagina a pagina, uma parte Parquet por pagina, de forma resumivel.

    Existe porque a conversao monolitica de `ext-uk-04` (296 MB) nao cabe no
    limite de tempo de uma execucao e nao cabe confortavelmente na memoria de
    3,9 GB. Particionar tem dois efeitos colaterais bons: a conversao vira
    resumivel (parte ja escrita e pulada) e a poda passa a acontecer tambem no
    nivel de arquivo, nao so de grupo de linhas.
    """
    caminhos = sorted(glob.glob(str(Path(dir_paginas) / padrao)))
    if not caminhos:
        raise FileNotFoundError(f"nenhuma pagina em {dir_paginas}")
    dir_saida.mkdir(parents=True, exist_ok=True)

    feitas, puladas, n_feats = 0, 0, 0
    for p in caminhos:
        parte = dir_saida / (Path(p).stem + ".parquet")
        if parte.exists() and parte.stat().st_size > 0:
            puladas += 1
            continue
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        feats = d.get("features", []) or []
        if not feats:
            continue
        gdf = _preparar(gpd.GeoDataFrame.from_features(feats, crs=4326))
        gdf.to_parquet(parte, index=False, row_group_size=LINHAS_POR_GRUPO)
        n_feats += len(gdf)
        feitas += 1
        del d, feats, gdf

    partes = sorted(dir_saida.glob("*.parquet"))
    return {
        "origem": str(dir_paginas),
        "paginas_total": len(caminhos),
        "partes_escritas_agora": feitas,
        "partes_ja_existentes": puladas,
        "partes_no_disco": len(partes),
        "completo": len(partes) == len(caminhos),
        "features_gravadas_agora": n_feats,
        "bytes_saida": sum(p.stat().st_size for p in partes),
        "saida": str(dir_saida),
    }


def converter_arquivo(entrada: Path, saida: Path) -> dict:
    """Converte um GPKG/GeoJSON unico em GeoParquet ordenado."""
    gdf = gpd.read_file(entrada)
    n = len(gdf)
    gdf = _preparar(gdf)
    saida.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(saida, index=False, row_group_size=LINHAS_POR_GRUPO)
    return {
        "origem": str(entrada),
        "features_lidas": n,
        "features_gravadas": int(len(gdf)),
        "crs": CRS_ALVO,
        "saida": str(saida),
        "bytes_saida": saida.stat().st_size,
    }


def ler_bbox(
    caminho: Path, bbox: tuple[float, float, float, float],
    colunas: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Le so o que intersecta `bbox` (EPSG:27700), podando grupos de linhas.

    O filtro por colunas de bbox e uma intersecao de retangulos: descarta o que
    com certeza nao toca a consulta. O recorte fino de geometria, se necessario,
    e responsabilidade de quem chama -- aqui nao se altera geometria.
    """
    xmin, ymin, xmax, ymax = bbox
    filtro = [
        ("bbox_xmax", ">=", xmin), ("bbox_xmin", "<=", xmax),
        ("bbox_ymax", ">=", ymin), ("bbox_ymin", "<=", ymax),
    ]
    caminho = Path(caminho)
    if caminho.is_dir():
        # store particionado: descarta partes cujo bbox nem toca a consulta,
        # antes de abrir o arquivo
        partes = []
        for p in sorted(caminho.glob("*.parquet")):
            import pyarrow.parquet as pq

            md = pq.read_metadata(p)
            esq = {md.schema.column(i).name: i for i in range(md.num_columns)}
            toca = True
            if all(c in esq for c in ("bbox_xmin", "bbox_xmax", "bbox_ymin", "bbox_ymax")):
                xs0, ys0 = float("inf"), float("inf")
                xs1, ys1 = float("-inf"), float("-inf")
                for g in range(md.num_row_groups):
                    rg = md.row_group(g)
                    xs0 = min(xs0, rg.column(esq["bbox_xmin"]).statistics.min)
                    ys0 = min(ys0, rg.column(esq["bbox_ymin"]).statistics.min)
                    xs1 = max(xs1, rg.column(esq["bbox_xmax"]).statistics.max)
                    ys1 = max(ys1, rg.column(esq["bbox_ymax"]).statistics.max)
                toca = not (xs1 < xmin or xs0 > xmax or ys1 < ymin or ys0 > ymax)
            if toca:
                partes.append(gpd.read_parquet(p, columns=colunas, filters=filtro))
        if not partes:
            return gpd.GeoDataFrame(geometry=[], crs=CRS_ALVO)
        return gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=CRS_ALVO)
    return gpd.read_parquet(caminho, columns=colunas, filters=filtro)
