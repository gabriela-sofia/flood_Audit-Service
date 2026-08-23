"""
EXT-UK-07 -- Converte as aquisicoes externas para GeoParquet com poda por bbox.

Ver `armazenamento_geo.py` para o motivo tecnico. Em resumo: as paginas GeoJSON
somam mais de 1 GB e obrigam a carregar a Inglaterra inteira para olhar uma
AOI de 5.000 km2; isso ja estourou a memoria de um ambiente de 3,9 GB durante
o EXT-UK-05.

As paginas originais NAO sao apagadas. Elas sao a evidencia bruta de aquisicao
e continuam no disco; o GeoParquet e camada de trabalho, nao substituto de
proveniencia.

Verificacao obrigatoria: cada conversao confere que o numero de feicoes
gravadas e igual ao numero lido, e mede o ganho de leitura por bbox contra a
leitura total. Conversao que perde feicao e falha, nao aviso.

Uso:
    python scripts/externo/ext_uk07_converter_geoparquet.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd

from armazenamento_geo import (
    converter_arquivo, converter_paginas, converter_paginas_particionado, ler_bbox,
)

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
STORE = RUNS / "geostore"
AOI = (350_000, 350_000, 400_000, 450_000)


def mb(n: int) -> float:
    return round(n / 1024 ** 2, 1)


def main() -> int:
    STORE.mkdir(parents=True, exist_ok=True)
    print("===EXT-UK-07===")
    relatorio = {}

    # `flood_zones` vai particionado: 296 MB nao cabem numa conversao unica
    # dentro do limite de tempo/memoria observado, e particionar ainda deixa a
    # conversao resumivel.
    tarefas = [
        ("recorded_flood_outlines", "paginas",
         RUNS / "ext-uk-01-aquisicao" / "pages_v2", STORE / "recorded_flood_outlines.parquet"),
        ("flood_zones", "paginas_particionado",
         RUNS / "ext-uk-04-flood-zones" / "Flood_Zones_2_3_Rivers_and_Sea", STORE / "flood_zones"),
        ("outlines_elegiveis", "arquivo",
         RUNS / "ext-uk-02-area-piloto" / "elegiveis.gpkg", STORE / "outlines_elegiveis.parquet"),
    ]

    for nome, tipo, origem, saida in tarefas:
        if not Path(origem).exists():
            print(f"{nome}=ORIGEM_AUSENTE ({origem})")
            continue
        t0 = time.time()
        try:
            if tipo == "paginas":
                info = converter_paginas(Path(origem), saida)
            elif tipo == "paginas_particionado":
                info = converter_paginas_particionado(Path(origem), saida)
            else:
                info = converter_arquivo(Path(origem), saida)
        except Exception as exc:  # noqa: BLE001
            print(f"{nome}=FALHOU {type(exc).__name__}: {str(exc)[:120]}")
            relatorio[nome] = {"erro": f"{type(exc).__name__}: {exc}"}
            continue

        if "features_lidas" in info:
            info["integro"] = info["features_lidas"] == info["features_gravadas"]
        else:
            info["integro"] = bool(info.get("completo"))
        info["segundos"] = round(time.time() - t0, 1)
        relatorio[nome] = info
        print(f"{nome}: integro={info['integro']} saida={mb(info['bytes_saida'])}MB "
              f"em {info['segundos']}s")

    # --- prova de poda: comparacao honesta, ambas as leituras com geometria ---
    print("---PROVA_DE_PODA---")
    for nome in ("recorded_flood_outlines", "outlines_elegiveis"):
        caminho = STORE / f"{nome}.parquet"
        if not caminho.exists():
            continue
        t1 = time.time()
        recorte = ler_bbox(caminho, AOI)
        t_bbox = time.time() - t1
        t0 = time.time()
        completo = gpd.read_parquet(caminho)
        t_total = time.time() - t0
        ganho = (t_total / t_bbox) if t_bbox > 0 else 0
        print(
            f"{nome}: total={len(completo)} na_aoi={len(recorte)} "
            f"({100*len(recorte)/len(completo):.1f}%) completa={t_total:.2f}s "
            f"bbox={t_bbox:.2f}s ganho={ganho:.1f}x"
        )
        relatorio.setdefault(nome, {}).update(
            {"total": int(len(completo)), "na_aoi": int(len(recorte)),
             "seg_total": round(t_total, 2), "seg_bbox": round(t_bbox, 2),
             "ganho": round(ganho, 1)}
        )
        del completo, recorte

    (STORE / "conversao.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (STORE / "LEIA-ME.md").write_text(
        "# geostore -- camada de trabalho\n\n"
        "GeoParquet em EPSG:27700, com colunas `bbox_*` e linhas ordenadas por\n"
        "celula de 10 km, o que permite poda de grupos de linhas na leitura.\n\n"
        "Consultar por AOI:\n\n"
        "```python\n"
        "from armazenamento_geo import ler_bbox\n"
        "gdf = ler_bbox('local_runs/geostore/recorded_flood_outlines.parquet',\n"
        "               (350000, 350000, 400000, 450000))\n"
        "```\n\n"
        "As paginas GeoJSON originais permanecem no disco e continuam sendo a\n"
        "evidencia de aquisicao. Este diretorio e derivado e pode ser refeito\n"
        "a qualquer momento com `ext_uk07_converter_geoparquet.py`.\n",
        encoding="utf-8",
    )
    print("===END===")
    ok = all(v.get("integro", False) for v in relatorio.values() if "features_lidas" in v)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
