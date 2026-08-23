"""
EXT-UK-01 -- Aquisicao do rotulo externo: Recorded Flood Outlines (Environment Agency).

Objetivo desta etapa (aquisicao/organizacao apenas -- NAO e modelagem):
baixar integralmente o inventario de inundacoes REGISTRADAS da Inglaterra
(~31.696 poligonos) via OGC API Features, e produzir um inventario tabular
auditavel que permita decidir, com numero real, quais registros sao
elegiveis como POSITIVO datado para a linha causal SUSC.

Por que esta base: e a unica fonte publica encontrada no levantamento de
2026-08-06 que combina, na origem, (a) poligono de inundacao ocorrida,
(b) data de inicio/fim, e (c) separacao explicita de mecanismo
(`fluvial_f` / `coastal_f` / `tidal_f` + `flood_src` + `flood_caus`).

NAO faz: nenhuma extracao de feature, nenhum treino, nenhuma promocao de
negativo. Ausencia de poligono AQUI NAO E NEGATIVO -- a propria Environment
Agency documenta que ausencia de registro nao significa ausencia de
inundacao. A construcao de negativo e tarefa posterior e separada.

--------------------------------------------------------------------------
HISTORICO DE CORRECAO (2026-08-06, mesma sessao)
A primeira versao paginava com o parametro `offset`. O servidor da Defra
IGNORA `offset` silenciosamente: devolveu a mesma primeira pagina 32 vezes
(32.000 registros lidos contra 31.696 anunciados, e todas as contagens por
categoria saindo como multiplos exatos de 32 -- assinatura inequivoca de
duplicacao). Esta versao pagina seguindo o link `rel="next"` devolvido pela
propria API (mecanismo padrao do OGC API Features, nao burlavel pelo
servidor) e deduplica por `rec_out_id`, com verificacao dura no final.
--------------------------------------------------------------------------

Fonte    : Environment Agency / Defra Data Services Platform
Dataset  : 8c75e700-d465-11e4-8b5b-f0def148f590
Licenca  : Open Government Licence v3.0
Atribuic.: (c) Environment Agency copyright and/or database right

Uso:
    conda activate revp-susc
    python scripts/externo/ext_uk01_download_recorded_flood_outlines.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import requests

BASE = (
    "https://environment.data.gov.uk/geoservices/datasets/"
    "8c75e700-d465-11e4-8b5b-f0def148f590/ogc/features/v1/"
    "collections/Recorded_Flood_Outlines/items"
)
REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "local_runs" / "ext-uk-01-aquisicao"
PAGEDIR = OUTDIR / "pages_v2"
PAGE_SIZE = 1000
MAX_RETRIES = 5
TIMEOUT = 300
MAX_PAGES = 500  # trava de seguranca contra loop infinito de `next`

FIELDS = [
    "rec_out_id", "rec_grp_id", "name", "start_date", "end_date",
    "flood_src", "flood_caus", "fm_status", "hfm_status", "data_src",
    "fluvial_f", "coastal_f", "tidal_f", "data_prov", "data_qual",
]


def get_json(url: str, params: dict | None = None) -> dict:
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"falha em {url}: {last}")


def next_link(payload: dict) -> str | None:
    for link in payload.get("links", []) or []:
        if link.get("rel") == "next" and link.get("href"):
            tipo = (link.get("type") or "").lower()
            if "json" in tipo or not tipo:
                return link["href"]
    return None


def main() -> int:
    # pages_v2 sempre do zero: a v1 continha duplicatas e nao e reaproveitavel
    if PAGEDIR.exists():
        shutil.rmtree(PAGEDIR)
    PAGEDIR.mkdir(parents=True, exist_ok=True)

    head = get_json(BASE, {"limit": 1, "f": "application/json"})
    total = int(head.get("numberMatched") or 0)
    if total == 0:
        print("ERRO: numberMatched=0")
        return 1

    url: str | None = BASE
    params: dict | None = {"limit": PAGE_SIZE, "f": "application/json"}
    n_paginas = 0

    while url and n_paginas < MAX_PAGES:
        payload = get_json(url, params)
        params = None  # os links `next` ja vem com querystring completa
        feats = payload.get("features", []) or []
        if not feats:
            break
        (PAGEDIR / f"page_{n_paginas:04d}.geojson").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        n_paginas += 1
        print(f"  pagina {n_paginas} feats={len(feats)}", file=sys.stderr)
        url = next_link(payload)

    # --- consolidacao com deduplicacao por rec_out_id ---
    vistos: set[str] = set()
    rows: list[dict] = []
    brutos = 0
    for page_path in sorted(PAGEDIR.glob("page_*.geojson")):
        payload = json.loads(page_path.read_text(encoding="utf-8"))
        for feat in payload.get("features", []) or []:
            props = feat.get("properties", {}) or {}
            rid = str(props.get("rec_out_id"))
            brutos += 1
            if rid in vistos:
                continue
            vistos.add(rid)
            rows.append({k: props.get(k) for k in FIELDS})

    csv_path = OUTDIR / "recorded_flood_outlines_atributos.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # --- verificacao dura: unico ~= numberMatched ---
    cobertura = len(rows) / total if total else 0.0
    integridade = "OK" if cobertura >= 0.99 else "FALHA_COBERTURA"
    duplicatas = brutos - len(rows)

    def dist(field: str, top: int = 12) -> list[tuple[str, int]]:
        return Counter(str(r.get(field)) for r in rows).most_common(top)

    anos = Counter(
        (r["start_date"] or "")[:4] for r in rows if (r["start_date"] or "")[:4].isdigit()
    )
    sem_data = sum(1 for r in rows if not r.get("start_date"))
    mecanismo = Counter(
        f"F={r.get('fluvial_f')}|C={r.get('coastal_f')}|T={r.get('tidal_f')}" for r in rows
    )

    # recorte candidato a positivo: fluvial/pluvial puro, com data, sem mar/mare
    def puro_terrestre(r: dict) -> bool:
        return (
            str(r.get("coastal_f")) == "False"
            and str(r.get("tidal_f")) == "False"
            and bool(r.get("start_date"))
        )

    elegiveis = [r for r in rows if puro_terrestre(r)]
    elegiveis_2000 = [r for r in elegiveis if (r["start_date"] or "")[:4] >= "2000"]

    inventario = {
        "total_api": total,
        "features_brutas": brutos,
        "unicos": len(rows),
        "duplicatas_descartadas": duplicatas,
        "cobertura": round(cobertura, 4),
        "integridade": integridade,
        "paginas": n_paginas,
        "sem_start_date": sem_data,
        "ano_min": min(anos) if anos else None,
        "ano_max": max(anos) if anos else None,
        "flood_src": dist("flood_src"),
        "flood_caus": dist("flood_caus"),
        "mecanismo": mecanismo.most_common(12),
        "por_decada": sorted(Counter(a[:3] + "0" for a in anos.elements()).items()),
        "elegiveis_terrestres_com_data": len(elegiveis),
        "elegiveis_desde_2000": len(elegiveis_2000),
    }
    (OUTDIR / "inventario.json").write_text(
        json.dumps(inventario, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    gpkg_status = "NAO_TENTADO"
    try:
        import geopandas as gpd
        import pandas as pd

        partes = [gpd.read_file(p) for p in sorted(PAGEDIR.glob("page_*.geojson"))]
        gdf = gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=partes[0].crs)
        gdf = gdf.drop_duplicates(subset="rec_out_id")
        gdf.to_file(OUTDIR / "recorded_flood_outlines.gpkg", driver="GPKG")
        gpkg_status = f"OK n={len(gdf)} crs={gdf.crs}"
    except Exception as exc:  # noqa: BLE001
        gpkg_status = f"FALHOU: {type(exc).__name__}: {str(exc)[:120]}"

    print("===EXT-UK-01===")
    print(f"INTEGRIDADE={integridade} cobertura={cobertura:.3f}")
    print(f"TOTAL_API={total} BRUTOS={brutos} UNICOS={len(rows)} DUPS={duplicatas}")
    print(f"PAGINAS={n_paginas} SEM_DATA={sem_data} ANOS={inventario['ano_min']}..{inventario['ano_max']}")
    print("FLOOD_SRC=" + "; ".join(f"{k}:{v}" for k, v in inventario["flood_src"]))
    print("FLOOD_CAUS=" + "; ".join(f"{k}:{v}" for k, v in inventario["flood_caus"]))
    print("MECANISMO=" + "; ".join(f"{k}:{v}" for k, v in inventario["mecanismo"]))
    print("DECADA=" + "; ".join(f"{k}:{v}" for k, v in inventario["por_decada"]))
    print(f"ELEGIVEIS_TERRESTRES={len(elegiveis)} DESDE_2000={len(elegiveis_2000)}")
    print(f"GPKG={gpkg_status}")
    print("===END===")
    return 0 if integridade == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
