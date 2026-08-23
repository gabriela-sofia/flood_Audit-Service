"""
EXT-UK-04 -- Aquisicao do Flood Map for Planning (Flood Zones) na AOI do piloto.

AOI definida em EXT-UK-03, por evidencia (nao por escolha):
    janelas 50km `7_7` + `7_8`, contiguas, que sao #1 e #2 tanto no ranking
    geral quanto no ranking pluvial isolado.
    BNG (EPSG:27700): E 350.000-400.000 / N 350.000-450.000
    Noroeste da Inglaterra -- eixo Manchester-Warrington-Wigan-Bolton-Blackburn.
    262 eventos distintos, 1.517 outlines, ~158 datas, 2000-2025.

PARA QUE SERVE ESTA CAMADA -- ler com atencao, porque o uso errado invalidaria
o trabalho inteiro:

  As Flood Zones sao MODELADAS, nao observadas. Elas NAO sao rotulo positivo e
  NAO sao evidencia de inundacao ocorrida. O uso pretendido aqui e o inverso:
  servir de FILTRO DE CONTAMINACAO para a construcao de negativos.

  O problema real: a propria Environment Agency documenta que "a ausencia de
  Recorded Flood Outline numa area nao significa que ela nunca inundou, apenas
  que nao temos registro". Logo, tratar toda area sem outline como negativo
  contamina a classe negativa com area que inundou e nao foi registrada.

  A mitigacao: exigir do candidato a negativo duas condicoes simultaneas --
  (a) fora de qualquer Recorded Flood Outline, e (b) fora da Flood Zone
  modelada. A condicao (b) remove justamente a area que e fisicamente
  plausivel de inundar mas nao tem registro, que e a fonte da contaminacao.

  Isso NAO promove nada a negativo. Apenas adquire a camada que tornara essa
  decisao possivel e auditavel depois, em tarefa propria.

Fonte  : Environment Agency / Defra Data Services Platform
Dataset: 04532375-a198-476e-985e-0579a0a11b47 (Flood Map for Planning - Flood Zones)
Licenca: Open Government Licence v3.0

Uso:
    python scripts/externo/ext_uk04_zonas_inundacao_aoi.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from pyproj import Transformer

DATASET = "04532375-a198-476e-985e-0579a0a11b47"
ROOT = (
    f"https://environment.data.gov.uk/geoservices/datasets/{DATASET}"
    "/ogc/features/v1"
)
REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "local_runs" / "ext-uk-04-flood-zones"

# AOI em British National Grid (EPSG:27700)
AOI_BNG = (350_000, 350_000, 400_000, 450_000)
TIMEOUT = 300
MAX_RETRIES = 5
PAGE_SIZE = 500
MAX_PAGES = 400


def get_json(url: str, params: dict | None = None) -> dict | None:
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    print(f"  FALHA {url}: {type(last).__name__}")
    return None


def next_link(payload: dict) -> str | None:
    for link in payload.get("links", []) or []:
        if link.get("rel") == "next" and link.get("href"):
            return link["href"]
    return None


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # AOI para CRS84, que e o que o parametro bbox da OGC API espera
    tr = Transformer.from_crs(27700, 4326, always_xy=True)
    x0, y0, x1, y1 = AOI_BNG
    lon0, lat0 = tr.transform(x0, y0)
    lon1, lat1 = tr.transform(x1, y1)
    bbox = f"{lon0:.6f},{lat0:.6f},{lon1:.6f},{lat1:.6f}"

    print("===EXT-UK-04===")
    print(f"AOI_BNG={AOI_BNG}")
    print(f"AOI_WGS84={bbox}")

    # --- 1. descobrir colecoes ---
    cols = get_json(f"{ROOT}/collections", {"f": "application/json"})
    if not cols:
        print("COLECOES=FALHOU")
        print("===END===")
        return 1
    ids = [c.get("id") for c in cols.get("collections", [])]
    print("COLECOES=" + "; ".join(str(i) for i in ids))
    (OUTDIR / "collections.json").write_text(
        json.dumps(cols, indent=2), encoding="utf-8"
    )

    # --- 2. baixar cada colecao restrita a AOI ---
    resumo = {}
    for cid in ids:
        if not cid:
            continue
        base = f"{ROOT}/collections/{cid}/items"
        head = get_json(base, {"limit": 1, "bbox": bbox, "f": "application/json"})
        if head is None:
            resumo[cid] = {"erro": "head_falhou"}
            continue
        total = head.get("numberMatched")
        props = {}
        if head.get("features"):
            props = head["features"][0].get("properties", {}) or {}
        print(f"---{cid}--- na_aoi={total} campos={list(props.keys())[:10]}")

        subdir = OUTDIR / str(cid)
        subdir.mkdir(exist_ok=True)
        url: str | None = base
        params: dict | None = {"limit": PAGE_SIZE, "bbox": bbox, "f": "application/json"}
        n_pag, n_feat = 0, 0
        while url and n_pag < MAX_PAGES:
            payload = get_json(url, params)
            params = None
            if not payload:
                break
            feats = payload.get("features", []) or []
            if not feats:
                break
            (subdir / f"page_{n_pag:04d}.geojson").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            n_pag += 1
            n_feat += len(feats)
            url = next_link(payload)

        resumo[cid] = {
            "numberMatched": total,
            "paginas": n_pag,
            "features_salvas": n_feat,
            "campos": list(props.keys()),
        }
        print(f"  BAIXADO paginas={n_pag} features={n_feat}")

    (OUTDIR / "resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (OUTDIR / "PROVENIENCIA.md").write_text(
        "# Proveniencia - Flood Map for Planning (Flood Zones)\n"
        "- Fonte: Environment Agency / Defra Data Services Platform\n"
        f"- Dataset ID: {DATASET}\n"
        "- Endpoint: OGC API Features v1\n"
        f"- Data de acesso: {time.strftime('%Y-%m-%d %H:%M')}\n"
        "- Licenca: Open Government Licence v3.0\n"
        "- Atribuicao: (c) Environment Agency copyright and/or database right\n"
        f"- Recorte: AOI BNG {AOI_BNG} (EXT-UK-03)\n"
        "- NATUREZA DO DADO: MODELADO, nao observado.\n"
        "- Uso pretendido: FILTRO DE CONTAMINACAO para construcao de negativo.\n"
        "- Uso PROIBIDO: como rotulo positivo, como evidencia de inundacao\n"
        "  ocorrida, ou como variavel do modelo.\n"
        "- Status: AQUISICAO_EXPLORATORIA - nenhum negativo promovido.\n",
        encoding="utf-8",
    )
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
