"""Rodada 3: transporte alternativo do MESMO FABDEM V1-2 (espelho GEE `projects/sat-io/open-datasets/FABDEM`) quando a
origem data.bris.ac.uk recusa conexao. Nao muda produto nem cadeia: so o caminho dos bytes.

  conferir   le no GEE a mesma janela de um DEM ja baixado da origem (`local_runs/mdt-fabdem/*_fabdem_4326.tif`) na
             grade nativa (1 arcsec, EPSG:4326, mesmo transform) e compara pixel a pixel; so habilita o transporte se
             as diferencas forem nulas
  baixar     grava `<nome>_fabdem_4326.tif` com o mesmo grid que `baixar_fabdem.baixar` produziria para o plano
             (mesma bbox, mesma resolucao, mesmo nodata), em blocos para respeitar o limite do computePixels

Uso:
    python scripts/ia/rodada3_fabdem_gee.py conferir [--ref EMSR680_AOI01]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for s in ("servico", "terreno", "sar"):
    sys.path.insert(0, str(REPO / "scripts" / s))
from proveniencia import agora_utc, sha256_arquivo  # noqa: E402

COLECAO = "projects/sat-io/open-datasets/FABDEM"
SAIDA = REPO / "local_runs" / "confirmacao-rodada3" / "v1" / "confirmacao"
BLOCO = 2048
NODATA = -9999.0


def _ee():
    import warnings
    warnings.filterwarnings("ignore", message=".*restricted mode.*")
    import ee
    ee.Initialize()
    return ee


def ler_janela(ee, transform, largura: int, altura: int) -> np.ndarray:
    """Grade identica (EPSG:4326, mesmo transform afim), em blocos; NODATA fora da cobertura."""
    img = ee.ImageCollection(COLECAO).mosaic().select(0).unmask(NODATA).toFloat()
    out = np.full((altura, largura), NODATA, np.float32)
    for i0 in range(0, altura, BLOCO):
        for j0 in range(0, largura, BLOCO):
            h, w = min(BLOCO, altura - i0), min(BLOCO, largura - j0)
            req = {"expression": img, "fileFormat": "NPY",
                   "grid": {"dimensions": {"width": w, "height": h},
                            "affineTransform": {"scaleX": transform.a, "shearX": 0, "translateX": transform.c + j0 * transform.a,
                                                "shearY": 0, "scaleY": transform.e, "translateY": transform.f + i0 * transform.e},
                            "crsCode": "EPSG:4326"}}
            a = np.load(io.BytesIO(ee.data.computePixels(req)))
            out[i0:i0 + h, j0:j0 + w] = a[a.dtype.names[0]] if a.dtype.names else a
    return out


def conferir(ref: str = "EMSR680_AOI01") -> dict:
    import rasterio
    ee = _ee()
    arq = REPO / "local_runs" / "mdt-fabdem" / f"{ref}_fabdem_4326.tif"
    with rasterio.open(arq) as r:
        a = r.read(1)
        tr, (H, W) = r.transform, r.shape
    b = ler_janela(ee, tr, W, H)
    validos = (a != NODATA) & (b != NODATA)
    d = np.abs(a[validos].astype(np.float64) - b[validos])
    res = {"tipo": "conferencia_transporte_fabdem_gee", "quando_utc": agora_utc(), "referencia": str(arq.relative_to(REPO)),
           "referencia_sha256": sha256_arquivo(arq), "colecao": COLECAO, "pixels": int(a.size),
           "validos_ambos": int(validos.sum()), "nodata_divergente": int(((a == NODATA) != (b == NODATA)).sum()),
           "max_abs": float(d.max()) if d.size else None, "pixels_diferentes": int((d > 0).sum()),
           "identico": bool(d.size and d.max() == 0 and ((a == NODATA) == (b == NODATA)).all())}
    (SAIDA / "fabdem_gee_conferencia.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def baixar(plano: dict, saida: Path) -> dict:
    """Substitui `baixar_fabdem.baixar` com o mesmo grid nativo: origem no canto NW da bbox alinhado a 1 arcsec."""
    import rasterio
    from rasterio.transform import from_origin
    res, grude = 1 / 3600, 1e-6  # mesma grade e mesmo arredondamento "para fora" de baixar_fabdem.baixar
    o, s, l, n = plano["bbox_4326"]
    # bordas da grade FABDEM em (k - 1/2) arco-segundo (centro do pixel no arco-segundo inteiro)
    c0, c1 = int(np.floor(o / res + .5 + grude)), int(np.ceil(l / res + .5 - grude))
    r_s, r_n = int(np.floor(s / res + .5 + grude)), int(np.ceil(n / res + .5 - grude))
    W, H = c1 - c0, r_n - r_s
    tr = from_origin((c0 - .5) * res, (r_n - .5) * res, res, res)
    a = ler_janela(_ee(), tr, W, H)
    if (a == NODATA).all():
        raise SystemExit(f"{plano['nome']}: FABDEM GEE sem cobertura na bbox")
    saida.mkdir(parents=True, exist_ok=True)
    arq = saida / f"{plano['nome']}_fabdem_4326.tif"
    with rasterio.open(arq, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32", crs="EPSG:4326",
                       transform=tr, nodata=NODATA, compress="deflate") as d:
        d.write(a, 1)
    man = {**plano, "transporte": "GEE " + COLECAO, "conferencia": str((SAIDA / "fabdem_gee_conferencia.json").relative_to(REPO))}
    arq.with_name(f"{plano['nome']}_fabdem_manifesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"arquivo": str(arq), "shape": [H, W]}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["conferir"])
    ap.add_argument("--ref", default="EMSR680_AOI01")
    a = ap.parse_args()
    print(json.dumps(conferir(a.ref), ensure_ascii=False, indent=1))
