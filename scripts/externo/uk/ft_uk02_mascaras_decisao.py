"""
FT-UK-02 -- Rasteriza as mascaras que implementam o criterio N1-N4.

Implementa, na grade de 30 m do FT-UK-01, o criterio escrito em
`docs/metodologia_cientifica/ext_uk_adjudicacao_negativo_v1.md`:

  N1  fora de TODO Recorded Flood Outline, sem restricao de data.
      Sem restricao de data e proposital: se inundou em 1953, nao serve como
      negativo hoje. Usar so os elegiveis (>=2000) contaminaria a classe.
  N2  fora de toda Flood Zone modelada (FZ2 e FZ3).
  N3  a pelo menos 400 m de qualquer area N1/N2.
  N4  pareado por contexto construido -- aqui entra o ESA WorldCover.

E aplica tambem a regra R3 do contrato de reducao a pontos: agua permanente
NAO e inundacao. O JRC Global Surface Water remove rio, lago e reservatorio
antes de qualquer amostragem.

SAIDAS (todas na grade 3333x1666, EPSG:27700, 30 m):
  mask_rfo_qualquer.tif   1 = dentro de outline registrado, qualquer data
  mask_rfo_elegivel.tif   1 = dentro de outline elegivel (com data, >=2000,
                              nao costeiro, nao de mare)
  mask_fz.tif             1 = dentro de Flood Zone modelada
  dist_inundavel.tif      distancia em metros ate a area N1/N2 mais proxima
  worldcover.tif          classe ESA WorldCover reamostrada (vizinho mais proximo)
  agua_permanente.tif     1 = ocorrencia de agua JRC > 50%
  classe_adjudicacao.tif  0=indefinido 1=POS_CAND 2=EXCLUIDO 3=NEG_CAND

NAO faz: nao amostra ponto, nao promove rotulo, nao extrai feature de modelo.
Produz as mascaras que tornam a amostragem auditavel.

Uso:
    python scripts/externo/ft_uk02_mascaras_decisao.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_store import ler_bbox  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
STORE = RUNS / "geostore"
FEAT = RUNS / "aq-features"
RAST = RUNS / "ft-uk-01-rasters"
OUT = RUNS / "ft-uk-02-mascaras"

AOI_BNG = (350_000, 350_000, 400_000, 450_000)
BUFFER_M = 400          # criterio N3
AGUA_LIMIAR = 50        # ocorrencia JRC em % acima da qual e agua permanente
WC_CONSTRUIDO = 50      # codigo ESA WorldCover para Built-up


def gravar(caminho: Path, arr: np.ndarray, perfil: dict) -> None:
    """Grava tolerando mount sem permissao de unlink (ver FT-UK-01)."""
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "s.tif"
        with rasterio.open(tmp, "w", **perfil) as dst:
            dst.write(arr, 1)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "rb") as o, open(caminho, "wb") as d:
            shutil.copyfileobj(o, d)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===FT-UK-02===")

    with rasterio.open(RAST / "elevation_m.tif") as src:
        forma = (src.height, src.width)
        transform = src.transform
        perfil_base = src.profile.copy()
    perfil_u8 = {**perfil_base, "dtype": "uint8", "nodata": 255}
    perfil_f32 = {**perfil_base, "dtype": "float32", "nodata": np.nan}
    print(f"GRADE={forma} transform_res={transform.a}m")

    # ---------------- N1: outlines, qualquer data ----------------
    rfo = ler_bbox(STORE / "recorded_flood_outlines.parquet", AOI_BNG)
    m_rfo = rasterize(
        ((g, 1) for g in rfo.geometry if g is not None and not g.is_empty),
        out_shape=forma, transform=transform, fill=0, dtype="uint8",
        all_touched=True,
    )
    print(f"RFO_QUALQUER poligonos={len(rfo)} celulas={int(m_rfo.sum()):,} "
          f"({100*m_rfo.mean():.2f}% da AOI)")
    gravar(OUT / "mask_rfo_qualquer.tif", m_rfo, perfil_u8)

    # elegiveis, separados por tema -- base dos positivos
    eleg = ler_bbox(STORE / "outlines_elegiveis.parquet", AOI_BNG)
    m_eleg = rasterize(
        ((g, 1) for g in eleg.geometry if g is not None and not g.is_empty),
        out_shape=forma, transform=transform, fill=0, dtype="uint8",
        all_touched=True,
    )
    print(f"RFO_ELEGIVEL poligonos={len(eleg)} celulas={int(m_eleg.sum()):,} "
          f"({100*m_eleg.mean():.2f}%)")
    gravar(OUT / "mask_rfo_elegivel.tif", m_eleg, perfil_u8)

    for tema in ("fluvial", "pluvial"):
        sub = eleg[eleg["tema"] == tema]
        if len(sub) == 0:
            continue
        m = rasterize(
            ((g, 1) for g in sub.geometry if g is not None and not g.is_empty),
            out_shape=forma, transform=transform, fill=0, dtype="uint8",
            all_touched=True,
        )
        print(f"  tema={tema} poligonos={len(sub)} celulas={int(m.sum()):,}")
        gravar(OUT / f"mask_pos_{tema}.tif", m, perfil_u8)

    # ---------------- N2: flood zones modeladas ----------------
    fz = ler_bbox(STORE / "flood_zones", AOI_BNG)
    m_fz = rasterize(
        ((g, 1) for g in fz.geometry if g is not None and not g.is_empty),
        out_shape=forma, transform=transform, fill=0, dtype="uint8",
        all_touched=True,
    )
    print(f"FLOOD_ZONE poligonos={len(fz)} celulas={int(m_fz.sum()):,} "
          f"({100*m_fz.mean():.2f}%)")
    gravar(OUT / "mask_fz.tif", m_fz, perfil_u8)

    # ---------------- N3: distancia ate area inundavel ----------------
    from scipy.ndimage import distance_transform_edt

    inundavel = (m_rfo > 0) | (m_fz > 0)
    dist = distance_transform_edt(~inundavel, sampling=transform.a).astype("float32")
    print(f"DIST_INUNDAVEL mediana={np.median(dist):.0f}m p90={np.percentile(dist,90):.0f}m "
          f"max={dist.max():.0f}m")
    gravar(OUT / "dist_inundavel.tif", dist, perfil_f32)

    # ---------------- N4: contexto construido ----------------
    import glob

    wc = np.zeros(forma, dtype="uint8")
    for tif in sorted(glob.glob(str(FEAT / "worldcover" / "*.tif"))):
        with rasterio.open(tif) as src:
            tmp = np.zeros(forma, dtype="uint8")
            reproject(
                source=rasterio.band(src, 1), destination=tmp,
                dst_transform=transform, dst_crs=perfil_base["crs"],
                resampling=Resampling.nearest, src_nodata=0, dst_nodata=0,
            )
            wc = np.where(tmp > 0, tmp, wc)
    classes, contagens = np.unique(wc, return_counts=True)
    print("WORLDCOVER=" + "; ".join(
        f"{int(c)}:{100*n/wc.size:.1f}%" for c, n in zip(classes, contagens) if n > 0))
    gravar(OUT / "worldcover.tif", wc, perfil_u8)

    # ---------------- R3: agua permanente ----------------
    agua = np.zeros(forma, dtype="uint8")
    gsw = sorted(glob.glob(str(FEAT / "gsw" / "*.tif")))
    if gsw:
        with rasterio.open(gsw[0]) as src:
            occ = np.zeros(forma, dtype="uint8")
            reproject(
                source=rasterio.band(src, 1), destination=occ,
                dst_transform=transform, dst_crs=perfil_base["crs"],
                resampling=Resampling.bilinear, src_nodata=255, dst_nodata=0,
            )
        agua = (occ > AGUA_LIMIAR).astype("uint8")
    print(f"AGUA_PERMANENTE celulas={int(agua.sum()):,} ({100*agua.mean():.2f}%)")
    gravar(OUT / "agua_permanente.tif", agua, perfil_u8)

    # ---------------- classe de adjudicacao ----------------
    classe = np.zeros(forma, dtype="uint8")
    classe[m_eleg > 0] = 1                                   # POS_CAND
    ainda = classe == 0
    classe[ainda & inundavel] = 2                             # EXCLUIDO
    ainda = classe == 0
    neg = ainda & (dist >= BUFFER_M) & (agua == 0)
    classe[neg] = 3                                           # NEG_CAND
    # o que sobrou (ainda==0 e nao neg) fica 0 = indefinido: fora de tudo mas
    # dentro do buffer de 400 m, ou agua permanente. Nao e negativo nem positivo.
    gravar(OUT / "classe_adjudicacao.tif", classe, perfil_u8)

    total = classe.size
    cont = {int(k): int(v) for k, v in zip(*np.unique(classe, return_counts=True))}
    nomes = {0: "INDEFINIDO", 1: "POS_CAND", 2: "EXCLUIDO", 3: "NEG_CAND"}
    for k in (1, 2, 3, 0):
        v = cont.get(k, 0)
        print(f"{nomes[k]}={v:,} ({100*v/total:.1f}%) ~{v*(transform.a**2)/1e6:.0f}km2")

    # composicao de cobertura do solo por classe -- insumo do pareamento N4
    print("---COMPOSICAO_WORLDCOVER_POR_CLASSE---")
    comp = {}
    for k in (1, 3):
        sel = classe == k
        cls, n = np.unique(wc[sel], return_counts=True)
        d = {int(c): round(100 * x / max(sel.sum(), 1), 1) for c, x in zip(cls, n)}
        comp[nomes[k]] = d
        print(f"{nomes[k]}: " + "; ".join(f"{c}:{p}%" for c, p in sorted(d.items())))
    constr_pos = comp.get("POS_CAND", {}).get(WC_CONSTRUIDO, 0)
    constr_neg = comp.get("NEG_CAND", {}).get(WC_CONSTRUIDO, 0)
    print(f"CONSTRUIDO pos={constr_pos}% neg={constr_neg}% "
          f"razao={(constr_pos/constr_neg if constr_neg else float('inf')):.2f}")

    (OUT / "resumo.json").write_text(json.dumps({
        "grade": list(forma), "res_m": transform.a, "buffer_m": BUFFER_M,
        "agua_limiar_pct": AGUA_LIMIAR,
        "poligonos": {"rfo_qualquer": len(rfo), "rfo_elegivel": len(eleg),
                      "flood_zone": len(fz)},
        "classes": {nomes[k]: cont.get(k, 0) for k in (0, 1, 2, 3)},
        "composicao_worldcover": comp,
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"TEMPO={round(time.time()-t0,1)}s")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
