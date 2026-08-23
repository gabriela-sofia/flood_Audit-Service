"""
FT-UK-01 -- Prepara a grade metrica da AOI e deriva declividade e TWI.

ENTRADAS (ja no disco, adquiridas por aq_master.py):
  aq-features/dem_glo30/   9 tiles Copernicus GLO-30, EPSG:4326
  aq-features/hand_glo30/  9 tiles Global 30m HAND, EPSG:4326

POR QUE REPROJETAR ANTES DE QUALQUER COISA:
os tiles vem em graus decimais. Calcular declividade sobre grade em graus da
resultado errado, porque um grau de longitude nao mede o mesmo que um grau de
latitude -- na latitude da AOI (53N) a diferenca passa de 40%. Toda derivada
topografica e calculada aqui SOMENTE depois de reprojetar para EPSG:27700
(British National Grid, metrico), na resolucao nativa de 30 m.

O QUE E DERIVADO AQUI:
  elevation_m  -- direto do DEM reprojetado
  slope_deg    -- gradiente em graus, calculado na grade metrica
  twi_dinf     -- ln(a / tan(beta)), com area de contribuicao por D-infinity
  hand_m       -- do produto global, NAO recalculado

RESSALVA METODOLOGICA QUE PRECISA APARECER NO ARTIGO:
no v12 brasileiro, `hand_m_dinf` foi calculado pela autora com roteamento
D-infinity. Aqui o HAND vem PRONTO do produto global de 30 m, que usa a sua
propria cadeia de derivacao. As duas grandezas medem a mesma coisa
conceitualmente -- altura acima da drenagem mais proxima -- mas nao sao o
mesmo calculo. Isso e uma diferenca real entre as regioes e deve ser
declarada, nao dissimulada. O TWI, esse sim, e calculado aqui com D-infinity,
igual ao v12.

NAO faz: nao amostra ponto, nao monta dataset, nao treina.

Uso:
    python scripts/externo/ft_uk01_preparar_rasters_aoi.py
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
FEAT = RUNS / "aq-features"
OUT = RUNS / "ft-uk-01-rasters"

AOI_BNG = (350_000, 350_000, 400_000, 450_000)
AOI_LL = (-3.25, 52.60, -1.50, 54.40)   # com folga, para nao truncar borda
RES = 30.0                               # metros
CRS_ALVO = "EPSG:27700"


def mosaico_recortado(padrao: str) -> tuple[np.ndarray, dict]:
    """Junta os tiles e recorta na AOI, ainda em graus."""
    arquivos = sorted(glob.glob(str(FEAT / padrao)))
    if not arquivos:
        raise FileNotFoundError(padrao)
    fontes = [rasterio.open(a) for a in arquivos]
    try:
        arr, transform = merge(fontes, bounds=AOI_LL)
        perfil = fontes[0].profile.copy()
    finally:
        for f in fontes:
            f.close()
    perfil.update(height=arr.shape[1], width=arr.shape[2], transform=transform,
                  count=1, driver="GTiff", compress="deflate")
    return arr[0], perfil


def gravar_raster(caminho: Path, arr: np.ndarray, perfil: dict) -> Path:
    """
    Grava GeoTIFF de forma tolerante a sistema de arquivos montado.

    A pasta de trabalho e um mount que NAO permite unlink, e o rasterio apaga o
    arquivo antes de reescrever ("Deleting ... failed: Operation not permitted").
    Solucao: escrever num temporario local e copiar os bytes por cima, porque
    abrir em 'wb' trunca o arquivo em vez de apaga-lo.
    """
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "saida.tif"
        with rasterio.open(tmp, "w", **perfil) as dst:
            dst.write(arr, 1)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "rb") as origem, open(caminho, "wb") as destino:
            shutil.copyfileobj(origem, destino)
    return caminho


def para_bng(dados: np.ndarray, perfil: dict, nome: str, reamostragem) -> Path:
    """Reprojeta para EPSG:27700 numa grade fixa de 30 m alinhada a AOI."""
    largura = int((AOI_BNG[2] - AOI_BNG[0]) / RES)
    altura = int((AOI_BNG[3] - AOI_BNG[1]) / RES)
    transform_alvo = rasterio.transform.from_origin(
        AOI_BNG[0], AOI_BNG[3], RES, RES
    )
    destino = np.full((altura, largura), np.nan, dtype="float32")
    reproject(
        source=dados.astype("float32"),
        destination=destino,
        src_transform=perfil["transform"],
        src_crs=perfil["crs"],
        dst_transform=transform_alvo,
        dst_crs=CRS_ALVO,
        resampling=reamostragem,
        src_nodata=perfil.get("nodata"),
        dst_nodata=np.nan,
    )
    perfil_saida = dict(
        driver="GTiff", height=altura, width=largura, count=1,
        dtype="float32", crs=CRS_ALVO, transform=transform_alvo,
        nodata=np.nan, compress="deflate", tiled=True,
    )
    return gravar_raster(OUT / f"{nome}.tif", destino, perfil_saida)


def declividade(z: np.ndarray, res: float) -> np.ndarray:
    """Declividade em graus pelo metodo de Horn, sobre grade metrica."""
    dzdy, dzdx = np.gradient(z, res, res)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def twi_dinf(z: np.ndarray, res: float, slope_deg: np.ndarray) -> np.ndarray:
    """
    TWI = ln(a / tan(beta)), com area de contribuicao por D-infinity.

    Usa pysheds com roteamento 'dinf', o mesmo tipo de roteamento do v12.
    O `tan(beta)` recebe um piso pequeno para evitar divisao por zero em area
    plana -- sem isso o TWI explodiria justamente onde ele mais importa.
    """
    from pysheds.grid import Grid
    from pysheds.view import Raster, ViewFinder

    import pyproj

    # o pysheds exige um CRS valido no ViewFinder; passar None quebra no pyproj
    vf = ViewFinder(
        shape=z.shape,
        affine=rasterio.transform.from_origin(AOI_BNG[0], AOI_BNG[3], res, res),
        crs=pyproj.Proj(CRS_ALVO, preserve_units=True),
        nodata=np.nan,
    )
    dem = Raster(np.nan_to_num(z, nan=np.nanmin(z)).astype("float64"), viewfinder=vf)
    grid = Grid.from_raster(dem)

    dem_p = grid.fill_pits(dem)
    dem_p = grid.fill_depressions(dem_p)
    dem_p = grid.resolve_flats(dem_p)
    fdir = grid.flowdir(dem_p, routing="dinf")
    acc = grid.accumulation(fdir, routing="dinf")

    a = (np.asarray(acc, dtype="float64") + 1.0) * res   # area especifica
    tanb = np.tan(np.radians(np.clip(slope_deg, 0.05, None)))
    return np.log(a / tanb).astype("float32")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===FT-UK-01===")
    print(f"AOI_BNG={AOI_BNG} res={RES}m crs={CRS_ALVO}")

    dem_ll, perfil_dem = mosaico_recortado("dem_glo30/*.tif")
    print(f"DEM mosaico={dem_ll.shape} em graus")
    p_dem = para_bng(dem_ll, perfil_dem, "elevation_m", Resampling.bilinear)
    del dem_ll

    hand_ll, perfil_hand = mosaico_recortado("hand_glo30/*.tif")
    print(f"HAND mosaico={hand_ll.shape} em graus")
    p_hand = para_bng(hand_ll, perfil_hand, "hand_m", Resampling.bilinear)
    del hand_ll

    with rasterio.open(p_dem) as src:
        z = src.read(1)
        perfil_bng = src.profile.copy()
    print(f"GRADE_BNG={z.shape} celulas={z.size:,}")

    s = declividade(z, RES)
    gravar_raster(OUT / "slope_deg.tif", s.astype("float32"), perfil_bng)
    print(f"SLOPE min={np.nanmin(s):.2f} med={np.nanmedian(s):.2f} max={np.nanmax(s):.2f}")

    t = twi_dinf(z, RES, s)
    gravar_raster(OUT / "twi_dinf.tif", t, perfil_bng)
    print(f"TWI min={np.nanmin(t):.2f} med={np.nanmedian(t):.2f} max={np.nanmax(t):.2f}")

    with rasterio.open(p_hand) as src:
        h = src.read(1)
    print(f"HAND min={np.nanmin(h):.2f} med={np.nanmedian(h):.2f} max={np.nanmax(h):.2f}")
    print(f"ELEV min={np.nanmin(z):.1f} med={np.nanmedian(z):.1f} max={np.nanmax(z):.1f}")

    resumo = {
        "aoi_bng": AOI_BNG, "res_m": RES, "crs": CRS_ALVO,
        "forma": list(z.shape), "celulas": int(z.size),
        "camadas": ["elevation_m", "slope_deg", "twi_dinf", "hand_m"],
        "estatisticas": {
            "elevation_m": [float(np.nanmin(z)), float(np.nanmedian(z)), float(np.nanmax(z))],
            "slope_deg": [float(np.nanmin(s)), float(np.nanmedian(s)), float(np.nanmax(s))],
            "twi_dinf": [float(np.nanmin(t)), float(np.nanmedian(t)), float(np.nanmax(t))],
            "hand_m": [float(np.nanmin(h)), float(np.nanmedian(h)), float(np.nanmax(h))],
        },
        "ressalva_hand": (
            "HAND vem do produto global de 30 m, com cadeia de derivacao propria. "
            "No v12 brasileiro o HAND foi calculado localmente com D-infinity. "
            "Mesma grandeza conceitual, calculo diferente -- declarar como limitacao."
        ),
        "segundos": round(time.time() - t0, 1),
    }
    (OUT / "resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"TEMPO={resumo['segundos']}s")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
