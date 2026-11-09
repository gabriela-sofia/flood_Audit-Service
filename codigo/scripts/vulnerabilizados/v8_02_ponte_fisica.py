"""v8: ponte fisica -- atributos do contrato Q (MT05/MT11) materializados nas coortes brasileiras.

Mesmas definicoes do treino (mt11_confirmatorio_transdutivo.atributos):
  ln_sca_twi      = twi_dinf + ln tan(max(declive, 0,01 grau))     <- contrato do Q (nome canonico ln_sca)
  ln_sca_raster   = ln(dinf_sca) lido do raster original            <- SCA original; so diagnostico de equivalencia
  dist_drenagem_m, dist_canal10_m = mt02_atributos.distancias_da_unidade(tile) (EDT na unidade de terreno inteira)
  declive_p50     = mt11.declive_p50(tile) (raster decimado, sem ponto nem rotulo)
  jrc_ocorrencia  = agua_permanente_jrc.tif do tile (observabilidade/agua anterior; nao entra no Q)
Construido (GHSL) e areia (SoilGrids) ficam para v8_02b (Earth Engine). NoData nunca vira zero.
Coortes: rmpa, manaus, curitiba (v3), recife e rmr (grades v6).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_comum import TER, V3A, V6, V8, agora, etapa, g8, sha_hex

D = V8 / "ponte"
FONTES = {"rmpa": V3A / "rmpa.parquet", "manaus": V3A / "manaus.parquet", "curitiba": V3A / "curitiba.parquet",
          "recife": V6 / "recife_avaliacao/grade.parquet", "rmr": V6 / "rmr_avaliacao/grade.parquet"}


def amostrar_raster(arq: Path, lon, lat) -> np.ndarray:
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import rowcol
    with rasterio.open(arq) as s:
        px, py = Transformer.from_crs(4326, s.crs, always_xy=True).transform(lon, lat)
        lin, col = (np.asarray(v) for v in rowcol(s.transform, px, py))
        ok = (lin >= 0) & (lin < s.height) & (col >= 0) & (col < s.width)
        out = np.full(len(lon), np.nan)
        if ok.any():
            r0, r1, c0, c1 = lin[ok].min(), lin[ok].max() + 1, col[ok].min(), col[ok].max() + 1
            a = s.read(1, window=((r0, r1), (c0, c1))).astype("float64")
            v = a[lin[ok] - r0, col[ok] - c0]
            v[(v == s.nodata) | (v < -1e4)] = np.nan
            out[ok] = v
    return out


def main():
    from mt02_atributos import distancias_da_unidade
    from mt11_confirmatorio_transdutivo import declive_p50
    D.mkdir(parents=True, exist_ok=True)
    for coorte, fonte in FONTES.items():
        alvo = D / f"{coorte}.parquet"
        if alvo.exists():
            continue
        t0 = time.time()
        g = pd.read_parquet(fonte, columns=["ponto_id", "tile", "lon", "lat", "slope_deg", "twi_dinf"])
        partes = []
        for tile, q in g.groupby("tile"):
            pasta = TER / tile
            lon, lat = q.lon.to_numpy(), q.lat.to_numpy()
            d1, d2, lado = distancias_da_unidade(pasta, lon, lat)
            sca = amostrar_raster(pasta / "dinf_sca.tif", lon, lat)
            jrc = amostrar_raster(pasta / "agua_permanente_jrc.tif", lon, lat) if (pasta / "agua_permanente_jrc.tif").exists() else np.full(len(q), np.nan)
            partes.append(pd.DataFrame({"ponto_id": q.ponto_id.to_numpy(), "tile": tile,
                                        "ln_sca_twi": q.twi_dinf.to_numpy() + np.log(np.tan(np.radians(np.clip(q.slope_deg.to_numpy(), .01, None)))),
                                        "ln_sca_raster": np.log(np.where(sca > 0, sca, np.nan)),
                                        "dist_drenagem_m": d1, "dist_canal10_m": d2, "lado_unidade_km": lado,
                                        "declive_p50": declive_p50(pasta), "jrc_ocorrencia": jrc}))
            print(coorte, tile, len(q), flush=True)
        out = pd.concat(partes, ignore_index=True)
        g8(alvo, out)
        f = np.isfinite(out.ln_sca_twi) & np.isfinite(out.ln_sca_raster)
        info = {"quando_utc": agora(), "linhas": len(out), "segundos": time.time() - t0, "fonte_sha256": sha_hex(fonte),
                "finitos": {c: int(np.isfinite(out[c]).sum()) for c in ["ln_sca_twi", "ln_sca_raster", "dist_drenagem_m", "dist_canal10_m", "jrc_ocorrencia"]},
                "equivalencia_ln_sca": {"n": int(f.sum()), "spearman": float(out.loc[f, ["ln_sca_twi", "ln_sca_raster"]].corr("spearman").iloc[0, 1]),
                                        "mediana_dif": float((out.ln_sca_twi - out.ln_sca_raster)[f].median()),
                                        "p95_abs_dif": float((out.ln_sca_twi - out.ln_sca_raster)[f].abs().quantile(.95))}}
        g8(alvo.with_suffix(".json"), info)
        etapa("v8_02_ponte", coorte=coorte, **{k: info[k] for k in ["linhas", "segundos"]})
        print(coorte, info, flush=True)


if __name__ == "__main__":
    main()
