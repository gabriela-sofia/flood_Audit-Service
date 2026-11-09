"""MT02, passo 4: atributos de conexao e de restricao da saida, derivados do acervo (sem aquisicao).

Hipoteses escritas ANTES de medir (falhas do MT01: enxurrada e territorios planos):

  conexao   dist_drenagem_m      distancia horizontal a celula de drenagem D-inf do contrato
                                  (limiar 0,1123 km2). Enxurrada concentra no caminho do fluxo:
                                  mais perto -> mais inundado; efeito maior em relevo forte.
            dist_canal10_m        distancia a celula com area contribuinte >= 10 km2 (canal de porte).
                                  ATENCAO: em chip pequeno a area acumulada e truncada no recorte;
                                  `area_truncavel` marca unidades com menos de 20 km de lado.
  saida     dist_mar_rio_km      DIST_SINK do HydroBASINS nivel 12: distancia ao longo do rio ate
                                  o sumidouro; se o sumidouro e o mar (ENDO = 0), mare e remanso
                                  pesam mais perto dele. Sumidouro endorreico vira NaN declarado.
            elevation_m          cota absoluta FABDEM/MDT (EGM2008 ~ nivel do mar): perto do mar,
                                  cota sobre o mar pode importar mais que altura sobre a drenagem.

nao_e: canal, dique e costa sao proxies; nada aqui mede capacidade de saida, mare ou obra.
Uso:
    python scripts/metodo/mt02_atributos.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "geo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc, sha256_arquivo  # noqa: E402
import mt01_bacias_globais as bg  # noqa: E402
import mt01_populacao as mp  # noqa: E402

SAIDA = mp.RUNS / "mecanismos-transferiveis" / "v2"
CANAL_KM2 = 10.0
LADO_MINIMO_KM = 20.0


def distancias_da_unidade(caminho: Path, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    import rasterio
    from rasterio.transform import rowcol
    from scipy import ndimage
    with rasterio.open(caminho / "streams_dinf.tif") as s:
        canal = s.read(1) == 1
        tr, crs, res = s.transform, s.crs, s.res[0]
    with rasterio.open(caminho / "dinf_accum_cells.tif") as s:
        acum = s.read(1)
    grande = acum * res * res / 1e6 >= CANAL_KM2
    lado_km = min(canal.shape) * res / 1000
    from pyproj import Transformer
    px, py = Transformer.from_crs(4326, crs, always_xy=True).transform(x, y)
    lin, col = rowcol(tr, px, py)
    lin, col = np.asarray(lin), np.asarray(col)
    dentro = (lin >= 0) & (lin < canal.shape[0]) & (col >= 0) & (col < canal.shape[1])
    out = []
    for mascara in (canal, grande):
        d = np.full(len(x), np.nan)
        if mascara.any():
            e = ndimage.distance_transform_edt(~mascara) * res
            d[dentro] = e[lin[dentro], col[dentro]]
        out.append(d)
    return out[0], out[1], lado_km


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    SAIDA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = pd.read_parquet(mp.SAIDA / "populacao.parquet",
                        columns=["ponto_id", "fonte", "papel", "lon", "lat", "unidade_terreno"])
    p = p[p.fonte.isin(mp.FONTES_CONTRASTE) & p.papel.isin(["positivo_episodico", "negativo_observado"])]
    u = pd.read_parquet(mp.ACERVO).set_index("unidade")
    partes = []
    for k, (un, g) in enumerate(p.groupby("unidade_terreno")):
        try:
            d1, d2, lado = distancias_da_unidade(Path(u.loc[un, "caminho"]), g.lon.to_numpy(), g.lat.to_numpy())
            erro = None
        except Exception as e:  # registra; nao inventa
            d1 = d2 = np.full(len(g), np.nan)
            lado, erro = np.nan, str(e)[:120]
        partes.append(pd.DataFrame({"ponto_id": g.ponto_id.to_numpy(), "dist_drenagem_m": d1, "dist_canal10_m": d2,
                                    "lado_unidade_km": lado, "erro_raster": erro}))
        if k % 100 == 0:
            print(k, un, round(time.time() - t0), flush=True)
    a = pd.concat(partes, ignore_index=True)
    a["area_truncavel"] = a.lado_unidade_km < LADO_MINIMO_KM
    # bacia nivel 12: distancia ao sumidouro ao longo do rio
    import bacias_hydrobasins as hb
    bg.registrar_regioes()
    b = hb.Bacias()
    coords = p[["lon", "lat"]].round(5).drop_duplicates().reset_index(drop=True)
    t, _, _ = b.nivel(12)
    loc = b.localizar(coords.lon.to_numpy(), coords.lat.to_numpy(), 12)
    info = t.set_index("HYBAS_ID")[["DIST_SINK", "ENDO"]]
    coords["dist_mar_rio_km"] = info.DIST_SINK.reindex(loc.HYBAS_ID.to_numpy()).to_numpy()
    coords["sumidouro_endorreico"] = info.ENDO.reindex(loc.HYBAS_ID.to_numpy()).to_numpy() == 2
    coords.loc[coords.sumidouro_endorreico, "dist_mar_rio_km"] = np.nan
    p2 = p.assign(_lon=p.lon.round(5), _lat=p.lat.round(5)).merge(
        coords.rename(columns={"lon": "_lon", "lat": "_lat"}), on=["_lon", "_lat"], how="left")
    a = a.merge(p2[["ponto_id", "dist_mar_rio_km", "sumidouro_endorreico"]], on="ponto_id", how="left")
    el = pd.read_parquet(mp.DS06, columns=["ponto_id", "elevation_m"])
    a = a.merge(el, on="ponto_id", how="left").sort_values("ponto_id")
    arq = SAIDA / "atributos_conexao_saida.parquet"
    a.to_parquet(arq, index=False)
    man = {"tipo": "mt02_atributos", "quando_utc": agora_utc(), "pontos": int(len(a)),
           "preenchimento": {c: round(float(a[c].notna().mean()), 4) for c in
                             ("dist_drenagem_m", "dist_canal10_m", "dist_mar_rio_km", "elevation_m")},
           "unidades_com_erro": int(a.erro_raster.notna().groupby(a.erro_raster.notna()).size().get(True, 0)),
           "area_truncavel": float(a.area_truncavel.mean()), "canal_km2": CANAL_KM2,
           "codigo": sha256_arquivo(Path(__file__)), "saida": sha256_arquivo(arq),
           "entrada_populacao": sha256_arquivo(mp.SAIDA / "populacao.parquet"), "segundos": round(time.time() - t0)}
    (SAIDA / "atributos_conexao_saida.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(man, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
