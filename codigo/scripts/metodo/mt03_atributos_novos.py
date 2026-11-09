"""MT03: atributos de pontos NOVOS (fora das 43 unidades) no mesmo formato do MT02/MT03, e escore das candidatas
congeladas. Usado pelas unidades CEMS de desenvolvimento e pelo piloto UrbanSARFloods.

Tudo sem rotulo: terreno da cadeia canonica do proprio recorte, cobertura pelo mesmo `imagem` do cob01 (GEE,
lotes pequenos), conexao do raster local (como o MT02), conexao HydroSHEDS, bacia nivel 12 (HydroBASINS),
contexto do raster inteiro (quantis, como o MT01). Piso local: quantil 0,01 do recorte (minimo 30 pontos).
"""
from __future__ import annotations

import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "geo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
import mt01_populacao as mp  # noqa: E402

V3 = mp.RUNS / "mecanismos-transferiveis" / "v3"


def terreno_nos_pontos(d: pd.DataFrame, pasta: Path) -> pd.DataFrame:
    import cems03_amostrar_pontos as c3
    from pyproj import Transformer
    epsg = c3._crs_do_terreno(pasta)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(d.lon.to_numpy(), d.lat.to_numpy())
    v = c3.amostrar_terreno(list(zip(x, y)), pasta)
    for k, val in v.items():
        d[k] = val
    return d


def cobertura(d: pd.DataFrame, ano: int, lotes: Path, rotulo: str) -> pd.DataFrame:
    import cob01_cobertura_e_solo as cob
    import ee
    ee.Initialize()
    cob.LOTES = lotes
    lotes.mkdir(parents=True, exist_ok=True)
    pedidos = [{"nome": f"lote_{rotulo}_{i // 2000:04d}_{ano}", "ano": ano,
                "pontos": d.iloc[i:i + 2000][["ponto_id", "lat", "lon"]]} for i in range(0, len(d), 2000)]
    with ThreadPoolExecutor(3) as ex:
        arqs = list(ex.map(lambda l: cob.coletar_lote(ee, l), pedidos))
    c = pd.concat([pd.read_parquet(a) for a in arqs])[["ponto_id", "construido_frac", "areia_pct_0_30"]]
    return d.merge(c, on="ponto_id", how="left")


def completar(d: pd.DataFrame, pasta: Path, ano: int, rotulo_grupo: str, lotes: Path,
              grupo: str | None = None) -> pd.DataFrame:
    """d: ponto_id, lon, lat, classe, evento_id, unidade (+ terreno se ja amostrado). pasta: terreno do recorte."""
    import mt02_atributos as a2
    import mt03_conexao_hydrosheds as hs
    if "hand_m" not in d:
        d = terreno_nos_pontos(d, pasta)
        d = d[np.isfinite(d[["elevation_m", "hand_m", "slope_deg", "twi_dinf"]]).all(axis=1)]
        d = d[d.get("agua_permanente_jrc", 0) == 0].copy()
    g = grupo or "unidade_espacial"
    if g not in d:
        d[g] = rotulo_grupo
    piso = d.groupby(g).elevation_m.transform(lambda s: s.quantile(.01))
    n = d.groupby(g).elevation_m.transform("size")
    d["elevation_local_m"] = (d.elevation_m - piso).where(n >= 30)
    d["ln_sca"] = d.twi_dinf + np.log(np.tan(np.radians(d.slope_deg.clip(lower=0.01))))
    if "construido_frac" not in d:
        d = cobertura(d, ano, lotes, rotulo_grupo)
    partes = []
    for un, gg in d.groupby(g):
        caminho = pasta if (pasta / "run_manifest.json").exists() else pasta / un
        d1, d2, lado = a2.distancias_da_unidade(caminho, gg.lon.to_numpy(), gg.lat.to_numpy())
        ctx = mp.contexto_das_unidades(pd.DataFrame({"unidade": [un], "familia_mdt": ["fabdem"],
                                                     "compativel": [True], "caminho": [str(caminho)]}))
        partes.append(pd.DataFrame({"ponto_id": gg.ponto_id.to_numpy(), "dist_drenagem_m": d1, "dist_canal10_m": d2,
                                    "lado_unidade_km": lado, "declive_p50": ctx.declive_p50.iloc[0],
                                    "hand_p90": ctx.hand_p90.iloc[0]}))
    d = d.merge(pd.concat(partes), on="ponto_id", how="left")
    d["area_truncavel"] = d.lado_unidade_km < a2.LADO_MINIMO_KM
    b, _ = mp.localizar_bacias(d.lon.to_numpy(), d.lat.to_numpy())
    for c in b.columns:
        d[c] = b[c].to_numpy()
    d["ln_up_area_l12"] = np.log(d.UP_AREA_L12.where(d.UP_AREA_L12 > 0))
    import bacias_hydrobasins as hb
    import mt01_bacias_globais as bg
    bg.registrar_regioes()
    bb = hb.Bacias()
    t, _, _ = bb.nivel(12)
    loc = bb.localizar(d.lon.to_numpy(), d.lat.to_numpy(), 12)
    info = t.set_index("HYBAS_ID")[["DIST_SINK", "ENDO"]]
    d["dist_mar_rio_km"] = info.DIST_SINK.reindex(loc.HYBAS_ID.to_numpy()).to_numpy()
    d["sumidouro_endorreico"] = info.ENDO.reindex(loc.HYBAS_ID.to_numpy()).to_numpy() == 2
    d.loc[d.sumidouro_endorreico, "dist_mar_rio_km"] = np.nan
    h = hs.atributos_de(d[["ponto_id", "lon", "lat"]].assign(grupo=d[g].to_numpy()))
    d = d.merge(h, on="ponto_id", how="left")
    # colunas que o MT02/MT03 esperam com outro nome
    d["construido_v2"], d["areia_v2"] = d.construido_frac, d.areia_pct_0_30
    d["motivo_areia_pct_0_30"] = np.where(d.areia_pct_0_30.notna(), "valor", "mascara_da_camada")
    d["origem"] = "mt03_novo"
    d["processo_ponto"] = d.get("processo_ponto", "nao_aplicavel")
    d["familia_mdt"] = "fabdem"
    return d.reset_index(drop=True)


def pontuar(d: pd.DataFrame) -> pd.DataFrame:
    import mt03_experimento as e3
    with (V3 / "candidatas_congeladas.pkl").open("rb") as f:
        c = pickle.load(f)
    x = e3.caracteristicas(d)
    out = pd.DataFrame({"A0_regra_hand": -d.hand_m.to_numpy()}, index=d.index)
    for nome, m in c["modelos"].items():
        out[nome] = m.escore(x)  # ausente vira a mediana do ajuste (Padronizador do MT01), como no treino
    return out
