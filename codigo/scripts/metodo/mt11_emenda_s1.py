"""MT11, emenda 8 (antes de qualquer escore): órbita S1 escolhida pela cobertura dos PONTOS, não pela contagem de datas.

Defeito (17/10/2026): `_pilha_s1` escolhe a órbita relativa com mais datas que TOCAM o tile de 1 grau. A
pilha de 16/10 trabalhava em blocos de 7,7 km, sempre dentro da faixa da órbita; num tile de 1 grau a
órbita mais frequente pode cobrir só um canto. Nos três primeiros eventos extraídos, a órbita escolhida
cobria 2% (GFD:4483) e 9% (GFD:4434) dos pontos, e a admissão por data (>= 90% dos pontos com leitura)
recusou TODAS as datas. Nenhum escore tinha sido calculado.

Correção, mantendo uma órbita por tile e todo o resto da receita registrada:
  órbita   a que cobre mais pontos do evento pela pegada (`system:footprint`) das suas cenas na janela
           [início-425 d, início-60 d]; empate: mais datas, depois menor número
  pegada   pontos dentro da união das pegadas dessa órbita; fora dela F fica ausente (sai do HF)
  admissão a mesma regra (>= 90% com leitura, mediana VV em [-25,-3] dB, neve), calculada sobre os
           pontos da pegada; `frequencia` e `qualidade` originais
  qualidade os 2000 pixels aleatórios do tile restritos à mesma pegada
Atributos de terreno, distâncias, cobertura e espelho: funções originais, sem mudança.
Saída em `atributos_e8/`; `atributos/` (extração original de 3 eventos) fica preservado.
Uso (pela trava): python scripts/metodo/mt11_emenda_s1.py atributos
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for s in ("metodo", "geo", "servico", "suscetibilidade", "terreno", "sar"):
    sys.path.insert(0, str(REPO / "scripts" / s))
import mt11_confirmatorio_transdutivo as m11  # noqa: E402
from proveniencia import sha256_arquivo  # noqa: E402

SAIDA = m11.SAIDA / "atributos_e8"
CFG = m11.CFG


def _colecao(ee, tile, inicio):
    t0 = dt.date.fromisoformat(inicio)
    return (ee.ImageCollection("COPERNICUS/S1_GRD").filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")).filterBounds(ee.Geometry.Rectangle(tile))
            .filterDate(str(t0 - dt.timedelta(days=CFG["janela_pre_dias"][0])), str(t0 - dt.timedelta(days=CFG["janela_pre_dias"][1]))))


def escolher_orbita(cenas: list[dict], lon: np.ndarray, lat: np.ndarray) -> dict:
    """cenas: [{orbita, data, anel:[[lon,lat],...]}]. Devolve órbita, datas, máscara da pegada e cobertura por órbita."""
    import shapely
    from shapely.geometry import Polygon
    pts = shapely.points(lon, lat)
    por = {}
    for c in cenas:
        por.setdefault(c["orbita"], {"datas": set(), "polis": []})
        por[c["orbita"]]["datas"].add(c["data"])
        por[c["orbita"]]["polis"].append(Polygon(c["anel"]).buffer(0))
    if not por:
        return {"orbita": None, "datas": [], "pegada": np.zeros(len(lon), bool), "cobertura": {}}
    cob = {}
    for o, v in por.items():
        u = shapely.union_all(v["polis"])
        cob[o] = shapely.covers(u, pts)
    orb = max(por, key=lambda o: (int(cob[o].sum()), len(por[o]["datas"]), -o))
    return {"orbita": orb, "datas": sorted(por[orb]["datas"])[-CFG["max_datas"]:], "pegada": cob[orb],
            "cobertura": {int(o): float(c.mean()) for o, c in cob.items()}}


def cenas_da_janela(ee, tile, inicio) -> list[dict]:
    col = _colecao(ee, tile, inicio)
    orb, ts, fp = (col.aggregate_array(p).getInfo() for p in ("relativeOrbitNumber_start", "system:time_start", "system:footprint"))
    return [{"orbita": int(o), "data": dt.datetime.fromtimestamp(t / 1000, dt.UTC).date().isoformat(), "anel": f["coordinates"]}
            for o, t, f in zip(orb, ts, fp)]


def imagem_da_orbita(ee, tile, inicio, crs, orb, datas):
    """O mesmo multilook de `_pilha_s1`, para a órbita e as datas dadas."""
    col = _colecao(ee, tile, inicio)
    bandas = []
    for k, d in enumerate(datas):
        dia = (col.filter(ee.Filter.eq("relativeOrbitNumber_start", orb))
               .filterDate(d, str(dt.date.fromisoformat(d) + dt.timedelta(days=1))).select("VV").mosaic())
        lin = ee.Image(10).pow(dia.divide(10))
        ml = (lin.setDefaultProjection(crs=crs, scale=10).reduceResolution(ee.Reducer.mean(), maxPixels=64)
              .reproject(crs=crs, scale=30).log10().multiply(10))
        bandas.append(ml.rename(f"VV_{k:02d}"))
    return ee.Image.cat(bandas)


def frequencia_na_pegada(V: np.ndarray, datas: list[str], lat: float, pegada: np.ndarray) -> dict:
    """Regra original aplicada só aos pontos da pegada; fora dela tudo fica ausente (nunca zero)."""
    out = {"F_v1": np.full(len(V), np.nan), "F_e1": np.full(len(V), np.nan), "n_pre": np.zeros(len(V), int), "datas_admitidas": 0}
    if pegada.sum() == 0 or not datas:
        return out
    f = m11.frequencia(V[pegada], datas, lat)
    for k in ("F_v1", "F_e1", "n_pre"):
        out[k][pegada] = f[k]
    out["datas_admitidas"] = f["datas_admitidas"]
    return out


def s1_no_tile(ee, ev, d: pd.DataFrame, crs: str) -> tuple[pd.DataFrame, dict]:
    tile = ev["tiles"][0]["tile_1grau"]
    lat0 = (tile[1] + tile[3]) / 2
    cenas = cenas_da_janela(ee, tile, ev["inicio"])
    esc = escolher_orbita(cenas, d.lon.to_numpy(), d.lat.to_numpy())
    vazio = d.assign(F_v1=np.nan, F_e1=np.nan, n_pre=0, arvore=np.nan, construido_wc=np.nan, jrc_occ_s1=np.nan, pegada_s1=esc["pegada"])
    if esc["orbita"] is None or not esc["pegada"].any():
        return vazio, {"orbita": esc["orbita"], "datas": len(esc["datas"]), "datas_admitidas": 0, "regra": None,
                       "cobertura_por_orbita": esc["cobertura"], "fracao_pontos_na_pegada": float(esc["pegada"].mean())}
    img = imagem_da_orbita(ee, tile, ev["inicio"], crs, esc["orbita"], esc["datas"])
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).rename("jrc_occ")
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    estat = ee.Image.cat([wc.eq(10).toFloat().reduceResolution(ee.Reducer.mean(), maxPixels=64).rename("arvore"),
                          wc.eq(50).toFloat().reduceResolution(ee.Reducer.mean(), maxPixels=64).rename("construido_wc"),
                          jrc]).reproject(crs=crs, scale=30)
    tudo = img.addBands(estat)
    nomes = [f"VV_{k:02d}" for k in range(len(esc["datas"]))]
    t = m11._amostrar_tabela(ee, tudo, d[["ponto_id", "lat", "lon"]], 30, crs).set_index("ponto_id").reindex(d.ponto_id)
    fr = frequencia_na_pegada(t.reindex(columns=nomes).to_numpy(float), esc["datas"], lat0, esc["pegada"])
    aleat = ee.FeatureCollection.randomPoints(ee.Geometry.Rectangle(tile), CFG["pontos_qualidade"], CFG["semente_pontos"])
    pq = pd.DataFrame([{"ponto_id": f"q{i}", "lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1]}
                       for i, f in enumerate(aleat.getInfo()["features"])])
    peg_q = escolher_orbita([c for c in cenas if c["orbita"] == esc["orbita"]], pq.lon.to_numpy(), pq.lat.to_numpy())["pegada"]
    tq = m11._amostrar_tabela(ee, tudo, pq, 30, crs).set_index("ponto_id").reindex(pq.ponto_id)
    fq = frequencia_na_pegada(tq.reindex(columns=nomes).to_numpy(float), esc["datas"], lat0, peg_q)
    obs_q = (tq.arvore.to_numpy(float) <= CFG["arvore_max"]) & (tq.construido_wc.to_numpy(float) <= CFG["construido_max"])
    jq = np.where(obs_q, tq.jrc_occ.to_numpy(float), np.nan)
    qual = {r: m11.qualidade(fq[f"F_{r}"], jq, fq["n_pre"]) for r in ("v1", "e1")}
    out = d.assign(F_v1=fr["F_v1"], F_e1=fr["F_e1"], n_pre=fr["n_pre"], arvore=t.arvore.to_numpy(float),
                   construido_wc=t.construido_wc.to_numpy(float), jrc_occ_s1=t.jrc_occ.to_numpy(float), pegada_s1=esc["pegada"])
    return out, {"orbita": esc["orbita"], "datas": len(esc["datas"]), "datas_admitidas": fr["datas_admitidas"], "qualidade": qual,
                 "regra": "e1" if qual["e1"]["passa"] else ("v1" if qual["v1"]["passa"] else None),
                 "cobertura_por_orbita": esc["cobertura"], "fracao_pontos_na_pegada": float(esc["pegada"].mean()),
                 "pontos_qualidade_na_pegada": int(peg_q.sum())}


def atributos():
    """Mesma sequência de `m11.atributos`, com o S1 desta emenda; grava em atributos_e8/."""
    import ee
    import gfdq02_derivar_quarentena as gq
    from gfd02_amostrar_terreno import _amostrar_no_tile
    from mt02_atributos import distancias_da_unidade
    import mt11_emenda as em
    em.validar()
    ee.Initialize()
    SAIDA.mkdir(parents=True, exist_ok=True)
    for ev in m11.ler(m11.SAIDA / "selecao.json")["eventos"]:
        alvo = SAIDA / f"{ev['evento_id'].replace(':', '_')}.parquet"
        if alvo.exists() and alvo.with_suffix(".json").exists() and m11.ler(alvo.with_suffix(".json"))["sha256"] == sha256_arquivo(alvo):
            continue
        t = ev["tiles"][0]
        p = pd.read_csv(m11.SAIDA / "pontos" / f"pontos_gfd{ev['id']}_{t['tile']}.csv")
        tile = p.tile.iloc[0]
        if em.terreno_ok(m11.TER / tile)["estado"] != "completo":
            print(ev["evento_id"], "terreno incompleto", flush=True)
            continue
        s = _amostrar_no_tile(p, m11.TER / tile)
        d, saiu = gq.espelho_do_ds06(p.reset_index(drop=True).assign(**{k: v for k, v in s.items()}))
        if len(d) == 0:
            # emenda 9: o espelho pré-registrado retirou todos os pontos; nada a consultar, o evento sai por regra
            vazio = d.assign(ln_sca=np.nan, dist_drenagem_m=np.nan, dist_canal10_m=np.nan, declive_p50=np.nan, construido_v2=np.nan,
                             areia_v2=np.nan, F_v1=np.nan, F_e1=np.nan, n_pre=0, arvore=np.nan, construido_wc=np.nan,
                             jrc_occ_s1=np.nan, pegada_s1=False, unidade=ev["unidade"], evento_id=ev["evento_id"])
            vazio.to_parquet(alvo, index=False)
            m11.escrever(alvo.with_suffix(".json"), {"evento_id": ev["evento_id"], "linhas": 0, "saiu_no_espelho": saiu,
                                                     "s1": {"orbita": None, "datas": 0, "datas_admitidas": 0, "regra": None,
                                                            "motivo": "sem linhas após o espelho do ds06"},
                                                     "sha256": sha256_arquivo(alvo), "emenda": 9})
            print(ev["evento_id"], 0, "sem linhas após o espelho", flush=True)
            continue
        d["ln_sca"] = d.twi_dinf + np.log(np.tan(np.radians(np.clip(d.slope_deg, .01, None))))
        d1, d2, lado = distancias_da_unidade(m11.TER / tile, d.lon.to_numpy(), d.lat.to_numpy())
        d["dist_drenagem_m"], d["dist_canal10_m"], d["lado_unidade_km"] = d1, d2, lado
        d["declive_p50"] = m11.declive_p50(m11.TER / tile)
        ano = int(ev["inicio"][:4])
        d = d.merge(m11.cobertura(ee, d, ano), on="ponto_id", how="left")
        d["construido_v2"], d["areia_v2"] = d.construido_frac, d.areia_pct_0_30
        d, info = s1_no_tile(ee, ev, d, f"EPSG:{int(d.crs_metrico.iloc[0])}")
        d["unidade"], d["evento_id"] = ev["unidade"], ev["evento_id"]
        tmp = alvo.with_suffix(".parquet.part")
        d.to_parquet(tmp, index=False)
        tmp.replace(alvo)
        m11.escrever(alvo.with_suffix(".json"), {"evento_id": ev["evento_id"], "linhas": len(d), "saiu_no_espelho": saiu,
                                                 "s1": info, "ano_cobertura": ano, "sha256": sha256_arquivo(alvo), "emenda": 8})
        print(ev["evento_id"], len(d), info.get("regra"), info.get("datas_admitidas"), round(info.get("fracao_pontos_na_pegada", 0), 2), flush=True)
    return {"eventos_com_atributos": len(list(SAIDA.glob("*.parquet")))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["atributos"])
    print(json.dumps({"atributos": atributos}[ap.parse_args().acao](), ensure_ascii=False, default=str))
