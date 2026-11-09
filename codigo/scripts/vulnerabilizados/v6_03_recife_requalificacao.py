"""v6: requalificacao de Recife por criterios locais verificaveis (pos-v5; a triagem v5 fica preservada).

Regra fixada antes das consultas:
- Episodio: sequencia de dias com relatos administrativos de agua (categoria 'relato_administrativo_agua',
  sem duplicatas de conteudo), lacuna <= 1 dia, total >= 20 relatos. Datas SEDEC sao de REGISTRO (sem hora).
  Janela de observacao: [primeiro dia, ultimo dia + 1]; +2 se o episodio dura >= 3 dias (persistencia ou chuva repetida).
- Nuvem: eo:cloud_cover None -> 'desconhecida' (nao entra como limpa nem como nublada); 0 e valido (corrige v5).
- S2: para cada cena na janela, fracao valida local do SCL (2,4,5,6,7) em cada FCU de Recife (setores CD_TIPO=1,
  malha PE 2022); FCU 'observavel' se >= 0,8. A classe agua nao e gravada.
- S1 (Planetary Computer RTC): cenas na janela e par anterior na mesma orbita relativa em ate 24 dias; suporte
  potencial = fracao nao construida (WorldCover 2021 != 50) de cada FCU; duplo espalhamento urbano fica desconhecido.
- Viabilidade: S2 se >= 8 FCU observaveis numa cena; S1 parcial se ha par e >= 8 FCU com >= 20% nao construido.
Parada: sem viabilidade, registrar a lacuna por episodio e nao adquirir pixels alem do SCL/metadados.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v6_comum import TV, V6, agora, congelar6, g6, sha_hex

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
D = V6 / "recife"
ES = "https://earth-search.aws.element84.com/v1/search"
PC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
BBOX = [-35.02, -8.16, -34.86, -7.93]


def stac(url, colecao, d0, d1):
    corpo = {"collections": [colecao], "bbox": BBOX, "datetime": f"{d0}T00:00:00Z/{d1}T23:59:59Z", "limit": 100}
    req = urllib.request.Request(url, data=json.dumps(corpo).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)["features"]


def nuvem_cena(f):
    v = f["properties"].get("eo:cloud_cover")
    return None if v is None else float(v)


def episodios():
    regs = pd.concat([pd.read_parquet(p, columns=["data_registro", "categoria", "duplicado_conteudo"])
                      for p in sorted((TV / "v3/recife").glob("registros_*.parquet"))])
    a = regs[regs.categoria.eq("relato_administrativo_agua") & ~regs.duplicado_conteudo.fillna(False).astype(bool)]
    dia = a.groupby(pd.to_datetime(a.data_registro).dt.date).size().sort_index()
    eps, atual = [], []
    for d_, n in dia.items():
        if atual and (d_ - atual[-1][0]).days > 2:
            eps.append(atual)
            atual = []
        atual.append((d_, int(n)))
    if atual:
        eps.append(atual)
    out = []
    for e in eps:
        tot = sum(n for _, n in e)
        if tot >= 20:
            dur = (e[-1][0] - e[0][0]).days + 1
            out.append({"inicio": e[0][0], "fim": e[-1][0], "dias_com_relato": len(e), "duracao_dias": dur, "relatos": tot,
                        "janela_fim": e[-1][0] + timedelta(days=2 if dur >= 3 else 1)})
    return pd.DataFrame(out)


def fcu_recife():
    import geopandas as gpd
    z = TV / "fontes/setores_2022/PE_setores_CD2022.zip"
    shp = [n for n in zipfile.ZipFile(z).namelist() if n.endswith(".shp")][0]
    g = gpd.read_file(f"/vsizip/{z.as_posix()}/{shp}", columns=["CD_SETOR", "CD_MUN", "CD_TIPO", "CD_FCU", "NM_FCU"])
    g = g[g.CD_MUN.eq("2611606") & g.CD_TIPO.eq("1")]
    return g.dissolve("CD_FCU", aggfunc={"NM_FCU": "first"}).reset_index()


def fracao_por_fcu(href, fcu, classes_validas=(2, 4, 5, 6, 7)):
    import rasterio
    from rasterio import features
    from rasterio.windows import from_bounds
    with rasterio.open("/vsicurl/" + href) as s:
        f2 = fcu.to_crs(s.crs)
        w = from_bounds(*f2.total_bounds, transform=s.transform).round_offsets().round_lengths()
        a = s.read(1, window=w, boundless=True, fill_value=0)
        tr = s.window_transform(w)
    idx = features.rasterize(zip(f2.geometry, range(1, len(f2) + 1)), out_shape=a.shape, transform=tr, fill=0, dtype="int32")
    ok = np.isin(a, classes_validas)
    tot = np.bincount(idx.ravel(), minlength=len(f2) + 1)[1:]
    val = np.bincount(idx.ravel(), weights=ok.ravel(), minlength=len(f2) + 1)[1:]
    return np.where(tot > 0, val / np.maximum(tot, 1), np.nan)


def nao_construido(fcu):
    import rasterio
    from rasterio import features
    from rasterio.windows import from_bounds
    u = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_S09W036_Map.tif"
    with rasterio.open(u) as s:
        f2 = fcu.to_crs(s.crs)
        w = from_bounds(*f2.total_bounds, transform=s.transform).round_offsets().round_lengths()
        a = s.read(1, window=w)
        tr = s.window_transform(w)
    idx = features.rasterize(zip(f2.geometry, range(1, len(f2) + 1)), out_shape=a.shape, transform=tr, fill=0, dtype="int32")
    tot = np.bincount(idx.ravel(), minlength=len(f2) + 1)[1:]
    nc = np.bincount(idx.ravel(), weights=(a != 50).ravel(), minlength=len(f2) + 1)[1:]
    return np.where(tot > 0, nc / np.maximum(tot, 1), np.nan)


def main():
    congelar6(D / "protocolo.json", {"versao": "v6-recife-1", "regras": __doc__, "codigo_sha256": sha_hex(Path(__file__)),
                                     "triagem_v5_preservada": "local_runs/transferencia-vulnerabilizados/v5_execucao_comunidades/recife/"})
    ep = episodios()
    fcu = fcu_recife()
    fcu["frac_nao_construido"] = nao_construido(fcu)
    s1_suporte = int((fcu.frac_nao_construido >= .2).sum())
    linhas = []
    for e in ep.itertuples():
        d0, d1 = e.inicio.isoformat(), e.janela_fim.isoformat()
        s2 = stac(ES, "sentinel-2-l2a", d0, d1)
        melhor = 0
        detalhes = []
        for f in s2:
            fr = fracao_por_fcu(f["assets"]["scl"]["href"], fcu)
            obs = int(np.nansum(fr >= .8))
            melhor = max(melhor, obs)
            detalhes.append({"item": f["id"], "nuvem_cena": nuvem_cena(f), "fcu_observaveis": obs, "fracao_media": float(np.nanmean(fr))})
        s1 = stac(PC, "sentinel-1-rtc", d0, d1)
        pares = []
        for f in s1:
            orb = f["properties"].get("sat:relative_orbit")
            t = date.fromisoformat(f["properties"]["datetime"][:10])
            pre = [g for g in stac(PC, "sentinel-1-rtc", (t - timedelta(days=24)).isoformat(), (t - timedelta(days=1)).isoformat())
                   if g["properties"].get("sat:relative_orbit") == orb]
            pares.append({"pos": f["id"], "hora_utc": f["properties"]["datetime"][11:16], "orbita": orb, "pre_mesma_orbita": [g["id"] for g in pre]})
        viavel_s2 = melhor >= 8
        viavel_s1 = any(p["pre_mesma_orbita"] for p in pares) and s1_suporte >= 8
        linhas.append({"inicio": d0, "fim": e.fim.isoformat(), "janela_fim": d1, "duracao_dias": e.duracao_dias, "relatos": e.relatos,
                       "s2_cenas": len(s2), "s2_nuvem_desconhecida": int(sum(nuvem_cena(f) is None for f in s2)),
                       "s2_max_fcu_observaveis": melhor, "s2_detalhes": detalhes, "s1_cenas": len(s1), "s1_pares": pares,
                       "viavel_s2": viavel_s2, "viavel_s1_parcial": viavel_s1})
        print(d0, d1, e.relatos, "s2", len(s2), melhor, "s1", len(s1), viavel_s2, viavel_s1, flush=True)
    t = pd.DataFrame(linhas)
    g6(D / "episodios.csv", t.drop(columns=["s2_detalhes", "s1_pares"]))
    g6(D / "fcu_suporte_s1.csv", fcu.drop(columns="geometry"))
    g6(D / "resultado.json", {"quando_utc": agora(), "fcu_recife": int(len(fcu)), "fcu_com_20pct_nao_construido": s1_suporte,
                              "episodios": linhas,
                              "fichas": [{"aoi": "FCU de Recife (2611606)", "episodio": f"{r['inicio']}..{r['janela_fim']}",
                                          "decisao": "adquirir janelas S2" if r["viavel_s2"] else ("adquirir par S1 so no suporte nao construido" if r["viavel_s1_parcial"] else "nao adquirir"),
                                          "lacuna": "sem cena S2 com >= 8 FCU observaveis" if not r["viavel_s2"] else "-"} for r in linhas]})
    print(t[["inicio", "janela_fim", "relatos", "s2_cenas", "s2_max_fcu_observaveis", "s1_cenas", "viavel_s2", "viavel_s1_parcial"]].to_string())
    print("FCU", len(fcu), "com >=20% nao construido", s1_suporte)


if __name__ == "__main__":
    main()
