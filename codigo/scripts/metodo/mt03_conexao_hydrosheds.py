"""MT03, passo 3: conexao com dominio a montante suficiente (HydroRIVERS + acumulacao HydroSHEDS 15").

POR QUE
No MT02 a distancia a canal >= 10 km2 vinha da acumulacao D-inf do raster da propria unidade. Em todo chip
Sen1Floods11/UFO, e em muitas AOIs CEMS, a bacia e cortada na borda: o rio grande que atravessa o recorte
entra com a area contribuinte que nasce DENTRO do recorte. A HydroSHEDS calcula a acumulacao no continente
inteiro, entao nenhum ponto tem a bacia cortada -- ao custo de resolucao (15" ~ 460 m; linhas de rio do
HydroRIVERS na mesma grade).

O QUE SAI, por ponto (sem rotulo):
  dist_rio_{T}_m      distancia ao trecho HydroRIVERS mais proximo com UPLAND_SKM >= T, T em 10/100/1000/10000 km2
  upa_15s_km2         area a montante na celula 15" do ponto (celulas x area da celula)
  upa_15s_max3_km2    maximo da janela 3x3 (deslocamento de ~1 celula entre a linha de rio e o ponto)
e, por unidade, o TRUNCAMENTO da representacao do chip:
  razao_area_local    maior area acumulada no raster local / maior UPLAND_SKM do HydroRIVERS na caixa do raster
  pontos perto de canal >=10 km2 pelo HydroRIVERS que o raster local nao enxerga

Fonte: HydroSHEDS v1 (Lehner et al. 2008; Lehner & Grill 2013), licenca HydroSHEDS (uso nao comercial com
atribuicao), mesma familia do HydroBASINS ja usado.

nao_e: rede observada; e rede derivada de MDT de 3" agregada a 15". Distancia a rio nao mede capacidade de
saida nem conexao hidraulica real.

Uso:
    python scripts/metodo/mt03_conexao_hydrosheds.py baixar
    python scripts/metodo/mt03_conexao_hydrosheds.py atributos [--populacao mt01|cems_dev]
    python scripts/metodo/mt03_conexao_hydrosheds.py truncamento
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "geo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402
import mt01_populacao as mp  # noqa: E402

RAIZ = mp.RUNS / "hydrosheds-rios"
V3 = mp.RUNS / "mecanismos-transferiveis" / "v3"
BASE = "https://data.hydrosheds.org/file"
REGIOES = ("eu", "af", "as", "na", "sa", "au")
LIMIARES_KM2 = (10, 100, 1000, 10000)
TETO_M = 50_000.0
UA = {"User-Agent": "REV-P-pesquisa/1.0 (uso academico)"}
# caixas das regioes HydroSHEDS v1 (oeste, sul, leste, norte), do documento tecnico
CAIXAS = {"eu": (-14, 12, 70, 62), "af": (-19, -35, 55, 38), "as": (57, -12, 180, 61), "na": (-138, 5, -52, 62),
          "sa": (-93, -56, -32, 15), "au": (112, -56, 180, -10)}
# o HydroRIVERS cobre tambem acima de 60N (regioes ar/si nao baixadas): ponto acima de 62N fica declarado


def _arquivos(r: str) -> dict:
    return {"rios": (f"{BASE}/HydroRIVERS/HydroRIVERS_v10_{r}_shp.zip", RAIZ / f"HydroRIVERS_v10_{r}_shp.zip"),
            "acc": (f"{BASE}/hydrosheds-v1-acc/hyd_{r}_acc_15s.zip", RAIZ / f"hyd_{r}_acc_15s.zip")}


def baixar() -> dict:
    RAIZ.mkdir(parents=True, exist_ok=True)
    man = {}
    for r in REGIOES:
        for nome, (url, destino) in _arquivos(r).items():
            if not destino.exists():
                tmp = destino.with_suffix(".parcial")
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as resp, \
                        open(tmp, "wb") as f:
                    total = int(resp.headers.get("Content-Length", 0))
                    while bloco := resp.read(1 << 20):
                        f.write(bloco)
                if total and tmp.stat().st_size != total:
                    raise RuntimeError(f"{url}: {tmp.stat().st_size} de {total} bytes")
                tmp.replace(destino)
            man[destino.name] = {"url": url, "bytes": destino.stat().st_size, "sha256": sha256_arquivo(destino)}
            print(destino.name, man[destino.name]["bytes"], flush=True)
    m = {"tipo": "hydrosheds_rios_e_acumulacao", "quando_utc": agora_utc(), "arquivos": man,
         "licenca": "HydroSHEDS license (nao comercial, atribuicao): https://www.hydrosheds.org/terms-of-use",
         "citacao": "Lehner, Verdin & Jarvis 2008; Lehner & Grill 2013 (HydroRIVERS v1.0)"}
    (RAIZ / "manifesto.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"arquivos": len(man), "gb": round(sum(v["bytes"] for v in man.values()) / 1e9, 2)}


def regiao_de(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    out = np.full(len(lon), "", dtype=object)
    for r in REGIOES:  # ordem fixa; sobreposicao das caixas resolvida pela primeira
        o, s, l, n = CAIXAS[r]
        m = (out == "") & (lon >= o) & (lon <= l) & (lat >= s) & (lat <= n)
        out[m] = r
    return out


def _rios_da_regiao(r: str, caixas: list[tuple]) -> pd.DataFrame:
    """Uma passada pelo HydroRIVERS da regiao, guardando so trechos que tocam alguma caixa (lon/lat)."""
    import fiona
    from shapely import STRtree, box
    from shapely.geometry import shape
    zipf = _arquivos(r)["rios"][1].as_posix()
    arv = STRtree([box(*c) for c in caixas])
    linhas = []
    with fiona.open(f"zip://{zipf}!HydroRIVERS_v10_{r}_shp/HydroRIVERS_v10_{r}.shp") as f:
        for ft in f:
            g = shape(ft["geometry"])
            if len(arv.query(g)):
                linhas.append({"UPLAND_SKM": float(ft["properties"]["UPLAND_SKM"]), "geometry": g})
    return pd.DataFrame(linhas, columns=["UPLAND_SKM", "geometry"])


def _acc_url(r: str) -> str:
    return f"/vsizip/{_arquivos(r)['acc'][1].as_posix()}/hyd_{r}_acc_15s.tif"


def atributos_de(p: pd.DataFrame) -> pd.DataFrame:
    """p: ponto_id, lon, lat, grupo (unidade de processamento). Devolve distancias e area a montante."""
    import rasterio
    from pyproj import Transformer
    from shapely import STRtree, points
    from rasterio.windows import Window
    out = []
    from shapely import box
    p = p.assign(regiao=regiao_de(p.lon.to_numpy(), p.lat.to_numpy()))
    m = 0.6  # ~60 km: cobre o teto de 50 km
    cx = p.groupby(["grupo", "regiao"]).agg(o=("lon", "min"), s=("lat", "min"), l=("lon", "max"), n=("lat", "max"))
    por_regiao = {}
    for r, c in cx.groupby(level=1):
        if r:
            caixas = [(a.o - m, a.s - m, a.l + m, a.n + m) for a in c.itertuples()]
            cache = V3 / "conexao" / f"_rios_{r}_{sha256_json(caixas)[7:19]}.parquet"
            if cache.exists():
                from shapely import from_wkb
                t = pd.read_parquet(cache)
                por_regiao[r] = pd.DataFrame({"UPLAND_SKM": t.UPLAND_SKM, "geometry": [from_wkb(b) for b in t.wkb]})
            else:
                from shapely import to_wkb
                por_regiao[r] = _rios_da_regiao(r, caixas)
                cache.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"UPLAND_SKM": por_regiao[r].UPLAND_SKM,
                              "wkb": [to_wkb(g) for g in por_regiao[r].geometry]}).to_parquet(cache, index=False)
            print("regiao", r, len(por_regiao[r]), flush=True)
    arvores = {r: STRtree(list(t.geometry)) for r, t in por_regiao.items() if len(t)}
    for (grupo, r), g in p.groupby(["grupo", "regiao"]):
        base = {"ponto_id": g.ponto_id.to_numpy()}
        if r == "":
            out.append(pd.DataFrame({**base, "regiao_hydrosheds": "fora"}))
            continue
        o, s, l, n = g.lon.min(), g.lat.min(), g.lon.max(), g.lat.max()
        tr = por_regiao[r]
        rios = tr.iloc[arvores[r].query(box(o - m, s - m, l + m, n + m))] if r in arvores else tr
        lon0, lat0 = (o + l) / 2, (s + n) / 2
        utm = (32600 if lat0 >= 0 else 32700) + int((lon0 + 180) // 6) + 1
        t = Transformer.from_crs(4326, utm, always_xy=True).transform
        px, py = t(g.lon.to_numpy(), g.lat.to_numpy())
        pts = points(px, py)
        reg = dict(base)
        from shapely.ops import transform
        rp = rios.assign(geometry=[transform(t, x) for x in rios.geometry]) if len(rios) else rios
        for lim in LIMIARES_KM2:
            sub = rp[rp.UPLAND_SKM >= lim] if len(rp) else rp
            d = np.full(len(g), TETO_M)
            if len(sub):
                arv = STRtree(list(sub.geometry))
                idx, dist = arv.query_nearest(pts, return_distance=True, all_matches=False)
                d[idx[0]] = np.minimum(dist, TETO_M)
            reg[f"dist_rio_{lim}_m"] = d
        with rasterio.open(_acc_url(r)) as src:
            from rasterio.transform import rowcol
            rr, cc = rowcol(src.transform, g.lon.to_numpy(), g.lat.to_numpy())
            rr, cc = np.asarray(rr), np.asarray(cc)
            r0, c0 = max(rr.min() - 1, 0), max(cc.min() - 1, 0)
            r1, c1 = min(rr.max() + 2, src.height), min(cc.max() + 2, src.width)
            a = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(float)
            nd = src.nodata
            if nd is not None:
                a[a == nd] = np.nan
            a[a < 0] = np.nan
            cel_km2 = (15 / 3600 * 111.32) ** 2 * np.cos(np.radians(g.lat.to_numpy()))
            i, j = rr - r0, cc - c0
            reg["upa_15s_km2"] = a[i, j] * cel_km2
            pad = np.pad(a, 1, constant_values=np.nan)
            viz = np.nanmax(np.stack([pad[i + di, j + dj] for di in (0, 1, 2) for dj in (0, 1, 2)]), axis=0)
            reg["upa_15s_max3_km2"] = viz * cel_km2
        reg["regiao_hydrosheds"] = r
        reg["maior_upland_na_caixa_km2"] = float(rios.UPLAND_SKM.max()) if len(rios) else np.nan
        out.append(pd.DataFrame(reg))
        print(grupo, r, len(g), flush=True)
    return pd.concat(out, ignore_index=True)


def atributos(populacao: str) -> dict:
    t0 = time.time()
    if populacao == "mt01":
        p = pd.read_parquet(mp.SAIDA / "populacao.parquet",
                            columns=["ponto_id", "fonte", "papel", "lon", "lat", "unidade_terreno"])
        p = p[p.fonte.isin(mp.FONTES_CONTRASTE) & p.papel.isin(["positivo_episodico", "negativo_observado"])]
        p = p.rename(columns={"unidade_terreno": "grupo"})
        p["grupo"] = p.grupo.fillna("sem_unidade")
        arq = V3 / "conexao" / "atributos_hydrosheds_mt01.parquet"
    else:
        p = pd.read_parquet(V3 / "cems_desenvolvimento" / "pontos.parquet", columns=["ponto_id", "lon", "lat",
                                                                                    "unidade_espacial"])
        p = p.rename(columns={"unidade_espacial": "grupo"})
        arq = V3 / "conexao" / "atributos_hydrosheds_cems_dev.parquet"
    arq.parent.mkdir(parents=True, exist_ok=True)
    a = atributos_de(p[["ponto_id", "lon", "lat", "grupo"]])
    a.to_parquet(arq, index=False)
    man = {"tipo": "mt03_atributos_hydrosheds", "populacao": populacao, "quando_utc": agora_utc(),
           "pontos": int(len(a)), "saida": sha256_arquivo(arq), "codigo": sha256_arquivo(Path(__file__)),
           "fontes": json.loads((RAIZ / "manifesto.json").read_text(encoding="utf-8"))["arquivos"],
           "preenchimento": {c: round(float(a[c].notna().mean()), 4) for c in a.columns if c != "ponto_id"
                             and a[c].dtype.kind == "f"},
           "fora_das_regioes": int((a.regiao_hydrosheds == "fora").sum()), "segundos": round(time.time() - t0)}
    arq.with_suffix(".json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return {k: man[k] for k in ("pontos", "preenchimento", "fora_das_regioes", "segundos")}


def truncamento() -> dict:
    """Quanto a representacao local (chip/AOI) perde da bacia: por unidade de terreno, sem rotulo."""
    import rasterio
    a = pd.read_parquet(V3 / "conexao" / "atributos_hydrosheds_mt01.parquet")
    m2 = pd.read_parquet(mp.RUNS / "mecanismos-transferiveis" / "v2" / "atributos_conexao_saida.parquet",
                         columns=["ponto_id", "dist_canal10_m", "lado_unidade_km", "area_truncavel"])
    p = pd.read_parquet(mp.SAIDA / "populacao.parquet", columns=["ponto_id", "fonte", "unidade_terreno"])
    d = a.merge(m2, on="ponto_id").merge(p, on="ponto_id")
    u = pd.read_parquet(mp.ACERVO).set_index("unidade")
    linhas = []
    for un, g in d.groupby("unidade_terreno"):
        try:
            with rasterio.open(Path(u.loc[un, "caminho"]) / "dinf_accum_cells.tif") as s:
                acum = s.read(1).astype(float)
                if s.nodata is not None:
                    acum[acum == s.nodata] = np.nan
                res = s.res[0]
            maior_local, erro = float(np.nanmax(acum)) * res * res / 1e6, None
        except Exception as e:  # noqa: BLE001 -- registra ausencia, nao inventa
            maior_local, erro = np.nan, str(e)[:120]
        maior_global = float(g.maior_upland_na_caixa_km2.max())
        perto_global = g.dist_rio_10_m <= 300
        linhas.append({"unidade_terreno": un, "fonte": g.fonte.iloc[0], "pontos": len(g), "erro_raster": erro,
                       "maior_area_local_km2": maior_local, "maior_upland_hydrorivers_km2": maior_global,
                       "razao_area_local": maior_local / maior_global if maior_global and maior_global > 0 else np.nan,
                       "frac_perto_rio10_global": float(perto_global.mean()),
                       "frac_perto_rio10_global_sem_canal10_local": float(
                           (perto_global & ~(g.dist_canal10_m <= 300)).mean()),
                       "spearman_dist_local_global": float(g[["dist_canal10_m", "dist_rio_10_m"]].corr("spearman").iloc[0, 1])
                       if g.dist_canal10_m.notna().sum() > 10 else np.nan})
    t = pd.DataFrame(linhas)
    t.to_csv(V3 / "conexao" / "truncamento_por_unidade.csv", index=False)
    r = {"unidades": len(t), "unidades_com_erro_de_leitura": int(t.erro_raster.notna().sum()),
         "razao_area_local_mediana_por_fonte": t.groupby("fonte").razao_area_local.median().round(4).to_dict(),
         "frac_unidades_razao_menor_10pct": t.groupby("fonte").apply(lambda x: float((x.razao_area_local < .1).mean())).round(3).to_dict(),
         "pontos_perto_de_rio10_global_que_o_local_nao_ve": t.groupby("fonte").apply(
             lambda x: float(np.average(x.frac_perto_rio10_global_sem_canal10_local, weights=x.pontos))).round(4).to_dict(),
         "spearman_local_global_mediano": t.groupby("fonte").spearman_dist_local_global.median().round(3).to_dict()}
    (V3 / "conexao" / "truncamento.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=("baixar", "atributos", "truncamento"))
    ap.add_argument("--populacao", default="mt01", choices=("mt01", "cems_dev"))
    a = ap.parse_args()
    r = {"baixar": baixar, "truncamento": truncamento}.get(a.acao)
    r = r() if r else atributos(a.populacao)
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
