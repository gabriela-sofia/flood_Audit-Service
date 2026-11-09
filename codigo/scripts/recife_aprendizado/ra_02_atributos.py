"""Atributos por objeto rua-bairro em blocos separaveis.

Q30   -- atributos do nucleo Q limpo (Qsep_sel_fis) na cadeia de 30 m, medias dos pontos da
         grade v6 a ate 45 m da rua (fallback: ponto mais proximo a ate 90 m). O escore Q do
         objeto e o proprio modelo congelado aplicado a essas medias.
DREN  -- cadastro EMLURB (fotografia atual): entradas BL+CG por 100 m de rua (faixa de 15 m),
         distancia media a rede, fracao das entradas a <= 5 m de uma linha de rede (ligacao
         candidata), fracao da rua junto a canaleta. Densidade nao e capacidade nem causa.
SAIDA -- distancia minima da rua a canal/rio cadastrado (topografica; nao e nivel observado).
PE3D  -- MDT nativo de 1 m agregado a 2 m por media (so ruas na cobertura das 8 folhas):
         posicao relativa da rua ao entorno (50 e 200 m), profundidade de depressao (superficie
         condicionada por preenchimento MENOS a original, mantidas separadas), fracao da rua em
         depressao >= 0,10 m, declividade longitudinal, desnivel ate o canal mais proximo.

Chuva nao entra: a tarefa e estatica (relatos de 2014-2024, sem hora); nao ha forcamento por
objeto e a mistura de fontes de chuva do acervo ja foi diagnosticada.
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np
import pandas as pd

from ra_comum import DREN, EPSG, MACRO, OUT, PE3D_ZIPS, ROOT, V6R, V9P0, Etapa, gravar_json, rel, sha256

sys.path[:0] = [str(ROOT / "scripts/vulnerabilizados"), str(ROOT / "scripts/metodo")]
REF = OUT / "referencias"
ATR = OUT / "atributos"
Q_FEAT = ["f_hand", "f_sca", "f_declive", "f_construido", "f_areia", "f_dist_dren", "f_dist_canal"]
DREN_FEAT = ["dren_entradas_100m", "dren_dist_rede_m", "dren_frac_ligadas_5m", "dren_frac_canaleta"]
SAIDA_FEAT = ["saida_log_dist_canal"]
PE3D_FEAT = ["pe3d_rel50", "pe3d_rel200", "pe3d_depr_max", "pe3d_frac_depr10", "pe3d_decl_long", "pe3d_desnivel_canal"]
TOLS = [1.0, 2.0, 5.0, 10.0]


def amostrar_linha(g, passo):
    pts = []
    if g is None or g.is_empty:
        return pts
    for parte in getattr(g, "geoms", [g]):
        if parte.is_empty or parte.geom_type not in ("LineString", "LinearRing") or parte.length == 0:
            continue
        n = max(int(parte.length // passo), 1)
        pts += [parte.interpolate(i / n, normalized=True) for i in range(n + 1)]
    return pts


def bloco_q(obj):
    import geopandas as gpd
    from v8_04_campanha import caracteristicas, coorte
    with open(V9P0 / "modelos.pkl", "rb") as f:
        q = pickle.load(f)["Qsep_sel_fis"]
    d = coorte("recife")
    grade = pd.read_parquet(V6R / "grade.parquet", columns=["ponto_id", "lon", "lat"])
    d = d.merge(grade, on="ponto_id")
    x = caracteristicas(d)
    x["ponto_id"] = d.ponto_id.values
    x["hand_m"] = d.hand_m.values
    pts = gpd.GeoDataFrame(x, geometry=gpd.points_from_xy(d.lon, d.lat), crs=4326).to_crs(EPSG)
    cols = [c for c in x.columns if c != "ponto_id"]
    buf = obj[["id_obj", "geometry"]].copy()
    buf["geometry"] = buf.buffer(45)
    j = gpd.sjoin(pts, buf, predicate="within")
    agg = j.groupby("id_obj")[cols].mean()
    agg["q_n_pontos"] = j.groupby("id_obj").size()
    falta = obj[~obj.id_obj.isin(agg.index)]
    if len(falta):
        near = gpd.sjoin_nearest(falta[["id_obj", "geometry"]], pts, max_distance=90, distance_col="d")
        near = near.sort_values("d").drop_duplicates("id_obj")
        extra = near.set_index("id_obj")[cols]
        extra["q_n_pontos"] = 0
        agg = pd.concat([agg, extra])
    agg["q_escore"] = q.escore(agg[cols])
    pad = {"principais": list(q.principais), "b": list(map(float, q.b)),
           "interacoes": [list(p) for p in q.interacoes], "g": list(map(float, q.g)),
           "mu": q.pad.mu.to_dict(), "sd": q.pad.sd.to_dict(), "med": q.pad.med.to_dict()}
    return agg, pad, len(d)


def bloco_dren(obj):
    import geopandas as gpd
    from shapely.strtree import STRtree
    el = gpd.read_file(DREN / "layer_00_elementos_da_rede_epsg31985.gpkg", columns=["OBJECTID", "TIPO", "COD_MICROBACIA"])
    rede = gpd.read_file(DREN / "layer_01_rede_epsg31985.gpkg", columns=["OBJECTID"])
    can = gpd.read_file(DREN / "layer_02_canaleta_epsg31985.gpkg", columns=["OBJECTID"])
    entradas = el[el.TIPO.astype(str).str.strip().isin(["BL", "CG"])].copy()
    arv_rede = STRtree(list(rede.geometry))
    d_ent, _ = None, None
    idx_e, dist_e = arv_rede.query_nearest(list(entradas.geometry), return_distance=True, all_matches=False)
    dist_ent = np.full(len(entradas), np.inf)
    dist_ent[idx_e[0]] = dist_e
    entradas["dist_rede"] = dist_ent
    # sensibilidade do grafo candidato (global, descritiva)
    sens = []
    ext = []
    for g in rede.geometry:
        cs = list(g.coords)
        ext += [cs[0], cs[-1]]
    ext = np.asarray(ext)
    from scipy.spatial import cKDTree
    kd = cKDTree(ext)
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    n = len(ext)
    mesma_linha = np.c_[np.arange(0, n, 2), np.arange(1, n, 2)]      # as duas pontas de cada linha
    for t in TOLS:
        pares = np.array(sorted(kd.query_pairs(t)), dtype=int).reshape(-1, 2)
        ij = np.vstack([mesma_linha, pares])
        m = sp.coo_matrix((np.ones(len(ij)), (ij[:, 0], ij[:, 1])), shape=(n, n))
        ncomp, _ = connected_components(m, directed=False)
        sens.append({"tolerancia_m": t, "linhas": len(rede), "componentes": int(ncomp),
                     "entradas_ligadas_frac": float(np.mean(dist_ent <= t)), "entradas": len(entradas)})
    buf = obj[["id_obj", "geometry"]].copy()
    buf["geometry"] = buf.buffer(15)
    j = gpd.sjoin(entradas, buf, predicate="within")
    n_ent = j.groupby("id_obj").size()
    lig = j.assign(l=(j.dist_rede <= 5).astype(float)).groupby("id_obj").l.mean()
    can_u = can.buffer(5).union_all()
    arv_can = None
    linhas = []
    for _, o in obj.iterrows():
        pts = amostrar_linha(o.geometry, 10)
        ii, dd = arv_rede.query_nearest(pts, return_distance=True, all_matches=False)
        linhas.append({"id_obj": o.id_obj, "dren_dist_rede_m": float(np.minimum(dd, 200).mean()),
                       "dren_frac_canaleta": float(o.geometry.intersection(can_u).length / max(o.geometry.length, 1))})
    out = pd.DataFrame(linhas).set_index("id_obj")
    out["dren_entradas_100m"] = (n_ent.reindex(out.index).fillna(0) / obj.set_index("id_obj").comprimento_m * 100).values
    out["dren_frac_ligadas_5m"] = lig.reindex(out.index)
    out["dren_sem_entrada"] = out.dren_frac_ligadas_5m.isna().astype(int)
    out["dren_frac_ligadas_5m"] = out.dren_frac_ligadas_5m.fillna(0.0)
    micro = j.groupby("id_obj").COD_MICROBACIA.agg(lambda s: s.dropna().astype(str).mode().iloc[0] if s.notna().any() else None)
    out["microbacia_moda"] = micro.reindex(out.index)
    return out, pd.DataFrame(sens), {"entradas_BL_CG": len(entradas), "linhas_rede": len(rede), "canaletas": len(can)}


def bloco_saida(obj):
    import geopandas as gpd
    from shapely.strtree import STRtree
    c = gpd.read_file(MACRO / "layer_02_canais_epsg31985.gpkg", columns=["NOME"])
    r = gpd.read_file(MACRO / "layer_03_rios_e_riachos_epsg31985.gpkg", columns=["NOME"])
    linhas = list(c.geometry) + list(r.geometry)
    arv = STRtree(linhas)
    out = []
    for _, o in obj.iterrows():
        pts = amostrar_linha(o.geometry, 10)
        _, dd = arv.query_nearest(pts, return_distance=True, all_matches=False)
        out.append({"id_obj": o.id_obj, "saida_dist_canal_m": float(dd.min()), "saida_log_dist_canal": float(np.log1p(dd.min()))})
    return pd.DataFrame(out).set_index("id_obj"), linhas


def mosaico_pe3d():
    import rasterio
    from rasterio.merge import merge
    from rasterio.enums import Resampling
    srcs = [rasterio.open(f"/vsizip/{z.as_posix()}/{z.stem}.tif") for z in sorted(PE3D_ZIPS.glob("*.zip"))]
    arr, tr = merge(srcs, res=2.0, nodata=-9999.0, resampling=Resampling.average)
    perfil = srcs[0].profile
    for s in srcs:
        s.close()
    perfil.update(height=arr.shape[1], width=arr.shape[2], transform=tr, nodata=-9999.0, compress="deflate", tiled=True,
                  driver="GTiff", count=1, dtype="float32")
    p = ATR / "pe3d_mdt_2m_original.tif"
    with rasterio.open(p, "w", **perfil) as d:
        d.write(arr[0].astype("float32"), 1)
    return p


def bloco_pe3d(obj, linhas_canal):
    import rasterio
    import whitebox
    from rasterio.features import geometry_mask
    from rasterio.windows import Window
    from rasterio.windows import transform as transform_janela
    from shapely.strtree import STRtree
    orig = mosaico_pe3d()
    cond = ATR / "pe3d_mdt_2m_condicionado_fill.tif"
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    for _ in range(3):
        if cond.exists():
            cond.unlink()
        wbt.fill_depressions(str(orig), str(cond), fix_flats=False)
        if cond.exists():
            break
    with rasterio.open(orig) as r:
        z = r.read(1).astype("float64")
        tr, shape = r.transform, r.shape
    z[z <= -9000] = np.nan
    with rasterio.open(cond) as r:
        zc = r.read(1).astype("float64")
    zc[~np.isfinite(z)] = np.nan
    prof = zc - z
    with rasterio.open(ATR / "pe3d_profundidade_depressao_2m.tif", "w", driver="GTiff", height=shape[0], width=shape[1],
                       count=1, dtype="float32", crs=EPSG, transform=tr, nodata=np.nan, compress="deflate") as d:
        d.write(prof.astype("float32"), 1)
    arv_c = STRtree(linhas_canal)
    linhas = []
    for _, o in obj.iterrows():
        g = o.geometry
        b200 = g.buffer(200)
        minx, miny, maxx, maxy = b200.bounds
        c0, r0 = ~tr * (minx, maxy)
        c1, r1 = ~tr * (maxx, miny)
        r0, c0 = max(int(r0), 0), max(int(c0), 0)
        r1, c1 = min(int(r1) + 1, shape[0]), min(int(c1) + 1, shape[1])
        if r1 <= r0 or c1 <= c0:
            continue
        wtr = transform_janela(Window(c0, r0, c1 - c0, r1 - r0), tr)
        zz, pp = z[r0:r1, c0:c1], prof[r0:r1, c0:c1]
        m_rua = ~geometry_mask([g.buffer(3)], out_shape=zz.shape, transform=wtr)
        m50 = ~geometry_mask([g.buffer(50)], out_shape=zz.shape, transform=wtr)
        m200 = ~geometry_mask([b200], out_shape=zz.shape, transform=wtr)
        zr = zz[m_rua & np.isfinite(zz)]
        if len(zr) < 3:
            continue
        med = np.median(zr)
        cand = arv_c.query(g.buffer(500))
        desn = np.nan
        if len(cand):
            zcs = []
            for k in cand:
                for p in amostrar_linha(linhas_canal[k].intersection(g.buffer(500)), 10):
                    cc, rr = ~tr * (p.x, p.y)
                    if 0 <= int(rr) < shape[0] and 0 <= int(cc) < shape[1] and np.isfinite(z[int(rr), int(cc)]):
                        zcs.append(z[int(rr), int(cc)])
            if zcs:
                desn = med - np.median(zcs)
        pr = pp[m_rua & np.isfinite(pp)]
        linhas.append({"id_obj": o.id_obj, "pe3d_z_mediana": float(med),
                       "pe3d_rel50": float(med - np.nanmedian(zz[m50])), "pe3d_rel200": float(med - np.nanmedian(zz[m200])),
                       "pe3d_depr_max": float(pr.max()) if len(pr) else 0.0,
                       "pe3d_frac_depr10": float(np.mean(pr >= 0.10)) if len(pr) else 0.0,
                       "pe3d_decl_long": float((np.percentile(zr, 90) - np.percentile(zr, 10)) / max(g.length, 1) * 100),
                       "pe3d_desnivel_canal": float(desn), "pe3d_pixels_rua": int(len(zr))})
    return pd.DataFrame(linhas).set_index("id_obj"), {"original": rel(orig), "condicionado": rel(cond),
                                                      "sha_original": sha256(orig), "sha_condicionado": sha256(cond)}


def main() -> int:
    import geopandas as gpd
    e = Etapa("ra_02_atributos")
    ATR.mkdir(parents=True, exist_ok=True)
    obj = gpd.read_file(REF / "objetos_rua_bairro.gpkg")
    t0 = time.time()
    aq, padq, n_grade = bloco_q(obj)
    print(f"[Q] {len(aq)} objetos ({time.time() - t0:.0f}s)", flush=True)
    ad, sens, info_d = bloco_dren(obj)
    sens.to_csv(ATR / "grafo_drenagem_sensibilidade.csv", index=False)
    print(f"[DREN] ({time.time() - t0:.0f}s)", flush=True)
    asd, linhas_canal = bloco_saida(obj)
    print(f"[SAIDA] ({time.time() - t0:.0f}s)", flush=True)
    ap, info_p = bloco_pe3d(obj[obj.em_pe3d.astype(bool)], linhas_canal)
    print(f"[PE3D] {len(ap)} objetos ({time.time() - t0:.0f}s)", flush=True)
    t = obj.drop(columns="geometry").set_index("id_obj")
    t = t.join(aq).join(ad).join(asd).join(ap)
    t["tem_q"] = t.q_escore.notna()
    t["tem_pe3d"] = t.pe3d_rel50.notna()
    t = t.reset_index()
    t.to_parquet(ATR / "atributos_objetos.parquet", index=False)
    gravar_json(ATR / "nucleo_q_congelado.json", {**padq, "origem": rel(V9P0 / "modelos.pkl"),
                                                   "sha256_pkl": sha256(V9P0 / "modelos.pkl"),
                                                   "nota": "Qsep_sel_fis: lambda e coeficientes so nas 29 unidades de ajuste_fisico (v9 P0)"})
    dic = pd.DataFrame([{"atributo": a, "bloco": b} for b, fs in
                        [("Q30", Q_FEAT + ["q_escore"]), ("DREN", DREN_FEAT), ("SAIDA", SAIDA_FEAT), ("PE3D", PE3D_FEAT)] for a in fs])
    dic.to_csv(ATR / "dicionario_atributos.csv", index=False)
    resumo = {"objetos": len(t), "com_q": int(t.tem_q.sum()), "com_pe3d": int(t.tem_pe3d.sum()),
              "grade_v6_pontos": n_grade, "drenagem": info_d, "pe3d": info_p,
              "faltantes": t[Q_FEAT + DREN_FEAT + SAIDA_FEAT + PE3D_FEAT].isna().sum().to_dict(),
              "sha_atributos": sha256(ATR / "atributos_objetos.parquet"),
              "parede_s": round(time.time() - t0, 1)}
    gravar_json(ATR / "atributos.json", resumo)
    e.saida(*ATR.glob("*"))
    e.fechar(**{k: v for k, v in resumo.items() if k not in ("faltantes",)})
    import json
    print(json.dumps(resumo, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
