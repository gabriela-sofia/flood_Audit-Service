"""v6: triangulacao da hipotese fisica N1b (agua sob dossel/construido perdida pelo S2) com a mancha UFRGS de 06/05/2024.

Fonte: Zenodo 14617475 (v1.4, CC-BY-4.0), camadas `rhguaiba-*_planet-skysat_inundacao_obs_06052024`: limites de inundacao
interpretados em Planet/SkySat, cota krigada menos FABDEM, validada com pontos de campo. NAO e independente do terreno
(depende de MDE), mas e independente do S2 e do S1 usados no projeto. Protocolo congelado antes de abrir a camada.
Subcomandos: protocolo (py312) | executar (revp-susc)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v6_comum import TV, V6, agora, congelar6, g6, sha_hex

D = V6 / "triangulacao_ufrgs"
GPKG = TV / "fontes/zenodo_14617475/cheias_rhguaiba_2024_db_v14.gpkg"
PONTOS = V6 / "observacao_sar/pontos.parquet"

PROTOCOLO = {
    "versao": "v6-ufrgs-1",
    "pergunta": "pontos S2-negativos sob arvore ou construidos caem dentro da mancha UFRGS mais que S2-negativos abertos no mesmo decil de HAND?",
    "populacao": "pontos RMPA de v6/observacao_sar/pontos.parquet (FCU + baixa renda, adaptacao + teste) dentro do suporte",
    "camadas": "todas as camadas cujo nome contem 'planet-skysat_inundacao_obs_06052024' (as descricoes do glossario de Cai e Pardo estao trocadas; usa-se a geometria)",
    "suporte": "fecho convexo das feicoes de cada camada, sem buffer; fora do suporte = nao observado (nunca negativo)",
    "metrica": "fracao dentro da mancha por cobertura (aberto/arvore/construido) x estado_s2",
    "controle": "diferenca arvore-aberto e construido-aberto entre S2-negativos, media sobre decis de HAND ponderada pelo numero de arvore/construido no decil; bootstrap por grupo (2000, semente 20261021), IC 95%",
    "decisao": "N1b fisico sustentado nesta fonte se IC da diferenca arvore-aberto (S2-negativos) > 0,05; contrariado se IC < 0,05 e inclui ou fica abaixo de 0; insuficiente se < 30 pontos S2-negativos de arvore no suporte",
    "dependencia": "mancha usa FABDEM: concordancia com HAND e esperada e nao e validacao do HAND; por isso a comparacao e dentro do decil de HAND",
    "sem_revisao_humana": True,
}


ADENDO_01 = {
    "versao": "v6-ufrgs-adendo-01",
    "motivo": "na v1.4 do Zenodo a camada usa data AAAAMMDD (rhguaiba_planet-skysat_inundacao_obs_20240506); a regra original (06052024) casou 0 camadas e a execucao 1 nao abriu feicao alguma (resultado.json preservado, suporte 0)",
    "regra": "camadas cujo nome contem 'planet-skysat_inundacao' e 'obs' com data 06052024 ou 20240506",
    "saida": "resultado_a01.json / fracoes_a01.csv",
}


def protocolo():
    p = congelar6(D / "protocolo.json", {**PROTOCOLO, "codigo_sha256": sha_hex(Path(__file__)), "pontos_sha": sha_hex(PONTOS),
                                         "abriu_camada": False})
    print(p["quando_utc"])


def _dif_por_decil(q, cob):
    linhas = []
    for dec, z in q.groupby("decil_hand"):
        a, b = z[z.cobertura.eq(cob)], z[z.cobertura.eq("aberto")]
        if len(a) and len(b):
            linhas.append((len(a), a.dentro.mean() - b.dentro.mean()))
    if not linhas:
        return np.nan
    w, d = np.array(linhas).T
    return float((w * d).sum() / w.sum())


def executar():
    import geopandas as gpd
    from shapely import contains_xy
    import shapely
    prot = congelar6(D / "protocolo.json", {k: v for k, v in PROTOCOLO.items()})
    congelar6(D / "protocolo_adendo_01.json", ADENDO_01)
    camadas = [c for c in gpd.list_layers(GPKG).name if "planet-skysat_inundacao" in c and "obs" in c and ("06052024" in c or "20240506" in c)]
    p = pd.read_parquet(PONTOS)
    p["dentro"] = False
    p["suporte"] = False
    meta = {}
    for c in camadas:
        m = gpd.read_file(GPKG, layer=c).to_crs("EPSG:4326")
        m = m[m.geometry.notna() & ~m.geometry.is_empty]
        geoms = shapely.force_2d(shapely.make_valid(m.geometry.values))
        hull = shapely.convex_hull(shapely.multipoints(shapely.get_coordinates(geoms)))
        pts = shapely.points(p.lon.to_numpy(), p.lat.to_numpy())
        sup = contains_xy(hull, p.lon.to_numpy(), p.lat.to_numpy())
        idx_pt, _ = shapely.STRtree(geoms).query(pts, predicate="intersects")
        den = sup & np.isin(np.arange(len(p)), idx_pt)
        p["suporte"] |= sup
        p["dentro"] |= den
        meta[c] = {"feicoes": int(len(m)), "pontos_suporte": int(sup.sum()), "pontos_dentro": int(den.sum()), "bbox": [float(v) for v in m.total_bounds]}
        print(c, meta[c], flush=True)
    s = p[p.suporte].copy()
    tab = s.groupby(["cobertura", "estado_s2"]).dentro.agg(["mean", "size"]).reset_index()
    neg = s[s.estado_s2.eq("negativo")]
    res = {"camadas": meta, "pontos_suporte": int(len(s)), "por_municipio_grupo": int(s.grupo.nunique())}
    rng = np.random.default_rng(20261021)
    grupos = neg.grupo.unique()
    for cob in ["arvore", "construido"]:
        n = int(neg.cobertura.eq(cob).sum())
        if n < 30 or len(grupos) < 5:
            res[cob] = {"estado": "insuficiente", "n": n}
            continue
        por_g = {g: z for g, z in neg.groupby("grupo")}
        bs = []
        for _ in range(2000):
            amostra = pd.concat([por_g[g] for g in rng.choice(grupos, len(grupos), replace=True)])
            bs.append(_dif_por_decil(amostra, cob))
        bs = np.array(bs)
        iv = [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))]
        est = "sustentado" if iv[0] > .05 else "contrariado" if iv[1] < .05 else "inconclusivo"
        res[cob] = {"n_s2neg": n, "diferenca_vs_aberto_mesmo_decil": _dif_por_decil(neg, cob), "ic95": iv, "estado": est}
    g6(D / "fracoes_a01.csv", tab)
    g6(D / "resultado_a01.json", {"adendo": "protocolo_adendo_01.json", "quando_utc": agora(), "protocolo_quando": prot["quando_utc"], "gpkg_sha256": sha_hex(GPKG), "resultados": res})
    print(tab.round(3).to_string())
    print({k: v for k, v in res.items() if k != "camadas"})


if __name__ == "__main__":
    {"protocolo": protocolo, "executar": executar}[sys.argv[1]]()
