"""v6: avaliacao congelada nos 14 outros municipios da Concentracao Urbana de Recife (territorio novo, episodio conhecido).

Dependencia declarada: mesmo episodio (chuvas de 28/05/2022) e mesma cena S2 de 01/06/2022 ja usados em Recife
municipio (v6_04). Rotulos destes municipios nunca foram abertos. Modelos e regras abaixo foram decididos com base no
desenvolvimento (RMPA, Recife municipio) antes de construir esta referencia.
Subcomandos: grade (revp-susc) | covariaveis (py312) | protocolo (py312) | predicoes (py312) | referencia (revp-susc) | avaliacao (py312)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v6_comum import REPO, RUNS_TV, V3A, V5, V6, agora, congelar6, g6, ler, sha_hex

D = V6 / "rmr_avaliacao"
AQUI = Path(__file__).resolve().parent
TILES = {"gfd_tile_S08W035": (-35, -34, -8, -7), "gfd_tile_S09W035": (-35, -34, -9, -8),
         "gfd_tile_S08W036": (-36, -35, -8, -7), "gfd_tile_S09W036": (-36, -35, -9, -8)}
CLASSES = ["fcu", "baixa_renda", "comparador_urbano", "urbano_sem_renda"]
EVENTO = {"25MBM": "S2B_25MBM_20220601_0_L2A", "25LBL": "S2B_25LBL_20220601_0_L2A"}
NIVEL = 1 - 0.05 / 4

PROTOCOLO = {
    "versao": "v6-rmr-1",
    "natureza": "confirmacao delimitada: territorio novo (14 municipios da CU Recife, exceto Recife), episodio e cena ja usados em Recife municipio",
    "populacao": "setores Censo 2022 da CU 'Recife/PE' exceto 2611606; pontos de 60 m; admitidos so pontos com SCL valido na cena do evento (sem rotulo)",
    "referencia": {"evento": EVENTO, "pre": "por tile: ultima cena nos 90 dias antes de 2022-05-28 com fracao valida media >= 0,8 nos pontos FCU do tile",
                   "agua": "MNDWI > 0 em 20 m", "permanente": "agua valida na cena pre", "estados": "rotulo_referencia()"},
    "modelos_congelados": {
        "A0": "-HAND", "A1_log": "v5 A1 internacional completo (14 unidades fora do Brasil)",
        "GBM_ext": "HistGradientBoosting monotonico (HAND-, declive-, TWI+, construido livre, canal-), depth 3, 200 it, taxa 0,05, folha 20, nos 14 doadores internacionais, pesos balanceados",
        "A1_ctx7": "logistica balanceada nos 7 doadores internacionais mais proximos do descritor sem rotulo DESTA populacao (p90 HAND, mediana declive, fracao construida)",
        "A3_rmpa_fcu100": "v5 A3 internacional completo fcu_comunidades B=100, media de 6 sementes",
        "A3_rmpa_uni400": "v5 A3 urbano_uniforme B=400", "A2e_rmpa_uni400": "v5 A2_estrito urbano_uniforme B=400"},
    "primarios": {"C1": "GBM_ext - A0 (H1: valor do conhecimento estrangeiro nao linear sobre HAND)",
                  "C2": "A1_ctx7 - A1_log (H-C: selecao de doadores por contexto)",
                  "C3": "A3_rmpa_fcu100 - A1_log (H3: transporte do aprendizado brasileiro)",
                  "C4": "A3_rmpa_uni400 - A2e_rmpa_uni400 (H2: prior estrangeiro no transporte)",
                  "populacao": "todas as classes urbanas, AUC agrupada; bootstrap por agrupamento (comunidade FCU ou setor), 2000",
                  "nivel": NIVEL, "efeito_minimo": 0.01,
                  "decisao": "ganho_relevante se inferior > 0,01; ganho se > 0; perda se superior < 0; senao inconclusivo",
                  "nao_testavel": "< 20 positivos ou < 30 agrupamentos com positivo"},
    "secundarios": ["mesmos contrastes por classe (FCU, baixa renda, comparador)", "macro FCU (>=5/5) se >= 8 comunidades",
                    "recusa estratificada (M4) nao aplicavel: sem preditor de competencia ajustado para esta populacao"],
    "parada": "avaliacao unica",
}


def grade():
    import rasterio
    import geopandas as gpd
    import zipfile
    from rasterio import features
    from rasterio.warp import transform as wtransform
    from rasterio.windows import Window, from_bounds
    from tv01_universo_escassez import universo_nacional
    z = RUNS_TV / "fontes/setores_2022/PE_setores_CD2022.zip"
    shp = [n for n in zipfile.ZipFile(z).namelist() if n.endswith(".shp")][0]
    st = gpd.read_file(f"/vsizip/{z.as_posix()}/{shp}", columns=["CD_SETOR"]).merge(universo_nacional(), on="CD_SETOR", how="left")
    st = st[st.NM_CONCURB.eq("Recife/PE") & st.CD_MUN.ne("2611606") & st.classe_territorial.isin(CLASSES)].reset_index(drop=True).to_crs("EPSG:32725")
    st["idx"] = np.arange(1, len(st) + 1)
    partes = []
    for tile, (lo0, lo1, la0, la1) in TILES.items():
        base = REPO / "local_runs/ter-01-cadeia-harmonizada" / tile
        with rasterio.open(base / "hand_dinf.tif") as s:
            w = from_bounds(*st.total_bounds, transform=s.transform).round_offsets().round_lengths().intersection(Window(0, 0, s.width, s.height))
            tr, hand, nod = s.window_transform(w), s.read(1, window=w), s.nodata
        idx = features.rasterize(zip(st.geometry, st.idx), out_shape=hand.shape, transform=tr, fill=0, dtype="int32")
        lin, col = np.indices(hand.shape)
        r0, c0 = int(w.row_off), int(w.col_off)
        par = ((lin + r0) % 2 == 0) & ((col + c0) % 2 == 0) & (idx > 0)
        xs, ys = (np.asarray(v) for v in rasterio.transform.xy(tr, lin[par], col[par]))
        lon, lat = (np.asarray(v) for v in wtransform("EPSG:32725", "EPSG:4326", xs, ys))
        nuc = (lon >= lo0) & (lon < lo1) & (lat >= la0) & (lat < la1)
        df = pd.DataFrame({"tile": tile, "x": xs[nuc], "y": ys[nuc], "lon": lon[nuc], "lat": lat[nuc], "idx": idx[par][nuc], "hand_m": hand[par][nuc].astype("float64")})
        df.loc[df.hand_m == nod, "hand_m"] = np.nan
        for nome, arq in [("slope_deg", "slope_deg_wbt.tif"), ("twi_dinf", "twi_dinf.tif")]:
            with rasterio.open(base / arq) as s:
                a = s.read(1, window=w)
                df[nome] = a[par][nuc].astype("float64")
                df.loc[df[nome] == s.nodata, nome] = np.nan
        partes.append(df)
        print(tile, len(df), flush=True)
    g = pd.concat(partes, ignore_index=True).merge(pd.DataFrame(st.drop(columns="geometry"))[["idx", "CD_SETOR", "CD_MUN", "NM_MUN", "CD_FCU", "NM_FCU", "classe_territorial", "pessoas"]], on="idx").drop(columns="idx")
    for c in ["hand_m", "slope_deg", "twi_dinf"]:
        g.loc[(g[c] < -1000) | ~np.isfinite(g[c]), c] = np.nan
    g["terreno_valido"] = g[["hand_m", "slope_deg", "twi_dinf"]].notna().all(axis=1)
    g.insert(0, "ponto_id", [f"RMR_{i:07d}" for i in range(len(g))])
    g6(D / "grade.parquet", g)
    print(len(g), g.classe_territorial.value_counts().to_dict())


def covariaveis():
    from v2_03_covariaveis import dist_canal, worldcover
    g = pd.read_parquet(D / "grade.parquet", columns=["ponto_id", "tile", "x", "y", "lon", "lat"])
    c = g[["ponto_id"]].copy()
    c["worldcover"] = worldcover(g)
    c["construido"] = (c.worldcover == 50).astype(int)
    c["dist_canal_m"] = dist_canal(g)
    g6(D / "covariaveis.parquet", c)


ADENDO_01 = {
    "versao": "v6-rmr-adendo-01",
    "motivo": "em Recife municipio a cena pre (2022-04-02) era cena de episodio e contava agua do episodio como permanente",
    "regra": "cena pre candidata e descartada se sua data cair em [inicio, janela_fim + 3 dias] de qualquer episodio de v6/recife/episodios.csv",
    "busca_stac": "bbox da grade RMR em vez do bbox de Recife municipio",
    "registrado_antes_da_referencia": True,
}


def protocolo():
    codigo = {p.name: sha_hex(p) for p in [AQUI / "v6_07_rmr.py", AQUI / "v5_aprendizagem.py", AQUI / "v2_aprendizagem.py"]}
    p = congelar6(D / "protocolo.json", {**PROTOCOLO, "codigo_sha256": codigo, "grade_sha": sha_hex(D / "grade.parquet"),
                                         "covariaveis_sha": sha_hex(D / "covariaveis.parquet"), "coeficientes_v5_sha": sha_hex(V5 / "s1_s2/coeficientes.jsonl")})
    print(p["quando_utc"])


def predicoes():
    import v2_aprendizagem as ap
    import v5_aprendizagem as v5
    from comum_tv import bloquear_rede
    from v6_06_metodo import gbm
    bloquear_rede()
    if (D / "referencia.parquet").exists():
        raise SystemExit("referencia ja existe")
    g = pd.read_parquet(D / "grade.parquet").merge(pd.read_parquet(D / "covariaveis.parquet"), on="ponto_id")
    g = g[g.terreno_valido & g.dist_canal_m.notna() & (g.worldcover > 0)].reset_index(drop=True)
    cols = ap.COM_CANAL
    doad = pd.read_parquet(V3A / "doadores.parquet")
    tr = doad[doad.pool.eq("internacional") & doad.papel.eq("ajuste") & doad.dist_canal_m.notna() & doad.construido.notna()]
    esc = ap.Escala(v5.transformar(tr, cols))
    z = lambda q: esc(v5.transformar(q, cols))
    zg = z(g)
    out = g[["ponto_id"]].copy()
    out["A0"] = -g.hand_m.to_numpy()
    th1, _ = v5.ajustar(z(tr), tr.classe.to_numpy(), balancear=True)
    out["A1_log"] = v5.escore(th1, zg)
    w = np.where(tr.classe == 1, len(tr) / (2 * tr.classe.sum()), len(tr) / (2 * (len(tr) - tr.classe.sum())))
    out["GBM_ext"] = gbm().fit(z(tr), tr.classe, sample_weight=w).predict_proba(zg)[:, 1]
    def desc(q):
        return pd.Series({"hand": np.log1p(q.hand_m.clip(lower=0).quantile(.9)), "slope": np.log1p(q.slope_deg.clip(lower=0).median()), "constr": q.construido.mean()})
    alvo = desc(g)
    du = tr.groupby("unidade_doadora").apply(desc, include_groups=False)
    sd = du.std(ddof=0).replace(0, 1)
    sel = list((((du - alvo) / sd) ** 2).sum(axis=1).sort_values(kind="stable").index[:7])
    q = tr[tr.unidade_doadora.isin(sel)]
    thc, _ = v5.ajustar(z(q), q.classe.to_numpy(), balancear=True)
    out["A1_ctx7"] = v5.escore(thc, zg)
    coef = [json.loads(l) for l in (V5 / "s1_s2/coeficientes.jsonl").read_text(encoding="utf-8").splitlines()]
    sel_c = lambda **k: [c for c in coef if all(c.get(a) == b for a, b in k.items())]
    t = v5.transformar(g, cols)
    for nome, pol, b, br in [("A3_rmpa_fcu100", "fcu_comunidades", 100, "A3"), ("A3_rmpa_uni400", "urbano_uniforme", 400, "A3")]:
        out[nome] = np.mean([v5.escore(np.array(c["theta"]), zg) for c in sel_c(pool="internacional", camadas="completo", politica=pol, B=b, braco=br)], axis=0)
    cs = sel_c(pool="internacional", camadas="completo", politica="urbano_uniforme", B=400, braco="A2_estrito")
    out["A2e_rmpa_uni400"] = np.mean([v5.escore(np.array(c["theta"]), ((t[pd.Series(c["escala_media"]).index] - pd.Series(c["escala_media"])) / pd.Series(c["escala_desvio"])).to_numpy()) for c in cs], axis=0)
    g6(D / "predicoes.parquet", out)
    g6(D / "predicoes_registro.json", {"quando_utc": agora(), "sha_predicoes": sha_hex(D / "predicoes.parquet"), "doadores_ctx7": sel,
                                       "descritor_alvo": alvo.to_dict(), "pontos": len(out)})
    print(len(out), sel)


def referencia():
    from rasterio.warp import transform as wtransform
    from tv06_referencia_rmpa import amostrar, indices, ler_janela
    from comum_tv import rotulo_referencia
    import v6_03_recife_requalificacao as rq
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    sha_pred = ler(D / "predicoes_registro.json")["sha_predicoes"]
    g = pd.read_parquet(D / "grade.parquet", columns=["ponto_id", "lon", "lat", "classe_territorial"])
    xs, ys = (np.asarray(v) for v in wtransform("EPSG:4326", "EPSG:32725", g.lon.to_numpy(), g.lat.to_numpy()))
    b = (xs.min() - 200, ys.min() - 200, xs.max() + 200, ys.max() + 200)
    base = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/25/{t1}/{t2}/{a}/{m}/{item}/{banda}.tif"
    def url(item, banda):
        dia, tile = item.split("_")[2], item.split("_")[1]
        return "/vsicurl/" + base.format(t1=tile[2], t2=tile[3:], a=dia[:4], m=int(dia[4:6]), item=item, banda=banda)
    fcu = g.classe_territorial.eq("fcu").to_numpy()
    est = {k: np.full(len(g), "desconhecido", dtype=object) for k in ["estado"]}
    meta = {}
    if (D / "referencia.parquet").exists():
        raise SystemExit("referencia ja existe")
    congelar6(D / "protocolo_adendo_01.json", {**ADENDO_01, "referencia_existia": False})
    ep = pd.read_csv(V6 / "recife/episodios.csv", parse_dates=["inicio", "janela_fim"])
    def em_episodio(f):
        d = pd.Timestamp(f["properties"]["datetime"][:10])
        return bool(((d >= ep.inicio) & (d <= ep.janela_fim + pd.Timedelta(days=3))).any())
    rq.BBOX = [float(g.lon.min()), float(g.lat.min()), float(g.lon.max()), float(g.lat.max())]
    cand = [f for f in rq.stac(rq.ES, "sentinel-2-l2a", "2022-02-27", "2022-05-27") if not em_episodio(f)]
    for tile, ev_item in EVENTO.items():
        cols_ev = {bd: amostrar(*ler_janela(url(ev_item, bd), b, 20.0)[:2], xs, ys, 0) for bd in ["SCL", "B03", "B11", "B08"]}
        pre = None
        for f in sorted([f for f in cand if f"_{tile}_" in f["id"]], key=lambda f: f["properties"]["datetime"], reverse=True):
            try:
                scl, tr, _ = ler_janela(url(f["id"], "SCL"), b, 20.0)
            except Exception:
                continue
            v = amostrar(scl, tr, xs, ys, 0)
            cobre = v[fcu] != 0
            if cobre.any() and np.isin(v[fcu][cobre], [2, 4, 5, 6, 7]).mean() >= .8:
                pre = f["id"]
                break
        if pre is None:
            meta[tile] = {"estado": "sem_pre"}
            continue
        cols_pr = {bd: amostrar(*ler_janela(url(pre, bd), b, 20.0)[:2], xs, ys, 0) for bd in ["SCL", "B03", "B11", "B08"]}
        me, _ = indices(cols_ev["B03"], cols_ev["B11"], cols_ev["B08"])
        mp, _ = indices(cols_pr["B03"], cols_pr["B11"], cols_pr["B08"])
        st = rotulo_referencia(me > 0, mp > 0, np.zeros(len(g), bool), np.isin(cols_ev["SCL"], [2, 4, 5, 6, 7]), np.isin(cols_pr["SCL"], [2, 4, 5, 6, 7]))
        usar = (est["estado"] == "desconhecido") & (cols_ev["SCL"] != 0)
        est["estado"][usar] = st[usar]
        meta[tile] = {"evento": ev_item, "pre": pre, "cobertos_evento": int((cols_ev["SCL"] != 0).sum())}
        print(tile, meta[tile], flush=True)
    out = g[["ponto_id"]].copy()
    out["estado"] = est["estado"]
    g6(D / "referencia.parquet", out)
    g6(D / "referencia.json", {"quando_utc": agora(), "sha_predicoes_ja_registradas": sha_pred, "leu_predicoes": False, "tiles": meta,
                               "estados": out.estado.value_counts().to_dict()})
    print(out.estado.value_counts().to_dict())


ADENDO_02 = {
    "versao": "v6-rmr-adendo-02",
    "motivo": "pela regra congelada + adendo 01 nenhum tile tem cena pre >= 0,8 valida (melhor 0,747, S2B_25LBL_20220422); referencia v1 ficou toda desconhecida e esta preservada; o diagnostico usou so SCL, nenhuma banda de agua",
    "regra_pre": "por ponto: observacao SCL valida mais recente entre as cenas S2 do mesmo tile em [2022-02-27, 2022-05-27], fora de [inicio, janela_fim + 3] dos episodios; permanente se MNDWI > 0 nela; pre_desconhecido se nenhuma",
    "evento": "inalterado (S2B 2022-06-01, 25MBM primeiro e 25LBL nos pontos ainda desconhecidos)",
    "saidas": "referencia_a02.parquet / referencia_a02.json; a avaliacao primaria passa a ler a02",
    "resultado_v1": "nao_testavel (sem cena pre admissivel) - preservado",
    "registrado_antes_de_ler_agua": True,
}


def referencia_a02():
    from rasterio.warp import transform as wtransform
    from tv06_referencia_rmpa import amostrar, indices, ler_janela
    from comum_tv import rotulo_referencia
    import v6_03_recife_requalificacao as rq
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    if (D / "referencia_a02.parquet").exists():
        raise SystemExit("referencia a02 ja existe")
    congelar6(D / "protocolo_adendo_02.json", ADENDO_02)
    sha_pred = ler(D / "predicoes_registro.json")["sha_predicoes"]
    g = pd.read_parquet(D / "grade.parquet", columns=["ponto_id", "lon", "lat"])
    xs, ys = (np.asarray(v) for v in wtransform("EPSG:4326", "EPSG:32725", g.lon.to_numpy(), g.lat.to_numpy()))
    b = (xs.min() - 200, ys.min() - 200, xs.max() + 200, ys.max() + 200)
    base = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/25/{t1}/{t2}/{a}/{m}/{item}/{banda}.tif"
    def url(item, banda):
        dia, tile = item.split("_")[2], item.split("_")[1]
        return "/vsicurl/" + base.format(t1=tile[2], t2=tile[3:], a=dia[:4], m=int(dia[4:6]), item=item, banda=banda)
    def bandas(item):
        return {bd: amostrar(*ler_janela(url(item, bd), b, 20.0)[:2], xs, ys, 0) for bd in ["SCL", "B03", "B11", "B08"]}
    ep = pd.read_csv(V6 / "recife/episodios.csv", parse_dates=["inicio", "janela_fim"])
    def em_episodio(f):
        d = pd.Timestamp(f["properties"]["datetime"][:10])
        return bool(((d >= ep.inicio) & (d <= ep.janela_fim + pd.Timedelta(days=3))).any())
    rq.BBOX = [float(g.lon.min()), float(g.lat.min()), float(g.lon.max()), float(g.lat.max())]
    cand = [f for f in rq.stac(rq.ES, "sentinel-2-l2a", "2022-02-27", "2022-05-27") if not em_episodio(f)]
    estado = np.full(len(g), "desconhecido", dtype=object)
    meta = {}
    for tile, ev_item in EVENTO.items():
        ev = bandas(ev_item)
        pre_val = np.zeros(len(g), bool)
        pre_agua = np.zeros(len(g), bool)
        usadas = []
        for f in sorted([f for f in cand if f"_{tile}_" in f["id"]], key=lambda f: f["properties"]["datetime"], reverse=True):
            try:
                pr = bandas(f["id"])
            except Exception as e:
                usadas.append({"cena": f["id"], "erro": str(e)[:120]})
                continue
            novo = ~pre_val & np.isin(pr["SCL"], [2, 4, 5, 6, 7])
            mp, _ = indices(pr["B03"], pr["B11"], pr["B08"])
            pre_agua[novo] = mp[novo] > 0
            pre_val |= novo
            usadas.append({"cena": f["id"], "pontos_preenchidos": int(novo.sum())})
        me, _ = indices(ev["B03"], ev["B11"], ev["B08"])
        st = rotulo_referencia(me > 0, pre_agua, np.zeros(len(g), bool), np.isin(ev["SCL"], [2, 4, 5, 6, 7]), pre_val)
        usar = (estado == "desconhecido") & (ev["SCL"] != 0)
        estado[usar] = st[usar]
        meta[tile] = {"evento": ev_item, "cenas_pre": usadas, "cobertos_evento": int((ev["SCL"] != 0).sum())}
        print(tile, len(usadas), flush=True)
    out = g[["ponto_id"]].copy()
    out["estado"] = estado
    g6(D / "referencia_a02.parquet", out)
    g6(D / "referencia_a02.json", {"quando_utc": agora(), "sha_predicoes_ja_registradas": sha_pred, "leu_predicoes": False, "tiles": meta,
                                   "estados": out.estado.value_counts().to_dict()})
    print(out.estado.value_counts().to_dict())


def avaliacao():
    from comum_tv import auc_mann_whitney, bloquear_rede, bootstrap_auc, intervalo
    import v5_aprendizagem as v5
    bloquear_rede()
    if ler(D / "referencia_a02.json")["sha_predicoes_ja_registradas"] != sha_hex(D / "predicoes.parquet"):
        raise SystemExit("ordem predicao -> referencia nao confere")
    if (D / "avaliacao.json").exists():
        raise SystemExit("avaliacao unica ja executada")
    g = pd.read_parquet(D / "grade.parquet").merge(pd.read_parquet(D / "predicoes.parquet"), on="ponto_id").merge(pd.read_parquet(D / "referencia_a02.parquet"), on="ponto_id")
    k = g[g.estado.isin(["positivo", "negativo"])].copy()
    k["y"] = k.estado.eq("positivo").astype(int)
    k["grupo"] = np.where(k.classe_territorial.eq("fcu"), "FCU:" + k.CD_FCU.astype(str), "SET:" + k.CD_SETOR.astype(str))
    pares = [("C1", "GBM_ext", "A0"), ("C2", "A1_ctx7", "A1_log"), ("C3", "A3_rmpa_fcu100", "A1_log"), ("C4", "A3_rmpa_uni400", "A2e_rmpa_uni400")]
    cols = sorted({c for _, a, b in pares for c in (a, b)})
    res = {}
    for nome_pop, q, nivel in [("todas", k, NIVEL), ("fcu", k[k.classe_territorial.eq("fcu")], .95), ("baixa_renda", k[k.classe_territorial.eq("baixa_renda")], .95),
                               ("comparador_urbano", k[k.classe_territorial.eq("comparador_urbano")], .95)]:
        gp = q[q.y.eq(1)].grupo.nunique()
        if q.y.sum() < 20 or gp < 30:
            res[nome_pop] = {"estado": "nao_testavel", "positivos": int(q.y.sum()), "agrupamentos_com_positivo": int(gp)}
            continue
        bs = bootstrap_auc(q, "grupo", cols, 2000, 20261021)
        linhas = {}
        for nome, a, b in pares:
            iv = intervalo(bs[:, cols.index(a)] - bs[:, cols.index(b)], nivel)
            dfr = auc_mann_whitney(q.y, q[a]) - auc_mann_whitney(q.y, q[b])
            linhas[nome] = {"a": a, "b": b, "auc_a": auc_mann_whitney(q.y, q[a]), "auc_b": auc_mann_whitney(q.y, q[b]), "diferenca": dfr, "intervalo": iv, "nivel": nivel,
                            "estado": None if iv is None else "ganho_relevante" if iv[0] > .01 else "ganho" if iv[0] > 0 else "perda" if iv[1] < 0 else "inconclusivo"}
        res[nome_pop] = {"positivos": int(q.y.sum()), "n": int(len(q)), "agrupamentos": int(q.grupo.nunique()), "agrupamentos_com_positivo": int(gp),
                         "contrastes": linhas, "auc": {c: auc_mann_whitney(q.y, q[c]) for c in cols}}
    f = k[k.classe_territorial.eq("fcu")]
    pg = v5.auc_por_grupo(f.y.to_numpy(), f.A0.to_numpy(), f.grupo.to_numpy(), 5)
    res["fcu_macro_elegiveis_5x5"] = int(len(pg))
    g6(D / "avaliacao.json", {"quando_utc": agora(), "protocolo_sha256": sha_hex(D / "protocolo.json"), "adendos": ["protocolo_adendo_01.json", "protocolo_adendo_02.json"],
                              "resultado_regra_original": "nao_testavel (referencia.json: sem cena pre admissivel)", "referencia": "referencia_a02.parquet", "resultados": res})
    print(json.dumps(res, default=float, indent=1)[:4000])


if __name__ == "__main__":
    {"grade": grade, "covariaveis": covariaveis, "protocolo": protocolo, "predicoes": predicoes, "referencia": referencia, "referencia_a02": referencia_a02, "avaliacao": avaliacao}[sys.argv[1]]()
