"""Ledger de fontes, grafo de dependencia, particoes e amostras da campanha de encostas.

Nada aqui ajusta modelo. As particoes sao fixadas por semente e por geometria,
antes de qualquer previsao, e gravadas com hash para o protocolo congelar.

Papeis experimentais
--------------------
Idai (doador, Zimbabue, ciclone de 2019): pontos de FONTE (ruptura) contra celulas
  do suporte mapeado fora de qualquer poligono de fonte, percurso ou fluxo. O
  inventario e por evento e completo no suporte declarado pelos autores; ausencia
  de poligono e ausencia mapeada naquele evento, sujeita a omissao do mapeador.
  Blocos de 10 km: ajuste (70% dos positivos), desenvolvimento (15%) e reserva de
  origem/retencao (15%).
Petropolis, teste (episodio de fev/2022): 81 poligonos INPE/Charter de
  deslizamento. Suporte = envoltoria convexa das 85 feicoes do inventario (onde o
  mapeador comprovadamente trabalhou). Fundo = celulas do suporte fora de qualquer
  feicao: NAO rotulado, nao encosta estavel. Poligonos inteiros: ruptura+percurso.
Petropolis, atualizacao local (1933-2015): relatos UFRJ/GEOCART de deslizamento,
  geocodificados por endereco, FORA do suporte de teste + 300 m. Fundo de
  atualizacao = dominio de relato (ate 300 m de qualquer registro do inventario),
  para nao ensinar ao modelo "onde ha gente" em vez de "onde a encosta rompe".
Petropolis, teste secundario de transferencia (so braços sem rotulo local):
  todos os relatos historicos de deslizamento contra o dominio de relato.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pa_comum import (CHARTER, EPSG_IDAI, EPSG_PETROPOLIS, ESTUDO, FONTES, OUT, SEMENTE, TERRENO,
                      Etapa, gravar_json, rel, sha256)
from pa_02_terreno import ATRIBUTOS

REF = OUT / "referencias"
BLOCO_IDAI_M = 10_000
BUFFER_EXCLUSAO_TESTE_M = 300
RAIO_DOMINIO_RELATO_M = 300
RAIO_PROXIMO_RELATO_M = 60
NEG_POR_POS_IDAI = 4


def _grade(regiao):
    import rasterio
    r = rasterio.open(TERRENO / regiao / "atributos_encosta.tif")
    return r


def _pilha(r):
    a = r.read().astype("float32")
    return a


def _rc(r, xs, ys):
    t = r.transform
    colunas = np.floor((np.asarray(xs) - t.c) / t.a).astype(int)
    linhas = np.floor((np.asarray(ys) - t.f) / t.e).astype(int)
    return linhas, colunas


def _feat(pilha, linhas, colunas):
    ok = (linhas >= 0) & (linhas < pilha.shape[1]) & (colunas >= 0) & (colunas < pilha.shape[2])
    out = np.full((len(linhas), pilha.shape[0]), np.nan, dtype="float32")
    out[ok] = pilha[:, linhas[ok], colunas[ok]].T
    return out


def _raster(r, geoms, all_touched=False, valores=None):
    from rasterio.features import rasterize
    if valores is None:
        formas = [(g, 1) for g in geoms]
    else:
        formas = list(zip(geoms, valores))
    return rasterize(formas, out_shape=(r.height, r.width), transform=r.transform, fill=0,
                     all_touched=all_touched, dtype="int32")


def ufrj_gdf():
    import geopandas as gpd
    d = json.loads((FONTES / "ufrj_geocart/eventos_ufrj_geocart_gfi.json").read_text(encoding="utf-8"))
    df = pd.DataFrame([f["properties"] for f in d["features"]])
    df["status_n"] = (df.status.astype(str).str.upper().str.replace("É", "E")
                      .replace({"BAIXO": "BAIXA"}))
    df["precisao_n"] = df.precisao.astype(str).str.replace(" ", "_").replace({"RANGER_INTERPOLATED": "RANGE_INTERPOLATED"})
    df["data_dt"] = pd.to_datetime(df.data, format="%d/%m/%Y", errors="coerce")
    df["localizado"] = (df.latitude != 0) & (df.longitude != 0) & ~df.precisao_n.eq("NAO_LOCALIZADO")
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4674)
    return g.to_crs(EPSG_PETROPOLIS)


def ledger(ufrj, charter, idai_src, idai_ext, muni) -> pd.DataFrame:
    aq = json.loads((FONTES / "aquisicao.json").read_text(encoding="utf-8"))
    L = []
    def add(**k):
        L.append(k)
    add(fonte="INPE/Charter 0864 Eventos_Desastre", arquivo=rel(CHARTER), sha256=sha256(CHARTER),
        evento="Petropolis fev/2022", unidade="poligono", n=int((charter.tipoevento == "Deslizamento").sum()),
        natureza="ruptura+percurso (cicatriz inteira, nao separada)", papel="teste primario (positivos)",
        instrumento="interpretacao de imagem orbital pos-evento (Carta Internacional)", escala="poligonos 100-21.600 m2",
        incerteza="data por poligono e area inspecionada nao declaradas; suporte = envoltoria das feicoes",
        independencia="mesmo episodio de DRM/SGB 2022; nao usado em ajuste")
    add(fonte="INPE/Charter 0864 Eventos_Desastre", arquivo=rel(CHARTER), sha256=sha256(CHARTER),
        evento="Petropolis fev/2022", unidade="poligono", n=int((charter.tipoevento == "Fluxo de detritos").sum()),
        natureza="percurso/deposito (fluxo de detritos)", papel="conexao (contexto); fora do alvo de ruptura",
        instrumento="idem", escala="3 poligonos", incerteza="idem", independencia="idem")
    add(fonte="INPE/Charter 0864 Eventos_Desastre", arquivo=rel(CHARTER), sha256=sha256(CHARTER),
        evento="Petropolis fev/2022", unidade="poligono", n=int((charter.tipoevento != "Deslizamento").sum() - (charter.tipoevento == "Fluxo de detritos").sum()),
        natureza="agua (inundacao)", papel="excluido do fundo de teste; agua sem suporte local",
        instrumento="idem", escala="1 poligono", incerteza="idem", independencia="idem")
    loc = ufrj[ufrj.localizado]
    for tipo, papel in [("deslizamento", "atualizacao local (fora do suporte de teste) e teste secundario de transferencia"),
                        ("inundacao", "define dominio de relato; nao rotulo de encosta"),
                        ("alagamento", "define dominio de relato"), ("sem informacao", "define dominio de relato"),
                        ("deslizamento/inundacao", "define dominio de relato; processo misto")]:
        sub = loc[loc.tipo_de_ev.eq(tipo)]
        add(fonte="UFRJ/GEOCART eventos_ufrj_geocart (SIG municipal)",
            arquivo=aq["ufrj"]["consulta"]["arquivo"], sha256=aq["ufrj"]["consulta"]["sha256"],
            evento=f"historico {sub.data_dt.min():%Y}-{sub.data_dt.max():%Y}" if len(sub) else "historico",
            unidade="relato geocodificado", n=int(len(sub)),
            natureza={"deslizamento": "relato de ruptura com dano (local do dano, nao da cicatriz)",
                      "inundacao": "agua", "alagamento": "agua"}.get(tipo, "desconhecido/misto"),
            papel=papel, instrumento="jornais (IPT) e compilacao GEOCART; geocodificacao por endereco",
            escala="ponto; precisao ROOFTOP a APPROXIMATE",
            incerteza="erro posicional de dezenas a centenas de m; vies de relato para areas ocupadas e estradas",
            independencia="fontes jornalisticas; independentes da imagem de 2022")
    add(fonte="Dille et al. 2026 Idai Landslide Source Points", arquivo=aq["idai"]["arquivos"][0]["arquivo"],
        sha256=aq["idai"]["arquivos"][0]["sha256"], evento="ciclone Idai, mar/2019, Zimbabue", unidade="ponto de fonte",
        n=int(len(idai_src)), natureza="ruptura (fonte)", papel="doador: positivos de ajuste/dev/reserva por bloco",
        instrumento="mapeamento em imagem (inventario dos autores)", escala="ponto",
        incerteza="omissao do mapeador; suporte = envoltoria dos pontos de fonte", independencia="evento e regiao distintos de Petropolis")
    for zona in ["Depletion zone", "Run-out zone", "Debris-rich Flow"]:
        add(fonte="Dille et al. 2026 Idai Landslide Extent Maps", arquivo=aq["idai"]["arquivos"][1]["arquivo"],
            sha256=aq["idai"]["arquivos"][1]["sha256"], evento="ciclone Idai, mar/2019", unidade="poligono",
            n=int((idai_ext.Zone == zona).sum()),
            natureza={"Depletion zone": "ruptura (zona de deplecao)", "Run-out zone": "percurso/deposito",
                      "Debris-rich Flow": "agua com detritos"}[zona],
            papel="excluido do fundo do doador (nao e ausencia)", instrumento="idem", escala="poligono",
            incerteza="idem", independencia="mesma equipe/imagem dos pontos de fonte")
    add(fonte="SIG municipal areas_de_risco_2025_def_civil", arquivo=rel(ESTUDO / "fontes/areas_risco_2025_municipal.geojson"),
        sha256=sha256(ESTUDO / "fontes/areas_risco_2025_municipal.geojson"), evento="zoneamento 2025 (pos-2022)",
        unidade="poligono", n=int(len(muni)), natureza="avaliacao tecnica", papel="contexto tecnico; NAO preditor nem validacao de 2022",
        instrumento="vistoria/zoneamento da Defesa Civil", escala="setor", incerteza="data por feicao ausente",
        independencia="pode ter usado o conhecimento de 2022")
    add(fonte="svc-03 escore_petropolis (modelo fluvial_serra)", arquivo="local_runs/svc-03-grade/escore_petropolis.tif",
        sha256=sha256(OUT.parents[1] / "svc-03-grade/escore_petropolis.tif"), evento="nao se aplica", unidade="celula 120 m",
        n=None, natureza="inferencia de agua (transferencia sem referencia local)", papel="componente agua da resposta territorial",
        instrumento="modelo servido; congelado", escala="120 m", incerteza="sem referencia local", independencia="nao usa encostas")
    add(fonte="SGB 2022 relatorio + 11 anexos; DRM-RJ relatorio final", arquivo=rel(ESTUDO / "fontes"),
        sha256=None, evento="Petropolis 2022", unidade="documento", n=None, natureza="vistoria/avaliacao tecnica (PDF)",
        papel="nao vetorial; sem uso como rotulo", instrumento="vistoria expedita / mapeamento", escala="localidade",
        incerteza="nao e inventario vetorial completo", independencia="mesmo episodio do teste")
    add(fonte="USGS Hurricane Maria (Porto Rico)", arquivo=None, sha256=None, evento="2017", unidade="ponto",
        n=None, natureza="ruptura", papel="nao adquirido: desafio anti-robo (HTTP 403)", instrumento=None,
        escala=None, incerteza=None, independencia=None)
    add(fonte="Landslide4Sense", arquivo=None, sha256=None, evento="varios (sismicos e chuva)", unidade="patch",
        n=None, natureza="deteccao pos-evento", papel="auditado; nao baixado (tarefa e gatilhos incompativeis)",
        instrumento="Sentinel-2 + ALOS", escala="128x128", incerteza=None, independencia=None)
    add(fonte="Ferreira et al. 2025 / Alcantara et al. 2023 (dados)", arquivo=None, sha256=None, evento=None, unidade=None,
        n=None, natureza=None, papel="dados mediante solicitacao; nao solicitados (sem contato externo)",
        instrumento=None, escala=None, incerteza=None, independencia=None)
    return pd.DataFrame(L)


def grafo() -> dict:
    nos = ["charter_2022", "sgb_2022", "drm_2022", "municipal_2025", "ufrj_ipt", "ufrj_geocart", "idai_fontes",
           "idai_extensao", "fabdem", "svc03_agua", "censo_2022"]
    arestas = [
        ("charter_2022", "sgb_2022", "mesmo episodio; equipes distintas; possivel uso de imagens comuns"),
        ("charter_2022", "drm_2022", "mesmo episodio; DRM mapeou cicatrizes do mesmo evento"),
        ("drm_2022", "municipal_2025", "zoneamento posterior pode incorporar o mapeamento DRM/2022"),
        ("charter_2022", "municipal_2025", "zoneamento posterior ao episodio de teste: nao independente"),
        ("ufrj_ipt", "ufrj_geocart", "mesma compilacao (dissertacao UFRJ), mesmo geocodificador"),
        ("idai_fontes", "idai_extensao", "mesma equipe e mesmas imagens"),
        ("fabdem", "charter_2022", "atributo compartilhado por todos os bracos; nao rotulo"),
        ("fabdem", "idai_fontes", "mesmo produto de terreno no doador e no destino"),
        ("fabdem", "svc03_agua", "agua usa outro MDT/derivacao (svc-03); nao compartilha coeficientes"),
    ]
    return {"nos": nos, "arestas": [{"de": a, "para": b, "relacao": c} for a, b, c in arestas],
            "regra": "fontes ligadas por mesma imagem/equipe nao contam como confirmacoes independentes"}


def main() -> int:
    import geopandas as gpd
    from shapely.geometry import box

    e = Etapa("pa_03_referencias")
    REF.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)

    charter = gpd.read_file(CHARTER).to_crs(EPSG_PETROPOLIS).reset_index(drop=True)
    charter["id_feicao"] = np.arange(len(charter))
    ufrj = ufrj_gdf()
    D = FONTES / "idai"
    idai_src = gpd.read_file(D / "idai_fontes/Idai Landslide Source Points.shp").to_crs(EPSG_IDAI)
    idai_ext = gpd.read_file(D / "idai_extensao/Idai Landslide Extent Maps.shp").to_crs(EPSG_IDAI)
    muni = gpd.read_file(ESTUDO / "fontes/areas_risco_2025_municipal.geojson").to_crs(EPSG_PETROPOLIS)
    e.entrada(CHARTER, FONTES / "ufrj_geocart/eventos_ufrj_geocart_gfi.json", D, ESTUDO / "fontes/areas_risco_2025_municipal.geojson")

    lg = ledger(ufrj, charter, idai_src, idai_ext, muni)
    lg.to_csv(REF / "ledger_fontes.csv", index=False, encoding="utf-8-sig")
    gravar_json(REF / "grafo_fontes.json", grafo())

    # ------------------------------------------------------------ Idai
    r = _grade("idai")
    P = _pilha(r)
    suporte = idai_src.union_all().convex_hull
    m_sup = _raster(r, [suporte]).astype(bool) & np.isfinite(P).all(0)
    m_ext = _raster(r, list(idai_ext.geometry), all_touched=True).astype(bool)
    from scipy.ndimage import binary_dilation
    viz = binary_dilation(m_ext, iterations=2)                    # ~60 m
    li, co = _rc(r, idai_src.geometry.x, idai_src.geometry.y)
    pos_cells = pd.DataFrame({"l": li, "c": co}).drop_duplicates()
    pos_cells = pos_cells[(pos_cells.l >= 0) & (pos_cells.l < r.height) & (pos_cells.c >= 0) & (pos_cells.c < r.width)]
    pos_cells = pos_cells[m_sup[pos_cells.l, pos_cells.c]]
    m_pos = np.zeros_like(m_sup)
    m_pos[pos_cells.l, pos_cells.c] = True
    m_pos_viz = binary_dilation(m_pos, iterations=2)
    elegivel_neg = m_sup & ~viz & ~m_pos_viz
    nl, nc = np.nonzero(elegivel_neg)
    k = min(len(nl), NEG_POR_POS_IDAI * len(pos_cells))
    escolha = rng.choice(len(nl), size=k, replace=False)
    neg_cells = pd.DataFrame({"l": nl[escolha], "c": nc[escolha]})
    idai = pd.concat([pos_cells.assign(y=1), neg_cells.assign(y=0)], ignore_index=True)
    xs, ys = r.xy(idai.l.values, idai.c.values)
    idai["x"], idai["yc"] = np.asarray(xs), np.asarray(ys)
    idai[ATRIBUTOS] = _feat(P, idai.l.values, idai.c.values)
    idai = idai.dropna(subset=ATRIBUTOS).reset_index(drop=True)
    idai["bloco"] = ((idai.x // BLOCO_IDAI_M).astype(int).astype(str) + "_" + (idai.yc // BLOCO_IDAI_M).astype(int).astype(str))
    pb = idai.groupby("bloco").y.sum()
    ordem = pb.index.values[rng.permutation(len(pb))]
    acum = pb.loc[ordem].cumsum() / pb.sum()
    papel = np.where(acum <= 0.70, "ajuste", np.where(acum <= 0.85, "desenvolvimento", "reserva_origem"))
    mapa = dict(zip(ordem, papel))
    idai["papel"] = idai.bloco.map(mapa)
    idai["id"] = ["idai_%07d" % i for i in range(len(idai))]
    idai.to_parquet(REF / "amostra_idai.parquet", index=False)
    blocos_idai = idai.groupby(["bloco", "papel"]).y.agg(["sum", "count"]).reset_index()
    blocos_idai.to_csv(REF / "particao_idai_blocos.csv", index=False)
    area_sup_idai = float(suporte.area / 1e6)
    r.close()

    # ------------------------------------------------------------ Petropolis
    r = _grade("petropolis")
    P = _pilha(r)
    valido = np.isfinite(P).all(0)
    hull = charter.union_all().convex_hull
    m_teste = _raster(r, [hull]).astype(bool) & valido
    m_excl = _raster(r, [hull.buffer(BUFFER_EXCLUSAO_TESTE_M)]).astype(bool)
    desl = charter[charter.tipoevento.eq("Deslizamento")]
    lab = _raster(r, list(desl.geometry), valores=list(desl.id_feicao + 1))      # centro da celula
    for _, row in desl.iterrows():                                               # cicatriz sem centro de celula
        if not (lab == row.id_feicao + 1).any():
            p = row.geometry.representative_point()
            li_, co_ = r.index(p.x, p.y)
            lab[li_, co_] = row.id_feicao + 1
    qualquer = _raster(r, list(charter.geometry), all_touched=True).astype(bool)
    tl, tc = np.nonzero(m_teste)
    teste = pd.DataFrame({"l": tl, "c": tc})
    teste["id_cicatriz"] = lab[tl, tc] - 1
    teste["y"] = (teste.id_cicatriz >= 0).astype(int)
    teste["fundo"] = (~qualquer[tl, tc]) & (teste.y == 0)
    teste = teste[(teste.y == 1) | teste.fundo].reset_index(drop=True)
    xs, ys = r.xy(teste.l.values, teste.c.values)
    teste["x"], teste["yc"] = np.asarray(xs), np.asarray(ys)
    teste[ATRIBUTOS] = _feat(P, teste.l.values, teste.c.values)
    teste["bloco500"] = ((teste.x // 500).astype(int).astype(str) + "_" + (teste.yc // 500).astype(int).astype(str))
    teste["id"] = ["t22_%06d" % i for i in range(len(teste))]
    teste.to_parquet(REF / "teste_petropolis_2022.parquet", index=False)

    loc = ufrj[ufrj.localizado].copy()
    li, co = _rc(r, loc.geometry.x, loc.geometry.y)
    loc["l"], loc["c"] = li, co
    loc = loc[(loc.l >= 0) & (loc.l < r.height) & (loc.c >= 0) & (loc.c < r.width)]
    loc = loc[valido[loc.l, loc.c]]
    loc["dentro_exclusao"] = m_excl[loc.l, loc.c]
    desl_h = loc[loc.tipo_de_ev.eq("deslizamento")]
    pool = desl_h[~desl_h.dentro_exclusao & desl_h.status_n.isin(["ALTA", "MEDIA"])]
    pool_sitios = (pool.groupby(["l", "c"]).agg(n_relatos=("id", "size"), ids_ufrj=("id", lambda s: ",".join(map(str, sorted(s)))),
                                               ano_min=("data_dt", "min"), ano_max=("data_dt", "max")).reset_index())
    pool_sitios[ATRIBUTOS] = _feat(P, pool_sitios.l.values, pool_sitios.c.values)
    pool_sitios["id"] = ["h_%04d" % i for i in range(len(pool_sitios))]
    xs, ys = r.xy(pool_sitios.l.values, pool_sitios.c.values)
    pool_sitios["x"], pool_sitios["yc"] = np.asarray(xs), np.asarray(ys)
    pool_sitios.to_parquet(REF / "pool_atualizacao_local.parquet", index=False)

    # dominio de relato: ate 300 m de qualquer registro localizado (qualquer tipo)
    m_rel_pts = np.zeros(valido.shape, bool)
    m_rel_pts[loc.l, loc.c] = True
    from scipy.ndimage import distance_transform_edt
    dist_rel = distance_transform_edt(~m_rel_pts) * 30.0
    m_desl = np.zeros(valido.shape, bool)
    m_desl[desl_h.l, desl_h.c] = True
    dist_desl = distance_transform_edt(~m_desl) * 30.0
    m_dom = (dist_rel <= RAIO_DOMINIO_RELATO_M) & valido
    dl, dc = np.nonzero(m_dom)
    dom = pd.DataFrame({"l": dl, "c": dc})
    dom["dentro_exclusao"] = m_excl[dl, dc]
    dom["perto_deslizamento"] = dist_desl[dl, dc] <= RAIO_PROXIMO_RELATO_M
    dom[ATRIBUTOS] = _feat(P, dl, dc)
    dom["id"] = ["d_%06d" % i for i in range(len(dom))]
    dom.to_parquet(REF / "dominio_relato.parquet", index=False)

    # teste secundario: todos os sitios historicos de deslizamento (sem rotulo local nos bracos avaliados)
    hist = (desl_h.groupby(["l", "c"]).agg(n_relatos=("id", "size")).reset_index())
    hist[ATRIBUTOS] = _feat(P, hist.l.values, hist.c.values)
    hist["id"] = ["hs_%04d" % i for i in range(len(hist))]
    hist.to_parquet(REF / "teste_historico_sitios.parquet", index=False)

    # fundo do municipio inteiro (diagnostico de vies de amostragem)
    vl, vc = np.nonzero(valido & ~m_excl)
    k = 40_000
    esc = rng.choice(len(vl), size=k, replace=False)
    mun = pd.DataFrame({"l": vl[esc], "c": vc[esc]})
    mun[ATRIBUTOS] = _feat(P, mun.l.values, mun.c.values)
    mun["id"] = ["m_%06d" % i for i in range(len(mun))]
    mun.to_parquet(REF / "fundo_municipio_diagnostico.parquet", index=False)
    r.close()

    resumo = {
        "idai": {"suporte_km2": area_sup_idai, "positivos_celulas": int(idai.y.sum()),
                 "negativos_celulas": int((idai.y == 0).sum()),
                 "por_papel": idai.groupby("papel").y.agg(["sum", "count"]).to_dict(),
                 "blocos_por_papel": blocos_idai.groupby("papel").bloco.nunique().to_dict(),
                 "pontos_fonte_total": int(len(idai_src)),
                 "negativo_significa": "celula do suporte a >60 m de qualquer poligono/ponto mapeado no evento Idai"},
        "petropolis_teste_2022": {"suporte_km2": float(hull.area / 1e6), "celulas_suporte_validas": int(m_teste.sum()),
                                  "cicatrizes": int(teste.loc[teste.y == 1, "id_cicatriz"].nunique()),
                                  "celulas_positivas": int(teste.y.sum()), "celulas_fundo": int(teste.fundo.sum()),
                                  "prevalencia_celular": float(teste.y.mean()),
                                  "blocos500": int(teste.bloco500.nunique())},
        "pool_local": {"relatos_deslizamento_localizados": int(len(desl_h)),
                       "dentro_exclusao": int(desl_h.dentro_exclusao.sum()),
                       "relatos_pool_alta_media": int(len(pool)), "sitios_unicos": int(len(pool_sitios)),
                       "relatos_por_sitio_max": int(pool_sitios.n_relatos.max()),
                       "anos": [str(pool.data_dt.min().date()), str(pool.data_dt.max().date())]},
        "dominio_relato": {"celulas": int(len(dom)), "fora_exclusao": int((~dom.dentro_exclusao).sum()),
                           "elegiveis_fundo": int((~dom.dentro_exclusao & ~dom.perto_deslizamento).sum())},
        "teste_historico": {"sitios": int(len(hist)), "relatos": int(hist.n_relatos.sum())},
        "parametros": {"bloco_idai_m": BLOCO_IDAI_M, "buffer_exclusao_teste_m": BUFFER_EXCLUSAO_TESTE_M,
                       "raio_dominio_relato_m": RAIO_DOMINIO_RELATO_M, "raio_proximo_relato_m": RAIO_PROXIMO_RELATO_M,
                       "neg_por_pos_idai": NEG_POR_POS_IDAI, "semente": SEMENTE},
        "hashes": {p.name: sha256(p) for p in sorted(REF.glob("*.parquet"))},
    }
    gravar_json(REF / "referencias.json", resumo)
    e.saida(*REF.glob("*"))
    e.fechar(**{k: v for k, v in resumo.items() if k != "hashes"})
    print(json.dumps({k: v for k, v in resumo.items() if k != "hashes"}, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
