"""Auditoria pre-treino: o que cada dataset observa, o que permite aprender e o que ainda compromete o aprendizado.

-------------------------------------------------------------------------
O QUE ESTE GERADOR FAZ
-------------------------------------------------------------------------

Le os artefatos em disco (nunca numero copiado a mao) e grava:

  local_runs/auditoria-pre-treino/resultado.json     linhagem, checagens, prontidao
  local_runs/auditoria-pre-treino/sensibilidades.json avaliacoes de DESENVOLVIMENTO
  docs/AUDITORIA_PRE_TREINO.md                        a sintese unica, gerada daqui

Cada achado segue: problema -> evidencia -> impacto no aprendizado ->
correcao -> verificacao -> pendencia, e tem uma classe:

  defeito_corrigivel     erro de codigo/protocolo que os dados atuais corrigem
  falta_observacional    nenhuma programacao resolve: falta observar
  limitacao_de_desenho   a pergunta que o dado responde e mais estreita que a dita
  hipotese_nao_testada   plausivel, sem teste ainda

-------------------------------------------------------------------------
O QUE ELE NAO FAZ
-------------------------------------------------------------------------

Nao reroda teste confirmatorio, nao le saida de modelo nos eventos de
teste, nao muda admissao, rotulo nem modelo servido. As sensibilidades usam
so eventos CONHECIDOS (ds06), com regime exploratorio declarado. Nos testes
confirmatorios le apenas a composicao do rotulo (regime hidrico, distancia do
tile ao evento), que nao depende de modelo.

Uso:
    python scripts/metodo/auditoria_pre_treino.py                   # tudo (sensibilidade em cache)
    python scripts/metodo/auditoria_pre_treino.py --refazer-sensibilidades
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "suscetibilidade", "geo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

RUNS = REPO / "local_runs"
SAIDA = RUNS / "auditoria-pre-treino"
DOC = REPO / "docs" / "AUDITORIA_PRE_TREINO.md"
SCRIPTS = REPO / "scripts"
SEMENTE = 20261006

# --------------------------------------------------------------------------
# Linhagem declarada. O estado (existe, hash, consumidores) e verificado no disco.
# `aprende`: o que este dataset pode ensinar a um PESO. Corpus/memoria fica
# separado: reconstruir indice nao e aprendizado de pesos.
# --------------------------------------------------------------------------
DATASETS = (
    {"id": "tabela_unica", "caminho": "ds-05-tabela-unica/tabela_unica_v2.parquet",
     "produtor": "scripts/suscetibilidade/ds05_admissao_consolidacao.py", "papel": "canonico",
     "aprende": "pesos tabulares, so via ds06 e admissao", "chave": "tabela_unica_v2.parquet"},
    {"id": "tabela_admitidos", "caminho": "ds-05-tabela-unica/tabela_unica_admitidos_v2.parquet",
     "produtor": "scripts/suscetibilidade/ds05_admissao_consolidacao.py", "papel": "derivado",
     "aprende": "subconjunto admitido da tabela unica", "chave": "tabela_unica_admitidos_v2"},
    {"id": "pool_fluvial", "caminho": "ds-05-tabela-unica/tabela_unica_pool_fluvial_v2.parquet",
     "produtor": "scripts/suscetibilidade/ds05_admissao_consolidacao.py", "papel": "derivado",
     "aprende": "pool fluvial historico", "chave": "tabela_unica_pool_fluvial_v2"},
    {"id": "ds06_vigente", "caminho": "ds-06-dataset-de-treino/{vigente}/pontos.parquet",
     "produtor": "scripts/suscetibilidade/ds06_dataset_de_treino.py", "papel": "canonico",
     "aprende": "pesos tabulares de suscetibilidade (logistica/GBM)", "chave": "versao_atual"},
    {"id": "anotacoes_ds06", "caminho": "verdade-observacional/anotacoes_ds06_{vigente}.parquet",
     "produtor": "scripts/metodo/verdade_observacional.py", "papel": "derivado",
     "aprende": "nada sozinho: evento canonico e regime hidrico por ponto", "chave": "verdade_observacional"},
    {"id": "gfd_pontos", "caminho": "gfd-brasil/pontos_gfd_com_terreno.csv",
     "produtor": "scripts/geo/gfd02_amostrar_terreno.py", "papel": "fonte_derivada",
     "aprende": "rotulo GFD + terreno + JRC occurrence por ponto", "chave": "pontos_gfd_com_terreno"},
    {"id": "gfd_quarentena", "caminho": "gfd-quarentena/pontos_quarentena.parquet",
     "produtor": "scripts/geo/gfdq02_derivar_quarentena.py", "papel": "teste_confirmatorio_examinado",
     "aprende": "nada: teste lido uma vez (gfdq03)", "chave": "gfd-quarentena"},
    {"id": "rfd_quarentena", "caminho": "rfd-quarentena-brasil/pontos_quarentena.parquet",
     "produtor": "scripts/geo/rfd02_eventos_brasileiros_novos.py", "papel": "teste_confirmatorio_examinado",
     "aprende": "nada: teste lido uma vez (rfd03)", "chave": "rfd-quarentena-brasil"},
    {"id": "conhecidos", "caminho": "reservas-examinadas/conhecidos_2026_10_03.json",
     "produtor": "scripts/metodo/reservas_examinadas.py", "papel": "fronteira_confirmatoria",
     "aprende": "nada: lista imutavel do que ja informou escolha", "chave": "conhecidos_2026_10_03"},
    {"id": "floodnet_janelas", "caminho": "floodnet-03-auditoria/janelas_sensor_dia.parquet",
     "produtor": "scripts/externo/fn02_auditar_serie_floodnet.py", "papel": "canonico",
     "aprende": "contraste pluvial observado sensor x dia (NYC)", "chave": "janelas_sensor_dia"},
    {"id": "ctx04_card", "caminho": "ctx-04-terreno-versus-contexto/card.json",
     "produtor": "scripts/metodo/ctx04_terreno_versus_contexto.py", "papel": "resultado",
     "aprende": "pares evento x posto (contexto fluvial)", "chave": "ctx-04-terreno-versus-contexto"},
    {"id": "n1_amostra", "caminho": "campanha-integrada-20261003/N1/amostra.parquet",
     "produtor": "scripts/metodo/ci_n1_encoder_por_ponto.py", "papel": "derivado",
     "aprende": "pontos GFD com janela 128 px (3,84 km) para o encoder/vizinhanca", "chave": "N1"},
    {"id": "n1_janelas", "caminho": "campanha-integrada-20261003/N1/janelas_f16.npy",
     "produtor": "scripts/metodo/ci_n1_encoder_por_ponto.py", "papel": "cache",
     "aprende": "cache das janelas do N1 (5 camadas wbt30, float16)", "chave": "janelas_f16"},
    {"id": "s1f11_split_teste", "caminho": "n1a-sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv",
     "produtor": "dados publicos Sen1Floods11 v1.1", "papel": "canonico",
     "aprende": "segmentacao de agua S1 (frente A)", "chave": "flood_test_data"},
    {"id": "s1gfloods_inventario", "caminho": "sar-pre-pos/campanha_2026_10_06/inventario.json",
     "produtor": "scripts/sar/dados_pre_pos.py", "papel": "derivado",
     "aprende": "mudanca entre duas datas (PNG renderizado)", "chave": "inventario.json"},
    {"id": "serra03_manifesto", "caminho": "serra-03-rotulo-sensor-duplo/v2/manifesto.json",
     "produtor": "scripts/geo/serra03_rotulo_por_sensor_duplo.py", "papel": "resultado",
     "aprende": "nada admitido: 0 rotulos", "chave": "serra-03-rotulo-sensor-duplo"},
    {"id": "rag_corpus", "caminho": "rag-corpus/manifest.json",
     "produtor": "scripts/recuperacao (indice)", "papel": "corpus_documental",
     "aprende": "NENHUM peso: memoria/indice para a camada de resposta", "chave": "rag-corpus"},
)

# --------------------------------------------------------------------------
# Hash integral com cache por (tamanho, mtime): arquivo nao mudou, hash nao muda.
# --------------------------------------------------------------------------
def _sha256(p: Path, cache: dict) -> str:
    st = p.stat()
    chave = f"{p.relative_to(REPO)}|{st.st_size}|{int(st.st_mtime)}"
    if chave in cache:
        return cache[chave]
    h = hashlib.sha256()
    with p.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 24), b""):
            h.update(bloco)
    cache[chave] = "sha256:" + h.hexdigest()
    return cache[chave]


def _scripts_texto() -> dict:
    out = {}
    for p in SCRIPTS.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        try:
            out[str(p.relative_to(REPO)).replace("\\", "/")] = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
    return out


def consumidores(chave: str, textos: dict, produtor: str) -> list[str]:
    return sorted(k for k, t in textos.items() if chave in t and k != produtor
                  and not k.endswith("auditoria_pre_treino.py"))


def linhagem(cache: dict) -> dict:
    from ds06_dataset_de_treino import versao_atual
    vigente = versao_atual().name
    textos = _scripts_texto()
    itens = []
    for ds in DATASETS:
        caminho = RUNS / ds["caminho"].format(vigente=vigente)
        r = {k: v for k, v in ds.items() if k != "chave"}
        r["caminho"] = str(caminho.relative_to(REPO)).replace("\\", "/")
        r["existe"] = caminho.exists()
        if caminho.exists():
            r["bytes"] = caminho.stat().st_size
            r["sha256"] = _sha256(caminho, cache)
        r["consumidores"] = consumidores(ds["chave"], textos, ds["produtor"])
        r["estado"] = ("ausente" if not caminho.exists() else
                       "sem_consumidor_no_codigo" if not r["consumidores"] and ds["papel"] not in
                       ("teste_confirmatorio_examinado", "resultado", "corpus_documental", "cache") else "em_uso")
        itens.append(r)
    versoes = []
    for d in sorted((RUNS / "ds-06-dataset-de-treino").glob("v*/manifesto.json"), key=lambda p: int(p.parent.name[1:])):
        pv = d.parent / "pontos.parquet"
        fixados = sorted(k for k, t in textos.items() if f"ds-06-dataset-de-treino/{d.parent.name}/" in t)
        versoes.append({"versao": d.parent.name, "sha256": _sha256(pv, cache) if pv.exists() else None,
                        "estado": "vigente" if d.parent.name == vigente else
                        ("historica_com_consumidor_fixo" if fixados else "historica"),
                        "consumidores_que_fixam_esta_versao": fixados})
    return {"ds06_vigente": vigente, "datasets": itens, "versoes_ds06": versoes,
            "cadeia": ("fontes -> aquisicao (geo/*, externo/*) -> processamento (gfd02, cems03, ter02) -> "
                       "rotulos/papel (ds03) -> reducao por fonte (ds04) -> admissao/consolidacao (ds05) -> "
                       "uso por finalidade (admissao_cientifica) -> dataset de treino (ds06) -> recortes "
                       "(generalizacao_por_reservas, curva, cortes_por_evento) -> treino/teste -> "
                       "consumidores (fechamento, svc02, ciclo de contexto)"),
            "separacao": ("corpus documental/RAG (rag-corpus) nao entra em nenhum ajuste de peso; "
                          "os pesos tabulares vem so do ds06, os neurais so do Sen1Floods11/S1GFloods/ter-01")}


# --------------------------------------------------------------------------
# Leitura do que ja existe
# --------------------------------------------------------------------------
def _j(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _km(a, b, c, d) -> float:
    p = math.pi / 180
    h = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def _ds06() -> tuple[pd.DataFrame, str]:
    from ds06_dataset_de_treino import versao_atual
    o = versao_atual()
    return pd.read_parquet(o / "pontos.parquet"), o.name


def _teste(q: str) -> pd.DataFrame:
    t = pd.read_parquet(RUNS / q / "pontos_quarentena.parquet")
    t = t[(t.mecanismo == "FLUVIAL_ENXURRADA") & (t.classe_relevo == "PLANO_OU_ONDULADO")]
    return t.dropna(subset=["hand_m", "twi_dinf", "slope_deg", "elevation_local_m"])


# --------------------------------------------------------------------------
# Checagens. Cada uma devolve evidencia medida; o achado e montado depois.
# --------------------------------------------------------------------------
def chk_dominio_gfd(d: pd.DataFrame) -> dict:
    import verdade_observacional as vo
    codigo = (SCRIPTS / "geo" / "rfd01_replica_gfd.py").read_text(encoding="utf-8")
    g = d[d.fonte == "gfd"]
    return {"mascara_no_codigo": len(re.findall(r"updateMask\(declive\.lte\(5\)\)", codigo)),
            "declarado": vo.DOMINIO_DO_ROTULO["gfd"]["mascara_de_terreno"],
            "declive_fabdem_mediano_por_classe": g.groupby("classe").slope_deg.median().round(3).to_dict(),
            "fracao_declive_fabdem_acima_de_5_por_classe": (g.slope_deg > 5).groupby(g.classe).mean().round(4).to_dict(),
            "classe_relevo_do_gfd_no_ds06": g.classe_relevo.value_counts().to_dict()}


TER01 = RUNS / "ter-01-cadeia-harmonizada"
BLOCO_PX = 8          # 8 x 30 m = 240 m, o suporte do GMTED2010 7,5" (~231 m) que o GFD usa
LIMIAR_GFD_GRAUS = 5.0


def declive_grosso(dem: np.ndarray, lin: int, col: int, px: float) -> float:
    """Declive (graus) na escala de ~240 m: media de blocos 8x8 em 3x3 e diferenca central.

    Aproxima `ee.Terrain.slope(GMTED2010)` com o FABDEM local. Nao e o GMTED: serve para
    comparar populacoes no MESMO procedimento, nao para reproduzir pixel a pixel.
    """
    h = BLOCO_PX * 3 // 2
    w = dem[lin - h:lin + h, col - h:col + h]
    if w.shape != (BLOCO_PX * 3, BLOCO_PX * 3) or not np.isfinite(w).all():
        return np.nan
    m = w.reshape(3, BLOCO_PX, 3, BLOCO_PX).mean(axis=(1, 3))
    passo = BLOCO_PX * px
    dzdx = (m[1, 2] - m[1, 0]) / (2 * passo)
    dzdy = (m[0, 1] - m[2, 1]) / (2 * passo)
    return float(np.degrees(np.arctan(np.hypot(dzdx, dzdy))))


def declive_grosso_nos_pontos(pts: pd.DataFrame, por_classe: int = 150, tiles_max: int | None = None,
                              semente: int = SEMENTE) -> pd.DataFrame:
    """`pts`: tile (nome da pasta do ter-01), x_m, y_m no CRS do tile, classe."""
    import rasterio
    out = []
    tiles = sorted(pts.tile.unique(), key=lambda t: hashlib.sha256(f"{semente}|{t}".encode()).hexdigest())
    for t in tiles[:tiles_max] if tiles_max else tiles:
        arq = TER01 / t / "dem_filled.tif"
        if not arq.exists():
            continue
        g = pts[pts.tile == t]
        g = pd.concat([x.sample(min(len(x), por_classe), random_state=semente) for _, x in g.groupby("classe")])
        with rasterio.open(arq) as r:
            dem = r.read(1).astype("float64")
            if r.nodata is not None:
                dem[dem == r.nodata] = np.nan
            linhas, colunas = rasterio.transform.rowcol(r.transform, g.x_m.to_numpy(), g.y_m.to_numpy())
            px = abs(r.res[0])
        for (_, row), li, co in zip(g.iterrows(), linhas, colunas):
            out.append({"tile": t, "classe": int(row.classe), "declive_240m": declive_grosso(dem, int(li), int(co), px)})
    return pd.DataFrame(out)


def chk_mascara_de_declive(cache_dir: Path = SAIDA) -> dict:
    """O corte de declive do algoritmo virou NoData (fora das duas classes) ou negativo?

    Controle: os pontos da replica (`rfd`), onde o `updateMask` e garantido -- a fracao
    acima de 5 graus neles mede o erro da aproximacao FABDEM-240 m x GMTED. Se os
    NEGATIVOS do treino GFD passarem muito disso e os positivos nao, o pixel ingreme
    entrou como negativo no produto publicado: o rotulo seria circular com o declive.
    """
    alvo = cache_dir / "mascara_de_declive.json"
    feito = _j(alvo)
    if feito:
        return feito
    p = pd.read_csv(RUNS / "gfd-brasil" / "pontos_gfd_com_terreno.csv", usecols=["tile", "x_m", "y_m", "classe_obs"])
    p = p[p.classe_obs.isin(["INUNDACAO", "NAO_INUNDADO_OBSERVADO"])]
    p = p.assign(classe=(p.classe_obs == "INUNDACAO").astype(int), tile="gfd_tile_" + p.tile.astype(str))
    grupos = {"gfd_treino": declive_grosso_nos_pontos(p, tiles_max=150)}
    for nome, q in (("rfd_controle_mascarado", "rfd-quarentena-brasil"), ("gfd_quarentena", "gfd-quarentena")):
        t = pd.read_parquet(RUNS / q / "pontos_quarentena.parquet", columns=["tile", "x_m", "y_m", "classe"])
        grupos[nome] = declive_grosso_nos_pontos(t.dropna())
    r = {"suporte_m": BLOCO_PX * 30, "limiar_graus": LIMIAR_GFD_GRAUS,
         "dem": "FABDEM 30 m do ter-01 (dem_filled), media em blocos de 240 m; o GFD usa GMTED2010 7,5\"",
         "por_grupo": {}}
    for nome, g in grupos.items():
        g = g.dropna()
        r["por_grupo"][nome] = {str(c): {"pontos": int((g.classe == c).sum()),
                                         "acima_do_limiar": round(float((g[g.classe == c].declive_240m > LIMIAR_GFD_GRAUS).mean()), 4),
                                         "p99_graus": round(float(g[g.classe == c].declive_240m.quantile(0.99)), 2)}
                                for c in (0, 1)}
    ctrl = r["por_grupo"]["rfd_controle_mascarado"]
    tre = r["por_grupo"]["gfd_treino"]
    r["leitura"] = ("negativo ingreme alem do controle: o produto publicado deixou pixel ingreme virar negativo"
                    if tre["0"]["acima_do_limiar"] > 3 * max(ctrl["0"]["acima_do_limiar"], 0.005)
                    else "negativo ingreme no nivel do controle mascarado: o corte virou NoData nas duas classes")
    cache_dir.mkdir(parents=True, exist_ok=True)
    alvo.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def chk_regime_hidrico(d: pd.DataFrame) -> dict:
    import verdade_observacional as vo
    g = d[d.fonte == "gfd"][["ponto_id", "classe"]].merge(
        pd.read_csv(vo.PONTOS_GFD, usecols=["ponto_id", "jrc_occurrence"]), on="ponto_id", how="left")
    out = {"treino_gfd_ds06": vo.composicao_hidrica(g)}
    for q in ("gfd-quarentena", "rfd-quarentena-brasil"):
        out[q] = vo.composicao_hidrica(_teste(q))
    return out


def chk_selecao_caso_completo() -> dict:
    import generalizacao_por_reservas as g
    d, _, f = g.carregar()
    _, _, ft = g.carregar(g.TERRENO)
    wc = d.groupby("fonte").size().to_dict()
    exec_ = {}
    for q, nome in (("gfd-quarentena", "gfdq03"), ("rfd-quarentena-brasil", "rfd03")):
        e = _j(RUNS / q / "executado.json") or {}
        exec_[nome] = {"linhas_treino_declaradas": e.get("linhas_treino"),
                       "treino_completo_em": f["populacao_completa_em"],
                       "teste_completo_em": list(g.TERRENO),
                       "mesma_regra": False}
    return {"protocolo_congelado": {"linhas_por_fonte": wc, "perda_por_classe": f["perda_por_classe"]},
            "populacao_de_terreno": {"perda_por_classe": ft["perda_por_classe"]},
            "executores_confirmatorios": exec_}


def chk_identidade() -> dict:
    ident = _j(RUNS / "verdade-observacional" / "identidade.json") or {}
    return {"faixa_temporal_por_fonte": ident.get("faixa_temporal_por_fonte"),
            "pares_equivalentes": ident.get("pares_equivalentes", []),
            "consultados_fora_da_tabela": len(ident.get("consultados_fora_da_tabela", [])),
            "hash": ident.get("hash")}


def _janelas_teste_rfd() -> pd.DataFrame:
    s = _j(RUNS / "rfd-quarentena-brasil" / "selecao.json")
    t = pd.read_parquet(RUNS / "rfd-quarentena-brasil" / "pontos_quarentena.parquet", columns=["evento_id", "lat", "lon"])
    caixa = t.groupby("evento_id").agg(lat_min=("lat", "min"), lat_max=("lat", "max"),
                                       lon_min=("lon", "min"), lon_max=("lon", "max")).reset_index()
    jan = pd.DataFrame([{"evento_id": f"RFDBR:{e['eventid']}", "inicio": pd.Timestamp(e["inicio"]),
                         "fim": pd.Timestamp(e["fim"])} for e in s["eventos"]])
    return jan.merge(caixa, on="evento_id")


def _janelas_teste_gfdq() -> pd.DataFrame:
    """A selecao do gfdq01 nao grava data do evento: so a caixa entra, e a identidade fica no ID.

    Sem data, a equivalencia entre fontes nao e verificavel aqui; a leitura parcial
    (mesmo ID GFD) e."""
    t = pd.read_parquet(RUNS / "gfd-quarentena" / "pontos_quarentena.parquet", columns=["evento_id", "lat", "lon"])
    caixa = t.groupby("evento_id").agg(lat_min=("lat", "min"), lat_max=("lat", "max"),
                                       lon_min=("lon", "min"), lon_max=("lon", "max")).reset_index()
    return caixa.assign(inicio=pd.NaT, fim=pd.NaT)


def chk_testes_confirmatorios() -> dict:
    """Composicao do rotulo e identidade dos eventos de teste. Nao le saida de modelo."""
    import reservas_examinadas as rx
    import verdade_observacional as vo
    s = _j(RUNS / "rfd-quarentena-brasil" / "selecao.json")
    dist = []
    for e in s["eventos"]:
        for tl in e["tiles"]:
            x0, y0, x1, y1 = tl["tile_1grau"]
            dist.append({"evento": f"RFDBR:{e['eventid']}", "tile": tl["tile"],
                         "km_do_centroide_gdacs": round(_km(e["lat"], e["lon"], (y0 + y1) / 2, (x0 + x1) / 2), 1)})
    km = np.array([x["km_do_centroide_gdacs"] for x in dist])
    # A regra declarada do rfd02 (500 km, 45 dias, "evento conhecido de qualquer fonte") aplicada
    # tambem aos eventos CONSULTADOS fora da tabela (pacote de serra): o que ela teria tirado.
    cons = vo.consultados()
    cons = cons.dropna(subset=["inicio"]) if len(cons) else cons
    regra_rfd02 = []
    for e in s["eventos"]:
        for _, c in cons.iterrows():
            cy, cx = (c.lat_min + c.lat_max) / 2, (c.lon_min + c.lon_max) / 2
            dias = abs((pd.Timestamp(e["inicio"]) - c.inicio).days)
            km_ = _km(e["lat"], e["lon"], cy, cx)
            if km_ <= 500 and dias <= 45:
                regra_rfd02.append({"evento": f"RFDBR:{e['eventid']}", "consultado": c.evento_id,
                                    "dias": int(dias), "km": round(km_, 1)})
    # O proprio exame (registrar_exame depois da leitura) e ignorado: a pergunta e o que a
    # identidade entre fontes teria dito ANTES da leitura.
    reg_rfd = rx.regime([f"RFDBR:{e['eventid']}" for e in s["eventos"]], "rfd", examinados=set(),
                        janelas=_janelas_teste_rfd())
    jg = _janelas_teste_gfdq()
    reg_gfdq = rx.regime(sorted(jg.evento_id), "gfd", examinados=set(), janelas=jg) if len(jg) else None
    return {"rfd_distancia_tile_centroide_km": {"tiles": int(len(km)), "mediana": float(np.median(km)),
                                                "p25": float(np.percentile(km, 25)), "max": float(km.max()),
                                                "fracao_acima_de_200km": round(float((km > 200).mean()), 3)},
            "rfd_regra_declarada_aplicada_aos_consultados": regra_rfd02,
            "rfd_regime_com_identidade_entre_fontes_ignorando_o_proprio_exame": reg_rfd,
            "gfdq_regime_com_identidade_entre_fontes_ignorando_o_proprio_exame": reg_gfdq}


def chk_s1gfloods(cache: dict) -> dict:
    """Tiles adjacentes entre particoes: borda direita de um chip contra a esquerda de outro.

    Nulo: a mesma borda contra a mesma borda (direita x direita) nunca e adjacencia.
    """
    inv_p = RUNS / "sar-pre-pos" / "campanha_2026_10_06" / "inventario.json"
    inv = _j(inv_p)
    if not inv:
        return {"estado": "inventario ausente"}
    h = _sha256(inv_p, cache)
    alvo = SAIDA / "s1gfloods_adjacencia.json"
    feito = _j(alvo)
    if feito and feito.get("inventario_sha256") == h:
        return feito
    r = adjacencia_entre_particoes(Path(inv["raiz"]) / "B", [p["nome"] for p in inv["pares"]],
                                   np.array([p["split"] for p in inv["pares"]]))
    r["inventario_sha256"] = h
    SAIDA.mkdir(parents=True, exist_ok=True)
    alvo.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def _z(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    s = np.linalg.norm(a, axis=1, keepdims=True)
    s[s < 1e-6] = np.inf
    return a / s


def adjacencia(bordas: dict, corr_min: float = 0.9, mad_max: float = 12.0) -> dict:
    """Pares (i, j) em que a borda de saida de i continua na borda de entrada de j, mutuamente."""
    pares = {}
    for sai, entra in (("R", "L"), ("B", "T")):
        A, Bm = _z(bordas[sai]), _z(bordas[entra])
        C = A @ Bm.T
        np.fill_diagonal(C, -1)
        j, i2 = C.argmax(1), C.argmax(0)
        c = C[np.arange(len(j)), j]
        mad = np.abs(bordas[sai] - bordas[entra][j]).mean(1)
        ok = (c > corr_min) & (mad < mad_max) & (i2[j] == np.arange(len(j)))
        pares[sai + entra] = [(int(a), int(j[a])) for a in np.flatnonzero(ok)]
    nulo = {}
    for lado in ("R", "T"):
        A = _z(bordas[lado])
        C = A @ A.T
        np.fill_diagonal(C, -1)
        j = C.argmax(1)
        c = C[np.arange(len(j)), j]
        mad = np.abs(bordas[lado] - bordas[lado][j]).mean(1)
        nulo[lado + lado] = int(((c > corr_min) & (mad < mad_max)).sum())
    return {"pares": pares, "nulo": nulo}


def adjacencia_entre_particoes(pasta: Path, nomes: list, split: np.ndarray) -> dict:
    from PIL import Image
    bordas = {k: [] for k in "LRTB"}
    for n in nomes:
        b = np.asarray(Image.open(pasta / n).convert("L"), dtype=np.float32)
        bordas["L"].append(b[:, 0]); bordas["R"].append(b[:, -1])
        bordas["T"].append(b[0, :]); bordas["B"].append(b[-1, :])
    bordas = {k: np.array(v) for k, v in bordas.items()}
    a = adjacencia(bordas)
    todos = [p for v in a["pares"].values() for p in v]
    cruzam = [(i, j) for i, j in todos if split[i] != split[j]]
    teste_viz = sorted({nomes[x] for i, j in cruzam for x, y in ((i, j), (j, i))
                        if split[x] == "teste" and split[y] == "treino"})
    return {"chips": len(nomes), "pares_adjacentes": len(todos), "pares_que_cruzam_particao": len(cruzam),
            "nulo_mesma_borda": a["nulo"], "chips_de_teste": int((split == "teste").sum()),
            "chips_de_teste_com_vizinho_no_treino": len(teste_viz),
            "regra": "borda que continua na outra: correlacao > 0,9, diferenca media < 12 niveis, melhor par mutuo"}


def chk_floodnet() -> dict:
    aud = _j(RUNS / "floodnet-03-auditoria" / "resumo.json") or {}
    f1 = (SCRIPTS / "metodo" / "ci_f1_floodnet_negativo_observado.py").read_text(encoding="utf-8")
    fn2 = (SCRIPTS / "externo" / "fn02_auditar_serie_floodnet.py").read_text(encoding="utf-8")
    return {"janelas": aud.get("por_classe"), "criterios": aud.get("criterios"),
            "f1_usa_so_lamina_e_sem_lamina": 'isin(["lamina", "sem_lamina"])' in f1,
            "nao_observado_nunca_negativo": '"nao_observado"' in fn2 and "~j.funcionando" in fn2,
            "dia": "UTC nas duas pontas (fn02 floor('D') em UTC; MRMS 24h 00Z-00Z)",
            "f1": (_j(RUNS / "campanha-integrada-20261003" / "F1" / "card.json") or {}).get("forcamento_dentro_do_sensor"),
            "f1s": (_j(RUNS / "campanha-integrada-20261003" / "F1s" / "card.json") or {}).get("forcamento_dentro_do_sensor")}


def chk_floodnet_defasagem() -> dict:
    """Lamina num dia UTC com a chuva no dia anterior: o evento cruzou 00Z (20h em Nova York).

    Diagnostico post hoc, sem ajuste: conta quantas janelas `lamina` tem chuva MRMS
    < 5 mm no proprio dia e >= 10 mm no dia anterior, e mede a AUC da chuva dentro do
    sensor com max(D-1, D) ao lado da do F1 (so D). Nao substitui o F1.
    """
    import ci_f1_floodnet_negativo_observado as f1
    from sklearn.metrics import roc_auc_score
    j = pd.read_parquet(f1.AUD)
    j["dia"] = pd.to_datetime(j.dia, utc=True)
    d = j[j.classe.isin(["lamina", "sem_lamina"])].reset_index(drop=True)
    d["y"] = (d.classe == "lamina").astype(int)
    d["chuva_d"] = f1.chuva(d)
    d["chuva_d1"] = f1.chuva(d.assign(dia=d.dia - pd.Timedelta(days=1)))
    d = d.dropna(subset=["chuva_d"])
    lam = d[d.y == 1]
    cruzou = int(((lam.chuva_d < 5) & (lam.chuva_d1 >= 10)).sum())

    def auc(col):
        v = [roc_auc_score(g.y, g[col]) for _, g in d.dropna(subset=[col]).groupby("deployment_id")
             if g.y.nunique() == 2 and g.y.sum() >= 3]
        return {"sensores": len(v), "auc_media": round(float(np.mean(v)), 4) if v else None}

    d["chuva_max_d1_d"] = d[["chuva_d", "chuva_d1"]].max(axis=1)
    return {"janelas_lamina": int(len(lam)), "lamina_com_chuva_so_no_dia_anterior": cruzou,
            "fracao": round(cruzou / max(len(lam), 1), 3),
            "auc_dentro_do_sensor_so_D": auc("chuva_d"), "auc_dentro_do_sensor_max_D1_D": auc("chuva_max_d1_d"),
            "regime": "post hoc, exploratorio; o F1 congelado continua valendo como foi"}


def chk_serra() -> dict:
    """Le as versoes v1/v2 (as que ci_e3 e o fechamento leem); a copia antiga da raiz esta em quarentena."""
    base = RUNS / "serra-03-rotulo-sensor-duplo"
    vers = {v: _j(base / v / "manifesto.json") or {} for v in ("v1", "v2")}
    e3 = _j(RUNS / "campanha-integrada-20261003" / "E3" / "card.json") or {}
    ing = (e3.get("cems_por_contexto") or {}).get("INGREME", {})
    return {"rotulos_admitidos": {v: m.get("rotulos_admitidos") for v, m in vers.items()},
            "testavel": {v: m.get("testavel") for v, m in vers.items()},
            "grupos_por_classe": {v: m.get("grupos_por_classe") for v, m in vers.items()},
            "copia_divergente_em_quarentena": _j(base / "_quarentena" / "MOTIVO.json"),
            "raiz_sem_manifesto": not (base / "manifesto.json").exists(),
            "cems_ingreme_positivos": ing.get("positivos"),
            "pacote_fundo_de_vale": {k: v.get("fracao_fundo_de_vale") for k, v in
                                     ((e3.get("pacote_de_adjudicacao") or {}).get("por_evento") or {}).items()}}


def chk_contexto_fluvial() -> dict:
    c = _j(RUNS / "ctx-04-terreno-versus-contexto" / "card.json") or {}
    recus = c.get("montagem", {}).get("pares_recusados", [])
    motivos = pd.Series([r["motivo"] for r in recus]).value_counts().to_dict() if recus else {}
    return {"desenho": c.get("desenho"), "pares_recusados_por_motivo": motivos}


def chk_vizinhanca() -> dict:
    base = RUNS / "campanha-integrada-20261003"
    n1b, v2b = _j(base / "N1b" / "card.json") or {}, _j(base / "V2b" / "card.json") or {}
    return {"resolucao_do_rotulo_gfd_m": 250, "janela_m": 3840,
            "n1b_diferencas": n1b.get("diferencas_ic95_por_evento"),
            "v2b_dentro_do_evento_todas_as_escalas": {f: (v2b.get("resultado") or {}).get(f, {}).get("terreno_mais_todas_as_escalas")
                                                      for f in ("gfd", "cems")}}


def chk_admissao(d: pd.DataFrame) -> dict:
    import admissao_cientifica as adm
    v = adm.validade(d)
    return {"linhas": int(len(d)), "falham_treino": int((~v.uso_treino).sum()),
            "motivos": v.loc[~v.uso_treino, "motivo_treino"].value_counts().head(5).to_dict()}


def chk_consumidores() -> dict:
    fech = (SCRIPTS / "metodo" / "relatorio_de_fechamento.py").read_text(encoding="utf-8")
    camp = (SCRIPTS / "metodo" / "relatorio_campanha_integrada.py").read_text(encoding="utf-8")
    return {"fechamento_fixa_versao_do_ds06": bool(re.search(r"ds-06-dataset-de-treino/v\d+/", fech)),
            "campanha_separa_falha_de_desenho": "falha de desenho" in camp}


# --------------------------------------------------------------------------
# Sensibilidades de DESENVOLVIMENTO (eventos conhecidos; exploratorio).
# --------------------------------------------------------------------------
MODELOS_TERRENO = ("M0_regra_hand", "M1_logistica_terreno", "M3_gbm_terreno")


def _cv_terreno(s0: pd.DataFrame, k: int, sal: str = "a1") -> dict:
    import generalizacao_por_reservas as g
    atrib = g.dobras(sorted(s0.evento_id.unique()), k, sal)
    s = s0.assign(dobra=s0.evento_id.map(atrib)).reset_index(drop=True)
    pred = {m: np.full(len(s), np.nan) for m in MODELOS_TERRENO}
    for f in range(k):
        teste_ev = sorted(e for e, x in atrib.items() if x == f)
        tr, te = g.separar(s, teste_ev)
        pos = np.flatnonzero(s.dobra.to_numpy() == f)
        for m in MODELOS_TERRENO:
            fn = g.ajustar_modelo(m, tr)
            if fn is not None:
                pred[m][pos] = fn(te)
    return {"s": s, "pred": pred}


def _comparacoes(s: pd.DataFrame, pred: dict, rng) -> dict:
    import generalizacao_por_reservas as g
    out = {"medidas": {m: {k: v for k, v in x.items() if k in ("auc_agrupada_fora_da_dobra",
                                                                "auc_mediana_dentro_do_evento", "eventos_com_auc")}
                       for m, x in g.medir(s, pred).items()}}
    for a, b in (("M1_logistica_terreno", "M0_regra_hand"), ("M3_gbm_terreno", "M1_logistica_terreno")):
        c = g.comparar(s, pred[a], pred[b], rng)
        out[f"{a}|{b}"] = {k: c.get(k) for k in ("eventos_pareados", "eventos_a_favor", "dentro_do_evento_mediana",
                                                "ic95_media_dentro", "p_wilcoxon")} | {"veredito": g.veredito(c)}
    return out


def sensibilidades() -> dict:
    """S1 populacao (caso completo x terreno inteiro) e S2 regime hidrico, no GFD conhecido."""
    import generalizacao_por_reservas as g
    import reservas_examinadas as rx
    import verdade_observacional as vo
    inicio = time.time()
    rng = np.random.default_rng(SEMENTE)
    out = {"regime": "exploratorio", "por_que": "eventos conhecidos do ds06; nenhum evento de teste confirmatorio",
           "semente": SEMENTE, "estimadores": "os de generalizacao_por_reservas (M0, M1, M3), dobras a1"}
    populacoes = {}
    for nome, vetor in (("caso_completo_do_protocolo", g.TODAS), ("terreno_completo", g.TERRENO)):
        d, versao, f = g.carregar(vetor)
        for fonte in ("gfd", "cems"):
            s0 = d[d.fonte == fonte]
            cv = _cv_terreno(s0, min(g.DOBRAS[fonte], s0.evento_id.nunique()))
            populacoes[(nome, fonte)] = cv
            out[f"S1_{fonte}_{nome}"] = {"linhas": int(len(s0)), "eventos": int(s0.evento_id.nunique()),
                                         "fracao_positiva": round(float(s0.classe.mean()), 4),
                                         "perda_por_classe": f["perda_por_classe"][fonte],
                                         "regime_dos_eventos": rx.regime(sorted(s0.evento_id.unique()), fonte)["regime"],
                                         **_comparacoes(cv["s"], cv["pred"], rng)}
    for fonte in ("gfd", "cems"):
        if out[f"S1_{fonte}_caso_completo_do_protocolo"]["linhas"] == out[f"S1_{fonte}_terreno_completo"]["linhas"]:
            raise RuntimeError(f"{fonte}: as duas populacoes sao iguais -- o corte nao esta sendo isolado")
    cv = populacoes[("terreno_completo", "gfd")]
    s = cv["s"].merge(pd.read_csv(vo.PONTOS_GFD, usecols=["ponto_id", "jrc_occurrence"]), on="ponto_id", how="left")
    reg = vo.regime_hidrico(s.jrc_occurrence.to_numpy()).to_numpy()
    estratos = {"positivo_nunca_ou_raro": np.isin(reg, ["nunca_agua", "agua_rara"]),
                "positivo_recorrente": reg == "agua_recorrente"}
    for nome, mask_pos in estratos.items():
        manter = (s.classe.to_numpy() == 0) | ((s.classe.to_numpy() == 1) & mask_pos)
        ss = s[manter].reset_index(drop=True)
        pp = {m: p[manter] for m, p in cv["pred"].items()}
        out[f"S2_{nome}"] = {"positivos": int(ss.classe.sum()), "negativos_todos": int((ss.classe == 0).sum()),
                             **_comparacoes(ss, pp, rng)}
    out["segundos"] = round(time.time() - inicio)
    return out


def chk_alvos(d: pd.DataFrame) -> dict:
    """Alvo por tarefa, cortes e a composicao do rotulo por corte e por epoca do JRC. Nenhuma metrica de modelo."""
    import verdade_observacional as vo
    occ = pd.read_csv(vo.PONTOS_GFD, usecols=["ponto_id", "jrc_occurrence"])
    g = d[d.fonte == "gfd"][["ponto_id", "classe", "janela_inicio"]].merge(occ, on="ponto_id")
    s = _j(RUNS / "rfd-quarentena-brasil" / "selecao.json")
    ini = {f"RFDBR:{e['eventid']}": e["inicio"] for e in s["eventos"]}
    t = _teste("rfd-quarentena-brasil")
    t = t.assign(dentro=vo.epoca_jrc_contem(t.evento_id.map(ini)).to_numpy())
    q = _teste("gfd-quarentena")
    return {"alvos": vo.ALVOS, "cortes": vo.CORTES, "jrc": vo.JRC,
            "por_corte": {"treino_gfd": vo.composicao_por_corte(g.jrc_occurrence, g.classe),
                          "teste_sul_americano": vo.composicao_por_corte(q.jrc_occurrence, q.classe),
                          "teste_brasil": vo.composicao_por_corte(t.jrc_occurrence, t.classe)},
            "epoca": {"treino_gfd_dentro_da_epoca": round(float(vo.epoca_jrc_contem(g.janela_inicio).mean()), 3),
                      "teste_brasil_linhas_dentro": round(float(t.dentro.mean()), 3),
                      "teste_brasil_positivos_recorrentes_dentro": vo.composicao_hidrica(t[t.dentro])["1"]["agua_recorrente"],
                      "teste_brasil_positivos_recorrentes_fora": vo.composicao_hidrica(t[~t.dentro])["1"]["agua_recorrente"]}}


def chk_perda_por_contexto() -> dict:
    """Perda do caso completo por fonte x territorio x classe, contra a populacao de terreno."""
    import generalizacao_por_reservas as g
    a, _, _ = g.carregar(g.TERRENO)
    b, _, _ = g.carregar(g.TODAS)
    out = {}
    for (f, t), x in a.groupby(["fonte", "territorio"]):
        y = b[(b.fonte == f) & (b.territorio == t)]
        out[f"{f}|{t}"] = {"linhas": int(len(x)), **g.perda_por_classe(x, y)}
    return out


# Consumidor -> como define a populacao. A coluna `prova` e um trecho que TEM de estar no codigo:
# se o codigo mudar, a checagem acusa e a tabela deixa de valer.
CONSUMIDORES_POPULACAO = (
    ("generalizacao_por_reservas v1", "metodo/generalizacao_por_reservas.py", "def carregar(vetor_da_populacao=TODAS)",
     "caso completo nas 5 comparacoes (pre-registro de 03/10)", "preservado como medido"),
    ("generalizacao_por_reservas v2", "metodo/generalizacao_por_reservas.py", "d, versao, filtro = carregar(TERRENO)",
     "terreno nas comparacoes de terreno; cobertura/solo citadas da v1", "migrado em 07/10"),
    ("gfdq03 / rfd03", "metodo/gfdq03_teste_confirmatorio.py", "d, origem, _ = g.carregar()",
     "treino em caso completo, teste em terreno", "registrado; confirmatorio nao se refaz"),
    ("resposta_lugar_novo", "metodo/resposta_lugar_novo.py", "gen.carregar(POPULACAO_DO_MODELO_CONFIRMADO)",
     "a do modelo confirmado (caso completo)", "amarrado de proposito: outra populacao seria variante nao confirmada"),
    ("politica_sob_orcamento v1", "metodo/politica_sob_orcamento.py",
     '"v1": {"saida": SAIDA, "populacao": POPULACAO_DO_PRE_REGISTRO',
     "caso completo (pre-registro de 03/10)", "preservado como medido"),
    ("politica_sob_orcamento v2", "metodo/politica_sob_orcamento.py",
     '"v2": {"saida": SAIDA / "v2_populacao_da_tarefa", "populacao": gen.TERRENO, "espacial": True}',
     "terreno; externo sem tile do treino pela coordenada", "migrado em 07/10 com pre-registro proprio"),
    ("recusa_por_dominio v1", "metodo/recusa_por_dominio.py",
     '"v1": {"saida": SAIDA, "populacao": POPULACAO_DO_PRE_REGISTRO',
     "caso completo (pre-registro de 03/10)", "preservado como medido"),
    ("recusa_por_dominio v2", "metodo/recusa_por_dominio.py",
     '"v2": {"saida": SAIDA / "v2_populacao_da_tarefa", "populacao": gen.TERRENO, "espacial": True}',
     "terreno; entre fontes sem tile do treino pela coordenada", "migrado em 07/10 com pre-registro proprio"),
    ("curva_de_aprendizado", "metodo/curva_de_aprendizado.py", 'adm.filtrar(s, "avaliacao_espacial", list(VETOR))',
     "vetor de terreno na admissao; caso completo so na ablacao de cobertura/solo", "ja na populacao da tarefa"),
    ("sinais_marginais", "metodo/sinais_marginais.py", 'groupby(["contexto", "fonte"])',
     "cada variavel onde existe; contexto x fonte", "nao afetado: nenhuma celula mistura fontes"),
    ("celula_para_treino", "metodo/celula_para_treino.py", 'adm.filtrar(d, "treino", FEATURES)',
     "o vetor do modelo da celula", "nao afetado por populacao; identidade: ver cascata"),
    ("ci_e2", "metodo/ci_e2_sensibilidade_negativo.py", 'adm.filtrar(flu, "treino", VETOR)',
     "o vetor do E2 (hand, twi)", "identidade corrigida no E2c"),
    ("ctx04", "metodo/ctx04_terreno_versus_contexto.py", 'adm.filtrar(d, "avaliacao_espacial", TERRENO)',
     "terreno", "ja na populacao da tarefa"),
)


def chk_populacoes() -> list[dict]:
    out = []
    for nome, arq, prova, populacao, estado in CONSUMIDORES_POPULACAO:
        texto = (SCRIPTS / arq).read_text(encoding="utf-8")
        out.append({"consumidor": nome, "arquivo": f"scripts/{arq}", "populacao": populacao, "estado": estado,
                    "prova_no_codigo": prova in texto})
    return out


def _veredito_a1(r: dict | None, fonte: str, chave: str):
    try:
        c = r["estudos"][fonte]["atribuicoes"]["a1"]["comparacoes"][chave]
        return {"veredito": c["veredito"], "mediana": c["dentro_do_evento_mediana"], "ic": c["ic95_media_dentro"],
                "a_favor": f"{c['eventos_a_favor']}/{c['eventos_pareados']}"}
    except (KeyError, TypeError):
        return None


def chk_cascata() -> dict:
    """O que foi rerodado por causa das correcoes, contra o resultado preservado."""
    base = RUNS / "campanha-integrada-20261003"
    e2, e2c = _j(base / "E2" / "card.json"), _j(base / "E2c" / "card.json")
    v1 = _j(RUNS / "generalizacao-por-reservas" / "resultado.json")
    v2 = _j(RUNS / "generalizacao-por-reservas" / "v2_populacao_da_tarefa" / "resultado.json")
    comp = {}
    for fonte in ("gfd", "cems"):
        for chave in ("M1_logistica_terreno_vs_M0_regra_hand", "M3_gbm_terreno_vs_M1_logistica_terreno"):
            comp[f"{fonte}|{chave}"] = {"v1_caso_completo": _veredito_a1(v1, fonte, chave),
                                        "v2_populacao_de_terreno": _veredito_a1(v2, fonte, chave),
                                        "eventos_v1": (v1 or {}).get("estudos", {}).get(fonte, {}).get("eventos"),
                                        "eventos_v2": (v2 or {}).get("estudos", {}).get(fonte, {}).get("eventos")}
    ef = {f"{x['treino']}->{x['teste']}": {k: x.get(k) for k in ("linhas_teste", "linhas_retiradas_por_lugar",
                                                                 "chave_de_lugar", "estado")}
          | {"auc_mediana_dentro": {m: x["medidas"][m]["auc_mediana_dentro_do_evento"] for m in x.get("medidas", {})}}
          for x in (v2 or {}).get("entre_fontes", [])}
    terr = {t["territorio"]: (t.get("M1_vs_M0") or {}).get("veredito", t.get("estado"))
            for t in (v2 or {}).get("estudos", {}).get("gfd", {}).get("territorio_reservado", [])}
    return {"e2": {"original": (e2 or {}).get("diferenca_B_menos_A_ic95_por_evento"),
                   "e2c": (e2c or {}).get("diferenca_B_menos_A_ic95_por_evento"),
                   "eventos_unidos": (e2c or {}).get("eventos_canonicos_unidos"),
                   "relacao_A_condicional": {"original": ((e2 or {}).get("relacao_fisica") or {}).get("A", {}).get("condicional"),
                                             "e2c": ((e2c or {}).get("relacao_fisica") or {}).get("A", {}).get("condicional")},
                   "estado": "medido" if e2c else "pendente (na fila da trava)"},
            "generalizacao": {"comparacoes_a1": comp, "entre_fontes_v2": ef, "territorio_reservado_v2": terr,
                              "estado": "medido" if v2 else "pendente (na fila da trava)"},
            "nao_rerodados": {
                "sinais_marginais": "cada celula e contexto x fonte: o par GFD:4577/S1F11:Bolivia nunca cai junto",
                "relacoes": "depende de sinais e celulas, nenhum dos dois mudou",
                "curva_de_aprendizado": "estudos de uma fonte so e populacao ja no vetor de terreno",
                "celulas_de_contexto": ("43 linhas S1F11:Bolivia na celula norte, a 234 km dos pontos do GFD:4577 "
                                        "(6.092 linhas): dependencia de evento sem pixel compartilhado; nao reajustado "
                                        "por um efeito abaixo da resolucao (o E2c, com o mesmo tipo de uniao, mudou "
                                        "a AUC na quarta casa)"),
                "politica_e_recusa": "pre-registros de 03/10 amarrados ao caso completo; migrar exige pre-registro novo",
                "confirmatorios": "nao se refazem"}}


def chk_sar_sessao() -> dict:
    """Le (sem tocar) o card da sessao SAR de independencia por evento."""
    c = _j(RUNS / "sar-independencia" / "campanha_2026_10_06" / "card.json")
    if not c:
        return {"estado": "card ausente"}
    i = c.get("independencia", {})
    return {"particao": "evento inteiro no teste; area sobreposta e copia de imagem proibidas entre particoes "
                        "(scripts/sar/dados_pares_geograficos.conferir_particoes)",
            "pre_pos_menos_pos_por_evento": {k: i.get(k) for k in ("n_eventos", "media", "ic95_bootstrap_evento",
                                                                   "p_wilcoxon", "estado")},
            "eventos_a_favor": sum(v > 0 for v in (i.get("diferencas_por_evento") or {}).values()),
            "regime": c.get("regime"), "promovido": c.get("promovido")}


def chk_migracoes_v2() -> dict:
    """Politica e recusa: v1 (caso completo, lugar por nome) contra v2 (terreno, lugar por coordenada)."""
    out = {}
    for nome, pasta in (("recusa_por_dominio", "recusa-por-dominio"), ("politica_sob_orcamento", "politica-sob-orcamento")):
        v1 = _j(RUNS / pasta / "resultado.json")
        v2 = _j(RUNS / pasta / "v2_populacao_da_tarefa" / "resultado.json")
        if nome == "recusa_por_dominio":
            resumo = lambda r: None if not r else {  # noqa: E731
                "primario": r.get("primario"),
                "cortes": {k: {"estado": v.get("estado"), "eventos": v.get("eventos"),
                               "dominio_mediana_ordena": (v.get("seletores") or {}).get("dominio_mediana", {}).get("ordena_melhor_que_o_acaso")}
                           for k, v in r.get("cortes", {}).items()}}
        else:
            resumo = lambda r: None if not r else {  # noqa: E731
                str(x["orcamento"]): {k: x.get(k) for k in ("estado", "reservas_restritivas", "mediana_lacuna_menos_aleatoria",
                                                           "reservas_a_favor", "p_wilcoxon", "regressao_externa_mediana")}
                for x in r.get("vereditos", [])}
        out[nome] = {"v1": resumo(v1), "v2": resumo(v2), "estado": "medido" if v2 else "pendente (na fila da trava)"}
    return out


def chk_rastreio_regras() -> dict:
    import rastreio_de_impacto as ri
    e = ri.estado()
    novos = [n for n, d in ri.NOS.items() if d.get("impressoes", {}).keys() & set(ri.REGRAS)]
    return {"regras": {k: f() for k, f in ri.REGRAS.items()},
            "nos_com_regra": {n: {"regras": sorted(set(ri.NOS[n]["impressoes"]) & set(ri.REGRAS)),
                                  "estado": e[n]["estado"]} for n in novos}}


def chk_ms1() -> dict:
    r = _j(RUNS / "ms1-posicao-topografica" / "resultado.json")
    m = _j(RUNS / "posicao-topografica" / "v1" / "manifesto.json")
    if not r:
        return {"estado": "pendente (na fila da trava)", "derivacao": (m or {}).get("nan_por_coluna")}
    tab = {}
    for corte, x in r["cortes"].items():
        for cid, c in x["comparacoes"].items():
            tab[f"{corte}|{cid}"] = {"veredito": c["veredito"], "mediana": c["dentro_do_evento_mediana"],
                                     "ic": c["ic95_media_dentro"], "a_favor": f"{c['eventos_a_favor']}/{c['eventos_pareados']}"}
    auc = {corte: {mod: v["auc_mediana_dentro_do_evento"] for mod, v in x["medidas"].items()} for corte, x in r["cortes"].items()}
    return {"estado": "medido", "primario": r["primario"], "comparacoes": tab, "auc_mediana_dentro": auc,
            "perda_por_classe": r["populacao"]["perda_por_classe_ao_exigir_posicao"],
            "coef_cems_para_gfd": r["cortes"]["cems_para_gfd"]["coeficientes"],
            "derivacao": {k: (m or {}).get(k) for k in ("linhas", "unidades", "nan_por_coluna", "sem_mdt")}}


def chk_ms2() -> dict:
    r = _j(RUNS / "ms2-quadro-do-negativo" / "resultado.json")
    if not r:
        return {"estado": "pendente (na fila da trava)"}
    comp = {k: {"veredito": v["veredito"], "mediana": v["dentro_do_evento_mediana"], "ic": v["ic95_media_dentro"],
                "a_favor": f"{v['eventos_a_favor']}/{v['eventos_pareados']}"} for k, v in r["comparacoes"].items()}
    auc = {q: {m: v["auc_mediana_dentro_do_evento"] for m, v in x.items()} for q, x in r["medidas_no_gfd"].items()}
    desc = {q: {m: v["auc_mediana_dentro_do_evento"] for m, v in x.items()} for q, x in r["descritivo_gfd_para_cems"].items()}
    return {"estado": "medido", "pre_registro": r["pre_registro"], "quadros": r["linhas_por_quadro"],
            "teste_gfd": r["teste_gfd"], "comparacoes": comp, "auc_mediana_dentro_no_gfd": auc,
            "coeficientes": r["coeficientes"], "descritivo_gfd_para_cems": desc}


def chk_queda() -> dict:
    r = _j(RUNS / "queda-entre-fontes" / "resultado.json")
    if not r:
        return {"estado": "pendente"}
    dirc = {f: {v: x["auc_mediana_dentro_crescente"] for v, x in t.items() if v in ("hand_m", "twi_dinf", "slope_deg", "pos_min_2km")}
            for f, t in r["direcao_por_fonte"].items()}
    sat = r["saturacao_cems_para_gfd"]
    return {"estado": "medido", "satura": sat["satura"], "auc_expit_linear": [sat["auc_mediana_dentro_expit"],
                                                                               sat["auc_mediana_dentro_linear"]],
            "auc_dentro_por_variavel_crescente": dirc,
            "twi_eventos_acima_de_0_5": {f: t["twi_dinf"]["eventos_acima_de_0_5"] for f, t in r["direcao_por_fonte"].items()},
            "spearman_hand_twi_dentro": r["spearman_hand_twi_dentro_mediana"], "escala": r["escala_cems_para_gfd"]}


def _resumo_comparacoes(r: dict | None, chave: str = "comparacoes") -> dict | None:
    if not r:
        return None
    return {k: {"veredito": v.get("veredito"), "mediana": v.get("dentro_do_evento_mediana"),
                "ic": v.get("ic95_media_dentro"), "a_favor": f"{v.get('eventos_a_favor')}/{v.get('eventos_pareados')}"}
            for k, v in r.get(chave, {}).items()}


def chk_a16() -> dict:
    r = _j(RUNS / "auditoria-elevacao-local" / "resultado.json")
    if not r:
        return {"estado": "pendente"}
    e = r["estatica"]
    return {"estado": "medido", "le_a_classe": e["le_a_classe"], "transdutivo": e["transdutivo"],
            "teste_usa_os_proprios_pontos_rotulados": e["teste_usa_os_proprios_pontos_rotulados"],
            "consumidores": [k for k, v in e["consumidores"].items() if v["usa"]],
            "por_fonte": r["por_fonte"], "auc": r["auc"], "piso_do_mdt_disponivel": r["piso_do_mdt_disponivel"]}


def chk_posicao_v2() -> dict:
    m = _j(RUNS / "posicao-topografica" / "v2" / "manifesto.json")
    m1 = _j(RUNS / "posicao-topografica" / "v1" / "manifesto.json")
    if not m:
        return {"estado": "pendente"}
    return {"estado": "medido", "linhas": m["linhas"], "unidades": m["unidades"], "sem_mdt": m["sem_mdt"],
            "nan_por_coluna": m["nan_por_coluna"], "pos_min_negativo": m["pos_min_negativo"],
            "z_mdt_igual_a_elevation_m": m["z_mdt_igual_a_elevation_m"],
            "v1": {k: (m1 or {}).get(k) for k in ("linhas", "unidades", "nan_por_coluna")}}


def chk_populacoes_alvo() -> dict:
    import verdade_observacional as vo
    return {k: {"pergunta": v["pergunta"], "delimitacao_usa_o_desfecho": v["delimitacao_usa_o_desfecho"],
                "consultavel_antes_do_evento": v["consultavel_antes_do_evento"], "fontes": list(v["fontes"]),
                "permite": v["permite"], "nao_permite": v["nao_permite"]} for k, v in vo.POPULACOES_ALVO.items()}


def chk_ms3() -> dict:
    r = _j(RUNS / "ms3-decomposicao-do-quadro" / "resultado.json")
    if not r:
        return {"estado": "pendente"}
    return {"estado": "medido", "auc_mediana_dentro": {k: v["auc_mediana_dentro_do_evento"]
                                                       for k, v in r["medidas_no_gfd"].items()},
            "comparacoes": _resumo_comparacoes(r), "B_20_sementes_p10_p50_p90": r["B_descritivo_20_sementes"]["p10_p50_p90"],
            "coef_twi": {k: v.get("twi_dinf") for k, v in r["coeficientes"].items()}}


def chk_tc01() -> dict:
    out = {}
    for fase in ("piloto", "desenvolvimento"):
        m = _j(RUNS / "tc01-territorio-consultavel" / fase / "manifesto.json")
        out[fase] = None if not m else {"por_estado": m["por_estado"], "admissao": m["admissao"],
                                        "selecao_de_tiles": m.get("selecao_de_tiles"), "falhas": len(m.get("falhas", []))}
    return out


def chk_replica_territorio() -> dict:
    out = {}
    for v, pasta in (("rfd04", "rfd-replica-no-territorio"), ("rfd05", "rfd-replica-no-territorio-v5")):
        r = _j(RUNS / pasta / "validacao.json")
        out[v] = None if not r else {k: r.get(k) for k in ("eventos_avaliaveis", "recall_mediano", "falso_positivo_mediano",
                                                           "falso_positivo_p90", "razao_de_prevalencia_mediana", "passou")}
    return out


def chk_br03() -> dict:
    r = _j(RUNS / "br-preparo-sem-rotulo" / "resumo.json")
    if not r:
        return {"estado": "pendente"}
    t = pd.read_csv(RUNS / "br-preparo-sem-rotulo" / "eventos.csv", dtype={"eventid": str})
    c = t[t.estado == "candidato_novo"]
    visiveis = c[c.dias_visiveis > 0]
    return {"estado": "medido", "rotulos_lidos": r["rotulos_lidos"], "por_estado": r["por_estado"],
            "fonte_do_catalogo": r["fonte_do_catalogo"], "candidatos": r["candidatos"],
            "candidatos_visiveis": sorted(visiveis.eventid.tolist()),
            "candidatos_sem_treino_nas_bacias_e_vizinhas": r["candidatos_sem_treino_nas_bacias_e_vizinhas"],
            "catalogo_hash": r["catalogo_hash"]}


def chk_ms4() -> dict:
    r = _j(RUNS / "ms4-bracos-de-desenvolvimento" / "resultado.json")
    if not r:
        return {"estado": "pendente"}
    return {"estado": "medido", "linhas": r["linhas"], "eventos": r["eventos"],
            "auc_mediana_dentro": {k: v["auc_mediana_dentro_do_evento"] for k, v in r["medidas"].items()},
            "comparacoes": _resumo_comparacoes(r),
            "piso_amostral_descritivo": {k: r["descritivo_A1e_vs_A1"].get(k) for k in ("veredito", "dentro_do_evento_mediana")},
            "custo_de_ajuste_s": r["custo_de_ajuste_s"], "custo_do_encoder": r["custo_do_encoder"]}


def chk_ms5() -> dict:
    r = _j(RUNS / "ms5-recusa-decomposta" / "resultado.json")
    if not r:
        return {"estado": "pendente"}
    return {"estado": "medido", "sinais": {k: {kk: v[kk] for kk in ("estado", "eventos", "delta_mediano", "eventos_a_favor",
                                                                    "fracao_do_ganho_do_oraculo_mediana")}
                                           for k, v in r["sinais"].items()},
            "escopo": r["escopo"]}


def chk_ms6() -> dict:
    r = _j(RUNS / "ms6-quadro-de-treino-para-consulta" / "resultado.json")
    if not r:
        return {"estado": "pendente"}
    return {"estado": "medido", "linhas": r["linhas"],
            "auc_mediana_no_territorio": {k: v["auc_mediana_dentro_do_evento"] for k, v in r["medidas_no_territorio"].items()},
            "ap_no_territorio": {k: v["average_precision"] for k, v in r["medidas_no_territorio"].items()},
            "auc_mediana_na_vizinhanca": {k: v["auc_mediana_dentro_do_evento"]
                                          for k, v in r["medidas_na_vizinhanca_descritivo"].items()},
            "comparacoes": _resumo_comparacoes(r)}


def chk_trava() -> dict:
    texto = (REPO / "scripts" / "manutencao" / "executar_serializado.py").read_text(encoding="utf-8")
    diario = RUNS / ".trava-pesada.log.jsonl"
    eventos = [json.loads(x)["evento"] for x in diario.read_text(encoding="utf-8").splitlines()] if diario.exists() else []
    return {"criacao_atomica": "os.O_EXCL" in texto, "libera_so_o_dono": "nao_liberou_alheia" in texto,
            "diario": str(diario.relative_to(REPO)), "eventos_no_diario": pd.Series(eventos).value_counts().to_dict()
            if eventos else {}}


# --------------------------------------------------------------------------
# Achados, interpretacao, prontidao e proposta
# --------------------------------------------------------------------------
def ms1_estado(c: dict) -> str:
    # medido ou nao, a resposta para lugar novo e os confirmatorios ainda usam a variavel
    return "parcialmente_resolvido"


def ms1_impacto(c: dict) -> str:
    m = c["ms1"]
    if m.get("estado") != "medido":
        return "medicao pendente na fila da trava"
    t = m["comparacoes"]

    def v(k):
        x = t.get(k) or {}
        return f"`{x.get('veredito')}` {x.get('mediana')} ({x.get('a_favor')})"
    return (f"C1 piso amostral: GFD {v('gfd_dentro|C1')}, CEMS {v('cems_dentro|C1')}, CEMS->GFD {v('cems_para_gfd|C1')}. "
            f"C2 posicao sem rotulo (primario): GFD {v('gfd_dentro|C2')}, CEMS->GFD {v('cems_para_gfd|C2')}, "
            f"GFD->CEMS {v('gfd_para_cems|C2')}. C3 multiescala: GFD {v('gfd_dentro|C3')}, CEMS->GFD {v('cems_para_gfd|C3')}. "
            "Leitura: no GFD o piso nao acrescenta e a posicao sem rotulo sim; no CEMS e entre fontes o piso acrescenta, o "
            "que nao separa sinal fisico de vazamento; o primario CEMS->GFD fica dominado pelo quadro do negativo (A17) e "
            "nao informa sobre a posicao")


def achados(c: dict, sens: dict | None) -> list[dict]:
    rh, sc, tc, al = c["regime_hidrico"], c["selecao_caso_completo"], c["testes_confirmatorios"], c["alvos"]
    s1, s1t = (sens or {}).get("S1_gfd_caso_completo_do_protocolo", {}), (sens or {}).get("S1_gfd_terreno_completo", {})
    c1, c1t = (sens or {}).get("S1_cems_caso_completo_do_protocolo", {}), (sens or {}).get("S1_cems_terreno_completo", {})
    s2a, s2b = (sens or {}).get("S2_positivo_nunca_ou_raro", {}), (sens or {}).get("S2_positivo_recorrente", {})
    M1M0, M3M1 = "M1_logistica_terreno|M0_regra_hand", "M3_gbm_terreno|M1_logistica_terreno"

    def v(x, par):
        return (x.get(par) or {}).get("veredito", "nao medido")

    mk = c["dominio_gfd"]["mascara"]["por_grupo"]
    sa, ss = c["s1gfloods"], c["sar_sessao"]
    pares = c["identidade"]["pares_equivalentes"]
    cas = c["cascata"]
    tr = c["trava"]
    base = [
        {"id": "A1", "titulo": "Filtro de declive do GFD/rfd01: dominio, nao negativo", "estado": "resolvido",
         "classe": "limitacao_de_desenho",
         "problema": "o algoritmo mascara declive > 5 graus (GMTED2010) antes de decidir agua; nao era declarado e nao se sabia se virava NoData ou negativo",
         "evidencia": (f"no codigo: `updateMask(declive.lte(5))` {c['dominio_gfd']['mascara_no_codigo']}x; medido a ~240 m com o FABDEM: "
                       f"negativos do treino acima de 5 graus {mk['gfd_treino']['0']['acima_do_limiar']:.2%} contra "
                       f"{mk['rfd_controle_mascarado']['0']['acima_do_limiar']:.2%} no controle com mascara garantida (replica)"),
         "impacto": ("o corte virou NoData nas duas classes; nao ha negativo 'por construcao' no declive. Restam vieses: o negativo de "
                     "vale encaixado so vem da parte plana (atenua o contraste de declive), positivo encostado em parede ingreme conta "
                     "como interior, e o corte e no GMTED, nao no slope_deg do modelo. O efeito de declive/HAND vale dentro do dominio"),
         "correcao": "DOMINIO_DO_ROTULO com verificacao empirica e vieses restantes; chk_mascara_de_declive reprodutivel",
         "verificacao": "test_auditoria_pre_treino::test_declive_grosso_plano_e_rampa; test_dominio_declarado_bate_com_o_codigo_da_replica",
         "pendencia": "servir modelo GFD em terreno com declive GMTED > 5 graus e extrapolacao declarada"},
        {"id": "A2", "titulo": "Alvo e estratos: agua nao permanente, episodica e recorrente", "estado": "resolvido",
         "classe": "limitacao_de_desenho",
         "problema": "o alvo dos modelos nao estava nomeado; 'positivo episodico' incluia varzea recorrente",
         "evidencia": (f"positivos recorrentes (10 <= occ < 50): treino {rh['treino_gfd_ds06']['1']['agua_recorrente']:.1%}, sul-americano "
                       f"{rh['gfd-quarentena']['1']['agua_recorrente']:.1%}, Brasil {rh['rfd-quarentena-brasil']['1']['agua_recorrente']:.1%}; "
                       f"no Brasil, fora da epoca do JRC (pos-2021) {al['epoca']['teste_brasil_positivos_recorrentes_fora']:.1%} e dentro "
                       f"{al['epoca']['teste_brasil_positivos_recorrentes_dentro']:.1%}: nao e artefato do evento pesar na propria occurrence; "
                       f"a ordem se mantem nos cortes 5/10/20 {al['por_corte']['teste_brasil']}"),
         "impacto": (f"o alvo aprendido e agua nao permanente. Recorrencia e estrato de leitura, nunca exclusao nem prova de memorizacao. "
                     f"Desenvolvimento (S2): logistica x HAND `{v(s2a, M1M0)}` nos positivos nunca/raros e `{v(s2b, M1M0)}` nos "
                     f"recorrentes; GBM x logistica `{v(s2a, M3M1)}` / `{v(s2b, M3M1)}`"),
         "correcao": "verdade_observacional.ALVOS, CORTES, JRC, epoca_jrc_contem, composicao_por_corte",
         "verificacao": "test_verdade_observacional (regime, epoca, composicao por corte)",
         "pendencia": "JRC nao ve agua sob dossel: recorrencia medida e limite inferior (Amazonia)"},
        {"id": "A3", "titulo": "Populacao definida pelas variaveis da tarefa",
         "estado": ("resolvido" if all(v["estado"] == "medido" for v in c["migracoes_v2"].values())
                    and all(p["prova_no_codigo"] for p in c["populacoes"]) else "parcialmente_resolvido"),
         "classe": "defeito_corrigivel",
         "problema": "SoilGrids, campo auxiliar fora do vetor de terreno, definia a populacao do protocolo e do treino dos confirmatorios",
         "evidencia": (f"perda no GFD: positivos {sc['protocolo_congelado']['perda_por_classe']['gfd']['1']:.1%}, negativos "
                       f"{sc['protocolo_congelado']['perda_por_classe']['gfd']['0']:.1%}; CEMS: positivos "
                       f"{sc['protocolo_congelado']['perda_por_classe']['cems']['1']:.1%}, negativos {sc['protocolo_congelado']['perda_por_classe']['cems']['0']:.1%}; "
                       f"consumidores verificados no codigo: {sum(p['prova_no_codigo'] for p in c['populacoes'])}/{len(c['populacoes'])}"),
         "impacto": (f"GFD: logistica x HAND `{v(s1, M1M0)}` -> `{v(s1t, M1M0)}`, GBM x logistica `{v(s1, M3M1)}` -> `{v(s1t, M3M1)}`. "
                     f"CEMS (eventos {c1.get('eventos')} -> {c1t.get('eventos')}): GBM x logistica `{v(c1, M3M1)}` -> `{v(c1t, M3M1)}`: "
                     "o ganho exploratorio do GBM no CEMS dependia do corte. Transferencia CEMS -> GFD na v2: logistica "
                     "0,573 de AUC agrupada contra 0,738 da regra de HAND (v1: 0,786); entram 14.089 negativos do CEMS "
                     "planos e perto da agua (HAND mediano 1,46 m) e o sinal do TWI na logistica do CEMS troca (+0,24 -> "
                     "-0,22) enquanto o peso da elevacao local triplica; a causa dessa queda e o quadro do negativo (A17)"),
         "correcao": "carregar(vetor_da_populacao), checar_populacao, protocolo v2 da generalizacao; populacao explicita em todo consumidor",
         "verificacao": "test_checar_populacao_recusa_regra_assimetrica, test_v2_so_mexe_na_populacao_e_preserva_o_pre_registro_v1",
         "pendencia": ("politica e recusa v2: " + "; ".join(f"{k} {v['estado']}" for k, v in c["migracoes_v2"].items()))},
        {"id": "A4", "titulo": "Identidade canonica com evidencia; independencia espacial separada", "estado": "resolvido",
         "classe": "defeito_corrigivel",
         "problema": "regime e dobras comparavam ID so dentro da fonte; a transferencia entre instrumentos comparava NOME de grupo",
         "evidencia": (f"pares: {[(p['a'], p['b'], p['status'], p.get('km_entre_amostras_mediana')) for p in pares]}; lugar compartilhado "
                       f"(fracao das linhas de B em tile de A): gfd->cems {c['identidade']['lugar_compartilhado'].get('gfd->cems')}, "
                       f"gfd->sen1floods11 {c['identidade']['lugar_compartilhado'].get('gfd->sen1floods11')}"),
         "impacto": ("os pares GFD x Sen1Floods11 sao provavel mesmo evento com amostras a ~220-235 km: dependencia de evento, nao "
                     "pixel duplicado. Correcao da sintese anterior: sinais_marginais NAO era afetado (contexto x fonte). "
                     "Periodos distintos (GFD 2002-2018, CEMS 2023-2026) nao tornam as fontes independentes no espaco"),
         "correcao": "identidade com status/evidencia/ambiguidade; chave_espacial por coordenada; entre_fontes(espacial=True); E2c",
         "verificacao": "test_qualificar_separa_catalogo_de_coincidencia_e_mede_distancia, test_entre_fontes_espacial_retira_lugar_compartilhado",
         "pendencia": "celulas de contexto nao reajustadas (efeito de 43 linhas; ver cascata)"},
        {"id": "A5", "titulo": "Teste Brasil: janela x bacia nivel 4", "estado": "limite_registrado", "classe": "limitacao_de_desenho",
         "problema": "a regiao rotulada e a bacia nivel 4 inteira; tiles longe do evento catalogado",
         "evidencia": (f"tile-centroide: mediana {tc['rfd_distancia_tile_centroide_km']['mediana']:.0f} km; "
                       f"consulta anterior pela regra do rfd02: {tc['rfd_regra_declarada_aplicada_aos_consultados']}"),
         "impacto": ("vale como 'janelas novas de agua nao permanente em bacias e tiles ja vistos', com 75% de positivos recorrentes. "
                     "O ganho do GBM e compativel com estrutura local -- inclusive onde a agua recorre --, sem demonstrar memorizacao"),
         "correcao": "limite registrado; consultados entram no regime", "verificacao": "test_consultados_entram_como_ressalva",
         "pendencia": "proximo teste por bacia e footprint do evento, nao por raio fixo"},
        {"id": "A6", "titulo": "Teste sul-americano: leitura parcial anterior", "estado": "limite_registrado",
         "classe": "limitacao_de_desenho",
         "problema": "GFD:2782 teve 25 negativos lidos antes da selecao, sem entrar no ds05",
         "evidencia": f"{tc['gfdq_regime_com_identidade_entre_fontes_ignorando_o_proprio_exame']}",
         "impacto": "nenhuma escolha de modelo usou esses pontos; veredito mantido com ressalva",
         "correcao": "consultados", "verificacao": "test_consultados_entram_como_ressalva", "pendencia": "nenhuma"},
        {"id": "A7", "titulo": "SAR pre/pos: o ganho intercalado nao sobrevive a particao por evento", "estado": "resolvido_na_leitura",
         "classe": "defeito_corrigivel",
         "problema": "S1GFloods particionado por hash com tiles vizinhos da mesma cena nas tres particoes; alvo definido como mudanca",
         "evidencia": (f"S1GFloods: {sa.get('chips_de_teste_com_vizinho_no_treino')} de {sa.get('chips_de_teste')} chips de teste com tile "
                       f"vizinho no treino (nulo {sa.get('nulo_mesma_borda')}); Sen1Floods11 bitemporal com evento inteiro no teste "
                       f"(sessao SAR): pre_pos - pos {ss.get('pre_pos_menos_pos_por_evento')}, {ss.get('eventos_a_favor')} de 10 eventos a favor"),
         "impacto": "o +0,16 do S1GFloods e interpolacao dentro da cena; com evento reservado o pre/pos nao ajudou (estimativa negativa, indecidivel)",
         "correcao": "checagem de adjacencia (somente leitura); a sessao SAR ja particiona por evento, area e copia",
         "verificacao": "test_adjacencia_acha_tiles_vizinhos_e_respeita_o_nulo",
         "pendencia": "S1GFloods so volta a ser lido se reparticionado por mosaico; continua desenvolvimento"},
        {"id": "A8", "titulo": "FloodNet: regras corretas, contraste curto", "estado": "bloqueado_por_observacao",
         "classe": "falta_observacional",
         "problema": "janela sensor x dia UTC; 9% das laminas com chuva so no dia anterior",
         "evidencia": f"{c['floodnet'].get('defasagem_00z')}",
         "impacto": "forcamento dentro do sensor medido em NYC; terreno entre sensores sem dias; Brasil sem negativo pluvial",
         "correcao": "nenhuma no codigo", "verificacao": "checagem dos filtros",
         "pendencia": "proximo protocolo pluvial fixa a janela de chuva antes de ler"},
        {"id": "A9", "titulo": "Serra: cena depois do pico", "estado": "bloqueado_por_observacao", "classe": "falta_observacional",
         "problema": "cenas 2-6 dias apos o pico; copia antiga admitia seco pos-pico",
         "evidencia": f"admitidos {c['serra']['rotulos_admitidos']}; copia em quarentena {(c['serra'].get('copia_divergente_em_quarentena') or {}).get('sha256')}",
         "impacto": "nenhum aprendizado de serra brasileira", "correcao": "quarentena da copia; checagem le v1/v2",
         "verificacao": "test_serra_raiz_sem_manifesto_divergente", "pendencia": "observacao no pico ou campo"},
        {"id": "A10", "titulo": "Contexto fluvial: resolucao em eventos", "estado": "bloqueado_por_observacao",
         "classe": "falta_observacional", "problema": "11 eventos com posto utilizavel",
         "evidencia": f"{c['contexto_fluvial']['pares_recusados_por_motivo']}", "impacto": "nao resolve ganho < ~0,12 de AUC agrupada",
         "correcao": "nenhuma", "verificacao": "ctx04/E4", "pendencia": "~70 eventos com posto nao regularizado"},
        {"id": "A11", "titulo": "Vizinhanca e encoder", "estado": "limite_registrado", "classe": "hipotese_nao_testada",
         "problema": "rotulo de 250 m, janela de 3,84 km; ganho dentro do evento so no GFD",
         "evidencia": f"{c['vizinhanca']['n1b_diferencas']}", "impacto": "sem vantagem da representacao aprendida sobre estatistica simples",
         "correcao": "a vizinhanca simples e referencia obrigatoria", "verificacao": "N1b/V1c/V2b",
         "pendencia": "testar vizinhanca com rotulo de 30 m em evento reservado"},
        {"id": "A12", "titulo": "Consumidores e copias desatualizados", "estado": "resolvido", "classe": "defeito_corrigivel",
         "problema": "fechamento fixava ds06 v6; campanha listava falha de desenho; manifesto de serra divergente",
         "evidencia": f"{c['consumidores']}", "impacto": "documentos apontavam para dados que ninguem le",
         "correcao": "versao vigente, falha de desenho separada, quarentena", "verificacao": "test_consumidores_atualizados",
         "pendencia": "README local da pesquisadora nao tocado"},
        {"id": "A13", "titulo": "Negativo do GFD visto em qualquer dia", "estado": "bloqueado_por_observacao",
         "classe": "hipotese_nao_testada", "problema": "clear_views por ponto nao gravado",
         "evidencia": "colunas de pontos_gfd_com_terreno.csv", "impacto": "efeito no ranking entre modelos desconhecido",
         "correcao": "nenhuma sem nova leitura do Earth Engine", "verificacao": "—",
         "pendencia": "gravar clear_views/duration por ponto (cota EE)"},
        {"id": "A14", "titulo": "Trava pesada: exclusao mutua e retomada", "estado": "resolvido", "classe": "defeito_corrigivel",
         "problema": ("verificacao e escrita nao atomicas; o fim de um trabalho apagava a trava de outro (em 06/10 a campanha SAR "
                      "ficou sem trava); PID casado como substring; nenhum diario. A causa exata da sobreposicao de 23:23 nao esta "
                      "em registro nenhum (hipotese: --forcar ou corrida). Depois da primeira correcao, o diario mostrou "
                      "outro defeito: as 16:37:20 UTC de 07/10 o teste de vida pelo tasklist devolveu 'morto' para o "
                      "pid 22608, que ainda rodava, e a trava viva foi afastada como orfa"),
         "evidencia": f"{tr}",
         "impacto": "dois trabalhos pesados juntos em maquina de 15,5 GB",
         "correcao": ("O_EXCL, token por dono, liberar so o proprio, orfa afastada para historico (nunca apagada), "
                      "diario append-only, vida pelo proprio Windows (OpenProcess/GetExitCodeProcess) e morte "
                      "confirmada em duas leituras antes de afastar"),
         "verificacao": "tests/test_executar_serializado.py (8 testes, diretorio temporario; inclui falso morto)",
         "pendencia": "wrapper que ja estava rodando com codigo anterior usa o teste de vida antigo ate terminar"},
        {"id": "A15", "titulo": "Rastreio de impacto ve regra, nao so arquivo", "estado": "resolvido",
         "classe": "defeito_corrigivel",
         "problema": ("o rastreio comparava hash de arquivo: mudar o corte de agua recorrente, a populacao de uma comparacao, a "
                      "regra de identidade ou as dobras nao envelhecia nada a jusante"),
         "evidencia": f"{c['rastreio_regras']['nos_com_regra']}",
         "impacto": "derivacoes novas (v2, E2c, posicao topografica, MS1, politica e recusa v2) passam a ter carimbo e regra",
         "correcao": ("regra_de_rotulo, regra_de_populacao, regra_de_identidade, regra_de_particao como impressoes; "
                      "nos novos com ganchos exigir/carimbar"),
         "verificacao": ("test_rastreio_de_impacto::test_mudar_regra_sem_mudar_arquivo_envelhece_o_consumidor, "
                         "test_regra_de_populacao_muda_quando_o_padrao_de_carregar_muda"),
         "pendencia": "nos antigos (celulas, sinais, curva) seguem so com hash de arquivo; nenhuma regra deles mudou"},
    ]
    novos = {"a16", "ms1", "queda", "ms2", "posicao_v2", "populacoes_alvo", "tc01", "ms3",
             "replica_territorio", "br03", "ms5"}
    faltantes = sorted(novos - set(c))
    if faltantes:
        return base + [{"id": "SNAPSHOT", "titulo": "Snapshot anterior as checagens atuais",
                        "estado": "parcialmente_resolvido", "classe": "defeito_corrigivel",
                        "problema": "o codigo atual exige campos que nao foram medidos neste snapshot",
                        "evidencia": {"checagens_ausentes": faltantes},
                        "impacto": "A16-A24 nao sao reconstruidos com valores presumidos ou resultados de outra rodada",
                        "correcao": "preservar os achados disponiveis e nomear explicitamente a diferenca de versao",
                        "verificacao": "test_achados_tem_a_forma_completa",
                        "pendencia": "consultar o fechamento versionado atual; regenerar a auditoria so em janela protegida"}]
    return base + [
        {"id": "A16", "titulo": "elevation_local_m depende do rotulo amostrado", "estado": ms1_estado(c),
         "classe": "defeito_corrigivel",
         "problema": ("pela implementacao (ds06.referencia_local, gfdq02): piso = quantil 1% de elevation_m dos pontos "
                      "ADMITIDOS da unidade (tile GFD, AOI CEMS), calculado antes de separar dobras. Nao le a classe. "
                      "Tres coisas distintas: (a) vazamento direto do desfecho -- pequeno: o piso so dos negativos muda a "
                      "AUC dentro do evento em <= 0,007; (b) dependencia do desenho amostral -- a extensao amostrada e a "
                      "faixa de 2 km da mancha (GFD) ou a AOI do analista (CEMS), e o piso e transdutivo (90,6% das "
                      "unidades GFD misturam eventos de dobras diferentes); (c) incompatibilidade treino-inferencia -- "
                      "numa consulta nao ha amostra rotulada, o piso nao e calculavel do mesmo jeito; os confirmatorios "
                      "calcularam o piso sobre os pontos rotulados do proprio teste"),
         "evidencia": f"auditoria {c['a16']}; MS1 {c['ms1'].get('comparacoes')}",
         "impacto": ms1_impacto(c) + (". O piso importa ENTRE unidades (cada evento GFD cobre varios tiles). AUC "
                                      "dentro do evento, sem piso / piso do MDT da unidade (sem rotulo) / piso dos pontos: "
                                      "GFD 0,602 / 0,689 / 0,717; CEMS 0,714 / 0,722 / 0,760. Cerca de 3/4 do ganho no GFD "
                                      "e normalizacao entre lugares obtivel sem rotulo; ~0,03 (GFD) e ~0,04 (CEMS) vem da "
                                      "extensao amostrada, que e desenhada pela inundacao"),
         "correcao": ("auditoria_elevacao_local.py separa as tres causas; substituto sem rotulo calculavel na consulta "
                      "(posicao_no_ponto v2, A18); consumidores sinalizados, pesos NAO trocados"),
         "verificacao": ("tests/test_posicao_no_ponto.py; local_runs/auditoria-elevacao-local/resultado.json; "
                         "pre-registro ms1 em local_runs/ms1-posicao-topografica"),
         "pendencia": ("nenhum modelo servido usa elevation_local_m; a resposta para lugar novo e os confirmatorios usam: "
                       "trocar o vetor exige teste prospectivo, nao edicao")},
        {"id": "A17", "titulo": "Queda CEMS -> GFD: quadro do negativo, nao fisica invertida", "estado": "resolvido_na_leitura",
         "classe": "limitacao_de_desenho",
         "problema": ("a logistica de terreno treinada no CEMS cai ao acaso dentro do evento GFD (0,504) enquanto a regra "
                      "de HAND fica em 0,7335; o TWI troca de sinal entre os ajustes"),
         "evidencia": f"diagnostico {c['queda']}; MS2 {c['ms2'].get('comparacoes')} {c['ms2'].get('auc_mediana_dentro_no_gfd')}",
         "impacto": ("nao e saturacao numerica (expit = preditor linear); nenhuma variavel inverte de direcao entre fontes "
                     "(TWI alto indica cheia nas duas); o sinal negativo do TWI no CEMS e condicional ao HAND (colinear "
                     "dentro do evento); o contraste de HAND do GFD vive abaixo de 1 m, ~0,03 desvio do HAND do CEMS, que "
                     "tira o negativo da AOI inteira. Harmonizar o CEMS ao quadro de 2 km do GFD leva a mesma logistica a "
                     "0,770 no GFD (T1 pre-registrado, 69/69 eventos). VanDerWal et al. (2009): a extensao do fundo muda "
                     "pesos e desempenho"),
         "correcao": ("diagnostico_queda_entre_fontes.py e ms2_quadro_do_negativo.py (pre-registrado antes de ler o MS1); "
                      "a v2 da generalizacao continua valendo como medida, mas a causa deixa de ser lida como falta de "
                      "transferencia fisica"),
         "verificacao": ("local_runs/queda-entre-fontes/resultado.json; local_runs/ms2-quadro-do-negativo/resultado.json; "
                         "tests/test_posicao_topografica.py::test_ms2_quadro_2km_reproduz_o_desenho_do_gfd"),
         "pendencia": ("exploratorio em eventos conhecidos: nao autoriza afirmar que o CEMS transfere para o GFD; o quadro "
                       "e escolhido pela pergunta (A19), e o MS3 (A20) separa os fatores")},
        {"id": "A18", "titulo": "Substituto do piso: posicao no ponto a 30 m, a mesma funcao na consulta",
         "estado": "resolvido" if c["posicao_v2"].get("estado") == "medido" else "parcialmente_resolvido",
         "classe": "defeito_corrigivel",
         "problema": ("a posicao v1 media o ponto a 30 m contra o minimo de medias de 240 m (0,7% de pos_min_1km < 0), "
                      "aceitava janela truncada na borda do tile (numa consulta a janela e inteira) e procurava o MDT do "
                      "CEMS so por AOI (o EMSR720 tem um MDT para as cinco AOIs: 10.792 pontos perdidos)"),
         "evidencia": f"{c['posicao_v2']}",
         "impacto": ("o MS1 foi medido com a v1 e fica como medido; MS4/MS6 usam a v2. O MDT e dem_filled (WBT "
                     "fill_depressions): a consulta so reproduz o valor pela mesma cadeia sobre o mesmo MDT de base"),
         "correcao": ("posicao_no_ponto.py: suporte de 30 m no ponto e na janela, janela fora do raster -> NaN, NoData "
                      "interno aceito ate metade, `no_ponto` le so a janela e chama a mesma funcao do lote; terreno CEMS "
                      "na raiz da ativacao como cems03"),
         "verificacao": ("test_posicao_no_ponto: nunca negativo, borda, NoData, consulta == lote num GeoTIFF sintetico, "
                         "raiz da ativacao"),
         "pendencia": "o servico nao chama no_ponto ainda: ligar exige teste prospectivo do vetor (A16)"},
        {"id": "A19", "titulo": "Populacoes-alvo declaradas; territorio consultavel construido em desenvolvimento",
         "estado": "parcialmente_resolvido", "classe": "limitacao_de_desenho",
         "problema": ("vizinhanca da inundacao (GFD), AOI do analista (CEMS) e territorio consultavel respondem perguntas "
                      "diferentes; o buffer de 2 km e construido a partir da propria inundacao e nao existe numa consulta "
                      "antes do evento; nenhuma fonte do projeto amostrava o territorio"),
         "evidencia": f"populacoes {c['populacoes_alvo']}; TC01 {c['tc01']}",
         "impacto": ("nivel de AUC nao se compara entre quadros (no territorio o negativo longe e facil: -HAND 0,79-0,89 "
                     "no piloto); o quadro e escolhido pela pergunta, nunca pela AUC"),
         "correcao": ("verdade_observacional.POPULACOES_ALVO + leitura_entre_populacoes; tc01_territorio_consultavel.py: "
                      "grade de 1 km sobre o tile de 1 grau, tiles sorteados (o piloto escolhia por contagem de pontos e "
                      "isso depende do desfecho), sem visada nunca e negativo"),
         "verificacao": "test_populacoes_alvo_declaram_pergunta_e_consulta; test_tc01_estado_sem_visada_nao_e_negativo",
         "pendencia": ("TC01 e desenvolvimento (eventos conhecidos); o tile ainda e escolhido onde houve evento: nao e "
                       "probabilidade de ocorrencia; a replica precisa valer no territorio (A21)")},
        {"id": "A20", "titulo": "MS3: a extensao do negativo explica a queda entre quadros no experimento",
         "estado": "resolvido_na_leitura" if c["ms3"].get("estado") == "medido" else "limite_registrado",
         "classe": "limitacao_de_desenho",
         "problema": "o MS2 mudava extensao, margem, quantidade e distribuicao de HAND ao mesmo tempo",
         "evidencia": f"{c['ms3']}",
         "impacto": ("tirar so o negativo a mais de 2 km leva 0,503 -> 0,767 (69/69); margem de 250 m sem diferenca; "
                     "quantidade sem efeito proprio (20 sementes: mediana 0,5055); reponderar HAND explica ~1/9; o TWI "
                     "condicional negativo vem do negativo longe"),
         "correcao": "ms3_decomposicao_do_quadro.py, pre-registrado, um fator por condicao",
         "verificacao": "test_ms3_condicoes_isolam_um_fator; test_logistica_ponderada_com_pesos_iguais_e_a_comum",
         "pendencia": ("observabilidade (mapa de analista x visada MODIS) nao e separavel com os dados atuais: a extensao "
                       "e explicacao sustentada no experimento, nao causa exclusiva provada")},
        {"id": "A21", "titulo": "Replica do GFD no territorio, longe da mancha",
         "estado": ("resolvido" if (c["replica_territorio"].get("rfd05") or {}).get("passou")
                    else "bloqueado_por_observacao" if c["replica_territorio"].get("rfd05") else "parcialmente_resolvido"),
         "classe": "falta_observacional",
         "problema": ("a replica foi validada (concordancia 1,0) so nos pontos da faixa de 2 km; o territorio tem positivo "
                      "raro e negativo longe, onde o falso positivo e o recall nunca foram medidos"),
         "evidencia": f"{c['replica_territorio']}",
         "impacto": ("RFD04 (replica computada a 1 km, refletancia media) e RFD05 (projecao nativa, mudanca declarada "
                     "depois do RFD04) reprovaram no recall: com limiares operacionais a replica perde ~1/3 dos pixels "
                     "inundados do GFD longe do interior da mancha; no territorio esse positivo perdido vira negativo "
                     "observado -- erro de ROTULO no teste, nao do modelo. Sem terceira tentativa por limiar. "
                     "Pareamento por coordenada incompleto em alguns tiles (igual nas duas versoes) registrado"),
         "correcao": "rfd04_replica_no_territorio.py --versao rfd05 (pre-registro proprio, motivo declarado)",
         "verificacao": "local_runs/rfd-replica-no-territorio{,-v5}/validacao.json",
         "pendencia": ("a avaliacao brasileira de CONSULTA nao tem rotulo valido: decidir entre (1) aceitar 'agua "
                       "detectada pela replica' como alvo declarado, (2) fazer a avaliacao no quadro de vizinhanca, onde a "
                       "replica foi validada, declarada condicional ao evento, ou (3) outro sensor (SAR) com validacao "
                       "propria")},
        {"id": "A22", "titulo": "Avaliacao brasileira: novidade, espaco e visibilidade sem ler rotulo",
         "estado": "limite_registrado", "classe": "limitacao_de_desenho",
         "problema": ("o catalogo GDACS 2022-2025 e o mesmo que o rfd02 percorreu: 28 eventos ja tiveram rotulo consultado "
                      "ou lido; o GDACS de cheia e disparado pelo GloFAS (modelo), nao por observacao"),
         "evidencia": f"{c['br03']}",
         "impacto": ("sobram 24 candidatos nunca examinados, 13 com dia visivel; quase todos tem ponto de treino na "
                     "sub-bacia nivel 6 ou vizinha: 'bacia nao vista' so existe tirando essas bacias do treino"),
         "correcao": "br03_preparo_sem_rotulo.py: snapshot com hash, novidade por identidade, HydroBASINS 6, nuvem MODIS",
         "verificacao": "test_br03_novidade_examinado_vence_e_sem_poligono_nao_e_candidato",
         "pendencia": ("retrospectivo novo nao e previsao futura; >= 8 eventos avaliaveis nao e garantido (13 visiveis, "
                       "classes ainda desconhecidas); o pre-registro fica congelado sem leitura")},
        {"id": "A23", "titulo": "Recusa: elegibilidade, novidade e incerteza medidas separadamente",
         "estado": "resolvido_na_leitura" if c["ms5"].get("estado") == "medido" else "limite_registrado",
         "classe": "hipotese_nao_testada",
         "problema": "a recusa por dominio misturava escopo cientifico, distancia de atributo e incerteza, e nao ordenou o erro",
         "evidencia": f"{c['ms5']}",
         "impacto": ("so incerteza ou novidade com AURC abaixo do risco medio por evento podem recusar por erro; "
                     "elegibilidade fica como escopo; nenhum sinal e confianca calibrada"),
         "correcao": "ms5_recusa_decomposta.py, pre-registrado, sobre as predicoes fora da dobra do MS4",
         "verificacao": "test_ms5_aurc_oraculo_aleatorio_e_pior",
         "pendencia": "nada muda nas recusas de seguranca do servico; calibracao exige teste proprio"},
        {"id": "A24", "titulo": "Rastreio: carimbo de hora dentro do JSON envelhece dependentes",
         "estado": "limite_registrado", "classe": "defeito_corrigivel",
         "problema": ("o rastreio compara bytes de arquivo: identidade.json regenerado com conteudo identico (mesmo hash de "
                      "conteudo) envelheceu oito nos so pelo quando_utc"),
         "evidencia": "08/10: verdade_observacional regenerado, hash de conteudo a4ec1a55 antes e depois, parquet identico",
         "impacto": "falso envelhecimento: recarimbo manual com evidencia; nenhum envelhecimento verdadeiro escondido",
         "correcao": "nenhuma ainda: impressao normalizada de JSON (sem campos de hora) exige recarimbar todo o grafo",
         "verificacao": "historico de carimbos em RASTREIO_DE_IMPACTO",
         "pendencia": "trocar a impressao de JSON por conteudo normalizado numa janela sem cascata ativa"},
    ]


def interpretacao(c: dict) -> list[dict]:
    """Limites por conclusao. Ressalva nao invalida tudo; exploratorio nao autoriza servir."""
    cas = c["cascata"]["generalizacao"]["comparacoes_a1"]
    return [
        {"conclusao": "logistica de terreno > regra de HAND em territorio novo (sul-americano)", "regime": "confirmatorio (1 teste)",
         "recorrencia": "55% dos positivos recorrentes", "populacao": "treino caso completo, teste terreno",
         "identidade_e_consulta": "GFD:2782 com 25 negativos lidos antes", "espaco": "sem tile em comum com o treino",
         "leitura": ("mantida, com as ressalvas; nao vale para o Brasil nem para serra. A16: o braco logistico usou "
                     "elevation_local_m com piso dos pontos rotulados do teste -- a conclusao vale para essa variavel e "
                     "nao se reproduz numa consulta; a regra de HAND nao e afetada")},
        {"conclusao": "GBM > logistica em territorio conhecido (Brasil)", "regime": "confirmatorio (1 teste)",
         "recorrencia": "75% dos positivos recorrentes (79% fora da epoca do JRC)", "populacao": "treino caso completo, teste terreno",
         "identidade_e_consulta": "evento = janela x bacia nivel 4; Petropolis consultado 33 dias antes, 44 km",
         "espaco": "100% em tile do treino", "leitura": "mantida como 'estrutura local em lugar visto'; nao autoriza servir GBM"},
        {"conclusao": "logistica x HAND no Brasil conhecido", "regime": "confirmatorio (1 teste)", "recorrencia": "idem",
         "populacao": "idem", "identidade_e_consulta": "idem", "espaco": "idem", "leitura": "indecidivel, como medido"},
        {"conclusao": "GBM > logistica no CEMS (exploratorio)", "regime": "exploratorio",
         "recorrencia": "nao medida no CEMS", "populacao": f"caso completo; v2 {cas.get('cems|M3_gbm_terreno_vs_M1_logistica_terreno')}",
         "identidade_e_consulta": "CEMS sem par no GFD", "espaco": "lugar compartilhado com o GFD em 22% das linhas (outra epoca)",
         "leitura": "nao sustentada fora do corte de SoilGrids: deixa de ser citada como ganho do GBM no CEMS"},
        {"conclusao": "logistica > HAND e GBM > logistica no GFD (exploratorio)", "regime": "exploratorio",
         "recorrencia": "ganho nos dois estratos (S2)", "populacao": f"v2 {cas.get('gfd|M1_logistica_terreno_vs_M0_regra_hand')}",
         "identidade_e_consulta": "sem duplicata dentro do GFD", "espaco": "tile do teste fora do treino",
         "leitura": "direcao mantida; nivel de AUC muda com a populacao e nao se compara entre v1 e v2"},
        {"conclusao": "transferencia GFD <-> CEMS", "regime": "exploratorio", "recorrencia": "nao medida",
         "populacao": "v2: terreno", "identidade_e_consulta": "sem evento em comum",
         "espaco": f"v2 retira lugar por coordenada: {c['cascata']['generalizacao']['entre_fontes_v2']}",
         "leitura": ("v1 dizia que a logistica transfere melhor entre instrumentos; na populacao de terreno, com lugar "
                     "retirado por coordenada, a logistica treinada no CEMS cai abaixo da regra de HAND no GFD: a "
                     "afirmacao segue nao sustentada. MS2 (A17) mostra que a queda vem do quadro do negativo: com o CEMS "
                     "no quadro de 2 km a mesma logistica volta a 0,770 no GFD. Isso explica a queda; nao reabilita a "
                     "afirmacao de transferencia, que exigiria teste em evento nao visto")},
        {"conclusao": "recusa por dominio ordena o erro", "regime": "exploratorio",
         "recorrencia": "—", "populacao": f"v1 caso completo; v2 terreno ({c['migracoes_v2']['recusa_por_dominio']['estado']})",
         "identidade_e_consulta": "—", "espaco": "v2 retira lugar por coordenada",
         "leitura": ("nao sustentada nas duas versoes: no GFD o seletor de dominio fica pior que o acaso; entre fontes o "
                     "controle por tamanho ordena tanto quanto ele")},
        {"conclusao": "pre/pos SAR ajuda a segmentar", "regime": "desenvolvimento",
         "recorrencia": "JRC tira permanente, nao sazonal", "populacao": "—",
         "identidade_e_consulta": "rotulos Sen1Floods11 ja examinados", "espaco": "S1GFloods intercalado; Sen1Floods11 por evento",
         "leitura": "nao sustentada: o ganho some com evento reservado"},
        {"conclusao": "politica por lacuna", "regime": "exploratorio", "recorrencia": "—",
         "populacao": f"v1 caso completo; v2 terreno ({c['migracoes_v2']['politica_sob_orcamento']['estado']})",
         "identidade_e_consulta": "—", "espaco": "v2 retira lugar do externo por coordenada",
         "leitura": "indecidivel em 4 e 8, not_testable em 16, nas duas versoes: sem vantagem consistente"},
        {"conclusao": "para consultar territorio, treinar no territorio e melhor que na vizinhanca", "regime": "exploratorio (MS6)",
         "recorrencia": "nao estratificada", "populacao": "TC01 desenvolvimento x ds06 GFD", "identidade_e_consulta": "os mesmos 69 eventos, dobras iguais",
         "espaco": "tiles repetem entre eventos (desenvolvimento)",
         "leitura": f"{(c['ms6'].get('comparacoes') or {})}"},
        {"conclusao": "encoder de terreno acrescenta alem da vizinhanca simples", "regime": "exploratorio (MS4 K5)",
         "recorrencia": "nao estratificada", "populacao": "amostra N1 (GFD)", "identidade_e_consulta": "eventos conhecidos",
         "espaco": "dobras por evento", "leitura": f"{((c['ms4'].get('comparacoes') or {}).get('K5'))}"},
        {"conclusao": "posicao topografica sem rotulo acrescenta ao terreno do ponto", "regime": "exploratorio (MS1)",
         "recorrencia": "nao estratificada", "populacao": "terreno + posicao finita",
         "identidade_e_consulta": "eventos conhecidos do ds06", "espaco": "entre fontes sem tile comum",
         "leitura": ("sustentada no GFD (C2 ganho) e GFD->CEMS; indecidivel no CEMS; varias escalas nao acrescentam a uma "
                     "no GFD e acrescentam no CEMS. Autoriza so propor o vetor para teste prospectivo")},
    ]


def prontidao(c: dict) -> list[dict]:
    return [
        {"tarefa": "ordenar suscetibilidade fluvial de planicie (agua nao permanente)", "estado": "pronto_com_ressalva",
         "pode_alimentar": "ds06 GFD/CEMS planicie na populacao de terreno, agrupado por evento canonico e tile",
         "contaminaria": "caso completo em SoilGrids; dobras por nome de grupo entre fontes; ler recorrencia como erro"},
        {"tarefa": "transferencia para territorio novo", "estado": "pronto_com_ressalva",
         "pode_alimentar": "treino como acima; teste so com bacia e tile fora do treino",
         "contaminaria": "eventos conhecidos, examinados ou consultados; tile do treino; raio fixo como garantia"},
        {"tarefa": "transferencia dentro do Brasil para lugar nao visto", "estado": "bloqueado_por_observacao",
         "pode_alimentar": "271 tiles de 1 grau no Brasil sem treino nem teste rfd (precisam de evento observado)",
         "contaminaria": "reusar os 16 eventos do rfd03"},
        {"tarefa": "nao linearidade (GBM)", "estado": "pronto_com_ressalva",
         "pode_alimentar": "so como braco secundario declarado", "contaminaria": "citar o ganho do CEMS v1 como evidencia"},
        {"tarefa": "probabilidade territorial", "estado": "bloqueado_por_observacao",
         "pode_alimentar": "eventos com area por classe gravada (V4)", "contaminaria": "fracao positiva da amostra como area"},
        {"tarefa": "serra brasileira", "estado": "bloqueado_por_observacao", "pode_alimentar": "nada hoje",
         "contaminaria": "seco na data da cena como negativo; GFD de serra"},
        {"tarefa": "pluvial urbano no Brasil", "estado": "bloqueado_por_observacao", "pode_alimentar": "nada hoje",
         "contaminaria": "ausencia de chamado como negativo; NYC como validacao do Brasil"},
        {"tarefa": "forcamento pluvial dentro do sensor (NYC)", "estado": "pronto_com_ressalva",
         "pode_alimentar": "janelas lamina/sem_lamina do fn02", "contaminaria": "nao_observado/desconhecido como seco; janela 00Z sem defasagem declarada"},
        {"tarefa": "contexto fluvial", "estado": "bloqueado_por_observacao", "pode_alimentar": "ctx04 como diagnostico",
         "contaminaria": "variavel nova sobre 11 eventos"},
        {"tarefa": "segmentacao de agua SAR (Sen1Floods11)", "estado": "pronto_com_ressalva",
         "pode_alimentar": "particao por evento da sessao SAR (desenvolvimento)", "contaminaria": "chamar de teste virgem"},
        {"tarefa": "mudanca pre/pos SAR", "estado": "parcialmente_resolvido",
         "pode_alimentar": "Sen1Floods11 bitemporal por evento (desenvolvimento)", "contaminaria": "S1GFloods com particao por hash"},
        {"tarefa": "representacao aprendida de terreno", "estado": "pronto_com_ressalva",
         "pode_alimentar": "janelas do N1 com vizinhanca simples como referencia", "contaminaria": "comparar so com o terreno do ponto"},
        {"tarefa": "politica de aquisicao", "estado": "medido_sem_vantagem",
         "pode_alimentar": "simulacao retrospectiva v2 (populacao de terreno)",
         "contaminaria": "custo de rede simulado como real; ler indecidivel como ganho"},
        {"tarefa": "recusa por dominio", "estado": "nao_sustentado",
         "pode_alimentar": "nada como seletor; controle por tamanho como referencia", "contaminaria": "servir recusa por dominio"},
        {"tarefa": "posicao topografica sem rotulo no vetor", "estado": "pronto_com_ressalva",
         "pode_alimentar": "posicao_no_ponto v2 (30 m, mesma funcao da consulta) como candidato num teste prospectivo",
         "contaminaria": "elevation_local_m (piso dos pontos rotulados) em lugar novo ou teste; a v1 de 240 m"},
        {"tarefa": "consulta de territorio delimitado antes do evento", "estado":
            ("pronto_com_ressalva" if (c["replica_territorio"].get("rfd05") or {}).get("passou") else "bloqueado_por_observacao"),
         "pode_alimentar": "TC01 desenvolvimento (GFD conhecido) para treino e desenvolvimento",
         "contaminaria": "buffer da propria inundacao como requisito da consulta; AUC do territorio comparada a da vizinhanca"},
        {"tarefa": "recusa por erro (incerteza ou novidade)", "estado":
            ("pronto_com_ressalva" if any(v["estado"] == "ordena" for v in (c["ms5"].get("sinais") or {}).values())
             else "medido_sem_vantagem" if c["ms5"].get("estado") == "medido" else "parcialmente_resolvido"),
         "pode_alimentar": "o sinal que ordena, como recusa declarada de desenvolvimento",
         "contaminaria": "distancia de dominio apresentada como confianca calibrada; remover recusa de seguranca"},
        {"tarefa": "transferencia entre instrumentos (CEMS <-> GFD)", "estado": "decisao_de_alvo",
         "pode_alimentar": "os dois quadros, declarados", "contaminaria": "treinar num quadro e ler no outro sem dizer"},
    ]


PROPOSTA = {
    "estado": "proposta para decisao da pesquisadora; NAO congelada como pre-registro",
    "pergunta": "a logistica de terreno ordena agua nao permanente melhor que HAND em territorio brasileiro que o treino nao viu?",
    "selecao_por_mecanismo": ("cheia fluvial de planicie; fora: enxurrada de serra, rompimento de barragem, evento costeiro; "
                              "o mecanismo vem do catalogo e da forma do footprint, nao do resultado"),
    "selecao_por_bacia": ("sub-bacias HydroBASINS nivel 6 que tocam a area afetada do GDACS (modelo, nao a mancha "
                          "observada): delimitacao independente do desfecho; o TREINO perde os pontos dessas sub-bacias e "
                          "das vizinhas, porque quase todo candidato tem treino nelas (br03)"),
    "candidatos": "br03: eventos candidato_novo com >= 1 dia visivel, lista congelada antes da leitura",
    "populacao_alvo": "territorio_consultavel (A19); rotulo = replica, so se o portao do RFD05 passar (A21)",
    "selecao_por_periodo": "2022-2025 (fora do GFD e da epoca do JRC: occurrence nao inclui o evento)",
    "suporte_espacial": ("rotulo MODIS 250 m da replica; terreno 30 m agregado ao mesmo suporte para a leitura; tile e "
                         "sub-bacia fora do treino; 100 km NAO e criterio"),
    "alvo": "agua nao permanente (occurrence < 50), lida tambem nos estratos nunca/rara e recorrente, cortes ja fixados",
    "populacao": "carregar(TERRENO) no treino; teste completo no mesmo vetor (checar_populacao)",
    "vetor": ("M1 = hand_m, twi_dinf, slope_deg, SEM elevation_local_m (piso dos pontos rotulados, A16); candidato "
              "secundario base + pos_min_2km (MS1); nada derivado do rotulo do teste"),
    "quadro_do_negativo": ("o da replica MODIS (positivo a > 250 m do negativo; negativo na faixa de 2 km), o mesmo do "
                           "treino GFD; a leitura vale para essa pergunta, nao para a paisagem inteira (A17)"),
    "comparacoes": {"primaria": "M1 logistica x M0 HAND",
                    "secundarias": ["base + pos_min_2km x M1", "vizinhanca simples (estatisticas da janela) x M1",
                                    "M3 GBM x M1", "candidato aprendido x vizinhanca"],
                    "condicoes_iguais": "mesmas linhas, mesmas janelas, mesmo treino, mesma semente"},
    "efeito_relevante": "0,01 de AUC dentro do evento (o do protocolo congelado)",
    "avaliacao": "uma leitura; Wilcoxon pareado por evento; >= 8 eventos com as duas classes; IC por bootstrap de eventos",
    "orcamento": "tarefas em lote do Earth Engine; derivacao de terreno local dos tiles novos; ajuste em minutos",
    "separacao": {"desenvolvimento": "ds06 conhecido (v2)", "regressao": "reavaliar contextos conhecidos; queda > 0,01 barra",
                  "confirmacao": "so os eventos novos, uma vez, depois registrados como examinados"},
    "aquisicao_que_destrava": ("feito sem rotulo: snapshot GDACS, HydroBASINS 6, nuvem MODIS (br03); falta: replica "
                               "validada no territorio (RFD05) e a leitura unica dos rotulos dos candidatos"),
}


def markdown(r: dict) -> str:
    c = r["checagens"]
    L = ["# Auditoria pré-treino: o que cada dataset observa e o que permite aprender", "",
         f"**Gerado de** `scripts/metodo/auditoria_pre_treino.py` em {r['quando_utc']}. **Não editar à mão.**", "",
         "**Não é.** validação científica: checagem verde diz que a cadeia faz o que declara; não fecha lacuna "
         "observacional. Nenhum teste confirmatório foi rerodado; nos eventos de teste só a composição do rótulo foi lida.", "",
         "## Estado em uma linha por achado", "", "| achado | estado | classe |", "|---|---|---|"]
    L += [f"| {a['id']} {a['titulo']} | `{a['estado']}` | `{a['classe']}` |" for a in r["achados"]]
    L += ["", "## 1. Cadeia real dos dados", "", r["linhagem"]["cadeia"], "", r["linhagem"]["separacao"], "",
          "| dataset | papel | estado | sha256 | consumidores | aprende |", "|---|---|---|---|---|---|"]
    for x in r["linhagem"]["datasets"]:
        L.append(f"| `{x['caminho']}` | {x['papel']} | {x['estado']} | `{(x.get('sha256') or '—')[:23]}…` | "
                 f"{len(x['consumidores'])} | {x['aprende']} |")
    L += ["", "Versões do ds06: " + "; ".join(f"`{v['versao']}` {v['estado']}" for v in r["linhagem"]["versoes_ds06"]) + ".",
          "Hashes completos em `local_runs/auditoria-pre-treino/resultado.json`.", "",
          "## 2. Alvos por tarefa", ""]
    L += [f"- **`{k}`:** {v['definicao']}. Uso: {v['uso']}." for k, v in c["alvos"]["alvos"].items()]
    L += ["", "Cortes: " + "; ".join(f"{k}: {v}" for k, v in c["alvos"]["cortes"].items()) + ".",
          f"JRC: {c['alvos']['jrc']['produto']}, época {c['alvos']['jrc']['epoca']}, denominador {c['alvos']['jrc']['denominador']}; "
          f"{c['alvos']['jrc']['limites']}.", "",
          "| conjunto | fração com occurrence ≥ corte (< 50), negativos / positivos |", "|---|---|"]
    for k, x in c["alvos"]["por_corte"].items():
        L.append(f"| {k} | " + "; ".join(f"≥{cut}: {v['0']:.1%} / {v['1']:.1%}" for cut, v in x.items()) + " |")
    L += ["", f"Época: {c['alvos']['epoca']}.", "", "## 3. Achados: problema → evidência → impacto → correção → verificação → pendência", ""]
    for a in r["achados"]:
        L += [f"### {a['id']}. {a['titulo']} (`{a['estado']}`, `{a['classe']}`)", "",
              f"- **Problema:** {a['problema']}", f"- **Evidência:** {a['evidencia']}",
              f"- **Impacto no aprendizado:** {a['impacto']}", f"- **Correção:** {a['correcao']}",
              f"- **Verificação:** {a['verificacao']}", f"- **Pendência:** {a['pendencia']}", ""]
    L += ["## 4. População por consumidor", "", "| consumidor | população | estado | prova no código |", "|---|---|---|---|"]
    L += [f"| {p['consumidor']} | {p['populacao']} | {p['estado']} | {'sim' if p['prova_no_codigo'] else '**NÃO**'} |"
          for p in c["populacoes"]]
    L += ["", "Perda do caso completo por fonte × território (fração tirada de cada classe):", "",
          "| fonte · território | linhas (terreno) | negativos | positivos |", "|---|---:|---:|---:|"]
    L += [f"| {k} | {v['linhas']} | {v['0']} | {v['1']} |" for k, v in c["perda_por_contexto"].items()]
    cas = c["cascata"]
    L += ["", "## 5. Cascata de desenvolvimento", "",
          f"**E2c** (dobras por evento canônico; {cas['e2']['estado']}): diferença B−A original {cas['e2']['original']} → "
          f"E2c {cas['e2']['e2c']}; eventos unidos {cas['e2']['eventos_unidos']}; coeficiente condicional do HAND em A "
          f"{cas['e2']['relacao_A_condicional']}.", "",
          f"**Generalização v2** (população de terreno; {cas['generalizacao']['estado']}). Resultado da v1 preservado; "
          "níveis de AUC não se comparam entre v1 e v2.", "",
          "| fonte · comparação | v1 caso completo | v2 terreno |", "|---|---|---|"]
    for k, x in cas["generalizacao"]["comparacoes_a1"].items():
        def f(y):
            return "—" if not y else f"`{y['veredito']}` {y['mediana']} {y['ic']} ({y['a_favor']})"
        L.append(f"| {k} (eventos {x['eventos_v1']} → {x['eventos_v2']}) | {f(x['v1_caso_completo'])} | {f(x['v2_populacao_de_terreno'])} |")
    L += ["", f"Entre fontes na v2 (lugar retirado por coordenada): {cas['generalizacao']['entre_fontes_v2']}.",
          f"Território reservado na v2 (M1 × M0): {cas['generalizacao']['territorio_reservado_v2']}.", "",
          "Não rerodados, com o motivo:", ""] + [f"- **{k}:** {v}" for k, v in cas["nao_rerodados"].items()]
    L += ["", "## 6. Interpretação dos testes existentes, por conclusão", "",
          "| conclusão | regime | recorrência | população | identidade e consulta | espaço | leitura |", "|---|---|---|---|---|---|---|"]
    L += [f"| {x['conclusao']} | {x['regime']} | {x['recorrencia']} | {x['populacao']} | {x['identidade_e_consulta']} | "
          f"{x['espaco']} | {x['leitura']} |" for x in r["interpretacao"]]
    L += ["", "## 7. SAR: coordenação com a sessão ativa", "",
          f"Partição da sessão SAR: {c['sar_sessao'].get('particao')}. Resultado por evento: {c['sar_sessao'].get('pre_pos_menos_pos_por_evento')}.",
          "", "Para confirmar a mudança pré/pós de forma independente faltam: eventos fora do Sen1Floods11 e do S1GFloods "
          "(2020 em diante), par S1 da mesma órbita com data da cena anterior identificada (não mediana anual), rótulo "
          "feito sem ver o modelo e com classe desconhecida, água permanente e sazonal do ano anterior separadas, partição "
          "por evento e mosaico, e o critério de 0,025 de IoU fixado antes. Os dados já examinados continuam desenvolvimento "
          "mesmo reparticionados.", "",
          "## 7b. Migrações v2, MS1, MS2 e queda entre fontes", "",
          f"Recusa por domínio e política sob orçamento, v1 × v2: {c['migracoes_v2']}.", "",
          f"MS1 (posição topográfica sem rótulo): {c['ms1']}.", "",
          f"MS2 (quadro do negativo): {c['ms2']}.", "",
          f"Diagnóstico da queda CEMS → GFD: {c['queda']}.", "",
          "## 7c. A16, populações-alvo, território consultável, Brasil e recusa", "",
          f"A16 pela implementação: {c['a16']}.", "",
          f"Posição no ponto v2: {c['posicao_v2']}.", "",
          f"MS3 (decomposição do quadro): {c['ms3']}.", "",
          f"TC01 (território consultável): {c['tc01']}.", "",
          f"Réplica no território (RFD04/RFD05): {c['replica_territorio']}.", "",
          f"Preparo brasileiro sem rótulo (BR03): {c['br03']}.", "",
          f"MS4 (braços de desenvolvimento): {c['ms4']}.", "",
          f"MS5 (recusa decomposta): {c['ms5']}.", "",
          f"MS6 (quadro de treino para consulta): {c['ms6']}.", "",
          "## 8. Trava de trabalho pesado", "", f"{c['trava']}", "",
          "## 9. Prontidão por tarefa", "", "| tarefa | estado | pode alimentar aprendizado | ainda contaminaria |", "|---|---|---|---|"]
    L += [f"| {p['tarefa']} | `{p['estado']}` | {p['pode_alimentar']} | {p['contaminaria']} |" for p in r["prontidao"]]
    L += ["", "## 10. Próximo experimento geoespacial (proposta, não congelada)", "",
          f"`{r['proposta_hash']}` em `local_runs/auditoria-pre-treino/proposta_proximo_experimento.json`.", ""]
    L += [f"- **{k.replace('_', ' ')}:** {v}" for k, v in r["proposta"].items()]
    s = r.get("sensibilidades")
    if s:
        L += ["", "## 11. Sensibilidades de desenvolvimento (exploratório, eventos conhecidos)", "",
              f"Regime `{s['regime']}`; semente {s['semente']}; {s['estimadores']}.", "",
              "| recorte | tamanho | logística × HAND | GBM × logística |", "|---|---:|---|---|"]
        for k in ("S1_gfd_caso_completo_do_protocolo", "S1_gfd_terreno_completo", "S1_cems_caso_completo_do_protocolo",
                  "S1_cems_terreno_completo", "S2_positivo_nunca_ou_raro", "S2_positivo_recorrente"):
            x = s.get(k)
            if not x:
                continue
            n = f"{x['linhas']} linhas, {x['eventos']} eventos" if "linhas" in x else f"{x['positivos']} positivos"
            cel = [f"`{x[p]['veredito']}` mediana {x[p]['dentro_do_evento_mediana']}, IC {x[p]['ic95_media_dentro']}, "
                   f"{x[p]['eventos_a_favor']}/{x[p]['eventos_pareados']}"
                   for p in ("M1_logistica_terreno|M0_regra_hand", "M3_gbm_terreno|M1_logistica_terreno")]
            L.append(f"| {k} | {n} | " + " | ".join(cel) + " |")
    L += ["", f"`{r['hash_resumo']}`", ""]
    return "\n".join(L)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refazer-sensibilidades", action="store_true")
    a = ap.parse_args()
    from proveniencia import agora_utc, sha256_json
    SAIDA.mkdir(parents=True, exist_ok=True)
    cache_p = SAIDA / "hashes_cache.json"
    cache = _j(cache_p) or {}
    d, versao = _ds06()
    ident = chk_identidade()
    ident["lugar_compartilhado"] = (_j(RUNS / "verdade-observacional" / "identidade.json") or {}).get(
        "lugar_compartilhado_entre_fontes", {})
    c = {"dominio_gfd": {**chk_dominio_gfd(d), "mascara": chk_mascara_de_declive()}, "regime_hidrico": chk_regime_hidrico(d),
         "alvos": chk_alvos(d), "selecao_caso_completo": chk_selecao_caso_completo(),
         "perda_por_contexto": chk_perda_por_contexto(), "populacoes": chk_populacoes(), "identidade": ident,
         "testes_confirmatorios": chk_testes_confirmatorios(), "s1gfloods": chk_s1gfloods(cache),
         "sar_sessao": chk_sar_sessao(), "floodnet": {**chk_floodnet(), "defasagem_00z": chk_floodnet_defasagem()},
         "serra": chk_serra(), "contexto_fluvial": chk_contexto_fluvial(), "vizinhanca": chk_vizinhanca(),
         "admissao": chk_admissao(d), "consumidores": chk_consumidores(), "cascata": chk_cascata(), "trava": chk_trava(),
         "migracoes_v2": chk_migracoes_v2(), "rastreio_regras": chk_rastreio_regras(), "ms1": chk_ms1(),
         "ms2": chk_ms2(), "queda": chk_queda(), "a16": chk_a16(), "posicao_v2": chk_posicao_v2(),
         "populacoes_alvo": chk_populacoes_alvo(), "ms3": chk_ms3(), "tc01": chk_tc01(),
         "replica_territorio": chk_replica_territorio(), "br03": chk_br03(), "ms4": chk_ms4(), "ms5": chk_ms5(),
         "ms6": chk_ms6()}
    del d
    sp = SAIDA / "sensibilidades.json"
    sens = _j(sp)
    if a.refazer_sensibilidades or not sens or sens.get("dataset") != versao:
        sens = sensibilidades()
        sens["dataset"] = versao
        sens["quando_utc"] = agora_utc()
        sp.write_text(json.dumps(sens, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    proposta_hash = sha256_json(PROPOSTA)
    (SAIDA / "proposta_proximo_experimento.json").write_text(
        json.dumps({"hash": proposta_hash, "proposta": PROPOSTA}, ensure_ascii=False, indent=1), encoding="utf-8")
    r = {"tipo": "auditoria_pre_treino", "dataset": versao, "linhagem": linhagem(cache), "checagens": c,
         "achados": achados(c, sens), "interpretacao": interpretacao(c), "prontidao": prontidao(c),
         "proposta": PROPOSTA, "proposta_hash": proposta_hash, "sensibilidades": sens}
    cache_p.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    r["hash_resumo"] = sha256_json(json.loads(json.dumps({k: v for k, v in r.items() if k != "sensibilidades"},
                                                         default=str)))
    r["quando_utc"] = agora_utc()
    (SAIDA / "resultado.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    DOC.write_text(markdown(r), encoding="utf-8")
    print(f"GRAVADO={DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
