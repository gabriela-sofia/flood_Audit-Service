"""Acervo espacial: o que o disco ja tem, onde, por qual cadeia, e para que pode ser usado.

-------------------------------------------------------------------------
A LACUNA
-------------------------------------------------------------------------

O terreno do projeto esta em `ter-01` (1.650 diretorios, ~120 GiB), mas cada
consumidor o encontrava do seu jeito: o lote pelo NOME da unidade do evento
(`posicao_no_ponto.pasta_da_unidade`), o servico pela caixa do tile mais central
(`resolver_camadas.CoberturaPorTiles`, so dentro do Brasil e sem cadeia nem
MDT), o encoder pelo prefixo do diretorio. Nenhum respondia "o que cobre esta
coordenada, com que cadeia, e o que falta". Medido em 09/10/2026: o pacote
congelado perdeu 136.949 pontos observados por "vetor nao finito" -- 85.725 so
porque a janela de 2 km de `pos_min_2km` saia do raster da unidade do evento,
embora os tiles GFD tenham margem de 0,1 grau e se sobreponham.

-------------------------------------------------------------------------
O QUE ESTE MODULO FAZ
-------------------------------------------------------------------------

  indice      tabelas persistentes em `local_runs/acervo-espacial/`: unidades de
              terreno (extensao, nucleo, CRS, resolucao, cadeia, MDT de entrada,
              licenca), camadas (dtype, NoData, impressao, ligacao fisica),
              conjuntos de pontos e pegadas de observacao (eventos, reservas,
              candidatos). Incremental: so reabre o que mudou de tamanho/data.
  consulta    `Acervo.consultar_ponto` / `consultar_aoi`: cobertura LIDA no
              raster (nao so caixa), compatibilidade de cadeia e de MDT,
              alternativas com seus valores, lacunas separadas em falta de
              atributo, de observacao e de independencia.
  contexto    `descrever` e a UNICA funcao que transforma janelas de raster em
              atributos. O lote (`contextualizar`) e a consulta passam por ela;
              so muda quem le a janela (memoria ou disco).

-------------------------------------------------------------------------
REGRAS QUE NAO SAO OPCIONAIS
-------------------------------------------------------------------------

  unidade     cada ponto le TODOS os atributos de UMA unidade: nao se costura
              HAND de dois tiles -- HAND e TWI dependem da drenagem derivada
              sobre a extensao inteira do raster (Renno et al. 2008, Remote Sens.
              Environ. 112: 3469-3481; Nobre et al. 2011, J. Hydrol. 404: 13-29),
              e dois tiles derivados separadamente tem redes diferentes perto da
              borda. O "mosaico" e a escolha por ponto da unidade compativel em
              que ele esta mais longe da borda.
  ordem       compativel > mais longe da borda > nome; vence a primeira com
              HAND, TWI, declive, elevacao e pos_min_2km finitos; se nenhuma,
              a primeira, e o que falta fica desconhecido com o motivo.
  compativel  cadeia canonica declarada (30 m, 0,1123 km2, TWI ln(SCA/tan))
              E MDT da familia de referencia (FABDEM V1-2, agregacao por media).
              A outra familia (MDT agregado por bilinear, origem nao declarada no
              run_manifest) e indexada e consultavel, nunca misturada.
  borda       janela que sai do raster e desconhecida, nao truncada (a mesma
              regra de `posicao_no_ponto`).
  ausencia    NoData, fora de toda unidade e janela incompleta ficam NaN com
              motivo. Nada e copiado de um vizinho; ausencia de observacao nao
              e negativo.
  reservas    pegadas de reservas e candidatos vem so de metadados ja gravados
              pelos protocolos; o indice nunca abre os pacotes nem le classe.

nao_e: nao e aquisicao (nao baixa nada), nao deriva terreno, nao decide rotulo
e nao muda o escore servido.

Uso:
    python scripts/geo/acervo_espacial.py --indexar
    python scripts/geo/acervo_espacial.py --verificar
    python scripts/geo/acervo_espacial.py --ponto -51.07 -12.07
    python scripts/geo/acervo_espacial.py --aoi -51.2 -12.2 -51.0 -12.0 --passo-m 1000
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
for _pasta in ("servico", "suscetibilidade", "metodo", "terreno"):
    sys.path.insert(0, str(REPO / "scripts" / _pasta))

import posicao_no_ponto as pp  # noqa: E402

RUNS = REPO / "local_runs"
TER01 = RUNS / "ter-01-cadeia-harmonizada"
MDT_FABDEM = RUNS / "mdt-fabdem"
SAIDA = RUNS / "acervo-espacial"
VERSAO = "v1"
ENCODER = RUNS / "terreno-encoder" / "encoder_amostra_chip_s7_manifesto.json"

# Cadeia de referencia: a do pacote congelado (ter-01, WhiteboxTools 2.4.0).
RESOLUCAO_M = 30.0
LIMIAR_CANAL_KM2 = 0.1123
FORMULA_TWI = "ln(SCA / tan(slope))"
REAMOSTRAGEM_REFERENCIA = "average"

# arquivo -> (atributo, estado epistemico, unidade de medida, o que e)
CAMADAS = {
    "dem_filled.tif": ("elevation_m", "medido", "m",
                       "MDT de entrada agregado a 30 m, depressoes preenchidas (WBT); FABDEM ja e um MDT corrigido por modelo"),
    "slope_deg_wbt.tif": ("slope_deg", "derivado", "graus", "declive WBT sobre dem_filled"),
    "hand_dinf.tif": ("hand_m", "derivado", "m", "altura acima da drenagem D-inf (limiar 0,1123 km2) do proprio raster"),
    "twi_dinf.tif": ("twi_dinf", "derivado", "1", "ln(SCA/tan(declive)), D-inf"),
    "dinf_sca.tif": ("dinf_sca", "derivado", "m", "area especifica de contribuicao D-inf"),
    "dinf_accum_cells.tif": ("dinf_accum_cells", "derivado", "celulas", "acumulacao D-inf em celulas"),
    "streams_dinf.tif": ("streams_dinf", "derivado", "1", "rede de drenagem extraida pelo limiar"),
    "agua_permanente_jrc.tif": ("agua_permanente_jrc", "medido_externo", "1", "JRC GSW v1.4 occurrence >= 50"),
}
ARQUIVO_DO_ATRIBUTO = {v[0]: k for k, v in CAMADAS.items()}
ATRIBUTOS_DO_PIXEL = ("elevation_m", "hand_m", "twi_dinf", "slope_deg")
NUCLEO = ("hand_m", "twi_dinf", "slope_deg", "elevation_m", "pos_min_2km")
CAMADAS_OBRIGATORIAS = tuple(ARQUIVO_DO_ATRIBUTO[a] for a in ATRIBUTOS_DO_PIXEL)

# Vizinhanca simples: a definicao do N1b/V1/MS4 (janela de 128 px das cinco camadas
# do encoder, log1p nas acumulacoes, normalizada pelo manifesto do encoder,
# invalido -> 0, media/desvio/p10/p50/p90 por camada). Diferenca declarada: aqui a
# janela que sai do raster e desconhecida (no N1 bastava metade valida), e nao ha o
# arredondamento float16 do cache do N1.
CAMADAS_VIZINHANCA = ("dem_filled", "slope_deg_wbt", "hand_dinf", "dinf_sca", "dinf_accum_cells")
CAMADAS_EM_LOG = ("dinf_sca", "dinf_accum_cells")
SEM_DADO = -32768.0
LADO_VIZINHANCA = 128
ESTATISTICAS = ("media", "desvio", "p10", "p50", "p90")
COLUNAS_VIZINHANCA = tuple(f"viz_{c}_{e}" for e in ESTATISTICAS for c in CAMADAS_VIZINHANCA)

LIDAS = (*CAMADAS_OBRIGATORIAS, "dinf_sca.tif", "dinf_accum_cells.tif")
K_POSICAO = pp.raio_em_celulas(max(pp.RAIOS_M), RESOLUCAO_M)
K_LEITURA = max(K_POSICAO, LADO_VIZINHANCA // 2)
COLUNAS_CONTEXTO = (*ATRIBUTOS_DO_PIXEL, *pp.COLUNAS, *COLUNAS_VIZINHANCA)

FABDEM = {"produto": "FABDEM V1-2", "doi": "10.5523/bris.s5hqmjcdj8yo2ibzi9b4ew3sn",
          "artigo": "Hawker et al. 2022, Environ. Res. Lett., doi:10.1088/1748-9326/ac4d4f",
          "licenca": ("CC BY-NC-SA 4.0 segundo license.txt e a descricao do deposito (conferido em 09/10/2026); o campo "
                      "de metadados do repositorio indica Non-Commercial Government Licence; uso nao comercial"),
          "data_do_relevo": "nao registrada no acervo: FABDEM e o Copernicus GLO-30 com floresta e edificacao removidas"}
LICENCAS = {
    "gfd": "Global Flood Database v1 (Tellman et al. 2021, Nature 596: 80-86), CC BY 4.0 -- declarada em tc01",
    "cems": "https://mapping.emergency.copernicus.eu/terms-and-conditions/ -- declarada em obs03",
    "gdacs_glofas": "catalogo GDACS; licenca do produto nao declarada localmente",
    "jrc": "JRC Global Surface Water v1.4 (Pekel et al. 2016); licenca nao declarada no manifesto local",
}


# ---------------------------------------------------------------------------
# Impressao de arquivo
# ---------------------------------------------------------------------------

def impressao(p: Path, completo: bool = False) -> dict:
    """Tamanho, data, identidade fisica e um hash.

    `sha256` completo so quando pedido ou para arquivo pequeno (<= 8 MiB); acima
    disso o hash e PARCIAL (primeiro e ultimo MiB + tamanho) e vem rotulado como
    tal -- detecta troca de arquivo, nao corrupcao no meio. `id_arquivo` repete
    entre ligacoes fisicas (hardlink): mesmo dado, nao copia.
    """
    st = p.stat()
    pequeno = st.st_size <= 8 << 20
    h = hashlib.sha256()
    with p.open("rb") as f:
        if completo or pequeno:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
            tipo = "sha256"
        else:
            h.update(f.read(1 << 20))
            f.seek(-(1 << 20), 2)
            h.update(f.read(1 << 20))
            h.update(str(st.st_size).encode())
            tipo = "sha256_parcial"
    return {"bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns), "id_arquivo": f"{st.st_dev}:{st.st_ino}",
            "ligacoes": int(st.st_nlink), "hash": f"{tipo}:{h.hexdigest()}"}


def _sha_json(obj) -> str:
    texto = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _agora() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(p).as_posix()


# ---------------------------------------------------------------------------
# Unidades de terreno
# ---------------------------------------------------------------------------

def tipo_da_unidade(nome: str) -> str:
    if nome.startswith("gfd_tile_"):
        return "tile_gfd"
    if nome.startswith("n1chip_"):
        return "chip_sen1floods11"
    if re.fullmatch(r"EMS[RN]\d+_AOI\d+", nome):
        return "aoi_cems"
    if re.fullmatch(r"EMS[RN]\d+", nome):
        return "ativacao_cems"
    if nome.endswith("_harmonizado"):
        return "regiao_servida"
    return "outro"


def nucleo_do_tile(nome: str) -> list | None:
    """Tile GFD de 1 grau pelo nome: `gfd_tile_S12W051` -> [-51, -12, -50, -11] (o FABDEM tem 0,1 grau de margem)."""
    m = re.fullmatch(r"gfd_tile_([NS])(\d{2})([EW])(\d{3})", nome)
    if not m:
        return None
    lat = int(m[2]) * (1 if m[1] == "N" else -1)
    lon = int(m[4]) * (1 if m[3] == "E" else -1)
    return [float(lon), float(lat), float(lon + 1), float(lat + 1)]


def procedencia_do_mdt(pasta: Path, manifesto: dict) -> dict:
    """De onde veio o MDT de entrada, pelo que os manifestos dizem. Nao infere origem que nao foi gravada."""
    dtm = str(manifesto.get("dtm_input") or "")
    amostragem = manifesto.get("resampling_dem")
    nome = Path(dtm.replace("\\", "/")).name
    if "fabdem" in nome.lower():
        plano = MDT_FABDEM / nome.replace("_4326.tif", "_manifesto.json")
        dados = {}
        if plano.exists():
            try:
                dados = json.loads(plano.read_text(encoding="utf-8"))
            except ValueError:
                dados = {}
        urls = [t.get("url", "") for t in dados.get("tiles", [])]
        origem = ("data.bris.ac.uk" if urls and all("data.bris.ac.uk" in u for u in urls)
                  else ("misto_ou_outro" if urls else "plano_ausente"))
        return {"familia_mdt": f"fabdem_{amostragem}", "mdt_produto": dados.get("produto", "fabdem"),
                "mdt_tipo": dados.get("tipo_dem"), "mdt_origem": origem,
                "mdt_plano": _rel(plano) if plano.exists() else None,
                "mdt_plano_hash": dados.get("hash_resumo"), "mdt_licenca": FABDEM["licenca"],
                "mdt_doi": FABDEM["doi"], "mdt_bbox_pedida": dados.get("bbox_4326")}
    return {"familia_mdt": f"outro_{amostragem}", "mdt_produto": "nao_declarado_no_run_manifest",
            "mdt_tipo": None, "mdt_origem": nome or "nao_declarada", "mdt_plano": None, "mdt_plano_hash": None,
            "mdt_licenca": "nao_declarada_localmente", "mdt_doi": None, "mdt_bbox_pedida": None}


def cadeia(manifesto: dict | None) -> tuple[bool, str]:
    """A cadeia declarada serve ao contrato? Mesmo criterio de `dados_terreno.cadeia_declarada`, mais a agregacao."""
    if manifesto is None:
        return False, "sem run_manifest: cadeia desconhecida"
    desvios = []
    px = manifesto.get("pixel_size") or []
    if not (len(px) == 2 and all(abs(float(p) - RESOLUCAO_M) < 1e-6 for p in px)):
        desvios.append(f"resolucao {px}")
    limiar = manifesto.get("stream_area_km2")
    if limiar is None:
        desvios.append("limiar de canal por percentil, nao por area")
    elif abs(float(limiar) - LIMIAR_CANAL_KM2) > 1e-9:
        desvios.append(f"limiar de canal {limiar} km2")
    if manifesto.get("twi_formula") != FORMULA_TWI:
        desvios.append(f"formula do TWI {manifesto.get('twi_formula')!r}")
    return (not desvios), ("; ".join(desvios) or "cadeia canonica declarada")


def _cabecalho(arq: Path) -> dict:
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(arq) as r:
        t = r.transform
        bbox = transform_bounds(r.crs, "EPSG:4326", *r.bounds, densify_pts=21)
        return {"crs": r.crs.to_string() if r.crs else None, "epsg": r.crs.to_epsg() if r.crs else None,
                "largura": int(r.width), "altura": int(r.height), "res_m": float(abs(t.a)),
                "transform": [float(t.a), float(t.b), float(t.c), float(t.d), float(t.e), float(t.f)],
                "bbox": [float(x) for x in bbox], "dtype": r.dtypes[0],
                "nodata": None if r.nodata is None else float(r.nodata),
                "compressao": (r.compression.value if r.compression else None)}


def _ler_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None


def indexar_terreno(raiz: Path = TER01, anterior: pd.DataFrame | None = None,
                    camadas_anteriores: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Unidades e camadas de `raiz`. So reabre raster cujo tamanho ou data mudou desde o indice anterior."""
    antes = {}
    if camadas_anteriores is not None and len(camadas_anteriores):
        for r in camadas_anteriores.to_dict("records"):
            antes[(r["unidade"], r["arquivo"])] = r
    unid_antes = {r["unidade"]: r for r in anterior.to_dict("records")} if anterior is not None and len(anterior) else {}
    unidades, camadas = [], []
    diario = {"reaproveitadas": 0, "reabertas": 0, "novas": 0, "removidas": 0}
    vistos = set()
    for d in sorted(p for p in raiz.iterdir() if p.is_dir()) if raiz.exists() else []:
        man_arq = d / "run_manifest.json"
        man = _ler_json(man_arq)
        presentes = [a for a in CAMADAS if (d / a).exists()]
        if not presentes:
            continue
        vistos.add(d.name)
        linhas = []
        mudou = d.name not in unid_antes
        for a in presentes:
            st = (d / a).stat()
            velho = antes.get((d.name, a))
            if velho is not None and velho["bytes"] == st.st_size and velho["mtime_ns"] == st.st_mtime_ns:
                linhas.append(velho)
                diario["reaproveitadas"] += 1
                continue
            mudou = True
            diario["reabertas" if velho is not None else "novas"] += 1
            try:
                cab = _cabecalho(d / a)
                erro = None
            except Exception as e:  # noqa: BLE001 -- raster ilegivel fica registrado, nao derruba o indice
                cab, erro = {}, f"{type(e).__name__}: {e}"[:200]
            atributo, estado, um, oque = CAMADAS[a]
            linhas.append({"unidade": d.name, "arquivo": a, "atributo": atributo, "estado_epistemico": estado,
                           "unidade_de_medida": um, "o_que_e": oque, "erro_de_leitura": erro,
                           **{k: cab.get(k) for k in ("crs", "largura", "altura", "res_m", "dtype", "nodata",
                                                      "compressao")},
                           "transform": json.dumps(cab.get("transform")), "bbox": json.dumps(cab.get("bbox")),
                           **impressao(d / a)})
        camadas.extend(linhas)
        if not mudou and d.name in unid_antes and unid_antes[d.name].get("run_manifest_hash") == (
                impressao(man_arq)["hash"] if man_arq.exists() else None):
            unidades.append(unid_antes[d.name])
            continue
        ref = next((r for r in linhas if r["arquivo"] == "dem_filled.tif" and not r["erro_de_leitura"]),
                   next((r for r in linhas if not r["erro_de_leitura"]), None))
        ok, motivo = cadeia(man)
        proc = procedencia_do_mdt(d, man or {})
        geom = {}
        if ref is not None:
            geom = {"crs": ref["crs"], "res_m": ref["res_m"], "largura": ref["largura"], "altura": ref["altura"],
                    "transform": ref["transform"], "bbox": ref["bbox"]}
        divergentes = sorted({r["arquivo"] for r in linhas if ref is not None and not r["erro_de_leitura"]
                              and (r["transform"] != ref["transform"] or r["crs"] != ref["crs"]
                                   or r["largura"] != ref["largura"])})
        familia_ok = proc["familia_mdt"] == f"fabdem_{REAMOSTRAGEM_REFERENCIA}"
        faltam = [a for a in CAMADAS_OBRIGATORIAS if a not in presentes]
        incompat = []
        if not ok:
            incompat.append(f"cadeia: {motivo}")
        if not familia_ok:
            incompat.append(f"mdt da familia {proc['familia_mdt']}, referencia fabdem_{REAMOSTRAGEM_REFERENCIA}")
        if divergentes:
            incompat.append(f"camadas em grade diferente da dem_filled: {divergentes}")
        if faltam:
            incompat.append(f"faltam camadas {faltam}")
        unidades.append({
            "unidade": d.name, "tipo": tipo_da_unidade(d.name), "caminho": _rel(d), **geom,
            "nucleo": json.dumps(nucleo_do_tile(d.name)),
            "camadas": ",".join(presentes), "cadeia_no_contrato": ok, "cadeia_motivo": motivo,
            "stream_area_km2": (man or {}).get("stream_area_km2"), "twi_formula": (man or {}).get("twi_formula"),
            "reamostragem_mdt": (man or {}).get("resampling_dem"), "wbt": (man or {}).get("wbt_version"),
            "derivado_em": (man or {}).get("quando_utc"), "enxugado": bool((man or {}).get("enxugado")),
            **proc, "run_manifest_hash": impressao(man_arq)["hash"] if man_arq.exists() else None,
            "compativel": not incompat, "motivo_incompatibilidade": "; ".join(incompat) or None,
            "impressao_unidade": _sha_json(sorted((r["arquivo"], r["hash"], r["bytes"]) for r in linhas)),
        })
    diario["removidas"] = len(set(unid_antes) - vistos)
    return pd.DataFrame(unidades), pd.DataFrame(camadas), diario


# ---------------------------------------------------------------------------
# Conjuntos de pontos e pegadas de observacao
# ---------------------------------------------------------------------------

def _ultimo(padrao: str) -> Path | None:
    achados = sorted(RUNS.glob(padrao), key=lambda p: p.stat().st_mtime)
    return achados[-1] if achados else None


def registro_de_conjuntos() -> list[dict]:
    """Conjuntos tabulares que alimentam ou cercam o aprendizado, com papel e uso autorizado.

    `ler_coordenadas` falso: o indice nao abre o arquivo alem do hash (reserva ou
    pacote bruto). Classe nunca e lida aqui.
    """
    pacote = _ultimo("pacote-de-treino/*/manifesto.json")
    curado = _ultimo("tc01-curado/v1/*/pontos.parquet")
    itens = [
        {"id": "ds05_tabela_unica_v2", "caminho": RUNS / "ds-05-tabela-unica/tabela_unica_v2.parquet",
         "papel": "canonico", "uso_autorizado": "fonte de ds06 e admissao", "instrumento": "varias fontes",
         "mecanismo": "varios", "populacao": "varias", "ler_coordenadas": True},
        *[{"id": f"ds06_{v.name}", "caminho": v / "pontos.parquet",
           "papel": "canonico" if v.name == "v7" else "historico",
           "uso_autorizado": "pesos tabulares (vigente)" if v.name == "v7" else "linhagem; nao treinar",
           "instrumento": "varias fontes", "mecanismo": "varios", "populacao": "vizinhanca/AOI por fonte",
           "ler_coordenadas": v.name == "v7"}
          for v in sorted((RUNS / "ds-06-dataset-de-treino").glob("v*")) if (v / "pontos.parquet").exists()],
        {"id": "tc01_desenvolvimento", "caminho": RUNS / "tc01-territorio-consultavel/desenvolvimento/pontos.parquet",
         "papel": "derivado", "uso_autorizado": "origem do tc01 curado; repeticoes de coordenada/evento",
         "instrumento": "MODIS (GFD v1)", "mecanismo": "fluvial de planicie", "populacao": "territorio_consultavel",
         "ler_coordenadas": True},
        *([{"id": "tc01_curado_v1", "caminho": curado, "papel": "canonico",
            "uso_autorizado": "desenvolvimento: base do pacote congelado", "instrumento": "MODIS (GFD v1)",
            "mecanismo": "fluvial de planicie", "populacao": "territorio_consultavel", "ler_coordenadas": True}]
          if curado else []),
        *([{"id": "pacote_fisico_congelado", "caminho": pacote.parent / "fisico.parquet", "papel": "congelado",
            "uso_autorizado": "treino de desenvolvimento pelo protocolo do pacote; nao alterar",
            "instrumento": "MODIS (GFD v1)", "mecanismo": "fluvial de planicie", "populacao": "territorio_consultavel",
            "ler_coordenadas": True}] if pacote else []),
        {"id": "gfd_quarentena", "caminho": RUNS / "gfd-quarentena/pontos_quarentena.parquet",
         "papel": "teste_confirmatorio_examinado", "uso_autorizado": "nenhum ajuste; ja lido uma vez (gfdq03)",
         "instrumento": "MODIS (GFD v1)", "mecanismo": "fluvial", "populacao": "vizinhanca_da_inundacao",
         "ler_coordenadas": True},
        {"id": "rfd_quarentena_brasil", "caminho": RUNS / "rfd-quarentena-brasil/pontos_quarentena.parquet",
         "papel": "teste_confirmatorio_examinado", "uso_autorizado": "nenhum ajuste; ja lido uma vez (rfd03)",
         "instrumento": "MODIS (replica do algoritmo GFD)", "mecanismo": "fluvial", "populacao": "janela x bacia",
         "ler_coordenadas": True},
        {"id": "cache_janelas_n1", "caminho": RUNS / "campanha-integrada-20261003/N1/janelas_f16.npy",
         "papel": "cache", "uso_autorizado": "reproduzir N1/V1/MS4; refazivel a partir do ter-01",
         "instrumento": "terreno", "mecanismo": "-", "populacao": "amostra N1", "ler_coordenadas": False},
    ]
    for regiao, z in (("sa", RUNS / "hydrobasins/v1c/bruto/hybas_sa_lev01-12_v1c.zip"),
                      ("eu", RUNS / "hydrobasins/v1c/bruto_eu/hybas_eu_lev01-12_v1c.zip")):
        itens.append({"id": f"hydrobasins_v1c_{regiao}", "caminho": z, "papel": "referencia_externa",
                      "uso_autorizado": ("unidade hidrologica e conectividade; sem redistribuicao avulsa; atribuicao "
                                         "Exhibit B (HydroSHEDS v1 License Agreement)"),
                      "instrumento": "HydroSHEDS 15 arc-seg (SRTM)", "mecanismo": "-", "populacao": "-",
                      "ler_coordenadas": False})
    dom = RUNS / "dominio-aquatico/tc01/dominio_por_ponto.parquet"
    itens.append({"id": "dominio_aquatico_tc01", "caminho": dom, "papel": "derivado",
                  "uso_autorizado": "contexto e sensibilidade; WBM nao exclui; nao reescreve estado observacional",
                  "instrumento": "Copernicus DEM WBM (2011-2015) + JRC GSW v1.4 (1984-2021)", "mecanismo": "-",
                  "populacao": "territorio_consultavel (TC01)", "ler_coordenadas": False})
    for v2 in sorted(RUNS.glob("contexto-desenvolvimento/v2/*/extensao.parquet")):
        itens.append({"id": f"contexto_v2_{v2.parent.name}", "caminho": v2, "papel": "derivado",
                      "uso_autorizado": "desenvolvimento; v1 + extensao por ponto_id (bacia, dominio, borda)",
                      "instrumento": "MODIS (GFD v1) + contexto", "mecanismo": "fluvial de planicie",
                      "populacao": "territorio_consultavel", "ler_coordenadas": False})
        pa = v2.parent / "populacao_alvo_v2.parquet"
        if pa.exists():
            itens.append({"id": f"populacao_alvo_v2_{v2.parent.name}", "caminho": pa, "papel": "candidato",
                          "uso_autorizado": ("populacao candidata coerente com o contrato do alvo (occurrence JRC v1.4 "
                                             "< 50 no ponto); nao congelada; pacote antigo preservado"),
                          "instrumento": "MODIS (GFD v1)", "mecanismo": "fluvial de planicie",
                          "populacao": "territorio_consultavel, alvo_v2", "ler_coordenadas": False})
    cat = RUNS / "candidatos-independentes/catalogo.json"
    itens.append({"id": "catalogo_candidatos_independentes", "caminho": cat, "papel": "catalogo",
                  "uso_autorizado": "planejar aquisicao/confirmacao; so metadados, nenhum rotulo aberto",
                  "instrumento": "varios", "mecanismo": "a confirmar", "populacao": "-", "ler_coordenadas": False})
    for p in sorted(RUNS.glob("reservas-observacionais/*/aquisicoes/*/manifesto.json")):
        itens.append({"id": f"reserva_cems_{p.parent.name[:12]}", "caminho": p, "papel": "reserva",
                      "uso_autorizado": "nenhum ate protocolo independente; rotulo nao lido", "instrumento": "CEMS (S1)",
                      "mecanismo": "a confirmar", "populacao": "aoi_do_analista", "ler_coordenadas": False})
    for p in sorted(RUNS.glob("observacoes-dirigidas/*/aquisicoes/*/manifesto.json")):
        itens.append({"id": f"candidato_gfm_{p.parent.name[:12]}", "caminho": p, "papel": "candidato",
                      "uso_autorizado": "auditoria de observabilidade; 0 admitidos", "instrumento": "GFM (S1) / OPERA",
                      "mecanismo": "a confirmar", "populacao": "-", "ler_coordenadas": False})
    return itens


def indexar_conjuntos(anteriores: pd.DataFrame | None = None) -> pd.DataFrame:
    import pyarrow.parquet as pq
    antes = {r["id"]: r for r in anteriores.to_dict("records")} if anteriores is not None and len(anteriores) else {}
    linhas = []
    for item in registro_de_conjuntos():
        p = Path(item["caminho"])
        base = {k: v for k, v in item.items() if k != "caminho"}
        base["caminho"] = _rel(p)
        if not p.exists():
            linhas.append({**base, "existe": False})
            continue
        imp = impressao(p, completo=p.suffix != ".npy")
        velho = antes.get(item["id"])
        if velho is not None and velho.get("hash") == imp["hash"]:
            linhas.append({**velho, **base})
            continue
        extra = {"linhas": None, "eventos": None, "bbox": None, "colunas": None}
        if p.suffix == ".parquet":
            pf = pq.ParquetFile(p)
            nomes = pf.schema_arrow.names
            extra["linhas"] = int(pf.metadata.num_rows)
            extra["colunas"] = ",".join(nomes)
            if item["ler_coordenadas"]:
                lon_c = "lon" if "lon" in nomes else ("longitude" if "longitude" in nomes else None)
                lat_c = "lat" if "lat" in nomes else ("latitude" if "latitude" in nomes else None)
                ev_c = "evento_id" if "evento_id" in nomes else None
                cols = [c for c in (lon_c, lat_c, ev_c) if c]
                if lon_c and lat_c:
                    t = pd.read_parquet(p, columns=cols)
                    extra["bbox"] = json.dumps([float(t[lon_c].min()), float(t[lat_c].min()),
                                                float(t[lon_c].max()), float(t[lat_c].max())])
                    if ev_c:
                        extra["eventos"] = int(t[ev_c].nunique())
        linhas.append({**base, "existe": True, "rotulo_lido_pelo_indice": False, **imp, **extra})
    return pd.DataFrame(linhas)


def indexar_pegadas(unidades: pd.DataFrame) -> pd.DataFrame:
    """Onde ha observacao de evento e com que papel. Caixa de triagem, nao identidade nem mancha."""
    linhas = []
    ds06 = RUNS / "ds-06-dataset-de-treino" / "v7" / "pontos.parquet"
    caixas = {r["unidade"]: json.loads(r["bbox"]) for r in unidades.to_dict("records") if r.get("bbox")}
    if ds06.exists():
        cols = ["fonte", "evento_id", "grupo_validacao", "mecanismo", "janela_inicio", "janela_fim",
                "resolucao_nativa_m", "procedencia"]
        d = pd.read_parquet(ds06, columns=cols)
        d = d[d.fonte.isin(["gfd", "cems"])].drop_duplicates(["evento_id", "grupo_validacao"])
        for (fonte, ev), g in d.groupby(["fonte", "evento_id"]):
            bx = []
            for grupo in g.grupo_validacao.unique():
                nome = ("gfd_tile_" + grupo.split("_", 1)[1]) if fonte == "gfd" else grupo
                if fonte == "gfd" and nucleo_do_tile(nome):
                    bx.append(nucleo_do_tile(nome))
                elif nome in caixas:
                    bx.append(caixas[nome])
                elif nome.split("_AOI", 1)[0] in caixas:
                    bx.append(caixas[nome.split("_AOI", 1)[0]])
            if not bx:
                continue
            a = np.array(bx)
            r0 = g.iloc[0]
            linhas.append({"evento_id": ev, "fonte": fonte, "papel": "desenvolvimento_conhecido",
                           "instrumento": "MODIS 250 m (GFD v1)" if fonte == "gfd" else "CEMS rapid mapping",
                           "mecanismo": r0.mecanismo, "janela_inicio": str(r0.janela_inicio),
                           "janela_fim": str(r0.janela_fim), "resolucao_m": float(r0.resolucao_nativa_m),
                           "caixa": json.dumps([float(a[:, 0].min()), float(a[:, 1].min()),
                                                float(a[:, 2].max()), float(a[:, 3].max())]),
                           "partes": json.dumps(bx), "origem_da_caixa": "tiles/AOIs do ds06 v7",
                           "rotulo_acessivel": True, "licenca": LICENCAS.get(fonte)})
    for prot in sorted(RUNS.glob("reservas-observacionais/*/protocolo.json")):
        m = _ler_json(prot) or {}
        for a in m.get("ativacoes", []):
            at = a.get("ativacao", a)
            ext = at.get("extent") or ""
            nums = [float(x) for x in re.findall(r"-?\d+\.\d+", ext)]
            if len(nums) < 4:
                continue
            lons, lats = nums[0::2], nums[1::2]
            linhas.append({"evento_id": at.get("code"), "fonte": "cems", "papel": "reserva",
                           "instrumento": "CEMS (Sentinel-1 segundo o catalogo)", "mecanismo": "a confirmar",
                           "janela_inicio": str(at.get("evento_data")), "janela_fim": None, "resolucao_m": None,
                           "caixa": json.dumps([min(lons), min(lats), max(lons), max(lats)]),
                           "partes": None, "origem_da_caixa": f"extent dos metadados da ativacao ({_rel(prot)})",
                           "rotulo_acessivel": False, "licenca": m.get("licenca")})
    br = RUNS / "br-preparo-sem-rotulo" / "eventos.csv"
    if br.exists():
        e = pd.read_csv(br, dtype={"eventid": str})
        for r in e.itertuples(index=False):
            if any(pd.isna(getattr(r, c)) for c in ("lon_min", "lat_min", "lon_max", "lat_max")):
                continue
            papel = "candidato" if r.estado == "candidato_novo" else "teste_examinado"
            linhas.append({"evento_id": f"GDACS:{r.eventid}", "fonte": "gdacs_glofas", "papel": papel,
                           "instrumento": f"catalogo {r.fonte_catalogo} (modelo; nao e observacao)",
                           "mecanismo": "a confirmar", "janela_inicio": str(r.inicio), "janela_fim": str(r.fim),
                           "resolucao_m": None,
                           "caixa": json.dumps([float(r.lon_min), float(r.lat_min), float(r.lon_max), float(r.lat_max)]),
                           "partes": None, "origem_da_caixa": "caixa do poligono GDACS (br03)",
                           "rotulo_acessivel": False, "licenca": LICENCAS["gdacs_glofas"]})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Indice: gravar, carregar, verificar
# ---------------------------------------------------------------------------

def indexar(saida: Path = SAIDA, raiz: Path = TER01, conjuntos: bool = True) -> dict:
    saida.mkdir(parents=True, exist_ok=True)
    inicio = time.time()
    u0 = pd.read_parquet(saida / "unidades.parquet") if (saida / "unidades.parquet").exists() else None
    c0 = pd.read_parquet(saida / "camadas.parquet") if (saida / "camadas.parquet").exists() else None
    k0 = pd.read_parquet(saida / "conjuntos.parquet") if (saida / "conjuntos.parquet").exists() else None
    unidades, camadas, diario = indexar_terreno(raiz, u0, c0)
    tabelas = {"unidades": unidades, "camadas": camadas}
    if conjuntos:
        tabelas["conjuntos"] = indexar_conjuntos(k0)
        tabelas["pegadas"] = indexar_pegadas(unidades)
    hashes = {}
    for nome, t in tabelas.items():
        tmp = saida / f"{nome}.parquet.part"
        t.to_parquet(tmp, index=False)
        tmp.replace(saida / f"{nome}.parquet")
        hashes[nome] = impressao(saida / f"{nome}.parquet", completo=True)["hash"]
    fisicos = camadas.groupby("id_arquivo").size() if len(camadas) else pd.Series(dtype=int)
    man = {"tipo": "acervo_espacial", "versao": VERSAO, "quando_utc": _agora(),
           "codigo_sha256": impressao(Path(__file__), completo=True)["hash"], "raiz_terreno": _rel(raiz),
           "tabelas": hashes, "diario": diario,
           "unidades": int(len(unidades)), "compativeis": int(unidades.compativel.sum()) if len(unidades) else 0,
           "por_tipo": unidades.tipo.value_counts().to_dict() if len(unidades) else {},
           "por_familia_mdt": unidades.familia_mdt.value_counts().to_dict() if len(unidades) else {},
           "camadas": int(len(camadas)),
           "camadas_ligadas_a_outro_caminho": int((camadas.ligacoes > 1).sum()) if len(camadas) else 0,
           "mesmo_arquivo_fisico_dentro_do_indice": int((fisicos > 1).sum()),
           "erros_de_leitura": int(camadas.erro_de_leitura.notna().sum()) if len(camadas) else 0,
           "referencia": {"resolucao_m": RESOLUCAO_M, "limiar_canal_km2": LIMIAR_CANAL_KM2, "twi": FORMULA_TWI,
                          "familia_mdt": f"fabdem_{REAMOSTRAGEM_REFERENCIA}", "fabdem": FABDEM},
           "segundos": round(time.time() - inicio, 1),
           "nao_e": "nao baixa, nao deriva, nao le classe; hash parcial em arquivo > 8 MiB e declarado como tal"}
    if conjuntos:
        man["conjuntos"] = int(len(tabelas["conjuntos"]))
        man["pegadas_por_papel"] = tabelas["pegadas"].papel.value_counts().to_dict() if len(tabelas["pegadas"]) else {}
    man["hash_resumo"] = _sha_json({k: v for k, v in man.items() if k not in ("quando_utc", "segundos", "diario")})
    (saida / "manifesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    with (saida / "diario.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"quando_utc": man["quando_utc"], **diario, "hash_resumo": man["hash_resumo"]}) + "\n")
    return man


def verificar(saida: Path = SAIDA) -> dict:
    """Tabelas batem com o manifesto, e os arquivos indexados continuam com o mesmo tamanho e data."""
    man = json.loads((saida / "manifesto.json").read_text(encoding="utf-8"))
    falhas = [n for n, h in man["tabelas"].items()
              if impressao(saida / f"{n}.parquet", completo=True)["hash"] != h]
    c = pd.read_parquet(saida / "camadas.parquet")
    raiz = REPO / man["raiz_terreno"]
    mudaram, sumiram = [], []
    for r in c.itertuples(index=False):
        p = raiz / r.unidade / r.arquivo
        if not p.exists():
            sumiram.append(f"{r.unidade}/{r.arquivo}")
            continue
        st = p.stat()
        if st.st_size != r.bytes or st.st_mtime_ns != r.mtime_ns:
            mudaram.append(f"{r.unidade}/{r.arquivo}")
    novas = sorted({d.name for d in raiz.iterdir() if d.is_dir() and any((d / a).exists() for a in CAMADAS)}
                   - set(c.unidade))
    return {"tabelas_integras": not falhas, "tabelas_divergentes": falhas, "camadas_conferidas": int(len(c)),
            "mudaram": mudaram[:20], "n_mudaram": len(mudaram), "sumiram": sumiram[:20], "n_sumiram": len(sumiram),
            "unidades_nao_indexadas": novas[:20], "n_unidades_nao_indexadas": len(novas),
            "atual": not (falhas or mudaram or sumiram or novas)}


# ---------------------------------------------------------------------------
# Geometria: onde o ponto cai em cada unidade, sem ler pixel
# ---------------------------------------------------------------------------

def geometria(unidades: pd.DataFrame, lon: np.ndarray, lat: np.ndarray) -> pd.DataFrame:
    """Para cada (ponto, unidade cuja caixa o contem): linha/coluna, distancia a borda e ao nucleo.

    Vetorizado por unidade. Devolve indice do ponto, unidade, lin, col, distancia
    a borda do raster em celulas e metros, e se o ponto esta no nucleo do tile.
    """
    from pyproj import Transformer
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    ordem_lon = np.argsort(lon, kind="mergesort")
    lon_ord = lon[ordem_lon]
    partes = []
    for u in unidades.itertuples(index=False):
        if not u.bbox:
            continue
        b = json.loads(u.bbox)
        a0 = int(np.searchsorted(lon_ord, b[0], side="left"))
        a1 = int(np.searchsorted(lon_ord, b[2], side="right"))
        faixa = ordem_lon[a0:a1]
        dentro = np.sort(faixa[(lat[faixa] >= b[1]) & (lat[faixa] <= b[3])])
        if not len(dentro):
            continue
        t = json.loads(u.transform)
        x, y = Transformer.from_crs("EPSG:4326", u.crs, always_xy=True).transform(lon[dentro], lat[dentro])
        colf = (np.asarray(x) - t[2]) / t[0]
        linf = (np.asarray(y) - t[5]) / t[4]
        col, lin = np.floor(colf).astype(np.int64), np.floor(linf).astype(np.int64)
        no_raster = (lin >= 0) & (lin < u.altura) & (col >= 0) & (col < u.largura)
        if not no_raster.any():
            continue
        dentro, lin, col = dentro[no_raster], lin[no_raster], col[no_raster]
        borda = np.minimum.reduce([lin, col, u.altura - 1 - lin, u.largura - 1 - col])
        nuc = json.loads(u.nucleo) if u.nucleo and u.nucleo != "null" else None
        if nuc:
            lo, la = lon[dentro], lat[dentro]
            no_nucleo = (lo >= nuc[0]) & (lo < nuc[2]) & (la >= nuc[1]) & (la < nuc[3])
        else:
            no_nucleo = np.ones(len(dentro), bool)
        partes.append(pd.DataFrame({"i": dentro, "unidade": u.unidade, "lin": lin, "col": col,
                                    "borda_celulas": borda, "borda_m": borda * float(u.res_m),
                                    "no_nucleo": no_nucleo, "compativel": bool(u.compativel)}))
    if not partes:
        return pd.DataFrame(columns=["i", "unidade", "lin", "col", "borda_celulas", "borda_m", "no_nucleo",
                                     "compativel", "ordem"])
    g = pd.concat(partes, ignore_index=True)
    # A ordem que lote e consulta seguem: compativel, mais longe da borda, nome.
    g = g.sort_values(["i", "compativel", "borda_celulas", "unidade"], ascending=[True, False, False, True],
                      kind="mergesort")
    g["ordem"] = g.groupby("i").cumcount()
    return g.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Leitores de janela: memoria (lote) e disco (consulta). Mesma saida.
# ---------------------------------------------------------------------------

def _limpar(a: np.ndarray, nodata, arquivo: str = "dem_filled.tif") -> np.ndarray:
    """NoData vira NaN. Camada do vetor em float64 continua float64: o indice de 09/10 achou dem/hand em
    float64 em todos os 1.649 rasters e o lote do TC01 leu o valor completo (cortar para float32 mudava
    elevation_m em ate 7,6e-6 m no piloto). Acumulacoes vao a float32, como na janela do N1."""
    manter = a.dtype == np.float64 and arquivo in CAMADAS_OBRIGATORIAS
    a = a.astype(np.float64 if manter else np.float32, copy=False)
    if nodata is not None and np.isfinite(nodata):
        a[a == a.dtype.type(nodata)] = np.nan
    return a


class LeitorDeDisco:
    """Consulta: le so a janela de cada camada em torno do ponto (boundless; fora do raster vira NaN e `fora`)."""

    def __init__(self, pasta: Path):
        self.pasta = Path(pasta)

    def janelas(self, lin: int, col: int, k: int = K_LEITURA, arquivos=None) -> tuple[dict, np.ndarray]:
        import rasterio
        from rasterio.windows import Window
        out, fora = {}, None
        for a in arquivos or [x for x in LIDAS if (self.pasta / x).exists()]:
            with rasterio.open(self.pasta / a) as r:
                w = Window(col - k, lin - k, 2 * k + 1, 2 * k + 1)
                z = r.read(1, window=w, boundless=True, fill_value=np.nan if "float" in r.dtypes[0] else 0)
                z = _limpar(z, r.nodata, a)
                if fora is None:
                    ls = np.arange(lin - k, lin + k + 1)[:, None]
                    cs = np.arange(col - k, col + k + 1)[None, :]
                    fora = (ls < 0) | (ls >= r.height) | (cs < 0) | (cs >= r.width)
                z[fora] = np.nan
                out[a] = z
        return out, fora


class LeitorEmMemoria:
    """Lote: carrega cada camada uma vez, com moldura de NaN de K_LEITURA celulas, e fatia janelas."""

    def __init__(self, pasta: Path, arquivos=None, k: int = K_LEITURA):
        import rasterio
        self.pasta, self.k = Path(pasta), k
        self.camadas = {}
        for a in arquivos or [x for x in LIDAS if (Path(pasta) / x).exists()]:
            with rasterio.open(self.pasta / a) as r:
                z = _limpar(r.read(1), r.nodata, a)
                h, w = z.shape
            m = np.full((h + 2 * k, w + 2 * k), np.nan, z.dtype)
            m[k:k + h, k:k + w] = z
            self.camadas[a] = m
            self.forma = (h, w)
        h, w = self.forma
        self.fora = np.ones((h + 2 * k, w + 2 * k), bool)
        self.fora[k:k + h, k:k + w] = False

    def janelas(self, lin: int, col: int, k: int = K_LEITURA, arquivos=None) -> tuple[dict, np.ndarray]:
        if k > self.k:
            raise ValueError("janela maior que a moldura carregada")
        a0, b0 = lin + self.k - k, col + self.k - k
        s = (slice(a0, a0 + 2 * k + 1), slice(b0, b0 + 2 * k + 1))
        return ({a: z[s] for a, z in self.camadas.items() if arquivos is None or a in arquivos},
                self.fora[s])


# ---------------------------------------------------------------------------
# A funcao unica: janelas -> atributos
# ---------------------------------------------------------------------------

_NORMALIZACAO = None


def normalizacao_do_encoder(manifesto: Path = ENCODER) -> tuple[np.ndarray, np.ndarray, str]:
    global _NORMALIZACAO
    if _NORMALIZACAO is None or _NORMALIZACAO[2] != str(manifesto):
        m = json.loads(Path(manifesto).read_text(encoding="utf-8"))
        _NORMALIZACAO = (np.asarray(m["normalizacao"]["media"], np.float32).reshape(-1, 1, 1),
                         np.asarray(m["normalizacao"]["desvio"], np.float32).reshape(-1, 1, 1), str(manifesto))
    return _NORMALIZACAO


def pilha_da_vizinhanca(janelas: dict, fora: np.ndarray, normalizacao=None) -> np.ndarray | None:
    """Janela 128 x 128 das cinco camadas, transformada como no N1. None se sai do raster ou tem < 50% valido."""
    k = (fora.shape[0] - 1) // 2
    a, b = k - LADO_VIZINHANCA // 2, k + LADO_VIZINHANCA // 2      # Window(col-64, lin-64, 128, 128) do N1
    if any(f"{c}.tif" not in janelas for c in CAMADAS_VIZINHANCA) or fora[a:b, a:b].any():
        return None
    pilha = np.stack([janelas[f"{c}.tif"][a:b, a:b] for c in CAMADAS_VIZINHANCA]).astype(np.float32, copy=True)
    pilha[pilha == SEM_DADO] = np.nan
    valido = np.isfinite(pilha).all(0)
    if valido.mean() < 0.5:
        return None
    for i, c in enumerate(CAMADAS_VIZINHANCA):
        if c in CAMADAS_EM_LOG:
            pilha[i] = np.log1p(np.where(valido & (pilha[i] > 0), pilha[i], 0.0))
    media, desvio, _ = normalizacao or normalizacao_do_encoder()
    pilha = (pilha - media) / desvio
    pilha[:, ~valido] = 0.0
    return pilha


def estatisticas_da_vizinhanca(pilhas: np.ndarray) -> np.ndarray:
    """25 numeros por janela: a funcao `estatisticas_multiescala` do V1, na escala de 128 px."""
    from ci_v1_vizinhanca_multiescala import estatisticas_multiescala
    # float32 e a precisao em que o lote guarda: os percentis saem em float64 e a consulta devolvia mais
    # digitos que a linha materializada (achado da demonstracao de 09/10).
    return estatisticas_multiescala(pilhas, np.arange(len(pilhas)),
                                    escalas=(LADO_VIZINHANCA,))[LADO_VIZINHANCA].astype(np.float32)


def posicao_da_janela(z: np.ndarray, fora: np.ndarray, px: float = RESOLUCAO_M) -> dict:
    """pos_min/dev nas tres escalas, com a regra de borda de `posicao_no_ponto.no_ponto`.

    Medido em 09/10/2026: em `posicao_no_ponto`, `dev_*` do lote (vista do raster
    inteiro) e da consulta (copia da janela) diferem ate ~1e-5 relativo -- media e
    desvio em float32 dependem da ordem da soma, que depende do layout. `pos_min`
    (minimo e uma subtracao) nao depende e fica em float32, igual ao pacote
    congelado; `dev` sai de uma copia contigua em float64, a mesma nos dois caminhos.
    """
    k = (z.shape[0] - 1) // 2
    # o MDT entra em float32 como em `posicao_no_ponto._ler`/`no_ponto`; a copia em float64 tem os mesmos
    # valores, e a diferenca z0 - min em float64 e exata, entao arredondar para float32 reproduz o pos_min
    # do lote bit a bit (arredondamento correto IEEE 754) com uma unica chamada de `descritores`.
    z64 = np.array(z, dtype=np.float32).astype(np.float64)
    out = {c: np.nan for c in pp.COLUNAS}
    for r in pp.RAIOS_M:
        kk = pp.raio_em_celulas(r, px)
        sub = (slice(k - kk, k + kk + 1), slice(k - kk, k + kk + 1))
        if fora[sub].any():
            continue
        e = f"{r // 1000}km"
        d = pp.descritores(z64[sub], kk, kk, px)
        p = d[f"pos_min_{e}"]
        out[f"pos_min_{e}"] = float(np.float32(p)) if np.isfinite(p) else np.nan
        out[f"dev_{e}"] = d[f"dev_{e}"]
    return out


def descrever(janelas: dict, fora: np.ndarray, px: float = RESOLUCAO_M, vizinhanca: bool = True,
              normalizacao=None) -> tuple[dict, np.ndarray | None]:
    """A UNICA traducao de janelas para atributos. Devolve (atributos, pilha da vizinhanca ou None).

    As estatisticas da vizinhanca sao calculadas por quem chama, em bloco, com
    `estatisticas_da_vizinhanca` -- a mesma funcao para uma janela ou para mil.
    """
    k = (fora.shape[0] - 1) // 2
    out = {}
    for a in ATRIBUTOS_DO_PIXEL:
        z = janelas.get(ARQUIVO_DO_ATRIBUTO[a])
        v = np.nan if z is None or fora[k, k] else float(z[k, k])
        out[a] = v if np.isfinite(v) and v != SEM_DADO else np.nan
    dem = janelas.get("dem_filled.tif")
    out.update(posicao_da_janela(dem, fora, px) if dem is not None else {c: np.nan for c in pp.COLUNAS})
    pilha = pilha_da_vizinhanca(janelas, fora, normalizacao) if vizinhanca else None
    return out, pilha


def motivo_da_ausencia(atributo: str, valores: dict, borda_celulas: int, px: float = RESOLUCAO_M) -> str | None:
    if np.isfinite(valores.get(atributo, np.nan)):
        return None
    if not np.isfinite(valores.get("elevation_m", np.nan)):
        return "mdt_sem_dado_no_ponto"
    if atributo.startswith(("pos_min", "dev")):
        r = int(atributo.split("_")[-1].replace("km", "")) * 1000
        if borda_celulas < pp.raio_em_celulas(r, px):
            return "janela_sai_do_raster"
        return "janela_com_menos_de_metade_valida_ou_plana"
    if atributo.startswith("viz_"):
        if borda_celulas < LADO_VIZINHANCA // 2:
            return "janela_sai_do_raster"
        return "janela_com_menos_de_metade_valida"
    if atributo == "hand_m":
        return "hand_sem_drenagem_alcancavel_no_raster"
    return "camada_sem_dado_no_ponto"


# ---------------------------------------------------------------------------
# Consulta
# ---------------------------------------------------------------------------

class Acervo:
    """Indice carregado. `consultar_ponto` e `contextualizar` seguem a mesma ordem e a mesma funcao."""

    def __init__(self, saida: Path = SAIDA, raiz: Path | None = None, hidrologia: bool = True,
                 dominio: bool = True):
        self.hidrologia, self.dominio = hidrologia, dominio
        self._bacias = None
        self.saida = Path(saida)
        self.man = json.loads((self.saida / "manifesto.json").read_text(encoding="utf-8"))
        self.raiz = Path(raiz) if raiz else REPO / self.man["raiz_terreno"]
        u = pd.read_parquet(self.saida / "unidades.parquet")
        tem = u.camadas.fillna("").apply(lambda s: all(a in s.split(",") for a in CAMADAS_OBRIGATORIAS))
        self.unidades = u[tem & u.bbox.notna()].reset_index(drop=True)
        self.todas = u
        p = self.saida / "pegadas.parquet"
        self.pegadas = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        self._leitores: dict = {}

    def leitor(self, unidade: str, em_memoria: bool = False):
        if not em_memoria:
            return LeitorDeDisco(self.raiz / unidade)
        if unidade not in self._leitores:
            if len(self._leitores) >= 2:
                self._leitores.clear()
            self._leitores[unidade] = LeitorEmMemoria(self.raiz / unidade)
        return self._leitores[unidade]

    def candidatos(self, lon: float, lat: float) -> pd.DataFrame:
        return geometria(self.unidades, np.array([lon]), np.array([lat]))

    def observacoes(self, lon: float, lat: float) -> list[dict]:
        if self.pegadas.empty:
            return []
        out = []
        for r in self.pegadas.to_dict("records"):
            b = json.loads(r["caixa"])
            if not (b[0] <= lon <= b[2] and b[1] <= lat <= b[3]):
                continue
            partes = json.loads(r["partes"]) if isinstance(r.get("partes"), str) else None
            if partes and not any(p[0] <= lon <= p[2] and p[1] <= lat <= p[3] for p in partes):
                continue
            out.append({k: r[k] for k in ("evento_id", "fonte", "papel", "instrumento", "mecanismo", "janela_inicio",
                                          "janela_fim", "rotulo_acessivel", "origem_da_caixa")})
        return out

    def consultar_ponto(self, lon: float, lat: float, vizinhanca: bool = True) -> dict:
        g = self.candidatos(lon, lat)
        avaliadas, escolhida = [], None
        for c in g.itertuples(index=False):
            jan, fora = self.leitor(c.unidade).janelas(int(c.lin), int(c.col))
            vals, pilha = descrever(jan, fora, vizinhanca=vizinhanca)
            if pilha is not None:
                vals.update(zip(COLUNAS_VIZINHANCA, estatisticas_da_vizinhanca(pilha[None])[0].tolist()))
            elif vizinhanca:
                vals.update({k: np.nan for k in COLUNAS_VIZINHANCA})
            completo = all(np.isfinite(vals[a]) for a in NUCLEO)
            avaliadas.append({"unidade": c.unidade, "ordem": int(c.ordem), "compativel": bool(c.compativel),
                              "borda_m": float(c.borda_m), "borda_celulas": int(c.borda_celulas),
                              "no_nucleo": bool(c.no_nucleo), "nucleo_finito": completo, "valores": vals})
            if escolhida is None and completo:
                escolhida = avaliadas[-1]
        if escolhida is None and avaliadas:
            escolhida = avaliadas[0]
        return self._resposta(lon, lat, avaliadas, escolhida, vizinhanca)

    def _resposta(self, lon, lat, avaliadas, escolhida, vizinhanca) -> dict:
        obs = self.observacoes(lon, lat)
        lacunas = []
        atributos = {}
        nomes = (*ATRIBUTOS_DO_PIXEL, *pp.COLUNAS, *(COLUNAS_VIZINHANCA if vizinhanca else ()))
        for a in nomes:
            if escolhida is None:
                atributos[a] = {"valor": None, "estado": "desconhecido", "motivo": "nenhuma_unidade_de_terreno_cobre"}
                continue
            v = escolhida["valores"].get(a, np.nan)
            estado = ("medido" if a == "elevation_m" else "derivado") if np.isfinite(v) else "desconhecido"
            mot = motivo_da_ausencia(a, escolhida["valores"], escolhida["borda_celulas"])
            atributos[a] = {"valor": float(v) if np.isfinite(v) else None, "estado": estado,
                            "unidade": escolhida["unidade"], "motivo": mot}
        faltam = [a for a, x in atributos.items() if x["estado"] == "desconhecido"]
        if faltam:
            outras = [x["unidade"] for x in avaliadas if x is not escolhida and x["nucleo_finito"]]
            lacunas.append({"tipo": "falta_de_atributo", "atributos": faltam,
                            "motivos": sorted({atributos[a]["motivo"] for a in faltam}),
                            "resolve_por_derivacao": "talvez: derivar terreno numa unidade que cubra a janela"
                            if not outras else "ja ha alternativa compativel com o nucleo finito"})
        if escolhida is not None and not escolhida["compativel"]:
            lacunas.append({"tipo": "fonte_incompativel", "unidade": escolhida["unidade"],
                            "motivo": self._motivo_incompat(escolhida["unidade"])})
        legiveis = [o for o in obs if o["rotulo_acessivel"]]
        if not legiveis:
            lacunas.append({"tipo": "falta_de_observacao",
                            "motivo": ("so ha observacao reservada ou candidata aqui (rotulo nao lido)" if obs
                                       else "nenhuma pegada de observacao no acervo cobre o ponto"),
                            "nao_e": "ausencia de observacao nao e negativo"})
        if any(o["papel"] in ("desenvolvimento_conhecido", "teste_examinado") for o in obs):
            lacunas.append({"tipo": "falta_de_independencia",
                            "motivo": "lugar ja visto em desenvolvimento ou teste examinado: avaliar aqui nao e independente",
                            "eventos": sorted({o["evento_id"] for o in obs
                                               if o["papel"] in ("desenvolvimento_conhecido", "teste_examinado")})[:20]})
        dom = self._dominio_aquatico(lon, lat) if self.dominio else {"estado": "nao_consultado"}
        if dom.get("dominio_aquatico") == "fora_do_dominio_terrestre":
            lacunas.append({"tipo": "fora_do_dominio_terrestre", "estado_aquatico": dom["estado_aquatico"],
                            "motivo": "mar ou agua permanente: o descritor de terreno existe mas nao descreve "
                                      "suscetibilidade de terra aqui"})
        elif dom.get("estado") in ("indisponivel", "nao_consultado"):
            agua = self._agua_permanente(escolhida, lon, lat) if escolhida is not None else None
            if agua is None and escolhida is not None:
                lacunas.append({"tipo": "mascara_de_agua_ausente",
                                "motivo": ("sem dominio aquatico e sem agua_permanente_jrc na unidade: mar e lago tem "
                                           "valor no dem_filled e aparecem como terreno plano")})
            elif agua == 1:
                lacunas.append({"tipo": "agua_permanente", "motivo": "JRC occurrence >= 50 no ponto"})
        hid = self._hidrologia_do_ponto(lon, lat) if self.hidrologia else {"estado": "nao_consultado"}
        if hid.get("estado") != "unica":
            lacunas.append({"tipo": "unidade_hidrologica_desconhecida",
                            "motivo": {"fora_de_toda_bacia": "fora de toda sub-bacia HydroBASINS (mar ou fora da regiao)",
                                       "ambigua_na_borda": "ponto na fronteira entre sub-bacias",
                                       }.get(hid.get("estado"), "HydroBASINS indisponivel nesta consulta")})
        alternativas = []
        for x in avaliadas:
            if x is escolhida:
                continue
            dif = {}
            if escolhida is not None:
                for a in NUCLEO:
                    va, vb = x["valores"].get(a, np.nan), escolhida["valores"].get(a, np.nan)
                    dif[a] = float(va - vb) if np.isfinite(va) and np.isfinite(vb) else None
            alternativas.append({"unidade": x["unidade"], "ordem": x["ordem"], "compativel": x["compativel"],
                                 "borda_m": x["borda_m"], "no_nucleo": x["no_nucleo"],
                                 "nucleo_finito": x["nucleo_finito"], "diferenca_para_escolhida": dif,
                                 "motivo_incompatibilidade": None if x["compativel"]
                                 else self._motivo_incompat(x["unidade"])})
        cobertura = ("ausente" if escolhida is None else
                     "completa" if escolhida["nucleo_finito"] and escolhida["compativel"] else "parcial")
        return {"ponto": {"lon": lon, "lat": lat}, "cobertura": cobertura,
                "escolhida": None if escolhida is None else
                {k: escolhida[k] for k in ("unidade", "ordem", "compativel", "borda_m", "no_nucleo", "nucleo_finito")}
                | {"regra": "compativel > mais longe da borda > nome; primeira com nucleo finito"},
                "atributos": atributos, "alternativas": alternativas, "observacoes": obs, "lacunas": lacunas,
                "dominio_aquatico": dom, "hidrologia": hid,
                "indice": self.man["hash_resumo"],
                "nao_e": "nao e escore nem rotulo; desconhecido nao e zero e ausencia de observacao nao e negativo"}

    def _hidrologia_do_ponto(self, lon: float, lat: float) -> dict:
        """Sub-bacias HydroBASINS (niveis 6, 8 e 12) e sistema fluvial; 'indisponivel' se as tabelas faltam."""
        try:
            import bacias_hydrobasins as bh
            if self._bacias is None:
                self._bacias = bh.Bacias()
            out = {"fonte": "HydroBASINS v1c (Lehner & Grill 2013)", "hash": self._bacias.man["hash_resumo"]}
            for n in (6, 8, 12):
                r = self._bacias.localizar([lon], [lat], n).iloc[0]
                out[f"nivel_{n:02d}"] = int(r.HYBAS_ID)
                out["estado"] = r.estado_bacia
                if n == 6:
                    out["sistema_fluvial"] = int(r.MAIN_BAS)
            out["nao_e"] = "drenagem de 500 m (SRTM), nao a do ter-01; ID diferente nao garante independencia"
            return out
        except (FileNotFoundError, ImportError, OSError) as e:
            return {"estado": "indisponivel", "motivo": f"{type(e).__name__}: {e}"[:160]}

    def _dominio_aquatico(self, lon: float, lat: float) -> dict:
        """Estado aquatico pela regra de `dominio_aquatico` (WBM em cache + JRC remoto); sem rede vira desconhecido."""
        try:
            import dominio_aquatico as da
            r = da.amostrar([lon], [lat]).iloc[0]
            return {"estado": "consultado", "regra": da.VERSAO_REGRA, "estado_aquatico": r.estado_aquatico,
                    "dominio_aquatico": r.dominio_aquatico, "wbm": None if not np.isfinite(r.wbm) else int(r.wbm),
                    "jrc_occurrence": None if not np.isfinite(r.jrc_occurrence) else int(r.jrc_occurrence),
                    "jrc_seasonality_2021": None if not np.isfinite(r.jrc_seasonality_2021)
                    else int(r.jrc_seasonality_2021)}
        except Exception as e:  # noqa: BLE001 -- rede ou cache indisponivel: estado desconhecido declarado
            return {"estado": "indisponivel", "motivo": f"{type(e).__name__}: {e}"[:160]}

    def _agua_permanente(self, escolhida: dict, lon: float, lat: float) -> int | None:
        """1/0 da mascara JRC da unidade no ponto; None se a unidade nao tem a camada."""
        arq = self.raiz / escolhida["unidade"] / "agua_permanente_jrc.tif"
        if not arq.exists():
            return None
        import rasterio
        from pyproj import Transformer
        with rasterio.open(arq) as r:
            x, y = Transformer.from_crs("EPSG:4326", r.crs, always_xy=True).transform(lon, lat)
            v = next(r.sample([(x, y)]))[0]
            return None if r.nodata is not None and v == r.nodata else int(v)

    def _motivo_incompat(self, unidade: str) -> str | None:
        r = self.todas[self.todas.unidade == unidade]
        return None if r.empty else r.iloc[0].motivo_incompatibilidade

    def consultar_aoi(self, caixa: list[float], passo_m: float = 1000.0, vizinhanca: bool = False,
                      maximo: int = 5000) -> dict:
        """Grade regular dentro da caixa, passada pelo MESMO caminho do lote. Cobertura lida, nao so caixa."""
        lon0, lat0, lon1, lat1 = caixa
        dlat = passo_m / 111_320.0
        dlon = passo_m / (111_320.0 * max(0.2, math.cos(math.radians((lat0 + lat1) / 2))))
        lons = np.arange(lon0 + dlon / 2, lon1, dlon)
        lats = np.arange(lat0 + dlat / 2, lat1, dlat)
        if len(lons) * len(lats) > maximo:
            raise ValueError(f"grade de {len(lons) * len(lats)} pontos acima do maximo {maximo}; aumente o passo")
        LO, LA = np.meshgrid(lons, lats)
        pts = pd.DataFrame({"ponto_id": np.arange(LO.size), "lon": LO.ravel(), "lat": LA.ravel()})
        t = contextualizar(pts, self, vizinhanca=vizinhanca)
        nucleo = np.isfinite(t[list(NUCLEO)].to_numpy(float)).all(1)
        caixas_compat = self.unidades[self.unidades.compativel]
        return {"caixa": caixa, "passo_m": passo_m, "pontos": int(len(t)),
                "com_unidade": int(t.unidade_terreno.notna().sum()),
                "nucleo_finito": int(nucleo.sum()),
                "fracao_nucleo_finito": round(float(nucleo.mean()), 4) if len(t) else None,
                "em_unidade_incompativel": int((t.unidade_compativel == False).sum()),  # noqa: E712
                "por_unidade": t.unidade_terreno.value_counts(dropna=False).head(20).to_dict(),
                "motivos_de_ausencia": t.motivo_ausencia.value_counts().to_dict(),
                "unidades_compativeis_que_tocam": int(len(geometria(caixas_compat, pts.lon.to_numpy(),
                                                                    pts.lat.to_numpy()).unidade.unique())),
                "observacoes": self.observacoes_na_caixa(caixa),
                "nao_e": "cobertura de descritor, nao de observacao; caixa de pegada e triagem, nao mancha"}

    def observacoes_na_caixa(self, caixa) -> list[dict]:
        if self.pegadas.empty:
            return []
        out = []
        for r in self.pegadas.to_dict("records"):
            b = json.loads(r["caixa"])
            if b[0] <= caixa[2] and b[2] >= caixa[0] and b[1] <= caixa[3] and b[3] >= caixa[1]:
                out.append({k: r[k] for k in ("evento_id", "fonte", "papel", "rotulo_acessivel")})
        return out


# ---------------------------------------------------------------------------
# Lote
# ---------------------------------------------------------------------------

BLOCO_VIZINHANCA = 512


def _avaliar_unidade(leitor, unidade: str, idx: np.ndarray, geo: pd.DataFrame, vizinhanca: bool) -> pd.DataFrame:
    """Pilhas da vizinhanca resumidas a cada bloco: guarda-las ate o fim custava ~4 GB por unidade grande."""
    linhas, pilhas, onde = [], [], []
    viz = np.full((len(geo), len(COLUNAS_VIZINHANCA)), np.nan, np.float32) if vizinhanca else None

    def _resumir():
        if pilhas:
            viz[onde] = estatisticas_da_vizinhanca(np.stack(pilhas))
            pilhas.clear()
            onde.clear()
    for j, r in enumerate(geo.itertuples(index=False)):
        jan, fora = leitor.janelas(int(r.lin), int(r.col))
        vals, pilha = descrever(jan, fora, vizinhanca=vizinhanca)
        linhas.append(vals)
        if pilha is not None:
            pilhas.append(pilha)
            onde.append(j)
            if len(pilhas) == BLOCO_VIZINHANCA:
                _resumir()
    _resumir()
    t = pd.DataFrame(linhas, columns=[*ATRIBUTOS_DO_PIXEL, *pp.COLUNAS])
    if vizinhanca:
        t[list(COLUNAS_VIZINHANCA)] = viz
    t.insert(0, "i", idx)
    t["unidade_terreno"] = unidade
    t["borda_celulas"] = geo.borda_celulas.to_numpy()
    t["no_nucleo"] = geo.no_nucleo.to_numpy()
    t["unidade_compativel"] = geo.compativel.to_numpy()
    t["ordem_candidato"] = geo.ordem.to_numpy()
    return t


def _tarefa(args) -> pd.DataFrame:
    """Uma (rodada, unidade). Funcao de modulo para rodar em processo separado; grava o checkpoint."""
    raiz, unidade, gu, vizinhanca, em_memoria, arq = args
    leitor = LeitorEmMemoria(Path(raiz) / unidade) if em_memoria else LeitorDeDisco(Path(raiz) / unidade)
    t = _avaliar_unidade(leitor, unidade, gu.i.to_numpy(), gu, vizinhanca)
    if arq is not None:
        t.to_parquet(Path(arq).with_suffix(".part"), index=False)
        Path(arq).with_suffix(".part").replace(arq)
    return t


def contextualizar(pontos: pd.DataFrame, acervo: Acervo, vizinhanca: bool = True, em_memoria: bool | None = None,
                   checkpoint: Path | None = None, progresso: bool = False, processos: int = 1) -> pd.DataFrame:
    """Contexto de muitos pontos com a MESMA ordem e a MESMA funcao da consulta.

    Rodada r avalia, para cada ponto pendente, o r-esimo candidato; quem sai com
    o nucleo finito fica, quem nao sai passa ao seguinte. Ponto sem candidato com
    nucleo finito fica com o primeiro candidato e o que falta e desconhecido.
    Com `checkpoint`, cada (rodada, unidade) e gravada e a retomada a pula.
    `processos` > 1 distribui unidades entre processos; o resultado nao depende disso.
    """
    lon, lat = pontos.lon.to_numpy(float), pontos.lat.to_numpy(float)
    geo = geometria(acervo.unidades, lon, lat)
    n = len(pontos)
    if em_memoria is None:
        em_memoria = n > 2000
    if checkpoint:
        Path(checkpoint).mkdir(parents=True, exist_ok=True)
    pendentes = np.arange(n)
    aceitos, primeira = [], None
    max_ordem = int(geo.ordem.max()) + 1 if len(geo) else 0
    inicio = time.time()
    for rodada in range(max_ordem):
        g = geo[(geo.ordem == rodada) & geo.i.isin(pendentes)]
        if g.empty:
            continue
        grupos = list(g.groupby("unidade", sort=True))
        prontos, tarefas = {}, []
        for unidade, gu in grupos:
            arq = Path(checkpoint) / f"r{rodada:02d}_{unidade}.parquet" if checkpoint else None
            if arq is not None and arq.exists():
                t = pd.read_parquet(arq)
                if not np.array_equal(np.sort(t.i.to_numpy()), np.sort(gu.i.to_numpy())):
                    raise RuntimeError(f"checkpoint {arq.name} de outra lista de pontos")
                prontos[unidade] = t
            else:
                tarefas.append((str(acervo.raiz), unidade, gu, vizinhanca, em_memoria, arq))
        feitos = 0

        def _relatar(unidade, m):
            if progresso:
                print(f"rodada {rodada} {feitos}/{len(tarefas)} {unidade} {m} pts {round(time.time() - inicio)} s",
                      flush=True)
        if processos > 1 and len(tarefas) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=processos) as pool:
                futuros = {pool.submit(_tarefa, tk): tk[1] for tk in tarefas}
                for fu in as_completed(futuros):
                    prontos[futuros[fu]] = fu.result()
                    feitos += 1
                    _relatar(futuros[fu], len(prontos[futuros[fu]]))
        else:
            for tk in tarefas:
                prontos[tk[1]] = _tarefa(tk)
                feitos += 1
                _relatar(tk[1], len(prontos[tk[1]]))
        t = pd.concat([prontos[u] for u, _ in grupos], ignore_index=True)
        ok = np.isfinite(t[list(NUCLEO)].to_numpy(float)).all(1)
        aceitos.append(t[ok])
        if rodada == 0:
            primeira = t[~ok]
        pendentes = np.setdiff1d(pendentes, t.i[ok].to_numpy())
    partes = list(aceitos)
    if primeira is not None:
        partes.append(primeira[primeira.i.isin(pendentes)])
    out = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame({"i": np.array([], int)})
    faltam = np.setdiff1d(np.arange(n), out.i.to_numpy())
    if len(faltam):
        out = pd.concat([out, pd.DataFrame({"i": faltam})], ignore_index=True)
    for c in (*ATRIBUTOS_DO_PIXEL, *pp.COLUNAS, *(COLUNAS_VIZINHANCA if vizinhanca else ())):
        if c not in out:
            out[c] = np.nan
    for c in ("unidade_terreno", "borda_celulas", "no_nucleo", "unidade_compativel", "ordem_candidato"):
        if c not in out:
            out[c] = None
    out = out.sort_values("i").reset_index(drop=True)
    if not np.array_equal(out.i.to_numpy(), np.arange(n)):
        raise RuntimeError("contexto com ponto duplicado ou perdido")
    cand = geo.groupby("i").agg(n_candidatos=("unidade", "size"), n_compativeis=("compativel", "sum"))
    out = out.join(cand, on="i")
    out[["n_candidatos", "n_compativeis"]] = out[["n_candidatos", "n_compativeis"]].fillna(0).astype(int)
    out["nucleo_finito"] = np.isfinite(out[list(NUCLEO)].to_numpy(float)).all(1)
    motivo = np.full(n, None, dtype=object)
    for j in np.flatnonzero(~out.nucleo_finito.to_numpy()):
        r = out.iloc[j]
        if pd.isna(r.unidade_terreno):
            motivo[j] = "nenhuma_unidade_de_terreno_cobre"
        else:
            vals = {a: float(r[a]) for a in NUCLEO}
            motivo[j] = motivo_da_ausencia(next(a for a in NUCLEO if not np.isfinite(vals[a])), vals,
                                           int(r.borda_celulas))
    out["motivo_ausencia"] = motivo
    out.insert(0, "ponto_id", pontos.ponto_id.to_numpy())
    return out.drop(columns=["i"])


# ---------------------------------------------------------------------------

def _json_seguro(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--indexar", action="store_true")
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--ponto", nargs=2, type=float, metavar=("LON", "LAT"))
    ap.add_argument("--aoi", nargs=4, type=float, metavar=("LON0", "LAT0", "LON1", "LAT1"))
    ap.add_argument("--passo-m", type=float, default=1000.0)
    ap.add_argument("--sem-vizinhanca", action="store_true")
    a = ap.parse_args()
    if a.indexar:
        m = indexar()
        print(json.dumps({k: m[k] for k in ("unidades", "compativeis", "por_familia_mdt", "diario", "segundos",
                                            "hash_resumo")}, ensure_ascii=False))
    if a.verificar:
        print(json.dumps(verificar(), ensure_ascii=False))
    if a.ponto:
        r = Acervo().consultar_ponto(*a.ponto, vizinhanca=not a.sem_vizinhanca)
        print(json.dumps(r, ensure_ascii=False, indent=1, default=_json_seguro))
    if a.aoi:
        r = Acervo().consultar_aoi(list(a.aoi), a.passo_m, vizinhanca=not a.sem_vizinhanca)
        print(json.dumps(r, ensure_ascii=False, indent=1, default=_json_seguro))
    if not (a.indexar or a.verificar or a.ponto or a.aoi):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
