"""Pontos de uma ativacao CEMS: positivo e negativo OBSERVADOS, com terreno.

-------------------------------------------------------------------------
POR QUE ISTO EXISTE
-------------------------------------------------------------------------

`cems01_pacote_areal` declara, na primeira linha, que NAO amostra ponto. E o
`ds01_montar_multirregiao` le `local_runs/cems-emsr*/pontos_*.csv` como se esse
arquivo aparecesse sozinho. Entre os dois havia um passo ad-hoc: os pontos do
EMSR734 foram gerados em 11/08/2026 por codigo que nao ficou no repositorio.

Enquanto for ad-hoc, cada ativacao nova custa o mesmo trabalho manual -- e e
exatamente a dependencia que a frente de aprendizado autonomo existe para
remover.

-------------------------------------------------------------------------
O QUE E POSITIVO E O QUE E NEGATIVO, e o que sai das duas classes
-------------------------------------------------------------------------

    observado    = (areaOfInterest INTERSECAO imageFootprint)
    positivo     = observado INTERSECAO observedEvent[event_type 5-Flood]
    deslizamento = observedEvent[event_type 6-Mass Movement]
    negativo     = observado MENOS positivo MENOS deslizamento

A intersecao com o footprint e obrigatoria: area da AOI que a imagem nao cobriu
NAO foi observada, e nao pode virar negativo. E o deslizamento sai das DUAS
classes -- nao e inundacao, e tambem nao e negativo util, porque e outro perigo
com terreno alterado. As tres regras vem do `br-01-cems-negativo`, que fez isso
a mao para o EMSR720, e este modulo as reproduz em codigo.

-------------------------------------------------------------------------
AGUA PERMANENTE: O QUE O EMSR720 NAO TEM, E POR QUE ISSO VAI DECLARADO
-------------------------------------------------------------------------

O pacote do EMSR734 tem TRES classes -- 1.080 nao-inundado, 1.080 AGUA
PERMANENTE e 190 inundacao --, e a agua permanente veio de camada externa,
porque o produto EMS nao a marca.

No EMSR720 o `obj_desc` tem so `Riverine flood` (1.490) e `Landslide` (497).
Nao existe agua permanente delineada. Entao o canal do rio -- que ja estava la
antes do evento -- cai dentro do NEGATIVO, e um modelo treinado assim
aprenderia "leito de rio = nao inundado", que e o inverso da fisica.

Este modulo NAO resolve isso inventando o rotulo. Ele:

  1. declara a limitacao no manifesto, com o nome do campo que faltou;
  2. grava `sobre_canal_modelado`, marcando se o ponto cai na rede de drenagem
     que a cadeia canonica extraiu (limiar 0,1123 km2).

O canal modelado NAO e observacao de agua, e por isso ele nao vira classe aqui:
usar modelo para definir rotulo e o erro que o projeto persegue. Ele fica como
COLUNA, e quem decidir excluir o leito faz isso a jusante, declarando.

-------------------------------------------------------------------------
A AMOSTRAGEM, e por que espacamento fixo e nao sorteio
-------------------------------------------------------------------------

Espacamento de 150 m sobre a grade de 30 m do proprio raster derivado -- cada
5o pixel em x e y, no centro da celula. E deterministico: duas corridas dao os
mesmos pontos, sem semente. Sorteio exigiria guardar a semente para reproduzir,
e o projeto ja pagou por amostragem irreproduzivel uma vez (`stable_rank`
incluindo `csv_path`).

Amostrar TODO pixel de 30 m daria ~280 mil pontos em 257 km2, todos
espacialmente autocorrelacionados dentro de 5 grupos. Mais linha nao e mais
informacao quando o grupo nao muda.

O GRUPO E A AOI, e nao a ativacao. Isso vem do `ds04.reduzir_cems`: "no CEMS o
grupo de validacao E a AOI: e onde o analista desenhou". Agrupar por ativacao
deu EPV 1,67 no projeto e ja foi corrigido em 09/08/2026.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao promove rotulo, nao treina, nao toca em gate e nao escreve no ds01 -- ele
grava o CSV que o ds01 le, e o status sai CANDIDATO. Nao baixa nada: exige o
produto extraido e o terreno ja derivado pela cadeia canonica.

Uso:
    python scripts/geo/cems03_amostrar_pontos.py --ativacao EMSR720 \\
        --extraido local_runs/br-01-cems-negativo/extraido \\
        --terreno local_runs/ter-01-cadeia-harmonizada/EMSR720
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "geo"))

from cems01_pacote_areal import (  # noqa: E402
    CAMADA_AOI, CAMADA_EVENTO, CAMADA_FOOTPRINT, TIPO_DESLIZAMENTO,
    TIPO_INUNDACAO, ler_camada,
)
from proveniencia import agora_utc, sha256_json  # noqa: E402

RUNS = REPO / "local_runs"
PASSO_M = 150.0
LICENCA = "Copernicus - uso livre com atribuicao"
FONTE = "copernicus_ems_rapid_mapping"
CAMADAS_TERRENO = {
    "elevation_m": "dem_filled.tif",
    "hand_m": "hand_dinf.tif",
    "slope_deg": "slope_deg_wbt.tif",
    "twi_dinf": "twi_dinf.tif",
}
CANAL = "streams_dinf.tif"
# Presentes em todo `_source_v*.xml` de toda ativacao: sao a base auxiliar
# do EMS, e nao a cena do evento.
DATAS_DE_CADASTRO = {"2008-06-01", "2010-12-08"}
# Gerada por `agua_permanente_jrc`, alinhada ao raster de terreno. Quando ela
# existe, o ponto sobre agua permanente vira classe PROPRIA -- nao positivo,
# porque a agua ja estava la, e nao negativo, porque e agua. E a mesma regra
# do `baseline_otsu`: `valido & (jrc == 0)`.
AGUA_PERMANENTE = "agua_permanente_jrc.tif"
MANIFESTO_DA_CADEIA = "run_manifest.json"


def terreno_da_aoi(terreno: Path, ativacao: str, aoi_id: str) -> Path:
    """O diretorio de terreno desta AOI, aceitando raiz OU diretorio unico.

    -------------------------------------------------------------------------
    POR QUE OS DOIS FORMATOS
    -------------------------------------------------------------------------

    Quando este modulo foi escrito, a ativacao tinha UM diretorio de terreno
    (`ter-01/EMSR720`). A fila de derivacao passou a derivar por AOI
    (`ter-01/EMSR838_AOI03`), que e o formato que o `cems07`, o `gfd02` e o
    encoder leem -- e das 15 ativacoes ja aderidas, so a EMSR720 tem o
    diretorio antigo.

    Derivar de novo a ativacao inteira so para satisfazer a assinatura custaria
    horas de CPU e dezenas de gigabytes (a EMSR774 sozinha tem 108.114 km2)
    para reproduzir raster que ja existe. Entao a assinatura passa a aceitar a
    RAIZ: se o caminho recebido tem `run_manifest.json`, e o diretorio unico de
    sempre; se nao tem, e a raiz, e cada AOI resolve o seu.
    """
    if (terreno / MANIFESTO_DA_CADEIA).exists():
        return terreno
    candidato = terreno / f"{ativacao}_{aoi_id}"
    if (candidato / MANIFESTO_DA_CADEIA).exists():
        return candidato
    raise FileNotFoundError(
        f"sem terreno para {ativacao}_{aoi_id}: nem {terreno} tem "
        f"{MANIFESTO_DA_CADEIA}, nem {candidato} existe")


def _crs_do_terreno(terreno: Path) -> int:
    d = json.loads((terreno / "run_manifest.json").read_text(encoding="utf-8"))
    return int(str(d["crs"]).split(":")[1])


def data_do_produto(pasta: Path) -> str | None:
    """A data da cena POS-EVENTO de que este produto foi desenhado.

    -------------------------------------------------------------------
    ESTA FUNCAO LIA O CARIMBO DO METADADO, E NAO A DATA DA IMAGEM
    -------------------------------------------------------------------

    Ate 14/09/2026 ela varria os `_source_v*.xml` com uma expressao regular de
    `20\\d{2}-\\d{2}-\\d{2}` e devolvia a menor data que sobrasse. O que essa
    data e, olhando o XML do EMSR720 AOI02:

        <dateStamp><gco:Date>2024-07-02</gco:Date></dateStamp>

    E o `dateStamp` do ISO 19139 -- quando o REGISTRO DE METADADO foi criado.
    A cena esta em outro lugar, no `_source_v*.dbf` do mesmo produto:

        source_nam  RADARSAT-2     src_date  04/05/2024
        source_tm   T22:10:00Z     eventphase  Post-event

    A diferenca foi medida nos 185 produtos de delineacao dos 25 pacotes em
    disco: mediana de UM dia, e o Rio Grande do Sul como valor extremo com
    mediana de 59 dias e maximo de 72. A enchente do RS foi em 30/04-05/05 e o
    metadado dos mapas foi carimbado em junho e julho.

    Foi dessa leitura que nasceu a frase "os DEL sao de 26/06 a 15/07, que e a
    segunda cheia do Rio Grande do Sul" -- nao ha segunda cheia nesses
    produtos: sao os mesmos mapas de maio, publicados depois.

    O XML fica como ULTIMO recurso, e quando ele for usado o manifesto diz.
    """
    import re

    achados = []
    for dbf in sorted(pasta.rglob("*_source_v*.dbf")):
        try:
            import fiona
            with fiona.open(dbf) as src:
                for f in src:
                    p = dict(f["properties"])
                    if "post" not in str(p.get("eventphase") or "").lower():
                        continue
                    m = re.search(r"(\d{2})/(\d{2})/(\d{4})",
                                  str(p.get("src_date") or ""))
                    if m:
                        achados.append(f"{m.group(3)}-{m.group(2)}-{m.group(1)}")
        except Exception:                                        # noqa: BLE001
            continue
    if achados:
        # a mais recente: a delineacao descreve o estado no ultimo instante
        # observado, e nao no primeiro
        return max(achados)

    datas = set()
    for arq in pasta.rglob("*_source_v*.xml"):
        texto = arq.read_text(encoding="utf-8", errors="replace")
        datas |= set(re.findall(r"20\d{2}-\d{2}-\d{2}", texto))
    datas -= DATAS_DE_CADASTRO
    return min(datas) if datas else None


def camadas_da_aoi(extraido: Path, aoi_id: str, epsg: int,
                   janela: tuple | None = None) -> dict:
    """As tres camadas do EMS para UMA AOI, no CRS do terreno.

    `janela` restringe aos produtos desenhados DENTRO dela, e existe porque uma
    ativacao pode cobrir mais de um evento. O EMSR720 e o caso: os produtos GRA
    sao de 07-08/05/2024 e os DEL sao de 26/06 a 15/07 -- a segunda cheia do Rio
    Grande do Sul naquele ano.
    Unir os dois rotula com maio E julho ao mesmo tempo, e o radar e de um dia
    so. Medido: 1,5% da area de inundacao vinha exclusivamente de jun/jul.
    """
    from datetime import date

    from shapely import union_all

    pasta = extraido
    todos = [p for p in sorted(pasta.iterdir())
             if p.is_dir() and f"_{aoi_id}_" in p.name]
    if not todos:
        raise SystemExit(f"nenhum produto para {aoi_id} em {extraido}")

    alvos, descartados = [], []
    for p in todos:
        d = data_do_produto(p)
        if janela is None:
            alvos.append(p)
            continue
        if d and janela[0] <= date.fromisoformat(d) <= janela[1]:
            alvos.append(p)
        else:
            descartados.append({"produto": p.name, "data": d,
                                "por_que": "fora da janela da campanha"})
    if not alvos:
        raise SystemExit(
            f"nenhum produto de {aoi_id} dentro da janela {janela}")

    def juntar(sufixo: str, filtro=None):
        pecas, ilegiveis = [], []
        for d in alvos:
            feicoes, ruins = ler_camada(d, sufixo, epsg)
            ilegiveis += ruins
            for g, props in feicoes:
                if filtro is None or filtro(props):
                    pecas.append(g)
        return (union_all(pecas) if pecas else None), ilegiveis

    aoi, r1 = juntar(CAMADA_AOI)
    footprint, r2 = juntar(CAMADA_FOOTPRINT)
    cheia, r3 = juntar(
        CAMADA_EVENTO,
        lambda p: str(p.get("event_type", "")).startswith(TIPO_INUNDACAO))
    massa, r4 = juntar(
        CAMADA_EVENTO,
        lambda p: str(p.get("event_type", "")).startswith(TIPO_DESLIZAMENTO))
    return {"aoi": aoi, "footprint": footprint, "cheia": cheia,
            "massa": massa, "ilegiveis": r1 + r2 + r3 + r4,
            "produtos_usados": [p.name for p in alvos],
            "produtos_descartados": descartados}


def classes_da_aoi(c: dict) -> dict:
    """observado, positivo e negativo -- com o footprint obrigatorio."""
    if c["aoi"] is None or c["footprint"] is None:
        return {"observado": None, "positivo": None, "negativo": None}
    observado = c["aoi"].intersection(c["footprint"])
    positivo = (observado.intersection(c["cheia"])
                if c["cheia"] is not None else None)
    negativo = observado
    if positivo is not None:
        negativo = negativo.difference(positivo)
    if c["massa"] is not None:
        negativo = negativo.difference(c["massa"])
    return {"observado": observado, "positivo": positivo, "negativo": negativo}


def pontos_na_geometria(geom, terreno: Path, passo: float) -> list:
    """Centros de celula espacados `passo`, dentro da geometria."""
    import numpy as np
    import rasterio
    from shapely import contains_xy

    if geom is None or geom.is_empty:
        return []
    with rasterio.open(terreno / CAMADAS_TERRENO["elevation_m"]) as src:
        transformacao = src.transform
        largura, altura = src.width, src.height
        px = abs(transformacao.a)
    salto = max(1, int(round(passo / px)))
    x0, y0, x1, y1 = geom.bounds
    colunas = np.arange(0, largura, salto)
    linhas = np.arange(0, altura, salto)
    # centro da celula: +0,5 em coluna e linha
    xs = transformacao.c + (colunas + 0.5) * transformacao.a
    ys = transformacao.f + (linhas + 0.5) * transformacao.e
    xs = xs[(xs >= x0) & (xs <= x1)]
    ys = ys[(ys >= y0) & (ys <= y1)]
    if not len(xs) or not len(ys):
        return []
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    dentro = contains_xy(geom, gx, gy)
    return list(zip(gx[dentro].tolist(), gy[dentro].tolist()))


def amostrar_terreno(coords: list, terreno: Path) -> dict:
    """Valor de cada camada nos pontos, e o sinalizador de canal modelado."""
    import numpy as np
    import rasterio

    saida = {}
    for nome, arquivo in CAMADAS_TERRENO.items():
        with rasterio.open(terreno / arquivo) as src:
            v = np.array([x[0] for x in src.sample(coords)], dtype="float64")
            nd = src.nodata
            if nd is not None:
                v[v == nd] = np.nan
            saida[nome] = v
    for chave, arquivo in (("sobre_canal_modelado", CANAL),
                           ("agua_permanente_jrc", AGUA_PERMANENTE)):
        alvo = terreno / arquivo
        if alvo.exists():
            with rasterio.open(alvo) as src:
                c = np.array([x[0] for x in src.sample(coords)],
                             dtype="float64")
                nd = src.nodata
                saida[chave] = np.where(
                    np.isnan(c) | ((c == nd) if nd is not None else False), 0,
                    (c > 0).astype(int))
        else:
            saida[chave] = np.zeros(len(coords), dtype=int)
    return saida


def construir(ativacao: str, extraido: Path, terreno: Path,
              passo: float = PASSO_M, janela: tuple | None = None) -> dict:
    import numpy as np
    from pyproj import Transformer

    aois = sorted({p.name.split("_")[1] for p in extraido.iterdir()
                   if p.is_dir() and p.name.startswith(ativacao)})
    linhas, por_aoi, ilegiveis = [], {}, []
    sem_terreno, epsgs = [], {}
    n = 0
    for aoi_id in aois:
        try:
            terreno_aoi = terreno_da_aoi(terreno, ativacao, aoi_id)
        except FileNotFoundError as e:
            # AOI sem terreno e AUSENCIA DECLARADA, e nao motivo para abortar a
            # ativacao inteira: as outras AOIs dela continuam validas.
            sem_terreno.append({"aoi": aoi_id, "motivo": str(e)})
            print(f"  [sem terreno] {ativacao}_{aoi_id}", flush=True)
            continue
        epsg = _crs_do_terreno(terreno_aoi)
        epsgs[aoi_id] = epsg
        tr = Transformer.from_crs(epsg, 4326, always_xy=True)
        c = camadas_da_aoi(extraido, aoi_id, epsg, janela)
        ilegiveis += c["ilegiveis"]
        cl = classes_da_aoi(c)
        contagem = {}
        for classe, geom in (("INUNDACAO", cl["positivo"]),
                             ("NAO_INUNDADO_OBSERVADO", cl["negativo"])):
            coords = pontos_na_geometria(geom, terreno_aoi, passo)
            contagem[classe] = len(coords)
            if not coords:
                continue
            vals = amostrar_terreno(coords, terreno_aoi)
            for i, (x, y) in enumerate(coords):
                lon, lat = tr.transform(x, y)
                # AGUA PERMANENTE TEM PRECEDENCIA sobre as duas classes.
                #
                # Ela nao e positivo -- a agua ja estava la antes do evento --
                # e nao e negativo util, porque e agua. E a mesma regra que o
                # `baseline_otsu` aplica com `valido & (jrc == 0)`, e o
                # `MAPA_CLASSE` do ds01 ja reserva a classe 9 para ela.
                #
                # Sem isso o leito do rio entra como negativo de TWI altissimo
                # e INVERTE o coeficiente da variavel: medido no EMSR720, de
                # +0,4408 para -0,1727.
                efetiva = ("AGUA_PERMANENTE"
                           if int(vals["agua_permanente_jrc"][i])
                           else classe)
                linhas.append({
                    "classe_obs": efetiva, "aoi_id": aoi_id,
                    "x_m": round(x, 1), "y_m": round(y, 1),
                    "lon": lon, "lat": lat,
                    **{k: (None if np.isnan(vals[k][i]) else
                           round(float(vals[k][i]), 6))
                       for k in CAMADAS_TERRENO},
                    "worldcover": None,
                    "ativacao": ativacao,
                    "grupo_cv": f"{ativacao}_{aoi_id}",
                    "crs_metrico": epsg, "resolucao_m": 30.0,
                    "fonte": FONTE, "licenca": LICENCA,
                    "sobre_canal_modelado":
                        int(vals["sobre_canal_modelado"][i]),
                    "agua_permanente_jrc":
                        int(vals["agua_permanente_jrc"][i]),
                    "classe_do_poligono": classe,
                    "ponto_id": f"{ativacao}_{n:06d}",
                })
                n += 1
        por_aoi[aoi_id] = {
            "contagem": contagem,
            "produtos_usados": c["produtos_usados"],
            "produtos_descartados": c["produtos_descartados"],
            "km2": {k: round(v.area / 1e6, 4) if v is not None else 0.0
                    for k, v in cl.items()},
        }
    return {"linhas": linhas, "por_aoi": por_aoi, "ilegiveis": ilegiveis,
            "aois_sem_terreno": sem_terreno,
            # O CRS e POR AOI desde que o terreno passou a ser resolvido por
            # AOI: uma ativacao pode cruzar fuso, e um numero so mentiria.
            "epsg": epsgs, "aois": aois, "janela": janela}


def gravar(ativacao: str, dados: dict, terreno: Path,
           passo: float) -> tuple[Path, dict]:
    destino = RUNS / f"cems-emsr{ativacao.replace('EMSR', '').lower()}"
    destino.mkdir(parents=True, exist_ok=True)
    csv_path = destino / f"pontos_{ativacao.lower()}.csv"
    linhas = dados["linhas"]
    if linhas:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(linhas)

    por_classe = {}
    for l in linhas:
        por_classe[l["classe_obs"]] = por_classe.get(l["classe_obs"], 0) + 1
    sobre_canal = sum(1 for l in linhas
                      if l["classe_obs"] == "NAO_INUNDADO_OBSERVADO"
                      and l["sobre_canal_modelado"])
    tem_mascara = (terreno / AGUA_PERMANENTE).exists()
    reclassificados = {}
    for l in linhas:
        if l["classe_obs"] == "AGUA_PERMANENTE":
            k = l["classe_do_poligono"]
            reclassificados[k] = reclassificados.get(k, 0) + 1
    # As classes que TREINAM sao 0 e 1; a 9 e carregada e excluida pelo ds05.
    treinaveis = {k: v for k, v in por_classe.items()
                  if k in ("INUNDACAO", "NAO_INUNDADO_OBSERVADO")}
    minoria = min(treinaveis.values()) if treinaveis else 0
    manifesto = {
        "tipo": "pontos_observados_de_ativacao_cems",
        "quando_utc": agora_utc(),
        "ativacao": ativacao,
        "crs_metrico": dados["epsg"],
        "passo_da_amostragem_m": passo,
        "amostragem": ("espacamento fixo no centro da celula da grade de 30 m "
                       "do proprio raster derivado; deterministico, sem semente"),
        "aois": dados["aois"],
        "grupos": len(dados["aois"]),
        "campanha": ([str(dados["janela"][0]), str(dados["janela"][1])]
                     if dados.get("janela") else None),
        "por_que_a_campanha": (
            "uma ativacao pode cobrir mais de um evento, e a janela restringe "
            "aos produtos desenhados dentro dela. ATENCAO ao motivo antigo "
            "registrado aqui -- 'os DEL do EMSR720 sao de 26/06 a 15/07, a "
            "segunda cheia do RS' -- ele estava errado: aquelas datas eram o "
            "dateStamp do metadado, e as cenas pos-evento dos mesmos produtos "
            "sao de 04-05/05/2024. Ver `data_do_produto`"),
        "grupo_e": ("a AOI, nao a ativacao: e onde o analista desenhou, e "
                    "agrupar por ativacao deu EPV 1,67 no projeto"),
        "pontos": len(linhas),
        "por_classe": por_classe,
        "epv_com_4_variaveis": round(minoria / 4, 1) if minoria else 0,
        "por_aoi": dados["por_aoi"],
        "camadas_ilegiveis": dados["ilegiveis"],
        "aois_sem_terreno": dados.get("aois_sem_terreno", []),
        "terreno_resolvido_por": (
            "AOI (`ter-01/<ativacao>_<aoi>`) quando o caminho recebido e a raiz; "
            "diretorio unico quando ele tem run_manifest.json"),
        "pacote_incompleto": bool(dados["ilegiveis"]),
        "terreno": {
            "diretorio": str(terreno),
            "cadeia": json.loads(
                (terreno / "run_manifest.json").read_text(encoding="utf-8"))
            if (terreno / "run_manifest.json").exists() else
            {"raiz": str(terreno),
             "nota": "terreno resolvido por AOI; ver aois_sem_terreno"},
        },
        "agua_permanente": {
            "separada": tem_mascara,
            "fonte": ("mascara JRC gerada por `agua_permanente_jrc`, com o "
                      "corte calibrado contra o `JRCWaterHand` do "
                      "Sen1Floods11" if tem_mascara else None),
            "por_que_precisou": (
                "o produto do EMS traz so `Riverine flood` e `Landslide` em "
                "`obj_desc`: nao ha agua permanente delineada, ao contrario do "
                "pacote do EMSR734, onde ela veio de camada externa"),
            "consequencia_se_nao_separar": (
                "o leito do rio cai no NEGATIVO com TWI altissimo e INVERTE o "
                "coeficiente da variavel -- medido aqui: +0,4408 para -0,1727"),
            "regra": ("sai das DUAS classes, como o deslizamento: nao e "
                      "positivo porque a agua ja estava la, e nao e negativo "
                      "util porque e agua. Mesma regra do `baseline_otsu`, "
                      "`valido & (jrc == 0)`"),
            "reclassificados": reclassificados,
            "canal_modelado_como_conferencia": (
                f"`sobre_canal_modelado` marca {sobre_canal} negativos sobre a "
                "rede de drenagem derivada; ele NAO define classe -- usar "
                "modelo para rotular e o erro que o projeto persegue -- e "
                "serve de conferencia independente da mascara observada"),
        },
        "definicao": {
            "observado": "areaOfInterest INTERSECAO imageFootprint",
            "positivo": f"observado INTERSECAO observedEvent[{TIPO_INUNDACAO}]",
            "negativo": f"observado MENOS positivo MENOS [{TIPO_DESLIZAMENTO}]",
            "por_que_o_footprint": ("area da AOI que a imagem nao cobriu nao "
                                    "foi observada e nao pode virar negativo"),
        },
        "status": ("CANDIDATO. Escreve o CSV que o ds01 le e nada mais: "
                   "nenhum rotulo promovido, nenhum modelo treinado, gate C4 "
                   "inalterado"),
        "nao_e": ("nao e rotulo de referencia e nao autoriza claim preditivo: "
                  "e a amostra de pontos observados de uma ativacao"),
        "arquivo": str(csv_path),
    }
    manifesto["hash_resumo"] = sha256_json(
        {k: v for k, v in manifesto.items() if k != "quando_utc"})
    (destino / "resumo_pontos.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, manifesto


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ativacao", required=True)
    p.add_argument("--extraido", type=Path, required=True)
    p.add_argument("--terreno", type=Path, required=True)
    p.add_argument("--passo", type=float, default=PASSO_M)
    p.add_argument("--campanha", nargs=2, metavar=("INICIO", "FIM"),
                   help="AAAA-MM-DD AAAA-MM-DD; so produtos desenhados nessa "
                        "janela entram, para que rotulo e radar sejam do mesmo "
                        "evento")
    a = p.parse_args()

    janela = None
    if a.campanha:
        from datetime import date
        janela = (date.fromisoformat(a.campanha[0]),
                  date.fromisoformat(a.campanha[1]))
    dados = construir(a.ativacao, a.extraido, a.terreno, a.passo, janela)
    csv_path, manifesto = gravar(a.ativacao, dados, a.terreno, a.passo)
    print(json.dumps({k: v for k, v in manifesto.items()
                      if k not in ("por_aoi", "terreno")},
                     indent=2, ensure_ascii=False))
    print(f"\nGRAVADO={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
