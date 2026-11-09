"""FABDEM para uma AOI: o MDT que o projeto aceita, e so a janela dela.

-------------------------------------------------------------------------
POR QUE FABDEM E NAO O QUE O PLANO DESCOBRE SOZINHO
-------------------------------------------------------------------------

`aoi_pipeline.planejar` descobre `cop-dem-glo-30`, e o proprio alerta dele diz:

    "produto e DSM (inclui dossel e edificacao), nao MDT. HAND derivado dele
     carrega o offset da altura da cobertura, que nao e terreno"

E `bloqueia_por_politica` sai FALSO. Ou seja: a politica avisa e nao impede.
Em Petropolis quem barrou foi decisao humana, que trocou por FABDEM -- e nao
existia script, foi ad-hoc. Este arquivo existe para que a proxima regiao nao
dependa de alguem lembrar.

FABDEM e o Copernicus com floresta e edificacao removidas (Hawker et al. 2022),
e `aoi_pipeline.classificar_dem("fabdem")` ja o classifica como DTM.

-------------------------------------------------------------------------
SO A JANELA, E POR QUE ISSO IMPORTA
-------------------------------------------------------------------------

O FABDEM e distribuido em zips de 10x10 graus, cada um com centenas de tiles de
1x1. Baixar o zip para usar 257 km2 seria trazer gigabytes para ler megabytes.

A leitura usa `/vsizip//vsicurl/`, que abre o tile DENTRO do zip remoto por
range request e le apenas a janela pedida -- o mesmo caminho que resolveu
Petropolis. `bytes_lidos_estimados` no manifesto registra o tamanho da janela,
nao o do zip.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao reprojeta e nao deriva terreno. O DEM sai em EPSG:4326, nativo, porque
quem tem contrato sobre reprojecao e resolucao e `derivar_aoi_nova`: 30 m,
agregacao por MEDIA, fuso UTM da AOI. Duas reprojecoes em serie criariam um
raster que nenhum dos dois modulos declara.

Nao decide nada sobre rotulo, mecanismo ou promocao.

REDE: o padrao e DRY RUN. Sem `--baixar` ele imprime o plano -- tiles, zip,
URL e bytes -- e nao toca na rede.

Uso:
    python scripts/terreno/baixar_fabdem.py --aoi-parquet local_runs/br-01-cems-negativo/aoi.parquet --nome EMSR720_AOI01
    python scripts/terreno/baixar_fabdem.py --bbox -51.93 -29.33 -51.59 -29.02 --nome EMSR720_AOI01 --baixar
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from proveniencia import agora_utc, sha256_json  # noqa: E402

BASE = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn"
# 1 arco-segundo, a grade nativa do FABDEM.
RESOLUCAO_GRAUS = 1.0 / 3600.0
# Folga em CELULA para grudar a janela na grade sem ganhar uma celula extra
# quando a coordenada pedida ja cai exatamente na borda de uma.
GRUDE = 1e-6
SUFIXO_TILE = "_FABDEM_V1-2.tif"
SUFIXO_ZIP = "_FABDEM_V1-2.zip"
SAIDA = REPO / "local_runs" / "mdt-fabdem"

# -------------------------------------------------------------------------
# A MENSAGEM DO GDAL MENTE SOBRE A CAUSA, e isso custou 120 janelas.
#
# Medido em 14/09/2026: com DOIS lotes de derivacao rodando em paralelo contra
# `data.bris.ac.uk`, 96 de 100 janelas do lote brasileiro e 24 de 25 do lote
# tropical morreram com
#
#     RasterioIOError: '/vsizip//vsicurl/.../S05W043_FABDEM_V1-2.tif' does not
#     exist in the file system, and is not recognized as a supported dataset
#
# O tile EXISTE: listado no zip e aberto em seguida, ele devolve 3600x3600. E
# a mesma janela que falhou, reexecutada SOZINHA, derivou em 470 s sem erro.
#
# O que o `/vsicurl` faz quando a range request e estrangulada ou cai e
# reportar o arquivo como inexistente -- ausencia e indistinguivel de recusa
# na mensagem. Sem tentativa, uma falha de rede de um segundo joga fora vinte
# minutos de derivacao e o log diz que o dado nao existe.
TENTATIVAS_POR_TILE = 4
ESPERA_INICIAL_S = 20.0

# -------------------------------------------------------------------------
# E HA UMA TERCEIRA CAUSA, QUE A TENTATIVA SOZINHA NAO ALCANCA: A PENDURA.
#
# Medido em 15/09/2026: o lote parou em `gfd_tile_S32W055` as 03h41 e
# continuou VIVO, sem escrever uma linha, por sete horas. Nenhuma mensagem de
# tentativa apareceu -- porque nao houve erro. A leitura nao voltou.
#
# O `/vsicurl` nao tem timeout por padrao. Conexao que para de entregar bytes
# sem fechar deixa o `rasterio.open` esperando para sempre, e a repeticao de
# `_com_tentativas` so age sobre EXCECAO -- ela nunca chega. Erro e pendura sao
# falhas opostas: uma liga o tratamento, a outra o desliga. As variaveis abaixo
# convertem a segunda na primeira, e so entao as tentativas passam a valer.
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "300")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")


# -------------------------------------------------------------------------
# E HA UMA SEGUNDA CAUSA, QUE A PRIMEIRA CORRECAO NAO ALCANCA.
#
# Depois de por tentativas, quatro janelas continuaram morrendo -- sempre nos
# mesmos tiles. Listando o arquivo remoto em 14/09/2026:
#
#     S40W060-S30W050_FABDEM_V1-2.zip      59 tiles, S33W051 NAO esta la
#     S40W050-S30W040_FABDEM_V1-2.zip      o zip inteiro NAO existe
#
# O FABDEM nao publica tile sem terra. `S33W051` e a Lagoa dos Patos e o bloco
# S40W050-S30W040 e Atlantico aberto. A ausencia e REAL, e nenhuma tentativa a
# conserta.
#
# As duas causas dao a MESMA mensagem do GDAL, e por isso precisam ser
# separadas antes e nao depois: listar o zip responde "existe ou nao existe"
# sem depender de a leitura ter sido estrangulada.
#
# E tile de oceano ausente NAO e perda hidrologica: agua nao contribui area a
# montante. Truncar terra no talvegue rebaixa a rede e sobe o HAND -- foi o
# que rebaixou Valparaiso --, mas truncar mar nao tira nada de bacia nenhuma.
# Por isso o oceano vira lacuna DECLARADA e a janela segue, enquanto qualquer
# outra falha continua abortando.
AUSENCIA_NA_MENSAGEM = ("404", "does not exist", "no such file",
                        "not found", "cannot open")


def _falha_e_ausencia(e: Exception) -> bool:
    """A listagem falhou porque o zip NAO EXISTE, ou porque a rede caiu?

    As duas causas sao opostas e chegavam com a mesma cara -- conjunto vazio --,
    e a consequencia era o pior tipo de erro deste projeto: em 21/09/2026 nove
    janelas em pleno Mato Grosso foram registradas como "bloco sem terra:
    oceano" porque a listagem do zip remoto falhou por rede. O mesmo par de
    causas ja tinha mordido o projeto no GDAL (commit 7b586a5).
    """
    texto = f"{type(e).__name__}: {e}".lower()
    return any(marca in texto for marca in AUSENCIA_NA_MENSAGEM)


def _tiles_do_zip(url_zip: str) -> set:
    """O que o arquivo remoto REALMENTE contem.

    Conjunto vazio significa UMA coisa so: o zip nao existe (bloco de 10 graus
    sem terra). Falha de rede e repetida por `_com_tentativas` e, se insistir,
    sobe como excecao -- porque tratar rede instavel como oceano inventa
    ausencia de dado onde ha dado.
    """
    import fiona

    def listar():
        return set(fiona.listdir(url_zip))

    try:
        return _com_tentativas(listar, url_zip)
    except Exception as e:                                       # noqa: BLE001
        if _falha_e_ausencia(e):
            return set()
        raise


def _com_tentativas(fn, alvo: str, tentativas: int = TENTATIVAS_POR_TILE,
                    espera: float = ESPERA_INICIAL_S):
    """Repete `fn` com espera dobrando, e so entao desiste com o motivo."""
    ultimo = None
    for i in range(tentativas):
        try:
            return fn()
        except Exception as e:                                   # noqa: BLE001
            ultimo = e
            if i + 1 < tentativas:
                print(f"    tentativa {i + 1}/{tentativas} falhou em {alvo}: "
                      f"{type(e).__name__}; esperando "
                      f"{espera * (2 ** i):.0f}s", flush=True)
                time.sleep(espera * (2 ** i))
    raise RuntimeError(
        f"{alvo}: {tentativas} tentativas falharam. Ultimo erro: "
        f"{type(ultimo).__name__}: {str(ultimo)[:200]}") from ultimo


# -------------------------------------------------------------------------
# NAO EXISTE CACHE DE TILE AQUI, E ISSO FOI MEDIDO E NAO SUPOSTO.
#
# Em 15/09/2026 implementei um cache de tile inteiro, argumentando que
# `/vsizip//vsicurl` sobre entrada comprimida nao consegue saltar e portanto
# ler a janela custaria o mesmo que ler o tile. A medida derrubou o argumento,
# no mesmo tile e na mesma conexao:
#
#     abrir o tile remoto ..................... 2,2 s
#     ler uma janela de 600x600 .............. 24,4 s
#     ler o TILE INTEIRO de 3600x3600 ....... 380,0 s
#
# Os GeoTIFF do FABDEM sao LADRILHADOS em blocos de 256x256 com deflate POR
# BLOCO. O GDAL pede por range apenas os blocos que a janela toca, e ler o tile
# inteiro custa quinze vezes mais do que ler o pedaco de que se precisa.
#
# A "redundancia de 1,40x" que motivou o cache tambem estava mal lida: as 476
# leituras dos 341 tiles distintos sao leituras de JANELA, e janelas vizinhas
# tocam PARTES DIFERENTES do tile compartilhado. Cachear o tile inteiro
# gastaria 380 s para poupar duas ou tres leituras de 24 s.
#
# Fica registrado porque a intuicao e convidativa e esta errada: aqui, baixar
# menos e pedir menos, e nao guardar mais.


def _rotulo(lat: int, lon: int) -> str:
    """Canto sudoeste no formato do FABDEM: S30W052, N01E010."""
    ns = f"{'S' if lat < 0 else 'N'}{abs(lat):02d}"
    ew = f"{'W' if lon < 0 else 'E'}{abs(lon):03d}"
    return f"{ns}{ew}"


def tiles_da_bbox(oeste: float, sul: float, leste: float, norte: float) -> list:
    """Os tiles de 1x1 que cobrem a bbox, e o zip de 10x10 de cada um.

    O canto do tile e o PISO da coordenada, e o do zip e o piso da dezena --
    e por isso um tile em -29,33 mora no tile S30 e no zip S30W060, mesmo o
    numero "29" nao aparecendo em lugar nenhum do nome.
    """
    saida = []
    for lat in range(math.floor(sul), math.floor(norte) + 1):
        for lon in range(math.floor(oeste), math.floor(leste) + 1):
            zip_lat = math.floor(lat / 10) * 10
            zip_lon = math.floor(lon / 10) * 10
            nome_zip = (f"{_rotulo(zip_lat, zip_lon)}-"
                        f"{_rotulo(zip_lat + 10, zip_lon + 10)}{SUFIXO_ZIP}")
            nome_tile = f"{_rotulo(lat, lon)}{SUFIXO_TILE}"
            saida.append({
                "tile": nome_tile,
                "zip": nome_zip,
                "bbox_do_tile": [lon, lat, lon + 1, lat + 1],
                "url": f"/vsizip//vsicurl/{BASE}/{nome_zip}/{nome_tile}",
            })
    return saida


def bbox_do_parquet(caminho: Path) -> tuple:
    """Bbox em EPSG:4326 a partir de um parquet de geometria do projeto."""
    import pyarrow.parquet as pq
    from pyproj import Transformer
    from shapely import from_wkb

    tabela = pq.read_table(caminho)
    geom = from_wkb(tabela.column("geometry")[0].as_py())
    x0, y0, x1, y1 = geom.bounds
    # Os parquets areais do projeto sao gravados em CRS metrico (o EMSR720 em
    # 31982) e nao carregam a declaracao no arquivo. Coordenada em centenas de
    # milhares e metro, nao grau -- e converter na duvida evitaria o erro
    # silencioso de pedir um tile do meio do Atlantico.
    if max(abs(x0), abs(y0)) > 180:
        crs = _crs_do_parquet(caminho)
        tr = Transformer.from_crs(crs, 4326, always_xy=True)
        x0, y0 = tr.transform(x0, y0)
        x1, y1 = tr.transform(x1, y1)
    return x0, y0, x1, y1


def _crs_do_parquet(caminho: Path) -> int:
    """O CRS declarado ao lado do parquet, ou erro.

    O `resumo.json` do pacote areal grava `crs_metrico`. Adivinhar o fuso a
    partir da coordenada funcionaria na maioria dos casos e falharia em
    silencio nos outros, e um DEM da regiao errada derivaria terreno inteiro
    sem que nada denunciasse.
    """
    resumo = caminho.parent / "resumo.json"
    if resumo.exists():
        d = json.loads(resumo.read_text(encoding="utf-8"))
        if d.get("crs_metrico"):
            return int(d["crs_metrico"])
    raise SystemExit(
        f"CRS do parquet nao declarado: esperado `crs_metrico` em {resumo}")


def planejar(oeste: float, sul: float, leste: float, norte: float,
             nome: str) -> dict:
    tiles = tiles_da_bbox(oeste, sul, leste, norte)
    # 1 grau de FABDEM e 3600x3600 a ~0,000278 grau. A janela e a fracao da
    # bbox dentro do tile, e cada celula sao 4 bytes.
    graus_lat = norte - sul
    graus_lon = leste - oeste
    celulas = (graus_lat / RESOLUCAO_GRAUS) * (graus_lon / RESOLUCAO_GRAUS)
    return {
        "tipo": "plano_de_fabdem",
        "quando_utc": agora_utc(),
        "nome": nome,
        "bbox_4326": [round(oeste, 6), round(sul, 6),
                      round(leste, 6), round(norte, 6)],
        "produto": "fabdem",
        "tipo_dem": "DTM",
        "por_que_este_produto": (
            "o GLO-30 que o aoi_pipeline descobre e DSM, e a politica do "
            "projeto avisa sem bloquear; FABDEM e o Copernicus com floresta e "
            "edificacao removidas, e ja e classificado como DTM"),
        "tiles": tiles,
        "n_tiles": len(tiles),
        "celulas_da_janela_estimadas": int(celulas),
        "bytes_lidos_estimados": int(celulas * 4),
        "leitura": "janela por range request dentro do zip remoto, via "
                   "/vsizip//vsicurl/ -- o zip inteiro nao e baixado",
        "nao_e": ("nao reprojeta, nao deriva terreno e nao decide rotulo: o "
                  "DEM sai nativo em EPSG:4326 e quem tem contrato sobre 30 m "
                  "e agregacao e derivar_aoi_nova"),
    }


def baixar(plano: dict, saida: Path = SAIDA) -> dict:
    """Le a janela de cada tile e grava um DEM 4326 unico."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.windows import Window, from_bounds

    oeste, sul, leste, norte = plano["bbox_4326"]
    saida.mkdir(parents=True, exist_ok=True)
    partes, lidos, sem_terra = [], [], []
    catalogo: dict = {}
    for t in plano["tiles"]:
        # O NOME DO TILE VEM DO PLANO, e nao de partir a URL.
        #
        # `t["url"].rsplit("/", 1)` quebrava com `ValueError: not enough
        # values to unpack` sempre que a URL nao tinha barra -- que e o caso
        # dos planos montados a mao pelos testes, com caminho local. Cinco
        # testes de `test_janela_do_fabdem` reprovaram por isso desde o
        # conserto do oceano, e eu so os vi hoje porque nao rodei a suite
        # depois daquele conserto.
        #
        # O plano ja carrega `tile` e `zip` declarados por `tiles_da_bbox`;
        # usa-los remove a analise sintatica do meio do caminho.
        nome_tile = t["tile"]
        sufixo = f"/{nome_tile}"
        url_zip = (t["url"][:-len(sufixo)] if t["url"].endswith(sufixo)
                   else None)
        if url_zip is None:
            # Caminho que nao aponta para dentro de um arquivo remoto: nao ha
            # listagem a consultar, e a ausencia real aparece na leitura.
            catalogo[nome_tile] = {nome_tile}
            url_zip = nome_tile
        elif url_zip not in catalogo:
            catalogo[url_zip] = _tiles_do_zip(url_zip)
        if nome_tile not in catalogo[url_zip]:
            sem_terra.append({"tile": t["tile"],
                              "motivo": ("nao listado no arquivo FABDEM: "
                                         "bloco sem terra")})
            lidos.append({"tile": t["tile"], "shape": [0, 0],
                          "pulado": "sem cobertura FABDEM (oceano)"})
            continue
        with _com_tentativas(lambda u=t["url"]: rasterio.open(u),
                             t["tile"]) as src:
            # A JANELA E RECORTADA NA INTERSECAO, e nao pedida crua.
            #
            # `from_bounds` com uma bbox que sai do tile devolve janela com
            # offset NEGATIVO. O `read` clipa em silencio -- vem menos dado --
            # mas `window_transform` usa a origem NAO clipada, entao a parte e
            # declarada 0,1 grau fora do lugar e o `merge` empilha os tiles
            # deslocados. O resultado sai com 100% de celula valida e aparencia
            # correta: pedido de 1,2 x 1,2 grau virou 1,0 x 1,0 deslocado.
            #
            # Nao aparecia porque toda AOI usada ate aqui cabia DENTRO de um
            # tile so. A margem de bacia do `cadeia_em_lote` foi o primeiro
            # pedido a cruzar borda de tile.
            b, inverso = src.bounds, ~src.transform
            recorte = (max(oeste, b.left), max(sul, b.bottom),
                       min(leste, b.right), min(norte, b.top))
            if recorte[0] >= recorte[2] or recorte[1] >= recorte[3]:
                lidos.append({"tile": t["tile"], "shape": [0, 0],
                              "bounds_do_tile": list(b),
                              "pulado": "sem intersecao com a bbox pedida"})
                continue
            # A JANELA E GRUDADA NA GRADE DA FONTE, PARA FORA.
            #
            # `from_bounds` devolve janela FRACIONARIA e o `read` a arredonda
            # por conta dele -- e nao sempre para fora. Um pedido desalinhado
            # meia celula saiu uma celula CURTO no leste, o que e perda de
            # dado silenciosa. Arredondar aqui, com piso no canto noroeste e
            # teto no sudeste, faz o mosaico CONTER o pedido sempre; e o que
            # permite ao portao abaixo cobrar cobertura em vez de tolerancia.
            c0, r0 = inverso * (recorte[0], recorte[3])
            c1, r1 = inverso * (recorte[2], recorte[1])
            coluna = max(0, math.floor(c0 + GRUDE))
            linha = max(0, math.floor(r0 + GRUDE))
            janela = Window(
                coluna, linha,
                min(src.width, math.ceil(c1 - GRUDE)) - coluna,
                min(src.height, math.ceil(r1 - GRUDE)) - linha)
            dados = _com_tentativas(
                lambda s=src, w=janela: s.read(1, window=w), t["tile"])
            perfil = src.profile.copy()
            perfil.update(height=dados.shape[0], width=dados.shape[1],
                          transform=src.window_transform(janela),
                          compress="deflate")
            parte = saida / f"_{plano['nome']}_{t['tile']}"
            with rasterio.open(parte, "w", **perfil) as fora:
                fora.write(dados, 1)
            partes.append(parte)
            lidos.append({"tile": t["tile"], "shape": list(dados.shape),
                          "nodata": src.nodata,
                          "bounds_do_tile": list(b),
                          "bounds_do_recorte": [round(x, 6) for x in recorte]})
    if not partes:
        # O motivo VIAJA na mensagem: "nenhum tile intersecta" e "todos os
        # tiles falharam na leitura" pedem acoes opostas -- a primeira encerra
        # a janela, a segunda pede repetir quando a rede voltar.
        pulados = [l.get("pulado") for l in lidos if l.get("pulado")]
        raise SystemExit(
            f"{plano['nome']}: {len(plano['tiles'])} tiles no plano e nenhuma "
            f"parte lida (motivos: {sorted(set(pulados)) or 'nenhum declarado'}) "
            f"-- nenhum tile do FABDEM intersecta "
            f"{plano['bbox_4326']}")

    destino = saida / f"{plano['nome']}_fabdem_4326.tif"
    if len(partes) == 1:
        partes[0].replace(destino)
    else:
        arrays = [rasterio.open(p) for p in partes]
        mosaico, transformacao = merge(arrays)
        perfil = arrays[0].profile.copy()
        perfil.update(height=mosaico.shape[1], width=mosaico.shape[2],
                      transform=transformacao, compress="deflate")
        for a in arrays:
            a.close()
        with rasterio.open(destino, "w", **perfil) as fora:
            fora.write(mosaico[0], 1)
        for p in partes:
            p.unlink()

    with rasterio.open(destino) as src:
        v = src.read(1).astype("float64")
        valido = v != (src.nodata if src.nodata is not None else -9999.0)
        estat = {
            "crs": str(src.crs), "shape": [src.height, src.width],
            "res_graus": [abs(src.transform.a), abs(src.transform.e)],
            "bounds": list(src.bounds),
            "nodata": src.nodata,
            "n_validas": int(valido.sum()),
            "n_total": int(v.size),
            "elev_min": round(float(v[valido].min()), 4) if valido.any() else None,
            "elev_max": round(float(v[valido].max()), 4) if valido.any() else None,
            "elev_media": round(float(v[valido].mean()), 4) if valido.any() else None,
        }
    # PORTAO DE GEOMETRIA, e ele existe porque a falha acima passou.
    #
    # Um DEM no lugar errado deriva terreno inteiro sem que nada denuncie, e o
    # projeto ja pagou por isso duas vezes: o desalinhamento de grade do Earth
    # Engine que "parecia correto", e o FABDEM deslocado que este portao pega.
    #
    # O criterio e ASSIMETRICO, e a primeira corrida real ensinou por que.
    # `read(window=...)` arredonda a janela PARA FORA, entao o mosaico sai ate
    # uma celula MAIOR que o pedido. Com tolerancia simetrica de meia celula o
    # portao reprovou um mosaico correto por 0,000278 grau no sul -- que e
    # exatamente 1 arco-segundo, a propria grade do FABDEM.
    #
    #   FALTAR qualquer coisa   -> recusa: o pedido nao foi coberto
    #   SOBRAR ate 1,5 celula   -> aceita: e o arredondamento da janela
    #   SOBRAR mais que isso    -> recusa: mosaico maior que o pedido
    # O PEDIDO ENCOLHE ATE A TERRA QUE EXISTE, e o quanto fica declarado.
    #
    # Sem isto o portao de geometria recusaria toda janela costeira por
    # "faltam celulas no sul", que e verdade e nao e defeito: falta oceano.
    # Com isto ele continua cobrando que o mosaico cubra a TERRA pedida.
    if sem_terra:
        usados = [x for x in lidos if "bounds_do_recorte" in x]
        if usados:
            oeste = max(oeste, min(x["bounds_do_recorte"][0] for x in usados))
            sul = max(sul, min(x["bounds_do_recorte"][1] for x in usados))
            leste = min(leste, max(x["bounds_do_recorte"][2] for x in usados))
            norte = min(norte, max(x["bounds_do_recorte"][3] for x in usados))
    pedido = [oeste, sul, leste, norte]
    obtido = estat["bounds"]
    # Positivo = furo naquele lado; negativo = sobra.
    falta = {"oeste": obtido[0] - oeste, "sul": obtido[1] - sul,
             "leste": leste - obtido[2], "norte": norte - obtido[3]}
    epsilon = RESOLUCAO_GRAUS * 1e-3
    problemas = []
    for lado, f in falta.items():
        if f > epsilon:
            problemas.append(f"{lado}: FALTAM {f / RESOLUCAO_GRAUS:.2f} celulas")
        elif -f > RESOLUCAO_GRAUS * 1.5:
            problemas.append(f"{lado}: SOBRAM {-f / RESOLUCAO_GRAUS:.2f} celulas")
    if problemas:
        raise SystemExit(
            f"{plano['nome']}: mosaico do FABDEM fora do pedido "
            f"{[round(x, 6) for x in pedido]} -> "
            f"{[round(x, 6) for x in obtido]}: " + "; ".join(problemas))

    plano = dict(plano)
    plano.update({"estado": "baixado", "arquivo": str(destino),
                  "tiles_lidos": lidos, "raster": estat,
                  "tiles_sem_cobertura_fabdem": sem_terra,
                  "por_que_oceano_nao_aborta": (
                      "o FABDEM nao publica tile sem terra; agua nao "
                      "contribui area a montante, entao a lacuna e declarada "
                      "e a janela segue. Truncar TERRA continua abortando"),
                  "conferencia_de_geometria": {
                      "bbox_pedida": [round(x, 6) for x in pedido],
                      "bounds_do_mosaico": [round(x, 6) for x in obtido],
                      "celulas_faltando_por_lado": {
                          k: round(v / RESOLUCAO_GRAUS, 3)
                          for k, v in falta.items()},
                      "criterio": ("faltar qualquer coisa recusa; sobrar ate "
                                   "1,5 celula e o arredondamento da janela"),
                      "por_que": (
                          "DEM no lugar errado deriva terreno inteiro sem "
                          "denunciar; a janela recortada na intersecao ja "
                          "conserta a causa, e este portao impede a "
                          "reincidencia")}})
    plano["hash_resumo"] = sha256_json(
        {k: v for k, v in plano.items() if k != "quando_utc"})
    (saida / f"{plano['nome']}_fabdem_manifesto.json").write_text(
        json.dumps(plano, indent=2, ensure_ascii=False), encoding="utf-8")
    return plano


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nome", required=True)
    p.add_argument("--bbox", type=float, nargs=4,
                   metavar=("OESTE", "SUL", "LESTE", "NORTE"))
    p.add_argument("--aoi-parquet", type=Path)
    p.add_argument("--baixar", action="store_true",
                   help="toca a rede; sem ele o modulo so imprime o plano")
    a = p.parse_args()

    if a.aoi_parquet:
        oeste, sul, leste, norte = bbox_do_parquet(a.aoi_parquet)
    elif a.bbox:
        oeste, sul, leste, norte = a.bbox
    else:
        raise SystemExit("informe --bbox ou --aoi-parquet")

    plano = planejar(oeste, sul, leste, norte, a.nome)
    if not a.baixar:
        plano["estado"] = "dry_run"
        plano["baixou_algo"] = False
        print(json.dumps(plano, indent=2, ensure_ascii=False))
        print("\nDRY RUN: nada foi baixado. Rode com --baixar para tocar a rede.")
        return 0

    print(f"lendo {plano['n_tiles']} tile(s) de FABDEM, janela de "
          f"~{plano['bytes_lidos_estimados']/1e6:.1f} MB...", file=sys.stderr)
    feito = baixar(plano)
    print(json.dumps(feito, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
