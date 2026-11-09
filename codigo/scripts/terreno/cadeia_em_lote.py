"""A cadeia canonica sobre uma LISTA de janelas, e nao sobre uma AOI por vez.

-------------------------------------------------------------------------
POR QUE ESTE ARQUIVO EXISTE
-------------------------------------------------------------------------

O `ter-01` tem 786 diretorios e o repositorio nao tem quem os tenha produzido
em lote: `derivar_aoi_nova` faz UMA janela por chamada, e a varredura que
gerou os 786 foi ad-hoc. O mesmo padrao que o `baixar_fabdem` foi escrito para
encerrar -- "em Petropolis quem barrou foi decisao humana, e nao existia
script, foi ad-hoc" -- estava de pe aqui.

Ele passou a fazer falta por dois motivos ao mesmo tempo:

  * o `gfd01` entregou 149.179 pontos brasileiros de negativo OBSERVADO, e o
    `ds05` os recusa por `cadeia_global_nao_comparavel` enquanto nao houver
    terreno da cadeia canonica sob eles;
  * o `EMSR702` esta baixado, com data resolvida, e sem nenhuma AOI derivada.

-------------------------------------------------------------------------
O CUSTO MEDIDO, e ele nao e o que eu supus
-------------------------------------------------------------------------

Eu tinha registrado que derivar 838 mil km2 a 30 m era decisao de custo
proibitiva. Medido, e duas vezes, porque a primeira medicao tambem enganava.

Os 786 diretorios do `ter-01` dao 246.298.777 celulas em 0,34 h -> 203.150
celulas/s, e o maior deles, `EMSR778_AOI01`, tem 28.468.088 celulas. Um tile de
1 grau a 30 m tem ~13 milhoes: cabe com folga, e ate aqui a conclusao de que o
custo nao e proibitivo esta certa.

Mas 203.150 celulas/s e o `elapsed_s` do `run_manifest`, e ele cronometra SO a
cadeia do WhiteboxTools. O primeiro tile deste lote foi cronometrado inteiro:

    gfd_tile_S22W053   13.018.644 celulas   3726 x 3494   EPSG:32722
        cadeia do WhiteboxTools      23,1 s
        janela FABDEM + reprojecao  ~220,8 s
        total                        243,8 s

O custo e REDE. Estimar o lote por 203.150 celulas/s o subestima em 3,8x, e a
estimativa que eu anunciei para 25 tiles -- 0,63 h -- estava errada por isso.

-------------------------------------------------------------------------
A MARGEM DE BACIA, e por que ela nao e enfeite
-------------------------------------------------------------------------

Cortar a janela no tile trunca a bacia na borda. O `dinf_sca` de uma celula
proxima da borda soma so a area contribuinte DENTRO da janela, entao o canal
inicia tarde, a rede sai rala, e o `elevation_above_stream` -- o HAND -- sai
alto. Esse e o mecanismo que rebaixou Valparaiso (`susc_20h`: HAND 50,9 m com
o rio a 688 m) e o projeto ja pagou por ele.

A janela derivada e, por isso, MAIOR que o tile: `MARGEM_DE_BACIA_GRAUS` de
cada lado. O ponto so e amostravel no tile interno, e `gfd02` confere isso.

O que a margem NAO resolve, e fica declarado: rio de talvegue principal cuja
area contribuinte vem de centenas de quilometros acima continua truncado. Por
isso cada ponto amostrado leva a distancia dele para a borda da derivacao, e
quem for modelar pode cortar por ela em vez de confiar num numero unico.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao amostra ponto, nao decide rotulo, nao promove modelo e nao inventa
cadeia: ele chama `baixar_fabdem` e `derivar_aoi_nova`, que sao quem tem
contrato sobre DEM e sobre os 30 m / 0,1123 km2.

Nao reescreve diretorio existente sem `--refazer`: os 786 do `ter-01` ja
foram usados para medir.

REDE: o padrao e DRY RUN. Sem `--baixar` ele imprime o plano -- tiles, bytes,
celulas e horas estimadas -- e nao toca na rede nem no disco.

Uso:
    python scripts/terreno/cadeia_em_lote.py --tiles-gfd 1
    python scripts/terreno/cadeia_em_lote.py --tiles-gfd 25 --baixar
    python scripts/terreno/cadeia_em_lote.py --janelas minhas.json --baixar
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "terreno"))

from proveniencia import agora_utc, sha256_json  # noqa: E402

import baixar_fabdem as fab  # noqa: E402
import derivar_aoi_nova as der  # noqa: E402

RUNS = REPO / "local_runs"
PONTOS_GFD = RUNS / "gfd-brasil"
SAIDA = RUNS / "ter-03-lote-de-tiles"
TER01 = RUNS / "ter-01-cadeia-harmonizada"

# 0,1 grau ~ 11 km de contexto de bacia em cada lado. Escolhido contra o
# limiar de canal do projeto: 0,1123 km2 de area contribuinte inicia um canal,
# o que sao ~340 m de caminho de fluxo no pior caso, entao 11 km cobrem a rede
# local com margem larga. Nao cobre talvegue principal -- ver o cabecalho.
MARGEM_DE_BACIA_GRAUS = 0.1

# MEDIDO, e em duas parcelas, porque a soma engana.
#
# Os 786 diretorios do ter-01 dao 246.298.777 celulas em 0,34 h -> 203.150
# celulas/s, mas esse numero e o `elapsed_s` do `run_manifest`, que cronometra
# SO a cadeia do WhiteboxTools. O primeiro tile deste lote separou as parcelas:
#
#     gfd_tile_S22W053   13.018.644 celulas
#         cadeia do WhiteboxTools      23,1 s   (563.000 celulas/s)
#         janela FABDEM + reprojecao  ~220,8 s
#         total                        243,8 s
#
# Ou seja: o custo e REDE, e nao processamento. Estimar por 203.150 cel/s
# subestimava o lote em 3,8x -- e eu tinha anunciado 0,63 h para 25 tiles.
CELULAS_POR_SEGUNDO_WHITEBOX = 563_000
SEGUNDOS_POR_TILE_DE_REDE = 221.0

GRAU_FABDEM = 1.0 / 3600.0 * 1.0  # 1 arco-segundo
RESOLUCAO_FABDEM_GRAUS = 0.0002777777777777778


def _rotulo_do_tile(lat: int, lon: int) -> str:
    """Mesmo formato do FABDEM, para que o nome diga onde o raster esta."""
    return fab._rotulo(lat, lon)


def tiles_do_gfd(quantos: int) -> list[dict]:
    """Os tiles de 1 grau com mais ponto GFD dentro, do mais denso ao menos.

    A ordem e por densidade e nao por evento: um tile pode receber ponto de
    varios eventos, e derivar terreno duas vezes para o mesmo lugar seria a
    mesma conta com dois nomes -- o erro que o `dinf_sca`/`facc_dinf` do
    `derivar_aoi_nova` documenta.
    """
    import pandas as pd

    arquivos = sorted(PONTOS_GFD.glob("pontos_gfd*.csv"))
    if not arquivos:
        raise SystemExit(f"sem pontos GFD em {PONTOS_GFD}")
    d = pd.concat([pd.read_csv(p) for p in arquivos], ignore_index=True)
    d["tlat"] = d.lat.apply(math.floor).astype(int)
    d["tlon"] = d.lon.apply(math.floor).astype(int)
    contagem = (d.groupby(["tlat", "tlon"])
                .agg(pontos=("ponto_id", "size"),
                     eventos=("evento_gfd", "nunique"))
                .sort_values("pontos", ascending=False))
    saida = []
    for (lat, lon), linha in contagem.head(quantos).iterrows():
        saida.append({
            "nome": f"gfd_tile_{_rotulo_do_tile(lat, lon)}",
            "tile_1grau": [lon, lat, lon + 1, lat + 1],
            "pontos_gfd_no_tile": int(linha.pontos),
            "eventos_gfd_no_tile": int(linha.eventos),
        })
    return saida


def _fuso_utm(lon: float) -> int:
    """O fuso UTM que contem a longitude. Mesma conta do `derivar_aoi_nova`."""
    return int((lon + 180) // 6) + 1


def _distorcao_do_fuso(bbox: list[float]) -> dict:
    """Quanto a janela se afasta do meridiano central do fuso escolhido.

    `derivar_aoi_nova` reprojeta para UM fuso, escolhido pelo centro da janela.
    Uma janela de 1,2 grau pode cruzar a borda do fuso, e ai a ponta de fora
    fica a mais de 3 graus do meridiano central. Isso nao e defeito -- e o
    comportamento do UTM --, mas e ESCOLHA, e escolha silenciosa foi o que fez
    o projeto aceitar grade desalinhada do Earth Engine porque "parecia
    correta". Entao a conta sai medida no manifesto.
    """
    oeste, _, leste, _ = bbox
    fuso = _fuso_utm((oeste + leste) / 2)
    meridiano_central = -180 + (fuso - 1) * 6 + 3
    afastamento = max(abs(oeste - meridiano_central),
                      abs(leste - meridiano_central))
    # Erro de escala do UTM transverso: k ~ 1 + (x/R)^2 / 2, com x a distancia
    # ao meridiano central. Em metro por celula de 30 m.
    x_km = afastamento * 111.32
    erro_relativo = (x_km / 6371.0) ** 2 / 2
    return {
        "fuso": fuso,
        "cruza_borda_de_fuso": _fuso_utm(oeste) != _fuso_utm(leste),
        "graus_do_meridiano_central": round(afastamento, 3),
        "erro_de_escala_relativo": round(erro_relativo, 6),
        "erro_por_celula_de_30m_cm": round(erro_relativo * 30 * 100, 2),
    }


def _com_margem(tile: list[float], margem: float) -> list[float]:
    oeste, sul, leste, norte = tile
    return [round(oeste - margem, 6), round(sul - margem, 6),
            round(leste + margem, 6), round(norte + margem, 6)]


def planejar(janelas: list[dict], margem: float = MARGEM_DE_BACIA_GRAUS) -> dict:
    total_celulas = total_bytes = 0
    plano = []
    for j in janelas:
        bbox = _com_margem(j["tile_1grau"], margem)
        p = fab.planejar(*bbox, j["nome"])
        # Celulas DEPOIS da reprojecao para 30 m, que e o que o WhiteboxTools
        # processa -- e nao as celulas de 1 arco-segundo do FABDEM.
        lat_media = (bbox[1] + bbox[3]) / 2
        km_lat = (bbox[3] - bbox[1]) * 111.32
        km_lon = (bbox[2] - bbox[0]) * 111.32 * math.cos(math.radians(lat_media))
        celulas_30m = int((km_lat * 1000 / 30) * (km_lon * 1000 / 30))
        total_celulas += celulas_30m
        total_bytes += p["bytes_lidos_estimados"]
        plano.append({**j, "bbox_derivada": bbox, "n_tiles_fabdem": p["n_tiles"],
                      "bytes_fabdem_estimados": p["bytes_lidos_estimados"],
                      "celulas_a_30m": celulas_30m,
                      "utm": _distorcao_do_fuso(bbox),
                      "ja_existe": (TER01 / j["nome"] / "run_manifest.json").exists()})
    return {
        "tipo": "plano_de_cadeia_em_lote",
        "quando_utc": agora_utc(),
        "janelas": plano,
        "n_janelas": len(plano),
        "margem_de_bacia_graus": margem,
        "por_que_a_margem": (
            "cortar no tile trunca a area contribuinte na borda: o canal "
            "inicia tarde, a rede sai rala e o HAND sai alto -- o mecanismo "
            "que rebaixou Valparaiso. A janela derivada e maior que o tile, e "
            "o ponto so e amostravel no tile interno"),
        "o_que_a_margem_nao_resolve": (
            "talvegue principal com area contribuinte de centenas de km acima "
            "continua truncado; por isso cada ponto leva a distancia para a "
            "borda da derivacao"),
        "janelas_que_cruzam_borda_de_fuso": sum(
            1 for j in plano if j["utm"]["cruza_borda_de_fuso"]),
        "pior_erro_por_celula_de_30m_cm": max(
            (j["utm"]["erro_por_celula_de_30m_cm"] for j in plano), default=0.0),
        "por_que_isso_e_aceito": (
            "derivar_aoi_nova reprojeta para UM fuso, escolhido pelo centro; "
            "a janela de 1,2 grau pode cruzar a borda e a ponta de fora passa "
            "de 3 graus do meridiano central. O erro de escala resultante e "
            "medido acima, em centimetro por celula de 30 m, e fica no "
            "manifesto em vez de ser suposto pequeno"),
        "celulas_a_30m_total": total_celulas,
        "bytes_fabdem_estimados_total": total_bytes,
        "horas_estimadas": round(
            (total_celulas / CELULAS_POR_SEGUNDO_WHITEBOX
             + len(plano) * SEGUNDOS_POR_TILE_DE_REDE) / 3600, 2),
        "base_da_estimativa": (
            f"{CELULAS_POR_SEGUNDO_WHITEBOX:,} celulas/s na cadeia do "
            f"WhiteboxTools mais {SEGUNDOS_POR_TILE_DE_REDE:.0f} s por tile de "
            "janela FABDEM e reprojecao, medidos no gfd_tile_S22W053; a parcela "
            "que manda e a REDE"),
        "nao_e": (
            "nao amostra ponto, nao decide rotulo e nao promove modelo: chama "
            "baixar_fabdem e derivar_aoi_nova, que tem o contrato do DEM e "
            "dos 30 m / 0,1123 km2"),
    }


def executar(plano: dict, refazer: bool = False,
             maximo: int | None = None) -> dict:
    """`maximo` fecha o lote depois de N janelas DERIVADAS nesta execucao.

    A fila da noite tem centenas de janelas e leva dias. Enquanto ela roda, a
    trava de `executar_serializado` fica com ela, e nenhum outro trabalho pesado
    entra -- que e o oposto do que a trava existe para fazer. Com um teto por
    execucao a fila vira uma fatia: roda N, libera a vez, e a proxima chamada
    continua de onde parou, porque pular o que ja tem manifesto ja e o padrao.
    """
    SAIDA.mkdir(parents=True, exist_ok=True)
    feitos, falhas, pulados = [], [], []
    for j in plano["janelas"]:
        if maximo is not None and len(feitos) + len(falhas) >= maximo:
            print(f"  [teto] {maximo} janelas nesta execucao; o resto fica "
                  "para a proxima", flush=True)
            break
        nome = j["nome"]
        if j["ja_existe"] and not refazer:
            pulados.append({"nome": nome, "por_que": "run_manifest.json ja existe"})
            print(f"  [pula] {nome}: ja derivado", flush=True)
            continue
        inicio = time.time()
        try:
            p = fab.planejar(*j["bbox_derivada"], nome)
            baixado = fab.baixar(p)
            dem = Path(baixado["arquivo"])
            m = der.derivar(dem, nome)
            feitos.append({
                "nome": nome,
                "dem": str(dem),
                "dem_bytes": dem.stat().st_size,
                "shape": m.get("shape"),
                # `derivar` grava `shape`, e nao `n_total`: ler a chave que ele
                # nao escreve gravava `null` aqui e zerava o total do lote.
                "celulas": (m["shape"][0] * m["shape"][1]
                            if m.get("shape") else None),
                "segundos_do_whitebox": m.get("elapsed_s"),
                "crs": m.get("crs"),
                "pixel_size": m.get("pixel_size"),
                "stream_area_km2": m.get("stream_area_km2"),
                "wbt_version": m.get("wbt_version"),
                "segundos": round(time.time() - inicio, 1),
                "pontos_gfd_no_tile": j.get("pontos_gfd_no_tile"),
            })
            print(f"  [ok]  {nome}: {m.get('shape')} "
                  f"{m.get('pixel_size')} limiar {m.get('stream_area_km2')} "
                  f"em {feitos[-1]['segundos']}s", flush=True)
        except (Exception, SystemExit) as e:        # noqa: BLE001
            # `SystemExit` ENTRA AQUI DE PROPOSITO, e isso nao e descuido.
            #
            # `SystemExit` herda de `BaseException`, entao `except Exception`
            # nao o pega -- e o portao de geometria do `baixar_fabdem` recusa
            # justamente com `SystemExit`. Sem esta linha, uma janela recusada
            # no tile 17 mataria o lote e perderia os 8 seguintes, contra o
            # que este bloco declara fazer.
            #
            # Falha de UMA janela nao para o lote, e nao vira silencio: ela
            # entra no manifesto com o texto do erro, e o `--refazer` nao e
            # necessario para retomar porque `ja_existe` pula o que ficou
            # pronto.
            falhas.append({"nome": nome, "erro": f"{type(e).__name__}: {e}",
                           "segundos": round(time.time() - inicio, 1)})
            print(f"  [FALHA] {nome}: {type(e).__name__}: {e}",
                  flush=True)
    resumo = {
        "tipo": "cadeia_em_lote_executada",
        "quando_utc": agora_utc(),
        "margem_de_bacia_graus": plano["margem_de_bacia_graus"],
        "pedidos": plano["n_janelas"],
        "derivados": len(feitos),
        "pulados": len(pulados),
        "falhas": len(falhas),
        "celulas_derivadas": sum(f["celulas"] or 0 for f in feitos),
        "segundos_do_whitebox": round(
            sum(f["segundos_do_whitebox"] or 0 for f in feitos), 1),
        "segundos": round(sum(f["segundos"] for f in feitos), 1),
        "pontos_gfd_cobertos": sum(f.get("pontos_gfd_no_tile") or 0
                                   for f in feitos),
        "detalhe": feitos,
        "pulados_detalhe": pulados,
        "falhas_detalhe": falhas,
        "nao_e": plano["nao_e"],
    }
    resumo["hash_resumo"] = sha256_json(
        {k: v for k, v in resumo.items() if k != "quando_utc"})
    (SAIDA / "lote.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    return resumo


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tiles-gfd", type=int,
                   help="os N tiles de 1 grau com mais ponto GFD dentro")
    p.add_argument("--janelas", type=Path,
                   help="JSON com [{nome, tile_1grau:[O,S,L,N]}, ...]")
    p.add_argument("--margem", type=float, default=MARGEM_DE_BACIA_GRAUS)
    p.add_argument("--baixar", action="store_true",
                   help="toca a rede e o disco; sem ele so imprime o plano")
    p.add_argument("--refazer", action="store_true")
    p.add_argument("--maximo", type=int, default=None,
                   help="derive no maximo N janelas nesta execucao e libere a "
                        "vez; a proxima chamada continua de onde parou")
    a = p.parse_args()

    if a.tiles_gfd:
        janelas = tiles_do_gfd(a.tiles_gfd)
    elif a.janelas:
        janelas = json.loads(a.janelas.read_text(encoding="utf-8"))
    else:
        raise SystemExit("informe --tiles-gfd N ou --janelas arquivo.json")

    plano = planejar(janelas, a.margem)
    print(f"=== {plano['n_janelas']} janelas | margem "
          f"{plano['margem_de_bacia_graus']} grau | "
          f"{plano['celulas_a_30m_total']:,} celulas a 30 m | "
          f"{plano['bytes_fabdem_estimados_total'] / 1e6:,.0f} MB de FABDEM | "
          f"~{plano['horas_estimadas']} h (rede + WhiteboxTools)")
    print(f"    {plano['janelas_que_cruzam_borda_de_fuso']} janelas cruzam borda "
          f"de fuso UTM; pior erro de escala "
          f"{plano['pior_erro_por_celula_de_30m_cm']} cm por celula de 30 m")
    for j in plano["janelas"]:
        marca = "JA EXISTE" if j["ja_existe"] else "        "
        fuso = f"UTM{j['utm']['fuso']}{'*' if j['utm']['cruza_borda_de_fuso'] else ' '}"
        print(f"  {marca} {j['nome']:24} bbox {j['bbox_derivada']} "
              f"{fuso} {j['n_tiles_fabdem']} tile(s) "
              f"{j['celulas_a_30m']:>12,} cel "
              f"{j.get('pontos_gfd_no_tile', 0):>6} pts GFD")

    if not a.baixar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        (SAIDA / "plano.json").write_text(
            json.dumps(plano, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nDRY RUN. Plano em {SAIDA / 'plano.json'}. "
              "Use --baixar para executar.")
        return 0

    r = executar(plano, a.refazer, a.maximo)
    print(f"\nderivados={r['derivados']} pulados={r['pulados']} "
          f"falhas={r['falhas']} | {r['celulas_derivadas']:,} celulas em "
          f"{r['segundos']}s | {r['pontos_gfd_cobertos']:,} pontos GFD cobertos")
    print(f"GRAVADO={SAIDA / 'lote.json'}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
