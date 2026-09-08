"""Chicago 311: a segunda cidade, para testar se a lei do TWI se repete.

POR QUE ESTA CIDADE, E POR QUE AGORA:

Nova York ensinou que, dentro do mesmo dia e do mesmo lugar, o indice topografico
de umidade separa o chamado de alagamento do chamado de outro assunto, e que o
efeito CRESCE com a chuva (`twi_dinf x chuva` = +0,0397, IC95 [+0,0221; +0,0560]).
O HAND deu nulo nas tres medidas.

Uma lei fisica se repete em outra cidade. Um artefato de Nova York nao. Enquanto
o resultado existir num lugar so, transferi-lo para Recife seria trocar fisica
por geografia -- que e exatamente o erro que este projeto ja matou uma vez.

Chicago serve ao teste por quatro motivos, todos verificados antes de escrever
este arquivo:

  1. tem os DOIS analogos dos tipos de Nova York -- `Water On Street Complaint`
     (superficie, como `street flooding`) e `Water in Basement Complaint`
     (subsuperficie, como `sewer backup`). Isso permite testar tambem a ORDEM:
     em Nova York a via respondeu 1,96 e o refluxo 1,40, que e a ordem que a
     fisica manda;
  2. 150.523 chamados positivos COM coordenada, 99,8% de cobertura, de 2018-12 a
     hoje -- serie mais longa que a de Nova York;
  3. esta na cobertura do MRMS, entao a chuva vem a 1 km como la;
  4. e plana. Isso NAO e defeito: e onde a convergencia decide o encharcamento,
     porque sem declive nao ha para onde a agua ir. Se o TWI so funcionasse com
     relevo, ele nao serviria para a cidade brasileira que mais interessa.

-------------------------------------------------------------------------
A ARMADILHA DO SP156, QUE AQUI ESTA DECLARADA
-------------------------------------------------------------------------

Sao Paulo caiu de servivel para 0,4753 -- abaixo do acaso -- quando o leitor de
rotulo foi corrigido: o que estava sendo contado como "hidrico" era limpeza de
bueiro e isencao de IPTU. Chicago tem exatamente a mesma armadilha, e ela e
grande:

    Sewer Cleaning Inspection Request      27.980   manutencao, nao ocorrencia
    Sewer Cave-In Inspection Request       13.043   falha de infraestrutura
    Alley Sewer Inspection Request          4.104   manutencao
    No Water Complaint                     11.997   abastecimento, nao chuva
    Low Water Pressure Complaint            5.626   abastecimento
    Water Lead Test Kit/Visit Request      47.900   qualidade da agua potavel
    Water Quality Concern                   1.981   qualidade da agua potavel

Esses tipos falam de agua e NAO sao ocorrencia de alagamento. Eles saem das duas
classes -- nao viram positivo, e tambem nao viram fundo. Deixa-los no fundo seria
pior que ignora-los: seriam chamados cuja localizacao depende da rede de agua e
esgoto, e nao do processo geral de reclamar.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao deriva terreno, nao junta com chuva, nao promove fonte e nao vira variavel.
Baixa, registra sha256 e declara. Promover e trabalho do `ds01`, com os
criterios de sempre.

Uso:
    python scripts/externo/aq_chicago01_chamados.py --amostra
    python scripts/externo/aq_chicago01_chamados.py --completo
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "local_runs" / "aq-chicago-01"

BASE = "https://data.cityofchicago.org/resource/v6vf-nfxy.json"
AGENTE = "REV-P/1.0 (pesquisa academica de suscetibilidade a inundacao)"

# Mesma janela do estudo de Nova York, para que a comparacao entre as duas
# cidades nao carregue diferenca de periodo junto com diferenca de lugar.
INICIO = "2020-01-01T00:00:00"

# Ocorrencia de agua na superficie ou no subsolo do imovel. Sao os dois, e nao
# mais: cada um tem um analogo direto em Nova York.
TIPOS_POSITIVOS = (
    "Water On Street Complaint",      # analogo de `street flooding`
    "Water in Basement Complaint",    # analogo de `sewer backup`
)

# Falam de agua e nao sao ocorrencia. Saem das DUAS classes. Ver o docstring.
TIPOS_EXCLUIDOS = (
    "Sewer Cleaning Inspection Request",
    "Sewer Cave-In Inspection Request",
    "Alley Sewer Inspection Request",
    "No Water Complaint",
    "Low Water Pressure Complaint",
    "Water Lead Test Kit Request",
    "Water Lead Test Visit Request",
    "Water Quality Concern",
    # NAO TEM LUGAR, e por isso nao pode ser fundo. Medido: destes dois tipos,
    # 194.625 chamados caem numa unica coordenada -- 41,87183 / -87,67985, o
    # endereco da propria central -- e 46.907 caem no aeroporto. Juntos com uma
    # terceira copia da central, eram 70,5% do fundo em TRES pontos.
    #
    # O efeito medido foi grave: as tres variaveis de terreno inverteram de
    # sinal ao mesmo tempo, porque a comparacao deixou de ser "endereco alagavel
    # contra endereco nao alagavel" e virou "endereco contra a central de
    # atendimento". Em Nova York as tres maiores coordenadas do fundo somam
    # 2,69%, e por isso o mesmo desenho funcionou la.
    "311 INFORMATION ONLY CALL",
    "Aircraft Noise Complaint",
)

# Rede de seguranca, para o caso de outro tipo com o mesmo defeito aparecer sem
# que alguem perceba: coordenada que sozinha carrega mais que esta fracao do
# fundo e centroide, nao endereco. O pior caso de Nova York e 1,41%.
FRACAO_MAXIMA_POR_COORDENADA = 0.02

COLUNAS = ("sr_number", "sr_type", "created_date", "latitude", "longitude")
PAGINA = 50000
INTERVALO_S = 0.35

# O FUNDO E POR DIA, E NAO UM TETO DE LINHAS. Por que isto e correcao e nao
# preferencia:
#
# a primeira versao pedia 400.000 linhas de fundo ORDENADAS POR DATA. Chicago tem
# milhoes de chamados, entao as 400.000 primeiras cobriram 78 dias -- de
# 2020-01-01 a 2020-03-18 -- contra os 2.438 dias dos positivos. Como o estrato
# do desenho e o DIA, e um dia so informa quando tem as duas classes, sobraram
# 78 dias e 1.772 positivos de 135.983, e o teste saiu nulo. O nulo era da
# amostragem, nao do territorio.
#
# Com teto POR DIA todo dia recebe fundo, e ha um ganho de desenho junto: o
# esforco de fundo passa a ser constante entre dias, entao dia de muito chamado
# nao pesa mais que dia de pouco.
FUNDO_POR_DIA = 150

# E A COTA DO DIA SAI DE UM PONTO SORTEADO DELE, NAO DO COMECO. Por que:
#
# a consulta ordena por `created_date`, entao pedir as 150 primeiras de cada dia
# entrega as 150 de LOGO APOS A MEIA-NOITE -- com ~5.100 chamados/dia, uma
# janela sempre igual de ~40 minutos. Chamado de madrugada nao e amostra justa
# de quem liga: o assunto e o bairro de quem telefona as 00h30 nao sao os de
# quem telefona as 14h. E o fundo so cancela a propensao a reclamar se for
# `target-group background` do dia INTEIRO -- do contrario a hora entra como
# confundidor, e hora se correlaciona com bairro, que se correlaciona com
# terreno, que e exatamente o que o desenho mede.
#
# O deslocamento e sorteado com semente fixa: a corrida reproduz, e a posicao do
# bloco varia entre dias em vez de ser a mesma hora sempre.
#
# E O SORTEIO E SOBRE A CONTAGEM REAL DO DIA, nao sobre um teto fixo. Um teto
# fixo de 1.200 foi medido e ainda truncava: com ~5.100 chamados/dia ele so
# alcanca o primeiro quarto do dia, e o fundo saiu com mediana as 8h e NADA
# depois do meio-dia. Contar antes custa um pedido a mais por dia e faz o bloco
# poder cair em qualquer hora.
#
# O QUE ISTO NAO RESOLVE: dentro de UM dia o bloco continua contiguo, entao
# aquele dia isolado ainda tem recorte de hora. O que se corrige e o acumulado
# entre os 2.438 dias, que e a escala em que o coeficiente e estimado. Fica
# declarado no manifesto, e medido depois pela hora do fundo contra a dos casos.
SEMENTE_FUNDO = 20260903


def _sessao() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = AGENTE
    return s


def _lista_sql(valores) -> str:
    return "(" + ", ".join("'" + v.replace("'", "''") + "'" for v in valores) + ")"


def baixar(sessao, where: str, teto: int | None,
           quieto: bool = False) -> list[dict]:
    """Pagina a consulta ate o fim ou ate o teto declarado."""
    linhas, offset = [], 0
    while True:
        # Nunca pedir mais do que o teto ainda admite: a amostra chegava a
        # baixar 50 mil linhas para usar 5 mil, o que e cota gasta a toa num
        # portal publico.
        falta = PAGINA if teto is None else min(PAGINA, teto - len(linhas))
        if falta <= 0:
            break
        p = {"$select": ",".join(COLUNAS), "$where": where,
             "$order": "created_date", "$limit": falta, "$offset": offset}
        r = sessao.get(BASE, params=p, timeout=300)
        r.raise_for_status()
        lote = r.json()
        linhas.extend(lote)
        if not quieto:
            print(f"    +{len(lote):,} (total {len(linhas):,})", flush=True)
        if len(lote) < falta:
            break
        offset += falta
        if teto and len(linhas) >= teto:
            break
        time.sleep(INTERVALO_S)
    return linhas


def _bloco(sessao, where: str, limite: int, deslocamento: int,
           tentativas: int = 3) -> list[dict]:
    """Um pedido so, a partir de `deslocamento`, com espera crescente no erro.

    Sao ~2.400 pedidos numa corrida completa contra um portal publico sem token.
    Uma falha isolada nao pode virar dia vazio em silencio -- dia vazio some do
    estrato, e estrato que some encolhe o teste sem aparecer como erro. Por isso
    a tentativa se repete antes de o dia ser dado como perdido.
    """
    p = {"$select": ",".join(COLUNAS), "$where": where,
         "$order": "created_date", "$limit": limite, "$offset": deslocamento}
    for n in range(tentativas):
        try:
            r = sessao.get(BASE, params=p, timeout=300)
            r.raise_for_status()
            return r.json()
        except Exception:                                        # noqa: BLE001
            if n == tentativas - 1:
                raise
            time.sleep(2.0 * (n + 1))
    return []


def contar(sessao, where: str, tentativas: int = 3) -> int:
    """Quantos chamados de fundo o dia teve. E o que torna o sorteio uniforme."""
    p = {"$select": "count(*) AS n", "$where": where}
    for n in range(tentativas):
        try:
            r = sessao.get(BASE, params=p, timeout=300)
            r.raise_for_status()
            corpo = r.json()
            return int(corpo[0]["n"]) if corpo else 0
        except Exception:                                        # noqa: BLE001
            if n == tentativas - 1:
                raise
            time.sleep(2.0 * (n + 1))
    return 0


def fundo_do_dia(sessao, where: str, cota: int,
                 rng) -> tuple[list[dict], int, int]:
    """A cota do dia, a partir de um ponto sorteado sobre o dia INTEIRO.

    Devolve tambem o tamanho do dia e o deslocamento usado. Dia que cabe na cota
    vem inteiro, e ai nao ha recorte de hora nenhum para declarar.
    """
    n = contar(sessao, where)
    if n <= cota:
        return _bloco(sessao, where, cota, 0), n, 0
    deslocamento = rng.randrange(0, n - cota + 1)
    return _bloco(sessao, where, cota, deslocamento), n, deslocamento


def gravar(linhas: list[dict], destino: Path) -> dict:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUNAS))
        w.writeheader()
        for l in linhas:
            w.writerow({c: l.get(c, "") for c in COLUNAS})
    b = destino.read_bytes()
    com_coord = sum(1 for l in linhas if l.get("latitude"))
    datas = [l["created_date"][:10] for l in linhas if l.get("created_date")]
    return {
        "arquivo": destino.name,
        "linhas": len(linhas),
        "com_coordenada": com_coord,
        "pct_com_coordenada": round(100.0 * com_coord / len(linhas), 2) if linhas else 0.0,
        "primeira_data": min(datas) if datas else None,
        "ultima_data": max(datas) if datas else None,
        "sha256": hashlib.sha256(b).hexdigest(),
        "bytes": len(b),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--amostra", action="store_true",
                   help="baixa o minimo para conferir a fonte")
    p.add_argument("--completo", action="store_true",
                   help="baixa a serie inteira dos positivos e o fundo ate o teto")
    ns = p.parse_args()
    if not (ns.amostra or ns.completo):
        print("escolha --amostra ou --completo")
        return 2

    sessao = _sessao()
    OUT.mkdir(parents=True, exist_ok=True)
    teto_pos = 5000 if ns.amostra else None

    onde_base = (f"created_date >= '{INICIO}' AND latitude IS NOT NULL")
    print("=== CHICAGO 311 ===")
    print(f"positivos: {list(TIPOS_POSITIVOS)}")
    where_pos = f"{onde_base} AND sr_type IN {_lista_sql(TIPOS_POSITIVOS)}"
    positivos = baixar(sessao, where_pos, teto_pos)
    info_pos = gravar(positivos, OUT / "chicago311_positivos.csv")

    # O fundo vem da MESMA base e da MESMA janela dos positivos: e isso que faz
    # dele `target-group background` e nao um sorteio de pontos no mapa.
    dias = sorted({l["created_date"][:10] for l in positivos
                   if l.get("created_date")})
    if ns.amostra:
        dias = dias[:20]
    print("")
    print(f"fundo: {FUNDO_POR_DIA} por dia, nos {len(dias):,} dias dos positivos")
    fora = _lista_sql(TIPOS_POSITIVOS + TIPOS_EXCLUIDOS)
    rng = random.Random(SEMENTE_FUNDO)
    fundo, do_inicio, perdidos = [], [], []
    for i, dia in enumerate(dias, start=1):
        where_dia = (f"created_date >= '{dia}T00:00:00' "
                     f"AND created_date <= '{dia}T23:59:59' "
                     f"AND latitude IS NOT NULL AND sr_type NOT IN {fora}")
        try:
            lote, n_dia, _ = fundo_do_dia(sessao, where_dia, FUNDO_POR_DIA, rng)
            fundo.extend(lote)
            if n_dia <= FUNDO_POR_DIA:
                do_inicio.append(dia)
        except Exception as e:                                   # noqa: BLE001
            perdidos.append({"dia": dia, "erro": type(e).__name__})
            print(f"    {dia}: {type(e).__name__}")
        if i % 200 == 0 or i == len(dias):
            print(f"    {i}/{len(dias)} dias | {len(fundo):,} linhas | "
                  f"do inicio {len(do_inicio)} | perdidos {len(perdidos)}",
                  flush=True)
        time.sleep(INTERVALO_S)
    # Guarda de centroide: aplicada DEPOIS da consulta, porque ela e sobre a
    # distribuicao do que veio e nao sobre o que foi pedido.
    antes = len(fundo)
    contagem = {}
    for l in fundo:
        chave = (round(float(l["latitude"]), 5), round(float(l["longitude"]), 5))
        contagem[chave] = contagem.get(chave, 0) + 1
    limite = max(1, int(FRACAO_MAXIMA_POR_COORDENADA * antes))
    centroides = {k for k, n in contagem.items() if n > limite}
    if centroides:
        fundo = [l for l in fundo
                 if (round(float(l["latitude"]), 5),
                     round(float(l["longitude"]), 5)) not in centroides]
        print(f"    guarda de centroide: {len(centroides)} coordenada(s) acima de "
              f"{100 * FRACAO_MAXIMA_POR_COORDENADA:.0f}% removida(s), "
              f"{antes - len(fundo):,} linhas")
    info_fundo = gravar(fundo, OUT / "chicago311_fundo.csv")
    info_fundo["removidos_por_centroide"] = antes - len(fundo)
    info_fundo["coordenadas_centroide"] = [list(c) for c in sorted(centroides)]

    tipos_fundo = {}
    for l in fundo:
        t = l.get("sr_type") or ""
        tipos_fundo[t] = tipos_fundo.get(t, 0) + 1

    manifesto = {
        "tipo": "aquisicao_chamados_chicago",
        "gerado_por": "scripts/externo/aq_chicago01_chamados.py",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fonte": "chicago311",
        "regiao": "Chicago, EUA",
        "portal": BASE,
        "modo": "amostra" if ns.amostra else "completo",
        "janela_declarada": INICIO[:10],
        "positivos": {"tipos": list(TIPOS_POSITIVOS), **info_pos},
        "fundo": {"papel": "target-group background: chamado da MESMA base, na "
                           "mesma janela, sobre outro assunto",
                  "teto_por_dia": FUNDO_POR_DIA,
                  "dias_pedidos": len(dias),
                  "dias_cobertos": len({l["created_date"][:10] for l in fundo
                                        if l.get("created_date")}),
                  "dias_perdidos": perdidos,
                  "por_que_por_dia": (
                      "um teto de linhas ordenado por data cobriu 78 dias de "
                      "2.438 e derrubou o teste inteiro; o estrato do desenho e "
                      "o dia, e dia sem fundo nao informa"),
                  "posicao_no_dia": (
                      "deslocamento sorteado sobre a contagem real do dia, "
                      "semente fixa"),
                  "semente": SEMENTE_FUNDO,
                  "dias_que_couberam_na_cota": len(do_inicio),
                  "por_que_sortear": (
                      "as primeiras N linhas do dia sao as de logo apos a "
                      "meia-noite; o fundo so cancela a propensao a reclamar se "
                      "amostrar o dia inteiro, senao a hora entra como "
                      "confundidor e hora se correlaciona com bairro"),
                  "vies_que_permanece": (
                      "dentro de um dia o bloco e contiguo, entao aquele dia "
                      "isolado ainda tem recorte de hora; o que se corrige e o "
                      "acumulado entre dias, que e a escala em que o coeficiente "
                      "e estimado"),
                  "tipos_distintos": len(tipos_fundo),
                  "tipos_mais_comuns": sorted(tipos_fundo.items(),
                                              key=lambda x: -x[1])[:10],
                  **info_fundo},
        "excluidos_das_duas_classes": {
            "tipos": list(TIPOS_EXCLUIDOS),
            "por_que": ("falam de agua e nao sao ocorrencia de alagamento: "
                        "manutencao de rede, falha de infraestrutura e qualidade "
                        "da agua potavel. No fundo eles seriam piores que "
                        "ausentes, porque a localizacao deles depende da rede de "
                        "agua e esgoto e nao do processo geral de reclamar. E a "
                        "mesma armadilha que derrubou o modelo de Sao Paulo de "
                        "servivel para 0,4753"),
        },
        "para_que_serve": (
            "testar se `twi_dinf x chuva`, medido em Nova York, se repete em "
            "outra cidade. Lei fisica se repete; artefato de uma cidade nao"),
        "nao_e": (
            "nao e rotulo promovido, nao e variavel e nao entrou em modelo: e "
            "aquisicao declarada, com sha256 por arquivo. Chamado do cidadao "
            "mede assistencia solicitada, nao extensao de agua"),
    }
    (OUT / "manifesto.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/externo/aq_chicago01_chamados.py --completo\n",
        encoding="utf-8")

    print(f"\npositivos: {info_pos['linhas']:,} "
          f"({info_pos['pct_com_coordenada']}% com coordenada) "
          f"{info_pos['primeira_data']} a {info_pos['ultima_data']}")
    print(f"fundo    : {info_fundo['linhas']:,} em "
          f"{len(tipos_fundo)} tipos distintos")
    print(f"GRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
