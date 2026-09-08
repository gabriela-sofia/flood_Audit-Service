"""Adquire as centrais de atendimento ao cidadao que carregam o padrao de Curitiba.

POR QUE ESTA AQUISICAO EXISTE:

`aud_desenho03` mostrou que o desenho que faz Curitiba funcionar -- chamado do
cidadao como positivo, chamado da mesma base em categoria nao-hidrologica como
fundo -- nao e propriedade de Curitiba. E o padrao das centrais 156 no Brasil e
311 nos Estados Unidos, e existe em varias cidades com dados abertos.

Este modulo adquire essas fontes. Ele nao decide se elas servem: `aud_desenho03`
tem os seis requisitos, e a verificacao roda contra o dado baixado.

-------------------------------------------------------------------------
POR QUE EM DUAS FASES, E NAO TUDO DE UMA VEZ
-------------------------------------------------------------------------

O SP156 sao 59 CSVs e mais de tres gigabytes, com TRES esquemas diferentes ao
longo do tempo (2012-2014, 2015-2019, 2020 em diante). Se a base nao trouxer
coordenada por chamado, os tres gigabytes nao respondem nada -- o eixo do
terreno exige positivos e negativos na mesma celula, e sem ponto nao ha celula.

O projeto ja gastou muito tempo adquirindo primeiro e descobrindo o desenho
depois; foi esse o motivo de `aud_desenho03` existir. Entao:

  --amostra    baixa o minimo para responder "esta fonte tem o que e preciso?"
  --completo   baixa a serie inteira, e so faz sentido depois da amostra passar

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao rotula, nao junta com o terreno, nao entra em `ds01` e nao vira variavel.
Ele baixa, registra sha256 e declara o que veio. Promover qualquer coisa disso
a positivo ou a fundo e trabalho do `ds01`, sob os criterios de sempre.

Uso:
    python scripts/externo/aq_chamados01_centrais_de_atendimento.py --amostra
    python scripts/externo/aq_chamados01_centrais_de_atendimento.py --completo --fonte nyc311
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "local_runs" / "aq-chamados-01"

AGENTE = "REV-P/1.0 (pesquisa academica de suscetibilidade a inundacao)"
INTERVALO_MINIMO_S = 1.0
TIMEOUT_S = 900

# Quantas linhas bastam para ler o esquema e decidir se a fonte tem coordenada,
# data e categoria. Nao e amostra estatistica: e leitura de contrato.
LINHAS_DA_AMOSTRA = 5000

FONTES = {
    "nyc311": {
        "regiao": "Nova York, EUA",
        "tipo": "socrata",
        "dataset": "9qq5-d465",
        "base": "https://data.cityofnewyork.us",
        "descricao": "311 - Street Flooding and Sewer Backups",
        "papel": "positivos (recorte tematico ja pronto)",
    },
    "nyc311_fundo": {
        "regiao": "Nova York, EUA",
        "tipo": "socrata",
        "dataset": "erm2-nwe9",
        "base": "https://data.cityofnewyork.us",
        "descricao": "311 Service Requests from 2020 to Present",
        "papel": "fundo da mesma base (target-group background)",
        # A base completa passa de vinte milhoes de linhas. O fundo nao precisa
        # ser exaustivo: precisa vir do MESMO processo de observacao, e isso ja
        # vale com uma amostra grande. `--completo` aqui pega um teto declarado.
        "teto_de_linhas": 400000,
        # A janela deixa de ser escrita a mao: ela e DERIVADA do recorte
        # tematico em disco, por `janela_dos_positivos_em_disco`. A constante
        # abaixo fica so como ultimo recurso, e descreve a janela da AMOSTRA de
        # 5.000 linhas -- que e justamente o que ficou errado quando a
        # exportacao completa entrou e ninguem reavaliou a constante. Uma
        # constante comparada consigo mesma nao verifica nada.
        "pasta_dos_positivos": "nyc311",
        "janela_dos_positivos": ("2026-05-24T00:00:00", "2026-07-30T00:00:00"),
    },
    "sp156": {
        "regiao": "Sao Paulo, BR",
        "tipo": "ckan",
        "portal": "https://dados.prefeitura.sp.gov.br",
        "pacote": "dados-do-sp156",
        "descricao": "SP156/SIGRC -- solicitacoes",
        "papel": "positivos e fundo na mesma base",
        # So o esquema de 2020 em diante; os anteriores tem outro dicionario.
        "filtro_no_nome": ("TRI 2020", "TRI 2021", "TRI 2022", "TRI 2023",
                           "TRI 2024", "TRI 2025", "TRI 2026"),
    },
    "fortaleza_copdc": {
        "regiao": "Fortaleza, BR",
        "tipo": "ckan",
        "portal": "https://dados.fortaleza.ce.gov.br",
        "pacote": "ocorrencias-copdc",
        "descricao": "ocorrencias da Coordenadoria de Protecao e Defesa Civil",
        "papel": "a verificar: nasce de chamado ou de acao, como em Recife?",
    },
}


def baixar(url: str, destino: Path, *, ultima: list) -> dict:
    """Uma requisicao, com intervalo minimo e sha256 do que chegou."""
    espera = INTERVALO_MINIMO_S - (time.time() - ultima[0])
    if espera > 0:
        time.sleep(espera)
    req = urllib.request.Request(url, headers={"User-Agent": AGENTE})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        dados = r.read()
    ultima[0] = time.time()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(dados)
    return {
        "url": url,
        "arquivo": str(destino.relative_to(REPO)),
        "bytes": len(dados),
        "sha256": hashlib.sha256(dados).hexdigest(),
    }


def recursos_ckan(cfg: dict, ultima: list) -> list:
    """Os CSVs do pacote, ja filtrados pelo esquema que interessa."""
    url = f"{cfg['portal']}/api/3/action/package_show?id={cfg['pacote']}"
    espera = INTERVALO_MINIMO_S - (time.time() - ultima[0])
    if espera > 0:
        time.sleep(espera)
    req = urllib.request.Request(url, headers={"User-Agent": AGENTE})
    with urllib.request.urlopen(req, timeout=120) as r:
        pacote = json.loads(r.read())["result"]
    ultima[0] = time.time()
    csvs = [x for x in pacote["resources"]
            if str(x.get("format", "")).lower().strip(".") == "csv"]
    filtro = cfg.get("filtro_no_nome")
    if filtro:
        csvs = [x for x in csvs if any(f in (x.get("name") or "") for f in filtro)]
    return csvs


# ---------------------------------------------- fundo casado no tempo
#
# POR QUE ESTRATIFICAR POR MES, e por que espalhar dentro do mes.
#
# Pedir as N linhas mais recentes de uma janela unica foi o que produziu, em
# 31/08/2026, um fundo de tres semanas contra positivos de seis anos: 96,3% dos
# positivos caiam em dia sem nenhum negativo, e a data sozinha separava as
# classes com AUC 0,9815. O portao do `aq_chamados03` recusa essa promocao.
#
# Um pedido por MES resolve a janela. Mas um pedido por mes com `$limit` pequeno
# e `$order=created_date` traz so os primeiros dias de cada mes -- a janela fica
# certa e o DIA fica errado, e o contraste diario morre do mesmo jeito. Por isso
# cada mes e contado primeiro e depois lido em FATIAS espalhadas por `$offset`:
# a amostra passa a cobrir o mes inteiro.
COLUNAS_DO_FUNDO = ("unique_key", "created_date", "complaint_type",
                    "descriptor", "latitude", "longitude")
FATIAS_POR_MES = 4


def _meses(inicio: str, fim: str) -> list:
    """Primeiro dia de cada mes no intervalo, inclusive."""
    a = datetime.fromisoformat(inicio[:10]).replace(day=1)
    b = datetime.fromisoformat(fim[:10]).replace(day=1)
    saida = []
    while a <= b:
        prox = (a.replace(day=28) + timedelta(days=8)).replace(day=1)
        saida.append((a.strftime("%Y-%m-%dT00:00:00"),
                      prox.strftime("%Y-%m-%dT00:00:00")))
        a = prox
    return saida


def _consultar(url: str, ultima: list) -> bytes:
    espera = INTERVALO_MINIMO_S - (time.time() - ultima[0])
    if espera > 0:
        time.sleep(espera)
    req = urllib.request.Request(url, headers={"User-Agent": AGENTE})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        dados = r.read()
    ultima[0] = time.time()
    return dados


def janela_dos_positivos_em_disco(pasta_positivos: str) -> tuple | None:
    """A janela REAL do recorte tematico ja baixado.

    Derivada do arquivo, e nao escrita a mao. A constante anterior descrevia a
    janela de uma amostra de 5.000 linhas, e ficou errada no instante em que a
    exportacao completa entrou -- e ninguem percebeu, porque uma constante
    comparada consigo mesma nao verifica nada.
    """
    import csv as _csv

    menor = maior = None
    for caminho in sorted((OUT / pasta_positivos).glob("*.csv")):
        with caminho.open(encoding="utf-8", errors="replace", newline="") as fh:
            leitor = _csv.reader(fh)
            cabecalho = next(leitor, None)
            if not cabecalho:
                continue
            idx = next((i for i, c in enumerate(cabecalho)
                        if c.strip().lower() in ("created_date", "created date")),
                       None)
            if idx is None:
                continue
            for linha in leitor:
                if idx >= len(linha) or not linha[idx]:
                    continue
                cru = linha[idx]
                if len(cru) >= 10 and cru[4] == "-":
                    v = cru[:10]
                elif len(cru) >= 10 and cru[2] == "/":
                    # `05/24/2026 10:00:00 AM`, o formato da exportacao
                    mes, dia, ano = cru[:10].split("/")
                    v = ano + "-" + mes + "-" + dia
                else:
                    continue
                menor = v if menor is None or v < menor else menor
                maior = v if maior is None or v > maior else maior
    if menor is None:
        return None
    return (menor + "T00:00:00", maior + "T23:59:59")


def adquirir_fundo_por_mes(nome: str, cfg: dict, linhas_por_mes: int,
                           ultima: list) -> dict:
    """Fundo estratificado por mes, espalhado dentro do mes por `$offset`."""
    base, ds = cfg["base"], cfg["dataset"]
    janela = (janela_dos_positivos_em_disco(cfg["pasta_dos_positivos"])
              or cfg.get("janela_dos_positivos"))
    if not janela:
        raise RuntimeError(
            nome + ": nao ha positivos em disco para derivar a janela; baixe "
            "o recorte tematico antes")
    meses = _meses(*janela)
    destino_dir = OUT / nome
    destino_dir.mkdir(parents=True, exist_ok=True)
    select = urllib.parse.quote(",".join(COLUNAS_DO_FUNDO))
    ordem = urllib.parse.quote("created_date")
    por_fatia = max(linhas_por_mes // FATIAS_POR_MES, 1)

    arquivos, falhas, contagens = [], [], []
    for i, (ini, fim) in enumerate(meses, start=1):
        onde = urllib.parse.quote(
            "created_date >= '" + ini + "' AND created_date < '" + fim + "'")
        try:
            bruto = _consultar(
                base + "/resource/" + ds + ".json?$select=count(1)%20as%20n"
                "&$where=" + onde, ultima)
            n_mes = int(json.loads(bruto)[0]["n"])
        except Exception as e:                                    # noqa: BLE001
            falhas.append({"mes": ini[:7], "etapa": "contagem",
                           "erro": type(e).__name__ + ": " + str(e)[:160]})
            continue
        contagens.append({"mes": ini[:7], "linhas_no_mes": n_mes})
        if n_mes == 0:
            continue
        # fatias igualmente espacadas: cobrem inicio, meio e fim do mes
        passo = max(n_mes // FATIAS_POR_MES, 1)
        partes = []
        for k in range(FATIAS_POR_MES):
            deslocamento = min(k * passo, max(n_mes - 1, 0))
            url = (base + "/resource/" + ds + ".csv?$select=" + select
                   + "&$where=" + onde + "&$order=" + ordem
                   + "&$limit=" + str(por_fatia)
                   + "&$offset=" + str(deslocamento))
            try:
                partes.append(_consultar(url, ultima))
            except Exception as e:                                # noqa: BLE001
                falhas.append({"mes": ini[:7], "fatia": k,
                               "erro": type(e).__name__ + ": " + str(e)[:160]})
        if not partes:
            continue
        # um cabecalho so; as demais fatias entram sem a primeira linha
        corpo = [partes[0]]
        for b in partes[1:]:
            if b.count(b"\n") > 1:
                corpo.append(b.split(b"\n", 1)[1])
        conteudo = b"".join(corpo)
        alvo = destino_dir / (ds + "_fundo_" + ini[:7] + ".csv")
        alvo.write_bytes(conteudo)
        arquivos.append({
            "arquivo": str(alvo.relative_to(REPO)), "mes": ini[:7],
            "bytes": len(conteudo),
            "sha256": hashlib.sha256(conteudo).hexdigest(),
            "linhas_no_mes": n_mes, "fatias": len(partes),
        })
        print("    " + str(i) + "/" + str(len(meses)) + " " + ini[:7]
              + "  mes=" + format(n_mes, ",") + "  baixado="
              + format(len(conteudo) / 1e6, ".1f") + " MB", flush=True)
    return {
        "fonte": nome, "tipo": "socrata", **cfg,
        "modo": "fundo_estratificado_por_mes",
        "janela_derivada_dos_positivos": list(janela),
        "meses": len(meses), "linhas_por_mes_pedidas": linhas_por_mes,
        "fatias_por_mes": FATIAS_POR_MES,
        "colunas": list(COLUNAS_DO_FUNDO),
        "contagem_por_mes": contagens,
        "arquivos": arquivos, "falhas": falhas,
        "por_que_estratificado": (
            "pedir as N linhas mais recentes de uma janela unica produziu "
            "fundo de tres semanas contra positivos de seis anos; a data "
            "passava a separar as classes com AUC 0,9815"),
    }


def adquirir_socrata(nome: str, cfg: dict, completo: bool, ultima: list) -> dict:
    base, ds = cfg["base"], cfg["dataset"]
    destino_dir = OUT / nome
    if completo and nome == "nyc311":
        # exportacao completa do recorte tematico
        url = f"{base}/api/views/{ds}/rows.csv?accessType=DOWNLOAD"
        arq = baixar(url, destino_dir / f"{ds}_completo.csv", ultima=ultima)
    elif completo:
        teto = cfg.get("teto_de_linhas", LINHAS_DA_AMOSTRA)
        # O FUNDO PRECISA VIR DO MESMO PERIODO DOS POSITIVOS. Pedir as ultimas
        # N linhas por `created_date DESC` traz so os dias mais recentes, e se
        # os positivos forem de outro periodo os dois nunca dividem um dia --
        # o eixo da chuva morre por recorte de aquisicao, e nao por
        # propriedade da fonte. Foi o que aconteceu na primeira tentativa:
        # positivos de maio a julho contra fundo de agosto, 0% de data mista.
        janela = cfg.get("janela_dos_positivos")
        onde = (f"&$where={urllib.parse.quote(f'created_date between ' + chr(39) + janela[0] + chr(39) + ' and ' + chr(39) + janela[1] + chr(39))}"
                if janela else "")
        url = (f"{base}/resource/{ds}.csv?$limit={teto}{onde}"
               f"&$order={urllib.parse.quote('created_date DESC')}")
        arq = baixar(url, destino_dir / f"{ds}_teto{teto}.csv", ultima=ultima)
    else:
        url = f"{base}/resource/{ds}.csv?$limit={LINHAS_DA_AMOSTRA}"
        arq = baixar(url, destino_dir / f"{ds}_amostra.csv", ultima=ultima)
    return {"fonte": nome, "tipo": "socrata", **cfg, "arquivos": [arq]}


def adquirir_ckan(nome: str, cfg: dict, completo: bool, ultima: list) -> dict:
    recursos = recursos_ckan(cfg, ultima)
    if not completo:
        # o menor recurso basta para ler o esquema, e custa menos a quem serve
        recursos = sorted(recursos, key=lambda x: int(x.get("size") or 1 << 40))[:1]
    arquivos, falhas = [], []
    for r in recursos:
        alvo = OUT / nome / f"{r['id'][:8]}_{(r.get('name') or 'recurso')[:40]}.csv"
        try:
            arquivos.append({**baixar(r["url"], alvo, ultima=ultima),
                             "recurso": r.get("name")})
        except Exception as e:                                    # noqa: BLE001
            falhas.append({"recurso": r.get("name"), "url": r.get("url"),
                           "erro": f"{type(e).__name__}: {str(e)[:200]}"})
        print(f"    {len(arquivos)}/{len(recursos)} "
              f"{(r.get('name') or '')[:44]:44s} "
              f"{arquivos[-1]['bytes']/1e6 if arquivos else 0:8.1f} MB", flush=True)
    return {"fonte": nome, "tipo": "ckan", **cfg,
            "recursos_encontrados": len(recursos),
            "arquivos": arquivos, "falhas": falhas}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--completo", action="store_true",
                   help="serie inteira; sem isto, so o minimo para ler o esquema")
    p.add_argument("--amostra", action="store_true",
                   help="explicita o padrao: so o minimo para ler o esquema")
    p.add_argument("--fonte", default=None, choices=sorted(FONTES))
    p.add_argument("--fundo-por-mes", type=int, default=None, metavar="N",
                   help="fundo estratificado: N linhas por mes, espalhadas "
                        "dentro do mes, na janela dos positivos em disco")
    ns = p.parse_args()
    if ns.completo and ns.amostra:
        p.error("--amostra e --completo se excluem")

    alvos = {ns.fonte: FONTES[ns.fonte]} if ns.fonte else FONTES
    modo = ("fundo_por_mes" if ns.fundo_por_mes
            else "completo" if ns.completo else "amostra")
    print(f"modo={modo}  fontes={list(alvos)}")

    ultima = [0.0]
    resultados, falhas = {}, {}
    for nome, cfg in alvos.items():
        print(f"\n[{nome}] {cfg['descricao']}")
        try:
            if ns.fundo_por_mes:
                if cfg["tipo"] != "socrata":
                    raise RuntimeError(
                        "fundo por mes so existe para socrata; para CKAN o "
                        "fundo ja vem do mesmo arquivo dos positivos")
                r = adquirir_fundo_por_mes(nome, cfg, ns.fundo_por_mes, ultima)
            else:
                fn = (adquirir_socrata if cfg["tipo"] == "socrata"
                      else adquirir_ckan)
                r = fn(nome, cfg, ns.completo, ultima)
            resultados[nome] = r
            b = sum(a["bytes"] for a in r["arquivos"])
            print(f"    OK {len(r['arquivos'])} arquivo(s), {b/1e6:.1f} MB")
        except Exception as e:                                    # noqa: BLE001
            falhas[nome] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"    FALHOU {falhas[nome]}")

    OUT.mkdir(parents=True, exist_ok=True)
    manifesto = {
        "artefato": "aq_chamados01_centrais_de_atendimento",
        "schema_version": "1.0.0",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modo": modo,
        "fontes_adquiridas": resultados,
        "fontes_com_falha": falhas,
        "total_bytes": sum(a["bytes"] for r in resultados.values()
                           for a in r["arquivos"]),
        "nao_e": ("nao rotula, nao junta com terreno, nao entra no ds01 e nao "
                  "vira variavel; baixa, registra sha256 e declara"),
        "verificacao_de_requisitos": (
            "roda scripts/externo/aq_chamados02_verificar_requisitos.py"),
    }
    sufixo = f"_{ns.fonte}" if ns.fonte else ""
    alvo = OUT / f"manifesto_{modo}{sufixo}.json"
    alvo.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/externo/aq_chamados01_centrais_de_atendimento.py --amostra\n"
        "python scripts/externo/aq_chamados01_centrais_de_atendimento.py --completo\n",
        encoding="utf-8")
    print(f"\ntotal: {manifesto['total_bytes']/1e6:.1f} MB")
    print(f"MANIFESTO={alvo}")
    return 0 if resultados else 1


if __name__ == "__main__":
    raise SystemExit(main())
