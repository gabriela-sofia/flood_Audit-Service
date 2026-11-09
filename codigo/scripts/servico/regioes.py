"""Quem e cada regiao -- por LUGAR, e nunca pelo nome do download.

-------------------------------------------------------------------------
O DEFEITO QUE ESTE MODULO EXISTE PARA ENCERRAR
-------------------------------------------------------------------------

Em 13/09/2026 o Rio Grande do Sul virou regiao servida e o produto mentiu
sobre ele em TRES lugares ao mesmo tempo:

    maturidade_da_regiao         disse que o RS nao entrou no treino
    caixas_das_regioes           nao deu caixa a ele
    regioes_com_referencia_local disse que ele nao tem ponto rotulado

O RS tem 10.792 pontos rotulados, positivo e negativo ambos OBSERVADOS -- o
melhor dado do projeto. As tres funcoes erraram pelo mesmo motivo, e o motivo
nao foi um bug: foi uma SUPOSICAO que valia e deixou de valer.

    enquanto cada regiao servida era sua propria fonte na tabela unica
    (`recife`, `curitiba`, `nyc`, `sao_paulo`), perguntar `fonte == regiao`
    respondia certo, e ninguem precisava declarar nada

O pool areal do Copernicus EMS quebrou isso. O RS e um RECORTE da fonte
`cems` -- cinco AOIs da ativacao EMSR720 --, e a string `rio_grande_do_sul`
nao aparece em `fonte` nenhuma. Cada consumidor que perguntava pela fonte
concluiu, sozinho e em silencio, que a regiao nao existia.

A primeira correcao foi um mapa `regiao -> (fonte, prefixo de AOI)`. Ela
conserta os tres casos e NAO conserta a classe do defeito: continua sendo
casamento de string com nome de download, e volta a quebrar na proxima fonte
que chegar com outro nome para o mesmo lugar -- que e exatamente o que a
aquisicao autonoma vai fazer.

-------------------------------------------------------------------------
A REGRA: REGIAO E UM LUGAR
-------------------------------------------------------------------------

Uma regiao e uma EXTENSAO GEOGRAFICA. Pertencer a ela e cair dentro dela.
Nenhum consumidor pergunta de que fonte o ponto veio, nem como o arquivo se
chamava, nem por qual pipeline ele entrou.

Consequencia pratica, e e a que importa para a aquisicao autonoma: quando
chegar dado novo do mesmo lugar sob outro nome de fonte, ele JA e daquela
regiao. Ninguem precisa lembrar de acrescentar a string em lugar nenhum.

-------------------------------------------------------------------------
DE ONDE VEM A EXTENSAO, ja que ela nao pode vir do nome
-------------------------------------------------------------------------

De uma SEMENTE declarada uma vez por regiao: a evidencia que estabelece onde
aquele lugar fica. Sao dois tipos, e os dois sao verificaveis:

    rotulo   as linhas rotuladas daquela regiao na tabela unica
    terreno  a cadeia de terreno da regiao, quando nao ha rotulo nenhum

Petropolis existe so pelo segundo tipo -- zero ponto rotulado, e mesmo assim
um lugar perfeitamente definido, porque a cadeia de terreno cobre ele. Isso e
o que "trabalhar com terreno e contexto" significa aqui em vez de slogan: a
regiao sem rotulo continua sendo uma regiao.

A semente e usada UMA vez, para derivar a extensao. Depois dela, tudo e
espacial.

-------------------------------------------------------------------------
FAIL-CLOSED
-------------------------------------------------------------------------

Regiao sem extensao derivavel nao ganha extensao inventada: ela simplesmente
nao e resolvivel, e o contrato recusa. Zero, nivel do mar e caixa continental
sao todos piores que recusar, e o projeto ja pagou por dois deles.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

UNICA = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_v2.parquet"
GRADE = REPO / "local_runs" / "svc-03-grade"

# -------------------------------------------------------------------------
# A SEMENTE DE CADA REGIAO, declarada uma vez e com o motivo escrito.
#
# `rotulo`  -> {"fonte": ..., opcionalmente "aoi_prefixo": ...}
# `terreno` -> {"grade": "<nome do csv de grade>"}
#
# Acrescentar regiao nova e acrescentar UMA entrada aqui. Nao ha segunda
# lista, nem terceira: as caixas, os baselines e o teste de pertencer ao
# ajuste saem todos daqui.
# -------------------------------------------------------------------------
SEMENTES = {
    "recife": {"tipo": "rotulo", "fonte": "recife"},
    "curitiba": {"tipo": "rotulo", "fonte": "curitiba"},
    # NOVA YORK E LABORATORIO, E NAO REGIAO DO PRODUTO.
    #
    # Ela continua aqui inteira -- extensao, pertencimento, modelo servido --
    # porque e onde o projeto mediu o mecanismo: no estrato celula x dia, o
    # TWI responde e o efeito dele cresce com a chuva. Esse achado sustenta o
    # que o produto afirma sobre terreno, e sair daqui o apagaria.
    #
    # O que ela nao e: um lugar que o produto oferece. O produto e brasileiro,
    # e uma carta do Brasil com um pino em Manhattan nao descreve nem o
    # produto nem a ciencia. `no_produto` separa as duas coisas num campo, em
    # vez de um filtro escondido na camada de apresentacao.
    "nyc": {"tipo": "rotulo", "fonte": "nyc", "no_produto": False},
    "sao_paulo": {"tipo": "rotulo", "fonte": "sao_paulo"},
    # O RS e um recorte de `cems`: a semente nomeia a ativacao UMA vez, e dali
    # em diante ele e um lugar como qualquer outro.
    "rio_grande_do_sul": {"tipo": "rotulo", "fonte": "cems",
                          "aoi_prefixo": "EMSR720"},
    # Zero ponto rotulado, e ainda assim um lugar: a extensao vem da cadeia de
    # terreno. E o caso que prova que a identidade nao depende de rotulo.
    "petropolis": {"tipo": "terreno", "grade": "grade_petropolis.csv"},
    # -------------------------------------------------------------------------
    # AS CINCO MACRORREGIOES, e por que a unidade e esta
    # -------------------------------------------------------------------------
    #
    # O negativo observado do Global Flood Database cobre 25 UFs, e a tentacao
    # e servir UF. A regra de EPV do projeto nao deixa: medido em 16/09/2026,
    # das 25 UFs so o RS (31 grupos) comporta as tres variaveis, sete comportam
    # uma, e DEZESSETE comportam zero. Em macrorregiao todas comportam pelo
    # menos duas, e Sul e Centro-Oeste comportam as tres.
    #
    # A macrorregiao e a unidade mais FINA em que o orcamento fecha. Nao e
    # recorte cartografico escolhido por gosto -- e onde o dado para de
    # sustentar divisao.
    #
    # A caixa sai dos pontos rotulados da propria macrorregiao, como em toda
    # regiao daqui: a de Recife tambem e o casco dos pontos de Recife, e nao o
    # limite municipal. Servir a caixa administrativa inteira afirmaria
    # cobertura onde nao ha grade derivada.
    "norte": {"tipo": "macrorregiao", "fonte": "gfd", "regiao_br": "N"},
    "nordeste": {"tipo": "macrorregiao", "fonte": "gfd", "regiao_br": "NE"},
    "centro_oeste": {"tipo": "macrorregiao", "fonte": "gfd", "regiao_br": "CO"},
    "sudeste": {"tipo": "macrorregiao", "fonte": "gfd", "regiao_br": "SE"},
    "sul": {"tipo": "macrorregiao", "fonte": "gfd", "regiao_br": "S"},
}

_cache: dict = {}

# -------------------------------------------------------------------------
# O CARTAO DE IDENTIDADE, e por que ele existe
# -------------------------------------------------------------------------
#
# A tabela unica tem 531 MB e nao vai para o repositorio. Sem ela, este modulo
# nao derivava extensao nenhuma -- e SEM EXTENSAO O CONTRATO RECUSA TUDO em
# G2. Na pratica: quem clonava o projeto recebia um servico que nao respondia
# uma requisicao sequer, e o sintoma (`region_not_supported` em toda regiao)
# nao dizia que o que faltava era um arquivo de meio giga.
#
# O que a tabela unica fornece a QUEM SERVE, porem, nao sao 531 MB: sao tres
# fatos por regiao -- a caixa, se ha ponto rotulado dentro, e o piso de
# elevacao. Isso cabe em dois quilobytes, e e o que este cartao materializa.
#
# A tabela continua sendo a FONTE: quando ela esta no disco, e dela que tudo
# sai, e o cartao e apenas regravavel a partir dela (`--gravar`). O cartao so
# responde quando ela nao esta -- e declara que foi ele quem respondeu, para
# ninguem confundir "derivado agora" com "lido de um artefato de ontem".
IDENTIDADE = REPO / "produto" / "identidade_das_regioes.json"


def _identidade() -> dict:
    """O cartao materializado, ou `{}` quando ele nao existe."""
    if "identidade" not in _cache:
        dados = {}
        if IDENTIDADE.exists():
            import json
            try:
                dados = json.loads(IDENTIDADE.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                dados = {}
        _cache["identidade"] = dados.get("regioes", dados) or {}
    return _cache["identidade"]


def _do_cartao(regiao: str, campo: str):
    return (_identidade().get(regiao) or {}).get(campo)


def _tabela():
    import pandas as pd
    if "unica" not in _cache:
        if not UNICA.exists():
            _cache["unica"] = None
        else:
            _cache["unica"] = pd.read_parquet(
                UNICA, columns=["fonte", "aoi", "lat", "lon", "classe",
                                "admitido", "elevation_baseline_m",
                                "elevation_m"])
    return _cache["unica"]


def _linhas_da_semente(regiao: str):
    """As linhas que a SEMENTE aponta. Usado so para derivar a extensao."""
    cfg = SEMENTES.get(regiao)
    if not cfg or cfg["tipo"] not in ("rotulo", "macrorregiao"):
        return None
    d = _tabela()
    if d is None:
        return None
    m = (d.fonte == cfg["fonte"]) & (d.admitido == True)  # noqa: E712
    m &= d.classe.isin([0, 1])
    if cfg.get("aoi_prefixo"):
        m &= d.aoi.astype(str).str.startswith(cfg["aoi_prefixo"], na=False)
    s = d[m]
    if cfg["tipo"] == "macrorregiao":
        rot = _macrorregiao_das_linhas(s)
        if rot is None:
            return None
        s = s[rot == cfg["regiao_br"]]
    return None if s.empty else s


def _macrorregiao_das_linhas(s):
    """A macrorregiao de cada linha, por ponto-em-poligono na malha do IBGE.

    NAO por prefixo do nome do tile: um tile de 1 grau atravessa divisa, e o
    prefixo devolveria a divisa errada sem dar erro -- que e exatamente como o
    `mosaic()` do Earth Engine zerou o relevo do Brasil inteiro em 14/09 sem
    levantar excecao.

    Devolve `None` quando a malha nao esta no disco. Quem chama trata isso como
    semente nao derivavel e cai no cartao de identidade, que e o mesmo caminho
    ja usado quando a tabela unica esta ausente.
    """
    if "macro" in _cache:
        pronto = _cache["macro"]
        return pronto.reindex(s.index) if pronto is not None else None
    try:
        sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))
        import br04_o_brasil_como_professor as br
        if not br.MALHA.exists():
            _cache["macro"] = None
            return None
        d = _tabela()
        from cache_regiao import ler_cache, gravar_cache, calcular_macrorregiao
        pasta_cache = REPO / "local_runs" / "cache-macrorregioes" / "v1"
        pronto = ler_cache(d, br.MALHA, pasta_cache, Path(br.__file__))
        if pronto is None:
            pronto = calcular_macrorregiao(d, br)
            gravar_cache(d, br.MALHA, pasta_cache, pronto, Path(br.__file__))
        _cache["macro"] = pronto
    except Exception:
        _cache["macro"] = None
        return None
    pronto = _cache["macro"]
    return pronto.reindex(s.index) if pronto is not None else None


def extensao(regiao: str) -> dict | None:
    """A caixa da regiao, derivada da semente. `None` se nao for derivavel."""
    if regiao in _cache.get("ext", {}):
        return _cache["ext"][regiao]
    cfg = SEMENTES.get(regiao)
    cx = None
    if cfg and cfg["tipo"] in ("rotulo", "macrorregiao"):
        s = _linhas_da_semente(regiao)
        if s is not None:
            cx = {"lat_min": float(s.lat.min()), "lat_max": float(s.lat.max()),
                  "lon_min": float(s.lon.min()), "lon_max": float(s.lon.max()),
                  "origem": "rotulo", "n_da_semente": int(len(s))}
    elif cfg and cfg["tipo"] == "terreno":
        import pandas as pd
        p = GRADE / cfg["grade"]
        if p.exists():
            g = pd.read_csv(p, usecols=["lat", "lon"])
            cx = {"lat_min": float(g.lat.min()), "lat_max": float(g.lat.max()),
                  "lon_min": float(g.lon.min()), "lon_max": float(g.lon.max()),
                  "origem": "cadeia de terreno", "n_da_semente": int(len(g))}
    if cx is None and cfg:
        # A semente nao pode ser derivada agora (tabela unica ou grade fora do
        # disco). O cartao responde, e diz que foi ele: continua sendo a mesma
        # caixa, derivada da mesma semente, so que numa execucao anterior.
        do_cartao = _do_cartao(regiao, "extensao")
        if do_cartao:
            cx = {**do_cartao, "lido_do_cartao": True}
    _cache.setdefault("ext", {})[regiao] = cx
    return cx


def caixas() -> dict:
    """Toda regiao com extensao derivavel. Substitui `caixas_das_regioes`."""
    return {r: extensao(r) for r in SEMENTES if extensao(r)}


def do_produto() -> list:
    """As regioes que o PRODUTO oferece -- nem toda regiao declarada e uma.

    A diferenca existe por um caso so, e ele esta declarado na semente: Nova
    York e onde o mecanismo foi medido, e nao um lugar que a tela oferece.
    Perguntar aqui, e nao no frontend, mantem a resposta a mesma para qualquer
    consumidor -- mapa, cartao ou relatorio.
    """
    return [r for r, cfg in SEMENTES.items() if cfg.get("no_produto", True)]


def territorio(apenas_do_produto: bool = False) -> dict:
    """Toda regiao declarada, com extensao real quando for derivavel.

    Este inventario nao decide servibilidade, maturidade ou score. Ele existe
    para consumidores cartograficos mostrarem tambem onde o contrato ainda se
    cala, sem transformar ausencia de modelo servido em ausencia de lugar.

    `apenas_do_produto` recorta para o que a tela oferece. O padrao continua
    sendo TUDO: quem audita a ciencia precisa ver Nova York.
    """
    nomes = do_produto() if apenas_do_produto else list(SEMENTES)
    return {r: {"extensao": extensao(r)} for r in nomes}


def _poligono(regiao: str):
    """O poligono da macrorregiao, uniao das UFs dela na malha do IBGE.

    -------------------------------------------------------------------------
    POR QUE A CAIXA NAO BASTA AQUI, E BASTA PARA CIDADE
    -------------------------------------------------------------------------

    Macrorregiao brasileira nao e retangulo. Medido em 16/09/2026: a caixa do
    Centro-Oeste vai de lat -22,48 a -9,84 e lon -57,96 a -47,00; a do Nordeste
    vai de -10,99 a -3,00 e -48,72 a -36,01. Elas se cruzam no canto de
    Tocantins e Bahia -- e nao ha ordem de caixas que resolva isso, porque a
    sobreposicao e da GEOMETRIA, e nao da implementacao.

    Para Recife ou Curitiba a caixa e uma aproximacao boa e continua valendo.
    Para macrorregiao ela afirmaria pertencimento onde nao ha, e o efeito seria
    concreto: um ponto no canto poderia sair como Sudeste (`mvp_local`, porque
    o leave-one-region-out de la estoura o teto da faixa) ou como Centro-Oeste
    (`validado`). Maturidade diferente para o mesmo ponto, conforme a ordem de
    um dicionario.

    A malha e a mesma que o `br04` usa para atribuir regiao aos pontos do
    treino. Usar outra aqui deixaria o produto e a ciencia discordando sobre
    onde as coisas ficam.
    """
    cfg = SEMENTES.get(regiao)
    if not cfg or cfg["tipo"] != "macrorregiao":
        return None
    cache = _cache.setdefault("poly", {})
    if regiao in cache:
        return cache[regiao]
    poly = None
    try:
        sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))
        import json as _json
        import br04_o_brasil_como_professor as br
        from shapely.geometry import shape
        from shapely.ops import unary_union
        if br.MALHA.exists():
            g = _json.loads(br.MALHA.read_text(encoding="utf-8"))
            geoms = [shape(f["geometry"]) for f in g["features"]
                     if br.REGIAO_DA_UF.get(
                         br.UF_DO_CODIGO.get(f["properties"]["codarea"]))
                     == cfg["regiao_br"]]
            if geoms:
                poly = unary_union(geoms)
    except Exception:
        poly = None
    cache[regiao] = poly
    return poly


def dentro(regiao: str, lat: float, lon: float) -> bool:
    """Pertencer e cair dentro -- da caixa, e do poligono quando ele existe.

    A caixa e sempre o primeiro filtro, porque e barata e porque toda regiao
    tem uma. Quando a semente e de macrorregiao, a caixa sozinha nao decide:
    ver `_poligono`.

    Sem a malha no disco o poligono e `None` e a caixa volta a decidir. Isso e
    degradacao declarada e nao silenciosa: `pertencimento_e_por` diz qual das
    duas respondeu, e quem audita consegue ver que o recorte ficou grosso.
    """
    cx = extensao(regiao)
    if not cx:
        return False
    na_caixa = (cx["lat_min"] <= lat <= cx["lat_max"]
                and cx["lon_min"] <= lon <= cx["lon_max"])
    if not na_caixa:
        return False
    poly = _poligono(regiao)
    if poly is None:
        return True
    from shapely.geometry import Point
    return bool(poly.covers(Point(lon, lat)))


def pertencimento_e_por(regiao: str) -> str:
    """`poligono` ou `caixa` -- o que de fato decidiu o `dentro`."""
    cfg = SEMENTES.get(regiao)
    if not cfg or cfg["tipo"] != "macrorregiao":
        return "caixa"
    return "poligono" if _poligono(regiao) is not None else "caixa"


def linhas_da_regiao(regiao: str):
    """As linhas da tabela que caem DENTRO da regiao -- de qualquer fonte.

    E aqui que a promessa do modulo se paga: dado novo do mesmo lugar entra
    sozinho, sob qualquer nome de fonte.
    """
    cx = extensao(regiao)
    d = _tabela()
    if not cx or d is None:
        return None
    m = ((d.admitido == True) & d.classe.isin([0, 1])  # noqa: E712
         & d.lat.between(cx["lat_min"], cx["lat_max"])
         & d.lon.between(cx["lon_min"], cx["lon_max"]))
    s = d[m]
    return None if s.empty else s


def tem_referencia_local(regiao: str) -> bool:
    """Ha ponto rotulado DENTRO da regiao, venha ele de que fonte vier."""
    if _tabela() is None:
        return bool(_do_cartao(regiao, "tem_referencia_local"))
    return linhas_da_regiao(regiao) is not None


def pontos_rotulados(regiao: str) -> int:
    """Quantos pontos rotulados caem dentro da regiao.

    Existe para que quem so precisa da CONTAGEM nao carregue a tabela inteira
    -- e para que ela continue existindo quando a tabela nao esta no disco.
    """
    if _tabela() is None:
        return int(_do_cartao(regiao, "pontos_rotulados") or 0)
    s = linhas_da_regiao(regiao)
    return 0 if s is None else int(len(s))


def baseline_de_elevacao(regiao: str) -> float | None:
    """O piso de elevacao da regiao, calculado NO LUGAR, ou `None`.

    -------------------------------------------------------------------------
    POR QUE O CRITERIO MUDOU EM 24/09/2026
    -------------------------------------------------------------------------

    A versao anterior lia `elevation_baseline_m` das linhas da regiao e exigia
    que o valor fosse UNICO. Esse campo e calculado no `ds04` como o P1 da
    elevacao dentro da propria FONTE -- e o suporte dele e a fonte, nao o
    lugar. Duas consequencias, as duas medidas:

      * tile do GFD, cuja fonte tem escala nacional, cai dentro da caixa de
        uma cidade e traz um segundo baseline. UM ponto bastava para a regiao
        ficar sem piso: em 22/09 Recife, Sao Paulo e Curitiba sairam com
        `None`, e a funcao recusava com razao -- escolher entre dois baselines
        por conta propria tornaria o valor irreproduzivel;
      * para fonte de escala nacional o proprio baseline ja era errado em toda
        parte. O piso do GFD e 2,46 m e o piso local mediano sob os pontos
        dele e 132,91 m; o do CEMS e 0,00 m contra 55,34 m.

    O piso de uma regiao e propriedade do LUGAR. Entao ele passa a ser o P1 da
    `elevation_m` das linhas que caem na regiao, venham da fonte que vierem --
    que e a mesma definicao do `ds04`, aplicada ao suporte certo.

    -------------------------------------------------------------------------
    POR QUE ISTO NAO MEXE EM NENHUM ESCORE
    -------------------------------------------------------------------------

    O escore nao passa por aqui. `svc02` e `svc03` derivam `elevation_rel_m` do
    `elevacao_baseline_por_regiao` declarado pelo MODELO SERVIDO -- que e o
    baseline com que ele foi ajustado, e trocar isso sem reajustar produziria
    feature inconsistente com o ajuste. Esta funcao responde outra pergunta, a
    do cartao de identidade das regioes: qual o piso deste lugar.

    -------------------------------------------------------------------------
    O P1 ABSORVE O INTRUSO, E ESSA ERA A INTENCAO ORIGINAL
    -------------------------------------------------------------------------

    Medido em 24/09 sobre a tabela: Curitiba da 874,633978 com o tile intruso
    dentro, valor IDENTICO ao que dava sem ele; Sao Paulo muda 0,04 m com 28
    intrusos; Recife muda 0,0003 m com 10. Um percentil e robusto a punhado de
    ponto por construcao -- foi por isso que o projeto escolheu P1 e nao
    minimo. O defeito nunca foi o intruso; era exigir unicidade de um valor
    cujo suporte era a fonte.

    Medido de novo em 03/10/2026, depois do lote GFD: Curitiba tem 36 pontos
    estrangeiros (2% das linhas, acima do 1% do P1) e o piso muda 0,17 m
    (874,634 sem eles, 874,465 com). A regra nao mudou; o "identico" de 24/09
    deixou de valer, e o teste passou a proteger a definicao com o efeito
    limitado, e nao o numero.

    As duas recusas continuam, e a segunda e a que o teste protege:

      * sem linha com elevacao na regiao -> `None`;
      * piso ZERO -> `None`, porque zero significa que o P1 caiu no nivel do
        mar e a variavel relativa degenerou em absoluta.
    """
    if _tabela() is None:
        # As recusas ja foram aplicadas quando o cartao foi gravado: o que
        # esta la e `None` ou um baseline valido.
        b = _do_cartao(regiao, "baseline_de_elevacao")
        return None if b is None else float(b)
    s = linhas_da_regiao(regiao)
    if s is None or "elevation_m" not in s.columns:
        return None
    import pandas as pd
    e = pd.to_numeric(s["elevation_m"], errors="coerce").dropna()
    if e.empty:
        return None
    b = round(float(e.quantile(0.01)), 6)
    return None if b == 0.0 else b


def gravar_cartao(caminho: Path = IDENTIDADE) -> dict:
    """Materializa a identidade das regioes a partir da tabela unica.

    Fail-closed: sem a tabela no disco nao ha o que gravar, e sobrescrever o
    cartao com o que o proprio cartao respondeu seria congelar uma copia de
    uma copia sem nunca voltar a fonte.
    """
    import json
    if _tabela() is None:
        raise SystemExit(
            f"ABORTADO: {UNICA} ausente. O cartao so se grava a partir da "
            "tabela unica -- ela e a fonte, e ele e o resumo dela.")
    regioes_ = {}
    for r in SEMENTES:
        cx = extensao(r)
        regioes_[r] = {
            "extensao": {k: v for k, v in (cx or {}).items()
                         if k != "lido_do_cartao"} or None,
            "tem_referencia_local": tem_referencia_local(r),
            "baseline_de_elevacao": baseline_de_elevacao(r),
            "pontos_rotulados": pontos_rotulados(r),
        }
    corpo = {
        "tipo": "identidade_das_regioes",
        "fonte": f"derivado de {UNICA.name} por regioes.py::gravar_cartao",
        "para_que_serve": (
            "permitir que o contrato resolva G2 sem a tabela unica no disco; "
            "sao os unicos tres fatos que quem SERVE precisa dela"),
        "nao_e": ("nao substitui a tabela unica para ajustar modelo, medir "
                  "desempenho ou auditar rotulo -- so para servir"),
        "regioes": regioes_,
    }
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(corpo, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    _cache.pop("identidade", None)
    return corpo


def esta_no_ajuste(regiao: str, modelo: dict) -> bool:
    """A regiao contribuiu com linha para o ajuste deste modelo?

    Pergunta por LUGAR. `fontes_do_ajuste` responde por nome de fonte e por
    isso dizia que o RS -- cinco AOIs dentro de `cems` -- nao estava no treino
    de um modelo ajustado justamente sobre ele.

    QUEM DECLARA, DECIDE -- inclusive quando declara que a lista e VAZIA.
    -------------------------------------------------------------------
    A checagem por fonte e uma inferencia para modelo que nao declarou nada, e
    ela erra nos dois sentidos quando regiao e fonte nao coincidem. Ate 13/09
    so o primeiro erro tinha aparecido (dizer que o RS nao estava no treino
    quando estava). O segundo apareceu ao tirar o RS do pool areal: o modelo
    passou a nao ter nenhuma linha dele, e a inferencia por fonte continuava
    dizendo que sim -- porque as linhas do RS sao `cems`, e `cems` segue no
    ajuste pelas outras treze ativacoes.

    Por isso `regioes_no_ajuste`, quando PRESENTE, e a resposta inteira: lista
    vazia significa "nenhuma regiao declarada entrou", e nao "nao sei".
    """
    if "regioes_no_ajuste" in modelo:
        return regiao in {str(r).lower()
                          for r in (modelo["regioes_no_ajuste"] or [])}
    if regiao in {str(f).lower() for f in modelo.get("fontes_do_ajuste", {})}:
        return True
    fontes = {str(f).lower() for f in modelo.get("fontes_do_ajuste", {})}
    if not fontes:
        return False
    s = linhas_da_regiao(regiao)
    return s is not None and bool(
        set(s.fonte.astype(str).str.lower()) & fontes)


def diagnostico() -> list:
    """Uma linha por regiao, para auditoria e para o teste de coerencia."""
    saida = []
    for r in sorted(SEMENTES):
        cx = extensao(r)
        s = linhas_da_regiao(r)
        saida.append({
            "regiao": r,
            "semente": SEMENTES[r]["tipo"],
            "resolvivel": cx is not None,
            "origem_da_extensao": cx and cx["origem"],
            "linhas_dentro": 0 if s is None else int(len(s)),
            "fontes_dentro": [] if s is None else sorted(
                set(s.fonte.astype(str))),
            "tem_referencia_local": tem_referencia_local(r),
            "baseline": baseline_de_elevacao(r),
        })
    return saida


if __name__ == "__main__":
    import json
    if "--gravar" in sys.argv:
        corpo = gravar_cartao()
        print(f"GRAVADO={IDENTIDADE}")
        for r, v in corpo["regioes"].items():
            cx = v["extensao"]
            print(f"  {r:20s} extensao={'sim' if cx else 'NAO'}  "
                  f"rotulados={v['pontos_rotulados']:>6}  "
                  f"baseline={v['baseline_de_elevacao']}")
    else:
        print(json.dumps(diagnostico(), indent=2, ensure_ascii=False))
