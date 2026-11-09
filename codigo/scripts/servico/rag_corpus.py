"""
Corpus RAG local e curado do REV-P.

F3 do roadmap. Pequeno, local, citavel e versionado. Nao le `logs/`, nao varre
dado privado e nao inventa fonte.

-------------------------------------------------------------------------
AS TRES CLASSES DE AUTORIDADE, e por que uma nao pode falar pela outra
-------------------------------------------------------------------------

Um corpus unico com "documentos" dentro trata como equivalentes coisas que
respondem perguntas diferentes. Aqui elas sao separadas na indexacao, e a
separacao e verificada na hora de citar:

  projeto      verdade sobre o ESTADO do REV-P: maturidade por regiao,
               contrato, model card, o que foi e o que nao foi validado.
               Um paper nao pode dizer que Curitiba esta validada.

  cientifica   literatura e metodologia: D-infinity, HAND, TWI, Firth,
               validacao estruturada, model cards, alucinacao de citacao.
               A documentacao interna nao promove um metodo a consenso.

  operacional  documentacao de dados, APIs e ferramentas: WhiteboxTools,
               Open-Meteo, Copernicus DEM, GEE. Diz COMO uma camada e
               obtida, nunca o que ela significa cientificamente.

A regra e uma so: CLASSE NAO SOBRESCREVE CLASSE. Um claim sobre o estado do
projeto so e sustentado por `projeto`; um claim metodologico so por
`cientifica`. Ver `rag_verificador_citacao.classe_exigida`.

-------------------------------------------------------------------------
O QUE CADA CHUNK CARREGA
-------------------------------------------------------------------------

`chunk_id`, `source_id`, `source_path`, `titulo`, `tipo_fonte`,
`classe_autoridade`, `hash_fonte`, `hash_chunk`, `versao`, `secao`, `ordem`,
`texto`, e `doi`/`url` SOMENTE quando literalmente presentes no proprio
trecho. Um DOI que esta no arquivo mas nao no chunk nao pertence ao chunk:
seria a origem exata do erro que o corpus existe para impedir -- citar uma
referencia que "estava por perto".

`hash_fonte` e do arquivo inteiro, `hash_chunk` do texto do trecho. Os dois,
porque as perguntas sao duas: "o documento mudou?" e "este trecho mudou?".

-------------------------------------------------------------------------
INJECAO
-------------------------------------------------------------------------

Um chunk e DADO. Se ele fala em imperativo com o sistema, alguem esta usando
o corpus como canal de instrucao. A deteccao acontece na INDEXACAO (fica
gravada no manifesto, visivel numa auditoria) e de novo na RECUPERACAO (o
trecho sai do contexto). Duas vezes de proposito: o indice pode ter sido
construido antes de a regra existir.
"""

from __future__ import annotations


def _declarar_metadados_de_saida(dados):
    """Metadados de procedencia na copia serializada; nao modifica o objeto de calculo."""
    if not isinstance(dados, dict):
        return dados
    return {"tipo": "registro_rag_corpus",
            "gerado_por": 'scripts/servico/rag_corpus.py',
            "nao_e": "Registro do produtor rag_corpus: declarar a procedencia nao revalida resultados nem autoriza promocao; conservar os estados e limites especificos do artefato.",
            **dados}


import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from claims import texto_tem_injecao  # noqa: E402
from proveniencia import (  # noqa: E402
    Artefato, Grafo, agora_utc, sha256_arquivo, sha256_json, sha256_texto,
)

OUT = REPO / "local_runs" / "rag-corpus"
DB = OUT / "corpus.sqlite"
MANIFEST = OUT / "manifest.json"
PROVENIENCIA = OUT / "proveniencia.json"

PROJETO = "projeto"
CIENTIFICA = "cientifica"
OPERACIONAL = "operacional"
CLASSES = (PROJETO, CIENTIFICA, OPERACIONAL)

# A publicação atual consulta o artigo e os documentos preparados junto dele.
# Resultados de avaliações anteriores permanecem nas evidências, com seu contexto.
PUBLICACAO = REPO.parent
DOCUMENTOS = [
    (PUBLICACAO / "README.md", PROJETO),
    (PUBLICACAO / "artigo/fonte/artigo.tex", PROJETO),
    (PUBLICACAO / "docs/METODO.md", PROJETO),
    (PUBLICACAO / "docs/CONTRIBUICOES.md", PROJETO),
    (PUBLICACAO / "docs/DADOS_E_FONTES.md", PROJETO),
    (PUBLICACAO / "docs/REPRODUCAO.md", OPERACIONAL),
    (PUBLICACAO / "docs/CODIGO.md", OPERACIONAL),
]

DEFAULT_SOURCES = [caminho for caminho, _ in DOCUMENTOS]
CLASSE_POR_CAMINHO = {caminho: classe for caminho, classe in DOCUMENTOS}

# Referencias primarias. O `texto` e uma descricao curta do que a referencia
# estabelece -- ela e o que sustenta ou nao sustenta um claim, e por isso
# precisa ser fiel e curta, nunca um resumo generoso. DOI e URL sao os
# oficiais; nenhum foi escrito de memoria: todos vem da secao de referencias
# do roadmap, que ja e um documento revisado do projeto.
REFERENCIAS = [
    {
        "titulo": "Tarboton 1997 D-infinity flow directions",
        "doi": "10.1029/96WR03137",
        "url": "https://doi.org/10.1029/96WR03137",
        "texto": ("D-infinity representa direcao de fluxo como angulo continuo "
                  "e reparte o fluxo entre os dois pixels vizinhos que cercam "
                  "esse angulo, em vez de mandar tudo para um unico vizinho "
                  "como o D8."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Renno et al. 2008 HAND",
        "doi": "10.1016/j.rse.2008.03.018",
        "url": "https://www.sciencedirect.com/science/article/pii/S003442570800120X",
        "texto": ("HAND e um descritor topografico que da, para cada celula, a "
                  "altura acima da drenagem mais proxima seguindo o caminho de "
                  "fluxo; e descritor de terreno, nao previsao de inundacao."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Beven e Kirkby 1979 TOPMODEL e indice topografico",
        "doi": "10.1080/02626667909491834",
        "url": "https://eprints.whiterose.ac.uk/id/eprint/111293/",
        "texto": ("O indice topografico combina area de contribuicao a montante "
                  "e declividade local para descrever a propensao de uma celula "
                  "a saturar."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Firth 1993 Bias reduction of maximum likelihood estimates",
        "doi": "10.1093/biomet/80.1.27",
        "url": "https://doi.org/10.1093/biomet/80.1.27",
        "texto": ("Penalizacao de Jeffreys reduz o vies de primeira ordem da "
                  "maxima verossimilhanca e produz estimativas finitas mesmo "
                  "sob separacao."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Heinze e Schemper 2002 separacao em regressao logistica",
        "doi": "10.1002/sim.1047",
        "url": "https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.1047",
        "texto": ("Sob separacao a maxima verossimilhanca nao converge; a "
                  "penalizacao de Firth e apresentada como solucao pratica."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Roberts et al. 2017 structured cross-validation",
        "doi": "10.1111/ecog.02881",
        "url": "https://florianhartig.github.io/publications/roberts2017cross/",
        "texto": ("Dados com estrutura espacial, temporal ou hierarquica exigem "
                  "validacao que respeite essa estrutura; validacao aleatoria "
                  "superestima desempenho."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Peduzzi et al. 1996 eventos por variavel",
        "doi": "10.1016/S0895-4356(96)00236-3",
        "url": "https://doi.org/10.1016/S0895-4356(96)00236-3",
        "texto": ("Estudo de simulacao sobre o numero de eventos por variavel "
                  "em regressao logistica, base do orcamento EPV."),
        "classe": CIENTIFICA,
    },
    # -------------------------------------------------------------------
    # AUTO-SUPERVISAO E ESCALA
    #
    # Identificadores verificados contra a fonte em 2026-09-03 e registrados na
    # secao 3 de docs/REALINHAMENTO_AUTO_SUPERVISAO_E_ESCALA_2026_09_03.md --
    # mesma politica das referencias acima. Nenhum foi escrito de memoria.
    #
    # Entram como `cientifica` porque sustentam afirmacao METODOLOGICA sobre
    # como se pre-treina sem rotulo e como dado, capacidade e compute escalam
    # juntos. Foi a ausencia delas no corpus que deixou a frente neural
    # concluir sobre dado o que era defeito de amostragem.
    {
        "titulo": "He et al. 2022 Masked Autoencoders",
        "doi": "10.48550/arXiv.2111.06377",
        "url": "https://arxiv.org/abs/2111.06377",
        "texto": ("MAE mascara trechos da entrada e reconstroi o que escondeu. "
                  "Dois desenhos centrais: arquitetura encoder-decoder "
                  "assimetrica, com decoder leve, e mascaramento de proporcao "
                  "ALTA da entrada -- 75% -- para que a tarefa auto-"
                  "supervisionada seja nao trivial."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Hoffmann et al. 2022 Compute-Optimal (Chinchilla)",
        "doi": "10.48550/arXiv.2203.15556",
        "url": "https://arxiv.org/abs/2203.15556",
        "texto": ("Para treino compute-optimal o tamanho do modelo e o numero "
                  "de amostras de treino devem escalar IGUALMENTE: a cada "
                  "duplicacao do modelo, duplicar o dado de treino. Modelos "
                  "grandes estavam significativamente subtreinados por escalar "
                  "parametros mantendo o dado constante."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Reed et al. 2023 Scale-MAE",
        "doi": "10.48550/arXiv.2212.14532",
        "url": "https://arxiv.org/abs/2212.14532",
        "texto": ("MAE consciente de escala para dado geoespacial: a AREA DE "
                  "TERRA coberta pela imagem determina a escala da codificacao "
                  "posicional, e nao a resolucao em pixels. Primeiro MAE auto-"
                  "supervisionado com codificacao posicional de escala e "
                  "piramides laplacianas."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "El-Yaniv e Wiener 2010 selective classification",
        "doi": "",
        "url": "https://jmlr.org/papers/v11/el-yaniv10a.html",
        "texto": ("Formaliza predicao seletiva: o classificador pode se abster, "
                  "trocando cobertura por risco menor na regiao onde responde."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Lewis et al. 2020 Retrieval-Augmented Generation",
        "doi": "",
        "url": "https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html",
        "texto": ("RAG combina um gerador parametrico com memoria nao "
                  "parametrica recuperada de um indice externo."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Guu et al. 2020 REALM",
        "doi": "",
        "url": "https://proceedings.mlr.press/v119/guu20a.html",
        "texto": ("REALM usa recuperacao de documentos durante pre-treino e "
                  "inferencia para capturar conhecimento de modo mais modular "
                  "e interpretavel do que armazenar tudo implicitamente nos "
                  "parametros."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Borgeaud et al. 2022 RETRO",
        "doi": "",
        "url": "https://proceedings.mlr.press/v162/borgeaud22a.html",
        "texto": ("RETRO melhora modelos de linguagem condicionando a geracao "
                  "a chunks recuperados de uma base externa, abrindo caminho "
                  "para memoria explicita em vez de depender apenas de mais "
                  "parametros."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Hsueh et al. 2024 pitfalls of knowledge editing",
        "doi": "10.18653/v1/2024.findings-emnlp.550",
        "url": "https://aclanthology.org/2024.findings-emnlp.550/",
        "texto": ("Edicao de conhecimento em LLMs pode causar distorcao de "
                  "conhecimento e deterioracao de capacidades gerais; por isso "
                  "atualizar fatos dentro dos pesos exige avaliacao propria de "
                  "efeitos colaterais."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Mitchell et al. 2019 Model Cards for Model Reporting",
        "doi": "10.1145/3287560.3287596",
        "url": "https://doi.org/10.1145/3287560.3287596",
        "texto": ("Model cards documentam uso pretendido, desempenho "
                  "desagregado, limites e contexto de avaliacao de modelos."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Gebru et al. 2021 Datasheets for datasets",
        "doi": "10.1145/3458723",
        "url": "https://doi.org/10.1145/3458723",
        "texto": ("Datasheets documentam motivacao, composicao, coleta e usos "
                  "recomendados de um conjunto de dados."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Walters e Wilder 2023 citacoes fabricadas por ChatGPT",
        "doi": "10.1038/s41598-023-41032-5",
        "url": "https://pubmed.ncbi.nlm.nih.gov/37679503/",
        "texto": ("Medicao empirica de fabricacao e erro em citacoes "
                  "bibliograficas geradas por modelo de linguagem."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Oquab et al. 2023 DINOv2",
        "doi": "10.48550/arXiv.2304.07193",
        "url": "https://doi.org/10.48550/arXiv.2304.07193",
        "texto": ("DINOv2 aprende features visuais gerais sem supervisao; sao "
                  "descritores visuais, nao evidencia causal de processo "
                  "hidrologico."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Wilkinson et al. 2016 FAIR Guiding Principles",
        "doi": "10.1038/sdata.2016.18",
        "url": "https://doi.org/10.1038/sdata.2016.18",
        "texto": ("Principios FAIR para gestao de dados: localizavel, "
                  "acessivel, interoperavel e reusavel, com metadados ricos."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Beven et al. 2015 comunicacao de incerteza em inundacao",
        "doi": "10.1080/15715124.2014.917318",
        "url": "https://www.tandfonline.com/doi/abs/10.1080/15715124.2014.917318",
        "texto": ("Discute como comunicar incerteza em mapeamento de inundacao "
                  "sem transformar faixa de incerteza em falsa precisao."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "WhiteboxTools manual de analise hidrologica",
        "doi": "",
        "url": "https://jblindsay.github.io/wbt_book/available_tools/hydrological_analysis.html",
        "texto": ("Manual oficial das ferramentas hidrologicas do "
                  "WhiteboxTools, incluindo a implementacao de D-infinity."),
        "classe": OPERACIONAL,
    },
    {
        "titulo": "Copernicus DEM GLO-30 readme oficial",
        "doi": "",
        "url": "https://copernicus-dem-30m.s3.amazonaws.com/readme.html",
        "texto": ("O Copernicus DEM GLO-30 e um modelo digital de SUPERFICIE "
                  "(DSM): inclui dossel e edificacao, e nao equivale a um MDT."),
        "classe": OPERACIONAL,
    },
    {
        "titulo": "Open-Meteo Historical Weather API",
        "doi": "",
        "url": "https://github.com/open-meteo/open-meteo/blob/main/openapi/historical-weather.yml",
        "texto": ("Especificacao OpenAPI do historico horario e diario "
                  "derivado de ERA5 e ERA5-Land."),
        "classe": OPERACIONAL,
    },
    {
        "titulo": "Google Earth Engine Sentinel-1 guia oficial",
        "doi": "",
        "url": "https://developers.google.com/earth-engine/guides/sentinel1",
        "texto": ("Descreve o pre-processamento aplicado a colecao Sentinel-1 "
                  "GRD no Earth Engine e a heterogeneidade entre cenas."),
        "classe": OPERACIONAL,
    },
    # -------------------------------------------------------------------
    # EIXO SOCIAL
    #
    # DOI e URL vem literalmente da secao 8 de
    # docs/REALINHAMENTO_EIXO_HUMANO_POLITICAS_PUBLICAS_2026_09_01.md, que e
    # documento revisado do projeto -- mesma politica de proveniencia das
    # referencias acima. Nenhum identificador foi escrito de memoria.
    #
    # A divisao de classe aqui e deliberada e vale mais que a contagem: so
    # Cutter e Rufat entram como `cientifica`, porque so eles sustentam
    # afirmacao METODOLOGICA sobre como se mede vulnerabilidade social. SVI,
    # Climate Just, IBGE e Periferia Viva sao fonte de dado e enquadramento
    # institucional -- dizem COMO um indicador e obtido e por qual programa,
    # nunca o que ele significa cientificamente. Promover uma pagina de
    # agencia a literatura seria deixar o dado se auto-justificar.
    {
        "titulo": "Cutter, Boruff e Shirley 2003 SoVI vulnerabilidade social",
        "doi": "10.1111/1540-6237.8402002",
        "url": "https://doi.org/10.1111/1540-6237.8402002",
        "texto": ("Vulnerabilidade social a perigos ambientais pode ser medida "
                  "por indicadores socioeconomicos e demograficos agregados, "
                  "mas os fatores relevantes variam por lugar; nao existe um "
                  "conjunto unico valido em qualquer territorio."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "Rufat, Tate, Burton e Maroof 2015 vulnerabilidade social a enchentes",
        "doi": "10.1016/j.ijdrr.2015.09.013",
        "url": "https://doi.org/10.1016/j.ijdrr.2015.09.013",
        "texto": ("Revisao de estudos de vulnerabilidade social a enchentes: "
                  "demografia, status socioeconomico e saude aparecem como "
                  "drivers frequentes, enquanto percepcao de risco e capacidade "
                  "de resposta tambem importam e sao menos capturadas pelos "
                  "indices."),
        "classe": CIENTIFICA,
    },
    {
        "titulo": "CDC/ATSDR Social Vulnerability Index",
        "doi": "",
        "url": "https://www.atsdr.cdc.gov/place-health/php/svi/index.html",
        "texto": ("Indice territorial usado para preparo e resposta a "
                  "emergencias, organizado em dominios de status "
                  "socioeconomico, composicao familiar, raca/etnia, moradia e "
                  "transporte."),
        "classe": OPERACIONAL,
    },
    {
        "titulo": "Climate Just mapeamento de vulnerabilidade e enchente",
        "doi": "",
        "url": "https://www.climatejust.org.uk/mapping/",
        "texto": ("Mapeamento que combina exposicao a enchente e "
                  "vulnerabilidade social para planejamento local, mantendo os "
                  "dois eixos como camadas separadas."),
        "classe": OPERACIONAL,
    },
    {
        "titulo": "IBGE e CEMADEN populacao em areas de risco no Brasil",
        "doi": "",
        "url": ("https://www.ibge.gov.br/geociencias/informacoes-ambientais/"
                "estudos-ambientais/21538-populacao-em-areas-de-risco-no-brasil.html"),
        "texto": ("Base brasileira que associa areas de risco mapeadas a dados "
                  "censitarios para subsidiar monitoramento, alerta e gestao de "
                  "risco."),
        "classe": OPERACIONAL,
    },
    {
        "titulo": "IBGE Censo 2022 tipologia territorial oficial",
        "doi": "",
        "url": ("https://www.ibge.gov.br/geociencias/organizacao-do-territorio/"
                "tipologias-do-territorio/15788-favelas-e-comunidades-urbanas.html"),
        "texto": ("Tipologia territorial oficial do IBGE, publicada no Censo "
                  "2022, usada no REV-P apenas como indicador agregado de "
                  "vulnerabilidade territorial e precariedade urbanistica. "
                  "Nao deve ser devolvida como rotulo direto da area."),
        "classe": OPERACIONAL,
    },
    {
        "titulo": "IBGE caracteristicas urbanisticas do entorno dos domicilios",
        "doi": "",
        "url": ("https://www.ibge.gov.br/geociencias/organizacao-do-territorio/"
                "tipologias-do-territorio/24702-caracteristicas-urbanisticas-do-"
                "entorno-dos-domicilios.html"),
        "texto": ("Levantamento censitario do entorno do domicilio: "
                  "pavimentacao, bueiro ou boca de lobo, esgoto a ceu aberto, "
                  "lixo acumulado e arborizacao, por setor."),
        "classe": OPERACIONAL,
    },
    {
        "titulo": "Ministerio das Cidades Periferia Viva urbanizacao territorial",
        "doi": "",
        "url": ("https://www.gov.br/cidades/pt-br/novo-pac-selecoes-2026/"
                "periferia-viva-urbanizacao-de-favelas/"),
        "texto": ("Programa federal que liga periferias urbanas a risco "
                  "socioambiental, infraestrutura, planejamento, informacao e "
                  "participacao social; e enquadramento institucional, nao "
                  "evidencia sobre um territorio especifico."),
        "classe": OPERACIONAL,
    },
]

PADRAO_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
PADRAO_URL = re.compile(r"https?://[^\s)\]>]+")
PADRAO_TITULO_SECAO = re.compile(r"^(#{1,6})\s+(.*)$")

ALVO_CHUNK = 1400


def _titulo(texto: str, fallback: str) -> str:
    for linha in texto.splitlines():
        limpa = linha.strip(" #\t")
        if limpa:
            return limpa[:160]
    return fallback


def _so_referencia(paragrafo: str) -> bool:
    """O paragrafo e apenas um item de referencia solto?

    Chunk que corta a referencia do trecho que ela sustenta produz duas
    metades inuteis: uma afirmacao sem fonte e uma fonte sem afirmacao. Um
    paragrafo curto que e essencialmente um DOI/URL e grudado no anterior.
    """
    limpo = paragrafo.strip()
    if len(limpo) > 240:
        return False
    return bool(PADRAO_DOI.search(limpo) or PADRAO_URL.search(limpo))


def chunks_por_secao(texto: str, alvo: int = ALVO_CHUNK) -> list:
    """Chunking deterministico que preserva secao e nao corta referencia.

    Deterministico quer dizer: mesma entrada, mesmos chunks, sempre -- em
    qualquer maquina e em qualquer ordem de execucao. Sem isso o `hash_chunk`
    nao serve para comparar duas construcoes do corpus, que e o unico motivo
    de ele existir.
    """
    secao = ""
    blocos: list = []       # (secao, paragrafo)
    for paragrafo in re.split(r"\n\s*\n", texto):
        limpo = paragrafo.strip()
        if not limpo:
            continue
        cabecalho = None
        for linha in limpo.splitlines():
            m = PADRAO_TITULO_SECAO.match(linha.strip())
            if m:
                cabecalho = m.group(2).strip()
                break
        if cabecalho:
            secao = cabecalho
        blocos.append((secao, limpo))

    saida: list = []
    atual, secao_atual = "", ""
    for sec, paragrafo in blocos:
        muda_secao = bool(atual) and sec != secao_atual
        estoura = bool(atual) and len(atual) + len(paragrafo) + 2 > alvo
        if (muda_secao or estoura) and not _so_referencia(paragrafo):
            saida.append({"secao": secao_atual, "texto": atual})
            atual, secao_atual = paragrafo, sec
        else:
            atual = paragrafo if not atual else atual + "\n\n" + paragrafo
            secao_atual = secao_atual or sec
    if atual:
        saida.append({"secao": secao_atual, "texto": atual})
    return saida


RAIZ_MEMORIA = REPO / "local_runs" / "memoria-cientifica"
MEMORIA = RAIZ_MEMORIA / "chunks_rag_verificados.json"


def _chunks_da_memoria(caminho: Path = None) -> list:
    """Chunks `verificado` da memoria cientifica, ou lista vazia.

    Le a raiz E os subdiretorios: os manifestos de execucao ficam na raiz e os
    commits em `commits/`. Sao duas fontes de auto-declaracao diferentes --
    uma diz o que a execucao mediu, a outra diz o que a autora entendeu no dia
    -- e as duas sao conhecimento do projeto.

    Vazio quando a memoria nao existe -- e nao erro. O corpus tem de continuar
    construivel num clone limpo, onde `local_runs/` esta ausente por ser
    git-ignored. Memoria e informacao acumulada, nunca pre-requisito.
    """
    if caminho is not None:
        caminhos = [Path(caminho)]
    else:
        caminhos = [MEMORIA]
        if RAIZ_MEMORIA.exists():
            caminhos += sorted(RAIZ_MEMORIA.glob("*/chunks_rag_verificados.json"))

    saida, vistos = [], set()
    for c in caminhos:
        if not c.exists():
            continue
        try:
            for chunk in json.loads(c.read_text(encoding="utf-8")):
                # `chunk_id` repetido entre fontes sobrescreveria em silencio.
                if chunk.get("chunk_id") in vistos:
                    continue
                vistos.add(chunk.get("chunk_id"))
                saida.append(chunk)
        except Exception:
            continue
    return saida


COLUNAS = ("chunk_id", "source_id", "source_path", "titulo", "secao",
           "tipo_fonte", "classe_autoridade", "hash_fonte", "hash_chunk",
           "versao", "doi", "url", "tem_injecao", "conteudo")


def _conectar(db: Path) -> tuple:
    con = sqlite3.connect(db)
    indexadas = "titulo, secao, conteudo"
    nao_indexadas = [c for c in COLUNAS if c not in ("titulo", "secao", "conteudo")]
    try:
        con.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5("
            + ", ".join(f"{c} UNINDEXED" for c in nao_indexadas)
            + f", {indexadas})")
        # A ordem das colunas no INSERT tem de seguir a da CREATE, e nao a de
        # `COLUNAS`: FTS5 posiciona as UNINDEXED antes das indexadas aqui.
        return con, "fts5", tuple(nao_indexadas) + ("titulo", "secao", "conteudo")
    except sqlite3.OperationalError:
        con.execute("CREATE TABLE chunks("
                    + ", ".join(f"{c} TEXT" for c in COLUNAS) + ")")
        return con, "like", COLUNAS


def construir(out: Path = OUT, sources: list | None = None,
              referencias: list | None = None,
              memoria: list | None = None) -> dict:
    """Monta o indice. `sources`, `referencias` e `memoria` sao as TRES origens.

    `memoria` existia como origem e nao existia como parametro: quem chamava
    `construir(sources=[x], referencias=[])` para montar um corpus isolado
    recebia, sem saber, a memoria cientifica inteira junto. O teste de injecao
    passou meses passando por isso -- ele afirmava que a busca voltava vazia, e
    voltava mesmo, mas por a memoria daquele momento nao casar com a consulta.
    Ao crescer de 55 para 79 chunks em 02/09, um casou, e o defeito apareceu.

    Uma funcao que aceita duas origens para isolar e fixa a terceira nao isola:
    ela engana quem chama.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "corpus.sqlite"
    manifest = out / "manifest.json"
    if db.exists():
        db.unlink()

    con, indice, ordem = _conectar(db)
    sources = DEFAULT_SOURCES if sources is None else sources
    referencias = REFERENCIAS if referencias is None else referencias
    memoria = _chunks_da_memoria() if memoria is None else list(memoria)
    grafo = Grafo(rotulo="rag_corpus")
    registros: list = []
    linhas: list = []
    com_injecao: list = []

    def inserir(reg: dict) -> None:
        con.execute(
            f"INSERT INTO chunks({', '.join(ordem)}) VALUES "
            f"({', '.join('?' for _ in ordem)})",
            tuple(reg[c] for c in ordem))

    for path in sources:
        path = Path(path)
        if path.is_absolute() and path.is_relative_to(PUBLICACAO):
            rel = str(path.relative_to(PUBLICACAO))
        else:
            rel = str(path)
        rel = rel.replace("\\", "/")
        if not path.exists():
            registros.append({"source_id": rel, "source_path": rel,
                              "existe": False, "chunks": 0})
            continue
        texto = path.read_text(encoding="utf-8")
        titulo = _titulo(texto, path.name)
        classe = CLASSE_POR_CAMINHO.get(path, PROJETO)
        hash_fonte = sha256_arquivo(path)
        partes = chunks_por_secao(texto)
        for i, parte in enumerate(partes, start=1):
            corpo = parte["texto"]
            injecao = texto_tem_injecao(corpo)
            reg = {
                "chunk_id": f"{rel}#{i:04d}",
                "source_id": rel,
                "source_path": rel,
                "titulo": titulo,
                "secao": parte["secao"],
                "tipo_fonte": "documento_revp",
                "classe_autoridade": classe,
                "hash_fonte": hash_fonte,
                "hash_chunk": sha256_texto(corpo),
                "versao": None,
                # DOI/URL do PROPRIO trecho. Um identificador que esta no
                # arquivo mas nao neste chunk nao pertence a ele.
                "doi": ";".join(sorted(set(PADRAO_DOI.findall(corpo)))),
                "url": ";".join(sorted(set(PADRAO_URL.findall(corpo)))),
                "tem_injecao": "1" if injecao else "0",
                "conteudo": corpo,
            }
            inserir(reg)
            linhas.append(reg)
            if injecao:
                com_injecao.append({"chunk_id": reg["chunk_id"],
                                    "injecao_detectada": injecao})
        grafo.registrar(Artefato(id=rel, tipo="documento", hash=hash_fonte,
                                 origem="repositorio_local"))
        registros.append({
            "source_id": rel, "source_path": rel, "existe": True,
            "sha256": hash_fonte,
            "titulo": titulo, "chunks": len(partes),
            "tipo_fonte": "documento_revp", "classe_autoridade": classe,
        })

    for i, ref in enumerate(referencias, start=1):
        source_id = f"referencias/{i:03d}"
        corpo = (f"{ref['titulo']}\nDOI: {ref['doi']}\nURL: {ref['url']}\n"
                 f"{ref['texto']}")
        classe = ref.get("classe", CIENTIFICA)
        hash_fonte = sha256_texto(corpo)
        reg = {
            "chunk_id": f"{source_id}#0001", "source_id": source_id,
            "source_path": source_id, "titulo": ref["titulo"], "secao": "",
            "tipo_fonte": "referencia_primaria", "classe_autoridade": classe,
            "hash_fonte": hash_fonte, "hash_chunk": hash_fonte,
            "versao": None, "doi": ref["doi"], "url": ref["url"],
            "tem_injecao": "1" if texto_tem_injecao(corpo) else "0",
            "conteudo": corpo,
        }
        inserir(reg)
        linhas.append(reg)
        grafo.registrar(Artefato(id=source_id, tipo="documento",
                                 hash=hash_fonte, origem="referencia_curada",
                                 extra={"doi": ref["doi"], "url": ref["url"]}))
        registros.append({
            "source_id": source_id, "source_path": source_id, "existe": True,
            "sha256": hash_fonte,
            "titulo": ref["titulo"], "chunks": 1,
            "tipo_fonte": "referencia_primaria", "classe_autoridade": classe,
            "doi": ref["doi"], "url": ref["url"],
        })

    # ------------------------------------------------ memoria cientifica
    #
    # O que o projeto MEDIU, e nao so o que ele escreveu. Ate 02/09 o corpus
    # indexava documentos e referencias -- as duas coisas que alguem senta e
    # redige. Os 260 manifestos de execucao em `local_runs/`, que carregam o
    # numero, o hash e a limitacao declarada de cada rodada, ficavam fora, e o
    # efeito era o sistema nao lembrar do que ele proprio tinha acabado de
    # medir.
    #
    # Entram SO os registros `verificado`, e a verificacao e mecanica: o
    # artefato declara `tipo` e o script gerador esta no repositorio. O que nao
    # passa fica na memoria e aparece na auditoria, sem virar contexto.
    for reg_memoria in memoria:
        corpo = reg_memoria["texto"]
        hash_fonte = reg_memoria.get("hash_fonte") or sha256_texto(corpo)
        reg = {
            "chunk_id": reg_memoria["chunk_id"],
            "source_id": reg_memoria["source_id"],
            "source_path": reg_memoria["source_id"],
            "titulo": reg_memoria["titulo"], "secao": "",
            "tipo_fonte": "memoria_cientifica",
            "classe_autoridade": reg_memoria.get("classe_autoridade", PROJETO),
            "hash_fonte": hash_fonte,
            "hash_chunk": reg_memoria.get("hash_chunk") or sha256_texto(corpo),
            "versao": None, "doi": "", "url": reg_memoria.get("url") or "",
            "tem_injecao": "1" if texto_tem_injecao(corpo) else "0",
            "conteudo": corpo,
        }
        inserir(reg)
        linhas.append(reg)
        registros.append({
            "source_id": reg["source_id"], "source_path": reg["source_id"],
            "existe": True, "sha256": hash_fonte, "titulo": reg["titulo"],
            "chunks": 1, "tipo_fonte": "memoria_cientifica",
            "classe_autoridade": reg["classe_autoridade"],
        })

    for reg in linhas:
        grafo.registrar(Artefato(id=reg["chunk_id"], tipo="chunk",
                                 hash=reg["hash_chunk"],
                                 origem=reg["classe_autoridade"]))
        grafo.relacionar(reg["chunk_id"], "EXTRAIDO_DE", reg["source_id"])

    con.commit()
    con.close()

    por_classe = {c: sum(1 for r in linhas if r["classe_autoridade"] == c)
                  for c in CLASSES}
    # O hash do corpus cobre os chunks, e nao o instante da construcao: e o
    # que permite dizer "o corpus mudou" em vez de "o corpus foi reconstruido".
    hash_corpus = sha256_json(
        [{"chunk_id": r["chunk_id"], "hash_chunk": r["hash_chunk"],
          "classe_autoridade": r["classe_autoridade"]} for r in linhas])

    resultado = {
        "tipo": "rag_corpus_local_curado",
        "quando_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "indice": indice,
        "db": str(db),
        "chunk_count": len(linhas),
        "chunks_por_classe": por_classe,
        "classes_de_autoridade": list(CLASSES),
        "hash_corpus": hash_corpus,
        "sources": registros,
        "chunks_com_injecao": com_injecao,
        "logs_excluidos": True,
        "politica": (
            "resposta so pode citar fonte que apareca literalmente no corpus; "
            "classe de autoridade nao sobrescreve classe; fonte ausente vira "
            "nao_encontrado, nunca resposta confiante"),
    }
    manifest.write_text(json.dumps(_declarar_metadados_de_saida(resultado), indent=2, ensure_ascii=False),
                        encoding="utf-8")
    grafo.gravar(out / "proveniencia.json", com_timestamp=False)
    return resultado


def main() -> int:
    resultado = construir()
    print(json.dumps({
        "manifest": str(MANIFEST), "db": str(DB),
        "chunks": resultado["chunk_count"],
        "por_classe": resultado["chunks_por_classe"],
        "hash_corpus": resultado["hash_corpus"],
        "indice": resultado["indice"],
        "com_injecao": len(resultado["chunks_com_injecao"]),
        "quando_utc": agora_utc(),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
