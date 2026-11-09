"""
Suporte cientifico -- a ontologia geral de sustentacao do REV-P.

O QUE ESTE ARQUIVO E, e por que ele nao e "o arquivo da temporalidade":

o REV-P ja sabia dizer se uma REGIAO tem modelo (`maturidade`) e se uma
REQUISICAO cai dentro do dominio do ajuste (`G5`). O que ele nao sabia dizer,
de forma estruturada, e se uma PROPRIEDADE CIENTIFICA do modelo esta
sustentada por evidencia -- e, principalmente, com que motivo ela nao esta.

Curitiba revelou a falta: "o holdout de 2026 caiu para 0,52" virou, em texto,
"Curitiba nao generaliza no tempo". Sao coisas diferentes. A primeira e
resultado; a segunda e conclusao; e a terceira -- "entao o produto deve
recusar Curitiba" -- e politica. Este arquivo separa as tres.

    evidencia -> avaliacao cientifica -> estado de suporte -> politica -> decisao

-------------------------------------------------------------------------
VOCABULARIO, e de onde ele veio
-------------------------------------------------------------------------

Nao e vocabulario novo. `scripts/challengers/autopsia_curitiba.py` ja usava
`supported` / `not_supported` / `inconclusive` para julgar hipoteses. O que
falta la e o quarto estado, e a falta dele e exatamente o defeito que este
arquivo existe para corrigir: sem `not_testable`, "nao deu para medir" cai em
`not_supported`, e ausencia de teste vira falha demonstrada.

    supported      ha evidencia suficiente, segundo protocolo declarado, para
                   sustentar a propriedade avaliada.
    not_supported  existe teste identificavel E a evidencia e suficiente para
                   NAO sustentar a propriedade avaliada.
    inconclusive   ha evidencia relevante, mas ela nao identifica suficiente-
                   mente a conclusao.
    untested       a evidencia necessaria EXISTE, e nenhum teste foi declarado
                   ou executado neste escopo.
    not_testable   a evidencia necessaria nao existe, ou o desenho nao permite
                   o teste.

As nao-equivalencias sao parte do vocabulario, e nao comentario:

    inconclusive  != not_supported
    not_testable  != not_supported
    untested      != not_testable       (ver abaixo -- e a distincao de 22/09)
    supported     != validado           (maturidade e outro eixo)
    ausencia de falha != supported

-------------------------------------------------------------------------
POR QUE `untested` EXISTE (2026-09-22)
-------------------------------------------------------------------------

`not_testable` estava carregando dois estados que pedem acoes OPOSTAS, e a
matriz os mostrava identicos:

    RS, Petropolis   `sem_dado_para_montar_teste` -- nao ha ponto rotulado na
                     regiao. Sair disso exige CAMPANHA DE AQUISICAO, que pode
                     nem ser possivel.
    Curitiba, NYC,   `protocolo_nao_declara_teste_neste_escopo`, com
    SP, pool         `dados_permitiriam_teste = true` -- o dado sustentaria o
    pluvial          teste; ninguem declarou o corte. Sair disso exige UMA
                     LINHA DE PROTOCOLO.

Eram 4 de 13 escopos no lado barato, invisiveis porque a palavra era a mesma.
O projeto nao conseguia distinguir, olhando a matriz, o que precisa de
aquisicao do que precisa de um corte declarado -- e por isso a fila de
trabalho nunca saia da matriz, saia da memoria de quem estava por perto.

A consequencia operacional dos dois continua igual (nenhum dos dois demonstra
suporte, e nenhum dos dois promove). O que muda e que a razao agora sobrevive
ate o consumidor, e com ela a ACAO PARA SAIR -- ver `ACOES_PARA_SAIR`.

-------------------------------------------------------------------------
A CONSEQUENCIA PODE SER IGUAL; A RAZAO NAO PODE
-------------------------------------------------------------------------

A politica operacional e conservadora: so `supported` conta como suporte
demonstrado. `inconclusive` e `not_testable` levam ao mesmo efeito pratico --
suporte nao demonstrado -- e e por isso que a razao precisa ser persistida
separada. Colapsar os quatro estados num booleano no momento de decidir
apagaria, no registro, a diferenca entre "medimos e nao sustenta", "medimos e
nao deu para separar" e "nao havia como medir".

-------------------------------------------------------------------------
O QUE ESTE ARQUIVO NAO FAZ
-------------------------------------------------------------------------

Nao avalia nada. Ele nao le dado, nao roda modelo e nao decide estado: quem
deriva o estado temporal e `scripts/temporalidade/dimensao_temporal.py`, a
partir de artefato. Aqui ficam so o vocabulario, o escopo, a politica e a
matriz -- para que uma segunda dimensao (espacial, fisica) possa entrar depois
sem inventar uma segunda linguagem cientifica.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------ estados

SUPPORTED = "supported"
NOT_SUPPORTED = "not_supported"
INCONCLUSIVE = "inconclusive"
UNTESTED = "untested"
NOT_TESTABLE = "not_testable"

ESTADOS_DE_SUPORTE = (SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE, UNTESTED,
                      NOT_TESTABLE)

DEFINICAO = {
    SUPPORTED: ("ha evidencia suficiente, segundo o protocolo declarado, para "
                "sustentar a propriedade avaliada neste escopo"),
    NOT_SUPPORTED: ("existe teste identificavel e a evidencia e suficiente "
                    "para nao sustentar a propriedade avaliada neste escopo"),
    INCONCLUSIVE: ("ha evidencia relevante, mas ela nao identifica "
                   "suficientemente a conclusao neste escopo"),
    UNTESTED: ("a evidencia necessaria para o teste existe neste escopo, e "
               "nenhum teste foi declarado ou executado sobre ela"),
    NOT_TESTABLE: ("a evidencia necessaria para o teste nao existe, ou o "
                   "desenho nao permite o teste neste escopo"),
}

NAO_EQUIVALE = {
    INCONCLUSIVE: [NOT_SUPPORTED, UNTESTED],
    UNTESTED: [NOT_SUPPORTED, NOT_TESTABLE, INCONCLUSIVE],
    NOT_TESTABLE: [NOT_SUPPORTED, SUPPORTED, UNTESTED],
    NOT_SUPPORTED: [INCONCLUSIVE, NOT_TESTABLE, UNTESTED],
    SUPPORTED: ["validado", NOT_TESTABLE, UNTESTED],
}

# ---------------------------------------------------------------- dimensoes
#
# Uma dimensao so e AVALIAVEL aqui com protocolo formal escrito. Declarar
# `spatial_support` sem protocolo produziria uma celula vazia que, lida de
# longe, parece avaliacao pendente -- e nao "dimensao que este projeto ainda
# nao sabe avaliar". A abstracao existe; a avaliacao nao e fabricada.

TEMPORAL = "temporal"

DIMENSOES = {
    TEMPORAL: {
        "pergunta": ("qual e o suporte temporal demonstravel para o uso deste "
                     "modelo nesta regiao e neste enquadramento?"),
        "tem_protocolo_formal": True,
        "protocolo": "scripts/temporalidade/ontologia_temporal.py",
    },
    "spatial": {
        "pergunta": "o modelo transporta entre blocos espaciais independentes?",
        "tem_protocolo_formal": False,
        "protocolo": None,
    },
    "domain": {
        "pergunta": "a requisicao cai no dominio multivariado do ajuste?",
        "tem_protocolo_formal": False,
        "protocolo": None,
    },
    "local_reference": {
        "pergunta": "ha referencia observacional local com que verificar?",
        "tem_protocolo_formal": False,
        "protocolo": None,
    },
    "physical_consistency": {
        "pergunta": "os efeitos tem o sinal fisico esperado com IC firme?",
        "tem_protocolo_formal": False,
        "protocolo": None,
    },
}

DIMENSOES_AVALIAVEIS = tuple(
    d for d, v in DIMENSOES.items() if v["tem_protocolo_formal"])


# ----------------------------------------------------------------- politica
#
# A politica e conservadora e esta declarada num lugar so, porque ela e o
# ponto em que uma conclusao cientifica vira consequencia de produto. Mudar
# aqui e decisao, e nao detalhe de implementacao.

EFEITO_SUPORTE_DEMONSTRADO = "suporte_demonstrado"
EFEITO_SUPORTE_REFUTADO = "suporte_refutado"
EFEITO_SUPORTE_NAO_DEMONSTRADO = "suporte_nao_demonstrado"

EFEITO_OPERACIONAL = {
    SUPPORTED: EFEITO_SUPORTE_DEMONSTRADO,
    NOT_SUPPORTED: EFEITO_SUPORTE_REFUTADO,
    INCONCLUSIVE: EFEITO_SUPORTE_NAO_DEMONSTRADO,
    UNTESTED: EFEITO_SUPORTE_NAO_DEMONSTRADO,
    NOT_TESTABLE: EFEITO_SUPORTE_NAO_DEMONSTRADO,
}

MOTIVO_DO_EFEITO = {
    SUPPORTED: "a evidencia sustenta a propriedade sob o protocolo declarado",
    NOT_SUPPORTED: "a evidencia identificavel nao sustenta a propriedade",
    INCONCLUSIVE: ("politica conservadora: evidencia relevante existe mas nao "
                   "identifica a conclusao, entao o suporte nao esta "
                   "demonstrado"),
    UNTESTED: ("politica conservadora: o dado sustentaria o teste, mas "
               "nenhum foi executado neste escopo, entao o suporte nao esta "
               "demonstrado -- e a ausencia e de teste, nao de evidencia"),
    NOT_TESTABLE: ("politica conservadora: nao ha teste identificavel neste "
                   "escopo, entao o suporte nao esta demonstrado"),
}

# ------------------------------------------------- a acao que tira do estado
#
# O QUE ISTO ACRESCENTA, e por que nao e enfeite de relatorio:
#
# ate 22/09 a matriz dizia o estado e a razao, e parava ali. Quem quisesse
# saber o que fazer com um escopo bloqueado tinha de conhecer o projeto. O
# resultado previsivel: 4 escopos que precisavam de UMA LINHA DE PROTOCOLO
# ficaram parados ao lado de 2 que precisavam de campanha de aquisicao, sem
# que ninguem visse a diferenca.
#
# A acao e DERIVADA da razao tecnica, nunca escrita a mao por escopo -- do
# contrario ela seria mais uma tabela congelada, que e o defeito que este
# projeto ja achou dez vezes. Quem publica a razao publica a acao junto.

CUSTO_AQUISICAO = "aquisicao"        # exige dado que nao existe; pode falhar
CUSTO_PROTOCOLO = "protocolo"        # exige declarar e congelar um teste
CUSTO_EXECUCAO = "execucao"          # o teste existe e precisa rodar/ampliar
CUSTO_DESENHO = "desenho"            # a pergunta nao e identificavel como esta
CUSTO_NENHUM = "nenhum"              # ja demonstrado

CLASSES_DE_CUSTO = (CUSTO_AQUISICAO, CUSTO_PROTOCOLO, CUSTO_EXECUCAO,
                    CUSTO_DESENHO, CUSTO_NENHUM)

# Da mais barata para a mais cara. A ordem existe para que a fila saia
# ordenada por onde o projeto avanca mais por unidade de esforco, e nao por
# ordem alfabetica de regiao.
ORDEM_DE_CUSTO = {
    CUSTO_PROTOCOLO: 0,
    CUSTO_EXECUCAO: 1,
    CUSTO_DESENHO: 2,
    CUSTO_AQUISICAO: 3,
    CUSTO_NENHUM: 9,
}


def validar_classe_de_custo(classe: str) -> str:
    if classe not in CLASSES_DE_CUSTO:
        raise ValueError(
            f"classe de custo desconhecida: {classe!r}; conhecidas: "
            f"{list(CLASSES_DE_CUSTO)}")
    return classe


def acao_para_sair(razao_tecnica: str, mapa: dict,
                   estado: str | None = None) -> dict:
    """A acao que tiraria este escopo do estado, lida do mapa da dimensao.

    O mapa vive na dimensao (hoje `dimensao_temporal.ACOES_PARA_SAIR`), e nao
    aqui: quem conhece as razoes tecnicas e quem as produz. Este modulo so
    garante o formato e a classe de custo declarada.

    A CHAVE E O PAR (razao, estado), e nao a razao sozinha. A primeira versao
    deste mapa errou exatamente nisso, e o erro apareceu na primeira execucao:
    `todos_os_cortes_discriminam_acima_do_acaso` sai `supported` quando o
    desenho identifica a pergunta e `inconclusive` quando nao identifica. Com
    a razao como chave unica, os quatro escopos do segundo caso receberam a
    acao do primeiro -- "nada a fazer" -- e teriam sumido da fila carregando o
    confundimento que e justamente o que falta separar.

    Razao sem acao mapeada devolve `mapeada: false` em vez de inventar, e a
    fila mostra isso como lacuna do mapa -- informacao util, e nao escopo sem
    trabalho pendente.
    """
    entrada = mapa.get(razao_tecnica)
    if not entrada:
        return {
            "acao": None,
            "classe_de_custo": None,
            "o_que_produz": None,
            "mapeada": False,
            "observacao": (
                f"razao tecnica {razao_tecnica!r} nao esta no mapa de acoes "
                "desta dimensao; isto e lacuna do mapa, e nao ausencia de "
                "trabalho pendente"),
        }
    por_estado = entrada.get("por_estado") or {}
    escolhida = por_estado.get(estado) if estado else None
    if escolhida is None:
        escolhida = {k: v for k, v in entrada.items() if k != "por_estado"}
    # A checagem e por AUSENCIA DE ACAO, e nao por "o estado nao esta no
    # `por_estado`". A primeira versao perguntava a segunda coisa e so quando
    # `estado is not None` -- entao uma entrada que declara apenas
    # `por_estado`, consultada sem estado, caia num `KeyError` em vez de
    # devolver `mapeada: false`. Chamador que nao sabe o estado e legitimo
    # (a assinatura o torna opcional), e um helper de diagnostico nao pode
    # derrubar quem o chama.
    if "acao" not in escolhida:
        return {
            "acao": None,
            "classe_de_custo": None,
            "o_que_produz": None,
            "mapeada": False,
            "observacao": (
                f"a razao {razao_tecnica!r} so tem acao declarada por estado, "
                f"e o estado {estado!r} nao esta entre os declarados "
                f"({sorted(por_estado)})"),
        }
    validar_classe_de_custo(escolhida["classe_de_custo"])
    return {**escolhida, "mapeada": True}


def validar_estado(estado: str) -> str:
    if estado not in ESTADOS_DE_SUPORTE:
        raise ValueError(
            f"estado de suporte desconhecido: {estado!r}; conhecidos: "
            f"{list(ESTADOS_DE_SUPORTE)}")
    return estado


def suporte_demonstrado(estado: str) -> bool:
    """Unico predicado que autoriza tratar a propriedade como sustentada."""
    return validar_estado(estado) == SUPPORTED


def bloqueia_promocao(estado: str) -> bool:
    """Tudo que nao e `supported` bloqueia -- por motivos que seguem distintos."""
    return not suporte_demonstrado(estado)


def efeito_operacional(estado: str) -> dict:
    validar_estado(estado)
    return {
        "efeito": EFEITO_OPERACIONAL[estado],
        "motivo": MOTIVO_DO_EFEITO[estado],
        "suporte_demonstrado": suporte_demonstrado(estado),
    }


# -------------------------------------------- traducao do vocabulario temporal
#
# A avaliacao temporal continua sendo feita por `ontologia_temporal`, que ja
# existia e ja separa teste invalido de falha. Este mapa e TRADUCAO, e nao um
# segundo julgamento: se o dia em que os dois discordarem chegar, e porque
# alguem escreveu uma regra nova aqui em vez de la.

DE_TEMPORAL_LEGADO = {
    "TEMPORAL_SUPPORT": SUPPORTED,
    "TEMPORAL_FAILURE": NOT_SUPPORTED,
    "TEMPORAL_INCONCLUSIVE": INCONCLUSIVE,
    "TEMPORAL_UNTESTED": UNTESTED,
    "TEMPORAL_TEST_INVALID": NOT_TESTABLE,
}


def do_vocabulario_temporal(status: str) -> str:
    if status not in DE_TEMPORAL_LEGADO:
        raise ValueError(f"status temporal desconhecido: {status!r}")
    return DE_TEMPORAL_LEGADO[status]


# ------------------------------------------------------------------- escopo

@dataclass(frozen=True)
class EscopoDeSuporte:
    """A quem o suporte pertence.

    NUNCA `curitiba.temporal = inconclusive`. Suporte e propriedade de
    REGIAO x MODELO x PROTOCOLO x JANELA -- e nao de regiao. Curitiba tem um
    modelo proprio nao servido e e servida por um modelo transferido; o
    suporte temporal de um nao e o do outro, e guardar por regiao apagaria
    justamente essa diferenca.
    """

    region_id: str
    model_id: str
    protocol_id: str
    evidence_version: str
    training_window: str | None = None
    evaluation_window: str | None = None
    evaluation_domain: str | None = None
    # Janela dos DADOS da regiao, que nao e a janela de treino do modelo. Num
    # modelo de transferencia as duas nao tem nada a ver uma com a outra, e
    # publicar so uma delas faria a janela local parecer treino.
    region_data_window: str | None = None
    forcing_source: str | None = None
    dataset_version: str | None = None
    observation_process: str | None = None
    label_definition: str | None = None

    def chave(self) -> tuple:
        return (self.region_id, self.model_id, self.protocol_id,
                self.evidence_version)

    def para_dict(self) -> dict:
        d = {
            "region_id": self.region_id,
            "model_id": self.model_id,
            "protocol_id": self.protocol_id,
            "evidence_version": self.evidence_version,
            "training_window": self.training_window,
            "evaluation_window": self.evaluation_window,
            "evaluation_domain": self.evaluation_domain,
            "region_data_window": self.region_data_window,
        }
        for campo in ("forcing_source", "dataset_version",
                      "observation_process", "label_definition"):
            valor = getattr(self, campo)
            if valor is not None:
                d[campo] = valor
        return d


@dataclass
class AvaliacaoDeSuporte:
    """Uma conclusao de suporte, com tudo que a torna auditavel.

    Os blocos sao separados de proposito. `support` sozinho e um rotulo; o que
    permite alguem discordar com fundamento e ver o escopo, o protocolo, a
    evidencia, as limitacoes, a proveniencia e a consequencia lado a lado.
    """

    dimension: str
    support: str
    scope: EscopoDeSuporte
    razao_tecnica: str
    evidence: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    provenance: list = field(default_factory=list)
    avaliado_em: str | None = None
    # A acao que tiraria este escopo do estado, derivada da razao tecnica pelo
    # mapa da dimensao. Fica na avaliacao, e nao num relatorio ao lado, porque
    # quem le o estado precisa ver o proximo passo sem abrir outro arquivo.
    acao_para_sair: dict | None = None

    def __post_init__(self):
        validar_estado(self.support)
        if self.dimension not in DIMENSOES:
            raise ValueError(f"dimensao desconhecida: {self.dimension!r}")
        if not DIMENSOES[self.dimension]["tem_protocolo_formal"]:
            raise ValueError(
                f"dimensao {self.dimension!r} nao tem protocolo formal: a "
                "abstracao existe, mas avaliacao sem protocolo seria fabricada")

    @property
    def protocol_id(self) -> str:
        return self.scope.protocol_id

    @property
    def operational_effect(self) -> str:
        return EFEITO_OPERACIONAL[self.support]

    def para_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "support": self.support,
            "support_significa": DEFINICAO[self.support],
            "nao_equivale_a": NAO_EQUIVALE[self.support],
            "scope": self.scope.para_dict(),
            "protocol_id": self.scope.protocol_id,
            "razao_tecnica": self.razao_tecnica,
            "evidence": list(self.evidence),
            "limitations": list(self.limitations),
            "provenance": list(self.provenance),
            "operational_effect": self.operational_effect,
            "operational_effect_motivo": MOTIVO_DO_EFEITO[self.support],
            "acao_para_sair": self.acao_para_sair,
            "avaliado_em": self.avaliado_em,
        }


# ------------------------------------------------------------------- matriz

NOME_DA_MATRIZ = "scientific_support_matrix.json"


class MatrizDeSuporte:
    """O artefato central, e o unico lugar de onde os consumidores leem.

    SVC-02, SVC-07, a camada de claims e o frontend leem daqui. Se cada um
    derivasse o proprio estado temporal, o projeto teria duas definicoes de
    temporalidade -- que e o defeito que a frente inteira existe para evitar.
    """

    def __init__(self, avaliacoes: list | None = None,
                 gerado_em: str | None = None):
        self.avaliacoes = list(avaliacoes or [])
        self.gerado_em = gerado_em

    def adicionar(self, avaliacao: AvaliacaoDeSuporte) -> "MatrizDeSuporte":
        self.avaliacoes.append(avaliacao)
        return self

    @staticmethod
    def _dict(a) -> dict:
        return a if isinstance(a, dict) else a.para_dict()

    def consultar(self, region_id: str, model_id: str,
                  dimension: str = TEMPORAL) -> dict | None:
        """A avaliacao deste escopo, ou None -- nunca a de um escopo vizinho.

        Devolver a avaliacao do modelo local quando quem perguntou serve por
        transferencia seria herdar suporte que ninguem mediu. `None` e a
        resposta correta, e ela tem consequencia declarada em quem chama.
        """
        alvo = (str(region_id).lower(), str(model_id))
        for a in self.avaliacoes:
            d = self._dict(a)
            if d.get("dimension") != dimension:
                continue
            escopo = d.get("scope") or {}
            if (str(escopo.get("region_id")).lower(),
                    str(escopo.get("model_id"))) == alvo:
                return d
        return None

    def por_dimensao(self, dimension: str) -> list:
        return [self._dict(a) for a in self.avaliacoes
                if self._dict(a).get("dimension") == dimension]

    def regioes_avaliadas(self, dimension: str = TEMPORAL) -> set:
        return {str((self._dict(a).get("scope") or {}).get("region_id")).lower()
                for a in self.avaliacoes
                if self._dict(a).get("dimension") == dimension}

    def fila_de_trabalho(self) -> list:
        """Os escopos sem suporte demonstrado, ordenados pelo custo de sair.

        Esta e a diferenca entre uma matriz que RELATA e uma que ORIENTA. Ela
        nao decide nada e nao promove nada -- so torna visivel, em ordem, o
        que o projeto ganharia por unidade de esforco. Um escopo a uma linha
        de protocolo de distancia deixa de aparecer igual a um que depende de
        campanha de aquisicao.

        Ordena por classe de custo e, dentro dela, por escopo -- nunca por
        ordem de insercao, que dependeria de quem rodou o que primeiro.
        """
        itens = []
        for a in self.avaliacoes:
            d = self._dict(a)
            estado = d.get("support")
            if estado is None or suporte_demonstrado(estado):
                continue
            acao = d.get("acao_para_sair") or {}
            escopo = d.get("scope") or {}
            itens.append({
                "dimension": d.get("dimension"),
                "region_id": escopo.get("region_id"),
                "model_id": escopo.get("model_id"),
                "support": estado,
                "razao_tecnica": d.get("razao_tecnica"),
                "acao": acao.get("acao"),
                "classe_de_custo": acao.get("classe_de_custo"),
                "o_que_produz": acao.get("o_que_produz"),
                "acao_mapeada": bool(acao.get("mapeada")),
            })
        return sorted(itens, key=lambda i: (
            ORDEM_DE_CUSTO.get(i["classe_de_custo"], 8),
            str(i["region_id"]), str(i["model_id"])))

    def para_dict(self) -> dict:
        fila = self.fila_de_trabalho()
        por_custo = {}
        for i in fila:
            por_custo.setdefault(str(i["classe_de_custo"]), []).append(
                f"{i['region_id']}/{i['model_id']}")
        return {
            "tipo": "scientific_support_matrix",
            # v2 (2026-09-22): `untested` separado de `not_testable`, e cada
            # avaliacao sem suporte passa a carregar a acao para sair. Subir a
            # versao e obrigatorio: quem le `estados_possiveis` para validar
            # um payload rejeitaria o estado novo sem isto.
            "versao": "2",
            "gerado_em": self.gerado_em,
            "estados_possiveis": list(ESTADOS_DE_SUPORTE),
            "definicoes": DEFINICAO,
            "nao_equivale": NAO_EQUIVALE,
            "dimensoes_declaradas": DIMENSOES,
            "dimensoes_avaliaveis": list(DIMENSOES_AVALIAVEIS),
            "politica_operacional": {
                "efeito_por_estado": EFEITO_OPERACIONAL,
                "motivo_por_estado": MOTIVO_DO_EFEITO,
                "so_supported_demonstra": True,
                "consequencia_igual_razao_diferente": (
                    "inconclusive, untested e not_testable levam ao mesmo "
                    "efeito e permanecem estados distintos no registro"),
            },
            "classes_de_custo": list(CLASSES_DE_CUSTO),
            "fila_de_trabalho": fila,
            "fila_por_classe_de_custo": por_custo,
            "o_que_a_fila_e": (
                "os escopos sem suporte demonstrado, ordenados pelo custo de "
                "sair do estado; e derivada da razao tecnica, nunca escrita a "
                "mao, e nao decide nem promove nada"),
            "avaliacoes": [self._dict(a) for a in self.avaliacoes],
            "nao_e": ("nao e maturidade, nao e portao do contrato v1 e nao "
                      "promove modelo: e a dimensao cientifica separada"),
        }

    def gravar(self, destino: Path) -> Path:
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(
            json.dumps(self.para_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8")
        return destino

    @classmethod
    def carregar(cls, origem: Path) -> "MatrizDeSuporte | None":
        if not origem.exists():
            return None
        try:
            d = json.loads(origem.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if d.get("tipo") != "scientific_support_matrix":
            return None
        return cls(avaliacoes=d.get("avaliacoes") or [],
                   gerado_em=d.get("gerado_em"))


def ausencia_declarada(region_id: str, model_id: str,
                       dimension: str = TEMPORAL,
                       motivo: str = "") -> dict:
    """O que responder quando a matriz nao cobre o escopo perguntado.

    Nao e `not_testable`: `not_testable` e uma CONCLUSAO derivada de artefato,
    e isto aqui e a ausencia da propria conclusao. Confundir os dois faria a
    falta da matriz parecer resultado cientifico.
    """
    return {
        "dimension": dimension,
        "support": None,
        "avaliado": False,
        "scope": {"region_id": region_id, "model_id": model_id},
        "motivo": motivo or (
            "nenhuma avaliacao de suporte foi encontrada para este escopo; "
            "ausencia de avaliacao nao e conclusao cientifica"),
        "operational_effect": EFEITO_SUPORTE_NAO_DEMONSTRADO,
        "operational_effect_motivo": (
            "politica conservadora: sem avaliacao, o suporte nao esta "
            "demonstrado"),
    }
