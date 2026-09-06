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
    not_testable   a evidencia necessaria nao existe, ou o desenho nao permite
                   o teste.

As nao-equivalencias sao parte do vocabulario, e nao comentario:

    inconclusive  != not_supported
    not_testable  != not_supported
    supported     != validado           (maturidade e outro eixo)
    ausencia de falha != supported

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
NOT_TESTABLE = "not_testable"

ESTADOS_DE_SUPORTE = (SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE, NOT_TESTABLE)

DEFINICAO = {
    SUPPORTED: ("ha evidencia suficiente, segundo o protocolo declarado, para "
                "sustentar a propriedade avaliada neste escopo"),
    NOT_SUPPORTED: ("existe teste identificavel e a evidencia e suficiente "
                    "para nao sustentar a propriedade avaliada neste escopo"),
    INCONCLUSIVE: ("ha evidencia relevante, mas ela nao identifica "
                   "suficientemente a conclusao neste escopo"),
    NOT_TESTABLE: ("a evidencia necessaria para o teste nao existe, ou o "
                   "desenho nao permite o teste neste escopo"),
}

NAO_EQUIVALE = {
    INCONCLUSIVE: [NOT_SUPPORTED],
    NOT_TESTABLE: [NOT_SUPPORTED, SUPPORTED],
    NOT_SUPPORTED: [INCONCLUSIVE, NOT_TESTABLE],
    SUPPORTED: ["validado", NOT_TESTABLE],
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
    NOT_TESTABLE: EFEITO_SUPORTE_NAO_DEMONSTRADO,
}

MOTIVO_DO_EFEITO = {
    SUPPORTED: "a evidencia sustenta a propriedade sob o protocolo declarado",
    NOT_SUPPORTED: "a evidencia identificavel nao sustenta a propriedade",
    INCONCLUSIVE: ("politica conservadora: evidencia relevante existe mas nao "
                   "identifica a conclusao, entao o suporte nao esta "
                   "demonstrado"),
    NOT_TESTABLE: ("politica conservadora: nao ha teste identificavel neste "
                   "escopo, entao o suporte nao esta demonstrado"),
}


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

    def para_dict(self) -> dict:
        return {
            "tipo": "scientific_support_matrix",
            "versao": "1",
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
                    "inconclusive e not_testable levam ao mesmo efeito e "
                    "permanecem estados distintos no registro"),
            },
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
