"""
Modelo formal de claims do REV-P, e o verificador que os resolve sozinho.

-------------------------------------------------------------------------
O QUE MUDA EM RELACAO AO VERIFICADOR ANTERIOR
-------------------------------------------------------------------------

O verificador do SVC-06 julgava FRASE: recebia um texto, aplicava padroes e
devolvia `(aprovada, motivo)`. Isso funcionava, e tinha dois limites que so
aparecem quando o gerador de texto melhora:

  1. o veredito era binario e opaco. Uma frase reprovada dizia UM motivo, o
     primeiro encontrado, e a auditoria nao sabia dizer o que MAIS estava
     errado nela -- nem o que estava certo.

  2. a unidade errada. Uma frase pode conter cinco afirmacoes, quatro
     corretas e uma inventada. Julgar a frase inteira e a decisao certa para
     PUBLICAR (frase e a menor unidade legivel), e a decisao errada para
     AUDITAR: some a informacao de quais afirmacoes o gerador de texto acerta.

Este modulo separa as duas coisas. A unidade de VERIFICACAO passa a ser o
claim -- uma afirmacao tipada, com predicado, valor e origem declarada. A
unidade de PUBLICACAO continua sendo a frase, e a regra que as liga e uma so:

    uma frase so e publicada quando TODOS os claims dela estao `verificado`.

Fail-closed por construcao. Claim que o verificador nao consegue resolver
contra uma origem autorizada nao fica em duvida: fica `sem_suporte`, e
`sem_suporte` derruba a frase. Nao ha estado "provavelmente ok".

-------------------------------------------------------------------------
A REGRA QUE DA NOME AO ARQUIVO
-------------------------------------------------------------------------

    DECLARAR UMA ORIGEM NAO TORNA A ORIGEM VERDADEIRA.

`origens_declaradas` e o que o gerador de texto ALEGA. `evidencias_verificadas` e o
que o verificador ENCONTROU sozinho, indo ler payload, cartao do modelo, contrato,
trecho recuperado e manifesto de proveniencia. Os dois campos existem
separados de proposito: no dia em que forem o mesmo campo, o sistema tera
voltado a confiar no gerador -- e nao havera como notar.

-------------------------------------------------------------------------
O QUE O VERIFICADOR BLOQUEIA
-------------------------------------------------------------------------

Cada item abaixo tem um teste adversarial em
`tests/test_claims_adversarial.py` que falha se o claim for publicado.

     1  numero que nao aparece em origem autorizada
     2  numero alterado (o mesmo campo, valor diferente)
     3  maturidade diferente da que o payload declara
     4  promocao para `validado`, em qualquer regiao
     5  previsao de ocorrencia futura de evento
     6  escore apresentado como probabilidade de o evento acontecer
     7  variavel sem sinal fisico verificado apresentada como determinante
     8  direcao fisica contraria a que o payload declara
     9  efeito com IC que cruza zero apresentado como certeza
    10  Sentinel/DINO como causa
    11  similaridade visual como confirmacao
    12  referencia que nao foi recuperada do corpus
    13  DOI/URL que nao existe em nenhuma fonte recuperada
    14  fonte real cujo trecho nao sustenta o claim
    15  documento recuperado que tenta instruir o sistema
    16  claim de qualquer tipo sem origem autorizada
    17  ausencia, NaN ou recusa apresentada como seguranca/baixo risco
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

# ----------------------------------------------------------------- estados
#
# Tres, e nao dois. `sem_suporte` existe porque "nao consegui verificar" e
# uma informacao diferente de "verifiquei e esta errado" -- para quem le a
# auditoria. Para quem PUBLICA, os dois tem o mesmo efeito, e essa e a parte
# fail-closed: a distincao serve ao diagnostico, nunca a liberacao.

VERIFICADO = "verificado"
REJEITADO = "rejeitado"
SEM_SUPORTE = "sem_suporte"
ESTADOS = (VERIFICADO, REJEITADO, SEM_SUPORTE)

TIPOS_CLAIM = (
    "claim_numero",
    "claim_status",
    "claim_maturidade",
    "claim_variavel",
    "claim_direcao",
    "claim_limitacao",
    "claim_fonte",
    "claim_proveniencia",
    "claim_evidencia_visual",
    # Social e tipo proprio porque vulnerabilidade/exposicao nao e maturidade
    # do modelo nem variavel causal. A camada social pode mudar prioridade
    # publica, nunca escore fisico.
    "claim_social",
    # Extra em relacao ao minimo, e nao por gosto: previsao de evento e a
    # fronteira cientifica numero um do REV-P (o produto responde
    # suscetibilidade do terreno). Enfiar isso dentro de `claim_status`
    # esconderia, na auditoria, exatamente a violacao mais grave.
    "claim_previsao",
    # Suporte cientifico e tipo proprio, e nao um caso de `claim_maturidade`.
    # Maturidade fala de EXISTIR modelo proprio da regiao; suporte temporal
    # fala de a relacao aprendida TRANSPORTAR para eventos posteriores. Sao
    # eixos diferentes, e "Curitiba foi validada temporalmente" viola o
    # segundo sem tocar no primeiro -- Curitiba nem modelo proprio servido tem.
    "claim_suporte_cientifico",
)

# Predicados autorizados. Vocabulario fechado: predicado novo entra por
# decisao, com verificador escrito junto. Um predicado sem verificador e um
# claim que sempre passa -- que e o defeito que este arquivo existe para
# impedir.
PREDICADOS = {
    "valor_numerico": "claim_numero",
    "status_da_resposta": "claim_status",
    "gate_que_nao_fechou": "claim_status",
    "maturidade_do_modelo": "claim_maturidade",
    "efeito_de_variavel": "claim_variavel",
    "direcao_de_variavel": "claim_direcao",
    "certeza_de_efeito": "claim_direcao",
    "limitacao_do_modelo": "claim_limitacao",
    "referencia_citada": "claim_fonte",
    "origem_de_artefato": "claim_proveniencia",
    "papel_da_evidencia_visual": "claim_evidencia_visual",
    "ocorrencia_futura": "claim_previsao",
    "escore_como_probabilidade_de_evento": "claim_previsao",
    "veredito_de_seguranca": "claim_previsao",
    "suporte_temporal": "claim_suporte_cientifico",
    "causa_da_degradacao_temporal": "claim_suporte_cientifico",
    "metrica_isolada_como_suporte": "claim_suporte_cientifico",
    "limitacao_temporal": "claim_suporte_cientifico",
    "classe_social_da_area": "claim_social",
    "prioridade_publica": "claim_social",
    "rotulo_territorial_sensivel": "claim_social",
}

# `validado` em qualquer flexao, e a forma NEGADA dele. As duas precisam ser
# reconhecidas, e por motivos opostos: a afirmativa e a promocao proibida; a
# negada ("nenhuma regiao esta validada") e VERDADEIRA no REV-P e util para o
# leitor. Reprovar as duas seria fail-closed levado ao ponto de apagar
# informacao correta -- que e o outro modo de falha do verificador.
PADRAO_VALIDADO = re.compile(r"\bvalidad[oa]s?\b", re.IGNORECASE)
PADRAO_VALIDADO_NEGADO = re.compile(
    r"\b(?:nao|nenhum[ao]?|sem|ainda\s+nao|nunca)\b[^.;]{0,40}?"
    r"\bvalidad[oa]s?\b", re.IGNORECASE)

MATURIDADES_CONHECIDAS = (
    "validado", "mvp_local", "transferencia_caracterizada",
    "transferencia_sem_referencia_local", "nao_suportada",
)
STATUS_CONHECIDOS = (
    "ok", "insufficient_data", "region_not_supported",
    "insufficient_social_data",
)

CLASSES_SOCIAIS_CONHECIDAS = ("alta", "media", "baixa", "nao_identificavel")
PRIORIDADES_PUBLICAS_CONHECIDAS = (
    "prioridade_publica_forte",
    "prioridade_tecnica",
    "prioridade_de_observacao",
    "prioridade_social_contextual",
    "prioridade_rotineira",
    "prioridade_de_dado",
    "prioridade_de_vistoria_e_coleta",
    "prioridade_de_evidencia",
    # Fisica alta sem cobertura social. Sem esta entrada o
    # verificador rejeitaria como "prioridade publica desconhecida"
    # justamente a classe criada para o caso que mais importa.
    "prioridade_tecnica_e_de_dado",
)
PADRAO_CLASSE_SOCIAL = re.compile(
    r"(?:vulnerabilidade|exposicao|exposição)[^.;]{0,45}?"
    r"\b(alta|media|m[eé]dia|baixa|nao\s+identificavel|não\s+identificável)\b"
    r"|\b(alta|media|m[eé]dia|baixa|nao\s+identificavel|não\s+identificável)\b"
    r"[^.;]{0,45}?(?:vulnerabilidade|exposicao|exposição)",
    re.IGNORECASE)
PADRAO_PRIORIDADE_PUBLICA = re.compile(
    r"prioridade_(?:publica_forte|tecnica_e_de_dado|tecnica|de_observacao|"
    r"social_contextual|rotineira|de_dado|de_vistoria_e_coleta|"
    r"de_evidencia)",
    re.IGNORECASE)
PADRAO_ROTULO_TERRITORIAL_SENSIVEL = re.compile(
    r"\b(?:esta|essa|a)\s+area\s+(?:e|eh|é|foi|esta|está|fica|"
    r"foi\s+classificada\s+como|e\s+classificada\s+como)[^.;]{0,60}?"
    r"(?:favela|comunidade\s+urbana|fcu)\b",
    re.IGNORECASE)

# ------------------------------------------------- suporte temporal na prosa
#
# A ordem importa: a primeira expressao que casar decide o ESTADO que a frase
# tentou afirmar. As formas negadas vem antes das afirmativas pelo mesmo
# motivo de `PADRAO_VALIDADO_NEGADO`: "nao generaliza no tempo" contem
# "generaliza no tempo", e casar a afirmativa primeiro leria a frase ao
# contrario.
#
# `not_testable` tem padrao proprio, e nao e sinonimo de `not_supported`.
# "Petropolis nao possui evidencia local suficiente para realizar o teste" e
# verdadeiro; "Petropolis falhou no teste temporal" e falso, e a diferenca
# entre os dois e a coisa toda que a dimensao existe para preservar.
SUPORTE_NA_PROSA = (
    ("not_supported", re.compile(
        r"(?:nao|nunca)\s+(?:se\s+)?(?:generaliza|transporta|sustenta|"
        r"permanece\s+estavel)\s+(?:\w+\s+){0,3}?tempo"
        r"|colapso\s+temporal\s+(?:comprovad|provad|demonstrad)"
        r"|falh(?:ou|a|ou\s+no)\s+(?:\w+\s+){0,3}?(?:teste\s+)?temporal"
        r"|instabilidade\s+temporal\s+(?:demonstrad|comprovad)"
        r"|suporte\s+temporal\s+(?:refutad|negad)", re.IGNORECASE)),
    ("not_testable", re.compile(
        r"nao\s+(?:e|foi|ha\s+como)\s+testave"
        r"|nao\s+(?:existe|ha)\s+teste\s+(?:temporal|prospectiv)"
        r"|nenhum\s+teste\s+(?:temporal|prospectiv)"
        r"|sem\s+(?:evidencia|referencia|inventario|dado)[^.;]{0,60}?"
        r"(?:para\s+)?(?:realizar|montar|sustentar)[^.;]{0,40}?"
        r"(?:teste|holdout|corte)"
        r"|nao\s+possui\s+evidencia[^.;]{0,60}?(?:teste|holdout|prospectiv)"
        r"|nao\s+(?:e|foi)\s+possivel\s+(?:realizar|montar)[^.;]{0,40}?teste",
        re.IGNORECASE)),
    ("inconclusive", re.compile(
        r"inconclusiv|nao\s+identifica[^.;]{0,40}?(?:conclusao|temporal)"
        r"|evidencia\s+temporal\s+(?:e\s+)?(?:atualmente\s+)?insuficiente",
        re.IGNORECASE)),
    ("supported", re.compile(
        r"validad[oa]s?\s+temporalmente"
        r"|temporalmente\s+(?:estavel|valid|sustentad)"
        r"|permanece\s+estave\w*\s+(?:\w+\s+){0,2}?tempo"
        # O vao entre "temporal" e o participio aceita ate duas palavras
        # ("esta comprovada", "ja foi demonstrada") e recusa qualquer negacao
        # dentro dele -- senao "estabilidade temporal nao esta comprovada"
        # seria lida como afirmacao de estabilidade.
        r"|estabilidade\s+temporal\s+(?:(?!\bnao\b)\w+\s+){0,2}?"
        r"(?:comprovad|demonstrad|estabelecid)"
        r"|passou\s+(?:no|nos)\s+(?:teste|cortes?)\s+(?:temporal|temporais|"
        r"prospectiv\w*)"
        r"|(?:generaliza|transporta|se\s+sustenta)\s+(?:\w+\s+){0,3}?tempo"
        r"|suporte\s+temporal\s+(?:(?!\bnao\b)\w+\s+){0,2}?"
        r"(?:demonstrad|comprovad|sustentad)",
        re.IGNORECASE)),
)


def estados_afirmados(plana: str) -> list:
    """Todos os estados que a frase afirma, e nao so o primeiro.

    A varredura respeita a prioridade da tabela e MASCARA o trecho ja
    consumido antes de testar o proximo padrao. As duas coisas juntas sao
    necessarias:

    - sem prioridade + mascara, "nao generaliza no tempo" casaria tambem como
      afirmacao de estabilidade, pelo pedaco "generaliza no tempo";
    - so com prioridade, parando no primeiro achado, "como nao e testavel, o
      modelo permanece estavel no tempo" seria lida apenas como `not_testable`
      -- e passaria, porque o payload realmente e `not_testable`. A segunda
      metade da frase, que afirma estabilidade, escapava inteira.

    Uma frase pode afirmar dois estados. Quando afirma, os dois sao
    verificados, e basta um nao bater com o payload para ela cair.
    """
    achados, restante = [], plana
    for estado, padrao in SUPORTE_NA_PROSA:
        m = padrao.search(restante)
        if not m:
            continue
        achados.append(estado)
        restante = restante[:m.start()] + " " * (m.end() - m.start()) \
            + restante[m.end():]
    return achados

# Causa da degradacao. Nenhum desenho deste projeto identifica concept drift
# nem mudanca climatica separadamente de forcing, sampling e observacao --
# esta e a conclusao explicita da autopsia de Curitiba. Afirmar a causa e
# sempre alem do que o payload autoriza, e por isso o predicado existe.
PADRAO_CAUSA_TEMPORAL = re.compile(
    r"concept\s*drift|deriva\s+de\s+conceito|mudanca\s+climatica"
    r"|mudanca\s+de\s+clima|porque\s+o\s+clima\s+mudou"
    r"|a\s+relacao\s+(?:fisica\s+)?deixou\s+de\s+valer", re.IGNORECASE)

# Metrica isolada virando suporte. O estado NAO sai de uma AUC: sai de
# identificabilidade, numero de cortes e intervalo -- e um dos casos reais do
# projeto tem AUC 0,95 num corte so e estado `inconclusive`. A barreira exige
# um CONECTIVO inferencial, para nao reprovar frase descritiva ("o suporte e
# sustentado; os cortes tem AUC media 0,7992"), que e correta e util.
PADRAO_METRICA = re.compile(
    r"\bauc\b|area\s+sob\s+a\s+curva|\bacuracia\b|\bbrier\b", re.IGNORECASE)
PADRAO_INFERENCIA = re.compile(
    r"\b(?:porque|pois|logo|portanto|entao|assim|ja\s+que|uma\s+vez\s+que)\b"
    r"|\b(?:mostra|prova|demonstra|indica|comprova|confirma|garante|"
    r"significa|implica)\s+que\b"
    r"|\bcom\s+auc\b|\bcomo\s+a\s+auc\b|\bdado\s+que\b|\bpela\s+auc\b",
    re.IGNORECASE)

# Limitacao temporal afirmada na prosa. Ela e VERIFICAVEL contra o payload:
# a dimensao publica `limitations`, e uma limitacao inventada e tao ruim
# quanto um numero inventado.
PADRAO_LIMITACAO_TEMPORAL = re.compile(
    r"(?:limitacao|limite|restricao|ressalva)[^.;]{0,60}?"
    r"(?:temporal|no\s+tempo|prospectiv)"
    r"|(?:temporal|prospectiv\w*)[^.;]{0,40}?"
    r"(?:tem|possui|carrega)\s+(?:a\s+)?limitac", re.IGNORECASE)


@dataclass
class Claim:
    """Uma afirmacao tipada extraida de uma frase, com veredito proprio."""

    tipo: str
    predicado: str
    valor: object
    origens_declaradas: list = field(default_factory=list)
    estado_verificacao: str = SEM_SUPORTE
    evidencias_verificadas: list = field(default_factory=list)
    motivo_rejeicao: str | None = None

    def __post_init__(self):
        if self.tipo not in TIPOS_CLAIM:
            raise ValueError(f"tipo de claim desconhecido: {self.tipo!r}")
        if self.predicado not in PREDICADOS:
            raise ValueError(f"predicado nao autorizado: {self.predicado!r}")
        if PREDICADOS[self.predicado] != self.tipo:
            raise ValueError(
                f"predicado {self.predicado!r} pertence a "
                f"{PREDICADOS[self.predicado]!r}, nao a {self.tipo!r}")
        if self.estado_verificacao not in ESTADOS:
            raise ValueError(f"estado invalido: {self.estado_verificacao!r}")

    @property
    def publicavel(self) -> bool:
        return self.estado_verificacao == VERIFICADO

    def aprovar(self, *evidencias) -> "Claim":
        self.estado_verificacao = VERIFICADO
        self.evidencias_verificadas = [e for e in evidencias if e]
        self.motivo_rejeicao = None
        return self

    def rejeitar(self, motivo: str) -> "Claim":
        self.estado_verificacao = REJEITADO
        self.motivo_rejeicao = motivo
        return self

    def sem_apoio(self, motivo: str) -> "Claim":
        self.estado_verificacao = SEM_SUPORTE
        self.motivo_rejeicao = motivo
        return self

    def para_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ lexico

def sem_acento(s: str) -> str:
    """`elevacao` e a palavra acentuada equivalente sao a mesma, aqui."""
    return "".join(c for c in unicodedata.normalize("NFD", (s or "").lower())
                   if unicodedata.category(c) != "Mn")


# Tolerancia da comparacao numerica: nao e "quase igual", e o arredondamento
# com que o proprio payload publica os numeros.
CASAS = (0, 1, 2, 3, 4)

PADRAO_NUMERO = re.compile(r"-?\d+(?:[.,]\d+)?")

# Direcao causal. O PLURAL nao e detalhe de regex: a frase por regra escreve
# no singular ("HAND = 13,5 m pesa contra"), e o qwen2.5:3b escreve como
# gente ("elevacao e declividade PESAM a favor"). Sem o plural, a checagem
# inteira era pulada e a frase APROVADA -- publicando fisica invertida em
# portugues fluente, que e a forma mais convincente de estar errado.
PADRAO_DIRECAO = re.compile(
    r"pesa(?:m|ram|va|vam|ndo)?\s+(a\s+favor|contra)", re.IGNORECASE)

# A familia causal inteira, e nao um verbo so. Um verificador ancorado no
# verbo que o PROPRIO projeto escreve nao esta verificando texto de terceiro:
# esta confiando nele.
PADRAO_CAUSAL = re.compile(
    r"\b(?:"
    r"pesa(?:m|ram|va|vam|ndo)?|"
    r"influencia(?:m|do|da|dos|das|ndo)?|influi(?:r|em)?|"
    r"contribu(?:i|em|iu|indo)|"
    r"caus(?:a|am|ou|ar|ando|ado|ada|ados|adas|ava|avam)|"
    r"favorece(?:m|u)?|"
    r"aumenta(?:m|r|ndo)?|reduz(?:em|ir|indo)?|diminui(?:em|r|ndo)?|"
    r"determina(?:m|do|da)?|explica(?:m|do|da)?|"
    r"responsav(?:el|eis)|associad(?:o|a|os|as)|"
    r"depende(?:m)?|impacta(?:m|do|da)?"
    r")\b", re.IGNORECASE)

# Verbos que transformam contribuicao do estimador em causa do fenomeno. A
# frase auditavel "pesa a favor/contra" descreve coeficiente x valor e nao
# pertence a este subconjunto.
PADRAO_CAUSA_FORTE = re.compile(
    r"\b(?:caus(?:a|am|ou|ar|ando|ado|ada|ados|adas|ava|avam)|"
    r"determina(?:m|do|da)?|responsav(?:el|eis)|prova|comprova|garante)\b",
    re.IGNORECASE)

# O que o gerador de texto escreve quando quer dizer a mesma coisa que o payload. Sem
# isto o nome nao casa, a variavel some da checagem -- e sumir e pior que ser
# reprovada, porque some em silencio.
APELIDOS = {
    "hand_m": ("hand", "altura acima da drenagem", "altura sobre a drenagem",
               "altura acima do canal"),
    "twi_dinf": ("twi", "umidade topografica", "topografico de umidade",
                 "indice de umidade", "umidade do terreno"),
    "elevation_m": ("elevacao", "altitude", "elevation"),
    "slope_deg": ("declividade", "inclinacao", "slope"),
    "rain_max_24h": ("pico de chuva", "chuva em 24", "chuva de 24",
                     "chuva maxima", "rain_max"),
    "rain_decay_index": ("chuva antecedente", "decaimento", "rain_decay",
                         "chuva acumulada"),
}

# Previsao de ocorrencia futura. O contrato responde suscetibilidade do
# terreno; qualquer coisa que projete o EVENTO esta fora do que o artefato
# sustenta, por mais bem escrita que esteja.
TERMOS_PREVISAO = (
    "vai alagar", "vai inundar", "vai enchente", "certamente", "com certeza",
    "garantido", "previsao de enchente", "prevemos", "ira alagar",
    "irá alagar", "risco de vida", "evacue", "evacuar", "vai transbordar",
    "deve alagar", "alagara", "inundara", "amanha", "nos proximos dias",
    "na proxima chuva", "quando chover",
)

# Escore virando probabilidade de o evento acontecer. O payload define
# `escore` como media, sobre os pontos, da probabilidade que o MODELO
# AJUSTADO atribui a cada ponto -- que nao e a chance de a area alagar.
PADRAO_PROB_DE_EVENTO = re.compile(
    r"(?:chance|probabilidade|risco|possibilidade)\s+(?:de\s+)?"
    r"(?:\d+[.,]?\d*\s*%\s*)?(?:de\s+)?"
    r"(?:alagar|inundar|enchente|inundacao|alagamento|transbordar|"
    r"que\s+alague|que\s+inunde)", re.IGNORECASE)
PADRAO_PERCENTUAL_DE_CHANCE = re.compile(
    r"\d+[.,]?\d*\s*%\s*de\s*(?:chance|probabilidade|risco)", re.IGNORECASE)

# Veredito de seguranca. NAO EXISTE campo no payload que autorize dizer que
# uma area e segura, tem baixo risco ou esta livre de enchente -- e a ausencia
# desse campo e uma decisao, nao um esquecimento. O contrato publica escore
# RELATIVO de suscetibilidade, e uma recusa publica silencio.
#
# Este padrao nasceu de dois casos adversariais que passavam por todas as
# outras checagens porque nao citavam numero, nao nomeavam variavel e nao
# afirmavam direcao:
#
#   "Como a variavel esta ausente, a area tem baixo risco."
#   "A requisicao foi recusada, entao o local e seguro."
#
# Os dois convertem AUSENCIA em GARANTIA, que e a leitura mais perigosa que o
# produto pode induzir -- e a unica em que o erro custa vida, nao reputacao.
PADRAO_SEGURANCA = re.compile(
    r"\b(?:"
    r"baixo\s+risco|risco\s+baixo|sem\s+risco|nenhum\s+risco|"
    r"nao\s+(?:ha|existe|oferece|apresenta|corre)\s+risco|"
    r"(?:e|sao|esta|estao|fica|considerad[oa]s?)\s+segur[oa]s?|"
    r"local\s+segur[oa]|area\s+segur[oa]|zona\s+segur[oa]|"
    r"livre\s+de\s+(?:enchente|inundac|alagament)|"
    r"fora\s+de\s+perigo|pode\s+ficar\s+tranquil"
    r")\b", re.IGNORECASE)

# Certeza sobre efeito. Usada junto com `ic_cruza_zero`: coeficiente cujo IC
# cruza zero e um efeito que o ajuste NAO estabeleceu, e dizer que ele
# "prova" algo e apresentar incerteza como certeza.
TERMOS_CERTEZA = (
    "prova", "comprova", "demonstra que", "garante", "sem duvida",
    "inequivoc", "e certo que", "indiscutiv", "conclusiv", "seguramente",
)

# Evidencia visual. Somente revisao por decisao de projeto: DINOv2 gera representacoes
# visuais gerais, nao prova causal hidrologica.
PADRAO_VISUAL = re.compile(
    r"\b(?:sentinel(?:-?[12])?|dino(?:v2)?|sar|embedding|"
    r"similaridade\s+visual|imagem\s+de\s+satelite|imagens\s+de\s+satelite)\b",
    re.IGNORECASE)
VERBOS_CONFIRMACAO = (
    "confirma", "comprova", "prova", "valida", "atesta", "reforca",
    "corrobora", "evidencia que", "mostra que", "indica que",
)

# Citacao bibliografica.
#
# O DOI PRECISA SER APARADO, e a razao e chata e importante: a sintaxe de DOI
# admite ponto e parentese, entao `10.1029/96WR03137.` no fim de uma frase
# casa COM o ponto final. O identificador com o ponto grudado nunca bate com o
# do corpus, e a citacao correta era reprovada -- reprovar fonte legitima e
# tao ruim quanto aprovar fonte falsa, e mais dificil de perceber.
PADRAO_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
PADRAO_URL = re.compile(r"https?://[^\s)\]\",]+")

# `(Autor, 1997)` -- citacao parentetica.
PADRAO_CITACAO_AUTOR_ANO = re.compile(
    r"\(([A-ZÀ-Ý][A-Za-zÀ-ÿ.'-]+(?:\s+(?:e|and|et\s+al\.?|&)\s*"
    r"[A-Za-zÀ-ÿ.'-]*)?),?\s*(1[89]\d{2}|20\d{2})\)")
# `Tarboton (1997)` -- citacao narrativa, com o autor FORA dos parenteses.
# Faltava, e a lacuna era silenciosa: a frase "a abordagem segue Tarboton
# (1997)" nao gerava claim de fonte nenhum. So o `1997` era visto, como
# numero -- entao a frase caia pelo motivo errado, e uma frase igual com um
# ano que existisse no payload teria passado citando uma fonte nunca
# recuperada.
PADRAO_CITACAO_NARRATIVA = re.compile(
    r"\b([A-ZÀ-Ý][A-Za-zÀ-ÿ.'-]+(?:\s+(?:e|and|et\s+al\.?|&)\s*"
    r"[A-Za-zÀ-ÿ.'-]*)?)\s*\((1[89]\d{2}|20\d{2})\)")

# Pontuacao que encosta no fim de um identificador e nao faz parte dele.
FIM_DE_FRASE = ".,;:!?)]}\"'"


def aparar_identificador(ident: str) -> str:
    return (ident or "").strip().rstrip(FIM_DE_FRASE)


# Palavras que nao distinguem nada. Sem esta lista, "de", "que" e "modelo"
# dariam sobreposicao alta entre qualquer claim e qualquer trecho, e a
# checagem de suporte aprovaria tudo -- que e pior que nao existir.
VAZIAS = {
    "que", "com", "para", "dos", "das", "uma", "uns", "por", "nao", "mais",
    "como", "esta", "este", "essa", "esse", "seu", "sua", "pelo", "pela",
    "sao", "foi", "ser", "ter", "tem", "the", "and", "for", "with", "from",
    "aqui", "onde", "quando", "entre", "sobre", "todo", "toda", "cada",
    "conforme", "segundo", "ver", "veja", "acordo",
}


def termos_de_conteudo(texto: str) -> set:
    return {t for t in re.findall(r"[a-z0-9_]{3,}", sem_acento(texto or ""))
            if t not in VAZIAS}


def sustenta(claim: str, trecho: str, minimo: float = 0.34) -> tuple:
    """O trecho sustenta o claim? Devolve `(bool, fracao)`.

    Sobreposicao de termos de conteudo: grosseira de proposito e declarada
    como tal. Nao entende o texto, e nao finge entender. Pega o caso que
    importa -- citar uma fonte que nao tem nada a ver com a afirmacao -- e
    deixa passar parafrase habilidosa.

    MORA AQUI, e nao no verificador de citacao, porque e uma regra do
    verificador de CLAIMS: uma citacao com DOI correto e trecho fora de
    assunto passava na checagem de presenca literal e publicava. O caso que
    revelou isso foi "Conforme 10.1029/96WR03137, Curitiba esta validada" --
    DOI real, recuperado de verdade, sustentando uma promocao de maturidade
    que o artigo de Tarboton obviamente nao sustenta.
    """
    termos_claim = termos_de_conteudo(claim)
    if not termos_claim:
        return False, 0.0
    fracao = len(termos_claim & termos_de_conteudo(trecho)) / len(termos_claim)
    return fracao >= minimo, round(fracao, 3)


def mascarar_identificadores(frase: str) -> str:
    """Troca DOI e URL por espacos, preservando o comprimento.

    UM DOI NAO E UM NUMERO DA FRASE, e tratar como se fosse reprovava toda
    citacao legitima: `10.1029/96WR03137` faz o extrator numerico ver
    "10.1029", que nao aparece em campo nenhum do payload -- porque nao
    deveria mesmo aparecer. A frase inteira caia com um motivo que nao tinha
    nada a ver com o defeito.

    Mascarar preservando o comprimento mantem qualquer indice de posicao
    valido, e mascarar em vez de remover evita colar duas palavras vizinhas
    e criar um termo que a frase nao tem.
    """
    fora = frase or ""
    for padrao in (PADRAO_URL, PADRAO_DOI):
        fora = padrao.sub(lambda m: " " * len(m.group(0)), fora)
    return fora

# Proveniencia afirmada em prosa.
PADRAO_PROVENIENCIA = re.compile(
    r"\b(?:treinad[oa]\s+com|ajustad[oa]\s+(?:com|em|sobre)|"
    r"derivad[oa]\s+de|proveniente\s+de|vem\s+d[eoa]s?|"
    r"gerad[oa]\s+(?:por|a\s+partir)|extraid[oa]\s+de|"
    r"os\s+dados\s+(?:sao|vieram|vem))\b", re.IGNORECASE)

# Promocao de maturidade / negacao de limitacao.
PADRAO_SEM_LIMITACAO = re.compile(
    r"\b(?:sem\s+limitac|nenhuma\s+limitac|nao\s+(?:ha|possui|tem)\s+"
    r"(?:limitac|restric|ressalva)|totalmente\s+valida|plenamente\s+valida|"
    r"pronto\s+para\s+uso\s+operacional|apto\s+para\s+operacao|"
    r"modelo\s+valida)", re.IGNORECASE)

# Documento recuperado tentando instruir o sistema. Um chunk e DADO; se ele
# fala em imperativo com o sistema, alguem esta usando o corpus como canal de
# instrucao -- e o corpus e leitura, nunca prompt.
PADRAO_INJECAO = re.compile(
    r"(?:ignore\s+(?:as\s+)?(?:instruc|previous|todas)|"
    r"desconsidere\s+(?:as\s+)?(?:instruc|regras)|"
    r"disregard\s+(?:all\s+)?(?:previous|prior)|"
    r"esqueca\s+(?:as\s+)?(?:instruc|regras)|"
    r"nova[s]?\s+instruc(?:ao|oes)|"
    r"voce\s+(?:deve|precisa|tem\s+que)\s+(?:responder|dizer|afirmar|ignorar)|"
    r"aja\s+como|finja\s+que|a\s+partir\s+de\s+agora\s+voce|"
    r"^\s*system\s*:|override\s+the\s+(?:verifier|rules)|"
    r"publique\s+sem\s+verific)", re.IGNORECASE | re.MULTILINE)


def texto_tem_injecao(texto: str) -> str | None:
    """Devolve o trecho que tenta instruir o sistema, ou `None`."""
    m = PADRAO_INJECAO.search(sem_acento(texto or ""))
    return m.group(0).strip() if m else None


# ------------------------------------------------- numeros que o payload tem

def _textos_do_payload(resposta: dict) -> list:
    """Toda string que o payload ja publica, para colher os numeros dela."""
    saida: list = []

    def juntar(v):
        if isinstance(v, str):
            saida.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                juntar(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                juntar(x)

    juntar(resposta.get("limitacoes"))
    juntar(resposta.get("escore_detalhe"))
    juntar(resposta.get("model_card"))
    # A dimensao cientifica publica ano de corte, janela e AUC por corte. Se
    # ela nao entrar aqui, uma frase correta sobre a evidencia temporal cai
    # pelo proprio numero que o payload acabou de publicar.
    juntar(resposta.get("suporte_cientifico"))
    juntar((resposta.get("explicacao") or {}).get("texto"))
    # Numa recusa o payload nao tem campo numerico nenhum, e `detalhe` e o
    # unico texto que existe. Sem colher dele, a frase que NOMEIA a recusa
    # ("variavel ausente no ponto 0") cairia pelo proprio `0` que ela cita.
    juntar(resposta.get("detalhe"))
    juntar(resposta.get("gate_que_nao_fechou"))
    # E os NOMES dos campos, nao so os valores. O `95` de "intervalo de 95%"
    # nao e valor de nada -- e o nivel de confianca, que esta no proprio nome
    # do campo `ic95`. Colher da chave torna a regra auto-mantida: se o campo
    # um dia virar `ic90`, o 90 passa a ser aceito e o 95 deixa de ser.
    saida.extend(_chaves(resposta))
    return saida


def _chaves(v, acc: list | None = None) -> list:
    acc = [] if acc is None else acc
    if isinstance(v, dict):
        for k, x in v.items():
            acc.append(str(k))
            _chaves(x, acc)
    elif isinstance(v, (list, tuple)):
        for x in v:
            _chaves(x, acc)
    return acc


def numeros_permitidos(resposta: dict) -> set:
    """Todo numero que o payload publica, em todos os arredondamentos dele."""
    brutos: list = []

    def juntar(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            brutos.append(float(v))
        elif isinstance(v, dict):
            for x in v.values():
                juntar(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                juntar(x)

    for chave in ("escore", "ic95", "n_pontos", "dominio"):
        juntar(resposta.get(chave))
    mc = resposta.get("model_card") or {}
    for chave in ("desempenho", "ajustado_em", "orcamento_epv"):
        juntar(mc.get(chave))
    for item in (resposta.get("explicacao") or {}).get("itens") or []:
        juntar(item.get("valor_medio"))
        juntar(item.get("contribuicao"))
    for v in (mc.get("sinal_por_variavel") or {}).values():
        juntar(v.get("coef"))
        juntar(v.get("ic95"))

    permitidos: set = set()
    for x in brutos:
        # a forma percentual conta: 0,6461 pode ser citado como 64,61
        for candidato in (x, x * 100):
            for c in CASAS:
                permitidos.add(f"{round(candidato, c):.{c}f}")

    # Numeros que o payload publica DENTRO do proprio texto tambem contam.
    # Sem isto o verificador reprovava "intervalo de 95%": o 95 nao e valor de
    # campo nenhum, mas o payload o escreve em "IC95" e em "percentil
    # 2,5-97,5". Reprovar frase correta e tao grave quanto aprovar frase
    # errada -- apaga informacao boa sem deixar rastro para o leitor.
    for texto in _textos_do_payload(resposta):
        for bruto in PADRAO_NUMERO.findall(texto):
            try:
                v = float(bruto.replace(",", "."))
            except ValueError:
                continue
            for c in CASAS:
                permitidos.add(f"{round(v, c):.{c}f}")
    return permitidos


def numero_bate(bruto: str, permitidos: set) -> bool:
    """O numero tem de bater NA PRECISAO EM QUE FOI ESCRITO.

    A primeira versao aceitava qualquer arredondamento, e isso furava o
    verificador inteiro: `0.9999` passava porque `round(0.9999, 0) == 1` e o
    payload publica `1.0` em `cobertura`. Todo numero perto de um inteiro era
    aprovado -- inclusive um escore inventado.
    """
    texto = bruto.replace(",", ".")
    try:
        v = float(texto)
    except ValueError:
        return False
    casas = len(texto.split(".")[1]) if "." in texto else 0
    if casas > max(CASAS):
        return False
    return f"{round(v, casas):.{casas}f}" in permitidos


def variaveis_citadas(frase: str, itens: list) -> list:
    """Quais variaveis do payload a frase nomeia -- nome, chave ou apelido."""
    alvo = sem_acento(frase)
    achadas = []
    for item in itens:
        chave = item.get("variavel")
        nomes = [chave, item.get("nome", "")] + list(APELIDOS.get(chave, ()))
        if any(n and sem_acento(n) in alvo for n in nomes):
            achadas.append(item)
    return achadas


# ------------------------------------------------------ contexto autorizado

@dataclass
class ContextoAutorizado:
    """Tudo contra o que um claim pode ser resolvido, e nada alem disso.

    Cada campo e uma AUTORIDADE distinta, e elas nao se substituem: um trecho
    de literatura nao autoriza um numero do payload, e o payload nao autoriza
    um DOI. E por isso que sao campos separados e nao um saco de contexto.
    """

    resposta: dict = field(default_factory=dict)
    trechos: list = field(default_factory=list)      # chunks REALMENTE recuperados
    proveniencia: dict | None = None                 # manifesto PROV-lite
    evidencia_visual: dict | None = None
    # A pergunta NAO e autoridade -- ela nao sustenta claim nenhum. Ela viaja
    # aqui porque o gerador de texto precisa saber o que foi perguntado para escrever
    # sobre o assunto certo; o que ele escrever continua sendo verificado
    # contra payload, trecho e manifesto, e nunca contra a pergunta.
    pergunta: str | None = None
    _numeros: set | None = None

    @property
    def numeros(self) -> set:
        if self._numeros is None:
            self._numeros = numeros_permitidos(self.resposta)
        return self._numeros

    @property
    def itens(self) -> list:
        return (self.resposta.get("explicacao") or {}).get("itens") or []

    @property
    def verificadas(self) -> set:
        return set((self.resposta.get("model_card") or {}).get(
            "variaveis_fisicamente_verificadas", []))

    @property
    def sinal(self) -> dict:
        return (self.resposta.get("model_card") or {}).get(
            "sinal_por_variavel") or {}

    def trechos_limpos(self) -> list:
        """Trechos recuperados que NAO tentam instruir o sistema.

        Um chunk com injecao nao e sanitizado nem reescrito: e removido do
        contexto e registrado. Reescrever conteudo suspeito para "deixar
        seguro" e apostar que a reescrita cobre todos os casos.
        """
        return [t for t in self.trechos
                if not texto_tem_injecao(t.get("trecho") or t.get("conteudo") or "")]

    def trechos_com_injecao(self) -> list:
        saida = []
        for t in self.trechos:
            achado = texto_tem_injecao(t.get("trecho") or t.get("conteudo") or "")
            if achado:
                saida.append({**t, "injecao_detectada": achado})
        return saida


# --------------------------------------------------------------- extracao
#
# Extrair NAO e julgar. Esta secao so olha a frase e diz o que ela tentou
# afirmar; todos os claims saem daqui com `estado_verificacao = SEM_SUPORTE`,
# e e a secao seguinte que os resolve. Manter as duas separadas e o que
# permite auditar "o gerador de texto afirmou X e X foi rejeitado" em vez de so
# "a frase caiu".

def extrair_claims(frase: str, ctx: ContextoAutorizado) -> list:
    claims: list = []
    baixa = frase.lower()
    plana = sem_acento(frase)

    # 1-2 numeros -- lidos da frase SEM os identificadores bibliograficos,
    # que tem digitos e nao sao grandezas do payload.
    for bruto in PADRAO_NUMERO.findall(mascarar_identificadores(frase)):
        claims.append(Claim(tipo="claim_numero", predicado="valor_numerico",
                            valor=bruto, origens_declaradas=["payload"]))

    # 5-6 previsao de evento
    for termo in TERMOS_PREVISAO:
        if sem_acento(termo) in plana:
            claims.append(Claim(tipo="claim_previsao",
                                predicado="ocorrencia_futura", valor=termo))
            break
    if PADRAO_PROB_DE_EVENTO.search(plana) or PADRAO_PERCENTUAL_DE_CHANCE.search(plana):
        claims.append(Claim(tipo="claim_previsao",
                            predicado="escore_como_probabilidade_de_evento",
                            valor=frase.strip()))
    achado_seguranca = PADRAO_SEGURANCA.search(plana)
    if achado_seguranca:
        claims.append(Claim(tipo="claim_previsao",
                            predicado="veredito_de_seguranca",
                            valor=achado_seguranca.group(0).strip()))

    # 3-4 maturidade
    for m in MATURIDADES_CONHECIDAS:
        if sem_acento(m) in plana:
            claims.append(Claim(tipo="claim_maturidade",
                                predicado="maturidade_do_modelo", valor=m,
                                origens_declaradas=["payload.maturidade"]))
    # A FLEXAO NAO E DETALHE. O vocabulario servido escreve `validado`, e a
    # prosa escreve "Curitiba esta validADA". Casar so a forma exata deixava
    # passar a promocao mais direta que existe -- em portugues correto e no
    # genero da regiao.
    promocao = PADRAO_VALIDADO.search(plana)
    if promocao and not any(c.tipo == "claim_maturidade" and c.valor == "validado"
                            for c in claims):
        negado = bool(PADRAO_VALIDADO_NEGADO.search(plana))
        claims.append(Claim(
            tipo="claim_maturidade", predicado="maturidade_do_modelo",
            valor="nao_validado" if negado else "validado",
            origens_declaradas=["payload.maturidade"]))

    # status / gate
    for s in STATUS_CONHECIDOS:
        if s in baixa:
            claims.append(Claim(tipo="claim_status",
                                predicado="status_da_resposta", valor=s,
                                origens_declaradas=["payload.status"]))
    gate = ctx.resposta.get("gate_que_nao_fechou")
    if gate and sem_acento(str(gate)) in plana:
        claims.append(Claim(tipo="claim_status",
                            predicado="gate_que_nao_fechou", valor=gate,
                            origens_declaradas=["payload.gate_que_nao_fechou"]))

    # 7 efeito de variavel (qualquer verbo da familia causal)
    if PADRAO_CAUSAL.search(frase):
        citadas = variaveis_citadas(frase, ctx.itens)
        if citadas:
            for item in citadas:
                claims.append(Claim(
                    tipo="claim_variavel", predicado="efeito_de_variavel",
                    valor=item["variavel"],
                    origens_declaradas=["model_card.variaveis_fisicamente_verificadas"]))
        elif ctx.itens and not PADRAO_VISUAL.search(frase):
            claims.append(Claim(tipo="claim_variavel",
                                predicado="efeito_de_variavel", valor=None))

    # 8 direcao -- por oracao, porque uma frase pode dar direcao a varias
    # variaveis de uma vez (a propria frase por regra faz isso).
    if PADRAO_DIRECAO.search(frase):
        oracoes = [o for o in re.split(r"[;,]", frase) if PADRAO_DIRECAO.search(o)]
        oracoes = oracoes or [frase]
        alguma = False
        for oracao in oracoes:
            dito = PADRAO_DIRECAO.search(oracao).group(1).replace(" ", "")
            for item in variaveis_citadas(oracao, ctx.itens):
                alguma = True
                claims.append(Claim(
                    tipo="claim_direcao", predicado="direcao_de_variavel",
                    valor={"variavel": item["variavel"], "sentido": dito},
                    origens_declaradas=["explicacao.itens[].sentido"]))
        if not alguma:
            claims.append(Claim(tipo="claim_direcao",
                                predicado="direcao_de_variavel", valor=None))

    # 9 certeza sobre efeito
    if any(t in plana for t in TERMOS_CERTEZA):
        for item in variaveis_citadas(frase, ctx.itens):
            claims.append(Claim(
                tipo="claim_direcao", predicado="certeza_de_efeito",
                valor=item["variavel"],
                origens_declaradas=["model_card.sinal_por_variavel[].ic95"]))

    # 10-11 evidencia visual
    if PADRAO_VISUAL.search(frase):
        papel = "descritivo"
        if PADRAO_CAUSAL.search(frase):
            papel = "causal"
        if any(v in plana for v in VERBOS_CONFIRMACAO):
            papel = "confirmacao"
        claims.append(Claim(
            tipo="claim_evidencia_visual",
            predicado="papel_da_evidencia_visual", valor=papel,
            origens_declaradas=["evidencia_visual"]))

    # 12-14 fontes
    vistos_fonte = set()

    def fonte(valor: str, origem: str) -> None:
        valor = aparar_identificador(valor)
        if valor and valor.lower() not in vistos_fonte:
            vistos_fonte.add(valor.lower())
            claims.append(Claim(tipo="claim_fonte",
                                predicado="referencia_citada",
                                valor=valor, origens_declaradas=[origem]))

    for doi in PADRAO_DOI.findall(frase):
        fonte(doi, "rag.doi")
    for url in PADRAO_URL.findall(frase):
        fonte(url, "rag.url")
    for autor, ano in (PADRAO_CITACAO_AUTOR_ANO.findall(frase)
                       + PADRAO_CITACAO_NARRATIVA.findall(frase)):
        fonte(f"{autor} {ano}", "rag.titulo")

    # proveniencia
    if PADRAO_PROVENIENCIA.search(plana):
        claims.append(Claim(tipo="claim_proveniencia",
                            predicado="origem_de_artefato",
                            valor=frase.strip(),
                            origens_declaradas=["proveniencia", "model_card"]))

    # suporte temporal -- estado afirmado e causa afirmada, separados
    afirmados = estados_afirmados(plana)
    for estado in afirmados:
        claims.append(Claim(
            tipo="claim_suporte_cientifico", predicado="suporte_temporal",
            valor=estado,
            origens_declaradas=["payload.suporte_cientifico.temporal"]))
    estado_afirmado = afirmados[0] if afirmados else None
    # Metrica isolada sustentando o estado: so quando a frase INFERE o estado
    # a partir dela. A ordem importa -- sem estado afirmado nao ha inferencia
    # de suporte a barrar, e citar AUC continua livre.
    if (estado_afirmado and PADRAO_METRICA.search(plana)
            and PADRAO_INFERENCIA.search(plana)):
        claims.append(Claim(
            tipo="claim_suporte_cientifico",
            predicado="metrica_isolada_como_suporte",
            valor=estado_afirmado,
            origens_declaradas=["payload.suporte_cientifico.temporal"]))
    if PADRAO_LIMITACAO_TEMPORAL.search(plana):
        claims.append(Claim(
            tipo="claim_suporte_cientifico", predicado="limitacao_temporal",
            valor=frase.strip(),
            origens_declaradas=["payload.suporte_cientifico.temporal"]))
    achado_causa = PADRAO_CAUSA_TEMPORAL.search(plana)
    if achado_causa:
        claims.append(Claim(
            tipo="claim_suporte_cientifico",
            predicado="causa_da_degradacao_temporal",
            valor=achado_causa.group(0).strip(),
            origens_declaradas=["payload.suporte_cientifico.temporal"]))

    # limitacao negada / promocao dissimulada
    if PADRAO_SEM_LIMITACAO.search(plana):
        claims.append(Claim(tipo="claim_limitacao",
                            predicado="limitacao_do_modelo",
                            valor="negada", origens_declaradas=["payload.limitacoes"]))

    # Camada social. A pergunta pode ser sensivel; a resposta so pode afirmar
    # classe ou prioridade que o payload social ja declarou. Rotulo territorial
    # oficial e fonte/proveniencia, nao etiqueta para usuario final.
    for m in PADRAO_CLASSE_SOCIAL.findall(frase):
        bruto = next((p for p in m if p), "")
        valor = sem_acento(bruto).replace(" ", "_")
        claims.append(Claim(
            tipo="claim_social", predicado="classe_social_da_area",
            valor=valor, origens_declaradas=[
                "payload.eixos.vulnerabilidade_exposicao.classe_relativa"]))
    for m in PADRAO_PRIORIDADE_PUBLICA.findall(plana):
        claims.append(Claim(
            tipo="claim_social", predicado="prioridade_publica",
            valor=m.lower(), origens_declaradas=[
                "payload.prioridade_publica.classe"]))
    if PADRAO_ROTULO_TERRITORIAL_SENSIVEL.search(plana):
        claims.append(Claim(
            tipo="claim_social", predicado="rotulo_territorial_sensivel",
            valor=frase.strip(), origens_declaradas=[
                "payload.eixos.vulnerabilidade_exposicao.fontes"]))
    return claims


# ------------------------------------------------------------- verificacao

def _verificar_numero(c: Claim, ctx: ContextoAutorizado) -> Claim:
    if numero_bate(str(c.valor), ctx.numeros):
        return c.aprovar("payload")
    return c.rejeitar(
        f"numero {c.valor!r} nao aparece em campo nenhum do payload")


def _verificar_maturidade(c: Claim, ctx: ContextoAutorizado) -> Claim:
    real = ctx.resposta.get("maturidade")
    if c.valor == "nao_validado":
        if real == "validado":
            return c.rejeitar(
                "nega validacao que o payload declara; o texto contradiz o "
                "proprio campo `maturidade`")
        return c.aprovar("payload.maturidade")
    if c.valor == "validado":
        # Regra 4, e ela e absoluta: nenhuma regiao do REV-P esta `validado`,
        # e nao existe payload que autorize a palavra. Nao ha comparacao a
        # fazer aqui -- ha uma promocao a impedir.
        return c.rejeitar(
            "declara maturidade `validado`; nenhuma regiao do REV-P esta "
            "validada e o vocabulario servido nao contem esse estado")
    if real is None:
        return c.sem_apoio("o payload nao declara maturidade nesta resposta")
    if c.valor != real:
        return c.rejeitar(
            f"declara maturidade {c.valor!r}, e o payload declara {real!r}")
    return c.aprovar("payload.maturidade")


def _verificar_status(c: Claim, ctx: ContextoAutorizado) -> Claim:
    if c.predicado == "gate_que_nao_fechou":
        real = ctx.resposta.get("gate_que_nao_fechou")
        if real and sem_acento(str(real)) == sem_acento(str(c.valor)):
            return c.aprovar("payload.gate_que_nao_fechou")
        return c.rejeitar(f"nomeia gate {c.valor!r}, e o payload aponta {real!r}")
    real = ctx.resposta.get("status")
    if real is None:
        return c.sem_apoio("o payload nao declara status")
    if c.valor != real:
        return c.rejeitar(f"declara status {c.valor!r}, e o payload traz {real!r}")
    return c.aprovar("payload.status")


def _verificar_previsao(c: Claim, ctx: ContextoAutorizado) -> Claim:
    if c.predicado == "ocorrencia_futura":
        return c.rejeitar(
            f"afirma previsao de evento ({c.valor!r}); o contrato responde "
            "suscetibilidade do terreno, nao ocorrencia futura")
    if c.predicado == "veredito_de_seguranca":
        status = ctx.resposta.get("status")
        contexto = ("a resposta e uma recusa, que e silencio"
                    if status and status != "ok"
                    else "o payload publica escore relativo de suscetibilidade")
        return c.rejeitar(
            f"declara seguranca ({c.valor!r}); nenhum campo do contrato "
            f"autoriza veredito de seguranca -- {contexto}, e ausencia de "
            "dado nunca significa baixo risco")
    return c.rejeitar(
        "apresenta o escore como probabilidade de o evento acontecer; o "
        "payload define escore como media da probabilidade que o modelo "
        "ajustado atribui aos pontos da area, e nao como chance de alagar")


def _verificar_variavel(c: Claim, ctx: ContextoAutorizado,
                        frase: str = "") -> Claim:
    if c.valor is None:
        return c.rejeitar(
            "afirma causa sem nomear variavel que o payload reconheca")
    if c.valor not in ctx.verificadas:
        return c.rejeitar(
            f"afirma causa sobre {c.valor}, que nao tem sinal fisico "
            "verificado; sobre ela o ajuste nao estabeleceu efeito, e negar "
            "tambem e afirmar")
    if str(c.valor).startswith("rain_"):
        ader = ((ctx.resposta.get("rain_evidence") or {}).get(
            "temporal_adherence") or
            (ctx.resposta.get("model_card") or {}).get("aderencia_temporal") or {})
        if ader.get("risco") == "alto" and PADRAO_CAUSA_FORTE.search(frase):
            return c.rejeitar(
                f"atribui efeito causal a {c.valor}, mas a aderencia temporal "
                "do ajuste e alta: a chuva pode identificar a data do evento "
                "em vez de discriminar lugares")
    return c.aprovar("model_card.variaveis_fisicamente_verificadas")


def _verificar_direcao(c: Claim, ctx: ContextoAutorizado) -> Claim:
    if c.predicado == "certeza_de_efeito":
        info = ctx.sinal.get(c.valor) or {}
        if info.get("ic_cruza_zero"):
            return c.rejeitar(
                f"apresenta como certeza o efeito de {c.valor}, cujo IC95 "
                f"{info.get('ic95')} cruza zero; incerteza declarada no "
                "artefato nao vira certeza na prosa")
        if not info:
            return c.sem_apoio(
                f"nao ha `sinal_por_variavel` para {c.valor} neste payload")
        return c.aprovar("model_card.sinal_por_variavel")
    if c.valor is None:
        return c.rejeitar(
            "afirma direcao causal sem nomear variavel que o payload reconheca")
    variavel, dito = c.valor["variavel"], c.valor["sentido"]
    item = next((i for i in ctx.itens if i["variavel"] == variavel), None)
    if item is None:
        return c.sem_apoio(f"{variavel} nao aparece nos itens da explicacao")
    if variavel not in ctx.verificadas:
        return c.rejeitar(
            f"{variavel} nao tem sinal fisico verificado; afirmar direcao "
            "reabriria o bug de fisica invertida")
    if item["sentido"].replace(" ", "") != dito:
        return c.rejeitar(
            f"diz que {variavel} pesa {dito!r}, e o payload declara "
            f"{item['sentido']!r}")
    return c.aprovar("explicacao.itens")


def _verificar_evidencia_visual(c: Claim, ctx: ContextoAutorizado) -> Claim:
    if c.valor == "causal":
        return c.rejeitar(
            "trata evidencia visual (Sentinel/DINO) como causa; ela e "
            "somente revisao e nunca entra em variavel, gate ou escore")
    if c.valor == "confirmacao":
        return c.rejeitar(
            "trata similaridade visual como confirmacao; semelhanca de "
            "embedding nao confirma inundacao nem valida escore")
    ev = ctx.evidencia_visual or (ctx.resposta.get("evidencia_visual") or None)
    if not ev:
        return c.sem_apoio(
            "cita evidencia visual sem bloco `evidencia_visual` recuperado "
            "nesta resposta")
    return c.aprovar("evidencia_visual")


def _verificar_fonte(c: Claim, ctx: ContextoAutorizado, frase: str = "") -> Claim:
    limpos = ctx.trechos_limpos()
    if not limpos:
        if ctx.trechos:
            return c.rejeitar(
                "a unica fonte recuperada tentava instruir o sistema e foi "
                "descartada do contexto")
        return c.sem_apoio(
            f"cita {c.valor!r} sem nenhum trecho recuperado do corpus; "
            "referencia conhecida 'de memoria' nao e referencia recuperada")
    alvo = sem_acento(str(c.valor))
    encontrados = []
    for t in limpos:
        campos = " ".join(str(t.get(k) or "") for k in
                          ("doi", "url", "titulo", "trecho", "conteudo"))
        if alvo and alvo in sem_acento(campos):
            encontrados.append(t)
    if not encontrados:
        return c.rejeitar(
            f"cita {c.valor!r}, que nao aparece literalmente em nenhum dos "
            f"{len(limpos)} trecho(s) recuperados")

    # Condicao 4: o trecho tem de sustentar A FRASE, e nao apenas existir.
    # Sem isto, um DOI real recuperado de verdade autorizava qualquer
    # afirmacao colada ao lado dele.
    afirmacao = mascarar_identificadores(frase or str(c.valor))
    melhor, apoio = 0.0, None
    for t in encontrados:
        corpo = " ".join(str(t.get(k) or "") for k in
                         ("titulo", "trecho", "conteudo"))
        ok, fracao = sustenta(afirmacao, corpo)
        if fracao > melhor:
            melhor, apoio = fracao, t
        if ok:
            return c.aprovar(t.get("chunk_id") or t.get("source_id") or "corpus")
    return c.rejeitar(
        f"cita {c.valor!r}, que existe e foi recuperada, mas cujo trecho nao "
        f"sustenta a afirmacao (sobreposicao {melhor} com "
        f"{(apoio or {}).get('chunk_id', 'o melhor trecho')})")


def _verificar_proveniencia(c: Claim, ctx: ContextoAutorizado) -> Claim:
    if not ctx.proveniencia:
        return c.sem_apoio(
            "afirma origem de artefato sem manifesto de proveniencia no "
            "contexto")
    artefatos = ctx.proveniencia.get("artefatos") or []
    plana = sem_acento(str(c.valor))
    for a in artefatos:
        nome = str(a.get("id", "")).split("/")[-1]
        if nome and sem_acento(nome) in plana:
            return c.aprovar(a.get("id"))
    for chave in ("estimador", "mecanismo"):
        valor = (ctx.resposta.get("model_card") or {}).get(chave)
        if valor and sem_acento(str(valor)) in plana:
            return c.aprovar(f"model_card.{chave}")
    return c.rejeitar(
        "afirma origem que nao casa com nenhum artefato do manifesto de "
        "proveniencia nem com o cartao do modelo")


def _verificar_suporte_cientifico(c: Claim, ctx: ContextoAutorizado) -> Claim:
    """Resolve contra `payload.suporte_cientifico.temporal`, e so contra ele.

    Duas regras que parecem uma so:

    1. o ESTADO afirmado tem de ser o estado do escopo DESTA resposta. A prosa
       nao pode importar o suporte de outro par regiao x modelo -- "o piloto
       ingles prova que Curitiba e temporalmente estavel" afirma `supported`
       sobre um escopo cujo estado e outro, e cai aqui;
    2. a CAUSA da degradacao nunca e sustentavel. Nenhum desenho do projeto
       separa concept drift de mudanca de forcing, de amostragem e de
       observacao -- a propria autopsia de Curitiba conclui isso. Afirmar a
       causa e sempre alem do payload, mesmo quando a degradacao e real.
    """
    temporal = (ctx.resposta.get("suporte_cientifico") or {}).get("temporal") or {}

    if c.predicado == "causa_da_degradacao_temporal":
        return c.rejeitar(
            f"atribui causa a degradacao temporal ({c.valor!r}); nenhum "
            "desenho do REV-P identifica concept drift ou mudanca climatica "
            "separadamente de mudanca de forcing, de amostragem e de "
            "observacao")

    if c.predicado == "metrica_isolada_como_suporte":
        return c.rejeitar(
            "infere o suporte temporal a partir de uma metrica isolada; o "
            "estado sai de identificabilidade, numero de cortes e intervalo, "
            "e nao de uma AUC -- ha caso no proprio projeto com AUC 0,95 num "
            "corte so e estado `inconclusive`")

    if c.predicado == "limitacao_temporal":
        declaradas = temporal.get("limitations") or []
        if not declaradas:
            return c.sem_apoio(
                "afirma limitacao temporal e o payload nao declara nenhuma "
                "para este par regiao x modelo")
        alvo = sem_acento(str(c.valor))
        for lim in declaradas:
            if sustenta(alvo, sem_acento(str(lim)))[0]:
                return c.aprovar("payload.suporte_cientifico.temporal.limitations")
        return c.rejeitar(
            f"afirma limitacao temporal que nao casa com nenhuma das "
            f"{len(declaradas)} declaradas no payload")

    real = temporal.get("support")
    if real is None:
        return c.sem_apoio(
            "afirma suporte temporal e o payload nao traz avaliacao para este "
            "par regiao x modelo; ausencia de avaliacao nao sustenta nem nega")
    if c.valor != real:
        return c.rejeitar(
            f"declara suporte temporal {c.valor!r}, e o payload declara "
            f"{real!r} para {temporal.get('scope', {}).get('region_id')!r} com "
            f"o modelo {temporal.get('scope', {}).get('model_id')!r}")
    return c.aprovar("payload.suporte_cientifico.temporal")


def _verificar_limitacao(c: Claim, ctx: ContextoAutorizado) -> Claim:
    limitacoes = ctx.resposta.get("limitacoes") or []
    criterios = (ctx.resposta.get("model_card") or {}).get(
        "criterios_nao_atingidos") or []
    return c.rejeitar(
        f"nega limitacoes que o payload declara ({len(limitacoes)} limitacao(oes), "
        f"{len(criterios)} criterio(s) nao atingido(s)); nenhuma regiao do "
        "REV-P esta validada")


def _verificar_social(c: Claim, ctx: ContextoAutorizado) -> Claim:
    social = (ctx.resposta.get("eixos") or {}).get(
        "vulnerabilidade_exposicao") or {}
    prioridade = ctx.resposta.get("prioridade_publica") or {}

    if c.predicado == "rotulo_territorial_sensivel":
        return c.rejeitar(
            "rotula diretamente a area como favela/comunidade urbana; a "
            "tipologia oficial pode aparecer como fonte ou indicador agregado, "
            "nao como etiqueta de usuario")

    if c.predicado == "classe_social_da_area":
        real = social.get("classe_relativa")
        if real is None:
            return c.sem_apoio(
                "afirma classe social sem payload social autorizado")
        if c.valor not in CLASSES_SOCIAIS_CONHECIDAS:
            return c.rejeitar(f"classe social desconhecida: {c.valor!r}")
        if c.valor != real:
            return c.rejeitar(
                f"declara vulnerabilidade/exposicao {c.valor!r}, e o payload "
                f"social declara {real!r}")
        return c.aprovar(
            "payload.eixos.vulnerabilidade_exposicao.classe_relativa")

    if c.predicado == "prioridade_publica":
        real = prioridade.get("classe")
        if real is None:
            return c.sem_apoio(
                "afirma prioridade publica sem payload social autorizado")
        if c.valor not in PRIORIDADES_PUBLICAS_CONHECIDAS:
            return c.rejeitar(f"prioridade publica desconhecida: {c.valor!r}")
        if c.valor != real:
            return c.rejeitar(
                f"declara prioridade {c.valor!r}, e o payload social declara "
                f"{real!r}")
        return c.aprovar("payload.prioridade_publica.classe")

    return c.sem_apoio(f"predicado social sem verificador: {c.predicado}")


_VERIFICADORES = {
    "claim_numero": _verificar_numero,
    "claim_maturidade": _verificar_maturidade,
    "claim_status": _verificar_status,
    "claim_previsao": _verificar_previsao,
    "claim_variavel": _verificar_variavel,
    "claim_direcao": _verificar_direcao,
    "claim_evidencia_visual": _verificar_evidencia_visual,
    "claim_fonte": _verificar_fonte,
    "claim_proveniencia": _verificar_proveniencia,
    "claim_limitacao": _verificar_limitacao,
    "claim_suporte_cientifico": _verificar_suporte_cientifico,
    "claim_social": _verificar_social,
}

# Ordem de severidade para escolher O motivo que a frase carrega. Nao muda o
# veredito (qualquer claim nao verificado ja derruba a frase); muda o que a
# auditoria mostra primeiro, e por isso a fronteira cientifica vem antes do
# detalhe numerico.
SEVERIDADE = (
    "claim_previsao", "claim_maturidade", "claim_social",
    "claim_suporte_cientifico",
    "claim_direcao", "claim_variavel", "claim_evidencia_visual",
    "claim_fonte", "claim_proveniencia", "claim_limitacao", "claim_numero",
    "claim_status",
)


def verificar_claims(claims: list, ctx: ContextoAutorizado,
                    frase: str = "") -> list:
    """Resolve cada claim contra as origens autorizadas, sozinho.

    O verificador NAO le `origens_declaradas` para decidir: ele vai a origem
    que o TIPO do claim exige. `origens_declaradas` fica no registro para que
    a auditoria possa comparar o que o gerador de texto alegou com o que foi de fato
    encontrado -- que e uma informacao diferente, e util.
    """
    for c in claims:
        verificador = _VERIFICADORES.get(c.tipo)
        if verificador is None:
            c.sem_apoio(f"nao ha verificador para {c.tipo}")
            continue
        try:
            if c.tipo in ("claim_fonte", "claim_variavel"):
                verificador(c, ctx, frase)
            else:
                verificador(c, ctx)
        except Exception as e:                  # verificador que quebra fecha
            c.sem_apoio(f"verificacao falhou: {type(e).__name__}: {e}")
    return claims


@dataclass
class Veredito:
    aprovada: bool
    motivo: str
    claims: list

    def para_dict(self) -> dict:
        return {"aprovada": self.aprovada, "motivo": self.motivo or None,
                "claims": [c.para_dict() for c in self.claims]}


def avaliar_frase(frase: str, ctx: ContextoAutorizado) -> Veredito:
    """A unidade de publicacao continua sendo a frase; a de prova, o claim."""
    claims = verificar_claims(extrair_claims(frase, ctx), ctx, frase)
    barrados = [c for c in claims if not c.publicavel]
    if not barrados:
        return Veredito(True, "", claims)
    ordem = {t: i for i, t in enumerate(SEVERIDADE)}
    pior = min(barrados, key=lambda c: ordem.get(c.tipo, 99))
    return Veredito(False, pior.motivo_rejeicao or f"{pior.tipo} sem suporte",
                    claims)


def resumo_de_claims(claims: list) -> dict:
    """Contagem por tipo e estado, para relatorio e benchmark."""
    resumo = {"total": len(claims), "publicaveis": 0, "por_tipo": {},
              "por_estado": {e: 0 for e in ESTADOS}}
    for c in claims:
        resumo["por_estado"][c.estado_verificacao] += 1
        alvo = resumo["por_tipo"].setdefault(
            c.tipo, {e: 0 for e in ESTADOS})
        alvo[c.estado_verificacao] += 1
        if c.publicavel:
            resumo["publicaveis"] += 1
    return resumo
