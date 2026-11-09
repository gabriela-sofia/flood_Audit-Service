"""D1 -- os casos que medem a quarta condicao da cadeia de citacao.

-------------------------------------------------------------------------
O LIMITE QUE O PROPRIO CODIGO JA DECLARAVA
-------------------------------------------------------------------------

`claims.sustenta` diz de si mesma:

    grosseira de proposito e declarada como tal. Nao entende o texto, e nao
    finge entender. Pega o caso que importa -- citar uma fonte que nao tem nada
    a ver com a afirmacao -- e deixa passar parafrase habilidosa.

A frente D existe para medir esse "deixa passar". Nao para consertar antes de
medir: baseline primeiro, como nas tres frentes anteriores.

-------------------------------------------------------------------------
DE ONDE VEM O GABARITO
-------------------------------------------------------------------------

Os trechos sao REAIS -- copiados do corpus, das referencias que ja estao
indexadas. Os claims foram escritos a mao para este conjunto, e isso fica
declarado: nao ha como gerar automaticamente um par cujo lexico casa e o
sentido nao, porque essa e exatamente a relacao que nenhum metodo lexico
consegue construir.

E o mesmo estatuto dos 18 casos adversariais do REV-P-EVAL, que tambem foram
escritos a mao. Um conjunto assim nao mede populacao; mede se a barreira pega
os modos de falha que alguem conseguiu imaginar.

-------------------------------------------------------------------------
QUATRO FAMILIAS, e as duas do meio sao as que importam
-------------------------------------------------------------------------

  sustenta_direto      o trecho diz o que o claim afirma, com as mesmas
                       palavras. O piso: se falhar aqui, o metodo nao serve.

  sustenta_parafrase   o trecho diz o que o claim afirma, com OUTRAS palavras.
                       Falso negativo aqui e fonte certa reprovada -- e uma
                       fonte certa reprovada e tao real quanto uma errada
                       aprovada.

  nao_sustenta_lexico  o lexico casa muito e o sentido nao. Direcao invertida,
                       numero trocado, generalizacao indevida. E a "parafrase
                       habilidosa" que a docstring admite deixar passar.

  nao_sustenta_obvio   o trecho nao tem nada a ver. O caso que o metodo atual
                       ja pega, mantido para provar que nada regrediu.
"""

from __future__ import annotations

# Trechos reais, copiados do corpus indexado. A chave e o `source_id` de onde
# eles vieram, para que a auditoria possa conferir a copia.
TRECHOS = {
    "tarboton_dinf": (
        "D-infinity representa direcao de fluxo como angulo continuo e reparte "
        "o fluxo entre os dois pixels vizinhos que cercam esse angulo, em vez "
        "de mandar tudo para um unico vizinho como o D8."),
    "renno_hand": (
        "HAND e um descritor topografico que da, para cada celula, a altura "
        "acima da drenagem mais proxima seguindo o caminho de fluxo; e "
        "descritor de terreno, nao previsao de inundacao."),
    "beven_twi": (
        "O indice topografico combina area de contribuicao a montante e "
        "declividade local para descrever a propensao de uma celula a saturar."),
    "firth_1993": (
        "Penalizacao de Jeffreys reduz o vies de primeira ordem da maxima "
        "verossimilhanca e produz estimativas finitas mesmo sob separacao."),
    "cutter_sovi": (
        "Vulnerabilidade social a perigos ambientais pode ser medida por "
        "indicadores socioeconomicos e demograficos agregados, mas os fatores "
        "relevantes variam por lugar; nao existe um conjunto unico valido em "
        "qualquer territorio."),
    "rufat_2015": (
        "Revisao de estudos de vulnerabilidade social a enchentes: demografia, "
        "status socioeconomico e saude aparecem como drivers frequentes, "
        "enquanto percepcao de risco e capacidade de resposta tambem importam "
        "e sao menos capturadas pelos indices."),
    "peduzzi_epv": (
        "Simulacao com regressao logistica mostra que menos de dez eventos por "
        "variavel produz vies e cobertura de intervalo abaixo do nominal."),
}

SUSTENTA_DIRETO = "sustenta_direto"
SUSTENTA_PARAFRASE = "sustenta_parafrase"
NAO_SUSTENTA_LEXICO = "nao_sustenta_lexico"
NAO_SUSTENTA_OBVIO = "nao_sustenta_obvio"
FAMILIAS = (SUSTENTA_DIRETO, SUSTENTA_PARAFRASE,
            NAO_SUSTENTA_LEXICO, NAO_SUSTENTA_OBVIO)

CASOS = [
    # ------------------------------------------------ sustenta, direto
    {"id": "dinf_direto", "familia": SUSTENTA_DIRETO, "trecho": "tarboton_dinf",
     "claim": "D-infinity reparte o fluxo entre os dois pixels vizinhos que "
              "cercam o angulo, em vez de mandar tudo para um unico vizinho.",
     "sustenta": True},
    {"id": "hand_direto", "familia": SUSTENTA_DIRETO, "trecho": "renno_hand",
     "claim": "HAND da a altura acima da drenagem mais proxima seguindo o "
              "caminho de fluxo.",
     "sustenta": True},
    {"id": "twi_direto", "familia": SUSTENTA_DIRETO, "trecho": "beven_twi",
     "claim": "O indice topografico combina area de contribuicao a montante e "
              "declividade local.",
     "sustenta": True},
    {"id": "sovi_direto", "familia": SUSTENTA_DIRETO, "trecho": "cutter_sovi",
     "claim": "Os fatores relevantes de vulnerabilidade social variam por "
              "lugar.",
     "sustenta": True},

    # -------------------------------------------- sustenta, parafrase
    # Mesma afirmacao, vocabulario deslocado. Reprovar aqui e reprovar a fonte
    # certa, que e um erro tao real quanto aprovar a errada.
    {"id": "dinf_parafrase", "familia": SUSTENTA_PARAFRASE,
     "trecho": "tarboton_dinf",
     "claim": "O metodo divide a drenagem entre celulas adjacentes em vez de "
              "concentrar toda a descarga numa so.",
     "sustenta": True},
    {"id": "hand_parafrase", "familia": SUSTENTA_PARAFRASE,
     "trecho": "renno_hand",
     "claim": "A medida informa o desnivel de cada ponto ate o curso d'agua "
              "que o drena, e nao antecipa alagamento.",
     "sustenta": True},
    {"id": "firth_parafrase", "familia": SUSTENTA_PARAFRASE,
     "trecho": "firth_1993",
     "claim": "A penalizacao produz coeficientes finitos mesmo quando as "
              "classes se separam perfeitamente.",
     "sustenta": True},
    {"id": "epv_parafrase", "familia": SUSTENTA_PARAFRASE,
     "trecho": "peduzzi_epv",
     "claim": "Com poucos casos por preditor, o intervalo de confianca cobre "
              "menos do que promete.",
     "sustenta": True},

    # -------------------------------------- NAO sustenta, lexico casa
    # A "parafrase habilidosa" que a docstring de `sustenta` admite deixar
    # passar: quase todas as palavras do claim estao no trecho, e a afirmacao
    # e outra.
    {"id": "dinf_invertido", "familia": NAO_SUSTENTA_LEXICO,
     "trecho": "tarboton_dinf",
     "claim": "D-infinity manda todo o fluxo para um unico pixel vizinho, como "
              "o D8 faz com o angulo continuo.",
     "sustenta": False},
    {"id": "hand_como_previsao", "familia": NAO_SUSTENTA_LEXICO,
     "trecho": "renno_hand",
     "claim": "HAND e uma previsao de inundacao para cada celula, calculada "
              "pela altura acima da drenagem mais proxima.",
     "sustenta": False},
    {"id": "twi_causal", "familia": NAO_SUSTENTA_LEXICO, "trecho": "beven_twi",
     "claim": "A declividade local e a area de contribuicao a montante "
              "determinam que a celula vai saturar.",
     "sustenta": False},
    {"id": "sovi_universal", "familia": NAO_SUSTENTA_LEXICO,
     "trecho": "cutter_sovi",
     "claim": "Existe um conjunto unico de indicadores socioeconomicos e "
              "demograficos valido para medir vulnerabilidade social em "
              "qualquer territorio.",
     "sustenta": False},
    {"id": "rufat_completo", "familia": NAO_SUSTENTA_LEXICO,
     "trecho": "rufat_2015",
     "claim": "Os indices de vulnerabilidade social a enchentes capturam bem "
              "percepcao de risco e capacidade de resposta.",
     "sustenta": False},
    {"id": "firth_elimina", "familia": NAO_SUSTENTA_LEXICO,
     "trecho": "firth_1993",
     "claim": "A penalizacao de Jeffreys elimina o vies da maxima "
              "verossimilhanca sob separacao.",
     "sustenta": False},
    {"id": "epv_invertido", "familia": NAO_SUSTENTA_LEXICO,
     "trecho": "peduzzi_epv",
     "claim": "Menos de dez eventos por variavel produz cobertura de intervalo "
              "acima do nominal e nenhum vies.",
     "sustenta": False},

    # ------------------------------------------ NAO sustenta, obvio
    # O metodo atual ja pega estes. Ficam para provar que nada regrediu.
    {"id": "tarboton_curitiba", "familia": NAO_SUSTENTA_OBVIO,
     "trecho": "tarboton_dinf",
     "claim": "Curitiba esta validada como regiao servida do REV-P.",
     "sustenta": False},
    {"id": "hand_renda", "familia": NAO_SUSTENTA_OBVIO, "trecho": "renno_hand",
     "claim": "A renda media dos setores censitarios de Recife caiu no ultimo "
              "censo.",
     "sustenta": False},
    {"id": "sovi_dinf", "familia": NAO_SUSTENTA_OBVIO, "trecho": "cutter_sovi",
     "claim": "O algoritmo D-infinity reparte fluxo entre pixels vizinhos.",
     "sustenta": False},
]


def caso_completo(caso: dict) -> dict:
    return {**caso, "texto_do_trecho": TRECHOS[caso["trecho"]]}


def todos() -> list[dict]:
    return [caso_completo(c) for c in CASOS]


def por_familia() -> dict:
    return {f: [caso_completo(c) for c in CASOS if c["familia"] == f]
            for f in FAMILIAS}
