"""A regua da cadeia neural: nenhum veredito sem o poder que o sustenta.

-------------------------------------------------------------------------
O ERRO QUE ESTE ARQUIVO IMPEDE
-------------------------------------------------------------------------

Tres experimentos da frente neural foram lidos como refutacao:

    AOIs                 +0,0182 / -0,0020 / +0,0972   "sinal_consistente: false"
    amostragem por area  -0,0197 / -0,0253 / -0,0746   "0/3 sementes"
    escala do pote       ganho nao monotono            "mais dado nao ajuda"

Nenhum deles podia refutar coisa alguma. Com tres sementes e o desvio que
aquelas mesmas corridas mediram, o menor efeito detectavel era de 2,3 a 4,2
vezes MAIOR que o efeito em questao. O experimento nao respondeu "nao"; ele
respondeu "nao sei", e a diferenca foi perdida no caminho entre o JSON e a
conclusao.

O custo disso e assimetrico e por isso ele importa: um "nao decidivel" lido
como "refutado" ENCERRA a linha de investigacao. Foi assim que "mais terreno
nao melhora a representacao" virou conclusao sobre o dado quando era
propriedade do desenho.

-------------------------------------------------------------------------
A REGRA, e ela e um portao e nao uma sugestao
-------------------------------------------------------------------------

Existem TRES vereditos, e `INDECIDIVEL` e o padrao:

  EFEITO_DETECTADO      o teste de POSTOS pareado (Wilcoxon) rejeita a
                        hipotese nula. So aqui se afirma ganho. O teste e de
                        postos e nao t porque este instrumento tem cauda
                        pesada documentada -- ver o bloco TESTE_PRIMARIO
                        abaixo e `docs/PRE_REGISTRO_TESTE_DE_POSTOS_2026_09_04`.
                        Exige no minimo 6 sementes: abaixo disso o menor p
                        possivel ja e maior que 0,05.

  AUSENCIA_DEMONSTRADA  o intervalo inteiro cabe dentro de +-`relevancia`.
                        Isto e equivalencia, e exige uma `relevancia`
                        DECLARADA ANTES -- "que diferenca seria grande o
                        bastante para mudar o que eu faco?". Sem esse numero
                        nao existe demonstracao de ausencia, existe so ausencia
                        de demonstracao.

  INDECIDIVEL           todo o resto, acompanhado do MDE e de quantas sementes
                        faltam para decidir.

`exigir_relevancia_declarada` levanta erro quando alguem tenta concluir
ausencia sem o numero. E portao e nao aviso pela mesma razao de sempre no
REV-P: o modo de falhar aqui e silencioso e a conclusao errada sobrevive por
meses.

-------------------------------------------------------------------------
O QUE ESTE ARQUIVO NAO FAZ
-------------------------------------------------------------------------

Nao promove nada e nao escolhe modelo -- isso e a catraca. Ele so responde
"este numero decide alguma coisa?", e responde do mesmo jeito para resultado
favoravel e desfavoravel. O experimento do pre-treino, que e o resultado
POSITIVO da frente, sai daqui como t=2,14 e p=0,166 com 3 sementes: ele nao
podia ser positivo de jeito nenhum, porque com 3 sementes nem o teste de sinal
alcanca 0,05. O que se sustenta dele e a reducao de amplitude (2,86x), que e
afirmacao sobre variancia e nao passa por este teste.
"""

from __future__ import annotations

import math
from statistics import median

from scipy import stats

# O TESTE PRIMARIO E DE POSTOS, e nao o t. A razao esta medida.
#
# Esta regua mede perda de reconstrucao, e esse instrumento tem cauda pesada
# DOCUMENTADA: duas corridas divergiram e ficaram registradas -- 1,1935 contra
# um nivel de 0,75, e 4,5631 contra um nivel de 0,9. Uma linhagem que diverge
# assim nao e "um resultado ruim": e uma corrida que falhou mecanicamente, e o
# t pareado nao tem defesa nenhuma contra a MAGNITUDE dela.
#
# O caso concreto: no braco de pote crescente, as seis sementes melhoraram
# (+0,110 a +0,311, e a divergente +3,865). O t deu p=0,233 e INDECIDIVEL --
# porque a divergente inflou o desvio para 1,49. Wilcoxon, que so olha sinal e
# ordem, deu p=0,031.
#
# Entao a promocao passa a depender do teste de postos, e o t continua ao lado
# para a contabilidade de poder (MDE, sementes necessarias), que e o que o
# teste de postos nao fornece.
TESTE_PRIMARIO = "wilcoxon_pareado"

# Com n sementes, o menor p possivel num teste de sinal e 2 * 0,5^n. Abaixo de
# 6 sementes, NENHUM teste distribution-free chega a 0,05 -- nem com todas as
# sementes concordando. Os experimentos de 3 sementes da frente neural eram,
# por construcao, incapazes de veredito positivo por esse caminho.
SEMENTES_MINIMAS = 6

EFEITO_DETECTADO = "EFEITO_DETECTADO"
AUSENCIA_DEMONSTRADA = "AUSENCIA_DEMONSTRADA"
INDECIDIVEL = "INDECIDIVEL"
VEREDITOS = (EFEITO_DETECTADO, AUSENCIA_DEMONSTRADA, INDECIDIVEL)

PODER_PADRAO = 0.80
ALFA_PADRAO = 0.05


class VereditoSemPoder(Exception):
    """Tentativa de concluir ausencia sem declarar o que seria relevante."""


def exigir_relevancia_declarada(relevancia: float | None) -> float:
    """O portao. Ausencia so pode ser concluida contra um tamanho declarado."""
    if relevancia is None or relevancia <= 0:
        raise VereditoSemPoder(
            "concluir ausencia exige `relevancia` > 0 declarada antes da "
            "medida: sem ela o resultado e INDECIDIVEL, e chamar isso de "
            "refutacao encerra a investigacao com base no desenho e nao no "
            "dado")
    return float(relevancia)


def mde(desvio: float, n: int, poder: float = PODER_PADRAO,
        alfa: float = ALFA_PADRAO) -> float:
    """Menor efeito detectavel do teste t pareado, aproximado.

    `(t_{1-alfa/2} + t_{poder}) * s / sqrt(n)`, com `n-1` graus de liberdade.
    E aproximacao: o teste exato usa t nao-central, e o valor sai um pouco
    otimista. Fica declarado em vez de escondido, porque a decisao que ele
    apoia -- "vale a pena rodar mais sementes?" -- nao muda na terceira casa.
    """
    if n < 2 or desvio <= 0:
        return float("inf")
    gl = n - 1
    return float((stats.t.ppf(1 - alfa / 2, gl) + stats.t.ppf(poder, gl))
                 * desvio / math.sqrt(n))


def sementes_necessarias(desvio: float, efeito_alvo: float,
                         poder: float = PODER_PADRAO,
                         alfa: float = ALFA_PADRAO, teto: int = 200) -> int:
    """Quantas sementes o efeito-alvo exigiria. O preco, antes de gastar.

    Este e o numero acionavel do arquivo: numa maquina que treina em CPU, saber
    de antemao que a resposta custa 40 corridas evita descobrir isso depois de
    tres e chamar o resultado de nulo.
    """
    efeito_alvo = abs(efeito_alvo)
    if efeito_alvo <= 0 or desvio <= 0:
        return 0
    for n in range(2, teto + 1):
        if mde(desvio, n, poder, alfa) <= abs(efeito_alvo):
            return n
    return teto + 1


def _p_de_postos(difs) -> float | None:
    """Wilcoxon pareado. Olha sinal e ordem, nunca magnitude.

    E por isso que ele sobrevive a uma linhagem que divergiu: uma diferenca de
    +3,865 no meio de outras cinco entre +0,110 e +0,311 vale, para ele,
    exatamente o mesmo que "a maior das seis".
    """
    nao_nulos = [d for d in difs if d != 0]
    if len(nao_nulos) < 1:
        return 1.0
    try:
        return float(stats.wilcoxon(nao_nulos).pvalue)
    except ValueError:
        return None


def _p_de_sinal(difs) -> float | None:
    """Teste de sinal binomial -- o piso do que se pode afirmar.

    Mais fraco que Wilcoxon e imune ate a ordem. Fica no relatorio porque e o
    numero que torna obvio, sem estatistica nenhuma, que 3 sementes nao
    decidem: com 3, o melhor p possivel e 0,25.
    """
    nao_nulos = [d for d in difs if d != 0]
    if not nao_nulos:
        return 1.0
    positivos = sum(1 for d in nao_nulos if d > 0)
    return float(stats.binomtest(positivos, len(nao_nulos), 0.5).pvalue)


def _divergentes(difs, fator: float = 5.0) -> list[int]:
    """Indices cuja magnitude e multiplo grosseiro da mediana das magnitudes.

    Nao serve para excluir nada -- serve para que uma corrida que falhou
    mecanicamente apareca no registro em vez de so alargar o desvio em
    silencio.
    """
    magnitudes = sorted(abs(d) for d in difs)
    if not magnitudes:
        return []
    meio = median(magnitudes)
    if meio <= 0:
        return []
    return [i for i, d in enumerate(difs) if abs(d) > fator * meio]


def analisar_pareado(difs, relevancia: float | None = None,
                     poder: float = PODER_PADRAO,
                     alfa: float = ALFA_PADRAO) -> dict:
    """A diferenca pareada por semente, com o veredito que ela sustenta.

    Pareado porque o desenho e pareado: a mesma semente roda os dois bracos, e
    a variabilidade que sobra entre bracos e menor que a variabilidade entre
    sementes. Usar a amplitude entre sementes como ruido do efeito -- que foi o
    que os relatorios anteriores fizeram -- superestima o ruido e esconde
    efeito real.
    """
    difs = [float(d) for d in difs]
    n = len(difs)
    if n < 2:
        return {"n": n, "veredito": INDECIDIVEL,
                "por_que": "menos de duas sementes nao tem desvio",
                "diferencas": difs}

    media = sum(difs) / n
    desvio = math.sqrt(sum((d - media) ** 2 for d in difs) / (n - 1))
    gl = n - 1
    erro = desvio / math.sqrt(n) if desvio > 0 else 0.0
    t = media / erro if erro > 0 else math.copysign(math.inf, media or 1.0)
    p = float(2 * stats.t.sf(abs(t), gl)) if erro > 0 else 0.0
    largura = float(stats.t.ppf(1 - alfa / 2, gl)) * erro
    ic = (media - largura, media + largura)
    detectavel = mde(desvio, n, poder, alfa)

    p_postos = _p_de_postos(difs)
    p_sinal = _p_de_sinal(difs)
    positivos = sum(1 for d in difs if d > 0)
    divergentes = _divergentes(difs)

    if p_postos is not None and p_postos < alfa and media != 0:
        veredito = EFEITO_DETECTADO
        por_que = (f"teste de postos pareado p={p_postos:.4f}: "
                   f"{positivos}/{n} sementes no mesmo sentido")
    elif relevancia is not None and relevancia > 0 and abs(ic[0]) < relevancia \
            and abs(ic[1]) < relevancia:
        veredito = AUSENCIA_DEMONSTRADA
        por_que = (f"o IC inteiro cabe dentro de +-{relevancia:g}, que foi "
                   f"declarado como o menor efeito que mudaria a decisao")
    elif n < SEMENTES_MINIMAS:
        veredito = INDECIDIVEL
        por_que = (f"{n} sementes: o menor p possivel num teste de sinal e "
                   f"{2 * 0.5 ** n:.4f}, entao nenhum resultado aqui pode "
                   f"chegar a {alfa}")
    else:
        veredito = INDECIDIVEL
        por_que = (f"teste de postos p={p_postos}; o efeito medido "
                   f"({media:.4f}) e menor que o menor detectavel com {n} "
                   f"sementes ({detectavel:.4f})")

    return {
        "n": n,
        "diferencas": [round(d, 6) for d in difs],
        "media": round(media, 6),
        "mediana": round(median(difs), 6),
        "desvio": round(desvio, 6),
        "erro_padrao": round(erro, 6),
        "teste_primario": TESTE_PRIMARIO,
        "p_postos": round(p_postos, 5) if p_postos is not None else None,
        "p_sinal": round(p_sinal, 5) if p_sinal is not None else None,
        "sementes_no_mesmo_sentido": f"{positivos}/{n}",
        # As linhagens que divergiram ficam NOMEADAS em vez de removidas.
        # Remover seria escolher os dados depois de ve-los; o teste de postos
        # ja neutraliza a magnitude delas, e o registro deixa a corrida
        # quebrada visivel para quem for investigar a causa.
        "divergentes": divergentes,
        # O t continua aqui pela contabilidade de poder, e nao pelo veredito.
        "t": round(t, 4) if math.isfinite(t) else None,
        "p": round(p, 4),
        "veredito_parametrico": (EFEITO_DETECTADO if p < alfa
                                 else INDECIDIVEL),
        "ic95": [round(ic[0], 6), round(ic[1], 6)],
        "mde": round(detectavel, 6),
        "razao_mde_efeito": (round(detectavel / abs(media), 2)
                             if media else None),
        # `abs`: um efeito negativo custa as mesmas sementes que o positivo
        # de mesmo tamanho. Sem isso, todo resultado desfavoravel voltava
        # "precisa de 0 sementes", que e a resposta que faria a linha ser
        # encerrada de novo pelo mesmo motivo de antes.
        "sementes_para_decidir": sementes_necessarias(desvio, abs(media),
                                                      poder, alfa)
                                 if media else None,
        "sinais_iguais": len({d > 0 for d in difs if d != 0}) <= 1,
        "poder": poder,
        "alfa": alfa,
        "relevancia_declarada": relevancia,
        "veredito": veredito,
        "por_que": por_que,
        "nao_e": ("INDECIDIVEL nao e ausencia de efeito; e ausencia de "
                  "resolucao. Tratar um como o outro foi o que encerrou tres "
                  "linhas de investigacao da frente neural"),
    }


def concluir_ausencia(difs, relevancia: float | None,
                      poder: float = PODER_PADRAO,
                      alfa: float = ALFA_PADRAO) -> dict:
    """Caminho unico para afirmar "nao ha efeito". Passa pelo portao.

    Existe separado de `analisar_pareado` de proposito: quem quer afirmar
    ausencia tem de chamar a funcao que cobra a `relevancia`, e nao pode
    chegar la por descuido lendo o campo `veredito` de outra.
    """
    exigir_relevancia_declarada(relevancia)
    return analisar_pareado(difs, relevancia=relevancia, poder=poder,
                            alfa=alfa)


def limiar_da_catraca(difs, alfa: float = ALFA_PADRAO) -> float:
    """O ganho minimo que a catraca aceita: o limite superior do ruido.

    Quando ha pares, a catraca decide pelo IC da diferenca -- e este numero e a
    largura dele. Quando ha uma corrida so, ele vira o limiar fixo da arena, e
    a corrida unica precisa bater esse valor para promover.
    """
    a = analisar_pareado(difs, alfa=alfa)
    if a["n"] < 2:
        return float("inf")
    return round(abs(a["media"] - a["ic95"][0]), 6)
