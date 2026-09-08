"""A regua comum das quatro frentes: risco contra cobertura.

-------------------------------------------------------------------------
O ENQUADRAMENTO QUE ESTAVA NO CORPUS E EU NAO USEI
-------------------------------------------------------------------------

El-Yaniv e Wiener (2010), ja indexado como `cientifica` no corpus do REV-P:

    "Formaliza predicao seletiva: o classificador pode se abster, trocando
     cobertura por risco menor na regiao onde responde."

As quatro frentes fazem exatamente isso, e eu medi cada uma com uma regua
diferente:

  A  recusa o chip quando a margem e baixa          -> media IoU no que sobra
  B  (nao tinha abstencao)                           -> Spearman
  C  devolve lista vazia quando nada casa            -> recall@k
  D  derruba a frase quando o juiz discorda          -> acuracia

Sao quatro seletores. E um seletor NAO se avalia pela qualidade media do que
ele deixa passar -- essa metrica premia recusar tudo menos o caso mais facil.

-------------------------------------------------------------------------
O ERRO QUE ISTO CORRIGE, e ele e meu
-------------------------------------------------------------------------

Na frente A eu conclui que "a recusa nao paga" porque o ganho de IoU dentro de
um orcamento de 10% de cobertura foi nulo. O criterio estava errado em dois
sentidos.

Primeiro: eu nunca perguntei se a confianca ORDENAVA melhor que o acaso. Um
seletor so existe se a ordem que ele impoe tiver informacao; recusar ao acaso
tambem "melhora" a media do que sobra em qualquer conjunto com variancia.

Segundo: comparar dois seletores num unico ponto de cobertura e comparar duas
curvas por um pixel. A medida correta e a curva inteira -- e a area sob ela.

-------------------------------------------------------------------------
AS TRES MEDIDAS
-------------------------------------------------------------------------

  curva        para cada cobertura c, o risco entre os c fracao mais confiantes
  AURC         area sob essa curva; MENOR e melhor
  ganho_sobre_acaso   AURC do seletor contra AURC de ordenacao aleatoria

A terceira e a que responde "a confianca e informativa?", e ela e a pergunta
que separa um seletor de um corte arbitrario. Sem ela, qualquer numero de
cobertura parcial parece bom.
"""

from __future__ import annotations

import numpy as np


def curva_risco_cobertura(confianca, erro) -> dict:
    """Risco acumulado ao percorrer as amostras da mais confiante para a menos.

    `erro` e 1 quando a decisao esta errada e 0 quando esta certa -- pode ser
    binario ou continuo (um IoU invertido, por exemplo). O risco em cobertura
    `c` e a media do erro entre as `c` fracoes mais confiantes.

    Empates sao ordenados de forma estavel: um seletor que nao distingue dois
    casos nao pode ser premiado por uma ordem que ele nao produziu.
    """
    confianca = np.asarray(confianca, dtype=np.float64)
    erro = np.asarray(erro, dtype=np.float64)
    if confianca.shape != erro.shape or confianca.size == 0:
        raise ValueError("confianca e erro precisam ter o mesmo tamanho nao vazio")

    ordem = np.argsort(-confianca, kind="stable")
    erro_ordenado = erro[ordem]
    n = len(erro_ordenado)
    acumulado = np.cumsum(erro_ordenado)
    indices = np.arange(1, n + 1)
    return {
        "cobertura": (indices / n).tolist(),
        "risco": (acumulado / indices).tolist(),
        "n": n,
        "risco_em_cobertura_total": round(float(erro.mean()), 6),
    }


def aurc(confianca, erro) -> float:
    """Area sob a curva risco-cobertura. Menor e melhor.

    Integrada por trapezio sobre a cobertura, que e o padrao da literatura de
    predicao seletiva. Um classificador perfeito tem AURC 0; um que nao ordena
    nada tem AURC igual ao risco global.
    """
    c = curva_risco_cobertura(confianca, erro)
    return round(float(np.trapezoid(c["risco"], c["cobertura"])), 6)


def aurc_do_acaso(erro, repeticoes: int = 200, seed: int = 20260902) -> dict:
    """AURC de ordenacao aleatoria -- o piso contra o qual todo seletor mede.

    Existe porque recusar ao acaso TAMBEM melhora a media do que sobra em
    qualquer conjunto com variancia. Sem esta referencia, um seletor inutil
    parece util.
    """
    erro = np.asarray(erro, dtype=np.float64)
    rng = np.random.default_rng(seed)
    valores = [aurc(rng.permutation(len(erro)).astype(np.float64), erro)
               for _ in range(repeticoes)]
    return {
        "media": round(float(np.mean(valores)), 6),
        "p05": round(float(np.percentile(valores, 5)), 6),
        "p95": round(float(np.percentile(valores, 95)), 6),
        "repeticoes": repeticoes,
    }


def risco_em(confianca, erro, cobertura_alvo: float) -> float | None:
    """Risco numa cobertura fixa -- util para comparar num ponto declarado."""
    c = curva_risco_cobertura(confianca, erro)
    for cob, risco in zip(c["cobertura"], c["risco"]):
        if cob >= cobertura_alvo:
            return round(float(risco), 6)
    return None


def avaliar_seletor(nome: str, confianca, erro,
                    coberturas=(0.5, 0.7, 0.9, 1.0),
                    seed: int = 20260902) -> dict:
    """Retrato completo de um seletor, com o acaso ao lado."""
    a = aurc(confianca, erro)
    acaso = aurc_do_acaso(erro, seed=seed)
    ganho = round(acaso["media"] - a, 6)
    return {
        "seletor": nome,
        "n": int(np.asarray(erro).size),
        "risco_sem_abstencao": round(float(np.asarray(erro, dtype=float).mean()), 6),
        "aurc": a,
        "aurc_do_acaso": acaso,
        "ganho_sobre_acaso": ganho,
        # O criterio que separa seletor de corte arbitrario: a ordem tem de
        # cair FORA do intervalo do acaso, e nao apenas ser diferente da media.
        "ordena_melhor_que_o_acaso": bool(a < acaso["p05"]),
        "risco_por_cobertura": {
            str(c): risco_em(confianca, erro, c) for c in coberturas},
        "leitura": ("AURC menor e melhor; se `ordena_melhor_que_o_acaso` for "
                    "falso, a confianca nao carrega informacao e qualquer "
                    "ganho de cobertura parcial e ilusao"),
    }


def comparar(seletores: dict, erro, **kw) -> dict:
    """`{nome: confianca}` contra o MESMO vetor de erro.

    Mesmo erro de proposito: comparar seletores medidos em conjuntos ou tarefas
    diferentes e a forma mais facil de eleger o errado.
    """
    saida = {nome: avaliar_seletor(nome, conf, erro, **kw)
             for nome, conf in seletores.items()}
    validos = {n: v for n, v in saida.items() if v["ordena_melhor_que_o_acaso"]}
    melhor = min(validos, key=lambda n: validos[n]["aurc"]) if validos else None
    return {
        "seletores": saida,
        "melhor": melhor,
        "por_que": ("menor AURC entre os que ordenam melhor que o acaso; "
                    "nenhum vence se nenhum ordena"),
    }
