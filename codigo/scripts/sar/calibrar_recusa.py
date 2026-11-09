"""A7 -- calibracao do limiar de recusa, na validacao.

-------------------------------------------------------------------------
O CRITERIO E DECLARADO ANTES DE VER A CURVA
-------------------------------------------------------------------------

Recusar melhora quase qualquer metrica: basta recusar os chips dificeis e o
IoU dos que sobram sobe. Um limiar escolhido depois de olhar o resultado e,
por construcao, o limiar que fez o numero ficar bonito.

Entao o criterio vem primeiro, e e este:

    o MAIOR limiar cujo custo de cobertura nao passa de `custo_maximo`

`custo_maximo` e a fracao de chips que se aceita perder. O ganho de IoU nao
entra na escolha -- ele e reportado como CONSEQUENCIA. Se recusar 10% dos
chips nao melhorar nada, isso aparece no relatorio, e a leitura correta passa a
ser "a recusa nao esta separando o que ela deveria separar".

-------------------------------------------------------------------------
VALIDACAO, NUNCA TESTE
-------------------------------------------------------------------------

A varredura roda no split de validacao. Calibrar no teste transformaria o teste
em conjunto de ajuste, e o numero final deixaria de medir o que promete -- sem
que nada na saida denunciasse.

O holdout da Bolivia tambem nao entra: ele responde a pergunta da etapa A6, e
gastar holdout em calibracao e a forma mais comum de perder um holdout.

-------------------------------------------------------------------------
O SEGUNDO CRITERIO, e por que um so nao bastava
-------------------------------------------------------------------------

O criterio do orcamento acima responde "qual o maior limiar que cabe em 10% de
cobertura perdida?". Ele e honesto e foi declarado antes da curva -- e responde
uma pergunta que nao decide nada sozinha.

Medido em 02/09/2026, ele escolheu limiar 0,05, que recusou 1 chip de 88 e
levou o ganho de IoU a 1e-06. A leitura registrada na epoca foi "a confianca da
rede nao separa o que deveria". Essa leitura estava errada, e o proprio
projeto ja tinha o instrumento para mostrar isso: sob a regua de predicao
seletiva, a mesma confianca tem AURC 0,5604 contra 0,7284 do acaso. Ela SEPARA.
O que nao separava era o ponto de operacao -- recusar 1 chip de 88 nao remove
risco nenhum, qualquer que seja a qualidade do ordenamento.

Comparar dois seletores num unico ponto de cobertura e comparar duas curvas por
um pixel (`predicao_seletiva.py`, e o erro que aquele modulo foi escrito para
corrigir). Entao entra um segundo criterio, declarado aqui antes de qualquer
curva ser olhada:

    1. a confianca so e usavel se ORDENAR melhor que o acaso (AURC contra o
       AURC de ordenacao aleatoria, no proprio split de validacao);
    2. entre os limiares que sobram, escolher o que remove MAIS RISCO POR
       UNIDADE DE COBERTURA abandonada;
    3. nunca abaixo de `cobertura_minima` -- um seletor que responde sobre 20%
       dos chips e um seletor que desistiu.

Se (1) falhar, nao ha escolha: o modulo devolve limiar `None` com o motivo, e a
recusa continua desligada. Nao ha ponto de operacao bom sobre um ordenamento
que nao tem informacao.

-------------------------------------------------------------------------
O QUE A RECUSA SIGNIFICA
-------------------------------------------------------------------------

Chip recusado nao e chip sem agua. E chip sobre o qual a rede nao tem posicao,
e o contrato do REV-P ja tem nome para isso: `nao_identificavel`. A regra 5 das
frentes neurais -- erro da rede e recusa, nao silencio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))

from avaliar_rede import MARGEM_NA_PREDICAO, avaliar  # noqa: E402
from baseline_otsu import SAIDA  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

LIMIARES_PADRAO = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


def varrer(rotulo: str = "v1", split: str = "validacao",
           limiares: tuple = LIMIARES_PADRAO,
           modo_confianca: str = MARGEM_NA_PREDICAO) -> list[dict]:
    """Um ponto por limiar: quanto se recusa, e o que sobra vale quanto."""
    pontos = []
    total = None
    for limiar in limiares:
        r = avaliar(rotulo, split,
                    limiar_confianca=None if limiar == 0.0 else limiar,
                    modo_confianca=modo_confianca)["resumo"]
        avaliados = r["chips_avaliados"]
        recusados = r["chips_recusados"]
        if total is None:
            total = avaliados + recusados
        pontos.append({
            "limiar": limiar,
            "chips_avaliados": avaliados,
            "chips_recusados": recusados,
            "custo_de_cobertura": round(recusados / total, 6) if total else None,
            "iou_micro": r["micro"]["iou"],
            "iou_micro_sem_agua_permanente":
                r["micro_sem_agua_permanente"]["iou"],
            "iou_macro": r["macro"]["iou"],
            "iou_macro_sem_agua_permanente":
                r["macro_sem_agua_permanente"]["iou"],
        })
    return pontos


def escolher(pontos: list[dict], custo_maximo: float = 0.1) -> dict:
    """O maior limiar dentro do orcamento de cobertura. So isso.

    O IoU nao participa da escolha -- se participasse, o criterio seria "o
    limiar que fez o numero subir", e a recusa viraria maquiagem.
    """
    elegiveis = [p for p in pontos
                 if p["custo_de_cobertura"] is not None
                 and p["custo_de_cobertura"] <= custo_maximo
                 and p["chips_avaliados"] > 0]
    if not elegiveis:
        return {"limiar": None,
                "por_que": (f"nenhum limiar recusa menos que {custo_maximo:.0%} "
                            "dos chips; a rede nao tem confianca separavel "
                            "neste orcamento")}
    escolhido = max(elegiveis, key=lambda p: p["limiar"])
    sem_recusa = next((p for p in pontos if p["limiar"] == 0.0), None)
    ganho = None
    if sem_recusa and escolhido["iou_micro_sem_agua_permanente"] is not None \
            and sem_recusa["iou_micro_sem_agua_permanente"] is not None:
        ganho = round(escolhido["iou_micro_sem_agua_permanente"]
                      - sem_recusa["iou_micro_sem_agua_permanente"], 6)
    return {
        **escolhido,
        "custo_maximo_declarado": custo_maximo,
        "ganho_de_iou_sem_agua_permanente": ganho,
        "por_que": ("maior limiar dentro do orcamento de cobertura; o ganho de "
                    "IoU e consequencia relatada, nao criterio de escolha"),
        "leitura_se_ganho_nulo": ("recusar sem melhorar significa que a "
                                  "confianca da rede nao separa o que deveria"),
    }


def curva_de_risco(rotulo: str = "v1", split: str = "validacao",
                   modo_confianca: str = MARGEM_NA_PREDICAO) -> dict:
    """O par (confianca, erro) de cada chip, na regua comum das frentes.

    `erro` e `1 - IoU sem agua permanente`. A agua permanente sai porque ela e
    metade do que o limiar acerta sem saber nada: com ela dentro, o risco de um
    chip de rio largo fica baixo por causa do rio, e a ordenacao pareceria boa
    sem que a rede tivesse contribuido.

    Chip sem IoU definido (nenhuma agua verdadeira e nenhuma predita) fica de
    fora: nao ha erro a ordenar ali, e imputar zero faria o chip mais facil do
    conjunto parecer o melhor caso de confianca.
    """
    sys.path.insert(0, str(REPO / "scripts" / "neural"))
    from predicao_seletiva import avaliar_seletor

    r = avaliar(rotulo, split, limiar_confianca=None,
                modo_confianca=modo_confianca)
    pares = [(c["certeza"], c["iou_sem_permanente"]) for c in r["por_chip"]
             if c.get("certeza") is not None
             and c.get("iou_sem_permanente") is not None]
    if not pares:
        return {"medivel": False,
                "motivo": ("nenhum chip com certeza e IoU definidos; sem par "
                           "(confianca, erro) nao ha curva")}
    confianca_ = [p[0] for p in pares]
    erro = [1.0 - p[1] for p in pares]
    medida = avaliar_seletor("A_rede_sar_" + split, confianca_, erro)
    return {
        "medivel": True,
        "n": len(pares),
        "o_que_e_erro": "1 - IoU do chip, sem agua permanente",
        "o_que_e_confianca": modo_confianca,
        **medida,
    }


def escolher_por_curva(pontos: list[dict], curva: dict,
                       cobertura_minima: float = 0.5) -> dict:
    """Mais risco removido por cobertura abandonada, e so se a curva ordenar.

    O criterio esta escrito no topo deste arquivo, antes de qualquer numero.
    Aqui ele so e aplicado.
    """
    if not curva.get("medivel"):
        return {"limiar": None,
                "por_que": curva.get("motivo", "curva nao medivel")}
    if not curva.get("ordena_melhor_que_o_acaso"):
        return {
            "limiar": None,
            "aurc": curva.get("aurc"),
            "aurc_do_acaso": (curva.get("aurc_do_acaso") or {}).get("media"),
            "por_que": ("a confianca nao ordena melhor que o acaso neste "
                        "split; nao existe ponto de operacao bom sobre um "
                        "ordenamento sem informacao, e a recusa fica desligada"),
        }

    sem_recusa = next((p for p in pontos if p["limiar"] == 0.0), None)
    base = (sem_recusa or {}).get("iou_micro_sem_agua_permanente")
    if base is None:
        return {"limiar": None,
                "por_que": "sem ponto de referencia em limiar 0,0"}

    elegiveis = []
    for p in pontos:
        custo = p.get("custo_de_cobertura")
        iou = p.get("iou_micro_sem_agua_permanente")
        if custo is None or iou is None or custo <= 0:
            continue
        if (1.0 - custo) < cobertura_minima:
            continue
        elegiveis.append({**p, "ganho": round(iou - base, 6),
                          "ganho_por_cobertura": round((iou - base) / custo, 6)})
    if not elegiveis:
        return {"limiar": None,
                "cobertura_minima_declarada": cobertura_minima,
                "por_que": ("nenhum limiar recusa alguma coisa mantendo a "
                            f"cobertura acima de {cobertura_minima:.0%}")}

    escolhido = max(elegiveis, key=lambda p: p["ganho_por_cobertura"])
    if escolhido["ganho"] <= 0:
        return {
            "limiar": None,
            "melhor_candidato": escolhido["limiar"],
            "ganho_do_melhor_candidato": escolhido["ganho"],
            "por_que": ("a curva ordena, mas nenhum ponto de operacao dentro "
                        "da cobertura minima remove risco; recusar aqui "
                        "custaria cobertura sem comprar nada"),
        }
    return {
        **escolhido,
        "cobertura_minima_declarada": cobertura_minima,
        "criterio": ("maior ganho de IoU por unidade de cobertura abandonada, "
                     "entre os limiares que preservam a cobertura minima"),
        "so_vale_porque": ("a confianca ordena melhor que o acaso neste split "
                           f"(AURC {curva.get('aurc')} contra "
                           f"{(curva.get('aurc_do_acaso') or {}).get('media')})"),
        "nao_e": ("nao e promocao do elo de observacao: quem decide se a rede "
                  "pode falar sobre area que ninguem conferiu continua sendo o "
                  "contrato, e este numero e so o ponto de operacao dela"),
    }


def calibrar(rotulo: str = "v1", split: str = "validacao",
             custo_maximo: float = 0.1,
             modo_confianca: str = MARGEM_NA_PREDICAO,
             cobertura_minima: float = 0.5) -> dict:
    if split == "teste":
        raise ValueError(
            "calibrar no teste transforma o teste em conjunto de ajuste; "
            "use `validacao`")
    if split == "bolivia":
        raise ValueError(
            "o holdout responde a etapa A6; gastar holdout em calibracao e a "
            "forma mais comum de perder um holdout")
    pontos = varrer(rotulo, split, modo_confianca=modo_confianca)
    curva = curva_de_risco(rotulo, split, modo_confianca)
    resultado = {
        "tipo": "calibracao_de_recusa",
        "etapa": "A7",
        "quando_utc": agora_utc(),
        "checkpoint": rotulo,
        "split": split,
        "modo_confianca": modo_confianca,
        "criterio": ("o maior limiar cujo custo de cobertura nao passa de "
                     f"{custo_maximo:.0%}; o IoU nao participa da escolha"),
        "curva": pontos,
        "escolhido": escolher(pontos, custo_maximo),
        # OS DOIS CRITERIOS VIAJAM JUNTOS, e o segundo nao apaga o primeiro.
        # Quem ja consumia `escolhido` continua lendo o mesmo campo; quem
        # precisa do ponto de operacao util le `escolhido_por_curva`. Trocar o
        # significado de um campo existente seria mudar o comportamento de quem
        # consome sem que nada na saida denunciasse.
        "risco_cobertura": curva,
        "escolhido_por_curva": escolher_por_curva(pontos, curva,
                                                  cobertura_minima),
        "nao_e": ("chip recusado nao e chip sem agua: e chip sobre o qual a "
                  "rede nao tem posicao, e o contrato chama isso de "
                  "`nao_identificavel`"),
    }
    resultado["hash_resultado"] = sha256_json(
        {k: v for k, v in resultado.items() if k != "quando_utc"})
    return resultado


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotulo", default="v1")
    parser.add_argument("--split", default="validacao")
    parser.add_argument("--custo-maximo", type=float, default=0.1)
    parser.add_argument("--cobertura-minima", type=float, default=0.5)
    parser.add_argument("--modo-confianca", default=MARGEM_NA_PREDICAO)
    # Gravar e o PADRAO. A forma anterior (`--gravar` opcional) criava
    # artefato que envelhece em silencio: em 22/09/2026 o `casos.json` do
    # produto estava desatualizado desde antes da migracao para parquet e um
    # teste passou contra ele por dias. `--gravar` segue aceito para nao
    # quebrar chamador existente.
    parser.add_argument("--gravar", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--nao-gravar", dest="gravar", action="store_false",
                        help="nao escreve artefato; so imprime o resumo")
    args = parser.parse_args()

    resultado = calibrar(args.rotulo, args.split, args.custo_maximo,
                         args.modo_confianca, args.cobertura_minima)
    if args.gravar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        destino = SAIDA / f"calibracao_recusa_{args.rotulo}.json"
        destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        resultado["gravado_em"] = str(destino)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
