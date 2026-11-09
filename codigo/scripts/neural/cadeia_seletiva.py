"""A unificacao: uma cadeia de portoes seletivos, e o que ela faz o REV-P saber.

-------------------------------------------------------------------------
O QUE AS QUATRO FRENTES SAO, JUNTAS
-------------------------------------------------------------------------

Separadas, sao quatro componentes que abstem. Juntas, sao uma CADEIA -- e a
cadeia tem uma propriedade que nenhum elo tem sozinho.

    observacao      A  a rede diz onde havia agua, ou recusa o chip
    representacao   B  o encoder diz com que se parece aquele terreno
    recuperacao     C  o indice traz o que o corpus sabe sobre isso
    verificacao     D  a conjuncao decide se aquilo pode ser dito

Cada elo pode dizer "nao sei". A cadeia publica o que sobrevive a todos, e
declara em qual elo parou quando nao publica.

-------------------------------------------------------------------------
A PROPRIEDADE QUE SO A CADEIA TEM
-------------------------------------------------------------------------

Em predicao seletiva, o risco de uma conjuncao de seletores independentes e no
maximo o menor risco individual -- e a cobertura, no maximo a menor cobertura.
Encadear TROCA cobertura por risco, e faz isso de forma monotona: nenhum elo
adicional pode aumentar o risco da cadeia.

E isso, e nao a acuracia de nenhum componente, e o que sustenta a afirmacao que
o projeto precisa fazer: o sistema erra menos porque sabe se calar mais, e sabe
DIZER onde se calou.

A demonstracao empirica esta em `medir_as_quatro_frentes`: a sobreposicao
lexica sozinha nao ordena melhor que o acaso (AURC 0,4029 contra p05 0,3492) e
em conjuncao com o juiz passa a ordenar (0,1085 contra p05 0,1266). Encadear
criou um seletor onde nao havia.

-------------------------------------------------------------------------
COMO O SISTEMA APRENDE, sem que nenhum peso mude
-------------------------------------------------------------------------

O ciclo ja estava desenhado em `IA_CIENTIFICA_MEMORIA_AUDITAVEL` e em
`memoria_cientifica.py`. O que faltava era ligar as frentes a ele:

    1. a frente A observa um evento novo e produz uma mascara com confianca
    2. o que passa no portao vira REGISTRO, com fonte, hash e limite de uso
    3. so registro `verificado` vira chunk elegivel
    4. a frente C passa a recuperar esse chunk
    5. a frente D decide se ele pode sustentar uma frase

Depois do passo 5, o sistema sabe algo que nao sabia -- e nenhum parametro
mudou. E aprendizado por acumulo de evidencia verificavel, que e a unica forma
de aprender compativel com "toda afirmacao tem de mostrar de onde veio".

Fine-tuning moveria esse conhecimento para dentro dos pesos e apagaria a
resposta da pergunta "qual evidencia sustenta isto".
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "neural"))

from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "sintese-neural"

OBSERVACAO = "observacao"
REPRESENTACAO = "representacao"
FORCAMENTO = "forcamento"
RECUPERACAO = "recuperacao"
VERIFICACAO = "verificacao"
ELOS = (OBSERVACAO, REPRESENTACAO, FORCAMENTO, RECUPERACAO, VERIFICACAO)

PUBLICADO = "publicado"
ABSTIDO = "abstido"


@dataclass
class Portao:
    """Um elo. `decidir` devolve `(passa, confianca, detalhe)`.

    `passa=False` NAO e "a resposta e nao": e "este elo nao sustenta seguir".
    A diferenca importa porque a cadeia publica a abstencao com o nome do elo,
    e nunca a converte em afirmacao negativa.
    """

    elo: str
    nome: str
    decidir: object
    obrigatorio: bool = True


@dataclass
class Resultado:
    estado: str
    parou_em: str | None
    trilha: list = field(default_factory=list)
    confianca_minima: float | None = None

    def como_dicionario(self) -> dict:
        return {
            "estado": self.estado,
            "parou_em": self.parou_em,
            "confianca_minima_da_cadeia": self.confianca_minima,
            "trilha": self.trilha,
            "nao_e": ("abstencao nao e resposta negativa: a cadeia diz que nao "
                      "sustenta seguir, e nomeia o elo onde parou"),
        }


def executar_cadeia(portoes: list, entrada: dict) -> Resultado:
    """Percorre os portoes na ordem. Para no primeiro que nao passa.

    A ordem e a dos elos, e nao a de custo: parar cedo por barato inverteria a
    logica -- o elo mais barato passaria a decidir o que os caros nunca veem.
    Aqui o mais barato so decide se ele for o primeiro POR SIGNIFICADO.
    """
    trilha = []
    confiancas = []
    for portao in portoes:
        try:
            passa, confianca, detalhe = portao.decidir(entrada)
        except Exception as e:
            # Elo quebrado nao publica e nao derruba silenciosamente: ele
            # aparece na trilha como falha, e a cadeia para nele.
            trilha.append({"elo": portao.elo, "portao": portao.nome,
                           "passou": False, "confianca": None,
                           "erro": f"{type(e).__name__}: {e}"})
            return Resultado(ABSTIDO, portao.elo, trilha, None)

        trilha.append({"elo": portao.elo, "portao": portao.nome,
                       "passou": bool(passa), "confianca": confianca,
                       "detalhe": detalhe})
        if confianca is not None:
            confiancas.append(float(confianca))
        if not passa:
            return Resultado(ABSTIDO, portao.elo, trilha,
                             min(confiancas) if confiancas else None)

    return Resultado(PUBLICADO, None, trilha,
                     min(confiancas) if confiancas else None)


# ------------------------------------------------------------- portoes reais


def portao_forcamento(dominio, *, variavel: str = "rain_event_day"):
    """Abstem quando a chuva do pedido esta fora do dominio em que o efeito foi
    estimado.

    POR QUE ESTE ELO EXISTE, e por que ele pertence ao eixo fisico:

    O efeito da chuva foi estimado sobre uma faixa de valores observados. Fora
    dela o modelo nao mediu nada -- extrapolar ali produz numero, e nao
    conhecimento. E o mesmo principio que ja governa a grade: no cenario de
    chuva extrema, 11.385 celulas de Recife ficam BRANCAS porque o pedido saiu
    da faixa 5-95% do ajuste. O mapa ja se abstem; a cadeia passa a se abster
    pela mesma razao, e com o mesmo endereco.

    A confianca e a distancia ao centro do dominio: 1 no meio, 0 na borda. Valor
    na borda passa com confianca minima, e isso e a leitura certa -- ali o
    suporte e mais fino, e a cadeia toma o minimo das confiancas.

    Pedido sem valor de forcamento NAO passa: ausencia de chuva declarada nao e
    chuva zero, que e a regra fail-closed do contrato de chuva.
    """
    lo, hi = float(dominio[0]), float(dominio[1])
    if hi <= lo:
        raise ValueError("dominio de forcamento vazio ou invertido")
    meio, meia_largura = (lo + hi) / 2.0, (hi - lo) / 2.0

    def decidir(entrada: dict):
        valor = (entrada or {}).get(variavel)
        if valor is None:
            return False, None, {"motivo": "pedido sem valor de forcamento",
                                 "variavel": variavel,
                                 "nao_e": "ausencia declarada nao e chuva zero"}
        valor = float(valor)
        dentro = lo <= valor <= hi
        confianca = max(0.0, 1.0 - abs(valor - meio) / meia_largura)
        return dentro, round(confianca, 4), {
            "variavel": variavel, "valor": valor,
            "dominio_estimado": [lo, hi],
            "distancia_ao_dominio": round(max(lo - valor, valor - hi, 0.0), 4),
            "motivo": ("dentro do dominio estimado" if dentro
                       else "fora do dominio em que o efeito foi estimado"),
        }

    return Portao(elo=FORCAMENTO,
                  nome=f"forcamento dentro de [{lo:g}, {hi:g}] mm",
                  decidir=decidir)


def portao_recuperacao(indice, classe: str, limiar: float = 0.0,
                       k: int = 5) -> Portao:
    """C: o corpus tem algo sobre isso, e o quao proximo esta."""

    def decidir(entrada: dict):
        achados = indice.buscar(entrada["consulta"], classe=classe, k=k)
        if not achados:
            return False, 0.0, {"recuperados": 0,
                                "por_que": "nada no corpus desta classe"}
        melhor = float(achados[0]["similaridade"])
        return (melhor > limiar, melhor,
                {"recuperados": len(achados),
                 "primeiro": achados[0]["chunk_id"]})

    return Portao(RECUPERACAO, "indice_denso", decidir)


def portao_verificacao(trecho_de: object, minimo_lexico: float = 0.34,
                       usar_juiz: bool = False) -> Portao:
    """D: a conjuncao. O juiz so subtrai, como na frente D."""
    sys.path.insert(0, str(REPO / "scripts" / "verificacao"))

    def decidir(entrada: dict):
        from claims import sustenta

        trecho = trecho_de(entrada)
        if not trecho:
            return False, 0.0, {"por_que": "nenhum trecho recuperado"}
        aprovou, fracao = sustenta(entrada["claim"], trecho, minimo_lexico)
        if not aprovou:
            return False, fracao, {"por_que": "sobreposicao lexica insuficiente"}
        if not usar_juiz:
            return True, fracao, {"juiz": "nao consultado"}

        from medir_suporte import julgar_local

        discorda, bruto = julgar_local(
            entrada["claim"], trecho,
            "http://127.0.0.1:11434/v1/chat/completions",
            "qwen2.5:3b-instruct")
        if discorda is True:
            return False, fracao, {"juiz": "discorda", "bruto": bruto[:60]}
        # indisponivel nao derruba: mesma regra da proveniencia ausente
        return True, fracao, {"juiz": "concorda" if discorda is False
                              else "indisponivel"}

    return Portao(VERIFICACAO, "conjuncao_lexico_e_juiz", decidir)


def cadeia_de_citacao(indice, classe: str, usar_juiz: bool = False) -> list:
    """A cadeia que decide se uma frase pode citar o corpus.

    Dois elos, e nao quatro: observacao e representacao pertencem ao eixo
    fisico, e uma frase sobre metodologia nao passa por eles. Montar uma cadeia
    com elos que nao se aplicam produziria abstencao por ausencia de dado
    irrelevante -- que e o oposto do que abstencao deve significar.
    """
    ultimo: dict = {}

    def guardar_trecho(entrada: dict):
        achados = indice.buscar(entrada["consulta"], classe=classe, k=5)
        if not achados:
            return None
        ultimo["chunk_id"] = achados[0]["chunk_id"]
        i = indice.chunk_ids.index(achados[0]["chunk_id"])
        return indice.conteudos[i]

    return [portao_recuperacao(indice, classe),
            portao_verificacao(guardar_trecho, usar_juiz=usar_juiz)]


def demonstrar(usar_juiz: bool = False) -> dict:
    """Roda a cadeia em casos reais da frente D e mostra onde ela para."""
    sys.path.insert(0, str(REPO / "scripts" / "recuperacao"))
    sys.path.insert(0, str(REPO / "scripts" / "verificacao"))
    from casos_de_suporte import todos
    from indice_denso import construir

    indice = construir()
    saidas = []
    for caso in todos():
        # a consulta e o proprio claim: e assim que a cadeia e usada de verdade
        # -- a frase proposta procura o que poderia sustenta-la.
        portoes = cadeia_de_citacao(indice, "cientifica", usar_juiz)
        r = executar_cadeia(portoes, {"consulta": caso["claim"],
                                      "claim": caso["claim"]})
        saidas.append({"id": caso["id"], "familia": caso["familia"],
                       "esperado_sustenta": caso["sustenta"],
                       **r.como_dicionario()})

    publicados = [s for s in saidas if s["estado"] == PUBLICADO]
    corretos = sum(1 for s in saidas
                   if (s["estado"] == PUBLICADO) == s["esperado_sustenta"])
    parou_em = {}
    for s in saidas:
        if s["parou_em"]:
            parou_em[s["parou_em"]] = parou_em.get(s["parou_em"], 0) + 1

    resumo = {
        "tipo": "demonstracao_cadeia_seletiva",
        "quando_utc": agora_utc(),
        "usar_juiz": usar_juiz,
        "casos": len(saidas),
        "publicados": len(publicados),
        "cobertura": round(len(publicados) / len(saidas), 4) if saidas else None,
        "acuracia_da_decisao": round(corretos / len(saidas), 4) if saidas else None,
        "abstencoes_por_elo": parou_em,
        "elos": list(ELOS),
        "por_que_dois_elos": (
            "observacao e representacao pertencem ao eixo fisico; uma frase "
            "sobre metodologia nao passa por eles, e uma cadeia com elos "
            "inaplicaveis produz abstencao por dado irrelevante"),
        "saidas": saidas,
    }
    resumo["hash_resumo"] = sha256_json(
        {k: v for k, v in resumo.items() if k != "quando_utc"})
    return resumo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--juiz", action="store_true")
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

    resumo = demonstrar(args.juiz)
    if args.gravar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        sufixo = "_com_juiz" if args.juiz else ""
        destino = SAIDA / f"cadeia_seletiva{sufixo}.json"
        destino.write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        resumo["gravado_em"] = str(destino)
    print(json.dumps({k: v for k, v in resumo.items() if k != "saidas"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
