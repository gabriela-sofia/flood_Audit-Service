"""
Consulta cognitiva do REV-P: tres modos, tres autoridades, nenhuma promocao.

-------------------------------------------------------------------------
MODOS DE CONSULTA E AUTORIDADE DAS FONTES
-------------------------------------------------------------------------

Uma consulta sem separacao de fontes pode sustentar uma resposta com a
fonte inadequada. O projeto distingue os papeis para evitar o seguinte erro:
sustentar uma afirmacao sobre o ESTADO do REV-P com um paper, ou uma
afirmacao metodologica com a documentacao interna. As duas parecem certas e
as duas sao invalidas.

Entao a autoridade e escolhida ANTES da pergunta ser respondida:

  inferencia    autoridade = payload + contrato + model card.
                Responde "por que este escore", "o que e o IC95", "por que
                foi recusado". Nao consulta literatura -- nao precisa, e
                consultar abriria a porta para o paper explicar o payload.

  cientifico    autoridade = corpus de classe `cientifica`.
                Responde "o que e HAND", "por que D-infinity", "por que
                validacao temporal". So cita trecho recuperado.

  social       autoridade = payload social + corpus operacional.
               Responde "quem pode ser mais afetado", "quais fatores de
               vulnerabilidade/exposicao aparecem", "qual prioridade publica".
               Nao responde se a area "e" um rotulo sensivel.

  proveniencia  autoridade = manifestos, artefatos e corpus de classe
                `projeto`. Responde "de onde veio esse numero", "que dataset
                treinou este modelo", "qual o hash do artefato servido".

-------------------------------------------------------------------------
O QUE A LLM NAO GANHA AQUI
-------------------------------------------------------------------------

Nenhuma ferramenta de escrita. Nenhuma execucao. Nenhum acesso a `logs/`.
As unicas capacidades sao LEITURA de escopo fechado -- `rag_busca.buscar` com
classe fixada pelo modo, e o manifesto de proveniencia da propria resposta --
e o resultado continua passando pelo verificador de claims antes de virar
texto. A LLM ganhou assunto, nao autoridade.

Roteamento e DETERMINISTICO: uma tabela de termos, nao um classificador. Se um
dia precisar de modelo para rotear, o roteador tera de ser verificado como
qualquer outro componente -- e ate la, uma tabela que erra de forma previsivel
e melhor que um acerto que ninguem consegue auditar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))

import precedencia_documental  # noqa: E402
from claims import ContextoAutorizado, sem_acento  # noqa: E402
from proveniencia import bloco_de_proveniencia  # noqa: E402
from rag_busca import buscar  # noqa: E402
from rag_corpus import CIENTIFICA, OPERACIONAL, PROJETO  # noqa: E402

MODELOS_SERVIDOS = REPO / "local_runs" / "svc-01-modelos"

INFERENCIA = "inferencia"
CIENTIFICO = "cientifico"
SOCIAL = "social"
PROVENIENCIA = "proveniencia"
MODOS = (INFERENCIA, CIENTIFICO, SOCIAL, PROVENIENCIA)

# Veredito proprio, e nao `sem_suporte`: `sem_suporte` descreve um claim que
# nao achou apoio, e aqui nem se chegou a gerar claim. O payload e que era o
# errado para a pergunta -- e quem le a auditoria precisa conseguir separar
# "a LLM afirmou o que nao podia" de "perguntaram na rota errada".
RECUSADO_POR_PAYLOAD = "recusado_payload_incompativel_com_o_modo"

# Cada modo declara de onde pode tirar apoio. `classe_rag = None` significa
# "este modo nao consulta o corpus" -- e nao "consulta tudo".
AUTORIDADE = {
    INFERENCIA: {
        "classe_rag": None,
        "usa_proveniencia": False,
        "descreve": "payload da resposta, contrato v1 e model card do modelo servido",
    },
    CIENTIFICO: {
        "classe_rag": CIENTIFICA,
        "usa_proveniencia": False,
        "descreve": "corpus de literatura e metodologia, so trecho recuperado",
    },
    SOCIAL: {
        "classe_rag": OPERACIONAL,
        "usa_proveniencia": False,
        "descreve": ("payload social, contrato social e fontes operacionais "
                     "recuperadas; nao rotula territorio sensivel"),
    },
    PROVENIENCIA: {
        "classe_rag": PROJETO,
        "usa_proveniencia": True,
        "descreve": "manifesto de proveniencia, artefatos e documentacao do projeto",
    },
}

TERMOS_DE_PROVENIENCIA = (
    "de onde", "proveniencia", "procedencia", "origem", "hash", "artefato",
    "versao do modelo", "quem gerou", "foi treinado", "dataset", "manifesto",
    "rastre", "cadeia",
)
TERMOS_CIENTIFICOS = (
    "o que e", "por que se usa", "literatura", "referencia", "artigo",
    "paper", "metodo", "metodologia", "d-infinity", "dinfinity", "hand",
    "twi", "firth", "bootstrap", "validacao cruzada", "epv", "doi",
    "estado da arte", "como se calcula", "definicao",
)
TERMOS_SOCIAIS = (
    "vulnerabilidade social", "vulnerabilidade territorial",
    "vulnerabilidade e exposicao", "vulnerabilidade e exposição",
    "vulnerabilidade/exposicao", "vulnerabilidade/exposição",
    "exposicao social", "exposição social", "quem pode ser mais afetado",
    "mais afetado", "populacao vulneravel", "população vulnerável",
    "prioridade publica", "prioridade pública", "acao publica", "ação pública",
    "politica publica", "política pública", "precariedade urbanistica",
    "precariedade urbanística", "desigualdade territorial", "segregacao",
    "segregação", "territorio vulneravel", "território vulnerável",
    "favela", "comunidade urbana",
)


# O que separa "o que E vulnerabilidade social" de "qual a vulnerabilidade
# social DESTA AREA". As duas mencionam o mesmo assunto e pedem coisas
# opostas: a primeira quer literatura, a segunda quer o payload da AOI.
#
# Subconjunto explicito, e nao `TERMOS_CIENTIFICOS` inteiro, porque "como se
# calcula a prioridade publica desta area?" tambem casaria com um termo
# cientifico -- e mandar essa para o modo cientifico devolveria a recusa por
# payload incompativel numa pergunta perfeitamente legitima sobre a AOI.
TERMOS_DE_DEFINICAO = (
    "o que e", "o que sao", "definicao", "conceito de", "literatura",
    "referencia", "paper", "artigo", "estado da arte",
)


def escolher_modo(pergunta: str) -> str:
    """Roteamento deterministico. Proveniencia vence ciencia, que vence inferencia.

    A ordem nao e arbitraria: "de onde veio o HAND deste ponto" e uma
    pergunta de PROVENIENCIA que menciona um termo cientifico, e responder
    com literatura seria responder outra coisa.

    Pelo mesmo criterio, assunto social + pedido de DEFINICAO e pergunta
    cientifica: Cutter 2003 e Rufat 2015 entraram no corpus como `cientifica`,
    e o modo social consulta `operacional`. Sem esta regra, "o que e
    vulnerabilidade social a enchentes?" iria ao modo social e nunca
    alcancaria a literatura que existe no corpus exatamente para responde-la.
    """
    plana = sem_acento(pergunta or "")
    if any(t in plana for t in TERMOS_DE_PROVENIENCIA):
        return PROVENIENCIA
    if any(sem_acento(t) in plana for t in TERMOS_SOCIAIS):
        if any(t in plana for t in TERMOS_DE_DEFINICAO):
            return CIENTIFICO
        return SOCIAL
    if any(t in plana for t in TERMOS_CIENTIFICOS):
        return CIENTIFICO
    return INFERENCIA


CONFIRMATORIO = REPO / "local_runs" / "gfd-quarentena" / "executado.json"


def registro_de_confirmacoes(caminho: Path = CONFIRMATORIO) -> dict | None:
    """O que um teste confirmatorio PROSPECTIVO sustentou, lido do artefato.

    Ausente, ilegivel ou com regime diferente de `confirmatorio_prospectivo`
    devolve None -- e None faz todo claim de generalizacao sair sem suporte.
    O escopo territorial e declarado aqui porque o artefato o declara em
    prosa (`nao_e`): os 16 eventos do teste de 03/10 sao da America do Sul
    fora do Brasil.
    """
    try:
        r = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if r.get("regime") != "confirmatorio_prospectivo" or not r.get("comparacoes"):
        return None
    return {"origem": str(caminho.relative_to(REPO)).replace("\\", "/"),
            "regime": r["regime"], "eventos": len(r.get("eventos_teste") or []),
            "escopo_territorial": "america_do_sul_fora_do_brasil",
            "comparacoes": {k: {"veredito": v.get("veredito")} for k, v in r["comparacoes"].items()}}


def montar_contexto(pergunta: str, resposta: dict, modo: str | None = None,
                    limite: int = 5) -> tuple:
    """Devolve `(modo, ContextoAutorizado)` -- so com o que o modo autoriza."""
    modo = modo or escolher_modo(pergunta)
    if modo not in MODOS:
        raise ValueError(f"modo desconhecido: {modo!r}; conhecidos: {list(MODOS)}")
    regra = AUTORIDADE[modo]

    trechos: list = []
    descartados: list = []
    if regra["classe_rag"] is not None:
        achado = buscar(pergunta, limite=limite, classe=regra["classe_rag"])
        # Cada trecho sai com o estado do documento de onde veio. `anotar`
        # existia desde 31/08 e ninguem o chamava: um trecho de documento
        # superado (ou de commit revisado) chegava ao gerador sem etiqueta, e o
        # verificador de citacao o aprovava porque o trecho existe mesmo.
        trechos = precedencia_documental.anotar(achado["resultados"])
        descartados = achado["descartados_por_injecao"]

    prov = None
    if regra["usa_proveniencia"]:
        try:
            prov = bloco_de_proveniencia(resposta, modelos_dir=MODELOS_SERVIDOS)
        except Exception:
            # Proveniencia e informacao SOBRE a resposta, nunca condicao para
            # ela. Falhou, o modo segue sem ela -- e todo claim de
            # proveniencia sai `sem_suporte`, que derruba a frase. Fail-closed
            # sem derrubar a consulta.
            prov = None

    ctx = ContextoAutorizado(
        resposta=resposta, trechos=trechos, proveniencia=prov,
        evidencia_visual=resposta.get("evidencia_visual"),
        pergunta=pergunta, confirmacoes=registro_de_confirmacoes())
    ctx.descartados_por_injecao = descartados          # so para a auditoria
    return modo, ctx


def incompatibilidade(modo: str, resposta: dict) -> str | None:
    """O modo pedido combina com o payload recebido? Devolve o motivo, ou None.

    ESTE E O BURACO QUE A ROTA `/consulta` ABRIU, e ele nao era fail-closed:
    era resposta errada com etiqueta certa. Com o modo `social` no roteador,
    "quem pode ser mais afetado aqui?" passou a rotear para `social` -- mas a
    rota fisica entrega o payload da inferencia, que nao tem
    `eixos.vulnerabilidade_exposicao`. O verificador social entao derrubava
    todo claim social por falta de payload, e o que sobrava e era publicado
    era o paragrafo FISICO de sempre: escore, IC, TWI, chuva antecedente.

    A pergunta era sobre pessoas e a resposta falava de terreno, sob o rotulo
    "autoridade: payload social". Isso e exatamente o modo de falha que o
    REV-P-EVAL batizou de `pertinencia` -- publicou sempre, respondeu nunca --
    e aqui ele vinha com uma etiqueta que afirmava o contrario do conteudo.

    A checagem vale nos dois sentidos. Payload social sob modo fisico teria o
    defeito espelhado: o `svc06` monta o resumo pelo TIPO DE PAYLOAD, entao o
    modo `inferencia` receberia o resumo social e responderia prioridade
    publica onde perguntaram escore.
    """
    from svc06_explicacao_llm import eh_payload_social

    social_no_payload = eh_payload_social(resposta)
    if modo == SOCIAL and not social_no_payload:
        return ("a pergunta foi roteada para o modo social, e o payload "
                "recebido e da inferencia fisica: nao ha "
                "`eixos.vulnerabilidade_exposicao` nem `prioridade_publica` "
                "para sustentar afirmacao sobre exposicao ou prioridade. "
                "A resposta social vem de POST /social/consulta, que calcula "
                "a camada social da mesma AOI antes de perguntar")
    if modo != SOCIAL and social_no_payload:
        return (f"a pergunta foi roteada para o modo {modo!r}, e o payload "
                "recebido e da camada social: ele nao carrega escore, IC nem "
                "model card, e a camada social nunca altera o payload fisico. "
                "A resposta fisica vem de POST /consulta")
    return None


def responder(pergunta: str, resposta: dict, motor=None,
              modo: str | None = None, limite: int = 5) -> dict:
    """Sintese verificada para uma pergunta, dentro da autoridade do modo.

    Aditivo: nao le nem altera nenhum campo cientifico de `resposta`. O que
    sai e um bloco proprio -- e mesmo ele so contem frase que passou pelo
    verificador de claims.
    """
    from svc06_explicacao_llm import sintetizar

    escolhido = modo or escolher_modo(pergunta)
    motivo = incompatibilidade(escolhido, resposta) if escolhido in MODOS else None
    if motivo:
        # Recusa ANTES de chamar o motor. Nao ha nada que a LLM possa escrever
        # aqui que o verificador consiga sustentar, e deixa-la escrever produz
        # a resposta errada com a etiqueta certa.
        return {
            "pergunta": pergunta,
            "modo": escolhido,
            "autoridade": AUTORIDADE[escolhido]["descreve"],
            "classe_de_autoridade_consultada": None,
            "sintese": None,
            "auditoria_sintese": {
                "veredito": RECUSADO_POR_PAYLOAD,
                "por_que": motivo,
                "frases_publicadas": 0,
            },
            "fontes": [],
            "proveniencia": None,
            "descartados_por_injecao": [],
            "por_que_null": motivo,
            "nao_e": ("a recusa e da consulta, nao da resposta: os campos "
                      "acima do bloco `consulta` seguem valendo"),
        }

    modo, ctx = montar_contexto(pergunta, resposta, escolhido, limite)
    saida = sintetizar(resposta, motor, evidencia=False, contexto=ctx)
    auditoria = saida["auditoria_sintese"]

    fontes = [
        {k: t.get(k) for k in ("chunk_id", "source_id", "titulo", "secao",
                               "classe_autoridade", "hash_chunk", "doi", "url")}
        for t in ctx.trechos_limpos()
    ]
    return {
        "pergunta": pergunta,
        "modo": modo,
        "autoridade": AUTORIDADE[modo]["descreve"],
        "classe_de_autoridade_consultada": AUTORIDADE[modo]["classe_rag"],
        "sintese": saida["sintese"],
        "auditoria_sintese": auditoria,
        "fontes": fontes,
        "proveniencia": ctx.proveniencia,
        "descartados_por_injecao": getattr(ctx, "descartados_por_injecao", []),
        # Sempre presente, mesmo quando ha sintese: o transporte copia um
        # conjunto FIXO de chaves, e um campo que so existe as vezes vira
        # `KeyError` no unico caminho em que ele importa -- o da recusa.
        "por_que_null": None,
        "nao_e": ("a consulta explica a decisao ja tomada; ela nao calcula "
                  "escore, nao move gate e nao promove maturidade"),
    }


def main() -> int:
    from svc02_contrato_inferencia import (
        _requisicoes_de_demonstracao, caixas_das_regioes, carregar_modelos,
        inferir, regioes_com_referencia_local,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--modo", choices=list(MODOS), default=None)
    args = parser.parse_args()

    modelos = carregar_modelos()
    if not modelos:
        print("ABORTADO: modelos servidos ausentes; rode svc01 primeiro")
        return 1
    _, req = _requisicoes_de_demonstracao(modelos)[0]
    resposta = inferir(req, modelos, caixas_das_regioes(),
                       regioes_com_referencia_local())
    print(json.dumps(responder(args.pergunta, resposta, modo=args.modo),
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
