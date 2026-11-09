"""
SVC-06 -- Explicacao em linguagem natural, com verificador de grounding.

O QUE ESTE ARQUIVO E:

uma camada que le o payload JA DECIDIDO pelo contrato e produz um paragrafo em
portugues. Ela nao calcula escore, nao le `camadas` cru, nao ve rotulo e nao
consulta portao nenhum. O que ela recebe e o mesmo dict que `/inferencia`
devolve, filtrado por uma lista branca (`resumo_para_o_motor`).

E ADITIVA, e isso e uma decisao de contrato e nao de implementacao. O campo
`explicacao` -- a frase gerada por regra, que existe desde o E6 -- fica intacto.
O que sai daqui entra em dois campos novos e opcionais, que um cliente da v1
pode ignorar:

  `sintese`            a prosa APROVADA pelo verificador, ou `None` quando
                       nenhuma frase passou. Nunca o texto determinstico
                       disfarcado de prosa gerada.
  `auditoria_sintese`  motor, frase a frase julgada, motivo de cada reprovacao
                       e `por_que_null`.

Pela politica de `docs/CONTRATO_VERSIONAMENTO.md` acrescentar campo opcional
nao quebra versao, e foi assim que `escore_detalhe` e `procedencia_das_camadas`
entraram. Se um dia a sintese SUBSTITUIR `explicacao`, o significado daquele
campo muda e passa a ser `v2` -- decisao explicita, nunca efeito colateral.

-------------------------------------------------------------------------
O VERIFICADOR, e por que ele e o centro deste arquivo
-------------------------------------------------------------------------

A camada de explicacao por regra nao podia divergir do payload: ela era montada
da mesma contribuicao que entrou no escore. Um gerador de linguagem PODE
divergir -- e vai, mais cedo ou mais tarde. Entao o texto nao e publicado como
veio: cada frase passa por duas checagens, e frase que nao passa e DESCARTADA,
nunca marcada e mantida.

  NUMEROS   todo numero do texto tem de aparecer, com o mesmo arredondamento,
            em algum campo do payload -- escore, extremos do IC, coeficiente,
            valor de variavel, AUC, contagem do ajuste. Numero que nao bate
            derruba a frase inteira, porque um numero inventado no meio de uma
            frase correta e pior que uma frase claramente errada.

  DIRECAO   frase que diga que uma variavel "pesa a favor" ou "pesa contra"
            tem de bater com o `sentido` que o payload ja declarou para ela E
            com a variavel estar entre as `fisicamente_verificada`. E a mesma
            trava do E6 (secao 5.2 da doc consolidada): sem ela o texto pode
            reabrir o bug de "elevacao pesa a favor" -- fisica invertida dita
            em portugues fluente, que e a forma mais convincente de estar
            errado.

  CAUSA     frase que afirme causa -- com QUALQUER verbo da familia, e nao so
            com `pesa` -- so pode nomear variavel com sinal fisico verificado.
            Sobre as outras o ajuste nao estabeleceu efeito, e negar ("nao
            contribui") tambem e afirmar. Ver `POLITICA_VARIAVEL_NAO_VERIFICADA`.

Se nada sobreviver, `sintese` e `None` e `auditoria_sintese.por_que_null` diz
por que. Fail-closed: a ausencia de texto novo e um RESULTADO, nao uma falha a
esconder -- e nao pode ser preenchida por baixo do pano com a frase por regra,
que e o que a tela ja mostra no campo `explicacao`.

-------------------------------------------------------------------------
O MOTOR E PLUGAVEL, e o padrao vem sem LLM nenhuma
-------------------------------------------------------------------------

A escolha entre API comercial e modelo local ficou em aberto, e nao precisa ser
tomada para este arquivo existir. `MotorDeReferencia` e deterministico, roda
sem rede e sem GPU, e serve para duas coisas: ser o padrao e ser o alvo dos
testes -- teste que depende de rede nao e teste.

`MotorHTTP` e o adaptador para uma API comercial. Ele so age quando a chave
esta no ambiente e o egresso foi pedido explicitamente. Nenhum teste da suite o
aciona.

`MotorLocal` e o adaptador para um servidor HTTP local (Ollama ou qualquer
coisa que fale a mesma API de `/chat/completions`) -- decisao tomada em 26/08:
motor local, CPU-only, sem GPU no ambiente (Tabela I do artigo). So age quando
`LOCAL_LLM_URL` esta configurada, e `motor_padrao()` o embrulha em
`MotorComFallback`: se o servidor local estiver fora do ar ou estourar o
timeout, a sintese cai para `MotorDeReferencia` de forma declarada, nunca
disfarcada -- ver `MotorComFallback`. Sem `LOCAL_LLM_URL`, o comportamento
padrao continua identico ao de antes desta decisao.

O verificador nao sabe nem se importa com qual motor gerou o texto. E de
proposito: se ele so funcionasse com um motor, nao estaria verificando o texto,
estaria confiando no gerador.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from evidencia_visual import consultar as consultar_evidencia_visual  # noqa: E402

# ------------------------------------------------------------------ lexico
#
# O LEXICO E O VERIFICADOR MORAM EM `claims.py`, e isto nao e reorganizacao
# de gaveta. Enquanto o unico consumidor era este arquivo, ter os padroes
# aqui era o certo. Agora o benchmark (`revp_eval_llm.py`), a consulta
# cognitiva (`consulta_cognitiva.py`) e o verificador de citacao precisam das
# MESMAS regras -- e duas copias de uma regra de seguranca divergem, sempre.
#
# O que continua exportado daqui e a API que o resto do projeto ja importa:
# `verificar_frase`, `verificar`, `claims_da_frase`, `_numeros_permitidos`.
# Mudou a implementacao, nao o contrato.
from claims import (  # noqa: E402
    APELIDOS, CASAS, PADRAO_CAUSAL, PADRAO_DIRECAO, PADRAO_NUMERO,
    TERMOS_PREVISAO, ContextoAutorizado, avaliar_frase, extrair_claims,
    numero_bate as _numero_bate,
    numeros_permitidos as _numeros_permitidos,
    resumo_de_claims,
    sem_acento as _sem_acento,
    variaveis_citadas as _variaveis_citadas,
    verificar_claims,
)
from claims import _chaves, _textos_do_payload  # noqa: E402,F401

# Nome antigo, mantido porque `auditar_motor_local.py` e os testes o citam.
TERMOS_PROIBIDOS = TERMOS_PREVISAO



# --------------------------------------------------------- o que o motor ve

def resumo_para_o_motor(resposta: dict) -> dict:
    """Lista branca do que o gerador de texto pode ver.

    Mandar o payload inteiro seria mais facil e estaria errado: `camadas` cru
    nao esta la (o contrato nunca ecoa a requisicao), mas `replicas_bootstrap`
    e o resto do artefato servido nao tem por que atravessar esta fronteira.
    Lista branca em vez de lista negra porque campo novo no payload deve
    aparecer aqui por decisao, e nao por esquecimento.
    """
    if _eh_payload_social(resposta):
        return _resumo_social_para_o_motor(resposta)
    if resposta.get("status") != "ok":
        return {
            "status": resposta.get("status"),
            "gate_que_nao_fechou": resposta.get("gate_que_nao_fechou"),
            "detalhe": resposta.get("detalhe"),
            "regiao": resposta.get("regiao"),
            "maturidade": resposta.get("maturidade"),
            "contrato": resposta.get("contrato"),
        }
    mc = resposta.get("model_card") or {}
    return {
        "status": "ok",
        "contrato": resposta.get("contrato"),
        "regiao": resposta.get("regiao"),
        "maturidade": resposta.get("maturidade"),
        "unidade_de_resposta": resposta.get("unidade_de_resposta"),
        "escore": resposta.get("escore"),
        "ic95": resposta.get("ic95"),
        "n_pontos": resposta.get("n_pontos"),
        "variaveis_usadas": resposta.get("variaveis_usadas"),
        "dominio": resposta.get("dominio"),
        "limitacoes": resposta.get("limitacoes"),
        "explicacao_por_regra": (resposta.get("explicacao") or {}).get("texto"),
        "itens": (resposta.get("explicacao") or {}).get("itens"),
        "variaveis_fisicamente_verificadas": mc.get(
            "variaveis_fisicamente_verificadas", []),
        "mecanismo": mc.get("mecanismo"),
        "desempenho": mc.get("desempenho"),
        "criterios_nao_atingidos": mc.get("criterios_nao_atingidos"),
        "limites_de_uso": mc.get("limites_de_uso"),
        # A dimensao temporal entra na lista branca porque a LLM precisa poder
        # EXPLICAR o estado -- e a unica forma de ela nunca o decidir e ele
        # chegar pronto. O que atravessa e o estado, a razao e o efeito; a
        # decisao ja foi tomada pela matriz cientifica, fora daqui.
        "suporte_temporal": _suporte_temporal_para_o_motor(resposta),
    }


def eh_payload_social(resposta: dict) -> bool:
    """O payload e uma resposta da camada social, e nao da inferencia fisica?

    Publica de proposito: `consulta_cognitiva` precisa da MESMA resposta para
    decidir se o modo pedido combina com o payload recebido. Duas copias desta
    pergunta divergiriam, e a divergencia apareceria do pior jeito possivel --
    o modo social montando o resumo fisico e publicando prosa de escore para
    uma pergunta sobre quem pode ser mais afetado.
    """
    return (
        resposta.get("contrato") == "social_v0_review_only"
        or "prioridade_publica" in resposta
        or "vulnerabilidade_exposicao" in (resposta.get("eixos") or {})
    )


# Nome anterior, mantido: era privado e so tinha uso interno, mas apagar um
# ponto de entrada nunca e de graca.
_eh_payload_social = eh_payload_social


def _resumo_social_para_o_motor(resposta: dict) -> dict:
    eixos = resposta.get("eixos") or {}
    social = eixos.get("vulnerabilidade_exposicao") or {}
    fisica = eixos.get("suscetibilidade_fisica") or {}
    lacuna = eixos.get("lacuna_de_conhecimento_publico") or {}
    prioridade = resposta.get("prioridade_publica") or {}
    return {
        "tipo_de_payload": "social",
        "status": resposta.get("status"),
        "contrato": resposta.get("contrato"),
        "unidade_de_resposta": resposta.get("unidade_de_resposta"),
        "regiao_social": resposta.get("regiao_social"),
        "aoi": resposta.get("aoi"),
        "fisica": {
            "status": fisica.get("status"),
            "escore_recebido": fisica.get("escore_recebido"),
            "classe_relativa": fisica.get("classe_relativa"),
            "maturidade": fisica.get("maturidade"),
            "alterado_pela_camada_social": fisica.get(
                "alterado_pela_camada_social"),
        },
        "vulnerabilidade_exposicao": {
            "status": social.get("status"),
            "cobertura_da_aoi": social.get("cobertura_da_aoi"),
            "indicadores_agregados": social.get("indicadores_agregados"),
            "media_relativa": social.get("media_relativa"),
            "classe_relativa": social.get("classe_relativa"),
            "fontes": social.get("fontes"),
            "nao_e": social.get("nao_e"),
        },
        "lacuna_de_conhecimento_publico": {
            "status": lacuna.get("status"),
            "lacunas": lacuna.get("lacunas"),
            "nao_e": lacuna.get("nao_e"),
        },
        "prioridade_publica": {
            "classe": prioridade.get("classe"),
            "razao": prioridade.get("razao"),
            "nao_e": prioridade.get("nao_e"),
        },
        "invariantes": resposta.get("invariantes"),
        "pergunta_do_usuario": resposta.get("pergunta_do_usuario"),
        "instrucao_de_linguagem": (
            "Nao rotule a area como favela ou comunidade urbana. Quando a "
            "tipologia oficial for relevante, descreva como indicador agregado "
            "de vulnerabilidade territorial e cite a fonte.")
    }


def _suporte_temporal_para_o_motor(resposta: dict) -> dict | None:
    t = (resposta.get("suporte_cientifico") or {}).get("temporal")
    if not t:
        return None
    return {
        "estado": t.get("support"),
        "significa": t.get("support_significa") or t.get("motivo"),
        "nao_equivale_a": t.get("nao_equivale_a"),
        "razao_tecnica": t.get("razao_tecnica"),
        "escopo": t.get("scope"),
        "limitacoes": t.get("limitations"),
        "e_portao_do_contrato": t.get("e_portao_do_contrato"),
        "a_llm_pode": "explicar este estado",
        "a_llm_nao_pode": ("alterar, inferir ou completar este estado, nem "
                           "atribuir causa a degradacao temporal"),
    }


# ------------------------------------------------------------- verificador
#
# `_numeros_permitidos`, `_numero_bate`, `_textos_do_payload` e `_chaves`
# passaram para `claims.py` e sao reexportados no topo deste arquivo. Nada
# mudou no que eles fazem; mudou quem mais precisa deles.



def _contexto(resposta: dict, contexto=None) -> ContextoAutorizado:
    """Normaliza o contexto autorizado, para quem ainda passa so o payload.

    Chamador antigo (`verificar_frase(frase, resposta, permitidos)`) nao
    conhece contexto nenhum: recebe um `ContextoAutorizado` so com o payload,
    e todo claim que exigir RAG ou proveniencia sai `sem_suporte` -- que
    derruba a frase. Fail-closed no default, e nao no caso especial.
    """
    if isinstance(contexto, ContextoAutorizado):
        return contexto
    return ContextoAutorizado(
        resposta=resposta,
        trechos=list((contexto or {}).get("trechos") or []),
        proveniencia=(contexto or {}).get("proveniencia"),
        evidencia_visual=(contexto or {}).get("evidencia_visual")
        or resposta.get("evidencia_visual"),
        pergunta=(contexto or {}).get("pergunta"))


def verificar_frase(frase: str, resposta: dict,
                    permitidos=None, contexto=None) -> tuple:
    """Devolve (aprovada, motivo). Motivo so importa quando reprovada.

    A ASSINATURA NAO MUDOU, e o terceiro parametro ficou ignorado de
    proposito: `permitidos` era o conjunto de numeros pre-calculado, e quem
    calcula isso agora e o proprio `ContextoAutorizado`, sob demanda e em
    cache. Remover o parametro quebraria `auditar_motor_local.py` e uma duzia
    de testes por nenhum ganho; aceita-lo e ignora-lo custa uma linha e
    mantem o contrato interno de pe.
    """
    return tuple(avaliar_frase(frase, _contexto(resposta, contexto))
                 .para_dict().get(k) for k in ("aprovada", "motivo"))[0:2]


def claims_da_frase(frase: str, resposta: dict, contexto=None) -> dict:
    """Resumo achatado do que a frase tentou afirmar, sem veredito.

    Formato antigo, mantido porque o cartao e os testes o consomem. O registro
    COMPLETO -- tipo, predicado, valor, origens declaradas, estado e motivo --
    sai em `auditoria_sintese.frases[].claims_formais`, produzido por
    `claims.avaliar_frase`. Este aqui responde "o que a frase disse"; aquele
    responde "e o que disso ficou de pe".
    """
    ctx = _contexto(resposta, contexto)
    formais = extrair_claims(frase, ctx)
    direcao = None
    for c in formais:
        if c.tipo == "claim_direcao" and isinstance(c.valor, dict):
            direcao = c.valor["sentido"].replace(" ", "_").lower()
            break
    return {
        "claim_numero": [c.valor for c in formais if c.tipo == "claim_numero"],
        "claim_variavel": [c.valor for c in formais
                           if c.tipo == "claim_variavel" and c.valor],
        "claim_direcao": direcao,
        "claim_causal": any(c.tipo in ("claim_variavel", "claim_direcao")
                            for c in formais),
        "claim_maturidade": [c.valor for c in formais
                             if c.tipo == "claim_maturidade"],
        "claim_status": [c.valor for c in formais
                         if c.tipo == "claim_status"
                         and c.predicado == "status_da_resposta"],
        "claim_previsao": [c.valor for c in formais
                           if c.tipo == "claim_previsao"],
        "claim_fonte": [c.valor for c in formais if c.tipo == "claim_fonte"],
        "claim_social": [c.valor for c in formais if c.tipo == "claim_social"],
        "claim_evidencia_visual": [c.valor for c in formais
                                   if c.tipo == "claim_evidencia_visual"],
    }


def _pontuar(frase: str) -> str:
    """Fecha a frase, para que o paragrafo publicado seja legivel.

    Afirmacao vinda de saida estruturada costuma chegar sem ponto: o modelo
    entende `texto` como campo, nao como frase. Juntar sem pontuar produz uma
    parede unica que o leitor nao consegue segmentar -- e que, na proxima
    verificacao, voltaria a ser tratada como frase unica.
    """
    frase = frase.strip()
    return frase if not frase or frase[-1] in ".!?" else frase + "."


def verificar(texto: str, resposta: dict, contexto=None,
              segmentos: list | None = None) -> dict:
    """Passa frase a frase. Frase reprovada e descartada, nunca so marcada.

    A frase e a unidade de PUBLICACAO; o claim e a unidade de PROVA. Uma
    frase so sai quando todos os claims dela estao `verificado` -- e a
    auditoria guarda os dois niveis, para que "a frase caiu" venha sempre
    acompanhado de qual afirmacao a derrubou.

    `segmentos` E A FRONTEIRA QUE A SAIDA ESTRUTURADA ENTREGA, e vale a pena
    dizer por que ela nao e um detalhe. Segmentar prosa por `[.!?]` funciona
    enquanto o gerador escreve como gente. Rodando o qwen2.5:3b com
    `response_format`, as afirmacoes voltaram corretamente separadas no JSON e
    SEM ponto final -- o modelo leu `texto` como campo, nao como frase. O
    split por regex colou as nove numa unica frase, e a primeira rejeicao
    derrubou as oito restantes junto, incluindo as verificaveis.

    Perder frase correta e tao grave quanto publicar frase errada: apaga
    informacao boa sem deixar rastro. Quando o gerador ja disse onde uma
    afirmacao termina, essa fronteira e melhor que qualquer heuristica de
    pontuacao -- e continua sendo so uma fronteira de TEXTO: nada do que o
    gerador declara sobre o conteudo dela e aceito sem verificacao.
    """
    ctx = _contexto(resposta, contexto)
    if segmentos:
        frases = [_pontuar(f) for f in segmentos if str(f).strip()]
    else:
        frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+", texto or "")
                  if f.strip()]
    julgadas, aprovadas, todos = [], [], []
    for frase in frases:
        v = avaliar_frase(frase, ctx)
        todos.extend(v.claims)
        julgadas.append({
            "frase": frase,
            "aprovada": v.aprovada,
            "motivo": v.motivo or None,
            "claims": claims_da_frase(frase, resposta, ctx),
            "claims_formais": [c.para_dict() for c in v.claims],
        })
        if v.aprovada:
            aprovadas.append(frase)
    descartados = ctx.trechos_com_injecao()
    saida = {
        "frases": julgadas,
        "n_aprovadas": len(aprovadas),
        "n_reprovadas": len(frases) - len(aprovadas),
        "texto_aprovado": " ".join(aprovadas),
        "claims": resumo_de_claims(todos),
        "regra": ("frase so e publicada quando todos os claims dela estao "
                  "verificados contra payload, model card, trecho recuperado "
                  "ou manifesto de proveniencia"),
    }
    if descartados:
        # Um documento do corpus que fala em imperativo com o sistema nao e
        # sanitizado nem reescrito: sai do contexto e aparece aqui. Reescrever
        # conteudo suspeito para "deixar seguro" e apostar que a reescrita
        # cobre todos os casos.
        saida["trechos_descartados_por_injecao"] = [
            {k: t.get(k) for k in ("chunk_id", "source_path", "titulo",
                                   "injecao_detectada")}
            for t in descartados]
    return saida



# ------------------------------------------------------------------ motores

class MotorDeReferencia:
    """Deterministico, sem rede e sem GPU. E o padrao.

    Nao e uma LLM e nao finge ser. Monta o paragrafo das mesmas grandezas que o
    payload ja tem, na mesma ordem, com as mesmas palavras -- entao passa no
    verificador por construcao, o que o torna o alvo certo para testar o
    verificador sem depender de gerador externo.
    """

    nome = "referencia_deterministica"
    e_llm = False

    def gerar(self, resumo: dict) -> str:
        if resumo.get("tipo_de_payload") == "social":
            social = resumo["vulnerabilidade_exposicao"]
            prioridade = resumo["prioridade_publica"]
            fisica = resumo["fisica"]
            return (
                "A area consultada tem vulnerabilidade/exposicao social "
                f"{social.get('classe_relativa')}, com cobertura social "
                "declarada no payload. "
                f"A prioridade publica e {prioridade.get('classe')}. "
                "Essa leitura nao altera o escore fisico recebido."
            )
        if resumo.get("status") != "ok":
            return (f"A requisicao foi recusada no portao "
                    f"{resumo.get('gate_que_nao_fechou')}: "
                    f"{resumo.get('detalhe')}. Nenhum escore foi calculado.")
        partes = [
            f"A area consultada em {resumo['regiao']} recebeu escore de "
            f"suscetibilidade {resumo['escore']}, com intervalo de 95% entre "
            f"{resumo['ic95'][0]} e {resumo['ic95'][1]}.",
            f"A resposta vale para a area inteira, e a regiao esta em "
            f"maturidade {resumo['maturidade']}.",
        ]
        if resumo.get("explicacao_por_regra"):
            partes.append(resumo["explicacao_por_regra"])
        if resumo.get("criterios_nao_atingidos"):
            partes.append("O modelo desta regiao nao atinge "
                          f"{len(resumo['criterios_nao_atingidos'])} criterio(s) "
                          "de leitura fixados antes do ajuste.")
        return " ".join(partes)


class MotorHTTP:
    """Adaptador para API comercial. Nunca acionado por teste.

    So age com chave no ambiente E `permitir_egresso=True` -- duas condicoes,
    porque uma chave esquecida no ambiente nao pode virar chamada de rede
    dentro de um repositorio que e fail-closed por regra.
    """

    e_llm = True

    def __init__(self, modelo: str | None = None,
                 variavel_da_chave: str = "REV_P_MODELO_API_KEY",
                 permitir_egresso: bool = False):
        self.modelo = modelo or os.environ.get("REV_P_MODELO_HTTP", "")
        self.variavel_da_chave = variavel_da_chave
        self.permitir_egresso = permitir_egresso
        self.nome = f"http:{self.modelo}"

    def disponivel(self) -> tuple[bool, str]:
        if not self.permitir_egresso:
            return False, "egresso nao autorizado nesta chamada"
        if not self.modelo:
            return False, "modelo HTTP nao configurado"
        if not os.environ.get(self.variavel_da_chave):
            return False, f"{self.variavel_da_chave} ausente no ambiente"
        return True, ""

    def gerar(self, resumo: dict) -> str:
        ok, motivo = self.disponivel()
        if not ok:
            raise RuntimeError(f"motor indisponivel: {motivo}")
        raise NotImplementedError(
            "adaptador declarado, chamada nao implementada: a escolha do motor "
            "ficou em aberto. Implementar aqui nao muda nada no contrato nem "
            "no verificador -- e essa e a razao de a fronteira existir")


PROMPT_SISTEMA = (
    "Voce escreve um paragrafo curto em portugues sobre suscetibilidade a "
    "enchente, usando SOMENTE os numeros e nomes que aparecem no JSON a "
    "seguir. Nao invente numero nenhum: todo numero citado tem que aparecer, "
    "com o mesmo arredondamento, em `escore`, `ic95`, `n_pontos`, `dominio` "
    "ou nos coeficientes do `model_card`. Se disser que uma variavel 'pesa a "
    "favor' ou 'pesa contra', ela precisa estar em "
    "`variaveis_fisicamente_verificadas` e bater com o `sentido` que os "
    "`itens` de `explicacao_por_regra` ja declaram. Nunca afirme previsao de "
    "evento (nunca diga 'vai alagar', 'certamente', 'evacue') -- o campo "
    "responde suscetibilidade do terreno, nao previsao. Se `status` nao for "
    "`ok`, explique a recusa pelo `gate_que_nao_fechou` e `detalhe`, sem "
    "inventar escore. Se o JSON tiver `pergunta_do_usuario`, responda A ELA "
    "usando apenas o que esta no JSON; se o JSON nao contiver o que a "
    "pergunta pede, diga que a informacao nao esta disponivel em vez de "
    "preencher. Se houver `trechos_recuperados`, use apenas o texto deles "
    "para afirmacoes de literatura, e cite o DOI ou URL que aparece no "
    "proprio trecho -- nunca um que voce conheca de outro lugar. Se o JSON "
    "for `tipo_de_payload: social`, responda sobre vulnerabilidade/exposicao, "
    "lacunas e prioridade publica usando apenas o payload social; nunca diga "
    "que a area 'e favela' ou 'e comunidade urbana'. A tipologia oficial pode "
    "ser descrita como indicador agregado de vulnerabilidade territorial, com "
    "fonte, mas nao como etiqueta da area."
)


PROMPT_SISTEMA_ESTRUTURADO = (
    PROMPT_SISTEMA + " Responda APENAS com um objeto JSON no formato "
    '{"afirmacoes": [{"texto": ..., "tipo": ..., "origem_declarada": ...}]}. '
    "Cada afirmacao deve ser UMA frase completa em portugues, terminada em "
    "ponto, que possa ser lida sozinha. `tipo` e um de: numero, maturidade, "
    "status, variavel, direcao, limitacao, proveniencia, fonte, social. "
    "`origem_declarada` e o nome do campo do JSON de onde voce tirou a "
    "afirmacao. Nao escreva nada fora do JSON."
)

# Schema rigido: `additionalProperties: false` e todos os campos obrigatorios.
# Schema frouxo faz o modelo devolver algo que valida e nao serve, e o parser
# vira adivinhacao.
FORMATO_CLAIMS = {
    "type": "json_schema",
    "json_schema": {
        "name": "afirmacoes_revp",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "afirmacoes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "texto": {"type": "string"},
                            "tipo": {
                                "type": "string",
                            "enum": ["numero", "maturidade", "status",
                                     "variavel", "direcao", "limitacao",
                                     "proveniencia", "fonte", "social"],
                            },
                            "origem_declarada": {"type": "string"},
                        },
                        "required": ["texto", "tipo", "origem_declarada"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["afirmacoes"],
            "additionalProperties": False,
        },
    },
}

CERCA_MARKDOWN = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _afirmacoes_do_json(bruto: str) -> list:
    """Le a saida estruturada, e NAO confia nela por ser JSON valido.

    Um modelo pequeno devolve, com frequencia, JSON perfeitamente formado com
    o campo errado, texto vazio ou a lista fora do envelope. Cada uma dessas
    formas e tratada aqui como falha de schema -- que cai para prosa -- e nao
    como "quase certo, da para aproveitar". Aproveitar meia estrutura seria a
    porta dos fundos pela qual o sistema volta a confiar no gerador.
    """
    import json as _json

    texto = CERCA_MARKDOWN.sub("", (bruto or "").strip()).strip()
    dados = _json.loads(texto)
    if not isinstance(dados, dict) or "afirmacoes" not in dados:
        raise ValueError("JSON sem a chave `afirmacoes`")
    itens = dados["afirmacoes"]
    if not isinstance(itens, list) or not itens:
        raise ValueError("`afirmacoes` nao e lista nao vazia")
    saida = []
    for item in itens:
        if not isinstance(item, dict) or not str(item.get("texto") or "").strip():
            raise ValueError("afirmacao sem texto")
        saida.append({
            "texto": str(item["texto"]).strip(),
            "tipo": str(item.get("tipo") or "").strip() or None,
            "origem_declarada": str(item.get("origem_declarada") or "").strip()
            or None,
        })
    return saida


class MotorLocal:
    """Adaptador para servidor HTTP local com API de chat compatível.

    A diferenca central para o `MotorHTTP` e o que esta em jogo: aqui nao ha
    egresso -- o servidor roda na propria maquina (Ollama, llama.cpp --server,
    LM Studio, ou qualquer coisa que fale o mesmo formato de
    `/chat/completions`), entao nao existe a trava de `permitir_egresso`. O
    que existe em troca e um timeout curto: CPU sem GPU (10 nucleos, 15,5 GB,
    Tabela I do artigo) pode ser lento, e uma sintese que nunca chega nao pode
    travar `/inferencia`. Quem chama este motor via `motor_padrao()` recebe
    fallback automatico para `MotorDeReferencia` quando o servidor esta fora
    do ar ou estoura o timeout -- ver `MotorComFallback`.
    """

    e_llm = True

    def __init__(self, url: str | None = None, modelo: str | None = None,
                 timeout: float | None = None, estruturado: bool | None = None):
        self.url = url if url is not None else os.environ.get("LOCAL_LLM_URL")
        self.modelo = modelo or os.environ.get("LOCAL_LLM_MODEL", "local")
        self.timeout = (timeout if timeout is not None else
                        float(os.environ.get("LOCAL_LLM_TIMEOUT", "30")))
        self.nome = f"local:{self.modelo}"
        self.estruturado = (
            estruturado if estruturado is not None
            else os.environ.get("LOCAL_LLM_ESTRUTURADO", "1") not in ("0", ""))
        # `None` = ainda nao testado. Vira True/False na primeira chamada, e e
        # o que a auditoria publica como `capacidades`. Um campo que ja
        # nascesse `False` diria "nao suporta" antes de alguem ter perguntado.
        self.capacidades = {"structured_output": None}
        self.modo_ultimo = None
        self.ultimas_afirmacoes = []

    def disponivel(self) -> tuple[bool, str]:
        if not self.url:
            return False, "LOCAL_LLM_URL nao configurada"
        return True, ""

    def _chamar(self, corpo: dict) -> dict:
        import json as _json
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            self.url, data=_json.dumps(corpo).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return _json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detalhe = ""
            try:
                detalhe = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            raise RuntimeError(
                f"servidor local recusou: HTTP {e.code} {detalhe}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError(f"servidor local inacessivel: {e}") from e

    @staticmethod
    def _conteudo(payload: dict) -> str:
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"resposta do servidor local em formato inesperado: {e}") from e

    def gerar(self, resumo: dict) -> str:
        """Tenta claims estruturados; cai para prosa livre se nao der.

        A ORDEM IMPORTA, e a razao nao e desempenho. Pedir JSON com schema faz
        o modelo COMPROMETER-SE com afirmacoes discretas -- cada uma com o
        texto que sera publicado e a origem que ele ALEGA ter usado. Isso nao
        torna nenhuma afirmacao verdadeira (o verificador continua resolvendo
        tudo sozinho), mas transforma "uma parede de prosa que passou ou nao"
        em "seis afirmacoes, quatro verificadas e duas rejeitadas, e o que
        cada uma alegava". A auditoria fica comparavel entre modelos, que e o
        que o REV-P-EVAL precisa medir.

        Quando o servidor nao aceita `response_format` -- versao antiga, outro
        runtime, modelo sem suporte -- a capacidade e registrada como `False`
        e a MESMA chamada segue em prosa. Fallback declarado, nunca disfarce:
        `capacidades` e `modo_ultimo` dizem o que aconteceu.
        """
        import json as _json

        ok, motivo = self.disponivel()
        if not ok:
            raise RuntimeError(f"motor indisponivel: {motivo}")

        self.ultimas_afirmacoes = []
        entrada = {"role": "user",
                   "content": _json.dumps(resumo, ensure_ascii=False)}
        base = {
            "model": self.modelo,
            "messages": [{"role": "system", "content": PROMPT_SISTEMA}, entrada],
            "temperature": 0,
        }

        if self.estruturado and self.capacidades["structured_output"] is not False:
            corpo = {
                "model": self.modelo,
                "messages": [
                    {"role": "system", "content": PROMPT_SISTEMA_ESTRUTURADO},
                    entrada,
                ],
                "temperature": 0,
                "response_format": FORMATO_CLAIMS,
            }
            try:
                afirmacoes = _afirmacoes_do_json(self._conteudo(self._chamar(corpo)))
                self.capacidades["structured_output"] = True
                self.modo_ultimo = "json_schema"
                self.ultimas_afirmacoes = afirmacoes
                return " ".join(a["texto"].strip() for a in afirmacoes
                                if a.get("texto"))
            except RuntimeError as e:
                if "inacessivel" in str(e):
                    raise        # servidor fora do ar nao e falta de schema
                self.capacidades["structured_output"] = False
            except (ValueError, TypeError, KeyError):
                # JSON valido nao e JSON correto: schema errado tambem cai
                # para prosa, em vez de publicar uma estrutura que ninguem
                # sabe ler.
                self.capacidades["structured_output"] = False

        self.modo_ultimo = "prosa"
        return self._conteudo(self._chamar(base))


class MotorComFallback:
    """Tenta um motor primario; se ele falhar, usa `MotorDeReferencia`.

    A troca acontece dentro de uma unica chamada de `gerar()`, mas nunca fica
    escondida: depois de gerar, `.nome` e `.e_llm` refletem o motor que
    REALMENTE produziu o texto -- nunca o que foi pedido primeiro.
    `sintetizar` le esses dois atributos depois do `gerar()` para montar
    `auditoria_sintese`, entao a troca aparece la como
    `motor: "referencia_deterministica"`, `e_llm: false`, igual a qualquer
    chamada que nunca tentou LLM nenhuma. `.motor_pedido`,
    `.fallback_ativado` e `.motivo_fallback` guardam o que foi tentado, para
    quem quiser auditar -- essa e a diferenca entre "trocar por baixo do
    pano" e "trocar e declarar".
    """

    def __init__(self, primario, alternativo=None):
        self.primario = primario
        self.alternativo = alternativo or MotorDeReferencia()
        self.motor_pedido = (getattr(primario, "nome", None)
                             if primario else None)
        self.fallback_ativado = False
        self.motivo_fallback = None
        self.nome = self.motor_pedido or self.alternativo.nome
        self.e_llm = (bool(getattr(primario, "e_llm", False))
                      if primario else False)

    def gerar(self, resumo: dict) -> str:
        if self.primario is not None:
            disponivel, motivo = True, ""
            checar = getattr(self.primario, "disponivel", None)
            if callable(checar):
                disponivel, motivo = checar()
            if disponivel:
                try:
                    texto = self.primario.gerar(resumo)
                    self.nome = self.primario.nome
                    self.e_llm = bool(getattr(self.primario, "e_llm", False))
                    return texto
                except Exception as e:
                    motivo = f"{type(e).__name__}: {e}"
            self.fallback_ativado = True
            self.motivo_fallback = motivo or "motor primario indisponivel"
        self.nome = self.alternativo.nome
        self.e_llm = bool(getattr(self.alternativo, "e_llm", False))
        return self.alternativo.gerar(resumo)


def motor_padrao():
    """O motor que `sintetizar()` usa quando ninguem injeta um.

    Sem `LOCAL_LLM_URL` configurada, e o `MotorDeReferencia` puro -- exatamente
    o comportamento de antes desta decisao, e o que os testes existentes
    continuam exercitando sem mudanca. Com a variavel configurada, vira
    `MotorLocal` com fallback automatico para o mesmo `MotorDeReferencia`:
    a escolha de motor local, CPU-only, feita em 26/08.
    """
    url = os.environ.get("LOCAL_LLM_URL")
    if not url:
        return MotorDeReferencia()
    return MotorComFallback(MotorLocal())


# ------------------------------------------------------------------ camada

def sintetizar(resposta: dict, motor=None, evidencia: bool = True,
               contexto=None) -> dict:
    """Devolve a resposta com `sintese` e `auditoria_sintese` anexados.

    OS DOIS CAMPOS, e por que sao dois:

      `sintese`            str  -- a prosa que passou pelo verificador
                           None -- nenhuma frase passou, ou nao houve texto

      `auditoria_sintese`  dict -- motor, frase a frase julgada, motivo de cada
                           reprovacao, erro do motor e `por_que_null`

    `sintese = None` NAO e ausencia de resposta: e o resultado de o verificador
    ter reprovado tudo. Por isso ele nunca vira o texto determinstico por baixo
    do pano -- quem le a resposta precisa distinguir "a prosa foi aprovada" de
    "a prosa foi rejeitada e o que sobrou e a frase por regra". A versao
    anterior deste arquivo devolvia o determinstico dentro do proprio campo, e
    era exatamente essa distincao que se perdia.

    `explicacao` -- a frase por regra do E6 -- fica intacta nos dois casos, e e
    ela que a tela mostra quando `sintese` e nula.

    Aditivo por construcao: a resposta que entra nao e modificada.
    """
    motor = motor or motor_padrao()
    ctx = _contexto(resposta, contexto)
    resumo = resumo_para_o_motor(resposta)
    # A PERGUNTA PRECISA CHEGAR AO GERADOR, e a falta disso foi um defeito
    # real de integracao: a camada de consulta escolhia o modo pela pergunta,
    # montava o contexto certo... e mandava ao motor so o payload. O modelo
    # nunca via o que tinha sido perguntado, entao reprocessava o payload e
    # devolvia o MESMO paragrafo para "o que e HAND", "explique o IC95" e
    # "de onde veio este modelo". O benchmark mostrava cobertura 1,0 -- todas
    # as perguntas produziram texto publicavel -- enquanto nenhuma tinha sido
    # respondida. Cobertura sem pertinencia e uma metrica que mente.
    #
    # A pergunta entra como CONTEXTO, e nao como autoridade: nada do que ela
    # afirmar autoriza claim nenhum, e o verificador continua resolvendo tudo
    # contra payload, trecho e manifesto.
    if ctx.pergunta:
        resumo = {**resumo, "pergunta_do_usuario": ctx.pergunta}
    # Trecho recuperado entra no que o motor VE, mas ja sem os que tentavam
    # instruir o sistema: o corpus e leitura, nunca canal de instrucao.
    if ctx.trechos:
        resumo = {**resumo, "trechos_recuperados": [
            {k: t.get(k) for k in ("chunk_id", "titulo", "classe_autoridade",
                                   "doi", "url", "trecho")}
            for t in ctx.trechos_limpos()]}

    try:
        bruto = motor.gerar(resumo)
        erro = None
    except Exception as e:  # motor indisponivel nao pode derrubar a inferencia
        bruto, erro = "", f"{type(e).__name__}: {e}"

    afirmacoes = list(getattr(motor, "ultimas_afirmacoes", None) or [])
    segmentos = [a["texto"] for a in afirmacoes] or None
    checagem = verificar(bruto, resposta, ctx, segmentos) if bruto else {
        "frases": [], "n_aprovadas": 0, "n_reprovadas": 0, "texto_aprovado": "",
        "claims": resumo_de_claims([]),
        "regra": "motor nao produziu texto"}
    if afirmacoes:
        # A origem que o GERADOR alegou, lado a lado com a que o verificador
        # ENCONTROU (em `claims_formais[].evidencias_verificadas`). Sao dois
        # campos porque sao duas coisas: no dia em que virarem um so, o
        # sistema tera voltado a confiar no gerador sem ninguem notar.
        for item, a in zip(checagem["frases"], afirmacoes):
            item["origem_declarada_pelo_gerador"] = a.get("origem_declarada")
            item["tipo_declarado_pelo_gerador"] = a.get("tipo")

    sintese = checagem["texto_aprovado"] or None

    if sintese:
        por_que_null = None
    elif erro:
        por_que_null = f"o motor nao produziu texto: {erro}"
    elif checagem["frases"]:
        por_que_null = (f"nenhuma das {len(checagem['frases'])} frase(s) do "
                        "gerador passou no verificador; o cartao mostra a "
                        "frase determinstica do proprio payload")
    else:
        por_que_null = "o motor nao produziu texto"

    auditoria = {
        "motor": getattr(motor, "nome", "desconhecido"),
        "e_llm": bool(getattr(motor, "e_llm", False)),
        "aprovadas": checagem["n_aprovadas"],
        "reprovadas": checagem["n_reprovadas"],
        "frases": checagem["frases"],
        "claims": checagem.get("claims"),
        "modo_do_gerador": getattr(motor, "modo_ultimo", None),
        "capacidades_do_motor": getattr(motor, "capacidades", None),
        "regra": checagem["regra"],
        "erro_do_motor": erro,
        "texto_bruto_do_gerador": bruto or None,
        "por_que_null": por_que_null,
        "nao_e": ("nao substitui o campo `explicacao`, que continua sendo a "
                  "frase gerada por regra sobre o payload"),
    }
    if checagem.get("trechos_descartados_por_injecao"):
        auditoria["trechos_descartados_por_injecao"] = checagem[
            "trechos_descartados_por_injecao"]
    if getattr(motor, "fallback_ativado", False):
        auditoria["fallback_ativado"] = True
        auditoria["motor_pedido"] = motor.motor_pedido
        auditoria["motivo_fallback"] = motor.motivo_fallback

    fora = {**resposta, "sintese": sintese, "auditoria_sintese": auditoria}
    if evidencia:
        fora["evidencia_visual"] = consultar_evidencia_visual(resposta)
    return fora


def anexar_evidencia_visual(resposta: dict) -> dict:
    """`evidencia_visual` sozinha, para quem ja sintetizou.

    Existe para a rota `/explicacao` nao gerar o texto duas vezes: ela recebe
    do `/inferencia` uma resposta que ja passou pelo verificador, e so precisa
    acrescentar a consulta de evidencia.
    """
    if "evidencia_visual" in resposta:
        return resposta
    return {**resposta, "evidencia_visual": consultar_evidencia_visual(resposta)}
