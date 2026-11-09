"""Precedencia documental: qual documento VIGE, e qual foi superado.

POR QUE ISTO E UM BURACO REAL DO RAG, e nao mais um filtro:

`rag_corpus` separa QUEM PODE FALAR -- as tres classes de autoridade. Nenhuma
delas separa QUANDO. O projeto produziu sete documentos em cinco dias, e por
similaridade de texto um documento de 27/08 responde perfeitamente bem sobre um
estado que mudou em 31/08. A recuperacao acerta o trecho e erra a epoca, e o
verificador de citacao aprova, porque o trecho existe mesmo.

Este modulo nao substitui nada disso. Ele acrescenta o eixo que faltava, e
`rag_busca` passa a poder anotar cada trecho com o estado do documento de onde
ele veio.

-------------------------------------------------------------------------
CINCO ESTADOS, E NENHUM DELES E INFERIDO DO TEXTO
-------------------------------------------------------------------------

  vigente       o documento mais recente da serie, sem ninguem o superando
  superado      OUTRO documento declara, por escrito, que o substitui ou revisa
  historico     documento anterior de uma serie datada, sem supersessao
                declarada: descreve um estado que era verdade naquele dia
  experimental  o proprio documento declara que especifica algo NAO SERVIDO
  draft         o proprio documento se declara rascunho

A regra que sustenta os cinco e a mesma de `CLASSE_POR_CAMINHO`: a relacao e
DECLARADA no ponto de curadoria ou no cabecalho do proprio documento, nunca
adivinhada por semelhanca. Inferir supersessao de similaridade deixaria um
documento se aposentar por acidente -- e o modo de falha seria silencioso, que
e o unico que este repositorio trata como inaceitavel.

-------------------------------------------------------------------------
O QUE ELE NAO FAZ
-------------------------------------------------------------------------

Nao apaga documento, nao remove trecho do indice e nao decide resposta. Um
documento superado continua citavel -- ele so passa a ser citavel COM A
ETIQUETA, e a resposta consegue dizer "o vigente e X; o anterior, Y, dizia
outra coisa".

Uso:
    python scripts/servico/precedencia_documental.py
    python scripts/servico/precedencia_documental.py --consultar docs/DIMENSAO_TEMPORAL_2026_08_28.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from proveniencia import agora_utc, sha256_arquivo  # noqa: E402

OUT = REPO / "local_runs" / "rag-precedencia"
INDICE = OUT / "precedencia_documental.json"

VIGENTE = "vigente"
SUPERADO = "superado"
HISTORICO = "historico"
EXPERIMENTAL = "experimental"
DRAFT = "draft"
ESTADOS = (VIGENTE, SUPERADO, HISTORICO, EXPERIMENTAL, DRAFT)

DEFINICAO = {
    VIGENTE: "documento mais recente da serie, sem supersessao declarada",
    SUPERADO: "outro documento declara por escrito que o substitui ou revisa",
    HISTORICO: ("documento anterior de uma serie datada, sem supersessao "
                "declarada: descreve um estado que era verdade naquele dia"),
    EXPERIMENTAL: "o proprio documento declara especificar algo nao servido",
    DRAFT: "o proprio documento se declara rascunho",
}

PODE_SUSTENTAR_ESTADO_ATUAL = {VIGENTE: True, SUPERADO: False, HISTORICO: False,
                               EXPERIMENTAL: False, DRAFT: False}

# SERIES curadas. Duas datas nao fazem uma serie: o que faz e responderem a
# MESMA pergunta. Agrupar por semelhanca de titulo juntaria `CHUVA_ESTADO` com
# `CIENCIA_DO_FENOMENO` -- que falam de chuva e respondem coisas diferentes.
SERIES = {
    # `CRONOGRAMA_V2` foi REMOVIDO do projeto em 09/09. Ele declarava-se
    # superado pelo proprio cabecalho da v3, e a decisao foi apagar o superado
    # em vez de mante-lo citavel com etiqueta. Fica o registro de que a serie
    # teve um elo entre o roadmap de 27/08 e a v3.
    "cronograma": [
        "docs/ROADMAP_PRODUTO_TCC_PROGRAMACAO_CIENTIFICA_2026_08_27.md",
        "docs/CRONOGRAMA_V3_2026_09_05_A_17_11.md",
    ],
    "contrato_de_inferencia": [
        "docs/CONTRATO_VERSIONAMENTO.md",
        "docs/CONTRATO_V2_CANDIDATO_TEMPORAL.md",
    ],
    # A ADR e serie propria: ela DECIDE sobre o contrato candidato sem ser uma
    # versao dele. Junta-las faria a decisao superar o que ela decide.
    "decisao_do_portao_temporal": [
        "docs/ADR_001_PORTAO_TEMPORAL_G6.md",
    ],
    "chuva_identificacao": [
        "docs/chuva/REV_P_CHUVA_VALIDACAO_COMPLETA.md",
        "docs/chuva/IDENTIFICABILIDADE_CAUSAL_2026_08_30.md",
        "docs/CHUVA_ESTADO_COMPLETO_2026_08_30.md",
        "docs/FORCAMENTO_AUTOPAREADO_2026_08_31.md",
    ],
    # A SUPERSESSAO AQUI E PARCIAL, e por isso o documento fica.
    #
    # Na limpeza de 09/09 este documento chegou a ser apagado por sair como
    # `superado` no indice. Foi restaurado: o `FORCAMENTO_AUTOPAREADO` declara
    # que revisa a §8 -- os dois ganhos de AUC medidos em janela descasada --
    # e diz, na mesma linha, que "o restante daquele documento (MDT, promocao,
    # contagens) permanece vigente". Sao 7 das 8 secoes, e elas guardam os dois
    # MDT LiDAR reais e a validacao que sustenta SP156 e NYC311 como fontes.
    #
    # O indice marca o ARQUIVO como superado porque o eixo dele e o documento
    # inteiro. A §8 carrega o aviso no proprio corpo; e la que a etiqueta cabe.
    "promocao_de_fontes": [
        "docs/AQUISICAO_E_MUDANCAS_2026_08_31.md",
        "docs/PROMOCAO_SP156_NYC311_2026_08_31.md",
    ],
    # Duas series, e nao uma: `FRONTEIRA_DE_VALIDADE` responde ate onde o
    # modelo servido vale; `DIMENSAO_TEMPORAL` define a ontologia de suporte.
    # Mesma data e assunto vizinho nao fazem uma delas superar a outra.
    "fronteira_de_validade": [
        "docs/FRONTEIRA_DE_VALIDADE_2026_08_28.md",
    ],
    "dimensao_temporal": [
        "docs/DIMENSAO_TEMPORAL_2026_08_28.md",
    ],
    "execucao_da_sessao": [
        "docs/EXECUCAO_2026_08_31_FORCAMENTO_E_PRECEDENCIA.md",
    ],
    "camada_cognitiva": [
        "docs/CAMADA_COGNITIVA_AUDITAVEL_2026_08_27.md",
        "docs/IA_CIENTIFICA_MEMORIA_AUDITAVEL_2026_09_01.md",
    ],
    "frontend": [
        "docs/FRONTEND_CIENTIFICO_F1_F2_2026_08_28.md",
    ],
    "estado_do_modelo": [
        "outputs_public/model/ESTADO_DO_MODELO.md",
    ],
    "estado_tecnico": [
        "docs/DOCUMENTACAO_TECNICA_CONSOLIDADA.md",
        "docs/RELATORIO_PROGRAMACAO_PRODUTO_TCC_2026_08_27.md",
    ],
    "ciencia_do_fenomeno": [
        "docs/CIENCIA_DO_FENOMENO_2026_08_30.md",
    ],
    "execucao_local": [
        "docs/GUIA_EXECUCAO_PRODUTO_LOCAL.md",
    ],
    "pesquisa_com_usuario": [
        "docs/pesquisa_usuario/protocolo_pesquisa_usuario_cartao_resposta_v1.md",
    ],
    "conceitos": [
        "docs/CONCEITOS_MAL_POSICIONADOS_2026_08_30.md",
    ],
    # Duas series para o eixo social, e nao uma. `CAMADA_SOCIAL_TERRITORIAL`
    # responde o que a camada social FAZ -- contrato v0, dominios, o piloto de
    # Curitiba. `REALINHAMENTO_EIXO_HUMANO` responde qual e o EIXO do produto e
    # quem e o publico. Mesma data e assunto vizinho nao fazem uma superar a
    # outra, pela mesma razao de `fronteira_de_validade` e `dimensao_temporal`.
    #
    "camada_social": [
        "docs/CAMADA_SOCIAL_TERRITORIAL_2026_09_01.md",
    ],
    # `IA_CIENTIFICA_MEMORIA_AUDITAVEL` entra na serie da camada cognitiva, e
    # nao numa propria: ele responde a MESMA pergunta que
    # `CAMADA_COGNITIVA_AUDITAVEL` -- como a LLM adquire conhecimento sem ganhar
    # autoridade -- e a primeira linha dele diz que realinha aquele documento.
    # Sendo mais recente e sem supersessao declarada por escrito, ele passa a
    # `vigente` e o de 27/08 vira `historico`: continua citavel, com etiqueta,
    # e para de sustentar afirmacao sobre o estado ATUAL da camada cognitiva.
    "eixo_humano": [
        "docs/REALINHAMENTO_EIXO_HUMANO_POLITICAS_PUBLICAS_2026_09_01.md",
    ],
    # ------------------------------------------------------------------
    # 02/09 a 07/09, e por que cada um vem SOZINHO na sua serie.
    #
    # Esta tabela parou de ser atualizada em 01/09 e treze documentos ficaram
    # invisiveis para o indice. O efeito nao foi neutro: o `CRONOGRAMA_V2`
    # continuou saindo como `vigente` mesmo com a v3 declarando, por escrito,
    # que o substituia -- o motor nao conseguia ver quem o superava.
    #
    # Nenhum dos treze declara supersessao no cabecalho. Foi verificado um a
    # um, e nao inferido do assunto. Entao cada um entra em serie propria e
    # sai como `vigente, unico da serie`, que e o que o arquivo sustenta.
    #
    # Agrupar por tema seria conveniente e ERRADO: juntar os quatro documentos
    # de chuva de 02/09 faria o ultimo aposentar os outros tres por data, e a
    # regra do modulo e que supersessao se DECLARA. Quando algum deles de fato
    # substituir outro, quem declara e o cabecalho -- nao esta tabela.
    "plano_ia_geoespacial": [
        "docs/PLANO_IA_GEOESPACIAL_APRENDIZADO_AUTONOMO.md",
    ],
    "frentes_neurais": [
        "docs/PLANO_FRENTES_NEURAIS.md",
    ],
    "auto_supervisao_e_escala": [
        "docs/REALINHAMENTO_AUTO_SUPERVISAO_E_ESCALA_2026_09_03.md",
    ],
    "aprendizado_acumulativo": [
        "docs/APRENDIZADO_ACUMULATIVO_2026_09_04.md",
    ],
    "pre_registro_do_teste_de_postos": [
        "docs/PRE_REGISTRO_TESTE_DE_POSTOS_2026_09_04.md",
    ],
    "teste_da_lei": [
        "docs/TESTE_DA_LEI_CHICAGO_2026_09_04.md",
    ],
    "chuva_qual_dado_ensina": [
        "docs/CHUVA_QUAL_DADO_ENSINA_2026_09_02.md",
    ],
    "plano_da_chuva_e_base_areal": [
        "docs/PLANO_CHUVA_E_BASE_AREAL_2026_09_02.md",
    ],
    "gatilho_da_chuva": [
        "docs/GATILHO_DA_CHUVA_PLANO_2026_09_02.md",
    ],
    "execucao_do_plano_da_chuva": [
        "docs/EXECUCAO_PLANO_CHUVA_2026_09_02.md",
    ],
    "arvore_do_projeto": [
        "docs/REV_P_ARVORE_COMPLETA_DO_PROJETO_2026_09_03_ATUALIZADA.md",
    ],
    "comunicacao_social": [
        "docs/PERGUNTAS_TRABALHO_E_COMUNICACAO_SOCIAL_2026_09_03.md",
    ],
    # O master de 07/09 e o mais recente do projeto e declara o proprio papel:
    # planejamento e poster sao contexto, e o estado tecnico e determinado
    # pelos artefatos. Serie propria -- ele nao supera documento nenhum, ele
    # MAPEIA todos.
    "documento_master": [
        "docs/DOCUMENTO_MASTER_TCC_REV_P_2026_09_07.md",
    ],
    "raiz_do_fracasso_pluvial": [
        "docs/RAIZ_DO_FRACASSO_PLUVIAL_2026_09_11.md",
    ],
    # Mesma serie, e o de 14/09 SUPERA o de 11/09 em dois pontos nomeados no
    # cabecalho dele: "o EMSR720 e a unica ativacao brasileira" (ha uma segunda,
    # EMSN194, em outro modulo do mesmo servico) e "19 eventos do GFD no
    # Brasil" (sao 87 os que tocam o territorio; 19 e quantos o DFO ROTULOU de
    # brasileiros). O resto do documento de 11/09 continua valendo, inclusive
    # as quatro ressalvas sobre o GFD -- por isso ele fica na lista, atras.
    "onde_tem_negativo_observado": [
        "docs/ONDE_O_BRASIL_TEM_NEGATIVO_OBSERVADO_2026_09_14.md",
        "docs/ONDE_TEM_NEGATIVO_OBSERVADO_2026_09_11.md",
    ],
    # Serie propria, e nao continuacao das duas acima: `raiz_do_fracasso`
    # diagnosticou que a raiz esta no negativo, `onde_tem_negativo_observado`
    # levantou onde ele existe, e este MEDE a diferenca que os dois supunham.
    # Ele nao supera nenhum dos dois -- eles seguem valendo, e este acrescenta
    # o numero que faltava.
    "o_que_o_rs_ensina": [
        "docs/O_QUE_O_RIO_GRANDE_DO_SUL_ENSINA_2026_09_13.md",
    ],
    # Serie propria. Ele CONTINUA `onde_tem_negativo_observado` -- aquele diz
    # onde o negativo existe, este diz o que o produto passou a responder com
    # ele -- mas nao o supera: os numeros de cobertura nacional de 14/09
    # seguem valendo e sao citados daqui.
    "brasil_servido_por_macrorregiao": [
        "docs/O_BRASIL_SERVIDO_POR_MACRORREGIAO_2026_09_17.md",
    ],
    # O fechamento do marco N0. Serie propria: ele nao supera documento nenhum,
    # e o cronograma v3 que o previu continua vigente -- o N0 e uma linha da
    # tabela de marcos daquele documento, cumprida.
    "n0_arvore_limpa": [
        "docs/N0_ARVORE_LIMPA_E_SUITE_REGISTRADA_2026_09_11.md",
    ],
    # O fechamento do marco G6. Nao entra na serie do contrato nem na da ADR:
    # ele RELATA a execucao dos seis passos, e um relato que superasse a
    # decisao faria o registro de como se decidiu desaparecer do indice.
    "g6_adocao": [
        "docs/G6_ADOCAO_DA_POLITICA_TEMPORAL_2026_09_12.md",
    ],
    # ------------------------------------------------------------------
    # 01/10 a 04/10. A tabela parou de novo em 17/09, e o estado atual do
    # projeto -- admissao por finalidade, lacre aberto, generalizacao por
    # evento, o primeiro teste confirmatorio -- ficou fora do indice. Cada um
    # entra em serie propria, pela mesma regra de 02-07/09: nenhum declara
    # supersessao alem da que o proprio cabecalho escreve.
    #
    # A unica supersessao desta leva e a do `CIENCIA_DO_APRENDIZADO`, e ela
    # vem do cabecalho dele: revisa o REALINHAMENTO (o achado "confiabilidade
    # de convergencia" era o ponto de parada, nao a inicializacao) e o
    # APRENDIZADO_ACUMULATIVO (o 6/6 compara uma epoca sobre pote diverso, nao
    # rodadas encadeadas). Os dois saem como `superado` no documento inteiro --
    # o eixo deste indice e o arquivo --, e o aviso trecho a trecho fica no
    # corpo deles, onde a etiqueta cabe.
    "ciencia_do_aprendizado": [
        "docs/CIENCIA_DO_APRENDIZADO_2026_10_03.md",
    ],
    "pre_registro_transferencia_pretreino": [
        "docs/PRE_REGISTRO_TRANSFERENCIA_PRETREINO_2026_10_03.md",
    ],
    # Gerado de artefatos, nao editado a mao: e o retrato do estado atual.
    "fechamento_cientifico": [
        "docs/FECHAMENTO_CIENTIFICO_OPERACIONAL.md",
    ],
    "campanha_integrada": [
        "docs/CAMPANHA_INTEGRADA_2026_10_03.md",
    ],
    "admissao_cientifica": [
        "docs/ADMISSAO_CIENTIFICA.md",
    ],
    "contrato_do_fenomeno": [
        "docs/CONTRATO_DO_FENOMENO.md",
    ],
}

# COMMITS REVISADOS. Commit nao tem cabecalho onde declarar que foi revisto, e
# a mensagem dele e imutavel de proposito -- reescrever historia nao e opcao.
# Mas a `memoria_cientifica` indexa mensagens de commit como trecho, e uma
# mensagem refutada continuaria saindo da busca sem etiqueta. A relacao fica
# declarada AQUI, com o documento que a sustenta e o que exatamente caiu; o
# resto da mensagem nao e tocado pela declaracao.
COMMITS_REVISADOS = {
    "03ff3b0": {
        "revisado_por": "CIENCIA_DO_APRENDIZADO_2026_10_03.md",
        "o_que_caiu": ("'sem pre-treino o ajuste fino e loteria': as corridas usavam o peso da "
                       "ULTIMA epoca; com a epoca escolhida pela validacao (16 sementes, 03/10) o "
                       "braco aleatorio tem amplitude 0,068 contra 0,146 do pre-treino, e a H1 "
                       "(pre-treino ajuda?) e indecidivel"),
    },
}

# Relacoes declaradas NO CABECALHO do documento. So as primeiras linhas sao
# lidas: uma mencao no corpo -- "o cronograma v1 dizia X" -- e citacao, e nao
# declaracao de supersessao.
LINHAS_DE_CABECALHO = 25
PADRAO_RELACAO = re.compile(
    r"^\*{0,2}(substitui|revisa|supersede)\*{0,2}\s*:?\s*(.+)$",
    re.IGNORECASE)
PADRAO_ARQUIVO = re.compile(r"`([^`]+\.md)`|([A-Za-z0-9_\-.]+\.md)")
PADRAO_DATA_NO_NOME = re.compile(r"(20\d{2})[_-](\d{2})[_-](\d{2})")
# `**Data:** 31/08/2026` tem os dois-pontos DENTRO dos asteriscos; exigir a
# ordem `asteriscos, dois-pontos` faria o cabecalho mais comum do repositorio
# nao casar, e a data cairia silenciosamente para o nome do arquivo.
PADRAO_DATA_DECLARADA = re.compile(
    r"^[*_\s]*data[*_\s:]*(\d{2})/(\d{2})/(20\d{2})", re.IGNORECASE)

PADRAO_EXPERIMENTAL = re.compile(
    r"estado:\s*especificac|nada aqui esta servido|nada aqui está servido|"
    r"candidato, com o portao|candidato, com o portão", re.IGNORECASE)
PADRAO_DRAFT = re.compile(r"^\s*(draft|rascunho)\b", re.IGNORECASE)
# O terceiro estado auto-declaravel, e o unico que uma ESPECIFICACAO pode
# alcancar sozinha: ela foi implementada. `CONTRATO_V2_CANDIDATO_TEMPORAL`
# passou por isso em 12/09 -- o titulo dele continua dizendo "candidato",
# corretamente, porque era o que ele era quando foi escrito. Quem diz que
# deixou de ser e o cabecalho, e nao a data nem a posicao na serie.
PADRAO_SUPERADO = re.compile(
    r"estado:\s*\**superado|^\*\*superado por:\*\*",
    re.IGNORECASE | re.MULTILINE)


def _cabecalho(caminho: Path) -> list[str]:
    texto = caminho.read_text(encoding="utf-8", errors="replace")
    return texto.splitlines()[:LINHAS_DE_CABECALHO]


def data_do_documento(caminho: Path) -> tuple[str | None, str]:
    """A data e a PROCEDENCIA dela -- nunca a data sozinha.

    `mtime` seria a resposta facil e a errada: reindexar o repositorio mudaria
    a precedencia de todo mundo sem ninguem ter escrito nada.
    """
    for linha in _cabecalho(caminho):
        m = PADRAO_DATA_DECLARADA.search(linha.strip())
        if m:
            d, mes, ano = m.groups()
            return f"{ano}-{mes}-{d}", "declarada no cabecalho"
    m = PADRAO_DATA_NO_NOME.search(caminho.name)
    if m:
        return "-".join(m.groups()), "no nome do arquivo"
    return None, "nao declarada"


def relacoes_declaradas(caminho: Path) -> list[dict]:
    """`Substitui:` / `Revisa:` no cabecalho, com o alvo nomeado."""
    saida = []
    for linha in _cabecalho(caminho):
        m = PADRAO_RELACAO.match(linha.strip())
        if not m:
            continue
        verbo, resto = m.group(1).lower(), m.group(2)
        for cru, simples in PADRAO_ARQUIVO.findall(resto):
            alvo = (cru or simples).strip()
            if alvo and alvo != caminho.name:
                saida.append({"verbo": verbo, "alvo": Path(alvo).name,
                              "linha": linha.strip()[:200]})
    return saida


def _auto_declarado(caminho: Path) -> str | None:
    """A ordem importa: `superado` e checado ANTES de `especificacao`.

    Um documento implementado costuma manter no titulo a palavra que o
    descrevia quando foi escrito -- "candidato", "proposta". Checar
    especificacao primeiro devolveria `experimental` para sempre, e o indice
    seguiria dizendo que nada ali esta servido depois de servido.
    """
    cabecalho = "\n".join(_cabecalho(caminho))
    if PADRAO_SUPERADO.search(cabecalho):
        return SUPERADO
    if PADRAO_DRAFT.search(cabecalho):
        return DRAFT
    if PADRAO_EXPERIMENTAL.search(cabecalho):
        return EXPERIMENTAL
    return None


def construir(series: dict | None = None, raiz: Path = REPO) -> dict:
    series = series or SERIES
    docs: dict[str, dict] = {}
    for serie, caminhos in series.items():
        for rel in caminhos:
            p = raiz / rel
            if not p.exists():
                docs[Path(rel).name] = {
                    "caminho": rel, "serie": serie, "existe": False,
                    "estado": None,
                    "motivo": "declarado na serie e ausente do repositorio"}
                continue
            data, procedencia = data_do_documento(p)
            docs[p.name] = {
                "caminho": rel, "serie": serie, "existe": True,
                "data": data, "procedencia_da_data": procedencia,
                "sha256": sha256_arquivo(p),
                "auto_declarado": _auto_declarado(p),
                "declara": relacoes_declaradas(p),
                "superado_por": [],
            }

    # Supersessao declarada: quem diz que substitui/revisa outro.
    for nome, d in docs.items():
        for rel in d.get("declara", []):
            alvo = docs.get(rel["alvo"])
            if alvo is not None:
                alvo["superado_por"].append(
                    {"documento": nome, "verbo": rel["verbo"],
                     "declaracao": rel["linha"]})
            else:
                d.setdefault("declaracoes_sem_alvo_no_indice", []).append(rel)

    # Estado. Ordem deliberada: o que o documento declara sobre si mesmo vem
    # ANTES da posicao na serie -- uma especificacao nao vira "vigente" so por
    # ser a mais recente.
    por_serie: dict[str, list] = {}
    for nome, d in docs.items():
        por_serie.setdefault(d["serie"], []).append(nome)
    for serie, nomes in por_serie.items():
        # Um documento que se declara ESPECIFICACAO ou RASCUNHO nao sucede
        # ninguem. Sem esta linha, `CONTRATO_V2_CANDIDATO_TEMPORAL` -- que diz
        # na primeira linha que nada ali esta servido -- tornaria o contrato
        # SERVIDO historico, que e o oposto do que ele afirma.
        sucessores = [n for n in nomes
                      if docs[n].get("existe") and not docs[n].get("auto_declarado")]
        datados = [n for n in sucessores if docs[n].get("data")]
        so_um_ordenavel = len(sucessores) == 1
        # EMPATE DE DATA. `max()` escolheria um dos dois pela ordem do
        # dicionario, e a escolha ficaria invisivel. Dois documentos do mesmo
        # dia, sem ninguem declarar supersessao, sao os DOIS vigentes -- e a
        # resposta tem de citar os dois.
        ultima = max((docs[n]["data"] for n in datados), default=None)
        empatados = [n for n in datados if docs[n]["data"] == ultima]
        mais_recente = empatados[0] if len(empatados) == 1 else None
        for nome in nomes:
            d = docs[nome]
            if not d.get("existe"):
                continue
            if d["auto_declarado"]:
                d["estado"] = d["auto_declarado"]
                d["motivo"] = "o proprio documento se declara assim"
            elif d["superado_por"]:
                d["estado"] = SUPERADO
                d["motivo"] = (f"{d['superado_por'][0]['documento']} declara que "
                               f"o {d['superado_por'][0]['verbo']}")
            elif nome in empatados or (not datados and so_um_ordenavel):
                d["estado"] = VIGENTE
                d["motivo"] = (
                    "mais recente da serie sem supersessao declarada"
                    if len(empatados) == 1 else
                    f"empate de data com {len(empatados)} documentos da serie: "
                    "sem supersessao declarada, nenhum supera o outro"
                    if empatados else
                    "unico da serie, sem data e sem supersessao")
            elif not d.get("data") and not so_um_ordenavel:
                # Sem data nao da para ordenar, e o projeto trata isso como
                # impossibilidade de demonstrar vigencia -- nao como vigencia.
                d["estado"] = HISTORICO
                d["motivo"] = ("sem data declarada e a serie tem outros "
                               "documentos: nao ha como demonstrar vigencia")
            else:
                d["estado"] = HISTORICO
                d["motivo"] = (f"a serie tem documento mais recente "
                               f"({ultima}) e ninguem declarou supersessao")
            d["pode_sustentar_estado_atual"] = PODE_SUSTENTAR_ESTADO_ATUAL[d["estado"]]
            d["vigente_da_serie"] = sorted(empatados) or None
            if len(empatados) > 1:
                d["empate_de_data"] = sorted(empatados)

    return {
        "tipo": "precedencia_documental",
        "versao": "1",
        "gerado_em": agora_utc(),
        "estados": list(ESTADOS),
        "definicoes": DEFINICAO,
        "pode_sustentar_estado_atual": PODE_SUSTENTAR_ESTADO_ATUAL,
        "regra": ("relacao declarada no ponto de curadoria ou no cabecalho do "
                  "proprio documento; nunca inferida de similaridade"),
        "series": {s: sorted(n) for s, n in por_serie.items()},
        "documentos": docs,
        "nao_e": ("nao apaga documento, nao remove trecho do indice e nao "
                  "decide resposta: acrescenta o eixo do tempo a citacao"),
    }


def carregar(caminho: Path = INDICE) -> dict | None:
    if not caminho.exists():
        return None
    try:
        d = json.loads(caminho.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return d if d.get("tipo") == "precedencia_documental" else None


def estado_de(source_path: str, indice: dict | None = None) -> dict:
    """O estado do documento de onde um trecho veio.

    Documento fora do indice devolve `conhecido: False` -- e nao `vigente`.
    Tratar ausencia de curadoria como vigencia seria deixar qualquer arquivo
    novo falar pelo estado atual do projeto sem ninguem ter olhado.
    """
    indice = indice if indice is not None else (carregar() or {})
    m = re.search(r"commit:([0-9a-f]{7,40})", str(source_path))
    if m:
        h = m.group(1)
        rev = next((v for k, v in COMMITS_REVISADOS.items() if h.startswith(k) or k.startswith(h)), None)
        if rev:
            return {"conhecido": True, "documento": f"commit:{h}", "serie": "commits",
                    "estado": SUPERADO, "motivo": f"{rev['revisado_por']} revisa: {rev['o_que_caiu']}",
                    "pode_sustentar_estado_atual": False, "superado_por": [
                        {"documento": rev["revisado_por"], "verbo": "revisa"}]}
        return {"conhecido": False, "documento": f"commit:{h}", "estado": None,
                "pode_sustentar_estado_atual": False,
                "motivo": "mensagem de commit: registra o que se sabia naquele dia, nao o estado atual"}
    nome = Path(str(source_path)).name
    d = (indice.get("documentos") or {}).get(nome)
    if not d or not d.get("estado"):
        return {"conhecido": False, "documento": nome,
                "estado": None, "pode_sustentar_estado_atual": False,
                "motivo": ("documento fora do indice de precedencia; ausencia "
                           "de curadoria nao e vigencia")}
    return {
        "conhecido": True, "documento": nome, "serie": d["serie"],
        "estado": d["estado"], "motivo": d["motivo"], "data": d.get("data"),
        "procedencia_da_data": d.get("procedencia_da_data"),
        "pode_sustentar_estado_atual": d["pode_sustentar_estado_atual"],
        "vigente_da_serie": d.get("vigente_da_serie"),
        "superado_por": d.get("superado_por") or [],
    }


def anotar(resultados: list, indice: dict | None = None) -> list:
    """Anota trechos de `rag_busca.buscar` com o estado do documento.

    Nao filtra. Quem decide o que fazer com um trecho superado e quem responde
    -- e a resposta correta quase nunca e esconder: e dizer qual e o vigente e
    o que o anterior dizia.
    """
    indice = indice if indice is not None else (carregar() or {})
    anotados = []
    for r in resultados:
        d = dict(r)
        d["precedencia"] = estado_de(
            d.get("source_path") or d.get("source_id") or "", indice)
        anotados.append(d)
    return anotados


def resumo_para_resposta(anotados: list) -> dict:
    """O que a resposta precisa dizer sobre a epoca das suas fontes."""
    vigentes = [a for a in anotados
                if (a["precedencia"].get("estado") == VIGENTE)]
    nao_vigentes = [a for a in anotados
                    if a["precedencia"].get("estado") not in (VIGENTE, None)]
    desconhecidos = [a for a in anotados if not a["precedencia"]["conhecido"]]
    return {
        "trechos": len(anotados),
        "de_documento_vigente": len(vigentes),
        "de_documento_nao_vigente": len(nao_vigentes),
        "de_documento_fora_do_indice": len(desconhecidos),
        "documentos_vigentes_usados": sorted(
            {a["precedencia"]["documento"] for a in vigentes}),
        "documentos_superados_usados": sorted(
            {a["precedencia"]["documento"] for a in nao_vigentes}),
        "algum_claim_de_estado_atual_sem_documento_vigente":
            bool(nao_vigentes or desconhecidos) and not vigentes,
        "o_que_a_resposta_deve_dizer": (
            "citar o vigente e nomear o superado quando os dois forem usados; "
            "trecho de documento nao vigente nao sustenta afirmacao sobre o "
            "estado ATUAL do projeto"),
    }


def gravar(indice: dict, destino: Path = INDICE) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(indice, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    return destino


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--consultar", default=None,
                    help="caminho de um documento; imprime o estado dele")
    ns = ap.parse_args()

    indice = construir()
    if ns.consultar:
        print(json.dumps(estado_de(ns.consultar, indice), indent=2,
                         ensure_ascii=False))
        return 0

    destino = gravar(indice)
    print("===PRECEDENCIA DOCUMENTAL===")
    for serie in sorted(indice["series"]):
        print(f"\n{serie}")
        for nome in indice["series"][serie]:
            d = indice["documentos"][nome]
            if not d.get("existe"):
                print(f"  {nome:52s} AUSENTE  {d['motivo']}")
                continue
            print(f"  {nome:52s} {str(d['data']):10s} {d['estado']:12s} "
                  f"{d['motivo']}")
    contagem: dict[str, int] = {}
    for d in indice["documentos"].values():
        if d.get("estado"):
            contagem[d["estado"]] = contagem.get(d["estado"], 0) + 1
    print(f"\npor estado: {contagem}")
    print(f"GRAVADO={destino}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
