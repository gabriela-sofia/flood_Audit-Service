"""
SVC-02 -- O contrato de inferencia: portoes, escore com IC, model card, explicacao.

O QUE ESTE ARQUIVO E, e o que ele nao e:

e a implementacao, dentro do REV-P, do contrato que a Secao II do plano
descreve -- "a requisicao declara geometria, CRS, periodo e camadas; a resposta
devolve status, maturidade da regiao, escore com intervalo, variaveis usadas e
limitacoes, sempre junto de um model card". Ate agora esse contrato existia como
texto e como MVP no repositorio privado; aqui ele vira funcao pura, auditavel e
testavel, sem rede e sem servidor.

NAO e um servidor HTTP. A escolha e deliberada: o que precisa ser auditado e o
CONTRATO -- quais portoes existem, em que ordem, o que cada recusa significa e
como o escore e construido. Envolver isso em FastAPI e trabalho de transporte,
nao de metodo, e traria dependencia de rede para dentro de um repositorio que e
fail-closed por regra. `inferir()` e a funcao que um servidor chamaria.

-------------------------------------------------------------------------
OS PORTOES, na ordem em que sao avaliados
-------------------------------------------------------------------------

A ordem importa: cada portao so e avaliado se o anterior fechou, e a resposta
nomeia o PRIMEIRO que falhou. Recusar por "faltou HAND" quando a geometria nem
era valida esconderia o erro real de quem chamou.

  G1 geometria_valida        ha pelo menos um ponto, CRS declarado e suportado,
                             latitude e longitude finitas e dentro do globo.
                             Falha -> insufficient_data.
  G2 regiao_resolvida        a regiao vem declarada na requisicao ou e deduzida
                             pela caixa envolvente das regioes que existem na
                             tabela unica. Falha -> region_not_supported.
  G3 modelo_para_a_regiao    existe modelo servivel mapeado para a regiao.
                             Falha -> region_not_supported, nunca escore por
                             analogia.
  G4 variaveis_presentes     toda variavel do modelo esta presente e finita em
                             todo ponto. Falha -> insufficient_data.
  G5 dominio_coberto         quantas variaveis da requisicao caem dentro da
                             faixa 5-95% que o modelo viu no ajuste. Acima do
                             limite de extrapolacao -> insufficient_data; abaixo
                             dele, entra como limitacao declarada.

G5 existe porque `metodo_aplicacao_sem_rotulo_local_v1.md` mediu que 0% dos
pontos de Curitiba caiam na faixa de `elevation_m` do treino -- diferenca de
2,76 desvios. Aquilo era uma recomendacao num documento; aqui e um portao que o
servico avalia por requisicao.

-------------------------------------------------------------------------
G6 -- O PORTAO TEMPORAL, IMPLEMENTADO E ATIVO POR MATURIDADE
-------------------------------------------------------------------------

  G6 suporte_temporal        ha suporte temporal suficiente, segundo a
                             politica cientifica aplicavel a este par
                             regiao x modelo? Falha -> insufficient_data.

Ele NAO testa "AUC temporal acima de um limiar". A entrada e a avaliacao de
suporte da matriz cientifica (`scripts/temporalidade/dimensao_temporal.py`), e
o que ele le e o ESTADO -- `supported`, `not_supported`, `inconclusive`,
`not_testable` -- com a razao preservada em cada caso.

`POLITICA_PORTAO_TEMPORAL = "por_maturidade"` desde 12/09/2026, pela ADR-001.
A constante nomeia a politica em vez de ligar ou desligar o portao, e a razao
e que "ligado" nunca disse COMO: das tres saidas medidas em sombra sobre as
mesmas requisicoes, `portao` recusa sempre -- e com a evidencia disponivel
nenhuma das tres regioes tem suporte `supported` no proprio escopo, entao o
produto inteiro pararia de emitir -- e `exibir` e o comportamento da v1.

`por_maturidade` recusa onde a recusa CONTRADIZ o que o produto ja afirma:

  `validado` / `mvp_local`   o produto diz que a regiao tem modelo proprio
                             ajustado nela; emitir sem suporte temporal e
                             afirmar validade que ninguem mediu. RECUSA.
  `transferencia_*`          o payload ja declara que o modelo nao viu a
                             regiao. A recusa nao acrescentaria cautela: so
                             apagaria o escore. O suporte viaja como
                             LIMITACAO DECLARADA.

Efeito medido: Recife passa a recusar; Curitiba e Petropolis seguem emitindo.
A demonstracao deixa de ser "o produto responde" e passa a ser "responde onde
pode e recusa onde afirmaria o que nao mediu".

Isso e mudanca de comportamento de recusa, que `docs/CONTRATO_VERSIONAMENTO.md`
classifica como `v2` -- e e por isso que a versao subiu no mesmo passo. A
analise que exigiu a versao esta em
`local_runs/temporalidade-integracao/contract_impact.json` (`V2_REQUIRED`), e a
decisao em `docs/ADR_001_PORTAO_TEMPORAL_G6.md`.

O campo `suporte_cientifico` continua existindo e continua descritivo onde a
politica nao decide: `e_portao_do_contrato` responde por ESCOPO, e nao pela
politica em geral -- em Petropolis ele e `false` e em Recife e `true`, com a
mesma evidencia por tras.

-------------------------------------------------------------------------
MATURIDADE DA REGIAO -- o eixo de predicao seletiva
-------------------------------------------------------------------------

  validado                    modelo proprio, sinais fisicos corretos, IC das
                              causais sem cruzar zero e AUC agrupada dentro da
                              faixa declarada.
  mvp_local                   modelo proprio da regiao, mas algum criterio de
                              leitura nao atingido. O escore sai com a falha
                              declarada -- ver POLITICA_CRITERIO_NAO_ATINGIDO.
  transferencia_caracterizada a regiao nao entrou no treino em nenhuma linha; o
                              que se pode declarar e a distancia de dominio, nao
                              acerto contra rotulo local.
  nao_suportada               sem modelo: region_not_supported.

POLITICA_CRITERIO_NAO_ATINGIDO decide o que fazer quando o modelo da regiao
existe mas nao atinge o criterio de leitura fixado antes (por exemplo, AUC
abaixo da faixa 0,70-0,88). Duas posturas defensaveis:

  "declara"  devolve o escore com a falha escrita em `limitacoes` e maturidade
             rebaixada a mvp_local. Coerente com "resultado negativo e
             publicado" -- esconder o escore esconderia tambem o quanto ele e
             fraco.
  "recusa"   devolve insufficient_data. Coerente com fail-closed estrito.

O padrao e "declara", e a escolha esta AQUI, exposta, porque ela muda o que o
produto responde para Recife: com a fonte de chuva corrigida o LOO-AUC de
Recife e 0,6339, abaixo da faixa. Trocar esta constante e uma decisao da
Gabriela, nao um detalhe de implementacao.

-------------------------------------------------------------------------
O ESCORE, o intervalo e a unidade de resposta
-------------------------------------------------------------------------

A unidade que o contrato devolve e a AREA, nunca o pixel -- "validar no mesmo
grao que se entrega e o que torna o resultado auditavel". O escore da AOI e a
media dos escores dos seus pontos.

O IC vem das replicas de bootstrap gravadas pelo SVC-01: cada replica de
coeficiente projeta um escore para cada ponto, a media da AOI e recalculada por
replica, e o intervalo e o percentil 2,5-97,5 dessas medias. Nao ha reajuste em
tempo de requisicao.

-------------------------------------------------------------------------
A CAMADA DE EXPLICACAO
-------------------------------------------------------------------------

Gera por REGRAS, lendo o payload ja decidido. O plano declarou essa rota como
alternativa caso a explicacao divergisse do payload; adota-la desde o inicio
torna a divergencia impossivel por construcao, e nao por verificacao posterior.
A contribuicao de cada variavel e `coeficiente_padronizado x z(valor)`, medida
no mesmo espaco em que o modelo decide, e a frase so descreve o sinal dessa
contribuicao. Nao ha texto livre nem modelo de linguagem envolvido.

Uso:
    python scripts/servico/svc02_contrato_inferencia.py --demonstracao
    python scripts/servico/svc02_contrato_inferencia.py --requisicao arquivo.json
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))
sys.path.insert(0, str(REPO / "scripts"))
from chuva.contrato_chuva import (  # noqa: E402
    contract as rain_contract, scenario_evidence,
)
from ds03_esquema_alvo import VERSAO  # noqa: E402

import politica_de_portao_temporal as politica_temporal  # noqa: E402
import suporte_cientifico as suporte  # noqa: E402
from cadeia_no_contrato import bloco_da_cadeia  # noqa: E402

RUNS = REPO / "local_runs"
MODELOS = RUNS / "svc-01-modelos"
UNICA = RUNS / "ds-05-tabela-unica" / f"tabela_unica_{VERSAO}.parquet"
OUT = RUNS / "svc-02-contrato"
MATRIZ_CIENTIFICA = (RUNS / "temporalidade-integracao"
                     / suporte.NOME_DA_MATRIZ)

CRS_SUPORTADOS = ("EPSG:4326",)
VERSAO_CONTRATO = "v3"
POLITICA_CRITERIO_NAO_ATINGIDO = "declara"   # ou "recusa" -- ver cabecalho
MAX_PCT_EXTRAPOLACAO = 50.0                  # acima disto, G5 recusa

# A POLITICA de G6, adotada em 12/09/2026 pela ADR-001. Ela deixou de ser um
# booleano porque "ligado" nao dizia COMO: as tres saidas admissiveis estao em
# `politica_de_portao_temporal`, foram medidas em sombra sobre as mesmas
# requisicoes, e `por_maturidade` e a unica em que a consequencia da recusa
# depende do que o produto esta afirmando.
#
# Efeito medido, e ele e o argumento: Recife (`mvp_local`) passa a RECUSAR --
# dizer que a regiao tem modelo proprio ajustado nela, sem suporte temporal, e
# afirmar validade que ninguem demonstrou. Curitiba e Petropolis seguem
# emitindo, porque `transferencia_*` ja declara que o modelo nao viu a regiao,
# e ali a recusa nao acrescentaria cautela: so apagaria o escore.
POLITICA_PORTAO_TEMPORAL = "por_maturidade"

# Modos temporais opt-in. O padrao continua sendo o retrato estrutural da v2;
# declarar uma data nao transforma uma grade estatica em retrato daquele dia.
TIPOS_DE_RESPOSTA_TEMPORAL = (
    "retrato_estrutural",
    "retrospectiva_observada",
    "cenario_futuro",
    "pesquisa_reprodutivel",
)

STATUS_OK = "ok"
STATUS_SEM_DADO = "insufficient_data"
STATUS_SEM_REGIAO = "region_not_supported"

# A ESCADA PUBLICA DA v2, na ordem em que e avaliada. Ela esta aqui, nomeada,
# porque a ordem e os nomes fazem parte do que a versao promete: a resposta
# nomeia o PRIMEIRO portao que nao fechou, entao acrescentar um degrau muda o
# significado de toda recusa -- e foi por isso que acrescentar
# `suporte_temporal` custou uma versao. Na v1 ele viajava so como campo
# descritivo; a trava que exigia sua ausencia daqui esta em
# `test_contrato_retrato.py`, reescrita junto com a versao.
PORTOES_PUBLICOS = (
    "geometria_valida", "regiao_resolvida", "modelo_para_a_regiao",
    "variaveis_presentes", "dominio_coberto", "suporte_temporal",
)

# O vocabulario oficial de maturidade, e o que cada nivel AUTORIZA afirmar.
# Fica aqui, num lugar so, porque o projeto ja teve dois vocabularios em
# circulacao: o do `susc_20e` (`available`/`limited_evidence`/`insufficient`) e
# este. Quem consome a resposta precisa de um.
AUTORIZA_POR_MATURIDADE = {
    "validado": ("escore com desempenho verificado contra rotulo local; "
                 "nenhuma regiao do projeto esta neste nivel"),
    "mvp_local": ("escore de modelo proprio da regiao, carregando na resposta "
                  "o criterio de leitura que o ajuste nao atingiu"),
    "transferencia_caracterizada": (
        "escore de modelo que nao viu a regiao no treino, mas a regiao tem "
        "inventario local com que verificar depois; a distancia de dominio e "
        "declarada, o acerto nao"),
    "transferencia_sem_referencia_local": (
        "escore por semelhanca de terreno, nunca afirmacao de acerto: nao ha "
        "nenhum ponto rotulado na regiao com que verificar, nem hoje nem "
        "retroativamente"),
    "nao_suportada": "nenhum escore; a requisicao e recusada antes de calcular",
}

NOME_LEGIVEL = {
    "hand_m": "altura acima da drenagem (HAND)",
    "twi_dinf": "indice topografico de umidade (TWI)",
    "elevation_m": "elevacao",
    "slope_deg": "declividade",
    "rain_max_24h": "pico de chuva em 24 h",
    "rain_decay_index": "chuva antecedente com decaimento",
}
UNIDADE = {"hand_m": "m", "twi_dinf": "", "elevation_m": "m",
           "slope_deg": "graus", "rain_max_24h": "mm", "rain_decay_index": "mm"}


# ------------------------------------------------------------ carregamento

def carregar_modelos(diretorio: Path = MODELOS) -> dict:
    """Os modelos SERVIDOS -- os que o indice nomeia, e nao os que o disco tem.

    -------------------------------------------------------------------------
    POR QUE O INDICE MANDA, E NAO O GLOB
    -------------------------------------------------------------------------

    Ate 17/09/2026 esta funcao carregava todo `*.json` do diretorio, e isso
    parecia inofensivo porque todo json ali tinha sido escrito pelo `svc01`.
    Deixou de ser verdade quando scripts de medida passaram a gravar artefato
    no MESMO diretorio (`rs02` grava `fluvial_areal_observado`,
    `sp01` grava `pluvial_urbano_observado`). Eles nao sao catalogo; sao
    medidas, e ficam ao lado por conveniencia.

    O efeito medido foi concreto: `fluvial_areal_observado`, congelado em
    14/09 sobre `fonte == "cems"`, continuava declarando servir o Rio Grande
    do Sul. Em `regioes()` a iteracao e alfabetica e a ultima entrada vence,
    entao ele SOBRESCREVIA o `areal_brasil_observado` e o RS voltava de
    `validado` para `transferencia_caracterizada` -- sem erro, sem aviso, e
    com o numero errado saindo pela API.

    `indice.json` existe exatamente para dizer o que e servido; ele e escrito
    pelo `svc01` no fim da construcao, com os modelos daquela execucao. Um
    json solto no diretorio deixa de virar modelo servido por acidente de
    nome de arquivo.

    Sem indice no disco a funcao NAO cai no glob: devolve vazio, e o contrato
    recusa tudo em G3. Cair no glob seria servir o que sobrou de execucoes
    antigas justamente quando o catalogo se perdeu.
    """
    if not diretorio.exists():
        return {}
    indice = diretorio / "indice.json"
    if not indice.exists():
        return {}
    nomes = json.loads(indice.read_text(encoding="utf-8")).get("modelos", {})
    modelos = {}
    for nome in nomes:
        p = diretorio / f"{nome}.json"
        if not p.exists():
            continue
        modelos[nome] = json.loads(p.read_text(encoding="utf-8"))
    return modelos


def caixas_das_regioes(caminho: Path = UNICA) -> dict:
    """Delegado a `regioes.py`, que resolve por LUGAR e nao por nome de fonte.

    A versao anterior iterava uma lista fixa de fontes e derivava a caixa de
    `d.fonte == fonte`. Isso nao dava caixa ao Rio Grande do Sul, que e um
    recorte de `cems`, e por isso o G2 recusava a regiao inteira mesmo com
    10.792 pontos rotulados dentro dela.

    A lista fixa some: a fonte da verdade e `regioes.SEMENTES`, e acrescentar
    regiao nova e acrescentar uma entrada la -- nunca em quatro lugares.
    """
    import regioes as _reg
    return {r: {k: v for k, v in cx.items()
                if k in ("lat_min", "lat_max", "lon_min", "lon_max")}
            for r, cx in _reg.caixas().items()}


def regioes_com_referencia_local(caminho: Path = UNICA) -> set:
    """Delegado a `regioes.py`: ha ponto rotulado DENTRO da regiao?

    A versao anterior devolvia `set(d.fonte.unique())` -- nomes de fonte. O
    Rio Grande do Sul nao aparecia ali, e o contrato concluia que a regiao com
    o melhor dado observado do projeto nao tinha referencia local nenhuma.

    Nao e o mesmo que "regiao validada": Curitiba tem rotulo e o projeto
    decidiu nao usa-lo como criterio de aprovacao. E o que separa "da para
    verificar depois" de "nao ha nem o que verificar".
    """
    import regioes as _reg
    return {r for r in _reg.SEMENTES if _reg.tem_referencia_local(r)}


def carregar_matriz_cientifica(caminho: Path = MATRIZ_CIENTIFICA):
    """A matriz de sustentacao cientifica, ou None se ela nao foi gerada.

    None NAO e `not_testable`: `not_testable` e conclusao derivada de
    artefato, e a falta da matriz e a ausencia da propria conclusao. Quem
    consome recebe `ausencia_declarada`, que diz exatamente isso.
    """
    return suporte.MatrizDeSuporte.carregar(caminho)


def suporte_cientifico_do_escopo(regiao: str, modelo: dict,
                                 matriz=None, maturidade: str | None = None) -> dict:
    """O bloco `suporte_cientifico` da resposta, para um par regiao x modelo.

    A consulta e pelo PAR. Curitiba e servida por `fluvial_planicie`, que tem
    holdout prospectivo no proprio pool de ajuste -- e nao em Curitiba.
    Perguntar so pela regiao devolveria a evidencia do modelo local de
    Curitiba (que nunca foi servido) ou a do pool; as duas seriam suporte
    herdado de escopo que ninguem mediu.
    """
    nome = modelo.get("nome") if isinstance(modelo, dict) else str(modelo)
    aval = matriz.consultar(regiao, nome) if matriz is not None else None
    if aval is None:
        bloco = suporte.ausencia_declarada(
            regiao, nome, suporte.TEMPORAL,
            motivo=("a matriz de sustentacao cientifica nao existe ou nao "
                    "cobre este par regiao x modelo; gere com "
                    "python scripts/temporalidade/dimensao_temporal.py"))
    else:
        bloco = dict(aval)
        bloco["avaliado"] = True
    # `e_portao_do_contrato` responde sobre ESTE escopo, e nao sobre a
    # politica em geral. Sob `por_maturidade` a mesma evidencia decide em
    # Recife e nao decide em Petropolis; um campo que dissesse `true` para os
    # dois faria o leitor de Petropolis achar que o escore dele passou por um
    # portao que nunca foi aplicado ali.
    decide_aqui = politica_temporal.decidir(
        POLITICA_PORTAO_TEMPORAL, maturidade, bloco.get("support"))
    bloco["e_portao_do_contrato"] = bool(
        POLITICA_PORTAO_TEMPORAL != politica_temporal.EXIBIR
        and str(maturidade) in politica_temporal.AFIRMA_VALIDADE_LOCAL)
    bloco["politica_do_portao"] = POLITICA_PORTAO_TEMPORAL
    # O dicionario de `decidir` NAO entra cru aqui: ele carrega a chave
    # `gate_que_nao_fechou`, que na resposta significa "esta requisicao foi
    # recusada por este portao". Dentro de um bloco descritivo ela apareceria
    # em respostas de SUCESSO e na listagem de regioes -- e quem procurar esse
    # nome para achar recusas encontraria lugares onde nao houve nenhuma.
    bloco["a_politica_decide_neste_escopo"] = bool(decide_aqui["recusa"])
    bloco["motivo_da_politica"] = decide_aqui["motivo"]
    bloco["o_que_este_campo_e"] = (
        "dimensao cientifica avaliada sempre; sob a politica "
        f"{POLITICA_PORTAO_TEMPORAL!r} ela DECIDE onde a maturidade afirma "
        "validade local e viaja como limitacao declarada onde o produto ja "
        f"declara transferencia (contrato {VERSAO_CONTRATO})")
    return {"temporal": bloco,
            "dimensoes_sem_protocolo_formal": [
                d for d, v in suporte.DIMENSOES.items()
                if not v["tem_protocolo_formal"]]}


def g6_suporte_temporal(regiao: str, modelo: dict, matriz=None,
                        maturidade: str | None = None) -> tuple:
    """O portao temporal, sob a politica adotada -- nao mais um booleano.

    Ele nao pergunta "a AUC temporal passou do limiar?". Ele pergunta se o
    ESTADO de suporte deste escopo autoriza prosseguir, e quem le o estado e
    `suporte_cientifico.suporte_demonstrado` -- um lugar so, para que o
    reajuste do SVC-07 e o contrato nao usem definicoes diferentes.

    O QUE MUDOU NA v2: quem transforma estado em decisao passou a ser
    `politica_de_portao_temporal.decidir`, e nao este arquivo. A separacao e o
    ponto: trocar de politica nao toca em quem avalia, e a avaliacao nao muda
    de valor porque a politica mudou. A cadeia continua

        evidencia -> avaliacao -> suporte -> POLITICA -> decisao

    e este `if` e o ultimo elo, so.

    Os cinco estados nao sao colapsados: `inconclusive`, `untested` e
    `not_testable` levam a mesma consequencia e o `detalhe` continua dizendo
    qual dos tres foi, porque os tres nao pedem o mesmo trabalho para mudar.

    A RECUSA DIZ O QUE FALTA (22/09). Ate aqui ela nomeava o estado e a razao
    tecnica e parava. Quem lesse "estado 'not_testable'" nao tinha como saber
    se aquilo pedia uma linha de protocolo ou uma campanha de aquisicao -- e
    as duas coisas ficaram na mesma fila por semanas. A acao viaja junto,
    derivada da razao, nunca escrita aqui.
    """
    bloco = suporte_cientifico_do_escopo(
        regiao, modelo, matriz, maturidade)["temporal"]
    estado = bloco.get("support")
    decisao = politica_temporal.decidir(
        POLITICA_PORTAO_TEMPORAL, maturidade, estado)
    if not decisao["recusa"]:
        return True, None, bloco
    razao = bloco.get("razao_tecnica") or bloco.get("motivo") or "sem avaliacao"
    acao = bloco.get("acao_para_sair") or {}
    sufixo = ""
    if acao.get("acao"):
        sufixo = (f"; para sair deste estado: {acao['acao']} "
                  f"(custo: {acao.get('classe_de_custo')})")
    return False, _falha(
        "suporte_temporal", STATUS_SEM_DADO,
        f"suporte temporal nao demonstrado para {regiao!r} com o modelo "
        f"{modelo.get('nome')!r}: estado {estado!r} ({razao}); a maturidade "
        f"{maturidade!r} afirma validade local, e emitir escore aqui seria "
        f"afirmar validade que ninguem mediu{sufixo}",
        {"regiao": regiao, "maturidade": maturidade,
         "politica_do_portao": POLITICA_PORTAO_TEMPORAL,
         "acao_para_sair": acao or None,
         "suporte_cientifico": {"temporal": bloco}}), bloco


def contexto_temporal_da_requisicao(requisicao: dict) -> tuple[bool, dict]:
    """Interpreta a pergunta temporal sem inventar validade para uma data."""
    tipo = requisicao.get("tipo_de_resposta") or "retrato_estrutural"
    if tipo not in TIPOS_DE_RESPOSTA_TEMPORAL:
        return False, {"motivo": f"tipo_de_resposta desconhecido: {tipo!r}"}

    periodo = requisicao.get("periodo") or {}
    data_alvo = requisicao.get("data_alvo") or periodo.get("fim")
    origem_data = ("data_alvo" if requisicao.get("data_alvo") else
                   "periodo.fim_legado" if periodo.get("fim") else None)
    corte = requisicao.get("data_corte_conhecimento")

    datas = {}
    for nome, valor in (("data_alvo", data_alvo),
                        ("data_corte_conhecimento", corte)):
        if valor is None:
            continue
        try:
            datas[nome] = date.fromisoformat(str(valor))
        except ValueError:
            return False, {"motivo": f"{nome} deve usar ISO 8601 YYYY-MM-DD"}

    base = {
        "tipo_de_resposta": tipo,
        "data_alvo": data_alvo,
        "origem_da_data_alvo": origem_data,
        "data_corte_conhecimento": corte,
        "especifico_da_data_alvo": False,
        "estado": "declarado_nao_usado",
    }

    if tipo == "retrato_estrutural":
        base.update({
            "estado": "nao_aplicavel" if data_alvo is None else "declarado_nao_usado",
            "motivo": ("o retrato usa camadas estruturais e/ou cenarios; uma "
                       "data declarada nao torna a grade especifica daquele dia"),
            "leitura_autorizada": "suscetibilidade fisica estrutural",
            "nao_e": "retrospectiva observada nem previsao para uma data",
        })
        return True, base

    if data_alvo is None:
        return False, {**base, "estado": "nao_identificavel",
                       "motivo": f"{tipo} exige data_alvo"}

    if tipo == "pesquisa_reprodutivel":
        return False, {
            **base, "estado": "nao_identificavel",
            "motivo": ("NÃO IDENTIFICÁVEL COM OS DADOS ATUAIS: o projeto nao "
                       "possui snapshot versionado de modelo, dados e fontes "
                       "por data de corte do conhecimento"),
        }

    if corte and datas["data_corte_conhecimento"] > datas["data_alvo"]:
        return False, {**base, "estado": "corte_temporal_invalido",
                       "motivo": ("data_corte_conhecimento posterior a "
                                  "data_alvo permitiria informacao futura")}

    if tipo == "retrospectiva_observada":
        evidencia = requisicao.get("evidencia_temporal_entrada") or {}
        if datas["data_alvo"] > date.today():
            return False, {**base, "estado": "data_futura",
                           "motivo": ("retrospectiva_observada nao aceita "
                                      "data futura")}
        if not evidencia.get("executado") or not evidencia.get("janela_chuva"):
            return False, {
                **base, "estado": "nao_identificavel",
                "motivo": ("NÃO IDENTIFICÁVEL COM OS DADOS ATUAIS: a data foi "
                           "declarada, mas nao ha aquisicao observada e "
                           "rastreavel das condicoes temporais daquela data"),
            }
        base.update({
            "estado": "condicionado_a_entrada_observada",
            "especifico_da_data_alvo": True,
            "janela_temporal_usada": evidencia["janela_chuva"],
            "fonte_temporal": evidencia.get("fonte"),
            "leitura_autorizada": ("suscetibilidade condicionada as entradas "
                                   "observadas declaradas para a data"),
            "nao_e": ("validacao local naquela data nem probabilidade "
                       "observada de alagamento"),
        })
        return True, base

    if not requisicao.get("cenario_de_chuva"):
        return False, {**base, "estado": "cenario_ausente",
                       "motivo": ("cenario_futuro exige cenario_de_chuva; o "
                                  "contrato nao escolhe forcing em silencio")}
    base.update({
        "estado": "cenario_declarado",
        "cenario_de_chuva": requisicao.get("cenario_de_chuva"),
        "leitura_autorizada": "cenario condicional para a data declarada",
        "nao_e": "previsao meteorologica nem observacao da data-alvo",
    })
    return True, base


def modelo_da_regiao(regiao: str, modelos: dict) -> dict | None:
    for m in modelos.values():
        if regiao in m.get("regioes_servidas", []) and m.get("servivel"):
            return m
    return None


# ------------------------------------------------------------------ portoes

def _falha(gate: str, status: str, detalhe: str, extra: dict | None = None) -> dict:
    r = {"status": status, "gate_que_nao_fechou": gate, "detalhe": detalhe,
         "escore": None, "contrato": VERSAO_CONTRATO}
    if extra:
        r.update(extra)
    return r


def g1_geometria(req: dict) -> tuple[bool, dict | None, list]:
    geo = req.get("geometria") or {}
    crs = geo.get("crs")
    if crs not in CRS_SUPORTADOS:
        return False, _falha("geometria_valida", STATUS_SEM_DADO,
                             f"CRS {crs!r} nao suportado; suportados: "
                             f"{list(CRS_SUPORTADOS)}"), []
    pontos = geo.get("pontos") or []
    if not pontos:
        return False, _falha("geometria_valida", STATUS_SEM_DADO,
                             "nenhum ponto na geometria"), []
    for i, p in enumerate(pontos):
        try:
            lat, lon = float(p["lat"]), float(p["lon"])
        except (KeyError, TypeError, ValueError):
            return False, _falha("geometria_valida", STATUS_SEM_DADO,
                                 f"ponto {i} sem lat/lon numericos"), []
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return False, _falha("geometria_valida", STATUS_SEM_DADO,
                                 f"ponto {i} com coordenada nao finita"), []
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return False, _falha("geometria_valida", STATUS_SEM_DADO,
                                 f"ponto {i} fora do globo"), []
    return True, None, pontos


def g2_regiao(req: dict, pontos: list, caixas: dict) -> tuple[bool, dict | None, str]:
    """A regiao declarada, ou a MAIS ESPECIFICA que contem a geometria inteira.

    -------------------------------------------------------------------------
    POR QUE "MAIS ESPECIFICA" E NAO "A PRIMEIRA QUE CASA"
    -------------------------------------------------------------------------

    Ate 16/09/2026 esta funcao devolvia a primeira caixa que continha os
    pontos, e isso bastava porque nenhuma das seis regioes se sobrepunha --
    `test_as_extensoes_nao_se_sobrepoem` guardava exatamente essa premissa.

    As cinco macrorregioes quebram a premissa por construcao, e nao por
    descuido: Curitiba fica DENTRO do Sul, Recife dentro do Nordeste, Sao Paulo
    dentro do Sudeste. Aninhamento aqui e o desenho certo -- o produto passa a
    responder no Brasil todo, e continua tendo lugares com inventario proprio
    dentro dele.

    Com caixas aninhadas, "a primeira que casa" vira sorteio decidido pela
    ordem de insercao de um dicionario: a mesma requisicao em Curitiba poderia
    ser servida pelo modelo de Curitiba ou pelo areal brasileiro conforme a
    ordem em que as sementes foram escritas. Nenhuma das duas respostas e
    errada isoladamente; o que e inaceitavel e nao saber qual sai.

    A menor caixa que contem a geometria e a regra: ela devolve o lugar com
    inventario local quando ele existe, e a macrorregiao quando nao existe.
    """
    declarada = (req.get("regiao") or "").strip().lower()
    _reg = _modulo_regioes()
    if declarada:
        if _e_macrorregiao(_reg, declarada):
            ok, motivo = territorio_da_macrorregiao(declarada, pontos, _reg)
            if not ok:
                return False, _falha("regiao_resolvida", STATUS_SEM_REGIAO, motivo), ""
        return True, None, declarada
    candidatas = [
        (nome, c) for nome, c in caixas.items()
        if all(c["lat_min"] <= float(p["lat"]) <= c["lat_max"]
               and c["lon_min"] <= float(p["lon"]) <= c["lon_max"]
               for p in pontos)]
    # A caixa e o filtro barato; para MACRORREGIAO o poligono do IBGE decide (v3).
    # Ate a v2 havia dois atalhos: se o poligono excluia todas as candidatas, a
    # lista voltava para as caixas (um ponto no Paraguai saia como `sul`, modelo
    # brasileiro, maturidade `validado`); e qualquer erro na malha era engolido.
    # Agora macrorregiao sem territorio verificado sai da lista, e nao volta.
    excluidas = []
    refinadas = []
    for nome, c in candidatas:
        if _e_macrorregiao(_reg, nome) or (_reg is None and nome in MACRORREGIOES_DO_CONTRATO):
            ok, motivo = territorio_da_macrorregiao(nome, pontos, _reg)
            if not ok:
                excluidas.append(motivo)
                continue
        refinadas.append((nome, c))
    candidatas = refinadas
    if not candidatas and excluidas:
        return False, _falha("regiao_resolvida", STATUS_SEM_REGIAO,
                             "geometria cai na caixa de macrorregiao mas nao no territorio dela: "
                             + "; ".join(excluidas)), ""
    if candidatas:
        def area(c):
            return ((c["lat_max"] - c["lat_min"])
                    * (c["lon_max"] - c["lon_min"]))
        # O nome desempata: duas caixas de area identica tem de resolver para o
        # mesmo lugar em toda execucao, e nao pela ordem do dicionario.
        nome, _ = min(candidatas, key=lambda kv: (area(kv[1]), kv[0]))
        return True, None, nome
    return False, _falha("regiao_resolvida", STATUS_SEM_REGIAO,
                         "geometria nao cai em nenhuma regiao com modelo, e a "
                         "requisicao nao declarou `regiao`"), ""


MACRORREGIOES_DO_CONTRATO = {"norte", "nordeste", "centro_oeste", "sudeste", "sul"}


def _modulo_regioes():
    try:
        import regioes
        return regioes
    except Exception:  # noqa: BLE001 -- sem o modulo, macrorregiao fica nao verificavel (e recusada), nao aceita
        return None


def _e_macrorregiao(reg, nome: str) -> bool:
    if reg is None:
        return nome in MACRORREGIOES_DO_CONTRATO
    cfg = reg.SEMENTES.get(nome)
    return bool(cfg and cfg.get("tipo") == "macrorregiao")


def territorio_da_macrorregiao(nome: str, pontos: list, reg=None, geometria=None) -> tuple[bool, str]:
    """TODOS os pontos (ou a geometria inteira) dentro do poligono IBGE da macrorregiao; borda conta como dentro.

    Sem malha no disco ou sem o modulo de regioes a verificacao nao existe, e entao a resposta e NAO: o modelo
    areal brasileiro nao responde onde o territorio nao foi conferido (v3; sem fallback para a caixa).
    """
    reg = reg if reg is not None else _modulo_regioes()
    poly = reg._poligono(nome) if reg is not None else None
    if poly is None:
        return False, f"territorio de {nome!r} nao verificavel: malha IBGE ausente ou ilegivel (sem fallback para caixa)"
    from shapely.geometry import Point
    if geometria is not None and not poly.covers(geometria):
        return False, f"geometria fora (ou parcialmente fora) do territorio de {nome!r}"
    fora = [i for i, p in enumerate(pontos) if not poly.covers(Point(float(p["lon"]), float(p["lat"])))]
    if fora:
        return False, f"{len(fora)} de {len(pontos)} pontos fora do territorio de {nome!r}"
    return True, ""


def g3_modelo(regiao: str, modelos: dict) -> tuple[bool, dict | None, dict]:
    m = modelo_da_regiao(regiao, modelos)
    if m is None:
        return False, _falha("modelo_para_a_regiao", STATUS_SEM_REGIAO,
                             f"sem modelo ajustado e validado para {regiao!r}; "
                             "nenhum escore por analogia",
                             {"regiao": regiao,
                              "maturidade": "nao_suportada"}), {}
    return True, None, m


def _tipo_do_escore(modelo: dict) -> str:
    """O rotulo que a amostra do ajuste sustenta -- servido como `tipo_se_v2`.

    Chamar de "probabilidade projetada" um escore ajustado contra FUNDO promete
    uma escala que o dado nao tem: ali a prevalencia nao e identificavel e so a
    ordem sobrevive. `ordinal_relativo` diz o que o numero e.

    Na v1 este valor NAO substitui `tipo`. Ele viaja ao lado, porque trocar o
    rotulo muda o que um cliente ja escrito le -- e o retrato do contrato
    guarda justamente isso. A troca e uma decisao de versao, nao um efeito
    colateral de uma auditoria.
    """
    leitura = (modelo.get("amostra_negativa") or {}).get("leitura_admitida")
    if leitura == "ordinal":  # noqa: RET505
        return "ordinal_relativo"
    if leitura == "absoluta_condicionada":
        return "probabilidade_projetada_condicionada"
    return "probabilidade_projetada"


DERIVADAS_RELATIVAS = {"elevation_rel_m": "elevation_m"}


def derivar_relativas(pontos: list, modelo: dict, regiao: str) -> tuple:
    """Reconstroi as variaveis relativas que o modelo exige e o DEM nao da.

    O geoprocessamento sob demanda entrega `elevation_m`, que e o que se le de
    um raster. O vetor servido usa `elevation_rel_m`, que e a mesma elevacao
    contra o P1 da propria regiao -- e essa constante e propriedade da REGIAO,
    nao da requisicao: tira-la da AOI daria um numero diferente a cada recorte
    e o escore deixaria de ser reproduzivel.

    Por isso o baseline vem do artefato servido, gravado pelo `SVC-01` com a
    mesma definicao do `ds04`. Regiao sem baseline nao tem a variavel derivada
    e cai no G4 como variavel ausente -- recusa declarada, e nunca zero, que
    seria tratar o nivel do mar como piso local: exatamente o erro que a troca
    de variavel corrigiu.

    Ponto que ja trouxe a relativa pronta fica intocado: quem tem o valor de
    origem manda sobre a reconstrucao.
    """
    exigidas = [f for f in modelo.get("features", []) if f in DERIVADAS_RELATIVAS]
    if not exigidas:
        return pontos, {}
    baselines = modelo.get("elevacao_baseline_por_regiao") or {}
    base = baselines.get(regiao)
    derivadas = {}
    for f in exigidas:
        origem = DERIVADAS_RELATIVAS[f]
        if base is None:
            derivadas[f] = {"derivada": False,
                            "motivo": f"sem baseline de elevacao para {regiao!r}"}
            continue
        n = 0
        for pt in pontos:
            camadas = pt.get("camadas", pt)
            if camadas.get(f) is not None:
                continue
            v = camadas.get(origem)
            if v is None:
                continue
            camadas[f] = float(v) - float(base)
            n += 1
        derivadas[f] = {"derivada": True, "a_partir_de": origem,
                        "baseline_m": base, "pontos_derivados": n,
                        "definicao": "elevation_m menos o P1 da propria regiao"}
    return pontos, derivadas


def g4_variaveis(pontos: list, modelo: dict,
                 sem_cobertura: str = "recusar"
                 ) -> tuple[bool, dict | None, np.ndarray, dict]:
    """As variaveis exigidas, ponto a ponto.

    `sem_cobertura` decide o que fazer com ponto incompleto, e o padrao
    continua sendo `recusar` -- o comportamento de sempre.

    POR QUE `descartar` EXISTE:

    A resposta do contrato e por AREA, e a area vira uma malha de pontos. Com
    `recusar`, um unico ponto sem dado derruba a area inteira. Isso foi
    medido e o desperdicio e grande: em 40 areas sorteadas sobre celulas reais
    da grade de Curitiba, 18 foram recusadas -- e 14 dessas tinham 80% ou mais
    dos pontos com dado completo. Havia recusa por 1 ponto em 25.

    O QUE `descartar` NAO PODE FAZER, e por isso a resposta declara tanta
    coisa: o ponto sem cobertura NAO e ponto sem suscetibilidade. E, pior, a
    ausencia NAO e aleatoria -- ela se concentra em borda de cobertura, corpo
    d'agua e fora do limite do territorio. Descartar esses pontos e uma
    analise de casos completos sobre dado que falta de forma sistematica, e
    o escore resultante vale para a PARTE COBERTA da area, podendo estar
    enviesado em relacao a area inteira. Quem le precisa receber isso junto
    com o numero, e nao depois.
    """
    feats = modelo["features"]
    linhas, usados, descartados = [], [], []
    for i, p in enumerate(pontos):
        camadas = p.get("camadas", p)
        linha, completo = [], True
        for f in feats:
            v = camadas.get(f)
            if v is None or not np.isfinite(float(v)):
                if sem_cobertura != "descartar":
                    return False, _falha(
                        "variaveis_presentes", STATUS_SEM_DADO,
                        f"variavel {f!r} ausente ou nao finita no ponto {i}",
                        {"variaveis_exigidas": feats}), np.empty(0), {}
                descartados.append({"ponto": i, "variavel": f})
                completo = False
                break
            linha.append(float(v))
        if completo:
            linhas.append(linha)
            usados.append(i)

    cobertura = {
        "pontos_pedidos": len(pontos),
        "pontos_usados": len(usados),
        "pontos_descartados": len(descartados),
        "fracao_coberta": (round(len(usados) / len(pontos), 4)
                           if pontos else 0.0),
        "politica": sem_cobertura,
    }
    if descartados:
        cobertura["primeiros_descartados"] = descartados[:5]
        cobertura["nao_e"] = (
            "ponto descartado nao e ponto sem suscetibilidade: e ponto onde o "
            "modelo nao tem dado para responder")
        cobertura["vies"] = (
            "a ausencia de cobertura nao e aleatoria (borda, corpo d'agua, "
            "fora do limite do territorio): o escore vale para a parte "
            "coberta e pode nao representar a area inteira")

    if not linhas:
        return False, _falha(
            "variaveis_presentes", STATUS_SEM_DADO,
            "nenhum ponto da area tem todas as variaveis exigidas",
            {"variaveis_exigidas": feats, "cobertura": cobertura}
        ), np.empty(0), cobertura

    return True, None, np.array(linhas, dtype=float), cobertura


def _distancia_do_ajuste(X: np.ndarray, modelo: dict, faixa: dict) -> dict:
    """Quao longe do centro do ajuste esta a area, olhando o ponto INTEIRO.

    -------------------------------------------------------------------------
    O QUE ISTO ACRESCENTA AO G5, e e uma lacuna real
    -------------------------------------------------------------------------

    O G5 pergunta, variavel por variavel, se os pontos caem na faixa 5-95% do
    ajuste. E uma pergunta MARGINAL: ela olha uma variavel de cada vez e nunca
    olha a combinacao.

    Um ponto pode estar confortavelmente dentro de todas as faixas e ainda ser
    uma COMBINACAO que o ajuste nao viu -- HAND alto com TWI alto, por exemplo,
    onde cada um sozinho e comum e os dois juntos nao sao. O G5 aprova esse
    ponto, e aprova certo: nenhuma variavel dele esta extrapolando. Mas quem le
    o escore nao fica sabendo que a area e incomum.

    Isso importa agora porque a AOI passou a ser desenhada a mao. Enquanto as
    consultas vinham de celulas da grade, a combinacao era, por construcao, uma
    que a regiao produz. Um poligono arbitrario nao tem essa garantia.

    -------------------------------------------------------------------------
    O QUE ISTO NAO E, e os tres limites sao declarados na resposta
    -------------------------------------------------------------------------

    1. NAO E O ENCODER DE TERRENO. O encoder (frente B) mede distancia de
       dominio no espaco de embedding, e la a propriedade foi MEDIDA: entre
       tres seletores, `distancia_do_centro` foi o unico a ordenar melhor que o
       acaso. Aqui a distancia e euclidiana padronizada nas proprias variaveis
       do modelo -- outro estimador da mesma ideia, e a propriedade medida
       daquele NAO transfere para este.

    2. NAO CAPTURA CORRELACAO. O artefato servido traz media e desvio por
       variavel (`padronizacao`), e nao a matriz de covariancia do ajuste. Sem
       ela nao ha distancia de Mahalanobis, e esta distancia trata as
       variaveis como independentes. Ela e um resumo diferente do G5, e nao
       um resumo estritamente melhor.

    3. NAO TEM LIMIAR. Nenhum corte foi calibrado para ela, entao ela nao
       recusa nada e nao entra em `status`. Anexar um limiar aqui sem
       calibracao seria exatamente o que a A7 existe para impedir do outro lado
       do projeto.
    """
    feats = modelo["features"]

    # BLOCO DESCRITIVO NAO PODE DERRUBAR PORTAO.
    #
    # `g5_dominio` decidia com `features` e `faixa_dominio_5_95` e mais nada.
    # Ao pendurar esta distancia no detalhe dele, eu passei a exigir tambem
    # `padronizacao` -- e quebrei todo chamador que monta modelo sintetico para
    # exercitar o portao, que e exatamente o caso dos testes de dominio. Um
    # numero que so descreve nao pode impedir a decisao de acontecer.
    padronizacao = modelo.get("padronizacao") or {}
    if any(f not in padronizacao for f in feats):
        ausentes = [f for f in feats if f not in padronizacao]
        return {
            "calculavel": False,
            "motivo": ("o artefato servido nao traz `padronizacao` para "
                       f"{ausentes}; sem o centro do ajuste nao ha distancia "
                       "a ele"),
            "e_portao": False,
        }

    mu = np.array([padronizacao[f]["media"] for f in feats])
    sd = np.array([padronizacao[f]["desvio"] for f in feats])
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd

    # Dividir pela raiz do numero de variaveis mantem a escala comparavel entre
    # modelos de tamanhos diferentes: Petropolis tem uma variavel e Recife tem
    # cinco, e sem isto a distancia de Recife seria maior so por ser 5-D.
    d = np.linalg.norm(Z, axis=1) / math.sqrt(len(feats))

    dentro_de_todas = np.ones(len(X), dtype=bool)
    for j, f in enumerate(feats):
        lo, hi = faixa[f]
        dentro_de_todas &= (X[:, j] >= lo) & (X[:, j] <= hi)

    resumo = {
        "calculavel": True,
        "unidade": "desvios-padrao do ajuste, por variavel",
        "definicao": ("norma euclidiana do ponto padronizado pelo centro do "
                      "ajuste, dividida pela raiz do numero de variaveis"),
        "p50": round(float(np.median(d)), 4),
        "p90": round(float(np.percentile(d, 90)), 4),
        "maximo": round(float(np.max(d)), 4),
        "pontos_dentro_de_todas_as_faixas": int(dentro_de_todas.sum()),
        "pontos": int(len(X)),
        "e_portao": False,
        "por_que_existe": ("o G5 olha uma variavel de cada vez; um ponto pode "
                           "estar dentro de todas as faixas e ainda ser uma "
                           "combinacao que o ajuste nao viu"),
        "nao_e": ("nao e o encoder de terreno, nao captura correlacao entre as "
                  "variaveis (o artefato servido nao traz covariancia) e nao "
                  "tem limiar calibrado: e descritiva, e nao decide nada"),
    }
    if dentro_de_todas.any():
        resumo["p90_entre_os_dentro_de_todas_as_faixas"] = round(
            float(np.percentile(d[dentro_de_todas], 90)), 4)
    return resumo


def g5_dominio(X: np.ndarray, modelo: dict) -> tuple[bool, dict | None, dict]:
    feats = modelo["features"]
    faixa = modelo["faixa_dominio_5_95"]
    dentro, fora = {}, []
    for j, f in enumerate(feats):
        lo, hi = faixa[f]
        pct = 100.0 * float(np.mean((X[:, j] >= lo) & (X[:, j] <= hi)))
        dentro[f] = round(pct, 1)
        if pct < MAX_PCT_EXTRAPOLACAO:
            fora.append(f)
    pct_variaveis_fora = 100.0 * len(fora) / len(feats)
    detalhe = {"cobertura_por_variavel_pct": dentro,
               "variaveis_em_extrapolacao": fora,
               "distancia_ao_centro_do_ajuste": _distancia_do_ajuste(X, modelo,
                                                                     faixa)}
    if pct_variaveis_fora > MAX_PCT_EXTRAPOLACAO:
        return False, _falha("dominio_coberto", STATUS_SEM_DADO,
                             f"{len(fora)} de {len(feats)} variaveis fora da "
                             "faixa 5-95% do ajuste: o escore seria extrapolacao",
                             detalhe), detalhe
    return True, None, detalhe


# -------------------------------------------------------------------- escore

def escore_com_ic(X: np.ndarray, modelo: dict) -> dict:
    """Escore da AREA e IC95 pelas replicas de bootstrap gravadas no SVC-01."""
    feats = modelo["features"]
    mu = np.array([modelo["padronizacao"][f]["media"] for f in feats])
    sd = np.array([modelo["padronizacao"][f]["desvio"] for f in feats])
    Z = (X - mu) / sd

    beta = np.array([modelo["coef"][f] for f in feats])
    eta = modelo["intercepto"] + Z @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    escore = float(np.mean(p))

    rep = np.array(modelo.get("replicas_bootstrap") or [])
    if rep.size:
        etas = rep[:, 0][None, :] + Z @ rep[:, 1:].T           # (n_pontos, n_rep)
        medias = np.mean(1.0 / (1.0 + np.exp(-etas)), axis=0)  # (n_rep,)
        lo, hi = np.percentile(medias, [2.5, 97.5])
        ic = [round(float(lo), 4), round(float(hi), 4)]
        n_rep = int(rep.shape[0])
    else:
        ic, n_rep = None, 0

    return {"escore": round(escore, 4), "ic95": ic, "n_replicas": n_rep,
            "escore_por_ponto": [round(float(v), 4) for v in p],
            "z": Z, "contribuicoes": (Z * beta)}


def maturidade_da_regiao(regiao: str, modelo: dict,
                         com_referencia_local: set | None = None) -> str:
    """Quatro niveis, e a diferenca entre os dois de transferencia importa.

    Curitiba e Petropolis sao ambas servidas por modelo que nao as viu, mas nao
    sao o mesmo caso: Curitiba tem inventario local (que o projeto decidiu nao
    usar como criterio de aprovacao, `metodo_aplicacao_sem_rotulo_local_v1.md`),
    e Petropolis nao tem nenhum ponto rotulado. Chamar as duas de
    `transferencia_caracterizada` esconderia que numa delas existe um caminho
    para verificar depois, e na outra nao existe nem isso.
    """
    if regiao not in modelo.get("regioes_servidas", []):
        return "nao_suportada"
    # `fontes_do_ajuste` E CHAVEADO POR FONTE, E REGIAO NEM SEMPRE E FONTE.
    #
    # Enquanto cada regiao servida era sua propria fonte (`recife`, `curitiba`),
    # olhar so as fontes bastava. O pool areal quebrou isso: o Rio Grande do Sul
    # entra no ajuste como cinco AOIs da fonte `cems`, entao a checagem por
    # fonte dizia que ele NAO estava no treino -- e o contrato devolvia
    # `transferencia_sem_referencia_local`, cuja definicao e "nao ha nenhum
    # ponto rotulado na regiao". O RS tem 1.585 positivos e 9.207 negativos
    # OBSERVADOS. Era a declaracao mais errada que o produto podia emitir sobre
    # o melhor dado que ele tem.
    #
    # Por isso o modelo passa a declarar `regioes_no_ajuste` quando regiao e
    # fonte nao coincidem, e as duas listas sao consultadas.
    import regioes as _reg
    treinado_na_regiao = _reg.esta_no_ajuste(regiao, modelo)
    if treinado_na_regiao:
        # A falha pode ser DO MODELO ou DESTA regiao. Um modelo nacional que
        # fecha em media pode nao fechar num pedaco do pais, e chamar esse
        # pedaco de `validado` pelo desempenho medio seria dizer que a medida
        # local nao importa. `falhas_de_criterio_por_regiao` vem do
        # leave-one-region-out, e o `svc01` a LE do artefato do br04 em vez de
        # carregar numero fixo.
        por_regiao = (modelo.get("falhas_de_criterio_por_regiao") or {})
        if modelo.get("falhas_de_criterio") or por_regiao.get(regiao):
            return "mvp_local"
        return "validado"
    if com_referencia_local is not None and regiao not in com_referencia_local:
        return "transferencia_sem_referencia_local"
    return "transferencia_caracterizada"


# --------------------------------------------------------------- explicacao

def explicar(feats: list[str], X: np.ndarray, contribuicoes: np.ndarray,
             escore: float, modelo: dict | None = None) -> dict:
    """Le o payload ja decidido e descreve o sinal de cada contribuicao.

    Nao ha texto livre: cada frase e montada da mesma grandeza que entrou no
    escore, entao divergir do payload e impossivel por construcao.

    O FILTRO FISICO, e por que ele e necessario
    -------------------------------------------
    Descrever a contribuicao de TODA variavel produzia frase falsa sobre o
    mundo. Em Recife a versao anterior dizia, ao pe da letra, que "elevacao
    pesa a favor" e "declividade pesa a favor" -- ou seja, que estar mais alto
    e em terreno mais ingreme AUMENTA a suscetibilidade a alagamento. A conta
    estava certa (`coef x z`, a mesma que entrou no escore); o coeficiente e
    que tem sinal contrario a fisica, e a frase apenas o repetia em portugues.

    Entao a frase passa a citar so as variaveis que o SVC-01 marcou como
    `fisicamente_verificada`: sinal batendo com o esperado E IC95 sem cruzar
    zero. Sinal certo com IC cruzando zero tambem sai -- o ajuste nao
    estabeleceu efeito nenhum sobre aquela variavel, e afirmar sentido seria
    inventar.

    `itens` continua COMPLETO, com todas as variaveis: o filtro e sobre o que
    a frase afirma, nao sobre o que o payload mostra. Quem audita precisa ver
    a contribuicao de todas; quem le a frase nao pode receber afirmacao falsa.
    Cada item carrega `fisicamente_verificada` e `citada_na_frase`.
    """
    sinal = (modelo or {}).get("sinal_por_variavel", {})
    verificadas = {f for f, v in sinal.items() if v.get("fisicamente_verificada")}
    # Modelo antigo, sem o bloco de sinal: nao ha o que filtrar, e calar tudo
    # seria pior que a frase completa. O campo `filtro` declara qual caso e.
    tem_filtro = bool(sinal)

    medias_contrib = contribuicoes.mean(axis=0)
    medias_valor = X.mean(axis=0)
    ordem = np.argsort(-np.abs(medias_contrib))
    itens = []
    for j in ordem:
        f = feats[j]
        c = float(medias_contrib[j])
        v = float(medias_valor[j])
        ok = (f in verificadas) if tem_filtro else True
        sentido = "a favor" if c > 0 else ("contra" if c < 0 else "neutro")
        itens.append({
            "variavel": f, "nome": NOME_LEGIVEL.get(f, f),
            "valor_medio": round(v, 3), "unidade": UNIDADE.get(f, ""),
            "contribuicao": round(c, 4),
            "sentido": sentido,
            "fisicamente_verificada": bool(ok),
            "citada_na_frase": bool(ok and sentido != "neutro"),
        })
    frases = [f"{i['nome']} = {i['valor_medio']}{(' ' + i['unidade']).rstrip()} "
              f"pesa {i['sentido']}" for i in itens if i["citada_na_frase"]]
    omitidas = [i["variavel"] for i in itens if not i["fisicamente_verificada"]]
    return {
        "gerador": "regras_sobre_o_payload",
        "escore_explicado": round(escore, 4),
        "itens": itens,
        "filtro": ("so variaveis com sinal fisico verificado e IC95 sem cruzar "
                   "zero" if tem_filtro else "nenhum: modelo sem bloco de sinal"),
        "variaveis_omitidas_da_frase": omitidas,
        "por_que_omitidas": (
            "entram no escore, mas o ajuste nao estabeleceu sentido fisico "
            "confiavel para elas -- afirmar seria inventar") if omitidas else None,
        "texto": ("; ".join(frases) + ".") if frases else
                 ("nenhuma variavel do modelo tem sentido fisico verificado: "
                  "o escore sai sem frase de explicacao" if tem_filtro else
                  "nenhuma variavel se afasta da media do ajuste."),
    }


def afirmacao_autorizada(modelo: dict) -> dict:
    """O que o escore pode AFIRMAR, pela regra de uso da admissao (03/10/2026).

    Modelo ajustado sobre ausencia de registro mede propensao a registrar, nao
    inundacao. Fica FORA do `model_card`, cujos campos sao os da v1: entra como
    campo opcional da resposta, como `cadeia_neural`, e nao muda quem e recusado.
    """
    sys.path.insert(0, str(REPO / "scripts" / "metodo"))
    from afirmacao_dos_servidos import estimando_do_modelo
    a = estimando_do_modelo(modelo)
    return {"estimando": a["estimando"], "frase": a["frase"], "motivos": a["motivos"]}


def model_card(modelo: dict, regiao: str, maturidade: str) -> dict:
    d = modelo.get("desempenho", {})
    return {
        "modelo": modelo["nome"],
        "regiao": regiao,
        "maturidade": maturidade,
        "mecanismo": modelo.get("mecanismo"),
        "variaveis": modelo["features"],
        "estimador": "regressao logistica penalizada de Firth",
        "ajustado_em": {
            "n": modelo.get("n"), "grupos": modelo.get("grupos"),
            "grupos_positivos": modelo.get("grupos_positivos"),
            "grupos_negativos": modelo.get("grupos_negativos"),
            "fontes": modelo.get("fontes_do_ajuste"),
            "niveis_de_negativo": modelo.get("niveis_de_negativo"),
            "prevalencia": modelo.get("prevalencia_do_ajuste"),
        },
        "desempenho": {"auc_agrupada": d.get("auc_cv"), "folds": d.get("folds"),
                       "unidade_de_reamostragem":
                           modelo.get("unidade_de_reamostragem")},
        # O orcamento de EPV sai do proprio artefato em vez de ficar so no
        # relatorio: quem le o escore precisa saber quantas variaveis o
        # conjunto comportava, nao quantas foram usadas.
        "orcamento_epv": {
            "variaveis_usadas": len(modelo["features"]),
            "variaveis_permitidas": modelo.get("orcamento_estrito"),
            "eventos_por_variavel": round(
                min(modelo.get("grupos_positivos") or 0,
                    modelo.get("grupos_negativos") or 0) / len(modelo["features"]), 2),
            "minimo_exigido": 10,
        },
        "sinal_por_variavel": modelo.get("sinal_por_variavel", {}),
        "variaveis_fisicamente_verificadas": modelo.get(
            "variaveis_fisicamente_verificadas", []),
        # Aderencia temporal do CONJUNTO DE AJUSTE: as duas classes foram
        # observadas nos mesmos dias? Nao confundir com `suporte_cientifico
        # .temporal`, que pergunta sobre janela futura. Este vem antes: se as
        # classes nao dividem datas, uma variavel que varia no tempo pode
        # estar separando dias em vez de lugares.
        "aderencia_temporal": modelo.get("aderencia_temporal", {}),
        "criterios_nao_atingidos": modelo.get("falhas_de_criterio", []),
        "limites_de_uso": modelo.get("limitacoes_declaradas", []),
        "por_que_este_conjunto": modelo.get("por_que_este_conjunto"),
        "ic_do_escore": ("bootstrap do preditor linear reamostrando grupos, "
                         f"{modelo.get('n_replicas', 0)} replicas"),
        "nao_e": ("nao e previsao de evento nem mapa operacional: e escore de "
                  "predisposicao do terreno, com as limitacoes acima"),
    }


def rain_evidence(modelo: dict, requisicao: dict, X: np.ndarray,
                  dominio: dict) -> dict:
    feats = list(modelo["features"])
    rain = [f for f in feats if f.startswith("rain_")]
    scenario = (requisicao.get("cenario_de_chuva") or "").strip() or None
    if not rain:
        return {
            "used": False,
            "scenario": scenario,
            "reason": "regional model does not use rain; supplied rain cannot alter score",
        }
    values = {f: {
        "mean": round(float(np.mean(X[:, feats.index(f)])), 6),
        "min": round(float(np.min(X[:, feats.index(f)])), 6),
        "max": round(float(np.max(X[:, feats.index(f)])), 6),
    } for f in rain}
    meta = rain_contract()
    scenario_meta = (scenario_evidence(
        scenario, modelo.get("faixa_dominio_5_95")) if scenario else None)
    return {
        "used": True,
        "provider": meta["provider"],
        "product": meta["product"],
        "scenario": scenario,
        "values_used": values,
        "reference_period": None,
        "period_policy": meta["period_policy"],
        "decay_k": meta["variables"]["rain_decay_index"]["decay_parameter"],
        "window_days": meta["lookback_days"],
        "spatial_resolution_degrees": meta["spatial_resolution_degrees"],
        "timezone": meta["timezone"],
        "method_version": meta["method_version"],
        "training_domain": {f: modelo.get("faixa_dominio_5_95", {}).get(f)
                            for f in rain},
        "inside_domain": not bool(set(rain) & set(
            dominio.get("variaveis_em_extrapolacao") or [])),
        "extrapolation": bool(set(rain) & set(
            dominio.get("variaveis_em_extrapolacao") or [])),
        "scenario_evidence": scenario_meta,
        "temporal_adherence": modelo.get("aderencia_temporal", {}),
        "limitations": [
            "scenario values are forcing assumptions, not measurements at each point",
            "rain_max_24h is not physically verified in the served model",
            "low shared-date support prevents a causal rain claim",
        ],
    }


# ---------------------------------------------------------------- cobertura

def regioes(modelos: dict | None = None,
            com_referencia_local: set | None = None,
            matriz_cientifica=None) -> dict:
    """A consulta de cobertura -- o que um `GET /regioes` devolveria.

    POR QUE ELA EXISTE AQUI, e nao num servidor:

    a Tela 1 do esboco de demonstrador precisa listar as AOIs e a maturidade
    de cada uma ANTES de qualquer escore. Isso e pergunta de contrato, nao de
    transporte: quais regioes tem modelo, em que nivel de maturidade, e o que
    cada nivel autoriza afirmar. O verbo HTTP e o nome do transporte que
    ainda nao existe; a resposta e esta.

    O vocabulario de maturidade e o unico oficial do projeto -- `mvp_local`,
    `transferencia_caracterizada`, `transferencia_sem_referencia_local`,
    `nao_suportada`. O servidor antigo (`susc_20e`) usava outro
    (`available` / `limited_evidence` / `insufficient`), incompativel, e esta
    funcao existe tambem para que so haja um.

    A maturidade NAO e escrita a mao aqui: sai de `maturidade_da_regiao()`, a
    mesma que `inferir()` chama. Listar e inferir nao podem discordar.
    """
    modelos = carregar_modelos() if modelos is None else modelos
    if com_referencia_local is None:
        com_referencia_local = regioes_com_referencia_local()
    if matriz_cientifica is None:
        matriz_cientifica = carregar_matriz_cientifica()

    # A LISTAGEM PERGUNTA A MESMA FUNCAO QUE A INFERENCIA.
    #
    # A versao anterior varria os modelos e escrevia `saida[regiao]` a cada
    # um, entao quando dois modelos declaravam a mesma regiao o ULTIMO em
    # ordem alfabetica vencia -- enquanto `inferir()` usa `modelo_da_regiao`,
    # que devolve o PRIMEIRO servivel. As duas respostas podiam divergir, e
    # divergiram: em 17/09/2026 o `GET /regioes` dizia
    # `transferencia_caracterizada` para o Rio Grande do Sul e a inferencia
    # devolvia `validado`, no mesmo instante, com os mesmos artefatos.
    #
    # Perguntar `modelo_da_regiao` aqui e o que torna a promessa do docstring
    # verificavel em vez de intencionada.
    saida = {}
    candidatas = {r for m in modelos.values() if m.get("servivel")
                  for r in m.get("regioes_servidas", [])}
    for regiao in sorted(candidatas):
        m = modelo_da_regiao(regiao, modelos)
        if m is None:
            continue
        nome = m.get("nome")
        mat = maturidade_da_regiao(regiao, m, com_referencia_local)
        d = m.get("desempenho", {})
        saida[regiao] = {
            "maturidade": mat,
            "o_que_a_maturidade_autoriza": AUTORIZA_POR_MATURIDADE[mat],
            "modelo": nome,
            "mecanismo": m.get("mecanismo"),
            "variaveis_exigidas": m["features"],
            "auc_agrupada": d.get("auc_cv"),
            "criterios_nao_atingidos": m.get("falhas_de_criterio", []),
            "tem_referencia_local": regiao in com_referencia_local,
            "limitacoes": m.get("limitacoes_declaradas", []),
            # Maturidade e suporte temporal sao EIXOS DIFERENTES, e a
            # listagem mostra os dois lado a lado justamente porque a
            # leitura errada e natural: `mvp_local` fala de ter modelo
            # proprio, e nao de a relacao aprendida transportar no tempo.
            # A maturidade entra porque `e_portao_do_contrato` responde
            # por escopo: sem ela a listagem diria `false` ate onde o
            # portao de fato decide.
            "suporte_cientifico": suporte_cientifico_do_escopo(
                regiao, m, matriz_cientifica, mat),
        }
    return {
        "contrato": VERSAO_CONTRATO,
        "regioes": saida,
        "maturidades_possiveis": list(AUTORIZA_POR_MATURIDADE),
        "estados_de_suporte_possiveis": list(suporte.ESTADOS_DE_SUPORTE),
        "suporte_nao_e_maturidade": (
            "maturidade diz se existe modelo proprio da regiao; suporte "
            "temporal diz se ha evidencia de que a relacao aprendida "
            "transporta para eventos posteriores. Um nao determina o outro"),
        "politica_do_portao_temporal": POLITICA_PORTAO_TEMPORAL,
        "o_que_a_politica_faz": politica_temporal.DESCRICAO[
            POLITICA_PORTAO_TEMPORAL],
        "fora_desta_lista": ("qualquer outra geometria devolve "
                             f"{STATUS_SEM_REGIAO}; nenhum escore por analogia"),
        # DERIVADO, e nao declarado. Este campo era a constante "validado", e
        # a constante estava certa ate 17/09/2026 -- quando o negativo
        # observado brasileiro levou quatro macrorregioes e o Rio Grande do
        # Sul ao nivel, ela virou uma afirmacao falsa que ninguem tinha de
        # lembrar de atualizar. Agora ele e lido da propria listagem: se
        # amanha uma regiao subir ou cair de nivel, este campo acompanha.
        "niveis_que_nenhuma_regiao_alcanca": [
            nivel for nivel in AUTORIZA_POR_MATURIDADE
            if nivel != "nao_suportada"
            and not any(d["maturidade"] == nivel for d in saida.values())],
        "regioes_por_maturidade": {
            nivel: sorted(r for r, d in saida.items()
                          if d["maturidade"] == nivel)
            for nivel in AUTORIZA_POR_MATURIDADE
            if any(d["maturidade"] == nivel for d in saida.values())},
    }


# ------------------------------------------------------------------ contrato

def inferir(requisicao: dict, modelos: dict | None = None,
            caixas: dict | None = None,
            com_referencia_local: set | None = None,
            matriz_cientifica=None) -> dict:
    """A funcao que um servidor chamaria. Sem rede, sem estado, sem efeito."""
    modelos = carregar_modelos() if modelos is None else modelos
    caixas = caixas_das_regioes() if caixas is None else caixas
    if com_referencia_local is None:
        com_referencia_local = regioes_com_referencia_local()
    if matriz_cientifica is None:
        matriz_cientifica = carregar_matriz_cientifica()

    ok, falha, pontos = g1_geometria(requisicao)
    if not ok:
        return falha
    ok, falha, regiao = g2_regiao(requisicao, pontos, caixas)
    if not ok:
        return falha
    ok, falha, modelo = g3_modelo(regiao, modelos)
    if not ok:
        return falha
    pontos, derivadas = derivar_relativas(pontos, modelo, regiao)
    ok, falha, X, cobertura = g4_variaveis(
        pontos, modelo, requisicao.get("pontos_sem_cobertura") or "recusar")
    if not ok:
        return falha
    ok, falha, dominio = g5_dominio(X, modelo)
    if not ok:
        return falha

    # A MATURIDADE E CALCULADA ANTES DE G6, e nao depois, porque a politica
    # `por_maturidade` precisa dela para decidir. Isso nao mexe na ordem da
    # escada: G6 continua sendo o sexto degrau e continua sendo avaliado
    # depois de G5 -- o que mudou e o instante em que um valor ja disponivel
    # passa a ser lido. Requisicao que falha em G1-G5 retorna antes daqui e
    # nunca chega a calcular maturidade, como era.
    maturidade = maturidade_da_regiao(regiao, modelo, com_referencia_local)

    # A semantica da data pertence ao contrato, nao ao transporte. Ela entra
    # no mesmo degrau temporal e nunca transforma data declarada em evidencia.
    ok_contexto, contexto_temporal = contexto_temporal_da_requisicao(requisicao)
    if not ok_contexto:
        return _falha(
            "suporte_temporal", STATUS_SEM_DADO,
            contexto_temporal["motivo"],
            {"regiao": regiao, "maturidade": maturidade,
             "contexto_temporal": contexto_temporal})

    # G6 e avaliado ANTES de calcular o escore. Quem decide se ele recusa e a
    # politica, nao este bloco: sob `por_maturidade`, escopo que declara
    # transferencia passa e o estado de suporte viaja como limitacao.
    ok, falha, _ = g6_suporte_temporal(
        regiao, modelo, matriz_cientifica, maturidade)
    if not ok:
        return falha
    limitacoes = list(modelo.get("limitacoes_declaradas", []))
    if dominio["variaveis_em_extrapolacao"]:
        limitacoes.append(
            "extrapolacao de dominio em "
            f"{dominio['variaveis_em_extrapolacao']}: a requisicao esta fora da "
            "faixa 5-95% que o modelo viu no ajuste")
    if cobertura.get("pontos_descartados"):
        # Vai para `limitacoes` e nao so para um bloco proprio: quem le a
        # resposta rapido le o escore e as limitacoes, e "este escore cobre
        # 84% da area que voce desenhou" pertence a essa leitura.
        limitacoes.append(
            f"cobertura parcial da area: {cobertura['pontos_usados']} de "
            f"{cobertura['pontos_pedidos']} pontos tinham todas as variaveis "
            f"({cobertura['fracao_coberta']:.0%}). O escore vale para a parte "
            "coberta; ponto sem cobertura nao e ponto sem suscetibilidade, e "
            "a ausencia nao e aleatoria")
    if modelo.get("falhas_de_criterio"):
        limitacoes.append(
            "criterio de leitura nao atingido no ajuste: "
            + "; ".join(modelo["falhas_de_criterio"]))
        if POLITICA_CRITERIO_NAO_ATINGIDO == "recusa":
            return _falha("modelo_dentro_dos_criterios", STATUS_SEM_DADO,
                          "o modelo da regiao nao atinge os criterios de leitura "
                          "fixados antes do ajuste",
                          {"regiao": regiao, "maturidade": maturidade,
                           "criterios_nao_atingidos": modelo["falhas_de_criterio"]})

    r = escore_com_ic(X, modelo)
    explicacao = explicar(modelo["features"], X, r["contribuicoes"],
                          r["escore"], modelo)
    # Contrato v1: evidência nova viaja dentro do bloco extensível de
    # explicação, sem criar uma nova chave top-level nem alterar o model card.
    explicacao["evidencia_chuva"] = rain_evidence(
        modelo, requisicao, X, dominio)
    resposta = {
        "status": STATUS_OK,
        "contrato": VERSAO_CONTRATO,
        "regiao": regiao,
        "maturidade": maturidade,
        "unidade_de_resposta": "area",
        "n_pontos": int(X.shape[0]),
        "cobertura_da_area": cobertura,
        "escore": r["escore"],
        "ic95": r["ic95"],
        # A nomenclatura fica explicita porque o esboco de telas ja confundiu
        # as duas coisas: o que sai daqui e PROBABILIDADE PROJETADA pelo modelo
        # ajustado sobre a requisicao, e nao desempenho fora da amostra. AUC e
        # propriedade do modelo e vive no `model_card`; escore e da requisicao.
        "escore_detalhe": {
            # `tipo` NAO muda na v1. O retrato do contrato o trava, e trocar
            # o rotulo do escore e mudanca de comportamento para quem consome
            # -- `CONTRATO_VERSIONAMENTO.md` classifica isso como v2. Segue o
            # mesmo caminho da dimensao temporal: o valor correto viaja como
            # campo NOVO e OPCIONAL, e ligar a troca e decisao de versao.
            "tipo": "probabilidade_projetada",
            "tipo_se_v2": _tipo_do_escore(modelo),
            "definicao": ("media, sobre os pontos da area, da probabilidade que "
                          "o modelo ajustado atribui a cada ponto"),
            # A leitura que o numero admite nao e escolha de apresentacao: ela
            # sai da natureza da amostra negativa do ajuste. Com FUNDO -- pontos
            # que so nao tem registro, e nao observacao de que ali nao inundou --
            # a prevalencia nao e identificavel e o nivel absoluto depende de
            # quantos pontos de fundo alguem sorteou. O frontend ja dizia isso
            # numa string de glossario; aqui ele viaja junto do numero, para
            # quem consome a API e nunca ve a tela.
            "leitura_admitida": (modelo.get("amostra_negativa") or {}).get(
                "leitura_admitida", "nao_declarada"),
            "por_que_esta_leitura": (modelo.get("amostra_negativa") or {}).get(
                "fundamento", "a amostra negativa do ajuste nao foi declarada"),
            "amostra_negativa_do_ajuste": modelo.get("amostra_negativa"),
            "variaveis_derivadas_na_requisicao": derivadas or None,
            "ic95_obtido_por": ("percentil 2,5-97,5 das projecoes das replicas "
                                "de bootstrap de grupos gravadas no artefato "
                                "servido; nenhum reajuste em tempo de requisicao"),
            "n_replicas": r["n_replicas"],
            "escala": [0.0, 1.0],
            "nao_e": ("nao e AUC nem escore fora da amostra; nao ha p-valor no "
                      "artefato servido, e o contrato nao inventa um"),
        },
        "variaveis_usadas": modelo["features"],
        "dominio": dominio,
        "limitacoes": limitacoes,
        # Entrou na v1 como campo opcional e descritivo. Na v2 a dimensao
        # decide -- mas so onde a maturidade afirma validade local, e o bloco
        # diz por escopo, em `e_portao_do_contrato`, se decidiu aqui. Uma
        # resposta de SUCESSO que traga `true` ali e a de uma regiao cujo
        # suporte foi demonstrado; `false`, a de quem serve por transferencia.
        "suporte_cientifico": suporte_cientifico_do_escopo(
            regiao, modelo, matriz_cientifica, maturidade),
        # Campo NOVO e OPCIONAL, pelo mesmo caminho do `suporte_cientifico`:
        # mostra a trilha da cadeia neural e nomeia o elo onde o sistema se
        # calaria. Nao e portao -- `status` sai de G1-G5 e nao muda por ele.
        "cadeia_neural": bloco_da_cadeia(modelo, X, dominio),
        # Campo NOVO e OPCIONAL (03/10), pelo mesmo caminho: o que o escore pode
        # afirmar (propensao a registrar x inundacao). Nao e portao.
        "afirmacao_autorizada": afirmacao_autorizada(modelo),
        "model_card": model_card(modelo, regiao, maturidade),
        "explicacao": explicacao,
    }
    if (requisicao.get("tipo_de_resposta") is not None
            or requisicao.get("data_alvo") is not None
            or requisicao.get("data_corte_conhecimento") is not None):
        resposta["contexto_temporal"] = contexto_temporal
    return resposta


# ------------------------------------------------------------- demonstracao

def _requisicoes_de_demonstracao(modelos: dict) -> list[tuple[str, dict]]:
    """Uma requisicao por estado do contrato, com valores da propria base."""
    import pandas as pd

    demos = []
    if UNICA.exists():
        d = pd.read_parquet(UNICA)
        d = d[(d.admitido == True) & d.classe.isin([0, 1])]  # noqa: E712
        for regiao in ("recife", "curitiba"):
            s = d[d.fonte == regiao].dropna(
                subset=["lat", "lon", "elevation_m", "slope_deg", "hand_m",
                        "twi_dinf", "rain_max_24h", "rain_decay_index"]).head(12)
            if s.empty:
                continue
            pontos = [{"lat": float(r.lat), "lon": float(r.lon),
                       "camadas": {f: float(getattr(r, f)) for f in
                                   ("elevation_m", "slope_deg", "hand_m",
                                    "twi_dinf", "rain_max_24h",
                                    "rain_decay_index")}}
                      for r in s.itertuples()]
            demos.append((f"{regiao}: area com todas as camadas", {
                "geometria": {"tipo": "aoi", "crs": "EPSG:4326", "pontos": pontos},
                "periodo": {"inicio": "2024-01-01", "fim": "2024-12-31"}}))

    # Petropolis nao tem nenhuma linha na tabela unica, entao os valores vem da
    # grade derivada do proprio raster de terreno (SVC-03). Inventar numero aqui
    # seria pior que nao ter demonstracao -- ja aconteceu uma vez.
    grade = RUNS / "svc-03-grade" / "grade_petropolis.csv"
    if grade.exists():
        # As camadas vem do MODELO que serve Petropolis, e nunca de uma lista
        # escrita aqui. A versao anterior fixava `{"hand_m": ...}`, escrita
        # quando `fluvial_serra` tinha uma variavel so; quando o modelo passou
        # a exigir `twi_dinf` tambem, a demonstracao continuou mandando uma, o
        # G4 recusou por `variaveis_presentes` e Petropolis SUMIU de
        # `regioes_exercitadas` -- sem erro nenhum, que e a forma de defeito
        # que este projeto mais repete: a lista congelada tres passos acima.
        serra = modelos.get("fluvial_serra") or {}
        exigidas = [f for f in (serra.get("features") or []) if f != "elevation_rel_m"]
        g = pd.read_csv(grade, nrows=2000).dropna(subset=["escore"]).head(10)
        faltam = [f for f in exigidas if f not in g.columns]
        if not g.empty and exigidas and not faltam:
            pontos = [{"lat": float(r.lat), "lon": float(r.lon),
                       "camadas": {f: float(getattr(r, f)) for f in exigidas}}
                      for r in g.itertuples()]
            demos.append(("petropolis: area sem nenhum ponto rotulado", {
                "regiao": "petropolis",
                "geometria": {"tipo": "aoi", "crs": "EPSG:4326",
                              "pontos": pontos}}))
        elif faltam:
            # Declarado, e nao silenciado: uma coluna que a grade nao tem e
            # lacuna de derivacao, nao motivo para mandar meia requisicao.
            demos.append((
                "petropolis: grade sem as colunas que o modelo exige "
                f"({', '.join(faltam)})", {
                    "regiao": "petropolis",
                    "geometria": {"tipo": "aoi", "crs": "EPSG:4326",
                                  "pontos": [{"lat": float(g.iloc[0].lat),
                                              "lon": float(g.iloc[0].lon),
                                              "camadas": {}}]}}))
    demos.append(("petropolis: sem as camadas que o modelo exige", {
        "regiao": "petropolis",
        "geometria": {"tipo": "aoi", "crs": "EPSG:4326",
                      "pontos": [{"lat": -22.505, "lon": -43.178,
                                  "camadas": {}}]}}))
    demos.append(("variavel faltando", {
        "regiao": "recife",
        "geometria": {"tipo": "ponto", "crs": "EPSG:4326",
                      "pontos": [{"lat": -8.05, "lon": -34.9,
                                  "camadas": {"hand_m": 3.0}}]}}))
    demos.append(("CRS nao suportado", {
        "regiao": "recife",
        "geometria": {"tipo": "ponto", "crs": "EPSG:31985",
                      "pontos": [{"lat": -8.05, "lon": -34.9}]}}))
    demos.append(("geometria fora de toda regiao com modelo", {
        "geometria": {"tipo": "ponto", "crs": "EPSG:4326",
                      "pontos": [{"lat": 48.85, "lon": 2.35,
                                  "camadas": {"hand_m": 1.0, "twi_dinf": 9.0}}]}}))
    return demos


def avaliar_integracao(manifesto: dict, requisicao: dict) -> dict:
    """Contrato da IA integrada de pesquisa; nenhuma promoção por empacotamento.

    A percepção pós-evento nunca é entrada da ordem física. O consumidor não
    pode transformar a demonstração em consulta territorial validada.
    """
    uso = requisicao.get("uso", "consulta_lugar_novo")
    fisica = manifesto.get("fisica") or {}
    modelo = {"features": fisica.get("features_brutas", []),
              "contexto_cientifico": fisica.get("contexto_cientifico", {})}
    gate = avaliar_uso_contextual(modelo, uso)
    motivos = list(gate["motivos"])
    if not requisicao.get("pontos") and not requisicao.get("chip"):
        motivos.append("informe pontos físicos ou chip de desenvolvimento")
    if len(requisicao.get("pontos") or []) > 5000:
        motivos.append("consulta excedeu 5000 pontos")
    if requisicao.get("data_alvo") or requisicao.get("profundidade"):
        motivos.append("integração atual não demonstra previsão datada ou profundidade")
    if manifesto.get("promovido") is not False:
        motivos.append("manifesto não declara estado de pesquisa")
    return {"versao": "integracao_v1", "uso": uso, "autorizado": not motivos,
            "motivos": motivos, "maturidade": "desenvolvimento",
            "escore_fisico_usa_sar": False, "probabilidade_calibrada": False,
            "validacao_local": False, "promovido": False}


def avaliar_uso_contextual(modelo: dict, uso: str, *, verificacao_independente: dict | None = None) -> dict:
    """Distingue reproduzir um resultado de consultar um lugar antes do evento.

    Portao explicito para a linhagem de pesquisa nova; nao altera os pesos nem
    a politica G1-G6 dos modelos legados. Ausencia de metadados nao e permissao.
    A verificacao deve vir do auditor, nunca da declaracao do proprio modelo.
    Esta funcao confere o vinculo; nao produz nem autentica a auditoria.
    """
    motivos = []
    usos = {"demonstracao_historica", "consulta_lugar_novo"}
    if uso not in usos:
        motivos.append("uso cientifico desconhecido")
    contexto = modelo.get("contexto_cientifico") or {}
    if uso == "consulta_lugar_novo":
        if contexto.get("populacao") != "territorio_consultavel":
            motivos.append("populacao de treino nao sustenta consulta de territorio delimitado antes do evento")
        if contexto.get("preditores_calculaveis_sem_rotulo") is not True:
            motivos.append("compatibilidade treino-inferencia sem amostra rotulada nao demonstrada")
        if "elevation_local_m" in (modelo.get("features") or []):
            motivos.append("elevation_local_m depende da amostra; nao substituir por piso do MDT em silencio")
        evidencia = contexto.get("avaliacao_independente") or {}
        if evidencia.get("estado") != "supported" or not evidencia.get("artefato_sha256"):
            motivos.append("avaliacao independente da variante e da pergunta de consulta nao demonstrada")
        from proveniencia import sha256_json
        verificada = verificacao_independente or {}
        if (verificada.get("estado") != "verificado"
                or verificada.get("modelo_sha256") != sha256_json(modelo)
                or verificada.get("artefato_sha256") != evidencia.get("artefato_sha256")):
            motivos.append("declaracao de suporte nao substitui auditoria independente vinculada a esta variante")
    return {"uso": uso, "autorizado": not motivos, "motivos": motivos,
            "versao": "uso_contextual_v1", "populacao": contexto.get("populacao"),
            "nao_e": "nao e autenticacao da auditoria, promocao, confianca calibrada ou prova de causalidade"}


def main() -> int:
    modelos = carregar_modelos()
    if not modelos:
        print(f"ABORTADO: {MODELOS} ausente ou vazio. Gere com: "
              "python scripts/servico/svc01_construir_modelos_servidos.py")
        return 1

    if "--regioes" in sys.argv:
        print(json.dumps(regioes(modelos), indent=2, ensure_ascii=False))
        return 0

    if "--requisicao" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--requisicao") + 1])
        r = inferir(json.loads(p.read_text(encoding="utf-8")), modelos)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    caixas = caixas_das_regioes()
    print("===SVC-02: CONTRATO DE INFERENCIA===")
    print(f"contrato={VERSAO_CONTRATO}  politica_criterio_nao_atingido="
          f"{POLITICA_CRITERIO_NAO_ATINGIDO}")
    print(f"modelos servidos: "
          f"{[n for n, m in modelos.items() if m.get('servivel')]}")
    print(f"caixas de regiao deduzidas da base: {list(caixas)}")

    respostas = []
    for titulo, req in _requisicoes_de_demonstracao(modelos):
        r = inferir(req, modelos, caixas)
        respostas.append({"caso": titulo, "requisicao": req, "resposta": r})
        print(f"\n--- {titulo} ---")
        print(f"  status={r['status']}", end="")
        if r["status"] == "ok":
            print(f"  regiao={r['regiao']}  maturidade={r['maturidade']}")
            print(f"  escore={r['escore']}  IC95={r['ic95']}  "
                  f"({r['n_pontos']} pontos, unidade={r['unidade_de_resposta']})")
            print(f"  variaveis={r['variaveis_usadas']}")
            print(f"  dominio={r['dominio']['cobertura_por_variavel_pct']}")
            print(f"  explicacao: {r['explicacao']['texto']}")
            for lim in r["limitacoes"]:
                print(f"  limitacao: {lim}")
        else:
            print(f"  gate={r['gate_que_nao_fechou']}")
            print(f"  detalhe: {r['detalhe']}")

    # a demonstracao guarda o payload sem as replicas, que sao do modelo
    (OUT / "respostas_demonstracao.json").write_text(
        json.dumps(respostas, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/servico/svc01_construir_modelos_servidos.py\n"
        "python scripts/servico/svc02_contrato_inferencia.py --demonstracao\n"
        "python -m pytest tests/test_svc_contrato_inferencia.py -q\n",
        encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
