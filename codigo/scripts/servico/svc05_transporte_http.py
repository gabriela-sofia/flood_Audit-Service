"""
SVC-05 -- O transporte HTTP do contrato de inferencia.

O QUE ESTE ARQUIVO E, e principalmente o que ele NAO e:

e transporte. Nada aqui decide. As rotas recebem o corpo da requisicao, chamam
`inferir()` ou `regioes()` do SVC-02, e devolvem o que veio -- sem reformatar,
sem renomear campo, sem interpretar status. Se uma regra de negocio aparecer
neste arquivo, ela esta no lugar errado.

A razao e a mesma que fez o SVC-02 nascer como funcao pura: o que precisa ser
auditado e o CONTRATO -- quais portoes existem, em que ordem, o que cada recusa
significa. Duplicar essa auditoria na camada HTTP criaria dois lugares onde a
verdade pode divergir, e a experiencia do projeto e que ela diverge.

-------------------------------------------------------------------------
O QUE FOI APROVEITADO DO PROTOTIPO DE JULHO, e o que foi deixado la
-------------------------------------------------------------------------

`outputs_public/data/susc_20e_api_contrato_inferencia_recife/` tem uma API
FastAPI que rodou de verdade em julho. Dela vem o PADRAO -- Pydantic para
validacao e precompute no `startup` -- e nada mais.

Nao foi aproveitado, de proposito:

  * `POST /score` e `GET /regions`. As rotas aqui sao `/inferencia` e
    `/regioes`, em portugues, que e o vocabulario do esboco de telas e dos
    docstrings do SVC-02 -- a fonte mais recente e mais auditada. Duas
    convencoes convivendo e como quem chama escolhe a errada.
  * `ScoreResponse` e os campos dele. O vocabulario de maturidade daquele
    prototipo (`available` / `limited_evidence` / `insufficient`) foi
    descontinuado, e ha teste no SVC-02 que falha se ele reaparecer. Os
    modelos Pydantic aqui sao construidos sobre os nomes que o SVC-02 usa hoje.

-------------------------------------------------------------------------
PRECOMPUTE NO STARTUP
-------------------------------------------------------------------------

`carregar_modelos()` le 1000 replicas de bootstrap por modelo;
`caixas_das_regioes()` e `regioes_com_referencia_local()` leem a tabela unica
inteira (43 MB); `carregar_grades()` monta tres KD-trees sobre ate 172 mil
celulas. Nada disso muda entre requisicoes. Carregar por requisicao tornaria o
servico inutil, e a decisao ja tinha sido tomada e documentada em julho.

-------------------------------------------------------------------------
SAUDE: a guarda que hoje so existe no terminal
-------------------------------------------------------------------------

`svc02.main()` aborta com "ABORTADO: ... ausente ou vazio" quando
`local_runs/svc-01-modelos/` nao tem artefato. Numa API isso precisa acontecer
ANTES de aceitar trafego, e nao como um 500 no meio da primeira requisicao.
`GET /saude` responde 503 enquanto o servico nao tiver o que servir.

-------------------------------------------------------------------------
O LIMITE DE PONTOS
-------------------------------------------------------------------------

O escore da area e a media sobre os pontos, e o IC projeta 1000 replicas por
ponto. Uma requisicao com um milhao de pontos nao e um erro do cliente que o
servidor deva absorver: e uma matriz de 10^9 elementos. `MAX_PONTOS` recusa com
422 antes de alocar qualquer coisa.

Uso:
    python scripts/servico/svc05_transporte_http.py --porta 8123

`uvicorn scripts.servico.svc05_transporte_http:app` NAO funciona: `scripts/`
nao e pacote (nao tem __init__.py), e o uvicorn resolve o alvo por importacao
de modulo. O `main()` daqui chama o uvicorn com o objeto `app` ja importado,
que e o caminho que roda.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))
sys.path.insert(0, str(REPO / "scripts" / "social"))

from log_auditoria import LogDeAuditoria  # noqa: E402
from geoprocessamento_sob_demanda import adquirir  # noqa: E402
from resolver_camadas import CENARIOS_CHUVA, carregar_grades, resolver  # noqa: E402
from consulta_cognitiva import responder  # noqa: E402
from proveniencia import bloco_de_proveniencia  # noqa: E402
from svc06_explicacao_llm import (  # noqa: E402
    anexar_evidencia_visual, sintetizar,
)
from svc02_contrato_inferencia import (  # noqa: E402
    AUTORIZA_POR_MATURIDADE, VERSAO_CONTRATO, caixas_das_regioes,
    carregar_modelos, inferir, modelo_da_regiao, regioes,
    regioes_com_referencia_local,
)
from camada_social import (  # noqa: E402
    VERSAO_CONTRATO_SOCIAL, avaliar_aoi_social, carregar_unidades_sociais_json,
)

MAX_PONTOS = 5_000
TITULO = "REV-P -- contrato de inferencia (SVC-05, MVP local)"

# `/inferencia` passa pelo verificador de grounding e anexa `sintese` e
# `auditoria_sintese`. Com o motor padrao -- deterministico, sem rede -- isso
# custa microssegundos e nao pode falhar; a chave existe para o dia em que um
# motor de rede entrar no lugar dele, quando "quem so quer o escore nao deve
# pagar por texto" volta a valer e esta linha vira `False`.
SINTESE_EM_INFERENCIA = True

# `proveniencia` -- bloco OPCIONAL com a cadeia do payload: qual modelo servido
# gerou o escore, com que hash, treinado com que dataset. Aditivo pela politica
# de `docs/CONTRATO_VERSIONAMENTO.md`, igual a `escore_detalhe` e
# `procedencia_das_camadas`: cliente v1 que nao conhece o campo ignora.
#
# NAO E CONDICAO PARA A RESPOSTA. Se a montagem falhar, a resposta sai sem o
# bloco -- a cadeia de proveniencia e informacao SOBRE o resultado, e trocar
# um escore valido por um 500 porque um hash nao pode ser lido seria inverter
# o que importa.
PROVENIENCIA_EM_INFERENCIA = True
MODELOS_SERVIDOS = REPO / "local_runs" / "svc-01-modelos"
UNIDADES_SOCIAIS_OFICIAIS = {
    "curitiba": (
        REPO / "local_runs" / "camada-social-territorial" / "ibge-curitiba"
        / "unidades_sociais_curitiba_ibge_v1.json"
    ),
}


# ------------------------------------------------------------------ esquema
#
# Os modelos Pydantic existem para validar a entrada e gerar o /docs. A SAIDA
# e devolvida como veio do SVC-02, sem passar por modelo de resposta: filtrar
# o payload por um esquema declarado aqui faria esta camada decidir o que o
# contrato mostra, que e exatamente o que ela nao pode fazer.

class Ponto(BaseModel):
    lat: float = Field(..., description="latitude em EPSG:4326")
    lon: float = Field(..., description="longitude em EPSG:4326")
    camadas: dict[str, float] | None = Field(
        default=None,
        description=("valores das variaveis do modelo. Se omitido, o servico "
                     "resolve pela celula da grade de 120 m que contem o ponto"))


class Geometria(BaseModel):
    tipo: str = "aoi"
    crs: str = Field(default="EPSG:4326",
                     description="unico CRS suportado hoje")
    pontos: list[Ponto]


class Periodo(BaseModel):
    inicio: str | None = None
    fim: str | None = None


class RequisicaoInferencia(BaseModel):
    geometria: Geometria
    regiao: str | None = Field(
        default=None,
        description="se omitida, e deduzida pela caixa envolvente das regioes")
    periodo: Periodo | None = Field(
        default=None,
        description=("campo legado. Na rota comum e aceito e declarado como "
                     "nao usado; com ingestao sob demanda, `fim` pode ser a "
                     "data de referencia da janela de chuva e a resposta "
                     "declara esse uso limitado. Novas consultas devem usar "
                     "tipo_de_resposta e data_alvo"))
    tipo_de_resposta: Literal[
        "retrato_estrutural", "retrospectiva_observada", "cenario_futuro",
        "pesquisa_reprodutivel"] | None = Field(
            default=None,
            description=("semantica temporal opt-in; omissao preserva o "
                         "retrato estrutural da v2"))
    data_alvo: str | None = Field(
        default=None,
        description=("data ISO YYYY-MM-DD sobre a qual a pergunta e feita; "
                     "nao cria validade temporal por si so"))
    data_corte_conhecimento: str | None = Field(
        default=None,
        description=("data maxima das fontes admissiveis numa reproducao; "
                     "hoje causa recusa porque ainda nao ha snapshots datados"))
    cenario_de_chuva: Literal[tuple(CENARIOS_CHUVA)] | None = Field(  # type: ignore[valid-type]
        default=None,
        description=("obrigatorio quando o modelo da regiao usa chuva e a "
                     "requisicao nao traz as camadas. Nao ha padrao: escolher "
                     "um em silencio esconderia o valor que gerou o escore"))
    ingestao_sob_demanda: bool = Field(
        default=False,
        description=("deriva as camadas do raster publico quando o ponto cai "
                     "fora da grade das tres regioes. Custa rede e minutos, "
                     "entao e opt-in por requisicao -- o padrao continua sendo "
                     "recusar em vez de sair baixando"))
    pontos_sem_cobertura: Literal["recusar", "descartar"] = Field(
        default="recusar",
        description=("o que fazer com ponto da area que nao tem todas as "
                     "variaveis. `recusar` (padrao) derruba a area inteira, "
                     "como sempre. `descartar` responde pela parte coberta e "
                     "declara a cobertura -- util porque uma area desenhada "
                     "quase sempre encosta na borda da grade, e recusar tudo "
                     "por 1 ponto em 25 desperdica area com dado"))
    sintese: bool = Field(
        default=True,
        description=("gerar a prosa em linguagem natural nesta resposta. "
                     "O padrao e True por compatibilidade, mas quem so quer "
                     "saber se a area e suscetivel deve pedir False: com "
                     "motor LLM local a sintese domina o tempo da resposta, "
                                     "e o escore ja esta pronto antes dela"))


class AOISocial(BaseModel):
    lat_min: float = Field(..., description="latitude minima em EPSG:4326")
    lon_min: float = Field(..., description="longitude minima em EPSG:4326")
    lat_max: float = Field(..., description="latitude maxima em EPSG:4326")
    lon_max: float = Field(..., description="longitude maxima em EPSG:4326")


class RequisicaoSocialAOI(BaseModel):
    aoi: AOISocial
    fisica: dict[str, Any] = Field(
        ...,
        description=("payload fisico ja produzido pelo SVC-02 ou equivalente; "
                     "a camada social le, mas nao altera esse payload"))
    regiao: str = Field(
        default="curitiba",
        description=("regiao social carregada. Hoje so ha fonte oficial "
                     "integrada para Curitiba"))
    cobertura_minima: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=("fracao minima da AOI coberta por unidades sociais para "
                     "responder status ok"))


class RequisicaoSocialConsulta(RequisicaoSocialAOI):
    pergunta: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="pergunta sobre a resposta social ja calculada")


# ------------------------------------------------------------------ estado

class Servico:
    """Tudo que e carregado uma vez e reusado por todas as requisicoes."""

    def __init__(self) -> None:
        self.modelos: dict = {}
        self.caixas: dict = {}
        self.com_referencia_local: set = set()
        self.grades: dict = {}
        self.log: LogDeAuditoria | None = None
        self.unidades_sociais: dict[str, list] = {}
        self.motivo_social_nao_pronto: dict[str, str] = {}
        self.pronto = False
        self.motivo_nao_pronto = "ainda nao carregado"

    def carregar(self) -> None:
        # Zerar ANTES de tentar. Sem isto, uma recarga que falha deixa o
        # `pronto` da carga anterior de pe, e `/saude` responde 200 sobre um
        # servico que perdeu os artefatos -- o oposto do que a rota existe
        # para fazer.
        self.pronto = False
        self.carregar_social()
        self.modelos = carregar_modelos()
        if not self.modelos:
            self.motivo_nao_pronto = (
                "local_runs/svc-01-modelos/ ausente ou vazio; gere com: "
                "python scripts/servico/svc01_construir_modelos_servidos.py")
            return
        self.caixas = caixas_das_regioes()
        self.com_referencia_local = regioes_com_referencia_local()
        self.grades = carregar_grades()
        self.log = LogDeAuditoria()
        self.pronto = True
        self.motivo_nao_pronto = ""

    def carregar_social(self) -> None:
        """Carrega fontes sociais oficiais sem condicionar a inferencia fisica."""

        self.unidades_sociais = {}
        self.motivo_social_nao_pronto = {}
        for regiao, caminho in UNIDADES_SOCIAIS_OFICIAIS.items():
            if not caminho.exists():
                self.motivo_social_nao_pronto[regiao] = (
                    f"fonte social ausente: {caminho}")
                continue
            try:
                unidades = carregar_unidades_sociais_json(caminho)
            except Exception as e:
                self.motivo_social_nao_pronto[regiao] = (
                    f"{type(e).__name__}: {e}")
                continue
            if not unidades:
                self.motivo_social_nao_pronto[regiao] = (
                    f"fonte social vazia: {caminho}")
                continue
            self.unidades_sociais[regiao] = unidades
            self.motivo_social_nao_pronto.pop(regiao, None)


servico = Servico()


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    servico.carregar()
    yield


app = FastAPI(title=TITULO, version=VERSAO_CONTRATO, lifespan=ciclo_de_vida)

# O mesmo serviço expõe o motor integrado de pesquisa. Os portões e a
# maturidade das rotas legadas continuam decididos pelo SVC-02.
sys.path.insert(0, str(REPO / "scripts/ia"))
from servico_ia_integrada import criar_router as _router_ia_integrada
app.include_router(_router_ia_integrada())

# Medidas sociais por polígonos oficiais, com três contratos separados.
# A consulta é independente da disponibilidade do escore físico.
from social07_servico_medidas import criar_router as _router_medidas_sociais
app.include_router(_router_medidas_sociais())

# Resposta integrada v1: orquestra as rotas acima numa jornada so, com uma autoridade fisica declarada.
from resposta_integrada import criar_router as _router_resposta_integrada
app.include_router(_router_resposta_integrada())


def _exigir_pronto() -> None:
    if not servico.pronto:
        raise HTTPException(status_code=503, detail=servico.motivo_nao_pronto)


def _unidades_sociais(regiao: str) -> tuple[str, list]:
    normalizada = regiao.strip().lower()
    unidades = servico.unidades_sociais.get(normalizada)
    if not unidades:
        motivo = servico.motivo_social_nao_pronto.get(
            normalizada, "regiao social nao carregada")
        raise HTTPException(status_code=503, detail={
            "contrato": VERSAO_CONTRATO_SOCIAL,
            "regiao": normalizada,
            "motivo": motivo,
            "nao_e": "baixa vulnerabilidade social",
        })
    return normalizada, unidades


# ------------------------------------------------------------------- rotas

@app.get("/saude")
def saude() -> JSONResponse:
    """Responde 503 enquanto nao houver artefato servido.

    A guarda que o `svc02.main()` ja tinha, mas que so aparecia no terminal.
    """
    corpo: dict[str, Any] = {
        "pronto": servico.pronto,
        "contrato": VERSAO_CONTRATO,
        "modelos_carregados": sorted(servico.modelos),
        "regioes_com_grade": sorted(servico.grades),
        "celulas_por_regiao": {r: len(g) for r, g in servico.grades.items()},
    }
    if not servico.pronto:
        corpo["motivo"] = servico.motivo_nao_pronto
        return JSONResponse(status_code=503, content=corpo)
    return JSONResponse(status_code=200, content=corpo)


@app.get("/regioes")
def rota_regioes() -> dict:
    """Cobertura e maturidade, sem alterar nada do que o SVC-02 devolve."""
    _exigir_pronto()
    return regioes(servico.modelos, servico.com_referencia_local)


@app.get("/social/saude")
def rota_social_saude() -> dict:
    regioes_sociais = {}
    for regiao, caminho in UNIDADES_SOCIAIS_OFICIAIS.items():
        unidades = servico.unidades_sociais.get(regiao) or []
        regioes_sociais[regiao] = {
            "pronto": bool(unidades),
            "unidades_carregadas": len(unidades),
            "fonte": str(caminho),
            "motivo": servico.motivo_social_nao_pronto.get(regiao),
        }
    return {
        "contrato": VERSAO_CONTRATO_SOCIAL,
        "regioes": regioes_sociais,
        "nao_e": "estado de prontidao do modelo fisico",
    }


@app.post("/social/aoi")
def rota_social_aoi(requisicao: RequisicaoSocialAOI) -> dict:
    regiao, unidades = _unidades_sociais(requisicao.regiao)
    resposta = avaliar_aoi_social(
        requisicao.aoi.model_dump(),
        unidades,
        requisicao.fisica,
        cobertura_minima=requisicao.cobertura_minima,
    )
    saida = {**resposta, "regiao_social": regiao}
    # Campos novos têm seu próprio contrato: não reclassificam os eixos v0.
    if regiao == "curitiba":
        from social07_servico_medidas import consultar
        try:
            saida["medidas_territoriais"] = consultar(regiao, requisicao.aoi.model_dump())
        except (ValueError, FileNotFoundError, KeyError) as e:
            saida["medidas_territoriais"] = {"contrato": "social_medidas_v1",
                "estado": "not_testable", "motivo": str(e),
                "indice_validado": None, "desigualdade_confirmada": False}
    return saida


@app.post("/social/consulta")
def rota_social_consulta(requisicao: RequisicaoSocialConsulta) -> dict:
    base = RequisicaoSocialAOI(**{
        k: v for k, v in requisicao.model_dump(exclude_none=True).items()
        if k != "pergunta"})
    resposta = rota_social_aoi(base)
    try:
        consulta = responder(requisicao.pergunta, resposta, modo="social")
    except Exception as e:
        return {**resposta, "consulta": {
            "pergunta": requisicao.pergunta,
            "modo": "social",
            "sintese": None,
            "erro": f"{type(e).__name__}: {e}",
            "por_que_null": ("a camada cognitiva social falhou; a prioridade "
                             "publica e os eixos sociais acima nao dependem "
                             "dela")}}
    return {**resposta, "consulta": {
        k: consulta[k] for k in
        ("pergunta", "modo", "autoridade", "classe_de_autoridade_consultada",
         "sintese", "auditoria_sintese", "fontes", "proveniencia",
         "descartados_por_injecao", "por_que_null", "nao_e")}}


@app.post("/inferencia")
def rota_inferencia(requisicao: RequisicaoInferencia, http: Request) -> dict:
    """Chama `inferir()` e devolve o resultado. Nao decide nada."""
    _exigir_pronto()
    bruta = requisicao.model_dump(exclude_none=True)

    n = len(bruta["geometria"]["pontos"])
    if n > MAX_PONTOS:
        raise HTTPException(
            status_code=422,
            detail=(f"{n} pontos acima do limite de {MAX_PONTOS}: o IC projeta "
                    "1000 replicas por ponto, e a requisicao seria recusada "
                    "depois de alocar a matriz inteira"))

    # A regiao precisa ser resolvida ANTES de preencher camadas, porque cada
    # regiao tem grade e modelo proprios. Aqui isso e so consulta: quem decide
    # de verdade continua sendo o portao G2/G3 dentro de `inferir()`.
    regiao = (bruta.get("regiao") or "").strip().lower()
    if not regiao:
        regiao = _deduzir_regiao(bruta["geometria"]["pontos"])
    modelo = modelo_da_regiao(regiao, servico.modelos) if regiao else None

    procedencia = None
    if modelo is not None:
        bruta, procedencia = resolver(bruta, modelo, servico.grades, regiao)
        # A ingestao so entra DEPOIS de o resolvedor dizer que o ponto esta
        # fora de toda celula servivel -- e a fronteira que o `resolver_camadas`
        # desenhou de proposito. Sem isto, uma requisicao dentro da cobertura
        # pagaria download por nada.
        if (requisicao.ingestao_sob_demanda
                and procedencia["pontos_fora_de_toda_celula_servivel"] > 0):
            periodo = bruta.get("periodo") or {}
            data_referencia = bruta.get("data_alvo") or periodo.get("fim")
            plano = adquirir(bruta["geometria"]["pontos"],
                             data_referencia, online=True)
            procedencia["ingestao_sob_demanda"] = {
                k: v for k, v in plano.items() if k != "camadas"}
            if plano.get("executado"):
                bruta["evidencia_temporal_entrada"] = {
                    "executado": True,
                    "origem": "ingestao_sob_demanda",
                    "fonte": (plano.get("chuva") or {}).get("fonte"),
                    "janela_chuva": (plano.get("chuva") or {}).get("janela"),
                }
            for ponto, camadas in zip(bruta["geometria"]["pontos"],
                                      plano.get("camadas") or []):
                for f in modelo["features"]:
                    if ponto["camadas"].get(f) is None and camadas.get(f) is not None:
                        ponto["camadas"][f] = camadas[f]

    resposta = inferir(bruta, servico.modelos, servico.caixas,
                       servico.com_referencia_local)
    if procedencia is not None:
        resposta = {**resposta, "procedencia_das_camadas": procedencia}

    periodo_foi_usado = bool(
        requisicao.ingestao_sob_demanda
        and procedencia
        and (procedencia.get("ingestao_sob_demanda") or {}).get("executado"))
    if (requisicao.periodo is not None and resposta.get("status") == "ok"
            and not periodo_foi_usado):
        resposta = {**resposta, "campos_ignorados": {
            "periodo": ("declarado na requisição e não usado: as camadas vêm "
                        "da grade, que não é datada, e a chuva entra por "
                        "`cenario_de_chuva`, que é percentil da distribuição "
                        "observada e não uma data. O escore acima NÃO é "
                        "específico do período pedido")}}
    elif requisicao.periodo is not None and periodo_foi_usado:
        resposta = {**resposta, "campos_utilizados": {
            "periodo": ("`periodo.fim` foi usado somente como data de "
                        "referência da janela de chuva na ingestão sob demanda. "
                        "Isso condiciona a entrada temporal, mas NÃO demonstra "
                        "validade temporal do modelo nem transforma o retorno "
                        "em previsão para o período inteiro")}}

    resposta = _sintetizar(resposta, pedida=requisicao.sintese)
    resposta = _anexar_proveniencia(resposta)

    if servico.log is not None:
        servico.log.registrar(requisicao.model_dump(exclude_none=True),
                              resposta, cliente=_cliente(http))
    return resposta


def _sintetizar(resposta: dict, pedida: bool = True) -> dict:
    """Anexa `sintese` e `auditoria_sintese`, sem poder derrubar a inferencia.

    O `try` nao e decoracao. O escore, os portoes e o model card ja estao
    prontos quando esta funcao roda: deixar uma falha do gerador de texto
    transformar uma resposta valida em 500 seria trocar o que importa pelo que
    e acessorio. Falhou, sai sem sintese -- e o cartao mostra a frase por regra,
    que nunca dependeu disto.

    `pedida=False` e a consequencia pratica desse mesmo raciocinio. Quando
    esta rota foi unificada com `/explicacao`, o argumento foi que gerar o
    texto "custa microssegundos" -- e custava, com o motor deterministico.
    Com `LOCAL_LLM_URL` apontando para o motor local, a mesma chamada passou a
    custar cerca de 30 segundos, medidos, e esse custo e CONSTANTE: nao muda
    entre 1 e 81 pontos, porque nao e calculo, e espera. A inferencia
    cientifica em si mede 0,000 s.
    Fazer quem so quer saber se a area e suscetivel esperar meio minuto por
    prosa inverte a prioridade do produto. Com `sintese: false` a resposta
    cientifica sai na hora, e quem quiser o texto pede numa segunda chamada.
    """
    if not pedida:
        return {**resposta, "sintese": None,
                "auditoria_sintese": {
                    "motor": None, "e_llm": False, "aprovadas": 0,
                    "reprovadas": 0, "frases": [],
                    "por_que_null": ("a síntese não foi pedida nesta "
                                     "requisição (`sintese: false`); o escore, "
                                     "os portões e o model card acima não "
                                     "dependem dela")}}
    if not SINTESE_EM_INFERENCIA:
        return resposta
    try:
        return sintetizar(resposta, evidencia=False)
    except Exception as e:                       # nunca derruba a inferencia
        return {**resposta, "sintese": None,
                "auditoria_sintese": {
                    "motor": None, "e_llm": False, "aprovadas": 0,
                    "reprovadas": 0, "frases": [],
                    "erro_do_motor": f"{type(e).__name__}: {e}",
                    "por_que_null": ("a camada de sintese falhou; o escore e os "
                                     "portoes acima nao dependem dela")}}


def _anexar_proveniencia(resposta: dict) -> dict:
    """Anexa o bloco `proveniencia`, e nunca derruba a inferencia por causa dele.

    Mesmo racional do `try` em `_sintetizar`: o escore, os portoes e o model
    card ja estao prontos quando esta funcao roda.
    """
    if not PROVENIENCIA_EM_INFERENCIA:
        return resposta
    try:
        return {**resposta,
                "proveniencia": bloco_de_proveniencia(
                    resposta, modelos_dir=MODELOS_SERVIDOS)}
    except Exception as e:                        # nunca derruba a inferencia
        return {**resposta, "proveniencia": {
            "erro": f"{type(e).__name__}: {e}",
            "nao_e": ("bloco descritivo; nenhum campo cientifico da resposta "
                      "depende dele")}}


def _deduzir_regiao(pontos: list[dict]) -> str:
    """Pergunta ao PORTAO, e nao a uma copia dele.

    Esta funcao repetia a logica antiga do G2 -- a primeira caixa que casa --,
    e ela so escolhe a camada e o modelo; quem decide e `inferir()`. Enquanto
    nenhuma caixa se sobrepunha as duas copias concordavam por acidente. Com as
    macrorregioes (Curitiba dentro do Sul, caixas do Centro-Oeste e do Nordeste
    cruzadas em Tocantins) elas passariam a divergir: o transporte preencheria
    camadas com a grade de uma regiao e o contrato julgaria a requisicao como
    outra. Delegar e o que garante que as duas respostas sao a mesma.
    """
    from svc02_contrato_inferencia import g2_regiao
    ok, _, nome = g2_regiao({}, pontos, servico.caixas)
    return nome if ok else ""


def _cliente(http: Request) -> str | None:
    return http.client.host if http.client else None


@app.post("/explicacao")
def rota_explicacao(requisicao: RequisicaoInferencia, http: Request) -> dict:
    """A mesma resposta de `/inferencia`, com `evidencia_visual` anexada.

    O QUE MUDOU, e por que a rota continua existindo:

    `sintese` e `auditoria_sintese` passaram a sair tambem em `/inferencia` --
    com o motor padrao, deterministico, gerar o texto custa microssegundos, e
    separar as rotas por esse custo deixou de se pagar. O que ainda distingue
    esta rota e `evidencia_visual`, que consulta o corpus de vetores e e a
    parte cara.

    A sintese NAO e gerada de novo aqui: `rota_inferencia` ja passou pelo
    verificador, e chamar o motor duas vezes na mesma requisicao poderia
    devolver dois textos diferentes para a mesma resposta -- que e o defeito
    que o verificador existe para impedir, reintroduzido pelo transporte.

    O campo `explicacao` original continua intacto nas duas rotas -- ver
    `docs/CONTRATO_VERSIONAMENTO.md`.
    """
    _exigir_pronto()
    resposta = rota_inferencia(requisicao, http)
    return anexar_evidencia_visual(resposta)


class RequisicaoConsulta(RequisicaoInferencia):
    """A mesma requisicao de inferencia, com uma pergunta em cima dela.

    Herda em vez de duplicar porque a consulta NAO e uma rota paralela: ela
    responde sobre uma inferencia, e portanto precisa da inferencia inteira.
    Aceitar so a pergunta abriria uma porta para responder sem payload -- que
    e responder sem autoridade.
    """

    pergunta: str = Field(..., min_length=3, max_length=500,
                          description="pergunta sobre a resposta ja calculada")
    modo: Literal["inferencia", "cientifico", "proveniencia"] | None = Field(
        None, description="autoridade consultada; omitido usa roteamento "
                          "deterministico por termo")


@app.post("/consulta")
def rota_consulta(requisicao: RequisicaoConsulta, http: Request) -> dict:
    """Pergunta sobre a resposta, dentro da autoridade do modo escolhido.

    ESTA ROTA NAO CALCULA NADA. Ela chama `/inferencia` internamente, pega o
    payload ja decidido e passa para a camada cognitiva. O escore, os portoes
    e o model card sao os mesmos byte a byte -- e existe um teste de
    equivalencia que falha se deixarem de ser (`test_fronteira_arquitetural`).

    O que muda entre modos e de onde a explicacao pode tirar apoio, nunca o
    que o contrato decidiu.
    """
    _exigir_pronto()
    base = RequisicaoInferencia(**{
        k: v for k, v in requisicao.model_dump(exclude_none=True).items()
        if k not in ("pergunta", "modo")})
    resposta = rota_inferencia(base, http)
    try:
        consulta = responder(requisicao.pergunta, resposta, modo=requisicao.modo)
    except Exception as e:                        # nunca derruba a inferencia
        return {**resposta, "consulta": {
            "pergunta": requisicao.pergunta, "modo": requisicao.modo,
            "sintese": None,
            "erro": f"{type(e).__name__}: {e}",
            "por_que_null": ("a camada cognitiva falhou; o escore e os "
                             "portoes acima nao dependem dela")}}
    # Os campos cientificos vem do payload; a consulta entra num bloco proprio
    # para que nenhum deles possa ser sobrescrito por acidente de merge.
    return {**resposta, "consulta": {
        k: consulta[k] for k in
        ("pergunta", "modo", "autoridade", "classe_de_autoridade_consultada",
         "sintese", "auditoria_sintese", "fontes", "proveniencia",
         "descartados_por_injecao", "por_que_null", "nao_e")}}


@app.get("/")
def raiz() -> dict:
    return {
        "servico": TITULO,
        "contrato": VERSAO_CONTRATO,
        "rotas": {
            "POST /inferencia": ("escore da area, com portoes, IC, model card "
                                 "e a sintese ja verificada"),
            "POST /social/aoi": ("prioridade publica e vulnerabilidade social "
                                 "por AOI, sem alterar o escore fisico"),
            "POST /social/consulta": ("pergunta em linguagem natural sobre a "
                                      "resposta social ja calculada"),
            "POST /explicacao": "a mesma resposta, com evidencia_visual anexada",
            "POST /consulta": ("pergunta em linguagem natural sobre a resposta, "
                               "respondida dentro da autoridade do modo"),
            "GET /regioes": "cobertura e maturidade por regiao",
            "GET /social/saude": "fontes sociais oficiais carregadas por regiao",
            "POST /social/medidas/consulta": "exposição, componentes de vulnerabilidade e oportunidade de observação com contratos separados",
            "GET /social/medidas/saude": "integridade dos estudos sociais registrados; não declara validação científica",
            "POST /integrada/v1/consulta": "jornada unica AOI -> autoridade fisica declarada -> historico, SAR, suporte, explicacao, fontes, social, cartao e relatorio",
            "GET /saude": "503 enquanto nao houver artefato servido",
            "GET /docs": "esquema interativo gerado pelo FastAPI",
        },
        "maturidades": AUTORIZA_POR_MATURIDADE,
        "nao_e": ("nao e previsao de evento nem mapa operacional: e escore de "
                  "predisposicao do terreno, com limitacoes declaradas por "
                  "regiao na propria resposta"),
    }


def main() -> int:
    import uvicorn

    from log_de_servico import redirecionar_se_pedido

    # `--log` existe para quando este processo sobe como servico, sem terminal
    # onde alguem leia a saida. Nao muda nada quando ele roda a mao.
    redirecionar_se_pedido()

    porta = 8123
    if "--porta" in sys.argv:
        porta = int(sys.argv[sys.argv.index("--porta") + 1])
    uvicorn.run(app, host="127.0.0.1", port=porta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
