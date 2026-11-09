"""Camada social territorial do REV-P.

Esta camada responde a uma pergunta diferente da inferencia fisico-hidrologica:
quem pode ser mais afetado, onde a acao publica deveria ser priorizada e quais
lacunas de observacao precisam ser declaradas.

Ela nao calcula enchente, nao treina modelo e nao altera escore fisico. O
contrato e review-only nesta fase porque as fontes oficiais ainda precisam ser
adquiridas e versionadas.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "local_runs" / "camada-social-territorial"
VERSAO_CONTRATO_SOCIAL = "social_v0_review_only"

DOMINIOS_SOCIAIS = (
    "exposicao_populacional",
    "condicao_socioeconomica",
    "composicao_demografica",
    "raca_genero",
    "moradia_infraestrutura",
    "mobilidade_acesso",
    "lacuna_institucional",
)

# -------------------------------------------------------------------------
# HIERARQUIA DA AUSENCIA SOCIAL
# -------------------------------------------------------------------------
#
# O projeto ja trata a ausencia como informacao no eixo dos eventos: o
# `nivel_negativo` do `ds03_esquema_alvo` existe justamente porque "ausencia
# nao constitui registro de ausencia". A camada social precisava do mesmo
# eixo, e nao tinha: toda falta de dado saia como um unico
# `nao_identificavel`.
#
# O motivo de isso importar aqui e o proprio objeto de estudo. Um territorio
# pode nao ter dado porque a fonte nao alcanca aquele tipo de lugar -- e essa
# ausencia NAO e neutra: ela e o efeito, medido, da mesma exclusao que a
# camada social existe para descrever. Colapsa-la com "ainda nao consultamos"
# apaga a distincao mais importante do eixo humano.
#
# Nenhum destes estados e baixa vulnerabilidade. A hierarquia serve para dizer
# QUE ACAO a ausencia justifica, nunca para relaxar a leitura.
AUSENCIA_FONTE_INEXISTENTE = "fonte_inexistente_para_o_territorio"
AUSENCIA_FONTE_NAO_ALCANCA = "fonte_existe_e_nao_alcanca_a_aoi"
AUSENCIA_COBERTURA_PARCIAL = "cobertura_parcial_abaixo_do_minimo"
AUSENCIA_DOMINIO_NAO_MEDIDO = "dominio_ausente_na_fonte_que_cobre"
SEM_AUSENCIA = "cobertura_suficiente"

NIVEIS_AUSENCIA_SOCIAL = (
    SEM_AUSENCIA,
    AUSENCIA_DOMINIO_NAO_MEDIDO,
    AUSENCIA_COBERTURA_PARCIAL,
    AUSENCIA_FONTE_NAO_ALCANCA,
    AUSENCIA_FONTE_INEXISTENTE,
)

ACAO_POR_AUSENCIA = {
    SEM_AUSENCIA: "nenhuma coleta social adicional e exigida para responder",
    AUSENCIA_DOMINIO_NAO_MEDIDO: (
        "a fonte cobre o territorio e nao mede este dominio; ampliar o "
        "conjunto de indicadores, nao o recorte"),
    AUSENCIA_COBERTURA_PARCIAL: (
        "ha unidades sociais na AOI abaixo do minimo de cobertura; o que foi "
        "medido segue valendo como indicio e nao como agregado"),
    AUSENCIA_FONTE_NAO_ALCANCA: (
        "existe fonte social na regiao e ela nao alcanca esta AOI; verificar "
        "recorte, malha e atualizacao antes de concluir qualquer coisa"),
    AUSENCIA_FONTE_INEXISTENTE: (
        "nenhuma fonte social carregada cobre este tipo de territorio; a "
        "ausencia e do inventario publico, e ela mesma e um achado"),
}

FONTES_CANDIDATAS = [
    {
        "fonte_id": "ibge_censo_2022_setores",
        "nome": "IBGE Censo 2022 - agregados por setores censitarios",
        "url": "https://www.ibge.gov.br/estatisticas/sociais/trabalho/22827-censo-demografico-2022.html",
        "escala": "setor censitario",
        "dominios": "condicao_socioeconomica;composicao_demografica;raca_genero",
        "uso_admitido": "contexto agregado e prioridade territorial",
        "nao_e": "causa de enchente ou rotulo do modelo fisico",
        "status_no_revp": "candidata_nao_adquirida",
    },
    {
        "fonte_id": "ibge_favelas_comunidades_urbanas_2022",
        "nome": "IBGE Censo 2022 - Favelas e Comunidades Urbanas",
        "url": "https://www.ibge.gov.br/geociencias/organizacao-do-territorio/tipologias-do-territorio/15788-favelas-e-comunidades-urbanas.html",
        "escala": "tipologia territorial oficial do IBGE",
        "dominios": "moradia_infraestrutura;lacuna_institucional",
        "uso_admitido": (
            "indicador agregado de vulnerabilidade territorial e contexto de "
            "precariedade urbanistica"),
        "nao_e": "rotulo direto de usuario, identidade territorial ou confirmacao de alagamento",
        "status_no_revp": "candidata_nao_adquirida",
    },
    {
        "fonte_id": "ibge_entorno_domicilios_2022",
        "nome": "IBGE - caracteristicas urbanisticas do entorno dos domicilios",
        "url": "https://www.ibge.gov.br/geociencias/organizacao-do-territorio/tipologias-do-territorio/24702-caracteristicas-urbanisticas-do-entorno-dos-domicilios.html",
        "escala": "face/setor conforme produto publicado",
        "dominios": "moradia_infraestrutura;mobilidade_acesso",
        "uso_admitido": "infraestrutura urbana que agrava impacto e resposta",
        "nao_e": "variavel hidrologica do escore",
        "status_no_revp": "candidata_nao_adquirida",
    },
    {
        "fonte_id": "ibge_cemaden_bater_2018",
        "nome": "IBGE/CEMADEN - populacao em areas de risco no Brasil",
        "url": "https://www.ibge.gov.br/geociencias/informacoes-ambientais/estudos-ambientais/21538-populacao-em-areas-de-risco-no-brasil.html",
        "escala": "area de risco monitorada por municipio",
        "dominios": "exposicao;lacuna_institucional",
        "uso_admitido": "referencia institucional de exposicao a risco",
        "nao_e": "validacao local automatica do REV-P",
        "status_no_revp": "referencia_metodologica_nao_integrada",
    },
    {
        "fonte_id": "ministerio_cidades_periferia_viva",
        "nome": "Ministerio das Cidades - Periferia Viva / Periferia Sem Risco",
        "url": "https://www.gov.br/cidades/pt-br/novo-pac-selecoes-2026/periferia-viva-urbanizacao-de-favelas/",
        "escala": "poligono de intervencao KML/KMZ quando publicado pelo proponente",
        "dominios": "moradia_infraestrutura;lacuna_institucional;mobilidade_acesso",
        "uso_admitido": "ponte com politica publica, intervencao e planejamento",
        "nao_e": "fonte de treino do modelo fisico",
        "status_no_revp": "referencia_de_enquadramento",
    },
]


@dataclass(frozen=True)
class BBox:
    """Retangulo em EPSG:4326 usado no experimento v0.

    A versao operacional deve trocar isto por geometria real. O retangulo e
    suficiente para testar o contrato, os invariantes e a matriz de prioridade.
    """

    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

    def area(self) -> float:
        largura = max(0.0, self.lon_max - self.lon_min)
        altura = max(0.0, self.lat_max - self.lat_min)
        return largura * altura

    def overlap_area(self, other: "BBox") -> float:
        lon_min = max(self.lon_min, other.lon_min)
        lon_max = min(self.lon_max, other.lon_max)
        lat_min = max(self.lat_min, other.lat_min)
        lat_max = min(self.lat_max, other.lat_max)
        return BBox(lat_min, lon_min, lat_max, lon_max).area()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BBox":
        return cls(
            lat_min=float(data["lat_min"]),
            lon_min=float(data["lon_min"]),
            lat_max=float(data["lat_max"]),
            lon_max=float(data["lon_max"]),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "lat_min": self.lat_min,
            "lon_min": self.lon_min,
            "lat_max": self.lat_max,
            "lon_max": self.lon_max,
        }


@dataclass(frozen=True)
class UnidadeSocial:
    unidade_id: str
    nome: str
    geometria: BBox
    indicadores: dict[str, float]
    fonte: str
    observacao: str = ""

    def normalizados(self) -> dict[str, float]:
        saida = {}
        for dominio, valor in self.indicadores.items():
            if dominio not in DOMINIOS_SOCIAIS:
                continue
            saida[dominio] = max(0.0, min(1.0, float(valor)))
        return saida


def inventario_fontes() -> list[dict[str, str]]:
    """Fontes candidatas, sem alegar que ja foram adquiridas."""

    return list(FONTES_CANDIDATAS)


def unidades_demo() -> list[UnidadeSocial]:
    """Casos sinteticos declarados para testar contrato antes de dado oficial."""

    return [
        UnidadeSocial(
            unidade_id="demo_periferia_alta_vulnerabilidade",
            nome="territorio vulneravel demonstrativo",
            geometria=BBox(-8.080, -34.930, -8.040, -34.890),
            indicadores={
                "exposicao_populacional": 0.86,
                "condicao_socioeconomica": 0.88,
                "composicao_demografica": 0.72,
                "raca_genero": 0.81,
                "moradia_infraestrutura": 0.91,
                "mobilidade_acesso": 0.76,
                "lacuna_institucional": 0.84,
            },
            fonte="sintetico_para_teste_de_contrato",
            observacao="nao representa setor censitario real",
        ),
        UnidadeSocial(
            unidade_id="demo_area_baixa_vulnerabilidade",
            nome="territorio menos vulneravel demonstrativo",
            geometria=BBox(-8.120, -34.890, -8.085, -34.850),
            indicadores={
                "exposicao_populacional": 0.22,
                "condicao_socioeconomica": 0.18,
                "composicao_demografica": 0.24,
                "raca_genero": 0.21,
                "moradia_infraestrutura": 0.16,
                "mobilidade_acesso": 0.20,
                "lacuna_institucional": 0.28,
            },
            fonte="sintetico_para_teste_de_contrato",
            observacao="nao representa setor censitario real",
        ),
        UnidadeSocial(
            unidade_id="demo_infraestrutura_intermediaria",
            nome="territorio intermediario demonstrativo",
            geometria=BBox(-8.070, -34.875, -8.035, -34.835),
            indicadores={
                "exposicao_populacional": 0.57,
                "condicao_socioeconomica": 0.50,
                "composicao_demografica": 0.45,
                "raca_genero": 0.48,
                "moradia_infraestrutura": 0.62,
                "mobilidade_acesso": 0.55,
                "lacuna_institucional": 0.58,
            },
            fonte="sintetico_para_teste_de_contrato",
            observacao="nao representa setor censitario real",
        ),
    ]


def carregar_unidades_sociais_json(caminho: Path) -> list[UnidadeSocial]:
    """Carrega unidades sociais geradas por pipelines de aquisicao oficiais."""

    data = json.loads(caminho.read_text(encoding="utf-8"))
    unidades = data.get("unidades_sociais", data if isinstance(data, list) else [])
    saida = []
    for item in unidades:
        saida.append(UnidadeSocial(
            unidade_id=str(item["unidade_id"]),
            nome=str(item.get("nome") or item["unidade_id"]),
            geometria=BBox.from_dict(item["geometria"]),
            indicadores={k: float(v) for k, v in
                         (item.get("indicadores") or {}).items()
                         if v is not None},
            fonte=str(item.get("fonte") or "fonte_nao_declarada"),
            observacao=str(item.get("observacao") or ""),
        ))
    return saida


def classe_relativa(valor: float | None) -> str:
    if valor is None:
        return "nao_identificavel"
    if valor >= 0.67:
        return "alta"
    if valor >= 0.34:
        return "media"
    return "baixa"


def classe_fisica(escore_fisico: float | None, status_fisico: str = "ok") -> str:
    if status_fisico != "ok" or escore_fisico is None:
        return "nao_identificavel"
    return classe_relativa(float(escore_fisico))


def cobertura_social(aoi: BBox, unidades: list[UnidadeSocial]) -> list[dict[str, Any]]:
    area_aoi = aoi.area()
    if area_aoi <= 0:
        return []
    partes = []
    for unidade in unidades:
        area = aoi.overlap_area(unidade.geometria)
        if area <= 0:
            continue
        partes.append({
            "unidade_id": unidade.unidade_id,
            "nome": unidade.nome,
            "fracao_da_aoi": area / area_aoi,
            "indicadores": unidade.normalizados(),
            "fonte": unidade.fonte,
            "observacao": unidade.observacao,
        })
    return partes


def agregar_indicadores(partes: list[dict[str, Any]]) -> tuple[dict[str, float], float]:
    cobertura = sum(float(p["fracao_da_aoi"]) for p in partes)
    if cobertura <= 0:
        return {}, 0.0
    agregado: dict[str, float] = {}
    for dominio in DOMINIOS_SOCIAIS:
        numerador = 0.0
        peso = 0.0
        for parte in partes:
            indicadores = parte["indicadores"]
            if dominio not in indicadores:
                continue
            w = float(parte["fracao_da_aoi"])
            numerador += w * float(indicadores[dominio])
            peso += w
        if peso > 0:
            agregado[dominio] = round(numerador / peso, 4)
    return agregado, round(min(cobertura, 1.0), 4)


def nivel_de_ausencia_social(partes: list[dict[str, Any]],
                             indicadores: dict[str, float],
                             fracao: float,
                             cobertura_minima: float,
                             unidades: list[UnidadeSocial]) -> str:
    """Por que o dado social faltou -- e nao apenas que faltou.

    A ordem das perguntas vai do mais especifico ao mais grave, porque a acao
    publica que cada resposta justifica e diferente: ampliar indicadores,
    revisar recorte, ou registrar que o inventario publico nao alcanca aquele
    tipo de territorio.
    """
    if not unidades:
        return AUSENCIA_FONTE_INEXISTENTE
    if not partes:
        return AUSENCIA_FONTE_NAO_ALCANCA
    if fracao < cobertura_minima or not indicadores:
        return AUSENCIA_COBERTURA_PARCIAL
    if len(indicadores) < len(DOMINIOS_SOCIAIS):
        return AUSENCIA_DOMINIO_NAO_MEDIDO
    return SEM_AUSENCIA


def prioridade_publica(classe_susc: str, classe_social: str,
                       evidencia: str) -> tuple[str, str]:
    """Matriz qualitativa: prioridade nao e novo escore fisico."""

    if classe_social == "nao_identificavel":
        # A fisica alta NAO pode ser apagada pela ausencia social. Antes desta
        # ramificacao, uma AOI com escore 0,82 e sem setor censitario
        # utilizavel recebia `prioridade_de_dado` -- exatamente a mesma classe
        # de uma AOI seca e sem dado. As duas ausencias sao iguais; as duas
        # areas nao sao, e a que some da fila e justamente a que combina
        # suscetibilidade alta com territorio nao inventariado.
        if classe_susc == "alta":
            return (
                "prioridade_tecnica_e_de_dado",
                "suscetibilidade fisica alta sem cobertura social: a acao "
                "tecnica ja se justifica, e nao se sabe quem esta exposto; "
                "coletar dado social nao adia a vistoria",
            )
        return (
            "prioridade_de_dado",
            "sem cobertura social suficiente; coletar ou integrar fonte oficial",
        )
    if classe_susc == "nao_identificavel":
        if classe_social == "alta":
            return (
                "prioridade_de_vistoria_e_coleta",
                "territorio vulneravel sem inferencia fisica suficiente; nao declarar risco baixo",
            )
        return (
            "prioridade_de_evidencia",
            "inferencia fisica ausente ou recusada; resposta publica deve pedir nova evidencia",
        )
    if classe_susc == "alta" and classe_social == "alta":
        return (
            "prioridade_publica_forte",
            "suscetibilidade fisica alta coincide com alta vulnerabilidade/exposicao",
        )
    if classe_susc == "alta":
        return (
            "prioridade_tecnica",
            "suscetibilidade fisica alta; acao publica segue relevante mesmo sem alta vulnerabilidade social",
        )
    if classe_social == "alta" and evidencia != "forte":
        return (
            "prioridade_de_observacao",
            "alta vulnerabilidade com evidencia fisica limitada; priorizar vistoria/coleta, nao previsao",
        )
    if classe_social == "alta":
        return (
            "prioridade_social_contextual",
            "vulnerabilidade alta pede comunicacao e preparo, sem alterar a leitura fisica",
        )
    return (
        "prioridade_rotineira",
        "sem coincidencia demonstrativa de alta suscetibilidade e alta vulnerabilidade",
    )


def avaliar_aoi_social(
    aoi: dict[str, Any],
    unidades: list[UnidadeSocial],
    inferencia_fisica: dict[str, Any] | None = None,
    cobertura_minima: float = 0.25,
) -> dict[str, Any]:
    """Avalia contexto social de uma AOI sem alterar o payload fisico."""

    caixa = BBox.from_dict(aoi)
    fisica = dict(inferencia_fisica or {})
    partes = cobertura_social(caixa, unidades)
    indicadores, fracao = agregar_indicadores(partes)
    media_social = None
    if indicadores:
        media_social = round(sum(indicadores.values()) / len(indicadores), 4)
    classe_social = classe_relativa(media_social)
    status_social = "ok" if fracao >= cobertura_minima and indicadores else "insufficient_social_data"
    nivel_ausencia = nivel_de_ausencia_social(
        partes, indicadores, fracao, cobertura_minima, unidades)

    status_fisico = fisica.get("status", "nao_informado")
    escore_fisico = fisica.get("escore")
    classe_susc = classe_fisica(escore_fisico, status_fisico)
    evidencia = fisica.get("maturidade") or "nao_declarada"
    prioridade, razao = prioridade_publica(classe_susc, classe_social, evidencia)
    if status_social != "ok":
        prioridade, razao = prioridade_publica(classe_susc, "nao_identificavel", evidencia)

    return {
        "contrato": VERSAO_CONTRATO_SOCIAL,
        "status": status_social,
        "unidade_de_resposta": "area",
        "aoi": caixa.as_dict(),
        "eixos": {
            "suscetibilidade_fisica": {
                "status": status_fisico,
                "escore_recebido": escore_fisico,
                "classe_relativa": classe_susc,
                "maturidade": fisica.get("maturidade"),
                "fonte": "payload_do_svc02_ou_equivalente",
                "alterado_pela_camada_social": False,
            },
            "vulnerabilidade_exposicao": {
                "status": status_social,
                "cobertura_da_aoi": fracao,
                "indicadores_agregados": indicadores,
                "media_relativa": media_social,
                "classe_relativa": classe_social if status_social == "ok" else "nao_identificavel",
                "fontes": sorted({p["fonte"] for p in partes}),
                "nao_e": "variavel causal do alagamento",
            },
            "lacuna_de_conhecimento_publico": {
                "status": "declarada" if status_social != "ok" else "avaliada",
                "lacunas": lacunas_sociais(indicadores, status_social),
                # POR QUE faltou, e nao so que faltou. Ver a hierarquia no topo
                # do modulo: a ausencia de fonte para um tipo de territorio e
                # ela mesma um achado sobre o inventario publico.
                "nivel_ausencia_social": nivel_ausencia,
                "acao_que_a_ausencia_justifica": ACAO_POR_AUSENCIA[nivel_ausencia],
                "nao_e": "baixo risco",
            },
        },
        "partes_intersectadas": partes,
        "prioridade_publica": {
            "classe": prioridade,
            "razao": razao,
            "nao_e": "previsao individual de evento",
        },
        "invariantes": {
            "nao_altera_escore_fisico": True,
            "nao_treina_modelo": True,
            "nao_cria_rotulo": True,
            "ausencia_social_nao_e_baixa_vulnerabilidade": True,
            "vulnerabilidade_nao_e_causa_hidrologica": True,
        },
    }


def lacunas_sociais(indicadores: dict[str, float], status_social: str) -> list[str]:
    lacunas = [d for d in DOMINIOS_SOCIAIS if d not in indicadores]
    if status_social != "ok":
        lacunas.append("cobertura_geografica_social_insuficiente")
    return lacunas


def casos_experimentais() -> list[dict[str, Any]]:
    return [
        {
            "caso": "alta_fisica_alta_social",
            "aoi": {"lat_min": -8.075, "lon_min": -34.925,
                    "lat_max": -8.050, "lon_max": -34.900},
            "fisica": {"status": "ok", "escore": 0.82, "maturidade": "mvp_local"},
        },
        {
            "caso": "alta_fisica_baixa_social",
            "aoi": {"lat_min": -8.115, "lon_min": -34.885,
                    "lat_max": -8.095, "lon_max": -34.860},
            "fisica": {"status": "ok", "escore": 0.82, "maturidade": "mvp_local"},
        },
        {
            "caso": "fisica_recusada_alta_social",
            "aoi": {"lat_min": -8.075, "lon_min": -34.925,
                    "lat_max": -8.050, "lon_max": -34.900},
            "fisica": {"status": "insufficient_data", "escore": None,
                       "maturidade": None, "gate_que_nao_fechou": "dominio_coberto"},
        },
        {
            "caso": "alta_fisica_sem_cobertura_social",
            "aoi": {"lat_min": -9.500, "lon_min": -36.000,
                    "lat_max": -9.450, "lon_max": -35.950},
            "fisica": {"status": "ok", "escore": 0.82,
                       "maturidade": "transferencia_sem_referencia_local"},
        },
        {
            "caso": "media_fisica_social_intermediaria",
            "aoi": {"lat_min": -8.065, "lon_min": -34.870,
                    "lat_max": -8.040, "lon_max": -34.845},
            "fisica": {"status": "ok", "escore": 0.51,
                       "maturidade": "transferencia_caracterizada"},
        },
    ]


def executar_experimentos() -> dict[str, Any]:
    unidades = unidades_demo()
    resultados = []
    for caso in casos_experimentais():
        r = avaliar_aoi_social(caso["aoi"], unidades, caso["fisica"])
        r["caso"] = caso["caso"]
        resultados.append(r)

    nao_colapso = experimento_nao_colapso(resultados)
    matriz = matriz_demonstrativa()
    return {
        # Sem `tipo` o coletor de conhecimento nao consegue absorver este
        # arquivo, e a camada social inteira ficava fora da memoria da IA --
        # ela media e o sistema esquecia.
        "tipo": "experimentos_camada_social",
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "contrato": VERSAO_CONTRATO_SOCIAL,
        "metodo": "intersecao territorial por bbox demonstrativo; sem dado oficial integrado",
        "fontes_candidatas": inventario_fontes(),
        "casos": resultados,
        "experimento_nao_colapso": nao_colapso,
        "matriz_prioridade": matriz,
        "limitacoes": [
            "unidades sociais demonstrativas nao representam setores censitarios reais",
            "geometria v0 usa bbox e deve ser substituida por intersecao de poligonos",
            "fontes oficiais ainda precisam de aquisicao, licenca, versao e schema",
            "prioridade publica nao e previsao individual nem probabilidade de evento",
        ],
        "nao_e": ("risco fisico: a camada social contextualiza e prioriza o "
                  "que o eixo fisico ja disse, e nunca fabrica suscetibilidade "
                  "onde o terreno nao a sustenta"),
        "proximos_experimentos": [
            "adquirir setores censitarios e tipologias territoriais oficiais por municipio piloto",
            "cruzar AOIs reais do SVC-03 com setores oficiais",
            "medir lacuna de registro comparando vulnerabilidade alta e densidade de chamados",
            "rodar pesquisa de usuario com modo gestor publico e modo publico",
        ],
    }


def experimento_nao_colapso(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    por_caso = {r["caso"]: r for r in resultados}
    a = por_caso["alta_fisica_alta_social"]
    b = por_caso["alta_fisica_baixa_social"]
    c = por_caso["fisica_recusada_alta_social"]
    escore_a = a["eixos"]["suscetibilidade_fisica"]["escore_recebido"]
    escore_b = b["eixos"]["suscetibilidade_fisica"]["escore_recebido"]
    return {
        "status": "PASS" if escore_a == escore_b and c["eixos"]["suscetibilidade_fisica"]["escore_recebido"] is None else "FAIL",
        "mesmo_escore_fisico_com_prioridades_diferentes": escore_a == escore_b
        and a["prioridade_publica"]["classe"] != b["prioridade_publica"]["classe"],
        "fisica_recusada_nao_vira_baixo_risco": (
            c["eixos"]["suscetibilidade_fisica"]["classe_relativa"] == "nao_identificavel"
            and "vistoria" in c["prioridade_publica"]["classe"]
        ),
        "conclusao": (
            "camada social altera prioridade e lacuna declarada, nunca escore fisico"
        ),
    }


def matriz_demonstrativa() -> list[dict[str, str]]:
    linhas = []
    for fisica in ("alta", "media", "baixa", "nao_identificavel"):
        for social in ("alta", "media", "baixa", "nao_identificavel"):
            prioridade, razao = prioridade_publica(fisica, social, "mvp_local")
            linhas.append({
                "classe_suscetibilidade_fisica": fisica,
                "classe_vulnerabilidade_social": social,
                "prioridade_publica": prioridade,
                "razao": razao,
            })
    return linhas


def escrever_csv(path: Path, rows: list[dict[str, Any]], campos: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in campos})


def escrever_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def achatar_caso(caso: dict[str, Any]) -> dict[str, Any]:
    fisica = caso["eixos"]["suscetibilidade_fisica"]
    social = caso["eixos"]["vulnerabilidade_exposicao"]
    lacuna = caso["eixos"]["lacuna_de_conhecimento_publico"]
    return {
        "caso": caso["caso"],
        "status_social": caso["status"],
        "status_fisico": fisica["status"],
        "escore_fisico_recebido": fisica["escore_recebido"],
        "classe_fisica": fisica["classe_relativa"],
        "classe_social": social["classe_relativa"],
        "media_social": social["media_relativa"],
        "cobertura_social_da_aoi": social["cobertura_da_aoi"],
        "prioridade_publica": caso["prioridade_publica"]["classe"],
        "razao": caso["prioridade_publica"]["razao"],
        "lacunas": ";".join(lacuna["lacunas"]),
    }


def relatorio_markdown(resultado: dict[str, Any]) -> str:
    linhas = [
        "# Camada social territorial - experimento v0",
        "",
        "Este experimento cria a primeira prova de contrato da camada social do REV-P.",
        "Ele nao valida vulnerabilidade social local; demonstra que a camada social",
        "pode ser mantida separada do escore fisico e usada para prioridade publica,",
        "lacuna de evidencia e comunicacao responsavel.",
        "",
        "## Resultado",
        "",
        f"- contrato: `{resultado['contrato']}`",
        f"- casos demonstrativos: {len(resultado['casos'])}",
        f"- nao-colapso: `{resultado['experimento_nao_colapso']['status']}`",
        "",
        "## Invariante principal",
        "",
        "A vulnerabilidade social nao aumenta nem reduz o escore fisico de enchente.",
        "Ela altera a prioridade publica, a linguagem e a necessidade de vistoria ou",
        "coleta de dado.",
        "",
        "## Casos",
        "",
        "| caso | fisica | social | prioridade | leitura |",
        "|---|---|---|---|---|",
    ]
    for caso in resultado["casos"]:
        r = achatar_caso(caso)
        linhas.append(
            f"| {r['caso']} | {r['classe_fisica']} | {r['classe_social']} | "
            f"{r['prioridade_publica']} | {r['razao']} |"
        )
    linhas += [
        "",
        "## Limitacoes",
        "",
    ]
    linhas += [f"- {x}" for x in resultado["limitacoes"]]
    linhas += [
        "",
        "## Proximos experimentos",
        "",
    ]
    linhas += [f"- {x}" for x in resultado["proximos_experimentos"]]
    linhas.append("")
    return "\n".join(linhas)


def gravar_resultados(resultado: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    escrever_json(output_dir / "experimentos_sociais_v0.json", resultado)
    escrever_csv(
        output_dir / "inventario_fontes_sociais_v0.csv",
        resultado["fontes_candidatas"],
        ["fonte_id", "nome", "url", "escala", "dominios",
         "uso_admitido", "nao_e", "status_no_revp"],
    )
    escrever_csv(
        output_dir / "casos_sociais_v0.csv",
        [achatar_caso(c) for c in resultado["casos"]],
        ["caso", "status_social", "status_fisico", "escore_fisico_recebido",
         "classe_fisica", "classe_social", "media_social",
         "cobertura_social_da_aoi", "prioridade_publica", "razao", "lacunas"],
    )
    escrever_csv(
        output_dir / "matriz_prioridade_social_v0.csv",
        resultado["matriz_prioridade"],
        ["classe_suscetibilidade_fisica", "classe_vulnerabilidade_social",
         "prioridade_publica", "razao"],
    )
    (output_dir / "relatorio_camadas_sociais_v0.md").write_text(
        relatorio_markdown(resultado), encoding="utf-8")
    (output_dir / "commands.txt").write_text(
        "python scripts/social/camada_social.py\n"
        "python -m pytest tests/test_camada_social.py -q\n",
        encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa experimentos v0 da camada social territorial.")
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--unidades-json", default=None,
                        help="JSON com unidades sociais oficiais ja integradas")
    parser.add_argument("--aoi-json", default=None,
                        help="JSON com bbox da AOI em EPSG:4326")
    parser.add_argument("--fisica-json", default=None,
                        help="JSON com resposta fisica ou resumo do svc02")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.unidades_json or args.aoi_json or args.fisica_json:
        if not (args.unidades_json and args.aoi_json and args.fisica_json):
            raise SystemExit(
                "--unidades-json, --aoi-json e --fisica-json devem ser usados juntos")
        unidades = carregar_unidades_sociais_json(Path(args.unidades_json))
        aoi = json.loads(Path(args.aoi_json).read_text(encoding="utf-8"))
        fisica = json.loads(Path(args.fisica_json).read_text(encoding="utf-8"))
        resposta = avaliar_aoi_social(aoi, unidades, fisica)
        output_dir.mkdir(parents=True, exist_ok=True)
        destino = output_dir / "resposta_social_aoi.json"
        destino.write_text(json.dumps(resposta, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        print("===CAMADA SOCIAL AOI===")
        print(f"status={resposta['status']}")
        print(f"prioridade={resposta['prioridade_publica']['classe']}")
        print(f"GRAVADO={destino}")
        print("===END===")
        return 0

    resultado = executar_experimentos()
    gravar_resultados(resultado, output_dir)
    print("===CAMADA SOCIAL TERRITORIAL===")
    print(f"contrato={resultado['contrato']}")
    print(f"casos={len(resultado['casos'])}")
    print(f"nao_colapso={resultado['experimento_nao_colapso']['status']}")
    print(f"GRAVADO={output_dir}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
