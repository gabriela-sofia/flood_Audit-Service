"""Etapa 4 -- a resposta do produto mostra a cadeia inteira, e onde ela parou.

-------------------------------------------------------------------------
ADITIVO, E NUNCA PORTAO
-------------------------------------------------------------------------

O contrato v1 ja tem os portoes que decidem: G1 a G5, mais o G6 quando ligado.
Este bloco NAO decide nada. Ele nasce como campo novo e opcional, pelo mesmo
caminho que `suporte_cientifico` seguiu -- um cliente da v1 que o ignore
continua lendo certo, e `status` sai exatamente igual com ou sem ele.

Se um dia a cadeia neural passar a recusar resposta, isso e mudanca de
comportamento para quem consome, e o `CONTRATO_VERSIONAMENTO.md` classifica
como v2. Ligar essa troca e decisao de versao, e nao efeito colateral de um
bloco informativo.

-------------------------------------------------------------------------
O ELO QUE NAO SE APLICA APARECE, E APARECE COMO NAO APLICAVEL
-------------------------------------------------------------------------

Uma requisicao de escore nao passa pela rede SAR, nao extrai embedding de
terreno e nao cita o corpus. Tres dos cinco elos nao se aplicam.

Omiti-los daria a impressao de cadeia curta e aprovada. Eles entram nomeados,
com o motivo -- porque ausencia de elo nao e elo que passou, exatamente como
ausencia de registro nao e registro de ausencia.

-------------------------------------------------------------------------
O ELO QUE SE APLICA E O FORCAMENTO
-------------------------------------------------------------------------

O efeito da chuva foi estimado sobre uma faixa de valores observados, e fora
dela o modelo nao mediu nada. A grade ja se abstem por isso -- 11.385 celulas
de Recife ficam brancas no cenario extremo --, e aqui a cadeia se abstem pela
mesma razao e com o mesmo endereco.

O dominio vem do proprio modelo servido (`faixa_dominio_5_95`), e nunca de um
numero escrito aqui.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "neural"))

from cadeia_seletiva import (  # noqa: E402
    ELOS, FORCAMENTO, OBSERVACAO, RECUPERACAO, REPRESENTACAO, VERIFICACAO,
    executar_cadeia, portao_forcamento,
)

# Por que cada elo fica de fora de uma requisicao de escore.
NAO_APLICAVEIS = {
    OBSERVACAO: ("a requisicao pede escore de area e nao processa cena SAR; "
                 "observacao de agua entra pela memoria, nunca pelo escore"),
    REPRESENTACAO: ("nenhuma saida do encoder de terreno esta em "
                    "`variaveis_usadas`: entrar exigiria orcamento de EPV e a "
                    "trava de sinal fisico"),
    RECUPERACAO: ("nada e citado do corpus nesta rota; a cadeia de citacao "
                  "pertence a explicacao textual, fora do escore"),
    VERIFICACAO: ("sem citacao nao ha o que verificar; o verificador de claims "
                  "continua obrigatorio onde ha frase publicada"),
}


def _variavel_de_forcamento(modelo: dict) -> str | None:
    """A variavel de chuva que o MODELO usa, e nao uma escolhida aqui.

    Modelo sem variavel de chuva nao tem elo de forcamento -- e isso e
    diferente de ter e falhar. `rain_evidence` ja faz a mesma distincao para o
    bloco de evidencia.
    """
    for f in modelo.get("features") or []:
        if str(f).startswith("rain_"):
            return str(f)
    return None


def bloco_da_cadeia(modelo: dict, X, dominio: dict) -> dict:
    """A trilha da cadeia para esta requisicao. Informativo, nunca decisorio."""
    import numpy as np

    variavel = _variavel_de_forcamento(modelo)
    nao_aplicaveis = dict(NAO_APLICAVEIS)
    portoes, entrada = [], {}

    if variavel is None:
        nao_aplicaveis[FORCAMENTO] = (
            "o modelo da regiao nao usa variavel de chuva; nao ha forcamento "
            "estimado para conferir")
    else:
        faixa = (modelo.get("faixa_dominio_5_95") or {}).get(variavel)
        indice = list(modelo["features"]).index(variavel)
        if not faixa or float(faixa[1]) <= float(faixa[0]):
            nao_aplicaveis[FORCAMENTO] = (
                f"o modelo nao declara faixa 5-95% utilizavel para {variavel}")
        else:
            portoes.append(portao_forcamento(faixa, variavel=variavel))
            # O valor conferido e a MEDIA sobre os pontos da area, que e a
            # mesma reducao que produz o escore. Conferir ponto a ponto daria
            # uma cadeia por ponto, e a resposta e da area.
            entrada[variavel] = float(np.mean(np.asarray(X)[:, indice]))

    resultado = executar_cadeia(portoes, entrada) if portoes else None
    bloco = {
        "elos_da_cadeia": list(ELOS),
        "elos_avaliados": [p.elo for p in portoes],
        "elos_nao_aplicaveis": {k: v for k, v in nao_aplicaveis.items()
                                if k not in {p.elo for p in portoes}},
        "cobertura_por_variavel_pct": dominio.get("cobertura_por_variavel_pct"),
        "e_portao_do_contrato": False,
        "nao_e": ("portao: o `status` desta resposta sai de G1-G5 (e do G6 "
                  "quando ligado) e nao muda por causa deste bloco; a cadeia "
                  "aqui informa em qual elo o sistema se calaria"),
    }
    if resultado is None:
        bloco.update({"estado": "nao_avaliada", "parou_em": None,
                      "trilha": [], "confianca_minima_da_cadeia": None,
                      "por_que": "nenhum elo desta cadeia se aplica a esta rota"})
    else:
        trilha = resultado.como_dicionario()
        # As duas advertencias sao diferentes e as duas importam: a da cadeia
        # diz que abster nao e negar, e a do contrato diz que este bloco nao
        # decide. `update` cru apagava a segunda com a primeira em silencio.
        aviso_da_cadeia = trilha.pop("nao_e", None)
        bloco.update(trilha)
        if aviso_da_cadeia:
            bloco["nao_e"] = f"{bloco['nao_e']}. E {aviso_da_cadeia}"
    return bloco
