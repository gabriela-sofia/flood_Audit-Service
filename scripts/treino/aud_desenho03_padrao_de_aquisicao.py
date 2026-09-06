"""O padrao que faz Curitiba funcionar, e como testar uma fonte antes de adquiri-la.

POR QUE ESTE MODULO EXISTE:

Curitiba e a unica das seis fontes que informa sobre chuva, e por muito tempo
isso pareceu sorte. Nao e: e uma propriedade da fonte, e ela pode ser descrita
e procurada.

  positivos   `siac156_curitiba_v1` -- chamados hidrologicos ao 156, a central
              de atendimento ao cidadao.
  negativos   `siac156_curitiba_non_hydrological_category_v1` -- chamados da
              MESMA base, em categoria nao-hidrologica.

Duas propriedades saem disso de graca, e sao as duas que faltam em toda parte:

  DATA POR OCORRENCIA   um chamado existe porque alguem ligou naquele dia. A
                        data e do evento por construcao, e nao carimbo de cena
                        ou de ativacao. Curitiba tem 642 datas para 1.680
                        pontos; sen1floods11 tem 11 para 30.431.

  FUNDO DA MESMA BASE   quem liga por outro assunto passou pelo mesmo processo
                        de observacao. E o target-group background de Phillips
                        et al. (2009), que cancela o viesde esforco amostral em
                        vez de introduzi-lo.

-------------------------------------------------------------------------
A DIFERENCA ENTRE CURITIBA E RECIFE, QUE E O CRITERIO INTEIRO
-------------------------------------------------------------------------

Recife tambem vem da defesa civil, tambem tem fundo da mesma instituicao, e
mesmo assim nao informa sobre chuva. A diferenca esta em QUEM origina o
registro:

  Curitiba   DEMANDA DO CIDADAO. "Tem agua na minha rua" e um evento datado.
  Recife     ACAO DA INSTITUICAO. `Monitoramento` (60%) e `Vistoria` (18%) sao
             agenda, e a data e a da agenda -- dai os 2,30 mm de mediana no dia
             registrado, iguais aos 2,25 dos negativos.

Nao e questao de qualidade da fonte. As duas sao boas; elas respondem perguntas
diferentes porque sao originadas por processos diferentes.

-------------------------------------------------------------------------
O QUE ESTE MODULO FAZ, E O QUE NAO FAZ
-------------------------------------------------------------------------

Ele transforma o padrao em requisitos verificaveis e avalia uma fonte candidata
ANTES de qualquer esforco de aquisicao -- que e onde o projeto gastou mais
tempo historicamente, adquirindo primeiro e descobrindo o desenho depois.

NAO baixa nada, nao acessa rede e nao decide adquirir. Requisito que ninguem
verificou fica `None` e aparece como desconhecido, nunca como atendido.

Uso:
    python scripts/treino/aud_desenho03_padrao_de_aquisicao.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "local_runs" / "aud-desenho-03"

# Cada requisito nasce de uma medida ja feita neste projeto, e nao de boa
# pratica generica. A coluna `porque` guarda essa origem.
REQUISITOS = {
    "origem_e_demanda_do_cidadao": {
        "porque": ("Recife tem fundo da mesma instituicao e ainda assim da "
                   "2,30 mm no dia registrado, porque o positivo e acao "
                   "programada; o chamado do cidadao e datado por construcao"),
        # So a chuva depende disto. Recife e originada por acao institucional e
        # ainda assim sustenta o eixo do terreno com 100% dos pontos em celula
        # mista: para saber ONDE a agua chega basta que alguem tenha ido la, e
        # nao importa quem decidiu ir. O que a agenda estraga e a DATA.
        "necessario_para": ["chuva"],
    },
    "fundo_da_mesma_base": {
        "porque": ("target-group background (Phillips et al. 2009): sem ele o "
                   "modelo aprende onde a instituicao vai, nao onde a agua chega"),
        "necessario_para": ["terreno", "chuva"],
    },
    "geolocalizacao_por_ocorrencia": {
        "porque": ("o eixo do terreno exige positivos e negativos na mesma "
                   "celula; sem ponto nao ha celula"),
        "necessario_para": ["terreno", "chuva"],
    },
    "data_por_ocorrencia": {
        "porque": ("93% da base atual e carimbo de lote, e ai a chuva e "
                   "constante dentro do estrato por construcao"),
        "necessario_para": ["chuva"],
    },
    "classes_dividem_o_dia": {
        "porque": ("o Reino Unido tem data fina e 0% dos pontos em data mista: "
                   "toda diferenca de chuva vira diferenca entre datas"),
        "necessario_para": ["chuva"],
    },
    "terreno_derivavel": {
        "porque": ("HAND e TWI sao o que responde no eixo do terreno; sem DEM "
                   "utilizavel a fonte nao entra em modelo nenhum"),
        "necessario_para": ["terreno", "chuva"],
    },
}

# Fontes candidatas. Um campo so e `True` quando ha evidencia verificada; o que
# ninguem checou fica `None`. Nenhuma delas foi adquirida por este modulo.
CANDIDATAS = {
    "curitiba_siac156": {
        "regiao": "Curitiba, BR", "estado": "ja_no_projeto",
        "fonte": "SIAC 156 -- dadosabertos.curitiba.pr.gov.br",
        "origem_e_demanda_do_cidadao": True, "fundo_da_mesma_base": True,
        "geolocalizacao_por_ocorrencia": True, "data_por_ocorrencia": True,
        "classes_dividem_o_dia": True, "terreno_derivavel": True,
        "nota": "a referencia do padrao: 2,6 pontos por data, razao 6,18 na chuva",
    },
    "recife_sedec": {
        "regiao": "Recife, BR", "estado": "ja_no_projeto",
        "fonte": "registro de atendimentos da defesa civil -- dados.recife.pe.gov.br",
        "origem_e_demanda_do_cidadao": False, "fundo_da_mesma_base": True,
        "geolocalizacao_por_ocorrencia": True, "data_por_ocorrencia": True,
        "classes_dividem_o_dia": False, "terreno_derivavel": True,
        "nota": ("Monitoramento 60% e Vistoria 18%: acao programada. A fonte "
                 "bruta separa `Data` de `Data_da_Acao` e o esquema achatou"),
    },
    "sp156": {
        "regiao": "Sao Paulo, BR", "estado": "candidata_nao_verificada",
        "fonte": ("SP156/SIGRC -- dados.prefeitura.sp.gov.br/dataset/dados-do-sp156; "
                  "alagamentos da defesa civil no GeoSampa"),
        "origem_e_demanda_do_cidadao": True, "fundo_da_mesma_base": None,
        "geolocalizacao_por_ocorrencia": None, "data_por_ocorrencia": None,
        "classes_dividem_o_dia": None, "terreno_derivavel": None,
        "nota": ("analogo direto do 156 de Curitiba, com historico declarado "
                 "de 2013 em diante; a maior base municipal candidata"),
    },
    "rio1746": {
        "regiao": "Rio de Janeiro, BR", "estado": "candidata_nao_verificada",
        "fonte": "central 1746 -- data.rio (IPP)",
        "origem_e_demanda_do_cidadao": True, "fundo_da_mesma_base": None,
        "geolocalizacao_por_ocorrencia": None, "data_por_ocorrencia": None,
        "classes_dividem_o_dia": None, "terreno_derivavel": None,
        "nota": "mesma arquitetura de central unica; relevo misto planicie-encosta",
    },
    "fortaleza_copdc": {
        "regiao": "Fortaleza, BR", "estado": "candidata_nao_verificada",
        "fonte": "COPDC via CIOPS -- dados.fortaleza.ce.gov.br",
        "origem_e_demanda_do_cidadao": None, "fundo_da_mesma_base": None,
        "geolocalizacao_por_ocorrencia": None, "data_por_ocorrencia": None,
        "classes_dividem_o_dia": None, "terreno_derivavel": None,
        "nota": ("ocorrencias de protecao e defesa civil; verificar se o "
                 "registro nasce de chamado ou de acao, como em Recife"),
    },
    "nyc311": {
        "regiao": "Nova York, EUA", "estado": "candidata_nao_verificada",
        "fonte": ("NYC Open Data -- '311 Street Flooding and Sewer Backups' "
                  "(9qq5-d465) e a base 311 completa como fundo"),
        "origem_e_demanda_do_cidadao": True, "fundo_da_mesma_base": None,
        "geolocalizacao_por_ocorrencia": None, "data_por_ocorrencia": None,
        "classes_dividem_o_dia": None, "terreno_derivavel": None,
        "nota": ("o 311 e o 156 dos EUA e ja tem recorte tematico pronto de "
                 "alagamento de rua; base completa desde 2010"),
    },
}


def avaliar(c: dict) -> dict:
    """O que a candidata cumpre, o que falta e o que ninguem verificou."""
    def estado(alvo):
        req = [r for r, v in REQUISITOS.items() if alvo in v["necessario_para"]]
        falta = [r for r in req if c.get(r) is False]
        aberto = [r for r in req if c.get(r) is None]
        if falta:
            return "nao_serve", falta, aberto
        if aberto:
            return "indeterminado", falta, aberto
        return "serve", falta, aberto

    st, ft, ab = estado("terreno")
    sc, fc, ac = estado("chuva")
    return {
        "regiao": c["regiao"], "estado": c["estado"], "fonte": c["fonte"],
        "eixo_terreno": st, "eixo_chuva": sc,
        "requisitos_reprovados": sorted(set(ft) | set(fc)),
        "requisitos_nao_verificados": sorted(set(ab) | set(ac)),
        "nota": c["nota"],
    }


def fila_de_prioridade(aval: dict) -> list:
    """Candidata que pode destravar o eixo da chuva vem antes, e nao a maior.

    O projeto tem seis fontes que respondem sobre terreno e uma que responde
    sobre chuva. Fonte nova que so acrescenta terreno soma; fonte que
    acrescenta chuva muda o que o projeto consegue afirmar.
    """
    def peso(item):
        _, v = item
        if v["estado"] == "ja_no_projeto":
            return (3, 0)
        # menos requisitos em aberto = mais barato de verificar
        pode_chuva = v["eixo_chuva"] in ("serve", "indeterminado")
        return (0 if pode_chuva else 1, len(v["requisitos_nao_verificados"]))
    return [k for k, _ in sorted(aval.items(), key=peso)]


def main() -> int:
    aval = {k: avaliar(v) for k, v in CANDIDATAS.items()}
    fila = fila_de_prioridade(aval)

    relatorio = {
        "artefato": "aud_desenho03_padrao_de_aquisicao",
        "schema_version": "1.0.0",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pergunta": ("que fonte tem o desenho de Curitiba, e o que falta "
                     "verificar antes de gastar esforco de aquisicao"),
        "padrao": {
            "nome": "central de atendimento ao cidadao com categoria",
            "positivos": "chamados da categoria hidrologica",
            "negativos": "chamados da MESMA base em categoria nao-hidrologica",
            "por_que_funciona": ("o chamado e datado por construcao e o fundo "
                                 "passou pelo mesmo processo de observacao"),
            "o_que_o_distingue_de_Recife": (
                "demanda do cidadao contra acao programada da instituicao"),
        },
        "requisitos": REQUISITOS,
        "candidatas": aval,
        "fila_de_prioridade": fila,
        "criterio_da_fila": ("candidata que pode destravar o eixo da chuva vem "
                             "antes da que so acrescenta terreno"),
        "nao_e": ("nao baixa nada, nao acessa rede e nao decide adquirir; "
                  "requisito nao verificado aparece como desconhecido"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destino = OUT / "padrao_de_aquisicao.json"
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/treino/aud_desenho03_padrao_de_aquisicao.py\n",
        encoding="utf-8")

    print(f"{'candidata':20s} {'terreno':14s} {'chuva':14s} {'aberto':>6s}  regiao")
    for k in fila:
        v = aval[k]
        print(f"{k:20s} {v['eixo_terreno']:14s} {v['eixo_chuva']:14s} "
              f"{len(v['requisitos_nao_verificados']):6d}  {v['regiao']}")
    print(f"\nfila: {' > '.join(fila)}")
    print(f"\nRELATORIO={destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
