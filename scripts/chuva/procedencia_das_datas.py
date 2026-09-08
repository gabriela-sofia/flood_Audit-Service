"""A data do evento e mesmo a data do evento?

POR QUE ESTA AUDITORIA EXISTE:

`data_evento` define a janela inteira da chuva -- antecedente e gatilho. Se ela
nao for a data em que a agua apareceu, tudo que se calcula a partir dela mede
outra coisa, e nenhum diagnostico do ajuste acusa isso: a coluna existe, e
preenchida, e tem o formato certo.

O gatilho recem-adquirido deu a primeira medida direta. Em Recife ele separa
mal as classes -- 2,30 mm de mediana nos positivos contra 2,25 nos negativos --
e dois virgula tres milimetros nao inundam nada. Ou a chuva nao explica o
evento, ou a data nao e a do evento.

-------------------------------------------------------------------------
O QUE A AUDITORIA ENCONTROU
-------------------------------------------------------------------------

PRIMEIRO: `precisao_data` existia no pipeline e nao chegava ao modelo. O
`chuva03` a produz declarando o custo de cada origem -- data de chip, data
embutida no nome do arquivo, data de ATIVACAO do CEMS que cobre varias AOIs --
e o `ds04` a descartava na linha em que limpava colunas de merge, por
proximidade e nao por decisao. `data_evento` era uma coluna de valor sem coluna
de procedencia, que e exatamente o que a regra de ouro do `ds03` proibe.
CORRIGIDO em 30/08/2026: a coluna esta no esquema-alvo e viaja com o dado.

E ela traz uma distincao que o projeto tratava como uma coisa so: PRECISAO nao
e GRANULARIDADE. `chip_stac` e a melhor precisao possivel -- a data e da cena
que o analista rotulou, sem aproximacao -- e ainda assim carimba 2.766 pontos
com o mesmo dia. Sao 54.475 pontos que declaram a melhor precisao e entregam
carimbo de lote. So a granularidade diz se a fonte pode informar sobre chuva.

SEGUNDO: essa procedencia perdida e reconstruivel por uma medida simples --
quantos pontos compartilham cada data. E ela parte a base em duas populacoes:

  POR OCORRENCIA        curitiba (3,2 pontos por data), recife (1,4), e a chuva
                        do dia registrado se comporta como se espera: Curitiba
                        da 6,80 mm nos positivos contra 1,10 nos negativos,
                        razao 6,2. Onde a data e da ocorrencia, o gatilho
                        separa as classes.

  CARIMBO DE LOTE       sen1floods11 (30.431 pontos em 11 datas, 2.766 por
                        data), ufo (24.044 em 18 datas) e cems (23.915 em 12).
                        Sao 78.390 dos 84.077 pontos: 93% da base tem data de
                        cena de satelite ou de ativacao de emergencia, e nao de
                        ocorrencia. No ufo a mediana da chuva no dia registrado
                        e ZERO, com 8.025 pontos secos.

Isso nao e defeito de aquisicao: uma cena Sentinel-1 tem uma data so, e a
ativacao do CEMS cobre a emergencia inteira. E a informacao de que aquela data
vale para milhares de pontos ao mesmo tempo estava em `precisao_data` e foi
descartada antes do modelo.

TERCEIRO: em Recife, onde a data E por ocorrencia, a fonte ainda assim nao e de
evento. `v4_canonical_base` sao 132 positivos a 2,30 mm, vindos de
`registro_de_atendimentos_da_defesa_civil`, cuja solicitacao predominante e
`Monitoramento` (60%) e `Vistoria` (18%) -- acompanhamento programado, nao
resposta de emergencia no dia da agua. Ao lado, os 8 positivos do decreto 35669
dao 112 mm de mediana no dia registrado. A mesma fonte carrega duas populacoes
que o modelo trata como uma, e a fonte bruta as separa em colunas diferentes
(`Data` de ocorrencia e `Data_da_Acao`) que o esquema-alvo achatou numa so.

-------------------------------------------------------------------------
O QUE ISSO SIGNIFICA PARA A IDENTIFICABILIDADE
-------------------------------------------------------------------------

A medida anterior atribuia a nao-identificabilidade da chuva a celula de 11 km.
Ela e maior do que isso: em 93% da base a data tambem e compartilhada. Um
estrato (celula, dia) nesses lotes junta milhares de pontos das duas classes com
exatamente o mesmo numero de chuva -- por construcao, e nao por coincidencia.

O QUE ESTE MODULO NAO FAZ: nao corrige data, nao reclassifica ponto, nao
remove linha e nao exclui fonte. Ele mede e declara.

Uso:
    python scripts/chuva/procedencia_das_datas.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from chuva.contrato_chuva import (  # noqa: E402
    METHOD_VERSION, SPATIAL_RESOLUTION_DEGREES, TRIGGER_VARIABLE,
    parse_event_dates,
)

TABELA = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_admitidos_v2.csv"
GATILHO = REPO / "local_runs" / "chuva-gatilho" / "gatilho_por_celula_data.csv"
OUT = REPO / "local_runs" / "chuva-procedencia-datas"

# Abaixo disto, a chuva do dia registrado nao sustenta leitura pluvial do
# evento. Nao e limiar de decisao do produto: e o piso abaixo do qual a
# diferenca entre positivo e negativo deixa de ser atribuivel a chuva.
MM_MINIMO_PARA_LEITURA_PLUVIAL = 5.0


def carregar() -> pd.DataFrame:
    d = pd.read_csv(TABELA, low_memory=False)
    d = d[d.classe.isin([0, 1])].copy()
    d["ts"] = parse_event_dates(d.data_evento)
    return d


def cobertura_da_precisao(d: pd.DataFrame) -> dict:
    """`precisao_data` sobreviveu ate a tabela que ajusta o modelo?

    Ate 30/08/2026 a resposta era nao: o `chuva03` a produzia e o `ds04` a
    descartava junto das colunas de merge. Restituida ao esquema-alvo, ela
    passa a viajar com o dado, e esta funcao deixa de reconstruir procedencia
    para apenas conferir que ela chegou.
    """
    presente = "precisao_data" in d.columns
    r = {
        "coluna_presente_na_tabela_unica": presente,
        "produzida_por": "scripts/treino/chuva03_resolver_datas_evento.py",
        "declarada_no_esquema_alvo": presente,
    }
    if not presente:
        r["consequencia"] = ("`data_evento` e coluna de valor sem coluna de "
                             "procedencia, contra a regra de ouro do ds03")
        return r
    r["valores"] = {k: int(v) for k, v in
                    d.precisao_data.value_counts().items()}
    return r


def precisao_versus_granularidade(d: pd.DataFrame, analise: dict) -> dict:
    """A precisao que a fonte DECLARA e a granularidade que ela ENTREGA.

    Sao duas perguntas diferentes e o projeto tratava como uma. `chip_stac` e
    a melhor precisao possivel -- a data e da cena que o analista rotulou, sem
    aproximacao. E ainda assim ela carimba 2.766 pontos com o mesmo dia,
    porque uma cena Sentinel-1 cobre uma area inteira de uma vez.

    Precisao alta e granularidade grossa convivem, e so a segunda diz se a
    fonte pode informar sobre chuva.
    """
    if "precisao_data" not in d.columns or not analise.get("avaliado"):
        return {"avaliado": False}
    gran = {f: max(procs.values(), key=lambda v: v["n"])["granularidade"]
            for f, procs in analise["por_fonte"].items()}
    saida = {}
    for (fonte, prec), s in d.groupby(["fonte", "precisao_data"]):
        g = gran.get(fonte)
        if g is None:
            continue
        saida[f"{fonte}/{prec}"] = {
            "pontos": int(len(s)),
            "precisao_declarada": prec,
            "granularidade_medida": g,
            "declara_bem_e_carimba_lote": bool(
                prec in ("chip_stac", "nome_do_chip")
                and g == "carimbo_de_lote"),
        }
    return {
        "avaliado": True,
        "por_fonte_e_precisao": saida,
        "leitura": ("precisao alta nao implica granularidade fina: a data pode "
                    "estar exatamente certa para uma cena e ainda assim valer "
                    "para milhares de pontos ao mesmo tempo"),
    }


# Pontos por data continua sendo reportado porque e legivel, mas ele NAO
# decide: numa cidade de doze milhoes de habitantes, dois mil chamados por dia
# e volume normal, cada um com sua data verdadeira, e a razao acusaria lote
# onde nao ha. Foi o que o SP156 mostrou.
#
# O que decide e quantas das datas POSSIVEIS de fato aparecem. As datas
# possiveis sao limitadas pelo que for menor: o numero de pontos (n pontos nao
# podem ocupar mais que n dias) ou o tamanho da janela coberta. Se cada
# registro traz a propria data, essa cobertura vai a um; se todos herdam a data
# de uma cena ou de uma ativacao, ela vai a zero.
#
# Assim Recife -- 205 datas para 278 pontos espalhados por dezessete anos --
# aparece como por ocorrencia, que e o que ela e, enquanto a densidade sobre a
# janela sozinha a condenaria junto com as fontes de satelite.
COBERTURA_DE_DATAS_OCORRENCIA = 0.4
COBERTURA_DE_DATAS_LOTE = 0.05

PONTOS_POR_DATA_LOTE = 100.0
PONTOS_POR_DATA_OCORRENCIA = 10.0


def _dias_no_intervalo(ts) -> int | None:
    """Tamanho da janela coberta, em dias, ou None se nao houver duas datas."""
    v = ts.dropna()
    if len(v) < 2:
        return None
    return int((v.max() - v.min()).days) + 1


def granularidade_da_data(n: int, datas: int,
                          dias_no_intervalo: int | None = None) -> dict:
    """A data e por ocorrencia ou carimbo de lote?

    Com `dias_no_intervalo` conhecido usa a cobertura de datas, que e a medida
    correta. Sem ele cai na razao de pontos por data, que confunde volume com
    lote -- e por isso o intervalo deve ser informado sempre que existir.
    """
    if not datas:
        return {"pontos_por_data": None, "cobertura_de_datas": None,
                "granularidade": "indeterminada"}
    razao = n / datas
    base = {"pontos_por_data": round(razao, 1)}
    if dias_no_intervalo and dias_no_intervalo > 0:
        possiveis = min(n, dias_no_intervalo)
        # capado em 1: dias distintos em UTC podem exceder a janela medida por
        # arredondamento nas pontas, e cobertura maior que o total nao existe
        cobertura = min(datas / possiveis, 1.0) if possiveis else 0.0
        if cobertura >= COBERTURA_DE_DATAS_OCORRENCIA:
            grau = "por_ocorrencia"
        elif cobertura <= COBERTURA_DE_DATAS_LOTE:
            grau = "carimbo_de_lote"
        else:
            grau = "intermediaria"
        return {**base, "cobertura_de_datas": round(cobertura, 4),
                "dias_no_intervalo": int(dias_no_intervalo),
                "granularidade": grau}
    if razao >= PONTOS_POR_DATA_LOTE:
        grau = "carimbo_de_lote"
    elif razao <= PONTOS_POR_DATA_OCORRENCIA:
        grau = "por_ocorrencia"
    else:
        grau = "intermediaria"
    return {**base, "cobertura_de_datas": None, "granularidade": grau}


def por_procedencia(d: pd.DataFrame) -> dict:
    """Chuva no dia registrado, por origem do registro."""
    if not GATILHO.exists():
        return {"avaliado": False,
                "motivo": "gatilho ausente; rode scripts/chuva/adquirir_gatilho.py"}
    g = pd.read_csv(GATILHO)
    res = SPATIAL_RESOLUTION_DEGREES
    z = d[d.ts.notna()].copy()
    z["dia"] = z.ts.dt.strftime("%Y-%m-%d")
    z["cl_lat"] = (z.lat / res).round() * res
    z["cl_lon"] = (z.lon / res).round() * res
    z = z.merge(g[["cl_lat", "cl_lon", "dia", TRIGGER_VARIABLE]],
                on=["cl_lat", "cl_lon", "dia"], how="left")

    saida = {}
    for (fonte, proc), s in z.groupby(["fonte", "procedencia"]):
        v = s[TRIGGER_VARIABLE].dropna().to_numpy(dtype=float)
        if not len(v):
            continue
        datas = int(s.ts.dt.normalize().nunique())
        mediana = round(float(np.median(v)), 2)
        saida.setdefault(fonte, {})[proc] = {
            "procedencia": proc,
            "n": int(len(s)),
            "classes": sorted(s.classe.unique().tolist()),
            "gatilho_resolvido": int(len(v)),
            "mediana_mm_no_dia": mediana,
            "maximo_mm_no_dia": round(float(np.max(v)), 2),
            "pontos_com_chuva_zero": int((v == 0).sum()),
            "datas_distintas": datas,
            **granularidade_da_data(int(len(s)), datas,
                                    _dias_no_intervalo(s.ts)),
            "sustenta_leitura_pluvial": mediana >= MM_MINIMO_PARA_LEITURA_PLUVIAL,
        }
    return {"avaliado": True, "limiar_mm": MM_MINIMO_PARA_LEITURA_PLUVIAL,
            "por_fonte": saida, "agregado": agregado_da_base(saida)}


def agregado_da_base(por_fonte: dict) -> dict:
    """Quanto da base inteira depende de data carimbada em lote."""
    todas = [v for procs in por_fonte.values() for v in procs.values()]
    total = sum(v["n"] for v in todas)
    lote = sum(v["n"] for v in todas if v["granularidade"] == "carimbo_de_lote")
    return {
        "pontos": total,
        "pontos_com_data_de_lote": lote,
        "pct_com_data_de_lote": round(100 * lote / total, 1) if total else None,
        "procedencias_de_lote": sorted(
            v["procedencia"] for v in todas
            if v["granularidade"] == "carimbo_de_lote"),
    }


def contraste_positivo_negativo(analise: dict) -> dict:
    """Onde positivos e negativos foram registrados em dias de chuva igual."""
    if not analise.get("avaliado"):
        return {"avaliado": False}
    achados = {}
    for fonte, procs in analise["por_fonte"].items():
        pos = [v for v in procs.values() if v["classes"] == [1]]
        neg = [v for v in procs.values() if v["classes"] == [0]]
        if not pos or not neg:
            continue
        # A comparacao e entre as procedencias DOMINANTES, e nao entre medias
        # ponderadas: com media, oito pontos de 112 mm puxam cento e trinta e
        # dois de 2,3 mm e o contraste some justamente onde ele existe.
        dom_p = max(pos, key=lambda v: v["n"])
        dom_n = max(neg, key=lambda v: v["n"])
        mp, mn = dom_p["mediana_mm_no_dia"], dom_n["mediana_mm_no_dia"]
        achados[fonte] = {
            "procedencia_dominante_positiva": dom_p["procedencia"],
            "n_positivos": dom_p["n"],
            "mediana_mm_positivos": mp,
            "procedencia_dominante_negativa": dom_n["procedencia"],
            "n_negativos": dom_n["n"],
            "mediana_mm_negativos": mn,
            "razao": round(mp / mn, 2) if mn else None,
            "procedencias_positivas": len(pos),
            "procedencias_negativas": len(neg),
            "positivos_indistinguiveis_dos_negativos": bool(
                mn > 0 and mp / mn < 1.5),
        }
    return {"avaliado": True, "por_fonte": achados}


def main() -> int:
    if not TABELA.exists():
        print(f"tabela unica ausente: {TABELA}")
        return 2
    d = carregar()
    analise = por_procedencia(d)

    relatorio = {
        "artefato": "chuva_procedencia_das_datas",
        "schema_version": "1.0.0",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method_version": METHOD_VERSION,
        "pergunta": "a data registrada e a data em que a agua apareceu?",
        "precisao_da_data": cobertura_da_precisao(d),
        "chuva_no_dia_registrado": analise,
        "precisao_versus_granularidade": precisao_versus_granularidade(d, analise),
        "contraste": contraste_positivo_negativo(analise),
        "fonte_bruta_de_recife": {
            "arquivo": ("data/raw/recife/seced_defesa_civil/"
                        "registro_de_atendimentos_da_defesa_civil__atendimentos_*.csv"),
            "o_que_registra": "atendimentos da defesa civil, nao eventos",
            "colunas_de_data": {
                "Data": "data de ocorrencia; vazia em parte dos registros",
                "Data_da_Acao": "data da acao de resposta ou acompanhamento",
            },
            "solicitacao_predominante": {
                "Monitoramento": "60% das linhas",
                "Vistoria": "18% das linhas",
            },
            "leitura": ("acompanhamento programado domina a base; a data de um "
                        "monitoramento nao e a data do alagamento"),
            "referencia": ("outputs_public/data/susc_20a_aquisicao_eventos_reais_"
                           "recife/reports/gap5_positivos_sem_data_v4.md"),
        },
        "nao_e": ("nao corrige data, nao reclassifica ponto e nao remove linha: "
                  "mede e declara"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    destino = OUT / "procedencia_das_datas.json"
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/chuva/procedencia_das_datas.py\n", encoding="utf-8")

    p = relatorio["precisao_da_data"]
    print(f"precisao_data na tabela unica: {p['coluna_presente_na_tabela_unica']}")
    if analise.get("avaliado"):
        print(f"\nchuva no dia registrado (limiar {MM_MINIMO_PARA_LEITURA_PLUVIAL} mm):")
        for fonte, procs in sorted(analise["por_fonte"].items()):
            for proc, v in sorted(procs.items(),
                                  key=lambda kv: -kv[1]["mediana_mm_no_dia"]):
                marca = "ok " if v["sustenta_leitura_pluvial"] else "NAO"
                print(f"  {marca} {fonte:9s} {proc[:40]:40s} n={v['n']:6d} "
                      f"mediana={v['mediana_mm_no_dia']:7.2f} mm  "
                      f"{v['pontos_por_data']:>7} pt/data  {v['granularidade']}")
        for fonte, c in relatorio["contraste"].get("por_fonte", {}).items():
            if c["positivos_indistinguiveis_dos_negativos"]:
                print(f"\n  {fonte}: {c['procedencia_dominante_positiva'][:28]} "
                      f"({c['n_positivos']} pos, {c['mediana_mm_positivos']} mm)"
                      f" vs {c['procedencia_dominante_negativa'][:26]} "
                      f"({c['n_negativos']} neg, {c['mediana_mm_negativos']} mm)"
                      f" — razao {c['razao']}: indistinguiveis")
    print(f"\nRELATORIO={destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
