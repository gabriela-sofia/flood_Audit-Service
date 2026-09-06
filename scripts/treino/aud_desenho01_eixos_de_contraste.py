"""Para que serve cada fonte: o contraste que ela oferece, e o que ela nao responde.

POR QUE ESTA AUDITORIA EXISTE:

o projeto acumulou seis fontes e tratou todas como "dados de inundacao". Elas
nao sao a mesma coisa, e a diferenca nao esta na qualidade -- esta em QUAL
PERGUNTA cada uma consegue responder. Uma fonte pode ser impecavel para estimar
predisposicao do terreno e ser estruturalmente incapaz de dizer qualquer coisa
sobre chuva. Isso nao aparece em nenhuma metrica de ajuste: o modelo roda, o
coeficiente sai, o intervalo fecha, e a resposta e sobre outra coisa.

-------------------------------------------------------------------------
OS DOIS EIXOS
-------------------------------------------------------------------------

Um par positivo/negativo so informa sobre uma variavel se os dois membros do
par DIFEREM naquela variavel. Isso parte a base em dois eixos independentes:

EIXO DO TERRENO (espacial)
    Positivos e negativos dentro da MESMA celula de chuva. A chuva e a mesma
    para os dois -- ela esta controlada por construcao --, entao o que os
    distingue e a posicao no terreno. Este e o eixo em que HAND e TWI
    respondem, e e por isso que eles respondem.

EIXO DA CHUVA (temporal)
    Positivos e negativos que diferem em chuva. Aqui esta a armadilha: dentro
    de uma celula-dia a chuva e CONSTANTE por construcao do produto (0,1 grau,
    diario). Todo par dentro de uma celula-dia tem diferenca de chuva
    exatamente zero, e nao informa nada sobre ela -- por mais pontos que tenha.
    Para informar, o par precisa cruzar celulas ou dias, e ai a diferenca de
    chuva vem acompanhada de diferenca de terreno ou de data.

A consequencia e que os dois eixos competem: o desenho que controla a chuva
para medir terreno e o mesmo que impede medir chuva. Nao e defeito de
aquisicao nem de curadoria. E a estrutura do dado.

-------------------------------------------------------------------------
O QUE MUDA A LEITURA DO EIXO TEMPORAL
-------------------------------------------------------------------------

`procedencia_das_datas` mostrou que 93% da base tem data carimbada em lote --
cena de satelite ou ativacao de emergencia. Numa fonte assim, "positivos e
negativos na mesma data" e trivialmente verdadeiro e nao significa pareamento:
significa que a fonte inteira tem doze datas. Por isso a granularidade da data
entra aqui, e nao como enfeite: sem ela, o eixo temporal parece existir onde
ele e um artefato de como a fonte foi montada.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao remove fonte, nao repondera, nao ajusta modelo e nao decide o que entra no
vetor. Ele mede o contraste disponivel e declara, por fonte, quais perguntas
aquela fonte pode responder.

Uso:
    python scripts/treino/aud_desenho01_eixos_de_contraste.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from chuva.contrato_chuva import (  # noqa: E402
    METHOD_VERSION, SPATIAL_RESOLUTION_DEGREES, parse_event_dates,
)
from chuva.procedencia_das_datas import granularidade_da_data  # noqa: E402

TABELA = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_admitidos_v2.csv"
OUT = REPO / "local_runs" / "aud-desenho-01"

# Abaixo disto o eixo nao sustenta leitura: sao poucos pares para que a
# diferenca observada seja atribuivel a variavel, e nao ao acaso da amostra.
# Nao e teste de hipotese -- e piso de existencia do contraste.
PCT_MINIMO_DO_EIXO = 10.0


def celula(serie: pd.Series) -> pd.Series:
    return (serie / SPATIAL_RESOLUTION_DEGREES).round().astype(str)


def preparar(caminho: Path = TABELA) -> pd.DataFrame:
    d = pd.read_csv(caminho, low_memory=False)
    d = d[d.classe.isin([0, 1])].copy()
    d["dia"] = parse_event_dates(d.data_evento).dt.strftime("%Y-%m-%d")
    d["cel"] = celula(d.lat) + "_" + celula(d.lon)
    d["cel_dia"] = d.cel + "@" + d.dia.fillna("sem_data")
    return d


def pct_em_estrato_misto(s: pd.DataFrame, chave: str) -> float:
    """Fracao de pontos que dividem o estrato com a classe oposta."""
    if not len(s):
        return 0.0
    mistos = s.groupby(chave).classe.nunique()
    return round(100 * float(s[chave].isin(mistos[mistos == 2].index).mean()), 1)


def dias_no_intervalo(s: pd.DataFrame) -> int | None:
    """A janela que a fonte cobre, para separar volume de carimbo de lote."""
    v = pd.to_datetime(s["dia"], format="%Y-%m-%d", errors="coerce").dropna()
    if len(v) < 2:
        return None
    return int((v.max() - v.min()).days) + 1


def eixos_da_fonte(s: pd.DataFrame) -> dict:
    """O contraste que esta fonte oferece, nos dois eixos."""
    datas = int(s.dia.nunique())
    gran = granularidade_da_data(len(s), datas, dias_no_intervalo(s))

    # Eixo do terreno: pontos das duas classes na mesma celula de chuva.
    pct_terreno = pct_em_estrato_misto(s, "cel")

    # Eixo da chuva: os pontos que dividem celula-dia tem diferenca de chuva
    # ZERO por construcao. Quanto maior este numero, menos a chuva pode
    # explicar as classes -- e o complemento e o unico lugar onde ela poderia,
    # ao custo de a comparacao passar a cruzar lugares ou dias.
    pct_chuva_neutralizada = pct_em_estrato_misto(s, "cel_dia")
    pct_data_mista = pct_em_estrato_misto(s, "dia")

    serve_terreno = pct_terreno >= PCT_MINIMO_DO_EIXO
    # A chuva so teria chance se sobrasse contraste FORA da celula-dia e a data
    # fosse da ocorrencia. Com carimbo de lote, o que sobra e diferenca entre
    # lotes, e nao entre condicoes meteorologicas de pontos comparaveis.
    serve_chuva = (gran["granularidade"] == "por_ocorrencia"
                   and pct_data_mista >= PCT_MINIMO_DO_EIXO)

    responde = [p for p, ok in (("predisposicao_do_terreno", serve_terreno),
                                ("efeito_da_chuva", serve_chuva)) if ok]
    return {
        "pontos": int(len(s)),
        "positivos": int((s.classe == 1).sum()),
        "negativos": int((s.classe == 0).sum()),
        "celulas": int(s.cel.nunique()),
        "datas": datas,
        **gran,
        "pct_pontos_em_celula_mista": pct_terreno,
        "pct_pontos_em_data_mista": pct_data_mista,
        "pct_pontos_com_chuva_neutralizada": pct_chuva_neutralizada,
        "responde": responde or ["nenhuma_pergunta_do_contrato"],
        "cega_para_chuva": not serve_chuva,
    }


def motivo_da_cegueira(e: dict) -> str:
    """Por que esta fonte nao informa sobre chuva, na linguagem do fenomeno."""
    if not e["cega_para_chuva"]:
        return ""
    if e["granularidade"] == "carimbo_de_lote":
        return (f"a data e carimbo de lote ({e['pontos_por_data']} pontos por "
                f"data em {e['datas']} datas): ela marca a cena ou a ativacao, "
                f"nao a ocorrencia")
    if e["pct_pontos_em_data_mista"] < PCT_MINIMO_DO_EIXO:
        return (f"positivos e negativos quase nunca dividem o dia "
                f"({e['pct_pontos_em_data_mista']}% dos pontos): toda diferenca "
                f"de chuva e diferenca entre datas, nao entre lugares")
    return "contraste insuficiente nos dois criterios"


def auditar(d: pd.DataFrame) -> dict:
    por_fonte = {f: eixos_da_fonte(s) for f, s in d.groupby("fonte")}
    for e in por_fonte.values():
        e["motivo_da_cegueira"] = motivo_da_cegueira(e)
    total = sum(e["pontos"] for e in por_fonte.values())
    if not total:
        return {"por_fonte": por_fonte, "agregado": {"pontos": 0}}
    cegas = sum(e["pontos"] for e in por_fonte.values() if e["cega_para_chuva"])
    terreno = sum(e["pontos"] for e in por_fonte.values()
                  if "predisposicao_do_terreno" in e["responde"])
    return {
        "por_fonte": por_fonte,
        "agregado": {
            "pontos": total,
            "pct_que_sustenta_eixo_do_terreno": round(100 * terreno / total, 1),
            "pct_cego_para_chuva": round(100 * cegas / total, 1),
            "fontes_que_informam_sobre_chuva": sorted(
                f for f, e in por_fonte.items() if not e["cega_para_chuva"]),
        },
    }


def main() -> int:
    if not TABELA.exists():
        print(f"tabela unica ausente: {TABELA}")
        return 2
    r = auditar(preparar())

    relatorio = {
        "artefato": "aud_desenho01_eixos_de_contraste",
        "schema_version": "1.0.0",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method_version": METHOD_VERSION,
        "pergunta": "que contraste cada fonte oferece, e o que ela nao pode responder",
        "eixos": {
            "predisposicao_do_terreno": (
                "positivos e negativos na mesma celula de chuva: a chuva esta "
                "controlada por construcao e o que distingue e o terreno"),
            "efeito_da_chuva": (
                "exige data de ocorrencia E positivos e negativos dividindo o "
                "dia; dentro de uma celula-dia a diferenca de chuva e zero"),
        },
        "pct_minimo_do_eixo": PCT_MINIMO_DO_EIXO,
        **r,
        "nao_e": ("nao remove fonte, nao repondera, nao ajusta modelo e nao "
                  "decide o que entra no vetor"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destino = OUT / "eixos_de_contraste.json"
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/treino/aud_desenho01_eixos_de_contraste.py\n",
        encoding="utf-8")

    print(f"{'fonte':13s} {'n':>6s} {'cels':>5s} {'datas':>6s} {'pt/data':>8s} "
          f"{'terreno':>8s} {'chuva=0':>8s}  responde")
    for f, e in sorted(r["por_fonte"].items()):
        print(f"{f:13s} {e['pontos']:6d} {e['celulas']:5d} {e['datas']:6d} "
              f"{e['pontos_por_data']:8.1f} "
              f"{e['pct_pontos_em_celula_mista']:7.1f}% "
              f"{e['pct_pontos_com_chuva_neutralizada']:7.1f}%  "
              f"{', '.join(e['responde'])}")
    a = r["agregado"]
    print(f"\nsustenta o eixo do terreno: {a['pct_que_sustenta_eixo_do_terreno']}% dos pontos")
    print(f"cego para chuva:            {a['pct_cego_para_chuva']}% dos pontos")
    print(f"informam sobre chuva:       {a['fontes_que_informam_sobre_chuva'] or 'nenhuma'}")
    print(f"\nRELATORIO={destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
