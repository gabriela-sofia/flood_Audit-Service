"""MRMS a 1 km contra ERA5-Land a 11 km, nas MESMAS linhas.

POR QUE ESTE MODULO EXISTE, e o erro que ele corrige:

O canario de 8 dias da aquisicao MRMS mediu "fracao da variancia entre lugares
no mesmo dia" e deu 0,2138, contra 0,075 do ERA5-Land. Foi reportado como o
ganho de E3. Os dois numeros estavam errados como comparacao, por dois motivos
independentes:

  1. AMOSTRA. A fracao e intra/total. Com 8 dias, a variancia ENTRE dias quase
     nao existe, entao a fracao intra infla por construcao. Na serie inteira o
     MRMS da 0,0622 -- nao 0,2138.

  2. BASE. O 0,075 do ERA5 foi calculado sobre a serie (celula x dia, 19
     celulas) e o do MRMS sobre os pontos (468 mil chamados). Bases diferentes
     nao se comparam. Nas mesmas linhas o ERA5 da 0,0474 e o MRMS 0,0622.

E ha um terceiro problema, que e o mais interessante: **a fracao agregada pune o
MRMS por ele estar certo.** Em dia seco o radar ve zero em toda parte, e a
variacao espacial e zero -- correta. A reanalise do ERA5 devolve valores
pequenos e DIFERENTES entre suas celulas no mesmo dia seco, o que conta como
variacao espacial no denominador sem ser sinal de chuva nenhum. Como dois tercos
dos dias sao secos, o agregado premia o ruido.

A pergunta certa nao e "quanto varia no agregado", e sim **"nos dias em que
chove, o produto distingue bairros?"**.

O QUE NAO FAZ: nao ajusta modelo, nao promove variavel.

Uso:
    python scripts/chuva/mrms_vs_era5_nyc.py
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
    SPATIAL_RESOLUTION_DEGREES as RES, parse_event_dates,
)

RUNS = REPO / "local_runs"
TABELA = RUNS / "ds-05-tabela-unica" / "tabela_unica_admitidos_v2.csv"
ERA5 = RUNS / "chuva-serie-celula" / "serie_diaria_por_celula_nyc.csv"
MRMS = RUNS / "chuva-mrms-nyc" / "mrms_por_ponto_nyc.csv"
OUT = RUNS / "chuva-mrms-nyc"

PRODUTOS = {"era5_land_11km": "rain_event_day", "mrms_1km": "mrms_qpe_24h_mm"}
# Acima disto o dia conta como dia de chuva. Abaixo, o radar ver zero em toda
# parte e correto, e nao ausencia de resolucao.
CHUVA_MINIMA_MM = 1.0


def carregar() -> pd.DataFrame:
    d = pd.read_csv(TABELA, low_memory=False,
                    usecols=["ponto_id", "fonte", "classe", "lat", "lon",
                             "data_evento"])
    d = d[(d.fonte == "nyc") & d.classe.isin([0, 1])].copy()
    d["ts"] = parse_event_dates(d.data_evento)
    d = d[d.ts.notna()].copy()
    d["dia"] = d.ts.dt.strftime("%Y-%m-%d")
    d["cl_lat"] = ((d.lat / RES).round() * RES).round(4)
    d["cl_lon"] = ((d.lon / RES).round() * RES).round(4)

    era = pd.read_csv(ERA5, usecols=["cl_lat", "cl_lon", "dia",
                                     "rain_event_day"])
    era = era.drop_duplicates(["cl_lat", "cl_lon", "dia"])
    era["cl_lat"] = era.cl_lat.round(4)
    era["cl_lon"] = era.cl_lon.round(4)
    mrms = pd.read_csv(MRMS).drop_duplicates(["ponto_id", "dia"])

    m = (d.merge(era, on=["cl_lat", "cl_lon", "dia"], how="inner")
          .merge(mrms, on=["ponto_id", "dia"], how="inner"))
    return m.dropna(subset=list(PRODUTOS.values()))


def descrever(sel: pd.DataFrame) -> dict:
    saida = {"pontos": int(len(sel)), "dias": int(sel.dia.nunique())}
    for nome, col in PRODUTOS.items():
        g = sel.groupby("dia")[col]
        total = float(sel[col].var())
        intra = float((sel[col] - g.transform("mean")).var())
        saida[nome] = {
            "valores_distintos_por_dia_mediana": int(g.nunique().median()),
            "amplitude_no_dia_mm_mediana": round(float((g.max() - g.min()).median()), 3),
            "desvio_no_dia_mm_mediano": round(float(g.std().median()), 4),
            "variancia_total": round(total, 3),
            "variancia_intra_dia": round(intra, 3),
            "fracao_intra_dia": round(intra / total, 4) if total else None,
        }
    return saida


def main() -> int:
    m = carregar()
    chuvoso = m[m.groupby("dia").mrms_qpe_24h_mm.transform("median")
                > CHUVA_MINIMA_MM]
    recortes = {"todos_os_dias": descrever(m),
                "dias_de_chuva": descrever(chuvoso)}

    print("=== MRMS 1 km CONTRA ERA5-Land 11 km, mesmas linhas ===")
    for recorte, r in recortes.items():
        print(f"\n--- {recorte}: {r['pontos']:,} pontos em {r['dias']:,} dias")
        print(f"{'':18s} {'distintos/dia':>14s} {'amplitude mm':>13s} "
              f"{'desvio mm':>10s} {'fracao intra':>13s}")
        for nome in PRODUTOS:
            v = r[nome]
            print(f"{nome:18s} {v['valores_distintos_por_dia_mediana']:14d} "
                  f"{v['amplitude_no_dia_mm_mediana']:13.2f} "
                  f"{v['desvio_no_dia_mm_mediano']:10.3f} "
                  f"{v['fracao_intra_dia']:13.4f}")

    chuva = recortes["dias_de_chuva"]
    ganho = (chuva["mrms_1km"]["valores_distintos_por_dia_mediana"]
             / max(chuva["era5_land_11km"]["valores_distintos_por_dia_mediana"], 1))

    manifesto = {
        "tipo": "chuva_mrms_vs_era5_nyc",
        "gerado_por": "scripts/chuva/mrms_vs_era5_nyc.py",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chuva_minima_mm": CHUVA_MINIMA_MM,
        "recortes": recortes,
        "ganho_de_distincao_espacial_em_dia_de_chuva": round(ganho, 2),
        "correcao_de_leitura_anterior": {
            "reportado_no_canario": 0.2138,
            "por_que_estava_errado": (
                "8 dias: com pouca variancia ENTRE dias, a fracao intra infla "
                "por construcao; e o valor do ERA5 usado na comparacao vinha de "
                "outra base (serie celula x dia, nao pontos)"),
            "valor_correto_na_serie_inteira": recortes["todos_os_dias"][
                "mrms_1km"]["fracao_intra_dia"],
            "valor_do_era5_nas_mesmas_linhas": recortes["todos_os_dias"][
                "era5_land_11km"]["fracao_intra_dia"],
        },
        "leitura": (
            "no agregado o MRMS parece pouco melhor, e a fracao chega a punir o "
            "produto certo: em dia seco o radar ve zero em toda parte, o que e "
            "verdade e conta como variacao espacial nula, enquanto a reanalise "
            "devolve valores pequenos e diferentes entre celulas sem que chova. "
            "Nos dias de chuva, que sao os que interessam, o MRMS distingue "
            f"{ganho:.0f}x mais valores dentro do mesmo dia"),
        "nao_e": (
            "nao e medida de acerto de nenhum dos dois produtos contra "
            "pluviometro: e comparacao de quanta distincao espacial cada um "
            "oferece nos mesmos pontos e nas mesmas datas"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mrms_vs_era5.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nem dia de chuva o MRMS distingue {ganho:.0f}x mais valores por dia")
    print(f"GRAVADO={OUT / 'mrms_vs_era5.json'}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
