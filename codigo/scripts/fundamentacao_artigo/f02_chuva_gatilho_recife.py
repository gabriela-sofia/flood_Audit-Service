"""F02: a chuva do dia aumenta os relatos de alagamento em Recife? Desenho autopareado no tempo.

Mesmo desenho e estimador já usados em Nova York e São Paulo (`scripts/chuva/case_crossover.py`):
Poisson condicional com estratos ano-mês x dia da semana. Cada dia é comparado aos outros dias do
mesmo mês e mesmo dia da semana, então tudo que não muda no lugar (terreno, drenagem, hábito de
registrar) se cancela. O relato da SEDEC é a ocorrência registrada; a data é `data_acao`.

Protocolo declarado antes de estimar:
  - casos: relatos SEDEC 2014-2024 sem duplicata de conteúdo, contados por dia (cidade inteira);
  - exposição primária: chuva do dia (ERA5, Open-Meteo, ponto -8,05/-34,90), efeito por 10 mm;
  - sensibilidades: chuva de 48 h (dia anterior + dia), porque a ação pode ocorrer no dia seguinte;
    estação INMET A301 (Recife) em 2020-2023 (o arquivo de 2024 não traz a A301), medida no chão;
  - controle negativo: chuva de 30 dias depois, que não pode causar o relato de hoje (deve dar ~1);
  - leitura: "sustentado" se o IC95% por bootstrap de estratos excluir 1 e o controle negativo incluir 1.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts/chuva"))
from case_crossover import _blocos, ajustar  # noqa: E402

SAIDA = RAIZ / "local_runs/fundamentacao-artigo-20261028"
RELATOS = RAIZ / "local_runs/recife-aprendizado/v1/referencias/relatos_sedec_geocodificados.parquet"
ERA5 = SAIDA / "chuva/era5_recife_diario.json"
INMET = RAIZ / "data/external_raw/inmet/historical"
POR_MM, B, SEMENTE, FUTURO = 10.0, 2000, 20261028, 30


def serie_era5() -> pd.Series:
    d = json.loads(ERA5.read_text(encoding="utf-8"))["daily"]
    return pd.Series(d["precipitation_sum"], index=pd.to_datetime(d["time"]), dtype=float)


def serie_inmet() -> pd.Series:
    partes = []
    for ano in range(2020, 2024):
        z = zipfile.ZipFile(INMET / f"inmet_{ano}.zip")
        nome = next(n for n in z.namelist() if "_A301_" in n)
        t = pd.read_csv(io.BytesIO(z.read(nome)), sep=";", skiprows=8, encoding="latin-1", decimal=",")
        data = pd.to_datetime(t.iloc[:, 0].astype(str).str.replace("/", "-"))
        chuva = pd.to_numeric(t.iloc[:, 2], errors="coerce")
        chuva[chuva < 0] = np.nan
        horas = chuva.groupby(data).count()
        diario = chuva.groupby(data).sum(min_count=1)
        diario[horas < 20] = np.nan       # dia com menos de 20 h medidas fica ausente, não zero
        partes.append(diario)
    return pd.concat(partes).sort_index()


def casos_por_dia() -> pd.Series:
    r = pd.read_parquet(RELATOS)
    r = r[~r.duplicado_conteudo.astype(bool)]
    return r.groupby(pd.to_datetime(r.data_acao).dt.normalize()).size()


def estimar(dias: pd.DataFrame, exposicao: str) -> dict:
    d = dias.dropna(subset=[exposicao]).copy()
    d["estrato"] = d.index.strftime("%Y-%m") + "|" + d.index.dayofweek.astype(str)
    agr = d.groupby("estrato").agg(casos=("casos", "sum"), var=(exposicao, "var"), n=("casos", "size"))
    informativos = agr.index[(agr.casos > 0) & (agr.n > 1) & (agr["var"].fillna(0) > 0)]
    d = d[d.estrato.isin(informativos)].sort_values("estrato")
    X = d[[exposicao]].to_numpy(float)
    y = d.casos.to_numpy(float)
    inicio = _blocos(pd.factorize(d.estrato, sort=False)[0])
    beta = ajustar(X, y, inicio)[0]
    comeco, tamanho = inicio[:-1], np.diff(inicio)
    rng = np.random.default_rng(SEMENTE)
    reps = []
    for _ in range(B):
        ordem = rng.integers(0, len(comeco), len(comeco))
        sz = tamanho[ordem]
        fim = np.cumsum(sz)
        idx = np.arange(int(fim[-1])) + np.repeat(comeco[ordem] - np.concatenate([[0], fim[:-1]]), sz)
        reps.append(ajustar(X[idx], y[idx], np.concatenate([[0], fim]))[0])
    lo, hi = np.percentile(np.exp(POR_MM * np.asarray(reps)), [2.5, 97.5])
    return {"razao_de_taxas_por_10mm": float(np.exp(POR_MM * beta)), "ic95": [float(lo), float(hi)],
            "ic_exclui_1": bool(lo > 1 or hi < 1), "estratos_informativos": int(len(informativos)),
            "casos_em_estrato_informativo": int(y.sum()), "dias": int(len(d))}


def montar(chuva: pd.Series, casos: pd.Series) -> pd.DataFrame:
    dias = pd.DataFrame({"chuva_dia": chuva})
    dias["chuva_48h"] = chuva + chuva.shift(1)
    dias["chuva_futuro_30d"] = chuva.shift(-FUTURO)
    dias["casos"] = casos.reindex(dias.index).fillna(0).astype(int)
    return dias


def main():
    casos = casos_por_dia()
    saida = {"pergunta": "a chuva do dia aumenta os relatos de alagamento no mesmo lugar (Recife)?",
             "desenho": "autopareado no tempo, estratos ano-mes x dia da semana, Poisson condicional",
             "regra": "sustentado se IC95% da exposicao exclui 1 e o do controle futuro inclui 1",
             "relatos_sem_duplicata": int(casos.sum()), "dias_com_relato": int(len(casos)),
             "periodo_relatos": [str(casos.index.min().date()), str(casos.index.max().date())], "series": {}}
    era5 = montar(serie_era5(), casos).loc["2014-01-01":"2024-12-31"]
    inm = montar(serie_inmet(), casos).loc["2020-01-01":"2023-12-31"]
    for nome, dias in (("era5_2014_2024", era5), ("inmet_a301_2020_2023", inm)):
        r = {e: estimar(dias, e) for e in ("chuva_dia", "chuva_48h", "chuva_futuro_30d")}
        principal, controle = r["chuva_dia"], r["chuva_futuro_30d"]
        r["leitura"] = ("sustentado" if principal["ic95"][0] > 1 and controle["ic95"][0] <= 1 <= controle["ic95"][1]
                        else "nao sustentado")
        r["relatos_no_periodo"] = int(dias.casos.sum())
        saida["series"][nome] = r
        print(nome, json.dumps(r, default=float), flush=True)
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "f02_chuva_gatilho_recife.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
