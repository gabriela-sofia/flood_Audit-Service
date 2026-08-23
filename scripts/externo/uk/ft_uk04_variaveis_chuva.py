"""
FT-UK-04 -- Anexa as duas features de chuva do v12 a amostra do piloto.

ENTRADAS:
  local_runs/ft-uk-03-amostra/amostra_candidata_uk_v1.csv
  local_runs/aq-chirps3/{rnl,sat}/chirps3_*_{AAAA-MM}.nc

AS DUAS FEATURES, e como cada uma e definida:

  rain_max_24h            precipitacao do dia do evento, no pixel CHIRPS que
                          contem o ponto.
  rain_decay_index_api    Antecedent Precipitation Index. Soma ponderada da
                          chuva dos dias anteriores, com peso k^i decaindo
                          geometricamente:
                              API = soma_{i=1..N} k^i * P(dia - i)
                          k = 0,9 e N = 14, iguais aos do v12, para que a
                          variavel signifique a mesma coisa nas duas regioes.

O PROBLEMA DO NEGATIVO SEM DATA, e a decisao tomada:
positivo tem data de evento; negativo nao tem. Atribuir chuva ao negativo
exige escolher uma data, e essa escolha nao e neutra:

  - dar ao negativo a chuva de um dia seco produziria separacao perfeita e
    falsa -- o modelo aprenderia "choveu = enchente", que e circular;
  - dar a media climatologica apagaria a variabilidade que o positivo tem.

A decisao aqui e PAREAMENTO TEMPORAL: cada negativo recebe a data de um evento
positivo sorteado, do mesmo estrato de cobertura do solo. Assim positivo e
negativo compartilham a mesma distribuicao de datas, e a chuva deixa de ser um
atalho para distinguir as classes. O que sobra de sinal na chuva e interacao
com o terreno, que e o que interessa.

Essa decisao esta registrada tambem em `docs/metodologia_cientifica/` e precisa
aparecer no artigo -- e uma escolha de desenho, nao um detalhe de implementacao.

NAO faz: nao treina, nao avalia, nao promove rotulo.

Uso:
    python scripts/externo/ft_uk04_variaveis_chuva.py
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
AMOSTRA = RUNS / "ft-uk-03-amostra" / "amostra_candidata_uk_v1.csv"
CHIRPS = RUNS / "aq-chirps3"
OUT = RUNS / "ft-uk-04-chuva"

K_DECAY = 0.9      # igual ao v12
N_DIAS = 14        # igual ao v12
SEMENTE = 20260807


def carregar_serie(flavor: str):
    """Abre todos os NetCDF mensais como um unico dataset."""
    import xarray as xr

    arquivos = sorted((CHIRPS / flavor).glob("*.nc"))
    if not arquivos:
        raise FileNotFoundError(
            f"nenhum NetCDF em {CHIRPS/flavor} -- rode aq_chirps3_aoi.py --baixar"
        )
    # `open_mfdataset` exige dask; abrir e concatenar direto evita a dependencia
    # e, no volume desta AOI (36x37 pixels por dia), cabe folgado em memoria.
    partes = [xr.open_dataset(a) for a in arquivos]
    return xr.concat(partes, dim="time").sortby("time")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEMENTE)
    print("===FT-UK-04===")

    df = pd.read_csv(AMOSTRA, parse_dates=["start_date"])
    print(f"AMOSTRA n={len(df):,} pos={int((df.rotulo==1).sum()):,} "
          f"neg={int((df.rotulo==0).sum()):,}")

    # --- pareamento temporal do negativo ---
    datas_por_wc = (df[df.rotulo == 1]
                    .groupby("worldcover")["start_date"]
                    .apply(lambda s: s.dropna().dt.date.tolist())
                    .to_dict())
    todas = df.loc[df.rotulo == 1, "start_date"].dropna().dt.date.tolist()

    def sortear(wc):
        pool = datas_por_wc.get(wc) or todas
        return pool[rng.integers(len(pool))]

    faltando = df["start_date"].isna()
    df.loc[faltando, "start_date"] = [
        pd.Timestamp(sortear(w)) for w in df.loc[faltando, "worldcover"]
    ]
    df["data_pareada"] = faltando
    print(f"NEGATIVOS_COM_DATA_PAREADA={int(faltando.sum()):,}")

    resultados = {}
    for flavor in ("rnl", "sat"):
        try:
            ds = carregar_serie(flavor)
        except FileNotFoundError as exc:
            print(f"[{flavor}] AUSENTE: {exc}")
            continue
        print(f"[{flavor}] serie carregada: {dict(ds.sizes)}")

        precip = ds["precip"]
        lats = precip["y"].values
        lons = precip["x"].values
        iy = np.abs(lats[None, :] - df["lat"].values[:, None]).argmin(axis=1)
        ix = np.abs(lons[None, :] - df["lon"].values[:, None]).argmin(axis=1)

        tempos = pd.to_datetime(precip["time"].values).date
        idx_tempo = {d: i for i, d in enumerate(tempos)}
        cubo = precip.values  # (time, y, x) -- AOI pequena, cabe em memoria

        rmax = np.full(len(df), np.nan, dtype="float32")
        api = np.full(len(df), np.nan, dtype="float32")
        sem_data = 0
        for n, (d, yy, xx) in enumerate(zip(df["start_date"].dt.date, iy, ix)):
            t = idx_tempo.get(d)
            if t is None:
                sem_data += 1
                continue
            rmax[n] = cubo[t, yy, xx]
            soma = 0.0
            for i in range(1, N_DIAS + 1):
                ti = idx_tempo.get(d - dt.timedelta(days=i))
                if ti is not None:
                    v = cubo[ti, yy, xx]
                    if np.isfinite(v):
                        soma += (K_DECAY ** i) * float(v)
            api[n] = soma

        df[f"rain_max_24h_{flavor}"] = rmax
        df[f"rain_decay_index_api_{flavor}"] = api
        ok = int(np.isfinite(rmax).sum())
        print(f"[{flavor}] com_chuva={ok:,}/{len(df):,} sem_data_na_serie={sem_data:,}")
        print(f"[{flavor}] rain_max_24h  pos={np.nanmedian(rmax[df.rotulo==1]):.2f} "
              f"neg={np.nanmedian(rmax[df.rotulo==0]):.2f}")
        print(f"[{flavor}] api           pos={np.nanmedian(api[df.rotulo==1]):.2f} "
              f"neg={np.nanmedian(api[df.rotulo==0]):.2f}")
        resultados[flavor] = {"com_chuva": ok, "sem_data": sem_data}
        ds.close()

    saida = OUT / "amostra_candidata_uk_v2_com_chuva.csv"
    df.to_csv(saida, index=False)

    # concordancia entre os dois flavors -- a analise de sensibilidade
    if "rnl" in resultados and "sat" in resultados:
        a = df["rain_max_24h_rnl"]
        b = df["rain_max_24h_sat"]
        m = a.notna() & b.notna()
        if m.sum() > 10:
            print(f"CORRELACAO_RNL_SAT_rain_max_24h={a[m].corr(b[m]):.3f} (n={int(m.sum()):,})")
            print(f"DIFERENCA_MEDIANA={float((a[m]-b[m]).median()):.3f}mm")

    (OUT / "resumo.json").write_text(json.dumps({
        "n": int(len(df)), "k_decay": K_DECAY, "n_dias": N_DIAS,
        "semente": SEMENTE,
        "negativos_com_data_pareada": int(faltando.sum()),
        "flavors": resultados,
        "decisao_pareamento_temporal": (
            "Negativo nao tem data propria. Recebe a data de um evento positivo "
            "sorteado no mesmo estrato de cobertura do solo, para que positivo e "
            "negativo compartilhem a distribuicao de datas e a chuva nao vire "
            "atalho de classificacao."
        ),
        "aviso": "Amostra CANDIDATA. Nenhum modelo treinado.",
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SAIDA={saida.name} TEMPO={round(time.time()-t0,1)}s")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
