"""Auditoria da serie FloodNet 2020-2023: quando o sensor OBSERVOU, e o que observou.

O catalogo de eventos so tem positivo. A serie de ~1 min permite dizer, para
um sensor e um intervalo, se ele estava funcionando -- e so entao "sem lamina"
vira observacao, e nao ausencia de registro.

-------------------------------------------------------------------------
JANELA = sensor (deployment) x dia UTC
-------------------------------------------------------------------------

Funcionando, se TODAS valem (fixadas antes de olhar resultado):
  cobertura   leituras >= 80% das esperadas pelo intervalo mediano do sensor
  lacuna      nenhum intervalo entre leituras > 30 min
  quadros     perda de pacote (saltos do `f_cnt`) <= 20% no dia
  bateria     `batt_v` minimo >= 3,4 V
  tipo        `deploy_type` pluvial (instalacao de mare fica fora)

Classe da janela funcionando:
  lamina       `depth_proc_mm` >= 25 mm em pelo menos uma leitura
  sem_lamina   `depth_proc_mm` == 0 E `depth_filt_mm` == 0 em todas as leituras
               -> "sem lamina acima de 10 mm no ponto do sensor naquele dia";
               10 mm e o corte do proprio `depth_filt_mm`, nao ausencia absoluta
  desconhecido o resto: filtro e bruto discordam, ou lamina entre 10 e 25 mm

Janela que nao funcionou e `nao_observado`, nunca negativo.

Nao e mecanismo pluvial demonstrado: o sensor mede agua na rua; chuva, mare e
rompimento de rede chegam iguais. A separacao vem do `deploy_type` e, depois,
do pareamento com chuva -- que NAO e feito aqui. O FloodNet instala onde ja
houve alagamento: a amostra de lugares nao representa a cidade.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ENTRADA = REPO / "local_runs" / "floodnet-02-serie-2020-2023"
SAIDA = REPO / "local_runs" / "floodnet-03-auditoria"
COBERTURA_MIN = 0.80
LACUNA_MAX_MIN = 30.0
PERDA_QUADROS_MAX = 0.20
BATERIA_MIN_V = 3.4
LAMINA_MM = 25.0
CORTE_FILTRO_MM = 10.0
COLUNAS = ["time", "deployment_id", "depth_filt_mm", "depth_proc_mm", "batt_v", "f_cnt", "lat", "lon", "deploy_type"]


def janelas_do_arquivo(f: Path) -> pd.DataFrame:
    d = pd.read_csv(f, usecols=COLUNAS, dtype={"deployment_id": "category", "deploy_type": "category",
                                               "depth_filt_mm": "float32", "depth_proc_mm": "float32",
                                               "batt_v": "float32", "lat": "float32", "lon": "float32"})
    d["time"] = pd.to_datetime(d.time, utc=True, format="mixed")
    d = d.sort_values(["deployment_id", "time"])
    d["dia"] = d.time.dt.floor("D")
    d["dt_min"] = d.groupby("deployment_id").time.diff().dt.total_seconds() / 60
    salto = d.groupby("deployment_id").f_cnt.diff()
    # contador reinicia (salto negativo) em reinicio do aparelho: nao e perda
    d["perdidos"] = np.where((salto > 1) & (salto < 10_000), salto - 1, 0)
    g = d.groupby(["deployment_id", "dia"])
    j = g.agg(leituras=("time", "size"), maior_lacuna_min=("dt_min", "max"), perdidos=("perdidos", "sum"),
              bateria_min=("batt_v", "min"), proc_max=("depth_proc_mm", "max"), filt_max=("depth_filt_mm", "max"),
              lat=("lat", "first"), lon=("lon", "first"), tipo=("deploy_type", "first")).reset_index()
    return j


def classificar(j: pd.DataFrame, intervalo: pd.Series) -> pd.DataFrame:
    esperadas = 1440.0 / j.deployment_id.map(intervalo)
    j["cobertura"] = (j.leituras / esperadas).clip(upper=1.0)
    j["perda_quadros"] = j.perdidos / (j.perdidos + j.leituras)
    j["funcionando"] = ((j.cobertura >= COBERTURA_MIN) & (j.maior_lacuna_min.fillna(0) <= LACUNA_MAX_MIN)
                        & (j.perda_quadros <= PERDA_QUADROS_MAX) & (j.bateria_min >= BATERIA_MIN_V)
                        & (j.tipo.astype(str).str.lower() == "pluvial"))
    j["classe"] = np.select(
        [~j.funcionando, j.proc_max >= LAMINA_MM, (j.proc_max == 0) & (j.filt_max == 0)],
        ["nao_observado", "lamina", "sem_lamina"], default="desconhecido")
    return j


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(REPO / "scripts" / "servico"))
    from proveniencia import agora_utc
    SAIDA.mkdir(parents=True, exist_ok=True)
    arquivos = sorted(ENTRADA.glob("floodnet-flood-sensor-*.csv"))
    partes, intervalos = [], []
    for f in arquivos:
        j = janelas_do_arquivo(f)
        partes.append(j)
        print(f"  {f.name}: {len(j)} janelas", flush=True)
    j = pd.concat(partes, ignore_index=True)
    # dia partido entre dois arquivos mensais: soma leituras, pior lacuna/bateria, maior lamina
    j = j.groupby(["deployment_id", "dia"]).agg(
        leituras=("leituras", "sum"), maior_lacuna_min=("maior_lacuna_min", "max"), perdidos=("perdidos", "sum"),
        bateria_min=("bateria_min", "min"), proc_max=("proc_max", "max"), filt_max=("filt_max", "max"),
        lat=("lat", "first"), lon=("lon", "first"), tipo=("tipo", "first")).reset_index()
    # intervalo nominal do sensor: mediana das leituras por dia cheio, convertida em minutos
    intervalo = (1440.0 / j.groupby("deployment_id").leituras.quantile(0.9)).clip(lower=0.5)
    j = classificar(j, intervalo)
    j.to_parquet(SAIDA / "janelas_sensor_dia.parquet", index=False)
    resumo = {"tipo": "auditoria_floodnet", "quando_utc": agora_utc(), "arquivos": len(arquivos),
              "janelas": int(len(j)), "sensores": int(j.deployment_id.nunique()),
              "por_classe": j.classe.value_counts().to_dict(),
              "por_tipo": j.tipo.astype(str).value_counts().to_dict(),
              "sensores_com_sem_lamina": int(j[j.classe == "sem_lamina"].deployment_id.nunique()),
              "criterios": {"cobertura_min": COBERTURA_MIN, "lacuna_max_min": LACUNA_MAX_MIN,
                            "perda_quadros_max": PERDA_QUADROS_MAX, "bateria_min_v": BATERIA_MIN_V,
                            "lamina_mm": LAMINA_MM, "corte_do_filtro_mm": CORTE_FILTRO_MM},
              "nao_e": ("sem_lamina e 'sem agua acima de 10 mm no ponto do sensor naquele dia', nao seco absoluto; "
                        "nao separa mecanismo nem representa a cidade (instalacao dirigida por historico)")}
    (SAIDA / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
