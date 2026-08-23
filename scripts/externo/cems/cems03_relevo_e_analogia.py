"""
CEMS-03 -- Analogia de terreno medida onde o fenomeno acontece.

POR QUE ESTE SCRIPT SUBSTITUI O RANKING DO cems01:

o cems01 pontuava cada ativacao lendo o DEM numa janela de ~16 km em torno do
CENTROIDE DA ATIVACAO. O centroide e a media das AOIs. Quando a ativacao cobre
um pais inteiro, ele cai onde nao ha AOI nenhuma.

Verificacao de 09-10/08/2026: o primeiro colocado do ranking de Petropolis
(EMSR867, Madagascar) foi escolhido por um centroide com 513 m de relevo e
19 graus de declividade -- no planalto central. As AOIs efetivamente mapeadas
estao no LITORAL, a 4,7 e 11,5 m de altitude, com 70 m de relevo. Cinco dos
seis analogos baixados sao planicies de inundacao.

Petropolis tem 1.152 m de relevo local e 81% da area acima de 15 graus. Os
analogos entregavam de 23 a 113 m. O conjunto de treino nao continha terreno
ingreme -- nao porque analogo de serra nao exista, mas porque os escolhidos
nao eram analogos.

A REGRA QUE DECORRE, e que este script implementa:

    semelhanca medida no centroide e CANDIDATO.
    so a verificacao por AOI confirma analogia.

O centroide e geometria administrativa da ativacao, nao do fenomeno. A unidade
fisica e a AOI: e onde o analista olhou, onde o poligono foi desenhado e onde
os pontos sao amostrados.

POR QUE `relevo_local` E NAO `elevacao`:
Curitiba e Petropolis tem praticamente a MESMA elevacao mediana (919 m contra
907 m) e relevos que diferem por um fator de onze (100 m contra 1.152 m).
Elevacao nao distingue serra de planalto. A amplitude altimetrica distingue.

CRITERIO DE ACEITACAO, declarado antes da leitura (nao ajustavel por resultado):
    relevo local >= 400 m  E  fracao da area acima de 15 graus >= 0.25
Petropolis da 1.152 m / 81%. La Reunion da 1.108 m / 26%. Toda planicie do
conjunto atual fica de fora por larga margem.

NAO faz: nao baixa pacote vetorial, nao promove rotulo, nao treina, nao ajusta
o criterio para caber mais AOI.

Uso:
    python scripts/externo/cems03_relevo_e_analogia.py --screen
    python scripts/externo/cems03_relevo_e_analogia.py --verificar
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")
# NAO definir CPL_VSIL_CURL_USE_HEAD=NO -- quebra o /vsicurl. Ver aq_chirps3_v2.

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
INV = RUNS / "n1c-cems" / "activations_raw.json"
DATASET = RUNS / "ds-01-multirregiao" / "dataset_multirregiao_v1.csv"
OUT = RUNS / "cems-02-analogos-v2"

# Perfil de referencia, medido no Copernicus DEM GLO-30 em 2026-08-10.
PETROPOLIS = {
    "lat": -22.505, "lon": -43.178,
    "elev_med": 906.8, "relevo_local": 1152.2,
    "slope_med": 24.8, "slope_p90": 37.8, "frac_gt15": 0.81, "frac_gt25": 0.49,
}

# Criterio de aceitacao -- declarado antes de olhar o resultado.
MIN_RELEVO_LOCAL = 400.0
MIN_FRAC_GT15 = 0.25

JANELA = 0.08     # ~18 km de lado
THREADS = 16


def nome_tile(lat: float, lon: float) -> str:
    la, lo = math.floor(lat), math.floor(lon)
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"


def perfil(lat: float, lon: float, meia: float = JANELA) -> dict | None:
    """Perfil de relevo de CONTEXTO numa janela ao redor do ponto.

    Devolve None em vez de mascarar falha: sem DEM nao ha julgamento de
    analogia, e um valor inventado seria pior que a ausencia.
    """
    import numpy as np
    import rioxarray

    warnings.filterwarnings("ignore")
    t = nome_tile(lat, lon)
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{t}/{t}.tif"
    da = rioxarray.open_rasterio(url, chunks=None, masked=True, lock=False)
    da = da.rio.clip_box(minx=lon - meia, miny=lat - meia,
                         maxx=lon + meia, maxy=lat + meia).load()
    z = np.asarray(da.squeeze(), dtype="float64")
    if z.size < 100 or not np.isfinite(z).any():
        return None
    passo_y = 30.0
    passo_x = 30.0 * max(math.cos(math.radians(lat)), 0.1)
    gy, gx = np.gradient(z, passo_y, passo_x)
    s = np.degrees(np.arctan(np.hypot(gx, gy)))
    zf, sf = z[np.isfinite(z)], s[np.isfinite(s)]
    return {
        "elev_med": float(np.median(zf)),
        "relevo_local": float(np.percentile(zf, 95) - np.percentile(zf, 5)),
        "slope_med": float(np.median(sf)),
        "slope_p90": float(np.percentile(sf, 90)),
        "frac_gt15": float((sf > 15).mean()),
        "frac_gt25": float((sf > 25).mean()),
    }


def aprovado(p: dict) -> bool:
    return (p["relevo_local"] >= MIN_RELEVO_LOCAL
            and p["frac_gt15"] >= MIN_FRAC_GT15)


def semelhanca(p: dict) -> float:
    """0 a 1 contra o perfil de Petropolis. Serve para ORDENAR, nao para aprovar."""
    d_rel = abs(p["relevo_local"] - PETROPOLIS["relevo_local"])
    d_slo = abs(p["slope_med"] - PETROPOLIS["slope_med"])
    d_f15 = abs(p["frac_gt15"] - PETROPOLIS["frac_gt15"])
    return (0.4 * (1 - min(d_rel, 1200) / 1200)
            + 0.4 * (1 - min(d_slo, 25) / 25)
            + 0.2 * (1 - d_f15))


def centroide(wkt: str | None):
    m = re.search(r"POINT\s*\(\s*([-\d.eE]+)\s+([-\d.eE]+)\s*\)", wkt or "")
    return (float(m.group(2)), float(m.group(1))) if m else (None, None)


def screen() -> int:
    """Fila de CANDIDATOS. Le no centroide -- limitacao declarada no cabecalho."""
    import pandas as pd

    if not INV.exists():
        print(f"ABORTADO: {INV} ausente. Rode n1c_cems_activations.py antes.")
        return 1
    raw = json.loads(INV.read_text(encoding="utf-8"))
    alvos = [r for r in raw
             if str(r.get("category")) in ("Flood", "Storm", "Mass movement")]
    print(f"===CEMS03_SCREEN=== ativacoes={len(alvos)}")

    def uma(r):
        la, lo = centroide(r.get("centroid"))
        if la is None:
            return None
        try:
            p = perfil(la, lo)
        except Exception as exc:  # noqa: BLE001
            print(f"   FALHA_DEM {r['code']} {type(exc).__name__}")
            return None
        if not p:
            return None
        p.update(code=r["code"], name=r.get("name"), cat=str(r.get("category")),
                 paises=", ".join(r.get("countries") or []),
                 lat=round(la, 3), lon=round(lo, 3),
                 n_aois=r.get("n_aois"), data=str(r.get("eventTime"))[:10])
        p["sim_petropolis"] = round(semelhanca(p), 4)
        p["passaria_criterio"] = aprovado(p)
        return p

    with ThreadPoolExecutor(THREADS) as ex:
        res = [x for x in ex.map(uma, alvos) if x]

    d = pd.DataFrame(res).sort_values("sim_petropolis", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "screen_relevo_centroide_v2.csv", index=False)
    print(f"AVALIADAS={len(d)} CANDIDATOS_QUE_PASSARIAM={int(d.passaria_criterio.sum())}")
    cols = ["code", "cat", "paises", "n_aois", "relevo_local", "slope_med",
            "frac_gt15", "sim_petropolis"]
    print(d.head(12)[cols].round(2).to_string(index=False))
    print("LEMBRETE: isto e CANDIDATO. Rode --verificar depois de baixar.")
    print("===END===")
    return 0


def verificar() -> int:
    """Verificacao por AOI -- a unica que confirma analogia."""
    import pandas as pd

    if not DATASET.exists():
        print(f"ABORTADO: {DATASET} ausente. Rode ds01_montar_multirregiao.py antes.")
        return 1
    df = pd.read_csv(DATASET, low_memory=False)
    df = df[df["regiao"].str.startswith("analogo_")]
    if df.empty:
        print("ABORTADO: nenhuma regiao 'analogo_*' no dataset.")
        return 1

    alvos = []
    for reg, g in df.groupby("regiao"):
        for aoi, s in g.groupby("grupo_cv"):
            alvos.append((reg, aoi, float(s.lat.median()), float(s.lon.median()), len(s)))
    print(f"===CEMS03_VERIFICAR=== aois={len(alvos)}")

    def uma(a):
        reg, aoi, la, lo, n = a
        try:
            p = perfil(la, lo)
        except Exception as exc:  # noqa: BLE001
            print(f"   FALHA_DEM {aoi} {type(exc).__name__}")
            return None
        if not p:
            return None
        p.update(regiao=reg, aoi=aoi, lat=round(la, 3), lon=round(lo, 3), n_pontos=n)
        p["classe_relevo"] = "INGREME" if aprovado(p) else "PLANO_OU_ONDULADO"
        p["analogo_de_petropolis"] = aprovado(p)
        p["sim_petropolis"] = round(semelhanca(p), 4)
        return p

    with ThreadPoolExecutor(THREADS) as ex:
        res = [x for x in ex.map(uma, alvos) if x]
    if not res:
        print("ABORTADO: nenhuma AOI perfilada. Nao vou emitir veredito sem leitura.")
        return 2

    d = pd.DataFrame(res)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "verificacao_relevo_por_aoi.csv", index=False)

    ap = int(d.analogo_de_petropolis.sum())
    print(f"AOIS_PERFILADAS={len(d)} APROVADAS_COMO_INGREME={ap}")
    print(f"CRITERIO: relevo_local>={MIN_RELEVO_LOCAL:.0f}m E frac_gt15>={MIN_FRAC_GT15}")
    print("--- por ativacao ---")
    r = d.groupby("regiao").agg(
        aois=("aoi", "size"), ingremes=("analogo_de_petropolis", "sum"),
        relevo_med=("relevo_local", "median"), slope_med=("slope_med", "median"))
    print(r.round(1).to_string())
    print(f"--- referencia Petropolis: relevo={PETROPOLIS['relevo_local']:.0f}m "
          f"slope={PETROPOLIS['slope_med']:.1f} frac_gt15={PETROPOLIS['frac_gt15']:.2f}")
    if ap == 0:
        print("VEREDITO: NENHUMA AOI e analoga a Petropolis. O conjunto de treino "
              "nao contem terreno ingreme. Baixe as ativacoes da fila do --screen.")
    print("===END===")
    return 0


def main() -> int:
    if "--screen" in sys.argv:
        return screen()
    if "--verificar" in sys.argv:
        return verificar()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
