"""
CEMS-01 -- Encontra ativacoes analogas as regioes do projeto, por caracteristica.

O PROBLEMA QUE RESOLVE:
a consulta ao portal CEMS filtrada por Brasil (2026-08-07) devolveu apenas
quatro ativacoes no pais inteiro, e nenhuma cobre Recife, Curitiba ou
Petropolis. A rota direta esta encerrada.

A rota indireta e o que este script implementa: se nao existe ativacao NAS
regioes, procura-se ativacao COM AS CARACTERISTICAS das regioes. O objetivo
nao e substituir dado local -- e ensinar ao modelo o que cada coisa do evento
e, usando eventos com estrutura fisica comparavel e, sobretudo, com a
separacao de mecanismo que o Brasil nao publica.

PERFIL DE CADA REGIAO (valores de referencia, nao estimados aqui):

  Recife      -8,05 / -34,88   ~10 m    costeiro, plano, tropical umido
                                        fenomeno: pluvial urbano / drenagem
  Curitiba   -25,43 / -49,27   ~930 m   planalto interior, suave ondulado,
                                        subtropical; fenomeno: fluvial urbano
  Petropolis -22,51 / -43,18   ~840 m   serra, montanhoso, subtropical de
                                        altitude; fenomeno: flash flood COM
                                        movimento de massa

CRITERIOS DE PONTUACAO, e por que cada um:

  clima      diferenca de latitude absoluta. E o proxy mais barato de regime
             de precipitacao e sazonalidade. Nao substitui Koppen, mas separa
             tropical de temperado sem depender de dado externo.
  relevo     elevacao e declividade medianas numa janela ao redor do centroide,
             lidas do Copernicus DEM GLO-30 por COG (so na etapa --relevo, que
             exige rede). E o criterio que mais importa para Petropolis:
             flash flood em serra nao se parece com cheia de planicie.
  mecanismo  palavra-chave de movimento de massa no nome da ativacao. Bonus
             para Petropolis, penalidade para Recife e Curitiba, onde
             deslizamento seria confusao e nao analogia.
  riqueza    numero de AOIs. Mais AOI = mais area declarada como observada =
             mais negativo por observacao.

LIMITACAO HONESTA: a presenca efetiva de poligonos `6-Mass Movement` so e
conhecida depois de baixar o pacote vetorial. O nome da ativacao e um indicio,
nao uma garantia. Por isso a saida e uma FILA DE VERIFICACAO ordenada, nao uma
selecao final.

NAO faz: nao baixa pacote vetorial, nao promove rotulo, nao treina.

Uso:
    python scripts/externo/cems01_analogos_por_regiao.py
    python scripts/externo/cems01_analogos_por_regiao.py --relevo   # exige rede
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
INV = RUNS / "n1c-cems" / "activations_flood.json"
OUT = RUNS / "cems-01-analogos"

REGIOES = {
    "Recife": {
        "lat": -8.05, "lon": -34.88, "elev": 10, "slope": 1.5,
        "fenomeno": "pluvial urbano costeiro, terreno plano",
        "quer_massa": False,
    },
    "Curitiba": {
        "lat": -25.43, "lon": -49.27, "elev": 930, "slope": 3.0,
        "fenomeno": "fluvial urbano em planalto, relevo suave ondulado",
        "quer_massa": False,
    },
    "Petropolis": {
        "lat": -22.51, "lon": -43.18, "elev": 840, "slope": 18.0,
        "fenomeno": "flash flood em serra COM movimento de massa",
        "quer_massa": True,
    },
}

PALAVRAS_MASSA = ("landslide", "mudslide", "mud flow", "debris", "mass movement",
                  "frana", "glissement", "deslizamento", "erdrutsch")

JANELA_GRAUS = 0.15   # ~16 km de lado ao redor do centroide


def centroide(wkt: str | None):
    m = re.search(r"POINT\s*\(\s*([-\d.eE]+)\s+([-\d.eE]+)\s*\)", wkt or "")
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def nome_tile_dem(lat: float, lon: float) -> str:
    import math
    la, lo = math.floor(lat), math.floor(lon)
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"


def relevo_do_centroide(lat: float, lon: float) -> tuple[float | None, float | None]:
    """Le uma janela do DEM global por COG e devolve (elev_mediana, slope_mediana)."""
    import numpy as np
    import rioxarray

    t = nome_tile_dem(lat, lon)
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{t}/{t}.tif"
    try:
        da = rioxarray.open_rasterio(url, chunks=None, masked=True, lock=False)
        da = da.rio.clip_box(minx=lon - JANELA_GRAUS, miny=lat - JANELA_GRAUS,
                             maxx=lon + JANELA_GRAUS, maxy=lat + JANELA_GRAUS).load()
        z = np.asarray(da.squeeze(), dtype="float64")
        if z.size < 9 or not np.isfinite(z).any():
            return None, None
        # passo metrico aproximado na latitude do ponto
        passo_y = 30.0
        passo_x = 30.0 * max(np.cos(np.radians(lat)), 0.1)
        gy, gx = np.gradient(z, passo_y, passo_x)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        return float(np.nanmedian(z)), float(np.nanmedian(slope))
    except Exception:  # noqa: BLE001
        return None, None


def main() -> int:
    com_relevo = "--relevo" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===CEMS01_ANALOGOS===")

    if not INV.exists():
        print(f"ABORTADO: rode n1c_cems_ativacoes.py antes ({INV} ausente)")
        return 1
    fl = json.loads(INV.read_text(encoding="utf-8"))
    print(f"ATIVACOES_INUNDACAO={len(fl)}")

    linhas = []
    for r in fl:
        lon, lat = centroide(r.get("centroid"))
        if lat is None:
            continue
        nome = str(r.get("name") or "")
        linhas.append({
            "code": r.get("code"), "name": nome,
            "lat": lat, "lon": lon,
            "data": str(r.get("eventTime"))[:10],
            "n_aois": int(r.get("n_aois") or 0),
            "n_products": int(r.get("n_products") or 0),
            "paises": ", ".join(r.get("countries") or []),
            "indicio_massa": any(p in nome.lower() for p in PALAVRAS_MASSA),
        })
    df = pd.DataFrame(linhas)
    print(f"COM_CENTROIDE={len(df)} indicio_de_massa={int(df.indicio_massa.sum())}")

    if com_relevo:
        print("--- lendo relevo do DEM global (COG) ---")
        elevs, slopes = [], []
        for i, r in df.iterrows():
            e, s = relevo_do_centroide(r["lat"], r["lon"])
            elevs.append(e)
            slopes.append(s)
            if (i + 1) % 20 == 0:
                print(f"   {i+1}/{len(df)}", flush=True)
        df["elev_med"] = elevs
        df["slope_med"] = slopes
        ok = int(df["elev_med"].notna().sum())
        print(f"RELEVO_OBTIDO={ok}/{len(df)}")
    else:
        df["elev_med"] = None
        df["slope_med"] = None
        print("RELEVO=NAO_LIDO (rode com --relevo, exige rede)")

    resumo = {}
    for regiao, p in REGIOES.items():
        d = df.copy()
        # clima: 0 a 1, penalizando diferenca de latitude absoluta
        d["s_clima"] = 1 - (d["lat"].abs() - abs(p["lat"])).abs().clip(0, 40) / 40
        # relevo, se disponivel
        if d["elev_med"].notna().any():
            d["s_elev"] = 1 - (d["elev_med"] - p["elev"]).abs().clip(0, 1500) / 1500
            d["s_slope"] = 1 - (d["slope_med"] - p["slope"]).abs().clip(0, 25) / 25
        else:
            d["s_elev"] = 0.5
            d["s_slope"] = 0.5
        # mecanismo
        d["s_mec"] = d["indicio_massa"].map(
            {True: 1.0, False: 0.35} if p["quer_massa"] else {True: 0.2, False: 1.0})
        # riqueza de AOI (log suave)
        import numpy as np
        d["s_aoi"] = np.log1p(d["n_aois"]) / np.log1p(max(d["n_aois"].max(), 1))

        d["escore"] = (0.30 * d["s_clima"] + 0.20 * d["s_elev"] + 0.20 * d["s_slope"]
                      + 0.20 * d["s_mec"] + 0.10 * d["s_aoi"])
        d = d.sort_values("escore", ascending=False)
        d["url"] = "https://rapidmapping.emergency.copernicus.eu/" + d["code"].astype(str)
        d.to_csv(OUT / f"analogos_{regiao.lower()}.csv", index=False)

        print(f"---{regiao}--- ({p['fenomeno']})")
        for _, r in d.head(6).iterrows():
            rel = ""
            if pd.notna(r["elev_med"]):
                rel = f" elev={r['elev_med']:.0f}m decl={r['slope_med']:.1f}deg"
            massa = " [indicio de massa]" if r["indicio_massa"] else ""
            print(f"  {r.escore:.3f} {r['code']} {str(r['name'])[:48]:48s} "
                  f"lat={r['lat']:+.1f} aois={r['n_aois']}{rel}{massa}")
        resumo[regiao] = d.head(6)[
            ["code", "name", "lat", "lon", "data", "n_aois", "escore", "url"]
        ].to_dict("records")

    (OUT / "fila_de_verificacao.json").write_text(
        json.dumps({
            "gerado_em": time.strftime("%Y-%m-%d %H:%M"),
            "com_relevo": com_relevo,
            "perfis": REGIOES,
            "ranking": resumo,
            "aviso": ("FILA DE VERIFICACAO, nao selecao final. A presenca de "
                      "poligonos '6-Mass Movement' so e confirmada apos baixar "
                      "o pacote vetorial da ativacao."),
        }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"TEMPO={round(time.time()-t0,1)}s")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
