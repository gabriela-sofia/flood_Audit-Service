"""
CEMS-04 -- Fila de aquisicao construida sobre a GEOMETRIA REAL das AOIs.

O QUE ESTE SCRIPT DESTRAVA:

ate 10/08/2026 a selecao de analogos era feita lendo o DEM no CENTROIDE DA
ATIVACAO, porque nao se conhecia forma de obter a geometria das AOIs sem baixar
o pacote vetorial inteiro. O resultado foi que cinco dos seis analogos baixados
sao planicies de inundacao: o primeiro colocado do ranking de Petropolis
(EMSR867, Madagascar) tem centroide no planalto central e AOIs no litoral, a
4,7 e 11,5 m de altitude.

O endpoint abaixo resolve isso:

    dashboard-api/aois/?activation__code=EMSR867

Ele devolve o POLIGONO de cada AOI. Sao 1.109 AOIs em todo o catalogo publico,
771 delas em ativacoes de Flood/Storm/Mass movement, obtidas em UMA requisicao.
Com isso a analogia passa a ser medida onde o fenomeno foi mapeado, e nao onde
a media aritmetica das AOIs calhou de cair.

Observacao sobre o parametro: `?activation=` e `?code=` sao ACEITOS E IGNORADOS
pela API -- devolvem as 1.109 AOIs sem filtrar. Somente `?activation__code=`
filtra. Um filtro silenciosamente ignorado e a mesma familia de erro que ja
custou caro neste projeto (paginacao do EA, `offset` ignorado), por isso o
script confere que toda linha devolvida pertence ao codigo pedido.

O QUE CONTINUA MANUAL, e por que:
o endpoint `products/` responde 403 -- exige autenticacao. O pacote vetorial
(o zip com `observedEventA`, `areaOfInterestA`, `imageFootprintA`) precisa ser
baixado do portal a mao. Este script produz a lista exata do que baixar, com
link, para que o trabalho manual seja minimo e sem desperdicio.

CRITERIO DE ACEITACAO -- declarado antes da leitura, nao ajustavel por resultado:
    relevo_local >= 400 m  E  frac_gt15 >= 0.25
Petropolis: 1.152 m / 81%. Curitiba, com a MESMA elevacao, da 100 m / 3% --
por isso o criterio e sobre amplitude altimetrica e nao sobre elevacao.

NAO faz: nao baixa raster de evento, nao extrai feature, nao promove rotulo,
nao treina, nao afrouxa o criterio para caber mais AOI.

Uso:
    python scripts/externo/cems04_fila_de_aquisicao.py --aois       # 1 requisicao
    python scripts/externo/cems04_fila_de_aquisicao.py --perfilar   # DEM, resumivel
    python scripts/externo/cems04_fila_de_aquisicao.py --fila       # o que baixar
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import urllib.request
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
OUT = RUNS / "cems-04-fila"
API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api"
PORTAL = "https://rapidmapping.emergency.copernicus.eu"

CATEGORIAS = ("Flood", "Storm", "Mass movement")

PETROPOLIS = {"elev_med": 906.8, "relevo_local": 1152.2, "slope_med": 24.8,
              "slope_p90": 37.8, "frac_gt15": 0.81, "frac_gt25": 0.49}

MIN_RELEVO_LOCAL = 400.0
MIN_FRAC_GT15 = 0.25
MIN_AOIS_PARA_VALER_O_DOWNLOAD = 2   # 1 AOI = 1 grupo de CV; nao compensa

JANELA = 0.05      # ~11 km de lado
THREADS = 24


# ------------------------------------------------------------------ utilitarios

def _get(url: str, tentativas: int = 4):
    ultimo = None
    for k in range(tentativas):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except Exception as exc:  # noqa: BLE001
            ultimo = exc
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"{url} -> {type(ultimo).__name__}: {ultimo}")


def centroide_wkt(wkt: str | None):
    """Centroide simples do anel do poligono. Basta: a janela do DEM tem 11 km."""
    pts = re.findall(r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", wkt or "")
    if not pts:
        return None, None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return sum(ys) / len(ys), sum(xs) / len(xs)


def nome_tile(lat: float, lon: float) -> str:
    la, lo = math.floor(lat), math.floor(lon)
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"


def perfil(lat: float, lon: float, meia: float = JANELA) -> dict | None:
    import numpy as np
    import rioxarray

    warnings.filterwarnings("ignore")
    t = nome_tile(lat, lon)
    da = rioxarray.open_rasterio(
        f"https://copernicus-dem-30m.s3.amazonaws.com/{t}/{t}.tif",
        chunks=None, masked=True, lock=False)
    da = da.rio.clip_box(minx=lon - meia, miny=lat - meia,
                         maxx=lon + meia, maxy=lat + meia).load()
    z = np.asarray(da.squeeze(), dtype="float64")
    if z.size < 100 or not np.isfinite(z).any():
        return None
    gy, gx = np.gradient(z, 30.0, 30.0 * max(math.cos(math.radians(lat)), 0.1))
    s = np.degrees(np.arctan(np.hypot(gx, gy)))
    zf, sf = z[np.isfinite(z)], s[np.isfinite(s)]
    return {"elev_med": float(np.median(zf)),
            "relevo_local": float(np.percentile(zf, 95) - np.percentile(zf, 5)),
            "slope_med": float(np.median(sf)),
            "slope_p90": float(np.percentile(sf, 90)),
            "frac_gt15": float((sf > 15).mean()),
            "frac_gt25": float((sf > 25).mean())}


def aprovado(p: dict) -> bool:
    return p["relevo_local"] >= MIN_RELEVO_LOCAL and p["frac_gt15"] >= MIN_FRAC_GT15


def semelhanca(p: dict) -> float:
    return (0.4 * (1 - min(abs(p["relevo_local"] - PETROPOLIS["relevo_local"]), 1200) / 1200)
            + 0.4 * (1 - min(abs(p["slope_med"] - PETROPOLIS["slope_med"]), 25) / 25)
            + 0.2 * (1 - abs(p["frac_gt15"] - PETROPOLIS["frac_gt15"])))


# ------------------------------------------------------------------ etapa 1

def baixar_aois() -> int:
    """Uma requisicao. 1.109 AOIs com poligono."""
    OUT.mkdir(parents=True, exist_ok=True)
    print("===CEMS04_AOIS===")
    d = _get(f"{API}/aois/?limit=5000")
    todas = d.get("results") or []
    if len(todas) != d.get("count"):
        print(f"ABORTADO: API disse count={d.get('count')} e devolveu {len(todas)}. "
              f"Nao vou trabalhar com inventario truncado.")
        return 2
    print(f"AOIS_TOTAIS={len(todas)}")

    if not INV.exists():
        print(f"ABORTADO: {INV} ausente. Rode n1c_cems_activations.py antes.")
        return 1
    raw = json.loads(INV.read_text(encoding="utf-8"))
    meta = {r["code"]: r for r in raw if str(r.get("category")) in CATEGORIAS}
    sel = [a for a in todas if a.get("activationCode") in meta]
    print(f"AOIS_EM_{'/'.join(CATEGORIAS)}={len(sel)} ativacoes={len(meta)}")

    (OUT / "aois_catalogo.json").write_text(
        json.dumps(sel, ensure_ascii=False), encoding="utf-8")
    (OUT / "ativacoes_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"GRAVADO={OUT/'aois_catalogo.json'}")
    print("===END===")
    return 0


# ------------------------------------------------------------------ etapa 2

def perfilar() -> int:
    """Relevo por AOI. Resumivel: pula o que ja esta no CSV."""
    import pandas as pd

    alvo = OUT / "aois_catalogo.json"
    if not alvo.exists():
        print(f"ABORTADO: {alvo} ausente. Rode --aois antes.")
        return 1
    sel = json.loads(alvo.read_text(encoding="utf-8"))

    destino = OUT / "relevo_por_aoi.csv"
    feitos: set = set()
    if destino.exists():
        try:
            ja = pd.read_csv(destino)
            feitos = set(zip(ja["code"].astype(str), ja["aoi_num"].astype(str)))
        except Exception:  # noqa: BLE001
            feitos = set()
    pend = [a for a in sel
            if (str(a.get("activationCode")), str(a.get("number"))) not in feitos]
    print(f"===CEMS04_PERFILAR=== total={len(sel)} ja_feitos={len(feitos)} pendentes={len(pend)}")
    if not pend:
        print("NADA_A_FAZER")
        print("===END===")
        return 0

    falhas: list[str] = []

    def uma(a):
        la, lo = centroide_wkt(a.get("extent"))
        if la is None:
            falhas.append(f"{a.get('activationCode')}#{a.get('number')} sem_geometria")
            return None
        try:
            p = perfil(la, lo)
        except Exception as exc:  # noqa: BLE001
            falhas.append(f"{a.get('activationCode')}#{a.get('number')} {type(exc).__name__}")
            return None
        if not p:
            falhas.append(f"{a.get('activationCode')}#{a.get('number')} dem_vazio")
            return None
        p.update(code=a.get("activationCode"), aoi_nome=a.get("name"),
                 aoi_num=a.get("number"), lat=round(la, 4), lon=round(lo, 4))
        p["ingreme"] = aprovado(p)
        p["sim_petropolis"] = round(semelhanca(p), 4)
        return p

    # blocos: grava a cada 60 para que interrupcao nao perca trabalho
    t0 = time.time()
    for i in range(0, len(pend), 60):
        bloco = pend[i:i + 60]
        with ThreadPoolExecutor(THREADS) as ex:
            res = [x for x in ex.map(uma, bloco) if x]
        if res:
            modo = "a" if destino.exists() else "w"
            pd.DataFrame(res).to_csv(destino, mode=modo,
                                     header=(modo == "w"), index=False)
        print(f"  {min(i+60, len(pend))}/{len(pend)} "
              f"({(time.time()-t0)/60:.1f}min, falhas={len(falhas)})", flush=True)

    d = pd.read_csv(destino)
    print(f"PERFILADAS={len(d)} INGREMES={int(d.ingreme.sum())} FALHAS={len(falhas)}")
    for f in falhas[:10]:
        print(f"   FALHA {f}")
    print("===END===")
    return 0


# ------------------------------------------------------------------ etapa 3

def fila() -> int:
    """A lista do que baixar a mao, ordenada, com link."""
    import pandas as pd

    destino = OUT / "relevo_por_aoi.csv"
    if not destino.exists():
        print(f"ABORTADO: {destino} ausente. Rode --perfilar antes.")
        return 1
    d = pd.read_csv(destino)
    meta = json.loads((OUT / "ativacoes_meta.json").read_text(encoding="utf-8"))

    ing = d[d.ingreme]
    print("===CEMS04_FILA===")
    print(f"AOIS_PERFILADAS={len(d)} INGREMES={len(ing)} "
          f"({100*len(ing)/max(len(d),1):.1f}%)")
    print(f"CRITERIO relevo_local>={MIN_RELEVO_LOCAL:.0f}m E frac_gt15>={MIN_FRAC_GT15}")

    g = (ing.groupby("code")
            .agg(aois_ingremes=("aoi_num", "size"),
                 relevo_med=("relevo_local", "median"),
                 slope_med=("slope_med", "median"),
                 frac_gt15=("frac_gt15", "median"),
                 sim=("sim_petropolis", "max"))
            .reset_index())
    g["aois_totais"] = g.code.map(lambda c: int((d.code == c).sum()))
    g["nome"] = g.code.map(lambda c: (meta.get(c) or {}).get("name"))
    g["paises"] = g.code.map(lambda c: ", ".join((meta.get(c) or {}).get("countries") or []))
    g["data"] = g.code.map(lambda c: str((meta.get(c) or {}).get("eventTime"))[:10])
    g["url"] = PORTAL + "/" + g.code
    g = g[g.aois_ingremes >= MIN_AOIS_PARA_VALER_O_DOWNLOAD]
    g = g.sort_values(["aois_ingremes", "sim"], ascending=False)
    g.to_csv(OUT / "fila_de_download.csv", index=False)

    print(f"\nATIVACOES_COM_>={MIN_AOIS_PARA_VALER_O_DOWNLOAD}_AOIS_INGREMES={len(g)}")
    cols = ["code", "paises", "aois_ingremes", "aois_totais",
            "relevo_med", "slope_med", "sim"]
    print(g.head(20)[cols].round(2).to_string(index=False))

    print("\n--- BAIXE ESTES PACOTES VETORIAIS A MAO (products/ exige autenticacao) ---")
    for _, r in g.head(8).iterrows():
        print(f"  {r.code}  {r.aois_ingremes:>2}/{r.aois_totais:<3} AOIs ingremes  "
              f"{str(r.paises)[:28]:28s} {r.url}")
    print("\nDepois de baixar, para cada zip:")
    print("  python scripts/externo/cems02_pipeline_ativacao.py --zip <zip> --codigo <EMSRxxx>")
    print("  python scripts/suscetibilidade/ds01_montar_multirregiao.py")
    print("  python scripts/externo/cems03_relevo_e_analogia.py --verificar")
    print("===END===")
    return 0


def main() -> int:
    if "--aois" in sys.argv:
        return baixar_aois()
    if "--perfilar" in sys.argv:
        return perfilar()
    if "--fila" in sys.argv:
        return fila()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
