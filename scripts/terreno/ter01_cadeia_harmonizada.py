"""
TER-01 -- A cadeia de terreno auditada, aplicada a todas as regioes.

O PROBLEMA QUE ISTO RESOLVE:

hoje o projeto tem DUAS cadeias diferentes produzindo colunas com o MESMO NOME.

  Recife (auditada, 27/07/2026)
      entrada  MDT real de 10 m
      HAND     DERIVADO: fill -> D-inf -> acumulacao -> canais no percentil 98
               -> elevacao acima do canal
      slope    WhiteboxTools v2.4.0
      TWI      SCA D-infinito + slope do WBT
      prova    r=1,0 e max_abs_diff=0,0 contra o raster de referencia

  Analogos CEMS (cems02, atual)
      entrada  Copernicus GLO-30, 30 m
      HAND     NAO DERIVADO -- baixado pronto do produto global glo-30-hand,
               que tem definicao propria de rede de drenagem
      slope    gradiente do numpy
      TWI      pysheds, com tan(slope) truncado em 0,05 grau

`hand_m` numa tabela e `hand_m` na outra nao sao a mesma grandeza. HAND e TWI
sao fortemente dependentes de escala e da definicao de canal; trocar de
instrumento no meio invalida qualquer comparacao de coeficiente entre regioes.

ONDE ISSO E FATAL, e nao apenas feio: Petropolis ja tem HAND derivado pela
cadeia WhiteboxTools. Treinar no HAND global e predizer no HAND-WBT e aplicar
o modelo a uma variavel diferente daquela que ele aprendeu. Harmonizar nao e
refino -- e pre-condicao da propagacao.

A DECISAO DE RESOLUCAO, e por que ela nao e uma concessao:

MDT de 10 m so existe no Brasil. Haiti, Equador, Honduras e Sri Lanka tem, no
melhor caso, 30 m. Como coeficiente so e comparavel entre regioes derivadas do
mesmo jeito, a cadeia roda a **30 m em todas**, inclusive em Recife.

Recife ja tem a versao de 10 m auditada. Derivar Recife tambem a 30 m e
compara-la a de 10 m MEDE o efeito da resolucao -- isto e, quanto de HAND e
TWI se perde nos analogos por nao existir MDT fino la. Isso deixa de ser
limitacao declarada e vira quantidade medida.

O QUE E AUDITADO EM CADA RODADA (mesmo padrao do recife_validation):
  run_manifest.json   versao do WBT, CRS, shape, pixel, parametros, sha256 de
                      cada saida, tempo, plataforma
  compare_*.json      contra a derivacao anterior: pearson, mean/max_abs_diff,
                      p95, p99, fracao abaixo de 1e-3, e se a grade bate

O LIMIAR DE CANAL E EM AREA, NAO EM PERCENTIL -- correcao de 12/08/2026:

a primeira versao deste script herdou o `stream_percentile = 98,0` do
recife_validation. Testada na EMSR789_AOI06, produziu HAND com mediana de
70,7 m contra 36,4 m do produto global (r=0,815). A causa nao e resolucao: e
que **o percentil e relativo as celulas da janela**.

    Recife          1.122,7 celulas a 10 m  =  0,112 km2 de area contribuinte
    EMSR789_AOI06     826,1 celulas a 30 m  =  0,743 km2

Seis vezes e meia de diferenca. O mesmo "percentil 98" define uma rede de
drenagem diferente em cada AOI, porque depende do tamanho e do relevo da
janela. Percentil serve para uma regiao unica; para comparar regioes, nao.

Como HAND e literalmente "altura acima do canal mais proximo", mudar a
definicao de canal muda a variavel. O limiar passa a ser fixado em AREA
CONTRIBUINTE, que e grandeza fisica e independe de resolucao e de janela:

    AREA_CANAL_KM2 = 0,1123   <- exatamente o limiar efetivo do recife_validation

Isso preserva continuidade com a cadeia auditada e a torna transferivel. O
percentil correspondente continua sendo calculado e gravado no manifesto,
como diagnostico: se ele variar muito entre AOIs, e sinal de que a densidade
de drenagem varia -- informacao, nao erro.

CONVENCOES FIXADAS (mudar qualquer uma invalida a comparabilidade):
  area contribuinte de canal   0,1123 km2  (herdada do recife_validation)
  piso de declividade          0,05 grau   (evita divisao por zero no TWI)
  TWI                          ln(SCA / tan(slope))
  reamostragem                 bilinear para DEM
  resolucao                    30 m no UTM local da AOI

NAO faz: nao treina, nao promove rotulo, nao altera dataset existente, nao
ajusta percentil de canal por regiao para melhorar resultado.

Uso:
    python scripts/terreno/ter01_cadeia_harmonizada.py --checar-ambiente
    python scripts/terreno/ter01_cadeia_harmonizada.py --aoi EMSR847_AOI31
    python scripts/terreno/ter01_cadeia_harmonizada.py --lote ingremes
    python scripts/terreno/ter01_cadeia_harmonizada.py --regiao recife_30m \
        --bbox -35.05,-8.15,-34.85,-7.95
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")
# NAO definir CPL_VSIL_CURL_USE_HEAD=NO -- quebra o /vsicurl. Ver aq_chirps3_v2.

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
OUT = RUNS / "ter-01-cadeia-harmonizada"
CACHE = RUNS / "cache-global" / "dem"
CATALOGO = RUNS / "cems-04-fila" / "aois_catalogo.json"
RELEVO = RUNS / "cems-02-analogos-v2" / "verificacao_relevo_por_aoi.csv"

# --- convencoes fixadas. Mudar invalida a comparabilidade entre regioes. ---
# Limiar de canal em AREA CONTRIBUINTE, nao em percentil. 0,1123 km2 e o limiar
# efetivo do recife_validation (1.122,7 celulas x 100 m2). Ver cabecalho.
AREA_CANAL_KM2 = 0.1123
PISO_SLOPE_GRAUS = 0.05
RESOLUCAO_M = 30.0
MARGEM_GRAUS = 0.02          # folga alem da AOI, para a drenagem nao ser cortada
# Ajustavel por `--margem`. Existe porque numa janela pequena a folga padrao nao
# basta: o ponto cai perto da borda, o roteamento de fluxo nao alcanca canal
# nenhum e o HAND sai nodata mesmo com elevacao e declividade validas. Foi o que
# aconteceu com 2.002 pontos do UFO em 51 chips. Aumentar a folga e a correcao
# fisica -- o canal existe, so estava fora do recorte.


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def epsg_utm(lat: float, lon: float) -> int:
    z = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + z


def nome_tile(lat: float, lon: float) -> str:
    la, lo = math.floor(lat), math.floor(lon)
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"


# ------------------------------------------------------------------ ambiente

def checar_ambiente() -> int:
    print("===TER01_AMBIENTE===")
    ok = True
    try:
        import whitebox
        wbt = whitebox.WhiteboxTools()
        wbt.verbose = False
        v = wbt.version().splitlines()[0] if hasattr(wbt, "version") else "?"
        print(f"whitebox     OK  {v}")
    except Exception as exc:  # noqa: BLE001
        print(f"whitebox     FALTA ({type(exc).__name__}) -> pip install whitebox")
        ok = False
    for m in ("rasterio", "numpy", "pandas", "pyproj"):
        try:
            __import__(m)
            print(f"{m:12s} OK")
        except Exception:  # noqa: BLE001
            print(f"{m:12s} FALTA")
            ok = False
    print(f"AMBIENTE={'PRONTO' if ok else 'INCOMPLETO'}")
    print("===END===")
    return 0 if ok else 1


# ------------------------------------------------------------------ DEM

def preparar_dem(bbox_ll, label: str, destino: Path) -> dict:
    """Baixa tiles GLO-30, mosaica e reprojeta ao UTM local a 30 m."""
    import numpy as np
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    warnings.filterwarnings("ignore")
    # Retomada: DEM ja preparado e legivel e reaproveitado. Sem isto, uma
    # execucao interrompida deixa um .tif parcial que o rasterio nao consegue
    # sobrescrever em pasta montada, e a AOI falha para sempre.
    if destino.exists():
        try:
            with rasterio.open(destino) as r:
                a = r.read(1)
                # DEM gravado antes da mascara de mar tem celulas <= 0 e faria o
                # WBT travar de novo. Nesse caso ignora o cache e regenera.
                if np.isfinite(a).any() and not (np.isfinite(a) & (a <= 0)).any():
                    return {"crs": str(r.crs), "shape": [int(r.height), int(r.width)],
                            "pixel_size": [RESOLUCAO_M, RESOLUCAO_M],
                            "bbox_ll": list(bbox_ll),
                            "n_valid": int(np.isfinite(a).sum()),
                            "n_total": int(a.size),
                            "dem_reaproveitado": True}
        except Exception:  # noqa: BLE001
            pass          # ilegivel: segue e tenta regerar

    minx, miny, maxx, maxy = bbox_ll
    minx -= MARGEM_GRAUS; miny -= MARGEM_GRAUS
    maxx += MARGEM_GRAUS; maxy += MARGEM_GRAUS
    CACHE.mkdir(parents=True, exist_ok=True)

    tiles = []
    for la in range(math.floor(miny), math.floor(maxy) + 1):
        for lo in range(math.floor(minx), math.floor(maxx) + 1):
            t = nome_tile(la + 0.5, lo + 0.5)
            local = CACHE / f"{t}.tif"
            url = f"https://copernicus-dem-30m.s3.amazonaws.com/{t}/{t}.tif"
            tiles.append(local if local.exists() else url)

    abertos = []
    for t in tiles:
        try:
            abertos.append(rasterio.open(str(t)))
        except Exception:  # noqa: BLE001
            continue          # tile de oceano nao existe; ausencia e esperada
    if not abertos:
        raise RuntimeError(f"{label}: nenhum tile DEM disponivel para {bbox_ll}")

    arr, tr = merge(abertos, bounds=(minx, miny, maxx, maxy))
    perfil = abertos[0].profile
    for a in abertos:
        a.close()

    lat_c, lon_c = (miny + maxy) / 2, (minx + maxx) / 2
    dst_crs = f"EPSG:{epsg_utm(lat_c, lon_c)}"
    tr2, w, h = calculate_default_transform(
        perfil["crs"], dst_crs, arr.shape[-1], arr.shape[-2],
        left=minx, bottom=miny, right=maxx, top=maxy, resolution=RESOLUCAO_M)

    saida = np.full((h, w), np.nan, dtype="float32")
    reproject(source=arr[0], destination=saida,
              src_transform=tr, src_crs=perfil["crs"],
              dst_transform=tr2, dst_crs=dst_crs,
              src_nodata=perfil.get("nodata"), dst_nodata=np.nan,
              resampling=Resampling.bilinear)

    # MASCARA DE MAR -- corrigido em 12/08/2026 depois de a EMSR734_AOI03 travar.
    # Aquela AOI e caribenha: 98% do raster e oceano a exatamente 0 m. O
    # `fill_depressions` do WhiteboxTools entra em resolucao de flats sobre uma
    # planicie plana de dezenas de milhares de celulas e nao termina. Nao e lento:
    # e degenerado.
    # Rotear fluxo sobre o mar nao tem significado hidrologico, e HAND acima do
    # nivel do mar e zero por definicao. Celulas <= 0 m viram nodata.
    # LIMITACAO DECLARADA: terra abaixo do nivel do mar (poldere holandes, vale
    # do Jordao) seria mascarada indevidamente. Nao ha nenhuma no conjunto atual.
    mar = np.isfinite(saida) & (saida <= 0)
    frac_mar = float(mar.mean())
    saida[mar] = np.nan
    if frac_mar > 0.5:
        print(f"    [{label}] {frac_mar*100:.0f}% do recorte e mar/nivel zero -- mascarado")

    valido = np.isfinite(saida)
    if valido.sum() < 500:
        raise RuntimeError(
            f"{label}: so {int(valido.sum())} celulas de terra apos mascarar o mar. "
            f"Recorte pequeno demais para derivar drenagem com limiar de "
            f"{AREA_CANAL_KM2} km2 (exige {AREA_CANAL_KM2/((RESOLUCAO_M**2)/1e6):.0f} celulas).")

    destino.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destino, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs=dst_crs, transform=tr2,
                       nodata=np.nan, compress="deflate") as d:
        d.write(saida, 1)
    return {"crs": dst_crs, "shape": [int(h), int(w)],
            "pixel_size": [RESOLUCAO_M, RESOLUCAO_M],
            "bbox_ll": [minx, miny, maxx, maxy],
            "n_valid": int(np.isfinite(saida).sum()),
            "n_total": int(saida.size)}


# ------------------------------------------------------------------ cadeia

def derivar(dem: Path, outdir: Path, label: str, meta_dem: dict) -> dict:
    """fill -> D-inf -> SCA -> slope -> canais(p98) -> HAND -> TWI."""
    import numpy as np
    import rasterio
    import whitebox

    outdir.mkdir(parents=True, exist_ok=True)
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    wbt.set_working_dir(str(outdir.resolve()))
    t0 = time.time()

    p = {k: str((outdir / f"{k}.tif").resolve()) for k in
         ("dem_filled", "dinf_accum_cells", "dinf_sca", "slope_deg_wbt",
          "streams_dinf", "hand_dinf", "twi_dinf")}

    wbt.fill_depressions(str(dem.resolve()), p["dem_filled"])
    wbt.d_inf_flow_accumulation(p["dem_filled"], p["dinf_accum_cells"],
                                out_type="cells", log=False, clip=False)
    wbt.d_inf_flow_accumulation(p["dem_filled"], p["dinf_sca"],
                                out_type="specific contributing area",
                                log=False, clip=False)
    wbt.slope(p["dem_filled"], p["slope_deg_wbt"], units="degrees")

    with rasterio.open(p["dinf_accum_cells"]) as s:
        acc = s.read(1).astype("float64")
        perfil = s.profile
    valido = np.isfinite(acc)
    if not valido.any():
        raise RuntimeError(f"{label}: acumulacao D-inf vazia -- cadeia quebrada")
    # Limiar em area contribuinte -> celulas, usando o pixel REAL deste raster.
    area_px_km2 = (RESOLUCAO_M * RESOLUCAO_M) / 1e6
    limiar = AREA_CANAL_KM2 / area_px_km2
    # percentil correspondente: diagnostico de densidade de drenagem, nao criterio
    percentil_equiv = float((acc[valido] < limiar).mean() * 100.0)

    wbt.extract_streams(p["dinf_accum_cells"], p["streams_dinf"],
                        threshold=limiar, zero_background=False)
    wbt.elevation_above_stream(p["dem_filled"], p["streams_dinf"], p["hand_dinf"])

    # TWI = ln(SCA / tan(slope)), com piso de declividade declarado
    with rasterio.open(p["dinf_sca"]) as s:
        sca = s.read(1).astype("float64")
    with rasterio.open(p["slope_deg_wbt"]) as s:
        slo = s.read(1).astype("float64")
    tanb = np.tan(np.radians(np.clip(slo, PISO_SLOPE_GRAUS, None)))
    with np.errstate(divide="ignore", invalid="ignore"):
        twi = np.log(np.where(sca > 0, sca, np.nan) / tanb)
    perfil.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    with rasterio.open(p["twi_dinf"], "w", **perfil) as d:
        d.write(twi.astype("float32"), 1)

    # Diagnostico de rede de drenagem. Uma AOI pequena ou muito plana pode
    # produzir pouquissimas celulas de canal -- a EMSR734_AOI03, ilha de 2 km,
    # gera 12. Nesse caso o HAND fica definido em pouca area e os pontos que
    # cairem fora viram nodata (e sao descartados pelo ter02, corretamente).
    # Nao e erro, mas nao pode ficar invisivel.
    with rasterio.open(p["streams_dinf"]) as s:
        st = s.read(1).astype("float64")
        nd_st = s.nodata
    if nd_st is not None:
        st[np.isclose(st, nd_st)] = np.nan
    n_canal = int(np.nansum(st > 0))
    if n_canal < 50:
        print(f"    [{label}] REDE DEGENERADA: {n_canal} celulas de canal. "
              f"HAND definido em pouca area; pontos fora viram nodata.")

    # verificacao antes de declarar sucesso: camada vazia e erro, nao aviso
    vazias = []
    for k in ("hand_dinf", "twi_dinf", "slope_deg_wbt"):
        with rasterio.open(p[k]) as s:
            a = s.read(1)
            if not np.isfinite(a).any() or float(np.nanstd(a)) == 0.0:
                vazias.append(k)
    if vazias:
        raise RuntimeError(f"{label}: camadas vazias ou constantes: {vazias}")

    manifesto = {
        "region_label": label, "dtm_input": str(dem),
        "outdir": str(outdir), **meta_dem,
        "stream_area_km2": AREA_CANAL_KM2,
        "stream_threshold_cells": limiar,
        "stream_percentile_equivalente": round(percentil_equiv, 3),
        "n_celulas_canal": n_canal,
        "piso_slope_graus": PISO_SLOPE_GRAUS,
        "twi_formula": "ln(SCA / tan(slope))",
        "resampling_dem": "bilinear",
        "wbt_version": wbt.version().splitlines()[0] if hasattr(wbt, "version") else "?",
        "elapsed_s": round(time.time() - t0, 1),
        "python": sys.version.split()[0], "platform": sys.platform,
        "outputs": p,
        "sha256": {k: sha256(Path(v)) for k, v in p.items() if Path(v).exists()},
    }
    (outdir / "run_manifest.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifesto


# ------------------------------------------------------------------ compara

def comparar(gerado: Path, referencia: Path, rotulo: str, outdir: Path) -> dict:
    """Mesmo formato do recife_validation/compare_*.json."""
    import numpy as np
    import rasterio

    with rasterio.open(gerado) as a, rasterio.open(referencia) as b:
        A, B = a.read(1).astype("float64"), b.read(1).astype("float64")
        grade = (a.shape == b.shape and str(a.crs) == str(b.crs)
                 and a.transform.almost_equals(b.transform, precision=1e-6))
        crs_a, crs_b, sh_a, sh_b = str(a.crs), str(b.crs), list(a.shape), list(b.shape)

    r = {"label": rotulo, "path_generated": str(gerado),
         "path_reference": str(referencia),
         "shape_generated": sh_a, "shape_reference": sh_b,
         "crs_generated": crs_a, "crs_reference": crs_b, "grid_match": bool(grade),
         "n_valid_generated": int(np.isfinite(A).sum()),
         "n_valid_reference": int(np.isfinite(B).sum())}

    if not grade:
        r["comparavel"] = False
        r["motivo"] = "grades diferentes -- comparacao pixel a pixel nao se aplica"
    else:
        m = np.isfinite(A) & np.isfinite(B)
        d = A[m] - B[m]
        r.update(comparavel=True, n_compared=int(m.sum()),
                 pearson_r=float(np.corrcoef(A[m], B[m])[0, 1]) if m.sum() > 2 else None,
                 mean_abs_diff=float(np.abs(d).mean()),
                 max_abs_diff=float(np.abs(d).max()),
                 mean_signed_diff=float(d.mean()),
                 p95_abs_diff=float(np.percentile(np.abs(d), 95)),
                 p99_abs_diff=float(np.percentile(np.abs(d), 99)),
                 frac_abs_diff_lt_1e_3=float((np.abs(d) < 1e-3).mean()),
                 mean_generated=float(A[m].mean()), mean_reference=float(B[m].mean()),
                 std_generated=float(A[m].std()), std_reference=float(B[m].std()))
    (outdir / f"compare_{rotulo}.json").write_text(
        json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    return r


# ------------------------------------------------------------------ alvos

def bbox_da_aoi(nome_aoi: str):
    if not CATALOGO.exists():
        raise RuntimeError(f"{CATALOGO} ausente. Rode cems04 --aois antes.")
    cat = json.loads(CATALOGO.read_text(encoding="utf-8"))
    for a in cat:
        rot = f"{a.get('activationCode')}_AOI{int(a.get('number')):02d}"
        if rot == nome_aoi:
            pts = re.findall(r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", a.get("extent") or "")
            xs = [float(x) for x, _ in pts]
            ys = [float(y) for _, y in pts]
            return (min(xs), min(ys), max(xs), max(ys))
    raise RuntimeError(f"AOI {nome_aoi} nao encontrada no catalogo")


def aois_ingremes() -> list[str]:
    import pandas as pd
    if not RELEVO.exists():
        raise RuntimeError(f"{RELEVO} ausente. Rode cems03 --verificar antes.")
    d = pd.read_csv(RELEVO)
    return sorted(d.loc[d.analogo_de_petropolis, "aoi"].astype(str).tolist())


def rodar_um(label: str, bbox) -> bool:
    outdir = OUT / label
    if (outdir / "run_manifest.json").exists():
        print(f"[{label}] JA_DERIVADO")
        return True
    try:
        dem = outdir / f"{label}_dem_30m.tif"
        meta = preparar_dem(bbox, label, dem)
        m = derivar(dem, outdir, label, meta)
        print(f"[{label}] OK shape={m['shape']} {m['crs']} "
              f"canal={m['stream_threshold_cells']:.1f} cells "
              f"(p{m['stream_percentile_equivalente']:.1f}) "
              f"({m['elapsed_s']}s)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] FALHA {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    global MARGEM_GRAUS
    args = sys.argv[1:]
    if "--checar-ambiente" in args:
        return checar_ambiente()
    if "--margem" in args:
        MARGEM_GRAUS = float(args[args.index("--margem") + 1])
        print(f"MARGEM ajustada para {MARGEM_GRAUS} grau")

    OUT.mkdir(parents=True, exist_ok=True)
    alvos: list[tuple[str, tuple]] = []

    if "--aoi" in args:
        nome = args[args.index("--aoi") + 1]
        alvos.append((nome, bbox_da_aoi(nome)))
    elif "--regiao" in args and "--bbox" in args:
        nome = args[args.index("--regiao") + 1]
        bb = tuple(float(x) for x in args[args.index("--bbox") + 1].split(","))
        alvos.append((nome, bb))
    elif "--lote" in args and args[args.index("--lote") + 1] in ("ingremes", "todas"):
        import pandas as pd

        qual = args[args.index("--lote") + 1]
        if qual == "ingremes":
            nomes = aois_ingremes()
        else:
            # todas as AOIs do dataset, ingremes ou nao. Necessario para que
            # planicie e serra entrem no mesmo modelo: enquanto so as ingremes
            # estiverem harmonizadas, os dois lados usam instrumentos diferentes.
            if not RELEVO.exists():
                print(f"ABORTADO: {RELEVO} ausente. Rode cems03 --verificar antes.")
                return 1
            nomes = sorted(pd.read_csv(RELEVO)["aoi"].astype(str).tolist())
        for a in nomes:
            try:
                alvos.append((a, bbox_da_aoi(a)))
            except Exception as exc:  # noqa: BLE001
                print(f"[{a}] SEM_BBOX {exc}")
    else:
        print(__doc__)
        return 0

    print(f"===TER01=== alvos={len(alvos)} resolucao={RESOLUCAO_M}m "
          f"area_canal={AREA_CANAL_KM2}km2")

    # LOTE ISOLADO POR SUBPROCESSO -- adotado em 12/08/2026.
    # Duas AOIs travaram o lote inteiro: uma ilha caribenha com 98% de mar
    # (corrigido pela mascara) e outra que ainda nao foi diagnosticada. Uma AOI
    # patologica nao pode custar o lote. Cada uma roda em processo proprio com
    # teto de tempo; quem estourar fica registrado e o lote segue.
    # `--aoi` sozinho continua rodando no processo atual, para depurar.
    if "--lote" in args:
        import subprocess

        # Teto por AOI. 120 s e o suficiente para a grande maioria (a mediana
        # fica em 2 a 30 s), mas AOIs grandes como as de Honduras e do Caribe
        # precisam de mais. Ajustavel: --teto 600.
        teto_s = int(args[args.index("--teto") + 1]) if "--teto" in args else 120
        ok, estourou, falhou = 0, [], []
        for i, (nome, _bb) in enumerate(alvos, 1):
            if (OUT / nome / "run_manifest.json").exists():
                ok += 1
                continue
            try:
                r = subprocess.run(
                    [sys.executable, "-u", str(Path(__file__).resolve()),
                     "--aoi", nome],
                    capture_output=True, text=True, timeout=teto_s)
                saida = (r.stdout or "").strip().splitlines()
                for ln in saida:
                    if "OK shape" in ln or "REDE DEGENERADA" in ln or "mascarado" in ln:
                        print(f"  {ln.strip()}", flush=True)
                if (OUT / nome / "run_manifest.json").exists():
                    ok += 1
                else:
                    falhou.append(nome)
                    print(f"  [{nome}] FALHOU: "
                          f"{(saida[-1] if saida else (r.stderr or '')[:120])}", flush=True)
            except subprocess.TimeoutExpired:
                estourou.append(nome)
                print(f"  [{nome}] ESTOUROU {teto_s}s -- pulado", flush=True)
            if i % 20 == 0:
                print(f"  ... {i}/{len(alvos)} (ok={ok} estourou={len(estourou)} "
                      f"falhou={len(falhou)})", flush=True)

        print(f"\nDERIVADOS={ok}/{len(alvos)}  ->  {OUT}")
        if estourou:
            print(f"ESTOURARAM_O_TETO ({len(estourou)}): {estourou}")
            print("  Rode com --aoi <nome> para diagnosticar sem teto.")
        if falhou:
            print(f"FALHARAM ({len(falhou)}): {falhou}")
        print("===END===")
        return 0 if ok == len(alvos) else 1

    ok = sum(1 for nome, bb in alvos if rodar_um(nome, bb))
    print(f"\nDERIVADOS={ok}/{len(alvos)}  ->  {OUT}")
    print("===END===")
    return 0 if ok == len(alvos) else 1


if __name__ == "__main__":
    raise SystemExit(main())
