"""MT11: o espelho GEE do FABDEM reproduz o arquivo de Bristol em mais de uma janela?

A rodada 3 conferiu UMA janela (EMSR680_AOI01: 7.429.851 pixels idênticos) e habilitou o espelho
`projects/sat-io/open-datasets/FABDEM` como transporte. A memória de 01/10/2026 registrava 18 de 40
arquivos locais diferentes, sem artefato localizado. Este conferidor refaz a medição com artefato:

  referências  janelas `local_runs/mdt-fabdem/*_fabdem_4326.tif` baixadas de data.bris.ac.uk (manifesto
               sem `transporte`), ordenadas por sha256("mt11fab|" + nome), uma por tile de 1 grau central
               distinto, até N
  leitura      `rodada3_fabdem_gee.ler_janela` (só importada), grade nativa 1 arcsec EPSG:4326, mesmo transform
  comparação   recorte central de até 1024 x 1024 pixels: forma, transform, nodata, pixels diferentes, |dif| máx

Nenhum rótulo é lido; nada é gravado em `mdt-fabdem`. Saída: `local_runs/mt11-transdutivo/fabdem_transporte.json`.
Uso: python scripts/metodo/mt11_fabdem_transporte.py [--n 40]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for s in ("ia", "servico", "terreno"):
    sys.path.insert(0, str(REPO / "scripts" / s))
import rodada3_fabdem_gee as r3  # noqa: E402
from proveniencia import agora_utc, sha256_arquivo  # noqa: E402

MDT = REPO / "local_runs" / "mdt-fabdem"
SAIDA = REPO / "local_runs" / "mt11-transdutivo" / "fabdem_transporte.json"
LADO = 1024
TOL_CELULA = 1e-6   # diferença de origem só de ponto flutuante (medida: 1e-10 a 1e-12 célula)


def regiao(o, s, l, n) -> str:
    x, y = (o + l) / 2, (s + n) / 2
    return ("asia" if x >= 60 and y > -12 else "europa" if -12 <= x < 45 and y >= 35 else
            "america_norte_central" if x < -50 and y >= 13 else "oceania" if x >= 110 and y <= -10 else
            "africa" if -20 <= x < 55 and -36 <= y < 35 else "america_sul" if x < -30 else "outro")


def referencias(n: int, fora_america_sul: bool = False) -> list[dict]:
    vistos, out = set(), []
    tifs = sorted(MDT.glob("*_fabdem_4326.tif"), key=lambda p: hashlib.sha256(f"mt11fab|{p.name}".encode()).hexdigest())
    for t in tifs:
        man = t.with_name(t.name.replace("_fabdem_4326.tif", "_fabdem_manifesto.json"))
        if not man.exists():
            continue
        m = json.loads(man.read_text(encoding="utf-8"))
        if m.get("transporte"):
            continue
        o, s, l, nn = m["bbox_4326"]
        if fora_america_sul and regiao(o, s, l, nn) == "america_sul":
            continue
        tile = f"{math.floor((s + nn) / 2)}_{math.floor((o + l) / 2)}"
        if tile in vistos:
            continue
        vistos.add(tile)
        out.append({"tif": t, "tile_central": tile, "regiao": regiao(o, s, l, nn), "zips": sorted({x["zip"] for x in m.get("tiles", [])})})
        if len(out) >= n:
            break
    return out


def comparar(ee, ref: dict) -> dict:
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(ref["tif"]) as r:
        H, W = r.shape
        h, w = min(LADO, H), min(LADO, W)
        i0, j0 = (H - h) // 2, (W - w) // 2
        win = Window(j0, i0, w, h)
        a = r.read(1, window=win)
        tr = r.window_transform(win)
        nod = r.nodata
    b = r3.ler_janela(ee, tr, w, h)
    na, nb = (a == nod) if nod is not None else np.isnan(a), b == r3.NODATA
    v = ~na & ~nb
    d = np.abs(a[v].astype(np.float64) - b[v])
    return {"referencia": str(ref["tif"].relative_to(REPO)), "sha256": sha256_arquivo(ref["tif"]),
            "tile_central": ref["tile_central"], "regiao": ref.get("regiao"), "zips": ref["zips"], "forma": [h, w], "nodata_ref": nod,
            "validos_ambos": int(v.sum()), "nodata_divergente": int((na != nb).sum()),
            "pixels_diferentes": int((d > 0).sum()), "max_abs": float(d.max()) if d.size else None,
            "p50_abs_dif": float(np.median(d[d > 0])) if (d > 0).any() else 0.,
            "identico": bool(d.size and d.max() == 0 and (na == nb).all())}


def caminho_de_gravacao(nomes: list[str]) -> dict:
    """Regrava pelo espelho (`rodada3_fabdem_gee.baixar`, o mesmo que a cadeia usaria) janelas já baixadas de
    Bristol, num diretório temporário, e compara arquivo contra arquivo: forma, transform, CRS, nodata, tipo, valores."""
    import tempfile
    import rasterio
    linhas = []
    for nome in nomes:
        man = json.loads((MDT / f"{nome}_fabdem_manifesto.json").read_text(encoding="utf-8"))
        ref = MDT / f"{nome}_fabdem_4326.tif"
        try:
            with tempfile.TemporaryDirectory(prefix="mt11_fab_") as tmp:
                r3.baixar({k: man[k] for k in ("nome", "bbox_4326", "tiles")}, Path(tmp))
                with rasterio.open(ref) as a, rasterio.open(Path(tmp) / ref.name) as b:
                    va, vb = a.read(1), b.read(1)
                    dt_cel = max(abs(x - y) / abs(a.transform.a) for x, y in zip(tuple(a.transform)[:6], tuple(b.transform)[:6]))
                    meta = {"forma": [list(a.shape), list(b.shape)], "transform_igual": a.transform == b.transform,
                            "transform_max_delta_celulas": dt_cel,
                            "crs_igual": a.crs == b.crs, "nodata": [a.nodata, b.nodata], "dtype": [a.dtypes[0], b.dtypes[0]]}
            igual_forma = va.shape == vb.shape
            dif = int((va != vb).sum()) if igual_forma else None
            linhas.append({"referencia": nome, "regiao": regiao(*man["bbox_4326"]), **meta, "pixels": int(va.size),
                           "pixels_diferentes": dif, "identico": bool(igual_forma and dif == 0 and meta["transform_max_delta_celulas"] < TOL_CELULA
                                                                     and meta["crs_igual"] and meta["nodata"][0] == meta["nodata"][1])})
        except Exception as e:  # noqa: BLE001
            linhas.append({"referencia": nome, "erro": f"{type(e).__name__}: {str(e)[:160]}"})
        print(json.dumps(linhas[-1], default=str), flush=True)
    out = {"tipo": "mt11_caminho_de_gravacao_fabdem", "quando_utc": agora_utc(), "codigo_sha256": sha256_arquivo(Path(__file__)),
           "leitor_sha256": sha256_arquivo(Path(r3.__file__)), "linhas": linhas,
           "identicas": sum(bool(x.get("identico")) for x in linhas), "total": len(linhas)}
    SAIDA.with_name("fabdem_caminho_gravacao.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--fora-america-sul", action="store_true",
                    help="só referências fora da América do Sul (regiões das janelas MT11); grava *_regional.json")
    ap.add_argument("--caminho-gravacao", nargs="+", help="nomes de janelas locais para regravar pelo espelho")
    a = ap.parse_args()
    if a.caminho_gravacao:
        caminho_de_gravacao(a.caminho_gravacao)
        return
    global SAIDA
    if a.fora_america_sul:
        SAIDA = SAIDA.with_name("fabdem_transporte_regional.json")
    ee = r3._ee()
    linhas = []
    for ref in referencias(a.n, a.fora_america_sul):
        try:
            linhas.append(comparar(ee, ref))
        except Exception as e:  # noqa: BLE001 - falha de transporte fica registrada, nunca vira identidade
            linhas.append({"referencia": str(ref["tif"].relative_to(REPO)), "erro": f"{type(e).__name__}: {str(e)[:160]}"})
        print(json.dumps({k: linhas[-1].get(k) for k in ("referencia", "identico", "pixels_diferentes", "max_abs", "erro")}), flush=True)
    ok = [x for x in linhas if "erro" not in x]
    res = {"tipo": "mt11_conferencia_transporte_fabdem", "quando_utc": agora_utc(), "colecao": r3.COLECAO,
           "codigo_sha256": sha256_arquivo(Path(__file__)), "leitor_sha256": sha256_arquivo(Path(r3.__file__)),
           "recorte": f"centro de até {LADO}x{LADO} pixels", "referencias": len(linhas), "comparadas": len(ok),
           "identicas": sum(x["identico"] for x in ok), "diferentes": sum(not x["identico"] for x in ok),
           "zips_com_diferenca": sorted({z for x in ok if not x["identico"] for z in x["zips"]}),
           "zips_identicos": sorted({z for x in ok if x["identico"] for z in x["zips"]}), "linhas": linhas,
           "alcance": "janelas locais conferidas; não certifica tile não conferido"}
    SAIDA.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("referencias", "comparadas", "identicas", "diferentes")}))


if __name__ == "__main__":
    main()
