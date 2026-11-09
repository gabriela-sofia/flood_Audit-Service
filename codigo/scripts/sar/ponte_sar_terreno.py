"""Ponte SAR <-> terreno: grade, observabilidade e procedencia por chip Sen1Floods11.

Versao `ponte_sar_terreno_v1` (11/10/2026). Reprojeta a pilha `ter-01`
(UTM, ~16 m, WhiteboxTools 2.4.0) para a grade do chip SAR posterior (EPSG:4326,
512x512, ~10 m), bilinear, com NoData do WBT e NaN fora do suporte: nunca vira
zero. Atributos usados: HAND (D-inf), declive e elevacao relativa ao proprio chip
(DEM menos o percentil 1 dos pixels validos do chip; sem rotulo).
Fora por truncamento: area de contribuicao especifica e acumulacao sao
calculadas so dentro do recorte (MT02); o HAND herda a rede de drenagem
recortada e leva essa ressalva. O DEM e contemporaneo da derivacao, nao do evento.
Os rotulos Sen1Floods11 sao manuais e nao usam HAND: o contraste nao e circular.
Normalizacao fixa sem dado:
  HAND: log1p(clip(0,100))/log1p(100) -> [-1,1]; declive: clip(0,45) -> [-1,1];
  elevacao relativa: clip(0,50 m) -> [-1,1]; invalido -> 0 e fora da mascara.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

REPO = Path(__file__).resolve().parents[2]
RAIZ_TER = REPO / "local_runs" / "ter-01-cadeia-harmonizada"
CACHE = REPO / "local_runs" / "ponte-sar-terreno" / "v1" / "terreno"
SEM_DADO = -32768.0
CAMADAS = ("hand_dinf", "slope_deg_wbt", "dem_filled")
CANAIS = ("hand", "declive", "elevacao_relativa")
EXCLUIDAS = {"dinf_sca": "truncada no recorte", "dinf_accum_cells": "truncada no recorte"}


def diretorio(nome, raiz=None):
    raiz = Path(raiz or RAIZ_TER)
    for d in (raiz / f"n1chip_{nome}_LabelHand", raiz / f"n1chip_{nome}", raiz / nome):
        if d.exists():
            return d
    raise FileNotFoundError(f"sem terreno para {nome}")


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def normalizar(hand, declive, elev_rel):
    h = np.log1p(np.clip(hand, 0, 100))/np.log1p(100)*2 - 1
    s = np.clip(declive, 0, 45)/45*2 - 1
    e = np.clip(elev_rel, 0, 50)/50*2 - 1
    return np.stack([h, s, e]).astype(np.float32)


def alinhar(r, raiz=None):
    """(canais normalizados [3,H,W], mascara de validade, procedencia) na grade do SAR posterior."""
    d = diretorio(r["nome"], raiz)
    with rasterio.open(r["caminhos"]["depois"]) as f:
        dst_crs, dst_tr, forma = f.crs, f.transform, (f.height, f.width)
    brutos, prov = {}, {"fontes": {}, "reamostragem": "bilinear", "grade_destino": {
        "crs": str(dst_crs), "transform": list(dst_tr)[:6], "forma": list(forma)}}
    for c in CAMADAS:
        p = d / f"{c}.tif"
        with rasterio.open(p) as f:
            src = f.read(1).astype(np.float32)
            src[(src == SEM_DADO) | ~np.isfinite(src)] = np.nan
            dst = np.full(forma, np.nan, np.float32)
            reproject(src, dst, src_transform=f.transform, src_crs=f.crs, dst_transform=dst_tr, dst_crs=dst_crs,
                      resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
            prov["fontes"][c] = {"arquivo": str(p), "sha256": _sha(p), "crs": str(f.crs),
                                 "resolucao_m": abs(f.transform.a)}
        brutos[c] = dst
    valido = np.logical_and.reduce([np.isfinite(v) for v in brutos.values()])
    if not valido.any():
        raise ValueError(f"terreno sem pixel valido na grade do SAR: {r['nome']}")
    base = np.percentile(brutos["dem_filled"][valido], 1)
    x = normalizar(np.nan_to_num(brutos["hand_dinf"]), np.nan_to_num(brutos["slope_deg_wbt"]),
                   np.nan_to_num(brutos["dem_filled"]) - base)
    x[:, ~valido] = 0
    prov.update({"fracao_valida": float(valido.mean()), "elevacao_base_p1_m": float(base),
                 "excluidas": EXCLUIDAS, "ressalva_hand": "drenagem recortada no chip"})
    return x, valido, prov


_MEM = {}


def terreno(r, cache=None, raiz=None):
    """Cacheado em disco (float16) e em memoria; recalcula se a fonte mudou."""
    if r["nome"] in _MEM:
        return _MEM[r["nome"]]
    cache, raiz = Path(cache or CACHE), Path(raiz or RAIZ_TER)
    f, fj = cache / f"{r['nome']}.npz", cache / f"{r['nome']}.json"
    if f.exists() and fj.exists():
        prov = json.loads(fj.read_text(encoding="utf-8"))
        if all(_sha(v["arquivo"]) == v["sha256"] for v in prov["fontes"].values()):
            z = np.load(f)
            _MEM[r["nome"]] = (z["x"].astype(np.float32), z["v"], prov)
            return _MEM[r["nome"]]
    x, v, prov = alinhar(r, raiz)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, x=x.astype(np.float16), v=v)
    fj.write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    _MEM[r["nome"]] = (x.astype(np.float16).astype(np.float32), v, prov)
    return _MEM[r["nome"]]
