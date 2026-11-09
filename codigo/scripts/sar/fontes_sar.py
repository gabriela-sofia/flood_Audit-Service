"""Adapter Sen1Floods11 + GEOID: um contrato de chip para a mesma tarefa SAR.

Alvo comum: agua observada NAO permanente na visada posterior.
  Sen1Floods11: 1 = agua fora da agua permanente JRC; 0 = seco; -1 = desconhecido,
                permanente JRC ou sem leitura do sensor (mesma regra de ci.alvo).
  GEOID:        classe 2 (inundada) = 1; classe 0 = 0; classe 1 (permanente CEMS) = -1;
                255, validade 0, mascara GDAL ou SAR nao finito/nao positivo = -1.
JRC e CEMS sao referencias diferentes de agua permanente: alinhar o alvo nao prova
que observam o mesmo. Desconhecido nunca vira negativo.

Entrada: dB VV/VH em quatro canais na ordem [VVpos, VHpos, VVpre, VHpre].
  modo "pos":        posterior duplicado; suporte so do posterior.
  modo "pre_pos":    anterior datado real (so GEOID com pre_pos_primario).
  modo "pos_par":    posterior duplicado com EXATAMENTE o suporte do par real;
                     controle do pre_pos, mesmos pixels e canais.
Normalizacao fixa, sem estatistica de dado: recorte nos limites do v1 e escala
afim para [-1, 1] por banda. Nao usa contexto futuro, validacao ou teste.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

import campanha_independencia_autonomia as ci
from dados_pares_geograficos import LIMITES_DB, imagens

MODOS = ("pos", "pre_pos", "pos_par")
_CACHE = {}
_HASH_OK = set()


def normalizar_fixo(x):
    """dB -> [-1, 1] por banda, recortando nos limites declarados; independe dos dados."""
    z = np.empty_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        lo, hi = LIMITES_DB[i % 2]
        z[i] = (np.clip(x[i], lo, hi) - (lo+hi)/2) / ((hi-lo)/2)
    return z


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def _conferir(r):
    for k, p in r["caminhos"].items():
        if p in _HASH_OK:
            continue
        esperado = r.get("hashes_arquivos_pai", {}).get(k)
        if esperado is None or _sha(p) != esperado:
            raise ValueError(f"arquivo GEOID alterado ou sem hash: {p}")
        _HASH_OK.add(p)


def _ler_geoid(r):
    """Le o recorte 512 sem reamostrar; devolve dB [pos, pre], classes e mascaras."""
    _conferir(r)
    col, row, w, h = r["janela"]
    win = Window(col, row, w, h)
    with rasterio.open(r["caminhos"]["post"]) as f:
        pos, mpos = f.read(window=win).astype(np.float64), f.read_masks(window=win).min(0) > 0
    with rasterio.open(r["caminhos"]["pre"]) as f:
        pre, mpre = f.read(window=win).astype(np.float64), f.read_masks(window=win).min(0) > 0
    with rasterio.open(r["caminhos"]["label"]) as f:
        y, mrot = f.read(1, window=win), f.read_masks(1, window=win) > 0
    with rasterio.open(r["caminhos"]["validity"]) as f:
        val, mval = f.read(1, window=win), f.read_masks(1, window=win) > 0
    if not set(np.unique(y)).issubset({0, 1, 2, 255}) or not set(np.unique(val)).issubset({0, 1}):
        raise ValueError("classes ou validade fora do contrato GEOID")
    vis_pos = mpos & np.all(np.isfinite(pos) & (pos > 0), 0)
    vis_pre = mpre & np.all(np.isfinite(pre) & (pre > 0), 0)
    rotulado = (val == 1) & mval & mrot & (y != 255)
    db = np.zeros((4,) + y.shape, np.float32)
    db[:2] = np.where(vis_pos, 10*np.log10(np.where(vis_pos, pos, 1)), 0)
    db[2:] = np.where(vis_pre, 10*np.log10(np.where(vis_pre, pre, 1)), 0)
    return db, y, vis_pos & rotulado, vis_pos & vis_pre & rotulado


def carregar(r):
    """(x_dB [4,H,W] em ordem pos,pre; alvo por modo; suporte por modo). Cacheado por chip."""
    if r["nome"] in _CACHE:
        return _CACHE[r["nome"]]
    if r["fonte"] == "sen1":
        x, sensor = imagens(r["caminhos"]["antes"], r["caminhos"]["depois"])
        x = np.asarray(x, np.float32)
        y, _, _ = ci.alvo(r, sensor)
        alvo = {"pos": y.astype(np.int8)}
        suporte = {"pos": sensor.copy()}
        classes = None
    elif r["fonte"] == "geoid":
        x, classes, vis, vis_par = _ler_geoid(r)
        alvo, suporte = {}, {}
        for modo, v in (("pos", vis), ("pre_pos", vis_par), ("pos_par", vis_par)):
            a = np.full(classes.shape, -1, np.int8)
            ok = v & (classes != 1)
            a[ok] = (classes[ok] == 2).astype(np.int64)
            alvo[modo], suporte[modo] = a, v
    else:
        raise ValueError(f"fonte desconhecida: {r['fonte']}")
    _CACHE[r["nome"]] = (x, alvo, suporte, classes)
    return _CACHE[r["nome"]]


def entrada(r, modo="pos", janela=None):
    if modo not in MODOS:
        raise ValueError("modo desconhecido")
    if modo != "pos" and not (r["fonte"] == "geoid" and r.get("pre_pos_primario")):
        raise ValueError("pre/pos real so com par datado admitido")
    x, _, suporte, _ = carregar(r)
    v = suporte["pos" if modo == "pos" else "pre_pos"]
    if janela is not None:
        a, b, s = janela
        x, v = x[:, a:a+s, b:b+s], v[a:a+s, b:b+s]
    z = normalizar_fixo(x)
    if modo in ("pos", "pos_par"):
        z[2:] = z[:2]
    z[:, ~v] = 0
    return z, v


def alvo(r, modo="pos", janela=None):
    _, a, _, _ = carregar(r)
    y = a["pos" if modo == "pos" else "pre_pos"]
    if janela is not None:
        p, q, s = janela
        y = y[p:p+s, q:q+s]
    # Guardado em int8 para caber 418 chips em memoria; a perda exige int64.
    return y.astype(np.int64)


def contagem(r, modo="pos"):
    """Suporte apos todas as mascaras; classes GEOID brutas so para descrever."""
    _, a, suporte, classes = carregar(r)
    y = a["pos" if modo == "pos" else "pre_pos"]
    out = {"positivo": int((y == 1).sum()), "negativo": int((y == 0).sum()),
           "ignorado": int((y == -1).sum()), "visivel": int(suporte["pos" if modo == "pos" else "pre_pos"].sum()),
           "total": int(y.size)}
    if classes is not None:
        out["permanente_ignorado"] = int(((classes == 1) & suporte["pos" if modo == "pos" else "pre_pos"]).sum())
    return out


def chips_sen1(pares, divisoes=None):
    out = []
    for r in pares:
        c = dict(r)
        c["fonte"] = "sen1"
        out.append(c)
    return out


def chips_geoid(manifesto_preparo):
    out = []
    for c in manifesto_preparo["chips"]:
        r = dict(c)
        r["fonte"] = "geoid"
        r["grade"] = {"forma": [c["janela"][3], c["janela"][2]], "bounds": c["bounds"], "crs": c["crs"]}
        r["hashes"] = {"depois": c["hashes_arquivos_pai"]["post"], "antes": c["hashes_arquivos_pai"]["pre"]}
        out.append(r)
    return out


def limpar_cache():
    _CACHE.clear()
