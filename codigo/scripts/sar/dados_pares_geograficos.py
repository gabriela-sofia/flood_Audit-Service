"""Contrato geografico bitemporal: agua observada, desconhecido e agua permanente."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[2]
JRC = REPO / "local_runs/n1a-sen1floods11/v1.1/data/flood_events/HandLabeled/JRCWaterHand"
_CACHE_IMAGENS = {}
_CACHE_ROTULOS = {}
LIMITES_DB = ((-23., 0.), (-28., -5.))


def recortar_db(x):
    z = x.copy()
    for i in range(z.shape[0]):
        z[i] = np.clip(z[i], *LIMITES_DB[i % 2])
    return z


def sha(caminho):
    h = hashlib.sha256()
    with Path(caminho).open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def identidade(nome):
    partes = Path(nome).stem.split("_")
    if len(partes) < 2 or not partes[1].isdigit():
        raise ValueError("identidade de chip nao reconhecida")
    return "_".join(partes[:2])


def indexar(raiz):
    itens = {}
    for p in sorted(Path(raiz).rglob("*.tif")):
        chave = identidade(p.name)
        if chave in itens:
            raise ValueError("produto com identidade duplicada")
        itens[chave] = p
    return itens


def grade(f):
    return {"crs": str(f.crs), "transform": list(f.transform),
            "forma": [f.height, f.width], "bounds": list(f.bounds)}


def conferir_grade(f, referencia):
    g = grade(f)
    # Escritas independentes variam nas ultimas casas: tolerancia de um
    # milionesimo de pixel, sem deslocar nem reamostrar qualquer matriz.
    tolerancia = max(abs(referencia["transform"][0]), abs(referencia["transform"][4])) * 1e-6
    if (not f.crs or g["crs"] != referencia["crs"] or g["forma"] != referencia["forma"]
            or not np.allclose(g["transform"], referencia["transform"], rtol=0, atol=tolerancia)
            or not np.allclose(g["bounds"], referencia["bounds"], rtol=0, atol=tolerancia)):
        raise ValueError("CRS, extensao ou grade divergente; nao reamostrar silenciosamente")


def janela_anterior(f, referencia):
    """Aceita um recorte na mesma malha; nunca interpola pixels ausentes."""
    t, r = list(f.transform), referencia["transform"]
    tol = max(abs(r[0]), abs(r[4])) * 1e-6
    if (not f.crs or str(f.crs) != referencia["crs"] or r[1] != 0 or r[3] != 0
            or not np.allclose([t[0], t[1], t[3], t[4]], [r[0], r[1], r[3], r[4]], rtol=0, atol=tol)):
        raise ValueError("grade divergente na referencia anterior")
    col, row = (t[2]-r[2])/r[0], (t[5]-r[5])/r[4]
    c, a = int(round(col)), int(round(row))
    h, w = referencia["forma"]
    if abs(col-c) > 1e-6 or abs(row-a) > 1e-6 or min(a, c) < 0 or a+f.height > h or c+f.width > w:
        raise ValueError("grade divergente: anterior nao e recorte inteiro da referencia")
    return a, c, f.height, f.width


def inventariar(raiz, jrc=JRC):
    raiz = Path(raiz)
    metadados = json.loads((raiz / "Sen1Floods11_Metadata.geojson").read_text(encoding="utf-8"))
    eventos = {f["properties"]["location"]: f["properties"] for f in metadados["features"]}
    # Mekong e o nome dos chips do evento Cambodia no catalogo oficial.
    if "Cambodia" in eventos:
        eventos["Mekong"] = eventos["Cambodia"]
    dirs = [p for p in (raiz / "extraidos").iterdir() if p.is_dir()]
    produtos = {}
    for nome, termo in (("rotulo", "label"), ("antes", "pre"), ("depois", "post")):
        candidatos = [p for p in dirs if termo in p.name.lower()]
        if len(candidatos) != 1:
            raise ValueError(f"diretorio de {nome} ambiguo ou ausente: {dirs}")
        produtos[nome] = indexar(candidatos[0])
    base = set(produtos["rotulo"])
    if any(set(p) - base for p in produtos.values()):
        raise ValueError("imagem sem rotulo correspondente")
    completos = set.intersection(*(set(p) for p in produtos.values()))
    excluidos = [{"nome": nome, "motivo": "par incompleto na fonte publicada",
                  "produtos_ausentes": [k for k, p in produtos.items() if nome not in p]}
                 for nome in sorted(base - completos)]
    linhas, recusados = [], []
    for nome in sorted(produtos["rotulo"]):
        if nome not in completos:
            continue
        caminhos = {k: str(v[nome].resolve()) for k, v in produtos.items()}
        caminhos["jrc"] = str((Path(jrc) / f"{nome}_JRCWaterHand.tif").resolve())
        try:
            with rasterio.open(caminhos["rotulo"]) as f:
                g = grade(f)
                y = f.read()[0]
                if f.count != 1 or not set(np.unique(y)).issubset({-1, 0, 1}):
                    raise ValueError("rotulo fora do contrato -1/0/1")
            for k in ("antes", "depois", "jrc"):
                with rasterio.open(caminhos[k]) as f:
                    if k == "antes":
                        janela = janela_anterior(f, g)
                    else:
                        conferir_grade(f, g)
                    if f.count != (1 if k == "jrc" else 2):
                        raise ValueError(f"{k}: bandas divergentes")
                    if k == "jrc" and not set(np.unique(f.read()[0])).issubset({-1, 0, 1}):
                        raise ValueError("JRC fora do contrato")
            pais = nome.split("_")[0]
            e = eventos[pais]
            data = datetime.strptime(e["s1_date"], "%Y/%m/%d").date().isoformat()
            original = Path(jrc).parent / "LabelHand" / f"{nome}_LabelHand.tif"
            hash_anterior = sha(original) if original.exists() else None
            linhas.append({"nome": nome, "evento": f"sen1:{e['ID']}:{data}",
                           "local": pais, "data_pos_catalogo": data,
                           "orbita_catalogo": e["orbit"], "orbita_relativa_catalogo": e["rel_orbit_num"],
                           "antes": "mediana do ano anterior conforme artigo; datas individuais ausentes",
                           "grade": g, "caminhos": caminhos,
                           "janela_anterior": list(janela),
                           "cobertura_anterior": "recorte inteiro sem interpolacao; exterior desconhecido",
                           "bandas": "VV/VH conforme artigo; descricoes anteriores podem estar ausentes",
                           "precisao_de_calculo": "float32; arquivos e hashes originais preservados",
                           "rotulo_original_sha256": hash_anterior,
                           "rotulo_identico_ao_preexistente": hash_anterior == sha(caminhos["rotulo"]),
                           "hashes": {k: sha(p) for k, p in caminhos.items()}})
        except (ValueError, KeyError, rasterio.errors.RasterioError) as erro:
            recusados.append({"nome": nome, "motivo": str(erro)})
    return {"fonte": "https://zenodo.org/records/7946594", "pares": linhas,
            "recusados": recusados, "excluidos_sem_par": excluidos,
            "rotulos_publicados": len(base),
            "eventos": dict(Counter(r["evento"] for r in linhas)),
            "metadados_sha256": sha(raiz / "Sen1Floods11_Metadata.geojson"),
            "rotulos_identicos_ao_corpus_preexistente": sum(r["rotulo_identico_ao_preexistente"] for r in linhas),
            "alvo": "agua observada; primaria fora de agua permanente JRC",
            "regime": "desenvolvimento_por_evento: rotulos Sen1Floods11 ja examinados no projeto",
            "promovido": False}


def sobrepoe(a, b):
    ax0, ay0, ax1, ay1 = a["grade"]["bounds"]
    bx0, by0, bx1, by1 = b["grade"]["bounds"]
    return min(ax1, bx1) > max(ax0, bx0) + 1e-10 and min(ay1, by1) > max(ay0, by0) + 1e-10


def conferir_particoes(treino, validacao, teste):
    partes = [treino, validacao, teste]
    if any(not p for p in partes):
        raise ValueError("particao vazia")
    for i in range(3):
        for j in range(i + 1, 3):
            for a in partes[i]:
                for b in partes[j]:
                    if a["evento"] == b["evento"] or sobrepoe(a, b):
                        raise ValueError("evento ou area compartilhada entre particoes")
                    # Uma copia de qualquer data tambem nao pode atravessar a reserva.
                    if set(a["hashes"][k] for k in ("antes", "depois")) & set(b["hashes"][k] for k in ("antes", "depois")):
                        raise ValueError("copia de imagem entre particoes")


def particao(pares, teste_eventos, validacao_eventos):
    if set(teste_eventos) & set(validacao_eventos):
        raise ValueError("teste e validacao com evento compartilhado")
    te = [r for r in pares if r["evento"] in teste_eventos]
    va = [r for r in pares if r["evento"] in validacao_eventos]
    tr = [r for r in pares if r["evento"] not in set(teste_eventos) | set(validacao_eventos)]
    conferir_particoes(tr, va, te)
    return tr, va, te


@lru_cache(maxsize=24)
def imagens(antes, depois):
    if (antes, depois) in _CACHE_IMAGENS:
        x, sensor, i = _CACHE_IMAGENS[(antes, depois)]
        return x[i], sensor[i]
    with rasterio.open(antes) as a, rasterio.open(depois) as b:
        row, col, h, w = janela_anterior(a, grade(b))
        anterior = np.full((2, b.height, b.width), np.nan, dtype=np.float32)
        mascara = np.zeros((2, b.height, b.width), dtype=np.uint8)
        anterior[:, row:row+h, col:col+w] = a.read()
        mascara[:, row:row+h, col:col+w] = a.read_masks()
        x = np.concatenate([b.read(), anterior]).astype(np.float32)
        valido = np.concatenate([b.read_masks(), mascara]).all(0) & np.isfinite(x).all(0)
    return x, valido


@lru_cache(maxsize=24)
def rotulo(caminho, jrc):
    if (caminho, jrc) in _CACHE_ROTULOS:
        y, v, npv, i = _CACHE_ROTULOS[(caminho, jrc)]
        return y[i].astype(np.int64), v[i], npv[i]
    with rasterio.open(caminho) as f, rasterio.open(jrc) as p:
        y, permanente = f.read()[0].astype(np.int64), p.read()[0]
        valido = (y != -1) & (f.read_masks()[0] != 0)
        sem_permanente = valido & (permanente == 0) & (p.read_masks()[0] != 0)
    return y, valido, sem_permanente


def entrada(r, estat, modo="pre_pos", janela=None):
    x, v = imagens(r["caminhos"]["antes"], r["caminhos"]["depois"])
    if janela is not None:
        a, b, s = janela
        x, v = x[:, a:a+s, b:b+s], v[a:a+s, b:b+s]
    x = recortar_db(x)
    media = np.asarray(estat["media"], dtype=np.float32)[:, None, None]
    desvio = np.asarray(estat["desvio"], dtype=np.float32)[:, None, None]
    x = (x - np.tile(media, (2, 1, 1))) / np.tile(desvio, (2, 1, 1))
    x[:, ~v] = 0
    if modo == "pos":
        x[2:] = x[:2]
    elif modo == "pos_delta":
        x[2:] = x[:2] - x[2:]
    elif modo != "pre_pos":
        raise ValueError("entrada desconhecida")
    return x, v


def materializar_cache(inv, destino):
    """Cache sem reamostragem; verifica hashes antes de reutilizar."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    assinatura = hashlib.sha256(json.dumps(inv, sort_keys=True).encode()).hexdigest()
    manifesto = destino / "cache.json"
    linhas = inv["pares"]
    h, w = linhas[0]["grade"]["forma"]
    specs = {"x": (np.float32, (len(linhas), 4, h, w)),
             "sensor": (np.bool_, (len(linhas), h, w)),
             "y": (np.int16, (len(linhas), h, w)),
             "v": (np.bool_, (len(linhas), h, w)), "npv": (np.bool_, (len(linhas), h, w))}
    if manifesto.exists():
        m = json.loads(manifesto.read_text(encoding="utf-8"))
        if m["inventario"] != assinatura or any(sha(destino / f"{k}.npy") != m["hashes"][k] for k in specs):
            raise ValueError("cache divergente ou corrompido")
    else:
        arrays = {k: np.lib.format.open_memmap(destino / f"{k}.temporario.npy", mode="w+", dtype=d, shape=s)
                  for k, (d, s) in specs.items()}
        for i, r in enumerate(linhas):
            if r["grade"]["forma"] != [h, w]:
                raise ValueError("cache exige forma comum")
            c = r["caminhos"]
            arrays["x"][i], arrays["sensor"][i] = imagens(c["antes"], c["depois"])
            arrays["y"][i], arrays["v"][i], arrays["npv"][i] = rotulo(c["rotulo"], c["jrc"])
        for array in arrays.values():
            array.flush()
        del array, arrays
        for k in specs:
            (destino / f"{k}.temporario.npy").replace(destino / f"{k}.npy")
        m = {"inventario": assinatura, "hashes": {k: sha(destino / f"{k}.npy") for k in specs}}
        manifesto.write_text(json.dumps(m, indent=2), encoding="utf-8")
    arrays = {k: np.load(destino / f"{k}.npy", mmap_mode="r", allow_pickle=False) for k in specs}
    for i, r in enumerate(linhas):
        c = r["caminhos"]
        _CACHE_IMAGENS[(c["antes"], c["depois"])] = (arrays["x"], arrays["sensor"], i)
        _CACHE_ROTULOS[(c["rotulo"], c["jrc"])] = (arrays["y"], arrays["v"], arrays["npv"], i)
    imagens.cache_clear()
    rotulo.cache_clear()


def normalizacao(linhas):
    soma, quadrado, n = np.zeros(2), np.zeros(2), 0
    for r in linhas:
        x, v = imagens(r["caminhos"]["antes"], r["caminhos"]["depois"])
        x = recortar_db(x)
        for t in (x[:2], x[2:]):
            z = t[:, v].astype(np.float64)
            soma += z.sum(1)
            quadrado += (z * z).sum(1)
            n += z.shape[1]
    if not n:
        raise ValueError("treino sem pixels radiometricos validos")
    mu = soma / n
    sd = np.sqrt(np.maximum(quadrado / n - mu ** 2, 1e-6))
    return {"media": mu.tolist(), "desvio": sd.tolist(),
            "limites_db": LIMITES_DB,
            "origem": "somente imagens do treino inicial; congelada durante aquisicao"}
