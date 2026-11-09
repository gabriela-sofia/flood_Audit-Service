"""Pares SAR: a imagem anterior e a observacao do evento nao sao intercambiaveis.

O pacote S1GFloods usa PNG renderizado, nao VV/VH em dB. Nao se misturam
esses canais com o Sen1Floods11 nem se inventam datas a partir do nome.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

_CACHE = {}


def ler_imagem(caminho: Path) -> np.ndarray:
    with Image.open(caminho) as imagem:
        a = np.asarray(imagem).copy()
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"{caminho}: esperado PNG RGB uint8, nao converter silenciosamente")
    return a


@lru_cache(maxsize=96)
def ler_par(raiz: str, nome: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if raiz in _CACHE:
        imagens, rotulos, indices = _CACHE[raiz]
        i = indices[nome]
        return imagens[i, :, :, :3], imagens[i, :, :, 3:], rotulos[i].astype(np.int64)
    p = Path(raiz)
    antes = ler_imagem(p / "A" / nome)
    depois = ler_imagem(p / "B" / nome)
    with Image.open(p / "Label" / nome) as imagem:
        alvo = np.asarray(imagem).copy()
    if alvo.ndim == 3:
        if not np.array_equal(alvo[..., 0], alvo[..., 1]) or not np.array_equal(
                alvo[..., 0], alvo[..., 2]):
            raise ValueError(f"{nome}: rotulo colorido sem legenda")
        alvo = alvo[..., 0]
    if antes.shape != depois.shape or alvo.shape != antes.shape[:2]:
        raise ValueError(f"{nome}: grades do par e do rotulo divergem")
    if not set(np.unique(alvo)).issubset({0, 255}):
        raise ValueError(f"{nome}: rotulo fora de 0/255; nao binarizar desconhecidos")
    return antes, depois, (alvo == 255).astype(np.int64)


def digest(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str((a.shape, str(a.dtype))).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def inventariar(raiz: Path, mapa: Path | None = None, progresso=None) -> dict:
    """Sem mapa de eventos, a particao e tecnica e exclusivamente exploratoria.

    Componentes de imagens identicas ficam juntos. Isso evita copia exata,
    mas NAO garante separacao de cenas adjacentes, territorios ou eventos.
    """
    ler_par.cache_clear()
    _CACHE.pop(str(raiz.resolve()), None)
    nomes = sorted(p.name for p in (raiz / "A").glob("*.png"))
    if not nomes or any(set(nomes) != {p.name for p in (raiz / sub).glob("*.png")}
                        for sub in ("B", "Label")):
        raise ValueError("A, B e Label precisam conter os mesmos pares nao vazios")
    pais = {n: n for n in nomes}

    def encontrar(n):
        while pais[n] != n:
            pais[n] = pais[pais[n]]
            n = pais[n]
        return n

    hashes = {}
    vistos = {}
    for i, nome in enumerate(nomes):
        a, b, y = ler_par(str(raiz), nome)
        ha, hb = digest(a), digest(b)
        hashes[nome] = {"antes": ha, "depois": hb, "rotulo": digest(y)}
        for h in (ha, hb):
            if h in vistos:
                pais[encontrar(nome)] = encontrar(vistos[h])
            vistos[h] = nome
        if progresso and (i + 1) % 500 == 0:
            progresso(i + 1, len(nomes))
    componentes = {}
    for n in nomes:
        componentes.setdefault(encontrar(n), []).append(n)
    grupos = {n: min(ns) for ns in componentes.values() for n in ns}
    declarado = json.loads(mapa.read_text(encoding="utf-8")) if mapa else None
    if declarado is not None and set(declarado) != set(nomes):
        raise ValueError("mapa de eventos deve cobrir exatamente os pares")
    linhas = []
    for nome in nomes:
        if declarado is not None:
            item = declarado[nome]
            if not item.get("evento") or item.get("split") not in {
                    "treino", "validacao", "teste"}:
                raise ValueError("mapa exige evento e split explicitos")
            split, evento = item["split"], item["evento"]
        else:
            balde = int(hashlib.sha256(grupos[nome].encode()).hexdigest()[:8], 16) % 10
            split = "teste" if balde == 0 else "validacao" if balde == 1 else "treino"
            evento = None
        linhas.append({"nome": nome, "split": split, "evento": evento,
                       "componente_copia": grupos[nome], "hashes": hashes[nome]})
    for chave in ("componente_copia", "evento"):
        divisao = {}
        for r in linhas:
            if r[chave] is not None:
                divisao.setdefault(r[chave], set()).add(r["split"])
        if any(len(v) > 1 for v in divisao.values()):
            raise ValueError(f"{chave} compartilhado entre particoes")
    if any(not any(r["split"] == s for r in linhas)
           for s in ("treino", "validacao", "teste")):
        raise ValueError("particao vazia")
    return {"raiz": str(raiz.resolve()), "pares": linhas,
            "regime": "exploratorio",
            "eventos_identificados": declarado is not None,
            "independencia_espacial": "nao_verificada: PNG sem georreferencia",
            "unidade": "RGB renderizado 0..255, nao dB",
            "ordem_temporal": "A antes e B evento conforme publicacao; datas ausentes",
            "fonte": "https://github.com/Tamer-Saleh/S1GFlood-Detection"}


def materializar_cache(inventario: dict, destino: Path) -> None:
    """Poucos arquivos sequenciais evitam reabrir 16 mil PNGs a cada epoca.

    O cache nao muda os pixels. Hashes completos e o inventario permitem
    recusar um cache antigo ou corrompido, sem depender so de sua existencia.
    """
    from adquirir_pares_s1gfloods import hash_arquivo

    destino.mkdir(parents=True, exist_ok=True)
    assinatura = hashlib.sha256(json.dumps(inventario, sort_keys=True).encode()).hexdigest()
    m = destino / "cache.json"
    arquivos = [destino / "imagens.npy", destino / "rotulos.npy"]
    if m.exists():
        dados = json.loads(m.read_text(encoding="utf-8"))
        if dados["inventario"] != assinatura or any(
                hash_arquivo(p) != dados["hashes"][p.name] for p in arquivos):
            raise ValueError("cache divergente: usar uma rodada nova")
    else:
        raiz = inventario["raiz"]
        linhas = inventario["pares"]
        a, _, _ = ler_par(raiz, linhas[0]["nome"])
        h, w = a.shape[:2]
        temporarios = [p.with_suffix(".temporario.npy") for p in arquivos]
        imagens = np.lib.format.open_memmap(temporarios[0], mode="w+", dtype=np.uint8,
                                            shape=(len(linhas), h, w, 6))
        rotulos = np.lib.format.open_memmap(temporarios[1], mode="w+", dtype=np.uint8,
                                            shape=(len(linhas), h, w))
        for i, r in enumerate(linhas):
            a, b, y = ler_par(raiz, r["nome"])
            if a.shape[:2] != (h, w):
                raise ValueError("cache exige chips de mesma forma")
            imagens[i, :, :, :3], imagens[i, :, :, 3:] = a, b
            rotulos[i] = y
        imagens.flush()
        rotulos.flush()
        del imagens, rotulos
        for origem, alvo in zip(temporarios, arquivos):
            origem.replace(alvo)
        dados = {"inventario": assinatura,
                 "hashes": {p.name: hash_arquivo(p) for p in arquivos}}
        m.write_text(json.dumps(dados, indent=2), encoding="utf-8")
    _CACHE[inventario["raiz"]] = (
        np.load(arquivos[0], mmap_mode="r", allow_pickle=False),
        np.load(arquivos[1], mmap_mode="r", allow_pickle=False),
        {r["nome"]: i for i, r in enumerate(inventario["pares"])})
    ler_par.cache_clear()


def normalizacao(raiz: str, linhas: list[dict]) -> dict:
    """Mesma escala para as duas datas, calculada somente no treino."""
    soma = np.zeros(3, dtype=np.float64)
    quadrado = soma.copy()
    n = 0
    for r in linhas:
        if r["split"] != "treino":
            continue
        for a in ler_par(raiz, r["nome"])[:2]:
            x = a.reshape(-1, 3).astype(np.float64) / 255
            soma += x.sum(0)
            quadrado += (x * x).sum(0)
            n += len(x)
    if n == 0:
        raise ValueError("normalizacao sem treino")
    media = soma / n
    desvio = np.sqrt(np.maximum(quadrado / n - media ** 2, 1e-8))
    return {"media": media.tolist(), "desvio": desvio.tolist(),
            "origem": "somente pares de treino, duas datas"}


def entrada(a: np.ndarray, b: np.ndarray, estat: dict, modo: str) -> np.ndarray:
    """Seis canais nos tres bracos: a capacidade da U-Net nao muda."""
    media = np.asarray(estat["media"], dtype=np.float32)
    desvio = np.asarray(estat["desvio"], dtype=np.float32)
    antes = (a.astype(np.float32) / 255 - media) / desvio
    depois = (b.astype(np.float32) / 255 - media) / desvio
    if modo == "pos":
        auxiliar = depois
    elif modo == "pre_pos":
        auxiliar = antes
    elif modo == "pos_delta":
        auxiliar = depois - antes
    else:
        raise ValueError(f"modo desconhecido: {modo}")
    return np.concatenate([depois, auxiliar], axis=2).transpose(2, 0, 1).copy()
