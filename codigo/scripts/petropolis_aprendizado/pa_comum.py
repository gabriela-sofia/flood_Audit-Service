"""Caminhos, hashes e registro comuns da linhagem petropolis-aprendizado/v1.

A linhagem e aditiva: nao escreve fora de `local_runs/petropolis-aprendizado/v1/`
e nunca altera os produtores congelados que le (estudo multiperigos, v3, svc-03,
produto/). Cada etapa grava o que leu, o que escreveu e quanto custou em
`execucao_etapas.jsonl`, para que o manifesto final possa ser conferido.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "local_runs" / "petropolis-aprendizado" / "v1"
FONTES = OUT / "fontes"
TERRENO = OUT / "terreno"

ESTUDO = ROOT / "local_runs/petropolis-multiperigos/estudo_20261024_v1"
CHARTER = ROOT / "local_runs/transferencia-vulnerabilizados/v3/fontes/petropolis_charter0864/Eventos_Desastre.shp"
SVC03_TIF = ROOT / "local_runs/svc-03-grade/escore_petropolis.tif"
SVC03_RES = ROOT / "local_runs/svc-03-grade/resultado.json"

EPSG_PETROPOLIS = 31983
EPSG_IDAI = 32736
RES_M = 30.0
SEMENTE = 20261024

# Fontes que esta linhagem le e nao pode alterar; conferidas antes e depois.
PROTEGIDAS_GLOB = [
    ("local_runs/petropolis-multiperigos/estudo_20261024_v1", "**/*"),
    ("local_runs/transferencia-vulnerabilizados/v3/fontes/petropolis_charter0864", "*"),
    ("local_runs/transferencia-vulnerabilizados/v3", "manifesto*.json"),
    ("local_runs/transferencia-vulnerabilizados/v10_ciclo_contextual", "manifesto.json"),
    ("local_runs/svc-03-grade", "*petropolis*"),
    ("local_runs/svc-03-grade", "resultado.json"),
    ("produto/modelos", "*"),
    ("produto/grades", "*"),
]


def sha256(p: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bloco), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def hashes_protegidas() -> dict:
    saida = {}
    for base, padrao in PROTEGIDAS_GLOB:
        for p in sorted((ROOT / base).glob(padrao)):
            if p.is_file():
                saida[rel(p)] = sha256(p)
    return saida


def gravar_json(caminho: Path, objeto) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(objeto, ensure_ascii=False, indent=2, default=_padrao),
                       encoding="utf-8")
    return caminho


def ler_json(caminho: Path):
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def _padrao(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return rel(o)
    raise TypeError(type(o))


def ambiente() -> dict:
    versoes = {}
    for m in ["numpy", "pandas", "sklearn", "scipy", "geopandas", "rasterio", "shapely",
              "pyproj", "matplotlib", "whitebox"]:
        try:
            versoes[m] = __import__(m).__version__
        except Exception:                                            # noqa: BLE001
            versoes[m] = None
    return {"python": sys.version.split()[0], "executavel": sys.executable,
            "plataforma": platform.platform(), "versoes": versoes}


class Etapa:
    """Registra inicio, fim, custo e artefatos de uma etapa no diario da linhagem."""

    def __init__(self, nome: str):
        self.nome = nome
        self.t0 = time.time()
        self.c0 = time.process_time()
        self.saidas: list[str] = []
        self.entradas: list[str] = []

    def entrada(self, *ps):
        self.entradas += [rel(p) for p in ps]

    def saida(self, *ps):
        self.saidas += [rel(p) for p in ps]

    def fechar(self, **extra):
        OUT.mkdir(parents=True, exist_ok=True)
        reg = {"etapa": self.nome, "fim": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "parede_s": round(time.time() - self.t0, 1),
               "cpu_s": round(time.process_time() - self.c0, 1),
               "pid": os.getpid(), "comando": " ".join(sys.argv),
               "entradas": sorted(set(self.entradas)), "saidas": sorted(set(self.saidas)), **extra}
        with open(OUT / "execucao_etapas.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(reg, ensure_ascii=False, default=_padrao) + "\n")
        return reg
