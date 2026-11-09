"""Caminhos, hashes e registro comuns da linhagem recife-aprendizado/v1.

Aditiva: escreve so em `local_runs/recife-aprendizado/v1/`. Le em modo somente
leitura o acervo de PROJETO (PE3D, cadastro EMLURB, base de enderecos), o
estudo de drenagem, a qualificacao SEDEC da v3, a grade/coorte da v6 e o nucleo
Q da v9/v10. Cada etapa registra entradas, saidas e custo em execucao_etapas.jsonl.
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
OUT = ROOT / "local_runs" / "recife-aprendizado" / "v1"
PROJ = Path(os.environ.get("REV_P_DADOS_PROJETO", str(ROOT / "data")))
PE3D_ZIPS = PROJ / "raw/recife/pe3d_mdt"
DREN = PROJ / "processed/recife/esig_emlurb_drenagem"
MACRO = PROJ / "processed/recife/esig_emlurb_macrodrenagem"
END = PROJ / "processed/recife/official_address_base"
ESTUDO = ROOT / "local_runs/recife-drenagem/estudo_20261024_v1"
V3R = ROOT / "local_runs/transferencia-vulnerabilizados/v3/recife"
SEDEC_RAW = ROOT / "local_runs/acq-recife-01-sedec/raw"
V6R = ROOT / "local_runs/transferencia-vulnerabilizados/v6_evidencia_robusta/recife_avaliacao"
V9P0 = ROOT / "local_runs/transferencia-vulnerabilizados/v9_evolucao_abrangente/p0_regularizacao"
EPSG = 31985
SEMENTE = 20261024

PROTEGIDAS = [
    (ESTUDO, "**/*"), (V3R, "*"), (V6R, "*"), (V9P0, "*"),
    (ROOT / "local_runs/transferencia-vulnerabilizados/v10_ciclo_contextual", "manifesto.json"),
    (ROOT / "local_runs/transferencia-vulnerabilizados/v8_validacao_integrada", "grafo_particoes.json"),
    (ROOT / "local_runs/ds-05-tabela-unica", "*"),
    (ROOT / "produto/modelos", "*"), (ROOT / "produto/grades", "*"),
    (PE3D_ZIPS, "*.zip"), (DREN, "*.gpkg"), (MACRO, "*.gpkg"), (END, "*.gpkg"),
    (SEDEC_RAW, "*.csv"),
]


def sha256(p: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bloco), b""):
            h.update(b)
    return h.hexdigest()


def rel(p) -> str:
    p = Path(p)
    for base in (ROOT, PROJ.parent):
        try:
            return str(p.resolve().relative_to(base)).replace("\\", "/")
        except ValueError:
            continue
    return str(p).replace("\\", "/")


def hashes_protegidas() -> dict:
    saida = {}
    for base, padrao in PROTEGIDAS:
        for p in sorted(Path(base).glob(padrao)):
            if p.is_file():
                saida[rel(p)] = sha256(p)
    return saida


def _padrao(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return rel(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(type(o))


def gravar_json(caminho: Path, objeto) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(objeto, ensure_ascii=False, indent=2, default=_padrao), encoding="utf-8")
    return caminho


def ler_json(caminho: Path):
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def ambiente() -> dict:
    versoes = {}
    for m in ["numpy", "pandas", "sklearn", "scipy", "geopandas", "rasterio", "shapely", "pyproj", "matplotlib", "whitebox"]:
        try:
            versoes[m] = __import__(m).__version__
        except Exception:                                            # noqa: BLE001
            versoes[m] = None
    return {"python": sys.version.split()[0], "executavel": sys.executable,
            "plataforma": platform.platform(), "versoes": versoes}


class Etapa:
    def __init__(self, nome: str):
        self.nome, self.t0, self.c0 = nome, time.time(), time.process_time()
        self.saidas, self.entradas = [], []

    def entrada(self, *ps):
        self.entradas += [rel(p) for p in ps]

    def saida(self, *ps):
        self.saidas += [rel(p) for p in ps]

    def fechar(self, **extra):
        OUT.mkdir(parents=True, exist_ok=True)
        reg = {"etapa": self.nome, "fim": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "parede_s": round(time.time() - self.t0, 1), "cpu_s": round(time.process_time() - self.c0, 1),
               "pid": os.getpid(), "comando": " ".join(sys.argv),
               "entradas": sorted(set(self.entradas)), "saidas": sorted(set(self.saidas)), **extra}
        with open(OUT / "execucao_etapas.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(reg, ensure_ascii=False, default=_padrao) + "\n")
        return reg
