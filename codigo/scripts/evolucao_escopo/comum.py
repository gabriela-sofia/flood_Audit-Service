"""Execucao aditiva do estudo de escopo: somente acervo local, versoes preservadas."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
OUT = RUNS / "evolucao-escopo" / "v1"
NAO_E = "desenvolvimento retrospectivo com acervo local; nao e confirmacao nova, promocao, previsao, profundidade ou identificacao causal"
for pasta in ("servico", "metodo", "geo", "sar", "terreno", "social", "ia", "planejamento", "fechamento", "recuperacao", "verificacao", "manutencao"):
    sys.path.append(str(REPO / "scripts" / pasta))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def agora():
    return datetime.now(timezone.utc).isoformat()


def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rel(p):
    return Path(p).resolve().relative_to(REPO).as_posix()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def gravar(p, valor):
    p = Path(p).resolve()
    if not p.is_relative_to(OUT.resolve()):
        raise ValueError("escrita fora da linhagem de evolucao")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".json":
        if not isinstance(valor, dict):
            valor = {"registros": valor}
        valor = {"tipo": "evolucao_escopo:" + p.relative_to(OUT).with_suffix("").as_posix(),
                 "nao_e": NAO_E, "gerado_por": rel(__file__), **valor}
        texto = json.dumps(valor, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(texto, encoding="utf-8")
    elif p.suffix == ".csv":
        tmp = p.with_suffix(".csv.tmp")
        valor.to_csv(tmp, index=False)
    elif p.suffix in (".md", ".txt"):
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(valor, encoding="utf-8")
    else:
        raise ValueError(p.suffix)
    tmp.replace(p)
    return p


def congelar(p, valor):
    p = Path(p)
    if p.exists():
        existente = ler(p)
        for k, v in valor.items():
            if existente.get(k) != v:
                raise ValueError(f"protocolo congelado diverge em {k}: use outra versao")
        return existente
    gravar(p, {**valor, "quando_utc": agora()})
    return ler(p)


def bloquear_rede():
    import socket
    def recusada(*args, **kwargs):
        raise RuntimeError("rede bloqueada na evolucao do acervo local")
    socket.socket.connect = recusada
    socket.socket.connect_ex = recusada


def nome_fase(fase):
    return OUT / fase / "resultado.json"
