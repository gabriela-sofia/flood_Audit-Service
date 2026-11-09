"""Evolução com o acervo local (EAL) — utilidades comuns da linhagem `local_runs/evolucao-acervo-local/v1/`.

Toda fase devolve um dict com `estado` em ESTADOS. Nenhuma fase acessa rede: `bloquear_rede()` troca a conexão
de socket por uma recusa explícita, para que um caminho que acione rede implicitamente (Earth Engine, requests,
tile remoto do rasterio) falhe com nome e não em silêncio. O modo inspeção (`EAL_INSPECAO=1` ou `--inspecao`)
recusa qualquer escrita, inclusive na própria linhagem.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for s in ("evolucao_local", "ia", "mt12", "metodo", "servico", "sar", "geo", "terreno", "suscetibilidade", "manutencao", "chuva"):
    p = str(REPO / "scripts" / s)
    if p not in sys.path:
        sys.path.append(p)

RAIZ_EAL = REPO / "local_runs" / "evolucao-acervo-local"
LINHAGEM = RAIZ_EAL / "v1"
LINHAGEM2 = RAIZ_EAL / "v2"  # segunda etapa; a v1 fica preservada
ESTADOS = ("sucesso", "impedido", "sem_dados", "falha")


class RedeBloqueada(RuntimeError):
    pass


class EscritaRecusada(RuntimeError):
    pass


_socket_connect_original = socket.socket.connect


def bloquear_rede() -> None:
    def _recusa(self, endereco):  # noqa: ARG001
        raise RedeBloqueada(f"rede desativada nesta campanha (tentativa de conexão a {endereco!r})")
    socket.socket.connect = _recusa
    socket.socket.connect_ex = _recusa
    os.environ.setdefault("NO_PROXY", "*")
    os.environ["GDAL_HTTP_TIMEOUT"] = "1"
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".nenhuma"


def liberar_rede() -> None:
    """Só para testes: devolve a conexão original."""
    socket.socket.connect = _socket_connect_original


def inspecao() -> bool:
    return os.environ.get("EAL_INSPECAO") == "1"


def agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


NAO_E_EAL = ("resultado da linhagem EAL: experimento sobre o acervo local ja baixado, sem rede e sem rotulo novo; nao "
            "e validacao externa, nao promove modelo nem regiao e vale so para a populacao declarada no protocolo da fase")


def _declarado(p: Path, o):
    """Catraca de manifesto (19/10/2026): JSON sem `tipo` recebe o caminho dentro da linhagem como tipo e o `nao_e` da
    EAL. Quem ja declara o proprio tipo nao e alterado; artefatos ja gravados nao sao reescritos."""
    if not isinstance(o, dict) or "tipo" in o or "artefato" in o:
        return o
    rel = Path(p).resolve().relative_to(RAIZ_EAL.resolve()).with_suffix("").as_posix()
    return {"tipo": f"evolucao_acervo_local:{rel}", "nao_e": o.get("nao_e", NAO_E_EAL),
            **{k: v for k, v in o.items() if k != "nao_e"}}


def gravar(p: Path, o) -> Path:
    if inspecao():
        raise EscritaRecusada(f"modo inspeção: escrita recusada em {p}")
    p = Path(p)
    if not str(p.resolve()).startswith(str(RAIZ_EAL.resolve())):
        raise EscritaRecusada(f"escrita fora da linhagem recusada: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    if os.environ.get("EAL_ETAPA") == "2" and str(p.resolve()).startswith(str(LINHAGEM.resolve())):
        raise EscritaRecusada(f"etapa 2 nao escreve na linhagem v1: {p}")
    if p.suffix == ".json":
        tmp.write_text(json.dumps(_declarado(p, o), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    elif p.suffix == ".csv":
        o.to_csv(tmp, index=False)
    else:
        raise ValueError(f"formato não previsto: {p.suffix}")
    tmp.replace(p)
    return p


def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def resultado(fase: str, estado: str, **kw) -> dict:
    if estado not in ESTADOS:
        raise ValueError(f"estado inválido: {estado}")
    return {"fase": fase, "estado": estado, "quando_utc": agora(), **kw}


def livre_gib() -> float:
    return shutil.disk_usage(REPO).free / 1024 ** 3


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)
