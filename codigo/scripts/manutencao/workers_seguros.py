"""Workers e etapas com propagacao de falha e completude verificada (17/10/2026).

Incidente que motivou: na campanha `fechamento_aprendizado_transferencia_v1`, `executar_paralelo`
devolvia os codigos dos workers mas a CLI saia com 0; o orquestrador seguiu para o modelo final
com um worker relancado ainda ativo, fora da posse da trava.

Regras:
  - o pai espera TODOS os workers, mesmo depois de um falhar: nenhum fica vivo fora da trava;
  - codigo do conjunto = 0 so se todos sairem com 0; pausa (3) e propagada como 3; outro erro como 1;
  - etapa seguinte so roda com codigo 0 E manifesto de completude (todas as saidas esperadas
    existem e passam na checagem de integridade).

Uso (biblioteca):
    r = rodar_workers([[py, script, "--worker", "0"], ...], [log0, log1])
    sys.exit(r["codigo"])
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable

PAUSA = 3


def codigo_do_conjunto(codigos: list[int]) -> int:
    if all(c == 0 for c in codigos):
        return 0
    if all(c in (0, PAUSA) for c in codigos):
        return PAUSA
    return 1


def rodar_workers(cmds: list[list[str]], logs: list[Path], cwd: Path | None = None) -> dict:
    """Inicia todos, espera todos, devolve codigos e o codigo do conjunto. Nunca abandona um vivo."""
    t0 = time.monotonic()
    procs = []
    for cmd, log in zip(cmds, logs):
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        f = open(log, "a", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd), f))
    codigos = []
    for p, f in procs:
        codigos.append(p.wait())
        f.close()
    return {"codigos": codigos, "codigo": codigo_do_conjunto(codigos), "segundos_relogio": time.monotonic() - t0}


def json_valido(p: Path) -> bool:
    try:
        json.loads(Path(p).read_text(encoding="utf-8"))
        return True
    except (OSError, ValueError):
        return False


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def manifesto_completude(esperados: Iterable[Path], checar: Callable[[Path], bool] | None = None,
                         destino: Path | None = None) -> dict:
    """Todas as saidas esperadas existem e passam na checagem? Grava o manifesto se `destino`."""
    faltando, invalidos, ok = [], [], {}
    for p in map(Path, esperados):
        if not p.exists():
            faltando.append(str(p))
        elif checar is not None and not checar(p):
            invalidos.append(str(p))
        else:
            ok[str(p)] = sha256(p)
    r = {"tipo": "manifesto_de_completude", "completo": not faltando and not invalidos,
         "esperados": len(ok) + len(faltando) + len(invalidos), "ok": len(ok),
         "faltando": faltando, "invalidos": invalidos, "hashes": ok}
    if destino is not None:
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        Path(destino).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def exigir(codigo: int, completude: dict | None = None, etapa: str = "") -> None:
    """Bloqueia a etapa seguinte. Levanta SystemExit com o codigo propagado."""
    if codigo != 0:
        raise SystemExit(codigo)
    if completude is not None and not completude["completo"]:
        raise SystemExit(f"{etapa}: saidas ausentes/invalidas ({len(completude['faltando'])}/{len(completude['invalidos'])})")
