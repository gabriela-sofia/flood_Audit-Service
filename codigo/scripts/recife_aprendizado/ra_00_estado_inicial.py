"""Estado inicial: hashes protegidos (inclusive fontes de PROJETO lidas), ambiente, trava, energia, processos."""
from __future__ import annotations

import subprocess
import sys

from ra_comum import OUT, ROOT, Etapa, ambiente, gravar_json, hashes_protegidas

sys.path.insert(0, str(ROOT / "scripts" / "manutencao"))
import executar_serializado as ser  # noqa: E402


def main() -> int:
    destino = OUT / "estado_inicial.json"
    if destino.exists():
        print("estado inicial ja registrado; a v1 nao e sobrescrita")
        return 1
    e = Etapa("ra_00_estado_inicial")
    try:
        procs = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python*", "/FO", "CSV"],
                               capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:                                         # noqa: BLE001
        procs = f"falha: {exc}"
    estado = {"ambiente": ambiente(), "energia": ser.energia(), "trava_pesada": ser.estado(),
              "processos_python": procs, "protegidas_sha256": hashes_protegidas(),
              "nota": "hashes antes de qualquer mutacao; PROJETO aberto so para leitura"}
    gravar_json(destino, estado)
    e.saida(destino)
    e.fechar(n_protegidas=len(estado["protegidas_sha256"]))
    print(f"{len(estado['protegidas_sha256'])} arquivos protegidos registrados; trava={estado['trava_pesada']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
