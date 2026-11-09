"""HydroBASINS v1c para as regioes que faltavam (Africa, Asia, America do Norte, Oceania).

MT01 (mecanismos transferiveis, 10/10/2026) usa eventos CEMS, Sen1Floods11 e UFO
fora da America do Sul e da Europa. Sem bacia nesses continentes, a dependencia
montante-jusante e o "mesmo sistema fluvial" so poderiam ser aproximados por
distancia -- e o Mississippi mostra que distancia nao basta (Omaha, Kansas e
St. Louis ficam a centenas de km e drenam para o mesmo exutorio).

Nao edita `scripts/geo/bacias_hydrobasins.py`: o hash dele esta no pre-registro
do TT01. As regioes novas entram em `REGIOES` so em tempo de execucao, com o
tamanho de cada zip conferido no servidor em 10/10/2026 (HEAD Content-Length).
Mesma licenca ja aceita para SA/EU (HydroSHEDS v1, sem redistribuicao avulsa).

nao_e: nao e rotulo, nao e mancha, nao muda nenhuma particao ja registrada.

Uso:
    python scripts/metodo/mt01_bacias_globais.py --converter af as na au
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "geo"))
import bacias_hydrobasins as hb  # noqa: E402

BYTES = {"af": 549_296_874, "as": 376_386_706, "na": 316_655_770, "au": 194_799_278}
NIVEIS = (6, 8, 12)


def registrar_regioes() -> dict:
    """Acrescenta as regioes novas ao dicionario do modulo (copia em memoria; arquivo intocado)."""
    for r, b in BYTES.items():
        hb.REGIOES.setdefault(r, {"zip": hb.RAIZ / f"bruto_{r}" / f"hybas_{r}_lev01-12_v1c.zip",
                                  "tabelas": hb.RAIZ / f"tabelas_{r}", "bytes": b})
    return hb.REGIOES


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--converter", nargs="+", choices=sorted(BYTES))
    a = ap.parse_args()
    registrar_regioes()
    for r in a.converter or []:
        if (hb.REGIOES[r]["tabelas"] / "manifesto.json").exists():
            print(r, "ja convertida", flush=True)
            continue
        t0 = time.time()
        m = hb.converter(niveis=NIVEIS, regiao=r)
        print(r, json.dumps({n: v["topologia"]["poligonos"] for n, v in m["niveis"].items()}),
              round(time.time() - t0), "s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
