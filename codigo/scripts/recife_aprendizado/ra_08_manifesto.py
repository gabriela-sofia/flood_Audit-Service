"""Fecha o manifesto da linhagem depois da ultima mutacao; `--conferir` recalcula tudo."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from ra_comum import OUT, ROOT, ambiente, gravar_json, hashes_protegidas, ler_json, rel, sha256

EXTERNOS = ["docs/RECIFE_APRENDIZADO_ALAGAMENTO_RESULTADOS_2026_10_24.md", "tests/test_recife_aprendizado_v1.py"]


def listar():
    arq = {rel(p): sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "manifesto.json"}
    arq.update({rel(p): sha256(p) for p in sorted((ROOT / "scripts/recife_aprendizado").glob("*.py"))})
    arq.update({x: sha256(ROOT / x) for x in EXTERNOS})
    return arq


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--conferir", action="store_true")
    ns = a.parse_args()
    m = OUT / "manifesto.json"
    ini = ler_json(OUT / "estado_inicial.json")["protegidas_sha256"]
    if ns.conferir:
        man = ler_json(m)
        dif = [k for k, h in man["arquivos"].items() if not (ROOT / k).exists() or sha256(ROOT / k) != h]
        agora = hashes_protegidas()
        prot = [k for k in ini if agora.get(k) != ini[k]]
        print({"arquivos": len(man["arquivos"]), "divergentes": dif, "protegidas_alteradas": prot})
        return 0 if not dif and not prot else 1
    agora = hashes_protegidas()
    gravar_json(m, {"linhagem": "recife-aprendizado/v1", "fechado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "ambiente": ambiente(), "protocolo_sha256": sha256(OUT / "protocolo/protocolo_v1.json"),
                    "emendas_sha256": {n: sha256(OUT / "protocolo" / n) for n in ("emenda_1.json", "emenda_2.json")},
                    "verificacao_ok": ler_json(OUT / "verificacao_independente.json")["ok"],
                    "protegidas": {"n": len(ini), "alteradas": [k for k in ini if agora.get(k) != ini[k]]},
                    "arquivos": listar(), "nota": "fechado depois da ultima mutacao"})
    print("manifesto fechado:", len(ler_json(m)["arquivos"]), "arquivos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
