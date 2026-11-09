"""Fecha o manifesto da linhagem depois da ultima mutacao e confere.

`--conferir` recalcula os hashes de tudo que o manifesto lista e das fontes protegidas.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from pa_comum import OUT, ROOT, ambiente, gravar_json, hashes_protegidas, ler_json, rel, sha256

EXTERNOS = ["docs/PETROPOLIS_APRENDIZADO_ENCOSTAS_RESULTADOS_2026_10_24.md",
            "docs/ARTIGO_PETROPOLIS_ENCOSTAS_SECOES_2026_10_24.md",
            "tests/test_petropolis_aprendizado_v1.py"]


def listar():
    arquivos = {rel(p): sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "manifesto.json"}
    for p in sorted((ROOT / "scripts/petropolis_aprendizado").glob("*.py")):
        arquivos[rel(p)] = sha256(p)
    for x in EXTERNOS:
        arquivos[x] = sha256(ROOT / x)
    return arquivos


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--conferir", action="store_true")
    ns = a.parse_args()
    m = OUT / "manifesto.json"
    if ns.conferir:
        man = ler_json(m)
        dif = [k for k, h in man["arquivos"].items() if not (ROOT / k).exists() or sha256(ROOT / k) != h]
        ini = ler_json(OUT / "estado_inicial.json")["protegidas_sha256"]
        agora = hashes_protegidas()
        prot = [k for k in ini if agora.get(k) != ini[k]]
        print({"arquivos": len(man["arquivos"]), "divergentes": dif, "protegidas_alteradas": prot})
        return 0 if not dif and not prot else 1
    ini = ler_json(OUT / "estado_inicial.json")["protegidas_sha256"]
    agora = hashes_protegidas()
    man = {"linhagem": "petropolis-aprendizado/v1", "fechado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "ambiente": ambiente(), "protocolo_sha256": sha256(OUT / "protocolo/protocolo_v1.json"),
           "emenda1_sha256": sha256(OUT / "protocolo/emenda_1.json"),
           "verificacao_ok": ler_json(OUT / "verificacao_independente.json")["ok"],
           "protegidas": {"n": len(ini), "alteradas": [k for k in ini if agora.get(k) != ini[k]]},
           "arquivos": listar(),
           "nota": "fechado depois da ultima mutacao; qualquer arquivo novo ou alterado depois disto e fora do manifesto"}
    gravar_json(m, man)
    print(f"manifesto: {len(man['arquivos'])} arquivos; protegidas alteradas: {man['protegidas']['alteradas']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
