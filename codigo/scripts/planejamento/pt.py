"""PT — planejamento-tese. CLI por fases, sem rede, saída JSON.

    python -X utf8 scripts/planejamento/pt.py <fase>

Fases leves: matriz | aprendizado | suporte | ledger | jornada | entrega
Fase pesada (só pelo serializador, na tomada): contexto_v2
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FASES = {"contexto_v2": ("pt_contexto_v2", "executar"), "contexto_v2_congelar": ("pt_contexto_v2", "congelar"),
         "aprendizado": ("pt_aprendizado", "executar"), "suporte": ("pt_suporte", "executar"),
         "ledger": ("pt_ledger", "executar"), "jornada": ("pt_jornada", "executar"),
         "matriz": ("pt_matriz", "executar"), "entrega": ("pt_entrega", "executar"),
         "social_equivalencia": ("pt_social_equivalencia", "executar"),
         "n2_congelar": ("pt_n2", "congelar"), "n2_piloto": ("pt_n2", "piloto"), "n2": ("pt_n2", "executar"),
         "n2_analisar": ("pt_n2", "analisar"), "continuidade": ("pt_continuidade", "executar"),
         "suporte_aninhado": ("pt_suporte_aninhado", "executar"),
         "jornada_integrada": ("pt_jornada_integrada", "executar"),
         "aquisicao_congelar": ("pt_aquisicao_laco", "congelar"), "aquisicao_laco": ("pt_aquisicao_laco", "executar"),
         "tipagem_cems06": ("pt_tipagem_cems06", "executar"), "escrita": ("pt_escrita", "executar"),
         "correspondencia": ("pt_correspondencia", "executar"),
         "sar_terreno_congelar": ("pt_sar_terreno_aoi", "congelar"), "sar_terreno": ("pt_sar_terreno_aoi", "executar"),
         "sar_terreno_sensor": ("pt_sar_terreno_aoi", "exploratorio_sensor"), "sar_terreno_datas": ("pt_sar_terreno_aoi", "auditoria_datas"),
         "jornada_aoi": ("pt_jornada_aoi", "executar"),
         "impedimentos": ("pt_impedimentos", "executar"), "capsula": ("pt_capsula", "executar")}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fase", choices=list(FASES))
    a = ap.parse_args()
    import pt_comum
    pt_comum.bloquear_rede()
    mod, fn = FASES[a.fase]
    try:
        r = getattr(importlib.import_module(mod), fn)()
        r = {"fase": a.fase, "estado": "sucesso", **(r if isinstance(r, dict) else {"saida": r})}
    except Exception as e:  # noqa: BLE001
        r = {"fase": a.fase, "estado": "falha", "erro": f"{type(e).__name__}: {e}", "rastro": traceback.format_exc()[-3000:]}
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str)[:15000])
    return 0 if r["estado"] == "sucesso" else 1


if __name__ == "__main__":
    raise SystemExit(main())
