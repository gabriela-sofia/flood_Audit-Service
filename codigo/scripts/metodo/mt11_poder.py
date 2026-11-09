"""MT11, preparo: poder do teste transdutivo pela regra do projeto, a partir do MT10 pré-evento.

Reamostra as diferenças por componente F_transf - Q da sensibilidade só com datas anteriores
ao evento (18 componentes CEMS), centradas e deslocadas para cada efeito verdadeiro, e aplica
a regra de decisão: sustentado se média >= +0,01 e IC95% bootstrap > 0; ganho relevante
excluído se o limite superior < +0,01. Supõe que a dispersão entre unidades GFD seja a dos
componentes CEMS, o que é premissa e não medida.
Uso: python scripts/metodo/mt11_poder.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FONTE = REPO / "local_runs/mecanismos-transferiveis/v10/teto_transdutivo_v1/sensibilidade_pre_evento.json"
SAIDA = REPO / "local_runs/mt11-transdutivo/poder.json"
UNIDADES = (12, 16, 20, 24, 32)
EFEITOS = (0.0, .010, .014, .020, .029)
REPLICAS, BOOT, MARGEM = 1000, 1000, .01
NIVEIS = (.95, 1 - .05 / 3)   # 1-0,05/3: três primárias do MT11 (HQ1, HQ2, HF)


def main():
    d = np.array(list(json.loads(FONTE.read_text(encoding="utf-8"))["contrastes"]["F_transf-Q"]["por_componente"].values()))
    centro = d - d.mean()
    rng = np.random.default_rng(20261017)
    tab = []
    for nivel in NIVEIS:
        q = (1 - nivel) / 2
        for n in UNIDADES:
            for delta in EFEITOS:
                sus = exc = 0
                for _ in range(REPLICAS):
                    x = rng.choice(centro, n) + delta
                    lo, hi = np.quantile(rng.choice(x, (BOOT, n)).mean(1), [q, 1 - q])
                    sus += x.mean() >= MARGEM and lo > 0
                    exc += hi < MARGEM
                tab.append({"nivel": nivel, "unidades": n, "efeito": delta, "p_sustentado": sus / REPLICAS,
                            "p_ganho_excluido": exc / REPLICAS})
    out = {"tipo": "mt11_poder_simulado", "fonte": str(FONTE.relative_to(REPO)), "n_fonte": len(d),
           "media_fonte": float(d.mean()), "dp_fonte": float(d.std(ddof=1)), "tabela": tab,
           "premissa": "dispersão entre unidades novas igual à dos 18 componentes CEMS do MT10"}
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in tab:
        print(r)


if __name__ == "__main__":
    main()
