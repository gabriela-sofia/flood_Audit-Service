"""F03: intervalo da vantagem da função transferida sobre o HAND nas cinco coortes de comunidades.

A tabela da v9 (b_tabular) registrou só a AUC pontual. As previsões por ponto ficaram em
`b_tabular/predicoes_{coorte}.parquet`; aqui o contraste Q_sel_ext - HAND é pareado nos mesmos
pontos de teste FCU e o IC95% vem de 2.000 reamostragens de comunidades inteiras (grupo FCU).
A referência é água observada pelo Sentinel-2 (MNDWI > 0) contra pontos observados sem água na
mesma data, isto é, ausência observada, e não fundo sem relato.

Regra declarada antes: "sustentado" se o limite inferior do IC95% for > 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts/vulnerabilizados"))
from comum_tv import AucPonderavel, intervalo, pesos_bootstrap  # noqa: E402

PRED = RAIZ / "local_runs/transferencia-vulnerabilizados/v9_evolucao_abrangente/b_tabular"
SAIDA = RAIZ / "local_runs/fundamentacao-artigo-20261028"
N_BOOT, SEMENTE = 2000, 20261028


def main():
    saida = {"regra": "sustentado se limite inferior do IC95% > 0", "grupo": "comunidade FCU", "bootstrap": N_BOOT,
             "referencia": "Sentinel-2 MNDWI > 0 contra observado sem agua na mesma cena", "por_coorte": {}}
    for c in ["rmpa", "manaus", "curitiba", "recife", "rmr"]:
        p = pd.read_parquet(PRED / f"predicoes_{c}.parquet")
        p = p[p.classe_territorial.eq("fcu")].reset_index(drop=True)
        cols = ["Q_sel_ext", "A0_hand", "menos_declive"]
        fun = {k: AucPonderavel(p.y, p[k]) for k in cols}
        pontual = {k: fun[k](np.ones(len(p))) for k in cols}
        reps = np.array([[fun[k](w) for k in cols] for w in pesos_bootstrap(p.grupo, N_BOOT, SEMENTE)])
        res = {"pontos": int(len(p)), "positivos": int(p.y.sum()), "comunidades": int(p.grupo.nunique()),
               "comunidades_com_positivo": int(p[p.y == 1].grupo.nunique()),
               "auc": {k: float(v) for k, v in pontual.items()},
               "ic95_auc_Q": intervalo(reps[:, 0], 0.95)}
        for ref, j in (("A0_hand", 1), ("menos_declive", 2)):
            ic = intervalo(reps[:, 0] - reps[:, j], 0.95)
            res[f"Q_menos_{ref}"] = {"delta": float(pontual["Q_sel_ext"] - pontual[ref]), "ic95": ic,
                                    "leitura": "sustentado" if ic and ic[0] > 0 else ("contrario" if ic and ic[1] < 0 else "inconclusivo")}
        saida["por_coorte"][c] = res
        print(c, json.dumps(res, default=float), flush=True)
    (SAIDA / "f03_ic_transferencia_coortes.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
