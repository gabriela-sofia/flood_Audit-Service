"""F06: a função transferida ordena as ruas relatadas de Recife acima do acaso? IC da própria AUC.

A campanha de Recife deu IC só para a diferença contra o HAND. Aqui o IC é da AUC dentro do bairro
(A1_Q), reamostrando bairros de teste inteiros em 2.000 réplicas, com a mesma definição da campanha:
u_i = fração das ruas sem relato do mesmo bairro abaixo da rua relatada i; AUC = média dos u_i.
Regra: sustentado se o limite inferior do IC95% for > 0,5.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
PRED = RAIZ / "local_runs/recife-aprendizado/v1/resultados/pred_ruas.parquet"
SAIDA = RAIZ / "local_runs/fundamentacao-artigo-20261028"
N_BOOT, SEMENTE = 2000, 20261028


def u_por_rua(t: pd.DataFrame, col: str) -> pd.DataFrame:
    linhas = []
    for bairro, g in t.groupby("CBAIRRCODI"):
        fundo = g.loc[g.y == 0, col].to_numpy()
        for v in g.loc[g.y == 1, col]:
            linhas.append({"bairro": bairro, "u": ((fundo < v).sum() + 0.5 * (fundo == v).sum()) / len(fundo)})
    return pd.DataFrame(linhas)


def main():
    p = pd.read_parquet(PRED)
    t = p[p.papel == "teste"]
    saida = {"regra": "sustentado se limite inferior do IC95% > 0,5", "grupo": "bairro", "bootstrap": N_BOOT}
    rng = np.random.default_rng(SEMENTE)
    for col, nome in (("A1_Q", "funcao_transferida"), ("A0_regra", "hand")):
        u = u_por_rua(t, col)
        bairros = u.bairro.unique()
        soma = u.groupby("bairro").u.agg(["sum", "count"])
        reps = []
        for _ in range(N_BOOT):
            esc = soma.loc[rng.choice(bairros, len(bairros))]
            reps.append(esc["sum"].sum() / esc["count"].sum())
        lo, hi = np.percentile(reps, [2.5, 97.5])
        saida[nome] = {"auc_dentro_do_bairro": float(u.u.mean()), "ic95": [float(lo), float(hi)],
                       "ruas_relatadas": int(len(u)), "bairros": int(len(bairros)), "acima_do_acaso": bool(lo > 0.5)}
        rng = np.random.default_rng(SEMENTE)
    print(json.dumps(saida, ensure_ascii=False))
    (SAIDA / "f06_ic_auc_ruas_recife.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
