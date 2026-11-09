"""F07: transferência por estrato territorial, com a mesma reamostragem do F03 e previsões já salvas.

O F03 deu intervalo só às FCU. Aqui o mesmo contraste vai à coorte inteira e aos estratos de baixa renda e de
comparador urbano, porque a tese é de transferência para o destino e FCU é um estrato avaliado dentro dele.
Nenhum modelo é ajustado. Vale a regra da v8 (`v8_04b_analise.contrastes`): com menos de 20 positivos ou menos de
10 grupos com positivo, o estrato é descritivo e não recebe intervalo.

Uso:
    python scripts/fundamentacao_artigo/f07_estratos_coortes.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts/vulnerabilizados"))
from comum_tv import AucPonderavel, intervalo, pesos_bootstrap  # noqa: E402

PRED = RAIZ / "local_runs/transferencia-vulnerabilizados/v9_evolucao_abrangente/b_tabular"
SAIDA = RAIZ / "local_runs/fundamentacao-artigo-20261106"
N_BOOT, SEMENTE = 2000, 20261106
COLUNAS = ["Q_sel_ext", "A0_hand", "menos_declive"]
COORTES = ["rmpa", "manaus", "curitiba", "recife", "rmr"]
ESTRATOS = {"todas": None, "fcu": "fcu", "baixa_renda": "baixa_renda", "comparador": "comparador_urbano"}


def leitura(ic):
    if not ic:
        return "sem_intervalo"
    return "sustentado" if ic[0] > 0 else ("contrario" if ic[1] < 0 else "inconclusivo")


def medir(p):
    positivos, grupos_pos = int(p.y.sum()), int(p[p.y == 1].grupo.nunique())
    fun = {k: AucPonderavel(p.y, p[k]) for k in COLUNAS}
    pontual = {k: fun[k](np.ones(len(p))) for k in COLUNAS}
    res = {"pontos": int(len(p)), "positivos": positivos, "grupos": int(p.grupo.nunique()),
           "grupos_com_positivo": grupos_pos, "auc": {k: float(v) for k, v in pontual.items()}}
    if positivos < 20 or grupos_pos < 10:
        res["estado"] = "descritivo_suporte_insuficiente"
        return res
    reps = np.array([[fun[k](w) for k in COLUNAS] for w in pesos_bootstrap(p.grupo, N_BOOT, SEMENTE)])
    res["estado"] = "avaliado"
    res["ic95_auc_Q"] = intervalo(reps[:, 0], 0.95)
    for ref, j in (("A0_hand", 1), ("menos_declive", 2)):
        ic = intervalo(reps[:, 0] - reps[:, j], 0.95)
        res[f"Q_menos_{ref}"] = {"delta": float(pontual["Q_sel_ext"] - pontual[ref]), "ic95": ic, "leitura": leitura(ic)}
    return res


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    saida = {"tipo": "fundamentacao_artigo:f07/estratos_coortes",
             "nao_e": "novo treino nem confirmacao: reanalise de previsoes salvas em coortes ja examinadas; AUC mede ordenacao",
             "gerado_por": "scripts/fundamentacao_artigo/f07_estratos_coortes.py",
             "regra": "sustentado se limite inferior do IC95% > 0; descritivo se positivos < 20 ou grupos com positivo < 10 (regra da v8)",
             "bootstrap": N_BOOT, "semente": SEMENTE, "por_coorte": {}}
    for coorte in COORTES:
        p = pd.read_parquet(PRED / f"predicoes_{coorte}.parquet", columns=["grupo", "classe_territorial", "y", *COLUNAS])
        p = p[p.y.isin([0, 1])].reset_index(drop=True)
        saida["por_coorte"][coorte] = {}
        for nome, classe in ESTRATOS.items():
            d = p if classe is None else p[p.classe_territorial.eq(classe)].reset_index(drop=True)
            saida["por_coorte"][coorte][nome] = medir(d)
            print(coorte, nome, json.dumps(saida["por_coorte"][coorte][nome], default=float), flush=True)
        (SAIDA / "f07_estratos_coortes.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
