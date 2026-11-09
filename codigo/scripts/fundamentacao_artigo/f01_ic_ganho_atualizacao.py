"""F01: intervalo do ganho de atualização nas coortes do ciclo v10, reamostrando comunidades.

O ciclo v10 registrou só médias por semente. Os pesos de cada atualização (theta, c) ficaram em
`ii_ciclo/lotes.parquet`, então as previsões do teste são reconstruídas sem rodar o ciclo de novo:
escore = Q + z @ theta + c. O contraste é pareado nos mesmos pontos de teste FCU:
AUC(estratégia, B=400) - AUC(congelado), média das 6 sementes, com bootstrap de comunidades (CD_FCU).

Regra declarada antes de olhar o intervalo: "sustentado" se o limite inferior do IC95% for > 0.
Não altera o ciclo, os pools nem o protocolo v10; escreve em local_runs/fundamentacao-artigo-20261028/.
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
from v10_comum import V10, auc, nucleo_limpo  # noqa: E402

SAIDA = RAIZ / "local_runs/fundamentacao-artigo-20261028"
B_FINAL, N_BOOT, SEMENTE_BOOT = 400, 2000, 20261028
ESTRATEGIAS = ["margem_v9_gate", "uniforme_gate"]


def main():
    from v8_04_campanha import FEAT, caracteristicas, coorte, z_externa
    Q = nucleo_limpo()
    mu, sd, med = Q.pad.mu[FEAT], Q.pad.sd[FEAT], Q.pad.med[FEAT]
    lotes = pd.read_parquet(V10 / "ii_ciclo/lotes.parquet")
    final = lotes[(lotes.pool == "todas") & (lotes.consultados == B_FINAL)]
    saida = {"regra": "sustentado se limite inferior do IC95% > 0", "B": B_FINAL, "bootstrap": N_BOOT,
             "grupo": "CD_FCU", "populacao": "pontos de teste FCU positivo/negativo", "por_coorte": {}}
    for c in ["rmpa", "manaus", "curitiba"]:
        d = coorte(c)
        x = caracteristicas(d)
        te = (d.particao.eq("teste") & d.estado_primaria.isin(["positivo", "negativo"])
              & d.classe_territorial.eq("fcu")).to_numpy()
        q, z = Q.escore(x)[te], z_externa(x, mu, sd, med)[te]
        y = d.estado_primaria.eq("positivo").to_numpy()[te].astype(int)
        grupos = d.CD_FCU.astype(str).to_numpy()[te]
        res = {"pontos": int(te.sum()), "positivos": int(y.sum()), "comunidades": int(len(set(grupos))),
               "auc_congelado": auc(y, q)}
        for estr in ESTRATEGIAS:
            linhas = final[(final.coorte == c) & (final.estrategia == estr)].sort_values("semente")
            escores = []
            for _, r in linhas.iterrows():
                th = np.asarray(json.loads(r.theta))
                escores.append(q + (z @ th + r.c if r.ativo else 0))
            aucs = [auc(y, s) for s in escores]
            conferencia = float(np.mean(aucs)) - float(linhas.auc_teste_fcu.mean())
            base, novos = AucPonderavel(y, q), [AucPonderavel(y, s) for s in escores]
            delta = np.array([np.mean([a(w) for a in novos]) - base(w)
                              for w in pesos_bootstrap(pd.Series(grupos), N_BOOT, SEMENTE_BOOT)])
            ic = intervalo(delta, 0.95)
            ganho = float(np.mean(aucs)) - res["auc_congelado"]
            res[estr] = {"auc_media_sementes": float(np.mean(aucs)), "ganho": ganho, "ic95": ic,
                         "sementes_favoraveis": int(sum(a > res["auc_congelado"] for a in aucs)),
                         "sementes_ativas": int(linhas.ativo.sum()),
                         "leitura": ("sustentado" if ic and ic[0] > 0 else "contrario" if ic and ic[1] < 0 else "inconclusivo")
                         if linhas.ativo.any() else "sem atualizacao (filtro de suporte)",
                         "conferencia_com_curvas": conferencia}
        saida["por_coorte"][c] = res
        print(c, json.dumps(res, default=float)[:600], flush=True)
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "f01_ic_ganho_atualizacao.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1, default=float),
                                                          encoding="utf-8")


if __name__ == "__main__":
    main()
