"""F04: encosta em Petrópolis (acima do acaso? água separa de encosta?) e custo de retenção na RMPA.

Parte A, Petrópolis 2022, previsões salvas em `petropolis-aprendizado/v1/resultados/pred_teste_2022.parquet`:
  AUC por cicatriz (média das células da cicatriz contra o fundo, depois média das cicatrizes), com
  bootstrap conjunto de cicatrizes e de blocos de fundo de 500 m, como na campanha original.
  A1_log = relação aprendida no Idai; A0_regra = declive; D2_agua_svc03 = escore de água servido.
  Perguntas: (a) a relação do Idai ordena acima do acaso (IC > 0,5)? (b) o escore de água ordena as
  cicatrizes abaixo do acaso (IC < 0,5)? (c) a relação de encosta supera o escore de água (IC de Δ > 0)?

Parte B, retenção na RMPA (ciclo v10, margem com filtro, B=400): AUC por unidade externa de retenção,
  modelo atualizado menos congelado, média das 6 sementes; IC por bootstrap das unidades e contagem de
  unidades com perda. Pesos reconstruídos de `ii_ciclo/lotes.parquet`, como em F01.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts/vulnerabilizados"))
from comum_tv import intervalo  # noqa: E402

SAIDA = RAIZ / "local_runs/fundamentacao-artigo-20261028"
N_BOOT, SEMENTE = 2000, 20261028


def auc_por_cicatriz(score, y, cic, bloco, w_cic=None, w_bloco=None):
    fundo = y == 0
    s0, b0 = score[fundo], bloco[fundo]
    peso0 = np.ones(len(s0)) if w_bloco is None else w_bloco[b0]
    ordem = np.argsort(s0, kind="mergesort")
    s0, peso0 = s0[ordem], peso0[ordem]
    acum = np.concatenate([[0], np.cumsum(peso0)])
    total = acum[-1]
    pos = ~fundo
    sp, cp = score[pos], cic[pos]
    abaixo = acum[np.searchsorted(s0, sp, "left")]
    empate = acum[np.searchsorted(s0, sp, "right")] - abaixo
    u = (abaixo + 0.5 * empate) / total
    medias = pd.Series(u).groupby(cp).mean()
    if w_cic is None:
        return float(medias.mean())
    w = w_cic[medias.index.to_numpy()]
    return float((medias.to_numpy() * w).sum() / w.sum())


def parte_a():
    p = pd.read_parquet(RAIZ / "local_runs/petropolis-aprendizado/v1/resultados/pred_teste_2022.parquet")
    y = p.y.to_numpy()
    cic = pd.factorize(p.id_cicatriz.where(p.y == 1, "fundo"))[0]
    bloco = pd.factorize(p.bloco500)[0]
    colunas = {"idai": "A1_log", "declive": "A0_regra", "agua": "D2_agua_svc03"}
    pontual = {k: auc_por_cicatriz(p[c].to_numpy(float), y, cic, bloco) for k, c in colunas.items()}
    cic_pos = np.unique(cic[y == 1])
    rng = np.random.default_rng(SEMENTE)
    reps = []
    for _ in range(N_BOOT):
        w_c = np.zeros(cic.max() + 1)
        np.add.at(w_c, rng.choice(cic_pos, len(cic_pos)), 1)
        blocos = np.unique(bloco[y == 0])
        w_b = np.zeros(bloco.max() + 1)
        np.add.at(w_b, rng.choice(blocos, len(blocos)), 1)
        reps.append([auc_por_cicatriz(p[c].to_numpy(float), y, cic, bloco, w_c, w_b) for c in colunas.values()])
    reps = np.array(reps)
    r = {"cicatrizes": int(len(cic_pos)), "celulas_fundo": int((y == 0).sum()), "auc_por_cicatriz": pontual,
         "ic95": {k: intervalo(reps[:, i], 0.95) for i, k in enumerate(colunas)}}
    r["idai_acima_do_acaso"] = r["ic95"]["idai"][0] > 0.5
    r["agua_abaixo_do_acaso"] = r["ic95"]["agua"][1] < 0.5
    d = reps[:, 0] - reps[:, 2]
    r["idai_menos_agua"] = {"delta": pontual["idai"] - pontual["agua"], "ic95": intervalo(d, 0.95)}
    d2 = reps[:, 0] - reps[:, 1]
    r["idai_menos_declive"] = {"delta": pontual["idai"] - pontual["declive"], "ic95": intervalo(d2, 0.95)}
    return r


def parte_b():
    from v10_comum import V10, nucleo_limpo, papeis
    from v8_04_campanha import FEAT, externo, z_externa
    import mt01_experimento as e1
    Q = nucleo_limpo()
    mu, sd, med = Q.pad.mu[FEAT], Q.pad.sd[FEAT], Q.pad.med[FEAT]
    tr, x, _ = externo()
    ret = tr.unidade.map(papeis()).eq("retencao").to_numpy()
    R, qR, zR = tr[ret].reset_index(drop=True), Q.escore(x[ret]), z_externa(x[ret], mu, sd, med)
    lotes = pd.read_parquet(V10 / "ii_ciclo/lotes.parquet")
    final = lotes[(lotes.coorte == "rmpa") & (lotes.pool == "todas") & (lotes.consultados == 400)
                  & (lotes.estrategia == "margem_v9_gate")].sort_values("semente")

    def por_unidade(s):
        return {u: e1.auc_media_por_unidade(g, s[g.index.to_numpy()]) for u, g in R.groupby("unidade")}

    base = por_unidade(qR)
    novos = [por_unidade(qR + zR @ np.asarray(json.loads(r.theta)) + r.c) for _, r in final.iterrows()]
    unidades = sorted(base)
    delta = np.array([np.mean([n[u] for n in novos]) - base[u] for u in unidades])
    rng = np.random.default_rng(SEMENTE)
    reps = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(N_BOOT)]
    return {"unidades_de_retencao": len(unidades), "delta_por_unidade": dict(zip(unidades, delta.tolist())),
            "T_B_media_das_unidades": float(delta.mean()), "unidades_com_perda": int((delta < 0).sum()),
            "ic95_bootstrap_de_unidades": intervalo(np.array(reps), 0.95),
            "conferencia_media_de_unidades": float(np.mean([np.mean(list(n.values())) for n in novos])
                                                   - np.mean(list(base.values())) - final.auc_retencao.mean()
                                                   + lotes[(lotes.coorte == "rmpa") & (lotes.pool == "todas")
                                                           & (lotes.estrategia == "congelado")].auc_retencao.mean())}


def main():
    saida = {"petropolis": parte_a()}
    print(json.dumps(saida["petropolis"], default=float), flush=True)
    saida["retencao_rmpa"] = parte_b()
    print(json.dumps(saida["retencao_rmpa"], default=float), flush=True)
    (SAIDA / "f04_encosta_e_retencao.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1, default=float),
                                                        encoding="utf-8")


if __name__ == "__main__":
    main()
