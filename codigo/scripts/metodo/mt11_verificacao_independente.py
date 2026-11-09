"""MT11: verificação independente do resultado emendado, sem as funções de medida do avaliador.

Recalcula, a partir de `escores_emendado.parquet` e de `completude.csv`:
  AUC por evento por contagem de pares (`verificar_fechamento_independente.auc_pares`)
  média por unidade por outro caminho (dicionários, não groupby/unstack)
  IC por bootstrap reimplementado com a mesma semente e o mesmo número de réplicas
e compara com `resultado_emendado.json`. Grava `verificacao_independente.json`. Não altera nada.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/metodo"))
from verificar_fechamento_independente import auc_pares  # noqa: E402

D = REPO / "local_runs/mt11-transdutivo"
CFG = json.loads((D / "protocolo.json").read_text(encoding="utf-8"))["config"]


def contraste(v: list[float]) -> dict:
    v = np.array(v, float)
    rng = np.random.default_rng(CFG["semente_bootstrap"])
    b = rng.choice(v, (CFG["bootstrap"], len(v))).mean(1)
    q = (1 - CFG["nivel"]) / 2
    return {"n": len(v), "media": float(v.mean()), "ic": [float(x) for x in np.quantile(b, [q, 1 - q])], "a_favor": int((v > 0).sum())}


def main():
    s = pd.read_parquet(D / "escores_emendado.parquet")
    t = pd.read_csv(D / "completude.csv").set_index("evento_id")
    r = json.loads((D / "resultado_emendado.json").read_text(encoding="utf-8"))
    obs = ((s.arvore <= CFG["arvore_max"]) & (s.construido_wc <= CFG["construido_max"]) & (s.slope_deg <= CFG["declive_max_graus"])
           & (s.jrc_occ_s1 < CFG["jrc_permanente"]) & (s.n_pre >= CFG["min_datas_pre"]) & s.F.notna()).to_numpy()
    mapas = {"congelada_original": t.unidade_centro.to_dict(),
             "emendada_primaria": {e: (None if t.loc[e, "ligado_primario"] else t.loc[e, "componente_primario"]) for e in t.index},
             "sensibilidade_exutorio": {e: (None if t.loc[e, "ligado_exutorio"] else t.loc[e, "componente_exutorio"]) for e in t.index}}
    out, dif = {}, 0.
    for nome, mapa in mapas.items():
        res = {}
        for hip, (a, b, so_obs) in {"HQ1": ("Q_conexao", "A0_regra_hand", False), "HQ2": ("Q_conexao", "Q_base", False),
                                    "HF": ("F_transf", "Q_conexao", True)}.items():
            por_unidade: dict = {}
            for ev in sorted(set(s.evento_id)):
                u = mapa.get(ev)
                if u is None or (isinstance(u, float) and np.isnan(u)):
                    continue
                m = (s.evento_id == ev).to_numpy() & (obs if so_obs else True)
                y = s.classe.to_numpy()[m]
                if m.sum() < CFG["minimo_pontos_evento"] or len(set(y)) < 2:
                    continue
                d = auc_pares(y, s[a].to_numpy()[m]) - auc_pares(y, s[b].to_numpy()[m])
                por_unidade.setdefault(u, []).append(d)
            v = [float(np.mean(x)) for _, x in sorted(por_unidade.items())]
            c = contraste(v)
            ref = r["leituras"][nome][hip]
            d_m = abs(c["media"] - ref["media"])
            d_ic = max(abs(x - y) for x, y in zip(c["ic"], ref["ic"])) if "ic" in ref else 0.
            dif = max(dif, d_m)
            res[hip] = {**c, "dif_media": d_m, "dif_ic": d_ic, "n_confere": c["n"] == ref["n"], "a_favor_confere": c["a_favor"] == ref["a_favor"]}
        out[nome] = res
    v = {"tipo": "mt11_verificacao_independente", "max_dif_media": dif, "leituras": out,
         "nota": "IC pode diferir se a ordem das unidades na reamostragem diferir; média, n e contagem a favor devem conferir"}
    (D / "verificacao_independente.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({n: {h: {k: x[k] for k in ("n_confere", "a_favor_confere", "dif_media", "dif_ic")} for h, x in y.items()}
                      for n, y in out.items()}, indent=1))


if __name__ == "__main__":
    main()
