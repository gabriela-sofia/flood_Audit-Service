"""F08: contas descritivas citadas no artigo, feitas sobre artefatos existentes. Nenhum modelo é ajustado.

  ruas        independência do escore em relação ao comprimento da rua dentro do bairro e contraste pareado por
              comprimento (Recife, teste). O relato acompanha a extensão da rua; a pergunta é se o escore herda isso.
  comunidades quantas FCU passaram a ter observação física, a partir da tabela por comunidade da v10.
  fontes      correspondência entre relato e Sentinel-2 por bairro em Recife/2022 (Tabela de fontes da v9).
  chip        HAND e radar nos pontos do CEMS no RS: o terreno separa mais que o retorno de radar naquele lugar.

Uso:
    python scripts/fundamentacao_artigo/f08_ruas_e_comunidades.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "local_runs/fundamentacao-artigo-20261106"
RECIFE = RAIZ / "local_runs/recife-aprendizado/v1"
V10 = RAIZ / "local_runs/transferencia-vulnerabilizados/v10_ciclo_contextual"
SEMENTE, N_BOOT = 20261106, 2000


def posicao(alvo, comparadores):
    return (comparadores < alvo).mean() + 0.5 * (comparadores == alvo).mean()


def auc_no_bairro(t, coluna):
    u = []
    for _, b in t.groupby("CBAIRRCODI"):
        comparadores = b.loc[b.y == 0, coluna].to_numpy()
        u += [posicao(v, comparadores) for v in b.loc[b.y == 1, coluna].to_numpy()]
    return float(np.mean(u))


def pareado_por_bairro(t, fator):
    """Soma das posições e contagem por bairro. Cada rua relatada é comparada só com ruas do mesmo bairro de
    comprimento até `fator` vezes maior ou menor, e entra se houver ao menos 3 comparadoras."""
    linhas = []
    for bairro, b in t.groupby("CBAIRRCODI"):
        neg = b[b.y == 0]
        soma_q, soma_h, n = 0.0, 0.0, 0
        for _, r in b[b.y == 1].iterrows():
            viz = neg[(neg.comp >= r.comp / fator) & (neg.comp <= r.comp * fator)]
            if len(viz) >= 3:
                soma_q += posicao(r["A1_Q"], viz["A1_Q"].to_numpy())
                soma_h += posicao(r["A0_regra"], viz["A0_regra"].to_numpy())
                n += 1
        linhas.append((bairro, soma_q, soma_h, n))
    return np.array([x[1:] for x in linhas], dtype=float)


def ruas():
    p = pd.read_parquet(RECIFE / "resultados/pred_ruas.parquet", columns=["id_obj", "papel", "y", "CBAIRRCODI", "A0_regra", "A1_Q"])
    a = pd.read_parquet(RECIFE / "atributos/atributos_objetos.parquet", columns=["id_obj", "comprimento_m"])
    t = p.merge(a, on="id_obj", how="left").rename(columns={"comprimento_m": "comp"})
    t = t[t.papel == "teste"].copy()
    rho = lambda col: float(t.groupby("CBAIRRCODI")[[col, "comp"]].apply(lambda b: spearmanr(b[col], b["comp"]).statistic).mean())
    res = {"ruas": int(len(t)), "relatadas": int(t.y.sum()), "bairros": int(t.CBAIRRCODI.nunique()),
           "auc_Q": auc_no_bairro(t, "A1_Q"), "auc_hand": auc_no_bairro(t, "A0_regra"), "auc_comprimento": auc_no_bairro(t, "comp"),
           "rho_medio_no_bairro_Q_comprimento": rho("A1_Q"), "rho_medio_no_bairro_hand_comprimento": rho("A0_regra")}
    rng = np.random.default_rng(SEMENTE)
    for fator in (1.5, 2.0):
        m = pareado_por_bairro(t, fator)
        pesos = rng.multinomial(len(m), np.ones(len(m)) / len(m), size=N_BOOT)
        total = pesos @ m
        deltas = (total[:, 0] - total[:, 1]) / np.where(total[:, 2] > 0, total[:, 2], np.nan)
        q, h, n = m[:, 0].sum() / m[:, 2].sum(), m[:, 1].sum() / m[:, 2].sum(), int(m[:, 2].sum())
        res[f"pareado_fator_{fator}"] = {"auc_Q": float(q), "auc_hand": float(h), "delta": float(q - h), "positivos_usados": n,
                                         "ic95_delta": [float(x) for x in np.nanpercentile(deltas, [2.5, 97.5])]}
    return res


def comunidades():
    c = pd.read_csv(V10 / "v_social/por_comunidade_fcu.csv")
    c["com_positivo"] = c.positivos >= 1
    c["duas_classes"] = (c.positivos >= 1) & (c.negativos >= 1)
    c["cinco_cinco"] = (c.positivos >= 5) & (c.negativos >= 5)
    por = c.groupby("coorte")[["com_positivo", "duas_classes", "cinco_cinco"]].sum().astype(int)
    por["fcu"] = c.groupby("coorte").CD_FCU.nunique()
    return {"fcu_total": int(c.CD_FCU.nunique()), "com_positivo": int(c[c.com_positivo].CD_FCU.nunique()),
            "duas_classes": int(c[c.duas_classes].CD_FCU.nunique()), "cinco_cinco": int(c[c.cinco_cinco].CD_FCU.nunique()),
            "por_coorte": por.to_dict("index")}


def fontes():
    tabela = {"relato_com_s2": 19, "relato_sem_s2": 16, "sem_relato_com_s2": 36, "sem_relato_sem_s2": 23}
    razao, p = fisher_exact([[19, 16], [36, 23]])
    return {**tabela, "bairros": 94, "uniao_com_registro": 19 + 16 + 36, "so_s2": 55, "razao_de_chances": float(razao), "p_fisher": float(p)}


def chip():
    d = pd.read_csv(RAIZ / "local_runs/cems-emsr720/pontos_emsr720.csv")
    colunas = list(d.columns)
    res = {"colunas": colunas, "classes": d.classe_obs.value_counts().to_dict()}
    for nome in [c for c in colunas if c.lower() in ("hand_m", "hand", "hand_dinf", "vh_db", "vv_db")]:
        res[nome] = d.groupby("classe_obs")[nome].median().to_dict()
    return res


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    saida = {"tipo": "fundamentacao_artigo:f08/ruas_e_comunidades",
             "nao_e": "novo treino nem confirmacao; contagens e reanalise pos-hoc de artefatos existentes",
             "gerado_por": "scripts/fundamentacao_artigo/f08_ruas_e_comunidades.py",
             "comunidades": comunidades(), "fontes": fontes(), "chip": chip(), "ruas": ruas()}
    (SAIDA / "f08_ruas_e_comunidades.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(saida, ensure_ascii=False, indent=1)[:3500])


if __name__ == "__main__":
    main()
