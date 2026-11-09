"""V1: o ganho da vizinhanca e de escala geomorfologica ou de sobreposicao espacial?

O N1b mostrou que estatistica simples da janela de 3,84 km supera o embedding.
Aqui a mesma amostra do N1 (GFD de planicie, 69 eventos) e as mesmas janelas
(cache normalizado como no treino do encoder) respondem duas perguntas:

  escala       resumo da janela em lados de 4, 8, 16, 32, 64 e 128 px (120 m a
               3,84 km), um modelo por escala. O rotulo do GFD tem 250 m: ganho
               que so aparece acima de ~8 px pode ser escala do rotulo ou do
               relevo -- o V2 (CEMS) separa as duas.
  sobreposicao dobras por EVENTO (como o N1) contra dobras por COMPONENTE: o
               grafo evento-tile ligado, de modo que nenhum tile e nenhum evento
               aparecem dos dois lados. Se o ganho cai muito no componente, a
               vizinhanca estava reconhecendo o LUGAR visto no treino.

Resumo por escala: media, desvio e percentis 10/50/90 de cada camada (25
numeros), padronizados no treino da dobra. Logistica L2, mesmas linhas.
Desenvolvimento em eventos conhecidos: exploratorio.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
N1 = REPO / "local_runs" / "campanha-integrada-20261003" / "N1"
ESCALAS = (4, 8, 16, 32, 64, 128)
REF = ["hand_m", "twi_dinf"]


def estatisticas_multiescala(jan: np.ndarray, idx: np.ndarray, escalas=ESCALAS, lote: int = 256) -> dict:
    lado = jan.shape[-1]
    out = {s: [] for s in escalas}
    for i in range(0, len(idx), lote):
        x = np.asarray(jan[idx[i:i + lote]], dtype=np.float32)
        for s in escalas:
            a = (lado - s) // 2
            r = x[:, :, a:a + s, a:a + s].reshape(len(x), x.shape[1], -1)
            q = np.percentile(r, [10, 50, 90], axis=2)
            out[s].append(np.concatenate([r.mean(2), r.std(2), q[0], q[1], q[2]], axis=1))
    return {s: np.concatenate(v) for s, v in out.items()}


def dobras_por_evento(d: pd.DataFrame, semente: int, k: int = 5) -> np.ndarray:
    rng = np.random.default_rng(semente)
    u = np.asarray(d.evento_id.unique(), dtype=object)
    rng.shuffle(u)
    return d.evento_id.map({e: i % k for i, e in enumerate(u)}).to_numpy()


def componentes(d: pd.DataFrame, chave_lugar: str) -> np.ndarray:
    """Componente conexo do grafo bipartido evento-lugar (union-find)."""
    pai: dict = {}

    def raiz(x):
        while pai.setdefault(x, x) != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x
    for e, t in d[["evento_id", chave_lugar]].drop_duplicates().itertuples(index=False):
        a, b = raiz(("e", e)), raiz(("t", t))
        if a != b:
            pai[a] = b
    return np.array([str(raiz(("e", e))) for e in d.evento_id])


def dobras_por_componente(d: pd.DataFrame, chave_lugar: str, semente: int, k: int = 5) -> tuple[np.ndarray, int]:
    comp = componentes(d, chave_lugar)
    tam = pd.Series(comp).value_counts()
    rng = np.random.default_rng(semente)
    ordem = list(tam.index)
    rng.shuffle(ordem)
    carga = np.zeros(k)
    atrib = {}
    # componente grande primeiro na dobra mais leve: dobras de tamanho parecido
    for c in sorted(ordem, key=lambda c: -tam[c]):
        j = int(np.argmin(carga))
        atrib[c] = j
        carga[j] += tam[c]
    return np.array([atrib[c] for c in comp]), int(len(tam))


def prever(d: pd.DataFrame, blocos: dict, dobra: np.ndarray) -> dict:
    y = d.classe.to_numpy()
    T = d[REF].to_numpy(float)
    prev = {}
    for nome, B in blocos.items():
        p = np.full(len(d), np.nan)
        for k in np.unique(dobra):
            te, tr = dobra == k, dobra != k
            if len(np.unique(y[tr])) < 2:
                continue
            X = T if B is None else np.column_stack([T, B])
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            m = LogisticRegression(C=1.0, max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
            p[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
        prev[nome] = p
    return prev


def medir(d: pd.DataFrame, prev: dict, ref: str, rng, reps: int = 500) -> dict:
    y = d.classe.to_numpy()
    evs = d.evento_id.to_numpy()
    idx = {e: np.flatnonzero(evs == e) for e in np.unique(evs)}

    def dentro(v):
        a = [roc_auc_score(y[i], v[i]) for i in idx.values() if len(np.unique(y[i])) == 2]
        return round(float(np.median(a)), 4)
    met = {n: {"auc_agrupada": round(float(roc_auc_score(y, v)), 4), "auc_mediana_dentro_do_evento": dentro(v),
               "brier": round(float(brier_score_loss(y, v)), 4)} for n, v in prev.items()}
    difs = {n: [] for n in prev if n != ref}
    for _ in range(reps):
        i = np.concatenate([idx[e] for e in rng.choice(list(idx), len(idx))])
        if len(np.unique(y[i])) == 2:
            for n in difs:
                difs[n].append(roc_auc_score(y[i], prev[n][i]) - roc_auc_score(y[i], prev[ref][i]))
    ic = {n: np.percentile(v, [2.5, 97.5]).round(4).tolist() for n, v in difs.items()}
    return {"metricas": met, "diferenca_contra_" + ref + "_ic95_por_evento": ic, "eventos": len(idx)}


def blocos_de_escala(S: dict) -> dict:
    b = {"terreno": None}
    b.update({f"terreno_mais_janela_{s}px": S[s] for s in S})
    b["terreno_mais_todas_as_escalas"] = np.column_stack([S[s] for s in S])
    return b


def executar(p: dict, saida: Path) -> dict:
    semente = p["sementes"][0]
    a = pd.read_parquet(N1 / "amostra.parquet")
    a = a[a.janela_valida].reset_index(drop=True)
    jan = np.load(N1 / "janelas_f16.npy", mmap_mode="r")
    S = estatisticas_multiescala(jan, a.indice_janela.to_numpy())
    np.savez_compressed(saida / "estatisticas_multiescala.npz", **{str(k): v for k, v in S.items()})
    rng = np.random.default_rng(semente)
    blocos = blocos_de_escala(S)
    por_evento = medir(a, prever(a, blocos, dobras_por_evento(a, semente)), "terreno", rng)
    dob_c, n_comp = dobras_por_componente(a, "tile_dir", semente)
    por_componente = medir(a, prever(a, blocos, dob_c), "terreno", rng)
    chave = "diferenca_contra_terreno_ic95_por_evento"
    ganho = {n: {"por_evento": por_evento[chave][n], "por_componente": por_componente[chave][n]}
             for n in por_evento[chave]}
    queda = {n: round(por_evento["metricas"][n]["auc_agrupada"] - por_componente["metricas"][n]["auc_agrupada"], 4)
             for n in por_evento["metricas"]}
    escala_minima = next((s for s in ESCALAS if ganho[f"terreno_mais_janela_{s}px"]["por_componente"][0] > 0), None)
    return {"tipo": "vizinhanca_multiescala", "pergunta": p["pergunta"], "linhas": int(len(a)),
            "eventos": int(a.evento_id.nunique()), "tiles": int(a.tile_dir.nunique()),
            "componentes_evento_tile": n_comp, "escalas_px": list(ESCALAS), "pixel_m": 30,
            "dobras_por_evento": por_evento, "dobras_por_componente": por_componente,
            "ganho_sobre_terreno": ganho, "queda_de_auc_evento_para_componente": queda,
            "menor_escala_com_ganho_no_componente_px": escala_minima,
            "veredito": ("ganho_sobrevive_a_separacao_espacial" if escala_minima else
                         "ganho_nao_sobrevive_a_separacao_espacial"),
            "interpretacao_permitida": ("desenvolvimento em eventos conhecidos; a separacao por componente remove "
                                        "lugar repetido, nao remove o desenho de amostragem do GFD (positivo no "
                                        "interior da mancha, negativo na faixa de 2 km) -- isso o V2 testa")}
