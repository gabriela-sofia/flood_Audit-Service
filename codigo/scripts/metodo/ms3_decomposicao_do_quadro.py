"""MS3: o que, DENTRO do quadro do negativo, muda a transferencia CEMS -> GFD? Um fator por condicao.

-------------------------------------------------------------------------
DE ONDE VEM
-------------------------------------------------------------------------

MS2 (pre-registrado) mostrou que treinar o CEMS no quadro de 2 km do GFD leva a
logistica de terreno de 0,504 a 0,770 de AUC mediana dentro do evento GFD. O
quadro de 2 km muda varias coisas ao mesmo tempo: tira negativos longe da
mancha (extensao), tira a margem de 250 m entre classes (incerteza de borda),
reduz linhas (quantidade) e muda a distribuicao de HAND dos negativos. MS2
sustenta o quadro como explicacao NO experimento; nao separa a causa.

-------------------------------------------------------------------------
DESENHO (pre_registro.json antes de qualquer ajuste)
-------------------------------------------------------------------------

  A  aoi              CEMS como esta
  B  aoi_tamanho      A subamostrado ao numero de positivos e negativos de E
                      (semente fixa; 20 sementes descritivas)
  C  aoi_sem_margem   A sem pontos a <= 250 m da outra classe (mantem o longe)
  D  aoi_sem_longe    A sem negativos a > 2 km de um positivo (mantem a margem)
  E  quadro_2km       C e D juntos (o quadro do MS2)
  F  aoi_reponderado  A com pesos nos negativos que igualam os decis de HAND aos
                      dos negativos de E (logistica ponderada); extensao igual a A

  modelo      logistica de hand_m, twi_dinf, slope_deg (a base do MS1/MS2)
  teste       GFD sem tile de 1 grau ocupado pelo CEMS, o mesmo em todas
  comparacoes cada condicao x A, e E x D, E x C; generalizacao_por_reservas.veredito
  primario    nenhum: decomposicao diagnostica, todas lidas pela mesma regra
  nao testavel observabilidade (mapa de analista no CEMS x visada MODIS no GFD):
              nenhuma condicao a separa com os dados atuais

nao_e: nao reabilita "o CEMS transfere para o GFD"; desenvolvimento em eventos
conhecidos. Uma condicao com ganho e explicacao no experimento, nao causa unica.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

import generalizacao_por_reservas as gen  # noqa: E402
import ms1_posicao_topografica as ms1  # noqa: E402
import ms2_quadro_do_negativo as ms2  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "ms3-decomposicao-do-quadro"
SEMENTE = 20261008
SEMENTES_DESCRITIVAS = 20
CONDICOES = ("A_aoi", "B_aoi_tamanho", "C_aoi_sem_margem", "D_aoi_sem_longe", "E_quadro_2km", "F_aoi_reponderado")
COMPARACOES = (("B_vs_A", "B_aoi_tamanho", "A_aoi", "quantidade"),
               ("C_vs_A", "C_aoi_sem_margem", "A_aoi", "margem de borda"),
               ("D_vs_A", "D_aoi_sem_longe", "A_aoi", "extensao"),
               ("F_vs_A", "F_aoi_reponderado", "A_aoi", "distribuicao de HAND dos negativos"),
               ("E_vs_A", "E_quadro_2km", "A_aoi", "replica do MS2"),
               ("E_vs_D", "E_quadro_2km", "D_aoi_sem_longe", "margem depois da extensao"),
               ("E_vs_C", "E_quadro_2km", "C_aoi_sem_margem", "extensao depois da margem"))
VISTO_ANTES = ("MS2 lido (07/10): E x A ganho +0,25, 69/69; diagnostico entre fontes lido: direcao marginal igual nas "
               "duas fontes, contraste de HAND do GFD 0,03 desvio do CEMS")


def pre_registro() -> dict:
    return json.loads(json.dumps({
        "condicoes": CONDICOES, "comparacoes": [list(c) for c in COMPARACOES], "primario": None,
        "modelo": list(ms1.BASE), "estimador": "logistica IRLS com intercepto, padronizada no treino (ponderada so em F)",
        "populacao": "carregar(TERRENO) + posicao v1 finita (a do MS2)",
        "teste": "GFD sem tile de 1 grau do CEMS", "margem_m": ms2.MARGEM_M, "faixa_m": ms2.FAIXA_M,
        "reponderacao": "decis de hand_m dos negativos de E; peso = fracao em E / fracao em A por decil",
        "regra": "generalizacao_por_reservas.veredito", "nao_testavel": "observabilidade",
        "visto_antes": VISTO_ANTES, "semente": SEMENTE, "sementes_descritivas": SEMENTES_DESCRITIVAS}))


def gravar_pre_registro(destino: Path = SAIDA / "pre_registro.json") -> str:
    corpo = pre_registro()
    h = sha256_json(corpo)
    if destino.exists():
        antigo = json.loads(destino.read_text(encoding="utf-8"))
        if antigo["hash"] != h:
            raise RuntimeError(f"pre-registro mudou ({antigo['hash']} -> {h}): crie MS4")
        return h
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"quando_utc": agora_utc(), "hash": h, "criterios": corpo},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
    return h


def distancias(c: pd.DataFrame) -> pd.DataFrame:
    """Por AOI: distancia de cada ponto ao ponto mais proximo da outra classe (m)."""
    from sklearn.neighbors import BallTree
    c = c.reset_index(drop=True)
    d_outra = np.full(len(c), np.nan)
    for _, idx in c.groupby("grupo_validacao").indices.items():
        s = c.iloc[idx]
        pos, neg = s[s.classe == 1], s[s.classe == 0]
        if pos.empty or neg.empty:
            continue
        rad = lambda x: np.radians(x[["lat", "lon"]].to_numpy(float))  # noqa: E731
        d_outra[idx[(s.classe == 0).to_numpy()]] = BallTree(rad(pos), metric="haversine").query(rad(neg), k=1)[0][:, 0] * 6_371_000
        d_outra[idx[(s.classe == 1).to_numpy()]] = BallTree(rad(neg), metric="haversine").query(rad(pos), k=1)[0][:, 0] * 6_371_000
    return c.assign(d_outra_m=d_outra)


def condicoes(c: pd.DataFrame, rng) -> dict:
    c = distancias(c)
    tem = c.d_outra_m.notna()
    margem = c.d_outra_m > ms2.MARGEM_M
    perto = (c.classe == 1) | (c.d_outra_m <= ms2.FAIXA_M)
    out = {"A_aoi": c[tem], "C_aoi_sem_margem": c[tem & margem], "D_aoi_sem_longe": c[tem & perto],
           "E_quadro_2km": c[tem & margem & perto]}
    e = out["E_quadro_2km"]
    out["B_aoi_tamanho"] = subamostra(out["A_aoi"], e, rng)
    out["F_aoi_reponderado"] = out["A_aoi"].assign(peso=pesos_hand(out["A_aoi"], e))
    return out


def subamostra(a: pd.DataFrame, alvo: pd.DataFrame, rng) -> pd.DataFrame:
    partes = []
    for k in (0, 1):
        x = a[a.classe == k]
        n = min(len(x), int((alvo.classe == k).sum()))
        partes.append(x.iloc[rng.choice(len(x), n, replace=False)])
    return pd.concat(partes)


def pesos_hand(a: pd.DataFrame, e: pd.DataFrame) -> np.ndarray:
    cortes = np.unique(np.quantile(e.loc[e.classe == 0, "hand_m"], np.linspace(0, 1, 11)))
    if len(cortes) < 2:
        cortes = np.array([-np.inf, np.inf])          # decis colapsados: um intervalo so, peso 1
    cortes[0], cortes[-1] = -np.inf, np.inf
    fa = pd.cut(a.loc[a.classe == 0, "hand_m"], cortes).value_counts(normalize=True)
    fe = pd.cut(e.loc[e.classe == 0, "hand_m"], cortes).value_counts(normalize=True)
    razao = (fe / fa).replace([np.inf], np.nan).fillna(0.0)
    w = np.ones(len(a))
    neg = (a.classe == 0).to_numpy()
    w[neg] = pd.cut(a.loc[a.classe == 0, "hand_m"], cortes).map(razao).astype(float).fillna(0.0).to_numpy()
    # o total de peso dos negativos fica igual ao numero deles: so a forma muda
    w[neg] *= neg.sum() / w[neg].sum()
    return w


def logistica_ponderada(X: np.ndarray, y: np.ndarray, w: np.ndarray, iteracoes: int = 30) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(A.shape[1])
    for _ in range(iteracoes):
        p = expit(A @ b)
        W = w * p * (1 - p) + 1e-9
        H = A.T @ (A * W[:, None]) + 1e-6 * np.eye(A.shape[1])
        passo = np.linalg.solve(H, A.T @ (w * (y - p)))
        b += passo
        if np.abs(passo).max() < 1e-8:
            break
    return b


def ajustar(tr: pd.DataFrame) -> callable:
    if "peso" not in tr:
        return ms1.ajustar(tr, ms1.BASE)
    v = list(ms1.BASE)
    X = tr[v].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    b = logistica_ponderada((X - mu) / sd, tr.classe.to_numpy(float), tr.peso.to_numpy(float))
    f = lambda te: expit(b[0] + ((te[v].to_numpy(float) - mu) / sd) @ b[1:])  # noqa: E731
    f.coef = dict(zip(v, np.round(b[1:], 4).tolist()))
    return f


def executar() -> dict:
    import posicao_topografica as pt
    import verdade_observacional as vo
    h = gravar_pre_registro()
    d, versao, _ = gen.carregar(gen.TERRENO)
    d = pt.juntar(d)
    d = d[np.isfinite(d[list(ms1.POSICAO)].to_numpy(float)).all(axis=1)]
    c = d[d.fonte == "cems"].reset_index(drop=True)
    g = d[d.fonte == "gfd"]
    teste = g[~vo.chave_espacial(g).isin(set(vo.chave_espacial(c)))].reset_index(drop=True)
    rng = np.random.default_rng(SEMENTE)
    cond = condicoes(c, rng)
    pred, coef, med = {}, {}, {}
    for k, tr in cond.items():
        fn = ajustar(tr)
        pred[k] = fn(teste)
        coef[k] = fn.coef
    med = ms1.medir(teste, pred)
    comps = {}
    for cid, a, b, q in COMPARACOES:
        r = gen.comparar(teste, pred[a], pred[b], rng)
        comps[cid] = {"pergunta": q, "a": a, "b": b, **r, "veredito": gen.veredito(r)}
    # B descritivo: 20 subamostras, a mediana dentro do evento de cada uma
    b_desc = []
    for s in range(SEMENTES_DESCRITIVAS):
        sub = subamostra(cond["A_aoi"], cond["E_quadro_2km"], np.random.default_rng(SEMENTE + 1 + s))
        pe = gen.auc_por_evento(teste, ms1.ajustar(sub, ms1.BASE)(teste))
        b_desc.append(round(float(np.median(list(pe.values()))), 4))
    r = {"tipo": "ms3_decomposicao_do_quadro", "dataset": versao, "pre_registro": h, "regime": "exploratorio",
         "linhas": {k: {"linhas": int(len(x)), "positivos": int(x.classe.sum()),
                        "hand_mediano_neg": round(float(x.loc[x.classe == 0, "hand_m"].median()), 2)}
                    for k, x in cond.items()},
         "teste_gfd": {"linhas": int(len(teste)), "eventos": int(teste.evento_id.nunique())},
         "medidas_no_gfd": med, "coeficientes": coef, "comparacoes": comps,
         "B_descritivo_20_sementes": {"medianas_dentro": b_desc,
                                      "p10_p50_p90": [round(float(x), 4) for x in np.percentile(b_desc, [10, 50, 90])]},
         "nao_testavel": "observabilidade (mapa de analista x visada MODIS)",
         "nao_e": "nao reabilita transferencia geral; explicacao no experimento, nao causa unica"}
    r["hash_resumo"] = sha256_json(json.loads(json.dumps(r, default=float)))
    r["quando_utc"] = agora_utc()
    (SAIDA / "resultado.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return r


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-registrar", action="store_true")
    a = ap.parse_args()
    if a.pre_registrar:
        print(gravar_pre_registro())
        return 0
    inicio = time.time()
    r = executar()
    print(json.dumps({k: v["veredito"] for k, v in r["comparacoes"].items()}), round(time.time() - inicio), "s")
    print(f"GRAVADO={SAIDA / 'resultado.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
