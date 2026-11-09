"""MS5: elegibilidade, novidade dos atributos e incerteza preditiva -- qual ordena o erro, separadamente?

-------------------------------------------------------------------------
POR QUE SEPARAR
-------------------------------------------------------------------------

A recusa por dominio (v1 e v2) nao ordenou o erro melhor que o acaso no GFD.
"Dominio" misturava tres coisas: (1) elegibilidade cientifica -- a pergunta
vale para este lugar? (contrato: mecanismo, relevo, instrumento); (2)
novidade dos atributos -- o ponto esta longe do que o treino viu?; (3)
incerteza preditiva -- o modelo esta indeciso ou instavel aqui? So (2) e (3)
podem ser comparadas contra o erro; (1) e regra de escopo e fica como
cobertura. Nenhum sinal aqui e confianca calibrada.

-------------------------------------------------------------------------
DESENHO (pre_registro.json antes de qualquer leitura)
-------------------------------------------------------------------------

  linhas      predicoes fora da dobra do MS4 (amostra N1, 5 dobras por evento)
  erro        Brier de cada ponto na predicao A2 (base + pos_min_2km)
  sinais      novidade_mahalanobis  distancia de Mahalanobis ao treino da dobra
                                     (vetor padronizado de A2)
              novidade_knn          distancia media aos 10 vizinhos do treino
                                     (ate 20.000 linhas de treino, semente fixa)
              incerteza_margem      1 - |2p - 1| da predicao A2
              incerteza_bootstrap   desvio de 20 reajustes de A2 com eventos de
                                     treino reamostrados
  referencia  ordem aleatoria: AURC esperada = risco medio do evento
  leitura     por evento: delta = AURC(sinal) - risco medio; "ordena" se a
              mediana < 0, Wilcoxon p < 0,05 e >= 8 eventos; senao
              "nao_ordena" (mediana >= 0) ou "indecidivel"
  descritivo  fracao do ganho do oraculo; risco com cobertura 80/90/95%
  escopo      cobertura da elegibilidade do contrato no ds06 (sem AURC)

nao_e: nao remove recusas de seguranca do servico, nao calibra confianca,
nao muda nenhum modelo servido.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

import ms4_bracos_de_desenvolvimento as ms4  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "ms5-recusa-decomposta"
SEMENTE = 20261008
VETOR = ms4.BASE + ("pos_min_2km",)
SINAIS = ("novidade_mahalanobis", "novidade_knn", "incerteza_margem", "incerteza_bootstrap")
REPETICOES_BOOTSTRAP = 20
VIZINHOS = 10
MAX_TREINO_KNN = 20_000
COBERTURAS = (0.8, 0.9, 0.95)


def pre_registro() -> dict:
    return json.loads(json.dumps({
        "linhas": "ms4 predicoes_fora_da_dobra", "erro": "Brier da predicao A2_base_pos2km",
        "sinais": SINAIS, "vetor": VETOR, "repeticoes_bootstrap": REPETICOES_BOOTSTRAP, "vizinhos": VIZINHOS,
        "referencia": "AURC esperada da ordem aleatoria = risco medio do evento",
        "regra": "ordena: mediana(delta) < 0, Wilcoxon p < 0,05, >= 8 eventos; nao_ordena: mediana >= 0; resto indecidivel",
        "descritivo": ["fracao do ganho do oraculo", "risco nas coberturas 0,8/0,9/0,95"],
        "escopo": "cobertura da elegibilidade do contrato (sem AURC)", "semente": SEMENTE}))


def gravar_pre_registro(destino: Path = SAIDA / "pre_registro.json") -> str:
    corpo = pre_registro()
    h = sha256_json(corpo)
    if destino.exists():
        antigo = json.loads(destino.read_text(encoding="utf-8"))
        if antigo["hash"] != h:
            raise RuntimeError(f"pre-registro mudou ({antigo['hash']} -> {h}): crie MS6")
        return h
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"quando_utc": agora_utc(), "hash": h, "criterios": corpo},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
    return h


def aurc(perda: np.ndarray, incerteza: np.ndarray) -> float:
    """Area sob risco x cobertura: aceita primeiro o mais certo (menor incerteza)."""
    ordem = np.argsort(incerteza, kind="stable")
    risco = np.cumsum(perda[ordem]) / np.arange(1, len(perda) + 1)
    return float(risco.mean())


def risco_na_cobertura(perda: np.ndarray, incerteza: np.ndarray, c: float) -> float:
    ordem = np.argsort(incerteza, kind="stable")
    n = max(1, int(round(c * len(perda))))
    return float(perda[ordem][:n].mean())


def sinais(d: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(SEMENTE)
    X = d[list(VETOR)].to_numpy(float)
    y = d.classe.to_numpy(int)
    out = {s: np.full(len(d), np.nan) for s in SINAIS}
    p = d["A2_base_pos2km"].to_numpy(float)
    out["incerteza_margem"] = 1 - np.abs(2 * p - 1)
    for f in sorted(d.dobra.unique()):
        tr, te = (d.dobra != f).to_numpy(), (d.dobra == f).to_numpy()
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = (X - mu) / sd
        cov = np.cov(Z[tr], rowvar=False) + 1e-6 * np.eye(Z.shape[1])
        inv = np.linalg.inv(cov)
        dz = Z[te] - Z[tr].mean(0)
        out["novidade_mahalanobis"][te] = np.sqrt(np.einsum("ij,jk,ik->i", dz, inv, dz))
        idx = np.flatnonzero(tr)
        if len(idx) > MAX_TREINO_KNN:
            idx = rng.choice(idx, MAX_TREINO_KNN, replace=False)
        nn = NearestNeighbors(n_neighbors=VIZINHOS).fit(Z[idx])
        out["novidade_knn"][te] = nn.kneighbors(Z[te])[0].mean(1)
        evs = d.evento_id.to_numpy()[tr]
        unicos = np.unique(evs)
        reps = []
        for _ in range(REPETICOES_BOOTSTRAP):
            esc = rng.choice(unicos, len(unicos), replace=True)
            linhas = np.concatenate([np.flatnonzero(tr)[evs == e] for e in esc])
            if len(np.unique(y[linhas])) < 2:
                continue
            m = LogisticRegression(C=1.0, max_iter=3000).fit(Z[linhas], y[linhas])
            reps.append(m.predict_proba(Z[te])[:, 1])
        out["incerteza_bootstrap"][te] = np.std(reps, axis=0)
    return d.assign(**out)


def ler(d: pd.DataFrame) -> dict:
    from scipy.stats import wilcoxon
    perda_total = (d["A2_base_pos2km"] - d.classe) ** 2
    d = d.assign(perda=perda_total)
    res = {}
    for s in SINAIS:
        deltas, frac, cobs = [], [], {c: [] for c in COBERTURAS}
        for _, g in d.groupby("evento_id"):
            if len(g) < 20:
                continue
            perda, inc = g.perda.to_numpy(), g[s].to_numpy()
            ok = np.isfinite(inc)
            perda, inc = perda[ok], inc[ok]
            media = perda.mean()
            a, oraculo = aurc(perda, inc), aurc(perda, perda)
            deltas.append(a - media)
            frac.append((media - a) / (media - oraculo) if media > oraculo else np.nan)
            for c in COBERTURAS:
                cobs[c].append(risco_na_cobertura(perda, inc, c) - media)
        deltas = np.array(deltas)
        p = float(wilcoxon(deltas).pvalue) if len(deltas) >= 8 and np.any(deltas != 0) else None
        med = float(np.median(deltas)) if len(deltas) else None
        if med is None:
            estado = "not_testable"
        elif med >= 0:
            estado = "nao_ordena"
        elif p is not None and p < 0.05 and len(deltas) >= 8:
            estado = "ordena"
        else:
            estado = "indecidivel"
        res[s] = {"estado": estado, "eventos": int(len(deltas)), "delta_mediano": round(med, 5) if med is not None else None,
                  "eventos_a_favor": int((deltas < 0).sum()), "p_wilcoxon": p,
                  "fracao_do_ganho_do_oraculo_mediana": round(float(np.nanmedian(frac)), 3) if len(frac) else None,
                  "risco_menos_medio_por_cobertura": {str(c): round(float(np.median(v)), 5) for c, v in cobs.items()}}
    return res


def escopo() -> dict:
    import contrato_do_fenomeno as cf
    from ds06_dataset_de_treino import versao_atual
    t = pd.read_parquet(versao_atual() / "pontos.parquet", columns=["fonte", "mecanismo", "classe_relevo"])
    elegivel = (t.mecanismo == "FLUVIAL_ENXURRADA") & (t.classe_relevo == "PLANO_OU_ONDULADO")
    resolvido = pd.Series([cf.relevo_resolvido(f, r) for f, r in zip(t.fonte, t.classe_relevo)], index=t.index)
    return {"linhas_ds06": int(len(t)),
            "elegivel_para_planicie_fluvial_por_fonte": t.assign(e=elegivel).groupby("fonte").e.mean().round(3).to_dict(),
            "relevo_resolvido_pelo_instrumento_por_fonte": t.assign(r=resolvido).groupby("fonte").r.mean().round(3).to_dict(),
            "leitura": "elegibilidade e regra de escopo: diz onde a pergunta vale, nao onde o modelo erra"}


def executar() -> dict:
    h = gravar_pre_registro()
    d = pd.read_parquet(ms4.SAIDA / "predicoes_fora_da_dobra.parquet")
    d = sinais(d)
    r = {"tipo": "ms5_recusa_decomposta", "pre_registro": h, "regime": "exploratorio",
         "linhas": int(len(d)), "eventos": int(d.evento_id.nunique()),
         "sinais": ler(d), "escopo": escopo(),
         "nao_e": "nao remove recusa de seguranca, nao calibra confianca, nao muda modelo servido"}
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
    r = executar()
    print(json.dumps({k: v["estado"] for k, v in r["sinais"].items()}, ensure_ascii=False))
    print(f"GRAVADO={SAIDA / 'resultado.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
