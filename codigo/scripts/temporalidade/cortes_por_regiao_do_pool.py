"""Suporte temporal medido POR REGIAO, quando a regiao e recorte de um pool.

-------------------------------------------------------------------------
POR QUE ESTE ARQUIVO EXISTE
-------------------------------------------------------------------------

Em 17/09/2026 quatro macrorregioes e o Rio Grande do Sul chegaram a
`validado`, servidos por `areal_brasil_observado`. A primeira requisicao real
numa delas -- Cuiaba, no Centro-Oeste -- leu o terreno do tile, passou por
G1-G5 e foi recusada em G6:

    suporte temporal nao demonstrado para 'centro_oeste' com o modelo
    'areal_brasil_observado': a matriz nao cobre este par

O portao estava certo. A matriz tinha evidencia temporal forte para o
CONJUNTO (`dominio_de_ajuste:areal_brasil_observado`, 18 cortes, `supported`)
e nenhuma para a regiao. `MatrizDeSuporte.consultar` recusa herdar, e recusa
com razao: "o Brasil transporta no tempo" nao implica "o Centro-Oeste
transporta no tempo" -- o agregado nacional pode esconder uma regiao que
nao transporta.

Este arquivo MEDE a pergunta no escopo em que ela e feita.

-------------------------------------------------------------------------
O DESENHO
-------------------------------------------------------------------------

Para cada data de evento do pool, em ordem:

    treino  = todo o pool com data ANTERIOR ao corte (o pais inteiro)
    teste   = so as linhas DA REGIAO com data igual ou posterior ao corte

O treino e nacional porque o modelo servido e nacional: medir um modelo
ajustado so na regiao responderia sobre outro modelo, que ninguem serve. O
teste e regional porque e a regiao que recebe o escore.

O estimador e o do modelo servido -- Firth, com a mesma padronizacao do
`svc01`. As variaveis sao as do modelo servido, na mesma ordem.

O IC95 de cada corte vem de bootstrap por GRUPO (tile do GFD), 1000
replicas: celulas do mesmo tile compartilham a mesma cheia, e reamostrar
linha estreitaria o intervalo com pseudo-replicacao -- o erro que o projeto
ja mediu em Recife (IC x1,13 ao passar de linha para celula).

Um corte so e avaliavel quando o teste da regiao tem as duas classes em pelo
menos `MINIMO_GRUPOS` grupos. Corte abaixo disso nao e reprovacao: e corte
que nao existe, e fica de fora sem virar AUC.

-------------------------------------------------------------------------
O QUE ELE NAO FAZ
-------------------------------------------------------------------------

Nao decide suporte. Grava cortes com AUC e IC; quem os julga e
`dimensao_temporal.derivar_suporte`, com a mesma regua de todo escopo do
projeto (minimo de 3 cortes, IC inteiro acima de 0,5 em TODOS).

Uso:
    python scripts/temporalidade/cortes_por_regiao_do_pool.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))

import susc_firth_shim  # noqa: F401,E402
import br04_o_brasil_como_professor as br  # noqa: E402
import regioes as R  # noqa: E402
import svc01_construir_modelos_servidos as svc01  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "temporalidade-cortes-por-regiao"
MODELO = "areal_brasil_observado"
FONTE = "gfd"
N_BOOT = 1000
SEMENTE = 20260917
MINIMO_GRUPOS = 3

SIGLA_DA_MACRORREGIAO = {"norte": "N", "nordeste": "NE", "centro_oeste": "CO",
                         "sudeste": "SE", "sul": "S"}


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """AUC por postos (Mann-Whitney). None quando falta uma das classes."""
    from scipy.stats import rankdata
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _ic_por_grupo(y, p, grupos, rng) -> list | None:
    codigos, inverso = np.unique(grupos, return_inverse=True)
    indices = [np.flatnonzero(inverso == k) for k in range(len(codigos))]
    aucs = []
    for _ in range(N_BOOT):
        escolha = rng.integers(0, len(indices), len(indices))
        idx = np.concatenate([indices[k] for k in escolha])
        a = _auc(y[idx], p[idx])
        if a is not None:
            aucs.append(a)
    if len(aucs) < N_BOOT * 0.9:
        return None
    return [round(float(np.percentile(aucs, 2.5)), 4),
            round(float(np.percentile(aucs, 97.5)), 4)]


def linhas_por_regiao(pool: pd.DataFrame, modelo: dict) -> dict:
    """As linhas do pool que pertencem a cada regiao que o modelo serve.

    Macrorregiao por poligono da malha do IBGE (a mesma do `br04` e de
    `regioes.dentro`); regiao nomeada, como o RS, pela extensao declarada em
    `regioes.py`. Nunca por prefixo de nome.
    """
    com = br.com_regiao(pool)
    saida = {}
    for regiao in modelo.get("regioes_no_ajuste", []):
        if regiao in SIGLA_DA_MACRORREGIAO:
            saida[regiao] = com[com.regiao_br == SIGLA_DA_MACRORREGIAO[regiao]]
        else:
            cx = R.extensao(regiao)
            if not cx:
                saida[regiao] = com.iloc[0:0]
                continue
            m = ((com.lat >= cx["lat_min"]) & (com.lat <= cx["lat_max"])
                 & (com.lon >= cx["lon_min"]) & (com.lon <= cx["lon_max"]))
            saida[regiao] = com[m]
    return saida


def main() -> int:
    alvo = svc01.OUT / f"{MODELO}.json"
    if not alvo.exists():
        print(f"ABORTADO: {alvo} ausente; gere com "
              f"python scripts/servico/svc01_construir_modelos_servidos.py "
              f"--so {MODELO}")
        return 1
    modelo = json.loads(alvo.read_text(encoding="utf-8"))
    feats = modelo["features"]

    d = br.carregar()
    pool = d[d.fonte == FONTE].dropna(subset=feats + ["grupo_cv", "dt"]).copy()
    por_regiao = linhas_por_regiao(pool, modelo)
    datas = sorted(pool.dt.dt.date.dropna().unique())
    print(f"pool {FONTE}: {len(pool):,} linhas, {pool.grupo_cv.nunique()} grupos, "
          f"{len(datas)} datas; regioes: "
          + ", ".join(f"{r}={len(x):,}" for r, x in por_regiao.items()))

    rng = np.random.default_rng(SEMENTE)
    resultado = {r: {"cortes": [], "cortes_nao_avaliaveis": 0}
                 for r in por_regiao}
    t0 = time.time()
    for corte in datas[1:]:
        tr = pool[pool.dt.dt.date < corte]
        if tr.classe.nunique() < 2 or tr.grupo_cv.nunique() < MINIMO_GRUPOS:
            continue
        X = tr[feats].to_numpy(float)
        mu, sd = svc01.padronizar(X)
        m = svc01.ajustar((X - mu) / sd, tr.classe.to_numpy(int))
        for regiao, linhas in por_regiao.items():
            te = linhas[linhas.dt.dt.date >= corte]
            grupos_com_duas = te.groupby("grupo_cv").classe.nunique()
            if (te.classe.nunique() < 2
                    or te.grupo_cv.nunique() < MINIMO_GRUPOS):
                resultado[regiao]["cortes_nao_avaliaveis"] += 1
                continue
            p = m.predict_proba((te[feats].to_numpy(float) - mu) / sd)[:, 1]
            y = te.classe.to_numpy(int)
            auc = _auc(y, p)
            ic = _ic_por_grupo(y, p, te.grupo_cv.to_numpy(), rng)
            resultado[regiao]["cortes"].append({
                "rotulo": f"teste_{corte}",
                "corte": str(corte),
                "auc": round(auc, 4) if auc is not None else None,
                "ic95": ic,
                "n_teste": int(len(te)),
                "grupos_teste": int(te.grupo_cv.nunique()),
                "grupos_com_as_duas_classes": int((grupos_com_duas == 2).sum()),
                "positivos_teste": int(y.sum()),
                "n_treino": int(len(tr)),
                "grupos_treino": int(tr.grupo_cv.nunique()),
            })
        print(f"  corte {corte}: treino {len(tr):,} ({time.time()-t0:.0f}s)",
              flush=True)

    print()
    for regiao, r in resultado.items():
        cs = r["cortes"]
        if not cs:
            print(f"  {regiao:<20} nenhum corte avaliavel")
            continue
        piores = min(cs, key=lambda c: (c["ic95"] or [0])[0])
        print(f"  {regiao:<20} {len(cs):>2} cortes | AUC "
              f"{min(c['auc'] for c in cs):.4f}-{max(c['auc'] for c in cs):.4f} | "
              f"pior IC {piores['ic95']} em {piores['corte']}")

    man = {
        "tipo": "cortes_prospectivos_por_regiao_do_pool",
        "quando_utc": agora_utc(),
        "modelo": MODELO,
        "fonte_do_pool": FONTE,
        "variaveis": feats,
        "estimador": "Firth, padronizacao do svc01 -- o do modelo servido",
        "desenho": ("treino = pool inteiro antes do corte; teste = linhas da "
                    "regiao no corte e depois; IC95 por bootstrap de grupo "
                    f"com {N_BOOT} replicas"),
        "minimo_de_grupos_no_teste": MINIMO_GRUPOS,
        "semente": SEMENTE,
        "regioes": resultado,
        "nao_e": ("nao julga suporte: grava cortes com AUC e IC, e quem decide "
                  "e dimensao_temporal.derivar_suporte com a regua comum. Nao "
                  "herda o veredito do dominio de ajuste para a regiao -- "
                  "mede a regiao"),
    }
    man["hash_resumo"] = sha256_json(man)
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "cortes.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGRAVADO={SAIDA}/cortes.json")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
