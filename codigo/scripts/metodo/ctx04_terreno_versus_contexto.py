"""O contexto do evento acrescenta ao terreno? Mesmo corte, mesmos pontos, dois modelos.

-------------------------------------------------------------------------
DESENHO, FIXADO ANTES DO AJUSTE
-------------------------------------------------------------------------

  M0  terreno:            hand_m, twi_dinf
  M1  terreno+contexto:   M0 + pico_percentil + log1p(chuva 7 d ate o pico)
                          + hand x pico_percentil + hand x chuva

Contexto por par evento x estacao (ctx03): so pares com bacia conferida,
pico COMPLETO e chuva calculada. Pico incompleto nao ancora nada.

Corte: deixa-um-EVENTO-fora. Todos os pares do evento reservado saem juntos
-- o mesmo evento em duas estacoes daria ao treino o contexto que o teste
vai usar. Unidade independente: o evento; o IC das diferencas reamostra
eventos.

Peso: cada par pesa igual no ajuste e nas metricas. Sem isso o par com
2.500 pontos decide pelo de 80.

So GFD na analise principal: um protocolo de amostragem (faixa e passo do
gfd01). A fracao de positivos de um par depende do protocolo; misturar o
CEMS, amostrado de outro jeito, faria o contexto aprender a fonte.

O QUE O DESENHO PERMITE CONCLUIR. Dentro de um par o contexto e constante:
ele nao reordena pontos do mesmo par (so muda a inclinacao do HAND). Ganho,
se houver, aparece em calibracao entre eventos (Brier, log-loss) e na AUC
agrupada; a AUC media DENTRO do par e quase a mesma por construcao. Isso e
reconstrucao historica (contexto ate o pico); previsao exigiria so
informacao anterior ao instante-alvo.

nao_e: nao e modelo servido nem promocao. E a medida do ganho do contexto,
inclusive ausencia de ganho.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

from proveniencia import agora_utc, sha256_json  # noqa: E402

CTX = REPO / "local_runs" / "ctx-03-contexto-em-lote"
SAIDA = REPO / "local_runs" / "ctx-04-terreno-versus-contexto"
DOC = REPO / "docs" / "CONTEXTO_HIDROLOGICO_FLUVIAL.md"
TERRENO = ["hand_m", "twi_dinf"]
CONTEXTO = ["pico_percentil", "chuva_log"]
SEMENTE = 20261003
BOOTSTRAP = 1000
MINIMO_EVENTOS = 8


def montar() -> tuple[pd.DataFrame, dict]:
    import admissao_cientifica as adm
    from ds06_dataset_de_treino import versao_atual
    origem = versao_atual()
    p = pd.read_parquet(origem / "pontos.parquet")
    lig = pd.read_parquet(CTX / "pontos_ligados.parquet")
    par = pd.read_csv(CTX / "contexto_por_par.csv")
    ok = ((par.bacia_estado == "bacia_conferida") & (par.pico_incompleto == False)  # noqa: E712
          & par.pico_percentil.notna() & par.chuva_acum_7d_ate_pico_mm.notna())
    recusa = par.loc[~ok, ["evento", "estacao"]].assign(motivo=np.select(
        [par.bacia_estado[~ok] != "bacia_conferida", par.pico_incompleto[~ok] != False,  # noqa: E712
         par.pico_percentil[~ok].isna()],
        ["bacia_nao_conferida", "pico_incompleto_ou_ausente", "sem_percentil"], "sem_chuva"))
    par = par.loc[ok, ["evento", "estacao", "pico_percentil", "chuva_acum_7d_ate_pico_mm"]]
    d = (p.merge(lig, on="ponto_id").merge(par, on=["evento", "estacao"]))
    d, barrado = adm.filtrar(d, "avaliacao_espacial", TERRENO)
    d = d.assign(chuva_log=np.log1p(d.chuva_acum_7d_ate_pico_mm.astype(float)),
                 par=d.evento + "|" + d.estacao.astype(str))
    return d, {"dataset": origem.name, "pares_recusados": recusa.to_dict("records"),
               "barrado_na_admissao": barrado}


def desenho(d: pd.DataFrame, com_contexto: bool) -> np.ndarray:
    X = d[TERRENO].to_numpy(float)
    if not com_contexto:
        return X
    c = d[CONTEXTO].to_numpy(float)
    return np.column_stack([X, c, X[:, :1] * c])


def ajustar_prever(tr: pd.DataFrame, te: pd.DataFrame, com_contexto: bool) -> np.ndarray:
    Xtr, Xte = desenho(tr, com_contexto), desenho(te, com_contexto)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    w = 1.0 / tr.groupby("par").par.transform("size").to_numpy()
    m = LogisticRegression(C=1.0, max_iter=2000).fit((Xtr - mu) / sd, tr.classe, sample_weight=w)
    return m.predict_proba((Xte - mu) / sd)[:, 1]


def metricas(y, p, w, pares) -> dict:
    dentro = []
    for g in np.unique(pares):
        m = pares == g
        if len(np.unique(y[m])) == 2:
            dentro.append(roc_auc_score(y[m], p[m]))
    return {"auc_agrupada": float(roc_auc_score(y, p, sample_weight=w)),
            "auc_media_dentro_do_par": float(np.mean(dentro)) if dentro else None,
            "brier": float(brier_score_loss(y, p, sample_weight=w)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), sample_weight=w))}


def executar(d: pd.DataFrame) -> dict:
    eventos = sorted(d.evento.unique())
    if len(eventos) < MINIMO_EVENTOS:
        return {"estado": "not_testable", "motivo": f"{len(eventos)} eventos < {MINIMO_EVENTOS}"}
    previsto = {False: np.zeros(len(d)), True: np.zeros(len(d))}
    for ev in eventos:
        te = (d.evento == ev).to_numpy()
        for cc in (False, True):
            previsto[cc][te] = ajustar_prever(d[~te], d[te], cc)
    y = d.classe.to_numpy()
    w = 1.0 / d.groupby("par").par.transform("size").to_numpy()
    pares = d.par.to_numpy()
    m0, m1 = metricas(y, previsto[False], w, pares), metricas(y, previsto[True], w, pares)
    rng = np.random.default_rng(SEMENTE)
    idx_ev = {e: np.flatnonzero(d.evento.to_numpy() == e) for e in eventos}
    difs = {k: [] for k in m0}
    for _ in range(BOOTSTRAP):
        amostra = rng.choice(eventos, size=len(eventos), replace=True)
        i = np.concatenate([idx_ev[e] for e in amostra])
        if len(np.unique(y[i])) < 2:
            continue
        a, b = metricas(y[i], previsto[False][i], w[i], pares[i]), metricas(y[i], previsto[True][i], w[i], pares[i])
        for k in difs:
            if a[k] is not None and b[k] is not None:
                difs[k].append(b[k] - a[k])
    ic = {k: np.percentile(v, [2.5, 97.5]).round(4).tolist() for k, v in difs.items() if len(v) >= 100}
    # Ganho declarado so se o IC da diferenca exclui zero na direcao boa.
    melhora = {"auc_agrupada": ic.get("auc_agrupada", [0, 0])[0] > 0,
               "brier": ic.get("brier", [0, 0])[1] < 0,
               "log_loss": ic.get("log_loss", [0, 0])[1] < 0}
    estado = "supported" if any(melhora.values()) else (
        "not_supported" if all((ic.get(k, [0, 0])[0] <= 0 <= ic.get(k, [0, 0])[1]) for k in melhora) else "inconclusive")
    return {"estado": estado, "eventos": len(eventos), "pares": int(d.par.nunique()),
            "estacoes": int(d.estacao.nunique()), "linhas": int(len(d)),
            "positivos": int(y.sum()), "negativos": int((1 - y).sum()),
            "terreno": {k: round(v, 4) if v is not None else None for k, v in m0.items()},
            "terreno_mais_contexto": {k: round(v, 4) if v is not None else None for k, v in m1.items()},
            "diferenca_ic95_por_evento": ic, "melhora_com_ic_fora_de_zero": melhora,
            "replicas_validas": len(difs["brier"])}


def markdown(r: dict) -> str:
    L = ["# Contexto hidrológico do evento no fluvial", "",
         f"**Gerado de** `scripts/metodo/ctx04_terreno_versus_contexto.py` em {r['quando_utc']}. **Não editar à mão.**",
         "", f"**Não é.** {r['nao_e']}", ""]
    for nome, e in r["analises"].items():
        L += [f"## {nome}", ""]
        if e.get("estado") == "not_testable":
            L += [f"`not_testable`: {e['motivo']}", ""]
            continue
        L += [f"Estado: **`{e['estado']}`** — {e['eventos']} eventos, {e['pares']} pares evento×estação, "
              f"{e['estacoes']} estações, {e['positivos']:,} positivos / {e['negativos']:,} negativos observados."
              .replace(",", "."), "",
              "| métrica | terreno | terreno + contexto | diferença (IC95 por evento) |", "|---|---:|---:|---|"]
        for k in e["terreno"]:
            L.append(f"| {k} | {e['terreno'][k]} | {e['terreno_mais_contexto'][k]} | "
                     f"{e['diferenca_ic95_por_evento'].get(k, '—')} |")
        L.append("")
    L += ["## Pares recusados antes do ajuste", ""]
    L += [f"- {x['evento']} · {x['estacao']}: `{x['motivo']}`" for x in r["montagem"]["pares_recusados"]] or ["- nenhum"]
    L += ["", "## Leitura obrigatória", "",
          "- Contexto é constante dentro do par: não reordena pontos do mesmo par; a AUC média dentro do par quase não muda por construção.",
          "- Ganho só conta com IC95 da diferença fora de zero, reamostrando **eventos**.",
          "- É reconstrução histórica: a chuva vai até o pico. Previsão exigiria só informação anterior ao instante-alvo.",
          "- Percentil do pico com telemetria de 15 min contra história diária superestima o percentil (ver `ctx03`).",
          "", f"`{r['hash_resumo']}`", ""]
    return "\n".join(L)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    import rastreio_de_impacto as ri
    ri.exigir_atual(ri.NOS["terreno_versus_contexto"]["depende"], "ctx04")
    d, mont = montar()
    analises = {"GFD (protocolo único de amostragem)": executar(d[d.fonte == "gfd"]),
                "Todas as fontes (sensibilidade; mistura protocolos)": executar(d)}
    r = {"tipo": "terreno_versus_contexto", "quando_utc": agora_utc(), "montagem": mont,
         "desenho": {"terreno": TERRENO, "contexto": CONTEXTO, "interacao": "hand x contexto",
                     "corte": "deixa-um-evento-fora", "peso": "igual por par", "bootstrap": BOOTSTRAP,
                     "semente": SEMENTE, "minimo_eventos": MINIMO_EVENTOS},
         "analises": analises,
         "nao_e": ("nao e modelo servido nem promocao: e a medida do ganho do contexto do evento sobre o "
                   "terreno, inclusive ausencia de ganho")}
    r["hash_resumo"] = sha256_json({k: v for k, v in r.items() if k != "quando_utc"})
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "card.json").write_text(json.dumps(r, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    DOC.write_text(markdown(r), encoding="utf-8")
    ri.carimbar("terreno_versus_contexto", "ctx04")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("estado", "eventos", "pares", "terreno",
                                                                   "terreno_mais_contexto", "diferenca_ic95_por_evento")}
                      for k, v in analises.items()}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
