"""Onde o modelo de suscetibilidade generaliza e onde falha, com incerteza separada.

-------------------------------------------------------------------------
A PERGUNTA
-------------------------------------------------------------------------

Com os dados admitidos (fluvial de planicie, GFD e CEMS, negativo observado),
o que um modelo aprende que transfere para cheia nao vista, e onde falha?

  M0 regra_hand          escore = -HAND. Zero parametro: o baseline fisico.
  M1 logistica_terreno   as 4 variaveis das hipoteses sustentadas.
  M2 logistica_tcs       + construido_frac, argila, areia (cobertura/solo).
  M3 gbm_terreno         nao linear, hiperparametros FIXADOS aqui, sem busca.
  M4 gbm_tcs             idem com cobertura/solo.

Todas as variantes usam AS MESMAS LINHAS (completas nas 7 variaveis), as
mesmas dobras e o mesmo teste. Comparacoes declaradas em COMPARACOES.

-------------------------------------------------------------------------
DESENHO: validacao cruzada agrupada por EVENTO
-------------------------------------------------------------------------

Cada evento cai em exatamente uma dobra de teste; o treino daquela dobra
perde os eventos de teste E todo tile que eles tocam. Cada evento recebe uma
previsao fora da amostra, e a inferencia e feita sobre EVENTOS -- a unidade
independente que existe. (Um primeiro desenho com 10 reservas de 25% deu 3
eventos de teste no CEMS e um bootstrap degenerado; foi descartado no teste
de fumaca, antes de qualquer pre-registro, e o que foi visto ali esta
declarado em `visto_antes_do_pre_registro`.)

-------------------------------------------------------------------------
REGIME: EXPLORATORIO, e por que
-------------------------------------------------------------------------

`reservas_examinadas` mostrou que nao ha reserva intacta: o vetor foi
escolhido com todos os eventos e o lacre de 01/10 foi aberto em 02/10. Tudo
aqui e desenvolvimento. Os criterios sao gravados com hash antes da primeira
execucao (`pre_registro.json`); uma execucao com criterio diferente falha.

-------------------------------------------------------------------------
TRES INCERTEZAS, nunca somadas sem dizer
-------------------------------------------------------------------------

  eventos do teste   Wilcoxon e bootstrap sobre EVENTOS. Ponto e tile do mesmo
                     evento nao sao independentes.
  atribuicao         qual evento caiu em qual dobra: 3 atribuicoes por sal,
                     relatadas como sensibilidade, nao como replica.
  ordem de treino    so na curva diversidade x volume.

Duas AUCs: agrupada (pontos de todas as cheias juntos) e dentro do evento
(ordenar lugares DENTRO de uma cheia). Uma pode piorar enquanto a outra
melhora; nenhuma e "a certa". A AUC agrupada fora da dobra mistura escores
de modelos ajustados em dobras diferentes, e isso fica dito.

nao_e: nao e modelo servido, nao promove nada, nao e probabilidade territorial
(a amostragem e condicionada ao footprint do produto) e nao e causalidade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "generalizacao-por-reservas"
TERRENO = ("hand_m", "twi_dinf", "slope_deg", "elevation_local_m")
COBERTURA_SOLO = ("construido_frac", "argila_pct_0_30", "areia_pct_0_30")
TODAS = TERRENO + COBERTURA_SOLO
FONTES = ("gfd", "cems")
DOBRAS = {"gfd": 5, "cems": 13}          # cems: um evento por dobra quando ha 13
SAIS_ATRIBUICAO = ("a1", "a2", "a3")
MAX_LINHAS_TREINO = 60_000
BOOTSTRAP = 500
MIN_LINHAS_EVENTO = 20
MIN_EVENTOS_PARA_INFERENCIA = 8
SEMENTE = 20261003
GBM = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
           l2_regularization=1.0, early_stopping=False)
MODELOS = {
    "M0_regra_hand": ("regra", ("hand_m",)),
    "M1_logistica_terreno": ("logistica", TERRENO),
    "M2_logistica_tcs": ("logistica", TODAS),
    "M3_gbm_terreno": ("gbm", TERRENO),
    "M4_gbm_tcs": ("gbm", TODAS),
}
COMPARACOES = (
    ("M1_logistica_terreno", "M0_regra_hand", "aprender pesos supera a regra fisica de uma variavel?"),
    ("M3_gbm_terreno", "M1_logistica_terreno", "nao linearidade acrescenta ao terreno?"),
    ("M2_logistica_tcs", "M1_logistica_terreno", "cobertura/solo acrescentam a logistica?"),
    ("M4_gbm_tcs", "M3_gbm_terreno", "cobertura/solo acrescentam ao gbm?"),
)
RELEVANCIA_AUC = 0.01
ALFA = 0.05
VISTO_ANTES = ("teste de fumaca de 03/10 com reservas de 25%: CEMS sal g01 (3 eventos de teste) -- "
               "M3>M1 e M2>M1 por ~0,03-0,04 de AUC agrupada; desenho descartado por bootstrap "
               "degenerado, nao pelo resultado")


def pre_registro() -> dict:
    corpo = {"dobras": DOBRAS, "sais_atribuicao": SAIS_ATRIBUICAO, "modelos": MODELOS, "gbm": GBM,
             "comparacoes": COMPARACOES, "relevancia_auc": RELEVANCIA_AUC, "alfa": ALFA,
             "regra": ("por comparacao e por fonte, na atribuicao a1: Wilcoxon pareado sobre a "
                       "diferenca de AUC DENTRO do evento; ganho/perda se p<alfa e |mediana|>="
                       "relevancia; sem_diferenca_relevante se o IC95 bootstrap por evento da media "
                       "cabe em +-relevancia; senao indecidivel. AUC agrupada e a2/a3 sao descritivas"),
             "min_eventos_para_inferencia": MIN_EVENTOS_PARA_INFERENCIA,
             "bootstrap": BOOTSTRAP, "max_linhas_treino": MAX_LINHAS_TREINO,
             "min_linhas_evento": MIN_LINHAS_EVENTO, "semente": SEMENTE,
             "visto_antes_do_pre_registro": VISTO_ANTES}
    return json.loads(json.dumps(corpo))


def gravar_pre_registro(destino: Path) -> str:
    """Primeira execucao grava; as seguintes so rodam com os mesmos criterios."""
    corpo = pre_registro()
    h = sha256_json(corpo)
    if destino.exists():
        antigo = json.loads(destino.read_text(encoding="utf-8"))
        if antigo["hash"] != h:
            raise RuntimeError(
                f"criterios mudaram depois do pre-registro ({antigo['hash']} -> {h}); "
                "mudar criterio depois de ver resultado e escolha retrospectiva")
        return h
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"quando_utc": agora_utc(), "hash": h, "criterios": corpo},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
    return h


def perda_por_classe(antes: pd.DataFrame, depois: pd.DataFrame) -> dict:
    """Fracao de cada classe que a selecao tirou. Perda desigual entre classes e selecao pelo desfecho."""
    out = {}
    for c in (0, 1):
        n = int((antes.classe == c).sum())
        out[str(c)] = round(1 - int((depois.classe == c).sum()) / n, 4) if n else None
    return out


def carregar(vetor_da_populacao=TODAS) -> tuple[pd.DataFrame, str, dict]:
    """Linhas GFD/CEMS de planicie admitidas, completas em `vetor_da_populacao`.

    O padrao (TODAS) e o do pre-registro de 03/10: as cinco comparacoes nas MESMAS
    linhas, o que exige SoilGrids. Medido em 06/10 (auditoria pre-treino): no GFD
    isso tira 24,1% dos positivos e 6,4% dos negativos (agua, urbano e varzea,
    onde o SoilGrids mascara) -- selecao pelo desfecho, e os executores
    confirmatorios (gfdq03, rfd03) treinaram nessa populacao e testaram sem ela.
    Para comparacoes so de terreno, `vetor_da_populacao=TERRENO` usa a populacao
    inteira; o padrao fica para reproduzir o protocolo congelado.
    """
    import admissao_cientifica as adm
    from ds06_dataset_de_treino import versao_atual
    origem = versao_atual()
    p = pd.read_parquet(origem / "pontos.parquet")
    d = p[(p.mecanismo == "FLUVIAL_ENXURRADA") & (p.classe_relevo == "PLANO_OU_ONDULADO")
          & p.fonte.isin(list(FONTES))]
    # A admissao recebe o vetor como `preditores` e ja exige terreno+cobertura finitos:
    # o corte por SoilGrids acontece AQUI, nao so no dropna. A perda e medida contra a
    # populacao admitida so com o terreno.
    antes, _ = adm.filtrar(d, "avaliacao_espacial", list(TERRENO))
    antes = antes.dropna(subset=["evento_id", "grupo_validacao"])
    d, barrado = adm.filtrar(d, "avaliacao_espacial", list(vetor_da_populacao))
    adm.exigir(d, "avaliacao_espacial", "generalizacao_por_reservas", list(vetor_da_populacao))
    d = d.dropna(subset=list(vetor_da_populacao) + ["evento_id", "grupo_validacao"]).copy()
    d["evento_id"] = d.evento_id.astype(str)
    return d, origem.name, {"barrado_na_admissao": barrado, "sem_cobertura_solo": len(antes) - len(d),
                            "populacao_completa_em": list(vetor_da_populacao),
                            "perda_por_classe": {f: perda_por_classe(antes[antes.fonte == f], d[d.fonte == f])
                                                 for f in FONTES}}


def checar_populacao(filtro_do_treino: dict, exigidas_no_teste, vetor) -> dict:
    """Treino e teste selecionados pela MESMA regra de completude?

    `filtro_do_treino` e o terceiro retorno de `carregar`; `exigidas_no_teste` as
    colunas cujo NaN o executor tira do teste. Coluna exigida so no treino e fora
    do vetor do modelo muda a populacao de um lado so: e o caso do gfdq03 e do
    rfd03 (treino completo em SoilGrids, teste nao), que fica registrado e nao
    pode ser refeito. Executor novo chama isto antes de ajustar e falha.
    """
    so_treino = sorted(set(filtro_do_treino["populacao_completa_em"]) - set(vetor) - set(exigidas_no_teste))
    if so_treino:
        raise ValueError(f"treino completo em {so_treino}, fora do vetor {list(vetor)} e nao exigido no teste: "
                         "populacoes diferentes; use carregar(vetor_da_populacao=<o vetor>)")
    return {"colunas_completas_so_no_treino": [], "mesma_regra": True}


def dobras(eventos, k: int, sal: str) -> dict:
    ordem = sorted(eventos, key=lambda e: hashlib.sha256((e + sal).encode()).hexdigest())
    return {e: i % k for i, e in enumerate(ordem)}


def reservar(eventos, sal: str, fracao: float = 0.25) -> list[str]:
    ordem = sorted(eventos, key=lambda e: hashlib.sha256((e + sal).encode()).hexdigest())
    return sorted(ordem[:max(1, round(len(ordem) * fracao))])


def separar(d: pd.DataFrame, teste_ev) -> tuple[pd.DataFrame, pd.DataFrame]:
    te = d[d.evento_id.isin(teste_ev)]
    tr = d[~d.evento_id.isin(teste_ev) & ~d.grupo_validacao.isin(te.grupo_validacao)]
    assert not set(tr.evento_id) & set(te.evento_id)
    assert not set(tr.grupo_validacao) & set(te.grupo_validacao)
    return tr, te


def logistica_com_intercepto(X: np.ndarray, y: np.ndarray, iteracoes: int = 30) -> np.ndarray | None:
    """O mesmo IRLS de `sinais_marginais.ajustar`, devolvendo TAMBEM o intercepto.

    A primeira versao deste modulo reaproveitava `ajustar` (que descarta o
    intercepto) e recuperava o intercepto por Newton a partir de zero. Em
    amostras pequenas esse Newton divergia (c = 3,7e11 medido em 03/10), todo
    escore saturava em 1 e a AUC caia para 0,5 por empate. O resultado com esse
    defeito ficou preservado como `resultado_defeito_intercepto_2026_10_03.json`.
    """
    A = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(A.shape[1])
    for _ in range(iteracoes):
        p = expit(A @ b)
        W = p * (1 - p)
        H = A.T @ (A * W[:, None]) + 1e-9 * np.eye(A.shape[1])
        passo = np.linalg.solve(H, A.T @ (y - p))
        b += passo
        if np.max(np.abs(passo)) < 1e-8:
            return b
    return b if np.all(np.isfinite(b)) else None


def ajustar_modelo(nome: str, tr: pd.DataFrame, semente: int = SEMENTE):
    tipo, vetor = MODELOS[nome]
    vetor = list(vetor)
    if tipo == "regra":
        return lambda te: -te["hand_m"].to_numpy(float)
    if tr.classe.nunique() < 2:
        return None
    if len(tr) > MAX_LINHAS_TREINO:
        tr = tr.sample(MAX_LINHAS_TREINO, random_state=semente)
    X = tr[vetor].to_numpy(float)
    y = tr.classe.to_numpy(int)
    if tipo == "logistica":
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        b = logistica_com_intercepto((X - mu) / sd, y.astype(float))
        if b is None:
            return None
        return lambda te: expit(b[0] + ((te[vetor].to_numpy(float) - mu) / sd) @ b[1:])
    m = HistGradientBoostingClassifier(random_state=semente, **GBM).fit(X, y)
    return lambda te: m.predict_proba(te[vetor].to_numpy(float))[:, 1]


def auc_por_evento(te: pd.DataFrame, escore: np.ndarray) -> dict:
    out = {}
    y = te.classe.to_numpy()
    for ev, idx in te.groupby("evento_id").indices.items():
        yy = y[idx]
        if len(idx) >= MIN_LINHAS_EVENTO and 0 < yy.sum() < len(yy):
            out[ev] = float(roc_auc_score(yy, escore[idx]))
    return out


def fora_da_dobra(s: pd.DataFrame, k: int, sal: str, modelos=None) -> tuple[pd.DataFrame, dict]:
    """Uma previsao fora da amostra por linha, para cada modelo."""
    modelos = list(modelos or MODELOS)
    atrib = dobras(sorted(s.evento_id.unique()), k, sal)
    s = s.assign(dobra=s.evento_id.map(atrib))
    pred = {m: np.full(len(s), np.nan) for m in modelos}
    suporte = []
    for f in range(k):
        teste_ev = sorted(e for e, x in atrib.items() if x == f)
        tr, te = separar(s, teste_ev)
        pos = np.flatnonzero(s.dobra.to_numpy() == f)
        suporte.append({"dobra": f, "eventos_teste": len(teste_ev), "linhas_teste": len(te),
                        "linhas_treino": len(tr), "eventos_treino": int(tr.evento_id.nunique()),
                        "linhas_de_treino_perdidas_por_tile_compartilhado":
                            int((~s.evento_id.isin(teste_ev)).sum() - len(tr))})
        for m in modelos:
            fn = ajustar_modelo(m, tr)
            if fn is not None:
                pred[m][pos] = fn(te)
    return s, {"atribuicao": sal, "dobras": suporte, "predicoes": pred}


def medir(s: pd.DataFrame, pred: dict) -> dict:
    y = s.classe.to_numpy()
    out = {}
    for m, p in pred.items():
        ok = np.isfinite(p)
        pe = auc_por_evento(s[ok], p[ok])
        out[m] = {"auc_agrupada_fora_da_dobra": round(float(roc_auc_score(y[ok], p[ok])), 4),
                  "average_precision": round(float(average_precision_score(y[ok], p[ok])), 4),
                  "auc_media_dentro_do_evento": round(float(np.mean(list(pe.values()))), 4) if pe else None,
                  "auc_mediana_dentro_do_evento": round(float(np.median(list(pe.values()))), 4) if pe else None,
                  "eventos_com_auc": len(pe),
                  "brier": (round(float(brier_score_loss(y[ok], p[ok])), 4)
                            if MODELOS[m][0] != "regra" else None)}
    return out


def comparar(s: pd.DataFrame, a: np.ndarray, b: np.ndarray, rng) -> dict:
    pa, pb = auc_por_evento(s, a), auc_por_evento(s, b)
    comuns = sorted(set(pa) & set(pb))
    dif = np.array([pa[e] - pb[e] for e in comuns])
    y = s.classe.to_numpy()
    idx = s.groupby("evento_id").indices
    evs = list(idx)
    agrup, media = [], []
    for _ in range(BOOTSTRAP):
        sorteio = rng.choice(len(evs), len(evs))
        i = np.concatenate([idx[evs[k]] for k in sorteio])
        if 0 < y[i].sum() < len(i):
            agrup.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
        if len(dif):
            media.append(dif[rng.choice(len(dif), len(dif))].mean())
    r = {"eventos_pareados": len(comuns),
         "dentro_do_evento_mediana": round(float(np.median(dif)), 4) if len(dif) else None,
         "dentro_do_evento_media": round(float(dif.mean()), 4) if len(dif) else None,
         "eventos_a_favor": int((dif > 0).sum()),
         "ic95_media_dentro": np.percentile(media, [2.5, 97.5]).round(4).tolist() if media else None,
         "agrupada": round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 4),
         "ic95_agrupada_por_evento": np.percentile(agrup, [2.5, 97.5]).round(4).tolist() if agrup else None}
    if len(dif) >= MIN_EVENTOS_PARA_INFERENCIA and np.any(dif != 0):
        r["p_wilcoxon"] = round(float(wilcoxon(dif).pvalue), 5)
    return r


def veredito(c: dict) -> str:
    if c["eventos_pareados"] < MIN_EVENTOS_PARA_INFERENCIA or "p_wilcoxon" not in c:
        return "not_testable"
    med = c["dentro_do_evento_mediana"]
    if c["p_wilcoxon"] < ALFA and abs(med) >= RELEVANCIA_AUC:
        return "ganho" if med > 0 else "perda"
    lo, hi = c["ic95_media_dentro"]
    if -RELEVANCIA_AUC <= lo and hi <= RELEVANCIA_AUC:
        return "sem_diferenca_relevante"
    return "indecidivel"


def territorios(d: pd.DataFrame, rng) -> list[dict]:
    """Territorio inteiro fora: nenhum evento que o toque, nenhum tile dele, no treino."""
    out = []
    for t in sorted(d.territorio.unique()):
        te = d[d.territorio == t]
        tr = d[(d.territorio != t) & ~d.evento_id.isin(te.evento_id)
               & ~d.grupo_validacao.isin(te.grupo_validacao)]
        base = {"territorio": t, "linhas_teste": len(te), "eventos_teste": int(te.evento_id.nunique()),
                "eventos_treino": int(tr.evento_id.nunique()), "linhas_treino": len(tr)}
        if tr.classe.nunique() < 2 or te.classe.nunique() < 2:
            out.append({**base, "estado": "not_testable", "motivo": "uma classe so depois do corte"})
            continue
        esc = {n: ajustar_modelo(n, tr)(te) for n in ("M0_regra_hand", "M1_logistica_terreno",
                                                       "M3_gbm_terreno")}
        cmp = comparar(te, esc["M1_logistica_terreno"], esc["M0_regra_hand"], rng)
        out.append({**base, "estado": "medido", "medidas": medir(te, esc),
                    "M1_vs_M0": {**cmp, "veredito": veredito(cmp)}})
    return out


def entre_fontes(d: pd.DataFrame, espacial: bool = False) -> list[dict]:
    """Treina numa fonte, mede na outra. Reconciliado em 06/10 (`verdade_observacional.identidade`):
    GFD cobre 2002-2018 e CEMS 2023-2026, nenhuma cheia em comum -- transferencia de
    instrumento E de epoca ao mesmo tempo."""
    out = []
    for a, b in (("gfd", "cems"), ("cems", "gfd")):
        tr, te = d[d.fonte == a], d[d.fonte == b]
        antes = len(te)
        if espacial:
            # O nome do grupo muda de natureza entre fontes (tile x AOI): so a coordenada
            # acha o lugar compartilhado (verdade_observacional.chave_espacial).
            import verdade_observacional as vo
            te = te[~vo.chave_espacial(te).isin(set(vo.chave_espacial(tr)))]
        else:
            te = te[~te.grupo_validacao.isin(tr.grupo_validacao)]
        if te.classe.nunique() < 2:
            out.append({"treino": a, "teste": b, "estado": "not_testable", "linhas_teste": len(te),
                        "linhas_retiradas_por_lugar": antes - len(te),
                        "motivo": "depois de tirar o lugar compartilhado sobra uma classe ou nada"})
            continue
        esc = {n: ajustar_modelo(n, tr)(te) for n in ("M0_regra_hand", "M1_logistica_terreno",
                                                       "M3_gbm_terreno")}
        out.append({"treino": a, "teste": b, "linhas_teste": len(te), "linhas_retiradas_por_lugar": antes - len(te),
                    "chave_de_lugar": "tile de 1 grau pela coordenada" if espacial else "nome do grupo",
                    "eventos_teste": int(te.evento_id.nunique()), "medidas": medir(te, esc)})
    return out


def diversidade_versus_volume(d: pd.DataFrame, sal: str, ordens: int, rng) -> dict:
    """Mesmo volume, 4 eventos contra todos: a diferenca com as duas incertezas separadas."""
    tr, te = separar(d, reservar(sorted(d.evento_id.unique()), sal))
    por_ev = {e: s for e, s in tr.groupby("evento_id")}
    evs = sorted(por_ev)
    y = te.classe.to_numpy()
    idx = te.groupby("evento_id").indices
    lista = list(idx)
    if len(lista) < MIN_EVENTOS_PARA_INFERENCIA:
        return {"reserva": sal, "estado": "not_testable", "eventos_teste": len(lista)}
    sorteios = [rng.choice(len(lista), len(lista)) for _ in range(200)]
    por_ordem, matriz, volumes = [], [], []
    for _ in range(ordens):
        perm = list(rng.permutation(evs))
        poucos = pd.concat([por_ev[e] for e in perm[:4]])
        muitos = tr.sample(min(len(poucos), len(tr)), random_state=int(rng.integers(1 << 31)))
        fa = ajustar_modelo("M1_logistica_terreno", muitos)
        fb = ajustar_modelo("M1_logistica_terreno", poucos)
        if fa is None or fb is None:
            continue
        a, b = fa(te), fb(te)
        volumes.append(len(poucos))
        por_ordem.append(float(roc_auc_score(y, a) - roc_auc_score(y, b)))
        linha = []
        for s in sorteios:
            i = np.concatenate([idx[lista[k]] for k in s])
            linha.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i])
                         if 0 < y[i].sum() < len(i) else np.nan)
        matriz.append(linha)
    media_por_bootstrap = np.nanmean(np.array(matriz), axis=0)
    return {"reserva": sal, "estado": "medido", "eventos_teste": len(lista), "ordens": len(por_ordem),
            "linhas_por_braco_mediana": int(np.median(volumes)),
            "diferenca_mediana_entre_ordens": round(float(np.median(por_ordem)), 4),
            "faixa_entre_ordens_95": np.percentile(por_ordem, [2.5, 97.5]).round(4).tolist(),
            "ordens_a_favor_da_diversidade": int(sum(x > 0 for x in por_ordem)),
            "ic95_eventos_do_teste_da_media": np.nanpercentile(media_por_bootstrap, [2.5, 97.5]).round(4).tolist()}


SAIDA_V2 = SAIDA / "v2_populacao_da_tarefa"
MODELOS_TERRENO = ("M0_regra_hand", "M1_logistica_terreno", "M3_gbm_terreno")
COMPARACOES_TERRENO = tuple(c for c in COMPARACOES if c[0] in MODELOS_TERRENO and c[1] in MODELOS_TERRENO)


def pre_registro_v2() -> dict:
    """A v2 muda so a POPULACAO de cada comparacao e a chave de lugar entre fontes.

    Nao e cega: a auditoria pre-treino de 06/10 (sensibilidade S1) ja mediu M0/M1/M3 na
    populacao de terreno com as dobras a1. A v2 e desenvolvimento, como a v1.
    """
    return {"base": pre_registro(), "versao": "v2",
            "populacao": {"comparacoes_de_terreno": "carregar(TERRENO): admitidas e completas no vetor do modelo",
                          "comparacoes_de_cobertura_solo": "carregar(TODAS): as da v1, citadas por hash, nao refeitas"},
            "comparacoes": [list(c) for c in COMPARACOES_TERRENO],
            "entre_fontes": "teste sem tile de 1 grau ocupado pelo treino (verdade_observacional.chave_espacial)",
            "identidade": "GFD e CEMS nao tem par equivalente entre si (verdade_observacional.identidade)",
            "visto_antes_do_pre_registro": ("auditoria_pre_treino S1 (06/10): M1xM0 e M3xM1 na populacao de "
                                            "terreno, dobras a1, GFD e CEMS"),
            "nao_e": "nao substitui a v1: a v1 fica como medida, na populacao dela"}


def executar_v2(ordens: int = 20) -> int:
    import rastreio_de_impacto as ri
    import reservas_examinadas as rx
    _entrada, _saida = ri.ganchos("generalizacao_v2", "generalizacao_por_reservas --protocolo v2")
    _entrada()
    h = sha256_json(pre_registro_v2())
    alvo = SAIDA_V2 / "pre_registro.json"
    SAIDA_V2.mkdir(parents=True, exist_ok=True)
    if alvo.exists() and json.loads(alvo.read_text(encoding="utf-8"))["hash"] != h:
        raise RuntimeError("pre-registro v2 mudou depois de gravado: crie v3")
    if not alvo.exists():
        alvo.write_text(json.dumps({"quando_utc": agora_utc(), "hash": h, "criterios": pre_registro_v2()},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    d, versao, filtro = carregar(TERRENO)
    rng = np.random.default_rng(SEMENTE)
    inicio = time.time()
    estudos = {}
    for fonte in FONTES:
        parcial = SAIDA_V2 / f"parcial_{fonte}.json"
        if parcial.exists():
            p = json.loads(parcial.read_text(encoding="utf-8"))
            if p.get("pre_registro") == h and p.get("dataset") == versao:
                estudos[fonte] = p["estudo"]
                continue
        s0 = d[d.fonte == fonte]
        k = min(DOBRAS[fonte], s0.evento_id.nunique())
        atribuicoes = {}
        for sal in SAIS_ATRIBUICAO:
            s, cv = fora_da_dobra(s0, k, sal, MODELOS_TERRENO)
            cmp = {}
            for a, b, q in COMPARACOES_TERRENO:
                c = comparar(s, cv["predicoes"][a], cv["predicoes"][b], rng)
                cmp[f"{a}_vs_{b}"] = {"pergunta": q, **c, "veredito": veredito(c)}
            atribuicoes[sal] = {"dobras": cv["dobras"], "medidas": medir(s, cv["predicoes"]), "comparacoes": cmp,
                                "auc_por_evento_M1": auc_por_evento(s, cv["predicoes"]["M1_logistica_terreno"])}
            print(f"v2 {fonte} {sal} {round(time.time() - inicio)}s "
                  f"{ {k2: v['veredito'] for k2, v in cmp.items()} }", flush=True)
        estudos[fonte] = {"eventos": int(s0.evento_id.nunique()), "linhas": len(s0), "dobras": k,
                          "fracao_positiva": round(float(s0.classe.mean()), 4),
                          "regime": rx.regime(sorted(s0.evento_id.unique()), fonte)["regime"],
                          "atribuicoes": atribuicoes}
        parcial.write_text(json.dumps({"pre_registro": h, "dataset": versao, "estudo": estudos[fonte]},
                                      ensure_ascii=False, default=float), encoding="utf-8")
    g = d[d.fonte == "gfd"]
    estudos["gfd"]["diversidade_versus_volume"] = [diversidade_versus_volume(g, sal, ordens, rng)
                                                   for sal in ("v1", "v2", "v3")]
    estudos["gfd"]["territorio_reservado"] = territorios(g, rng)
    v1 = SAIDA / "resultado.json"
    r = {"tipo": "generalizacao_por_reservas", "versao_do_protocolo": "v2", "dataset": versao,
         "pre_registro": h, "regime": "exploratorio", "filtro": filtro, "estudos": estudos,
         "entre_fontes": entre_fontes(d, espacial=True),
         "comparacoes_de_cobertura_solo": {"origem": str(v1.relative_to(REPO)),
                                           "sha256": "sha256:" + hashlib.sha256(v1.read_bytes()).hexdigest()
                                           if v1.exists() else None,
                                           "por_que": "populacao caso completo, a mesma da v1: nao refeitas"},
         "nao_comparavel_com_v1": ("as AUC mudam porque a populacao muda (entra o que o SoilGrids mascara: agua, "
                                   "urbano, varzea); compare vereditos e direcao, nao o nivel da AUC"),
         "nao_e": "nao e modelo servido nem promocao; nao e probabilidade territorial; nao e causalidade"}
    r["hash_resumo"] = sha256_json(json.loads(json.dumps(r, default=float)))
    r["quando_utc"] = agora_utc()
    r["segundos"] = round(time.time() - inicio)
    (SAIDA_V2 / "resultado.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float),
                                             encoding="utf-8")
    print(f"GRAVADO={SAIDA_V2 / 'resultado.json'}")
    _saida()
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ordens", type=int, default=20)
    ap.add_argument("--protocolo", choices=("v1", "v2"), default="v1",
                    help="v1: populacao do pre-registro de 03/10 (caso completo); v2: populacao da tarefa")
    args = ap.parse_args()
    if args.protocolo == "v2":
        return executar_v2(args.ordens)
    import reservas_examinadas as rx
    h = gravar_pre_registro(SAIDA / "pre_registro.json")
    d, versao, filtro = carregar()
    rng = np.random.default_rng(SEMENTE)
    inicio = time.time()
    estudos = {}
    for fonte in FONTES:
        # Checkpoint por fonte: uma execucao interrompida retoma daqui, e so com
        # o MESMO pre-registro e o mesmo dataset -- senao recalcula.
        parcial = SAIDA / f"parcial_{fonte}.json"
        if parcial.exists():
            p = json.loads(parcial.read_text(encoding="utf-8"))
            if p.get("pre_registro") == h and p.get("dataset") == versao:
                estudos[fonte] = p["estudo"]
                print(f"{fonte}: retomado de {parcial.name}", flush=True)
                continue
        s0 = d[d.fonte == fonte]
        k = min(DOBRAS[fonte], s0.evento_id.nunique())
        atribuicoes = {}
        for sal in SAIS_ATRIBUICAO:
            s, cv = fora_da_dobra(s0, k, sal)
            medidas = medir(s, cv["predicoes"])
            cmp = {}
            for a, b, q in COMPARACOES:
                c = comparar(s, cv["predicoes"][a], cv["predicoes"][b], rng)
                cmp[f"{a}_vs_{b}"] = {"pergunta": q, **c, "veredito": veredito(c)}
            pe = auc_por_evento(s, cv["predicoes"]["M1_logistica_terreno"])
            atribuicoes[sal] = {"dobras": cv["dobras"], "medidas": medidas, "comparacoes": cmp,
                                "auc_por_evento_M1": pe}
            print(f"{fonte} {sal} {round(time.time() - inicio)}s "
                  f"{ {k2: v['veredito'] for k2, v in cmp.items()} }", flush=True)
        info = s0.groupby("evento_id").agg(territorio=("territorio", "first"), linhas=("classe", "size"),
                                           prevalencia=("classe", "mean"))
        pe = pd.Series(atribuicoes["a1"]["auc_por_evento_M1"], name="auc_M1")
        erros = info.join(pe, how="inner").sort_values("auc_M1")
        estudos[fonte] = {
            "eventos": int(s0.evento_id.nunique()), "linhas": len(s0), "dobras": k,
            "regime": rx.regime(sorted(s0.evento_id.unique()), fonte)["regime"],
            "atribuicoes": atribuicoes,
            "eventos_abaixo_de_0_6_M1": int((erros.auc_M1 < 0.6).sum()),
            "piores_eventos_M1": erros.head(8).reset_index().round(3).to_dict("records"),
            "auc_M1_por_territorio": erros.groupby("territorio").auc_M1.agg(["count", "median"]).round(3)
                                         .reset_index().to_dict("records")}
        SAIDA.mkdir(parents=True, exist_ok=True)
        parcial.write_text(json.dumps({"pre_registro": h, "dataset": versao, "estudo": estudos[fonte]},
                                      ensure_ascii=False, default=float), encoding="utf-8")
    estudos["gfd"]["diversidade_versus_volume"] = [
        diversidade_versus_volume(d[d.fonte == "gfd"], sal, args.ordens, rng) for sal in ("v1", "v2", "v3")]
    estudos["gfd"]["territorio_reservado"] = territorios(d[d.fonte == "gfd"], rng)
    r = {"tipo": "generalizacao_por_reservas", "dataset": versao, "pre_registro": h,
         "regime": "exploratorio", "por_que_exploratorio": rx.lacre_aberto().get("causa"),
         "filtro": filtro, "estudos": estudos, "entre_fontes": entre_fontes(d),
         "observacao_sar": ("not_testable como preditor: a agua observada no evento e o alvo, nao "
                            "contexto pre-evento; usa-la seria vazamento"),
         "embedding_terreno": ("not_testable: o encoder da frente B so tem embedding dos chips do "
                               "Sen1Floods11, nao destes pontos; comparar exigiria extrair embedding "
                               "por ponto com o mesmo protocolo"),
         "contexto_hidrologico": ("fora deste experimento: ctx04 cobre 11 eventos GFD e ficou "
                                  "not_supported; incluir aqui reduziria todas as linhas a esses eventos"),
         "nao_e": ("nao e modelo servido nem promocao; nao e probabilidade territorial; nao e "
                   "causalidade; atribuicoes de dobra nao sao replicas independentes")}
    r["hash_resumo"] = sha256_json(r)
    r["quando_utc"] = agora_utc()
    r["segundos"] = round(time.time() - inicio)
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "resultado.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float),
                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
