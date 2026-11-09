"""
Autopsia temporal de Curitiba: descobrir a fronteira, nao defender o modelo.

-------------------------------------------------------------------------
O QUE SE SABE ANTES DE COMECAR
-------------------------------------------------------------------------

Curitiba tem desempenho razoavel em validacao embaralhada e colapso proximo
do acaso em holdout temporal. Diagnosticos anteriores nao explicaram a quebra
de forma trivial. Este arquivo NAO repete aqueles diagnosticos: ele decompoe
a quebra em partes que podem ser refutadas separadamente.

As sete hipoteses estao escritas em `protocolo_experimental.HIPOTESES`, ANTES
de qualquer numero. Cada uma termina em `supported`, `not_supported` ou
`inconclusive` -- nunca `proved`, porque nada aqui identifica causa.

-------------------------------------------------------------------------
A DISTINCAO QUE ORGANIZA A ANALISE
-------------------------------------------------------------------------

Um modelo pode degradar no tempo por duas razoes diferentes, que pedem
respostas opostas:

  P(X) mudou       covariate shift. As areas consultadas em 2026 nao se
                   parecem com as de treino. A resposta e ABSTENCAO -- o
                   modelo nao deveria ter respondido.

  P(Y|X) mudou     a mesma configuracao de terreno passou a ter outro
                   desfecho. A resposta e outra: o modelo esta errado, e
                   abster nao conserta.

O discriminador de periodo separa as duas de forma OPERACIONAL, e nao causal:
se um classificador distingue 2026 do passado usando so as features, ha
covariate shift mensuravel. Se NAO distingue e o modelo mesmo assim cai,
concept shift permanece POSSIVEL -- e "possivel" e o mais forte que este
desenho autoriza dizer.

-------------------------------------------------------------------------
O QUE ESTE ARQUIVO NAO FAZ
-------------------------------------------------------------------------

Nao treina modelo servido para Curitiba. A regiao e servida por
TRANSFERENCIA (`fluvial_planicie`), e continua sendo. O conjunto
`curitiba_autopsia` existe so para responder por que a serie temporal quebra
-- se algum resultado aqui parecer sugerir um modelo local, falta o
inventario que a memoria do projeto ja registra como ausente.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "challengers"))

import matriz_validacao as mv  # noqa: E402
from protocolo_experimental import CRITERIOS_FISICOS, SEED, protocolo  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "experimentos-cientificos"
ANO_TESTE = 2026

SUPPORTED = "supported"
NOT_SUPPORTED = "not_supported"
INCONCLUSIVE = "inconclusive"


# ------------------------------------------------- 1. distribuicao temporal

def distribuicao_temporal(c: mv.Conjunto) -> pd.DataFrame:
    d = c.dados.assign(ano=c.datas.dt.year, _g=c.grupos)
    linhas = []
    for ano, g in d.groupby("ano"):
        linha = {
            "ano": int(ano), "n": len(g),
            "positivos": int(g.classe.sum()),
            "negativos": int((g.classe == 0).sum()),
            "prevalencia": round(float(g.classe.mean()), 4),
            "grupos": int(g._g.nunique()),
            "grupos_positivos": int(g[g.classe == 1]._g.nunique()),
            "datas_distintas": int(g.data_evento.nunique()),
        }
        for f in c.features:
            linha[f"{f}_mediana"] = round(float(g[f].median()), 4)
            linha[f"{f}_p05"] = round(float(g[f].quantile(0.05)), 4)
            linha[f"{f}_p95"] = round(float(g[f].quantile(0.95)), 4)
        linhas.append(linha)
    return pd.DataFrame(linhas).sort_values("ano")


# ------------------------------------------------------ 2. drift de covariaveis

def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Diferenca padronizada de medias, com desvio combinado.

    Interpretavel sem tabela: |SMD| < 0,1 e diferenca desprezivel; > 0,25 e
    substancial. Ao contrario de um teste de hipotese, nao cresce com n --
    e com 1680 linhas qualquer teste rejeita tudo.
    """
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt((sa ** 2 + sb ** 2) / 2)
    if pooled == 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled)


def _psi(a: np.ndarray, b: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index sobre quantis do TREINO."""
    cortes = np.unique(np.quantile(a, np.linspace(0, 1, n_bins + 1)))
    if len(cortes) < 3:
        return 0.0
    ha = np.histogram(a, bins=cortes)[0].astype(float)
    hb = np.histogram(b, bins=cortes)[0].astype(float)
    ha, hb = ha / max(ha.sum(), 1), hb / max(hb.sum(), 1)
    # epsilon evita log(0); o valor e pequeno o bastante para nao dominar
    eps = 1e-6
    ha, hb = np.clip(ha, eps, None), np.clip(hb, eps, None)
    return float(np.sum((hb - ha) * np.log(hb / ha)))


def drift_de_covariaveis(c: mv.Conjunto, ano_teste: int = ANO_TESTE) -> pd.DataFrame:
    from scipy.stats import ks_2samp

    anos = c.datas.dt.year
    tr = (anos < ano_teste).to_numpy(na_value=False)
    te = (anos == ano_teste).to_numpy(na_value=False)
    linhas = []
    for f in c.features:
        a = c.dados.loc[tr, f].to_numpy(dtype=float)
        b = c.dados.loc[te, f].to_numpy(dtype=float)
        if len(a) < 5 or len(b) < 5:
            continue
        lo, hi = np.quantile(a, [0.05, 0.95])
        ks = ks_2samp(a, b)
        linhas.append({
            "feature": f,
            "n_treino": len(a), "n_teste": len(b),
            "media_treino": round(float(np.mean(a)), 4),
            "media_teste": round(float(np.mean(b)), 4),
            "smd": round(_smd(a, b), 4),
            "smd_magnitude": ("desprezivel" if abs(_smd(a, b)) < 0.1
                              else "moderada" if abs(_smd(a, b)) < 0.25
                              else "substancial"),
            "ks_stat": round(float(ks.statistic), 4),
            "ks_pvalue": float(ks.pvalue),
            "psi": round(_psi(a, b), 4),
            "psi_leitura": ("estavel" if _psi(a, b) < 0.1
                            else "moderado" if _psi(a, b) < 0.25 else "grande"),
            "p05_treino": round(float(lo), 4), "p95_treino": round(float(hi), 4),
            "pct_teste_fora_5_95_do_treino": round(
                float(np.mean((b < lo) | (b > hi)) * 100), 2),
        })
    return pd.DataFrame(linhas)


# ------------------------------------------------------- 3. drift condicional

def discriminador_de_periodo(c: mv.Conjunto, ano_teste: int = ANO_TESTE) -> dict:
    """Um classificador distingue 2026 do passado usando SO as features?

    Este e um diagnostico OPERACIONAL, e a distincao importa: AUC alta aqui
    diz que os periodos sao separaveis no espaco de features -- covariate
    shift mensuravel. NAO diz que o shift causou a degradacao do modelo, e
    nao identifica mecanismo.

    A importancia por feature sai como `review-only` de proposito: ela indica
    onde olhar, e nao qual variavel e responsavel.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    anos = c.datas.dt.year
    alvo = (anos == ano_teste).to_numpy(na_value=False).astype(int)
    marcado = (anos <= ano_teste).to_numpy(na_value=False)
    X = c.X[marcado]
    z = alvo[marcado]
    g = c.grupos[marcado]
    if len(np.unique(z)) < 2:
        return {"auc": None, "motivo": "sem os dois periodos"}

    p = np.full(len(z), np.nan)
    k = min(5, len(np.unique(g)))
    for tr, te in GroupKFold(n_splits=k).split(X, z, g):
        if len(np.unique(z[tr])) < 2:
            continue
        mu = X[tr].mean(0)
        sd = np.where(X[tr].std(0) == 0, 1, X[tr].std(0))
        m = LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=SEED)
        m.fit((X[tr] - mu) / sd, z[tr])
        p[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    ok = ~np.isnan(p)
    if ok.sum() == 0 or len(np.unique(z[ok])) < 2:
        return {"auc": None, "motivo": "nenhum fold com os dois periodos"}

    mu = X.mean(0)
    sd = np.where(X.std(0) == 0, 1, X.std(0))
    cheio = LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=SEED)
    cheio.fit((X - mu) / sd, z)
    auc = float(roc_auc_score(z[ok], p[ok]))
    return {
        "auc": round(auc, 4),
        "leitura": ("periodos praticamente indistinguiveis" if auc < 0.6
                    else "periodos parcialmente separaveis" if auc < 0.75
                    else "periodos claramente separaveis"),
        "covariate_shift_mensuravel": bool(auc >= 0.6),
        "importancia_review_only": {
            f: round(float(abs(co)), 4)
            for f, co in zip(c.features, np.asarray(cheio.coef_).ravel())},
        "nao_e": ("diagnostico operacional de separabilidade; nao identifica "
                  "causa nem prova concept drift"),
    }


# -------------------------------------- 4. relacao variavel-desfecho por periodo

def efeitos_por_periodo(c: mv.Conjunto, n_boot: int = 300) -> pd.DataFrame:
    """Coeficiente de cada variavel dentro de cada ano, com IC de grupos.

    Ajustar dentro do ano responde "a relacao mudou?", que e diferente de
    "o modelo do passado erra no futuro". Um sinal que inverte entre periodos
    e evidencia de instabilidade da relacao; ICs que se sobrepoem sao
    evidencia de que a amostra nao distingue.
    """
    from firthlogist import FirthLogisticRegression

    anos = c.datas.dt.year
    rng = np.random.default_rng(SEED)
    linhas = []
    for ano in sorted(anos.dropna().unique()):
        sel = (anos == ano).to_numpy(na_value=False)
        X, y, g = c.X[sel], c.y[sel], c.grupos[sel]
        if len(np.unique(y)) < 2 or len(y) < 30:
            for f in c.features:
                linhas.append({"ano": int(ano), "feature": f, "coef": None,
                               "motivo": "amostra insuficiente no periodo"})
            continue
        mu = X.mean(0)
        sd = np.where(X.std(0) == 0, 1, X.std(0))
        Z = (X - mu) / sd
        m = FirthLogisticRegression(max_iter=500, skip_pvals=True, skip_ci=True)
        m.fit(Z, y)
        coefs = np.asarray(m.coef_).ravel()

        unicos = np.unique(g)
        por_grupo = {u: np.where(g == u)[0] for u in unicos}
        replicas = []
        for _ in range(n_boot):
            escolha = rng.choice(unicos, size=len(unicos), replace=True)
            idx = np.concatenate([por_grupo[u] for u in escolha])
            if len(np.unique(y[idx])) < 2:
                continue
            try:
                mb = FirthLogisticRegression(max_iter=300, skip_pvals=True,
                                             skip_ci=True)
                mb.fit(Z[idx], y[idx])
                replicas.append(np.asarray(mb.coef_).ravel())
            except Exception:
                continue
        arr = np.array(replicas) if replicas else None
        for j, f in enumerate(c.features):
            lo = hi = None
            if arr is not None and len(arr) >= 30:
                lo = round(float(np.percentile(arr[:, j], 2.5)), 4)
                hi = round(float(np.percentile(arr[:, j], 97.5)), 4)
            esperado = CRITERIOS_FISICOS["sinais_esperados"].get(f)
            cruza = None if lo is None else bool(lo <= 0 <= hi)
            linhas.append({
                "ano": int(ano), "feature": f,
                "coef": round(float(coefs[j]), 4),
                "ic95_lo": lo, "ic95_hi": hi,
                "ic_cruza_zero": cruza,
                "direcao_esperada": esperado,
                "direcao_estimada": int(np.sign(coefs[j])),
                "veredito": ("inconclusivo" if cruza in (None, True)
                             else "consistente"
                             if np.sign(coefs[j]) == np.sign(esperado)
                             else "invertido"),
                "n": int(len(y)), "grupos": int(len(unicos)),
                "replicas_bootstrap": 0 if arr is None else len(arr),
            })
    return pd.DataFrame(linhas)


# --------------------------------------------------- 5. decomposicao por evento

def desempenho_por_evento(c: mv.Conjunto, modelo, ano_teste: int = ANO_TESTE,
                          min_n: int = 20) -> pd.DataFrame:
    """AUC do modelo do passado dentro de cada evento do periodo de teste.

    Responde a pergunta que uma AUC agregada esconde: o colapso e geral, ou
    sao poucos eventos muito ruins puxando a media?
    """
    from sklearn.metrics import roc_auc_score

    anos = c.datas.dt.year
    tr = (anos < ano_teste).to_numpy(na_value=False)
    te = (anos == ano_teste).to_numpy(na_value=False)
    if tr.sum() < 30 or te.sum() < 10:
        return pd.DataFrame()

    X, y = c.X, c.y
    mu = X[tr].mean(0)
    sd = np.where(X[tr].std(0) == 0, 1, X[tr].std(0))
    ajustado = modelo.ajustar((X[tr] - mu) / sd, y[tr])
    p = np.full(len(y), np.nan)
    p[te] = modelo.prever(ajustado, (X[te] - mu) / sd)

    d = c.dados.assign(_p=p, _y=y, _te=te, _data=c.datas)
    linhas = []
    for data, g in d[d._te].groupby("_data"):
        if len(g) < min_n or g._y.nunique() < 2:
            linhas.append({"evento": str(pd.Timestamp(data).date()),
                           "n": len(g), "positivos": int(g._y.sum()),
                           "auc": None,
                           "motivo": ("amostra pequena" if len(g) < min_n
                                      else "uma classe so")})
            continue
        linhas.append({
            "evento": str(pd.Timestamp(data).date()),
            "n": len(g), "positivos": int(g._y.sum()),
            "prevalencia": round(float(g._y.mean()), 4),
            "auc": round(float(roc_auc_score(g._y, g._p)), 4),
            "escore_medio": round(float(g._p.mean()), 4),
        })
    return pd.DataFrame(linhas).sort_values("n", ascending=False)


# ------------------------------------------------------- 6. cartografia do erro

def cartografia_do_erro(c: mv.Conjunto, modelo, ano_teste: int = ANO_TESTE,
                        lado: float = 0.02) -> tuple:
    """Onde o modelo erra no periodo de teste, por bloco e por faixa de HAND."""
    anos = c.datas.dt.year
    tr = (anos < ano_teste).to_numpy(na_value=False)
    te = (anos == ano_teste).to_numpy(na_value=False)
    if tr.sum() < 30 or te.sum() < 10:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X, y = c.X, c.y
    mu = X[tr].mean(0)
    sd = np.where(X[tr].std(0) == 0, 1, X[tr].std(0))
    ajustado = modelo.ajustar((X[tr] - mu) / sd, y[tr])
    p = modelo.prever(ajustado, (X[te] - mu) / sd)

    d = c.dados[te].copy()
    d["escore"] = p
    d["y"] = y[te]
    d["predito"] = (p >= 0.5).astype(int)
    d["erro"] = (d.predito != d.y).astype(int)
    d["tipo_erro"] = np.where(
        d.erro == 0, "acerto",
        np.where(d.predito == 1, "falso_positivo", "falso_negativo"))

    pontos = d[["lat", "lon", "escore", "y", "predito", "erro", "tipo_erro",
                "hand_m", "twi_dinf", "slope_deg", "elevation_m"]].copy()

    d["bloco_lat"] = (d.lat / lado).round().astype("int64")
    d["bloco_lon"] = (d.lon / lado).round().astype("int64")
    por_bloco = (d.groupby(["bloco_lat", "bloco_lon"])
                 .agg(n=("y", "size"), positivos=("y", "sum"),
                      taxa_erro=("erro", "mean"),
                      escore_medio=("escore", "mean"),
                      lat=("lat", "mean"), lon=("lon", "mean"))
                 .reset_index())
    por_bloco["taxa_erro"] = por_bloco.taxa_erro.round(4)
    por_bloco["escore_medio"] = por_bloco.escore_medio.round(4)

    faixas = pd.qcut(d.hand_m, q=min(5, d.hand_m.nunique()), duplicates="drop")
    por_hand = (d.assign(faixa=faixas.astype(str))
                .groupby("faixa")
                .agg(n=("y", "size"), prevalencia=("y", "mean"),
                     taxa_erro=("erro", "mean"),
                     escore_medio=("escore", "mean"))
                .reset_index().round(4))
    return pontos, por_bloco, por_hand


# ------------------------------------------------------------- 7. hipoteses

def _veredito_h1(drift: pd.DataFrame, disc: dict) -> dict:
    substanciais = drift[drift.smd.abs() >= 0.25].feature.tolist()
    moderadas = drift[(drift.smd.abs() >= 0.1) & (drift.smd.abs() < 0.25)].feature.tolist()
    auc = disc.get("auc")
    if auc is None:
        return {"veredito": INCONCLUSIVE,
                "por_que": "discriminador de periodo nao pode ser ajustado"}
    if auc >= 0.6 or substanciais:
        return {"veredito": SUPPORTED,
                "por_que": (f"discriminador de periodo com AUC {auc}; "
                            f"features com SMD substancial: {substanciais or 'nenhuma'}; "
                            f"moderada: {moderadas or 'nenhuma'}"),
                "evidencia": ["feature_drift.csv", "hypotheses.json"]}
    return {"veredito": NOT_SUPPORTED,
            "por_que": (f"discriminador de periodo com AUC {auc} e nenhuma "
                        "feature com SMD substancial: os periodos nao sao "
                        "separaveis no espaco de features"),
            "evidencia": ["feature_drift.csv"]}


def _veredito_h2(por_bloco: pd.DataFrame) -> dict:
    if por_bloco.empty or len(por_bloco) < 3:
        return {"veredito": INCONCLUSIVE,
                "por_que": "blocos espaciais insuficientes no periodo de teste"}
    grandes = por_bloco[por_bloco.n >= 10]
    if len(grandes) < 3:
        return {"veredito": INCONCLUSIVE,
                "por_que": "menos de tres blocos com amostra suficiente"}
    desvio = float(grandes.taxa_erro.std())
    media = float(grandes.taxa_erro.mean())
    concentrado = desvio > 0.15 and grandes.taxa_erro.max() > media + 2 * desvio
    return {"veredito": SUPPORTED if concentrado else NOT_SUPPORTED,
            "por_que": (f"taxa de erro por bloco: media {media:.3f}, desvio "
                        f"{desvio:.3f}, maximo {grandes.taxa_erro.max():.3f} "
                        f"em {len(grandes)} blocos com n>=10"),
            "evidencia": ["erro_por_bloco.csv"]}


def _veredito_h4(efeitos: pd.DataFrame) -> dict:
    """H4 fala em sinal OU MAGNITUDE, e o primeiro corte deste avaliador so
    testava sinal.

    O defeito nao era teorico. `rain_decay_index` -- o efeito mais forte e
    mais consistente do conjunto -- vai de +1,03 (IC [0,52; 1,70]) em 2023 a
    -0,007 (IC [-0,37; 0,34]) em 2026. O SINAL nao inverte com IC firme dos
    dois lados, entao a checagem antiga devolvia `not_supported` para o achado
    mais importante da autopsia: o preditor dominante DESAPARECEU.

    Duas formas de mudanca contam, e as duas exigem IC:

      INVERSAO       sinais opostos, com IC firme em ambos os periodos.
      MAGNITUDE      ICs DISJUNTOS entre dois periodos. Nao se sobrepor e a
                     forma honesta de dizer "distinguivel"; comparar
                     coeficientes pontuais diria que tudo muda sempre.
    """
    validos = efeitos.dropna(subset=["coef", "ic95_lo", "ic95_hi"])
    if validos.empty:
        return {"veredito": INCONCLUSIVE, "por_que": "sem efeitos estimaveis"}

    inversoes, magnitudes = [], []
    for f, g in validos.groupby("feature"):
        firmes = g[g.ic_cruza_zero == False]  # noqa: E712
        if len(set(firmes.direcao_estimada.tolist())) > 1:
            inversoes.append({"feature": f, "anos": firmes.ano.tolist(),
                              "sinais": firmes.direcao_estimada.tolist()})
        linhas = g.sort_values("ano").to_dict("records")
        for i in range(len(linhas)):
            for j in range(i + 1, len(linhas)):
                a, b = linhas[i], linhas[j]
                if a["ic95_hi"] < b["ic95_lo"] or b["ic95_hi"] < a["ic95_lo"]:
                    magnitudes.append({
                        "feature": f,
                        "de": {"ano": int(a["ano"]), "coef": a["coef"],
                               "ic95": [a["ic95_lo"], a["ic95_hi"]]},
                        "para": {"ano": int(b["ano"]), "coef": b["coef"],
                                 "ic95": [b["ic95_lo"], b["ic95_hi"]]},
                    })

    if inversoes or magnitudes:
        partes = []
        if inversoes:
            partes.append(f"{len(inversoes)} variavel(is) invertem o sinal com "
                          "IC firme nos dois periodos")
        if magnitudes:
            alvos = sorted({m["feature"] for m in magnitudes})
            partes.append(f"{len(magnitudes)} par(es) de periodos com IC "
                          f"DISJUNTO em {', '.join(alvos)}")
        return {"veredito": SUPPORTED,
                "por_que": "; ".join(partes),
                "inversoes_de_sinal": inversoes,
                "mudancas_de_magnitude": magnitudes,
                "evidencia": ["efeitos_por_periodo.csv"]}

    n_firmes = int((validos.ic_cruza_zero == False).sum())  # noqa: E712
    return {"veredito": NOT_SUPPORTED if n_firmes >= 3 else INCONCLUSIVE,
            "por_que": (f"nenhuma inversao de sinal nem IC disjunto entre "
                        f"periodos; {n_firmes} de {len(validos)} efeitos tem "
                        "IC que nao cruza zero"),
            "evidencia": ["efeitos_por_periodo.csv"]}


def _veredito_h5(holdout: dict, grouped: dict) -> dict:
    auc_h = holdout.get("auc")
    auc_g = grouped.get("auc")
    slope = holdout.get("calibration_slope")
    if auc_h is None or auc_g is None:
        return {"veredito": INCONCLUSIVE, "por_que": "holdout sem AUC estimavel"}
    ranking_caiu = (auc_g - auc_h) > 0.05
    calib_ruim = slope is not None and abs(slope - 1.0) > 0.3
    if ranking_caiu:
        return {"veredito": NOT_SUPPORTED,
                "por_que": (f"a AUC caiu de {auc_g} (embaralhado) para {auc_h} "
                            f"(temporal): o ranking degradou, e nao so a escala"),
                "evidencia": ["metrics.csv"]}
    if calib_ruim:
        return {"veredito": SUPPORTED,
                "por_que": (f"AUC preservada ({auc_h} vs {auc_g}) com "
                            f"calibration_slope {slope}"),
                "evidencia": ["metrics.csv"]}
    return {"veredito": INCONCLUSIVE,
            "por_que": f"AUC {auc_h} e slope {slope} nao separam as duas causas"}


def _veredito_h6(com_chuva: dict, sem_chuva: dict) -> dict:
    a, b = com_chuva.get("auc"), sem_chuva.get("auc")
    if a is None or b is None:
        return {"veredito": INCONCLUSIVE, "por_que": "uma das variantes falhou"}
    delta = b - a
    if delta > 0.05:
        return {"veredito": SUPPORTED,
                "por_que": (f"remover a chuva MELHORA o holdout temporal de "
                            f"{a} para {b} (delta {delta:+.4f}): as features "
                            "de chuva atrapalham a generalizacao temporal"),
                "evidencia": ["metrics.csv"]}
    if abs(delta) <= 0.05:
        return {"veredito": NOT_SUPPORTED,
                "por_que": (f"remover a chuva move o holdout de {a} para {b} "
                            f"(delta {delta:+.4f}), dentro do ruido"),
                "evidencia": ["metrics.csv"]}
    return {"veredito": NOT_SUPPORTED,
            "por_que": (f"remover a chuva PIORA o holdout de {a} para {b} "
                        f"(delta {delta:+.4f}): a chuva contribui")}


def _veredito_h7(por_hand: pd.DataFrame, por_evento: pd.DataFrame) -> dict:
    uteis = []
    if not por_evento.empty and "auc" in por_evento:
        uteis = por_evento.dropna(subset=["auc"]).query("auc >= 0.65").evento.tolist()
    faixas_ruins = 0
    if not por_hand.empty:
        faixas_ruins = int((por_hand.taxa_erro > 0.4).sum())
    if uteis:
        return {"veredito": SUPPORTED,
                "por_que": (f"{len(uteis)} evento(s) do periodo de teste mantem "
                            f"AUC >= 0,65: {uteis[:5]}"),
                "evidencia": ["desempenho_por_evento.csv"]}
    return {"veredito": NOT_SUPPORTED if not por_hand.empty else INCONCLUSIVE,
            "por_que": (f"nenhum evento do periodo de teste mantem AUC >= 0,65; "
                        f"{faixas_ruins} faixa(s) de HAND com taxa de erro > 0,4"),
            "evidencia": ["desempenho_por_evento.csv", "erro_por_faixa_hand.csv"]}


def executar(saida: Path = SAIDA) -> dict:
    from harness_comparacao import (
        ChallengerFirth, ChallengerGAMSpline, ChallengerGBMMonotonico,
    )

    p = protocolo()
    c = mv.carregar("curitiba_autopsia")
    saida.mkdir(parents=True, exist_ok=True)

    dist = distribuicao_temporal(c)
    drift = drift_de_covariaveis(c)
    disc = discriminador_de_periodo(c)
    efeitos = efeitos_por_periodo(c)

    firth = ChallengerFirth()
    por_evento = desempenho_por_evento(c, firth)
    pontos, por_bloco, por_hand = cartografia_do_erro(c, firth)

    # H3 e H6 precisam de avaliacoes sob a MESMA matriz, e nao de numeros
    # calculados a parte -- senao a comparacao seria entre dois desenhos.
    resultados = []
    for modelo in (firth, ChallengerGAMSpline(), ChallengerGBMMonotonico(
            [int(np.sign(CRITERIOS_FISICOS["sinais_esperados"][f]))
             for f in c.features])):
        for esquema, kw in ((mv.GROUPED, {}), (mv.HOLDOUT, {"ano_corte": ANO_TESTE}),
                            (mv.FORWARD, {"anos_teste": [2024, 2025, 2026]}),
                            (mv.SPATIAL, {}), (mv.LOEO, {})):
            resultados.append(mv.avaliar_modelo(modelo, c, esquema, p, **kw))

    # H6: mesmo conjunto, mesma matriz, so sem as features de chuva.
    sem_chuva_feats = [f for f in c.features if not f.startswith("rain")]
    c_sem = mv.Conjunto(nome="curitiba_autopsia_sem_chuva", dados=c.dados,
                        features=sem_chuva_feats, grupos=c.grupos, y=c.y,
                        datas=c.datas, hash_arquivo=c.hash_arquivo)
    res_sem = mv.avaliar_modelo(firth, c_sem, mv.HOLDOUT, p, ano_corte=ANO_TESTE)
    resultados.append(res_sem)

    def _achar(modelo_nome, esquema, dataset="curitiba_autopsia"):
        for r in resultados:
            if (r["modelo"] == modelo_nome and r["esquema"] == esquema
                    and r["dataset"] == dataset):
                return r
        return {}

    h_grouped = _achar("firth_servido", mv.GROUPED)
    h_holdout = _achar("firth_servido", mv.HOLDOUT)
    gam_holdout = _achar("gam_spline_aditivo", mv.HOLDOUT)
    gbm_holdout = _achar("gbm_monotonico", mv.HOLDOUT)

    melhor_nao_linear = max(
        [x for x in (gam_holdout.get("auc"), gbm_holdout.get("auc"))
         if x is not None], default=None)
    if melhor_nao_linear is None or h_holdout.get("auc") is None:
        h3 = {"veredito": INCONCLUSIVE, "por_que": "holdout sem AUC estimavel"}
    elif melhor_nao_linear - h_holdout["auc"] > 0.05:
        h3 = {"veredito": SUPPORTED,
              "por_que": (f"o melhor nao linear atinge {melhor_nao_linear} no "
                          f"holdout temporal contra {h_holdout['auc']} do linear"),
              "evidencia": ["metrics.csv"]}
    else:
        h3 = {"veredito": NOT_SUPPORTED,
              "por_que": (f"o melhor nao linear atinge {melhor_nao_linear} no "
                          f"holdout temporal contra {h_holdout['auc']} do "
                          "linear: a nao linearidade nao recupera a "
                          "generalizacao temporal"),
              "evidencia": ["metrics.csv"]}

    hipoteses = {
        "H1": {"enunciado": p["hipoteses"][0]["enunciado"],
               **_veredito_h1(drift, disc)},
        "H2": {"enunciado": p["hipoteses"][1]["enunciado"],
               **_veredito_h2(por_bloco)},
        "H3": {"enunciado": p["hipoteses"][2]["enunciado"], **h3},
        "H4": {"enunciado": p["hipoteses"][3]["enunciado"],
               **_veredito_h4(efeitos)},
        "H5": {"enunciado": p["hipoteses"][4]["enunciado"],
               **_veredito_h5(h_holdout, h_grouped)},
        "H6": {"enunciado": p["hipoteses"][5]["enunciado"],
               **_veredito_h6(h_holdout, res_sem)},
        "H7": {"enunciado": p["hipoteses"][6]["enunciado"],
               **_veredito_h7(por_hand, por_evento)},
    }

    dist.to_csv(saida / "curitiba_distribuicao_temporal.csv", index=False)
    drift.to_csv(saida / "feature_drift.csv", index=False)
    efeitos.to_csv(saida / "efeitos_por_periodo.csv", index=False)
    por_evento.to_csv(saida / "desempenho_por_evento.csv", index=False)
    por_bloco.to_csv(saida / "erro_por_bloco.csv", index=False)
    por_hand.to_csv(saida / "erro_por_faixa_hand.csv", index=False)
    pontos.to_csv(saida / "erro_por_ponto.csv", index=False)
    mv.linhas_para_csv(resultados).to_csv(
        saida / "curitiba_metrics.csv", index=False)
    mv.folds_para_csv(resultados).to_csv(
        saida / "curitiba_folds.csv", index=False)

    # A unidade "evento" NAO sustenta avaliacao em Curitiba, e isso e um
    # resultado: das 126 datas do periodo de teste, nenhuma tem as duas
    # classes com amostra suficiente. Um LOEO rodado assim mede os poucos
    # folds sobreviventes, com prevalencia de 0,93 -- e apresentar aquela AUC
    # como "generalizacao para evento novo" seria descrever outra coisa.
    eventos_avaliaveis = int(por_evento.auc.notna().sum()) if not por_evento.empty else 0
    limitacoes_estruturais = [{
        "o_que": "avaliacao por evento em Curitiba",
        "estado": "bloqueado",
        "por_que": (f"{len(por_evento)} datas no periodo de teste, "
                    f"{eventos_avaliaveis} com as duas classes e amostra "
                    "suficiente; a data agrupa 4 a 43 pontos, quase sempre "
                    "todos positivos"),
        "consequencia": ("leave_one_event_out sobrevive com poucos folds e "
                         "prevalencia de teste ~0,93; a AUC resultante nao "
                         "responde 'generaliza para evento novo'"),
        "o_que_falta": ("negativos datados no mesmo evento, ou uma unidade de "
                        "evento mais grossa que a data"),
    }]

    relatorio = {
        "tipo": "autopsia_temporal_curitiba",
        "limitacoes_estruturais": limitacoes_estruturais,
        "quando_utc": agora_utc(),
        "hash_do_protocolo": p["hash_do_protocolo"],
        "dataset_hash": c.hash_arquivo,
        "conjunto": c.extra,
        "ano_de_teste": ANO_TESTE,
        "discriminador_de_periodo": disc,
        "hipoteses": hipoteses,
        "resultados": resultados,
        "hash_da_matriz": mv.hash_da_matriz(resultados),
        "nao_promove": ("Curitiba continua servida por transferencia; nada "
                        "aqui treina modelo local nem muda maturidade"),
    }
    (saida / "hypotheses.json").write_text(
        json.dumps({"hash_do_protocolo": p["hash_do_protocolo"],
                    "hipoteses": hipoteses}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (saida / "curitiba_autopsia.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    return relatorio


def main() -> int:
    r = executar()
    print(json.dumps({
        "hash_do_protocolo": r["hash_do_protocolo"],
        "conjunto": r["conjunto"],
        "discriminador_auc": r["discriminador_de_periodo"].get("auc"),
        "hipoteses": {k: v["veredito"] for k, v in r["hipoteses"].items()},
        "saida": str(SAIDA),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
