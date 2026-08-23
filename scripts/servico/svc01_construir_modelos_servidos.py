"""
SVC-01 -- Constroi os modelos que o contrato de inferencia serve.

POR QUE ESTE ARQUIVO EXISTE:

o `revp_contrato_inferencia_v0_revalidacao_cientifica.md` (23/07/2026) terminou
com duas pendencias declaradas como bloqueadoras de qualquer codigo de API:

  (a) a semantica do `confidence_interval` por escore -- bootstrap preditivo ou
      delta method -- "precisa ser tomada antes de qualquer codigo de API, nao
      depois";
  (b) o orcamento de EPV do teste A/B com DINO.

A (b) ja esta resolvida por decisao anterior do projeto: o DINO foi descartado
como variavel na Fase 1 e vive so como evidencia visual, entao o teste A/B nao e
pre-requisito de nada aqui.

A (a) e resolvida NESTE arquivo, e a decisao esta escrita abaixo para nao virar
escolha silenciosa.

-------------------------------------------------------------------------
DECISAO: o IC do escore e bootstrap do preditor linear, reamostrando GRUPOS
-------------------------------------------------------------------------

O que se faz: reamostram-se os GRUPOS do conjunto de ajuste com reposicao, N
vezes; para cada reamostra o Firth e reajustado; guardam-se as N replicas de
coeficientes. O IC de um escore e o percentil das N projecoes daquele ponto.

Por que nao o delta method: ele depende da aproximacao assintotica do
erro-padrao, que e discutivelmente fragil exatamente no regime de n pequeno
que motivou usar Firth. Usar Firth por causa do n pequeno e depois propagar
incerteza por aproximacao assintotica seria incoerente.

Por que reamostrar GRUPO e nao linha: e a regra U2 do projeto
(`ext_uk_adjudicacao_negativo_v1.md`). Reamostrar linha infla a precisao por
pseudo-replicacao, e o IC sairia estreito demais -- que e o modo de falha mais
perigoso num numero que vai para dentro de uma resposta de API.

Custo: o reajuste custa ~0,2 s no maior conjunto, entao N=1000 custa ~200 s por
modelo. Isso acontece AQUI, uma vez, e nao por requisicao: o artefato servido
guarda as replicas de coeficiente, e a requisicao so projeta. Servir e
multiplicacao de matriz.

-------------------------------------------------------------------------
QUAIS MODELOS SAO SERVIDOS, e por que cada um
-------------------------------------------------------------------------

`recife_pluvial`   Recife e PLUVIAL_URBANO: a agua vem da chuva excedendo a
                   drenagem, nao do canal subindo. Modelo proprio, com as seis
                   variaveis, como em `mod-pluv-01`.

`fluvial_planicie` Ajustado no estrato PLANO_OU_ONDULADO **excluindo a regiao
                   alvo brasileira**: entram so as fontes com evidencia real
                   (CEMS, Sen1Floods11, piloto ingles). Curitiba fica fora do
                   ajuste por decisao de metodo, nao por falta de dado -- e a
                   regra de `metodo_aplicacao_sem_rotulo_local_v1.md`: a regiao
                   alvo nao entra no treino em nenhuma linha.

`fluvial_serra`    Estrato INGREME, com o numero de variaveis que o orcamento
                   de EPV da classe minoritaria permite. Serve as regioes de
                   serra -- e e o modelo que Petropolis usaria se algum dia
                   tiver variaveis fisicas extraidas.

-------------------------------------------------------------------------
O QUE O ARTEFATO SERVIDO GUARDA, e por que
-------------------------------------------------------------------------

coeficientes e padronizacao   sem `mu`/`sd` do treino o escore de uma requisicao
                              nova nao e comparavel ao do ajuste.
replicas de bootstrap         para o IC por escore, sem reajustar nada em tempo
                              de requisicao.
faixa 5-95% de cada variavel  para o portao de dominio: o contrato precisa saber
                              se a requisicao esta dentro do que o modelo viu.
                              E o achado de `metodo_aplicacao_sem_rotulo_local`
                              virando regra executavel em vez de recomendacao.
desempenho e criterios        para o `model_card`: AUC agrupada, sinais
                              conferidos, o que falhou.
prevalencia do ajuste         um escore de regressao logistica herda a
                              prevalencia do conjunto de treino; declarar e o
                              minimo para quem le o numero.

NAO FAZ: nao serve nada, nao decide maturidade de regiao (isso e do SVC-02),
nao promove rotulo e nao autoriza uso preditivo.

Uso:
    python scripts/servico/svc01_construir_modelos_servidos.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))
import susc_firth_adaptador  # noqa: F401,E402
from ds03_esquema_alvo import VERSAO  # noqa: E402

RUNS = REPO / "local_runs"
POOL = RUNS / "ds-05-tabela-unica" / f"tabela_unica_pool_fluvial_{VERSAO}.csv"
UNICA = RUNS / "ds-05-tabela-unica" / f"tabela_unica_{VERSAO}.csv"
OUT = RUNS / "svc-01-modelos"

TOPO = ["elevation_m", "slope_deg", "hand_m", "twi_dinf"]
CHUVA = ["rain_max_24h", "rain_decay_index"]
CAUSAIS = ["hand_m", "twi_dinf"]
SINAL_EXIGIDO = {"hand_m": -1, "twi_dinf": +1}
NEGATIVO_ADMITIDO = ("observado", "exclusao_qualificada", "nao_aplicavel")

EPV_MINIMO = 10
N_BOOT = 1000
SEMENTE = 20260820
AUC_MIN, AUC_MAX = 0.70, 0.88

# Regioes-alvo brasileiras: nunca entram no ajuste do modelo que as serve.
REGIOES_ALVO = ("curitiba", "recife")


def ajustar(X: np.ndarray, y: np.ndarray):
    from firthlogist import FirthLogisticRegression

    m = FirthLogisticRegression(max_iter=500, skip_pvals=True, skip_ci=True)
    m.fit(X, y)
    return m


def padronizar(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = X.mean(0), X.std(0)
    return mu, np.where(sd == 0, 1, sd)


def orcamento_estrito(s: pd.DataFrame) -> int:
    """Quantas variaveis o conjunto comporta: grupos da classe rara / 10."""
    por_classe = s.groupby("classe").grupo_cv.nunique()
    if len(por_classe) < 2:
        return 0
    return int(por_classe.min()) // EPV_MINIMO


def auc_agrupada(s: pd.DataFrame, feats: list[str]) -> dict:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    X = s[feats].to_numpy(dtype=float)
    y = s.classe.to_numpy().astype(int)
    g = s.grupo_cv.to_numpy()
    n_folds = min(5, len(np.unique(g)))
    if n_folds < 2:
        return {"auc_cv": None, "motivo": "grupos insuficientes"}
    p = np.full(len(y), np.nan)
    for i_tr, i_te in GroupKFold(n_splits=n_folds).split(X, y, g):
        if len(np.unique(y[i_tr])) < 2 or len(np.unique(y[i_te])) < 2:
            continue
        mu, sd = padronizar(X[i_tr])
        m = ajustar((X[i_tr] - mu) / sd, y[i_tr])
        p[i_te] = m.predict_proba((X[i_te] - mu) / sd)[:, 1]
    ok = ~np.isnan(p)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return {"auc_cv": None, "motivo": "nenhum fold com as duas classes"}
    return {"auc_cv": round(float(roc_auc_score(y[ok], p[ok])), 4),
            "folds": n_folds, "cobertura": round(float(ok.mean()), 3)}


def construir(nome: str, s: pd.DataFrame, feats: list[str], meta: dict) -> dict:
    print(f"\n=== {nome} ===")
    print(f"  n={len(s):,} grupos={s.grupo_cv.nunique():,} "
          f"variaveis={feats}")
    orc = orcamento_estrito(s)
    if len(feats) > orc:
        print(f"  ABORTADO: {len(feats)} variaveis acima do orcamento de EPV ({orc})")
        return {"nome": nome, "servivel": False,
                "motivo": f"{len(feats)} variaveis acima do orcamento de EPV ({orc})",
                "orcamento_estrito": orc, "features": feats, **meta}

    X = s[feats].to_numpy(dtype=float)
    y = s.classe.to_numpy().astype(int)
    mu, sd = padronizar(X)
    m = ajustar((X - mu) / sd, y)
    coef = [float(c) for c in m.coef_.ravel()[:len(feats)]]
    intercepto = float(np.ravel(m.intercept_)[0])

    rng = np.random.default_rng(SEMENTE)
    grupos = s.grupo_cv.to_numpy()
    unicos = np.unique(grupos)
    idx = {g: np.flatnonzero(grupos == g) for g in unicos}
    replicas, t0 = [], time.time()
    for k in range(N_BOOT):
        esc = rng.choice(unicos, size=len(unicos), replace=True)
        i = np.concatenate([idx[g] for g in esc])
        yb = y[i]
        if len(np.unique(yb)) < 2:
            continue
        try:
            mb = ajustar((X[i] - mu) / sd, yb)
        except Exception:
            continue
        replicas.append([float(np.ravel(mb.intercept_)[0])]
                        + [float(c) for c in mb.coef_.ravel()[:len(feats)]])
        if (k + 1) % 250 == 0:
            print(f"    bootstrap {k + 1}/{N_BOOT} ({time.time() - t0:.0f}s)")

    arr = np.array(replicas)
    ic_coef = {f: [round(float(v), 4) for v in np.percentile(arr[:, j + 1], [2.5, 97.5])]
               for j, f in enumerate(feats)} if len(arr) >= 100 else {}

    desempenho = auc_agrupada(s, feats)
    falhas = []
    for f, esperado in SINAL_EXIGIDO.items():
        if f not in feats:
            continue
        c = coef[feats.index(f)]
        if np.sign(c) != esperado:
            falhas.append(f"sinal de {f} e {c:+.4f}")
        if f in ic_coef and ic_coef[f][0] <= 0 <= ic_coef[f][1]:
            falhas.append(f"IC95 de {f} cruza zero")
    auc = desempenho.get("auc_cv")
    if auc is not None and not (AUC_MIN <= auc <= AUC_MAX):
        falhas.append(f"AUC {auc} fora da faixa [{AUC_MIN}; {AUC_MAX}]")

    faixa = {f: [round(float(np.percentile(X[:, j], 5)), 4),
                 round(float(np.percentile(X[:, j], 95)), 4)]
             for j, f in enumerate(feats)}

    print(f"  AUC agrupada={auc}  coef={dict(zip(feats, [round(c, 4) for c in coef]))}")
    print(f"  bootstrap: {len(arr)} replicas validas de {N_BOOT} "
          f"({time.time() - t0:.0f}s)")
    print(f"  falhas={falhas if falhas else 'nenhuma'}")

    return {
        "nome": nome, "servivel": True, "features": feats,
        "intercepto": round(intercepto, 6),
        "coef": {f: round(c, 6) for f, c in zip(feats, coef)},
        "ic95_coef": ic_coef,
        "padronizacao": {f: {"media": round(float(mu[j]), 6),
                             "desvio": round(float(sd[j]), 6)}
                         for j, f in enumerate(feats)},
        "faixa_dominio_5_95": faixa,
        "replicas_bootstrap": [[round(v, 6) for v in linha] for linha in arr.tolist()],
        "n_replicas": int(len(arr)),
        "desempenho": desempenho,
        "orcamento_estrito": orc,
        "prevalencia_do_ajuste": round(float(y.mean()), 4),
        "n": int(len(s)), "grupos": int(s.grupo_cv.nunique()),
        "grupos_positivos": int(s.loc[s.classe == 1, "grupo_cv"].nunique()),
        "grupos_negativos": int(s.loc[s.classe == 0, "grupo_cv"].nunique()),
        "fontes_do_ajuste": s.fonte.value_counts().to_dict(),
        "niveis_de_negativo": s.loc[s.classe == 0, "nivel_negativo"]
                               .value_counts().to_dict(),
        "falhas_de_criterio": falhas,
        "veredito": "COERENTE_COM_CRITERIOS" if not falhas else "FORA_DOS_CRITERIOS",
        **meta,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===SVC-01: CONSTRUCAO DOS MODELOS SERVIDOS===")

    for p in (POOL, UNICA):
        if not p.exists():
            print(f"ABORTADO: {p} ausente. Gere com: "
                  "python scripts/suscetibilidade/ds05_admissao_consolidacao.py")
            return 1

    pool = pd.read_csv(POOL, low_memory=False)
    pool = pool[pool.classe.isin([0, 1])
                & pool.nivel_negativo.isin(NEGATIVO_ADMITIDO)].copy()
    unica = pd.read_csv(UNICA, low_memory=False)
    unica = unica[(unica.admitido == True) & unica.classe.isin([0, 1])].copy()  # noqa: E712

    modelos = []

    # 1. Recife -- pluvial urbano, modelo proprio
    rec = unica[unica.fonte == "recife"].dropna(subset=TOPO + CHUVA + ["grupo_cv"])
    modelos.append(construir(
        "recife_pluvial", rec, TOPO + CHUVA,
        {"mecanismo": "PLUVIAL_URBANO",
         "regioes_servidas": ["recife"],
         "por_que_este_conjunto":
             "Recife tem modelo proprio porque o mecanismo e outro: a agua vem "
             "da chuva excedendo a drenagem, nao do canal subindo. Somar Recife "
             "ao conjunto fluvial assumiria um processo gerador unico",
         "limitacoes_declaradas": [
             "negativo por ausencia de registro, o nivel mais fraco da hierarquia",
             "positivos e negativos dividem 5 das 205 datas: a chuva separa dias, "
             "nao lugares (ext_chuva_estado_do_projeto_v1.md)",
             "HAND nao separa as classes em Recife -- num evento pluvial HAND nao "
             "e o mecanismo",
         ]}))

    # 2. Planicie fluvial -- sem nenhuma linha da regiao alvo
    pla = pool[(pool.classe_relevo == "PLANO_OU_ONDULADO")
               & (~pool.fonte.isin(REGIOES_ALVO))].dropna(
                   subset=TOPO + ["grupo_cv"])
    modelos.append(construir(
        "fluvial_planicie", pla, TOPO,
        {"mecanismo": "FLUVIAL_ENXURRADA",
         "classe_relevo": "PLANO_OU_ONDULADO",
         "regioes_servidas": ["curitiba"],
         "por_que_este_conjunto":
             "a regiao alvo nao entra no treino em nenhuma linha: o proposito e "
             "prever onde nao ha inventario local, e validar contra o rotulo da "
             "propria regiao seria a pergunta errada "
             "(metodo_aplicacao_sem_rotulo_local_v1.md)",
         "limitacoes_declaradas": [
             "elevation_m e altitude absoluta, nao grandeza causal comparavel "
             "entre cidades de altitude de base diferente -- o portao de dominio "
             "do SVC-02 mede isso por requisicao",
             "chuva fora do conjunto: nao discrimina na escala em que o modelo "
             "compara (ext_chuva_estado_do_projeto_v1.md)",
         ]}))

    # 3. Serra fluvial -- orcamento de EPV manda no numero de variaveis
    ser = pool[(pool.classe_relevo == "INGREME")
               & (~pool.fonte.isin(REGIOES_ALVO))].dropna(
                   subset=TOPO + ["grupo_cv"])
    orc_ser = orcamento_estrito(ser)
    feats_ser = (CAUSAIS if orc_ser >= 2 else ["hand_m"])[:max(orc_ser, 1)]
    modelos.append(construir(
        "fluvial_serra", ser, feats_ser,
        {"mecanismo": "FLUVIAL_ENXURRADA",
         "classe_relevo": "INGREME",
         "regioes_servidas": ["petropolis"],
         "por_que_petropolis_e_servida":
             "decisao tomada em 20/08/2026, ao executar o E5. Dois documentos "
             "divergiam: o esboco de telas declarava Petropolis como "
             "region_not_supported, e ext_criterios_de_acerto_v1.md secao 6 dizia "
             "que 'para PREDIZER em Petropolis nao falta nada; o que falta e a "
             "validacao'. O proprio E5 resolve: ele manda levar o modelo as TRES "
             "regioes e caracterizar a distancia de dominio, com a evidencia de "
             "que nao se afirme acerto onde nao ha inventario local. Servir com "
             "maturidade `transferencia_sem_referencia_local` e limitacoes "
             "declaradas atende as duas coisas; recusar impediria o E5",
         "por_que_este_conjunto":
             f"o estrato ingreme tem {ser.loc[ser.classe == 1, 'grupo_cv'].nunique()} "
             "grupos positivos; o orcamento de EPV da classe minoritaria decide "
             "quantas variaveis cabem, e nao o desempenho",
         "limitacoes_declaradas": [
             "nenhuma AOI brasileira no ajuste: os grupos ingremes sao todos "
             "estrangeiros",
             "Petropolis nao tem nenhuma linha na tabela unica: nenhum ponto "
             "rotulado, nenhum inventario local, nenhuma validacao possivel hoje",
             "o escore para Petropolis e predicao por semelhanca de terreno, "
             "nunca afirmacao de acerto",
         ]}))

    for m in modelos:
        (OUT / f"{m['nome']}.json").write_text(
            json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")

    indice = {
        "gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "semente": SEMENTE, "n_boot": N_BOOT,
        "decisao_do_ic": ("bootstrap do preditor linear reamostrando grupos; "
                          "resolve a pendencia declarada em "
                          "revp_contrato_inferencia_v0_revalidacao_cientifica.md "
                          "secao 3.5"),
        "modelos": {m["nome"]: {
            "servivel": m["servivel"],
            "features": m["features"],
            "n": m.get("n"), "grupos": m.get("grupos"),
            "auc_cv": m.get("desempenho", {}).get("auc_cv"),
            "veredito": m.get("veredito", "NAO_SERVIVEL"),
            "regioes_servidas": m.get("regioes_servidas", []),
        } for m in modelos},
        "segundos": round(time.time() - t0, 1),
        "nao_e": ("nao e validacao operacional: e o artefato que o contrato "
                  "de inferencia serve, com limitacoes declaradas por modelo"),
    }
    (OUT / "indice.json").write_text(
        json.dumps(indice, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/servico/svc01_construir_modelos_servidos.py\n"
        "python scripts/servico/svc02_contrato_inferencia.py --demonstracao\n"
        "python -m pytest tests/test_svc_contrato_inferencia.py -q\n",
        encoding="utf-8")

    print(f"\n--- INDICE ---")
    for nome, d in indice["modelos"].items():
        print(f"  {nome:20s} servivel={d['servivel']} auc={d['auc_cv']} "
              f"veredito={d['veredito']}")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
