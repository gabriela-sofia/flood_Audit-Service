"""
MOD-PROSP-02 -- Holdout temporal (E4/M3) sobre a TABELA UNICA.

POR QUE ESTE SCRIPT EXISTE, SE O MOD-PROSP-01 JA RODOU:
o MOD-PROSP-01 (12/08/2026) respondeu a pergunta e deu
`PROSPECTIVAMENTE_ESTAVEL`. Ele nao e descartado -- e o precedente que este
script replica. Mas ele tem tres pendencias que impedem de encerrar o E4:

  1. RODOU NA BASE ERRADA. Le `ds-01-multirregiao`, montada antes da
     harmonizacao. A base congelada do E2/M1 e `ds-05-tabela-unica`
     (16/08/2026). Na ds-05 o piloto ingles tem 401 grupos, nao 328.

  2. A DATA DO NEGATIVO E ARTEFATO. Os grupos sao puros: 201 eventos
     so-positivos (`EV_*`) e 200 blocos espaciais de 5 km so-negativos
     (`NEG_r_c`). O bloco negativo nao tem data de evento: ele carrega as
     datas dos positivos para os quais foi amostrado -- mediana de 11,5
     datas distintas por bloco, ate 35. Ordenar bloco por data minima empurra
     quase todo negativo para os cortes antigos, e a prevalencia do teste
     sobe de 0,297 para 0,933 ao longo dos folds do MOD-PROSP-01.

  3. NAO TEM INTERVALO DE CONFIANCA. A regra U2 de
     `ext_uk_adjudicacao_negativo_v1.md` exige IC por reamostragem NO NIVEL
     DO GRUPO em todo desempenho reportado. O MOD-PROSP-01 reporta AUC nua.
     Um dos folds dele mede AUC sobre 31 negativos.

O QUE ESTE SCRIPT FAZ:
roda o mesmo desenho de janela expansiva na base congelada, em DUAS variantes
declaradas ANTES de olhar qualquer resultado, com IC de grupos em todo fold.

-------------------------------------------------------------------------
DESENHO DECLARADO ANTES DA EXECUCAO (nada aqui pode ser revisto depois de
ver numero -- revisar a posteriori seria decisao pos-hoc)
-------------------------------------------------------------------------

BASE: o pool fluvial da tabela unica, na versao que o contrato do ds03
declara (`VERSAO`) -- nao um nome fixo: quando o esquema avanca, o script
segue junto em vez de ler silenciosamente a tabela anterior.

ESTRATO PRIMARIO: `fonte == "uk"`, escolhido por COBERTURA DE CALENDARIO --
criterio que nao depende de nenhum resultado: o piloto ingles cobre 21 anos
(2000-2025) e as demais fontes cobrem 3 ou 4. Um holdout temporal precisa de
horizonte, e so ele tem.

ESTRATOS SECUNDARIOS: as demais fontes rodam o mesmo desenho e sao
reportadas com a limitacao medida ao lado. Nao entram como resposta do E4;
entram para que "so o UK sustenta o teste" seja um numero conferivel em
`viabilidade_por_fonte.csv` e nao uma afirmacao. Sen1Floods11 nao sustenta
nenhum fold (11 grupos em 11 datas); Curitiba sustenta folds pela contagem
mas tem 114 negativos contra 1.238 positivos, com quase um grupo por ponto.

POR QUE NAO O POOL INTEIRO: no pool, corte temporal e corte de fonte. Um
corte em 2020 poe Sen1Floods11+UK no treino e CEMS+UK+Curitiba no teste, e
faz o estrato INGREME saltar de 775 para 5.762 pontos. Isso mede
transferencia entre fontes, nao estabilidade temporal. O script grava a
matriz ano x fonte em `confusao_fonte_periodo.csv` para que a afirmacao seja
conferivel.

CONJUNTOS DE VARIAVEIS: TERRENO (4) e COMPLETO (6, com chuva). Os dois
rodam sempre; nenhum e escolhido pelo resultado.

VARIANTE `heranca` -- replicacao literal do MOD-PROSP-01 na base nova.
  Data do grupo = menor data dos seus pontos. Todos os 401 grupos entram na
  mesma ordenacao temporal. Serve para responder "o resultado de 12/08
  sobrevive a troca de base?" e nada alem disso -- ela herda o defeito 2.

VARIANTE `bloco` -- corrige o defeito 2.
  O eixo temporal se aplica so ao POSITIVO, que e quem tem data de evento.
  Os 200 blocos negativos sao espaciais: entram por sorteio deterministico
  (semente declarada), sem que nenhum bloco apareca dos dois lados do mesmo
  fold, e o teste recebe blocos ate aproximar 1:1 dos positivos do fold.
  Justificativa fisica: suscetibilidade do negativo e propriedade do
  terreno, nao da data; o bloco nao inundou em nenhuma das 25 datas.

QUAL VARIANTE VALE PARA QUAL FONTE -- decidido pela NATUREZA DO NEGATIVO,
que a estrutura do dado revela sem precisar de arbitrio: se nenhum grupo tem
as duas classes, o negativo foi amostrado a parte e sua data e herdada
(UK, `exclusao_qualificada`) -- vale a `bloco`. Se os grupos sao mistos, o
negativo foi observado dentro da mesma AOI na mesma data do evento
(CEMS, `observado`) -- a data e real e vale a `heranca`. O script mede a
pureza dos grupos e registra a variante aplicavel por fonte; para o estrato
primario as duas rodam, porque e a comparacao entre elas que demonstra o
defeito 2.

JANELA DE TESTE: `max(8, n_unidades_ordenadas // 10)` -- mesma regra do
MOD-PROSP-01. Na variante `heranca` a unidade e o grupo (401 -> 40); na
`bloco` e o evento positivo (201 -> 20).

TRAVA DE EPV: um fold so e avaliado se o treino tiver, EM CADA CLASSE, pelo
menos `10 x n_variaveis` grupos. O MOD-PROSP-01 contava so o total de grupos;
contar assim deixa passar treino de 40 eventos contra 2 negativos, que
produz numero sem significar nada. EPV sempre quis dizer "eventos da classe
rara por variavel" -- aqui a regra e aplicada as duas classes, que e o que
ela significa. A trava nao altera nenhum fold do estrato primario (conferido
antes e depois); ela derruba os folds degenerados dos secundarios.

IC95: bootstrap percentil com N=1000 reamostrando GRUPOS do teste (regra
U2). Reamostra invalida (uma so classe) e descartada e contada.

RELATO: cada fold declara n de pontos e n de grupos, positivos e negativos
dos dois lados (regra U3).

CRITERIOS, ja fixados em `ext_criterios_de_acerto_v1.md` antes desta rodada:
faixa aceitavel 0,70-0,88; queda abaixo de 0,60 em fold com EPV suficiente e
degradacao real. Veredito com a mesma logica do MOD-PROSP-01:
  tendencia < -0,5 E algum fold < 0,60  -> DEGRADACAO_TEMPORAL
  media >= 0,70 E nenhum fold < 0,60    -> PROSPECTIVAMENTE_ESTAVEL
  caso contrario                        -> INCONCLUSIVO_VARIANCIA_ALTA

NAO FAZ: nao ajusta hiperparametro, nao escolhe corte nem variante pelo
resultado, nao declara modelo final, nao promove rotulo. O negativo ingles e
`exclusao_qualificada`, nao observacao de nao-ocorrencia.

Uso:
    python scripts/suscetibilidade/mod_prosp02_holdout_temporal_ds05.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susc_firth_adaptador  # noqa: F401,E402
from ds03_esquema_alvo import VERSAO  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
DATASET = RUNS / "ds-05-tabela-unica" / f"tabela_unica_pool_fluvial_{VERSAO}.csv"
OUT = RUNS / "mod-prosp-02"

FONTE_SERIE = "uk"
TOPO = ["elevation_m", "slope_deg", "hand_m", "twi_dinf"]
CHUVA = ["rain_max_24h", "rain_decay_index"]
CONJUNTOS = {"TERRENO": TOPO, "COMPLETO": TOPO + CHUVA}
NEGATIVO_ADMITIDO = ("observado", "exclusao_qualificada", "nao_aplicavel")

EPV_MINIMO = 10
MIN_GRUPOS_TESTE = 8
N_BOOT = 1000
SEMENTE = 20260820

AUC_MIN, AUC_MAX = 0.70, 0.88
AUC_DEGRADACAO = 0.60


# ----------------------------------------------------------------- ajuste

def ajustar(X: np.ndarray, y: np.ndarray):
    from firthlogist import FirthLogisticRegression

    m = FirthLogisticRegression(max_iter=500, skip_pvals=True, skip_ci=True)
    m.fit(X, y)
    return m


def auc_com_ic(y: np.ndarray, p: np.ndarray, grupos: np.ndarray,
               rng: np.random.Generator) -> dict:
    """AUC pontual e IC95 percentil reamostrando GRUPOS (regra U2).

    Reamostrar linhas infla a precisao por pseudo-replicacao: 40 pontos de um
    mesmo evento nao sao 40 observacoes independentes.
    """
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, p))
    unicos = np.unique(grupos)
    por_grupo = {g: np.flatnonzero(grupos == g) for g in unicos}
    amostras = []
    for _ in range(N_BOOT):
        escolhidos = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([por_grupo[g] for g in escolhidos])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        amostras.append(roc_auc_score(yb, p[idx]))
    if len(amostras) < 100:
        return {"auc": round(auc, 4), "ic95": None, "boot_validos": len(amostras)}
    lo, hi = np.percentile(amostras, [2.5, 97.5])
    return {"auc": round(auc, 4), "ic95": [round(float(lo), 4), round(float(hi), 4)],
            "boot_validos": len(amostras)}


def avaliar_fold(tr: pd.DataFrame, te: pd.DataFrame, features: list[str],
                 rng: np.random.Generator) -> dict | None:
    """Ajusta no passado, mede no futuro. Devolve None se faltar contraste."""
    from sklearn.metrics import roc_auc_score

    if tr.classe.nunique() < 2 or te.classe.nunique() < 2:
        return None
    Xtr = tr[features].to_numpy(dtype=float)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd == 0, 1, sd)
    ytr = tr.classe.to_numpy().astype(int)
    m = ajustar((Xtr - mu) / sd, ytr)
    p_tr = m.predict_proba((Xtr - mu) / sd)[:, 1]
    p_te = m.predict_proba((te[features].to_numpy(dtype=float) - mu) / sd)[:, 1]
    yte = te.classe.to_numpy().astype(int)

    r = auc_com_ic(yte, p_te, te.grupo_cv.to_numpy(), rng)
    auc_tr = float(roc_auc_score(ytr, p_tr))
    return {
        "n_treino": int(len(tr)), "n_teste": int(len(te)),
        "grupos_treino": int(tr.grupo_cv.nunique()),
        "grupos_teste": int(te.grupo_cv.nunique()),
        "eventos_treino": int(tr.loc[tr.classe == 1, "grupo_cv"].nunique()),
        "eventos_teste": int(te.loc[te.classe == 1, "grupo_cv"].nunique()),
        "pos_treino": int(ytr.sum()), "neg_treino": int(len(ytr) - ytr.sum()),
        "pos_teste": int(yte.sum()), "neg_teste": int(len(yte) - yte.sum()),
        "prevalencia_treino": round(float(ytr.mean()), 3),
        "prevalencia_teste": round(float(yte.mean()), 3),
        "epv_treino": round(tr.grupo_cv.nunique() / len(features), 1),
        "epv_eventos_treino": round(
            tr.loc[tr.classe == 1, "grupo_cv"].nunique() / len(features), 1),
        "auc_prospectivo": r["auc"], "ic95": r["ic95"],
        "boot_validos": r["boot_validos"],
        "auc_treino": round(auc_tr, 4), "lacuna": round(auc_tr - r["auc"], 4),
        "coef": {f: round(float(c), 4)
                 for f, c in zip(features, m.coef_.ravel()[:len(features)])},
    }


# --------------------------------------------------------------- variantes

def epv_ok(tr: pd.DataFrame, features: list[str]) -> bool:
    """EPV da CLASSE MINORITARIA do treino, em grupos, por variavel.

    Contar so grupos do treino, ou so eventos positivos, deixa passar fold
    degenerado: no estrato de Curitiba, com 114 negativos contra 1.238
    positivos, a alocacao 1:1 do teste levava quase todo negativo para o lado
    de fora e sobrava treino com 2 grupos negativos -- um ajuste que produz
    numero mas nao significa nada. EPV sempre foi "eventos da classe rara por
    variavel"; esta funcao aplica isso as duas classes, que e o que a regra
    quer dizer. Ela nao afeta o estrato primario (checado: os folds do UK sao
    identicos com e sem ela) -- e a trava que faltava para os secundarios.
    """
    por_classe = tr.groupby("classe").grupo_cv.nunique()
    if len(por_classe) < 2:
        return False
    return bool(por_classe.min() >= EPV_MINIMO * len(features))


def datas_por_grupo(u: pd.DataFrame) -> pd.Series:
    """Data de referencia de cada grupo -- a menor data dos seus pontos."""
    return pd.Series(u.groupby("grupo_cv")["data"].min()).sort_values()


def grupos_puros(u: pd.DataFrame) -> bool:
    """Nenhum grupo tem as duas classes -> o negativo foi amostrado a parte.

    E a assinatura estrutural do negativo por `exclusao_qualificada`: ele nao
    vem da mesma AOI observada na data do evento, entao a data que carrega e
    herdada. Grupo misto e o contrario -- negativo observado, data real.
    """
    return bool(u.groupby("grupo_cv").classe.nunique().max() == 1)


def variante_aplicavel(u: pd.DataFrame) -> str:
    return "bloco" if grupos_puros(u) else "heranca"


def folds_heranca(u: pd.DataFrame, features: list[str],
                  rng: np.random.Generator) -> list[dict]:
    """Replicacao literal do MOD-PROSP-01: todo grupo ordenado pela data minima."""
    datas = datas_por_grupo(u)
    ordem = datas.index.tolist()
    janela = max(MIN_GRUPOS_TESTE, len(ordem) // 10)
    min_treino = EPV_MINIMO * len(features)

    linhas, i = [], min_treino
    while i + MIN_GRUPOS_TESTE <= len(ordem):
        tr = u[u.grupo_cv.isin(ordem[:i])]
        te = u[u.grupo_cv.isin(ordem[i:i + janela])]
        if epv_ok(tr, features):
            f = avaliar_fold(tr, te, features, rng)
            if f is not None:
                f = {"variante": "heranca", "corte": str(datas.loc[ordem[i]].date()),
                     "janela_grupos": janela, **f}
                linhas.append(f)
        i += janela
    return linhas


def folds_bloco(u: pd.DataFrame, features: list[str],
                rng: np.random.Generator) -> list[dict]:
    """Eixo temporal so no positivo; bloco negativo alocado por sorteio."""
    pos = u[u.classe == 1]
    datas_ev = datas_por_grupo(pos)
    eventos = datas_ev.index.tolist()
    blocos = np.array(sorted(u.loc[u.classe == 0, "grupo_cv"].unique()))
    permutados = blocos[np.random.default_rng(SEMENTE).permutation(len(blocos))]
    pontos_por_bloco = u[u.classe == 0].groupby("grupo_cv").size()

    janela = max(MIN_GRUPOS_TESTE, len(eventos) // 10)
    min_treino = EPV_MINIMO * len(features)

    linhas, i, k = [], min_treino, 0
    while i + MIN_GRUPOS_TESTE <= len(eventos):
        ev_tr, ev_te = eventos[:i], eventos[i:i + janela]
        alvo = int(u[u.grupo_cv.isin(ev_te)].shape[0])

        # blocos de teste: a partir de um deslocamento que anda com o fold, para
        # que nenhum bloco fique sempre do mesmo lado ao longo da trajetoria
        blocos_te, acumulado, j = [], 0, 0
        while acumulado < alvo and j < len(permutados):
            b = permutados[(k + j) % len(permutados)]
            blocos_te.append(b)
            acumulado += int(pontos_por_bloco[b])
            j += 1
        k = (k + j) % len(permutados)
        blocos_tr = [b for b in permutados if b not in set(blocos_te)]

        tr = u[u.grupo_cv.isin(set(ev_tr) | set(blocos_tr))]
        te = u[u.grupo_cv.isin(set(ev_te) | set(blocos_te))]
        assert not (set(tr.grupo_cv) & set(te.grupo_cv)), "grupo dos dois lados"

        f = avaliar_fold(tr, te, features, rng) if epv_ok(tr, features) else None
        if f is not None:
            f = {"variante": "bloco", "corte": str(datas_ev.loc[eventos[i]].date()),
                 "janela_grupos": janela, "blocos_negativos_teste": len(blocos_te),
                 **f}
            linhas.append(f)
        i += janela
    return linhas


# ------------------------------------------------------------- diagnostico

def viabilidade_por_fonte(pool: pd.DataFrame, n_variaveis: int) -> pd.DataFrame:
    """Quantas fontes do pool sustentam janela expansiva? Medido, nao afirmado."""
    linhas = []
    for fonte, sub in pool.groupby("fonte"):
        datas = datas_por_grupo(sub)
        ordem = datas.index.tolist()
        janela = max(MIN_GRUPOS_TESTE, len(ordem) // 10)
        min_treino = EPV_MINIMO * n_variaveis
        folds, i = 0, min_treino
        while i + MIN_GRUPOS_TESTE <= len(ordem):
            te = sub[sub.grupo_cv.isin(ordem[i:i + janela])]
            tr = sub[sub.grupo_cv.isin(ordem[:i])]
            if te.classe.nunique() == 2 and tr.classe.nunique() == 2:
                folds += 1
            i += janela
        linhas.append({
            "fonte": fonte, "pontos": len(sub), "grupos": sub.grupo_cv.nunique(),
            "eventos_positivos": sub.loc[sub.classe == 1, "grupo_cv"].nunique(),
            "datas_distintas": sub["data"].nunique(),
            "inicio": str(sub["data"].min().date()),
            "fim": str(sub["data"].max().date()),
            "anos_cobertos": sub["data"].dt.year.nunique(),
            "positivos": int((sub.classe == 1).sum()),
            "negativos": int((sub.classe == 0).sum()),
            "prevalencia": round(float((sub.classe == 1).mean()), 3),
            "pontos_por_grupo": round(len(sub) / sub.grupo_cv.nunique(), 1),
            "grupos_puros": grupos_puros(sub),
            "variante_aplicavel": variante_aplicavel(sub),
            "folds_sustentaveis": folds,
        })
    return pd.DataFrame(linhas).sort_values("anos_cobertos", ascending=False)


def confusao_fonte_periodo(pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Por que o corte temporal no pool inteiro seria corte de fonte."""
    pool = pool.copy()
    pool["ano"] = pool["data"].dt.year
    m = pool.groupby(["ano", "fonte"]).grupo_cv.nunique().unstack(fill_value=0)
    corte = pd.Timestamp("2020-01-01")
    lados = {}
    for nome, sub in (("treino_antes_de_2020", pool[pool["data"] < corte]),
                      ("teste_2020_em_diante", pool[pool["data"] >= corte])):
        lados[nome] = {
            "pontos": len(sub), "grupos": int(sub.grupo_cv.nunique()),
            "fontes": sub.fonte.value_counts().to_dict(),
            "classe_relevo": sub.classe_relevo.value_counts().to_dict(),
        }
    return m, {
        "corte_ilustrativo": str(corte.date()),
        "lados": lados,
        "leitura": ("nenhuma fonte aparece dos dois lados com peso comparavel: "
                    "o corte temporal no pool e um corte de fonte disfarcado, e "
                    "por isso o E4 e medido no estrato de serie longa"),
    }


# ------------------------------------------------------------------ resumo

def resumir(linhas: list[dict]) -> dict:
    a = np.array([f["auc_prospectivo"] for f in linhas], dtype=float)
    tend = (float(np.corrcoef(np.arange(len(a)), a)[0, 1])
            if len(a) > 2 else float("nan"))
    degradou = int((a < AUC_DEGRADACAO).sum())
    na_faixa = int(((a >= AUC_MIN) & (a <= AUC_MAX)).sum())
    if np.isfinite(tend) and tend < -0.5 and degradou > 0:
        veredito = "DEGRADACAO_TEMPORAL"
    elif a.mean() >= AUC_MIN and degradou == 0:
        veredito = "PROSPECTIVAMENTE_ESTAVEL"
    else:
        veredito = "INCONCLUSIVO_VARIANCIA_ALTA"
    return {
        "folds": len(a), "auc_medio": round(float(a.mean()), 4),
        "auc_mediana": round(float(np.median(a)), 4),
        "auc_min": round(float(a.min()), 4), "auc_max": round(float(a.max()), 4),
        "desvio": round(float(a.std()), 4),
        "tendencia": round(tend, 3) if np.isfinite(tend) else None,
        "folds_na_faixa": na_faixa, "folds_degradados": degradou,
        "prevalencia_teste_min": round(min(f["prevalencia_teste"] for f in linhas), 3),
        "prevalencia_teste_max": round(max(f["prevalencia_teste"] for f in linhas), 3),
        "veredito": veredito,
    }


def rodar_estrato(u: pd.DataFrame, variantes: tuple[str, ...],
                  rotulo: str) -> tuple[dict, list[dict]]:
    """Roda o desenho declarado num estrato, para cada conjunto de variaveis."""
    construtores = {"heranca": folds_heranca, "bloco": folds_bloco}
    resultados, todas = {}, []
    for nome_conj, features in CONJUNTOS.items():
        resultados[nome_conj] = {"features": features, "n_variaveis": len(features)}
        for nome_var in variantes:
            rng = np.random.default_rng(SEMENTE)
            linhas = construtores[nome_var](u, features, rng)
            print(f"\n---{rotulo} / {nome_conj} / variante {nome_var}---")
            if not linhas:
                resultados[nome_conj][nome_var] = {
                    "folds": 0, "veredito": "SEM_FOLD_COM_EPV_SUFICIENTE"}
                print("  nenhum fold com EPV suficiente")
                continue
            for f in linhas:
                ic = (f"[{f['ic95'][0]:.4f}; {f['ic95'][1]:.4f}]"
                      if f["ic95"] else "IC indisponivel")
                print(f"  corte={f['corte']:12s} treino={f['grupos_treino']:4d}g"
                      f"/{f['eventos_treino']:3d}ev  teste={f['grupos_teste']:4d}g "
                      f"({f['pos_teste']:5d}+/{f['neg_teste']:5d}-, "
                      f"prev={f['prevalencia_teste']:.2f})  "
                      f"AUC={f['auc_prospectivo']:.4f} {ic}")
            r = resumir(linhas)
            resultados[nome_conj][nome_var] = {**r, "detalhe": linhas}
            print(f"  media={r['auc_medio']:.4f} mediana={r['auc_mediana']:.4f} "
                  f"min={r['auc_min']:.4f} max={r['auc_max']:.4f} "
                  f"tendencia={r['tendencia']}")
            print(f"  na faixa [{AUC_MIN}; {AUC_MAX}]={r['folds_na_faixa']}/"
                  f"{r['folds']}  abaixo de {AUC_DEGRADACAO}="
                  f"{r['folds_degradados']}/{r['folds']}")
            print(f"  VEREDITO={r['veredito']}")
            todas.extend([{"estrato": rotulo, "conjunto": nome_conj, **f}
                          for f in linhas])
    return resultados, todas


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===MOD-PROSP-02: HOLDOUT TEMPORAL SOBRE A TABELA UNICA===")

    if not DATASET.exists():
        print(f"ABORTADO: {DATASET} ausente. Gere com: "
              "python scripts/suscetibilidade/ds05_admissao_consolidacao.py")
        return 1

    pool = pd.read_csv(DATASET, low_memory=False)
    pool = pool[pool.classe.isin([0, 1])
                & pool.nivel_negativo.isin(NEGATIVO_ADMITIDO)].copy()
    pool["data"] = pd.to_datetime(pool.data_evento.astype(str).str.slice(0, 10),
                                  errors="coerce")
    pool = pool.dropna(subset=["data", "grupo_cv"] + CONJUNTOS["COMPLETO"])

    viab = viabilidade_por_fonte(pool, len(TOPO))
    viab.to_csv(OUT / "viabilidade_por_fonte.csv", index=False)
    print("\n---VIABILIDADE POR FONTE (trava de EPV com 4 variaveis)---")
    print(viab.to_string(index=False))

    matriz, confusao = confusao_fonte_periodo(pool)
    matriz.to_csv(OUT / "confusao_fonte_periodo.csv")
    print("\n---CORTE UNICO EM 2020 NO POOL (por que nao se mede aqui)---")
    for nome, d in confusao["lados"].items():
        print(f"  {nome}: {d['pontos']:,} pontos, {d['grupos']} grupos, "
              f"fontes={d['fontes']}")

    u = pool[pool.fonte == FONTE_SERIE].copy()
    print(f"\n---ESTRATO DE SERIE LONGA: fonte={FONTE_SERIE}---")
    print(f"  pontos={len(u):,} grupos={u.grupo_cv.nunique()} "
          f"eventos={u.loc[u.classe == 1, 'grupo_cv'].nunique()} "
          f"blocos_negativos={u.loc[u.classe == 0, 'grupo_cv'].nunique()} "
          f"datas={u['data'].nunique()} "
          f"periodo={u['data'].min().date()} -> {u['data'].max().date()}")
    datas_por_bloco = u[u.classe == 0].groupby("grupo_cv")["data"].nunique()
    print(f"  datas distintas por bloco negativo: mediana="
          f"{datas_por_bloco.median():.0f} max={datas_por_bloco.max()} "
          "-- e por isso que a variante `bloco` existe")

    resultados, todas = rodar_estrato(u, ("heranca", "bloco"), FONTE_SERIE)

    secundarios = {}
    for fonte in sorted(f for f in pool.fonte.unique() if f != FONTE_SERIE):
        sub = pool[pool.fonte == fonte].copy()
        variante = variante_aplicavel(sub)
        print(f"\n===ESTRATO SECUNDARIO: {fonte} "
              f"(variante aplicavel: {variante})===")
        print(f"  pontos={len(sub):,} grupos={sub.grupo_cv.nunique()} "
              f"anos_cobertos={sub['data'].dt.year.nunique()} "
              f"pos={int((sub.classe == 1).sum()):,} "
              f"neg={int((sub.classe == 0).sum()):,}")
        res, linhas = rodar_estrato(sub, (variante,), fonte)
        secundarios[fonte] = {
            "variante_aplicavel": variante,
            "grupos_puros": grupos_puros(sub),
            "anos_cobertos": int(sub["data"].dt.year.nunique()),
            "positivos": int((sub.classe == 1).sum()),
            "negativos": int((sub.classe == 0).sum()),
            "pontos_por_grupo": round(len(sub) / sub.grupo_cv.nunique(), 1),
            "conjuntos": res,
        }
        todas.extend(linhas)

    if not todas:
        print("ABORTADO: nenhum fold avaliavel em nenhuma combinacao.")
        return 2

    df = pd.DataFrame(todas)
    df["coef"] = df["coef"].apply(json.dumps)
    df["ic95"] = df["ic95"].apply(lambda v: json.dumps(v) if v else "")
    df.to_csv(OUT / "folds.csv", index=False)

    (OUT / "resultado.json").write_text(json.dumps({
        "entrada": str(DATASET.relative_to(REPO)).replace("\\", "/"),
        "estrato": FONTE_SERIE,
        "base_do_estrato": {
            "pontos": int(len(u)), "grupos": int(u.grupo_cv.nunique()),
            "eventos_positivos": int(u.loc[u.classe == 1, "grupo_cv"].nunique()),
            "blocos_negativos": int(u.loc[u.classe == 0, "grupo_cv"].nunique()),
            "datas_distintas": int(u["data"].nunique()),
            "periodo": [str(u["data"].min().date()), str(u["data"].max().date())],
            "datas_por_bloco_negativo_mediana": float(datas_por_bloco.median()),
            "datas_por_bloco_negativo_max": int(datas_por_bloco.max()),
        },
        "epv_minimo": EPV_MINIMO, "n_boot": N_BOOT, "semente": SEMENTE,
        "criterios": {"faixa": [AUC_MIN, AUC_MAX], "degradacao": AUC_DEGRADACAO},
        "conjuntos": resultados,
        "estratos_secundarios": secundarios,
        "leitura_dos_secundarios": (
            "nao respondem o E4 e nao entram no veredito: CEMS e Curitiba "
            "cobrem 3 e 4 anos, entao o que medem e variacao dentro de um "
            "mesmo regime, nao horizonte. Estao aqui para que a escolha do "
            "estrato primario seja conferivel em vez de afirmada"),
        "confusao_fonte_periodo": confusao,
        "precedente": {
            "mod-prosp-01": "ds-01, 328 grupos, 9 folds, AUC medio 0,7686, "
                            "PROSPECTIVAMENTE_ESTAVEL, sem IC",
            "o_que_muda_aqui": ["base congelada ds-05",
                                "variante `bloco` sem data artificial de negativo",
                                "IC95 por bootstrap de grupos em todo fold"],
        },
        "segundos": round(time.time() - t0, 1),
        "nao_e": ("nao e validacao operacional nem autoriza uso preditivo: e "
                  "holdout temporal num piloto de exclusao qualificada, medido "
                  "num unico pais"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    (OUT / "commands.txt").write_text(
        "python scripts/suscetibilidade/ds05_admissao_consolidacao.py\n"
        "python scripts/suscetibilidade/mod_prosp02_holdout_temporal_ds05.py\n"
        "python -m pytest tests/test_mod_prosp02_holdout_temporal.py -q\n",
        encoding="utf-8")

    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
