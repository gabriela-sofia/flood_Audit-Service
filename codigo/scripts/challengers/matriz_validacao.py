"""
Matriz canonica de validacao: uma API, seis esquemas, nenhum fold caseiro.

-------------------------------------------------------------------------
POR QUE UMA API UNICA
-------------------------------------------------------------------------

Quando cada experimento monta os proprios folds, dois numeros na mesma tabela
podem ter sido produzidos por desenhos diferentes -- e a tabela nao mostra
isso. Foi assim que Curitiba foi promovida uma vez: a validacao media outra
coisa, e o numero parecia comparavel.

Aqui todo resultado passa por `avaliar_modelo(modelo, dataset, esquema,
protocolo)` e sai com o mesmo registro, incluindo `protocol_hash` e
`dataset_hash`. Comparar dois registros com hashes diferentes levanta
`ProtocoloDivergente` -- ver `protocolo_experimental`.

-------------------------------------------------------------------------
OS SEIS ESQUEMAS, e o que cada um pode e nao pode responder
-------------------------------------------------------------------------

  grouped_kfold      DIAGNOSTICO. Embaralha no tempo, entao mede "quanto sinal
                     ha neste conjunto", nunca "isto generaliza". E o numero
                     que o modelo servido publica, e o unico que a promocao de
                     Curitiba usou -- por isso ele carrega
                     `apenas_diagnostico: True` no registro.

  leave_one_event_out  cada evento/data vira teste uma vez. Responde
                     "generaliza para um evento nao visto?", que e a pergunta
                     de um produto que sera usado no proximo evento.

  holdout_temporal   corte unico por data: passado treina, futuro testa. E a
                     pergunta operacional literal.

  forward_temporal   varios cortes crescentes. Distingue "colapsou de vez" de
                     "degrada com o tempo" -- um holdout unico nao distingue.

  spatial_block      blocos de 0,05 grau. Mede dependencia espacial residual.

  leave_region_out   treina fora da regiao alvo e testa nela. E o desenho da
                     TRANSFERENCIA, que e como Curitiba e Petropolis sao
                     servidas de verdade.

Cada esquema devolve `n_folds_validos` E `n_folds_descartados`, com motivo.
Fold sem as duas classes nao entra na media -- e some sem deixar rastro se
ninguem contar, que e como uma AUC vira a media de tres folds fingindo ser
cinco.
"""

from __future__ import annotations

import json

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "challengers"))

from protocolo_experimental import (  # noqa: E402
    CORTES_ESPACIAIS, CRITERIOS_FISICOS, DATASETS, SEED, TABELA, protocolo,
)
from proveniencia import sha256_arquivo, sha256_json  # noqa: E402

GROUPED = "grouped_kfold"
LOEO = "leave_one_event_out"
HOLDOUT = "holdout_temporal"
FORWARD = "forward_temporal"
SPATIAL = "spatial_block"
REGION = "leave_region_out"
ESQUEMAS = (GROUPED, LOEO, HOLDOUT, FORWARD, SPATIAL, REGION)

# `grouped_kfold` embaralha o tempo. Ele responde "ha sinal aqui?" e nao
# "isto generaliza?" -- e a diferenca entre as duas ja custou uma promocao
# indevida neste projeto.
APENAS_DIAGNOSTICO = {GROUPED}


def parse_data(s: pd.Series) -> pd.Series:
    """`format='mixed'` NAO e detalhe.

    Sem ele, o pandas infere o formato do primeiro elemento nao nulo e
    transforma 93,6% das datas validas do pool em NaT: a fonte `uk` grava com
    hora e as demais sem. Uma analise temporal sobre 6% dos dados teria
    passado por analise temporal.

    -------------------------------------------------------------------------
    `utc=True` TAMBEM NAO E DETALHE, e foi medido antes de ser escrito
    -------------------------------------------------------------------------

    A `uk` passou a gravar OFFSET, e nao so hora. Com offset em parte das
    linhas e nenhum no resto, o pandas levanta `Mixed timezones detected` e o
    modulo inteiro morre -- e como esta e a API canonica de validacao, morria
    junto todo experimento que passa por ela. Foi o que derrubou
    `experimentos_cientificos` e `petropolis_lacunas`.

    A correcao poderia ser perigosa: normalizar para UTC desloca o dia de
    qualquer carimbo cujo horario local cruze a meia-noite, e o dia e o que os
    esquemas `holdout` e `forward` cortam. Por isso foi medido na tabela antes:

        588.380 linhas (99,4%)  data pura, sem hora e sem fuso
          3.738 linhas (0,6%)   com fuso, TODAS da `uk`, TODAS em `+00:00`
              0 linhas          mudam de dia de calendario ao normalizar

    Como o unico offset presente e o proprio UTC, a normalizacao nao move
    nenhuma data. O `tz_localize(None)` devolve tudo ingenuo, que e o que as
    99,4% ja eram -- sem ele, comparar estas datas com um corte ingenuo
    levantaria erro mais adiante.

    Se algum dia entrar fonte com offset diferente de zero, este comentario
    deixa de valer e a conta tem de ser refeita: `test_data_nao_desloca_dia`
    reprova.
    """
    return pd.to_datetime(s.astype("string"), errors="coerce",
                          format="mixed", utc=True).dt.tz_localize(None)


# ------------------------------------------------------------------ dataset

@dataclass
class Conjunto:
    nome: str
    dados: pd.DataFrame
    features: list
    grupos: np.ndarray
    y: np.ndarray
    datas: pd.Series
    hash_arquivo: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def X(self) -> np.ndarray:
        return self.dados[self.features].to_numpy(dtype=float)


def carregar(nome: str) -> Conjunto:
    import svc01_construir_modelos_servidos as svc01

    cfg = DATASETS[nome]
    caminho = TABELA / cfg["arquivo"]
    d = pd.read_parquet(caminho)
    if "admitido" in d.columns and cfg["arquivo"].startswith("tabela_unica_admitidos"):
        d = d[d.admitido == True]  # noqa: E712
    d = d[d.classe.isin([0, 1])].query(cfg["filtro"])
    d = d.dropna(subset=cfg["features"] + ["grupo_cv", "lat", "lon"]).copy()
    grupos = svc01.grupo_nunca_mais_fino_que_a_celula(d).to_numpy()
    return Conjunto(
        nome=nome, dados=d, features=list(cfg["features"]), grupos=grupos,
        y=d.classe.to_numpy().astype(int), datas=parse_data(d.data_evento),
        hash_arquivo=sha256_arquivo(caminho),
        extra={"n": int(len(d)), "n_grupos": int(len(np.unique(grupos)))})


# ------------------------------------------------------------------- folds
#
# Cada gerador devolve (rotulo, indices_treino, indices_teste). Nenhum deles
# olha `y` para decidir o corte -- olhar seria escolher o fold pelo resultado.

def _folds_grouped(c: Conjunto, n_splits: int = 5):
    from sklearn.model_selection import GroupKFold

    k = min(n_splits, len(np.unique(c.grupos)))
    if k < 2:
        return
    for i, (tr, te) in enumerate(GroupKFold(n_splits=k).split(c.X, c.y, c.grupos), 1):
        yield f"fold_{i}", tr, te


def _folds_evento(c: Conjunto, max_eventos: int = 40):
    """Um evento = uma data. Datas ausentes ficam sempre no treino.

    Elas nao podem virar um fold de teste: uma data que nao existe nao e um
    evento nao visto, e um fold construido sobre ela mediria ruido.
    """
    datas = c.datas
    validas = datas.dropna().unique()
    if len(validas) < 2:
        return
    # Eventos maiores primeiro: um fold de teste com tres pontos nao produz
    # AUC interpretavel, e gastar folds nele empurra os grandes para fora.
    ordem = (pd.Series(datas.values).value_counts().index.tolist())[:max_eventos]
    idx = np.arange(len(c.y))
    for data in ordem:
        te = idx[(datas == data).to_numpy()]
        tr = idx[(datas != data).to_numpy() | datas.isna().to_numpy()]
        if len(te) == 0 or len(tr) == 0:
            continue
        yield f"evento_{pd.Timestamp(data).date()}", tr, te


def _folds_holdout(c: Conjunto, ano_corte: int):
    anos = c.datas.dt.year
    idx = np.arange(len(c.y))
    tr = idx[(anos < ano_corte).to_numpy(na_value=False)]
    te = idx[(anos == ano_corte).to_numpy(na_value=False)]
    if len(tr) and len(te):
        yield f"treino<{ano_corte}_teste{ano_corte}", tr, te


def _folds_por_data(c: Conjunto, cortes: list, minimo_grupos: int = 5):
    """Encadeamento progressivo por DATA, e nao por ano.

    O ano e granularidade boa para uma cidade com anos de serie, e ruim para
    um pool de desastres: o pool areal do Copernicus EMS tem TREZE datas
    distintas em apenas tres anos civis, entao cortar por ano oferece no
    maximo dois cortes -- sempre abaixo do minimo de tres que a matriz de
    sustentacao exige para sair de `inconclusive`.

    O limite era da regua, e nao do dado: por data o mesmo pool oferece onze
    cortes avaliaveis.

    `minimo_grupos` existe porque um corte com duas AOIs de um lado nao mede
    transporte, mede sorte -- e a media de uma AUC dessas entraria na serie
    com o mesmo peso das outras.
    """
    idx = np.arange(len(c.y))
    dias = c.datas.dt.date
    for corte in cortes:
        m_tr = (dias < corte).to_numpy(na_value=False)
        m_te = (dias >= corte).to_numpy(na_value=False)
        tr, te = idx[m_tr], idx[m_te]
        if not len(tr) or not len(te):
            continue
        if len(set(c.y[tr])) < 2 or len(set(c.y[te])) < 2:
            continue
        if (len(set(c.grupos[tr])) < minimo_grupos
                or len(set(c.grupos[te])) < minimo_grupos):
            continue
        yield f"treino<{corte}_teste>={corte}", tr, te


def _folds_forward(c: Conjunto, anos_teste: list = None, datas_teste: list = None):
    """Por ano (como sempre foi) ou por data, quando o conjunto pedir.

    `datas_teste` e opcional e aditivo: nenhum conjunto existente muda de
    comportamento por causa dele.
    """
    if datas_teste:
        yield from _folds_por_data(c, datas_teste)
        return
    for ano in (anos_teste or []):
        yield from _folds_holdout(c, ano)


def _folds_spatial(c: Conjunto, lado: float = None, n_splits: int = 5):
    from sklearn.model_selection import GroupKFold

    lado = lado or CORTES_ESPACIAIS["bloco_graus"]
    bloco = ((c.dados.lat / lado).round().astype("int64").astype(str) + "_"
             + (c.dados.lon / lado).round().astype("int64").astype(str)).to_numpy()
    k = min(n_splits, len(np.unique(bloco)))
    if k < 2:
        return
    for i, (tr, te) in enumerate(GroupKFold(n_splits=k).split(c.X, c.y, bloco), 1):
        yield f"bloco_{i}", tr, te


def _folds_region(c: Conjunto, coluna: str = "fonte"):
    valores = c.dados[coluna].astype(str).to_numpy()
    idx = np.arange(len(c.y))
    for v in sorted(set(valores)):
        te = idx[valores == v]
        tr = idx[valores != v]
        if len(te) and len(tr):
            yield f"fora_de_{v}", tr, te


def gerar_folds(c: Conjunto, esquema: str, **kw):
    if esquema == GROUPED:
        return list(_folds_grouped(c, **kw))
    if esquema == LOEO:
        return list(_folds_evento(c, **kw))
    if esquema == HOLDOUT:
        return list(_folds_holdout(c, **kw))
    if esquema == FORWARD:
        return list(_folds_forward(c, **kw))
    if esquema == SPATIAL:
        return list(_folds_spatial(c, **kw))
    if esquema == REGION:
        return list(_folds_region(c, **kw))
    raise ValueError(f"esquema desconhecido: {esquema!r}; conhecidos: {ESQUEMAS}")


# ------------------------------------------------------------------ metricas

def calibracao(y, p) -> dict:
    """Slope e intercepto da recalibracao logistica de Cox."""
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logito = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2 or np.allclose(logito, logito[0]):
        return {"calibration_slope": None, "calibration_intercept": None,
                "motivo": "sem variacao suficiente para recalibrar"}
    m = LogisticRegression(max_iter=1000, C=1e12)
    m.fit(logito, y)
    return {"calibration_slope": round(float(m.coef_[0][0]), 4),
            "calibration_intercept": round(float(m.intercept_[0]), 4)}


def _auc_ic95_por_grupo(y, p, grupos, n_boot: int = 300,
                        seed: int = SEED) -> list | None:
    """IC da AUC reamostrando GRUPO, e nao linha.

    Reamostrar linha em Curitiba, onde ha ate seis observacoes na mesma
    celula, produz um IC estreito que descreve a duplicacao e nao a incerteza.
    """
    from sklearn.metrics import roc_auc_score

    unicos = np.unique(grupos)
    if len(unicos) < 5:
        return None
    rng = np.random.default_rng(seed)
    por_grupo = {g: np.where(grupos == g)[0] for g in unicos}
    aucs = []
    for _ in range(n_boot):
        escolha = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([por_grupo[g] for g in escolha])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    if len(aucs) < 30:
        return None
    return [round(float(np.percentile(aucs, 2.5)), 4),
            round(float(np.percentile(aucs, 97.5)), 4)]


def _sinais_fisicos(coefs, feats: list) -> dict:
    esperados = CRITERIOS_FISICOS["sinais_esperados"]
    if coefs is None:
        return {"disponivel": False,
                "motivo": "estimador nao expoe efeito por variavel"}
    por_variavel = {}
    for f, c in zip(feats, coefs):
        alvo = esperados.get(f)
        por_variavel[f] = {
            "direcao_esperada": alvo,
            "direcao_estimada": int(np.sign(c)) if c else 0,
            "coef": round(float(c), 4),
            "veredito": ("inconclusivo" if alvo is None or c == 0
                         else "consistente" if np.sign(c) == np.sign(alvo)
                         else "invertido"),
        }
    return {"disponivel": True, "por_variavel": por_variavel,
            "n_consistente": sum(1 for v in por_variavel.values()
                                 if v["veredito"] == "consistente"),
            "n_invertido": sum(1 for v in por_variavel.values()
                               if v["veredito"] == "invertido")}


# ------------------------------------------------------------------- API

def avaliar_modelo(modelo, conjunto: Conjunto, esquema: str,
                   protocolo_dict: dict | None = None,
                   com_ic: bool = True, **kw_folds) -> dict:
    """A unica porta de avaliacao desta frente.

    `modelo` precisa de `.nome`, `.ajustar(X, y)`, `.prever(m, X)` e
    `.coeficientes(m)`. Nada mais -- e de proposito: um estimador que
    precisasse de tratamento especial aqui estaria sendo avaliado sob outro
    desenho.
    """
    from sklearn.metrics import (
        balanced_accuracy_score, brier_score_loss, roc_auc_score,
    )

    p_dict = protocolo_dict or protocolo()
    folds = gerar_folds(conjunto, esquema, **kw_folds)
    X, y, grupos = conjunto.X, conjunto.y, conjunto.grupos

    p = np.full(len(y), np.nan)
    registros, descartados = [], []
    for rotulo, tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            descartados.append({"fold": rotulo, "motivo": "treino com uma classe"})
            continue
        if len(np.unique(y[te])) < 2:
            descartados.append({"fold": rotulo, "motivo": "teste com uma classe"})
            continue
        # VAZAMENTO: a padronizacao ve SO o treino. Ajustar no conjunto
        # inteiro passa media e desvio do teste para o treino -- pequeno,
        # invisivel e suficiente para inflar um holdout.
        mu = X[tr].mean(0)
        sd = np.where(X[tr].std(0) == 0, 1, X[tr].std(0))
        ajustado = modelo.ajustar((X[tr] - mu) / sd, y[tr])
        pred = modelo.prever(ajustado, (X[te] - mu) / sd)
        p[te] = pred
        registros.append({
            "fold": rotulo,
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "train_groups": int(len(np.unique(grupos[tr]))),
            "test_groups": int(len(np.unique(grupos[te]))),
            "prevalence_train": round(float(y[tr].mean()), 4),
            "prevalence_test": round(float(y[te].mean()), 4),
            "train_period": _periodo(conjunto.datas.iloc[tr]),
            "test_period": _periodo(conjunto.datas.iloc[te]),
            "auc": round(float(roc_auc_score(y[te], pred)), 4),
            "brier": round(float(brier_score_loss(y[te], pred)), 4),
            # IC POR FOLD, e nao so no agregado.
            #
            # A matriz de sustentacao recusa uma serie de cortes com
            # `cortes_sem_intervalo_para_separar_queda_de_ruido`: sem
            # intervalo, uma AUC que cai de 0,73 para 0,64 entre dois cortes
            # pode ser deriva real ou pode ser o tamanho do conjunto de teste,
            # e a serie inteira fica `inconclusive` por isso.
            #
            # O IC agregado nao resolve: ele descreve o esquema todo, e a
            # pergunta temporal e sobre a variacao ENTRE cortes. Reamostra
            # grupo, e nao linha, pela mesma razao de sempre -- em um corte com
            # 25 AOIs de teste, reamostrar linha mediria a duplicacao dentro
            # da mancha e nao a incerteza entre desastres.
            #
            # Devolve `None` quando o corte tem menos de cinco grupos: um
            # intervalo ali seria mais enganoso que a ausencia dele.
            "auc_ic95": _auc_ic95_por_grupo(y[te], pred, grupos[te]),
        })

    ok = ~np.isnan(p)
    base = {
        "modelo": getattr(modelo, "nome", "desconhecido"),
        "modelo_id": getattr(modelo, "id_protocolo", None),
        "dataset": conjunto.nome,
        "esquema": esquema,
        "apenas_diagnostico": esquema in APENAS_DIAGNOSTICO,
        "features": conjunto.features,
        "n_folds_validos": len(registros),
        "n_folds_descartados": len(descartados),
        "folds_descartados": descartados,
        "folds": registros,
        "protocol_hash": p_dict["hash_do_protocolo"],
        "dataset_hash": conjunto.hash_arquivo,
    }
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return {**base, "auc": None,
                "motivo": "nenhuma predicao com as duas classes"}

    yo, po, go = y[ok], p[ok], grupos[ok]
    mu = X.mean(0)
    sd = np.where(X.std(0) == 0, 1, X.std(0))
    coefs = modelo.coeficientes(modelo.ajustar((X - mu) / sd, y))
    aucs = [r["auc"] for r in registros]

    return {
        **base,
        "n_train": int(sum(r["n_train"] for r in registros) / max(1, len(registros))),
        "n_test": int(ok.sum()),
        "prevalence_train": round(float(np.mean(
            [r["prevalence_train"] for r in registros])), 4),
        "prevalence_test": round(float(yo.mean()), 4),
        "auc": round(float(roc_auc_score(yo, po)), 4),
        "auc_ic95": _auc_ic95_por_grupo(yo, po, go) if com_ic else None,
        "auc_desvio_entre_folds": (round(float(np.std(aucs)), 4)
                                   if len(aucs) > 1 else None),
        "auc_min_fold": round(float(np.min(aucs)), 4) if aucs else None,
        "balanced_accuracy": round(float(
            balanced_accuracy_score(yo, (po >= 0.5).astype(int))), 4),
        "brier": round(float(brier_score_loss(yo, po)), 4),
        **calibracao(yo, po),
        # `coverage` aqui e a fracao do conjunto que recebeu predicao. A
        # abstencao por dominio e outra coisa e vive no laboratorio de domain
        # shift; misturar as duas faria a cobertura de fold parecer decisao
        # cientifica.
        "coverage": round(float(ok.mean()), 4),
        "abstention_rate": round(float(1 - ok.mean()), 4),
        "physical_signs": _sinais_fisicos(coefs, conjunto.features),
        "uncertainty": {"metodo": "bootstrap de grupos, 300 replicas",
                        "unidade": "grupo fundido com a celula"},
    }


def _periodo(datas: pd.Series) -> dict:
    validas = datas.dropna()
    if validas.empty:
        return {"inicio": None, "fim": None, "n_datado": 0}
    return {"inicio": str(validas.min().date()), "fim": str(validas.max().date()),
            "n_datado": int(len(validas))}


def linhas_para_csv(resultados: list) -> pd.DataFrame:
    """Achata a lista de resultados no `metrics.csv` da rodada."""
    linhas = []
    for r in resultados:
        linhas.append({
            "dataset": r["dataset"], "modelo": r["modelo"],
            "modelo_id": r.get("modelo_id"), "esquema": r["esquema"],
            "apenas_diagnostico": r["apenas_diagnostico"],
            "n_folds_validos": r["n_folds_validos"],
            "n_folds_descartados": r["n_folds_descartados"],
            "n_train": r.get("n_train"), "n_test": r.get("n_test"),
            "prevalence_train": r.get("prevalence_train"),
            "prevalence_test": r.get("prevalence_test"),
            "auc": r.get("auc"),
            "auc_ic95_lo": (r.get("auc_ic95") or [None, None])[0],
            "auc_ic95_hi": (r.get("auc_ic95") or [None, None])[1],
            "auc_desvio_entre_folds": r.get("auc_desvio_entre_folds"),
            "auc_min_fold": r.get("auc_min_fold"),
            "balanced_accuracy": r.get("balanced_accuracy"),
            "brier": r.get("brier"),
            "calibration_intercept": r.get("calibration_intercept"),
            "calibration_slope": r.get("calibration_slope"),
            "coverage": r.get("coverage"),
            "abstention_rate": r.get("abstention_rate"),
            "n_sinal_consistente": (r.get("physical_signs") or {}).get("n_consistente"),
            "n_sinal_invertido": (r.get("physical_signs") or {}).get("n_invertido"),
            "protocol_hash": r["protocol_hash"],
            "dataset_hash": r["dataset_hash"],
        })
    return pd.DataFrame(linhas)


def folds_para_csv(resultados: list) -> pd.DataFrame:
    linhas = []
    for r in resultados:
        for f in r.get("folds", []):
            linhas.append({
                "dataset": r["dataset"], "modelo": r["modelo"],
                "esquema": r["esquema"], "fold": f["fold"],
                "n_train": f["n_train"], "n_test": f["n_test"],
                "train_groups": f["train_groups"], "test_groups": f["test_groups"],
                "prevalence_train": f["prevalence_train"],
                "prevalence_test": f["prevalence_test"],
                "train_period_ini": f["train_period"]["inicio"],
                "train_period_fim": f["train_period"]["fim"],
                "test_period_ini": f["test_period"]["inicio"],
                "test_period_fim": f["test_period"]["fim"],
                "auc": f["auc"], "brier": f["brier"],
                # O IC POR FOLD PRECISA CHEGAR AO CSV, e nao so ao dicionario.
                #
                # `avaliar_modelo` passou a calcula-lo em 13/09, e ele morria
                # aqui: este serializador nomeia as colunas uma a uma, entao
                # chave nova no registro simplesmente nao viaja. O sintoma foi
                # silencioso -- a matriz seguiu dizendo
                # `cortes_sem_intervalo_para_separar_queda_de_ruido` com o
                # intervalo ja calculado do outro lado da funcao.
                #
                # Vai como JSON porque e um par [lo, hi]: `None` continua vazio
                # na celula, e quem le sabe distinguir "nao calculado" de
                # "calculado e largo".
                "auc_ic95": (json.dumps(f["auc_ic95"])
                             if f.get("auc_ic95") is not None else None),
                "protocol_hash": r["protocol_hash"],
            })
    return pd.DataFrame(linhas)


def hash_da_matriz(resultados: list) -> str:
    return sha256_json([
        {k: r.get(k) for k in ("dataset", "modelo", "esquema", "auc",
                               "n_folds_validos", "protocol_hash")}
        for r in resultados])
