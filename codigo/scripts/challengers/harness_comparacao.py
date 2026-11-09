"""
Harness de comparacao de estimadores. Review-only, sem promocao.

-------------------------------------------------------------------------
O QUE ESTE ARQUIVO E, e o que ele NAO E
-------------------------------------------------------------------------

E um banco de provas: pega os MESMOS dados, a MESMA unidade independente e os
MESMOS folds que produziram o modelo servido, e roda estimadores diferentes em
cima disso. Serve para responder "o Firth esta deixando sinal na mesa?" com um
numero em vez de com opiniao.

NAO E um caminho de promocao. Nenhuma saida daqui pode virar modelo servido:

  * grava so em `local_runs/challengers-comparacao/`, nunca em
    `local_runs/svc-01-modelos/`;
  * o veredito e sempre `COMPARACAO_REVIEW_ONLY` -- nao existe `PROMOVIVEL`
    neste arquivo, nem como string;
  * `PORTAO_DE_ESCRITA` recusa qualquer destino dentro do diretorio servido, e
    o teste `test_challengers_harness.py` falha se essa trava sair.

A razao e a mesma de sempre no REV-P: trocar de estimador porque a AUC subiu
0,02 num desenho que nao foi feito para essa pergunta e como Curitiba foi
promovida uma vez -- por validacao que media outra coisa.

-------------------------------------------------------------------------
O DESENHO E REUSADO, e nao redecidido
-------------------------------------------------------------------------

A comparacao so vale se o desenho for identico. Entao nada aqui e inventado:

  UNIDADE   `grupo_nunca_mais_fino_que_a_celula` do SVC-01. E a regra U2 do
            projeto: nenhum grupo mais fino que a celula de 1e-4 grau, fundido
            por componente conexa. Sem isso, Recife reamostra LINHA e todo
            intervalo sai estreito demais por pseudo-replicacao.

  FOLDS     `GroupKFold` com o mesmo numero de splits do SVC-01, sobre o mesmo
            vetor de grupos. Validacao aleatoria nao entra aqui em hipotese
            nenhuma -- nem como referencia, porque numero que nao pode ser
            usado como prova nao deveria estar na tabela.

  FEATURES  as mesmas do modelo servido correspondente, na mesma ordem. Dar
            mais variaveis ao challenger e compara-lo com outra pergunta.

  EPV       o orcamento estrito do SVC-01 continua valendo. Um challenger que
            so ganha estourando o orcamento nao ganhou.

-------------------------------------------------------------------------
O QUE E MEDIDO, e por que a AUC sozinha nao decide
-------------------------------------------------------------------------

  auc_agrupada     discriminacao sob o desenho correto.
  brier            erro quadratico da probabilidade -- discriminacao e
                   calibracao juntas.
  calibracao       slope e intercepto da regressao do logito observado sobre o
                   predito. Slope longe de 1 e um modelo confiante demais (ou
                   de menos), e num produto que publica IC isso importa mais
                   que meio ponto de AUC.
  estabilidade     desvio-padrao da AUC entre folds. Um modelo que acerta em
                   media e erra feio em um fold nao e melhor: e mais arriscado.
  sinais_fisicos   so onde o estimador e interpretavel. Um ganho de AUC com
                   sinal fisico invertido nao e ganho -- e o mesmo defeito que
                   `variaveis_fisicamente_verificadas` existe para conter.

-------------------------------------------------------------------------
O QUE ESTE HARNESS NAO RESOLVE, e fica registrado
-------------------------------------------------------------------------

A comparacao roda sob o desenho do SVC-01 -- GroupKFold espacial-por-grupo.
Isso responde "qual estimador aproveita melhor este conjunto sob esta
validacao". NAO responde transferencia entre regioes nem holdout temporal, que
exigem desenho proprio (E3/E4) e decisao metodologica que nao esta neste
arquivo. Ver `EXPERIMENTOS_PROPOSTOS` no fim: eles ficam declarados como
proposta, e nao improvisados aqui.
"""

from __future__ import annotations


def _declarar_metadados_de_saida(dados):
    """Metadados de procedencia na copia serializada; nao modifica o objeto de calculo."""
    if not isinstance(dados, dict):
        return dados
    return {"tipo": "registro_harness_comparacao",
            "gerado_por": 'scripts/challengers/harness_comparacao.py',
            "nao_e": "Registro do produtor harness_comparacao: declarar a procedencia nao revalida resultados nem autoriza promocao; conservar os estados e limites especificos do artefato.",
            **dados}


import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))

from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "challengers-comparacao"
SERVIDO = REPO / "local_runs" / "svc-01-modelos"

VEREDITO = "COMPARACAO_REVIEW_ONLY"
SEED = 20260827


def PORTAO_DE_ESCRITA(destino: Path) -> Path:
    """Recusa qualquer destino dentro do diretorio de modelos servidos.

    Nao e paranoia: um `--saida` mal digitado seria o caminho mais curto entre
    "rodei uma comparacao" e "troquei o modelo do produto sem revisao". A
    trava e uma linha e fecha essa porta inteira.
    """
    destino = Path(destino).resolve()
    if destino == SERVIDO.resolve() or SERVIDO.resolve() in destino.parents:
        raise PermissionError(
            f"destino {destino} esta dentro de {SERVIDO}: este harness e "
            "review-only e nunca escreve em modelo servido")
    return destino


# ------------------------------------------------------------- challengers
#
# Cada challenger declara se e INTERPRETAVEL. Isso decide se `sinais_fisicos`
# tem valor para ele -- e um GBM sem sinal por variavel nao e um GBM ruim: e um
# GBM, e a tabela nao deve fingir que a coluna esta vazia por acaso.

class ChallengerFirth:
    """O estimador SERVIDO. Entra como referencia, nunca como concorrente.

    Sem ele na tabela, os outros numeros nao teriam contra o que ser lidos.
    """

    nome = "firth_servido"
    id_protocolo = "M0"
    interpretavel = True
    papel = "referencia (estimador servido)"

    def ajustar(self, X, y):
        from firthlogist import FirthLogisticRegression

        m = FirthLogisticRegression(max_iter=500, skip_pvals=True, skip_ci=True)
        m.fit(X, y)
        return m

    def prever(self, m, X):
        return m.predict_proba(X)[:, 1]

    def coeficientes(self, m):
        return list(np.asarray(m.coef_).ravel())


class ChallengerLogisticaL2:
    nome = "logistica_l2"
    id_protocolo = "M1"
    interpretavel = True
    papel = "challenger linear regularizado"

    def ajustar(self, X, y):
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=SEED)
        m.fit(X, y)
        return m

    def prever(self, m, X):
        return m.predict_proba(X)[:, 1]

    def coeficientes(self, m):
        return list(np.asarray(m.coef_).ravel())


class ChallengerGAMSpline:
    """Logistica sobre splines por variavel: nao linear e ainda aditivo.

    Aditivo importa: mantem "efeito de uma variavel" como uma coisa que existe,
    o que um modelo com interacao livre perde. O sinal fisico e lido pela
    inclinacao media da funcao ajustada, e nao por um coeficiente -- e por isso
    `coeficientes` devolve `None`: nao ha um.
    """

    nome = "gam_spline_aditivo"
    id_protocolo = "M2"
    interpretavel = True
    papel = "challenger nao linear aditivo"

    def __init__(self, n_nos: int = 4):
        self.n_nos = n_nos

    def ajustar(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer

        # Poucos nos de proposito: o orcamento EPV do projeto e apertado, e um
        # spline com muitos nos gasta graus de liberdade que o conjunto nao
        # tem. Um challenger que so ganha por flexibilidade que o EPV nao
        # comporta nao ganhou.
        m = make_pipeline(
            SplineTransformer(n_knots=self.n_nos, degree=3,
                              include_bias=False),
            LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=SEED))
        m.fit(X, y)
        return m

    def prever(self, m, X):
        return m.predict_proba(X)[:, 1]

    def coeficientes(self, m):
        return None


class ChallengerInteracoes:
    """Linear com POUCAS interacoes pre-especificadas, e nenhuma descoberta.

    A diferenca entre este challenger e uma busca por interacao e a ordem das
    decisoes. Os pares vem de `INTERACOES_PRE_ESPECIFICADAS`, escritos no
    protocolo com justificativa fisica ANTES de qualquer resultado. Testar
    todos os pares e ficar com o melhor produziria, com seis variaveis, quinze
    candidatos e quase certamente um "achado" -- e nao haveria como saber
    depois se ele foi encontrado ou procurado.

    O termo de interacao entra como produto das variaveis JA padronizadas, e
    nao como produto das brutas: sem isso o termo carrega a escala de duas
    variaveis e domina a penalizacao sozinho.
    """

    nome = "interacoes_pre_especificadas"
    interpretavel = True
    papel = "challenger linear com interacoes fisicamente justificadas"
    id_protocolo = "M3"

    def __init__(self, features: list, pares: list | None = None):
        from protocolo_experimental import INTERACOES_PRE_ESPECIFICADAS

        self.features = list(features)
        declarados = pares if pares is not None else [
            tuple(i["par"]) for i in INTERACOES_PRE_ESPECIFICADAS]
        # so pares cujas DUAS variaveis existem neste conjunto
        self.pares = [(a, b) for a, b in declarados
                      if a in self.features and b in self.features]
        self.indices = [(self.features.index(a), self.features.index(b))
                        for a, b in self.pares]

    def _expandir(self, X):
        if not self.indices:
            return X
        extras = [X[:, i] * X[:, j] for i, j in self.indices]
        return np.column_stack([X] + extras)

    def ajustar(self, X, y):
        from firthlogist import FirthLogisticRegression

        m = FirthLogisticRegression(max_iter=500, skip_pvals=True, skip_ci=True)
        m.fit(self._expandir(X), y)
        return m

    def prever(self, m, X):
        return m.predict_proba(self._expandir(X))[:, 1]

    def coeficientes(self, m):
        # so os coeficientes das variaveis ORIGINAIS entram na checagem de
        # sinal fisico: o sinal esperado de `hand_m x rain_max_24h` nao esta
        # definido no projeto, e inventar um seria criar expectativa fisica
        # para caber no resultado.
        return list(np.asarray(m.coef_).ravel()[:len(self.features)])

    def nomes_das_interacoes(self) -> list:
        return [f"{a}_x_{b}" for a, b in self.pares]


class ChallengerGBMMonotonico:
    """Arvore com restricao de monotonicidade no sinal fisico esperado.

    A restricao nao e ajuste fino: e o que separa "modelo flexivel" de "modelo
    flexivel que respeita a fisica". Sem ela, o GBM aprende a inverter HAND em
    parte do dominio e paga isso com AUC melhor e explicacao impossivel.
    """

    nome = "gbm_monotonico"
    id_protocolo = "M4"
    interpretavel = False
    papel = "challenger nao linear com restricao fisica"

    def __init__(self, sinais: list | None = None):
        self.sinais = sinais

    def ajustar(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier

        restricao = self.sinais if self.sinais else [0] * X.shape[1]
        m = HistGradientBoostingClassifier(
            max_iter=200, max_depth=3, learning_rate=0.05,
            monotonic_cst=restricao, random_state=SEED)
        m.fit(X, y)
        return m

    def prever(self, m, X):
        return m.predict_proba(X)[:, 1]

    def coeficientes(self, m):
        return None


# ------------------------------------------------------------------ metricas

def _calibracao(y, p) -> dict:
    """Slope e intercepto da recalibracao logistica de Cox.

    Slope 1 e intercepto 0 e calibracao perfeita. Slope < 1 significa
    probabilidades espalhadas demais -- confianca excessiva, que num produto
    que publica IC e pior que meio ponto de AUC a menos.
    """
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logito = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return {"slope": None, "intercepto": None,
                "motivo": "uma classe so no conjunto avaliado"}
    m = LogisticRegression(max_iter=1000, penalty=None)
    m.fit(logito, y)
    return {"slope": round(float(m.coef_[0][0]), 4),
            "intercepto": round(float(m.intercept_[0]), 4)}


def _sinais_fisicos(coefs, feats: list, esperados: dict) -> dict:
    if coefs is None:
        return {"disponivel": False,
                "motivo": "estimador nao expoe efeito por variavel"}
    saida = {}
    for f, c in zip(feats, coefs):
        alvo = esperados.get(f)
        saida[f] = {
            "coef": round(float(c), 4),
            "sinal_esperado": alvo,
            "sinal_bate": None if alvo is None else bool(np.sign(c) == np.sign(alvo)),
        }
    return {"disponivel": True, "por_variavel": saida,
            "n_batem": sum(1 for v in saida.values() if v["sinal_bate"]),
            "n_com_expectativa": sum(1 for v in saida.values()
                                     if v["sinal_esperado"] is not None)}


def avaliar_challenger(challenger, X, y, grupos, feats: list,
                       esperados: dict, n_folds: int) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    p = np.full(len(y), np.nan)
    aucs_por_fold, folds_usados = [], 0
    for i_tr, i_te in GroupKFold(n_splits=n_folds).split(X, y, grupos):
        if len(np.unique(y[i_tr])) < 2 or len(np.unique(y[i_te])) < 2:
            continue
        mu = X[i_tr].mean(0)
        sd = np.where(X[i_tr].std(0) == 0, 1, X[i_tr].std(0))
        modelo = challenger.ajustar((X[i_tr] - mu) / sd, y[i_tr])
        p[i_te] = challenger.prever(modelo, (X[i_te] - mu) / sd)
        folds_usados += 1
        aucs_por_fold.append(float(roc_auc_score(y[i_te], p[i_te])))

    ok = ~np.isnan(p)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return {"estimador": challenger.nome, "papel": challenger.papel,
                "auc_agrupada": None,
                "motivo": "nenhum fold com as duas classes"}

    mu = X.mean(0)
    sd = np.where(X.std(0) == 0, 1, X.std(0))
    modelo_cheio = challenger.ajustar((X - mu) / sd, y)
    coefs = challenger.coeficientes(modelo_cheio)

    return {
        "estimador": challenger.nome,
        "papel": challenger.papel,
        "interpretavel": challenger.interpretavel,
        "folds_usados": folds_usados,
        "auc_agrupada": round(float(roc_auc_score(y[ok], p[ok])), 4),
        "auc_por_fold": [round(a, 4) for a in aucs_por_fold],
        # Um modelo que acerta em media e erra feio num fold nao e melhor:
        # e mais arriscado, e o produto responde area por area.
        "estabilidade_desvio_auc": (round(float(np.std(aucs_por_fold)), 4)
                                    if len(aucs_por_fold) > 1 else None),
        "brier": round(float(brier_score_loss(y[ok], p[ok])), 4),
        "calibracao": _calibracao(y[ok], p[ok]),
        "sinais_fisicos": _sinais_fisicos(coefs, feats, esperados),
    }


# -------------------------------------------------------------- conjuntos

def conjuntos_do_projeto() -> list:
    """Reconstroi os conjuntos do SVC-01, sem redecidir nada.

    Le a MESMA tabela, aplica a MESMA admissao e a MESMA fusao de grupos. Se o
    SVC-01 mudar, isto muda junto -- que e o comportamento certo: um harness
    que congela sua propria copia do conjunto compara com um passado.
    """
    import pandas as pd

    import svc01_construir_modelos_servidos as svc01

    if not svc01.POOL.exists() or not svc01.UNICA.exists():
        return []

    pool = pd.read_parquet(svc01.POOL)
    pool = pool[pool.classe.isin([0, 1])
                & pool.nivel_negativo.isin(svc01.NEGATIVO_ADMITIDO)].copy()
    unica = pd.read_parquet(svc01.UNICA)
    unica = unica[(unica.admitido == True)  # noqa: E712
                  & unica.classe.isin([0, 1])].copy()

    saida = []
    rec = unica[unica.fonte == "recife"].dropna(
        subset=svc01.TOPO + svc01.CHUVA + ["grupo_cv", "lat", "lon"])
    saida.append(("recife_pluvial", rec, svc01.TOPO + svc01.CHUVA))

    pla = pool[(pool.classe_relevo == "PLANO_OU_ONDULADO")
               & (~pool.fonte.isin(svc01.REGIOES_ALVO))].dropna(
                   subset=svc01.TOPO + ["grupo_cv", "lat", "lon"])
    saida.append(("fluvial_planicie", pla, svc01.TOPO))

    ser = pool[(pool.classe_relevo == "INGREME")
               & (~pool.fonte.isin(svc01.REGIOES_ALVO))].dropna(
                   subset=svc01.TOPO + ["grupo_cv", "lat", "lon"])
    orc = svc01.orcamento_estrito(ser)
    feats = (svc01.CAUSAIS if orc >= 2 else ["hand_m"])[:max(orc, 1)]
    saida.append(("fluvial_serra", ser, feats))
    return [(nome, s, f) for nome, s, f in saida if not s.empty]


def comparar_conjunto(nome: str, s, feats: list) -> dict:
    import svc01_construir_modelos_servidos as svc01

    grupos = svc01.grupo_nunca_mais_fino_que_a_celula(s).to_numpy()
    X = s[feats].to_numpy(dtype=float)
    y = s.classe.to_numpy().astype(int)
    n_folds = min(5, len(np.unique(grupos)))
    esperados = {f: svc01.SINAL_FISICO.get(f) for f in feats}
    restricao = [int(np.sign(esperados.get(f) or 0)) for f in feats]

    if n_folds < 2:
        return {"conjunto": nome, "features": feats,
                "bloqueado": "grupos insuficientes para GroupKFold"}

    challengers = [
        ChallengerFirth(),
        ChallengerLogisticaL2(),
        ChallengerGAMSpline(),
        ChallengerGBMMonotonico(restricao),
    ]
    resultados = []
    for c in challengers:
        try:
            resultados.append(
                avaliar_challenger(c, X, y, grupos, feats, esperados, n_folds))
        except Exception as e:
            # Challenger que quebra e um DADO, e nao um motivo para a tabela
            # inteira sumir. Some da comparacao com o motivo escrito.
            resultados.append({"estimador": c.nome, "papel": c.papel,
                               "auc_agrupada": None,
                               "erro": f"{type(e).__name__}: {e}"})
    return {
        "conjunto": nome,
        "features": feats,
        "n_linhas": int(len(s)),
        "n_grupos": int(len(np.unique(grupos))),
        "grupos_por_classe": {
            str(k): int(v) for k, v in
            s.assign(_g=grupos).groupby("classe")._g.nunique().items()},
        "epv": round(float(min(
            s.assign(_g=grupos).groupby("classe")._g.nunique()) / max(1, len(feats))), 2),
        "n_folds": int(n_folds),
        "desenho": ("GroupKFold sobre componente conexa de grupo e celula; "
                    "nenhuma validacao aleatoria"),
        "resultados": resultados,
    }


# ------------------------------------------------------ experimentos abertos
#
# Escritos como PROPOSTA, e nao rodados. Cada um exige uma decisao metodologica
# que nao esta no repositorio hoje; improvisar a decisao aqui produziria um
# numero que alguem leria depois como resultado.

EXPERIMENTOS_PROPOSTOS = [
    {
        "id": "CH-E1",
        "pergunta": "algum challenger transfere melhor entre regioes?",
        "por_que_nao_rodou": (
            "exige desenho de transferencia (treinar fora da regiao alvo e "
            "avaliar nela) e uma referencia local para a regiao alvo. "
            "Curitiba tem colapso temporal conhecido e Petropolis nao tem "
            "referencia local adequada; sem resolver isso antes, a metrica "
            "mediria a falta de rotulo, e nao o estimador"),
        "o_que_falta": "definir a unidade de avaliacao na regiao alvo",
    },
    {
        "id": "CH-E2",
        "pergunta": "algum challenger degrada menos em holdout temporal?",
        "por_que_nao_rodou": (
            "o holdout temporal fechado do projeto (E4) usa cortes por data "
            "com IC de grupos, e nem todo conjunto aqui tem data utilizavel "
            "-- a data do bloco negativo e artefato conhecido. Rodar o corte "
            "sobre uma data artefactual produziria uma curva de degradacao "
            "que nao existe"),
        "o_que_falta": "declarar quais conjuntos tem data admissivel",
    },
    {
        "id": "CH-E3",
        "pergunta": "o ganho nao linear sobrevive ao orcamento EPV?",
        "por_que_nao_rodou": (
            "o EPV conta variaveis, e um spline gasta mais graus de liberdade "
            "que uma variavel linear. Nao ha no repositorio uma convencao de "
            "quantos EPV um termo nao linear custa; escolher uma agora seria "
            "escolher o resultado"),
        "o_que_falta": "convencao de custo de EPV para termo nao linear",
    },
]


def comparar(saida: Path = SAIDA) -> dict:
    saida = PORTAO_DE_ESCRITA(saida)
    conjuntos = conjuntos_do_projeto()
    comparacoes = [comparar_conjunto(nome, s, feats)
                   for nome, s, feats in conjuntos]
    relatorio = {
        "tipo": "challengers_comparacao_review_only",
        "quando_utc": agora_utc(),
        "veredito": VEREDITO,
        "seed": SEED,
        "desenho_reusado_de": "scripts/servico/svc01_construir_modelos_servidos.py",
        "conjuntos": comparacoes,
        "experimentos_propostos": EXPERIMENTOS_PROPOSTOS,
        "nao_promove": (
            "nenhum resultado aqui muda modelo servido, maturidade ou contrato; "
            "trocar o estimador do produto exige protocolo proprio e revisao "
            "humana"),
        "hash_do_desenho": sha256_json(
            [{"conjunto": c.get("conjunto"), "features": c.get("features"),
              "n_folds": c.get("n_folds")} for c in comparacoes]),
    }
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "resultado.json").write_text(
        json.dumps(_declarar_metadados_de_saida(relatorio), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    (saida / "RELATORIO.md").write_text(_markdown(relatorio), encoding="utf-8")
    return relatorio


def _markdown(r: dict) -> str:
    linhas = ["# Challengers - comparacao review-only", "",
              f"Veredito: `{r['veredito']}`.", "",
              f"Desenho reusado de `{r['desenho_reusado_de']}`.", "",
              f"> {r['nao_promove']}", ""]
    for c in r["conjuntos"]:
        linhas += [f"## {c['conjunto']}", ""]
        if c.get("bloqueado"):
            linhas += [f"BLOQUEADO: {c['bloqueado']}", ""]
            continue
        linhas += [
            f"Features: `{', '.join(c['features'])}` · grupos {c['n_grupos']} · "
            f"EPV {c['epv']} · folds {c['n_folds']}", "",
            "| estimador | papel | AUC agrupada | desvio entre folds | Brier | "
            "slope calib. | sinais fisicos |",
            "|---|---|---|---|---|---|---|"]
        for x in c["resultados"]:
            sf = x.get("sinais_fisicos") or {}
            if sf.get("disponivel"):
                sinal = f"{sf['n_batem']}/{sf['n_com_expectativa']} batem"
            else:
                sinal = "nao interpretavel"
            cal = (x.get("calibracao") or {}).get("slope")
            linhas.append(
                f"| `{x['estimador']}` | {x.get('papel', '')} | "
                f"{x.get('auc_agrupada')} | {x.get('estabilidade_desvio_auc')} | "
                f"{x.get('brier')} | {cal} | {sinal} |")
        linhas.append("")
    linhas += ["## Experimentos propostos, nao executados", ""]
    for e in r["experimentos_propostos"]:
        linhas += [f"### {e['id']} — {e['pergunta']}", "",
                   f"- **por que nao rodou:** {e['por_que_nao_rodou']}",
                   f"- **o que falta:** {e['o_que_falta']}", ""]
    return "\n".join(linhas)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saida", default=str(SAIDA))
    args = parser.parse_args()
    r = comparar(Path(args.saida))
    print(json.dumps({
        "veredito": r["veredito"],
        "conjuntos": [c["conjunto"] for c in r["conjuntos"]],
        "hash_do_desenho": r["hash_do_desenho"],
        "experimentos_propostos": [e["id"] for e in r["experimentos_propostos"]],
        "saida": args.saida,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
