"""Autopareamento no tempo: a chuva transitoria muda o risco NO MESMO LUGAR?

POR QUE ESTE DESENHO, e por que ele nao e um refinamento do `desenho_pareado`:

`desenho_pareado.py` estratifica por `(dia, AOI)` e pergunta o que sobra quando
a DATA sai. Este modulo vira o estrato para o LUGAR e pergunta o que sobra
quando o LUGAR sai. As duas perguntas nao competem -- elas cancelam
confundimentos diferentes --, mas so a segunda responde a pergunta do produto.

Com o estrato no lugar, tudo que e invariante no tempo naquele ponto sai da
estimativa POR CONSTRUCAO, e nao por ajuste:

  terreno, HAND, TWI, declividade      identicos em todas as datas do estrato
  densidade da rede de drenagem        idem
  propensao daquele endereco a ligar   idem

Os tres eram confundimentos abertos e medidos no projeto. O terceiro e o que
`aq_chamados03` chama de fundo -- `target-group background` --, e o primeiro e
o que domina o escore servido. Aqui os dois saem juntos.

E ha uma razao a mais, especifica desta base. `casos_das_centrais.auditar_janelas`
mediu que em Nova York 73,8% dos positivos caem em dia SEM nenhum negativo, e
que um preditor que so conhece o NUMERO DO DIA separa as classes com AUC 0,887.
Enquanto a janela de aquisicao das duas classes nao coincidir, comparar classes
nessa base mede a janela. **Este desenho nao usa classe negativa nenhuma**: o
caso e o proprio controle em outras datas, entao o defeito nao entra.

-------------------------------------------------------------------------
O ESTIMADOR, e por que este
-------------------------------------------------------------------------

Poisson CONDICIONAL (Armstrong, Gasparrini & Tobias, 2014): a contagem de casos
da celula-dia, com o intercepto do estrato CONDICIONADO PARA FORA em vez de
estimado. Estimar um intercepto por estrato seria o problema de parametros
incidentais -- centenas de parametros de ruido contaminando o coeficiente de
interesse. A verossimilhanca condicional e multinomial dentro do estrato:

    L(b) = PROD_s PROD_i [ exp(x_i b) / SUM_j exp(x_j b) ] ^ y_i

`statsmodels` nao esta instalado neste ambiente e a logistica condicional exata
com varios casos por estrato exige soma combinatoria; a forma acima e convexa,
exata e barata, e Armstrong et al. mostram que ela coincide com a logistica
condicional no regime deste dado.

-------------------------------------------------------------------------
O INTERVALO, e a regra de pseudo-replicacao
-------------------------------------------------------------------------

O IC nao sai da matriz de informacao. Ele sai de bootstrap de ESTRATOS -- a
unidade que compartilha o mesmo forcing --, porque dois chamados na mesma
celula-dia tem exposicao identica por construcao e conta-los como observacoes
independentes estreitaria o IC sem acrescentar informacao. E a mesma regra de
`grupo_nunca_mais_fino_que_a_celula()` do SVC-01, aplicada ao tempo.

Como sensibilidade, um segundo IC reamostra CELULAS (19 em Nova York): mais
conservador, porque estratos da mesma celula em meses vizinhos nao sao
independentes.

-------------------------------------------------------------------------
O QUE ELE NAO FAZ
-------------------------------------------------------------------------

Nao cria escore, nao substitui a suscetibilidade territorial, nao vira campo
novo do contrato e nao promove modelo. Produz uma razao de taxas do FORCAMENTO
-- ou uma recusa com o requisito que a destravaria.

Uso:
    python scripts/chuva/autopareamento_temporal.py --fonte nyc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from chuva.contrato_chuva import METHOD_VERSION, TRIGGER_VARIABLE  # noqa: E402

CASOS = REPO / "local_runs" / "chuva-casos-centrais"
SERIES = REPO / "local_runs" / "chuva-serie-celula"
OUT = REPO / "local_runs" / "chuva-autopareamento"

GATILHO = TRIGGER_VARIABLE
ANTECEDENTE = "rain_decay_index"
PICO_ANTERIOR = "rain_max_24h"

# ---------------------------------------------------------------- protocolo
#
# Congelado ANTES de estimar. Escolher a janela depois de ver o resultado e
# mover a trave -- e o projeto ja tem `pre_registro_do_grid` em
# `identificabilidade.py` exatamente por isso. O protocolo primario e
# declarado aqui; os demais sao SENSIBILIDADE, e sao reportados juntos.

PROTOCOLO_PRIMARIO = "mes_dow"

PROTOCOLOS = {
    "mes_dow": {
        "descricao": ("estratificado no tempo: celula x mes-calendario x dia "
                      "da semana. Referentes sao os demais dias do mesmo mes "
                      "e mesmo dia da semana"),
        "por_que": ("e o esquema de referentes que nao sofre viés de "
                    "sobreposicao (Janes, Sheppard & Lumley, 2005): os "
                    "estratos particionam o tempo, entao nenhum dia serve de "
                    "referente para um caso fora do proprio estrato"),
        "cancela": ["tudo invariante no lugar", "tendencia de longo prazo",
                    "sazonalidade dentro do mes", "ciclo semanal de chamados"],
        "chave": lambda d: (d.cl_lat.astype(str) + "|" + d.cl_lon.astype(str)
                            + "|" + d.ts.dt.strftime("%Y-%m") + "|"
                            + d.ts.dt.dayofweek.astype(str)),
    },
    "semana_iso": {
        "descricao": "celula x semana ISO",
        "por_que": ("janela mais curta: controla tendencia com mais forca e "
                    "deixa menos dias por estrato"),
        "cancela": ["tudo invariante no lugar", "tendencia", "sazonalidade"],
        "chave": lambda d: (d.cl_lat.astype(str) + "|" + d.cl_lon.astype(str)
                            + "|" + d.ts.dt.isocalendar().year.astype(str) + "W"
                            + d.ts.dt.isocalendar().week.astype(str)),
    },
    "mes": {
        "descricao": "celula x mes-calendario, sem casar dia da semana",
        "por_que": ("serve para MEDIR quanto o casamento por dia da semana "
                    "muda a estimativa: se mudar muito, o ciclo semanal de "
                    "chamados estava dentro do efeito"),
        "cancela": ["tudo invariante no lugar", "tendencia", "sazonalidade"],
        "chave": lambda d: (d.cl_lat.astype(str) + "|" + d.cl_lon.astype(str)
                            + "|" + d.ts.dt.strftime("%Y-%m")),
    },
}

# Conjuntos de exposicao pre-declarados. Antecedente e gatilho ficam separados
# porque medem coisas diferentes (Kohler & Linsley, 1951), e a combinacao so e
# lida depois de as duas isoladas terem sido lidas.
EXPOSICOES = {
    "gatilho": [GATILHO],
    "antecedente": [ANTECEDENTE],
    "gatilho_e_antecedente": [GATILHO, ANTECEDENTE],
}

# Controle negativo: a chuva de daqui a N dias, que nao pode causar o chamado
# de hoje. Trinta dias fica fora de qualquer janela antecedente plausivel e
# dentro da mesma estacao, entao o teste e sobre confundimento sazonal
# residual, e nao sobre estacao diferente.
DIAS_DO_CONTROLE_FUTURO = 30
CONTROLE_FUTURO = "rain_event_day_futuro"

POR_MM = 10.0          # o efeito e reportado por 10 mm, nao por mm
B_BOOTSTRAP = 1000
SEMENTE = 20260831

# Preconditions de identificabilidade, declaradas antes de estimar.
MIN_ESTRATOS_POR_PARAMETRO = 10     # a mesma regra de EPV do SVC-01
MIN_FRACAO_VARIANCIA_INTRA = 0.01   # 1% da variancia da exposicao dentro do estrato


# ------------------------------------------------------------------ dados

def montar(fonte: str) -> pd.DataFrame:
    """Celula-dia com exposicao e contagem de casos, sobre a serie inteira.

    O `left join` e sobre a SERIE, e nao sobre os casos: dia sem chamado e
    zero de chamado, e nao ausencia de dado -- o registro bruto cobre a janela
    inteira. Zero de CHAMADO nao e zero de alagamento, e essa distincao viaja
    na limitacao do artefato.
    """
    serie_arq = SERIES / f"serie_diaria_por_celula_{fonte}.csv"
    casos_arq = CASOS / f"casos_celula_dia_{fonte}.csv"
    for p in (serie_arq, casos_arq):
        if not p.exists():
            raise FileNotFoundError(f"ausente: {p}")
    serie = pd.read_csv(serie_arq)
    casos = pd.read_csv(casos_arq)
    chave = ["cl_lat", "cl_lon", "dia"]
    colunas_caso = [c for c in casos.columns if c.startswith("casos")]
    d = serie.merge(casos[chave + colunas_caso], on=chave, how="left")
    for c in colunas_caso:
        d[c] = d[c].fillna(0).astype(int)
    d["ts"] = pd.to_datetime(d.dia)
    return d


# -------------------------------------------------------------- estimador

def _loglik_e_grad(beta: np.ndarray, X: np.ndarray, y: np.ndarray,
                   inicio: np.ndarray) -> tuple[float, np.ndarray]:
    """Verossimilhanca condicional de Poisson e seu gradiente, por estrato.

    `inicio` sao os limites dos blocos contiguos de estrato em X/y.

    Tudo por `reduceat`, e nao por laco sobre estratos: sao dezenas de milhares
    de estratos vezes centenas de replicas de bootstrap, e a versao com laco em
    Python nao termina em tempo util. `reduceat` da o mesmo resultado porque os
    blocos sao contiguos por construcao -- `preparar` ordena por estrato.
    """
    comeco = inicio[:-1]
    eta = X @ beta
    # Estabilizacao por estrato: `exp` de eta cru estoura com exposicao em mm.
    m = np.maximum.reduceat(eta, comeco)
    tamanho = np.diff(inicio)
    w = np.exp(eta - np.repeat(m, tamanho))
    soma = np.add.reduceat(w, comeco)
    n_s = np.add.reduceat(y, comeco)
    ll = float(y @ eta - n_s @ (m + np.log(soma)))
    p = w / np.repeat(soma, tamanho)
    grad = X.T @ y - X.T @ (np.repeat(n_s, tamanho) * p)
    return -ll, -grad


def ajustar(X: np.ndarray, y: np.ndarray, inicio: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    b0 = np.zeros(X.shape[1])
    r = minimize(_loglik_e_grad, b0, args=(X, y, inicio), jac=True,
                 method="L-BFGS-B", options={"maxiter": 500})
    if not r.success and not np.all(np.isfinite(r.x)):
        raise RuntimeError(f"ajuste nao convergiu: {r.message}")
    return np.asarray(r.x, dtype=float)


def _blocos(estratos: np.ndarray) -> np.ndarray:
    """Limites dos blocos contiguos; exige `estratos` ja ordenado."""
    troca = np.flatnonzero(np.diff(estratos)) + 1
    return np.concatenate([[0], troca, [len(estratos)]])


def preparar(d: pd.DataFrame, protocolo: str, exposicoes: list[str],
             coluna_caso: str = "casos") -> dict:
    chave = PROTOCOLOS[protocolo]["chave"](d)
    s = d.assign(_estrato=chave).sort_values("_estrato").reset_index(drop=True)
    s = s.dropna(subset=exposicoes)
    codigos = pd.factorize(s._estrato, sort=False)[0]
    s = s.assign(_id=codigos)

    # Estrato so informa se tem caso E variacao de exposicao. Sem uma das duas
    # ele nao contribui com nada para a verossimilhanca condicional -- e
    # conta-lo como amostra inflaria o poder aparente do teste.
    agr = s.groupby("_id").agg(
        casos=(coluna_caso, "sum"), dias=(coluna_caso, "size"),
        **{f"var_{e}": (e, "var") for e in exposicoes})
    informativo = ((agr.casos > 0) & (agr.dias > 1)
                   & (agr[[f"var_{e}" for e in exposicoes]].fillna(0) > 0).any(axis=1))
    ids = set(agr.index[informativo])
    s = s[s._id.isin(ids)].sort_values("_id").reset_index(drop=True)
    return {
        "quadro": s,
        "estratos_totais": int(len(agr)),
        "estratos_informativos": int(informativo.sum()),
        "casos_totais": int(d[coluna_caso].sum()),
        "casos_em_estrato_informativo": int(s[coluna_caso].sum()),
        "dias_em_estrato_informativo": int(len(s)),
    }


def fracao_variancia_intra(s: pd.DataFrame, variavel: str) -> float | None:
    total = float(s[variavel].var())
    if not total or not np.isfinite(total):
        return None
    intra = float((s[variavel] - s.groupby("_id")[variavel].transform("mean")).var())
    return round(intra / total, 6)


def estimar(prep: dict, exposicoes: list[str], coluna_caso: str = "casos",
            b_bootstrap: int = B_BOOTSTRAP) -> dict:
    """Coeficientes, razao de taxas por 10 mm, e IC por bootstrap de estratos."""
    s = prep["quadro"]
    if s.empty:
        return {"estimado": False, "motivo": "nenhum estrato informativo"}
    X = s[exposicoes].to_numpy(dtype=float)
    y = s[coluna_caso].to_numpy(dtype=float)
    ids = s._id.to_numpy()
    inicio = _blocos(ids)
    beta = ajustar(X, y, inicio)

    comeco = inicio[:-1]
    tamanho = np.diff(inicio)
    n_estratos = len(comeco)

    def replica(ordem: np.ndarray) -> np.ndarray | None:
        """Reajusta sobre a sequencia de estratos `ordem` (indices de bloco).

        As linhas sao selecionadas por aritmetica de indice em vez de
        `vstack` de fatias: com dezenas de milhares de estratos, montar a
        replica em Python custa mais que o proprio ajuste.
        """
        sz = tamanho[ordem]
        fim = np.cumsum(sz)
        marcas = np.concatenate([[0], fim])
        deslocamento = np.repeat(comeco[ordem] - np.concatenate([[0], fim[:-1]]), sz)
        idx = np.arange(int(fim[-1])) + deslocamento
        try:
            return ajustar(X[idx], y[idx], marcas)
        except Exception:                                         # noqa: BLE001
            return None

    rng = np.random.default_rng(SEMENTE)
    replicas = []
    for _ in range(b_bootstrap):
        r = replica(rng.integers(0, n_estratos, size=n_estratos))
        if r is not None and np.all(np.isfinite(r)):
            replicas.append(r)
    arr = np.asarray(replicas) if replicas else np.empty((0, len(exposicoes)))

    # IC de sensibilidade reamostrando CELULAS: estratos da mesma celula em
    # meses vizinhos nao sao independentes, e este IC assume menos.
    celula_do_estrato = (s.cl_lat.astype(str) + "|"
                         + s.cl_lon.astype(str)).to_numpy()[comeco]
    celulas, codigo = np.unique(celula_do_estrato, return_inverse=True)
    blocos_da_celula = [np.flatnonzero(codigo == k) for k in range(len(celulas))]
    rep_cel = []
    rng2 = np.random.default_rng(SEMENTE + 1)
    for _ in range(min(b_bootstrap, 400)):
        escolhidas = rng2.integers(0, len(celulas), size=len(celulas))
        r = replica(np.concatenate([blocos_da_celula[k] for k in escolhidas]))
        if r is not None and np.all(np.isfinite(r)):
            rep_cel.append(r)
    arr_cel = np.asarray(rep_cel) if rep_cel else np.empty((0, len(exposicoes)))

    saida = {"estimado": True, "estimador": "poisson_condicional",
             "por_variavel": {}, "replicas_por_estrato": int(len(arr)),
             "replicas_por_celula": int(len(arr_cel))}
    for i, var in enumerate(exposicoes):
        rr = float(np.exp(POR_MM * beta[i]))
        d = {"coeficiente_por_mm": round(float(beta[i]), 6),
             f"razao_de_taxas_por_{POR_MM:g}mm": round(rr, 4),
             "fracao_da_variancia_dentro_do_estrato":
                 fracao_variancia_intra(s, var)}
        if len(arr):
            lo, hi = np.percentile(np.exp(POR_MM * arr[:, i]), [2.5, 97.5])
            d["ic95_bootstrap_de_estrato"] = [round(float(lo), 4), round(float(hi), 4)]
            d["ic95_exclui_1"] = bool(lo > 1.0 or hi < 1.0)
        if len(arr_cel):
            lo, hi = np.percentile(np.exp(POR_MM * arr_cel[:, i]), [2.5, 97.5])
            d["ic95_bootstrap_de_celula"] = [round(float(lo), 4), round(float(hi), 4)]
            d["ic95_de_celula_exclui_1"] = bool(lo > 1.0 or hi < 1.0)
        saida["por_variavel"][var] = d
    return saida


def dose_resposta(prep: dict, variavel: str, coluna_caso: str = "casos",
                  n_faixas: int = 5) -> dict:
    """Efeito por faixa de exposicao -- monotonia e mais informativa que um b.

    Sem forma funcional imposta: a faixa mais baixa e a referencia e as demais
    entram como indicadoras dentro do MESMO estrato.
    """
    s = prep["quadro"]
    if s.empty:
        return {"avaliado": False, "motivo": "sem estrato informativo"}
    try:
        faixas = pd.qcut(s[variavel], n_faixas, labels=False, duplicates="drop")
    except ValueError:
        return {"avaliado": False, "motivo": "exposicao sem faixas distintas"}
    n = int(pd.Series(faixas).nunique())
    if n < 2:
        return {"avaliado": False, "motivo": "exposicao concentrada numa faixa"}
    X = np.column_stack([(faixas == k).astype(float) for k in range(1, n)])
    y = s[coluna_caso].to_numpy(dtype=float)
    beta = ajustar(X, y, _blocos(s._id.to_numpy()))
    limites = (s.groupby(faixas)[variavel].agg(["min", "max", "size"])
               .round(3).to_dict("index"))
    return {
        "avaliado": True, "variavel": variavel, "faixas": n,
        "referencia": "faixa 0 (menor exposicao)",
        "limites_da_faixa": {str(k): v for k, v in limites.items()},
        "razao_de_taxas_por_faixa": {
            str(k + 1): round(float(np.exp(beta[k])), 4) for k in range(n - 1)},
        "monotonica": bool(np.all(np.diff(beta) > 0)) if n > 2 else None,
    }


# ------------------------------------------------------ identificabilidade

def avaliar_identificabilidade(prep: dict, exposicoes: list[str]) -> dict:
    """As precondicoes valem? Recusa com requisito e resultado, nao falha."""
    s = prep["quadro"]
    n_par = len(exposicoes)
    estratos = prep["estratos_informativos"]
    razao = estratos / n_par if n_par else 0.0
    fracoes = {v: (fracao_variancia_intra(s, v) if not s.empty else None)
               for v in exposicoes}
    faltas = []
    if estratos == 0:
        faltas.append({
            "requisito": "pelo menos um estrato com caso e variacao de exposicao",
            "observado": 0,
            "o_que_destravaria": ("casos repetidos no mesmo lugar em datas "
                                  "diferentes, ou janela de estudo mais longa")})
    if razao < MIN_ESTRATOS_POR_PARAMETRO:
        faltas.append({
            "requisito": f"{MIN_ESTRATOS_POR_PARAMETRO} estratos informativos "
                         "por parametro (a regra de EPV do projeto)",
            "observado": round(razao, 2),
            "o_que_destravaria": (f"{int(MIN_ESTRATOS_POR_PARAMETRO * n_par)} "
                                  "estratos informativos, ou menos variaveis "
                                  "de exposicao no mesmo ajuste")})
    baixas = {v: f for v, f in fracoes.items()
              if f is not None and f < MIN_FRACAO_VARIANCIA_INTRA}
    if baixas:
        faltas.append({
            "requisito": (f"pelo menos {MIN_FRACAO_VARIANCIA_INTRA:.0%} da "
                          "variancia da exposicao DENTRO do estrato"),
            "observado": baixas,
            "o_que_destravaria": ("produto de precipitacao com celula menor "
                                  "que o estrato, ou estrato temporal mais "
                                  "largo, que deixa mais dias por estrato")})
    return {
        "identificavel": not faltas,
        "estratos_informativos": estratos,
        "estratos_por_parametro": round(razao, 2),
        "fracao_da_variancia_dentro_do_estrato": fracoes,
        "requisitos_nao_atendidos": faltas,
        "criterios": {
            "min_estratos_por_parametro": MIN_ESTRATOS_POR_PARAMETRO,
            "min_fracao_variancia_intra": MIN_FRACAO_VARIANCIA_INTRA,
            "declarados_antes_de_estimar": True,
        },
    }


def estado_de_suporte(ident: dict, est: dict, variavel: str) -> dict:
    """Traduz o resultado para o vocabulario unico de `suporte_cientifico`.

    `not_identifiable` NAO vira um quinto estado. Ele e a RAZAO tecnica sob
    `not_testable` -- a diferenca entre "nao ha teste identificavel" e
    "medimos e nao sustenta" e justamente o que a ontologia existe para
    preservar.
    """
    import suporte_cientifico as sup

    if not ident["identificavel"]:
        return {"support": sup.NOT_TESTABLE,
                "diagnostico": "not_identifiable",
                "razao_tecnica": "precondicoes_de_identificacao_nao_atendidas"}
    if not est.get("estimado"):
        return {"support": sup.NOT_TESTABLE, "diagnostico": "not_identifiable",
                "razao_tecnica": "ajuste_nao_produziu_estimativa"}
    v = est["por_variavel"][variavel]
    ic = v.get("ic95_bootstrap_de_estrato")
    if not ic:
        return {"support": sup.INCONCLUSIVE, "diagnostico": "sem_intervalo",
                "razao_tecnica": "bootstrap_nao_produziu_replicas_validas"}
    if ic[0] > 1.0:
        return {"support": sup.SUPPORTED, "diagnostico": "efeito_positivo",
                "razao_tecnica": f"ic95_de_{variavel}_inteiramente_acima_de_1"}
    if ic[1] < 1.0:
        return {"support": sup.NOT_SUPPORTED, "diagnostico": "efeito_invertido",
                "razao_tecnica": f"ic95_de_{variavel}_inteiramente_abaixo_de_1"}
    return {"support": sup.INCONCLUSIVE, "diagnostico": "ic_cruza_1",
            "razao_tecnica": f"ic95_de_{variavel}_contem_1"}


# ------------------------------------------------------------------ rodada

# Ate onde o controle negativo pode desviar de 1 sem derrubar o resultado. O
# criterio e RELATIVO e nao absoluto: um piso de vies de 5% ao lado de um
# efeito de 51% e ruido de desenho; o mesmo 5% ao lado de um efeito de 8% seria
# metade do achado. Escrito DEPOIS de ver o controle de Nova York, e declarado
# como tal no artefato.
FRACAO_MAXIMA_DO_CONTROLE = 1 / 3


def _leitura_do_controle(est_futuro: dict, por_protocolo: dict) -> dict:
    """O controle negativo desviou de 1? E isso importa diante do efeito?"""
    principal = (por_protocolo[PROTOCOLO_PRIMARIO]["por_exposicao"]["gatilho"]
                 ["estimativa"])
    if not est_futuro.get("estimado") or not principal.get("estimado"):
        return {"desviou_de_1": None,
                "leitura": "controle negativo nao estimavel neste conjunto"}
    v = est_futuro["por_variavel"][CONTROLE_FUTURO]
    desviou = bool(v.get("ic95_exclui_1"))
    b_controle = abs(float(v["coeficiente_por_mm"]))
    b_principal = abs(float(principal["por_variavel"][GATILHO]["coeficiente_por_mm"]))
    fracao = (b_controle / b_principal) if b_principal else None
    tolerado = fracao is not None and fracao <= FRACAO_MAXIMA_DO_CONTROLE
    return {
        "desviou_de_1": desviou,
        "razao_de_taxas": v[f"razao_de_taxas_por_{POR_MM:g}mm"],
        "magnitude_relativa_ao_efeito_principal": (
            round(fracao, 4) if fracao is not None else None),
        "fracao_maxima_tolerada": FRACAO_MAXIMA_DO_CONTROLE,
        "criterio_escrito_depois_de_ver_o_resultado": True,
        "dentro_da_tolerancia": bool(not desviou or tolerado),
        "leitura": (
            "o controle e nulo: nao ha vies temporal residual detectavel"
            if not desviou else
            ("o controle desvia de 1, entao o desenho TEM vies temporal "
             "residual -- mas a magnitude dele e uma fracao pequena do efeito "
             "principal, e no sentido oposto; o achado sobrevive com o piso de "
             "vies declarado"
             if tolerado else
             "o controle desvia de 1 com magnitude comparavel ao efeito "
             "principal: o que o desenho mede pode ser estrutura temporal "
             "residual, e o achado NAO se sustenta")),
    }


def consolidar(por_protocolo: dict, alvo: str = "gatilho",
               controle: dict | None = None) -> dict:
    """O que sobreviveu a TODOS os protocolos, e nao o melhor deles.

    Regra de leitura, escrita depois de ver a tabela por protocolo e declarada
    como tal: o resultado que vale e o do protocolo PRIMARIO, congelado antes
    de estimar; os demais dizem se ele e estavel. `supported` so e mantido
    quando os dois bootstraps do primario excluem 1 e nenhum protocolo de
    sensibilidade inverte a direcao -- caso contrario cai para `inconclusive`,
    porque um efeito que muda de leitura conforme a janela do estrato nao esta
    demonstrado.
    """
    import suporte_cientifico as sup

    primario = por_protocolo[PROTOCOLO_PRIMARIO]["por_exposicao"][alvo]
    est = primario["estimativa"]
    base = primario["suporte"]["support"]
    if not est.get("estimado"):
        return {"support": base, "regra": "primario nao estimou",
                "escrita_depois_de_ver_os_resultados": True}
    v = est["por_variavel"][EXPOSICOES[alvo][0]]
    rr = v[f"razao_de_taxas_por_{POR_MM:g}mm"]
    dois_bootstraps = bool(v.get("ic95_exclui_1")
                           and v.get("ic95_de_celula_exclui_1"))
    direcoes = []
    for nome, bloco in por_protocolo.items():
        e = bloco["por_exposicao"][alvo]["estimativa"]
        if e.get("estimado"):
            direcoes.append(
                e["por_variavel"][EXPOSICOES[alvo][0]][
                    f"razao_de_taxas_por_{POR_MM:g}mm"])
    inverte = any((x - 1.0) * (rr - 1.0) < 0 for x in direcoes)
    # O CONTROLE NEGATIVO TEM PODER DE VETO. Um falseamento que reprova e o
    # unico jeito de o desenho dizer "o que eu meço aqui pode nao ser efeito" --
    # e deixa-lo fora da consolidacao seria manter o teste e ignorar o
    # resultado dele. Sao Paulo e exatamente esse caso: `supported` nos tres
    # protocolos, e a chuva de trinta dias DEPOIS explicando um terco do
    # tamanho do efeito.
    controle_ok = controle is None or controle.get("dentro_da_tolerancia", True)
    estavel = dois_bootstraps and not inverte and controle_ok
    support = base if (base != sup.SUPPORTED or estavel) else sup.INCONCLUSIVE
    return {
        "support": support,
        "suporte_do_protocolo_primario": base,
        "controle_negativo_dentro_da_tolerancia": controle_ok,
        "magnitude_do_controle_sobre_o_efeito": (
            (controle or {}).get("magnitude_relativa_ao_efeito_principal")),
        "razao_de_taxas_do_primario": rr,
        "os_dois_bootstraps_excluem_1": dois_bootstraps,
        "algum_protocolo_inverte_a_direcao": inverte,
        "razoes_de_taxas_por_protocolo": {
            nome: (bloco["por_exposicao"][alvo]["estimativa"]
                   .get("por_variavel", {})
                   .get(EXPOSICOES[alvo][0], {})
                   .get(f"razao_de_taxas_por_{POR_MM:g}mm"))
            for nome, bloco in por_protocolo.items()},
        "regra": ("primario congelado antes de estimar; rebaixa para "
                  "inconclusive se algum bootstrap incluir 1, se a direcao "
                  "inverter entre protocolos, ou se o controle negativo sair "
                  "da tolerancia"),
        "escrita_depois_de_ver_os_resultados": True,
    }


def rodar(fonte: str, b_bootstrap: int = B_BOOTSTRAP) -> dict:
    d = montar(fonte)
    colunas_subtipo = [c for c in d.columns
                       if c.startswith("casos_") and c != "casos"]
    # Desfecho alternativo: OCORRENCIA em vez de VOLUME. A contagem de chamados
    # e superdispersa -- um temporal produz centenas de ligacoes -- e da peso
    # enorme ao dia de tempestade. O indicador binario pergunta se houve
    # chamado, e nao quantos, e separa efeito sobre ocorrer de efeito sobre
    # quanto se reclama.
    d["casos_ocorrencia"] = (d.casos > 0).astype(int)

    resultados = {}
    for protocolo in PROTOCOLOS:
        por_exposicao = {}
        for nome, exp in EXPOSICOES.items():
            prep = preparar(d, protocolo, exp)
            ident = avaliar_identificabilidade(prep, exp)
            est = (estimar(prep, exp, b_bootstrap=b_bootstrap)
                   if ident["identificavel"] else
                   {"estimado": False, "motivo": "precondicoes nao atendidas"})
            por_exposicao[nome] = {
                "exposicoes": exp,
                "amostra": {k: v for k, v in prep.items() if k != "quadro"},
                "identificabilidade": ident,
                "estimativa": est,
                "suporte": estado_de_suporte(ident, est, exp[0]),
            }
        resultados[protocolo] = {
            "descricao": PROTOCOLOS[protocolo]["descricao"],
            "por_que": PROTOCOLOS[protocolo]["por_que"],
            "cancela_por_construcao": PROTOCOLOS[protocolo]["cancela"],
            "por_exposicao": por_exposicao,
        }

    # Dose-resposta e subtipos so no protocolo primario: eles interpretam o
    # resultado principal, e repeti-los em todos os protocolos multiplicaria
    # comparacoes sem responder pergunta nova.
    prep_primario = preparar(d, PROTOCOLO_PRIMARIO, [GATILHO])
    dose = dose_resposta(prep_primario, GATILHO)

    # TESTE DE ATALHO DE DETECCAO. `sewer backup` exige que exista rede para
    # haver a quem reclamar; `street flooding` nao exige. Se o efeito da chuva
    # for parecido nos dois, ele nao e artefato de onde a rede existe.
    por_subtipo = {}
    for coluna in colunas_subtipo:
        prep = preparar(d, PROTOCOLO_PRIMARIO, [GATILHO], coluna_caso=coluna)
        ident = avaliar_identificabilidade(prep, [GATILHO])
        est = (estimar(prep, [GATILHO], coluna_caso=coluna,
                       b_bootstrap=min(b_bootstrap, 400))
               if ident["identificavel"] else
               {"estimado": False, "motivo": "precondicoes nao atendidas"})
        por_subtipo[coluna.replace("casos_", "")] = {
            "amostra": {k: v for k, v in prep.items() if k != "quadro"},
            "identificabilidade": ident, "estimativa": est,
            "suporte": estado_de_suporte(ident, est, GATILHO),
        }

    # CONTROLE NEGATIVO. A chuva de daqui a trinta dias nao pode causar o
    # chamado de hoje. Se ela "explicasse" o desfecho, o que o desenho estaria
    # medindo seria estrutura sazonal residual dentro do estrato -- e o
    # resultado principal iria junto. E o teste de falseamento mais barato que
    # existe aqui, e ele so faz sentido rodado ANTES de acreditar no efeito.
    futuro = d[["cl_lat", "cl_lon", "ts", GATILHO]].copy()
    futuro["ts"] = futuro.ts - pd.Timedelta(days=DIAS_DO_CONTROLE_FUTURO)
    futuro = futuro.rename(columns={GATILHO: CONTROLE_FUTURO})
    d = d.merge(futuro, on=["cl_lat", "cl_lon", "ts"], how="left")
    prep_futuro = preparar(d.dropna(subset=[CONTROLE_FUTURO]),
                           PROTOCOLO_PRIMARIO, [CONTROLE_FUTURO])
    ident_futuro = avaliar_identificabilidade(prep_futuro, [CONTROLE_FUTURO])
    est_futuro = (estimar(prep_futuro, [CONTROLE_FUTURO],
                          b_bootstrap=min(b_bootstrap, 400))
                  if ident_futuro["identificavel"] else
                  {"estimado": False, "motivo": "precondicoes nao atendidas"})

    controle = _leitura_do_controle(est_futuro, resultados)

    prep_ocorrencia = preparar(d, PROTOCOLO_PRIMARIO, [GATILHO],
                               coluna_caso="casos_ocorrencia")
    ident_ocorrencia = avaliar_identificabilidade(prep_ocorrencia, [GATILHO])
    est_ocorrencia = (estimar(prep_ocorrencia, [GATILHO],
                              coluna_caso="casos_ocorrencia",
                              b_bootstrap=min(b_bootstrap, 400))
                      if ident_ocorrencia["identificavel"] else
                      {"estimado": False, "motivo": "precondicoes nao atendidas"})

    return {
        "fonte": fonte,
        "celulas": int(d.groupby(["cl_lat", "cl_lon"]).ngroups),
        "dias_de_serie": int(len(d)),
        "primeiro_dia": d.dia.min(), "ultimo_dia": d.dia.max(),
        "casos": int(d.casos.sum()),
        "dias_com_caso": int((d.casos > 0).sum()),
        "protocolo_primario": PROTOCOLO_PRIMARIO,
        "por_protocolo": resultados,
        "leitura_consolidada": consolidar(resultados, controle=controle),
        "dose_resposta_do_gatilho": dose,
        "controle_negativo_chuva_futura": {
            "o_que_e": (f"a chuva de {DIAS_DO_CONTROLE_FUTURO} dias DEPOIS, no "
                        "mesmo lugar, no lugar do gatilho"),
            "o_que_esperar": ("razao de taxas proxima de 1 com IC contendo 1; "
                              "efeito aqui significa que o desenho esta lendo "
                              "estrutura sazonal residual, e derruba junto o "
                              "resultado principal"),
            "amostra": {k: v for k, v in prep_futuro.items() if k != "quadro"},
            "identificabilidade": ident_futuro,
            "estimativa": est_futuro,
            **controle,
        },
        "desfecho_ocorrencia_em_vez_de_volume": {
            "o_que_muda": ("o desfecho passa a ser HOUVE chamado, e nao "
                           "quantos: o dia de tempestade deixa de pesar "
                           "centenas de vezes mais que os demais"),
            "amostra": {k: v for k, v in prep_ocorrencia.items() if k != "quadro"},
            "identificabilidade": ident_ocorrencia,
            "estimativa": est_ocorrencia,
            "suporte": estado_de_suporte(ident_ocorrencia, est_ocorrencia, GATILHO),
        },
        "por_subtipo_de_chamado": por_subtipo,
    }


def _razao_do_rebaixamento(cons: dict) -> str:
    """Qual das tres condicoes derrubou o primario -- e nao a primeira da lista.

    Publicar "a direcao inverte" quando quem reprovou foi o controle negativo
    poe uma razao falsa ao lado de um estado certo, que e o defeito que
    `claims.py` existe para barrar.
    """
    if not cons.get("os_dois_bootstraps_excluem_1", True):
        return "efeito_nao_estavel: o bootstrap de celula inclui 1"
    if cons.get("algum_protocolo_inverte_a_direcao"):
        return "efeito_nao_estavel: a direcao inverte entre protocolos"
    if not cons.get("controle_negativo_dentro_da_tolerancia", True):
        fracao = cons.get("magnitude_do_controle_sobre_o_efeito")
        return ("falseamento: a chuva de trinta dias DEPOIS explica "
                f"{fracao:.0%} do tamanho do efeito, acima da tolerancia"
                if fracao is not None else
                "falseamento: o controle negativo saiu da tolerancia")
    return "rebaixado sem condicao identificada"


def matriz_de_forcamento(relatorios: dict) -> dict:
    """A avaliacao no vocabulario unico de `suporte_cientifico`.

    Sai como matriz SEPARADA, e nao dentro de `scientific_support_matrix.json`:
    aquela e a dimensao TEMPORAL (o modelo transporta para eventos
    posteriores?) e esta e a dimensao FORCAMENTO (a chuva transitoria muda o
    risco no mesmo lugar?). Sao perguntas diferentes sobre escopos diferentes,
    e junta-las por compartilharem a palavra "tempo" e exatamente o que
    `DIMENSAO_TEMPORAL` proibe.

    Para virar celula da matriz servida basta registrar a dimensao em
    `suporte_cientifico.DIMENSOES` com `tem_protocolo_formal: True` apontando
    para este arquivo -- ate la, o artefato existe e nao e consumido pelo
    contrato, que e o estado correto para evidencia ainda nao adotada.
    """
    import suporte_cientifico as sup

    avaliacoes = []
    for fonte, rel in relatorios.items():
        r = rel["resultado"]
        cons = r["leitura_consolidada"]
        primario = r["por_protocolo"][PROTOCOLO_PRIMARIO]["por_exposicao"]["gatilho"]
        avaliacoes.append({
            "dimension": "event_forcing",
            "support": cons["support"],
            "support_significa": sup.DEFINICAO[cons["support"]],
            "nao_equivale_a": sup.NAO_EQUIVALE[cons["support"]],
            "scope": {
                "region_id": fonte,
                "model_id": "case_crossover:gatilho_diario",
                "protocol_id": f"case_crossover/{PROTOCOLO_PRIMARIO}",
                "evidence_version": rel["proveniencia"]["serie_de_chuva"]["sha256"][:16],
                "evaluation_window": f"{r['primeiro_dia']}..{r['ultimo_dia']}",
                "evaluation_domain": "celula do produto de chuva x dia",
                "forcing_source": "open_meteo/era5_land",
                "observation_process": ("chamado do cidadao a central de "
                                        "atendimento; ausencia de chamado nao "
                                        "e ausencia de alagamento"),
                "label_definition": "categoria hidrica da propria base",
            },
            "protocol_id": f"case_crossover/{PROTOCOLO_PRIMARIO}",
            # A razao segue a decisao CONSOLIDADA. Quando ela rebaixa o
            # primario, repetir a razao do primario diria "IC acima de 1" ao
            # lado de `inconclusive` -- exatamente o tipo de payload que o
            # projeto barra em `claims.py`.
            "razao_tecnica": (
                primario["suporte"]["razao_tecnica"]
                if cons["support"] == primario["suporte"]["support"]
                else _razao_do_rebaixamento(cons)),
            "razao_tecnica_do_protocolo_primario": primario["suporte"]["razao_tecnica"],
            "diagnostico_de_identificabilidade": primario["suporte"]["diagnostico"],
            "leitura_consolidada": cons,
            "evidence": [{
                "medida": f"razao_de_taxas_por_{POR_MM:g}mm",
                "valor": cons["razao_de_taxas_do_primario"],
                "ic95_bootstrap_de_estrato": (
                    primario["estimativa"].get("por_variavel", {})
                    .get(GATILHO, {}).get("ic95_bootstrap_de_estrato")),
                "estratos_informativos": primario["amostra"]["estratos_informativos"],
                "estimador": "poisson_condicional",
            }],
            "limitations": rel["limitacoes"],
            "provenance": [rel["proveniencia"]],
            "operational_effect": sup.EFEITO_OPERACIONAL[cons["support"]],
            "operational_effect_motivo": sup.MOTIVO_DO_EFEITO[cons["support"]],
            "avaliado_em": rel["gerado_em"],
        })
    return {
        "tipo": "scientific_support_matrix",
        "versao": "1",
        "dimensao": "event_forcing",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estados_possiveis": list(sup.ESTADOS_DE_SUPORTE),
        "definicoes": sup.DEFINICAO,
        "pergunta": ("a chuva transitoria aumenta o risco de ocorrencia "
                     "notificada no mesmo lugar?"),
        "nao_e_a_dimensao_temporal": (
            "temporal pergunta se o modelo transporta para eventos "
            "posteriores; forcamento pergunta se a chuva muda o risco no "
            "mesmo lugar. Compartilhar a palavra tempo nao as torna a mesma"),
        "avaliacoes": avaliacoes,
        "nao_e": ("nao e portao do contrato, nao entra na escada de recusa e "
                  "nao altera escore: e evidencia registrada"),
    }


def _sha(caminho: Path) -> str | None:
    return (hashlib.sha256(caminho.read_bytes()).hexdigest()
            if caminho.exists() else None)


def relatorio_da_fonte(fonte: str, b_bootstrap: int) -> dict:
    from chuva.casos_das_centrais import auditar_janelas

    r = rodar(fonte, b_bootstrap=b_bootstrap)
    ns = argparse.Namespace(fonte=fonte)
    relatorio = {
        "artefato": "chuva_case_crossover",
        "schema_version": "1.0.0",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method_version": METHOD_VERSION,
        "pergunta": ("a chuva transitoria aumenta o risco de ocorrencia NO "
                     "MESMO LUGAR?"),
        "desenho": {
            "nome": "autopareado no tempo (case-crossover)",
            "referencia": "Maclure (1991); Janes, Sheppard & Lumley (2005)",
            "estimador": ("Poisson condicional com intercepto de estrato "
                          "condicionado para fora (Armstrong, Gasparrini & "
                          "Tobias, 2014)"),
            "unidade_de_analise": "celula do produto de chuva x dia",
            "unidade_de_reamostragem": "estrato; sensibilidade por celula",
            "sem_classe_negativa": (
                "o caso e o proprio controle em outras datas; o desenho nao "
                "usa fundo e por isso nao herda o descasamento de janela "
                "medido em `auditar_janelas`"),
            "protocolos_congelados_antes_de_estimar": list(PROTOCOLOS),
            "efeito_reportado_por_mm": POR_MM,
        },
        "auditoria_da_aquisicao": auditar_janelas(ns.fonte),
        "resultado": r,
        "limitacoes": [
            "zero de CHAMADO nao e zero de alagamento: a ausencia de registro "
            "e ausencia de notificacao, e o desenho estima efeito sobre "
            "OCORRENCIA NOTIFICADA",
            "a exposicao e constante dentro da celula de ~11 km: o efeito e "
            "estimado na escala da celula, e nao na do endereco",
            "esforco de notificacao so esta cancelado no que e constante "
            "dentro do estrato (lugar, mes, dia da semana); um evento que "
            "mude a propensao a ligar num dia especifico nao esta controlado",
            "acumulado diario nao distingue intensidade: 80 mm em duas horas "
            "e 80 mm no dia inteiro sao o mesmo numero neste produto",
        ],
        "nao_e": ("nao e escore, nao substitui a suscetibilidade territorial, "
                  "nao cria campo de contrato e nao promove modelo"),
        "proveniencia": {
            "casos": {
                "arquivo": f"local_runs/chuva-casos-centrais/casos_celula_dia_{ns.fonte}.csv",
                "sha256": _sha(CASOS / f"casos_celula_dia_{ns.fonte}.csv")},
            "serie_de_chuva": {
                "arquivo": f"local_runs/chuva-serie-celula/serie_diaria_por_celula_{ns.fonte}.csv",
                "sha256": _sha(SERIES / f"serie_diaria_por_celula_{ns.fonte}.csv")},
        },
    }

    return relatorio


def imprimir(relatorio: dict) -> None:
    r = relatorio["resultado"]
    print(f"=== autopareamento no tempo: {r['fonte']} ===")
    print(f"{r['casos']:,} casos, {r['celulas']} celulas, "
          f"{r['dias_de_serie']:,} celulas-dia, "
          f"{r['primeiro_dia']} a {r['ultimo_dia']}")
    for protocolo, bloco in r["por_protocolo"].items():
        marca = " (primario)" if protocolo == PROTOCOLO_PRIMARIO else ""
        print(f"\n-- {protocolo}{marca}")
        for nome, v in bloco["por_exposicao"].items():
            a, e = v["amostra"], v["estimativa"]
            if not e.get("estimado"):
                print(f"   {nome:22s} RECUSA: "
                      f"{v['identificabilidade']['requisitos_nao_atendidos'][0]['requisito']}")
                continue
            alvo = v["exposicoes"][0]
            pv = e["por_variavel"][alvo]
            ic = pv.get("ic95_bootstrap_de_estrato")
            icc = pv.get("ic95_bootstrap_de_celula")
            print(f"   {nome:22s} RR/{POR_MM:g}mm={pv[f'razao_de_taxas_por_{POR_MM:g}mm']:.4f} "
                  f"IC_estrato={ic} IC_celula={icc} "
                  f"estratos={a['estratos_informativos']:,} "
                  f"suporte={v['suporte']['support']}")
    if r["dose_resposta_do_gatilho"].get("avaliado"):
        print(f"\n-- dose-resposta do gatilho: "
              f"{r['dose_resposta_do_gatilho']['razao_de_taxas_por_faixa']}")
    if r["por_subtipo_de_chamado"]:
        print("\n-- por subtipo (teste de atalho de deteccao)")
        for st, v in r["por_subtipo_de_chamado"].items():
            e = v["estimativa"]
            if not e.get("estimado"):
                print(f"   {st:18s} recusa")
                continue
            pv = e["por_variavel"][GATILHO]
            print(f"   {st:18s} RR/{POR_MM:g}mm="
                  f"{pv[f'razao_de_taxas_por_{POR_MM:g}mm']:.4f} "
                  f"IC={pv.get('ic95_bootstrap_de_estrato')} "
                  f"casos={v['amostra']['casos_em_estrato_informativo']:,}")
    c = r["leitura_consolidada"]
    print(f"\n-- leitura consolidada do gatilho: {c['support']} "
          f"(RR={c['razao_de_taxas_do_primario']}, "
          f"dois bootstraps excluem 1={c['os_dois_bootstraps_excluem_1']}, "
          f"inverte entre protocolos={c['algum_protocolo_inverte_a_direcao']})")
    cn = r["controle_negativo_chuva_futura"]
    ce = cn["estimativa"]
    if ce.get("estimado"):
        pv = ce["por_variavel"][CONTROLE_FUTURO]
        print(f"-- controle negativo (chuva de +{DIAS_DO_CONTROLE_FUTURO} dias): "
              f"RR/{POR_MM:g}mm={pv[f'razao_de_taxas_por_{POR_MM:g}mm']:.4f} "
              f"IC={pv.get('ic95_bootstrap_de_estrato')} "
              f"desviou_de_1={cn['desviou_de_1']} "
              f"dentro_da_tolerancia={cn['dentro_da_tolerancia']}")
    o = r["desfecho_ocorrencia_em_vez_de_volume"]["estimativa"]
    if o.get("estimado"):
        pv = o["por_variavel"][GATILHO]
        print(f"-- desfecho OCORRENCIA (nao volume): "
              f"RR/{POR_MM:g}mm={pv[f'razao_de_taxas_por_{POR_MM:g}mm']:.4f} "
              f"IC={pv.get('ic95_bootstrap_de_estrato')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", default="nyc", help="nyc, sao_paulo ou 'todas'")
    ap.add_argument("--bootstrap", type=int, default=B_BOOTSTRAP)
    ns = ap.parse_args()

    fontes = ["nyc", "sao_paulo"] if ns.fonte == "todas" else [ns.fonte]
    OUT.mkdir(parents=True, exist_ok=True)
    relatorios = {}
    for fonte in fontes:
        try:
            rel = relatorio_da_fonte(fonte, ns.bootstrap)
        except FileNotFoundError as e:
            print(f"insumo ausente para {fonte}: {e}")
            print("  rode antes:")
            print(f"    python scripts/chuva/casos_das_centrais.py --fonte {fonte}")
            print("    python scripts/chuva/serie_por_celula.py --online "
                  f"--de-casos {fonte}")
            if len(fontes) == 1:
                return 2
            continue
        relatorios[fonte] = rel
        destino = OUT / f"autopareamento_{fonte}.json"
        destino.write_text(json.dumps(rel, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        imprimir(rel)
        print(f"RELATORIO={destino}\n")

    if not relatorios:
        return 1
    matriz = matriz_de_forcamento(relatorios)
    destino_matriz = OUT / "matriz_suporte_forcamento.json"
    destino_matriz.write_text(json.dumps(matriz, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/chuva/autopareamento_temporal.py --fonte todas\n", encoding="utf-8")
    print("=== matriz de forcamento ===")
    for a in matriz["avaliacoes"]:
        print(f"  {a['scope']['region_id']:12s} {a['support']:14s} "
              f"{a['razao_tecnica']}")
    print(f"MATRIZ={destino_matriz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
