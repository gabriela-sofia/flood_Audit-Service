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
como feature na Fase 1 e vive so como evidencia visual, entao o teste A/B nao e
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

CORRECAO DE 23/08/2026 -- a regra U2 nao estava sendo cumprida em Recife.
O `grupo_cv` vinha um por PONTO (269 grupos, 269 pontos), enquanto a base tem
71 pontos empilhados em 26 celulas: reamostrar aquele grupo era reamostrar
linha, e o IC de Recife saia estreito demais exatamente pelo motivo que a
regra existe para impedir. `grupo_nunca_mais_fino_que_a_celula()` passa a
valer para todos os modelos, como invariante e nao como excecao de Recife.

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

`areal_brasil_observado`
                   Desde 17/09/2026. Global Flood Database brasileiro, positivo
                   e negativo ambos observados por sensor, 125 tiles. Serve as
                   cinco macrorregioes do IBGE -- a unidade mais fina em que a
                   regra de EPV fecha -- e o Rio Grande do Sul. As regioes NO
                   AJUSTE sao contadas pelo mesmo recorte do contrato, nunca
                   declaradas; a que e servida sem estar no ajuste (o RS)
                   recebe a AUC contra o rotulo local dela.

Uso parcial: `--so <modelo>` reconstroi so aquele e PRESERVA os outros do
disco, com o `gerado_em` original.

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
import susc_firth_shim  # noqa: F401,E402
from ds03_esquema_alvo import VERSAO  # noqa: E402
from aud_desenho02_natureza_dos_negativos import (  # noqa: E402
    negativos_da_fonte,
)

RUNS = REPO / "local_runs"
POOL = RUNS / "ds-05-tabela-unica" / f"tabela_unica_pool_fluvial_{VERSAO}.parquet"
UNICA = RUNS / "ds-05-tabela-unica" / f"tabela_unica_{VERSAO}.parquet"
OUT = RUNS / "svc-01-modelos"

# `elevation_rel_m` e nao `elevation_m`. A troca nao e preferencia teorica: na
# transferencia que este servico faz -- pool sem Curitiba aplicado a Curitiba --
# a absoluta poe 0,0% dos pontos do alvo dentro do dominio do treino, contra
# 94,5% da relativa (aud_desenho04). O `ds03` ja declarava `elevation_rel_m`
# como regra permanente do contrato, e o vetor servido nao a usava.
TOPO = ["elevation_rel_m", "slope_deg", "hand_m", "twi_dinf"]

# `rain_max_24h` saiu. Ela nao e o pico do evento que o nome sugere: e o maior
# acumulado diario da janela ANTERIOR, uma parcela da soma que o
# `rain_decay_index` ja acumula (correlacao 0,66 a 0,92, e menor ou igual em
# 77% a 94% dos casos). Duas variaveis competindo pela mesma informacao, e so
# uma com interpretacao fisica propria.
CHUVA = ["rain_decay_index"]

# CAUSAIS sao as variaveis cujo sinal foi fixado ANTES do ajuste como criterio
# de leitura. Elas continuam sendo as unicas que derrubam o veredito -- mudar
# isso depois de ver o resultado seria mover a trave.
CAUSAIS = ["hand_m", "twi_dinf"]

# SINAL_FISICO cobre TODAS as variaveis. Nao decide veredito: decide o que a
# camada de explicacao pode dizer, e entra como limitacao declarada quando o
# ajuste contraria a fisica. A diferenca entre as duas listas e deliberada.
SINAL_FISICO = {
    "elevation_m": -1,       # mais alto, menos agua acumulada
    "elevation_rel_m": -1,   # idem, medido contra o piso da propria regiao
    "slope_deg": -1,         # mais ingreme, a agua escorre em vez de empocar
    "hand_m": -1,            # mais acima da drenagem, menos alcancavel
    "twi_dinf": +1,          # mais umidade topografica acumulada
    "rain_max_24h": +1,      # mais chuva, mais agua
    "rain_decay_index": +1,  # mais chuva antecedente, solo mais saturado
}
SINAL_EXIGIDO = {f: SINAL_FISICO[f] for f in CAUSAIS}
NEGATIVO_ADMITIDO = ("observado", "exclusao_qualificada", "nao_aplicavel")

# A grade em que o ds05 conta duplicidade (`intra_fonte_mesma_celula`). Dois
# pontos dentro da mesma celula nao sao observacoes independentes.
GRADE_CELULA_GRAUS = 1e-4

EPV_MINIMO = 10
N_BOOT = 1000
SEMENTE = 20260820
AUC_MIN, AUC_MAX = 0.70, 0.88

# Piso logico, e nao criterio de leitura. Fora da faixa [AUC_MIN, AUC_MAX] o
# modelo serve com a limitacao declarada; ABAIXO deste piso ele nao serve, e a
# diferenca nao e de grau: um escore que ordena pior que sorteio nao sustenta
# nem a leitura ordinal.
AUC_ACASO = 0.5

# Regioes-alvo brasileiras: nunca entram no ajuste do modelo que as serve.
REGIOES_ALVO = ("curitiba", "recife")

# As cinco macrorregioes do IBGE, servidas pelo modelo areal brasileiro. A
# atribuicao de cada ponto a uma delas e por ponto-em-poligono sobre a malha
# oficial (`br04`), e nao por prefixo de nome de tile: um tile de 1 grau
# atravessa divisa, e o prefixo diria a divisa errada sem dar erro.
MACRORREGIOES = ("norte", "nordeste", "centro_oeste", "sudeste", "sul")


def grupo_nunca_mais_fino_que_a_celula(s: pd.DataFrame) -> pd.Series:
    """Funde grupos que compartilham celula, e devolve o rotulo de grupo a usar.

    POR QUE ISTO EXISTE, e por que so aparece em Recife:

    a regra U2 do projeto manda reamostrar GRUPO e nao linha, para o IC nao
    sair estreito demais por pseudo-replicacao. Em Recife essa protecao nao
    agia: o `grupo_cv` vinha um por PONTO (269 grupos para 269 pontos), e a
    base tem 71 pontos empilhados em 26 celulas -- ate 6 linhas na mesma
    coordenada, com feature e classe identicas. Reamostrar aquele `grupo_cv`
    era, na pratica, reamostrar linha: exatamente o que a regra proibia.

    Nas fontes fluviais o grupo ja e a AOI ou o evento, muito mais grosso que a
    celula, e nada muda. Por isso a correcao e escrita como INVARIANTE e nao
    como excecao de Recife: nenhum grupo pode ser mais fino que a celula.

    A fusao e por componente conexa (union-find) e nao por "renomear para a
    celula": um grupo legitimo pode cobrir varias celulas, e uma celula pode
    ser tocada por mais de um grupo. Renomear quebraria o primeiro caso; a
    componente conexa preserva os dois.

    O rotulo final da componente e o MENOR `grupo_cv` original que ela contem.
    Isso importa: quando nada funde, o rotulo sai identico ao de antes, e o
    GroupKFold reproduz os mesmos folds. Sem essa garantia, a correcao mexeria
    na AUC dos modelos fluviais -- onde nao ha pseudo-replicacao nenhuma para
    corrigir -- so porque o nome do grupo mudou.
    """
    pai: dict = {}

    def achar(x):
        pai.setdefault(x, x)
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    def unir(a, b):
        ra, rb = achar(a), achar(b)
        if ra != rb:
            pai[ra] = rb

    celula = (
        (s.lat / GRADE_CELULA_GRAUS).round().astype("int64").astype(str)
        + "_"
        + (s.lon / GRADE_CELULA_GRAUS).round().astype("int64").astype(str)
    )
    grupos = s.grupo_cv.astype(str)
    for g, c in zip(grupos, celula):
        unir(("g", g), ("c", c))

    rotulo: dict = {}
    for g in grupos.unique():
        raiz = achar(("g", g))
        if raiz not in rotulo or g < rotulo[raiz]:
            rotulo[raiz] = g
    return pd.Series([rotulo[achar(("g", g))] for g in grupos], index=s.index)


def ajustar(X: np.ndarray, y: np.ndarray):
    from firthlogist import FirthLogisticRegression

    m = FirthLogisticRegression(max_iter=500, skip_pvals=True, skip_ci=True)
    m.fit(X, y)
    return m


def padronizar(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = X.mean(0), X.std(0)
    return mu, np.where(sd == 0, 1, sd)


# Lado do bloco espacial usado quando a fusao por celula colapsa a unidade de
# validacao. Nao e ajuste fino: e a menor escala em que Nova York e Sao Paulo
# ainda tem centenas de blocos, e vem da blocagem espacial de Roberts et al.
# (2017), que o projeto ja usa como referencia para dado com estrutura espacial.
BLOCO_ESPACIAL_KM = 1.0
GRAUS_POR_KM = 1.0 / 111.0


def bloco_espacial(s: pd.DataFrame, km: float = BLOCO_ESPACIAL_KM) -> pd.Series:
    """Rotulo de bloco de `km` de lado, para usar como unidade de validacao."""
    g = km * GRAUS_POR_KM
    return ((s.lat / g).round().astype(int).astype(str) + "_"
            + (s.lon / g).round().astype(int).astype(str))


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


def aderencia_temporal(s: pd.DataFrame, feats: list[str]) -> dict:
    """Positivos e negativos compartilham as mesmas datas?

    O QUE ESTA METRICA MEDE, e por que ela e diferente do suporte temporal:

    `suporte_cientifico.temporal` pergunta se o desempenho se sustenta numa
    janela FUTURA a do ajuste. Esta pergunta e outra, e vem antes: dentro do
    proprio conjunto de ajuste, as duas classes foram observadas nos MESMOS
    dias?

    Quando nao foram, qualquer variavel que varie no tempo -- a chuva, aqui --
    deixa de medir "quanto choveu neste lugar" e passa a identificar "que dia
    e este". E, se os dias positivos sao dias diferentes dos negativos, o
    modelo pode separar as classes acertando a DATA e nao o LUGAR. O escore
    continua saindo, a AUC continua sendo calculavel, e nada no ajuste acusa:
    o defeito e do desenho amostral, nao do estimador.

    Isto ja estava escrito no model card de Recife, mas como frase fixa no
    codigo -- "positivos e negativos dividem 5 das 205 datas". A afirmacao
    mais importante do card era a unica que ninguem recalculava. Agora ela e
    derivada da propria tabela que ajustou o modelo, e muda quando a base
    mudar.

    Duas fracoes, e a segunda importa mais: `datas_com_ambas_as_classes` conta
    dias; `pontos_em_data_compartilhada` conta observacoes. Um conjunto pode
    ter metade dos dias mistos e quase nenhum ponto neles.
    """
    coluna = "data_evento"
    usa_tempo = sorted(f for f in feats if f.startswith("rain_"))
    if coluna not in s.columns:
        return {"avaliada": False,
                "motivo": f"a tabela do ajuste nao tem a coluna {coluna!r}",
                "variaveis_que_dependem_do_tempo": usa_tempo}

    # `format="mixed"` nao e detalhe: a coluna tem `YYYY-MM-DD` e
    # `YYYY-MM-DD HH:MM:SS` juntos, e sem isto o pandas infere UM formato pelo
    # inicio e coage todo o resto a NaT -- Recife inteiro sumiria em silencio.
    # `utc=True` pelo mesmo motivo, um nivel acima: no conjunto multirregional
    # convivem timestamps ingenuos e com fuso explicito, e ai o pandas levanta
    # em vez de coagir. Comparamos DIAS de evento, nao hora local.
    ts = pd.to_datetime(s[coluna], format="mixed", utc=True, errors="coerce")
    com = s[ts.notna()]
    dias = ts[ts.notna()].dt.normalize()
    sem_data = int(len(s) - len(com))
    if not len(com):
        return {"avaliada": False, "motivo": "nenhuma linha do ajuste tem data",
                "linhas_sem_data": sem_data,
                "variaveis_que_dependem_do_tempo": usa_tempo}

    por_dia = com.assign(_dia=dias.values).groupby("_dia").classe.agg(
        ["min", "max", "size"])
    ambas = por_dia[(por_dia["min"] == 0) & (por_dia["max"] == 1)]
    pontos_ambas = int(ambas["size"].sum())

    frac_datas = len(ambas) / len(por_dia)
    frac_pontos = pontos_ambas / len(com)

    saida = {
        "avaliada": True,
        "datas_distintas": int(len(por_dia)),
        "datas_com_ambas_as_classes": int(len(ambas)),
        "fracao_de_datas_com_ambas_as_classes": round(frac_datas, 4),
        "pontos_em_data_compartilhada": pontos_ambas,
        "fracao_de_pontos_em_data_compartilhada": round(frac_pontos, 4),
        "linhas_sem_data": sem_data,
        "janela": {"inicio": str(por_dia.index.min().date()),
                   "fim": str(por_dia.index.max().date())},
        "variaveis_que_dependem_do_tempo": usa_tempo,
        "o_que_mede": ("se as duas classes foram observadas nos mesmos dias "
                       "dentro do conjunto de ajuste"),
        "nao_e": ("nao e o suporte temporal: aquele pergunta se o desempenho "
                  "se sustenta em janela futura, e este pergunta se o desenho "
                  "amostral separa lugares ou apenas dias"),
    }

    # O risco so se concretiza quando o modelo USA variavel que varia no
    # tempo. Sem chuva no conjunto, baixa aderencia continua sendo um fato do
    # desenho, mas nao ha por onde ele virar atalho.
    if not usa_tempo:
        saida["risco"] = "nao_aplicavel"
        saida["risco_motivo"] = (
            "nenhuma variavel do modelo depende do tempo: a data nao tem por "
            "onde entrar no escore")
    elif frac_pontos < 0.25:
        saida["risco"] = "alto"
        saida["risco_motivo"] = (
            f"apenas {frac_pontos:.1%} dos pontos estao em datas que contem as "
            f"duas classes, e o modelo usa {usa_tempo}: nesta base, saber a "
            "data e quase saber a classe, e a chuva pode estar separando dias "
            "em vez de lugares")
    elif frac_pontos < 0.75:
        saida["risco"] = "medio"
        saida["risco_motivo"] = (
            f"{frac_pontos:.1%} dos pontos estao em datas com as duas classes; "
            f"parte do sinal de {usa_tempo} ainda pode ser data e nao lugar")
    else:
        saida["risco"] = "baixo"
        saida["risco_motivo"] = (
            f"{frac_pontos:.1%} dos pontos estao em datas que contem as duas "
            "classes: a variavel temporal e comparada dentro do mesmo dia")
    return saida


def baselines_das_regioes(unica: pd.DataFrame, regioes: list[str]) -> dict:
    """O piso de elevacao de cada regiao servida, para reconstruir a relativa.

    `elevation_rel_m` so existe contra um baseline, e o baseline e propriedade
    da REGIAO -- P1 da elevacao dentro da fonte, exatamente como o `ds04` o
    define. Numa requisicao o servico recebe `elevation_m` derivada do DEM e
    precisa do mesmo numero que o ajuste usou; tira-lo da propria AOI daria
    outro valor a cada recorte, e o escore deixaria de ser reproduzivel.

    Regiao sem baseline nao recebe a chave, e o `SVC-02` recusa em vez de
    inventar zero -- que seria tratar o nivel do mar como piso local, o erro
    que a troca de variavel existe para corrigir.
    """
    saida = {}
    for r in regioes:
        s = unica.loc[unica.fonte == r, "elevation_baseline_m"].dropna()
        if s.empty:
            continue
        v = s.unique()
        # constante por fonte por construcao do ds04; se nao for, algo mudou
        # no redutor e o servico nao pode escolher um valor por conta propria
        if len(v) != 1:
            print(f"  AVISO baseline nao unico em {r}: {len(v)} valores; "
                  f"regiao fica sem baseline")
            continue
        saida[r] = round(float(v[0]), 6)
    return saida


def construir(nome: str, s: pd.DataFrame, feats: list[str], meta: dict) -> dict:
    print(f"\n=== {nome} ===")
    s = s.copy()
    grupos_antes = int(s.grupo_cv.nunique())
    s["grupo_cv"] = grupo_nunca_mais_fino_que_a_celula(s)
    grupos_depois = int(s.grupo_cv.nunique())
    if grupos_depois != grupos_antes:
        print(f"  celula funde grupos: {grupos_antes:,} -> {grupos_depois:,} "
              f"({grupos_antes - grupos_depois:,} eram pseudo-replicacao)")
    print(f"  n={len(s):,} grupos={grupos_depois:,} variaveis={feats}")
    orc = orcamento_estrito(s)
    # FALLBACK DE UNIDADE DE VALIDACAO. A fusao por celula (regra U2) encontra
    # dependencia espacial real, e em base de chamados densa ela encontra
    # DEMAIS: o mesmo endereco reaparece em muitos dias, a componente conexa
    # une todos e sobra um grupo so -- Nova York vai de 69 grupos para 1, Sao
    # Paulo de 30 para 1. Um grupo nao valida nada, e concluir "nao ha
    # evidencia" dali seria confundir colapso da unidade com falta de dado.
    #
    # Onde isso acontece, a unidade passa a ser o BLOCO ESPACIAL de 1 km, que
    # e a resposta padrao da literatura para estrutura espacial (Roberts et al.
    # 2017) e que o projeto ja cita. Onde a fusao NAO colapsa -- Recife,
    # Curitiba, as fluviais -- nada muda: o fallback so dispara no caso que ele
    # existe para tratar, e fica declarado no artefato.
    unidade = "componente_conexa_de_celulas"
    if len(feats) > orc and grupos_antes >= EPV_MINIMO and grupos_depois <= 2:
        s = s.assign(grupo_cv=bloco_espacial(s))
        grupos_depois = int(s.grupo_cv.nunique())
        orc = orcamento_estrito(s)
        unidade = f"bloco_espacial_{BLOCO_ESPACIAL_KM:g}km"
        print(f"  unidade de validacao -> {unidade}: {grupos_depois:,} grupos, "
              f"orcamento {orc}")
    if len(feats) > orc:
        # Quando a fusao por celula colapsa o conjunto em pouquissimos grupos,
        # dizer so "EPV 0" esconde a causa. A regra U2 nao esta errada -- ela
        # esta encontrando dependencia espacial real --, mas numa base de
        # chamados densa o mesmo endereco reaparece em muitos dias e a
        # componente conexa engole o desenho temporal inteiro. Quem ler o
        # artefato precisa distinguir "faltam eventos" de "a unidade de
        # validacao colapsou".
        colapso = grupos_antes >= EPV_MINIMO and grupos_depois <= 2
        motivo = (
            f"a fusao por celula reduziu {grupos_antes:,} grupos a "
            f"{grupos_depois}: o mesmo local reaparece em dias diferentes e a "
            f"componente conexa une todos. Nao ha unidade de validacao cruzada "
            f"neste conjunto sob a regra atual (U2 espacial)"
            if colapso else
            f"{len(feats)} variaveis acima do orcamento de EPV ({orc})")
        print(f"  ABORTADO: {motivo}")
        return {"nome": nome, "servivel": False, "motivo": motivo,
                "orcamento_estrito": orc,
        "unidade_de_validacao": unidade, "features": feats,
                "grupos_antes_da_fusao": int(grupos_antes),
                "grupos_depois_da_fusao": int(grupos_depois),
                "colapso_da_unidade_de_validacao": bool(colapso), **meta}

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

    # Verificacao de sinal de TODAS as variaveis. Uma variavel so e dada como
    # fisicamente verificada quando as duas coisas valem: o sinal bate com o
    # esperado E o IC95 nao cruza zero. Sinal certo com IC cruzando zero nao e
    # coerencia fisica -- e o ajuste nao tendo estabelecido nada sobre ela.
    sinal_por_variavel, invertidas, indefinidas = {}, [], []
    for j, f in enumerate(feats):
        c = coef[j]
        esperado = SINAL_FISICO.get(f)
        ic = ic_coef.get(f)
        cruza = bool(ic and ic[0] <= 0 <= ic[1])
        bate = None if esperado is None else bool(np.sign(c) == esperado)
        sinal_por_variavel[f] = {
            "coef": round(c, 6), "sinal_esperado": esperado,
            "sinal_bate": bate, "ic95": ic, "ic_cruza_zero": cruza,
            "fisicamente_verificada": bool(bate and not cruza),
        }
        if bate is False:
            invertidas.append(f)
        elif bate and cruza:
            indefinidas.append(f)

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

    # Sinal invertido em variavel nao-causal nao derruba o veredito -- o
    # criterio de leitura foi fixado antes do ajuste sobre as causais, e
    # ampliar a trava depois de ver o resultado seria mover a trave. Mas
    # tambem nao pode ficar escondido: entra como limitacao declarada, e a
    # camada de explicacao para de citar a variavel.
    limitacoes = list(meta.get("limitacoes_declaradas", []))

    # A aderencia temporal vira limitacao DERIVADA. Ate aqui esta afirmacao
    # existia como frase fixa no codigo ("positivos e negativos dividem 5 das
    # 205 datas") -- a mais importante do card era a unica que ninguem
    # recalculava, e ela ficaria congelada se a base mudasse.
    aderencia = aderencia_temporal(s, feats)
    if aderencia.get("risco") in ("alto", "medio"):
        limitacoes.append(
            "aderencia temporal "
            f"{aderencia['risco']}: {aderencia['risco_motivo']}")
    if invertidas:
        limitacoes.append(
            "sinal contrario ao fisicamente esperado em "
            f"{', '.join(invertidas)}: a variavel entra no escore mas fica fora "
            "da frase de explicacao")
    if indefinidas:
        limitacoes.append(
            f"IC95 cruza zero em {', '.join(indefinidas)}: o ajuste nao "
            "estabeleceu o efeito, e a variavel fica fora da frase de explicacao")
    meta = {**meta, "limitacoes_declaradas": limitacoes}

    faixa = {f: [round(float(np.percentile(X[:, j], 5)), 4),
                 round(float(np.percentile(X[:, j], 95)), 4)]
             for j, f in enumerate(feats)}

    print(f"  AUC agrupada={auc}  coef={dict(zip(feats, [round(c, 4) for c in coef]))}")
    print(f"  bootstrap: {len(arr)} replicas validas de {N_BOOT} "
          f"({time.time() - t0:.0f}s)")
    print(f"  falhas={falhas if falhas else 'nenhuma'}")
    verificadas = [f for f, v in sinal_por_variavel.items()
                   if v["fisicamente_verificada"]]
    print(f"  sinal fisico verificado em: {verificadas or 'nenhuma'}")
    if invertidas:
        print(f"  sinal INVERTIDO em: {invertidas} (fora da frase de explicacao)")
    if indefinidas:
        print(f"  IC cruza zero em: {indefinidas} (fora da frase de explicacao)")

    print(f"  aderencia temporal: risco={aderencia.get('risco')} "
          f"pontos em data compartilhada="
          f"{aderencia.get('fracao_de_pontos_em_data_compartilhada')}")

    # PISO DO ACASO: abaixo dele o modelo nao e fraco, e ANTI-INFORMATIVO.
    #
    # `AUC_MIN = 0,70` produz `falhas_de_criterio` e nao impede servir -- e a
    # politica esta certa: Recife serve com 0,6422 e a limitacao declarada,
    # porque 0,64 ordena melhor que sorteio e a unica leitura reivindicada e a
    # ORDINAL. Abaixo de 0,50 essa leitura tambem cai: a ordem produzida e pior
    # que aleatoria, e nao ha frase honesta que a acompanhe.
    #
    # Isto apareceu quando o rotulo de Sao Paulo foi separado entre ocorrencia
    # e manutencao. Com pedido de limpeza de bueiro dentro, a AUC era 0,6223 --
    # o terreno prevendo onde a rede entope. Com so ocorrencia de Defesa Civil,
    # ela cai para 0,4753, e o achado e cientifico: em pluvial urbano o terreno
    # nao separa alagamento, que e o que o model card ja declarava como
    # limitacao. Servir 0,4753 seria publicar essa limitacao como escore.
    if auc is not None and auc <= AUC_ACASO:
        print(f"  NAO SERVIVEL: AUC {auc} nao supera o acaso ({AUC_ACASO})")
        return {
            "nome": nome, "servivel": False,
            "motivo": (f"AUC agrupada {auc} nao supera o acaso ({AUC_ACASO}): "
                       "a ordem produzida e pior que aleatoria, e a leitura "
                       "ordinal -- a unica que o projeto reivindica -- nao se "
                       "sustenta"),
            "features": feats, "desempenho": desempenho,
            "aderencia_temporal": aderencia,
            "falhas_de_criterio": falhas,
            "veredito": "ABAIXO_DO_ACASO",
            "sinal_por_variavel": sinal_por_variavel,
            "variaveis_fisicamente_verificadas": [
                f for f, v in sinal_por_variavel.items()
                if v["fisicamente_verificada"]],
            "orcamento_estrito": orc,
            # A recusa carrega os MESMOS campos de dado que a aceitacao: sao
            # eles que respondem "por que falhou?", e sem eles a ficha do
            # reprovado nao sustenta a propria recusa.
            **campos_do_dado(s, unidade, grupos_antes),
            **meta,
        }

    return {
        "nome": nome, "servivel": True, "features": feats,
        "aderencia_temporal": aderencia,
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
        # Que leitura o escore admite, derivada da natureza da amostra
        # negativa e nao de escolha de apresentacao. Com fundo -- amostra
        # do ambiente disponivel, e nao observacao de que ali nao inundou
        # -- a prevalencia nao e identificavel, e nenhuma correcao de
        # intercepto a recupera: so a ordem entre celulas sobrevive.
        **campos_do_dado(s, unidade, grupos_antes),
        "elevacao_baseline_por_regiao": meta.pop("elevacao_baseline_por_regiao", {}),
        "sinal_por_variavel": sinal_por_variavel,
        "variaveis_fisicamente_verificadas": [
            f for f, v in sinal_por_variavel.items() if v["fisicamente_verificada"]],
        "falhas_de_criterio": falhas,
        "veredito": "COERENTE_COM_CRITERIOS" if not falhas else "FORA_DOS_CRITERIOS",
        **meta,
    }


def falhas_por_regiao_do_loro(macrorregioes) -> dict:
    """O criterio de leitura aplicado A CADA macrorregiao, lido do `br04`.

    -------------------------------------------------------------------------
    POR QUE POR REGIAO, E POR QUE LIDO E NAO ESCRITO
    -------------------------------------------------------------------------

    `falhas_de_criterio` e do MODELO, e o modelo areal brasileiro fecha sem
    falha nenhuma. Mas a pergunta que interessa a quem recebe um escore em Belem
    nao e "o modelo nacional fecha?" -- e "ele fecha AQUI?". A medida honesta
    disso e o leave-one-region-out: ajusta nas outras quatro macrorregioes e
    cobra nesta, contra rotulo local.

    A faixa [AUC_MIN, AUC_MAX] vale nos dois sentidos. AUC abaixo de 0,70 diz
    que o modelo nao ordena; AUC acima de 0,88 e igualmente uma bandeira, e
    normalmente significa que a particao ficou facil -- lugares muito parecidos
    dos dois lados, ou uma classe concentrada em poucos grupos. Nenhum dos dois
    e motivo para esconder o escore; os dois sao motivo para ele sair com a
    falha declarada e a maturidade em `mvp_local`.

    O numero e LIDO do artefato do `br04` e nunca digitado aqui. Digitar
    congelaria a medida na versao de hoje, que e o defeito que este projeto ja
    encontrou dez vezes -- a ultima delas neste mesmo arquivo, com o pool areal
    preso em `fonte == "cems"` enquanto 78.694 pontos brasileiros esperavam.

    Sem o artefato em disco nao ha rebaixamento e nao ha invencao: devolve
    vazio, e `por_que_sem_loro` no model_card diz que a medida nao estava la.
    """
    alvo = RUNS / "br-04-professor" / "professor.json"
    if not alvo.exists():
        return {}
    try:
        bloco = json.loads(alvo.read_text(encoding="utf-8"))["C_leave_one_region_out"]
    except (KeyError, ValueError):
        return {}
    sigla = {"norte": "N", "nordeste": "NE", "centro_oeste": "CO",
             "sudeste": "SE", "sul": "S"}
    fora = {}
    for reg in macrorregioes:
        d = bloco.get(sigla.get(reg, ""))
        if not d:
            continue
        auc = d.get("auc_agregada")
        fisica = d.get("fisica_sustentada")
        falhas = []
        if auc is None:
            continue
        if auc < AUC_MIN:
            falhas.append(
                f"leave-one-region-out {auc} abaixo de {AUC_MIN}: o modelo "
                f"nacional nao ordena nesta macrorregiao quando nao a viu")
        if auc > AUC_MAX:
            falhas.append(
                f"leave-one-region-out {auc} acima de {AUC_MAX}: separacao "
                f"boa demais e bandeira, nao aprovacao -- normalmente a "
                f"particao ficou facil")
        if fisica is not None and fisica < 3:
            falhas.append(
                f"leave-one-region-out sustenta {fisica} de 3 sinais fisicos")
        if falhas:
            fora[reg] = falhas
    return fora


def regioes_com_linhas_no_ajuste(conjunto: pd.DataFrame, candidatas) -> dict:
    """Quantas linhas do ajuste caem DENTRO de cada regiao candidata.

    -------------------------------------------------------------------------
    POR QUE CONTAR, E NAO DECLARAR
    -------------------------------------------------------------------------

    `regioes.esta_no_ajuste` obedece a `regioes_no_ajuste` quando o modelo a
    declara -- "quem declara, decide". Em 17/09/2026 essa lista foi escrita a
    mao com o Rio Grande do Sul dentro, justificada por "24.985 linhas do GFD
    na caixa do RS". A caixa usada naquela conta foi desenhada a olho sobre o
    estado inteiro; a extensao DECLARADA do RS e a da ativacao EMSR720, e
    dentro dela o GFD tem ZERO linhas. O contrato passou a chamar de `validado`
    uma regiao que o modelo nunca viu.

    Contar pelo mesmo recorte que o contrato usa elimina a classe do erro:
    macrorregiao pelo poligono da malha do IBGE (o do `br04` e de
    `regioes.dentro`), lugar nomeado pela extensao de `regioes.py`.
    """
    sys.path.insert(0, str(REPO / "scripts" / "servico"))
    import br04_o_brasil_como_professor as br
    import regioes as _reg

    sigla = {"norte": "N", "nordeste": "NE", "centro_oeste": "CO",
             "sudeste": "SE", "sul": "S"}
    com = br.com_regiao(conjunto[["lat", "lon"]].copy())
    contagem = {}
    for regiao in candidatas:
        if regiao in sigla:
            contagem[regiao] = int((com.regiao_br == sigla[regiao]).sum())
            continue
        cx = _reg.extensao(regiao)
        if not cx:
            contagem[regiao] = 0
            continue
        contagem[regiao] = int(((conjunto.lat >= cx["lat_min"])
                                & (conjunto.lat <= cx["lat_max"])
                                & (conjunto.lon >= cx["lon_min"])
                                & (conjunto.lon <= cx["lon_max"])).sum())
    return contagem


def desempenho_em_rotulo_local(modelo: dict, linhas: pd.DataFrame) -> dict:
    """AUC do modelo JA AJUSTADO contra o rotulo observado de uma regiao.

    Para regiao servida por transferencia que TEM inventario proprio -- o RS,
    com a delineacao do Copernicus EMS -- esta e a medida que
    `transferencia_caracterizada` promete: acerto verificavel contra rotulo
    local, com o modelo que nao o viu. Nao reajusta nada: projeta os
    coeficientes servidos, com a padronizacao servida.
    """
    from sklearn.metrics import roc_auc_score

    feats = modelo.get("features") or []
    s = linhas.dropna(subset=feats)
    s = s[s.classe.isin([0, 1])]
    if s.empty or s.classe.nunique() < 2:
        return {"avaliado": False,
                "motivo": "sem as duas classes observadas na regiao"}
    z = np.full(len(s), float(modelo.get("intercepto", 0.0)))
    for f in feats:
        pad = modelo["padronizacao"][f]
        z += modelo["coef"][f] * (s[f].to_numpy(float) - pad["media"]) / pad["desvio"]
    por_aoi = {}
    for aoi, g in s.assign(_z=z).groupby("aoi"):
        if g.classe.nunique() == 2:
            por_aoi[str(aoi)] = round(float(roc_auc_score(g.classe, g._z)), 4)
    return {
        "avaliado": True,
        "auc_na_regiao": round(float(roc_auc_score(s.classe, z)), 4),
        "n": int(len(s)), "positivos": int(s.classe.sum()),
        "auc_por_aoi": por_aoi,
        "aois_acima_do_acaso": int(sum(v > 0.5 for v in por_aoi.values())),
        "aois_avaliadas": len(por_aoi),
        "nao_e": ("nao e suporte temporal nem desempenho no ajuste: e o modelo "
                  "nacional aplicado a um rotulo que ele nao viu, de outro "
                  "instrumento (delineacao de analista, nao MODIS)"),
    }




def campos_do_dado(amostra, unidade: str, grupos_antes) -> dict:
    """O que descreve a AMOSTRA, e nao o ajuste.

    POR QUE ISTO VIROU FUNCAO

    Estes campos existiam so no retorno do modelo SERVIVEL. O modelo reprovado
    saia com 18 campos contra 35, e os 17 que sumiam eram em boa parte estes --
    `grupos_positivos`, `prevalencia_do_ajuste`, `niveis_de_negativo`,
    `unidade_de_validacao`. Ou seja: o registro da recusa perdia exatamente os
    campos que explicariam a recusa, e quem fosse entender por que Sao Paulo
    ficou abaixo do acaso nao encontrava com quantos grupos positivos ele foi
    ajustado.

    Para um projeto cuja tese e governanca auditavel de inferencia, a recusa e
    metade do que ele afirma saber fazer. Ela nao pode ser registrada pela
    metade.

    Duplicar a lista nos dois retornos daria duas definicoes de "o que descreve
    o dado", e elas divergiriam no primeiro campo novo. Uma funcao, dois
    chamadores.
    """
    return {
        "prevalencia_do_ajuste": round(float(amostra.classe.mean()), 4),
        "amostra_negativa": negativos_da_fonte(amostra),
        "n": int(len(amostra)),
        "grupos": int(amostra.grupo_cv.nunique()),
        "grupos_antes_da_fusao_por_celula": grupos_antes,
        "unidade_de_reamostragem": (
            "componente conexa de grupo e celula: nenhum grupo mais fino que a "
            f"celula de {GRADE_CELULA_GRAUS} grau"
            if unidade == "componente_conexa_de_celulas" else
            f"bloco espacial de {BLOCO_ESPACIAL_KM:g} km (Roberts et al. 2017): "
            "a fusao por celula colapsou a unidade neste conjunto, porque o "
            "mesmo local reaparece em muitos dias"),
        "unidade_de_validacao": unidade,
        "grupos_positivos": int(
            amostra.loc[amostra.classe == 1, "grupo_cv"].nunique()),
        "grupos_negativos": int(
            amostra.loc[amostra.classe == 0, "grupo_cv"].nunique()),
        "fontes_do_ajuste": amostra.fonte.value_counts().to_dict(),
        "niveis_de_negativo": amostra.loc[amostra.classe == 0, "nivel_negativo"]
                                     .value_counts().to_dict(),
    }


def declarar_a_ficha(m: dict) -> dict:
    """Acrescenta `tipo` e `nao_e` a ficha, no UNICO lugar onde ela e escrita.

    POR QUE AQUI, e nao em cada `return`

    O ajuste sai por tres caminhos -- servivel, abaixo do acaso e abortado por
    orcamento -- e cada um monta o proprio dicionario. Declarar nos tres
    significaria tres textos que divergem no primeiro dia em que alguem editar
    um so. Aqui a ficha ja esta pronta e o que ela E ja esta decidido.

    O `nao_e` MUDA conforme o veredito, e essa e a razao de a funcao existir em
    vez de uma constante: a limitacao de um modelo que serve nao e a mesma de
    um que ficou abaixo do acaso, e um `nao_e` generico seria decoracao.

    Incidente: em 23/09/2026 seis das oito fichas servidas nao declaravam nem
    `tipo` nem `nao_e`, e as duas unicas que declaravam eram as duas ORFAS --
    fora do indice, sem servir nada. O coletor de conhecimento nao absorve
    manifesto mudo, entao o que o projeto mais afirma era o que ele menos
    conseguia lembrar.
    """
    if m.get("servivel") is True:
        nao_e = ("nao autoriza leitura fora da faixa de dominio declarada, nao "
                 "afirma validade temporal -- que e medida em outro artefato -- "
                 "e nao transforma ordenacao em previsao de evento")
    elif m.get("veredito") == "ABAIXO_DO_ACASO":
        nao_e = ("nao e escore utilizavel: a ordem produzida e pior que "
                 "aleatoria. A ficha existe para registrar a medida, e cita-la "
                 "como resultado do projeto seria publicar a limitacao como "
                 "achado")
    else:
        nao_e = ("o ajuste nao fechou os criterios do projeto; a ficha registra "
                 "por que, e nao autoriza usar o modelo para responder nada")
    return {"tipo": "ficha_de_modelo_servido", "nao_e": nao_e, **m}

def main(so: tuple = ()) -> int:
    """Constroi os modelos servidos. `so` limita a quais, e o resto e PRESERVADO.

    -------------------------------------------------------------------------
    POR QUE `so` EXISTE, e por que o indice e MESCLADO e nao reescrito
    -------------------------------------------------------------------------

    Sem ele a unica forma de acrescentar um modelo era reajustar os seis, e o
    custo nao e teorico: em 17/09/2026 essa execucao levou 2,5 horas de relogio
    reconstruindo cinco modelos cujas entradas nao tinham mudado desde as
    18:45, e MORREU no sexto -- sem traceback, com a maquina em 2,4 GB livres
    disputando memoria com o WhiteboxTools. Cinco ajustes identicos foram
    refeitos para nada, e o unico que interessava foi o que nao coube.

    Com `so`, mexer num modelo custa aquele modelo. O indice e lido do disco e
    MESCLADO: quem nao foi reconstruido continua exatamente como estava, com o
    artefato que o produziu intacto ao lado. Reescrever o indice inteiro a
    partir de uma execucao parcial apagaria do catalogo os modelos que nao
    entraram nela -- e o contrato passaria a recusar regiao que ele servia,
    sem ninguem ter decidido isso.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===SVC-01: CONSTRUCAO DOS MODELOS SERVIDOS===")

    for p in (POOL, UNICA):
        if not p.exists():
            print(f"ABORTADO: {p} ausente. Gere com: "
                  "python scripts/suscetibilidade/ds05_admissao_consolidacao.py")
            return 1

    pool = pd.read_parquet(POOL)
    pool = pool[pool.classe.isin([0, 1])
                & pool.nivel_negativo.isin(NEGATIVO_ADMITIDO)].copy()
    unica = pd.read_parquet(UNICA)
    unica = unica[(unica.admitido == True) & unica.classe.isin([0, 1])].copy()  # noqa: E712

    modelos = []

    def construir_ou_preservar(nome: str, *args, **kwargs) -> dict:
        """Ajusta o modelo, ou devolve o artefato que ja esta no disco.

        Preservar e LER o json anterior, e nao fabricar um substituto: o que
        volta e byte a byte o que a execucao anterior gravou, com o proprio
        `gerado_em` dela. Assim o indice mesclado nunca carimba data nova em
        ajuste velho -- que e a forma preferida deste projeto de mentir sem
        errar uma conta.
        """
        if not so or nome in so:
            return construir(nome, *args, **kwargs)
        anterior = OUT / f"{nome}.json"
        if anterior.exists():
            print(f"\n=== {nome}: preservado do disco (--so) ===", flush=True)
            return json.loads(anterior.read_text(encoding="utf-8"))
        print(f"\n=== {nome}: pulado, e nao havia artefato no disco ===",
              flush=True)
        return {"nome": nome, "servivel": False,
                "motivo": "nao construido nesta execucao e ausente do disco"}

    # 1. Recife -- pluvial urbano, modelo proprio
    rec = unica[unica.fonte == "recife"].dropna(subset=TOPO + CHUVA + ["grupo_cv", "lat", "lon"])
    modelos.append(construir_ou_preservar(
        "recife_pluvial", rec, TOPO + CHUVA,
        {"mecanismo": "PLUVIAL_URBANO",
         "regioes_servidas": ["recife"],
         # Declarar, e nao deixar inferir: `esta_no_ajuste` so cai na
         # inferencia por caixa quando o modelo cala, e la ela pergunta se
         # ALGUMA fonte do ajuste tem linha dentro da caixa da regiao -- o que
         # um unico ponto intruso responde por acidente. Aqui o modelo e local
         # e a resposta e a propria regiao.
         "regioes_no_ajuste": ["recife"],
         "elevacao_baseline_por_regiao": baselines_das_regioes(unica, ["recife"]),
         "por_que_este_conjunto":
             "Recife tem modelo proprio porque o mecanismo e outro: a agua vem "
             "da chuva excedendo a drenagem, nao do canal subindo. Somar Recife "
             "ao conjunto fluvial assumiria um processo gerador unico",
         "limitacoes_declaradas": [
             "negativo por ausencia de registro, o nivel mais fraco da hierarquia "
             "-- 98,4% dos negativos sao fundo, e o escore so sustenta ordem",
             "rain_max_24h saiu do vetor: media a janela ANTERIOR ao evento e "
             "e parcela da soma que rain_decay_index acumula",
             "HAND nao separa as classes em Recife -- num evento pluvial HAND nao "
             "e o mecanismo",
         ]}))

    # 1b. Pluvial urbano das cidades novas -- modelo LOCAL, como Recife.
    #
    # A primeira tentativa aplicou aqui o desenho de Curitiba, excluindo a
    # regiao alvo do treino, e o resultado refutou a escolha: `pluvial_nyc`
    # ajustado em Recife mais Sao Paulo deu AUC 0,4385 na CV do PROPRIO treino,
    # abaixo do acaso. O conjunto e heterogeneo demais -- Recife na planicie
    # costeira a 14 m e Sao Paulo no planalto a 763 m --, e a validacao por
    # grupos separa por cidade, entao o modelo treina numa e e cobrado na outra.
    #
    # O erro foi de premissa, e nao de ajuste. Curitiba e servida por
    # transferencia porque NAO TEM inventario local; Nova York e Sao Paulo
    # entraram COM inventario proprio -- chamados do cidadao, com rotulo,
    # geometria e data. Para quem tem dado local, a dinamica que se aplica e a
    # de Recife: modelo local, e a maturidade que vem com ele.
    pluvial = unica[(unica.mecanismo == "PLUVIAL_URBANO")
                    & unica.classe.isin([0, 1])]
    for alvo in ("nyc", "sao_paulo"):
        conj = pluvial[pluvial.fonte == alvo].dropna(
            subset=TOPO + ["grupo_cv", "lat", "lon"])
        if conj.empty or conj.classe.nunique() < 2:
            print(f"\n=== pluvial_{alvo}: sem conjunto de treino ===")
            continue
        base = baselines_das_regioes(unica, [alvo])
        # NOVA YORK E CONTEXTO, E NAO LUGAR ATENDIDO (23/09/2026).
        #
        # `regioes.SEMENTES` ja declarava `no_produto: False` para ela, com a
        # razao escrita: "o produto e brasileiro, e uma carta do Brasil com um
        # pino em Manhattan nao descreve nem o produto nem a ciencia". Mas o
        # modelo continuava saindo com `regioes_servidas: ["nyc"]`, entao um
        # campo dizia que ela nao esta no produto e o modelo servido dizia que
        # ela e regiao atendida. Duas partes do contrato discordando sobre o
        # mesmo caso -- a mesma forma do defeito de Curitiba.
        #
        # O MODELO CONTINUA SENDO AJUSTADO, e de proposito: e nele que o
        # mecanismo terreno-sob-gatilho foi medido (TWI responde, e o efeito
        # cresce com a chuva). Tirar o ajuste apagaria a evidencia junto com a
        # oferta. O que sai e a AFIRMACAO DE SERVICO, e so ela.
        serve_alguem = [] if alvo == "nyc" else [alvo]
        modelos.append(construir_ou_preservar(
            f"pluvial_urbano_{alvo}", conj, TOPO,
            {"mecanismo": "PLUVIAL_URBANO",
             "classe_relevo": "PLANO_OU_ONDULADO",
             "regioes_servidas": serve_alguem,
             "papel": ("contexto_cientifico" if alvo == "nyc"
                       else "regiao_servida"),
             "regioes_no_ajuste": [alvo],
             "elevacao_baseline_por_regiao": base,
             "por_que_este_conjunto":
                 f"{alvo} tem inventario proprio -- chamados do cidadao com "
                 "rotulo, geometria e data --, entao o modelo e local, como o "
                 "de Recife. Transferir de outra cidade foi tentado e refutado: "
                 "o conjunto Recife+Sao Paulo da AUC 0,4385 na CV do proprio "
                 "treino, porque planicie costeira e planalto nao formam um "
                 "processo gerador unico",
             "limitacoes_declaradas": [
                 "mecanismo pluvial urbano: a agua vem da chuva excedendo a "
                 "drenagem, e a CAPACIDADE da rede nao esta no vetor -- HAND e "
                 "TWI descrevem para onde a agua corre naturalmente, e na "
                 "cidade ela corre para onde a engenharia mandou",
                 "chuva fora do conjunto: o gatilho `rain_event_day` esta "
                 "adquirido e medido (leva Nova York de 0,6090 para 0,7395), e "
                 "nao entra porque Recife nao o tem e porque a chuva do dia do "
                 "evento e posterior a decisao de rotular o ponto",
                 "negativo de fundo: chamado de outro assunto da oportunidade "
                 "de observacao, nao vistoria; o escore sustenta ordem, nao "
                 "valor absoluto",
             ]}))

    # 2. Planicie fluvial -- sem nenhuma linha da regiao alvo
    pla = pool[(pool.classe_relevo == "PLANO_OU_ONDULADO")
               & (~pool.fonte.isin(REGIOES_ALVO))].dropna(
                   subset=TOPO + ["grupo_cv", "lat", "lon"])
    modelos.append(construir_ou_preservar(
        "fluvial_planicie", pla, TOPO,
        {"mecanismo": "FLUVIAL_ENXURRADA",
         "classe_relevo": "PLANO_OU_ONDULADO",
         "regioes_servidas": ["curitiba"],
         # LISTA VAZIA, E DECLARADA. O conjunto e construido com
         # `~pool.fonte.isin(REGIOES_ALVO)`: Curitiba nao entra em nenhuma
         # linha, de proposito. Sem esta declaracao `esta_no_ajuste` caia na
         # inferencia por caixa, achava o tile do GFD que invade a caixa de
         # Curitiba e respondia `True` -- e a maturidade virava `validado`
         # para uma regiao que o modelo nunca viu. E o MESMO ponto intruso que
         # apagou o baseline de elevacao dela: um so defeito, dois sintomas.
         "regioes_no_ajuste": [],
         "elevacao_baseline_por_regiao": baselines_das_regioes(unica, ["curitiba"]),
         "por_que_este_conjunto":
             "a regiao alvo nao entra no treino em nenhuma linha: o proposito e "
             "prever onde nao ha inventario local, e validar contra o rotulo da "
             "propria regiao seria a pergunta errada "
             "(metodo_aplicacao_sem_rotulo_local_v1.md)",
         "limitacoes_declaradas": [
             "elevation_rel_m e a elevacao contra o P1 da propria regiao; a "
             "absoluta saiu do vetor em 30/08/2026 porque punha 0,0% dos pontos "
             "de Curitiba no dominio do treino, contra 94,5% da relativa",
             "chuva fora do conjunto: nao discrimina na escala em que o modelo "
             "compara (ext_chuva_estado_do_projeto_v1.md)",
         ]}))

    # 3. Serra fluvial -- orcamento de EPV manda no numero de variaveis
    ser = pool[(pool.classe_relevo == "INGREME")
               & (~pool.fonte.isin(REGIOES_ALVO))].dropna(
                   subset=TOPO + ["grupo_cv", "lat", "lon"])
    orc_ser = orcamento_estrito(ser)
    feats_ser = (CAUSAIS if orc_ser >= 2 else ["hand_m"])[:max(orc_ser, 1)]
    modelos.append(construir_ou_preservar(
        "fluvial_serra", ser, feats_ser,
        {"mecanismo": "FLUVIAL_ENXURRADA",
         "classe_relevo": "INGREME",
         "regioes_servidas": ["petropolis"],
         # Mesma construcao que a planicie: a regiao alvo nao entra. Hoje a
         # inferencia por caixa acerta aqui por acidente (nenhuma fonte do
         # ajuste tem linha dentro de Petropolis); declarar tira o acidente
         # do caminho antes que uma AOI nova o transforme em erro.
         "regioes_no_ajuste": [],
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

    # 4. Areal brasileiro observado -- o modelo que o negativo do GFD abriu
    #
    # -------------------------------------------------------------------------
    # POR QUE ELE EXISTE, E POR QUE SO AGORA
    # -------------------------------------------------------------------------
    #
    # Ate 16/09/2026 o unico modelo areal do projeto era `fluvial_areal_observado`,
    # ajustado pelo `rs02` sobre `fonte == "cems"` -- 23.915 pontos, todos de
    # ativacao do Copernicus. Naquele ajuste o `slope_deg` tinha IC95
    # [-0,4181; +0,0520]: CRUZAVA ZERO, e a variavel saia como
    # `fisicamente_verificada: false`. Com uma causal nao verificada nenhuma
    # regiao podia chegar a `validado`.
    #
    # O que mudou nao foi o metodo, foi a amostra. O Global Flood Database
    # nacional trouxe 78.694 pontos brasileiros em 125 tiles com as duas classes
    # OBSERVADAS, e com eles os tres IC passam a excluir zero:
    #
    #     slope_deg  -0,3036  [-0,7541; -0,0176]
    #     hand_m     -3,3492  [-4,3327; -2,5451]
    #     twi_dinf   +0,5962  [+0,4448; +0,7551]
    #
    # O `slope_deg` passava RASPANDO -- limite superior -0,0176. Em 18/09/2026,
    # com o leito permanente fora do positivo (`gfd02`) e 230 tiles, o IC dele
    # foi para [-0,8616; -0,1682], e a frase fixa que o declarava raspando saiu
    # das limitacoes: ela estava sendo publicada sobre um ajuste que ja nao a
    # sustentava. Se um IC voltar a cruzar zero, a limitacao DERIVADA de
    # `construir` o declara.
    #
    # -------------------------------------------------------------------------
    # POR QUE SO O BRASIL, E NAO BRASIL MAIS MUNDO
    # -------------------------------------------------------------------------
    #
    # O protocolo canonico avaliou as duas opcoes nos mesmos esquemas, e o
    # conjunto brasileiro ganha em TODOS eles -- 0,8200 contra 0,7881 no
    # kfold agrupado, 0,8255 contra 0,8113 no forward, 0,8091 contra 0,7279 no
    # holdout, 0,8214 contra 0,7901 no bloco espacial. E ganha tambem no que
    # nao e desempenho: `areal_brasil_observado` fecha com ZERO sinais
    # invertidos em todos os esquemas, e `areal_observado_multi` carrega um
    # sinal invertido em todos. Somar o mundo aqui piora as duas coisas.
    #
    # -------------------------------------------------------------------------
    # POR QUE MACRORREGIAO, E NAO UF
    # -------------------------------------------------------------------------
    #
    # Pela regra de EPV do proprio projeto (grupos da classe rara // 10), medida
    # em 16/09: das 25 UFs com dado areal brasileiro, UMA comporta as tres
    # variaveis (RS, 31 grupos), sete comportam uma, e dezessete comportam
    # ZERO. Em macrorregiao todas comportam pelo menos duas, e Sul e
    # Centro-Oeste comportam as tres. A macrorregiao e a unidade mais fina em
    # que o orcamento fecha -- e nao uma escolha de conveniencia cartografica.
    areal_br = pool[pool.fonte == "gfd"].dropna(
        subset=CAUSAIS + ["slope_deg", "grupo_cv", "lat", "lon"])
    linhas_por_regiao_areal = (
        regioes_com_linhas_no_ajuste(
            areal_br, list(MACRORREGIOES) + ["rio_grande_do_sul"])
        if (not so or "areal_brasil_observado" in so) else {})
    modelos.append(construir_ou_preservar(
        "areal_brasil_observado", areal_br, ["slope_deg"] + CAUSAIS,
        {"mecanismo": "FLUVIAL_AREAL",
         # O RS entra NOMEADO alem da macrorregiao que o contem. A caixa dele
         # e menor que a do Sul, entao `g2` resolve para ele -- e sem estar
         # aqui o portao seguinte devolveria `region_not_supported` no melhor
         # dado do projeto, o defeito de 13/09/2026 por outra porta.
         #
         # Mas SERVIR nao e ESTAR NO AJUSTE. Quem entrou no ajuste e CONTADO,
         # nao declarado -- ver `regioes_com_linhas_no_ajuste`.
         "regioes_servidas": list(MACRORREGIOES) + ["rio_grande_do_sul"],
         "linhas_do_ajuste_por_regiao": linhas_por_regiao_areal,
         "regioes_no_ajuste": sorted(
             r for r, n in linhas_por_regiao_areal.items() if n > 0),
         "falhas_de_criterio_por_regiao": falhas_por_regiao_do_loro(MACRORREGIOES),
         "o_que_falhas_de_criterio_por_regiao_significa":
             "a macrorregiao continua servida; o escore sai com a falha escrita "
             "e a maturidade rebaixada a mvp_local, pela mesma politica "
             "`declara` que vale para Recife",
         "por_que_este_conjunto":
             "positivo e negativo vem AMBOS de delimitacao areal observada por "
             "sensor -- `flooded` e `clear_views` do Global Flood Database --, "
             "e nenhuma linha aqui e negativo por ausencia de registro, que e o "
             "nivel mais fraco da hierarquia e o que explica a fisica invertida "
             "de Recife",
         "unidade_de_validacao_declarada":
             "tile de 1 grau inteiro fora por vez: corte aleatorio de linhas "
             "poria celulas vizinhas da mesma mancha nos dois lados e mediria "
             "interpolacao dentro da cheia, nao generalizacao",
         "limitacoes_declaradas": [
             "elevation_rel_m fora do vetor: o baseline de um conjunto de "
             "escala nacional cai no nivel do mar e a variavel degenera em "
             "elevacao absoluta, que mede onde a AOI fica e nao o relevo dela",
             "chuva fora do vetor: no pool areal ela e do evento e nao varia "
             "dentro da mancha, entao nao e estimavel neste desenho",
             "a resolucao nativa do rotulo e 250 m (MODIS): manchas urbanas "
             "finas e drenagem de bairro estao abaixo do que o sensor separa",
             "as macrorregioes ESTAO no ajuste: o escore nelas e otimista na "
             "mesma medida que qualquer ponto de treino, e a medida honesta de "
             "generalizacao e o leave-one-region-out do br04, nao a AUC daqui",
         ]}))

    # Regiao servida pelo modelo areal SEM estar no ajuste, mas com rotulo
    # proprio: a medida que `transferencia_caracterizada` promete e o acerto do
    # modelo que nao a viu contra esse rotulo. Hoje e o caso do RS, cuja
    # delineacao do Copernicus EMS nunca entrou num ajuste com o GFD.
    areal = next((m for m in modelos
                  if m.get("nome") == "areal_brasil_observado"), None)
    if areal is not None and areal.get("servivel") and (
            not so or "areal_brasil_observado" in so):
        import regioes as _reg
        por_regiao = {}
        for regiao in areal.get("regioes_servidas", []):
            if regiao in (areal.get("regioes_no_ajuste") or []):
                continue
            cfg = _reg.SEMENTES.get(regiao) or {}
            if cfg.get("tipo") != "rotulo":
                continue
            linhas = unica[unica.fonte == cfg["fonte"]]
            if cfg.get("aoi_prefixo"):
                linhas = linhas[linhas.aoi.astype(str).str.startswith(
                    cfg["aoi_prefixo"], na=False)]
            por_regiao[regiao] = desempenho_em_rotulo_local(areal, linhas)
            r = por_regiao[regiao]
            if r.get("avaliado"):
                print(f"  {regiao}: fora do ajuste, contra rotulo local "
                      f"AUC {r['auc_na_regiao']} ({r['aois_acima_do_acaso']}/"
                      f"{r['aois_avaliadas']} AOIs acima do acaso)")
        areal["desempenho_em_rotulo_local_por_regiao"] = por_regiao

    for m in modelos:
        (OUT / f"{m['nome']}.json").write_text(
            json.dumps(declarar_a_ficha(m), indent=2, ensure_ascii=False),
            encoding="utf-8")

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
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--so", nargs="*", default=[],
                     help="construir apenas estes modelos; os demais sao "
                          "PRESERVADOS do disco, com o gerado_em deles")
    raise SystemExit(main(tuple(_ap.parse_args().so)))
