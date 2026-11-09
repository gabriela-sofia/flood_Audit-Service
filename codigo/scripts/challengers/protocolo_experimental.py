"""
Protocolo experimental congelado. Escrito ANTES de qualquer resultado.

-------------------------------------------------------------------------
POR QUE UM PROTOCOLO, e por que ele tem hash
-------------------------------------------------------------------------

A pergunta desta rodada nao e "o REV-P funciona?". E "sob quais condicoes ele
funciona, falha ou deve recusar?". Essas duas perguntas pedem disciplinas
opostas: a primeira premia liberdade analitica, a segunda a destroi.

Com quatro estimadores, seis esquemas de validacao, tres conjuntos e sete
metricas, ha mais de quinhentas combinacoes. Se as escolhas forem feitas
DEPOIS de ver os numeros, sempre existe uma combinacao em que o challenger
ganha -- e nao ha como saber, meses depois, se ela foi encontrada ou
procurada.

Entao tudo que define o experimento e fixado aqui, e o `hash_do_protocolo`
viaja junto de cada resultado. Dois resultados so podem ser comparados quando
os hashes batem; `exigir_mesmo_protocolo` recusa a comparacao silenciosa, que
e a forma mais comum de p-hacking acidental -- ninguem decide fazer, acontece.

-------------------------------------------------------------------------
O QUE ESTE PROTOCOLO NAO FAZ
-------------------------------------------------------------------------

Nao promove nada. Nao existe caminho, aqui ou em qualquer arquivo desta
frente, entre um resultado bom e o modelo servido. `REGRA_DE_NAO_PROMOCAO` e
verificada por teste.

E nao decide sozinho: quando duas especificacoes empatam dentro da tolerancia,
`REGRA_DE_DESEMPATE` manda ficar com a mais simples -- e, no limite, com o
modelo servido. Empate nao e motivo para trocar.
"""

from __future__ import annotations


def _declarar_metadados_de_saida(dados):
    """Metadados de procedencia na copia serializada; nao modifica o objeto de calculo."""
    if not isinstance(dados, dict):
        return dados
    return {"tipo": "registro_protocolo_experimental",
            "gerado_por": 'scripts/challengers/protocolo_experimental.py',
            "nao_e": "Registro do produtor protocolo_experimental: declarar a procedencia nao revalida resultados nem autoriza promocao; conservar os estados e limites especificos do artefato.",
            **dados}


import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))

from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "challengers-protocolo"
TABELA = REPO / "local_runs" / "ds-05-tabela-unica"

SEED = 20260828

# ----------------------------------------------------------------- datasets
#
# Os mesmos arquivos que o SVC-01 le. Nada e recortado aqui de forma nova: o
# recorte por mecanismo e relevo continua sendo o do modelo servido, porque
# comparar estimadores sob um recorte diferente compararia outra pergunta.

DATASETS = {
    "recife_pluvial": {
        "arquivo": "tabela_unica_admitidos_v2.parquet",
        "filtro": "fonte == 'recife'",
        "mecanismo": "PLUVIAL_URBANO",
        "features": ["elevation_m", "slope_deg", "hand_m", "twi_dinf",
                     "rain_max_24h", "rain_decay_index"],
    },
    "fluvial_planicie": {
        "arquivo": "tabela_unica_pool_fluvial_v2.parquet",
        "filtro": ("classe_relevo == 'PLANO_OU_ONDULADO' and "
                   "fonte not in ('curitiba', 'recife')"),
        "mecanismo": "FLUVIAL_ENXURRADA",
        "features": ["elevation_m", "slope_deg", "hand_m", "twi_dinf"],
    },
    "fluvial_serra": {
        "arquivo": "tabela_unica_pool_fluvial_v2.parquet",
        "filtro": ("classe_relevo == 'INGREME' and "
                   "fonte not in ('curitiba', 'recife')"),
        "mecanismo": "FLUVIAL_ENXURRADA",
        "features": ["hand_m", "twi_dinf"],
    },
    # O POOL AREAL, que e o unico com negativo OBSERVADO nas duas classes.
    #
    # Ele entra aqui por uma razao de estrutura, e nao de conveniencia: a
    # matriz de sustentacao cientifica le `experimentos-cientificos/folds.csv`
    # para saber se um escopo tem holdout temporal, e esse arquivo so recebe o
    # que passou pela API canonica. Enquanto o pool areal ficasse fora daqui,
    # o dominio de ajuste de `fluvial_areal_observado` continuaria
    # `not_testable` por "protocolo nao declara teste neste escopo" -- e a
    # razao seria verdadeira, ainda que o teste fosse perfeitamente possivel.
    #
    # O que ha para testar: o pool vai de 2024-05-02 a 2026-03-02 em treze
    # datas, e o corte prospectivo e avaliavel em onze delas. Medido fora da
    # API em 13/09, a AUC sobe de 0,6339 para 0,8034 (mediana por AOI)
    # conforme desastres se acumulam -- mas aquele numero nao entra em matriz
    # nenhuma justamente por ter sido medido fora. Aqui ele passa a ser
    # produzido pelo mesmo caminho dos outros.
    #
    # `elevation_m` fica fora do vetor: no pool o baseline e 0,0 e a elevacao
    # relativa degenera em absoluta, que entre paises mede o pais.
    "fluvial_areal_observado": {
        "arquivo": "tabela_unica_admitidos_v2.parquet",
        "filtro": "fonte == 'cems'",
        "mecanismo": "FLUVIAL_ENXURRADA",
        "features": ["slope_deg", "hand_m", "twi_dinf"],
    },
    # O NEGATIVO OBSERVADO BRASILEIRO, e o que ele muda nao e o tamanho.
    #
    # O `gfd` traz 33.172 pontos admitidos em 24 tiles, com negativo
    # `observado` -- a mesma qualidade do CEMS e do Sen1Floods11, e o unico
    # negativo observado do projeto em territorio brasileiro alem do EMSR720.
    #
    # E ele traz DATA: 16 janelas de evento entre 2002-01-15 e 2017-01-05. Com
    # isso o Rio Grande do Sul DEIXA DE SER a data mais antiga do pool areal,
    # que era a razao estrutural do `not_testable` -- "todo corte prospectivo
    # poe o RS do lado do treino". Vinte e dois anos antes dele existem agora.
    #
    # A resolucao do rotulo e 250 m contra os 30 m da feature, e isso nao se
    # disfarca: `resolucao_nativa_m` carrega os 250 e `precisao_data` carrega
    # `janela_do_evento`, porque a banda `flooded` e o maximo acumulado entre
    # inicio e fim e nenhum pixel tem instante proprio.
    "areal_brasil_observado": {
        "arquivo": "tabela_unica_admitidos_v2.parquet",
        "filtro": "fonte == 'gfd'",
        "mecanismo": "FLUVIAL_ENXURRADA",
        "features": ["slope_deg", "hand_m", "twi_dinf"],
    },
    # As duas fontes areais JUNTAS, que e onde a serie temporal existe.
    #
    # Separadas, cada uma responde uma pergunta truncada: o CEMS cobre
    # 2024-2026 em treze datas e o GFD cobre 2002-2017 em dezesseis, e nenhuma
    # das duas sozinha tem serie longa. Juntas sao 148 grupos e 24 anos.
    #
    # O que isso NAO resolve: as duas medem com instrumentos diferentes --
    # analista sobre radar de 1,5 a 20 m contra MODIS de 250 m --, e um corte
    # prospectivo que separe 2017 de 2024 separa TAMBEM instrumento de
    # instrumento. Quem ler a serie tem de ler isso junto; por isso as duas
    # ficam tambem como conjuntos proprios, e nao so combinadas.
    "areal_observado_multi": {
        "arquivo": "tabela_unica_admitidos_v2.parquet",
        "filtro": "fonte in ('cems', 'gfd')",
        "mecanismo": "FLUVIAL_ENXURRADA",
        "features": ["slope_deg", "hand_m", "twi_dinf"],
    },
    # Curitiba entra como conjunto PROPRIO apenas para a autopsia temporal --
    # nunca como treino de modelo servido. A regiao continua sendo servida por
    # transferencia, e nada aqui muda isso.
    # 03/10/2026: o contrato do fenomeno declara o 156 de Curitiba
    # MISTO_NAO_SEPARAVEL e ele saiu do pool FLUVIAL; a autopsia passou a ler
    # da tabela admitida com o mecanismo do contrato. Mesmas linhas (1.680),
    # mesmas features, mesma pergunta temporal -- so a etiqueta e a fonte do
    # arquivo mudaram, e nenhuma conclusao de mecanismo sai daqui.
    "curitiba_autopsia": {
        "arquivo": "tabela_unica_admitidos_v2.parquet",
        "filtro": "fonte == 'curitiba' and mecanismo == 'MISTO_NAO_SEPARAVEL'",
        "mecanismo": "MISTO_NAO_SEPARAVEL",
        "features": ["elevation_m", "slope_deg", "hand_m", "twi_dinf",
                     "rain_max_24h", "rain_decay_index"],
        "so_para": "diagnostico temporal; nunca vira modelo servido",
    },
}

# --------------------------------------------------------------- unidade
#
# A regra U2 do projeto. Nao e um detalhe de implementacao: em Curitiba o
# `grupo_cv` original vem por `unidade_observacao` -- 1471 grupos para 1680
# linhas -- e reamostrar aquilo e reamostrar LINHA. Todo intervalo sai
# estreito demais por pseudo-replicacao.

UNIDADE_INDEPENDENTE = {
    "regra": "grupo_nunca_mais_fino_que_a_celula",
    "implementacao": "svc01_construir_modelos_servidos.grupo_nunca_mais_fino_que_a_celula",
    "celula_graus": 1e-4,
    "fusao": "componente conexa (union-find) entre grupo_cv e celula",
    "por_que": ("nenhum grupo pode ser mais fino que a celula; sem isso o IC "
                "sai estreito demais por pseudo-replicacao"),
}

# ------------------------------------------------------------ data e cortes

DATA = {
    "coluna": "data_evento",
    "parse": "pd.to_datetime(..., format='mixed', errors='coerce')",
    "por_que_mixed": (
        "sem `format='mixed'` o pandas infere o formato do primeiro elemento "
        "nao nulo e transforma 93,6% das datas validas do pool em NaT -- a "
        "fonte `uk` grava com hora ('2024-11-23 00:00:00') e as demais sem. "
        "Uma analise temporal sobre 6% dos dados nao seria uma analise "
        "temporal"),
    "datas_suspeitas": {
        "2025-01-01": ("656 negativos da fonte `uk` compartilham esta data; "
                       "e sentinela de bloco, nao data de evento observado"),
    },
}

CORTES_TEMPORAIS = {
    # Recife tem a serie mais longa do projeto -- 205 datas distintas entre
    # 2008 e 2025 -- e era o UNICO conjunto sem teste temporal, porque o corte
    # estava fixado em 2026 e a serie dele termina em 2025. O lado de teste
    # saia vazio e a matriz registrava, corretamente,
    # `corte_declarado_nao_materializou_nenhum_fold`.
    #
    # Nao faltava dado: faltava o corte cair dentro da serie. Por ano sao doze
    # cortes; por data, 189. O encadeamento por data entra aqui pelo mesmo
    # motivo do pool areal -- a granularidade anual joga fora estrutura
    # temporal que existe e esta medida.
    "recife_pluvial": {
        "esquema": "forward_temporal",
        "por_que": ("205 datas distintas em treze anos; o corte por ano usa "
                    "doze delas e descarta o resto da serie"),
        "forward_por_data": True,
        "o_que_isso_nao_resolve": (
            "o negativo de Recife continua sendo ausencia de registro, e "
            "nenhum corte temporal conserta isso -- ver "
            "`ref01_relacao_sob_negativo_observado`, onde Recife nao produz "
            "um unico estrato utilizavel"),
    },
    "fluvial_areal_observado": {
        "esquema": "holdout_temporal",
        "treino": "ano <= 2025",
        "teste": "ano == 2026",
        "por_que": ("o pool vai de 2024-05-02 a 2026-03-02 e o corte de 2026 "
                    "deixa 99 AOIs no treino e 25 no teste -- a pergunta "
                    "operacional de servir hoje o que foi ajustado ate ontem"),
        # POR DATA, E NAO POR ANO, e o motivo e do dado e nao do gosto.
        #
        # O pool tem treze datas distintas em tres anos civis. Cortar por ano
        # oferece DOIS cortes, sempre abaixo do minimo de tres que a matriz
        # exige para sair de `inconclusive` -- e o `inconclusive` resultante
        # seria uma afirmacao sobre a granularidade da regua, nao sobre o que
        # o dado sustenta. Por data sao ONZE cortes avaliaveis, cada um com
        # ao menos cinco AOIs de cada lado.
        "forward_por_data": True,
        "forward": [2025, 2026],
        "o_rs_esta_sempre_no_treino": (
            "a ativacao EMSR720 e a data MAIS ANTIGA do pool, entao todo corte "
            "prospectivo a poe do lado do treino. O `not_testable` do Rio "
            "Grande do Sul e estrutural: nao se resolve com mais dado do pool, "
            "so com um segundo evento na propria regiao"),
    },
    "areal_brasil_observado": {
        "esquema": "holdout_temporal",
        "treino": "ano <= 2011",
        "teste": "ano >= 2012",
        "por_que": ("dezesseis janelas de evento entre 2002 e 2017; o corte "
                    "de 2012 deixa nove janelas no treino e sete no teste"),
        "forward_por_data": True,
    },
    "areal_observado_multi": {
        "esquema": "holdout_temporal",
        "treino": "ano <= 2017",
        "teste": "ano >= 2024",
        "por_que": ("o combinado vai de 2002 a 2026 e o corte de 2024 separa "
                    "o que o MODIS viu do que o analista desenhou -- a "
                    "pergunta operacional e tambem, inseparavelmente, uma "
                    "troca de instrumento"),
        "forward_por_data": True,
    },
    "curitiba_autopsia": {
        "esquema": "holdout_temporal",
        "treino": "ano <= 2025",
        "teste": "ano == 2026",
        "por_que": ("o colapso relatado esta em 2026; o corte reproduz a "
                    "pergunta operacional -- servir hoje o que foi ajustado "
                    "ate ontem"),
        "forward": [2024, 2025, 2026],
    },
}

CORTES_ESPACIAIS = {
    "esquema": "spatial_block",
    "bloco_graus": 0.05,
    "por_que": ("bloco maior que a celula de agrupamento e menor que a AOI: "
                "separa vizinhanca sem virar leave-one-region-out disfarcado"),
}

# ----------------------------------------------------------------- features
#
# Lista branca. Uma feature nova entra por decisao documentada, nunca por
# estar disponivel na tabela.

FEATURES_PERMITIDAS = ["elevation_m", "slope_deg", "hand_m", "twi_dinf",
                       "rain_max_24h", "rain_decay_index"]

TRANSFORMACOES_PERMITIDAS = {
    "padronizacao": "media e desvio calculados SO no treino de cada fold",
    "spline": "SplineTransformer grau 3, no maximo 4 nos, ajustado so no treino",
    "interacao": "somente pares pre-especificados em INTERACOES_PRE_ESPECIFICADAS",
    "proibido": ["qualquer transformacao ajustada no conjunto inteiro",
                 "selecao de feature por desempenho no teste",
                 "feature derivada do rotulo",
                 "imputacao com estatistica do teste"],
}

# Interacoes declaradas ANTES de ver resultado, com justificativa fisica. Par
# que nao esteja aqui nao pode ser testado nesta rodada -- e essa e a regra
# que impede a busca por interacao que "funciona".
INTERACOES_PRE_ESPECIFICADAS = [
    {
        "par": ["hand_m", "rain_max_24h"],
        "justificativa_fisica": (
            "a altura acima da drenagem so vira barreira efetiva em relacao ao "
            "volume que chega; a mesma cota protege menos sob chuva maior"),
    },
    {
        "par": ["hand_m", "twi_dinf"],
        "justificativa_fisica": (
            "estar baixo em relacao ao canal e acumular area a montante sao "
            "condicoes que se reforcam: a agua precisa chegar E ter onde ficar"),
    },
    {
        "par": ["slope_deg", "twi_dinf"],
        "justificativa_fisica": (
            "declividade e area de contribuicao sao os dois termos do proprio "
            "indice topografico; testar a interacao e testar se a forma "
            "logaritmica do TWI captura a relacao no dominio observado"),
    },
]

# --------------------------------------------------------------- metricas

METRICAS = {
    "discriminacao": ["auc", "balanced_accuracy"],
    "erro_probabilistico": ["brier"],
    "calibracao": ["calibration_intercept", "calibration_slope"],
    "cobertura": ["coverage", "abstention_rate"],
    "estabilidade": ["auc_desvio_entre_folds"],
    "incerteza": ["auc_ic95_bootstrap_de_grupos"],
    "principal": None,
    "por_que_sem_principal": (
        "nao existe um numero que decida. Um modelo com AUC maior e sinal "
        "fisico invertido nao ganhou; um com AUC igual e calibracao pior "
        "tambem nao. A decisao le o painel inteiro -- ver PAINEL_DE_DECISAO"),
}

CRITERIOS_FISICOS = {
    "sinais_esperados": {
        "elevation_m": -1, "slope_deg": -1, "hand_m": -1,
        "twi_dinf": +1, "rain_max_24h": +1, "rain_decay_index": +1,
    },
    "vereditos": ["consistente", "inconclusivo", "invertido"],
    "regra_inconclusivo": "IC95 do efeito cruza zero",
    "regra_bloqueio": (
        "modelo com sinal `invertido` em variavel de mecanismo nao e elegivel "
        "para revisao humana, por melhor que seja a AUC"),
}

CRITERIOS_CALIBRACAO = {
    "slope_ideal": 1.0,
    "intercepto_ideal": 0.0,
    "tolerancia_slope": 0.15,
    "por_que": ("slope < 1 e confianca excessiva. Num produto que publica IC, "
                "isso importa mais que meio ponto de AUC"),
    "nao_recalibra_servido": True,
}

CRITERIOS_ABSTENCAO = {
    "regra_servida_hoje": "G5: faixa 5-95% do ajuste, por variavel",
    "metodos_comparados": ["faixa_atual", "quantil_univariado",
                           "mahalanobis_robusta", "knn_padronizado",
                           "detector_treino_vs_alvo"],
    "criterio_de_utilidade": (
        "a distancia de dominio so e util se estiver ASSOCIADA a degradacao "
        "observada. Metodo que separa dominio mas nao prediz erro nao vira "
        "regra"),
    "nao_altera_g5": True,
}

# ------------------------------------------------------- modelos candidatos
#
# Complexidade crescente e declarada. M0 e a referencia; os demais so existem
# para responder "o servido esta deixando sinal na mesa?".

MODELOS_CANDIDATOS = [
    {"id": "M0", "nome": "firth_servido", "familia": "linear penalizado",
     "hiperparametros": {"max_iter": 500}, "busca": None,
     "papel": "referencia (estimador servido)"},
    {"id": "M1", "nome": "logistica_l2", "familia": "linear regularizado",
     "hiperparametros": {"class_weight": "balanced", "max_iter": 2000},
     "busca": None, "papel": "challenger linear"},
    {"id": "M2", "nome": "gam_spline_aditivo", "familia": "nao linear aditivo",
     "hiperparametros": {"n_knots": 4, "degree": 3}, "busca": None,
     "papel": "nao linearidade univariada controlada"},
    {"id": "M3", "nome": "interacoes_pre_especificadas",
     "familia": "linear com interacao", "hiperparametros": {},
     "busca": None,
     "papel": "poucas interacoes fisicamente justificaveis"},
    {"id": "M4", "nome": "gbm_monotonico", "familia": "arvore restrita",
     "hiperparametros": {"max_iter": 200, "max_depth": 3,
                         "learning_rate": 0.05},
     "busca": None,
     "papel": "nao linear com restricao fisica de monotonicidade"},
]

# Sem busca de hiperparametro nesta rodada, e a ausencia e a decisao: com os
# tamanhos de amostra aqui, uma busca precisaria de fold interno, e o custo
# em graus de liberdade nao se paga para responder a pergunta desta rodada.
BUSCA_DE_HIPERPARAMETRO = {
    "ativa": False,
    "por_que": ("valores fixos e declarados. Se um dia houver busca, ela roda "
                "em fold INTERNO ao treino -- o holdout final nunca participa "
                "de selecao"),
}

REGRA_DE_DESEMPATE = {
    "ordem": ["coerencia_fisica", "generalizacao_temporal", "calibracao",
              "estabilidade", "simplicidade"],
    "tolerancia_auc": 0.02,
    "regra": ("diferenca de AUC menor que a tolerancia e empate. Em empate "
              "vence o mais simples; empatando com M0, fica M0"),
    "por_que": "empate nao e motivo para trocar o que ja esta servido",
}

REGRA_DE_NAO_PROMOCAO = {
    "vale_para": "toda esta frente experimental",
    "regra": ("nenhum resultado desta frente altera modelo servido, contrato, "
              "maturidade de regiao ou portao G5"),
    "saida_maxima": "eligible_for_human_review = true/false",
    "proibido": ["promovido", "promovivel", "champion automatico",
                 "recalibrar o contrato servido"],
}

PAINEL_DE_DECISAO = [
    "temporal_generalization", "spatial_generalization", "calibration",
    "physical_coherence", "domain_coverage", "complexity", "stability",
]

# Hipoteses da autopsia de Curitiba, escritas ANTES de calcular. Cada uma
# termina em supported / not_supported / inconclusive -- nunca `proved`.
HIPOTESES = [
    {"id": "H1", "enunciado": "2026 apresenta covariate shift mensuravel",
     "como_testar": "SMD, KS e PSI por feature entre <=2025 e 2026; "
                    "discriminador de periodo",
     "refuta_se": "todas as features com |SMD| pequeno e AUC do discriminador "
                  "proxima de 0,5"},
    {"id": "H2", "enunciado": "o colapso e concentrado em subconjunto espacial",
     "como_testar": "AUC por bloco espacial de 0,05 grau no periodo de teste",
     "refuta_se": "degradacao homogenea entre blocos"},
    {"id": "H3", "enunciado": "as relacoes fisicas mantem o sinal, mas a "
                              "fronteira e nao linear",
     "como_testar": "M2/M4 contra M0 no MESMO holdout temporal",
     "refuta_se": "nao linearidade nao melhora o holdout temporal"},
    {"id": "H4", "enunciado": "as relacoes fisicas mudam de sinal ou magnitude "
                              "entre periodos",
     "como_testar": "coeficiente por periodo com IC bootstrap de grupos",
     "refuta_se": "ICs se sobrepoem e nenhum sinal inverte"},
    {"id": "H5", "enunciado": "o problema principal e calibracao, nao ranking",
     "como_testar": "AUC vs Brier/slope no holdout; recalibracao review-only",
     "refuta_se": "AUC cai junto com a calibracao"},
    {"id": "H6", "enunciado": "a chuva introduz separacao temporal incompativel",
     "como_testar": "modelo com e sem features de chuva no mesmo holdout",
     "refuta_se": "remover chuva nao altera a degradacao temporal"},
    {"id": "H7", "enunciado": "existe subconjunto de dominio onde o modelo "
                              "permanece util",
     "como_testar": "AUC por faixa de distancia de dominio no periodo de teste",
     "refuta_se": "nenhuma faixa mantem AUC util"},
]

VAZAMENTOS_PROIBIDOS = [
    "observacao do mesmo grupo em treino e teste",
    "observacao do mesmo evento/data em treino e teste no esquema temporal",
    "padronizacao ajustada fora do treino",
    "selecao de feature ou hiperparametro usando o holdout final",
    "feature derivada do rotulo",
    "periodo de teste presente no treino",
]


def _hash_dos_datasets() -> dict:
    return {nome: sha256_arquivo(TABELA / cfg["arquivo"])
            for nome, cfg in DATASETS.items()}


def protocolo() -> dict:
    corpo = {
        "tipo": "protocolo_experimental_revp",
        "versao": "1",
        "seed": SEED,
        "datasets": DATASETS,
        "unidade_independente": UNIDADE_INDEPENDENTE,
        "data": DATA,
        "cortes_temporais": CORTES_TEMPORAIS,
        "cortes_espaciais": CORTES_ESPACIAIS,
        "features_permitidas": FEATURES_PERMITIDAS,
        "transformacoes_permitidas": TRANSFORMACOES_PERMITIDAS,
        "interacoes_pre_especificadas": INTERACOES_PRE_ESPECIFICADAS,
        "metricas": METRICAS,
        "criterios_fisicos": CRITERIOS_FISICOS,
        "criterios_calibracao": CRITERIOS_CALIBRACAO,
        "criterios_abstencao": CRITERIOS_ABSTENCAO,
        "modelos_candidatos": MODELOS_CANDIDATOS,
        "busca_de_hiperparametro": BUSCA_DE_HIPERPARAMETRO,
        "regra_de_desempate": REGRA_DE_DESEMPATE,
        "regra_de_nao_promocao": REGRA_DE_NAO_PROMOCAO,
        "painel_de_decisao": PAINEL_DE_DECISAO,
        "hipoteses": HIPOTESES,
        "vazamentos_proibidos": VAZAMENTOS_PROIBIDOS,
    }
    # O hash cobre o PROTOCOLO, e nao os dados: trocar a tabela sem trocar o
    # desenho e uma mudanca real, mas de outra natureza, e por isso os hashes
    # dos arquivos ficam FORA do hash do protocolo e viajam ao lado dele.
    corpo["hash_do_protocolo"] = sha256_json(corpo)
    corpo["hash_dos_datasets"] = _hash_dos_datasets()
    return corpo


class ProtocoloDivergente(RuntimeError):
    """Comparar resultados de protocolos diferentes e comparar outra coisa."""


def exigir_mesmo_protocolo(*resultados) -> str:
    """Recusa comparar resultados gerados sob protocolos diferentes.

    E a trava contra o p-hacking que ninguem decide fazer: uma tabela que
    junta o holdout de ontem com o de hoje, depois de o corte ter mudado,
    parece uma comparacao e nao e.
    """
    hashes = {r.get("hash_do_protocolo") for r in resultados}
    # A AUSENCIA vem PRIMEIRO, e a ordem importa: "nao sei sob qual desenho
    # isto foi gerado" e um problema diferente e mais grave que "os desenhos
    # diferem", e reporta-lo como divergencia esconderia a causa real.
    if None in hashes:
        raise ProtocoloDivergente(
            "resultado sem `hash_do_protocolo`: nao ha como saber sob qual "
            "desenho ele foi gerado")
    if len(hashes) > 1:
        # `sorted` sobre um conjunto com None e str levantaria TypeError e
        # engoliria a mensagem util; `key=str` mantem o erro legivel.
        raise ProtocoloDivergente(
            "resultados gerados sob protocolos diferentes: "
            f"{sorted(hashes, key=str)}; nao sao comparaveis")
    return hashes.pop()


def conferir_contra_o_congelado(saida: Path = SAIDA) -> dict:
    """O desenho em codigo e o desenho congelado em disco sao o mesmo?

    -------------------------------------------------------------------------
    POR QUE ESTA FUNCAO PRECISOU EXISTIR
    -------------------------------------------------------------------------

    `exigir_mesmo_protocolo` compara resultados ENTRE SI, e nunca contra o
    disco. Num mesmo run todos os resultados carregam o hash do codigo por
    construcao, entao a trava sempre passou -- e passou enquanto o arquivo
    congelado em 13/09 declarava 4 datasets e o codigo passava a declarar 7,
    incluindo o `areal_brasil_observado`, que sustenta o projeto inteiro.

    O efeito pratico: a frase "avaliado sob protocolo pre-registrado" era falsa
    para os conjuntos acrescentados depois do congelamento, e nada acusava.

    -------------------------------------------------------------------------
    POR QUE ELA NAO LEVANTA, E POR QUE `gravar` NAO ENTRA NA CADEIA
    -------------------------------------------------------------------------

    Recongelar a cada rodada destruiria a propriedade que o congelamento existe
    para dar: um desenho fixado DEPOIS de ver o resultado nao e pre-registro, e
    uma cadeia que recongela sozinha produz exatamente isso, com carimbo.

    Entao o congelamento continua sendo um ato deliberado -- alguem roda
    `python scripts/challengers/protocolo_experimental.py` -- e o que a cadeia
    faz e CONFERIR. A divergencia e reportada, nao levantada: ela e um achado
    sobre o estado do projeto, e derrubar a campanha no meio nao a conserta.
    """
    arq = saida / "protocolo.json"
    p = protocolo()
    if not arq.exists():
        return {"congelado_existe": False, "igual": False,
                "por_que": "nenhum desenho congelado em disco"}
    d = json.loads(arq.read_text(encoding="utf-8"))
    divergentes = [k for k in p
                   if k not in ("hash_dos_datasets", "hash_do_protocolo")
                   and sha256_json(p.get(k)) != sha256_json(d.get(k))]
    return {
        "congelado_existe": True,
        "igual": p["hash_do_protocolo"] == d.get("hash_do_protocolo"),
        "hash_em_codigo": p["hash_do_protocolo"],
        "hash_congelado": d.get("hash_do_protocolo"),
        "congelado_em_utc": d.get("congelado_em_utc"),
        "chaves_divergentes": divergentes,
        "datasets_no_codigo_e_nao_no_congelado": [
            x for x in p.get("datasets", {}) if x not in d.get("datasets", {})],
        "o_que_a_divergencia_significa": (
            "conjunto que esta no codigo e nao no congelado NAO foi avaliado "
            "sob protocolo pre-registrado, e a frase nao vale para ele"),
    }


def gravar(saida: Path = SAIDA) -> dict:
    """Congela o desenho. Ato deliberado, com o hash anterior preservado.

    O congelamento novo NAO cobre resultado ja produzido: quem rodou antes
    rodou sob outro desenho, e dizer o contrario seria pre-registro retroativo,
    que e a fraude que o pre-registro existe para impedir. Por isso o arquivo
    carrega `substitui` e `nao_cobre`, escritos no proprio artefato.
    """
    anterior = None
    arq = saida / "protocolo.json"
    if arq.exists():
        try:
            velho = json.loads(arq.read_text(encoding="utf-8"))
            anterior = {"hash": velho.get("hash_do_protocolo"),
                        "congelado_em_utc": velho.get("congelado_em_utc"),
                        "datasets": list(velho.get("datasets") or [])}
        except Exception:
            anterior = {"hash": None, "erro": "congelado anterior ilegivel"}
    p = protocolo()
    saida.mkdir(parents=True, exist_ok=True)
    arq.write_text(
        json.dumps(_declarar_metadados_de_saida({**p,
                    "congelado_em_utc": agora_utc(),
                    "substitui": anterior,
                    "nao_cobre": (
                        "resultado gerado ANTES deste congelamento nao foi "
                        "pre-registrado sob ele; o hash que cada resultado "
                        "carrega diz sob qual desenho ele nasceu")}),
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    return p


def main() -> int:
    p = gravar()
    print(json.dumps({
        "hash_do_protocolo": p["hash_do_protocolo"],
        "datasets": list(p["datasets"]),
        "modelos": [m["id"] for m in p["modelos_candidatos"]],
        "hipoteses": [h["id"] for h in p["hipoteses"]],
        "saida": str(SAIDA),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
