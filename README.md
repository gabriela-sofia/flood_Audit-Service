# Suscetibilidade urbana a inundacoes

Da evidencia heterogenea ao servico de inferencia auditavel.

Este repositorio organiza uma cadeia reprodutivel para estimar suscetibilidade
fisico-hidrologica a inundacoes em areas urbanas. O projeto separa tres coisas que
nao devem ser confundidas: o dado de entrada, o ajuste estatistico treinado com
fontes curadas e o contrato de inferencia que decide quando uma resposta pode ser
entregue.

Suscetibilidade, aqui, e a predisposicao fisica do terreno a receber, concentrar ou
reter agua sob um dado contexto de chuva. A resposta do servico nao e promessa de
ocorrencia, alerta em tempo real, diagnostico individual ou medida de vulnerabilidade
social. Quando a evidencia local nao permite uma afirmacao, o repositorio registra a
lacuna em vez de transformar ausencia de dado em resultado negativo.

`REV-P` e o nome interno do projeto. Ele aparece em prefixos de arquivos, estagios e
artefatos historicos, mas nao altera o escopo tecnico do repositorio.

## Comece por aqui

| Se voce quer | Leia |
|---|---|
| Entender a cadeia tecnica e os portoes de resposta | [`docs/METODO.md`](docs/METODO.md) |
| Ver o que esta provado, o que e inconclusivo e o que nao foi testado | [`docs/EVIDENCIA.md`](docs/EVIDENCIA.md) |
| Conferir fontes, recortes e limites de uso | [`docs/FONTES.md`](docs/FONTES.md) |
| Consultar o estado atual do pacote publico | [`docs/ESTADO_ATUAL.md`](docs/ESTADO_ATUAL.md) |
| Ver os artefatos curados incluidos nesta versao | [`docs/ARTEFATOS_CURADOS.md`](docs/ARTEFATOS_CURADOS.md) |
| Entender os modelos servidos | [`modelo/MODELO.md`](modelo/MODELO.md) |
| Entender a base de dados consolidada | [`modelo/DADOS.md`](modelo/DADOS.md) |

## Estado atual

O pacote publico contem os scripts de treino curado, os scripts de servico e os
artefatos consolidados necessarios para auditar o estado atual do projeto.

| Frente | Estado |
|---|---|
| Tabela unica | Manifesto v2 consolidado com 595.865 observacoes de entrada, 568.265 admitidas e 65.070 pontos no pool fluvial. As tabelas completas nao entram no pacote publico por tamanho; o manifesto registra os arquivos, tamanhos e checksums esperados. |
| Modelos servidos | Cinco candidatos auditados. Quatro sao serviveis pelo contrato atual: Recife pluvial, NYC pluvial urbano, fluvial de planicie e fluvial de serra. Sao Paulo pluvial urbano fica registrado como nao servivel. |
| Contrato de inferencia | Implementado como funcao pura, com portoes em ordem declarada, maturidade regional, distancia de dominio, explicacao estruturada e recusa quando o caso nao atende ao contrato. |
| Grade de aplicacao | Gerada a 120 m para Recife, Curitiba e Petropolis, com CSV, GeoTIFF, PNG e resumo JSON. |
| Chuva | Analise caso-controle temporal suporta associacao entre chuva antecedente e eventos em NYC e Sao Paulo. Esse resultado qualifica o forcamento, mas nao promove modelo de suscetibilidade sozinho. |
| Temporalidade | Matrizes registram casos `supported`, `inconclusive` e `not_testable`. Onde o protocolo nao separa efeito temporal de ruido ou confundimento, o resultado permanece inconclusivo. |
| Camada social | Camada territorial v0 para Curitiba em modo de revisao. Ela nao altera escore fisico, nao treina modelo e nao cria rotulo hidrologico. |
| Recuperacao de fonte | Recuperador denso supera busca lexical na tarefa auditada. Esse componente serve para rastreabilidade de fontes, nao para alterar portoes ou escores. |

## Resultados numericos versionados

| Modelo | Uso no contrato | n | grupos | AUC | Veredito |
|---|---:|---:|---:|---:|---|
| Recife pluvial | Recife | 269 | 227 | 0,6422 | Servivel, fora dos criterios fortes |
| Pluvial urbano NYC | NYC | 471.727 | 751 | 0,6089 | Servivel, fora dos criterios fortes |
| Pluvial urbano Sao Paulo | Nao servido | 7.146 | 1.438 | 0,4753 | Abaixo do acaso |
| Fluvial planicie | Curitiba | 56.654 | 509 | 0,7334 | Coerente com criterios |
| Fluvial serra | Petropolis | 5.162 | 24 | 0,7916 | Coerente com criterios |

| Regiao | Celulas validas | Celulas serviveis | Mediana do escore | p90 | Maturidade |
|---|---:|---:|---:|---:|---|
| Recife | 56.666 | 54.963 | 0,5808 | 0,7785 | `mvp_local` |
| Curitiba | 65.275 | 64.310 | 0,2746 | 0,5804 | `transferencia_caracterizada` |
| Petropolis | 172.015 | 157.005 | 0,3460 | 0,5634 | `transferencia_sem_referencia_local` |

## Estrutura

```text
docs/                    metodo, evidencia, fontes e estado publico
modelo/                  cartao do modelo, ficha dos dados e artefatos curados
modelo/execucoes/        saidas versionadas de treino, contrato, grade e auditorias
modelo/grade/            grade publica por regiao a 120 m
scripts/treino/          scripts curados de preparacao, treino e auditoria
scripts/servico/         modelos servidos, contrato de inferencia e grade
scripts/terreno/         cadeia fisico-hidrologica de terreno
scripts/externo/         rotinas de aquisicao e preparacao de fontes externas
scripts/social/          camada social territorial em revisao
scripts/temporalidade/   auditorias temporais e ontologia de estados
tests/                   testes de regressao da cadeia publica
```

## Executar

```bash
conda env create -f environment.yml
conda activate revp-susc
python -m pytest tests -q
```

Alguns testes pulam quando a tabela unica completa nao esta presente no checkout. Isso
e esperado no pacote publico: os artefatos grandes sao descritos por manifesto e podem
ser regenerados pela cadeia de scripts curados.
