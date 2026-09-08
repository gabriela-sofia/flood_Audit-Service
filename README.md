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
| Aprender o projeto do zero, ou ensina-lo | [`docs/ROTEIRO_DE_ESTUDO.md`](docs/ROTEIRO_DE_ESTUDO.md) |
| Entender o vocabulario antes dos numeros | [`docs/CONCEITOS.md`](docs/CONCEITOS.md) |
| Entender a cadeia tecnica e os portoes de resposta | [`docs/METODO.md`](docs/METODO.md) |
| Ver como as partes se encaixam e o caminho de uma celula | [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) |
| Ver o que esta provado, o que e inconclusivo, o que nao foi testado e o que foi reprovado | [`docs/EVIDENCIA.md`](docs/EVIDENCIA.md) |
| Entender como se mede, e como nao se mede | [`docs/VALIDACAO.md`](docs/VALIDACAO.md) |
| Ler as limitacoes reunidas num lugar so | [`docs/LIMITACOES.md`](docs/LIMITACOES.md) |
| Ver o retrato de cada regiao | [`docs/REGIOES.md`](docs/REGIOES.md) |
| Saber o que pode e o que nao pode ser afirmado | [`docs/AFIRMACOES.md`](docs/AFIRMACOES.md) |
| Conferir fontes, recortes e limites de uso | [`docs/FONTES.md`](docs/FONTES.md) |
| Consultar o estado atual do pacote publico | [`docs/ESTADO_ATUAL.md`](docs/ESTADO_ATUAL.md) |
| Ver os artefatos curados incluidos nesta versao | [`docs/ARTEFATOS_CURADOS.md`](docs/ARTEFATOS_CURADOS.md) |
| Entender os modelos servidos, com pesos e dominio | [`modelo/MODELO.md`](modelo/MODELO.md) |
| Entender a base de dados consolidada e o esquema | [`modelo/DADOS.md`](modelo/DADOS.md) |

## Estado atual

O pacote publico contem os scripts de treino curado, os scripts de servico, os scripts
das frentes de chuva e transferencia, e os artefatos consolidados necessarios para
auditar o estado atual do projeto.

| Frente | Estado |
|---|---|
| Tabela unica | Manifesto v2 consolidado com 595.865 observacoes de entrada, 568.265 admitidas e 65.070 pontos no pool fluvial. As tabelas completas nao entram no pacote publico por tamanho; o manifesto registra os arquivos, tamanhos e checksums esperados. |
| Modelos servidos | Cinco candidatos auditados. Quatro sao serviveis pelo contrato atual: Recife pluvial, NYC pluvial urbano, fluvial de planicie e fluvial de serra. Sao Paulo pluvial urbano fica registrado como nao servivel. |
| Contrato de inferencia | Implementado como funcao pura, com portoes em ordem declarada, maturidade regional, distancia de dominio, explicacao estruturada e recusa quando o caso nao atende ao contrato. |
| Grade de aplicacao | Gerada a 120 m para Recife, Curitiba e Petropolis, com CSV, GeoTIFF, PNG e resumo JSON. A chuva entra como cenario, nao como camada. |
| Chuva como forcamento | Analise autopareada no tempo sustenta associacao entre chuva antecedente e ocorrencia notificada no mesmo lugar, em NYC e Sao Paulo. Esse resultado qualifica o forcamento, mas nao promove modelo de suscetibilidade sozinho. |
| Transferencia entre cidades | Testada com o mesmo estimador em Nova York e Chicago. Chuva e declividade replicam. `twi_dinf`, a interacao `twi_dinf` x chuva e `elevation_rel_m` invertem de sinal na segunda cidade. **A transferencia esta reprovada neste teste.** |
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

Os escores das tres regioes vem de tres modelos diferentes, sem calibracao comum, e nao
sao comparaveis entre si. O de Recife ainda depende do cenario de chuva escolhido.

| Termo | Nova York | Chicago | Replicou |
|---|---:|---:|---|
| `mrms_qpe_24h_mm` | +0,1886 | +0,4416 | sim |
| `slope_deg` | −0,0151 | −0,0111 | sim |
| `twi_dinf` | +0,0390 | −0,0102 | **nao, inverteu** |
| `twi_dinf` x chuva | +0,0397 | −0,0110 | **nao, inverteu** |
| `elevation_rel_m` | −0,0088 | +0,0397 | **nao, inverteu** |

## Limitacoes principais

O detalhamento esta em [`docs/LIMITACOES.md`](docs/LIMITACOES.md). Em resumo:

- nenhuma das tres regioes servidas tem inventario local usado como criterio de aprovacao;
- Petropolis nao tem uma unica linha na tabela unica: o escore dela e predicao por semelhanca de terreno;
- em Recife, 98,4% dos negativos sao ausencia de registro, e apenas 3,7% dos pontos estao em datas que contem as duas classes;
- o unico teste de transferencia entre cidades reprovou o sinal de `twi_dinf` e de `elevation_rel_m`;
- a temporalidade permanece `inconclusive` ou `not_testable` em todos os escopos avaliados;
- a tabela unica registra uma violacao de dominio aberta em `unidade_agrupamento`.

## Estrutura

```text
docs/                    conceitos, metodo, arquitetura, evidencia, validacao,
                         limitacoes, regioes, afirmacoes, fontes, estado e roteiro
modelo/                  cartao do modelo, ficha dos dados e artefatos curados
modelo/execucoes/        saidas versionadas de treino, contrato, grade e auditorias
scripts/treino/          scripts curados de preparacao, treino e auditoria
scripts/servico/         modelos servidos, contrato de inferencia, grade e proveniencia
scripts/chuva/           contrato de chuva, autopareamento temporal e terreno sob gatilho
scripts/terreno/         cadeia fisico-hidrologica de terreno
scripts/externo/         rotinas de aquisicao e preparacao de fontes externas
scripts/social/          camada social territorial em revisao
scripts/temporalidade/   auditorias temporais e ontologia de estados
scripts/neural/          cadeia seletiva das frentes auxiliares
scripts/dino/            rotinas historicas de revisao visual, fora do contrato servido
outputs_public/          pacote historico por estagio, anterior a esta organizacao
tests/                   testes de regressao da cadeia publica
```

## Executar

```bash
conda env create -f environment.yml
conda activate revp-susc
python -m pytest tests -q
```

Alguns testes pulam quando a tabela unica completa nao esta presente no checkout, e
outros pulam porque guardam contratos de camadas que ficam fora deste pacote publico
(apresentacao da resposta, tela e reajuste periodico). Os dois casos declaram o motivo
do salto. Os artefatos grandes sao descritos por manifesto e podem ser regenerados pela
cadeia de scripts curados.
