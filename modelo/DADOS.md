# Dados

## Escopo

Este arquivo descreve a base consolidada usada pelos scripts publicos. O pacote inclui
manifestos, resumos e artefatos de servico. As tabelas completas da tabela unica nao sao
versionadas neste repositorio por tamanho.

## Tabela unica v2

Fonte: `modelo/execucoes/ds-05-tabela-unica/manifesto_v2.json`.

| Item | Valor |
|---|---:|
| Observacoes de entrada | 595.865 |
| Observacoes admitidas | 568.265 |
| Pool fluvial | 65.070 |
| Grupos admitidos | 6.873 |
| Grupos no pool fluvial | 2.002 |

## Arquivos completos registrados por manifesto

| Arquivo | Tamanho registrado |
|---|---:|
| `tabela_unica_v2.csv` | 218.051.209 bytes |
| `tabela_unica_admitidos_v2.csv` | 207.116.610 bytes |
| `tabela_unica_pool_fluvial_v2.csv` | 24.359.964 bytes |
| `rejeitados_com_motivo.csv` | 2.115.746 bytes |

Esses arquivos sao produtos reprodutiveis da cadeia, mas nao fazem parte do pacote
publico enxuto. O manifesto tambem registra o SHA-256 de cada um, alem da versao v1
anterior, para que uma reproducao possa ser conferida sem os arquivos em maos.

## Esquema alvo

Fonte: `modelo/execucoes/ds-03-esquema/esquema_alvo_v1.json`. Versao `v1`, 24 colunas.

Classes de rotulo:

| Codigo | Classe |
|---:|---|
| 1 | `INUNDACAO` |
| 2 | `MOVIMENTO_DE_MASSA` |
| 0 | `NAO_INUNDADO` |
| 9 | `AGUA_PERMANENTE` |
| −1 | `SEM_DADO` |

Variaveis fisicas: `elevation_m`, `slope_deg`, `hand_m`, `twi_dinf`, `rain_max_24h`,
`rain_decay_index`. As quatro primeiras que vem do terreno formam o bloco
`variaveis_terreno`.

### Regra de ouro

> Nenhuma coluna de valor sem a coluna de procedencia correspondente; ausencia e nulo
> declarado, nunca estimativa.

### Criterios de admissao

| Criterio | Regra |
|---|---|
| A — dentro da AOI | O ponto tem AOI declarada e coordenada finita dentro dela. |
| B — mesma cadeia | As quatro variaveis de terreno vem de uma cadeia declarada e nao `global`. Global e outro **instrumento**, nao outra resolucao. |
| C — grupo identificavel | Existe `grupo_cv` nao nulo, para que a divisao de folds nao parta um grupo. |

### Resolucao unica

Desde 13/08/2026 a resolucao e unica em todo o projeto: **30 m, cadeia WhiteboxTools,
limiar de canal 0,1123 km2**. Isso substitui a regra anterior de resolucao por
mecanismo, e linha canonica fora de `wbt30` passa a ser erro de reducao.

O motivo e medido, nao estetico: TWI a 10 m e TWI a 30 m **nao sao a mesma variavel**
(pearson 0,293 em Recife, 0,205 em Curitiba). A regra antiga era "ou todo o TWI vem de
30 m, ou o TWI sai do conjunto"; com resolucao unica ela passa a ser satisfeita por
construcao, e nao por vigilancia.

### Regra da chuva

Duas fontes de precipitacao nao podem ocupar a mesma coluna. Um pool so e licito dentro
de uma unica `fonte_chuva`.

## Criterio do pool fluvial

O pool fluvial da tabela unica v2 e definido por:

```text
cadeia_terreno = wbt30
E mecanismo = FLUVIAL_ENXURRADA
E as quatro variaveis de terreno presentes
```

Variaveis exigidas: `elevation_m`, `slope_deg`, `hand_m`, `twi_dinf`.

Esse criterio garante que os modelos fluviais sejam ajustados apenas com dados
compativeis com o mecanismo fisico declarado. NYC, Sao Paulo, UFO e Recife nao entram no
pool: 65.070 dos 568.265 pontos admitidos sao elegiveis, distribuidos em 2.002 grupos.

## Fontes admitidas

| Fonte | Entrada | Admitidos | Pool fluvial |
|---|---:|---:|---:|
| NYC | 471.727 | 471.727 | 0 |
| Sen1Floods11 | 45.340 | 30.431 | 30.431 |
| CEMS | 36.418 | 25.483 | 25.483 |
| Urban Flood Observations | 25.800 | 24.044 | 0 |
| UK | 7.476 | 7.476 | 7.476 |
| Sao Paulo | 7.146 | 7.146 | 0 |
| Curitiba | 1.680 | 1.680 | 1.680 |
| Recife | 278 | 278 | 0 |

## Fontes fora da tabela unica

Estas fontes existem no pacote e NAO entram na tabela unica nem em modelo servido. Elas
sustentam auditorias de chuva e de transferencia.

| Fonte | Papel | Manifesto |
|---|---|---|
| Chicago 311 | Casos e fundo para o teste de transferencia entre cidades | `modelo/execucoes/chicago-teste-da-lei/insumos/aquisicao_chamados_manifesto.json` |
| MDT de Chicago e leitura de terreno nos pontos | Terreno da segunda cidade, pela mesma cadeia D-infinity 30 m | `modelo/execucoes/chicago-teste-da-lei/insumos/terreno_nos_pontos_manifesto.json` |
| MRMS 1 km, Chicago | Chuva de radar calibrada por pluviometro | `modelo/execucoes/chicago-teste-da-lei/insumos/chuva_mrms_manifesto.json` |
| MRMS 1 km, Nova York | Chuva de radar calibrada por pluviometro | `modelo/execucoes/nyc-terreno-sob-gatilho/insumos/chuva_mrms_manifesto.json` |
| Series por celula e casos das centrais | Insumos do autopareamento temporal | `modelo/execucoes/chuva-autopareamento/insumos/` |

O MRMS cobre apenas os Estados Unidos. Nenhum resultado que dependa dele transfere
sozinho para regiao brasileira, e os artefatos declaram isso.

## Regras de uso

1. Nao imputar observacoes ausentes.
2. Nao tratar fonte sem dado local como validacao negativa.
3. Nao misturar vulnerabilidade social com suscetibilidade fisica.
4. Nao promover modelo por evidencia auxiliar sem alteracao explicita do contrato.
5. Preservar `not_testable` e `inconclusive` como estados cientificos distintos.
6. Nao rebaixar modelo servido por auditoria auxiliar: uma reprovacao de transferencia entra na documentacao e nas limitacoes, nao nos pesos.

## Limitacoes conhecidas da base

| Limitacao | Onde esta registrada |
|---|---|
| `unidade_agrupamento` recebe `chamado` em 478.873 linhas, fora do dominio declarado | `violacoes_de_contrato` no manifesto v2 |
| Pontos empilhados na mesma celula dentro da mesma fonte (402.714 em NYC, 1.124 em Sao Paulo, 812 em Curitiba, 430 em Sen1Floods11, 92 em CEMS, 86 em UFO, 71 em Recife) | `duplicidade.intra_fonte_mesma_celula` |
| Nenhum ponto de Petropolis | Ausencia da fonte na contagem por fonte |

O empilhamento e a razao pela qual o grupo de reamostragem nunca pode ser mais fino que
a celula: duas linhas na mesma coordenada, com as mesmas variaveis e a mesma classe, nao
sao duas observacoes independentes.

## Artefatos incluidos

Os artefatos curados ficam em `modelo/execucoes/`:

| Diretorio | Conteudo |
|---|---|
| `ds-03-esquema/` | Esquema alvo da tabela unica. |
| `ds-05-tabela-unica/` | Manifesto v2 e contagem por fonte. |
| `aud-chuva-02/` | Auditoria da escala do contraste de chuva por fonte. |
| `mod-prosp-02/` | Holdout temporal sobre a tabela unica. |
| `mod-serra-03/` | Selecao do modelo de relevo ingreme. |
| `svc-01-modelos/` | Modelos candidatos, indice e comandos. |
| `svc-02-contrato/` | Demonstracoes e cobertura do contrato. |
| `svc-03-grade/` | Grades, figuras e resumo da aplicacao a 120 m. |
| `chuva-autopareamento/` | Autopareamento temporal, matriz de forcamento e manifestos dos insumos. |
| `nyc-terreno-sob-gatilho/` | Efeito do terreno sob o gatilho de chuva em Nova York. |
| `chicago-teste-da-lei/` | Teste de transferencia do efeito para a segunda cidade. |
| `temporalidade/` | Auditorias temporais por recorte. |
| `temporalidade-integracao/` | Matrizes de suporte e reconciliacao. |
| `neural-catraca/` | Experimento confirmatorio de reconstrucao mascarada. |
| `sintese-neural/` | Sintese da cadeia seletiva. |
| `recuperacao/` | Comparacao de recuperadores de fonte. |
| `camada-social-territorial/` | Camada social v0 em revisao. |
