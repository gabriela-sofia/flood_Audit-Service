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
publico enxuto.

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

## Regras de uso

1. Nao imputar observacoes ausentes.
2. Nao tratar fonte sem dado local como validacao negativa.
3. Nao misturar vulnerabilidade social com suscetibilidade fisica.
4. Nao promover modelo por evidencia auxiliar sem alteracao explicita do contrato.
5. Preservar `not_testable` e `inconclusive` como estados cientificos distintos.

## Artefatos incluidos

Os artefatos curados ficam em `modelo/execucoes/`:

| Diretorio | Conteudo |
|---|---|
| `svc-01-modelos/` | Modelos candidatos, indice e comandos. |
| `svc-02-contrato/` | Demonstracoes e cobertura do contrato. |
| `svc-03-grade/` | Grades, figuras e resumo da aplicacao a 120 m. |
| `ds-05-tabela-unica/` | Manifesto e contagem por fonte. |
| `chuva-case-crossover/` | Auditoria de chuva antecedente. |
| `temporalidade/` | Auditorias temporais por recorte. |
| `temporalidade-integracao/` | Matrizes de suporte e reconciliacao. |
| `neural-catraca/` | Experimento confirmatorio de reconstrucao mascarada. |
| `sintese-neural/` | Sintese da cadeia seletiva. |
| `recuperacao/` | Comparacao de recuperadores de fonte. |
| `camada-social-territorial/` | Camada social v0 em revisao. |
