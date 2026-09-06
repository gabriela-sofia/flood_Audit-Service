# Artefatos curados

Este arquivo lista o que entra no pacote publico como resultado curado. A lista nao
substitui os manifestos; ela orienta revisao e manutencao.

## Treino e consolidacao

| Caminho | Papel |
|---|---|
| `modelo/execucoes/ds-05-tabela-unica/manifesto_v2.json` | Manifesto da tabela unica v2. |
| `modelo/execucoes/ds-05-tabela-unica/relatorio_contagem_por_fonte.csv` | Contagem resumida por fonte. |
| `scripts/treino/ds01_*` a `scripts/treino/ds05_*` | Consolidacao e verificacao da tabela unica. |
| `scripts/treino/chuva02_*` a `scripts/treino/chuva04_*` | Auditoria de chuva antecedente. |
| `scripts/treino/aud_*` | Auditorias de desenho, chuva e integridade. |

## Servico

| Caminho | Papel |
|---|---|
| `modelo/execucoes/svc-01-modelos/indice.json` | Indice dos modelos candidatos e servidos. |
| `modelo/execucoes/svc-02-contrato/` | Demonstracoes e cobertura do contrato. |
| `modelo/execucoes/svc-03-grade/resultado.json` | Resumo das grades por regiao. |
| `modelo/grade/` | Grades publicas em CSV, GeoTIFF e PNG. |
| `scripts/servico/svc01_construir_modelos_servidos.py` | Construcao dos modelos servidos. |
| `scripts/servico/svc02_contrato_inferencia.py` | Contrato de inferencia. |
| `scripts/servico/svc03_grade_suscetibilidade.py` | Geracao da grade publica. |

## Evidencias auxiliares

| Caminho | Papel |
|---|---|
| `modelo/execucoes/chuva-case-crossover/` | Associacao entre chuva antecedente e eventos nos recortes testados. |
| `modelo/execucoes/temporalidade/` | Auditorias temporais por protocolo. |
| `modelo/execucoes/temporalidade-integracao/` | Matriz de suporte, reconciliacao e estado temporal. |
| `modelo/execucoes/recuperacao/` | Comparacao de recuperadores de fonte. |
| `modelo/execucoes/neural-catraca/` | Experimento confirmatorio de reconstrucao mascarada. |
| `modelo/execucoes/sintese-neural/` | Sintese seletiva da frente neural. |
| `modelo/execucoes/camada-social-territorial/` | Camada social territorial v0 em revisao. |

## Criterio de inclusao

Um resultado entra neste pacote quando atende a pelo menos uma das condicoes:

1. E consumido diretamente pelo contrato de inferencia.
2. Explica uma grade publica versionada.
3. Documenta uma evidencia auxiliar citada no planejamento ou no poster.
4. Registra uma limitacao que impede afirmacao mais forte.
5. Resume arquivo grande que nao deve ser versionado integralmente.

Resultados exploratorios sem ligacao com essas condicoes permanecem fora do pacote
curado.
