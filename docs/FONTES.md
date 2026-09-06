# Fontes

Este documento registra as fontes que aparecem na cadeia publica. Ele descreve uso
tecnico e limite de interpretacao; nao transforma fonte observacional em validacao
universal.

## Fontes consolidadas na tabela unica

| Fonte | Papel na cadeia | Observacoes |
|---|---|---|
| Recife | Evidencia pluvial local | Fonte municipal de eventos; usada no modelo `recife_pluvial`. |
| Curitiba | Referencia local para transferencia fluvial | Eventos SIAC 156 entram como dado local de referencia, nao como modelo local validado. |
| NYC | Evidencia pluvial urbana em larga escala | Usada no modelo `pluvial_urbano_nyc` e na auditoria de chuva antecedente. |
| Sao Paulo | Evidencia pluvial urbana testada | O candidato regional fica abaixo do acaso e nao e servido. A chuva antecedente tem suporte como forcamento no recorte testado. |
| UK | Base fluvial com grupos de validacao | Entra no pool fluvial. |
| CEMS | Evidencia fluvial de evento | Entra no pool fluvial com rastreio por evento. |
| Sen1Floods11 | Evidencia global de inundacao | Entra no pool fluvial. |
| Urban Flood Observations | Observacoes urbanas heterogeneas | Entra na tabela admitida, fora do pool fluvial principal. |

## Fonte de chuva

A auditoria de chuva usa serie meteorologica harmonizada para construir o
`rain_decay_index` e para a analise caso-controle temporal. O resultado da chuva e
interpretado como suporte ao forcamento de evento, nao como prova de que o modelo de
suscetibilidade generaliza para todos os recortes.

## Fonte social

A camada social territorial v0 usa recortes censitarios de Curitiba para organizar
exposicao, perfil socioeconomico, demografia, raca/genero, moradia/infraestrutura,
mobilidade/acesso e lacunas institucionais.

Essa camada e paralela ao escore fisico. Ela nao muda o contrato de inferencia, nao
altera pesos e nao cria rotulos de inundacao.

## Regra de leitura

Todo dado publico deve ser lido com tres perguntas:

1. Qual fonte sustenta esta afirmacao?
2. Qual artefato versionado registra o processamento?
3. O resultado e `supported`, `inconclusive`, `not_testable` ou uma recusa do contrato?

Quando uma dessas respostas nao estiver disponivel, o texto publico deve declarar a
lacuna.
