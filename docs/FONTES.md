# Fontes

Este documento registra as fontes que aparecem na cadeia publica. Ele descreve uso
tecnico e limite de interpretacao; nao transforma fonte observacional em validacao
universal.

## Fontes consolidadas na tabela unica

| Fonte | Papel na cadeia | Observacoes |
|---|---|---|
| Recife | Evidencia pluvial local | Fonte municipal de eventos; usada no modelo `recife_pluvial`. Os negativos sao ausencia de registro. |
| Curitiba | Referencia local para transferencia fluvial | Eventos SIAC 156 entram como dado local de referencia, nao como modelo local validado. |
| NYC | Evidencia pluvial urbana em larga escala | Usada no modelo `pluvial_urbano_nyc` e nas auditorias de chuva. |
| Sao Paulo | Evidencia pluvial urbana testada | O candidato regional fica abaixo do acaso e nao e servido. A chuva antecedente tem suporte como forcamento no recorte testado. |
| UK | Base fluvial com grupos de validacao | Entra no pool fluvial. |
| CEMS | Evidencia fluvial de evento | Entra no pool fluvial com rastreio por evento. |
| Sen1Floods11 | Evidencia global de inundacao | Entra no pool fluvial. |
| Urban Flood Observations | Observacoes urbanas heterogeneas | Entra na tabela admitida, fora do pool fluvial principal. |

Petropolis nao aparece nesta tabela porque nao tem nenhuma linha na base. Essa ausencia
e a razao da maturidade `transferencia_sem_referencia_local`.

## Fontes de chuva

| Produto | Resolucao | Onde e usado | Limite |
|---|---|---|---|
| Open-Meteo / ERA5-Land | celula de cerca de 11 km | `rain_decay_index` do modelo de Recife e desenho autopareado em NYC e Sao Paulo | Nao mede intensidade intradiaria; na escala da grade de 120 m a chuva nao varia, por isso entra como cenario. |
| MRMS QPE 24 h | 1 km | Terreno sob gatilho em Nova York e o teste de Chicago | Estimativa de radar calibrada por pluviometro, nao medida de pluviometro. Cobre apenas os Estados Unidos. |

A escolha do produto muda a pergunta que da para fazer. Com a chuva a 11 km ela nao
varia dentro do dia numa cidade, e o estrato precisa ser celula x dia; com a chuva a
1 km ela varia entre bairros no mesmo dia, e o estrato pode ser o dia inteiro. Os dois
desenhos estao no pacote, e cada artefato declara qual usou.

## Fontes fora do modelo, usadas em auditoria

| Fonte | Papel | Por que nao entra em modelo |
|---|---|---|
| Chicago 311 | Segunda cidade do teste de transferencia | O objetivo e testar se um efeito medido em Nova York reaparece; promover a fonte a treino confundiria as duas perguntas. |
| MDT de Chicago | Terreno da segunda cidade, pela mesma cadeia D-infinity de 30 m | Idem. |

O fundo de Chicago e `target-group background`: chamado da MESMA base, na mesma janela,
sobre outro assunto. Ele existe para cancelar a propensao daquele endereco a acionar a
central, e o manifesto registra o teto por dia, a posicao sorteada dentro do dia, a
semente e o vies que permanece.

## Fonte social

A camada social territorial v0 usa recortes censitarios do IBGE (Censo 2022) em
Curitiba para organizar exposicao, perfil socioeconomico, demografia, raca/genero,
moradia/infraestrutura, mobilidade/acesso e lacunas institucionais. O manifesto registra
URL, tamanho e SHA-256 de cada arquivo de origem.

Essa camada e paralela ao escore fisico. Ela nao muda o contrato de inferencia, nao
altera pesos e nao cria rotulos de inundacao. Ausencia de dado social nao e baixa
vulnerabilidade.

## Regra de leitura

Todo dado publico deve ser lido com tres perguntas:

1. Qual fonte sustenta esta afirmacao?
2. Qual artefato versionado registra o processamento?
3. O resultado e `supported`, `inconclusive`, `not_testable`, reprovado, ou uma recusa do contrato?

Quando uma dessas respostas nao estiver disponivel, o texto publico deve declarar a
lacuna.
