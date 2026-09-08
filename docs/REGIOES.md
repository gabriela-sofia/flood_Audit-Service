# Regiao por regiao

Cada regiao entra no projeto com um papel diferente. Confundir os papeis e o erro que
transforma "aplicamos um modelo transferido" em "validamos localmente".

## Resumo

| Regiao | Papel | Modelo servido | Maturidade | Principal limite |
|---|---|---|---|---|
| Recife | MVP local pluvial | `recife_pluvial` | `mvp_local` | Negativos por ausencia e aderencia temporal fragil |
| Curitiba | Aplicacao central em planicie | `fluvial_planicie` | `transferencia_caracterizada` | Transferencia, nao modelo local servido |
| Petropolis | Serra por transferencia | `fluvial_serra` | `transferencia_sem_referencia_local` | Sem nenhuma linha local na tabela unica |
| NYC | Referencia urbana externa | `pluvial_urbano_nyc` | `referencia_externa` | Chamado e notificacao, nao inundacao |
| Sao Paulo | Teste urbano brasileiro | nenhum | reprovado no fisico local | AUC abaixo do acaso |
| Chicago | Segunda cidade do teste de transferencia | nenhum | fora do produto | Nao e regiao servida; existe para testar um efeito |

## Recife

| Aspecto | Estado |
|---|---|
| Papel | MVP local pluvial |
| Modelo servido | `recife_pluvial` |
| Variaveis | `elevation_rel_m`, `slope_deg`, `hand_m`, `twi_dinf`, `rain_decay_index` |
| Maturidade na grade | `mvp_local` |
| AUC curada | 0,6422 (269 pontos, 227 grupos) |
| Grade | 57.994 celulas, 56.666 validas, 54.963 serviveis, 1.703 recusadas |
| Escore (cenario de chuva mediano) | p50 0,5808 / p90 0,7785 |
| Principal forca | Unica regiao com base local rotulada e chuva antecedente no vetor |
| Principal limitacao | 98,4% dos negativos sao ausencia de registro; so 3,7% dos pontos em datas com as duas classes |

**Como explicar:** Recife mostra a cadeia local completa — dado, modelo, contrato, grade
e explicacao. Ao mesmo tempo e o exemplo de honestidade do projeto: a AUC e moderada, o
proprio indice marca `FORA_DOS_CRITERIOS`, e o escore sustenta ordem, nao valor.

## Curitiba

| Aspecto | Estado |
|---|---|
| Papel | Regiao central do trabalho, aplicacao em planicie |
| Modelo servido | `fluvial_planicie`, ajustado com Sen1Floods11, CEMS e UK |
| Maturidade na grade | `transferencia_caracterizada` |
| AUC do modelo usado | 0,7334 (56.654 pontos, 509 grupos) |
| Grade | 67.521 celulas, 65.275 validas, 64.310 serviveis, 965 recusadas |
| Escore | p50 0,2746 / p90 0,5804 |
| Camada social | IBGE Censo 2022, 3.190 setores, 254 em FCU |
| Temporalidade | `not_testable` no escopo declarado; diagnostico local `inconclusive` |
| Presenca na tabela unica | 1.680 pontos admitidos, 1.471 grupos |
| Principal forca | Contrato, grade e camada social territorial completos |
| Principal limitacao | O modelo servido nao foi ajustado nela |

**Como explicar:** Curitiba e o caso mais importante para entender maturidade. Nao se
pode dizer "temos modelo local validado de Curitiba" quando o servido e o
`fluvial_planicie`. A formulacao correta e aplicacao por transferencia caracterizada,
com camada social local e auditorias explicitas.

A elevacao relativa existe por causa dela: a elevacao absoluta punha 0,0% dos pontos de
Curitiba no dominio de treino, contra 94,5% da relativa.

## Petropolis

| Aspecto | Estado |
|---|---|
| Papel | Aplicacao serrana por transferencia |
| Modelo servido | `fluvial_serra`, ajustado com CEMS e Sen1Floods11 |
| Variavel central | `hand_m`, coeficiente −1,4423 (IC −3,5545 a −0,7455) |
| Maturidade na grade | `transferencia_sem_referencia_local` |
| AUC do modelo usado | 0,7916 (5.162 pontos, **24 grupos**, 19 positivos) |
| Grade | 176.818 celulas, 172.015 validas, 157.005 serviveis, 15.010 recusadas |
| Escore | p50 0,3460 / p90 0,5634 |
| Presenca na tabela unica | **nenhuma linha** |
| Principal forca | Coerencia fisica do HAND em relevo ingreme |
| Principal limitacao | Nenhuma referencia local rotulada, e nenhuma AOI brasileira no ajuste |

**Como explicar:** Petropolis e leitura fisica transferida, nunca acuracia local
provada. Os 24 grupos do estrato ingreme sao o orcamento que obriga o modelo a uma unica
variavel — e a razao de o resultado, ainda que dentro dos criterios, ser fragil por
construcao.

## NYC

| Aspecto | Estado |
|---|---|
| Papel | Serie urbana externa para pluvial e para as frentes de chuva |
| Modelo | `pluvial_urbano_nyc`, AUC 0,6089 (471.727 pontos, 751 grupos) |
| Chuva como forcamento | Razao de risco por 10 mm = 1,5099 (IC 1,4645 a 1,5610) |
| Terreno sob gatilho | `twi_dinf` +0,0390 e `twi_dinf` x chuva +0,0397, ambos aprendidos |
| Presenca na tabela unica | 471.727 pontos admitidos, 2.402 grupos |
| Principal forca | Base grande e associacao forte entre chuva e chamado |
| Principal limitacao | Chamado e processo de notificacao; 100% dos negativos sao fundo |

O gatilho `rain_event_day` levaria NYC de 0,6090 para 0,7395 e **nao entra** no conjunto
servido: Recife nao o tem, e a chuva do dia do evento e posterior a decisao de rotular o
ponto.

## Sao Paulo

| Aspecto | Estado |
|---|---|
| Papel | Teste urbano brasileiro |
| Modelo fisico local | `pluvial_urbano_sao_paulo`, AUC 0,4753, **nao servido** |
| Chuva como forcamento | Razao de risco por 10 mm = 1,2158 (IC 1,0470 a 1,4998) |
| Presenca na tabela unica | 7.146 pontos admitidos, 1.976 grupos |
| Principal leitura | O pipeline reprova o modelo fisico local e mantem o suporte da chuva |

`elevation_rel_m` (+0,7226) e `slope_deg` (+0,1704) saem com sinal contrario ao esperado,
com IC excluindo zero. Esse e o padrao de um ajuste que aprendeu outra coisa que nao o
terreno.

## Chicago

| Aspecto | Estado |
|---|---|
| Papel | Segunda cidade do teste de transferencia; **nao e regiao servida** |
| Amostra | 496.972 pontos, 134.542 positivos, 2.438 dias informativos |
| Replicou | chuva (+0,4416) e declividade (−0,0111) |
| Inverteu | `twi_dinf`, `twi_dinf` x chuva e `elevation_rel_m`, com ICs que nao se sobrepoem |
| Nao pode responder | HAND: p95 de 7,985 m contra limiar de 10 m, sem contraste |
| Principal leitura | Um efeito medido numa cidade so nao distingue lei fisica de particularidade daquela cidade |

Chicago nao entra na tabela unica, nao promove variavel e nao ajusta modelo servido. Ela
existe para responder uma pergunta que nenhuma quantidade de dado de Nova York
responderia. Detalhes em `docs/EVIDENCIA.md`, secao 7.
