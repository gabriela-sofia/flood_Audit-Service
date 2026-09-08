# Cartao do modelo

## Uso pretendido

O modelo estima suscetibilidade fisico-hidrologica a inundacoes em regioes declaradas
no contrato. A saida e uma resposta auditavel por geometria ou celula de grade, com
escore, intervalo, maturidade regional, distancia de dominio e explicacao estruturada.

O modelo nao e alerta em tempo real, previsao meteorologica, diagnostico individual,
medida de vulnerabilidade social ou prova de risco para uma pessoa ou imovel.

## Modelos candidatos

Fonte: `modelo/execucoes/svc-01-modelos/indice.json`, gerado em `2026-09-01T15:58:25`,
semente `20260820`, `n_boot=1000`.

| Modelo | Servido | Regiao | n | grupos | AUC | Veredito | Mecanismo |
|---|---|---|---:|---:|---:|---|---|
| `recife_pluvial` | Sim | Recife | 269 | 227 | 0,6422 | `FORA_DOS_CRITERIOS` | pluvial urbano |
| `pluvial_urbano_nyc` | Sim | NYC | 471.727 | 751 | 0,6089 | `FORA_DOS_CRITERIOS` | pluvial urbano |
| `pluvial_urbano_sao_paulo` | Nao | Sao Paulo | 7.146 | 1.438 | 0,4753 | `ABAIXO_DO_ACASO` | pluvial urbano |
| `fluvial_planicie` | Sim | Curitiba | 56.654 | 509 | 0,7334 | `COERENTE_COM_CRITERIOS` | fluvial/enxurrada |
| `fluvial_serra` | Sim | Petropolis | 5.162 | 24 | 0,7916 | `COERENTE_COM_CRITERIOS` | fluvial/enxurrada |

"Servido" significa que ha artefato usado pelo contrato — nao que o modelo esteja livre
de limitacao. O `recife_pluvial` e servido E tem veredito fora dos criterios no proprio
indice; ele precisa ser explicado como MVP local com limitacoes, nunca como validacao.

A coluna `grupos` e parte do resultado, nao um detalhe: a AUC e medida por validacao
cruzada agrupada, e 24 grupos (19 positivos) e o orcamento que obriga o `fluvial_serra`
a usar uma unica variavel. Ler a AUC sem ler os grupos e ler metade.

## Pesos por modelo

Coeficientes padronizados (variavel centrada e escalada pela media e desvio do proprio
ajuste). Leia direcao e forca relativa DENTRO de cada modelo; coeficiente de modelos
diferentes nao e comparavel. Intervalos por bootstrap do preditor linear reamostrando
grupos, 1.000 replicas.

### `recife_pluvial`

Intercepto `0,178908`. Prevalencia do ajuste `0,539`. Fonte: `recife`, 269 pontos.

| Variavel | Coeficiente | IC 95% | Sinal esperado | Leitura |
|---|---:|---|---:|---|
| `rain_decay_index` | +0,4999 | 0,1958 a 0,9571 | + | Chuva antecedente sustenta sinal positivo. Fisicamente verificada. |
| `twi_dinf` | +0,4372 | 0,1076 a 0,9608 | + | Coerente com convergencia e acumulo. Fisicamente verificada. |
| `slope_deg` | +0,4028 | 0,0416 a 0,8948 | − | Sinal contrario ao esperado; entra no escore e fica fora da explicacao. |
| `elevation_rel_m` | +0,3271 | −0,3778 a 1,1346 | − | Sinal contrario e IC cruzando zero. |
| `hand_m` | −0,0872 | −0,7439 a 0,5230 | − | IC cruza zero: o ajuste nao estabeleceu o efeito. |

Amostra negativa: 145 positivos e 124 negativos, dos quais **122 sao fundo (98,4%)** e
apenas 2 sao exclusao qualificada. Leitura admitida: **ordinal** — so a ordem entre
celulas e sustentada, porque com amostra majoritariamente de fundo a prevalencia nao e
identificavel e o nivel absoluto nao e propriedade do terreno.

Aderencia temporal: 205 datas distintas, **5 com as duas classes** (2,44%), e apenas
3,72% dos pontos em data compartilhada. O modelo usa `rain_decay_index`, que depende do
tempo: nesta base, saber a data e quase saber a classe.

Falhas de criterio registradas: IC95 de `hand_m` cruza zero; AUC 0,6422 fora da faixa
[0,70; 0,88].

### `pluvial_urbano_nyc`

Intercepto `−1,296237`. Prevalencia do ajuste `0,2219`. Fonte: `nyc`, 471.727 pontos.

| Variavel | Coeficiente | IC 95% | Sinal esperado | Leitura |
|---|---:|---|---:|---|
| `twi_dinf` | +0,2119 | 0,1672 a 0,2597 | + | Sinal positivo estavel. Fisicamente verificada. |
| `slope_deg` | −0,1807 | −0,2515 a −0,1092 | − | Sinal coerente. Fisicamente verificada. |
| `hand_m` | −0,0641 | −0,2278 a 0,1167 | − | IC cruza zero. |
| `elevation_rel_m` | −0,0464 | −0,2575 a 0,1446 | − | IC cruza zero. |

Amostra negativa: 104.653 positivos e 367.074 negativos, **100% fundo**. Leitura
admitida: ordinal. Unidade de reamostragem: bloco espacial de 1 km — a fusao por celula
colapsou a unidade neste conjunto, porque o mesmo local reaparece em muitos dias.

Limitacao central do mecanismo: a agua vem da chuva excedendo a drenagem, e a
**capacidade da rede nao esta no vetor**. HAND e TWI descrevem para onde a agua corre
naturalmente; na cidade ela corre para onde a engenharia mandou.

O gatilho `rain_event_day` esta adquirido e medido — ele leva Nova York de 0,6090 para
0,7395 — e nao entra no conjunto servido por duas razoes declaradas: Recife nao o tem, e
a chuva do dia do evento e posterior a decisao de rotular o ponto.

### `pluvial_urbano_sao_paulo` — nao servido

| Item | Valor |
|---|---:|
| n | 7.146 |
| grupos | 1.438 |
| AUC | 0,4753 |
| Servivel | Nao |
| Veredito | `ABAIXO_DO_ACASO` |

Motivo registrado no artefato: a AUC agrupada nao supera o acaso, entao a ordem
produzida e pior que aleatoria e a leitura ordinal — a unica que o projeto reivindica —
nao se sustenta. `elevation_rel_m` (+0,7226) e `slope_deg` (+0,1704) saem com sinal
contrario ao esperado, ambos com IC excluindo zero.

Este resultado e importante justamente por ser negativo: mostra o pipeline reprovando
quando a evidencia nao sustenta o uso.

### `fluvial_planicie`

Intercepto `−0,630888`. Prevalencia do ajuste `0,4428`. Fontes do ajuste:
`sen1floods11` 29.650, `cems` 19.528, `uk` 7.476.

| Variavel | Coeficiente | IC 95% | Sinal esperado | Leitura |
|---|---:|---|---:|---|
| `hand_m` | −2,5967 | −3,5426 a −1,7926 | − | Sinal forte e coerente: menor HAND, maior suscetibilidade. Fisicamente verificada. |
| `twi_dinf` | +0,4124 | 0,3435 a 0,4645 | + | Coerente com convergencia. Fisicamente verificada. |
| `slope_deg` | +0,3097 | 0,1146 a 0,4492 | − | Sinal contrario ao esperado; fica fora da explicacao. |
| `elevation_rel_m` | −0,0535 | −0,3250 a 0,0328 | − | IC cruza zero. |

Amostra negativa: 25.085 positivos e 31.569 negativos, **0% fundo** — 27.831 sao
ausencia observada e 3.738 exclusao qualificada. Leitura admitida: **absoluta**, porque
os negativos sao observados ou qualificados. E o unico dos modelos pluviais/fluviais
servidos com essa propriedade, junto do `fluvial_serra`.

Nenhuma falha de criterio registrada.

### `fluvial_serra`

Intercepto `−0,626171`. Prevalencia do ajuste `0,4117`. Fontes do ajuste: `cems` 4.387,
`sen1floods11` 775.

| Variavel | Coeficiente | IC 95% | Sinal esperado | Leitura |
|---|---:|---|---:|---|
| `hand_m` | −1,4423 | −3,5545 a −0,7455 | − | Menor altura acima da drenagem, maior suscetibilidade. Fisicamente verificada. |

Amostra negativa: 2.125 positivos e 3.037 negativos, **0% fundo**, todos ausencia
observada. Leitura admitida: absoluta. 24 grupos, 19 positivos: e esse orcamento que
limita o modelo a uma variavel.

Nenhuma falha de criterio registrada — mas nenhuma AOI brasileira entrou no ajuste, e
Petropolis nao tem uma linha na tabela unica.

## Estabilidade das variaveis entre cidades

Fonte: `modelo/execucoes/chicago-teste-da-lei/`.

| Termo | Nova York | Chicago | Replicou |
|---|---:|---:|---|
| `mrms_qpe_24h_mm` | +0,1886 | +0,4416 | sim |
| `slope_deg` | −0,0151 | −0,0111 | sim |
| `twi_dinf` | +0,0390 | −0,0102 | **nao, inverteu** |
| `elevation_rel_m` | −0,0088 | +0,0397 | **nao, inverteu** |

Isso nao altera nenhum modelo servido — auditoria de transferencia nao promove nem
rebaixa modelo — mas restringe o que pode ser dito sobre generalizacao. Ver
`docs/EVIDENCIA.md`, secao 7.

## Dominio de covariaveis

Faixa 5–95 usada pelo portao `G5_DOMINIO_COVARIAVEL`, por modelo:

| Modelo | `elevation_rel_m` | `slope_deg` | `hand_m` | `twi_dinf` | `rain_decay_index` |
|---|---|---|---|---|---|
| `recife_pluvial` | 0,0045 a 65,7758 | 0,1863 a 17,5814 | 0,0 a 46,3916 | 4,953 a 12,754 | 5,9962 a 68,8708 |
| `pluvial_urbano_nyc` | 1,4565 a 40,7301 | 0,0635 a 4,0611 | 0,0009 a 31,1821 | 6,8098 a 14,9181 | — |
| `fluvial_planicie` | 3,179 a 315,8319 | 0,0001 a 11,053 | 0,0 a 27,4659 | 5,8065 a 15,6099 | — |
| `fluvial_serra` | — | — | 0,0 a 187,4298 | — | — |

Caso fora da faixa nao e recusado automaticamente: ele sai marcado como extrapolacao,
com a distancia ao dominio declarada na resposta.

## Maturidade regional

| Regiao | Modelo | Maturidade | O que a maturidade quer dizer |
|---|---|---|---|
| Recife | `recife_pluvial` | `mvp_local` | Ha dado local rotulado, com as limitacoes acima. |
| Curitiba | `fluvial_planicie` | `transferencia_caracterizada` | Ha dado local de referencia, mas o modelo servido nao foi ajustado nela. |
| Petropolis | `fluvial_serra` | `transferencia_sem_referencia_local` | Nao ha nenhuma linha local na tabela unica. |
| NYC | `pluvial_urbano_nyc` | `referencia_externa` | Serve de referencia urbana externa. |

Sao Paulo aparece como evidencia analisada, mas o candidato regional nao e servido. A
escala completa de maturidade esta em [`docs/CONCEITOS.md`](../docs/CONCEITOS.md).

## Grade publica

Fonte: `modelo/execucoes/svc-03-grade/resultado.json`.

| Regiao | Cenario | p50 | p90 | Cobertura servivel |
|---|---|---:|---:|---:|
| Recife | mediana observada de chuva | 0,5808 | 0,7785 | 97,0% |
| Recife | p90 observado de chuva | 0,7323 | 0,8740 | 97,0% |
| Recife | maximo observado de chuva | 0,9015 | 0,9599 | 79,9% |
| Curitiba | unico, sem chuva | 0,2746 | 0,5804 | 98,5% |
| Petropolis | unico, sem chuva | 0,3460 | 0,5634 | 91,3% |

As linhas de regioes diferentes nao sao comparaveis entre si: sao modelos diferentes,
com vetores diferentes e sem calibracao comum.

## Condicoes de recusa

O contrato deve recusar ou marcar extrapolacao quando:

1. A regiao nao esta no contrato.
2. O modelo regional nao e servivel.
3. A maturidade minima nao esta declarada.
4. Variaveis obrigatorias estao ausentes.
5. O caso sai do dominio de covariaveis sem regra explicita de uso.

## Limitacoes

O inventario completo esta em [`docs/LIMITACOES.md`](../docs/LIMITACOES.md). Os pontos
que mais restringem a leitura deste cartao:

- nenhuma das tres regioes servidas tem inventario local usado como criterio de aprovacao;
- Petropolis nao tem nenhuma linha na tabela unica, e o escore dela e semelhanca de terreno;
- em Recife, 98,4% dos negativos sao ausencia de registro, e a data quase determina a classe;
- o unico teste de transferencia entre cidades reprovou o sinal de `twi_dinf` e de `elevation_rel_m`;
- as auditorias temporais continuam em `inconclusive` e `not_testable`;
- a camada social e paralela ao escore fisico e permanece em revisao.

Nenhum resultado deste pacote deve ser lido como validacao prospectiva geral para o
Brasil.
