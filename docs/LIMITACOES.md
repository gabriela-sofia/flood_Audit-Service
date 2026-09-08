# Limitacoes

Este arquivo reune, num lugar so, as limitacoes que os artefatos declaram. Ele nao
suaviza nenhuma delas e nao acrescenta nenhuma que o artefato nao registre. Cada bloco
cita o arquivo onde a limitacao esta gravada.

A regra que organiza este documento: **evidencia e limitacao andam juntas**. Uma tabela
de resultado sem a limitacao ao lado nao e um resultado resumido, e um resultado
diferente.

## 1. Do desenho, e valem para tudo

Fonte: `modelo/execucoes/svc-03-grade/resultado.json`, campo `nao_e`.

- Nenhuma das tres regioes servidas tem inventario local usado como criterio de aprovacao. A grade nao e afirmacao de acerto.
- Ausencia de registro nao e ausencia de inundacao. Chamado que nao existiu, mascara que nao foi gerada e ativacao que nao aconteceu sao lacunas, nao negativos.
- Suscetibilidade fisica e vulnerabilidade social sao coisas diferentes e nao se somam.
- Escore de regioes diferentes vem de modelos diferentes, sem calibracao comum, e nao e comparavel entre regioes.

## 2. Por regiao servida

### Recife — `mvp_local`

Fonte: `modelo/execucoes/svc-03-grade/resultado.json`, regiao `recife`, campo `limitacoes`.

| Limitacao | Consequencia |
|---|---|
| 98,4% dos negativos sao fundo (ausencia de registro), o nivel mais fraco da hierarquia | O escore so sustenta ORDEM, nao probabilidade absoluta. |
| Apenas 3,7% dos pontos estao em datas que contem as duas classes, e o modelo usa `rain_decay_index` | Nesta base, saber a data e quase saber a classe: a chuva pode estar separando dias em vez de lugares. |
| `hand_m` nao separa as classes | Num evento pluvial o HAND nao e o mecanismo. |
| `elevation_rel_m` e `slope_deg` saem com sinal contrario ao fisicamente esperado | Entram no escore e ficam fora da frase de explicacao. |
| O IC95 de `hand_m` cruza zero | O ajuste nao estabeleceu o efeito. |
| `rain_max_24h` saiu do vetor | Media a janela ANTERIOR ao evento e e parcela da soma que `rain_decay_index` ja acumula. |

### Curitiba — `transferencia_caracterizada`

Fonte: `modelo/execucoes/svc-03-grade/resultado.json`, regiao `curitiba`.

| Limitacao | Consequencia |
|---|---|
| Nao existe modelo local servido para Curitiba | Ela e servida pelo `fluvial_planicie`, ajustado fora dela. |
| `elevation_rel_m` e a elevacao contra o P1 da propria regiao; a absoluta punha 0,0% dos pontos no dominio de treino, contra 94,5% da relativa | A variavel resolve DOMINIO, e o efeito dela entre cidades e instavel (secao 4). |
| A chuva ficou fora do conjunto | Ela nao discrimina na escala em que este modelo compara. |
| `slope_deg` sai com sinal contrario ao esperado | Fica fora da frase de explicacao. |
| O IC95 de `elevation_rel_m` cruza zero | O ajuste nao estabeleceu o efeito. |

### Petropolis — `transferencia_sem_referencia_local`

Fonte: `modelo/execucoes/svc-03-grade/resultado.json`, regiao `petropolis`.

| Limitacao | Consequencia |
|---|---|
| Petropolis nao tem NENHUMA linha na tabela unica | Nenhum ponto rotulado, nenhum inventario local, nenhuma validacao local possivel hoje. |
| Nenhuma AOI brasileira entrou no ajuste do `fluvial_serra` | Os grupos ingremes sao todos estrangeiros. |
| O estrato ingreme tem 24 grupos, 19 deles positivos, e por isso o modelo usa uma unica variavel | AUC de 0,7916 sobre 24 grupos e um resultado fragil por construcao, e o artefato registra as duas leituras de EPV que sustentam a escolha. |
| O escore e predicao por semelhanca de terreno | Nunca afirmacao de acerto. |

## 3. Sao Paulo e NYC

| Recorte | Limitacao |
|---|---|
| Sao Paulo, modelo fisico | AUC 0,4753, abaixo do acaso. O candidato NAO e servido, e isso esta no indice de modelos, nao so no texto. |
| Sao Paulo, chuva | O forcamento tem suporte (RR 1,2158). Suporte de forcamento nao promove modelo de suscetibilidade. |
| NYC | Serve como referencia externa. O modelo tem AUC 0,6089, fora dos criterios fortes. |

## 4. Transferencia entre cidades

Fonte: `modelo/execucoes/chicago-teste-da-lei/teste_da_lei_chicago.json`.

Esta e a limitacao mais recente e a que mais restringe o que o pacote pode afirmar.

| Termo | Nova York | Chicago | Leitura |
|---|---:|---:|---|
| `mrms_qpe_24h_mm` | +0,1886 | +0,4416 | Replicou. |
| `slope_deg` | −0,0151 | −0,0111 | Replicou. |
| `hand_m` | −0,0058 | −0,0179 | Nulo em NYC; em Chicago nao ha contraste para testar. |
| `twi_dinf` | +0,0390 | −0,0102 | **Inverteu**, com ICs que nao se sobrepoem. |
| `twi_dinf` x chuva | +0,0397 | −0,0110 | **Inverteu**, com ICs que nao se sobrepoem. |
| `elevation_rel_m` | −0,0088 | +0,0397 | **Inverteu**, com ICs que nao se sobrepoem. |

Consequencias que o pacote assume:

1. O TWI nao esta demonstrado como mecanismo geral de acumulo urbano. Ele funcionou em Nova York.
2. `elevation_rel_m` resolve o problema de DOMINIO entre regioes; ela nao esta demonstrada como tendo efeito estavel entre regioes. Essas duas propriedades sao diferentes e o metodo agora as separa.
3. As maturidades `transferencia_caracterizada` (Curitiba) e `transferencia_sem_referencia_local` (Petropolis) continuam validas como ROTULOS de quanto se sabe — e este resultado e mais uma razao para elas nao serem promovidas.
4. Chicago nao autoriza a conclusao inversa: sao duas cidades dos Estados Unidos, com centrais de atendimento e clima proprios, e nenhuma delas e brasileira.

Chicago nao pode responder sobre o HAND: o p95 do `hand_m` na cidade e 7,985 m contra
limiar declarado de 10 m. Um nulo ali seria nulo por falta de contraste.

## 5. Chuva como forcamento

Fonte: `modelo/execucoes/chuva-autopareamento/matriz_suporte_forcamento.json`.

- Zero chamado nao significa zero inundacao.
- A exposicao e aproximada em celula de cerca de 11 km no desenho autopareado.
- O esforco de notificacao so esta controlado se for constante dentro do estrato.
- Chuva diaria nao mede intensidade intradiaria.
- O resultado nao altera automaticamente escore fisico nenhum.

## 6. Temporalidade

Fonte: `modelo/execucoes/temporalidade-integracao/scientific_support_matrix.json`.

Nenhum escopo avaliado chegou a `supported`. Todos estao em `inconclusive` ou
`not_testable`. Nao ha validacao prospectiva ampla neste pacote, e a frente temporal
nao deve ser apresentada como se houvesse.

## 7. Dados

Fonte: `modelo/execucoes/ds-05-tabela-unica/manifesto_v2.json`.

| Limitacao | Onde |
|---|---|
| A coluna `unidade_agrupamento` recebe o valor `chamado` em 478.873 linhas, fora do dominio declarado | `violacoes_de_contrato` no manifesto. Pendente de revisao de contrato. |
| Ha empilhamento de pontos na mesma celula dentro de varias fontes (402.714 em NYC, 1.124 em Sao Paulo, 812 em Curitiba, 71 em Recife) | `duplicidade.intra_fonte_mesma_celula`. E por isso que o grupo de reamostragem nunca pode ser mais fino que a celula. |
| As tabelas completas nao estao versionadas | Representadas por nome, tamanho e checksum no manifesto. |

## 8. Frentes auxiliares

| Frente | Limite |
|---|---|
| Camada social | v0 de revisao. Nao altera escore fisico, nao treina modelo, nao cria rotulo hidrologico, e ausencia de dado social nao e baixa vulnerabilidade. |
| Recuperacao de fonte | Serve rastreabilidade. Nao altera portao, peso ou escore. |
| Cadeia seletiva / neural | Reconstrucao mascarada em recortes reservados, nao deteccao direta de agua. O veredito parametrico do proprio experimento e `INDECIDIVEL`, ao lado do veredito por postos. |
| DINO e revisao visual | Rotinas historicas mantidas no repositorio; nao participam do escore servido. |

## 9. O que este pacote NAO afirma

- Nao afirma previsao de inundacao, alerta em tempo real ou diagnostico de imovel.
- Nao afirma validacao operacional de nenhum modelo.
- Nao afirma que ausencia de registro seja evidencia negativa.
- Nao afirma transferencia demonstrada entre regioes.
- Nao afirma transporte HTTP, interface publica ou operacao continua.
