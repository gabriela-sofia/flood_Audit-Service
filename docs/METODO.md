# Metodo

Este documento descreve a cadeia publica do projeto. Ele nao substitui os scripts:
cada numero citado precisa estar em artefato versionado em `modelo/execucoes/`.

## 1. Premissa

O projeto mede suscetibilidade fisico-hidrologica. A variavel de interesse nao e dano,
vulnerabilidade social, exposicao populacional, risco individual ou previsao operacional.
A resposta esperada e uma estimativa auditavel de predisposicao fisica do terreno sob
condicoes representadas pelas covariaveis disponiveis.

O contrato so responde quando a combinacao entre regiao, modelo, variaveis e distancia
de dominio e compativel com o que foi treinado ou explicitamente aceito como
transferencia. Fora disso, a saida correta e recusa ou alerta de extrapolacao.

## 2. Cadeia publica

1. Curadoria de fontes observacionais e negativas.
2. Harmonizacao espacial e derivacao de variaveis fisicas de terreno.
3. Consolidacao da tabela unica e dos pools de ajuste.
4. Treino dos candidatos regionais e fisiograficos.
5. Selecao dos modelos servidos.
6. Execucao do contrato de inferencia.
7. Geracao de grades de aplicacao.
8. Auditorias complementares de chuva, temporalidade, recuperacao de fonte e camada social.
9. Teste de transferencia entre cidades, que mede se um efeito medido num lugar reaparece em outro.

## 3. Variaveis principais

| Variavel | Papel | Estado da evidencia |
|---|---|---|
| `elevation_rel_m` | Elevacao relativa ao P1 da propria regiao. Foi adotada no lugar da absoluta porque a absoluta punha 0,0% dos pontos de Curitiba no dominio de treino, contra 94,5% da relativa. | O sinal dela **inverte** entre Nova York e Chicago (`docs/EVIDENCIA.md`, secao 7). Ela resolve o problema de DOMINIO; ela nao esta demonstrada como invariante entre cidades. |
| `slope_deg` | Declividade local. | Replicou nas duas cidades testadas, com o sinal fisico esperado. |
| `hand_m` | Altura acima da drenagem mais proxima. | Nulo em Nova York; em Chicago nao ha contraste suficiente para testar (p95 de 7,985 m contra limiar de 10 m). Sustenta sozinho o modelo `fluvial_serra`, num estrato ingreme de 24 grupos. |
| `twi_dinf` | Indice topografico de umidade com escoamento D-infinity. | Mecanismo dominante em Nova York; o sinal **inverte** em Chicago. |
| `rain_decay_index` | Chuva antecedente agregada por decaimento temporal, quando disponivel no recorte. | Entra so no modelo de Recife. A chuva como forcamento de evento tem suporte em NYC e Sao Paulo. |

A leitura correta desta tabela: variavel util para ORDENAR dentro de uma regiao nao e a
mesma coisa que variavel com efeito estavel ENTRE regioes. As duas ultimas colunas do
pacote separam essas duas perguntas de proposito.

## 4. Treino curado

Os scripts publicos de treino ficam em `scripts/treino/`. O diretorio inclui a
consolidacao de dados, verificacoes de chuva, desenho experimental e selecao de
modelos. O nome do diretorio e deliberado: ele contem somente o material que entra na
cadeia curada deste pacote publico.

O manifesto da tabela unica v2 registra:

| Campo | Valor |
|---|---:|
| Observacoes de entrada | 595.865 |
| Observacoes admitidas | 568.265 |
| Pool fluvial | 65.070 |
| Grupos admitidos | 6.873 |
| Grupos no pool fluvial | 2.002 |

As tabelas completas ficam fora do pacote publico por tamanho. O manifesto registra os
nomes, tamanhos e checksums esperados para reproducao.

## 5. Modelos servidos

A etapa `svc-01-modelos` produz o indice de candidatos. O contrato atual usa apenas
modelos marcados como serviveis.

| Modelo | Regiao servida | Variaveis | n / grupos | Estado |
|---|---|---|---:|---|
| `recife_pluvial` | Recife | `elevation_rel_m`, `slope_deg`, `hand_m`, `twi_dinf`, `rain_decay_index` | 269 / 227 | Servivel, desempenho fora dos criterios fortes |
| `pluvial_urbano_nyc` | NYC | `elevation_rel_m`, `slope_deg`, `hand_m`, `twi_dinf` | 471.727 / 751 | Servivel, desempenho fora dos criterios fortes |
| `pluvial_urbano_sao_paulo` | Nao servida | `elevation_rel_m`, `slope_deg`, `hand_m`, `twi_dinf` | 7.146 / 1.438 | Nao servivel |
| `fluvial_planicie` | Curitiba | `elevation_rel_m`, `slope_deg`, `hand_m`, `twi_dinf` | 56.654 / 509 | Servivel, coerente com criterios |
| `fluvial_serra` | Petropolis | `hand_m` | 5.162 / 24 | Servivel, coerente com criterios |

## 6. Contrato de inferencia

O contrato fica em `scripts/servico/svc02_contrato_inferencia.py`. Ele centraliza a
decisao cientifica e devolve resposta estruturada. As demais camadas devem apresentar a
resposta, nao reinterpretar portoes.

Os portoes publicos sao:

| Portao | Funcao |
|---|---|
| `G1_REGIAO_SUPORTADA` | A regiao precisa estar declarada no contrato. |
| `G2_MODELO_SERVIVEL` | O modelo regional precisa estar marcado como servivel. |
| `G3_MATURIDADE_MINIMA` | A maturidade regional precisa estar explicita. |
| `G4_VARIAVEIS_OBRIGATORIAS` | Todas as variaveis exigidas pelo modelo precisam estar presentes. |
| `G5_DOMINIO_COVARIAVEL` | O caso precisa estar dentro do dominio ou sair marcado como extrapolacao. |

## 7. Grades de aplicacao

A grade publica e gerada por `scripts/servico/svc03_grade_suscetibilidade.py` e fica em
`modelo/execucoes/svc-03-grade/`.

| Regiao | Resolucao | Artefatos |
|---|---:|---|
| Recife | 120 m | CSV, GeoTIFF, PNG, resumo JSON |
| Curitiba | 120 m | CSV, GeoTIFF, PNG, resumo JSON |
| Petropolis | 120 m | CSV, GeoTIFF, PNG, resumo JSON |

A chuva entra na grade como CENARIO, nao como camada: ela nao varia na escala da celula
de 120 m — Recife inteiro cabe em cerca de 4 celulas do produto de precipitacao — entao
ela desloca o escore sem mudar o ordenamento. Celula fora do dominio sai como NaN e
nunca como zero.

## 8. Frente de chuva

A frente de chuva tem dois papeis distintos, e o pacote nao os mistura.

1. **Variavel do modelo**: `rain_decay_index`, so no `recife_pluvial`.
2. **Pergunta cientifica separada**: a chuva transitoria aumenta o risco de ocorrencia
   notificada no mesmo lugar?

A segunda pergunta usa desenho autopareado no tempo (case-crossover), implementado em
`scripts/chuva/autopareamento_temporal.py`. O estrato e o LUGAR, entao terreno, rede de
drenagem e propensao daquele endereco a acionar a central saem da estimativa por
construcao. O desenho nao usa classe negativa nenhuma: o caso e o proprio controle em
outras datas, e por isso ele nao herda o descasamento de janela entre positivos e fundo
que foi medido nas bases das centrais.

Cadeia reprodutivel:

```text
aq_chamados01_centrais_de_atendimento.py   chamados brutos das centrais
aq_chamados02_verificar_requisitos.py      leitor unico de categoria e data
casos_das_centrais.py                      casos por celula x dia
serie_por_celula.py                        serie de chuva por celula
autopareamento_temporal.py                 estimativa e matriz de forcamento
```

## 9. Teste de transferencia entre cidades

Um efeito medido num lugar so nao distingue lei fisica de particularidade daquele lugar.
O pacote implementa esse teste em duas etapas:

```text
gatilho_terreno_nyc.py          efeito do terreno com chuva a 0,1 grau
terreno_sob_gatilho_1km.py      efeito do terreno com chuva a 1 km (MRMS), estrato = dia
terreno_sob_gatilho_chicago.py  o MESMO estimador, importado, aplicado a segunda cidade
```

O estimador e importado, nunca reescrito. Se as duas cidades passassem por
implementacoes diferentes, uma diferenca de resultado poderia vir do codigo em vez do
territorio, e nao haveria como saber qual.

O resultado esta em `docs/EVIDENCIA.md`, secao 7: chuva e declividade replicam; `twi_dinf`,
`twi_dinf` x chuva e `elevation_rel_m` invertem. O pacote registra a reprovacao e nao
altera modelo servido por causa dela — auditoria complementar nao promove nem rebaixa
modelo dentro do contrato.

## 10. Auditorias complementares

Chuva, temporalidade, recuperacao de fonte, cadeia seletiva, transferencia entre cidades
e camada social sao auditadas como evidencias auxiliares. Elas nao mudam pesos, portoes,
maturidade ou promocao de modelo dentro do contrato de inferencia.

Quando uma frente e inconclusiva, o estado deve continuar escrito como inconclusivo.
Quando o protocolo nao testa uma hipotese naquele recorte, o estado deve continuar
escrito como nao testavel. Quando o teste existe e reprova, como em Chicago, o estado
deve continuar escrito como reprovado — e nao desaparecer do texto publico.
