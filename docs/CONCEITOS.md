# Conceitos

Este arquivo define o vocabulario do projeto. Ele existe porque quase todo erro de
leitura de um sistema de risco ambiental comeca em confundir dois conceitos vizinhos:
suscetibilidade com risco, ausencia de registro com ausencia de evento, transferencia
com validacao.

## 1. Suscetibilidade

Suscetibilidade e a predisposicao fisica de um local a apresentar inundacao, alagamento
ou acumulo, dadas caracteristicas do terreno e, quando o contrato daquela regiao inclui,
chuva antecedente.

Ela **nao** e igual a:

| Nao e | Por que |
|---|---|
| Risco total | Risco exigiria combinar ameaca, exposicao, vulnerabilidade, capacidade de resposta e incerteza. O projeto mantem essas dimensoes separadas. |
| Dano esperado | Dano depende de o que existe no lugar, nao so de para onde a agua vai. |
| Previsao meteorologica | O projeto nao preve chuva futura. |
| Vulnerabilidade social | Fragilidade social nao e causa hidrologica. |
| Confirmacao de evento | Escore alto nao afirma que houve inundacao ali. |

## 2. Contrato fisico

O contrato fisico e a fronteira que impede o projeto de servir inferencia fora das
condicoes minimas declaradas. Ele responde cinco perguntas, nesta ordem:

1. Quais variaveis precisam existir?
2. Qual modelo regional ou transferido deve ser usado?
3. A celula esta dentro de dominio aceitavel?
4. Se nao estiver, qual motivo de recusa deve ser registrado?
5. Qual maturidade acompanha a inferencia?

O contrato e mais importante que qualquer camada auxiliar. Se uma celula nao passa no
contrato, o sistema preserva a recusa — nenhuma camada externa promove um caso recusado.

A implementacao esta em `scripts/servico/svc02_contrato_inferencia.py`, como funcao
pura. Os portoes estao em `docs/METODO.md`, secao 6.

## 3. Maturidade regional

Maturidade e o nivel de sustentacao operacional por regiao e mecanismo. Ela nao e uma
nota de qualidade do projeto; e um **rotulo de escopo** que viaja junto com a resposta.

| Maturidade | Interpretacao | Usada hoje em |
|---|---|---|
| `mvp_local` | Ha modelo local servido para a regiao, com validacao e limitacoes registradas. | Recife |
| `transferencia_caracterizada` | Ha aplicacao por transferencia com caracterizacao de dominio. | Curitiba |
| `transferencia_sem_referencia_local` | Ha aplicacao por transferencia sem rotulo local suficiente para validacao regional. | Petropolis |
| `referencia_externa` | Regiao externa que serve de referencia, nao de alvo de produto. | NYC |
| `diagnostico_local` | Ha treino ou analise local para diagnostico, sem produto servido. | Curitiba local, Sao Paulo |
| `inconclusivo` | O teste feito nao permite afirmar suporte nem falha forte. | escopos temporais |
| `nao_testavel` | A pergunta nao pode ser testada com os dados atuais. | escopos temporais |

A leitura importante: subir de maturidade exige **dado novo**, nao ajuste de modelo.
Nenhuma auditoria auxiliar promove maturidade.

## 4. Recusa

Recusa e resultado tecnico valido, nao falha do sistema. Ela protege quem consulta
contra inferencia indevida.

Motivos possiveis:

- variavel fisica ausente;
- celula fora do dominio do modelo;
- regiao sem maturidade suficiente para aquele uso;
- base temporal insuficiente;
- lacuna de referencia local.

Na grade publica as recusas sao contadas e publicadas: 1.703 celulas em Recife, 965 em
Curitiba, 15.010 em Petropolis. Elas fazem parte do produto e nao devem sumir da
comunicacao.

## 5. Negativo observado versus ausencia de registro

Esta e a distincao que mais muda conclusao no projeto inteiro.

| Estado | Exemplo | Interpretacao correta |
|---|---|---|
| Negativo observado | Classe negativa com evidencia observacional especifica (mascara, vistoria, mancha) | Pode entrar como contraste, se o contrato permitir. |
| Exclusao qualificada | Area ou periodo onde se sabe algo sobre nao ocorrencia, ou area fora da mancha | Tem valor, mas depende da fonte. |
| Ausencia de registro | Nao ha chamado, noticia, mascara nem ponto | **Nao prova ausencia de inundacao.** |
| Fundo amostral | Pontos de contraste sem confirmacao (`target-group background`) | Util para treino, com limitacao forte de interpretacao. |

A consequencia pratica esta no cartao do modelo: quando a amostra negativa e
majoritariamente fundo, a prevalencia nao e identificavel e **so a ordem entre celulas e
sustentada** — a leitura admitida vira ordinal. Quando os negativos sao observados ou
qualificados, a leitura pode ser absoluta.

| Modelo | Fundo na amostra negativa | Leitura admitida |
|---|---:|---|
| `recife_pluvial` | 98,4% | ordinal |
| `pluvial_urbano_nyc` | 100,0% | ordinal |
| `fluvial_planicie` | 0,0% | absoluta |
| `fluvial_serra` | 0,0% | absoluta |

## 6. Variaveis fisicas

### 6.1 HAND — altura acima da drenagem mais proxima

Mede quantos metros o nivel da agua precisaria subir naquele ponto para alcancar a
drenagem mais proxima.

- valores menores indicam proximidade vertical de drenagens;
- em cheia fluvial, HAND baixo costuma ser coerente com maior suscetibilidade;
- em alagamento urbano pluvial, HAND explica menos, porque drenagem artificial e
  microtopografia interferem.

No projeto isso aparece medido, nao suposto: `hand_m` e o coeficiente dominante nos dois
modelos fluviais (−2,5967 na planicie, −1,4423 na serra) e tem IC cruzando zero nos dois
modelos pluviais urbanos.

### 6.2 TWI D-infinity — indice topografico de umidade

Mede tendencia de convergencia: quanta area drena para o ponto em relacao a facilidade
com que a agua sai dele.

- valores maiores sugerem maior acumulacao potencial;
- D-infinity distribui o fluxo entre direcoes continuas, em vez de escolher uma das
  oito direcoes da grade;
- depende fortemente da qualidade do MDT e do processamento hidrologico.

Regra do projeto: TWI a 10 m e TWI a 30 m **nao sao a mesma variavel** (pearson 0,293 em
Recife, 0,205 em Curitiba). Desde 13/08/2026 a resolucao e unica — 30 m, cadeia
WhiteboxTools, limiar de canal 0,1123 km2 — e a regra passa a ser satisfeita por
construcao em vez de por vigilancia.

### 6.3 Declividade

Mede inclinacao local. O sinal esperado depende do mecanismo: em baixa declividade a
agua acumula; em alta declividade ha escoamento rapido e enxurrada.

Quando o sinal observado contradiz a expectativa simples, o projeto **registra a
limitacao** em vez de esconder: `slope_deg` sai positivo em Recife e no `fluvial_planicie`,
contra o esperado, e por isso entra no escore e fica fora da frase de explicacao.

### 6.4 Elevacao relativa

Elevacao contra o P1 da propria regiao, em vez de altitude absoluta. Ela existe para
resolver um problema concreto: a elevacao absoluta punha **0,0% dos pontos de Curitiba**
no dominio de treino do ajuste externo, contra 94,5% da relativa.

Atencao a distincao que o teste de Chicago obrigou o projeto a fazer: `elevation_rel_m`
resolve **dominio** entre regioes; ela nao esta demonstrada como tendo **efeito estavel**
entre regioes. Sao duas propriedades diferentes.

### 6.5 Chuva antecedente — `rain_decay_index`

Resume precipitacao antecedente com decaimento temporal, representando memoria recente
de chuva. Entra apenas no `recife_pluvial`.

- nao e previsao de chuva;
- nao e radar em tempo real;
- na grade de 120 m ela entra como **cenario**, nao como camada, porque nao varia na
  escala da celula.

`rain_max_24h` saiu do vetor: ela media a janela anterior ao evento e ja e parcela da
soma que o indice de decaimento acumula.

## 7. Glossario

| Termo | Definicao |
|---|---|
| AUC | Metrica de discriminacao entre classes no desenho de validacao usado. Nao e probabilidade de inundacao. |
| Autopareamento no tempo (case-crossover) | Desenho que compara exposicao em dias de caso e de controle no mesmo local, cancelando o que e fixo naquele lugar. |
| Bootstrap | Reamostragem para estimar incerteza. Aqui reamostra grupos, nunca linhas. |
| Contrato | Regra que permite escore ou forca recusa. |
| D-infinity | Metodo de direcao de fluxo continua para derivar indices hidrologicos. |
| Extrapolacao | Caso fora da faixa 5–95 do dominio de treino. Nao e recusa: sai marcado, com a distancia declarada. |
| Firth | Correcao de vies para regressao logistica com separacao ou poucos eventos. |
| Fundo (`target-group background`) | Ponto de contraste sem confirmacao de evento. Nao e negativo observado. |
| HAND | Altura acima da drenagem mais proxima. |
| Leitura ordinal | So a ordem entre celulas e sustentada; o valor absoluto nao e propriedade do terreno. |
| Maturidade | Nivel de sustentacao operacional e regional do resultado. |
| Mecanismo | `PLUVIAL_URBANO` ou `FLUVIAL_ENXURRADA`. Modelo so serve regiao do mesmo mecanismo. |
| Pool fluvial | Subconjunto harmonizado compativel com o mecanismo fluvial/enxurrada. |
| Recusa | Saida valida quando o contrato nao permite inferencia. |
| Suscetibilidade | Predisposicao fisica a inundacao ou acumulo. |
| Transferencia | Uso de modelo ajustado fora da regiao local, com limites declarados. |
| TWI | Indice topografico de umidade e convergencia. |
| Vulnerabilidade | Fragilidade social e exposicao. Nao e variavel causal hidrologica. |
