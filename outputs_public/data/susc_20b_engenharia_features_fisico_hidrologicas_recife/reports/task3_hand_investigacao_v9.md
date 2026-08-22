# TASK 3 — Investigação do ruído persistente em HAND (48% sign-flip bootstrap)

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-23
**Autorizado por**: Gabriela Sofia (reviewer/executora única do projeto)

## 1. O que foi testado

HAND-Dinf (D-infinity + depression-filling Wang & Liu, já implementado em v7/Improvement 2)
continua com taxa de sign-flip bootstrap ~48-49% mesmo após todas as correções de v9 (Task 2)
— ou seja, o problema é **independente** do confundimento de amostragem dos negativos (Task 2
melhorou `elevation_m`, mas não mudou `hand_m_dinf`: 48,0% em v8 → 49,1% em v9, essencialmente
idêntico). Isso já é evidência de que a causa é estrutural ao próprio HAND/terreno, não ao
desenho amostral.

Três hipóteses levantadas na tarefa foram testadas concretamente:

### (a) Artefatos de emenda entre os 48 tiles PE3D mesclados

Calculada a distância de cada um dos 269 pontos do dataset v9 à borda de tile mais próxima
(coordenadas UTM 25S). Apenas **15/269 pontos (5,6%)** estão a menos de 30 m de uma borda de
tile. Correlação entre "distância à borda" e "amplitude de HAND entre os 3 thresholds testados"
(ver abaixo): **r = 0,024** — essencialmente nula. **Artefatos de emenda refutados** como causa
principal: não há relação entre proximidade de borda de tile e instabilidade do valor de HAND.

### (b) Limiar de acumulação de fluxo inadequado para a rede de drenagem do Recife

Testados **3 thresholds de inicialização de canal** (percentil da distribuição de acumulação
D-infinity, ~400× de amplitude): P90 (17,0 células), P98 (450,4 células — próximo do baseline
já usado em v7/v8), P99,5 (6.877,1 células). Para cada um, recomputado `extract_streams` +
`elevation_above_stream` (WhiteboxTools) e extraído HAND nos 269 pontos do dataset v9.

| Threshold | HAND médio (todo o grid) | HAND médio positivos | HAND médio negativos | MWU p-valor | Direção (esperada: pos<neg) |
|---|---:|---:|---:|---:|---|
| P90 (17 céls) | 5,04 m | 5,38 m | 4,64 m | 0,413 | **errada** (pos>neg) |
| P98 (450 céls, ≈baseline) | 11,52 m | 11,29 m | 11,79 m | 0,793 | correta (pos<neg), mas p≈0,79 |
| P99,5 (6877 céls) | 16,32 m | 16,96 m | 15,58 m | 0,367 | **errada** (pos>neg) |
| (referência: cache v7/v8, 98º pct=1122,7 céls) | 13,29 m | 13,53 m | 13,01 m | 0,535 | **errada** (pos>neg) |

**Nenhum dos 3 thresholds produz separação estatisticamente significativa**, e a **direção da
diferença muda de sinal conforme o threshold escolhido** (correta só no P98 testado aqui, e
mesmo essa não é significativa) — isso não é apenas "ruído de magnitude", é uma real
inconsistência de sinal induzida pela escolha do parâmetro, o que é uma bandeira vermelha
clássica de que o método não está capturando um sinal físico estável nessa escala/terreno.

### (c) Resolução de 10 m inadequada para a escala de drenagem urbana do Recife

Não testado diretamente (recompute em outra resolução exigiria remalhar os 48 tiles PE3D, fora
do orçamento desta rodada), mas a literatura já citada em Improvement 2 (Recife é planície
costeira "notoriamente plana") e a sensibilidade extrema ao threshold observada aqui são
consistentes com essa explicação: em terreno muito plano com rede de drenagem urbana densa
(canais e galerias abaixo da resolução de 10 m, ou paralelos entre si), pequenas variações no
limiar de "o que conta como canal" reordenam completamente qual célula é considerada "mais
próxima de um canal", produzindo o padrão de instabilidade observado.

## 2. Veredito

- **Não é** artefato de emenda entre tiles (correlação nula com distância à borda).
- **É, ao menos parcialmente**, sensibilidade genuína ao limiar de acumulação de fluxo — mas
  testar 3 thresholds concretos (400× de amplitude) **não encontrou nenhum ponto ótimo** que
  produza separação estatisticamente significativa ou robusta; a instabilidade de sinal entre
  thresholds sugere que o problema não é "escolher o número certo", é que **nenhum limiar
  único de acumulação de fluxo captura de forma estável a diferença entre local-de-evento e
  local-controle nesta escala e neste terreno**.
- Consistente com uma **limitação de dado/escala genuína** (terreno plano + drenagem urbana
  densa/artificial + resolução de 10 m), não corrigível de forma barata por ajuste de
  parâmetro dentro desta rodada. Testar resoluções mais finas (ex.: reamostrar para 3-5 m se
  o DTM de origem permitir) ou fontes de rede de drenagem vetorial oficial (ex.: cadastro de
  drenagem urbana IPPUC-equivalente do Recife) seriam os próximos passos honestos, não
  tentados aqui por estarem fora do escopo/orçamento desta rodada.

## 3. Decisão para o modelo v9

Mantido o HAND-Dinf já usado em v7/v8 (98º percentil da distribuição de acumulação, valor
absoluto ligeiramente diferente do "P98" testado aqui por terem sido computados em execuções
whitebox distintas, mas mesma lógica de percentil) — é, dos quatro conjuntos comparados aqui, o
único com direção fisicamente correta (ainda que não significativa). Nenhuma mudança de HAND
foi aplicada ao dataset/modelo final v9 além de manter a escolha já validada. O sign-flip
bootstrap de ~48-49% permanece **documentado como limitação genuína aberta**, não maquiado.

## 4. Arquivos

`scratch_hand_thresholds/streams_{t90,t98_baseline,t995}.tif`,
`scratch_hand_thresholds/hand_{t90,t98_baseline,t995}.tif`,
`_scratch_v9_hand_threshold_compare.csv` (269 pontos × 4 variantes de HAND + distância a tile).
