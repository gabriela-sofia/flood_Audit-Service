# GAP 7 — Poder estatístico em n=282 (141/141)

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-22

## IC do AUC (Hanley & McNeil 1982, ajuste single-fit full-data)

| AUC | SE | IC 95% (±1,96·SE) | Intervalo |
|---|---:|---:|---|
| 0,757 (v3 logreg) | 0,0286 | ±0,056 | [0,701, 0,813] |
| 0,766 (v3 GBM) | 0,0282 | ±0,055 | [0,711, 0,821] |
| 0,826 (v2 logreg) | 0,0248 | ±0,049 | [0,777, 0,875] |

**Mas o desvio-padrão empírico observado no block-CV pooled (150 folds: 3 tamanhos de bloco ×
10 seeds × 5 folds) é 0,098 (v3 logreg)** — quase o dobro da SE de Hanley-McNeil de um único
ajuste full-data. Isso reflete variância adicional real de particionamento espacial (blocos
diferentes geram AUCs de fold muito diferentes: mínimo 0,341, máximo 0,980) — a incerteza real
de deploy espacial é maior do que a fórmula clássica (que assume amostragem i.i.d., não blocos
espacialmente correlacionados) sugeriria.

## Bootstrap de coeficientes (1000 reamostragens, feature set v3, n=270)

| Feature | Ponto (logreg padr.) | IC 95% bootstrap | Cruza zero? | % inversões de sinal |
|---|---:|---|---|---:|
| `elevation_m` | +0,646 | [+0,327, +1,079] | Não | 0,0% |
| `slope_deg` | −0,547 | [−0,979, −0,228] | Não | 0,0% |
| `hand_m` | +0,051 | [−0,288, +0,448] | **Sim** | 34,5% |
| `twi` | +0,128 | [−0,172, +0,426] | **Sim** | 18,1% |
| `impermeabilizacao_proxy` | −0,058 | [−0,430, +0,353] | **Sim** | 32,5% |
| `rain_max_24h_chirps` | +0,565 | [−0,004, +1,386] | quase | 2,8% |
| `rain_decay_index_api_chirps` | +0,876 | [+0,318, +1,471] | Não | 0,0% |

## Interpretação

- **Robustos em n=282**: `elevation_m`, `slope_deg`, `rain_decay_index_api_chirps` (IC nunca
  cruza zero, 0% de inversão de sinal em 1000 reamostragens) — incluindo o achado
  "incoerente" de elevação, que **não é ruído de amostra pequena**, é um efeito estável (reforça
  o veredito do Gap 1: confundidor sistemático de desenho, não flutuação aleatória).
- **Frágeis/não confiáveis em n=282**: `hand_m` (34,5% de inversão), `impermeabilizacao_proxy`
  (32,5%), `twi` (18,1%) — conclusões a nível de coeficiente individual sobre essas três
  features devem ser tratadas como **provisórias**, não afirmações fortes.

## Veredito

**n=282 é adequado para separar "efeito real e estável" de "ruído" via bootstrap** (como
demonstrado acima), mas **não é grande o suficiente para produzir estimativas de coeficiente
individualmente precisas** para features de efeito mais fraco (HAND, TWI, uso do solo
neutralizado). O headline AUC (0,757±0,098 logreg / 0,766±0,092 GBM) deve ser lido com a
largura real do IC em mente (~±0,05 a ±0,10 dependendo do método), não como um número pontual
preciso. As conclusões sobre o coeficiente de elevação (Gap 1) são estatisticamente robustas
nesta amostra — o problema ali é de desenho/confundidor, não de tamanho de amostra.

**Item fora de escopo, apenas registrado**: a discrepância de AUC não resolvida em Petrópolis
(0,428 holdout único vs. 0,812 blockCV, do trabalho v3 original de Petrópolis, diferente deste
Recife v3) permanece um item aberto separado, não investigado nesta rodada.

## Arquivos
- Cálculo direto (`scipy.stats`, bootstrap manual com `numpy.random.default_rng(20260722)`,
  1000 iterações) sobre `dataset_v3_features_finais.csv`
