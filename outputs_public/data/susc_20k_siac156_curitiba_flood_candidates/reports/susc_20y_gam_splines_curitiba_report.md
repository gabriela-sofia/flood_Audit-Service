# SUSC-20Y — GAM aditivo com splines por feature, Curitiba

**Status**: EXECUTADO, resultado misto (melhora parcial, não recupera totalmente o GBM).

## Motivação

O SUSC-20X testou interação-produto e limiar binário dentro do modelo linear e falhou em
recuperar o sinal não-linear do GBM (melhor tentativa 0,5295 vs GBM 0,5888 — gap 0,0593). A
limitação apontada naquele relatório era não ter testado um GAM (generalized additive model)
com spline por feature, que captura a *forma* real da curva em vez de um único corte ou produto.

Diferença chave de método em relação ao SUSC-20X: o GAM continua **aditivo por construção** —
cada feature entra com seu próprio bloco de spline (B-spline cúbica, nós por quantil),
concatenados e passados por uma única regressão logística. **Não há nenhum termo cruzado entre
features.** Isso faz do GAM um teste mais rigoroso, não mais permissivo, que o SUSC-20X: se o
GAM recuperar o sinal do GBM, a não-linearidade é por-feature (boa notícia, interpretável, cada
feature ganha uma curva de efeito isolada e plotável). Se não recuperar, reforça que há
interação real entre features (estrutura de ordem mais alta), não capturável por soma de curvas
independentes.

## Método

- `SplineTransformer` (sklearn 1.7) por feature, independente — nós por quantil, sem termo de
  bias (intercepto vem da regressão logística).
- Bases concatenadas → `LogisticRegression(penalty="l2", class_weight="balanced")`.
- Config default: 5 nós, grau 3, C=1,0 (mesmos valores usados como ponto de partida nos outros
  diagnósticos desta rodada).
- Holdout temporal (treino 2023-2025, teste 2026), spatial block CV (bairro, 5 folds) e grade de
  sensibilidade a hiperparâmetro (5 níveis de nós × 2 graus × 3 valores de C = 30 combinações).
- Curva de efeito aditivo por feature: como o modelo é aditivo, o efeito de cada feature é
  `spline_feature(x) · coef_feature` — recuperável diretamente, sem precisar "congelar" as
  outras features (partial dependence tradicional precisaria disso; aqui já vem congelado pela
  própria forma do modelo).

## Resultado

| medida | valor |
|---|---:|
| baseline linear (5 features, sem spline) | 0,5246 |
| GAM holdout 2026, config default (5 nós, grau 3, C=1) | 0,5445 |
| GAM spatial block CV, config default (média 5 folds) | 0,6509 |
| GAM melhor holdout na grade de 30 combos | 0,575 |
| GBM (referência, SUSC-20U) | 0,5888 |
| gap pro GBM (melhor GAM vs GBM) | 0,0138 |
| % combos da grade acima do baseline linear | 96,7% |
| % combos da grade que igualam/superam o GBM | 0% |

**O GAM melhora sobre o linear puro (0,5246 → até 0,575) mas não fecha o gap pro GBM.** O gap
caiu de 0,0593 (melhor tentativa do SUSC-20X) para 0,0138 (melhor GAM) — uma redução de ~77% no
gap, a aproximação mais próxima do GBM conseguida com um modelo aditivo/interpretável até agora
nesta série. Ainda assim, nenhuma das 30 configurações testadas alcança ou supera o patamar do
GBM.

### Curvas de efeito aditivo (config default, ajustado no treino 2023-2025 completo)

| feature | direção geral | monotônica? | direção bate com sinal causal esperado? |
|---|---|---|---|
| `slope_deg` | decrescente | não (3 trocas de sinal na inclinação) | sim |
| `hand_m_dinf` | decrescente | não (2 trocas) | sim |
| `twi_dinf` | decrescente | não (2 trocas) | **não** |
| `rain_peak_residual_orthogonalized` | decrescente | não (2 trocas) | **não** |
| `rain_decay_index_api_chirps` | crescente | não (3 trocas) | sim |

3 de 5 features batem com a direção causal esperada (documentada desde o SUSC-20M/20N);
`twi_dinf` já havia aparecido como anômalo no SUSC-20V (GBM). Nenhuma das 5 curvas é monotônica
— todas têm pelo menos 2 trocas de sinal na inclinação, consistente com a leitura de que a
relação real não é uma função simples/linear, mesmo isoladamente por feature.

## Leitura

Resultado misto, mas informativo em duas direções:

1. **O GAM aditivo captura uma fração real do sinal não-linear** (redução de 77% no gap vs a
   tentativa anterior), o que sugere que boa parte da não-linearidade que o GBM explora é sim
   por-feature (forma de cada curva isolada), não pura interação entre pares. Isso é
   consistente com uma leitura fisicamente razoável: a relação entre `hand_m_dinf` e risco não
   precisa ser linear (proximidade da rede de drenagem pode ter efeito de limiar natural), e o
   GAM consegue captar isso sem sacrificar a aditividade.
2. **Mas não fecha o gap inteiro** — resta ~0,014 de AUC que o GAM aditivo não alcança. A leitura
   mais direta é que parte (menor, mas real) do sinal do GBM vem de interação genuína entre
   features (ex.: o efeito de `hand_m_dinf` mudar de acordo com o nível de `rain_decay_index`),
   que nenhum modelo aditivo — por mais flexível que seja a curva de cada feature — consegue
   capturar por definição.

**Isso não muda a decisão metodológica primária**: a rota causal continua sendo o modelo
linear/Firth com as 5 features originais (interpretável, coeficiente único, sob EPV). O GAM é
registrado como o diagnóstico que mais se aproximou de reconciliar performance e
interpretabilidade nesta série, mas segue sendo um diagnóstico à parte, não uma substituição da
rota primária — a mesma regra que vem sendo aplicada desde o SUSC-20U.

## Limitações

1. Nós de spline colocados por quantil (`knots="quantile"`), não otimizados via busca fina; a
   grade de 30 combos cobre uma faixa razoável (3-8 nós, grau 2-3, C 0,1-10) mas não é
   exaustiva.
2. Curva de efeito reportada no treino completo (2023-2025); não foi re-extraída fold a fold no
   spatial block CV — a forma pode variar levemente entre folds, não testado aqui.
3. GAM aditivo por construção não pode, por definição, capturar interação real entre pares de
   features — se essa é a fonte do gap restante, nenhuma variação de nós/grau vai fechá-lo; só
   um termo de interação explícito (testado no SUSC-20X, sem sucesso com produto/limiar simples)
   ou uma superfície de spline 2D (não tentada) poderia.
4. `firthlogist` (usado na rota primária linear) não foi testado com base de spline — o GAM
   aqui usa `LogisticRegression` padrão (penalizada), não Firth; comparação direta de
   coeficiente/p-valor com a rota primária não é direta.

## Arquivos

- `scripts/pipeline_v20y_gam_spline_curitiba.py`
- `results/v20y_hyperparam_grid.csv`, `v20y_additive_effect_curves.csv`, `v20y_all_reports.json`
- `tests/test_susc_20y_gam_spline.py` (6 testes)

## Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20y_gam_spline_curitiba.py
```
