# SUSC-20Z — GAM + interação tensor-spline (2D), Curitiba — fecha a vertente GAM

**Status**: EXECUTADO, resultado negativo (não melhora sobre o GAM aditivo puro do SUSC-20Y).
Encerra a vertente GAM/spline nesta série.

## Motivação

O SUSC-20Y (GAM aditivo puro) fechou 77% do gap entre linear (0,5246) e GBM (0,5888), mas não
o gap inteiro (melhor config = 0,575, gap restante 0,0138). A limitação #3 daquele relatório
apontava a causa provável: um GAM aditivo, por definição, não representa interação genuína
entre pares de features — só uma superfície de spline 2D (tensor-product, o termo `te()` do
mgcv/R) poderia. Este script testa exatamente isso, nos mesmos 3 pares já usados no SUSC-20X
(`rain_decay × hand`, `rain_peak × hand`, `rain_decay × rain_peak`), como continuação direta e
última etapa desta vertente.

## Método

Termo de interação tensor-spline: produto tensorial das bases de spline de duas features —
para cada par de funções-base (uma de cada feature), uma coluna nova = produto das duas. Isso é
adicionado ao MESMO GAM aditivo do SUSC-20Y (5 features, spline própria cada), não substitui a
parte aditiva. Testado por par, todos os 3 combinados, e uma grade de sensibilidade a
hiperparâmetro do tensor (nós 3/4/5 × grau 1/2/3, 9 combinações, sempre com os 3 pares juntos).

## Resultado

| configuração | nº features totais | AUC holdout 2026 |
|---|---:|---:|
| GAM aditivo puro (referência, SUSC-20Y, config default) | 30 | 0,5445 |
| + tensor `rain_decay × hand` | 46 | 0,5504 |
| + tensor `rain_peak × hand` | 46 | 0,5480 |
| + tensor `rain_decay × rain_peak` | 46 | 0,5474 |
| + os 3 tensores combinados | 78 | 0,5546 |
| melhor da grade de hiperparâmetro do tensor (nós=5, grau=1) | 78 | 0,5631 |
| *referência: melhor GAM aditivo puro (grade do SUSC-20Y)* | *—* | *0,575* |
| *referência: GBM (SUSC-20U)* | *5* | *0,5888* |

**A interação tensor-spline melhora sobre o baseline aditivo default (0,5445 → até 0,5631) mas
não supera o melhor GAM puramente aditivo já encontrado no SUSC-20Y (0,575)** — ou seja,
adicionar o termo de interação explícito não trouxe ganho líquido sobre simplesmente dar mais
liberdade à parte aditiva (mais nós/grau, como já testado no SUSC-20Y). O gap para o GBM, que
era 0,0138 com o melhor GAM aditivo, sobe para 0,0257 quando medido contra o melhor resultado
desta rodada (porque a referência de comparação é a mesma, 0,5888, mas o resultado desta rodada
não superou o do SUSC-20Y).

## Leitura

A vertente GAM/spline está, com isso, **exaurida com as ferramentas testadas**: aditivo puro
(SUSC-20Y) chega mais perto (0,575) que aditivo+tensor (0,5631, este script). O motivo mais
provável não é falta de capacidade de representar interação — o tensor-spline pode, em
princípio, aproximar qualquer superfície suave 2D — mas overfitting: cada par de tensor
adiciona 16-25 colunas extras com apenas ~1179 linhas de treino, e a regularização L2 usada não
foi suficiente pra extrair sinal líquido sem também acomodar ruído. Combinado com a leitura do
SUSC-20X (interação produto simples também não ajudou) e do SUSC-20Y (aditivo puro fecha a
maior parte do gap), a leitura mais consistente é: **a maior parte da não-linearidade que o GBM
captura é por-feature (uma curva por variável, sem depender de outra), e o resíduo restante
(~0,01-0,03 de AUC) provavelmente vem de estrutura de ordem mais alta (3+ features) ou splits
regionais específicos** que uma árvore de decisão explora naturalmente e nenhuma forma de GAM
(aditivo ou com par de interação) reproduz com os dados disponíveis.

**Não invalida nenhum achado anterior desta série.** Rota primária continua linear/Firth
interpretável sob EPV. GAM aditivo (SUSC-20Y) permanece o diagnóstico mais próximo de
reconciliar interpretabilidade e performance, mas nenhuma variante testada (produto simples,
limiar, GAM aditivo, GAM+tensor) fecha o gap inteiro pro GBM.

## Limitações

1. Regularização L2 fixa (não testada L1/elastic-net, que poderia zerar colunas de tensor
   pouco úteis e reduzir overfitting).
2. Tensor testado só nos mesmos 3 pares do SUSC-20X (as 3 features mais importantes segundo o
   GBM) — não testada interação envolvendo `slope_deg`/`twi_dinf`, nem interação de 3 vias.
3. N de treino (1179) é pequeno pra bases de tensor com 16-25 colunas por par; um ajuste com
   mais regularização (ex. GAM penalizado tipo mgcv, não disponível no sklearn) poderia extrair
   mais sinal sem overfitting — não tentado aqui por falta de biblioteca instalável no sandbox
   (mesma limitação de shap/xgboost documentada no SUSC-20V).

## Arquivos

- `scripts/pipeline_v20z_gam_tensor_interaction_curitiba.py`
- `results/v20z_tensor_interaction_configs.csv`, `v20z_tensor_hyperparam_grid.csv`,
  `v20z_all_reports.json`
- `tests/test_susc_20z_gam_tensor_interaction.py` (5 testes)

## Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20z_gam_tensor_interaction_curitiba.py
```
