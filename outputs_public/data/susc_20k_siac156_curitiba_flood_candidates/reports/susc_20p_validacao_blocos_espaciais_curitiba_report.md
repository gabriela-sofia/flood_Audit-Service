# SUSC-20P — Validação por blocos espaciais (bairro), Curitiba

**Status**: EXECUTADO. GroupKFold(n_splits=5) por bairro, 1471 unidades, 73 bairros. Rota
primária do SUSC-20N/20M/20O (5 features, `elevation_m` fora — decisão causal mantida, não
revisitada aqui).

## Motivação

O SUSC-20O mostrou que o AUC cai de 0,6459 (LOO-CV embaralhado) para 0,5246 (holdout temporal
2026). A literatura de suscetibilidade a enchente aponta duas fontes possíveis pra esse tipo de
colapso, e elas pedem diagnósticos diferentes:

1. **Vazamento espacial** — autocorrelação entre unidades do mesmo bairro infla o CV
   embaralhado porque duas queixas do mesmo bairro (ou vizinhas) caem uma no treino e outra no
   teste; o modelo "decora" identidade de vizinhança em vez de física de terreno. Estudos
   comparando random split vs. spatial block CV relatam AUC 5–15% mais alto sob split aleatório
   ([ScienceDirect, "Next generation data-driven flood susceptibility modelling with spatial
   machine learning"](https://www.sciencedirect.com/science/article/pii/S2468227625005514)).
2. **Deriva temporal/administrativa** — o padrão de queixas do SIAC 156 muda ano a ano
   (2025 com proporção de positivos desigual; 2026 é ano parcial sem toda a sazonalidade) —
   mecanismo distinto de vazamento espacial, já levantado como hipótese aberta no SUSC-20O.

Este script isola a hipótese (1): bloco = bairro (unidade administrativa já catalogada, não
grade arbitrária), nenhum bairro aparece simultaneamente em treino e teste do mesmo fold
(`GroupKFold`, sklearn).

## Resultado

**AUC médio por bloco espacial: 0,6442** (desvio 0,0316; min 0,6230; max 0,6999; 5 folds).

| validação | AUC |
|---|---:|
| LOO-CV (embaralhada, SUSC-20N) | 0,6459 |
| 5-fold repetido 50× (embaralhada, SUSC-20N) | 0,6440 |
| **spatial block CV (bairro nunca visto, SUSC-20P)** | **0,6442 (±0,032)** |
| holdout temporal (2026 nunca visto, SUSC-20O) | 0,5246 |

| fold | bairros no teste | EPV (treino) | n treino | n teste | pos/neg teste | AUC |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 15 | 69,2 | 1165 | 293 | 216/77 | 0,6230 |
| 1 | 14 | 67,0 | 1164 | 294 | 206/88 | 0,6999 |
| 2 | 14 | 65,2 | 1168 | 290 | 193/97 | 0,6371 |
| 3 | 15 | 69,0 | 1171 | 287 | 209/78 | 0,6283 |
| 4 | 15 | 68,0 | 1164 | 294 | 211/83 | 0,6329 |

Todos os 5 folds passam o piso EPV (20,0) com folga (65,2–69,2). Cada um dos 73 bairros de
Curitiba sai inteiro para teste em exatamente 1 fold — nunca aparece em treino e teste do mesmo
fold (checado por assert no código, coberto por teste).

## Leitura

O AUC por bloco espacial (0,6442) é estatisticamente indistinguível do LOO-CV embaralhado
(0,6459) — a diferença está dentro do desvio entre folds (±0,032) e é muito menor que a queda
observada no holdout temporal (0,6459→0,5246). Isso quer dizer: **o modelo generaliza bem para
bairros inteiros nunca vistos no treino.** Não há evidência de que o CV embaralhado do SUSC-20N
estivesse inflado por autocorrelação espacial/identidade de vizinhança.

Isso concentra a explicação do colapso na segunda hipótese do SUSC-20O, ainda não testada
isoladamente: deriva temporal/administrativa (composição do SIAC 156 muda de ano pra ano de
forma não-física, e/ou 2026 parcial tem sazonalidade de chuva não comparável ao treino anual).
Não é mais "uma entre duas hipóteses não-excludentes" — é a hipótese remanescente depois de
descartar vazamento espacial com evidência direta.

**Não é motivo pra trocar de método**: confirma que o problema de generalização é temporal, não
espacial, o que estreita o próximo teste útil (ex.: comparar composição ano-a-ano do SIAC 156
diretamente, ou testar holdout por safra chuvosa em vez de ano civil) em vez de continuar
generalizando sobre "o modelo não generaliza".

## Limitações

1. Bloco = bairro, não grade de distância fixa — bairros de Curitiba têm tamanho e formato
   heterogêneos; um bairro pequeno espacialmente adjacente a um bairro grande do fold de treino
   ainda pode ter alguma proximidade física não capturada por essa unidade administrativa.
2. `GroupKFold` do sklearn não randomiza a atribuição de grupos a folds (é determinístico dado a
   ordem de entrada) — não há repetição com sementes diferentes aqui, ao contrário do 5-fold
   repetido 50× do SUSC-20N.
3. Não testa a hipótese de deriva temporal/administrativa — fica para uma rodada futura,
   explicitamente fora de escopo aqui (uma tarefa por vez).
4. `elevation_m` não foi testada aqui — decisão causal do SUSC-20N mantida sem revisão.

## Arquivos

- `scripts/pipeline_v20p_spatial_block_cv_curitiba.py`
- `results/v20p_all_reports.json`, `v20p_spatial_block_cv_fold_results.csv`
- `tests/test_susc_20p_spatial_block_cv.py` (5 testes)

## Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20p_spatial_block_cv_curitiba.py
```
