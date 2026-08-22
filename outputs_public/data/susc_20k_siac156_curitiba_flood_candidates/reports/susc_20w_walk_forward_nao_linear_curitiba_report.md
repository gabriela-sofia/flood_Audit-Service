# SUSC-20W — Walk-forward multi-corte para 4 classes de modelo, Curitiba

**Status**: EXECUTADO. Checagem de robustez decisiva para o achado do SUSC-20U/20V: a
vantagem não-linear **não é específica de 2026** — aparece nos 3 cortes prospectivos
disponíveis.

## Motivação

Antes de aceitar o achado do SUSC-20U/20V (modelos não-lineares superam o linear em 2026),
era preciso checar uma explicação alternativa: será que o modelo mais flexível está só
explorando ruído específico do conjunto de teste de 2026 (overfitting ao teste), não captando
estrutura real? Se a vantagem só aparecer em 2026 e não nos outros cortes prospectivos
(2023→2024, 2023-24→2025), essa seria a leitura mais provável. Se aparecer nos 3, a leitura
muda pra "existe não-linearidade real nas features, presente em qualquer corte".

## Resultado

| modelo | teste 2024 | teste 2025 | teste 2026 |
|---|---:|---:|---:|
| Linear (baseline) | 0,6282 | 0,6652 | 0,5246 |
| GBM | **0,6776** | 0,6587 | **0,5888** |
| AdaBoost | **0,6481** | **0,6799** | **0,5771** |
| Random Forest (d=3) | **0,6841** | **0,6949** | **0,5611** |

| ano de teste | modelos não-lineares acima do linear |
|---:|---:|
| 2024 | 3 de 3 |
| 2025 | 2 de 3 (GBM fica 0,0065 abaixo do linear neste corte) |
| 2026 | 3 de 3 |

## Leitura

**A vantagem não-linear aparece nos 3 cortes, não só em 2026** — 8 de 9 comparações
modelo×ano ficam acima do linear (a única exceção é o GBM em 2025, por uma margem mínima de
0,0065). Isso descarta a hipótese de que o ganho de 2026 seja um artefato de overfitting ao
ruído específico daquele conjunto de teste — é consistente com existir não-linearidade real
nas 5 features causais, presente em qualquer recorte temporal, não uma peculiaridade de 2026.

Ao mesmo tempo, **2026 continua sendo o ano mais difícil de prever pra qualquer classe de
modelo** — o AUC absoluto de 2026 é o mais baixo em todas as 4 linhas da tabela, mesmo para os
modelos não-lineares (0,56-0,59, contra 0,63-0,68 em 2024 e 0,66-0,69 em 2025). Isso é
consistente com a deriva já diagnosticada no SUSC-20Q (a relação chuva↔queixa muda
especificamente em 2026): a não-linearidade **compensa parcialmente** essa deriva (todos os
modelos melhoram sobre o linear), mas **não a resolve** (2026 segue sendo o pior ano pra
qualquer modelo).

## Síntese

Duas conclusões independentes, ambas reais:

1. **Existe não-linearidade genuína e generalizável** nas 5 features causais — não é
   específica de 2026, aparece em praticamente todo corte prospectivo testado.
2. **A deriva específica de 2026 continua sem explicação causal identificada** — nenhum
   modelo, linear ou não-linear, chega perto do desempenho que tem em 2024/2025 quando
   testado em 2026.

Essas duas conclusões não competem entre si — são achados complementares sobre partes
diferentes do mesmo problema.

## Limitações

1. Só 3 cortes disponíveis (2023-2026) — poder estatístico limitado pra generalizar o padrão
   "não-linear > linear" com alta confiança.
2. GBM ficou levemente abaixo do linear em 2025 — não investigado a fundo se é ruído ou sinal
   real de que o gradient boosting específico é menos robusto que Random Forest/AdaBoost neste
   recorte.
3. Não decompõe se a mesma estrutura não-linear (mesmos limiares) se repete nos 3 cortes ou se
   cada um encontra um padrão diferente — ficaria pra uma rodada de aprofundamento.

## Arquivos

- `scripts/pipeline_v20w_nonlinear_walk_forward_curitiba.py`
- `results/v20w_all_reports.json`, `v20w_nonlinear_walk_forward_all_models.csv`
- `tests/test_susc_20w_nonlinear_walk_forward.py` (4 testes)

## Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20w_nonlinear_walk_forward_curitiba.py
```
