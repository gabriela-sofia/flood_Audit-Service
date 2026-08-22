# SUSC-20T — Mais 3 diagnósticos (lançamento de app real + 2 técnicas de literatura), Curitiba

**Status**: EXECUTADO, três resultados negativos.

## 1. Divisão pelo lançamento do CuritibaApp (2026-03-25)

Achado de notícia real: em 25/03/2026 a Prefeitura de Curitiba lançou um app municipal
unificado com IA própria, absorvendo o 156 gradualmente
([bemparana.com.br](https://www.bemparana.com.br/noticias/parana/app-prefeitura-curitiba-unifica-servicos/),
[curitiba.pr.gov.br](https://www.curitiba.pr.gov.br/noticias/central-156-da-prefeitura-de-curitiba-lanca-nova-plataforma-para-respostas-dos-protocolos/75293)).
Isso cai dentro da janela de teste (jan-jul/2026) — testamos se a quebra de AUC se concentra
no período pós-lançamento.

| janela | n teste | AUC | fração positivo |
|---|---:|---:|---:|
| pré-app (01/01–21/03/2026) | 158 | **0,4341** | 69,6% |
| pós-app (26/03–29/07/2026) | 121 | **0,4656** | 51,6% |

**Resultado**: o colapso NÃO se concentra no período pós-app — o período pré-app (antes do
app existir) tem AUC ainda pior (0,4341 vs. 0,4656). A composição de positivo/negativo muda
visivelmente (70%→52%), mas isso não explica o colapso de AUC, que já está presente antes da
mudança de sistema. **Hipótese refutada.**

## 2. `rain_max_24h_chirps` como feature alternativa

Coluna já presente no dataset, nunca usada como feature de modelo (mais simples que os
índices de pico/decaimento atuais).

| ano | n | correlação com label |
|---:|---:|---:|
| 2023 | 266 | 0,0153 |
| 2024 | 351 | −0,0801 |
| 2025 | 572 | 0,1037 |
| 2026 | 282 | 0,0160 |

Correlação fraca e inconsistente em **todos** os anos, não só 2026 — nunca foi um sinal forte.
Holdout temporal (terreno + esta feature no lugar dos 2 índices atuais): AUC=0,523, igual ao
baseline (0,5246). **Não ajuda — não é feature alternativa viável.**

## 3. Treino com peso de recência (decaimento exponencial)

| meia-vida | peso 2023 | peso 2024 | peso 2025 | AUC holdout |
|---:|---:|---:|---:|---:|
| 1 ano | 0,25 | 0,50 | 1,00 | 0,5241 |
| 2 anos | 0,50 | 0,71 | 1,00 | 0,5223 |

Ambos praticamente idênticos ao baseline (0,5246) — dar mais peso aos anos recentes não
recupera desempenho. **Não ajuda.**

## Síntese

Três vertentes adicionais testadas, três negativas. Somando com SUSC-20P/Q/R/S, o conjunto de
explicações e correções testadas e descartadas agora inclui: vazamento espacial, sazonalidade,
ruído de amostra, deriva administrativa/metadado agregado, ENOS, viés de rótulo negativo
contaminado (PU bagging), mudança de sistema de atendimento (lançamento de app real, com data
verificada), feature de chuva alternativa mais simples, e ponderação por recência. Nenhuma
recupera o desempenho prospectivo.

## Limitações

1. Divisão pré/pós-app é um corte único, sem repetição possível (só existe 1 data de
   lançamento).
2. A modernização do 156 em si foi descrita como incorporação "aos poucos" ao novo app — o
   corte de 25/03 é o melhor proxy disponível, mas a transição pode ter sido gradual, não
   instantânea.
3. `rain_max_24h_chirps` não substitui os dois índices atuais nas outras validações (só
   testado no holdout temporal simples).
4. Peso de recência testado só com 2 meia-vidas (1 e 2 anos) — não variado exaustivamente.

## Arquivos

- `scripts/pipeline_v20t_more_diagnostics_curitiba.py`
- `results/v20t_all_reports.json`
- `tests/test_susc_20t_more_diagnostics.py` (4 testes)

## Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20t_more_diagnostics_curitiba.py
```
