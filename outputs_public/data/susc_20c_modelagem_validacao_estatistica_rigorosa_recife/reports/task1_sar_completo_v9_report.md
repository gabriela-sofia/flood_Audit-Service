# TASK 1 — Sentinel-1 SAR pre/post-evento: extração completa (v8 n=20 → v9 n=113)

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-23
**Autorizado por**: Gabriela Sofia (reviewer/executora única do projeto)

## 1. O que foi escalado

v8 tinha identificado 124/181 pontos primários (68,5%) com par Sentinel-1 GRD pré/pós real
dentro de ±15 dias (Earth Search STAC, `s3://sentinel-s1-l1c` anônimo), mas só processara o
pixel (calibração radiométrica manual) para 24/124 por limite de tempo/engenharia — **não** de
disponibilidade de dados.

Nesta rodada, **os 124/124 pontos elegíveis foram processados** (mesmo método validado de v8,
sem mudança de metodologia — só escala): geolocalização por ajuste afim local (GCPs ±0,6° da
AOI do Recife), parse real do XML de calibração por cena (`calibration-iw-{vv,vh}.xml`),
`sigma0 = DN²/A²`, `backscatter_dB = 10·log10(sigma0)`.

### Bug real encontrado e corrigido durante o escalonamento

O cache de geolocalização/calibração por cena (`_scratch_sar_scene_geoloc_calib_cache_v9.pkl`)
era mutado por múltiplas threads sem lock, causando `RuntimeError: dictionary changed size
during iteration` intermitente e, pior, **travamentos indefinidos** de alguns workers presos em
threads não daemon que o `ThreadPoolExecutor` (`with` context manager) esperava terminar antes
de sair — isso travava o processo inteiro por chamada, mesmo com orçamento de tempo interno
baixo. Corrigido com `threading.Lock` em torno do dicionário compartilhado + `shutdown(wait=False,
cancel_futures=True)` + `os._exit(0)` no fim (não espera threads penduradas). Após a correção, o
throughput subiu de ~1-4 pontos/chamada para ~15-20 pontos/chamada. 16 pontos que haviam falhado
por essa race condition (`dictionary changed size...`) foram reprocessados com sucesso após o
fix — **124/124 pontos elegíveis têm extração completa, 0 erros remanescentes**.

## 2. Amostra final utilizável

- **124 pares brutos** (97 positivos / 27 negativos, gap ≤15 dias cada lado).
- **11 pares ambíguos** (mesma cena retornada para "antes" e "depois" por limite de fronteira
  dia-inteiro) excluídos por ordem temporal indeterminada — igual critério do v8.
- **n=113 pares genuínos** (89 positivos / 24 negativos) com ΔVV e ΔVH calculados
  (`sar_backscatter_pairs_v9_final.csv`).

Isso é **5,65× o n=20 do v8** (13 pos / 7 neg → 89 pos / 24 neg).

## 3. Resultado estatístico (n=113, vs n=20 do v8)

| Feature | Positivos (n=89) | Negativos (n=24) | MWU p (bicaudal) | MWU p (unicaudal, hipótese física) |
|---|---:|---:|---:|---:|
| ΔVV (pós−pré, dB) | média −1,06 / mediana −1,67 | média −2,30 / mediana −2,56 | **0,455** | 0,775 (hipótese não sustentada) |
| ΔVH (pós−pré, dB) | (mesmo n) | (mesmo n) | **0,446** | — mesma direção invertida |

Comparação com v8 (n=20): p=0,20 (VV) / p=0,76 (VH); direção já invertida vs. hipótese.

## 4. Coerência física

Hipótese: superfícies alagadas mostram VV mais negativo pós-evento (reflexão especular da água
parada) → esperado ΔVV mais negativo nos positivos. **A direção observada permanece invertida**
mesmo em n=113: os negativos mostram queda de VV/VH numericamente maior que os positivos, e
nenhuma diferença é estatisticamente significativa.

## 5. Veredito honesto

Com n=20, a conclusão só podia ser "inconclusivo" (amostra pequena demais para decidir).
Com **n=113 (5,65× maior)**, a direção invertida **se mantém** e a magnitude do p-valor não se
aproxima de significância (0,45 vs 0,20/0,76 — na verdade menos extremo, não mais). Isto é uma
evidência bem mais forte do que o n=20 permitia, e aponta para uma **conclusão negativa real**,
não apenas "não pudemos detectar sinal": a hipótese simples de reflexão especular em água calma
e aberta **não parece se sustentar no contexto urbano/estuarino do Recife** neste desenho
(escala de pixel único 5×5, geolocalização aproximada ~10-20m, mistura de superfícies
urbanas/vegetadas/água no mesmo pixel, e o próprio conjunto de "positivos" sendo eventos de
alagamento urbano transitório, não necessariamente lâminas de água abertas e estáveis no
momento exato da passagem do satélite).

**Decisão**: `sar_delta_vv_db` / `sar_delta_vh_db` **não foram integrados** como features do
modelo primário v9 — anexados apenas como colunas informativas
(`sar_delta_vv_db_INFORMATIONAL_NOT_MODELED` / `..._vh_...`) em `dataset_v9_final.csv` para
transparência e uso exploratório futuro, não como preditor.

## 6. Arquivos

- `extract_sar_backscatter_v9_full.py` — extração completa (124/124), com o fix de
  concorrência documentado.
- `_scratch_sar_usable_full124.csv` — os 124 pares elegíveis (derivados do cache STAC do v8).
- `_scratch_sar_backscatter_cache_v9.csv` — cache bruto ponto-a-ponto (124 linhas).
- `sar_backscatter_pairs_v9_merged_raw.csv` — merge completo com metadados de cena/gap.
- `sar_backscatter_pairs_v9_final.csv` — **n=113 pares válidos finais** (não-ambíguos, com
  ΔVV/ΔVH calculados) usado nas estatísticas acima.
