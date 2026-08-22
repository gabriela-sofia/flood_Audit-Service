# FIX 3 — Aquisição CHIRPS + FIX 2 — Redução do par hidrológico de chuva (Recife v3)

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-22
**Autorizado por**: Gabriela Sofia (reviewer/executora única)

## FIX 3 — Aquisição de precipitação gridded real

### Tentativa 1: CHIRPS (Climate Hazards Center, UCSB) — **SUCESSO**

- Fonte: `https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/{ano}/` —
  GeoTIFFs diários públicos, sem autenticação, resolução 0,05° (~5,5 km).
- **2190 datas-calendário únicas reais** foram identificadas como necessárias (união de todas
  as `event_date` dos 282 pontos do v2 + janela de lookback de 14 dias cada, deduplicada) e
  **todas as 2190 foram baixadas com sucesso** (100%), via download concorrente (8–14 threads,
  com retry/backoff em HTTP 429) e leitura em memória (gunzip + `rasterio.MemoryFile`, sem
  gravar os ~57 MB descomprimidos em disco por data) de uma janela pequena (35×30 px) cobrindo
  os 14 pixels CHIRPS distintos usados pelos 282 pontos.
- Volume real transferido: ~7,2 GB comprimidos (2190 arquivos `.tif.gz`, ~3,3 MB cada).
- Script: `_scratch_chirps_fetch.py` (resumable, grava incrementalmente `date,status,array_json`
  em `_scratch_chirps_daily_grid_values.csv`).

### Tentativa GPM IMERG: **NÃO NECESSÁRIA** — CHIRPS já obtido com sucesso, IMERG (que
exigiria login NASA Earthdata, indisponível nesta sessão sandboxed conforme já documentado em
tentativas anteriores do projeto) não foi tentado.

### Cobertura resultante

- **273/282 pontos** têm `event_date` válida (9 positivos sem data, herdados do v2).
- **273/273 (100%) desses pontos obtiveram os 14/14 dias completos** de lookback via CHIRPS —
  incluindo os **63 pontos pós-2021-11-12** (fim dos dados de precipitação do INMET A301), que
  antes só tinham cobertura diária via ANA (`rain_granularity=daily_only_calendar_day_max`,
  sem nenhum dia faltante). **Isso fecha de fato a lacuna real documentada no v2** — não apenas
  parcialmente: cobertura contínua, sem buracos, para todo o período 2014–2024.

### Sanity check: grid CHIRPS vs. estação ANA/INMET (período de sobreposição)

| Métrica | Pearson r | Spearman r | n sobreposição |
|---|---:|---:|---:|
| `rain_max_24h_chirps` vs. `rain_max_24h_mm` (estação) | **0,658** | 0,654 | 273 |
| `rain_decay_index_api_chirps` vs. `rain_decay_index_api` (estação) | **0,777** | 0,724 | 273 |

Correlação moderada-forte, confirmando que o produto gridded **não diverge descontroladamente**
da estação — mas não é idêntico (esperado: CHIRPS é uma média de área de 0,05°/~5,5 km,
enquanto a estação é um sensor pontual; `rain_max_24h_chirps` é sempre um máximo de
dia-calendário — mesma limitação de granularidade diária já documentada para as estações ANA no
v2 — **não** um pico sub-diário rolante real como o `INMET_A301` fornecia para 72/282 pontos no
v2, então a comparação mistura granularidades para esse subconjunto, reportado honestamente).

### Decisão

Com a aquisição CHIRPS bem-sucedida e o sanity check aprovado, **v3 usa os valores CHIRPS
(`rain_max_24h_chirps`, `rain_decay_index_api_chirps`) como as features de chuva finais do
modelo**, substituindo os valores baseados em estação (mantidos no dataset apenas para
comparação/auditoria, não usados em `FEATURE_COLS_V3`).

---

## FIX 2 — Redução para o par hidrológico (rain_decay_index_api + rain_max_24h)

### Antes (v2, 4 features de chuva, baseadas em estação)

| Par | r |
|---|---:|
| rain_sum_7d ↔ rain_decay_index_api | **0,935** |
| rain_max_24h ↔ rain_decay_index_api | 0,811 |
| rain_sum_3d ↔ rain_decay_index_api | 0,811 |
| rain_max_24h ↔ rain_sum_3d | 0,514 |

**VIF individual (regredindo cada uma nas outras 3)**:

| Feature | VIF |
|---|---:|
| rain_decay_index_api | **22,09** |
| rain_sum_7d | 9,66 |
| rain_max_24h | 4,30 |
| rain_sum_3d | 3,62 |

### Depois (v3, par reduzido, valores CHIRPS)

| Par | Pearson r | VIF (par de 2 features) | n |
|---|---:|---:|---:|
| rain_max_24h_chirps ↔ rain_decay_index_api_chirps | **0,756** | **2,33** | 271 |

Para referência, o mesmo par calculado com os valores de estação (não usado no modelo final):
r=0,806, VIF=2,85 (271→273 pontos).

### Veredito

Colinearidade **drasticamente reduzida** (VIF máximo caiu de 22,1 para 2,33 — bem abaixo do
limiar convencional de preocupação VIF>5–10). Sinais na regressão logística padronizada
(dataset completo, `evaluate_physical_coherence()`): **ambas as features de chuva saem com
sinal positivo coerente e concordância entre logreg e GBM** —
`rain_max_24h_chirps` coef=+0,568 (GBM corr +0,482), `rain_decay_index_api_chirps` coef=+0,871
(GBM corr +0,446, a feature de maior peso no modelo). Nenhum alerta de inconsistência de chuva
disparado (ver `coerencia_fisica_v3.md`).

## Arquivos
- `_scratch_chirps_fetch.py`, `_scratch_chirps_build_daily_table.py` — aquisição/processamento
- `chirps_daily_grid_table.csv` — tabela diária real por pixel CHIRPS (14 pixels, 2190 datas)
- `_scratch_chirps_sanity_check.json` — números do sanity check
- `pipeline_recife_v3.py::build_chirps_features()`, `::vif_two_features()` — código reprodutível
