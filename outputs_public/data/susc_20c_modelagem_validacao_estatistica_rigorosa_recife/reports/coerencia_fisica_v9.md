# RECIFE MODELO v9 — Auditoria de Coerência Física

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-23

## Regras de coerência (hipóteses físicas a priori)

| Feature | Hipótese | Sinal esperado no coeficiente |
|---|---|---:|
| slope_deg | Terreno mais íngreme drena mais rápido → menos suscetível | β < 0 |
| hand_m_dinf | Mais alto acima do canal de drenagem → menos suscetível | β < 0 |
| twi_dinf | Maior índice de umidade topográfica → mais suscetível | β > 0 |
| rain_decay_index_api_chirps / rain_peak_residual | Mais chuva recente/intensa → mais suscetível | β > 0 |
| elevation_m | Menor elevação → mais suscetível (mas confundido com geografia de bairro, ver Task 2) | β < 0 |
| sar_delta_vv/vh_db (informativo, não modelado) | Alagamento → queda maior de VV pós-evento | β < 0 |

## Resultado v9 (Firth multivariado, n=260)

| Feature | Passa? | Nota |
|---|---|---|
| slope_deg | **Sim** | β=−0,160, sinal correto, não sig. (p=0,254), bootstrap flip 13,2% |
| hand_m_dinf | Sinal correto, robustez falha | β=−0,012, sinal correto mas ~0 e p=0,96; bootstrap flip **49,1%** — ver `task3_hand_investigacao.md` (limitação genuína, não CRS/tile) |
| twi_dinf | **Sim, robusto** | β=+0,282, sinal correto, p=0,045, bootstrap flip **1,7%** — feature mais confiável do modelo |
| rain_decay_index_api_chirps | **Sim, robusto** | β=+0,850, sinal correto, p<0,001, bootstrap flip **0,0%** — feature mais forte e mais confiável |
| rain_peak_residual_orthogonalized | Sinal errado | β=−0,071, esperado positivo; não significativo (p=0,637); bootstrap flip 28,7% — efeito residual fraco, não interpretável isoladamente |
| elevation_m | Sinal errado, mas corrigido de "falsamente confiante" para "honestamente instável" | β=+0,268 (esperado negativo); p=0,372 (não sig., era 0,047 sig. em v8); bootstrap flip 18,0% (era 2,7% em v8) — ver `task2_elevacao_crs_investigacao.md` |
| sar_delta_vv/vh_db | Não avaliável como sinal de modelo | Direção observada invertida vs. hipótese em n=113 (Task 1); não incluído no modelo |

**Passagem estrita (sinal correto E bootstrap flip <5%)**: 2/6 features modeladas
(`twi_dinf`, `rain_decay_index_api_chirps`).
**Passagem de sinal apenas (ignorando robustez)**: 3/6 (`slope_deg`, `hand_m_dinf`, `twi_dinf`,
`rain_decay_index_api_chirps` teriam sinal correto — 4/6, mas `hand_m_dinf` tem coeficiente
≈0 e não informativo na prática).

## Leitura honesta

O modelo v9 é dominado por **duas features de chuva/umidade** (`rain_decay_index_api_chirps` e
`twi_dinf`) que carregam sinal real, fisicamente coerente e estatisticamente estável. As
features de terreno estático (`elevation_m`, `hand_m_dinf`, `slope_deg`) continuam fracas ou
ruidosas — não por erro de processamento (ambas investigadas a fundo nesta rodada: Task 2
refutou bug de CRS para elevação e isolou o confundimento de amostragem real; Task 3 testou 3
thresholds de drenagem para HAND e não achou ponto ótimo). Isso é consistente com a leitura
recorrente do projeto desde v5/v7: Recife é fisiograficamente muito plano, e HAND/elevação
"estático" carregam menos sinal discriminativo do que a componente dinâmica de chuva neste
n e nesta escala de dado.
