# SUSC-20Q — Bateria exaustiva de diagnósticos do colapso de AUC prospectivo, Curitiba

**Status**: EXECUTADO. 6 diagnósticos independentes, todos sobre dado já existente (nenhuma
aquisição nova). Rota primária do SUSC-20N/20M/20O/20P (5 features, `elevation_m` fora).

## Motivação

Pedido explícito (2026-08-02): testar exaustivamente toda vertente que a literatura oferece
pra diagnosticar o colapso de AUC 0,6459 (LOO-CV embaralhado, SUSC-20N) → 0,5246 (holdout
temporal 2026, SUSC-20O). O SUSC-20P já descartou vazamento espacial (spatial block CV =
0,6442, igual ao embaralhado). Esta rodada isola as demais hipóteses.

## Os 6 diagnósticos

### 1. Bootstrap CI no AUC do holdout temporal

`n_teste=279`, 2000 reamostras bootstrap do conjunto de teste (classificador ajustado 1x no
treino, não re-treinado por reamostra).

| métrica | valor |
|---|---:|
| AUC ponto | 0,5246 |
| IC 95% | [0,4535; 0,5959] |
| IC inclui 0,5 (acaso)? | **sim** |
| IC exclui a referência embaralhada (0,6459)? | **sim** |

**Leitura**: o IC inclui 0,5 — o AUC do holdout temporal **não é estatisticamente distinguível
de acaso** a 95%. Ao mesmo tempo, o IC exclui com folga o AUC de 0,6459 medido em CV
embaralhado — a queda em si é real, não ruído de amostra pequena. Achado mais forte que uma
simples "queda": o modelo pode não ter poder preditivo prospectivo genuíno em 2026.

### 2. Ablação terreno-só vs. chuva-só sob holdout temporal

| grupo | n features | AUC holdout 2026 |
|---|---:|---:|
| terreno-só (slope/HAND/TWI) | 3 | 0,5213 |
| chuva-só (peak/decay) | 2 | **0,4984** |
| completo (5 features) | 5 | 0,5246 |

**Leitura**: nenhum dos dois grupos generaliza — chuva-só fica literalmente abaixo do acaso.
Refuta a hipótese inicial de que "chuva carrega a deriva e terreno se mantém": os dois grupos
colapsam juntos, o que aponta pra uma mudança mais geral na relação feature↔label em 2026, não
localizada num subconjunto de variável.

### 3. Walk-forward multi-corte

| treino | teste | EPV (treino) | AUC |
|---|---:|---:|---:|
| 2023 | 2024 | 21,0 | **0,6282** |
| 2023, 2024 | 2025 | 41,8 | **0,6652** |
| 2023, 2024, 2025 | 2026 | 63,0 | **0,5246** |

**Leitura — achado mais importante da bateria**: o modelo generaliza prospectivamente **bem**
pra 2024 e pra 2025 (AUC 0,63 e 0,67 — iguais ou melhores que o CV embaralhado). O colapso é
**específico de 2026**, não uma falha geral de generalização temporal. Isso muda a pergunta de
"o modelo generaliza no tempo?" (resposta: geralmente sim) pra "o que é diferente
especificamente em 2026?".

### 4. Holdout casado por estação

Treino = Jan-Jul de 2023, 2024, 2025 (mesma janela de mês do teste); teste = Jan-Jul 2026.

| desenho | n treino | n teste | AUC |
|---|---:|---:|---:|
| casado por estação (Jan-Jul → Jan-Jul) | 747 | 279 | 0,5219 |
| holdout original (ano-cheio → Jan-Jul parcial, SUSC-20O) | 1179 | 279 | 0,5246 |

**Leitura**: controlar sazonalidade não recupera desempenho (0,5219 ≈ 0,5246). **Descarta a
hipótese de sazonalidade não-comparável** como explicação — não é "2026 é só Jan-Jul e por
isso é diferente".

### 5. Estabilidade de coeficiente Firth por ano

| feature | 2023 (p) | 2024 (p) | 2025 (p) | 2026 (p) |
|---|---|---|---|---|
| `rain_decay_index_api_chirps` | +0,588 (0,0002) | +0,435 (0,0022) | +0,631 (<0,0001) | **+0,032 (0,795)** |
| `rain_peak_residual_orthogonalized` | −0,412 (0,0033) | −0,511 (<0,0001) | −0,142 (0,217) | **+0,015 (0,903)** |
| `hand_m_dinf` | −0,316 (0,031) | +0,014 (0,901) | −0,063 (0,614) | −0,173 (0,194) |
| `twi_dinf` | −0,055 (0,741) | +0,363 (0,038) | +0,077 (0,591) | −0,049 (0,731) |
| `slope_deg` | +0,204 (0,201) | −0,143 (0,320) | +0,017 (0,891) | −0,022 (0,875) |

**Leitura — mecanismo direto do colapso**: as duas features de chuva são o sinal mais forte e
consistente em 2023-2025 (mesmo sinal, quase sempre p<0,01) e **ficam completamente nulas em
2026** (coeficiente perto de zero, p>0,79 nas duas). É exatamente o padrão que explica os
diagnósticos 2 e 3: a relação chuva↔queixa que sustentava o modelo nos 3 anos anteriores não
aparece nos dados de 2026. Terreno nunca foi um sinal forte e consistente em nenhum ano (só
`hand_m_dinf` em 2023 e `twi_dinf` em 2024, isolados) — não é isso que muda em 2026, é a
ausência do sinal de chuva que normalmente carregava o modelo.

### 6. Estabilidade de composição/metadado por ano

| year | n | strong (%, pos) | Drenagem (%, pos) | rain_status OK | rain_n_days_found |
|---:|---:|---:|---:|---|---:|
| 2023 | 266 | 95,0% | 97,5% | sim | 14,0 |
| 2024 | 351 | 89,9% | 86,2% | sim | 14,0 |
| 2025 | 572 | 90,1% | 82,1% | sim | 14,0 |
| 2026 | 282 | 90,8% | 93,1% | sim | 14,0 |

**Leitura**: nenhuma métrica de qualidade/composição do dado (confiança de geocodificação,
categoria de queixa, cobertura de janela de chuva, status do dado de chuva) é visivelmente
diferente em 2026. **Descarta deriva administrativa/de qualidade de dado como explicação
visível** nesses termos — 2026 parece metodologicamente comparável aos outros anos.

## Síntese

Depois de 6 diagnósticos (mais o SUSC-20P antes desta rodada), descartamos:

- Vazamento espacial (SUSC-20P: spatial block CV = CV embaralhado)
- Sazonalidade não-comparável (diagnóstico 4: casar a estação não recupera AUC)
- Deriva administrativa/qualidade de dado visível em metadado (diagnóstico 6: composição
  estável ano a ano)
- Que o colapso seja ruído de amostra pequena (diagnóstico 1: a queda é real, fora do IC)
- Que seja um problema geral de generalização temporal do modelo (diagnóstico 3: 2024 e 2025
  generalizam bem prospectivamente)

O que fica, com evidência direta: **2026 especificamente tem uma relação chuva↔queixa
diferente dos 3 anos anteriores** (diagnóstico 5) — as duas features de chuva, que carregavam
o modelo em 2023-2025, ficam nulas nesse ano. Isso é consistente com o diagnóstico 2 (chuva-só
cai abaixo do acaso) e com o diagnóstico 3 (só 2026 falha).

Duas leituras não-excludentes ficam em aberto, **nenhuma testável com o dado atual sem nova
aquisição**:

1. **Anomalia hidrometeorológica real em 2026** — se o regime de chuva de 2026 for atípico
   (ex.: padrão diferente de intensidade/decaimento), a relação estatística aprendida nos anos
   anteriores pode genuinamente não se aplicar — não seria falha do modelo, seria mudança real
   do fenômeno. Testável com índice de anomalia climática (ENSO/ONI) ou comparação de chuva
   total anual real (fora do escopo desta rodada, exigiria puxar mais uma série externa).
   Ver [nota] abaixo.
2. **Censura/incompletude do ano parcial** — mesmo com metadado de qualidade estável (diagnóstico
   6), é possível que queixas de 2026 ainda em processamento no SIAC 156 tenham características
   sistematicamente diferentes das já consolidadas (não capturado pelos campos de metadado
   auditados). Sem forma de testar isso sem re-consultar a fonte depois que mais tempo passar.

**Nota**: a comparação direta com uma série de chuva total anual independente (não condicionada
a queixa) e/ou índice ENSO ficou fora desta rodada por exigir puxar uma série nova — está
documentada como próximo passo explícito, não decidida nem executada aqui (ver Limitações).

**Não é motivo pra trocar de método, reponderar feature ou re-otimizar** — é um resultado real
que estreita a pergunta de "o modelo não generaliza" (imprecisa) pra "a relação chuva-queixa
observada em 2026 é diferente da observada em 2023-2025, por razão física ou de completude de
dado ainda não determinada" (precisa e testável).

## Limitações

1. Diagnóstico 1 usa reamostragem bootstrap do teste, não do treino — mede incerteza da
   avaliação, não incerteza do ajuste do modelo.
2. Walk-forward (diagnóstico 3) só tem 3 cortes possíveis (4 anos de dado) — poder estatístico
   limitado pra generalizar "2026 é a exceção" com alta confiança.
3. Diagnóstico 5 ajusta Firth por ano com EPV marginal (~21, piso 20) — coeficientes por ano
   têm mais incerteza que o modelo agregado.
4. As duas hipóteses da síntese não foram testadas — nenhuma decisão de redesenho foi tomada
   aqui.
5. Redesenho de amostragem negativa (pareamento temporal positivo-negativo, PU-learning) é uma
   vertente de literatura real e documentada separadamente como opção de metodologia maior,
   não diagnóstico — não executado nesta rodada, exige aprovação explícita antes de rodar.

## Arquivos

- `scripts/pipeline_v20q_exhaustive_temporal_diagnostics_curitiba.py`
- `results/v20q_all_reports.json`, `v20q_ablacao_terreno_vs_chuva.csv`,
  `v20q_walk_forward_cutoffs.csv`, `v20q_coef_stability_by_year.csv`,
  `v20q_metadata_composition_stability_by_year.csv`
- `tests/test_susc_20q_exhaustive_temporal_diagnostics.py` (7 testes)

## Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20q_exhaustive_temporal_diagnostics_curitiba.py
```
