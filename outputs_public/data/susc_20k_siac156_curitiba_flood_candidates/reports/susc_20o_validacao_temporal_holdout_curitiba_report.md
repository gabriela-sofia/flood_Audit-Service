# SUSC-20O — Validação temporal holdout, Curitiba

**Status**: EXECUTADO. Treino 2023–2025 (1179 unidades, 315 neg/864 pos), teste 2026-parcial
(279 unidades, 108 neg/171 pos). Rota primária do SUSC-20N (5 features, `elevation_m` fora —
decisão causal mantida, não revisitada aqui).

## Motivação

O LOO-CV e o 5-fold repetido usados no SUSC-20M/20N embaralham unidades no tempo: um ponto de
2023 pode treinar o modelo que classifica um ponto de 2025, e vice-versa. Isso superestima
capacidade preditiva real se houver qualquer deriva temporal (composição do SIAC 156 muda de
ano pra ano, cobertura de bairro muda, distribuição da chuva antecedente muda). Holdout temporal
é o teste mais honesto disponível com os dados que existem: treinar só no passado, avaliar num
ano nunca visto.

## Resultado

**AUC holdout temporal (2026 nunca visto): 0,5246.**

Contra LOO-AUC 0,6459 e 5-fold-repetido 0,6440 do SUSC-20N (mesma rota, mesmas features, CV
embaralhada). A queda é grande — o modelo é quase indistinguível de aleatório fora da janela
temporal em que foi treinado, apesar de "funcionar" na validação cruzada padrão.

| validação | AUC |
|---|---:|
| LOO-CV (embaralhada, SUSC-20N) | 0,6459 |
| 5-fold repetido 50× (embaralhada, SUSC-20N) | 0,6440 |
| **holdout temporal (2026 nunca visto, SUSC-20O)** | **0,5246** |

### Coeficientes Firth ajustados só no treino (2023–2025)

| feature | coef. (treino only) | p | IC 95% | sinal | coef. (dataset completo, 20N) |
|---|---:|---:|---|---|---:|
| `rain_decay_index_api_chirps` | **+0,5356** | **<0,0001** | [0,374; 0,706] | ✓ | +0,4278 |
| `rain_peak_residual_orthogonalized` | **−0,3898** | **<0,0001** | [−0,525; −0,257] | ✗ | −0,3276 |
| `hand_m_dinf` | −0,1428 | 0,0582 | [−0,290; 0,005] | ✓ | −0,1401 (p=0,032, sig.) |
| `twi_dinf` | +0,1462 | 0,1296 | [−0,041; 0,352] | ✓ | +0,0761 |
| `slope_deg` | +0,0236 | 0,7734 | [−0,136; 0,189] | ✗ | −0,0026 |

`hand_m_dinf` perde a significância (p=0,058, vs. p=0,032 no dataset completo) quando ajustado
só nos 3 anos de treino — consistente com o achado geral de N do SUSC-20N: menos dado, menos
poder estatístico. Coeficientes de chuva mantêm magnitude e direção parecidas.

## Leitura

A queda de AUC 0,64→0,52 quando o teste é genuinamente prospectivo (não embaralhado) é o
resultado mais direto já obtido de que o modelo primário de Curitiba **não generaliza bem no
tempo**. Duas leituras não-excludentes, nenhuma testável com os dados atuais:

1. **Composição do SIAC 156 muda por ano** — 2025 tem desproporcionalmente mais positivos
   (569 brutos, ~464 unidades) que os outros anos (ver funil do SUSC-20K2); se essa mudança
   correlaciona com algo não-físico (ex.: mudança de política de atendimento, campanha de
   conscientização que aumenta reclamação em bairros específicos), o modelo aprende esse
   padrão administrativo, não o físico-hidrológico.
2. **2026 é ano parcial (jan–jul)**, então a distribuição sazonal de chuva do teste não é
   comparável à distribuição anual completa do treino — vieses sazonais (época seca vs.
   chuvosa) podem estar desalinhados entre treino e teste por desenho, não por falha do modelo.

Nenhuma das duas foi testada aqui — ficam como hipótese, não conclusão. **Não é motivo pra
trocar de método ou re-otimizar features**: é resultado real, documentado, que deve pesar na
leitura de qualquer AUC de validação cruzada reportado daqui pra frente — o número de 0,64 do
SUSC-20N mede separabilidade dentro da distribuição observada, não capacidade preditiva
prospectiva.

## Limitações

1. Um único corte temporal (não há anos suficientes pra repetir o holdout com múltiplos cortes
   tipo walk-forward).
2. Teste é ano parcial (7 meses), não comparável em sazonalidade ao treino completo.
3. `elevation_m` não foi testada aqui — decisão causal do SUSC-20N mantida sem revisão.
4. Não distingue as duas hipóteses da seção "Leitura" — nenhuma foi investigada.

## Arquivos

- `scripts/pipeline_v20o_temporal_holdout_curitiba.py`
- `results/v20o_all_reports.json`, `v20o_firth_train_only_coefs.csv`, `v20o_univariate_train_only.csv`
- `tests/test_susc_20o_temporal_holdout.py` (6 testes)

## Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20o_temporal_holdout_curitiba.py
```
