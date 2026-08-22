# SUSC-21A — GBM com restrição monotônica causal, Curitiba

**Status**: EXECUTADO. Resultado positivo-parcial: não supera o GBM irrestrito nem o melhor
GAM em AUC, mas é o único modelo não-linear testado nesta série com 100% de conformidade
causal garantida por construção.

## Motivação

Nova vertente, distinta da família GAM (SUSC-20Y/20Z, encerrada). Em vez de tentar *traduzir*
a não-linearidade de um GBM já ajustado sem restrição pra um termo interpretável, esta rodada
restringe o próprio ajuste do GBM a nunca contrariar o sinal causal já estabelecido (SUSC-20M/
20N): `HistGradientBoostingClassifier(monotonic_cst=...)` força a relação entre cada feature e
o log-odds de risco a ser monotônica na direção física conhecida — `slope_deg` e `hand_m_dinf`
decrescentes (mais declividade/mais distância da drenagem → menos risco), `twi_dinf`,
`rain_peak_residual_orthogonalized` e `rain_decay_index_api_chirps` crescentes.

Isso opera diretamente o mandato do projeto — "o modelo não deve descobrir enchentes; deve
refletir relações físicas conhecidas" — de um jeito que nem o linear nem o GBM irrestrito
conseguem sozinhos: o linear só permite 1 taxa constante por feature; o GBM irrestrito captura
qualquer forma, inclusive uma que contraria o sinal causal (caso documentado do `twi_dinf` no
SUSC-20V e de 2/5 features no SUSC-20Y). O GBM monotônico fica no meio: tão flexível quanto o
GBM em forma (limiares, platôs, taxas variáveis), mas **nunca pode inverter direção** — por
construção, nunca produziria uma curva anômala como a do `twi_dinf`.

## Método

`MONOTONIC_CST` reaproveita `EXPECTED_SIGN` (já definido em `pipeline_v20o`, mesma convenção
+1/-1) sem reescrever a direção causal em lugar nenhum. Comparação pareada monotônico vs.
irrestrito nos mesmos hiperparâmetros (isola o "custo" de impor a restrição). Holdout temporal,
spatial block CV (bairro), grade de sensibilidade a hiperparâmetro (27 combos: profundidade
1-3 × iterações 50/100/200 × taxa de aprendizado 0,02/0,05/0,1) e um *sanity check* — a
restrição realmente produz uma partial dependence monotônica? (testado tanto pela PD agregada
quanto por uma verificação direta na função de decisão bruta, linha fixa, 1 feature variando
por vez — necessário porque funções de árvore em degrau têm trechos planos que um detector
ingênuo de troca de sinal confundiria com violação).

## Resultado

| medida | monotônico | irrestrito (mesmos hiperparâmetros) |
|---|---:|---:|
| holdout 2026, config default | 0,5515 | 0,5739 |
| spatial block CV, média 5 folds | 0,6385 | 0,6590 |
| melhor da grade de 27 combos | 0,5561 | — (não regridado; ver SUSC-20U/20V) |

| referência | AUC |
|---|---:|
| baseline linear (5 features) | 0,5246 |
| GAM aditivo, melhor config (SUSC-20Y) | 0,575 |
| GBM irrestrito (SUSC-20U) | 0,5888 |
| **GBM monotônico, melhor config (este script)** | **0,5561** |

**Custo de impor monotonicidade** (mesma config default): 0,0224 de AUC. **Gap pro GBM
irrestrito** (melhor config de cada): 0,0327. O GBM monotônico fica levemente abaixo do melhor
GAM aditivo (0,5561 vs 0,575) e não recupera o patamar do GBM irrestrito — mas continua **acima
do baseline linear em 100% das 27 configurações testadas**.

### Sanity check — conformidade causal

| feature | direção esperada | direção obtida | bate? | monotônica? |
|---|---|---|---|---|
| `slope_deg` | decrescente | decrescente | sim | sim |
| `hand_m_dinf` | decrescente | decrescente | sim | sim |
| `twi_dinf` | crescente | crescente | sim | sim |
| `rain_peak_residual_orthogonalized` | crescente | crescente | sim | sim |
| `rain_decay_index_api_chirps` | crescente | crescente | sim | sim |

**5 de 5 — 100% de conformidade causal**, garantida por construção (não é coincidência da
rodada; a restrição impede matematicamente qualquer violação). Isso contrasta diretamente com o
GBM irrestrito (SUSC-20V: 4/5, `twi_dinf` anômalo) e o GAM aditivo (SUSC-20Y: 3/5, `twi_dinf` e
`rain_peak_residual_orthogonalized` anômalos).

## Leitura

Este é o primeiro modelo não-linear desta série que **nunca pode contrariar o conhecimento
causal já estabelecido**, ao custo de abrir mão de uma fração pequena de AUC (0,0224 a 0,0327,
dependendo da comparação) frente às versões irrestritas. Isso reformula a pergunta da série: em
vez de "como recuperar 100% do sinal do GBM com um modelo interpretável", que se mostrou muito
difícil (SUSC-20X/20Y/20Z), a pergunta mais produtiva pode ser "qual o melhor modelo não-linear
que **nunca viola física conhecida**" — e a resposta, com as ferramentas testadas, é o GBM
monotônico, não o GBM irrestrito nem o GAM aditivo (ambos toleram pelo menos 1 feature em
direção anômala).

Isso não resolve a lacuna original desta investigação (o colapso de AUC em 2026 documentado
desde o SUSC-20O) — nenhum modelo testado nesta série inteira (SUSC-20P a 21A) chega perto de
recuperar o AUC de 2024/2025 (0,63-0,68) no corte de 2026 (melhor resultado nesta rodada:
0,5561). Mas estabelece um candidato mais defensável cientificamente que o GBM irrestrito, caso
a decisão futura seja usar algum modelo não-linear em produção: um GBM monotônico é flexível o
suficiente pra capturar limiares e platôs reais, e rígido o suficiente pra nunca "inventar"
enchente onde a física diz que não deveria haver risco crescente.

## Limitações

1. `class_weight="balanced"` usado em ambas as versões (monotônica e irrestrita) — não testado
   sem balanceamento, que poderia mudar o custo relativo da restrição.
2. A restrição monotônica do sklearn se aplica por feature isoladamente; não impede interação
   entre features anômala (ex.: uma combinação especifica de `hand_m_dinf` alto E
   `rain_decay_index_api_chirps` baixo ainda pode produzir comportamento não intuitivo, mesmo
   que cada eixo isolado seja monotônico).
3. Grade de 27 combos não é exaustiva (mesmos limites usados no SUSC-20U/20V pra comparação
   direta); um espaço de busca maior poderia melhorar levemente o resultado.
4. Não testado com Firth/EPV formal (é diagnóstico de árvore, fora da rota causal primária,
   mesma ressalva de todos os diagnósticos GBM/GAM desta série).

## Arquivos

- `scripts/pipeline_v21a_monotonic_constrained_curitiba.py`
- `results/v21a_hyperparam_grid_monotonic.csv`, `v21a_monotonic_pd_check.csv`,
  `v21a_all_reports.json`
- `tests/test_susc_21a_monotonic_constrained.py` (6 testes)

## Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v21a_monotonic_constrained_curitiba.py
```
