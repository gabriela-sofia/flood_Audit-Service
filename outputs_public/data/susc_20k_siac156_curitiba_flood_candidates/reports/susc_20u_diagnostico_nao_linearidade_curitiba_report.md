# SUSC-20U — Diagnóstico de não-linearidade (GBM raso), Curitiba

**Status**: EXECUTADO. **Primeiro resultado positivo desta linha de investigação inteira**
(SUSC-20P a 20U). Diagnóstico, não proposta de rota primária — decisão de aprofundar ou não
fica com a orientação humana.

## Motivação

Toda vertente linear testada até aqui (Firth supervisionado, PU bagging, peso de recência,
feature de chuva alternativa) fica presa perto de AUC≈0,52 no holdout temporal de 2026. Este
diagnóstico pergunta uma coisa distinta de tudo que já foi testado: será que a relação entre
as features físico-causais e o rótulo é **não-linear** (limiares, interação entre variáveis),
e um modelo linear simplesmente não consegue capturá-la — independente de qual seja a causa da
mudança ano a ano já diagnosticada?

**Importante**: isso não é uma alternativa às explicações já descartadas (ENOS, app,
amostragem negativa) — é uma pergunta ortogonal, sobre a FORMA da relação, não sobre a CAUSA da
mudança.

## Método

Gradient boosting raso (`GradientBoostingClassifier`, scikit-learn) sobre as MESMAS 5 features
causais de sempre — nenhuma feature nova, nenhuma orbital/proxy, `elevation_m` continua fora.
Nenhuma busca de hiperparâmetro pra escolher o "melhor" — testado com configuração default
razoável e depois auditado com grade de sensibilidade (27 combinações) pra checar se o
resultado é frágil ou robusto.

## Resultado

| validação | baseline linear (Firth/logistic) | GBM raso |
|---|---:|---:|
| holdout temporal (2026 nunca visto) | 0,5246 | **0,5888** |
| IC 95% (bootstrap, 2000 reamostras) | [0,4535; 0,5959] — inclui 0,5 | **[0,5188; 0,6589] — exclui 0,5** |
| spatial block CV (bairro, 5 folds) | 0,6442 | **0,6664** |

O IC do GBM **exclui 0,5 (acaso)** — diferente do baseline linear, cujo IC incluía. Isso quer
dizer: o GBM tem sinal preditivo genuíno e estatisticamente distinguível de aleatório no
holdout temporal, o que o modelo linear não tinha.

### Robustez a hiperparâmetro (27 combinações: profundidade 1-3 × nº árvores 50-200 × taxa de
aprendizado 0,02-0,10)

| | valor |
|---|---:|
| AUC mínimo na grade | 0,5467 |
| AUC máximo na grade | 0,6287 |
| AUC mediano na grade | 0,5859 |
| fração de configurações com AUC>0,55 | 92,6% (25 de 27) |

**Não é um resultado frágil de uma configuração sortuda** — a esmagadora maioria da grade fica
acima de 0,55, com mediana 0,586, muito acima do baseline linear (0,5246) em praticamente toda
a grade testada.

### Importância de feature (média, configuração de referência)

| feature | importância |
|---|---:|
| `rain_decay_index_api_chirps` | 0,431 |
| `rain_peak_residual_orthogonalized` | 0,239 |
| `hand_m_dinf` | 0,189 |
| `slope_deg` | 0,096 |
| `twi_dinf` | 0,045 |

Padrão fisicamente coerente com o resto do projeto: chuva domina (67% combinado), HAND (a
feature de terreno causalmente mais central) é a segunda mais importante, `twi_dinf` a menos
— não é um padrão aleatório/sem sentido físico.

## Leitura

Isso reabre uma pergunta que nenhum dos diagnósticos anteriores tinha feito: a explicação do
colapso pode não ser (só) sobre O QUE muda em 2026, mas sobre COMO a relação chuva-terreno-
queixa é estruturada — se for não-linear/com limiares (ex.: chuva só importa acima de um
certo decaimento, ou a interação chuva×HAND é o que realmente importa, não os efeitos
principais separados), um modelo linear nunca vai capturar isso bem, em nenhum ano, e a
diferença entre 2023-2025 e 2026 pode ser sobre ONDE nos dados aquela porção não-linear do
espaço de features caiu, não sobre a física mudar.

**Isto não resolve o problema — mas é o primeiro resultado desta bateria inteira (SUSC-20P a
20U) que aponta um caminho ainda não esgotado.**

## Limitações e ressalvas importantes (não escondidas)

1. **GBM não respeita o mesmo piso de EPV que rege a rota Firth.** Uma árvore rasa já tem mais
   parâmetros efetivos que 5 coeficientes lineares — o piso EPV=20 que protege contra
   overfitting em regressão logística não tem equivalente direto aqui. Isso é uma limitação
   real, não um detalhe.
2. **Interpretabilidade**: `feature_importances_` do sklearn não são coeficientes causais —
   não dão sinal, não dão intervalo de confiança, não respondem "aumentar X em 1 unidade muda
   o risco em quanto". Isso conflita diretamente com a prioridade do projeto por
   interpretabilidade e coerência científica.
3. Não testado se a melhora persiste com repetição de holdout temporal (só há 1 corte
   2023-25→2026 disponível, mesma limitação do SUSC-20O).
4. Não é uma proposta de rota primária — é diagnóstico. Promover GBM a candidato de produção
   exigiria decisão humana explícita, dado o conflito direto com a prioridade declarada do
   projeto por interpretabilidade sobre performance.
5. Não decompõe QUAL não-linearidade/interação especificamente está sendo capturada (ex.: via
   partial dependence ou SHAP) — ficaria pra uma rodada de aprofundamento, se decidido.

## Arquivos

- `scripts/pipeline_v20u_nonlinear_diagnostic_curitiba.py`
- `results/v20u_all_reports.json`, `v20u_gbm_spatial_block_cv.csv`,
  `v20u_gbm_hyperparam_sensitivity.csv`
- `tests/test_susc_20u_nonlinear_diagnostic.py` (4 testes)

## Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20u_nonlinear_diagnostic_curitiba.py
```
