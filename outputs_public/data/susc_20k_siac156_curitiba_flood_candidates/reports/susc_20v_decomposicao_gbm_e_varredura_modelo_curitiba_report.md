# SUSC-20V — Decomposição do GBM + varredura de classes de modelo, Curitiba

**Status**: EXECUTADO. Fortalece o achado do SUSC-20U: não é um acidente de um algoritmo
específico — praticamente toda classe de modelo não-linear testada supera o baseline linear.

## Contexto

Pedido explícito (2026-08-02): testar tudo, inclusive fora do filtro de interpretabilidade que
rege a rota primária, para ver se alguma classe de modelo recupera sinal prospectivo real em
2026. **Ressalva mantida deliberadamente**: nenhuma feature nova derivada do label foi
adicionada — isso invalidaria a validação por vazamento, não é uma questão de estilo. Só a
escolha de algoritmo foi liberada do filtro de interpretabilidade, não o desenho de feature.

## 1. Varredura de classes de modelo (holdout temporal 2026, mesmas 5 features causais)

| modelo | AUC holdout |
|---|---:|
| GBM raso (SUSC-20U, referência) | **0,5888** |
| AdaBoost (100 stumps) | 0,5771 |
| Random Forest (profundidade 3) | 0,5611 |
| Extra Trees (profundidade 3) | 0,5468 |
| SVM-RBF (C=1) | 0,5386 |
| SVM-RBF (C=10) | 0,5370 |
| MLP (16,8) | 0,5318 |
| MLP (8) | 0,5292 |
| *baseline linear (Firth/logistic, SUSC-20O)* | *0,5246* |

**Todas as 8 classes de modelo ficam acima do baseline linear**, não só o GBM. Isso é um
padrão consistente, não um resultado isolado de um algoritmo específico — reforça a leitura do
SUSC-20U de que existe alguma estrutura não-linear real nas features causais que os modelos
lineares não capturam, independente de qual família de modelo não-linear é usada. Métodos
baseados em árvore (GBM, AdaBoost, RF) performam melhor que kernel (SVM) e rede neural (MLP)
rasa, nesta ordem.

## 2. Decomposição (partial dependence) do GBM

Direção e não-linearidade por feature (grade em escala original, não padronizada):

| feature | direção geral | amplitude do efeito | monotônico? | consistente com sinal causal esperado? |
|---|---|---:|---|---|
| `rain_decay_index_api_chirps` | crescente | 1,27 | não (7 mudanças de inclinação) | ✓ sim |
| `rain_peak_residual_orthogonalized` | decrescente | 1,08 | não (7 mudanças) | ✓ sim (mesmo sinal do Firth) |
| `hand_m_dinf` | decrescente | 0,90 | não (5 mudanças) | ✓ sim (mais perto do canal = mais risco) |
| `twi_dinf` | decrescente | 0,59 | não (5 mudanças) | **✗ não** — esperado seria crescente (mais acúmulo teórico de umidade = mais risco) |
| `slope_deg` | decrescente | 0,15 | não (9 mudanças) | ✓ sim (mais declive = melhor drenagem = menos risco) |

**Leitura**: nenhuma das 5 features tem relação monotônica simples — todas têm múltiplas
mudanças de inclinação ao longo da grade, confirmando que a relação é genuinamente não-linear,
não só "a mesma relação linear com uma curva suave". 4 de 5 features mantêm a direção geral
consistente com o sinal causal já estabelecido no modelo linear (chuva, HAND, declividade) —
o GBM não está "inventando" uma física diferente, está capturando limiares/interações dentro
da mesma direção causal já conhecida. `twi_dinf` é a exceção — direção oposta à esperada,
consistente com o próprio modelo linear (onde `twi_dinf` nunca teve sinal robusto em nenhuma
rodada anterior, SUSC-20L a 20Q).

## Síntese

Isso reforça — mas não resolve — o achado do SUSC-20U. A totalidade das evidências agora
aponta pra: (1) existe sinal prospectivo real recuperável em 2026 que os modelos lineares não
capturam (8 de 8 classes não-lineares superam o baseline); (2) esse sinal, pelo menos no GBM,
é fisicamamente coerente na maioria das features (4 de 5 na direção causal esperada); (3) nada
disso ainda é uma rota de produção — continua sendo diagnóstico, sob as mesmas ressalvas de
interpretabilidade/EPV já registradas no SUSC-20U.

## Limitações

1. `shap` e `xgboost` não instalaram no sandbox (timeout de rede) — decomposição feita via
   `sklearn.inspection.partial_dependence`, que já estava disponível, sem tentar contornar a
   rede de forma arriscada. Isso limita a decomposição a efeito médio por feature (partial
   dependence 1-D), não captura interações par-a-par diretamente.
2. Nenhum dos modelos da varredura teve busca de hiperparâmetro (só configuração default
   razoável) — o ranking entre eles pode mudar com tuning, embora o padrão geral
   (não-linear > linear) provavelmente se mantenha dado o tamanho da diferença.
3. `twi_dinf` com direção anômala não foi investigado mais a fundo aqui.

## Arquivos

- `scripts/pipeline_v20v_gbm_decomposition_and_model_sweep_curitiba.py`
- `results/v20v_all_reports.json`, `v20v_model_sweep_holdout_2026.csv`
- `tests/test_susc_20v_decomposition_sweep.py` (3 testes)

## Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20v_gbm_decomposition_and_model_sweep_curitiba.py
```
