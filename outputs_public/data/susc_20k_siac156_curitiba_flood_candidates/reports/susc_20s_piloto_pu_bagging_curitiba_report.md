# SUSC-20S — Piloto de redesenho de amostragem negativa (PU bagging), Curitiba

**Status**: EXECUTADO, resultado negativo (redesenho não resolve o colapso de generalização
temporal). Piloto, não substitui a rota primária de produção.

## Motivação

Segunda vertente decidida na revisão de literatura de 2026-08-02 (a primeira, ONI/ENOS, foi
refutada pelo dado real no SUSC-20R). Ataca o confundimento estrutural já documentado desde o
SUSC-20K/20N: "negativo" em Curitiba é uma queixa não-hidrológica numa data arbitrária, não uma
confirmação de ausência de enchente — literatura chama isso de "case-control sampling with
contaminated controls" (PBLC, *Land* 2022, DOI 10.3390/land11111971).

## Método e escolha justificada

Método: **PU bagging** ([Mordelet & Vert, "A bagging SVM to learn from positive and unlabeled
examples", arXiv:1010.0772](https://arxiv.org/pdf/1010.0772)). Positivos ficam fixos em toda
bag (rótulo confiável); a cada bag, reamostra com reposição do pool "não-rotulado" (nosso atual
"negativo") um subconjunto do mesmo tamanho dos positivos; ajusta um classificador; agrega a
predição média no conjunto de teste ao longo de 300 bags.

**Por que PU bagging e não o modelo bayesiano espacial de sub-reporte** ([Agostini, Pierson &
Garg, AAAI-24](https://arxiv.org/abs/2312.11754)), que também estava na lista de referências:
o modelo bayesiano suaviza risco por correlação espacial entre queixas, criando uma superfície
contínua derivada da própria densidade de queixa — próximo demais de um "score/proxy derivado
do label", proibido pelas regras fixas do projeto ("nunca usar scores, thresholds, proxies ou
variáveis derivadas do label como features"; "o modelo não deve descobrir enchentes"). PU
bagging não inventa nenhuma variável nova nem suaviza no espaço — só muda como o classificador
trata o rótulo "negativo" (de "confirmado ausente" pra "não confirmado"), usando as MESMAS 5
features físico-causais de sempre.

## Resultado

| validação | baseline (supervisionado, já commitado) | PU bagging (este piloto) |
|---|---:|---:|
| holdout temporal (2026 nunca visto) | 0,5246 (SUSC-20O) | **0,5245** |
| spatial block CV (bairro, 5 folds) | 0,6442 (SUSC-20P) | **0,6443** |

Diferença: -0,0001 e +0,0001 respectivamente — **estatisticamente e praticamente nula**. PU
bagging não muda o comportamento do modelo em nenhuma das duas validações.

## Leitura

**O redesenho de amostragem negativa via PU bagging não resolve o colapso de generalização
temporal.** Isso é coerente com o diagnóstico do SUSC-20Q: o mecanismo do colapso não é rótulo
negativo contaminado (que PU bagging corrigiria) — é que a relação física chuva↔queixa que
sustentava o modelo em 2023-2025 simplesmente não aparece nos dados de 2026 (coeficientes de
chuva caem a zero, diagnóstico 5 do SUSC-20Q). PU bagging resolve viés de rótulo; não resolve
mudança de distribuição/relação entre feature e rótulo ao longo do tempo — são problemas
diferentes, e o nosso é o segundo, não o primeiro.

Com isso, as duas vertentes decididas na revisão de literatura (ONI/ENOS no SUSC-20R,
redesenho de amostragem negativa aqui) voltaram resultado negativo, honesto, real. Isso não é
um beco sem saída — é uma eliminação rigorosa que **fortalece a conclusão do SUSC-20Q**: o
colapso de 2026 é uma propriedade real e ainda não explicada desse período específico, não um
artefato de desenho metodológico corrigível com as ferramentas testadas até agora.

## Limitações

1. Piloto com 300 bags — não testado com mais bags (custo computacional maior, retorno
   marginal esperado baixo dado o resultado já nulo).
2. `LogisticRegression` padrão (não Firth) dentro de cada bag — substituição registrada
   explicitamente: o bagging já reduz a variância que Firth corrige em amostra pequena, mas
   não foi comparado bag-a-bag contra uma versão com Firth.
3. Tamanho de reamostra do pool não-rotulado = tamanho do conjunto de positivos (convenção
   padrão de Mordelet & Vert) — não testadas outras razões.
4. Recife não foi tocado — mudança de metodologia usada em produção em outra região exigiria
   decisão humana separada, fora de escopo aqui.
5. Não é uma promoção de rota — a rota primária de Curitiba continua sendo a Firth
   supervisionada de 5 features (SUSC-20N), sem mudança.

## Arquivos

- `scripts/pipeline_v20s_pu_bagging_pilot_curitiba.py`
- `results/v20s_all_reports.json`, `v20s_pu_bagging_spatial_block_cv.csv`
- `tests/test_susc_20s_pu_bagging_pilot.py` (5 testes)

## Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20s_pu_bagging_pilot_curitiba.py
```
