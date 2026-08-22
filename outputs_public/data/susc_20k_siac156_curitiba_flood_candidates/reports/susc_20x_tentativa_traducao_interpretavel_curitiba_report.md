# SUSC-20X — Tentativa de tradução interpretável da não-linearidade, Curitiba

**Status**: EXECUTADO, resultado negativo. Tentativa honesta de resolver a tensão
interpretabilidade×performance flagueada no SUSC-20U/20V — não conseguiu.

## Motivação

O SUSC-20U/20V/20W estabeleceu que existe não-linearidade real e generalizável nas 5 features
causais (GBM e outras 7 classes de modelo superam o linear em quase todo corte temporal). A
pergunta natural: dá pra capturar esse sinal com um termo explícito e interpretável dentro do
próprio modelo linear (Firth/logistic), preservando coeficiente, p-valor e IC — em vez de
aceitar um modelo caixa-preta?

## Método

Dois tipos de termo, ambos lidos diretamente da partial dependence do GBM (SUSC-20V/20W), não
escolhidos por busca:

1. **Interação produto** (par de features multiplicadas) — forma clássica de capturar
   interação num GLM.
2. **Indicador de limiar binário**, no ponto exato onde a partial dependence do GBM mostra o
   salto mais acentuado: `hand_m_dinf<4m` (PD cai de 0,84 pra −0,05 nesse intervalo),
   `rain_decay_index_api_chirps>20` (PD sobe de −0,62 pra +0,44), `rain_peak_residual<−4` (PD
   cai de +0,43 pra +0,01).

## Resultado

| configuração | nº features | AUC holdout 2026 |
|---|---:|---:|
| base (5 features, linear) | 5 | 0,5246 |
| + `rain_decay × hand_m_dinf` | 6 | 0,5246 |
| + `rain_peak × hand_m_dinf` | 6 | 0,5200 |
| + `rain_decay × rain_peak` | 6 | 0,5295 |
| + as 3 interações produto | 8 | 0,5252 |
| + limiar `hand_m_dinf<4` | 6 | 0,5262 |
| + limiar `rain_decay>20` | 6 | 0,5279 |
| + limiar `rain_peak<−4` | 6 | 0,5184 |
| + os 3 limiares | 8 | 0,5224 |
| *referência: GBM (SUSC-20U)* | *5* | *0,5888* |

**Nenhuma configuração chega perto do GBM.** A melhor (interação `rain_decay × rain_peak`)
fica em 0,5295 — uma melhora de 0,005 sobre o baseline, dentro do ruído, muito longe dos 0,588
do GBM.

## Leitura

A tentativa de "traduzir" a não-linearidade pra um termo simples e interpretável **falhou**.
Isso é informativo: sugere que o que o GBM captura não é uma interação produto suave nem um
limiar aditivo simples — é provavelmente uma estrutura mais complexa (interação de ordem mais
alta entre 3+ features, ou splits diferentes em subregiões diferentes do espaço de features,
o que uma árvore captura naturalmente e um GLM com poucos termos não consegue replicar sem
uma especificação muito mais elaborada).

**Isso não invalida o achado do SUSC-20U/20V/20W** — só estabelece que resolver a tensão
interpretabilidade×performance não é trivial com os recursos tentados aqui. A escolha, por
ora, continua sendo a documentada desde o SUSC-20U: manter a rota primária linear/Firth
(interpretável, causal, sob EPV) e tratar o achado não-linear como diagnóstico registrado, não
como substituto.

## Limitações

1. Só testadas interações par-a-par entre as 3 features mais importantes (`rain_decay`,
   `rain_peak`, `hand_m_dinf`) — não testadas interações de 3ª ordem nem envolvendo
   `slope_deg`/`twi_dinf`.
2. Limiares lidos visualmente do grid de partial dependence, não otimizados (ex.: via
   change-point detection formal) — um limiar mais preciso poderia performar diferente.
3. Não tentado GAM (generalized additive model) com termos spline por feature, que capturaria
   a forma da partial dependence mais fielmente que um único limiar binário — ficaria pra uma
   rodada futura, se decidido.

## Arquivos

- `scripts/pipeline_v20x_interaction_translation_attempt_curitiba.py`
- `results/v20x_all_reports.json`, `v20x_interaction_translation_attempts.csv`
- `tests/test_susc_20x_interaction_translation.py` (4 testes)

## Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20x_interaction_translation_attempt_curitiba.py
```
