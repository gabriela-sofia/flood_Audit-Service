# SUSC-20R — Correlação ONI/ENOS real vs. colapso de AUC prospectivo, Curitiba

**Status**: EXECUTADO, resultado negativo (hipótese não sustentada pelo dado real).

## Motivação

O SUSC-20Q deixou em aberto a hipótese de anomalia hidrometeorológica real explicando o
colapso específico de 2026. A revisão de literatura de 2026-08-02 achou a Nota Técnica SIMEPAR
(fonte primária, 2026-06-11) confirmando El Niño se formando no Paraná a partir do inverno
2026 — mas notou, sem testar, que a linha do tempo bruta do ONI não parecia dar um corte
limpo. Este script formaliza o teste.

## Método

Índice ONI real (Oceanic Niño Index, médias móveis de 3 meses de anomalia de TSM na região
Niño 3.4), fonte: [Golden Gate Weather Services, "El Niño and La Niña Years and Intensities"
(Jan Null, CCM)](https://ggweather.com/enso/oni.htm), baseado no [ONI v5 do NOAA/CPC](https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php).
Consultado 2026-08-02. Extraída a média dos 5 meses Jan-Jul (JFM, FMA, MAM, AMJ, MJJ) por ano,
mesma janela do holdout casado por estação do SUSC-20Q, comparada com o AUC de teste do
walk-forward (SUSC-20Q diagnóstico 3) e o coeficiente de `rain_decay_index_api_chirps` por ano
(SUSC-20Q diagnóstico 5).

**Limitação documentada, não contornada**: a fonte consultada está "atualizada até dezembro de
2025" — **não existe valor numérico publicado de ONI pra Jan-Jul de 2026**. Não foi estimado
nem inventado; a comparação pra 2026 fica com essa célula vazia por desenho.

## Resultado

| ano | ONI médio Jan-Jul | AUC teste (walk-forward) | coef. `rain_decay` | p |
|---:|---:|---:|---:|---:|
| 2023 | +0,274 (neutro→El Niño formando) | — (só treino) | 0,588 | 0,0002 |
| 2024 | **+0,882 (El Niño forte)** | **0,6282** | 0,435 | 0,0022 |
| 2025 | −0,068 (neutro) | **0,6652** | 0,631 | <0,0001 |
| 2026 | **não publicado** | 0,5246 | 0,032 | 0,7948 |

## Leitura

**A hipótese ENOS não é sustentada pelos dados reais disponíveis.** 2024 — o ano com a maior
anomalia ENOS real dos que têm dado publicado (El Niño forte, ONI médio +0,88) — generalizou
bem (AUC teste=0,6282). 2025 — ano ENOS-neutro (ONI médio −0,07) — generalizou ainda melhor
(AUC teste=0,6652). Se anomalia ENOS explicasse falha de generalização, esperaríamos o padrão
oposto (2024 falhando, não 2025 sendo o melhor). Isso vai na direção contrária da hipótese
levantada na revisão de literatura.

Não é possível fechar a pergunta sobre 2026 especificamente por falta de dado publicado — mas
o teste com o dado que existe (2023-2025) já é suficiente pra não seguir tratando "anomalia
ENOS" como explicação provável sem evidência adicional. Consistente com a ressalva de
honestidade já registrada no PLANO_ACAO antes de rodar este script.

## Limitações

1. n=2 anos com ONI publicado E AUC de teste (2024, 2025) — não dá pra rodar teste de
   correlação formal (poder estatístico zero com n=2); a leitura é descritiva/direcional, não
   inferencial.
2. 2026 fica sem comparação numérica — a Nota Técnica SIMEPAR (qualitativa) ainda aponta El
   Niño se formando a partir de jun/2026, o que tecnicamente cairia fora da maior parte da
   janela Jan-Jul do teste (a maior parte do semestre foi neutra/transicional, segundo a
   mesma fonte) — mas isso é leitura qualitativa da notícia, não do índice numérico.
3. Não exclui outras explicações hidrometeorológicas não capturadas por ONI (ex.: eventos
   convectivos locais de mesoescala, que a própria Nota SIMEPAR menciona como mecanismo
   distinto de "El Niño aumenta a chuva média").

## Síntese pra decisão

Com este resultado, a vertente "testar ONI real" está concluída — resultado negativo, honesto,
registrado. Consistente com a decisão já tomada de seguir em sequência pra a segunda vertente
(redesenho de amostragem negativa via correção de sub-reporte/PU-learning), que ataca o
confundimento estrutural na raiz em vez de depender de uma explicação climática externa que não
se sustentou.

## Arquivos

- `scripts/pipeline_v20r_oni_enso_correlation_curitiba.py`
- `results/v20r_oni_enso_correlation.json`
- `tests/test_susc_20r_oni_enso_correlation.py` (5 testes)

## Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20r_oni_enso_correlation_curitiba.py
```
