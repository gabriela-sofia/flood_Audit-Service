# SUSC-20H2 — Processamento real: Petrópolis, evento 05/04/2025

**Status**: RESULTADO REAL, NÃO ADJUDICADO. Bug de performance corrigido no pipeline; o par de
cenas testado não reproduz o padrão limpo de Valparaíso/2022 — nem claramente ruído, nem
claramente candidato único. Nenhuma decisão forçada.

## Bug real corrigido em `detect_water_candidates.py`

O laço de agregação por cluster fazia `np.where(labels == lab)` uma vez por componente
conectado, varrendo a imagem inteira a cada iteração — O(n_clusters × tamanho_da_imagem). Nos
testes sintéticos (poucos clusters) isso não aparecia; com dado real (4.335 componentes brutos
na primeira cena) o processo trava indefinidamente. Substituído por agregação vetorizada
(`scipy.ndimage.sum/mean/center_of_mass`), que faz a mesma conta em O(imagem) uma vez, não por
cluster. Roda em ~5s agora. Os 18 testes existentes continuam passando.

## Cenas processadas

Antes: **2025-02-16** (cloudCover 1,15%, S2C). Depois: **2025-04-09** (28,24%, S2A) e
**2025-04-07** (42,93%, S2C) — as duas opções recomendadas pela varredura de nuvem real da
rodada anterior. 4 bandas (B03/B08/B11/B12), grade idêntica (1848×2500, EPSG:4326, 10m),
conferida antes de rodar.

## Resultado: confundimento de cena inteira, igual ao padrão que já tinha derrubado a tentativa de fev/2022

| Comparação | Pixels no gate físico | Pixels no consenso 2-de-3 | % da cena | Clusters ≥20px |
|---|---:|---:|---:|---:|
| 16/02 × 09/04 (cena inteira) | 184.516 | 158.515 | 3,43% | 627 |
| 16/02 × 07/04 (cena inteira) | 311.738 | 57.771 | 1,25% | 330 |

Os clusters não se concentram no canal do Rio Quitandinha nem no Centro Histórico — cobrem toda
a extensão da AOI (lat -22,55 a -22,30, lon -43,25 a -43,05, ou seja, o município inteiro).

**Repetido numa janela restrita ao núcleo urbano** (lon -43,21 a -43,14, lat -22,535 a -22,475 —
Centro Histórico, Quitandinha, Valparaíso, Alto da Serra), mesma técnica que isolou o sinal de
Valparaíso em 2022:

| Comparação (janela urbana) | % da janela | Clusters ≥20px |
|---|---:|---:|
| 16/02 × 09/04 | 3,74% | 62 |
| 16/02 × 07/04 | 0,76% | 22 |

Mesmo restrito, ainda 6× (09/04) e 1,3× (07/04) acima do baseline de ruído aleatório (~0,6%,
mesma referência usada na adjudicação de fev/2022), e ainda espalhado em múltiplos focos
separados, não um cluster único e concentrado.

## Interpretação honesta, sem forçar veredito

Duas explicações reais concorrem, nenhuma descartada:

1. **Confundimento sistemático entre as cenas** — 16/02 (S2C, verão) vs abril (S2A/S2C, outono):
   mesmo controlando estação, sensores diferentes e ~7-8 semanas de intervalo real podem gerar
   diferença radiométrica ampla sem relação com o evento.
2. **Sinal real, mas difuso, não pontual** — o evento de 05/04/2025 foi documentado como severo
   e amplo (300mm/24h, decreto de emergência, 128 desalojados, 1.802 afetados, múltiplos
   bairros). Solo saturado, lama e vegetação danificada numa área grande produzem exatamente a
   assinatura NIR/SWIR observada, sem ser "lâmina d'água parada" concentrada — diferente do
   padrão de Valparaíso/2022, que foi um transbordamento de canal específico.

Nenhuma das duas foi confirmada nem descartada nesta rodada. Não escolhi um cluster "vencedor"
arbitrariamente — isso seria repetir o erro já cometido e corrigido com Valparaíso (adjudicar
por sinal de índice antes de checar coerência física).

## O que não foi feito (deliberado)

Filtro topográfico da seção 2.1 (HAND/TWI/declividade) não foi aplicado aos clusters — com o
sinal já disperso e sem candidato único destacado, rodar o filtro em 600+ clusters não
economiza tempo real; primeiro precisa decidir se vale reprocessar com outra referência "antes"
ou aceitar que este evento não vai produzir um candidato pontual pelo método atual.

## Registros

`registries/v20h2_petropolis_05abr2025_cena_inteira_x_09abr.csv` (627 linhas) e
`registries/v20h2_petropolis_05abr2025_cena_inteira_x_07abr.csv` (330 linhas) — todos os
clusters brutos, `adjudicated=false` em todos.
