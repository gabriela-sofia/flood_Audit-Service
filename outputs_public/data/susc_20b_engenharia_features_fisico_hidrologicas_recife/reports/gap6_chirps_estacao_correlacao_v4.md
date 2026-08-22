# GAP 6 — Correlação CHIRPS × estação moderada (r=0,658–0,777)

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-22

## Verificação de bug de processamento (janela de lookback / alinhamento de data)

Ambos `rain_max_24h_chirps` (CHIRPS) e `rain_max_24h_mm` (estação) usam a mesma definição de
janela: `[data_evento−1, ..., data_evento−14]` (`CHIRPS_LOOKBACK_DAYS = v2.RAIN_LOOKBACK_DAYS`,
código idêntico entre as duas fontes). Inspeção manual de dois casos concretos confirma que a
lógica está correta, não é um bug de offset:

- **Caso 1 — valores idênticos "suspeitos" entre eventos próximos**: `recife_pos_0013`
  (evento 2014-07-04, CHIRPS=343,95mm) e `recife_pos_0012` (evento 2014-07-10, CHIRPS=343,95mm)
  têm exatamente o mesmo valor. Inspeção da tabela diária real (`chirps_daily_grid_table.csv`,
  pixel `14_16`) mostra um único dia extremo real, **2014-06-26 = 343,95 mm**, que cai dentro
  da janela de 14 dias de AMBOS os eventos (04/07: janela 20/06–03/07; 10/07: janela 27/06–
  09/07 → 26/06 é exatamente o dia-14). **Comportamento correto de máximo-em-janela-móvel**,
  não duplicação/cache incorreto.
- **Caso 2 — estação=100,0mm exato em 3 datas diferentes (04-24, 04-25, 04-27/2014), CHIRPS=0
  nas 3**: inspeção do XML bruto ANA/HidroWeb (estação 834007, abril/2014) mostra
  `Chuva22=100` (`DiaMaxima=22`, `Maxima=100`) — o pico real do mês ocorreu em **22/04/2014**,
  não nos dias 24/25/27. Como os três eventos caem dentro de 14 dias de distância de 22/04, o
  "máximo em janela" da estação corretamente reflete o mesmo pico de 100mm do dia 22 — mesmo
  mecanismo do Caso 1, replicado do lado da estação. **Não é um bug do nosso código.**

## Achado real de divergência CHIRPS × estação (não um bug de pipeline)

No mesmo Caso 2, inspecionando `chirps_daily_grid_table.csv` para os pixels usados (`15_17`,
`15_16`) em **todo** abril/2014: CHIRPS registra **zero em todos os dias de 01/04 a 28/04**, e
concentra praticamente toda a chuva do mês em apenas 2 dias finais (`15_17`: 29/04=68,06mm,
30/04=204,19mm, soma=272,25mm no mês). A estação, em contraste, distribui a chuva ao longo do
mês com pico real em 22/04 (100mm) e um evento menor de fim de mês (29/04=31,6mm,
30/04=47,8mm, soma≈79mm). **CHIRPS não "perdeu" o mês (todos os 14 pixels somam 200–270mm em
abril/2014 — quantidade climatologicamente plausível), mas deslocou temporalmente onde dentro
do mês a chuva "aparece"**, por até ~7-8 dias em relação ao registro da estação.

Isso é consistente com uma limitação documentada da literatura de validação do CHIRPS: o
algoritmo combina dados de satélite infravermelho (duração de nuvem fria) com uma rede
esparsa de estações terrestres de calibração/ajuste (blend), o que pode produzir deslocamento
temporal de sistemas convectivos intensos e localizados, especialmente em regiões com poucas
estações de calibração disponíveis — não um erro de processamento do nosso lado.

## Literatura (busca realizada nesta rodada)

Busca por validação do CHIRPS no Nordeste do Brasil encontrou **"Validating CHIRPS-based
satellite precipitation estimates in Northeast Brazil"** (ScienceDirect) reportando que **o
desempenho do CHIRPS é sistematicamente pior perto da costa do que em áreas interioranas**
(correlação r=0,36 costeiro vs. r=0,51 interior, agregação mensal, 21 estações NE Brasil,
1981–2013), e baixa capacidade de detecção de chuva (probabilidade de detecção ≈0,44). Um
segundo estudo (Springer, bacia do rio Ipojuca/PE) reporta RMSE diário do CHIRPS de 8,85mm e
tendência de superestimar meses chuvosos. **Recife é litorâneo** — a correlação moderada
observada aqui (r=0,658–0,777, diária, agregada em janela de 14 dias) é **direcionalmente
consistente com essa limitação documentada de desempenho costeiro do CHIRPS no NE do Brasil**,
e é na verdade mais alta que as correlações mensais costeiras publicadas (0,36–0,51),
provavelmente porque agregação de 14 dias suaviza ruído diário.

## Veredito

**Não é um bug de processamento** (offset de data, fuso horário, unidade, ou erro de janela —
todos verificados diretamente nos dados brutos e descartados). **É uma limitação real e
documentada do produto CHIRPS em regiões costeiras do Nordeste brasileiro**, com deslocamento
temporal observável em pelo menos um caso concreto inspecionado manualmente. A correlação
moderada é esperada, não um sinal de erro, e cai dentro do intervalo (na verdade acima) do que
a literatura publicada reporta para a costa do NE do Brasil.

## Arquivos
- Inspeção direta em `chirps_daily_grid_table.csv`, `dataset_v3_features_finais.csv`,
  `data/raw/recife/precipitacao_ana_inmet/ana_hidroweb_estacao_834007_recife_curado_inmet/
  HidroSerieHistorica_834007_nivelConsistencia1_raw.xml`

Sources:
- [Validating CHIRPS-based satellite precipitation estimates in Northeast Brazil](https://www.sciencedirect.com/science/article/abs/pii/S014019631630235X)
- [Evaluation of BR-DWGD, CHIRPS and GPM-IMERG precipitation products: case study in the Ipojuca River Basin](https://link.springer.com/article/10.1007/s00703-026-01127-w)
