# Lead B — ANA Feature Service: estações reais de rio para Recife

**Status**: concluído (endpoint real encontrado, consulta espacial executada, séries
históricas reais baixadas e analisadas)

## Endpoint REST real encontrado

`https://www.snirh.gov.br/arcgis/rest/services` — diretório ArcGIS REST **público, sem
autenticação**, respondendo `f=json`. Pastas relevantes: `Hidroweb_BH` (inventário de
estações), `Telemetria_BH`, `SGH`, `DADOSABERTOS`.

Layer usado: `Hidroweb_BH/INVENTARIOS_ESTACOES/MapServer/0` (Feature Layer de pontos,
`geometryType=esriGeometryPoint`), com campos `EST_CD_FLU` (código fluviométrico),
`EST_CD_PLU` (pluviométrico), `RIO_NM`, `MUN_NM`, `ULTIMAATUALIZACAO`, `OPERANDO`,
`POSSUI_DADOS`.

## Consulta espacial (bbox Recife)

Query `geometry=-35.05,-8.15,-34.85,-7.9&geometryType=esriGeometryEnvelope&inSR=4326
&spatialRel=esriSpatialRelIntersects&where=EST_CD_FLU IS NOT NULL` retornou **24 estações
fluviométricas reais** na área metropolitana do Recife, incluindo:

| Código | Nome | Rio | Município | Última atualização (metadado) |
|---|---|---|---|---|
| 39189000 | Parque de Santana | Rio Capibaribe | Recife | 05/07/2005 |
| 39189500 | Ilha do Retiro | Rio Capibaribe | Recife | 05/07/2005 |
| 39098050 | Captação Beberibe | Rio Beberibe | Recife | 25/07/2012 |
| 39098200 | Linha do Tiro | Rio Morno | Recife | 05/07/2005 |
| **39187800** | São Lourenço da Mata II | **Rio Capibaribe** | São Lourenço da Mata (RMR, upstream) | 31/01/2017 |
| **39187900** | Açude Maria da Luz | **Rio Pixaó** (afluente do Capibaribe) | São Lourenço da Mata (RMR) | 22/12/2015 |

(Query por `MUN_NM='RECIFE'` sozinha retornou 42 estações no total, incluindo
pluviométricas.) Nenhuma estação nomeada "Tejipió" apareceu como fluviométrica — só existe
como estação pluviométrica convencional ("TIJIPIO", 00834011).

## Download de séries históricas reais

O metadado `ULTIMAATUALIZACAO` do Feature Service **subestima a disponibilidade real** — ao
consultar o endpoint SOAP/GET legado `telemetriaws1.ana.gov.br/ServiceANA.asmx
/HidroSerieHistorica` diretamente, encontrou-se:

- **Estação 39187800 (Rio Capibaribe, São Lourenço da Mata II) — Vazão (m³/s)**: série diária
  real e contínua de **1990-01-01 a 2026-03-31** (13.178 dias com valor), muito além do que o
  metadado (2017) sugeria.
- **Estação 39187900 (Rio Pixaó) — Cota (nível, cm)**: série diária real de **2018-01-11 a
  2026-07-21** (2.806 dias com valor) — cobre exatamente a janela dos dois eventos-decreto do
  Lead A (2022 e 2026).
- As demais 5 estações fluviométricas dentro do município de Recife propriamente dito
  (Parque de Santana, Ilha do Retiro, Captação Beberibe, Linha do Tiro, Ponte Caxanga)
  retornaram genuinamente "Sem dados para esta estação" em toda a janela testada (1990–2026,
  `tipoDados` 1/2/3) — consistente com o metadado `ULTIMAATUALIZACAO` de 2005/2012: são
  estações convencionais desativadas há muito tempo, sem série digitalizada acessível por
  este endpoint.

## Análise de corroboração hidrológica

Limiar de "alta vazão/cota": **percentil 95 da própria série da estação** (conforme
instruído). p95 Vazão (Capibaribe) = 34,99 m³/s (n=13.178 dias); p95 Cota (Pixaó) = 9.911 cm
(n=2.806 dias).

**14 das 101 datas positivas já existentes em `dataset_v9_final.csv`** mostram vazão e/ou
cota ≥ p95 no mesmo dia — corroboração hidrológica independente e real (fonte diferente da
chuva CHIRPS/Open-Meteo já usada):
2014-05-14, 2014-06-26, 2014-06-27, 2014-09-08, 2016-04-17, 2016-04-18, 2016-05-30,
2017-06-01, 2017-07-21, 2021-05-13, 2023-06-27, 2023-06-30, 2024-04-17, 2024-06-15.

**Achado mais forte**: a data do decreto 35.669 (**28/05/2022**, Lead A) apresenta vazão de
**165,45 m³/s no Rio Capibaribe** — percentil 99,48% de toda a série de 36 anos (top 0,52%
histórico) — e cota de 9.920 cm no Rio Pixaó (acima do p95) — confirmação hidrológica
independente e muito forte da magnitude excepcional do evento.

A data do decreto 39.714 (**02/05/2026**) não tem leitura de vazão (série Capibaribe termina
em 2026-03-31) nem de cota exata nesse dia (gap de dados entre 2026-04-30 e a leitura
seguinte — possível falha de sensor durante o próprio evento extremo, não confirmável
causalmente, apenas registrado como coincidência honesta).

## Uso no dataset

As leituras de vazão/cota são **evidência de corroboração temporal/de área** (estações estão
fora do município de Recife, a montante no Capibaribe/Pixaó) — não foram usadas para criar
novos pontos de treino com lat/lon dentro de Recife, exatamente como instruído ("usável
apenas como corroboração de área/tempo, não um novo registro pontual"). Nenhum novo ponto foi
adicionado a partir do Lead B; seu valor é de validação cruzada do Lead A e de 14 datas já
existentes no dataset v9.

## Achado adicional real (não solicitado, mas notável)

A série de vazão do Capibaribe mostra o pico histórico absoluto em **1990-07-29 (1.048,58
m³/s)** — coincide exatamente com um evento no catálogo DFO/Global Flood Database (Lead C,
registro ID 440: 1990-07-29 a 1990-07-31, "Heavy rain", 27 mortos, centróide -8,034/-35,051,
bem próximo de Recife) — outra confirmação cruzada real entre as duas fontes independentes,
reportada apenas como observação qualitativa (fora do período coberto pelo dataset atual).
