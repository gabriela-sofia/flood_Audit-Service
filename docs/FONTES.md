# Fontes

Registro das fontes de evidência do projeto: de onde cada dado veio, o que ele fornece,
quando foi adquirido e por que é aceito como evidência válida.

Toda fonte listada é **pública e oficial** ou publicação científica revisada por pares.
Nenhum dado foi comprado, estimado por proxy ou obtido por acesso privado. Onde a
aquisição foi feita por consulta programática, o endpoint está declarado e é reproduzível.

---

## 1. Evento oficial — Recife

| Fonte | O que fornece | Origem | Aquisição |
|---|---|---|---|
| **SEDEC / Defesa Civil do Recife** | 154 registros de ocorrência de enchente, geocodificados | Registro oficial municipal | 21/07/2026 |
| **Diário Oficial do Recife (DOME)** | Decretos de situação de emergência e calamidade por enchente | `dome.recife.pe.gov.br` — acervo pesquisável desde 30/04/2015 | 07/2026 |
| **ANA / SNIRH** | Inventário e séries históricas de estações fluviométricas na bacia | `snirh.gov.br/arcgis/rest/services`, camada `Hidroweb_BH/INVENTARIOS_ESTACOES` — ArcGIS REST público, sem autenticação | 07/2026 |
| **Global Flood Database v1** | Eventos de inundação mapeados por MODIS entre 2000 e 2018 | Repositório oficial do Cloud to Street, declarado no *Data availability* de Tellman et al. (Nature, 2021) | 07/2026 |

A geocodificação usou Nominatim (OpenStreetMap) ao vivo, com limite de 1,1 s por
requisição, restrição `countrycodes=br` e validação por caixa envolvente de Recife
(−8,35 a −7,85 de latitude; −35,20 a −34,75 de longitude). Cada registro tem log próprio.

## 2. Evento oficial — Curitiba

| Fonte | O que fornece | Origem | Aquisição |
|---|---|---|---|
| **SIAC 156** | 1.045 reclamações de enchente e alagamento com endereço, na janela 2023–2026 | Central 156 da Prefeitura de Curitiba, `curitiba.pr.gov.br` | 07–08/2026 |
| **SIAC 156 — categorias independentes de chuva** | 114 pontos de negativo por exclusão qualificada | Mesma base, filtrada por categoria causalmente independente de precipitação | 08/2026 |
| **Diário Oficial do Paraná** | Decretos de emergência | `diariooficial.cepe.com.br` | 07/2026 |

O endereço de cada reclamação passou por deduplicação, geocodificação por Nominatim e
adjudicação física contra HAND, TWI e declividade — a mesma régua aplicada em Recife.

## 3. Evento e negativo — frente externa

| Fonte | O que fornece | Origem | Aquisição |
|---|---|---|---|
| **Copernicus EMS Rapid Mapping** | 25.249 pontos de negativo observado em 119 AOIs; ativação EMSR720 no Rio Grande do Sul com 216,55 km² | Serviço da Comissão Europeia, API pública de ativações | 08/2026 |
| **Environment Agency (Reino Unido)** | 7.476 pontos de negativo por exclusão qualificada (3.738 / 3.738) em 201 eventos independentes, a partir dos *Recorded Flood Outlines* e do *Flood Map for Planning* | Dado aberto do governo britânico | 08/2026 |
| **Sen1Floods11** | 4.831 recortes de 512×512 a 10 m, cobrindo 120.406 km² em 11 eventos, com rótulo manual de três estados | Publicação científica com dado aberto | 08/2026 |
| **Urban Flood Observations (UFO)** | 215 recortes de 1024×1024 a 3 m em contexto urbano | Zenodo | 08/2026 |
| **ESA WorldCover** | Cobertura do solo a 10 m, usada no critério de exclusão qualificada | Agência Espacial Europeia, dado aberto | 08/2026 |

## 4. Terreno

| Fonte | O que fornece | Origem |
|---|---|---|
| **PE3D** | Modelo digital de terreno a 1 m para Pernambuco | Programa PE3D, Governo de Pernambuco |
| **Copernicus DEM GLO-30** | Modelo digital de superfície a 30 m, cobertura global | Copernicus, dado aberto |
| **FABDEM** | Modelo de terreno derivado, com edificação e vegetação removidas | Universidade de Bristol, uso acadêmico |

HAND, TWI e declividade **não** vêm de produto genérico: são rederivados por D-infinity
(Tarboton, 1997) pela cadeia própria do projeto, na mesma resolução para toda a base. A
reprodução foi conferida pixel a pixel contra o raster de referência em Recife e em
Curitiba.

## 5. Chuva

| Fonte | O que fornece | Origem |
|---|---|---|
| **Open-Meteo / ERA5-Land** | Precipitação diária, produto único em toda a base, cobertura 99,99% | Reanálise ERA5-Land (Muñoz-Sabater et al., 2021), servida por API pública sem autenticação |

## 6. Observação orbital

| Fonte | O que fornece | Origem |
|---|---|---|
| **Sentinel-2 (L2A)** | Bandas para detecção de lâmina d'água pré e pós-evento | Copernicus, via Earth Search / Planetary Computer |
| **Sentinel-1 (GRD)** | Retroespalhamento SAR em VV/VH, que enxerga através de nuvem | Copernicus, via catálogo público |
| **S2ID — Sistema Integrado de Informações sobre Desastres** | Janelas de evento por município, identificadas por código IBGE | Ministério da Integração e do Desenvolvimento Regional, `atlasdigital.mdr.gov.br` |

## 7. Referências científicas de método

| Referência | Uso no projeto |
|---|---|
| Tellman et al., *Nature*, 2021 | Global Flood Database e a magnitude da exposição a inundação |
| Nobre et al., *Journal of Hydrology*, 2011 | Definição de HAND |
| Beven & Kirkby, *Hydrological Sciences Bulletin*, 1979 | Índice topográfico de umidade |
| Tarboton, *Water Resources Research*, 1997 | Repartição de escoamento por D-infinity |
| Firth, *Biometrika*, 1993 | Estimador penalizado para evento raro |
| Peduzzi et al., *Journal of Clinical Epidemiology*, 1996 | Piso de eventos por preditor |
| Muñoz-Sabater et al., *Earth System Science Data*, 2021 | ERA5-Land |
| Abnar & Zuidema, *ACL*, 2020 | Rollout de atenção, na fila de revisão visual |

---

## 8. Por que estas fontes são aceitas

O rótulo positivo é **evento oficialmente registrado e geocodificado** por órgão público,
nunca interpretação de imagem. O negativo é declarado em três níveis — observado, por
exclusão qualificada e ausência de registro — e a proporção entre eles é reportada sempre
que um ajuste os mistura.

Toda fonte entra na mesma cadeia de derivação de terreno e na mesma resolução, o que
torna a variável comparável entre regiões e entre países. A unidade de validação é o
evento, não o ponto.

O detalhamento de cada rodada de aquisição — data, script, artefato e veredito — está em
[`EVIDENCIA.md`](EVIDENCIA.md). A composição da base está em
[`../modelo/DADOS.md`](../modelo/DADOS.md).
