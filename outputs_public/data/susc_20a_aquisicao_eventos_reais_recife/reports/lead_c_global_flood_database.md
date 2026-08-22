# Lead C — Global Flood Database (Tellman et al. 2021): mirror sem autenticação GEE

**Status**: concluído — mirror real encontrado e usado sem autenticação Earth Engine.

## Mirror encontrado

O "Data availability" da Nature (`nature.com/articles/s41586-021-03695-w`) declara
explicitamente dois caminhos: (1) Google Earth Engine (`GLOBAL_FLOOD_DB_MODIS_EVENTS_V1`,
requer auth — já confirmado bloqueado na rodada v11) e (2) **site Cloud to Street +
repositório GitHub oficial** `github.com/cloudtostreet/MODIS_GlobalFloodDatabase`. Este
repositório expõe:

1. **`data/shp_files/dfo_polys_20191203.shp/.dbf/.shx/.prj`** — o catálogo completo de
   polígonos do Dartmouth Flood Observatory (DFO) usado como entrada do GFD, com **4.825
   registros globais, 1985–2019** (mais amplo que os 913 eventos MODIS-validados finais),
   baixado diretamente via `raw.githubusercontent.com` (sem qualquer autenticação).
2. **Bucket público do Google Cloud Storage `gs://gfd_v1_4`**, listável e baixável via HTTPS
   simples (`storage.googleapis.com/storage/v1/b/gfd_v1_4/o`) sem gsutil nem conta GEE —
   contém os GeoTIFFs de cada um dos 913 eventos validados (`flooded`, `duration`,
   `clear_views`, `clear_perc`, `jrc_perm_water`), nomeados `DFO_{id}_From_{data}_to_{data}.zip`.

Isto **contradiz o bloqueio reportado na rodada v11** ("requer autenticação Earth Engine,
indisponível") — havia uma rota alternativa real e pública que não tinha sido tentada até o
fim.

## Filtragem para Brasil / Pernambuco / Recife

Do catálogo DFO completo (4.825 registros), **112 são do Brasil**. Destes, **13 têm polígono
(não apenas centróide) que intersecta uma caixa costeira ampla de Pernambuco**
(lon -35,3/-34,7, lat -8,3/-7,7) — mas os polígonos brutos do DFO são propositalmente
grosseiros (ex.: registro 308 cobre lon -53 a -34, metade do Nordeste) e servem apenas como
região de busca de imagem, não como extensão de enchente validada.

Dos candidatos com ano ≥2000 (janela real de cobertura do GFD final validado por MODIS),
apenas **2 de 9 testados** existem de fato no bucket `gfd_v1_4` (ou seja, passaram pela
validação MODIS e entraram nos 913 eventos finais): `DFO_3291` (2008-03-30 a 2008-04-22) e
`DFO_3667` (2010-06-22 a 2010-06-30, o conhecido desastre de Alagoas/Zona da Mata Sul —
seu polígono **não intersecta** a caixa de Recife).

## Evento confirmado com pixels reais sobre Recife

**DFO_3291** (`GLIDE FL-2008-000045-BRA`, 30/03/2008–22/04/2008, "Heavy rain", 36 mortos,
190.000 desalojados, país todo): o raster de 5 bandas (EPSG:4326) foi baixado
(42,99 MB) e recortado exatamente na caixa de Recife (lon -35,05/-34,85, lat -8,15/-7,9).
Resultado real (não fabricado): **167 pixels com `flooded=1`** na janela; após excluir água
permanente (`jrc_perm_water=1`, água/canal do rio já existente), restam **13 pixels de
inundação nova genuína**, agrupados em lon -34,89/-34,87, lat -8,11/-8,05, duração 1–12 dias.

Geocodificação reversa (Nominatim) do centróide desses 13 pixels (-34,8801, -8,0830) confirma:
**bairro Brasília Teimosa, Recife/PE** — comunidade costeira baixa historicamente conhecida
por alagamentos, fisicamente coerente (elevação extraída = 3,01 m, HAND = 6,18 m, slope
baixo).

## Novo registro

| point_id | data | lat | lon | tier |
|---|---|---|---|---|
| LEADC_2008_0001 | 2008-03-30 | -8.082997 | -34.880064 | `global_flood_database_modis_event_extent_centroid` (centróide de cluster de pixels MODIS reais, não endereço/bairro-oficial) |

Esta data (**2008-03-30**) é anterior a toda a cobertura do `dataset_v9_final.csv`
(2014-01-03 em diante) — é, de fato, "terreno historicamente novo",
via satélite MODIS validado (não via decreto/jornal).

## Não incorporado (documentado, não descartado por preguiça)

Os outros 111 registros Brasil do DFO não geraram novos pontos: 109 estão fora da caixa de
Recife (outras regiões do Brasil) ou, mesmo intersectando a caixa ampla de PE, não
correspondem a nenhum arquivo real no bucket `gfd_v1_4` (não passaram na validação MODIS final
— prováveis motivos: nebulosidade excessiva, evento de curta duração incompatível com o
compósito de 3 dias, ou simplesmente fora do escopo 2000–2018 do produto final). Isso foi
verificado por consulta direta ao bucket (HTTP), não presumido.

## Resultado quantitativo

**1 novo registro positivo** (evento-extensão-centróide, tier mais baixo dos três leads, mas
real e documentado), incorporado ao pipeline v9 completo e ao `dataset_v12_final.csv`.
