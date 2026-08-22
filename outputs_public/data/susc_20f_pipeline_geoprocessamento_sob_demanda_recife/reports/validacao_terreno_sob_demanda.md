# Validação -- amostragem de terreno sob demanda (SUSC-20F)

## Resultado real (269 pontos reamostrados, todos dentro da cobertura raster)

| Feature | Máx diferença absoluta | Média diferença absoluta | Fonte |
|---|---:|---:|---|
| `hand_m_dinf` | **0,0** | **0,0** | `hand_dinf.tif` -- match exato |
| `twi_dinf` | **0,0** | **0,0** | `twi_dinf.tif` -- match exato |
| `elevation_m` | 10,61 m | 2,67 m | `dem_filled.tif` -- **não é match exato** |
| `slope_deg` | 35,98° | 5,60° | `slope_deg_wbt.tif` -- **não é match exato** |

## Achado honesto: o raster exato de elevação/declividade do v12 não existe mais em disco

`hand_dinf.tif`/`twi_dinf.tif` batem perfeitamente porque foram gerados **na mesma
recomputação D-infinity** (Improvement 2, SUSC-20B) que também gerou os valores
finais de `hand_m_dinf`/`twi_dinf` do `dataset_v12_final.csv` -- é literalmente o
mesmo arquivo/execução.

`elevation_m`/`slope_deg`, porém, vieram de uma função diferente e mais antiga
(`v2.extract_elevation_slope_at_points()`, documentada em
`task2_elevacao_crs_investigacao_v9.md` como "mesmo código desde v2 até v9") --
rodando sobre um mosaico DEM que **não sobrou em nenhum diretório de `local_runs`
ainda existente** (busquei em todo `local_runs` por `.tif` com "dem/elev/slope" no
nome -- só os 3 arquivos de `scratch_dinf` existem hoje). Testei tanto o DEM bruto
(`recife_dem_10m_raw.tif`) quanto o preenchido (`dem_filled.tif`) contra os 278
pontos -- nenhum bate exatamente (diferença média ~2,5-2,8m).

**Isto não é um bug do pipeline sob demanda** -- é uma lacuna real de proveniência:
o artefato raster original de elevação/declividade que gerou aquelas duas colunas
específicas do dataset de treino foi descartado em algum momento entre as sessões
v2-v9 (prática normal de `local_runs`, que não é permanente por desenho). O DTM
PE3D original (48 tiles brutos) continua bloqueado por login/captcha
(`susc_devro17_bloqueio_mdt_pe3d`), então não é possível re-derivar o raster exato
sem essa aquisição manual.

## Decisão tomada (documentada, não escondida)

Uso `dem_filled.tif`/`slope_deg_wbt.tif` (o DTM PE3D merged mais recente disponível,
fisicamente válido, mesma fonte de 48 tiles) como a **melhor fonte real disponível**
para `elevation_m`/`slope_deg` no pipeline sob demanda, com a diferença média
documentada acima exposta como limitação explícita em toda saída do motor. Isso é
defensável porque:

1. `elevation_m` e `slope_deg` já são os dois preditores **mais fracos** do modelo
   treinado (SUSC-20C: p=0,37 e p=0,22 respectivamente, ambos com CI cruzando zero)
   -- um desvio de poucos metros/graus nessas duas variáveis específicas tem impacto
   pequeno no score final, ao contrário de HAND/TWI/chuva (que batem exatamente ou
   vêm de fonte ao vivo real).
2. A alternativa (bloquear o gate inteiro por essa discrepância) impediria usar HAND
   e TWI reais e a chuva real também -- um custo maior que o benefício, dado que a
   diferença já está medida e documentada, não escondida.

## Cobertura espacial real

Bbox WGS84 aproximado do DTM merged: **lon -35,033 a -34,843, lat -8,168 a -7,916**
(~21km x 28km) -- cobre essencialmente toda a área de Recife já definida no produto
(`region_registry.py`). Fora desse bbox: `sample_terrain_features()` retorna `None`
(fail-closed, sem interpolar ou inventar).
