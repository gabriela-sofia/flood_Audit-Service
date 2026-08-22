# SUSC-20F -- Pipeline de geoprocessamento sob demanda (Recife)

## Objetivo

Fecha a limitação mais importante deixada explícita no SUSC-20E: a API só respondia
`ok` para geometrias que contêm um dos 269 pontos já conhecidos do v12. Este pipeline
calcula as 6 features físicas em tempo real para **qualquer** coordenada dentro da
cobertura real disponível -- não mais restrito aos pontos de treino.

## Duas fontes reais, dois resultados de validação diferentes (relatados sem esconder nada)

### Chuva (Open-Meteo ERA5-Land archive API, pública, sem autenticação)

Validado contra 30 pontos reais do v12 que já usavam essa fonte: `rain_decay_index_api_chirps`
bate **exatamente** (diff=0,0) e `rain_peak_residual_orthogonalized` bate a nível de
arredondamento de ponto flutuante (diff máx=0,0027). Match completo -- mesma fórmula,
mesma API, aplicado aos coeficientes de ortogonalização já treinados
(`open_meteo_era5_land_archive_api`: beta=0,392, intercept=2,3942).

### Terreno (DTM PE3D merged, D-infinity)

`hand_m_dinf`/`twi_dinf`: match **exato** (mesma execução D-infinity que gerou os
valores de treino). `elevation_m`/`slope_deg`: **não batem exatamente** (diferença
média ~2,7m / ~5,6°) -- o raster original específico dessas duas colunas não existe
mais em nenhum `local_runs` (busca completa confirmou). Usei o DTM merged mais
recente disponível (mesma fonte de 48 tiles PE3D, fisicamente válido) como melhor
aproximação real, documentando a diferença explicitamente em vez de escondê-la. Ver
`validacao_terreno_sob_demanda.md` para a investigação completa. Impacto prático
limitado: são as duas features mais fracas do modelo treinado (p=0,37 e p=0,22).

## Cobertura real

Bbox WGS84 do DTM merged: lon -35,033 a -34,843, lat -8,168 a -7,916 (~21km x 28km,
cobre essencialmente toda a região de Recife já definida no produto). Fora disso:
`sample_terrain_features()` retorna `None` -- fail-closed, sem interpolar.

## Arquivos

- `scripts/terrain_features_on_demand.py` -- amostragem real de raster
- `scripts/rain_features_on_demand.py` -- chuva ao vivo + ortogonalização já treinada
- `scripts/on_demand_feature_engine.py` -- combina os dois em 1 vetor de 6 features
- `scripts/validate_terrain_on_demand.py`, `scripts/validate_rain_on_demand.py` -- validação contra os 269/97 pontos reais
- `results/validate_terrain_on_demand_summary.json`, `results/validate_rain_on_demand_summary.json`

## Variáveis de ambiente necessárias (nenhum path privado hardcoded)

- `REVP_SUSC20F_TERRAIN_RASTER_DIR` -- diretório local com os 4 rasters (privado, PROJETO)
- Chuva não precisa de variável -- API pública, sem chave
