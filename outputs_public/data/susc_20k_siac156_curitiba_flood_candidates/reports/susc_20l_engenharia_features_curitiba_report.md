# SUSC-20L — Engenharia de features físico-hidrológicas, Curitiba (SIAC 156)

**Status**: FEATURES EXTRAÍDAS, DATASET COMPLETO. Nenhum modelo treinado nesta etapa.
**Escopo**: só a Tarefa 1 (features). A modelagem/validação (SUSC-20M, espelho do SUSC-20C)
está travada atrás da decisão de EPV documentada na seção 6.

Entrada: `v20k2_dataset_positivos_curitiba_siac156_v1.csv` (1238 linhas) +
`v20k3_dataset_negativos_curitiba_siac156_v1.csv` (119 linhas).
Saída: `registries/v20l_dataset_curitiba_features_v1.csv` — **1357 linhas × 42 colunas**,
todas as colunas de proveniência dos dois registries preservadas.

---

## 1. Schema-alvo — completo

As 9 colunas do schema de Recife (`dataset_eventos_features_v12_final.csv`) estão todas
presentes: `elevation_m`, `slope_deg`, `hand_m_dinf`, `twi_dinf`, `rain_max_24h_chirps`,
`rain_decay_index_api_chirps`, `mapbiomas_class_2023`, `rain_data_source`,
`rain_peak_residual_orthogonalized`.

Além delas, uma coluna `*_status` por feature de terreno e por MapBiomas (`OK` /
`NODATA_NO_PIXEL` / `FORA_DA_EXTENSAO_DO_RASTER`), `rain_status`, `rain_n_days_found`,
`rain_grid_lat` / `rain_grid_lon` (célula de grade real que forneceu a chuva), e as duas
colunas de unidade de observação da seção 5.

## 2. Terreno — amostragem, não recálculo

Rasters de SUSC-20G (`local_runs/susc_20g_hand_twi_dinfinity_generico/curitiba/`, MDT
SGB/CPRM 10 m, EPSG:31982, WhiteboxTools 2.4.0 D-infinity). Mesmas regras de NODATA do
`read_hand_twi_slope_at_point.py`. `elevation_m` sai do MDT bruto (`curitiba_dtm_10m.tif`),
não do `dem_filled.tif` hidro-condicionado — espelha a extração original do v12, que
amostrou os tiles PE3D crus.

| feature | OK | NODATA_NO_PIXEL | FORA_DA_EXTENSÃO |
|---|---:|---:|---:|
| `elevation_m` | 1357 | 0 | 0 |
| `slope_deg` | 1357 | 0 | 0 |
| `hand_m_dinf` | **1343** | **14** | 0 |
| `twi_dinf` | 1357 | 0 | 0 |

**Os 14 NODATA de HAND não são erro de extensão** — a expectativa de "~0 porque os pontos
vêm de dentro da bbox" se confirmou para 3 das 4 features, e falhou só para HAND. Verificado
diretamente no raster: nesses pixels o MDT, a declividade, o TWI e a acumulação D-infinity
têm valor válido; só `hand_dinf.tif` é `-9999`. Causa: `ElevationAboveStream` deixa sem
valor a célula cujo caminho de fluxo a jusante não alcança nenhuma célula de canal (limiar
de 98º percentil, 847 células) antes de sair da máscara válida do MDT. São 14 linhas / **10
unidades de observação / 9 coordenadas distintas**, todas positivas, todas na franja
norte-oeste da cobertura (Pilarzinho, Butiatuvinha, Santa Cândida, Atuba). 0,9% das
unidades. Sob a exclusão listwise que o pipeline de Recife usa (`dropna`), elas saem do
modelo multivariado — e não saem nenhum negativo por esse motivo.

## 3. Chuva — Open-Meteo ERA5-Land, fórmula literal de SUSC-20B

Fórmula idêntica a `fetch_rain_leadA_positives.py`: janela de 14 dias terminando no dia
**anterior** ao `event_date` (o dia do evento nunca entra — trava anti-vazamento herdada do
v12), `rain_max_24h` = máximo diário da janela, índice API com decaimento `k=0.85`.

Nota sobre o nome das colunas: o sufixo `_chirps` é **legado do v12 de Recife**, onde 181
pontos vieram de CHIRPS e 97 de Open-Meteo na mesma coluna; quem declara a fonte real é
`rain_data_source`. Curitiba tem fonte única, `open_meteo_era5_land_archive_api` — a mesma
API que o script de referência de fato chama. Divergência declarada: timezone
`America/Sao_Paulo` em vez de `America/Recife` (fuso local correto de cada região; ambos
UTC−3 sem horário de verão no período 2023-2026, então os cortes diários coincidem).

Ponto negativo usa a `event_date` real do próprio registro (data da ocorrência
não-hidrológica), nunca data sintética — mesma regra de Recife.

**Cobertura: 1357/1357 com janela completa de 14/14 dias reais. 0 falha, 0 janela parcial.**

Otimização de rede (não muda valor): as 848 coordenadas únicas resolvem para apenas **10
células de grade distintas** do arquivo ERA5-Land. Em vez de 848 requisições da mesma série,
o script faz uma passada curta de descoberta de célula em lote e depois 1 requisição de span
completo por célula, com `assert` de que o centro da célula resolve para si mesmo. A
equivalência foi verificada de ponta a ponta: para 20 linhas sorteadas, o script refez a
requisição isolada de 14 dias **nas coordenadas originais do ponto** e comparou —
**0 divergências** (`local_runs/susc_20l_curitiba_features/qa_rain_window_equivalence.json`).

Limitação real que isso expõe: a resolução espacial efetiva da chuva em Curitiba é de 10
células (~7,8 km × ~5,3 km cada) para 848 localizações. Num mesmo dia de evento, a maioria
dos pontos compartilha exatamente o mesmo valor de chuva. A chuva discrimina **entre datas**,
quase não discrimina **entre pontos na mesma data**. O mesmo vale para o v12 de Recife e
nunca foi quantificado lá.

## 4. MapBiomas

Coleção 9 (`mapbiomas_collection90_integration_v1`), banda `classification_2023`, 30 m — o
mesmo asset público de SUSC-19B/19F. **848/848 coordenadas com classe válida, 0 NODATA.**

| classe | descrição | linhas |
|---:|---|---:|
| 24 | área urbanizada | 1318 |
| 21 | mosaico de agropecuária | 31 |
| 3 | formação florestal | 7 |
| 12 | formação campestre | 1 |

97,1% em classe 24. Como preditor, essa variável é quase constante em Curitiba — vale
registrar antes de qualquer expectativa de que ela carregue sinal.

## 5. Unidade de observação — achado que muda a aritmética de N

Os registries de entrada têm repetição de `point_id`: **1238 linhas positivas ↔ 1088
`point_id` únicos**, **119 linhas negativas ↔ 103 `point_id` únicos**. Isso já estava
documentado na seção 4 do relatório 20K2/K3 como "duplicação real modesta (mesma reclamação
categorizada duas vezes, ex.: `Drenagem/Alagamentos` + `Proteção ao cidadão - GM/Enchentes`)"
— mas o efeito sobre N não tinha sido puxado adiante.

Como as features são função apenas de (lat, lon, `event_date`), essas linhas repetidas têm
feature **bit a bit idêntica** (verificado em teste). Contá-las como observações
independentes é pseudo-replicação: infla N sem trazer informação nova.

Chaveando por (lat, lon, `event_date`) — a unidade físico-hidrológica real:

| | linhas | unidades de observação |
|---|---:|---:|
| positivos | 1238 | **1045** |
| negativos | 119 | **103** |
| total | 1357 | **1148** |

As duas colunas `observation_unit_key` e `is_duplicate_observation_unit` marcam isso no
dataset. **Nada foi removido** — a escolha de unidade de análise é decisão da etapa de
modelagem, não da extração.

## 6. Gate EPV — checado antes do modelo, como exigido

Classe minoritária = negativo. Com listwise deletion nas 6 features de Recife, nenhum
negativo é perdido (os 14 NODATA de HAND são todos positivos).

| base de contagem | n negativo | EPV com 6 features | EPV com 5 features |
|---|---:|---:|---:|
| linhas brutas | 119 | 19,83 | 23,80 |
| **unidades de observação** | **103** | **17,17** | **20,60** |

O piso de 20 da revisão de literatura do projeto
(`docs/metodologia_cientifica/revp_revisao_literatura_alinhamento_metodos_v1.md`) **não é
atingido em nenhuma das duas contagens com 6 features**. Pela contagem correta (unidades),
5 features passa por margem estreita (20,60).

Nenhum modelo multivariado foi rodado. A decisão entre reduzir para 5 features (rota a) ou
ampliar N de negativo antes (rota b) fica registrada como aberta — ver seção 8.

## 7. Ortogonalização

`orthogonalize_per_source`, lógica idêntica a `build_v12_dataset.py`, **refitada nos pontos
de Curitiba** (não herda o β/intercepto de Recife — regiões e regimes de chuva distintos).
Fonte única, um ajuste:

```
open_meteo_era5_land_archive_api:
  n_used = 1357   beta = 0.3582   intercept = 9.3202
  pearson_r(rain_max_24h, api)  antes = 0.6722   depois = 0.0
```

## 8. Limitações e decisões em aberto

1. **EPV não fecha com 6 features** (seção 6) — decisão de rota pendente.
2. **10 células de chuva para a cidade inteira** (seção 3): a chuva é quase um efeito de
   data, não de local. Se o modelo achar sinal em chuva, isso é discriminação temporal
   entre eventos, não espacial entre pontos.
3. **MapBiomas quase constante** (97,1% classe 24) — poder discriminativo próximo de zero
   por construção.
4. **10 unidades positivas sem HAND** saem do modelo por listwise deletion.
5. **Cobertura do MDT**: 36% do bbox do raster é NODATA (área fora da máscara válida do MDT
   SGB/CPRM). Todos os 1357 pontos caíram dentro da máscara válida para MDT/declividade/TWI.
6. **`elevation_m` do MDT bruto, não do `dem_filled`**: espelha a extração original do v12.
   A diferença entre os dois só existe em depressões preenchidas; não foi quantificada aqui.
7. Positivos e negativos vêm da **mesma fonte administrativa** (SIAC 156) e do mesmo
   processo de geocodificação, o que controla viés de fonte — mas continuam sendo pontos
   administrativos geocodificados por rua, não verdade cartográfica de campo. A regra de uso
   permitido/proibido herdada dos registries de entrada foi preservada linha a linha.

## 9. Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/build_v20l_curitiba_features.py
python -m pytest tests/test_susc_20l_curitiba_features.py -q
```

Requer os rasters de SUSC-20G em `local_runs/` e Earth Engine autenticado. Caches de rede
(série de chuva por célula, classes MapBiomas) ficam em
`local_runs/susc_20l_curitiba_features/` — rodadas seguintes não repetem chamada externa.
