# SUSC-20G — HAND/TWI D-infinity como unidade genérica, validada contra Recife

**Status**: INFRAESTRUTURA_VALIDADA | **Data**: 2026-07-27
**Escopo**: só engenharia. Nenhum ponto foi amostrado, nenhum rótulo criado, nenhum modelo
treinado. Curitiba e Petrópolis continuam com N=1 positivo cada — abaixo do piso EPV (~20-30)
documentado em `docs/metodologia_cientifica/revp_linhagem_coleta_curitiba_petropolis_paridade_recife.md`.

## 1. O problema que esta rodada fecha

O cálculo real de HAND/TWI D-infinity usado no v12 de Recife existia apenas descrito em prosa
(`improvement2_hand_twi_dinf_report.md`, PROJETO). Não havia script: só um amostrador de pontos
que lia rasters já prontos. Consequência prática: nenhuma região nova podia receber HAND/TWI
pelo mesmo método, e o próprio Recife não era reproduzível a partir do MDT.

## 2. Scripts entregues

| Arquivo | Papel |
|---|---|
| `scripts/compute_hand_twi_dinfinity.py` | encadeamento D-infinity completo; MDT de entrada e diretório de saída são parâmetros |
| `scripts/prepare_region_dtm.py` | converte grade Esri ArcInfo Binary Grid (formato SGB/CPRM) em GeoTIFF, com reamostragem opcional |
| `scripts/compare_rasters.py` | comparação pixel a pixel entre raster gerado e raster de referência |
| `tests/test_susc_20g_hand_twi_dinfinity.py` (raiz do repo) | invariantes em MDT sintético + regressão real contra Recife |

Sequência implementada, idêntica à documentada no v7/v12:
`fill_depressions_wang_and_liu(fix_flats=True)` → `d_inf_flow_accumulation(out_type="cells")` →
`d_inf_flow_accumulation(out_type="sca")` → `slope(units="degrees")` → limiar de drenagem no
percentil 98 da acumulação em células → `extract_streams` → `elevation_above_stream` (HAND) →
`wetness_index(sca, slope)` (TWI).

Dependência instalada nesta máquina para viabilizar a execução: pacote Python `whitebox` 2.3.6
(binário WhiteboxTools v2.4.0, baixado pelo próprio pacote na primeira execução).

## 3. Validação contra Recife — resultado real

Entrada: o **mesmo** MDT merged de 10 m que gerou o v12 (48 tiles PE3D, EPSG:31985,
2777×2086, 3.579.890 células válidas). Saída comparada contra `hand_dinf.tif` e `twi_dinf.tif`
já existentes.

Parâmetros reproduzidos antes da comparação de raster:

| Parâmetro | Documentado no v12 | Reproduzido aqui |
|---|---:|---:|
| Células válidas | 3.579.890 | 3.579.890 |
| Limiar de drenagem (percentil 98, células) | 1122,74 | 1122,7383 |
| Cobertura HAND (% das células válidas) | 97,27% | 97,27% |
| Cobertura TWI | 100,00% | 100,00% |

Comparação pixel a pixel (`registries/v20g_recife_regression_comparison.json`):

| Raster | Pixels comparados | Pearson r | Diferença média | Diferença máxima |
|---|---:|---:|---:|---:|
| `hand_dinf.tif` | 3.482.047 | **1,0000000** | 0,0 m | **0,0 m** |
| `twi_dinf.tif` | 3.579.890 | **1,0000000** | 0,0 | **0,0** |

Não é "parecido": é **reprodução bit a bit**. Média e desvio-padrão coincidem em todas as casas
decimais impressas (HAND média 13,6067 m / desvio 15,6100; TWI média 5,8404 / desvio 4,0331 nos
dois lados). O critério de aceite fixado para esta rodada era r ≥ 0,99; o resultado é exato.

**O que isso autoriza e o que não autoriza**: autoriza aplicar o script em região nova com o
mesmo método do v12. Não diz nada sobre qualidade preditiva de HAND/TWI — o próprio relatório do
v12 registra que D-infinity corrigiu o sinal do TWI mas não tornou nenhuma das duas features
estatisticamente robusta naquele tamanho de amostra.

## 4. Execução em Curitiba e Petrópolis

Só depois da validação acima. Fontes: os MDTs/MDEs SGB/CPRM já confirmados localmente.

| | Curitiba | Petrópolis |
|---|---|---|
| Pasta na fonte | `01.MDS_Hipsometria/mdt/` | `MDE/pt_sirgas_utm/` |
| Formato nativo | Esri ArcInfo Binary Grid, float32 | idem |
| Resolução nativa | **2,5 m** | **30,13 m** |
| Grade processada | 10 m (agregação `average`) | 30,13 m (nativa, sem reamostragem) |
| CRS atribuído | EPSG:31982 (SIRGAS 2000 / UTM 22S) | ~~EPSG:31984~~ → **EPSG:31983** (SIRGAS 2000 / UTM 23S) — ver correção em `susc_20g2_petropolis_mdt_terreno_nu_report.md` §4 |
| Grade | 3301×2062 | 1483×1506 |
| Células válidas | 4.357.585 | 1.218.649 |
| Elevação (min–max) | 861,8 – 1022,2 m | 26,5 – 2233,6 m |
| Limiar de drenagem (P98) | 847,37 células | 826,76 células |
| Cobertura HAND | 96,79% | 97,53% |
| HAND (mediana / máx) | 8,93 m / 77,37 m | 103,83 m / 1269,76 m |
| TWI (mediana) | 6,53 | 6,00 |

Rasters gerados: `hand_dinf.tif` e `twi_dinf.tif` por região, em
`local_runs/susc_20g_hand_twi_dinfinity_generico/<região>/` (diretório git-ignored, como manda o
a regra fixa do projeto). Inventário sanitizado em `registries/v20g_hand_twi_dinf_readiness_registry.csv`.

### Ressalvas registradas, não contornadas

1. **Petrópolis é MDE, não MDT.** O ZIP do SGB/CPRM para Petrópolis não contém pasta de modelo
   de terreno — as pastas de topo são `Declividade`, `Fusao`, `Hipsometria`, `MDE`,
   `Relevo_sombreado`. O caminho indicado na tarefa (`MDE/pt_sirgas_utm/`) é modelo de
   **superfície**. Coerente com o achado já registrado de que MDE ≠ MDT (offset da ordem de
   metros em área vegetada/edificada). Curitiba, ao contrário, tem pasta `mdt` explícita.
   **Resolvido em SUSC-20G/2** com FABDEM V1-2 (terreno nu, 30 m) — ver
   `susc_20g2_petropolis_mdt_terreno_nu_report.md`.
2. **Resoluções diferentes entre regiões.** 10 m em Recife e Curitiba, 30,13 m em Petrópolis. O
   limiar de drenagem é percentil da distribuição de acumulação em células, e célula de 30 m
   drena área 9× maior que célula de 10 m: o HAND de Petrópolis (mediana 103,8 m) **não é
   numericamente comparável** ao de Recife (7,1 m). Parte da diferença é relevo real (Serra dos
   Órgãos vs planície costeira), parte é resolução. Não foi feita reamostragem de Petrópolis
   para 10 m de propósito — interpolar 30 m para 10 m não cria informação.
3. **Curitiba foi agregada de 2,5 m para 10 m** para ficar na mesma grade do v12. O dado nativo
   de 2,5 m continua disponível: basta trocar `--target-res` no `prepare_region_dtm.py`.
4. **44 células de Curitiba (0,001%) têm TWI negativo**, todas em `slope = 90°` — parede vertical
   na borda de nodata da grade. Artefato de borda conhecido, quantificado, sem efeito prático.
5. **Cobertura de HAND < 100% em toda região** (97,27% Recife, 96,79% Curitiba, 97,53%
   Petrópolis): células que não alcançam a rede de drenagem extraída no limiar P98. É o mesmo
   comportamento do v12, não uma degradação nova.

## 5. Testes

`python -m pytest tests/test_susc_20g_hand_twi_dinfinity.py -v` → **12 passaram, 0 falharam**
(16,05 s).

Cobrem: geração dos 7 rasters do encadeamento; grade de saída idêntica à de entrada; limiar igual
ao percentil pedido; HAND ≥ 0 e HAND = 0 exatamente sobre a drenagem; percentil maior produz
drenagem menor; MDT inexistente falha alto (não silencioso); comparador detecta grade divergente e
reconhece raster idêntico; reamostragem preserva extensão; e a **regressão real contra os rasters
do v12 de Recife** (limiar 1122,74 ± 0,01; r ≥ 0,99; diferença máxima ≈ 0). Se os rasters de
referência não estiverem montados na máquina, a regressão é **pulada com motivo explícito** — não
passa por omissão.

## 6. O que continua bloqueado

Nada aqui aproxima Curitiba ou Petrópolis de um modelo. HAND/TWI são raster de prontidão: não
foram amostrados em ponto nenhum, não viraram feature de dataset, não tocaram em rótulo. O
bloqueio segue sendo N=1 positivo por região.
