# GAP 3 — Cobertura HAND (35,7%) e preenchimento de depressões

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-22

## Tentativa de instalação

- `pip install richdem --break-system-packages`: **FALHOU** — não é erro de código, é uma
  limitação real do sandbox: cada chamada de shell deste ambiente é efêmera (nenhum estado ou
  processo em segundo plano sobrevive entre chamadas, confirmado empiricamente: um processo
  `nohup ... &` iniciado numa chamada não aparece mais em `ps aux` na chamada seguinte). O
  build da extensão C++ do richdem (`Building wheel for richdem`) não termina dentro da janela
  de execução de uma única chamada (~45 s), e não há como continuar o build entre chamadas.
  Documentado como limitação de sandbox, não contornado.
- `pip install whitebox --break-system-packages`: **SUCESSO** — pacote Python puro (wheel sem
  compilação), instala em segundos. No primeiro uso, baixa o binário pré-compilado
  WhiteboxTools (Rust, algoritmo de Wang & Liu / Priority-Flood real, não uma aproximação) —
  também funcionou sem timeout.

## Recomputação real do HAND com preenchimento de depressões

1. DEM mesclado 10 m real do v2 (`_scratch_hand_dem_10m.npy`, 2777×2086, EPSG:31985,
   `nodata=-9999`) escrito como GeoTIFF real.
2. `whitebox.WhiteboxTools().fill_depressions(..., fix_flats=True)` — Wang & Liu (2006) real,
   executado em 0,88 s (excluindo I/O).
3. D8 + acumulação de fluxo + HAND recomputados com o **mesmo código exato** do
   `pipeline_recife_v2.py::compute_recife_hand()` (mesma lógica de pointer-doubling,
   `stream_thresh=30` células), agora sobre o DEM preenchido.

### Replicação de controle (DEM bruto, sem preenchimento)

`n_sinks=231.670`, `n_resolved=1.278.765/3.579.890` → **35,72% resolvido** — bate exatamente
com os 35,7% já reportados no v2 (confirma que o array de cache e a reimplementação são
idênticos ao pipeline original).

### Resultado com preenchimento de depressões

| Métrica | DEM bruto (v2/v3) | DEM preenchido (v4) |
|---|---:|---:|
| Sinks sem vizinho descendente | 231.670 | **1.481** (−99,4%) |
| Células HAND resolvidas / válidas | 1.278.765 / 3.579.890 | **3.570.783 / 3.579.890** |
| % resolvido (grade completa) | 35,72% | **99,75%** |
| Hit direto nos 282 pontos (sem fallback 150m) | 33,7% (95/282) | **100% (282/282)** |
| Pontos resolvidos (≤150m tolerância) | 281/282 (99,6%) | **282/282 (100%)** |

**Melhoria real e substancial de cobertura**: de 35,7% para 99,75% da grade, e de 33,7% para
100% de hit direto nos 282 pontos — elimina por completo a necessidade de fallback por
tolerância de busca (e, portanto, o canal de missingness diferencial identificado no Gap 1).

## O preenchimento resolve a incoerência de sinal?

**Não totalmente, mas melhora**: com HAND preenchido, `hand_m_filled` passa a ser **fisicamente
coerente** na auditoria multivariada do v4 (logreg coef=−0,054, GBM corr=−0,040, ambos
negativos concordantes — ver `coeficientes_coerencia_v4.csv`), contra o v3 onde `hand_m` era
incoerente (+0,050/+0,075). A correlação univariada com o label continua fraca
(corr=+0,087, quase nula) — o preenchimento resolve o problema de **cobertura/missingness**,
não cria um sinal físico forte que não existia (esperado: HAND é apenas parcialmente
informativo em terreno tão plano quanto Recife).

## Decisão

**Substituir `hand_m` (v3, sem preenchimento) por `hand_m_filled` (v4, com preenchimento Wang &
Liu real via WhiteboxTools) em `FEATURE_COLS_V4`.** Justificativa: cobertura passa de 35,7%
para 99,75% sem custo de qualidade (mesmo método D8/threshold, mesma fonte de DEM), e o sinal
passa a ser fisicamente coerente na auditoria multivariada — mudança de feature genuinamente
justificada pelos dois critérios do projeto (dado real, não fabricado; melhoria mensurável).

## Arquivos
- `/tmp/recife_dem_10m_raw.tif`, `/tmp/recife_dem_10m_filled.tif` — DEM antes/depois (não
  persistidos no repositório, artefatos de sessão)
- `_scratch_hand_filled_10m.npy` — grade HAND preenchida (2777×2086)
- `_scratch_dataset_with_hand_filled.csv` — `hand_m_filled` extraído nos 282 pontos
- `dataset_v4_features_finais.csv` — dataset final com `hand_m_filled` incorporado
