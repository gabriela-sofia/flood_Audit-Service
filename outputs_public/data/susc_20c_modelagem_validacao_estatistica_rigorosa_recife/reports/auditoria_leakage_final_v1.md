# Auditoria de Leakage/Artefato — Modelo Oficial Recife v1

**Status**: PRIMEIRO_MODELO_OFICIAL_RECIFE_COM_INCERTEZA_DOCUMENTADA
**Data**: 2026-07-22
**Autorizado por**: Gabriela Sofia (reviewer/executora única)
**Escopo**: `dataset_final_oficial_recife.csv` (339 pontos: 141 positivos SEDEC-inundação
geocodificados + 198 negativos SEDEC-não-inundação geocodificados)

## Motivação

O histórico do projeto já teve um caso confirmado de leakage severo por tipo de geometria (v1,
corrigido em v2 — ver `local_runs/treino_exploratorio_diagnostico_v2/RELATORIO_COMPARATIVO_V1_VS_V2.md`).
Antes de declarar este modelo "oficial", repetimos a auditoria especificamente para o par
positivos-SEDEC-inundação vs negativos-SEDEC-não-inundação, que vêm de dois sub-pipelines de
geocodificação diferentes.

## Achado NOVO adicional — `rain_missing` removido do vetor de features (encontrado durante o ajuste do modelo, não na varredura inicial)

Ao ajustar o modelo oficial, o indicador `rain_missing` (criado para marcar os 9 positivos sem
data real de evento, que recebem chuva antecedente imputada por mediana) foi checado quanto a
separação por rótulo e **encontrado como um proxy quase-perfeito do rótulo**:

| `rain_missing` | label=0 (negativo) | label=1 (positivo) |
|---|---:|---:|
| 0 | 198 | 132 |
| 1 | 0 | **9** |

**100% dos casos com `rain_missing=1` são positivos** (os 9 positivos sem data de evento; os 198
negativos SEMPRE recebem uma data de referência reamostrada por construção metodológica do v5 —
nunca ficam sem chuva calculável). Isso significa que `rain_missing`, se usado como feature,
funcionaria como um proxy derivado do rótulo (viola a regra fixa do projeto de "não usar proxy
derivado do rótulo como feature"), mesmo sendo pequeno em magnitude (9/339 = 2,7% dos casos).

**Ação corretiva**: `rain_missing` foi **removido do vetor de features do modelo oficial**
(mantido apenas como coluna de auditoria no CSV consolidado). O vetor final de features é:
`elevation_m`, `slope_deg`, `rain_1d_mm_imputed`, `rain_3d_mm_imputed`, `rain_7d_mm_imputed` (5
features, todas de Camada 1 física-hidrológica; imputação por mediana sem indicador). A
validação cruzada robusta e os coeficientes/SHAP reportados nos demais arquivos deste diretório
já refletem essa correção (rodados sem `rain_missing`).

## Resultado — achados NOVOS (colunas de proveniência, não usadas como feature)

Estas colunas **existem no CSV consolidado para fins de documentação/auditoria**, mas **NUNCA
entram no vetor de features do modelo** — separam o rótulo de forma perfeita ou quase perfeita
exatamente porque descrevem qual sub-pipeline gerou o registro, não um sinal físico de risco:

| Coluna | Separação por label | Motivo |
|---|---|---|
| `source_layer` | **100% perfeita** (`recife_seced_geocoded`=141 pos/0 neg; `sedec_defesa_civil_non_flood_v3_full_geocode`=0 pos/198 neg) | Identifica o pipeline de origem, não risco |
| `geometry_type` | **100% perfeita** (`point_real_geocoded`=pos; `Point`=neg) | Rótulo de nomenclatura interna do pipeline, coincide 1:1 com `source_layer` |
| `classe` | **100% perfeita** (positivos = única classe "flood..."; negativos = 6 classes: árvore em risco, desabamento, deslizamento/barreira, incêndio, invasão, muro) | `classe` é, por definição, o texto-fonte do tipo de ocorrência — usar como feature seria usar o próprio rótulo disfarçado |
| `point_id` (prefixo) | **100% perfeita** (`recife_pos_*`=positivos; `REC_NEGV3_*`=negativos) | Identificador de linhagem, nunca deveria (e não é) usado como feature numérica/categórica |
| `reference_date_is_synthetic` | **100% perfeita** (False=positivos datados; True=negativos, que recebem data de referência reamostrada) | Por construção metodológica documentada em v5 (não há data real de "não evento"); não é sinal de risco físico |

**Verificação explícita item-a-item**:
- Remanescentes de texto de tipo de ocorrência (`classe`) → **SIM, encontrado**, tratado como
  metadado, nunca como feature.
- Prefixos de ID de registro (`point_id`) → **SIM, encontrado** (`recife_pos_` vs `REC_NEGV3_`),
  nunca usado como feature.
- Diferenças de distribuição de confiança de geocodificação entre os dois grupos → **verificado,
  diferença pequena e não decisiva** (ver abaixo).

## Resultado — features candidatas ao modelo (checagem limpa)

| Feature | Distribuição por label | Veredito |
|---|---|---|
| `confidence_tier` (strong/medium) | medium: 8,6% dos negativos vs 6,4% dos positivos; strong: 91,4% vs 93,6% | Diferença pequena, direção não sistemática o suficiente para configurar leakage; **não incluída no modelo headline** de qualquer forma (não é feature física) |
| `rain_station_used` (ANA_834007/ANA_834017/INMET_A301) | proporções variam (ex. ANA_834017: 37,9% neg vs 50,0% pos) mas todas as 3 estações aparecem em ambas as classes, sem separação perfeita | Limpo — reflete geografia dos pontos, não vazamento de rótulo |
| `rain_station_dist_m` | médias próximas (6393 m negativos vs 5191 m positivos) | Limpo — sobreposição de distribuição, não separação categórica |
| `rain_1d_mm` / `rain_3d_mm` / `rain_7d_mm` | médias quase idênticas entre classes (ex. rain_7d: 57,3 mm neg vs 54,1 mm pos) | Limpo — nenhuma diferença suspeita de artefato de pipeline |
| `elevation_m` / `slope_deg` | fonte idêntica (PE3D MDT, mesmo método corrigido de janela 31×31 px) para ambas as classes, extraída pelo mesmo código (`sample_dem_recife`, ver `build_recife_v3_dataset.py`) | Limpo — sem diferença de método por classe |
| `neighborhood` | 18/43 bairros contêm ambas as classes | Sobreposição parcial esperada (geografia real), não é separação artificial |
| `mde_source` (tile PE3D) | 13/27 tiles contêm ambas as classes | Sobreposição parcial esperada |

## Veredito da auditoria

**Resultado: LIMPO para o conjunto de features efetivamente usado no modelo** (elevação,
declividade, chuva antecedente 1/3/7d, indicador de imputação). **Achado confirmado (não é
achado novo, mas foi reverificado explicitamente para este dataset)**: as colunas de
proveniência/identificação (`source_layer`, `classe`, `geometry_type`, prefixo de `point_id`,
`reference_date_is_synthetic`) separam o rótulo de forma perfeita por construção — foram
mantidas no CSV consolidado apenas como metadado de auditoria, com a coluna
`excluded_from_model_reason` documentando explicitamente por que nunca devem ser usadas como
feature preditiva. Nenhuma dessas colunas entra no vetor `X` do treino em nenhum momento deste
pipeline (v3, v5, ou este v1 oficial).

**Diferença em relação ao caso de leakage do v1→v2**: naquele caso, o leakage estava DENTRO do
vetor de features real do modelo (tipo de geometria correlacionado com fonte de patch estrutural
diferente). Aqui, a separação perfeita existe apenas em colunas de metadado que já eram, por
desenho, excluídas do vetor de treino — a checagem serviu para confirmar isso explicitamente,
não para corrigir um novo vazamento ativo.
