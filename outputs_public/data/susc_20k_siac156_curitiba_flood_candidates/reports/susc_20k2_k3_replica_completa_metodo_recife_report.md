# SUSC-20K2/K3 — Réplica completa do método SEDEC/Recife em Curitiba (SIAC 156)

**Status**: RESULTADO REAL, POSITIVO. Curitiba deixa de ter N=1 e passa a ter **1238 positivos +
119 negativos**, minerados/geocodificados/pareados com o mesmo rigor metodológico usado no
dataset v12 de Recife. N deixa de ser o gargalo do projeto para esta região.

**Escopo**: só a etapa de aquisição de pontos (positivo + negativo). Nenhuma feature física foi
extraída, nenhum modelo foi treinado nesta rodada -- isso é trabalho separado (seção 7).

---

## 1. Por que esta rodada existe

O SUSC-20K (relatório anterior) tinha achado o SIAC 156 como fonte administrativa real
equivalente ao SEDEC de Recife, mas só tinha minerado 4 datas de evento conhecidas (33 pontos
geocodificados). Uma investigação da linhagem completa de Recife (script
`build_recife_seced_geocoding_plan_v1.py` + `qa_recife_seced_geocoding_ready_records_v1.py` +
`run_geocoding.py`) mostrou que o método real de Recife **não** dependia de datas de evento
pré-conhecidas: minerava a categoria estruturada ("alagamento") no cadastro administrativo
inteiro, ano a ano, ao longo de 11 anos (2014-2025), geocodificava tudo que passasse no QA, e
aceitava em bloco (strong+medium) como positivo -- sem revisão manual individual por ponto. Esta
rodada replica exatamente essa lógica pro SIAC 156.

## 2. Etapa 1 — Mineração multi-ano (positivos)

Baixados 4 arquivos ano-completo do SIAC 156 via o endpoint AJAX real do portal
(`/ConjuntoDado/DownloadArquivos/`) + espelho UFPR: **2023, 2024, 2025 (ano-completo,
`156_-_Base_de_Dados_2023.csv` da UFPR e `YYYY-12-31_...csv` do portal, que são extratos
ano-a-data acumulados) e 2026 (parcial, até 29/07)**. 2.580.703 solicitações citywide no total.

Escopo temporal (2023-2026, não 2014-2023 como Recife): decisão explícita, documentada aqui, não
omissão. Motivos: (a) o registro de eventos-catálogo `v20i` do projeto só cobre 2023-2026 --
minerar 2016-2022 sem esse catálogo de referência perde a checagem cruzada independente que os
4 anos atuais tiveram (seção 3); (b) cada arquivo ano-completo do SIAC 156 é 6-10x maior que o
equivalente de Recife (118-150 MB vs. 10-16 MB) -- oito anos custaria ~1 GB de download bruto,
desproporcional ao ganho esperado. Se um N ainda maior for necessário no futuro, os anos 2016-2022
estão disponíveis no espelho UFPR (rollups anuais confirmados) como próximo passo natural.

`scripts/extract_flood_complaints_siac156.py` (já existente, reaproveitado sem alteração)
filtrado sobre os 4 arquivos inteiros: **1238 reclamações hidrológicas reais** (173 em 2023, 292
em 2024, 569 em 2025, 204 em 2026-parcial). Categoria vem estruturada no próprio SIAC 156
(`Assunto`/`Subdivisao` = "Alagamentos"/"Enchentes, inundações ou alagamentos"), não texto livre
como no SEDEC -- não precisou de lista de termos/bloqueio de deslizamento por keyword: confirmado
por varredura direta que "Deslizamento de terra" é categoria SIAC estruturalmente separada
("Segurança de edificações e imóveis"), 0 contaminação.

## 3. Confirmação independente: pico real nos 4 dias de evento conhecidos (herdado do SUSC-20K)

Já demonstrado no relatório anterior e não refeito aqui: médias de 61,5 reclamações/dia nos 4 dias
de evento FORTE do `v20i` vs. mediana de fundo 2,0 em 40 dias sem evento catalogado -- pico de
10-30x, evidência de que a categoria captura eventos reais.

## 4. Etapa 2 — QA/dedup

`scripts/qa_dedup_siac156.py` (novo, espelha `qa_recife_seced_geocoding_ready_records_v1.py`):
deduplicação por (logradouro normalizado + bairro + data + categoria), reconhecimento de bairro
contra os **75 bairros oficiais** (extraídos do próprio SIAC 156 -- ano de 2025 tem exatamente 75
bairros distintos, bate com a contagem oficial do município), checagem de rua genérica/vazia.

Resultado: **1238/1238 `ready_for_geocoding`** -- 0 `needs_manual_cleanup`, 0 `reject`. Muito mais
limpo que o QA de Recife (71 de 289 prontos, resto precisando limpeza manual) porque o `Logradouro`
do SIAC 156 é campo estruturado de rua, sem mistura com texto livre de descrição de ocorrência.
1160 chaves únicas (endereço+bairro+data+categoria) dentre as 1238 linhas -- duplicação real
modesta (mesma reclamação categorizada duas vezes, ex.: "Drenagem/Alagamentos" +
"Proteção ao cidadão - GM/Enchentes..." pro mesmo endereço/data).

## 5. Etapa 3 — Geocodificação em massa (positivos)

`scripts/geocode_nominatim.py` (já existente, reaproveitado): 867-889 endereços únicos (variação
por diferença de capitalização na query), Nominatim ao vivo, 1,1s entre chamadas, mesma régua
strong/medium/failed do relatório SEDEC. Rodada levou ~35 execuções encadeadas (limite de tempo
por chamada de shell), cache atômico garantindo retomada sem perda de progresso nem duplicação de
consulta.

**Resultado: 1238/1238 geocodificados (100%), 1121 strong + 117 medium, 0 falhas** (1 falha de
timeout de rede na primeira passada, resolvida com nova tentativa). Taxa de sucesso muito acima
dos 48,8% de Recife -- lá o `Endereco` tinha número de casa mascarado na fonte
(`relatorio_geocodificacao_sedec.md:24`); o `Logradouro` do SIAC 156 nunca teve número pra começo
de conversa (rua limpa por desenho do sistema), então nunca falha por ofuscação.

747 coordenadas únicas dentre os 1238 pontos geocodificados -- mesma proporção de repetição
espacial que Recife (105 únicas em 141 positivos): ruas cronicamente alagáveis aparecem em mais de
um evento ao longo dos 4 anos. Isso é sinal real (recorrência), não duplicata de processamento.

## 6. Etapa 4 — Dataset final de positivos

`scripts/build_final_dataset.py` (novo): aceita em bloco strong+medium como positivo, **mesma
regra final de Recife** (confirmada na investigação: `stage1_build_points_v2.py:124`, "Recife
POSITIVES: same geocoded points as v1 (unchanged)" -- sem segunda camada de verificação manual
individual apesar do QA ter recomendado isso pra Recife). Cada linha carrega `qa_record_id`,
`confidence_tier`, `nominatim_display_name`, `nominatim_osm_id`, `occurrence_phenomenon` (nota
explícita de exclusão de deslizamento) -- mesmo padrão de proveniência do
`dataset_eventos_features_v12_final.csv`.

**`v20k2_dataset_positivos_curitiba_siac156_v1.csv`: 1238 linhas, `region=Curitiba`, `label=1`.**

## 7. Etapa 5 — Amostragem de ponto negativo (SUSC-20K3)

Documento `revp_criterio_ponto_negativo_recife_e_replicacao_curitiba_petropolis.md` travava
explicitamente a amostragem de negativo em Curitiba enquanto N=1 positivo ("o pareamento por
bairro produziria negativos em exatamente 1 bairro"). Com N=1238 (73 bairros distintos), a
trava foi removida -- as 4 verificações V1-V4 da seção 4.1 foram respondidas:

| # | Verificação | Resposta pra Curitiba |
|---|---|---|
| V1 | Cadastro de ocorrências não-hidrológicas, multianual, com endereço/data? | Sim -- o próprio SIAC 156, mesma fonte dos positivos |
| V2 | Coordenada própria ou precisa geocodificar? | Precisa geocodificar (mesmo pipeline Nominatim) |
| V3 | Categorias causalmente independentes de chuva **em Curitiba**? | Lista construída do zero, ver `scripts/negative_categories_curitiba.py` |
| V4 | Camada oficial de bairro pro pareamento? | Sim -- 75 bairros extraídos do SIAC 156 |

### 7.1 Categorias negativas (V3) -- não herdadas de Recife

`scripts/negative_categories_curitiba.py` documenta, categoria por categoria, o motivo causal:
29 pares (Assunto, Subdivisão) aceitos como independentes de chuva em Curitiba (Iluminação
Pública, Trânsito/fiscalização, Cartão-transporte, IPTU, Animais, Coleta de resíduos, Praças,
Sinalização, etc.) e 7 excluídos explicitamente por dependerem de chuva:

- **Drenagem** (todo o grupo) -- é o próprio sistema causal da enchente.
- **Pavimentação** (todo o grupo) -- dano de pavimento se agrava com chuva/erosão.
- **Árvore ou galho caído** (com/sem bloqueio) -- queda por temporal/vento.
- **Risco de Leptospirose/Roedores em bueiro** -- ligado a água parada pós-enchente.
- **Limpeza – dengue** -- combate a foco ligado a água parada, geralmente pós-chuva.
- **Cabos danificados** -- ambíguo (pode ser queda de galho em temporal), excluído por precaução.

### 7.2 Mineração + pareamento geográfico (já nascendo pareado, sem remendo posterior)

`scripts/mine_negative_candidates_siac156.py`: filtra pela allowlist de categoria **e** exige que
o bairro já tenha positivo real (condição 5) -- a correção que Recife só aplicou depois no v9
(caiu de 39 bairros dispersos pra 3 em comum com positivo) foi incorporada desde a primeira
rodada aqui. Amostragem determinística por hash estável (sem semente aleatória), 40
candidatos/ano × 4 anos = **160 candidatos negativos brutos** (137 endereços únicos), usando os
73 bairros com positivo real.

### 7.3 Geocodificação + checagem de colisão

Mesmo `geocode_nominatim.py`: **160/160 geocodificados, 137 strong + 23 medium, 0 falhas.**
Checagem de colisão espacial contra os 1238 positivos (raio 30 m, mesmo critério de Recife):
**41 colisões removidas** (mesma rua tem reclamação de enchente E de outra categoria) -- **119
negativos finais**.

**`v20k3_dataset_negativos_curitiba_siac156_v1.csv`: 119 linhas, `region=Curitiba`, `label=0`,
`negative_source_type=siac156_curitiba_non_hydrological_category_v1`.**

### 7.4 Proporção positivo:negativo

**1238:119 (≈10,4:1)**, mais desbalanceada que a de Recife (154:124 ≈ 1,24:1). Isso é efeito do
teto conservador de 40/ano escolhido nesta rodada, não limite de dado disponível -- o pool bruto
de categorias negativas válidas e pareadas por bairro é muito maior que 160 (Curitiba tem alto
volume de solicitações 156 em geral). Aumentar `--max-por-ano` é a alavanca direta se um conjunto
mais balanceado for necessário; não decidido nesta rodada.

## 8. O que isso muda pro projeto

Curitiba tinha N=1 positivo (pixel MODIS em Juvevê, 2022) e N=0 negativo. Agora tem **1238
positivos + 119 negativos**, ambos com proveniência completa e critério causal documentado.
Isso:

1. Resolve o gargalo de N que bloqueava qualquer feature -- mesmo no piso conservador EPV≥30, o
   grupo menor (119 negativos) sustenta ~4 features; folga muito maior que isso é alcançável
   ajustando a amostragem de negativo.
2. Muda o status de Curitiba de "achado isolado sem contexto estatístico" pra "dataset com escala
   comparável (e em N absoluto, maior) que o único modelo real do projeto".
3. **Não** torna Curitiba pronta pra treino ainda -- faltam extração de features
   físico-hidrológicas (SUSC-20B equivalente: HAND/TWI já tem raster pronto em
   `local_runs/susc_20g_hand_twi_dinfinity_generico/curitiba/`; falta chuva CHIRPS e MapBiomas) e
   modelagem/validação estatística (SUSC-20C, Firth). Isso é decisão e trabalho de uma próxima
   rodada, não executado aqui.

## 9. Scripts e testes

| Arquivo | Papel |
|---|---|
| `scripts/negative_categories_curitiba.py` | allowlist de categorias negativas causalmente justificadas pra Curitiba (V3) |
| `scripts/mine_negative_candidates_siac156.py` | mineração + pareamento geográfico de candidatos negativos, amostragem determinística |
| `scripts/build_final_dataset.py` | monta dataset final de positivos (aceite em bloco strong/medium) |

`tests/test_susc_20k2_siac156_pipeline_completo.py` (12 testes) +
`tests/test_susc_20k3_siac156_negativos.py` (9 testes): **21 testes novos, todos passando**
(mais os 18 já existentes do SUSC-20K original -- 39 no total pro pipeline SIAC 156 completo).

## 10. Limitações reconhecidas

- Escopo 2023-2026 (não 2014-2023 como Recife) -- decisão documentada na seção 2, não omissão.
- Proporção positivo:negativo desbalanceada (10,4:1) por teto conservador de amostragem.
- Repetição espacial (747 coordenadas únicas em 1238 positivos) não foi colapsada -- mantida
  como está porque Recife também manteve (105/141), e colapsar descartaria a informação temporal
  de recorrência sem justificativa causal pra fazê-lo.
- Petrópolis segue sem fonte equivalente (documentado no SUSC-20K original).
- Nenhuma feature física foi extraída; nenhum modelo foi treinado ou validado.
