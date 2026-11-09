# v10 — cadeia limpa, doadores contextuais, ciclo rastreável, memória limitada e lacuna social (2026-10-24)

Execução de `docs/PROMPT_CONTINUIDADE_POS_V9_2026_10_24.md`. A v9 fica preservada como histórico (`v9_evolucao_abrangente/`, manifesto intacto). Linhagem nova: `local_runs/transferencia-vulnerabilizados/v10_ciclo_contextual/`, com 57 arquivos no manifesto, fechado depois da última mutação e conferido sem divergência. A verificação independente deu `ok`. Os testes `tests/test_v10_ciclo_contextual.py` e os da v9 passaram (19); o log está em `testes/pytest_v10_v9.txt`.

Tudo aqui é **desenvolvimento em coortes já examinadas**. Nada é confirmação e nenhum peso ativo foi alterado. As reservas (EMSR868/871, MT12 e Bolívia) não foram abertas, ninguém foi contatado e nenhuma revisão foi simulada.

## 0. Núcleo limpo e defeito herdado corrigido

- **Núcleo.** Em todas as frentes o núcleo é o `Qsep_sel_fis` da v9, com λ escolhido e coeficientes ajustados **só** nas 29 unidades de `ajuste_fisico`. O pickle é idêntico ao manifesto da v9 (sha conferido).
- **Papéis separados:**
  - competência: 8 unidades, divididas em A (4) para o estimador e B (4) para a calibração;
  - retenção: 6 unidades, nunca vistas por nenhum ajuste;
  - coortes BR: partição v3 `teste`.
- **Vazamento no pool da v9 F (fato observado, corrigido).** O pool de aquisição tomou a partição do 1º ponto de cada agrupamento. Em Curitiba, 3 FCU têm pontos de teste e de adaptação, e com isso **33 negativos de teste estavam no pool**. Na primeira rodada do ciclo, 280 consultas caíram neles. A rodada foi arquivada em `ii_ciclo_r1_pool_com_teste/`, e a rodada válida (protocolo v10-ii-2) remove esses IDs. A verificação agora mostra 0 consultas em teste, e o replay determinístico dos IDs consultados é idêntico nas 3 coortes. O número de Curitiba da v9 F tinha esse vazamento. O efeito é pequeno (33 negativos de cerca de 199 mil), mas o contrato estava violado.

## 1. Transferência negativa (Frente I) — quanto ganho resta na cadeia limpa

**I.b — região retirada de todo o aprendizado.** λ, padronização e coeficientes foram refeitos sem a região. Cada região foi comparada com 12 controles: 6 remoções aleatórias com o mesmo número de unidades e 6 pareadas por fonte. O Δ abaixo é a média do AUC FCU nas 5 coortes, menos o núcleo.

| região removida | Δ AUC FCU | controles (mín / mediana / máx) | posto entre 12 |
|---|---|---|---|
| África/Oriente Médio (10 unid.) | +0,008 | −0,059 / +0,001 / +0,020 | 9 |
| Américas (8) | +0,000 | −0,025 / +0,009 / +0,022 | 4 |
| Ásia (5) | −0,011 | −0,033 / −0,012 / +0,006 | 7 |
| Europa (6) | −0,003 | −0,033 / −0,010 / +0,014 | 8 |

Na retenção externa, os Δ ficaram entre −0,006 e +0,004, também dentro dos controles. **Na cadeia limpa não há transferência negativa regional distinguível de remover unidades ao acaso.**

**I.a + suplemento — o estimando da v9 (Q fixo + resíduo MLP).** Média de 5 sementes, AUC FCU:
- O **resíduo completo piora o núcleo** em 4 de 5 coortes: RMPA 0,770 contra 0,874, Recife 0,774 contra 0,859, Curitiba 0,831 contra 0,877, Manaus 0,876 contra 0,915. A exceção é a RMR, com 0,839 contra 0,773.
- Tirar África/OM **do resíduo** dá entre +0,017 e +0,043 sobre o resíduo completo: RMPA +0,043, Curitiba +0,040, Manaus +0,030, RMR +0,030, Recife +0,017.
- Os 6 controles pareados em **linhas e positivos removidos** (±25%) ficam quase todos negativos: Curitiba entre −0,184 e +0,008. Os 6 aleatórios por número de unidades ficam entre −0,182 e +0,054.
- Conclusão: a "transferência negativa" vista na v9 existe **dentro do resíduo** e supera os controles pareados. Mas o resíduo sem África continua abaixo do núcleo limpo em RMPA, Recife e Curitiba. O ganho da v9 era a recuperação parcial de um dano causado pelo próprio resíduo, não ganho sobre o Q.
- Não foram pareados mecanismo nem qualidade. A região é o continente pela coordenada mediana, não uma caracterização hidroclimática.

**I.c — doadores contextuais.** O k foi escolhido por k-NN em descritores sem rótulo (p90 HAND, declive, construído, relevo), com deixa-uma-unidade-física-de-fora. k = 15 empatou com k = 20 (0,7604), à frente de 10 (0,7564) e 5 (0,7489).
- Nas 6 unidades de retenção, o contextual **supera todos os físicos em 6 de 6**, por +0,001 a +0,034 (EMSR838: 0,591 contra 0,557). Fica acima do máximo de 8 sorteios de k em 4 de 6.
- Nas coortes BR, **piora o FCU** em Curitiba (0,836 contra 0,877), Recife (0,822 contra 0,859) e RMR (0,732 contra 0,773), e fica abaixo da média do k aleatório. Em RMPA e Manaus fica igual ao núcleo.
- Conclusão: a seleção por contexto ajuda fora do Brasil por margens pequenas. **Não é uma regra para comunidades brasileiras** e não deve ser servida.

**I.e — qualidade da evidência.** Retirar positivos UFO com occurrence_v8 ≥ 75 (178 pontos) ou com recurrence_v9 ≥ 75 (446) muda o AUC em no máximo 0,003. É o mesmo tamanho de 5 retiradas aleatórias de igual número. A recorrência não muda a conclusão da v8.

## 2. Ciclo integrado (Frente II)

O ciclo vai da consulta à revelação, depois ao filtro de suporte (≥5 positivos, ≥5 negativos, ≥3 agrupamentos com positivo), à atualização por offset (τ = 25) e, por fim, ao teste e à retenção.
- Configuração: lotes de 25 até 400 consultas, 6 sementes, 3 coortes, pools "todas" e "fcu".
- Registro por consulta: ID, ordem, escore e posto prévios, estado revelado, grupo e custo. Desconhecidos gastam orçamento.
- Registro por lote: θ, c, motivo, AUC no **teste admissível inteiro** e AUC de retenção (`consultas.parquet`, `lotes.parquet`).

Resultado em 400 consultas, pool "todas", diferenças pareadas por semente:

| | RMPA FCU | RMPA retenção | Manaus | Curitiba |
|---|---|---|---|---|
| margem v9 + filtro − congelado | **+0,052** (6/6) | **−0,029** | 0 (filtro não atinge) | 0 (filtro não atinge) |
| margem v9 + filtro − uniforme + filtro | +0,023 (6/6) | −0,018 | 0 | 0 |
| uniforme + filtro − congelado | +0,029 | −0,010 | 0 | 0 |
| margem com filtro − sem filtro | +0,021 | −0,011 | FCU −0,009 / retenção **+0,078** | 0 |

- **A margem da v9 se reproduz.** No pool FCU da RMPA, margem fica 0,920 e uniforme 0,886, ou +0,034, contra +0,040 na v9.
- **Só a RMPA tem densidade de positivos para o filtro.** Em Manaus e Curitiba, 400 consultas por margem revelam 0 a 1 positivo e o filtro corretamente não atualiza. Em Manaus, o mesmo critério sem filtro atualiza com 5 positivos e **perde 0,078 de retenção**.
- **O filtro não basta.** Em Curitiba, `margem_p90 + filtro` atinge o suporte em 375 consultas e colapsa: FCU 0,628 e retenção 0,605. O critério 5/5/3 é necessário, mas não garante nada.
- **O ciclo aprende mais por consulta onde há positivos, mas não preserva a capacidade.** Toda atualização na RMPA custa de 0,010 a 0,029 de retenção externa.

## 3. Memória realmente limitada (Frente III)

- **Reservatório.** O reservatório guarda z, y, q e o logit **no momento da inserção**; o teste unitário confirma que o logit nunca é recalculado. O aprendiz não tem acesso ao histórico completo.
- **Seleção de M e α.** Feita só nas 8 unidades de competência (desenvolvimento): M = 200 e α = 0,3. Os dois ficaram na **borda da grade** e as diferenças de desenvolvimento são ≤ 0,006, então a seleção é fraca.
- **Desenho.** 3 ordens × 6 sementes, v4 urbano uniforme, B = 400, com filtro.

Diferenças pareadas na etapa final (IC 95% por bootstrap de pares):

| | retenção externa | RMPA FCU | Manaus FCU | Curitiba FCU |
|---|---|---|---|---|
| DER limitado − congelado | **−0,022** [−0,028; −0,016] | +0,017 | +0,015 | +0,020 |
| replay limitado − congelado | −0,023 [−0,032; −0,016] | +0,009 (IC cruza 0) | +0,014 | +0,020 |
| DER limitado − replay limitado | +0,002 [−0,001; +0,005] | +0,008 (IC cruza 0) | +0,002 | 0,000 |
| DER limitado − variante v9 (memória reconstruída) | **−0,005** [−0,008; −0,002] | +0,001 | +0,002 | 0,000 |
| só logits − DER | +0,002 | −0,005 | −0,002 | 0,000 |

- **Com memória honesta, a distilação não supera o replay simples** na retenção.
- A variante v9, que tinha acesso privilegiado ao histórico, retinha 0,005 a mais. A vantagem relatada na v9 incluía esse acesso.
- **Todos os aprendizes perdem cerca de 0,02 de retenção externa**, inclusive a referência conjunta com histórico completo (0,707 contra 0,733). Adaptar ao Brasil custa retenção externa, e a memória limitada não evita isso.

## 4. Competência (Frente IV)

- **Separação.** Os auxiliares foram escolhidos em dobras de unidades físicas: GB monotônico com profundidade 2 e lr 0,03, e MLP com largura 64. O ensemble tem 5 MLP (`ens_mlp_s11..15.pt`). O estimador de competência foi treinado em A e o CRC calibrado em B.
- **Ranking de tranquilizadoras erradas** (positivos ditos "menor") com cobertura de 0,7:
  - retenção externa: margem 1.038, aprendido 1.108, discordância 1.266, aleatório 1.741;
  - RMR: aprendido 283, margem 313, aleatório 881.
  - O estimador aprendido é dominado pelo coeficiente da margem (5,0). **A margem continua sem superação clara**, como na v9.
- **CRC é diagnóstico, não garantia.** λ = 0,089 com α = 0,02. A perda empírica na retenção foi 0,009, mas cobre só cerca de 31% dos pontos. Nas coortes BR a prevalência baixa torna o limite trivial. A garantia exigiria permutabilidade entre B e o destino, e ela não vale.
- **Custo por comunidade** com orçamento de 5% está em `custo_por_comunidade_orc05.csv`. **Revisões concluídas: 0 em todas as linhas.**

## 5. Lacuna observacional nas comunidades (Frente V)

**Recife, episódio de 28/05/2022.**
- **Relatos usados.** 147 relatos de água da SEDEC entre 28/05 e 07/06, sem duplicata de conteúdo (no ano todo, 49 duplicatas foram removidas). A precisão é **de bairro**. Os relatos caem em 35 bairros, 98,6% deles casados com a grade v6 pelo atributo de bairro do DBF de setores IBGE 2022.
- **Triangulação.** Relatos por bairro contra positivos S2 por ponto observável:
  - contagem: Spearman −0,10 (p = 0,32);
  - taxa por 1.000 pontos: −0,13 (p = 0,21).
  - Contra o escore do núcleo: 0,06 e 0,09.
- **Parcela não identificada.** Em 16 dos 35 bairros com relato, a referência S2 não tem nenhum positivo.
- **Independência.** Os relatos não ensinaram nenhum modelo. A janela da SEDEC foi usada na v6 para escolher a data da cena, uma dependência de **data**, não de local.
- **Pontos geocodificados.** São 9 na janela, 8 deles em 28/05, e 5 estão sensíveis à borda de 30 m. O setor de Várzea tem 3 positivos em 558 pontos observáveis e o de Torre tem 4 em 38. Os demais setores têm 0 positivos, e o de Passarinho não está na grade.
- Leitura: **a referência S2 e os relatos não concordam no nível de bairro.** Isso não diz qual dos dois está certo. Ausência de chamado não é seco, e coincidência não é acurácia.

**Curitiba, 28/10 a 12/11/2023.** Há 39 chamados geocodificados harmonizados e **só 1 cai numa FCU**, com 15 negativos e 0 positivos de referência. A fonte não serve para avaliar comunidades.

**Por comunidade FCU** (`por_comunidade_fcu.csv`). As colunas são: dados, referência utilizável, desconhecido, positivos e negativos, consultas da Frente II (somadas sobre 6 sementes × 2 pools), carga de encaminhamento da Frente IV e situação. A tabela abaixo mostra a parcela dos pontos FCU em comunidades **avaliáveis** (≥5 positivos e ≥5 negativos):

| coorte | comunidades | avaliáveis 5×5 | pontos avaliáveis | pontos sem referência | consultas por margem com positivo revelado |
|---|---|---|---|---|---|
| RMPA | 293 | 27 | 19,0% | 5,4% | 366 |
| RMR | 433 | 6 | 5,6% | 20,3% | — (fora do ciclo) |
| Manaus | 236 | 5 | 3,4% | 0% | 0 |
| Recife | 294 | 0 | 0% | 13,2% | — |
| Curitiba | 410 | 0 | 0% | 9,3% | 0 |

**PPS v2 (RMPA), preparado.**
- O plano de análise foi congelado antes de qualquer resposta (`plano_analise_pps.json`). O sha da chave confere e há 0 respostas, então a análise com respostas **não foi executada**.
- Estimativas pelo desenho, a partir do estado S2 já conhecido da chave, para a parcela da população **sem referência**:
  - baixa renda 0,31 (EP 0,08);
  - comparador urbano 0,23 (EP 0,07);
  - FCU 0,075 (EP 0,036).
- É nessa parcela que a revisão humana teria de responder.

**Que parcela da lacuna social foi avaliável:**
- Pelo critério 5×5, entre 0% e 19% dos pontos FCU, conforme a coorte.
- Recife e Curitiba ficam em 0%.
- A triangulação externa foi possível só em Recife, no nível de bairro, e não concordou com a referência.

## 6. Frente VII (representações)

Não executada. Nenhum diagnóstico desta rodada apontou uma hipótese específica que uma nova representação testaria. Fica declarado como não executado, não como resultado negativo.

## Respostas às quatro perguntas

1. **Quanto ganho resta com a cadeia limpa?** Nenhum ganho de seleção de doadores sobre o núcleo nas coortes BR. O ganho da v9 vinha do resíduo, e o resíduo fica abaixo do núcleo.
2. **A seleção de doadores funciona além do resíduo?** Fora do Brasil, levemente (6 de 6 unidades de retenção, até +0,034). No Brasil, não: piora 3 coortes. A remoção regional no refit completo não se distingue de controles aleatórios.
3. **O ciclo aprende mais por consulta preservando capacidade?** Aprende mais por consulta só onde há positivos (RMPA, +0,023 sobre uniforme). Não preserva: custa de 0,018 a 0,029 de retenção. Em 2 de 3 coortes o filtro fica corretamente fechado. Um filtro satisfeito ainda permitiu colapso (Curitiba, p90).
4. **Que parcela da lacuna social foi avaliável?** Entre 0% e 19% dos pontos FCU com 5×5. Os relatos de Recife não concordam com a referência no nível de bairro, e em Curitiba os chamados geocodificados não chegam às FCU.

## Limitações declaradas

- As coortes já foram examinadas em rodadas anteriores. Remoções e pareamentos foram feitos com 6 conjuntos; os efeitos pequenos (≤0,01) não são distinguíveis.
- Os modelos refeitos em I.b e I.e não foram persistidos (são determinísticos a partir das sementes). Foram gravados λ, composição removida e AUC por conjunto. Os pesos persistidos são `ens_mlp_s*.pt` (IV); o ciclo grava θ e c por lote.
- Na Frente V, contagens de relato por bairro não são normalizadas por exposição. A taxa por pontos da grade é só um substituto para isso.

## Arquivos

- Código:
  - `scripts/vulnerabilizados/v10_{comum,fila,i_doadores,i_suplemento,ii_ciclo,iii_memoria,iv_competencia,v_social,z_sintese}.py`;
  - `tests/test_v10_ciclo_contextual.py`.
- Figura: `docs/figuras/v10_ciclo_contextual_2026_10_24.{png,svg}`.
- Linhagem: `local_runs/transferencia-vulnerabilizados/v10_ciclo_contextual/` (protocolos congelados, `execucao_etapas.jsonl`, `verificacao_independente.json`, `manifesto.json`).
