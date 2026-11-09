# Fechamento científico operacional

**Gerado de** `scripts/metodo/relatorio_de_fechamento.py` em 2026-10-08T01:36:20.533866Z. **Não editar à mão.**

**Não é.** prova de que a ciência acabou: software correto não é validade observacional. Cada seção diz o que vale para qual uso e o que segue bloqueado.

## 0. Validade das avaliações: não há reserva intacta

O lacre de 01/10 (13 eventos GFD, `generalizacao_protegida` r2) está **aberto**: aberto por `curva_de_aprendizado:gfd_planicie` (13 eventos); `politica_retrospectiva` (13 eventos). Causa: a mesma ordenacao por sha256 do evento foi usada para o lacre (20%) e para os testes de 02/10 (25%): o primeiro esta contido no segundo.

Além disso, sinais_marginais mediu as relacoes com todos os eventos, e o lacre foi aberto em 02/10; so evento posterior ao congelamento pode confirmar. **Toda avaliação abaixo sobre o ds06 é exploratória.** Uma partição nova criada agora não seria intacta: os rótulos já informaram escolhas.

Conjunto conhecido congelado em `local_runs\reservas-examinadas\conhecidos_2026_10_03.json` (`sha256:cbe3da243deadbaf…`): gfd 69, cems 34, sen1floods11 11, uk 401, ufo 18 eventos. Só evento ausente dessa lista pode compor teste `confirmatorio_prospectivo` (`reservas_examinadas.regime`).

Os números das seções 6 e 7 foram medidos nesse lacre aberto; ficam preservados como medidos, com esse regime.

## 1. Admissão científica

Regra única de uso (`admissao_cientifica.py` v1) sobre 1.641.356 linhas da tabela única. Consumidores que ajustam ou medem chamam `exigir` e falham com linha não autorizada: `celula_para_treino`, `generalizacao_protegida`, `curva_de_aprendizado`, `ctx04`, política retrospectiva.

| uso | linhas autorizadas |
|---|---:|
| `exploracao` | 1.641.356 |
| `treino` | 763.155 |
| `avaliacao_espacial` | 759.301 |
| `avaliacao_temporal` | 513.525 |
| `avaliacao_externa` | 513.530 |
| `afirmacao` | 513.530 |

Fontes recusadas como referência (motivo gravado no módulo):

- `opera_dswx_hls_serra`: 03/10/2026: em serra brasileira, controle seco de ago/2024 deu 278 km2 de agua nova contra 103 km2 no evento de mai/2024; mascara de sombra nao muda
- `ufrgs_rs2024_observada`: IoU 0,23-0,30 com a agua do S2 de 06/05/2024; trecho so-UFRGS tem HAND p90 15 m: extensao sobre o MDE, nao delineacao observada
- `gfm_copernicus`: nao auditada: algoritmos do GFM usam mascara de HAND -- circular para testar HAND; exclusoes do radar precisam virar classe desconhecida
- `geoglows_vazao`: simulacao forcada por ERA5; no Taquari atrasou o pico de 2024 em 5 dias e nao viu set/2023: nao e observacao
- `mapbiomas_rs2024_area_atingida`: nota tecnica MapBiomas (jun/2024): deteccao de mudanca S1/S2/Planet, depois exclui pixel em encosta por limiar de HAND e separa deslizamento por declive de 7% -- circular para testar HAND e declive; inclui solo saturado e mancha modelada de POA (Possanti 2024); sem metrica de acuracia. So exploracao
- `zenodo_17177153`: titulo promete mapeamento de areas inundadas RS 2024, mas o pacote publicado contem ganho/perda de vegetacao por hexagono de 2 km: nao ha delineacao de cheia
- `zenodo_11151528`: limiar de NDWI de um sensor (S2) que inclui area umida e agua permanente, publicado so como imagem (PNG/PDF), sem vetor: nao e independente do que o projeto computa e nao e utilizavel como camada

Datas resolvidas na rodada de 02/10 (texto fixo, não lido de artefato): 15.566 linhas CEMS com a cena do produto (`cena_do_produto_ems`), 5 seguem sem data e ficam fora de todo uso temporal (EMSR861 AOI22/23); 376.722 linhas GFD datadas pelo catálogo de eventos.

## 2. Impacto e revalidação

Cada gerador exige a montante atual e carimba ao fim; histórico append-only em `local_runs/rastreio-de-impacto/carimbos.jsonl`.

| nó | estado | motivos |
|---|---|---|
| `tabela_unica` | `atual` | — |
| `contrato` | `atual` | — |
| `admissao` | `atual` | — |
| `dataset_de_treino` | `atual` | — |
| `celulas_de_contexto` | `atual` | — |
| `sinais_marginais` | `atual` | — |
| `relacoes` | `atual` | — |
| `conhecimento` | `atual` | — |
| `curva_de_aprendizado` | `atual` | — |
| `contexto_hidrologico` | `atual` | — |
| `modelos_servidos` | `sem_carimbo` | nunca revalidado sob o rastreio |
| `afirmacao_dos_servidos` | `atual` | — |
| `verdade_observacional` | `desatualizado` | codigo_mudou:scripts/metodo/verdade_observacional.py |
| `generalizacao_v2` | `herdado` | a_montante_nao_atual:verdade_observacional |
| `e2c_identidade_canonica` | `herdado` | a_montante_nao_atual:verdade_observacional |
| `recusa_por_dominio_v2` | `herdado` | a_montante_nao_atual:verdade_observacional |
| `politica_sob_orcamento_v2` | `herdado` | a_montante_nao_atual:verdade_observacional |
| `posicao_topografica` | `atual` | — |
| `ms1_posicao_topografica` | `herdado` | a_montante_nao_atual:verdade_observacional |
| `ms2_quadro_do_negativo` | `herdado` | a_montante_nao_atual:verdade_observacional |
| `ms3_decomposicao_do_quadro` | `herdado` | a_montante_nao_atual:verdade_observacional |
| `posicao_no_ponto` | `atual` | — |
| `auditoria_elevacao_local` | `sem_carimbo` | nunca revalidado sob o rastreio |
| `ms4_bracos_de_desenvolvimento` | `sem_carimbo` | nunca revalidado sob o rastreio |
| `ms5_recusa_decomposta` | `sem_carimbo` | nunca revalidado sob o rastreio |
| `tc01_territorio_consultavel` | `atual` | — |
| `br03_preparo_sem_rotulo` | `desatualizado` | codigo_mudou:scripts/metodo/verdade_observacional.py |
| `ms6_quadro_de_treino_para_consulta` | `sem_carimbo` | nunca revalidado sob o rastreio |
| `queda_entre_fontes` | `herdado` | a_montante_nao_atual:verdade_observacional |
| `terreno_versus_contexto` | `atual` | — |

## 3. Fluvial: contexto do evento

Triagem antes de baixar: 71 eventos fluviais brasileiros, 101 pares evento×estação elegíveis (postos em barragem/açude excluídos), 40 escolhidos por diversidade: 40 estações, 19 eventos, territórios centro_oeste, nordeste, norte, sul.

Bacias: {'bacia_conferida': 33, 'bacia_divergente': 7}. Pares com pico completo: 24 de 40. Pontos ligados a uma única estação: 11.325.

- **GFD (protocolo único de amostragem)**: `not_supported` — 11 eventos, 16 pares. Terreno {'auc_agrupada': 0.8303, 'auc_media_dentro_do_par': 0.7711, 'brier': 0.1643, 'log_loss': 0.4937}; terreno+contexto {'auc_agrupada': 0.7309, 'auc_media_dentro_do_par': 0.7783, 'brier': 0.1995, 'log_loss': 0.5781}; diferença IC95 por evento {'auc_agrupada': [-0.2069, 0.0408], 'auc_media_dentro_do_par': [0.0013, 0.0137], 'brier': [-0.0123, 0.0939], 'log_loss': [-0.031, 0.2247]}.
- **Todas as fontes (sensibilidade; mistura protocolos)**: `not_supported` — 12 eventos, 18 pares. Terreno {'auc_agrupada': 0.8211, 'auc_media_dentro_do_par': 0.7698, 'brier': 0.1628, 'log_loss': 0.4984}; terreno+contexto {'auc_agrupada': 0.6714, 'auc_media_dentro_do_par': 0.7707, 'brier': 0.2091, 'log_loss': 0.6455}; diferença IC95 por evento {'auc_agrupada': [-0.3392, 0.0229], 'auc_media_dentro_do_par': [-0.0002, 0.0022], 'brier': [-0.0041, 0.1305], 'log_loss': [-0.0039, 0.3807]}.

## 4. Pluvial: negativos observados

FloodNet público adquirido: 3.422 eventos, 491 sensores (424 sem maré). Negativos observados: **0** — catalogo publico tem so eventos; sensor sem evento e ausencia de registro -- seco, desligado ou em manutencao sao indistinguiveis sem a serie continua e o funcionamento.

**Bloqueado por evidência pública insuficiente**, não por pedido: o catálogo continua só positivo e nenhum negativo é fabricado. A busca delimitada de fontes abertas está na seção 12.

## 5. Serra: referência independente

Protocolo OPERA com controles secos (regra fixada antes: evento ≥ 3.0× o maior controle):

- `HLS_RS`: `recusado` — maior controle 278.4 km², razões {'RS_set2023': 0.29, 'RS_nov2023': 0.03, 'RS_mai2024': 0.37, 'RS_jun2025_HLS': 0.08}
- `HLS_SP`: `candidato_a_adjudicacao` — maior controle 3.4 km², razões {'SP_fev2023': 3.65}
- `S1_RS`: `candidato_a_adjudicacao` — maior controle 0.1 km², razões {'RS_jun2025_S1': 19.0}

Candidato a adjudicação não é rótulo: libera revisão humana, nada além.

Pacote de adjudicação: 5 eventos prontos, bacias Cai, Litoral Norte SP, Piabanha, Taquari-Antas; amostra estratificada por relevo, nunca por HAND; classe padrão `desconhecido`.

**Bloqueado por revisão humana:** adjudicacao humana: rotulo incerto exige revisao, nao inferencia automatica.

## 6. Hipóteses e aprendizado

Hipóteses na versão `sha256:7feb6989dacc3b71`: 5 sustentadas — FLUVIAL_ENXURRADA·elevation_local_m·marginal; FLUVIAL_ENXURRADA·hand_m·marginal; FLUVIAL_ENXURRADA·slope_deg·marginal; FLUVIAL_ENXURRADA·twi_dinf·marginal; FLUVIAL_ENXURRADA·hand_m·condicional_ao_vetor. Estimandos medidos: marginal e condicional; contribuição individual e causalidade não medidas.

- curva `gfd_planicie`: com o mesmo volume (~9576 linhas), 4 eventos dao AUC 0.768 [0.7237; 0.7954] e 24 eventos dao 0.78 [0.7734; 0.7860]; multiplicar os pontos dos mesmos 4 eventos leva de 0.7491 a 0.768; todos os eventos inteiros, 0.7795 [0.7795; 0.7795]
- curva `cems_planicie`: com o mesmo volume (~3449 linhas), 4 eventos dao AUC 0.7941 [0.7061; 0.8215] e 25 eventos dao 0.7852 [0.7790; 0.7975]; multiplicar os pontos dos mesmos 4 eventos leva de 0.7656 a 0.7941; todos os eventos inteiros, 0.7864 [0.7864; 0.7864]

As faixas da curva são dispersão entre ORDENS de treino com o teste fixo, não incerteza sobre os eventos de teste: 0,780 contra 0,768 não é melhora demonstrada. A seção 8 separa as duas.

## 7. Avaliação da política

Retrospectiva, mesmo orçamento (40.000 linhas), mesma base e teste: AUC protegida lacuna 0.7939, fixa 0.7825, aleatória (mediana) 0.7862; percentil da lacuna entre 30 aleatórias: 1.0.
- reserva `r1`: lacuna 0.7656, fixa 0.7656, aleatória 0.7656, percentil 1.00 — **orçamento não restringiu a escolha: problema de desenho, não resultado**
- reserva `r2`: lacuna 0.7896, fixa 0.7896, aleatória 0.7896, percentil 1.00 — **orçamento não restringiu a escolha: problema de desenho, não resultado**
- reserva `r3`: lacuna 0.7805, fixa 0.7801, aleatória 0.7809, percentil 0.23
- reserva `r4`: lacuna 0.7824, fixa 0.7835, aleatória 0.7822, percentil 0.63

Das 2 reservas em que o orçamento restringiu, a política por lacuna ficou no topo das aleatórias em 0. **Sem evidência de vantagem consistente** da política por lacuna; a vantagem da primeira reserva não se repetiu. Leitura descritiva: aquisição simulada sobre o acervo em disco, não custo de rede real.

## 8. Generalização por evento (exploratória)

Validação cruzada agrupada por evento (cada evento testado uma vez; tiles do teste fora do treino), mesmas linhas para os 5 modelos, critérios com hash `sha256:09846b5e3c859319…` gravados antes. Regime: **exploratorio**.

### GFD — 69 eventos, 354.063 linhas, 5 dobras.

| modelo | AUC agrupada | AUC mediana dentro do evento | AP | Brier |
|---|---:|---:|---:|---:|
| `M0_regra_hand` | 0.7443 | 0.7441 | 0.464 | — |
| `M1_logistica_terreno` | 0.7867 | 0.8006 | 0.5543 | 0.171 |
| `M2_logistica_tcs` | 0.7471 | 0.779 | 0.4935 | 0.1837 |
| `M3_gbm_terreno` | 0.8009 | 0.8215 | 0.574 | 0.1663 |
| `M4_gbm_tcs` | 0.7716 | 0.79 | 0.523 | 0.1817 |

| comparação | eventos | a favor | mediana dentro | IC95 média dentro | p | Δ agrupada [IC por evento] | veredito |
|---|---:|---:|---:|---|---:|---|---|
| aprender pesos supera a regra fisica de uma variavel? | 69 | 64 | 0.048 | [0.0364; 0.0526] | 0.0 | 0.0424 [0.0297; 0.0558] | **ganho** (a2/a3: ganho, ganho) |
| nao linearidade acrescenta ao terreno? | 69 | 48 | 0.0191 | [0.0086; 0.0239] | 0.0001 | 0.0142 [0.0022; 0.0262] | **ganho** (a2/a3: ganho, indecidivel) |
| cobertura/solo acrescentam a logistica? | 69 | 18 | -0.0131 | [-0.0401; -0.0173] | 1e-05 | -0.0396 [-0.0555; -0.027] | **perda** (a2/a3: perda, perda) |
| cobertura/solo acrescentam ao gbm? | 69 | 19 | -0.0144 | [-0.0292; -0.0114] | 7e-05 | -0.0292 [-0.0422; -0.0168] | **perda** (a2/a3: perda, perda) |

Eventos com AUC dentro < 0,6 para a logística de terreno: 0. Piores: GFD:4325 (centro_oeste, 6349 linhas, prev. 0.461) 0.683; GFD:4335 (norte, 3583 linhas, prev. 0.083) 0.705; GFD:4326 (norte, 3163 linhas, prev. 0.045) 0.706; GFD:3007 (norte, 3399 linhas, prev. 0.029) 0.712; GFD:4220 (norte, 4946 linhas, prev. 0.321) 0.715.

AUC dentro do evento (M1) por território: brasil_fora_da_malha 0.825 (n=1); centro_oeste 0.759 (n=7); externo:gfd 0.801 (n=4); nordeste 0.801 (n=3); norte 0.799 (n=28); sudeste 0.874 (n=3); sul 0.82 (n=23).

### CEMS — 13 eventos, 20.975 linhas, 13 dobras. Um evento por dobra: as atribuições a2/a3 são idênticas por construção e não medem sensibilidade.

| modelo | AUC agrupada | AUC mediana dentro do evento | AP | Brier |
|---|---:|---:|---:|---:|
| `M0_regra_hand` | 0.7748 | 0.7775 | 0.6277 | — |
| `M1_logistica_terreno` | 0.7645 | 0.7512 | 0.6443 | 0.1927 |
| `M2_logistica_tcs` | 0.7654 | 0.7567 | 0.6429 | 0.1912 |
| `M3_gbm_terreno` | 0.7822 | 0.7878 | 0.662 | 0.1882 |
| `M4_gbm_tcs` | 0.7848 | 0.7889 | 0.6704 | 0.1894 |

| comparação | eventos | a favor | mediana dentro | IC95 média dentro | p | Δ agrupada [IC por evento] | veredito |
|---|---:|---:|---:|---|---:|---|---|
| aprender pesos supera a regra fisica de uma variavel? | 12 | 5 | -0.0046 | [-0.0125; 0.0145] | 1.0 | -0.0102 [-0.0286; 0.0087] | **indecidivel** (a2/a3: indecidivel, indecidivel) |
| nao linearidade acrescenta ao terreno? | 12 | 11 | 0.0305 | [0.0201; 0.0504] | 0.00146 | 0.0176 [0.0011; 0.0378] | **ganho** (a2/a3: ganho, ganho) |
| cobertura/solo acrescentam a logistica? | 12 | 8 | 0.0083 | [-0.0045; 0.02] | 0.38037 | 0.0008 [-0.0117; 0.017] | **indecidivel** (a2/a3: indecidivel, indecidivel) |
| cobertura/solo acrescentam ao gbm? | 12 | 6 | -0.0024 | [-0.016; 0.0109] | 0.96973 | 0.0027 [-0.0207; 0.0193] | **indecidivel** (a2/a3: indecidivel, indecidivel) |

Eventos com AUC dentro < 0,6 para a logística de terreno: 1. Piores: CEMS:EMSR754 (externo:cems, 1906 linhas, prev. 0.46) 0.518; CEMS:EMSR796 (externo:cems, 235 linhas, prev. 0.506) 0.673; CEMS:EMSR851 (externo:cems, 2398 linhas, prev. 0.492) 0.721; CEMS:EMSR847 (externo:cems, 3813 linhas, prev. 0.413) 0.734; CEMS:EMSR778 (externo:cems, 1647 linhas, prev. 0.501) 0.735.

AUC dentro do evento (M1) por território: externo:cems 0.74 (n=11); sul 0.883 (n=1).

### Território reservado (GFD)

Território inteiro fora; nenhum evento que o toque e nenhum tile dele no treino.

| território | eventos teste/treino | AUC agrupada M0 / M1 / M3 | M1 vs M0 (veredito) |
|---|---|---|---|
| brasil_fora_da_malha | 61/8 | 0.7851 / 0.8276 / 0.8215 | ganho |
| centro_oeste | 44/25 | 0.7123 / 0.7671 / 0.779 | ganho |
| externo:gfd | 11/58 | 0.7191 / 0.7472 / 0.7646 | ganho |
| nordeste | 7/62 | 0.8344 / 0.8371 / 0.88 | not_testable |
| norte | 32/37 | 0.7054 / 0.771 / 0.8076 | ganho |
| sudeste | 10/59 | 0.8601 / 0.8757 / 0.9227 | not_testable |
| sul | 28/41 | 0.8062 / 0.8364 / 0.8553 | ganho |

### Diversidade × volume com as duas incertezas separadas (GFD)

Mesmo volume: todos os eventos subamostrados contra 4 eventos inteiros. A faixa entre ordens é incerteza de *quais* eventos treinam; o IC é incerteza dos *eventos de teste*.

| reserva | eventos teste | linhas por braço | Δ AUC mediana | faixa entre ordens | IC95 eventos de teste | ordens a favor |
|---|---:|---:|---:|---|---|---:|
| v1 | 17 | 2480 | 0.015 | [-0.0173; 0.0941] | [0.0148; 0.0297] | 15/19 |
| v2 | 17 | 3907 | 0.0111 | [-0.0159; 0.0592] | [0.0083; 0.0214] | 18/20 |
| v3 | 17 | 5845 | 0.0099 | [-0.0036; 0.0238] | [0.0064; 0.0108] | 16/20 |

### Transferência entre instrumentos

| treino → teste | eventos | AUC agrupada M0 / M1 / M3 | AUC mediana dentro M0 / M1 / M3 |
|---|---:|---|---|
| gfd → cems | 13 | 0.7748 / 0.7788 / 0.7513 | 0.7775 / 0.7643 / 0.7637 |
| cems → gfd | 69 | 0.7443 / 0.7856 / 0.7594 | 0.7441 / 0.7882 / 0.759 |

Identidade entre as fontes (`verdade_observacional`): GFD 2002-01-15–2018-11-07, CEMS 2023-05-16–2026-03-05; pares GFD×CEMS equivalentes: nenhum. A transferência é de instrumento e de época ao mesmo tempo, sem cheia em comum.

- SAR como preditor: not_testable como preditor: a agua observada no evento e o alvo, nao contexto pre-evento; usa-la seria vazamento.
- Embedding de terreno: not_testable: o encoder da frente B so tem embedding dos chips do Sen1Floods11, nao destes pontos; comparar exigiria extrair embedding por ponto com o mesmo protocolo.
- Contexto hidrológico: fora deste experimento: ctx04 cobre 11 eventos GFD e ficou not_supported; incluir aqui reduziria todas as linhas a esses eventos.

## 8b. Teste confirmatório prospectivo (GFD, leitura única)

Regime **confirmatorio_prospectivo**: 16 eventos GFD fora de `conhecidos_2026_10_03.json`, sem tile em comum com o treino (69 eventos, dataset v7). Plano `sha256:e78a5bbd6b5edfbd…`, seleção `sha256:ecce4bf486ad823b…`, ressalvas gravadas antes da leitura. Os eventos ficaram registrados como examinados: não confirmam de novo.

| comparação | eventos | a favor | mediana dentro | IC95 média dentro | p | Δ agrupada [IC] | veredito |
|---|---:|---:|---:|---|---:|---|---|
| M1_logistica_terreno × M0_regra_hand | 16 | 13 | 0.0409 | [0.0091; 0.0487] | 0.01099 | 0.0409 [0.0261; 0.0566] | **ganho** |
| M3_gbm_terreno × M1_logistica_terreno | 16 | 8 | 0.0063 | [-0.0555; 0.0215] | 0.66855 | -0.0285 [-0.0867; 0.0209] | **indecidivel** |

**Não é:** nao e validacao no Brasil (os eventos novos sao de fora dele); nao testa cobertura/solo; uma execucao so. Os eventos são de países vizinhos: o ganho vale para evento novo e território novo da América do Sul, não para o Brasil.

## 9. Política sob orçamento restritivo (simulação retrospectiva, exploratória)

8 reservas, aquisição sequencial, custo por evento = 1 + linhas/5000 (anotação 0: rótulo do produto GFD), 30 trajetórias aleatórias por reserva e orçamento, regressão medida no CEMS (contexto externo). Pré-registro `sha256:18484aead15d382c…`. Prioridade por lacuna é heurística, não política aprendida.

| orçamento | reservas restritivas | lacuna − mediana aleatória | reservas a favor | p | ganho sobre a base | regressão externa | veredito |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4.0 | 8/8 | 0.0027 | 5 | 0.1094 | 0.0048 | 0.0062 | **indecidivel** |
| 8.0 | 8/8 | -0.0011 | 3 | 0.7422 | 0.0069 | 0.0052 | **indecidivel** |
| 16.0 | 3/8 | — | — | — | — | — | `not_testable` |

Preservado: a conclusao de 02/10 -- sem vantagem consistente da lacuna sob 40.000 linhas -- continua valendo para aquele desenho.

## 10. Cadeia neural (frente A)

- **H1** pré-treino curto × aleatório, 16 sementes, época pela validação: 7/16, mediana -0.006337, p postos 0.29785 → **INDECIDIVEL** (relevância 0,025 declarada antes; não alterada). Amplitude entre sementes aleatório 0.068097 × pré-treino 0.145924: a 'loteria' era o ponto de parada, não a inicialização.
- **H2** crescente × fixo, orcamento pareado: 5/6, p 0.21875 → **INDECIDIVEL** (releitura).
- **H2** crescente × fixo, orcamento desigual: 4/10, p 0.8457 → **INDECIDIVEL** (releitura).
- **H3** limiar de recusa 0,25 (validação) no teste, leitura única: IoU sem água permanente 0.448927 → 0.591549 com cobertura 0.555556; ordena: True.

Hipóteses abertas, não falhas comprovadas: máscara sem token (0 = escondido = sem leitura = média) e deriva de normalização pré-treino → ajuste fino (~0,16 desvio). Nenhuma decisão prática depende delas hoje; pré-treino longo não executado por isso. Detalhe em `docs/CIENCIA_DO_APRENDIZADO_2026_10_03.md`.

## 11. A cadeia de aprendizado que de fato executa

| elo | o que existe | evidência |
|---|---|---|
| observação | produtos GFD/CEMS/S1F11 e centrais 156/311 | `ds-05-tabela-unica/tabela_unica_v2.parquet` |
| adjudicação/admissão | `admissao_cientifica` por finalidade (não é validação) | `admissao-cientifica/retrato.json` |
| amostras contextualizadas | ds06: papel, relevo, território, evento, grupo | `ds-06-dataset-de-treino/v7/pontos.parquet` |
| treino | logística de terreno / GBM fixado (pesos tabulares); U-Net e MAE (pesos neurais) | `generalizacao-por-reservas/resultado.json` |
| avaliação protegida | **sem reserva intacta**; prospectiva congelada | `reservas-examinadas/retrato.json` |
| candidato | nenhum modelo novo candidato nesta rodada | — |
| promoção/recusa | catraca (neural) e svc02/gates (servido); nada promovido | `neural-catraca/experimento_pote_crescente_redecidido.json` |
| monitoramento | rastreio de impacto (atualização, não validade) | `rastreio-de-impacto/carimbos.jsonl` |
| novo ciclo | política por lacuna: heurística, simulação retrospectiva | `politica-sob-orcamento/resultado.json` |

Três coisas diferentes que se chamam "aprender" e não se misturam: **pesos** (logística/GBM por ajuste; U-Net/MAE por gradiente), **corpus/memória** (RAG e LLM reconstroem índice; nenhum peso muda) e **política** (regra de escolha de dados; hoje heurística escrita à mão, não aprendida). Nenhuma saída de SAR, embedding, memória ou LLM entra no escore físico nem nos gates do svc02.

## 12. Consumidores, fontes públicas e resultados negativos

**Datas lidas da tabela atual:** CEMS 15.566 linhas com a cena do produto e 5 sem data; GFD 521.726 de 521.726 linhas com início de janela.

**Uso temporal depende da pergunta** (data de cena CEMS é observação, não início nem pico):

| pergunta | linhas | exige |
|---|---:|---|
| `ordem_entre_eventos` | 513.525 | reservar eventos posteriores; a data so precisa ordenar eventos |
| `forcamento_no_instante` | 0 | ligar chuva/nivel ao instante da ocorrencia; exige data da propria ocorrencia, e cena, ativacao ou inicio de janela nao sao |
| `previsao` | 51.478 | estimar antes do evento: alem da data, todo preditor tem de existir antes dele -- o DEM nao existia para evento anterior a sua publicacao |

**Modelos servidos: o que cada escore pode afirmar** (campo opcional `afirmacao_autorizada` de toda resposta do `svc02`, fora do `model_card` da v1; não muda quem é recusado):

| modelo | servível | afirmação autorizada | motivo |
|---|---|---|---|
| `areal_brasil_observado` | True | `suscetibilidade_a_inundacao` | — |
| `fluvial_areal_observado` | True | `suscetibilidade_a_inundacao` | — |
| `fluvial_planicie` | True | `suscetibilidade_com_ressalva` | negativo inferido por exclusao no ajuste |
| `fluvial_serra` | True | `suscetibilidade_a_inundacao` | — |
| `pluvial_urbano_nyc` | True | `propensao_a_registrar` | negativo de ausencia de registro no ajuste |
| `pluvial_urbano_observado` | True | `mecanismo_nao_sustentado` | fontes ['ufo'] observam ['MISTO_NAO_SEPARAVEL'], nao PLUVIAL_URBANO |
| `pluvial_urbano_sao_paulo` | False | `propensao_a_registrar` | negativo de ausencia de registro no ajuste |
| `recife_pluvial` | True | `propensao_a_registrar` | negativo de ausencia de registro no ajuste |

O nó `modelos_servidos` fica `sem_carimbo`: os modelos não foram reajustados sobre a tabela atual (reajustar muda o produto e exige decisão de versionamento).

**UFO:** o produtor define não-inundado incluindo nuvem; medido no chip, 10 de 12.900 negativos sem chão visível ({'visivel': 12890, 'nuvem': 7, 'sem_dado': 3}) viraram SEM_DADO no ds06 v7. Driver declarado por evento: um único pluvial (Houston/Harvey, composto).

**Pluvial, busca delimitada sem pedido nem credencial** (C1 observação contínua ou imagem do mesmo evento; C2 funcionamento; C3 mecanismo separável; C4 local e janela; C5 acesso público):

| fonte | estado | por quê |
|---|---|---|
| FloodNet NYC (Open Data aq7i-eu5q, kb2e-tjy3) | adquirida; so positivos (3.422 eventos) | o catalogo usado so tem eventos; a serie de ~1 min 2020-2023 e publica (Zenodo 10211443, profundidade bruta/filtrada/processada), mas acessa-la aceita a licenca NYU/CUNY nao comercial: nao auditada ate decisao de quem responde pelo projeto (corrigido em 04/10/2026; antes dizia 'so mediante pedido', o que era falso) |
| Civil IoT Taiwan / WRA, sensores de alagamento em via (SensorThings) | sondada ao vivo | unica que cumpre C1-C5 no papel |
| WRA IoT (iot.wra.gov.tw) | excluida | exige cadastro e credencial de API |
| Portal Civil IoT (ci.taiwan.gov.tw/dsp), historico | certificado TLS expirado em 03/10/2026 | nao se desliga verificacao TLS: sem ela a procedencia do dado nao e garantida |
| UFO (Zenodo 10.5281/zenodo.19698577) | ja no projeto como MISTO_NAO_SEPARAVEL | um unico evento pluvial, e composto: nao da independencia nem separa mecanismo |
| SuDS / Sunny Day Flooding (Carolina do Norte) | nao adquirida | mecanismo dominante de mare/costeiro |
| Centrais 156/CGE (SP), SEDEC (Recife), 311 (NYC, Chicago) | ja no projeto | registro de ocorrencia: so positivo e propensao a registrar |
| SGB/SACE, CEMADEN (Brasil) | nao serve | medem rio e chuva, nao agua na rua |

Sonda ao vivo do Civil IoT Taiwan em 2026-10-03T15:15:18.173306Z: HTTP 500, operante=False. nenhuma fonte desta busca entrega hoje negativo pluvial observado ja auditado: Taiwan, a unica que cumpre os criterios no papel, esta inoperante; a serie publica do FloodNet 2020-2023 existe, depende de aceite de licenca e ainda nao foi auditada. Uso bloqueado: treino, avaliacao e afirmacao do mecanismo PLUVIAL_URBANO; Brasil nao e validado por NYC/Taiwan.

**Teste confirmatório único** (`gfdq03`, regime `confirmatorio_prospectivo`, plano `sha256:e78a5bbd6b5ed…`, seleção `sha256:ecce4bf486ad8…`): treino 69 eventos GFD conhecidos (354063 linhas); teste 16 eventos GFD novos da América do Sul fora do Brasil, sem evento nem tile em comum (63841 linhas), em quarentena até a leitura.

| comparação | eventos | a favor | mediana dentro | IC95 média dentro | p | veredito |
|---|---:|---:|---:|---|---:|---|
| M1_logistica_terreno × M0_regra_hand | 16 | 13 | 0.0409 | [0.0091, 0.0487] | 0.01099 | **ganho** |
| M3_gbm_terreno × M1_logistica_terreno | 16 | 8 | 0.0063 | [-0.0555, 0.0215] | 0.66855 | **indecidivel** |

Ressalvas gravadas antes da leitura: transferencia territorial: os 16 eventos sao da America do Sul fora do Brasil e o treino e quase todo brasileiro (mais eventos externo:gfd); o teste mede evento novo E territorio novo ao mesmo tempo -- perda da logistica para a regra de HAND ali nao separa as duas causas | minimo de eventos: veredito exige >= 8 eventos pareados com AUC dentro do evento (>= 20 linhas e as duas classes); se a derivacao deixar menos, o veredito e not_testable para aquela comparacao, e isso e desfecho valido, nao motivo para relaxar o criterio | depois da leitura os 16 eventos viram examinados: o resultado grava a lista de eventos de teste para entrar em testes_registrados do reservas_examinadas; conhecidos_2026_10_03.json nao muda

Os 16 eventos passaram a examinados (`reservas_examinadas.registrar_exame`): não confirmam mais nada.

**Réplica do algoritmo do GFD** (`rfd01`, código público MIT; MODIS C6.1, JRC GSW 1.4): portão pré-registrado nos 19 eventos brasileiros do GFD -- concordância mediana 1.0, cobertura mediana 0.9617, 19 eventos acima do mínimo: **passou = True**. Rodada como tarefas em lote (cota do Earth Engine em modo restrito); equivalência lote × interativo conferida em 200/200 pontos.

**Teste confirmatório único no Brasil** (`rfd03`, regime `confirmatorio_prospectivo`): treino 69 eventos GFD; teste 16 cheias brasileiras 2019-2025 do GDACS rotuladas pela réplica (60283 linhas; 100% em tile do treino: evento novo, lugar conhecido).

| comparação | eventos | a favor | mediana dentro | IC95 média dentro | p | veredito |
|---|---:|---:|---:|---|---:|---|
| M1_logistica_terreno × M0_regra_hand | 16 | 11 | 0.0393 | [-0.0051, 0.0434] | 0.14386 | **indecidivel** |
| M3_gbm_terreno × M1_logistica_terreno | 16 | 15 | 0.0399 | [0.0266, 0.0551] | 9e-05 | **ganho** |

Leitura conjunta dos dois confirmatórios: a logística de terreno supera a regra de HAND em território novo (América do Sul) e fica indecidível no Brasil conhecido, com a mesma direção; o GBM acrescenta à logística no Brasil conhecido e não em território novo -- compatível com ele aprender estrutura do lugar visto no treino. Ressalvas gravadas antes: rotulo da replica do GFD (rfd01, MODIS C6.1, JRC GSW 1.4), validada nos 19 eventos brasileiros do GFD; nao e o produto GFD | lugar conhecido permitido: o teste mede evento novo (2019-2025) em territorio ja visto no treino; junto com o sul-americano (evento e territorio novos) separa, em parte, evento de territorio | eventos do GDACS (GLOFAS/JRC) e nao do DFO; janela fromdate-todate do GDACS | Terra em deriva de orbita desde 2022: menos imagens por dia nos eventos recentes | minimo de 8 eventos pareados; abaixo disso o veredito e not_testable, desfecho valido | depois da leitura os eventos viram examinados (registrar_exame)

**Pluvial: série FloodNet 2020-2023 auditada** (exploratório: primeira análise sobre dado recém-auditado, sensores instalados onde já alagou; licença aceita pela pesquisadora em 04/10/2026; `fn01`/`fn02`/F1). Janela sensor×dia, só sensor pluvial funcionando (cobertura, lacuna, perda de quadros e bateria, critérios fixados antes):

- janelas: {'nao_observado': 13127, 'sem_lamina': 3955, 'desconhecido': 1533, 'lamina': 190}; sensores com `sem_lamina`: 45. `sem_lamina` = nada acima de 10 mm no ponto do sensor naquele dia, não seco absoluto.
- validação com o catálogo: {'lamina': 27, 'nao_observado': 19, 'sem_lamina': 0, 'desconhecido': 0} -- nenhum evento do catálogo cai em janela `sem_lamina`.
- forçamento dentro do sensor: AUC da chuva MRMS 0.6942 (IC [0.6085, 0.7919], 12 sensores); o F1 usa 186 lâmina / 3935 sem lâmina porque exclui os dias sem grade MRMS.
- terreno entre sensores no mesmo dia com ≥ 20.0 mm: HAND {'dias': 6, 'auc_media': 0.6389, 'ic95_media_por_dia': [0.4, 0.8501]}, TWI {'dias': 6, 'auc_media': 0.4736, 'ic95_media_por_dia': [0.2556, 0.6514]} -- poucos dias com as duas classes: não sustenta conclusão sobre terreno.
- limites: o FloodNet instala onde já alagou (contraste entre lugares propensos, não a cidade); o critério de perda de quadros descartou 10 dos 19 eventos pluviais não observados -- mais estrito que o necessário, não alterado depois de visto; Nova York não valida o Brasil.

**Serra: delineações públicas auditadas e recusadas como referência**

- `mapbiomas_rs2024_area_atingida`: nota tecnica MapBiomas (jun/2024): deteccao de mudanca S1/S2/Planet, depois exclui pixel em encosta por limiar de HAND e separa deslizamento por declive de 7% -- circular para testar HAND e declive; inclui solo saturado e mancha modelada de POA (Possanti 2024); sem metrica de acuracia. So exploracao
- `zenodo_17177153`: titulo promete mapeamento de areas inundadas RS 2024, mas o pacote publicado contem ganho/perda de vegetacao por hexagono de 2 km: nao ha delineacao de cheia
- `zenodo_11151528`: limiar de NDWI de um sensor (S2) que inclui area umida e agua permanente, publicado so como imagem (PNG/PDF), sem vetor: nao e independente do que o projeto computa e nao e utilizavel como camada

**Serra: rótulo por dois sensores** (S2 óptico + S1 radar, controle seco, regra v1, base S1 de 60 dias): rótulos admitidos 0, grupos por classe {'agua_de_cheia': 0, 'seco_observado': 0}, testável False, para revisão humana 97.

- rs_taquari_mai2024: cena 4 dias após o evento; classes {'desconhecido': 200}; desconhecido por {'sem_s1_na_janela': 195, 'nuvem_ou_sombra_no_s2': 5}
- rs_cai_mai2024: cena 4 dias após o evento; classes {'desconhecido': 200}; desconhecido por {'sem_s1_na_janela': 192, 'nuvem_ou_sombra_no_s2': 8}
- rs_taquari_jun2025: cena 6 dias após o evento; classes {'seco_na_data_da_cena': 139, 'desconhecido': 61}; desconhecido por {'sensores_discordam': 45, 'nuvem_ou_sombra_no_s2': 15, 'ambiguo_entre_limiares': 1}
- sp_litoral_norte_fev2023: cena 6 dias após o evento; classes {'seco_na_data_da_cena': 138, 'desconhecido': 62}; desconhecido por {'sensores_discordam': 51, 'nuvem_ou_sombra_no_s2': 11}
- rj_petropolis_fev2022: cena 2 dias após o evento; classes {'desconhecido': 200}; desconhecido por {'sem_s1_na_janela': 144, 'nuvem_ou_sombra_no_s2': 56}

**Serra: rótulo por dois sensores** (S2 óptico + S1 radar, controle seco, regra v2, base S1 de 365 dias): rótulos admitidos 0, grupos por classe {'agua_de_cheia': 0, 'seco_observado': 0}, testável False, para revisão humana 241.

- rs_taquari_mai2024: cena 4 dias após o evento; classes {'seco_na_data_da_cena': 112, 'desconhecido': 88}; desconhecido por {'sensores_discordam': 83, 'nuvem_ou_sombra_no_s2': 5}
- rs_cai_mai2024: cena 4 dias após o evento; classes {'seco_na_data_da_cena': 122, 'desconhecido': 78}; desconhecido por {'sensores_discordam': 70, 'nuvem_ou_sombra_no_s2': 8}
- rs_taquari_jun2025: cena 6 dias após o evento; classes {'seco_na_data_da_cena': 139, 'desconhecido': 61}; desconhecido por {'sensores_discordam': 45, 'nuvem_ou_sombra_no_s2': 15, 'ambiguo_entre_limiares': 1}
- sp_litoral_norte_fev2023: cena 6 dias após o evento; classes {'seco_na_data_da_cena': 147, 'desconhecido': 53}; desconhecido por {'sensores_discordam': 42, 'nuvem_ou_sombra_no_s2': 11}
- rj_petropolis_fev2022: cena 2 dias após o evento; classes {'desconhecido': 200}; desconhecido por {'sem_s1_na_janela': 144, 'nuvem_ou_sombra_no_s2': 56}

Leitura: seco numa cena de 4 a 6 dias depois do pico não é ausência de cheia em vale de serra; nenhuma cena limpa viu água de cheia nos pontos sorteados; o S1 não tem passagem regular no sul do Brasil. A revisão humana com as mesmas imagens herda o mesmo limite temporal: a referência de serra segue bloqueada por falta de observação pública no instante do pico, e não por falta de revisão.

**Fluvial: auditoria do `not_supported` do ctx04** (protocolo original preservado):

- operacional: 19 de 40 pares usados; perdidos {'bacia_divergente_ou_falha': 7, 'pico_incompleto_ou_ausente': 13, 'sem_percentil_ou_chuva': 1}; 11 dos usados têm usina ou barragem dentro da bacia; 9 pares de bacias aninhadas no mesmo evento;
- resolução: com 11 eventos, a meia-largura do IC95 da diferença é {'auc_agrupada': 0.1239, 'auc_media_dentro_do_par': 0.0062, 'brier': 0.0531, 'log_loss': 0.1279}: ganho menor que isso é indistinguível de zero;
- como a perda acontece: erro da fração positiva por par {'terreno': 0.1617, 'terreno_mais_contexto': 0.2154} (o contexto desloca interceptos entre eventos); o sinal do percentil troca entre dobras: True;
- saturação, como hipótese: percentil entre 0.9137 e 0.9999; correlação chuva × fração positiva -0.414, negativa, o que aponta para a faixa de amostragem do GFD e não para a física.

Decisão: nenhum experimento novo de variáveis. O limite dominante é a resolução em eventos independentes e o desenho de amostragem; acrescentar variáveis sobre 11 eventos seria busca até ganho. O que mudaria a conclusão: mais eventos independentes com posto não regularizado. Quantos depende do efeito que importa: com a variância medida (meia-largura 0,124 em 11 eventos), resolver 0,05 de AUC pediria da ordem de 70 eventos, se a variância por evento se mantiver -- estimativa, não garantia.

### Restrições desta seção

| lacuna | por quê | uso |
|---|---|---|
| pluvial urbano | negativo observado existe só em Nova York (série FloodNet auditada: sensor×dia sem lâmina > 10 mm), em sensores instalados onde já alagou; terreno entre sensores sem dias suficientes | forçamento dentro do sensor medido em NYC; treino e afirmação para o Brasil seguem bloqueados |
| serra brasileira | o CEMS observa relevo íngreme (14 eventos fora do Brasil, 2 grupos brasileiros, 320 positivos), mas nenhuma observação pública no pico dos 5 eventos do pacote; o pacote tem 1,5-4% de pontos em fundo de vale (E3); delineações públicas recusadas | afirmação sobre serra bloqueada; casos ambíguos listados em `serra-03-rotulo-sensor-duplo/*/revisao_humana_reduzida.csv` |
| fluvial com contexto | 11 eventos não resolvem ganho menor que 0,12 de AUC agrupada | inconclusivo por resolução |
| modelos servidos | ajustados antes da admissão central; Recife e NYC medem propensão a registrar | afirmação declarada no cartão; reajuste depende de decisão de versionamento |
| testes de Curitiba (5) | expectativas atualizadas ao contrato vigente, garantias preservadas | resolvido |

## 12b. Camada de resposta: memória, verificador, recusa e lugar novo

- **Memória:** corpus com 3813 trechos; precedência com {'historico': 7, 'vigente': 42, 'superado': 4}. Trechos de documento superado ou de commit revisado chegam à resposta etiquetados (`consulta_cognitiva` chama `precedencia_documental.anotar`), e o vigente sai ao lado.
- **Verificador:** `claim_generalizacao` barra generalização afirmada sem teste confirmatório, no Brasil, do GBM, ou sem território delimitado; certeza sobre efeito de variável ("prova", "garante") é sempre recusada.
- **REV-P-EVAL:** `APROVADO`; 24 ataques, 24 bloqueados, 0 vazados, 0 sem payload (14 com alvo substituto declarado).
- **Recusa por domínio (exploratório, pré-registrado):** dentro do GFD (primário) 0.209976 vs acaso p05 0.18945 → ordena `False`; CEMS → GFD 0.195779 vs acaso p05 0.19856 → ordena `True`. Dentro da mesma fonte a regra recusa terreno extremo e fácil (Brier dos recusados 0,02–0,09 contra ~0,17 dos aceitos); só entre instrumentos ela ordena o erro. A regra não foi alterada.
- **Lugar novo:** {'explicacao_do_modelo_disponivel': 16, 'recusado': 7}; incerteza por bootstrap de eventos `executado` (200 réplicas). Artefato histórico: não autoriza consulta anterior ao evento. A consulta nova exige população compatível e preditores calculáveis sem a amostra rotulada.

## 13. Contribuição real de cada frente para a IA final

| frente | contribuição medida | estado |
|---|---|---|
| terreno (logística) | supera a regra de HAND no GFD: exploratório `ganho`, **confirmatório em território novo `ganho`** (seção 8b, 16 eventos fora do Brasil); confirmatório em território conhecido (Brasil, seção 12) `indecidivel`; no CEMS: `indecidivel` | evidência na população histórica; não autoriza consulta territorial |
| não linearidade (GBM) | exploratório GFD `ganho`, CEMS `ganho`; confirmatório em território novo `indecidivel`, em território conhecido (Brasil) `ganho`. Esses contrastes delimitam evidência, não demonstram impossibilidade universal de transferência | hipótese de desenvolvimento; os eventos dos dois testes não reconfirmam |
| cobertura/solo | GFD `perda`, CEMS `indecidivel`: a associação não transfere entre eventos, e o dado é do ano do evento com preenchimento não aleatório | fora do vetor |
| política por lacuna | 4.0: `indecidivel`, 8.0: `indecidivel`, 16.0: `not_testable` | heurística; simulação retrospectiva |
| pré-treino SAR (A8) | H1 `INDECIDIVEL` | não promovido; congelado sem decisão prática |
| rede SAR + recusa (A0/A7) | recusa ordena no teste: `True` | observação, nunca preditor do escore |
| encoder de terreno (B) | sobre HAND+TWI (2 variáveis; nível de AUC não comparável à seção 8) acrescenta [0.0139, 0.0359] (N1, `acrescimo`), mas estatística simples da mesma janela de 3,84 km o supera: embedding − vizinhança [-0.0216, -0.0038] (N1b, post hoc, `vizinhanca_supera_embedding`) | N1b compara substituição; não exclui ganho incremental ao acrescentar o encoder (MS4 K5) |
| vizinhança da janela | ganho DENTRO do evento: GFD 0.0832 [0.0721, 0.0947] (68/69); CEMS 0.0053 [-0.0333, 0.041] (14/27) (V2b, post hoc). No CEMS o ganho do V2 é só entre eventos | ganho dentro do evento so no GFD: compativel com o desenho/escala do GFD, nao demonstra relevo; não é afirmado como relevo |
| RAG / juiz / LLM | redigem a partir de objetos medidos; não atualizam pesos | camada de resposta, fora do escore |

## 13b. Auditoria pré-treino: o que limita as seções acima

Síntese única em `docs/AUDITORIA_PRE_TREINO.md` (gerada em 2026-10-07T05:36:56.802183Z, `sha256:6b6227a70ab38810…`). Nenhum teste confirmatório foi rerodado; os vereditos das seções 8b e 12 ficam como medidos, com estes limites:

| achado | classe | impacto |
|---|---|---|
| A1 Filtro de declive do GFD/rfd01: dominio, nao negativo | `limitacao_de_desenho` | o corte virou NoData nas duas classes; nao ha negativo 'por construcao' no declive. Restam vieses: o negativo de vale encaixado so vem da parte plana (atenua o contraste de declive), positivo encostado em parede ingreme conta como interior, e o corte e no GMTED, nao no slope_deg do modelo. O efeito de declive/HAND vale dentro do dominio |
| A2 Alvo e estratos: agua nao permanente, episodica e recorrente | `limitacao_de_desenho` | o alvo aprendido e agua nao permanente. Recorrencia e estrato de leitura, nunca exclusao nem prova de memorizacao. Desenvolvimento (S2): logistica x HAND `ganho` nos positivos nunca/raros e `ganho` nos recorrentes; GBM x logistica `ganho` / `ganho` |
| A3 Populacao definida pelas variaveis da tarefa | `defeito_corrigivel` | GFD: logistica x HAND `ganho` -> `ganho`, GBM x logistica `ganho` -> `ganho`. CEMS (eventos 13 -> 33): GBM x logistica `ganho` -> `indecidivel`: o ganho exploratorio do GBM no CEMS dependia do corte. Transferencia CEMS -> GFD na v2: logistica 0,573 de AUC agrupada contra 0,738 da regra de HAND (v1: 0,786); entram 14.089 negativos do CEMS planos e perto da agua (HAND mediano 1,46 m) e o sinal do TWI na logistica do CEMS troca (+0,24 -> -0,22) enquanto o peso da elevacao local triplica; a causa dessa queda e o quadro do negativo (A17) |
| A4 Identidade canonica com evidencia; independencia espacial separada | `defeito_corrigivel` | os pares GFD x Sen1Floods11 sao provavel mesmo evento com amostras a ~220-235 km: dependencia de evento, nao pixel duplicado. Correcao da sintese anterior: sinais_marginais NAO era afetado (contexto x fonte). Periodos distintos (GFD 2002-2018, CEMS 2023-2026) nao tornam as fontes independentes no espaco |
| A5 Teste Brasil: janela x bacia nivel 4 | `limitacao_de_desenho` | vale como 'janelas novas de agua nao permanente em bacias e tiles ja vistos', com 75% de positivos recorrentes. O ganho do GBM e compativel com estrutura local -- inclusive onde a agua recorre --, sem demonstrar memorizacao |
| A6 Teste sul-americano: leitura parcial anterior | `limitacao_de_desenho` | nenhuma escolha de modelo usou esses pontos; veredito mantido com ressalva |
| A7 SAR pre/pos: o ganho intercalado nao sobrevive a particao por evento | `defeito_corrigivel` | o +0,16 do S1GFloods e interpolacao dentro da cena; com evento reservado o pre/pos nao ajudou (estimativa negativa, indecidivel) |
| A8 FloodNet: regras corretas, contraste curto | `falta_observacional` | forcamento dentro do sensor medido em NYC; terreno entre sensores sem dias; Brasil sem negativo pluvial |
| A9 Serra: cena depois do pico | `falta_observacional` | nenhum aprendizado de serra brasileira |
| A10 Contexto fluvial: resolucao em eventos | `falta_observacional` | nao resolve ganho < ~0,12 de AUC agrupada |
| A11 Vizinhanca e encoder | `hipotese_nao_testada` | sem vantagem da representacao aprendida sobre estatistica simples |
| A12 Consumidores e copias desatualizados | `defeito_corrigivel` | documentos apontavam para dados que ninguem le |
| A13 Negativo do GFD visto em qualquer dia | `hipotese_nao_testada` | efeito no ranking entre modelos desconhecido |
| A14 Trava pesada: exclusao mutua e retomada | `defeito_corrigivel` | dois trabalhos pesados juntos em maquina de 15,5 GB |
| A15 Rastreio de impacto ve regra, nao so arquivo | `defeito_corrigivel` | derivacoes novas (v2, E2c, posicao topografica, MS1, politica e recusa v2) passam a ter carimbo e regra |
| A16 elevation_local_m depende do rotulo amostrado | `defeito_corrigivel` | C1 piso amostral: GFD `sem_diferenca_relevante` 0.0025 (50/69), CEMS `ganho` 0.0273 (25/28), CEMS->GFD `ganho` 0.0256 (62/69). C2 posicao sem rotulo (primario): GFD `ganho` 0.0303 (59/69), CEMS->GFD `sem_diferenca_relevante` 0.0069 (68/69), GFD->CEMS `ganho` 0.0395 (22/28). C3 multiescala: GFD `sem_diferenca_relevante` 0.0038 (42/69), CEMS->GFD `ganho` 0.1122 (59/69). Leitura: no GFD o piso nao acrescenta e a posicao sem rotulo sim; no CEMS e entre fontes o piso acrescenta, o que nao separa sinal fisico de vazamento; o primario CEMS->GFD fica dominado pelo quadro do negativo (A17) e nao informa sobre a posicao |
| A17 Queda CEMS -> GFD: quadro do negativo, nao fisica invertida | `limitacao_de_desenho` | nao e saturacao numerica (expit = preditor linear); nenhuma variavel inverte de direcao entre fontes (TWI alto indica cheia nas duas); o sinal negativo do TWI no CEMS e condicional ao HAND (colinear dentro do evento); o contraste de HAND do GFD vive abaixo de 1 m, ~0,03 desvio do HAND do CEMS, que tira o negativo da AOI inteira. Harmonizar o CEMS ao quadro de 2 km do GFD leva a mesma logistica a 0,770 no GFD (T1 pre-registrado, 69/69 eventos). VanDerWal et al. (2009): a extensao do fundo muda pesos e desempenho |

Prontidão por tarefa: ordenar suscetibilidade fluvial de planicie (agua nao permanente) `pronto_com_ressalva`; transferencia para territorio novo `pronto_com_ressalva`; transferencia dentro do Brasil para lugar nao visto `bloqueado_por_observacao`; nao linearidade (GBM) `pronto_com_ressalva`; probabilidade territorial `bloqueado_por_observacao`; serra brasileira `bloqueado_por_observacao`; pluvial urbano no Brasil `bloqueado_por_observacao`; forcamento pluvial dentro do sensor (NYC) `pronto_com_ressalva`; contexto fluvial `bloqueado_por_observacao`; segmentacao de agua SAR (Sen1Floods11) `pronto_com_ressalva`; mudanca pre/pos SAR `parcialmente_resolvido`; representacao aprendida de terreno `pronto_com_ressalva`; politica de aquisicao `medido_sem_vantagem`; recusa por dominio `nao_sustentado`; posicao topografica sem rotulo no vetor `pronto_com_ressalva`; transferencia entre instrumentos (CEMS <-> GFD) `decisao_de_alvo`.

## 14. Próxima decisão experimental

Os dois testes confirmatórios disponíveis **foram executados**: território novo (seção 8b) e território conhecido no Brasil (seção 12). Seus resultados permanecem restritos à população e ao vetor avaliados. A auditoria A16 mostrou dependência amostral de `elevation_local_m`; TC01 e MS6 mudam o quadro de consulta. Portanto, antes de nova confirmação, é preciso estabilizar alvo, população e variante calculável na inferência. A prioridade brasileira é **evento em território não visto no treino**, com observabilidade elegível e ausência nos registros conhecido e examinado. A comparação antiga não deve ser reutilizada silenciosamente se o vetor ou a pergunta mudar: a variante corrigida exige desenho próprio, congelado antes de abrir observações independentes. Reexaminar os testes consumidos não cria nova confirmação.

## 15. Framework atual e fronteira de consulta

A síntese reproduzível das frentes e dos critérios de conclusão está em `docs/FECHAMENTO_DO_FRAMEWORK.md`, gerada por `scripts/metodo/fechamento_do_framework.py`. O manifesto registra hashes, integridade e ausências; isso não substitui validade observacional.

- **MS4 K5:** `ganho`, delta mediano dentro do evento 0.0152 de AUC, IC da média [0.0072, 0.0179]. Compara acrescentar embedding à vizinhança, não substituí-la. Desenvolvimento em eventos conhecidos, sem promoção ou confirmação territorial.
- **MS5:** novidade_mahalanobis `nao_ordena`; novidade_knn `nao_ordena`; incerteza_margem `ordena`; incerteza_bootstrap `ordena`. Ordenação de erro no desenvolvimento não é probabilidade calibrada nem regra servida.
- **MS6:** código disponível, resultado ainda ausente nesta síntese.
- **Consulta:** `resposta_lugar_novo.py` recusa por padrão a linhagem de piso amostral; `--demonstracao-historica` explicita a reprodução antiga. Isso não modifica o escore dos modelos servidos.

## Lacunas restantes e o que as resolveria sem pedido, contato ou credencial

| lacuna | estado | o que resolveria |
|---|---|---|
| avaliação confirmatória | nenhuma reserva intacta no ds06 (seção 0) | eventos GFD/CEMS públicos ausentes de `conhecidos_2026_10_03.json`, avaliados uma vez com os critérios já congelados |
| pluvial urbano | negativo observado só em Nova York, em sensores instalados onde já alagou (série FloodNet auditada, seção 12): chuva ordena dentro do sensor; terreno entre sensores indecidível (6 dias) | para o Brasil, nada: NYC não valida o Brasil. O que resolveria aqui: observação contínua brasileira com funcionamento do sensor, ou imagem do mesmo evento com área seca visível e mecanismo pluvial separável |
| serra brasileira | sem observação pública no pico (seção 12) | cena SAR/óptica no dia do pico; até lá, afirmação sobre serra bloqueada |
| CEMS pré-2023 (Petrópolis 2022) | fora desta rodada: exige credencial nova | sem alternativa pública verificada; o uso fica restrito ao que existe |
