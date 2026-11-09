# Evolução abrangente pós-v8 (v9): resultados executados

23/10/2026. Execução de `docs/PROMPT_EVOLUCAO_ABRANGENTE_POS_V8_2026_10_22.md`.

- **Linhagem:** `local_runs/transferencia-vulnerabilizados/v9_evolucao_abrangente/` (658 arquivos, 2,72 GB, manifesto fechado depois do registro da etapa).
- **Código:** `scripts/vulnerabilizados/v9_*.py`. **Testes:** `tests/test_v9_evolucao_abrangente.py`.
- **Figura:** `docs/figuras/v9_evolucao_abrangente_2026_10_22.png`.

**Natureza:** tudo é **desenvolvimento ou reanálise** de coortes já examinadas (RMPA/2024, Manaus/2021, Curitiba/2023, Recife e RMR/2022, sendo Recife e RMR o mesmo episódio). Nada aqui é confirmação.

**Preservação:**
- v1–v8 intactos; as 15 fontes protegidas foram conferidas por hash.
- Bolívia (Sen1Floods11) trancada; nenhuma reserva aberta; nenhum peso promovido; LaTeX intocado; nenhum commit.

## 1. Execução real

| Etapa | Duração | Dados que ensinaram |
|---|---:|---|
| P0 regularização | 1,6 min | 36.331 linhas fora do Brasil (59 eventos, 43 unidades); grade λ 4 × 5 em 3 dobras de unidades estrangeiras |
| P0 continuidade | 1,2 min | Orçamentos v4 (3 políticas × B 25/100/400 × 6 sementes), 1.296 ajustes |
| B tabular | 3,3 h | 96 configurações em 6 famílias (halving 1 → 3 dobras → 2 finalistas × 3 sementes); ajuste final em todo o fora-do-Brasil |
| F aquisição | 9 min | Pools com todos os estados: RMPA 209.474, Manaus 48.913, Curitiba 149.027 pontos |
| G contínua | 1 min | 3 ordens × 6 sementes × 6 estratégias |
| H incerteza | 1 min | Deep ensemble de 5 MLPs nas unidades de ajuste físico; estimador nas 8 unidades de competência |
| E generalização | 20 min | 6 objetivos × 3 sementes, 5 regiões deixadas de fora, 10 ablações de região |
| C patches | 31 min de extração + 2 h de treino | 36.331 patches externos e 110.545 de coortes (32×32×5); 5 famílias × 4 configurações + 3 sementes |
| C multimodal / D conjunta | 20 min + 71 min | 431 chips Sen1Floods11 a 30 m; 8 braços × 3 sementes |
| Contrastes, síntese, reavaliação | 3,5 min | Bootstrap de 300 reamostragens por comunidade/setor |

**Operação:**
- Duas paradas por bateria: a fila parou e retomou sem perder etapas.
- Uma suspensão do PC durante a Frente B, que retomou sozinha.

**Falhas próprias, corrigidas e registradas:**
- `etapa(nome=...)` colidia com o parâmetro `nome` (pego pelo Pyright antes da execução).
- Sorteio do curriculum sem limite de tamanho (pego por teste antes da execução).
- Avaliação do braço só-radar com o terreno real: AUC 0,30 era artefato. Reavaliado a partir dos pesos salvos: 0,93–0,94.
- Unpickle das classes da Frente B gravadas como `__main__`.

## 2. P0: controles que precisavam ser corrigidos

**Regularização só no exterior** (`p0_regularizacao/`):
- A seleção agrupada só em unidades estrangeiras escolheu λ_p = 0,01 e λ_i = 0,1; a herdada com Brasil era 0,001 e 100.
- Nas coortes, a diferença para o histórico é ≤ 0,002 de AUC. Só RMPA-todas tem intervalo que exclui zero (−0,0017 [−0,0029; −0,0006]); nas FCU, todos os contrastes são inconclusivos e centrados em zero.
- **O ganho estrangeiro persiste sem nenhuma escolha feita com rótulos brasileiros.**
- O Q_sep (para competência e retenção) também foi re-selecionado só nas unidades físicas.

**Continuidade do Q** (`p0_continuidade/`):
- A adaptação parte da função completa (efeitos principais, interações e imputação) com uma correção offset que começa em zero.
- **Paridade exata:** nas 162 recusas, a função final é idêntica ao Q (diferença máxima 0,0).
- **Simplificação para 7 efeitos principais (usada na v8):** Spearman ≥ 0,9994 com o Q completo e |ΔAUC| ≤ 0,002. A descontinuidade era desprezível, agora medida.
- **Filtro de suporte** (≥ 5 positivos, ≥ 5 negativos, ≥ 3 agrupamentos com positivo): sem suporte, conserva o Q.

| B = 400, uniforme | Correção sem filtro | Com filtro | Q (B = 0) |
|---|---:|---:|---:|
| Curitiba FCU | 0,819 | **0,893** | 0,893 |
| Manaus FCU | 0,898 | **0,925** | 0,920 |
| RMPA todas | 0,926 | **0,926** | 0,906 |

**Manifesto:** a nova rotina registra a etapa antes de calcular os hashes; o histórico v8 foi preservado.

## 3. Frente A: qualificação de dados

**JRC** (`a_jrc/`):
- **Bandas e estados:** consultadas `occurrence`, `recurrence`, `seasonality` e `max_extent` com máscara explícita. Ficam separados falha de consulta (0), água nunca detectada (54.935 pontos) e histórico.
- **Correção da v8:** 26 pontos que a v8 tratou como nulos tinham histórico.
- **Água preexistente:** pela **recurrence**, 37% dos positivos UFO (4.783/12.900) caem em pixels com água em ≥ 75% dos anos. Nos negativos UFO são 3,9%; nos positivos de inundação do Sen1Floods11, 17,6%.
- **Leitura:** a evidência de água preexistente no UFO é mais forte que os 12,4% medidos pela `occurrence` na v8.
- **Limites:** 1984–2021 inclui anos posteriores a muitos eventos, e nenhum rótulo foi reclassificado.

**Qualidade como hipótese** (Frente E): a ponderação dos positivos UFO recorrentes e o curriculum (CEMS + Sen1Floods11 primeiro) **empataram com o ERM** em todas as regiões e coortes.

## 4. Frente B: modelos tabulares e correção residual

Zero-shot, AUC nas FCU; contraste contra o Q com IC 95% por comunidade/setor (`contrastes_bootstrap.csv`).

| FCU | Q | GB monotônico | Boosting residual | MLP residual | FT-Transformer |
|---|---:|---:|---:|---:|---:|
| RMPA | 0,880 | 0,921 (+0,041 inc.) | 0,907 (+0,027 inc.) | 0,842 (**−0,039**) | 0,877 (inc.) |
| Manaus | 0,920 | 0,911 (inc.) | 0,915 (inc.) | 0,903 (inc.) | 0,892 (**−0,028**) |
| Curitiba (11 pos.) | 0,893 | 0,861 | 0,889 | 0,879 | 0,868 |
| Recife | 0,866 | 0,798 (**−0,069**) | 0,859 (inc.) | 0,854 (inc.) | 0,848 (**−0,019**) |
| RMR | 0,809 | 0,859 (+0,050 inc.) | 0,881 (+0,072 inc.) | 0,863 (inc.) | 0,839 (inc.) |

- **Nenhuma família supera o Q com intervalo** nas FCU. Árvores sobre os atributos físicos ganham em pontos estimados na RMPA e na RMR, mas perdem em Recife. Redes diretas perdem.
- **Contra o −declive:** GB monotônico e boosting residual vencem em Recife (+0,08 / +0,14) e na RMR (+0,14 / +0,17), com intervalo; na RMPA, inconclusivo.
- **Monotonia contra livre:** diferença ≤ 0,03 sem padrão.
- **TabPFN:** não usado; a licença dos pesos exige aceite e a base excede o regime publicado.

## 5. Frente C: representação espacial e multimodal

**Patches de terreno** (encoder `encoder_v2_autosup`, pré-treinado sem rótulos em chips Sen1Floods11, que incluem o terreno, não o rótulo, de uma unidade de retenção):
- **Como correção residual sobre o Q, piora a RMPA com intervalo** (−0,015 a −0,038 em todas as famílias). Em Manaus, +0,010 a +0,015 inconclusivo; nas demais, inconclusivo ou pior.
- **A representação pré-treinada tem conteúdo:** sem o Q, o embedding pré-treinado congelado chega a 0,904 na RMPA-todas e 0,836 em Manaus. O embedding aleatório dá 0,835 e 0,723.
- **Mas não é universal:** em Curitiba e Recife o aleatório chega a superar o pré-treinado sem o Q.
- **Ajuste fino contra rede do zero:** diferenças pequenas e de sinal variável.
- **Leitura:** o pré-treino acrescenta informação de forma do terreno, mas ela não se soma ao Q nas coortes brasileiras.

**Observação multimodal** (água visível, chips Sen1Floods11 a 30 m, AUC por pixel no teste, 3 sementes):

| Braço | Radar presente | Radar ausente |
|---|---:|---:|
| Fusão com máscara de modalidade | 0,927–0,952 (IoU 0,50–0,55) | 0,81–0,87 |
| Só radar (reavaliado) | 0,930–0,942 (IoU 0,57–0,59) | — |
| Só terreno | — | 0,85–0,87 |

- **Com radar presente**, o terreno não acrescenta à observação.
- **Sem radar**, a fusão com máscara mantém a utilidade do terreno.
- **O S1 é co-evento:** serve à observação, nunca à suscetibilidade antecedente.

## 6. Frente D: aprendizagem conjunta

Encoder compartilhado entre a suscetibilidade (patches externos, sem radar) e a observação (chips), mesmo número de passos, média de 3 sementes:

| Braço | Suscetibilidade FCU: RMPA / Manaus / Curitiba / Recife / RMR | Observação (radar presente / ausente) |
|---|---|---|
| Separada | 0,844 / 0,944 / 0,847 / 0,857 / 0,778 | 0,942 / 0,833 |
| Gradiente bloqueado | 0,861 / 0,935 / 0,875 / 0,861 / 0,796 | 0,933 / 0,798 |
| Conjunta | 0,866 / 0,931 / 0,868 / 0,859 / 0,796 | 0,953 / 0,842 |
| **Pareamento rompido** | **0,868 / 0,933 / 0,868 / 0,866 / 0,800** | 0,355 / 0,525 |

- **Gradientes:** cosseno médio de −0,037, com 57% dos passos em conflito leve.
- **O controle de pareamento rompido iguala a conjunta na suscetibilidade.** O ganho da conjunta sobre a separada vem de regularização ou de mais passos no encoder, não do conhecimento da tarefa observacional.
- **Benefício do compartilhamento: não identificado.** A ponderação de perdas não foi testada, porque o conflito diagnosticado é leve.

## 7. Frente E: generalização entre domínios

- **Deixando cada região de fora:**
  - ERM ≈ balanceado ≈ CORAL, com o GroupDRO pior (−0,01 a −0,05);
  - ponderação por qualidade e curriculum ≈ ERM.
- **Zero-shot nas coortes:** todas as variantes de MLP residual ficam **abaixo do Q com intervalo** na RMPA (−0,06 a −0,11), em Manaus e em Recife; na RMR, inconclusivo.
- **Ablação de região (1 semente, descritivo):** sem as 12 unidades da África/Oriente Médio, as FCU melhoram contra o controle de mesmo tamanho. É candidato a transferência negativa por região, a testar com mais sementes e no Q.

| FCU | Sem África/OM | Controle de mesmo tamanho |
|---|---:|---:|
| RMPA | 0,850 | 0,815 |
| RMR | 0,875 | 0,827 |
| Manaus | 0,920 | 0,886 |
| Recife | 0,859 | 0,819 |

## 8. Frente F: aquisição ativa

Pools com todos os estados, de modo que consultar um desconhecido gasta orçamento; 6 sementes, rótulos escondidos até a consulta.

| 400 consultas | Uniforme | Margem | BADGE | Diversidade | Mistura | Alto escore | Oráculo (teto) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Positivos, FCU RMPA | 32 | 55 | 65 | 52 | 49 | 197 | 400 |
| AUC teste FCU RMPA (Q 0,880) | 0,884 | **0,925** | 0,906 | 0,902 | 0,905 | 0,852 | 0,880 |
| Positivos, FCU Manaus | 0,3 | 0 | 1,5 | 4,3 | 3,0 | 9 | 14 |
| Consultas sem rótulo, FCU Manaus | 23 | 18 | 30 | 42 | 30 | 124 | 26 |
| Positivos, FCU Curitiba | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- **Margem dá mais ganho por consulta:** +0,040 nas FCU da RMPA sobre a uniforme.
- **Achar positivos não basta:** "alto escore" acha 6 vezes mais e piora o teste, porque a amostra fica enviesada; também gasta 5 vezes mais consultas sem rótulo em Manaus.
- **O oráculo só de positivos não ajuda:** sem negativos, o filtro recusa a atualização.
- **Curitiba FCU:** nenhuma estratégia cria positivo onde a referência não tem.

## 9. Frente G: crescimento contínuo

Média de 3 ordens × 6 sementes; retenção em 6 unidades externas que não ajustaram nada.

| Estratégia | Ganho na RMPA | Retenção das coortes anteriores | Retenção externa (final − início) |
|---|---:|---:|---:|
| Regularizada (v8) | +0,017 | −0,031 (RMPA) | −0,042 |
| Com filtro de suporte | +0,021 | −0,004 | −0,026 |
| Replay (memória 200) | +0,027 | −0,009 | −0,029 |
| **DER++ (replay + destilação de logits)** | +0,019 | **≈ 0** | **−0,020** |
| Conjunta (não sequencial) | +0,031 | — | −0,036 |

**Conservar a função anterior corta o esquecimento pela metade, e a destilação é a mais conservadora.** Nenhuma estratégia zera o esquecimento externo.

## 10. Frente H: incerteza e encaminhamento

Respostas "menor suscetibilidade" erradas com 70% de cobertura; alertas nunca recusados.

| Coorte | Margem | Aprendido | Combinação | DI | Discordância | Ensemble 5 redes | Aleatório |
|---|---:|---:|---:|---:|---:|---:|---:|
| RMPA | **33** | 34 | 192 | 621 | 668 | 820 | 639 |
| RMR | 294 | **293** | 301 | 1.004 | 868 | 1.071 | 797 |
| Recife | 77 | **76** | 90 | 107 | 118 | 152 | 121 |
| Manaus | **34** | **34** | 36 | 46 | 47 | 57 | 51 |
| Curitiba | 135 | 135 | 154 | 113 | 146 | **102** | 140 |

- **A margem continua sendo a referência.** O estimador aprendido apenas a reproduz.
- **Ensemble profundo, domínio e discordância ficam piores que o aleatório**, exceto em Curitiba-todas.
- **Orçamentos fixos de encaminhamento** (`orcamentos_encaminhamento.csv`): com 10% dos pontos encaminhados, a margem reduz as erradas da RMPA de 870 (aleatório) para 382.
- **Controle conformal (CRC):** calibrado em prevalência 0,42, cumpre α = 2% nas coortes, mas aceita só 9–225 pontos baixos. A mudança de prevalência o torna inútil aqui; vale como diagnóstico.
- **Revisões concluídas: 0.**

## 11. Lacuna nas comunidades (`s_lacuna/`)

- **Observabilidade (fração de pontos com referência S2 válida):**

  | Coorte | FCU | Baixa renda | Comparador |
  |---|---:|---:|---:|
  | RMR | 0,61 | 0,44 | 0,62 |
  | Recife | 0,82 | 0,90 | 0,91 |
  | RMPA | 0,95 | 0,78 | 0,90 |

- **Comunidades sem nenhum positivo:** 82–99% das FCU.
- **Areia do SoilGrids:** mascarada em 59–97% dos pontos urbanos. Nas FCU, esse atributo do Q é quase sempre a mediana imputada.

A lacuna está na referência, e é maior no Nordeste e na baixa renda da RMR.

## 12. O que ficou sustentado, inconclusivo, refutado ou não testável

| Alegação | Estado | Artefato · unidades · limite |
|---|---|---|
| Ganho estrangeiro do Q persiste sem seleção brasileira de hiperparâmetros | **Sustentado no domínio testado** | `p0_regularizacao` · 5 coortes, 4 episódios · desenvolvimento |
| Atualizar só com suporte preserva o Q e evita a degradação v8 | **Sustentado (desenvolvimento)** | `p0_continuidade` · 3 coortes × 18 orçamentos · paridade 0,0 |
| Modelos tabulares mais fortes superam o Q | **Inconclusivo**; redes diretas pioram | `contrastes_bootstrap` · 5 coortes |
| Árvores físicas superam −declive no Nordeste | Sustentado em Recife/RMR (1 episódio) | idem |
| Representação espacial pré-treinada acrescenta ao Q | **Refutado no desenho testado** (RMPA); inconclusivo em Manaus | `c_espacial` |
| Pré-treino tem informação de terreno | Sustentado descritivamente (RMPA, Manaus) | sondas sem Q |
| Compartilhamento entre tarefas transfere conhecimento | **Não identificado** (controle rompido = conjunta) | `d_conjunta` · 3 sementes |
| Generalização entre domínios (CORAL/DRO) ajuda | **Refutado no desenho testado** | `e_generalizacao` |
| Aquisição por margem aprende mais por consulta | Sustentado na RMPA (desenvolvimento); **não testável** nas FCU de Curitiba | `f_aquisicao` · 6 sementes |
| Crescimento sem esquecimento | **Refutado**; DER++ reduz o esquecimento sem zerar | `g_continua` |
| Nova técnica de incerteza supera a margem | **Refutado** (empata ou perde) | `h_incerteza` |
| Redução da lacuna social | **Não testável** (0 revisões; referência FCU escassa) | `s_lacuna`, PPS v2 |
| Fila, pools, patches, chips e contratos | Apenas implementado (infraestrutura qualificada) | manifesto |

**Quanto do ganho estrangeiro persistiu após os controles:** todo. A regularização externa muda menos de 0,002, a função completa é preservada e o Q segue como a referência que nenhuma família nova superou com intervalo nas FCU.

**Extensões com informação nova:**
- filtro de suporte;
- margem como política de aquisição;
- DER++;
- recurrence do JRC;
- ablação de região África/Oriente Médio;
- a observação de que terreno e radar se substituem, e não se somam, na tarefa observacional.

## 13. Verificação

- **Reprodução a partir dos pesos salvos** (`verificacao_independente.json`, ok): Q exportado com diferença de 1,8 × 10⁻¹⁵; MLP residual recarregado, 1,2 × 10⁻⁷; encoder espacial recarregado, 0,0.
- **Testes:** 53 aprovados (v5–v9; 10 da v9, com 1 pulado antes da execução da P0). São QA de software, não sustentação científica.
- **Retomada:**

```text
py312 scripts/vulnerabilizados/v9_fila.py estado
py312 scripts/vulnerabilizados/v9_fila.py executar          # roda so o pendente, pelo serializador
py312 scripts/vulnerabilizados/v9_fila.py marcar NOME pendente
```

## 14. Próximos passos que os resultados justificam

1. Ablação de região no **Q** (não só no MLP), com várias sementes e controle de tamanho, para testar a transferência negativa da África/Oriente Médio.
2. Aquisição por margem com filtro de suporte como política padrão de adaptação; testar em episódio novo.
3. DER++ sobre a correção offset como regra de atualização contínua, com retenção em unidades nunca usadas.
4. Confirmação em 4–12 episódios novos (planejamento v8/revisão), com Q, filtro e margem congelados.
