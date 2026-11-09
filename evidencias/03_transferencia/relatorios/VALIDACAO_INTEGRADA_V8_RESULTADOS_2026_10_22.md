# Validação integrada v8: ponte física MT01/MT11 → comunidades brasileiras, poucos rótulos e limites

22/10/2026. Execução de `docs/PROMPT_EXECUCAO_PESADA_TESE_2026_10_22.md`.

- **Linhagem:** `local_runs/transferencia-vulnerabilizados/v8_validacao_integrada/` (162 arquivos, 349 MB, manifesto SHA-256).
- **Código:** `scripts/vulnerabilizados/v8_*.py`. **Testes:** `tests/test_v8_validacao_integrada.py`.
- **Figura:** `docs/figuras/v8_campanha_integrada_2026_10_22.png` / `.svg`.

**Natureza:** tudo aqui é **desenvolvimento ou reanálise**. RMPA/2024, Manaus/2021, Curitiba/2023, Recife e RMR/2022 já tinham sido examinados. Nenhum resultado desta campanha é confirmação.

**Preservação:**
- 15 fontes protegidas foram conferidas por hash no início e no fim, sem divergência: coortes v3, IDs v4, coeficientes v5, população MT01, pesos MT05, geografia v7, auditoria, chave selada v6 (não aberta) e referências v6.
- Nenhuma reserva aberta (EMSR868/871, MT12), nenhum peso ativo promovido, LaTeX intocado, nenhum commit.
- Ninguém foi contatado.

## 1. O que foi realmente treinado e com quanto dado

| Modelo | Dados de ajuste | Saída salva |
|---|---|---|
| Q reprodução (MT05) | 91.446 linhas, 130 eventos, 43 unidades | reproduziu os pesos congelados com diferença 0,0 |
| Q fora do Brasil (terreno, base, conexão) | 36.331 linhas, 59 eventos, 0 pontos no Brasil (CEMS 16.633, UFO 11.698, Sen1Floods11 8.000) | `q_externo/modelos.pkl`, `pesos.json` |
| Q fora da América do Sul | 30.075 linhas, 48 eventos | idem |
| Q com rótulos permutados (controle) | 36.331 linhas | idem |
| Ablações de fonte + 12 controles de tamanho | 19.000–36.000 linhas cada | `q_fontes/` |
| Q_sep (grafo de partições) | 19.407 linhas, 34 eventos | `q_sep.pkl` |
| GBM monotônico externo (busca de 4 configurações por validação agrupada externa) | 36.331 linhas | `campanha/modelos_externos_v8.pkl` |
| Adaptadores locais: 1.682 ajustes (3 coortes × 3 políticas × 3 orçamentos × 6 sementes × 11 braços) | 25/100/400 rótulos consultados (IDs v4) | `campanha/ajustes.parquet`, com θ de cada ajuste |
| Crescimento sequencial (4 estratégias + conjunta, 6 sementes) | 400 rótulos por etapa | `crescimento/` |
| Estimador de competência (2 versões) | 8 unidades externas de competência | `competencia*/` |

**Contagem honesta dos 1.682 adaptadores:**

| Estado | Ajustes | Onde |
|---|---:|---|
| Ajustados com duas classes | 872 | todos os 576 da RMPA |
| Cópias do prior | 486 | Manaus e Curitiba |
| Impossíveis (uma classe) | 324 | Manaus e Curitiba |

Nos orçamentos FCU de Curitiba, **zero positivo em todas as sementes**.

**Avaliação (partição de teste, só positivo/negativo):**

| Coorte | Pontos | Positivos | Positivos FCU |
|---|---:|---:|---:|
| RMPA | 135.564 | 6.117 | 241 |
| Manaus | 63.452 | 215 | 48 |
| Curitiba | 198.944 | 883 | 11 |
| Recife | 42.736 | 448 | 31 |
| RMR | 67.099 | 2.882 | 366 |

## 2. Achados que corrigem contratos, referências e comparações

1. **O Q congelado não era transferência sem rótulo para o Brasil.** A população de ajuste do MT05 contém **CEMS:EMSR720 (RS, maio/2024)**, o mesmo episódio da RMPA, e 50.646 linhas no Brasil (55%). O Q foi reajustado sob o mesmo contrato só com eventos inteiramente fora do Brasil. Aplicado à RMPA, o Q congelado ficou +0,006 acima do externo (IC [+0,005; +0,008]): é o efeito esperado de ter visto o episódio.
2. **A ligação de bacia não é a de episódio.** A unidade MT01 `CEMS:EMSR720` também agrupa eventos da Bolívia (2025), do Paraguai (2018) e do Uruguai (2019), ligados por rede hidrológica. A variante fora da América do Sul, sem eles, mudou ≤ 0,005. País, bacia e episódio ficam declarados como eixos separados.
3. **UFO e água preexistente** (`contrato/evidencia_agua_anterior_jrc.parquet`, JRC GSW 1.4):
   - **1.442 dos 11.640 positivos UFO (12,4%)** estão sobre água recorrente em ≥ 75% do histórico, contra 6 dos 12.394 negativos.
   - Os positivos de inundação do Sen1Floods11 têm 0 recorrente forte; a água permanente do Sen1Floods11 cai 100% nas faixas recorrente/sazonal, o que confere a leitura.
   - Nada foi reclassificado. A tarefa "água nova" deixa os positivos UFO como **não adjudicados**.
   - Retirá-los do ajuste mudou ≤ 0,005 nas coortes, contra controles de mesmo tamanho.
4. **SCA reconstruída × original:** a rota TWI do contrato concorda com o `dinf_sca` do raster em Manaus e Curitiba (Spearman 0,99), mas não em terreno plano (RMPA 0,91, RMR 0,91, Recife 0,79). O nome canônico `ln_sca` continua sendo a rota do contrato; a SCA original fica registrada como diagnóstico.
5. **Cobertura e solo:** a primeira rota de download (100/250 m) tinha dois defeitos:
   - o NoData 0 apagava construído = 0;
   - a grade diferente dava 62–68% de concordância com a API.

   A rota final, na grade de 30 m da API, concorda em 87–93% (construído) e 94–98% (areia). A primeira tentativa está preservada.
6. **JRC:** a primeira leitura devolvia nulo em 100% dos pontos (o redutor grava a propriedade como `first`), e depois "nulo" foi lido como falta de observação, quando significa água nunca detectada. Os dois defeitos foram corrigidos e registrados.

## 3. Contrato de alvos (`contrato/`)

**1.641.356 IDs únicos** (DS06 + contexto fora do DS06), cada um com rótulo e valor nativos, alvo harmonizado, estado observacional, instrumento, resolução, janela, episódio, processo, geografia, família de terreno, dependências da referência, elegibilidade por tarefa e motivo de exclusão.

| Tarefa | Elegível | Com ressalva | Fora |
|---|---:|---:|---:|
| Suscetibilidade retrospectiva | 81.914 | 431.616 (GFD na escala de 250 m; positivo misto do UFO) | 1.124.088; mais 3.738 só-positivo UK |
| Água visível | 120.431 | 521.726 | 999.199 |
| Água nova pré/pós | 51.487 | 439.347 | 1.137.622; mais 12.900 UFO não adjudicados |
| Notificação administrativa | 242.033 positivos | 735.770 ausências de registro (não são seco) | 663.553 |
| Observabilidade | 654.485 | — | 986.871 |

Coortes brasileiras S2 (`contrato_coortes_brasileiras.parquet`): os estados positivo, negativo, permanente, desconhecido e pré-desconhecido ficam preservados; desconhecido nunca vira negativo. As contagens por tarefa, fonte, país, episódio e comunidade estão em `contagens_*.csv`.

## 4. Ponte física e contexto (desenvolvimento)

**Sem rótulo local.** AUC agrupada nas FCU:

| Coorte | HAND | −declive | Q fora do Brasil | Q congelado | GBM externo | A1 v3 (linha territorial) |
|---|---:|---:|---:|---:|---:|---:|
| RMPA | 0,841 | **0,910** | 0,881 | 0,889 | 0,883 | 0,807 |
| Manaus | 0,874 | 0,808 | **0,921** | 0,923 | 0,905 | 0,915 |
| Curitiba (11 pos.) | 0,826 | 0,698 | 0,892 | 0,886 | 0,845 | 0,920 |
| Recife | 0,802 | 0,715 | **0,866** | 0,863 | 0,818 | — |
| RMR | 0,693 | 0,716 | **0,809** | 0,802 | 0,717 | — |

A melhor regra física escolhida **nos dados externos** foi o HAND (0,738 contra −declive 0,703 e TWI 0,702).

**Contrastes com IC 95% por comunidade/setor** (`analise/contrastes_sem_rotulo.csv`):

- **Q fora do Brasil − HAND:**
  - todas: RMPA +0,018 [+0,008; +0,027]; Recife +0,044 [+0,021; +0,067]; RMR +0,085 [+0,055; +0,106]; Manaus +0,036 e Curitiba +0,007 inconclusivos;
  - FCU: Manaus +0,047 [+0,024; +0,072]; Recife +0,064 [+0,023; +0,117]; RMR +0,116 [+0,073; +0,138]; RMPA +0,040 inconclusivo; Curitiba descritivo.
- **Q fora do Brasil − (−declive):** **perde na RMPA** (todas −0,016 [−0,027; −0,001]; FCU −0,029 inconclusivo). Vence em Manaus (+0,106), Curitiba (+0,057), Recife (+0,088) e RMR (+0,069).
- **Dentro das comunidades** (macro FCU ≥ 5/5; 10 RMPA, 4 Manaus, 6 RMR): na RMPA, Q 0,714 contra HAND 0,579 e −declive 0,659; na RMR, 0,751 contra 0,671 e 0,642. Agregado e macro são estimandos diferentes e não concordam na RMPA.
- **Conhecimento contra estrutura:** Q − Q permutado = +0,30 a +0,63 em todas as coortes.
- **Blocos:**
  - **cobertura/solo** (Q_base − Q_terreno): +0,044 Recife, +0,091 RMR (FCU +0,108 e +0,141); inconclusivo no Sul e em Manaus;
  - **conexão** (Q_conexão − Q_base): **piora** a RMPA (−0,023) e a RMR (−0,025). No MT11 externo ela ajudava.
- **Não linear:** GBM externo − Q: RMR −0,066, Recife −0,016, Manaus FCU −0,017; só Curitiba +0,008. É a segunda vez que o modelo não linear não se transporta (v6 RMR).
- **Rota parcial sem areia:** |dif| ≤ 0,004. A imputação pela mediana do contrato não muda o resultado.
- **Fontes** (efeito líquido do controle de tamanho, descritivo):
  - retirar UFO piora RMR FCU (−0,018) e Curitiba FCU (−0,010);
  - retirar CEMS **melhora** RMR FCU (+0,059) e Curitiba FCU (+0,028), mas piora Recife todas (−0,021) e RMPA (−0,008);
  - **mais volume não demonstrou complementaridade:** a contribuição de cada fonte muda de sinal entre territórios.

## 5. Poucos rótulos (`analise/contrastes_poucos_rotulos.csv`, `curvas_aprendizagem.csv`)

**RMPA:** duas classes em 100% dos orçamentos; positivos médios 2,5 / 10,3 / 41,3 na política uniforme.

| Contraste (τ = 25, uniforme, todas) | B = 25 | B = 100 | B = 400 |
|---|---|---|---|
| F1 prior externo − prior neutro (mesma escala e τ) | **+0,011 [+0,001; +0,025]** | +0,003 inc. | **+0,003 [+0,002; +0,006]** |
| F2 escala externa − escala local (sem prior) | **−0,016 [−0,031; −0,006]** | −0,005 inc. | −0,003 |
| F3 prior externo − prior permutado | **+0,020 [+0,008; +0,037]** | +0,005 | +0,004 |
| F4 prior externo com rótulos − mesmo sem rótulo (B = 0) | +0,005 | **+0,016** | **+0,021 [+0,015; +0,026]** |

- **Nas FCU da RMPA:** F1 +0,050 (B = 25) inconclusivo; F4 +0,021 [+0,001; +0,052] com B = 400.
- **Leitura:** o benefício está no **conteúdo do prior** e é pequeno, concentrado em B = 25. A escala externa sozinha atrapalha.
- **Manaus e Curitiba:** os F1 de +0,29 a +0,42 **não são aprendizagem**. Comparam cópia do prior com ajuste impossível de uma classe. Com os poucos positivos que existem, **atualizar piora** (Curitiba B = 400, F4: −0,011 [−0,018; −0,005]).
- **GBM local:** inferior ao prior em Manaus (FCU −0,147) e Curitiba (−0,040).
- **Custo de rótulos** (`custo_rotulos.csv`): 400 consultados e 400 válidos, mas 0 a 2,8 positivos em Manaus e Curitiba. A política FCU de Curitiba não obteve as duas classes em nenhuma semente.

## 6. Crescimento sequencial e retenção (`crescimento/`)

**Desenho:** Manaus → Curitiba → RMPA, 400 rótulos por etapa. A retenção é medida em 6 unidades externas que nunca ensinaram nada.

| Estratégia | Retenção externa: início → após Manaus → final | RMPA: início → final | FCU final (RMPA / Manaus / Curitiba) |
|---|---|---|---|
| Sem atualização | 0,732 → 0,732 → 0,732 | 0,904 → 0,904 | 0,876 / 0,918 / 0,879 |
| Regularizada (τ = 25) | 0,732 → 0,680 → 0,724 | 0,904 → 0,922 | 0,891 / 0,930 / 0,893 |
| Direta (τ = 1) | 0,732 → 0,635 → 0,697 | 0,904 → 0,933 | 0,910 / 0,931 / 0,887 |
| Replay (200 exemplos) | 0,732 → 0,680 → 0,721 | 0,904 → 0,928 | 0,900 / 0,931 / 0,902 |
| Conjunta (referência não sequencial) | final 0,708 | 0,937 | 0,914 / 0,934 / 0,893 |

- **Etapas pobres pioram:** atualizar com ~3 positivos de Manaus piora o **próprio teste de Manaus** (0,789 → 0,777), a RMPA (0,904 → 0,865) e a retenção.
- **Ganho só com contraste:** o ganho aparece na etapa RMPA, com ~41 positivos.
- **Esquecimento final na retenção externa:** −0,008 (regularizada), −0,011 (replay), −0,035 (direta).
- **Replay** não superou claramente a atualização regularizada.

## 7. Competência, recusa e quem recebe resposta (`competencia/`, `competencia_tranquilizadora/`)

- **Separação:** o Q_sep foi ajustado nas unidades físicas, o estimador de competência em 8 outras e a avaliação nas 6 de retenção e nas coortes. A retenção não ensina competência; na v7 ensinava.
- **Versão 1 (erro da regra de topo de 20%) — falha registrada:** ensinado em prevalência de ~45%, o estimador aprendeu que o erro está no alerta.

  Nos destinos de 2–5%, reduziu o erro dos aceitos (RMPA 0,170 → 0,013 com 70% de cobertura), mas **aceitou só 321 de 6.117 positivos (FCU: 11 de 241)**.

  A calibração por decil ordena bem (correlação 0,85–0,93), mas superestima o erro (previsto 0,26 contra observado 0,17 na RMPA).
- **Versão 2 (uma modificação declarada pós-diagnóstico):** o alerta nunca é recusado, e a competência só decide entre "menor suscetibilidade com suporte" e "evidência insuficiente".

  Com 70% de cobertura, as respostas tranquilizadoras erradas ficam assim:

  | Coorte | Aprendido | Aleatório | Só margem |
  |---|---:|---:|---:|
  | RMPA, todas | 41 | 631 | 35 |
  | RMPA, FCU | 0 | 35 | 0 |
  | RMR, todas | 460 | 892 | 315 |
  | RMR, FCU | 26 | 130 | 25 |
  | Retenção externa | 998 | 1.719 | 1.044 |

  - **O estimador aprendido praticamente reproduz a margem.** Domínio (DI) e qualidade não ajudaram; o DI piorou.
  - **Cobertura FCU com resposta:** 0,62 (RMPA), 0,70 (Manaus), 0,73 (RMR), contra 0,70 no total.
  - **Encaminhados como evidência insuficiente:** RMPA 957 positivos (FCU 50); RMR 989 (FCU 182). **Revisões concluídas: 0**; encaminhar não é revisar.
- **Três estados** por ponto: maior suscetibilidade (alerta), menor com suporte e evidência insuficiente, com contagens por classe e por comunidade (`por_comunidade_c70.csv`).

## 8. Revisão independente e suficiência

- **Amostra v1** (`revisao_probabilistica/`), dois estágios com sorteio simples de setores: não viesada, mas imprecisa (DP de 0,075 para uma prevalência substituta de 0,063). Preservada.
- **Amostra v2** (`revisao_probabilistica_v2/`), PPS sistemático por tamanho com unidades de certeza e 4 pontos por setor:
  - 480 pontos, 120 setores (FCU 50, baixa renda 35, comparador 35);
  - todos os estados e toda a distribuição de escores entram;
  - π conhecida; os pesos somam exatamente o tamanho de cada estrato;
  - DP da prevalência substituta: **0,021**, igual ao EP estimado (0,022);
  - chave selada, formulário cego e trilha de adjudicação.

  **Estado: 0 respostas, sem revisor.** O pacote v6 (215 pontos) continua sendo o diagnóstico dos extremos.
- **Suficiência para a próxima confirmação** (`suficiencia_confirmacao.csv`):
  - contraste Q fora do Brasil − HAND nas FCU, com episódios como unidade (4 em desenvolvimento, efeito médio 0,061, DP 0,022): **2 episódios** no cenário otimista, **10** no conservador;
  - macro por comunidade FCU (DP 0,165): **26 a 132 comunidades com ≥ 5/5**; o desenvolvimento inteiro tem 20.
- **Outro episódio:** nenhum candidato admissível novo foi encontrado no acervo. Blumenau/2023 e Manaus/2022 seguem reprovados na triagem v6, e as janelas não foram ampliadas. Petrópolis segue sem referência hidrológica separada de movimento de massa.

## 9. Síntese: o que passou, o que falhou

| Alegação | Estado | Artefato · população · contraste · unidades · limite |
|---|---|---|
| Relações físicas aprendidas fora do Brasil ordenam a água nova em comunidades brasileiras sem rótulo local | **Sustentado no domínio testado (desenvolvimento)** | `analise/contrastes_sem_rotulo.csv` · 5 coortes, 4 episódios · Q − HAND > 0 nas FCU em Manaus, Recife e RMR, e em todas na RMPA, Recife e RMR · agrupamentos por comunidade/setor · desenvolvimento já examinado; não confirmatório |
| Vantagem sobre a melhor referência física simples | **Inconclusivo / heterogêneo** | idem, N2 · −declive vence na RMPA agregada; Q vence nas demais e dentro das comunidades da RMPA · 4 episódios · a melhor regra difere por território |
| Bloco de cobertura/solo ajuda | Sustentado no Nordeste; inconclusivo no Sul (desenvolvimento) | N6 · Recife e RMR · 1 episódio (PE-2022) |
| Bloco de conexão MT11 ajuda no Brasil | **Refutado nestas coortes** (piora RMPA e RMR) | N7 · 2 episódios |
| Não linear externo transporta melhor | **Refutado nestas coortes** | N8 · RMR, Recife, Manaus |
| Prior externo facilita aprender com poucos rótulos | Sustentado pequeno na RMPA com B = 25 (+0,011); **não testável** em Manaus e Curitiba (orçamentos sem positivos) | F1/F3 · RMPA · 1 episódio · efeito pequeno |
| Rótulos locais acrescentam ao modelo sem rótulo | Sustentado na RMPA (B ≥ 100); **refutado** em Curitiba (B = 400 piora) | F4 · 2 episódios |
| Crescimento sequencial sem esquecimento | **Refutado** na formulação testada | `crescimento/` · etapas com ~3 positivos pioram o próprio destino e a retenção; esquecimento final de −0,008 a −0,035 · 6 sementes, 1 sequência |
| Competência aprendida reconhece limites sem abandonar positivos | v1 **refutada** (recusa os positivos); v2 **inconclusiva** (não supera a margem) · ganho real só do encaminhamento por margem | `competencia*/` · 5 coortes + 6 unidades externas |
| Redução da lacuna nas FCU | **Não testável** (sem revisão humana; referência FCU escassa: 11 positivos em Curitiba) | amostra v2 pronta, 0 respostas |
| Aprendizagem compartilhada / encoder multimodal | **Apenas não implementado**: sem tarefa e referência que o teste aqui | matriz C17–C20 |
| Contratos, ponte, amostra PPS, partições | **Apenas implementado** (qualificação de dados, não resultado da tese) | `contrato/`, `ponte/`, `revisao_probabilistica_v2/`, `grafo_particoes.json` |

**Matriz das 32 camadas:** `matriz_32_camadas_v8.csv`. Em 21 camadas houve uso efetivo (ensina, condiciona, qualifica, avalia, contextualiza ou rastreia), com entradas, saídas e contribuição. **11 ficam "disponível"**, sem crédito pelo ganho de outra.

## 10. Verificação e reprodução

- **Verificação independente** (`verificacao_independente.json`, `ok = true`):
  - escores sem rótulo recalculados a partir do `modelos.pkl` em todas as coortes, com diferença 0,0;
  - 42 ajustes sorteados, com AUC recalculada dos θ salvos, diferença 0,0;
  - Q congelado reproduzido pelo pipeline v8, diferença 0,0.
- **Testes:** **85 aprovados** (10 v8 + regressão v1–v7 + auditoria). Os testes v8 cobrem PPS (inclusão empírica contra π em 4.000 sorteios), regras semânticas do contrato, leitura do JRC, partições disjuntas por evento, reprodução da campanha a partir dos pesos e formulários sem chave.
- **Execução:** carga pesada só pelo serializador e na tomada; a sessão começou na bateria (47%) e só fez trabalho leve até a energia voltar. Tempos e retomadas estão em `execucao_etapas.jsonl`.
- **Interrupção registrada:** o bootstrap inicial (1.000 reamostragens) levaria mais de 1 h por coorte; o processo desta sessão foi interrompido e refeito com 300 reamostragens, declarado no script.

**Comandos (a partir da raiz):**

```text
py312 = C:\Users\gabriela\AppData\Local\Programs\Python\Python312\python.exe -X utf8
S     = scripts/manutencao/executar_serializado.py --esperar-min 30 --
py312 scripts/vulnerabilizados/v8_00_estado.py
py312 S py312 scripts/vulnerabilizados/v8_01_q_externo.py
py312 S py312 scripts/vulnerabilizados/v8_02_ponte_fisica.py
py312 scripts/vulnerabilizados/v8_02b_cobertura_solo.py          # Earth Engine; lotes retomaveis
py312 S py312 scripts/vulnerabilizados/v8_03_contrato_alvos.py
py312 scripts/vulnerabilizados/v8_03b_evidencia_agua_anterior.py # Earth Engine
py312 S py312 scripts/vulnerabilizados/v8_01b_q_fontes.py
py312 scripts/vulnerabilizados/v8_04_campanha.py protocolo && py312 S py312 scripts/vulnerabilizados/v8_04_campanha.py executar
py312 S py312 scripts/vulnerabilizados/v8_04b_analise.py
py312 S py312 scripts/vulnerabilizados/v8_04c_fontes_avaliacao.py
py312 S py312 scripts/vulnerabilizados/v8_05_particao_crescimento.py
py312 S py312 scripts/vulnerabilizados/v8_06_competencia.py
py312 S py312 scripts/vulnerabilizados/v8_06b_competencia_tranquilizadora.py
py312 scripts/vulnerabilizados/v8_07_amostra_probabilistica.py && py312 scripts/vulnerabilizados/v8_07b_amostra_pps.py
py312 scripts/vulnerabilizados/v8_08_suficiencia_camadas.py
py312 S py312 scripts/vulnerabilizados/v8_09_verificacao_manifesto.py
py312 scripts/vulnerabilizados/v8_10_figura.py
py312 -m pytest tests/test_v8_validacao_integrada.py tests/test_v7_aproveitamento_acervo.py tests/test_v6_evidencia_robusta.py tests/test_v5_comunidades.py -q
```

## 11. Condições exatas para destravar o que ficou não testável

1. **Confirmação:** 2 a 10 episódios independentes com referência S2 qualificada (ou equivalente) e FCU com as duas classes. Os contrastes Q_ext − HAND e P_ext − L_escext ficam congelados antes, com o modelo e as regras desta linhagem.
2. **Eficiência com poucos rótulos em Manaus e Curitiba:** orçamentos que contenham positivos. Isso exige referência com mais água nas FCU; aumentar B não resolve: 400 consultados deram 0 a 2,8 positivos.
3. **Lacuna e competência socialmente avaliadas:** um revisor humano cego responder ao pacote PPS v2 (480 pontos). A designação é da usuária.
4. **Água sob dossel e preexistente:** adjudicação humana dos 1.442 positivos UFO recorrentes antes de usá-los na tarefa "água nova".

Texto editorial: `docs/TEXTO_ARTIGO_VALIDACAO_INTEGRADA_V8_2026_10_22.md`.
