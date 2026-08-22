# RECIFE MODELO — Trajetória Completa v1 → v9

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-23
**Autorizado por**: Gabriela Sofia (reviewer/executora única do projeto)
**Não fabricação**: nenhum dado fabricado em nenhum estágio; toda limitação documentada
explicitamente; nenhuma alegação de "validado"/"confirmado"/"ground truth" usada.

## Tabela-resumo

| Estágio | Mudança principal | n primário (pos/neg reais) | LOO-AUC primário |
|---|---|---:|---:|
| v1-v3 | Pipeline oficial inicial, CHIRPS, TWI D8, pseudo-absência por uso-do-solo | ~163/282 (misto real+pseudo) | ~0,60 (secundário inflado por confundimento de amostragem) |
| v4/Gaps | Auditoria: confundimento de elevação (Gap1), colapso de variância de impermeabilização (Gap2), depression-filling HAND (Gap3), sanity-check SAR (Gap4), poder estatístico n=282 avaliado como insuficiente (Gap7) | — | — |
| v5 | Split explícito primário (real-vs-real) vs. secundário (pseudo-absência); Firth + bootstrap + LOO-CV como metodologia padrão | 163 (127/36) | **0,602** |
| v6 | Modelo em dois estágios (P(susceptibilidade estática) × P(perigo dinâmico\|susceptibilidade)) | (mesmo dado) | (avaliação de probabilidade conjunta, não um único AUC comparável) |
| v7 | Improvement 1 (mais eventos reais via retry de geocoding), Improvement 2 (HAND/TWI D8→D-infinity), Improvement 3 (ortogonalização de chuva), Improvement 4 (pareamento por bairro — só na pseudo-absência SECUNDÁRIA) | 181 (145/36) | **0,6225** |
| v8 | Task A: +80 negativos reais EMLURB "156" (sem critério de bairro); Task B: SAR Sentinel-1 tentado, n=20 processado (inconclusivo, não integrado) | 261 (145/116), 251 completos | **0,7032** (posteriormente identificado como parcialmente inflado por confundimento geográfico — ver v9/Task 2) |
| **v9** | Task 1: SAR completo (124/124, n=113 válidos) — direção ainda invertida, ainda não integrado (achado negativo mais forte, não apenas inconclusivo); Task 2: confundimento geográfico dos 80 negativos do v8 diagnosticado e corrigido (reseleção bairro-sobreposta, 88 negativos); Task 3: HAND testado em 3 thresholds de drenagem — limitação genuína documentada, não corrigível barato | **269 (145/124), 260 completos** | **0,6578** |

## O que mudou em cada estágio, honestamente

- **v1-v4**: pipeline funcional, mas com confundimento de amostragem sério (pseudo-absências de
  qualquer lugar do município, gerando separabilidade artificial). Gaps 1-7 diagnosticaram isso
  sem ainda corrigir de forma completa.
- **v5**: primeira separação limpa entre análise primária (real-vs-real, a que importa para
  qualquer alegação causal) e secundária (presence-background, mais dado mas mais viés). LOO-AUC
  primário modesto (0,602) — honestamente fraco, mas metodologicamente correto pela primeira vez.
- **v6**: reformulação conceitual (dois estágios), não uma melhoria de métrica única.
- **v7**: 4 melhorias de engenharia de feature (mais dado real, D-infinity, ortogonalização,
  pareamento por bairro) moveram o LOO-AUC muito pouco (0,602→0,6225) — diagnóstico correto na
  época: "o gargalo é poder estatístico (n pequeno), não especificação de features".
- **v8**: testou esse diagnóstico com aquisição real de mais dado (n quase dobrou nos
  negativos). LOO-AUC subiu real e mensuravelmente (+0,081) — confirmação inicial de que mais
  dado real ajuda. Mas o SAR temporal não pôde ser processado além de n=20 (limite de
  engenharia/tempo, não de dado disponível) e permaneceu inconclusivo. **Não detectado na época**:
  os 80 novos negativos tinham um viés geográfico severo (só 3/39 bairros sobrepostos com os
  positivos) que inflava artificialmente parte do ganho de AUC.
- **v9** (esta rodada): as três lacunas deixadas em aberto por v8 foram fechadas com
  investigação real, não com narrativa:
  1. **SAR completo** (124/124 pontos, n=113 pares válidos, 5,65× o n anterior) — a direção
     invertida se **mantém** e o p-valor não se aproxima de significância; isso é agora uma
     conclusão negativa honesta e bem mais sólida (não apenas "n pequeno demais"), não integrada
     ao modelo.
  2. **Confundimento de elevação**: isolado como reintrodução de viés de amostragem geográfica
     nos 80 negativos do v8 (não um bug de CRS — pipeline validado diretamente contra 3
     landmarks conhecidos). Corrigido por reseleção (88 negativos reais, mesmo pool, critério de
     sobreposição de bairro). O coeficiente de elevação deixou de ser "significativo e errado"
     (p=0,047) para "honestamente instável e não significativo" (p=0,372, sign-flip 18%) —
     ainda sinal fisicamente incorreto na média, mas sem falsa confiança estatística.
  3. **HAND**: 3 thresholds de acumulação de fluxo testados (~400× de amplitude); nenhum produz
     separação significativa nem direção estável; artefatos de emenda entre tiles descartados
     (correlação nula com distância à borda). Documentado como limitação genuína de
     escala/terreno, não maquiado como "quase resolvido".
  4. **Resultado líquido do LOO-AUC**: caiu de 0,7032 (v8) para **0,6578** (v9) ao remover o
     confundimento geográfico — uma queda real e esperada, análoga ao que já havia acontecido no
     próprio projeto quando o pareamento por bairro foi aplicado à análise secundária em v7
     (AUC caiu de ~0,74-0,78 para ~0,69 pelo mesmo motivo). Um número mais baixo, mas construído
     sobre um desenho amostral mais defensável.

## Features estatisticamente robustas (bootstrap sign-flip < 5%) — contagem atual

**2 de 6** features do modelo primário v9: `twi_dinf` (1,7%) e `rain_decay_index_api_chirps`
(0,0%). Ambas com sinal fisicamente correto e coeficiente estatisticamente estável sob
reamostragem. As outras 4 (`elevation_m`, `slope_deg`, `hand_m_dinf`,
`rain_peak_residual_orthogonalized`) permanecem instáveis ou com sinal incorreto — sem mudança
de contagem em relação a v8 nas duas features robustas (eram as mesmas duas), mas com melhora de
honestidade estatística em `elevation_m`.

## Veredito final: maduro/estável, ou ainda limitado por tamanho de amostra?

**Ainda fundamentalmente limitado por tamanho de amostra e composição geográfica dos negativos
reais disponíveis** — não por falta de rigor metodológico. Nove iterações de trabalho real (v1
a v9) já cobriram: correção de confundimento de amostragem (repetidas vezes, incluindo nesta
rodada), recomputação de HAND/TWI com D-infinity e 3 thresholds distintos, ortogonalização de
chuva por fonte, aquisição de SAR real em escala completa, Firth + bootstrap + LOO-CV como
metodologia padrão desde v5. O que **não mudou** ao longo de 9 iterações é o fato estrutural de
que o Recife só tem ~124-181 eventos positivos reais bem documentados e um pool de negativos
reais (não-fabricados) inerentemente mais restrito e geograficamente enviesado por natureza (são
ocorrências de infraestrutura elétrica, não amostras aleatórias do território). As duas
features que carregam sinal real e estável (`twi_dinf`, chuva) são exatamente as que capturam a
componente dinâmica/meteorológica do problema — a componente estática de terreno (elevação,
HAND) continua fraca depois de repetidas tentativas honestas de correção, o que é, em si, um
resultado científico defensável (não um fracasso de engenharia): em terreno tão plano quanto o
Recife, índices topográficos estáticos parecem carregar pouco sinal residual além do que a chuva
recente já explica, dado o n disponível. **Não é uma vitória completa, nem um fracasso — é o
retrato honesto de um modelo que já extraiu a maior parte do sinal extraível dos dados reais
disponíveis nesta escala, com os limites remanescentes documentados e não escondidos.**

## Arquivos desta rodada (v9)

`task1_sar_completo_report.md`, `task2_elevacao_crs_investigacao.md`,
`task3_hand_investigacao.md`, `dataset_v9_final.csv`, `analise_primaria_v9.md`,
`coerencia_fisica_v9.md`, este relatório, além de todos os scripts/CSVs intermediários listados
em cada relatório de task.
