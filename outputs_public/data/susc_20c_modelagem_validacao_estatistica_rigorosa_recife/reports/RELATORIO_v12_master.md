# v12 — Extração Final das 3 Leads Pendentes + Re-treino

**Baseline**: `dataset_v9_final.csv` (n=269, 145 pos / 124 neg, LOO-AUC=0,6578)
**Resultado**: `dataset_v12_final.csv` (n=278, 154 pos / 124 neg, **269 completos com todas as
features** [268→269 pela adição de 9 pontos completos], **LOO-AUC=0,6781**)

## Resumo por lead

| Lead | Resultado real | Novos registros |
|---|---|---|
| A — Diário Oficial Recife (DOME) | 2 decretos de enchente novos confirmados por leitura integral de PDF (35.669/2022-05-28; 39.714/2026-05-02); bairros extraídos de matéria oficial anexa ao decreto de 2022 | **8** (bairro-level) |
| B — ANA Feature Service | Endpoint REST real (`snirh.gov.br/arcgis/rest/services`), 24 estações fluviométricas reais na RMR; 2 com série histórica real baixada (1990-2026 e 2018-2026); corroboração hidrológica de 14 datas já existentes + confirmação forte do evento 2022-05-28 (percentil 99,48%) | **0** (usado só como corroboração, conforme instruído) |
| C — Global Flood Database | Mirror sem GEE encontrado (GitHub + bucket GCS público `gfd_v1_4`); 1 evento MODIS-validado (DFO_3291, 2008) com pixels reais de inundação nova sobre Recife (bairro Brasília Teimosa) | **1** (evento-extensão-centróide) |
| **Total** | | **9 novos pontos positivos reais** |

## Re-treino (mesma metodologia exata do v9: Firth penalizado + bootstrap N=1000 + LOO/k-fold CV)

- **LOO-AUC: 0,6781** (v9: 0,6578; Δ = +0,0203)
- Repeated 5-fold (50 repetições): média 0,6747 (desvio 0,0115; min 0,6496; max 0,6934)
- **Coerência física preservada**: mesmos sinais de coeficiente que v9 em todas as 6
  features; `rain_decay_index_api_chirps` continua o preditor mais forte e estatisticamente
  significativo (p<0,0001, CI 95% não cruza zero); `twi_dinf` significativo (p=0,046);
  `hand_m_dinf` continua sem sinal robusto (48,7% de flips no bootstrap) — **limitação já
  documentada em v9, não piorada nem miraculosamente corrigida pelos 9 pontos novos** (honesto:
  n=9 é pequeno demais para resolver esse problema estrutural pré-existente).
- Nenhuma feature nova mudou de direção esperada para inesperada; nenhuma piora de
  significância. O ganho de AUC é modesto e consistente com adicionar 9 pontos reais bem
  distribuídos (8 costeiros/baixos + 1 evento de 2008), não uma mudança artificial radical.

## Arquivos desta rodada (`local_runs/recife_modelo_v12_extracao_final/`)

- `lead_a_diario_oficial_decretos.md`, `lead_b_ana_estacoes_reais.md`,
  `lead_c_global_flood_database.md` — relatórios por lead (fontes, URLs, method, achados)
- `dataset_v12_final.csv` — dataset final (278 linhas)
- `build_v12_dataset.py`, `pipeline_v12_primary.py` — scripts de construção e re-treino
- `primaria_v12_univariate_mannwhitney.csv`, `primaria_v12_firth_multivariate_coefs.csv`,
  `primaria_v12_bootstrap_coefs.csv`, `primaria_v12_predictive_auc.json`,
  `all_reports_v12_primary.json`, `v12_orthogonalization_stats.json`
- `_scratch_new_positives_leadA_full.csv`, `_scratch_new_positive_leadC_full.csv` — registros
  novos com features completas antes da fusão

## Avaliação honesta de fechamento

**Lead A**: fechado para a rodada — todos os PDFs candidatos (52) foram abertos e lidos;
decretos de enchente confirmados foram extraídos até o limite real da fonte (o corpo do
decreto não nomeia bairros; quando uma matéria anexa nomeia, foi extraída). Pendência
concreta remanescente: se um dia a Prefeitura publicar um boletim de vistoria/bairros para o
evento de 2026-05-02 em edição futura, vale reextrair — não existe ainda nas edições
disponíveis até 23/07/2026.

**Lead B**: fechado — endpoint encontrado, consultado e as duas séries realmente disponíveis
foram baixadas por completo. As 5 estações fluviométricas dentro do próprio município do
Recife estão genuinamente sem dado digital (não é bloqueio de acesso, é ausência de dado na
fonte desde 2005/2012). Pendência concreta: solicitar ao ANA (SIC/LAI) a digitalização de
réguas convencionais mais antigas, se se quiser estender a série — fora do escopo de acesso
automatizado.

**Lead C**: fechado — mirror real sem GEE encontrado e usado; catálogo Brasil completo (112
registros) checado; apenas 1 evento tinha raster MODIS validado real intersectando Recife.
Pendência concreta: o catálogo GFD final tem só 913 eventos globais (2000–2018); eventos DFO
brasileiros fora dessa lista (a maioria dos 112) nunca tiveram raster gerado — não há mais
nada a extrair desse mirror para esses casos, é um limite genuíno do produto, não de acesso.

**Conclusão geral**: as 3 avenidas de aquisição de dado real foram levadas à conclusão prática
nesta sessão (não apenas "portal encontrado, dado não extraído"). Resta como avenida real,
mas fora do escopo de acesso público automatizado: (1) abrir novas edições do DOME à medida
que forem publicadas para o evento de maio/2026; (2) solicitação formal de microdados à
ANA/CEMADEN via LAI; (3) nenhuma pendência sabida no Global Flood Database além do limite de
cobertura do próprio produto.
