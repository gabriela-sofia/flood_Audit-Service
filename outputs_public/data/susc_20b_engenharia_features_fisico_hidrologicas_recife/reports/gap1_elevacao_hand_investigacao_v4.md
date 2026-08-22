# GAP 1 — Elevação e HAND com sinal incoerente (investigação de causa-raiz)

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-22
**Autorizado por**: Gabriela Sofia (reviewer/executora única)

## Resumo do problema (herdado do v3)

Mesmo após o Fix 1 (pseudo-ausência estratificada por uso do solo, que eliminou o confundidor
urbano: corr(elevação,impermeabilização) −0,24→0,02), `elevation_m` permaneceu com sinal
positivo incoerente (+0,53 no v2 → +0,65 no v3, piorou). `hand_m` permaneceu incoerente e
quase nulo (35,7% de cobertura raster).

## Hipótese 1 — Missingness diferencial de HAND como canal de vazamento

Contagens reais (150 m de tolerância, `dataset_v3_features_finais.csv`):

| Grupo | Resolvido (qualquer, ≤150m) | Hit direto (dist=0, sem fallback) |
|---|---:|---:|
| `flood_positive` (n=141) | 141/141 (100%) | 55/141 (39,0%) |
| `real_clean_sedec_negative` (n=22) | 22/22 (100%) | 14/22 (63,6%) |
| `spatiotemporal_pseudo_absence...` (n=119) | 118/119 (99,2%) | 26/118 (22,0%) |

**Achado real**: há sim uma diferença sistemática na taxa de hit direto vs. fallback entre
classes (pseudo-ausências recorrem ao fallback 78% das vezes vs. 61% dos positivos vs. 36% dos
negativos reais; Mann-Whitney nas distâncias de busca label1 vs label0: p=0,103, não
significativo a 5%). Porém `hand_m` médio não difere muito entre hit direto (8,71 m) e fallback
(9,92 m) — **o mecanismo de fallback em si não introduz um viés direcional forte o suficiente
para explicar a magnitude do coeficiente de elevação**. Contribuinte real, mas secundário.

## Hipótese 2 — Confundidor de clustering espacial / identidade de bairro

Todos os 282 pontos foram associados a um bairro real (join espacial contra
`data/raw/recife/official_address_base/bairros_do_recife.geojson`, 94 polígonos, 100% de
resolução, buffer de fallback ≤500 m quando fora do polígono).

- **75,2% dos 141 positivos** vêm de apenas **10 bairros** (de 94 totais); os positivos ocupam
  29 bairros distintos, contra 62 para os negativos/pseudo-ausências (maior dispersão espacial
  dos negativos, muito menor concentração dos positivos).
- Regressão OLS `elevation_m ~ label` (sem efeito fixo): coef=+9,64 m, p=0,00026, R²=0,047.
- Regressão OLS `elevation_m ~ label + C(bairro)` (efeito fixo de bairro, 67 categorias): coef
  de `label` cai para **+3,81 m, p=0,164 (não significativo)**, R² sobe para 0,596. Restringindo
  aos 24 bairros que têm as duas classes presentes (185 pontos): mesmo coeficiente (+3,81),
  p=0,193.

**Achado real**: a maior parte da associação bruta elevação↔label é absorvida por "em qual
bairro o ponto está" — ou seja, positivos vêm de um subconjunto específico de bairros que
tendem a ter cota mais alta (ex.: Cohab-Ibura de Cima, Nova Descoberta, Barro, Várzea, Água
Fria — comunidades de morro/encosta conhecidas em Recife), não de uma elevação alta
generalizada em toda a cidade.

## Achado decisivo — decomposição por tipo de fonte (real record vs. pseudo-ausência)

Ao decompor `elevation_m ~ label + is_real_record` (onde `is_real_record`=1 para
`flood_positive` OU `real_clean_sedec_negative`, =0 só para pseudo-ausência):

| Termo | Coef. | p-valor |
|---|---:|---:|
| Intercept | 16,63 | <0,001 |
| `label` (positivo vs. negativo, controlando fonte) | **+0,52** | **0,918 (não significativo)** |
| `is_real_record` (registro SEDEC real vs. pseudo-ausência sintética) | **+10,81** | **0,033** |

Confirmando: `flood_positive` (média 27,95 m) e `real_clean_sedec_negative` (média 27,44 m —
**ambos registros SEDEC reais**) têm elevação estatisticamente indistinguível (Mann-Whitney
p=0,907), enquanto ambos diferem significativamente da pseudo-ausência (média 16,63 m;
p=0,004 e p=0,024, respectivamente). **A elevação não separa inundação de não-inundação — ela
separa "ponto é um registro real da base SEDEC" de "ponto é uma pseudo-ausência
espaço-temporal sintética".**

## Hipótese 3 — artefatos de vazio/borda de tile PE3D no DEM

Usando o DEM mesclado 10 m real (`_scratch_hand_dem_10m.npy`, 2777×2086, cache do v2, mesmo
raster usado no pipeline), calculou-se a distância euclidiana de cada um dos 282 pontos até a
célula inválida (`-9999`) mais próxima (`scipy.ndimage.distance_transform_edt`). Distância
média: label=1 → 2.840,5 m; label=0 → 2.952,2 m (Mann-Whitney p=0,789). **Nenhum ponto está a
menos de 30 m de uma célula de vazio** (mínimo observado > 500 m em ambos os grupos).
**Hipótese descartada**: não há proximidade sistemática a vazios/bordas de mosaico, e não há
diferença entre classes.

## Auditoria caso a caso

**10 positivos de maior elevação**: concentrados em Cohab-Ibura de Cima (85,7 m), Várzea
(84,7 m), Nova Descoberta (82,7 m ×2), Barro (74,8 m, 70,5 m), Cohab-Ibura de Cima (71,7 m ×2),
Dois Unidos (71,5 m), Várzea (70,1 m) — **todas comunidades de encosta/morro reais e
conhecidas de Recife**, geograficamente coerentes com o fenômeno documentado de alagamento e
deslizamento em comunidades de morro (distinto de inundação costeira/fluvial de baixada).

**10 negativos/pseudo-ausências de menor elevação**: Peixinhos (1,07 m), Cabanga (1,80 m),
Boa Viagem (1,94 m, 2,01 m), Recife/Bairro do Recife centro histórico (2,16 m), Boa Vista
(2,18 m), Boa Viagem (2,32 m), São José (2,32 m), Imbiribeira (2,81 m, 2,84 m) — **bairros de
planície costeira/estuarina reais**, geograficamente coerentes com baixa cota.

Ambos os extremos são geograficamente reais e plausíveis — **não são erros de geocodificação**.
O problema não está nos pontos individuais, está no desenho amostral: os registros SEDEC reais
(positivos + negativos reais) vêm desproporcionalmente de um subconjunto de bairros de encosta
de alta densidade populacional, e as pseudo-ausências, amostradas uniformemente (com
estratificação apenas por classe MapBiomas) por todo o município, sub-representam esses
mesmos bairros.

## Veredito

**Não é um artefato residual de DEM/tile/vazio** (descartado empiricamente, Hipótese 3).
**É parcialmente um efeito de missingness diferencial de HAND** (Hipótese 1, real mas de
magnitude insuficiente para explicar o coeficiente de elevação).
**É principalmente um confundidor de desenho amostral remanescente**: "registro real
SEDEC vs. pseudo-ausência sintética" (não "inundação vs. não-inundação"), fortemente absorvido
por identidade de bairro (R² salta de 0,047 para 0,596 com efeito fixo de bairro; o
coeficiente de `label` deixa de ser significativo). O confundidor de USO DO SOLO do v2 (Fix 1)
foi corrigido com sucesso; o confundidor real remanescente é de **catchment de notificação/
concentração espacial de onde a SEDEC de fato recebe e registra chamados** — que inclui tanto
positivos quanto os 22 negativos reais, mas não as pseudo-ausências.

Isso **não invalida** os 10 casos de maior elevação como geograficamente reais (comunidades de
morro genuinamente sujeitas a alagamento/deslizamento é um fenômeno real e documentado na
literatura sobre Recife) — mas o COEFICIENTE estatístico de `elevation_m` no modelo atual
**não deve ser interpretado como "elevação mais alta → mais risco de inundação"**. Deve ser
interpretado como um artefato de desenho de amostragem (real-record vs. pseudo-ausência) ainda
não corrigido, que requereria uma nova geração de pseudo-ausências estratificada também por
bairro/densidade de notificação SEDEC (não só por classe de uso do solo) — fora do escopo desta
rodada (não implementado para evitar overfitting: 67 bairros x n=282 é insuficiente para efeito
fixo completo em produção, ver `gap7_poder_estatistico.md`).

## Robustez estatística (ver Gap 7 para detalhe completo)

Bootstrap (1000 reamostragens, n=270): IC 95% de `elevation_m` = [+0,327, +1,079], **nunca
cruza zero (0% de inversões de sinal)** — o efeito é estatisticamente robusto e estável em
n=282, não é ruído de amostra pequena. Isso reforça que a causa é sistemática (confundidor de
desenho), não uma flutuação aleatória que "resolveria sozinha" com mais iterações.

## Arquivos
- `_scratch_dataset_with_bairro_void.csv` — dataset com bairro (join espacial 94 polígonos) e
  distância a vazio de DEM
- `_scratch_dataset_with_hand_filled.csv` — HAND recomputado com preenchimento de depressões
  (ver `gap3_hand_depressoes.md`)
