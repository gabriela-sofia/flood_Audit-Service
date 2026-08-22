# FIX 4 — TWI estático + tentativa de aquisição Sentinel-1 SAR (Recife v3)

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-22
**Autorizado por**: Gabriela Sofia (reviewer/executora única)

## FIX 4a — TWI (Topographic Wetness Index)

`TWI = ln(a / tan(β))`, onde `a` = área de captação específica, aproximada como
`(acumulação_de_fluxo_D8_em_células + 1) × resolução_m`, e `β` = declividade local (gradiente
de diferenças centrais no mesmo DEM mesclado 10 m).

**Reuso real, não recomputação**: `a` foi construída a partir da grade de acumulação de fluxo
D8 **já calculada e cacheada pelo HAND do v2** (`local_runs/recife_modelo_oficial_v2/
_scratch_accum.npy`, `_scratch_hand_dem_10m.npy`, `_scratch_hand_transform_10m.json`) —
carregada diretamente, sem re-executar o passo de roteamento de fluxo D8/pointer-doubling.

- Grade: 2777×2086 células, 10 m, 3.579.890 células válidas.
- Limitação herdada do HAND (documentada, não escondida): sem preenchimento de depressões
  (no pit-fill), então `a` pode estar subestimada localmente em sumidouros não preenchidos.
- Piso de declividade mínima 0,1° aplicado para evitar `tan(β)→0` no terreno muito plano de
  Recife (atingido em 0,2% das células válidas — documentado, não escondido).
- TWI: min=−3,58, mediana=4,71, max=14,16.
- **Extração nos 282 pontos: 150 m de tolerância de busca (15 células, mesma disciplina do
  HAND) → 282/282 (100%) resolvidos.**

## FIX 4b — Tentativa de aquisição Sentinel-1 SAR

### Primeira tentativa: Microsoft Planetary Computer STAC — **INDISPONÍVEL (outage transiente confirmado)**

`https://planetarycomputer.microsoft.com/api/stac/v1` respondeu **200 OK rapidamente na raiz**
do catálogo, mas **todo endpoint que depende do backend de busca** (`/collections`,
`/collections/sentinel-1-grd`, `/search` via GET e POST) **retornou timeout consistente
(HTTP 000, sem resposta em 20-30s) ou 504 Gateway Timeout do Azure Front Door** — reproduzido
em múltiplas tentativas espaçadas no tempo, incluindo via `pystac-client`/`planetary-computer`
Python e via `curl` direto. **Documentado como indisponibilidade real do serviço no momento da
tentativa**, não um erro de código — não forçado, não contornado com credenciais.

### Segunda tentativa: Earth Search (Element84) + AWS Open Data — **SUCESSO**

`https://earth-search.aws.element84.com/v1` (STAC público, sem autenticação) respondeu
normalmente. Busca por `sentinel-1-grd` no bbox de Recife (2014–2024) retornou **588 cenas
reais**. Os assets `vv`/`vh` apontam para `s3://sentinel-s1-l1c/...` (bucket AWS Open Data);
**confirmado acessível de forma anônima via HTTPS simples** (`https://sentinel-s1-l1c.
s3.amazonaws.com/...`, GET 200 OK, `Accept-Ranges: bytes` — mesmo padrão de acesso anônimo
windowed-COG que já funcionou para o MapBiomas no v2).

**Limitação real de georreferenciamento**: os GeoTIFFs de medição GRD do Sentinel-1 **não são
georreferenciados por afim simples** (imagem em coordenadas slant/ground-range) — usam ~210
GCPs (Ground Control Points, EPSG:4326) por cena. A posição do pixel de cada ponto foi estimada
por **interpolação linear da grade de GCPs da própria cena** (`scipy.interpolate.griddata`,
fallback para `nearest` quando o ponto cai fora do fecho convexo dos GCPs) — uma aproximação
documentada (não ortorretificação pixel-perfeito), com janela de busca de ±40 px ao redor do
pixel estimado e valor tomado no pixel central (ou mediana da janela não-nula se o centro caiu
em nodata).

### Cobertura real obtida

- **138 cenas Sentinel-1 GRD distintas** necessárias (1 por data de evento mais próxima, entre
  os 273 pontos com `event_date`), buscadas e processadas (apenas banda VV, para limitar
  volume de requisições) via download concorrente (4 threads).
- **271/282 pontos (96,1%) obtiveram um valor real de backscatter VV** (Digital Number bruto,
  **NÃO calibrado para sigma0** — documentado, não afirmado como backscatter calibrado).
  2 falhas de leitura de rede pontuais, 9 pontos sem `event_date` (mesmos 9 do v2).
- **Comparação com Sentinel-2 óptico do v2**: cobertura SAR (96,1%) é **muito maior** que a
  cobertura óptica de patch do v2 (9,4%) — confirma que SAR penetra nuvens.
- **Gap de data real** (data da cena mais próxima vs. `event_date` do ponto): mediana **2,3
  dias**, p75=27,3 dias, máximo=480,3 dias (para positivos anteriores a 2015-04-28, início da
  cobertura Sentinel-1 real sobre Recife nos dados públicos indexados) — reportado
  integralmente, não escondido.

### Decisão de uso

Conforme a regra do projeto ("dado orbital é auxiliar"), **o SAR NÃO foi incluído em
`FEATURE_COLS_V3`** (o conjunto de features do modelo primário) — fica no dataset como coluna
auxiliar (`sar_vv_dn_uncalibrated`, `sar_scene_id`, `sar_date_gap_timedelta`) para uso
secundário/exploratório futuro, não forçado no modelo atual dado que (a) é DN bruto não
calibrado, (b) o gap de data é grande e heterogêneo para uma fração dos pontos, (c) o
georreferenciamento por GCP é uma aproximação.

## Arquivos
- `_scratch_fix4_twi.py`, `pipeline_recife_v3.py::compute_twi()/extract_twi_at_points()`
- `_scratch_sar_list_scenes.py`, `_scratch_sar_fetch.py` — busca e extração SAR
- `_scratch_sentinel1_scenes.csv` — 588 cenas indexadas
- `_scratch_sar_vv_values.csv` — 271 valores reais extraídos
- `_scratch_twi_report.json` — números completos do TWI
