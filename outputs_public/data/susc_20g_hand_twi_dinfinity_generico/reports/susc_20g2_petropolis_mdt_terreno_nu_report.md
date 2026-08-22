# SUSC-20G/2 — MDT de terreno nu para Petrópolis, e leitura observacional nos 2 pontos adjudicados

**Status**: INFRAESTRUTURA_RESOLVIDA_COM_RESSALVAS | **Data**: 2026-07-28
**Escopo**: fecha a pendência da seção 5 de
`docs/metodologia_cientifica/revp_linhagem_coleta_curitiba_petropolis_paridade_recife.md` — o
"MDT" de Petrópolis era um modelo de superfície. Nenhum modelo treinado, nenhuma feature criada,
nenhum rótulo. Curitiba e Petrópolis seguem com N=1 positivo cada.

---

## 1. Busca local primeiro (mínimo de dado necessário)

Varredura de `PROJETO/data/` por qualquer MDT de terreno nu de Petrópolis ainda não usado:

| Achado | Resultado |
|---|---|
| `produtos_mde_petropolis_rj.zip` em 3 caminhos distintos | **mesmo arquivo** (sha256 idêntico: `892745d1a3609369…`) — uma fonte, não três |
| `MDE/pt_sirgas_utm/` | 30,13 m — modelo de **superfície** (o que já se sabia) |
| `Hipsometria/hip_pt_utm23/` | **achado novo: 10 m**, 4466×4535, elevação contínua 40–2220 m |
| `Fusao/` | imagem ECW (ortomosaico), não é elevação |
| `bc_petropolis_rj.zip` | shapefiles de distritos/limites — **sem curvas de nível nem pontos cotados** |
| `ibge_ana_auxiliary/` | diretório **vazio** |
| DEM global já baixado (SRTM/COP30/ALOS/FABDEM) | **nenhum** em `PROJETO/data` nem em `local_runs` |

**O `Hipsometria` a 10 m é a mesma superfície do MDE**, só que na grade nativa: agregado a 30 m,
a diferença para o `MDE` é média 0,35 m, mediana 0,17 m, desvio 3,07 m sobre 1.218.647 pixels.
Ou seja: o pacote SGB/CPRM de Petrópolis melhora a **resolução** disponível (30 m → 10 m), mas
**não** contém terreno nu em lugar nenhum. Nada local resolvia a pendência.

## 2. Fonte externa 1 — Copernicus DEM GLO-30: tentada e **rejeitada**

Baixado por `/vsicurl` só a janela da bbox (`-43.25,-22.55,-43.05,-22.30`), tile
`Copernicus_DSM_COG_10_S23_00_W044_00_DEM`, bucket público `copernicus-dem-30m` (sem credencial):
900×720 pixels, ~2,6 MB lidos — não o tile inteiro.

Comparação contra o MDE do SGB/CPRM, mesma grade, 616.297 pixels:

| Métrica | GLO-30 − MDE_SGB |
|---|---:|
| Diferença média | −2,17 m |
| Mediana | −2,84 m |
| Desvio-padrão | 10,32 m |
| Pearson r | 0,9994 |

**Motivo da rejeição**: o produto é, por definição, um DSM — o próprio nome do arquivo é
`Copernicus_DSM`. E o número confirma: não há offset de escala de dossel entre ele e o MDE do
SGB; os dois descrevem a mesma superfície. (O `r = 0,9994` sozinho não prova nada — em relevo de
26 a 2233 m qualquer par de DEMs correlaciona alto; o que decide é a ausência de deslocamento
sistemático de 10-20 m em área florestada.) Trocar o MDE do SGB pelo GLO-30 não resolveria a
pendência, só trocaria um modelo de superfície por outro.

## 3. Fonte externa 2 — FABDEM V1-2: **aceita**, com evidência real de terreno nu

FABDEM = Copernicus DEM 30 m com florestas e edificações removidas (Hawker et al. 2022,
Universidade de Bristol). Acesso público, **sem login e sem token**: `data.bris.ac.uk`, dataset
`s5hqmjcdj8yo2ibzi9b4ew3sn`, bloco `S30W050-S20W040_FABDEM_V1-2.zip` (1,05 GB).

O bloco só é publicado como zip de 10°×10°. Para não baixar 1,05 GB por um tile, o servidor
aceita range request (`Accept-Ranges: bytes`) e o acesso foi feito por
`/vsizip//vsicurl/…/S23W044_FABDEM_V1-2.tif`, lendo só a janela da bbox — mesmos ~2,6 MB.
Licença CC BY-NC-SA 4.0 (uso não comercial), compatível com trabalho acadêmico; registrar a
citação em qualquer publicação.

### A verificação de que é terreno nu de fato (não suposição)

Diferença FABDEM − GLO-30 (mesmo grid, mesma origem, 633.444 pixels): média **−5,30 m**,
mediana −5,18 m, **93,07 %** dos pixels ≤ 0. Já é sinal na direção certa, mas o teste decisivo é
estratificar por uso do solo. MapBiomas Coleção 9, `classification_2022`, extraído via Earth
Engine na mesma bbox e reamostrado para a mesma grade:

| Classe MapBiomas 2022 | n pixels | Diferença média | Mediana | % ≤ 0 |
|---|---:|---:|---:|---:|
| Formação florestal | 361.115 | **−7,64 m** | −7,73 m | 99,4 % |
| Silvicultura | 326 | **−9,77 m** | −10,16 m | 100,0 % |
| Mosaico agricultura/pastagem | 121.434 | −3,07 m | −2,87 m | 94,0 % |
| Afloramento rochoso | 26.747 | −3,09 m | −1,58 m | 88,5 % |
| Área urbanizada | 37.066 | −1,96 m | −0,83 m | 79,1 % |
| Outra área não vegetada | 463 | −1,47 m | −0,74 m | 76,7 % |
| Pastagem | 85.956 | **−0,79 m** | −0,26 m | 72,8 % |
| Água | 287 | **+0,31 m** | 0,00 m | 56,8 % |

O rebaixamento é **onde tem dossel** (floresta −7,6 m; silvicultura −9,8 m) e **quase nulo onde
não tem** (pastagem −0,8 m; água +0,3 m). É exatamente o que separa MDT de MDS, medido, não
assumido. Área urbanizada cai só −1,96 m porque a célula de 30 m mistura telhado e rua — remoção
de edificação a 30 m é fraca por construção, e isso é limitação do produto, não erro.

Rugosidade local (desvio-padrão 3×3) caiu apenas 0,6 % (11,367 m → 11,302 m): em relevo de
serra, a rugosidade é dominada pela encosta, não pelo dossel. O critério "mais liso" **não**
discrimina aqui — reportado como tal em vez de forçado.

## 4. Erro real encontrado e corrigido: CRS de Petrópolis

Na rodada anterior o MDE de Petrópolis foi rotulado **EPSG:31984** (SIRGAS 2000 / UTM 24S). Está
errado: o `prj.adf` da fonte declara meridiano central −45°, que é a **zona 23S**, EPSG:31983. O
erro apareceu sozinho: a primeira comparação com o GLO-30 deu **0 pixels em comum**. Reprojetando
o centro da grade, EPSG:31984 põe Petrópolis em `lon −37,18` (mar de Sergipe), EPSG:31983 põe em
`lon −43,18` — a posição real.

O cálculo D-infinity em si não foi afetado (usa só a geometria métrica da grade, que estava
correta), mas o rótulo errado inviabilizaria qualquer amostragem em ponto. Os rasters de
Petrópolis foram regerados com EPSG:31983. Curitiba (EPSG:31982, meridiano central −51° = zona
22S) foi conferida e está correta.

## 5. Rasters gerados

`compute_hand_twi_dinfinity.py` (o mesmo script já validado bit a bit contra Recife, não
reescrito) rodado sobre o FABDEM:

**`local_runs/susc_20g_hand_twi_dinfinity_generico/petropolis_mdt_real/`** (git-ignored)
— `hand_dinf.tif`, `twi_dinf.tif`, `slope_deg_wbt.tif` + intermediários.

| | Petrópolis (FABDEM, MDT) |
|---|---|
| CRS / grade / pixel | EPSG:31983 / 932×698 / 30 m |
| Células válidas | 633.444 |
| Limiar de drenagem (P98) | 874,45 células |
| HAND (mín / mediana / máx) | 0,00 / 109,31 / 1066,26 m |
| TWI (mín / mediana / máx) | 2,43 / 5,58 / 41,55 |
| Declividade (mediana) | 23,57° |

### Ressalvas que continuam de pé

1. **Resolução**: FABDEM é 30 m (1 arco-segundo). Recife e Curitiba estão em 10 m. A paridade de
   *tipo de superfície* foi resolvida; a de *resolução* não, e não há como resolvê-la com FABDEM.
   Existe um grid local de 10 m (`Hipsometria`), mas é superfície — escolher um implica abrir mão
   do outro. Não foi feita nenhuma combinação dos dois: subtrair um do outro para "fabricar" um
   MDT de 10 m seria dado inventado.
2. **Recorte da bbox trunca a drenagem a montante.** HAND e acumulação foram calculados só dentro
   de `-43.25,-22.55,-43.05,-22.30`, conforme pedido. Bacia que entra pela borda é cortada, o que
   tende a **superestimar** HAND perto das bordas. Valparaíso está a ~2 km da borda oeste do
   recorte — não é caso de borda, mas o efeito não é zero.
3. **Limiar P98 é sensível à grade**: 874,45 células a 30 m não é o mesmo alvo hidrológico que
   1122,74 células a 10 m em Recife. HAND de Petrópolis segue não sendo diretamente comparável em
   valor absoluto ao de Recife.

---

## 6. Etapa 2 — Leitura observacional nos 2 pontos adjudicados

Checagem de coerência física. Não é feature, não é rótulo, não há modelo, não há ponto negativo
para comparar. Registro em `registries/v20g_observational_reading_adjudicated_points.csv`.

| Ponto | HAND (m) | TWI | Declividade (°) | Percentil regional HAND / TWI | Distância à drenagem |
|---|---:|---:|---:|---:|---:|
| **Curitiba / Juvevê** (−25,4177; −49,2557) MDT 10 m | **2,52** | **7,18** | **1,98** | 24,3 % / 63,1 % | 135 m |
| **Petrópolis / Valparaíso** (−22,51625; −43,18828) MDT FABDEM 30 m | **50,88** | **5,11** | **23,34** | 27,8 % / 31,9 % | 255 m |

**Curitiba / Juvevê** — coerente com ponto de enchente real: 2,5 m acima do nível de drenagem,
terreno praticamente plano (1,98°), TWI acima da mediana regional (7,18 vs 6,53), com pixel de
HAND = 0 a menos de 150 m.

**Petrópolis / Valparaíso** — **não** reproduz a assinatura clássica de inundação: 50,9 m acima
da drenagem, encosta de 23,3° (exatamente a mediana regional de 23,57°) e TWI abaixo da mediana
regional; é "baixo" só em termos relativos ao relevo de serra (percentil 27,8 % de HAND), o que é
compatível com enxurrada/escoamento em encosta, não com lâmina d'água acumulada — mas a leitura é
frágil a 30 m, com drenagem recortada pela bbox e fundo de vale estreito, e por isso é sinal de
alerta para reexame do candidato, não veredito sobre ele.

Nenhuma das duas leituras valida modelo algum: não há modelo, não há negativo, e N=1 por região.

## 7. Scripts adicionados neste passo

| Arquivo | Papel |
|---|---|
| `scripts/fetch_open_dem_bbox.py` | lê só a janela da bbox de um DEM aberto em COG via `/vsicurl` (ou `/vsizip//vsicurl`), sem credencial |
| `scripts/read_hand_twi_slope_at_point.py` | leitura observacional de HAND/TWI/declividade no pixel de um lat/lon |

`compute_hand_twi_dinfinity.py`, `prepare_region_dtm.py` e `compare_rasters.py` não foram
alterados. Suíte `tests/test_susc_20g_hand_twi_dinfinity.py`: **12 passaram, 0 falharam**.
