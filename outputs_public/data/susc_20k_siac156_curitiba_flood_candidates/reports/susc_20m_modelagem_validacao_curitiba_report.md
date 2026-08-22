# SUSC-20M — Modelagem e validação estatística, Curitiba (SIAC 156)

**Status**: RESULTADO REAL, FRACO. O modelo separa positivo de negativo acima do acaso
(LOO-AUC 0,605), mas **o único preditor estável é chuva antecedente**; as três features de
terreno têm intervalo de confiança cruzando zero. O resultado é reportado como veio.

Espelha `pipeline_v12_primary.py` (SUSC-20C): Mann-Whitney univariado → Firth multivariada →
bootstrap estratificado N=1000 → AUC preditivo (LOO + 5-fold repetido 50×). Mesmas funções,
mesmos parâmetros. Seed 20260731.

---

## 1. QA de ambiente — obrigatório antes de comparar com Recife

`firthlogist` exige Python <3.11 e a API privada `_validate_data` do scikit-learn, removida
na 1.6. Este ambiente é Python 3.12 / sklearn 1.8 / numpy 2.5. Solução: instalação com
`--ignore-requires-python` (o wheel é `py3-none-any`) + shim de 3 linhas restaurando
`_validate_data`.

Isso só é aceitável se produzir o mesmo número. Verificado: o pipeline re-roda a Firth
multivariada sobre o `dataset_eventos_features_v12_final.csv` de Recife e compara com os
coeficientes publicados em `primaria_v12_firth_multivariate_coefs.csv`.

> **status = REPRODUZIDO**, diferença máxima absoluta **0,0** em coeficiente, p-valor,
> IC inferior e IC superior, nas 6 features.

O QA roda a cada execução e aborta o pipeline se divergir.

## 2. Duas decisões tomadas antes de rodar

### 2.1 Unidade de análise = (lat, lon, `event_date`)

Os registries repetem a mesma ocorrência quando o SIAC 156 a categorizou duas vezes. Essas
linhas têm feature idêntica bit a bit; contá-las como observações independentes é
pseudo-replicação. **1357 linhas → 1148 unidades (1045 pos / 103 neg).**

### 2.2 Gate EPV checado ANTES do multivariado, com a contagem corrigida

| base | n negativo | EPV 6 feat. | EPV 5 feat. |
|---|---:|---:|---:|
| linhas | 119 | 19,83 | 23,80 |
| **unidades** | **103** | **17,17 ✗** | **20,60 ✓** |

Piso 20 (`revp_revisao_literatura_alinhamento_metodos_v1.md`). O primário roda com **5
features**; o bloco primário levanta `SystemExit` se o gate reprovar (testado).

Feature cortada: **`elevation_m`**, por razão física declarada antes de ver o resultado —
Curitiba é planalto (todos os pontos entre 850 e 1100 m), onde a elevação absoluta carrega
identidade de sub-bacia e não exposição local; HAND é a versão causalmente correta da mesma
ideia. Corroborado pelo v12 de Recife, onde `elevation_m` saiu com **sinal invertido**
(+0,2662, p=0,374), problema já diagnosticado em `task2_elevacao_crs_investigacao_v9.md`.
A seção 5 mostra o que essa decisão custou — inclusive contra ela.

Sinais esperados: revisados um a um, **nenhum alterado** em relação a Recife.

## 3. Resultado primário — 5 features, 1148 unidades, EPV 20,60

`n_used = 1138` (1035 pos / 103 neg; 10 unidades positivas perdidas por listwise deletion,
as sem HAND documentadas em SUSC-20L). Log-verossimilhança −321,72.

### Univariado (Mann-Whitney)

| feature | média pos | média neg | p | r rank-biserial | direção | sinal esperado |
|---|---:|---:|---:|---:|---|---:|
| `slope_deg` | 3,482 | 3,902 | 0,3958 | 0,051 | pos<neg | −1 ✓ |
| `hand_m_dinf` | 8,423 | 9,372 | 0,0531 | 0,115 | pos<neg | −1 ✓ |
| `twi_dinf` | 8,161 | 7,768 | 0,7490 | −0,019 | pos>neg | +1 ✓ |
| `rain_peak_residual_orthogonalized` | 0,064 | 2,096 | **0,0081** | 0,158 | pos<neg | +1 ✗ |
| `rain_decay_index_api_chirps` | 33,255 | 26,868 | **0,0006** | −0,206 | pos>neg | +1 ✓ |

### Firth multivariada

| feature | coef. padr. | IC 95% | p | sinal esperado | IC cruza zero |
|---|---:|---|---:|---|---|
| `slope_deg` | −0,1239 | [−0,338; 0,112] | 0,2904 | ✓ | sim |
| `hand_m_dinf` | −0,0422 | [−0,257; 0,185] | 0,7037 | ✓ | sim |
| `twi_dinf` | +0,0424 | [−0,216; 0,360] | 0,7671 | ✓ | sim |
| `rain_peak_residual_orthogonalized` | **−0,2771** | [−0,463; −0,081] | **0,0063** | **✗** | não |
| `rain_decay_index_api_chirps` | **+0,4228** | [0,183; 0,681] | **0,0004** | ✓ | não |

### Bootstrap estratificado (N=1000, 0 falhas)

| feature | média boot | IC 2,5–97,5% | cruza zero | % sign-flip |
|---|---:|---|---|---:|
| `slope_deg` | −0,1097 | [−0,345; 0,161] | sim | 19,9 |
| `hand_m_dinf` | −0,0393 | [−0,263; 0,197] | sim | **36,7** |
| `twi_dinf` | +0,0671 | [−0,162; 0,354] | sim | **35,0** |
| `rain_peak_residual_orthogonalized` | −0,2743 | [−0,438; −0,088] | não | 0,2 |
| `rain_decay_index_api_chirps` | +0,4334 | [0,176; 0,717] | não | 0,0 |

O bootstrap concorda com a Firth em tudo. Taxa de troca de sinal de 35-37% em `hand_m_dinf`
e `twi_dinf` significa que o sinal dessas features é essencialmente indeterminado nesta
amostra.

### AUC preditivo

| | valor |
|---|---:|
| LOO-CV AUC | **0,6048** |
| 5-fold repetido 50× (média ± dp) | **0,6007 ± 0,0100** |
| faixa 5-fold (mín–máx) | 0,5759 – 0,6193 |

## 4. Leitura — o que esses números permitem e não permitem dizer

**Observado.** Duas features têm efeito estável: chuva antecedente (`rain_decay_index`,
+0,42) e o resíduo de pico de chuva (`rain_peak_residual`, −0,28). As três de terreno têm
sinal na direção física esperada mas IC cruzando zero e sign-flip de até 37%.

**O que isso provavelmente é.** O eixo que o modelo achou é temporal, não espacial. Positivo
é reclamação de alagamento (acontece em dia de chuva); negativo é reclamação
não-hidrológica em data arbitrária. Um modelo que aprende "choveu antes" separa as duas
classes sem dizer nada sobre suscetibilidade do local. Isso é uma propriedade do desenho
positivo-vs-negativo administrativo, **não** um defeito de Curitiba — o v12 de Recife tem o
mesmo padrão, e mais forte: lá `rain_decay_index` = +0,9896 (p<0,0001) e o terreno também
era fraco (HAND −0,0001, `slope` −0,17 n.s.; só `twi` chegou a p=0,046). Curitiba
**replica** o padrão de Recife, com terreno ainda mais mudo.

Some-se a isso o achado de SUSC-20L: a chuva de Curitiba tem apenas **10 células ERA5-Land**
para 848 coordenadas. Dentro de uma mesma data, quase todos os pontos compartilham o mesmo
valor de chuva. A feature de chuva é, por construção, quase uma variável de data.

**Sinal invertido replicado.** `rain_peak_residual_orthogonalized` sai negativo em Recife
(−0,14, n.s.) e negativo e significativo em Curitiba (−0,28, p=0,006). Hipótese não testada:
controlado o antecedente, a reclamação associa-se mais a chuva acumulada/prolongada do que a
pico curto e intenso — compatível com saturação de drenagem urbana em vez de enxurrada. Não
há evidência local que sustente isso; fica como hipótese, não como achado.

**O que NÃO se pode dizer.** Que o modelo prevê alagamento; que HAND/TWI/declividade "não
importam" (o resultado é indeterminação, não ausência de efeito); que AUC 0,60 valida o
índice para uso operacional. AUC 0,60 com um preditor temporal dominante é um resultado
fraco e assim reportado — nenhum método foi trocado no meio por causa dele.

## 5. Sensibilidades — rotuladas, não-primárias

### S1 — as 6 features de Recife (EPV 17,17, **abaixo do piso**)

Rodada só para medir o custo da decisão da seção 2.2. **Não é resultado válido** pelo
critério de EPV do projeto.

| feature | coef. | IC 95% | p | sinal |
|---|---:|---|---:|---|
| `elevation_m` | **−0,3684** | [−0,617; −0,111] | **0,0054** | ✓ |
| `rain_decay_index_api_chirps` | +0,4007 | [0,160; 0,659] | 0,0008 | ✓ |
| `rain_peak_residual_orthogonalized` | −0,2682 | [−0,455; −0,071] | 0,0084 | ✗ |
| `hand_m_dinf` | +0,1563 | [−0,102; 0,425] | 0,2375 | ✗ |
| `slope_deg` | −0,0544 | [−0,278; 0,194] | 0,6480 | ✓ |
| `twi_dinf` | +0,0021 | [−0,256; 0,321] | 0,9411 | ✓ |

LOO-AUC 0,6249 (vs. 0,6048 do primário); 5-fold 0,6220 ± 0,0093.

**Isto contraria parcialmente a justificativa do corte, e fica registrado como tal**: em
Curitiba, ao contrário de Recife, `elevation_m` sai com o sinal fisicamente esperado
(negativo), IC não cruza zero, sign-flip de 0,2% no bootstrap, e melhora a AUC em ~0,02.
A decisão de cortar foi tomada *antes* de ver isso, por critério físico + gate EPV, e não
está sendo revertida a posteriori. O caminho correto para readmitir `elevation_m` é a
seção 6 — não relaxar o piso.

### S2 — 1357 linhas com pseudo-replicação (EPV 23,80)

Rodada só para medir o efeito da unidade de análise. `hand_m_dinf` passa a ser
significativo no univariado (p=0,0092, contra 0,0531 nas unidades) mas continua n.s. no
multivariado (p=0,347). LOO-AUC 0,6154. Confirma o esperado: repetir a mesma observação
infla significância aparente sem trazer informação — a escolha da seção 2.1 está certa.

## 6. Próxima rodada — número concreto

Para readmitir `elevation_m` com EPV ≥ 20 são necessárias **≥ 120 unidades de observação
negativas** (hoje 103): faltam no mínimo **17 unidades**, e o prudente é mirar ~150 para
absorver perda por listwise deletion e por colisão de geocodificação.

Caminho já existente, sem código novo:
`scripts/mine_negative_candidates_siac156.py --max-por-ano <maior>` + `geocode_nominatim.py`.
O script já pareia por bairro corretamente. Atenção ao contar o incremento: o teto útil é em
**unidades (lat, lon, data)**, não em linhas — pela taxa atual, 119 linhas renderam 103
unidades (~87%).

## 7. Limitações

1. **Desbalanceamento 10:1** (1035 pos / 103 neg). A Firth trata separação e viés de amostra
   pequena; a AUC usa `class_weight="balanced"`. Ainda assim, toda a precisão do modelo está
   ancorada em 103 negativos.
2. **Confundimento temporal** (seção 4) — limitação de desenho, herdada de Recife.
3. **Chuva com resolução de 10 células** para a cidade inteira (SUSC-20L §3).
4. **`mapbiomas_class_2023` não entrou no modelo** — 97,1% dos pontos em classe 24, poder
   discriminativo ~0 por construção. Mesma decisão do v12 de Recife, que também não a modela.
5. **10 unidades positivas sem HAND** removidas por listwise deletion (tratamento confirmado
   como idêntico ao de Recife).
6. **Nenhuma validação externa**: nada foi testado fora de Curitiba, nem em holdout temporal.
   LOO e k-fold repetido são validação interna.
7. **Positivos e negativos são pontos administrativos geocodificados por rua**, não verdade
   cartográfica de campo.

## 8. Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20m_curitiba_primary.py
python -m pytest tests/test_susc_20m_curitiba_modelagem.py -q
```

Dependência não trivial: `pip install --ignore-requires-python firthlogist` (ver seção 1).
Resultados em `outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/results/`.
