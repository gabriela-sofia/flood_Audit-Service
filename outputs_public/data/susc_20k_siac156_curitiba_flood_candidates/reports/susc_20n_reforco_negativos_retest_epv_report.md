# SUSC-20N — Reforço de negativos e retest do gate EPV, Curitiba

**Status**: EXECUTADO. Negativos de 103 → 426 unidades, EPV com 6 features de 17,17 → **70,5**
(passa o piso). Escopo restrito: só aumento de N e re-rodada. Nenhuma feature nova, nenhum
`EXPECTED_SIGN` alterado, nenhum método trocado.

**Revisão humana (pós-execução, 2026-08-01)**: apesar da EPV passar com 6 features,
`elevation_m` **não** foi promovida a rota primária — o corte original era causal (confundimento
com identidade de sub-bacia/bairro no planalto de Curitiba), não estatístico, e a EPV nunca foi
o motivo de fundo, só a confirmação formal. A sensibilidade S1 (6 features) mostra contribuição
preditiva nula de `elevation_m` (ΔAUC −0,0006) mesmo com sinal "correto" — o que não refuta o
confundimento, só mostra que reintroduzi-la não piora nem melhora a predição. **Rota primária
permanece 5 features** (agora com EPV 84,6, também folgada); 6 features segue como sensibilidade
S1. Ver docstring de `pipeline_v20m_curitiba_primary.py` para o raciocínio completo. Números
abaixo mantidos como calculados; a leitura textual foi ajustada para não chamar 6-features de
"primário".

---

## 1. Bloqueio encontrado e resolvido

Os 4 CSVs brutos do SIAC 156 usados no SUSC-20K2/K3 **não estavam mais no disco** e não havia
script de download versionado. Rebaixados do espelho UFPR (`dadosabertos.c3sl.ufpr.br/curitiba/156/`,
517,6 MB, autorizado) para `local_runs/susc_20n_siac156_negative_expansion/raw/` (git-ignored).

## 2. Achado de reprodutibilidade — a amostragem determinística não é reproduzível

`stable_rank()` em `mine_negative_candidates_siac156.py:52` inclui **`str(csv_path)`** no hash:

```python
"rank_key": stable_rank(str(csv_path), logradouro, bairro, data_criacao)
```

Como a ordenação define quem entra no corte `--max-por-ano`, **a amostra depende do caminho
passado na linha de comando**. Rodando com os mesmos 4 arquivos e `--max-por-ano 40`, o
resultado bate em contagem (160, 40/ano) mas **as 160 linhas são outras**: 0 de sobreposição com
`v20k3_negativos_brutos.csv`. O caminho da sessão original não é recuperável (tentativa de
inverter o hash sobre 15 diretórios × 5 nomes plausíveis: nenhum bate).

**Consequência**: `--max-por-ano 120` não é superconjunto de `--max-por-ano 40`. Esta rodada é
uma **união de dois sorteios** do mesmo pool, sob os mesmos critérios, não um incremento.
Todas as 119 linhas anteriores foram preservadas verbatim (mesmo `point_id`).

Não corrigi `stable_rank` — mudá-lo mudaria a amostra e está fora do escopo desta rodada. Para
a próxima, esta rodada passou **nomes de arquivo nus** (executando de dentro do diretório dos
CSVs), o que torna o `rank_key` estável entre máquinas. Reproduzir exige repetir isso.

## 3. Números

### Funil de negativos

| etapa | rodada anterior (20K3) | esta rodada (20N) | total |
|---|---:|---:|---:|
| `--max-por-ano` | 40 | **120** | — |
| brutos minerados | 160 | 480 | — |
| geocodificados | 160 (137 strong / 23 medium) | **480 (430 strong / 50 medium), 0 falhas** | — |
| descartados: duplicata de chave | — | 66 | — |
| descartados: colisão <30 m com positivo | 41 | **91** | — |
| linhas finais | 119 | +323 novas | **442** |
| **unidades (lat, lon, data)** | **103** | +323 | **426** |
| unidades pós-listwise deletion | 103 | — | **423** |

Positivos inalterados: 1238 linhas / 1045 unidades. Balanço de classes 10,4:1 → **2,4:1**.

### Gate EPV (6 features, classe minoritária = negativo)

| | antes (20M) | depois (20N) |
|---|---:|---:|
| n negativo pós-listwise | 103 | **423** |
| EPV com 6 features | 17,17 ✗ | **70,5 ✓** |
| EPV com 5 features | 20,60 ✓ | 84,6 ✓ |

Piso 20. **Passa com folga com 6 features** — mas ver "Revisão humana" acima: isso não reabre a
decisão causal de cortar `elevation_m`. Números de ambas as rotas reportados abaixo; a rota
primária continua sendo 5 features.

### Primário — 5 features, 1471 unidades, n_used 1458 (1035 pos / 423 neg)

**LOO-AUC 0,6459** · 5-fold repetido 50× **0,6440 ± ~0,003**.

| feature | coef. 20N (primário) | IC 95% | p (20N) | sinal | coef. 20M | p (20M) |
|---|---:|---|---:|---|---:|---:|
| `rain_decay_index_api_chirps` | **+0,4278** | [0,294; 0,567] | **<0,0001** | ✓ | +0,4228 | 0,0004 |
| `rain_peak_residual_orthogonalized` | **−0,3276** | [−0,444; −0,213] | **<0,0001** | **✗** | −0,2771 | 0,0063 |
| `hand_m_dinf` | **−0,1401** | [−0,267; −0,012] | **0,0317** | ✓ | −0,0422 | 0,7037 |
| `twi_dinf` | +0,0761 | [−0,073; 0,239] | 0,3267 | ✓ | +0,0424 | 0,7671 |
| `slope_deg` | −0,0026 | [−0,139; 0,137] | 0,9639 | ✓ | −0,1239 | 0,2904 |

**Achado novo desta rodada, não presente no relatório original**: com N maior e `elevation_m`
de fora (rota primária correta), `hand_m_dinf` passa a ser estatisticamente significativo e com
sinal fisicamente esperado (IC não cruza zero) — no 20M (n_neg=103) era nulo (p=0,70). É a
primeira feature de terreno com sinal robusto no modelo primário de Curitiba.

### Sensibilidade S1 — 6 features (elevation_m reintroduzida), mesmos 1458 casos

**LOO-AUC 0,6453** · 5-fold repetido 50× **0,6438 ± 0,0028** (faixa 0,6362–0,6503).
Log-verossimilhança −816,70.

| feature | coef. 20N (S1) | IC 95% | p (20N) | sinal | sign-flip boot | coef. 20M | p (20M) |
|---|---:|---|---:|---|---:|---:|---:|
| `rain_decay_index_api_chirps` | **+0,4213** | [0,287; 0,561] | **<0,0001** | ✓ | 0,0% | +0,4228 | 0,0004 |
| `rain_peak_residual_orthogonalized` | **−0,3252** | [−0,442; −0,210] | **<0,0001** | **✗** | 0,0% | −0,2771 | 0,0063 |
| `elevation_m` | **−0,1686** | [−0,325; −0,011] | **0,0361** | ✓ | 2,1% | *(fora)* | — |
| `twi_dinf` | +0,0617 | [−0,088; 0,225] | 0,4303 | ✓ | 17,6% | +0,0424 | 0,7671 |
| `hand_m_dinf` | −0,0469 | [−0,201; 0,108] | 0,5513 | ✓ | 27,6% | −0,0422 | 0,7037 |
| `slope_deg` | +0,0261 | [−0,113; 0,169] | 0,7151 | **✗** | 36,7% | −0,1239 | 0,2904 |

Note-se que `hand_m_dinf` some (deixa de ser significativo) quando `elevation_m` entra junto —
consistente com o argumento causal do docstring: as duas carregam parte da mesma informação
(altura relativa/absoluta), e `elevation_m` "rouba" a variância de `hand_m_dinf` no multivariado
sem ser a versão causalmente correta dela.

Univariado (Mann-Whitney): significativos em p<0,05 → `elevation_m` (p<0,0001),
`hand_m_dinf` (p=0,0001), `rain_peak_residual` (p<0,0001), `rain_decay_index` (p<0,0001).
Não significativos → `slope_deg` (p=0,0996), `twi_dinf` (p=0,3948).

### O que mudou em relação ao SUSC-20M

1. **`hand_m_dinf` vira significativo no primário** (−0,1401, p=0,0317, IC fora de zero) —
   era nulo no 20M (p=0,70, n_neg=103). Primeira feature de terreno com sinal robusto no modelo
   primário de Curitiba. Só aparece quando `elevation_m` fica de fora (ver nota da S1 acima —
   as duas competem pela mesma variância).
2. **`elevation_m`, reintroduzida só na sensibilidade S1, entra estável**: −0,1686, IC não cruza
   zero, sign-flip 2,1%, sinal fisicamente esperado — mas contribuição preditiva nula (ver
   resumo abaixo). Não muda a rota primária (ver "Revisão humana" no topo).
3. **`slope_deg` troca de sinal** entre 20M e 20N-primário (−0,1239 → −0,0026), segue não
   significativo nas duas rodadas — indeterminado, não é achado.
4. **`rain_peak_residual` mantém o sinal invertido** e ganha força (p 0,0063 → <0,0001 no
   primário). Terceira replicação do mesmo padrão (Recife v12, Curitiba 20M, Curitiba 20N).
5. **AUC sobe 0,6048 → 0,6459** no primário, e o desvio do 5-fold cai (mais N, estimativa mais
   estável). Continua um resultado fraco.

**A leitura de fundo do SUSC-20M não muda**: o eixo dominante segue sendo chuva antecedente,
que é temporal, não espacial. Mais negativos deram poder estatístico, não removeram o
confundimento de desenho.

### Resumo — primário vs. sensibilidades (todos acima do piso EPV agora)

| bloco | EPV | LOO-AUC |
|---|---:|---:|
| **primário — 5 features (sem `elevation_m`), unidades** | **84,60** | **0,6459** |
| S1 — 6 features (`elevation_m` reintroduzida), unidades | 70,50 | 0,6453 |
| S2 — 5 features, 1663 linhas com pseudo-replicação | 87,80 | 0,6551 |

S1 mostra que a contribuição preditiva de `elevation_m` é **praticamente nula** (ΔAUC −0,0006
em relação ao primário): ela é estatisticamente estável no coeficiente, mas não melhora
predição — não há ganho que justifique reabrir a decisão causal de cortá-la. S2 confirma de novo
que pseudo-replicação infla a AUC aparente (+0,0092 em relação ao primário).

## 4. Bug encontrado e corrigido no pipeline de features

O cache de série de chuva por célula (`rain_cell_series.json`, SUSC-20L) guardava a série mas
não o span coberto. Os negativos novos incluem datas de janeiro/2023, cuja janela de 14 dias
alcança dezembro/2022 — fora do span original (2023-01-02). Primeira execução saiu com **3
janelas truncadas** (`n_days_found` = 1, 10 e 11) em silêncio. Corrigido com checagem de
cobertura + refetch por célula; execução final tem **1680/1680 com 14/14 dias reais**.

QA de não-regressão: os 1357 pontos da rodada anterior mantêm valor bruto **idêntico**
(`max|diff| = 0` em elevação, declividade, HAND, TWI, chuva e MapBiomas). O único valor que
muda é `rain_peak_residual_orthogonalized`, por construção — a ortogonalização é refitada na
amostra maior (β 0,3582 → 0,3725, intercepto 9,3202 → 9,3971).

HAND `NODATA_NO_PIXEL`: 14 → **17 linhas** (3 negativos novos na mesma franja norte-oeste).

## 5. Limitações

1. **União de sorteios, não incremento** (seção 2). A amostra final é reprodutível a partir dos
   artefatos publicados, mas não a partir do comando original do 20K3.
2. **Confundimento temporal intacto** — negativos continuam pareados por bairro, não por data.
   É a limitação de fundo e nenhum aumento de N a resolve.
3. **Chuva ainda com 10 células ERA5-Land** para 1152 coordenadas (SUSC-20L §3).
4. **Colisão subiu de 25,6% para 19,0%** do lote (91/480) — mas em termos absolutos são 91
   pontos descartados por estarem a menos de 30 m de um positivo, o que é sinal de que o pool
   negativo e o positivo compartilham geografia de rua.
5. Nenhuma validação externa nem holdout temporal — segue tudo validação interna.
6. **`elevation_m` fora do primário permanece decisão causal, não estatística** — se uma
   próxima rodada quiser reabri-la, precisa de argumento causal novo (ex.: controlar por
   bairro/renda diretamente), não só mais N passando a EPV.

## 6. Reprodução

```
python outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts/pipeline_v20m_curitiba_primary.py --dataset ../registries/v20n_dataset_curitiba_features_v2.csv --primario 5feat --tag v20n
```

(rota primária correta; use `--primario 6feat` só para reproduzir a sensibilidade S1)

A mineração exige os 4 CSVs brutos em `local_runs/susc_20n_siac156_negative_expansion/raw/` e
tem que ser executada **de dentro desse diretório, com nomes de arquivo nus** (seção 2).
