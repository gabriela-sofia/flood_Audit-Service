# TASK 2 — Investigação do discrepância elevation_m: CRS bug (a) vs confundimento de amostragem (b)

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-23
**Autorizado por**: Gabriela Sofia (reviewer/executora única do projeto)

## 1. Pergunta

v8 afirma que `elevation_m` "mantém o problema de sinal/CRS já documentado" — mas v7's
Improvement 4 (pareamento por bairro) tinha relatado uma redução do gap de elevação de ~11,3 m
(significativo) para ~2,6 m (não significativo). São a mesma coisa, ou coisas diferentes?

## 2. Achado #1 — as duas afirmações não são sobre o mesmo par de grupos

Improvement 4 (v7) aplicou pareamento por bairro à **pseudo-absência sintética SECUNDÁRIA**
(pontos artificiais gerados por rejection sampling dentro do mesmo bairro do positivo pareado),
comparada aos positivos: gap caiu de 11,3 m → 2,6 m, p deixou de ser significativo. **Isso nunca
foi aplicado ao conjunto PRIMÁRIO real-vs-real** (positivos reais vs. negativos reais do
SEDEC/EMLURB) — os 36 negativos reais do v7 primário nunca passaram por nenhum pareamento
espacial; eles são localizações reais de ocorrências não-meteorológicas (queda de energia,
iluminação), fixas pela fonte.

Confirmado diretamente nos coeficientes Firth do conjunto primário:

| | v7 primário (36 neg reais) | v8 primário (116 neg reais) |
|---|---:|---:|
| `elevation_m` coef (padronizado) | +0,4953 | +0,5906 |
| `elevation_m` p-valor | 0,253 (não sig.) | **0,047 (sig., sinal errado)** |
| `elevation_m` CI cruza zero | Sim | **Não** |

**O sinal errado (positivo, esperado negativo) já estava presente em v7** — não é algo que
"quebrou de novo" em v8; é uma questão pré-existente que a v8 apenas tornou mais visível
(estatisticamente mais "confiante", na direção errada) ao adicionar mais negativos.

## 3. Verificação direta — é (a) bug de CRS/reprojeção?

`v2.extract_elevation_slope_at_points()` (função canônica usada por **todos** os builders de
features desde v2 até v9, incluindo `build_v8_new_negatives_base.py` dos 80 novos pontos
EMLURB — **mesmo código, mesmo caminho**, não um caminho novo/não testado) foi testada
diretamente:

- CRS dos 48 tiles PE3D: **EPSG:31985** (SIRGAS 2000 / UTM 25S) — CRS correto e válido para o
  Recife (longitude ~-34,9°, dentro da faixa -36° a -30° da zona 25S).
- Transformação `pyproj` de 3 landmarks conhecidos do Recife caiu dentro do envelope espacial
  dos 48 tiles, como esperado (ex.: Recife Centro em 293802E/9108254N, dentro do envelope
  276017-296878E / 9096713-9124483N).
- **Amostragem real de elevação em 3 pontos de referência conhecidos**:

| Local | Elevação extraída |
|---|---:|
| Recife Centro (baixada) | 3,34 m |
| Boa Viagem / praia (baixada) | 5,06 m |
| Alto José do Pinho (bairro de morro, nome literal "Alto") | **40,31 m**, slope 17,0° |

Os valores são **fisicamente corretos em sinal e magnitude** (baixadas ~3-5m, bairro de morro
~40m com slope alto). **(a) está refutada**: não há bug de CRS/reprojeção — o pipeline de
extração está correto e é idêntico para todos os pontos (originais e novos).

## 4. Verificação direta — é (b) confundimento de amostragem reintroduzido?

Comparação de geografia (bairro) entre positivos, os 36 negativos originais (v7) e os 80 novos
negativos EMLURB (v8, `build_v8_new_negatives_base.py` — amostragem determinística 8/ano/2015-
2024, **sem nenhum critério de bairro**):

| Grupo | n bairros distintos | Overlap com bairros-positivo (29 bairros) | Elevação média | MWU vs. positivos |
|---|---:|---:|---:|---:|
| Positivos (145) | 29 | — | 27,29 m | — |
| Negativos originais v7 (36) | 27 | 14/27 (52%) | 21,35 m | p=0,196 (não sig.) |
| **Negativos novos v8 (80, EMLURB, sem pareamento)** | **39** | **3/39 (7,7%)** | 19,10 m | **p=0,027 (sig.)** |
| Negativos originais vs. novos (entre si) | — | — | (21,3 vs 19,1) | p=0,662 (não sig. entre si) |

**Confirmado: (b) é a causa real.** Os 80 novos negativos foram amostrados de forma
determinística por ano, sem nenhum critério de sobreposição geográfica com os bairros onde
ocorrem os eventos de alagamento positivos — dispersos em 39 bairros com apenas 3 em comum com
a geografia dos positivos (vs. 14/27 já imperfeito dos negativos originais). Isso não é um bug
de extração; é um viés de seleção reintroduzido pela nova fonte de dados, que **amplificou** um
problema de desenho amostral que já existia de forma mais branda desde v7.

## 5. Correção aplicada (dado real, sem fabricação)

Não é possível "mover" negativos reais para outro bairro (isso seria fabricação). A correção
legítima é **reseleção dentro do mesmo pool real já obtido**: do pool bruto de 200 candidatos
reais do arquivo EMLURB "156" (`_scratch_emlurb156_sampled_negatives_raw.csv`, coordenadas e
datas reais da fonte, já baixado em v8), **88/200 candidatos** caem em bairros que também têm
eventos positivos reais — mais que os 80 originalmente escolhidos sem esse critério.
`build_v9_bairro_matched_new_negatives.py` reseleciona exatamente esses 88 candidatos (mesma
lógica de 8/ano, mesmo processamento de features), substituindo os 80 antigos.

**Resultado da correção**:

| | Negativos novos v8 (80, sem critério) | **Negativos novos v9 (88, bairro-sobreposto)** |
|---|---:|---:|
| Bairros distintos | 39 | 21 |
| Overlap com bairros-positivo | 3/39 (7,7%) | **21/21 (100%, por construção)** |
| Elevação média | 19,10 m | **25,84 m** (muito mais perto dos 27,29 m dos positivos) |
| MWU vs. positivos | p=0,027 (sig.) | **p=0,772 (não sig.)** |
| MWU conjunto completo (36 antigos + N novos) vs. positivos | p=0,047 (Firth, v8) | **ver §6** |

## 6. Coeficiente corrigido (Firth multivariado, dataset v9 completo, n=260)

| | v8 (n=251, 116 neg) | **v9 (n=260, 124 neg, 88 bairro-matched)** |
|---|---:|---:|
| `elevation_m` coef padronizado | +0,5906 | **+0,2678** |
| p-valor | 0,047 (sig., sinal errado) | **0,372 (não sig.)** |
| CI cruza zero | Não | **Sim** |
| Bootstrap sign-flip (N=1000) | 2,7% (falsamente "estável" no sinal errado) | **18,0%** (instável — mais honesto) |

**A correção não torna o sinal "certo"** (o coeficiente ainda é positivo, esperado negativo) —
mas remove a falsa confiança estatística que o confundimento de amostragem estava produzindo.
Em vez de um coeficiente "significativo e errado" (pior cenário: parece informativo mas não é),
agora temos um coeficiente honestamente instável e não-significativo, consistente com o que se
espera de uma feature que carrega um confundimento residual genuíno (elevação de bairro difere
entre "onde há eventos de alagamento reportados" e "onde há problemas de infraestrutura elétrica
reportados" por razões geográficas reais que não são puramente um artefato de amostragem, mas
também não constituem evidência de sinal físico direto).

## 7. Veredito

- **Não é (a)** — pipeline de CRS/extração validado como correto, mesmo caminho de código para
  todos os pontos.
- **É (b)**, especificamente: os 80 negativos novos do v8 nunca passaram por nenhum critério de
  correspondência geográfica com os positivos (diferente do Improvement 4 do v7, que só se
  aplicou à pseudo-absência sintética secundária, nunca ao conjunto primário real-vs-real).
- **Corrigido** via reseleção (não fabricação) dentro do pool real já coletado, priorizando
  candidatos em bairros com sobreposição de eventos positivos: 88/200 candidatos elegíveis
  (mais que os 80 originais).
- `elevation_m` no dataset v9 final: coef=+0,2678, p=0,372, sign-flip bootstrap 18,0% — ainda
  sinal fisicamente incorreto na média, mas agora honestamente não-significativo e não
  falsamente "confiante", em vez de significativo-e-errado.

## 8. Arquivos

`build_v9_bairro_matched_new_negatives.py`, `fetch_rain_v9_bairro_matched.py`,
`new_negatives_v9_bairro_matched_final.csv` (88 linhas), `dataset_v9_final.csv`,
`primaria_v9_firth_multivariate_coefs.csv`, `primaria_v9_bootstrap_coefs.csv`.
