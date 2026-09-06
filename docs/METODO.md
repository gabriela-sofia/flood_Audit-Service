# Metodo

Este documento descreve a cadeia publica do projeto. Ele nao substitui os scripts:
cada numero citado precisa estar em artefato versionado em `modelo/execucoes/`.

## 1. Premissa

O projeto mede suscetibilidade fisico-hidrologica. A variavel de interesse nao e dano,
vulnerabilidade social, exposicao populacional, risco individual ou previsao operacional.
A resposta esperada e uma estimativa auditavel de predisposicao fisica do terreno sob
condicoes representadas pelas covariaveis disponiveis.

O contrato so responde quando a combinacao entre regiao, modelo, variaveis e distancia
de dominio e compativel com o que foi treinado ou explicitamente aceito como
transferencia. Fora disso, a saida correta e recusa ou alerta de extrapolacao.

## 2. Cadeia publica

1. Curadoria de fontes observacionais e negativas.
2. Harmonizacao espacial e derivacao de variaveis fisicas de terreno.
3. Consolidacao da tabela unica e dos pools de ajuste.
4. Treino dos candidatos regionais e fisiograficos.
5. Selecao dos modelos servidos.
6. Execucao do contrato de inferencia.
7. Geracao de grades de aplicacao.
8. Auditorias complementares de chuva, temporalidade, recuperacao de fonte e camada social.

## 3. Variaveis principais

| Variavel | Papel |
|---|---|
| `elevation_rel_m` | Elevacao relativa dentro do contexto local. Substitui a leitura absoluta quando o objetivo e transferencia entre regioes. |
| `slope_deg` | Declividade local. |
| `hand_m` | Altura acima da drenagem mais proxima. |
| `twi_dinf` | Indice topografico de umidade com escoamento D-infinity. |
| `rain_decay_index` | Chuva antecedente agregada por decaimento temporal, quando disponivel no recorte. |

## 4. Treino curado

Os scripts publicos de treino ficam em `scripts/treino/`. O diretorio inclui a
consolidacao de dados, verificacoes de chuva, desenho experimental e selecao de
modelos. O nome do diretorio e deliberado: ele contem somente o material que entra na
cadeia curada deste pacote publico.

O manifesto da tabela unica v2 registra:

| Campo | Valor |
|---|---:|
| Observacoes de entrada | 595.865 |
| Observacoes admitidas | 568.265 |
| Pool fluvial | 65.070 |
| Grupos admitidos | 6.873 |
| Grupos no pool fluvial | 2.002 |

As tabelas completas ficam fora do pacote publico por tamanho. O manifesto registra os
nomes, tamanhos e checksums esperados para reproducao.

## 5. Modelos servidos

A etapa `svc-01-modelos` produz o indice de candidatos. O contrato atual usa apenas
modelos marcados como serviveis.

| Modelo | Regiao servida | Variaveis | Estado |
|---|---|---|---|
| `recife_pluvial` | Recife | `elevation_rel_m`, `slope_deg`, `hand_m`, `twi_dinf`, `rain_decay_index` | Servivel, desempenho fora dos criterios fortes |
| `pluvial_urbano_nyc` | NYC | `elevation_rel_m`, `slope_deg`, `hand_m`, `twi_dinf` | Servivel, desempenho fora dos criterios fortes |
| `pluvial_urbano_sao_paulo` | Nao servida | `elevation_rel_m`, `slope_deg`, `hand_m`, `twi_dinf` | Nao servivel |
| `fluvial_planicie` | Curitiba | `elevation_rel_m`, `slope_deg`, `hand_m`, `twi_dinf` | Servivel, coerente com criterios |
| `fluvial_serra` | Petropolis | `hand_m` | Servivel, coerente com criterios |

## 6. Contrato de inferencia

O contrato fica em `scripts/servico/svc02_contrato_inferencia.py`. Ele centraliza a
decisao cientifica e devolve resposta estruturada. As demais camadas devem apresentar a
resposta, nao reinterpretar portoes.

Os portoes publicos sao:

| Portao | Funcao |
|---|---|
| `G1_REGIAO_SUPORTADA` | A regiao precisa estar declarada no contrato. |
| `G2_MODELO_SERVIVEL` | O modelo regional precisa estar marcado como servivel. |
| `G3_MATURIDADE_MINIMA` | A maturidade regional precisa estar explicita. |
| `G4_VARIAVEIS_OBRIGATORIAS` | Todas as variaveis exigidas pelo modelo precisam estar presentes. |
| `G5_DOMINIO_COVARIAVEL` | O caso precisa estar dentro do dominio ou sair marcado como extrapolacao. |

## 7. Grades de aplicacao

A grade publica e gerada por `scripts/servico/svc03_grade_suscetibilidade.py` e fica em
`modelo/grade/` e `modelo/execucoes/svc-03-grade/`.

| Regiao | Resolucao | Artefatos |
|---|---:|---|
| Recife | 120 m | CSV, GeoTIFF, PNG, resumo JSON |
| Curitiba | 120 m | CSV, GeoTIFF, PNG, resumo JSON |
| Petropolis | 120 m | CSV, GeoTIFF, PNG, resumo JSON |

## 8. Auditorias complementares

Chuva, temporalidade, recuperacao de fonte, cadeia seletiva e camada social sao
auditadas como evidencias auxiliares. Elas nao mudam pesos, portoes, maturidade ou
promocao de modelo dentro do contrato de inferencia.

Quando uma frente e inconclusiva, o estado deve continuar escrito como inconclusivo.
Quando o protocolo nao testa uma hipotese naquele recorte, o estado deve continuar
escrito como nao testavel.
