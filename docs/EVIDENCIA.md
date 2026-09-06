# Evidencia

Este arquivo resume o que esta afirmado pelos artefatos curados. Planejamento e poster
sao tratados como documentos de sintese; quando houver conflito, prevalece o artefato
versionado em `modelo/execucoes/` e o codigo que o reproduz.

## 1. Tabela unica

Fonte: `modelo/execucoes/ds-05-tabela-unica/manifesto_v2.json`.

| Item | Valor |
|---|---:|
| Observacoes de entrada | 595.865 |
| Observacoes admitidas | 568.265 |
| Pool fluvial | 65.070 |
| Grupos admitidos | 6.873 |
| Grupos no pool fluvial | 2.002 |

Composicao admitida por fonte:

| Fonte | Admitidos | Pool fluvial | Grupos admitidos |
|---|---:|---:|---:|
| NYC | 471.727 | 0 | 2.402 |
| Sen1Floods11 | 30.431 | 30.431 | 11 |
| CEMS | 25.483 | 25.483 | 119 |
| Urban Flood Observations | 24.044 | 0 | 215 |
| UK | 7.476 | 7.476 | 401 |
| Sao Paulo | 7.146 | 0 | 1.976 |
| Curitiba | 1.680 | 1.680 | 1.471 |
| Recife | 278 | 0 | 278 |

Veredito: a tabela esta consolidada por manifesto publico. As tabelas completas nao
entram neste pacote por tamanho.

## 2. Modelos servidos

Fonte: `modelo/execucoes/svc-01-modelos/indice.json`.

| Modelo | n | Grupos | AUC | Servivel | Veredito |
|---|---:|---:|---:|---|---|
| `recife_pluvial` | 269 | 227 | 0,6422 | Sim | Fora dos criterios fortes, mas disponivel para Recife com maturidade local declarada. |
| `pluvial_urbano_nyc` | 471.727 | 751 | 0,6089 | Sim | Fora dos criterios fortes, mas disponivel para NYC. |
| `pluvial_urbano_sao_paulo` | 7.146 | 1.438 | 0,4753 | Nao | Abaixo do acaso; nao deve ser servido. |
| `fluvial_planicie` | 56.654 | 509 | 0,7334 | Sim | Coerente com criterios; usado para Curitiba. |
| `fluvial_serra` | 5.162 | 24 | 0,7916 | Sim | Coerente com criterios; usado para Petropolis. |

Veredito: Curitiba nao e descrita como modelo local validado. Ela e uma regiao servida
por transferencia caracterizada a partir do modelo fluvial de planicie.

## 3. Contrato de inferencia

Fonte: `modelo/execucoes/svc-02-contrato/`.

O contrato demonstra cobertura por regiao, portoes, maturidade e explicacao
estruturada. A decisao cientifica fica no modulo
`scripts/servico/svc02_contrato_inferencia.py`.

Veredito: o contrato esta implementado localmente como funcao pura. Transporte HTTP,
interface publica e operacao em tempo real nao sao afirmados por este pacote.

## 4. Grade de aplicacao

Fonte: `modelo/execucoes/svc-03-grade/resultado.json`.

| Regiao | Celulas validas | Serviveis | Serviveis/validas | p50 | p90 |
|---|---:|---:|---:|---:|---:|
| Recife | 56.666 | 54.963 | 97,0% | 0,5808 | 0,7785 |
| Curitiba | 65.275 | 64.310 | 98,5% | 0,2746 | 0,5804 |
| Petropolis | 172.015 | 157.005 | 91,3% | 0,3460 | 0,5634 |

Veredito: a grade publica existe em CSV, GeoTIFF e PNG para as tres regioes. As
checagens contra o contrato nao registram divergencias nos casos amostrados.

## 5. Chuva antecedente

Fonte: `modelo/execucoes/chuva-case-crossover/forcing_support_matrix.json`.

| Recorte | Razao de risco por 10 mm | IC | Estratos informativos | Suporte |
|---|---:|---|---:|---|
| NYC | 1,5099 | 1,4645 a 1,5610 | 8.552 | `supported` |
| Sao Paulo | 1,2158 | 1,0470 a 1,4998 | 1.753 | `supported` |

Veredito: a chuva antecedente tem suporte como forcamento de evento nos recortes
testados. Esse resultado nao promove sozinho nenhum modelo regional.

## 6. Temporalidade

Fonte: `modelo/execucoes/temporalidade-integracao/scientific_support_matrix.json`.

| Hipotese | Estado |
|---|---|
| Curitiba, fluvial de planicie | `not_testable` no escopo declarado. |
| Dominio de ajuste, fluvial de planicie | `inconclusive`. |
| Petropolis, fluvial de serra | `not_testable` por falta de dado local. |
| Dominio de ajuste, fluvial de serra | `inconclusive`. |
| NYC pluvial urbano | `not_testable`. |
| Recife pluvial | `not_testable`. |
| UK, protocolo prospectivo | `inconclusive`. |
| Curitiba, diagnostico local | `inconclusive`. |

Veredito: a frente temporal nao deve ser apresentada como validacao prospectiva ampla.
Ela registra limites de identificabilidade.

## 7. Recuperacao de fonte

Fonte: `modelo/execucoes/recuperacao/comparacao_recuperadores_k5.json`.

| Recuperador | Recall agregado | MRR agregado | Recall em morfologia |
|---|---:|---:|---:|
| Lexical | 0,7043 | 0,6886 | 0,1189 |
| Denso | 0,9675 | 0,8808 | 0,9482 |
| Hibrido | 0,8872 | 0,8847 | 0,4207 |

Veredito: o resultado de 0,9482 e especifico do subconjunto de morfologia. O agregado
do recuperador denso e 0,9675.

## 8. Cadeia seletiva

Fonte: `modelo/execucoes/neural-catraca/experimento_pote_crescente_redecidido.json`.

O experimento confirmatorio registra `n=6`, ganho mediano de `0,119623` e efeito
detectado por testes de postos e sinal. O proprio artefato limita a interpretacao:
trata-se de reconstrucao mascarada em recortes reservados, nao de deteccao direta de
agua.

## 9. Camada social territorial

Fonte: `modelo/execucoes/camada-social-territorial/ibge-curitiba/manifesto_ibge_curitiba_v1.json`.

Curitiba possui 3.190 setores na malha, 3.190 com unidade social e 254 em FCU no
recorte auditado. A camada social v0 e de revisao: nao altera escore fisico, nao treina
modelo, nao cria rotulo hidrologico e nao permite concluir baixa vulnerabilidade quando
faltam dados.
