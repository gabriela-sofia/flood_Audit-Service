# Cartao do modelo

## Uso pretendido

O modelo estima suscetibilidade fisico-hidrologica a inundacoes em regioes declaradas
no contrato. A saida e uma resposta auditavel por geometria ou celula de grade, com
escore, intervalo, maturidade regional, distancia de dominio e explicacao estruturada.

O modelo nao e alerta em tempo real, previsao meteorologica, diagnostico individual,
medida de vulnerabilidade social ou prova de risco para uma pessoa ou imovel.

## Modelos candidatos

Fonte: `modelo/execucoes/svc-01-modelos/indice.json`.

| Modelo | Servido | Regiao | AUC | Leitura |
|---|---|---|---:|---|
| `recife_pluvial` | Sim | Recife | 0,6422 | Modelo local minimo; uso permitido com maturidade declarada. |
| `pluvial_urbano_nyc` | Sim | NYC | 0,6089 | Modelo urbano servido com desempenho fora de criterio forte. |
| `pluvial_urbano_sao_paulo` | Nao | Sao Paulo | 0,4753 | Abaixo do acaso; nao deve ser usado para inferencia. |
| `fluvial_planicie` | Sim | Curitiba | 0,7334 | Transferencia caracterizada para planicie. |
| `fluvial_serra` | Sim | Petropolis | 0,7916 | Transferencia sem referencia local. |

## Variaveis

| Modelo | Variaveis |
|---|---|
| `recife_pluvial` | `elevation_rel_m`, `slope_deg`, `hand_m`, `twi_dinf`, `rain_decay_index` |
| `pluvial_urbano_nyc` | `elevation_rel_m`, `slope_deg`, `hand_m`, `twi_dinf` |
| `pluvial_urbano_sao_paulo` | `elevation_rel_m`, `slope_deg`, `hand_m`, `twi_dinf` |
| `fluvial_planicie` | `elevation_rel_m`, `slope_deg`, `hand_m`, `twi_dinf` |
| `fluvial_serra` | `hand_m` |

## Maturidade regional

| Regiao | Modelo | Maturidade |
|---|---|---|
| Recife | `recife_pluvial` | `mvp_local` |
| Curitiba | `fluvial_planicie` | `transferencia_caracterizada` |
| Petropolis | `fluvial_serra` | `transferencia_sem_referencia_local` |
| NYC | `pluvial_urbano_nyc` | `referencia_externa` |

Sao Paulo aparece como evidencia analisada, mas o candidato regional nao e servido.

## Grade publica

Fonte: `modelo/execucoes/svc-03-grade/resultado.json`.

| Regiao | p50 | p90 | Cobertura servivel |
|---|---:|---:|---:|
| Recife | 0,5808 | 0,7785 | 97,0% |
| Curitiba | 0,2746 | 0,5804 | 98,5% |
| Petropolis | 0,3460 | 0,5634 | 91,3% |

## Condicoes de recusa

O contrato deve recusar ou marcar extrapolacao quando:

1. A regiao nao esta no contrato.
2. O modelo regional nao e servivel.
3. A maturidade minima nao esta declarada.
4. Variaveis obrigatorias estao ausentes.
5. O caso sai do dominio de covariaveis sem regra explicita de uso.

## Limitacoes

Os resultados sao condicionados aos dados curados e aos dominios declarados. As
auditorias temporais ainda contem estados inconclusivos e nao testaveis. A camada social
e paralela ao escore fisico e permanece em revisao. Nenhum resultado deve ser lido como
validacao prospectiva geral para o Brasil.
