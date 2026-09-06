# Consolidacao da dimensao temporal do REV-P

Gerado em 2026-09-01T19:00:09.236577Z. Contrato `v1`, inalterado. Portao temporal: DESLIGADO.

## 1. Auditoria do handoff
13 de 13 afirmacoes confirmadas por execucao.

Nenhuma afirmacao refutada nesta rodada.

## 2. Ate onde cada teste temporal chegou
| region_id | model_id | estado_execucao | motivo_da_parada | support |
|---|---|---|---|---|
| curitiba | fluvial_planicie | EVALUABLE | nan | not_testable |
| dominio_de_ajuste:fluvial_planicie | fluvial_planicie | EVALUABLE | nan | inconclusive |
| petropolis | fluvial_serra | EVALUABLE | nan | not_testable |
| dominio_de_ajuste:fluvial_serra | fluvial_serra | EVALUABLE | nan | inconclusive |
| nyc | pluvial_urbano_nyc | DECLARED | protocolo_nao_declara_teste_neste_escopo | not_testable |
| recife | recife_pluvial | DECLARED | corte_declarado_nao_materializou_nenhum_fold | not_testable |
| uk | MOD-PROSP-02:TERRENO/bloco | EVALUABLE | nan | inconclusive |
| uk | MOD-PROSP-02:COMPLETO/bloco | EVALUABLE | nan | inconclusive |
| curitiba | SUSC-20O/20Q:logistica_l2_diagnostica | EVALUABLE | nan | inconclusive |

## 3. Suporte por regiao x modelo
| region_id | model_id | maturidade | support | razao_tecnica |
|---|---|---|---|---|
| curitiba | fluvial_planicie | transferencia_caracterizada | not_testable | protocolo_nao_declara_teste_neste_escopo |
| dominio_de_ajuste:fluvial_planicie | fluvial_planicie | nan | inconclusive | serie_curta_1_corte(s)_minimo_3 |
| petropolis | fluvial_serra | transferencia_sem_referencia_local | not_testable | sem_dado_para_montar_teste |
| dominio_de_ajuste:fluvial_serra | fluvial_serra | nan | inconclusive | serie_curta_1_corte(s)_minimo_3 |
| nyc | pluvial_urbano_nyc | mvp_local | not_testable | protocolo_nao_declara_teste_neste_escopo |
| recife | recife_pluvial | mvp_local | not_testable | corte_declarado_nao_materializou_nenhum_fold |
| uk | MOD-PROSP-02:TERRENO/bloco | nan | inconclusive | todos_os_cortes_discriminam_acima_do_acaso |
| uk | MOD-PROSP-02:COMPLETO/bloco | nan | inconclusive | todos_os_cortes_discriminam_acima_do_acaso |
| curitiba | SUSC-20O/20Q:logistica_l2_diagnostica | nan | inconclusive | cortes_sem_intervalo_para_separar_queda_de_ruido |

Maturidade e suporte temporal sao colunas diferentes de proposito: nenhuma determina a outra.

## 4. Shadow G6
3 de 7 casos mudariam de resposta se o portao temporal fosse ligado.

| case_id | region | v1_status | v1_gate | temporal_support | shadow_status | changed |
|---|---|---|---|---|---|---|
| C01 | recife | ok | - | not_testable | insufficient_data | True |
| C02 | curitiba | ok | - | not_testable | insufficient_data | True |
| C03 | petropolis | ok | - | not_testable | insufficient_data | True |
| C04 | petropolis | insufficient_data | variaveis_presentes | - | insufficient_data | False |
| C05 | recife | insufficient_data | variaveis_presentes | - | insufficient_data | False |
| C06 | recife | insufficient_data | geometria_valida | - | insufficient_data | False |
| C07 | - | region_not_supported | regiao_resolvida | - | region_not_supported | False |

## 5. Harness de challengers
`temporal_not_worse` deixou de ser calculado no harness e passou a ser DERIVADO do estado de suporte. Ausencia de teste, que antes entrava como `false` (isto e, como piora demonstrada), agora sai como `not_testable` -- e continua bloqueando revisao humana, pelo motivo certo

- avaliacoes conferidas: 12
- mudanca de elegibilidade: 0
- mudanca apenas semantica: 0

## 6. Proveniencia
6 artefatos distintos; integro: True.

## 7. Recuperacao de fonte
8 de 8 perguntas recuperam a fonte atual; termos proibidos em 0.

## 8. Contrato
- portoes publicos: ['geometria_valida', 'regiao_resolvida', 'modelo_para_a_regiao', 'variaveis_presentes', 'dominio_coberto']
- G6 existe internamente: True
- G6 publico: False
- G6 ativo: False
- decisoes da v1 alteradas: False
- **V2_REQUIRED**

com a evidencia atual, ativar o portao temporal faria a v1 deixar de emitir escore para curitiba (not_testable), petropolis (not_testable), nyc (not_testable), recife (not_testable). Mudar comportamento de recusa e mudanca semantica, e docs/CONTRATO_VERSIONAMENTO.md exige `v2` para isso.

## 9. O que esta rodada NAO mudou
- modelo servido: inalterado;
- maturidades: inalteradas;
- contrato: continua `v1`, com os mesmos cinco portoes publicos;
- portao temporal: desligado;
- nenhum challenger promovido;
- resultados historicos: preservados.
