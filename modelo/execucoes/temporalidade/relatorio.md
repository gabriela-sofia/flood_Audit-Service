# Auditoria de temporalidade do REV-P

## 1. O que validade temporal passou a significar no REV-P
Treinar com evidencia observavel ate t e testar em eventos posteriores a t sem vazamento, preservando separacao entre tempo fisico, tempo observacional, processo de rotulo e processo de amostragem.

## 2. Quais regioes suportam teste temporal
- Environment Agency/UK - MOD-PROSP-02 TERRENO bloco: IDENTIFICAVEL -> TEMPORAL_SUPPORT
- Curitiba 2026 - SUSC-20O/20Q: PARCIALMENTE_IDENTIFICAVEL -> TEMPORAL_INCONCLUSIVE
- Curitiba leave-one-event-out: NAO_IDENTIFICAVEL -> TEMPORAL_TEST_INVALID
- Copernicus EMS - capacidade temporal: PARCIALMENTE_IDENTIFICAVEL -> TEMPORAL_INCONCLUSIVE
- Sen1Floods11 - capacidade temporal: NAO_IDENTIFICAVEL -> TEMPORAL_TEST_INVALID
- Petropolis - capacidade temporal: NAO_IDENTIFICAVEL -> TEMPORAL_TEST_INVALID

## 3. Controle ingles
MOD-PROSP-02 e o controle positivo: identificavel, com AUC medio 0.7992 no TERRENO bloco.

## 4. Curitiba 2026
Ha degradacao observada no holdout futuro, mas a auditoria classifica a identificabilidade como parcial.

## 5. O AUC 0,52 mede o que?
Mede ranking na amostra administrativa futura de 2026 sob D0. Nao mede sozinho P(Y|X), clima ou falha pura do mecanismo.

## 6. Chuva
Curitiba: indices de chuva mudam e chuva isolada nao transporta no holdout historico.

## 7. Negativos
Curitiba: prevalencia anual varia e negativos exigem desenho temporal proprio.

## 8. Mecanismo fisico
Topografia fica majoritariamente estavel em drift; coeficientes fisicos por ano permanecem mistos e devem ser lidos como observacionais.

## 9. Placebos
periodo_por_chuva AUC=0.6411

## 10. Taxonomia final de Curitiba
TEMPORAL_INCONCLUSIVE: ha degradacao observada no holdout, mas a magnitude nao esta separada de forcing, sampling e observation shift

## 11. Consequencia para Recife, Petropolis e transferencia
Recife e Petropolis nao viram estabilidade temporal por ausencia de estrutura local suficiente; transferencia espacial nao implica transferencia temporal.

## 12. Consequencia para SVC-07 e promocao
Sem alteracao do contrato ou modelos. A avaliacao temporal proposta e modo de revisao.

## 13. Bugs/leakage encontrados
Nenhum novo vazamento corrigido nesta camada; a auditoria registra testes para futuro no treino, evento atravessando fold, scaler global e AUC de classe unica.

## 14. Testes
Executar `python -m pytest tests/ -q` apos gerar os artefatos.

## 15. Artefatos
`local_runs/temporalidade/` contem ontologia, catalogo historico, matrizes, diagnosticos e resumos JSON.

## 16. O que continua desconhecido
Curitiba ainda nao separa completamente forcing shift, sampling shift, observation shift e possivel conditional shift.

## 17. Proximo experimento de maior valor informativo
desenho event-matched para negativos de Curitiba, congelado antes do resultado, com oportunidade de observacao e janela temporal comparaveis
