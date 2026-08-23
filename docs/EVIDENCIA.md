# Evidência

Por que se pode confiar no que este repositório afirma.

Este documento é a linhagem científica do projeto: cada rodada que produziu um número
do artigo, na ordem em que aconteceu, com a data, o script que a executou, o artefato
que ela deixou e o veredito. Ele existe porque afirmação sem procedência é opinião — e
porque o critério de leitura foi fixado **antes** de cada rodada, o que só é verificável
se a ordem cronológica estiver escrita.

Substitui 22 documentos separados. Estado em 22/08/2026.

---

## 1. As quatro regras de decisão

Fixadas em 16/08/2026, valem para toda rodada posterior e explicam a maior parte das
recusas registradas na tabela abaixo.

| Regra | Enunciado | Consequência prática |
|---|---|---|
| **0** | Critério não se ajusta por resultado | A faixa de AUC esperada, o valor lido como suspeita de vazamento e o sinal obrigatório de cada variável ficam escritos antes de rodar. Nenhuma rodada foi refeita depois de olhar o teste |
| **1** | Folga de derivação de terreno se decide por métrica física, nunca por contagem de cobertura | Um produto de terreno não é aceito porque "cobre 98% da área", e sim porque a métrica bate contra o raster de referência |
| **2** | Ausência de registro não é negativo | Dos 442 pontos avaliados, 114 passam nos quatro critérios de exclusão qualificada; os 328 restantes seguem como `ausencia` e **não entram como negativo** |
| **3** | Estimador de decisão vem com intervalo | Nenhum número de decisão é reportado como ponto. O IC vem de bootstrap reamostrando **grupos**, nunca linhas |

---

## 2. A tabela-mestra das rodadas

Ordem cronológica. "Veredito" é o que a rodada decidiu, não o que se esperava dela.

| Data | Rodada | Script | Artefato | O que produziu | Veredito |
|---|---|---|---|---|---|
| 07/08 | Balanço da frente externa | — | — | Mapa das lacunas por região; nomeia `C4_BLOCKED_NO_FORMAL_NEGATIVES` como o gargalo real | Lacuna declarada |
| 07/08 | Contrato de redução a pontos | `pontos/n1f_reduzir_a_pontos.py` | `n1f` | A regra que impede contar o mesmo evento duas vezes ao converter polígono em ponto | Contrato fixado |
| 07/08 | Adjudicação do negativo — AOI Inglaterra | `uk/ext_uk05_contabilidade_negativo.py` | `ext-uk-05` | Os quatro critérios N1–N4 que definem exclusão qualificada | Critério fixado |
| 09/08 | **Critérios de acerto** | — | — | Faixa de AUC 0,70–0,88, limiar de suspeita de vazamento, sinal obrigatório por variável | **Fixado antes de tudo** |
| 09/08 | O que não é enchente (v1 e v2) | `mod_neg01_o_que_nao_e_enchente.py` | `mod-neg-01` | Mede o efeito da definição de negativo sobre o AUC | O ajuste de maior desempenho próprio é o que menos transfere |
| 10/08 | Análogos de Petrópolis | `cems/cems01_analogos_por_regiao.py` | `cems-02-analogos-v2` | Ranking de ativações análogas por relevo | **Falhou** — o centroide da ativação caía onde não há AOI |
| 11/08 | Modelo de encosta v1 | `mod_serra01_ingreme_2features.py` | `mod-serra-01` | Primeiro ajuste em terreno íngreme, 2 variáveis | Superado pela v2 (orçamento de EPV errado) |
| 12/08 | Cadeia de terreno harmonizada | `terreno/ter01_cadeia_harmonizada.py`, `ter02_reextrair_e_comparar.py` | `ter-01`, `ter-02` | Rederivação de HAND/TWI pela mesma cadeia D-infinity em todas as fontes | Reprodução bit a bit contra o raster de referência |
| 12/08 | HAND incomparável entre regiões | `terreno/ter01_cadeia_harmonizada.py` | `ter-01/{recife,curitiba,petropolis}` | Achado: o `hand_m` das três regiões **não era a mesma variável** antes da harmonização | Bloqueia comparação até rederivar |
| 12/08 | Modelo fluvial multirregião | `mod_mec01_fluvial_multirregiao.py` | `ds-02`, `mod-mec-01` | Primeiro ajuste com cadeia única e mecanismo único | Viável |
| 12/08 | Validação prospectiva v1 | `mod_prosp01_holdout_temporal.py`, `terreno/ter03_reextrair_brasil.py` | `mod-prosp-01` | Holdout temporal preliminar | `PROSPECTIVAMENTE_ESTAVEL`; Recife não pode ser somada ao conjunto externo |
| 12/08 | Resolução e mecanismo | — | — | Decide a resolução do MDT e a separação por mecanismo | Decisão fechada |
| 13/08 | **Tabela única e pool harmonizado** | `treino/ds03_esquema_alvo.py`, `ds04_reduzir_por_fonte.py`, `ds05_admissao_consolidacao.py` | `ds-03`, `ds-04`, `ds-05` | A base de **65.070 pontos** elegíveis ao ajuste fluvial, reduzidos de seis fontes | Base consolidada |
| 14/08 | Resolução única de 30 m | `terreno/ter05_harmonizar_uk.py`, `ter06_harmonizar_chips_nivel1.py` | `ter-05`, `ter-06` | Toda a base na mesma resolução, mesmo onde custa detalhe | Comparabilidade garantida |
| 16/08 | Chuva de fonte única em Recife | `mod_recife03_pluvial_fonte_unica.py` | `mod-recife-03` | Recife passa a usar produto único de chuva | LOO-AUC cai para 0,6409 — **abaixo da faixa fixada** |
| 16/08 | **Regras de decisão** | — | — | As quatro regras da §1 | Fixadas |
| 20/08 | **E3 — Ajuste por classe de relevo** | `treino/mod_serra03_relevo_ds05.py` | `modelo/execucoes/mod-serra-03/` | Serra 0,7916; planície 0,7245; transferência 0,7957 | `COERENTE_COM_CRITERIOS` |
| 20/08 | **E4 — Holdout temporal** | `treino/mod_prosp02_holdout_temporal_ds05.py` | `modelo/execucoes/mod-prosp-02/` | 201 eventos, 110 datas, 8 cortes, AUC médio 0,7992 | `COERENTE_COM_CRITERIOS`; Curitiba não sustenta o teste |
| 20/08 | **E5 — Grade de suscetibilidade** | `servico/svc03_grade_suscetibilidade.py` | `modelo/grade/` | 56.666 / 65.275 / 172.015 células a 120 m | Concluída |
| 20/08 | **E6 — Contrato de inferência** | `servico/svc01_construir_modelos_servidos.py`, `svc02_contrato_inferencia.py` | `modelo/execucoes/svc-01-modelos/`, `svc-02-contrato/` | Três modelos servidos, cinco portões, IC por bootstrap de grupos | Contrato executável; falta transporte HTTP |
| 20/08 | **Estado da chuva no projeto** | `treino/aud_chuva02_escala_do_contraste.py` | `modelo/execucoes/aud-chuva-02/` | Cobertura de 14% para 99,99%, produto único | A chuva **não discrimina** na escala do modelo (~11 km) |

Alguns scripts citados na coluna "Script" são de rodadas anteriores à consolidação e não
estão versionados neste repositório — a tabela os nomeia porque é o registro do que rodou,
não do que ficou.

---

## 3. Os achados que mudaram o rumo

**A variável derivada do rótulo (09/08).** Acurácia balanceada de 0,8855 com elas,
**0,4834** sem. O número alto era o modelo lendo a resposta. Rota fechada.

**O `hand_m` não era a mesma variável (12/08).** Antes da rederivação pela cadeia
própria, o HAND de Recife, Curitiba e Petrópolis vinha de produtos diferentes, em
resoluções diferentes. Comparar regiões nessas condições teria medido diferença de
produto e chamado de diferença de território.

**O centroide de Petrópolis (10/08).** O ranking de ativações análogas escolheu
EMSR867, em Madagascar, porque o centroide da ativação caía no planalto central com
513 m de relevo. As AOIs efetivamente mapeadas estavam no litoral, a 4,7 e 11,5 m de
altitude. A analogia passou a ser medida **onde o fenômeno acontece**, não no centroide.

**O Copernicus EMS não serve à AOI inglesa (07/08).** A API pública só expõe EMSR656 em
diante, de 2023; nenhuma das 7 ativações do Reino Unido cai na AOI do piloto. A hipótese
de usar CEMS como negativo naquela região caiu, e o negativo inglês passou a vir da
Environment Agency por exclusão qualificada.

**O orçamento de EPV do estrato íngreme (20/08).** O estrato tem 19 eventos positivos e
comporta **uma** variável, não quatro. O `MOD-SERRA-01`, que produziu os coeficientes
citados até então, usou duas. A conclusão sobreviveu; os números mudaram, e o texto
passou a citar os novos.

**Recife não atinge o próprio critério (16/08).** Com a fonte de chuva corrigida, o
LOO-AUC é 0,6409 — abaixo da faixa fixada em 09/08 — e o IC de `hand_m` cruza zero. O
serviço responde mesmo assim, com a falha escrita na resposta e a maturidade rebaixada
para `mvp_local`. Esconder o escore esconderia também o quanto ele é fraco.

---

## 4. O que a chuva custou

A chuva é a essência física do fenômeno e a variável mais cara de medir de forma
comparável. Três rodadas, nesta ordem:

1. Aquisição do CHIRPS por janela em COG, com cache mensal. Um mês só era gravado se
   todos os dias chegassem.
2. Reexecução: **163 minutos, zero meses novos**, 939 falhas num flavor e 1.297 de 1.297
   no outro. O servidor da fonte passou a recusar as requisições — tem proteção contra
   scraping documentada, e o pipeline bateu nele com 2.594 pedidos mais retentativas.
3. Reescrita com cache **por dia**, uma requisição por dia. Recuperou o que a segunda
   perdeu.

Depois disso, a auditoria de confundimento retirou o CHIRPS e a base passou a usar
**produto único (Open-Meteo/ERA5-Land)**, com cobertura de 99,99%. Os três scripts de
aquisição do CHIRPS não estão neste repositório: são a rota aposentada, e mantê-los
sugeriria que ela ainda vale.

O achado final é o mais importante: recuperada e unificada, a chuva **não discrimina na
escala do modelo**. Ela é medida em células de ~11 km enquanto o modelo compara pontos
dentro do mesmo evento. Entra como cenário, não como camada.

---

## 5. O que continua em aberto

| Item | Estado |
|---|---|
| Negativo formal nas três regiões brasileiras | `C4_BLOCKED_NO_FORMAL_NEGATIVES` permanece aberto — ver `METODO.md` §5 |
| Inventário local de Petrópolis | Ausente; solicitações formais a DRM-RJ e Defesa Civil sem resposta no período |
| Transporte HTTP do serviço | O contrato roda como função pura; falta expor |
| Holdout próprio de Curitiba | Não sustentável com 114 negativos contra 1.238 positivos |
| Reprodução entre ambientes | 21 de 22 testes de invariante passam. No ambiente fixado do projeto a reprodução bate exata; entre ambientes diferentes ainda depende de versão de biblioteca |
