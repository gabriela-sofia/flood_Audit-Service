# Matriz de evidências do artigo v22

As tabelas seguem a ordem do artigo de 10 de novembro. Cada linha identifica a comparação, o código e o registro que a sustenta. Os vínculos dos parágrafos, legendas e referências estão nos manifestos.

[Matriz completa em JSON](../manifestos/afirmacoes_artigo.json) · [Matriz em CSV](../manifestos/afirmacoes_artigo.csv)

## Tabela I — Relações físicas e teste externo

| Item | Resultado | Registro e código | Leitura |
|---|---|---|---|
| I.01 | Muçum: mediana do HAND  ·  2,2 m com água  ·  61,2 m no comparador | [Evidência](../evidencias/04_recife/fundamentacao-artigo-20261106/f08_ruas_e_comunidades.json) · [Código](../codigo/scripts/fundamentacao_artigo/f08_ruas_e_comunidades.py) | Muçum ilustra o contraste entre observações do episódio. |
| I.02 | Dentro da cheia, 331.492 pontos (a)  ·  -1,13  ·  [-1,31; -0,96] | [Evidência](../evidencias/02_aprendizado/mecanismos-transferiveis/v9/identificacao_causal_v1/resultado.json) · [Código](../codigo/scripts/metodo/mt09_identificacao_causal.py) | Associação por desvio-padrão de HAND dentro da cheia. |
| I.03 | Pares a até 3 km (a)  ·  -1,27  ·  [-1,48; -1,05] | [Evidência](../evidencias/02_aprendizado/mecanismos-transferiveis/v9/identificacao_causal_v1/resultado.json) · [Código](../codigo/scripts/metodo/mt09_identificacao_causal.py) | Comparação espacial de pontos próximos. |
| I.04 | HAND maior, menos água em 25 de 27 territórios. | [Evidência](../evidencias/02_aprendizado/relatorios/MECANISMOS_TRANSFERIVEIS_EXECUCAO_MT01_2026_10_10.md) · [Código](../codigo/scripts/metodo/mt01_experimento.py) | Sinal de HAND em ajustes territoriais individuais. |
| I.05 | Modelo geral - HAND  ·  +0,039  ·  [0,026; 0,054], 33 a favor | [Evidência](../evidencias/02_aprendizado/mecanismos-transferiveis/v1/resultado.json) · [Código](../codigo/scripts/metodo/mt01_experimento.py) | Unidades deixadas fora uma por vez. |
| I.06 | Retirar pistas de acumulação  ·  -0,064  ·  [-0,080; -0,048], 39 pioram | [Evidência](../evidencias/02_aprendizado/mecanismos-transferiveis/v1/resultado.json) · [Código](../codigo/scripts/metodo/mt01_experimento.py) | Ablação das pistas de acumulação. |
| I.07 | Núcleo - HAND (b)  ·  +0,055  ·  [0,024; 0,086] | [Evidência](../evidencias/02_aprendizado/mt11-fechamento/v2/reparo_grafo_3b.json) · [Código](../codigo/scripts/metodo/mt11_emenda.py) | Núcleo congelado com Brasil no ajuste e teste externo. IC98,33%. |
| I.08 | Com - sem atributos de conexão (b)  ·  +0,030  ·  [0,006; 0,053] | [Evidência](../evidencias/02_aprendizado/mt11-fechamento/v2/reparo_grafo_3b.json) · [Código](../codigo/scripts/metodo/mt11_emenda.py) | Incremento de conexão no teste externo. IC98,33%. |
| I.09 | Regressão - HAND, 16 cheias (c)  ·  +0,041  ·  13/16, p=0,011 | [Evidência](../evidencias/02_aprendizado/gfd-quarentena/executado.json) · [Código](../codigo/scripts/metodo/gfdq03_teste_confirmatorio.py) | Comparação em 16 cheias sul-americanas fora do Brasil. |
| I.10 | HAND (log)  ·  -0,981  ·  -0,904  ·  Areia  ·  -0,051  ·  -0,202 | [Evidência](../evidencias/02_aprendizado/transferencia-vulnerabilizados/v9_evolucao_abrangente/p0_regularizacao/pesos.json) · [Código](../codigo/scripts/vulnerabilizados/v9_p0_regularizacao.py) | Coeficientes padronizados das versões completa e com reserva. |
| I.11 | Distância ao canal  ·  -0,429  ·  -0,433  ·  Contribuição (log)  ·  +0,043  ·  +0,085 | [Evidência](../evidencias/02_aprendizado/transferencia-vulnerabilizados/v9_evolucao_abrangente/p0_regularizacao/pesos.json) · [Código](../codigo/scripts/vulnerabilizados/v9_p0_regularizacao.py) | Coeficientes padronizados das versões completa e com reserva. |
| I.12 | Declive  ·  -0,312  ·  -0,274  ·  Dist. à drenagem  ·  +0,030  ·  +0,055 | [Evidência](../evidencias/02_aprendizado/transferencia-vulnerabilizados/v9_evolucao_abrangente/p0_regularizacao/pesos.json) · [Código](../codigo/scripts/vulnerabilizados/v9_p0_regularizacao.py) | Coeficientes padronizados das versões completa e com reserva. |
| I.13 | Fração construída  ·  -0,210  ·  -0,170  ·    ·   · | [Evidência](../evidencias/02_aprendizado/transferencia-vulnerabilizados/v9_evolucao_abrangente/p0_regularizacao/pesos.json) · [Código](../codigo/scripts/vulnerabilizados/v9_p0_regularizacao.py) | Coeficientes padronizados das versões completa e com reserva. |

## Tabela II — Transferência para comunidades e recortes urbanos

| Item | Resultado | Registro e código | Leitura |
|---|---|---|---|
| II.01 | RMPA  ·  241 (17)  ·  0,880 [0,828; 0,923]  ·  —  ·  — | [Evidência](../evidencias/03_transferencia/fundamentacao-artigo-20261028/f03_ic_transferencia_coortes.json) · [Código](../codigo/scripts/fundamentacao_artigo/f03_ic_transferencia_coortes.py) | FCU da RMPA. Ganho sobre HAND inconclusivo. |
| II.02 | Manaus  ·  48 (10)  ·  0,920 [0,791; 0,967]  ·  +0,046 [0,020; 0,069]  ·  +0,111 | [Evidência](../evidencias/03_transferencia/fundamentacao-artigo-20261028/f03_ic_transferencia_coortes.json) · [Código](../codigo/scripts/fundamentacao_artigo/f03_ic_transferencia_coortes.py) | Transferência externa sem ajuste local. IC95% por grupos de FCU. |
| II.03 | Recife  ·  31 (19)  ·  0,866 [0,777; 0,935]  ·  +0,064 [0,012; 0,125]  ·  +0,151 | [Evidência](../evidencias/03_transferencia/fundamentacao-artigo-20261028/f03_ic_transferencia_coortes.json) · [Código](../codigo/scripts/fundamentacao_artigo/f03_ic_transferencia_coortes.py) | Transferência externa sem ajuste local. IC95% por grupos de FCU. |
| II.04 | RMR  ·  366 (35)  ·  0,809 [0,721; 0,866]  ·  +0,116 [0,069; 0,145]  ·  +0,093 | [Evidência](../evidencias/03_transferencia/fundamentacao-artigo-20261028/f03_ic_transferencia_coortes.json) · [Código](../codigo/scripts/fundamentacao_artigo/f03_ic_transferencia_coortes.py) | Transferência externa sem ajuste local. IC95% por grupos de FCU. |
| II.05 | RMPA  ·  6.117  ·  0,905  ·  +0,016 [0,005; 0,025]  ·  —  ·  0,884  ·  0,907 | [Evidência](../evidencias/03_transferencia/fundamentacao-artigo-20261106/f07_estratos_coortes.json) · [Código](../codigo/scripts/fundamentacao_artigo/f07_estratos_coortes.py) | Coorte completa. HAND fica abaixo do REV-P e declive fica acima. |
| II.06 | Manaus  ·  215  ·  0,791  ·  +0,036 [0,0004; 0,061]  ·  +0,105  ·  —  ·  0,804 | [Evidência](../evidencias/03_transferencia/fundamentacao-artigo-20261106/f07_estratos_coortes.json) · [Código](../codigo/scripts/fundamentacao_artigo/f07_estratos_coortes.py) | Coorte completa. Estrato de baixa renda com suporte insuficiente. |
| II.07 | RMC  ·  883  ·  0,811  ·  —  ·  +0,056  ·  —  ·  0,812 | [Evidência](../evidencias/03_transferencia/fundamentacao-artigo-20261106/f07_estratos_coortes.json) · [Código](../codigo/scripts/fundamentacao_artigo/f07_estratos_coortes.py) | Coorte completa com vantagem sobre declive. FCU descritivas com 11 positivos. |
| II.08 | Recife  ·  448  ·  0,781  ·  +0,043 [0,019; 0,068]  ·  +0,087  ·  0,844  ·  0,739 | [Evidência](../evidencias/03_transferencia/fundamentacao-artigo-20261106/f07_estratos_coortes.json) · [Código](../codigo/scripts/fundamentacao_artigo/f07_estratos_coortes.py) | Coorte completa e AUC nos estratos de Recife. |
| II.09 | RMR  ·  2.882  ·  0,775  ·  +0,085 [0,055; 0,106]  ·  +0,069  ·  0,770  ·  0,748 | [Evidência](../evidencias/03_transferencia/fundamentacao-artigo-20261106/f07_estratos_coortes.json) · [Código](../codigo/scripts/fundamentacao_artigo/f07_estratos_coortes.py) | Coorte completa e AUC nos estratos da RMR, que exclui Recife. |
| II.10 | Rótulos embaralhados  ·  Queda de AUC de 0,30 a 0,63 | [Evidência](../evidencias/03_transferencia/transferencia-vulnerabilizados/v8_validacao_integrada/analise/contrastes_sem_rotulo.csv) · [Código](../codigo/scripts/vulnerabilizados/v8_01_q_externo.py) | Controle com rótulos embaralhados. |
| II.11 | Outro sensor  ·  99% dos positivos da RMPA na mancha Planet/SkySat | [Evidência](../evidencias/03_transferencia/transferencia-vulnerabilizados/v6_evidencia_robusta/triangulacao_ufrgs/fracoes_a01.csv) · [Código](../codigo/scripts/vulnerabilizados/v6_08_ufrgs_triangulacao.py) | Triangulação no recorte comum entre sensores, com frações por cobertura. |

## Tabela III — Ruas com relatos em Recife

| Item | Resultado | Registro e código | Leitura |
|---|---|---|---|
| III.01 | 2014--2024: 1.648 relatos, 420 localizados até a rua, 96 ruas em 38 bairros. | [Evidência](../evidencias/04_recife/relatorios/RECIFE_APRENDIZADO_ALAGAMENTO_RESULTADOS_2026_10_24.md) · [Código](../codigo/scripts/recife_aprendizado/ra_01_referencias.py) | Geocodificação dos relatos e agrupamento rua–bairro. |
| III.02 | Em 16 de 71 bairros com registro, a ocorrência apareceu apenas nos relatos. | [Evidência](../evidencias/04_recife/fundamentacao-artigo-20261106/f08_ruas_e_comunidades.json) · [Código](../codigo/scripts/fundamentacao_artigo/f08_ruas_e_comunidades.py) | União dos bairros com registro no cotejo de relatos e imagem. |
| III.03 | AUC do REV-P  ·  0,669  ·  [0,564; 0,775] | [Evidência](../evidencias/04_recife/fundamentacao-artigo-20261028/f06_ic_auc_ruas_recife.json) · [Código](../codigo/scripts/fundamentacao_artigo/f06_ic_auc_ruas_recife.py) | AUC da versão com reserva dentro do bairro. |
| III.04 | AUC do HAND  ·  0,595  ·  [0,491; 0,711] | [Evidência](../evidencias/04_recife/fundamentacao-artigo-20261028/f06_ic_auc_ruas_recife.json) · [Código](../codigo/scripts/fundamentacao_artigo/f06_ic_auc_ruas_recife.py) | AUC do HAND nos mesmos comparadores. |
| III.05 | Ganho de AUC sobre o HAND  ·  +0,074  ·  [0,012; 0,159] | [Evidência](../evidencias/04_recife/relatorios/RECIFE_APRENDIZADO_ALAGAMENTO_RESULTADOS_2026_10_24.md) · [Código](../codigo/scripts/recife_aprendizado/ra_04_treino.py) | Diferença pareada. IC95% por bairro. |
| III.06 | Ruas com relato no topo de 20%  ·  42%  ·  HAND: 31% | [Evidência](../evidencias/04_recife/relatorios/RECIFE_APRENDIZADO_ALAGAMENTO_RESULTADOS_2026_10_24.md) · [Código](../codigo/scripts/recife_aprendizado/ra_04_treino.py) | Fração de ruas relatadas no quintil mais alto. |
| III.07 | Ganho em pares de comprimento semelhante^a  ·  +0,058  ·  [-0,024; 0,177] | [Evidência](../evidencias/04_recife/fundamentacao-artigo-20261106/f08_ruas_e_comunidades.json) · [Código](../codigo/scripts/fundamentacao_artigo/f08_ruas_e_comunidades.py) | Pareamento por comprimento até razão 1,5. Intervalo inclui zero. |
| III.08 | HAND (log)  ·  +0,751  ·  +0,204  ·  Areia  ·  +0,033  ·  +0,033 | [Evidência](../evidencias/04_recife/proveniencia/exemplos_ruas_e_figuras.json) · [Código](../codigo/scripts/recife_aprendizado/ra_04_treino.py) | Conta com a versão com reserva nas duas ruas de Estância. |
| III.09 | Dist. ao canal  ·  +0,853  ·  +0,093  ·  Área que drena (log)  ·  +0,129  ·  -0,013 | [Evidência](../evidencias/04_recife/proveniencia/exemplos_ruas_e_figuras.json) · [Código](../codigo/scripts/recife_aprendizado/ra_04_treino.py) | Conta com a versão com reserva nas duas ruas de Estância. |
| III.10 | Declive  ·  +0,272  ·  +0,123  ·  Dist. à drenagem  ·  -0,026  ·  +0,023 | [Evidência](../evidencias/04_recife/proveniencia/exemplos_ruas_e_figuras.json) · [Código](../codigo/scripts/recife_aprendizado/ra_04_treino.py) | Conta com a versão com reserva nas duas ruas de Estância. |
| III.11 | Fração construída  ·  -0,525  ·  -0,406  ·  Escore final  ·  1,487  ·  0,057 | [Evidência](../evidencias/04_recife/proveniencia/exemplos_ruas_e_figuras.json) · [Código](../codigo/scripts/recife_aprendizado/ra_04_treino.py) | Conta com a versão com reserva nas duas ruas de Estância. |
| III.12 | Número de relatos  ·  3  ·  4 | [Evidência](../evidencias/04_recife/proveniencia/exemplos_ruas_e_figuras.json) · [Código](../codigo/scripts/recife_aprendizado/ra_04_treino.py) | Conta com a versão com reserva nas duas ruas de Estância. |
| III.13 | Ruas com escore menor (de 74)  ·  72  ·  12 | [Evidência](../evidencias/04_recife/proveniencia/exemplos_ruas_e_figuras.json) · [Código](../codigo/scripts/recife_aprendizado/ra_04_treino.py) | Conta com a versão com reserva nas duas ruas de Estância. |

## Tabela IV — Encostas em Petrópolis

| Item | Resultado | Registro e código | Leitura |
|---|---|---|---|
| IV.01 | Modelo de água  ·  0,375  ·  [0,307; 0,445] | [Evidência](../evidencias/05_petropolis/fundamentacao-artigo-20261028/f04_encosta_e_retencao.json) · [Código](../codigo/scripts/fundamentacao_artigo/f04_encosta_e_retencao.py) | AUC por cicatriz. Fundo fora das cicatrizes observadas. |
| IV.02 | Modelo de encosta treinado no Idai  ·  0,660  ·  [0,593; 0,726] | [Evidência](../evidencias/05_petropolis/fundamentacao-artigo-20261028/f04_encosta_e_retencao.json) · [Código](../codigo/scripts/fundamentacao_artigo/f04_encosta_e_retencao.py) | AUC por cicatriz. Fundo fora das cicatrizes observadas. |
| IV.03 | Encosta - modelo de água  ·  +0,286  ·  [0,170; 0,396] | [Evidência](../evidencias/05_petropolis/fundamentacao-artigo-20261028/f04_encosta_e_retencao.json) · [Código](../codigo/scripts/fundamentacao_artigo/f04_encosta_e_retencao.py) | AUC por cicatriz. Fundo fora das cicatrizes observadas. |
| IV.04 | Externo com exemplos locais - só local (a)  ·  +0,126  ·  [0,070; 0,180] | [Evidência](../evidencias/05_petropolis/petropolis-aprendizado/v1/resultados/contrastes_T1.csv) · [Código](../codigo/scripts/petropolis_aprendizado/pa_05_treino.py) | Mesmo orçamento de 100 consultas. Externo com exemplos locais contra só local. |

## Tabela V — Atualização na RM de Porto Alegre

| Item | Resultado | Registro e código | Leitura |
|---|---|---|---|
| V.01 | Margem (AUC de 0,874 a 0,926)  ·  +0,052  ·  [0,020; 0,101]  ·  6/6 | [Evidência](../evidencias/06_atualizacao/fundamentacao-artigo-20261028/f01_ic_ganho_atualizacao.json) · [Código](../codigo/scripts/fundamentacao_artigo/f01_ic_ganho_atualizacao.py) | 400 consultas nas FCU da RMPA. Seis repetições do mesmo episódio. |
| V.02 | Pontos escolhidos por sorteio  ·  +0,029  ·  [0,009; 0,062]  ·  6/6 | [Evidência](../evidencias/06_atualizacao/fundamentacao-artigo-20261028/f01_ic_ganho_atualizacao.json) · [Código](../codigo/scripts/fundamentacao_artigo/f01_ic_ganho_atualizacao.py) | 400 consultas nas FCU da RMPA. Seis repetições do mesmo episódio. |
| V.03 | Atualização com 400 consultas  ·  +0,021  ·  [0,015; 0,026]  · | [Evidência](../evidencias/06_atualizacao/transferencia-vulnerabilizados/v8_validacao_integrada/analise/contrastes_poucos_rotulos.csv) · [Código](../codigo/scripts/vulnerabilizados/v8_01_q_externo.py) | Coorte inteira da RMPA. F4 com 400 consultas e prior externo. |
| V.04 | Origem após ajuste por margem  ·  -0,028  ·  [-0,056; -0,006]  · | [Evidência](../evidencias/05_petropolis/fundamentacao-artigo-20261028/f04_encosta_e_retencao.json) · [Código](../codigo/scripts/fundamentacao_artigo/f04_encosta_e_retencao.py) | Perda de retenção após ajuste na RMPA, medida em seis unidades externas. |
| V.05 | FCU, memória de 200 exemplos^a  ·  +0,017  ·  [0,006; 0,027]  · | [Evidência](../evidencias/06_atualizacao/transferencia-vulnerabilizados/v10_ciclo_contextual/iii_memoria/pareados_final.csv) · [Código](../codigo/scripts/vulnerabilizados/v10_iii_memoria.py) | DER com memória de 200 exemplos, no ciclo de três destinos. |

## Tabela VI — Cobertura observacional

| Item | Resultado | Registro e código | Leitura |
|---|---|---|---|
| VI.01 | Recorte total (a)  ·  1.708 FCU, 2.776.821 moradores | [Evidência](../evidencias/09_social/peso_das_regioes.json) · [Código](../codigo/scripts/fundamentacao_artigo/f08_ruas_e_comunidades.py) | Universo censitário de 1.666 FCU de água e 42 de Petrópolis. |
| VI.02 | FCU com água  ·  De 1 no acervo inicial a 123 de 1.666 no recorte ampliado | [Evidência](../evidencias/04_recife/fundamentacao-artigo-20261106/f08_ruas_e_comunidades.json) · [Código](../codigo/scripts/fundamentacao_artigo/f08_ruas_e_comunidades.py) | Acervo inicial e contagem no universo ampliado de 1.666 FCU. |
| VI.03 | Suporte nas 123 FCU  ·  119 com as duas classes. Em 38, há ≥5 pontos por classe. | [Evidência](../evidencias/04_recife/fundamentacao-artigo-20261106/f08_ruas_e_comunidades.json) · [Código](../codigo/scripts/fundamentacao_artigo/f08_ruas_e_comunidades.py) | Duas classes e suporte mínimo de cinco pontos de cada classe. |

## Tabela VII — Módulos de apoio

| Item | Resultado | Registro e código | Leitura |
|---|---|---|---|
| VII.01 | Mapa de água, 90 recortes (b)  ·  Sobreposição (IoU) da U-Net: 0,449 | [Evidência](../evidencias/07_percepcao/sar-inundacao/avaliacao_unet_v1_teste.json) · [Código](../codigo/scripts/sar/avaliar_rede.py) | IoU por pixel, sem água permanente, em 90 recortes reservados. |
| VII.02 | Pré-treino, 8 eventos (d)  ·  Pré-treinado - aleatório: ΔAP +0,041 [0,014; 0,069], 7/8, com 1 recorte; +0,050 com 3 | [Evidência](../evidencias/07_percepcao/pretreino-sar/v2/analise.json) · [Código](../codigo/scripts/sar/pretreino_v2.py) | Ganho de AP observado. IC do contraste primário cruza a margem de 0,025. |
| VII.03 | SAR contínuo, 8 unidades  ·  LwF: ΔIoU +0,049 [0,029; 0,068], limiar 0,2 | [Evidência](../evidencias/07_percepcao/planejamento-tese/v2/continuidade/diagnostico.json) · [Código](../codigo/scripts/planejamento/pt_continuidade.py) | Reagregação nas oito unidades comuns, com limiar 0,2. Estimativa integral no artefato. |
| VII.04 | Busca, 11.238 consultas  ·  Nas 5 primeiras: 44,35% híbrida vs. 36,65% lexical | [Evidência](../evidencias/08_resposta/evolucao-escopo/v1/cognicao/resultado.json) · [Código](../codigo/scripts/evolucao_escopo/cognicao.py) | Recall@5 em consultas sintéticas com trecho-alvo conhecido. |
| VII.05 | Juiz + regras vs. regras (18)  ·  Acerto: 0,722 vs. 0,444. Falsos positivos: 1 vs. 7 | [Evidência](../evidencias/08_resposta/verificacao-suporte/medida_de_suporte_com_juiz.json) · [Código](../codigo/scripts/verificacao/medir_suporte.py) | Comparação em 18 casos de sustentação. A combinação reduziu falsos positivos. |
| VII.06 | Água com resposta baixa (c)  ·  Margem vs. sorteio: RMPA 0 vs. 35, RMR 25 vs. 130 | [Evidência](../evidencias/08_resposta/transferencia-vulnerabilizados/v8_validacao_integrada/competencia_tranquilizadora/cobertura_tranquilizadoras.csv) · [Código](../codigo/scripts/vulnerabilizados/v8_06b_competencia_tranquilizadora.py) | Positivos com resposta baixa nas FCU, à meta de cobertura de 70% na coorte. |
| VII.07 | Reprodução do escore  ·  147.798 pontos, erro máximo 3,56×10^{-15} | [Evidência](../evidencias/08_resposta/ia-integrada-v2/fechamento_v1/fisica.json) · [Código](../codigo/scripts/ia/ia_integrada_v2.py) | Soma de pontos do MT11 e da rodada 3. Erro numérico máximo. |
| VII.08 | Conferência do serviço  ·  2.747 testes (20/10), 922 fontes por hash, 5 portões | [Evidência](../evidencias/08_resposta/evolucao-escopo/v1/qa_final/v3/resultado.json) · [Código](../codigo/scripts/servico/resposta_integrada.py) | Conferência da entrega de 20/10. Contagens de testes, hashes e portões do serviço. |

## Leitura conjunta com o pôster

O [pôster](../artigo/REV-P_poster.pdf) resume os mesmos ensaios. Na RMPA, a estimativa de +0,040 AUC sobre HAND nas FCU tem intervalo que inclui zero. A Tabela II do artigo indica essa ausência de vantagem sustentada com um traço. A [coorte FCU](../evidencias/03_transferencia/fundamentacao-artigo-20261028/f03_ic_transferencia_coortes.json) conserva o intervalo.

O pré-treino com um recorte rotulado por contexto ganhou 0,041 AP em média. O intervalo de 0,014 a 0,069 cruza a margem de 0,025. A [demonstração de Muçum](../evidencias/07_percepcao/demonstracao_mucum/reconstrucao_mucum.json) ilustra a reconstrução com o estado salvo. O [ensaio de oito eventos](../evidencias/07_percepcao/pretreino-sar/v2/analise.json) mede a diferença de AP.

O contraste LwF com limiar 0,2 aparece como +0,049 no artigo. O [artefato de reagregação](../evidencias/07_percepcao/planejamento-tese/v2/continuidade/diagnostico.json) conserva a estimativa integral, próxima de +0,04848, e os intervalos. Essa diferença de apresentação fica registrada na conferência.

A triagem da Tabela VII usa a [comparação com meta de 70% na coorte da avaliação integrada](../evidencias/08_resposta/transferencia-vulnerabilizados/v8_validacao_integrada/competencia_tranquilizadora/cobertura_tranquilizadoras.csv). A contagem dentro das FCU depende da coorte e da execução selecionadas.

A perda de 0,078 no ensaio de Manaus sem filtro foi medida na retenção da origem. O [registro do ciclo](../evidencias/06_atualizacao/relatorios/CICLO_CONTEXTUAL_V10_RESULTADOS_2026_10_24.md) identifica a população da perda.

