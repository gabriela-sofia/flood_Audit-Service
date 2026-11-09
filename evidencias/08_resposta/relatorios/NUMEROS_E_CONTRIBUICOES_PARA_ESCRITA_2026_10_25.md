# REV-P — números, capacidades e contribuições para a escrita

Retrato: 25/10/2026, Asia/Chita. Consolidação dirigida dos relatórios e artefatos disponíveis até as campanhas regionais de 24/10. Não é censo de todos os arquivos, novo experimento ou reexecução de toda a pesquisa. Inclui resultados positivos, inconclusivos, contrários, cobertura de dados e implementação. O anexo `ANEXO_CATALOGO_HISTORICO_68_RESULTADOS_2026_10_25.md` preserva os 68 registros de 20/10; as correções e campanhas posteriores abaixo têm prioridade na interpretação correspondente.

## Como usar este documento

- AUC mede ordenação; 0,88 não é 88% de acerto. ΔAUC de 0,05 não significa redução de risco de 5%.
- AP deve acompanhar prevalência e definição do alvo. IoU de segmentação não é intercambiável com AUC.
- Pontos, pixels, sementes, modelos e arquivos não são réplicas territoriais independentes.
- Não somar as amostras das campanhas: elas reutilizam dados. Recife e RMR compartilham o episódio de 2022.
- Cada número conserva tarefa, versão e população. Uma tarefa usa água observada por satélite; outra, ruas com relatos; outra, cicatrizes contra fundo.
- Os experimentos recentes em coortes conhecidas são desenvolvimento. Confirmações históricas ou delimitadas mantêm seu alcance próprio.
- Números de testes abaixo são resultados registrados em datas e suítes distintas. Apenas os 16 testes regionais foram rerodados nesta conversa em 25/10; não somar suítes sobrepostas.

## 1. O que a ferramenta faz

1. Qualifica observações e distingue presença, ausência admitida, fundo e desconhecido.
2. Extrai atributos territoriais e conserva fonte, data, escala, resolução e versão.
3. Aprende funções preditivas a partir das observações qualificadas para cada tarefa.
4. Transfere funções para destinos separados do ajuste e avalia a informação disponível no destino.
5. Consulta exemplos locais sob orçamento, registra informação revelada e testa atualizações.
6. Compara ganho local com manutenção do modelo e mede retenção na origem.
7. Integra terreno, água, encosta, infraestrutura e contexto social, mantendo tarefas distintas.
8. Produz candidatas de resposta territorial com fontes, limites e observação necessária.
9. Executa percepção orbital/neural e recuperação documental, com avaliações próprias.
10. Preserva pesos, previsões, protocolos, manifestos e trilhas de execução.

Isso é ferramenta de pesquisa executada. Não equivale a previsão datada, estimativa de profundidade, risco completo, causalidade identificada ou uso operacional irrestrito.

## 2. Bases e populações — números para métodos

| Número | Significado e versão | Fonte |
|---|---|---|
| 1.641.356 linhas | consolidação descrita na proposta de 06/10; fotografia histórica, não total atual do disco | S01 |
| 763.155 linhas | autorizadas para treino na mesma fotografia; elegibilidade não implica uso em todo treino | S01 |
| 513.530 linhas | autorizadas para afirmação na mesma fotografia; categorias sobrepostas | S01 |
| 91.446 linhas, 130 eventos, 43 unidades | amostra de ajuste reproduzida para Q histórico MT05 na v8 | S02 |
| 36.331 linhas, 59 eventos | ajuste externo v8/v9; CEMS 16.633, UFO 11.698, Sen1Floods11 8.000 | S02 |
| 30.075 linhas, 48 eventos | variante sem América do Sul da v8 | S02 |
| 19.407 linhas, 34 eventos | Q separado por papéis da v8 | S02 |
| 29 + 8 + 6 unidades | na cadeia limpa v10: ajuste físico, competência e retenção; as oito se dividem em quatro para ajuste e quatro para calibração | S04 |

As linhas brasileiras foram retiradas do ajuste externo e a v9 corrigiu a seleção histórica da regularização. Q_sel_ext usa as 43 unidades estrangeiras; Qsep_sel_fis usa as 29 físicas. São funções diferentes. Não chamar as seis unidades de retenção da v9 de inéditas em relação ao Q_sel_ext; a v10 corrige o papel na cadeia completa.

## 3. Conhecimento físico e fórmula — resultados para o núcleo científico

### MT01: relações aprendidas em destinos retidos

- **43 unidades de cinco continentes**, com destino deixado fora do ajuste.
- **33/43 unidades favoráveis** ao modelo G contra HAND.
- **G − HAND: +0,03909 AUC**, IC95% **[0,02567; 0,05380]**.
- **G − regra física de pesos iguais: aproximadamente +0,018**, IC95% **[0,010; 0,026]**.
- **Retirar acumulação/posição topográfica: −0,06357**, IC95% **[−0,08015; −0,04815]**.
- Sinal de HAND negativo em **25/27** ajustes territoriais individuais; magnitude variando aproximadamente **seis vezes**.
- Modificação contextual C−G **−0,00070**; hierárquico H−G **−0,00053**: sem superioridade demonstrada.

**O que escreve:** relações físicas aprendidas contêm informação transferível no desenho avaliado. Não são pesos causais universais. G inclui uma elevação dependente da amostra; não atribuir esses resultados diretamente à função Q de consulta por coordenadas. Eventos já examinados. Fonte: S05.

### MT11 e reparo de independência

- **30 componentes**: conexão−HAND **+0,05503**, IC98,333% **[0,02389; 0,08632]**.
- Mesmos 30: conexão−base **+0,02962**, IC98,333% **[0,00611; 0,05254]**. Positivo, mas limite inferior abaixo de 0,01.
- Histórico incremental: **15 componentes**, **+0,02942**, IC98,333% **[−0,00322; 0,06278]**: inconclusivo.
- Sensibilidade de exutório, conexão−HAND: **19 componentes**, **+0,04988**, IC **[0,00560; 0,09656]**.
- 3B após o grafo comum: **seis componentes**, abaixo dos **oito** exigidos. Não reutilizar o resultado anterior de oito como confirmação intacta.

**O que escreve:** ganho delimitado da conexão e limite do histórico, após reparação retrospectiva. MT11 e 3B não são réplicas somáveis. Fonte: S06 e catálogo.

### Cobertura, solo e ponderação — MT06/MT07

- Auditoria MT06: **510.348 pontos, 130 eventos, 43 componentes**; areia ausente em **64.067 pontos**.
- **57,4%** dessas ausências coincidem com classes de água/área úmida do WorldCover no cruzamento declarado; associação não identifica causa da máscara.
- MT06: **344 ajustes novos**, oito braços por 43 componentes.
- Cobertura conjunta: **+0,0048**, IC98,333% **[−0,0009; 0,0107]**, inconclusivo.
- Construído condicional: **+0,0049 [−0,0008; 0,0108]**, inconclusivo.
- Areia condicional: aproximadamente **−0,0001**, sem ganho sustentado.
- MT07: **43 retreinos** com peso por componente; diferença global **+0,0002**, IC95% **[−0,00229; 0,00301]**: sem ganho sustentado.

Não dizer que toda camada melhora todo lugar, nem atribuir ganho de um bloco conjunto a um atributo isolado. Fontes: S07/S08.

## 4. Transferência física para coortes brasileiras

### População de teste v8 — apenas estados positivo/negativo

| Coorte | Pontos | Positivos | Positivos FCU |
|---|---:|---:|---:|
| RMPA | 135.564 | 6.117 | 241 |
| Manaus | 63.452 | 215 | 48 |
| Curitiba | 198.944 | 883 | 11 |
| Recife município | 42.736 | 448 | 31 |
| RMR | 67.099 | 2.882 | 366 |

Fonte: S02. Não confundir a coorte Manaus com o pacote social am_piloto de Iranduba/Manacapuru.

### Q_sel_ext v9 — seleção de regularização estrangeira, FCU agrupadas

| Coorte | Q externo | HAND | Menos declive |
|---|---:|---:|---:|
| RMPA | 0,88035 | 0,84062 | 0,90995 |
| Manaus | 0,91955 | 0,87340 | 0,80836 |
| Curitiba | 0,89284 | 0,82595 | 0,69755 |
| Recife | 0,86637 | 0,80239 | 0,71534 |
| RMR | 0,80899 | 0,69292 | 0,71568 |

São valores descritivos de ordenação, não cinco confirmações independentes nem acurácia. Curitiba tem 11 positivos FCU. RMPA perde para menos declive no agregado. Fonte: S03 e `p0_regularizacao/auc_coortes.csv`, `b_tabular/auc_coortes.csv`.

### Aprendizado com poucos rótulos — v8

- **1.682 execuções de adaptadores**: **872** ajustados com duas classes, **486** cópias do prior, **324** impossíveis por uma classe. Não chamar todos de ajustes aprendidos.
- Orçamentos **25, 100 e 400 consultas**, seis sementes, três políticas de seleção.
- RMPA, B=25: prior externo−neutro **+0,011 [0,001; 0,025]**; externo−permutado **+0,020 [0,008; 0,037]**.
- RMPA, B=400: atualização−sem adaptação **+0,021 [0,015; 0,026]** na população todas; FCU **+0,021 [0,001; 0,052]**.
- Curitiba, B=400: uma atualização piora em **−0,011 [−0,018; −0,005]**.

Fonte: S02. A v10 abaixo é a referência para o ciclo integrado com núcleo limpo.

## 5. Ciclo de consulta e atualização — v10

- Lotes de **25**, até **400 consultas**, seis sementes, três coortes, pools todas/FCU.
- Filtro: pelo menos **cinco positivos, cinco negativos e três agrupamentos com positivo**. Condição experimental, não garantia universal.
- **Zero consultas em teste**; IDs reproduzidos de forma idêntica nas três coortes.
- RMPA, B=400, pool todas: margem+filtro−congelado **+0,052 AUC FCU**, favorável em **6/6 sementes**; custo de retenção externa **−0,029**.
- Mesmo desenho: margem+filtro−uniforme+filtro **+0,023**, favorável em **6/6**; diferença de retenção **−0,018**.
- Uniforme+filtro−congelado **+0,029 FCU**; retenção **−0,010**.
- Pool FCU RMPA: margem **0,920**, uniforme **0,886**, ganho **+0,034**.
- Manaus e Curitiba: margem com 400 consultas revela **0–1 positivo** na condição relatada; filtro não atualiza.
- Manaus: variante sem filtro perde **0,078 de retenção**; impedir essa atualização preserva capacidade.
- Curitiba: margem_p90 atinge o filtro em **375 consultas** e cai para **0,628 FCU**, retenção **0,605**. Satisfazer o filtro não basta.

**O que escreve:** houve aprendizado local útil e melhor aproveitamento de consultas na RMPA, com custo mensurado; também houve recusas e falhas reais. Não são seis replicações regionais. Fonte: S04.

## 6. Memória, retenção e reconhecimento de limites

### Memória limitada v10

- Memória selecionada **M=200**, α=**0,3**, na borda da grade; seleção fraca.
- **Três ordens × seis sementes**, 400 consultas por etapa.
- DER limitado−congelado, retenção: **−0,022**, IC95% **[−0,028; −0,016]**.
- Ganhos FCU finais associados: RMPA **+0,017**, Manaus **+0,015**, Curitiba **+0,020**; não transferir automaticamente o intervalo da retenção para esses ganhos.
- Replay−congelado: retenção **−0,023 [−0,032; −0,016]**.
- DER−replay: **+0,002 [−0,001; 0,005]**, sem superioridade clara.
- Referência conjunta: retenção **0,707** contra **0,733** inicial.

**O que escreve:** adaptação e memória foram executadas; preservar integralmente capacidade anterior não foi demonstrado. Os ganhos de estabilidade da v9 tinham acesso histórico que foi corrigido na v10. Fonte: S04.

### Competência e encaminhamento v10

- Ensemble de **cinco MLP**; competência em quatro unidades, calibração em quatro, avaliação externa em seis.
- A 70% de cobertura, positivos erroneamente classificados como menor na retenção: **1.038 margem; 1.108 aprendido; 1.266 discordância; 1.741 aleatório**.
- Na RMR: **283 aprendido; 313 margem; 881 aleatório**.
- CRC: α=**0,02**, λ=**0,089**, perda empírica de **0,009** na retenção com cobertura de aproximadamente **31%**.
- **Zero revisões concluídas**.

**O que escreve:** encaminhamento pode reduzir respostas baixas erradas, mas a competência aprendida não tem superioridade geral sobre margem. Não há garantia conformal de transporte entre esses domínios. Recusa por dado ausente é outra capacidade. Fonte: S04.

## 7. Recife — ruas, alagamento registrado e ferramenta urbana

### Dados e teste

- **1.648 relatos SEDEC**, 2014–2024; **420 geocodificados até a rua**.
- Geocodificação: **220** nomes exatos, **101** sem tipo, **84** por contenção única, **15** aproximados.
- **5.477 ruas**, **96** com relato, **38 bairros**.
- Teste: **19 bairros, 2.225 ruas, 48 positivas**.
- PE3D: **2.564 ruas, 38 positivas** no suporte declarado.
- **350 ajustes** e **2.000 reamostragens bootstrap**; não são novas observações.

### Resultados dentro do bairro

| Modelo | AUC | AP | Relatadas no quintil superior |
|---|---:|---:|---:|
| HAND | 0,595 | 0,028 | 31% |
| Q externo | 0,669 | 0,038 | 42% |
| Só local Q, B=48 | 0,684 | 0,041 | 35% |
| Q atualizado, B=48 | 0,679 | 0,040 | 40% |
| Prior permutado | 0,633 | 0,035 | 33% |

- Transferência Q−HAND **+0,07355**, IC95% **[0,01153; 0,15908]**, sustentada pela regra.
- Incremento da atualização aproximadamente **+0,010**, inconclusivo.
- Retenção externa: **0,733 antes; 0,735–0,737 depois**.
- AUC aparente com drenagem/saída: **0,756**; com controle por comprimento, incremento de drenagem/saída **+0,014**, inconclusivo.
- PE3D: ganho aparente protocolar **+0,062**; após controle por comprimento **−0,031**, contrário em amostra pequena. Não apresentar 0,796 do braço PE3D como benefício físico independente demonstrado.
- Coeficientes padronizados A4: HAND **−0,82**, distância ao canal **−0,56**, construído **−0,32**; são coeficientes do estimador, não pesos causais.

**Natureza:** ruas com relato contra fundo sem relato; não há negativos físicos observados. 2022 não tem endereço utilizável nessa campanha. Fonte: S09.

### Infraestrutura existente e efetivamente trabalhada

- **175.630 registros** de elementos de microdrenagem: BL **15.883**, CG **44.163**, PV **16.791**, CAN **61.791**, FC **37.002**. Nem todos são bueiros; as categorias são cadastrais.
- **74.379 registros de tubulação**, **11.382 de canaletas**.
- Macrodrenagem: **99 canais, 12 rios/riachos, cinco bacias**; representações relacionadas, não casos independentes.
- **Oito folhas PE3D de 1 m**; extração da campanha a **2 m**; núcleo físico a **30 m**.
- Molduras PE3D: **65,609 km²**; **129 FCU** intersectam; **111** com pelo menos 95% da área nas molduras.
- COTA_FUNDO: **4.025/175.630** preenchidos, com anomalias. Cadastro não permite presumir capacidade hidráulica ou funcionamento.

### Saída por comunidade e casos

- **295 FCU** na candidata; **228 cobertas**, **74** com rua relatada, **33** com resultado fora do ajuste.
- Universo censitário: **361.548 moradores** nas 295; **295.336** nas cobertas; **50.198** nas 33 com resultado fora do ajuste. Não são beneficiados ou atingidos medidos.
- Chico Mendes–Linha Nova: **7.786 moradores**, **256 BL/CG**, uma rua com **14 relatos**; bairro de ajuste, leitura descritiva.
- Encanta Moça: **8.878 moradores**, **151 BL/CG**; profundidade topográfica de depressão até **6,8 m** não é lâmina de inundação.
- Estrada dos Remédios–Rozenblit I: **1.487 moradores**, **136 BL/CG**.
- **Sete figuras** e candidata com terreno, captação, rede, saída, observação, inferência, contexto social, fontes e lacunas. Sem promoção do serviço ativo.

Fontes: S09/S10 e `territorio/cobertura_295.json`.

## 8. Petrópolis — encostas, transferência e integração territorial

### Dados, atributos e aprendizado na origem

- **42 FCU**, universo censitário **43.019 moradores**.
- Idai: **14.900 pontos de fonte** e **36.026 polígonos** do inventário adquirido. Não somar pontos/polígonos como casos independentes; ausência uniforme não demonstrada pela envoltória.
- UFRJ/GEOCART: **1.592 registros**, **1.401 de deslizamento**, 1933–2015; endereço de dano não é localização confirmada da ruptura.
- INPE/Charter 2022: **85 polígonos = 81 deslizamentos + três fluxos de detritos + uma inundação**; inundação sem interseção FCU nesse produto.
- **Sete atributos** FABDEM a **30 m**: dois declives, duas curvaturas, área contribuinte específica, amplitude em **210 m** e posição topográfica em **330 m**.
- Idai retido: **13 blocos de 10 km**; AUC logística **0,885**, boosting **0,897**, declive **0,877**; inventário contra fundo até comprovação de ausência.
- Coeficientes padronizados: declives **+1,32/+0,86**, amplitude **−0,78**, posição topográfica **−0,30**. Não são pesos causais.

### Transferência e atualização

- Teste: **81 cicatrizes**, **12.170 células de fundo**, suporte de **11,9 km²**; prevalência celular **2,2%** no desenho.
- Logística externa: AUC por cicatriz **0,66048**, IC95% **[0,587; 0,726]**; AP **0,052**; **40%** das cicatrizes no quintil superior.
- Declive: **0,65390 [0,577; 0,722]**; AP **0,049**; **38%** no quintil superior.
- Boosting: **0,61969**; AP **0,033**; **21%** no quintil superior.
- Externo−declive **+0,00658 [−0,01861; 0,03168]**, inconclusivo. Não afirmar equivalência.
- Retirar declive no destino custa aproximadamente **0,042 AUC**; retirar relevo, **0,016**.
- **350 ajustes locais**, B=**10 a 402 sítios**, e **2.000 reamostragens**.
- Atualização histórica, B=100: **−0,158 [−0,247; −0,065]**, piora.
- Externo+local−controle sem coeficientes externos: aproximadamente **+0,126**; real−permutado **+0,071** no contraste relatado. Estrutura externa limita deterioração; não é ganho sobre o congelado.
- Emenda de cinco bairros: externo **0,668**, só local **0,608**, combinação **0,648**, prior permutado **0,545**.
- Combinação−externo **−0,020 [−0,039; −0,003]**. Retenção Idai **0,883**; dependência espacial residual, mesmo episódio já examinado.
- Escore de água nas cicatrizes: AUC aproximadamente **0,37**, apenas **4%** no quintil superior. Baixo escore de água não significa segurança de encosta.

### Informação territorial — critérios corrigidos pela auditoria

- **Cinco FCU** intersectam o inventário cartográfico.
- **18 FCU** têm geocódigo histórico interno.
- União inventário/geocódigo interno: **20 FCU**.
- Ao incluir relatos a até **100 m**: **31 FCU**; **11** entram somente pela proximidade. Não chamar 31 de ocorrência física interna confirmada.
- **36 FCU** com contexto técnico 2025, posterior ao evento.
- **37 FCU** com pelo menos 80% das células nas faixas marginais do doador: diagnóstico de entradas, não validação de aplicabilidade.
- **33 FCU** com conexão de montante até 1 km a encosta no quintil superior.
- Candidata com **sete componentes** separados; **seis figuras**.
- Morro da Oficina–Rua dos Ferroviários: **1.025 moradores**, quatro cicatrizes e dois fluxos; **16.879 m²** de interseção declarada; não representa população atingida.
- Vila São Francisco A: **548 moradores**, cinco cicatrizes e **15.666 m²**; Idai recupera três de cinco no quintil superior, declive uma de cinco.
- Sargento Boening: **584 moradores**, duas cicatrizes e **13 relatos internos**; **38%** das células com algum atributo fora da faixa do doador.

Fontes: S11/S12. A estatística `renda_resp_media_ponderada` da candidata tem problema semântico apontado na auditoria: média ponderada de medianas não é renda média/per capita/mediana da FCU. Não usá-la assim na escrita.

## 9. Observabilidade e dimensão social

### V10: possibilidade de avaliação individual por comunidade

| Coorte | Comunidades na população | Com pelo menos 5/5 | Parcela dos pontos FCU nessas comunidades |
|---|---:|---:|---:|
| RMPA | 293 | 27 | 19,0% |
| Manaus | 236 | 5 | 3,4% |
| Curitiba | 410 | 0 | 0% |
| Recife | 294 | 0 | 0% |
| RMR | 433 | 6 | 5,6% |

Esse critério pertence à referência orbital; não contradiz as 33 FCU com resultados por rua de Recife. As 294 FCU da coorte orbital e as 295 da candidata municipal são populações diferentes.

- Recife/2022: **147 relatos**, **35 bairros**, **16 bairros com relato sem positivo S2**. Correlação de contagens **−0,10 (p=0,32)**; taxa por pontos **−0,13 (p=0,21)**. Não identifica qual fonte está correta.
- Curitiba/2023: **39 chamados** na janela; **um** em FCU, associado a 15 negativos e zero positivos da referência.
- PPS RMPA: **480 pontos, 120 setores**, com **50 setores FCU, 35 baixa renda, 35 comparador**, quatro pontos por setor. **Zero respostas humanas** no pacote relatado.
- PPS, estimativa da parcela sem referência pelo estado S2: baixa renda **0,31 (EP 0,08)**, comparador **0,23 (0,07)**, FCU **0,075 (0,036)**. Não são resultados da adjudicação humana.

Fonte: S04/S02.

### Medidas sociais e exposição

- **Três objetos distintos:** exposição, componentes de vulnerabilidade, cobertura científica.
- **9.186 setores:** Curitiba **3.190**, RS **5.924**, am_piloto **72**.
- **Sete componentes:** crianças, idosos e cinco indicadores de água/saneamento, cada um com seu denominador. Renda e cobertura explicitadas; não um índice sintético validado.
- Exposição condicional na grade censitária: população **[1.131; 16.068]**; domicílios ocupados particulares/coletivos **[445; 6.418]**.
- Sensibilidade de 30 m: erosão **[880; 11.949]** pessoas; dilatação **[2.161; 17.671]**. São limites geométricos, não IC ou vítimas.
- Piloto aberto: **400 pontos**, **191 anotáveis**, **209 não sei**: **160 nuvem, 28 sombra, 21 mistura**; nenhuma referência admitida nessa frente.
- Produtos posteriores RS: **15 produtos datados em seis dias**, cruzados com os **5.924 setores**, mantendo datas separadas.

Fontes: S13/S06. am_piloto pertence a Iranduba/Manacapuru, não automaticamente ao município de Manaus.

## 10. Percepção, representação e aprendizado conjunto

- V9, patches: **36.331 externos** e **110.545 das coortes**, dimensões **32×32×5**.
- **Cinco famílias × quatro configurações**, com três sementes nos finalistas; as famílias são alternativas, não componentes todos servidos ao mesmo tempo.
- Sem Q, embedding pré-treinado: AUC RMPA-todas **0,904**, Manaus **0,836**; aleatório **0,835/0,723**. Não universal: em outros destinos o aleatório pode vencer.
- Como resíduo sobre Q, famílias pioram RMPA aproximadamente **−0,015 a −0,038**; informação aprendida não demonstrou ganho incremental consistente.
- Observação multimodal: **431 chips Sen1Floods11 a 30 m**; oito braços, três sementes.
- Fusão com radar: AUC **0,927–0,952**, IoU **0,50–0,55**; só radar corrigido: AUC **0,930–0,942**, IoU **0,57–0,59**.
- Radar ausente: fusão AUC **0,81–0,87**; só terreno **0,85–0,87**. Radar coevento é observação, não previsão antecedente.
- Conjunta: AUC observação com/sem radar **0,953/0,842**; controle de pareamento rompido mantém desempenho de suscetibilidade semelhante. Benefício específico de compartilhamento não identificado; não atribuir uma causa única ao empate.
- Campanha JRC anterior: conjunto−parada **+0,00326**, IC97,5% **[−0,00062; 0,00774]**, 43 destinos; sem ganho relevante demonstrado de 0,01.
- Continuidade SAR, oito unidades: LwF−continuação **+0,05870 IoU [0,03403; 0,08337]** na v2.1 e **+0,04848 [0,02882; 0,06815]** na v3.1, limiar 0,2. Calibração por modelo reduz o contraste v3.1 para **+0,00892 [−0,00496; 0,02280]**, inconclusivo.
- Sonda histórica: **24 corridas**, quatro braços × seis sementes; **83 chips** em suporte comum, **72 com água**. Ganho de amostragem **+0,00648 IoU** com vazio=0; a conclusão muda com vazio=1. Não escolher apenas a convenção favorável.

Fontes: S03/S06. Para a reanálise mais limpa da atualização/retensão, usar v10. DINOv2 permanece em sua finalidade de revisão/representação, sem inferir validação pelo tamanho do embedding.

## 11. Sensores, chuva e limites causais

- FloodNet: **14 arquivos mensais disponíveis dos 15 previstos**, maio–setembro de 2021–2023.
- **5.800 sensor-dias, 41 sensores**: **1.848 sem lâmina instrumental, 138 com lâmina, 3.028 não observados, 786 desconhecidos**.
- Critério de lâmina: pelo menos **25 mm** em duas leituras consecutivas a cinco minutos. Não comprova continuidade entre as leituras.
- Após juntar chuva/defasagens: **1.906 linhas completas, 29 sensores**.
- **Dez meses informativos**, mínimo protocolar **12**; reparo local tinha teto **11** na avaliação de 20/10. Efeito primário não testável; não fabricar estimativa causal.
- Verificador distinto reproduziu **99 sensor-dias**.
- Ponte fluvial ANA–D8–S2: **quatro cenas, um episódio/componente**, sem identificação do mecanismo causal.

Fonte: S14/S06. Não transformar associação com notificação em observação física, profundidade no pixel ou efeito de uma intervenção.

## 12. Memória documental, consulta e engenharia científica

- Corpus avaliado: **3.813 chunks**, **11.238 consultas sintéticas**.
- Recall@5: lexical **36,65%**, LSA **39,05%**, híbrido **44,35%**. Consultas derivadas do corpus, sem avaliação independente de compreensão.
- Verificador lexical adversarial: **8/18 corretos**, **sete falsos positivos**, **três falsos negativos**. Não declarar sustentação automática de toda prosa.
- Registro estruturado confere identidade/hash e resultado exato; tem alcance distinto de verificação semântica.
- Integração v2: Q reproduzido em **65.549 pontos MT11 + 82.249 pontos rodada 3**, erro máximo **3,56×10⁻¹⁵**; histórico em **28.875**, erro **1,34×10⁻¹⁵**.
- Consulta sem rótulo em **200 pontos conhecidos de duas AOIs**: erro de escore até **2,67×10⁻¹⁵**. Paridade em pontos conhecidos, não validação de um novo território.
- Percepção empacotada: **222 chips**, 84 de retenção e 138 de avaliação; paridade com o pai, não ganho novo.
- Conferência regional: **82.249 pontos**, **73.616** com igualdade exata na região válida original, **8.563** passam à cobertura real, **61** NoData e **nove** sem região.
- Entrega local de 20/10: **2.747 testes aprovados, 81 dispensados, zero falhas**; **922 fontes por hash**, **24 sondas**, **183 verificações** na reprodução registrada. Não instalação limpa nem reprocessamento integral de rasters brutos.
- V8: **85 testes** registrados; v9: **53** registrados; v10: **19 testes v9/v10** registrados. Suítes sobrepostas: não somar.
- Campanhas regionais: **sete testes Recife + nove Petrópolis = 16**; rerodados em **25/10, 10,64 s**, sem dispensas.
- Conferência atual regional: **130 arquivos Recife + 453 Petrópolis**, zero divergências; métricas principais recalculadas.

Fontes: S06/S14/S15 e verificações regionais. A implementação oferece consulta por coordenadas nas áreas disponíveis, integridade, estados de maturidade, respostas integradas, formulários de revisão e transporte local. Não transforma todos os candidatos de pesquisa em modelos ativos promovidos.

## 13. Resultados que devem aparecer junto dos positivos

1. Transferência existe no desenho avaliado; vantagem sobre a melhor referência simples muda por território.
2. Atualização local melhora a RMPA em condições específicas; Recife ficou inconclusivo; Petrópolis teve deterioração nas formulações testadas.
3. Memória limitada não eliminou esquecimento.
4. Seleção contextual v10 melhora pequenas margens nas seis unidades externas, mas piora FCU em três coortes brasileiras; não é regra geral de uso.
5. Competência aprendida não supera claramente margem e não recebe garantia conformal de transporte.
6. Drenagem/PE3D de Recife são implementações reais; ganho independente não demonstrado após controle de comprimento.
7. Fusão, redes e treino conjunto são experimentos reais; não há benefício multimodal compartilhado universal demonstrado.
8. Água baixa não é segurança contra encosta; presença de relato não é mapa completo do fenômeno.
9. Diagnóstico de lacuna é contribuição atual; redução de danos e redução efetiva da lacuna social não foram demonstradas.

## 14. Ordem sugerida para a escrita

1. **Problema:** escassez e heterogeneidade das observações territoriais.
2. **Método:** qualificação → atributos → aprendizado externo → transferência → consulta/atualização → avaliação/retensão → resposta rastreável.
3. **Resultado principal:** conhecimento físico transferível (MT01/MT11 e destinos brasileiros, com suas linhagens).
4. **Demonstração de aprendizado:** ciclo RMPA v10, ganho e custo da atualização.
5. **Casos territoriais:** Recife por ruas; Petrópolis por encostas e integração entre processos.
6. **Informação social:** cobertura, comunidades, população censitária, observabilidade e exposição condicional.
7. **Experimentos de suporte:** representação, SAR, memória e recuperação documental, sem fundir tarefas.
8. **Limites:** referências, critérios de suporte, resultados contrários, retenção e confirmação independente.

Não existe um único total legítimo de “pontos”, “modelos treinados”, “pessoas beneficiadas” ou “territórios validados” somando todas essas tabelas. Para um total global seria necessário inventário deduplicado com definição de unidade e papel. Os números acima devem permanecer por campanha.

## Fontes do documento

- S01: `docs/PROPOSTA_CONCEITUAL_ATUAL_REV_P_2026_10_06.md`.
- S02: `docs/VALIDACAO_INTEGRADA_V8_RESULTADOS_2026_10_22.md`.
- S03: `docs/EVOLUCAO_ABRANGENTE_V9_RESULTADOS_2026_10_23.md`; usar correções de `docs/REVISAO_V9_DIRECAO_DA_TESE_2026_10_24.md` e S04.
- S04: `docs/CICLO_CONTEXTUAL_V10_RESULTADOS_2026_10_24.md`.
- S05: `docs/MECANISMOS_TRANSFERIVEIS_EXECUCAO_MT01_2026_10_10.md` e `local_runs/mecanismos-transferiveis/v1/resultado.json`.
- S06: `docs/RESULTADOS_PARA_ESCRITA_FINAL_ESCOPO_LOCAL_2026_10_20.md`, `docs/TEXTO_BASE_RESULTADOS_CONCLUSAO_ESCOPO_LOCAL_2026_10_20.md` e registro JSON de resultados de 20/10.
- S07: `docs/MT06_COBERTURA_SOLO_RESULTADOS_2026_10_17.md`.
- S08: `docs/MT07_BALANCEAMENTO_COMPONENTES_2026_10_17.md`.
- S09: `docs/RECIFE_APRENDIZADO_ALAGAMENTO_RESULTADOS_2026_10_24.md` e `local_runs/recife-aprendizado/v1/`.
- S10: `docs/RECIFE_ALAGAMENTO_URBANO_DRENAGEM_PE3D_2026_10_24.md`.
- S11: `docs/PETROPOLIS_APRENDIZADO_ENCOSTAS_RESULTADOS_2026_10_24.md` e `local_runs/petropolis-aprendizado/v1/`.
- S12: `docs/AUDITORIA_ENTREGA_PETROPOLIS_2026_10_24.md`, cujas correções interpretativas prevalecem sobre S11.
- S13: `docs/CONSOLIDACAO_SOCIAL_TRES_MEDIDAS_2026_10_12.md`.
- S14: `docs/FECHAMENTO_IA_INTEGRADA_V2_2026_10_18.md`.
- S15: `docs/VALIDACAO_CONTRIBUICOES_RECIFE_PETROPOLIS_2026_10_25.md` e `local_runs/validacao-contribuicoes-regioes-20261025/verificacao.json`.

Os caminhos relativos são fontes dentro do repositório REV-P. As leituras desta consolidação não abrem reservas e não substituem uma revisão bibliográfica de novidade científica.
