# O avanço da v9 e os próximos experimentos que a tese precisa

24/10/2026. Revisão do retorno anexado, do relatório v9, dos artefatos selecionados e do código executável. Não houve novo treino, abertura de chaves cegas, alteração da v9 ou promoção de modelo.

## 1. O que ficou mais forte

A campanha ampliou de fato a programação e o treinamento. Os resultados favorecem uma arquitetura de pesquisa composta por **conhecimento físico transferido + aquisição informativa + atualização condicionada ao suporte + indicação explícita de limites**. Isso conecta as partes da tese sem exigir que uma rede maior supere todos os controles.

Os avanços mais aproveitáveis são:

- **Q com seleção estrangeira:** a regularização passou a ser escolhida em unidades estrangeiras. Os valores de AUC mantiveram os ganhos de desenvolvimento com alterações pequenas em relação à receita histórica.
- **Continuidade da função:** o offset permite conservar o Q completo quando a atualização é recusada. A paridade relatada é uma propriedade útil e verificável de implementação; o benefício da regra de suporte tem avaliação de desenvolvimento.
- **Consulta informativa:** nas FCU da RMPA, a margem produziu ganho sobre a consulta uniforme em todas as seis sementes examinadas, com média de **+0,04033 AUC**, mínimo **+0,02142** e máximo **+0,06013**. São repetições de um mesmo episódio, não seis confirmações territoriais.
- **Estabilidade:** a variante de replay/distilação reduziu a alteração negativa na população medida. O significado de “externa” dessa população precisa da correção descrita abaixo.
- **Lacuna observacional:** a falta de referência tem denominadores territoriais explícitos. A observabilidade de 61% das FCU e 44% da baixa renda da RMR, registrada pela v9, já é um diagnóstico social/observacional relevante; ainda não é demonstração de que o método reduziu essa lacuna.

Não é necessário descartar as redes ou o pré-treino porque não superaram Q. Seus ensaios ajudam a localizar informação redundante, suporte insuficiente e limites do desenho. Uma configuração sem ganho não refuta a família inteira de métodos.

## 2. O que esta revisão conferiu

Linhagem nova: `local_runs/revisao-v9-20261024/v1/`.

- Estado da fila: **15 etapas marcadas como concluídas**. Isso foi lido sem relançar a fila.
- **79 artefatos selecionados** conferidos contra os hashes do manifesto v9, sem divergências. O manifesto lista 658 arquivos; os 658 não foram todos rechecados nesta revisão.
- **20 AUC** da P0 recalculadas a partir das previsões individuais, com `sklearn.metrics.roc_auc_score`: diferença máxima para a tabela de **2,22 × 10⁻¹⁶**.
- Contraste margem − uniforme recontado por semente a partir das curvas individuais de aquisição.
- Estabilidade recontada a partir do resumo por ordem/estratégia, mantendo a identificação de conjunto previamente visto.
- Grafo: 43 unidades, sendo 29 físicas, oito de competência e seis de retenção. A seleção `Q_sel_ext` usa as 43; `Qsep_sel_fis` usa 29.
- Código corrente v9 registrado por hash para fundamentar os achados de implementação.

Os 53 testes e a reprodução dos pesos são resultados da execução anterior, presentes no log e no verificador; não foram rerodados nesta revisão. O recálculo acima verifica métricas das previsões, não substitui nova inferência integral dos checkpoints.

## 3. Controles que mudam a interpretação

### 3.1 Retenção e competência ainda recebem conhecimento do conjunto reservado ao papel

`v9_g_continua.py` carrega **Q_sel_ext** e mede a “retenção externa” em seis unidades que fazem parte das 43 usadas para ajustar esse Q. As linhas de retenção vêm da mesma amostra externa carregada pelo treino. Assim, o número mede estabilidade em exemplos previamente vistos pelo núcleo; não mede retenção em unidades totalmente inéditas.

Isso preserva uma informação válida sobre estabilidade, mas a frase “seis unidades que nunca ajustaram nada” não descreve o pipeline completo.

`v9_h_incerteza.py` também usa **Q_sel_ext** como base. Embora os cinco resíduos neurais sejam ajustados só no papel físico, o Q e a configuração finalista escolhida na Frente B recebem dados dos demais papéis. Restringir o último ajuste não isola toda a cadeia.

**Correção:** usar Qsep_sel_fis ou outro núcleo ajustado e selecionado exclusivamente no papel físico. Fazer seleção de arquitetura/escala dentro desse papel, separar ajuste do estimador de competência e calibração, e só então avaliar retenção. Como essas unidades já foram examinadas, o novo ensaio continua desenvolvimento com dependências corrigidas; não as torna uma reserva inédita retroativamente.

### 3.2 A ablação regional retira dados apenas da correção neural

Na Frente E, Q e o pré-processamento permanecem ajustados com todas as regiões. O que muda é o conjunto usado para treinar a rede residual. O próprio código declara que o Q da avaliação leave-region-out já viu a região retida.

Portanto, a melhora sem África/Oriente Médio sugere um efeito da composição do **ajuste da correção residual sobre um Q global**. Ainda não estima o efeito de excluir aquela região de todo o aprendizado.

O controle remove o mesmo número de unidades, não necessariamente o mesmo número de linhas, classes, instrumentos ou mecanismos. A região é aproximada pela coordenada mediana da unidade, não uma caracterização hidroclimática completa.

**Correção:** repetir a remoção em Q, seleção de hiperparâmetros, padronização, representação e correção; comparar com controles de quantidade/composição. Separar isso do ensaio legítimo em que o Q global fica fixo e apenas o resíduo varia. Uma semente escolhida após resultados brasileiros não sustenta uma regra geral de excluir continentes.

### 3.3 A ponderação de qualidade não usou o novo indicador JRC

A Frente E lê `evidencia_agua_anterior_jrc.parquet` **da v8**, campo `jrc_ocorrencia`. Não utiliza a `recurrence` qualificada pela frente A da v9.

Logo, o resultado “qualidade empata com ERM” vale para a ponderação implementada com aquele proxy antigo. Não demonstra que o novo indicador foi testado ou que qualificação de referência é inútil.

**Correção:** registrar a banda e a versão efetivas e comparar occurrence, recurrence e, quando disponível, histórico anterior ao evento, com as mesmas amostras e controles. Os 37% de positivos UFO sobre alta recurrence não são uma nova medida do mesmo estimando de 12,4% da occurrence: banda e denominadores diferem. Nenhum dos dois números adjudica individualmente água pré-evento; 1984–2021 inclui anos posteriores a muitos eventos.

### 3.4 O replay é uma variante de distilação, com memória reconstruída

O script mantém todo o histórico acumulado para reconstruir a memória de 200 exemplos, e recalcula os logits dos exemplos antigos com o modelo corrente quando monta o novo buffer. Isso difere de conservar logits associados ao momento de inserção em uma memória limitada.

**Correção:** comparar explicitamente a variante atual com buffer limitado, logits preservados na inserção e replay de rótulos sem distilação, sob os mesmos custos. A referência conjunta pode usar histórico completo, mas esse armazenamento não deve ficar acessível ao aprendiz de memória limitada. O ganho observado não deve ser atribuído exclusivamente ao mecanismo canônico DER++ sem esse isolamento.

### 3.5 O controle conformal é diagnóstico heurístico

O estimador de risco é ajustado nos rótulos de competência e o limiar é escolhido nos mesmos rótulos. O critério usa a proporção de positivos entre os aceitos; é necessário verificar sua adequação à perda/hipóteses do procedimento conformal, não apenas aplicar uma correção finita sobre a taxa.

**Correção:** separar treino do risco, calibração e avaliação por dependências; especificar uma perda admissível e uma implementação de referência antes de enunciar garantia. A evidência existente descreve um comportamento conservador da heurística. Não identifica sozinha a mudança de prevalência como causa nem demonstra controle formal.

## 4. Interpretações que precisam permanecer delimitadas

- A política chamada **margem** na aquisição usa distância ao percentil 80 do escore entre candidatos livres, atualizado a cada lote. Não é automaticamente incerteza probabilística calibrada. Deve ser reproduzida com essa definição antes de variantes.
- O aumento de positivos com alto escore e a piora de teste são compatíveis com seleção pouco representativa; o experimento não isolou o viés de amostragem como única causa.
- “Pareamento rompido iguala conjunta” significa que o benefício específico da correspondência entre tarefas **não foi identificado**. Não prova sozinho que a causa é regularização ou mais passos. Esses efeitos pedem controles pareados adicionais.
- Não superar Q com intervalo não demonstra equivalência, superioridade universal do Q ou inutilidade das redes. Também não significa que todo ganho pontual de uma rede está confirmado.
- Curitiba não tem positivo **na referência do conjunto FCU de adaptação atual**. Isso não demonstra ausência de inundação na cidade/comunidade.
- A consulta ativa salva curvas, mas não persiste a sequência de IDs consultados e os offsets por lote. Essa trilha precisa existir para auditoria completa e reprodução sem rerodar toda a simulação.

## 5. Direção para a próxima campanha

### Frente I — transferência contextual com controles completos

Testar exclusão/seleção de doadores no Q completo e no resíduo, com distinção de hipóteses. Estudar mecanismos, fonte, qualidade, terreno, uso do solo e observabilidade como explicações da transferência negativa. Usar seleção exclusivamente externa; os territórios brasileiros já examinados permanecem desenvolvimento. Aumentar sementes mede estabilidade computacional, não cria replicação regional.

### Frente II — aquisição, atualização e estabilidade como um único ciclo

Integrar margem de aquisição, teste de suporte e offset completo. Comparar com consulta uniforme e com o gate isolado, sob o mesmo B total. Medir ganho de novo conhecimento, custo de desconhecidos e regressão após cada lote. Persistir IDs, escores disponíveis antes da consulta, probabilidades quando aplicáveis, rótulos revelados, modelo e motivo de atualização/recusa.

Reavaliar retenção com núcleo limpo e memória limitada. Esse ciclo é uma contribuição mais concreta para o objetivo de crescimento do que comparar apenas arquiteturas finais.

### Frente III — observabilidade e lacuna territorial como desfechos explícitos

Mapear quais comunidades e contextos estão fora da referência e do suporte. Usar geocodificação brasileira, chamados e fontes IBGE do acervo para triangulação na escala de localização realmente disponível. Um chamado qualificado pode sustentar presença observada; ausência de chamado continua desconhecida. Identificar sobreposição de chamadas/episódios e impedir circularidade se alguma fonte já ensinou o modelo.

Avaliar ganho apenas onde houver referência independente da intervenção e declarar a parcela ainda não identificada. O pacote cego aguarda revisão real; falta de revisor não deve bloquear programação, qualificação e experimentos nas partes testáveis. Nenhuma resposta humana será simulada.

### Frente IV — representações e treino conjunto sob diagnóstico

Prosseguir com patches/multimodalidade quando houver pergunta específica: ganho marginal sobre Q, observação sob modalidade ausente ou melhoria em contexto retido. Remover componentes de avaliação do pré-treino para alegações indutivas; manter a versão transdutiva como protocolo separado. Usar controles pareados de passos, regularização e correspondência entre tarefas.

O conhecimento útil pode estar na função de observação/qualificação, mesmo sem melhorar o escore físico. Isso precisa ser mostrado como resultado próprio, sem somar métricas de tarefas distintas como se fossem validação de uma IA única.

## 6. Conclusão admissível nesta etapa

A v9 sustenta avanços de desenvolvimento em transferência física, continuidade da função e eficiência de consulta na RMPA. A estabilidade por distilação é um sinal útil no conjunto visto. Generalização regional, retenção inédita e competência formal ainda exigem os controles de cadeia descritos acima. A lacuna de observação já foi quantificada em recortes; sua redução permanece sem demonstração.

O próximo trabalho deve **ampliar os contrastes que explicam os ganhos e conectar aquisição, atualização e limites em execução rastreável**, com as frentes computacionais viáveis avançando enquanto a confirmação independente é preparada.
