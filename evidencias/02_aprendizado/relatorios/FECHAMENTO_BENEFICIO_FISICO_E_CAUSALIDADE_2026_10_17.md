# Fechamento: benefício físico do aprendizado conjunto e explicação causal

Data: 17/10/2026. Experimentos MT08 (teto de informação) e MT09 (identificação causal), nos mesmos 43 componentes de desenvolvimento do MT05–MT07. Desenvolvimento retrospectivo: nenhuma reserva aberta, nenhum modelo promovido, nenhum módulo congelado alterado.

> **Correção de alcance (17/10/2026, após a [revisão da coordenação](REVISAO_COORDENACAO_MT11_2026_10_17.md)).** Os números deste documento não mudam; quatro frases foram substituídas porque excediam o que os resultados demonstram:
> - "não sobra relação compartilhável" e "compartilhável entre lugares: esgotado" → conclusões sobre as alternativas **efetivamente testadas**. Uma campanha finita de modelos não mede o teto de qualquer função possível das entradas, e o IC de T_gbm, [−0,0195; +0,0118], não exclui +0,01.
> - "a única via que restou" → "a alternativa testada que mostrou sinal".
> - "explicação causal: fechada" → evidência observacional convergente, sob pressupostos de identificação explícitos. Controles por evento e proximidade e a triangulação fortalecem a associação coerente com mecanismos, mas não eliminam por si o confundimento local nem identificam um efeito causal.
> - "frente fechada" → "ganho relevante excluído nas variantes e populações em que os intervalos o excluem".

As duas perguntas estavam abertas por motivos diferentes. A do aprendizado conjunto tinha cinco variantes nulas, mas nenhuma resposta sobre *por que* uma sexta seria diferente. A causal estava marcada `not_identified` no grafo de 16/10, por um motivo que vale para quantidades marginais e não para a comparação dentro do evento. Os dois desenhos abaixo atacam essas razões em vez de testar mais uma variante.

![Variantes conjuntas, teto das entradas e mecanismos por instrumento](../local_runs/mt08-mt09/fechamento_mt08_mt09.png)

## Resposta curta

1. **Benefício físico do aprendizado conjunto entre lugares: ganho relevante excluído nas variantes testadas.** As cinco variantes conjuntas e a mudança de objetivo (MT07) ficaram entre +0,0002 e +0,0032, com limite superior abaixo de +0,01. Entre as alternativas testadas no MT08, a logística global que viu os rótulos do próprio destino ganha +0,0001 [−0,0006; +0,0009]. As árvores treinadas só nos outros lugares dão −0,003, com IC [−0,0195; +0,0118] que **não exclui** +0,01. Isso limita as variantes e famílias de função testadas; não prova que nenhuma outra função das mesmas entradas aprenderia mais.
2. **A folga que existe é específica do lugar** (+0,016 a +0,03 em blocos espaciais) e só aparece com informação do próprio destino. Ela é coerente com o MT09: a direção dos mecanismos é invariante, a intensidade varia muito entre componentes (I² 0,93–0,95).
3. **A alternativa testada que mostrou sinal foi observar o destino sem rótulo.** A frequência de água SAR do próprio pixel, fora do evento, deu +0,029 [+0,001; +0,060] pela regra pré-registrada, mas cai para +0,014 [−0,001; +0,031] usando só datas anteriores ao evento. É a pista mais concreta do projeto para o "benefício físico", e ainda não está demonstrada.
4. **Explicação causal: evidência observacional convergente, sob pressupostos declarados.** HAND e distância ao canal mantêm a direção dentro do evento, entre vizinhos a ≤ 3 km e nas quatro famílias de instrumento, o que é coerente com mecanismos físicos; isso não elimina por si o confundimento local nem identifica um efeito causal; HAND tem dose-resposta monotônica. Declive, SCA e distância à drenagem não se sustentam como mecanismo, embora o modelo servido use SCA e distância à drenagem com peso positivo em 43/43 retreinos.

## 1. Benefício físico do aprendizado conjunto

### As seis variantes

| Variante | Conjunto − parada, 43 componentes | IC | Leitura |
|---|---:|---|---|
| SAR, primeira tarefa (v1) | +0,0015 | [−0,0018; +0,0045] | ganho relevante excluído |
| Frequência SAR | +0,0032 | [−0,0008; +0,0079] | ganho relevante excluído |
| Supervisão temporal SAR (P1) | +0,0016 | [−0,0037; +0,0065] | ganho relevante excluído |
| Alvo óptico S2 informativo (O3) | +0,0026 | [−0,0019; +0,0071] | ganho relevante excluído |
| Gradiente auxiliar sem oposição à física (v4, outra sessão) | +0,0017 | [+0,0004; +0,0029] | ganho **detectável e irrelevante** |
| Peso igual por componente (MT07, objetivo) | +0,0002 | [−0,0023; +0,0030] | ganho relevante excluído |

IC97,5% nas campanhas conjuntas e IC95% no MT07. Nos dois casos, "não sustentado" no projeto significa limite superior abaixo de +0,01, ou seja, ganho relevante **excluído**, e não apenas não demonstrado. A v4 é a primeira com limite inferior acima de zero. O contraste dela contra o rótulo auxiliar permutado deu +0,0006 [−0,0011; +0,0023]: o pouco que ganha vem da regularização do gradiente, e não do conteúdo da tarefa auxiliar.

### Por que uma sétima variante não mudaria isso (MT08)

Em todas as variantes, a cabeça física só recebe uma função das mesmas entradas (terreno, conexão, cobertura). A tarefa auxiliar muda *qual* função é aprendida, mas não acrescenta entrada. O MT08 mediu, nos mesmos 43 componentes e pontos de teste do MT06, quanto **qualquer** função dessas entradas consegue ordenar no destino retido. Foram 1.763 ajustes, 6.305 s, com protocolo congelado às 03:18:59 UTC, antes de existir o resultado da v4.

| Braço menos Q_conexao | Dobras aleatórias | Blocos espaciais de 0,05° | Leitura |
|---|---|---|---|
| Árvores com reforço, só outros lugares (T_gbm) | −0,0032 [−0,0195; +0,0118] | — | flexibilidade aprendida fora não ajuda |
| Logística Q_conexao global **vendo rótulos do destino** (P_lin) | +0,0009 [+0,0004; +0,0016] | +0,0001 [−0,0006; +0,0009] | forma atual já no próprio teto |
| Árvores globais vendo o destino (P_gbm) | +0,0485 [+0,0260; +0,0780] | +0,0158 [−0,0094; +0,0453] | folga cai a um terço sem vizinho espacial |
| Logística só no destino (L_lin) | +0,0303 [+0,0136; +0,0505] | +0,0116 [−0,0095; +0,0334] | indecidível |
| Árvores só no destino (L_gbm) | +0,0779 [+0,0479; +0,1101] | +0,0172 [−0,0083; +0,0476] | indecidível |
| **Teto global** (máx. P por componente) | **+0,0535 [+0,0322; +0,0822]** | +0,0315 [+0,0139; +0,0553] | folga existe |
| **Teto local** (máx. P e L) | **+0,0924 [+0,0671; +0,1215]** | +0,0458 [+0,0262; +0,0731] | folga existe |

IC97,5% por componente; primárias em negrito. AUC conferidas por contagem de pares (diferença 0); agregação refeita por outro caminho (diferença 10⁻¹⁷); T_gbm reproduzido bit a bit em três destinos sorteados.

**A predição registrada não se confirmou.** Eu esperava teto global abaixo de +0,01, o que fecharia a pergunta por limite de informação. O teto primário ficou em +0,053, com folga sustentada. A leitura correta é a que a decomposição permite:

- **Nas famílias testadas, a parte compartilhável não melhora.** A logística global não melhora nem quando vê os rótulos do destino (+0,0001 em blocos). As árvores aprendidas fora dão −0,003 [−0,0195; +0,0118], estimativa que não exclui +0,01. Isso é coerente com os nulos das variantes conjuntas, cujo encoder também é treinado sem informação do destino. A v4, que saiu depois do congelamento, ficou em +0,0017, dentro do registrado. Não é demonstração de que nenhuma função das mesmas entradas, aprendida fora, ganharia mais.
- **O que sobra é específico do lugar e não linear.** Só aparece quando o ajuste usa rótulos do próprio destino e permite curvatura. Mais da metade da folga aleatória é vizinhança espacial: em blocos, cada braço isolado fica indecidível (+0,012 a +0,017). O máximo por componente (+0,03) ainda carrega otimismo de seleção.
- **O MT09 diz por quê.** A direção de HAND e da distância ao canal é a mesma em quase todo componente, mas a intensidade varia muito (I² 0,93–0,95; intervalo de predição de HAND de −2,18 a −0,09). Um modelo compartilhado só aprende a intensidade média; o resto é do lugar.

### A alternativa testada que mostrou sinal: observar o destino sem rótulo (MT10)

> **Atualização (18/10/2026, fechamento do MT11).** Os números do MT10 abaixo não mudam. Esta seção trata de **desenvolvimento retrospectivo**. A avaliação externa veio depois, no MT11, e os três níveis não devem ser confundidos:
>
> | Nível | Estudo | População e rótulo | F_transf − Q_conexao | Leitura |
> |---|---|---|---|---|
> | Desenvolvimento retrospectivo | MT10 primária | 18 componentes CEMS de desenvolvimento; rótulo CEMS compartilha S1 com F; `freq_fora` inclui datas posteriores ao evento | +0,029 [+0,001; +0,060] | sustentado pela regra, em dados que ajudaram a desenhar o estudo |
> | Sensibilidade pré-evento | MT10, pós-hoc | os mesmos 18; só datas S1 anteriores ao evento | +0,014 [−0,001; +0,031] | indecidível; pensada depois da primária |
> | Avaliação externa retrospectiva | MT11 (grafo primário com `atributos_e8`) | 16 componentes GFD 2015–2018 nunca usados; rótulo MODIS; F só com datas anteriores ao evento; máscaras auxiliares (JRC até 2021, WorldCover 2021) posteriores | +0,0404, IC98,33% [+0,0034; +0,0822] | sustentado na análise registrada; após a reparação posterior de independência (grafo da rodada 3B), o GFD:4355 liga-se a evento examinado e o HF fica em +0,0294 [−0,0032; +0,0628], 15 componentes, indecidível; no exutório (10 componentes), +0,0281 [−0,0123; +0,0725], indecidível |
>
> O MT11 é a primeira avaliação externa da integração entre a física congelada e o histórico SAR do destino; o ganho é positivo, mas depende de um componente e não está confirmado depois da reparação de independência. É integração de informação local observada, não aprendizado neural compartilhado nem identificação de causa. Detalhes em [MT11_FECHAMENTO_E_INTEGRACAO_2026_10_18.md](MT11_FECHAMENTO_E_INTEGRACAO_2026_10_18.md).

Em todas as variantes, a tarefa auxiliar foi retida junto com o componente. Nenhuma podia levar informação do destino. Entre as formas testadas de aprendizado conjunto, a que pode alcançar a folga medida é **transdutiva**: supervisão sem rótulo de inundação, observada *no* destino. (Redação corrigida em 18/10/2026; antes dizia "a única forma… que ainda pode alcançar a folga", o que excedia as alternativas testadas.) Antes de treinar uma rede assim, o MT10 mediu o limite de informação dessa via com a frequência de água SAR do próprio pixel fora das janelas de evento (`freq_fora` da pilha S1), que é exatamente o que uma tarefa auxiliar no destino ensinaria.

Combinação [Q, F, Q×F] ajustada nos outros componentes e aplicada ao destino: usa a observação do destino, nunca o rótulo. População pré-registrada: 20 componentes com referência SAR utilizável pela regra de 17/10, pixel observável, ≥ 10 datas. 4.518 pontos; 18 componentes avaliáveis.

| Recorte | F_transf − Q | Leitura |
|---|---|---|
| **Primária (regra E1/v1, 18 componentes)** | **+0,029 [+0,001; +0,060]** | sustentado pela regra |
| Sem pixel com histórico de água JRC | +0,024 [−0,009; +0,062] | indecidível |
| Todos os 26 componentes com v1 (sem regra de qualidade) | −0,0005 [−0,005; +0,004] | ganho relevante excluído |
| Só datas anteriores ao evento (**pós-hoc**) | +0,014 [−0,001; +0,031] | indecidível |
| F sozinho | −0,082 | F não substitui a física; complementa |

A sensibilidade só com datas anteriores foi pensada depois de ver a primária. Ela existe porque `freq_fora` exclui o evento e 60 dias de transição, mas inclui datas *posteriores*: água ou mudança de superfície que persista além disso pareceria previsão. O ganho cai pela metade. Com os dados atuais não dá para separar quanto disso é persistência pós-evento e quanto é só F mais ruidoso, já que sobram cerca de 24 de 32 datas. Há ainda uma ressalva fixada antes: o rótulo CEMS é delimitado sobre S1/óptico, então parte do ganho de F pode ser erro compartilhado de instrumento, e não física.

**Leitura.** A observação SAR do destino é a primeira entrada sem rótulo que alcança o tamanho da folga honesta do MT08, mas o efeito ainda não está demonstrado: depende da regra de qualidade e encolhe sem as datas posteriores. Se existir, o "benefício físico" que o aprendizado conjunto procurava é um benefício da **observação do lugar**, não do compartilhamento entre lugares.

## 2. Explicação causal (MT09)

### Por que a pergunta não estava fechada, e o que muda

O grafo conceitual de 16/10 (`conjunta_grafo_causal.py`) declarou o efeito do relevo `not_identified` porque a amostra só existe com seleção S=1, que depende da inundação: o evento foi ativado porque alagou e os pontos foram sorteados por classe. Isso é verdade para a prevalência e para interceptos. **Não é verdade para a razão de chances dentro do evento**: quando a amostragem depende apenas do rótulo, as inclinações da logística continuam consistentes para as razões de chances da população (Prentice & Pyke, 1979, *Biometrika* 66:403). A seleção do evento e da AOI vai para o intercepto livre por evento, que todo o MT01–MT07 já usa.

Restam três ameaças: confundimento dentro do evento, erro de rótulo diferencial e falta de positividade. Cada uma recebeu um desenho próprio, com regra de leitura registrada antes das estimativas (protocolo `local_runs/mecanismos-transferiveis/v9/identificacao_causal_v1/protocolo.json`).

| Desenho | O que remove ou testa |
|---|---|
| C2 dentro do evento | seleção do evento/AOI (intercepto livre); invariância por intervalo de predição para um componente novo |
| C3 triangulação | erro de observação: quatro famílias de instrumento com erros diferentes (MODIS 250 m, CEMS analista, S1/S2 manual 10 m, urbano 3 m) |
| C4 pares a ≤ 3 km | todo confundidor que varia em escala maior que o par: AOI, chuva regional, família de MDT, exposição, uso regional do solo. Logística condicional exata de pares; 1 km como sensibilidade |
| C5 dose-resposta | forma prevista pela hidráulica de enchimento (monotônica em HAND) |
| C6 cobertura | erro diferencial do instrumento por classe WorldCover (SAR vê mal sob árvore e prédio) |
| C1 amostragem | consequência testável do argumento de Prentice–Pyke: inclinações com 100% e 25% dos negativos |

População: 331.492 pontos completos, 124 eventos com ≥ 20 pontos por classe, 39 componentes (4 componentes não têm evento com as duas classes nesse piso). Efeitos em log da razão de chances por desvio-padrão da variável (escala comum), média de efeitos aleatórios entre componentes.

### Resultado por variável

| Variável | Dentro do evento (C2) | Intervalo de predição | Pares ≤ 3 km (C4) | Pares ≤ 1 km | Famílias a favor | Nível | Sinal previsto pela hidráulica |
|---|---|---|---|---|---:|---|---|
| HAND | −1,13 [−1,31; −0,96]; OR 0,32 | [−2,18; −0,09] | −1,27 [−1,48; −1,05] | −1,23 [−1,44; −1,03] | 4/4 | **mecanismo local triangulado, invariante** | concorda (−) |
| Distância ao canal ≥ 10 km² | −0,49 [−0,56; −0,41]; OR 0,62 | [−0,94; −0,03] | −0,58 [−0,74; −0,43] | −0,63 [−0,82; −0,45] | 4/4 | **mecanismo local triangulado, invariante** | concorda (−) |
| Declive | −0,24 [−0,42; −0,06] | [−1,33; +0,85] | −0,29 [−0,51; −0,08] | −0,25 [−0,38; −0,12] | 2/4 | associação dependente do instrumento | concorda (−) |
| Área de contribuição (ln SCA) | −0,11 [−0,17; −0,04] | [−0,40; +0,18] | −0,11 [−0,21; +0,00] | −0,14 [−0,25; −0,03] | 2/4 | associação dependente do instrumento | **contraria** (previsto +) |
| Distância à drenagem | +0,03 [−0,06; +0,11] | [−0,46; +0,52] | −0,10 [−0,19; −0,01] | −0,16 [−0,26; −0,07] | 0/4 (GFD contra) | não sustentado | — |

"Invariante" segue a regra registrada (intervalo de predição do C2 fora de zero). Nos pares a 3 km, o intervalo de predição de HAND continua fora de zero ([−2,44; −0,09]); o da distância ao canal toca zero ([−1,39; +0,23]). Num componente novo, o sinal do canal é provável, mas não garantido.

Triangulação de HAND (dentro do evento): MODIS −1,63 [−1,78; −1,48], S1/S2 manual −1,32 [−1,55; −1,09], urbano 3 m −1,32 [−1,66; −0,98], CEMS −0,89 [−1,09; −0,70]. Distância ao canal: −0,53, −0,45, −0,57, −0,44, todas com IC fora de zero. Os pares a 3 km reproduzem a triangulação nas quatro famílias para as duas variáveis. GFD tem um único componente; ali a unidade é o evento, declaradamente dependente.

**Dose-resposta (C5).** Em relação a HAND < 0,5 m, o log da razão de chances cai a cada faixa: −0,68 (0,5–1 m), −0,96 (1–2 m), −1,24 (2–5 m), −1,89 (5–10 m), −2,63 (≥ 10 m), com IC fora de zero em todas e intervalo de predição fora de zero a partir de 1 m. A sequência é monotônica, como prevê o enchimento a partir da drenagem.

**Cobertura (C6, exploratório).** O efeito de HAND tem o mesmo sinal, com IC fora de zero, sob árvore (−1,02), campo (−1,24), agricultura (−1,27), construído (−0,55), solo nu (−0,81) e água/área úmida (−0,64); distância ao canal também, em todas as seis. O efeito encolhe onde o instrumento vê pior (construído, área úmida), mas não inverte. Um artefato de observação deveria concentrar o efeito nas classes em que a água aparece fácil, e não é o que acontece.

**Amostragem por classe (C1) — resultado que limita a magnitude.** Para HAND, cortar 75% dos negativos não muda a inclinação: diferença −0,03 [−0,09; +0,03]. Para declive (+0,25 [+0,12; +0,36]) e distância ao canal (−0,21 [−0,42; −0,09]), muda. Pelo teorema, isso só acontece se a logística linear estiver mal especificada, e o C5 mostra que está (o efeito de HAND não é linear). Consequência: o **sinal** da distância ao canal é identificado (estável em C1, C2, C3 e C4); a **magnitude** por desvio-padrão depende do desenho amostral e não deve ser citada como razão de chances populacional. Para HAND, sinal e magnitude passam.

**A explicação do modelo servido (C7).** Nos 43 retreinos Q_conexao do MT06, os pesos principais de HAND, distância ao canal e declive são negativos em 43/43, como os mecanismos. Os de distância à drenagem e de SCA são **positivos em 43/43**: o modelo usa essas duas variáveis em sentido que não se sustenta como mecanismo (distância à drenagem não é sustentada; SCA tem o sinal oposto ao hidráulico e só aparece positiva no efeito marginal, +0,34, isto é, por supressão quando HAND entra junto). Uma explicação do escore que cite essas duas variáveis descreve o estimador, não a causa.

**Forçamento (C8, leitura dos resultados encerrados, sem nova estimativa).** No autopareamento das centrais (mesmo lugar, estrato célula × mês × dia da semana), a chuva do dia aumenta a notificação: Nova York RR 1,51 [1,47; 1,56] por 10 mm, com controle de chuva 30 dias depois em 0,95; São Paulo 1,22 [1,04; 1,45], controle 1,09 [0,82; 1,39]. O desfecho é notificação, não lâmina d'água. Na pilha S1 dos componentes CEMS, a chuva de 72 h não move a detecção SAR entre datas: RR 1,007 [0,973; 1,025], controle futuro 0,999.

Verificação independente: as inclinações por evento foram reestimadas com o sklearn por outra formulação (C4 como logística sem intercepto sobre diferenças com sinal sorteado). Diferença máxima 0,0004 (C2) e 0,0009 (C4), contra efeitos de −0,5 a −1,3. A meta-análise refeita por fórmula escrita de novo reproduz as médias com diferença zero. Simulações nos testes confirmam as duas propriedades usadas: inclinação preservada sob amostragem por classe e confundidor regional removido pelo par a 1 km quando a logística comum fica viciada.

### O que a evidência sustenta e o que não sustenta

**Sustentado como associação observacional convergente, com pressupostos declarados.** Comparando dois pontos do mesmo evento e da mesma vizinhança de 3 km, o mais baixo em relação à drenagem e o mais próximo de um canal de ≥ 10 km² tem mais chance de estar inundado. Isso vale nas quatro famílias de instrumento, nas seis classes de cobertura e, para HAND, com forma monotônica de dose-resposta. É um mecanismo **estrutural-condicional**, invariante entre componentes no sentido do intervalo de predição. A explicação compatível com essa evidência é *posição vertical em relação à drenagem e conexão a um canal*, nessa ordem de força. Ler isso como efeito causal depende dos pressupostos abaixo, que não foram todos testados.

**Não é efeito de intervenção.** Posição no relevo não se manipula; a frase "se este ponto fosse 1 m mais alto" é um contrafactual hidráulico, não um experimento.

**Ameaças que o desenho não remove**, e o que cada uma exigiria:
- infraestrutura local (dique, galeria, bomba) correlacionada com o relevo dentro de 3 km: inventário com estado e data, só onde for o mecanismo reivindicado;
- erro do MDT correlacionado com cobertura no pixel: o C6 limita (mesmo sinal sob árvore e prédio), mas não elimina;
- componentes e eventos são os conhecidos: a confirmação em unidades novas continua necessária para a magnitude.

**Não sustentado como mecanismo:** declive (depende do instrumento; quase todo o efeito vem do MODIS), SCA (sinal contrário à hidráulica) e distância à drenagem (some quando HAND e canal entram). O forçamento tem efeito identificado na notificação do mesmo lugar, mas não na água vista pelo SAR entre datas; a ponte entre chuva e lâmina d'água continua sem observação adequada.

## 3. Consequências

### Para o artigo

- **Aprendizado conjunto:** apresentar como resultado negativo delimitado às variantes testadas, que tiveram ganho relevante excluído. A decomposição do MT08 mostra onde está a folga medida (não linear e específica do lugar) nas famílias testadas, sem provar o teto de qualquer função. É uma contribuição metodológica: medir a folga disponível antes de testar mais uma variante.
- **Causalidade:** "mecanismo estrutural-condicional triangulado", com a ressalva de Prentice–Pyke explícita. HAND com sinal e magnitude; distância ao canal só com sinal. Não usar "efeito causal" sem o qualificador.
- **Explicação do modelo servido:** as razões que citam SCA ou distância à drenagem descrevem o estimador. Cartões e contrato devem marcar essas duas variáveis como não-mecanismo (C7: peso positivo em 43/43, sem sustentação causal).

### Atualização das frentes 10 e 14

| Frente | Antes | Agora |
|---|---|---|
| 10. Aprendizado da IA integrada | ganho físico relevante não sustentado | Ganho relevante excluído nas cinco variantes conjuntas e no MT07; no MT08, nenhuma família testada sem informação do destino o atingiu (T_gbm não o exclui). A via transdutiva está em teste confirmatório (MT11) |
| 14. Previsão/profundidade/causalidade | causalidade fora da afirmação | Evidência observacional convergente para HAND e conexão (MT09), sob pressupostos declarados; efeito causal não identificado; efeito de intervenção e de infraestrutura fora; forçamento identificado só na notificação |

### O que fecharia a via transdutiva

Um teste novo, pré-registrado antes de ler qualquer resultado, com:
- F calculado **só com datas anteriores** ao evento (a forma que um serviço teria);
- unidades **não usadas** em MT05–MT10. Esta é uma decisão da pesquisadora, não uma etapa automática. As cinco ativações CEMS fluviais não examinadas (EMSR856, 855, 854, 850, 841) não têm pacote nem ponto, dão só leitura descritiva (o requisito registrado é ≥ 16 unidades) e, uma vez abertas, deixam de ser não examinadas, competindo com a confirmação dos pesos Q. As alternativas são eventos novos ou abrir reservas de outros protocolos;
- rótulo de **outro instrumento** onde houver (S2 da rodada 2), para separar física de erro compartilhado com o S1;
- margem +0,01 e o mesmo estimador F_transf do MT10.

Só depois desse resultado vale treinar a rede conjunta transdutiva (tarefa auxiliar de frequência no próprio destino). Se F não carregar a folga em unidades novas, a rede também não carregará.

Nada aqui altera o serviço, abre reserva ou promove modelo.

## Reprodução

Protocolos congelados antes de qualquer ajuste ou estimativa: MT08 às 03:18:59 UTC e MT09 antes da execução, com hashes de código, base e coordenadas. MT09 teve uma fumaça em um evento por família, sem estimativa impressa, registrada no protocolo.

```bash
python scripts/manutencao/executar_serializado.py -- python -X utf8 scripts/metodo/mt09_identificacao_causal.py executar
```

```bash
python scripts/manutencao/executar_serializado.py -- python -X utf8 scripts/metodo/mt08_teto_compartilhado.py executar
```

```bash
python scripts/manutencao/executar_serializado.py -- python -X utf8 scripts/metodo/mt10_teto_transdutivo.py executar
```

```bash
python -X utf8 scripts/metodo/mt10_sensibilidade_pre_evento.py
```

```bash
python -X utf8 scripts/metodo/mt08_mt09_verificacao_independente.py
```

O MT10 foi escrito e congelado depois de ler o MT08 (registrado no protocolo como `motivacao_mt08_lida`), antes de qualquer ajuste próprio. A sensibilidade só com datas anteriores ao evento é pós-hoc.

Artefatos: `local_runs/mecanismos-transferiveis/v8/teto_compartilhado_v1/`, `.../v9/identificacao_causal_v1/`, `.../v10/teto_transdutivo_v1/`, verificação em `.../verificacao_independente_mt08_mt09.json`, figura em `local_runs/mt08-mt09/`. Índice de evidência regenerado em `docs/ESTADO_CIENTIFICO_ATUAL_2026_10_17_V3.md` (o snapshot anterior fica preservado). Testes: `tests/test_mt08_teto_compartilhado.py`, `tests/test_mt09_identificacao_causal.py`, `tests/test_mt10_teto_transdutivo.py` (19 novos; 35 com os de MT06/MT07).
