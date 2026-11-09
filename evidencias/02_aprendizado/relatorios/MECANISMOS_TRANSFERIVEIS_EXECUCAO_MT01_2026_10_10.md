# Mecanismos transferíveis: execução do protocolo MT01

Data: 10/10/2026. Desenvolvimento prospectivo sobre eventos **já examinados** em ciclos anteriores. Não é confirmação independente. Nada foi promovido: escore servido, `svc02_contrato_inferencia.py`, portões e maturidade continuam intocados. As reservas EMSR868/EMSR871 não foram abertas; o código recusa abri-las.

Pré-registro `sha256:1e8a6f57…eb1` (`local_runs/mecanismos-transferiveis/v1/pre_registro.json`), gravado antes de qualquer ajuste oficial. Resultado `sha256:38cb4947…5b1`.

## Resposta curta

- **O que se mostrou transferível.** A relação de **acumulação**: quanto mais alto o lugar em relação à drenagem (HAND) e ao piso local do terreno, menos ele aparece inundado.
  - Aprendida em 42 territórios e aplicada ao 43º, sem nenhum rótulo dele, ordena a inundação observada melhor que a regra HAND pura em 43 territórios independentes de 5 continentes.
  - G − regra HAND = **+0,039** de AUC média por unidade (IC95% por unidade 0,026–0,054; 33/43 unidades a favor).
  - G − regra física de pesos iguais = **+0,018** (IC 0,010–0,026).
  - Tirar o bloco de acumulação custa **−0,064** (IC −0,080 a −0,048; perda em 39/43 unidades).
  - O sinal de HAND é negativo em 25 de 27 territórios ajustados isoladamente; a magnitude varia cerca de seis vezes entre eles.
  - Transfere-se a forma e a direção da relação, não um coeficiente idêntico.
- **Para quais processos e destinos.** Ordenação dentro do evento, para inundação **fluvial** (AUC 0,765), **costeira** (0,813) e **urbana mista** (UFO, 14 cidades: 0,807), inclusive para a América do Sul inteira sem rótulo sul-americano (0,768). Na **enxurrada** a transferência é mais fraca (0,718; só 9 unidades). **Alagamento pluvial urbano isolado** continua não testável.
- **O que o contexto modifica.** Nada que este protocolo tenha conseguido medir:
  - Os modificadores pré-declarados (escala do rio, relevo do território, HAND p90, bacia costeira) não mudam o desempenho (C − G = −0,0007; H − G = −0,0005; ambos não inferiores, nenhum superior).
  - O contexto aparece por outro caminho, no **teto**: territórios planos (deltas com maré, planície irrigada) e dominados por enxurrada transferem pior. Isso é exploratório.
  - O sinal do declive se divide entre territórios (14 negativos, 13 positivos) e nenhum modificador disponível explica essa divisão.
- **Quanto dado local ainda é necessário.**
  - **Para ordenar:** pouco. 20, 100 ou 500 rótulos de um evento do destino não melhoram o modelo transferido (−0,003, −0,003, +0,0003). Um modelo só com eles fica bem pior (−0,055 com 20 rótulos).
  - **Com muitos eventos locais:** na América do Sul, 4/5 dos 75 eventos somam +0,006 a +0,014 (pós-hoc).
  - **Para afirmar desempenho no destino:** ainda é preciso observação local independente. Sem ela, a ordenação é herdada, não validada.
- **Falhas que limitam a tese.**
  1. Não há unidade brasileira fora do componente sul-americano.
  2. O alagamento pluvial não tem negativo observado.
  3. A restrição da saída (diques, canais, maré, galerias) não está representada, e as falhas se concentram onde ela deveria mandar.
  4. A enxurrada tem pouco suporte.
  5. Todos os eventos já tinham sido lidos antes.
- **Avanço para "por que ali alaga?".**
  - O REV-P pode responder, para um lugar sem histórico, "porque está baixo em relação à drenagem e ao entorno". Essa é uma relação de acumulação herdada de 42 territórios independentes, com direção estável e desempenho medido fora deles.
  - Também pode dizer onde essa resposta é fraca: terreno plano, saída controlada por obra ou maré, chuva concentrada em encosta.
  - Não pode ainda dizer "porque a galeria não dá vazão" ou "porque a chuva foi X". Esses blocos não estão identificados.

## Estado de cada parte

| Estado | O quê |
|---|---|
| **Implementado** | População multi-fonte com processo por polígono CEMS; HydroBASINS em 6 regiões; grafo de dependência com 3 desenhos; contexto sem rótulo por raster inteiro; logística com intercepto por evento, interações e inclinação por território (MAP); braços de regra, geral, condicionado, parcial, ablações, fonte excluída, doadores com suporte igual, curva de territórios e poucos rótulos; analisador por unidade com bootstrap; verificação independente; 12 testes |
| **Executado** | 43/43 destinos do pré-registro; diagnóstico do TT01; ocorrência JRC em 513.530 pontos; pós-hoc (família de MDT, suporte, heterogeneidade, teto local, métricas secundárias corrigidas); sensibilidades JRC (43 unidades) e exutório (32 unidades), seção 6.9 |
| **Sustentado** | H1 (G > regra HAND; G > regra composta); necessidade preditiva da acumulação (H3); ausência de dependência da fonte do destino (H4: −0,0015, IC −0,0035 a 0,0001); G > doadores aleatórios com suporte igual (+0,017, IC 0,011–0,023) |
| **Não inferior, sem ganho** | C e H frente a G (H2): a parametrização por contexto pré-declarada não é necessária |
| **Exploratório** | Relevo × desempenho (Spearman 0,45–0,49); falhas planas e de enxurrada; referência costeira (elevação local > HAND em 5/6 eventos costeiros e também em 15/25 fluviais, então não é distintiva); teto local; heterogeneidade de coeficientes |
| **Inconclusivo** | Bloco de conexão (−0,003, IC −0,015 a 0,003); doadores por contexto × proximidade × aleatório com suporte igual; poucos rótulos com 100 e 500 (7 e 4 unidades) |
| **Refutado (nesta formulação)** | A expectativa de que modificadores de rio/relevo/costa mudassem a resposta. Os sinais das interações saíram estáveis entre as dobras, mas com magnitude desprezível (~10⁻³ na escala padronizada), e dois saíram contra a expectativa escrita: `ln_SCA×relevo` negativo e `HAND×costa` positivo. Isso refuta a parametrização, não a transferência |
| **Não testável** | Alagamento pluvial urbano isolado (sem negativo observado); transferência para destino brasileiro separado da América do Sul; calibração/probabilidade no destino (intercepto desconhecido; probabilidade condicionada ao evento não é anual); produção de escoamento como mecanismo (chuva quase constante dentro do evento; cobertura e solo confundidos com observação) |

## 1. O que é testável: população evento × território × processo × fonte × qualidade

Código: `scripts/metodo/mt01_populacao.py`. Entrada: ds06 v7 (763.155 linhas), acervo espacial e pacotes CEMS. Saída: `populacao.parquet`, `inventario.json`.

| Fonte | Papel | Eventos com duas classes | Processo | Instrumento / resolução | MDT |
|---|---|---:|---|---|---|
| GFD | 146.797 positivos / 260.785 negativos observados | 69 (América do Sul) | do evento: fluvial de grande escala | MODIS 250 m; nuvem e cheia rápida invisíveis | FABDEM |
| CEMS | 13.608 / 37.875 | 32 (5 continentes) | **no ponto**, pelo polígono: fluvial 8.883, enxurrada 2.144, costeira 1.877, misto 704 | rapid mapping 10–30 m | FABDEM 51% / outro 49% |
| Sen1Floods11 | 12.949 / 17.482 | 11 países | do evento: fluvial monçônico | rótulo manual S1/S2 10 m | outro |
| UFO | 11.640 / 12.394 | 18 (14 cidades) | do evento: urbano misto | imagem 3 m | outro |
| UK | 3.738 / 3.738 por exclusão | 0 | fluvial registrado | contornos EA | — |
| NYC, Chicago, SP, Curitiba, Recife | só presença (482 mil) | — | pluvial urbano de registro | centrais 156/311 | — |

- **Processo no ponto.** Só existe no CEMS: 100% dos positivos CEMS caíram num polígono `observedEvent` com `obj_desc`. GFD, Sen1Floods11 e UFO têm processo só do evento, declarado como tal.
- **Ausência.** O ds05 tem 735.654 linhas de ausência nas centrais. Elas são *target-group background* (Phillips et al., 2009): ordenam propensão a registrar, não suscetibilidade, e não entram no contraste.
- **Água frequente.** Amostrada a ocorrência JRC v1.4 nos 513.530 pontos.
  - ≥ 50% de ocorrência: 0,07% dos positivos GFD, 3,1% dos CEMS, 1,5% dos Sen1Floods11 e **17,0% dos UFO** (canais urbanos).
  - O ds06 já tinha limpado o GFD; o pacote TC01 tinha 16%.

**Blocos de processo e seus proxies** (nenhum é medida direta):

| Bloco | Proxy | Limite declarado |
|---|---|---|
| Acumulação / extravasamento | `hand_m`, `elevation_local_m` (contra o piso da unidade) | HAND é altura sobre a drenagem representada (FABDEM ou outro MDT a 30 m), não cota nem capacidade do canal |
| Concentração / conexão | `ln_SCA` = TWI + ln tan(declive), `slope_deg` | Área contribuinte D-inf; não vê galeria nem barreira |
| Produção de escoamento | `construido_frac` (GHSL), `areia_pct_0_30` (SoilGrids) | Confundidos com observação (ver seção 7). A areia marca planície aluvial tanto quanto infiltração. A chuva (ERA5-Land ~9 km) é quase constante dentro do evento e fica fora do contraste |
| Restrição da saída | — | Ausente: diques, canais, galerias, maré, remanso, operação |

## 2. Independência: por que o TT01 virou um componente e o que o MT01 fez

Código: `scripts/metodo/mt01_independencia.py`. Saídas: `diagnostico_tt01.json`, `independencia.json`, `unidades.parquet`.

**Diagnóstico do TT01, sem reajustar nada.**

| Ligação | Componentes |
|---|---:|
| só tile | 3 |
| só MAIN_BAS | 1 |
| só raster de terreno | 4 |
| tile + MAIN_BAS + raster | 1 |

- **Não são poucas pontes.** Mesmo retirando os 20 eventos de maior grau, sobram 4 componentes por tile.
- **Os eventos GFD são grandes.** 61 de 69 eventos cobrem mais de 10 tiles (mediana 15, máximo 18), e 217 tiles são revisitados por mais de um evento.
- **Lugar observado de novo.** 9,0% das células de 250 m aparecem em mais de um evento, e 3.935 delas com rótulos diferentes. São planícies que inundam repetidamente.
- **O MAIN_BAS junta anos diferentes no mesmo sistema.** O maior sistema tem 33 eventos.
- **Leitura.** O componente único é dependência real de **lugar** e de **sistema fluvial**, não rótulo compartilhado. O raster de terreno é covariável determinística: compartilhá-lo não vaza classe.

**O veredito do TT01 fica preservado.** A verificação do TT01 agora falha: `entrada/codigo alterado: local_runs\hydrobasins\v1c\tabelas\manifesto.json`. O ciclo de 10/10 (Europa e nível 1) alterou depois do pré-registro o manifesto das tabelas sul-americanas e o próprio `bacias_hydrobasins.py`; o `nivel_06.parquet` que o TT01 usou confere. Nada foi editado aqui; o defeito fica registrado.

**Grafo do MT01** (novo protocolo, registrado antes dos resultados):

- **Ligações:**
  - mesmo evento;
  - mesmo lugar (unidade espacial ou sub-bacia nível 8);
  - mesmo MAIN_BAS com janelas a menos de 30 dias (mesma onda de cheia);
  - pegadas a menos de 500 km com janelas a menos de 15 dias (mesmo sistema de chuva).
- **Bacias em todas as regiões do estudo.** O HydroBASINS foi estendido a África, Ásia, América do Norte e Oceânia: 1,44 GB, tamanhos conferidos no servidor, mesma licença já aceita para SA/EU. Isso foi feito em tempo de execução por `mt01_bacias_globais.py`, sem editar o módulo congelado no TT01.

| Desenho | Unidades | Maior unidade (eventos) | Unidades com fontes misturadas |
|---|---:|---:|---:|
| Primário (evento, lugar, onda, chuva) | **43** | 75 | 4 |
| Conservador (mesmo exutório, qualquer data) | 32 | 76 | 11 |
| Liberal (evento e lugar) | 45 | 75 | 4 |

- **Nenhum componente foi dissolvido.** As 43 unidades vêm de **acrescentar territórios de outros continentes**.
- **O componente sul-americano continua inteiro:** todo o GFD, EMSR720, EMSR799, EMSR853, Bolívia e Paraguai do Sen1Floods11, e UFO Uruguai.
- **A regra junta o que deve.** As cheias de 2019 do Missouri/Mississippi (Sen1Floods11 EUA; UFO Omaha, St. Joseph e Grafton) formam uma unidade só.
- **Verificação independente:** nenhuma unidade compartilha evento, unidade espacial ou sub-bacia nível 8 com outra.

## 3. Fundamentação que mudou decisões

| Fonte primária | Decisão no MT01 |
|---|---|
| Kratzert et al. 2019 (WRR, bacias não monitoradas); Tsai et al. 2021; Feng et al. 2022 | O alvo é uma relação compartilhada aplicada ao destino sem calibração local. A parametrização por atributos foi implementada como interações (C) e inclinação parcial (H), não como modelo diferenciável: falta série de forçamento e resposta para identificar parâmetros de processo |
| Gelman & Hill 2007 (*pooling* parcial) | No destino novo o efeito aleatório é zero; só os preditores de grupo levam informação. Por isso H só pode diferir de C pelo que o *pooling* muda no treino |
| Peters, Bühlmann & Meinshausen 2016; Rothenhäusler et al. 2021 (*anchor regression*, JRSSB 83:215) | Estabilidade entre ambientes ≠ causa. Reportam-se sinal por território e desempenho no pior destino. A *anchor regression* logística fica como próximo braço, não implementado |
| Roberts et al. 2017; Ploton et al. 2020; Wadoux et al. 2021 (contraponto) | Bloqueio pela pergunta (território novo) e pela dependência medida. Como a população não é amostra probabilística, nada aqui é acurácia de mapa |
| Phillips et al. 2009 (*target-group background*) | Ausência de chamado é fundo de grupo-alvo: o pluvial das centrais fica fora do contraste |
| Tellman et al. 2021 (GFD) | MODIS 250 m não vê cheia rápida nem urbana (300 eventos "pequenos ou rápidos" excluídos do próprio GFD). O processo de enxurrada só pode ser avaliado pelo CEMS |
| Johnson et al. 2019 (NWM-HAND) | HAND erra mais em baixo relevo, o que se reproduziu nas falhas (exploratório). Isso motiva a hipótese de recusa por relevo |
| Löwe et al. 2021 (U-FLOOD) | Forma do terreno + impermeabilização bastam para emular lâmina pluvial dentro de uma cidade. A transferência entre cidades não foi testada ali, e aqui também não pôde ser no pluvial |
| Hu et al. 2024; Pool et al. 2021 | Doadores comparados com suporte igual (mesmo número de eventos e teto por classe). Proximidade e semelhança são hipóteses, não regras |

## 4. Protocolo congelado

Código: `scripts/metodo/mt01_experimento.py`. O hash de cada entrada e de cada código está no pré-registro, e o executor recusa rodar se algum mudar.

- **População.** 510.348 linhas, 130 eventos e 43 unidades, com positivo e negativo observado nos eventos que têm as duas classes.
- **Retenção.** Cada destino retém o componente inteiro.
- **Orçamento.** Ajuste com até 400 linhas por evento e classe; teste com até 2.000; semente 20261010.
- **Perda.** Cada evento pesa 1 e cada classe metade dele, com intercepto por evento livre. O que se avalia é a ordem dentro do evento.
- **Atributos.** `log1p(HAND)`, `log1p(elev. local)`, `ln_SCA`, `log1p(declive)`, `construído`, `areia`, com imputação pela mediana do treino.
- **Correção antes do congelamento.** As *flags* de ausência foram retiradas depois do ensaio: a de solo ausente recebia +0,51 porque o SoilGrids falta sobre água e área construída.
- **Seleção.** Penalidades escolhidas em 3 dobras internas por território de treino; o destino nunca participa.
- **Métricas.**
  - Primária: AUC dentro do evento, com média igual dos eventos na unidade e depois das unidades. IC por bootstrap de unidades (2.000) e margem de relevância 0,01.
  - Mínimo de 8 unidades para inferir.
  - Calibração não é reportada.

## 5. Resultados principais (43 unidades, 130 eventos, 335.848 linhas de teste)

| Modelo | AUC média por unidade | Mediana | Pior unidade | Decil inferior | AP / prevalência* | Captura no topo 10%* | Captura no topo 20%* |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 regra HAND | 0,7376 | 0,7505 | 0,465 | 0,563 | 1,85 | 0,167 | 0,330 |
| A1 regra composta (sentidos físicos, pesos iguais) | 0,7587 | 0,7631 | 0,489 | 0,625 | 2,46 | 0,212 | 0,387 |
| **G geral** | **0,7767** | 0,7918 | 0,505 | 0,643 | 2,61 | 0,224 | 0,400 |
| C condicionado | 0,7760 | 0,7886 | 0,504 | 0,643 | 2,57 | 0,217 | 0,399 |
| H parcial | 0,7761 | 0,7886 | 0,505 | 0,643 | 2,56 | 0,218 | 0,400 |

\* Recalculado das previsões gravadas, com empate tratado; ver seção 7. A captura no topo vale para o quadro amostrado, não para a área. AUC não é percentual de acerto.

| Contraste | Δ médio | IC95% (unidades) | Unidades a favor | Veredito |
|---|---:|---|---:|---|
| G − regra HAND (H1) | +0,0391 | 0,0257 a 0,0538 | 33/43 | sustentado |
| G − regra composta (H1) | +0,0179 | 0,0102 a 0,0260 | 34/43 | sustentado |
| regra composta − regra HAND | +0,0212 | 0,0102 a 0,0334 | 31/43 | sustentado |
| C − G (H2) | −0,0007 | −0,0014 a −0,0001 | 17/43 | não inferior |
| H − G (H2) | −0,0005 | −0,0017 a 0,0007 | 16/43 | não inferior |
| sem acumulação − G (H3) | −0,0636 | −0,0801 a −0,0481 | 4/43 | perda relevante |
| sem conexão − G (H3) | −0,0035 | −0,0148 a 0,0032 | 28/43 | indecidível |
| sem produção − G (H3) | −0,0035 | −0,0085 a 0,0011 | 13/43 | sem perda relevante |
| sem a fonte do destino − G (H4) | −0,0015 | −0,0035 a 0,0001 | 16/42 | sem perda relevante |
| doadores contexto − próximos | −0,0051 | −0,0238 a 0,0102 | 20/43 | indecidível |
| doadores contexto − aleatórios | +0,0064 | −0,0075 a 0,0175 | 32/43 | indecidível |
| doadores próximos − aleatórios | +0,0115 | −0,0005 a 0,0244 | 27/43 | indecidível |
| G (tudo) − doadores aleatórios | +0,0166 | 0,0106 a 0,0230 | 36/43 | sustentado |

**Decisão pré-registrada:** a candidata é **G**. H e C são não inferiores, mas não melhores no pior destino, e G é o mais simples.

## 6. Fenômeno e atalhos

### 6.1 Por fonte do destino

| Destino | Unidades | Regra HAND | G |
|---|---:|---:|---:|
| CEMS | 23 | 0,725 | 0,766 |
| Sen1Floods11 | 8 | 0,764 | 0,781 |
| UFO (urbano misto) | 8 | 0,742 | 0,807 |
| América do Sul (componente) | 1 | 0,742 | 0,768 |

### 6.2 Por processo no ponto (CEMS)

Positivos de um processo contra os mesmos negativos observados do evento.

| Processo | Unidades | Eventos | Regra HAND | G |
|---|---:|---:|---:|---:|
| Fluvial | 20 | 25 | 0,734 | 0,765 |
| Costeira | 5 | 6 | 0,751 | 0,813 |
| Enxurrada | 9 | 10 | 0,695 | 0,718 |
| Misto | 4 | 4 | 0,795 | 0,834 |

O resultado fluvial não é transferido automaticamente para enxurrada nem para alagamento urbano: cada um foi medido separadamente, e o pluvial isolado não pôde ser medido.

### 6.3 Qualidade, instrumento e suporte

- **Fonte do destino excluída do treino:** −0,0015 (H4). O ganho não vem do instrumento do destino.
- **Família de MDT do destino (pós-hoc):** FABDEM, G 0,764 em 15 unidades; outro bilinear, G 0,784 em 28. A diferença acompanha a fonte (o GFD é todo FABDEM, os chips são todos outro MDT) e não é separável deles.
- **Peso por suporte (pós-hoc).** Ponderar as unidades por √min(positivos, negativos) dá G 0,777 e regra HAND 0,747. O resultado não depende das unidades minúsculas.
- **Doadores com o mesmo número de eventos (mediana 6) e o mesmo teto:** contexto, proximidade e sorteio empatam. O G com todos os territórios supera qualquer subconjunto de 6 eventos. Na seleção de doadores, o que pesa é diversidade e suporte, não semelhança.
- **Curva por número de territórios de treino:** 3 territórios 0,749; 6 0,767; 12 0,773; 42 0,777. Satura.

### 6.4 Interações e comportamento do modelo

- As interações de C ficaram ~10⁻³ na escala padronizada; a seleção interna escolheu penalidade alta em boa parte das dobras.
- A intervenção sintética (+1 desvio em cada atributo, por quartil de escala do rio e de relevo) dá um efeito de HAND de −0,93 a −0,94 em todos os quartis.
- Isso verifica o comportamento do modelo, não a causa no mundo.

### 6.5 Heterogeneidade entre territórios (pós-hoc)

G ajustado dentro de 27 unidades com suporte; coeficiente por desvio do conjunto todo:

| Atributo | Mediana | p10 a p90 | Negativos / positivos |
|---|---:|---|---|
| HAND | −0,93 | −1,36 a −0,21 | 25 / 2 |
| Elevação local | −0,63 | −2,17 a 0,02 | 23 / 4 |
| ln_SCA | +0,07 | −0,18 a 0,23 | 8 / 19 |
| Declive | 0,00 | −0,44 a 0,70 | 14 / 13 |
| Construído | −0,31 | −0,71 a −0,02 | 24 / 2 |
| Areia | +0,17 | −0,87 a 1,05 | 10 / 17 |

- **Exceções ao HAND negativo.** HAND ≥ 0 aparece em Cartum (UFO), onde a elevação local manda (−3,08), e em EMSR754, Vietnã costeiro (+0,65, com elevação local −2,17).
- **Leitura física possível.** Na água vinda do mar ou de rio largo e plano, a referência que importa é o piso local, não a drenagem desenhada pelo MDT. Isso é hipótese; o teste descritivo por processo não a distinguiu.

### 6.6 Onde falha (exploratório, depois de ver os resultados)

- **Destinos com G < 0,70.** São 9:
  - EMSR838 e S1F11 Paquistão (planície do Indo);
  - EMSR790, EMSR750 e EMSR696 (enxurrada);
  - UFO Can Tho e EMSR754 (delta com maré);
  - EMSR865 e EMSR857.
- **São mais planos.** Declive mediano de 0,55° contra 1,75°; HAND p90 de 9,7 m contra 18,6 m.
- **O desempenho sobe com o relevo.** Spearman com HAND p90 = 0,45 (p = 0,002) e com declive = 0,49 (p = 0,001) nas 43 unidades.
- **Corte de 10 m, escolhido depois.** Territórios planos dão G 0,735 contra 0,799; o G ainda supera a regra HAND nos planos (+0,047).
- **Diagnóstico.** Falta representação física (dique, canal, maré: restrição da saída), não suporte. C e H não corrigem.

### 6.7 Poucos rótulos locais (braço separado)

O evento de adaptação é o primeiro no tempo, e a avaliação é feita nos outros eventos do destino.

| n rotulados | Unidades | H sem rótulo local | H adaptado | Só local |
|---:|---:|---:|---:|---:|
| 20 | 9 | 0,763 | 0,760 | 0,708 |
| 100 | 7 | 0,761 | 0,758 | 0,745 |
| 500 | 4 | 0,778 | 0,778 | 0,770 |

- Adaptar a inclinação local com rótulos de **um evento** não ajuda a prever **outro evento** do mesmo território.
- O conhecimento transferido vale mais que um modelo local pequeno.
- Com menos de 8 unidades em 100 e 500, isso é descritivo.

### 6.8 Teto com dado local abundante (pós-hoc)

| Território | Eventos | Regra HAND | G sem rótulo local | G + 4/5 dos eventos locais | G só local |
|---|---:|---:|---:|---:|---:|
| América do Sul | 75 | 0,741 | 0,768 | 0,775 | 0,782 |
| Missouri/Mississippi 2019 | 7 | 0,862 | 0,893 | 0,893 | 0,888 |

Eventos do mesmo território dividem lugar. É exatamente o cenário "com dado local", e por isso a comparação é otimista para o local.

### 6.9 Sensibilidades pré-registradas

Só os braços A0, A1, G, C e H, com as mesmas regras do pré-registro.

| Sensibilidade | Unidades | G − regra HAND | G − regra composta | C − G | H − G |
|---|---:|---|---|---|---|
| Principal | 43 | +0,039 (0,026 a 0,054) | +0,018 (0,010 a 0,026) | −0,0007, não inferior | −0,0005, não inferior |
| Sem água frequente (JRC v1.4 ≥ 50 sai) | 43 | +0,040 (0,026 a 0,056) | +0,018 (0,010 a 0,026) | −0,0001, não inferior | +0,0015, não inferior |
| Mesmo exutório em qualquer data | 32 | +0,036 (0,021 a 0,054) | +0,014 (0,005 a 0,023) | −0,0005, não inferior | −0,0004, não inferior |

- **Os vereditos não mudam** em nenhuma das duas sensibilidades.
- **Estrato urbano (UFO).** Era o mais afetado pela água frequente (17% dos positivos). Sem ela, o G cai de 0,807 para 0,791 e a regra HAND de 0,742 para 0,722; o ganho do G sobre a regra se mantém.
- **Arquivos:** `sensibilidade_jrc.json`, `sensibilidade_exutorio.json` e `pos_hoc_destinos_jrc.json`.

## 7. Atalhos e defeitos encontrados

1. **Ausência de solo como rótulo.**
   - No ensaio, a *flag* "solo ausente" pegou +0,51.
   - O SoilGrids falta em 24% dos positivos e 6% dos negativos do GFD (água e área construída).
   - Retirada antes do congelamento.
2. **Cobertura ausente por versão do pipeline.**
   - Todas as 15.437 linhas CEMS datadas por cena (acrescentadas no ds06 v6) estão sem cobertura e solo, porque o `cob01` rodou sobre o ds06 v3. 93% dessas linhas são negativos.
   - O bloco "produção" no CEMS carrega esse padrão, e a ablação mostra que ele não é necessário.
   - Correção recomendada: refazer o `cob01` sobre o v7, como nova versão.
3. **AP e captura no topo com empate.**
   - O executor não tratava empate, e a regra HAND tem muitos HAND = 0. A AP/prevalência da regra saiu 4,36 no registro original; o correto é 1,85.
   - Recalculado das previsões, com os valores originais preservados nos registros dos destinos. A AUC (postos médios) não foi afetada; a verificação recalculou 650 AUCs com divergência máxima de 2,2×10⁻¹⁶.
4. **Verificação do TT01 quebrada** pelo manifesto de bacias de 10/10 (seção 2).
5. **Execução fora da trava.**
   - A trava estava com a grade canônica de outra sessão.
   - O MT01 rodou em prioridade baixa, com 2 *threads* e ~1,4 GB.

## 8. O que isso permite em "por que ali alaga?"

**Antes:** a relação HAND/terreno era sustentada dentro do GFD sul-americano, com um componente só e sem como saber se valia fora dele.

**Agora:**
- **O que se pode afirmar.** Num lugar sem histórico, a ordenação vem de uma relação de acumulação aprendida em 42 territórios independentes de 5 continentes e verificada no 43º.
  - Ela sobrevive à troca de instrumento e de continente.
  - A América do Sul é ordenada sem nenhum rótulo sul-americano, com AUC de 0,768.
  - Os rótulos locais acrescentam cerca de 0,01.
- **Como explicar.** A explicação para o usuário pode dizer "está baixo em relação à drenagem e ao entorno, a mesma relação que separa inundado de seco em 25 de 27 territórios observados".
- **Onde a resposta é fraca.** Ela pode marcar o caso em que é fraca: terreno plano com obra ou maré, encosta com enxurrada.

**O que não se pode afirmar:**
- causa de uma ocorrência;
- efeito de obra;
- probabilidade anual;
- desempenho validado num bairro brasileiro específico;
- qualquer coisa sobre alagamento pluvial isolado.

## 9. Fila dirigida por lacuna

Arquivo: `local_runs/mecanismos-transferiveis/v1/fila_de_aquisicao.json`. Nada foi adquirido além do HydroBASINS de 4 regiões e da ocorrência JRC nos pontos (leitura remota de poucos bytes por ponto).

| # | Lacuna | Falha medida | Já no acervo | Fonte, resolução, licença | Custo | Teste que destrava |
|---|---|---|---|---|---|---|
| 1 | Suporte (enxurrada) | 0,718; 9 unidades | 278 ativações CEMS de inundação no catálogo salvo, não usadas | CEMS, 10–30 m, termos CEMS + FABDEM CC BY-NC-SA | Pequeno | Transferência por processo com ≥ 8 unidades de enxurrada |
| 2 | Representação física (restrição da saída) | Falhas planas | Nada sobre dique, canal ou maré | OSM (canal, dreno, dique), ODbL | Pequeno | Ablação de bloco nas unidades planas |
| 3 | Observação (pluvial) | Não testável | FloodNet: 1 território só | UrbanSARFloods (licença a conferir) | A medir | Transferência urbana com negativo observado |
| 4 | Unidade brasileira | Não testável | Instrumento em validação (obs04) | S1 com referência sem HAND | Validar antes | Destino brasileiro fora do componente SA |
| 5 | Unidade nova para confirmação | Tudo já lido | EMSR926, 927, 933 (3, nunca abertas) | CEMS | Pequeno | Confirmação de H1 com G congelado ao chegar a 8 |
| 6 | Modificador climático | Declive 14/13 sem explicação | — | HydroATLAS, CC BY 4.0 (conferir) | A medir | H2 com aridez/clima pré-declarados |

## 10. Reprodução

```bash
python scripts/metodo/mt01_bacias_globais.py --converter af as na au
python scripts/metodo/mt01_populacao.py montar
python scripts/metodo/mt01_independencia.py diagnosticar_tt01
python scripts/metodo/mt01_independencia.py unidades
python scripts/metodo/mt01_ocorrencia_jrc.py
python scripts/metodo/mt01_experimento.py executar
python scripts/metodo/mt01_experimento.py executar --sensibilidade jrc
python scripts/metodo/mt01_experimento.py executar --sensibilidade exutorio
python scripts/metodo/mt01_relatorio.py analisar
python scripts/metodo/mt01_relatorio.py verificar
python scripts/metodo/mt01_pos_hoc.py
python -m pytest tests/test_mt01_mecanismos.py -q
```

O pré-registro já existe. O executor retoma por destino e recusa rodar se um hash mudar; versão nova exige pasta nova.

## Decisões para a pesquisadora

| Decisão | Alternativas | Consequência |
|---|---|---|
| Refazer `cob01` sobre o ds06 v7 | (a) refazer como nova versão; (b) manter e declarar | (a) tira o atalho de versão do bloco "produção" e muda o ds06; (b) a produção continua não interpretável no CEMS |
| Abrir as 3 ativações novas (EMSR926/927/933) só ao chegar a 8 | (a) esperar e acumular; (b) abrir já como descritivo | (a) preserva uma confirmação futura de H1; (b) gasta as unidades sem poder inferir |
| Baixar UrbanSARFloods | Depende da licença | Única rota no acervo público para negativo urbano observado em vários territórios |

Nenhuma dessas decisões muda o serviço. A promoção do G no contrato exigiria confirmação em unidades novas e decisão humana.

## Nota de 10/10/2026 (MT02): revisão de inferências e da base

Revisão feita depois do [MT02](MECANISMOS_TRANSFERIVEIS_MT02_BASE_CONEXAO_SUPORTE_2026_10_10.md). Os resultados acima ficam como foram medidos; o que muda é a leitura.

- **"Transferível" é associação, não causa.** Uma relação aprendida em outros territórios e verificada fora deles não prova a causa de nenhuma ocorrência.
- **Não inferioridade de C/H não prova ausência de efeito do contexto.** Mostra só que os modificadores testados, na forma linear penalizada, não melhoraram a ordenação. O próprio MT02 mostra o contexto ordenando o *erro* entre territórios (relevo e distância ao mar).
- **Poucos rótulos sem ganho não tornam o dado local dispensável.** Rótulos de um evento não ajudaram a prever outro evento do mesmo território. A validação local continua necessária, e a pergunta sobre adaptação com mais eventos locais fica aberta.
- **A cobertura do MT01 tinha um defeito de versão**, corrigido na v2: os negativos CEMS sem cobertura tinham sido coletados sem ano. Com a base corrigida, G_v2 − G_v1 = +0,0008 (não inferior). As conclusões do MT01 se mantêm.
