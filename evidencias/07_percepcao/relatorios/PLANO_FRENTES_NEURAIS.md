# REV-P — plano das frentes neurais

Documento de acompanhamento, revisado a cada sessão. Diferente dos outros
documentos de `docs/`, **este não entra no corpus RAG**: ele descreve trabalho
em andamento, muda a cada sessão e não é verdade sobre o estado do projeto até
que cada frente feche. Por isso também não está no índice de precedência
documental — o guardrail cobra registro de quem está no corpus, e este não está.

Marque `[x]` conforme fechar; `[~]` é etapa cujo código está pronto e
testado, mas que ainda depende de uma execução para produzir o número.
Cada frente é uma sessão.

---

## Por que estas quatro frentes, e nesta ordem

O objeto que aprende não é a LLM. Fine-tuning na camada de explicação move
conhecimento para dentro dos pesos, e a pergunta que o REV-P existe para
responder — *qual evidência sustenta esta afirmação?* — deixa de ter resposta
localizável.

O que aprende é o **fenômeno**. A LLM continua sendo a camada que não pode
mentir sobre o que a rede aprendeu.

A ordem não é arbitrária. A frente A produz observação; a B produz
representação de terreno e reaproveita a esteira que a A vai validar; C e D são
independentes das duas e melhoram a camada cognitiva sem tocar em nenhum portão
científico. Nenhuma bloqueia a outra — mas A antes de B porque A é a que
justifica cientificamente o resto.

---

## Regras que valem para as quatro

Estas não são recomendações. Elas são o que separa "o REV-P aprendeu" de "o
REV-P passou a ter uma rede dentro".

1. **Nenhuma saída neural entra em `variaveis_usadas` sem o orçamento de EPV**
   e a trava de sinal físico que valem para toda variável do contrato. A rota
   DINOv2 foi congelada exatamente por essa disciplina, e a disciplina é o
   resultado — não o obstáculo.
2. **Nenhum portão de `svc01`–`svc04` consulta uma rede** enquanto a frente não
   fechar com portão próprio, revisado e declarado.
3. **A rede é objeto de avaliação, não infraestrutura de entrega.** Toda frente
   entrega um número contra um baseline declarado, incluindo o caso em que ela
   perde.
4. **Peso que muda exige proveniência de peso**: hash do checkpoint, hash do
   dado de treino, seed, split, e o commit que produziu. `proveniencia.py` já
   tem o vocabulário (`TREINADO_COM`, `GERADO_POR`); não inventar outro.
5. **Erro da rede é recusa, não silêncio.** Uma segmentação que não confia em
   si mesma devolve `nao_identificavel`, nunca "sem água".
6. **Review-only até decisão humana.** Nenhuma frente promove nada
   automaticamente, pelo mesmo critério do SVC-07.

---

## Frente A — aprendizado de inundação em SAR

**Sessão 1.** A rota mais forte, e a que muda o que conta como observação.

### O que já está no disco

`local_runs/n1a-sen1floods11/` — 1,7 GB, CC BY 4.0, natureza declarada
`OBSERVADO`, adquirido em 2026-08-07. Status atual registrado na proveniência:
*"nenhuma feature extraida; nenhum modelo treinado."*

| item | valor |
|---|---|
| chips com rótulo manual | 446 |
| split oficial | 252 treino / 89 validação / 90 teste |
| holdout fora de região | 15 (Bolívia) |
| `S1Hand` | 2 bandas (VV, VH) em dB, float32, 512×512, EPSG:4326 |
| `LabelHand` | int16, `{-1 sem dado, 0 não-água, 1 água}` |
| `S1OtsuLabelHand` | **o baseline Otsu, já pronto, nos mesmos chips** |
| `JRCWaterHand` | água permanente — separa inundação de corpo d'água |
| ambiente | torch 2.11.0+cpu, sem CUDA |

Dois achados que decidem o desenho da frente:

- **O baseline já veio junto.** `S1OtsuLabelHand` é o mesmo threshold que o
  REV-P usa hoje para footprint. A comparação rede × Otsu roda no mesmo test
  set, sem construir comparador nenhum.
- **`JRCWaterHand` resolve a confusão que mais importa.** O `ds03` já distingue
  classe 9 (água permanente) de classe 1 (inundação); o dataset traz a máscara
  para fazer a mesma distinção no pixel.

### Etapas

- [x] **A1. Leitor e contrato de dado.** `scripts/sar/dados_sen1floods11.py`.
      `ler_rotulo` devolve **sempre** `(rotulo, valido)` — a assinatura é a
      barreira, porque uma função que devolvesse só o rótulo tornaria
      `rotulo == 1` a leitura óbvia, e essa leitura junta `-1` com `0`.
      Verificado nos 446 chips: todos 512×512, EPSG:4326, cinco produtos
      presentes, rótulo só com `{-1, 0, 1}`.
- [x] **A2. Baseline Otsu medido.** `scripts/sar/baseline_otsu.py`, split de
      teste (90 chips), publicado **antes** de existir rede. Resultado abaixo.
- [x] **A3. Split honesto.** Splits oficiais, sem reamostrar; disjunção
      verificada em vez de assumida (252/89/90/15). A Bolívia está trancada —
      inclusive nos testes, que rodam em `validacao`.
- [x] **A4. Modelo e esteira.** `scripts/sar/modelo_unet.py` e
      `scripts/sar/treino.py`. U-Net de 3 níveis, `base=16`, **482.610
      parâmetros**, saída em 2 canais com `CrossEntropyLoss(ignore_index=-1)`.
      Portão de reprodução fechado: mesma seed → mesmo `hash_estado`; seed
      diferente → hash diferente. Treino de fumaça gravado (2 épocas, 12 chips):
      perda 0,9062 → 0,7378, validação 0,6793 → 0,6589.
      **Falta o treino longo** — ver o comando abaixo.
- [x] **A5. Métrica contra o baseline.** Rodada com o peso de 30 épocas.
      Resultado misto, publicado abaixo com o sinal que deu.
- [x] **A6. Holdout fora de região (Bolívia).** Rodado uma única vez, depois
      de A5. Não degradou nas métricas sem água permanente — mas a leitura não é
      "transferiu bem": ver abaixo.
- [x] **A7. Recusa.** Calibrada na validação, e a primeira medida de confiança
      **falhou** — ver abaixo. Corrigida para medir a margem só onde a rede se
      compromete; a segunda curva discrimina.
- [x] **A8. Proveniência do peso.** O manifesto de treino carrega
      `hash_estado`, `hash_manifesto`, seed, split, normalização, parâmetros,
      versão do torch e o commit. Segundos de execução **não** entram no hash —
      um teste pegou esse defeito: dois treinos idênticos davam manifestos
      diferentes, e a diferença era a máquina estar mais ou menos ocupada.
- [x] **A9. Model card da rede.** `scripts/sar/model_card.py`, **gerado dos
      artefatos** e nunca escrito à mão. Campo sem artefato sai `AUSENTE` em vez
      de sumir. O limite que ele carrega está abaixo.

### O limite geográfico, medido (A9)

| split | países |
|---|---|
| treino | Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri-Lanka, USA |
| validação | *os mesmos dez* |
| teste | *os mesmos dez* |
| bolivia | Bolivia |

**Nenhum chip do Brasil.** O país mais próximo em continente e regime é o
Paraguai.

E o detalhe mais silencioso: treino, validação e teste repartem os **mesmos**
dez países. O split oficial mede generalização para cenas novas, **não** para
regiões novas. Um IoU alto no teste responde *"a rede aprendeu água em SAR?"* —
jamais *"a rede vale em Recife?"*. É por isso que A6 existe, e por isso a
Bolívia não pode aparecer em nenhum outro split (verificado em teste).

### O baseline, medido (A2 — split de teste, 90 chips)

| métrica | com água permanente | **sem água permanente** |
|---|---:|---:|
| IoU micro | 0,5458 | **0,3947** |
| F1 micro | 0,7062 | 0,5660 |
| IoU macro | 0,3274 | **0,2532** |
| F1 macro | 0,4155 | 0,3456 |

Máscara: 13,04% dos pixels fora por rótulo `-1`; 0% de abstenção do Otsu.
Hash do resumo: `sha256:d5f2cf3d…`. Artefatos em `local_runs/sar-inundacao/`.

**O achado que muda o discurso da frente.** Os verdadeiros-positivos do Otsu
caem de 1.881.424 para 993.023 quando se removem os pixels de água permanente:
**47% do que o baseline "acerta" é rio e lago que já estavam lá antes da
enchente.** Um método de limiar acha corpo d'água com facilidade, e acertar água
permanente é acertar a pergunta errada — a pergunta é inundação.

Isso é diretamente sobre o REV-P, porque o footprint SAR do projeto usa esse
mesmo tipo de método. E dá à frente A um alvo mais honesto que "superar 0,5458":
o número a bater é **0,3947 micro / 0,2532 macro**, e a rede tem `JRCWaterHand`
para aprender a distinção que o limiar não faz.

Segundo achado: micro (0,5458) é muito maior que macro (0,3274). O baseline é
carregado por chips com muita água e vai mal onde a lâmina é pequena — que é
justamente o caso urbano do REV-P.

### O treino longo (A4, pendente)

Ritmo medido nesta máquina, `torch 2.11.0+cpu`, 8 threads, sem GPU utilizável:
**~143 s de treino + ~80 s de validação por época** com os 252 chips. Ou seja
~3,7 min/época — 20 épocas em ~1h15, 40 em ~2h30.

```bash
python scripts/sar/treino.py --epocas 30 --lado 256 --lote 4 --threads 8 --gravar --rotulo v1
```

Três decisões do treino que estão no código e mudam o resultado:

1. **Normalização só com o treino.** Média e desvio de VV/VH saem exclusivamente
   do split de treino e vão gravados no checkpoint. Calcular sobre o conjunto
   todo seria vazamento, e vazamento não aparece em métrica nenhuma.
2. **NaN não é zero e não é predizível.** Pixel sem leitura de S1 vira a média
   do treino (zero após normalizar) **e** entra no ignore, junto com o `-1` do
   rótulo. Preencher sem ignorar ensinaria a rede que ausência de leitura parece
   água média.
3. **Recorte de 256 no treino, chip inteiro na avaliação.** O baseline foi
   medido em chip inteiro; comparar números medidos em unidades diferentes é a
   forma mais silenciosa de vencer uma comparação.

### O resultado (A5 — teste, 90 chips, peso de 30 épocas)

| IoU | rede | Otsu | diferença |
|---|---:|---:|---:|
| micro | 0,5285 | 0,5458 | −0,017 |
| **micro sem água permanente** | **0,4489** | **0,3947** | **+0,054** |
| macro | 0,3064 | 0,3274 | −0,021 |
| macro sem água permanente | 0,2407 | 0,2532 | −0,013 |

**Misto: ganha só no corte que interessa, perde nos outros três.** Sem promoção.

Estratificando por quanta água o chip tem — e isto **contradiz** o que o macro
sugeria:

| água no chip | n | rede | Otsu | delta |
|---|---:|---:|---:|---:|
| <1% | 25 | 0,0566 | 0,0417 | **+0,015** |
| 1–5% | 28 | 0,2452 | 0,2309 | **+0,014** |
| 5–20% | 17 | 0,3015 | 0,3698 | −0,068 |
| >20% | 15 | 0,4861 | 0,5318 | −0,046 |

A rede ganha nos 53 chips de **pouca** água e perde nos 32 de muita — o caso
urbano é lâmina pequena, então isto é a favor do REV-P. O macro parecia dizer o
oposto porque é dominado pela *magnitude* do IoU, não pela contagem. Ressalva
que não dá para omitir: em valor absoluto os dois métodos são ruins ali
(0,057 e 0,042); o ganho é relativo sobre base péssima.

### Transferência (A6 — Bolívia, 15 chips)

| | teste | Bolívia |
|---|---:|---:|
| IoU micro s/perm | 0,4489 | 0,4622 |
| precisão | 0,838 | 0,939 |
| recall | 0,491 | 0,477 |

Não degradou. Mas a leitura honesta não é "transferiu bem": precisão sobe para
0,94 e o recall fica em 0,48 — **a rede transferiu o conservadorismo**, não
necessariamente a competência. Com 15 chips, a incerteza não sustenta mais que
isso.

### A recusa, e o defeito que ela revelou (A7)

A primeira medida de confiança foi a margem média sobre **todos** os pixels
válidos, e a curva saiu plana: todos os limiares de 0,05 a 0,50 recusavam os
mesmos 3 chips, com ganho de IoU **0,0**. O critério imprimiu a frase que já
estava escrita para esse caso — *"recusar sem melhorar significa que a confiança
da rede não separa o que deveria"*.

O defeito: ~87% do chip é fundo óbvio que a rede acerta com probabilidade quase
zero, e **o fundo domina a média**. Num chip onde a rede prediz água com p=0,55,
a medida antiga dá 0,87 ("confiante") e a corrigida dá 0,10 ("em dúvida"). A
correção mede a margem só onde a rede **se compromete**.

Segunda curva, na validação, com orçamento de cobertura de 10% declarado antes:

| limiar | recusados | custo | IoU s/perm |
|---:|---:|---:|---:|
| 0,00 | 0 | 0,000 | 0,4212 |
| **0,05** | **1** | **0,011** | **0,4212** |
| 0,10 | 10 | 0,112 | 0,4379 |
| 0,20 | 26 | 0,292 | 0,4545 |
| 0,40 | 52 | 0,584 | 0,6355 |

A curva agora **discrimina** — recusar mais melhora o IoU de forma consistente.
Mas dentro do orçamento o escolhido é 0,05, com ganho ~0. O ponto seguinte
(0,10, ganho +0,017) fica 1,2 ponto percentual fora do orçamento, e **o
orçamento não foi mexido depois de ver a curva** — mexer seria a maquiagem que
o critério existe para impedir. E o IoU 0,6355 do limiar 0,40 custa recusar 58%
dos chips: é o número mais bonito da tabela e o mais desonesto de citar sozinho.

Conclusão da recusa, **corrigida em 2026-09-02**: esta leitura estava errada, e
o erro era do critério, não da rede.

Eu medi ganho de IoU num único ponto de cobertura, o que é comparar duas curvas
por um pixel — e nunca perguntei se a confiança **ordenava** melhor que o acaso.
Sob risco-cobertura (El-Yaniv e Wiener 2010, já no corpus do projeto):

| | AURC | acaso (média) | acaso (p05) | ordena? |
|---|---:|---:|---:|:--:|
| A — rede SAR | **0,5604** | 0,7284 | 0,6841 | **sim** |

A confiança da rede **carrega informação real**: recusando na ordem que ela
propõe, o risco em área cai de 0,738 para 0,560. O que não pagava era o
orçamento fixo de 10%, não o mecanismo.

### Critério de aceite da frente

> Dado um par VV/VH, o REV-P produz máscara de água com incerteza declarada,
> mede-se contra Otsu no mesmo conjunto, relata a queda fora de região e recusa
> quando não sabe — sem que nada disso toque no escore de suscetibilidade.

### O que esta frente NÃO faz

Não vira feature do modelo físico. Não substitui o footprint servido antes de
decisão humana. Não promove nenhum negativo do Sen1Floods11 para o dataset SUSC
— a proveniência do dado proíbe isso explicitamente, e a proibição continua
valendo depois do treino.

---

## Frente B — embeddings de terreno

**Sessão 2.** Reaproveita a esteira do DINO e ataca o gargalo diagnosticado.

### O que já existe, e o que o resultado negativo ensinou

`scripts/dino/` — 92 scripts, rota **congelada**. DINOv2 congelado, 768
dimensões, testado como feature em três comparações A/B independentes. Nulo nas
três. E o diagnóstico é a parte que importa:

> O gargalo medido não estava no embedding: estava na granularidade do ponto
> SEDEC.

Isso não diz que embedding não serve aqui. Diz que **nenhuma rede salva um
rótulo pontual**, e que embedding de imagem óptica não é a representação que o
fenômeno usa.

A esteira, porém, é boa e é reaproveitável inteira: manifesto de entrada →
preflight de ativo → extração → QA → análise estrutural → índice de evidência →
rastreador de proveniência. Isso é infraestrutura pronta.

### Etapas

- [x] **B1. Trocar o objeto, manter a esteira.** `scripts/terreno/dados_terreno.py`.
      A pilha vem do `ter-01`, que **já tinha terreno derivado para os 446
      chips** do Sen1Floods11 — cobertura de 100% nos quatro splits. Cinco
      camadas a 30 m: DEM preenchido, declividade, HAND D-∞, SCA e acumulação.
- [x] **B2. Encoder auto-supervisionado.** Autoencoder **mascarado** (blocos de
      16×16, 50%), 106.197 parâmetros, 40 épocas, embedding de 128 dimensões.
      Nenhum rótulo entra no treino — verificado em teste, por AST. Resultado
      abaixo.
- [x] **B3. Portão de reprodução.** `scripts/neural/proveniencia_de_peso.py`,
      extraído da frente A e agora usado pelas duas. Hash de estado, seed,
      commit, e **sem tempo de parede**.
- [ ] **B4. Uso estrutural primeiro.** k-NN, medoids, outliers, ordenação de
      fila de revisão — exatamente o que o DINO ficou autorizado a fazer.
      Critério: nada desloca escore.
- [ ] **B5. Teste A/B como feature, com o mesmo rigor de antes.** Critério:
      orçamento de EPV declarado antes de rodar, e resultado publicado com o
      sinal que der. Um segundo nulo é um segundo resultado.
- [ ] **B6. Comparação honesta com a rota congelada.** Se o embedding de terreno
      também der nulo, a conclusão fica mais forte, não mais fraca: o gargalo é
      do rótulo, e isso passa a ter duas evidências independentes.

### O baseline, medido antes do encoder (B2-baseline)

25 estatísticas descritivas, nenhum parâmetro aprendido, leave-one-country-out
contra a fração de água do mesmo rótulo manual da frente A:

**Spearman global 0,3821 · mediano entre países 0,4575**

E ele já respondeu algo: Mekong 0,70, USA 0,70, India 0,60 — contra Sri-Lanka
0,02, Ghana 0,07. A forma do terreno ordena onde a água ficou em alguns regimes
e **não ordena nada** em outros.

### O resultado do encoder (B2)

| | encoder | baseline | dif |
|---|---:|---:|---:|
| Spearman global | 0,3948 | 0,3821 | +0,013 |
| Spearman mediano entre países | 0,4554 | 0,4575 | −0,002 |

No agregado, **empate**. Mas o quadro por país é o achado:

| país | encoder | baseline | dif |
|---|---:|---:|---:|
| Sri-Lanka | 0,5201 | 0,0241 | **+0,496** |
| Ghana | 0,4100 | 0,0688 | **+0,341** |
| Nigeria | 0,7143 | 0,4945 | +0,220 |
| Somalia | 0,6045 | 0,4406 | +0,164 |
| Paraguay | 0,6224 | 0,4744 | +0,148 |
| Pakistan | 0,3540 | 0,3394 | +0,015 |
| Spain | 0,1565 | 0,1635 | −0,007 |
| India | 0,4291 | 0,5979 | −0,169 |
| Mekong | 0,4817 | 0,7035 | −0,222 |
| USA | 0,1476 | 0,7007 | **−0,553** |

**O encoder conserta exatamente onde o baseline falhava e quebra exatamente
onde ele ia bem.** Essa é a assinatura de duas representações
**complementares**, não de uma melhor que a outra: a estatística global
descreve "quanto mais baixo e plano, mais água", que basta em grande planície
(USA, Mekong) e não diz nada em terreno complexo (Sri-Lanka, Ghana).

### O que faltava: dado sem rótulo (v2, 2026-09-02)

O encoder v1 treinou em 252 chips rotulados. Mas o `ter-01` tinha **215 chips a
mais**, com as cinco camadas de terreno e **sem rótulo nenhum** — de 14 eventos
que não aparecem em split algum (Beira, Houston, Katmandu, Nova Gales do Sul,
Salt Lake, Quebec). Colisão com treino/teste/holdout: **zero**.

Auto-supervisão é a única família de aprendizado em que **dado sem rótulo é dado
útil**: o encoder aprende reconstruindo o que foi escondido, e para isso o rótulo
nunca fez falta.

Treino de **252 → 467 chips**, 40 épocas, nenhum rótulo, nenhuma curadoria:

| variante | global | mediano |
|---|---:|---:|
| baseline descritivo (25 dims) | 0,3821 | 0,4575 |
| encoder v1 (252 chips) | 0,3948 | 0,4554 |
| v1 + descritivo | 0,4835 | 0,4519 |
| **encoder v2 (467, +215 sem rótulo)** | **0,5274** | **0,5338** |
| v2 + descritivo | 0,5188 | 0,4632 |

**+0,133 sobre o v1 e +0,145 sobre o baseline** — mas ver a correção em B4
abaixo: **boa parte desse ganho era de mais passos de gradiente, não de mais
dado**. Com o controle correto, o dado sozinho vale entre +0,055 e +0,101.

Três leituras que continuam de pé:

1. **O mediano finalmente subiu** (0,4554 → 0,5338). O v1 empatava no mediano —
   o ganho dele estava só nos extremos. O v2 melhora de forma distribuída.
2. **O v2 sozinho supera o combinado do v1** (0,5274 contra 0,4835), e o
   combinado do v2 **piora** (0,5188): a representação aprendida já absorveu o
   que as estatísticas descritivas tinham a dizer.
3. **O USA voltou** — era o pior caso do v1 (0,1476, queda de −0,55 contra o
   baseline) e foi para 0,4937 (**+0,346**). Espanha: 0,1565 → 0,3817.

Não foi ganho em todo lugar: Somália caiu −0,40 e Ghana −0,21.

**A consequência prática:** cada raster novo que entrar no `ter-01` melhora a
representação **sem rótulo, sem curadoria e sem decisão humana**. Os pesos mudam
por gradiente, e a melhora é medida no mesmo protocolo com teste e holdout
intocados.

### B4 — a curva de escala, e a correção do número acima

O v1 e o v2 treinaram os dois por **40 épocas**. Mas um conjunto 1,85× maior
recebe 1,85× mais passos de gradiente por época — então aquele resultado
confundia **mais dado** com **mais treino**.

A curva de escala fixa o orçamento de passos em 1200 para todas as corridas. A
única coisa que muda é a diversidade do dado:

| fração | chips | passos | perda treino | Spearman global |
|---:|---:|---:|---:|---:|
| 0,00 | 252 | 1197 | 0,6999 | 0,3899 |
| 0,33 | 323 | 1200 | 0,6654 | **0,4653** |
| 0,66 | 394 | 1188 | 0,6222 | **0,4912** |
| 1,00 | 467 | 1170 | 0,6118 | 0,4451 |

| | 252 chips | 467 chips | ganho |
|---|---:|---:|---:|
| 40 épocas, passos livres | 0,3948 | 0,5274 | +0,133 |
| **1200 passos fixos** | 0,3899 | 0,4451 | **+0,055** |

**O que fica de pé:** a perda de treino cai monotonicamente (0,6999 → 0,6118) —
mais dado melhora a tarefa auto-supervisionada sem exceção. E **todos** os
pontos com dado extra superam o de 252 chips.

**O que não fica:** o ganho de +0,133 anunciado no v2. Com o controle, o dado
sozinho vale entre +0,055 e +0,101.

**O que ficou em aberto:** a curva não é monótona no Spearman — pico em 0,66 e
queda em 1,00. Com 1200 passos e uma única seed a variância é alta, então isso
pode ser ruído ou pode ser real. Rodando com mais seeds para separar.

### A combinação (exploratório — a hipótese nasceu do resultado)

Concatenando os dois vetores (153 dimensões):

| | combinado | baseline | dif |
|---|---:|---:|---:|
| Spearman global | **0,4835** | 0,3821 | **+0,102** |
| mediano entre países | 0,4519 | 0,4575 | −0,006 |

Oito dos dez países melhoram; só USA (−0,274) e Mekong (−0,070) perdem, e o USA
se recupera de 0,148 para 0,427. **O mediano não move porque o mediano é
robusto a extremos, e o ganho está justamente nos extremos ruins.**

Isto é exploratório e está declarado como tal no código: a hipótese veio dos
dados e roda no mesmo conjunto em que apareceu. O split de teste e o holdout
continuam intocados, e é só neles que a combinação poderia virar afirmação.

### Uma limitação do desenho, encontrada durante a corrida

O split de validação foi usado para **duas** coisas: monitorar o treino do
encoder e compor o conjunto de avaliação LOCO. Por isso o checkpoint usado é o
da **época 40, sem seleção** — escolher a época pelo mínimo da validação seria
selecionar com dados que depois entram na avaliação. A curva mostra sobreajuste
a partir da época ~20 (treino 0,684→0,602, validação 0,696→0,628 com
oscilação), então há desempenho sendo deixado na mesa em troca de um número que
significa o que diz.

### Critério de aceite da frente

> Existe uma representação aprendida do terreno, reprodutível por hash, usada em
> revisão estrutural, e testada como feature sob o mesmo orçamento de EPV que
> reprovou o DINOv2 — com o resultado publicado nos dois sentidos.

---

## Frente C — recuperação neural no RAG

**Sessão 3.** Independente das outras duas. Barata, e conserta um defeito medido.

### O defeito, medido

A busca é FTS5, lexical. Em 2026-09-01, no corpus real:

```
"favela comunidade urbana"      → 0 trechos
"favelas e comunidades urbanas" → a fonte certa
```

Falta de stemming. A fonte existe, está indexada, e a pergunta no singular não
a alcança. Num corpus que decide o que a LLM pode citar, isso é recuperação
falhando em silêncio.

### Etapas

- [x] **C1. Conjunto de avaliação.** 984 casos, gerados do próprio corpus em
      três níveis (literal, morfológico, lacunar) — sem anotação humana e sem
      circularidade. Baseline FTS5 medido antes de existir alternativa.
- [~] **C2. Índice denso local.** **Não é bi-encoder neural** — o Ollama local
      está sem `--embeddings` e nenhum modelo de sentença está em disco. É
      TF-IDF de n-gramas de **caractere** + SVD: denso, determinístico,
      offline, aprendido do corpus. Nomear melhor do que é seria o erro que o
      projeto persegue na LLM.
- [x] **C3. Híbrido por RRF.** Fusão por posição recíproca, `c=60` não ajustado
      aos dados. Resultado abaixo — e ele não é o que eu esperava.
- [x] **C4. Classe de autoridade continua mandando.** Verificado em teste para
      as três classes.
- [x] **C5. Detecção de injeção.** **Um buraco que eu abri e fechei** — ver
      abaixo.
- [x] **C6. Ganho publicado**, estratificado por nível e por classe.

### O baseline, e o defeito quantificado (C1)

984 casos, k=5:

| nível | recall@5 | MRR |
|---|---:|---:|
| literal | 0,9970 | 0,9812 |
| **morfológica** | **0,1189** | 0,0908 |
| lacunar | 0,9970 | 0,9939 |

**O agregado é 0,7043 e esconde tudo.** O FTS5 é quase perfeito no texto exato e
em termos de conteúdo soltos, e falha em 88% das consultas quando singular e
plural trocam. Essa é a medida do que, em 2026-09-01, apareceu como
`"favela comunidade urbana"` → 0 trechos.

Olhar só o agregado teria produzido "o FTS5 vai bem" — exatamente o erro que a
frente B ensinou a não cometer.

### O resultado (C3/C6)

| recuperador | agregado | literal | morfológica | lacunar |
|---|---:|---:|---:|---:|
| lexical FTS5 | 0,7043 | 0,9970 | 0,1189 | 0,9970 |
| **denso LSA** | **0,9675** | 0,9665 | **0,9482** | 0,9878 |
| híbrido RRF | 0,8872 | 0,9878 | 0,6768 | 0,9970 |

O denso resolve a morfologia: **+0,829**. E o híbrido, que eu esperava ser o
melhor dos dois, fica **atrás do denso puro** na morfologia (0,677 contra
0,948) — porque nas consultas morfológicas o lexical devolve lixo, e o RRF soma
esse lixo, empurrando o documento certo para baixo.

Fundir duas listas quando uma delas está errada não produz o melhor das duas.

### O caso que decidiria contra o denso, verificado

A cadeia de citação depende de literalidade: o identificador tem de aparecer
**literalmente** na fonte. Se o denso perdesse DOIs, o ganho não compensaria.

Testado nos 29 chunks com DOI, consultando o DOI exato:
**lexical 29/29 · denso 29/29 · híbrido 29/29.**

N-grama de caractere preserva identificador por construção. O medo não se
confirmou.

### O buraco que eu abri (C5)

A primeira versão do índice denso lia os `chunk_id` direto do SQLite e devolvia
o vizinho mais próximo — **sem passar pelo `texto_tem_injecao`** que o
`rag_busca` aplica na recuperação. Seria um caminho novo pelo qual um chunk que
fala em imperativo com o sistema chega ao contexto, contornando uma barreira
que existe desde a camada cognitiva.

Nada quebraria, a métrica melhoraria, e uma trava deixaria de valer. É o modo
de falha mais silencioso que uma frente de infraestrutura pode introduzir.

Fechado, com teste que consulta o próprio texto do chunk envenenado — o pior
caso, porque ele seria o vizinho mais próximo de si mesmo.

### Critério de aceite da frente

> A recuperação melhora num conjunto de avaliação declarado, sem que nenhum
> chunk citável passe a existir sem hash, classe e verificação.

---

## Frente D — verificação neural de suporte

**Sessão 4.** A última, e depende de C estar de pé. Estudar antes de implementar.

### A lacuna, declarada pelo próprio código

`sustenta()` é sobreposição de termos de conteúdo, e o arquivo diz o limite:

> pega fonte fora de assunto, não pega paráfrase habilidosa.

Essa é a quarta condição da cadeia de citação — a que costuma faltar. Uma
citação pode ter fonte real, DOI conferido, e o artigo não dizer aquilo.

### Etapas

- [x] **D1. Casos de paráfrase habilidosa.** 18 pares em quatro famílias,
      trechos **reais** do corpus e claims escritos à mão — não há como gerar
      automaticamente um par cujo léxico casa e o sentido não. Baseline medido.
- [x] **D2. Juiz local.** `qwen2.5:3b-instruct` via Ollama, temperatura 0,
      latência mediana 1,94 s. Vê **apenas** `(claim, trecho)` — nunca o payload
      nem o que foi gerado.
- [x] **D3. Conjunção.** O juiz **só subtrai**. Resultado abaixo.
- [~] **D4. O verificador é verificado.** Dez testes cobrem a assimetria e os
      dois piores casos do juiz. Falta plugá-lo no REV-P-EVAL como motor
      comparável.
- [x] **D5. Limite declarado.** `custo_declarado` e `regra` saem em toda
      medida; o custo em falso negativo é medido, não estimado.

### O baseline léxico, medido (D1)

18 casos, `claims.sustenta` como está servindo:

| família | n | acurácia | FP | FN |
|---|---:|---:|---:|---:|
| sustenta_direto | 4 | 1,00 | 0 | 0 |
| sustenta_parafrase | 4 | 0,25 | 0 | 3 |
| **nao_sustenta_lexico** | 7 | **0,00** | **7** | 0 |
| nao_sustenta_obvio | 3 | 1,00 | 0 | 0 |
| **geral** | 18 | **0,4444** | 7 | 3 |

A docstring de `sustenta` já dizia: *"pega o caso que importa e deixa passar
paráfrase habilidosa"*. Agora está medido — **0% na família da paráfrase
habilidosa**, e os 7 falsos positivos passariam.

### O resultado da conjunção (D3)

| família | léxico | conjunção | FP | FN |
|---|---:|---:|---|---|
| sustenta_direto | 1,00 | 1,00 | 0→0 | 0→0 |
| sustenta_parafrase | 0,25 | **0,00** | 0→0 | 3→**4** |
| nao_sustenta_lexico | 0,00 | **0,857** | **7→1** | 0→0 |
| nao_sustenta_obvio | 1,00 | 1,00 | 0→0 | 0→0 |
| **geral** | 0,4444 | **0,7222** | 7→1 | 3→4 |

**Corta 6 dos 7 falsos positivos e custa 1 falso negativo.** É o trade-off
previsto, agora com números: o juiz é conservador demais, e o desenho garante
que o conservadorismo só produz silêncio.

### A assimetria que quebra a circularidade

Usar um modelo de linguagem para verificar o que um modelo de linguagem
escreveu é circular por natureza. O que quebra a circularidade:

```text
publica  ⟺  a regra léxica aprova  E  o juiz não discorda
```

O juiz nunca vê o payload nem o que foi gerado — só o par `(claim, trecho)`,
verificado por assinatura em teste. E ele **só pode subtrair**.

Isso torna o pior caso inofensivo, e os dois piores casos estão em teste:

- juiz alucina "sustenta" para tudo → o sistema volta a ser o que era;
- juiz alucina "não sustenta" para tudo → publica menos, nunca errado;
- **juiz fora do ar → não derruba nada**, mesma regra da proveniência ausente.

A frente C reforçou por que conjunção e não fusão: lá, fundir duas listas
quando uma estava errada deu resultado **pior** que usar só a certa. Conjunção
não mistura — exige acordo.

### O custo, declarado em vez de escondido

A conjunção só aperta: **não conserta falso negativo**. A fonte certa que o
léxico reprova por falta de sobreposição continua reprovada, e o juiz não pode
salvá-la. A família `sustenta_parafrase` existe para medir esse custo — e ele
piorou (3→4).

No REV-P, publicar de menos é recuperável e publicar de mais não é. Mas isso é
uma escolha, e escolha declarada é diferente de efeito colateral.

### Critério de aceite da frente

> A quarta condição da cadeia de citação passa a pegar paráfrase, sem que o
> modelo neural possa **autorizar** sozinho nenhuma publicação.

### O risco a decidir antes de começar

Um verificador neural que erra para o lado permissivo é pior que nenhum, porque
dá confiança onde não há. Por isso D3 é conjunção e não votação, e por isso esta
frente vem depois — ela precisa que C tenha estabelecido como rodar um modelo
local com reprodutibilidade verificada.

---

---

## Síntese — as quatro na mesma régua

Cada frente foi medida com uma régua diferente (IoU, Spearman, recall@k,
acurácia), e isso escondeu o que elas têm em comum: **as quatro são seletores**.
Todas podem se abster, e um seletor não se avalia pela qualidade média do que
deixa passar — essa métrica premia recusar tudo menos o caso mais fácil.

A régua comum é a de El-Yaniv e Wiener (2010), que já estava indexada como
`cientifica` no corpus: risco contra cobertura, medido por AURC, sempre contra
o acaso da própria frente.

| frente | n | risco sem abstenção | AURC | acaso p05 | ordena melhor que o acaso? |
|---|---:|---:|---:|---:|:--:|
| A — rede SAR | 79 | 0,7380 | 0,5604 | 0,6841 | **sim** |
| B — embedding terreno | 10 | 0,5063 | 0,4507 | 0,3981 | não |
| C — recuperação densa | 984 | 0,0335 | **0,0113** | 0,0247 | **sim** |
| D — léxico sozinho | 18 | 0,5556 | 0,4029 | 0,3492 | **não** |
| D — conjunção | 18 | 0,2778 | 0,1085 | 0,1266 | **sim** |

Três coisas que só apareceram nesta régua:

1. **A frente A funciona como seletor** — e minha conclusão anterior estava
   errada, pelo critério e não pelo resultado.
2. **A sobreposição léxica sozinha NÃO ordena** (0,4029 dentro do intervalo do
   acaso). Isto é uma justificação da frente D independente da acurácia: o juiz
   não melhora um seletor bom, ele **cria** um seletor onde não havia.
3. **B não ordena** — e a razão é honesta: foi a única frente desenhada sem
   abstenção nativa, e a confiança que usei (tamanho da amostra) é fraca. Isso
   é uma lacuna de desenho, não um resultado sobre terreno.

AURC não se compara **entre** frentes: o erro de A é contínuo e o de C é
binário. A comparação válida é de cada frente com o seu próprio acaso.

## Registro de sessões

Uma linha por sessão fechada, com o commit local.

| sessão | frente | o que fechou | commit |
|---|---|---|---|
| 0 | — | encaixe do payload social, hierarquia da ausência, 4 ataques sociais no EVAL | `975c212` |
| 1 | A | **frente A completa** — A1 a A9, treino de 30 épocas, resultado misto sem promoção | `5846f96`, `80b5b9b`, `4a0713e`, `3409435`, `918502a`, `1709ad1` |
| 2 | B | B1–B3 + **v2 auto-supervisionado**: 215 chips sem rótulo levam 0,3948 → **0,5274** | `6e7f9a4`, `d447d67`, `2d62e47`, `0abcdb2` |
| 3 | C | C1–C6: defeito morfológico quantificado (0,119), denso resolve (0,948), híbrido RRF fica atrás; filtro de injeção herdado | `25b17bb` |
| 4 | D | D1–D3, D5: léxico 0,4444 → conjunção 0,7222; 7 FP → 1, custo 1 FN; juiz só subtrai | `2c868f7` |

## Como fazer o check comigo

No início de cada sessão, me peça o estado desta lista. Eu leio o documento, o
`git log` e os artefatos em `local_runs/`, e digo o que está feito, o que está
pela metade e o que mudou desde a última vez — sem confiar no que este arquivo
afirma, porque um `[x]` é uma alegação e o artefato é a evidência.
