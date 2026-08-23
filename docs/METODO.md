# Método

Como o serviço funciona e por que funciona assim. Reúne a premissa física, o vocabulário
do projeto, o que funcionou e o que foi descartado, como o contrato de inferência decide
e as métricas oficiais.

A procedência de cada número — data, script, artefato, veredito — está em
[`EVIDENCIA.md`](EVIDENCIA.md). O modelo servido está descrito em
[`../modelo/MODELO.md`](../modelo/MODELO.md) e a base em
[`../modelo/DADOS.md`](../modelo/DADOS.md).

Estado em 22/08/2026.

---

## 1. Síntese da linha causal (SUSC-20)

### 1.1 A premissa física

Suscetibilidade, aqui, é a predisposição do terreno a acumular água sob um dado
forçamento de chuva. Há enchente quando chega a um ponto mais água do que ele consegue
escoar ou armazenar. Isso se decompõe em três grandezas, e cada uma vira uma variável:

| Pergunta física | Variável | O que ela mede |
|---|---|---|
| Quanto chega? | Chuva antecedente | O forçamento — quanta água caiu na janela que importa |
| Para onde converge? | `twi_dinf` (índice topográfico de umidade) | Quanta encosta drena para aquela célula |
| Quanto precisa subir para ser alcançado? | `hand_m` (Height Above the Nearest Drainage) | A altura do ponto sobre o curso d'água que o drena — a lâmina que o rio precisa ganhar para chegar nele |

TWI e HAND exigem saber por onde a água desce. O projeto usa **D-infinity**, que reparte
o escoamento de cada célula entre direções contínuas em vez de despejar tudo em uma das
oito vizinhas. Isso importa em terreno urbano de baixa declividade, onde a discretização
em oito direções cria caminhos de drenagem que não existem.

O sinal de cada variável é fixado **antes** de rodar: HAND negativo (quanto mais alto
sobre a drenagem, menos suscetível), TWI positivo (quanto mais encosta converge, mais
suscetível). Coeficiente com sinal invertido é falha de mecanismo, não resultado.

### 1.2 A cadeia é única entre regiões

Toda a base passa pela mesma cadeia de derivação de terreno, na mesma resolução, mesmo
onde isso custa detalhe que o modelo não usa. Nenhuma das quatro fontes do ajuste
fluvial depende mais de HAND ou TWI vindos de produto genérico global: todas foram
rederivadas pela cadeia D-infinity do projeto. É o que torna a variável comparável entre
fontes — sem isso, a diferença entre Recife e o piloto inglês poderia ser diferença de
produto, não de território.

A reprodução foi conferida bit a bit contra o raster de referência em Recife e em
Curitiba. Petrópolis segue a mesma convenção, mas não tem ponto rotulado para comparar.

### 1.3 A chuva é de fonte única

A chuva é a essência física do fenômeno e a variável mais cara de medir de forma
comparável entre fontes. As datas de evento de Copernicus EMS, Sen1Floods11 e UFO
estavam em artefato local, em três formatos que nunca tinham sido lidos juntos.
Recuperadas, a cobertura de chuva passou de **14% para 99,99%** da base harmonizada, em
**produto único** (Open-Meteo/ERA5-Land). O CHIRPS foi o produto que a auditoria de
confundimento retirou.

Mas há um limite que o projeto declara em vez de contornar: a chuva é medida em células
de **~11 km**, enquanto o modelo compara pontos dentro do mesmo evento. Nessa escala ela
desloca o escore e **não muda o ordenamento**. Por isso ela entra no serviço como
*cenário*, não como camada.

### 1.4 A base

Tabela única harmonizada com **65.070 pontos elegíveis ao ajuste fluvial**, reduzidos a
partir de seis fontes. Positivos vêm de registro oficial geocodificado; o negativo é
tratado em três níveis declarados por linha:

| Nível | Fonte | Volume |
|---|---|---|
| **Observado** | Copernicus EMS | 25.249 pontos em 119 AOIs, mais a ativação EMSR720 no Rio Grande do Sul (216,55 km², proporção 5,94:1) |
| **Exclusão qualificada** | Environment Agency/UK | 7.476 pontos (3.738 / 3.738) em 201 eventos independentes |
| **Exclusão qualificada** | SIAC 156/Curitiba | 114 pontos antes classificados por ausência |
| **Ausência de registro** | Recife e Petrópolis | — |

Positivos: SEDEC/Recife (154) e SIAC 156/Curitiba (1.045), mais Diário Oficial e bases
internacionais.

Ausência de registro **não é ausência de evento**. O registro oficial nasce de quem
consegue reportar e ser atendido; a documentação disponível reflete a assistência
prestada, não a extensão real da água.

### 1.5 As onze etapas

| Etapa | O que entrega |
|---|---|
| `susc_20a` | Aquisição de evento real em Recife (SEDEC, ANA, Diário Oficial, bases internacionais) |
| `susc_20b` | Engenharia das features físico-hidrológicas |
| `susc_20c` | Modelagem e validação estatística rigorosa (Firth, v12) |
| `susc_20d` | Motor de inferência local |
| `susc_20e` | Contrato de API por região |
| `susc_20f` | Geoprocessamento sob demanda |
| `susc_20g` | HAND e TWI por D-infinity, genéricos por região |
| `susc_20h` | Candidatos de água por Sentinel-2 |
| `susc_20i` | Janelas de evento 2023–2026 |
| `susc_20j` | Candidatos de água por Sentinel-1 (SAR) |
| `susc_20k` | Curitiba: candidatos, negativos, features, modelagem e diagnósticos |

---

## 2. Mapeamento de turning points

### 2.1 O que funcionou

**Regressão logística penalizada de Firth como rota primária.** Escolhida por reduzir
viés em amostra pequena e entregar coeficiente com intervalo e direção física
verificável — não por desempenho. Num problema de evento raro, com poucos positivos e
exigência de defender mecanismo, um coeficiente interpretável com IC vale mais que um
ponto a mais de AUC.

**Validação agrupada por evento.** A unidade de informação é o evento, não o ponto. Isso
dissolveu significância que era pseudorreplicação: pontos do mesmo evento não são
observações independentes, e tratá-los como tal inflava a precisão. É a mesma unidade
que o produto devolve — o contrato responde por área, não por pixel —, e validar no
mesmo grão em que se entrega é o que torna o resultado auditável.

**Piso de eventos por preditor (EPV) verificado antes de cada ajuste.** O estrato íngreme
tem 19 eventos positivos e comporta **uma** variável, não quatro. A primeira versão do
modelo de serra usou duas; a conclusão sobreviveu, mas os números mudaram e o texto
passou a citar os novos. O orçamento de EPV vale para as duas classes, não só para a
rara.

**Rederivação única da cadeia de terreno.** Ver §1.2. Sem isso, transferência entre
regiões não seria interpretável.

**Recuperação da chuva.** De 14% para 99,99% de cobertura, em produto único — e a
descoberta, no mesmo movimento, de que ela não discrimina na escala do modelo (§1.3).

**A relação terreno–inundação não caduca.** Em janela expansiva de 25 anos, com a mesma
rota linear e as mesmas variáveis, o modelo não colapsa. Isso vale para um país e para
negativo por exclusão qualificada — não prova estabilidade em serra tropical.

### 2.2 O que falhou e foi descartado

**Variável derivada do rótulo.** Produz circularidade pura: a acurácia balanceada cai de
**0,8855 para 0,4834** quando essas variáveis saem. O número alto não era o modelo
funcionando, era o modelo lendo a resposta. Rota fechada.

**DINOv2 como preditor.** O codificador visual é congelado — nunca ajustado aos dados do
projeto. Composição estática de imagem orbital não carrega assinatura de evento pontual,
e isso foi testado em **três tentativas independentes**, todas fechando sem sinal. Numa
delas o teste formal de razão de verossimilhança deu significativo à primeira vista, mas
a correção de pseudorreplicação por patch mostrou que o sinal era artefato de amostra.
A rota está encerrada como candidata a variável. O DINOv2 permanece com um papel
delimitado e útil: medir similaridade entre áreas e **alimentar a fila de revisão
humana** — que é o código preservado em `scripts/dino/`.

**Modelo próprio de Curitiba como rota operacional.** O ajuste apresenta AUC 0,65 sob
validação embaralhada e colapsa para **0,52** em holdout temporal real de 2026. Sete
diagnósticos independentes descartaram vazamento espacial, sazonalidade, ruído de
amostra, deriva administrativa e correlação com El Niño/La Niña. Um GBM monotônico causal
confirmou não linearidade real no fenômeno, mas não resolveu a generalização temporal. O
oitavo diagnóstico não achou a causa: descartou a última explicação metodológica — o
holdout inglês mostra que o colapso não é propriedade do método — e nomeou o limite
amostral, **114 negativos contra 1.238 positivos**. Resultado negativo publicado, não
escondido.

**O ajuste de maior desempenho próprio é o que menos transfere.** A definição da classe
negativa decide o que o modelo aprende, e isso afeta o AUC obtido mais do que o fenômeno
em si. Por isso o negativo é declarado em três níveis e reportado em proporção sempre que
um modelo os mistura.

**Não linear como rota primária.** GBM e GAM entram como diagnóstico de não linearidade,
nunca como modelo de produção. Confirmam que há curvatura no fenômeno; não entregam
coeficiente defensável.

---

## 3. Decisão e funcionamento dos portões

O contrato de inferência roda como **função pura e auditável** — não é um servidor HTTP,
e isso é escolha. O que precisa ser auditado é o contrato: quais portões existem, em que
ordem, o que cada recusa significa e como o escore é construído. Falta só o transporte.

### 3.1 A ordem importa

Cada portão só é avaliado se o anterior fechou, e a resposta nomeia o **primeiro** que
falhou. Recusar por "faltou HAND" quando a geometria nem era válida esconderia o erro
real de quem chamou.

| # | Portão | O que verifica | A falha devolve |
|---|---|---|---|
| G1 | `geometria_valida` | Ao menos um ponto, CRS suportado, coordenadas finitas e dentro do globo | `insufficient_data` |
| G2 | `regiao_resolvida` | Região declarada, ou deduzida pela caixa envolvente das regiões que existem na base | `region_not_supported` |
| G3 | `modelo_para_a_regiao` | Existe modelo servível mapeado para a região | `region_not_supported` |
| G4 | `variaveis_presentes` | Toda variável do modelo presente e finita em todo ponto | `insufficient_data` |
| G5 | `dominio_coberto` | Quantas variáveis caem na faixa de 5–95% que o modelo viu | `insufficient_data` acima do limite |

### 3.2 Por que G5 existe

G5 é a novidade metodológica. Ele nasceu de uma medida: em Curitiba, **0% dos pontos**
caem na faixa de `elevation_m` do treino, a uma diferença padronizada de 2,76 desvios nos
pontos rotulados — e 5,05 desvios quando se mede sobre o território inteiro. Aquilo era
uma recomendação num documento; aqui virou portão avaliado a cada requisição.

Abaixo do limite de extrapolação, a variável entra como limitação declarada. Acima dele,
o serviço **recusa em vez de extrapolar**. O mesmo princípio governa a grade: célula fora
do domínio fica vazia no mapa em vez de receber escore baixo — recusar não é o mesmo que
dizer que ali é seguro.

### 3.3 Maturidade da região

O escore vem sempre acompanhado do quanto se pode confiar nele.

| Maturidade | Significa |
|---|---|
| `validado` | Modelo próprio, sinais corretos, IC das causais sem cruzar zero, AUC na faixa |
| `mvp_local` | Modelo próprio da região, mas algum critério de leitura não atingido |
| `transferencia_caracterizada` | A região não entrou no treino; declara-se distância de domínio, não acerto contra rótulo local |
| `transferencia_sem_referencia_local` | Não há nem inventário local para comparar; escore por semelhança de terreno, nunca afirmação de acerto |
| `region_not_supported` | Geometria que não cai em região alguma com modelo |

**Nenhuma região está em `validado`.** A região-alvo não entra no treino do modelo que a
serve: o propósito é prever onde não há inventário local, e validar contra o rótulo da
própria região seria a pergunta errada. Um teste guarda esse invariante.

### 3.4 O intervalo de confiança, e por que este

**Bootstrap do preditor linear, reamostrando grupos.** Os grupos do conjunto de ajuste
são reamostrados com reposição N=1000 vezes; cada reamostra reajusta o Firth; as réplicas
de coeficiente ficam gravadas no artefato servido. O IC de um escore é o percentil das N
projeções.

Não se usou o *delta method* porque ele depende da aproximação assintótica do erro-padrão
— discutivelmente frágil exatamente no regime de n pequeno que motivou usar Firth. Usar
Firth por causa do n pequeno e depois propagar incerteza por aproximação assintótica seria
incoerente com a própria escolha do estimador.

Reamostra-se grupo e não linha porque reamostrar linha infla a precisão por
pseudorreplicação: o IC sairia estreito demais, que é o modo de falha mais perigoso num
número que vai dentro de uma resposta de API.

### 3.5 A explicação não pode divergir do escore

A camada de explicação é gerada por regras sobre a contribuição que **já entrou** no
escore. Divergir entre explicação e resposta é impossível por construção, não por
verificação posterior.

### 3.6 A decisão que fica exposta

Quando o modelo da região existe mas não atinge o critério de leitura, há duas posturas
defensáveis, e uma constante visível escolhe entre elas — em vez de a escolha ficar
enterrada numa condição:

- **`declara`** (padrão): devolve o escore com a falha escrita nas limitações e a
  maturidade rebaixada. Coerente com "resultado negativo é publicado" — esconder o escore
  esconderia também o quanto ele é fraco.
- **`recusa`**: devolve `insufficient_data`. Coerente com falha fechada estrita.

---

## 4. Métricas oficiais de validação

### 4.1 E3 — Ajuste por classe de relevo

| Ajuste | AUC agrupada | Coeficientes (IC95) |
|---|---:|---|
| Serra | **0,7916** | `hand_m` −1,44 [−3,11; −0,83] |
| Planície | **0,7245** | `hand_m` −2,10 [−2,78; −1,56]; `twi_dinf` +0,40 [+0,33; +0,45] |
| Planície aplicada à serra | **0,7957** | — |

O modelo de planície aplicado à serra supera o que a serra alcança sozinha. A leitura não
é que a serra seja mais fácil: a relação é a mesma nos dois terrenos, e o que separa é
**quão bem cada um está estimado**. O estrato íngreme tem 19 eventos positivos e comporta
uma variável.

### 4.2 E4 — Holdout temporal

| Item | Valor |
|---|---|
| Janela | 201 eventos em 110 datas, entre 2000 e 2025 |
| Cortes | 8, todos na faixa 0,70–0,88 fixada antes de rodar |
| AUC médio | **0,7992** |
| Incerteza | IC95 por bootstrap de grupos em cada corte |
| Achado colateral | Curitiba não sustenta o teste — 114 negativos contra 1.238 positivos |

O corte temporal se aplica ao evento, que tem data; o negativo, amostrado por bloco
espacial, entra por sorteio declarado. O resultado refuta que o colapso temporal seja
propriedade do método; **não** prova estabilidade em serra tropical.

### 4.3 E5 e E6 — Grade e maturidade por região

Grade a 120 m, derivada da cadeia de terreno já existente, sem aquisição nova:

| Região | Células | Maturidade | O que a resposta carrega |
|---|---:|---|---|
| **Recife** | 56.666 | `mvp_local` | Modelo próprio (Firth v12, n=278: 154 positivos / 124 negativos, LOO-AUC 0,68) com o critério de leitura não atingido escrito na própria resposta |
| **Curitiba** | 65.275 | `transferencia_caracterizada` | Extrapolação de `elevation_m` declarada: 5,05 desvios do domínio de ajuste, nenhuma célula na faixa vista |
| **Petrópolis** | 172.015 | `transferencia_sem_referencia_local` | Escore por semelhança de terreno; 91,3% do território cabe na faixa de HAND do modelo de serra |

## 5. Protocolo C — como o projeto decide o que conta como evidência

O Protocolo C é a camada de auditoria que tentou estabelecer referência de campo
operacional em Recife a partir de Sentinel-2 pré/pós-evento, fontes oficiais e
embeddings visuais. Ele não está neste repositório como código — foi retirado da árvore
de trabalho na curadoria de entrega e permanece no histórico do repositório de origem.
Mas o **resultado** dele governa tudo o que veio depois, e por isso está aqui.

O que ele mediu: 2.654 patches avaliados, **zero** datas de produto Sentinel
confirmadas; 330 fontes escaneadas, 22 candidatos liberados, **nenhum** confirmado por
fonte institucional adquirida; 12 vínculos evento-patch, todos contextuais ou
temporalmente bloqueados.

A escala de níveis que ele deixou:

| Nível | Significa | Alcançado |
|---|---|---:|
| C1 | Contextual — evidência territorial documentada | 2 |
| C2 | Somente revisão — representação visual sem rótulo | 2 |
| C3 | Evento vinculado a patch, com data confirmada | 0 |
| C3+ | Temporal **e** espacial confirmados | 0 |
| C4 | Rótulo operacional — exige negativo formal explícito | 0 |

**Por que isso importa para o serviço.** O gate `C4_BLOCKED_NO_FORMAL_NEGATIVES` segue
aberto para Recife, Curitiba e Petrópolis. É daí que vem a hierarquia de três níveis do
negativo descrita em §1.4: como não existe negativo formal nas regiões brasileiras, o
projeto usa negativo **observado** (Copernicus EMS), **por exclusão qualificada**
(Environment Agency e 114 pontos de Curitiba) e declara **ausência de registro** onde não
tem nenhum dos dois. Sem o Protocolo C, essa hierarquia pareceria escolha arbitrária;
com ele, é consequência de uma auditoria que falhou de forma documentada.

O Protocolo C também fixou os guardrails que o código ainda carrega: nenhum artefato
pode marcar `ground_truth=true`, `can_train_model=true` ou
`can_create_operational_label=true`. O script `scripts/dino/revp_v1px_dino_queue_leakage_audit.py`
existe só para conferir isso a cada execução.

---

## 6. Glossário

O projeto tem vocabulário próprio, e ele aparece dentro do código, não só na prosa.

| Termo | O que significa aqui |
|---|---|
| **REV-P** | Código interno do projeto, herdado dos prefixos `revp_` nos arquivos e da nomenclatura de estágios (`SUSC-20A`, `MOD-SERRA-03`, `v1pv`). Não é um acrônimo com expansão definida |
| **Linha causal** | A sequência SUSC-20, de `20a` a `20k`: aquisição de evento real, features, modelo, motor, contrato. É o produto |
| **Patch** | Recorte espacial padronizado usado como unidade de análise visual, identificado por `REC_00205`, `CUR_00038`, `PET_00016` |
| **Ajuste fluvial** | O ajuste que estima suscetibilidade a **enchente**, e não a outro mecanismo — como o movimento de massa que também aparece em Petrópolis |
| **Unidade de validação** | O **evento**, ou a AOI, nunca o ponto. Pontos do mesmo evento não são observações independentes |
| **EPV** | Eventos por preditor. Piso que decide quantas variáveis um estrato comporta. Vale para as duas classes, não só para a rara |
| **Negativo observado** | Área que um analista examinou e onde não detectou inundação |
| **Exclusão qualificada** | Área excluída por critério explícito (os quatro critérios N1–N4), não por falta de registro |
| **Ausência de registro** | Nem observação nem exclusão: só não há dado. **Não é negativo** |
| **Maturidade** | O quanto se pode confiar no escore de uma região — ver §3.3 |
| **Fail-closed** | Diante de dado ausente ou ambíguo, o pipeline recusa e escreve o motivo, em vez de assumir um valor |
| **Review-only** | Artefato que serve à revisão humana e nunca alimenta treino nem vira rótulo. É o estatuto do DINOv2 no projeto |
| **Nível C1–C4** | A escala do Protocolo C — ver §5 |

---

## 7. Mapa do código

| Pasta | Papel | Arquivos |
|---|---|---:|
| `scripts/treino/` | Monta a tabela única (`ds03`→`ds05`), ajusta por classe de relevo (E3), roda o holdout temporal (E4), audita a escala da chuva | 9 |
| `scripts/servico/` | `svc01` treina e serializa os modelos servidos; `svc02` é o contrato com os cinco portões; `svc03` gera a grade (E5) | 3 |
| `scripts/terreno/` | `ter01`–`ter06`: a cadeia D-infinity que torna HAND e TWI comparáveis entre as seis fontes | 6 |
| `scripts/externo/` | Aquisição do negativo externo, em cinco estágios — ver abaixo | 29 |
| `scripts/dino/` | Fila de revisão visual. Nada aqui entra no modelo | 15 |
| `outputs_public/data/susc_20*/` | As onze etapas da linha causal, cada uma com script, dado curado e relatório | 191 |
| `tests/` | Regressão: 23 arquivos da linha causal, 7 do treino e do serviço | 30 |

A frente externa está organizada por estágio:

| Estágio | Conteúdo |
|---|---|
| `externo/comum/` | Armazenamento GeoParquet com poda por bbox e utilitários de aquisição |
| `externo/aquisicao/` | Sen1Floods11, UFO, Copernicus EMS e Global Flood Database — as quatro fontes do Nível 1 |
| `externo/cems/` | Seleção de ativações por analogia de relevo, fila de aquisição, download dos pacotes e o negativo observado do EMSR720 |
| `externo/uk/` | Piloto inglês: Recorded Flood Outlines, AOI, flood zones, contabilidade do negativo, WorldCover e as quatro etapas de feature |
| `externo/pontos/` | Redução de polígono a ponto e extração de features topográficas |

**Uma diferença que vale saber.** O `gates.py` de `susc_20e` é a versão anterior do
contrato: ele avalia geometria, CRS, região, modelo e variáveis — **sem o portão de
domínio**. O contrato atual, com os cinco portões da §3.1, está em
`scripts/servico/svc02_contrato_inferencia.py`. O `20e` fica como registro da etapa em
que a API foi desenhada.

---

## 8. Ambiente

São dois ambientes, de propósito: o modelo causal não carrega `torch`.

```bash
conda env create -f environment.yml
conda activate revp-susc
python -m pytest tests -q
```

| Ambiente | Para quê | Restrição |
|---|---|---|
| `environment.yml` | Linha causal, treino e serviço | Python `<3.11`, exigência do estimador de Firth |
| `requirements.txt` | Camada DINOv2 | Python padrão |

Parte dos testes lê artefatos de execução grandes demais para versionar — a tabela única
tem 189 MB e a grade por célula, 19 MB. Esses testes **pulam** em vez de falhar, e a
mensagem do skip traz o comando exato que os regenera. Os artefatos pequenos estão
versionados em `modelo/execucoes/`, e os testes os encontram automaticamente quando
`local_runs/` não existe.

---

## 9. Referências de método

1. Tellman, B. *et al.* Satellite imaging reveals increased proportion of population exposed to floods. *Nature*, 2021.
2. Nobre, A. D. *et al.* HAND, a new terrain descriptor using SRTM-DEM. *Journal of Hydrology*, 2011.
3. Beven, K. J.; Kirkby, M. J. A physically based, variable contributing area model of basin hydrology. *Hydrological Sciences Bulletin*, 1979.
4. Tarboton, D. G. A new method for the determination of flow directions and upslope areas in grid digital elevation models. *Water Resources Research*, 1997.
5. Firth, D. Bias reduction of maximum likelihood estimates. *Biometrika*, 1993.
6. Peduzzi, P. *et al.* A simulation study of the number of events per variable in logistic regression analysis. *Journal of Clinical Epidemiology*, 1996.
7. Muñoz-Sabater, J. *et al.* ERA5-Land: a global reanalysis dataset. *Earth System Science Data*, 2021.
8. Abnar, S.; Zuidema, W. Quantifying attention flow in transformers. *ACL*, 2020.
