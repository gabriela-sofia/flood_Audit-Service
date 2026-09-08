# Evidencia

Este arquivo resume o que esta afirmado pelos artefatos curados. Planejamento e poster
sao tratados como documentos de sintese; quando houver conflito, prevalece o artefato
versionado em `modelo/execucoes/` e o codigo que o reproduz.

Todo numero das tabelas abaixo e conferido contra o artefato citado pelo teste
`tests/test_documentacao_bate_com_artefato.py`. Numero que nao existir no artefato faz
a suite falhar.

## 1. Tabela unica

Fonte: `modelo/execucoes/ds-05-tabela-unica/manifesto_v2.json`.

| Item | Valor |
|---|---:|
| Observacoes de entrada | 595.865 |
| Observacoes admitidas | 568.265 |
| Pool fluvial | 65.070 |
| Grupos admitidos | 6.873 |
| Grupos no pool fluvial | 2.002 |

Composicao admitida por fonte:

| Fonte | Admitidos | Pool fluvial | Grupos admitidos |
|---|---:|---:|---:|
| NYC | 471.727 | 0 | 2.402 |
| Sen1Floods11 | 30.431 | 30.431 | 11 |
| CEMS | 25.483 | 25.483 | 119 |
| Urban Flood Observations | 24.044 | 0 | 215 |
| UK | 7.476 | 7.476 | 401 |
| Sao Paulo | 7.146 | 0 | 1.976 |
| Curitiba | 1.680 | 1.680 | 1.471 |
| Recife | 278 | 0 | 278 |

O manifesto tambem registra uma violacao de dominio aberta: a coluna
`unidade_agrupamento` recebe o valor `chamado` em 478.873 linhas, fora do dominio
declarado no esquema. Isso esta versionado no proprio manifesto e continua pendente de
revisao de contrato.

A cadeia foi rerodada em 08/09/2026 e reproduziu o manifesto byte a byte: os quatro
SHA-256 de `tabela_unica_v2.csv`, `tabela_unica_admitidos_v2.csv`,
`tabela_unica_pool_fluvial_v2.csv` e `rejeitados_com_motivo.csv` sao identicos aos
registrados aqui, e o unico campo que difere e `gerado_em`. Os artefatos ficaram como
estao: substitui-los trocaria um arquivo por outro identico e faria o historico parecer
que algo mudou.

Veredito: a tabela esta consolidada por manifesto publico e a consolidacao e
deterministica. As tabelas completas nao entram neste pacote por tamanho.

## 2. Modelos servidos

Fonte: `modelo/execucoes/svc-01-modelos/indice.json`.

| Modelo | n | Grupos | AUC | Servivel | Veredito |
|---|---:|---:|---:|---|---|
| `recife_pluvial` | 269 | 227 | 0,6422 | Sim | Fora dos criterios fortes, mas disponivel para Recife com maturidade local declarada. |
| `pluvial_urbano_nyc` | 471.727 | 751 | 0,6089 | Sim | Fora dos criterios fortes, mas disponivel para NYC. |
| `pluvial_urbano_sao_paulo` | 7.146 | 1.438 | 0,4753 | Nao | Abaixo do acaso; nao deve ser servido. |
| `fluvial_planicie` | 56.654 | 509 | 0,7334 | Sim | Coerente com criterios; usado para Curitiba. |
| `fluvial_serra` | 5.162 | 24 | 0,7916 | Sim | Coerente com criterios; usado para Petropolis. |

O numero de grupos importa tanto quanto a AUC. `fluvial_serra` tem 24 grupos, dos quais
19 positivos, e usa uma unica variavel (`hand_m`) exatamente por causa desse orcamento;
o artefato registra as duas leituras de EPV que sustentam a escolha. Ler 0,7916 sem ler
24 grupos e ler metade do resultado.

Veredito: Curitiba nao e descrita como modelo local validado. Ela e uma regiao servida
por transferencia caracterizada a partir do modelo fluvial de planicie.

## 3. Contrato de inferencia

Fonte: `modelo/execucoes/svc-02-contrato/`.

O contrato demonstra cobertura por regiao, portoes, maturidade e explicacao
estruturada. A decisao cientifica fica no modulo
`scripts/servico/svc02_contrato_inferencia.py`.

Veredito: o contrato esta implementado localmente como funcao pura. Transporte HTTP,
interface publica e operacao em tempo real nao sao afirmados por este pacote.

## 4. Grade de aplicacao

Fonte: `modelo/execucoes/svc-03-grade/resultado.json`.

| Regiao | Celulas validas | Serviveis | Serviveis/validas | Cenario | p50 | p90 |
|---|---:|---:|---:|---|---:|---:|
| Recife | 56.666 | 54.963 | 97,0% | mediana observada de chuva | 0,5808 | 0,7785 |
| Curitiba | 65.275 | 64.310 | 98,5% | unico, sem chuva | 0,2746 | 0,5804 |
| Petropolis | 172.015 | 157.005 | 91,3% | unico, sem chuva | 0,3460 | 0,5634 |

Os escores das tres linhas NAO sao comparaveis entre si. Sao tres modelos diferentes,
com vetores diferentes, sem calibracao comum. Em Recife o escore ainda depende do
cenario de chuva escolhido: sob a mediana observada o p50 e 0,5808; sob o p90 observado
sobe para 0,7323; sob o maximo observado vai a 0,9015 e a cobertura servivel cai de
97,0% para 79,9%. Comparar 0,5808 de Recife com 0,2746 de Curitiba mede a escolha de
modelo, nao o territorio.

### 4.1 Limitacoes que o proprio artefato declara

Estas linhas estao gravadas em `resultado.json`, por regiao, e sao parte do resultado.

**Recife**

- os negativos vem por ausencia de registro, o nivel mais fraco da hierarquia: 98,4% dos negativos sao fundo, e o escore so sustenta ordem;
- aderencia temporal alta: apenas 3,7% dos pontos estao em datas que contem as duas classes, e o modelo usa `rain_decay_index`. Nesta base, saber a data e quase saber a classe, e a chuva pode estar separando dias em vez de lugares;
- `hand_m` nao separa as classes em Recife: num evento pluvial o HAND nao e o mecanismo;
- `elevation_rel_m` e `slope_deg` saem com sinal contrario ao fisicamente esperado; entram no escore e ficam fora da frase de explicacao;
- o IC95 de `hand_m` cruza zero: o ajuste nao estabeleceu o efeito.

**Curitiba**

- `elevation_rel_m` e a elevacao contra o P1 da propria regiao; a absoluta saiu do vetor porque punha 0,0% dos pontos de Curitiba no dominio de treino, contra 94,5% da relativa;
- a chuva esta fora do conjunto: ela nao discrimina na escala em que este modelo compara;
- `slope_deg` sai com sinal contrario ao esperado e fica fora da frase de explicacao;
- o IC95 de `elevation_rel_m` cruza zero.

**Petropolis**

- nenhuma AOI brasileira entrou no ajuste: os grupos ingremes sao todos estrangeiros;
- Petropolis nao tem nenhuma linha na tabela unica: nenhum ponto rotulado, nenhum inventario local, nenhuma validacao local possivel hoje;
- o escore de Petropolis e predicao por semelhanca de terreno, nunca afirmacao de acerto.

Veredito: a grade publica existe em CSV, GeoTIFF e PNG para as tres regioes, e a
conferencia contra o contrato nao registrou divergencia em 25 celulas amostradas por
regiao. Isso valida a coerencia entre grade e contrato; nao valida acerto contra
inventario local, que nao existe em nenhuma das tres.

## 5. Chuva como forcamento de evento

Fonte: `modelo/execucoes/chuva-autopareamento/matriz_suporte_forcamento.json`.

Desenho autopareado no tempo (case-crossover): o estrato e o LUGAR, entao terreno,
drenagem e propensao daquele endereco a acionar a central saem da estimativa por
construcao. O desenho nao usa classe negativa: o caso e o proprio controle em outras
datas.

| Recorte | Razao de risco por 10 mm | IC 95% | Estratos informativos | Suporte |
|---|---:|---|---:|---|
| NYC | 1,5099 | 1,4645 a 1,5610 | 8.552 | `supported` |
| Sao Paulo | 1,2158 | 1,0470 a 1,4998 | 1.753 | `supported` |

Limitacoes declaradas: zero chamado nao e zero inundacao; a exposicao e aproximada em
celula de cerca de 11 km; o esforco de notificacao so esta controlado se for constante
dentro do estrato; chuva diaria nao mede intensidade intradiaria.

Veredito: a chuva antecedente tem suporte como forcamento de evento nos dois recortes
testados. Esse resultado nao promove sozinho nenhum modelo regional.

## 6. Terreno sob o gatilho, em Nova York

Fonte: `modelo/execucoes/nyc-terreno-sob-gatilho/terreno_sob_gatilho_1km.json`.

Com a chuva a 1 km (MRMS) ela passa a variar DENTRO do dia, e o estrato pode ser o dia
inteiro. Caso: chamado hidrico. Controle: chamado de outro assunto no mesmo dia.
Amostra: 381.919 pontos, 16.915 positivos, 376 dias informativos.

| Termo | Coeficiente | IC 95% | Sinal esperado | Aprendido |
|---|---:|---|---:|---|
| `twi_dinf` | +0,0390 | 0,0317 a 0,0472 | + | sim |
| `twi_dinf` x chuva | +0,0397 | 0,0221 a 0,0560 | + | sim |
| `slope_deg` | −0,0151 | −0,0184 a −0,0116 | − | sim |
| `elevation_rel_m` | −0,0088 | −0,0190 a −0,0002 | − | sim |
| `mrms_qpe_24h_mm` | +0,1886 | 0,1339 a 0,2221 | + | sim |
| `hand_m` | −0,0058 | −0,0136 a 0,0034 | − | nao |
| `hand_m` x chuva | +0,0022 | −0,0093 a 0,0105 | − | nao |

Veredito: em Nova York o TWI e o mecanismo, e o efeito dele cresce com a chuva. O HAND
e nulo. O proprio artefato declara que o MRMS cobre apenas os Estados Unidos e que este
resultado nao transfere sozinho para regiao brasileira.

## 7. O teste da segunda cidade: Chicago

Fonte: `modelo/execucoes/chicago-teste-da-lei/teste_da_lei_chicago.json`.

Enquanto o resultado da secao 6 existir num lugar so, nao ha como distinguir lei fisica
de particularidade de Nova York. Chicago responde exatamente essa pergunta, com o mesmo
estimador importado do modulo de Nova York — nao reescrito — para que uma diferenca de
resultado nao possa vir do codigo. Amostra: 496.972 pontos, 134.542 positivos, 2.438
dias informativos.

| Termo | NYC | Chicago | Mesmo sinal | ICs se sobrepoem | Aprendido nas duas |
|---|---:|---:|---|---|---|
| `twi_dinf` | +0,0390 | −0,0102 | **nao** | **nao** | nao |
| `twi_dinf` x chuva | +0,0397 | −0,0110 | **nao** | **nao** | nao |
| `elevation_rel_m` | −0,0088 | +0,0397 | **nao** | **nao** | nao |
| `hand_m` x chuva | +0,0022 | −0,0282 | nao | nao | nao |
| `slope_deg` | −0,0151 | −0,0111 | sim | sim | **sim** |
| `mrms_qpe_24h_mm` | +0,1886 | +0,4416 | sim | nao | **sim** |
| `hand_m` | −0,0058 | −0,0179 | sim | sim | nao |

Veredito, em tres partes:

1. **A chuva e a declividade replicam.** O forcamento aparece nas duas cidades com o
   sinal esperado e IC excluindo zero. Isso sustenta a secao 5 e a secao 6 na parte de
   chuva e de declive.
2. **O TWI e a interacao TWI x chuva INVERTEM.** O que era o mecanismo em Nova York sai
   com sinal oposto em Chicago, com IC que exclui zero e nao se sobrepoe ao de Nova
   York. Nao e ruido: e contradicao medida.
3. **`elevation_rel_m` tambem inverte.** Essa e a variavel que o metodo apresenta como
   a que permite comparar regioes diferentes. O unico teste de transferencia entre
   cidades que o projeto executou reprova o sinal dela na segunda cidade.

Chicago NAO pode responder sobre o HAND: o p95 do `hand_m` na cidade e 7,985 m contra
um limiar declarado de 10 m, entao quase toda a cidade esta no nivel da drenagem. Um
nulo de HAND aqui seria nulo por falta de contraste, que e outra coisa, e o artefato
declara isso.

O artefato tambem separa os dois tipos de chamado positivo, que tem significado fisico
diferente. Agua na via responde ao terreno; agua no porao responde muito menos:

| Termo | Agua na via | Agua no porao |
|---|---:|---:|
| `hand_m` | −0,0309, IC exclui zero | −0,0004, IC cruza zero |
| `elevation_rel_m` | −0,0262, sinal esperado | +0,0912, sinal invertido |
| `mrms_qpe_24h_mm` | +0,3998 | +0,4276 |
| `twi_dinf` | −0,0092, invertido | −0,0068, invertido |

O HAND separa as classes na via e nao separa no porao, que e o sentido fisico esperado:
altura acima da drenagem descreve escoamento superficial, nao refluxo de rede. A
inversao de `elevation_rel_m` e concentrada no porao. O corte por tipo nao existe no
artefato de Nova York deste pacote, entao a comparacao entre cidades por tipo nao esta
disponivel aqui.

### 7.1 O resultado defeituoso, mantido a proposito

O arquivo `teste_da_lei_chicago_defeito_centroide.json` e a versao anterior deste mesmo
teste, mantida no pacote porque a diferenca entre os dois e informacao. Dois defeitos de
amostragem foram corrigidos entre um e outro:

- o fundo cobria 78 dos 2.438 dias, porque o teto de linhas era global e ordenado por
  data; o estrato do desenho e o dia, e dia sem fundo nao informa. Passou a haver teto
  por dia, com posicao sorteada dentro do dia;
- boa parte do fundo caia em poucos centroides, porque o portal devolve coordenada de
  centroide quando nao tem o endereco exato.

A comparacao entre os dois arquivos mostra o que muda no resultado quando o desenho de
amostragem muda, e por isso o defeituoso fica versionado com o nome dizendo o que ele e.

### 7.2 O que Chicago muda no pacote

| Afirmacao | Estado depois de Chicago |
|---|---|
| Chuva antecedente aumenta risco notificado no mesmo lugar | Sustentada; replicou. |
| Declividade responde no sentido fisico esperado | Sustentada; replicou. |
| TWI e o mecanismo de acumulo em cidade | **Nao sustentada fora de Nova York.** |
| `elevation_rel_m` permite comparar regioes diferentes | **Nao sustentada; o sinal inverte.** |
| O modelo urbano de Nova York transfere para outra cidade | **Reprovada neste teste.** |
| Os modelos servidos mudam por causa de Chicago | Nao. Chicago nao ajusta modelo servido, nao promove variavel e nao altera portao. |

Chicago tambem nao autoriza sozinha a conclusao inversa: sao duas cidades dos Estados
Unidos, com centrais de atendimento e clima proprios, e nenhuma delas e brasileira.

## 8. Temporalidade

Fonte: `modelo/execucoes/temporalidade-integracao/scientific_support_matrix.json`.

| Hipotese | Estado |
|---|---|
| Curitiba, fluvial de planicie | `not_testable` no escopo declarado. |
| Dominio de ajuste, fluvial de planicie | `inconclusive`. |
| Petropolis, fluvial de serra | `not_testable` por falta de dado local. |
| Dominio de ajuste, fluvial de serra | `inconclusive`. |
| NYC pluvial urbano | `not_testable`. |
| Recife pluvial | `not_testable`. |
| UK, protocolo prospectivo | `inconclusive`. |
| Curitiba, diagnostico local | `inconclusive`. |

Veredito: a frente temporal nao deve ser apresentada como validacao prospectiva ampla.
Ela registra limites de identificabilidade.

## 9. Recuperacao de fonte

Fonte: `modelo/execucoes/recuperacao/comparacao_recuperadores_k5.json`, `k=5`, 984 casos.

| Recuperador | Recall agregado | MRR agregado | Recall em morfologia |
|---|---:|---:|---:|
| Lexical FTS5 | 0,7043 | 0,6886 | 0,1189 |
| Denso LSA | 0,9675 | 0,8808 | 0,9482 |
| Hibrido RRF | 0,8872 | 0,7678 | 0,6768 |

Veredito: o denso ganha no agregado e ganha muito em consulta morfologica, onde o
lexical acerta 0,1189. O hibrido fica entre os dois nas tres medidas. Este componente
serve para rastreabilidade de fontes; ele nao altera portao, peso ou escore.

## 10. Cadeia seletiva

Fonte: `modelo/execucoes/neural-catraca/experimento_pote_crescente_redecidido.json`.

O experimento confirmatorio registra `n=6`, ganho mediano de `0,119623` e efeito
detectado por testes de postos e sinal, com as 6 sementes no mesmo sentido. O proprio
artefato limita a interpretacao: trata-se de reconstrucao mascarada em recortes
reservados, nao de deteccao direta de agua. O veredito parametrico do mesmo artefato e
`INDECIDIVEL`, e as duas leituras ficam registradas lado a lado.

## 11. Predicao seletiva das frentes auxiliares

Fonte: `modelo/execucoes/sintese-neural/predicao_seletiva_das_frentes.json`.

As frentes auxiliares medem coisas incomparaveis entre si — erro continuo de mascara,
correlacao por pais, acerto binario de recuperacao. A regua que as torna comparaveis nao
e o desempenho: e a pergunta "a confianca desta frente ordena melhor que o acaso DELA
mesma?" (El-Yaniv e Wiener, 2010). A medida e AURC, area sob a curva risco-cobertura,
onde menor e melhor.

| Frente | Seletor | n | AURC | AURC do acaso (p05) | Ordena melhor que o acaso |
|---|---|---:|---:|---:|---|
| A | rede em SAR | 79 | 0,5604 | 0,6841 | sim |
| B | representacao de terreno | 10 | 0,4507 | 0,3981 | **nao** |
| C | recuperacao densa | 984 | 0,0113 | 0,0247 | sim |
| D lexico | decisao lexica | 18 | 0,4029 | 0,3492 | **nao** |
| D conjuncao | lexico e juiz em conjuncao | 18 | 0,1085 | 0,1266 | sim |

Leitura obrigatoria junto da tabela: **AURC de frentes diferentes nao se compara**. O
erro da frente A e continuo e o da frente C e binario; a comparacao valida e de cada
frente com o proprio acaso, nunca entre elas. A frente B tem 10 paises e a D tem 18 casos, entao
as curvas tem poucos pontos, o intervalo do acaso e largo, e a leitura correta e
qualitativa. A frente D mostra o efeito da conjuncao: o seletor lexico sozinho nao
ordena melhor que o acaso, e a conjuncao com o juiz ordena.

Veredito: onde `ordena_melhor_que_o_acaso` e falso, a confianca daquela frente nao
carrega informacao, e qualquer ganho aparente de cobertura parcial e ilusao. O artefato
declara isso por frente, em vez de reportar so o numero bom.

## 12. Cadeia seletiva

Fonte: `modelo/execucoes/sintese-neural/cadeia_seletiva.json`.

A cadeia encadeia quatro elos — observacao, representacao, recuperacao e verificacao —
e so publica a frase quando todos passam. Quando um elo nao sustenta, a cadeia **se
abstem e nomeia o elo onde parou**.

| Item | Valor |
|---|---:|
| Casos | 18 |
| Publicados | 12 |
| Cobertura | 0,6667 |
| Acuracia da decisao | 0,4444 |
| Abstencoes no elo de verificacao | 6 |

Duas leituras que o proprio artefato impoe:

1. **Abstencao nao e resposta negativa.** A cadeia diz que nao sustenta seguir, e nomeia
   onde parou. Contar abstencao como erro mediria outra coisa.
2. **Elo inaplicavel produz abstencao por dado irrelevante.** Observacao e representacao
   pertencem ao eixo fisico; uma frase sobre metodologia nao passa por eles, e uma
   cadeia com elos inaplicaveis se abstem sem que isso signifique falha do metodo.

Com acuracia de decisao em 0,4444 sobre 18 casos, esta e uma demonstracao de
comportamento seletivo, nao uma medida de desempenho operacional.

## 13. Camada social territorial

Fonte: `modelo/execucoes/camada-social-territorial/ibge-curitiba/manifesto_ibge_curitiba_v1.json`.

Curitiba possui 3.190 setores na malha, 3.190 com unidade social e 254 em FCU no
recorte auditado. A camada social v0 e de revisao: nao altera escore fisico, nao treina
modelo, nao cria rotulo hidrologico e nao permite concluir baixa vulnerabilidade quando
faltam dados.
