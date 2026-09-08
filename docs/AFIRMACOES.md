# O que pode e o que nao pode ser afirmado

Este arquivo existe para que a comunicacao publica do projeto nao ande mais rapido que a
evidencia dele. Cada afirmacao sustentada aponta para o artefato que a sustenta; cada
afirmacao vedada aponta para o motivo.

## 1. Afirmacoes sustentadas

| # | Afirmacao | Artefato |
|---|---|---|
| 1 | O projeto implementa uma arquitetura de inferencia de suscetibilidade fisica com contrato explicito, que preserva recusa e maturidade regional. | `scripts/servico/svc02_contrato_inferencia.py`, `modelo/execucoes/svc-02-contrato/` |
| 2 | A tabela unica v2 harmoniza 595.865 entradas e admite 568.265 linhas apos curadoria, com 8 fontes declaradas. | `ds-05-tabela-unica/manifesto_v2.json` |
| 3 | O modelo `fluvial_planicie` tem AUC curada 0,7334 e serve Curitiba como transferencia caracterizada. | `svc-01-modelos/indice.json` |
| 4 | O modelo `fluvial_serra` tem AUC curada 0,7916 sobre 24 grupos e serve Petropolis como transferencia sem referencia local. | `svc-01-modelos/fluvial_serra.json` |
| 5 | O modelo `recife_pluvial` e servido como MVP local com AUC curada 0,6422 e veredito `FORA_DOS_CRITERIOS` no proprio indice. | `svc-01-modelos/recife_pluvial.json` |
| 6 | O candidato pluvial urbano de Sao Paulo ficou abaixo do acaso (0,4753) e nao e servido. | `svc-01-modelos/pluvial_urbano_sao_paulo.json` |
| 7 | A grade de 120 m foi gerada para as tres regioes e conferida contra o contrato em 25 celulas por regiao, sem divergencia. | `svc-03-grade/resultado.json` |
| 8 | Chuva antecedente tem associacao positiva com ocorrencia notificada no mesmo lugar em NYC (1,5099) e Sao Paulo (1,2158), por desenho autopareado no tempo. | `chuva-autopareamento/matriz_suporte_forcamento.json` |
| 9 | Em Nova York, com a chuva a 1 km, o TWI e a interacao TWI x chuva sao aprendidos, e o HAND e nulo. | `nyc-terreno-sob-gatilho/terreno_sob_gatilho_1km.json` |
| 10 | O mesmo estimador aplicado a Chicago replica chuva e declividade e **inverte** TWI, TWI x chuva e elevacao relativa. | `chicago-teste-da-lei/teste_da_lei_chicago.json` |
| 11 | A camada social de Curitiba foi implementada com IBGE Censo 2022 e nao altera o escore fisico. | `camada-social-territorial/ibge-curitiba/manifesto_ibge_curitiba_v1.json` |
| 12 | O recuperador denso melhora cobertura semantica (0,9675 agregado, 0,9482 em morfologia) enquanto o lexical permanece util para citacao exata (0,9970 em literal). | `recuperacao/comparacao_recuperadores_k5.json` |
| 13 | O experimento pre-registrado de pote crescente detectou efeito por teste de postos, com 6/6 sementes no mesmo sentido e p = 0,03125. | `neural-catraca/experimento_pote_crescente_redecidido.json` |
| 14 | Duas das quatro frentes seletivas ordenam melhor que o proprio acaso; uma nao ordena. | `sintese-neural/predicao_seletiva_das_frentes.json` |
| 15 | A temporalidade possui apenas estados `inconclusive` e `not_testable`; nao ha validacao prospectiva ampla. | `temporalidade-integracao/scientific_support_matrix.json` |

## 2. Afirmacoes vedadas

| # | Nao dizer | Por que |
|---|---|---|
| 1 | "O sistema preve enchentes em qualquer cidade." | O projeto mede predisposicao fisica, nao preve ocorrencia; e ha tres regioes servidas, com maturidades diferentes. |
| 2 | "Curitiba tem modelo local validado operacionalmente." | O modelo servido em Curitiba e o `fluvial_planicie`, ajustado com Sen1Floods11, CEMS e UK. |
| 3 | "Petropolis foi validada localmente." | Petropolis nao tem uma unica linha na tabela unica. |
| 4 | "Ausencia de chamado significa ausencia de inundacao." | Ausencia de registro e o nivel mais fraco da hierarquia de negativos. |
| 5 | "A camada social aumenta ou diminui o escore fisico." | Invariante declarada no proprio manifesto social: nao altera escore, nao treina, nao rotula. |
| 6 | "Observacao remota ou representacao neural detecta enchente e alimenta o escore." | Frentes auxiliares nao promovem celula, nao mudam peso e nao alteram portao. |
| 7 | "A temporalidade foi comprovada." | Nenhum escopo avaliado chegou a `supported`. |
| 8 | "AUC e a probabilidade de inundacao daquela celula." | AUC e discriminacao entre pares no desenho de validacao. |
| 9 | "O escore e risco social total." | Risco exigiria ameaca, exposicao, vulnerabilidade e capacidade de resposta. |
| 10 | "O projeto resolve previsao meteorologica." | A chuva entra como variavel antecedente ou como cenario, nunca como previsao. |
| 11 | "O escore de Curitiba e maior ou menor que o de Recife." | Sao modelos diferentes, sem calibracao comum. A comparacao mede a escolha de modelo. |
| 12 | "O TWI e o mecanismo do alagamento urbano." | Ele foi mecanismo em Nova York e **inverteu** em Chicago. |
| 13 | "A elevacao relativa permite transferir entre regioes." | Ela resolve dominio; o efeito dela nao e estavel entre cidades. |
| 14 | "Todas as celulas da grade tem resposta." | 1.703 em Recife, 965 em Curitiba e 15.010 em Petropolis sao recusadas, e a recusa e parte do produto. |

## 3. Como formular os casos dificeis

### Temporalidade

**Correto:** a frente temporal foi incorporada como auditoria explicita; no estado
atual, parte das perguntas e inconclusiva ou nao testavel com os dados disponiveis, o
que impede afirmar validacao prospectiva ampla.

**Incorreto:** o modelo foi validado temporalmente nas regioes servidas.

### Observacao remota e representacao

**Correto:** o projeto incorpora frentes de representacao e sensoriamento remoto como
apoio de observacao e pesquisa, mantendo separada a inferencia fisica servida.

**Incorreto:** o sistema usa imagem para detectar enchentes e alimentar o escore final.

### Transferencia

**Correto:** Curitiba e Petropolis sao servidas por transferencia, com o nivel de
caracterizacao declarado na maturidade; o unico teste de transferencia de efeito entre
cidades que o projeto executou reprovou parte das variaveis.

**Incorreto:** o modelo generaliza para outras cidades.

### Sao Paulo

**Correto:** o candidato fisico local ficou abaixo do acaso e por isso nao e servido; o
mesmo recorte sustenta associacao entre chuva antecedente e chamado.

**Incorreto:** Sao Paulo nao funcionou. (Funcionou o que se pediu do pipeline: reprovar.)

## 4. A regra por tras de tudo

Um resultado entra na comunicacao publica quando:

1. existe artefato versionado que o sustenta, com caminho citado;
2. o estado dele esta declarado — sustentado, inconclusivo, nao testavel, reprovado ou
   fora de escopo;
3. a limitacao aparece **junto** do resultado, e nao numa nota adiante.

Resultado que reprova uma hipotese do projeto entra pelas mesmas tres condicoes, com o
mesmo peso de um que a sustenta. O checklist completo esta em
[`docs/VALIDACAO.md`](VALIDACAO.md), secao 7.
