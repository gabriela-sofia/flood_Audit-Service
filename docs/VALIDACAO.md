# Validacao

Este arquivo explica como o projeto mede o que mede, e como separar o que e produto do
que e pesquisa. Ele nao repete numeros: os numeros estao em `docs/EVIDENCIA.md`.

## 1. O que a AUC mede aqui

AUC mede capacidade discriminativa entre positivos e negativos dentro do desenho de
validacao usado. Nada alem disso.

Ela **nao** mede:

- probabilidade real de inundacao;
- risco absoluto;
- dano;
- validade em janela futura;
- transferencia garantida para outra cidade.

Uma AUC de 0,7334 no `fluvial_planicie` diz que, tomando um par formado por um positivo
e um negativo do proprio conjunto de ajuste, o modelo ordena o par corretamente em 73%
das vezes. Ela nao diz que 73% das celulas de Curitiba estao certas.

## 2. Por que a validacao e agrupada

Pontos proximos no espaco ou no tempo vazam informacao entre treino e teste. Sem
agrupamento, a AUC mede memorizacao de vizinhanca e sai otimista.

A regra do projeto: **o grupo nunca e mais fino que a celula**. A tabela unica registra
pontos empilhados na mesma celula dentro da mesma fonte — 402.714 em NYC, 1.124 em Sao
Paulo, 812 em Curitiba, 430 em Sen1Floods11, 92 em CEMS, 86 em UFO, 71 em Recife. Duas
linhas na mesma coordenada, com as mesmas variaveis e a mesma classe, nao sao duas
observacoes independentes.

Cada modelo declara a unidade que usou:

| Modelo | Unidade de reamostragem | Por que |
|---|---|---|
| `recife_pluvial` | componente conexa de grupo e celula | 71 pontos empilhados em 26 celulas; reamostrar o grupo original era reamostrar linha |
| `fluvial_planicie` | componente conexa de grupo e celula | idem |
| `fluvial_serra` | componente conexa de grupo e celula | idem |
| `pluvial_urbano_nyc` | bloco espacial de 1 km | a fusao por celula colapsou a unidade: o mesmo local reaparece em muitos dias |

## 3. Por que Firth ou regularizacao

Em base com separacao, poucos eventos ou desequilibrio, a logistica comum produz
coeficiente instavel — no limite, infinito. A correcao de Firth reduz esse vies.

No projeto isso pesa especialmente em:

- Recife, por `n` menor (269 pontos, 227 grupos);
- modelos com positivos raros;
- variaveis fisicas correlacionadas entre si;
- transferencia entre dominios diferentes.

**Cuidado que o pacote declara explicitamente:** ha duas camadas de estimacao no
projeto, e elas nao produzem o mesmo numero. Uma produz os coeficientes; a outra produz
a AUC de validacao cruzada. Sempre que um numero for citado, o texto precisa dizer de
qual camada ele saiu. Os coeficientes em `modelo/MODELO.md` e a AUC em
`svc-01-modelos/indice.json` vem de camadas diferentes do mesmo ajuste.

## 4. Bootstrap e intervalos

O intervalo nao sai da matriz de informacao. Ele sai de bootstrap **reamostrando
grupos** — a unidade que compartilha a mesma informacao — com 1.000 replicas e semente
fixa `20260820`.

Quando o IC cruza zero, o ajuste nao estabeleceu o efeito. A consequencia e concreta e
esta no codigo, nao so no texto: a variavel entra no escore e **fica fora da frase de
explicacao**. O mesmo vale para variavel com sinal contrario ao fisicamente esperado.

| Modelo | Variaveis fisicamente verificadas | Variaveis que ficam fora da explicacao |
|---|---|---|
| `recife_pluvial` | `twi_dinf`, `rain_decay_index` | `elevation_rel_m`, `slope_deg`, `hand_m` |
| `pluvial_urbano_nyc` | `slope_deg`, `twi_dinf` | `elevation_rel_m`, `hand_m` |
| `fluvial_planicie` | `hand_m`, `twi_dinf` | `elevation_rel_m`, `slope_deg` |
| `fluvial_serra` | `hand_m` | — |

## 5. Aderencia temporal nao e suporte temporal

Sao duas perguntas diferentes, e confundi-las produz conclusao errada nas duas direcoes.

| Pergunta | O que mede | Onde esta |
|---|---|---|
| **Aderencia temporal** | As duas classes foram observadas nos mesmos dias dentro do conjunto de ajuste? | campo `aderencia_temporal` de cada modelo |
| **Suporte temporal** | O desempenho se sustenta numa janela futura? | `temporalidade-integracao/scientific_support_matrix.json` |

A aderencia de Recife e o exemplo de por que isso importa: 205 datas distintas, apenas 5
com as duas classes, e 3,72% dos pontos em data compartilhada. Como o modelo usa
`rain_decay_index`, que depende do tempo, saber a data e quase saber a classe naquela
base. Isso e um problema do **desenho da amostra**, e nao aparece na AUC.

| Modelo | Datas distintas | Com as duas classes | Pontos em data compartilhada |
|---|---:|---:|---:|
| `recife_pluvial` | 205 | 5 (2,44%) | 3,72% |
| `pluvial_urbano_nyc` | 2.402 | 378 (15,74%) | 81,42% |
| `pluvial_urbano_sao_paulo` | 1.976 | 336 (17,00%) | 35,11% |
| `fluvial_planicie` | 132 | 21 (15,91%) | 92,71% |
| `fluvial_serra` | 9 | 9 (100,0%) | 100,0% |

## 6. Curadoria versus experimento

Quatro categorias, e so as duas primeiras entram no resultado servido.

| Categoria | Entra no servido? | Exemplo neste pacote |
|---|---|---|
| Contrato operacional | Sim | `svc02_contrato_inferencia.py`, grade SVC-03 |
| Modelo curado | Sim, se marcado servivel | `recife_pluvial`, `fluvial_planicie`, `fluvial_serra` |
| Diagnostico cientifico | Nao diretamente | Sao Paulo abaixo do acaso, temporalidade de Curitiba, teste de Chicago |
| Experimento exploratorio | Nao | cadeia seletiva, pote crescente, frentes neurais |

Um resultado de diagnostico pode ser **decisivo para o que se afirma** sem alterar nada
no que se serve. O teste de Chicago e o caso claro: ele reprova a transferencia do
efeito do TWI, muda o que a documentacao pode dizer, e nao mexe em um unico peso.

## 7. Checklist antes de publicar qualquer resultado

Antes de colocar uma frase no README, no poster, no planejamento ou numa apresentacao:

| Pergunta | Resposta exigida |
|---|---|
| Existe artefato versionado sustentando o numero? | Sim, com caminho. |
| O resultado e operacional, diagnostico ou demonstrativo? | Declarado explicitamente. |
| A regiao tem a maturidade correta na frase? | Sim. |
| A fonte do dado esta registrada? | Sim. |
| A validacao e agrupada, temporal ou apenas descritiva? | Declarada. |
| De qual camada de estimacao veio o numero? | Declarada. |
| Ausencia de dado foi tratada como ausencia de evento? | Nao. |
| A camada social alterou o escore fisico? | Nao. |
| Recuperacao, neural ou observacao remota alteraram o escore? | Nao. |
| As recusas foram preservadas na comunicacao? | Sim. |
| A limitacao aparece junto do resultado, e nao numa nota de rodape? | Sim. |

As tres primeiras e a ultima sao verificadas automaticamente pela suite:
`tests/test_documentacao_bate_com_artefato.py` confere numero contra artefato e
`tests/test_integridade_do_pacote.py` confere que todo artefato declara como foi
produzido.

## 8. Matriz de frentes

| Frente | Pergunta | Saida | Estado |
|---|---|---|---|
| Tabela unica | Os dados estao harmonizados? | Manifesto, contagens, rejeicoes | Executado |
| Modelos fisicos | O terreno discrimina casos? | Coeficientes, AUC, limitacoes | Executado |
| Contrato | A celula pode ser servida? | Escore ou recusa | Implementado e conferido |
| Grade | Como o territorio fica em celulas? | CSV, GeoTIFF, PNG, resumo | Executado |
| Temporalidade | O suporte persiste no tempo? | Matriz de suporte | Inconclusivo ou nao testavel |
| Chuva | Chuva aumenta notificacao no mesmo lugar? | Razao de risco por 10 mm | Suporte em NYC e Sao Paulo |
| Transferencia | O efeito se repete em outra cidade? | Comparacao de coeficientes | **Reprovada para TWI e elevacao relativa** |
| Social | Onde ha maior exposicao e vulnerabilidade? | Indicadores por setor | Implementado para Curitiba |
| Recuperacao | Como achar evidencia textual? | Comparacao em k=5 | Executado |
| Neural | A representacao aprende padrao util? | Experimento pre-registrado | Demonstrativo |
| Predicao seletiva | A confianca ordena melhor que o acaso? | AURC contra o acaso da propria frente | Tres seletores sim, dois nao |
