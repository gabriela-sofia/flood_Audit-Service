# Método do REV-P

O percurso conecta o problema territorial à observação, ao aprendizado e à resposta. Sua formulação completa está no [artigo atual](../artigo/REV-P_artigo_atual.pdf).

## 1. Qualificar a observação

Cada registro recebe um papel e uma finalidade admitida: presença do fenômeno, ausência observada, desconhecido ou água permanente. A qualificação conserva instrumento, lugar, data e condições de observação. Nuvem, sombra e falta de leitura permanecem desconhecidas.

As referências externas vêm de CEMS, Sen1Floods11, UFO e GFD, com resoluções e limites próprios. No Brasil, a referência óptica identifica água no episódio e compara com a imagem anterior. Água permanente é excluída. Relatos de Recife e cicatrizes de Petrópolis formam outras tarefas, com desenhos próprios.

Código: [curadoria e admissão](../codigo/scripts/metodo/README.md), [dados de aprendizado](../codigo/scripts/suscetibilidade/README.md), [Recife](../codigo/scripts/recife_aprendizado/README.md) e [Petrópolis](../codigo/scripts/petropolis_aprendizado/README.md).

Registro executado: [admissão das observações e usos permitidos](../evidencias/01_curadoria/README.md).

## 2. Descrever o território

O núcleo usa células de aproximadamente 30 × 30 m. MDT, declive e direção de escoamento dão origem à drenagem, HAND, área de contribuição e distâncias. Fração construída e areia completam os atributos. As unidades de terreno mantêm a família do MDT; famílias distintas não são costuradas no mesmo ponto.

Nas ruas, a agregação conserva a unidade rua-bairro. A chuva atua como gatilho temporal na análise dos relatos e permanece fora do escore físico descrito no artigo.

Código: [terreno](../codigo/scripts/terreno/README.md), [geoprocessamento](../codigo/scripts/geo/README.md) e [atributos dos casos territoriais](CODIGO.md).

## 3. Aprender e transferir a relação física

O núcleo ajusta regressão logística regularizada, com pesos por evento e classe e interceptos específicos de cena. O escore transferido reúne contribuições dos atributos e interações; pesos e padronização são preservados na transferência, e interceptos de cena não seguem para o destino.

O núcleo completo usa 43 unidades externas. O núcleo com reserva ajusta em 29, separa quatro para estimar respostas problemáticas, quatro para calibrar a recusa e seis para retenção. Os números de versões anteriores que aparecem em um ensaio específico conservam esse contexto; não são fundidos ao ajuste externo.

Código: [aprendizado físico](../codigo/scripts/metodo/README.md) e [transferência](../codigo/scripts/vulnerabilizados/README.md). Parâmetros: [modelos físicos](../modelos/fisico/README.md).

## 4. Avaliar por tarefa e por território

AUC verifica se positivos recebem escores maiores que seus comparadores. Os intervalos reamostram grupos territoriais inteiros. Populações com pouco suporte são descritivas; repetições de sementes não são novas replicações territoriais.

Água observada é comparada com ausência observada na mesma cena. Ruas com relato são comparadas com o fundo do mesmo bairro. Cicatrizes são comparadas com células de fundo, sem declarar esse fundo como ausência física comprovada.

Código: [fundamentação das tabelas e intervalos](../codigo/scripts/fundamentacao_artigo/README.md). Evidências: [matriz do artigo](MATRIZ_ARTIGO.md).

## 5. Aprender com referências locais

A atualização soma uma correção regularizada ao escore externo. Consultas por margem ou sorteio obedecem ao orçamento. O filtro de suporte exige cinco positivos, cinco negativos e três grupos com positivo. Ganho no destino e retenção na origem são medidos separadamente.

A memória conserva exemplos para os ensaios de continuidade. O artigo registra tanto ganhos quanto perdas de retenção; não afirma preservação integral do conhecimento.

Código: `v10_ii_ciclo.py`, `v10_iii_memoria.py` e `v10_iv_competencia.py`, em [atualização e transferência](../codigo/scripts/vulnerabilizados/README.md).

## 6. Perceber água e compor uma resposta

As redes SAR aprendem percepção de água; pré-treino, segmentação e continuidade têm medidas próprias. A memória documental recupera fontes, um verificador examina afirmações e o encaminhamento distingue respostas que precisam de mais evidência.

O contexto social acompanha o território sem modificar o escore físico. A resposta conserva a conta, as fontes, a versão e as condições de uso.

Código: [SAR](../codigo/scripts/sar/README.md), [recuperação documental](../codigo/scripts/recuperacao/README.md), [resposta](../codigo/scripts/servico/README.md) e [medidas sociais](../codigo/scripts/social/README.md).
