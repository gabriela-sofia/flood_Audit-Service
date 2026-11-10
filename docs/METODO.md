# Método do REV-P

O percurso conecta o problema territorial à observação, ao aprendizado e à resposta. Sua formulação completa está no [artigo atual](../artigo/REV-P_artigo_atual.pdf).

## 1. Qualificar a observação

O inventário reúne 1.641.356 observações com local, data, fonte, escala, classe e uso autorizado. A curadoria admite 763.155 registros para treino e 513.530 para avaliação externa. Cada ensaio define sua separação entre ajuste e teste. As classes são presença, comparador, água permanente e evidência insuficiente.

Na referência óptica, água nova exige MNDWI positivo durante a cheia e uma leitura anterior válida sem água. Em Muçum, as cenas Sentinel-2 são de 21 de abril e 6 de maio de 2024. A classificação de cena exclui nuvem, sombra e falta de leitura. A camada JRC ajuda a separar água permanente. A comparação com a mancha Planet/SkySat do Guaíba confere a referência em outro sensor.

Código: [curadoria e admissão](../codigo/scripts/metodo/README.md), [dados de aprendizado](../codigo/scripts/suscetibilidade/README.md), [Recife](../codigo/scripts/recife_aprendizado/README.md) e [Petrópolis](../codigo/scripts/petropolis_aprendizado/README.md).

Registro executado: [admissão das observações e usos permitidos](../evidencias/01_curadoria/README.md).

## 2. Descrever o território

O núcleo usa células de aproximadamente 30 × 30 m. MDT, declive e direção de escoamento dão origem à drenagem, HAND, área de contribuição e distâncias. Fração construída e areia completam os atributos. As unidades de terreno mantêm a família do MDT; famílias distintas não são costuradas no mesmo ponto.

Nas ruas, a agregação conserva a unidade rua-bairro. A chuva atua como gatilho temporal na análise dos relatos e permanece fora do escore físico descrito no artigo.

Código: [terreno](../codigo/scripts/terreno/README.md), [geoprocessamento](../codigo/scripts/geo/README.md) e [atributos dos casos territoriais](CODIGO.md).

## 3. Aprender e transferir a relação física

O núcleo ajusta regressão logística regularizada, com pesos por evento e classe e interceptos específicos de cena. O escore transferido reúne contribuições dos atributos e interações; pesos e padronização são preservados na transferência, e interceptos de cena não seguem para o destino.

Para transferir aos destinos brasileiros, o ajuste usa 36.331 pontos de 59 cheias no exterior, agrupadas em 43 unidades de evento e sub-bacia. São 16.633 pontos CEMS, 11.698 UFO e 8.000 Sen1Floods11. A versão externa completa atende à transferência para as FCU. A versão com reserva ajusta em 29 unidades, usa quatro para estimar o erro da resposta, quatro para calibrar a recusa e seis para medir retenção.

O teste externo da Tabela I usa outro núcleo, ajustado em 91.446 observações que incluem eventos brasileiros. O evento CEMS EMSR720 entra nesse núcleo congelado. O contraste com HAND é avaliado fora do país em 30 componentes, com IC98,33% no protocolo de três hipóteses. Os pesos aplicados às comunidades brasileiras pertencem ao ajuste sem pontos do Brasil.

Código: [aprendizado físico](../codigo/scripts/metodo/README.md) e [transferência](../codigo/scripts/vulnerabilizados/README.md). Parâmetros: [modelos físicos](../modelos/fisico/README.md).

## 4. Avaliar por tarefa e por território

AUC verifica se positivos recebem escores maiores que seus comparadores. Os intervalos reamostram grupos territoriais inteiros. Populações com pouco suporte são descritivas; repetições de sementes não são novas replicações territoriais.

Água nova é comparada a pontos válidos sem água na cheia. Em Recife, o modelo com reserva ordena ruas com relato em relação às demais ruas do mesmo bairro. O pareamento por comprimento examina o viés de registro. Em Petrópolis, o modelo treinado nas rupturas do Idai ordena as cicatrizes de 2022 diante de células de fundo. O modelo usa sete atributos FABDEM a 30 m e tem uma tarefa própria de encosta.

Código: [fundamentação das tabelas e intervalos](../codigo/scripts/fundamentacao_artigo/README.md). Evidências: [matriz do artigo](MATRIZ_ARTIGO.md).

## 5. Aprender com referências locais

A atualização soma uma correção regularizada ao escore externo. Consultas por margem ou sorteio obedecem ao orçamento. O filtro de suporte exige cinco positivos, cinco negativos e três grupos com positivo. Ganho no destino e retenção na origem são medidos separadamente.

A memória conserva exemplos para o ensaio em três destinos. Na RMPA, o ganho nas FCU e a perda nas seis unidades de retenção são medidas separadas. Em Manaus e na RMC, o filtro conservou a versão externa. A perda de 0,078 citada no artigo refere-se à retenção após uma atualização de Manaus sem filtro.

Código: `v10_ii_ciclo.py`, `v10_iii_memoria.py` e `v10_iv_competencia.py`, em [atualização e transferência](../codigo/scripts/vulnerabilizados/README.md).

## 6. Perceber água e compor uma resposta

O SAR VV/VH alimenta a U-Net que delimita água pixel a pixel. A avaliação usa 90 recortes de teste entre 446 anotados e apresenta IoU sem água permanente. No pré-treino mascarado, a rede reconstrói blocos ocultos de leitura válida. O encoder inicia o ajuste supervisionado, comparado com início aleatório em oito eventos e com 1, 3 ou 12 recortes rotulados por contexto. A precisão média mede esse ensaio. O intervalo do ganho com um recorte cruza a margem de 0,025. A continuidade com LwF mede outra questão, a retenção do aprendizado em sequência.

Por coordenada, a resposta reúne a posição relativa, o escore e a contribuição de cada variável. O encaminhamento distingue maior suscetibilidade, menor suscetibilidade com suporte e evidência insuficiente. O Censo, o mapa de água quando disponível e as referências científicas acompanham a conta. A busca é medida em consultas sintéticas, a verificação de frases tem um ensaio de 18 casos e a auditoria recompõe os escores registrados.

Código: [SAR](../codigo/scripts/sar/README.md), [recuperação documental](../codigo/scripts/recuperacao/README.md), [resposta](../codigo/scripts/servico/README.md) e [medidas sociais](../codigo/scripts/social/README.md).
