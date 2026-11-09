# Recife e Petrópolis: contribuições efetivas à primeira entrega

25/10/2026. Atualiza a leitura anterior, que não havia incorporado as campanhas `recife-aprendizado/v1` e `petropolis-aprendizado/v1`, executadas em 24/10. Estas linhagens acrescentam tarefas e implementações reais; os resultados de inundação orbital v6–v10 permanecem com seus próprios alvos. Nenhum modelo ativo ou artefato congelado foi modificado nesta conferência.

## Parecer científico

Ambas as regiões são contribuições efetivas. Recife demonstra transferência do núcleo físico para priorização de ruas com relato administrativo de alagamento em bairros fora do ajuste. Petrópolis demonstra transferência de uma relação de encosta para ordenação de cicatrizes inventariadas e uma integração territorial que distingue água, encosta e conexão. Nenhuma das duas demonstrou benefício adicional conclusivo da atualização local sobre seu modelo externo congelado nas formulações testadas.

Esses resultados sustentam a ferramenta como meio de transferir conhecimento e integrar evidências de naturezas distintas. O êxito e o insucesso dos ajustes locais são resultados científicos próprios; não se exige uma melhoria em todas as tarefas para reconhecer a contribuição da ferramenta. Tampouco se transforma implementação em confirmação de todas as hipóteses.

## Recife

- **Experimento real:** 1.648 relatos SEDEC de 2014–2024; 420 geocodificados até a rua; 96 ruas com relato em 38 bairros. Teste em 19 bairros: 2.225 ruas, 48 com relato. Treinamento de 350 ajustes, previsões individuais, pesos, controles e emendas salvos.
- **Transferência sustentada:** Q externo AUC dentro do bairro 0,66869; HAND 0,59514; contraste +0,07355, IC95% [0,01153; 0,15908]. É presença administrativa contra fundo sem relato; não especificidade, área inundada ou probabilidade física.
- **Aprendizado local:** incremento aproximadamente +0,010 com 48 ruas, inconclusivo. Retenção nas seis unidades externas preservada, cerca de 0,733 antes e 0,735–0,737 após atualização. Ajustar parâmetros é real; benefício adicional não foi demonstrado.
- **Contribuição territorial:** candidata para 295 FCU; 228 com cobertura, 74 associadas a rua relatada e 33 com resultados fora do ajuste. Os 50.198 moradores dessas 33 FCU são universo censitário, não pessoas atingidas ou beneficiadas. A unidade de rua não descreve automaticamente becos e interiores sem logradouro oficial.
- **Contribuição técnica:** extração real de PE3D, cadastro de drenagem, grafo candidato, condições de saída, fontes e lacunas. Ganhos aparentes de drenagem/PE3D foram revistos após controle de comprimento da rua. Drenagem fica inconclusiva e PE3D não sustenta ganho nesse controle; proximidade geométrica não demonstra capacidade hidráulica.

## Petrópolis

- **Experimento real:** modelo de encosta aprendido no inventário Idai/Zimbábue; aplicação a 81 cicatrizes de fevereiro de 2022; 350 ajustes locais com relatos e emenda de desenvolvimento com cicatrizes de outros bairros. Pesos, previsões e retenção salvos.
- **Transferência delimitada:** AUC média por cicatriz 0,66048, IC95% [0,587; 0,726], contra fundo não rotulado. Declive: 0,65390. Contraste +0,00658, IC95% [−0,01861; 0,03168], inconclusivo. Há ordenação acima do acaso no desenho presença–fundo; não superioridade, equivalência ou probabilidade física demonstrada.
- **Natureza da origem:** a auditoria constatou que o fundo do Idai foi construído pela envoltória do inventário; a cadeia não comprova inspeção uniforme de todas as células sem feição. Tratar também a origem como inventário contra fundo sem detecção registrada enquanto não houver documentação suficiente de ausência.
- **Aprendizado local:** relatos históricos pioraram a ordenação de cicatrizes; com B=100, delta aproximado −0,158. A estrutura externa limitou parte dessa deterioração, com controles favoráveis contra prior permutado. Atualizar com cicatrizes de outros bairros também não superou o congelado. Há dependência espacial residual nessa emenda, que permanece desenvolvimento do mesmo episódio.
- **Contribuição territorial:** resposta com água, encosta, conexão, contexto social, fontes, limites e observação necessária. Pela auditoria: cinco FCU têm interseção cartográfica; 20 têm interseção de inventário ou geocódigo interno; 31 ao admitir relatos a até 100 m. Os critérios não são intercambiáveis e relatos não provam ruptura no local do endereço.
- **Separação empírica dos processos:** o escore de água teve AUC de aproximadamente 0,37 nas cicatrizes. Isso demonstra que baixo escore de água não serve como evidência de segurança contra movimentos de massa nesse caso. A validação de inundação nas FCU de Petrópolis continua em aberto.

## Verificação e limites do parecer

Em 25/10, os testes específicos das duas campanhas foram reexecutados: **16 aprovados em 10,64 s**, sem dispensas. Intérprete: ambiente `revp-susc`; comando: `python -B -m pytest tests/test_recife_aprendizado_v1.py tests/test_petropolis_aprendizado_v1.py -q -p no:cacheprovider --basetemp local_runs/validacao-contribuicoes-regioes-20261025/pytest_tmp`.

A conferência atual de hashes e o recálculo das métricas primárias estão em `local_runs/validacao-contribuicoes-regioes-20261025/verificacao.json`, produzidos por `scripts/planejamento/verificar_contribuicoes_regionais_20261025.py`, sem gravar nas duas entregas. Não houve reexecução dos treinos nesta revisão.

Conferidos **130 arquivos de Recife e 453 de Petrópolis**, sem ausentes ou hashes divergentes. AUC primária recalculada de Recife: Q 0,6686913897290876 e HAND 0,5951376125907373; Petrópolis: externo 0,6604808657624983 e declive 0,6539014178587812. Reproduzem os números publicados, com diferença de aproximadamente 1,19e-9 no declive de Petrópolis.

Foram lidos os verificadores anteriores: Recife registra 242 linhas de métricas recalculadas, reinferência de 40 ajustes, zero bairros compartilhados entre ajuste/teste e zero consultas em teste; a auditoria distinta de Petrópolis registra reinferência de 352 modelos e conferência de 453 arquivos. Esses são resultados registrados anteriormente, não novas reinferências nesta revisão.

## Formulação para a entrega

Recife e Petrópolis demonstram como a mesma ferramenta organiza e testa conhecimento em tarefas territoriais distintas. Em Recife, o núcleo físico externo priorizou ruas com relato de alagamento em bairros retidos. Em Petrópolis, uma relação de terreno aprendida no exterior produziu ordenação das cicatrizes inventariadas, enquanto os testes locais identificaram deterioração por atualização e limites dos registros de dano. As duas campanhas produziram respostas territoriais rastreáveis, com evidências e lacunas explícitas. A contribuição é transferência e integração experimentalmente avaliadas, sem afirmar benefício líquido da atualização local em ambas ou fundir inundação e movimentos de massa em uma tarefa única.

## Fontes principais

- `docs/RECIFE_APRENDIZADO_ALAGAMENTO_RESULTADOS_2026_10_24.md`.
- `docs/PETROPOLIS_APRENDIZADO_ENCOSTAS_RESULTADOS_2026_10_24.md`.
- `docs/AUDITORIA_ENTREGA_PETROPOLIS_2026_10_24.md`.
- `local_runs/recife-aprendizado/v1/resultados/contrastes.csv`.
- `local_runs/petropolis-aprendizado/v1/resultados/contrastes_T1.csv`.
