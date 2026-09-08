# Artefatos curados

Este arquivo lista o que entra no pacote publico como resultado curado. A lista nao
substitui os manifestos; ela orienta revisao e manutencao.

A regra de completude: **todo diretorio versionado aparece nesta pagina**. Diretorio que
existe no repositorio e nao esta aqui e uma lacuna de curadoria, nao um detalhe.

## Treino e consolidacao

| Caminho | Papel |
|---|---|
| `modelo/execucoes/ds-03-esquema/` | Esquema alvo da tabela unica, em CSV e JSON. |
| `modelo/execucoes/ds-05-tabela-unica/manifesto_v2.json` | Manifesto da tabela unica v2, com nomes, tamanhos e checksums. |
| `modelo/execucoes/ds-05-tabela-unica/relatorio_contagem_por_fonte.csv` | Contagem resumida por fonte. |
| `modelo/execucoes/aud-chuva-02/` | Escala do contraste de chuva por fonte. |
| `modelo/execucoes/mod-prosp-02/` | Holdout temporal e viabilidade por fonte. |
| `modelo/execucoes/mod-serra-03/` | Selecao do modelo de relevo ingreme e coeficientes por relevo. |
| `scripts/treino/ds01_*` a `scripts/treino/ds05_*` | Consolidacao e verificacao da tabela unica. |
| `scripts/treino/chuva02_*` a `scripts/treino/chuva04_*` | Preparacao de chuva antecedente. |
| `scripts/treino/aud_*` | Auditorias de desenho, chuva e integridade. |
| `scripts/treino/mod_prosp02_*`, `mod_serra03_*` | Holdout temporal e modelo de relevo. |
| `scripts/treino/susc_firth_*`, `susc_ambiente_compat_comum.py` | Compatibilidade do estimador e do ambiente. |

## Servico

| Caminho | Papel |
|---|---|
| `modelo/execucoes/svc-01-modelos/` | Indice dos modelos candidatos e servidos, com um JSON por modelo. |
| `modelo/execucoes/svc-02-contrato/` | Demonstracoes e cobertura do contrato. |
| `modelo/execucoes/svc-03-grade/` | Grades publicas em CSV, GeoTIFF e PNG, mais o resumo por regiao. |
| `scripts/servico/svc01_construir_modelos_servidos.py` | Construcao dos modelos servidos. |
| `scripts/servico/svc02_contrato_inferencia.py` | Contrato de inferencia. |
| `scripts/servico/svc03_grade_suscetibilidade.py` | Geracao da grade publica. |
| `scripts/servico/resolver_camadas.py` | Resolucao das camadas exigidas por um caso. |
| `scripts/servico/suporte_cientifico.py` | Estados de suporte cientifico e matriz. |
| `scripts/servico/proveniencia.py` | Hash, tempo e bloco de proveniencia dos artefatos. |
| `scripts/servico/verificador_de_afirmacoes.py` | Barreira que so publica frase cujas afirmacoes o proprio verificador resolveu contra uma origem autorizada. |
| `scripts/servico/cadeia_no_contrato.py` | Encaixe aditivo da cadeia auxiliar no contrato. |

## Chuva e transferencia entre cidades

| Caminho | Papel |
|---|---|
| `modelo/execucoes/chuva-autopareamento/` | Autopareamento temporal em NYC e Sao Paulo, matriz de forcamento e manifestos dos insumos. |
| `modelo/execucoes/nyc-terreno-sob-gatilho/` | Efeito do terreno sob o gatilho de chuva em Nova York, a 0,1 grau e a 1 km. |
| `modelo/execucoes/chicago-teste-da-lei/` | O mesmo estimador aplicado a segunda cidade, com o resultado defeituoso anterior mantido para comparacao. |
| `scripts/chuva/contrato_chuva.py` | Contrato proprio da chuva: cenarios, papel fisico e versao do metodo. |
| `scripts/chuva/provedores.py` | Provedores de serie de chuva. |
| `scripts/chuva/serie_por_celula.py` | Serie diaria por celula. |
| `scripts/chuva/casos_das_centrais.py` | Casos por celula x dia a partir das centrais de atendimento. |
| `scripts/chuva/procedencia_das_datas.py` | Granularidade e procedencia da data de cada ocorrencia. |
| `scripts/chuva/autopareamento_temporal.py` | Desenho autopareado no tempo e matriz de forcamento. |
| `scripts/chuva/adquirir_mrms_nyc.py`, `adquirir_mrms_chicago.py` | Chuva de radar a 1 km. |
| `scripts/chuva/mrms_vs_era5_nyc.py` | Comparacao entre os dois produtos de chuva. |
| `scripts/chuva/gatilho_terreno_nyc.py` | Terreno sob gatilho, estrato celula x dia. |
| `scripts/chuva/terreno_sob_gatilho_1km.py` | Terreno sob gatilho, estrato dia, chuva a 1 km. |
| `scripts/chuva/terreno_sob_gatilho_chicago.py` | O teste da segunda cidade, com o estimador importado. |
| `scripts/externo/aq_chamados01_*`, `aq_chamados02_*` | Aquisicao e leitura unica das centrais de atendimento. |
| `scripts/externo/aq_chicago01_*`, `aq_chicago02_*`, `aq_chicago03_*` | Chamados, MDT e terreno nos pontos de Chicago. |

## Terreno e fontes externas

| Caminho | Papel |
|---|---|
| `scripts/terreno/ter01_*` a `ter06_*` | Cadeia fisico-hidrologica harmonizada e registro de auditoria regional. |
| `scripts/externo/aquisicao/`, `cems/`, `pontos/`, `uk/` | Aquisicao e preparacao das fontes externas que alimentam a tabela unica. |

## Evidencias auxiliares

| Caminho | Papel |
|---|---|
| `modelo/execucoes/temporalidade/` | Auditorias temporais por protocolo. |
| `modelo/execucoes/temporalidade-integracao/` | Matriz de suporte, reconciliacao e estado temporal. |
| `modelo/execucoes/recuperacao/` | Comparacao de recuperadores de fonte. |
| `modelo/execucoes/neural-catraca/` | Experimento confirmatorio de reconstrucao mascarada. |
| `modelo/execucoes/sintese-neural/` | Sintese seletiva da frente neural. |
| `modelo/execucoes/camada-social-territorial/` | Camada social territorial v0 em revisao. |
| `scripts/temporalidade/` | Ontologia, auditoria e dimensao temporal. |
| `scripts/social/` | Aquisicao de fontes sociais, camada social e recorte do IBGE em Curitiba. |
| `scripts/neural/cadeia_seletiva.py` | Encadeamento seletivo das frentes auxiliares. |

## Material historico mantido

| Caminho | Papel |
|---|---|
| `outputs_public/data/susc_*` | Pacote por estagio, anterior a esta organizacao. Fica versionado porque os testes `tests/test_susc_*` o usam como base de regressao. Nao e o estado atual e nao deve ser citado como resultado corrente. |
| `scripts/dino/` | Rotinas de revisao visual da fase anterior. Nao participam do escore servido, nao produzem artefato nesta versao e ficam como registro do caminho percorrido. |

## Documentacao

| Caminho | Papel |
|---|---|
| `README.md` | Porta de entrada: estado atual, resultados versionados, limitacoes principais e estrutura. |
| `docs/ROTEIRO_DE_ESTUDO.md` | Ordem para ler codigo e documentacao, quatro aulas e exercicios de leitura critica. |
| `docs/CONCEITOS.md` | Suscetibilidade, contrato, maturidade, recusa, hierarquia dos negativos, variaveis fisicas e glossario. |
| `docs/METODO.md` | Cadeia publica, portoes, treino curado, grade, frente de chuva e teste de transferencia. |
| `docs/ARQUITETURA.md` | Mapa em arvore, fluxo operacional, organizacao do codigo e o caminho de uma celula. |
| `docs/EVIDENCIA.md` | O que cada frente encontrou, com o artefato de origem em cada secao. |
| `docs/VALIDACAO.md` | O que a AUC mede, validacao agrupada, Firth, bootstrap, aderencia temporal e checklist. |
| `docs/LIMITACOES.md` | Todas as limitacoes reunidas, por desenho, regiao e frente. |
| `docs/REGIOES.md` | Retrato de cada regiao, incluindo Chicago como cidade de teste. |
| `docs/AFIRMACOES.md` | Afirmacoes sustentadas com artefato, afirmacoes vedadas com motivo. |
| `docs/FONTES.md` | Fontes da cadeia, produtos de chuva e limites de interpretacao. |
| `docs/ESTADO_ATUAL.md` | O que entra e o que fica fora deste pacote, e a regra de manutencao. |
| `docs/ARTEFATOS_CURADOS.md` | Este inventario. |
| `modelo/MODELO.md` | Cartao do modelo: pesos, intervalos, dominio, maturidade e limitacoes. |
| `modelo/DADOS.md` | Ficha dos dados: esquema, regra de ouro, criterios de admissao, contagens e pool fluvial. |

Toda tabela numerica destas paginas e conferida contra o artefato de origem por
`tests/test_documentacao_bate_com_artefato.py`.

## Criterio de inclusao

Um resultado entra neste pacote quando atende a pelo menos uma das condicoes:

1. E consumido diretamente pelo contrato de inferencia.
2. Explica uma grade publica versionada.
3. Documenta uma evidencia auxiliar citada no planejamento ou no poster.
4. Registra uma limitacao que impede afirmacao mais forte.
5. Resume arquivo grande que nao deve ser versionado integralmente.
6. E base de regressao de um teste versionado.

Resultado exploratorio sem ligacao com essas condicoes permanece fora do pacote curado.
Resultado que REPROVA uma hipotese do projeto entra pela condicao 4 e nunca e omitido.
