# REV-P

**Reconhecimento de Evidências Verificáveis para Predição**

Uma comunidade com poucos registros de cheia precisa de uma base para priorizar onde observar. O REV-P aprende como terreno e drenagem se relacionam com a água em cheias documentadas e aplica esse conhecimento a territórios brasileiros. Por coordenada, a ferramenta apresenta a posição relativa, a contribuição das variáveis, a versão do modelo, as fontes e o suporte da resposta. Quando há radar adequado, a delimitação de água acompanha a resposta. O Censo contextualiza quem vive no local.

O **artigo v22, de 10 de novembro de 2026**, define o recorte científico do repositório. O pôster apresenta a pergunta, o percurso do método e os resultados selecionados para a banca.

## Comece por aqui

| Para entender | Abra |
|---|---|
| A organização do repositório | [Guia de navegação](docs/REVISAO_DA_ORGANIZACAO.md) |
| O artigo atual | [PDF](artigo/REV-P_artigo_atual.pdf) e [fonte editável](artigo/fonte/artigo.tex) |
| O percurso em uma página | [Pôster](artigo/REV-P_poster.pdf) |
| O percurso do método | [Método](docs/METODO.md) e [admissão dos dados](evidencias/01_curadoria/README.md) |
| Cada resultado, seu código e sua evidência | [Matriz do artigo](docs/MATRIZ_ARTIGO.md) |
| O papel de cada pasta e rotina | [Organização do código](docs/CODIGO.md) |
| O que pode ser conferido com este pacote | [Reprodução e verificação](docs/REPRODUCAO.md) |
| Dados, instrumentos e referências | [Dados e fontes](docs/DADOS_E_FONTES.md) |
| A contribuição científica e territorial | [Contribuições](docs/CONTRIBUICOES.md) |

## Resultados organizados na ordem do artigo

| Tabela | Resultado ou desenho | Código principal | Evidência |
|---|---|---|---|
| I | Relações físicas, ganho externo e pesos com e sem reserva | [Método físico](codigo/scripts/metodo/README.md) | [Aprendizado](evidencias/02_aprendizado/README.md) |
| II | Transferência para comunidades e recortes urbanos | [Transferência](codigo/scripts/vulnerabilizados/README.md) | [Coortes brasileiras](evidencias/03_transferencia/README.md) |
| III | Ordenação e conta do escore nas ruas de Recife | [Recife](codigo/scripts/recife_aprendizado/README.md) | [Ruas e relatos](evidencias/04_recife/README.md) |
| IV | Modelo do Idai avaliado nas encostas de Petrópolis | [Petrópolis](codigo/scripts/petropolis_aprendizado/README.md) | [Cicatrizes e fundo](evidencias/05_petropolis/README.md) |
| V | Ganho com exemplos locais, retenção e memória | [Atualização](codigo/scripts/vulnerabilizados/README.md) | [Atualização na RMPA](evidencias/06_atualizacao/README.md) |
| VI | Cobertura observacional e suporte nas comunidades | [Contagens por coorte](codigo/scripts/fundamentacao_artigo/f08_ruas_e_comunidades.py) | [Cobertura](evidencias/09_social/README.md) |
| VII | Percepção, pré-treino, fontes, suporte e auditoria | [Radar](codigo/scripts/sar/README.md) e [resposta](codigo/scripts/servico/README.md) | [Percepção](evidencias/07_percepcao/README.md) e [resposta](evidencias/08_resposta/README.md) |

## Qual núcleo responde a cada pergunta

| Versão | Ajuste | Avaliação |
|---|---|---|
| Núcleo congelado | 91.446 observações, incluindo eventos brasileiros | Teste fora do país em 30 componentes. Ganho de +0,055 AUC sobre HAND no contraste registrado |
| Externa completa | 36.331 pontos de 59 cheias no exterior, agrupados em 43 unidades | Transferência sem ajuste local. Nas FCU de Manaus, Recife e RMR, os intervalos de ganho sobre HAND ficaram acima de zero |
| Externa com reserva | 29 unidades para ajuste e 14 reservadas | Ruas de Recife, atualização local, calibração da resposta e retenção. Seis unidades medem retenção |

Em Petrópolis, a tarefa é ordenar cicatrizes com um modelo de encosta aprendido no Idai. Na RMPA, a atualização por margem elevou a AUC nas FCU de 0,874 a 0,926, com perda média de 0,028 nas unidades de retenção. O [método](docs/METODO.md) identifica as referências e a regra de cada ensaio.

## Estrutura

```text
artigo/       artigo atual, pôster, fonte LaTeX e figuras
docs/         método, resultados, contribuições e orientações atuais
codigo/       rotinas selecionadas, dependências e testes originais
evidencias/   artefatos e relatórios ligados às tabelas do artigo
modelos/      pesos, padronização e estados dos modelos identificados
reproducao/   conferência do pacote e dos resultados registrados
testes/       testes da publicação e verificações portáveis
manifestos/   rastreabilidade, hashes e relação entre afirmações e arquivos
```

## Conferência rápida

Com Python 3.12:

```bash
python reproducao/verificar_pacote.py
python reproducao/conferir_resultados.py
```

Os dois comandos leem arquivos incluídos e não treinam modelos nem fazem downloads. Para os testes portáveis:

```bash
python -m pip install -r requirements-auditoria.txt
python -m pytest testes -q
```

As rotinas de pesquisa mantêm os caminhos de trabalho relativos à pasta `codigo/`. A [orientação de reprodução](docs/REPRODUCAO.md) distingue a conferência dos resultados publicados da repetição dos experimentos com dados externos.

## Como interpretar

AUC mede a frequência com que um positivo fica acima de seu comparador. Água nova, relato administrativo e cicatriz são referências de tarefas distintas. Em comunidades da RMPA, o ganho sobre HAND ficou inconclusivo. As FCU da RMC têm 11 positivos e entram como descrição. Os ensaios nas coortes brasileiras são de desenvolvimento.

A observação passou de uma para 123 FCU com água. Entre as 123, 119 têm as duas classes e 38 têm pelo menos cinco pontos de cada classe. O suporte para ordenar pontos e os totais censitários têm denominadores próprios. Os [manifestos](manifestos/README.md) registram a origem dos arquivos.
