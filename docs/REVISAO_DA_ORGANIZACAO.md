# Organização do repositório

O artigo atual define o recorte. A pasta reúne o PDF v16, sua fonte, os produtores e dependências dos resultados, os parâmetros identificados e as evidências selecionadas. O repositório reúne os arquivos que sustentam esse recorte.

## Como ficou

| Parte | Conteúdo preparado |
|---|---|
| Entrada | README com percurso do projeto e acesso às nove tabelas |
| Artigo | Um único PDF atual, fonte editável, três figuras e referências |
| Documentação | Método, contribuições, fontes, reprodução e mapa do código |
| Código | 304 arquivos Python em 24 famílias, com responsabilidade e entradas identificadas |
| Testes originais | 22 módulos selecionados, além dos testes do pacote |
| Evidências | Curadoria e resultados separados na ordem do artigo |
| Modelos | Núcleo completo, núcleo com reserva, encosta e percepção, com sua identificação |
| Rastreabilidade | 98 linhas de tabelas e 32 parágrafos ligados a códigos e fontes |

## O que conferir

1. Comece pelo [README](../README.md).
2. Abra o [método](METODO.md) para acompanhar o percurso da observação à resposta.
3. Siga a [matriz](MATRIZ_ARTIGO.md), na ordem das tabelas, e abra as evidências de cada bloco.
4. Confira a divisão das pastas em [organização do código](CODIGO.md).
5. Veja o que foi executado no [relatório de verificação](../manifestos/verificacao_preparacao.json).

## Critério de seleção

O pacote conserva o estado que sustenta o artigo. Nomes de rotinas que contêm versões continuam quando identificam uma implementação usada pelos resultados ou uma dependência necessária. Relatórios datados acompanham resultados ainda citados; a documentação de entrada apresenta a síntese atual. Versões antigas de artigo, arquivos de trabalho e a sequência de commits não compõem a publicação.

Os arquivos do trabalho local foram selecionados por conteúdo e dependência. O [manifesto de origem](../manifestos/origem.json) identifica o commit de referência e registra que há trabalho local ainda não commitado. Os hashes identificam o recorte preparado com mais precisão do que o commit isolado.

## Ajustes de organização

Os resultados e o PDF foram preservados. Na fonte LaTeX, mudaram apenas os caminhos das figuras. Os ajustes registrados incluem a pasta configurável de dados de Recife, a consulta à documentação atual, a padronização de descrições e identificadores, a retirada de referências a documentos ausentes e a configuração explícita do adaptador HTTP opcional. Os hashes anterior e preparado identificam as alterações. Os algoritmos de aprendizado, os pesos e os valores dos ensaios foram preservados.

Os resultados de busca da Tabela VIII conservam o corpus e as datas das respectivas avaliações. A indexação dos documentos atuais prepara a consulta desta publicação e não é apresentada como repetição daqueles ensaios.

## Reprodução

As conferências de integridade e transcrição usam somente os arquivos incluídos. A repetição integral de experimentos exige os dados externos e o ambiente descritos nos protocolos. A preparação foi conferida no ambiente local existente; não foi declarada uma instalação limpa em outra máquina. Os totais de testes desta preparação são separados dos 2.747 testes da entrega de 20/10 citada no artigo.

O repositório reúne o artigo, o código e as evidências selecionadas. Ambientes virtuais, caches e dados de trabalho ficam nas pastas locais de execução.
