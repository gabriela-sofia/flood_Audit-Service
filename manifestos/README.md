# Rastreabilidade da preparação

| Arquivo | O que identifica |
|---|---|
| [Origem](origem.json) | Recorte do artigo e identidade da fonte preparada |
| [Arquivos de origem](arquivos_origem.json) | Cópias, caminhos, tamanho, papel e SHA-256 |
| [Dependências do código](dependencias_codigo.json) | Entradas selecionadas e relações locais entre módulos |
| [Afirmações do artigo](afirmacoes_artigo.json) | Linhas das tabelas e parágrafos com suas fontes |
| [Planilha textual da matriz](afirmacoes_artigo.csv) | A mesma rastreabilidade em formato tabular |
| [Metadados bibliográficos](metadados_bibliograficos.json) | 35 entradas bibliográficas da fonte v22 |
| [Índice dos documentos atuais](indice_documental_atual.json) | Fontes e classes conferidas para a consulta preparada |
| [Verificação desta preparação](verificacao_preparacao.json) | Comandos e verificações executados no pacote organizado |

Hashes verificam identidade de arquivos. A ligação a uma fonte registra rastreabilidade; o regime e o alcance científico de cada resultado continuam sendo os descritos no artigo e nos relatórios de execução.

Quando há ajuste de portabilidade, o registro conserva `sha256_origem`, o hash da cópia preparada e a descrição da alteração. Não se substituem resultados por valores inventados.

Os relatórios de testes podem conter casos repetidos em suítes sobrepostas. O resumo final deduplica cada par módulo–nome do teste; o total desta preparação não se soma aos testes de entregas anteriores.
