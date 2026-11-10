# Reprodução e verificação

## O que conferir imediatamente

Na raiz do pacote:

```bash
python reproducao/verificar_pacote.py
python reproducao/conferir_resultados.py
```

A primeira rotina verifica hashes, arquivos, referências internas e cobertura da matriz. A segunda compara resultados estruturados com a apresentação no artigo e confere relações numéricas selecionadas. Nenhuma delas treina ou transforma os dados científicos.

Para os testes portáveis:

```bash
python -m pip install -r requirements-auditoria.txt
python -m pytest testes -q
```

O [relatório de conferência](../manifestos/verificacao_preparacao.json) registra o que foi efetivamente executado. QA de software e integridade editorial têm papéis diferentes da validação científica descrita no artigo.

## Código de pesquisa e dados externos

As rotinas originais ficam em `codigo/scripts/`. Seus caminhos relativos são calculados a partir de `codigo/`, preservando os importadores e a organização funcional do código de origem. Dependências locais selecionadas são identificadas no [grafo de código](../manifestos/dependencias_codigo.json).

Os artefatos agregados incluídos podem ser colocados nos caminhos esperados pelas rotinas:

```bash
python reproducao/materializar_evidencias.py
```

Essa operação copia somente os arquivos registrados para `codigo/local_runs/`. Não cria dados brutos, não completa entradas ausentes e não executa os produtores. A pasta materializada é ignorada pelo Git.

Para repetir experimentos, é necessário prover as entradas declaradas em cada protocolo e nas rotinas de aquisição, usar um ambiente compatível e escolher uma nova saída de execução. O pacote não apresenta um clone sem rasters como repetição integral dos experimentos.

Os testes originais selecionados ficam em `codigo/tests/` e são executados a partir de `codigo/`, com as dependências de pesquisa instaladas. Testes que exigem acervo externo conservam a regra de disponibilidade do teste original.

Os dados externos de Recife usam `codigo/data/` por padrão. A variável `REV_P_DADOS_PROJETO` permite indicar outra pasta de dados. Esse ajuste substitui o caminho pessoal fixo da cópia de origem; não altera atributos, pesos, protocolos ou métricas.

## Fontes atuais da resposta

O produtor `codigo/scripts/servico/rag_corpus.py` usa o artigo e os documentos atuais deste pacote como fontes sobre o projeto. A lista substitui referências a documentos de fases anteriores que não integram esta publicação. As classes científica, operacional e do projeto continuam separadas.

Para criar o índice de consulta atual, a partir de `codigo/`:

```bash
python scripts/servico/rag_corpus.py
```

O índice é gerado em `codigo/local_runs/rag-corpus/` para consultar a documentação atual. A avaliação da busca da Tabela VII usou um corpus de 3.813 trechos e 11.238 consultas sintéticas, registrado em seu [artefato de execução](../evidencias/08_resposta/evolucao-escopo/v1/cognicao/resultado.json).

## Artigo editável

Em `artigo/fonte/`, compile duas vezes:

```bash
pdflatex artigo.tex
pdflatex artigo.tex
```

A fonte v22 usa a pasta local `figuras/`. A bibliografia faz parte do arquivo `artigo.tex`. O PDF anexado e o pôster estão identificados por hash no [manifesto de origem](../manifestos/origem.json).

## Resposta por coordenada

A [consulta territorial](../codigo/scripts/ia/consulta_territorial_v2.py) recebe uma unidade admitida e coordenadas, lê os atributos dos rasters associados e registra a procedência. Ela requer as camadas de terreno e o pacote de modelo correspondente. Onde falta cobertura construída ou areia, o registro declara a imputação aplicada. O [serviço integrado](../codigo/scripts/servico/resposta_integrada.py) monta a resposta, o cartão e a conferência dos campos e do hash.

O [registro de paridade](../evidencias/08_resposta/ia-integrada-v2/fechamento_v1/fisica.json) confere 65.549 pontos do MT11 e 82.249 da rodada 3. Os 2.747 testes e as 922 fontes conferidas pertencem à entrega local de 20 de outubro. A suíte portátil deste repositório verifica o recorte publicado.
