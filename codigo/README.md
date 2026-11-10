# Código do recorte atual

As rotinas em `scripts/` implementam o método e os ensaios ligados ao artigo v22. A [matriz](../docs/MATRIZ_ARTIGO.md) aponta os produtores de cada resultado. A [organização por responsabilidade](../docs/CODIGO.md) explica as pastas, e o [grafo de seleção](../manifestos/dependencias_codigo.json) registra as dependências locais.

Os arquivos originais foram copiados com identidade conferida por SHA-256. Os produtores continuam esperando `data/` e `local_runs/` relativos a esta pasta. A [reprodução](../docs/REPRODUCAO.md) distingue o que está incluído do acervo externo necessário a novas execuções.

Uma configuração de caminho em `recife_aprendizado/ra_comum.py` foi tornada portável por `REV_P_DADOS_PROJETO`. O manifesto registra os hashes antes e depois; a lógica científica foi preservada.

O índice documental usa o artigo e os documentos atuais incluídos. Esse ajuste da lista de fontes também está registrado no manifesto, separado dos resultados datados dos ensaios de recuperação.
