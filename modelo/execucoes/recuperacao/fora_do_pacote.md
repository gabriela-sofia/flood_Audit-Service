# Por que a cadeia desta execucao fica fora do pacote publico

Os artefatos deste diretorio comparam tres recuperadores de trecho documental sobre o
mesmo conjunto de 984 consultas: busca lexical, densa e hibrida.

| Etapa | Script | Esta no pacote |
|---|---|---|
| Corpus documental e busca sobre ele | camada de corpus do repositorio de trabalho | **nao** |
| Avaliacao das consultas | `scripts/recuperacao/avaliacao_de_recuperacao.py` | **nao** |
| Indice denso | `scripts/recuperacao/indice_denso.py` | **nao** |
| Comparacao dos recuperadores | `scripts/recuperacao/comparar_recuperadores.py` | **nao** |

O motivo e de escopo. Essa cadeia depende do corpus documental do projeto — o texto
interno indexado, com a esteira de ingestao e o filtro de injecao que a acompanham.
Esse corpus nao e evidencia cientifica sobre inundacao: ele e memoria de trabalho do
projeto, com conteudo que nao cabe num pacote publico de metodo e resultado.

O que fica registrado aqui e o resultado da comparacao e o seu limite: o recuperador
serve para RASTREAR de onde veio uma afirmacao. Ele nao altera portao, peso, escore ou
maturidade, e nenhuma conclusao deste repositorio sobre suscetibilidade depende dele.

Os numeros publicados em `docs/EVIDENCIA.md` sao conferidos contra o JSON deste
diretorio pelo teste `tests/test_documentacao_bate_com_artefato.py`. O que nao da para
fazer deste checkout e regerar o JSON.
