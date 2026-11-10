# Organização do código

O código foi selecionado pelos produtores dos resultados do artigo e pelos testes correspondentes. A seleção inclui o fechamento de dependências locais, identificado no [grafo](../manifestos/dependencias_codigo.json). Não é uma cópia de todos os scripts da pasta de trabalho.

| Pasta | Responsabilidade | Entradas / arquivos Python |
|---|---|---|
| [challengers](../codigo/scripts/challengers/README.md) | Utilitários de avaliação importados pelos ensaios selecionados. | 0 / 4 |
| [chuva](../codigo/scripts/chuva/README.md) | Suporte à qualificação e ao tratamento temporal da chuva. | 0 / 7 |
| [evolucao_escopo](../codigo/scripts/evolucao_escopo/README.md) | Dependências da entrega de pesquisa e sua avaliação. | 1 / 4 |
| [evolucao_local](../codigo/scripts/evolucao_local/README.md) | Dependências de execução local consumidas pelo recorte. | 0 / 1 |
| [externo](../codigo/scripts/externo/README.md) | Aquisições externas consumidas pelas rotinas selecionadas. | 0 / 2 |
| [fechamento](../codigo/scripts/fechamento/README.md) | Dependências de consolidação e conferência da entrega. | 0 / 1 |
| [fundamentacao_artigo](../codigo/scripts/fundamentacao_artigo/README.md) | Cálculo das tabelas e intervalos do artigo, com população e estimando explícitos. | 6 / 6 |
| [geo](../codigo/scripts/geo/README.md) | Funções geográficas e preparação das camadas consumidas pelas entradas selecionadas. | 0 / 18 |
| [ia](../codigo/scripts/ia/README.md) | Integração de pesquisa, consulta, reprodução e avaliação dos componentes integrados. | 5 / 24 |
| [manutencao](../codigo/scripts/manutencao/README.md) | Serialização e suporte de execução importados pelo código selecionado. | 0 / 3 |
| [metodo](../codigo/scripts/metodo/README.md) | Aprendizado físico, admissão, independência territorial e protocolos de avaliação. | 13 / 63 |
| [neural](../codigo/scripts/neural/README.md) | Experimentos de representação e suporte à avaliação seletiva. | 1 / 8 |
| [petropolis_aprendizado](../codigo/scripts/petropolis_aprendizado/README.md) | Referências de encosta, aprendizado no Idai e avaliação de Petrópolis. | 14 / 14 |
| [planejamento](../codigo/scripts/planejamento/README.md) | Conferências e registros consumidos pelos produtores selecionados. | 1 / 5 |
| [recife_aprendizado](../codigo/scripts/recife_aprendizado/README.md) | Referências, atributos, teste por rua-bairro e explicação de Recife. | 12 / 12 |
| [recuperacao](../codigo/scripts/recuperacao/README.md) | Índice, recuperação de fontes e comparação entre recuperadores. | 4 / 4 |
| [sar](../codigo/scripts/sar/README.md) | Aquisição e qualificação SAR, pré-treino, segmentação e continuidade da percepção. | 8 / 31 |
| [servico](../codigo/scripts/servico/README.md) | Conta, fontes, contratos, explicação e composição da resposta territorial. | 2 / 23 |
| [social](../codigo/scripts/social/README.md) | Fontes censitárias, medidas territoriais, exposição e observabilidade. | 7 / 11 |
| [suscetibilidade](../codigo/scripts/suscetibilidade/README.md) | Esquemas, harmonização e admissão dos dados utilizados pelo núcleo. | 2 / 10 |
| [temporalidade](../codigo/scripts/temporalidade/README.md) | Estados e utilitários de qualificação temporal utilizados pelo recorte. | 0 / 4 |
| [terreno](../codigo/scripts/terreno/README.md) | Derivação e conferência do terreno, drenagem e atributos físicos. | 0 / 9 |
| [verificacao](../codigo/scripts/verificacao/README.md) | Verificações de integridade utilizadas pelo código selecionado. | 1 / 4 |
| [vulnerabilizados](../codigo/scripts/vulnerabilizados/README.md) | Transferência externa, partições, consulta, atualização, memória e competência. | 17 / 41 |

A execução das rotinas ocorre com `codigo/` como raiz de trabalho. Dados brutos e novas saídas de pesquisa ficam em `codigo/data/` e `codigo/local_runs/`, que não fazem parte da publicação. Os testes originais estão em [codigo/tests](../codigo/tests/README.md); os testes da preparação estão em [testes](../testes/README.md).
