# Estado atual do pacote publico

Data de organizacao: 2026-09-06.

Este pacote representa o estado publico do projeto apos curadoria dos scripts de treino,
servico, auditoria e artefatos consolidados. O objetivo e permitir leitura, revisao e
reproducao parcial sem incluir tabelas completas grandes ou material experimental fora
do contrato atual.

## Incluido

| Grupo | Incluido |
|---|---|
| Treino curado | Scripts em `scripts/treino/` para consolidacao, desenho experimental, chuva e selecao de modelos. |
| Servico | `svc01`, `svc02`, `svc03`, suporte cientifico, proveniencia e cadeia fora do contrato. |
| Artefatos servidos | Indice de modelos, demonstracoes do contrato e grade publica para Recife, Curitiba e Petropolis. |
| Auditorias | Chuva, temporalidade, recuperacao de fonte, cadeia seletiva e camada social territorial. |
| Documentacao | Metodo, evidencia, fontes, modelo, dados e lista de artefatos curados. |

## Nao incluido

| Grupo | Motivo |
|---|---|
| Tabelas completas da tabela unica | Arquivos grandes; representados por manifesto, tamanhos e checksums. |
| Pesos binarios pesados | Nao necessarios para explicar o contrato publico atual. |
| Rascunhos internos | Nao fazem parte da evidencia cientifica publica. |
| Experimentos sem ponte com o contrato | Permanecem fora do pacote curado. |

## Estado por frente

| Frente | Estado publico |
|---|---|
| Recife | Modelo pluvial local servido com maturidade `mvp_local`; desempenho fora dos criterios fortes. |
| Curitiba | Servida por transferencia caracterizada do modelo fluvial de planicie; nao e declarada como modelo local validado. |
| Petropolis | Servida por modelo fluvial de serra com transferencia sem referencia local. |
| Sao Paulo | Candidato pluvial urbano nao servido; auditoria de chuva antecedente tem suporte no recorte testado. |
| NYC | Modelo pluvial urbano servido como referencia externa e auditoria de chuva suportada. |
| Social Curitiba | Camada v0 de revisao, paralela ao escore fisico. |
| Temporalidade | Estados inconclusivos e nao testaveis preservados. |

## Regra de manutencao

Antes de afirmar novo resultado publico:

1. O script precisa estar em `scripts/` com caminho atual.
2. O artefato precisa estar em `modelo/execucoes/` ou documentado por manifesto.
3. A documentacao precisa apontar para o artefato correto.
4. O contrato precisa continuar dono dos portoes, maturidade e extrapolacao.
5. A linguagem precisa separar resultado suportado, inconclusivo, nao testavel e fora de escopo.
