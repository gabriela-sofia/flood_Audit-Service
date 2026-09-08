# Estado atual do pacote publico

Data de organizacao: 2026-09-07.

Este pacote representa o estado publico do projeto apos curadoria dos scripts de treino,
servico, chuva, transferencia, auditoria e artefatos consolidados. O objetivo e permitir
leitura, revisao e reproducao parcial sem incluir tabelas completas grandes ou material
experimental fora do contrato atual.

## Incluido

| Grupo | Incluido |
|---|---|
| Treino curado | Scripts em `scripts/treino/` para consolidacao, desenho experimental, chuva e selecao de modelos. |
| Servico | `svc01`, `svc02`, `svc03`, resolucao de camadas, suporte cientifico, proveniencia, verificador de afirmacoes e cadeia fora do contrato. |
| Chuva | Contrato de chuva, provedores, serie por celula, casos das centrais, procedencia das datas e autopareamento temporal. |
| Transferencia entre cidades | Terreno sob gatilho em Nova York, aquisicao de Chicago e o teste da segunda cidade com o estimador importado. |
| Artefatos servidos | Indice de modelos, demonstracoes do contrato e grade publica para Recife, Curitiba e Petropolis. |
| Auditorias | Chuva, transferencia, temporalidade, recuperacao de fonte, cadeia seletiva e camada social territorial. |
| Documentacao | Conceitos, metodo, arquitetura, evidencia, validacao, limitacoes, regioes, afirmacoes, fontes, roteiro de estudo, cartao do modelo, ficha dos dados e inventario de artefatos. |

## Nao incluido

| Grupo | Motivo |
|---|---|
| Tabelas completas da tabela unica | Arquivos grandes; representados por manifesto, tamanhos e checksums. |
| Tabelas brutas das centrais e de Chicago | Dezenas de MB por arquivo; representadas por manifesto com contagem, janela e checksum. |
| Pesos binarios pesados | Nao necessarios para explicar o contrato publico atual. |
| Camada de apresentacao, tela e reajuste periodico | Ficam fora do escopo deste pacote; os testes que as guardam sao pulados com o motivo declarado. |
| Rascunhos internos | Nao fazem parte da evidencia cientifica publica. |
| Experimentos sem ponte com o contrato | Permanecem fora do pacote curado. |

## Estado por frente

| Frente | Estado publico |
|---|---|
| Recife | Modelo pluvial local servido com maturidade `mvp_local`; desempenho fora dos criterios fortes e negativos por ausencia de registro. |
| Curitiba | Servida por transferencia caracterizada do modelo fluvial de planicie; nao e declarada como modelo local validado. |
| Petropolis | Servida por modelo fluvial de serra com transferencia sem referencia local; nenhuma linha local na tabela unica. |
| Sao Paulo | Candidato pluvial urbano nao servido; auditoria de chuva antecedente tem suporte no recorte testado. |
| NYC | Modelo pluvial urbano servido como referencia externa; auditoria de chuva suportada; efeito do terreno sob gatilho medido a 1 km. |
| Chicago | Teste de transferencia executado. Chuva e declividade replicam; `twi_dinf`, `twi_dinf` x chuva e `elevation_rel_m` invertem. Transferencia reprovada neste teste. |
| Social Curitiba | Camada v0 de revisao, paralela ao escore fisico. |
| Temporalidade | Estados inconclusivos e nao testaveis preservados; nenhum escopo em `supported`. |

## Cobertura de teste

`python -m pytest tests -q` cobre, alem das regressoes historicas por estagio:

| Frente | Testes |
|---|---|
| Contrato, grade e servico | `test_svc_contrato_inferencia.py`, `test_svc03_grade.py` |
| Chuva | `test_subsistema_chuva.py`, `test_chuva_autopareamento.py`, `test_chuva_serie_e_casos.py`, `test_chuva_procedencia_datas.py` |
| Transferencia entre cidades | `test_terreno_sob_gatilho.py`, `test_transferencia_entre_cidades.py` |
| Temporalidade | `test_suporte_cientifico_temporal.py` |
| Social | `test_camada_social.py`, `test_ibge_curitiba.py` |
| Neural | `test_cadeia_seletiva.py` |
| Integridade do proprio pacote | `test_documentacao_bate_com_artefato.py`, `test_integridade_do_pacote.py` |

Os dois ultimos cobrem, hoje: toda AUC, `n` e numero de grupos dos cinco candidatos;
todo coeficiente e intervalo dos quatro modelos servidos; toda faixa de dominio; a grade
das tres regioes; a razao de risco dos dois recortes de chuva; os sete termos da
comparacao entre Nova York e Chicago e o corte por tipo de chamado; as cinco linhas da
predicao seletiva; a cadeia seletiva; a aderencia temporal dos cinco modelos; a
composicao por fonte da tabela unica; e a contagem social.

Os dois ultimos existem para que um erro de curadoria falhe a suite em vez de sair
publicado: eles conferem numero de documento contra artefato, comando declarado contra
arquivo existente e campo de proveniencia contra script presente.

## Regra de manutencao

Antes de afirmar novo resultado publico:

1. O script precisa estar em `scripts/` com caminho atual.
2. O artefato precisa estar em `modelo/execucoes/` ou documentado por manifesto.
3. A documentacao precisa apontar para o artefato correto, e o numero precisa vir dele.
4. O contrato precisa continuar dono dos portoes, maturidade e extrapolacao.
5. A linguagem precisa separar resultado suportado, inconclusivo, nao testavel, reprovado e fora de escopo.
6. Resultado que reprova uma hipotese do projeto entra no pacote com o mesmo peso de um que a sustenta.
