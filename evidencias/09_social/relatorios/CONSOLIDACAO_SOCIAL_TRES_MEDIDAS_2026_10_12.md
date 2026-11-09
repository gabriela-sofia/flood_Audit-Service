# Três medidas sociais integradas ao REV-P

Uma IA integrada passa a consultar três objetos territoriais com contratos separados: exposição espacial, componentes de vulnerabilidade e cobertura científica condicionada à oportunidade de observação. A execução acrescentou dados oficiais, cruzamentos por polígonos reais, avaliações próprias e transporte HTTP. **A consolidação operacional não equivale à confirmação científica das três hipóteses.**

## Entrega executada

| Medida | Implementado e executado | Resultado e limite |
|---|---|---|
| Exposição | Interseção da delineação candidata CEMS EMSR720 com setores e grade populacional oficial; limites de localização e sensibilidade de borda | No universo geocodificado da grade: população entre **1.131 e 16.068**; domicílios ocupados particulares e coletivos entre **445 e 6.418**. Não são atingidos confirmados, perdas ou pessoas presentes em maio/2024. |
| Vulnerabilidade | Sete componentes com numerador, denominador e interpretação próprios, renda monetária original e cobertura por variável | **9.186 setores**: Curitiba 3.190; suporte regional do estudo RS 5.924; recorte amazônico 72. Sem índice sintético, média de medianas ou atribuição de valor municipal a pessoas. |
| Cobertura científica | Ligação dos 400 pontos do piloto aberto à malha, conferência do sorteio, pesos originais, estratos instrumentais, situação urbana/rural e contexto de renda; cinco contrastes e sensibilidade a três cortes de renda | Todos os pontos associados a setor. **191 anotáveis e 209 “não sei”**; 160 nuvem, 28 sombra, 21 mistura. Nenhuma referência admitida. Um evento e uma cena, sem confirmação de desigualdade. |

O nome histórico “Manaus” não é o município dos pontos: a ligação espacial encontrou setores de **Iranduba (1301852) e Manacapuru (1302504)**. A região servida é `am_piloto`, e cada feição conserva o código e nome municipal oficial. Não atribuir esses resultados à população de Manaus.

### Dados incorporados

Foram baixados e registrados **249.546.214 bytes**, incluindo três arquivos de agregados nacionais (demografia e características domiciliares partes 1 e 2), malhas AM/RS, o bloco `grade_id14` e seu mapa de articulação. Reutilizam-se os agregados básico/renda, malha PR e dicionários já existentes. Cada fonte tem endereço, tamanho, instante de aquisição e SHA-256. O Censo é **2022**; a data de publicação/aquisição não atualiza a população.

Fontes primárias: [agregados setoriais IBGE](https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios/Agregados_por_Setor_csv/), [grade oficial e articulação](https://geoftp.ibge.gov.br/recortes_para_fins_estatisticos/grade_estatistica/censo_2022/grade_estatistica/), [nota metodológica da grade](https://geoftp.ibge.gov.br/recortes_para_fins_estatisticos/grade_estatistica/censo_2022/Notas_metodologicas_grade_estatistica_2022.pdf). A nota define `TOTAL_DOM` como domicílios ocupados particulares **e coletivos** e exclui da grade geocodificação de qualidade 5/6. O tamanho da população excluída neste recorte é desconhecido: o máximo da grade só delimita o universo nela representado.

### Componentes e denominadores

| Componente | Numerador | Denominador | Pergunta sustentada |
|---|---|---|---|
| Crianças 0–9 | V01031 + V01032 | V01006 | Composição etária potencialmente relevante ao planejamento de apoio |
| Pessoas 60+ | V01040 + V01041 | V01006 | Composição etária; não mede dependência funcional |
| Sem água encanada | V00201 | V00001 | Condição domiciliar de abastecimento |
| Uso principal de água superficial | V00117 | V00001 | Forma de abastecimento; não certifica potabilidade |
| Sem ligação à rede de água | V00464 | V00001 | Infraestrutura domiciliar; alternativa não é inadequação automática |
| Sem banheiro nem sanitário | V00238 | V00001 | Condição de saneamento domiciliar |
| Esgoto em vala | V00313 | V00001 | Destinação declarada de esgoto; não comprova contaminação |

V00001 é DPPO. V01006 tem o próprio universo demográfico publicado. Os dicionários oficiais acompanham o manifesto. A agregação é **razão de somas apenas em pares completos**, com cobertura explícita; ausência/supressão ou denominador vazio produz `null`. A reconciliação encontrou 22/95/4 setores em Curitiba/RS/AM com demografia diferente ou ausente em relação ao básico: permanecem identificados na avaliação, sem substituir o denominador. Não houve numerador superior ao denominador nos sete componentes disponíveis.

Exemplo de cobertura em Curitiba: crianças em 3.112 setores; idosos em 3.147; sem água encanada em 3.044; água superficial em 3.155; sem ligação à rede em 2.869; sem banheiro em 3.125; esgoto em vala em 3.048. Uma proporção calculada no subconjunto com dado não descreve automaticamente o município inteiro.

## Avaliação de cada medida

**Exposição.** Confere unicidade/validade geométrica, sobreposição, cobertura da água pelo bloco de grade e contagens inteiras. Usa unidades totalmente contidas para o mínimo e todas as unidades intersectadas para o máximo. Não multiplica população por fração de área. A grade tem 4.658 células no envelope (3.758 de 200 m e 900 de 1 km), cobrindo 99,9999775% da geometria de água, dentro da tolerância numérica declarada. Com erosão de 30 m, os limites populacionais ficam [880, 11.949]; com dilatação de 30 m, [2.161, 17.671]. São perturbações exploratórias do suporte, não intervalos de confiança ou validação da delineação. Setores dão [0, 32.848] no seu universo; grade e setores não têm exatamente o mesmo universo e não podem ser apresentados como estimativas intercambiáveis. Água permanente não foi removida com validação neste pacote; a união contém delineações de diferentes datas.

**Vulnerabilidade.** A avaliação atual é de qualidade do dado, coerência dos denominadores, cobertura e agregação espacial. Mede componentes relevantes ao desenho, não prova o construto completo “vulnerabilidade” nem prediz dano. Renda é mediana do responsável **com rendimento**, não renda per capita, pobreza individual ou capacidade de resposta. Os componentes sanitários não substituem localização e funcionamento de hospitais, escolas, transporte e abrigos quando a pergunta exigir acesso a serviços. Essa hipótese exige outra camada e método de acessibilidade.

**Cobertura científica.** A análise preserva evento/cena/classe instrumental/situação. O limiar R$1.212 é descritivo, correspondente ao salário mínimo nominal de 2022; análises alternativas usam R$606 e R$2.424, sem classificar pessoas. Há três contrastes rurais com os dois grupos, dois deles com apenas dois setores acima do limiar. Os dois contrastes urbanos não têm o segundo grupo: a diferença fica `null`. Nos estratos rurais água/ambíguo/seco, diferenças descritivas de anotabilidade entre grupos são +5,35/−27,72/+5,12 pontos percentuais. Os sinais diferentes, a concentração espacial, a seleção por SAR e a ausência de replicação impedem uma conclusão geral. Ausência global de admissão também não prova igualdade de cobertura. “Anotável por IA” não é sensibilidade do sensor nem área/população cientificamente coberta.

## Uso no projeto

O SVC-05 registra `GET /social/medidas/saude` e `POST /social/medidas/consulta`. A segunda recebe `regiao` (`curitiba`, `rs_emsr720`, `am_piloto`) e opcionalmente `aoi` com `lon_min`, `lat_min`, `lon_max`, `lat_max`. A consulta social anterior de Curitiba passa a anexar `medidas_territoriais`, mantendo a trilha demonstrativa `social_v0_review_only` identificável.

Na AOI, componentes descrevem **os setores intersectados**; os limites de população referem-se ao recorte de estudo, não a inundação. Estudos regionais de água/grade/contrastes vêm explicitamente com seu suporte regional e não são atribuídos à AOI. A consulta da cobertura seleciona pontos da amostra dentro da AOI. Não há campo de entrada para promover SAR ou declarar uma máscara admitida. O serviço recusa arquivos alterados e caminhos fora do pacote; o cache não ignora a conferência de integridade. Nenhuma medida altera o escore físico, consome reservas independentes ou decide prioridade pública.

Artefatos principais:

- [Pacote setorial e dicionários](../local_runs/social-consolidacao/2026-10-12-v3/manifesto.json).
- [Exposição na grade](../local_runs/social-exposicao-grade/2026-10-12-v1/exposicao_grade.json).
- [Oportunidades e cobertura](../local_runs/social-consolidacao/2026-10-12-v3/am_piloto/cobertura_cientifica.json).
- [Contrastes de cobertura](../local_runs/social-avaliacao-cobertura/2026-10-12-v1/avaliacao.json).
- [Protocolo das próximas avaliações](PROTOCOLO_AVALIACAO_SOCIAL_2026_10_12.md).

As execuções preliminares v2 não são a entrega: a inspeção detectou uma variante oficial de coluna `setor`, corrigida com teste de regressão antes da consolidação v3. Os arquivos preliminares permanecem preservados e identificados para auditoria.

## Feito e falta fazer

Verificação final: **114 testes aprovados, 2 dispensados**, sem falhas, em 36,94 s. Inclui testes novos de denominadores/ausência, variantes das colunas oficiais, pesos, duplicação, reservas, integridade, recorte da AOI, grade, suporte comum e integração do SVC-05; também inclui regressões das rotas sociais/físicas existentes. Os dois dispensados pertencem ao transporte físico: uma região sem modelo e uma requisição Recife recusada por suporte temporal inconclusivo. Essa recusa foi preservada. Há aviso de depreciação do adaptador HTTP da versão instalada; ele não impediu os testes. O isolamento bloqueava o loopback interno usado pelo executor HTTP; a verificação concluída usou autorização de comunicação local.

Reprodução offline com as fontes adquiridas: instalar `requirements-social.txt`, executar `social06_consolidacao.py`, `social08_exposicao_grade.py` e `social09_avaliacao_cobertura.py`. Artefatos são imutáveis: repetir a mesma versão exige fontes/código idênticos; para mudanças, usar nova pasta/versionar os estudos complementares. `social04_aquisicao_complementar.py` faz aquisição incremental limitada de URLs oficiais. O pacote precisa ser transportado junto ao serviço; ausência retorna indisponibilidade ou `not_testable`, sem substituir por dados demonstrativos.

O relatório de testes e o manifesto de entrega estão em `local_runs/social-consolidacao/qa_20261012_verificado.xml` e `local_runs/social-consolidacao/entrega_20261012.json`.

**Feito:** aquisição; dados em escala adequada; interseções reais; sete componentes com denominadores; análise de exposição em duas escalas; oportunidade em pontos com pesos; comparações condicionadas e sensibilidade; integridade; API; avaliação de software e documentação.

**Ainda necessário à confirmação:** referência brasileira admitida para a finalidade específica; revisão humana cega; replicação de cobertura por eventos e cenas com suporte comum; avaliação externa do construto/impactos para alegações de vulnerabilidade; localização e acessibilidade de serviços quando requerida; avaliação humana própria para prioridade pública e compreensão da comunicação. Os testes de software não substituem essas evidências. As análises executadas aqui são exploratórias: nenhum protocolo posterior é apresentado como pré-registro anterior aos resultados.
