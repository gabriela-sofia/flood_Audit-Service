# REV-P — entrega integrada v2 de pesquisa, 18/10/2026

Foi implementada uma candidata de pesquisa que reúne, no mesmo estado, a relação física Q_conexao calculável sem piso amostral, o histórico admitido e a percepção neural já existente. A consulta por coordenadas e a interface local funcionam. A reprodução dos resultados e a qualificação física das séries FloodNet foram executadas. A candidata continua não promovida: integração de software não demonstra aprendizado conjunto nem identificação causal.

Esta entrega é aditiva. Preserva o modelo ativo anterior, suas rotas, pesos, resultados e reservas. Atualiza o diagnóstico científico com o grafo comum reparado da rodada 3B; relatórios anteriores permanecem históricos. Não foram feitos novos treinos ou downloads nesta etapa. As campanhas neurais anteriores e a aquisição ANA–S2 estão no [relatório precedente](APRENDIZADO_CONJUNTO_E_CAUSALIDADE_2026_10_18.md).

## 1. Resultado científico que pode sustentar o artigo

| Pergunta | Evidência atual | Conclusão delimitada |
|---|---|---|
| A conexão melhora a discriminação além de HAND? | MT11, grafo 3B: 30 componentes; ΔAUC +0,055027; IC98,333% [0,023891; 0,086319] | Sustentado na população admitida, retrospectivamente reparada |
| Acrescenta valor além do modelo físico base? | Mesmos 30 componentes: +0,029618 [0,006114; 0,052541] | Ganho positivo; limite inferior abaixo da margem relevante de 0,01 |
| A frequência histórica acrescenta valor? | HF: 15 componentes; +0,029418 [−0,003220; 0,062780] | Indecidível após o grafo comum; a conclusão positiva anterior não é a atual |
| A conclusão é robusta à regra de exutório? | HQ1: 19 componentes, +0,049878 [0,005597; 0,096559]; HQ2 e HF cruzam zero | Somente HQ1 sustenta sinal nessa sensibilidade |
| Há reprodução externa adicional da conexão? | 3B reparada: 8 componentes, +0,024534; IC95% [0,004738; 0,050107] | Positivo no mínimo de unidades; retirar uma cai abaixo do mínimo, ainda há referências por centroide |
| Aprendizado conjunto neural beneficia fisicamente? | Campanha anterior, oito rótulos por classe/evento: +0,006771; IC98,75% [0,004154; 0,009118]; conteúdo real − permutado +0,002717 [−0,003091; 0,007309] | Ganho pequeno abaixo de 0,01 e especificidade não sustentada. 32 rótulos por classe/evento também cruza zero |
| Chuva antecedente explica causalmente água física? | FloodNet qualificado: 10 meses informativos; protocolo exige 12 | `not_testable`; nenhum coeficiente causal ou RR calculado |

Os intervalos MT11 usam 2.000 reamostragens e o nível para três hipóteses da análise. As campanhas MT11 e 3B compartilham conexões no grafo: seus componentes não são somados como réplicas independentes. A reprodução própria confere números; não cria confirmação inédita. O grafo exclui GFD:4503 e GFD:4355 da primária MT11 por ligação a dados conhecidos, inclusive o caso de maior contribuição ao HF. A exclusão decorre da independência, não da direção do efeito.

Uma formulação defensável para o artigo é: “Relações de conexão hidrológica, exportadas para uma cadeia de consulta sem dependência do piso amostral, apresentam ganho de discriminação em populações delimitadas após reparação da independência. A contribuição adicional do histórico de água permanece inconclusiva. A integração neural e física foi implementada, mas o benefício físico relevante e específico do aprendizado conjunto e a identificação causal dos mecanismos permanecem não demonstrados.”

Não afirmar que todas as variáveis relevantes ou causas já são conhecidas. Coeficientes e decomposições aditivas explicam a decisão do estimador sob seus dados e hipóteses; não medem efeitos de intervenção.

## 2. O que foi programado e executado

- `scripts/ia/modelo_integrado_v2.py`: um estado PyTorch com Q, histórico e percepção. Exporta padronização, coeficientes, interações e assinatura da configuração; detecta troca de pesos e reordenação de configuração.
- `ia_integrada_v2.py`: pacote imutável, consulta sem classe e verificações de integridade. Histórico somente por identidade e coordenadas exatas. Fora do catálogo, permanece indisponível.
- `consulta_territorial_v2.py`: atributos físicos por coordenadas em unidades locais cadastradas. Não depende de rótulo ou mínimo de elevação da amostra. Núcleo físico faltante recusa; cobertura/solo opcionais usam a mediana de treino com imputação declarada. Não interpola cobertura de um ponto vizinho.
- `servico_ia_integrada_v2.py`: interface local própria, estado e consulta. Teste HTTP real confirmou resposta de pesquisa e recusa de previsão, produção territorial e caminhos inválidos. O servidor criado para teste foi encerrado.
- `revisao_humana_v2.py`: formulário cego com 49 itens, duas imagens por item, opção incerto, exportação CSV e declaração com checksum. Zero respostas preenchidas pela IA. O recebimento verifica identidade declarada, integridade e IDs, sem promover rótulo automaticamente.
- `floodnet_qualidade_v2_piloto.py`, `floodnet_fisico_v2.py` e verificadores: classificação instrumental, cruzamento com chuva e respeito à parada estatística. Verificação por leitura independente de 99 sensor-dias nos arquivos originais.
- `verificar_regional_integrada_v2.py`: conferência pendente da cadeia HydroSHEDS, em pasta própria, sem sobrescrever a rodada 3B.

O pacote está em `local_runs/ia-integrada-v2/pacote_v1`: `modelo.pt`, catálogo histórico, cobertura pontual e manifesto. O estado do modelo é único; catálogos são dados auxiliares, com hash. A percepção foi copiada do pai, sem retreino; não há nova representação compartilhada aprendida. O contexto social anterior continua descritivo e preservado na linhagem do manifesto, sem conversão em causa ou atributo social no pixel.

## 3. Avaliação da cadeia completa

Q reproduziu os escores salvos em 65.549 pontos MT11 e 82.249 da rodada 3; erro máximo 3,56×10⁻¹⁵. O histórico reproduziu 28.875 pontos, erro 1,34×10⁻¹⁵. São conferências numéricas sobre linhas conhecidas, não 147.798 unidades independentes ou pontos necessariamente admitidos em cada contraste científico.

Nos 200 pontos conhecidos de EMSR498_AOI01 e EMSR411_AOI01, a consulta por coordenadas reproduziu exatamente os seis atributos físicos centrais; erro de escore ≤2,67×10⁻¹⁵. A consulta não leu rótulos. As geometrias e a cobertura pontual pertencem ao acervo selecionado: isso não valida amostragem territorial nova.

A percepção foi reexecutada em 222 chips: 84 de retenção e 138 de avaliação. A conferência com o arquivo correto do pai confirmou erro zero em AP/perda. `percepcao.json` preserva a primeira procura de cache sem sucesso; a comparação posterior correta está em `paridade_percepcao_pai.json`. Paridade mantém o aprendizado existente; não é ganho novo.

A conferência regional abrangeu 82.249 pontos: 73.616 com dados válidos da v1 tiveram igualdade exata em todos os diagnósticos. Outros 8.563 passaram à região de cobertura real, em que a acumulação anterior era ausente. Permanecem 61 pontos com nodata na região v1 e 9 sem região. As grandes diferenças de distância nesse grupo refletem a correção da região/teto de busca anterior; não são mudança dos pesos Q. Os diagnósticos HydroSHEDS não são entradas de Q. Referências de geometria e seus limites continuam separados.

## 4. Água física contínua: avanço e limite causal

O arquivo contínuo FloodNet já estava no disco, além do catálogo de eventos. A descrição antiga “somente positivos públicos” está superada para este acervo. A fonte pública disponibiliza séries contínuas e orientações sobre ruído residual e limiar instrumental: [FloodNet, registro 10211443](https://zenodo.org/records/10211443). Foi utilizado apenas o material local, com manifesto de aquisição existente; licença e condições da fonte permanecem aplicáveis.

O protocolo adotou maio–setembro de 2021–2023, para delimitar interferência de neve, sem selecionar meses por resultado. Dos 15 arquivos mensais previstos, setembro/2023 não consta do arquivo público/local; a ausência foi registrada antes dos novos efeitos, sem substituição ou imputação. Processados os 14 disponíveis.

Admissão: pelo menos 95% de leituras de profundidade processada/filtrada, bateria e contador válidos; 80% dos intervalos de cinco minutos observados; lacuna máxima de 30 minutos incluindo bordas do dia; bateria mínima 3,4 V; perda de quadros ≤20%; sensor pluvial. Duplicatas conflitantes não são usadas. Lâmina requer ≥25 mm em dois intervalos consecutivos de cinco minutos; isso não comprova dez minutos contínuos acima do limiar. “Sem lâmina” exige valores processados e filtrados iguais a zero nas leituras válidas, sujeito à qualidade; indica ausência de sinal acima do piso instrumental de cerca de 10 mm naquele sensor, não rua absolutamente seca. Estados restantes são desconhecidos ou não observados.

| Estado | Sensor-dias |
|---|---:|
| Sem lâmina instrumental | 1.848 |
| Com lâmina instrumental | 138 |
| Não observado | 3.028 |
| Desconhecido | 786 |
| Total | 5.800, em 41 sensores |

O piloto de 2021, feito antes da ampliação, não é confirmação independente. A nova regra também corrige a frequência de amostragem: não é subconjunto estrito da classificação antiga. Um dia antes negativo passou a não observado; parte de dias antes não observados passou a elegível pela avaliação correta da cadência. Nenhuma versão anterior foi alterada.

Após juntar chuva MRMS e suas defasagens, há 1.906 linhas completas em 29 sensores. A análise de D−1, ajustada por D e D−2, estratifica sensor × ano-mês × dia da semana e reamostra meses. A amostra comum também exige D+30 para o controle futuro, inclusive na análise principal, como executado. Restaram 10 meses informativos, abaixo dos 12 fixados: primária e controle são `not_testable`, sem estimativa de razão de taxas. Não houve redução do mínimo após o resultado.

A verificação de implementação distinta reproduziu 99 sensor-dias, estratificados por mês e classe, e recontou os dez meses. Isso confere integridade instrumental da classificação em uma amostra dirigida; não é adjudicação humana nem valida ausência de ruído. Chuva em dia agregado é concomitante, sensores são localizados de forma dirigida, e funcionamento, seleção e filtragem podem depender da chuva. Mesmo com mais meses, a identificação causal exigirá defender esses pressupostos.

O resultado anterior positivo sobre notificações em Nova York continua sendo associação com registro administrativo. Não substitui lâmina física. No Taquari, a ponte ANA–D8–S2 admite só quatro cenas em um episódio/componente; não identifica o mecanismo fluvial. As perguntas fluvial e pluvial exigem desenhos e conclusões separados.

## 5. Pendências reais e ordem de fechamento

1. **Revisão humana:** realizar os 49 itens cegos, devolver CSV e declaração JSON; registrar incertos, adjudicar discrepâncias e executar o protocolo de referência sem acesso às predições. Esta seleção dirigida não estima acurácia populacional; o pacote completo de 212 itens e a referência brasileira continuam próprios, com desenho e ponderação adequados para qualquer inferência populacional.
2. **Replicação da conexão:** reutilizar o rascunho e a fila da 3B, obter geometrias neutras e resolver as oito referências por centroide antes dos rótulos. As 26 ativações candidatas fora da Europa não garantem número suficiente de componentes; o dimensionamento existente deve orientar a alegação.
3. **Aprendizado conjunto:** o experimento atual já respondeu negativamente ao critério de ganho relevante/específico. Uma próxima rodada precisa nova referência física auxiliar e hipótese mecanística, com parada/permutação e margem pré-definidas. Rodar mais sementes sobre os mesmos alvos não fecha a lacuna.
4. **Causal pluvial:** protocolo de uma coorte temporal completa, antes dos efeitos, visando cobertura que permita ao menos 12 meses informativos e diagnóstico de operação/ruído/seleção. Não completar apenas os dois meses mais favoráveis, nem mudar o mínimo. A série física qualificada agora existe e pode ser reaproveitada como desenvolvimento.
5. **Causal fluvial:** água física e nível/propagação contemporâneos, referência independente, repetição de episódios e ajuste de confundidores pertinentes. Máxima de régua ou saída do estimador não são profundidade no pixel.
6. **Uso territorial:** definir população, amostra externa, calibração e avaliação de erro/cobertura antes de afirmar probabilidade ou promover a candidata. Atribuição social, profundidade e previsão datada não decorrem do fechamento de software.

A matriz `entrega_v1/matriz_lacunas.csv` e `.json` mapeia as 15 frentes para alegação, estado, dado faltante, etapa e critério. “Pendente” indica exatamente o que bloqueia cada alegação; não invalida o resultado delimitado já obtido.

## 6. Verificação, preservação e reprodução

**100 testes Python distintos passaram:** 92 de regressão/contratos e candidata, quatro de adendo/revisão e quatro da qualidade FloodNet. Houve 26 avisos de depreciação anteriores na suíte de regressão. A interface humana passou por teste JavaScript de exportação e checksum com respostas sintéticas; nenhum arquivo de resposta humana foi criado. Houve teste HTTP real. QA de software não é validação científica.

O primeiro recebimento sintético expôs uma falha na rejeição de identidade artificial com sublinhado, corrigida e retestada. A conferência FloodNet teve inicialmente uma diferença de unidades temporais no verificador, corrigida com teste para microssegundos/nanossegundos; a classificação do produtor e o protocolo congelado não foram modificados depois dos resultados. Os logs preservam falhas e retomadas.

A avaliação parou inicialmente porque um cartão contextual histórico do pai mudara. O adendo operacional conferiu especificamente pesos, inventário, admissão, dados e protocolos SAR consumidos; o cartão não era entrada da avaliação. Hashes antigos não foram atualizados à força. O adendo, as fontes consumidas e a divergência contextual permanecem registrados.

Uso local, sem promover o modelo:

```powershell
$py = 'C:/Users/gabriela/AppData/Local/Programs/Python/Python312/python.exe'
& $py -X utf8 scripts/ia/servico_ia_integrada_v2.py --porta 8132
```

O serviço oferece `GET /ia/v2/estado` e `POST /ia/v2/consulta`; o contrato está em `entrega_v1/contrato.json`. Para um exemplo conhecido: enviar o JSON de `fechamento_v1/pedido_EMSR498_AOI01.json` ao POST. A consulta recusa uso produtivo em lugar novo, data-alvo e profundidade. O serviço de teste foi encerrado; este comando inicia uma nova sessão local por decisão do operador.

Abrir `local_runs/ia-integrada-v2/revisao_humana/revisao.html` no navegador para a revisão. Após a pessoa exportar os dois arquivos:

```powershell
& $py -X utf8 scripts/ia/revisao_humana_v2.py validar --csv CAMINHO/respostas_humanas.csv --declaracao CAMINHO/revisor_humano.json
```

O recebimento não promove automaticamente rótulos. A autoria permanece autodeclarada. Avaliações e conferências pesadas devem continuar passando pelo serializador do projeto, com alimentação e travas verificadas. Não executar novamente exportação no pacote existente: uma nova versão exige nova pasta.

`entrega_v1/registro.json` relaciona arquivos, tamanhos, hashes e QA; `codigo/` preserva scripts e testes usados; `ambiente.json` e `pacotes_instalados.txt` registram o ambiente. Essa fotografia não equivale a instalação limpa já testada em outra máquina. Dados brutos volumosos não são duplicados: ficam referidos pelos manifestos/protocolos originais. A conferência do registro recalcula cada hash. Modelo ativo e fontes do plano foram conferidos antes da entrega.

Nenhuma conclusão causal, promoção, validação brasileira ou revisão humana foi criada pela consolidação. O fechamento alcançado é o da candidata funcional de pesquisa e da evidência delimitada acima.
