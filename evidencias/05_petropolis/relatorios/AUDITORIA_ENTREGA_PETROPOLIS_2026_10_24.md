# Auditoria independente da entrega de Petrópolis

24/10/2026. Escopo: campanha `local_runs/petropolis-aprendizado/v1/`, scripts, nove testes, texto proposto para o artigo, seis figuras PNG, protocolos, estado Git e fontes protegidas listadas. Auditoria aditiva, sem alterar a entrega congelada, reexecutar treinos ou promover modelos. Não equivale a auditoria de todas as alterações históricas do REV-P ou a instalação limpa.

## Parecer

A execução de aprendizado é real, os resultados centrais reproduzem e os nove testes passaram novamente. Não há divergência nos 453 arquivos do manifesto ou nas 54 fontes protegidas verificadas. Entretanto, não foi encontrado commit da entrega neste checkout ou no outro worktree registrado, e existem correções científicas/editoriais necessárias antes de tratar o texto como pronto para submissão.

Resultado quantitativo preservado: modelo externo AUC por objeto 0,66048, declive 0,65390; ganho sobre declive inconclusivo. A atualização local histórica piora; a emenda com cicatrizes também não demonstra ganho sobre o congelado. Isso é diferente de falha de execução: os resultados negativos foram realmente produzidos e reproduzidos.

## O que foi reexecutado

- `tests/test_petropolis_aprendizado_v1.py`: **9 passed in 4.28s**, sem dispensas nessa execução.
- SHA-256 de **453 arquivos**, total de 1.439.910.808 bytes: zero ausentes e zero divergentes.
- SHA-256 das **54 fontes protegidas** listadas no estado inicial: zero divergentes.
- Seis amostras contra seus hashes congelados: todas iguais.
- Métricas de **357 vetores de previsão**: diferenças máximas de 1,19e-9 em AUC por objeto, 5,56e-17 em AUC celular e 9,89e-17 em AP.
- Reinferência de **352 modelos** (dois de origem e 350 ajustes locais) a partir dos pesos/coeficientes salvos: diferença máxima de 3,56e-15; zero ajustes locais sem convergência registrada.
- Métricas fora da dobra de **150 vetores** da emenda: diferença máxima de 2,23e-16. As 735 linhas de ajuste da emenda registram convergência.
- Partição Idai: zero blocos atribuídos a mais de um papel e zero células duplicadas na amostra.
- Recontagem independente da cobertura por comunidade e do afastamento espacial dos positivos da emenda.
- Inspeção visual das seis figuras PNG. SVGs tiveram integridade conferida; não foram todos renderizados separadamente.

Artefato desta auditoria: `local_runs/auditoria-petropolis-20261024/verificacao_independente_nova.json`. O verificador novo não importa as funções de métrica do produtor nem grava na pasta congelada.

## Achados que impedem declarar “tudo commitado e pronto”

### 1. Commit da entrega não encontrado

O checkout `C:/Users/gabriela/Documents/REV-P` está na branch `camada-cognitiva-e-fronteira-de-validade`, HEAD `62f0ee41d13b1adfb8905802dc1b9419f0acad44`, de 20/10/2026. O relatório, o código e o teste de Petrópolis estão não rastreados; não há commit de seus caminhos no histórico local consultado. Antes de adicionar esta auditoria, a inspeção encontrou 51 entradas modificadas e 497 não rastreadas no status (entradas de diretórios podem representar vários arquivos).

O outro checkout local inspecionado tem HEAD `4c7bad414b036c4e8bdda3371c69c541305ad640`, de maio/2026. Também não é o commit da entrega. Essa conferência foi somente leitura; não se presume inexistência de outro repositório não identificado.

`local_runs/` está ignorado pelo Git (`.gitignore`, linha 171). Mesmo um commit de código/documentação não incorporará automaticamente os pesos e dados científicos locais. O manifesto é verificável, mas sua presença em disco não é uma segunda cópia restaurável. Nenhum commit ou cópia de arquivo científico foi criado nesta auditoria.

### 2. Cobertura de 31 comunidades mistura presença interna e proximidade

Em `pa_06_territorio.py`, um relato a até 100 m é incluído no objeto `encosta.observado`. A recontagem dos próprios campos produz:

| Critério | FCU |
|---|---:|
| Inventário cartográfico intersecta a FCU | 5 |
| Há geocódigo histórico dentro da FCU | 18 |
| União de inventário e geocódigo dentro | **20** |
| União admitindo relato a até 100 m | **31** |
| Acréscimo apenas por proximidade, sem registro interno | **11** |

Portanto, a frase “31 comunidades com registro observado de encosta” precisa explicitar o vínculo por proximidade. Formulação adequada: “20 FCU com interseção de inventário ou geocódigo interno; 31 ao incluir relatos na vizinhança de até 100 m”. Mesmo um geocódigo interno mantém a incerteza posicional do relato e não comprova ruptura dentro da FCU. A população associada ao critério de 100 m não deve ser descrita como população de comunidades com ocorrência física confirmada.

Separar `presenca_cartografica_intersectada`, `relato_geocodificado_interno` e `relato_proximo_100m` na próxima versão. Regerar cobertura e figura 6 com esses critérios. Preservar a v1 como registro histórico.

### 3. Ausência mapeada na origem não está estabelecida pela envoltória

O código `pa_03_referencias.py` define o suporte Idai por `idai_src.union_all().convex_hull` e constrói negativos fora das feições mapeadas e suas dilatações. O catálogo adquirido descreve pontos e extensões de deslizamentos; não fornece nessa cadeia uma máscara de área uniformemente inspecionada que comprove ausência em toda a envoltória.

A existência de um inventário abrangente não basta para demonstrar que toda célula sem feição foi inspecionada e admitida como ausência. Antes de afirmar “referência de duas classes” e “ausência mapeada” na origem, ligar cada célula negativa a suporte/critério de inspeção documentado. Até essa comprovação, qualificar o contraste como fonte inventariada contra fundo sem detecção registrada. Os escores e métricas salvos continuam reproduzindo; muda a interpretação permitida da referência, não os valores calculados.

### 4. O artigo afirma equivalência e não inferioridade sem teste correspondente

Em `docs/ARTIGO_PETROPOLIS_ENCOSTAS_SECOES_2026_10_24.md`, a discussão afirma que o modelo externo “não é pior”, e a conclusão diz “equivalente à regra de declive”. O C1 é +0,00658, IC95 [-0,01861; +0,03168], classificado como inconclusivo pelo protocolo. Ausência de superioridade não estabelece equivalência ou não inferioridade.

Substituir por “não houve ganho demonstrado sobre a regra de declive neste desenho”. Não é necessário alterar resultado, treino ou limiar para corrigir a afirmação.

### 5. A renda agregada precisa de definição correta

`pa_06_territorio.py:190` calcula uma média ponderada por população de `renda_resp` setorial e a apresenta como `renda_resp_media_ponderada`, com estado social `observado`. A fonte local usa medianas do rendimento do responsável com rendimento. A média ponderada de medianas não é renda média, renda per capita ou mediana da FCU; ponderar por todos os moradores ainda troca a população do denominador.

Usar valores e cobertura por setor ou construir um indicador explicitamente descritivo da distribuição setorial. Não atribuir à comunidade uma estatística censitária de renda que a operação não identifica. A figuração territorial deve separar totais censitários de agregações calculadas.

### 6. O texto D5 troca o modelo do número citado

O CSV `resultados/emenda1/D5_receptor_T_hist.csv` mostra, nos geocódigos precisos: `A0_cel` (declive) = **0,32947**; `A1_cel` (modelo externo) = **0,37873**; `A1_montante7` = **0,46509**. O artigo relata a sequência do modelo externo e depois usa 0,33 sem indicar a mudança para declive. Corrigir a identificação: o valor do modelo nesse estrato é aproximadamente 0,38. A figura 4 apresenta essa sequência corretamente.

O resultado não autoriza eliminar o erro de geocodificação como explicação: o rótulo de precisão do geocódigo não é validação de campo do local da ruptura.

## Limites metodológicos adicionais, já parcialmente declarados

- A emenda aplica afastamento de 150 m ao fundo, não aos positivos. A auditoria encontrou positivos de ajuste a 30 m de células de teste nas dobras Alto da Serra e Chácara Flora, e a 60 m na dobra Centro. Atribuições de cicatrizes às dobras são distintas, mas há dependência espacial residual e sobreposição possível das janelas de atributos. Manter como desenvolvimento espacial do mesmo episódio; não chamar de avaliação espacialmente isolada ou independente.
- A padronização externa é compartilhada por A2 e A3. A2 é um controle sem coeficientes externos, não uma cadeia integralmente sem informação externa.
- Estar dentro de P1–P99 marginal em sete atributos é diagnóstico de faixa de entradas; não demonstra aplicabilidade multivariada, erro controlado ou validação em 37 FCU.
- Recusar a atualização após examinar o contraste é uma regra de seleção/uso. Não demonstra que uma competência de reconhecer erros foi aprendida e generaliza para unidades inéditas.
- O teste é presença–fundo no suporte derivado das feições. A ordenação acima de 0,5 é evidência nesse desenho; não identifica especificidade, calibração física ou segurança de toda Petrópolis.

## Relação com o pedido de execução

Houve aquisição real, treinamento real, controles pareados, avaliação, pesos, previsões, retenção, integração territorial, seis figuras e texto para revisão. A campanha não ficou apenas no planejamento. Entretanto, a expectativa de ganho após atualização local não se confirmou. Os valores sustentam aprendizado na tarefa de origem e ordenação transferida do inventário; não superioridade sobre declive nem benefício líquido da atualização local nas formulações testadas.

Portanto: **execução e reprodução aprovadas dentro do escopo auditado; conclusão científica e editorial exigem as correções acima; commit local da entrega não confirmado.** Não declarar validação integral da tese de aprendizado/adaptação em Petrópolis com base somente nos nove testes.

## Reprodução desta auditoria

O ambiente da entrega é `C:/Users/gabriela/miniconda3/envs/revp-susc/python.exe` (Python 3.10). Não existe `.venv/Scripts/python.exe` nem `setup_env.ps1` neste checkout; não foi instalada outra pilha. Ativar as pastas nativas do ambiente no PATH antes de importar SciPy e GDAL.

Teste executado:

```text
C:/Users/gabriela/miniconda3/envs/revp-susc/python.exe -B -m pytest tests/test_petropolis_aprendizado_v1.py -q -p no:cacheprovider --basetemp local_runs/auditoria-petropolis-20261024/pytest_tmp
```

Verificação nova, pelo serializador, sem treino:

```text
C:/Users/gabriela/AppData/Local/Programs/Python/Python312/python.exe -B scripts/manutencao/executar_serializado.py --esperar-min 1 -- C:/Users/gabriela/miniconda3/envs/revp-susc/python.exe -B scripts/planejamento/auditar_entrega_petropolis_20261024.py
```

O verificador próprio foi otimizado para evitar cópias redundantes de matrizes. Corrigiram-se também serialização de booleano NumPy e saída de console Unicode no ambiente. Esses incidentes pertencem à auditoria nova; não alteraram a entrega congelada ou seus resultados. Os testes da entrega não foram apresentados como reexecução de toda a suíte histórica do projeto.

O recálculo completo gravou o JSON novo com `ok_numerico=true`; nessa execução, apenas a impressão final no console falhou por codificação. A saída foi corrigida no verificador. A tentativa adicional de repetir o comando depois dessa correção aguardou o serializador e desistiu por haver processamento de Recife ativo (PID 6084). Esse processo não foi interrompido. As conferências numéricas citadas acima são as efetivamente concluídas e gravadas, não uma repetição posterior presumida.
