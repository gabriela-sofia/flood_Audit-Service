# v11 — Busca de Fontes Governamentais e Satelitais (Diário Oficial, Global Flood Database, ANA, dados.gov.br, UFPE)

**Data da execução**: 2026-07-23
**Baseline de comparação**: `local_runs/recife_modelo_v9_final/dataset_v9_final.csv` (n=269, LOO-AUC=0.6578)
**Resultado desta rodada**: nenhum registro novo de evento (ponto ou data) foi adicionado. Dataset e modelo permanecem em v9. Ver justificativa por lead abaixo — cada lead foi acessado de verdade (URLs e respostas reais documentadas), não descartado por suposição.

---

## Lead 1 — Diário Oficial do Recife (DOME)

**Acesso real**: `https://dome.recife.pe.gov.br/dome/buscar.php` — portal funcional, com busca textual real (não é só um formulário morto).

Achados concretos:
- O acervo pesquisável (`Acervo desde 30/04/2015`) só cobre **30/04/2015 em diante**. Edições anteriores exigem `https://www.recife.pe.gov.br/diariooficial-acervo/` — esse link retornou uma página com encoding ISO-8859-1 quebrado e sem indício de índice de busca full-text (aparenta ser um repositório de PDFs por data, não pesquisável por termo).
- A busca impõe **janela máxima de 12 meses por consulta** (mensagem literal do site: "Intervalo de consulta máximo de 12 meses"), então cobrir 2015–2026 exige ~11 consultas separadas por termo.
- Testei a janela 01/05/2022–30/04/2023 (o ano do megaevento de maio/2022) com o termo "situação de emergência": retornou **15+ edições** contendo o termo (2 páginas de resultados), incluindo a edição de 11/08/2022 com trecho literal: *"...extraordinários, no âmbito da Secretaria de Saúde, durante a Situação de Emergência declarada pelo Decreto..."*.
- Com o termo "calamidade" na mesma janela: **encontradas ~18+ edições**, várias citando o **Decreto estadual nº 50.434, de 15 de março de 2021**, que mantém "Estado de Calamidade Pública" no âmbito do Estado (esse decreto de 2021, pelo padrão de outras buscas, está associado a estiagem/seca em PE, não a enchente — não confirmado como decreto de inundação sem abrir o PDF integral).
- Os resultados de busca mostram apenas snippets curtos (uma frase de contexto) e o número/data da edição — **não o texto integral do decreto nem bairros afetados**. Para extrair datas de decreto e bairros citados seria necessário abrir cada PDF de edição individualmente (dezenas de PDFs), o que não foi feito nesta rodada por escopo/tempo.

**Conclusão do lead**: acervo é real, público e pesquisável por termo a partir de 2015 — mas (a) não estende a cobertura para antes de 2015 (o problema original), e (b) não produziu, nesta passada, nenhum decreto com bairro nomeado extraído e verificado. **Nenhum registro novo adicionado.** Fica como pendência concreta para uma sessão dedicada a abrir os PDFs das edições já localizadas (lista de datas acima) e extrair texto integral.

---

## Lead 2 — Diário Oficial do Estado de Pernambuco (CEPE)

**Acesso real**: `https://diariooficial.cepe.com.br/` — o fetch retornou **página em branco** (corpo vazio, sem HTML de conteúdo), consistente com um front-end que renderiza via JavaScript client-side sem fallback para requisição simples GET. Não há endpoint de busca acessível sem executar JS.

Achados via busca (não via acesso direto ao portal):
- Notícias confirmam publicação, em edição extra do Diário Oficial do Estado, do decreto que reconhece situação de emergência em **27 municípios** (incluindo Recife) por chuvas fortes — mencionado por CBN Recife e Agência Brasil, com validade de 180 dias.
- Também localizado: nota da SDS-PE ("Governador declara Estado de Calamidade Pública em algumas cidades atingidas pelas chuvas").
- Nenhum desses decretos foi acessado em texto integral (a busca no portal CEPE não é executável sem JS neste ambiente).

**Conclusão do lead**: fonte real, mas **inacessível a extração de texto nesta sessão** — página requer JavaScript e não há endpoint de API/busca documentado publicamente. Nenhum registro novo adicionado.

---

## Lead 3 — Dartmouth Flood Observatory / Global Flood Database (Cloud to Street)

**Acesso real**: catálogo Earth Engine (`developers.google.com/.../GLOBAL_FLOOD_DB_MODIS_EVENTS_V1`) e site do produto (`global-flood-database.cloudtostreet.info`) acessados com sucesso.

Achados concretos:
- Dataset real: **913 eventos de inundação mapeados globalmente, 2000-02-17 a 2018-12-10**, 250m de resolução (bandas a 30m para alguns produtos), derivado de imagens MODIS Terra/Aqua, com bandas `flooded`, `duration`, `clear_views`, `clear_perc`, `jrc_perm_water`, e metadados por evento (`dfo_dead`, `dfo_displaced`, `dfo_severity`, `dfo_main_cause`, centróide lat/lon do polígono DFO).
- **Limitação crítica e definitiva**: a cobertura temporal (2000–2018) **termina antes do evento catastrófico de maio/2022 em Recife** (>130 mortos na região metropolitana), que é justamente o evento mais bem documentado e citado nas buscas. Não há garantia de que algum evento anterior a 2018 no Recife tenha passado no controle de qualidade do GFD (não confirmado — exigiria consulta ao asset via Earth Engine).
- **Acesso aos dados em si**: requer autenticação Google Earth Engine (`ee.ImageCollection('GLOBAL_FLOOD_DB/MODIS_EVENTS/V1')`, executável só via Code Editor ou API Python com conta registrada) — não há export CSV/REST público sem essa autenticação, e este ambiente não tem credenciais de Earth Engine configuradas.

**Conclusão do lead**: fonte real e cientificamente citável, porém (a) sua janela temporal não cobre o evento mais relevante para Recife, e (b) a extração de qualquer evento pré-2018 exigiria acesso GEE que não está disponível neste ambiente. **Nenhum registro extraído/adicionado.**

---

## Lead 4 — ANA HidroWeb (Capibaribe, Beberibe, Tejipió)

**Acesso real**: `dadosabertos.ana.gov.br` (portal ArcGIS Hub) e `snirh.gov.br/hidroweb/serieshistoricas` acessados — ambos retornaram apenas metadados de página (título, descrição do serviço), sem tabela de dados embutida na resposta simples (front-end dinâmico ArcGIS/Angular).
- `www.ana.gov.br/hidrowebservicos/EstacoesTelemetricas.asmx` **redirecionou para a home do gov.br/ana** — o endpoint SOAP legado não respondeu como serviço de dados via GET simples neste ambiente.

Achados confirmados via busca (não via download direto):
- ANA de fato disponibiliza séries históricas de cota/vazão por estação fluviométrica via HidroWeb, incluindo estatísticas hidrometeorológicas por estação (confirmado pelo dataset "Índices e Estatísticas Hidrometeorológicas — Estação Fluviométrica" no portal `dadosabertos.ana.gov.br`).
- O rio Beberibe e o Capibaribe são cursos d'água urbanos confirmados em Recife, mas **não foi possível, nesta sessão, identificar o código de uma estação fluviométrica específica dentro do perímetro urbano de Recife** nem baixar sua série histórica — isso exigiria navegar a interface interativa do HidroWeb (mapa/inventário de estações) ou consultar a API REST do ArcGIS Feature Service com uma query espacial (bbox de Recife), o que não foi executado por limitação de escopo desta rodada (a página consultada só retornou metadados, não o serviço de features em si).

**Conclusão do lead**: fonte real e existente, mas **não convertida em dado extraído** nesta sessão — falta uma consulta direcionada ao Feature Service REST da ANA com bbox de Recife para listar códigos de estação, que fica como próximo passo concreto e específico (não "esgotado"). Nenhum registro adicionado.

---

## Lead 5 — dados.gov.br (catálogo federal)

**Acesso real**: `dados.gov.br/dados/conjuntos-dados?groups=meio-ambiente&q=inundação+recife` retornou explicitamente: **"Essa página requer javascript para funcionar corretamente"** — confirma que o catálogo federal é uma SPA (Angular/React) sem fallback de conteúdo estático, portanto não indexável por fetch simples neste ambiente.

Via busca externa (Google/Bing, não o portal em si):
- Confirma-se a existência do dataset municipal já conhecido — `Defesa Civil` em `dados.recife.pe.gov.br/ne/dataset/defesa-civil` (acessado com sucesso, ver detalhe abaixo) — mas nenhum dataset federal adicional específico de Pernambuco/Recife apareceu nos resultados de busca além do que já era conhecido (CPRM, APAC, mapeamentos de risco já referenciados na literatura).

**Detalhe extra obtido (dataset já conhecido, mas confirmando estrutura atual)**: o dataset `Defesa Civil` do Portal de Dados Abertos do Recife (CKAN 2.11.4) contém 4 recursos: dicionário de dados das Áreas de Risco (PDF/JSON), coordenadas geográficas da Região Sul (GeoJSON) e Áreas de Risco da Regional Sul (CSV) — última atualização registrada: **2 de março de 2026**. Isso é uma camada de "áreas de risco" (polígonos de risco geotécnico/geológico, mantidos por SEDEC), não um registro de eventos de inundação com data — mesma limitação já conhecida de rodadas anteriores (v9/v10).

**Conclusão do lead**: portal federal inacessível a fetch simples nesta sessão (requer JS); nenhum dataset novo e utilizável identificado além do que já estava mapeado. Nenhum registro adicionado.

---

## Lead 6 — Repositório Institucional UFPE

**Acesso real**: busca no repositório retornou 3 dissertações/TCCs relevantes, dois deles obtidos e lidos (texto extraído com sucesso):

1. **Bárbara Mantovani (2016)** — *"Mapeamento de risco a movimentos de massa e inundação em áreas urbanas do município de Camaragibe"* (`repositorio.ufpe.br/bitstream/123456789/17957/...`, PDF de 288 folhas, acessado e lido integralmente). Confirma-se: 102 áreas de risco a movimento de massa + **2 áreas de risco a inundação** mapeadas, mas o objeto de estudo é **Camaragibe (município vizinho, não Recife)**. Não aplicável ao dataset atual (escopo é Recife-only).
2. **Lucas de Siqueira Santos** — dissertação de mapeamento de suscetibilidade a inundações em **Jaboatão dos Guararapes** (modelo HAND + MDT PE3D LIDAR 1m) — também **município vizinho, não Recife**.
3. **Artigo (não é do repositório UFPE, mas de coautoria UFPE/IFPE)**: Lima et al. 2025, *"Mapeamento de Áreas Suscetíveis a Inundações na Cidade do Recife-PE/Brasil"* (Espaço em Revista, DOI 10.70261/er.v27i1.74841, acesso aberto CC-BY-4.0, PDF disponível). Este é o único achado **específico de Recife**. Confirma citação explícita a eventos históricos de 1966, 1975 e maio/2022 (>130 mortos na RMR), e usa AHP com 5 classes de suscetibilidade (pluviometria, declividade, uso do solo) — **é um modelo de suscetibilidade espacial, não um inventário de eventos com data/local pontual**; não contém tabela suplementar de pontos de ocorrência histórica extraível.

**Conclusão do lead**: repositório real e acessado com sucesso; os 2 trabalhos com dado geoespacial mais ricos (HAND, PE3D) são de **municípios vizinhos (Camaragibe, Jaboatão), não Recife**, então não substituem/complementam diretamente o dataset Recife-only sem uma decisão explícita de expandir escopo (não tomada aqui, fora do mandato desta tarefa). O único artigo Recife-específico encontrado é um mapa de suscetibilidade sem tabela de eventos pontuais anexa. **Nenhum registro adicionado.**

---

## Síntese

| Lead | Fonte real acessada? | Dado novo extraído? | Motivo específico de bloqueio (não "esgotado" genérico) |
|---|---|---|---|
| 1. Diário Oficial Recife | Sim (`dome.recife.pe.gov.br`, busca funcional) | Não | Acervo pesquisável só cobre 2015+; snippets de busca não trazem texto integral do decreto nem bairros; extração exigiria abrir dezenas de PDFs individuais (não feito nesta rodada) |
| 2. Diário Oficial PE (CEPE) | Portal acessado, conteúdo vazio | Não | Front-end 100% JS-renderizado, sem fallback estático nem API pública documentada |
| 3. Global Flood Database | Sim (catálogo GEE completo) | Não | Cobertura 2000-2018 não inclui o evento mais relevante (maio/2022); extração de eventos pré-2018 exige autenticação Earth Engine indisponível neste ambiente |
| 4. ANA HidroWeb | Portal acessado, só metadados | Não | Front-end dinâmico (ArcGIS Hub/Angular); endpoint SOAP legado redirecionou; faltou query espacial ao Feature Service REST (próximo passo concreto, não tentado ainda) |
| 5. dados.gov.br | Portal acessado, bloqueado | Não | SPA requer JavaScript ("Essa página requer javascript para funcionar corretamente") |
| 6. UFPE repositório | Sim, 3 documentos lidos | Não | Os 2 trabalhos com geodados ricos são de municípios vizinhos (Camaragibe, Jaboatão), não Recife; o único artigo Recife-específico é suscetibilidade sem tabela de eventos |

**Total de registros novos adicionados por camada de confiança**: 0 (gazette-decree-tier: 0; satellite-extent-tier: 0; gauge-tier: 0).

**Dataset**: inalterado — `dataset_v9_final.csv` (n=269) permanece a versão vigente.
**LOO-AUC**: inalterado — **0.6578** (v9), nenhum re-treino realizado (não há dado novo para justificar).

**Avaliação honesta**: esta rodada não encontrou terreno novo em termos de registros utilizáveis, mas por razões técnicas concretas e diferentes de "não encontrado" — cada lead tem um bloqueio de acesso específico e documentado (paywall de JS, janela temporal insuficiente, necessidade de autenticação GEE, escopo municipal vizinho). Os leads 1 (Diário Oficial Recife) e 4 (ANA Feature Service) são os mais promissores para uma sessão futura dedicada, pois ambos têm um caminho técnico claro e não tentado até o fim (abrir PDFs de edições já localizadas; query espacial ao Feature Service da ANA), diferentemente dos leads 2, 3 e 5, que têm bloqueios estruturais mais duros (JS-only front-end sem API; dependência de autenticação externa).
