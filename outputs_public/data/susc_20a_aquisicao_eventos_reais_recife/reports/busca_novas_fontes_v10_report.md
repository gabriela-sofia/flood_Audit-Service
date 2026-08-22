# Busca de Novas Fontes Reais — Recife (v10)

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-23
**Autorizado por**: Gabriela Sofia (reviewer/executora única do projeto)
**Objetivo**: buscar fontes REAIS e independentes de eventos de alagamento/inundação (positivos)
ou não-eventos (negativos) para Recife, ainda não usadas nas 9 iterações anteriores (v1-v9,
baseline atual n=269, LOO-AUC=0,6578).
**Não fabricação**: nenhuma coordenada, data ou valor fabricado. Toda fonte checada é
documentada com resultado real (encontrado/não encontrado/inacessível) e razão concreta.

---

## Resumo executivo

Nenhuma das 5 linhas de busca produziu pontos novos, geocodificáveis e não-duplicados
utilizáveis no pipeline de features (que exige lat/lon pontual). Duas fontes (S2ID / Atlas
Digital de Desastres) retornaram dados **reais e novos como registro administrativo**, mas
**sem resolução espacial pontual utilizável** — um achado negativo honesto e bem documentado,
não uma busca malsucedida. As demais 3 linhas (Wayback pré-2014, CEMADEN histórico, notícias)
também não geraram dado novo utilizável, por razões estruturais concretas detalhadas abaixo.

**Nenhuma atualização de dataset foi feita.** `dataset_v9_final.csv` (n=269, LOO-AUC=0,6578)
permanece a versão vigente. Não há `dataset_v10_expandido.csv` nem re-execução de modelo,
porque não havia dado real novo e utilizável para incorporar.

---

## Lead 1+2: S2ID (Sistema Integrado de Informações sobre Desastres) e Atlas Digital de
Desastres (CEPED/UFSC + Sedec/MIDR)

**Status: ENCONTRADO, MAS NÃO UTILIZÁVEL EM NÍVEL DE PONTO.**

O Atlas Digital de Desastres (`atlasdigital.mdr.gov.br`) é hoje o sucessor oficial e a
interface pública consolidada dos dados S2ID (1991–2025), mantido por Sedec/MIDR em parceria
com Ceped/UFSC — ou seja, as duas fontes das leads 1 e 2 convergem no mesmo banco de dados
público. Foi baixado o arquivo público completo (`BD_Atlas_1991_2025_v1.0_2026.04.23_Consolidado.csv`,
86.127.670 bytes, 145.734 linhas, disponível em
`https://atlasdigital.mdr.gov.br/arquivos/BD_Atlas_1991_2025_v1.0_2026.04.23_Consolidado.csv`)
e filtrado para `Nome_Municipio=Recife; Sigla_UF=PE`.

**16 registros oficiais para Recife (1991–2022)**, dos quais **8 são hidrológicos relevantes
para alagamento/inundação/enxurrada** (excluindo movimento de massa, estiagem, doenças
infecciosas):

| Data_Evento | Cod_Cobrade | Tipologia |
|---|---|---|
| 25/05/1991 | 12200 | Enxurradas |
| 22/06/1994 | 12200 | Enxurradas |
| 29/04/1996 | 12200 | Enxurradas |
| 23/06/1997 | 12200 | Enxurradas |
| 03/08/2000 | 12200 | Enxurradas |
| 12/08/2008 | 12100 | Inundações |
| 20/04/2010 | 12200 | Enxurradas |
| 13/06/2012 | 12300 | Alagamentos |
| 29/05/2022 | 13214 | Chuvas Intensas |

**Por que não são utilizáveis como novos pontos do dataset**:
- **7 dos 8 eventos (1991–2012) têm o campo `Setores Censitários` vazio** — ou seja, o registro
  oficial confirma data + tipo de desastre para o **município inteiro**, sem qualquer
  geometria, bairro ou endereço. Atribuir qualquer coordenada a esses eventos (mesmo um
  centróide do município) seria fabricar precisão espacial que a fonte não fornece —
  exatamente a limitação que os documentos internos do projeto já haviam antecipado
  (`fase2_decisao_label_curitiba_petropolis.md`: "S2ID confirma data e tipo de desastre
  administrativamente, mas não tem geometria").
- **O evento de 29/05/2022 tem 1.737 códigos de setor censitário listados** — na prática, isso
  cobre a maior parte da malha urbana do município (evento de escala municipal, a mesma chuva
  intensa de maio/2022 já amplamente representada no dataset atual via SEDEC). Usar centróides
  de setores censitários aqui não agregaria localização nova nem discriminação espacial
  significativa, e duplicaria um evento/data já saturado no dataset v9.
- Confirmado por checagem cruzada: `dataset_v9_final.csv` já contém registros datados de maio
  de 2022 (ex.: `NEWNEG_V9_BAIRROMATCH_070`, 2022-05-25) — não há sobreposição de datas exatas
  com 29/05/2022, mas o mesmo evento climático já está representado.

**Valor real, não-descartável**: os 7 eventos pré-2014 (1991, 1994, 1996, 1997, 2000, 2008,
2010) são uma confirmação histórica administrativa **genuína e nova** de que Recife tem
histórico de enxurradas/inundações reconhecido federalmente desde pelo menos 1991 — útil como
contexto qualitativo/narrativo na dissertação (ex.: discussão de recorrência histórica), mas
**não como ponto de treino**, por ausência estrutural de geometria na fonte.

---

## Lead 3: Extensão temporal do SEDEC via Wayback Machine (portal `dados.recife.pe.gov.br`)

**Status: CHECADO, GENUINAMENTE VAZIO — limitação estrutural do próprio portal, não de
acesso.**

- O primeiro snapshot do domínio `dados.recife.pe.gov.br` no Internet Archive é de
  **20/08/2013** — ou seja, o portal de dados abertos da Prefeitura do Recife só passou a
  existir cerca de 5 meses antes da data mais antiga já presente no dataset atual
  (2014-01-03).
- No snapshot de 20/08/2013, os únicos datasets existentes eram genéricos (área urbana, árvores
  tombadas, censo escolar) — **nenhum dataset de Defesa Civil ainda existia**.
- O primeiro dataset da Secretaria Executiva de Defesa Civil aparece nos snapshots de
  02/12/2013 e 22/02/2014, sob a URL `dataset/defesa-civil` (não a URL
  `dataset/registro-de-atendimentos-da-defesa-civil` já usada em v1-v9). Seu conteúdo, checado
  diretamente no HTML arquivado, era **apenas 3 recursos estáticos**: um dicionário de metadados
  em PDF, um GeoJSON de "Áreas de Risco da Região Sul" e um CSV de mapeamento de áreas de risco
  da Regional Sul — um mapa estático de áreas de risco, **sem registros de ocorrência com data
  de evento**, e cobrindo apenas uma regional (não o município inteiro.
- O dataset `registro-de-atendimentos-da-defesa-civil` (que é a fonte real usada nas iterações
  anteriores do projeto) só aparece no Wayback a partir de **2025** — não há snapshots
  arquivados de versões anteriores desse dataset especificamente, o que impede reconstruir
  versões históricas mais antigas dele via Wayback.

**Conclusão honesta**: não existe margem real para estender o SEDEC para trás de 2014 via
Wayback Machine — o portal em si não tinha dados de ocorrência de Defesa Civil antes disso.
Esta é uma limitação de existência de dado na fonte, não de acesso ou de esforço de busca.

---

## Lead 4: Arquivo histórico de alertas CEMADEN (distinto da API de estações em tempo real)

**Status: CHECADO, NÃO ENCONTRADO / NÃO PÚBLICO.**

- A seção oficial "Boletins e Relatórios" do CEMADEN (`gov.br/cemaden/.../monitoramento`)
  disponibiliza apenas boletins agregados nacionais/regionais (Boletim de Impactos de
  Extremos, Estado do Clima/Extremos/Desastres no Brasil, Monitoramento e Avaliação de Impactos
  da Seca, Monitoramento Hidrológico, Notas Técnicas) — nenhum é um banco de alertas históricos
  navegável por município/bairro.
- O "Mapa Interativo" (`mapainterativo.cemaden.gov.br`) só oferece download de séries de
  estações (pluviômetros, hidrológicas, Acqua, geotécnicas, radares) por UF/município/mês/ano —
  a mesma API de estações em tempo real já testada e esgotada em rodadas anteriores do projeto,
  não um arquivo de alertas distinto.
- Uma busca dirigida encontrou referência a uma plataforma computacional interna do CEMADEN
  (projeto "IDFAdmin", notícia de 28/11/2017) que faz retroanálise de alertas históricos — mas
  é uma ferramenta de uso interno dos pesquisadores do CEMADEN (estudos de caso em Salvador/BA e
  Bauru/SP), **sem interface pública de consulta**, e sem indicação de cobertura para Recife.
- Não foi localizada nenhuma URL pública funcional do tipo "alerta histórico por
  bairro/município" (`alerta.cemaden.gov.br`/`alerta2.cemaden.gov.br` não respondem com conteúdo
  navegável relevante).

**Conclusão honesta**: o CEMADEN não expõe publicamente um arquivo de alertas históricos
distinto da API de estações já exaurida — a lead é real como conceito, mas a ferramenta
correspondente é interna/não pública.

---

## Lead 5: Cross-referência de notícias (Diário de Pernambuco / Folha de Pernambuco) —
tier de confiança inferior, last-resort

**Status: RECONHECIMENTO LIMITADO REALIZADO; NÃO APROFUNDADO — baixo indício de rendimento
proporcional ao esforço.**

Como as leads 1–4 não produziram dado novo utilizável (mas também não foram "insuficientes" no
sentido de indicar que uma mineração jornalística maciça resolveria algo que fontes oficiais já
não resolvem — o gargalo é resolução espacial, não volume de eventos conhecidos), esta lead foi
tratada, conforme instruído, como último recurso e com expectativa de baixa confiança.

Uma busca exploratória no Diário de Pernambuco (que mantém arquivo digital próprio) retornou
majoritariamente **matérias qualitativas sobre pontos de alagamento recorrentes** (ex.: lista de
bairros cronicamente alagáveis como Afogados, Jordão, Várzea, Mustardinha, citada em reportagens
de 2016) — não eventos individuais com data e endereço específico e verificável, que é o que
seria necessário para gerar um ponto geocodável novo e não-duplicado com confiança defensável.
Extrair eventos individuais verdadeiramente novos exigiria leitura e verificação manual de
dezenas a centenas de matérias individuais, com risco relevante de (a) duplicar eventos já
presentes via SEDEC/EMLURB, ou (b) introduzir imprecisão de data/local a partir de texto
jornalístico não estruturado.

Dado o baixo indício de rendimento e o enquadramento explícito desta lead como tier inferior/
último recurso, **a mineração sistemática completa não foi realizada** — não foi "forçada" uma
extração de baixa confiança sem evidência de que valeria a pena. Isso é reportado como um limite
de escopo desta rodada, não como uma tentativa fracassada.

---

## Overlap / duplicação

Nenhum ponto novo foi adicionado, portanto não há risco de duplicação a mitigar nesta rodada.
A checagem de sobreposição de data feita para o evento de 29/05/2022 (Atlas Digital) contra
`dataset_v9_final.csv` está documentada na seção do Lead 1+2 acima.

---

## Veredito final

Das 5 linhas de busca, 2 (S2ID/Atlas Digital) retornaram dado administrativo real e novo, mas
estruturalmente não-espacializável em nível de ponto; as outras 3 (Wayback pré-2014, CEMADEN
histórico, notícias) confirmaram limitações genuínas de existência/acesso de dado, não falhas de
busca. **Este projeto, neste nível de acesso (sandbox sem credenciais institucionais, sem acesso
a bases pagas de imprensa, sem acordo de acesso a microdados do CEMADEN/Sedec), parece ter
efetivamente esgotado o cenário de dado real acessível publicamente para Recife.** Novas fontes
poderiam existir em nível de acesso mais alto (ex.: solicitação formal de microdados ao
CEMADEN via LAI/SIC, parceria direta com EMLURB/SEDEC para dados de vistoria com endereço,
ou assinatura de acervo de imprensa com busca full-text) — mas essas exigem ação institucional
fora do escopo de uma busca automatizada, não mais uma busca de portal público.

## Arquivos desta rodada (v10)

Apenas este relatório (`busca_novas_fontes_report.md`). Não há `dataset_v10_expandido.csv`,
`analise_primaria_v10.md` nem `RELATORIO_v10_COMPARATIVO.md` porque nenhum dado real novo e
utilizável foi obtido — `dataset_v9_final.csv` (n=269, LOO-AUC=0,6578) permanece a versão
vigente e nenhuma reexecução de modelo foi justificada.
