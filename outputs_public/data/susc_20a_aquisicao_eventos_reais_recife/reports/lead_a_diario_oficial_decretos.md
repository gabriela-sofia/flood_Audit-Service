# Lead A — Diário Oficial do Recife (DOME): extração de decretos de emergência por enchente

**Status**: concluído (PDFs abertos e lidos, não apenas snippets de busca)
**Fonte**: `https://dome.recife.pe.gov.br/dome/buscar.php` (acervo pesquisável desde 30/04/2015; PDFs em `https://dome.recife.pe.gov.br/upload_dome/`)

## Método

Busca por 6 termos ("situação de emergência", "calamidade", "alagamento", "inundação",
"enxurrada", "chuvas intensas") em 12 janelas de 12 meses cobrindo 30/04/2015–23/07/2026
(limite de 12 meses por consulta imposto pelo portal). Total de resultados brutos por termo:
situação de emergência=186, calamidade=208, alagamento=133, inundação=20, enxurrada=0,
chuvas intensas=35 (URLs de PDF únicas). Para filtrar decretos genuinamente de
enchente/chuva (não saúde/fiscal/estiagem), calculou-se a **interseção** entre o conjunto de
edições que citam termo(s) de status jurídico ("situação de emergência"/"calamidade") e o
conjunto que cita termo(s) hidrológico(s) ("alagamento"/"inundação"/"enxurrada"/"chuvas
intensas") na mesma edição — 52 edições candidatas. Todas as 52 foram baixadas
(`upload_dome/*.pdf`) e convertidas com `pdftotext -layout`; o texto completo foi varrido por
janelas de contexto (±1200 caracteres) ao redor de cada termo de status jurídico, checando
presença de termo hidrológico E ausência de indícios de contexto não-hídrico (COVID,
estiagem, saúde pública) antes de aceitar como decreto de enchente.

## Decretos de enchente confirmados (texto integral lido)

| Decreto | Data | COBRADE | Fonte (edição) |
|---|---|---|---|
| **Decreto Municipal nº 35.669** | **28/05/2022** | Chuvas Intensas 1.3.2.1.4, Alagamentos 1.2.3.0.0, Inundações 1.2.0.0 | Edição Extra nº 072, 28/05/2022 |
| Decreto nº 35.778 (prorrogação do 35.669) | 04/07/2022 | mesmo evento | DO 094, 05/07/2022 |
| **Decreto Municipal nº 39.714** | **02/05/2026** | Chuvas Intensas, Alagamentos, Inundações (189,9mm/24h) | Edição Extra nº 052, 02/05/2026 |
| Decretos nº 39.716 e nº 39.718 (alteram ementa/crédito suplementar do 39.714) | 04/05/2026 | mesmo evento | DO 053, 05/05/2026 |

Decretos descartados por não serem de enchente (lidos e confirmados não-hídricos): Decreto
33.511/2020 (COVID-19), Decreto 35.228/2021 (renovação de calamidade em contexto de
pandemia/fiscal, não hídrico).

## Checagem de duplicação vs `dataset_v9_final.csv` (269 pontos, 145 positivos, datas até 2025-01-23)

**Achado importante**: nenhuma data de 2022 ou 2026 existe como registro POSITIVO (label=1)
no dataset atual — só existem 6 registros NEGATIVOS de 2022 (bairro-matched) e nenhum de
2026. Ou seja, ao contrário da suposição inicial ("evento de 2022-05-29 provavelmente já
coberto"), **o evento catastrófico de 28/05/2022 não estava representado como positivo**.
Ambas as datas de decreto (2022-05-28 e 2026-05-02) são genuinamente novas.

## Extração de bairros

O texto do decreto em si (`35.669` e `39.714`) é **somente municipal** — declara emergência
"no âmbito do Município do Recife" sem listar bairros/ruas no corpo do decreto (achado
honesto: contradiz a expectativa inicial de que decretos brasileiros de enchente
frequentemente nomeiam bairros — nestes dois casos não nomeiam).

Entretanto, a **mesma edição extra** (nº 072, 28/05/2022) do Diário Oficial publica, junto ao
decreto, um boletim/matéria de imprensa oficial da Prefeitura ("Prefeitura do Recife ativa
Plano de Contingência...") citando explicitamente 14 escolas/creches abertas como abrigo por
bairro, e localidades de óbitos por deslizamento. Bairros citados (rede municipal e estadual):
**Várzea (Escola Célia Arraes; UR-7/Várzea), Passarinho (Marluce Santiago), Alto da Bela
Vista (Ibura), Linha do Tiro (Paulo VI), Torre (creche Santa Luzia), Roda de Fogo (creche
Miguel Arraes, Torrões), Monteiro (Silva Jardim), Cajueiro (Jarbas Pernambucano)**. Para o
decreto de 2026 (edição nº 052), nenhuma lista de bairros foi encontrada nas edições
subsequentes lidas (053, 055, 058, 060) além de menções genéricas não relacionadas ao evento
(ex.: instalação de contêiner da Guarda Municipal no bairro Recife Antigo).

## Geocodificação (Nominatim, mesmo método do projeto)

8 bairros/localidades geocodificados com sucesso (todos dentro de Recife/PE):

| point_id | bairro | lat | lon | resolução |
|---|---|---|---|---|
| LEADA_2022_0001 | Várzea | -8.045038 | -34.969173 | bairro administrativo |
| LEADA_2022_0002 | Passarinho | -7.981803 | -34.923180 | bairro administrativo |
| LEADA_2022_0003 | Alto da Bela Vista | -8.123130 | -34.941357 | sub-localidade (dentro de Ibura) |
| LEADA_2022_0004 | Linha do Tiro | -8.009652 | -34.905288 | bairro administrativo |
| LEADA_2022_0005 | Torre | -8.042511 | -34.907464 | bairro administrativo |
| LEADA_2022_0006 | Roda de Fogo | -8.056904 | -34.937826 | sub-localidade (dentro de Torrões) |
| LEADA_2022_0007 | Monteiro | -8.026322 | -34.928791 | bairro administrativo |
| LEADA_2022_0008 | Cajueiro | -8.012098 | -34.885663 | bairro administrativo |

**Tier de confiança**: `diario_oficial_decreto_gazette_news_bairro_level` — resolução de
bairro/sub-localidade (centroide), NÃO nível de endereço/ponto. Data do evento atribuída:
2022-05-28 (data de assinatura do decreto e da matéria oficial), consistente com o
enquadramento honesto exigido (não "validado", não "ground truth").

O decreto de 2026-05-02 **não** gerou pontos novos por falta de lista de bairros nas edições
lidas — permanece documentado apenas como achado real de data/decreto (nível município),
mesma limitação estrutural já identificada para o S2ID na rodada v10 (confirma
administrativamente, sem geometria).

## Resultado quantitativo

**8 novos registros positivos** (label=1) extraídos e incorporados ao pipeline v9 completo
(elevação/declividade PE3D, HAND/TWI D-infinito, MapBiomas 2023, chuva Open-Meteo ERA5-Land
com o mesmo método de v9). Ver `dataset_v12_final.csv` e relatório de re-treino.
