# SUSC-20K — SIAC 156 (Curitiba): equivalente administrativo ao SEDEC, achado e minerado

**Status**: RESULTADO REAL, POSITIVO, PARCIAL. Primeira fonte administrativa geocodificável real
pra Curitiba — mesma estrutura que fez o modelo de Recife funcionar (SEDEC: 91,6% dos 154
positivos vieram de reclamação administrativa geocodificada, não de imagem de satélite). Não
substitui adjudicação individual completa (mesmo padrão do SUSC-20A/Valparaíso/Juvevê); produz um
pool de candidatos com triagem física real, pronto pra próxima rodada de decisão.

**Escopo**: só Curitiba. Petrópolis não tem equivalente público (ver seção 5). Nenhum modelo
treinado, nenhum rótulo criado, nenhuma feature de imagem/score usada como entrada.

---

## 1. A fonte: SIAC 156

Sistema Integrado de Atendimento ao Cidadão — Central 156 de Curitiba. Portal
`dadosabertos.curitiba.pr.gov.br`, dataset "SIAC 156 - Dados Abertos". Estrutura confirmada no
dicionário de dados oficial:

| Campo | Papel |
|---|---|
| `Assunto` / `Subdivisao` | categoria do pedido — inclui literalmente "Alagamentos" e "Enchentes, inundações ou alagamentos" |
| `Logradouro` / `Bairro` | endereço de atendimento — geocodificável |
| `DataCriacao` | data do pedido |

Acesso: sem login, sem token, arquivo CSV diário (endpoint AJAX real descoberto:
`/ConjuntoDado/DownloadArquivos/`, servido por `mid-dadosabertos.curitiba.pr.gov.br`). Cada
arquivo diário é um extrato acumulado do mês corrente até aquela data (não só o dia). 572
arquivos diários disponíveis no portal (~1,5 ano de histórico rolante); mais antigo que isso,
existe espelho mensal na UFPR (`dadosabertos.c3sl.ufpr.br/curitiba/156/`, 2016–2022) e rollups
anuais — não usados nesta rodada.

## 2. Evidência independente: pico real de reclamação nos 4 dias de evento

Antes de geocodificar qualquer coisa, a contagem diária de reclamações com assunto de
enchente/alagamento já confirma os 4 eventos FORTE do registro `v20i` de forma independente da
notícia/S2ID (`registries/v20k_siac156_contagem_diaria_baseline.csv`, 44 dias):

| | Dias de evento (n=4) | Dias sem evento conhecido (n=40) |
|---|---:|---:|
| Média de reclamações/dia | **61,5** | 5,3 |
| Mediana | — | 2,0 |
| Máximo | 90 (18/02/2025) | 44 (13/02/2025 — pico não catalogado no v20i, não investigado aqui) |

Os 4 datas de evento (28/01/2025=57, 03/02/2025=46–57 conforme arquivo, 18/02/2025=90,
03/02/2026=53) formam um pico de **10 a 30× a mediana de fundo**. Isso é confirmação
independente e barata (sem geocodificar nada) de que o sinal administrativo captura os eventos
reais catalogados — não é ruído de fundo constante.

## 3. Extração: 251 reclamações brutas, 33 na lista de bairros da notícia

`scripts/extract_flood_complaints_siac156.py` filtra por `DataCriacao` exata + assunto de
enchente + (opcional) bairro batendo com a lista já registrada em `v20i` pra cada evento.

| Evento | Bairros esperados (fonte: notícia/S2ID já no v20i) | Reclamações flood totais no dia | Batendo bairro esperado |
|---|---|---:|---:|
| CUR_2025_02_03 | Boqueirão, Xaxim, Pinheirinho, Parolin, Portão, Vila Izabel, Seminário | 63 | 19 |
| CUR_2025_01_28 | Centro Cívico, Bairro Alto, Merces, Centro | 56 | 8 |
| CUR_2025_02_18 | Centro | 81 | 2 |
| CUR_2026_02_03 | Centro, Bairro Novo, Pilarzinho, Vista Alegre, Reboucas, Uberaba | 51 | 4 |

`bairro_esperado=False` não significa "não é enchente" — a lista de bairros do `v20i` vem só da
notícia/S2ID, que não é exaustiva (chuva forte atinge mais bairros do que o texto cita). Fica
registrado no CSV bruto (`v20k_siac156_candidatos_brutos.csv`, 251 linhas) pra uso futuro; esta
rodada só levou adiante os 33 que batem com o que já está catalogado, por rigor mínimo.

## 4. Geocodificação real: 33/33, 0 falhas

`scripts/geocode_nominatim.py`, mesmo método e mesma régua de confiança do relatório
`susc_20a_aquisicao_eventos_reais_recife/reports/relatorio_geocodificacao_sedec.md` (SEDEC
Recife): Nominatim ao vivo, 1,1s entre chamadas, `countrycodes=br`, bbox de Curitiba
(-49.45,-25.65,-49.10,-25.30), sem coordenada fabricada.

| Camada | N |
|---|---:|
| strong (nível de rua) | 29 |
| medium (fallback bairro) | 4 |
| failed | 0 |
| **Total** | **33** |

100% geocodificado — taxa muito acima dos 48,8% do Recife (lá o número de casa era mascarado na
fonte; aqui o `Logradouro` do SIAC 156 é rua limpa, sem mascaramento).

## 5. Adjudicação física: HAND/TWI/declividade nos rasters já existentes de Curitiba

`scripts/adjudicate_hand_twi_candidates.py` reaplica a mesma leitura observacional do SUSC-20G
(Juvevê/Valparaíso) — não recalcula raster: usa
`local_runs/susc_20g_hand_twi_dinfinity_generico/curitiba/` (MDT SGB/CPRM 10 m, D-infinity,
`hand_dinf.tif`/`twi_dinf.tif`/`slope_deg_wbt.tif`/`streams_dinf.tif`, já gerado e validado bit a
bit contra Recife no SUSC-20G). Critério: HAND ≤ mediana regional **e** declividade ≤ mediana
regional **e** TWI ≥ mediana regional — mesma leitura qualitativa já usada em Juvevê (que passa)
vs. Valparaíso (que não passa).

Mediana regional de Curitiba: TWI = 6,53, declividade = 3,55°.

| | N |
|---|---:|
| Pontos lidos (33 linhas, 28 coordenadas únicas) | 33 |
| Assinatura de enchente fisicamente plausível | **16** |
| Fora da assinatura clássica (HAND alto e/ou encosta e/ou TWI baixo) | 17 |

Dois exemplos de sinal muito forte (`v20k_siac156_candidatos_adjudicados.csv`):

- **Rua Tibagi, Centro (18/02/2025)** — HAND = 0,0 m, TWI = 15,1 (percentil 98,7%), declividade
  0,25°, **0 m de distância à drenagem extraída** — o evento de 18/02 foi justamente o
  transbordamento do Córrego Bigorrilho no Centro; esse ponto cai literalmente em cima da
  drenagem calculada.
- **Rua sem nome (Daisy Luci Berno), Parolin (03/02/2025)** — HAND = 0,0 m, TWI = 11,96
  (percentil 95,8%), declividade 1,89°.

### Leitura honesta dos 17 que não passam

Não é erro nem falso positivo do SIAC 156: "Alagamentos" no 156 cobre **dois fenômenos físicos
distintos** — transbordamento de curso d'água (assinatura HAND/TWI clássica, fundo de vale) e
alagamento superficial por bueiro/sarjeta entupidos (pode ocorrer em cota mais alta, longe de
drenagem mapeada, mesmo que a rua realmente tenha alagado). O filtro físico aqui separa os dois,
não descarta os 17 como "erro de dado" — eles continuam reais enquanto reclamação, só não têm a
assinatura de inundação fluvial que o modelo físico do projeto está construído pra capturar
(critério metodológico do projeto: o modelo deve refletir relações físicas conhecidas, e não
aceitar qualquer reclamação como prova de inundação-alvo).

## 6. O que isso muda pro N de Curitiba

Ainda **não** é adjudicação individual completa no sentido do SUSC-20A (cada ponto do Recife foi
verificado um a um antes de virar positivo do dataset v12; Valparaíso e Juvevê tiveram leitura
qualitativa individual, não só regra automática). O que esta rodada entrega:

1. Confirmação de que Curitiba **tem**, sim, uma fonte administrativa real equivalente ao SEDEC —
   pergunta que motivou esta rodada, respondida com evidência, não suposição.
2. Um pool de **16 candidatos com assinatura física plausível**, já geocodificados e com HAND/TWI/
   declividade/distância-à-drenagem documentados — pronto pra revisão individual (mesmo padrão
   Valparaíso/Juvevê) antes de qualquer promoção a N.
3. Evidência de pico estatístico independente em 4/4 datas de evento conhecidas — reduz a chance
   de que os candidatos sejam artefato de reclamação crônica constante.

**Decisão de N fica para a próxima rodada**, depois de revisão individual dos 16 (mesmo rigor
usado no Valparaíso), não automática por regra de threshold.

## 7. Petrópolis: sem equivalente público encontrado

Busca real (não contato direto com órgão, conforme regra do projeto): só existe uma "Consulta
Registro de Ocorrência" (`web2.petropolis.rj.gov.br/dfc/ro-digital/registros-ocorrencias`),
renderizada em JS, aparentando ser consulta de status por protocolo, não dataset público baixável.
Nenhum portal de dados abertos municipal com base de atendimento à Defesa Civil foi localizado.
Resultado negativo real, documentado — Petrópolis não tem a mesma infraestrutura de dados abertos
que Curitiba/Recife.

## 8. Limitações

- Cobertura de 4 dos 5 eventos FORTE do v20i. **CUR_2023_10_29** (rank 5) fica fora da janela
  diária do portal (~1,5 ano rolante); precisaria do espelho UFPR/rollup anual 2023, não buscado
  nesta rodada.
- Os 5 eventos MEDIO do v20i não foram testados ainda.
- Só os 33 candidatos com `bairro_esperado=True` foram geocodificados; os outros 218 do CSV bruto
  ficam disponíveis pra uma rodada futura, se fizer sentido ampliar.
- `Logradouro` sem número de casa às vezes (fallback "medium" no geocode) — mesma limitação já
  documentada no SEDEC do Recife.
- MDT usado é SGB/CPRM 10 m (mesma resolução do Recife, ao contrário do FABDEM 30 m usado em
  Petrópolis) — comparável em resolução com Recife, não com Petrópolis.

## 9. Scripts e testes

| Arquivo | Papel |
|---|---|
| `scripts/extract_flood_complaints_siac156.py` | filtra CSV bruto SIAC 156 por data/assunto/bairro |
| `scripts/geocode_nominatim.py` | geocodifica via Nominatim, cache atômico, mesma régua strong/medium/failed do SEDEC |
| `scripts/adjudicate_hand_twi_candidates.py` | lê HAND/TWI/declividade/distância-à-drenagem nos rasters já existentes do SUSC-20G |

`tests/test_susc_20k_siac156_flood_candidates.py`: 16 testes, sintéticos, sem rede — **16
passaram, 0 falharam**.
