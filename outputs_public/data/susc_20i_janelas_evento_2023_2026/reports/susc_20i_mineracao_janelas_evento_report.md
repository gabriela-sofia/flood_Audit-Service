# SUSC-20I — Mineração de janelas de evento 2023-2026 (Petrópolis e Curitiba)

**Data**: 2026-07-29 | **Escopo**: minerar fonte e ranquear janelas. Nenhuma imagem baixada,
nenhum candidato adjudicado, nenhum ponto criado. Curitiba segue N=1, Petrópolis N=0.

---

## 1. Fontes consultadas e o que cada uma deu

| Fonte | Rota | Resultado |
|---|---|---|
| S2ID — Danos Informados | POST público, sem login, ano a ano, 65 tipologias COBRADE | **funcionou**; 6 registros de Petrópolis e 29 de Curitiba em 2023-2026 |
| Notícia real (busca web) | corroboração por data, bairro e curso d'água | **funcionou** para 7 das janelas |
| Diário Oficial de Petrópolis | decreto de situação de emergência | **confirmou** o evento de 04-05/04/2025, com pluviometria por região |
| S2ID — Série Histórica | — | **não usada**: já documentado que o backend só tem dados até 2016 |

### 1.1 Bug real encontrado no filtro herdado

O script anterior (`s2id_danos_uf.py`) isola o município por substring do **nome**. O CSV vem em
latin-1 com "Petrópolis" acentuado, e `"petropolis" in "petrópolis"` é **falso** — a consulta
devolve zero silenciosamente. Foi exatamente o que aconteceu na primeira execução desta rodada:
"Petrópolis: 0" em todos os anos, inclusive em 2022, que sabidamente tem 3 registros.

Corrigido em `scripts/fetch_s2id_municipio.py`, que filtra pelo **código IBGE** presente no campo
`Protocolo` (`RJ-F-3303906-13120-20250405`). Reexecutado, reproduz os mesmos 6 registros de
Petrópolis. Sem essa correção, esta rodada teria concluído "Petrópolis esgotada" por engano.

### 1.2 Achado estrutural: Curitiba classifica o fenômeno, Petrópolis não

Petrópolis registra tudo por **gatilho meteorológico** (13214 em 2022, 13120 em 2024-2025) ou por
movimento de massa (11311, 11331) — nunca pela classe hidrológica. Já Curitiba usa **12300 –
Alagamentos** em 10 dos 29 registros do período. É a primeira vez nesta linhagem que o COBRADE
separa o fenômeno sozinho, e é por isso que o ranking de Curitiba é mais fundo que o de Petrópolis.

## 2. Petrópolis — 2 janelas reais

Escala de evidência: **forte** = registro oficial + notícia datada; **médio** = só um dos dois;
**fraco** = menção vaga.

| # | Data | Bairros / locais | Evidência hidrográfica | Fonte | Nível |
|---|---|---|---|---|---|
| 1 | **2025-04-05** | Centro Histórico; Av. Barão do Rio Branco; Alto da Serra; Independência; São Sebastião; Chácara da Flora | **Rio Quitandinha transbordou** e alagou o Centro Histórico | S2ID `RJ-F-3303906-13120-20250405` (Reconhecido; 30 desabrigados, 128 desalojados, 1.802 afetados) + decreto de emergência no D.O. municipal + Diário do Grande ABC / CNN / Sou Petrópolis | **FORTE** |
| 2 | **2024-03-22** | Quitandinha; Centro; Valparaíso; Duarte da Silveira; Itaipava; Independência; São Sebastião; Araras; Itamarati; Castelânea; Retiro; Roseiral | **Rios Quitandinha e Piabanha transbordaram**; ruas alagadas | S2ID `RJ-F-3303906-13120-20240322` (Reconhecido; 4 mortos, 188 desabrigados, 445 desalojados, 175.965 afetados) + Agência Brasil | **FORTE** |

Contexto quantitativo: 347 mm em 24 h em 05/04/2025 (São Sebastião >300 mm, Independência 296,81,
Alto da Serra 295,81, Chácara da Flora 248,2 — pluviômetros Cemaden); 196 mm em 12 h em 22/03/2024.

**Excluídos por fenômeno** (COBRADE de movimento de massa, fora do escopo de enchente):
22/01/2024 e 10/03/2025 (11311 – quedas/rolamentos de blocos), 06/02/2024 (11331 – corridas de
massa). Também excluído 20/08/2025 (14131 – incêndio florestal).

**Ressalva sobre a #2**: o evento de 22/03/2024 é **misto**. A notícia reporta transbordamento de
rio *e* deslizamentos nos mesmos bairros. Serve como janela, mas qualquer candidato dela nasce com
risco alto de ser encosta — exatamente o caso que o filtro topográfico da seção 2.1 existe para
pegar.

**Anos sem nada**: 2023 e 2026 têm **zero** registros de Petrópolis no S2ID, e nenhuma matéria
datada de enchente em Petrópolis para 2023 foi localizada. Esgotado para o período, não "não
procurado".

## 3. Curitiba — 10 janelas ranqueadas

| # | Data | Bairros / locais | Evidência hidrográfica | Fonte | Nível |
|---|---|---|---|---|---|
| 1 | **2025-02-03** | Boqueirão; Xaxim; Pinheirinho; Parolin; Portão; Vila Izabel; Seminário | **Ribeirão dos Padilhas transbordou no Xaxim** (resgate de 7 adultos e 35 crianças em creche); Arroio Boa Vista; Córrego Henry Ford | S2ID 12300 (4.040 afetados) + Banda B, verificado por leitura direta | **FORTE** |
| 2 | **2025-02-18** | Centro (Visconde de Naçar × Saldanha Marinho); escolas alagadas | **Córrego Bigorrilho subiu 2,29 m** e transbordou | S2ID 12300 (2.200 afetados) + massa.com.br / Banda B | **FORTE** |
| 3 | **2026-02-03** | Centro (Av. Vicente Machado); Bairro Novo; Pilarzinho; Vista Alegre; Rebouças; Uberaba; Linha Verde | ~40 pontos de alagamento em ao menos 10 bairros | S2ID 12300 (1.500 afetados) + Banda B / CBN Curitiba | **FORTE** |
| 4 | **2025-01-28** | Centro Cívico; Visconde de Naçar; Bairro Alto (Linha Verde); Mercês (Visconde do Rio Branco) | **Rio Ivo, canalizado, transbordou** | S2ID 12300 (12 desalojados, 1.988 afetados) + BandNews FM / Tribuna do PR | **FORTE** |
| 5 | **2023-10-29** | Caximba (extremo sul); monitoramento em Santa Felicidade, Pinheirinho, Butiatuvinha, Vila Izabel | risco de alagamento no Caximba; 94 mm em 24 h | S2ID 12300 (152 desalojados) + Banda B | **FORTE** ⚠ |
| 6 | 2025-02-27 | — | — | S2ID 12300 (2.000 afetados) | MÉDIO |
| 7 | 2026-01-14 | — | — | S2ID 12300 (264 afetados) | MÉDIO |
| 8 | 2025-12-14 | — | — | S2ID 12300 (240 afetados) | MÉDIO |
| 9 | 2023-10-12 | — | — | S2ID **13214** (19 desabrigados, 1.200 afetados) | MÉDIO |
| 10 | 2025-03-11 | — | — | S2ID 12300 (4 desalojados, 56 afetados) | MÉDIO |

⚠ **Divergência registrada na #5**: o S2ID informa 152 desalojados em 29/10/2023, e a matéria da
época afirma que a Defesa Civil não registrou desalojados. Não resolvi essa contradição — fica
anotada para ser checada antes de qualquer uso, não silenciada.

**Não ranqueados**: 22/11/2023 (12300 sem nenhum dano informado e sem notícia — FRACO) e 17
registros de incêndio, vendaval, granizo, deslizamento e produto perigoso, fora do escopo.

**Sobre as #6-#10**: o COBRADE 12300 é registro oficial de alagamento, mas sem notícia datada não
há bairro nem curso d'água para orientar o recorte da imagem. São janelas reais e utilizáveis —
só exigem uma busca de notícia dedicada antes de virarem alvo de aquisição.

### 3.1 Alto da Glória, Cristo Rei e Hugo Lange — verificados, sem data

Os três bairros foram checados especificamente, como pedido. Resultado real: **nenhuma data
específica de 2023-2026** foi encontrada, nem em registro S2ID nem em matéria datada. O que existe
é evidência de propensão crônica, sem evento datado: obras antienchente da prefeitura no Alto da
XV e Cristo Rei, alagamento recorrente há pelo menos oito anos em Hugo Lange, e galerias de
contenção do **Rio Juvevê** na Rua Camões (Hugo Lange) — a mesma drenagem do único ponto
adjudicado de Curitiba (Juvevê, 17/01/2022). Um snippet de busca mencionou alagamento na Rua
Schiller (Cristo Rei) em 18/01/2023, mas não consegui confirmar na fonte e **não** o incluí na
tabela.

Conclusão honesta: esses bairros são candidatos plausíveis de área, não janelas de evento. Só
entram quando houver data.

## 4. Artefatos

| Caminho | Conteúdo |
|---|---|
| `scripts/fetch_s2id_municipio.py` | consulta S2ID por UF + faixa de anos, filtrando por código IBGE (corrige o bug de acento) |
| `registries/v20i_event_windows_2023_2026.csv` | 35 janelas (as 12 ranqueadas + as excluídas com o motivo), com COBRADE, protocolo, danos, localidades, fonte, nível de evidência e `adjudicated=false` |

Reproduzir:

```bash
python outputs_public/data/susc_20i_janelas_evento_2023_2026/scripts/fetch_s2id_municipio.py --uf PR --ibge 4106902 --de 2023 --ate 2026 --outdir local_runs/susc_20i/s2id
```

## 5. O que esta rodada não fez

Não baixou imagem (Via A segue exigindo token EOSDIS ausente; Via B exige download manual
autenticado do Copernicus). Não adjudicou nada — a coluna `adjudicated` é `false` em todas as 35
linhas. Não criou ponto, não tocou em `region_registry.py`.
