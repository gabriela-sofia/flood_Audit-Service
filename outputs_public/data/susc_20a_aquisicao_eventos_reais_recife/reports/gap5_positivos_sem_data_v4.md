# GAP 5 — 9 positivos Recife sem data de evento

**Status**: EXPLORATORIO_DIAGNOSTICO_NAO_CANONICO | **Data**: 2026-07-22

## Identificação exata dos 9 registros

Via `qa_record_id` (cruzamento com `local_runs/recife_geocoding_completion/
recife_inundacao_geocoded_final.geojson`), os 9 positivos sem `event_date` foram todos
rastreados até linhas específicas do CSV bruto de origem:

`data/raw/recife/seced_defesa_civil/registro_de_atendimentos_da_defesa_civil__atendimentos_2014__
b81bf4cc-ae19-4a43-94a7-f11b3e0b662b.csv` (todos os 9 são do ano-arquivo 2014), linhas 644,
6195, 6213, 6454, 19757, 31831, 37401, 42090, 42310.

## Campo de data secundário encontrado

O CSV bruto tem 16 colunas, incluindo **`Data_da_Acao`** (data da ação de resposta da defesa
civil), distinta de `Data` (data de ocorrência, em branco nos 9 casos). **Os 9/9 registros têm
`Data_da_Acao` preenchida e internamente consistente com a coluna `Mês`** (ex.: linha 644,
Mês=Abril → Data_da_Acao=2014-04-15; linha 19757, Mês=Janeiro → Data_da_Acao=2014-01-23; linha
37401, Mês=Maio → Data_da_Acao=2014-05-12 — todas batem).

| Linha | Bairro | Ocorrência | Solicitação | `Data_da_Acao` |
|---:|---|---|---|---|
| 644 | Ibura de Baixo | Imóveis Alagados | Monitoramento Alagado | 2014-04-15 |
| 6195 | Campina do Barreto | Imóveis Alagados | Monitoramento Alagado | 2014-04-30 |
| 6213 | Campina do Barreto | Imóveis Alagados | Monitoramento Alagado | 2014-04-30 |
| 6454 | Campina do Barreto | Imóveis Alagados | Monitoramento Alagado | 2014-04-30 |
| 19757 | Imbiribeira | (vazio) | Monitoramento Alagado | 2014-01-23 |
| 31831 | Casa Forte | Imóveis Alagados | Monitoramento Alagado | 2014-06-05 |
| 37401 | Várzea | Alagamentos | Monitoramento Alagado | 2014-05-12 |
| 42090 | Linha do Tiro | Imóveis Alagados | Monitoramento Setor Alagado | 2014-05-29 |
| 42310 | Linha do Tiro | Imóveis Alagados | Monitoramento Setor Alagado | 2014-05-29 |

## Interpretação honesta — por que não foi simplesmente adotado

`Data_da_Acao` é semanticamente **a data da ação de acompanhamento/vistoria da defesa civil**,
não necessariamente a data exata do evento de alagamento em si — pode haver um atraso real
entre o alagamento e a visita de mapeamento/monitoramento (estes registros são majoritariamente
"Monitoramento Alagado" / "Mapeamento de Área de Risco", ações de acompanhamento contínuo, não
atendimentos de emergência no dia do evento). Promover esse campo silenciosamente para
`event_date` fabricaria uma precisão que a fonte não garante.

## Resultado

**9/9 têm um campo de data secundário real e utilizável, nenhum permanece "sem qualquer
data no registro bruto"** — mas o campo tem semântica diferente (data de ação/acompanhamento,
não data de ocorrência). Não promovido automaticamente a `event_date` nesta rodada (mudança de
semântica de dado requer decisão explícita da revisora, fora do escopo autorizado de "gap 1-3
apenas" para mudança de feature-set). **Recomendação para trabalho futuro**: usar
`Data_da_Acao` como `event_date_proxy` com flag explícita de incerteza (`date_type=
follow_up_action_not_occurrence`), o que resolveria os 9 casos com uma ressalva documentada, em
vez de excluí-los das features de chuva como hoje.

## Arquivos
- Consulta direta em `data/raw/recife/seced_defesa_civil/registro_de_atendimentos_da_defesa_civil__atendimentos_2014__b81bf4cc-ae19-4a43-94a7-f11b3e0b662b.csv`
- `local_runs/recife_geocoding_completion/recife_inundacao_geocoded_final.geojson` (linkagem
  `qa_record_id` → `source_row_index`)
