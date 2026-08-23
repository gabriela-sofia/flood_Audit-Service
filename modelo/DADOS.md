# Dados

Ficha da base, no formato de *datasheet for datasets* (Gebru et al., 2021). Descreve por
que a base existe, do que ela é feita, como foi coletada e rotulada, e o que foi
deliberadamente deixado de fora.

---

## 1. Motivação

A base existe para responder uma pergunta física: **a predisposição do terreno a
acumular água explica onde a enchente acontece?** Ela precisa, portanto, de dois lados —
onde inundou e onde comprovadamente não inundou — na mesma cadeia de derivação de
terreno, para que a comparação meça território e não diferença de produto.

A dificuldade que ela resolve é específica do Brasil: o registro oficial nasce de quem
consegue reportar e ser atendido. A documentação disponível reflete a assistência
prestada, não a extensão real da água. **Ausência de registro não é ausência de evento**,
e uma base que tratasse ausência como negativo aprenderia a geografia do atendimento
público, não a do relevo.

## 2. Composição

**65.070 pontos elegíveis ao ajuste fluvial**, reduzidos a partir de seis fontes, todas
na mesma cadeia de derivação D-infinity e na mesma resolução.

### Positivos

| Fonte | Região | Pontos |
|---|---|---:|
| SEDEC | Recife | 154 |
| SIAC 156 | Curitiba | 1.045 |
| Diário Oficial (decretos de emergência) | Recife | — |
| Bases internacionais | multi | — |

### Negativos, em três níveis declarados por linha

| Nível | Fonte | Volume | O que significa |
|---|---|---:|---|
| **Observado** | Copernicus EMS | 25.249 pontos em 119 AOIs | Área que um analista examinou e onde não detectou inundação. A ativação EMSR720 no Rio Grande do Sul entra com 216,55 km² na proporção 5,94:1 |
| **Exclusão qualificada** | Environment Agency / UK | 7.476 pontos (3.738 / 3.738) em 201 eventos independentes | Excluída pelos quatro critérios N1–N4 sobre os *Recorded Flood Outlines* |
| **Exclusão qualificada** | SIAC 156 / Curitiba | 114 pontos | Reclassificados pelo mesmo padrão de critérios; antes constavam como ausência |
| **Ausência de registro** | Recife e Petrópolis | — | Nem observação nem exclusão. **Não entra como negativo** |

Dos 442 pontos avaliados para exclusão qualificada em Curitiba, 114 passam nos quatro
critérios. Os 328 restantes seguem como ausência.

### Variáveis

| Variável | O que mede | Sinal esperado |
|---|---|---|
| `hand_m` | Altura sobre a drenagem mais próxima — a lâmina que o rio precisa ganhar para alcançar o ponto | negativo |
| `twi_dinf` | Índice topográfico de umidade por D-infinity — quanta encosta drena para ali | positivo |
| `slope_deg` | Declividade | — |
| `elevation_m` | Elevação | — |
| Chuva antecedente | Forçamento, em produto único | — |

O sinal é fixado **antes** de rodar. Coeficiente com sinal invertido é falha de
mecanismo, não resultado.

## 3. Coleta

| Etapa | Como | Onde no código |
|---|---|---|
| Evento oficial em Recife | Geocodificação de registros da SEDEC, decretos do Diário Oficial, estações da ANA | `outputs_public/data/susc_20a*/` |
| Evento oficial em Curitiba | Extração de reclamações do SIAC 156, deduplicação, geocodificação por Nominatim, adjudicação física por HAND/TWI | `outputs_public/data/susc_20k*/scripts/` |
| Fontes globais | Sen1Floods11, Urban Flood Observations, Copernicus EMS e Global Flood Database | `scripts/externo/aquisicao/` |
| Negativo inglês | *Recorded Flood Outlines* e *Flood Map for Planning* da Environment Agency, mais ESA WorldCover para o critério N4 | `scripts/externo/uk/` |
| Redução a ponto | Contrato que impede contar o mesmo evento duas vezes ao converter polígono em ponto | `scripts/externo/pontos/n1f_reduzir_a_pontos.py` |
| Terreno | Rederivação de HAND e TWI por D-infinity a partir do MDT, em toda a base | `scripts/terreno/` |
| Chuva | Open-Meteo / ERA5-Land, produto único, cobertura 99,99% | `scripts/treino/aud_chuva02_escala_do_contraste.py` |
| Consolidação | Esquema-alvo, redução por fonte, admissão | `scripts/treino/ds03`–`ds05` |

## 4. Rotulagem

O rótulo positivo é **evento oficialmente registrado e geocodificado**, não interpretação
de imagem. A unidade independente é o evento ou a AOI, nunca o ponto: pontos do mesmo
evento não são observações independentes, e tratá-los como tal inflava a significância
por pseudorreplicação.

Nenhum rótulo foi criado por modelo. O DINOv2 nunca produziu rótulo, alvo ou referência
de campo — ver `../docs/METODO.md` §5.

## 5. O que foi deixado de fora, e por quê

| Excluído | Motivo |
|---|---|
| Variável derivada do rótulo | Circularidade: acurácia balanceada de 0,8855 com elas, 0,4834 sem. O número alto era o modelo lendo a resposta |
| CHIRPS como fonte de chuva | Retirado pela auditoria de confundimento; a base passou a produto único ERA5-Land |
| UFO no ajuste fluvial | Declara mecanismo misto não separável. Tem a mesma cadeia de terreno, mas fica fora do ajuste |
| Ausência de registro como negativo | Regra 2 das regras de decisão — ver `../docs/EVIDENCIA.md` §1 |
| Copernicus EMS na AOI inglesa | A API pública só expõe ativações de 2023 em diante, e nenhuma das 7 do Reino Unido cai na AOI do piloto |
| Petrópolis do ajuste | Zero linhas: enchente e movimento de massa não estão separados nas fontes disponíveis |

## 6. Distribuição e manutenção

A base consolidada tem 189 MB e **não é versionada** — o repositório carrega os scripts
que a regeneram e os artefatos de resultado, não o dado bruto. As fontes primárias são
todas públicas: SEDEC, SIAC 156, ANA, Copernicus EMS, Environment Agency, Sen1Floods11,
Global Flood Database, ESA WorldCover, Open-Meteo/ERA5-Land.

Os dados curados por etapa da linha causal estão em
`outputs_public/data/susc_20*/registries/` e são versionados.

Solicitações formais a DRM-RJ, Defesa Civil de Petrópolis, SGB/CPRM e COMPDEC seguem sem
resposta no período do projeto — é o que mantém Petrópolis sem inventário local.

## 7. Limitações da base

- Nenhuma das três regiões brasileiras tem negativo formal aceito.
- A chuva está recuperada para 99,99% da base, mas em células de ~11 km — escala grossa
  demais para discriminar dentro de um mesmo evento.
- 21 de 22 testes de invariante da base passam. No ambiente fixado do projeto a
  reprodução bate exata; entre ambientes diferentes ainda depende de versão de
  biblioteca.
- O `hand_m` das três regiões brasileiras **não era a mesma variável** antes da
  harmonização de 12/08. Qualquer análise anterior a essa data não é comparável entre
  regiões.
