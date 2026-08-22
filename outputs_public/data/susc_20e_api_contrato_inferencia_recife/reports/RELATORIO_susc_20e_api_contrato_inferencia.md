# SUSC-20E -- API local do contrato de inferência (Fase 5, MVP)

## Objetivo

Implementa literalmente o rascunho v0 do contrato de inferência (`txtpragab.docx`,
extraído em `revp_fase2_decisoes_design_contrato.md`) como uma API HTTP local
(FastAPI/uvicorn), só depois de as Fases 1-3 terem números reais -- ordem já definida em
`PLANO_ACAO_produto_v1.md` seção 6.

## O que foi implementado

- `contract_schema.py`: schema de entrada/saída idêntico ao rascunho (`request_id`,
  `region.geometry`/`crs`, `period`, `requested_layers` → `status`, `region_maturity`,
  `score.value`/`confidence_interval`/`model_version`, `features_used`, `evidence`,
  `limitations`, `data_version`, `generated_at`).
- `region_registry.py`: estado real das 3 regiões (Recife=`available` com modelo v12;
  Curitiba=`limited_evidence` sem modelo, com nota apontando os Leads A/B/C reais desta
  sessão; Petrópolis=`insufficient`, mistura enchente/deslizamento não resolvida).
- `gates.py`: avalia em ordem -- geometria válida (shapely) → CRS conhecido → região
  suportada → modelo válido pra região → features físicas disponíveis. **Atualizado**:
  agora tenta primeiro os 269 pontos conhecidos do v12 e, se nenhum casar, cai para o
  pipeline sob demanda do SUSC-20F (terreno real amostrado de raster + chuva ao vivo
  Open-Meteo) quando o centróide da geometria cai dentro da cobertura real do DTM --
  ver `RELATORIO_susc_20f_pipeline_sob_demanda.md`.
- `engine_bridge.py`: reusa `susc_20d_score_engine.py` (Firth + bootstrap preditivo já
  validados), precomputando o ajuste + os 1000 draws de bootstrap **uma vez no startup**
  da API, não a cada request (otimização já antecipada na decisão da Fase 2).
- `app.py`: rota `POST /score` (contrato completo) + `GET /regions` (status por região).

## Validação end-to-end (rodado nesta sessão, servidor local, porta 8123)

| Caso de teste | status | region_maturity | score |
|---|---|---|---|
| Polígono contendo ponto real positivo conhecido (recife_pos_0001) | `ok` | `available` | 0.7737, CI [0.6692, 0.8718] -- idêntico ao audit do SUSC-20D pro mesmo ponto |
| Polígono em Curitiba | `insufficient_data` | `limited_evidence` | null (gate: sem modelo válido pra região) |
| Polígono em Petrópolis | `insufficient_data` | `insufficient` | null |
| Polígono fora de qualquer região conhecida (São Paulo) | `region_not_supported` | `insufficient` | null |
| Polígono em Recife sem nenhum ponto conhecido dentro | `insufficient_data` | `available` | null (gate: nenhum ponto com features físicas conhecidas na geometria) |

Todos os 5 casos se comportaram exatamente como o contrato exige: nunca inventa score
quando falta gate, nunca confunde `region_not_supported` (fora de cobertura) com
`insufficient_data` (dentro da cobertura, mas sem dado suficiente nesse ponto exato).

## Atualização -- pipeline sob demanda integrado (SUSC-20F)

A limitação abaixo (histórica, mantida para registro) **foi fechada nesta sessão**.
Novos testes reais:

| Caso de teste | status | score | fonte |
|---|---|---|---|
| Coordenada nova em Recife, **fora** dos 269 pontos conhecidos, dentro da cobertura DTM | `ok` | 0,3263, CI [0,0797; 0,716] | DTM real + Open-Meteo ao vivo (`observational_points_used=0`) |
| Coordenada dentro da região Recife mas **fora** da cobertura real do DTM | `insufficient_data` | null | gate: `fora_da_cobertura_real_do_dtm_e_sem_ponto_conhecido_na_geometria` |

Ver `RELATORIO_susc_20f_pipeline_sob_demanda.md` para a validação completa (terreno:
HAND/TWI match exato, elevação/declividade com ~2,7m/5,6° de diferença histórica
documentada; chuva: match exato contra 30 pontos reais).

## Limitação histórica (já resolvida, mantida como registro)

~~Os gates de DEM/declividade/HAND/TWI/chuva só são avaliados hoje checando se um dos
269 pontos conhecidos do v12 cai dentro da geometria pedida~~ -- resolvido pelo
SUSC-20F: qualquer coordenada dentro da cobertura real do DTM (~21km x 28km, cobre
essencialmente toda a região de Recife) agora recebe score real sob demanda.

## Como rodar

```
cd outputs_public/data/susc_20e_api_contrato_inferencia_recife/scripts
python -m uvicorn app:app --host 127.0.0.1 --port 8123
```

Variável opcional `REVP_SUSC20D_SENTINEL_DIR` (caminho privado local dos patches
Sentinel) habilita `evidence.dino_embedding_available`/`dino_patch_id` quando
configurada; sem ela, a API funciona normalmente e só omite essa evidência auxiliar
(nunca bloqueia o score por causa disso -- DINO é sempre opcional).

## O que ainda falta pro produto (não é este documento que resolve)

- Interface web (mapa + score + evidências) -- não iniciado.
- LLM de explicação sobre o payload estruturado -- não iniciado (a Fase 5 do plano só
  cobre a API/contrato; a camada de LLM é explicitamente posterior).
- Cobertura fora de Recife (Curitiba/Petrópolis) -- sem modelo estatístico próprio ainda.
- Reconciliar o raster exato de elevação/declividade (ver caveat documentado no SUSC-20F)
  se algum dia o DTM PE3D bruto voltar a ficar acessível sem captcha manual.
- Deploy real / infraestrutura compartilhada -- isto roda só localmente.
