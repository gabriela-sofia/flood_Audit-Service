# SUSC-20H — Pipelines reutilizáveis de aquisição: Via A bloqueada, Via B entregue

**Data**: 2026-07-28 | **Escopo**: transformar os dois protocolos de aquisição de candidatos
(hoje scripts ad hoc rodados uma vez) em unidades reutilizáveis e testadas. Nenhum candidato
novo adjudicado, nenhum modelo, nenhuma feature.

---

## 1. Via A — MODIS MCDWD_L3: **bloqueada, não implementada**

A condição para escrever o scanner era o download `.hdf` funcionar direto com o token, sem
clique no navegador. **Não funciona.** Quatro verificações reais, nesta ordem:

### 1.1 Não existe token EOSDIS neste ambiente

Varredura de variáveis de ambiente (`LAADS*`, `EARTHDATA*`, `EOSDIS*`, `NASA*`), de `~/.netrc`,
de arquivo de token no diretório do usuário e de credencial embutida em `.env`/`.json`/`.py` no
repositório e em PROJETO: **nada**. O único acerto textual da busca é um documento não
relacionado do protocolo C. Sem token, o teste "o token passa pelo gate de licença agora?" **não
pôde ser executado** — não é que falhou, é que não há credencial para testar.

### 1.2 A listagem do arquivo é pública e funciona

```
GET https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MCDWD_L3/2022/017.csv
HTTP 200 — CSV real com todos os tiles do dia
```

O tile de Curitiba/Petrópolis está lá, confirmado:
`MCDWD_L3.A2022017.h13v11.061.2025276221819.hdf`, 15.154.074 bytes.
Ou seja: **descobrir** o arquivo não exige credencial nenhuma.

### 1.3 O download do `.hdf` sem token cai no OAuth — mesmo erro do v5

```
GET .../2022/017/MCDWD_L3.A2022017.h13v11.061.2025276221819.hdf
HTTP=200  bytes=10783
url_effective = https://urs.earthdata.nasa.gov/oauth/authorize?response_type=code
                &client_id=A6th7HB-3EBoO7iOCiCLlA&redirect_uri=...%2Foauth%2Fcallback&state=...
```

10.783 bytes de HTML de login, não HDF binário — exatamente o mesmo sintoma registrado em
`RELATORIO_v5_NASA_MODIS_TOKEN_TENTATIVA.md`. O gate de licença/EULA continua no caminho.

### 1.4 Não há `.hdf` em disco para servir de fixture

A instrução previa usar "os HDFs já em mãos" como fixture do teste. Varredura de `REV-P/` e
`PROJETO/`: **zero arquivos `.hdf`**. O `RELATORIO_v7` explica por quê — os arquivos daquela
rodada foram **enviados manualmente** por você depois de autorizar o OAuth no navegador, nunca
baixados pelo pipeline, e não ficaram no repositório. `pyhdf` também não está instalado.

### 1.5 Conclusão

`scan_mcdwd_dates.py` **não foi escrito**. Escrevê-lo agora significaria entregar um scanner que
não pode baixar nada, com um leitor de `Flood_1Day_250m` que não pode ser testado contra nenhum
arquivo real — código não validado, contra a regra do projeto. As janelas novas (Curitiba verão
2023-2026) **não foram varridas**, e o número honesto de candidatos brutos desta rodada é
**nenhuma varredura executada**, não "0 candidatos".

**O que destrava, em ordem**: (1) token EOSDIS disponível como variável de ambiente; (2) um
download manual único pelo navegador logado, que registra o aceite de licença na conta;
(3) reteste do download programático. Só depois disso o scanner vale ser escrito — aí com o
`.hdf` real servindo de fixture, como planejado.

**Correção de método já registrada e que deve entrar no scanner quando ele existir**: a fórmula
de tile é a da grade geográfica (`h = floor((lon+180)/10)`, `v = floor((90−lat)/10)`), **não** a
sinusoidal. Pela fórmula certa, Curitiba e Petrópolis caem no **mesmo** tile `h13v11`
(lon[−50,−40], lat[−30,−20]) — o `h14v11` do v5 nunca existiu.

---

## 2. Via B — Sentinel-2 → candidatos: **entregue e testada**

`scripts/detect_water_candidates.py`. Generaliza o processamento que achou o cluster de
Valparaíso e incorpora o refinamento da revisão de literatura (seção 3).

### 2.1 Cadeia de decisão

1. **Filtro físico obrigatório** (o que separou água de nuvem no v14): no dia do evento
   `B08 < 0,15` **e** `B11 < 0,15`, e **não** era assim antes. Pixel reprovado aqui não vira
   candidato por mais que os índices concordem.
2. **Três índices**, calculados antes e depois:
   - NDWI = (B03 − B08)/(B03 + B08) — McFeeters 1996
   - MNDWI = (B03 − B11)/(B03 + B11) — Xu 2006
   - AWEI_nsh = 4·(B03 − B11) − (0,25·B08 + 2,75·B12) — **Feyisa et al. 2014**, *Remote Sensing
     of Environment* 140:23-35, doi:10.1016/j.rse.2013.08.029
3. **Consenso 2 de 3**: ao menos dois índices têm de subir mais que o próprio limiar de mudança.
4. **Cluster**: componentes conexos 8-vizinhos, tamanho mínimo configurável (default 20 px).

Saída: CSV de candidatos com centroide (linha/coluna, easting/northing e lon/lat quando há CRS),
contagem de pixels, média de reflectância antes/depois e `adjudicated=false`. **O script não
adjudica** — listar candidato e adjudicar pelo critério SUSC-20A continuam separados.

### 2.2 Achado real da implementação: AWEI exige B12, que o protocolo não baixava

AWEI_nsh usa SWIR2. O protocolo Via B da linhagem manda baixar B03, B08 e B11 — **sem B12 não há
como calcular AWEI**. E aí o "2 de 3" degeneraria no par NDWI+MNDWI, que é exatamente o que a
literatura aponta como insuficiente sozinho.

Decisão: o script **falha fechado** se B12 faltar, com mensagem explícita. Existe
`--allow-two-index-fallback` para rodar mesmo assim, e nesse caso todo cluster sai marcado
`consensus_basis = NDWI_MNDWI_ONLY_WEAKER` — degradação sempre visível, nunca silenciosa.

**Consequência prática**: o passo 1 do protocolo Via B passa a exigir **B03, B08, B11 e B12**
como Raw, TIFF 32-bit. Atualizado na seção 2 da linhagem.

### 2.3 Testes

`tests/test_susc_20h_sentinel2_water_candidates.py` — **18 passaram, 0 falharam** (5,0 s).

Cobrem: as três fórmulas contra o valor calculado à mão; divisão por zero virando NaN e não
exceção; o filtro físico nos quatro casos (escuro depois e não antes / já era escuro / continua
claro / escuro só em NIR); a regra de consenso parametrizada em 3-de-3, 2-de-3, 1-de-3 e 0-de-3;
consenso sem índice nenhum falhando alto; água plausível virando candidato; **nuvem brilhante
sendo rejeitada pelo filtro físico** (o falso-positivo real do v14); pixel escuro com um índice
só não passando; cluster abaixo do tamanho mínimo descartado; dois blocos separados virando dois
clusters ordenados por tamanho; **fail-closed sem B12** e o fallback marcando o cluster como mais
fraco; banda obrigatória ausente falhando alto; resolução de banda por nome de arquivo; e grades
diferentes entre bandas falhando alto em vez de reamostrar por conta própria.

Não há Sentinel-2 novo em disco nesta etapa — os testes usam arrays sintéticos mínimos, cada um
isolando uma regra. O script ainda **não foi rodado sobre cena real**; isso depende de uma
aquisição nova (agora com B12).
