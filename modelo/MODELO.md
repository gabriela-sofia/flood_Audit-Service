# Modelo

Cartão do modelo, no formato de *model card* (Mitchell et al., 2019). Descreve o que foi
treinado, para que serve, onde funciona e — principalmente — **onde não vale**.

O método está em [`../docs/METODO.md`](../docs/METODO.md); a procedência de cada número,
em [`../docs/EVIDENCIA.md`](../docs/EVIDENCIA.md); a base, em [`DADOS.md`](DADOS.md).

---

## 1. Identificação

| | |
|---|---|
| **Nome** | Suscetibilidade urbana a enchentes com base causal físico-hidrológica |
| **Versão** | 22/08/2026 |
| **Tipo** | Regressão logística penalizada de Firth. GBM monotônico e GAM entram só como diagnóstico, nunca em produção |
| **Uso pretendido** | Estimar a predisposição de uma **área** a acumular água sob um dado forçamento de chuva, com incerteza declarada e maturidade por região |
| **Unidade de resposta** | Área, nunca pixel — a mesma unidade em que o modelo é validado |
| **Fora de uso** | Previsão de evento, alerta operacional, decisão de evacuação, seguro, ou qualquer uso em que "sem escore" seja lido como "seguro" |

## 2. Os três modelos servidos

Estão em [`execucoes/svc-01-modelos/`](execucoes/svc-01-modelos/), serializados com as
réplicas de bootstrap que produzem o intervalo de confiança.

| Modelo | Ajustado em | Variáveis | AUC agrupada | Veredito |
|---|---|---:|---:|---|
| `recife_pluvial` | Recife, 269 pontos | 6 | 0,6409 | **`FORA_DOS_CRITERIOS`** |
| `fluvial_planicie` | 56.654 pontos, 509 grupos, **sem Recife e sem Curitiba** | 4 | 0,7336 | `COERENTE_COM_CRITERIOS` |
| `fluvial_serra` | 5.162 pontos, 24 grupos, estrangeiros | 1 | 0,7916 | `COERENTE_COM_CRITERIOS` |

**A região-alvo não entra no treino do modelo que a serve.** O propósito é prever onde
não há inventário local; validar contra o rótulo da própria região seria a pergunta
errada. Um teste guarda esse invariante.

**Recife não atinge o próprio critério.** Com a fonte de chuva corrigida, o LOO-AUC é
0,6409 — abaixo da faixa 0,70–0,88 fixada em 09/08, antes de rodar — e o IC de `hand_m`
cruza zero. O serviço responde mesmo assim, com a falha escrita na resposta e a
maturidade rebaixada. Esconder o escore esconderia também o quanto ele é fraco.

## 3. Desempenho

### Ajuste por classe de relevo (E3)

| Ajuste | AUC | Coeficientes (IC95) |
|---|---:|---|
| Serra | **0,7916** | `hand_m` −1,44 [−3,11; −0,83] |
| Planície | **0,7245** | `hand_m` −2,10 [−2,78; −1,56]; `twi_dinf` +0,40 [+0,33; +0,45] |
| Planície aplicada à serra | **0,7957** | — |

O modelo de planície aplicado à serra supera o que a serra alcança sozinha: a relação é
a mesma nos dois terrenos, e o que separa é quão bem cada um está estimado. O estrato
íngreme tem 19 eventos positivos e comporta **uma** variável.

### Holdout temporal (E4)

| | |
|---|---|
| Janela | 201 eventos em 110 datas, 2000–2025 |
| Cortes | 8, todos na faixa 0,70–0,88 fixada antes |
| AUC médio | **0,7992** |
| Incerteza | IC95 por bootstrap de grupos em cada corte |

Refuta que o colapso temporal seja propriedade do método. **Não** prova estabilidade em
serra tropical.

### Recife, modelo próprio

Firth v12, n=278 (154 positivos da SEDEC / 124 negativos). LOO-AUC 0,68; 5-fold repetido
0,67 ± 0,01. Os seis sinais de coeficiente preservam a coerência física esperada.

## 4. Onde o desempenho varia — e onde o modelo recusa

| Região | Maturidade | O que a resposta carrega |
|---|---|---|
| **Recife** | `mvp_local` | Modelo próprio com o critério de leitura não atingido, escrito na resposta |
| **Curitiba** | `transferencia_caracterizada` | Extrapolação de `elevation_m` declarada: 5,05 desvios do domínio, nenhuma célula na faixa vista |
| **Petrópolis** | `transferencia_sem_referencia_local` | Escore por semelhança de terreno; 91,3% do território cabe na faixa de HAND do modelo de serra |
| Qualquer outra | `region_not_supported` | Geometria que não cai em região com modelo |

**Nenhuma região está em `validado`.**

O portão de domínio (G5) recusa em vez de extrapolar acima do limite. Na grade, célula
fora do domínio fica **vazia** em vez de receber escore baixo: recusar não é o mesmo que
dizer que ali é seguro.

## 5. Limitações

- Nenhuma das três regiões brasileiras tem negativo formal aceito. O gate
  `C4_BLOCKED_NO_FORMAL_NEGATIVES` segue aberto, e a ausência é declarada, não
  contornada por proxy.
- A chuva não discrimina na escala do modelo: é medida em células de ~11 km enquanto o
  modelo compara pontos dentro do mesmo evento. Entra como **cenário**, não como camada.
- O contrato roda como função pura. Falta o transporte HTTP.
- O modelo próprio de Curitiba não generaliza: AUC 0,65 embaralhado, **0,52** em holdout
  temporal real de 2026. Sete diagnósticos independentes descartaram vazamento espacial,
  sazonalidade, ruído de amostra, deriva administrativa e ENOS. O limite é amostral —
  114 negativos contra 1.238 positivos.
- Petrópolis mistura enchente e movimento de massa nas fontes disponíveis, e o ajuste
  fluvial estima apenas enchente.
- O DINOv2 nunca entrou como variável. Três tentativas independentes fecharam sem sinal;
  numa delas o teste de razão de verossimilhança deu significativo até a correção de
  pseudorreplicação por patch mostrar que era artefato de amostra.

## 6. Incerteza

Bootstrap do preditor linear, reamostrando **grupos** com reposição, N=1000. Cada
reamostra reajusta o Firth; as réplicas de coeficiente ficam gravadas no artefato
servido, e o IC de um escore é o percentil das N projeções.

Não se usou o *delta method*: ele depende da aproximação assintótica do erro-padrão,
frágil exatamente no regime de n pequeno que motivou usar Firth. Reamostra-se grupo e não
linha porque reamostrar linha infla a precisão por pseudorreplicação — o IC sairia
estreito demais, que é o modo de falha mais perigoso num número que vai dentro de uma
resposta de API.

## 7. Artefatos versionados

| Caminho | Conteúdo | Gerado por |
|---|---|---|
| [`execucoes/svc-01-modelos/`](execucoes/svc-01-modelos/) | Os três modelos servidos com réplicas de bootstrap | `scripts/servico/svc01_construir_modelos_servidos.py` |
| [`execucoes/svc-02-contrato/`](execucoes/svc-02-contrato/) | As três respostas reais de demonstração | `scripts/servico/svc02_contrato_inferencia.py` |
| [`execucoes/mod-serra-03/`](execucoes/mod-serra-03/) | Coeficientes por classe de relevo (E3) | `scripts/treino/mod_serra03_relevo_ds05.py` |
| [`execucoes/mod-prosp-02/`](execucoes/mod-prosp-02/) | Cortes, folds e viabilidade por fonte (E4) | `scripts/treino/mod_prosp02_holdout_temporal_ds05.py` |
| [`execucoes/aud-chuva-02/`](execucoes/aud-chuva-02/) | Escala do contraste da chuva por fonte | `scripts/treino/aud_chuva02_escala_do_contraste.py` |
| [`execucoes/ds-03-esquema/`](execucoes/ds-03-esquema/) | Esquema-alvo da tabela única | `scripts/treino/ds03_esquema_alvo.py` |
| [`grade/`](grade/) | Escore por célula a 120 m, GeoTIFF das três regiões (E5) | `scripts/servico/svc03_grade_suscetibilidade.py` |

A tabela única (189 MB) e a grade por célula em CSV (19 MB) não são versionadas. Os
testes que as leem pulam com o comando que as regenera.

## 8. Reproduzir

```bash
conda activate revp-susc
python scripts/treino/ds05_admissao_consolidacao.py      # tabela única
python scripts/treino/mod_serra03_relevo_ds05.py         # E3
python scripts/treino/mod_prosp02_holdout_temporal_ds05.py  # E4
python scripts/servico/svc01_construir_modelos_servidos.py  # modelos servidos
python scripts/servico/svc03_grade_suscetibilidade.py    # E5
python -m pytest tests -q                                 # regressão
```
