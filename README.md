# REV-P

Suscetibilidade urbana a enchentes com base causal físico-hidrológica, em três regiões
brasileiras — Recife, Curitiba e Petrópolis — com validação externa no Reino Unido e nas
áreas cobertas pelo Copernicus EMS.

O projeto parte do caminho oposto ao usual: em vez de deixar um modelo descobrir padrões
em imagem de satélite, usa relações físico-hidrológicas já conhecidas como base causal e
testa essas relações contra evento real registrado por fonte oficial. Suscetibilidade,
aqui, é a predisposição do terreno a acumular água sob um dado forçamento de chuva —
quanto chega (chuva), para onde converge (TWI) e quanto precisa subir para ser alcançado
(HAND), com o escoamento repartido por D-infinity.

Recife é a entrega madura e auditada ponta a ponta. Curitiba e Petrópolis são resultados
parciais, reportados como vieram — inclusive quando o resultado é negativo.

---

## Estrutura

```text
REV-P/
├── docs/
│   ├── DOCUMENTACAO_TECNICA_CONSOLIDADA.md   # premissa, turning points, portões, métricas
│   └── tcc_exports/planejamento_entrega01/
│       └── main.pdf                          # documento de entrega
├── outputs_public/data/susc_20*/             # as 11 etapas da linha causal
├── scripts/dino/                             # apoio à fila de revisão visual (DINOv2)
├── tests/                                    # regressão da linha causal
├── environment.yml                           # ambiente conda (Firth)
└── requirements.txt                          # ambiente da camada DINOv2
```

Cada etapa em `outputs_public/data/susc_20*/` é autocontida: traz os scripts que a
produziram, os dados curados de saída e um relatório em `reports/`.

| Etapa | Entrega |
|---|---|
| `20a` | Aquisição de evento real em Recife |
| `20b` | Atributos físico-hidrológicos |
| `20c` | Modelagem e validação estatística (Firth) |
| `20d` | Motor de inferência local |
| `20e` | Contrato de API por região |
| `20f` | Geoprocessamento sob demanda |
| `20g` | HAND e TWI por D-infinity |
| `20h` / `20j` | Candidatos de água por Sentinel-2 e Sentinel-1 |
| `20i` | Janelas de evento 2023–2026 |
| `20k` | Curitiba: candidatos, negativos, features, modelagem e diagnósticos |

---

## Resultados

Base harmonizada com **65.070 pontos elegíveis ao ajuste fluvial**, reduzidos de seis
fontes, todas na mesma cadeia de derivação de terreno e com chuva de fonte única
(ERA5-Land, cobertura 99,99%).

| Região | Resultado |
|---|---|
| **Recife** | Firth, n=278 (154 positivos da SEDEC / 124 negativos), LOO-AUC 0,68. Motor de inferência e contrato entregues. Maturidade `mvp_local`. |
| **Curitiba** | 1.045 positivos do SIAC 156, 1.471 unidades de validação. O modelo próprio não generaliza (AUC 0,65 → 0,52 em holdout real) e não sustenta holdout próprio: 114 negativos contra 1.238 positivos. Resultado negativo documentado. |
| **Petrópolis** | Zero linhas na tabela única. Servido por transferência sem referência local — predição, nunca afirmação de acerto. |
| **Frente externa** | Holdout temporal: 201 eventos em 110 datas (2000–2025), 8 cortes na faixa 0,70–0,88, AUC médio **0,7992**. Por classe de relevo: serra 0,7916, planície 0,7245, transferência planície→serra 0,7957. |

Grade de aplicação a 120 m: 56.666 células em Recife, 65.275 em Curitiba, 172.015 em
Petrópolis. Célula fora do domínio do ajuste fica vazia no mapa em vez de receber escore
baixo — recusar não é o mesmo que dizer que ali é seguro.

**O que o projeto não afirma:** que o modelo de Curitiba esteja operacional, que o escore
de Petrópolis tenha sido validado contra evento local, ou que o DINOv2 substitua a base
físico-hidrológica. As três tentativas de promovê-lo a preditor fecharam sem sinal; ele
permanece apenas alimentando a fila de revisão humana.

O detalhamento — premissa física, o que funcionou e o que foi descartado, os cinco
portões do contrato e as métricas com intervalo — está em
[`docs/DOCUMENTACAO_TECNICA_CONSOLIDADA.md`](docs/DOCUMENTACAO_TECNICA_CONSOLIDADA.md).

---

## Executar

```bash
conda env create -f environment.yml
conda activate revp-susc
python -m pytest tests/test_susc_2*.py -q
```

Linha causal (Firth): ambiente conda com Python `<3.11`, ver `environment.yml`. Camada
DINOv2: `requirements.txt`, Python padrão. Toda a modelagem roda localmente, sem serviço
externo de treinamento.
