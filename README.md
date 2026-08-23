# Suscetibilidade urbana a enchentes com base causal físico-hidrológica

Da evidência heterogênea ao serviço de inferência auditável.

Enchentes são o desastre de maior recorrência no Brasil, e atingem primeiro quem não
escolheu onde morar. Este projeto parte do caminho oposto ao usual: em vez de deixar um
modelo descobrir padrões em imagem de satélite, usa relações físico-hidrológicas já
conhecidas como base causal e testa essas relações contra evento real registrado por
fonte oficial.

Suscetibilidade, aqui, é a predisposição do terreno a acumular água sob um dado
forçamento de chuva — **quanto chega** (chuva antecedente), **para onde converge** (índice
topográfico de umidade) e **quanto precisa subir para ser alcançado** (HAND), com o
escoamento repartido por D-infinity em vez de despejado numa das oito células vizinhas.

O produto final é um contrato de inferência: você entrega uma geometria, ele devolve um
escore por área com intervalo de confiança, a maturidade da região e a explicação — ou
recusa, dizendo qual portão falhou.

> `REV-P` é o código interno do projeto. Aparece nos prefixos `revp_` dos arquivos e na
> nomenclatura de estágios (`SUSC-20A`, `MOD-SERRA-03`, `v1pv`). Não é sigla com
> expansão: é só o nome que o repositório recebeu quando nasceu.

---

## Comece por aqui

| Se você quer | Leia |
|---|---|
| Entender **como funciona** e por que assim | [`docs/METODO.md`](docs/METODO.md) |
| Saber **por que confiar** nos números | [`docs/EVIDENCIA.md`](docs/EVIDENCIA.md) |
| Ver **o que foi treinado** e onde não vale | [`modelo/MODELO.md`](modelo/MODELO.md) |
| Ver **de que dados** ele saiu | [`modelo/DADOS.md`](modelo/DADOS.md) |

---

## Resultados

Base harmonizada com **65.070 pontos elegíveis ao ajuste fluvial**, reduzidos de seis
fontes, todas na mesma cadeia de derivação de terreno e com chuva de fonte única
(ERA5-Land, cobertura 99,99%).

| Região | Resultado | Maturidade |
|---|---|---|
| **Recife** | Firth, n=278 (154 positivos da SEDEC / 124 negativos), LOO-AUC 0,68. Motor de inferência e contrato entregues | `mvp_local` |
| **Curitiba** | 1.045 positivos do SIAC 156, 1.471 unidades de validação. Servida pelo ajuste fluvial de planície, com a distância de domínio declarada na resposta | `transferencia_caracterizada` |
| **Petrópolis** | Servida pelo ajuste de serra; 91,3% do território cabe na faixa de HAND que o modelo viu | `transferencia_sem_referencia_local` |

| Etapa | Resultado |
|---|---|
| **E3** — ajuste por classe de relevo | Serra 0,7916 (`hand_m` −1,44 [−3,11; −0,83]); planície 0,7245 (`hand_m` −2,10 [−2,78; −1,56]; `twi_dinf` +0,40 [+0,33; +0,45]); transferência planície→serra **0,7957** |
| **E4** — holdout temporal | 201 eventos em 110 datas (2000–2025), 8 cortes na faixa 0,70–0,88, AUC médio **0,7992** |
| **E5** — grade de aplicação | 56.666 células em Recife, 65.275 em Curitiba, 172.015 em Petrópolis, a 120 m |
| **E6** — contrato de inferência | Função pura com cinco portões em ordem declarada; falta o transporte HTTP |

---

## Estrutura

```text
├── docs/
│   ├── METODO.md          premissa física, portões, glossário, Protocolo C, mapa do código
│   ├── EVIDENCIA.md       tabela-mestra das rodadas: data, script, artefato, veredito
│   └── FONTES.md          de onde veio cada dado, quando, e por que é válido
├── modelo/
│   ├── MODELO.md          cartão do modelo — uso pretendido, desempenho, onde não vale
│   ├── DADOS.md           ficha da base — composição, coleta, o que ficou de fora
│   ├── execucoes/         artefatos das rodadas de treino e do contrato
│   └── grade/             escore por célula a 120 m, GeoTIFF das três regiões
├── scripts/
│   ├── treino/            tabela única, E3, E4, auditoria da chuva
│   ├── servico/           modelos servidos, contrato de inferência, grade
│   ├── terreno/           cadeia D-infinity harmonizada
│   ├── externo/           aquisição do negativo: Copernicus EMS, UK, fontes globais
│   └── dino/              fila de revisão visual (nada aqui entra no modelo)
├── outputs_public/data/   as onze etapas da linha causal, autocontidas
└── tests/                 regressão da linha causal, do treino e do serviço
```

Cada etapa em `outputs_public/data/susc_20*/` é autocontida: traz os scripts que a
produziram e os dados curados de saída. As fontes de todas elas estão consolidadas em
[`docs/FONTES.md`](docs/FONTES.md).

---

## Executar

```bash
conda env create -f environment.yml
conda activate revp-susc
python -m pytest tests -q
```

São dois ambientes, de propósito: `environment.yml` para a linha causal e o serviço
(Python `<3.11`, exigência do estimador de Firth) e `requirements.txt` para a camada
DINOv2. Toda a modelagem roda localmente, sem serviço externo de treinamento.

Parte dos testes lê artefatos grandes demais para versionar — a tabela única tem 189 MB.
Eles **pulam** em vez de falhar, e a mensagem traz o comando que os regenera. Para
reproduzir a cadeia inteira, ver [`modelo/MODELO.md`](modelo/MODELO.md) §8.
