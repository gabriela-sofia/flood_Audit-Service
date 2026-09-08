# Roteiro de estudo

Este arquivo e para quem precisa aprender o projeto do zero, ou ensina-lo. Ele comeca
pelo produto servido e so depois entra nas frentes auxiliares — a ordem inversa faz a
pessoa confundir experimento com nucleo operacional.

## 1. Ordem para ler o codigo

| # | Arquivo | Por que nesta posicao |
|---|---|---|
| 1 | `scripts/servico/svc02_contrato_inferencia.py` | E o nucleo. Decide escore, recusa, maturidade e limite. Tudo o mais existe em funcao dele. |
| 2 | `scripts/servico/svc03_grade_suscetibilidade.py` | Mostra o contrato aplicado em lote, celula por celula. |
| 3 | `scripts/servico/svc01_construir_modelos_servidos.py` | Mostra de onde vem o que o contrato serve, e por que um candidato e reprovado. |
| 4 | `scripts/treino/ds01_montar_multirregiao.py` ate `ds05_admissao_consolidacao.py` | A curadoria que produz a tabela unica. Vale tanto quanto o treino. |
| 5 | `scripts/treino/susc_firth_shim.py` | Por que o estimador e o que e. |
| 6 | `scripts/chuva/contrato_chuva.py` e `scripts/chuva/autopareamento_temporal.py` | Os dois papeis da chuva: variavel e pergunta cientifica separada. |
| 7 | `scripts/chuva/terreno_sob_gatilho_1km.py` e `terreno_sob_gatilho_chicago.py` | O teste de transferencia, com o estimador importado em vez de reescrito. |
| 8 | `scripts/social/camada_social.py` e `scripts/social/ibge_curitiba.py` | Camada paralela, com invariantes que a mantem paralela. |
| 9 | `scripts/temporalidade/dimensao_temporal.py` | Como um estado cientifico e produzido e preservado. |
| 10 | `scripts/servico/verificador_de_afirmacoes.py` | A barreira entre ter um resultado e poder publicar uma frase sobre ele. |
| 11 | `scripts/neural/cadeia_seletiva.py` e `scripts/neural/catraca.py` | Frente exploratoria, lida por ultimo de proposito. |
| 12 | `scripts/dino/` | Registro historico de revisao visual; nao participa do escore. |

## 2. Ordem para ler a documentacao

1. [`README.md`](../README.md) — o que o pacote entrega hoje.
2. [`docs/CONCEITOS.md`](CONCEITOS.md) — vocabulario, antes de qualquer numero.
3. [`docs/METODO.md`](METODO.md) — a cadeia e os portoes.
4. [`docs/ARQUITETURA.md`](ARQUITETURA.md) — como as partes se encaixam.
5. [`modelo/DADOS.md`](../modelo/DADOS.md) — de onde vem o dado.
6. [`modelo/MODELO.md`](../modelo/MODELO.md) — pesos, dominio e maturidade.
7. [`docs/VALIDACAO.md`](VALIDACAO.md) — como se mede e como nao se mede.
8. [`docs/EVIDENCIA.md`](EVIDENCIA.md) — o que cada frente encontrou.
9. [`docs/LIMITACOES.md`](LIMITACOES.md) — o que impede afirmacao mais forte.
10. [`docs/REGIOES.md`](REGIOES.md) — o retrato por regiao.
11. [`docs/AFIRMACOES.md`](AFIRMACOES.md) — o que pode e o que nao pode ser dito.

## 3. Quatro aulas

### Aula 1 — Problema, etica e contrato

Objetivo: entender por que o projeto existe e por que ele recusa.

1. A diferenca entre suscetibilidade, risco, evento e vulnerabilidade.
2. Por que ausencia de dado nao e ausencia de risco, e a hierarquia dos quatro tipos de
   negativo.
3. O contrato de inferencia como mecanismo de seguranca, e os cinco portoes em ordem.
4. Exemplos de saida: escore com intervalo, maturidade, e recusa com motivo.
5. Fechar com as afirmacoes permitidas e vedadas.

Leitura: `CONCEITOS.md`, `METODO.md` secao 6, `AFIRMACOES.md`.

### Aula 2 — Dados e variaveis fisicas

Objetivo: entender como dado bruto vira variavel util.

1. As oito fontes da tabela unica e o papel de cada uma.
2. A tabela unica v2: 595.865 de entrada, 568.265 admitidas, e o que saiu e por que.
3. HAND, TWI D-infinity, declividade, elevacao relativa e chuva antecedente — a
   intuicao fisica de cada uma.
4. O criterio do pool fluvial e por que ele existe.
5. Contagem por fonte, duplicidade intra-fonte e a violacao de dominio aberta.
6. Por que a curadoria vale tanto quanto o treino.

Leitura: `../modelo/DADOS.md`, `CONCEITOS.md` secao 6.

### Aula 3 — Modelos, pesos e validacao

Objetivo: entender o motor cientifico.

1. Regressao logistica com correcao de Firth, em linguagem simples: o que ela conserta.
2. Os coeficientes dos cinco candidatos, um a um.
3. Como ler um IC 95%, e o que acontece com a variavel quando ele cruza zero.
4. AUC e validacao agrupada: por que o grupo nunca e mais fino que a celula.
5. Aderencia temporal nao e suporte temporal.
6. Maturidade regional: por que subir exige dado novo, nao ajuste.
7. Comparar Recife, Curitiba, Petropolis, NYC e Sao Paulo — inclusive o que **nao** se
   pode comparar.

Leitura: `../modelo/MODELO.md`, `VALIDACAO.md`, `REGIOES.md`.

### Aula 4 — Frentes auxiliares e produto

Objetivo: entender o ecossistema sem misturar papeis.

1. A grade de 120 m: celulas validas, serviveis e recusadas.
2. Por que a chuva entra como cenario na grade e como variavel so em Recife.
3. Chuva como forcamento: o desenho autopareado e o que ele cancela por construcao.
4. O teste da segunda cidade, e o que ele reprovou.
5. Camada social territorial: paralela, com invariantes.
6. Recuperacao de fonte: rastreabilidade, nao decisao.
7. Frentes neurais e predicao seletiva: demonstrativo, com regua contra o proprio acaso.
8. Fechar com o checklist de auditoria.

Leitura: `EVIDENCIA.md`, `VALIDACAO.md` secoes 6 a 8.

## 4. Exercicios de leitura critica

Para verificar se a pessoa entendeu, peca que ela responda **usando artefato**:

1. Por que o `recife_pluvial` e servido mesmo com veredito `FORA_DOS_CRITERIOS`?
2. Por que o `fluvial_serra` usa uma unica variavel?
3. Por que o escore de Recife nao pode ser comparado com o de Curitiba?
4. O que muda na leitura do escore quando os negativos sao fundo em vez de observados?
5. Qual a diferenca entre `not_testable` e `inconclusive`, e por que preservar as duas?
6. Se Chicago reprovou o TWI, por que os modelos servidos nao mudaram?
7. Onde esta registrado que 1.703 celulas de Recife foram recusadas, e por que isso e
   parte do produto?
8. Que evidencia seria necessaria para Petropolis subir de maturidade?

## 5. Como reproduzir

```bash
conda env create -f environment.yml
conda activate revp-susc
python -m pytest tests -q
```

Cada pasta em `modelo/execucoes/` tem um `commands.txt` com a sequencia que a produziu.
Testes que pulam declaram o motivo: ou a tabela completa nao esta no checkout, ou a
camada guardada fica fora deste pacote publico.
