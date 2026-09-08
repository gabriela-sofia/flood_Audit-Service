# Por que parte da cadeia desta execucao fica fora do pacote publico

Os artefatos deste diretorio vem do experimento de pre-treino mascarado da frente
neural. A analise que decide o experimento esta no pacote; o treino que o produz, nao.

| Etapa | Script | Esta no pacote |
|---|---|---|
| Regua estatistica do experimento pareado | `scripts/neural/regua.py` | sim |
| Catraca: so grava campeao que bateu o anterior | `scripts/neural/catraca.py` | sim |
| Selo de proveniencia do peso | `scripts/neural/proveniencia_de_peso.py` | sim |
| Pre-treino mascarado e ajuste fino | `scripts/sar/pretreino_mascarado.py` e a frente SAR | **nao** |
| Experimento de pote crescente | `scripts/neural/experimento_pote_crescente.py` | **nao** |
| Volatilidade da rodada | `scripts/neural/volatilidade_da_rodada.py` | **nao** |
| Experimento de acumulacao | `scripts/neural/experimento_acumulacao.py` | **nao** |

O motivo e de escopo, nao de sigilo: esses tres scripts dependem da frente SAR inteira
(rede, dados Sen1Floods11, laco de treino em `torch`), que e um pacote de pesquisa com
outro tamanho, outras dependencias e outro tempo de execucao. Ele nao participa do
escore servido, nao entra no contrato de inferencia e nao sustenta nenhuma afirmacao
deste repositorio sobre suscetibilidade.

O que fica registrado aqui e o RESULTADO com a limitacao junto: reconstrucao mascarada
em recortes reservados, `n=6` sementes, ganho mediano `0,119623`, efeito detectado por
teste de postos e veredito parametrico `INDECIDIVEL`, com as duas leituras lado a lado.
Isso e verificavel lendo os JSON; nao e reproduzivel deste checkout.
