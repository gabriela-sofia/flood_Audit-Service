# Matriz do artigo atual

A matriz registra as 98 linhas das nove tabelas e os 32 parágrafos do método, resultados, discussão, conclusão e introdução. A [versão completa em JSON](../manifestos/afirmacoes_artigo.json) conserva o texto-fonte e seus vínculos; a [versão CSV](../manifestos/afirmacoes_artigo.csv) permite filtrar por tabela.

Rastrear uma afirmação não equivale a executar um novo experimento. As evidências distinguem artefatos estruturados de relatórios de execução; os regimes e limites científicos permanecem os do artigo.

## Tabela I — Desenhos de teste

[Abrir o desenho metodológico](METODO.md) e [o registro de admissão](../evidencias/01_curadoria/README.md)

| Item | Resultado ou desenho apresentado |
|---|---|
| I.01 | Presença  ·  água vista na cheia  ·  rua com relato à Defesa Civil  ·  cicatriz mapeada |
| I.02 | Comparação  ·  ponto sem água na mesma cena  ·  outras ruas do mesmo bairro  ·  células de fundo |
| I.03 | Fonte  ·  Sentinel-2; 697 positivos em FCU  ·  SEDEC (ref. sedec); 48 ruas em teste  ·  INPE/Charter (ref. inpe); 81 cicatrizes |
| I.04 | Modelo  ·  versão externa completa  ·  versão com reserva  ·  modelo de encosta |
| I.05 | Pergunta  ·  chega à mesma observação?  ·  chega ao que os moradores registram?  ·  o que muda com o processo? |
| I.06 | Cuidado  ·  nuvem fica desconhecida  ·  rua longa acumula relato; sem relato não é seco  ·  fundo não comprova ausência |

## Tabela II — Aprendizado físico

[Abrir evidências deste bloco](../evidencias/02_aprendizado/README.md)

| Item | Resultado ou desenho apresentado |
|---|---|
| II.01 | Geral - HAND, 43 unidades retidas  ·  +0,039 de AUC [0,026; 0,054]; 33 a favor |
| II.02 | Geral - pesos iguais  ·  +0,018 [0,010; 0,026] |
| II.03 | Sem o bloco de acumulação  ·  -0,064 [-0,080; -0,048]; perda em 39 |
| II.04 | Sem a fonte do destino  ·  -0,0015 [-0,0035; 0,0001] |
| II.05 | AUC fluvial, costeira, urbana mista  ·  0,765, 0,813 e 0,807 |
| II.06 | Am. do Sul sem rótulo sul-americano  ·  0,768 |
| II.07 | De 3 para 42 territórios no treino  ·  0,749 a 0,777; satura com a diversidade |
| II.08 | Sinal de HAND por território  ·  negativo em 25 de 27; varia cerca de 6 vezes |
| II.09 | HAND no evento, 331.492 pontos^{a}  ·  -1,13 [-1,31; -0,96]; 4 famílias de sensor |
| II.10 | HAND em pares a até 3 km^{a}  ·  -1,27 [-1,48; -1,05] |
| II.11 | HAND de 0,5 m a 10 m ou mais^{a}  ·  monotônico, -0,68 a -2,63 |
| II.12 | HAND em área construída^{a}  ·  -0,55; sinal mantido nas 6 coberturas |
| II.13 | Corte de 75% dos negativos, HAND  ·  -0,03 [-0,09; 0,03] no peso |
| II.14 | Árvores - logística  ·  lugar visto: +0,040; 15/16; p<0,001; lugar novo: 8/16 |
| II.15 | HAND (log)  ·  -0,981  ·  -0,904  ·  Areia  ·  -0,051  ·  -0,202 |
| II.16 | Distância ao canal  ·  -0,429  ·  -0,433  ·  Contribuição (log)  ·  +0,043  ·  +0,085 |
| II.17 | Declive  ·  -0,312  ·  -0,274  ·  Distância à drenagem  ·  +0,030  ·  +0,055 |
| II.18 | Fração construída  ·  -0,210  ·  -0,170  ·   ·   ·  |
| II.19 | 30 GFD^{b}  ·  núcleo - HAND: +0,055 [0,024; 0,086] |
| II.20 | Mesmos 30^{b}  ·  com - sem conexão: +0,030 [0,006; 0,053] |
| II.21 | 10 CEMS  ·  com - sem conexão: +0,017 [0,005; 0,030] |
| II.22 | 16 cheias^{c}  ·  logística - HAND: +0,041; 13/16; p=0,011 |

## Tabela III — Transferência

[Abrir evidências deste bloco](../evidencias/03_transferencia/README.md)

| Item | Resultado ou desenho apresentado |
|---|---|
| III.01 | RMPA  ·  241 (17)  ·  0,880 [0,828; 0,923]  ·  +0,040 [-0,002; 0,091]  ·  -0,030 |
| III.02 | Manaus  ·  48 (10)  ·  0,920 [0,791; 0,967]  ·  +0,046 [0,020; 0,069]  ·  +0,111 |
| III.03 | RMC  ·  11 (4)^dagger  ·  0,893^dagger  ·  +0,067^dagger  ·  +0,195^dagger |
| III.04 | Recife  ·  31 (19)  ·  0,866 [0,777; 0,935]  ·  +0,064 [0,012; 0,125]  ·  +0,151 |
| III.05 | RMR  ·  366 (35)  ·  0,809 [0,721; 0,866]  ·  +0,116 [0,069; 0,145]  ·  +0,093 |
| III.06 | RMPA  ·  6.117  ·  0,905  ·  +0,016 [0,005; 0,025]  ·  -0,017  ·  0,884  ·  0,907 |
| III.07 | Manaus  ·  215  ·  0,791  ·  +0,036 [0,000; 0,061]  ·  +0,105  ·  0,528^dagger  ·  0,804 |
| III.08 | RMC  ·  883  ·  0,811  ·  +0,006 [-0,012; 0,018]  ·  +0,056  ·  0,634^dagger  ·  0,812 |
| III.09 | Recife  ·  448  ·  0,781  ·  +0,043 [0,019; 0,068]  ·  +0,087  ·  0,844  ·  0,739 |
| III.10 | RMR  ·  2.882  ·  0,775  ·  +0,085 [0,055; 0,106]  ·  +0,069  ·  0,770  ·  0,748 |
| III.11 | Rótulos permutados  ·  a ordenação cai de 0,30 a 0,63: vem do aprendido |
| III.12 | Cobertura, solo (FCU)  ·  Recife +0,108 [0,045; 0,174]; RMR +0,141 [0,068; 0,170] |
| III.13 | Outro sensor  ·  99% dos positivos da RMPA na mancha Planet/SkySat (ref. ufrgs) |
| III.14 | 14 municípios^{c}  ·  árvores - HAND: +0,021 [0,000; 0,034]; alcança o não rotulado |
| III.15 | Dois novos^{d}  ·  AUC em FCU: 0,890 e 0,882; lugar nunca examinado |

## Tabela IV — Recife

[Abrir evidências deste bloco](../evidencias/04_recife/README.md)

| Item | Resultado ou desenho apresentado |
|---|---|
| IV.01 | Da fonte à unidade  ·  1.648 relatos, 2014--2024 to 420 até a rua to 96 unidades em 38 bairros to 48 em teste |
| IV.02 | População de teste  ·  48 ruas com relato em 19 bairros; 2.225 ruas; 42% das relatadas no quintil superior (HAND: 31%) |
| IV.03 | Núcleo, AUC no bairro  ·  0,669 [0,564; 0,775] |
| IV.04 | HAND, AUC no bairro  ·  0,595 [0,491; 0,711] |
| IV.05 | Vantagem no teste  ·  núcleo - HAND: +0,074 [0,012; 0,159] |
| IV.06 | Extensão da rua  ·  mediana de 556 m contra 154 m; AUC 0,743 |
| IV.07 | Escore times extensão, no bairro  ·  rho=0,03 |
| IV.08 | Núcleo - HAND, pareado^{a}  ·  +0,058 [-0,024; 0,177]; +0,011 [-0,053; 0,086] |
| IV.09 | Com relato, com e sem satélite^{b}  ·  19 e 16 bairros |
| IV.10 | Sem relato, com e sem satélite^{b}  ·  36 e 23; razão de chances 0,76; p=0,67 |
| IV.11 | Chuva por 10 mm, 1.372 relatos^{c}  ·  dia 1,31 [1,07; 4,63]; 48 h 1,46 [1,20; 2,73]; controle 0,78 [0,51; 1,06] |
| IV.12 | Recife, 295 FCU  ·  228 cobertas; 74 com rua relatada; 33 avaliadas (50.198 moradores) |
| IV.13 | HAND (log)  ·  +0,751  ·  +0,204 |
| IV.14 | Distância ao canal  ·  +0,853  ·  +0,093 |
| IV.15 | Declive  ·  +0,272  ·  +0,123 |
| IV.16 | Fração construída  ·  -0,525  ·  -0,406 |
| IV.17 | Areia  ·  +0,033  ·  +0,033 |
| IV.18 | Contribuição (log)  ·  +0,129  ·  -0,013 |
| IV.19 | Distância à drenagem  ·  -0,026  ·  +0,023 |
| IV.20 | Escore final  ·  1,487  ·  0,057 |
| IV.21 | Relatos / ruas com escore abaixo  ·  3 / 72 de 74  ·  4 / 12 de 74 |

## Tabela V — Petrópolis

[Abrir evidências deste bloco](../evidencias/05_petropolis/README.md)

| Item | Resultado ou desenho apresentado |
|---|---|
| V.01 | Encosta (Idai), 81 cicatrizes  ·  AUC 0,660 [0,593; 0,726]; -declive: +0,007; 12.170 células de fundo; 40% no quintil superior |
| V.02 | Modelo de água, mesmas cicatrizes  ·  AUC 0,375 [0,307; 0,445] |
| V.03 | Encosta - água  ·  +0,286 [0,170; 0,396] |
| V.04 | Petrópolis, B=10: externo e local - só local  ·  +0,222 [0,126; 0,307] |
| V.05 | Petrópolis, B=100: externo e local - só local  ·  +0,126 [0,070; 0,180] |
| V.06 | Petrópolis, B=100, endereços de dano - externo  ·  -0,158 [-0,247; -0,065] |
| V.07 | Petrópolis, 42 FCU  ·  registro interno de 5 para 20; 31 a até 100 m; 43.019 moradores |

## Tabela VI — Atualização

[Abrir evidências deste bloco](../evidencias/06_atualizacao/README.md)

| Item | Resultado ou desenho apresentado |
|---|---|
| VI.01 | RMPA, FCU, B=400: margem - congelado  ·  +0,052 [0,020; 0,101]; 6/6 |
| VI.02 | RMPA, FCU, B=400: sorteio - congelado  ·  +0,029 [0,009; 0,062]; 6/6 |
| VI.03 | RMPA, FCU, B=400: margem - sorteio  ·  +0,023; 6/6 |
| VI.04 | RMPA, coorte, B=400: atualizado - externo  ·  +0,021 [0,015; 0,026] |
| VI.05 | RMPA, coorte, B=25: prior - neutro  ·  +0,011 [0,001; 0,025] |
| VI.06 | RMPA, coorte, B=25: prior - permutado  ·  +0,020 [0,008; 0,037] |
| VI.07 | Retenção externa, RMPA, margem: T_B  ·  -0,028 [-0,056; -0,006] |
| VI.08 | RMPA, FCU, 3 destinos, memória de 200  ·  +0,017 [0,006; 0,027] |
| VI.09 | FCU da RMR^{c}  ·  prior - só local: +0,026 [0,014; 0,049] |

## Tabela VII — Percepção

[Abrir evidências deste bloco](../evidencias/07_percepcao/README.md)

| Item | Resultado ou desenho apresentado |
|---|---|
| VII.01 | U-Net times Otsu, IoU^{a}, 90 chips  ·  0,449 times 0,395, sem água permanente |
| VII.02 | Recusa pela confiança  ·  IoU 0,592, cobertura 0,556; AURC 0,560 times 0,684 |
| VII.03 | Radar e só terreno, 431 chips, AUC  ·  0,930--0,942 e 0,85--0,87 |
| VII.04 | LwF - continuação, IoU  ·  +0,049 [0,029; 0,068]; 8/8 |
| VII.05 | Conjunto diverso de pré-treino  ·  reconstrução: 6/6 sementes; p=0,031 |
| VII.06 | Pré-treino to segmentação  ·  indecidível; 7/16 sementes |

## Tabela VIII — Resposta

[Abrir evidências deste bloco](../evidencias/08_resposta/README.md)

| Item | Resultado ou desenho apresentado |
|---|---|
| VIII.01 | Busca densa times lexical, 984 casos  ·  0,948 times 0,119; AURC 0,011 times 0,025 |
| VIII.02 | Híbrida times lexical, 11.238 consultas  ·  recall@5: 44,35% times 36,65% |
| VIII.03 | Juiz e regra times só regra, 18 casos  ·  acurácia 0,722 times 0,444; falsos positivos 1 times 7 |
| VIII.04 | Tranquilizadoras erradas^{b}  ·  FCU RMPA 0 times 35; FCU RMR 25 times 130; retenção 1.038 times 1.741 |
| VIII.05 | Recusa estratificada, FCU da RMPA  ·  cobertura 0,40 to 0,70; erro 0,295 times 0,290 |
| VIII.06 | Escore reproduzido, 147.798 pontos  ·  erro máximo 3,56times10^{-15} |
| VIII.07 | Entrega local de 20/10; serviço  ·  2.747 testes; 922 fontes por hash; 5 portões; grade de 120 m |

## Tabela IX — Alcance social

[Abrir evidências deste bloco](../evidencias/09_social/README.md)

| Item | Resultado ou desenho apresentado |
|---|---|
| IX.01 | Destinos brasileiros  ·  1.708 FCU; 2.776.821 moradores |
| IX.02 | FCU, de 1.666  ·  de 1 para 123 positivas; 119 com duas classes; 38 com 5 e 5 |
| IX.03 | Observabilidade^{a}  ·  RMR 0,61 / 0,44 / 0,62; RMPA 0,95 / 0,78 / 0,90; Recife 0,82 / 0,90 / 0,91 |
| IX.04 | Medidas sociais  ·  9.186 setores; 7 componentes; EMSR720: 1.131 a 16.068 |
| IX.05 | Revisão cega, RMPA  ·  480 pontos, sem respostas ainda; desenho: desvio de 0,075 a 0,021 |
