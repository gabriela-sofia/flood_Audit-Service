# Petrópolis: aprendizado de encostas, transferência e integração territorial — resultados

24/10/2026. Execução de `docs/PROMPT_EXECUCAO_TREINO_PETROPOLIS_ARTIGO_2026_10_24.md`. Linhagem nova e aditiva: `local_runs/petropolis-aprendizado/v1/` (protocolo congelado `d218cfa4…`, emenda 1 `ea3293e6…`). Código em `scripts/petropolis_aprendizado/`, testes em `tests/test_petropolis_aprendizado_v1.py` (9 aprovados). Nenhum modelo ativo, reserva, selo, artefato DINO ou produtor anterior foi alterado: os 54 arquivos protegidos conferem com os hashes iniciais. Nada foi commitado, promovido ou enviado a terceiros.

## 1. Em uma frase

A ferramenta aprendeu, numa área com inventário completo (ciclone Idai, Zimbábue), uma relação de encosta que **transfere para Petrópolis acima do acaso** (AUC por cicatriz 0,66 no episódio de fevereiro de 2022), mas que **não supera a regra de declive** (0,65). As **observações locais disponíveis não melhoraram essa ordenação**: os relatos históricos de dano inverteram o modelo, e as cicatrizes de 2022 de outros bairros não acrescentaram ganho fora do bairro. O ganho demonstrado está na **informação territorial**: 31 das 42 comunidades passaram a ter registro observado de processo de encosta, contra 5 antes, sempre com origem, natureza e lacunas declaradas.

## 2. O que foi ensinado ao modelo, e com quais exemplos

| Fonte | Papel | Natureza | O que ela representa |
|---|---|---|---|
| Dille et al. (2026), ciclone Idai, 2019: 14.900 pontos de fonte e 36.026 polígonos (deplecão, percurso, fluxo com detritos) | doador | ruptura, contra ausência mapeada no evento | inventário por evento, completo no suporte declarado |
| INPE/Charter 0864, Petrópolis, fev/2022: 81 deslizamentos, 3 fluxos, 1 inundação | teste primário | cicatriz inteira (ruptura + percurso) | interpretação de imagem pós-evento |
| UFRJ/GEOCART (camada pública do SIG municipal): 1.592 registros, 1.401 deslizamentos (1933–2015) | atualização local (v1) e teste secundário | relato de dano por endereço | jornais (IPT) e GEOCART; posição do dano, não da cicatriz |
| SIG municipal, áreas de risco 2025 | contexto técnico | avaliação técnica | posterior a 2022; nunca usado como preditor nem como validação |
| svc-03 `fluvial_serra` | componente água | inferência congelada | transferência sem referência local |

O modelo recebeu **sete atributos de terreno anteriores ao evento**, derivados do mesmo MDT (FABDEM V1-2, 30 m) e da mesma cadeia WhiteboxTools do serviço, nas duas regiões:

- declividade média (3×3) e máxima (5×5);
- curvaturas em planta e em perfil;
- log da área contribuinte específica (D-infinito);
- amplitude em 210 m;
- posição topográfica em 330 m.

Ficaram de fora de propósito: HAND e TWI (variáveis da água), elevação absoluta (não transferível), qualquer imagem pós-evento e o zoneamento de 2025.

No Idai, a logística (λ = 100, escolhido no desenvolvimento do próprio Idai e situado no extremo da grade) aprendeu coeficientes padronizados positivos para as duas medidas de declive (+1,32 e +0,86). Dado o declive, a amplitude local (−0,78) e a posição topográfica (−0,30) entram com sinal negativo, e as curvaturas e a área contribuinte entram com pesos pequenos. São parâmetros do estimador sob este desenho. Não são pesos causais do fenômeno.

## 3. Resultados por pergunta

### 3.1 Aprendizado na origem — **demonstrado no Idai**

Nos 13 blocos de 10 km retidos do Idai (fonte contra ausência mapeada), a AUC foi 0,885 com a logística e 0,897 com boosting, contra 0,877 da regra de declive. Retirar o bloco de declive custa 0,023 de AUC; os outros blocos mudam menos de 0,005. Essa é uma referência de duas classes no suporte mapeado, sujeita à omissão do mapeador.

### 3.2 Transferência para Petrópolis sem rótulo local — **acima do acaso; ganho sobre o declive inconclusivo**

Teste primário: 81 cicatrizes e 12.170 células de fundo não rotulado em 11,9 km². A prevalência celular é de 2,2%.

| Braço | AUC por cicatriz (IC95) | AP celular | Cicatrizes no quintil superior |
|---|---|---|---|
| Regra de declive | 0,654 [0,577; 0,722] | 0,049 | 38% |
| Idai, logística | 0,660 [0,587; 0,726] | 0,052 | 40% |
| Idai, boosting | 0,620 [0,551; 0,682] | 0,033 | 21% |

O contraste primário C1 (Idai − declive) foi **+0,007 [−0,019; 0,032], inconclusivo** pela regra congelada (margem de 0,02, IC excluindo zero). O que se transferiu é essencialmente a relação com o declive e o relevo. No destino, retirar o declive custa 0,042, e retirar o relevo custa 0,016. O modelo não linear transferiu pior que o linear.

**Significado territorial.** Usar o conhecimento de outra região serrana ordena as encostas que romperam em 2022 melhor que o acaso, sem nenhum rótulo de Petrópolis. Porém o ganho aparente sobre "olhar o declive" não se distingue de zero neste teste. Trata-se de presença contra fundo: AUC e AP medem priorização de inventário, não especificidade nem probabilidade física.

### 3.3 Aprendizado local com relatos históricos (protocolo v1) — **contrário: piora**

Com B = 10 a 402 sítios históricos, sempre fora do suporte de teste + 300 m (verificado: zero sítios consultados no teste):

- **Só relatos (A2):** AUC por cicatriz de 0,37 a 0,41, **abaixo do acaso**.
- **Idai + relatos (A3, λ = 25):** cai de 0,63 (B = 10) para 0,43 (B = 402).
- **C2 (B = 100):** −0,158 [−0,247; −0,065], **contrário**.
- **C3 (externo contra só local, mesma regularização):** +0,126, sustentado. O conhecimento externo **limita o dano** de dados locais mal posicionados.
- **C4 (coeficientes reais contra permutados):** +0,071, sustentado. O efeito vem da estrutura aprendida, não só da regularização.
- **Retenção:** o Idai retido cai de 0,885 para 0,65 com B = 100 e para 0,36 com todos os sítios.

**Diagnóstico.** Os relatos ficam em posições baixas e côncavas: TPI mediano de −9 m e declive de 14°, contra 25° nas cicatrizes de 2022. Sem rótulo local, os modelos já os colocam abaixo do acaso (AUC 0,36–0,39 no teste secundário). Com o município inteiro como fundo, o coeficiente aprendido para o declive fica negativo (D1). O endereço marca **onde o dano foi registrado**, não onde a encosta rompeu.

O diagnóstico D5 (emenda 1) propagou o escore a montante por sete passos D8. A AUC subiu de 0,39 para 0,46, mas continuou abaixo de 0,5. Entre os geocódigos precisos, a inversão é maior (0,33). Logo, a leitura "o relato é o receptor do material" é **apenas parcialmente apoiada**, e não é ruído de geocodificação.

### 3.4 Aprendizado local com cicatrizes de 2022 (emenda 1, desenvolvimento) — **sem ganho fora do bairro**

A emenda foi desenhada depois do v1 e usa o episódio que já tinha sido o teste: é desenvolvimento, não confirmação. A validação cruzada usou cinco bairros (Alto da Serra 21, Caxambu 10, Coronel Veiga 14, Centro 22, Chácara Flora 14). O fundo de ajuste ficou a mais de 150 m do bairro retido; as cicatrizes vizinhas de outros bairros não tiveram esse afastamento.

| Braço (K = todas as cicatrizes dos outros bairros) | AUC por cicatriz fora do bairro |
|---|---|
| Regra de declive | 0,661 |
| Idai sem rótulo local | 0,668 |
| Só cicatrizes locais (λ = 25) | 0,608 |
| Idai + cicatrizes (λ = 25) | 0,648 |
| Controle com prior permutado | 0,545 |

- **E1_C2 (Idai + cicatrizes − Idai):** −0,020 [−0,039; −0,003], contrário no limite da margem.
- **E1_C3:** +0,040, sustentado. **E1_C4:** +0,103, sustentado.
- **Só local contra declive:** −0,053, contrário.
- **Retenção no Idai:** preservada (0,883).

**Significado.** Cicatrizes mapeadas ensinam uma relação com o sinal certo; os relatos não. Mas cerca de 60 cicatrizes de bairros vizinhos não bastam para superar o conhecimento externo nem o declive num bairro novo. A combinação preserva a origem e fica perto do modelo congelado. Aprendizado local com ganho **não foi demonstrado** em Petrópolis.

### 3.5 O que a ferramenta passou a informar sobre as comunidades — **implementado e medido como cobertura de informação**

A candidata (`territorio/candidata_resposta_territorial.json`) separa sete campos: `agua`, `encosta` (observado / contexto técnico / inferido / desconhecido), `conexao`, `contexto_social`, `fontes`, `limites` e `observacao_necessaria`. Nenhum escore é somado.

| Informação por FCU (de 42) | Ficha anterior | Agora |
|---|---|---|
| Registro observado de encosta | 5 | **31** (26 só por relato histórico até 100 m) |
| Contexto técnico 2025 | 36 | 36 |
| Inferência de encosta com ≥ 80% das células no domínio do doador | 0 | 37 |
| Conexão a montante (até 1 km) com encosta no quintil superior | 0 | 33 |
| Escore de água explícito, com maturidade | 0 | 42 |
| Sem nenhum registro observado | 37 | 11 |

A população das FCU com registro observado passou de **3.758 para 34.338** moradores, de um universo de 43.019. É o universo censitário de 2022, não pessoas atingidas. Os relatos que falharam como rótulo de treino são, como evidência territorial, a maior adição da campanha. Mostram recorrência histórica de dano nas comunidades, sem dizer onde a encosta rompe.

**Água versus encosta (D2).** O escore de água servido teve AUC por cicatriz de **0,37** nos deslizamentos de 2022, com apenas 4% das cicatrizes no seu quintil superior. As encostas que romperam estão justamente onde o escore de água é baixo. Por isso a ferramenta registra em cada ficha que um escore baixo de água não autoriza declarar segurança contra movimentos de massa. Esse resultado é a demonstração empírica da regra, não apenas uma ressalva.

**Atualização recusada.** O estado atualizado com relatos aparece na candidata como `recusado_pela_avaliacao`, com o motivo. A regra congelada atribui `inferido_nao_validado` ao estado transferido (C1 inconclusivo) e ao atualizado com cicatrizes (E1_C2 contrário).

### 3.6 Casos (gerados de `territorio/casos_territoriais.md`)

- **Morro da Oficina – Rua dos Ferroviários** (1.025 moradores): 4 cicatrizes e 2 fluxos de 2022 (16.879 m² na FCU, persistem à erosão de 10 m) e 6 relatos de 1949–2013. 77% da área está no zoneamento técnico. Quatro feições de 2022 drenam para a comunidade. O declive e o Idai recuperam 2 das 3 cicatrizes que tocam a FCU no quintil superior; o modelo atualizado com cicatrizes de outros bairros não recupera nenhuma. Água sem referência local.
- **Vila São Francisco A** (548): 5 cicatrizes (15.666 m², persistem a 10 m) e 7 relatos. Tem a maior área a montante entre os pilotos (1,51 km² até 1 km, 0,63 km² no quintil superior). O Idai recupera 3 de 5 cicatrizes; o declive, 1; a atualização, 0.
- **Sargento Boening** (584): 2 cicatrizes (2.010 m²) e 13 relatos dentro da FCU, o maior número entre os pilotos. Aqui a atualização com cicatrizes ajuda: 2 de 2 recuperadas, contra 1 de 2. Porém 38% das células têm algum atributo fora do domínio do doador: inferência em extrapolação.
- **Estrada do Paraíso A/B**: interseções com 2022 de 4 m² e 4.287 m² que quase desaparecem com 10 m de erosão — apoio fraco, possível efeito de borda. Há 14 e 12 relatos até 100 m.

O efeito da atualização muda de sinal entre comunidades. Isso reforça que o ganho agregado não é demonstração por comunidade.

## 4. Demonstrado, inconclusivo, não testável, implementado

- **Demonstrado (neste escopo):** aprendizado na origem; transferência acima do acaso para as cicatrizes de 2022; dano causado por rótulos locais de natureza incompatível, e proteção parcial dada pelo prior externo (C3, C4); preservação da origem quando a atualização usa cicatrizes; relação inversa entre o escore de água e as cicatrizes de encosta; aumento de cobertura de informação por comunidade.
- **Inconclusivo:** ganho da transferência sobre a regra de declive; ganho do boosting sobre a logística.
- **Contrário:** atualização com relatos históricos (C2) e, no limite, com cicatrizes de outros bairros (E1_C2).
- **Não testável aqui:**
  - suscetibilidade com ausências observadas em Petrópolis (não existem);
  - episódio independente de 2022;
  - separação ruptura/percurso no inventário local;
  - validação da água nas FCU;
  - cascata entre processos;
  - detecção pós-evento (sem imagem qualificada).
- **Implementado:** cadeia reprodutível; candidata territorial com sete componentes e estados; gate que recusa atualização reprovada; figuras e tabelas; verificação independente.

## 5. Bloqueios exatos e menor próximo experimento

1. **Ausências observadas em Petrópolis.** O próximo passo é um inventário com área vistoriada: por exemplo, o vetor das 611 cicatrizes e da área mapeada do DRM-RJ (2022), citado em PDF mas não disponível como vetor no acervo, ou uma amostragem própria de células inspecionadas em imagem, com revisão humana real. Isso converte presença–fundo em duas classes.
2. **Episódio independente.** O próximo passo é um inventário de outra chuva (por exemplo, a Região Serrana em 2011), testado com o modelo atual congelado.
3. **Dados de Ferreira et al. (2025) e Alcântara et al. (2023).** Estão disponíveis mediante solicitação. A solicitação exige contato da autora; ela não foi feita.
4. **USGS Hurricane Maria.** O acesso foi bloqueado por desafio anti-robô e não foi contornado. Seria um segundo doador de chuva tropical, para testar se a composição de doadores muda a transferência.
5. **Água nas FCU.** Continua sem referência local; o polígono de inundação de 2022 não intersecta FCU.

## 6. Custos e reprodução

- **Aquisição:** 2,0 MB (UFRJ, uma consulta WMS), 18,4 MB (Idai) e cerca de 57 MB de leitura parcial do FABDEM.
- **Processamento:**
  - terreno: 38 s;
  - referências: 9 s;
  - treino v1 (357 vetores de previsão, 350 ajustes locais e 2.000 réplicas de bootstrap): 148 s de parede;
  - emenda 1: 634 s, dos quais 552 s de bootstrap;
  - verificação: 42 s.
- **Execução:** todos os passos pesados rodaram pelo serializador, na tomada.
- **Comandos:** `local_runs/petropolis-aprendizado/v1/LEIAME.md`.
- **Incidentes:** registrados em `protocolo/emendas_e_incidentes.md`. São eles: DLL do SciPy sem PATH, antes de qualquer ajuste; tolerância float32 na verificação; conexão limitada a 1 km e critério de borda de 10 m, aplicados antes do texto e sem tocar métricas.

Texto para o artigo: `docs/ARTIGO_PETROPOLIS_ENCOSTAS_SECOES_2026_10_24.md`.
