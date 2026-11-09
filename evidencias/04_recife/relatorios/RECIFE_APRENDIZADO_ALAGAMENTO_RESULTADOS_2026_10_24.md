# Recife: alagamento urbano, transferência, aprendizado local e contexto urbano — resultados

24/10/2026. Execução de `docs/PROMPT_EXECUCAO_TREINO_RECIFE_ARTIGO_2026_10_24.md`.

- **Linhagem:** `local_runs/recife-aprendizado/v1/`, nova e aditiva. Protocolo `fc1465df…`; emendas 1 e 2 pós-teste.
- **Código e testes:** `scripts/recife_aprendizado/`; testes em `tests/test_recife_aprendizado_v1.py` (7 aprovados, junto com os 9 de Petrópolis).
- **Seções para o artigo:** `local_runs/recife-aprendizado/v1/texto_artigo_recife.md`.
- **Preservação:** as 115 fontes protegidas conferem com os hashes iniciais. O acervo de PROJETO foi lido sem alteração. Nada foi commitado, promovido ou enviado a terceiros. O modelo servido `recife_pluvial` não foi tocado.

## 1. Respostas diretas

- **O que foi ensinado, com que observações e que tarefa física?**
  - **Origem:** o núcleo Qsep_sel_fis foi ensinado fora do Brasil (29 unidades físicas da v10), com terreno de 30 m.
  - **Em Recife:** a tarefa foi ordenar ruas com relato de alagamento da Defesa Civil (2014–2024), geocodificado até a rua dentro do bairro (96 ruas, 38 bairros), contra as demais ruas do mesmo bairro. É presença–fundo: prioriza registro administrativo, não estima probabilidade física.
- **Qual conhecimento externo funcionou, e em qual população?** O Q, sem rótulo local, ordenou as ruas relatadas nos 19 bairros de teste com AUC dentro do bairro de 0,669, contra 0,595 do HAND. O contraste C1 foi de **+0,074 [0,012; 0,159], sustentado**. A retenção na origem foi preservada.
- **O que as observações locais acrescentaram fora do ajuste, e a qual custo?** Atualizar o Q com 10, 25 ou 48 ruas não mudou a ordenação (C2 +0,010, inconclusivo), e não houve custo de retenção.
- **O que PE3D e drenagem acrescentaram além do terreno de 30 m?**
  - **No protocolo:** C5 (drenagem + saída) deu +0,078 e C6 (PE3D) deu +0,062, ambos sustentados.
  - **Nas emendas pós-teste:** esse ganho é explicado pelo comprimento da rua. Ruas longas acumulam mais endereços e mais relatos, e a distância mínima ao canal diminui com a extensão. Controlando o comprimento, drenagem + saída ficam em +0,014 (inconclusivo) e o PE3D em −0,031 (contrário, n pequeno).
  - **Contribuição demonstrada:** nenhuma.
- **O que a ferramenta informa sobre uma comunidade concreta?** Para cada uma das 295 FCU, a resposta separa terreno, captação, rede, saída, água observada, inferência (com estado), contexto social, fontes, limites e próxima observação. Desse universo, 228 estão cobertas, 74 têm rua relatada e 33 têm resultado fora do ajuste (50.198 moradores, universo censitário).
- **Onde não respondeu, e qual observação permitiria o próximo teste?**
  - O episódio de 2022 não tem endereço utilizável.
  - Não há negativos observados.
  - Chuva subdiária e maré não foram adquiridas (o download do Cemaden tem CAPTCHA).
  - Cotas e seções são insuficientes para capacidade hidráulica e SWMM.
  - Os interiores de FCU não têm logradouro oficial.
  - **Próximo teste:** relatos datados com coordenada ou trecho por episódio, nível de canal ou maré no episódio, e uma microbacia com cotas e seções verificadas.

## 2. Referência e população

| Item | n |
|---|---:|
| Relatos SEDEC de água (2014–2024) | 1.648 |
| Geocodificados até a rua (nome exato 220, sem tipo 101, contenção única 84, aproximado 15) | 420 |
| Sem endereço (inclui todo 2022), não geocodificados, bairro não casado, canal | 728 / 436 / 41 / 22 |
| Ruas na população / com relato / bairros | 5.477 / 96 / 38 |
| Teste: bairros / ruas / positivas | 19 / 2.225 / 48 |
| Cobertura PE3D: ruas / positivas | 2.564 / 38 |

O geocodificador ganhou regras de limpeza de nomes antes do congelamento (auditoria de viabilidade, sem desempenho); um defeito de escape no regex foi corrigido. Detalhes em `protocolo/emendas_e_incidentes.md`.

## 3. Resultados (AUC dentro do bairro, teste)

| Braço | AUC | AP (prev. 2,2%) | Positivas no quintil superior do bairro |
|---|---:|---:|---:|
| Regra HAND 30 m | 0,595 | 0,028 | 31% |
| Q transferido | 0,669 | 0,038 | 42% |
| Só local, atributos Q (B = 48) | 0,684 | 0,041 | 35% |
| Q + atualização (B = 48) | 0,679 | 0,040 | 40% |
| Controle com prior permutado | 0,633 | 0,035 | 33% |
| Só local + drenagem/saída | 0,775 | 0,063 | 56% |
| Q atualizado + drenagem/saída | 0,756 | 0,055 | 52% |
| Só comprimento da rua (emenda 1) | 0,743 | 0,148 | — |
| Q atualizado + comprimento (emenda 1) | 0,819 | 0,070 | — |

No subconjunto PE3D, os valores foram:
- HAND: 0,457;
- regra "rua abaixo do entorno": 0,449;
- Q: 0,623;
- + drenagem/saída: 0,734;
- + PE3D: 0,796 (protocolo);
- com comprimento: 0,877 (Q + comprimento) contra 0,842 (+ PE3D).

**Retenção** nas 6 unidades de retenção, AUC dentro do evento: Q 0,733; A3 com λ = 25 entre 0,735 e 0,737; controle permutado com B = 10, 0,647.

**Coeficientes** (A4, padronizados):
- HAND: −0,82;
- distância ao canal (30 m): −0,56;
- construído: −0,32;
- log da distância ao canal: −0,37;
- drenagem: |coef| ≤ 0,16.

No PE3D, a posição relativa entra com −0,13 e a profundidade de depressão com +0,11. Esses parâmetros descrevem o estimador sob esta referência, não causas.

## 4. Demonstrado, inconclusivo, não testável, implementado

Detalhado em `tabelas/matriz_alegacao_evidencia_limite.csv`.

- **Demonstrado (escopo declarado):** transferência do Q acima do HAND para ruas relatadas; retenção preservada na atualização.
- **Inconclusivo:** ganho da atualização local; valor do conhecimento externo com ou sem blocos urbanos; contribuição de drenagem e saída sob controle de comprimento.
- **Não sustentado sob controle (pós-teste):** contribuição do PE3D. A regra PE3D simples ficou abaixo do acaso.
- **Não testável:** capacidade hidráulica; efeito de limpeza ou entupimento; alagamento por episódio (chuva, maré); SWMM; negativos observados.
- **Implementado:**
  - atributos PE3D, drenagem e saída por rua, com superfícies original e condicionada separadas;
  - grafo candidato com sensibilidade de tolerância;
  - candidata territorial para as 295 FCU com estados e alerta de comprimento;
  - 7 figuras, tabelas e verificação independente.

## 5. Casos

- **Chico Mendes – Linha Nova** (7.786 moradores, 256 BL/CG): uma rua relatada (14 relatos). Está em bairro de atualização, então a leitura é descritiva. O interior da comunidade não tem logradouro oficial, de modo que a unidade de rua não entra nos becos.
- **Encanta Moça** (8.878 moradores): sem rua relatada. Há depressão de até 6,8 m no PE3D e o canal fica a menos de 1 m.
- **Estrada dos Remédios – Rozenblit I** (1.487 moradores): uma rua relatada; canal a 0,4 m.
- **Soledade:** a Av. João de Barros, nos bairros vizinhos, fica nos percentis 0,90–0,99 com PE3D. O trecho do cruzamento publicado está num bairro sem relato geocodificado. As medidas publicadas (cinco valores, 2013–2014) continuam candidatas documentais, com a conferência visual pendente.
- **33 comunidades avaliadas:** na maioria, ganho aparente com A4; perdas em Vila Felicidade e Saramandaia. Várias compartilham a mesma avenida, e o ganho é o confundido pelo comprimento (Fig. 5).

## 6. Custos, reprodução e reconciliação

- **Custos:**
  - atributos: 203 s (PE3D 1 → 2 m, drenagem, Q);
  - treino (350 ajustes e 2.000 réplicas de bootstrap): 55 s;
  - emendas: 4 s;
  - verificação: 11 s.
- **Execução:** todos os passos pesados rodaram pelo serializador. Ele esperou corretamente por trabalho de outra sessão.
- **Comandos:** `local_runs/recife-aprendizado/v1/LEIAME.md`.
- **Reconciliação:** `tabelas/reconciliacao_historica.csv`. A v6–v10 (referência Sentinel-2) e o modelo servido continuam válidos nos seus alvos. Esta linhagem acrescenta uma tarefa de rua e o diagnóstico do viés de comprimento.
