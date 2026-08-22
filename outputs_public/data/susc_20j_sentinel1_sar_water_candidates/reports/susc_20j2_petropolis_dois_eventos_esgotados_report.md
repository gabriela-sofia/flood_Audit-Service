# SUSC-20J2 — Petrópolis: os 2 eventos reais de 2023-2026 esgotados, N=0 permanece

**Status**: RESULTADO REAL, EXAUSTIVO, NEGATIVO. Não é falta de tentativa: 4 métodos
independentes testados no evento mais forte, 2 métodos testados (e ambos estruturalmente
bloqueados) no segundo. Nenhum candidato novo adjudicável encontrado.

## Evento 1 — 05/04/2025 (Centro Histórico, Rio Quitandinha, S2ID forte)

Testado via Via B (óptico, `detect_water_candidates.py`) com **3 referências "antes"
independentes** e Via C (SAR, `detect_water_candidates_sar.py`) com 1 referência, todas
restritas ao corredor real dos rios Quitandinha/Piabanha (buffer OSM de 250m) e normalizadas
por z-score contra o deslocamento sistemático da própria cena:

| Referência "antes" | Sensor | Depois | % da cena (bruto) | Candidato no corredor real (z-score) |
|---|---|---|---:|---:|
| 16/02/2025 | S2C óptico | 09/04 | 3,43% | 0,000% |
| 16/02/2025 | S2C óptico | 07/04 | 1,25% | 0,010% |
| 28/11/2024 | S2A óptico | 09/04 | 3,45% | 0,000% |
| 03/03/2025 | S2B óptico | 09/04 | — | 0,000% |
| 03/03/2025 | S2B óptico | 07/04 | — | 0,010% |
| 03/04/2025 | S1A SAR (VV/VH) | 09/04 | — (3 clusters, 2.692px) | 0 clusters (corredor) |

Os 3 clusters SAR encontrados na cena inteira (10-11px cada, queda real de 4,5-4,6dB, dentro da
faixa publicada 3-8dB) ficam fora do corredor real — não perto do Centro/Quitandinha/Valparaíso.

**Conclusão real**: quatro métodos, quatro referências, mesmo resultado nulo no lugar
fisicamente esperado. Terreno de serra drena rápido; as janelas de imagem limpa disponíveis
(d+2 a d+4 do evento) provavelmente já pegam o pós-evento com a água escoada.

## Evento 2 — 22/03/2024 (Quitandinha+Piabanha, evento misto, S2ID forte mas já flagado como risco de mistura com deslizamento)

- **Via B (óptico)**: já documentado como sem par viável (nuvem 99-100% de d+1 a d+6, só abre em
  d+26 — água já teria escoado há muito).
- **Via C (SAR)**: testado agora. Único achado real e definitivo: **existe apenas 1 órbita
  Sentinel-1 cobrindo Petrópolis** (confirmado varrendo as 29 passagens de 2024 — todas na
  mesma órbita relativa, ciclo de 12 dias). Essa órbita tem uma **lacuna de cobertura real**
  sobre a porção sul do município — exatamente onde fica o núcleo urbano (Centro
  Histórico/Quitandinha) — confirmada em `2024-03-21`: 32% da cena sem dado, **100% de falha na
  janela urbana especificamente** (linhas 967-1250 de 1250, checado nos pixels reais, não só
  metadado). Como é sempre a mesma órbita, nenhuma outra data de 2024 resolveria isso.

**Conclusão real**: os dois métodos estão fechados pra esse evento, por motivos estruturais
diferentes e não contornáveis com mais tentativas de data.

## O que isso significa pro N de Petrópolis

N continua **0**. Isso não é uma falha de execução — é um resultado científico real: com as
duas janelas de evento mais bem documentadas de 2023-2026, e dois métodos de detecção
independentes testados com rigor (múltiplas referências, normalização estatística, restrição a
geometria real de drenagem), nenhum candidato pontual passou pelo filtro físico. Documentar isso
é mais correto que forçar um candidato fraco só para preencher N.

## Caminhos reais que sobram, não tentados ainda

1. **Outras janelas de evento**, menor confiança (médio, não forte) já listadas no registro
   `v20i_event_windows_2023_2026.csv` — não testadas ainda por imagem.
2. **Eventos históricos (1988, 2011)** mencionados em reportagem — exigiriam Landsat, sensor e
   protocolo totalmente diferentes, esforço maior.
3. Aceitar N=0 real e redirecionar esforço pra Curitiba (que tem COBRADE limpo e 10 janelas já
   ranqueadas) enquanto essa lacuna de Petrópolis fica documentada e aberta.

Nenhuma dessas é decidida aqui — é decisão da próxima rodada.
