# SUSC-20D -- Motor de inferencia local (MVP, Recife)

## Objetivo

Executa a Fase 3 do PLANO_ACAO_produto_v1.md: prova que o motor cientifico roda
ponta-a-ponta (score + intervalo + features + evidencia + limitacoes) antes de
gastar esforco em API/backend/interface.

## Validacao do motor antes do audit (sem isso, nao ha prova)

- Coeficientes Firth ponto-estimado: diferenca maxima vs SUSC-20C publicado =
  0.000046 (tolerancia 1e-3 -- passou).
- Bootstrap preditivo (N=1000, seed=20260723): diferenca maxima nos percentis
  2.5/97.5% vs SUSC-20C publicado = 0.0000.
- Evidencia DINO: indice carregado (109 pontos com patch DINO real anexado).

## Auditoria (n=269, 145 pos / 124 neg)

AUC in-sample deste motor: **0.7107**. LOO-AUC ja documentado em
SUSC-20C: **0.6781**. Nao sao o mesmo numero por desenho --
in-sample e otimista (mesmo dado usado pra treinar e avaliar); esta auditoria
nao substitui o LOO-AUC, so confirma que o motor (coeficientes fixos + score
por ponto) produz uma saida internamente coerente com o rotulo real.

CI (bootstrap preditivo) medio: largura media=0.2389,
mediana=0.2070.

## Limitacoes explicitas (sempre presentes na saida do motor)

- n=269 pontos completos (de 278 no v12), regiao=Recife apenas -- nao generaliza para outras cidades sem modelo proprio.
- LOO-AUC=0.6781 documentado (SUSC-20C): discriminacao modesta, nao e um detector de enchentes.
- DINO nao entra no score (Fase 1: ganho nao sobreviveu ao controle cluster-robusto por patch, ver revp_fase1_conclusao_dino_ab_test.md); aparece so como evidencia visual auxiliar quando disponivel.
- hand_m_dinf sem sinal robusto no bootstrap (48.7% de flips de sinal, ja documentado em SUSC-20C) -- tratar sua contribuicao individual com cautela.
- Este score nunca deve ser exibido como 'enchente detectada/confirmada' -- e uma suscetibilidade candidata com incerteza explicita.

## Arquivos

- `results/susc_20d_score_audit_v1.csv` -- score + CI + feature dominante + evidencia DINO por ponto
- `results/susc_20d_audit_summary.json` -- resumo da auditoria
