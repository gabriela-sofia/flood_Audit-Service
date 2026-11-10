# Percepção de água e pré-treino

- A U-Net sem pré-treino da Tabela VII tem seu estado e manifesto no [checkpoint de percepção](unet_sar.pt), na [ficha](ficha_modelo.json) e no [manifesto](manifesto_original.json).
- A reconstrução da Fig. 3 usa o [estado de pré-treino mascarado](pretreino_estado_rodada_geral_20260913.pt) e a [normalização registrada](encoder_pretreinado_rodada_geral_20260913_manifesto.json).
- O ganho em AP com 1, 3 ou 12 recortes foi medido no [ensaio de oito eventos](../../evidencias/07_percepcao/pretreino-sar/v2/analise.json). O efeito de LwF está na [reagregação em suporte comum](../../evidencias/07_percepcao/planejamento-tese/v2/continuidade/diagnostico.json).
