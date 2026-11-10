# Núcleos físicos do artigo v22

- [Versão externa completa](pesos_completa.json): `Q_sel_ext`, ajustada em 43 unidades externas.
- [Versão com reserva](pesos_com_reserva.json): `Qsep_sel_fis`, ajustada em 29 unidades.
- [Estado original do ajuste](estado_ajustado.pkl): arquivo serializado de origem, identificado por hash no inventário.

Os JSON incluem coeficientes, interações e padronizador. A transformação e a regra de escore estão em `codigo/scripts/metodo/` e `codigo/scripts/vulnerabilizados/`. O arquivo serializado conserva o estado de ajuste original e pode conter as variantes auxiliares dessa mesma execução; os dois JSON acima identificam as versões usadas nas afirmações atuais.

Os modelos exportados aqui são as versões ajustadas no exterior. A completa usa 43 unidades de 59 cheias e a com reserva ajusta em 29. As 14 restantes separam estimativa de erro, calibração da recusa e retenção. O núcleo congelado da Tabela I tem outro conjunto de ajuste, com 91.446 observações que incluem o Brasil. Seu resultado está registrado no [teste externo](../../evidencias/02_aprendizado/mt11-fechamento/v2/reparo_grafo_3b.json).
