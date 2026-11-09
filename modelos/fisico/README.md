# Núcleo físico

- [Versão externa completa](pesos_completa.json): `Q_sel_ext`, ajustada em 43 unidades externas.
- [Versão com reserva](pesos_com_reserva.json): `Qsep_sel_fis`, ajustada em 29 unidades.
- [Estado original do ajuste](estado_ajustado.pkl): arquivo serializado de origem, identificado por hash no inventário.

Os JSON incluem coeficientes, interações e padronizador. A transformação e a regra de escore estão em `codigo/scripts/metodo/` e `codigo/scripts/vulnerabilizados/`. O arquivo serializado conserva o estado de ajuste original e pode conter as variantes auxiliares dessa mesma execução; os dois JSON acima identificam as versões usadas nas afirmações atuais.
