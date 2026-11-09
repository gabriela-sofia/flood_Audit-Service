# Dados, instrumentos e fontes

O artigo conserva a origem e o papel das referências. A bibliografia completa está na [fonte do artigo](../artigo/fonte/artigo.tex); seus [metadados conferidos](../manifestos/metadados_bibliograficos.json) acompanham a preparação.

| Grupo | Fonte ou instrumento | Papel no projeto |
|---|---|---|
| Referências externas de cheia | CEMS, Sen1Floods11, UFO e GFD | Ensinar e avaliar, conforme o desenho de cada instrumento |
| Água nos destinos brasileiros | Sentinel-2, cenas anterior e do evento | Referência óptica qualificada |
| Radar | Sentinel-1 VV/VH | Percepção e aprendizado de água em tarefas próprias |
| Terreno | FABDEM ou outra família de MDT registrada | Derivação física a 30 m |
| Construído e solo | GHSL e SoilGrids | Atributos descritos no método |
| Água permanente | JRC | Exclusão e qualificação dos alvos |
| Recife | Atendimentos da Defesa Civil/SEDEC | Presença relatada por rua-bairro |
| Petrópolis | Inventário Idai, INPE/Charter e registros históricos | Aprendizado de encosta e referências com papéis distintos |
| Território e população | Malhas e Censo do IBGE | FCU, setores e contexto social |
| Chuva dos relatos | ERA5 diário | Gatilho temporal no mesmo lugar |

Os dados publicados em `evidencias/` são artefatos de resultados, protocolos e relatórios selecionados. Dados brutos, rasters e tabelas individuais volumosas permanecem como entradas externas para repetição integral dos experimentos. As rotinas de aquisição e preparação necessárias ao recorte foram mantidas no código selecionado.

As licenças e condições das fontes continuam aplicáveis. Incluir resultados agregados não substitui a autorização ou a aquisição dos dados de origem. A distinção entre presença, ausência observada, fundo e desconhecido faz parte da metodologia.
