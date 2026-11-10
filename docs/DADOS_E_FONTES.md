# Dados, instrumentos e fontes

O artigo v22 identifica o papel de cada referência. As 35 entradas bibliográficas estão na [fonte do artigo](../artigo/fonte/artigo.tex) e no [índice de referências](../manifestos/metadados_bibliograficos.json).

| Grupo | Fonte ou instrumento | Papel no projeto |
|---|---|---|
| Ajuste para transferência ao Brasil | CEMS, UFO e Sen1Floods11 | 36.331 pontos em 59 cheias exteriores, reunidos em 43 unidades |
| Teste fora do país | Manchas MODIS do GFD a 250 m | Referência reservada para o núcleo congelado ajustado com observações de vários países, incluindo o Brasil |
| Água nos destinos brasileiros | Sentinel-2, cenas anterior e do evento | MNDWI com leitura válida e comparação temporal para água nova |
| Conferência do Guaíba | Mancha Planet/SkySat de 6 de maio de 2024 | Triangulação por outro sensor no recorte comum de observação |
| Radar | Sentinel-1 VV/VH | Percepção e aprendizado de água em tarefas próprias |
| Terreno | FABDEM ou outra família de MDT registrada | Derivação física a 30 m |
| Construído e solo | GHSL e SoilGrids | Atributos descritos no método |
| Água permanente | JRC | Exclusão e qualificação dos alvos |
| Recife | Atendimentos da Defesa Civil/SEDEC | Presença relatada por rua-bairro |
| Petrópolis | Inventário Idai, INPE/Charter e registros históricos | Aprendizado de encosta e referências com papéis distintos |
| Território e população | Malhas e Censo do IBGE | FCU, setores e contexto social |
| Chuva | Dados meteorológicos por data | Contexto temporal do episódio, fora do escore |

Os dados publicados em `evidencias/` são artefatos de resultados, protocolos e relatórios selecionados. Dados brutos, rasters e tabelas individuais volumosas permanecem como entradas externas para repetição integral dos experimentos. As rotinas de aquisição e preparação necessárias ao recorte foram mantidas no código selecionado.

As licenças e condições das fontes continuam aplicáveis. Incluir resultados agregados não substitui a autorização ou a aquisição dos dados de origem. A distinção entre presença, ausência observada, fundo e desconhecido faz parte da metodologia.
