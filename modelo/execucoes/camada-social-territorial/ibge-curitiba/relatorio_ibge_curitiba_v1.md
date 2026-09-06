# Piloto social real - IBGE Curitiba

Este artefato integra dados oficiais do IBGE/Censo 2022 para criar a
primeira camada social territorial real do REV-P em Curitiba. O piloto
usa indicadores relativos dentro do municipio e nao altera nenhum escore
fisico-hidrologico.

## Fontes

- `malha_pr`: PR_setores_CD2022.zip (26283813 bytes, sha256 `0b289e80d3015e495f6c9619848b75a9507492b5eddf10683fea2ee1b0d82c8e`)
- `basico_br`: Agregados_por_setores_basico_BR_20260520.zip (15418417 bytes, sha256 `ec04624286233d699ebe69c7b9625744a1cfdfc8352126f30245bdef5e9bdc63`)
- `renda_br`: Agregados_por_setores_renda_responsavel_BR_20260508_csv.zip (9370011 bytes, sha256 `141c83e7635674e7e2c941e55f98811e538fcbf2e1e1125b0aa7892c260f4a23`)
- `dicionario_basico`: dicionario_de_dados_agregados_por_setores_censitarios_20260520.xlsx (118369 bytes, sha256 `0b8aedece57f6125d785b6aa2234cfd587e92dfb6ce5ca6ace8c67f140831344`)
- `dicionario_renda`: dicionario_de_dados_renda_responsavel_20260508.xlsx (10197 bytes, sha256 `fea6e2b2439eeb167c6fa9c136ed7cb13ebcb06f4c875d634e3e2578c668a48d`)

## Resultado

- setores na malha de Curitiba: 3190
- setores com agregados basicos: 3190
- setores com renda do responsavel: 3178
- unidades sociais geradas: 3190
- setores marcados como favela/comunidade urbana na malha: 254

## Cobertura por dominio

- exposicao_populacional: 3190 setores
- condicao_socioeconomica: 3168 setores
- moradia_infraestrutura: 3190 setores
- lacuna_institucional: 3190 setores

## Piloto de nao-colapso com dado real

| caso | setor | fisica | social | prioridade |
|---|---|---|---|---|
| curitiba_ibge_alta_vulnerabilidade | 410690205100181 | alta | alta | prioridade_publica_forte |
| curitiba_ibge_baixa_vulnerabilidade | 410690205010506 | alta | baixa | prioridade_tecnica |

## Limites

- O recorte ainda usa bbox de setor, nao intersecao poligonal completa.
- Renda, densidade e media de moradores nao esgotam vulnerabilidade social.
- O indicador de imputacao censitaria nao mede subregistro de chamado de enchente.
- Curitiba continua sendo laboratorio metodologico, nao centro final do projeto.
