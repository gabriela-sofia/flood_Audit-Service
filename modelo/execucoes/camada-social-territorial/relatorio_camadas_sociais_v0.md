# Camada social territorial - experimento v0

Este experimento cria a primeira prova de contrato da camada social do REV-P.
Ele nao valida vulnerabilidade social local; demonstra que a camada social
pode ser mantida separada do escore fisico e usada para prioridade publica,
lacuna de evidencia e comunicacao responsavel.

## Resultado

- contrato: `social_v0_review_only`
- casos demonstrativos: 5
- nao-colapso: `PASS`

## Invariante principal

A vulnerabilidade social nao aumenta nem reduz o escore fisico de enchente.
Ela altera a prioridade publica, a linguagem e a necessidade de vistoria ou
coleta de dado.

## Casos

| caso | fisica | social | prioridade | leitura |
|---|---|---|---|---|
| alta_fisica_alta_social | alta | alta | prioridade_publica_forte | suscetibilidade fisica alta coincide com alta vulnerabilidade/exposicao |
| alta_fisica_baixa_social | alta | baixa | prioridade_tecnica | suscetibilidade fisica alta; acao publica segue relevante mesmo sem alta vulnerabilidade social |
| fisica_recusada_alta_social | nao_identificavel | alta | prioridade_de_vistoria_e_coleta | territorio vulneravel sem inferencia fisica suficiente; nao declarar risco baixo |
| alta_fisica_sem_cobertura_social | alta | nao_identificavel | prioridade_tecnica_e_de_dado | suscetibilidade fisica alta sem cobertura social: a acao tecnica ja se justifica, e nao se sabe quem esta exposto; coletar dado social nao adia a vistoria |
| media_fisica_social_intermediaria | media | media | prioridade_rotineira | sem coincidencia demonstrativa de alta suscetibilidade e alta vulnerabilidade |

## Limitacoes

- unidades sociais demonstrativas nao representam setores censitarios reais
- geometria v0 usa bbox e deve ser substituida por intersecao de poligonos
- fontes oficiais ainda precisam de aquisicao, licenca, versao e schema
- prioridade publica nao e previsao individual nem probabilidade de evento

## Proximos experimentos

- adquirir setores censitarios e tipologias territoriais oficiais por municipio piloto
- cruzar AOIs reais do SVC-03 com setores oficiais
- medir lacuna de registro comparando vulnerabilidade alta e densidade de chamados
- rodar pesquisa de usuario com modo gestor publico e modo publico
