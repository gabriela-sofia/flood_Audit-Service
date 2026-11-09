# REV-P

**Reconhecimento de Evidências Verificáveis para Predição**

O REV-P é uma ferramenta de pesquisa em IA geoespacial que aprende relações físicas em cheias documentadas e examina sua transferência para territórios brasileiros com pouca referência local. A resposta reúne escore, explicação, fontes e condições de uso. Dados sociais contextualizam o território e permanecem separados do escore físico.

Esta organização corresponde ao **artigo v16, de 8 de novembro de 2026**. O artigo define os resultados e as contribuições deste pacote. As versões completa e com reserva do núcleo físico, o modelo de encosta e a rede de percepção conservam suas tarefas e avaliações próprias.

## Comece por aqui

| Para entender | Abra |
|---|---|
| A organização do repositório | [Guia de navegação](docs/REVISAO_DA_ORGANIZACAO.md) |
| O artigo atual | [PDF](artigo/REV-P_artigo_atual.pdf) e [fonte editável](artigo/fonte/artigo.tex) |
| O percurso do método | [Método](docs/METODO.md) e [admissão dos dados](evidencias/01_curadoria/README.md) |
| Cada resultado, seu código e sua evidência | [Matriz do artigo](docs/MATRIZ_ARTIGO.md) |
| O papel de cada pasta e rotina | [Organização do código](docs/CODIGO.md) |
| O que pode ser conferido com este pacote | [Reprodução e verificação](docs/REPRODUCAO.md) |
| Dados, instrumentos e referências | [Dados e fontes](docs/DADOS_E_FONTES.md) |
| A contribuição científica e territorial | [Contribuições](docs/CONTRIBUICOES.md) |

## Resultados organizados na ordem do artigo

| Tabela | Resultado ou desenho | Código principal | Evidência |
|---|---|---|---|
| I | Desenhos para água, relatos e encostas | [Fundamentação](codigo/scripts/fundamentacao_artigo/README.md) | [Método](docs/METODO.md) e [admissão dos dados](evidencias/01_curadoria/README.md) |
| II | Aprendizado físico, pesos e verificações | [Método físico](codigo/scripts/metodo/README.md) | [Aprendizado](evidencias/02_aprendizado/README.md) |
| III | Transferência para água nos destinos brasileiros | [Transferência](codigo/scripts/vulnerabilizados/README.md) | [Transferência](evidencias/03_transferencia/README.md) |
| IV | Relatos e explicação das ruas de Recife | [Recife](codigo/scripts/recife_aprendizado/README.md) | [Recife](evidencias/04_recife/README.md) |
| V | Modelo de encosta e referências de Petrópolis | [Petrópolis](codigo/scripts/petropolis_aprendizado/README.md) | [Petrópolis](evidencias/05_petropolis/README.md) |
| VI | Ganho local, retenção e memória | [Atualização](codigo/scripts/vulnerabilizados/README.md) | [Atualização](evidencias/06_atualizacao/README.md) |
| VII | Percepção da água e aprendizado nas redes | [SAR](codigo/scripts/sar/README.md) | [Percepção](evidencias/07_percepcao/README.md) |
| VIII | Fontes, encaminhamento e reprodução | [Recuperação](codigo/scripts/recuperacao/README.md) e [resposta](codigo/scripts/servico/README.md) | [Resposta](evidencias/08_resposta/README.md) |
| IX | Comunidades, alcance social e observabilidade | [Social](codigo/scripts/social/README.md) | [Alcance social](evidencias/09_social/README.md) |

## Estrutura

```text
artigo/       PDF atual, fonte LaTeX, figuras e referências
docs/         método, resultados, contribuições e orientações atuais
codigo/       rotinas selecionadas, dependências e testes originais
evidencias/   artefatos e relatórios ligados às tabelas do artigo
modelos/      pesos, padronização e estados dos modelos identificados
reproducao/   conferência do pacote e dos resultados registrados
testes/       testes da publicação e verificações portáveis
manifestos/   rastreabilidade, hashes e relação entre afirmações e arquivos
```

## Conferência rápida

Com Python 3.12:

```bash
python reproducao/verificar_pacote.py
python reproducao/conferir_resultados.py
```

Os dois comandos leem arquivos incluídos e não treinam modelos nem fazem downloads. Para os testes portáveis:

```bash
python -m pip install -r requirements-auditoria.txt
python -m pytest testes -q
```

As rotinas de pesquisa mantêm os caminhos de trabalho relativos à pasta `codigo/`. A [orientação de reprodução](docs/REPRODUCAO.md) distingue a conferência dos resultados publicados da repetição dos experimentos com dados externos.

## Como interpretar

AUC mede ordenação, e diferenças de AUC comparam desempenhos; não são porcentagens de acerto de um mapa nem redução de risco. Relato não substitui ausência observada e ausência de relato não comprova chão seco. Água e encosta têm modelos próprios. Resultados favoráveis, contrários, inconclusivos e com suporte insuficiente estão preservados nas evidências do artigo.

O ganho local é acompanhado de seu custo de retenção. Cobertura populacional não é contagem de atingidos. A revisão humana preparada continua registrada sem respostas. Os [manifestos](manifestos/README.md) identificam exatamente quais arquivos estão incluídos.
