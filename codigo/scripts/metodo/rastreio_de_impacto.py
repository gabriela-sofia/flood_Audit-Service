"""Rastreio de impacto: o que ficou desatualizado quando algo a montante mudou.

-------------------------------------------------------------------------
POR QUE ISTO EXISTE
-------------------------------------------------------------------------

A cadeia vai da evidencia ao claim:

    tabela unica -> contrato / admissao -> dataset de treino
                 -> celulas treinadas -> sinais -> relacoes -> conhecimento

Ate 03/10/2026 cada elo guardava algum hash, cada um do seu jeito, e nenhum
dizia aos de baixo que tinham envelhecido. O caso medido: o `ds04` passou a
datar 15.571 linhas do CEMS e, sem rastreio, `RELACOES_SUSTENTADAS.md` e
`CONHECIMENTO_DO_FENOMENO.md` continuariam afirmando sobre a tabela antiga
sem nenhum aviso. Auditoria em markdown nao basta: o ponto de entrada que
afirma precisa FALHAR quando o que ele le mudou desde a ultima revalidacao.

-------------------------------------------------------------------------
COMO FUNCIONA
-------------------------------------------------------------------------

Cada no declara: seu codigo, os nos de que depende, arquivos de dado bruto
que le direto, e suas saidas. Ao terminar, o gerador CARIMBA o no: grava o
hash do codigo, das saidas dos nos consumidos e das proprias saidas. O
carimbo vai para um historico append-only -- nada e apagado.

Um no fica:

  atual          carimbo existe e todos os hashes batem
  sem_carimbo    nunca foi revalidado sob este rastreio
  desatualizado  codigo, dado bruto, saida de dependencia ou saida propria
                 mudou depois do carimbo (os motivos vao listados)
  herdado        algum no a montante nao esta atual: o hash pode ate bater,
                 mas o que ele consumiu ja nao e o estado aceito

`exigir_atual(nos, quem)` e o que os pontos de entrada chamam: falha com a
lista do que envelheceu. `carimbar` recusa carimbar no cuja dependencia nao
esta atual -- revalida-se de cima para baixo.

Uso:
    python scripts/metodo/rastreio_de_impacto.py              # estado + doc
    python scripts/metodo/rastreio_de_impacto.py --carimbar tabela_unica
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
S = REPO / "scripts"
SAIDA = RUNS / "rastreio-de-impacto"
HISTORICO = SAIDA / "carimbos.jsonl"
DOC = REPO / "docs" / "RASTREIO_DE_IMPACTO.md"


def _ds06_atual() -> list[Path]:
    base = RUNS / "ds-06-dataset-de-treino"
    versoes = sorted((p.parent for p in base.glob("v*/manifesto.json")),
                     key=lambda d: int(d.name[1:]))
    return [versoes[-1] / "pontos.parquet", versoes[-1] / "manifesto.json"] if versoes else []


def _instrumentos_do_contrato() -> str:
    """Impressao so do que ds06 e admissao leem do contrato: instrumentos e escala x relevo.

    Acrescentar campo de relatorio ao contrato nao muda o dataset; mudar o
    mecanismo, o negativo ou o relevo resolvido de um instrumento muda.
    """
    sys.path.insert(0, str(S / "metodo"))
    sys.path.insert(0, str(S / "suscetibilidade"))
    sys.path.insert(0, str(S / "servico"))
    import contrato_do_fenomeno as cf
    import ds03_esquema_alvo as ds03
    from proveniencia import sha256_json
    return sha256_json({"instrumentos": cf.INSTRUMENTOS, "relevos": ds03.RELEVOS_RESOLVIDOS,
                        "papeis": ds03.PAPEL_EVIDENCIAL, "datas": ds03.PRECISAO_DATA})


def _modulos():
    for pasta in ("metodo", "suscetibilidade", "servico", "geo"):
        if str(S / pasta) not in sys.path:
            sys.path.insert(0, str(S / pasta))


# --------------------------------------------------------------------------
# IMPRESSOES DE REGRA (07/10/2026). Ate aqui o rastreio via so hash de ARQUIVO:
# mudar o corte de agua recorrente, a populacao de uma comparacao, a regra de
# identidade de evento ou a atribuicao de dobras nao envelhecia nada a jusante
# se o arquivo do consumidor nao mudasse. Cada regra abaixo e o OBJETO que
# define a semantica, hasheado onde quer que ele more. Mudar texto de
# docstring nao envelhece; mudar a regra envelhece.
# --------------------------------------------------------------------------
def regra_de_rotulo() -> str:
    """O que conta como positivo, negativo, permanente, recorrente e desconhecido."""
    _modulos()
    import ds03_esquema_alvo as ds03
    import verdade_observacional as vo
    from agua_permanente_jrc import LIMIAR_CALIBRADO
    from proveniencia import sha256_json
    return sha256_json({"papeis": ds03.PAPEL_EVIDENCIAL, "regimes": vo.REGIMES_HIDRICOS,
                        "alvos": sorted(vo.ALVOS), "cortes": sorted(vo.CORTES), "jrc_permanente": LIMIAR_CALIBRADO,
                        "epoca_jrc": vo.JRC["epoca"],
                        "mascara_gfd": vo.DOMINIO_DO_ROTULO["gfd"]["mascara_de_terreno"]})


def regra_de_populacao() -> str:
    """Que colunas definem a populacao de cada consumidor que ajusta ou mede."""
    _modulos()
    import generalizacao_por_reservas as g
    import inspect
    from proveniencia import sha256_json
    padrao = inspect.signature(g.carregar).parameters["vetor_da_populacao"].default
    return sha256_json({"terreno": g.TERRENO, "todas": g.TODAS, "fontes": g.FONTES,
                        "padrao_de_carregar": list(padrao),
                        "v2": g.pre_registro_v2()["populacao"]})


def regra_de_identidade() -> str:
    """Como dois registros viram o mesmo evento e como se compara lugar entre fontes."""
    _modulos()
    import verdade_observacional as vo
    from proveniencia import sha256_json
    return sha256_json({"folga_dias": vo.FOLGA_DIAS, "folga_graus": vo.FOLGA_GRAUS,
                        "prioridade": vo.PRIORIDADE, "chave_espacial": "tile 1 grau por coordenada"})


def regra_de_particao() -> str:
    """Dobras, reservas e o conjunto conhecido: o que pode ser treino, teste ou confirmacao."""
    _modulos()
    import generalizacao_por_reservas as g
    from proveniencia import sha256_json
    conhecidos = RUNS / "reservas-examinadas" / "conhecidos_2026_10_03.json"
    return sha256_json({"v1": g.pre_registro(), "v2": g.pre_registro_v2(),
                        "conhecidos": _hash(conhecidos)})


REGRAS = {"regra_de_rotulo": regra_de_rotulo, "regra_de_populacao": regra_de_populacao,
          "regra_de_identidade": regra_de_identidade, "regra_de_particao": regra_de_particao}


# Saida pode ser lista fixa ou funcao (dataset versionado: a versao vigente).
# `impressoes`: a parte de um modulo que o no le de fato, quando o arquivo
# inteiro seria dependencia larga demais.
NOS: dict[str, dict] = {
    "tabela_unica": {
        "codigo": [S / "suscetibilidade/ds03_esquema_alvo.py", S / "suscetibilidade/ds04_reduzir_por_fonte.py",
                   S / "suscetibilidade/ds05_admissao_consolidacao.py"],
        "dados": [RUNS / "cems-datas-de-imagem/datas.json", RUNS / "gfd-brasil/pontos_gfd_com_terreno.csv"],
        "depende": [],
        "saidas": [RUNS / "ds-05-tabela-unica/tabela_unica_v2.parquet"],
    },
    "contrato": {
        "codigo": [S / "metodo/contrato_do_fenomeno.py", S / "suscetibilidade/ds03_esquema_alvo.py"],
        "dados": [], "depende": ["tabela_unica"],
        "saidas": [RUNS / "contrato-do-fenomeno/contrato.json"],
    },
    "admissao": {
        "codigo": [S / "metodo/admissao_cientifica.py"],
        "impressoes": {"instrumentos_do_contrato": _instrumentos_do_contrato},
        "dados": [], "depende": ["tabela_unica"],
        "saidas": [RUNS / "admissao-cientifica/retrato.json"],
    },
    "dataset_de_treino": {
        "codigo": [S / "suscetibilidade/ds06_dataset_de_treino.py", S / "geo/cob01_cobertura_e_solo.py"],
        "impressoes": {"instrumentos_do_contrato": _instrumentos_do_contrato},
        "dados": [], "depende": ["tabela_unica"],
        "saidas": _ds06_atual,
    },
    "celulas_de_contexto": {
        "codigo": [S / "metodo/celula_para_treino.py", S / "metodo/ciclo_de_contexto.py",
                   S / "metodo/validacao_entre_contextos.py", S / "metodo/admissao_cientifica.py",
                   S / "servico/svc01_construir_modelos_servidos.py"],
        "dados": [], "depende": ["tabela_unica", "admissao"],
        "saidas": [RUNS / "contexto-treinado/resumo.json",
                   RUNS / "contexto-treinado/sintese_entre_contextos.json"],
    },
    "sinais_marginais": {
        "codigo": [S / "metodo/sinais_marginais.py"],
        "dados": [], "depende": ["dataset_de_treino"],
        "saidas": [RUNS / "sinais-marginais/sinais.json"],
    },
    "relacoes": {
        "codigo": [S / "metodo/relacoes_sustentadas.py"],
        "dados": [], "depende": ["contrato", "sinais_marginais", "celulas_de_contexto"],
        "saidas": [RUNS / "relacoes-sustentadas/registro.json"],
    },
    "conhecimento": {
        "codigo": [S / "metodo/conhecimento_do_fenomeno.py"],
        "dados": [], "depende": ["celulas_de_contexto"],
        "saidas": [RUNS / "contexto-treinado/conhecimento_do_fenomeno.json"],
    },
    "curva_de_aprendizado": {
        "codigo": [S / "metodo/curva_de_aprendizado.py", S / "metodo/admissao_cientifica.py"],
        "dados": [], "depende": ["dataset_de_treino"],
        "saidas": [RUNS / "curva-de-aprendizado/curva.json"],
    },
    # O piloto do Taquari (ctx01 rodado sozinho) foi substituido pelo lote:
    # o ctx01 segue como biblioteca do ctx03 e nao e mais no proprio.
    "contexto_hidrologico": {
        "codigo": [S / "geo/ctx01_contexto_hidrologico_do_evento.py", S / "geo/ctx02_triagem_eventos_estacoes.py",
                   S / "geo/ctx03_contexto_em_lote.py"],
        "dados": [RUNS / "ctx-02-triagem/triagem.json"], "depende": ["dataset_de_treino"],
        "saidas": [RUNS / "ctx-03-contexto-em-lote/contexto_em_lote.json",
                   RUNS / "ctx-03-contexto-em-lote/pontos_ligados.parquet"],
    },
    # Modelos servidos: ajustados antes da admissao central, sobre a tabela de
    # entao. O svc01 nao e reexecutado aqui -- reajustar muda o produto -- e o
    # no fica sem carimbo ate a decisao: o rastreio mostra que eles NAO foram
    # revalidados contra a tabela atual.
    "modelos_servidos": {
        "codigo": [S / "servico/svc01_construir_modelos_servidos.py"],
        "dados": [], "depende": ["tabela_unica", "admissao"],
        "saidas": lambda: sorted(p for p in (RUNS / "svc-01-modelos").glob("*.json")),
    },
    "afirmacao_dos_servidos": {
        "codigo": [S / "metodo/afirmacao_dos_servidos.py"],
        "impressoes": {"instrumentos_do_contrato": _instrumentos_do_contrato},
        "dados": [RUNS / "svc-01-modelos/indice.json"], "depende": [],
        "saidas": [RUNS / "afirmacao-dos-servidos/afirmacao.json"],
    },
    # Nos de 07/10/2026: as derivacoes da auditoria pre-treino, com impressoes de regra.
    "verdade_observacional": {
        "codigo": [S / "metodo/verdade_observacional.py"],
        "impressoes": {"regra_de_rotulo": regra_de_rotulo, "regra_de_identidade": regra_de_identidade},
        "dados": [RUNS / "gfd-brasil/pontos_gfd_com_terreno.csv",
                  RUNS / "serra-02-pacote-de-adjudicacao/pacote.json",
                  RUNS / "gfd-brasil/lote_que_tocam_o_brasil.json"],
        "depende": ["dataset_de_treino"],
        "saidas": lambda: [RUNS / "verdade-observacional/identidade.json"]
        + sorted((RUNS / "verdade-observacional").glob("anotacoes_ds06_v*.json")),
    },
    "generalizacao_v2": {
        "codigo": [S / "metodo/generalizacao_por_reservas.py", S / "metodo/admissao_cientifica.py"],
        "impressoes": {"regra_de_populacao": regra_de_populacao, "regra_de_particao": regra_de_particao,
                       "regra_de_identidade": regra_de_identidade},
        "dados": [], "depende": ["dataset_de_treino", "verdade_observacional"],
        "saidas": [RUNS / "generalizacao-por-reservas/v2_populacao_da_tarefa/resultado.json"],
    },
    "e2c_identidade_canonica": {
        "codigo": [S / "metodo/ci_e2_sensibilidade_negativo.py"],
        "impressoes": {"regra_de_identidade": regra_de_identidade},
        "dados": [], "depende": ["dataset_de_treino", "verdade_observacional"],
        "saidas": [RUNS / "campanha-integrada-20261003/E2c/card.json"],
    },
    "recusa_por_dominio_v2": {
        "codigo": [S / "metodo/recusa_por_dominio.py", S / "metodo/generalizacao_por_reservas.py"],
        "impressoes": {"regra_de_populacao": regra_de_populacao, "regra_de_identidade": regra_de_identidade},
        "dados": [], "depende": ["dataset_de_treino", "verdade_observacional"],
        "saidas": [RUNS / "recusa-por-dominio/v2_populacao_da_tarefa/resultado.json"],
    },
    "politica_sob_orcamento_v2": {
        "codigo": [S / "metodo/politica_sob_orcamento.py", S / "metodo/generalizacao_por_reservas.py"],
        "impressoes": {"regra_de_populacao": regra_de_populacao, "regra_de_identidade": regra_de_identidade,
                       "regra_de_particao": regra_de_particao},
        "dados": [], "depende": ["dataset_de_treino", "verdade_observacional"],
        "saidas": [RUNS / "politica-sob-orcamento/v2_populacao_da_tarefa/resultado.json"],
    },
    "posicao_topografica": {
        "codigo": [S / "suscetibilidade/posicao_topografica.py"],
        "dados": [], "depende": ["dataset_de_treino"],
        "saidas": [RUNS / "posicao-topografica/v1/manifesto.json", RUNS / "posicao-topografica/v1/pontos.parquet"],
    },
    "ms1_posicao_topografica": {
        "codigo": [S / "metodo/ms1_posicao_topografica.py", S / "metodo/generalizacao_por_reservas.py"],
        "impressoes": {"regra_de_populacao": regra_de_populacao, "regra_de_identidade": regra_de_identidade},
        "dados": [RUNS / "ms1-posicao-topografica/pre_registro.json"],
        "depende": ["dataset_de_treino", "verdade_observacional", "posicao_topografica"],
        "saidas": [RUNS / "ms1-posicao-topografica/resultado.json"],
    },
    "ms2_quadro_do_negativo": {
        "codigo": [S / "metodo/ms2_quadro_do_negativo.py", S / "metodo/ms1_posicao_topografica.py"],
        "impressoes": {"regra_de_populacao": regra_de_populacao, "regra_de_identidade": regra_de_identidade},
        "dados": [RUNS / "ms2-quadro-do-negativo/pre_registro.json"],
        "depende": ["dataset_de_treino", "verdade_observacional", "posicao_topografica"],
        "saidas": [RUNS / "ms2-quadro-do-negativo/resultado.json"],
    },
    "ms3_decomposicao_do_quadro": {
        "codigo": [S / "metodo/ms3_decomposicao_do_quadro.py", S / "metodo/ms2_quadro_do_negativo.py"],
        "impressoes": {"regra_de_populacao": regra_de_populacao, "regra_de_identidade": regra_de_identidade},
        "dados": [RUNS / "ms3-decomposicao-do-quadro/pre_registro.json"],
        "depende": ["dataset_de_treino", "verdade_observacional", "posicao_topografica"],
        "saidas": [RUNS / "ms3-decomposicao-do-quadro/resultado.json"],
    },
    "auditoria_elevacao_local": {
        "codigo": [S / "metodo/auditoria_elevacao_local.py", S / "suscetibilidade/ds06_dataset_de_treino.py"],
        "impressoes": {"regra_de_populacao": regra_de_populacao},
        "dados": [],
        "depende": ["dataset_de_treino", "posicao_no_ponto"],
        "saidas": [RUNS / "auditoria-elevacao-local/resultado.json"],
    },
    "ms4_bracos_de_desenvolvimento": {
        "codigo": [S / "metodo/ms4_bracos_de_desenvolvimento.py"],
        "impressoes": {"regra_de_particao": regra_de_particao},
        "dados": [RUNS / "ms4-bracos-de-desenvolvimento/pre_registro.json"],
        "depende": ["dataset_de_treino", "posicao_no_ponto"],
        "saidas": [RUNS / "ms4-bracos-de-desenvolvimento/resultado.json",
                   RUNS / "ms4-bracos-de-desenvolvimento/predicoes_fora_da_dobra.parquet"],
    },
    "ms5_recusa_decomposta": {
        "codigo": [S / "metodo/ms5_recusa_decomposta.py"],
        "dados": [RUNS / "ms5-recusa-decomposta/pre_registro.json"],
        "depende": ["ms4_bracos_de_desenvolvimento"],
        "saidas": [RUNS / "ms5-recusa-decomposta/resultado.json"],
    },
    "tc01_territorio_consultavel": {
        "codigo": [S / "geo/tc01_territorio_consultavel.py", S / "suscetibilidade/posicao_no_ponto.py"],
        "dados": [],
        "depende": ["dataset_de_treino"],
        "saidas": [RUNS / "tc01-territorio-consultavel/desenvolvimento/manifesto.json"],
    },
    "br03_preparo_sem_rotulo": {
        "codigo": [S / "geo/br03_preparo_sem_rotulo.py", S / "metodo/verdade_observacional.py"],
        "impressoes": {"regra_de_identidade": regra_de_identidade},
        "dados": [RUNS / "rfd-quarentena-brasil/selecao.json"],
        "depende": ["dataset_de_treino"],
        "saidas": [RUNS / "br-preparo-sem-rotulo/resumo.json"],
    },
    "ms6_quadro_de_treino_para_consulta": {
        "codigo": [S / "metodo/ms6_quadro_de_treino_para_consulta.py"],
        "impressoes": {"regra_de_populacao": regra_de_populacao, "regra_de_particao": regra_de_particao},
        "dados": [RUNS / "ms6-quadro-de-treino-para-consulta/pre_registro.json"],
        "depende": ["dataset_de_treino", "posicao_no_ponto", "tc01_territorio_consultavel"],
        "saidas": [RUNS / "ms6-quadro-de-treino-para-consulta/resultado.json"],
    },
    "posicao_no_ponto": {
        "codigo": [S / "suscetibilidade/posicao_no_ponto.py"],
        "dados": [],
        "depende": ["dataset_de_treino"],
        "saidas": [RUNS / "posicao-topografica/v2/manifesto.json", RUNS / "posicao-topografica/v2/pontos.parquet",
                   RUNS / "posicao-topografica/v2/unidades.parquet"],
    },
    "queda_entre_fontes": {
        "codigo": [S / "metodo/diagnostico_queda_entre_fontes.py", S / "metodo/ms1_posicao_topografica.py"],
        "impressoes": {"regra_de_populacao": regra_de_populacao, "regra_de_identidade": regra_de_identidade},
        "dados": [],
        "depende": ["dataset_de_treino", "verdade_observacional", "posicao_topografica"],
        "saidas": [RUNS / "queda-entre-fontes/resultado.json"],
    },
    "terreno_versus_contexto": {
        "codigo": [S / "metodo/ctx04_terreno_versus_contexto.py", S / "metodo/admissao_cientifica.py"],
        "dados": [], "depende": ["dataset_de_treino", "contexto_hidrologico"],
        "saidas": [RUNS / "ctx-04-terreno-versus-contexto/card.json"],
    },
    "acervo_espacial": {
        "codigo": [S / "geo/acervo_espacial.py", S / "suscetibilidade/posicao_no_ponto.py"],
        "dados": [], "depende": [],
        "saidas": [RUNS / "acervo-espacial/manifesto.json"],
    },
    "contexto_de_desenvolvimento": {
        "codigo": [S / "metodo/contexto_de_desenvolvimento.py", S / "geo/acervo_espacial.py",
                   S / "suscetibilidade/posicao_no_ponto.py"],
        "impressoes": {"regra_de_populacao": regra_de_populacao},
        "dados": sorted(RUNS.glob("tc01-curado/v1/*/manifesto.json"))[-1:]
        + sorted(RUNS.glob("pacote-de-treino/*/manifesto.json"))[-1:],
        "depende": ["acervo_espacial"],
        "saidas": lambda: sorted(RUNS.glob("contexto-desenvolvimento/v1/*/manifesto.json"),
                                 key=lambda p: p.stat().st_mtime)[-1:],
    },
    "bacias_hydrobasins": {
        "codigo": [S / "geo/bacias_hydrobasins.py"],
        "dados": [RUNS / "hydrobasins/v1c/bruto/hybas_sa_lev01-12_v1c.zip",
                  RUNS / "hydrobasins/v1c/bruto_eu/hybas_eu_lev01-12_v1c.zip"],
        "depende": [],
        "saidas": [RUNS / "hydrobasins/v1c/tabelas/manifesto.json",
                   RUNS / "hydrobasins/v1c/tabelas_eu/manifesto.json"],
    },
    "populacao_alvo_v2": {
        "codigo": [S / "metodo/contexto_extensao_v2.py"],
        "impressoes": {"regra_de_rotulo": regra_de_rotulo},
        "dados": [],
        "depende": ["contexto_extensao_v2"],
        "saidas": lambda: sorted(RUNS.glob("contexto-desenvolvimento/v2/*/populacao_alvo_v2.json"))[-1:],
    },
    "candidatos_independentes": {
        "codigo": [S / "metodo/candidatos_independentes.py"],
        "dados": [RUNS / "br-preparo-sem-rotulo/eventos.csv"],
        "depende": ["contexto_extensao_v2", "bacias_hydrobasins"],
        "saidas": [RUNS / "candidatos-independentes/manifesto.json"],
    },
    "grade_canonica_piloto": {
        "codigo": [S / "terreno/grade_canonica.py", S / "terreno/derivar_aoi_nova.py"],
        "dados": [RUNS / "grade-canonica-piloto/pre_registro.json"],
        "depende": [],
        "saidas": [RUNS / "grade-canonica-piloto/derivacao.json", RUNS / "grade-canonica-piloto/comparacao.json"],
    },
    "grade_canonica_v2": {
        "codigo": [S / "terreno/grade_canonica.py", S / "terreno/derivar_aoi_nova.py"],
        "dados": [RUNS / "grade-canonica-piloto/v2/pre_registro.json"],
        "depende": [],
        "saidas": [RUNS / "grade-canonica-piloto/v2/comparacao.json",
                   RUNS / "grade-canonica-piloto/v2/diagnosticos.json",
                   RUNS / "grade-canonica-piloto/v2/decisao.json"],
    },
    "pacote_candidato_v2": {
        "codigo": [S / "metodo/contexto_extensao_v2.py", S / "metodo/pacote_de_treino.py"],
        "dados": [],
        "depende": ["populacao_alvo_v2"],
        "saidas": lambda: sorted(RUNS.glob("pacote-de-treino-candidato/*/*/manifesto.json"))[-1:],
    },
    "aprendizado_pacote_v2": {
        "codigo": [S / "metodo/aprendizado_pacote_v2.py"],
        "dados": [RUNS / "aprendizado-pacote-v2/ap-v2-01/pre_registro.json"],
        "depende": ["pacote_candidato_v2"],
        "saidas": [RUNS / "aprendizado-pacote-v2/ap-v2-01/resultado.json"],
    },
    "instrumento_brasileiro": {
        "codigo": [S / "geo/obs04_instrumento_brasileiro.py"],
        "dados": [RUNS / "instrumento-brasileiro/v1/pre_registro.json",
                  RUNS / "instrumento-brasileiro/v2-dswx-hls/pre_registro.json",
                  RUNS / "instrumento-brasileiro/ib-03a/pre_registro.json",
                  RUNS / "instrumento-brasileiro/ib-03a/anotacao_congelada.json"],
        "depende": [],
        "saidas": [RUNS / "instrumento-brasileiro/v1/resultado.json",
                   RUNS / "instrumento-brasileiro/v2-dswx-hls/resultado.json",
                   RUNS / "instrumento-brasileiro/ib-03a/resultado.json"],
    },
    "elegibilidade_por_pergunta": {
        "codigo": [S / "metodo/candidatos_independentes.py"],
        "dados": [],
        "depende": ["candidatos_independentes", "instrumento_brasileiro"],
        "saidas": [RUNS / "candidatos-independentes/elegibilidade_por_pergunta.json"],
    },
    "plano_de_disco": {
        "codigo": [S / "manutencao/plano_de_disco.py", S / "manutencao/medir_compressao.py"],
        "dados": [],
        "depende": [],
        "saidas": [RUNS / "disco-plano/compressao_medida.json", RUNS / "disco-plano/plano.json"],
    },
    "dominio_aquatico": {
        "codigo": [S / "geo/dominio_aquatico.py"],
        "dados": [RUNS / "dominio-aquatico/piloto/pre_registro.json"],
        "depende": [],
        "saidas": [RUNS / "dominio-aquatico/piloto/resultado.json", RUNS / "dominio-aquatico/tc01/manifesto.json"],
    },
    "contexto_extensao_v2": {
        "codigo": [S / "metodo/contexto_extensao_v2.py", S / "geo/bacias_hydrobasins.py"],
        "dados": [],
        "depende": ["contexto_de_desenvolvimento", "bacias_hydrobasins", "dominio_aquatico"],
        "saidas": lambda: sorted(RUNS.glob("contexto-desenvolvimento/v2/*/manifesto.json"),
                                 key=lambda p: p.stat().st_mtime)[-1:],
    },
    "tc_ctx01_v2": {
        "codigo": [S / "metodo/tc_ctx01.py"],
        "dados": [RUNS / "tc-ctx01/v2/pre_registro.json"],
        "depende": ["contexto_extensao_v2"],
        "saidas": [RUNS / "tc-ctx01/v2/resultado.json"],
    },
}


class ImpactoDesatualizado(RuntimeError):
    """Um ponto de entrada leu algo que mudou depois da ultima revalidacao."""


def _hash(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return "sha256:" + h.hexdigest()


def _saidas(no: str, nos: dict) -> list[Path]:
    s = nos[no]["saidas"]
    return list(s()) if callable(s) else list(s)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)


def _impressao(no: str, nos: dict) -> dict:
    d = nos[no]
    return {
        "codigo": {**{_rel(p): _hash(p) for p in d["codigo"]},
                   **{f"impressao:{k}": f() for k, f in d.get("impressoes", {}).items()}},
        "dados": {_rel(p): _hash(p) for p in d["dados"]},
        "dependencias": {dep: {_rel(p): _hash(p) for p in _saidas(dep, nos)} for dep in d["depende"]},
        "saidas": {_rel(p): _hash(p) for p in _saidas(no, nos)},
    }


def carimbos(historico: Path = HISTORICO) -> dict:
    """Ultimo carimbo de cada no."""
    ultimo = {}
    if historico.exists():
        for linha in historico.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                c = json.loads(linha)
                ultimo[c["no"]] = c
    return ultimo


def estado(nos: dict = NOS, historico: Path = HISTORICO) -> dict:
    cs = carimbos(historico)
    proprio = {}
    for no in nos:
        c = cs.get(no)
        if c is None:
            proprio[no] = ("sem_carimbo", ["nunca revalidado sob o rastreio"])
            continue
        agora = _impressao(no, nos)
        motivos = []
        for chave in ("codigo", "dados", "saidas"):
            for arq, h in agora[chave].items():
                if c["impressao"][chave].get(arq) != h:
                    motivos.append(f"{chave}_mudou:{arq}")
        for dep, arqs in agora["dependencias"].items():
            for arq, h in arqs.items():
                if c["impressao"]["dependencias"].get(dep, {}).get(arq) != h:
                    motivos.append(f"dependencia_mudou:{dep}:{arq}")
        if any(h is None for h in agora["saidas"].values()):
            motivos.append("saida_ausente")
        proprio[no] = ("desatualizado", motivos) if motivos else ("atual", [])

    final: dict[str, dict] = {}

    def resolver(no: str) -> dict:
        if no in final:
            return final[no]
        est, mot = proprio[no]
        acima = [dep for dep in nos[no]["depende"] if resolver(dep)["estado"] != "atual"]
        if est == "atual" and acima:
            est, mot = "herdado", [f"a_montante_nao_atual:{dep}" for dep in acima]
        final[no] = {"estado": est, "motivos": mot,
                     "carimbado_em": cs.get(no, {}).get("quando_utc")}
        return final[no]

    for no in nos:
        resolver(no)
    return final


def exigir_atual(nos_exigidos, quem: str, nos: dict = NOS, historico: Path = HISTORICO) -> None:
    """Ponto de entrada: falha se algum no lido nao esta atual."""
    e = estado(nos, historico)
    ruins = {no: e[no] for no in nos_exigidos if e[no]["estado"] != "atual"}
    if ruins:
        detalhe = "; ".join(f"{no}={v['estado']} ({', '.join(v['motivos'][:3])})"
                            for no, v in ruins.items())
        raise ImpactoDesatualizado(
            f"{quem}: entradas nao revalidadas -- {detalhe}. Revalide de cima para baixo "
            f"e carimbe (rastreio_de_impacto.py --carimbar)")


def carimbar(no: str, quem: str = "manual", nos: dict = NOS, historico: Path = HISTORICO) -> dict:
    """Aceita o estado atual do no. Recusa se alguma dependencia nao esta atual."""
    from datetime import datetime, timezone
    e = estado(nos, historico)
    ruins = [dep for dep in nos[no]["depende"] if e[dep]["estado"] != "atual"]
    if ruins:
        raise ImpactoDesatualizado(
            f"nao carimbo {no}: dependencias nao atuais {ruins}; carimbe-as primeiro")
    imp = _impressao(no, nos)
    faltam = [a for a, h in imp["saidas"].items() if h is None]
    if faltam:
        raise ImpactoDesatualizado(f"nao carimbo {no}: saidas ausentes {faltam}")
    c = {"no": no, "quando_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
         "quem": quem, "impressao": imp}
    historico.parent.mkdir(parents=True, exist_ok=True)
    with open(historico, "a", encoding="utf-8") as f:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return c


def descendentes(no: str, nos: dict = NOS) -> list[str]:
    out, fila = [], [no]
    while fila:
        x = fila.pop()
        for n, d in nos.items():
            if x in d["depende"] and n not in out:
                out.append(n)
                fila.append(n)
    return out


def markdown(e: dict) -> str:
    L = ["# Rastreio de impacto", "",
         "**Gerado de** `scripts/metodo/rastreio_de_impacto.py`. **Não editar à mão.**", "",
         "**Não é.** nao e auditoria de leitura: os pontos de entrada que afirmam chamam "
         "`exigir_atual` e falham quando o que leem nao foi revalidado.", "",
         "| nó | depende de | estado | carimbado em | motivos |", "|---|---|---|---|---|"]
    for no, v in e.items():
        L.append(f"| `{no}` | {', '.join(NOS[no]['depende']) or '—'} | `{v['estado']}` | "
                 f"{v['carimbado_em'] or '—'} | {'; '.join(v['motivos'][:3]) or '—'} |")
    return "\n".join(L) + "\n"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--carimbar", nargs="*", default=[], help="nos a aceitar, de cima para baixo")
    a = ap.parse_args()
    for no in a.carimbar:
        carimbar(no, quem="cli")
        print(f"  carimbado: {no}")
    e = estado()
    for no, v in e.items():
        print(f"  {no:22s} {v['estado']:14s} {'; '.join(v['motivos'][:2])}")
    DOC.write_text(markdown(e), encoding="utf-8")
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "estado.json").write_text(json.dumps(e, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def ganchos(no: str, quem: str):
    """(entrada, saida) para o main de um gerador: exige a montante, carimba ao fim."""
    def entrada():
        exigir_atual(NOS[no]["depende"], quem)

    def saida():
        carimbar(no, quem)
    return entrada, saida
