"""Frente C: a recuperacao densa entra sem furar nenhuma garantia existente.

O que estes testes protegem nao e o recall -- e o fato de que um recuperador
novo herda TODAS as travas do que ele acompanha. Um indice que melhora a
metrica e contorna uma barreira e o pior resultado possivel desta frente.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "recuperacao"))
sys.path.insert(0, str(ROOT / "scripts" / "servico"))

pytest.importorskip("sklearn")

CORPUS = ROOT / "local_runs" / "rag-corpus" / "manifest.json"
sem_corpus = pytest.mark.skipif(
    not CORPUS.exists(), reason="corpus RAG ausente; rode rag_corpus.py")


# ------------------------------------------------- C1: o conjunto


def test_a_consulta_morfologica_troca_numero_e_nada_mais():
    """O nivel que mede o defeito real observado em 2026-09-01."""
    from avaliacao_de_recuperacao import LITERAL, MORFOLOGICA, consultas_do_chunk

    texto = ("As favelas e comunidades urbanas do Censo 2022 identificam "
             "desigualdade socioespacial por setor censitario declarado")
    c = consultas_do_chunk(texto, n_termos=6, semente=0)
    assert c[LITERAL] != c[MORFOLOGICA]
    assert len(c[LITERAL].split()) == len(c[MORFOLOGICA].split())


def test_as_consultas_sao_deterministicas():
    """Consulta que muda a cada execucao produz baseline incomparavel."""
    from avaliacao_de_recuperacao import consultas_do_chunk

    texto = "altura acima da drenagem mais proxima seguindo o caminho de fluxo"
    assert consultas_do_chunk(texto, 6, 3) == consultas_do_chunk(texto, 6, 3)


def test_consulta_rejeitada_conta_como_zero_e_nao_como_erro():
    """Um indice que engasga com a consulta do usuario esta falhando de verdade."""
    from avaliacao_de_recuperacao import recuperador_fts5

    assert recuperador_fts5('" OR 1=1 --', "projeto", 5) == []


# ------------------------------------- C4: classe nao sobrescreve classe


@sem_corpus
def test_o_denso_respeita_a_classe_de_autoridade():
    """Um indice denso nao pode furar regra de autoridade por ser denso."""
    from indice_denso import construir

    indice = construir()
    for classe in ("cientifica", "operacional", "projeto"):
        for r in indice.buscar("vulnerabilidade social", classe=classe, k=5):
            assert r["classe_autoridade"] == classe


# ----------------------------------------- C5: injecao nos dois pontos


@sem_corpus
def test_o_denso_nao_devolve_chunk_com_injecao():
    """A garantia que quase ficou de fora.

    A primeira versao lia os `chunk_id` direto do SQLite e devolvia o vizinho
    mais proximo, sem passar pelo `texto_tem_injecao`. Seria um caminho novo
    pelo qual um chunk que fala em imperativo com o sistema chega ao contexto.
    """
    import json
    import sqlite3

    from claims import texto_tem_injecao
    from indice_denso import construir

    manifest = json.loads(CORPUS.read_text(encoding="utf-8"))
    con = sqlite3.connect(manifest["db"])
    con.row_factory = sqlite3.Row
    linhas = [dict(r) for r in con.execute(
        "SELECT chunk_id, conteudo, classe_autoridade FROM chunks")]
    con.close()

    com_injecao = [r for r in linhas if texto_tem_injecao(r["conteudo"] or "")]
    if not com_injecao:
        pytest.skip("nenhum chunk com injecao no corpus atual")

    indice = construir()
    alvo = com_injecao[0]
    # consulta o proprio texto do chunk envenenado: e o pior caso, porque ele
    # seria o vizinho mais proximo de si mesmo.
    achados = indice.buscar(alvo["conteudo"][:200],
                            classe=alvo["classe_autoridade"], k=5)
    assert alvo["chunk_id"] not in {a["chunk_id"] for a in achados}
    assert any(d["chunk_id"] == alvo["chunk_id"]
               for d in indice.ultimos_descartados_por_injecao)


@sem_corpus
def test_com_injecao_ligado_o_chunk_aparece_para_a_auditoria():
    """O trecho nao e apagado: ele sai do contexto e fica visivel na auditoria."""
    import json
    import sqlite3

    from claims import texto_tem_injecao
    from indice_denso import construir

    manifest = json.loads(CORPUS.read_text(encoding="utf-8"))
    con = sqlite3.connect(manifest["db"])
    con.row_factory = sqlite3.Row
    linhas = [dict(r) for r in con.execute(
        "SELECT chunk_id, conteudo, classe_autoridade FROM chunks")]
    con.close()
    com_injecao = [r for r in linhas if texto_tem_injecao(r["conteudo"] or "")]
    if not com_injecao:
        pytest.skip("nenhum chunk com injecao no corpus atual")

    indice = construir()
    alvo = com_injecao[0]
    achados = indice.buscar(alvo["conteudo"][:200],
                            classe=alvo["classe_autoridade"], k=5,
                            com_injecao=True)
    assert alvo["chunk_id"] in {a["chunk_id"] for a in achados}


# ------------------------------------------------- C3: a fusao


def test_a_fusao_usa_posicao_e_nao_score():
    """Um score de FTS5 e um cosseno nao estao na mesma escala."""
    from comparar_recuperadores import fundir_rrf

    a = [{"chunk_id": "x"}, {"chunk_id": "y"}]
    b = [{"chunk_id": "y"}, {"chunk_id": "z"}]
    saida = fundir_rrf([a, b], k=3)
    ids = [s["chunk_id"] for s in saida]
    assert ids[0] == "y", "quem aparece bem nas duas listas sobe"
    assert set(ids) == {"x", "y", "z"}


def test_a_fusao_nao_inventa_documento():
    from comparar_recuperadores import fundir_rrf

    saida = fundir_rrf([[{"chunk_id": "a"}], [{"chunk_id": "b"}]], k=10)
    assert {s["chunk_id"] for s in saida} == {"a", "b"}


def test_a_comparacao_usa_a_regua_do_baseline():
    """Nao 'equivalente': a MESMA funcao, pela terceira frente seguida."""
    import avaliacao_de_recuperacao
    import comparar_recuperadores

    assert comparar_recuperadores.medir is avaliacao_de_recuperacao.medir
    assert comparar_recuperadores.montar_conjunto is \
        avaliacao_de_recuperacao.montar_conjunto
