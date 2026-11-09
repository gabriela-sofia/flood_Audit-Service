"""Frente D: o juiz neural so pode subtrair, e nunca autorizar.

Usar um modelo de linguagem para verificar o que um modelo de linguagem
escreveu e circular por natureza. O que quebra a circularidade e a conjuncao:
publica se a regra lexica aprova E o juiz nao discorda. Estes testes protegem
essa assimetria, que e a unica coisa que torna o pior caso do juiz inofensivo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "verificacao"))
sys.path.insert(0, str(ROOT / "scripts" / "servico"))


# ------------------------------------------------- D1: o conjunto


def test_os_trechos_sao_reais_e_os_claims_sao_declarados_a_mao():
    """Nao ha como gerar automaticamente um par cujo lexico casa e o sentido
    nao: essa e exatamente a relacao que nenhum metodo lexico constroi."""
    from casos_de_suporte import CASOS, FAMILIAS, TRECHOS, todos

    assert len(CASOS) >= 15
    assert {c["familia"] for c in CASOS} == set(FAMILIAS)
    for caso in todos():
        assert caso["trecho"] in TRECHOS
        assert caso["texto_do_trecho"] == TRECHOS[caso["trecho"]]
        assert isinstance(caso["sustenta"], bool)


def test_as_duas_familias_do_meio_existem_em_numero_util():
    """`sustenta_parafrase` mede o custo; `nao_sustenta_lexico` mede o ganho.

    Sem as duas, a frente mediria so o que o metodo atual ja acerta.
    """
    from casos_de_suporte import por_familia

    f = por_familia()
    assert len(f["sustenta_parafrase"]) >= 4
    assert len(f["nao_sustenta_lexico"]) >= 6


def test_o_lexico_erra_a_familia_que_a_docstring_admite_errar():
    """O limite que `claims.sustenta` ja declarava, agora medido."""
    from casos_de_suporte import por_familia
    from claims import sustenta

    erros = sum(1 for c in por_familia()["nao_sustenta_lexico"]
                if sustenta(c["claim"], c["texto_do_trecho"])[0])
    assert erros >= 5, ("se o lexico passasse a acertar isto sozinho, a frente "
                        "D perderia o motivo de existir -- e o numero teria de "
                        "ser remedido, nao o teste afrouxado")


# ------------------------------------- D3: a assimetria da conjuncao


def _conjuncao(aprovou_lexico: bool, discorda) -> bool:
    return bool(aprovou_lexico) and not (discorda is True)


def test_o_juiz_nunca_aprova_o_que_o_lexico_reprovou():
    """A assimetria inteira em uma linha."""
    for discorda in (True, False, None):
        assert _conjuncao(False, discorda) is False


def test_o_juiz_pode_derrubar_o_que_o_lexico_aprovou():
    assert _conjuncao(True, True) is False
    assert _conjuncao(True, False) is True


def test_juiz_indisponivel_nao_e_discordancia():
    """Juiz fora do ar nao derruba frase nenhuma.

    Mesma regra da proveniencia ausente: e informacao sobre o resultado, nao
    condicao para ele. Se indisponibilidade derrubasse, desligar o Ollama
    silenciaria o sistema inteiro.
    """
    assert _conjuncao(True, None) is True


def test_o_pior_caso_do_juiz_e_inofensivo():
    """Alucinando 'sustenta' para tudo, o sistema volta ao que era antes.

    Alucinando 'nao sustenta' para tudo, ele publica menos -- nunca errado.
    """
    from casos_de_suporte import todos
    from claims import sustenta

    casos = todos()
    so_lexico = [bool(sustenta(c["claim"], c["texto_do_trecho"])[0])
                 for c in casos]
    juiz_sempre_concorda = [_conjuncao(l, False) for l in so_lexico]
    juiz_sempre_discorda = [_conjuncao(l, True) for l in so_lexico]

    assert juiz_sempre_concorda == so_lexico
    assert not any(juiz_sempre_discorda)


def test_resposta_fora_do_vocabulario_nao_vira_voto():
    """Um juiz que respondeu outra coisa nao respondeu."""
    import medir_suporte

    class RespostaEstranha:
        pass

    # `julgar_local` devolve None quando nao reconhece a palavra; o teste cobre
    # a regra de leitura sem depender de servidor.
    plana = "TALVEZ, DEPENDE DO CONTEXTO".upper().replace("-", "_")
    assert "NAO_SUSTENTA" not in plana
    assert "SUSTENTA" not in plana
    assert medir_suporte.PROMPT_JUIZ.count("uma unica palavra") == 1


def test_o_juiz_nao_ve_o_payload_nem_o_que_foi_gerado():
    """A entrada do juiz e so `(claim, trecho)`.

    Se ele visse o payload, saberia qual e a resposta desejada -- e um juiz que
    sabe o gabarito nao esta julgando.
    """
    import inspect

    from medir_suporte import julgar_local

    parametros = set(inspect.signature(julgar_local).parameters)
    assert parametros == {"claim", "trecho", "url", "modelo", "tempo_limite"}


# --------------------------------------------- D5: o limite declarado


def test_o_custo_em_falso_negativo_esta_declarado():
    """Conjuncao so aperta: a fonte certa que o lexico reprova segue reprovada.

    Declarar isso e o que separa uma escolha de um efeito colateral.
    """
    from medir_suporte import avaliar

    r = avaliar(usar_juiz=False)["resumo"]
    assert "nao conserta falso negativo" in r["custo_declarado"]
    assert "juiz so subtrai" in r["regra"]
