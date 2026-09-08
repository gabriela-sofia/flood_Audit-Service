"""A cadeia: encadear troca cobertura por risco, e nomeia onde parou.

A propriedade que nenhum elo tem sozinho: em predicao seletiva, a conjuncao de
seletores tem risco no maximo igual ao menor risco individual, e cobertura no
maximo igual a menor cobertura. Encadear nao pode piorar o risco -- so pode
custar cobertura.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "neural"))
sys.path.insert(0, str(ROOT / "scripts" / "servico"))

from cadeia_seletiva import (  # noqa: E402
    ABSTIDO, ELOS, PUBLICADO, Portao, executar_cadeia,
)


def _portao(elo, nome, passa, confianca=0.9):
    return Portao(elo, nome, lambda e: (passa, confianca, {"eco": nome}))


def test_a_cadeia_publica_quando_todos_passam():
    r = executar_cadeia([_portao("recuperacao", "a", True),
                         _portao("verificacao", "b", True)], {})
    assert r.estado == PUBLICADO
    assert r.parou_em is None
    assert len(r.trilha) == 2


def test_a_abstencao_nomeia_o_elo_onde_parou():
    """Silencio sem endereco e indistinguivel de defeito."""
    r = executar_cadeia([_portao("recuperacao", "a", True),
                         _portao("verificacao", "b", False)], {})
    assert r.estado == ABSTIDO
    assert r.parou_em == "verificacao"


def test_a_cadeia_para_no_primeiro_que_nao_passa():
    """O elo seguinte nao chega a ser consultado."""
    chamou = []

    def registra(e):
        chamou.append("segundo")
        return True, 1.0, {}

    r = executar_cadeia([_portao("recuperacao", "a", False),
                         Portao("verificacao", "b", registra)], {})
    assert r.parou_em == "recuperacao"
    assert chamou == []


def test_elo_quebrado_nao_publica_e_aparece_na_trilha():
    """Excecao nao vira silencio: ela vira abstencao com o erro visivel."""
    def explode(e):
        raise RuntimeError("indice fora do ar")

    r = executar_cadeia([Portao("recuperacao", "a", explode)], {})
    assert r.estado == ABSTIDO
    assert r.parou_em == "recuperacao"
    assert "indice fora do ar" in r.trilha[0]["erro"]


def test_encadear_nunca_publica_mais_que_o_elo_mais_restritivo():
    """A monotonicidade que da sentido a cadeia.

    Se um elo sozinho recusa, a cadeia inteira recusa -- entao a cobertura da
    cadeia e no maximo a do elo mais restritivo, sempre.
    """
    casos = [{"i": i} for i in range(20)]
    permissivo = Portao("recuperacao", "p",
                        lambda e: (e["i"] % 2 == 0, 0.9, {}))
    restritivo = Portao("verificacao", "r",
                        lambda e: (e["i"] % 4 == 0, 0.8, {}))

    so_permissivo = sum(1 for c in casos
                        if executar_cadeia([permissivo], c).estado == PUBLICADO)
    so_restritivo = sum(1 for c in casos
                        if executar_cadeia([restritivo], c).estado == PUBLICADO)
    cadeia = sum(1 for c in casos
                 if executar_cadeia([permissivo, restritivo], c).estado
                 == PUBLICADO)

    assert cadeia <= min(so_permissivo, so_restritivo)


def test_a_confianca_da_cadeia_e_a_do_elo_mais_fraco():
    """Uma cadeia nao e mais confiavel que o seu pior elo."""
    r = executar_cadeia([_portao("recuperacao", "a", True, 0.9),
                         _portao("verificacao", "b", True, 0.2)], {})
    assert r.confianca_minima == pytest.approx(0.2)


def test_a_abstencao_nao_e_resposta_negativa():
    """A distincao que a frente inteira existe para manter."""
    r = executar_cadeia([_portao("verificacao", "b", False)], {})
    d = r.como_dicionario()
    assert "abstencao nao e resposta negativa" in d["nao_e"]
    assert d["estado"] == ABSTIDO


def test_os_elos_estao_na_ordem_do_significado():
    """Observacao antes de representacao, recuperacao antes de verificacao.

    Ordenar por custo faria o elo mais barato decidir o que os caros nunca veem.

    O teste passou a checar a ORDEM RELATIVA em vez da tupla exata quando o elo
    de forcamento entrou: travar a lista inteira faria qualquer elo novo, ainda
    que posto no lugar certo, reprovar. O que precisa ser protegido e a relacao
    entre eles, e nao o tamanho da cadeia.
    """
    posicao = {elo: i for i, elo in enumerate(ELOS)}
    assert posicao["observacao"] < posicao["representacao"]
    assert posicao["recuperacao"] < posicao["verificacao"]
    # O forcamento e fisico: ele pertence ao bloco de observacao e representacao,
    # e vem antes de qualquer elo textual.
    assert posicao["representacao"] < posicao["forcamento"] < posicao["recuperacao"]


CORPUS = ROOT / "local_runs" / "rag-corpus" / "manifest.json"


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus documental ausente")
def test_a_cadeia_real_troca_cobertura_por_acuracia():
    """A demonstracao empirica da propriedade, sem o juiz (deterministico)."""
    from cadeia_seletiva import demonstrar

    r = demonstrar(usar_juiz=False)
    assert 0.0 < r["cobertura"] <= 1.0
    assert r["abstencoes_por_elo"], "nenhuma abstencao seria suspeito"
    assert sum(r["abstencoes_por_elo"].values()) + r["publicados"] == r["casos"]


# ------------------------------------------------- elo de forcamento (E4.3)

def test_forcamento_fora_do_dominio_estimado_abstem():
    """Extrapolar produz numero, nao conhecimento -- e a cadeia diz onde parou."""
    from cadeia_seletiva import FORCAMENTO, portao_forcamento

    portao = portao_forcamento((3.3, 37.72))
    r = executar_cadeia([portao], {"rain_event_day": 100.2})
    assert r.estado == ABSTIDO
    assert r.parou_em == FORCAMENTO
    assert r.trilha[-1]["detalhe"]["distancia_ao_dominio"] > 0


def test_forcamento_dentro_do_dominio_passa_com_confianca_maior_no_centro():
    from cadeia_seletiva import PUBLICADO, portao_forcamento

    portao = portao_forcamento((0.0, 40.0))
    centro = executar_cadeia([portao], {"rain_event_day": 20.0})
    borda = executar_cadeia([portao], {"rain_event_day": 39.0})
    assert centro.estado == PUBLICADO and borda.estado == PUBLICADO
    assert centro.confianca_minima > borda.confianca_minima


def test_pedido_sem_chuva_declarada_nao_vira_chuva_zero():
    """Fail-closed: ausencia de valor e ausencia de dado, e nao 0 mm."""
    from cadeia_seletiva import ABSTIDO as _A, FORCAMENTO, portao_forcamento

    r = executar_cadeia([portao_forcamento((0.0, 40.0))], {})
    assert r.estado == _A
    assert r.parou_em == FORCAMENTO
    assert "nao e chuva zero" in r.trilha[-1]["detalhe"]["nao_e"]


def test_dominio_invertido_e_erro_de_configuracao_e_nao_abstencao_silenciosa():
    from cadeia_seletiva import portao_forcamento

    with pytest.raises(ValueError, match="dominio de forcamento"):
        portao_forcamento((40.0, 10.0))
