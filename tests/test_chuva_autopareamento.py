"""Autopareamento no tempo: o estimador, as precondicoes e as recusas.

Os testes que importam aqui nao sao "o codigo roda". Sao:

  1. o estimador recupera um efeito CONHECIDO em dado simulado, e recupera o
     nulo quando nao ha efeito -- sem isso, um numero bonito nao significa nada;
  2. o desenho cancela mesmo o que promete cancelar: um confundimento constante
     no lugar nao pode mover a estimativa;
  3. a recusa acontece pelas precondicoes DECLARADAS, e nao por acidente;
  4. `not_identifiable` nao vira um quinto estado da ontologia de suporte.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from chuva import autopareamento_temporal as cc  # noqa: E402


# ------------------------------------------------------------ dado simulado

def simular(n_celulas=12, dias=400, beta_por_mm=0.04, semente=7,
            propensao_por_celula=None) -> pd.DataFrame:
    """Celulas-dia com chuva aleatoria e casos gerados por Poisson.

    `propensao_por_celula` e o confundimento INVARIANTE NO LUGAR -- terreno,
    rede de drenagem, vontade de reclamar. Ele multiplica a taxa base daquela
    celula e nao varia no tempo: e exatamente o que o estrato deve cancelar.
    """
    rng = np.random.default_rng(semente)
    if propensao_por_celula is None:
        propensao_por_celula = np.ones(n_celulas)
    linhas = []
    datas = pd.date_range("2024-01-01", periods=dias, freq="D")
    for c in range(n_celulas):
        chuva = rng.gamma(0.6, 8.0, size=dias) * (rng.random(dias) < 0.35)
        taxa = 2.0 * propensao_por_celula[c] * np.exp(beta_por_mm * chuva)
        casos = rng.poisson(taxa)
        linhas.append(pd.DataFrame({
            "cl_lat": 40.0 + 0.1 * c, "cl_lon": -74.0,
            "dia": datas.strftime("%Y-%m-%d"), "ts": datas,
            cc.GATILHO: np.round(chuva, 4),
            cc.ANTECEDENTE: np.round(rng.gamma(1.0, 5.0, size=dias), 4),
            "casos": casos,
        }))
    return pd.concat(linhas, ignore_index=True)


def _rr(d: pd.DataFrame, exposicoes=None, b=120) -> dict:
    exposicoes = exposicoes or [cc.GATILHO]
    prep = cc.preparar(d, cc.PROTOCOLO_PRIMARIO, exposicoes)
    return cc.estimar(prep, exposicoes, b_bootstrap=b)


# ------------------------------------------------------------- o estimador

def test_recupera_o_efeito_conhecido_do_dado_simulado():
    beta = 0.04
    est = _rr(simular(beta_por_mm=beta))
    rr = est["por_variavel"][cc.GATILHO][f"razao_de_taxas_por_{cc.POR_MM:g}mm"]
    esperado = float(np.exp(cc.POR_MM * beta))
    assert abs(rr - esperado) < 0.06, (rr, esperado)
    lo, hi = est["por_variavel"][cc.GATILHO]["ic95_bootstrap_de_estrato"]
    assert lo <= esperado <= hi, (lo, esperado, hi)


def test_sem_efeito_o_intervalo_contem_1():
    est = _rr(simular(beta_por_mm=0.0, semente=11))
    lo, hi = est["por_variavel"][cc.GATILHO]["ic95_bootstrap_de_estrato"]
    assert lo <= 1.0 <= hi, (lo, hi)


def test_confundimento_constante_no_lugar_nao_move_a_estimativa():
    """A promessa central do desenho, verificada em vez de afirmada.

    Duas simulacoes IDENTICAS a menos da propensao por celula -- que varia
    cem vezes entre a primeira e a ultima. Se o estrato cancela mesmo, as duas
    estimativas tem de coincidir.
    """
    n = 12
    plano = simular(n_celulas=n, propensao_por_celula=np.ones(n), semente=3)
    escarpado = simular(n_celulas=n, semente=3,
                        propensao_por_celula=np.linspace(0.1, 10.0, n))
    a = _rr(plano, b=1)["por_variavel"][cc.GATILHO]["coeficiente_por_mm"]
    b = _rr(escarpado, b=1)["por_variavel"][cc.GATILHO]["coeficiente_por_mm"]
    assert abs(a - b) < 0.01, (a, b)


def test_a_verossimilhanca_condicional_bate_com_a_soma_por_estrato():
    """`reduceat` tem de dar o MESMO numero que o laco explicito.

    A versao vetorizada existe por desempenho; se ela divergisse do laco, o
    resultado inteiro seria de outra funcao objetivo.
    """
    d = simular(n_celulas=4, dias=90, semente=5)
    prep = cc.preparar(d, cc.PROTOCOLO_PRIMARIO, [cc.GATILHO])
    s = prep["quadro"]
    X = s[[cc.GATILHO]].to_numpy(dtype=float)
    y = s["casos"].to_numpy(dtype=float)
    inicio = cc._blocos(s._id.to_numpy())
    beta = np.array([0.03])

    ll_vet, grad_vet = cc._loglik_e_grad(beta, X, y, inicio)
    ll, grad = 0.0, np.zeros(1)
    for a, b in zip(inicio[:-1], inicio[1:]):
        e = X[a:b] @ beta
        yb, n = y[a:b], y[a:b].sum()
        if n == 0:
            continue
        w = np.exp(e - e.max())
        ll += float(yb @ e - n * (e.max() + np.log(w.sum())))
        grad += X[a:b].T @ yb - n * (X[a:b].T @ (w / w.sum()))
    assert ll_vet == pytest.approx(-ll, rel=1e-9)
    assert grad_vet == pytest.approx(-grad, rel=1e-9)


# --------------------------------------------------------- precondicoes

def test_estrato_sem_variacao_de_exposicao_nao_conta_como_amostra():
    """Chuva constante dentro do estrato nao informa nada -- e nao pode
    aparecer no denominador do orcamento de estratos por parametro."""
    d = simular(n_celulas=6, dias=200, semente=9)
    d[cc.GATILHO] = 5.0
    prep = cc.preparar(d, cc.PROTOCOLO_PRIMARIO, [cc.GATILHO])
    assert prep["estratos_informativos"] == 0
    ident = cc.avaliar_identificabilidade(prep, [cc.GATILHO])
    assert not ident["identificavel"]
    assert any("estrato" in f["requisito"] for f in ident["requisitos_nao_atendidos"])


def test_recusa_traz_o_requisito_que_destravaria():
    d = simular(n_celulas=1, dias=20, semente=2)
    d[cc.GATILHO] = 3.0
    ident = cc.avaliar_identificabilidade(
        cc.preparar(d, cc.PROTOCOLO_PRIMARIO, [cc.GATILHO]), [cc.GATILHO])
    assert not ident["identificavel"]
    for falta in ident["requisitos_nao_atendidos"]:
        assert falta["o_que_destravaria"], falta
        assert "observado" in falta


def test_precondicoes_sao_declaradas_antes_de_estimar():
    d = simular(n_celulas=8, dias=300, semente=4)
    ident = cc.avaliar_identificabilidade(
        cc.preparar(d, cc.PROTOCOLO_PRIMARIO, [cc.GATILHO]), [cc.GATILHO])
    assert ident["criterios"]["declarados_antes_de_estimar"] is True
    assert ident["criterios"]["min_estratos_por_parametro"] == 10


# ------------------------------------------------------------- ontologia

def test_nao_identificavel_e_razao_de_not_testable_e_nao_um_quinto_estado():
    import suporte_cientifico as sup

    d = simular(n_celulas=3, dias=40, semente=6)
    d[cc.GATILHO] = 1.0
    prep = cc.preparar(d, cc.PROTOCOLO_PRIMARIO, [cc.GATILHO])
    ident = cc.avaliar_identificabilidade(prep, [cc.GATILHO])
    estado = cc.estado_de_suporte(ident, {"estimado": False}, cc.GATILHO)
    assert estado["support"] == sup.NOT_TESTABLE
    assert estado["support"] in sup.ESTADOS_DE_SUPORTE
    assert estado["diagnostico"] == "not_identifiable"
    assert "not_identifiable" not in sup.ESTADOS_DE_SUPORTE


def test_ic_inteiramente_abaixo_de_1_e_not_supported_e_nao_inconclusive():
    import suporte_cientifico as sup

    ident = {"identificavel": True}
    est = {"estimado": True, "por_variavel": {
        cc.GATILHO: {"ic95_bootstrap_de_estrato": [0.71, 0.93]}}}
    assert cc.estado_de_suporte(ident, est, cc.GATILHO)["support"] == sup.NOT_SUPPORTED


def test_ic_que_cruza_1_e_inconclusive():
    import suporte_cientifico as sup

    ident = {"identificavel": True}
    est = {"estimado": True, "por_variavel": {
        cc.GATILHO: {"ic95_bootstrap_de_estrato": [0.95, 1.30]}}}
    assert cc.estado_de_suporte(ident, est, cc.GATILHO)["support"] == sup.INCONCLUSIVE


# ---------------------------------------------------------- consolidacao

def _bloco(rr, exclui_estrato, exclui_celula):
    return {"por_exposicao": {"gatilho": {
        "exposicoes": [cc.GATILHO],
        "estimativa": {"estimado": True, "por_variavel": {cc.GATILHO: {
            f"razao_de_taxas_por_{cc.POR_MM:g}mm": rr,
            "ic95_exclui_1": exclui_estrato,
            "ic95_de_celula_exclui_1": exclui_celula}}},
        "suporte": {"support": "supported"}}}}


def test_consolidacao_rebaixa_quando_o_bootstrap_de_celula_inclui_1():
    por_protocolo = {"mes_dow": _bloco(1.20, True, False),
                     "semana_iso": _bloco(1.15, True, True)}
    c = cc.consolidar(por_protocolo)
    assert c["support"] == "inconclusive"
    assert c["suporte_do_protocolo_primario"] == "supported"


def test_consolidacao_rebaixa_quando_a_direcao_inverte_entre_protocolos():
    por_protocolo = {"mes_dow": _bloco(1.30, True, True),
                     "semana_iso": _bloco(0.85, True, True)}
    assert cc.consolidar(por_protocolo)["support"] == "inconclusive"


def test_consolidacao_mantem_supported_quando_tudo_concorda():
    por_protocolo = {"mes_dow": _bloco(1.50, True, True),
                     "semana_iso": _bloco(1.32, True, True)}
    c = cc.consolidar(por_protocolo)
    assert c["support"] == "supported"
    assert c["escrita_depois_de_ver_os_resultados"] is True


# ------------------------------------------------------------- protocolos

def test_todo_protocolo_tem_chave_por_celula_e_por_tempo():
    d = simular(n_celulas=3, dias=120, semente=8)
    for nome, p in cc.PROTOCOLOS.items():
        chave = p["chave"](d)
        # duas celulas nunca compartilham estrato: o lugar entra na chave
        por_estrato = d.assign(_e=chave).groupby("_e").cl_lat.nunique()
        assert por_estrato.max() == 1, nome
        # e o estrato nao e o tempo inteiro: sobra contraste dentro dele
        assert d.assign(_e=chave)._e.nunique() > 3, nome


def test_o_protocolo_primario_esta_congelado_no_modulo():
    assert cc.PROTOCOLO_PRIMARIO in cc.PROTOCOLOS
    assert cc.PROTOCOLO_PRIMARIO == "mes_dow"


# --------------------------------------------------- controle negativo

def _bloco_principal(coef):
    return {"por_exposicao": {"gatilho": {"estimativa": {
        "estimado": True,
        "por_variavel": {cc.GATILHO: {"coeficiente_por_mm": coef}}}}}}


def _controle(coef, exclui_1):
    return {"estimado": True, "por_variavel": {cc.CONTROLE_FUTURO: {
        "coeficiente_por_mm": coef,
        f"razao_de_taxas_por_{cc.POR_MM:g}mm": round(float(2.718 ** (10 * coef)), 4),
        "ic95_exclui_1": exclui_1}}}


def test_controle_nulo_e_lido_como_ausencia_de_vies():
    r = cc._leitura_do_controle(_controle(0.0005, False),
                                {cc.PROTOCOLO_PRIMARIO: _bloco_principal(0.041)})
    assert r["desviou_de_1"] is False
    assert r["dentro_da_tolerancia"] is True
    assert "nao ha vies temporal residual" in r["leitura"]


def test_controle_pequeno_ao_lado_de_efeito_grande_e_piso_de_vies_declarado():
    """Nova York: controle -0,005/mm ao lado de efeito +0,041/mm."""
    r = cc._leitura_do_controle(_controle(-0.005, True),
                                {cc.PROTOCOLO_PRIMARIO: _bloco_principal(0.041)})
    assert r["desviou_de_1"] is True
    assert r["dentro_da_tolerancia"] is True
    assert r["magnitude_relativa_ao_efeito_principal"] < cc.FRACAO_MAXIMA_DO_CONTROLE
    assert "o achado sobrevive com o piso de vies declarado" in r["leitura"]


def test_controle_comparavel_ao_efeito_derruba_o_achado():
    r = cc._leitura_do_controle(_controle(-0.030, True),
                                {cc.PROTOCOLO_PRIMARIO: _bloco_principal(0.041)})
    assert r["dentro_da_tolerancia"] is False
    assert "NAO se sustenta" in r["leitura"]


def test_o_criterio_do_controle_declara_que_foi_escrito_depois():
    r = cc._leitura_do_controle(_controle(-0.005, True),
                                {cc.PROTOCOLO_PRIMARIO: _bloco_principal(0.041)})
    assert r["criterio_escrito_depois_de_ver_o_resultado"] is True


def test_a_chuva_futura_nao_pode_entrar_como_exposicao_do_resultado():
    """O controle e falseamento, e nao mais uma variavel candidata."""
    assert cc.CONTROLE_FUTURO not in sum(cc.EXPOSICOES.values(), [])


def test_controle_negativo_fora_da_tolerancia_derruba_a_consolidacao():
    """Sao Paulo: `supported` nos tres protocolos, e a chuva de trinta dias
    DEPOIS explicando um terco do tamanho do efeito. Manter o teste e ignorar
    o resultado dele seria pior do que nao ter o teste."""
    por_protocolo = {"mes_dow": _bloco(1.14, True, True),
                     "semana_iso": _bloco(1.11, True, True)}
    fora = {"dentro_da_tolerancia": False,
            "magnitude_relativa_ao_efeito_principal": 0.3335}
    c = cc.consolidar(por_protocolo, controle=fora)
    assert c["support"] == "inconclusive"
    assert c["controle_negativo_dentro_da_tolerancia"] is False
    assert c["magnitude_do_controle_sobre_o_efeito"] == 0.3335
    assert "controle negativo sair" in c["regra"]


def test_controle_dentro_da_tolerancia_preserva_supported():
    por_protocolo = {"mes_dow": _bloco(1.51, True, True),
                     "semana_iso": _bloco(1.32, True, True)}
    dentro = {"dentro_da_tolerancia": True,
              "magnitude_relativa_ao_efeito_principal": 0.1211}
    assert cc.consolidar(por_protocolo, controle=dentro)["support"] == "supported"


def test_sem_controle_a_consolidacao_nao_inventa_veto():
    por_protocolo = {"mes_dow": _bloco(1.51, True, True)}
    assert cc.consolidar(por_protocolo)["support"] == "supported"


def test_a_razao_do_rebaixamento_nomeia_a_condicao_que_falhou():
    """Publicar `a direcao inverte` quando quem reprovou foi o controle
    negativo poria uma razao falsa ao lado de um estado certo."""
    por_controle = cc._razao_do_rebaixamento({
        "os_dois_bootstraps_excluem_1": True,
        "algum_protocolo_inverte_a_direcao": False,
        "controle_negativo_dentro_da_tolerancia": False,
        "magnitude_do_controle_sobre_o_efeito": 0.3335})
    assert "falseamento" in por_controle and "33%" in por_controle

    por_bootstrap = cc._razao_do_rebaixamento({
        "os_dois_bootstraps_excluem_1": False,
        "algum_protocolo_inverte_a_direcao": False,
        "controle_negativo_dentro_da_tolerancia": True})
    assert "bootstrap de celula" in por_bootstrap

    por_direcao = cc._razao_do_rebaixamento({
        "os_dois_bootstraps_excluem_1": True,
        "algum_protocolo_inverte_a_direcao": True,
        "controle_negativo_dentro_da_tolerancia": True})
    assert "direcao inverte" in por_direcao
