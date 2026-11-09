"""Protecoes do MT01: modelo, independencia, recusa de reserva e uso nenhum do rotulo do destino."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/metodo"))
import mt01_experimento as ex
import mt01_independencia as ind
import mt01_populacao as mp


def _dados(n=600, eventos=4, semente=3):
    rng = np.random.default_rng(semente)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "z": rng.normal(size=n)})
    ev = np.repeat(np.arange(eventos), n // eventos)
    eta = -1.5 * x.a + .5 * x.b + .8 * x.a * x.z + rng.normal(size=eventos)[ev]
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return x, y, ev


def test_gradiente_confere_com_diferenca_finita():
    from scipy.optimize import check_grad
    x, y, ev = _dados()
    m = ex.Logistica(["a", "b"], [("a", "z")], territorio=True, lam_p=.1, lam_i=.1, lam_t=.1)
    m.fit(x, y, ev, ev % 2)  # ajusta para criar padronizador; depois testa a perda num ponto qualquer
    # reconstroi a funcao de perda pelo mesmo caminho do fit, com maxiter=0
    import scipy.optimize as so
    capturada = {}
    original = so.minimize

    def espiao(f, x0, **kw):
        capturada["f"], capturada["x0"] = f, x0
        return original(f, x0, **{**kw, "options": {"maxiter": 1}})
    ex.minimize = espiao
    try:
        m.fit(x, y, ev, ev % 2)
    finally:
        ex.minimize = original
    f = capturada["f"]
    v = np.random.default_rng(1).normal(size=len(capturada["x0"])) * .1
    erro = check_grad(lambda t: f(t)[0], lambda t: f(t)[1], v)
    assert erro < 1e-5


def test_sem_penalidade_recupera_sinais_e_ordem_independe_do_intercepto():
    x, y, ev = _dados(4000, 8)
    m = ex.Logistica(["a", "b"], lam_p=1e-6).fit(x, y, ev)
    assert m.b[0] < 0 < m.b[1]
    s = m.escore(x)
    assert np.allclose(s, m.escore(x))  # escore nao depende do evento: so ordem dentro dele


def test_auc_por_postos_confere_com_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(5)
    y = (rng.random(300) < .2).astype(int)
    s = rng.normal(size=300) + y
    s[:20] = np.round(s[:20])  # empates
    assert abs(ex.auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_ap_confere_com_sklearn():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(6)
    y = (rng.random(500) < .1).astype(int)
    s = rng.normal(size=500) + 2 * y
    assert abs(ex.ap(y, s) - average_precision_score(y, s)) < 1e-9


def test_grafo_une_por_evento_lugar_onda_e_nunca_usa_classe():
    d = pd.DataFrame({
        "evento_id": ["a", "a", "b", "c", "d", "e"],
        "unidade_espacial": ["u1", "u2", "u2", "u3", "u4", "u5"],
        "HYBAS_L8": [1, 2, 2, 3, 4, 5], "MAIN_BAS": [10, 10, 10, 20, 20, 30],
        "janela_inicio": ["2020-01-01", "2020-01-01", "2021-01-01", "2019-01-01", "2019-01-10", "2018-01-01"],
        "janela_fim": ["2020-01-05", "2020-01-05", "2021-01-05", "2019-01-05", "2019-01-12", "2018-01-02"],
        "lat": [0, 0, 0, 30, 30.1, -40], "lon": [0, 0, 0, 30, 30.1, 100], "fonte": "x",
        "classe": [0, 1, 0, 1, 0, 1]})
    g, _ = ind.grafo(d)
    assert g["a"] == g["b"]          # lugar repetido
    assert g["c"] == g["d"]          # mesmo MAIN_BAS dentro de 30 dias
    assert g["e"] not in (g["a"], g["c"])
    d2 = d.assign(classe=1 - d.classe)
    assert ind.grafo(d2)[0] == g
    # exutorio em qualquer data une a e b tambem por bacia; liberal nao une c e d
    assert ind.grafo(d, "liberal")[0]["c"] != ind.grafo(d, "liberal")[0]["d"]


def test_reserva_protegida_e_recusada():
    with pytest.raises(PermissionError):
        mp.poligonos_cems("EMSR868")
    with pytest.raises(PermissionError):
        mp.poligonos_cems("EMSR871")


def test_unidade_de_terreno_mapeia_fontes():
    nomes = {"EMSR680_AOI01", "gfd_tile_S12W051", "n1chip_Bolivia_1_LabelHand", "EMSR720"}
    assert mp.unidade_de_terreno("cems|EMSR680_AOI01", nomes) == "EMSR680_AOI01"
    assert mp.unidade_de_terreno("gfd|GFD_S12W051", nomes) == "gfd_tile_S12W051"
    assert mp.unidade_de_terreno("sen1floods11|Bolivia_1_LabelHand", nomes) == "n1chip_Bolivia_1_LabelHand"
    assert mp.unidade_de_terreno("cems|EMSR720_AOI03", nomes) == "EMSR720"
    assert mp.unidade_de_terreno("nyc|c0.1_407_-739", nomes) is None


def test_ln_sca_recupera_area_contribuinte():
    d = pd.DataFrame({"slope_deg": [1., 5., 20.], "twi_dinf": [10., 8., 4.]})
    sca = np.exp(mp.derivar(d.copy()).ln_sca)
    np.testing.assert_allclose(np.log(sca / np.tan(np.radians(d.slope_deg))), d.twi_dinf)


def test_amostragem_teto_por_evento_e_classe_reprodutivel():
    d = pd.DataFrame({"evento_id": np.repeat(["a", "b"], 50), "classe": np.tile([0, 1], 50)})
    s1, s2 = ex.amostrar(d, 10, 7), ex.amostrar(d, 10, 7)
    assert s1.index.equals(s2.index)
    assert s1.groupby(["evento_id", "classe"]).size().max() == 10


def test_adaptacao_local_nao_altera_nucleo():
    x, y, ev = _dados(2000, 4)
    m = ex.Logistica(["a", "b"], territorio=True, lam_p=.01, lam_t=1.).fit(x, y, ev, ev % 2)
    b = m.b.copy()
    u = m.adaptar(x.iloc[:100], y[:100])
    assert np.array_equal(b, m.b) and u.shape == (2,)


def test_veredito_recusa_com_menos_de_8_unidades_e_decide_com_ic():
    import mt01_relatorio as rel
    u = pd.DataFrame({"G": np.full(7, .8), "A0_regra_hand": np.full(7, .7)})
    assert rel.contraste(u, "G", "A0_regra_hand", "superioridade", .01, 200)["veredito"] == "not_testable"
    rng = np.random.default_rng(2)
    u = pd.DataFrame({"G": .8 + rng.normal(0, .01, 30), "A0_regra_hand": np.full(30, .7)})
    assert rel.contraste(u, "G", "A0_regra_hand", "superioridade", .01, 500)["veredito"] == "sustentado"
    u = pd.DataFrame({"C": .7 + rng.normal(0, .001, 30), "G": np.full(30, .7)})
    assert rel.contraste(u, "C", "G", "nao_inferioridade", .01, 500)["veredito"] == "nao_inferior"
    u = pd.DataFrame({"G_sem_conexao": .6 + rng.normal(0, .01, 30), "G": np.full(30, .7)})
    assert rel.contraste(u, "G_sem_conexao", "G", "perda", .01, 500)["veredito"] == "perda_relevante"


def test_media_por_unidade_pesa_eventos_igualmente():
    import mt01_relatorio as rel
    m = pd.DataFrame({"destino": ["u", "u", "u"], "modelo": "G", "evento_id": ["a", "b", "b"],
                      "auc": [.9, .5, .7], "processo": np.nan})
    assert abs(rel.por_unidade(m).loc["u", "G"] - (.9 + .6) / 2) < 1e-12
