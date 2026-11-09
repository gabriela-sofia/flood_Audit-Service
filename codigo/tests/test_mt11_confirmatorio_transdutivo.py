import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/metodo"))
import mt11_confirmatorio_transdutivo as m


def _datas(n, mes=5):
    return [f"2017-{mes:02d}-{d:02d}" for d in range(1, n + 1)]


def test_frequencia_v1_e_e1_por_ponto():
    V = np.full((3, 12), -10.)
    V[0, :6] = -20.   # água em metade das datas, mas o P75 do ponto é -10: E1 também vê
    V[1, :] = -16.    # sempre abaixo do limiar: v1 = 1, E1 = 0 (nunca cai 3 dB abaixo do próprio P75)
    f = m.frequencia(V, _datas(12), lat=5.)
    np.testing.assert_allclose(f["F_v1"], [.5, 1., 0.])
    np.testing.assert_allclose(f["F_e1"], [.5, 0., 0.])
    assert f["datas_admitidas"] == 12 and (f["n_pre"] == 12).all()


def test_admissao_recusa_cobertura_radiometria_e_neve():
    V = np.full((10, 4), -10.)
    V[:2, 0] = np.nan            # 80% com leitura < 90%
    V[:, 1] = -30.               # mediana fora de [-25, -3]
    f = m.frequencia(V, ["2017-05-01", "2017-05-13", "2017-05-25", "2017-06-06"], lat=5.)
    assert f["datas_admitidas"] == 2
    neve = m.frequencia(np.full((5, 2), -10.), ["2017-01-10", "2017-05-10"], lat=50.)
    assert neve["datas_admitidas"] == 1
    sul = m.frequencia(np.full((5, 2), -10.), ["2017-01-10", "2017-07-10"], lat=-50.)
    assert sul["datas_admitidas"] == 1


def test_qualidade_usa_so_contraste_jrc():
    F = np.r_[np.full(100, .02), np.full(50, .3)]
    jrc = np.r_[np.zeros(100), np.full(50, 30.)]
    n = np.full(150, 20)
    assert m.qualidade(F, jrc, n)["passa"]
    assert not m.qualidade(np.r_[np.full(100, .2), np.full(50, .3)], jrc, n)["passa"]
    assert not m.qualidade(F, jrc, np.full(150, 5))["passa"]


def test_escore_exportado_reproduz_a_logistica():
    rng = np.random.default_rng(3)
    x = pd.DataFrame({"Q": rng.normal(size=400), "F": rng.random(400)})
    x.loc[::7, "F"] = np.nan
    y = (rng.random(400) < 1 / (1 + np.exp(-(x.Q + 2 * x.F.fillna(0))))).astype(float)
    ev = np.repeat(["a", "b"], 200)
    lg = m.mt.e1.Logistica(["Q", "F"], [("Q", "F")], lam_p=.01, lam_i=.01)
    lg.fit(x, y, ev)
    e = {"coeficientes": dict(zip(lg.principais, lg.b.tolist())),
         "interacoes": [{"colunas": list(a), "peso": float(v)} for a, v in zip(lg.interacoes, lg.g)],
         "padronizador": {"colunas": lg.pad.mu.index.tolist(), "media": lg.pad.mu.to_dict(),
                          "desvio": lg.pad.sd.to_dict(), "mediana": lg.pad.med.to_dict()}}
    np.testing.assert_allclose(m.escore_exportado(x, e), lg.escore(x), atol=1e-12)


@pytest.mark.parametrize("valores,esperado", [
    (np.full(12, .03), "sustentado"), (np.full(12, .002), "ganho_excluido"),
    (np.full(12, -.03), "inferior"), (np.r_[np.full(6, -.03), np.full(6, .05)], "indecidivel"),
    (np.full(5, .05), "not_testable")])
def test_contraste(valores, esperado):
    rng = np.random.default_rng(1)
    assert m.contraste(pd.Series(valores + rng.normal(0, 1e-4, len(valores))))["veredito"] == esperado


def test_ordem_de_tiles_e_deterministica_e_tem_sal_proprio():
    assert m._ordem("4503|47_7") == m._ordem("4503|47_7")
    assert m._ordem("4503|47_7") != m._ordem("4503|47_8")
    assert m.SAL == "mt11|"
