"""Testes da linhagem petropolis-aprendizado/v1: metricas, prior, montante, contrato e integridade.

Nao validam a tese; conferem que as pecas fazem o que o protocolo diz.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts/petropolis_aprendizado"))
V1 = RAIZ / "local_runs/petropolis-aprendizado/v1"
precisa_linhagem = pytest.mark.skipif(not (V1 / "protocolo/protocolo_v1.json").exists(),
                                      reason="linhagem local ignorada pelo Git ausente")


@precisa_linhagem
def test_auc_obj_igual_a_forca_bruta():
    from pa_05_treino import auc_obj
    rng = np.random.default_rng(0)
    vp, vb = rng.normal(1, 1, 30), rng.normal(0, 1, 200)
    vp[3] = vb[5]                                       # empate
    cic = np.repeat(np.arange(10), 3)
    a, por = auc_obj(vp, cic, 10, vb)
    bruto = [np.mean([(np.sum(vb < x) + 0.5 * np.sum(vb == x)) / len(vb) for x in vp[cic == c]]) for c in range(10)]
    assert np.allclose(por, bruto) and abs(a - np.mean(bruto)) < 1e-12


@precisa_linhagem
def test_prior_forte_prende_coeficientes_e_prior_fraco_aprende():
    from pa_05_treino import ajustar_log
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + rng.normal(size=400) > 0).astype(float)
    b0 = np.array([0.0, 2.0, -1.0])
    forte = ajustar_log(X, y, 1e8, b0)
    fraco = ajustar_log(X, y, 1e-3, b0)
    assert np.allclose(forte["coef"], b0, atol=1e-4)
    assert fraco["coef"][0] > 0.5 and abs(fraco["coef"][1]) < 0.5


@precisa_linhagem
def test_rotulo_segue_a_margem_congelada():
    from pa_05_treino import rotulo
    assert rotulo(0.03, 0.001, 0.06) == "sustentado"
    assert rotulo(0.015, 0.001, 0.03) == "inconclusivo"          # IC exclui 0 mas abaixo da margem
    assert rotulo(-0.03, -0.06, -0.001) == "contrario"
    assert rotulo(0.05, -0.01, 0.1) == "inconclusivo"


@precisa_linhagem
def test_montante_max_propaga_so_rio_abaixo():
    from pa_05b_emenda1 import montante_max
    v = np.array([[5.0, 0.0, 0.0]])
    p = np.array([[1, 1, 1]])                     # ESRI 1 = leste
    out = montante_max(v, p, passos=1)
    assert out.tolist() == [[5.0, 5.0, 0.0]]
    assert montante_max(v, p, passos=2).tolist() == [[5.0, 5.0, 5.0]]
    p2 = np.array([[16, 16, 16]])                 # oeste: nada chega a leste
    assert montante_max(v, p2, passos=3).tolist() == [[5.0, 0.0, 0.0]]


@precisa_linhagem
def test_protocolo_congelado_antes_do_treino_e_amostras_iguais():
    from pa_comum import sha256
    prot = json.loads((V1 / "protocolo/protocolo_v1.json").read_text(encoding="utf-8"))
    etapas = [json.loads(l) for l in (V1 / "execucao_etapas.jsonl").read_text(encoding="utf-8").splitlines()]
    ordem = [e["etapa"] for e in etapas]
    assert ordem.index("pa_04_protocolo") < ordem.index("pa_05_treino")
    assert ordem.index("pa_04b_emenda1") < ordem.index("pa_05b_emenda1")
    for nome, h in prot["amostras_sha256"].items():
        assert sha256(V1 / "referencias" / nome) == h


@precisa_linhagem
def test_nenhum_sitio_consultado_no_teste():
    r = json.loads((V1 / "resultados/resultado_treino.json").read_text(encoding="utf-8"))
    assert r["consultas_em_teste"]["em_celula_de_teste"] == 0
    v = json.loads((V1 / "verificacao_independente.json").read_text(encoding="utf-8"))
    assert v["ok"] and v["sem_vazamento_pool_teste"]["sitios_pool_dentro_da_exclusao"] == 0


@precisa_linhagem
def test_todos_os_bracos_planejados_publicados():
    m = pd.read_csv(V1 / "resultados/metricas_bracos.csv")
    locais = {"A2_log_l1", "A2_log_l25", "A3_log_l1", "A3_log_l25", "A3perm_log_l25", "A2_hgb", "A3_hgb"}
    assert {"A0_regra", "A1_log", "A1_hgb"} <= set(m.braco)
    for b in locais:
        sub = m[m.braco == b]
        assert sorted(sub.B.astype(str).unique()) == sorted(["10", "30", "100", "300", "todos"])
        assert sub.groupby(sub.B.astype(str)).size().loc[["10", "30", "100", "300"]].eq(10).all()


@precisa_linhagem
def test_candidata_preserva_natureza_e_nao_declara_seguranca():
    c = json.loads((V1 / "territorio/candidata_resposta_territorial.json").read_text(encoding="utf-8"))
    assert c["contrato"] == ["agua", "encosta", "conexao", "contexto_social", "fontes", "limites", "observacao_necessaria"]
    assert len(c["comunidades"]) == 42
    for f in c["comunidades"]:
        assert set(c["contrato"]) <= set(f)
        assert "NAO autoriza" in f["agua"]["nota"]
        assert f["encosta"]["contexto_tecnico"]["estado"] == "contexto_tecnico"
        assert all(o["estado"] == "observado" for o in f["encosta"]["observado"])
        if not f["encosta"]["observado"]:
            assert any("nao e ausencia" in d for d in f["encosta"]["desconhecido"])
        assert "nao pessoas atingidas" in f["contexto_social"]["nota"]
        texto = json.dumps(f, ensure_ascii=False).lower()
        assert "seguro" not in texto.replace("seguranca contra", "")


@precisa_linhagem
def test_atualizacao_historica_recusada_quando_contraria():
    c = json.loads((V1 / "territorio/candidata_resposta_territorial.json").read_text(encoding="utf-8"))
    ct = pd.read_csv(V1 / "resultados/contrastes_T1.csv")
    contraria = ct[(ct.contraste == "C2_aprendizado_local") & (ct.B.astype(str) == "todos")].rotulo_regra.iloc[0] == "contrario"
    assert c["atualizacao_historica_recusada"] == contraria
