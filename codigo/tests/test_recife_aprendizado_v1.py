"""Testes da linhagem recife-aprendizado/v1: geocodificador, metrica dentro do bairro, contrato e integridade."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts/recife_aprendizado"))
V1 = RAIZ / "local_runs/recife-aprendizado/v1"
precisa = pytest.mark.skipif(not (V1 / "protocolo/protocolo_v1.json").exists(), reason="linhagem local ignorada pelo Git ausente")


@precisa
def test_normalizacao_de_logradouro_preserva_travessa_e_corta_complemento():
    from ra_01_referencias import nucleo, norm
    assert norm("1A TRAVESSA DA RUA E GUARANI") == "1A TRAVESSA GUARANI"
    assert norm("Rua Santa Teresa com Avenida Canaa") == "RUA SANTA TERESA"
    assert norm("R. Prof Antonio Luiz") == "RUA PROFESSOR ANTONIO LUIZ"
    assert nucleo("AVENIDA NORTE") <= nucleo("AVENIDA NORTE MIGUEL ARRAES DE ALENCAR")
    assert "\x01" not in (RAIZ / "scripts/recife_aprendizado/ra_01_referencias.py").read_text(encoding="utf-8")


@precisa
def test_percentil_dentro_do_bairro_ignora_outros_bairros():
    from ra_04_treino import pct_bairro
    v = np.array([0.9, 0.1, 0.5, 0.2, 100.0, 50.0])
    y = np.array([1, 0, 0, 1, 0, 0])
    b = np.array([1, 1, 1, 2, 2, 2])
    p = pct_bairro(v, y, b)
    assert p[0] == 1.0 and p[3] == 0.0               # o bairro 2 nao contamina o bairro 1


@precisa
def test_prior_forte_preserva_q_e_fraco_aprende():
    from ra_04_treino import ajustar
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = (X[:, 1] + rng.normal(size=300) > 0).astype(float)
    b0 = np.array([1.5, -1.0])
    assert np.allclose(ajustar(X, y, 1e9, b0)["coef"], b0, atol=1e-4)
    assert ajustar(X, y, 1e-3, b0)["coef"][1] > 0.3


@precisa
def test_particao_por_bairro_e_consultas_fora_do_teste():
    v = json.loads((V1 / "verificacao_independente.json").read_text(encoding="utf-8"))
    assert v["ok"] and v["sem_vazamento"]["bairros_em_comum"] == 0 and v["sem_vazamento"]["consultas_no_teste"] == 0


@precisa
def test_protocolo_antes_do_treino_e_emendas_depois():
    et = [json.loads(l)["etapa"] for l in (V1 / "execucao_etapas.jsonl").read_text(encoding="utf-8").splitlines()]
    assert et.index("ra_03_protocolo") < et.index("ra_04_treino") < et.index("ra_04b_emenda1") < et.index("ra_04c_emenda2")


@precisa
def test_todos_os_bracos_publicados():
    m = pd.read_csv(V1 / "resultados/metricas_bracos.csv")
    for b in ["A2_q_l1", "A2_q_l25", "A3_q_l1", "A3_q_l25", "A3perm_l25", "A2u_l25", "A4_l25", "A4_sem_DREN", "A4_sem_SAIDA", "A2_hgb", "A4_hgb"]:
        s = m[m.braco == b]
        assert sorted(s.B.astype(str).unique()) == ["10", "25", "todas"]
    assert {"A5_l25", "A4p_l25", "A2p_l25", "A0_pe3d"} <= set(m.braco)


@precisa
def test_candidata_preserva_naturezas_e_alerta_de_comprimento():
    c = json.loads((V1 / "territorio/candidata_resposta_territorial.json").read_text(encoding="utf-8"))
    assert len(c["comunidades"]) == 295
    assert c["contrato"] == ["terreno", "captacao", "rede", "saida", "agua_observada", "inferencia_aprendida",
                             "contexto_social", "fontes", "limites", "proxima_observacao"]
    e1 = pd.read_csv(V1 / "resultados/emenda1/contrastes_E1.csv").set_index("contraste")
    if e1.loc["E1c_urbano_controlado_comprimento", "rotulo_regra"] != "sustentado":
        assert c["estado_depois"] == "inferido_ganho_nao_sustentado_sob_controle_de_comprimento"
    for f in c["comunidades"]:
        assert f["captacao"]["estado"] == "contexto_tecnico" and f["rede"]["estado"] == "contexto_tecnico"
        if f["agua_observada"]["ruas_com_relato_sedec"] == 0:
            assert f["agua_observada"]["estado"] == "desconhecido"
        assert "nao pessoas atingidas" in f["contexto_social"]["nota"]
