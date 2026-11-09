"""Testes da linhagem v10: contratos de software (dados sinteticos) e coerencia das saidas salvas quando existirem.
Nao validam a tese."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts" / "vulnerabilizados"))
import v10_comum  # noqa: E402

V10 = v10_comum.V10


def test_reservatorio_limitado_e_logit_da_insercao():
    from v10_iii_memoria import Reservatorio
    r = Reservatorio(30, np.random.default_rng(0))
    for lote in range(10):
        n = 50
        r.inserir(np.full((n, 2), lote), np.ones(n), np.zeros(n), np.full(n, float(lote)))
        assert len(r.y) <= 30
    z, y, q, s = r.arrays()
    assert len(y) == 30 and r.n == 500
    assert np.all(z[:, 0] == s)  # o logit guardado e o do momento da insercao, nunca recalculado
    assert len(np.unique(s)) > 3  # amostragem de reservatorio mistura etapas


def test_reservatorio_vazio():
    from v10_iii_memoria import Reservatorio
    assert Reservatorio(5, np.random.default_rng(0)).arrays() is None


def test_suporte_exige_cinco_cinco_tres():
    from v10_iii_memoria import suporte
    y = np.r_[np.ones(5), np.zeros(5)]
    assert suporte(y, np.array(list("abcab") + ["x"] * 5))
    assert not suporte(y, np.array(list("aabbb") + ["x"] * 5))
    assert not suporte(np.r_[np.ones(4), np.zeros(6)], np.array(list("abcdxxxxxx")))
    assert not suporte(np.ones(10), np.array(list("abcdefghij")))


def test_nucleo_limpo_e_o_qsep_da_v9():
    import pickle
    p = v10_comum.V9 / "p0_regularizacao/modelos.pkl"
    if not p.exists():
        pytest.skip("v9 ausente")
    with p.open("rb") as f:
        M = pickle.load(f)
    Q = v10_comum.nucleo_limpo()
    assert Q.lam_p == M["Qsep_sel_fis"].lam_p and Q.lam_i == M["Qsep_sel_fis"].lam_i


def test_ciclo_registros_completos():
    c, lt = V10 / "ii_ciclo/consultas.parquet", V10 / "ii_ciclo/lotes.parquet"
    if not c.exists():
        pytest.skip("ciclo nao executado")
    q, lt = pd.read_parquet(c), pd.read_parquet(lt)
    assert {"ponto_id", "ordem", "lote", "escore_previo", "posto_previo", "estado_revelado", "grupo", "custo"} <= set(q.columns)
    assert not q.duplicated(["coorte", "pool", "estrategia", "semente", "ponto_id"]).any()
    assert {"theta", "c", "motivo", "auc_teste_todas", "auc_teste_fcu", "auc_retencao"} <= set(lt.columns)
    assert lt[lt.estrategia.eq("congelado")].ativo.eq(False).all()
    n = q.groupby(["coorte", "pool", "estrategia", "semente"]).size()
    assert n.eq(400).all()


def test_competencia_zero_revisoes():
    p = V10 / "iv_competencia/cobertura.csv"
    if not p.exists():
        pytest.skip("competencia nao executada")
    assert pd.read_csv(p).revisoes_concluidas.eq(0).all()


def test_crc_lambda_controla_perda_e_e_monotono_em_alfa():
    from v10_iv_competencia import lambda_crc
    rng = np.random.default_rng(3)
    risco = rng.random(2000)
    y = (rng.random(2000) < risco * .3).astype(int)
    baixo = np.ones(2000, bool)
    l1, l2 = lambda_crc(y, baixo, risco, .02), lambda_crc(y, baixo, risco, .05)
    assert l1 <= l2
    n = len(y)
    assert n / (n + 1) * np.mean((y == 1) & (risco <= l1)) + 1 / (n + 1) <= .02
    assert lambda_crc(y, baixo, risco, 1 / (n + 1) / 2) == -np.inf  # alfa abaixo do termo 1/(n+1): nada e aceito


def test_verificacao_independente():
    p = V10 / "verificacao_independente.json"
    if not p.exists():
        pytest.skip("sintese nao executada")
    assert json.loads(p.read_text(encoding="utf-8"))["ok"]
