"""Verificacao real do ambiente de treino da linha causal SUSC.

Este arquivo existe porque, em 04-05/08/2026, foi confirmado que:

1. `firthlogist` e `interpret` (`ExplainableBoostingClassifier`) nao publicam
   wheel/sdist compativel com Python >=3.11 -- reproduzido tanto num sandbox
   de teste (Python 3.10.12, instalou sem problema) quanto no ambiente
   (base) real da autora (Python 3.13.13, conda: pip recusou a instalacao,
   "Requires-Python <3.11" em todas as versoes disponiveis de `interpret`,
   nenhuma versao encontrada de `firthlogist`).
2. `firthlogist` depende de uma API privada do scikit-learn
   (`_validate_data`) removida na versao 1.6 -- reproduzido em Python
   3.10.12 com scikit-learn 1.7.2; corrigido com o shim de
   `susc_ambiente_compat_common.aplicar_shim_firthlogist_sklearn()`.

Se este arquivo for coletado num ambiente sem as duas libs instaladas (por
exemplo, o ambiente legado da linha DINO, que usa `requirements.txt` na raiz
e nao tem motivo pra ter `firthlogist`/`interpret`), os testes abaixo sao
PULADOS, nao falham -- isso nao e regressao do restante da suite, e um
ambiente diferente por desenho (ver docs/METODO.md §8).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

firthlogist = pytest.importorskip(
    "firthlogist", reason="ambiente sem firthlogist -- ver docs/METODO.md §8"
)
interpret = pytest.importorskip(
    "interpret", reason="ambiente sem interpret -- ver docs/METODO.md §8"
)

ROOT = Path(__file__).resolve().parents[1]
COMPAT_PATH = ROOT / "scripts" / "treino" / "susc_ambiente_compat_common.py"

SEED = 20260805  # data em que o ambiente foi verificado pela primeira vez


def _load_compat_module():
    spec = importlib.util.spec_from_file_location("susc_ambiente_compat_common", COMPAT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _dados_sinteticos(n: int, n_features: int = 5):
    """Dado 100% sintetico, sem nenhum dado real do projeto -- so pra testar
    se o motor de treino roda, nao pra medir desempenho de modelo nenhum."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(n, n_features))
    y = (X[:, 0] * 0.4 - X[:, 1] * 0.14 + rng.normal(size=n) * 1.5 > 0).astype(int)
    return X, y


def test_python_versao_compativel():
    """firthlogist/interpret exigem Python <3.11 -- confirmado real em 05/08/2026."""
    assert sys.version_info < (3, 11), (
        f"Python {sys.version_info.major}.{sys.version_info.minor} detectado -- "
        "firthlogist/interpret nao tem versao publicada compativel com >=3.11 "
        "(confirmado em 05/08/2026). Use o ambiente revp-susc (environment.yml)."
    )


def test_shim_firthlogist_e_idempotente():
    compat = _load_compat_module()
    primeira_chamada = compat.aplicar_shim_firthlogist_sklearn()
    segunda_chamada = compat.aplicar_shim_firthlogist_sklearn()
    # a segunda chamada nunca deve reaplicar o shim (o metodo ja existe agora)
    assert segunda_chamada is False
    assert isinstance(primeira_chamada, bool)


def test_firthlogist_treina_com_shim_aplicado():
    compat = _load_compat_module()
    compat.aplicar_shim_firthlogist_sklearn()
    from firthlogist import FirthLogisticRegression

    X, y = _dados_sinteticos(n=200)
    modelo = FirthLogisticRegression()
    modelo.fit(X, y)

    assert modelo.coef_.shape == (5,)
    assert np.all(np.isfinite(modelo.coef_))


def test_ebm_treina_no_tamanho_real_do_dataset_de_curitiba():
    """N=1458 e o tamanho real do dataset primario de Curitiba (SUSC-20N) --
    usado aqui so como escala de referencia pro benchmark. Dado 100%
    sintetico, sem nenhuma linha real do projeto."""
    from interpret.glassbox import ExplainableBoostingClassifier

    X, y = _dados_sinteticos(n=1458)
    modelo = ExplainableBoostingClassifier(random_state=SEED)
    modelo.fit(X, y)

    assert sorted(modelo.classes_.tolist()) == [0, 1]


def test_diagnostico_ambiente_nao_levanta_excecao():
    compat = _load_compat_module()
    info = compat.verificar_ambiente_treino_susc()

    assert info["python_compativel_firthlogist_interpret"] is True
    assert info["firthlogist_importa"] is True
    assert info["interpret_importa"] is True
