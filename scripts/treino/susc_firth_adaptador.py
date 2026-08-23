"""
Shim de compatibilidade entre `firthlogist` e scikit-learn >= 1.6.

PROBLEMA, ja documentado em docs/ambiente_treino_susc.md e encontrado pela
primeira vez no SUSC-20M (23/07/2026):
o `firthlogist` chama `BaseEstimator._validate_data`, API privada do
scikit-learn removida na versao 1.6. Em scikit-learn 1.7.2 o erro e

    AttributeError: 'FirthLogisticRegression' object has no attribute
    '_validate_data'

Nao e acidente de maquina: e incompatibilidade estavel entre as duas
bibliotecas, reproduzida de forma independente em 23/07/2026, 05/08/2026 e
07/08/2026, em tres ambientes diferentes.

POR QUE UM MODULO E NAO TRES LINHAS COLADAS:
ate aqui o shim era reescrito dentro de cada script que precisava dele. Isso
significa que um script novo que esqueca de coloca-lo falha em tempo de
execucao, no meio de uma rodada. Centralizar torna a correcao importavel e
auditavel -- e deixa o motivo escrito num lugar so.

USO:
    import susc_firth_adaptador  # noqa: F401  -- aplica ao importar
    from firthlogist import FirthLogisticRegression

A funcao publica `validate_data` do scikit-learn faz exatamente o que a
privada fazia; o shim so reexpoe com o nome antigo. Se um dia o firthlogist
publicar versao compativel, este modulo vira inofensivo -- ele so age quando
o atributo esta ausente.
"""

from __future__ import annotations


def aplicar() -> bool:
    """Instala o alias se necessario. Devolve True se precisou agir."""
    from sklearn.base import BaseEstimator

    if hasattr(BaseEstimator, "_validate_data"):
        return False

    try:
        from sklearn.utils.validation import validate_data
    except ImportError:  # scikit-learn muito antigo: nada a fazer
        return False

    def _validate_data(self, X="no_validation", y="no_validation", **kwargs):
        return validate_data(self, X=X, y=y, **kwargs)

    BaseEstimator._validate_data = _validate_data
    return True


APLICADO = aplicar()
