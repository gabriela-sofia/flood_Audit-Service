"""Compatibilidade entre `firthlogist` e versoes recentes do scikit-learn.

Por que este modulo existe
---------------------------
`firthlogist` chama `self._validate_data(...)`, um metodo "privado" (prefixo
`_`) da `BaseEstimator` do scikit-learn. Esse metodo foi removido na 1.6 e
substituido pela funcao publica `sklearn.utils.validation.validate_data`.
`firthlogist` (ate a versao 0.5.0, a mais recente no PyPI em 05/08/2026) nunca
foi atualizado pra essa mudanca -- toda chamada a `.fit()` levanta
`AttributeError: 'FirthLogisticRegression' object has no attribute
'_validate_data'` em qualquer ambiente com scikit-learn >=1.6.

Isso ja tinha sido encontrado e documentado no relatorio do SUSC-20M
(23/07/2026), rodando num ambiente com Python 3.12 / scikit-learn 1.8,
resolvido la com um shim de 3 linhas escrito direto no script daquela rodada.
Reproduzido de forma independente em 05/08/2026 (Python 3.10.12, scikit-learn
1.7.2, sandbox de verificacao de ambiente) -- ou seja, nao e uma falha
especifica de uma maquina, e uma incompatibilidade real e estavel entre as
duas bibliotecas. Este modulo existe pra nao copiar e colar esse patch em
cada script novo que use `firthlogist`.

O que NAO fazer
----------------
Nao usar isto pra "consertar" nenhum outro erro do `firthlogist`. E um patch
cirurgico pra um metodo especifico (`_validate_data`), nao um catch-all de
compatibilidade. Se aparecer outro erro de incompatibilidade, documentar
separadamente -- nao esticar este shim pra cobrir coisas novas sem prova de
que o resultado numerico continua identico ao publicado (mesmo criterio ja
usado no SUSC-20M: comparar coeficiente, p-valor e IC contra o que ja foi
publicado, diferenca maxima tolerada 1e-3).
"""
from __future__ import annotations

import sys

import sklearn.base


def aplicar_shim_firthlogist_sklearn() -> bool:
    """Restaura `BaseEstimator._validate_data` se ela tiver sido removida.

    Retorna ``True`` se o shim foi aplicado agora (ou seja, o metodo estava
    ausente e precisou ser recriado), ``False`` se ja nao era necessario
    (metodo ja existe nesta versao do scikit-learn, ou o shim ja tinha sido
    aplicado antes nesta mesma sessao Python). Nunca levanta excecao por
    conta propria -- se `validate_data` tambem nao existir (versao de
    scikit-learn muito diferente das testadas), o `ImportError` sobe puro,
    sem mascarar o problema.
    """
    if hasattr(sklearn.base.BaseEstimator, "_validate_data"):
        return False

    from sklearn.utils.validation import validate_data

    def _validate_data_shim(self, X, y=None, **kwargs):
        return validate_data(self, X, y, **kwargs)

    sklearn.base.BaseEstimator._validate_data = _validate_data_shim
    return True


def verificar_ambiente_treino_susc() -> dict:
    """Retrato real do ambiente atual pra diagnostico rapido.

    Nao levanta excecao por biblioteca faltando -- erros de import viram
    string no proprio dicionario, pra dar pra chamar isso de um script de
    diagnostico (ou de um teste) sem quebrar a execucao inteira.
    """
    info: dict[str, object] = {
        "python_version": sys.version.split()[0],
        "python_compativel_firthlogist_interpret": sys.version_info < (3, 11),
    }

    try:
        import sklearn

        info["sklearn_version"] = sklearn.__version__
    except Exception as exc:  # pragma: no cover - diagnostico, nao logica de negocio
        info["sklearn_version"] = f"erro: {exc}"

    info["shim_firthlogist_aplicado_agora"] = aplicar_shim_firthlogist_sklearn()

    try:
        import firthlogist  # noqa: F401

        info["firthlogist_importa"] = True
    except Exception as exc:  # pragma: no cover - diagnostico, nao logica de negocio
        info["firthlogist_importa"] = f"erro: {exc}"

    try:
        import interpret
        from interpret.glassbox import ExplainableBoostingClassifier  # noqa: F401

        info["interpret_importa"] = True
        info["interpret_version"] = interpret.__version__
    except Exception as exc:  # pragma: no cover - diagnostico, nao logica de negocio
        info["interpret_importa"] = f"erro: {exc}"

    return info


if __name__ == "__main__":
    import json

    print(json.dumps(verificar_ambiente_treino_susc(), indent=2, ensure_ascii=False))
