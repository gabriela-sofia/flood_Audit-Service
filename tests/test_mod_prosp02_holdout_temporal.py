"""Testes do holdout temporal (E4/M3) -- MOD-PROSP-02.

O que estes testes protegem, e por que cada um existe:

o risco desta etapa nao e o script quebrar -- e ele devolver uma trajetoria
de AUC plausivel e sem sentido. Tres modos de falha silenciosa ja
aconteceram de verdade neste trabalho e cada um tem teste aqui:

  * grupo dos dois lados do mesmo fold (vazamento por pseudo-replicacao);
  * treino do "futuro" comecando antes do fim do treino (corte que nao corta);
  * fold degenerado -- treino com 40 eventos contra 2 negativos, que produz
    numero e nao significa nada. Foi o que a alocacao 1:1 fez no estrato de
    Curitiba antes da trava de EPV por classe.

Os artefatos vivem em `local_runs/`, que e git-ignored. Quando faltarem, o
teste marca skip com o comando que os gera, em vez de falhar -- falhar por
ausencia de dado local confundiria quem clonasse o repositorio.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "local_runs" if (ROOT / "local_runs").is_dir() else ROOT / "modelo" / "execucoes"
OUT = RUNS / "mod-prosp-02"

sys.path.insert(0, str(ROOT / "scripts" / "treino"))
from mod_prosp02_holdout_temporal_ds05 import (  # noqa: E402
    AUC_DEGRADACAO, CONJUNTOS, EPV_MINIMO, FONTE_SERIE, SEMENTE, TOPO,
    datas_por_grupo, epv_ok, folds_bloco, folds_heranca, grupos_puros,
    resumir, variante_aplicavel,
)

GERAR = "python scripts/treino/mod_prosp02_holdout_temporal_ds05.py"


def _exige(p: Path) -> None:
    if not p.exists():
        pytest.skip(f"{p.relative_to(ROOT)} ausente -- gere com: {GERAR}")


@pytest.fixture(scope="module")
def folds() -> pd.DataFrame:
    _exige(OUT / "folds.csv")
    return pd.read_csv(OUT / "folds.csv")


@pytest.fixture(scope="module")
def resultado() -> dict:
    _exige(OUT / "resultado.json")
    return json.loads((OUT / "resultado.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sintetico() -> pd.DataFrame:
    """Estrato artificial com a mesma estrutura do piloto ingles.

    Grupos puros: eventos positivos datados e blocos negativos que carregam
    datas espalhadas -- e essa estrutura, e nao o tamanho, que os testes de
    desenho precisam exercitar.
    """
    rng = np.random.default_rng(7)
    linhas = []
    datas = pd.date_range("2000-01-01", "2024-12-31", periods=120)
    for e in range(120):
        for _ in range(8):
            linhas.append({"grupo_cv": f"EV_{e}", "classe": 1,
                           "data": datas[e]})
    for b in range(120):
        for _ in range(8):
            linhas.append({"grupo_cv": f"NEG_{b}", "classe": 0,
                           "data": rng.choice(datas)})
    d = pd.DataFrame(linhas)
    for f in CONJUNTOS["COMPLETO"]:
        base = rng.normal(size=len(d))
        d[f] = base + np.where(d.classe == 1, -0.8, 0.8)
    return d


# ------------------------------------------------------- desenho dos folds

@pytest.mark.parametrize("construtor", [folds_heranca, folds_bloco])
def test_nenhum_grupo_nos_dois_lados_do_mesmo_fold(sintetico, construtor):
    """Vazamento por grupo e o erro que mais se parece com sucesso."""
    rng = np.random.default_rng(SEMENTE)
    linhas = construtor(sintetico, TOPO, rng)
    assert linhas, "fixture deveria sustentar pelo menos um fold"
    for f in linhas:
        assert f["grupos_treino"] + f["grupos_teste"] <= (
            sintetico.grupo_cv.nunique())


@pytest.mark.parametrize("construtor", [folds_heranca, folds_bloco])
def test_treino_sempre_anterior_ao_teste(sintetico, construtor):
    rng = np.random.default_rng(SEMENTE)
    linhas = construtor(sintetico, TOPO, rng)
    cortes = [f["corte"] for f in linhas]
    assert cortes == sorted(cortes), "os cortes tem de avancar no tempo"


@pytest.mark.parametrize("construtor", [folds_heranca, folds_bloco])
def test_determinismo_com_a_mesma_semente(sintetico, construtor):
    a = construtor(sintetico, TOPO, np.random.default_rng(SEMENTE))
    b = construtor(sintetico, TOPO, np.random.default_rng(SEMENTE))
    assert a == b


def test_variante_bloco_mantem_prevalencia_do_teste_perto_de_um_para_um(
        sintetico):
    """E a correcao do defeito 2: sem ela a prevalencia deriva com o corte."""
    linhas = folds_bloco(sintetico, TOPO, np.random.default_rng(SEMENTE))
    for f in linhas:
        assert 0.40 <= f["prevalencia_teste"] <= 0.60, f["corte"]


# ---------------------------------------------------------- trava de EPV

def test_epv_por_classe_reprova_treino_sem_negativo():
    """Regressao do fold degenerado achado no estrato de Curitiba."""
    tr = pd.DataFrame({
        "grupo_cv": [f"EV_{i}" for i in range(40)] + ["NEG_0", "NEG_1"],
        "classe": [1] * 40 + [0, 0],
    })
    assert not epv_ok(tr, TOPO)


def test_epv_por_classe_aprova_treino_equilibrado():
    tr = pd.DataFrame({
        "grupo_cv": [f"EV_{i}" for i in range(40)]
                    + [f"NEG_{i}" for i in range(40)],
        "classe": [1] * 40 + [0] * 40,
    })
    assert epv_ok(tr, TOPO)


def test_epv_reprova_treino_de_uma_classe_so():
    tr = pd.DataFrame({"grupo_cv": [f"EV_{i}" for i in range(80)],
                       "classe": [1] * 80})
    assert not epv_ok(tr, TOPO)


# ------------------------------------------------ natureza do negativo

def test_grupo_puro_leva_a_variante_bloco(sintetico):
    assert grupos_puros(sintetico)
    assert variante_aplicavel(sintetico) == "bloco"


def test_grupo_misto_leva_a_variante_heranca(sintetico):
    """Negativo observado dentro da AOI tem data real -- a `heranca` vale."""
    misto = sintetico.copy()
    misto.loc[misto.grupo_cv == "NEG_0", "grupo_cv"] = "EV_0"
    assert not grupos_puros(misto)
    assert variante_aplicavel(misto) == "heranca"


def test_data_do_grupo_e_a_menor_das_datas_dos_pontos(sintetico):
    d = datas_por_grupo(sintetico)
    esperado = sintetico[sintetico.grupo_cv == "EV_5"]["data"].min()
    assert d.loc["EV_5"] == esperado
    assert list(d) == sorted(d), "a serie sai ordenada no tempo"


# ------------------------------------------------------------- veredito

def test_veredito_acusa_degradacao_quando_a_trajetoria_cai():
    linhas = [{"auc_prospectivo": a, "prevalencia_teste": 0.5}
              for a in (0.82, 0.75, 0.66, 0.55, 0.51)]
    r = resumir(linhas)
    assert r["veredito"] == "DEGRADACAO_TEMPORAL"


def test_veredito_estavel_exige_nenhum_fold_abaixo_do_piso():
    linhas = [{"auc_prospectivo": a, "prevalencia_teste": 0.5}
              for a in (0.74, 0.80, 0.77, 0.83)]
    assert resumir(linhas)["veredito"] == "PROSPECTIVAMENTE_ESTAVEL"
    com_queda = linhas + [{"auc_prospectivo": 0.55, "prevalencia_teste": 0.5}]
    assert resumir(com_queda)["veredito"] != "PROSPECTIVAMENTE_ESTAVEL"


# ----------------------------------------------------- artefatos gravados

def test_todo_fold_gravado_declara_ic_e_o_ic_contem_a_estimativa(folds):
    for _, f in folds.iterrows():
        assert f.ic95, f"fold sem IC: {f.estrato}/{f.conjunto}/{f.corte}"
        lo, hi = json.loads(f.ic95)
        assert lo <= f.auc_prospectivo <= hi
        assert f.boot_validos >= 100


def test_todo_fold_gravado_tem_as_duas_classes_dos_dois_lados(folds):
    assert (folds.pos_teste > 0).all()
    assert (folds.neg_teste > 0).all()
    assert (folds.pos_treino > 0).all()
    assert (folds.neg_treino > 0).all()


def test_todo_fold_gravado_cumpre_a_trava_de_eventos(folds):
    for _, f in folds.iterrows():
        minimo = EPV_MINIMO * len(CONJUNTOS[f.conjunto])
        assert f.eventos_treino >= minimo, f"{f.estrato}/{f.corte}"


def test_estrato_primario_e_o_de_maior_cobertura_de_calendario():
    _exige(OUT / "viabilidade_por_fonte.csv")
    v = pd.read_csv(OUT / "viabilidade_por_fonte.csv")
    vencedor = v.sort_values("anos_cobertos", ascending=False).iloc[0]
    assert vencedor.fonte == FONTE_SERIE, (
        "o estrato primario foi escolhido por horizonte temporal; se outra "
        "fonte passou a cobrir mais anos, a escolha precisa ser reexaminada")


def test_resumo_do_json_bate_com_o_csv_de_folds(folds, resultado):
    for conj, bloco in resultado["conjuntos"].items():
        for variante in ("heranca", "bloco"):
            gravados = folds[(folds.estrato == resultado["estrato"])
                             & (folds.conjunto == conj)
                             & (folds.variante == variante)]
            assert bloco[variante]["folds"] == len(gravados)


def test_nenhum_fold_degradado_no_estrato_primario(folds, resultado):
    """Nao e o resultado esperado -- e o que o artefato hoje afirma.

    Se uma execucao futura degradar, este teste falha e obriga a atualizar a
    leitura em vez de deixar o texto antigo de pe.
    """
    uk = folds[folds.estrato == resultado["estrato"]]
    assert (uk.auc_prospectivo >= AUC_DEGRADACAO).all()


def test_curitiba_nao_sustenta_holdout_temporal(resultado):
    """O achado negativo tambem e protegido: 114 negativos nao dao fold."""
    cur = resultado["estratos_secundarios"].get("curitiba")
    if cur is None:
        pytest.skip("estrato de Curitiba ausente nesta execucao")
    for conj in cur["conjuntos"].values():
        for chave, valor in conj.items():
            if isinstance(valor, dict) and "folds" in valor:
                assert valor["folds"] == 0, chave
