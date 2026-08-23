"""Testes do ajuste por classe de relevo (E3/M2 -- MOD-SERRA-03).

O que estes testes protegem:

o risco desta etapa e afirmar "o modelo funciona em serra" a partir de um ajuste
que a propria regra do projeto nao autoriza. Dois modos de falha concretos:

  * rodar mais variaveis do que o estrato comporta. O estrato ingreme tem 19
    grupos positivos; com 4 variaveis o EPV cai para 1,9 e o coeficiente vira
    ruido com aparencia de resultado;
  * ler o orcamento pela contagem de grupos totais, que e o que a v1 fazia e o
    que faz 24 grupos parecerem suficientes para 2 variaveis quando so 19 tem
    positivo.

Os artefatos vivem em `local_runs/`, git-ignored: ausencia vira skip com o
comando que os gera, nao falha.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "local_runs" if (ROOT / "local_runs").is_dir() else ROOT / "modelo" / "execucoes"
OUT = RUNS / "mod-serra-03"

sys.path.insert(0, str(ROOT / "scripts" / "treino"))
from mod_serra03_relevo_ds05 import (  # noqa: E402
    AUC_MAX, AUC_MIN, CONJUNTOS, EPV_MINIMO, SINAL_EXIGIDO, conferir,
    orcamento,
)

GERAR = "python scripts/treino/mod_serra03_relevo_ds05.py"


def _exige(p: Path) -> None:
    if not p.exists():
        pytest.skip(f"{p.relative_to(ROOT)} ausente -- gere com: {GERAR}")


@pytest.fixture(scope="module")
def resultado() -> dict:
    _exige(OUT / "resultado.json")
    return json.loads((OUT / "resultado.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def coeficientes() -> pd.DataFrame:
    _exige(OUT / "coeficientes_por_relevo.csv")
    return pd.read_csv(OUT / "coeficientes_por_relevo.csv")


# --------------------------------------------------- leituras do orcamento

def test_as_duas_leituras_do_orcamento_divergem_quando_a_classe_e_rara():
    """O caso do estrato ingreme, reduzido ao minimo."""
    s = pd.DataFrame({
        "grupo_cv": [f"g{i}" for i in range(24)],
        "classe": [1] * 19 + [0] * 5,
    })
    assert orcamento(s, "LEITURA_PRECEDENTE") == 2   # 24 grupos / 10
    assert orcamento(s, "LEITURA_ESTRITA") == 0      # 5 grupos da classe rara


def test_orcamento_e_zero_quando_falta_uma_classe():
    s = pd.DataFrame({"grupo_cv": [f"g{i}" for i in range(50)],
                      "classe": [1] * 50})
    for leitura in ("LEITURA_PRECEDENTE", "LEITURA_ESTRITA"):
        assert orcamento(s, leitura) == 0


def test_leitura_estrita_nunca_e_mais_permissiva_que_a_precedente():
    s = pd.DataFrame({"grupo_cv": [f"g{i}" for i in range(200)],
                      "classe": [1] * 130 + [0] * 70})
    assert (orcamento(s, "LEITURA_ESTRITA")
            <= orcamento(s, "LEITURA_PRECEDENTE"))


# ------------------------------------------------------ conferencia de sinal

def test_sinal_invertido_de_hand_e_reprovado():
    falhas = conferir({"hand_m": +0.9}, {"hand_m": [0.4, 1.4]}, 0.78)
    assert any("sinal de hand_m" in f for f in falhas)


def test_ic_que_cruza_zero_e_reprovado():
    falhas = conferir({"hand_m": -0.9}, {"hand_m": [-1.4, 0.2]}, 0.78)
    assert any("cruza zero" in f for f in falhas)


def test_auc_fora_da_faixa_e_reprovada():
    assert conferir({"hand_m": -1.0}, {"hand_m": [-1.5, -0.5]}, 0.62)
    assert conferir({"hand_m": -1.0}, {"hand_m": [-1.5, -0.5]}, 0.97)
    assert not conferir({"hand_m": -1.0}, {"hand_m": [-1.5, -0.5]}, 0.78)


def test_auc_muito_alta_e_lida_como_vazamento():
    falhas = conferir({"hand_m": -1.0}, {"hand_m": [-1.5, -0.5]}, 0.98)
    assert any("vazamento" in f for f in falhas)


def test_sinal_exigido_cobre_as_duas_causais():
    assert SINAL_EXIGIDO == {"hand_m": -1, "twi_dinf": +1}


# ----------------------------------------------------- artefatos gravados

def test_nenhum_ajuste_ultrapassa_a_leitura_precedente(resultado):
    for nome, e in resultado["estratos"].items():
        for conj, c in e.get("conjuntos", {}).items():
            if not c.get("rodado"):
                continue
            assert len(c["features"]) <= e["orcamento"]["LEITURA_PRECEDENTE"], (
                f"{nome}/{conj} rodou acima do orcamento das duas leituras")


def test_todo_ajuste_declara_em_qual_leitura_cabe(resultado):
    for nome, e in resultado["estratos"].items():
        for conj, c in e.get("conjuntos", {}).items():
            if not c.get("rodado"):
                continue
            assert set(c["cabe"]) == {"LEITURA_PRECEDENTE", "LEITURA_ESTRITA"}
            assert any(c["cabe"].values()), f"{nome}/{conj}"


def test_estrato_ingreme_comporta_uma_variavel_na_leitura_estrita(resultado):
    """Achado que muda o que se pode afirmar: 19 grupos positivos, nao 24.

    Se este numero mudar, `ext_modelo_de_encosta_v2.md` precisa mudar junto.
    """
    ing = resultado["estratos"].get("INGREME")
    if ing is None:
        pytest.skip("estrato ingreme ausente nesta execucao")
    assert ing["grupos_positivos"] == 19
    assert ing["orcamento"]["LEITURA_ESTRITA"] == 1
    assert ing["orcamento"]["LEITURA_PRECEDENTE"] == 2


def test_ajuste_de_uma_variavel_cabe_nas_duas_leituras_nos_dois_estratos(resultado):
    for nome in ("INGREME", "PLANO_OU_ONDULADO"):
        e = resultado["estratos"].get(nome)
        if e is None or "CAUSAL_1" not in e.get("conjuntos", {}):
            continue
        assert e["conjuntos"]["CAUSAL_1"]["cabe"]["LEITURA_ESTRITA"], nome


def test_os_sinais_fisicos_estao_corretos_em_todo_ajuste_rodado(coeficientes):
    for _, r in coeficientes.iterrows():
        if r.feature not in SINAL_EXIGIDO:
            continue
        esperado = SINAL_EXIGIDO[r.feature]
        assert (r.coef < 0) == (esperado < 0), (
            f"{r.estrato}/{r.conjunto}/{r.feature} com sinal {r.coef}")


def test_nenhum_ic_das_causais_cruza_zero(coeficientes):
    for _, r in coeficientes.iterrows():
        if r.feature not in SINAL_EXIGIDO:
            continue
        if pd.isna(r.ic95_lo) or pd.isna(r.ic95_hi):
            continue
        assert not (r.ic95_lo <= 0 <= r.ic95_hi), (
            f"{r.estrato}/{r.conjunto}/{r.feature}: IC [{r.ic95_lo}; {r.ic95_hi}]")


def test_toda_auc_gravada_esta_na_faixa_declarada(coeficientes):
    for auc in coeficientes.auc_cv.dropna().unique():
        assert AUC_MIN <= auc <= AUC_MAX, auc


def test_transferencia_da_planicie_para_a_serra_nao_colapsa(resultado):
    """Achado protegido: o modelo de planicie funciona em serra sem ter visto.

    Se deixar de valer, a afirmacao de que a relacao HAND/TWI e a mesma nos dois
    terrenos deixa de se sustentar e o texto tem de mudar.
    """
    t = resultado.get("transferencia", {})
    if not t:
        pytest.skip("transferencia ausente nesta execucao")
    for conj, d in t.items():
        auc = d["planicie_para_serra"]["auc"]
        assert auc is not None and auc >= AUC_MIN, f"{conj}: {auc}"


def test_estrato_sem_as_duas_classes_fica_fora_do_ajuste(resultado):
    nc = resultado["estratos"].get("NAO_CLASSIFICADO")
    if nc is None:
        pytest.skip("estrato NAO_CLASSIFICADO ausente")
    assert not nc.get("conjuntos")
    assert "motivo" in nc


def test_conjuntos_declarados_sao_os_que_o_script_conhece(resultado):
    conhecidos = set(CONJUNTOS)
    for e in resultado["estratos"].values():
        assert set(e.get("conjuntos", {})).issubset(conhecidos)
    assert resultado["epv_minimo"] == EPV_MINIMO
