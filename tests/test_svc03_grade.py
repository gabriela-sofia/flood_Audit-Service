"""Testes da grade de suscetibilidade (E5/M4 -- SVC-03).

O que estes testes protegem:

um mapa e o artefato mais facil de acreditar e o mais dificil de auditar: ele
parece resultado mesmo quando e extrapolacao. Os modos de falha que importam:

  * o mapa deixar de ser a mesma coisa que o contrato responde. Se a grade
    calcular o escore por um caminho e `inferir()` por outro, o produto passa a
    ter duas verdades. O script confere celulas sorteadas contra o contrato, e
    este teste guarda a conferencia;
  * preencher celula invalida em vez de descartar -- borda do raster, corpo
    d'agua, buraco de derivacao viram numero plausivel;
  * a chuva aparecer como se variasse no espaco. Em Recife ela e cenario
    declarado, e o mapa ordena terreno;
  * Petropolis receber escore sem que a maturidade diga que ali nao existe
    nenhum ponto rotulado para verificar.

Os artefatos vivem em `local_runs/`, git-ignored: ausencia vira skip.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "local_runs" if (ROOT / "local_runs").is_dir() else ROOT / "modelo" / "execucoes"
OUT = RUNS / "svc-03-grade"

sys.path.insert(0, str(ROOT / "scripts" / "servico"))
from svc03_grade_suscetibilidade import (  # noqa: E402
    CENARIOS_CHUVA, RASTER_DE, REGIOES, distancia_de_dominio,
)

GERAR = "python scripts/servico/svc03_grade_suscetibilidade.py"


def _exige(p: Path) -> None:
    if not p.exists():
        pytest.skip(f"{p.relative_to(ROOT)} ausente -- gere com: {GERAR}")


@pytest.fixture(scope="module")
def resultado() -> dict:
    _exige(OUT / "resultado.json")
    return json.loads((OUT / "resultado.json").read_text(encoding="utf-8"))


def _grades(resultado: dict) -> list[dict]:
    return [r for r in resultado["regioes"] if r.get("grade_gerada")]


# ------------------------------------------------------ desenho da grade

def test_as_tres_regioes_sao_processadas(resultado):
    assert {r["regiao"] for r in resultado["regioes"]} == set(REGIOES)


def test_toda_grade_declara_maturidade(resultado):
    for r in _grades(resultado):
        assert r["maturidade"] in (
            "validado", "mvp_local", "transferencia_caracterizada",
            "transferencia_sem_referencia_local"), r["regiao"]


def test_nenhuma_regiao_do_projeto_esta_validada(resultado):
    """Estado de hoje. Se mudar, e porque apareceu inventario -- leia antes."""
    for r in _grades(resultado):
        assert r["maturidade"] != "validado", r["regiao"]


def test_petropolis_declara_que_nao_tem_referencia_local(resultado):
    p = [r for r in resultado["regioes"] if r["regiao"] == "petropolis"]
    assert p, "petropolis ausente do resultado"
    p = p[0]
    if not p.get("grade_gerada"):
        pytest.skip("petropolis sem grade nesta execucao")
    assert p["maturidade"] == "transferencia_sem_referencia_local"


def test_curitiba_e_petropolis_nao_tem_a_mesma_maturidade(resultado):
    """A diferenca e o ponto: numa da para verificar depois, na outra nao."""
    m = {r["regiao"]: r["maturidade"] for r in _grades(resultado)}
    if "curitiba" in m and "petropolis" in m:
        assert m["curitiba"] != m["petropolis"]


def test_celula_invalida_e_descartada_e_contada(resultado):
    for r in _grades(resultado):
        assert r["celulas_validas"] <= r["celulas_da_grade"]
        assert 0 < r["pct_validas"] <= 100
        assert r["celulas_validas"] > 0, r["regiao"]


def test_a_grade_nao_inventa_resolucao(resultado):
    """A grade e subamostragem do pixel de 30 m, nunca reamostragem."""
    assert resultado["resolucao_m"] == 30 * resultado["passo"]
    for r in _grades(resultado):
        assert r["resolucao_m"] == resultado["resolucao_m"]


# ---------------------------------------------------- o mapa e o contrato

def test_o_mapa_bate_com_o_que_o_contrato_responde(resultado):
    """Se divergir, o produto passou a ter duas verdades."""
    for r in _grades(resultado):
        c = r["conferencia_contra_o_contrato"]
        assert c["celulas"] > 0, r["regiao"]
        assert c["divergencias"] == 0, (
            f"{r['regiao']}: {c['divergencias']} de {c['celulas']} celulas "
            "divergem entre a grade e inferir()")


# ------------------------------------------------------------ chuva/cenario

def test_chuva_entra_como_cenario_e_nunca_como_camada(resultado):
    for r in _grades(resultado):
        for v in r["variaveis_por_cenario"]:
            assert v not in RASTER_DE, (
                f"{r['regiao']}: {v} nao pode sair de raster nesta grade")


def test_so_recife_usa_cenario_de_chuva(resultado):
    for r in _grades(resultado):
        if r["regiao"] == "recife":
            assert set(r["cenarios"]) == set(CENARIOS_CHUVA)
        else:
            assert set(r["cenarios"]) == {"unico"}


def test_cenario_de_mais_chuva_nao_baixa_o_escore(resultado):
    """Sinal fisico do forcamento, medido na saida e nao no coeficiente."""
    rec = [r for r in _grades(resultado) if r["regiao"] == "recife"]
    if not rec:
        pytest.skip("recife sem grade nesta execucao")
    c = rec[0]["cenarios"]
    ordem = ["mediana_observada", "p90_observado", "maximo_observado"]
    medianas = [c[k]["escore_mediano"] for k in ordem if k in c]
    assert medianas == sorted(medianas), (
        f"escore mediano nao cresce com o cenario de chuva: {medianas}")


# ------------------------------------------------- distancia de dominio

def test_toda_grade_declara_distancia_de_dominio(resultado):
    """Entregavel explicito do E5."""
    for r in _grades(resultado):
        assert set(r["distancia_de_dominio"]) == set(r["variaveis_do_modelo"])
        for f, d in r["distancia_de_dominio"].items():
            assert 0 <= d["pct_dentro_da_faixa"] <= 100
            assert "diferenca_padronizada" in d


def test_distancia_de_dominio_e_zero_quando_a_grade_e_o_ajuste():
    import numpy as np

    modelo = {
        "features": ["hand_m"],
        "padronizacao": {"hand_m": {"media": 5.0, "desvio": 2.0}},
        "faixa_dominio_5_95": {"hand_m": [1.0, 9.0]},
    }
    X = np.array([[5.0], [5.0], [5.0]])
    d = distancia_de_dominio(X, modelo)["hand_m"]
    assert d["diferenca_padronizada"] == 0.0
    assert d["pct_dentro_da_faixa"] == 100.0


def test_curitiba_extrapola_elevacao_na_grade(resultado):
    """Achado que o portao de dominio existe para pegar; medido no territorio.

    Se deixar de valer, `metodo_aplicacao_sem_rotulo_local_v1.md` e o texto do
    servico precisam mudar junto.
    """
    cur = [r for r in _grades(resultado) if r["regiao"] == "curitiba"]
    if not cur or "elevation_m" not in cur[0]["distancia_de_dominio"]:
        pytest.skip("curitiba sem grade ou sem elevacao no modelo")
    d = cur[0]["distancia_de_dominio"]["elevation_m"]
    assert d["pct_dentro_da_faixa"] < 50.0, d


# --------------------------------------------------------- artefatos

def test_cada_grade_gera_csv_raster_e_figura(resultado):
    for r in _grades(resultado):
        for sufixo in (f"grade_{r['regiao']}.csv", f"escore_{r['regiao']}.tif",
                       f"escore_{r['regiao']}.png"):
            assert (OUT / sufixo).exists(), sufixo


def test_csv_da_grade_tem_escore_entre_zero_e_um(resultado):
    """Celula recusada pelo portao fica vazia; celula servivel tem escore valido."""
    for r in _grades(resultado):
        p = OUT / f"grade_{r['regiao']}.csv"
        _exige(p)
        d = pd.read_csv(p)
        colunas = [c for c in d.columns if c.startswith("escore")]
        assert colunas, r["regiao"]
        for c in colunas:
            servivel = d[c].notna()
            assert servivel.any(), f"{r['regiao']}/{c}: nenhuma celula servivel"
            assert d.loc[servivel, c].between(0, 1).all(), f"{r['regiao']}/{c}"
        assert len(d) == r["celulas_validas"]


def test_celula_recusada_pelo_dominio_fica_vazia_e_nao_zerada(resultado):
    """O que o servico recusaria nao pode virar escore baixo no mapa.

    Zero e um escore -- significa "muito pouco suscetivel". Recusa nao e isso:
    e "nao sei falar sobre esta celula". Confundir os dois pinta de seguro o
    lugar sobre o qual o modelo nao tem o que dizer.
    """
    for r in _grades(resultado):
        d = pd.read_csv(OUT / f"grade_{r['regiao']}.csv")
        principal = next(iter(r["cenarios"]))
        sufixo = "" if principal == "unico" else f"__{principal}"
        vazias = int(d[f"escore{sufixo}"].isna().sum())
        esperado = r["cenarios"][principal]["celulas_recusadas_por_dominio"]
        assert vazias == esperado, (
            f"{r['regiao']}: {vazias} celulas vazias no CSV contra "
            f"{esperado} recusadas pelo portao")


def test_csv_da_grade_tem_ic_coerente(resultado):
    for r in _grades(resultado):
        d = pd.read_csv(OUT / f"grade_{r['regiao']}.csv")
        for c in [c for c in d.columns if c.startswith("escore")]:
            sufixo = c[len("escore"):]
            lo, hi = d[f"ic95_lo{sufixo}"], d[f"ic95_hi{sufixo}"]
            ok = d[c].notna()
            assert (lo[ok] <= d.loc[ok, c]).all(), f"{r['regiao']}/{c}"
            assert (d.loc[ok, c] <= hi[ok]).all(), f"{r['regiao']}/{c}"
            assert lo[~ok].isna().all() and hi[~ok].isna().all(), (
                f"{r['regiao']}/{c}: celula sem escore com IC preenchido")


def test_o_resultado_declara_o_que_nao_e(resultado):
    assert "nao_e" in resultado
    for r in _grades(resultado):
        assert "nao_e" in r
