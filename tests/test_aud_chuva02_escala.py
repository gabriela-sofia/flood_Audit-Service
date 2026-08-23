"""Testes da auditoria de escala da chuva (AUD-CHUVA-02).

O que estes testes protegem:

o risco aqui nao e o script quebrar -- e ele devolver um veredito que parece
severo e nao mede nada. Dois modos de falha concretos:

  * confundir "a chuva nao varia dentro do grupo" com "o grupo tem um ponto
    so". Em Recife e Curitiba o grupo E o ponto; dizer que falta variacao
    intra-grupo ali seria descrever a unidade de validacao, nao a chuva. Foi
    exatamente o que a primeira versao do script fazia, e por isso o veredito
    virou dois.
  * medir AUC restrita a datas compartilhadas sem dizer sobre quantos pontos.
    Em Recife sao 10 pontos: o numero existe, a leitura nao.

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
OUT = RUNS / "aud-chuva-02"

sys.path.insert(0, str(ROOT / "scripts" / "treino"))
from aud_chuva02_escala_do_contraste import (  # noqa: E402
    FAIXA_ACASO, PCT_DATAS_COMPARTILHADAS, VARIAVEIS, veredito_espacial,
    veredito_temporal,
)

GERAR = "python scripts/treino/aud_chuva02_escala_do_contraste.py"


def _exige(p: Path) -> None:
    if not p.exists():
        pytest.skip(f"{p.relative_to(ROOT)} ausente -- gere com: {GERAR}")


@pytest.fixture(scope="module")
def tabela() -> pd.DataFrame:
    _exige(OUT / "escala_do_contraste_por_fonte.csv")
    return pd.read_csv(OUT / "escala_do_contraste_por_fonte.csv")


@pytest.fixture(scope="module")
def auditoria() -> dict:
    _exige(OUT / "auditoria_escala_chuva.json")
    return json.loads((OUT / "auditoria_escala_chuva.json").read_text(
        encoding="utf-8"))


# ------------------------------------------------------ logica do veredito

def test_grupo_de_um_ponto_nao_e_lido_como_falta_de_variacao():
    """A distincao que a primeira versao do script errava."""
    sem_uso = {"grupos_usaveis": 0, "auc_media_dentro_do_grupo": None}
    assert veredito_espacial(sem_uso, pontos_por_grupo=1.0) == "GRUPO_E_O_PONTO"
    assert veredito_espacial(sem_uso, pontos_por_grupo=18.6) == "GRUPOS_PUROS"


def test_auc_dentro_da_faixa_de_acaso_vira_ruido():
    for a in (FAIXA_ACASO[0], 0.5, FAIXA_ACASO[1]):
        d = {"grupos_usaveis": 40, "auc_media_dentro_do_grupo": a}
        assert veredito_espacial(d, 20.0) == "RUIDO_NA_ESCALA_DO_MODELO"


def test_auc_fora_da_faixa_vira_sinal():
    for a in (0.44, 0.62):
        d = {"grupos_usaveis": 40, "auc_media_dentro_do_grupo": a}
        assert veredito_espacial(d, 20.0) == "SINAL_NA_ESCALA_DO_MODELO"


def test_limiar_temporal_separa_disjunto_de_compartilhado():
    abaixo = {"pct_pontos_em_datas_compartilhadas": PCT_DATAS_COMPARTILHADAS - 0.1}
    acima = {"pct_pontos_em_datas_compartilhadas": PCT_DATAS_COMPARTILHADAS}
    assert veredito_temporal(abaixo) == "DATAS_DISJUNTAS"
    assert veredito_temporal(acima) == "DATAS_COMPARTILHADAS"


# ------------------------------------------------------ artefatos gravados

def test_toda_fonte_da_base_foi_auditada(auditoria):
    esperadas = {"cems", "curitiba", "recife", "sen1floods11", "ufo", "uk"}
    assert esperadas.issubset(set(auditoria["fontes"])), (
        "fonte da tabela unica ficou de fora da auditoria de escala")


def test_toda_fonte_usa_o_mesmo_produto_de_precipitacao(auditoria):
    produtos = {p for f in auditoria["fontes"].values()
                for p in f["fonte_chuva"]}
    assert produtos == {"open_meteo_era5_land"}, (
        f"a base deixou de ter produto unico: {produtos}")


def test_as_duas_variaveis_de_chuva_sao_auditadas(auditoria):
    for fonte, d in auditoria["fontes"].items():
        assert set(d["variaveis"]) == set(VARIAVEIS), fonte


def test_todo_par_declara_os_dois_vereditos(tabela):
    assert tabela.veredito_espacial.notna().all()
    assert tabela.veredito_temporal.notna().all()
    assert len(tabela) == 12, "6 fontes x 2 variaveis"


def test_auc_restrita_declara_o_n_junto(tabela):
    """Sem o n, 0,46 em Recife pareceria comparavel a 0,61 em Curitiba."""
    com_auc = tabela[tabela.auc_datas_compartilhadas.notna()]
    assert (com_auc.pct_datas_compartilhadas >= 0).all()
    for _, r in com_auc.iterrows():
        assert r.n > 0


def test_recife_e_a_fonte_com_datas_disjuntas(tabela):
    """Achado, nao expectativa: se mudar, o texto do projeto tem de mudar junto.

    Recife tem 5 datas com as duas classes em 205 -- 3,7% dos pontos. Enquanto
    for assim, a chuva de Recife separa dias, nao lugares, e nenhum texto pode
    ler o coeficiente de chuva como propriedade do terreno.
    """
    disjuntas = set(tabela.loc[tabela.veredito_temporal == "DATAS_DISJUNTAS",
                               "fonte"])
    assert disjuntas == {"recife"}, (
        f"o conjunto de fontes com datas disjuntas mudou: {disjuntas}. "
        "Reveja ext_chuva_estado_do_projeto_v1.md antes de seguir")


def test_nenhuma_fonte_mostra_sinal_de_chuva_na_escala_do_modelo(tabela):
    """Estado de hoje, protegido para que uma melhora nao passe despercebida.

    Se algum dia uma fonte sair para SINAL_NA_ESCALA_DO_MODELO, este teste
    falha -- e isso e uma boa noticia que precisa ser lida, nao um alarme.
    """
    com_sinal = tabela[tabela.veredito_espacial == "SINAL_NA_ESCALA_DO_MODELO"]
    assert com_sinal.empty, (
        "a chuva passou a discriminar dentro do grupo em "
        f"{list(com_sinal.fonte)} -- atualize a leitura do projeto")
