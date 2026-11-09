"""Contagens oficiais nao viram exposicao, zeros ou rotulos fisicos."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/social"))
import social01_censo_municipal as soc


@pytest.fixture
def fonte():
    meta = {"id": 6805, "nivelTerritorial": {"Administrativo": ["N6"]}, "classificacoes": [
        {"id": 11558, "categorias": [{"id": 1, "nome": "Total", "nivel": 0},
          {"id": 2, "nome": "Rede", "nivel": 1}, {"id": 3, "nome": "Outra", "nivel": 1},
          {"id": 4, "nome": "Subcategoria da rede", "nivel": 2}]}]}
    resultados = [{"classificacoes": [{"id": "11558", "categoria": {str(c): nome}}],
                   "series": [{"localidade": {"id": "2611606", "nome": "Recife (PE)", "nivel": {"id": "N6"}},
                               "serie": {"2022": valor}}]}
                  for c, nome, valor in ((1, "Total", "100"), (2, "Rede", "60"),
                                         (3, "Outra", "40"), (4, "Subcategoria", "50"))]
    return meta, [{"id": "381", "unidade": "Unidades", "resultados": resultados}]


def test_hierarquia_nao_soma_subcategoria(fonte):
    linhas, controle = soc.normalizar(*fonte, ("2611606",))
    assert controle[0]["soma_primeiro_nivel_confere"]
    assert linhas[1]["percentual_domicilios"] == 60


def test_ausencia_nao_vira_zero(fonte):
    fonte[1][0]["resultados"][2]["series"][0]["serie"]["2022"] = "..."
    linhas, controle = soc.normalizar(*fonte, ("2611606",))
    assert linhas[2]["domicilios"] is None
    assert linhas[2]["percentual_domicilios"] is None
    assert not controle[0]["categorias_completas"]


def test_total_divergente_recusa(fonte):
    fonte[1][0]["resultados"][0]["series"][0]["serie"]["2022"] = "90"
    with pytest.raises(ValueError, match="nao fecham"):
        soc.normalizar(*fonte, ("2611606",))


def test_serie_duplicada_recusa(fonte):
    fonte[1][0]["resultados"].append(copy.deepcopy(fonte[1][0]["resultados"][0]))
    with pytest.raises(ValueError, match="duplicada"):
        soc.normalizar(*fonte, ("2611606",))


def test_snapshot_repetido_e_adulteracao(fonte, tmp_path, monkeypatch):
    monkeypatch.setattr(soc, "obter_json", lambda url: fonte[0] if url.endswith("metadados") else fonte[1])
    arq = soc.executar(tmp_path, ("2611606",))
    texto = arq.read_text(encoding="utf-8")
    assert soc.executar(tmp_path, ("2611606",)) == arq
    assert arq.read_text(encoding="utf-8") == texto
    r = json.loads(texto)
    assert not r["usado_no_modelo_fisico"]
    assert not r["promovido"]
    (arq.parent / "indicadores.csv").write_text("alterado", encoding="utf-8")
    with pytest.raises(RuntimeError, match="adulterado"):
        soc.executar(tmp_path, ("2611606",))
