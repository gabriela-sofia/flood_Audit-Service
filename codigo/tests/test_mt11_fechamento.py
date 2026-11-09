"""MT11 fechamento: reprodução da auditoria, influência em precisão completa, população efetiva e reparo posterior."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for s in ("ia", "metodo", "servico", "geo", "suscetibilidade", "terreno", "sar"):
    sys.path.insert(0, str(REPO / "scripts" / s))
import mt11_fechamento as mf  # noqa: E402
from proveniencia import sha256_arquivo  # noqa: E402


def _ler(nome):
    p = mf.SAIDA / nome
    if not p.exists():
        pytest.skip(f"{nome} ainda não gerado")
    return json.loads(p.read_text(encoding="utf-8"))


def test_auditoria_reproduz_os_nove_contrastes_e_os_hashes():
    a = _ler("auditoria_recebida.json")
    assert a["todos_hashes_conferem"] and a["coordenadora_ic_reproduzido"]
    assert a["max_dif_media"] == 0 and a["max_dif_ic"] == 0
    assert len(a["contrastes_reproduzidos"]) == 9
    assert all(v["veredito_confere"] and v["n_confere"] for v in a["contrastes_reproduzidos"].values())


def test_influencia_sem_4355_vem_dos_dados_em_precisao_completa():
    i = _ler("influencia.json")["HF_primario_sem_GFD4355"]
    assert i["n"] == 15
    assert abs(i["media"] - 0.0294184) < 5e-8
    assert abs(i["ic98_33"][0] - (-0.0017075)) < 5e-8 and abs(i["ic98_33"][1] - 0.0632024) < 5e-8
    assert i["veredito"] != "sustentado"


def test_influencia_cobre_todas_as_retiradas_das_nove_combinacoes():
    t = pd.read_csv(mf.SAIDA / "influencia_completa.csv")
    n = t.groupby(["leitura", "hipotese"]).agg(r=("unidade_retirada", "size"), n=("n_original", "first"))
    assert len(n) == 9 and (n.r == n.n).all()


def test_populacao_efetiva_distingue_materializados_de_validos():
    p = {x["leitura"]: x for x in _ler("populacao_efetiva.json")["por_leitura"]}
    assert (p["congelada_original"]["pontos_materializados_na_leitura"], p["congelada_original"]["pontos_eventos_validos_Q"]) == (65549, 65530)
    assert (p["emendada_primaria"]["pontos_materializados_na_leitura"], p["emendada_primaria"]["pontos_eventos_validos_Q"]) == (52269, 52250)
    assert p["emendada_primaria"]["eventos_validos_HF"] == 17 and p["emendada_primaria"]["componentes_HF"] == 16
    assert all(x["HF_Q_e_F_mesmos_pontos"] for x in p.values())


def test_reparo_e_posterior_e_preserva_o_resultado_original():
    r = _ler("reparo_posterior.json")
    assert "não é pré-registro" in r["natureza"]
    assert r["resultado_original"] == sha256_arquivo(mf.MT11 / "resultado_emendado.json")
    assert r["mudancas_de_eventos"]["emendada_primaria"]["saem"] == ["GFD:4503"]


def test_nomes_publicos_preservam_a_chave_antiga():
    a = _ler("adendo_v1.json")["nomes_das_leituras"]
    assert a["congelada_original"]["nome_publico"] == "agrupamento_original_com_atributos_e8"
    assert all(v["chave_preservada"] == k and "atributos_e8" in v["atributos"] for k, v in a.items())


def test_reparo_grafo_3b_sai_por_independencia_e_preserva_original():
    p = mf.SAIDA_V2 / "reparo_grafo_3b.json"
    if not p.exists():
        pytest.skip("reparo 3B ainda não gerado")
    r = json.loads(p.read_text(encoding="utf-8"))
    assert r["resultado_original"] == sha256_arquivo(mf.MT11 / "resultado_emendado.json")
    assert r["mudancas_de_eventos"]["emendada_primaria"]["saem"] == ["GFD:4355", "GFD:4503"]
    causas = {x["causa"] for x in r["evidencia_eventos_que_saem_primario"]["GFD:4355"]}
    assert causas == {"onda"}
    assert r["contrastes"]["emendada_primaria"]["HF"]["original"]["veredito"] == "sustentado"
