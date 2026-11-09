import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "metodo"))

import admissao_cientifica as adm  # noqa: E402


def _linha(**kw):
    base = dict(ponto_id="p", fonte="cems", mecanismo="FLUVIAL_ENXURRADA", classe=1,
                nivel_negativo="nao_aplicavel", classe_relevo="PLANO_OU_ONDULADO",
                precisao_data="cena_do_produto_ems", data_evento="2024-05-08",
                grupo_cv="EMSR720_AOI01", admitido=True,
                hand_m=1.0, twi_dinf=8.0, slope_deg=1.0, elevation_m=30.0)
    base.update(kw)
    return base


def _v(*linhas):
    d = pd.DataFrame([_linha(ponto_id=f"p{i}", **l) for i, l in enumerate(linhas)])
    return d, adm.validade(d)


def test_positivo_observado_datado_serve_a_todos_os_usos():
    _, v = _v({})
    assert all(v.loc[0, f"uso_{u}"] for u in adm.USOS)


def test_ausencia_de_registro_nunca_vira_negativo():
    _, v = _v(dict(fonte="sao_paulo", mecanismo="PLUVIAL_URBANO", classe=0,
                   nivel_negativo="ausencia"))
    assert v.loc[0, "uso_exploracao"]
    assert not v.loc[0, "uso_treino"]
    assert v.loc[0, "motivo_treino"].startswith("papel_nao_ensina:desconhecido")


def test_sem_dado_e_agua_permanente_ficam_fora_do_binario():
    _, v = _v(dict(classe=-1), dict(classe=9))
    assert not v.uso_treino.any()
    assert set(v.motivo_treino) == {"papel_nao_ensina:desconhecido",
                                    "papel_nao_ensina:agua_permanente"}


def test_gfd_em_serra_nao_treina_porque_o_pixel_nao_resolve_o_relevo():
    _, v = _v(dict(fonte="gfd", classe_relevo="INGREME", precisao_data="janela_do_evento"))
    assert not v.loc[0, "uso_treino"]
    assert v.loc[0, "motivo_treino"] == "escala_do_instrumento_nao_resolve_o_relevo"


def test_negativo_por_exclusao_treina_mas_nao_avalia_nem_afirma():
    _, v = _v(dict(fonte="uk", classe=0, nivel_negativo="exclusao_qualificada",
                   precisao_data="nao_declarada", data_evento=None))
    assert v.loc[0, "uso_treino"]
    assert not v.loc[0, "uso_avaliacao_espacial"]
    assert not v.loc[0, "uso_afirmacao"]


def test_data_nao_resolvida_bloqueia_so_o_uso_temporal():
    _, v = _v(dict(precisao_data="nao_declarada", data_evento=None))
    assert v.loc[0, "uso_avaliacao_espacial"]
    assert not v.loc[0, "uso_avaliacao_temporal"]
    assert v.loc[0, "motivo_avaliacao_temporal"] == "data_nao_serve_a_pergunta:ordem_entre_eventos"


def test_terreno_nodata_nao_treina():
    _, v = _v(dict(hand_m=np.nan))
    assert v.loc[0, "motivo_treino"] == "terreno_incompleto"


def test_referencia_circular_nao_sustenta_afirmacao(monkeypatch):
    monkeypatch.setitem(adm.CAPACIDADE, "cems", {**adm.CAPACIDADE["cems"],
                                                 "circular_com": ["hand_m"]})
    _, v = _v({})
    assert v.loc[0, "uso_avaliacao_espacial"]
    assert not v.loc[0, "uso_afirmacao"]
    assert v.loc[0, "motivo_afirmacao"] == "referencia_circular_com_preditor"


def test_exigir_falha_com_linha_nao_autorizada_e_diz_o_motivo():
    d, _ = _v({}, dict(classe=0, nivel_negativo="ausencia"))
    with pytest.raises(adm.AdmissaoRecusada, match="papel_nao_ensina:desconhecido"):
        adm.exigir(d, "treino", "teste")
    ok, saiu = adm.filtrar(d, "treino")
    assert len(ok) == 1 and saiu == {"papel_nao_ensina:desconhecido": 1}
    adm.exigir(ok, "treino", "teste")


def test_mecanismo_que_o_instrumento_nao_sustenta_nao_treina():
    _, v = _v(dict(mecanismo="PLUVIAL_URBANO"))
    assert v.loc[0, "motivo_treino"] == "mecanismo_nao_justificado_pelo_instrumento"


def test_cena_do_cems_ordena_eventos_mas_nao_ancora_forcamento():
    d = pd.DataFrame([_linha()])
    assert adm.validade(d, pergunta_temporal="ordem_entre_eventos").loc[0, "uso_avaliacao_temporal"]
    v = adm.validade(d, pergunta_temporal="forcamento_no_instante")
    assert not v.loc[0, "uso_avaliacao_temporal"]
    assert v.loc[0, "motivo_avaliacao_temporal"] == "data_nao_serve_a_pergunta:forcamento_no_instante"


def test_previsao_recusa_evento_anterior_ao_dem():
    d = pd.DataFrame([_linha(fonte="gfd", precisao_data="janela_do_evento", data_evento="2011-07-23"),
                      _linha(data_evento="2024-05-08")])
    v = adm.validade(d, pergunta_temporal="previsao")
    assert v.uso_avaliacao_temporal.tolist() == [False, True]
    assert v.loc[0, "motivo_avaliacao_temporal"] == "preditor_publicado_depois_do_evento"
