"""Serie continua de chuva e leitura dos brutos das centrais.

Estes testes travam tres modos de falha que ja aconteceram neste repositorio,
e que nao aparecem como erro -- aparecem como numero plausivel:

  * janela antecedente incompleta virando zero de precipitacao;
  * arquivo com extensao `.csv` e conteudo XLSX lido como texto;
  * cabecalho de exportacao que nao casa com o esquema de API, fazendo um
    arquivo de 142 mil linhas sair da contagem em silencio.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "externo"))

from chuva import casos_das_centrais as cdc  # noqa: E402
from chuva import serie_por_celula as spc  # noqa: E402
from chuva.contrato_chuva import (  # noqa: E402
    LOOKBACK_DAYS, TRIGGER_VARIABLE, derive_rain_features,
    derive_trigger_feature,
)


def _serie(dias: int, valor: float = 2.0, inicio="2024-01-01") -> dict:
    datas = pd.date_range(inicio, periods=dias, freq="D").strftime("%Y-%m-%d")
    return {d: valor for d in datas}


# --------------------------------------------------------- serie por celula

def test_derivar_dia_usa_as_funcoes_do_contrato_e_nao_formula_local():
    serie = _serie(LOOKBACK_DAYS + 1)
    dia = sorted(serie)[-1]
    valores = spc.derivar_dia(serie, dia)
    janela = [serie[d] for d in sorted(serie)[:-1]]
    esperado = derive_rain_features(janela)
    assert valores["rain_max_24h"] == esperado["rain_max_24h"]
    assert valores["rain_decay_index"] == esperado["rain_decay_index"]
    assert (valores[TRIGGER_VARIABLE]
            == derive_trigger_feature([serie[dia]])[TRIGGER_VARIABLE])


def test_o_gatilho_e_o_dia_do_evento_e_nao_o_dia_anterior():
    """O erro que a primeira versao do `aq_chamados07` cometeu, travado."""
    serie = _serie(LOOKBACK_DAYS + 1, valor=0.0)
    dia = sorted(serie)[-1]
    anterior = sorted(serie)[-2]
    serie[dia] = 40.0
    serie[anterior] = 5.0
    valores = spc.derivar_dia(serie, dia)
    assert valores[TRIGGER_VARIABLE] == 40.0
    assert valores["rain_max_24h"] == 5.0


def test_janela_antecedente_incompleta_nao_vira_zero_de_chuva():
    serie = _serie(LOOKBACK_DAYS)          # falta um dia para a janela fechar
    assert spc.derivar_dia(serie, sorted(serie)[-1]) is None


def test_dia_fora_da_serie_e_ausencia_e_nao_dia_seco():
    serie = _serie(LOOKBACK_DAYS + 1)
    assert spc.derivar_dia(serie, "2030-06-01") is None


# --------------------------------------------------- leitura dos brutos

def test_arquivo_xlsx_com_extensao_csv_e_reconhecido_pelo_conteudo(tmp_path):
    falso = tmp_path / "parece_csv.csv"
    with zipfile.ZipFile(falso, "w") as z:
        z.writestr("qualquer.xml", "<a/>")
    assert cdc._formato_real(falso) == "zip/xlsx"


def test_texto_e_reconhecido_como_texto(tmp_path):
    p = tmp_path / "de_verdade.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    assert cdc._formato_real(p) == "texto"


def test_a_leitura_e_delegada_ao_leitor_compartilhado(tmp_path, monkeypatch):
    """Sem duplicata de leitor: o modulo le pelo mesmo caminho da promocao.

    A duplicata existiu e divergiu -- este modulo enxergava um trimestre de
    Sao Paulo enquanto `aq_chamados03` ja lia os vinte e seis. O teste trava a
    delegacao, e nao o resultado dela.
    """
    import aq_chamados02_verificar_requisitos as aq02

    for cfg in cdc.FONTES.values():
        assert cfg["fonte_do_leitor"] in aq02.ESQUEMAS

    chamadas = {}

    def falso_carregar(chave, registro=None, reduzir=None):
        chamadas["chave"] = chave
        chamadas["tem_reducao"] = reduzir is not None
        if registro is not None:
            registro.append({"arquivo": "x.csv", "ok": True, "linhas": 2,
                             "lido_como": "csv"})
        bruto = pd.DataFrame({
            "created_date": ["2026-05-24T10:00:00", "2026-05-25T10:00:00"],
            "latitude": [40.71, 40.72], "longitude": [-74.01, -74.02],
            "complaint_type": ["Sewer"] * 2,
            "descriptor": ["Street Flooding (SJ)", "Sewer Backup (SA)"]})
        return reduzir(bruto, aq02.ESQUEMAS[chave]) if reduzir else bruto

    monkeypatch.setattr(aq02, "carregar", falso_carregar)
    d, proc = cdc.carregar_bruto("nyc")
    assert chamadas["chave"] == "nyc311"
    assert chamadas["tem_reducao"] is True
    assert proc["sem_limite_de_linhas"] is True
    assert proc["arquivos_com_falha"] == []
    assert len(d) == 2


def test_a_reducao_por_arquivo_guarda_so_o_caso_hidrico():
    """O que sobe para a memoria e o caso, e nao o bruto inteiro."""
    import aq_chamados02_verificar_requisitos as aq02

    bruto = pd.DataFrame({
        "created_date": ["2026-05-24T10:00:00"] * 3,
        "latitude": [40.71, 40.72, None], "longitude": [-74.01, -74.02, -74.03],
        "complaint_type": ["Sewer", "Noise", "Sewer"],
        "descriptor": ["Street Flooding (SJ)", "Loud Music", "Sewer Backup"]})
    z = cdc._reduzir_para_casos("nyc")(bruto, aq02.ESQUEMAS["nyc311"])
    assert list(z.columns) == ["lat", "lon", "ts", "texto"]
    assert len(z) == 1  # o de ruido sai por categoria, o sem lat por geometria


@pytest.mark.parametrize("cabecalho", ["api", "exportacao"])
def test_os_dois_cabecalhos_da_mesma_base_produzem_o_mesmo_caso(cabecalho):
    """O defeito medido: o arquivo completo de Nova York usa o cabecalho de
    EXPORTACAO, o esquema conhecia so o de API, e 142.579 linhas sairam da
    contagem sem erro nenhum."""
    linhas = {
        "api": {"created_date": ["2026-05-24T10:00:00"],
                "latitude": [40.71], "longitude": [-74.01],
                "complaint_type": ["Sewer"],
                "descriptor": ["Street Flooding (SJ)"]},
        "exportacao": {"Created Date": ["05/24/2026 10:00:00 AM"],
                       "Latitude": [40.71], "Longitude": [-74.01],
                       "Problem (formerly Complaint Type)": ["Sewer"],
                       "Problem Detail (formerly Descriptor)":
                           ["Street Flooding (SJ)"]},
    }[cabecalho]
    z = cdc.normalizar(pd.DataFrame(linhas), "nyc")
    assert len(z) == 1
    assert z.dia.iloc[0] == "2026-05-24"
    assert z.subtipo.iloc[0] == "street_flooding"


def test_chamado_de_outro_assunto_nao_entra_como_caso():
    d = pd.DataFrame({"created_date": ["2026-05-24T10:00:00"],
                      "latitude": [40.71], "longitude": [-74.01],
                      "complaint_type": ["Noise"],
                      "descriptor": ["Loud Music/Party"]})
    assert cdc.normalizar(d, "nyc").empty


def test_caso_sem_geometria_ou_sem_data_nao_entra():
    d = pd.DataFrame({
        "created_date": ["2026-05-24T10:00:00", None, "2026-05-25T10:00:00"],
        "latitude": [None, 40.71, 40.72], "longitude": [-74.01, -74.01, -74.02],
        "complaint_type": ["Sewer"] * 3,
        "descriptor": ["Street Flooding (SJ)"] * 3})
    assert len(cdc.normalizar(d, "nyc")) == 1


def test_celula_dia_agrega_por_celula_do_produto_e_nao_por_endereco():
    """Dois chamados a 200 m um do outro tem a MESMA exposicao por construcao:
    conta-los como duas observacoes seria pseudo-replicacao do forcing."""
    d = pd.DataFrame({
        "created_date": ["2026-05-24T10:00:00"] * 2,
        "latitude": [40.712, 40.714], "longitude": [-74.011, -74.013],
        "complaint_type": ["Sewer"] * 2,
        "descriptor": ["Street Flooding (SJ)", "Sewer Backup (Use Comments) (SA)"]})
    cd = cdc.por_celula_dia(cdc.normalizar(d, "nyc"))
    assert len(cd) == 1
    assert int(cd.casos.iloc[0]) == 2
    assert int(cd.casos_street_flooding.iloc[0]) == 1
    assert int(cd.casos_sewer_backup.iloc[0]) == 1


# ------------------------------------------------- auditoria das janelas

def test_a_auditoria_denuncia_janela_descasada_entre_as_classes(monkeypatch,
                                                                tmp_path):
    """Positivos num periodo e negativos noutro: a data sozinha separa."""
    dias_pos = pd.date_range("2026-05-01", periods=40, freq="D")
    dias_neg = pd.date_range("2026-07-01", periods=40, freq="D")
    d = pd.concat([
        pd.DataFrame({"fonte": "x", "classe": 1, "data_evento":
                      dias_pos.strftime("%Y-%m-%d")}),
        pd.DataFrame({"fonte": "x", "classe": 0, "data_evento":
                      dias_neg.strftime("%Y-%m-%d")})])
    caminho = tmp_path / "tabela.csv"
    d.to_csv(caminho, index=False)
    monkeypatch.setattr(cdc, "TABELA", caminho)
    a = cdc.auditar_janelas("x")
    assert a["auditado"]
    assert a["pct_positivos_em_dia_sem_negativo"] == 100.0
    assert a["auc_de_um_preditor_que_so_conhece_o_dia"] > 0.95
    assert "nao sustenta leitura causal" in a["leitura"]


def test_a_auditoria_aprova_janela_coincidente(monkeypatch, tmp_path):
    dias = pd.date_range("2026-05-01", periods=60, freq="D").strftime("%Y-%m-%d")
    d = pd.concat([
        pd.DataFrame({"fonte": "x", "classe": 1, "data_evento": dias}),
        pd.DataFrame({"fonte": "x", "classe": 0, "data_evento": dias})])
    caminho = tmp_path / "tabela.csv"
    d.to_csv(caminho, index=False)
    monkeypatch.setattr(cdc, "TABELA", caminho)
    a = cdc.auditar_janelas("x")
    assert a["pct_positivos_em_dia_sem_negativo"] == 0.0
    assert a["auc_de_um_preditor_que_so_conhece_o_dia"] == pytest.approx(0.5, abs=0.02)


# --------------------------------------- equivalencia com o que ja existia

@pytest.mark.skipif(
    not (REPO / "local_runs" / "chuva-serie-celula"
         / "serie_diaria_por_celula.csv").exists(),
    reason="serie ainda nao construida neste ambiente")
def test_a_serie_fatiada_reproduz_o_que_o_provider_devolveu_por_janela():
    """Dois caminhos do mesmo contrato tem de dar o mesmo numero -- o projeto
    ja teve dois hashes do MESMO conteudo saindo diferentes."""
    serie = pd.read_csv(REPO / "local_runs" / "chuva-serie-celula"
                        / "serie_diaria_por_celula.csv")
    conf = spc.conferir_contra_aq07(serie)
    if not conf.get("conferido"):
        pytest.skip(conf.get("motivo", "sem referencia"))
    assert conf["pares_comparados"] > 0
    assert conf["divergencias"] == 0, conf["exemplos"]


def test_cobertura_do_rotulo_denuncia_fracao_pequena(monkeypatch, tmp_path):
    """Se o bruto tem quarenta vezes mais caso que a tabela, o modelo foi
    ajustado no que sobrou de uma falha de leitura -- e nao numa escolha."""
    cd = pd.DataFrame({"cl_lat": [40.7] * 3, "cl_lon": [-74.0] * 3,
                       "dia": ["2024-01-01", "2024-02-01", "2024-03-01"],
                       "casos": [400, 400, 200]})
    cd.to_csv(tmp_path / "casos_celula_dia_nyc.csv", index=False)
    monkeypatch.setattr(cdc, "OUT", tmp_path)
    c = cdc.cobertura_do_rotulo("nyc", positivos_na_tabela=25)
    assert c["medida"] is True
    assert c["casos_no_bruto_com_lugar_e_dia"] == 1000
    assert c["fracao_usada"] == 0.025
    assert "fracao do rotulo disponivel" in c["leitura"]


def test_cobertura_do_rotulo_aprova_quando_a_tabela_cobre_o_bruto(monkeypatch,
                                                                  tmp_path):
    cd = pd.DataFrame({"cl_lat": [40.7], "cl_lon": [-74.0],
                       "dia": ["2024-01-01"], "casos": [100]})
    cd.to_csv(tmp_path / "casos_celula_dia_nyc.csv", index=False)
    monkeypatch.setattr(cdc, "OUT", tmp_path)
    c = cdc.cobertura_do_rotulo("nyc", positivos_na_tabela=95)
    assert "cobre a maior parte" in c["leitura"]


def test_sem_o_agregado_a_cobertura_nao_e_inventada(monkeypatch, tmp_path):
    monkeypatch.setattr(cdc, "OUT", tmp_path)
    c = cdc.cobertura_do_rotulo("nyc", positivos_na_tabela=10)
    assert c["medida"] is False
    assert "rode" in c["motivo"]
