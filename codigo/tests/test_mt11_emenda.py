import json
import multiprocessing as mpc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
for s in ("metodo", "ia", "terreno", "servico", "geo", "sar"):
    sys.path.insert(0, str(RAIZ / "scripts" / s))
import mt11_emenda as em
import mt11_independencia as mi
import mt11_terreno_executar as te
import registro_aberturas as reg
import reservas_examinadas as rex


# ------------------------------------------------------------------ registro por fase

def test_registro_idempotente_e_compativel_com_consumidores(tmp_path):
    a = reg.registrar(["GFD:1", "GFD:2"], "gfd", "pontos_amostrados", "concluido", "mt11", "sha256:x", pasta=tmp_path)
    b = reg.registrar(["GFD:2", "GFD:1"], "gfd", "pontos_amostrados", "concluido", "mt11", "sha256:x", pasta=tmp_path)
    assert a == {"novos": 2, "ledger_compartilhado_gravado": True}
    assert b == {"novos": 0, "ledger_compartilhado_gravado": False}
    ledger = tmp_path / "examinados_apos_congelamento.jsonl"
    linhas = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(linhas) == 1 and linhas[0]["eventos"] == ["GFD:1", "GFD:2"] and linhas[0]["avaliacao"] == "mt11:pontos_amostrados"
    assert rex.examinados_depois("gfd", ledger) == {"GFD:1", "GFD:2"}   # consumidor existente lê o campo `eventos`


def test_intencao_nao_vai_ao_ledger_e_fases_ficam_separadas(tmp_path):
    reg.registrar(["GFD:9"], "gfd", "escores_avaliados", "intencao", "mt11", "sha256:x", pasta=tmp_path)
    assert not (tmp_path / "examinados_apos_congelamento.jsonl").exists()
    reg.registrar(["GFD:9"], "gfd", "rotulo_consultado_na_selecao", "concluido", "mt11", "sha256:x", pasta=tmp_path)
    e = reg.estado_por_evento("gfd", tmp_path)
    assert e[("GFD:9", "escores_avaliados")] == "intencao"
    assert e[("GFD:9", "rotulo_consultado_na_selecao")] == "concluido"
    with pytest.raises(ValueError):
        reg.registrar(["GFD:9"], "gfd", "auc_lida", "concluido", "mt11", "sha256:x", pasta=tmp_path)


def _escritor(args):
    pasta, k = args
    reg.registrar([f"GFD:{k}", "GFD:comum"], "gfd", "pontos_amostrados", "concluido", "mt11", "sha256:x", pasta=Path(pasta))


def test_concorrencia_sem_perda_nem_duplicata(tmp_path):
    with mpc.get_context("spawn").Pool(4) as p:
        p.map(_escritor, [(str(tmp_path), k) for k in range(12)])
    linhas = [json.loads(x) for x in (tmp_path / "aberturas_por_fase.jsonl").read_text(encoding="utf-8").splitlines()]
    chaves = [(r["evento_id"], r["fase"], r["estado"]) for r in linhas]
    assert len(chaves) == len(set(chaves)) == 13          # 12 próprios + "comum" uma vez só
    assert not (tmp_path / "aberturas_por_fase.jsonl.trava").exists()


# ------------------------------------------------------------------ integridade

def test_validar_recusa_peso_q_alterado(tmp_path, monkeypatch):
    falso = tmp_path / "modelos.pkl"
    falso.write_bytes(em.m11.Q_PKL.read_bytes() + b"x")
    monkeypatch.setattr(em.m11, "Q_PKL", falso)
    with pytest.raises(ValueError, match="q_pkl_sha256"):
        em.validar()


def test_avaliar_recusa_pendencia():
    with pytest.raises(SystemExit, match="pendências"):
        em.exigir_completude({"pendencias": [{"evento_id": "GFD:1", "motivo": "terreno ausente"}]})
    em.exigir_completude({"pendencias": []})


# ------------------------------------------------------------------ terreno e transporte

def _raster(arq, forma=(40, 40), x0=500000., valor=1.):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(arq, "w", driver="GTiff", height=forma[0], width=forma[1], count=1, dtype="float32",
                       crs="EPSG:32633", transform=from_origin(x0, 1000., 30, 30), nodata=-9999.) as d:
        d.write(np.full(forma, valor, np.float32), 1)


def test_terreno_ok_distingue_ausente_parcial_completo(tmp_path):
    assert em.terreno_ok(tmp_path)["estado"] == "ausente"
    (tmp_path / "run_manifest.json").write_text("{}")
    _raster(tmp_path / "dem_filled.tif")
    assert em.terreno_ok(tmp_path)["estado"] == "parcial"           # manifesto sem todos os derivados
    for r in em.RASTERS[1:]:
        _raster(tmp_path / r)
    assert em.terreno_ok(tmp_path)["estado"] == "completo"
    _raster(tmp_path / "twi_dinf.tif", x0=500030.)
    assert "divergentes" in em.terreno_ok(tmp_path)["motivo"]
    _raster(tmp_path / "twi_dinf.tif")
    _raster(tmp_path / "dem_filled.tif", valor=-9999.)
    assert em.terreno_ok(tmp_path)["estado"] == "parcial"           # DEM todo nodata


def test_classificacao_de_falha_de_transporte(tmp_path):
    assert te.classificar("SystemExit: x: FABDEM GEE sem cobertura na bbox", tmp_path) == "sem_cobertura_comprovada"
    assert te.classificar("SystemExit: 9 tiles e nenhuma parte lida (motivos: ['sem cobertura FABDEM (oceano)'])", tmp_path) == "transporte_falhou"
    assert te.classificar("EEException: Computation timed out", tmp_path) == "transporte_falhou"
    assert te.classificar("RuntimeError: whitebox returned 1", tmp_path) == "derivacao_falhou"
    assert te.classificar(None, tmp_path) == "parcial"               # sem erro mas sem derivados íntegros


def test_transporte_exige_tres_conferencias_identicas(tmp_path, monkeypatch):
    monkeypatch.setattr(te, "MT11", tmp_path)
    assert te.transporte_conferido()[0] is False
    for nome, d in (("fabdem_transporte.json", {"comparadas": 40, "identicas": 40}),
                    ("fabdem_transporte_regional.json", {"comparadas": 60, "identicas": 60}),
                    ("fabdem_caminho_gravacao.json", {"total": 5, "identicas": 4})):
        (tmp_path / nome).write_text(json.dumps(d))
    assert te.transporte_conferido()[0] is False                    # uma regravação diferente bloqueia
    (tmp_path / "fabdem_caminho_gravacao.json").write_text(json.dumps({"total": 5, "identicas": 5}))
    assert te.transporte_conferido()[0] is True


# ------------------------------------------------------------------ populações por evento

def _evento(n=60, classes=(0, 1), F_ok=True):
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"classe": np.resize(classes, n), "arvore": .1, "construido_wc": .1, "slope_deg": 2., "jrc_occ_s1": 0.,
                      "n_pre": 20, "F_v1": rng.random(n) if F_ok else np.nan, "F_e1": rng.random(n) if F_ok else np.nan})
    return d


def test_evento_sem_f_fica_fora_so_do_hf():
    r = em.estado_do_evento(_evento(), {"s1": {"regra": None}})
    assert r["avaliavel_Q"] and not r["avaliavel_HF"] and r["estado"] == "exclusao_tecnica_prevista"
    assert r["fracao_F_preenchido"] == 0.                            # F ausente não vira zero


def test_classes_insuficientes_e_populacao_hf_pareada():
    r = em.estado_do_evento(_evento(classes=(1,)), {"s1": {"regra": "e1"}})
    assert not r["avaliavel_Q"] and not r["avaliavel_HF"]
    d = _evento()
    d.loc[:29, "arvore"] = .9                                        # metade não observável pelo SAR
    r = em.estado_do_evento(d, {"s1": {"regra": "v1"}})
    assert r["pontos_Q"] == 60 and r["pontos_HF"] == 30 and r["avaliavel_HF"]


# ------------------------------------------------------------------ unidade compartilhada entre fontes e rodadas

def _no(ev, l8, mb, ini, lat=0., lon=0., un=None):
    return {"evento_id": ev, "fonte": ev.split(":")[0], "classe": 0, "janela_inicio": ini, "janela_fim": ini,
            "lat": lat, "lon": lon, "unidade_espacial": un or ev, "HYBAS_L8": l8, "MAIN_BAS": mb}


def test_grafo_liga_lugar_repetido_e_separa_o_que_nao_compartilha():
    j = pd.DataFrame([
        _no("MT11:GFD:1", 81, 61, "2016-01-01", lat=10, lon=10),
        _no("EXAM:EMSR9", 81, 61, "2020-05-01", lat=10, lon=10),      # mesma sub-bacia, outro ano: lugar repetido
        _no("MT11:GFD:2", 82, 62, "2017-01-01", lat=40, lon=40),
        _no("MT11:GFD:3", 83, 62, "2017-01-10", lat=40.5, lon=40.5),  # mesmo MAIN_BAS em 9 dias: mesma onda
        _no("R3:EMSR500", 84, 63, "2020-01-01", lat=-30, lon=-60),
        _no("MT11:GFD:4", 85, 64, "2018-01-01", lat=-20, lon=100),
        _no("DS06:UK:EV_1", 86, 64, "2010-01-01", lat=-20, lon=100),  # mesmo exutório, outra década
    ])
    p = mi.classificar_mt11(j, "primario")
    ev = {e["evento_id"]: e for e in p["eventos"]}
    assert ev["MT11:GFD:1"]["ligado_a_desenvolvimento_ou_examinado"]
    assert ev["MT11:GFD:2"]["componente"] == ev["MT11:GFD:3"]["componente"]
    assert not ev["MT11:GFD:4"]["ligado_a_desenvolvimento_ou_examinado"]
    e = {x["evento_id"]: x for x in mi.classificar_mt11(j, "exutorio")["eventos"]}
    assert e["MT11:GFD:4"]["ligado_a_desenvolvimento_ou_examinado"]  # o desenho conservador liga pelo exutório


# ------------------------------------------------------------------ emenda 8: órbita pela cobertura dos pontos

import mt11_emenda_s1 as e8  # noqa: E402


def test_orbita_que_cobre_os_pontos_vence_a_que_so_tem_mais_datas():
    canto = [[0, 0], [.1, 0], [.1, .1], [0, .1], [0, 0]]
    tudo = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    cenas = [{"orbita": 10, "data": f"2017-01-{d:02d}", "anel": canto} for d in range(1, 21)]
    cenas += [{"orbita": 20, "data": f"2017-02-{d:02d}", "anel": tudo} for d in range(1, 6)]
    rng = np.random.default_rng(0)
    lon, lat = rng.uniform(.2, .9, 50), rng.uniform(.2, .9, 50)
    r = e8.escolher_orbita(cenas, lon, lat)
    assert r["orbita"] == 20 and r["pegada"].all() and len(r["datas"]) == 5
    assert r["cobertura"][10] == 0.


def test_fora_da_pegada_f_fica_ausente_e_admissao_so_conta_a_pegada():
    V = np.full((10, 12), -10.)
    V[5:, :] = np.nan                                  # metade nunca lida (fora da faixa)
    pegada = np.r_[np.ones(5, bool), np.zeros(5, bool)]
    datas = [f"2017-05-{d:02d}" for d in range(1, 13)]
    f = e8.frequencia_na_pegada(V, datas, 5., pegada)
    assert f["datas_admitidas"] == 12                  # sem a pegada, 50% de leitura recusaria todas
    assert np.isnan(f["F_v1"][5:]).all() and (f["n_pre"][5:] == 0).all()
    assert (f["F_v1"][:5] == 0).all()
    assert e8.frequencia_na_pegada(V, datas, 5., np.zeros(10, bool))["datas_admitidas"] == 0


def test_evento_sem_linhas_apos_espelho_sai_por_regra():
    d = _evento(n=0)
    r = em.estado_do_evento(d, {"s1": {"regra": None}, "saiu_no_espelho": {"relevo_nao_resolvido_pelo_instrumento": 311}})
    assert not r["avaliavel_Q"] and not r["avaliavel_HF"] and r["estado"] == "exclusao_tecnica_prevista"
