"""Contratos do aprendizado continuo SAR; dados artificiais nunca comprovam ganho."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/sar"))
import continuidade_retencao as cr
import dados_pares_geograficos as dados

MINI = {"anotacoes_por_contexto": 2, "passos_por_etapa": 2, "passos_por_epoca": 1, "batch": 2,
        "base": 2, "recorte": 16, "threads": 1, "sementes": [7]}


def raster(p, a, x0):
    p.parent.mkdir(parents=True, exist_ok=True)
    a = a if a.ndim == 3 else a[None]
    with rasterio.open(p, "w", driver="GTiff", height=a.shape[1], width=a.shape[2], count=a.shape[0],
                       dtype=a.dtype, crs="EPSG:4326", transform=from_origin(x0, 30, .001, .001)) as f:
        f.write(a)


@pytest.fixture
def campanha(tmp_path):
    """Dez eventos com seis chips separados; ordem dos eventos fixada como na campanha."""
    raiz, jrc = tmp_path / "dados", tmp_path / "jrc"
    features = []
    for i in range(10):
        for j in range(6):
            nome = f"Ev{i}_{1000+i*10+j}"
            rng = np.random.default_rng(i*10+j)
            a = rng.normal(-18, 5, (2, 32, 32)).astype(np.float32)
            b = a + rng.normal(0, 2, a.shape).astype(np.float32)
            y = (b[0] < -20).astype(np.int16)
            y[0] = -1
            p = np.zeros((32, 32), np.int16)
            p[2:4] = 1
            x0 = -140 + i*10 + j*.05
            for pasta, valor, suf in (("PRE_S1", a, "S1Hand"), ("POST_S1", b, "S1Hand"), ("Labels", y, "LabelHand")):
                raster(raiz / "extraidos" / pasta / f"{nome}_{suf}.tif", valor, x0)
            raster(jrc / f"{nome}_JRCWaterHand.tif", p, x0)
        features.append({"properties": {"location": f"Ev{i}", "ID": i, "s1_date": "2018/01/01",
                                        "orbit": "ASCENDING", "rel_orbit_num": 10}})
    (raiz / "Sen1Floods11_Metadata.geojson").write_text(json.dumps({"features": features}), encoding="utf-8")
    inv = dados.inventariar(raiz, jrc)
    camp = tmp_path / "campanha"
    camp.mkdir()
    eventos = sorted(inv["eventos"])
    (camp / "inventario.json").write_text(json.dumps(inv), encoding="utf-8")
    (camp / "protocolo.json").write_text(json.dumps({"inventario": cr.ci.hash_json(inv), "ordem_eventos": eventos}),
                                         encoding="utf-8")
    diag = tmp_path / "diagnostico"
    diag.mkdir()
    (diag / "card.json").write_text(json.dumps({"aquisicao": {"liberada": True, "escolhido": {
        "braco": "mais_passos", "limiar": .2}}, "protocolo": {"config": {"modo": "pos"}}}), encoding="utf-8")
    return camp, diag, tmp_path


def linhas_grade(evento, colunas, linhas_n=1, passo=.046, largura=.046):
    out = []
    for i in range(linhas_n):
        for j in range(colunas):
            x0, y0 = j*passo, i*passo
            out.append({"nome": f"{evento}_{i}_{j}", "evento": evento,
                        "grade": {"bounds": [x0, y0, x0+largura, y0+largura]},
                        "hashes": {"antes": f"a{i}{j}", "depois": f"d{i}{j}"}})
    return out


def test_reserva_por_componentes_nao_toca_ajuste():
    rs = linhas_grade("e", 9, passo=.2)
    d = cr.dividir_contexto(rs, 1/3, 1.5)
    assert d["regra"] == "componentes_inteiros_por_hash" and len(d["retencao"]) == 3 and not d["tampao"]
    assert d == cr.dividir_contexto(list(reversed(rs)), 1/3, 1.5)


def test_bloco_contiguo_ganha_faixa_e_tampao():
    rs = linhas_grade("e", 6, linhas_n=5)
    d = cr.dividir_contexto(rs, 1/3, 1.5)
    assert d["regra"] == "faixa_do_eixo_longo_com_tampao" and len(d["retencao"]) == 10
    assert len(d["tampao"]) == 5 and len(d["ajuste"]) == 15
    viz = cr.adjacencia(rs, 1.5)
    idx = {r["nome"]: i for i, r in enumerate(rs)}
    assert not any(viz[idx[a], idx[b]] for a in d["ajuste"] for b in d["retencao"])


def test_separacao_recusa_vizinho_e_copia():
    rs = linhas_grade("e", 4)
    with pytest.raises(ValueError, match="vizinho"):
        cr.conferir_separacao(rs, {"ajuste": [rs[0]["nome"]], "retencao": [rs[1]["nome"]]}, 1.5)
    rs = linhas_grade("e", 4, passo=.2)
    rs[3]["hashes"]["depois"] = rs[0]["hashes"]["depois"]
    with pytest.raises(ValueError, match="copia"):
        cr.conferir_separacao(rs, {"ajuste": [rs[0]["nome"]], "retencao": [rs[3]["nome"]]}, 1.5)


def test_decisao_por_evento_e_poder():
    assert cr.decidir({str(i): .1 for i in range(5)}, .025, 6)["estado"] == "INDECIDIVEL"
    assert cr.decidir({str(i): .1+.01*i for i in range(8)}, .025, 6)["estado"] == "ganho_sustentado_no_desenvolvimento"
    assert cr.decidir({str(i): -.1-.01*i for i in range(8)}, .025, 6)["estado"] == "perda_sustentada_no_desenvolvimento"
    assert cr.decidir({str(i): (-1)**i*.001 for i in range(8)}, .025, 6)["estado"] == "sem_diferenca_relevante"
    assert cr.decidir({}, .025, 6)["estado"] == "not_testable"


def test_esquecimento_e_bwt():
    ctx = ["a", "b", "c"]
    serie = {1: {"a": .5}, 2: {"a": .6, "b": .4}, 3: {"a": .3, "b": .4, "c": .7}}
    r = cr.esquecimento(serie, ctx)
    assert r["esquecimento_por_contexto"] == pytest.approx({"a": .3, "b": 0.})
    assert r["bwt_por_contexto"] == pytest.approx({"a": -.2, "b": 0.})


def test_portao_usa_so_validacao():
    jobs = [{"braco": b, "seed": s, "etapa": 5, "validacao": {"macro_iou": v, "competente": True}}
            for s in (1, 2, 3) for b, v in (("reinicio", .4), ("continuacao_replay", .38))]
    assert cr.portao_e2(jobs, [1, 2, 3], .2, .025)["liberado"]
    jobs[1]["validacao"]["competente"] = False
    assert not cr.portao_e2(jobs, [1, 2, 3], .2, .025)["liberado"]


def test_ponta_a_ponta_retoma_e_nao_reescreve(campanha, monkeypatch):
    camp, diag, tmp = campanha
    saida = tmp / "saida"
    r = cr.executar(saida, camp, diag, MINI, pacote=None)
    concluido = cr.dc.ler(saida / "treino_concluido.json")
    jobs = concluido["jobs"]
    # 1 etapa compartilhada + 3 linhagens x 4 etapas + passos acumulados (+ E2 se o portao abrir)
    assert len(jobs) == 1 + 3*4 + 1 + (2*4 if concluido["portao_e2"]["liberado"] else 0)
    assert concluido["portao_e2"]["fonte"] == "somente validacao"
    e1 = next(j for j in jobs if j["braco"] == "etapa1_compartilhada")
    for b in ("continuacao", "continuacao_replay"):
        j2 = next(j for j in jobs if j["braco"] == b and j["etapa"] == 2)
        assert j2["pai"] == e1["pasta"]
        t = cr.dc.ler(Path(j2["pasta"]) / "treino.json")
        assert t["inicial"] == e1["hash_estado"]
    for j in jobs:
        t = cr.dc.ler(Path(j["pasta"]) / "treino.json")
        if j["braco"] in ("reinicio", "reinicio_passos_acumulados"):
            assert t["inicial"] is None and not t["antigas"]
    rep = cr.dc.ler(Path(next(j for j in jobs if j["braco"] == "continuacao_replay" and j["etapa"] == 3)["pasta"])
                    / "treino.json")
    assert len(rep["antigas"]) == 4 and len(rep["novas"]) == 2
    assert all(h["pixels_apresentados"]["de_antigas"] == MINI["batch"]//2 for h in rep["historico"])
    rpa = next(j for j in jobs if j["braco"] == "reinicio_passos_acumulados")
    assert rpa["passos"] == 5*MINI["passos_por_etapa"]
    custos = r["custos"]
    for braco, contidos in (("continuacao_replay", 5), ("continuacao", 5), ("reinicio", 1),
                            ("reinicio_passos_acumulados", 5)):
        assert custos[braco]["sementes"][7]["passos_contidos_no_modelo_final"] == contidos*MINI["passos_por_etapa"]
    assert custos["reinicio"]["sementes"][7]["passos_executados_na_linhagem"] == 5*MINI["passos_por_etapa"]
    assert not r["promovido"] and not r["altera_svc02"]
    # Retomada: nenhum peso muda e o resultado e reproduzido.
    antes = {j["pasta"]: (Path(j["pasta"]) / "melhor.pt").stat().st_mtime_ns for j in jobs}
    r2 = cr.executar(saida, camp, diag, MINI, pacote=None)
    assert antes == {p: (Path(p) / "melhor.pt").stat().st_mtime_ns for p in antes}
    assert r2["contrastes"] == json.loads(json.dumps(r["contrastes"]))
    with pytest.raises(ValueError, match="imutavel"):
        cr.executar(saida, camp, diag, {**MINI, "lr": .01}, pacote=None)


def test_avaliacao_recusa_antes_do_treino(campanha):
    camp, diag, tmp = campanha
    saida = tmp / "s2"
    cr.executar(saida, camp, diag, MINI, pacote=None, somente_preparar=True)
    protocolo, inv, part = cr.preparar(camp, diag, MINI, None)
    with pytest.raises(FileNotFoundError):
        cr.avaliar_tudo(saida, protocolo, inv, part)


def test_selecao_continuada_sem_oracle(campanha, monkeypatch):
    camp, diag, tmp = campanha
    monkeypatch.setattr(cr, "portao_e2", lambda *a, **k: {"liberado": True, "fonte": "teste"})
    saida = tmp / "s3"
    cr.executar(saida, camp, diag, MINI, pacote=None)
    jobs = cr.dc.ler(saida / "treino_concluido.json")["jobs"]
    assert {j["braco"] for j in jobs} >= {"replay_entropia", "replay_fixa"}
    sel = cr.dc.ler(saida / "replay_entropia_s7_e2" / "selecao.json")
    assert not sel["oracle_consultado_na_selecao"] and len(sel["escolhidos"]) == 2
    protocolo = cr.dc.ler(saida / "protocolo.json")
    ajuste = protocolo["divisoes"][protocolo["contextos_em_ordem"][1]]["ajuste"]
    assert set(sel["escolhidos"]) <= set(ajuste)
    with pytest.raises(ValueError, match="apenas identidade"):
        cr.aq.escolher([{"nome": "x", "evento": "e", "rotulo": 1}], "entropia", 1, 7, {"x": 1.})


def test_replay_exige_exemplos_antigos(campanha):
    camp, diag, tmp = campanha
    with pytest.raises(ValueError, match="replay sem"):
        cr.ajustar([{"nome": "x"}], [], "metade_antiga", None, 2, [], {}, 7, 2, MINI, tmp / "x", "a")


def test_relatorio_confere_pesos(campanha):
    import relatorio_continuidade as rel
    camp, diag, tmp = campanha
    saida = tmp / "s4"
    cr.executar(saida, camp, diag, MINI, pacote=None)
    texto = rel.escrever(saida).read_text(encoding="utf-8")
    assert "Matriz de ganho e retencao" in texto and "P2_acumulo_custo_marginal_igual" in texto
    peso = next(saida.glob("continuacao_s7_e3")) / "melhor.pt"
    peso.write_bytes(peso.read_bytes() + b"x")
    with pytest.raises(ValueError, match="peso alterado"):
        rel.escrever(saida)
