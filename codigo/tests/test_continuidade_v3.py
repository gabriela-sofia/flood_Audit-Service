"""Campanha v3 em dados artificiais: grupos separados, LwF mascarado, hibrido e analise. Nao e evidencia."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/sar"))
sys.path.insert(0, str(REPO / "tests"))
import continuidade_v3 as v3
import continuidade_retencao as v1
import fontes_sar as fs
from test_continuidade_retencao import MINI, campanha  # noqa: F401

CFG = {"anotacoes_por_contexto": 2, "passos_por_etapa": 2, "passos_por_epoca": 1, "batch": 2, "base": 2,
       "recorte": 16, "threads": 1, "sementes": [7], "semente_custo": 7}


def tif(p, a, dtype, x0):
    p.parent.mkdir(parents=True, exist_ok=True)
    a = a if a.ndim == 3 else a[None]
    with rasterio.open(p, "w", driver="GTiff", height=a.shape[1], width=a.shape[2], count=a.shape[0], dtype=dtype,
                       crs="EPSG:32630", transform=from_origin(x0, 640, 10, 10)) as f:
        f.write(a.astype(dtype))
    return str(p)


def sha(p):
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def preparo_geoid(raiz):
    chips, divisoes = [], {}
    for k, (ev, com_inundacao) in enumerate((("EMSR424", True), ("EMSR698", True), ("EMSR728", False))):
        nomes = []
        for t in range(2):
            rng = np.random.default_rng(k*10+t)
            y = np.zeros((64, 64), np.uint8)
            if com_inundacao:
                y[rng.random((64, 64)) < .2] = 2
            y[:4] = 1
            pos = np.where(y == 2, .003, .05)[None].repeat(2, 0) * rng.uniform(.8, 1.2, (2, 64, 64))
            pre = np.full((2, 64, 64), .05) * rng.uniform(.8, 1.2, (2, 64, 64))
            x0 = 10000*k + 1000*t
            base = raiz / ev / f"t{t}"
            cam = {"post": tif(base / "post.tif", pos, "float32", x0), "pre": tif(base / "pre.tif", pre, "float32", x0),
                   "label": tif(base / "label.tif", y, "uint8", x0),
                   "validity": tif(base / "val.tif", np.tile([1, 1, 1, 0], (64, 16)), "uint8", x0)}
            for row in (0, 32):
                for col in (0, 32):
                    nome = f"{ev}-1-{t}:r{row}:c{col}"
                    chips.append({"nome": nome, "evento": ev, "par": f"{ev}-1-{t}", "janela": [col, row, 32, 32],
                                  "crs": "EPSG:32630", "bounds": [x0+col*10, 0, x0+col*10+320, 320],
                                  "caminhos": cam, "hashes_arquivos_pai": {a: sha(b) for a, b in cam.items()},
                                  "pre_pos_primario": ev != "EMSR698"})
                    nomes.append(nome)
        divisoes[ev] = {"ajuste": nomes[:3], "retencao": nomes[3:6], "tampao": nomes[6:]}
    m = {"chips": chips, "divisoes": divisoes}
    f = raiz / "preparo.json"
    f.write_text(json.dumps(m), encoding="utf-8")
    return f


def test_v3_ponta_a_ponta(campanha, tmp_path):
    import analise_continuidade_v3 as an
    camp, diag, tmp = campanha
    v1.executar(tmp / "v1", camp, diag, MINI, pacote=None, somente_preparar=True)
    prep = preparo_geoid(tmp_path / "geoid")
    fs.limpar_cache()
    p, inv, todos = v3.preparar(tmp / "v1", prep, CFG)
    anotados = {n for c in p["anotacoes"]["7"].values() for n in c}
    assert not anotados & set().union(*map(set, p["grupos"].values()))
    assert all(n.startswith("EMSR698") for n in p["grupos"]["calibracao"])
    assert all(n.startswith("EMSR728") for n in p["grupos"]["controle"])
    assert any(n.startswith("EMSR424") for n in p["grupos"]["retencao"])
    assert p["primaria"]["piso_discreto"]["holm_pode_rejeitar"]
    r = v3.executar(tmp / "v3", CFG, tmp / "v1", prep, por_trava=False)
    assert r["estado"] == "pronto_para_analise"
    por_id, ordem, lin = v3.planejar(p)
    assert len(lin) == 6*5 + 6
    for o in range(6):
        assert len({l["jobs"][0] for l in lin if l["ordem"] == o and l["braco"] != "reinicio_acumulado"}) == 1
    # Ordem 0 termina em Madagascar (GEOID com faixas invalidas): etapa 6.
    t = {b: v3.dc.ler(tmp / "v3" / "jobs" / next(l for l in lin if l["braco"] == b and l["ordem"] == 0)["jobs"][5]
                      / "treino.json") for b in v3.BRACOS}
    assert t["lwf_mascarado"]["custo"]["passes_professor"] > 0
    assert t["lwf_mascarado"]["custo"]["pixels_distilados"] < t["lwf_legado"]["custo"]["pixels_distilados"]
    assert t["hibrido"]["custo"]["recortes_memoria"] > 0 and t["hibrido"]["custo"]["passes_professor"] > 0
    assert t["continuacao"]["custo"]["passes_professor"] == 0 and not t["continuacao"]["memoria"]
    assert len(t["replay_m1"]["memoria"]) == 5  # um chip por contexto antigo na etapa 6
    a = an.executar(tmp / "v3")
    pri = a["primaria"]["hibrido - lwf_mascarado"]
    assert pri["decisao_9_unidades"]["n"] == len(a["unidades"])
    assert pri["retencao_6"]["estado"] == "estimacao"
    assert all("T" in x and "limiar" in x for x in a["linhagens"])
    antes = {j: (tmp / "v3" / "jobs" / j / "melhor.pt").stat().st_mtime_ns for j in ordem}
    v3.executar(tmp / "v3", CFG, tmp / "v1", prep, por_trava=False)
    assert antes == {j: (tmp / "v3" / "jobs" / j / "melhor.pt").stat().st_mtime_ns for j in ordem}
    with pytest.raises(ValueError, match="imutavel"):
        v3.executar(tmp / "v3", {**CFG, "lr": .01}, tmp / "v1", prep, por_trava=False)


def test_kl_mascarada_ignora_nodata():
    import torch
    import nucleo_sar as nu
    s = torch.zeros(1, 2, 2, 2)
    z = torch.zeros(1, 2, 2, 2)
    z[0, 1, 0, 0] = 10.
    m = torch.tensor([[[False, True], [True, True]]])
    assert float(nu.kl_mascarada(s, z, 2., m)) == 0.
    assert float(nu.kl_mascarada(s, z, 2., None)) > 0.


def test_memoria_m1_admite_so_chip_rotulado():
    import continuidade_v3 as v3
    canon = ["a", "b"]
    anot = {"7": {"a": ["a1", "a2", "a3"], "b": ["b1", "b2"]}}
    cheio = {"7": {"a": {"a1": 5, "a2": 5, "a3": 5}, "b": {"b1": 5, "b2": 5}}}
    livre = v3.admitir_memoria(anot, cheio, canon)
    vazio = {"7": {"a": {n: (0 if n == livre["7"]["a"] else 5) for n in anot["7"]["a"]}, "b": cheio["7"]["b"]}}
    m = v3.admitir_memoria(anot, vazio, canon)
    assert m["7"]["a"] != livre["7"]["a"] and vazio["7"]["a"][m["7"]["a"]] > 0
    assert m["7"]["b"] == livre["7"]["b"]  # contexto sem chip vazio fica igual a v3
    with pytest.raises(ValueError, match="sem nenhum chip"):
        v3.admitir_memoria(anot, {"7": {"a": dict.fromkeys(anot["7"]["a"], 0), "b": cheio["7"]["b"]}}, canon)
