"""MT11, preparo: candidatos ao teste confirmatório da via transdutiva, SÓ por metadados.

Pergunta do teste (MT10): a frequência de água SAR do próprio lugar, só com datas ANTERIORES
ao evento e sem rótulo, acrescenta ordenação física ao Q_conexao em unidades nunca usadas?

Por que GFD 2015-2018: o rótulo do GFD é MODIS óptico, outro instrumento que não o S1 da
frequência (ataca o erro compartilhado que o MT10 não separava); o catálogo tem eventos
nunca examinados fora das bacias do MT; o S1 só existe desde out/2014 e o GFD termina em 2018.

Regra (nenhum rótulo é lido; nenhuma saída de modelo é vista):
  universo   GLOBAL_FLOOD_DB/MODIS_EVENTS/V1, início entre 2015 e 2018
  fora       conhecidos de 03/10; examinados depois do congelamento (gfdq03, rfd03, mt04);
             examinados na seleção gfdq01; causa sem "rain" ou com snow/cyclone/storm/dam/ice;
             centróide em tile de 1 grau tocado por qualquer ponto do ds06
  S1         datas distintas IW/VV no centróide em [início-425 d, início-60 d] (60 d = transição
             da pilha); elegível com >= 10 datas por órbita (média)
  unidade    MAIN_BAS do HydroBASINS nível 8 que contém o centróide; eventos na mesma bacia
             principal são UMA unidade; bacia principal de algum ponto do MT marca dependência
Saída: `local_runs/mt11-transdutivo/candidatos.csv` e `candidatos.json` com hash.

Limites: centróide é aproximação da mancha; a elegibilidade por tile (área de cada classe)
e a bacia final só se fixam no pré-registro, ainda sem leitura de rótulo por ponto.
Uso: python scripts/metodo/mt11_candidatos_transdutivo.py
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

import pandas as pd
import shapely
from shapely import wkb

REPO = Path(__file__).resolve().parents[2]
for p in ("suscetibilidade", "servico", "geo", "metodo"):
    sys.path.insert(0, str(REPO / "scripts" / p))
from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "mt11-transdutivo"
CONHECIDOS = REPO / "local_runs/reservas-examinadas/conhecidos_2026_10_03.json"
EXAMINADOS = REPO / "local_runs/reservas-examinadas/examinados_apos_congelamento.jsonl"
SELECAO_GFDQ = REPO / "local_runs/gfd-quarentena/selecao.json"
HYBAS = REPO / "local_runs/hydrobasins/v1c"
REGIOES = {"sa": "tabelas", "eu": "tabelas_eu", "af": "tabelas_af", "as": "tabelas_as", "au": "tabelas_au", "na": "tabelas_na"}
EXCLUI_CAUSA = ("snow", "cyclone", "storm", "dam", "ice")
ANOS = ("2015", "2018")
JANELA_PRE = (425, 60)
MIN_DATAS_POR_ORBITA = 10


def excluidos() -> set[str]:
    ids = {e for v in json.loads(CONHECIDOS.read_text(encoding="utf-8"))["por_fonte"].values() for e in v}
    for linha in EXAMINADOS.read_text(encoding="utf-8").splitlines():
        ids |= set(json.loads(linha)["eventos"])
    ids |= {f"GFD:{e['id']}" for e in json.loads(SELECAO_GFDQ.read_text(encoding="utf-8"))["examinados"]}
    return ids


def tiles_ds06() -> set[str]:
    from ds06_dataset_de_treino import versao_atual
    t = pd.read_parquet(versao_atual() / "pontos.parquet", columns=["lat", "lon"])
    return {f"{math.floor(a)}_{math.floor(o)}" for a, o in zip(t.lat, t.lon)}


def metadados(ee) -> list[dict]:
    col = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")
    props = ["id", "system:time_start", "dfo_main_cause", "dfo_country", "dfo_centroid_x", "dfo_centroid_y"]
    vals = {p: col.aggregate_array(p).getInfo() for p in props}
    out = []
    for t in zip(*vals.values()):
        r = dict(zip(props, t))
        r["inicio"] = dt.datetime.fromtimestamp(r.pop("system:time_start") / 1000, dt.UTC).date().isoformat()
        out.append(r)
    return out


def contar_s1(ee, x, y, inicio):
    t0 = dt.date.fromisoformat(inicio)
    q = (ee.ImageCollection("COPERNICUS/S1_GRD").filter(ee.Filter.eq("instrumentMode", "IW"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
         .filterBounds(ee.Geometry.Point([x, y]))
         .filterDate(str(t0 - dt.timedelta(days=JANELA_PRE[0])), str(t0 - dt.timedelta(days=JANELA_PRE[1]))))
    datas = q.aggregate_array("system:time_start").map(lambda v: ee.Date(v).format("YYYY-MM-dd")).distinct().size()
    orb = q.aggregate_array("relativeOrbitNumber_start").distinct().size()
    return ee.List([datas, orb]).getInfo()


def localizador():
    tabs = {r: pd.read_parquet(HYBAS / p / "nivel_08.parquet") for r, p in REGIOES.items()}

    def achar(x, y):
        pt = shapely.Point(x, y)
        for reg, t in tabs.items():
            m = t[(t.lon_min <= x) & (t.lon_max >= x) & (t.lat_min <= y) & (t.lat_max >= y)]
            for _, r in m.iterrows():
                if wkb.loads(r.wkb).covers(pt):
                    return reg, int(r.HYBAS_ID), int(r.MAIN_BAS)
        return None, None, None
    return achar


def main():
    import ee
    import mt03_experimento as e3
    ee.Initialize()
    fora, proibidos = excluidos(), tiles_ds06()
    mt = e3.carregar()
    mt_main = set(mt.MAIN_BAS.dropna().astype("int64"))
    achar = localizador()
    motivo, linhas = {}, []
    for r in metadados(ee):
        if not ANOS[0] <= r["inicio"][:4] <= ANOS[1]:
            continue
        eid, causa = f"GFD:{r['id']}", str(r["dfo_main_cause"]).lower()
        tile = f"{math.floor(r['dfo_centroid_y'])}_{math.floor(r['dfo_centroid_x'])}"
        m = ("conhecido_ou_examinado" if eid in fora else
             "causa" if "rain" not in causa or any(x in causa for x in EXCLUI_CAUSA) else
             "tile_do_ds06" if tile in proibidos else None)
        if m:
            motivo[m] = motivo.get(m, 0) + 1
            continue
        datas, orb = contar_s1(ee, r["dfo_centroid_x"], r["dfo_centroid_y"], r["inicio"])
        reg, l8, mb = achar(r["dfo_centroid_x"], r["dfo_centroid_y"])
        linhas.append({"evento_id": eid, "inicio": r["inicio"], "pais": r["dfo_country"], "causa": r["dfo_main_cause"],
                       "lon": r["dfo_centroid_x"], "lat": r["dfo_centroid_y"], "tile_centroide": tile,
                       "s1_datas_pre": datas, "s1_orbitas": orb, "datas_por_orbita": datas / max(orb, 1),
                       "regiao_hybas": reg, "hybas_l8": l8, "main_bas": mb,
                       "main_bas_do_mt": (mb in mt_main) if mb else None})
        print(eid, r["inicio"], r["dfo_country"], datas, orb, mb, flush=True)
    t = pd.DataFrame(linhas)
    t["elegivel_s1"] = t.datas_por_orbita >= MIN_DATAS_POR_ORBITA
    t["elegivel"] = t.elegivel_s1 & t.main_bas.notna() & (t.main_bas_do_mt == False)  # noqa: E712
    t["unidade"] = t.main_bas.map(lambda v: f"BAS:{int(v)}" if pd.notna(v) else None)
    SAIDA.mkdir(parents=True, exist_ok=True)
    t.to_csv(SAIDA / "candidatos.csv", index=False)
    el = t[t.elegivel]
    res = {"tipo": "mt11_candidatos_por_metadados", "quando_utc": agora_utc(), "regra": __doc__.split("Regra")[1].split("Saída")[0].strip(),
           "excluidos_por_motivo": motivo, "candidatos_apos_exclusao": len(t), "eventos_elegiveis": len(el),
           "unidades_elegiveis": int(el.unidade.nunique()),
           "unidades_com_20_datas": int(t[t.elegivel & (t.datas_por_orbita >= 20)].unidade.nunique()),
           "fora_da_cobertura_hybas": int(t.main_bas.isna().sum()), "rotulo_lido": False,
           "nao_e": "não é seleção final: elegibilidade por tile e bacia por mancha ficam para o pré-registro"}
    res["hash_lista"] = sha256_json(sorted(el.evento_id))
    (SAIDA / "candidatos.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "regra"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
