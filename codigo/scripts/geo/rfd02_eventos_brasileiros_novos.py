"""Cheias brasileiras 2019-2025 rotuladas pela replica do GFD, em quarentena, para o teste confirmatorio no Brasil.

-------------------------------------------------------------------------
A PERGUNTA QUE ESTE TESTE SEPARA
-------------------------------------------------------------------------

O teste sul-americano (`gfdq03`) mediu evento novo E territorio novo ao mesmo
tempo. Aqui o evento e novo (2019-2025, posterior a todo o GFD) e o LUGAR pode
ser conhecido: o treino ocupa 661 tiles de 1 grau, quase o Brasil inteiro, e
nao ha como fugir disso. Tile do treino e PERMITIDO e declarado.

-------------------------------------------------------------------------
REGRA (gravada com hash antes da primeira consulta de rotulo)
-------------------------------------------------------------------------

  so roda se `rfd01 --validar` passou no portao pre-registrado
  lista de eventos  GDACS FL, Brasil, 2019-01-01 a 2025-12-31, todos os niveis
                    de alerta; o snapshot da lista e gravado com hash
  fora              GDACS 1000220 (25/01/2019, rompimento de barragem em
                    Brumadinho: nao e cheia de chuva); todo evento a <= 500 km
                    e <= 45 dias de um evento brasileiro conhecido de qualquer
                    fonte (o unico pos-2018 e o CEMS EMSR720, RS, 02/05/2024)
  regiao            bacia HydroSHEDS nivel 4 que contem o centroide do GDACS
                    (o GFD usava as bacias nivel 4 do poligono do DFO), cortada
                    ao Brasil
  rotulo            `rfd01.dfo` com o limiar padrao 0,70/675 e composto de 3 dias,
                    janela do GDACS (fromdate a todate)
  tiles             1 grau com >= 5 km2 de positivo interior e >= 5 km2 de
                    negativo na faixa (`classes_do_rotulo`); ate 3 por evento por
                    hash; dois eventos novos nao dividem tile
  ordem             sha256("rfdbr|" + eventid); 16 eventos
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for pasta in ("servico", "geo", "suscetibilidade", "terreno"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

from proveniencia import agora_utc, sha256_json  # noqa: E402

Q = REPO / "local_runs" / "rfd-quarentena-brasil"
GDACS = ("https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?eventlist=FL&country=Brazil"
         "&fromDate=2019-01-01&toDate=2025-12-31&alertlevel=Green;Orange;Red")
EXCLUIDOS = {"1000220": "rompimento de barragem (Brumadinho, 25/01/2019): nao e cheia de chuva"}
CONHECIDOS_POS_2018 = [{"evento": "CEMS:EMSR720", "data": "2024-05-02", "lon": -51.972654, "lat": -28.982522}]
RAIO_KM, JANELA_DIAS = 500.0, 45
N_EVENTOS, TILES_POR_EVENTO, MIN_KM2 = 16, 3, 5.0
SAL = "rfdbr|"
GRADE_GFD = 0.002245788210298804   # grau por pixel do `flooded` do GFD


def regra() -> dict:
    return {"lista": GDACS, "excluidos": EXCLUIDOS, "conhecidos_pos_2018": CONHECIDOS_POS_2018,
            "raio_km": RAIO_KM, "janela_dias": JANELA_DIAS, "regiao": "HydroSHEDS hybas_4 que contem o centroide, cortada ao Brasil",
            "rotulo": "rfd01.dfo, limiar 0,70/675, composto 3 dias, janela fromdate-todate do GDACS",
            "grade": f"EPSG:4326 a {GRADE_GFD} grau (a do GFD)", "tiles_do_treino": "permitidos (lugar conhecido)",
            "n_eventos": N_EVENTOS, "tiles_por_evento": TILES_POR_EVENTO, "min_km2_por_classe_no_tile": MIN_KM2,
            "ordem": f"sha256('{SAL}' + eventid)"}


def pre_registrar() -> str:
    Q.mkdir(parents=True, exist_ok=True)
    alvo = Q / "pre_registro_selecao.json"
    r = regra()
    h = sha256_json(r)
    if alvo.exists():
        a = json.loads(alvo.read_text(encoding="utf-8"))
        if a["hash"] != h:
            raise RuntimeError(f"regra de selecao mudou depois do pre-registro ({a['hash']} -> {h})")
        return h
    alvo.write_text(json.dumps({"quando_utc": agora_utc(), "hash": h, "regra": r}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return h


def lista_gdacs() -> list[dict]:
    alvo = Q / "gdacs_snapshot.json"
    if not alvo.exists():
        with urllib.request.urlopen(GDACS, timeout=120) as r:
            d = json.load(r)
        ev = [{"eventid": str(f["properties"]["eventid"]), "inicio": f["properties"]["fromdate"][:10],
               "fim": f["properties"]["todate"][:10], "alerta": f["properties"].get("alertlevel"),
               "fonte": f["properties"].get("source"), "lon": f["geometry"]["coordinates"][0],
               "lat": f["geometry"]["coordinates"][1]} for f in d["features"] if f["geometry"]["type"] == "Point"]
        alvo.write_text(json.dumps({"quando_utc": agora_utc(), "url": GDACS, "eventos": ev, "hash": sha256_json(ev)},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    return json.loads(alvo.read_text(encoding="utf-8"))["eventos"]


def _km(a, b, c, d) -> float:
    p = math.pi / 180
    h = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def candidatos() -> tuple[list, dict]:
    from datetime import date
    fora = {}
    out = []
    for e in lista_gdacs():
        if e["eventid"] in EXCLUIDOS:
            fora[e["eventid"]] = EXCLUIDOS[e["eventid"]]
            continue
        perto = [k["evento"] for k in CONHECIDOS_POS_2018
                 if _km(e["lat"], e["lon"], k["lat"], k["lon"]) <= RAIO_KM
                 and abs((date.fromisoformat(e["inicio"]) - date.fromisoformat(k["data"])).days) <= JANELA_DIAS]
        if perto:
            fora[e["eventid"]] = f"coincide com evento conhecido {perto}"
            continue
        out.append(e)
    return sorted(out, key=lambda e: hashlib.sha256(f"{SAL}{e['eventid']}".encode()).hexdigest()), fora


def rotulo(e: dict):
    import ee
    from rfd01_replica_gfd import PADRAO, dfo
    br = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", "Brazil")).geometry()
    bacia = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_4").filterBounds(
        ee.Geometry.Point([e["lon"], e["lat"]])).geometry()
    roi = bacia.intersection(br, maxError=1000)
    img = dfo(roi, e["inicio"], e["fim"], PADRAO["b1b2"], PADRAO["b7"])
    grade = ee.Projection("EPSG:4326").scale(GRADE_GFD, GRADE_GFD)
    return img.reproject(grade), roi, grade


def tiles_elegiveis(e: dict, proibidos: set) -> list[dict]:
    import ee
    from gfd01_eventos_do_brasil import FAIXA_M, classes_do_rotulo
    img, roi, grade = rotulo(e)
    esc = 250.0
    pos, neg = classes_do_rotulo(img, esc, FAIXA_M)
    b = roi.bounds().coordinates().getInfo()[0]
    xs, ys = [c[0] for c in b], [c[1] for c in b]
    cel = [ee.Feature(ee.Geometry.Rectangle([lo, la, lo + 1, la + 1]), {"t": f"{la}_{lo}"})
           for la in range(math.floor(min(ys)), math.ceil(max(ys)))
           for lo in range(math.floor(min(xs)), math.ceil(max(xs))) if f"{la}_{lo}" not in proibidos]
    if not cel:
        return []
    area = ee.Image.pixelArea().divide(1e6)
    duas = area.updateMask(pos).rename("pos_km2").addBands(area.updateMask(neg).rename("neg_km2"))
    from rfd01_replica_gfd import exportar_e_ler
    # em lote: a cota do Earth Engine esta em modo restrito e a consulta interativa estoura o tempo
    r = exportar_e_ler(duas.reduceRegions(ee.FeatureCollection(cel), ee.Reducer.sum(), crs=grade, tileScale=8),
                       f"rfdbr_tiles_{e['eventid']}")
    ok = []
    for pr in r:
        p_, n_ = pr.get("pos_km2") or 0.0, pr.get("neg_km2") or 0.0
        if p_ >= MIN_KM2 and n_ >= MIN_KM2:
            la, lo = map(int, pr["t"].split("_"))
            ok.append({"tile": pr["t"], "tile_1grau": [lo, la, lo + 1, la + 1], "pos_km2": round(p_, 1),
                       "neg_km2": round(n_, 1), "prevalencia_real": round(p_ / (p_ + n_), 5)})
    return sorted(ok, key=lambda t: hashlib.sha256(f"{SAL}{e['eventid']}|{t['tile']}".encode()).hexdigest())[:TILES_POR_EVENTO]


LOTE_DE_CANDIDATOS = 6


def _celulas(e: dict):
    """Celulas de 1 grau na caixa da regiao do evento -- SEM excluir tile usado (isso e aplicado depois)."""
    import ee
    _, roi, _ = rotulo(e)
    b = roi.bounds().coordinates().getInfo()[0]
    xs, ys = [c[0] for c in b], [c[1] for c in b]
    return [ee.Feature(ee.Geometry.Rectangle([lo, la, lo + 1, la + 1]), {"t": f"{la}_{lo}", "eventid": e["eventid"]})
            for la in range(math.floor(min(ys)), math.ceil(max(ys)))
            for lo in range(math.floor(min(xs)), math.ceil(max(xs)))]


def _areas_do_lote(lote: list[dict]) -> dict:
    """Area de positivo interior e de negativo na faixa por celula, para VARIOS candidatos numa tarefa so.

    A elegibilidade de uma celula nao depende dos outros eventos; a regra de nao dividir tile entre eventos
    novos e aplicada depois, na ordem pre-registrada -- o resultado e o mesmo de uma tarefa por candidato,
    com uma fraccao das esperas na fila do modo restrito (04/10/2026).
    """
    import ee
    from gfd01_eventos_do_brasil import FAIXA_M, classes_do_rotulo
    from rfd01_replica_gfd import exportar_e_ler
    partes = []
    for e in lote:
        img, _, grade = rotulo(e)
        pos, neg = classes_do_rotulo(img, 250.0, FAIXA_M)
        area = ee.Image.pixelArea().divide(1e6)
        duas = area.updateMask(pos).rename("pos_km2").addBands(area.updateMask(neg).rename("neg_km2"))
        partes.append(duas.reduceRegions(ee.FeatureCollection(_celulas(e)), ee.Reducer.sum(), crs=grade, tileScale=8))
    nome = "rfdbr_lote_" + hashlib.sha256("|".join(e["eventid"] for e in lote).encode()).hexdigest()[:12]
    linhas = exportar_e_ler(ee.FeatureCollection(partes).flatten(), nome)
    out = {e["eventid"]: [] for e in lote}
    for pr in linhas:
        out[str(pr["eventid"])].append(pr)
    return out


def _escolher_tiles(e: dict, linhas: list[dict], usados: set) -> list[dict]:
    ok = []
    for pr in linhas:
        if pr["t"] in usados:
            continue
        p_, n_ = pr.get("pos_km2") or 0.0, pr.get("neg_km2") or 0.0
        if p_ >= MIN_KM2 and n_ >= MIN_KM2:
            la, lo = map(int, pr["t"].split("_"))
            ok.append({"tile": pr["t"], "tile_1grau": [lo, la, lo + 1, la + 1], "pos_km2": round(p_, 1),
                       "neg_km2": round(n_, 1), "prevalencia_real": round(p_ / (p_ + n_), 5)})
    return sorted(ok, key=lambda t: hashlib.sha256(f"{SAL}{e['eventid']}|{t['tile']}".encode()).hexdigest())[:TILES_POR_EVENTO]


def selecionar() -> dict:
    import ee
    v = json.loads((REPO / "local_runs" / "rfd-replica-gfd" / "validacao.json").read_text(encoding="utf-8"))
    if not v.get("passou"):
        raise SystemExit("a replica nao passou no portao de validacao: nao rotula evento novo")
    ee.Initialize()
    h = pre_registrar()
    cand, fora = candidatos()
    usados, escolhidos, examinados = set(), [], []
    areas: dict = {}
    i = 0
    while i < len(cand) and len(escolhidos) < N_EVENTOS:
        e = cand[i]
        if e["eventid"] not in areas:
            # le a tarefa por candidato ja feita (versao anterior), se houver; senao calcula o proximo lote
            try:
                from rfd01_replica_gfd import PASTA_EE, exportar_e_ler
                ee.data.getAsset(f"{PASTA_EE}/rfdbr_tiles_{e['eventid']}")
                areas[e["eventid"]] = exportar_e_ler(None, f"rfdbr_tiles_{e['eventid']}")
            except ee.EEException:
                lote = [c for c in cand[i:i + LOTE_DE_CANDIDATOS] if c["eventid"] not in areas]
                areas.update(_areas_do_lote(lote))
        t = _escolher_tiles(e, areas[e["eventid"]], usados)
        examinados.append({"eventid": e["eventid"], "tiles_elegiveis": len(t)})
        print(f"  GDACS:{e['eventid']} {e['inicio']}: {len(t)} tiles", flush=True)
        if t:
            escolhidos.append({**e, "evento_id": f"RFDBR:{e['eventid']}", "tiles": t})
            usados |= {x["tile"] for x in t}
        i += 1
    sel = {"tipo": "selecao_rfd_brasil", "quando_utc": agora_utc(), "pre_registro": h, "validacao_da_replica": v["pre_registro"],
           "candidatos": len(cand), "fora": fora, "examinados": examinados, "eventos": escolhidos,
           "nao_e": "nao e rotulo do GFD: e a replica do metodo, validada nos 19 eventos brasileiros do GFD"}
    (Q / "selecao.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    return sel


PONTOS_POR_CLASSE_NO_TILE = 1300   # o mesmo do gfdq02
EVENTOS_POR_TAREFA_DE_PONTOS = 4


def amostrar_pontos() -> int:
    """Pontos por tile escolhido, pelo mesmo stratifiedSample do gfd01/gfdq02, sobre o rotulo da replica."""
    import ee
    import pandas as pd
    from gfd01_eventos_do_brasil import FAIXA_M, PASSO_M, classes_do_rotulo
    from rfd01_replica_gfd import exportar_e_ler
    from gfdq02_derivar_quarentena import _rotulo
    ee.Initialize()
    sel = json.loads((Q / "selecao.json").read_text(encoding="utf-8"))
    saida = Q / "pontos"
    saida.mkdir(parents=True, exist_ok=True)
    # EVENTOS EM LOTE: o mesmo stratifiedSample por tile (mesma regiao, semente e grade), varios eventos
    # numa tarefa so -- cada tarefa espera 30-60 min na fila do modo restrito (05/10/2026)
    pendentes = [(ev, t) for ev in sel["eventos"] for t in ev["tiles"]
                 if not (saida / f"pontos_rfdbr{ev['eventid']}_{t['tile']}.csv").exists()]
    por_evento: dict = {}
    for ev, t in pendentes:
        por_evento.setdefault(ev["eventid"], (ev, []))[1].append(t)
    grupos = list(por_evento.values())
    for i in range(0, len(grupos), EVENTOS_POR_TAREFA_DE_PONTOS):
        lote = grupos[i:i + EVENTOS_POR_TAREFA_DE_PONTOS]
        amostras = []
        for ev, tiles in lote:
            img, _, grade = rotulo(ev)
            pos, neg = classes_do_rotulo(img, 250.0, FAIXA_M)
            rot = ee.Image(0).where(neg, 1).where(pos, 2).rename("classe_gfd").selfMask()
            for t in tiles:
                amostras.append(rot.addBands(ee.Image.pixelLonLat()).stratifiedSample(
                    numPoints=PONTOS_POR_CLASSE_NO_TILE, classBand="classe_gfd",
                    region=ee.Geometry.Rectangle(t["tile_1grau"]), scale=PASSO_M, projection=grade,
                    # geometries=True so porque exportar para asset recusa feicao sem geometria; o sorteio
                    # (regiao, semente, grade) e o mesmo, e lat/lon continuam vindo das bandas pixelLonLat
                    seed=20260911, geometries=True, tileScale=4
                ).map(lambda f, tt=t["tile"], ee_=ev["eventid"]: f.set({"tile_sel": tt, "evento_sel": ee_})))
        nome = "rfdbr_pontos_" + hashlib.sha256("|".join(ev["eventid"] for ev, _ in lote).encode()).hexdigest()[:12]
        linhas = exportar_e_ler(ee.FeatureCollection(amostras).flatten(), nome)
        for ev, tiles in lote:
            for t in tiles:
                la, lo = map(int, t["tile"].split("_"))
                sub = [x for x in linhas if x.get("tile_sel") == t["tile"] and str(x.get("evento_sel")) == ev["eventid"]
                       and x.get("latitude") is not None]
                pts = [{"ponto_id": f"RFDBR{ev['eventid']}_{t['tile']}_{k:05d}",
                        "classe_obs": "INUNDACAO" if int(x["classe_gfd"]) == 2 else "NAO_INUNDADO_OBSERVADO",
                        "lat": float(x["latitude"]), "lon": float(x["longitude"]), "evento_gfd": int(ev["eventid"]),
                        "grupo_cv": "GFD_" + _rotulo(la, lo), "tile": "gfd_tile_" + _rotulo(la, lo)}
                       for k, x in enumerate(sub)]
                pd.DataFrame(pts).to_csv(saida / f"pontos_rfdbr{ev['eventid']}_{t['tile']}.csv", index=False)
                print(f"  RFDBR:{ev['eventid']} {t['tile']}: {len(pts)} pontos", flush=True)
    return 0


def janelas() -> Path:
    from gfdq02_derivar_quarentena import _rotulo
    sel = json.loads((Q / "selecao.json").read_text(encoding="utf-8"))
    js = [{"nome": "gfd_tile_" + _rotulo(*map(int, t["tile"].split("_"))), "tile_1grau": t["tile_1grau"]}
          for ev in sel["eventos"] for t in ev["tiles"]]
    alvo = Q / "janelas.json"
    alvo.write_text(json.dumps(js, indent=1), encoding="utf-8")
    return alvo


def linhas() -> dict:
    """As regras do ds04/05/06 (espelho provado no gfdq02) sobre os pontos da replica."""
    import pandas as pd
    import gfdq02_derivar_quarentena as q2
    from gfd02_amostrar_terreno import _amostrar_no_tile
    prova = json.loads((q2.Q / "prova_do_espelho.json").read_text(encoding="utf-8"))
    if not prova.get("passou"):
        raise SystemExit("o espelho do ds06 nao esta provado (gfdq02 --provar-espelho)")
    p = pd.concat([pd.read_csv(f) for f in sorted((Q / "pontos").glob("pontos_rfdbr*.csv"))], ignore_index=True)
    partes, sem_terreno = [], []
    for tile, sub in p.groupby("tile"):
        if not (q2.TER / tile / "dem_filled.tif").exists():
            sem_terreno.append(tile)
            continue
        s = _amostrar_no_tile(sub, q2.TER / tile)
        partes.append(sub.reset_index(drop=True).assign(**{k: v for k, v in s.items()}))
    e, saiu = q2.espelho_do_ds06(pd.concat(partes, ignore_index=True))
    e["evento_id"] = "RFDBR:" + e.evento_gfd.astype(int).astype(str)
    e["fonte_do_rotulo"] = "replica_gfd_rfd01"
    e.to_parquet(Q / "pontos_quarentena.parquet", index=False)
    r = {"tipo": "pontos_quarentena_rfd_brasil", "quando_utc": agora_utc(), "pontos_amostrados": int(len(p)),
         "tiles_sem_terreno": sem_terreno, "linhas": int(len(e)), "saiu": saiu,
         "por_relevo": e.classe_relevo.value_counts().to_dict(), "eventos": int(e.evento_id.nunique()),
         "nao_e": "nao entra em ds03-ds06 antes do rfd03; rotulo da replica, nao do GFD"}
    (Q / "linhas_manifesto.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selecionar", action="store_true")
    ap.add_argument("--pontos", action="store_true")
    ap.add_argument("--janelas", action="store_true")
    ap.add_argument("--linhas", action="store_true")
    a = ap.parse_args()
    h = pre_registrar()
    if a.selecionar:
        s = selecionar()
        print(f"OK: {len(s['eventos'])} eventos, {sum(len(e['tiles']) for e in s['eventos'])} tiles")
    if a.pontos:
        amostrar_pontos()
    if a.janelas:
        print(janelas())
    if a.linhas:
        print(json.dumps(linhas(), ensure_ascii=False, indent=1))
    if not (a.selecionar or a.pontos or a.janelas or a.linhas):
        print(h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
