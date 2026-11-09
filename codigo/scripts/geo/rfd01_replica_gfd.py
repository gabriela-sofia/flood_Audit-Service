"""Replica do algoritmo do Global Flood Database (DFO/MODIS) para cheias posteriores a 2018.

-------------------------------------------------------------------------
DE ONDE VEM
-------------------------------------------------------------------------

Codigo publico do GFD (cloudtostreet/MODIS_GlobalFloodDatabase, licenca MIT,
Tellman et al. 2021, Nature): `flood_detection/modis.py`, `modis_toolbox.py`,
`utils/misc.py`. Reproduzido passo a passo:

  Terra+Aqua, GQ 250 m (b1, b2) unidos ao GA 500 m (b1, b3, b4, b7, state_1km)
  pansharpening do 500 m pela razao red_500/red_250
  razao (nir + 13,5) / (red + 1081,1) < b1b2;  red < 2027;  swir < b7
  agua no dia = os tres; composto de 3 dias = agua em >= 3 imagens Terra+Aqua
  (>= 2 antes de 04/07/2002); flooded = algum composto; duration = soma/2
  clear_views = imagens com cloud_state == 0 OU cloud_shadow == 0 (bits do state_1km)
  mascara de declive <= 5 (ee.Terrain.slope do GMTED2010); agua permanente JRC (transition == 1)
  janela: began - 2 dias ate ended + 3 dias

-------------------------------------------------------------------------
O QUE MUDA, POR FORCA, E E DECLARADO
-------------------------------------------------------------------------

  MODIS/006 foi descontinuado no Earth Engine -> MODIS/061 (recalibracao C6.1)
  JRC GSW 1.0 -> GSW 1.4 (mesma banda `transition`)
  bacias HydroSHEDS de Fusion Table (extinta) -> regiao de mapeamento dada aqui
  limiares por evento vinham do controle de qualidade do GFD; evento novo nao
  tem QC -> limiar padrao 0,70 / 675 (o que 17 dos 19 eventos brasileiros usaram)

-------------------------------------------------------------------------
O PORTAO: a replica so rotula evento novo se reproduzir o GFD nos conhecidos
-------------------------------------------------------------------------

`--validar` roda a replica nos 19 eventos brasileiros do GFD, com os limiares
que o proprio GFD gravou em cada um, e compara a CLASSE DO PROJETO
(`gfd01.classes_do_rotulo`: positivo interior / negativo na faixa) nos pontos
GFD ja amostrados. Criterio gravado em `pre_registro_validacao.json` antes do
primeiro resultado; se falhar, a replica nao rotula nada.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for pasta in ("servico", "geo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "rfd-replica-gfd"
PADRAO = {"b1b2": 0.70, "b7": 675.0}
PORTAO = {"concordancia_mediana_min": 0.90, "concordancia_evento_min": 0.80, "eventos_acima_min": 16,
          "cobertura_mediana_min": 0.80,
          "definicoes": {"concordancia": "entre pontos GFD rotulados que a replica tambem rotula, fracao com a mesma classe",
                         "cobertura": "fracao dos pontos GFD rotulados que a replica tambem rotula (positivo interior ou negativo na faixa)"}}


def _bandas_gq(c):
    return c.select(["sur_refl_b01", "sur_refl_b02"], ["red_250m", "nir_250m"])


def _bandas_ga(c):
    return c.select(["sur_refl_b01", "sur_refl_b03", "sur_refl_b04", "sur_refl_b07", "state_1km"],
                    ["red_500m", "blue", "green", "swir", "state_1km"])


def _unir(a, b):
    import ee
    j = ee.Join.inner().apply(a, b, ee.Filter.equals(leftField="system:time_start", rightField="system:time_start"))
    return ee.ImageCollection(j.map(lambda f: ee.Image.cat(f.get("primary"), f.get("secondary"))))


def _satelite(prefixo, roi, datas):
    import ee
    gq = ee.ImageCollection(f"MODIS/061/{prefixo}09GQ").filterDate(datas).filterBounds(roi)
    ga = ee.ImageCollection(f"MODIS/061/{prefixo}09GA").filterDate(datas).filterBounds(roi)
    return _unir(_bandas_gq(gq), _bandas_ga(ga))


def _preparar(img):
    import ee
    razao = img.select("red_500m").divide(img.select("red_250m"))
    ps = img.select("red_250m", "nir_250m", "state_1km").addBands(
        [img.select("blue").divide(razao), img.select("green").divide(razao), img.select("swir").divide(razao)])
    r = ps.expression("float(b('nir_250m') + 13.5) / float(b('red_250m') + 1081.1)").rename("b1b2_ratio")
    st = ps.select("state_1km")

    def bits(a, b, nome):
        return st.bitwiseAnd(sum(2 ** i for i in range(a, b + 1))).rightShift(a).rename(nome)
    return (ps.addBands(r).addBands([bits(0, 1, "cloud_state"), bits(2, 2, "cloud_shadow")])
            .copyProperties(img, ["system:time_start"]))


def dfo(roi, began: str, ended: str, b1b2: float, b7: float):
    """Imagem com flooded, duration, clear_views, jrc_perm_water -- composto de 3 dias."""
    import ee
    caixa = roi.bounds()
    datas = ee.DateRange(ee.Date(began).advance(-2, "day"), ee.Date(ended).advance(3, "day"))
    modis = (_satelite("MOD", caixa, datas).merge(_satelite("MYD", caixa, datas))
             .map(lambda i: _preparar(i.clip(caixa))).sort("system:time_start"))

    def agua(img):
        f = (img.select("b1b2_ratio").lt(b1b2).add(img.select("red_250m").lt(2027))
             .add(img.select("swir").lt(b7))).gte(3)
        return f.rename("w").set("system:time_start", img.get("system:time_start"))
    w = modis.map(agua)
    filtro = ee.Filter.And(ee.Filter.maxDifference(1000 * 60 * 60 * 24 * 2, leftField="system:time_start",
                                                   rightField="system:time_start"),
                           ee.Filter.greaterThanOrEquals(leftField="system:time_start", rightField="system:time_start"))
    j = ee.Join.saveAll(matchesKey="dfo_images", ordering="system:time_start").apply(w, w, filtro)
    minimo = 3 if began >= "2002-07-04" else 2
    comp = ee.ImageCollection(j.map(lambda i: ee.ImageCollection.fromImages(ee.Image(i).get("dfo_images"))
                                    .sum().gte(minimo).rename("flood_water")))
    freq = comp.sum().divide(2).toUint16()
    flooded = freq.gte(1).rename("flooded")
    limpas = modis.map(lambda i: i.select("cloud_state").eq(0).add(i.select("cloud_shadow").eq(0)).gt(0)
                       ).sum().rename("clear_views").toUint16()
    declive = ee.Terrain.slope(ee.Image("USGS/GMTED2010"))  # o do codigo original (ha um _FULL mais novo)
    perm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("transition").eq(1).unmask().rename("jrc_perm_water")
    return (flooded.addBands(freq.rename("duration")).addBands(limpas).updateMask(declive.lte(5))
            .addBands(perm).clip(roi))


def dfo_por_dia(roi, began: str, ended: str, b1b2: float, b7: float):
    """O mesmo resultado de `dfo`, com o composto montado por DIA e nao por par de imagens.

    No original cada imagem (2 por dia: Terra e Aqua) faz um `Join.saveAll` com as
    imagens dos 2 dias anteriores; como o produto diario tem `system:time_start`
    a 00:00, o composto de uma imagem do dia d e a soma de agua dos dias d-2..d,
    igual para as duas imagens do dia. Entao:

        composto_d = (agua_{d-2} + agua_{d-1} + agua_d) >= minimo
        soma       = sum_d  n_d * composto_d        (n_d = imagens do dia d)
        duration   = floor(soma / 2);  flooded = duration >= 1

    Mesma mascara (dia sem imagem fica mascarado, como a soma de colecao vazia).
    Custa O(dias) em vez de O(imagens^2) de metadados do join -- o que importa com
    a cota do Earth Engine em modo restrito (04/10/2026). Equivalencia provada
    em `provar_equivalencia`.
    """
    import ee
    caixa = roi.bounds()
    ini, fim = ee.Date(began).advance(-2, "day"), ee.Date(ended).advance(3, "day")
    modis = (_satelite("MOD", caixa, ee.DateRange(ini, fim)).merge(_satelite("MYD", caixa, ee.DateRange(ini, fim)))
             .map(lambda i: _preparar(i.clip(caixa))))

    def agua(img):
        return (img.select("b1b2_ratio").lt(b1b2).add(img.select("red_250m").lt(2027))
                .add(img.select("swir").lt(b7))).gte(3).rename("w")
    minimo = 3 if began >= "2002-07-04" else 2
    n_dias = fim.difference(ini, "day").round()
    dias = ee.List.sequence(0, n_dias.subtract(1))

    def do_dia(k):
        d = ini.advance(k, "day")
        imgs = modis.filterDate(d, d.advance(1, "day"))
        janela = modis.filterDate(d.advance(-2, "day"), d.advance(1, "day"))
        comp = janela.map(agua).sum().gte(minimo)
        return comp.multiply(imgs.size()).rename("s").set("n", imgs.size())
    soma = ee.ImageCollection(dias.map(do_dia)).filter(ee.Filter.gt("n", 0)).sum()
    freq = soma.divide(2).toUint16()
    flooded = freq.gte(1).rename("flooded")
    limpas = modis.map(lambda i: i.select("cloud_state").eq(0).add(i.select("cloud_shadow").eq(0)).gt(0)
                       ).sum().rename("clear_views").toUint16()
    declive = ee.Terrain.slope(ee.Image("USGS/GMTED2010"))
    perm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("transition").eq(1).unmask().rename("jrc_perm_water")
    return (flooded.addBands(freq.rename("duration")).addBands(limpas).updateMask(declive.lte(5))
            .addBands(perm).clip(roi))


def pre_registrar() -> str:
    SAIDA.mkdir(parents=True, exist_ok=True)
    alvo = SAIDA / "pre_registro_validacao.json"
    corpo = {"portao": PORTAO, "eventos": "os 19 eventos GFD brasileiros (gfd-brasil/catalogo.json)",
             "limiares": "os gravados pelo GFD em cada evento (threshold_b1b2, threshold_b7)",
             "comparacao": "classe do projeto (classes_do_rotulo, faixa 2 km, margem 250 m) nos pontos GFD ja amostrados",
             "operacional_para_evento_novo": {"limiares": PADRAO, "composto": "3Day"}}
    h = sha256_json(corpo)
    if alvo.exists():
        a = json.loads(alvo.read_text(encoding="utf-8"))
        if a["hash"] != h:
            raise RuntimeError(f"criterio da validacao mudou depois do pre-registro ({a['hash']} -> {h})")
        return h
    alvo.write_text(json.dumps({"quando_utc": agora_utc(), "hash": h, "criterio": corpo}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return h


def classe_nos_pontos(img_rotulo, esc, pts, proj, lote: int = 1000) -> list:
    """Classe do projeto (0 nenhuma, 1 negativo na faixa, 2 positivo interior) em cada ponto.

    Avaliada na MESMA grade em que o gfd01 amostrou: a projecao do `flooded` do
    GFD na escala max(250 m, nativa). Sem fixar a grade, `ee.Image(0)` nasce em
    EPSG:4326 de 1 grau e a transformada de distancia da faixa sai em outra
    unidade -- o autoteste GFD x GFD deu 0,64 assim (04/10/2026).
    """
    import ee
    from gfd01_eventos_do_brasil import FAIXA_M, PASSO_M, classes_do_rotulo
    grade = proj.atScale(max(PASSO_M, esc))
    pos, neg = classes_do_rotulo(img_rotulo.reproject(grade), esc, FAIXA_M)
    cls = ee.Image(0).where(neg, 1).where(pos, 2).rename("c").unmask(0).reproject(grade)
    # pontos ordenados por tile de 1 grau: cada consulta toca pouca area; lote que estoura o tempo
    # interativo do Earth Engine e partido em dois e reenviado (nao muda o valor de nenhum ponto)
    ordem = sorted(range(len(pts)), key=lambda k: (int(pts[k][0] // 1), int(pts[k][1] // 1)))
    pendentes = [ordem[i:i + lote] for i in range(0, len(ordem), lote)]
    out = {}
    while pendentes:
        ks = pendentes.pop()
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([pts[k][1], pts[k][0]]), {"k": k}) for k in ks])
        try:
            r = cls.reduceRegions(fc, ee.Reducer.first(), crs=grade, tileScale=8).getInfo()
        except ee.EEException as e:
            if "timed out" not in str(e) or len(ks) <= 50:
                raise
            pendentes += [ks[:len(ks) // 2], ks[len(ks) // 2:]]
            continue
        out.update({f["properties"]["k"]: f["properties"].get("first") for f in r["features"]})
    return [out.get(k) for k in range(len(pts))]


def validar() -> dict:
    import ee
    import pandas as pd
    ee.Initialize()
    h = pre_registrar()
    pts = pd.read_csv(REPO / "local_runs" / "gfd-brasil" / "pontos_gfd_com_terreno.csv",
                      usecols=["ponto_id", "evento_gfd", "classe_obs", "lat", "lon"]).drop_duplicates("ponto_id")
    col = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")
    feito = SAIDA / "validacao_por_evento.json"
    res = json.loads(feito.read_text(encoding="utf-8")) if feito.exists() else {}
    # AGUA_PERMANENTE e reclassificacao do gfd02 a 30 m (JRC occurrence) e sai do dataset: fica fora
    # da comparacao. Autoteste GFD x GFD nas outras duas classes: 498/498 (evento 3810, 04/10/2026).
    pts = pts[pts.classe_obs.isin(["INUNDACAO", "NAO_INUNDADO_OBSERVADO"])]
    # SO os 19 eventos do catalogo brasileiro, como o pre-registro diz. A primeira execucao percorreu
    # todos os eventos com ponto no arquivo (inclui os que so tocam o Brasil); o que ela calculou
    # fora dos 19 fica gravado como descritivo e nao entra no portao (04/10/2026).
    brasil = {e["id"] for e in json.loads((REPO / "local_runs" / "gfd-brasil" / "catalogo.json")
                                          .read_text(encoding="utf-8"))["eventos"]}
    pts = pts[pts.evento_gfd.isin(brasil)]
    for ev, g in pts.groupby("evento_gfd"):
        if str(ev) in res:
            continue
        gfd = ee.Image(col.filter(ee.Filter.eq("id", int(ev))).first())
        p = gfd.toDictionary(["threshold_b1b2", "threshold_b7"]).getInfo()
        ini = ee.Date(gfd.get("system:time_start")).format("yyyy-MM-dd").getInfo()
        fim = ee.Date(gfd.get("system:time_end")).format("yyyy-MM-dd").getInfo()
        esc = gfd.select("flooded").projection().nominalScale().getInfo()
        # a regiao do mapeamento e o footprint do GFD recortado aos pontos + 10 km: a classe de um ponto so
        # depende de pixels a menos de 2 km (faixa) + 250 m (margem), entao o recorte nao muda nenhuma classe
        caixa = ee.Geometry.Rectangle([float(g.lon.min()), float(g.lat.min()), float(g.lon.max()), float(g.lat.max())])
        roi = gfd.geometry().intersection(caixa.buffer(10_000), maxError=100)
        proj = gfd.select("flooded").projection()
        rep = dfo(roi, ini, fim, p["threshold_b1b2"], p["threshold_b7"])
        ll = list(zip(g.lat, g.lon))
        c_rep = classe_nos_pontos(rep, esc, ll, proj)
        c_gfd = (g.classe_obs == "INUNDACAO").map({True: 2, False: 1}).tolist()
        ambos = [(a, b) for a, b in zip(c_gfd, c_rep) if b in (1, 2)]
        res[str(ev)] = {"pontos": len(ll), "cobertura": round(len(ambos) / len(ll), 4),
                        "concordancia": round(sum(a == b for a, b in ambos) / len(ambos), 4) if ambos else None,
                        "positivos_gfd_que_a_replica_chama_de_negativo": int(sum(1 for a, b in ambos if a == 2 and b == 1)),
                        "negativos_gfd_que_a_replica_chama_de_positivo": int(sum(1 for a, b in ambos if a == 1 and b == 2)),
                        "limiares": p, "janela": [ini, fim]}
        feito.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  GFD:{ev}: {res[str(ev)]['concordancia']} conc., {res[str(ev)]['cobertura']} cob.", flush=True)
    import numpy as np
    fora = {k: v for k, v in res.items() if int(k) not in brasil}
    res = {k: v for k, v in res.items() if int(k) in brasil}
    conc = [v["concordancia"] for v in res.values() if v["concordancia"] is not None]
    cob = [v["cobertura"] for v in res.values()]
    passou = (np.median(conc) >= PORTAO["concordancia_mediana_min"]
              and sum(c >= PORTAO["concordancia_evento_min"] for c in conc) >= PORTAO["eventos_acima_min"]
              and np.median(cob) >= PORTAO["cobertura_mediana_min"])
    r = {"tipo": "validacao_replica_gfd", "quando_utc": agora_utc(), "pre_registro": h, "eventos": len(res),
         "concordancia_mediana": round(float(np.median(conc)), 4), "cobertura_mediana": round(float(np.median(cob)), 4),
         "eventos_com_concordancia_acima_do_minimo": int(sum(c >= PORTAO["concordancia_evento_min"] for c in conc)),
         "passou": bool(passou), "por_evento": res, "fora_do_pre_registro_descritivo": fora,
         "nao_e": "nao e validacao contra verdade de campo: e reproducao do produto GFD pelo mesmo metodo"}
    (SAIDA / "validacao.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


PASTA_EE = "projects/projetotransformador01/assets/rfd_validacao"


def exportar_e_ler(fc, nome: str, espera_s: int = 30) -> list[dict]:
    """Calcula `fc` como TAREFA (sem o limite interativo) num asset do projeto e devolve as propriedades.

    Se o asset ja existe, so le: a tarefa nao roda duas vezes. Falha alto se a tarefa nao completar.
    """
    import time
    import ee
    alvo = f"{PASTA_EE}/{nome}"
    try:
        ee.data.createAsset({"type": "FOLDER"}, PASTA_EE)
    except ee.EEException:
        pass
    try:
        ee.data.getAsset(alvo)
    except ee.EEException:
        t = ee.batch.Export.table.toAsset(fc, description=nome[:100], assetId=alvo)
        t.start()
        while t.status()["state"] in ("READY", "RUNNING"):
            time.sleep(espera_s)
        st = t.status()
        if st["state"] != "COMPLETED":
            raise RuntimeError(f"tarefa {nome} terminou em {st['state']}: {st.get('error_message')}")
    tab = ee.FeatureCollection(alvo)
    n = tab.size().getInfo()
    out = []
    for i in range(0, n, 5000):
        out += [f["properties"] for f in tab.toList(5000, i).getInfo()]
    return out


def _imagem_de_classe(img_rotulo, esc, proj):
    import ee
    from gfd01_eventos_do_brasil import FAIXA_M, PASSO_M, classes_do_rotulo
    grade = proj.atScale(max(PASSO_M, esc))
    pos, neg = classes_do_rotulo(img_rotulo.reproject(grade), esc, FAIXA_M)
    return ee.Image(0).where(neg, 1).where(pos, 2).rename("c").unmask(0).reproject(grade), grade


BLOCO_GRAUS = 3


def _blocos(g):
    """Bloco de 3 graus de cada ponto: regiao ~50x menor que a do evento inteiro, poucas tarefas."""
    return ((g.lat // BLOCO_GRAUS).astype(int).astype(str) + "_" + (g.lon // BLOCO_GRAUS).astype(int).astype(str))


def validar_em_lote(espera_s: int = 60) -> dict:
    """A mesma validacao de `validar`, com a amostragem como TAREFA do Earth Engine.

    Em modo restrito (cota mensal do nivel nao comercial esgotada, 04/10/2026) a
    consulta interativa estoura o limite de tempo nos eventos longos; tarefas em
    lote continuam permitidas, so mais lentas. Cada evento vira uma exportacao da
    classe nos pontos para um asset do proprio projeto; o calculo, os pontos e o
    portao sao os de `validar`.
    """
    import time
    import ee
    import numpy as np
    import pandas as pd
    ee.Initialize()
    h = pre_registrar()
    try:
        ee.data.createAsset({"type": "FOLDER"}, PASTA_EE)
    except ee.EEException:
        pass  # ja existe
    brasil = {e["id"] for e in json.loads((REPO / "local_runs" / "gfd-brasil" / "catalogo.json")
                                          .read_text(encoding="utf-8"))["eventos"]}
    pts = pd.read_csv(REPO / "local_runs" / "gfd-brasil" / "pontos_gfd_com_terreno.csv",
                      usecols=["ponto_id", "evento_gfd", "classe_obs", "lat", "lon"]).drop_duplicates("ponto_id")
    pts = pts[pts.classe_obs.isin(["INUNDACAO", "NAO_INUNDADO_OBSERVADO"]) & pts.evento_gfd.isin(brasil)]
    col = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")
    feito = SAIDA / "validacao_por_evento.json"
    res = json.loads(feito.read_text(encoding="utf-8")) if feito.exists() else {}
    for ev, g in pts.groupby("evento_gfd"):
        if str(ev) in res:
            continue
        alvo = f"{PASTA_EE}/gfd_{int(ev)}"
        try:
            ee.data.getAsset(alvo)
            existe = True
        except ee.EEException:
            existe = False
        partes = sorted(SAIDA.glob(f"_partes_{int(ev)}.json"))
        if not existe and partes:
            existe = True  # ja repartido: le as partes abaixo
        if not existe:
            gfd = ee.Image(col.filter(ee.Filter.eq("id", int(ev))).first())
            p = gfd.toDictionary(["threshold_b1b2", "threshold_b7"]).getInfo()
            ini = ee.Date(gfd.get("system:time_start")).format("yyyy-MM-dd").getInfo()
            fim = ee.Date(gfd.get("system:time_end")).format("yyyy-MM-dd").getInfo()
            proj = gfd.select("flooded").projection()
            esc = proj.nominalScale().getInfo()
            caixa = ee.Geometry.Rectangle([float(g.lon.min()), float(g.lat.min()), float(g.lon.max()), float(g.lat.max())])
            roi = gfd.geometry().intersection(caixa.buffer(10_000), maxError=100)
            cls, grade = _imagem_de_classe(dfo(roi, ini, fim, p["threshold_b1b2"], p["threshold_b7"]), esc, proj)
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]), {"ponto_id": pid})
                                       for pid, la, lo in zip(g.ponto_id, g.lat, g.lon)])
            t = ee.batch.Export.table.toAsset(cls.reduceRegions(fc, ee.Reducer.first(), crs=grade, tileScale=8),
                                              description=f"rfd_val_{int(ev)}", assetId=alvo)
            t.start()
            print(f"  GFD:{ev}: tarefa {t.id} ({len(g)} pontos, {ini}..{fim})", flush=True)
            while t.status()["state"] in ("READY", "RUNNING"):
                time.sleep(espera_s)
            st = t.status()
            json.dump({"ini": ini, "fim": fim, "limiares": p}, open(SAIDA / f"_meta_{int(ev)}.json", "w"))
            if st["state"] != "COMPLETED":
                if "timed out" not in str(st.get("error_message", "")):
                    raise RuntimeError(f"tarefa do evento {ev} terminou em {st['state']}: {st.get('error_message')}")
                # REPARTIR POR BLOCO DE 3 GRAUS: pontos espalhados (GFD 4159 vai de 19S a 4N) dao regiao gigante; cada tile
                # de 1 grau com a propria caixa + 10 km da a MESMA classe por ponto (so pixels a < 2,25 km contam)
                tiles = _blocos(g)
                json.dump(sorted(tiles.unique()), open(SAIDA / f"_partes_{int(ev)}.json", "w"))
                print(f"  GFD:{ev}: tempo esgotado no evento inteiro; repartindo em {tiles.nunique()} blocos", flush=True)
        meta = json.load(open(SAIDA / f"_meta_{int(ev)}.json"))
        if (SAIDA / f"_partes_{int(ev)}.json").exists():
            gfd = ee.Image(col.filter(ee.Filter.eq("id", int(ev))).first())
            proj = gfd.select("flooded").projection()
            esc = proj.nominalScale().getInfo()
            tiles = _blocos(g)
            linhas = []
            for tl, sub in g.groupby(tiles):
                caixa = ee.Geometry.Rectangle([float(sub.lon.min()), float(sub.lat.min()),
                                               float(sub.lon.max()), float(sub.lat.max())])
                roi = gfd.geometry().intersection(caixa.buffer(10_000), maxError=100)
                cls, grade = _imagem_de_classe(dfo(roi, meta["ini"], meta["fim"], meta["limiares"]["threshold_b1b2"],
                                                   meta["limiares"]["threshold_b7"]), esc, proj)
                fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]), {"ponto_id": pid})
                                           for pid, la, lo in zip(sub.ponto_id, sub.lat, sub.lon)])
                linhas += exportar_e_ler(cls.reduceRegions(fc, ee.Reducer.first(), crs=grade, tileScale=8),
                                         f"gfd_{int(ev)}_t{tl.replace('-', 'm')}")
        else:
            tab = ee.FeatureCollection(alvo)
            n = tab.size().getInfo()
            linhas = []
            for i in range(0, n, 5000):
                linhas += [f["properties"] for f in tab.toList(5000, i).getInfo()]
        c = pd.DataFrame(linhas).set_index("ponto_id")["first"].reindex(g.ponto_id)
        c_gfd = (g.classe_obs == "INUNDACAO").map({True: 2, False: 1}).to_numpy()
        ambos = [(a, b) for a, b in zip(c_gfd, c.to_numpy()) if b in (1, 2)]
        res[str(ev)] = {"pontos": int(len(g)), "cobertura": round(len(ambos) / len(g), 4),
                        "concordancia": round(sum(a == b for a, b in ambos) / len(ambos), 4) if ambos else None,
                        "positivos_gfd_que_a_replica_chama_de_negativo": int(sum(1 for a, b in ambos if a == 2 and b == 1)),
                        "negativos_gfd_que_a_replica_chama_de_positivo": int(sum(1 for a, b in ambos if a == 1 and b == 2)),
                        "limiares": meta["limiares"], "janela": [meta["ini"], meta["fim"]], "via": "tarefa_em_lote"}
        feito.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  GFD:{ev}: {res[str(ev)]['concordancia']} conc., {res[str(ev)]['cobertura']} cob.", flush=True)
    fora = {k: v for k, v in res.items() if int(k) not in brasil}
    res = {k: v for k, v in res.items() if int(k) in brasil}
    conc = [v["concordancia"] for v in res.values() if v["concordancia"] is not None]
    cob = [v["cobertura"] for v in res.values()]
    passou = (np.median(conc) >= PORTAO["concordancia_mediana_min"]
              and sum(c >= PORTAO["concordancia_evento_min"] for c in conc) >= PORTAO["eventos_acima_min"]
              and np.median(cob) >= PORTAO["cobertura_mediana_min"])
    r = {"tipo": "validacao_replica_gfd", "quando_utc": agora_utc(), "pre_registro": h, "eventos": len(res),
         "concordancia_mediana": round(float(np.median(conc)), 4), "cobertura_mediana": round(float(np.median(cob)), 4),
         "eventos_com_concordancia_acima_do_minimo": int(sum(c >= PORTAO["concordancia_evento_min"] for c in conc)),
         "passou": bool(passou), "por_evento": res, "fora_do_pre_registro_descritivo": fora,
         "nao_e": "nao e validacao contra verdade de campo: e reproducao do produto GFD pelo mesmo metodo"}
    (SAIDA / "validacao.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-registrar", action="store_true")
    ap.add_argument("--validar", action="store_true")
    ap.add_argument("--validar-em-lote", action="store_true",
                    help="a mesma validacao, com a amostragem como tarefa do Earth Engine (modo restrito)")
    a = ap.parse_args()
    if a.validar or a.validar_em_lote:
        r = validar_em_lote() if a.validar_em_lote else validar()
        print(json.dumps({k: v for k, v in r.items() if k != "por_evento"}, ensure_ascii=False, indent=1))
    else:
        print(pre_registrar())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
