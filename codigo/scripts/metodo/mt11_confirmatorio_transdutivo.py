"""MT11: teste confirmatório em unidades GFD nunca examinadas — pesos Q e via transdutiva.

Três hipóteses primárias, registradas antes de qualquer rótulo ser lido (IC 1-0,05/3 por
Bonferroni, margem +0,01 de AUC dentro do evento, unidade = bacia principal):

  HQ1  Q_conexao - A0_regra_hand   (no desenvolvimento MT05: ~+0,027)
  HQ2  Q_conexao - Q_base          (contribuição da conexão; +0,010 em todos, +0,027 no fluvial)
  HF   F_transf - Q_conexao        (frequência SAR do lugar, só datas antes do evento; MT10 +0,014)

Modelos congelados: Q_base e Q_conexao de `v5/sem_amostra_v1/candidatas/modelos.pkl` (MT05,
ajustados em todo o desenvolvimento, nenhuma reserva lida); F_transf = logística [Q, F, QxF]
com intercepto por evento, ajustada aqui UMA vez nos componentes CEMS do MT10 com F só
pré-evento e Q fora do componente (predição MT06), antes do pré-registro.

Fases (cada uma recusa rodar se a anterior não estiver gravada e com hash):
  congelar      ajusta e exporta F_transf; confere hash dos pesos Q
  registrar     grava protocolo.json (regras, hashes de código, candidatos, modelos, poder)
  selecionar    1 tile de 1 grau por evento: >= 5 km2 de cada classe (área do rótulo, só para
                saber se dá para amostrar), fora de tile do ds06/quarentena, sem dividir tile;
                ordem sha256("mt11|id")
  pontos        stratifiedSample do rótulo GFD (máscara gfd01), 1300 por classe no tile
  janelas       janelas para `cadeia_em_lote` (FABDEM + WhiteboxTools), roda à parte
  atributos     terreno nos pontos + espelho do ds06, distâncias (MT02), declive_p50 do raster,
                cobertura (cob01, scale 30), frequência S1 pré-evento nos pontos e qualidade
                da referência numa amostra aleatória do tile (sem rótulo)
  avaliar       escores congelados, AUC por evento, média por unidade, contrastes; registra os
                eventos como examinados
Uso: python scripts/metodo/mt11_confirmatorio_transdutivo.py <fase>
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import pickle
import sys
import time
import urllib.request
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "2")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for p in ("metodo", "geo", "servico", "suscetibilidade", "terreno", "sar"):
    sys.path.insert(0, str(REPO / "scripts" / p))
import mt05_transferencia_sem_amostra as mt  # noqa: E402
import mt11_candidatos_transdutivo as cand  # noqa: E402
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402

SAIDA = cand.SAIDA
TER = REPO / "local_runs" / "ter-01-cadeia-harmonizada"
Q_PKL = REPO / "local_runs/mecanismos-transferiveis/v5/sem_amostra_v1/candidatas/modelos.pkl"
Q_PKL_SHA = "sha256:945b6758761f746f3480ca48c6526760aa24d1a535231652c1001b6e9ea086bd"
QUARENTENA = REPO / "local_runs/gfd-quarentena/selecao.json"
EXAMINADOS = REPO / "local_runs/reservas-examinadas/examinados_apos_congelamento.jsonl"
SAL = "mt11|"
CFG = {
    "tiles_por_evento": 1, "min_km2_por_classe": 5.0, "pontos_por_classe_no_tile": 1300, "semente_pontos": 20260911,
    "janela_pre_dias": [425, 60], "max_datas": 60, "limiar_vv_db": -14.5, "e1_delta_db": 3.0, "e1_quantil": 75,
    "min_datas_pre": 10, "cobertura_minima_data": .9, "vv_mediana_admissivel": [-25.0, -3.0], "lat_neve": 40.0,
    "arvore_max": .5, "construido_max": .5, "declive_max_graus": 15.0, "jrc_permanente": 50,
    "qualidade_terra_max": .10, "qualidade_contraste_min": .05, "pontos_qualidade": 2000,
    "lam_f": {"p": .01, "i": .01}, "bootstrap": 2000, "semente_bootstrap": 7, "nivel": 1 - .05 / 3,
    "margem_auc": .01, "minimo_unidades": 8, "minimo_pontos_evento": 20,
}
HIPOTESES = {"HQ1": ("Q_conexao", "A0_regra_hand"), "HQ2": ("Q_conexao", "Q_base"), "HF": ("F_transf", "Q_conexao")}


def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def escrever(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_text(json.dumps(mt.rel.limpo(obj), ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, p)


def exigir(nome):
    if not (SAIDA / nome).exists():
        raise SystemExit(f"fase anterior ausente: {nome}")


# ------------------------------------------------------------------ congelar

def congelar():
    if (SAIDA / "f_transf.json").exists():
        return ler(SAIDA / "f_transf.json")
    if sha256_arquivo(Q_PKL) != Q_PKL_SHA:
        raise ValueError("pesos Q do MT05 mudaram")
    import mt10_sensibilidade_pre_evento as s10
    t, _ = s10.tabela_pre_evento()
    m = mt.e1.Logistica(["Q", "F"], [("Q", "F")], lam_p=CFG["lam_f"]["p"], lam_i=CFG["lam_f"]["i"])
    m.fit(t[["Q", "F"]], t.classe.to_numpy(float), t.evento_id.to_numpy())
    if not m.convergiu:
        raise ValueError("F_transf não convergiu")
    exp = {"coeficientes": dict(zip(m.principais, m.b.tolist())),
           "interacoes": [{"colunas": list(a), "peso": float(v)} for a, v in zip(m.interacoes, m.g)],
           "padronizador": {"colunas": m.pad.mu.index.tolist(), "media": m.pad.mu.to_dict(),
                            "desvio": m.pad.sd.to_dict(), "mediana": m.pad.med.to_dict()}}
    with (SAIDA / "f_transf.pkl").open("wb") as f:
        pickle.dump(m, f)
    r = {"tipo": "mt11_f_transf_congelado", "quando_utc": agora_utc(), "pontos": len(t), "eventos": int(t.evento_id.nunique()),
         "componentes": int(t.unidade.nunique()), "exportado": exp, "pkl_sha256": sha256_arquivo(SAIDA / "f_transf.pkl"),
         "q_pkl_sha256": Q_PKL_SHA, "fonte": "MT10 pré-evento (mt10_sensibilidade_pre_evento.tabela_pre_evento)",
         "nao_e": "ajuste em desenvolvimento conhecido; nenhum ponto GFD novo foi lido"}
    escrever(SAIDA / "f_transf.json", r)
    return r


def escore_exportado(x: pd.DataFrame, e: dict) -> np.ndarray:
    pad = e["padronizador"]
    z = {c: ((x[c].fillna(pad["mediana"][c]) - pad["media"][c]) / pad["desvio"][c]).fillna(0.).to_numpy() for c in pad["colunas"]}
    s = sum(z[c] * b for c, b in e["coeficientes"].items())
    for t in e["interacoes"]:
        a, b = t["colunas"]
        s = s + z[a] * z[b] * t["peso"]
    return np.asarray(s, float)


# ------------------------------------------------------------------ registrar

def registrar():
    exigir("f_transf.json")
    arq = SAIDA / "protocolo.json"
    codigos = [Path(__file__), Path(cand.__file__), Path(mt.__file__), Path(mt.e1.__file__),
               REPO / "scripts/metodo/mt10_sensibilidade_pre_evento.py", REPO / "scripts/metodo/mt11_poder.py",
               REPO / "scripts/geo/gfdq02_derivar_quarentena.py", REPO / "scripts/geo/gfd02_amostrar_terreno.py",
               REPO / "scripts/metodo/mt02_atributos.py", REPO / "scripts/geo/cob01_cobertura_e_solo.py",
               REPO / "scripts/terreno/cadeia_em_lote.py"]
    p = {"tipo": "MT11_pre_registro_antes_de_qualquer_rotulo", "config": CFG, "hipoteses": HIPOTESES,
         "candidatos_csv_sha256": sha256_arquivo(SAIDA / "candidatos.csv"),
         "candidatos_hash_lista": ler(SAIDA / "candidatos.json")["hash_lista"],
         "poder_sha256": sha256_arquivo(SAIDA / "poder.json"), "q_pkl_sha256": Q_PKL_SHA,
         "f_transf_sha256": sha256_arquivo(SAIDA / "f_transf.json"),
         "codigo_sha256": {str(c.relative_to(REPO)): sha256_arquivo(c) for c in codigos},
         "regras": {
             "unidade": "MAIN_BAS (HydroBASINS nível 8) do centro do tile escolhido; sem bacia no centro, a do centróide do evento",
             "evento_avaliavel": ">= 20 pontos e as duas classes depois do espelho do ds06",
             "valor_da_unidade": "média igual dos eventos da unidade",
             "decisao": "sustentado: média >= +0,01 e limite inferior > 0; ganho_excluido: limite superior < +0,01; "
                        "inferior: limite superior < 0; senão indecidível; < 8 unidades: not_testable",
             "HF_populacao": "pontos observáveis (árvore <= 0,5, construído <= 0,5, declive <= 15 graus, JRC < 50) com >= 10 "
                             "datas S1 admitidas antes do evento, em eventos (tile) cuja referência passa a qualidade JRC "
                             "(E1 se passa, senão v1); Q_conexao é avaliado no MESMO subconjunto para o contraste HF",
             "F": "VV multilook 30 m na órbita com mais datas no tile, datas em [início-425 d, início-60 d], no máximo as 60 "
                  "mais recentes; admissão por data nos pontos: >= 90% com leitura, mediana VV em [-25,-3] dB, sem mês de "
                  "neve acima de 40 graus; v1: VV <= -14,5 dB; E1: v1 e VV <= P75 do ponto - 3 dB",
             "qualidade": "amostra aleatória de 2000 pixels do tile (sem rótulo): detecção média em JRC=0 <= 0,10 e "
                          "sazonal (JRC 10-49) - terra >= 0,05",
             "sem_analise_interina": True, "examinados": "os eventos avaliados entram em examinados_apos_congelamento.jsonl"},
         "ressalvas_antes_da_leitura": [
             "evento novo e território novo ao mesmo tempo (Ásia, África, Europa, Oceania, EUA) contra treino Brasil/CEMS",
             "rótulo MODIS 250 m sobre pontos a 30 m; positivo interior e negativo na faixa de 2 km (gfd01)",
             "F_transf foi ajustado com Q fora do componente (MT06); aqui Q vem do modelo único congelado",
             "Earth Engine em modo restrito: falha de coleta vira evento sem F, nunca valor imputado"],
         "regime": "confirmatório; unidades nunca examinadas", "modelo_promovido": False}
    if arq.exists():
        old = ler(arq)
        if {k: v for k, v in old.items() if k != "quando_utc"} != mt.rel.limpo(p):
            raise ValueError("pré-registro mudou; não reescrever")
        return old
    p["quando_utc"] = agora_utc()
    escrever(arq, p)
    return p


def conferir_registro():
    exigir("protocolo.json")
    p = ler(SAIDA / "protocolo.json")
    for f, h in p["codigo_sha256"].items():
        if sha256_arquivo(REPO / f) != h:
            raise ValueError("código mudou após o pré-registro: " + f)
    if sha256_arquivo(SAIDA / "candidatos.csv") != p["candidatos_csv_sha256"]:
        raise ValueError("candidatos mudaram")
    return p


# ------------------------------------------------------------------ selecionar

def _ordem(texto: str) -> str:
    return hashlib.sha256((SAL + texto).encode()).hexdigest()


def tiles_elegiveis(ee, id_evento: int, proibidos: set) -> list[dict]:
    from gfd01_eventos_do_brasil import COLECAO, FAIXA_M, classes_do_rotulo
    img = ee.Image(ee.ImageCollection(COLECAO).filter(ee.Filter.eq("id", id_evento)).first())
    esc = img.select("flooded").projection().nominalScale().getInfo()
    pos, neg = classes_do_rotulo(img, esc, FAIXA_M)
    b = img.geometry().bounds().coordinates().getInfo()[0]
    xs, ys = [c[0] for c in b], [c[1] for c in b]
    cel = [ee.Feature(ee.Geometry.Rectangle([lo, la, lo + 1, la + 1]), {"t": f"{la}_{lo}"})
           for la in range(math.floor(min(ys)), math.ceil(max(ys)))
           for lo in range(math.floor(min(xs)), math.ceil(max(xs))) if f"{la}_{lo}" not in proibidos]
    if not cel:
        return []
    area = ee.Image.pixelArea().divide(1e6)
    duas = area.updateMask(pos).rename("pos_km2").addBands(area.updateMask(neg).rename("neg_km2"))
    r = duas.reduceRegions(ee.FeatureCollection(cel), ee.Reducer.sum(), scale=esc, tileScale=4).getInfo()
    ok = []
    for f in r["features"]:
        pr = f["properties"]
        p_, n_ = pr.get("pos_km2") or 0., pr.get("neg_km2") or 0.
        if p_ >= CFG["min_km2_por_classe"] and n_ >= CFG["min_km2_por_classe"]:
            la, lo = map(int, pr["t"].split("_"))
            ok.append({"tile": pr["t"], "tile_1grau": [lo, la, lo + 1, la + 1], "pos_km2": round(p_, 1), "neg_km2": round(n_, 1)})
    return sorted(ok, key=lambda t: _ordem(f"{id_evento}|{t['tile']}"))


def selecionar():
    conferir_registro()
    if (SAIDA / "selecao.json").exists():
        return ler(SAIDA / "selecao.json")
    import ee
    ee.Initialize()
    c = pd.read_csv(SAIDA / "candidatos.csv")
    c = c[c.elegivel].copy()
    c["ordem"] = [_ordem(e) for e in c.evento_id]
    proibidos = cand.tiles_ds06()
    proibidos |= {t["tile"] for e in ler(QUARENTENA)["eventos"] for t in e["tiles"]}
    achar = cand.localizador()
    eventos, examinados = [], []
    for r in c.sort_values("ordem").itertuples():
        idn = int(r.evento_id.split(":")[1])
        t = tiles_elegiveis(ee, idn, proibidos)
        examinados.append({"evento_id": r.evento_id, "tiles_elegiveis": len(t)})
        print(r.evento_id, r.pais, len(t), flush=True)
        if not t:
            continue
        esc = t[:CFG["tiles_por_evento"]]
        proibidos |= {x["tile"] for x in esc}
        lo, la = esc[0]["tile_1grau"][0] + .5, esc[0]["tile_1grau"][1] + .5
        _, _, mb = achar(lo, la)
        mb = mb if mb is not None else (int(r.main_bas) if pd.notna(r.main_bas) else None)
        eventos.append({"evento_id": r.evento_id, "id": idn, "pais": r.pais, "inicio": r.inicio,
                        "tiles": esc, "unidade": f"BAS:{mb}" if mb else None})
    s = {"tipo": "mt11_selecao", "quando_utc": agora_utc(), "protocolo_sha256": sha256_arquivo(SAIDA / "protocolo.json"),
         "examinados": examinados, "eventos": eventos, "unidades": len({e["unidade"] for e in eventos}),
         "nota": "área do rótulo lida só para saber se dá para amostrar; nenhum ponto nem modelo visto"}
    escrever(SAIDA / "selecao.json", s)
    return {"eventos": len(eventos), "unidades": s["unidades"]}


# ------------------------------------------------------------------ pontos e janelas

def _rotulo(la: int, lo: int) -> str:
    from cadeia_em_lote import _rotulo_do_tile
    return _rotulo_do_tile(la, lo)


def pontos():
    conferir_registro()
    exigir("selecao.json")
    import ee
    from gfd01_eventos_do_brasil import COLECAO, FAIXA_M, PASSO_M, classes_do_rotulo
    ee.Initialize()
    pasta = SAIDA / "pontos"
    pasta.mkdir(parents=True, exist_ok=True)
    for ev in ler(SAIDA / "selecao.json")["eventos"]:
        img = ee.Image(ee.ImageCollection(COLECAO).filter(ee.Filter.eq("id", ev["id"])).first())
        esc = img.select("flooded").projection().nominalScale().getInfo()
        pos, neg = classes_do_rotulo(img, esc, FAIXA_M)
        rot = ee.Image(0).where(neg, 1).where(pos, 2).rename("classe_gfd").selfMask()
        for t in ev["tiles"]:
            alvo = pasta / f"pontos_gfd{ev['id']}_{t['tile']}.csv"
            if alvo.exists():
                continue
            am = rot.addBands(ee.Image.pixelLonLat()).stratifiedSample(
                numPoints=CFG["pontos_por_classe_no_tile"], classBand="classe_gfd", region=ee.Geometry.Rectangle(t["tile_1grau"]),
                scale=max(PASSO_M, esc), projection=img.select("flooded").projection(), seed=CFG["semente_pontos"],
                geometries=False, tileScale=4)
            url = am.getDownloadURL(filetype="CSV", selectors=["classe_gfd", "latitude", "longitude"])
            with urllib.request.urlopen(url, timeout=900) as r:
                linhas = list(csv.DictReader(io.StringIO(r.read().decode("utf-8", errors="replace"))))
            la, lo = map(int, t["tile"].split("_"))
            pts = [{"ponto_id": f"MT11_{ev['id']}_{t['tile']}_{i:05d}",
                    "classe_obs": "INUNDACAO" if int(float(x["classe_gfd"])) == 2 else "NAO_INUNDADO_OBSERVADO",
                    "lat": float(x["latitude"]), "lon": float(x["longitude"]), "evento_gfd": ev["id"],
                    "grupo_cv": "GFD_" + _rotulo(la, lo), "tile": "gfd_tile_" + _rotulo(la, lo)}
                   for i, x in enumerate(linhas) if x.get("latitude")]
            pd.DataFrame(pts).to_csv(alvo, index=False)
            print(ev["evento_id"], t["tile"], len(pts), flush=True)
    return {"arquivos": len(list(pasta.glob("*.csv")))}


def janelas():
    conferir_registro()
    exigir("selecao.json")
    js = []
    for ev in ler(SAIDA / "selecao.json")["eventos"]:
        for t in ev["tiles"]:
            la, lo = map(int, t["tile"].split("_"))
            js.append({"nome": "gfd_tile_" + _rotulo(la, lo), "tile_1grau": t["tile_1grau"]})
    escrever(SAIDA / "janelas.json", js)
    return {"janelas": len(js), "ja_derivadas": sum((TER / j["nome"] / "dem_filled.tif").exists() for j in js)}


# ------------------------------------------------------------------ atributos

def declive_p50(pasta: Path, passo: int = 4) -> float:
    """Como `mt01_populacao.contexto_das_unidades`: p50 do raster inteiro decimado, sem ponto nem rótulo."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(pasta / "slope_deg_wbt.tif") as s:
        a = s.read(1, out_shape=(max(1, s.height // passo), max(1, s.width // passo)), resampling=Resampling.nearest).astype(float)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
    a[a < -1e4] = np.nan
    v = a[np.isfinite(a)]
    return float(np.percentile(v, 50)) if v.size else np.nan


def _amostrar_tabela(ee, img, pts: pd.DataFrame, escala: float, crs=None) -> pd.DataFrame:
    linhas = []
    for i in range(0, len(pts), 2500):
        parte = pts.iloc[i:i + 2500]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"ponto_id": r.ponto_id})
                                   for r in parte.itertuples()])
        for tentativa in range(4):
            try:
                kw = {"collection": fc, "reducer": ee.Reducer.first(), "scale": escala, "tileScale": 4}
                if crs:
                    kw["crs"] = crs
                r = img.reduceRegions(**kw).getInfo()
                linhas += [f["properties"] for f in r["features"]]
                break
            except Exception as e:  # noqa: BLE001 - Earth Engine em modo restrito: tenta de novo, depois registra
                if tentativa == 3:
                    raise
                print("  EE falhou, nova tentativa:", str(e)[:120], flush=True)
                time.sleep(20 * (tentativa + 1))
    return pd.DataFrame(linhas)


def cobertura(ee, d: pd.DataFrame, ano: int) -> pd.DataFrame:
    import cob01_cobertura_e_solo as cob
    t = _amostrar_tabela(ee, cob.imagem(ee, ano), d[["ponto_id", "lat", "lon"]], 30)
    return t.reindex(columns=["ponto_id", "construido_frac", "areia_pct_0_30", "worldcover_2021"])


def _pilha_s1(ee, tile, inicio: str, crs: str):
    """Datas pré-evento da órbita com mais datas no tile; VV multilook a 30 m na grade do terreno."""
    t0 = dt.date.fromisoformat(inicio)
    reg = ee.Geometry.Rectangle(tile)
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")).filterBounds(reg)
           .filterDate(str(t0 - dt.timedelta(days=CFG["janela_pre_dias"][0])), str(t0 - dt.timedelta(days=CFG["janela_pre_dias"][1]))))
    meta = col.aggregate_array("relativeOrbitNumber_start").zip(col.aggregate_array("system:time_start")).getInfo()
    por_orbita = {}
    for orb, ts in meta:
        por_orbita.setdefault(int(orb), set()).add(dt.datetime.fromtimestamp(ts / 1000, dt.UTC).date().isoformat())
    if not por_orbita:
        return None, [], None
    orb = max(por_orbita, key=lambda o: (len(por_orbita[o]), -o))
    datas = sorted(por_orbita[orb])[-CFG["max_datas"]:]
    bandas = []
    for k, d in enumerate(datas):
        dia = (col.filter(ee.Filter.eq("relativeOrbitNumber_start", orb))
               .filterDate(d, str(dt.date.fromisoformat(d) + dt.timedelta(days=1))).select("VV").mosaic())
        lin = ee.Image(10).pow(dia.divide(10))
        ml = (lin.setDefaultProjection(crs=crs, scale=10).reduceResolution(ee.Reducer.mean(), maxPixels=64)
              .reproject(crs=crs, scale=30).log10().multiply(10))
        bandas.append(ml.rename(f"VV_{k:02d}"))
    return ee.Image.cat(bandas), datas, orb


def frequencia(V: np.ndarray, datas: list[str], lat: float) -> dict:
    """V [pontos, datas] em dB (NaN sem leitura). Admissão por data nos pontos; detecção v1 e E1; F por ponto."""
    ok = []
    for j, d in enumerate(datas):
        col = V[:, j]
        valido = np.isfinite(col)
        mes = int(d[5:7])
        neve = abs(lat) > CFG["lat_neve"] and mes in ((12, 1, 2, 3) if lat > 0 else (6, 7, 8, 9))
        med = np.nanmedian(col) if valido.any() else np.nan
        ok.append(valido.mean() >= CFG["cobertura_minima_data"] and np.isfinite(med)
                  and CFG["vv_mediana_admissivel"][0] <= med <= CFG["vv_mediana_admissivel"][1] and not neve)
    ok = np.array(ok, bool)
    A = V[:, ok] if ok.any() else np.full((len(V), 0), np.nan)
    n = np.isfinite(A).sum(1)
    with np.errstate(invalid="ignore"):
        v1 = np.where(np.isfinite(A), A <= CFG["limiar_vv_db"], np.nan)
        p75 = np.nanpercentile(A, CFG["e1_quantil"], axis=1) if A.shape[1] else np.full(len(A), np.nan)
        e1 = np.where(np.isfinite(A), (A <= CFG["limiar_vv_db"]) & (A <= p75[:, None] - CFG["e1_delta_db"]), np.nan)
        f_v1 = np.where(n > 0, np.nansum(v1, 1) / np.maximum(n, 1), np.nan)
        f_e1 = np.where(n > 0, np.nansum(e1, 1) / np.maximum(n, 1), np.nan)
    return {"F_v1": f_v1, "F_e1": f_e1, "n_pre": n, "datas_admitidas": int(ok.sum())}


def qualidade(F: np.ndarray, jrc: np.ndarray, n: np.ndarray) -> dict:
    ok = n >= CFG["min_datas_pre"]
    terra, sazonal = ok & (jrc == 0), ok & (jrc >= 10) & (jrc < 50)
    t = float(np.nanmean(F[terra])) if terra.sum() else np.nan
    s = float(np.nanmean(F[sazonal])) if sazonal.sum() else np.nan
    passa = bool(np.isfinite(t) and np.isfinite(s) and t <= CFG["qualidade_terra_max"] and s - t >= CFG["qualidade_contraste_min"])
    return {"terra": t, "sazonal": s, "n_terra": int(terra.sum()), "n_sazonal": int(sazonal.sum()), "passa": passa}


def s1_no_tile(ee, ev, d: pd.DataFrame, crs: str) -> tuple[pd.DataFrame, dict]:
    tile = ev["tiles"][0]["tile_1grau"]
    img, datas, orb = _pilha_s1(ee, tile, ev["inicio"], crs)
    lat = (tile[1] + tile[3]) / 2
    if img is None:
        return d.assign(F_v1=np.nan, F_e1=np.nan, n_pre=0), {"orbita": None, "datas": 0}
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).rename("jrc_occ")
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    estat = ee.Image.cat([wc.eq(10).toFloat().reduceResolution(ee.Reducer.mean(), maxPixels=64).rename("arvore"),
                          wc.eq(50).toFloat().reduceResolution(ee.Reducer.mean(), maxPixels=64).rename("construido_wc"),
                          jrc]).reproject(crs=crs, scale=30)
    tudo = img.addBands(estat)
    nomes = [f"VV_{k:02d}" for k in range(len(datas))]
    t = _amostrar_tabela(ee, tudo, d[["ponto_id", "lat", "lon"]], 30, crs).set_index("ponto_id").reindex(d.ponto_id)
    V = t.reindex(columns=nomes).to_numpy(float)
    fr = frequencia(V, datas, lat)
    # qualidade em pixels aleatórios do tile: nenhum depende do rótulo
    aleat = ee.FeatureCollection.randomPoints(ee.Geometry.Rectangle(tile), CFG["pontos_qualidade"], CFG["semente_pontos"])
    pts_q = pd.DataFrame([{"ponto_id": f"q{i}", "lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1]}
                          for i, f in enumerate(aleat.getInfo()["features"])])
    tq = _amostrar_tabela(ee, tudo, pts_q, 30, crs).set_index("ponto_id").reindex(pts_q.ponto_id)
    fq = frequencia(tq.reindex(columns=nomes).to_numpy(float), datas, lat)
    obs_q = (tq.arvore.to_numpy(float) <= CFG["arvore_max"]) & (tq.construido_wc.to_numpy(float) <= CFG["construido_max"])
    jq = np.where(obs_q, tq.jrc_occ.to_numpy(float), np.nan)
    qual = {r: qualidade(fq[f"F_{r}"], jq, fq["n_pre"]) for r in ("v1", "e1")}
    out = d.assign(F_v1=fr["F_v1"], F_e1=fr["F_e1"], n_pre=fr["n_pre"], arvore=t.arvore.to_numpy(float),
                   construido_wc=t.construido_wc.to_numpy(float), jrc_occ_s1=t.jrc_occ.to_numpy(float))
    return out, {"orbita": orb, "datas": len(datas), "datas_admitidas": fr["datas_admitidas"], "qualidade": qual,
                 "regra": "e1" if qual["e1"]["passa"] else ("v1" if qual["v1"]["passa"] else None)}


def atributos():
    conferir_registro()
    exigir("selecao.json")
    import ee
    import gfdq02_derivar_quarentena as gq
    from gfd02_amostrar_terreno import _amostrar_no_tile
    from mt02_atributos import distancias_da_unidade
    ee.Initialize()
    pasta = SAIDA / "atributos"
    pasta.mkdir(parents=True, exist_ok=True)
    for ev in ler(SAIDA / "selecao.json")["eventos"]:
        alvo = pasta / f"{ev['evento_id'].replace(':', '_')}.parquet"
        if alvo.exists():
            continue
        t = ev["tiles"][0]
        p = pd.read_csv(SAIDA / "pontos" / f"pontos_gfd{ev['id']}_{t['tile']}.csv")
        tile = p.tile.iloc[0]
        if not (TER / tile / "dem_filled.tif").exists():
            print(ev["evento_id"], "sem terreno", flush=True)
            continue
        s = _amostrar_no_tile(p, TER / tile)
        bruto = p.reset_index(drop=True).assign(**{k: v for k, v in s.items()})
        d, saiu = gq.espelho_do_ds06(bruto)
        d["ln_sca"] = d.twi_dinf + np.log(np.tan(np.radians(np.clip(d.slope_deg, .01, None))))
        d1, d2, lado = distancias_da_unidade(TER / tile, d.lon.to_numpy(), d.lat.to_numpy())
        d["dist_drenagem_m"], d["dist_canal10_m"], d["lado_unidade_km"] = d1, d2, lado
        d["declive_p50"] = declive_p50(TER / tile)
        ano = int(ev["inicio"][:4])
        c = cobertura(ee, d, ano)
        d = d.merge(c, on="ponto_id", how="left")
        d["construido_v2"], d["areia_v2"] = d.construido_frac, d.areia_pct_0_30
        crs = f"EPSG:{int(d.crs_metrico.iloc[0])}"
        d, info_s1 = s1_no_tile(ee, ev, d, crs)
        d["unidade"], d["evento_id"] = ev["unidade"], ev["evento_id"]
        d.to_parquet(alvo, index=False)
        escrever(alvo.with_suffix(".json"), {"evento_id": ev["evento_id"], "linhas": len(d), "saiu_no_espelho": saiu,
                                             "s1": info_s1, "ano_cobertura": ano, "sha256": sha256_arquivo(alvo)})
        print(ev["evento_id"], len(d), info_s1.get("regra"), info_s1.get("datas_admitidas"), flush=True)
    return {"eventos_com_atributos": len(list(pasta.glob("*.parquet")))}


def fumaca():
    """Fases de atributos num evento JÁ EXAMINADO (quarentena gfdq, GFD:3452), sem escore nem leitura de AUC."""
    import ee
    import gfdq02_derivar_quarentena as gq
    from gfd02_amostrar_terreno import _amostrar_no_tile
    from mt02_atributos import distancias_da_unidade
    ee.Initialize()
    ev = next(e for e in ler(QUARENTENA)["eventos"] if e["id"] == 3452)
    t = ev["tiles"][0]
    p = pd.read_csv(REPO / "local_runs/gfd-quarentena/pontos" / f"pontos_gfd3452_{t['tile']}.csv").sample(300, random_state=1)
    tile = p.tile.iloc[0]
    s = _amostrar_no_tile(p, TER / tile)
    d, _ = gq.espelho_do_ds06(p.reset_index(drop=True).assign(**{k: v for k, v in s.items()}))
    d["ln_sca"] = d.twi_dinf + np.log(np.tan(np.radians(np.clip(d.slope_deg, .01, None))))
    d["dist_drenagem_m"], d["dist_canal10_m"], _ = distancias_da_unidade(TER / tile, d.lon.to_numpy(), d.lat.to_numpy())
    d["declive_p50"] = declive_p50(TER / tile)
    d = d.merge(cobertura(ee, d, 2017), on="ponto_id", how="left")
    d["construido_v2"], d["areia_v2"] = d.construido_frac, d.areia_pct_0_30
    evq = {"tiles": [t], "inicio": "2017-04-15"}
    d, info = s1_no_tile(ee, evq, d, f"EPSG:{int(d.crs_metrico.iloc[0])}")
    x = mt.caracteristicas(d)
    return {"linhas": len(d), "colunas_nulas": {c: float(x[c].isna().mean()) for c in x},
            "F_v1_preenchido": float(np.isfinite(d.F_v1).mean()), "n_pre_mediana": float(np.median(d.n_pre)),
            "s1": {k: v for k, v in info.items() if k != "qualidade"}, "qualidade": info.get("qualidade"),
            "nenhum_escore_calculado": True}


# ------------------------------------------------------------------ avaliar

def escores(d: pd.DataFrame, f_exp: dict) -> pd.DataFrame:
    with Q_PKL.open("rb") as f:
        modelos = pickle.load(f)
    x = mt.caracteristicas(d)
    out = d.copy()
    out["A0_regra_hand"] = -d.hand_m.to_numpy()
    for nome in ("Q_base", "Q_conexao"):
        out[nome] = modelos[nome].escore(x)
    regra = d.regra_unidade.to_numpy()
    out["F"] = np.where(regra == "e1", d.F_e1, np.where(regra == "v1", d.F_v1, np.nan))
    out["F_transf"] = escore_exportado(pd.DataFrame({"Q": out.Q_conexao, "F": out.F}), f_exp)
    return out


def contraste(dif: pd.Series) -> dict:
    d = dif.dropna()
    r = {"n": len(d), "media": float(d.mean()) if len(d) else None, "a_favor": int((d > 0).sum()), "por_unidade": d.to_dict()}
    if len(d) < CFG["minimo_unidades"]:
        return {**r, "veredito": "not_testable"}
    boot = np.random.default_rng(CFG["semente_bootstrap"]).choice(d.to_numpy(), (CFG["bootstrap"], len(d))).mean(1)
    q = (1 - CFG["nivel"]) / 2
    lo, hi = np.quantile(boot, [q, 1 - q])
    v = ("sustentado" if r["media"] >= CFG["margem_auc"] and lo > 0 else "inferior" if hi < 0 else
         "ganho_excluido" if hi < CFG["margem_auc"] else "indecidivel")
    return {**r, "ic": [float(lo), float(hi)], "nivel_ic": CFG["nivel"], "veredito": v}


def medir(s: pd.DataFrame, modelos: list[str]) -> pd.DataFrame:
    rows = []
    for (un, ev), g in s.groupby(["unidade", "evento_id"]):
        if len(g) < CFG["minimo_pontos_evento"] or g.classe.nunique() < 2:
            continue
        for m in modelos:
            rows.append({"unidade": un, "evento_id": ev, "modelo": m, "n": len(g), "auc": mt.e1.auc(g.classe.to_numpy(), g[m].to_numpy())})
    return pd.DataFrame(rows)


def avaliar():
    p = conferir_registro()
    exigir("selecao.json")
    if (SAIDA / "resultado.json").exists():
        return ler(SAIDA / "resultado.json")
    sel = ler(SAIDA / "selecao.json")
    f_exp = ler(SAIDA / "f_transf.json")["exportado"]
    partes, info = [], {}
    for ev in sel["eventos"]:
        arq = SAIDA / "atributos" / f"{ev['evento_id'].replace(':', '_')}.parquet"
        if not arq.exists():
            info[ev["evento_id"]] = "sem_atributos"
            continue
        meta = ler(arq.with_suffix(".json"))
        d = pd.read_parquet(arq)
        d["regra_unidade"] = meta["s1"].get("regra")
        partes.append(d)
    d = pd.concat(partes, ignore_index=True)
    s = escores(d, f_exp)
    s.to_parquet(SAIDA / "escores.parquet", index=False)
    mq = medir(s, ["A0_regra_hand", "Q_base", "Q_conexao"])
    obs = ((s.arvore <= CFG["arvore_max"]) & (s.construido_wc <= CFG["construido_max"]) & (s.slope_deg <= CFG["declive_max_graus"])
           & (s.jrc_occ_s1 < CFG["jrc_permanente"]) & (s.n_pre >= CFG["min_datas_pre"]) & s.F.notna())
    mf = medir(s[obs], ["Q_conexao", "F_transf"])
    uq = mq.groupby(["unidade", "modelo"]).auc.mean().unstack()
    uf = mf.groupby(["unidade", "modelo"]).auc.mean().unstack() if len(mf) else pd.DataFrame(columns=["Q_conexao", "F_transf"])
    pr = {"HQ1": contraste(uq.Q_conexao - uq.A0_regra_hand), "HQ2": contraste(uq.Q_conexao - uq.Q_base),
          "HF": contraste(uf.F_transf - uf.Q_conexao) if len(uf) else {"n": 0, "veredito": "not_testable"}}
    pd.concat([mq.assign(populacao="Q"), mf.assign(populacao="HF")]).to_parquet(SAIDA / "medidas.parquet", index=False)
    avaliados = sorted(set(mq.evento_id) | set(mf.evento_id))
    r = {"tipo": "MT11_resultado", "quando_utc": agora_utc(), "protocolo_sha256": sha256_arquivo(SAIDA / "protocolo.json"),
         "primarias": pr, "auc_media_unidades": {"Q": uq.mean().to_dict(), "HF": uf.mean().to_dict() if len(uf) else {}},
         "unidades_Q": len(uq), "unidades_HF": len(uf), "eventos_Q": int(mq.evento_id.nunique()), "eventos_HF": int(mf.evento_id.nunique()),
         "pontos": len(s), "pontos_HF": int(obs.sum()), "eventos_sem_atributos": info, "eventos_avaliados": avaliados,
         "regime": p["regime"], "modelo_promovido": False}
    escrever(SAIDA / "resultado.json", r)
    with EXAMINADOS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"quando_utc": r["quando_utc"], "fonte": "gfd", "avaliacao": "mt11_confirmatorio_transdutivo",
                            "eventos": avaliados}, ensure_ascii=False) + "\n")
    return {k: r[k] for k in ("unidades_Q", "unidades_HF", "eventos_Q", "eventos_HF")} | {
        "primarias": {k: {q: v.get(q) for q in ("n", "media", "ic", "veredito")} for k, v in pr.items()}}


if __name__ == "__main__":
    fases = {"fumaca": fumaca, "congelar": congelar, "registrar": registrar, "selecionar": selecionar, "pontos": pontos,
             "janelas": janelas, "atributos": atributos, "avaliar": avaliar}
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fase", choices=list(fases))
    a = ap.parse_args()
    print(json.dumps(fases[a.fase](), ensure_ascii=False, default=str))
