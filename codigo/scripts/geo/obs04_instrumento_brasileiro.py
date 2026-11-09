"""Instrumento observacional brasileiro: validar uma regra Sentinel-1 sem HAND contra referencia de analista.

-------------------------------------------------------------------------
POR QUE ESTE PASSO VEM ANTES DE QUALQUER CONFIRMACAO
-------------------------------------------------------------------------

O bloqueio de 10/10/2026 nao e falta de evento: e falta de um INSTRUMENTO que
observe agua de cheia no Brasil sem usar os descritores fisicos que a
confirmacao quer testar. O GFM usa HAND na mascara de exclusao (circular), a
replica MODIS foi reprovada e a mancha UFRGS 2024 interpola a cota sobre o
FABDEM (o limite da mancha vem do relevo). Sem instrumento validado, nenhum
evento novo vira unidade de teste.

-------------------------------------------------------------------------
O QUE E VALIDADO AQUI (desenvolvimento do instrumento, nao do modelo)
-------------------------------------------------------------------------

  tarefa       agua aberta de cheia NO INSTANTE da cena S1 posterior, por pixel
               de 10 m; nao "area inundada no pico" nem "nunca inundou"
  instrumento  regra fixa de mudanca em VV (a da serra03 v1, gravada em
               03/10/2026, antes deste passo): agua se VV_pos <= -15 dB e
               queda >= 3 dB; seco se VV_pos > -15 dB e queda < 1,5 dB;
               o resto e desconhecido. Nao usa HAND, DEM, declive ou JRC.
  referencia   rotulo GEOID-Flood (CC-BY-4.0) do EMSR720: delineacao CEMS de
               analista, revalidada manualmente pelos autores; 0 fundo,
               1 agua permanente, 2 inundada, 255 desconhecido. Independente
               do MODELO avaliado no REV-P; independencia de HAND NAO e
               verificada (provedores CEMS variam) -- por isso os erros saem
               por faixa de HAND, para expor esse vies se existir.
  papel        EMSR720 e evento conhecido do REV-P: aqui serve so para
               desenvolver/validar o instrumento. Nao e reserva, nao avalia o
               modelo de terreno, nao e evento novo.
  reserva      nao existe hoje reserva brasileira com referencia humana para o
               instrumento; o mesmo episodio (EMSN194) nao conta como reserva.

Criterios fixados ANTES de baixar ou ler qualquer pixel (pre_registro.json).

Uso:
    python scripts/geo/obs04_instrumento_brasileiro.py --pre-registrar
    python scripts/geo/obs04_instrumento_brasileiro.py --adquirir
    python scripts/geo/obs04_instrumento_brasileiro.py --avaliar
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in ("geo", "sar", "metodo", "servico", "suscetibilidade", "terreno"):
    sys.path.insert(0, str(REPO / "scripts" / _p))

RUNS = REPO / "local_runs"
SAIDA = RUNS / "instrumento-brasileiro" / "v1"
REVISAO = "868407460bf3db492f50730a57585916baa71dc6"
CATALOGO = RUNS / "geoid-catalogo-publico" / REVISAO / "geoid-flood" / "tile_catalog.parquet"
SERRA03 = RUNS / "serra-03-rotulo-sensor-duplo" / "v1" / "manifesto.json"
EVENTO = "EMSR720"
CAMADAS = ("s1grd", "label", "validity")
TOLERANCIA_PRIMARIA_H = 24
PASSO_CONTEXTO = 5          # 1 pixel de 10 m em cada 5x5 recebe contexto de terreno (50 m)
CRITERIOS = {
    "C1_precisao_agua": "precisao da classe agua_de_cheia contra referencia 2 (vs 0) >= 0,80 no agregado primario",
    "C2_revocacao_agua": "revocacao da referencia 2 entre pixels com referencia 0/2 e S1 valido >= 0,40",
    "C3_npv_seco": "entre pixels que o instrumento chama de seco, >= 0,95 sao referencia 0",
    "C4_cobertura": ">= 0,30 dos pixels com referencia 0/2 recebem agua ou seco (nao desconhecido)",
    "C5_incerteza": "limite inferior de 90% do bootstrap por tile da precisao >= 0,70",
    "decisao": ("C1-C5 passam -> instrumento ADMITIDO PARA DESENVOLVIMENTO no Brasil (nao confirmatorio); "
                "qualquer falha -> nao admitido, erros por contexto registrados; nenhum limiar e reajustado aqui"),
    "primario": f"tiles com |S1 posterior - delineacao posterior| <= {TOLERANCIA_PRIMARIA_H} h",
    "secundario": "todos os tiles, com a defasagem declarada; descritivo",
    "contextos": "HAND (0-5, 5-15, >15 m), declive (<2, 2-5, >5 graus), VV anterior (>-8 dB = alvo brilhante antes)",
}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def _sha_json(o) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _agora() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def regra() -> dict:
    """Limiares da serra03 v1, lidos do manifesto gravado em 03/10/2026; nada aqui os escolhe."""
    r = json.loads(SERRA03.read_text(encoding="utf-8"))["regra"]
    return {"vv_agua_db": r["vv_agua_db"], "queda_agua_db": r["queda_agua_db"], "queda_seco_db": r["queda_seco_db"]}


def classificar(vv_pre_db: np.ndarray, vv_pos_db: np.ndarray, rg: dict) -> np.ndarray:
    """0 seco, 1 agua de cheia, 255 desconhecido. Sem terreno, sem JRC."""
    queda = vv_pre_db - vv_pos_db
    ok = np.isfinite(vv_pre_db) & np.isfinite(vv_pos_db)
    agua = ok & (vv_pos_db <= rg["vv_agua_db"]) & (queda >= rg["queda_agua_db"])
    seco = ok & (vv_pos_db > rg["vv_agua_db"]) & (queda < rg["queda_seco_db"])
    out = np.full(vv_pos_db.shape, 255, np.uint8)
    out[seco] = 0
    out[agua] = 1
    return out


def pre_registrar() -> dict:
    SAIDA.mkdir(parents=True, exist_ok=True)
    arq = SAIDA / "pre_registro.json"
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    p = {"evento": EVENTO, "fonte": f"GEOID-Flood revisao {REVISAO} (CC-BY-4.0; Copernicus S1 e CEMS modificados)",
         "selecao": "todos os tiles do EMSR720 no catalogo; nenhuma escolha por qualidade, rotulo ou proporcao",
         "regra_instrumento": regra(), "regra_origem": {"arquivo": str(SERRA03.relative_to(REPO)), "sha256": _sha(SERRA03)},
         "criterios": CRITERIOS, "tolerancia_primaria_h": TOLERANCIA_PRIMARIA_H,
         "papel": "desenvolvimento do instrumento em evento conhecido; nao reserva, nao avaliacao do modelo de terreno",
         "nao_le": ["reservas EMSR868/871", "heldout GEOID", "rotulos antes deste pre-registro"],
         "codigo_sha256": _sha(Path(__file__))}
    p["hash"] = _sha_json(p)
    p["quando_utc"] = _agora()
    arq.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def _indice() -> dict:
    import obs01_aquisicao_dirigida as o1
    base = f"https://huggingface.co/datasets/links-ads/geoid-flood/resolve/{REVISAO}/"
    reg = o1.baixar(base + "shard_index.json.gz", SAIDA / "dados" / "shard_index.json.gz", 4 * 1024 ** 2)
    with gzip.open(reg["arquivo"], "rt", encoding="utf-8") as f:
        return json.loads(f.read(64 * 1024 ** 2))


def adquirir(teto_bytes: int = 256 * 1024 ** 2) -> dict:
    """Baixa so os membros do EMSR720 (S1 GRD anterior/posterior, rotulo, validade) por intervalo HTTP."""
    import pyarrow.parquet as pq
    import sar_obs01_pares_datados as so
    pre = pre_registrar()
    cat = pq.read_table(CATALOGO, columns=["event_id", "tile_id", "bbox_id", "utm_crs", "delineation_time_post",
                                           "delineation_time_pre", "event_time", "split"]).to_pandas()
    cat = cat[cat.event_id == EVENTO]
    chaves = {f"{r.event_id}-{r.tile_id}-{r.bbox_id}": r._asdict() for r in cat.itertuples(index=False)}
    indice = _indice()
    pedidos: dict[str, set] = {}
    por_chave = {k: {} for k in chaves}
    for s in indice["shards"]:
        camada = s["path"].split("/")[-2]
        if camada not in CAMADAS or "/train/" not in s["path"]:
            continue
        for nome in s["members"]:
            base = PurePosixPath(nome).name
            for k in chaves:
                if base.startswith(k + "_"):
                    pedidos.setdefault(s["path"], set()).add(nome)
                    por_chave[k].setdefault(camada, []).append(nome)
    shards = {s["path"]: s for s in indice["shards"]}
    destino = SAIDA / "dados"
    orc, arquivos, falhas = {"solicitados": 0}, [], []
    for caminho, nomes in sorted(pedidos.items()):
        try:
            r, b, n = so.extrair_selecionados(shards[caminho], nomes, destino, teto_bytes, orc)
            arquivos.extend(r)
        except Exception as e:  # registrar e seguir: falha de um shard nao vira ausencia de dado
            falhas.append({"shard": caminho, "erro": str(e)[:300]})
    out = {"tipo": "aquisicao_instrumento_brasileiro", "pre_registro": pre["hash"], "evento": EVENTO,
           "tiles_no_catalogo": len(chaves), "membros_por_tile": {k: {c: len(v) for c, v in d.items()}
                                                                   for k, d in por_chave.items()},
           "arquivos": arquivos, "falhas": falhas, "bytes_solicitados": orc["solicitados"],
           "licenca": "CC-BY-4.0; contem dados Copernicus Sentinel-1 modificados e produtos CEMS (c) Uniao Europeia",
           "reservas_consumidas": False, "quando_utc": _agora()}
    (SAIDA / "aquisicao.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


DATA = re.compile(r"_s1grd_(pre|post)_(\d{8}T\d{6})\.tif$")


def _tiles() -> dict:
    """Agrupa os arquivos baixados por tile: pre, post, label, validity."""
    out: dict[str, dict] = {}
    for arq in (SAIDA / "dados").rglob("*.tif"):
        chave = arq.name.split("_")[0]
        m = DATA.search(arq.name)
        if m:
            out.setdefault(chave, {})[m.group(1)] = arq
            out[chave][m.group(1) + "_data"] = pd.Timestamp(m.group(2))
        elif "_label" in arq.name:
            out.setdefault(chave, {})["label"] = arq
        elif "_validity" in arq.name:
            out.setdefault(chave, {})["validity"] = arq
    return out


def _db(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(x) & (x > 0), 10 * np.log10(x), np.nan)


def _metricas(ref: np.ndarray, ins: np.ndarray) -> dict:
    """ref em {0, 2} e ins em {0, 1, 255}, so pixels avaliaveis."""
    agua_ins, seco_ins = ins == 1, ins == 0
    ref2 = ref == 2
    tp, fp = int((agua_ins & ref2).sum()), int((agua_ins & ~ref2).sum())
    fn_total = int((ref2 & ~agua_ins).sum())
    seco_ok, seco_err = int((seco_ins & ~ref2).sum()), int((seco_ins & ref2).sum())
    n = len(ref)
    return {"pixels": n, "ref_inundada": int(ref2.sum()), "agua_ins": tp + fp, "tp": tp, "fp": fp,
            "precisao": tp / (tp + fp) if tp + fp else None,
            "revocacao": tp / (tp + fn_total) if tp + fn_total else None,
            "npv_seco": seco_ok / (seco_ok + seco_err) if seco_ok + seco_err else None,
            "cobertura": (tp + fp + seco_ok + seco_err) / n if n else None,
            "seco_ins": seco_ok + seco_err}


def _contexto(lon, lat):
    import acervo_espacial as ae
    ac = ae.Acervo(hidrologia=False, dominio=False)
    pts = pd.DataFrame({"ponto_id": np.arange(len(lon)), "lon": lon, "lat": lat})
    c = ae.contextualizar(pts, ac, vizinhanca=False, em_memoria=True)
    return c["hand_m"].to_numpy(float), c["slope_deg"].to_numpy(float)


def avaliar() -> dict:
    import rasterio
    from pyproj import Transformer
    pre = pre_registrar()
    rg = pre["regra_instrumento"]
    import pyarrow.parquet as pq
    cat = pq.read_table(CATALOGO, columns=["event_id", "tile_id", "bbox_id", "delineation_time_post"]).to_pandas()
    cat = cat[cat.event_id == EVENTO]
    delin = {f"{r.event_id}-{r.tile_id}-{r.bbox_id}": pd.Timestamp(r.delineation_time_post) for r in cat.itertuples()}
    linhas, por_tile = [], {}
    for chave, t in sorted(_tiles().items()):
        if not all(k in t for k in ("pre", "post", "label", "validity")):
            por_tile[chave] = {"estado": "incompleto", "tem": sorted(k for k in t if not k.endswith("_data"))}
            continue
        with rasterio.open(t["pre"]) as a, rasterio.open(t["post"]) as b, rasterio.open(t["label"]) as l, \
                rasterio.open(t["validity"]) as v:
            if not (a.transform == b.transform == l.transform == v.transform and a.crs == b.crs == l.crs):
                por_tile[chave] = {"estado": "grades_divergentes"}
                continue
            vv_pre, vv_pos = _db(a.read(1).astype(float)), _db(b.read(1).astype(float))
            ref, val = l.read(1), v.read(1)
            tr, crs = a.transform, a.crs
        ins = classificar(vv_pre, vv_pos, rg)
        gap_h = abs((t["post_data"] - delin.get(chave, pd.NaT)).total_seconds()) / 3600
        avaliavel = (val > 0) & np.isin(ref, (0, 2)) & np.isfinite(vv_pre) & np.isfinite(vv_pos)
        ii, jj = np.nonzero(avaliavel)
        d = pd.DataFrame({"tile": chave, "ref": ref[ii, jj], "ins": ins[ii, jj], "vv_pre": vv_pre[ii, jj],
                          "sub": ((ii % PASSO_CONTEXTO) == 0) & ((jj % PASSO_CONTEXTO) == 0), "i": ii, "j": jj})
        d["primario"] = gap_h <= TOLERANCIA_PRIMARIA_H
        x, y = tr * (d.j.to_numpy() + 0.5, d.i.to_numpy() + 0.5)
        d["x"], d["y"] = x, y
        linhas.append(d)
        por_tile[chave] = {"estado": "avaliado", "s1_pre": str(t["pre_data"]), "s1_pos": str(t["post_data"]),
                           "delineacao_pos": str(delin.get(chave)), "defasagem_h": round(gap_h, 2),
                           "primario": bool(gap_h <= TOLERANCIA_PRIMARIA_H), "crs": str(crs),
                           "pixels_avaliaveis": int(avaliavel.sum()),
                           "ref_1_agua_permanente_excluida": int((ref == 1).sum()),
                           "desconhecido_na_referencia": int(((ref == 255) | (val == 0)).sum()),
                           **_metricas(ref[avaliavel], ins[avaliavel])}
        print(chave, {k: por_tile[chave][k] for k in ("defasagem_h", "precisao", "revocacao", "npv_seco", "cobertura")},
              flush=True)
    d = pd.concat(linhas, ignore_index=True)
    # contexto de terreno so na subamostra; o terreno descreve o erro, nao fabrica rotulo
    s = d[d["sub"]].copy()
    lon = np.empty(len(s))
    lat = np.empty(len(s))
    for crs_, g in s.groupby(s.tile.map(lambda k: por_tile[k]["crs"])):
        lo, la = Transformer.from_crs(crs_, "EPSG:4326", always_xy=True).transform(g.x.to_numpy(), g.y.to_numpy())
        lon[s.index.get_indexer(g.index)], lat[s.index.get_indexer(g.index)] = lo, la
    hand, decl = _contexto(lon, lat)
    s["hand"], s["declive"] = hand, decl
    s["faixa_hand"] = pd.cut(s.hand, [-1e9, 5, 15, 1e9], labels=["0-5m", "5-15m", ">15m"])
    s["faixa_declive"] = pd.cut(s.declive, [-1e9, 2, 5, 1e9], labels=["<2", "2-5", ">5"])
    s["pre_brilhante"] = np.where(s.vv_pre > -8, "vv_pre>-8dB", "vv_pre<=-8dB")
    res = {"pre_registro": pre["hash"], "regra": rg, "por_tile": por_tile}
    for nome, sel in (("primario", d.primario), ("secundario_todos", np.ones(len(d), bool))):
        dd = d[sel]
        if dd.empty:
            res[nome] = {"estado": "not_testable", "motivo": "nenhum tile dentro da tolerancia temporal"}
            continue
        m = _metricas(dd.ref.to_numpy(), dd.ins.to_numpy())
        tiles = dd.tile.unique()
        rng = np.random.default_rng(20261010)
        boot = []
        grupos = {k: g for k, g in dd.groupby("tile")}
        for _ in range(2000):
            amostra = rng.choice(tiles, len(tiles), replace=True)
            tp = sum(int(((grupos[k].ins == 1) & (grupos[k].ref == 2)).sum()) for k in amostra)
            fp = sum(int(((grupos[k].ins == 1) & (grupos[k].ref == 0)).sum()) for k in amostra)
            if tp + fp:
                boot.append(tp / (tp + fp))
        m["precisao_ic90_por_tile"] = [float(np.quantile(boot, .05)), float(np.quantile(boot, .95))] if boot else None
        m["tiles"] = int(len(tiles))
        m["criterios"] = {
            "C1": m["precisao"] is not None and m["precisao"] >= 0.80,
            "C2": m["revocacao"] is not None and m["revocacao"] >= 0.40,
            "C3": m["npv_seco"] is not None and m["npv_seco"] >= 0.95,
            "C4": m["cobertura"] is not None and m["cobertura"] >= 0.30,
            "C5": m["precisao_ic90_por_tile"] is not None and m["precisao_ic90_por_tile"][0] >= 0.70}
        m["admitido_para_desenvolvimento"] = bool(all(m["criterios"].values())) if nome == "primario" else None
        ss = s[sel[s.index]]
        m["por_contexto"] = {c: {str(k): _metricas(g.ref.to_numpy(), g.ins.to_numpy()) for k, g in ss.groupby(c, observed=True)}
                             for c in ("faixa_hand", "faixa_declive", "pre_brilhante")}
        res[nome] = m
    res["nao_e"] = ("validacao de instrumento num evento conhecido (EMSR720), dentro de um episodio; nao e reserva, "
                    "nao confirma terreno e nao transfere a outro episodio sem nova referencia")
    res["quando_utc"] = _agora()
    res["hash_resumo"] = _sha_json({k: v for k, v in res.items() if k != "quando_utc"})
    (SAIDA / "resultado.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------
# v2 (10/10/2026, depois da v1 reprovada): instrumento OPTICO por granulo, OPERA DSWx-HLS.
#
# A v1 nao foi reajustada. A causa medida da reprovacao e estrutural: entre 01/2023 e
# 07/2024 o S1 passou 6 vezes sobre esses vales (orbita 24 D e uma aquisicao de emergencia
# na 163 A); nao existe par da mesma orbita a menos de 90 dias do evento, e o "antes" do
# GEOID e de set/2023, durante outra cheia. Trocar de LIMIAR na v1 seria escolher pelo
# resultado; trocar de INSTRUMENTO, com os mesmos criterios e pre-registro novo, e o passo
# seguinte declarado do desenvolvimento -- e o resultado vale so como desenvolvimento.

SAIDA_V2 = RUNS / "instrumento-brasileiro" / "v2-dswx-hls"
COLECAO_DSWX = "OPERA/DSWX/L3_V1/HLS"
WTR = {"agua": [1], "seco": [0], "desconhecido": [2, 252, 253, 254, 255]}


def pre_registrar_v2() -> dict:
    SAIDA_V2.mkdir(parents=True, exist_ok=True)
    arq = SAIDA_V2 / "pre_registro.json"
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    v1 = SAIDA / "resultado.json"
    p = {"evento": EVENTO, "versao": "v2-dswx-hls",
         "instrumento": (f"{COLECAO_DSWX} (GEE), camada WTR por granulo; agua = 1 (agua aberta); seco = 0; "
                         "agua parcial (2), neve, nuvem/sombra de nuvem, oceano e fill = desconhecido"),
         "regra_wtr": WTR, "usa_terreno": ("so a mascara de sombra de terreno do proprio produto (DEM Copernicus) "
                                           "para tornar desconhecido; sem HAND, sem declive, sem rotulo"),
         "granulo": "o de data mais proxima da delineacao posterior do tile; mesmo dia -> mosaico dos granulos do dia",
         "grade": "a do tile GEOID (UTM, 10 m); DSWx 30 m por vizinho mais proximo",
         "referencia": "a mesma da v1 (rotulo GEOID-Flood do EMSR720)",
         "criterios": CRITERIOS, "tolerancia_primaria_h": TOLERANCIA_PRIMARIA_H,
         "ressalva": ("a delineacao CEMS provavelmente foi desenhada sobre a mesma cena optica: mede concordancia "
                      "algoritmo x interprete na mesma imagem, nao o erro total do mapeamento"),
         "por_que_v2": "v1 reprovada; sem par S1 da mesma orbita a menos de 90 dias (6 cenas em 18 meses)",
         "v1_resultado": {"arquivo": str(v1.relative_to(REPO)), "sha256": _sha(v1)} if v1.exists() else None,
         "papel": "desenvolvimento do instrumento em evento conhecido; nao reserva, nao avalia o modelo de terreno",
         "codigo_sha256": _sha(Path(__file__))}
    p["hash"] = _sha_json(p)
    p["quando_utc"] = _agora()
    arq.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def _dswx_no_tile(perfil: dict, quando: pd.Timestamp) -> tuple[np.ndarray, str]:
    """WTR do granulo mais proximo da delineacao, amostrado na grade do tile GEOID."""
    import ee
    ee.Initialize()
    tr, crs, (h, w) = perfil["transform"], perfil["crs"], perfil["shape"]
    x0, y1 = tr.c, tr.f
    caixa = ee.Geometry.Rectangle([x0, y1 + tr.e * h, x0 + tr.a * w, y1], ee.Projection(crs), False)
    col = ee.ImageCollection(COLECAO_DSWX).filterBounds(caixa).filterDate(
        (quando - pd.Timedelta(days=3)).isoformat(), (quando + pd.Timedelta(days=3)).isoformat())
    tempos = col.aggregate_array("system:time_start").getInfo()
    if not tempos:
        return np.full((h, w), 255, np.uint8), "sem_granulo"
    alvo = min(tempos, key=lambda t: abs(pd.Timestamp(t, unit="ms") - quando))
    dia = pd.Timestamp(alvo, unit="ms")
    img = col.filterDate((dia - pd.Timedelta(hours=1)).isoformat(),
                         (dia + pd.Timedelta(hours=1)).isoformat()).select("WTR_Water_classification").mosaic()
    req = {"expression": img.unmask(255).toUint8(), "fileFormat": "NUMPY_NDARRAY",
           "grid": {"dimensions": {"width": w, "height": h},
                    "affineTransform": {"scaleX": tr.a, "shearX": 0, "translateX": x0,
                                        "shearY": 0, "scaleY": tr.e, "translateY": y1},
                    "crsCode": str(crs)}}
    arr = ee.data.computePixels(req)
    return np.asarray(arr["WTR_Water_classification"], np.uint8), str(dia)


def avaliar_v2() -> dict:
    import rasterio
    from pyproj import Transformer
    pre = pre_registrar_v2()
    import pyarrow.parquet as pq
    cat = pq.read_table(CATALOGO, columns=["event_id", "tile_id", "bbox_id", "delineation_time_post"]).to_pandas()
    cat = cat[cat.event_id == EVENTO]
    delin = {f"{r.event_id}-{r.tile_id}-{r.bbox_id}": pd.Timestamp(r.delineation_time_post) for r in cat.itertuples()}
    linhas, por_tile = [], {}
    for chave, t in sorted(_tiles().items()):
        if "label" not in t or "validity" not in t:
            continue
        with rasterio.open(t["label"]) as l, rasterio.open(t["validity"]) as v:
            ref, val = l.read(1), v.read(1)
            perfil = {"transform": l.transform, "crs": l.crs.to_string(), "shape": ref.shape}
        quando = delin[chave]
        wtr, granulo = _dswx_no_tile(perfil, quando)
        ins = np.full(wtr.shape, 255, np.uint8)
        ins[np.isin(wtr, WTR["seco"])] = 0
        ins[np.isin(wtr, WTR["agua"])] = 1
        gap_h = abs((pd.Timestamp(granulo) - quando).total_seconds()) / 3600 if granulo != "sem_granulo" else None
        avaliavel = (val > 0) & np.isin(ref, (0, 2))
        ii, jj = np.nonzero(avaliavel)
        d = pd.DataFrame({"tile": chave, "ref": ref[ii, jj], "ins": ins[ii, jj],
                          "sub": ((ii % PASSO_CONTEXTO) == 0) & ((jj % PASSO_CONTEXTO) == 0), "i": ii, "j": jj})
        d["primario"] = gap_h is not None and gap_h <= TOLERANCIA_PRIMARIA_H
        tr = perfil["transform"]
        d["x"], d["y"] = tr * (d.j.to_numpy() + 0.5, d.i.to_numpy() + 0.5)
        linhas.append(d)
        por_tile[chave] = {"estado": "avaliado" if avaliavel.any() else "sem_pixel_avaliavel", "granulo": granulo,
                           "delineacao_pos": str(quando), "defasagem_h": None if gap_h is None else round(gap_h, 2),
                           "primario": bool(d.primario.iloc[0]) if len(d) else False, "crs": perfil["crs"],
                           "wtr_contagem": {str(k): int(v) for k, v in zip(*np.unique(wtr[avaliavel], return_counts=True))},
                           **_metricas(ref[avaliavel], ins[avaliavel])}
        print(chave, {k: por_tile[chave][k] for k in ("granulo", "defasagem_h", "precisao", "revocacao", "npv_seco",
                                                      "cobertura")}, flush=True)
    d = pd.concat([x for x in linhas if len(x)], ignore_index=True)
    s = d[d["sub"]].copy()
    lon, lat = np.empty(len(s)), np.empty(len(s))
    for crs_, g in s.groupby(s.tile.map(lambda k: por_tile[k]["crs"])):
        lo, la = Transformer.from_crs(crs_, "EPSG:4326", always_xy=True).transform(g.x.to_numpy(), g.y.to_numpy())
        lon[s.index.get_indexer(g.index)], lat[s.index.get_indexer(g.index)] = lo, la
    s["hand"], s["declive"] = _contexto(lon, lat)
    s["faixa_hand"] = pd.cut(s.hand, [-1e9, 5, 15, 1e9], labels=["0-5m", "5-15m", ">15m"])
    s["faixa_declive"] = pd.cut(s.declive, [-1e9, 2, 5, 1e9], labels=["<2", "2-5", ">5"])
    res = {"pre_registro": pre["hash"], "versao": "v2-dswx-hls", "por_tile": por_tile}
    for nome, sel in (("primario", d.primario.to_numpy()), ("secundario_todos", np.ones(len(d), bool))):
        dd = d[sel]
        if dd.empty:
            res[nome] = {"estado": "not_testable", "motivo": "nenhum tile dentro da tolerancia temporal"}
            continue
        m = _metricas(dd.ref.to_numpy(), dd.ins.to_numpy())
        tiles = dd.tile.unique()
        grupos = {k: g for k, g in dd.groupby("tile")}
        rng, boot = np.random.default_rng(20261010), []
        for _ in range(2000):
            am = rng.choice(tiles, len(tiles), replace=True)
            tp = sum(int(((grupos[k].ins == 1) & (grupos[k].ref == 2)).sum()) for k in am)
            fp = sum(int(((grupos[k].ins == 1) & (grupos[k].ref == 0)).sum()) for k in am)
            if tp + fp:
                boot.append(tp / (tp + fp))
        m["precisao_ic90_por_tile"] = [float(np.quantile(boot, .05)), float(np.quantile(boot, .95))] if boot else None
        m["tiles"] = int(len(tiles))
        m["criterios"] = {
            "C1": m["precisao"] is not None and m["precisao"] >= 0.80,
            "C2": m["revocacao"] is not None and m["revocacao"] >= 0.40,
            "C3": m["npv_seco"] is not None and m["npv_seco"] >= 0.95,
            "C4": m["cobertura"] is not None and m["cobertura"] >= 0.30,
            "C5": m["precisao_ic90_por_tile"] is not None and m["precisao_ic90_por_tile"][0] >= 0.70}
        m["admitido_para_desenvolvimento"] = bool(all(m["criterios"].values())) if nome == "primario" else None
        ss = s[sel[s.index]]
        m["por_contexto"] = {c: {str(k): _metricas(g.ref.to_numpy(), g.ins.to_numpy())
                                 for k, g in ss.groupby(c, observed=True)} for c in ("faixa_hand", "faixa_declive")}
        res[nome] = m
    res["nao_e"] = ("concordancia algoritmo x interprete num evento conhecido; nao e reserva, nao confirma terreno, "
                    "nao vale para outro episodio sem referencia nova; optico so observa sem nuvem")
    res["quando_utc"] = _agora()
    res["hash_resumo"] = _sha_json({k: v for k, v in res.items() if k != "quando_utc"})
    (SAIDA_V2 / "resultado.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                             encoding="utf-8")
    return res


def diagnosticar_v2() -> dict:
    """Descritivo, depois do veredito: onde ficam os erros da v2 em relacao a borda da mancha de referencia.

    O DSWx e de 30 m e o rotulo de 10 m: erro a ate 30 m da borda e, em boa parte, resolucao. Nao muda criterio
    nem decisao; serve ao protocolo da proxima referencia (avaliar na grade do instrumento).
    """
    import rasterio
    from scipy.ndimage import distance_transform_edt
    import pyarrow.parquet as pq
    cat = pq.read_table(CATALOGO, columns=["event_id", "tile_id", "bbox_id", "delineation_time_post"]).to_pandas()
    cat = cat[cat.event_id == EVENTO]
    delin = {f"{r.event_id}-{r.tile_id}-{r.bbox_id}": pd.Timestamp(r.delineation_time_post) for r in cat.itertuples()}
    faixas = (0, 30, 90, 1e9)
    tot = {f: {"tp": 0, "fp": 0, "fn": 0} for f in ("ate_30m", "30_90m", "alem_90m")}
    for chave, t in sorted(_tiles().items()):
        if "label" not in t:
            continue
        with rasterio.open(t["label"]) as l, rasterio.open(t["validity"]) as v:
            ref, val = l.read(1), v.read(1)
            perfil = {"transform": l.transform, "crs": l.crs.to_string(), "shape": ref.shape}
        cache = SAIDA_V2 / "wtr" / f"{chave}.npy"
        if cache.exists():
            wtr = np.load(cache)
        else:
            wtr, _ = _dswx_no_tile(perfil, delin[chave])
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, wtr)
        av = (val > 0) & np.isin(ref, (0, 2))
        if not av.any():
            continue
        inund = ref == 2
        # distancia (m) a borda da mancha: dentro mede ate o seco, fora mede ate a agua
        dist = np.where(inund, distance_transform_edt(inund), distance_transform_edt(~inund)) * 10.0
        agua = wtr == 1
        for nome, lo, hi in (("ate_30m", 0, 30), ("30_90m", 30, 90), ("alem_90m", 90, 1e9)):
            f = av & (dist > lo) & (dist <= hi) if lo else av & (dist <= hi)
            tot[nome]["tp"] += int((f & agua & inund).sum())
            tot[nome]["fp"] += int((f & agua & ~inund).sum())
            tot[nome]["fn"] += int((f & ~agua & inund & (wtr == 0)).sum())
    for v in tot.values():
        v["precisao"] = round(v["tp"] / (v["tp"] + v["fp"]), 4) if v["tp"] + v["fp"] else None
        v["revocacao_entre_seco_ou_agua"] = round(v["tp"] / (v["tp"] + v["fn"]), 4) if v["tp"] + v["fn"] else None
    out = {"por_distancia_a_borda_da_referencia": tot, "faixas_m": faixas,
           "nao_e": "diagnostico depois do veredito; nao muda criterio nem admissao"}
    (SAIDA_V2 / "diagnostico_borda.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# IB-03a (10/10/2026): par S1 da MESMA orbita num evento brasileiro pos-2025, avaliado contra
# anotacao VISUAL CEGA em Sentinel-2. A anotacao e automatizada, nao humana: o resultado
# e "referencia visual por IA", serve ao desenvolvimento do instrumento e a decidir o insumo de
# anotacao (S2 x mosaicos); a referencia HUMANA continua exigida para confirmacao (subamostra
# re-anotada pela pesquisadora, kappa >= 0,8).

SAIDA_V3 = RUNS / "instrumento-brasileiro" / "ib-03a"
EVENTOS_BR = RUNS / "br-preparo-sem-rotulo" / "eventos.csv"
ELEGIBILIDADE = RUNS / "candidatos-independentes" / "elegibilidade_por_pergunta.json"
EXTRA_TAQUARI = {"eventid": "serra03:rs_taquari_jun2025", "inicio": "2025-06-16", "fim": "2025-07-01",
                 "bbox": [-51.940635, -29.411965, -51.605653, -29.050573]}
POR_ESTRATO = {"agua": 150, "seco": 150, "ambiguo": 100}
LADO_CHIP = 64                       # 64 x 10 m = 640 m
NUVEM_MAX_GRANULO = 40               # CLOUDY_PIXEL_PERCENTAGE do granulo S2
AREA_MIN_KM2 = 100                   # area comum S1 pre/pos x S2 x caixa, para caber a amostra
CRITERIO_INSUMO = {
    "anotavel_min": 0.80,
    "regra": ("entre pontos sem nuvem/sombra de nuvem na cena S2, se >= 80% recebem agua ou seco, o S2 basta. "
              "Se < 80% e o motivo dominante for 'misto' (resolucao), buscar imagem DATADA de maior resolucao; "
              "mosaico mensal (NICFI) nao serve, porque composto mensal nao fixa o instante da cena. Se o motivo "
              "dominante for nuvem, nenhum mosaico resolve: manter S1 + S2 e trocar de evento pela regra"),
}


def pre_registrar_v3() -> dict:
    SAIDA_V3.mkdir(parents=True, exist_ok=True)
    arq = SAIDA_V3 / "pre_registro.json"
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    p = {"id": "IB-03a", "papel": "desenvolvimento do instrumento; referencia visual por IA, nao humana",
         "pool_de_eventos": ("eventos brasileiros com inicio >= 2025-01-01 em br-preparo-sem-rotulo/eventos.csv, mais "
                             "RS jun/2025 (serra03); EXCLUI os candidatos independentes no desenho local (reservas "
                             "potenciais) listados em elegibilidade_por_pergunta.json"),
         "ordem": "data de inicio, depois id; o PRIMEIRO que cumprir a regra e o escolhido",
         "regra_do_evento": {"s1_pos": "IW VV no bbox, entre inicio e fim + 3 dias",
                             "s1_pre": "mesma orbita relativa e passagem, ate 24 dias antes do pos e antes do inicio",
                             "s2": f"S2 L2A a <= 1 dia do S1 pos, granulo com nuvem <= {NUVEM_MAX_GRANULO}%",
                             "so_metadados": True},
         "instrumento": {"regra": regra(), "grade": "30 m (UTM do evento)", "classes": ["agua", "seco", "ambiguo"],
                         "usa": "so VV pre/pos da mesma orbita; sem HAND, DEM, declive, rotulo"},
         "amostra": {"por_estrato_da_classe_do_instrumento": POR_ESTRATO, "nunca_por_terreno": True,
                     "exclui": ["JRC v1.4 occurrence >= 50 (contrato)", "fora da sobreposicao S1 pre/pos e S2"],
                     "semente": 20261010},
         "anotacao": {"quem": "anotador automatizado, visual, cego a HAND, DEM, escore, S1 e classe do instrumento",
                      "insumo": f"chips S2 {LADO_CHIP}x{LADO_CHIP} px a 10 m: pos cor verdadeira, pos falsa-cor "
                                "(B12/B8/B4) e pre falsa-cor; marca no centro",
                      "classes": ["agua", "seco", "nao_sei"],
                      "motivos": ["ok", "nuvem", "sombra", "misto", "urbano", "vegetacao_alagada?"],
                      "ordem": "embaralhada por hash do id; a chave de estrato fica em arquivo separado, lido so depois"},
         "metricas": "C1-C5 da v1 (precisao, revocacao, VPN do seco, cobertura, IC90 por bloco); "
                     "revocacao e cobertura ponderadas pelo inverso da fracao amostral de cada estrato",
         "criterios": CRITERIOS, "criterio_insumo_s2_x_mosaicos": CRITERIO_INSUMO,
         "confirmacao_humana": "subamostra aleatoria de 100 pontos re-anotada pela pesquisadora; kappa >= 0,8",
         "codigo_sha256": _sha(Path(__file__))}
    p["hash"] = _sha_json(p)
    p["quando_utc"] = _agora()
    arq.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def _pool() -> list[dict]:
    ind = {c["id"].split(":")[-1] for c in json.loads(ELEGIBILIDADE.read_text(encoding="utf-8"))["candidatos"]
           if c["independente_local"]}
    d = pd.read_csv(EVENTOS_BR, dtype={"eventid": str})
    d["inicio"] = pd.to_datetime(d.inicio, errors="coerce")
    d = d[(d.inicio >= "2025-01-01") & ~d.eventid.isin(ind)]
    pool = [{"eventid": r.eventid, "inicio": str(r.inicio.date()), "fim": str(r.fim),
             "bbox": [r.lon_min, r.lat_min, r.lon_max, r.lat_max], "estado_catalogo": r.estado}
            for r in d.itertuples()]
    pool.append({**EXTRA_TAQUARI, "estado_catalogo": "serra03"})
    return sorted(pool, key=lambda e: (e["inicio"], e["eventid"]))


def selecionar_evento(eventos=None, saida=None, pre=None) -> dict:
    """So metadados (datas, orbitas, nuvem do granulo). Para no primeiro evento que cumpre a regra."""
    import ee
    ee.Initialize()
    pre = pre_registrar_v3() if pre is None else pre
    saida = SAIDA_V3 if saida is None else Path(saida)
    log = []
    escolhido = None
    for e in (_pool() if eventos is None else eventos):
        g = ee.Geometry.Rectangle(e["bbox"])
        ini, fim = pd.Timestamp(e["inicio"]), pd.Timestamp(e["fim"]) + pd.Timedelta(days=3)
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(g)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .filterDate((ini - pd.Timedelta(days=30)).isoformat(), fim.isoformat()))
        props = s1.reduceColumns(ee.Reducer.toList(4), ["system:time_start", "relativeOrbitNumber_start",
                                                       "orbitProperties_pass", "system:index"]).get("list").getInfo()
        cenas = sorted({(pd.Timestamp(t, unit="ms"), o, p_, i) for t, o, p_, i in props})
        reg = {"evento": e["eventid"], "inicio": e["inicio"], "cenas_s1": len(cenas), "par": None, "s2": None}
        for t_pos, o, p_, i_pos in cenas:
            if not (ini <= t_pos <= fim):
                continue
            pres = [c for c in cenas if c[1] == o and c[2] == p_ and c[0] < ini and t_pos - c[0] <= pd.Timedelta(days=24)]
            if not pres:
                continue
            t_pre, _, _, i_pre = max(pres)
            # sobreposicao exigida pela amostra: S1 pos AND pre AND caixa do evento (defeito corrigido em 10/10:
            # a 1a versao so exigia que cada um tocasse a caixa, e o par escolhido nao cruzava o granulo S2)
            par = (ee.Image(f"COPERNICUS/S1_GRD/{i_pos}").geometry()
                   .intersection(ee.Image(f"COPERNICUS/S1_GRD/{i_pre}").geometry(), 100).intersection(g, 100))
            s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(par)
                  .filterDate((t_pos - pd.Timedelta(days=1)).isoformat(), (t_pos + pd.Timedelta(days=1)).isoformat())
                  .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", NUVEM_MAX_GRANULO))
                  .map(lambda im: im.set("area_comum_km2", im.geometry().intersection(par, 100).area(100).divide(1e6))))
            s2 = s2.filter(ee.Filter.gte("area_comum_km2", AREA_MIN_KM2))
            s2p = s2.reduceColumns(ee.Reducer.toList(4), ["system:index", "system:time_start",
                                                         "CLOUDY_PIXEL_PERCENTAGE", "area_comum_km2"]).get("list").getInfo()
            if s2p:
                reg["par"] = {"pos": [str(t_pos), i_pos], "pre": [str(t_pre), i_pre], "orbita": o, "passagem": p_}
                reg["s2"] = sorted([[i, str(pd.Timestamp(t, unit="ms")), c, round(a, 1)] for i, t, c, a in s2p],
                                   key=lambda x: (x[2], -x[3]))
                break
        log.append(reg)
        print(reg["evento"], reg["inicio"], reg["cenas_s1"], "ELEGIVEL" if reg["s2"] else "-", flush=True)
        if reg["s2"]:
            escolhido = {**e, **reg}
            break
    out = {"pre_registro": pre["hash"], "consultados": log, "escolhido": escolhido,
           "nao_e": "selecao por metadados; nenhum pixel lido", "quando_utc": _agora()}
    (saida / "selecao.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def _instrumento_ee(sel: dict):
    """Classe do instrumento S1 a 30 m (media em potencia linear), sem terreno: 1 agua, 0 seco, 2 ambiguo."""
    import ee
    rg = regra()
    def vv30(idx):
        img = ee.Image(f"COPERNICUS/S1_GRD/{idx}").select("VV")
        lin = ee.Image(10).pow(img.divide(10))
        proj = img.projection().atScale(30)
        return lin.reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(proj).log10().multiply(10)
    pos, pre_ = vv30(sel["par"]["pos"][1]), vv30(sel["par"]["pre"][1])
    queda = pre_.subtract(pos)
    agua = pos.lte(rg["vv_agua_db"]).And(queda.gte(rg["queda_agua_db"]))
    seco = pos.gt(rg["vv_agua_db"]).And(queda.lt(rg["queda_seco_db"]))
    classe = ee.Image(2).where(seco, 0).where(agua, 1).rename("classe").toInt()
    occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    return classe.updateMask(pos.mask().And(pre_.mask()).And(occ.lt(50)))


def amostrar_v3() -> dict:
    """Amostra estratificada pela classe do instrumento; a chave de estrato vai para arquivo separado."""
    import ee
    ee.Initialize()
    pre = pre_registrar_v3()
    sel = json.loads((SAIDA_V3 / "selecao.json").read_text(encoding="utf-8"))["escolhido"]
    s2_id = sel["s2"][0][0]
    s2 = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{s2_id}")
    regiao = (s2.geometry().intersection(ee.Geometry.Rectangle(sel["bbox"]), 10)
              .intersection(ee.Image(f"COPERNICUS/S1_GRD/{sel['par']['pos'][1]}").geometry(), 10)
              .intersection(ee.Image(f"COPERNICUS/S1_GRD/{sel['par']['pre'][1]}").geometry(), 10))
    classe = _instrumento_ee(sel)
    contagem = classe.reduceRegion(ee.Reducer.frequencyHistogram(), regiao, 30, maxPixels=1e10, tileScale=4).getInfo()
    fc = classe.stratifiedSample(numPoints=0, classBand="classe", region=regiao, scale=30, seed=20261010,
                                 classValues=[1, 0, 2], classPoints=[POR_ESTRATO["agua"], POR_ESTRATO["seco"],
                                                                     POR_ESTRATO["ambiguo"]],
                                 geometries=True, tileScale=4)
    feats = fc.getInfo()["features"]
    nomes = {1: "agua", 0: "seco", 2: "ambiguo"}
    pontos = []
    for f in feats:
        lon, lat = f["geometry"]["coordinates"]
        pid = hashlib.sha256(f"{lon:.6f},{lat:.6f}".encode()).hexdigest()[:10]
        pontos.append({"id": pid, "lon": lon, "lat": lat, "estrato": nomes[int(f["properties"]["classe"])]})
    pontos.sort(key=lambda p: p["id"])            # ordem cega: hash do id, nao o estrato
    for k, p in enumerate(pontos):
        p["codigo"] = f"P{k:03d}"
    chave = {"pre_registro": pre["hash"], "pixels_por_classe_na_regiao": contagem.get("classe"),
             "pontos": pontos, "nao_ler_antes_da_anotacao": True}
    (SAIDA_V3 / "chave_estratos.json").write_text(json.dumps(chave, ensure_ascii=False, indent=1), encoding="utf-8")
    cegos = [{k: p[k] for k in ("codigo", "id", "lon", "lat")} for p in pontos]
    (SAIDA_V3 / "pontos_cegos.json").write_text(json.dumps(cegos, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"pontos": len(pontos), "regiao_km2": round(regiao.area(10).getInfo() / 1e6, 1)}


def _s2_pre(sel: dict) -> str:
    """S2 anterior ao evento, mesmo granulo MGRS, menos nuvem em ate 60 dias antes do inicio (metadados)."""
    import ee
    tile = sel["s2"][0][0].split("_")[-1]
    ini = pd.Timestamp(sel["inicio"])
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filter(ee.Filter.eq("MGRS_TILE", tile[1:]))
           .filterDate((ini - pd.Timedelta(days=60)).isoformat(), ini.isoformat())
           .sort("CLOUDY_PIXEL_PERCENTAGE"))
    return col.first().get("system:index").getInfo()


def chips_v3(lado: int = LADO_CHIP) -> dict:
    """Pranchas de anotacao: so S2 (pos cor verdadeira, pos falsa-cor, pre falsa-cor). Sem S1, terreno ou classe."""
    import ee
    from PIL import Image, ImageDraw
    ee.Initialize()
    sel = json.loads((SAIDA_V3 / "selecao.json").read_text(encoding="utf-8"))["escolhido"]
    pontos = json.loads((SAIDA_V3 / "pontos_cegos.json").read_text(encoding="utf-8"))
    pos_id, pre_id = sel["s2"][0][0], _s2_pre(sel)
    bandas = ["B2", "B3", "B4", "B8", "B12"]
    img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{pos_id}").select(bandas).addBands(
        ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{pre_id}").select(bandas, [b + "_pre" for b in bandas]))
    crs = img.select("B4").projection().crs().getInfo()
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    pasta = SAIDA_V3 / "chips"
    pasta.mkdir(exist_ok=True)
    def esticar(a, teto):
        return (np.clip(a / teto, 0, 1) * 255).astype(np.uint8)
    paineis = {}
    for p in pontos:
        cache = pasta / f"{p['codigo']}.npz"
        if not cache.exists():
            x, y = tr.transform(p["lon"], p["lat"])
            x0, y1 = round(x / 10) * 10 - lado * 5, round(y / 10) * 10 + lado * 5
            arr = ee.data.computePixels({"expression": img.toFloat(), "fileFormat": "NUMPY_NDARRAY",
                                         "grid": {"dimensions": {"width": lado, "height": lado},
                                                  "affineTransform": {"scaleX": 10, "shearX": 0, "translateX": x0,
                                                                      "shearY": 0, "scaleY": -10, "translateY": y1},
                                                  "crsCode": crs}})
            np.savez_compressed(cache, **{b: np.asarray(arr[b], np.float32) for b in arr.dtype.names})
        a = np.load(cache)
        tc = np.dstack([esticar(a[b], 2500) for b in ("B4", "B3", "B2")])
        fc = np.dstack([esticar(a[b], 4000) for b in ("B12", "B8", "B4")])
        fp = np.dstack([esticar(a[b + "_pre"], 4000) for b in ("B12", "B8", "B4")])
        paineis[p["codigo"]] = [tc, fc, fp]
    # pranchas de 4 x 4 pontos; cada ponto com 3 paineis ampliados 3x e marca no centro (sem cobrir o pixel)
    esc, m = 3, 6
    cel = lado * esc
    larg, alt = 4 * (3 * cel + 2 * m + 20), 4 * (cel + 26)
    codigos = [p["codigo"] for p in pontos]
    pranchas = []
    for s in range(0, len(codigos), 16):
        folha = Image.new("RGB", (larg, alt), (255, 255, 255))
        dr = ImageDraw.Draw(folha)
        for k, cod in enumerate(codigos[s:s + 16]):
            ox, oy = (k % 4) * (3 * cel + 2 * m + 20), (k // 4) * (cel + 26)
            dr.text((ox + 2, oy + 2), cod, fill=(0, 0, 0))
            for j, painel in enumerate(paineis[cod]):
                im = Image.fromarray(painel).resize((cel, cel), Image.NEAREST)
                d2 = ImageDraw.Draw(im)
                c = cel // 2
                for dx, dy in ((-9, 0), (5, 0)):
                    d2.line([(c + dx, c), (c + dx + 4, c)], fill=(255, 0, 255), width=2)
                for dy in (-9, 5):
                    d2.line([(c, c + dy), (c, c + dy + 4)], fill=(255, 0, 255), width=2)
                folha.paste(im, (ox + j * (cel + m), oy + 14))
        arq = pasta.parent / f"prancha_{s // 16:02d}.png"
        folha.save(arq)
        pranchas.append(arq.name)
    out = {"s2_pos": pos_id, "s2_pre": pre_id, "pranchas": pranchas, "pontos": len(codigos),
           "paineis": "por ponto: [pos cor verdadeira | pos falsa-cor B12/B8/B4 | pre falsa-cor]; marca magenta no centro"}
    (SAIDA_V3 / "pranchas.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def congelar_anotacao_v3() -> dict:
    """Hash da anotacao ANTES de ler a chave de estratos; a avaliacao recusa anotacao alterada depois disto."""
    arq = SAIDA_V3 / "anotacao_ia.json"
    alvo = SAIDA_V3 / "anotacao_congelada.json"
    if alvo.exists():
        return json.loads(alvo.read_text(encoding="utf-8"))
    chave = SAIDA_V3 / "chave_estratos.json"
    r = {"anotacao_sha256": _sha(arq), "pontos": len(json.loads(arq.read_text(encoding="utf-8"))),
         "chave_sha256": _sha(chave), "chave_lida_antes": False, "quem_anotou": "anotador automatizado, visual, cego",
         "quando_utc": _agora()}
    alvo.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def avaliar_v3(blocos_graus: float = 0.05, reps: int = 2000) -> dict:
    """C1-C5 do instrumento S1 contra a referencia visual por IA, com pesos de estrato; e a decisao do insumo."""
    pre = pre_registrar_v3()
    cong = congelar_anotacao_v3()
    if _sha(SAIDA_V3 / "anotacao_ia.json") != cong["anotacao_sha256"]:
        raise RuntimeError("anotacao mudou depois de congelada")
    chave = json.loads((SAIDA_V3 / "chave_estratos.json").read_text(encoding="utf-8"))
    ann = json.loads((SAIDA_V3 / "anotacao_ia.json").read_text(encoding="utf-8"))
    d = pd.DataFrame(chave["pontos"])
    d["anot"] = d.codigo.map(lambda c: ann[c]["classe"])
    d["motivo"] = d.codigo.map(lambda c: ann[c]["motivo"])
    N = {"agua": 0, "seco": 0, "ambiguo": 0}
    for k, v in (chave["pixels_por_classe_na_regiao"] or {}).items():
        N[{"1": "agua", "0": "seco", "2": "ambiguo"}[str(k)]] = int(v)
    n = d.estrato.value_counts().to_dict()
    d["peso"] = d.estrato.map(lambda s: N[s] / n[s])
    d["bloco"] = (np.floor(d.lon / blocos_graus).astype(int).astype(str) + "_"
                  + np.floor(d.lat / blocos_graus).astype(int).astype(str))

    def metricas(x: pd.DataFrame) -> dict:
        ok = x[x.anot.isin(["agua", "seco"])]
        a, s = ok[ok.estrato == "agua"], ok[ok.estrato == "seco"]
        prec = (a.anot == "agua").mean() if len(a) else np.nan
        npv = (s.anot == "seco").mean() if len(s) else np.nan
        w_agua = ok.peso[ok.anot == "agua"]
        rev = (ok.peso[(ok.anot == "agua") & (ok.estrato == "agua")].sum() / w_agua.sum()) if w_agua.sum() else np.nan
        cob = ok.peso[ok.estrato.isin(["agua", "seco"])].sum() / ok.peso.sum() if ok.peso.sum() else np.nan
        return {"precisao": prec, "revocacao_ponderada": rev, "npv_seco": npv, "cobertura_ponderada": cob,
                "n_avaliaveis": int(len(ok)), "n_agua_inst": int(len(a)), "n_seco_inst": int(len(s))}
    m = metricas(d)
    rng = np.random.default_rng(20261010)
    blocos = d.bloco.unique()
    grupos = {b: g for b, g in d.groupby("bloco")}
    boot = []
    for _ in range(reps):
        x = pd.concat([grupos[b] for b in rng.choice(blocos, len(blocos), replace=True)])
        boot.append(metricas(x)["precisao"])
    boot = np.array([b for b in boot if np.isfinite(b)])
    m["precisao_ic90_por_bloco"] = [float(np.quantile(boot, .05)), float(np.quantile(boot, .95))]
    m["blocos"] = int(len(blocos))
    m = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}
    m["criterios"] = {"C1": bool(m["precisao"] >= 0.80), "C2": bool(m["revocacao_ponderada"] >= 0.40),
                      "C3": bool(m["npv_seco"] >= 0.95), "C4": bool(m["cobertura_ponderada"] >= 0.30),
                      "C5": bool(m["precisao_ic90_por_bloco"][0] >= 0.70)}
    m["admitido_para_desenvolvimento"] = bool(all(m["criterios"].values()))
    tab = {str(a): {str(b): int(v) for b, v in col.items()} for a, col in pd.crosstab(d.estrato, d.anot).to_dict().items()}
    # decisao do insumo (S2 x mosaicos), pela regra fixada antes
    sem_nuvem = d[~d.motivo.isin(["nuvem", "sombra"])]
    anotavel = float(sem_nuvem.anot.isin(["agua", "seco"]).mean()) if len(sem_nuvem) else None
    motivos = {str(k): int(v) for k, v in d[d.anot == "nao_sei"].motivo.value_counts().items()}
    dominante = max(motivos, key=motivos.get) if motivos else None
    if anotavel is not None and anotavel >= CRITERIO_INSUMO["anotavel_min"]:
        insumo = "S2 basta: entre pontos sem nuvem/sombra a fracao anotavel passa do limiar"
    elif dominante == "misto":
        insumo = "buscar imagem DATADA de maior resolucao; mosaico mensal nao serve (nao fixa o instante)"
    else:
        insumo = "nuvem domina: nenhum mosaico resolve; manter S1 + S2 e trocar de evento pela regra"
    res = {"pre_registro": pre["hash"], "anotacao": cong, "estrato_x_anotacao": tab,
           "pixels_por_estrato_na_regiao": N, "pontos_por_estrato": {str(k): int(v) for k, v in n.items()},
           "instrumento_s1_mesma_orbita": m,
           "insumo": {"fracao_anotavel_sem_nuvem_sombra": anotavel, "pontos_sem_nuvem_sombra": int(len(sem_nuvem)),
                      "nao_sei_por_motivo": motivos, "motivo_dominante": dominante, "decisao": insumo,
                      "nicfi": ("nao usado: composto mensal, acesso por cadastro na Planet; status apos o inicio de "
                                "2026 nao confirmado")},
           "nao_e": ("referencia visual por IA, nao humana; um evento (Manaus, jan/2025), uma cena; desenvolvimento do "
                     "instrumento; confirmacao exige subamostra re-anotada por pessoa (kappa >= 0,8)"),
           "quando_utc": _agora()}
    res["hash_resumo"] = _sha_json({k: v for k, v in res.items() if k != "quando_utc"})
    (SAIDA_V3 / "resultado.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                             encoding="utf-8")
    return res


def concordancia_humana() -> dict:
    """Revisão legada: exige os 100 códigos fixos e integridade da referência.

    Para registrar revisor/cegueira e preservar respostas versionadas, usar
    obs05_referencia_brasileira.py --humana CSV --revisor NOME --cega.
    """
    import obs05_referencia_brasileira as ib
    arq = SAIDA_V3 / "revisao_humana" / "amostra_para_anotar.csv"
    ib.verificar_piloto(SAIDA_V3)
    amostra = ib.ler(ib.SAIDA / "manifesto_revisao.json")["codigos_fixos"]
    out = ib.avaliar_respostas(ib.linhas_csv(arq), amostra, ib.ler(SAIDA_V3 / "anotacao_ia.json"))
    out["pontos"] = 100
    out["referencia_ia_atende_kappa"] = out["referencia_ia_validada"]
    out["referencia_ia_validada"] = False
    out["proveniencia_humana"] = "importar pelo IB-04 com identificação e declaração de cegueira; CSV legado não registra isso"
    out["respostas_sha256"] = ib.sha(arq)
    # Respostas parciais não apagam uma concordância histórica.
    if out["pontos_pendentes"] == 0:
        out["quando_utc"] = _agora()
        alvo = SAIDA_V3 / "revisao_humana" / ("concordancia_" + ib.sha(arq).split(":")[1][:16] + ".json")
        if not alvo.exists():
            alvo.write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-registrar", action="store_true")
    ap.add_argument("--adquirir", action="store_true")
    ap.add_argument("--avaliar", action="store_true")
    ap.add_argument("--pre-registrar-v2", action="store_true")
    ap.add_argument("--avaliar-v2", action="store_true")
    ap.add_argument("--diagnosticar-v2", action="store_true")
    ap.add_argument("--pre-registrar-v3", action="store_true")
    ap.add_argument("--selecionar-evento", action="store_true")
    ap.add_argument("--amostrar-v3", action="store_true")
    ap.add_argument("--chips-v3", action="store_true")
    ap.add_argument("--congelar-anotacao-v3", action="store_true")
    ap.add_argument("--avaliar-v3", action="store_true")
    ap.add_argument("--concordancia-humana", action="store_true")
    a = ap.parse_args()
    if a.concordancia_humana:
        print(json.dumps(concordancia_humana(), ensure_ascii=False, default=int))
    if a.congelar_anotacao_v3:
        print(congelar_anotacao_v3())
    if a.avaliar_v3:
        r = avaliar_v3()
        print(json.dumps({k: r[k] for k in ("estrato_x_anotacao", "pontos_por_estrato", "pixels_por_estrato_na_regiao",
                                            "instrumento_s1_mesma_orbita", "insumo")}, ensure_ascii=False, indent=1,
                         default=float))
    if a.amostrar_v3:
        print(amostrar_v3())
    if a.chips_v3:
        print(json.dumps(chips_v3(), ensure_ascii=False)[:600])
    if a.pre_registrar_v3:
        print(pre_registrar_v3()["hash"])
    if a.selecionar_evento:
        r = selecionar_evento()
        print(json.dumps(r["escolhido"], ensure_ascii=False, indent=1, default=str))
    if a.diagnosticar_v2:
        print(json.dumps(diagnosticar_v2(), ensure_ascii=False, indent=1))
    if a.pre_registrar_v2:
        print(pre_registrar_v2()["hash"])
    if a.avaliar_v2:
        r = avaliar_v2()
        print(json.dumps({k: {x: r[k].get(x) for x in ("precisao", "revocacao", "npv_seco", "cobertura",
                                                       "precisao_ic90_por_tile", "tiles", "criterios",
                                                       "admitido_para_desenvolvimento", "estado")}
                          for k in ("primario", "secundario_todos")}, ensure_ascii=False, indent=1, default=str))
    if a.pre_registrar:
        print(pre_registrar()["hash"])
    if a.adquirir:
        r = adquirir()
        print(json.dumps({k: r[k] for k in ("tiles_no_catalogo", "membros_por_tile", "falhas", "bytes_solicitados")},
                         ensure_ascii=False, indent=1))
    if a.avaliar:
        r = avaliar()
        print(json.dumps({k: r.get(k) for k in ("primario", "secundario_todos")}, ensure_ascii=False, indent=1,
                         default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
