"""Rodada 2, parte 3: referencia de agua de outro instrumento (Sentinel-2 L2A / Landsat L2) nos blocos da pilha.

Linhagem `referencia_fisica_forcamento_v1` (17/10/2026). Nada aqui treina ou le reserva.

Etapas (cada uma grava manifesto e recusa sobrescrever com entradas diferentes):
  selecionar     componentes FLUVIAIS ja classificados (>= 50% dos positivos CEMS com processo_ponto
                 fluvial), ordenados pelo numero de positivos fluviais; 2 blocos por componente com
                 mais positivos. Sem olhar predicao, erro do SAR ou JRC. Piloto = 4 primeiros;
                 ampliacao = ate 12 na MESMA ordem.
  oportunidades  por bloco: todas as passagens S1 admitidas x cenas S2 SR harmonizado e Landsat 8/9
                 C2 L2 (metadado Earth Engine: horario UTC, nuvem global, tile); Δt de cada par.
                 Contrato existente: |Δt| <= 1 h. Nao e relaxado.
  datas          instantes-alvo opticos (desenho "optico no proprio instante" quando nao ha par em 1 h):
                 por bloco, metade na janela de evento e metade fora, em ordem de nuvem global
                 crescente (so prioridade de download; nuvem global NAO admite nada).
  baixar         B3, B8, B11, SCL e probabilidade s2cloudless no grid do bloco (30 m; B11 nativa 20 m):
                 refletancias por media 10/20->30 m, SCL e nuvem por maximo/moda conservadores.
  admitir        por pixel: nao_observavel (SCL 0,1,3,8,9,10,11; s2cloudless >= 40; borda/sem dado),
                 permanente (JRC occurrence >= 50), agua (MNDWI >= 0), seco (MNDWI <= -0,15),
                 indeterminado (entre). Regra externa congelada (Xu 2006; folga declarada), nao ajustada
                 aqui. Agua em JRC = 0 e max_extent = 0 = agua_fora_do_historico; com historico = sazonal.
                 Indice optico automatico = alvo instrumental de outro instrumento, nao verdade de campo.
  revisao        amostra estratificada pela CLASSE OPTICA e componente (nunca pelo SAR, JRC ou predicao),
                 recortes PNG cegos, formulario CSV e importador. Revisao de IA nao conta como humana.

Uso:
    python scripts/ia/rodada2_referencia_optica.py selecionar|oportunidades|datas|baixar|admitir|revisao [--fase piloto|ampliacao]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for s in ("ia", "sar", "servico", "metodo", "manutencao"):
    sys.path.insert(0, str(REPO / "scripts" / s))
import conjunta_base as cb  # noqa: E402
import pilha_s1_mesmo_lugar as ps  # noqa: E402
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402

RF = REPO / "local_runs" / "referencia-fisica-forcamento" / "v1"
OPT = RF / "optica"
FASES = {"piloto": {"componentes": 4, "blocos": 2, "datas": 2, "teto_gib": 2.0},
         "ampliacao": {"componentes": 12, "blocos": 2, "datas": 4, "teto_gib": 6.0}}
TETO_PEDIDOS = 600
TOLERANCIA_CONTRATO_S = 3600
SCL_INVALIDO = (0, 1, 3, 8, 9, 10, 11)
NUVEM_MAX = 40
MNDWI_AGUA, MNDWI_SECO = 0.0, -0.15
JRC_PERMANENTE = 50
MIN_FRACAO_OBSERVAVEL = 0.5
CODIGO_CLASSE = {"nao_observavel": 0, "permanente": 1, "agua": 2, "seco": 3, "indeterminado": 4}


def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def gravar(p, o):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".part")
    t.write_text(json.dumps(o, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    t.replace(p)


# ===================================================================== selecao

def selecionar(fase: str) -> dict:
    arq = OPT / "selecao.json"
    pts = pd.read_parquet(cb.BASE / "pontos.parquet", columns=["unidade", "fonte", "classe", "processo_ponto", "lat", "lon"])
    pos = pts[(pts.fonte == "cems") & (pts.classe == 1)]
    proc = pos.groupby("unidade").processo_ponto.agg(lambda s: (s == "fluvial").mean())
    nflu = pos[pos.processo_ponto == "fluvial"].groupby("unidade").size()
    blocos = ler(ps.SAIDA / "blocos.json")["blocos"]
    der = pd.read_csv(ps.SAIDA / "blocos_derivados.csv").set_index("bloco")
    comps_pilha = {b["componente"] for b in blocos if b["bloco"] in der.index and bool(der.loc[b["bloco"], "em_b"])}
    fluviais = [c for c in sorted(comps_pilha) if proc.get(c, 0) >= .5]
    ordem = sorted(fluviais, key=lambda c: (-int(nflu.get(c, 0)), c))
    sel = {}
    for c in ordem[:FASES["ampliacao"]["componentes"]]:
        bs = [b for b in blocos if b["componente"] == c and b["bloco"] in der.index and bool(der.loc[b["bloco"], "em_b"])]
        bs = sorted(bs, key=lambda b: (-b["n_pos"], -b["n_pontos"], b["bloco"]))[:FASES["ampliacao"]["blocos"]]
        sel[c] = [b["bloco"] for b in bs]
    r = {"tipo": "selecao_congelada_referencia_optica", "regra": "componente fluvial (>=50% positivos CEMS fluviais) da pilha em (b), "
         "ordenado por positivos fluviais; 2 blocos com mais positivos; sem predicao, erro SAR ou JRC",
         "ordem": list(sel), "blocos": sel, "fracao_fluvial": {c: float(proc.get(c, 0)) for c in sorted(comps_pilha)},
         "positivos_fluviais": {c: int(nflu.get(c, 0)) for c in sel},
         "piloto": list(sel)[:FASES["piloto"]["componentes"]], "ampliacao": list(sel),
         "fontes": {str(p.relative_to(REPO)): sha256_arquivo(p) for p in (cb.BASE / "pontos.parquet", ps.SAIDA / "blocos.json",
                                                                          ps.SAIDA / "blocos_derivados.csv")}}
    if arq.exists() and ler(arq)["blocos"] != r["blocos"]:
        raise ValueError("selecao mudou depois de congelada")
    if not arq.exists():
        gravar(arq, {**r, "quando_utc": agora_utc()})
    return {"fase": fase, "componentes": r[fase], "fluviais_na_pilha": len(fluviais)}


def blocos_da_fase(fase: str) -> list[dict]:
    s = ler(OPT / "selecao.json")
    nomes = {n for c in s[fase] for n in s["blocos"][c][:FASES[fase]["blocos"]]}
    return [b for b in ler(ps.SAIDA / "blocos.json")["blocos"] if b["bloco"] in nomes]


# ===================================================================== oportunidades

def _catalogo_optico(ee, b: dict, ini: str, fim: str) -> list[dict]:
    g = ps._bloco_4326(b)
    geom = ee.Geometry.Polygon(list(g.exterior.coords))
    out = []
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(ini, fim)
    info = s2.reduceColumns(ee.Reducer.toList(4), ["system:index", "system:time_start", "CLOUDY_PIXEL_PERCENTAGE", "MGRS_TILE"]).getInfo()["list"]
    out += [{"colecao": "S2_SR_HARMONIZED", "id": i, "t_ms": t, "nuvem_global": n, "tile": tl} for i, t, n, tl in info]
    for col in ("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"):
        ls = ee.ImageCollection(col).filterBounds(geom).filterDate(ini, fim)
        info = ls.reduceColumns(ee.Reducer.toList(3), ["system:index", "system:time_start", "CLOUD_COVER"]).getInfo()["list"]
        out += [{"colecao": col.split("/")[1] + "_C2_L2", "id": i, "t_ms": t, "nuvem_global": n, "tile": None} for i, t, n in info]
    return out


def oportunidades(fase: str) -> dict:
    ee = ps._ee()
    man = pd.read_parquet(ps.SAIDA / "manifesto_admissao.parquet")
    adm = man[man.admitida.astype(bool)]
    linhas, opt_rows, erros = [], [], []
    for b in blocos_da_fase(fase):
        cache = OPT / "catalogos" / f"{b['bloco']}.json"
        a = adm[adm.bloco == b["bloco"]]
        t1 = pd.to_datetime(a.quando_utc, utc=True)
        ini, fim = str((t1.min() - pd.Timedelta(days=2)).date()), str((t1.max() + pd.Timedelta(days=2)).date())
        try:
            if cache.exists() and ler(cache)["janela"] == [ini, fim]:
                cat = ler(cache)["itens"]
            else:
                cat = _catalogo_optico(ee, b, ini, fim)
                gravar(cache, {"bloco": b["bloco"], "janela": [ini, fim], "itens": cat, "quando_utc": agora_utc(), "completo": True})
        except Exception as e:  # noqa: BLE001 - falha de consulta NAO e ausencia de cena
            erros.append({"bloco": b["bloco"], "erro": str(e)[:300]})
            continue
        to = pd.to_datetime([c["t_ms"] for c in cat], unit="ms", utc=True)
        for c, t in zip(cat, to):
            opt_rows.append({"bloco": b["bloco"], "componente": b["componente"], **c, "utc": t.isoformat()})
        for _, r in a.iterrows():
            ts = pd.Timestamp(r.quando_utc)
            d = np.abs((to - ts).total_seconds()) if len(to) else np.array([])
            k = int(np.argmin(d)) if len(d) else -1
            linhas.append({"bloco": b["bloco"], "componente": b["componente"], "s1_utc": ts.isoformat(), "janela": r.janela,
                           "optico_mais_proximo": cat[k]["id"] if k >= 0 else None,
                           "colecao": cat[k]["colecao"] if k >= 0 else None,
                           "delta_horas": float(d[k] / 3600) if k >= 0 else None,
                           "par_no_contrato_1h": bool(k >= 0 and d[k] <= TOLERANCIA_CONTRATO_S),
                           "pares_em_24h": int((d <= 86400).sum()) if len(d) else 0})
    q = pd.DataFrame(linhas)
    o = pd.DataFrame(opt_rows)
    q.to_parquet(OPT / f"oportunidades_s1_{fase}.parquet", index=False)
    o.to_parquet(OPT / f"cenas_opticas_{fase}.parquet", index=False)
    r = {"tipo": "quadro_de_oportunidades", "fase": fase, "quando_utc": agora_utc(), "blocos": int(q.bloco.nunique()) if len(q) else 0,
         "passagens_s1": len(q), "cenas_opticas": len(o), "pares_contrato_1h": int(q.par_no_contrato_1h.sum()) if len(q) else 0,
         "passagens_com_optico_em_24h": int((q.pares_em_24h > 0).sum()) if len(q) else 0,
         "delta_horas_mediana_mais_proximo": float(q.delta_horas.median()) if len(q) else None,
         "erros_de_consulta": erros, "completo": not erros,
         "leitura": "falha de consulta nao prova ausencia de cena; nuvem global e so prioridade, nao admissao"}
    gravar(OPT / f"oportunidades_{fase}.json", r)
    return {k: r[k] for k in ("blocos", "passagens_s1", "cenas_opticas", "pares_contrato_1h", "passagens_com_optico_em_24h",
                              "delta_horas_mediana_mais_proximo", "completo")}


# ===================================================================== datas-alvo

def datas(fase: str) -> dict:
    o = pd.read_parquet(OPT / f"cenas_opticas_{fase}.parquet")
    q = pd.read_parquet(OPT / f"oportunidades_s1_{fase}.parquet")
    from datetime import date
    evs = ler(ps.SAIDA / "blocos.json")["eventos"]
    n = FASES[fase]["datas"]
    alvo = []
    for bloco, g in o[o.colecao == "S2_SR_HARMONIZED"].groupby("bloco"):
        comp = g.componente.iloc[0]
        t = pd.to_datetime(g.utc, format="ISO8601")
        janelas = [(date.fromisoformat(e["j0"]), date.fromisoformat(e["j1"])) for e in evs if e["unidade"] == comp]
        g = g.assign(janela=[ps.classificar_data(x.date(), janelas) for x in t])  # mesma regra da pilha
        pares = set(q[(q.bloco == bloco) & q.par_no_contrato_1h].optico_mais_proximo)
        g = g.assign(par_1h=g.id.isin(pares)).sort_values(["par_1h", "nuvem_global"], ascending=[False, True])
        esc = pd.concat([g[g.janela == "evento"].head(n // 2), g[g.janela == "fora"].head(n - n // 2)])
        for _, r in esc.iterrows():
            alvo.append({"bloco": bloco, "componente": comp, "id": r.id, "utc": r.utc, "janela": r.janela,
                         "nuvem_global": r.nuvem_global, "par_s1_1h": bool(r.par_1h)})
    t = pd.DataFrame(alvo)
    gravar(OPT / f"datas_alvo_{fase}.json", {"tipo": "instantes_alvo_opticos", "fase": fase, "quando_utc": agora_utc(),
                                             "regra": "por bloco metade evento/metade fora; par S1 em 1 h primeiro; depois nuvem global crescente (prioridade, nao admissao)",
                                             "alvos": t.to_dict("records")})
    return {"alvos": len(t), "evento": int((t.janela == "evento").sum()) if len(t) else 0,
            "com_par_s1_1h": int(t.par_s1_1h.sum()) if len(t) else 0}


# ===================================================================== baixar

def baixar(fase: str) -> dict:
    ee = ps._ee()
    alvos = ler(OPT / f"datas_alvo_{fase}.json")["alvos"]
    blocos = {b["bloco"]: b for b in ler(ps.SAIDA / "blocos.json")["blocos"]}
    pasta = OPT / "rasters"
    pasta.mkdir(parents=True, exist_ok=True)
    reg = OPT / "downloads.jsonl"
    feitos = {json.loads(x)["arquivo"] for x in reg.read_text(encoding="utf-8").splitlines()} if reg.exists() else set()
    pedidos, erros_seguidos, ok = 0, 0, 0
    teto = FASES[fase]["teto_gib"] * 1024 ** 3
    for a in alvos:
        b = blocos[a["bloco"]]
        arq = pasta / f"{a['bloco']}__{a['id']}.npz"
        if str(arq) in feitos and arq.exists():
            continue
        usado = sum(f.stat().st_size for f in pasta.glob("*.npz"))
        if usado > teto or pedidos >= TETO_PEDIDOS or erros_seguidos >= 5:
            break
        t0 = time.monotonic()
        try:
            img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{a['id']}")
            nuv = ee.Image(f"COPERNICUS/S2_CLOUD_PROBABILITY/{a['id']}").select("probability")
            tr = [ps.PIXEL_M, 0, b["x0"], 0, -ps.PIXEL_M, b["y0"]]
            refl = (img.select(["B3", "B8", "B11"]).toFloat().reduceResolution(ee.Reducer.mean(), maxPixels=64)
                    .reproject(crs=b["crs"], crsTransform=tr))
            scl = img.select("SCL").reduceResolution(ee.Reducer.max(), maxPixels=64).reproject(crs=b["crs"], crsTransform=tr)
            pr = nuv.reduceResolution(ee.Reducer.max(), maxPixels=64).reproject(crs=b["crs"], crsTransform=tr)
            ex = ee.Image.cat([refl.unmask(-1).toInt16(), scl.unmask(0).toInt16().rename("SCL_MAX"),
                               img.select("SCL").reproject(crs=b["crs"], crsTransform=tr).unmask(0).toInt16().rename("SCL_CENTRO"),
                               pr.unmask(255).toInt16().rename("NUVEM_MAX")])
            arr = ps._pedido(ee, ex, b)
            pedidos += 1
            np.savez_compressed(arq, **{k: arr[k] for k in arr.dtype.names}, id=a["id"], utc=a["utc"])
            with reg.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"arquivo": str(arq), "id": a["id"], "bloco": a["bloco"], "sha256": sha256_arquivo(arq),
                                    "bytes": arq.stat().st_size, "segundos": time.monotonic() - t0, "quando_utc": agora_utc()}) + "\n")
            ok += 1
            erros_seguidos = 0
        except Exception as e:  # noqa: BLE001
            erros_seguidos += 1
            with reg.with_suffix(".erros.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": a["id"], "bloco": a["bloco"], "erro": str(e)[:300], "quando_utc": agora_utc()}) + "\n")
    return {"baixados_nesta_chamada": ok, "pedidos": pedidos, "erros_seguidos": erros_seguidos,
            "bytes_totais": sum(f.stat().st_size for f in pasta.glob("*.npz"))}


# ===================================================================== admitir

def classes_pixel(b3, b8, b11, scl_max, scl_centro, nuvem, jrc_occ, jrc_max):
    """Classe por pixel. Refletancia em 1e4 (S2 L2A harmonizado). -1 = sem dado."""
    valido = (b3 > 0) & (b11 > 0) & (b8 > 0)
    nao_obs = ~valido | np.isin(scl_centro, SCL_INVALIDO) | np.isin(scl_max, (8, 9, 10, 11)) | (nuvem >= NUVEM_MAX)
    with np.errstate(divide="ignore", invalid="ignore"):
        mndwi = (b3.astype(float) - b11) / (b3.astype(float) + b11)
    c = np.full(b3.shape, CODIGO_CLASSE["indeterminado"], np.int8)
    c[mndwi >= MNDWI_AGUA] = CODIGO_CLASSE["agua"]
    c[mndwi <= MNDWI_SECO] = CODIGO_CLASSE["seco"]
    c[jrc_occ >= JRC_PERMANENTE] = CODIGO_CLASSE["permanente"]
    c[nao_obs] = CODIGO_CLASSE["nao_observavel"]
    historico = (jrc_occ > 0) | (jrc_max > 0)
    return c, mndwi, historico


def admitir(fase: str) -> dict:
    alvos = {a["id"] + "|" + a["bloco"]: a for a in ler(OPT / f"datas_alvo_{fase}.json")["alvos"]}
    blocos = {b["bloco"]: b for b in ler(ps.SAIDA / "blocos.json")["blocos"]}
    cems = pd.read_parquet(cb.BASE / "pontos.parquet", columns=["evento_id", "fonte", "janela_inicio"])
    datas_produto = cems[cems.fonte == "cems"].groupby("evento_id").janela_inicio.agg(lambda s: set(map(str, s)))
    linhas = []
    for chave, a in alvos.items():
        arq = OPT / "rasters" / f"{a['bloco']}__{a['id']}.npz"
        b = blocos[a["bloco"]]
        if not arq.exists():
            linhas.append({**a, "estado": "nao_baixado", "motivo": "fora do teto, erro de consulta ou ainda nao baixado"})
            continue
        z = np.load(arq, allow_pickle=True)
        with np.load(ps.SAIDA / "pilhas" / f"{a['bloco']}.npz", allow_pickle=True) as p:
            jocc, jmax = p["jrc_occ"], p["jrc_max"]
        c, mndwi, hist = classes_pixel(z["B3"], z["B8"], z["B11"], z["SCL_MAX"], z["SCL_CENTRO"], z["NUVEM_MAX"], jocc, jmax)
        (OPT / "classes").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OPT / "classes" / f"{a['bloco']}__{a['id']}.npz", classe=c, mndwi=mndwi.astype(np.float32), historico=hist)
        obs = float((c != 0).mean())
        dia = pd.Timestamp(a["utc"]).date().isoformat()
        mesmo_dia_produto = any(dia in s for e, s in datas_produto.items() if e.split(":")[-1] in a["bloco"] or e == b.get("evento_id"))
        admit = obs >= MIN_FRACAO_OBSERVAVEL
        linhas.append({**a, "estado": "admitido" if admit else "recusado", "fracao_observavel": obs,
                       "motivo": None if admit else f"fracao observavel {obs:.2f} < {MIN_FRACAO_OBSERVAVEL}",
                       "pixels_agua": int((c == 2).sum()), "pixels_seco": int((c == 3).sum()),
                       "pixels_indeterminado": int((c == 4).sum()), "pixels_permanente": int((c == 1).sum()),
                       "agua_fora_do_historico": int(((c == 2) & ~hist).sum()), "agua_com_historico": int(((c == 2) & hist).sum()),
                       "instrumento": "Sentinel-2 MSI L2A harmonizado (GEE COPERNICUS/S2_SR_HARMONIZED) + s2cloudless",
                       "crs": b["crs"], "resolucao_efetiva_m": 30, "resolucao_nativa_m": {"B3": 10, "B8": 10, "B11": 20, "SCL": 20},
                       "regra": f"MNDWI>={MNDWI_AGUA} agua; <={MNDWI_SECO} seco; SCL/nuvem>={NUVEM_MAX} nao observavel",
                       "independencia": {"instrumental_do_SAR": True, "do_limiar_SAR": "limiar SAR calibrado no JRC (Landsat), nao no S2",
                                         "do_rotulo_CEMS": "suspeita" if mesmo_dia_produto else "data diferente do produto CEMS",
                                         "territorial": False, "selecao": "blocos escolhidos onde havia ponto CEMS; datas por nuvem global"},
                       "arquivo_sha256": sha256_arquivo(arq)})
    t = pd.DataFrame(linhas)
    t.to_parquet(OPT / f"admissao_{fase}.parquet", index=False)
    r = {"tipo": "admissao_referencia_optica", "fase": fase, "quando_utc": agora_utc(),
         "consultados": len(t), "baixados": int((t.estado != "nao_baixado").sum()), "admitidos": int((t.estado == "admitido").sum()),
         "recusados": int((t.estado == "recusado").sum()), "componentes_admitidos": int(t[t.estado == "admitido"].componente.nunique()),
         "blocos_admitidos": int(t[t.estado == "admitido"].bloco.nunique()),
         "datas_evento_admitidas": int(((t.estado == "admitido") & (t.janela == "evento")).sum()),
         "suspeita_mesma_cena_cems": int(((t.estado == "admitido") & (t.independencia.astype(str).str.contains("suspeita"))).sum()) if len(t) else 0,
         "nivel": "referencia de outro instrumento, automatica (nivel 2 com limites): nao e verdade de campo nem revisao humana",
         "vieses": ["datas escolhidas com nuvem global baixa (selecao por clima)", "optico nao ve agua sob nuvem/vegetacao densa",
                    "blocos onde havia ponto CEMS"]}
    gravar(OPT / f"admissao_{fase}.json", r)
    return {k: r[k] for k in ("consultados", "baixados", "admitidos", "recusados", "componentes_admitidos", "blocos_admitidos",
                              "datas_evento_admitidas", "suspeita_mesma_cena_cems")}


# ===================================================================== revisao humana (pacote)

def revisao(fase: str, por_estrato: int = 15, semente: int = 20261017) -> dict:
    """Amostra estratificada por classe optica x componente; recortes cegos (sem SAR, sem JRC, sem predicao)."""
    from PIL import Image
    t = pd.read_parquet(OPT / f"admissao_{fase}.parquet")
    t = t[t.estado == "admitido"]
    rng = np.random.default_rng(semente)
    pasta = OPT / "revisao" / fase
    (pasta / "recortes").mkdir(parents=True, exist_ok=True)
    form = []
    for _, a in t.iterrows():
        z = np.load(OPT / "rasters" / f"{a.bloco}__{a.id}.npz", allow_pickle=True)
        c = np.load(OPT / "classes" / f"{a.bloco}__{a.id}.npz")["classe"]
        for nome, cod in (("agua", 2), ("seco", 3), ("indeterminado", 4)):
            ii = np.argwhere(c[8:-8, 8:-8] == cod) + 8
            if not len(ii):
                continue
            for k in rng.choice(len(ii), min(por_estrato // max(len(t), 1) + 1, len(ii)), replace=False):
                i, j = ii[k]
                rid = sha256_json([a.bloco, a.id, int(i), int(j)])[7:19]
                rgb = np.stack([z[x][i - 8:i + 8, j - 8:j + 8] for x in ("B11", "B8", "B3")], -1).astype(float)
                rgb = np.clip(rgb / 3000 * 255, 0, 255).astype(np.uint8)
                im = Image.fromarray(rgb).resize((160, 160), Image.NEAREST)
                im.putpixel((80, 80), (255, 0, 255))
                im.save(pasta / "recortes" / f"{rid}.png")
                form.append({"id": rid, "arquivo": f"recortes/{rid}.png", "pergunta": "o pixel central (magenta) tem agua visivel?",
                             "resposta": "", "confianca": "", "revisor": "", "_bloco": a.bloco, "_cena": a.id, "_linha": int(i),
                             "_coluna": int(j), "_estrato_optico": nome})
    f = pd.DataFrame(form).sample(frac=1, random_state=semente)
    f[["id", "arquivo", "pergunta", "resposta", "confianca", "revisor"]].to_csv(pasta / "formulario_cego.csv", index=False)
    f.to_csv(pasta / "gabarito_oculto.csv", index=False)
    gravar(pasta / "instrucoes.json", {"tipo": "pacote_de_revisao_humana_cega", "recortes": len(f),
           "como": "abrir formulario_cego.csv; para cada recorte (falsa cor SWIR/NIR/verde, 16x16 pixels de 30 m) responder agua|seco|incerto e confianca 1-3",
           "nao_mostra": "classe optica automatica, SAR, JRC, predicao", "importar": "python scripts/ia/rodada2_referencia_optica.py importar_revisao --fase " + fase,
           "uso": "metade para desenvolvimento do instrumento, metade para avaliacao (separacao pelo id)", "revisao_ia_nao_conta": True})
    return {"recortes": len(f), "pasta": str(pasta)}


REVISORES_IA = ("codex", "claude", "gpt", "ia_", "_ia", "ia-", "llm", "modelo")
PARTICAO_V2 = REPO / "local_runs" / "confirmacao-rodada3" / "v1" / "revisao" / "particao_grupos_v2.csv"


def importar_revisao(fase: str) -> dict:
    """Somente revisao HUMANA declarada e verificavel (emenda rodada 3, 17/10/2026).

    Exige `revisor_humano.json` na pasta com tipo_revisor=humano, revisor_id e o sha256 do formulario preenchido;
    toda linha respondida tem de trazer esse revisor. Revisor com nome de IA, formulario igual ao da IA ou
    declaracao ausente/divergente sao recusados: respostas de IA vivem em `confirmacao-rodada3/v1/revisao/revisao_ia/`.
    A particao e por grupos (cena-bloco-componente, `particao_grupos_v2.csv`); a antiga por paridade do id misturava
    os mesmos 23 blocos nos dois papeis e nao e mais usada.
    """
    pasta = OPT / "revisao" / fase
    form = pasta / "formulario_cego.csv"
    f = pd.read_csv(form).merge(pd.read_csv(pasta / "gabarito_oculto.csv"), on="id", suffixes=("", "_g"))
    resp = f[f.resposta.astype(str).str.lower().isin(["agua", "seco", "incerto"])]
    if resp.empty:
        return {"estado": "sem_respostas_humanas", "recortes": len(f)}
    decl = pasta / "revisor_humano.json"
    if not decl.exists():
        raise SystemExit("revisor_humano.json ausente: revisao sem identidade verificavel nao e importada")
    d = ler(decl)
    rid = str(d.get("revisor_id", "")).strip()
    if d.get("tipo_revisor") != "humano" or not rid or any(x in rid.lower() for x in REVISORES_IA):
        raise SystemExit(f"declaracao nao e de revisor humano: {d}")
    if d.get("formulario_sha256") != sha256_arquivo(form):
        raise SystemExit("formulario preenchido difere do declarado pelo revisor")
    if set(resp.revisor.astype(str).str.strip()) != {rid}:
        raise SystemExit("linhas respondidas com revisor diferente do declarado")
    ia = REPO / "local_runs" / "coordenacao-rodada3" / "v1" / "revisao_ia" / "classificacao_visual_ia.csv"
    if ia.exists():
        r_ia = pd.read_csv(ia).set_index("id").resposta
        if resp.set_index("id").resposta.str.lower().equals(r_ia.reindex(resp.id).str.lower()):
            raise SystemExit("respostas identicas as da revisao de IA")
    part = pd.read_csv(PARTICAO_V2)[["id", "papel_v2"]]
    resp = resp.merge(part, on="id", how="left").rename(columns={"papel_v2": "papel"})
    if resp.papel.isna().any():
        raise SystemExit("recorte sem papel na particao por grupos")
    tab = pd.crosstab([resp.papel, resp._estrato_optico], resp.resposta.str.lower()).to_dict("index")
    gravar(pasta / "importado.json", {"respondidos": len(resp), "tipo_revisor": "humano", "revisor_id": rid,
                                      "declaracao": sha256_arquivo(decl), "particao": sha256_arquivo(PARTICAO_V2),
                                      "tabela": {str(k): v for k, v in tab.items()}, "quando_utc": agora_utc()})
    return {"respondidos": len(resp), "revisor_id": rid}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["selecionar", "oportunidades", "datas", "baixar", "admitir", "revisao", "importar_revisao"])
    ap.add_argument("--fase", default="piloto", choices=list(FASES))
    a = ap.parse_args()
    f = {"selecionar": selecionar, "oportunidades": oportunidades, "datas": datas, "baixar": baixar, "admitir": admitir,
         "revisao": revisao, "importar_revisao": importar_revisao}[a.acao]
    print(json.dumps(f(a.fase), ensure_ascii=False, indent=1, default=str)[:3000])
