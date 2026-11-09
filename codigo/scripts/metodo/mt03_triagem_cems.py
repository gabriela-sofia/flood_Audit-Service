"""MT03, passo 4: triagem CEMS por METADADOS e particao desenvolvimento x reserva ANTES de qualquer rotulo.

O que entra aqui e so o que o catalogo publico do Rapid Mapping devolve sobre a ativacao: categoria,
subcategoria, nome, data, AOIs (extensao), produtos (tipo, versao, camadas, imagens: sensor, classe de
resolucao, instante). A resposta publica tambem traz estatisticas de area afetada -- isso e informacao de
rotulo; o campo `stats` e descartado na leitura, antes de gravar, e nunca entra em triagem nem particao.

Etapas (cada uma grava e nao reescreve):
  catalogo     lista as ativacoes e baixa o detalhe das de inundacao/tempestade (sem `stats`)
  triar        exclusoes declaradas, estrato de processo pelo catalogo, qualidade pelos produtos
  unidades     grafo de dependencia do MT01 (evento, lugar L8, onda MAIN_BAS 30 d, chuva 500 km/15 d)
               sobre a populacao do MT01 + AOIs novas; ativacao ligada a evento ja examinado sai
  particionar  por estrato, ordem por sha256(semente|unidade); fracao de reserva fixada antes de olhar
               qualquer pacote, dimensionada no proprio registro

Exclusoes (contratos de outras frentes e deste projeto):
  EMSR868, EMSR871         reserva bruta do OBS03: nem detalhe e lido
  EMSR926, EMSR927, EMSR933 reserva de confirmacao do MT02
  EMSR858, EMSR866          arvore heldout do GEOID-Flood ainda nao usada (frente SAR)
  EMSR692, EMSR698, EMSR728, EMSR773  eventos em uso ou planejados pela frente SAR
  ativacoes ja no MT01/MT02 (34)

Uso:
    python scripts/metodo/mt03_triagem_cems.py catalogo
    python scripts/metodo/mt03_triagem_cems.py triar
    python scripts/metodo/mt03_triagem_cems.py unidades
    python scripts/metodo/mt03_triagem_cems.py particionar
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "geo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402
import mt01_independencia as ind  # noqa: E402
import mt01_populacao as mp  # noqa: E402

SAIDA = mp.RUNS / "mecanismos-transferiveis" / "v3" / "triagem_cems"
META = SAIDA / "metadados"
LISTA = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations-info/"
DETALHE = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/"
UA = {"User-Agent": "REV-P-pesquisa/1.0 (uso academico; triagem por metadados)"}
CATEGORIAS = {"Flood", "Storm"}
NUNCA_LER = {"EMSR868", "EMSR871"}
RESERVA_MT02 = {"EMSR926", "EMSR927", "EMSR933"}
FRENTE_SAR = {"EMSR858", "EMSR866", "EMSR692", "EMSR698", "EMSR728", "EMSR773"}
SEMENTE = "mt03-particao-v1"
PALAVRAS_RUPTURA = re.compile(r"glacial|glof|dam (break|failure|collapse|burst)|breach|outburst|lahar|levee fail",
                              re.I)


def _get(url: str) -> dict:
    for t in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return json.loads(r.read())
        except OSError:
            if t == 3:
                raise
            time.sleep(2 ** t)


def _sem_stats(o):
    if isinstance(o, dict):
        return {k: _sem_stats(v) for k, v in o.items() if k != "stats"}
    if isinstance(o, list):
        return [_sem_stats(v) for v in o]
    return o


def catalogo() -> dict:
    META.mkdir(parents=True, exist_ok=True)
    itens, url = [], LISTA + "?limit=100"
    while url:
        d = _get(url)
        itens += d["results"]
        url = d.get("next")
    (SAIDA / "lista.json").write_text(json.dumps({"quando_utc": agora_utc(), "itens": itens}, ensure_ascii=False,
                                                 indent=1), encoding="utf-8")
    baixados = 0
    for a in itens:
        c = a["code"]
        if a.get("category") not in CATEGORIAS or c in NUNCA_LER | RESERVA_MT02:
            continue
        arq = META / f"{c}.json"
        if arq.exists():
            continue
        bruto = _get(DETALHE + "?" + urllib.parse.urlencode({"code": c}))
        if len(bruto.get("results", [])) != 1:
            continue
        r = {"code": c, "url": DETALHE + "?code=" + c, "quando_utc": agora_utc(),
             "hash_resposta_original": sha256_json(bruto),
             "nota": "campo stats (estatistica de area afetada = informacao de rotulo) descartado antes de gravar",
             "ativacao": _sem_stats(bruto["results"][0])}
        arq.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        baixados += 1
    return {"ativacoes_listadas": len(itens), "detalhes_novos": baixados,
            "por_categoria": pd.Series([a.get("category") for a in itens]).value_counts().to_dict()}


def _processo(a: dict) -> str:
    texto = " ".join(str(a.get(k) or "") for k in ("name", "reason", "subCategory"))
    if PALAVRAS_RUPTURA.search(texto):
        return "ruptura_glof"
    sub = (a.get("subCategory") or "").lower()
    if a.get("category") == "Storm":
        return "tempestade_mista"
    if "flash" in sub:
        return "enxurrada"
    if "riverine" in sub:
        return "fluvial"
    if "coastal" in sub or "surge" in sub:
        return "costeira"
    return "inundacao_sem_subcategoria"


def _produtos(a: dict) -> list[dict]:
    out = []
    for aoi in a.get("aois", []):
        for p in aoi.get("products", []):
            camadas = [x["name"] for x in p.get("layers", [])]
            out.append({"aoi": aoi["number"], "aoi_nome": aoi.get("name"), "aoi_extent": aoi.get("extent"),
                        "tipo": p.get("type"), "monitoramento": p.get("monitoringNumber", 0),
                        "versao": (p.get("version") or {}).get("number"),
                        "status": (p.get("version") or {}).get("statusCode"),
                        "url": p.get("downloadPath"),
                        "tem_observado": any("observedEvent" in n for n in camadas),
                        "tem_modelado": any("modelledEvent" in n for n in camadas),
                        "imagens": [{"sensor": i.get("sensorType"), "nome": i.get("sensorName"),
                                     "resolucao": i.get("resolutionClass"), "quando": i.get("acquisitionTime")}
                                    for i in p.get("images", [])]})
    return out


def triar() -> dict:
    ja = set(json.loads((mp.RUNS / "reservas-examinadas" / "conhecidos_2026_10_03.json").read_text(
        encoding="utf-8")).get("por_fonte", {}).get("cems", []))
    ja = {x.split(":")[-1] for x in ja} | {p.name.split("_")[0] for p in mp.CEMS_ZIPS.glob("*_products.zip")}
    pop = pd.read_parquet(mp.SAIDA / "populacao.parquet", columns=["fonte", "evento_id"])
    ja |= set(re.findall(r"EMSR\d+", " ".join(pop.loc[pop.fonte == "cems", "evento_id"].astype(str).unique())))
    linhas = []
    for arq in sorted(META.glob("EMSR*.json")):
        a = json.loads(arq.read_text(encoding="utf-8"))["ativacao"]
        c = a["code"]
        prods = _produtos(a)
        dels = [p for p in prods if p["tipo"] == "DEL" and p["tem_observado"] and p["url"]]
        sensores = sorted({i["sensor"] for p in dels for i in p["imagens"] if i["sensor"]})
        resol = sorted({i["resolucao"] for p in dels for i in p["imagens"] if i["resolucao"]})
        motivo = ("reserva_sar_geoid" if c in FRENTE_SAR else "ja_examinado_no_mt" if c in ja
                  else "sem_delineacao_observada" if not dels
                  else "ativacao_aberta" if not a.get("closed") else "")
        linhas.append({"code": c, "nome": a.get("name"), "categoria": a.get("category"),
                       "subcategoria": a.get("subCategory"), "processo_catalogo": _processo(a),
                       "evento": a.get("eventTime"), "paises": ",".join(x["name"] for x in a.get("countries", [])),
                       "continente": a.get("continent"), "fechada": bool(a.get("closed")),
                       "aois_com_delineacao": len({p["aoi"] for p in dels}), "produtos_del": len(dels),
                       "monitoramentos": sum(p["monitoramento"] > 0 for p in dels),
                       "sensores": ",".join(sensores), "resolucoes": ",".join(resol),
                       "ultima_imagem": max((i["quando"] for p in dels for i in p["imagens"] if i["quando"]),
                                            default=None),
                       "motivo_exclusao": motivo})
    t = pd.DataFrame(linhas)
    t.to_csv(SAIDA / "triagem.csv", index=False)
    elegiveis = t[t.motivo_exclusao == ""]
    return {"detalhados": len(t), "exclusoes": t.motivo_exclusao.replace("", "elegivel").value_counts().to_dict(),
            "elegiveis_por_processo": elegiveis.processo_catalogo.value_counts().to_dict(),
            "elegiveis_por_continente": elegiveis.continente.value_counts().to_dict()}


def _pontos_da_aoi(wkt_txt: str, n: int = 6) -> np.ndarray:
    from shapely import wkt
    from shapely.geometry import Point
    g = wkt.loads(wkt_txt)
    x0, y0, x1, y1 = g.bounds
    xs, ys = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    pts = [(x, y) for x, y in zip(xs.ravel(), ys.ravel()) if g.contains(Point(x, y))]
    c = g.representative_point()
    return np.array(pts + [(c.x, c.y)])


def unidades() -> dict:
    t = pd.read_csv(SAIDA / "triagem.csv")
    el = t[t.motivo_exclusao.isna()].code.tolist()
    aois = []
    for c in el:
        a = json.loads((META / f"{c}.json").read_text(encoding="utf-8"))["ativacao"]
        prods = [p for p in _produtos(a) if p["tipo"] == "DEL" and p["tem_observado"] and p["url"]]
        fim = max((i["quando"] for p in prods for i in p["imagens"] if i["quando"]), default=a["eventTime"])
        for aoi in {p["aoi"]: p for p in prods}.values():
            pts = _pontos_da_aoi(aoi["aoi_extent"])
            for lon, lat in pts:
                aois.append({"evento_id": f"CEMS_NOVO:{c}", "unidade_espacial": f"{c}_AOI{aoi['aoi']:02d}",
                             "janela_inicio": a["eventTime"][:10], "janela_fim": str(fim)[:10],
                             "lon": lon, "lat": lat, "fonte": "cems_novo", "classe": 0})
    novos = pd.DataFrame(aois)
    b, hb_hash = mp.localizar_bacias(novos.lon.to_numpy(), novos.lat.to_numpy())
    novos = pd.concat([novos.reset_index(drop=True), b[["MAIN_BAS", "HYBAS_L8"]]], axis=1)
    pop = pd.read_parquet(mp.SAIDA / "populacao.parquet",
                          columns=["evento_id", "fonte", "classe", "janela_inicio", "janela_fim", "lat", "lon",
                                   "unidade_espacial", "HYBAS_L8", "MAIN_BAS", "papel"])
    pop = pop[pop.fonte.isin(mp.FONTES_CONTRASTE) & pop.papel.isin(["positivo_episodico", "negativo_observado"])]
    juntos = pd.concat([pop.drop(columns="papel"), novos], ignore_index=True)
    for c in ("HYBAS_L8", "MAIN_BAS"):
        juntos[c] = juntos[c].fillna(0).astype("int64")
    comp, lig = ind.grafo(juntos, "primario")
    novos_ev = sorted(novos.evento_id.unique())
    antigos_por_comp = pd.Series(comp).loc[sorted(set(pop.evento_id.dropna()))].groupby(lambda e: comp[e]).size()
    linhas = []
    for e in novos_ev:
        r = comp[e]
        linhas.append({"code": e.split(":")[1], "componente": r,
                       "ligada_a_examinado": bool(r in antigos_por_comp.index),
                       "sem_bacia": bool((novos.loc[novos.evento_id == e, "MAIN_BAS"].fillna(0) <= 0).all())})
    u = pd.DataFrame(linhas)
    livres = u[~u.ligada_a_examinado]
    nomes = {r: "U3_" + hashlib.sha256(("|".join(sorted(g.code))).encode()).hexdigest()[:10]
             for r, g in livres.groupby("componente")}
    u["unidade_nova"] = u.componente.map(nomes)
    u.to_csv(SAIDA / "unidades_novas.csv", index=False)
    novos.to_parquet(SAIDA / "aois_amostradas.parquet", index=False)
    return {"ativacoes_elegiveis": len(u), "ligadas_a_examinado": int(u.ligada_a_examinado.sum()),
            "unidades_novas": int(livres.componente.nunique()), "ligacoes": lig, "hydrobasins": hb_hash,
            "sem_bacia": int(u.sem_bacia.sum())}


def relevo() -> dict:
    """Declive mediano (MERIT DEM, 90 m) dentro das AOIs com delineacao: sem rotulo, so para estratificar."""
    import ee
    from shapely import wkt
    ee.Initialize()
    dem = ee.Image("MERIT/DEM/v1_0_3")
    declive = ee.Terrain.slope(dem)
    u = pd.read_csv(SAIDA / "unidades_novas.csv")
    linhas = []
    for c in u.code:
        a = json.loads((META / f"{c}.json").read_text(encoding="utf-8"))["ativacao"]
        prods = [p for p in _produtos(a) if p["tipo"] == "DEL" and p["tem_observado"] and p["url"]]
        for aoi in {p["aoi"]: p for p in prods}.values():
            g = wkt.loads(aoi["aoi_extent"]).simplify(0.002)
            geo = ee.Geometry(g.__geo_interface__)
            v = declive.reduceRegion(ee.Reducer.median(), geo, 90, maxPixels=1e9, bestEffort=True).getInfo()
            linhas.append({"code": c, "aoi": aoi["aoi"], "declive_mediano_graus": v.get("slope"),
                           "area_km2": g.area * 111.32 ** 2 * np.cos(np.radians(g.centroid.y))})
    r = pd.DataFrame(linhas)
    r.to_csv(SAIDA / "relevo_aois.csv", index=False)
    return {"aois": len(r), "declive_mediano": r.declive_mediano_graus.describe().round(2).to_dict()}


def poder(dp: float, efeito: float, n: int, reps: int = 4000, semente: int = 5) -> float:
    """Poder do criterio do projeto (IC 95% por bootstrap de unidades acima de zero), deltas ~ N(efeito, dp)."""
    rng = np.random.default_rng(semente)
    ok = 0
    for _ in range(reps // 20):
        v = rng.normal(efeito, dp, n)
        bs = rng.choice(v, (400, n)).mean(1)
        ok += np.percentile(bs, 2.5) > 0
    return ok / (reps // 20)


def dimensionar() -> dict:
    """Desvio entre unidades de cada contraste, medido no MT02 (desenvolvimento), e n por efeito relevante."""
    import mt01_relatorio as rel
    ok = [r for r in rel.carregar_destinos("../v2/destinos") if r["estado"] == "executado"]
    m = rel.medidas(ok)
    u = rel.por_unidade(m[m.recorte == "todos"].drop(columns="recorte"))
    proc = {k: rel.por_unidade(g.drop(columns="recorte")) for k, g in m[m.recorte.str.startswith("processo:")].groupby("recorte")}
    hip = {}

    def linha(nome, deltas, relevante):
        deltas = np.asarray(deltas, float)
        deltas = deltas[np.isfinite(deltas)]
        dp, ef = float(deltas.std(ddof=1)), float(deltas.mean())
        ns = {}
        for alvo, e in (("efeito_observado", ef), ("efeito_relevante", relevante)):
            ns[alvo] = {str(n): round(poder(dp, e, n), 2) for n in (6, 8, 12, 16, 24, 32, 48)}
        hip[nome] = {"unidades_no_mt02": int(len(deltas)), "delta_medio": round(ef, 4), "dp_entre_unidades": round(dp, 4),
                     "efeito_relevante": relevante, "poder_por_n": ns}

    linha("H1_G_v2_menos_regra_hand", u["G_v2"] - u["A0_regra_hand"], .02)
    for nome, chave, relevante in (("H_conexao_fluvial", "processo:fluvial", .01),
                             ("H_conexao_enxurrada", "processo:enxurrada", .02)):
        pu = proc.get(chave)
        if pu is not None:
            linha(nome, pu["G_conexao"] - pu["G_v2"], relevante)
    return hip


# Processo proprio pelo TEXTO do catalogo (lido antes de qualquer pacote): lago carstico que transborda por
# agua subterranea (turlough, EMSR718) e cheia por operacao de reservatorio (EMSR794). Nao sao enxurrada nem
# transbordamento fluvial comum; vao inteiros para a reserva, em estrato proprio, como o GLOF do MT02.
PROCESSO_PROPRIO = {"EMSR718": "lago_carstico_agua_subterranea", "EMSR794": "operacao_de_reservatorio"}


def particionar() -> dict:
    arq = SAIDA / "particao.json"
    if arq.exists():
        raise FileExistsError(arq)
    t = pd.read_csv(SAIDA / "triagem.csv")
    u = pd.read_csv(SAIDA / "unidades_novas.csv")
    r = pd.read_csv(SAIDA / "relevo_aois.csv")
    u = u[u.unidade_nova.notna()].merge(t[["code", "categoria", "continente", "evento"]], on="code")
    rel_ativ = r.groupby("code").apply(lambda g: np.average(g.declive_mediano_graus, weights=g.area_km2))
    u["declive"] = u.code.map(rel_ativ)
    est = {}
    for un, g in u.groupby("unidade_nova"):
        if g.code.isin(PROCESSO_PROPRIO).any():
            est[un] = "processo_proprio"
        elif (g.categoria == "Storm").all():
            est[un] = "tempestade"
        else:
            est[un] = "inundacao"
    est = pd.Series(est)
    decl = u.groupby("unidade_nova").declive.median()
    corte = float(decl[est == "inundacao"].median())
    est[(est == "inundacao") & (decl >= corte)] = "inundacao_relevo_alto"
    est[est == "inundacao"] = "inundacao_relevo_baixo"
    dim = dimensionar()
    regra = {"processo_proprio": "reserva inteira (estrato proprio, nenhuma hipotese de desenvolvimento usa)",
             "demais": ("dentro do estrato, ordem por sha256(semente|unidade); alternado comecando pela RESERVA "
                        "(reserva recebe a metade arredondada para cima)"),
             "estratos": ("tempestade pela categoria do catalogo; inundacao dividida pela mediana do declive "
                          f"mediano MERIT das AOIs ({corte:.2f} graus) -- so para equilibrar, nao e hipotese"),
             "por_que_metade": ("H1 com o efeito observado pede ~12 unidades para poder 0,9; a conexao na enxurrada "
                                "(efeito relevante 0,02) pede ~16 unidades COM enxurrada, que nenhuma particao das 22 "
                                "alcanca: a enxurrada fica em desenvolvimento conjunto (9 antigas + novas de "
                                "desenvolvimento) e a reserva protege H1 e a conexao fluvial")}
    linhas = []
    for estrato, g in est.groupby(est):
        ordem = sorted(g.index, key=lambda k: hashlib.sha256(f"{SEMENTE}|{k}".encode()).hexdigest())
        for k, un in enumerate(ordem):
            papel = "reserva" if estrato == "processo_proprio" or k % 2 == 0 else "desenvolvimento"
            linhas.append({"unidade_nova": un, "estrato": estrato, "papel": papel,
                           "declive_mediano": round(float(decl[un]), 3),
                           "ativacoes": sorted(u.loc[u.unidade_nova == un, "code"])})
    p = pd.DataFrame(linhas)
    out = {"tipo": "mt03_particao_cems", "quando_utc": agora_utc(), "semente": SEMENTE,
           "lido_antes": "so metadados do catalogo (sem stats) e declive MERIT; nenhum pacote, poligono ou pixel CEMS",
           "regra": regra, "processo_proprio": PROCESSO_PROPRIO, "dimensionamento": dim,
           "contagem": {str(k): v for k, v in p.groupby(["estrato", "papel"]).size().items()},
           "unidades": linhas,
           "contrato_da_reserva": ("nenhum pacote da reserva e baixado neste ciclo; abrir so com candidata e regra de "
                                   "decisao congeladas e n suficiente pelo dimensionamento; nada da reserva ajusta modelo"),
           "fontes": {str(f.relative_to(REPO)): sha256_arquivo(f) for f in
                      (SAIDA / "triagem.csv", SAIDA / "unidades_novas.csv", SAIDA / "relevo_aois.csv", Path(__file__))}}
    out["hash_resumo"] = sha256_json({k: v for k, v in out.items() if k != "quando_utc"})
    arq.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return {k: out[k] for k in ("contagem", "hash_resumo")} | {"unidades": [(l["unidade_nova"], l["estrato"], l["papel"], l["ativacoes"]) for l in linhas]}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=("catalogo", "triar", "unidades", "relevo", "dimensionar", "particionar"))
    a = ap.parse_args()
    r = {"catalogo": catalogo, "triar": triar, "unidades": unidades, "relevo": relevo, "dimensionar": dimensionar,
         "particionar": particionar}[a.acao]()
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
