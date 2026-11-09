"""MT03, passo 4: as unidades CEMS de DESENVOLVIMENTO da particao viram pontos com terreno -- e so elas.

Cada ativacao da particao `desenvolvimento` (triagem_cems/particao.json) passa por:
  baixar   por AOI, o produto DEL com observedEventA da PRIMEIRA observacao (DEL_PRODUCT; monitoramento so
           se nao houver), zip por AOI do Rapid Mapping. Codigo da reserva, protegido ou da frente SAR aborta.
  areal    observado = AOI INTERSECAO footprint; positivo = observedEvent 5-Flood; negativo = observado menos
           positivo menos movimento de massa (contrato do cems03, mesmas funcoes). obj_desc de cada poligono
           de cheia guardado para o processo no ponto.
  terreno  FABDEM na caixa da area observada + 0,1 grau de margem de bacia, cadeia canonica (30 m, 0,1123 km2)
           pelo `derivar_aoi_nova`, e mascara JRC >= 50 alinhada. Tudo em diretorio PROPRIO (nao no ter-01,
           nao no cems-05-pacotes: outras cadeias varrem esses diretorios e ingeririam estas ativacoes).
  pontos   grade de 150 m (cems03), terreno amostrado, agua permanente fora das duas classes, processo por
           obj_desc, piso local por AOI (quantil 0,01, minimo 30 pontos, regra do ds06), ln_sca do TWI.

Rotulos LIDOS aqui sao so os de desenvolvimento. Nenhuma linha vai para tabela unica, ds05 ou ds06.

Uso:
    python scripts/metodo/mt03_cems_desenvolvimento.py baixar
    python scripts/metodo/mt03_cems_desenvolvimento.py areal
    python scripts/metodo/mt03_cems_desenvolvimento.py terreno     (pesado: rede + WhiteboxTools; sob a trava)
    python scripts/metodo/mt03_cems_desenvolvimento.py pontos
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "geo", "terreno", "sar", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402
import mt01_populacao as mp  # noqa: E402
import mt03_triagem_cems as tri  # noqa: E402

V3 = mp.RUNS / "mecanismos-transferiveis" / "v3"
CD = V3 / "cems_desenvolvimento"
PACOTES, AREAL, MDT, TERRENO = CD / "pacotes", CD / "areal", CD / "mdt", CD / "terreno"
MAX_BYTES = 400 * 1024 ** 2
MARGEM_GRAUS = 0.1
QUANTIL_DO_PISO, MINIMO_PISO = 0.01, 30
CEMS_PROC = {"Riverine flood": "fluvial", "Flash flood": "enxurrada", "Coastal flood": "costeira"}


def particao() -> tuple[dict, set]:
    p = json.loads((tri.SAIDA / "particao.json").read_text(encoding="utf-8"))
    dev, reserva = {}, set()
    for u in p["unidades"]:
        for c in u["ativacoes"]:
            (dev.__setitem__(c, u["unidade_nova"]) if u["papel"] == "desenvolvimento" else reserva.add(c))
    proibidos = reserva | tri.NUNCA_LER | tri.RESERVA_MT02 | tri.FRENTE_SAR
    if set(dev) & proibidos:
        raise RuntimeError(f"desenvolvimento cruza reserva/protegido: {set(dev) & proibidos}")
    return dev, proibidos


def produtos_escolhidos(code: str) -> list[dict]:
    a = json.loads((tri.META / f"{code}.json").read_text(encoding="utf-8"))["ativacao"]
    prods = [p for p in tri._produtos(a) if p["tipo"] == "DEL" and p["tem_observado"] and p["url"]]
    out = []
    for aoi in sorted({p["aoi"] for p in prods}):
        cand = [p for p in prods if p["aoi"] == aoi]
        base = [p for p in cand if not p["monitoramento"]] or cand
        primeira = min(base, key=lambda p: (min((i["quando"] or "9999") for i in p["imagens"]) if p["imagens"]
                                            else "9999", p["monitoramento"], p["url"]))
        out.append({**primeira, "code": code, "evento": a["eventTime"]})
    return out


def baixar() -> dict:
    dev, proibidos = particao()
    feitos = []
    for code in sorted(dev):
        if code in proibidos:
            raise RuntimeError(code)
        for p in produtos_escolhidos(code):
            if urllib.parse.urlparse(p["url"]).hostname != "rapidmapping.emergency.copernicus.eu":
                raise ValueError(f"produto fora da origem: {p['url']}")
            destino = PACOTES / code / Path(urllib.parse.urlparse(p["url"]).path).name
            destino.parent.mkdir(parents=True, exist_ok=True)
            if not destino.exists():
                req = urllib.request.Request(p["url"], headers=tri.UA | {"Range": "bytes=0-0"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    total = int(r.headers["Content-Range"].split("/")[-1])
                if total > MAX_BYTES:
                    feitos.append({"code": code, "aoi": p["aoi"], "estado": "acima_do_teto", "bytes": total})
                    continue
                tmp = destino.with_suffix(".parcial")
                with urllib.request.urlopen(urllib.request.Request(p["url"], headers=tri.UA), timeout=600) as r, \
                        open(tmp, "wb") as f:
                    while b := r.read(1 << 20):
                        f.write(b)
                if tmp.stat().st_size != total:
                    raise RuntimeError(f"{destino.name}: {tmp.stat().st_size} de {total}")
                tmp.replace(destino)
            feitos.append({"code": code, "unidade_nova": dev[code], "aoi": p["aoi"], "aoi_nome": p["aoi_nome"],
                           "produto": destino.name, "url": p["url"], "bytes": destino.stat().st_size,
                           "sha256": sha256_arquivo(destino), "imagens": p["imagens"], "evento": p["evento"],
                           "estado": "baixado"})
            print(code, p["aoi"], destino.name, feitos[-1]["bytes"], flush=True)
    m = {"tipo": "mt03_cems_dev_pacotes", "quando_utc": agora_utc(), "pacotes": feitos,
         "particao": json.loads((tri.SAIDA / "particao.json").read_text(encoding="utf-8"))["hash_resumo"],
         "licenca": "Copernicus EMS: uso livre com atribuicao (European Union)"}
    (CD / "pacotes.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"pacotes": len(feitos), "mb": round(sum(f.get("bytes", 0) for f in feitos) / 1e6, 1),
            "acima_do_teto": sum(f["estado"] == "acima_do_teto" for f in feitos)}


def _gravar_geo(arq: Path, geo: list[dict], epsg: int) -> None:
    from shapely import to_wkb
    pd.DataFrame({"papel": [g["papel"] for g in geo], "wkb": [to_wkb(g["geometry"]) for g in geo],
                  "epsg": epsg}).to_parquet(arq, index=False)


def _ler_geo(arq: Path, epsg_destino: int | None = None) -> tuple[pd.DataFrame, int]:
    from pyproj import Transformer
    from shapely import from_wkb
    from shapely.ops import transform
    t = pd.read_parquet(arq)
    epsg = int(t.epsg.iloc[0])
    geoms = [from_wkb(b) for b in t.wkb]
    if epsg_destino and epsg_destino != epsg:
        f = Transformer.from_crs(epsg, epsg_destino, always_xy=True).transform
        geoms = [transform(f, g) for g in geoms]
        epsg = epsg_destino
    return pd.DataFrame({"papel": t.papel, "geometry": geoms}), epsg


def areal() -> dict:
    import cems01_pacote_areal as c1
    import cems03_amostrar_pontos as c3
    m = json.loads((CD / "pacotes.json").read_text(encoding="utf-8"))
    AREAL.mkdir(parents=True, exist_ok=True)
    linhas = []
    for p in m["pacotes"]:
        if p["estado"] != "baixado":
            continue
        aoi_id = f"AOI{p['aoi']:02d}"
        nome = f"{p['code']}_{aoi_id}"
        arq = AREAL / f"{nome}.parquet"
        if arq.exists():
            linhas.append(json.loads(arq.with_suffix(".json").read_text(encoding="utf-8")))
            continue
        with tempfile.TemporaryDirectory(prefix="mt03_cems_") as tmp:
            pasta = Path(tmp) / Path(p["produto"]).stem
            with zipfile.ZipFile(PACOTES / p["code"] / p["produto"]) as z:
                z.extractall(pasta)
            for interno in list(pasta.rglob("*.zip")):
                with zipfile.ZipFile(interno) as z:
                    z.extractall(interno.with_suffix(""))
            # centro da AOI pelo catalogo: CRS metrico fixo pela regra do cems01
            from shapely import wkt
            prod = next(x for x in produtos_escolhidos(p["code"]) if x["aoi"] == p["aoi"])
            c = wkt.loads(prod["aoi_extent"]).centroid
            epsg = c1.crs_metrico_de(c.x, c.y)
            cam = c3.camadas_da_aoi(Path(tmp), aoi_id, epsg)
            cls = c3.classes_da_aoi(cam)
            data = c3.data_do_produto(pasta) if hasattr(c3, "data_do_produto") else None
            feic, ileg = c1.ler_camada(Path(tmp), c1.CAMADA_EVENTO, epsg)
        cheias = [(g, str(pr.get("obj_desc"))) for g, pr in feic
                  if str(pr.get("event_type", "")).startswith(c1.TIPO_INUNDACAO)]
        geo = [{"papel": k, "geometry": cls[k]} for k in ("observado", "positivo", "negativo") if cls[k] is not None]
        geo += [{"papel": "massa", "geometry": cam["massa"]}] if cam["massa"] is not None else []
        geo += [{"papel": f"cheia|{d}", "geometry": g} for g, d in cheias]
        _gravar_geo(arq, geo, epsg)
        area = lambda g: (g.area / 1e6) if g is not None else 0.0  # noqa: E731
        r = {"nome": nome, "code": p["code"], "aoi": aoi_id, "unidade_nova": p["unidade_nova"], "epsg": epsg,
             "data_produto": data, "evento": p["evento"], "area_observada_km2": area(cls["observado"]),
             "area_positiva_km2": area(cls["positivo"]), "area_negativa_km2": area(cls["negativo"]),
             "poligonos_cheia_por_obj_desc": pd.Series([d for _, d in cheias]).value_counts().to_dict(),
             "ilegiveis": len(cam["ilegiveis"]) + len(ileg), "produtos_usados": cam["produtos_usados"],
             "imagens": p["imagens"]}
        arq.with_suffix(".json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        linhas.append(r)
        print(nome, round(r["area_observada_km2"], 1), round(r["area_positiva_km2"], 1),
              r["poligonos_cheia_por_obj_desc"], flush=True)
    t = pd.DataFrame(linhas)
    t.drop(columns=["imagens", "produtos_usados"]).to_csv(CD / "areal.csv", index=False)
    return {"aois": len(t), "com_positivo": int((t.area_positiva_km2 > 0).sum()),
            "obj_desc": pd.concat([pd.Series(x) for x in t.poligonos_cheia_por_obj_desc if x]).groupby(level=0).sum().to_dict()}


def _caixa_4326(arq: Path) -> list[float]:
    g, _ = _ler_geo(arq, 4326)
    o = g.loc[g.papel == "observado", "geometry"].iloc[0].bounds
    return [float(o[0]) - MARGEM_GRAUS, float(o[1]) - MARGEM_GRAUS, float(o[2]) + MARGEM_GRAUS, float(o[3]) + MARGEM_GRAUS]


def terreno(maximo: int | None = None) -> dict:
    import agua_permanente_jrc as jrc
    import baixar_fabdem as fab
    import derivar_aoi_nova as der
    t = pd.read_csv(CD / "areal.csv")
    t = t[t.area_positiva_km2 > 0]
    feitos, falhas = [], []
    for r in t.itertuples():
        destino = TERRENO / r.nome
        if (destino / "run_manifest.json").exists() and (destino / jrc_nome()).exists():
            continue
        if maximo is not None and len(feitos) + len(falhas) >= maximo:
            break
        t0 = time.time()
        try:
            caixa = _caixa_4326(AREAL / f"{r.nome}.parquet")
            plano = fab.planejar(*caixa, r.nome)
            b = fab.baixar(plano, saida=MDT)
            man = der.derivar(Path(b["arquivo"]), r.nome, raiz=TERRENO, epsg_metrico=int(r.epsg))
            jrc.mascara(destino / "dem_filled.tif", destino / jrc_nome())
            feitos.append({"nome": r.nome, "caixa": caixa, "shape": man.get("shape"),
                           "segundos": round(time.time() - t0, 1)})
            print("[ok]", r.nome, man.get("shape"), feitos[-1]["segundos"], flush=True)
        except (Exception, SystemExit) as e:  # noqa: BLE001 -- falha de uma AOI nao para o lote
            falhas.append({"nome": r.nome, "erro": f"{type(e).__name__}: {e}"[:300]})
            print("[FALHA]", r.nome, falhas[-1]["erro"], flush=True)
    log = CD / "terreno_execucoes.jsonl"
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"quando_utc": agora_utc(), "feitos": feitos, "falhas": falhas}, ensure_ascii=False) + "\n")
    return {"feitos": len(feitos), "falhas": falhas}


def jrc_nome() -> str:
    return "agua_permanente_jrc.tif"


def pontos() -> dict:
    import cems03_amostrar_pontos as c3
    from pyproj import Transformer
    from shapely import STRtree, points
    t = pd.read_csv(CD / "areal.csv")
    partes, sem = [], []
    for r in t.itertuples():
        destino = TERRENO / r.nome
        if not (destino / "run_manifest.json").exists():
            sem.append(r.nome)
            continue
        epsg_t = c3._crs_do_terreno(destino)
        g, _ = _ler_geo(AREAL / f"{r.nome}.parquet", epsg_t)
        geo = {k: (g.loc[g.papel == k, "geometry"].iloc[0] if (g.papel == k).any() else None)
               for k in ("positivo", "negativo")}
        linhas = []
        for classe, chave in ((1, "positivo"), (0, "negativo")):
            coords = c3.pontos_na_geometria(geo[chave], destino, c3.PASSO_M)
            if not coords:
                continue
            v = c3.amostrar_terreno(coords, destino)
            xy = np.array(coords)
            linhas.append(pd.DataFrame({"x_m": xy[:, 0], "y_m": xy[:, 1], "classe": classe, **v}))
        if not linhas:
            sem.append(r.nome)
            continue
        d = pd.concat(linhas, ignore_index=True)
        lon, lat = Transformer.from_crs(epsg_t, 4326, always_xy=True).transform(d.x_m.to_numpy(), d.y_m.to_numpy())
        d["lon"], d["lat"] = lon, lat
        cheias = g[g.papel.str.startswith("cheia|")]
        d["processo_ponto"] = "nao_aplicavel"
        if len(cheias) and (d.classe == 1).any():
            arv = STRtree(list(cheias.geometry))
            pos = d.index[d.classe == 1]
            idx, dist = arv.query_nearest(points(d.loc[pos, "x_m"], d.loc[pos, "y_m"]), return_distance=True,
                                          all_matches=False)
            desc = cheias.papel.str.split("|").str[1].to_numpy()[idx[1]]
            d.loc[pos[idx[0]], "processo_ponto"] = [CEMS_PROC.get(x, "outro:" + x) for x in desc]
        d["code"], d["aoi"], d["nome"] = r.code, r.aoi, r.nome
        d["unidade_nova"] = r.unidade_nova
        partes.append(d)
    d = pd.concat(partes, ignore_index=True)
    d = d[np.isfinite(d[["elevation_m", "hand_m", "slope_deg", "twi_dinf"]]).all(axis=1)]
    n_agua = int((d.agua_permanente_jrc == 1).sum())
    d = d[d.agua_permanente_jrc == 0].copy()
    piso = d.groupby("nome").elevation_m.transform(lambda s: s.quantile(QUANTIL_DO_PISO))
    n = d.groupby("nome").elevation_m.transform("size")
    d["elevation_local_m"] = (d.elevation_m - piso).where(n >= MINIMO_PISO)
    d["ln_sca"] = d.twi_dinf + np.log(np.tan(np.radians(d.slope_deg.clip(lower=0.01))))
    d["fonte"] = "cems"
    d["evento_id"] = "CEMS_NOVO:" + d.code
    d["unidade"] = d.unidade_nova
    d["unidade_espacial"] = d.nome
    d["unidade_terreno"] = "mt03dev/" + d.nome
    d["papel"] = np.where(d.classe == 1, "positivo_episodico", "negativo_observado")
    d = d.sort_values(["nome", "classe", "y_m", "x_m"]).reset_index(drop=True)
    d["ponto_id"] = "MT03DEV_" + d.nome + "_" + d.groupby("nome").cumcount().astype(str).str.zfill(7)
    arq = CD / "pontos.parquet"
    d.to_parquet(arq, index=False)
    m = {"tipo": "mt03_cems_dev_pontos", "quando_utc": agora_utc(), "pontos": int(len(d)),
         "por_unidade_classe": {f"{k[0]}|{k[1]}": int(v) for k, v in d.groupby(["unidade", "classe"]).size().items()},
         "processo_nos_positivos": d[d.classe == 1].processo_ponto.value_counts().to_dict(),
         "agua_permanente_excluida": n_agua, "aois_sem_terreno_ou_ponto": sem,
         "saida": sha256_arquivo(arq), "codigo": sha256_arquivo(Path(__file__))}
    (CD / "pontos.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return {k: m[k] for k in ("pontos", "processo_nos_positivos", "agua_permanente_excluida", "aois_sem_terreno_ou_ponto")}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=("baixar", "areal", "terreno", "pontos"))
    ap.add_argument("--maximo", type=int, default=None)
    a = ap.parse_args()
    r = terreno(a.maximo) if a.acao == "terreno" else {"baixar": baixar, "areal": areal, "pontos": pontos}[a.acao]()
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
