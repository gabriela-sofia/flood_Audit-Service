"""OBS01: buscar referencias observacionais sem alterar a base de treino.

O piloto audita o instrumento, nao mede modelos. GFM usa terreno no proprio
algoritmo: trocar MODIS por SAR nao cria verdade independente de HAND/declive.
Catalogar, baixar e admitir sao etapas distintas. Nenhuma admissao ocorre aqui.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/servico"))
from proveniencia import sha256_arquivo, sha256_json  # noqa: E402

STAC = "https://stac.eodc.eu/api/v1"
CEMS = "https://mapping.emergency.copernicus.eu/activations/api/activations/"
ZENODO = "https://zenodo.org/api/records/21430962"
MANUAL = "https://extwiki.eodc.eu/gfm_assets/gfm4.0_pum_2025.pdf"
CAMADAS = ("ensemble_flood_extent", "ensemble_water_extent", "exclusion_mask",
           "advisory_flags", "reference_water_mask")
REFERENCIAS = (
    "local_runs/br-preparo-sem-rotulo/eventos.csv",
    "local_runs/reservas-examinadas/conhecidos_2026_10_03.json",
    "local_runs/reservas-examinadas/examinados_apos_congelamento.jsonl",
    "local_runs/verdade-observacional/identidade.json",
)


def agora():
    return datetime.now(timezone.utc).isoformat()


def gravar(path, corpo):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    texto = json.dumps(corpo, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != corpo:
            raise ValueError(f"artefato imutavel divergente: {path}")
        return
    with path.open("x", encoding="utf-8") as f:
        f.write(texto)


def json_remoto(url):
    req = urllib.request.Request(url, headers={"User-Agent": "REV-P-observation-audit/1.0"})
    for tentativa in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                bruto = r.read(16 * 1024 * 1024 + 1)
            if len(bruto) > 16 * 1024 * 1024:
                raise ValueError("resposta JSON excedeu o limite")
            d = json.loads(bruto)
            if "error" in d:
                raise ValueError(f"erro da fonte: {d['error']}")
            return d
        except (OSError, TimeoutError):
            if tentativa == 2:
                raise
            time.sleep(2 ** tentativa)


def candidatos(repo=REPO):
    with (repo / REFERENCIAS[0]).open(encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f))
    saida = []
    for x in linhas:
        if x["estado"] != "candidato_novo":
            continue
        bbox = [float(x[k]) for k in ("lon_min", "lat_min", "lon_max", "lat_max")]
        if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
            raise ValueError("caixa geografica invalida")
        saida.append({"eventid": x["eventid"], "inicio": x["inicio"], "fim": x["fim"],
                      "bbox": bbox, "catalogo": x["fonte_catalogo"],
                      "estado": "candidato; mecanismo e identidade ainda exigem auditoria",
                      "territorio_novo": float(x["pontos_treino_nas_bacias_e_vizinhas"]) == 0})
    return sorted(saida, key=lambda x: x["eventid"])


def metadados_item(x):
    p = x["properties"]
    # Whitelist: flooded_pixels, proporcoes, previews e qualidade ligada ao
    # resultado nao entram no catalogo usado para ordenar a aquisicao.
    nomes = ("datetime", "start_datetime", "end_datetime", "Equi7Tile", "constellation",
             "proj:wkt2", "proj:shape", "proj:transform", "processing:version")
    return {"id": x["id"], "bbox": x["bbox"], "geometry": x.get("geometry"),
            "properties": {k: p[k] for k in nomes if k in p},
            "assets": {k: x["assets"][k] for k in CAMADAS if k in x["assets"]}}


def consultar_gfm(e):
    fim = (datetime.fromisoformat(e["fim"]) + timedelta(days=1)).date().isoformat()
    parametros = {"collections": "GFM", "bbox": ",".join(map(str, e["bbox"])),
                  "datetime": f"{e['inicio']}T00:00:00Z/{fim}T00:00:00Z", "limit": 100}
    url = STAC + "/search?" + urllib.parse.urlencode(parametros)
    vistos, itens, paginas = set(), {}, []
    while url:
        if url in vistos or len(vistos) >= 30:
            raise ValueError("paginacao incompleta ou ciclica; nao selecionar subconjunto")
        vistos.add(url)
        bruto = json_remoto(url)
        paginas.append({"url": url, "sha256_resposta": sha256_json(bruto),
                        "obtidos": len(bruto.get("features", []))})
        for x in bruto.get("features", []):
            item = metadados_item(x)
            if item["id"] in itens and itens[item["id"]] != item:
                raise ValueError("item repetido com metadados divergentes")
            itens[item["id"]] = item
        prox = [l for l in bruto.get("links", []) if l.get("rel") == "next"]
        if prox and prox[0].get("method", "GET") != "GET":
            raise ValueError("paginacao POST nao suportada")
        url = prox[0]["href"] if prox else None
        if url and urllib.parse.urlparse(url).hostname != "stac.eodc.eu":
            raise ValueError("paginacao fora da origem")
    return {"evento": e, "paginas": paginas, "itens": sorted(itens.values(), key=lambda x: x["id"]),
            "completo": True, "distribuicao_do_alvo_usada": False}


def escolher(catalogos, n_eventos, n_itens):
    if n_eventos <= 0 or n_itens <= 0:
        raise ValueError("orcamento de itens e eventos deve ser positivo")
    # Piloto de instrumento: limitar caixas grandes limita custo, nao melhora
    # o resultado. A regra nao serve de amostragem representativa do Brasil.
    aptos = [c for c in catalogos if c.get("completo") and c["itens"]
             and c["evento"]["bbox"][2] - c["evento"]["bbox"][0] <= 2
             and c["evento"]["bbox"][3] - c["evento"]["bbox"][1] <= 2]
    aptos.sort(key=lambda c: hashlib.sha256(("obs01-piloto|" + c["evento"]["eventid"]).encode()).hexdigest())
    selecao = []
    for c in aptos[:n_eventos]:
        e = c["evento"]
        meio = datetime.fromisoformat(e["inicio"]).replace(tzinfo=timezone.utc)
        meio += (datetime.fromisoformat(e["fim"]).replace(tzinfo=timezone.utc) - meio) / 2
        itens = sorted(c["itens"], key=lambda x: (
            abs((datetime.fromisoformat(x["properties"]["datetime"].replace("Z", "+00:00")) - meio).total_seconds()),
            x["id"]))
        # Granulos consecutivos da mesma passagem diferem por segundos. No
        # piloto limitado, guardar so um por dia evita chama-los de repeticao.
        datas, unicos = set(), []
        for x in itens:
            data = x["properties"]["datetime"][:10]
            if data not in datas:
                datas.add(data)
                unicos.append(x)
        selecao.append({"evento": e, "itens": unicos[:n_itens]})
    return selecao


def catalogar(saida, n_eventos, n_itens):
    referencias = {p: sha256_arquivo(REPO / p) for p in REFERENCIAS}
    regra = {"tipo": "obs01_piloto_observacional", "versao": 2, "referencias": referencias,
             "codigo": sha256_arquivo(Path(__file__)), "n_eventos": n_eventos, "n_itens": n_itens,
             "selecao": "candidatos BR03; caixas <=2 graus por lado; hash obs01-piloto|eventid; um item por dia UTC, mais proximo do meio da janela",
             "uso": "auditoria do instrumento; nao teste confirmatorio nem treinamento",
             "alvo": "classes publicadas em visadas datadas, nao ausencia durante todo o pico",
             "admissao_automatica": False, "promovido": False,
             "consulta_previa": "1101822: dois metadados STAC com estatisticas do produto consultados na exploracao; nao usar como reserva intacta",
             "independencia_fisica": "GFM usa DEM/HAND; nao e verdade independente desses descritores",
             "limites": ["GDACS/GLOFAS e candidato modelado, nao confirma ocorrencia",
                         "BR03 e triagem anterior, nao identidade ou mecanismo comprovados",
                         "caixas, bacias, eventos e datas ainda devem ser confrontados antes de inferir generalizacao"]}
    pasta = Path(saida) / sha256_json(regra).split(":")[-1]
    gravar(pasta / "regra.json", regra)
    if (pasta / "plano.json").exists():
        return pasta
    catalogos, falhas = [], []
    for e in candidatos():
        alvo = pasta / "catalogos" / f"{e['eventid']}.json"
        try:
            c = json.loads(alvo.read_text(encoding="utf-8")) if alvo.exists() else consultar_gfm(e)
            gravar(alvo, c)
            catalogos.append(c)
            print(f"CATALOGO GDACS:{e['eventid']} itens={len(c['itens'])}", flush=True)
        except (OSError, ValueError, KeyError) as exc:
            falhas.append({"eventid": e["eventid"], "erro": str(exc)})
    colecao = json_remoto(STAC + "/collections/GFM")
    gravar(pasta / "colecao_gfm.json", colecao)
    z = json_remoto(ZENODO)
    gravar(pasta / "zenodo.json", z)
    cems, url, vistos = [], CEMS + "?limit=1500", set()
    while url:
        if url in vistos:
            raise ValueError("catalogo CEMS ciclico")
        vistos.add(url)
        d = json_remoto(url)
        cems.extend(d["results"])
        url = d.get("next")
        if url:
            url = url.replace("http://", "https://", 1)
            if urllib.parse.urlparse(url).hostname != "mapping.emergency.copernicus.eu":
                raise ValueError("CEMS paginou fora da origem")
    gravar(pasta / "catalogo_cems.json", {"ativacoes": cems, "completo": True})
    conhecidos = json.loads((REPO / REFERENCIAS[1]).read_text(encoding="utf-8"))
    ids = {x.split(":", 1)[-1] for v in conhecidos["por_fonte"].values() for x in v}
    produtos_locais = {p.name.split("_", 1)[0] for p in (REPO / "local_runs/cems-05-pacotes").glob("*_products.zip")}
    floods = [x for x in cems if x["category"]["slug"] == "flood"]
    novos = [x for x in floods if x["code"].startswith("EMSR") and x["code"] not in ids | produtos_locais]
    brasileiros = [x for x in floods if "Brazil" in [p["short_name"] for p in x["countries"]]]
    plano = {"tipo": "plano_aquisicao_observacional", "regra_hash": sha256_json(regra),
             "gfm": escolher(catalogos, n_eventos, n_itens), "falhas_catalogo": falhas,
             "gfm_licenca_stac": colecao.get("license"),
             "licenca_gfm": "catalogo proprietary: uso e redistribuicao nao liberados automaticamente",
             "zenodo": {"registro": "21430962", "doi": z["metadata"].get("doi"),
                        "licenca": z["metadata"].get("license"), "arquivos": z["files"],
                        "evento_canonico": "CEMS:EMSR720", "evento_novo": False,
                        "uso": "comparacao de instrumento em evento conhecido; mosaico, sem hora por pixel"},
             "cems": {"total_catalogo": len(cems), "brasil_inundacao": brasileiros,
                      "nao_listados_localmente": novos,
                      "limite": "ID novo nao prova mecanismo, independencia de evento, territorio nem disponibilidade"},
             "referencias": referencias, "rotulos_gfm_lidos": False, "criado_utc": agora(),
             "promovido": False, "admitido_no_treino": False}
    plano["hash"] = sha256_json(plano)
    gravar(pasta / "plano.json", plano)
    return pasta


def baixar(url, destino, limite, checksum=None, tamanho=None):
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    meta = destino.with_suffix(destino.suffix + ".json")
    if destino.exists():
        if not meta.exists():
            raise ValueError("cache sem procedencia")
        r = json.loads(meta.read_text(encoding="utf-8"))
        if r["url"] != url or r["sha256"] != sha256_arquivo(destino) or destino.stat().st_size > limite:
            raise ValueError("cache divergente")
        return r
    parcial = destino.with_suffix(destino.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "REV-P-observation-audit/1.0"})
    lidos, md5 = 0, hashlib.md5()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            prometido = resp.headers.get("Content-Length")
            if prometido and int(prometido) > limite:
                raise ValueError("arquivo excede orcamento por camada")
            with parcial.open("wb") as f:
                while True:
                    b = resp.read(1 << 20)
                    if not b:
                        break
                    lidos += len(b)
                    if lidos > limite:
                        raise ValueError("download excede orcamento")
                    f.write(b)
                    md5.update(b)
        if prometido and lidos != int(prometido) or tamanho is not None and lidos != tamanho:
            raise ValueError("download truncado")
        if checksum is not None and checksum != "md5:" + md5.hexdigest():
            raise ValueError("checksum publicado divergente")
        parcial.replace(destino)
    except Exception:
        parcial.unlink(missing_ok=True)
        raise
    r = {"url": url, "arquivo": str(destino.resolve()), "bytes": lidos,
         "sha256": sha256_arquivo(destino), "md5": "md5:" + md5.hexdigest(), "quando_utc": agora()}
    gravar(meta, r)
    return r


def adquirir(pasta, limite_mb=64, total_mb=512, leve=False):
    plano = json.loads((pasta / "plano.json").read_text(encoding="utf-8"))
    if sha256_json({k: v for k, v in plano.items() if k != "hash"}) != plano["hash"]:
        raise ValueError("plano alterado")
    if any(sha256_arquivo(REPO / p) != h for p, h in plano["referencias"].items()):
        raise ValueError("inventario mudou: criar novo plano")
    total, feitos, falhas = 0, [], []
    grupos = plano["gfm"][:1] if leve else plano["gfm"]
    for grupo in grupos:
        for item in grupo["itens"][:1] if leve else grupo["itens"]:
            for k in CAMADAS:
                try:
                    if k not in item["assets"]:
                        raise ValueError("camada obrigatoria ausente")
                    url = item["assets"][k]["href"]
                    if urllib.parse.urlparse(url).hostname != "data.eodc.eu":
                        raise ValueError("asset fora da origem")
                    r = baixar(url, pasta / "gfm" / grupo["evento"]["eventid"] / item["id"] / f"{k}.tif",
                               min(limite_mb * 1024**2, total_mb * 1024**2 - total))
                    feitos.append({"evento": grupo["evento"]["eventid"], "item": item["id"], "camada": k, **r})
                    total += r["bytes"]
                    print(f"BAIXADO {grupo['evento']['eventid']} {k} {r['bytes']} bytes", flush=True)
                except (OSError, ValueError) as exc:
                    falhas.append({"evento": grupo["evento"]["eventid"], "item": item["id"], "camada": k, "erro": str(exc)})
    for x in plano["zenodo"]["arquivos"]:
        try:
            nome = x["key"]
            if nome not in ("Pre_Flood.zip", "During_Flood.zip"):
                raise ValueError("arquivo Zenodo nao previsto")
            r = baixar(x["links"]["self"], pasta / "optico_rs2024" / nome,
                       min(limite_mb * 1024**2, total_mb * 1024**2 - total), x["checksum"], x["size"])
            with zipfile.ZipFile(r["arquivo"]) as z:
                if z.testzip() is not None:
                    raise ValueError("ZIP corrompido")
                r = {**r, "membros": [{"nome": i.filename, "bytes_descompactados": i.file_size} for i in z.infolist()]}
            feitos.append({"evento": "CEMS:EMSR720", "instrumento": "DSWx-HLS mosaico", **r})
            total += r["bytes"]
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            falhas.append({"arquivo": x["key"], "erro": str(exc)})
    r = {"tipo": "aquisicao_observacional_obs01", "plano_hash": plano["hash"],
         "arquivos": feitos, "falhas": falhas, "bytes": total, "promovido": False,
         "admitido_no_treino": False, "novo_teste_confirmatorio": False,
         "etapa": "leve: primeiro item e mosaicos opticos; restante nao baixado" if leve else "lote do plano",
         "restante_do_plano_baixado": not leve and not falhas}
    destino = pasta / "aquisicoes" / sha256_json(r).split(":")[-1]
    gravar(destino / "manifesto.json", r)
    print(f"MANIFESTO={destino / 'manifesto.json'}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--saida", type=Path)
    ap.add_argument("--plano", type=Path)
    ap.add_argument("--eventos", type=int, default=4)
    ap.add_argument("--itens", type=int, default=2)
    ap.add_argument("--baixar", action="store_true")
    ap.add_argument("--leve", action="store_true", help="primeiro item e dois ZIPs, ate 16 MiB no total")
    a = ap.parse_args()
    if not a.saida and not a.plano:
        ap.error("informe --saida para catalogar ou --plano para retomar")
    pasta = a.plano.parent if a.plano else catalogar(a.saida, a.eventos, a.itens)
    print(f"PLANO={pasta / 'plano.json'}", flush=True)
    if a.baixar:
        r = adquirir(pasta, limite_mb=16, total_mb=16, leve=True) if a.leve else adquirir(pasta)
        return int(bool(r["falhas"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
