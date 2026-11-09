"""Aquisicao dirigida e incremental das fontes da campanha de encostas.

Cada fonte tem estimativa de tamanho antes do download, URL, data, SHA-256 e
licenca quando declarada. Nada e contornado: uma fonte atras de desafio
anti-robo e registrada como indisponivel e a campanha segue pelas outras.

Fontes:
  ufrj_geocart  inventario historico de eventos (Petropolis, 1933-2015), camada
                `eventos_ufrj_geocart` do SIG municipal, via WMS GetFeatureInfo
                publico (a camada nao e publicada por WFS). Uma consulta.
  idai          pontos de fonte e mapas de extensao (Dille et al., 2026), UNESCO
                IHP-WINS, CC BY-SA.
  usgs_maria    tentativa (Porto Rico, Hurricane Maria). ScienceBase responde com
                desafio Cloudflare: registrado, nao contornado.
  l4s           Landslide4Sense: so metadados do Zenodo para auditoria; nao baixado.
  fabdem        janelas FABDEM V1-2 (DTM 30 m) lidas por faixa no ZIP publico da
                Universidade de Bristol, para doador e destino pelo mesmo produto.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from datetime import datetime, timezone

from pa_comum import FONTES, Etapa, gravar_json, sha256, rel

UA = {"User-Agent": "REV-P pesquisa academica (aquisicao dirigida)"}
LIZMAP = ("https://sig.petropolis.rj.gov.br/sig398/index.php/lizmap/service?"
          "repository=publico&project=consulta_historica_eventos")
WINS_PKG = "https://ihp-wins.unesco.org/api/3/action/package_show?id=a8ee8e1f-2d15-4bd0-9676-c23bd05d54e0"
WINS_RES = {
    "idai_fontes.zip": "cebae5cf-0b07-49d4-aba0-c8a76358b712/download/idai-landslide-source-points.zip",
    "idai_extensao.zip": "f6178eaa-e0f6-41c0-b81c-1a3c9d1390cd/download/idai-landslide-extent-maps_zones.zip",
}
WINS_BASE = "https://ihp-wins.unesco.org/dataset/a8ee8e1f-2d15-4bd0-9676-c23bd05d54e0/resource/"
FABDEM = ("/vsizip//vsicurl/https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/"
          "{zip}_FABDEM_V1-2.zip/{tile}_FABDEM_V1-2.tif")
TETO_BYTES = 40_000_000

# Janelas definidas pela extensao das referencias, nao pelo resultado de modelo.
JANELAS = {
    "petropolis": {"bbox": (-43.42, -22.62, -42.93, -22.15), "epsg": 31983,
                   "tiles": [("S30W050-S20W040", "S23W044"), ("S30W050-S20W040", "S23W043")],
                   "por_que": "municipio 3303906 inteiro (42 FCU e inventarios)"},
    "idai": {"bbox": (32.40, -20.60, 33.08, -19.33), "epsg": 32736,
             "tiles": [("S30E030-S20E040", "S21E032"), ("S30E030-S20E040", "S21E033"),
                       ("S20E030-S10E040", "S20E032"), ("S20E030-S10E040", "S20E033")],
             "por_que": "extensao dos pontos de fonte Idai com margem de ~3 km"},
}


def agora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def baixar(url: str, destino, teto: int = TETO_BYTES) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r:
        tamanho = r.headers.get("Content-Length")
        if tamanho and int(tamanho) > teto:
            raise RuntimeError(f"{url} declara {tamanho} bytes, acima do teto {teto}")
        dados = r.read(teto + 1)
        url_final = r.geturl()
        tipo = r.headers.get("Content-Type")
    if len(dados) > teto:
        raise RuntimeError(f"{url} excedeu o teto de {teto} bytes")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(dados)
    return {"url": url, "url_final": url_final.split("?")[0], "content_type": tipo,
            "bytes": len(dados), "sha256": sha256(destino), "arquivo": rel(destino), "quando": agora()}


def ufrj() -> dict:
    cap = baixar(LIZMAP + "&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities",
                 FONTES / "ufrj_geocart" / "wms_capabilities.xml")
    # Uma consulta cobrindo a extensao declarada da camada (EPSG:31983
    # 674040-702767 / 7503954-7542856), com tolerancia que abrange tudo.
    gfi_url = (LIZMAP + "&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetFeatureInfo"
               "&LAYERS=eventos_ufrj_geocart&QUERY_LAYERS=eventos_ufrj_geocart&CRS=EPSG:31983"
               "&BBOX=673000,7503000,704000,7544000&WIDTH=1000&HEIGHT=1000&I=500&J=500"
               "&INFO_FORMAT=application/json&FEATURE_COUNT=10000&FI_POINT_TOLERANCE=1000&STYLES=")
    reg = baixar(gfi_url, FONTES / "ufrj_geocart" / "eventos_ufrj_geocart_gfi.json")
    feats = json.loads((FONTES / "ufrj_geocart" / "eventos_ufrj_geocart_gfi.json").read_text(encoding="utf-8"))["features"]
    ids = sorted(int(f["properties"]["id"]) for f in feats)
    completo = ids == list(range(1, len(ids) + 1))
    return {"fonte": "ufrj_geocart", "capabilities": cap, "consulta": reg, "feicoes": len(ids),
            "ids_contiguos_1_a_n": completo,
            "completude": ("completa por contiguidade de id nesta fotografia do servico" if completo
                           else "possivelmente incompleta: ids nao contiguos"),
            "acesso": "WMS GetFeatureInfo publico; camada consultavel no visualizador; WFS nao publicado (TYPENAME indisponivel)",
            "licenca": "nao declarada no servico; uso academico com citacao do SIG municipal e do GEOCART/UFRJ",
            "natureza": "relatos de ocorrencia com dano, compilados de jornais (IPT) e GEOCART, geocodificados por endereco"}


def idai() -> dict:
    pkg = baixar(WINS_PKG, FONTES / "idai" / "wins_package.json")
    meta = json.loads((FONTES / "idai" / "wins_package.json").read_text(encoding="utf-8"))["result"]
    regs = []
    for nome, caminho in WINS_RES.items():
        r = baixar(WINS_BASE + caminho, FONTES / "idai" / nome)
        with zipfile.ZipFile(FONTES / "idai" / nome) as z:
            r["conteudo"] = z.namelist()
            z.extractall(FONTES / "idai" / nome.replace(".zip", ""))
        regs.append(r)
    return {"fonte": "idai", "pacote": pkg, "arquivos": regs, "licenca": meta.get("license_title"),
            "metadata_modified": meta.get("metadata_modified"),
            "referencia": "Dille et al. (2026), NHESS 26, 2561, doi:10.5194/nhess-26-2561-2026; dados doi:10.63253/nii4g2ac",
            "gatilho": "chuva extrema do ciclone Idai, marco de 2019 (nao sismico)"}


def usgs_maria() -> dict:
    url = "https://www.sciencebase.gov/catalog/item/5d4c8b26e4b01d82ce8dfeb0?format=json"
    import urllib.error
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            corpo = r.read(4000).decode("utf-8", "replace")
            status = r.status
    except urllib.error.HTTPError as exc:
        corpo, status = exc.read(4000).decode("utf-8", "replace"), exc.code
    except Exception as exc:                                         # noqa: BLE001
        corpo, status = str(exc), None
    desafio = "Just a moment" in corpo or "challenge" in corpo.lower() or status in (403, 503)
    return {"fonte": "usgs_maria", "url": url, "doi": "10.5066/P9BVMD74", "status_http": status,
            "desafio_antirrobo": desafio, "quando": agora(),
            "decisao": ("indisponivel nesta execucao: desafio anti-robo; nao contornado" if desafio
                        else "resposta sem desafio; nao usado nesta versao")}


def l4s() -> dict:
    try:
        reg = baixar("https://zenodo.org/api/records/10463239", FONTES / "l4s" / "zenodo_record.json")
        d = json.loads((FONTES / "l4s" / "zenodo_record.json").read_text(encoding="utf-8"))
        arquivos = [{"nome": f.get("key"), "bytes": f.get("size")} for f in d.get("files", [])]
        lic = (d.get("metadata", {}).get("license") or {}).get("id")
    except Exception as exc:                                         # noqa: BLE001
        reg, arquivos, lic = {"erro": str(exc)}, [], None
    return {"fonte": "landslide4sense", "registro": reg, "arquivos": arquivos, "licenca": lic,
            "decisao": ("nao baixado: tarefa de deteccao pos-evento em patches sem georreferencia por regiao; "
                        "mistura gatilhos sismicos (Iburi, Gorkha) e nao ha imagem pos-evento de Petropolis "
                        "qualificada nesta linhagem para a tarefa complementar")}


def fabdem() -> dict:
    import numpy as np
    import rasterio
    import rasterio.transform
    from rasterio.env import Env
    from rasterio.merge import merge
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    saida = {}
    with Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".zip,.tif",
             GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3"):
        for nome, j in JANELAS.items():
            fontes = [rasterio.open(FABDEM.format(zip=z, tile=t)) for z, t in j["tiles"]]
            mosaico, transf = merge(fontes, bounds=j["bbox"], nodata=-9999.0)
            src_crs = fontes[0].crs
            for f in fontes:
                f.close()
            h, w = mosaico.shape[1:]
            dst_t, dw, dh = calculate_default_transform(src_crs, f"EPSG:{j['epsg']}", w, h,
                                                        *rasterio.transform.array_bounds(h, w, transf),
                                                        resolution=30.0)
            dst = np.full((dh, dw), -9999.0, dtype="float32")
            reproject(mosaico[0].astype("float32"), dst, src_transform=transf, src_crs=src_crs,
                      src_nodata=-9999.0, dst_transform=dst_t, dst_crs=f"EPSG:{j['epsg']}",
                      dst_nodata=-9999.0, resampling=Resampling.bilinear)
            arq = FONTES / "fabdem" / f"fabdem_{nome}_epsg{j['epsg']}_30m.tif"
            arq.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(arq, "w", driver="GTiff", height=dh, width=dw, count=1, dtype="float32",
                               crs=f"EPSG:{j['epsg']}", transform=dst_t, nodata=-9999.0,
                               compress="deflate", tiled=True) as d:
                d.write(dst, 1)
            valid = dst > -9000
            saida[nome] = {"arquivo": rel(arq), "sha256": sha256(arq), "tiles": j["tiles"],
                           "bbox_wgs84": j["bbox"], "por_que": j["por_que"], "epsg": j["epsg"],
                           "forma": [dh, dw], "celulas_validas": int(valid.sum()),
                           "bytes_lidos_estimados": int(mosaico.size * 4),
                           "elev_min_max": [float(dst[valid].min()), float(dst[valid].max())],
                           "reamostragem": "bilinear para 30 m no UTM local",
                           "licenca": "FABDEM V1-2, CC BY-NC-SA 4.0 (Hawker et al., 2022)", "quando": agora()}
    return saida


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--so", nargs="*", default=["ufrj", "idai", "usgs_maria", "l4s", "fabdem"])
    a = p.parse_args()
    e = Etapa("pa_01_aquisicao")
    destino = FONTES / "aquisicao.json"
    reg = json.loads(destino.read_text(encoding="utf-8")) if destino.exists() else {}
    funcs = {"ufrj": ufrj, "idai": idai, "usgs_maria": usgs_maria, "l4s": l4s, "fabdem": fabdem}
    for nome in a.so:
        print(f"[aquisicao] {nome}", flush=True)
        reg[nome] = funcs[nome]()
        gravar_json(destino, reg)
    e.saida(destino)
    e.fechar(fontes=a.so)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
