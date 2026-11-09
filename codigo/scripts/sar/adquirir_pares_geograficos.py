"""Adquire o Sen1Floods11 bitemporal publicado por Yadav, sem inventar metadados."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
REGISTRO = "https://zenodo.org/api/records/7946594"
ARQUIVOS = {
    "Labels-20230517T191741Z-001.zip": (2462219, "069b4c05eefb7a6e72c1adb34aaf1a24"),
    "POST_S1-20230517T191716Z-001.zip": (729719788, "40a505cfbd5318d94a9d5bd7aef88561"),
    "PRE_S1-20230517T191707Z-001.zip": (1520699949, "4a32637c56ea519bd3c4baca208b289d"),
}
METADADOS = "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/Sen1Floods11_Metadata.geojson"
BLOCO_BYTES = 32 * 1024 * 1024


def baixar_intervalos(sessao, url, parcial, tamanho):
    """Retoma bytes; so admite o arquivo depois do checksum integral."""
    bloco = BLOCO_BYTES

    def receber(intervalo):
        inicio, fim = intervalo
        cliente = requests.Session()
        cliente.trust_env = sessao.trust_env
        for tentativa in range(3):
            try:
                with cliente.get(url, headers={"Range": f"bytes={inicio}-{fim}"},
                                stream=True, timeout=(30, 45)) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {inicio}-{fim}/{tamanho}":
                        raise ValueError("servidor nao confirmou o intervalo requerido")
                    recebido = bytearray()
                    for b in r.iter_content(1024 * 1024):
                        recebido.extend(b)
                    if len(recebido) != fim - inicio + 1:
                        raise ValueError("intervalo incompleto")
                return recebido
            except (requests.RequestException, ValueError):
                if tentativa == 2:
                    raise
                time.sleep(2 ** tentativa)
        raise ValueError("intervalo nao recebido")

    with ThreadPoolExecutor(max_workers=4) as executor:
        while (parcial.stat().st_size if parcial.exists() else 0) < tamanho:
            inicio = parcial.stat().st_size if parcial.exists() else 0
            intervalos = [(i, min(i + bloco, tamanho)-1)
                          for i in range(inicio, min(inicio + 4 * bloco, tamanho), bloco)]
            # Grava na ordem do arquivo, nunca na ordem de chegada das respostas.
            for (_, fim), recebido in zip(intervalos, executor.map(receber, intervalos)):
                with parcial.open("ab") as f:
                    f.write(recebido)
                print(json.dumps({"arquivo": parcial.name, "bytes": fim + 1}), flush=True)


def hashes(caminho):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(caminho).open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(bloco)
            sha.update(bloco)
    return md5.hexdigest(), sha.hexdigest()


def extrair(arquivo, destino):
    destino = Path(destino).resolve()
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(arquivo) as z:
        for membro in z.infolist():
            alvo = (destino / membro.filename).resolve()
            if not alvo.is_relative_to(destino) or (membro.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("membro fora do destino ou link simbolico")
        z.extractall(destino)


def adquirir(destino, apenas_rotulos=False):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    sessao = requests.Session()
    sessao.trust_env = False
    resposta = sessao.get(REGISTRO, timeout=60)
    resposta.raise_for_status()
    registro = resposta.json()
    if registro["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("licenca diverge da referencia admitida")
    (destino / "registro_zenodo.json").write_text(json.dumps(registro, indent=2), encoding="utf-8")
    r = sessao.get(METADADOS, timeout=60)
    r.raise_for_status()
    if r.json().get("type") != "FeatureCollection":
        raise ValueError("catalogo de eventos invalido")
    (destino / "Sen1Floods11_Metadata.geojson").write_bytes(r.content)
    itens = {f["key"]: f for f in registro["files"]}
    admitidos = {}
    for nome, (tamanho, md5) in ARQUIVOS.items():
        if apenas_rotulos and not nome.startswith("Labels"):
            continue
        if itens[nome]["size"] != tamanho or itens[nome]["checksum"] != "md5:" + md5:
            raise ValueError("objeto publicado mudou")
        arquivo = destino / nome
        if not arquivo.exists():
            parcial = arquivo.with_suffix(".parcial")
            url = f"https://zenodo.org/records/7946594/files/{nome}?download=1"
            baixar_intervalos(sessao, url, parcial, tamanho)
            if parcial.stat().st_size != tamanho or hashes(parcial)[0] != md5:
                raise ValueError("download incompleto ou checksum divergente")
            os.replace(parcial, arquivo)
        hm, hs = hashes(arquivo)
        if arquivo.stat().st_size != tamanho or hm != md5:
            raise ValueError("arquivo local divergente")
        extrair(arquivo, destino / "extraidos")
        admitidos[nome] = {"bytes": tamanho, "md5": hm, "sha256": hs,
                           "url": itens[nome]["links"]["self"]}
        print(json.dumps({"conferido": nome, "sha256": hs}), flush=True)
    manifesto = {"fonte": "https://zenodo.org/records/7946594", "licenca": "CC-BY-4.0",
                 "autoria": "Ritu Yadav; Attentive Dual Stream Siamese U-Net, IGARSS 2022",
                 "arquivos": admitidos, "promovido": False,
                 "metadados_eventos": {"url": METADADOS,
                    "sha256": hashlib.sha256((destino / "Sen1Floods11_Metadata.geojson").read_bytes()).hexdigest()},
                 "antes": "mediana do ano anterior, mesma geometria SAR conforme artigo; nao cena unica",
                 "alvo_original": "agua 1, nao agua 0, desconhecido -1; nao inundacao nova garantida"}
    (destino / "aquisicao.json").write_text(json.dumps(manifesto, indent=2), encoding="utf-8")
    return manifesto


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destino", type=Path, default=REPO / "local_runs/sar-independencia/dados")
    p.add_argument("--apenas-rotulos", action="store_true")
    a = p.parse_args()
    adquirir(a.destino, a.apenas_rotulos)
