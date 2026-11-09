"""Baixa o arquivo publicado pelos autores; confere todos os bytes antes de usar.

Nao atribui data, orbita ou evento aos chips. Essas informacoes precisam vir
dos metadados da fonte, nunca do resultado da rede.
"""
from __future__ import annotations


def _declarar_metadados_de_saida(dados):
    """Metadados de procedencia na copia serializada; nao modifica o objeto de calculo."""
    if not isinstance(dados, dict):
        return dados
    return {"tipo": "registro_adquirir_pares_s1gfloods",
            "gerado_por": 'scripts/sar/adquirir_pares_s1gfloods.py',
            "nao_e": "Registro do produtor adquirir_pares_s1gfloods: declarar a procedencia nao revalida resultados nem autoriza promocao; conservar os estados e limites especificos do artefato.",
            **dados}


import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
URL = "https://huggingface.co/datasets/Tamer-Saleh/S1GFloods/resolve/main/S1GFloods.rar"
HASH = "4f1b57c342c661853dec0d7384709ce4bb9e53097ab31cd060cd4a3f915e3e67"
TAMANHO = 673724972


def hash_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as entrada:
        for bloco in iter(lambda: entrada.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def adquirir(destino: Path) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / "S1GFloods.rar"
    if not arquivo.exists():
        parcial = arquivo.with_suffix(".parcial")
        with requests.get(URL, stream=True, timeout=(30, 120)) as resposta:
            resposta.raise_for_status()
            with parcial.open("wb") as saida:
                for bloco in resposta.iter_content(1024 * 1024):
                    saida.write(bloco)
        if parcial.stat().st_size != TAMANHO or hash_arquivo(parcial) != HASH:
            raise ValueError("arquivo recebido diverge do objeto LFS publicado")
        parcial.rename(arquivo)
    if arquivo.stat().st_size != TAMANHO or hash_arquivo(arquivo) != HASH:
        raise ValueError("arquivo local diverge do objeto publicado")
    manifesto = {"url": URL, "sha256": HASH, "bytes": TAMANHO,
                 "fonte": "https://github.com/Tamer-Saleh/S1GFlood-Detection",
                 "alvo": "inundacao nova anotada pela fonte",
                 "nao_e": "VV/VH em dB, previsao ou validacao brasileira"}
    (destino / "aquisicao.json").write_text(
        json.dumps(_declarar_metadados_de_saida(manifesto), indent=2), encoding="utf-8")
    return arquivo


def extrair(arquivo: Path, destino: Path) -> None:
    """O pacote so pode conter os tres diretorios de imagens conhecidos."""
    nomes = subprocess.run(["tar", "-tf", str(arquivo)], check=True,
                           capture_output=True, text=True).stdout.splitlines()
    for nome in nomes:
        partes = Path(nome).parts
        if (Path(nome).is_absolute() or ".." in partes or not partes
                or partes[0] not in {"A", "B", "Label"}
                or (len(partes) > 1 and
                    (len(partes) != 2 or Path(nome).suffix != ".png"))):
            raise ValueError(f"membro inesperado no pacote: {nome}")
    destino.mkdir(parents=True, exist_ok=True)
    # tar recusa a leitura se o arquivo estiver corrompido; o hash ja foi conferido.
    subprocess.run(["tar", "-xf", str(arquivo), "-C", str(destino)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=Path,
                        default=REPO / "local_runs" / "sar-pre-pos" / "dados")
    parser.add_argument("--extrair", action="store_true")
    args = parser.parse_args()
    arquivo = adquirir(args.destino)
    if args.extrair:
        extrair(arquivo, args.destino / "pares")
    print(arquivo)
