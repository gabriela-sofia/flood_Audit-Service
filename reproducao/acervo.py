"""Leitura dos arquivos incluídos, a partir de sua origem registrada."""
from __future__ import annotations
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

def registro():
    return json.loads((RAIZ/'manifestos/arquivos_origem.json').read_text(encoding='utf-8'))['arquivos']

def caminho(origem):
    itens = [x for x in registro() if x['origem'] == origem]
    if len(itens) != 1:
        raise ValueError(f'Origem deve resolver uma vez: {origem}; encontrados {len(itens)}')
    p = (RAIZ/itens[0]['caminho']).resolve()
    if not p.is_relative_to(RAIZ):
        raise ValueError('Caminho fora do pacote')
    return p

def ler_json(origem):
    return json.loads(caminho(origem).read_text(encoding='utf-8-sig'))
