"""Aquisição pública limitada, incremental e registrada das fontes sociais."""


def _declarar_metadados_de_saida(dados):
    """Metadados de procedencia na copia serializada; nao modifica o objeto de calculo."""
    if not isinstance(dados, dict):
        return dados
    return {"tipo": "registro_social04_aquisicao_complementar",
            "gerado_por": 'scripts/social/social04_aquisicao_complementar.py',
            "nao_e": "Registro do produtor social04_aquisicao_complementar: declarar a procedencia nao revalida resultados nem autoriza promocao; conservar os estados e limites especificos do artefato.",
            **dados}

from pathlib import Path
import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'local_runs/social-fontes-complementares/2026-10-12'
BASE = 'https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios/'

def links():
    for folder in ('', 'Agregados_por_Setor_csv/'):
        html = urllib.request.urlopen(BASE + folder, timeout=60).read().decode('latin1')
        print(folder, re.findall(r'href="([^"]+\.(?:zip|xlsx))"', html))

def baixar(url, destino, limite=180_000_000):
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    registro = destino.with_suffix(destino.suffix + '.proveniencia.json')
    if destino.exists():
        if not registro.exists(): raise ValueError('arquivo sem proveniência')
        r = json.loads(registro.read_text(encoding='utf-8'))
        if hashlib.sha256(destino.read_bytes()).hexdigest() != r['sha256']: raise ValueError('hash divergente')
        return r
    parcial = destino.with_suffix(destino.suffix + '.part')
    h, tamanho = hashlib.sha256(), 0
    req = urllib.request.Request(url, headers={'User-Agent': 'REV-P-social/1.0'})
    with urllib.request.urlopen(req, timeout=120) as response, parcial.open('wb') as f:
        if int(response.headers.get('Content-Length', 0)) > limite: raise ValueError('fonte acima do limite')
        for bloco in iter(lambda: response.read(1 << 20), b''):
            tamanho += len(bloco)
            if tamanho > limite: raise ValueError('limite de download excedido')
            h.update(bloco); f.write(bloco)
    parcial.replace(destino)
    r = {'url': url, 'arquivo': destino.name, 'bytes': tamanho, 'sha256': h.hexdigest(),
         'adquirido_utc': datetime.now(timezone.utc).isoformat(), 'provedor': 'IBGE',
         'ano_referencia': 2022, 'uso': 'contexto agregado; não valida impacto ou modelo físico'}
    registro.write_text(json.dumps(_declarar_metadados_de_saida(r), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return r

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--listar', action='store_true')
    ap.add_argument('--url'); ap.add_argument('--arquivo')
    a = ap.parse_args()
    if a.listar: links()
    else:
        if not a.url or not a.arquivo or Path(a.arquivo).name != a.arquivo: ap.error('URL e nome simples obrigatórios')
        if not a.url.startswith(('https://ftp.ibge.gov.br/', 'https://geoftp.ibge.gov.br/')): ap.error('somente IBGE HTTPS')
        print(json.dumps(baixar(a.url, OUT/a.arquivo), ensure_ascii=False))
