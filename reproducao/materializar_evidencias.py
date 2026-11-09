"""Copia os arquivos publicados para os caminhos de trabalho esperados.

Recusa sobrescrever um arquivo diferente. Não cria dados ausentes, não
executa protocolos e não modifica a fonte publicada.
"""
from __future__ import annotations
import hashlib
import json
import shutil
from acervo import RAIZ, registro

def materializar():
    escritos=0
    for item in registro():
        if not item['origem'].startswith('local_runs/'):
            continue
        src=RAIZ/item['caminho']
        dst=(RAIZ/'codigo'/item['origem']).resolve()
        assert dst.is_relative_to((RAIZ/'codigo/local_runs').resolve())
        if dst.exists():
            if hashlib.sha256(dst.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Arquivo de trabalho diferente; preservado: '+str(dst))
            continue
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        escritos+=1
    return {'arquivos_materializados':escritos,'destino':'codigo/local_runs',
            'nota':'Somente os arquivos publicados. Entradas externas não são preenchidas.'}

if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(materializar(),ensure_ascii=False,indent=2))
