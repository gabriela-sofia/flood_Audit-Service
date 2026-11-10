"""Verifica o pacote organizado, sem dados externos, rede ou ajuste de modelos."""
from __future__ import annotations
import ast
import hashlib
import json
import re
from pathlib import Path

from acervo import RAIZ, registro

def verificar(raiz=RAIZ):
    raiz = Path(raiz).resolve()
    erros = []
    copias = registro()
    for item in copias:
        p = (raiz/item['caminho']).resolve()
        if not p.is_relative_to(raiz) or not p.is_file():
            erros.append('Arquivo de origem ausente: '+item['caminho'])
        elif hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            erros.append('Hash divergente: '+item['caminho'])
    grafo = json.loads((raiz/'manifestos/dependencias_codigo.json').read_text(encoding='utf-8'))
    for p in set(grafo['entradas']) | set(grafo['arestas']) | {q for v in grafo['arestas'].values() for q in v}:
        if not (raiz/p).is_file(): erros.append('Código ausente: '+p)
    py = list((raiz/'codigo').rglob('*.py'))
    py = [p for p in py if '__pycache__' not in p.parts and 'local_runs' not in p.parts]
    for p in py:
        try: ast.parse(p.read_text(encoding='utf-8-sig'), filename=p.relative_to(raiz).as_posix())
        except SyntaxError as e: erros.append(f'Sintaxe: {p.relative_to(raiz)}:{e.lineno}')
    matriz = json.loads((raiz/'manifestos/afirmacoes_artigo.json').read_text(encoding='utf-8'))
    ids = [c['id'] for c in matriz['afirmacoes']]
    if len(ids) != len(set(ids)): erros.append('ID repetido na matriz')
    for c in matriz['afirmacoes']:
        if not c['evidencias']:
            erros.append('Item sem evidência: '+c['id'])
        for p in c['codigo']+c['evidencias']:
            if not (raiz/p).is_file(): erros.append(f"Vínculo ausente em {c['id']}: {p}")
    obrigatorios = ['README.md','docs/METODO.md','docs/MATRIZ_ARTIGO.md','docs/CODIGO.md',
                    'docs/REPRODUCAO.md','docs/DADOS_E_FONTES.md','docs/CONTRIBUICOES.md']
    for p in obrigatorios:
        if not (raiz/p).is_file(): erros.append('Documento ausente: '+p)
    # Links da documentação preparada; relatórios de origem preservam o contexto de execução.
    docs = [raiz/'README.md',*(raiz/'docs').glob('*.md'),*(raiz/'modelos').rglob('README.md'),
            *(raiz/'codigo').rglob('README.md'),*(raiz/'evidencias').rglob('README.md'),
            raiz/'manifestos/README.md',raiz/'artigo/README.md']
    links = 0
    for p in docs:
        if not p.exists(): continue
        for alvo in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if re.match(r'^[a-z]+:|^#',alvo): continue
            alvo = alvo.split('#',1)[0].strip('<>')
            if not alvo: continue
            links += 1
            if not (p.parent/alvo).exists(): erros.append(f'Link ausente em {p.relative_to(raiz)}: {alvo}')
    for proibido in ('historico','archive','versoes_antigas'):
        if (raiz/proibido).exists(): erros.append('Pasta fora do recorte: '+proibido)
    origem = json.loads((raiz/'manifestos/origem.json').read_text(encoding='utf-8'))
    pdf = raiz/'artigo/REV-P_artigo_atual.pdf'
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != origem['artigo_sha256']:
        erros.append('PDF difere do artigo de referência')
    poster = raiz/'artigo/REV-P_poster.pdf'
    if hashlib.sha256(poster.read_bytes()).hexdigest() != origem['poster_sha256']:
        erros.append('PDF difere do pôster de referência')
    fonte = (raiz/'artigo/fonte/artigo.tex').read_text(encoding='utf-8')
    tabelas = re.findall(r'\\label\{tab:([^}]+)\}', fonte)
    if len(tabelas) != matriz['tabelas'] or len(tabelas) != len(set(tabelas)):
        erros.append('Tabelas do artigo divergem da matriz')
    for item in matriz['afirmacoes']:
        if item['texto_tex'] and item['texto_tex'] not in fonte:
            erros.append('Trecho ausente do artigo: '+item['id'])
    for figura in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', fonte):
        if not (raiz/'artigo/fonte/figuras'/figura).is_file():
            erros.append('Figura ausente: '+figura)
    citadas = {k for citacao in re.findall(r'\\cite\{([^}]+)\}',fonte) for k in citacao.split(',')}
    bibliografia = {k for k in re.findall(r'\\bibitem\{([^}]+)\}',fonte)}
    if citadas != bibliografia:
        erros.append('Chamadas bibliográficas divergem das referências do artigo')
    return {'tipo':'integridade_da_preparacao','ok':not erros,'erros':erros,
            'copias_com_hash_conferido':len(copias),'arquivos_python_analisados':len(py),
            'links_da_documentacao_conferidos':links,'linhas_tabeladas':matriz['afirmacoes_tabeladas'],
            'paragrafos_rastreados':matriz['paragrafos_rastreados'],
            'nota':'Identidade, presença, sintaxe e rastreabilidade; não é nova validação científica.'}

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    r=verificar()
    print(json.dumps(r,ensure_ascii=False,indent=2))
    raise SystemExit(0 if r['ok'] else 1)
