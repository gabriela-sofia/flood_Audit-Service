"""Qualifica pacotes historicos dirigidos a enxurrada, sem ingestao na base.

Seleciona primeira AOI DEL primaria por ativacao para diagnosticar o esquema;
essa fatia nao e uma amostra representativa da ativacao nem uma unidade admitida.
"""
from __future__ import annotations
import argparse
import re
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ('metodo', 'geo', 'servico'):
    sys.path.insert(0, str(REPO / 'scripts' / pasta))
from mt04_confirmacao import ler, gravar, sha256_arquivo, agora_utc
from mt04_aquisicao import protegidos, baixar_retomavel, get

SAIDA = REPO / 'local_runs/mecanismos-transferiveis/v4/enxurrada'
BASE = SAIDA.parent
INCLUIR_GRA = False
NOMES_HISTORICOS = False
EXPANDIR_ZIPS_INTERNOS = False


def extrair_seguro(z, destino):
    raiz = Path(destino).resolve()
    for m in z.infolist():
        p = PurePosixPath(m.filename.replace('\\', '/'))
        if p.is_absolute() or '..' in p.parts or any(':' in x for x in p.parts):
            raise ValueError('caminho inseguro no pacote')
        alvo = raiz.joinpath(*p.parts).resolve()
        if not alvo.is_relative_to(raiz):
            raise ValueError('caminho fora da pasta')
    z.extractall(raiz)


def planejar():
    t = ler(BASE / 'triagem_historica/resultado.json')
    ban = protegidos()
    rows = []
    for r in sorted(t['candidatos_textuais'], key=lambda x: (x['activationTime'], x['code'])):
        if r['code'] in ban:
            raise ValueError('protegida na fila')
        urls = sorted(u for u in r['pacotes'] if '_DEL_PRODUCT_' in u)
        if not urls and NOMES_HISTORICOS:
            urls = sorted(u for u in r['pacotes'] if re.search(r'_(?:\d+)?DELINEATION_',u) and '_MONIT' not in u)
        if not urls and INCLUIR_GRA:
            urls = sorted(u for u in r['pacotes'] if '_GRA_PRODUCT_' in u)
            if not urls and NOMES_HISTORICOS:
                urls = sorted(u for u in r['pacotes'] if re.search(r'_(?:\d+)?GRADING_',u) and '_MONIT' not in u)
        # Diagnostico do formato; os produtos restantes continuam explicitamente pendentes.
        if not urls:
            rows.append({'code': r['code'], 'estado': 'sem_DEL_primario', 'pacotes': len(r['pacotes'])})
            continue
        rows.append({'code': r['code'], 'url': urls[0], 'nome': r['nome'], 'centroid': r['centroid'],
                     'estado': 'diagnostico_de_uma_AOI', 'outros_produtos_pendentes': len(urls)-1})
    p = {'tipo': 'mt04_diagnostico_esquema_historico', 'quando_utc': agora_utc(), 'selecao': rows,
         'regra': 'primeira URL DEL_PRODUCT por ativacao; GRA_PRODUCT se autorizado e sem DEL; ordem temporal, sem selecao por rotulo ou desempenho',
         'grading_incluido': INCLUIR_GRA,
         'nomes_historicos': NOMES_HISTORICOS,
         'teto_total_bytes': 512*1024**2, 'teto_pacote_bytes': 64*1024**2,
         'papel': 'desenvolvimento; nunca confirmacao', 'reserva_consumida_por_este_lote': False,
         'nao_e': 'nao representa todas as AOIs nem confirma 16 unidades de enxurrada'}
    arq = SAIDA / 'pre_registro.json'
    if arq.exists():
        anterior = ler(arq)
        if anterior['selecao'] != rows:
            raise ValueError('selecao mudou; versionar')
        return anterior
    gravar(arq, p)
    return p


def adquirir():
    import requests
    p = planejar()
    acumulado, rows = 0, []
    for s in p['selecao']:
        if 'url' not in s:
            rows.append(s)
            continue
        code, url = s['code'], s['url']
        if urlparse(url).hostname != 'cems-mapping-website.s3.eu-west-1.amazonaws.com':
            raise ValueError('produto fora da origem')
        try:
            h = requests.head(url, timeout=60)
            h.raise_for_status()
            n = int(h.headers['Content-Length'])
            if n > p['teto_pacote_bytes'] or acumulado + n > p['teto_total_bytes']:
                rows.append({**s, 'estado': 'acima_do_teto', 'bytes': n})
                continue
            destino = SAIDA / 'pacotes' / code / Path(urlparse(url).path).name
            baixar_retomavel(url, destino, n, reserva_disco=2*1024**3)
            acumulado += n
            rows.append({**s, 'estado': 'adquirido', 'arquivo': str(destino.relative_to(REPO)),
                         'bytes': n, 'sha256': sha256_arquivo(destino), 'etag': h.headers.get('ETag')})
            print('vetor', code, n, flush=True)
        except Exception as exc:
            rows.append({**s, 'estado': 'falha', 'erro': str(exc)[:400]})
    r = {'tipo': 'mt04_pacotes_historicos', 'quando_utc': agora_utc(), 'pre_registro': sha256_arquivo(SAIDA/'pre_registro.json'),
         'pacotes': rows, 'bytes': acumulado, 'licenca': 'Copernicus EMS; atribuicao a Uniao Europeia; conferir metadados por produto'}
    gravar(SAIDA / 'aquisicao.json', r)
    return {'adquiridos': sum(x['estado']=='adquirido' for x in rows), 'bytes': acumulado}


def qualificar():
    import fiona
    from shapely.geometry import shape
    registros = []
    for s in ler(SAIDA / 'aquisicao.json')['pacotes']:
        if s['estado'] != 'adquirido':
            continue
        arq = REPO / s['arquivo']
        if sha256_arquivo(arq) != s['sha256']:
            raise ValueError('pacote alterado')
        (SAIDA / 'temporario').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='mt04_esquema_', dir=SAIDA/'temporario') as tmp:
            try:
                with zipfile.ZipFile(arq) as z:
                    extrair_seguro(z, tmp)
            except (zipfile.BadZipFile,ValueError,RuntimeError) as exc:
                registros.append({'code':s['code'],'camadas':[], 'estado':'pacote_invalido',
                                  'erro':str(exc)[:400],'flash_explicito_no_produto':None,
                                  'tem_AOI':False,'tem_footprint':False,'esquema_canonico_presente':False,
                                  'admitido':False,'motivo':'conteudo nao admitido; nao conta como ausencia de enxurrada'})
                continue
            camadas = []
            if EXPANDIR_ZIPS_INTERNOS:
                for interno in list(Path(tmp).rglob('*.zip')):
                    try:
                        with zipfile.ZipFile(interno) as z:
                            extrair_seguro(z,interno.with_suffix(''))
                    except (zipfile.BadZipFile,ValueError,RuntimeError) as exc:
                        camadas.append({'arquivo':interno.name,'erro':str(exc)[:400]})
            for f in sorted(Path(tmp).rglob('*.shp')):
                try:
                    with fiona.Env(SHAPE_RESTORE_SHX='YES'), fiona.open(f) as src:
                        campos = dict(src.schema['properties'])
                        numero = len(src)
                        crs = str(src.crs)
                    datas_invalidas_possiveis = [k for k,v in campos.items() if v in ('date','datetime')]
                    cols_uteis = ['event_type','obj_desc','eventphase','src_date','fcode','type','sbt ypdes','sbtypdes']
                    ignorar = [k for k in campos if k.lower() not in cols_uteis or k in datas_invalidas_possiveis]
                    with fiona.Env(SHAPE_RESTORE_SHX='YES'), fiona.open(f,ignore_fields=ignorar) as src:
                        props = [dict(x['properties']) for x in src] if len(ignorar)<len(campos) else []
                        descr = sorted({str(v) for x in props for k,v in x.items()
                                        if k.lower() in cols_uteis})
                        camadas.append({'arquivo': f.name, 'crs': crs, 'feicoes': numero,
                                        'campos': campos, 'descricoes': descr,
                                        'datas_nativas_nao_convertidas':datas_invalidas_possiveis,
                                        'data_observada_validada':False})
                except Exception as exc:
                    camadas.append({'arquivo': f.name, 'erro': str(exc)[:400]})
            eventos = [x for x in camadas if 'observedevent' in x['arquivo'].lower()]
            flash = any(any('flash flood' in v.lower() for v in x.get('descricoes', [])) for x in eventos)
            tem_aoi = any('areaofinterest' in x['arquivo'].lower() for x in camadas)
            tem_fp = any('imagefootprint' in x['arquivo'].lower() for x in camadas)
            registros.append({'code': s['code'], 'camadas': camadas, 'flash_explicito_no_produto': flash,
                              'tem_AOI': tem_aoi, 'tem_footprint': tem_fp,
                              'esquema_canonico_presente': bool(eventos and tem_aoi and tem_fp),
                              'admitido': False, 'motivo': 'diagnostico de uma AOI; falta completar cobertura, algebra, datas e independencia'})
    r = {'tipo': 'mt04_qualificacao_historica', 'quando_utc': agora_utc(), 'ativacoes': registros,
         'unidades_independentes_admitidas': 0, 'enxurrada_resolvida': False}
    gravar(SAIDA / 'qualificacao.json', r)
    return {'diagnosticadas': len(registros), 'com_flash_explicito': sum(bool(x['flash_explicito_no_produto']) for x in registros),
            'com_esquema_canonico': sum(x['esquema_canonico_presente'] for x in registros)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('acao', choices=['planejar','adquirir','qualificar'])
    ap.add_argument('--incluir-gra', action='store_true')
    ap.add_argument('--nomes-historicos', action='store_true')
    a = ap.parse_args()
    if a.incluir_gra:
        INCLUIR_GRA = True
        SAIDA = SAIDA / 'v2_com_grading'
    if a.nomes_historicos:
        NOMES_HISTORICOS = True
        SAIDA = BASE / 'enxurrada/v4_aliases_historicos'
    print({'planejar':planejar,'adquirir':adquirir,'qualificar':qualificar}[a.acao]())
