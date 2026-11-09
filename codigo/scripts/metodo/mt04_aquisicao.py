"""Busca dirigida: catalogo CEMS historico, referencias municipais e GT urbano.

Nao ingere dados nas cadeias compartilhadas nem promove modelos. Downloads
retomam por HTTP Range, verificam identidade/tamanho e preservam manifestos.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/metodo'))
from mt04_confirmacao import gravar, ler, agora_utc, sha256_arquivo, sha256_json
import mt03_triagem_cems as tri

SAIDA = REPO / 'local_runs/mecanismos-transferiveis/v4'
HF_REV = '0892ffa10cedb7ebd8456e1386b7a91615230429'
TAR = f'https://huggingface.co/datasets/S1Floodbenchmark/UrbanSARFloods_v1/resolve/{HF_REV}/urban_sar_floods.tar.gz'
EVENTOS = ['20190329_Iran', '20221013_Niger', '20230805_Hebei_1', '20230805_Hebei_2',
           '20191012_Japan', '20161011_Lumberton', '20220302_Coraki_Australia']
FLASH = re.compile(r'flash[ -]?flood|torrential|torrente|pluvial|cloudburst|enxurrada|rapid runoff', re.I)
UA = {'User-Agent': 'REV-P-pesquisa/1.0 (uso academico; aquisicao dirigida)'}


def get(url, **kw):
    r = requests.get(url, headers=UA, timeout=60, **kw)
    r.raise_for_status()
    return r


def protegidos():
    p = ler(tri.SAIDA / 'particao.json')
    return tri.NUNCA_LER | tri.RESERVA_MT02 | tri.FRENTE_SAR | {
        c for u in p['unidades'] if u['papel'] == 'reserva' for c in u['ativacoes']}


def catalogo():
    pasta = SAIDA / 'triagem_historica'
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / 'catalogo.json'
    if not arq.exists():
        url = 'https://mapping.emergency.copernicus.eu/activations/api/activations/?limit=1500'
        itens, vistos = [], set()
        while url:
            if url in vistos or urlparse(url).hostname != 'mapping.emergency.copernicus.eu':
                raise ValueError('paginacao fora da origem ou ciclica')
            vistos.add(url)
            d = get(url).json()
            itens.extend(tri._sem_stats(d['results']))
            url = d.get('next')
            if url:
                url = url.replace('http://', 'https://', 1)
        gravar(arq, {'quando_utc': agora_utc(), 'origem': sorted(vistos), 'ativacoes': itens})
    itens = ler(arq)['ativacoes']
    ban = protegidos()
    alvo = [a for a in itens if a['code'].startswith('EMSR') and a['code'] not in ban and int(a['code'][4:]) < 650
            and a.get('category', {}).get('name') in ('Flood', 'Storm')]
    def detalhe(a):
        code = a['code']
        dest = pasta / 'metadados' / f'{code}.json'
        if dest.exists():
            return ler(dest)
        url = f'https://mapping.emergency.copernicus.eu/activations/{code}/'
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(get(url).text, 'html.parser')
            tag = soup.find(string=re.compile('Activation Reason'))
            reason = tag.parent.parent.get_text(' ', strip=True) if tag else ''
            urls = sorted({x.get('href') for x in soup.select('a[href]')
                           if str(x.get('href', '')).endswith('_vector.zip')})
            r = {'code': code, 'nome': a['name'], 'centroid': a.get('centroid'),
                 'activationTime': a.get('activationTime'), 'categoria': a['category']['name'],
                 'reason': reason, 'indicacao_enxurrada': bool(FLASH.search(reason + ' ' + a['name'])),
                 'pacotes': urls, 'url': url, 'quando_utc': agora_utc(),
                 'nao_e': 'triagem textual; processo no poligono, negativo observado e independencia ainda nao admitidos'}
            gravar(dest, r)
            print('metadado', code, r['indicacao_enxurrada'], len(urls), flush=True)
            return r
        except Exception as exc:
            return {'code': code, 'erro': str(exc)[:400]}
    with ThreadPoolExecutor(3) as ex:
        rows = list(ex.map(detalhe, alvo))
    candidatos = [r for r in rows if r.get('indicacao_enxurrada') and r.get('pacotes')]
    existentes = set()
    for root in ('cems-05-pacotes', 'mecanismos-transferiveis/v3/cems_desenvolvimento/pacotes'):
        existentes |= {x for f in (REPO / 'local_runs' / root).rglob('*.zip') for x in re.findall(r'EMSR\d+', f.name)}
    for r in candidatos:
        r['ja_examinado_pacote'] = r['code'] in existentes
    out = {'tipo': 'mt04_triagem_historica', 'quando_utc': agora_utc(), 'catalogo_sha256': sha256_arquivo(arq),
           'ativacoes_catalogadas': len(itens), 'historicas_flood_storm': len(alvo),
           'candidatos_textuais': candidatos, 'falhas': [r for r in rows if 'erro' in r],
           'protegidos_excluidos': sorted(ban), 'rotulos_abertos': False,
           'unidades_independentes_admitidas': 0,
           'proxima': 'confrontar episodio/tempo/AOI/bacias, congelar desenho e admitir poligonos observados antes de contar unidades'}
    gravar(pasta / 'resultado.json', out)
    return {'catalogadas': len(itens), 'detalhadas': len(rows), 'candidatos': len(candidatos), 'falhas': len(out['falhas'])}


def camada(url, destino):
    meta = get(url, params={'f': 'json'}).json()
    if 'error' in meta:
        raise ValueError(str(meta['error']))
    ids = get(url + '/query', params={'f': 'json', 'where': '1=1', 'returnIdsOnly': 'true'}).json()
    if 'error' in ids or 'objectIds' not in ids:
        raise ValueError('consulta de IDs nao disponivel: ' + str(ids)[:400])
    ids = sorted(ids['objectIds'] or [])
    feats = []
    for i in range(0, len(ids), 500):
        d = get(url + '/query', params={'f': 'geojson', 'objectIds': ','.join(map(str, ids[i:i+500])),
                                      'outFields': '*', 'outSR': 4326, 'returnGeometry': 'true'}).json()
        if 'error' in d or d.get('exceededTransferLimit'):
            raise ValueError('consulta incompleta')
        feats.extend(d.get('features', []))
    campo = meta.get('objectIdField') or next((f['name'] for f in meta.get('fields', [])
                                             if f.get('type') == 'esriFieldTypeOID'), None)
    obtidos = [f['properties'].get(campo) for f in feats] if campo else []
    if len(feats) != len(ids) or (campo and set(obtidos) != set(ids)) or (ids and not campo):
        raise ValueError('IDs ou numero de feicoes divergentes')
    gravar(destino / 'dados.geojson', {'type': 'FeatureCollection', 'features': feats})
    gravar(destino / 'metadados.json', meta)
    return {'url': url, 'nome': meta['name'], 'feicoes': len(feats),
            'sha256': sha256_arquivo(destino / 'dados.geojson'), 'copyright': meta.get('copyrightText'),
            'licenca': 'acesso publico nao substitui termos de reutilizacao; revisar antes de admitir',
            'used_as_model_feature': False, 'lado_dique_validado': False,
            'historico_de_operacao_validado': False, 'quando_utc': agora_utc()}


def drenagem():
    pasta = SAIDA / 'referencias_hidraulicas'
    urls = [
        'https://gis-smamus.portoalegre.rs.gov.br/server/rest/services/01_PUBLICACOES/hidro_drenagem_superf_diag/FeatureServer/0',
        'https://geocuritiba.ippuc.org.br/server/rest/services/GeoCuritiba/Publico_Interno_GeoCuritiba_BaseCartografica_para_MC/MapServer/47',
    ]
    rows = []
    for i, url in enumerate(urls):
        try:
            r = camada(url, pasta / str(i))
        except Exception as exc:
            r = {'url': url, 'erro': str(exc)[:600], 'estado': 'indisponivel_nesta_consulta'}
        rows.append(r)
        print(r, flush=True)
    out = {'tipo': 'referencias_hidraulicas_publicas', 'quando_utc': agora_utc(), 'camadas': rows,
           'nao_e': 'rede cartografica nao prova capacidade de galeria, lado/cota de dique, funcionamento nem estado na data do evento'}
    gravar(pasta / 'manifesto.json', out)
    return out


def estruturas():
    pasta=SAIDA/'referencias_hidraulicas/v2_estruturas'
    urls=[
        'https://gis-smamus.portoalegre.rs.gov.br/server/rest/services/Hosted/Casas_de_Bomba___EBAP_210126/FeatureServer/0',
        'https://gis-smamus.portoalegre.rs.gov.br/server/rest/services/Hosted/Comportas_4D_210126/FeatureServer/0',
        'https://gis-smamus.portoalegre.rs.gov.br/server/rest/services/Hosted/Diques_170326/FeatureServer/0',
        'https://geocuritiba.ippuc.org.br/server/rest/services/GeoCuritiba/Publico_Interno_GeoCuritiba_BaseCartografica_para_MC/MapServer/28']
    rows=[]
    for i,url in enumerate(urls):
        try:
            r=camada(url,pasta/str(i))
        except Exception as exc:
            r={'url':url,'erro':str(exc)[:600],'estado':'indisponivel_nesta_consulta'}
        rows.append(r);print(r,flush=True)
    out={'tipo':'referencias_estruturas_hidraulicas','quando_utc':agora_utc(),'camadas':rows,
         'nao_e':'cadastro atual e projetos nao comprovam funcionamento na cheia historica nem poligono do lado protegido'}
    gravar(pasta/'manifesto.json',out)
    return out


def baixar_retomavel(url, destino, total, sha_esperado=None, reserva_disco=8*1024**3):
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + '.parcial')
    if destino.exists():
        if destino.stat().st_size != total or (sha_esperado and sha256_arquivo(destino) != 'sha256:' + sha_esperado):
            raise ValueError('arquivo existente diverge da origem congelada')
        return destino
    inicio = parcial.stat().st_size if parcial.exists() else 0
    if inicio > total:
        raise ValueError('parcial maior que origem')
    if shutil.disk_usage(destino.parent).free < total - inicio + reserva_disco:
        raise RuntimeError('disco insuficiente para download retomavel e margem de seguranca')
    for tentativa in range(4):
        inicio = parcial.stat().st_size if parcial.exists() else 0
        if inicio == total:
            break
        try:
            # parametro evita cache de URL assinada expirada sem mudar a revisao fixada.
            sep = '&' if '?' in url else '?'
            with requests.get(url + sep + f'download=true&retomada={inicio}', headers=UA | {'Range': f'bytes={inicio}-'},
                              timeout=(60, 180), stream=True) as r:
                r.raise_for_status()
                cr = r.headers.get('Content-Range', '')
                if r.status_code == 206 and not cr.startswith(f'bytes {inicio}-'):
                    raise ValueError('Range com inicio incorreto')
                if inicio and r.status_code != 206:
                    raise ValueError('servidor nao permite retomada')
                if cr and int(cr.split('/')[-1]) != total:
                    raise ValueError('origem mudou de tamanho')
                ultimo = inicio
                with parcial.open('ab' if inicio else 'wb') as f:
                    for b in r.iter_content(1024*1024):
                        f.write(b)
                        if f.tell() > total:
                            raise ValueError('download excedeu tamanho congelado')
                        if f.tell() - ultimo >= 512*1024**2:
                            ultimo = f.tell()
                            print('download', round(ultimo/1024**3, 2), '/', round(total/1024**3, 2), 'GiB', flush=True)
            if parcial.stat().st_size != total:
                raise OSError('fluxo incompleto')
            break
        except (requests.RequestException, OSError) as exc:
            print('retomada', tentativa + 1, str(exc)[:180], flush=True)
            if tentativa == 3:
                raise
            time.sleep(2**tentativa)
    if sha_esperado and sha256_arquivo(parcial) != 'sha256:' + sha_esperado:
        raise ValueError('SHA256 do arquivo completo diverge do LFS')
    parcial.replace(destino)
    return destino


def membro_gt(nome):
    p = PurePosixPath(nome)
    return (not p.is_absolute() and '..' not in p.parts and 'GT' in p.parts
            and p.name.endswith('_GT.tif') and p.name.split('_ID_')[0] in EVENTOS)


def urbano():
    pasta = SAIDA / 'urbansarfloods'
    a = ler(SAIDA / 'confirmacao/autorizacao.json')
    if not a.get('download_urbano_autorizado'):
        raise ValueError('download urbano nao autorizado')
    # LFS fornece tamanho e SHA256 do objeto na revisao congelada.
    url = f'https://huggingface.co/api/datasets/S1Floodbenchmark/UrbanSARFloods_v1/tree/{HF_REV}'
    info = next(x for x in get(url).json() if x['path'] == 'urban_sar_floods.tar.gz')
    total, sha = info['size'], info['lfs']['oid']
    pre = {'tipo': 'mt04_ampliacao_urbana', 'revisao': HF_REV, 'url': TAR, 'bytes': total, 'sha256_lfs': sha,
           'eventos': EVENTOS, 'grupos': 'Hebei_1 e Hebei_2 pertencem ao mesmo episodio; no maximo seis territorios',
           'licenca': 'CC-BY-4.0 declarada pelo produtor; rotulo semiautomatico',
           'regra': 'todas as GT destes eventos, sem selecao por desempenho; admitir negativo/mascaras/georreferencia antes de avaliar',
           'disco': 'arquivo comprimido temporario para retomada HTTP; eliminado apos extracao verificada do GT',
           'promovido': False, 'ajuste_na_reserva': False}
    arq_pre = pasta / 'pre_registro.json'
    if arq_pre.exists() and ler(arq_pre) != pre:
        raise ValueError('pre-registro urbano mudou')
    gravar(arq_pre, pre)
    archive = pasta / 'temporario/urban_sar_floods.tar.gz'
    pronto = pasta / 'extracao.json'
    if pronto.exists():
        r = ler(pronto)
        if all(sha256_arquivo(pasta / x['arquivo']) == x['sha256'] for x in r['rotulos']):
            return r
        raise ValueError('GT extraido alterado')
    baixar_retomavel(TAR, archive, total, sha)
    registros = []
    with tarfile.open(archive, mode='r|gz') as tar:
        for m in tar:
            if not m.isfile() or not membro_gt(m.name):
                continue
            dest = pasta / 'gt' / PurePosixPath(m.name).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, dest.with_suffix('.parcial').open('wb') as f:
                shutil.copyfileobj(src, f, 1024*1024)
            if dest.with_suffix('.parcial').stat().st_size != m.size:
                raise ValueError('extracao incompleta')
            dest.with_suffix('.parcial').replace(dest)
            registros.append({'arquivo': str(dest.relative_to(pasta)), 'membro': m.name,
                              'bytes': m.size, 'sha256': sha256_arquivo(dest)})
    if not registros:
        raise ValueError('nenhum GT elegivel; arquivo temporario preservado para diagnostico')
    r = {'tipo': 'extracao_GT_urbano', 'quando_utc': agora_utc(), 'pre_registro': sha256_arquivo(arq_pre),
         'rotulos': registros, 'bytes_GT': sum(x['bytes'] for x in registros),
         'estado': 'adquirido; admissao e avaliacao pendentes', 'archive_sha256': 'sha256:' + sha}
    gravar(pronto, r)
    # Apenas o arquivo temporario explicitamente produzido aqui; nenhuma remocao recursiva.
    if archive.resolve().parent != (pasta / 'temporario').resolve():
        raise ValueError('arquivo temporario fora da pasta autorizada')
    archive.unlink()
    return {'rotulos': len(registros), 'bytes_GT': r['bytes_GT'], 'estado': r['estado']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('acao', choices=['catalogo', 'drenagem', 'estruturas', 'urbano'])
    a = ap.parse_args()
    print(json.dumps({'catalogo': catalogo, 'drenagem': drenagem, 'estruturas': estruturas, 'urbano': urbano}[a.acao](), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
