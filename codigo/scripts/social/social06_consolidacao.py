"""Executa três estudos separados com dados reais e registra suas avaliações."""
from __future__ import annotations
import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
import pandas as pd
from shapely.geometry import box

from social02_medidas_territoriais import FONTES, REPO, gravar, numero, sha, verificar_fontes
from social04_aquisicao_complementar import OUT as NOVAS
from social05_contrato_medidas import INDICADORES, componentes, agregar_componentes, exposicao, cobertura_amostral

OUT = REPO / 'local_runs/social-consolidacao/2026-10-12-v3'
PILOTO = REPO / 'local_runs/instrumento-brasileiro/ib-03a'
CEMS = REPO / 'local_runs/br-01-cems-negativo'

def ler(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def fontes_novas():
    regs = {}
    for p in sorted(NOVAS.glob('*.proveniencia.json')):
        r = ler(p); arq = NOVAS/r['arquivo']
        if sha(arq) != r['sha256']: raise ValueError('fonte nova alterada: '+str(arq))
        regs[r['arquivo']] = r
    return regs

def agregado_multiplos(p, ids):
    """Leitura nacional em fluxo; guarda apenas setores selecionados."""
    with zipfile.ZipFile(p) as z:
        nomes = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if len(nomes) != 1: raise ValueError('CSV único obrigatório')
        with z.open(nomes[0]) as f:
            rd = csv.reader(io.TextIOWrapper(f, encoding='latin1'), delimiter=';')
            header = [c.upper() for c in next(rd)]
            chave = next((c for c in ('CD_SETOR','SETOR') if c in header),None)
            if chave is None: raise ValueError('coluna de setor desconhecida')
            pos = header.index(chave)
            wanted = {'V0001','V0007','V06006'} | {c for cs,d,_,_ in INDICADORES.values() for c in [*cs,d]}
            cols = [(i,c) for i,c in enumerate(header) if c in wanted]
            if not cols: raise ValueError('fonte sem variáveis esperadas')
            out = {}
            for row in rd:
                if not row: continue
                cid = row[pos]
                if cid in ids:
                    if cid in out: raise ValueError('duplicação de setor')
                    out[cid] = {c:row[i] for i,c in cols}
    return out

def definicoes(codigos):
    import openpyxl
    out = {}
    for p in sorted(FONTES.glob('dicionario*.xlsx')):
        w = openpyxl.load_workbook(p, read_only=True)
        for s in w.worksheets:
            for r in s.values:
                for i,c in enumerate(r[:-1]):
                    if str(c).upper() in codigos:
                        out[str(c).upper()] = {'descricao': str(r[i+1]), 'arquivo': p.name, 'aba': s.title}
        w.close()
    if set(out) != codigos: raise ValueError('definição oficial ausente: '+str(codigos-set(out)))
    return out

def validar_malha(g):
    if g.crs is None or g.empty or g.CD_SETOR.duplicated().any(): raise ValueError('malha vazia/sem CRS/duplicada')
    p = g.to_crs(6933)
    if not g.geometry.is_valid.all() or (p.geometry.area <= 0).any(): raise ValueError('polígono censitário inválido')
    # União versus soma detecta sobreposição indevida; pequenas diferenças numéricas são toleradas.
    a = p.geometry.area.sum(); u = p.geometry.union_all().area
    if abs(a-u)/a > 1e-6: raise ValueError('malha sobreposta: contagem duplicada')
    return {'setores': len(g), 'crs_origem': str(g.crs), 'sobreposicao_relativa': max(0,float((a-u)/a))}

def preparar_malhas():
    import geopandas as gpd  # só a construção do pacote precisa; o serviço e as funções puras não
    pts = ler(PILOTO/'pontos_cegos.json')
    bounds = (min(p['lon'] for p in pts), min(p['lat'] for p in pts),
              max(p['lon'] for p in pts), max(p['lat'] for p in pts))
    agua = gpd.read_parquet(CEMS/'positivo_inundacao.parquet').to_crs(4326)
    # Apenas o pacote de desenvolvimento já aberto; não consulta candidatos reservados.
    footprints = sorted((CEMS/'extraido').rglob('*imageFootprintA*.shp'))
    if not footprints: raise ValueError('sem footprint CEMS')
    frames = [gpd.read_file(p, engine='pyogrio').to_crs(4326) for p in footprints]
    suporte = gpd.GeoSeries(pd.concat([g.geometry for g in frames], ignore_index=True), crs=4326).union_all()
    rs = gpd.read_file(NOVAS/'RS_setores_CD2022.zip', bbox=suporte.bounds, engine='pyogrio')
    rs = rs[rs.intersects(suporte) & ~rs.intersection(suporte).is_empty].copy()
    am = gpd.read_file(NOVAS/'AM_setores_CD2022.zip', bbox=bounds, engine='pyogrio')
    am = am[am.intersects(box(*bounds))].copy()
    cur = gpd.read_file(FONTES/'PR_setores_CD2022.zip', where="CD_MUN = '4106902'", engine='pyogrio')
    return {'curitiba': cur, 'rs_emsr720': rs, 'am_piloto': am}, agua, suporte, footprints

def estudo_cobertura(am, tabelas):
    import geopandas as gpd
    sys.path.insert(0, str(REPO/'scripts/geo'))
    from obs05_referencia_brasileira import verificar_piloto
    verificar_piloto(PILOTO)
    pontos, anot = ler(PILOTO/'pontos_cegos.json'), ler(PILOTO/'anotacao_ia.json')
    desenho = ler(PILOTO/'chave_estratos.json'); estratos = {r['codigo']:r['estrato'] for r in desenho['pontos']}
    cont = Counter(estratos.values()); classes = {'seco':'0','agua':'1','ambiguo':'2'}
    # Correspondência original do diagnóstico IB-03a: 0 seco, 1 agua, 2 ambiguo.
    pre = ler(PILOTO/'pre_registro.json')
    if dict(cont) != pre['amostra']['por_estrato_da_classe_do_instrumento']: raise ValueError('desenho amostral divergente')
    cena = ler(PILOTO/'selecao.json')['escolhido']['s2'][0][0]
    g = gpd.GeoDataFrame(pontos, geometry=gpd.points_from_xy([p['lon'] for p in pontos],[p['lat'] for p in pontos]),crs=4326)
    j = gpd.sjoin(g, am[['CD_SETOR','CD_MUN','SITUACAO','geometry']].to_crs(4326), how='left', predicate='intersects')
    if j.codigo.duplicated().any(): raise ValueError('ponto em múltiplos setores; resolver borda explicitamente')
    rows = []
    for _,r in j.iterrows():
        setor = r.CD_SETOR if pd.notna(r.CD_SETOR) else None
        ren = numero(tabelas.get(setor,{}).get('V06006'))
        # Limiar apenas descritivo (1 SM de 2022 = R$1212); renda do responsável COM rendimentos.
        grupo = 'sem_dado' if ren is None else ('ate_1212_responsavel_com_renda' if ren <= 1212 else 'acima_1212_responsavel_com_renda')
        e = estratos[r.codigo]
        rows.append({'codigo':r.codigo, 'evento':'GDACS:1103101', 'cena':cena,
                     'lon':float(r.lon), 'lat':float(r.lat),
                     'estrato_instrumento':e, 'setor':setor,
                     'situacao':str(r.SITUACAO) if setor else 'fora_malha', 'grupo_renda':grupo,
                     'classe':anot[r.codigo]['classe'], 'motivo':anot[r.codigo]['motivo'],
                     'peso':desenho['pixels_por_classe_na_regiao'][classes[e]]/cont[e],
                     'referencia_admitida':False, 'revisao_humana_validada':False})
    resultado = cobertura_amostral(rows)
    resultado['motivos_nao_sei'] = dict(Counter(r['motivo'] for r in rows if r['classe']=='nao_sei'))
    resultado['fonte_renda'] = 'mediana setorial do responsável com rendimento; limiar descritivo R$1212 (SM/2022), sem classificar pessoas'
    resultado['avaliacao_propria'] = {'sorteio_original_conferido':True, 'cenas':1, 'eventos':1,
          'pontos_no_mesmo_setor_nao_sao_replicas_independentes':True,
          'inelegibilidade': ['referência por IA sem revisão humana', 'sem replicação por evento',
                             'amostra por classe SAR; comparações condicionadas ao estrato', 'renda de 2022, cena de 2025'],
          'sensibilidade_limiar_renda':[]}
    # Análise de sensibilidade: thresholds mantêm cada estrato instrumental separado.
    for corte in (606, 1212, 2424):
        copia = []
        for r in rows:
            rr = dict(r); ren = numero(tabelas.get(r['setor'],{}).get('V06006'))
            rr['grupo_renda'] = 'sem_dado' if ren is None else ('ate_limiar' if ren <= corte else 'acima_limiar')
            copia.append(rr)
        resultado['avaliacao_propria']['sensibilidade_limiar_renda'].append({'limiar':corte,'medidas':cobertura_amostral(copia)['medidas']})
    return resultado, rows

def executar(saida=OUT):
    saida = Path(saida); fontes = {**verificar_fontes(FONTES), **fontes_novas()}
    malhas, agua, suporte, footprints = preparar_malhas()
    qa = {k:validar_malha(g) for k,g in malhas.items()}
    ids = set().union(*(set(g.CD_SETOR) for g in malhas.values()))
    arqs = [FONTES/'Agregados_por_setores_basico_BR_20260520.zip',
            FONTES/'Agregados_por_setores_renda_responsavel_BR_20260508_csv.zip',
            NOVAS/'Agregados_por_setores_demografia_BR.zip',
            NOVAS/'Agregados_por_setores_caracteristicas_domicilio1_BR.zip',
            NOVAS/'Agregados_por_setores_caracteristicas_domicilio2_BR_20250417.zip']
    tabelas = {cid:{} for cid in ids}
    for p in arqs:
        dados = agregado_multiplos(p,ids)
        if not dados: raise ValueError('fonte sem setores do estudo: '+p.name)
        for cid,row in dados.items():
            for c,v in row.items():
                if c.startswith('V'):
                    if c in tabelas[cid]: raise ValueError('código sobreposto entre fontes: '+c)
                    tabelas[cid][c] = v
    codigos = {'V0001','V0007','V06006'} | {c for cs,d,_,_ in INDICADORES.values() for c in [*cs,d]}
    defs = definicoes(codigos)
    manifest = {'contrato':'social_medidas_v1', 'ano_censo':2022, 'fontes':fontes, 'definicoes':defs,
                'codigo_sha256':{Path(p).name:sha(p) for p in (__file__, REPO/'scripts/social/social05_contrato_medidas.py')},
                'regioes':{}, 'escore_fisico_alterado':False, 'promovido':False,
                'avaliacao_prioridade_publica':'nao_executada', 'avaliacao_compreensao_humana':'nao_executada'}
    por_setor = {cid:componentes(row) for cid,row in tabelas.items()}
    for nome,g in malhas.items():
        rows = []
        for cid in g.CD_SETOR:
            row = tabelas[cid]
            for c in ('V0001','V0007'):
                if numero(row.get(c)) is None: raise ValueError('básico sem contagem '+cid)
            r = {'CD_SETOR':cid,'populacao':numero(row['V0001']), 'domicilios_ocupados':numero(row['V0007']),
                 'renda_mediana_responsavel_com_rendimento':numero(row.get('V06006'))}
            for ind,vals in por_setor[cid].items():
                for c in ('numerador','denominador','valor','estado'): r[ind+'_'+c]=vals[c]
            rows.append(r)
        out = g[['CD_SETOR','CD_MUN','NM_MUN','SITUACAO','geometry']].merge(pd.DataFrame(rows),on='CD_SETOR',validate='one_to_one')
        arq = saida/nome/'setores.geojson'; gravar(arq,json.loads(out.to_crs(4326).to_json(drop_id=True,na='null')))
        comp = agregar_componentes([por_setor[cid] for cid in g.CD_SETOR])
        # Reconcilia universos; divergência entre básico e demografia não desaparece por renomeação.
        dif = [cid for cid in g.CD_SETOR if numero(tabelas[cid].get('V01006')) != numero(tabelas[cid].get('V0001'))]
        detalhe = {'setores':len(g), 'populacao_base':sum(r['populacao'] for r in rows),
                   'domicilios_ocupados_base':sum(r['domicilios_ocupados'] for r in rows),
                   'renda_setores_sem_dado':sum(r['renda_mediana_responsavel_com_rendimento'] is None for r in rows),
                   'componentes':comp, 'indice_vulnerabilidade_validado':False,
                   'avaliacao_propria':{**qa[nome], 'setores_demografia_diferente_basico':len(dif),
                     'setores_demografia_diferente_basico_ids':dif,
                     'componentes_inconsistentes':sum(v['estado']=='inconsistente' for cid in g.CD_SETOR for v in por_setor[cid].values()),
                     'validacao_de_impacto':False},
                   'geometria_sha256':sha(arq),'arquivo':str(arq.relative_to(saida)).replace('\\','/')}
        gravar(saida/nome/'vulnerabilidade.json',detalhe)
        manifest['regioes'][nome] = {'arquivo':detalhe['arquivo'],'sha256':sha(arq),'municipios':sorted(set(g.CD_MUN)),
                                    'setores':len(g),'vulnerabilidade_sha256':sha(saida/nome/'vulnerabilidade.json')}
    g = malhas['rs_emsr720'].to_crs(6933)
    unidades = [{'id':r.CD_SETOR,'geometria':r.geometry,'populacao':numero(tabelas[r.CD_SETOR]['V0001']),
                 'domicilios_ocupados':numero(tabelas[r.CD_SETOR]['V0007'])} for _,r in g.iterrows()]
    ref = {'id':'CEMS:EMSR720','tipo':'agua_observada','data':'2024-05-02 (início; produtos em múltiplas datas)',
           'fonte_sha256':sha(CEMS/'positivo_inundacao.parquet'),'uso':'desenvolvimento_exploratorio',
           'estado_admissao':'candidata', 'geometria':'união espacial de delineações; não um instantâneo simultâneo'}
    w = agua.to_crs(6933).geometry.union_all()
    exp = exposicao(unidades,w,ref)
    exp['avaliacao_propria'] = {'area_agua_km2':w.area/1e6, 'controle_sem_alocacao_por_area':True,
        'sensibilidade_suporte':{}}
    for metros in (-30,30):
        pol = w.buffer(metros)
        if not pol.is_empty:
            r = exposicao(unidades,pol,ref)
            exp['avaliacao_propria']['sensibilidade_suporte'][str(metros)] = r['limites_localizacao']
    exp['fontes_sha256'] = {'resumo':sha(CEMS/'resumo.json'),**{str(p.relative_to(CEMS)):sha(p) for p in footprints}}
    exp['limite_adicional'] = 'sem remoção de água permanente validada neste pacote; limite espacial exploratório não equivale a moradores atingidos por cheia'
    gravar(saida/'rs_emsr720/exposicao.json',exp)
    cov, ledger = estudo_cobertura(malhas['am_piloto'],tabelas)
    cov['fontes_sha256'] = {n:sha(PILOTO/n) for n in ('pontos_cegos.json','anotacao_ia.json','anotacao_congelada.json','chave_estratos.json','pre_registro.json','selecao.json')}
    gravar(saida/'am_piloto/cobertura_cientifica.json',cov); gravar(saida/'am_piloto/oportunidades_pontos.json',ledger)
    manifest['medidas'] = {'exposicao':'rs_emsr720/exposicao.json','vulnerabilidade':'por região em vulnerabilidade.json',
                           'cobertura_cientifica':'am_piloto/cobertura_cientifica.json'}
    manifest['artefatos'] = {str(p.relative_to(saida)).replace('\\','/'):sha(p) for p in sorted(saida.rglob('*.json')) if p.name!='manifesto.json'}
    # GeoJSON também deve ser protegido no leitor de serviço.
    manifest['artefatos'].update({str(p.relative_to(saida)).replace('\\','/'):sha(p) for p in sorted(saida.rglob('*.geojson'))})
    gravar(saida/'manifesto.json',manifest)
    return {'regioes':{k:v['setores'] for k,v in manifest['regioes'].items()},
            'exposicao':exp['limites_localizacao'], 'cobertura':{'pontos':cov['pontos'],'sem_setor':cov['pontos_sem_setor']},
            'indice_validado':False,'desigualdade_confirmada':False}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--saida',type=Path,default=OUT)
    print(json.dumps(executar(ap.parse_args().saida),ensure_ascii=False))
