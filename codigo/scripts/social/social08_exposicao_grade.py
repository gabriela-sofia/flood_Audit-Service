"""Exposição exploratória na grade oficial, com incerteza e universo próprios."""
import json
from pathlib import Path
import geopandas as gpd
from social02_medidas_territoriais import REPO, sha, gravar
from social04_aquisicao_complementar import OUT as FONTES
from social05_contrato_medidas import exposicao

OUT=REPO/'local_runs/social-exposicao-grade/2026-10-12-v1'
CEMS=REPO/'local_runs/br-01-cems-negativo'

def executar():
    p=FONTES/'grade_id14.zip'
    origem=json.loads(p.with_suffix('.zip.proveniencia.json').read_text(encoding='utf-8'))
    if sha(p)!=origem['sha256']:raise ValueError('grade alterada')
    agua=gpd.read_parquet(CEMS/'positivo_inundacao.parquet').to_crs(4326)
    # Carrega todo o envelope com margem para o experimento de +/-30m.
    b=agua.total_bounds; bbox=(b[0]-.002,b[1]-.002,b[2]+.002,b[3]+.002)
    g=gpd.read_file(p,bbox=bbox,engine='pyogrio').to_crs(6933)
    w=agua.to_crs(6933).geometry.union_all()
    if g.empty or g.ID_UNICO.duplicated().any() or not g.geometry.is_valid.all():raise ValueError('grade inválida')
    area=g.geometry.area.sum();uniao=g.geometry.union_all()
    if abs(area-uniao.area)/area>1e-6:raise ValueError('células sobrepostas')
    if w.difference(uniao).area/w.area>1e-6:raise ValueError('bloco de grade não cobre toda a água candidata')
    un=[{'id':r.ID_UNICO,'geometria':r.geometry,'populacao':r.TOTAL,'domicilios_ocupados':r.TOTAL_DOM} for _,r in g.iterrows()]
    ref={'id':'CEMS:EMSR720','tipo':'agua_observada','data':'maio/2024; união de produtos, não instantâneo',
         'fonte_sha256':sha(CEMS/'positivo_inundacao.parquet'),'uso':'desenvolvimento_exploratorio','estado_admissao':'candidata'}
    r=exposicao(un,w,ref)
    r['universo']={'populacao':'população vinculada à grade pelo IBGE; geocodificação níveis 1 a 4',
                  'domicilios_ocupados':'particulares E coletivos; não igual a V0007 dos setores',
                  'populacao_sem_geocodificacao_local':'desconhecida; limite superior não cobre moradores excluídos dos níveis 5/6',
                  'ano':2022,'sem_localizacao_individual':True}
    r['avaliacao_propria']={'celulas_envelope':len(g),'dimensoes':{k:int(v) for k,v in g.ID_UNICO.str.extract(r'^(200M|1KM)')[0].value_counts().items()},
                           'fracao_agua_coberta_grade':float(w.intersection(uniao).area/w.area),
                           'sobreposicao_relativa':max(0,float((area-uniao.area)/area)),
                           'sensibilidade_borda':{}}
    for metros in (-30,30):
        r['avaliacao_propria']['sensibilidade_borda'][str(metros)]=exposicao(un,w.buffer(metros),ref)['limites_localizacao']
    r['limites_adicionais']=['água permanente não removida com validação neste pacote',
                            'delineação candidata não identifica perdas ou atingidos',
                            'limites espaciais não são intervalos de confiança; defasagem censitária mantida']
    gp=OUT/'grade_recorte.geojson';gravar(gp,json.loads(g.to_crs(4326).to_json(drop_id=True,na='null')))
    r['fontes']={'grade':origem,'agua_sha256':sha(CEMS/'positivo_inundacao.parquet'),
                 'nota_metodologica':'https://geoftp.ibge.gov.br/recortes_para_fins_estatisticos/grade_estatistica/censo_2022/Notas_metodologicas_grade_estatistica_2022.pdf'}
    r['grade_recorte_sha256']=sha(gp);r['codigo_sha256']=sha(__file__)
    gravar(OUT/'exposicao_grade.json',r)
    gravar(OUT/'manifesto.json',{'contrato':'social_grade_exposicao_v1','artefatos':{'grade_recorte.geojson':sha(gp),
           'exposicao_grade.json':sha(OUT/'exposicao_grade.json')},'promovido':False,'escore_fisico_alterado':False})
    return r['limites_localizacao']

if __name__=='__main__':print(json.dumps(executar(),ensure_ascii=False))
