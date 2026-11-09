"""Consulta das três medidas auditáveis; independente do serviço físico."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from shapely.geometry import box, shape
from social02_medidas_territoriais import REPO, sha, limites_exposicao
from social05_contrato_medidas import INDICADORES, agregar_componentes, cobertura_amostral

PACOTE = REPO/'local_runs/social-consolidacao/2026-10-12-v3'
GRADE = REPO/'local_runs/social-exposicao-grade/2026-10-12-v1'
COBERTURA = REPO/'local_runs/social-avaliacao-cobertura/2026-10-12-v1'

def ler(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def verificar(pasta):
    pasta = Path(pasta); m = ler(pasta/'manifesto.json')
    if m.get('contrato') != 'social_medidas_v1': raise ValueError('contrato social incompatível')
    for nome,h in m['artefatos'].items():
        p = (pasta/nome).resolve()
        if not p.is_relative_to(pasta.resolve()) or sha(p)!=h: raise ValueError('artefato social alterado ou caminho inválido')
    return m

def estudo_adicional(pasta, arquivo, contrato):
    pasta=Path(pasta)
    if not (pasta/'manifesto.json').is_file():
        return {'estado':'not_testable','motivo':'estudo complementar ausente'}
    m=ler(pasta/'manifesto.json')
    if m.get('contrato')!=contrato:raise ValueError('contrato complementar incompatível')
    for nome,h in m['artefatos'].items():
        p=(pasta/nome).resolve()
        if not p.is_relative_to(pasta.resolve()) or sha(p)!=h:raise ValueError('estudo complementar alterado')
    return ler(pasta/arquivo)

@lru_cache(maxsize=4)
def _carregar(pasta, manifesto_hash):
    m = verificar(pasta)
    return m, {nome:ler(Path(pasta)/r['arquivo']) for nome,r in m['regioes'].items()}

def consultar(regiao, aoi=None, pasta=PACOTE):
    pasta=Path(pasta)
    # Hash de cada artefato conferido também em requisições seguintes; cache não mascara corrupção.
    m = verificar(pasta); _, dados = _carregar(str(pasta),sha(pasta/'manifesto.json'))
    if regiao not in dados: raise ValueError('região não coberta')
    zona=None
    if aoi is not None:
        vals=[aoi[k] for k in ('lon_min','lat_min','lon_max','lat_max')]
        import math
        if not all(math.isfinite(v) for v in vals) or not -180<=vals[0]<vals[2]<=180 or not -90<=vals[1]<vals[3]<=90:
            raise ValueError('AOI inválida')
        zona=box(*vals)
    features = dados[regiao]['features']
    subset = [f for f in features if zona is None or shape(f['geometry']).intersection(zona).area>0]
    cs=[]
    for f in subset:
        p=f['properties']; cs.append({n:{k:p.get(n+'_'+k) for k in ('numerador','denominador','valor','estado')} for n in INDICADORES})
    componentes = agregar_componentes(cs)
    # Exposição na AOI é apenas contexto censitário se não existe mancha elegível.
    limites=None
    if zona is not None:
        # Mesma reprojeção do GeoSeries.to_crs (pyproj always_xy), sem geopandas no serviço; ver social_projecao.
        from social_projecao import reprojetar
        geom = reprojetar([shape(f['geometry']) for f in subset], 4326, 6933)
        projzona = reprojetar([zona], 4326, 6933)[0]
        un = [{'id':f['properties']['CD_SETOR'],'geometria':g,
               **{k:f['properties'][k] for k in ('populacao','domicilios_ocupados')}} for f,g in zip(subset,geom)]
        limites=limites_exposicao(un,projzona,'recorte_de_estudo')
    exp=ler(pasta/'rs_emsr720/exposicao.json') if regiao=='rs_emsr720' else None
    grade=estudo_adicional(GRADE,'exposicao_grade.json','social_grade_exposicao_v1') if regiao=='rs_emsr720' else None
    cobertura={'estado':'not_testable','medidas':[], 'desigualdade_confirmada':False,
               'motivo':'sem estudo de oportunidade registrado nesta região'}
    if regiao=='am_piloto':
        cobertura=ler(pasta/'am_piloto/cobertura_cientifica.json')
        if zona is not None:
            from shapely.geometry import Point
            linhas=[r for r in ler(pasta/'am_piloto/oportunidades_pontos.json') if zona.covers(Point(r['lon'],r['lat']))]
            cobertura={**cobertura_amostral(linhas),'recorte':'pontos da amostra dentro da AOI',
                        'fonte_estudo': 'am_piloto/cobertura_cientifica.json'}
        cobertura['avaliacao_regional_contrastes']=estudo_adicional(COBERTURA,'avaliacao.json','social_cobertura_avaliacao_v1')
        cobertura['suporte_avaliacao_regional']='toda a amostra original; não somente pontos da AOI'
    return {'contrato':'social_medidas_v1','regiao':regiao,'ano_base':2022,'setores_intersectados':len(subset),
            'exposicao':{'estado':'not_testable' if exp is None else exp['estado'],
                         'contagem_confirmada':None,'contexto_aoi':limites,
                         'estudo_grade_regional':grade,
                         'estudo_exploratorio_regional':exp, 'suporte_estudo':'todo o pacote regional, não apenas a AOI' if exp else None},
            'vulnerabilidade':{'estado':'componentes_descritivos' if subset else 'sem_cobertura',
                              'componentes':componentes,'indice_validado':None,
                              'suporte':'totais/proporções dos setores intersectados, não das pessoas na parte da AOI'},
            'cobertura_cientifica':cobertura, 'prioridade_publica':None,
            'compreensao_humana_validada':False, 'escore_fisico_alterado':False,
            'fontes_manifesto_sha256':sha(pasta/'manifesto.json'),
            'limites':['sem inferência individual','2022 não é atualização populacional do evento',
                       'não estabelece prioridade pública ou perdas','sem índice composto ou média de medianas']}

def criar_router(pasta=PACOTE):
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, ConfigDict
    router=APIRouter(prefix='/social/medidas',tags=['medidas sociais'])
    class Entrada(BaseModel):
        model_config=ConfigDict(extra='forbid')
        regiao:str
        aoi:dict[str,float]|None=None
    # Sem anotações adiadas no parâmetro local: FastAPI precisa resolver o tipo.
    def consulta(req):
        try: return consultar(req.regiao,req.aoi,pasta)
        except (FileNotFoundError,ValueError,KeyError) as e:
            raise HTTPException(status_code=422 if isinstance(e,(ValueError,KeyError)) else 503, detail=str(e))
    consulta.__annotations__['req']=Entrada
    router.post('/consulta')(consulta)
    @router.get('/saude')
    def saude():
        try:
            m=verificar(pasta)
            return {'contrato':m['contrato'],'pronto':True,'regioes':m['regioes'],
                    'indice_vulnerabilidade_validado':False,'desigualdade_confirmada':False}
        except (FileNotFoundError,ValueError) as e:
            raise HTTPException(status_code=503,detail=str(e))
    return router
