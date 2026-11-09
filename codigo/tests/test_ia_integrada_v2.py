from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/ia'))
import ia_integrada_v2 as ia
from modelo_integrado_v2 import IAIntegradaV2,atributos,BRUTAS

def configs():return {'base_percepcao':4,'q':ia.ler(ia.Q_PATH)['Q_conexao'],'f':ia.ler(ia.F_PATH)['exportado']}
def rows():return pd.DataFrame([dict(zip(BRUTAS,[1,4,2,.1,40,30,100,3])),dict(zip(BRUTAS,[5,9,8,np.nan,np.nan,2,400,7]))])
def reference(x,cfg):
    p=cfg['padronizador'];z={c:((x[c].fillna(p['mediana'][c])-p['media'][c])/p['desvio'][c]).to_numpy() for c in p['colunas']}
    v=sum(z[c]*b for c,b in cfg['coeficientes'].items())
    for t in cfg['interacoes']:v=v+z[t['colunas'][0]]*z[t['colunas'][1]]*t['peso']
    return v

@pytest.fixture
def package(tmp_path):
    cfg=configs();m=IAIntegradaV2(cfg)
    torch.save(m.state_dict(),tmp_path/'modelo.pt')
    pd.DataFrame([{'ponto_id':'x','lat':-29.,'lon':-51.,'F':.3,'n_pre':10,'evento_id':'synthetic','regra_unidade':'e1'}]).to_parquet(tmp_path/'historico_admitido.parquet',index=False)
    pd.DataFrame(columns=['ponto_id','lat','lon','construido_v2','areia_v2']).to_parquet(tmp_path/'cobertura_pontual.parquet',index=False)
    ia.gravar(tmp_path/'manifesto.json',{'versao':ia.VERSAO,'promovido':False,'config':cfg,'arquivos':{n:ia.sha(tmp_path/n) for n in ('modelo.pt','historico_admitido.parquet','cobertura_pontual.parquet')},'limites':['pesquisa']})
    return tmp_path

def test_q_f_equivalem_e_explicacao_e_completa():
    cfg=configs();m=IAIntegradaV2(cfg);d=rows();x=atributos(d)
    with torch.no_grad():o=m(atributos_fisicos=m.tensores(d),frequencia=torch.tensor([.2,.8],dtype=torch.float64))
    q=reference(x,cfg['q']);f=reference(pd.DataFrame({'Q':q,'F':[.2,.8]}),cfg['f'])
    np.testing.assert_allclose(o['Q_conexao'],q,rtol=0,atol=1e-12);np.testing.assert_allclose(o['F_transf'],f,rtol=0,atol=1e-12)
    assert torch.equal(o['contribuicoes_q'].sum(1),o['Q_conexao'])
    assert torch.equal(o['contribuicoes_historico'].sum(1),o['F_transf'])

def test_consulta_nao_depende_de_outros_pontos_ou_classe(package):
    a=rows().replace({np.nan:None}).to_dict('records');a[0].update(ponto_id='x',lat=-29.,lon=-51.)
    solo=ia.consultar({'pontos':[a[0]]},package)['pontos'][0]
    both=ia.consultar({'pontos':a},package)['pontos'][0]
    assert solo==both
    a[0]['classe']=1
    assert ia.consultar({'pontos':[a[0]]},package)['pontos'][0]==solo

def test_historico_ausente_nao_vira_zero_e_coordenada_confere(package):
    point=rows().iloc[0].to_dict();point.update(ponto_id='x',lat=-29.,lon=-51.)
    assert 'F_transf' in ia.consultar({'pontos':[point]},package)['pontos'][0]
    point['lon']=-50.
    r=ia.consultar({'pontos':[point]},package)['pontos'][0]
    assert r['historico']['estado']=='indisponivel_para_este_ponto' and 'F_transf' not in r

def test_falha_fisica_e_uso_nao_suportado_recusam(package):
    p=rows().iloc[0].to_dict();p['hand_m']=None
    assert ia.consultar({'pontos':[p]},package)['status']=='recusado'
    assert ia.consultar({'uso':'consulta_lugar_novo','pontos':[p]},package)['status']=='recusado'
    assert ia.consultar({'pontos':[p],'data_alvo':'2027-01-01'},package)['status']=='recusado'
    p['hand_m']=float('inf')
    assert ia.consultar({'pontos':[p]},package)['status']=='recusado'

def test_peso_adulterado_e_recusado(package):
    (package/'modelo.pt').write_bytes(b'alterado')
    with pytest.raises(ValueError,match='alterado'):ia.carregar(package)

def test_configuracao_adulterada_e_recusada(package):
    p=package/'manifesto.json';meta=ia.ler(p)
    meta['config']['q']['padronizador']['colunas'].reverse();ia.gravar(p,meta)
    with pytest.raises(ValueError,match='configuração'):ia.carregar(package)

def test_frequencia_fora_de_intervalo_nao_imputada():
    m=IAIntegradaV2(configs())
    with pytest.raises(ValueError,match='frequência'):m(atributos_fisicos=m.tensores(rows()),frequencia=torch.tensor([float('nan'),1.5]))

def test_raster_sem_piso_amostral_e_fora_da_grade(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from pyproj import Transformer
    import consulta_territorial_v2 as terrain
    unit=tmp_path/'unidade';unit.mkdir();(unit/'run_manifest.json').write_text('{}')
    tf=from_origin(500000,6800000,30,30)
    specs={'hand_dinf.tif':np.full((16,16),2.),'slope_deg_wbt.tif':np.arange(256,dtype=float).reshape(16,16)/100,
      'twi_dinf.tif':np.full((16,16),10.),'streams_dinf.tif':np.eye(16),'dinf_accum_cells.tif':np.full((16,16),12000.)}
    for name,data in specs.items():
        with rasterio.open(unit/name,'w',driver='GTiff',height=16,width=16,count=1,dtype='float32',crs='EPSG:32722',transform=tf,nodata=-9999) as f:f.write(data.astype('float32'),1)
    to_ll=Transformer.from_crs(32722,4326,always_xy=True)
    lon,lat=to_ll.transform(500045,6799955);badlon,badlat=to_ll.transform(499000,6801000)
    one,_=terrain.extrair('unidade',[{'lat':lat,'lon':lon}],tmp_path)
    two,prov=terrain.extrair('unidade',[{'lat':lat,'lon':lon},{'lat':badlat,'lon':badlon}],tmp_path)
    pd.testing.assert_series_equal(one.iloc[0],two.iloc[0]);assert pd.isna(two.iloc[1].hand_m)
    assert not prov['rotulo_lido'] and not prov['piso_da_amostra'] and pd.isna(one.iloc[0].areia_v2)
    with pytest.raises(ValueError,match='fora'):terrain.extrair('../unidade',[{'lat':lat,'lon':lon}],tmp_path)

def test_interface_preserva_maturidade_e_rotas(package):
    from servico_ia_integrada_v2 import criar_app,PedidoV2
    app=criar_app(package);paths=app.openapi()['paths']
    assert set(paths)=={'/ia/v2/estado','/ia/v2/consulta'}
    endpoint=next(r.endpoint for r in app.routes if getattr(r,'path',None)=='/ia/v2/consulta')
    assert endpoint(PedidoV2(uso='consulta_lugar_novo',pontos=[rows().iloc[0].to_dict()]))['status']=='recusado'

