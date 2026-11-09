"""Erros científicos perigosos: denominadores, reservas, pesos e integridade."""
import json
import sys
import zipfile
from pathlib import Path
import pytest
from shapely.geometry import box

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/social'))
import social05_contrato_medidas as s
import social06_consolidacao as build
import social07_servico_medidas as api

def test_razao_das_somas_e_cobertura_sem_media_de_percentuais():
    a=s.componentes({'V01031':1,'V01032':0,'V01006':2})
    b=s.componentes({'V01031':1,'V01032':0,'V01006':10})
    c=s.componentes({'V01031':'X','V01032':0,'V01006':10})
    r=s.agregar_componentes([a,b,c])['criancas_0_9']
    assert r['proporcao_universo_com_dado']==pytest.approx(2/12)
    assert r['setores_sem_par_completo']==1

@pytest.mark.parametrize('d,estado',[(None,'ausente_ou_suprimido'),(0,'universo_vazio')])
def test_denominador_ausente_ou_vazio_nao_e_baixa_vulnerabilidade(d,estado):
    r=s.proporcao([0],d)
    assert r['valor'] is None and r['estado']==estado

def test_numerador_superior_ao_universo_nao_vira_percentual():
    assert s.proporcao([11],10)['estado']=='inconsistente'

def ref():
    return {'id':'fixture','tipo':'agua_observada','data':'2024-01-01','fonte_sha256':'fixture',
            'uso':'desenvolvimento_exploratorio','estado_admissao':'candidata'}

def test_candidato_tem_limites_exploratorios_sem_contagem_confirmada():
    un=[{'id':'A','geometria':box(0,0,1,1),'populacao':100,'domicilios_ocupados':20}]
    r=s.exposicao(un,box(0,0,.5,1),ref())
    assert r['estado']=='exploratorio_nao_admitido'
    assert r['limites_localizacao']['populacao']['maximo']==100
    assert r['contagem_confirmada'] is None

def test_reserva_independente_recusada_antes_do_cruzamento():
    r=ref();r['uso']='reserva_modelo'
    with pytest.raises(ValueError,match='reserva'):s.exposicao([],box(0,0,1,1),r)

def ponto(codigo,peso=1,classe='seco',setor='A'):
    return {'codigo':codigo,'peso':peso,'classe':classe,'setor':setor,'evento':'E','cena':'S',
            'estrato_instrumento':'seco','situacao':'Rural','grupo_renda':'grupo',
            'referencia_admitida':False,'revisao_humana_validada':False}

def test_pesos_de_sorteio_e_referencia_nao_admitida():
    r=s.cobertura_amostral([ponto('1',9),ponto('2',1,'nao_sei')])
    assert r['medidas'][0]['fracao_anotavel_ponderada']==.9
    assert r['medidas'][0]['fracao_referencia_admitida']==0
    assert not r['desigualdade_confirmada']

def test_pontos_mesmo_setor_nao_contam_como_setores_independentes():
    r=s.cobertura_amostral([ponto('1'),ponto('2')])
    assert r['medidas'][0]['n']==2 and r['medidas'][0]['setores']==1

def test_duplicata_nao_aumenta_oportunidade():
    with pytest.raises(ValueError,match='duplicado'):s.cobertura_amostral([ponto('1')]*2)

@pytest.mark.parametrize('peso',[0,-1,float('nan'),float('inf')])
def test_peso_invalido_recusado(peso):
    with pytest.raises(ValueError):s.cobertura_amostral([ponto('1',peso)])

def test_referencia_nao_pode_ser_admitida_sem_validacao():
    p=ponto('1');p['referencia_admitida']=True
    with pytest.raises(ValueError,match='validação'):s.cobertura_amostral([p])

@pytest.mark.parametrize('header',['CD_setor','setor','CD_SETOR'])
def test_variantes_oficiais_chave_csv_nao_apagam_dados(tmp_path,header):
    p=tmp_path/'f.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('f.csv',f'{header};V00201\nA;5\nB;6\n')
    assert build.agregado_multiplos(p,{'A'})=={'A':{'V00201':'5'}}

def test_chave_desconhecida_nao_finge_supressao(tmp_path):
    p=tmp_path/'f.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('f.csv','identificador;V00201\nA;5\n')
    with pytest.raises(ValueError,match='desconhecida'):build.agregado_multiplos(p,{'A'})

def test_pacote_real_conferido_e_http_independente_da_fisica():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app=FastAPI();app.include_router(api.criar_router())
    with TestClient(app) as c:
        r=c.get('/social/medidas/saude')
        assert r.status_code==200
        r=c.post('/social/medidas/consulta',json={'regiao':'curitiba','aoi':{'lon_min':-49.3592469,'lon_max':-49.3565437,'lat_min':-25.6275203,'lat_max':-25.6207649}})
        assert r.status_code==200,r.text
        x=r.json();assert x['setores_intersectados']==4
        assert x['exposicao']['contagem_confirmada'] is None
        assert x['vulnerabilidade']['indice_validado'] is None
        assert x['prioridade_publica'] is None and not x['escore_fisico_alterado']
        assert x['exposicao']['contexto_aoi']['limites_localizacao']['populacao']['maximo']==2141
        assert c.post('/social/medidas/consulta',json={'regiao':'curitiba','admitir_sar':True}).status_code==422

def test_regiao_sem_cobertura_nao_e_zero_vulnerabilidade():
    x=api.consultar('curitiba',{'lon_min':0,'lat_min':0,'lon_max':1,'lat_max':1})
    assert x['vulnerabilidade']['estado']=='sem_cobertura'
    assert all(v['proporcao_universo_com_dado'] is None for v in x['vulnerabilidade']['componentes'].values())

def test_recorte_da_cobertura_amostral_exclui_pontos_fora_da_aoi():
    x=api.consultar('am_piloto',{'lon_min':0,'lat_min':0,'lon_max':1,'lat_max':1})
    assert x['cobertura_cientifica']['pontos']==0
    assert x['cobertura_cientifica']['estado']=='not_testable'

def test_integridade_recusa_arquivo_alterado_antes_de_servir(tmp_path):
    p=tmp_path/'f.json';p.write_text('{}',encoding='utf-8')
    h=api.sha(p)
    (tmp_path/'manifesto.json').write_text(json.dumps({'contrato':'social_medidas_v1','artefatos':{'f.json':h}}),encoding='utf-8')
    p.write_text('{"alterado":true}',encoding='utf-8')
    with pytest.raises(ValueError,match='alterado'):api.verificar(tmp_path)

def test_pacote_real_sem_anomalias_e_com_infraestrutura():
    m=api.verificar(api.PACOTE)
    assert sum(r['setores'] for r in m['regioes'].values())==9186
    for regiao in m['regioes']:
        d=api.ler(api.PACOTE/regiao/'vulnerabilidade.json')
        assert d['avaliacao_propria']['componentes_inconsistentes']==0
        assert d['componentes']['sem_agua_encanada']['setores_com_par_completo']>0

def test_grade_real_preserva_universo_e_candidato():
    r=api.estudo_adicional(api.GRADE,'exposicao_grade.json','social_grade_exposicao_v1')
    assert r['estado']=='exploratorio_nao_admitido'
    assert r['contagem_confirmada'] is None
    assert 'coletivos' in r['universo']['domicilios_ocupados']
    assert r['avaliacao_propria']['fracao_agua_coberta_grade']>0.99999
    assert r['limites_localizacao']['populacao']['minimo']==1131
    assert r['limites_localizacao']['populacao']['maximo']==16068

def test_contraste_sem_grupo_comparavel_nao_tem_diferenca():
    from social09_avaliacao_cobertura import contrastes
    p=ponto('1');p['grupo_renda']='ate_1212_responsavel_com_renda'
    r=contrastes([p])
    assert r['contrastes'][0]['diferenca_ate_menos_acima'] is None
    assert r['contrastes'][0]['comparabilidade']=='sem_suporte_comum'
    assert r['contrastes'][0]['desigualdade_referencia_admitida'] is None

def test_contrastes_condicionam_classe_instrumental():
    from social09_avaliacao_cobertura import contrastes
    a=ponto('1');a['grupo_renda']='ate_1212_responsavel_com_renda'
    b=ponto('2');b['grupo_renda']='acima_1212_responsavel_com_renda';b['estrato_instrumento']='agua'
    assert all(x['diferenca_ate_menos_acima'] is None for x in contrastes([a,b])['contrastes'])

def test_api_entrega_estudos_complementares_sem_promover_agua_ou_igualdade():
    rs=api.consultar('rs_emsr720')
    assert rs['exposicao']['estudo_grade_regional']['universo']['ano']==2022
    assert rs['exposicao']['contagem_confirmada'] is None
    am=api.consultar('am_piloto')
    assert am['cobertura_cientifica']['pontos']==400
    assert am['cobertura_cientifica']['avaliacao_regional_contrastes']['desigualdade_confirmada'] is False

def test_manifesto_nao_aceita_caminho_fora_da_pasta(tmp_path):
    fora=tmp_path/'segredo.json';fora.write_text('{}',encoding='utf-8')
    pasta=tmp_path/'pacote';pasta.mkdir()
    (pasta/'manifesto.json').write_text(json.dumps({'contrato':'social_medidas_v1','artefatos':{'../segredo.json':api.sha(fora)}}),encoding='utf-8')
    with pytest.raises(ValueError,match='caminho'):api.verificar(pasta)

def test_svc05_registra_novas_rotas_e_anexa_medidas_a_consulta_legada(monkeypatch):
    sys.path.insert(0,str(ROOT/'scripts/servico'))
    import svc05_transporte_http as svc
    from camada_social import unidades_demo
    monkeypatch.setattr(svc.servico,'unidades_sociais',{'curitiba':unidades_demo()})
    assert '/social/medidas/consulta' in svc.app.openapi()['paths']
    req=svc.RequisicaoSocialAOI(regiao='curitiba',aoi={'lon_min':-49.3592469,'lon_max':-49.3565437,'lat_min':-25.6275203,'lat_max':-25.6207649},fisica={'status':'ok','escore':.8,'maturidade':'mvp_local'})
    r=svc.rota_social_aoi(req)
    assert r['medidas_territoriais']['contrato']=='social_medidas_v1'
    assert r['eixos']['suscetibilidade_fisica']['escore_recebido']==.8
    assert r['medidas_territoriais']['prioridade_publica'] is None
