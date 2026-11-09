"""Avalia a IA v2 sem retreino: paridade, grafo comum, consulta e percepção.
Reanálise posterior com escores conhecidos; não abre reserva ou confirmação nova.
"""
from pathlib import Path
import sys,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
REPO=Path(__file__).resolve().parents[2]
for folder in ('ia','metodo','servico','geo'):sys.path.insert(0,str(REPO/'scripts'/folder))
import ia_integrada_v2 as ia
import modelo_integrado_v2 as model
import consulta_territorial_v2 as territorial

OUT=ia.OUT/'fechamento_v1'
PACKAGE=ia.OUT/'pacote_v1'
R3=REPO/'local_runs/confirmacao-rodada3/v2'
MT11=REPO/'local_runs/mt11-transdutivo'
CFG={'bootstrap':2000,'semente':7,'nivel_ic':1-.05/3,'margem':.01,'minimo_componentes':8,
     'amostra_territorial_por_unidade':100,'unidades_territoriais':['EMSR498_AOI01','EMSR411_AOI01']}

def plan():
    files=[Path(__file__),REPO/'scripts/ia/modelo_integrado_v2.py',REPO/'scripts/ia/ia_integrada_v2.py',
      REPO/'scripts/ia/consulta_territorial_v2.py',REPO/'scripts/ia/servico_ia_integrada_v2.py',
      ia.Q_PATH,ia.F_PATH,MT11/'escores_emendado.parquet',
      R3/'grafo/nos.parquet',R3/'grafo/componentes.parquet',R3/'registro.json',
      R3/'reanalise/resultado_reparo.json',REPO/'local_runs/mt11-fechamento/v2/reparo_grafo_3b.json',
      REPO/'local_runs/confirmacao-rodada3/v1/confirmacao/predicoes.parquet']
    p={'tipo':'plano_avaliacao_pacote_v2','natureza':'posterior aos resultados; auditoria e reprodução retrospectiva, não confirmação nova',
       'config':CFG,'fontes':[ia.fonte(f) for f in files],
       'pesos':'Q e F congelados; percepção copiada do pai ativo; nenhum ajuste',
       'criterio_independencia_MT11':'grafo 3B; mesmo critério MT11: exclui ligado a desenvolvimento/examinado/reserva/MT04; outras confirmações não excluem, mas criam dependência entre estudos',
       'uso':'pesquisa e demonstração; sem promoção ou relaxamento do contrato legado',
       'parada':'todos os casos definidos, sem busca de efeito favorável'}
    target=OUT/'plano.json'
    if target.exists():
        if ia.ler(target)!=p:raise ValueError('plano/entradas mudaram: nova versão necessária')
    else:ia.gravar(target,p)
    return p

def auc(y,s):
    y=np.asarray(y);s=np.asarray(s,float);n1=int((y==1).sum());n0=int((y==0).sum())
    if not n1 or not n0 or not np.isfinite(s).all():raise ValueError('AUC sem suporte')
    return float((rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def resumo(d):
    d=d.sort_index();v=d.to_numpy(float);r={'n':len(v),'media':float(v.mean()) if len(v) else None,'a_favor':int((v>0).sum()),'por_unidade':d.to_dict()}
    if len(v)<CFG['minimo_componentes']:return {**r,'veredito':'not_testable'}
    rng=np.random.default_rng(CFG['semente']);bs=v[rng.integers(0,len(v),(CFG['bootstrap'],len(v)))].mean(axis=1)
    q=(1-CFG['nivel_ic'])/2;lo,hi=np.quantile(bs,[q,1-q])
    return {**r,'ic':[float(lo),float(hi)],'nivel_ic':CFG['nivel_ic'],
      'veredito':'sustentado' if v.mean()>=CFG['margem'] and lo>0 else ('nao_sustentado' if hi<CFG['margem'] else 'indecidivel')}

def avaliar_fisica(m):
    import mt11_fechamento as fechamento
    s,obs,_,_=fechamento.dados()
    checks={};pred={}
    for key,d in [('mt11',s),('rodada3',pd.read_parquet(REPO/'local_runs/confirmacao-rodada3/v1/confirmacao/predicoes.parquet'))]:
        with torch.no_grad():q=m(atributos_fisicos=m.tensores(d))['Q_conexao'].numpy()
        err=float(np.max(np.abs(q-d.Q_conexao.to_numpy())))
        if err>1e-10:raise ValueError('Q divergente '+key)
        checks[key]={'pontos':len(d),'erro_max_q':err}
        if key=='mt11':
            with torch.no_grad():f=m(atributos_fisicos=m.tensores(d[obs]),frequencia=torch.tensor(d.loc[obs,'F'].to_numpy(),dtype=torch.float64))['F_transf'].numpy()
            ef=float(np.max(np.abs(f-d.loc[obs,'F_transf'].to_numpy())))
            if ef>1e-10:raise ValueError('F divergente')
            s=s.copy();s['Q_conexao']=q;s.loc[obs,'F_transf']=f
            checks[key].update(pontos_historico=int(obs.sum()),erro_max_f=ef)
    n=pd.read_parquet(R3/'grafo/nos.parquet')
    excl=n.papeis.str.contains('desenvolvimento|examinado|reserva|confirmacao_MT04')
    mt=n[n.papeis.str.contains('confirmacao_MT11')].set_index('evento')
    expected=ia.ler(REPO/'local_runs/mt11-fechamento/v2/reparo_grafo_3b.json')['contrastes']
    contrasts={};metrics=[];max_auc=0.
    for name,col in [('emendada_primaria','componente_primario'),('sensibilidade_exutorio','componente_exutorio')]:
        mp=mt[col].where(~mt[col].isin(set(n.loc[excl,col])))
        z=s.assign(unidade=s.evento_id.map(mp)).dropna(subset=['unidade'])
        contrasts[name]={}
        for hypothesis,a,b,use_history in [('HQ1','Q_conexao','A0_regra_hand',False),('HQ2','Q_conexao','Q_base',False),('HF','F_transf','Q_conexao',True)]:
            t=z[obs.loc[z.index]] if use_history else z
            rows=[]
            for (unit,event),g in t.groupby(['unidade','evento_id']):
                if len(g)<20 or g.classe.nunique()!=2:continue
                aa=auc(g.classe,g[a]);bb=auc(g.classe,g[b])
                rows.append({'unidade':unit,'evento_id':event,'a':aa,'b':bb})
                metrics.append({'leitura':name,'hipotese':hypothesis,**rows[-1],'pontos':len(g)})
            table=pd.DataFrame(rows)
            delta=(table.groupby('unidade').a.mean()-table.groupby('unidade').b.mean())
            r=resumo(delta);ref=expected[name][hypothesis]
            if r['n']!=ref['n'] or abs(r['media']-ref['media'])>1e-12 or not np.allclose(r.get('ic'),ref.get('ic'),atol=1e-12,rtol=0) or r['veredito']!=ref['veredito']:
                raise ValueError('contraste não reproduz reparo '+name+'/'+hypothesis)
            contrasts[name][hypothesis]=r
    pd.DataFrame(metrics).to_parquet(OUT/'metricas_grafo_comum.parquet',index=False)
    r={'paridade':checks,'contrastes_grafo_3b':contrasts,'grafo_sha256':ia.sha(R3/'grafo/nos.parquet'),
       'regime':'reprodução retrospectiva após reparação; sem novas unidades, treino ou soma entre campanhas'}
    ia.gravar(OUT/'fisica.json',r)
    return r

def avaliar_territorio():
    d=pd.read_parquet(REPO/'local_runs/confirmacao-rodada3/v1/confirmacao/predicoes.parquet')
    results={};input_rows=[]
    for unit in CFG['unidades_territoriais']:
        g=d[d.nome==unit].sort_values('ponto_id').head(CFG['amostra_territorial_por_unidade'])
        if g.empty:raise ValueError('unidade definida sem dados: '+unit)
        points=g[['ponto_id','lat','lon']].to_dict('records')
        raw,prov=territorial.extrair(unit,points)
        diffs={}
        for col in model.NUCLEO:
            e=float(np.nanmax(np.abs(raw[col].to_numpy()-g[col].to_numpy())))
            if e>1e-7 or not np.array_equal(np.isfinite(raw[col]),np.isfinite(g[col])):raise ValueError('raster divergente '+unit+'/'+col+': '+str(e))
            diffs[col]=e
        pedido={'unidade':unit,'coordenadas':points}
        response=territorial.consultar_coordenadas(pedido,PACKAGE)
        scores=np.array([x['Q_conexao'] for x in response['pontos']])
        err=float(np.max(np.abs(scores-g.Q_conexao.to_numpy())))
        if err>1e-9:raise ValueError('consulta completa divergente')
        ia.gravar(OUT/f'pedido_{unit}.json',pedido);ia.gravar(OUT/f'resposta_{unit}.json',response)
        results[unit]={'pontos':len(g),'erro_atributos':diffs,'erro_max_escore':err,'sem_rotulo':not prov['rotulo_lido'],'piso_da_amostra':prov['piso_da_amostra']}
    ia.gravar(OUT/'equivalencia_territorial.json',results)
    return results

def avaliar_percepcao(m):
    import ia_integrada as old
    parent,parent_meta=old.carregar(old.RAIZ)
    _,inv,_,groups,_=old.protocolo_real(parent_meta)
    checked={}
    cache=old.resolver_pasta(old.RAIZ)/'avaliacao/resultado.json'
    expected=ia.ler(cache)['percepcao'] if cache.exists() else None
    for group,names in groups.items():
        values=old.medir_percepcao(m,[inv[n] for n in names])
        if expected:
            ref=expected[group]
            if abs(values['perda']-ref['perda'])>1e-8:raise ValueError('perda neural divergente')
            for event,row in values['por_evento'].items():
                if row['ap'] is not None and abs(row['ap']-ref['por_evento'][event]['ap'])>1e-10:raise ValueError('AP divergente')
        checked[group]=values
        print(json.dumps({'fase':'percepcao','grupo':group,'chips':len(names)}),flush=True)
    result={'grupos':checked,'comparacao_com_resultado_pai':expected is not None,'pesos_identicos':all(torch.equal(parent.percepcao.state_dict()[k],v) for k,v in m.percepcao.state_dict().items()),
       'regime':'mesmos dados conhecidos; nenhuma confirmação nova ou afirmação de benefício compartilhado'}
    ia.gravar(OUT/'percepcao.json',result)
    return result

def executar():
    OUT.mkdir(parents=True,exist_ok=True);plan();torch.set_num_threads(3)
    if not PACKAGE.exists():print(json.dumps(ia.exportar(PACKAGE),ensure_ascii=False),flush=True)
    m,meta=ia.carregar(PACKAGE)
    for name,call in [('fisica',lambda:avaliar_fisica(m)),('territorio',avaliar_territorio),('percepcao',lambda:avaliar_percepcao(m))]:
        marker=OUT/(name+'_concluido.json')
        if marker.exists():
            saved=ia.ler(marker)
            if saved['peso']!=meta['arquivos']['modelo.pt'] or saved['plano']!=ia.sha(OUT/'plano.json'):raise ValueError('etapa não corresponde ao pacote/plano')
            for rel,h in saved['saidas'].items():
                if ia.sha(OUT/rel)!=h:raise ValueError('saída alterada')
            continue
        before=set(OUT.glob('*.json'));call()
        after=set(OUT.glob('*.json'))
        outputs=after-before
        ia.gravar(marker,{'peso':meta['arquivos']['modelo.pt'],'plano':ia.sha(OUT/'plano.json'),
                         'saidas':{p.name:ia.sha(p) for p in outputs}})
        print(json.dumps({'fase':name,'estado':'concluido'}),flush=True)
    return {'pacote':str(PACKAGE),'avaliacao':str(OUT),'promovido':False,'treinos':0}

if __name__=='__main__':print(json.dumps(executar(),ensure_ascii=False))

