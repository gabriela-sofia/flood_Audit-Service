"""Selecao contextual nos oito componentes recuperados, com orcamento pareado e sem classe na selecao."""
from __future__ import annotations
import math
from comum import OUT, RUNS, agora, congelar, gravar, ler, rel, sha


def preparar():
    import fisica_local
    f = fisica_local.preparar()
    return congelar(OUT/'contexto_local/protocolo.json',{
        'tipo':'protocolo_contexto_local_recuperado','natureza':'novo desenvolvimento retrospectivo; fontes MT03 ja examinadas',
        'ativacoes':f['ativacoes'],'componentes':f['componentes'],'fonte':f['fonte'],
        'K':6,'sementes':[20261019,20261020,20261021],
        'preditores':['log1p_HAND','log1p_declive','ln_sca'],
        'descritores':'log1p(mediana declive),log1p(p90 HAND) do raster inteiro decimado4; escala so nos doadores admissiveis',
        'proximidade':'distancia geodesica entre centros das grades; medias dentro do componente',
        'bracos':['contexto','proximidade','aleatorio','geral_orcamento'],
        'orcamento':'teto1000 por componente/classe; todos bracos reduzidos ao mesmo total por classe; geral limitado ao mesmo total',
        'modelo':'logistica C1 balanced max_iter300; StandardScaler so no treino',
        'primarias':['contexto-proximidade','contexto-geral_orcamento'],
        'regra':'media de AUC por ativacao, depois componente, depois seed; IC97.5 t descritivo; margem .01; minimo8 componentes',
        'selecao_por_rotulo_ou_resultado':False,'codigo_sha256':sha(__file__),
        'nao_e':'reexecucao do Q_conexao congelado ou confirmacao independente; variante fisica local sem solo/cobertura'})


def executar():
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.warp import transform_bounds
    from pyproj import Geod
    from scipy import stats
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    cfg=preparar();grupos=cfg['componentes'];rows=[]
    td=RUNS/'mecanismos-transferiveis/v3/cems_desenvolvimento/terreno'
    for code,comp in grupos.items():
        for p in sorted(td.glob(f'{code}*/hand_dinf.tif')):
            vals={}
            for col,file,q in [('hand','hand_dinf.tif',.9),('slope','slope_deg_wbt.tif',.5)]:
                with rasterio.open(p.parent/file) as s:
                    a=s.read(1,out_shape=(max(1,s.height//4),max(1,s.width//4)),masked=True).compressed()
                    a=a[np.isfinite(a)&(a>-1e4)]
                    if not len(a):raise ValueError('descritor sem terreno valido')
                    vals[col]=float(np.quantile(a,q))
                    b=transform_bounds(s.crs,4326,*s.bounds)
            rows.append({'code':code,'componente':comp,'unidade':p.parent.name,
                         'hand':np.log1p(max(vals['hand'],0)),'slope':np.log1p(max(vals['slope'],0)),
                         'lat':(b[1]+b[3])/2,'lon':(b[0]+b[2])/2})
    desc=pd.DataFrame(rows)
    if desc.empty:raise ValueError('terreno completo dos descritores nao encontrado')
    desc=desc.groupby('componente')[['hand','slope','lat','lon']].mean()
    gravar(OUT/'contexto_local/descritores.csv',desc.reset_index())
    cs=sorted(desc.index)
    if len(cs)<8 or len(cs)-1<cfg['K']:
        out={'estado':'not_testable','componentes':len(cs),'motivo':'minimo8 ou K6 inviavel'}
        gravar(OUT/'contexto_local/resultado.json',out);return out
    d=pd.read_parquet(RUNS.parent/cfg['fonte'],columns=['code','classe','lat','lon','hand_m','slope_deg','ln_sca','agua_permanente_jrc'],filters=[('code','in',cfg['ativacoes'])])
    d=d[d.classe.isin([0,1]) & d.agua_permanente_jrc.eq(0)].copy()
    d=d[np.isfinite(d[['hand_m','slope_deg','ln_sca']]).all(axis=1)].copy()
    conflicts=d.groupby(['code','lat','lon']).classe.nunique();bad=conflicts[conflicts>1].index
    d=d[~d.set_index(['code','lat','lon']).index.isin(bad)].drop_duplicates(['code','lat','lon']).reset_index(drop=True)
    d['componente']=d.code.map(grupos);d=d[d.componente.isin(cs)].reset_index(drop=True)
    x=np.column_stack([np.log1p(d.hand_m.clip(lower=0)),np.log1p(d.slope_deg.clip(lower=0)),d.ln_sca.clip(-5,25)])
    res=[];selecoes=[];geod=Geod(ellps='WGS84')
    for target in cs:
        donors=[c for c in cs if c!=target];z=desc.loc[donors,['hand','slope']]
        mu=z.mean();sd=z.std(ddof=0).replace(0,1)
        delta=(z-desc.loc[target,['hand','slope']])/sd
        contexto=list((delta**2).sum(axis=1).sort_values(kind='stable').index[:cfg['K']])
        prox=sorted(donors,key=lambda c:(geod.inv(desc.loc[target,'lon'],desc.loc[target,'lat'],desc.loc[c,'lon'],desc.loc[c,'lat'])[2],c))[:cfg['K']]
        test=d[d.componente.eq(target)];it=d.index.get_indexer(test.index)
        for seed in cfg['sementes']:
            rng=np.random.default_rng(seed);random=list(rng.choice(donors,size=cfg['K'],replace=False))
            conjuntos={'contexto':contexto,'proximidade':prox,'aleatorio':random,'geral_orcamento':donors}
            amostras={}
            for nome,cgs in conjuntos.items():
                sub=d[d.componente.isin(cgs)]
                if target in set(sub.componente):raise ValueError('componente alvo vazou')
                amostras[nome]=sub.groupby(['componente','classe'],group_keys=False).apply(lambda q:q.sample(n=min(1000,len(q)),random_state=seed),include_groups=False)
            # groupby.apply remove colunas de agrupamento; indices originais
            # permanecem e sao usados para montar as matrizes, sem reordenar alvo.
            totals={cl:min(int((d.loc[a.index,'classe']==cl).sum()) for a in amostras.values()) for cl in (0,1)}
            if min(totals.values())==0:raise ValueError('orcamento comum sem uma classe')
            for nome,sub in amostras.items():
                sel=np.concatenate([rng.choice(sub.index[d.loc[sub.index,'classe'].to_numpy()==cl],totals[cl],replace=False) for cl in (0,1)])
                m=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=300,random_state=seed))
                m.fit(x[sel],d.loc[sel,'classe']);scores=m.decision_function(x[it]);te=test.copy();te['score']=scores
                for code,q in te.groupby('code'):
                    res.append({'destino':target,'ativacao':code,'seed':seed,'braco':nome,'auc':float(roc_auc_score(q.classe,q.score)) if q.classe.nunique()==2 else None})
                selecoes.append({'destino':target,'seed':seed,'braco':nome,'doadores':conjuntos[nome],'pontos_por_classe':totals})
            print(f'contexto recuperado: {target}, seed{seed}',flush=True)
    r=pd.DataFrame(res);gravar(OUT/'contexto_local/por_ativacao.csv',r)
    gravar(OUT/'contexto_local/selecoes.json',{'selecoes':selecoes})
    tab=r.groupby(['destino','braco','seed']).auc.mean().groupby(['destino','braco']).mean().unstack();contrastes={}
    for b in ['proximidade','geral_orcamento','aleatorio']:
        dif=tab.contexto-tab[b];n=len(dif);m=float(dif.mean());h=float(stats.t.ppf(.9875,n-1)*dif.std(ddof=1)/math.sqrt(n))
        contrastes[f'contexto-{b}']={'media':m,'ic97_5_t_descritivo':[m-h,m+h],'componentes':n,
            'estado':'ganho_relevante_exploratorio' if n>=8 and m>=.01 and m-h>0 else 'perda_exploratoria' if n>=8 and m+h<0 else 'inconclusivo'}
    out={'quando_utc':agora(),'componentes':len(cs),'ativacoes':int(d.code.nunique()),'contrastes':contrastes,
         'natureza':cfg['natureza'],'protocolo_sha256':sha(OUT/'contexto_local/protocolo.json'),'promovido':False,
         'limite':'variante local e nova coorte retrospectiva; nao substitui os resultados contextuais Q_conexao v2'}
    gravar(OUT/'contexto_local/resultado.json',out);return out


if __name__=='__main__':
    import argparse
    from comum import bloquear_rede
    p=argparse.ArgumentParser();p.add_argument('--preparar',action='store_true');a=p.parse_args();bloquear_rede()
    print(preparar() if a.preparar else executar())
