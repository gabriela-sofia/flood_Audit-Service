"""Variante fisica calculavel localmente nos casos CEMS recuperados, sem solo/cobertura imputados.

Novo desenvolvimento retrospectivo: nao confirma pesos Q ja usados em outros
estudos. Componente hidrologico retido em treino, escala e avaliacao.
"""
from __future__ import annotations
from comum import OUT, RUNS, agora, congelar, gravar, ler, rel, sha


def preparar():
    import pandas as pd
    ledger=RUNS/'planejamento-tese/v3-aoi/impedimentos/ativacoes_reabertas.csv'
    t=pd.read_csv(ledger);codes=sorted(t[t.estado_reaberto.eq('presente_ligavel_areal_e_terreno')].ativacao)
    nodes=pd.read_parquet(RUNS/'evolucao-acervo-local/v2/grafo_comum/grafo/nos.parquet').set_index('evento')
    grupos={}
    for c in codes:
        key=next((k for k in (f'CEMS:{c}',c) if k in nodes.index),None)
        if key is not None:grupos[c]=str(nodes.loc[key,'componente_primario'])
    return congelar(OUT/'fisica_local/protocolo.json',{
        'tipo':'protocolo_fisica_local_sem_cobertura','natureza':'desenvolvimento retrospectivo; referencias recuperadas de MT03 ja examinadas',
        'ativacoes':codes,'componentes':grupos,'preditores':['log1p_hand','log1p_declive','ln_sca'],
        'comparador':'-HAND no mesmo suporte','estimador':'logistica C1 balanced max_iter300; StandardScaler ajustado so no treino',
        'amostragem':'maximo1000 pontos por ativacao/classe, seed20261020; teste em todos pontos elegiveis do componente',
        'deduplicacao':'code,lat,lon; conflitos de classe removidos e contados; sem mistura de papeis',
        'avaliacao':'leave-one-componente-out; AUC por ativacao e media dentro de componente; IC95 descritivo',
        'fonte':rel(RUNS/'mecanismos-transferiveis/v3/cems_desenvolvimento/pontos.parquet'),
        'fonte_ledger_sha256':sha(ledger),'codigo_sha256':sha(__file__),
        'nao_e':'modelo promovido, Q congelado ou confirmacao independente nova'})


def executar():
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy import stats
    cfg=preparar();src=RUNS.parent/cfg['fonte']
    cols=['code','ponto_id','classe','hand_m','slope_deg','ln_sca','lat','lon','agua_permanente_jrc']
    d=pd.read_parquet(src,columns=cols,filters=[('code','in',cfg['ativacoes'])])
    n0=len(d);d=d[d.code.isin(cfg['componentes']) & d.classe.isin([0,1])].copy()
    fin=np.isfinite(d[['hand_m','slope_deg','ln_sca']]).all(axis=1)
    d=d[fin & d.agua_permanente_jrc.eq(0)].copy()
    cont=d.groupby(['code','lat','lon']).classe.nunique();bad=cont[cont>1].index
    conflicts=d.set_index(['code','lat','lon']).index.isin(bad)
    nconflito=int(conflicts.sum());d=d[~conflicts].drop_duplicates(['code','lat','lon']).copy()
    d['componente']=d.code.map(cfg['componentes'])
    if d.componente.nunique()<2:
        out={'estado':'not_testable','motivo':'menos de dois componentes locais','linhas':len(d)}
        gravar(OUT/'fisica_local/resultado.json',out);return out
    x=np.column_stack([np.log1p(d.hand_m.clip(lower=0)),np.log1p(d.slope_deg.clip(lower=0)),d.ln_sca.clip(-5,25)])
    y=d.classe.to_numpy();outrows=[];modelos=[]
    for comp in sorted(d.componente.unique()):
        sub=d[d.componente.ne(comp)];selected=[]
        for _,q in sub.groupby(['code','classe']):selected.extend(q.sample(n=min(1000,len(q)),random_state=20261020).index.tolist())
        pos=d.index.get_indexer(selected);test=d.componente.eq(comp).to_numpy()
        if set(d.iloc[pos].componente)&{comp}:raise ValueError('componente retido entrou no treino')
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=300,random_state=20261020))
        m.fit(x[pos],y[pos]);scores=m.decision_function(x[test]);te=d.loc[test].copy();te['score']=scores
        for code,q in te.groupby('code'):
            outrows.append({'ativacao':code,'componente':comp,'n':len(q),'positivos':int(q.classe.sum()),
                            'auc_local':float(roc_auc_score(q.classe,q.score)) if q.classe.nunique()==2 else None,
                            'auc_HAND':float(roc_auc_score(q.classe,-q.hand_m)) if q.classe.nunique()==2 else None})
        modelos.append({'retido':comp,'treino_componentes':sorted(sub.componente.unique()),'treino_pontos':len(pos),
                        'media':m[0].mean_.tolist(),'escala':m[0].scale_.tolist(),'coeficientes':m[1].coef_.tolist(),'intercepto':m[1].intercept_.tolist()})
        print(f'fisica local: componente retido {comp}',flush=True)
    tab=pd.DataFrame(outrows);gravar(OUT/'fisica_local/por_ativacao.csv',tab)
    tab['diferenca']=tab.auc_local-tab.auc_HAND;dif=tab.groupby('componente').diferenca.mean().dropna()
    media=float(dif.mean());n=len(dif);h=float(stats.t.ppf(.975,n-1)*dif.std(ddof=1)/np.sqrt(n)) if n>1 else None
    out={'quando_utc':agora(),'linhas_originais':n0,'linhas_elegiveis_deduplicadas':len(d),'linhas_com_conflito_excluidas':nconflito,
         'ativacoes':int(tab.ativacao.nunique()),'componentes':n,'delta_AUC_media_por_componente':media,
         'ic95_t_descritivo':[media-h,media+h] if h is not None else None,
         'minimo8_atingido':n>=8,'protocolo_sha256':sha(OUT/'fisica_local/protocolo.json'),
         'fonte_sha256':sha(src),'modelos':modelos,'promovido':False,
         'interpretacao':'variante de terreno calculavel sem GHSL/SoilGrids; resultado retrospectivo nos casos recuperados, nao confirmacao de Q'}
    gravar(OUT/'fisica_local/resultado.json',out);return out


if __name__=='__main__':
    from comum import bloquear_rede
    bloquear_rede();print(executar())
