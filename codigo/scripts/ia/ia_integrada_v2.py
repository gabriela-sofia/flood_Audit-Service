"""Pacote aditivo de pesquisa: exportação imutável e consulta sem rótulos.

Não altera o modelo ativo/contrato legado. Histórico só vem do catálogo admitido,
com identidade e coordenadas conferidas. Fora dele, F permanece indisponível.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from modelo_integrado_v2 import IAIntegradaV2,BRUTAS,NUCLEO,REPO,VERSAO
for name in ('metodo','servico','geo'):
    sys.path.insert(0,str(REPO/'scripts'/name))

OUT=REPO/'local_runs/ia-integrada-v2'
Q_PATH=REPO/'local_runs/mecanismos-transferiveis/v5/sem_amostra_v1/candidatas/pesos_e_padronizacao.json'
F_PATH=REPO/'local_runs/mt11-transdutivo/f_transf.json'

def ler(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gravar(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def fonte(p):return {'arquivo':str(Path(p).resolve().relative_to(REPO)),'sha256':sha(p)}


def exportar(destino=OUT/'pacote_v1'):
    import ia_integrada as anterior
    if Path(destino).exists():raise ValueError('destino já existe; não sobrescrever versão')
    old,meta=anterior.carregar(anterior.RAIZ)
    parent=anterior.resolver_pasta(anterior.RAIZ)
    q=ler(Q_PATH)['Q_conexao'];f=ler(F_PATH)['exportado']
    config={'base_percepcao':meta['base'],'q':q,'f':f}
    model=IAIntegradaV2(config);model.percepcao.load_state_dict(old.percepcao.state_dict())
    if any(not torch.equal(old.percepcao.state_dict()[k],v) for k,v in model.percepcao.state_dict().items()):raise ValueError('percepção mudou')
    import mt11_fechamento as mt
    s,obs,_,_=mt.dados()
    history=s[obs][['ponto_id','evento_id','lat','lon','F','n_pre','regra_unidade']].copy()
    if history.ponto_id.duplicated().any() or not history.F.between(0,1).all():raise ValueError('catálogo inválido')
    dst=Path(destino);dst.mkdir(parents=True)
    torch.save(model.state_dict(),dst/'modelo.pt')
    history.to_parquet(dst/'historico_admitido.parquet',index=False)
    cover_cols=['ponto_id','lat','lon','construido_v2','areia_v2']
    r3path=REPO/'local_runs/confirmacao-rodada3/v1/confirmacao/predicoes.parquet'
    cover=pd.concat([s[cover_cols],pd.read_parquet(r3path,columns=cover_cols)],ignore_index=True)
    if cover.ponto_id.duplicated().any():raise ValueError('identidade duplicada na cobertura')
    cover.to_parquet(dst/'cobertura_pontual.parquet',index=False)
    source=[Q_PATH,F_PATH,parent/'modelo.pt',parent/'manifesto.json',r3path,
      REPO/'local_runs/mt11-transdutivo/escores_emendado.parquet',
      REPO/'local_runs/mt11-transdutivo/resultado_emendado.json',
      REPO/'local_runs/confirmacao-rodada3/v2/registro.json',
      REPO/'local_runs/confirmacao-rodada3/v2/grafo/grafo.json',
      REPO/'local_runs/confirmacao-rodada3/v2/reanalise/resultado_reparo.json',
      REPO/'local_runs/mt11-fechamento/v1/registro.json',
      REPO/'local_runs/mt11-fechamento/v2/reparo_grafo_3b.json']
    manifest={'tipo':'ia_integrada_v2_pesquisa','versao':VERSAO,'quando_utc':datetime.now(timezone.utc).isoformat(),
      'config':config,'arquivos':{n:sha(dst/n) for n in ('modelo.pt','historico_admitido.parquet','cobertura_pontual.parquet')},
      'fontes':[fonte(p) for p in source],'codigo':[fonte(Path(__file__)),fonte(Path(__file__).with_name('modelo_integrado_v2.py'))],
      'historico':{'pontos':len(history),'eventos':int(history.evento_id.nunique()),
        'regra':'máscara de observabilidade e qualidade MT11 emendada; pelo menos 10 datas; catálogo retrospectivo conhecido',
        'temporalidade':'datas SAR anteriores ao evento; máscaras auxiliares posteriores, não avaliação prospectiva'},
      'fisica':{'origem':'Q_conexao MT05 congelado','features_brutas':list(BRUTAS),'piso_da_amostra':False,
        'populacao':'contrastes observados retrospectivos; não probabilidade territorial'},
      'percepcao':{'origem':str(parent.relative_to(REPO)),'modo':'posterior duplicado, sete canais; pesos idênticos ao pai'},
      'aprendizado':{'retreino':False,'representacao_compartilhada_entre_tarefas':False},
      'maturidade':'pesquisa_retrospectiva','promovido':False,'modelo_ativo_legado_alterado':False,
      'limites':['escores não são probabilidades, causa, profundidade ou previsão datada',
        'unificação preserva os estimadores; não é confirmação independente nova',
        'histórico do catálogo só no domínio observado; ausência não é frequência zero',
        'recusa por impossibilidade de entrada não certifica erro/confiabilidade fora do domínio']}
    # Mantém os metadados usados pelo leitor SAR legado, sem alterar seus arquivos.
    manifest['pai_percepcao_manifesto']=meta
    gravar(dst/'manifesto.json',manifest)
    return {'pasta':str(dst),'peso':manifest['arquivos']['modelo.pt'],'historico_pontos':len(history)}


def carregar(pasta=OUT/'pacote_v1'):
    root=Path(pasta);meta=ler(root/'manifesto.json')
    if meta['versao']!=VERSAO or meta['promovido'] is not False:raise ValueError('versão/maturidade incompatível')
    if set(meta['arquivos'])!={'modelo.pt','historico_admitido.parquet','cobertura_pontual.parquet'}:raise ValueError('inventário de pacote inválido')
    for name,h in meta['arquivos'].items():
        if sha(root/name)!=h:raise ValueError('artefato alterado: '+name)
    m=IAIntegradaV2(meta['config']);state=torch.load(root/'modelo.pt',weights_only=True,map_location='cpu')
    if not torch.equal(m.assinatura_config,state['assinatura_config']):raise ValueError('configuração não corresponde aos pesos')
    m.load_state_dict(state)
    if any(not torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('estado não finito')
    return m.eval(),meta


@torch.no_grad()
def consultar(pedido,pasta=OUT/'pacote_v1'):
    m,meta=carregar(pasta)
    base={'versao':VERSAO,'modelo_sha256':meta['arquivos']['modelo.pt'],'promovido':False,
          'maturidade':'pesquisa_retrospectiva','probabilidade_calibrada':False,'causa_identificada':False,'limites':meta['limites']}
    uso=pedido.get('uso','pesquisa_territorial')
    if uso not in ('pesquisa_territorial','demonstracao_retrospectiva') or pedido.get('data_alvo') or pedido.get('profundidade'):
        return {**base,'status':'recusado','motivo':'uso quantitativo validado, previsão ou profundidade não demonstrados nesta versão'}
    rows=pedido.get('pontos',[])
    if len(rows)>5000:raise ValueError('limite de 5000 pontos')
    out={**base,'status':'pesquisa','pontos':[]}
    cat=pd.read_parquet(Path(pasta)/'historico_admitido.parquet').set_index('ponto_id')
    for row in rows:
        d=pd.DataFrame([{c:row.get(c) for c in BRUTAS}])
        pid=row.get('ponto_id');response={'ponto_id':pid}
        cobertura=row.get('cobertura_terreno')
        if cobertura is not None:response['cobertura_terreno']=cobertura
        if d[list(NUCLEO)].isna().any(axis=None):
            motivo={'fora_da_grade':'ponto fora da grade do terreno da unidade; nenhuma métrica de área observada é atribuída',
                    'nodata_no_terreno':'terreno sem dado (nodata) no ponto; nenhum valor inventado'}.get(cobertura,'núcleo físico incompleto; nenhum valor inventado')
            out['pontos'].append({**response,'status':'recusado','motivo':motivo});continue
        try:t=m.tensores(d)
        except (ValueError,TypeError) as exc:
            out['pontos'].append({**response,'status':'recusado','motivo':str(exc)});continue
        h=None;motivo_historico=None
        if not pedido.get('usar_historico',True):motivo_historico='nao_solicitado'
        elif pid not in cat.index:motivo_historico='fora_do_catalogo_admitido'
        else:
            candidate=cat.loc[pid]
            if row.get('lat') is not None and row.get('lon') is not None and abs(float(row['lat'])-candidate.lat)<1e-9 and abs(float(row['lon'])-candidate.lon)<1e-9:h=candidate
            else:motivo_historico='coordenada_diverge_do_catalogo'
        freq=torch.tensor([h.F],dtype=torch.float64) if h is not None else None
        pred=m(atributos_fisicos=t,frequencia=freq)
        imputados=[c for c in BRUTAS if pd.isna(d[c].iloc[0])]
        response.update(status='pesquisa',Q_conexao=float(pred['Q_conexao'][0]),
          contribuicoes_q=dict(zip(m.fisica.nomes,pred['contribuicoes_q'][0].tolist())),
          atributos_imputados=imputados,
          confianca_local='não calibrada; não há intervalo de probabilidade por pixel',
          historico={'estado':'admitido_no_catalogo' if h is not None else 'indisponivel_para_este_ponto'})
        if motivo_historico:response['historico']['motivo']=motivo_historico
        if imputados:response['imputacao']=faixa_por_imputacao(m,d,imputados,freq)
        if h is not None:
            response.update(F_transf=float(pred['F_transf'][0]),contribuicoes_historico=dict(zip(m.historico.nomes,pred['contribuicoes_historico'][0].tolist())))
            response['historico'].update(frequencia=float(h.F),n_datas=int(h.n_pre),evento=h.evento_id,regra=h.regra_unidade)
        out['pontos'].append(response)
    if pedido.get('chip'):
        import ia_integrada as anterior
        inv=anterior.inventario(meta['pai_percepcao_manifesto'])
        if pedido['chip'] not in inv:raise ValueError('chip fora do inventário admitido')
        x,v,prov=anterior.cena(inv[pedido['chip']])
        p=m(cena=torch.from_numpy(x)[None])['logits_agua_observada'].softmax(1)[0,1].numpy()
        import rasterio
        with rasterio.open(inv[pedido['chip']]['caminhos']['jrc']) as src:
            support=v&(src.read(1)==0)&(src.read_masks(1)>0)
        out['observacao']={'chip':pedido['chip'],'pixels_validos':int(v.sum()),'pixels_fora_permanente':int(support.sum()),'escore_medio_no_suporte':float(p[support].mean()) if support.any() else None,
                          'rotulo_acessado':False,'procedencia':prov,'nao_e':'probabilidade calibrada ou causa'}
    if not rows and not pedido.get('chip'):raise ValueError('informe pontos ou chip')
    if rows and all(x['status']=='recusado' for x in out['pontos']) and 'observacao' not in out:out['status']='recusado'
    out['explicacao']='Contribuições somam os respectivos escores e descrevem o estimador, incluindo interações. Não atribuem causas físicas.'
    if any(x.get('imputacao') for x in out['pontos']):
        out['explicacao']+=' Termos que dependem de atributo imputado descrevem a mediana do treino, não o lugar; a faixa por imputação mostra quanto o escore muda no domínio válido do atributo ausente.'
    return out


# Domínio válido (unidades de entrada) dos atributos opcionais; usado só para mostrar a sensibilidade à imputação.
DOMINIO_OPCIONAL={'construido_v2':(0.,1.),'areia_v2':(0.,100.)}
FEATURE_DE={'construido_v2':'f_construido','areia_v2':'f_areia'}


def faixa_por_imputacao(m,d,imputados,freq=None):
    """Escore nos vértices do domínio dos atributos imputados (o escore é multilinear neles: os extremos estão nos
    vértices). Não é intervalo de confiança: é o quanto a ausência pode mover a ordenação."""
    import itertools
    opc=[c for c in imputados if c in DOMINIO_OPCIONAL]
    feats={FEATURE_DE[c] for c in opc}
    termos=[n for n in m.fisica.nomes if any(f in n.split('*') for f in feats)]
    if not opc:return {'atributos':imputados,'termos_dependentes':termos,'faixa':None}
    qs,fs=[],[]
    for vert in itertools.product(*[DOMINIO_OPCIONAL[c] for c in opc]):
        x=d.copy()
        for c,v in zip(opc,vert):x[c]=v
        p=m(atributos_fisicos=m.tensores(x),frequencia=freq)
        qs.append(float(p['Q_conexao'][0]))
        if freq is not None:fs.append(float(p['F_transf'][0]))
    r={'atributos':opc,'valor_imputado':'mediana do treino','termos_dependentes':termos,
       'Q_conexao_faixa':[min(qs),max(qs)],'nao_e':'intervalo de confiança ou probabilidade'}
    if fs:r['F_transf_faixa']=[min(fs),max(fs)]
    return r


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('acao',choices=['exportar','consultar']);ap.add_argument('--pasta',type=Path,default=OUT/'pacote_v1');ap.add_argument('--pedido',type=Path);ap.add_argument('--saida',type=Path)
    a=ap.parse_args();r=exportar(a.pasta) if a.acao=='exportar' else consultar(ler(a.pedido),a.pasta)
    if a.saida:gravar(a.saida,r)
    print(json.dumps(r,ensure_ascii=False,allow_nan=False))
