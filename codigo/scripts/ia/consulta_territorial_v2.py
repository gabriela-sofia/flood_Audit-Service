"""Coordenadas → atributos Q do raster inteiro → IA v2, sem classe ou piso amostral."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from modelo_integrado_v2 import REPO
from ia_integrada_v2 import consultar,fonte
for folder in ('metodo','geo'):sys.path.insert(0,str(REPO/'scripts'/folder))
import mt02_atributos as conexao
import mt01_populacao as populacao

RAIZ=REPO/'local_runs/confirmacao-rodada3/v1/confirmacao/terreno'
CAMADAS={'hand_m':'hand_dinf.tif','slope_deg':'slope_deg_wbt.tif','twi_dinf':'twi_dinf.tif'}


def extrair(unidade,pontos,raiz=RAIZ):
    if not pontos or len(pontos)>5000:raise ValueError('informe entre 1 e 5000 pontos')
    root=Path(raiz).resolve();folder=(root/unidade).resolve()
    if not folder.is_relative_to(root) or folder==root or not (folder/'run_manifest.json').is_file():raise ValueError('unidade fora do acervo admitido')
    d=pd.DataFrame(pontos)
    if not {'lat','lon'}.issubset(d):raise ValueError('coordenadas ausentes')
    if not np.isfinite(d[['lat','lon']].to_numpy(float)).all() or not d.lat.between(-90,90).all() or not d.lon.between(-180,180).all():raise ValueError('coordenadas inválidas')
    paths=[folder/'run_manifest.json'];grids=[]
    cobertura=np.full(len(d),'dentro',dtype=object)
    for key,name in CAMADAS.items():
        path=folder/name;paths.append(path)
        with rasterio.open(path) as src:
            if src.crs is None or not src.crs.is_projected:raise ValueError('terreno sem CRS métrico')
            grids.append((src.crs,src.transform,src.shape))
            x,y=Transformer.from_crs(4326,src.crs,always_xy=True).transform(d.lon.to_numpy(float),d.lat.to_numpy(float))
            r,c=rasterio.transform.rowcol(src.transform,x,y);r=np.asarray(r);c=np.asarray(c)
            inside=(r>=0)&(c>=0)&(r<src.height)&(c<src.width)
            values=np.array([v[0] for v in src.sample(zip(x,y))],float)
            values[~inside]=np.nan
            invalido=values< -1e4
            if src.nodata is not None:invalido|=values==src.nodata
            sem_dado=inside&invalido
            if src.nodata is not None:values[values==src.nodata]=np.nan
            values[values < -1e4]=np.nan;d[key]=values
            cobertura[(cobertura=='dentro')&sem_dado]='nodata_no_terreno'
            cobertura[~inside]='fora_da_grade'
    if len(set(grids))!=1:raise ValueError('grades diferentes')
    d['cobertura_terreno']=cobertura
    d['ln_sca']=d.twi_dinf+np.log(np.tan(np.radians(d.slope_deg.clip(lower=.01))))
    a,b,lado=conexao.distancias_da_unidade(folder,d.lon.to_numpy(float),d.lat.to_numpy(float))
    d['dist_drenagem_m']=a;d['dist_canal10_m']=b
    ctx=populacao.contexto_das_unidades(pd.DataFrame([{'unidade':unidade,'familia_mdt':'fabdem','compativel':True,'caminho':str(folder)}]))
    d['declive_p50']=ctx.declive_p50.iloc[0]
    # Não interpola cobertura de pontos antigos para um novo ponto. Ausência fica explícita.
    for key in ('construido_v2','areia_v2'):
        d[key]=np.nan
    paths += [folder/'streams_dinf.tif',folder/'dinf_accum_cells.tif']
    prov={'unidade':unidade,'fontes':[fonte(p) for p in paths],
          'contexto':'quantil do raster inteiro decimado, passo 4, como MT01/MT03',
          'lado_km':float(lado),'area_truncavel':bool(lado<20),
          'rotulo_lido':False,'piso_da_amostra':False,
          'cobertura_solo':'não adquiridos para a consulta; imputação explícita pelas medianas do treino',
          'limite':'acervo selecionado por eventos conhecidos; cálculo executável não valida suscetibilidade territorial'}
    return d,prov


def consultar_coordenadas(pedido,pasta):
    d,prov=extrair(pedido['unidade'],pedido['coordenadas'])
    coverage=pd.read_parquet(Path(pasta)/'cobertura_pontual.parquet').set_index('ponto_id')
    matched=0
    for i,row in d.iterrows():
        pid=row.get('ponto_id')
        if pid in coverage.index:
            c=coverage.loc[pid]
            if abs(row.lat-c.lat)<1e-9 and abs(row.lon-c.lon)<1e-9:
                d.loc[i,['construido_v2','areia_v2']]=[c.construido_v2,c.areia_v2];matched+=1
    prov['cobertura_solo']='cache pontual apenas com identidade e coordenadas exatas; demais ausências explicitamente imputadas'
    prov['pontos_com_cache_de_cobertura']=matched
    prov['amostragem_local_de_cobertura']={'construido_v2':'equivalencia_nao_demonstrada','areia_v2':'equivalencia_nao_sustentada',
      'evidencia':'local_runs/evolucao-acervo-local/v2/cobertura/resumo.json','substituto_nao_usado':'WorldCover; interpolacao de pontos antigos'}
    clean=d.replace({np.nan:None}).to_dict('records')
    response=consultar({'uso':pedido.get('uso','pesquisa_territorial'),'pontos':clean,
                       'usar_historico':pedido.get('usar_historico',True),
                       'data_alvo':pedido.get('data_alvo'),'profundidade':pedido.get('profundidade',False)},pasta)
    response['procedencia_territorial']=prov
    return response
