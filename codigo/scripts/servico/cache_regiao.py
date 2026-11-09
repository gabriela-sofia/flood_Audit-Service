"""Cache descartavel de ponto-em-poligono, vinculado a coordenadas, indice e malha."""
import hashlib
import json
from pathlib import Path

VERSAO='macro_ponto_poligono_within_v1'


def calcular_macrorregiao(d,motor):
    """Mesmo predicado estrito, preparando cada UF uma vez e filtrando por bbox.

    Pontos contidos em mais de uma UF voltam ao STRtree original, conservando
    inclusive sua regra de desempate. Bordas continuam excluidas por within.
    """
    import numpy as np
    import pandas as pd
    from shapely import contains_xy,prepare
    from shapely.geometry import shape
    dados=json.loads(Path(motor.MALHA).read_text(encoding='utf-8'))
    lon=d.lon.to_numpy();lat=d.lat.to_numpy()
    uf=np.full(len(d),None,dtype=object);quantidade=np.zeros(len(d),dtype='uint16')
    for f in dados['features']:
        g=shape(f['geometry']);prepare(g);x0,y0,x1,y1=g.bounds
        ix=np.flatnonzero((lon>=x0)&(lon<=x1)&(lat>=y0)&(lat<=y1))
        dentro=ix[contains_xy(g,lon[ix],lat[ix])]
        uf[dentro]=motor.UF_DO_CODIGO.get(f['properties']['codarea']);quantidade[dentro]+=1
    resultado=pd.Series([motor.REGIAO_DA_UF.get(u) if u else None for u in uf],index=d.index,name='regiao_br')
    ambiguos=np.flatnonzero(quantidade>1)
    if len(ambiguos):resultado.iloc[ambiguos]=motor.com_regiao(d.iloc[ambiguos].copy()).regiao_br.to_numpy()
    return resultado


def identidade(d,malha,codigo=None):
    import pandas as pd
    h=hashlib.sha256()
    h.update(VERSAO.encode())
    h.update(pd.util.hash_pandas_object(d[['lat','lon']],index=True).values.tobytes())
    h.update(Path(malha).read_bytes())
    h.update(Path(__file__).read_bytes())
    if codigo is not None:h.update(Path(codigo).read_bytes())
    return h.hexdigest()


def ler_cache(d,malha,pasta,codigo=None):
    import pandas as pd
    try:
        chave=identidade(d,malha,codigo);p=Path(pasta)/f'{chave}.parquet'
        mf=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
        if mf['identidade']!=chave or hashlib.sha256(p.read_bytes()).hexdigest()!=mf['conteudo_sha256']:
            return None
        s=pd.read_parquet(p)
        if not s.index.equals(d.index) or len(s)!=len(d) or list(s.columns)!=['regiao_br']:
            return None
        return s.regiao_br
    except (OSError,ValueError,KeyError):return None


def gravar_cache(d,malha,pasta,s,codigo=None):
    """Somente depois do calculo original; falha de cache nao altera o resultado."""
    try:
        chave=identidade(d,malha,codigo);pasta=Path(pasta);pasta.mkdir(parents=True,exist_ok=True)
        p=pasta/f'{chave}.parquet';tmp=p.with_suffix('.parquet.tmp')
        s.to_frame('regiao_br').to_parquet(tmp);tmp.replace(p)
        mf={'tipo':'cache_ponto_poligono_macrorregiao','nao_e':'nova classificacao cientifica ou promocao',
            'versao':VERSAO,'identidade':chave,'linhas':len(d),'predicado':'within da implementacao original',
            'invalidacao':'hash das coordenadas+indice, conteudo integral da malha e codigo',
            'conteudo_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        p.with_suffix('.json').write_text(json.dumps(mf,indent=2),encoding='utf-8')
    except (OSError,ValueError):pass
