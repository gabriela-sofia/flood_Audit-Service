"""Interface própria da candidata v2; preserva serviço/modelo ativo legados."""
from pathlib import Path
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from ia_integrada_v2 import OUT,carregar,consultar
from consulta_territorial_v2 import consultar_coordenadas

class PedidoV2(BaseModel):
    uso:str='pesquisa_territorial'
    pontos:list[dict]=Field(default_factory=list,max_length=5000)
    coordenadas:list[dict]=Field(default_factory=list,max_length=5000)
    unidade:str|None=None
    chip:str|None=None
    usar_historico:bool=True
    data_alvo:str|None=None
    profundidade:bool=False

def criar_app(pasta=OUT/'pacote_v1'):
    app=FastAPI(title='REV-P — IA integrada v2 de pesquisa',version='v2')
    @app.get('/ia/v2/estado')
    def estado():
        try:_,meta=carregar(pasta)
        except (OSError,ValueError,KeyError) as e:raise HTTPException(503,str(e)) from e
        return {k:meta[k] for k in ('versao','maturidade','promovido','fisica','historico','aprendizado')}
    @app.post('/ia/v2/consulta')
    def consulta(pedido:PedidoV2):
        try:
            if pedido.coordenadas:
                if not pedido.unidade or pedido.pontos or pedido.chip:raise ValueError('consulta por coordenadas exige unidade e não mistura outros modos')
                return consultar_coordenadas(pedido.model_dump(),pasta)
            return consultar(pedido.model_dump(),pasta)
        except (ValueError,KeyError,TypeError) as e:raise HTTPException(422,str(e)) from e
        except OSError as e:raise HTTPException(503,str(e)) from e
    return app

app=criar_app()
if __name__=='__main__':
    import argparse,uvicorn
    ap=argparse.ArgumentParser();ap.add_argument('--porta',type=int,default=8132);a=ap.parse_args()
    uvicorn.run(app,host='127.0.0.1',port=a.porta)
