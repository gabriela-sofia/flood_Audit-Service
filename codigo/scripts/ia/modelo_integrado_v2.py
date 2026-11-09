"""IA de pesquisa v2: Q congelado, histórico admitido e percepção no mesmo estado.

Não reentreina componentes nem representa identificação causal ou probabilidade.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/sar'))
from modelo_unet import UNetPequena

VERSAO='revp_ia_integrada_v2_pesquisa'
BRUTAS=('hand_m','ln_sca','slope_deg','construido_v2','areia_v2','dist_drenagem_m','dist_canal10_m','declive_p50')
FEATURES=('f_hand','f_sca','f_declive','f_construido','f_areia','f_dist_dren','f_dist_canal','z_relevo')
NUCLEO=('hand_m','ln_sca','slope_deg','dist_drenagem_m','dist_canal10_m','declive_p50')


def atributos(d):
    """Transformações MT05, calculáveis sem classe/evento ou piso da amostra."""
    missing=set(BRUTAS)-set(d.columns)
    if missing:raise ValueError('atributos ausentes: '+','.join(sorted(missing)))
    d=d[list(BRUTAS)].apply(pd.to_numeric,errors='raise')
    if np.isinf(d.to_numpy(float)).any():raise ValueError('infinito não é ausência')
    for name in ('dist_drenagem_m','dist_canal10_m','declive_p50'):
        if (d[name].dropna()<0).any():raise ValueError('valor físico inválido: '+name)
    for name,lo,hi in [('construido_v2',0,1),('areia_v2',0,100)]:
        if ((d[name].dropna()<lo)|(d[name].dropna()>hi)).any():raise ValueError('unidade/faixa inválida: '+name)
    return pd.DataFrame(dict(zip(FEATURES,[np.log1p(d.hand_m.clip(lower=0)),d.ln_sca.clip(-5,25),
        np.log1p(d.slope_deg.clip(lower=0)),d.construido_v2,d.areia_v2/100,
        np.log1p(d.dist_drenagem_m),np.log1p(d.dist_canal10_m),np.log1p(d.declive_p50)])),index=d.index)


class Exportado(nn.Module):
    """Parametrização aditiva exata, com interações explicitadas na explicação."""
    def __init__(self,exportado):
        super().__init__();self.colunas=tuple(exportado['padronizador']['colunas'])
        terms=list(exportado['coeficientes']);pairs=exportado['interacoes']
        self.nomes=tuple(terms)+tuple('*'.join(p['colunas']) for p in pairs)
        self.register_buffer('indices',torch.tensor([self.colunas.index(c) for c in terms],dtype=torch.long))
        self.register_buffer('pares',torch.tensor([[self.colunas.index(c) for c in p['colunas']] for p in pairs],dtype=torch.long).reshape(-1,2))
        self.register_buffer('pesos',torch.tensor(list(exportado['coeficientes'].values()),dtype=torch.float64))
        self.register_buffer('pesos_interacao',torch.tensor([p['peso'] for p in pairs],dtype=torch.float64))
        for name in ('media','desvio','mediana'):
            self.register_buffer(name,torch.tensor([exportado['padronizador'][name][c] for c in self.colunas],dtype=torch.float64))
        if not torch.isfinite(self.media).all() or not torch.isfinite(self.mediana).all() or not (self.desvio>0).all():raise ValueError('padronizador inválido')

    def termos(self,x):
        if x.ndim!=2 or x.shape[1]!=len(self.colunas):raise ValueError('dimensão incompatível')
        x=x.double()
        if torch.isinf(x).any():raise ValueError('infinito não admitido')
        z=(torch.where(torch.isnan(x),self.mediana,x)-self.media)/self.desvio
        main=z[:,self.indices]*self.pesos
        inter=z[:,self.pares[:,0]]*z[:,self.pares[:,1]]*self.pesos_interacao
        return torch.cat([main,inter],dim=1)

    def forward(self,x):return self.termos(x).sum(dim=1)


class IAIntegradaV2(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        signature=hashlib.sha256(json.dumps(config,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).digest()
        self.register_buffer('assinatura_config',torch.tensor(list(signature),dtype=torch.uint8))
        self.percepcao=UNetPequena(canais_entrada=7,base=config['base_percepcao'])
        self.fisica=Exportado(config['q'])
        self.historico=Exportado(config['f'])

    def forward(self,*,atributos_fisicos=None,frequencia=None,cena=None):
        if atributos_fisicos is None and cena is None:raise ValueError('nenhuma entrada')
        out={}
        if atributos_fisicos is not None:
            terms=self.fisica.termos(atributos_fisicos);q=terms.sum(dim=1)
            out.update(Q_conexao=q,contribuicoes_q=terms)
            if frequencia is not None:
                f=frequencia.double().reshape(-1)
                if len(f)!=len(q) or not torch.isfinite(f).all() or ((f<0)|(f>1)).any():raise ValueError('frequência inválida')
                xf=torch.stack([f if name=='F' else q for name in self.historico.colunas],dim=1)
                terms_f=self.historico.termos(xf)
                out.update(F_transf=terms_f.sum(dim=1),contribuicoes_historico=terms_f)
        if cena is not None:
            if cena.ndim!=4 or cena.shape[1]!=7 or min(cena.shape[2:])<8 or any(n%8 for n in cena.shape[2:]) or not torch.isfinite(cena).all():raise ValueError('cena inválida')
            out['logits_agua_observada']=self.percepcao(cena.float())
        return out

    def tensores(self,d):
        return torch.tensor(atributos(d)[list(self.fisica.colunas)].to_numpy(),dtype=torch.float64)
