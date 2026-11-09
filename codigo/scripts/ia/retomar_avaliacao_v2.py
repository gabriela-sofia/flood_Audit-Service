"""Retoma somente a avaliação SAR; dependências efetivas conferidas por tarefa.
Cartões científicos mutáveis não entram na inferência/métrica neural. Não edita o pai.
"""
import copy
from pathlib import Path
import json
import avaliar_integrada_v2 as base
import ia_integrada_v2 as ia
import ia_integrada as old

def fontes_sar(meta):
    required={str((old.AB/n).resolve()) for n in ('combinado_s7/melhor.pt','combinado_s7/treino.json','protocolo.json','resultado.json')}
    required.add(str(old.CONT.resolve()))
    inventory=[f for f in meta['fontes'] if Path(f['arquivo']).name=='inventario.json']
    if len(inventory)!=1:raise ValueError('inventário SAR não é único')
    required.add(str(Path(inventory[0]['arquivo']).resolve()))
    chosen=[f for f in meta['fontes'] if str(Path(f['arquivo']).resolve()) in required]
    if {str(Path(f['arquivo']).resolve()) for f in chosen}!=required:raise ValueError('dependência SAR ausente')
    for f in chosen:old.conferir(f)
    return chosen

def executar():
    base.plan()
    _,meta=ia.carregar(base.PACKAGE)
    parent=meta['pai_percepcao_manifesto'];chosen=fontes_sar(parent)
    changes=[{**f,'atual':ia.sha(f['arquivo']) if Path(f['arquivo']).exists() else None}
             for f in parent['fontes'] if not Path(f['arquivo']).exists() or ia.sha(f['arquivo'])!=f['sha256']]
    if any(Path(f['arquivo']).name!='cartoes_cientificos.json' for f in changes):
        raise ValueError('divergência adicional não contemplada pelo adendo')
    record={'tipo':'adendo_operacional_dependencias_por_tarefa','script':ia.fonte(Path(__file__)),
        'plano':ia.sha(base.OUT/'plano.json'),'fontes_sar_conferidas':chosen,'contexto_mutavel_nao_consumido':changes,
        'mudanca':'protocolo SAR confere apenas pesos/dados/protocolos SAR; também preserva a conferência do manifesto de admissão',
        'nao_altera':'pesos, amostras, máscaras, grupos, métrica, critérios ou resultados físicos',
        'cartoes_antigos_modificados':False}
    path=base.OUT/'adendo_contexto_mutavel.json'
    if path.exists():
        if ia.ler(path)!=record:raise ValueError('adendo mudou')
    else:ia.gravar(path,record)
    original=old.protocolo_real
    def guarded(metadata):
        scoped=copy.deepcopy(metadata);scoped['fontes']=fontes_sar(metadata)
        return original(scoped)
    old.protocolo_real=guarded
    try:return base.executar()
    finally:old.protocolo_real=original

if __name__=='__main__':print(json.dumps(executar(),ensure_ascii=False))

