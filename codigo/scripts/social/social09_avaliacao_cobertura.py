"""Contrastes de anotabilidade em oportunidades comparáveis do piloto aberto."""
from collections import defaultdict
import json
from pathlib import Path
from social02_medidas_territoriais import REPO, sha, gravar

PACOTE=REPO/'local_runs/social-consolidacao/2026-10-12-v3'
OUT=REPO/'local_runs/social-avaliacao-cobertura/2026-10-12-v1'

def contrastes(linhas):
    grupos=defaultdict(lambda:defaultdict(list))
    for r in linhas:
        if r['grupo_renda']=='sem_dado':continue
        grupos[(r['evento'],r['cena'],r['estrato_instrumento'],r['situacao'])][r['grupo_renda']].append(r)
    saida=[]
    for chave,gs in sorted(grupos.items()):
        baixo=gs.get('ate_1212_responsavel_com_renda',[]);alto=gs.get('acima_1212_responsavel_com_renda',[])
        def taxa(rs):
            return sum(r['peso'] for r in rs if r['classe']!='nao_sei')/sum(r['peso'] for r in rs) if rs else None
        tb,ta=taxa(baixo),taxa(alto)
        nb=len({r['setor'] for r in baixo if r['setor']});na=len({r['setor'] for r in alto if r['setor']})
        saida.append({'evento':chave[0],'cena':chave[1],'estrato_instrumento':chave[2],'situacao':chave[3],
                      'n_grupo_ate_limiar':len(baixo),'n_grupo_acima_limiar':len(alto),
                      'setores_grupo_ate_limiar':nb,'setores_grupo_acima_limiar':na,
                      'anotabilidade_grupo_ate_limiar':tb,'anotabilidade_grupo_acima_limiar':ta,
                      'diferenca_ate_menos_acima':tb-ta if tb is not None and ta is not None else None,
                      'comparabilidade': 'dois_grupos_presentes' if baixo and alto else 'sem_suporte_comum',
                      'replicacao_setorial_limitada':min(nb,na)<3,
                      'desigualdade_referencia_admitida':None,
                      'motivo_null':'nenhum ponto tem referência admitida; zero global de admissão não prova igualdade ou desigualdade'})
    return {'estimando':'diferença descritiva de anotabilidade visual por IA entre contextos de renda do responsável, condicionada a evento/cena/classe instrumental/situação',
            'contrastes':saida,'eventos_independentes':len({r['evento'] for r in linhas}),
            'desigualdade_confirmada':False,'causalidade':False,'efeito_institucional':None,
            'lacunas':['revisão humana cega','mais eventos e cenas com os dois grupos','plano de incerteza espacial antes da nova amostra',
                       'balanceamento da oportunidade/oclusão e mecanismo','renda 2022 não mede renda individual na cena de 2025']}

def executar():
    p=PACOTE/'am_piloto/oportunidades_pontos.json';m=json.loads((PACOTE/'manifesto.json').read_text(encoding='utf-8'))
    if sha(p)!=m['artefatos']['am_piloto/oportunidades_pontos.json']:raise ValueError('ledger alterado')
    r=contrastes(json.loads(p.read_text(encoding='utf-8')))
    r.update(ledger_sha256=sha(p),codigo_sha256=sha(__file__),promovido=False)
    gravar(OUT/'avaliacao.json',r)
    gravar(OUT/'manifesto.json',{'contrato':'social_cobertura_avaliacao_v1','artefatos':{'avaliacao.json':sha(OUT/'avaliacao.json')}})
    return r['contrastes']

if __name__=='__main__':print(json.dumps(executar(),ensure_ascii=False))
