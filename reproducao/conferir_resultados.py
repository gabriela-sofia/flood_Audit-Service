"""Confere resultados estruturados selecionados contra as tabelas do artigo.

Não reamostra, ajusta, baixa dados ou altera artefatos. O esperado vem dos
artefatos de execução incluídos; a fonte LaTeX é a apresentação conferida.
"""
from __future__ import annotations
import json
import re
from acervo import RAIZ, ler_json

def conferir():
    tex=(RAIZ/'artigo/fonte/artigo.tex').read_text(encoding='utf-8')
    tabelas=re.findall(r'\\begin\{resulttable\}\{[^}]*\}\{[^}]*\}(.*?)\\end\{resulttable\}',tex,re.S)
    romano={s:t for s,t in zip(['I','II','III','IV','V','VI','VII','VIII','IX'],tabelas)}
    checks=[]
    def valor(id,v,tabela,casas=3):
        escrito=f'{abs(float(v)):.{casas}f}'.replace('.',',')
        ok=bool(re.search(r'(?<![\d,])'+re.escape(escrito)+r'(?!\d)',romano[tabela]))
        checks.append({'id':id,'valor_artefato':float(v),'apresentacao':escrito,'tabela':tabela,'ok':ok})
    m=ler_json('local_runs/mecanismos-transferiveis/v1/resultado.json')['contrastes']
    for k,nome in [('G-A0_regra_hand','HAND'),('G-A1_regra_composta','pesos_iguais')]:
        valor('origem_'+nome,m[k]['delta_medio'],'II')
        for i,x in enumerate(m[k]['ic95']):valor(f'origem_{nome}_IC{i}',x,'II')
    mt=ler_json('local_runs/mt11-fechamento/v2/reparo_grafo_3b.json')['contrastes']['emendada_primaria']
    for k in ('HQ1','HQ2'):
        valor(k,mt[k]['media'],'II')
        for i,x in enumerate(mt[k]['ic']):valor(f'{k}_IC{i}',x,'II')
    f03=ler_json('local_runs/fundamentacao-artigo-20261028/f03_ic_transferencia_coortes.json')['por_coorte']
    for c in ('rmpa','manaus','recife','rmr'):
        r=f03[c]
        valor(c+'_FCU_AUC',r['auc']['Q_sel_ext'],'III')
        valor(c+'_FCU_HAND',r['Q_menos_A0_hand']['delta'],'III')
        for i,x in enumerate(r['ic95_auc_Q']):valor(f'{c}_AUC_IC{i}',x,'III')
    f07=ler_json('local_runs/fundamentacao-artigo-20261106/f07_estratos_coortes.json')['por_coorte']
    for c in ('rmpa','manaus','curitiba','recife','rmr'):
        valor(c+'_coorte',f07[c]['todas']['auc']['Q_sel_ext'],'III')
        valor(c+'_coorte_HAND',f07[c]['todas']['Q_menos_A0_hand']['delta'],'III')
        for estrato in ('baixa_renda','comparador'):
            valor(c+'_'+estrato,f07[c][estrato]['auc']['Q_sel_ext'],'III')
    ruas=ler_json('local_runs/fundamentacao-artigo-20261028/f06_ic_auc_ruas_recife.json')
    valor('ruas_modelo',ruas['funcao_transferida']['auc_dentro_do_bairro'],'IV')
    valor('ruas_HAND',ruas['hand']['auc_dentro_do_bairro'],'IV')
    pa=ler_json('local_runs/fundamentacao-artigo-20261028/f04_encosta_e_retencao.json')
    for k in ('idai','agua'):valor('Petropolis_'+k,pa['petropolis']['auc_por_cicatriz'][k],'V')
    valor('Petropolis_contraste',pa['petropolis']['idai_menos_agua']['delta'],'V')
    valor('retencao_RMPA',pa['retencao_rmpa']['T_B_media_das_unidades'],'VI')
    fa=ler_json('local_runs/fundamentacao-artigo-20261028/f01_ic_ganho_atualizacao.json')['por_coorte']['rmpa']
    for k in ('margem_v9_gate','uniforme_gate'):valor('atualizacao_'+k,fa[k]['ganho'],'VI')
    ps=ler_json('local_runs/transferencia-vulnerabilizados/v9_evolucao_abrangente/p0_regularizacao/pesos.json')
    for modelo in ('Q_sel_ext','Qsep_sel_fis'):
        for atributo,v in ps[modelo]['coeficientes'].items():valor(modelo+'_'+atributo,v,'II')
    sar=ler_json('local_runs/sar-inundacao/avaliacao_unet_v1_teste.json')
    valor('SAR_U-Net',sar['micro_sem_agua_permanente']['iou'],'VII')
    valor('SAR_Otsu',sar['comparacao_com_baseline']['micro_sem_agua_permanente']['iou']['otsu'],'VII')
    ch=ler_json('local_runs/fundamentacao-artigo-20261028/f02_chuva_gatilho_recife.json')['series']['era5_2014_2024']
    for k in ('chuva_dia','chuva_48h','chuva_futuro_30d'):valor(k,ch[k]['razao_de_taxas_por_10mm'],'IV',2)
    adm=ler_json('local_runs/admissao-cientifica/retrato.json')
    for k,v in [('observacoes',adm['linhas']),('treino',adm['por_uso']['treino']),
                ('afirmacao',adm['por_uso']['afirmacao'])]:
        escrito=f'{v:,}'.replace(',','.')
        checks.append({'id':'admissao_'+k,'valor_artefato':v,'apresentacao':escrito,
                       'bloco':'curadoria','ok':escrito in tex})
    exemplos=ler_json('docs/artigo_v9_metodo_imagens_tabelas_20261105/proveniencia/resultados_base_v8.json')['resposta']
    ruas_ids=('RB_79790_833','RB_56871_833')
    def linha_ruas(nome, valores, sinal=True):
        numeros=[]
        for v in valores:
            numero=f'{abs(v):.3f}'.replace('.',',')
            prefixo='$-$' if v<0 else '+' if sinal else ''
            numeros.append(prefixo+numero)
        esperado=nome+' & '+' & '.join(numeros)
        checks.append({'id':'exemplo_ruas_'+nome,'valores_artefato':valores,
                       'apresentacao':esperado,'tabela':'IV','ok':esperado in romano['IV']})
    for atributo,nome in [('f_hand','HAND (log)'),('f_dist_canal','Distância ao canal'),
                          ('f_declive','Declive'),('f_construido','Fração construída'),
                          ('f_areia','Areia'),('f_sca','Contribuição (log)'),
                          ('f_dist_dren','Distância à drenagem')]:
        linha_ruas(nome,[exemplos['contribuicoes'][i][atributo] for i in ruas_ids])
    linha_ruas('Escore final',[exemplos['escores'][i] for i in ruas_ids],sinal=False)
    return {'tipo':'conferencia_de_transcricao_estruturada','ok':all(c['ok'] for c in checks),
            'conferidos':len(checks),'divergencias':[c for c in checks if not c['ok']],
            'checagens':checks,'nota':'Resultados já executados; esta conferência não repete a validação científica.'}

if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    r=conferir()
    print(json.dumps(r,ensure_ascii=False,indent=2))
    raise SystemExit(0 if r['ok'] else 1)
