"""Confere medidas do artigo v22 nos registros de execução que as sustentam."""
from __future__ import annotations

import csv
from decimal import Decimal, ROUND_HALF_UP
import json
import re

from acervo import RAIZ, caminho, ler_json


def _linhas_csv(origem):
    with caminho('local_runs/'+origem).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def _selecionar(linhas, **filtros):
    registros = [linha for linha in linhas if all(linha[k] == str(v) for k, v in filtros.items())]
    if len(registros) != 1:
        raise ValueError(f'Filtro deve identificar um registro: {filtros}')
    return registros[0]


def _formatar(valor, casas):
    if isinstance(valor, str):
        return valor
    if isinstance(valor, int):
        return f'{valor:,}'.replace(',', '.')
    d = Decimal(str(valor)).quantize(Decimal(1).scaleb(-casas), rounding=ROUND_HALF_UP)
    return f'{d:.{casas}f}'.replace('.', ',')


def conferir():
    fonte = (RAIZ/'artigo/fonte/artigo.tex').read_text(encoding='utf-8')
    matriz = json.loads((RAIZ/'manifestos/afirmacoes_artigo.json').read_text(encoding='utf-8'))
    linhas = {a['id']: a for a in matriz['afirmacoes'] if a['tabela']}
    checks = []
    ressalvas = []

    def medir(item, valores, casas=3, trecho=None):
        if trecho is None:
            registro = linhas[item]
            assert registro['texto_tex'] in fonte, item
            trecho = registro['afirmacao']
        for valor in valores:
            escrito = _formatar(valor, casas)
            ok = bool(re.search(r'(?<![\d,.-])'+re.escape(escrito)+r'(?!\d|[,.]\d)', trecho))
            checks.append({'id': item, 'valor_artefato': valor, 'apresentacao': escrito, 'ok': ok})

    def run(p):
        return ler_json('local_runs/'+p)

    f08 = run('fundamentacao-artigo-20261106/f08_ruas_e_comunidades.json')
    hand = run('mecanismos-transferiveis/v9/identificacao_causal_v1/resultado.json')
    chip = f08['chip']['hand_m']
    medir('I.01', [chip['INUNDACAO'],chip['NAO_INUNDADO_OBSERVADO']], 1)
    dentro = hand['C2_dentro_do_evento']['f_hand']
    medir('I.02', [hand['dados']['linhas_completas'], dentro['media'], *dentro['ic']], 2)
    pares = hand['C4_pares_vizinhos']['3km']['f_hand']
    medir('I.03', [pares['media'], *pares['ic']], 2)
    contrastes = run('mecanismos-transferiveis/v1/resultado.json')['contrastes']
    geral = contrastes['G-A0_regra_hand']
    medir('I.05', [geral['delta_medio'], *geral['ic95'], geral['unidades_a_favor']])
    ablacao = contrastes['G_sem_acumulacao-G']
    medir('I.06', [ablacao['delta_medio'], *ablacao['ic95'], ablacao['unidades']-ablacao['unidades_a_favor']])
    mt11 = run('mt11-fechamento/v2/reparo_grafo_3b.json')['contrastes']['emendada_primaria']
    for item, k in [('I.07','HQ1'),('I.08','HQ2')]:
        medir(item,[mt11[k]['media'], *mt11[k]['ic']])
    gfd = run('gfd-quarentena/executado.json')['comparacoes']['M1_logistica_terreno|M0_regra_hand']
    medir('I.09', [gfd['dentro_do_evento_mediana'], f"{gfd['eventos_a_favor']}/{gfd['eventos_pareados']}", gfd['p_wilcoxon']])
    pesos = run('transferencia-vulnerabilizados/v9_evolucao_abrangente/p0_regularizacao/pesos.json')
    for n, campos in enumerate((('f_hand','f_areia'),('f_dist_canal','f_sca'),('f_declive','f_dist_dren'),('f_construido',)),10):
        medir(f'I.{n:02d}',[pesos[m]['coeficientes'][k] for k in campos for m in ('Q_sel_ext','Qsep_sel_fis')])

    fcu = run('fundamentacao-artigo-20261028/f03_ic_transferencia_coortes.json')['por_coorte']
    for n, destino in enumerate(('rmpa','manaus','recife','rmr'),1):
        r = fcu[destino]
        item = f'II.{n:02d}'
        medir(item,[r['positivos'],r['comunidades_com_positivo'],r['auc']['Q_sel_ext'],*r['ic95_auc_Q']])
        if destino != 'rmpa':
            medir(item,[r['Q_menos_A0_hand']['delta'],*r['Q_menos_A0_hand']['ic95'],r['Q_menos_menos_declive']['delta']])
    area = run('fundamentacao-artigo-20261106/f07_estratos_coortes.json')['por_coorte']
    for n,destino in enumerate(('rmpa','manaus','curitiba','recife','rmr'),5):
        r = area[destino]
        todos = r['todas']
        item = f'II.{n:02d}'
        medir(item,[todos['positivos'],todos['auc']['Q_sel_ext']])
        if destino != 'curitiba':
            medir(item,[todos['Q_menos_A0_hand']['delta']])
            for v in todos['Q_menos_A0_hand']['ic95']:
                medir(item,[v],4 if destino=='manaus' and v<0.001 else 3)
        if destino != 'rmpa':
            medir(item,[todos['Q_menos_menos_declive']['delta']])
        for estrato in ('baixa_renda','comparador'):
            if r[estrato]['estado'] == 'avaliado':
                medir(item,[r[estrato]['auc']['Q_sel_ext']])
    tr = _linhas_csv('transferencia-vulnerabilizados/v6_evidencia_robusta/triangulacao_ufrgs/fracoes_a01.csv')
    fracao = [float(r['mean']) for r in tr if r['estado_s2']=='positivo']
    checks.append({'id':'II.11','fracoes_por_cobertura':fracao,'apresentacao':'99%',
                   'ok': '99%' in linhas['II.11']['afirmacao'] and len(fracao)==3 and min(fracao)>=0.985})

    ruas = run('fundamentacao-artigo-20261028/f06_ic_auc_ruas_recife.json')
    medir('III.02',[f08['fontes']['relato_sem_s2'],f08['fontes']['uniao_com_registro']])
    for item, k in [('III.03','funcao_transferida'),('III.04','hand')]:
        medir(item,[ruas[k]['auc_dentro_do_bairro'], *ruas[k]['ic95']])
    medir('III.05',[ruas['funcao_transferida']['auc_dentro_do_bairro']-ruas['hand']['auc_dentro_do_bairro']])
    pareado = f08['ruas']['pareado_fator_1.5']
    medir('III.07',[pareado['delta'],*pareado['ic95_delta']])
    e = ler_json('docs/artigo_v9_metodo_imagens_tabelas_20261105/proveniencia/resultados_base_v8.json')['resposta']
    ids = ('RB_79790_833','RB_56871_833')
    for n,campos in enumerate((('f_hand','f_areia'),('f_dist_canal','f_sca'),('f_declive','f_dist_dren'),('f_construido',)),8):
        medir(f'III.{n:02d}',[e['contribuicoes'][i][k] for k in campos for i in ids])
    medir('III.11',[e['escores'][i] for i in ids])
    medir('III.13',[int(e['posicoes']['Rua Muriti'][0]),int(e['posicoes']['Av. Recife'][0]),74])

    f04 = run('fundamentacao-artigo-20261028/f04_encosta_e_retencao.json')
    p = f04['petropolis']
    for item,k in [('IV.01','agua'),('IV.02','idai')]:
        medir(item,[p['auc_por_cicatriz'][k],*p['ic95'][k]])
    medir('IV.03',[p['idai_menos_agua']['delta'],*p['idai_menos_agua']['ic95']])
    c3 = _selecionar(_linhas_csv('petropolis-aprendizado/v1/resultados/contrastes_T1.csv'),
                     B=100,contraste='C3_valor_conhecimento_externo')
    medir('IV.04',[float(c3[k]) for k in ('delta','ic95_lo','ic95_hi')])

    f01 = run('fundamentacao-artigo-20261028/f01_ic_ganho_atualizacao.json')['por_coorte']['rmpa']
    for item,k in [('V.01','margem_v9_gate'),('V.02','uniforme_gate')]:
        r = f01[k]
        medir(item,[r['ganho'],*r['ic95'], f"{r['sementes_favoraveis']}/{r['sementes_ativas']}"])
    medir('V.01',[f01['auc_congelado'],f01['margem_v9_gate']['auc_media_sementes']])
    origem = 'transferencia-vulnerabilizados/'
    f4 = _selecionar(_linhas_csv(origem+'v8_validacao_integrada/analise/contrastes_poucos_rotulos.csv'),
                    coorte='rmpa',populacao='todas',politica='urbano_uniforme',B=400,tau='t25',contraste='F4')
    medir('V.03',[float(f4['diferenca']),*json.loads(f4['ic95'])])
    r = f04['retencao_rmpa']
    medir('V.04',[r['T_B_media_das_unidades'],*r['ic95_bootstrap_de_unidades']])
    der = _selecionar(_linhas_csv(origem+'v10_ciclo_contextual/iii_memoria/pareados_final.csv'),
                     avaliado='rmpa',populacao='fcu',a='der_limitado',b='congelado')
    medir('V.05',[float(der['media']),*json.loads(der['ic95'])])

    c = f08['comunidades']
    censo = ler_json('docs/artigo_final_6_paginas_v5_20261029/images/peso_das_regioes.json')['_total_coortes']
    medir('VI.01',[censo['fcu']+42,censo['pessoas']+43019])
    medir('VI.02',[c['com_positivo'],c['fcu_total']])
    medir('VI.03',[c['duas_classes'],c['cinco_cinco']])

    u = run('sar-inundacao/avaliacao_unet_v1_teste.json')
    medir('VII.01',[u['micro_sem_agua_permanente']['iou']])
    pre = run('pretreino-sar/v2/analise.json')
    p1 = pre['primaria']['mae@1 - aleatorio@1']
    p3 = pre['secundaria']['mae@3 - aleatorio@3']
    medir('VII.02',[p1['n'],p1['media'],*p1['ic95_t'],p1['a_favor'],p3['media']])
    lwf = run('planejamento-tese/v2/continuidade/diagnostico.json')['reagregacao_mesmo_suporte']['iou_0_2']['v3_1_lwf_legado_menos_continuacao']
    medir('VII.03',[lwf['n'],*lwf['ic95_t']])
    exata = Decimal(str(lwf['media']))
    exibida = Decimal('0.049')
    intermediaria = exata.quantize(Decimal('0.0001'),rounding=ROUND_HALF_UP)
    compativel = intermediaria.quantize(Decimal('0.001'),rounding=ROUND_HALF_UP)==exibida
    checks.append({'id':'VII.03','valor_artefato':lwf['media'],'apresentacao':'0,049',
                   'ok':compativel and '0,049' in linhas['VII.03']['afirmacao']})
    ressalvas.append({'id':'VII.03','estimativa_integral':lwf['media'],'apresentacao_artigo':float(exibida),
        'arredondamento_direto':float(exata.quantize(Decimal('0.001'),rounding=ROUND_HALF_UP)),
        'compativel_com_arredondamento_intermediario':compativel})
    b = run('evolucao-escopo/v1/cognicao/resultado.json')
    medir('VII.04',[b['consultas']])
    medir('VII.04',[100*b['medidas'][k]['agregado']['recall_em_k'] for k in ('hibrido_rrf','lexical')],2)
    j = run('verificacao-suporte/medida_de_suporte_com_juiz.json')
    medir('VII.05',[j['casos'],*[j[k]['geral']['acuracia'] for k in ('conjuncao','lexico')],
                    *[j[k]['geral']['falsos_positivos'] for k in ('conjuncao','lexico')]])
    cv = _linhas_csv(origem+'v8_validacao_integrada/competencia_tranquilizadora/cobertura_tranquilizadoras.csv')
    for destino in ('rmpa','rmr'):
        medir('VII.06',[int(_selecionar(cv,destino=destino,metodo=m,cobertura_alvo=0.7,classe='fcu')['tranquilizadoras_erradas'])
                        for m in ('margem','aleatorio')])
    p = run('ia-integrada-v2/fechamento_v1/fisica.json')['paridade']
    medir('VII.07',[p['mt11']['pontos']+p['rodada3']['pontos']])
    erro = max(p[k]['erro_max_q'] for k in ('mt11','rodada3'))
    checks.append({'id':'VII.07','valor_artefato':erro,'limite_artigo':3.56e-15,
                   'ok':erro<=3.56e-15 and '3,56' in linhas['VII.07']['afirmacao']})
    qa = run('evolucao-escopo/v1/qa_final/v3/resultado.json')
    restauracao = run('evolucao-escopo/v1/restauracao/v2/anexo_codigo/resultado.json')
    medir('VII.08',[qa['passaram'],restauracao['fontes_conferidas']])

    adm = run('admissao-cientifica/retrato.json')
    medir('Método, admissão',[adm['linhas'],adm['por_uso']['treino'],adm['por_uso']['avaliacao_externa']],trecho=fonte)
    q = run(origem+'v8_validacao_integrada/q_externo/resultado.json')['subconjuntos']
    medir('Método, origens',[q['reproducao']['linhas'],q['fora_BR']['eventos'],q['fora_BR']['linhas'],
          q['fora_BR']['unidades'],*q['fora_BR']['por_fonte'].values()],trecho=fonte)
    checks.append({'id':'Método, separação dos ajustes','ok':q['reproducao']['contem_EMSR720'] and
                   not q['fora_BR']['contem_EMSR720'] and q['fora_BR']['pontos_no_brasil']==0})
    erros = [c for c in checks if not c['ok']]
    return {'tipo':'conferencia_de_transcricao_estruturada','versao':'v22','ok':not erros,
            'conferidos':len(checks),'divergencias':erros,'ressalvas':ressalvas,'checagens':checks,
            'nota':'Conferência de registros existentes. Resultados datados mantêm o protocolo e a população de origem.'}


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    resultado = conferir()
    print(json.dumps(resultado,ensure_ascii=False,indent=2))
    raise SystemExit(0 if resultado['ok'] else 1)
