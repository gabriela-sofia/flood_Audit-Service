"""Executa o protocolo MT03 autorizado, em pasta isolada e sem reajustar candidatas.

O protocolo original permanece imutavel. A abertura e registrada ANTES do primeiro
pacote; uma interrupcao nao devolve a reserva ao estado de intocada.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ('metodo', 'servico', 'geo', 'terreno', 'sar', 'suscetibilidade'):
    sys.path.insert(0, str(REPO / 'scripts' / pasta))
from proveniencia import agora_utc, sha256_arquivo, sha256_json
import mt03_triagem_cems as tri
import mt03_cems_desenvolvimento as cd
import mt03_atributos_novos as an
import mt01_experimento as e1
import mt01_relatorio as rel

V3 = tri.SAIDA.parent
PROTOCOLO = V3 / 'confirmacao/protocolo_confirmacao_v3.json'
SAIDA = REPO / 'local_runs/mecanismos-transferiveis/v4/confirmacao'
AUTORIZACAO = SAIDA / 'autorizacao.json'
MODELOS = ['A0_regra_hand', 'G_v2', 'G_terreno', 'G_conexao', 'G_rios10']


def ler(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def gravar(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(rel.limpo(d), ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.replace(p)


def conferir(protocolo=PROTOCOLO, autorizacao=AUTORIZACAO):
    p, a = ler(protocolo), ler(autorizacao)
    h = sha256_json({k: v for k, v in p.items() if k not in ('quando_utc', 'hash_resumo')})
    if h != p['hash_resumo'] or h != a['protocolo']:
        raise ValueError('protocolo divergente da autorizacao')
    if not a.get('abertura_autorizada'):
        raise ValueError('abertura nao autorizada')
    if sha256_arquivo(REPO / p['candidatas']['arquivo']) != p['candidatas']['sha256']:
        raise ValueError('pesos congelados divergentes')
    part = ler(tri.SAIDA / 'particao.json')
    if part['hash_resumo'] != p['reserva']['particao']:
        raise ValueError('particao divergente')
    if sha256_json({k: v for k, v in part.items() if k not in ('quando_utc', 'hash_resumo')}) != part['hash_resumo']:
        raise ValueError('integridade da particao falhou')
    unidades = [u for u in part['unidades'] if u['papel'] == 'reserva']
    if unidades != p['reserva']['unidades']:
        raise ValueError('lista da reserva divergente')
    permitidos = {c: u['unidade_nova'] for u in unidades for c in u['ativacoes']}
    proibidos = tri.NUNCA_LER | tri.RESERVA_MT02 | tri.FRENTE_SAR
    if set(permitidos) & proibidos:
        raise ValueError('reserva cruza outra frente protegida')
    return p, permitidos, proibidos


def configurar():
    p, permitidos, proibidos = conferir()
    # Reutiliza EXATAMENTE a cadeia declarada no protocolo, em processo separado.
    cd.CD = SAIDA / 'cems'
    cd.PACOTES, cd.AREAL, cd.MDT, cd.TERRENO = [cd.CD / n for n in ('pacotes', 'areal', 'mdt', 'terreno')]
    cd.particao = lambda: (permitidos, proibidos)
    cd.CD.mkdir(parents=True, exist_ok=True)
    return p, permitidos


def abrir():
    p, permitidos, _ = conferir()
    arq = SAIDA / 'abertura.json'
    if arq.exists():
        d = ler(arq)
        if d['protocolo'] != p['hash_resumo']:
            raise ValueError('abertura de outro protocolo')
        return d
    # Confronto antes de ler rotulos: pacotes ja presentes noutras cadeias e registro de exames.
    conflitos = []
    for raiz in ('cems-05-pacotes', 'cems-06-por-aoi-e-data'):
        for f in (REPO / 'local_runs' / raiz).rglob('*.zip'):
            if any(c in f.name for c in permitidos):
                conflitos.append(str(f.relative_to(REPO)))
    reg = REPO / 'local_runs/reservas-examinadas/examinados_apos_congelamento.jsonl'
    if reg.exists():
        for linha in reg.read_text(encoding='utf-8').splitlines():
            d = json.loads(linha)
            if d.get('avaliacao') != 'mt04_confirmacao':
                conflitos.extend(c for c in permitidos if c in re.findall(r'EMSR\d+', json.dumps(d)))
    if conflitos:
        raise ValueError(f'possivel exame previo: {conflitos}')
    d = {'tipo': 'abertura_reserva_mt03', 'quando_utc': agora_utc(), 'protocolo': p['hash_resumo'],
         'autorizacao': sha256_arquivo(AUTORIZACAO), 'ativacoes': sorted(permitidos),
         'estado': 'abertura iniciada; nunca voltar a declarar reserva intacta',
         'candidatas': p['candidatas'], 'independencia': 'grafo MT03 congelado; confronto adicional de pacotes/exames sem conflito'}
    gravar(arq, d)
    from reservas_examinadas import registrar_exame
    registrar_exame(['CEMS_NOVO:' + c for c in permitidos], 'cems', 'mt04_confirmacao')
    return d


def preparar():
    configurar()
    d = pd.read_parquet(cd.CD / 'pontos.parquet')
    te = e1.amostrar(d, e1.CONFIG['teto_teste_por_evento_classe'], e1.CONFIG['semente'])
    partes = []
    for code, g in te.groupby('code'):
        ano = int(ler(tri.META / f'{code}.json')['ativacao']['eventTime'][:4])
        arq = SAIDA / 'atributos' / f'{code}.parquet'
        if arq.exists():
            partes.append(pd.read_parquet(arq))
            continue
        g = g.copy()
        g['ponto_id'] = g.ponto_id.str.replace('MT03DEV_', 'MT04CONF_', regex=False)
        g['unidade_terreno'] = 'mt04conf/' + g.nome
        g = g.drop(columns=['elevation_local_m', 'ln_sca'], errors='ignore')
        x = an.completar(g, cd.TERRENO, ano, code, SAIDA / 'lotes_cobertura', grupo='unidade_espacial')
        arq.parent.mkdir(parents=True, exist_ok=True)
        x.to_parquet(arq, index=False)
        partes.append(x)
        print('atributos', code, len(x), flush=True)
    d = pd.concat(partes, ignore_index=True)
    d.to_parquet(SAIDA / 'pontos_avaliacao.parquet', index=False)
    return {'pontos': len(d), 'unidades': int(d.unidade.nunique())}


def medidas(d):
    from sklearn.metrics import roc_auc_score
    linhas = []
    for (un, ev), g in d.groupby(['unidade', 'evento_id']):
        recortes = {'todos': g, 'fluvial': g[(g.classe == 0) | (g.processo_ponto == 'fluvial')]}
        for rec, s in recortes.items():
            if s.classe.nunique() != 2 or len(s) < 20 or s.classe.sum() < 10:
                continue
            for m in MODELOS:
                if not np.isfinite(s[m]).all():
                    raise ValueError('escore nao finito')
                auc = roc_auc_score(s.classe, s[m])
                if abs(auc - e1.auc(s.classe.to_numpy(), s[m].to_numpy())) > 1e-9:
                    raise ValueError('divergencia de AUC')
                linhas.append({'unidade': un, 'evento': ev, 'recorte': rec, 'modelo': m, 'auc': auc})
    return pd.DataFrame(linhas)


def analisar(d, p):
    m = medidas(d)
    if m.empty:
        return {'estado': 'not_testable', 'motivo': 'nenhum evento com duas classes'}
    proprias = {u['unidade_nova'] for u in p['reserva']['unidades'] if u['estrato'] == 'processo_proprio'}
    def agrupar(t):
        e = t.groupby(['unidade', 'evento', 'modelo']).auc.mean().reset_index()
        return e.groupby(['unidade', 'modelo']).auc.mean().unstack()
    out = {'hipoteses': {}, 'processo_proprio': [], 'por_evento': m.to_dict('records')}
    for chave, rec, a, b in [('HC1', 'todos', 'G_v2', 'A0_regra_hand'),
                            ('HC2', 'fluvial', 'G_conexao', 'G_v2'), ('HC3', 'fluvial', 'G_rios10', 'G_v2')]:
        u = agrupar(m[(m.recorte == rec) & ~m.unidade.isin(proprias)])
        out['hipoteses'][chave] = {'hipotese': p['hipoteses_e_dimensionamento'][chave]['hipotese'],
                                  'regra': 'delta >= 0,01 e IC95 acima de zero; minimo 8 unidades',
                                  **rel.contraste(u, a, b, 'superioridade', .01, 2000),
                                  'por_unidade': u.reset_index().to_dict('records')}
    out['processo_proprio'] = agrupar(m[(m.recorte == 'todos') & m.unidade.isin(proprias)]).reset_index().to_dict('records')
    return out


def avaliar():
    p, _, _ = conferir()
    if not (SAIDA / 'abertura.json').exists():
        raise ValueError('abertura nao registrada')
    d = pd.read_parquet(SAIDA / 'pontos_avaliacao.parquet')
    d = pd.concat([d.reset_index(drop=True), an.pontuar(d).reset_index(drop=True)], axis=1)
    d.to_parquet(SAIDA / 'predicoes.parquet', index=False)
    r = {'tipo': 'mt04_confirmacao_protocolo_mt03', 'quando_utc': agora_utc(),
         'protocolo': p['hash_resumo'], 'candidatas': p['candidatas']['sha256'],
         'predicoes': sha256_arquivo(SAIDA / 'predicoes.parquet'), **analisar(d, p),
         'nao_confirma': p['o_que_nao_confirma'], 'promovido': False,
         'familia': 'HC3 secundaria; IC por hipotese, sem controle conjunto de multiplicidade no protocolo original'}
    gravar(SAIDA / 'resultado.json', r)
    return r


def executar():
    configurar()
    abrir()
    for nome, f in [('baixar', cd.baixar), ('areal', cd.areal), ('terreno', cd.terreno),
                    ('pontos', cd.pontos), ('preparar', preparar), ('avaliar', avaliar)]:
        print('ETAPA', nome, flush=True)
        r = f()
        gravar(SAIDA / f'etapa_{nome}.json', r)
        if nome == 'terreno' and r['falhas']:
            raise RuntimeError('terreno incompleto; retomar sem alterar protocolo')
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('acao', choices=['conferir', 'adquirir', 'executar', 'avaliar'])
    a = ap.parse_args()
    if a.acao == 'adquirir':
        configurar()
        abrir()
        r = cd.baixar()
        gravar(SAIDA / 'etapa_baixar.json', r)
    else:
        r = conferir()[0] if a.acao == 'conferir' else executar() if a.acao == 'executar' else avaliar()
    print(json.dumps(rel.limpo(r), ensure_ascii=False, indent=1)[:10000])


if __name__ == '__main__':
    main()
