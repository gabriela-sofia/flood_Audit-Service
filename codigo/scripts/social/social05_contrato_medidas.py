"""Contrato de três medidas territoriais, sem escore social sintético."""
from __future__ import annotations
import math
from collections import defaultdict
from social02_medidas_territoriais import numero, limites_exposicao

# Numeradores e denominadores do mesmo universo oficial, nunca total municipal.
INDICADORES = {
    'criancas_0_9': (['V01031', 'V01032'], 'V01006', 'sensibilidade_etaria',
                    'fração de moradores de 0 a 9 anos; não mede capacidade individual'),
    'idosos_60_mais': (['V01040', 'V01041'], 'V01006', 'sensibilidade_etaria',
                       'fração de moradores de 60 anos ou mais; não mede dependência funcional'),
    'sem_agua_encanada': (['V00201'], 'V00001', 'moradia_infraestrutura',
                         'fração de DPPO sem água encanada; não mede potabilidade'),
    'agua_superficial': (['V00117'], 'V00001', 'moradia_infraestrutura',
                        'fração de DPPO cujo abastecimento principal usa água superficial'),
    'sem_ligacao_rede_agua': (['V00464'], 'V00001', 'moradia_infraestrutura',
                            'fração de DPPO sem ligação à rede; alternativas não são automaticamente inadequadas'),
    'sem_banheiro_sanitario': (['V00238'], 'V00001', 'moradia_infraestrutura',
                              'fração de DPPO sem banheiro nem sanitário'),
    'esgoto_vala': (['V00313'], 'V00001', 'moradia_infraestrutura',
                    'fração de DPPO com destinação de esgoto em vala; não prova contaminação'),
}

def proporcao(valores, denominador):
    ns = [numero(v) for v in valores]; d = numero(denominador)
    if d is None or any(n is None for n in ns):
        return {'numerador': None, 'denominador': d, 'valor': None, 'estado': 'ausente_ou_suprimido'}
    n = sum(ns)
    if n > d: return {'numerador': n, 'denominador': d, 'valor': None, 'estado': 'inconsistente'}
    return {'numerador': n, 'denominador': d, 'valor': n/d if d else None,
            'estado': 'disponivel' if d else 'universo_vazio'}

def componentes(row):
    return {nome: {**proporcao([row.get(c) for c in cs], row.get(den)),
                   'eixo': eixo, 'interpretacao': texto, 'codigos_numerador': cs,
                   'codigo_denominador': den, 'ano': 2022}
            for nome, (cs, den, eixo, texto) in INDICADORES.items()}

def agregar_componentes(rows):
    """Razão de somas sobre pares completos; cobertura explícita por indicador."""
    saida = {}
    for nome in INDICADORES:
        vals = [r[nome] for r in rows]
        validos = [v for v in vals if v['estado'] in ('disponivel', 'universo_vazio')]
        n = sum(v['numerador'] for v in validos); d = sum(v['denominador'] for v in validos)
        saida[nome] = {'numerador_observado': n, 'denominador_observado': d,
                       'proporcao_universo_com_dado': n/d if d else None,
                       'setores_com_par_completo': len(validos), 'setores_sem_par_completo': len(vals)-len(validos),
                       'universo': 'apenas setores com numerador e denominador consistentes; sem imputação'}
    return saida

def exposicao(unidades, geometria, referencia):
    """Elegibilidade e semântica são distintas do simples cruzamento espacial."""
    for campo in ('id', 'tipo', 'data', 'fonte_sha256', 'uso', 'estado_admissao'):
        if not referencia.get(campo): raise ValueError('referência incompleta: '+campo)
    if referencia['uso'] in ('reserva_modelo', 'confirmacao_independente'):
        raise ValueError('reserva não pode ser aberta por este estudo de desenvolvimento')
    r = limites_exposicao(unidades, geometria, referencia['tipo'])
    elegivel = referencia['estado_admissao'] == 'admitida'
    return {**r, 'referencia': referencia, 'estado': 'descritivo_admitido' if elegivel else 'exploratorio_nao_admitido',
            'exposicao_confirmada': False, 'contagem_confirmada': None,
            'defasagem': 'população e moradias de 2022; não estima a população presente na data do evento'}

def cobertura_amostral(linhas):
    """Diagnóstico de oportunidade em pontos amostrados, por estrato do desenho.

    Não é cobertura areal. Pesos de expansão vêm exclusivamente do sorteio
    original. Anotabilidade visual por IA não é sensibilidade do sensor.
    """
    grupos = defaultdict(list); vistos = set()
    for r in linhas:
        chave = (r['evento'], r['cena'], r['codigo'])
        if chave in vistos: raise ValueError('ponto/cena duplicado')
        vistos.add(chave)
        if r['classe'] not in ('agua', 'seco', 'nao_sei'): raise ValueError('classe inesperada')
        if not math.isfinite(r['peso']) or r['peso'] <= 0: raise ValueError('peso amostral inválido')
        if r['referencia_admitida'] and not r['revisao_humana_validada']: raise ValueError('referência sem validação')
        # Regime/estrato instrumental mantém as comparações condicionadas ao desenho.
        grupos[(r['evento'], r['cena'], r['estrato_instrumento'], r['situacao'], r['grupo_renda'])].append(r)
    medidas = []
    for chave, rs in sorted(grupos.items()):
        w = sum(r['peso'] for r in rs)
        medidas.append(dict(zip(('evento','cena','estrato_instrumento','situacao','grupo_renda'), chave),
                            n=len(rs), setores=len({r['setor'] for r in rs if r['setor']}),
                            pontos_anotaveis=sum(r['classe'] != 'nao_sei' for r in rs),
                            fracao_anotavel_ponderada=sum(r['peso'] for r in rs if r['classe'] != 'nao_sei')/w,
                            fracao_referencia_admitida=sum(r['peso'] for r in rs if r['referencia_admitida'])/w))
    return {'estado': 'diagnostico_amostral' if linhas else 'not_testable', 'medidas': medidas,
            'pontos': len(linhas), 'pontos_sem_setor': sum(not r['setor'] for r in linhas),
            'suporte': 'pontos da amostra estratificada; não área nem pessoas',
            'desigualdade_confirmada': False, 'inferencias_causais': False,
            'avaliacao': 'um evento, referência por IA não validada; requer replicação independente e incerteza por blocos espaciais'}
