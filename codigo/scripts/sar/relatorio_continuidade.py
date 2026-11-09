"""Relatorio da campanha `continuidade_sar_v1`, lido so dos artefatos gravados.

Nao treina, nao infere, nao escolhe. Confere hashes de protocolo e pesos
antes de escrever. Valores com precisao completa ficam em resultado.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import continuidade_retencao as cr
from dados_pares_geograficos import sha

LOCAL = {"sen1:11:2019-05-22": "EUA", "sen1:2:2018-09-18": "Gana", "sen1:6:2017-06-28": "Paquistao",
         "sen1:9:2019-09-17": "Espanha", "sen1:3:2016-08-12": "India", "sen1:10:2017-05-30": "Sri Lanka",
         "sen1:7:2018-10-31": "Paraguai", "sen1:4:2018-08-05": "Mekong", "sen1:5:2018-09-21": "Nigeria",
         "sen1:8:2018-05-07": "Somalia"}
ORDEM = ("reinicio", "continuacao", "continuacao_replay", "reinicio_passos_acumulados", "replay_entropia",
         "replay_fixa")


def f(v, n=3):
    return "-" if v is None else f"{v:.{n}f}"


def nome(e):
    return LOCAL.get(e, e)


def conferir(pasta):
    pasta = Path(pasta)
    r = cr.dc.ler(pasta / "resultado.json")
    if sha(pasta / "protocolo.json") != r["protocolo_sha256"]:
        raise ValueError("protocolo divergente do resultado")
    for job, h in r["pesos"].items():
        if sha(pasta / job / "melhor.pt") != h["arquivo_sha256"]:
            raise ValueError(f"peso alterado depois da consolidacao: {job}")
    return r, cr.dc.ler(pasta / "protocolo.json"), cr.dc.ler(pasta / "matriz.json"), \
        cr.dc.ler(pasta / "treino_concluido.json")


def tabela_matriz(m, braco, ctx, teste, metrica):
    if braco not in m:
        return []
    cols = ctx + teste
    linhas = [f"| etapa | {' | '.join(nome(c) for c in cols)} |", "|---:|" + "---:|"*len(cols)]
    for etapa in sorted({int(e) for s in m[braco].values() for e in s}):
        vals = []
        for c in cols:
            xs = [m[braco][s][str(etapa)][c][metrica] for s in m[braco] if str(etapa) in m[braco][s]
                  and m[braco][s][str(etapa)].get(c, {}).get(metrica) is not None]
            marca = "**" if c in ctx and ctx.index(c)+1 == etapa else ""
            vals.append(f"{marca}{f(float(np.mean(xs)))}{marca}" if xs else "-")
        linhas.append(f"| {etapa} | {' | '.join(vals)} |")
    return linhas


def escrever(pasta):
    pasta = Path(pasta)
    r, p, mz, concl = conferir(pasta)
    m, ctx, teste = mz["matriz"], p["contextos_em_ordem"], p["avaliacao_desenvolvimento"]
    cfg = p["config"]
    L = ["# Aprendizado continuo SAR (`continuidade_sar_v1`)", "",
         f"Assinatura `{r['assinatura']}`; protocolo sha256 `{r['protocolo_sha256']}`.",
         "Desenvolvimento em eventos conhecidos. Nenhum peso promovido; svc02 e escore fisico intocados.", "",
         "## Desenho", "",
         f"Contextos em ordem: {' -> '.join(nome(c) for c in ctx)}. Validacao: "
         f"{', '.join(nome(e) for e in p['validacao'])}. Avaliacao de desenvolvimento: "
         f"{', '.join(nome(e) for e in teste)}.",
         f"{cfg['anotacoes_por_contexto']} anotacoes por contexto, iguais entre bracos; {cfg['passos_por_etapa']} "
         f"passos por etapa, lote {cfg['batch']}, recorte {cfg['recorte']}, U-Net base {cfg['base']}, "
         f"sementes {cfg['sementes']}. Limiar congelado {p['congelado_da_validacao']['limiar']} (validacao de 07/10).",
         f"Otimizador: {p['otimizador']}. Normalizacao: {p['normalizacao']}.", "",
         "| contexto | regra da reserva | ajuste | retencao | tampao |", "|---|---|---:|---:|---:|"]
    for c in ctx:
        d = p["divisoes"][c]
        L.append(f"| {nome(c)} | {d['regra']} | {len(d['ajuste'])} | {len(d['retencao'])} | {len(d['tampao'])} |")
    L += ["", "## Custos", "",
          "| braco | semente | anotacoes | buffer replay (chips) | passos no modelo final | passos executados na linhagem | segundos | epoca escolhida |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for b in ORDEM:
        for s, c in sorted(r["custos"].get(b, {}).get("sementes", {}).items()):
            L.append(f"| {b} | {s} | {c['anotacoes']} | {c['buffer_replay_chips']} | "
                     f"{c['passos_contidos_no_modelo_final']} | {c['passos_executados_na_linhagem']} | "
                     f"{c['segundos_na_linhagem']:.0f} | {c['melhor_epoca_final']} |")
    L += ["", "Reinicio e continuacao executam os mesmos passos por linhagem; o modelo final do reinicio contem so "
          "os da ultima etapa. O controle de passos acumulados iguala o que o modelo final contem.", ""]
    for metrica, titulo in (("iou", f"IoU no limiar {r['limiar']}"), ("ap", "AP aproximada")):
        L += [f"## Matriz de ganho e retencao: {titulo}", "",
              "Media das sementes. Linha = etapa (contextos ja vistos); negrito = contexto recem-aprendido. "
              "Colunas apos os contextos sao os eventos de avaliacao, nunca treinados.", ""]
        for b in ORDEM:
            t = tabela_matriz(m, b, ctx, teste, metrica)
            if t:
                L += [f"### {b}", "", *t, ""]
    L += ["## Esquecimento", "", "F = melhor desempenho anterior menos o final (Chaudhry et al. 2018); "
          "BWT = final menos o desempenho logo apos aprender (Lopez-Paz e Ranzato 2017). Media das sementes.", "",
          "| braco | metrica | esquecimento medio | BWT medio | por contexto (F) | por semente (F medio) |",
          "|---|---|---:|---:|---|---|"]
    for b in ORDEM:
        for metrica in ("iou", "ap"):
            x = r["retencao"].get(b, {}).get(metrica)
            if x:
                pc = ", ".join(f"{nome(c)} {f(v)}" for c, v in x["esquecimento_por_contexto"].items())
                ps = ", ".join(f"{s}: {f(v)}" for s, v in x["esquecimento_medio_por_semente"].items())
                L.append(f"| {b} | {metrica} | {f(x['esquecimento_medio'])} | {f(x['bwt_medio'])} | {pc} | {ps} |")
    L += ["", "## Contrastes pre-registrados", "", "Unidade = evento; sementes agregadas dentro do evento.", "",
          "| hipotese | metrica | eventos | media | a favor | IC95 | p | estado |", "|---|---|---:|---:|---|---|---:|---|"]
    for metrica in ("iou", "ap"):
        for h, c in r["contrastes"][metrica].items():
            ic = c.get("ic95_bootstrap_evento")
            L.append(f"| {h} | {metrica} | {c['n_eventos']} | {f(c.get('media'), 4)} | {c.get('a_favor', '-')} | "
                     f"{'-' if ic is None else f'[{ic[0]:.4f}, {ic[1]:.4f}]'} | {f(c.get('p_wilcoxon'), 4)} | "
                     f"{c['estado']} |")
    pt = r["portao_e2"]
    L += ["", f"Primaria: {p['hipoteses']['primaria']}.", "",
          "## Portao da aquisicao continuada (somente validacao)", "",
          f"Liberado: **{pt['liberado']}**; replay competente em todas as sementes: "
          f"{pt.get('replay_competente_todas_sementes')}; mediana IoU validacao replay "
          f"{f(pt.get('mediana_iou_validacao_replay'), 4)} vs reinicio {f(pt.get('mediana_iou_validacao_reinicio'), 4)}.",
          "", "## Validacao por etapa (decisao)", "",
          "| braco | semente | etapa | IoU macro validacao | competente |", "|---|---:|---:|---:|---|"]
    for j in concl["jobs"]:
        v = j["validacao"]
        L.append(f"| {j['braco']} | {j['seed']} | {j['etapa']} | {f(v['macro_iou'])} | {v['competente']} |")
    col = r["colapsos_tudo_fundo"]
    L += ["", "## Classe minoritaria", "",
          f"Casos em que o modelo nao marcou agua alguma num evento com agua observada (limiar congelado): "
          f"**{len(col)}**.", ""]
    for c in col[:30]:
        L.append(f"- {c['braco']} s{c['seed']} etapa {c['etapa']}: {nome(c['evento'])}")
    L += ["", "## Pesos e cartoes", "", f"{len(r['pesos'])} checkpoints com hash completo do estado e do arquivo. "
          "Cartoes dos modelos finais em `cartoes/`.", "",
          "## Limites", "",
          "- Dez eventos conhecidos; retencao tem no maximo quatro contextos antigos (abaixo do minimo de seis).",
          "- A reserva de retencao e do mesmo evento do ajuste, separada no espaco; nao e evento novo.",
          "- Uma ordem de contextos; efeito de ordem nao medido.",
          "- Entrada pos-evento; nao valida pre-pos, causalidade, Brasil ou previsao futura.",
          "- Nao e politica aprendida: a selecao continua sendo regra declarada."]
    (pasta / "relatorio.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return pasta / "relatorio.md"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pasta", type=Path, nargs="?", default=cr.SAIDA)
    print(escrever(ap.parse_args().pasta))
