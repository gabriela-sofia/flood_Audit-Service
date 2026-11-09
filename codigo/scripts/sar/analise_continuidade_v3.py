"""Analise pre-registrada de `continuidade_sar_v3`, lida so de artefatos gravados.

Por checkpoint: temperatura e limiar ajustados SO no grupo de calibracao
(Escocia); AP, IoU legado 0,2, IoU calibrado, Brier/NLL/ECE nas unidades de
decisao; falso positivo no controle (Alemanha). Unidade = evento; repeticoes
(ordem, semente) agregadas dentro do evento. Familia primaria com Holm sobre
Wilcoxon exato e TOST com nivel 1-2*alfa/k. Retencao e transferencia sempre
separadas, como estimacao. Nao treina, nao escolhe nada pelo resultado.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon

import calibracao_decisao as cd
import campanha_independencia_autonomia as ci
import continuidade_v3 as v3
import diagnostico_competencia as dc
from dados_pares_geograficos import sha


def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i])
    aj, mx = [0.]*len(ps), 0.
    for r, i in enumerate(o):
        mx = max(mx, min(1., (len(ps)-r)*ps[i]))
        aj[i] = mx
    return aj


def p_exato(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2 or not np.any(xs):
        return 1.
    return float(wilcoxon(xs, method="exact").pvalue) if np.count_nonzero(xs) == len(xs) else float(wilcoxon(xs).pvalue)


def eventos_necessarios(media, dp, margem):
    folga = abs(media) - margem
    if folga <= 0 or not np.isfinite(dp) or dp == 0:
        return None
    for n in range(2, 5001):
        if stats.t.ppf(.975, n-1)*dp/np.sqrt(n) < folga:
            return n
    return None


def estimar(difs, margem, k=1, alfa=.05, p_holm=None, decide=True, minimo=6):
    if not difs:
        return {"estado": "not_testable", "n": 0}
    xs = np.array(list(difs.values()), float)
    n, m = len(xs), float(xs.mean())
    r = {"n": n, "media": m, "mediana": float(np.median(xs)), "a_favor": f"{int((xs > 0).sum())}/{n}",
         "diferencas": difs, "margem": margem}
    if n < 2:
        return {**r, "estado": "INDECIDIVEL"}
    dp = float(xs.std(ddof=1))
    se = dp/np.sqrt(n)
    ic95 = [m - stats.t.ppf(.975, n-1)*se, m + stats.t.ppf(.975, n-1)*se]
    nivel = 1 - 2*alfa/k
    q = stats.t.ppf(1-(1-nivel)/2, n-1)
    ic_tost = [m - q*se, m + q*se]
    p = p_exato(xs)
    r.update({"dp": dp, "ic95_t": ic95, "ic_tost": ic_tost, "nivel_tost": nivel, "p_wilcoxon": p,
              "p_holm": p if p_holm is None else p_holm, "eventos_para_decidir": eventos_necessarios(m, dp, margem)})
    if not decide:
        r["estado"] = "estimacao"
    elif n < minimo:
        r["estado"] = "INDECIDIVEL"
    elif r["p_holm"] < alfa and ic95[0] > margem:
        r["estado"] = "ganho_sustentado_no_desenvolvimento"
    elif r["p_holm"] < alfa and ic95[1] < -margem:
        r["estado"] = "perda_sustentada_no_desenvolvimento"
    elif ic_tost[0] > -margem and ic_tost[1] < margem:
        r["estado"] = "equivalente_dentro_da_margem"
    else:
        r["estado"] = "INDECIDIVEL"
    return r


def medir_job(av, legado):
    cal = list(av["calibracao"].values())
    T = cd.ajustar_temperatura(cal)
    t = cd.escolher_limiar(cal, T)
    out = {}
    for g in ("retencao", "avaliacao", "validacao_checkpoint"):
        for ev, e in av[g].items():
            m = cd.medir_evento(e, t, T, legado)
            out[ev] = {"grupo": g, "ap": m["ap"], "iou_legado": m["legado"]["iou"], "iou_calibrado": m["calibrado"]["iou"],
                       "precisao_cal": m["calibrado"]["precisao"], "recall_cal": m["calibrado"]["recall"],
                       "brier": m["prob_sem_calibrar"]["brier"], "brier_cal": m["prob_calibrada"]["brier"],
                       "nll": m["prob_sem_calibrar"]["nll"], "nll_cal": m["prob_calibrada"]["nll"],
                       "ece": m["prob_sem_calibrar"]["ece"], "ece_cal": m["prob_calibrada"]["ece"],
                       "tp_fp_legado": m["legado"]["tp"]+m["legado"]["fp"], "classe": cd.classificar_falha(m)}
    ctrl = {}
    for ev, e in av["controle"].items():
        m = cd.medir_evento(e, t, T, legado)
        ctrl[ev] = {"fp_legado": m["legado"]["fracao_predita_agua"], "fp_calibrado": m["calibrado"]["fracao_predita_agua"]}
    return {"T": T, "limiar": t, "eventos": out, "controle": ctrl}


def analisar(saida):
    saida = Path(saida)
    p = dc.ler(saida / "protocolo.json")
    if dc.ler(saida / "avaliacao_concluida.json")["assinatura"] != ci.hash_json(p):
        raise ValueError("avaliacao de outro protocolo")
    por_id, _, linhagens = v3.planejar(p)
    cfg = p["config"]
    medidas = {jid: medir_job(dc.ler(saida / "jobs" / jid / "avaliacao.json"), cfg["limiar_legado"]) for jid in por_id}
    treinos = {jid: dc.ler(saida / "jobs" / jid / "treino.json") for jid in por_id}
    canon = p["contextos_canonicos"]
    retencao_ev = sorted({ev for jid in por_id for ev, x in medidas[jid]["eventos"].items() if x["grupo"] == "retencao"})
    teste_ev = sorted({ev for jid in por_id for ev, x in medidas[jid]["eventos"].items() if x["grupo"] == "avaliacao"})
    unidades = retencao_ev + teste_ev
    lin_reg = []
    for lin in linhagens:
        final = medidas[lin["jobs"][-1]]
        serie = {k: medidas[j]["eventos"] for k, j in enumerate(lin["jobs"], 1)}
        ctx = lin_ctx = p["ordens"][lin["ordem"]]
        reg = {**{c: lin[c] for c in ("ordem", "seed", "braco")}, "jobs": lin["jobs"], "ctx": ctx,
               "final": final["eventos"], "controle": final["controle"], "T": final["T"], "limiar": final["limiar"]}
        if lin["braco"] != "reinicio_acumulado":
            for met in ("ap", "iou_calibrado", "iou_legado"):
                f, bwt, aq = {}, {}, {}
                for i, c in enumerate(lin_ctx, 1):
                    aq[c] = serie[i][c][met] if c in serie[i] else None
                for i, c in enumerate(lin_ctx[:-1], 1):
                    vals = [serie[e][c][met] for e in range(i, len(lin_ctx)) if serie[e][c][met] is not None]
                    fim, ini = serie[len(lin_ctx)][c][met], serie[i][c][met]
                    if vals and fim is not None:
                        f[c] = max(vals) - fim
                    if fim is not None and ini is not None:
                        bwt[c] = fim - ini
                reg[f"esquecimento_{met}"], reg[f"bwt_{met}"], reg[f"aquisicao_{met}"] = f, bwt, aq
            reg["matriz_ap"] = {k: {ev: x["ap"] for ev, x in s.items()} for k, s in serie.items()}
        cs = [treinos[j]["custo"] for j in lin["jobs"]]
        reg["custo"] = {c: sum(x.get(c, 0) for x in cs) for c in ("passos", "recortes_novos", "recortes_memoria",
                                                                "pixels_supervisionados", "pixels_distilados",
                                                                "passes_professor", "segundos", "recortes_ancorados")}
        reg["custo"]["bytes_memoria_final"] = treinos[lin["jobs"][-1]]["bytes_memoria"]
        reg["melhor_epoca"] = [treinos[j]["melhor_epoca"] for j in lin["jobs"]]
        lin_reg.append(reg)

    def chave(r):
        return (r["ordem"], r["seed"])

    def por_evento(braco, met, pares=None, grupo=None):
        acc = defaultdict(list)
        for r in lin_reg:
            if r["braco"] != braco or (pares is not None and chave(r) not in pares):
                continue
            for ev, x in r["final"].items():
                if ev in unidades and x[met] is not None and (grupo is None or x["grupo"] == grupo):
                    acc[ev].append(x[met])
        return {ev: float(np.mean(v)) for ev, v in acc.items()}

    def reps(b):
        return {chave(r) for r in lin_reg if r["braco"] == b}

    def contraste(a, b, met, grupo=None):
        pares = reps(a) & reps(b)
        fa, fb = por_evento(a, met, pares, grupo), por_evento(b, met, pares, grupo)
        return {e: fa[e]-fb[e] for e in fa if e in fb}, len(pares)

    k = len(v3.PRIMARIA)
    brutos = [(a, b, *contraste(a, b, "ap")) for a, b in v3.PRIMARIA]
    aj = holm([p_exato(list(d.values())) for _, _, d, _ in brutos])
    primaria = {}
    for i, (a, b, d, npares) in enumerate(brutos):
        primaria[f"{a} - {b}"] = {
            "repeticoes_pareadas": npares,
            "decisao_9_unidades": estimar(d, cfg["margem_ap"], k, cfg["alfa"], aj[i]),
            "retencao_6": estimar(contraste(a, b, "ap", "retencao")[0], cfg["margem_ap"], decide=False),
            "transferencia_3": estimar(contraste(a, b, "ap", "avaliacao")[0], cfg["margem_ap"], decide=False)}
    secundaria = {}
    for a, b in v3.PRIMARIA + v3.SECUNDARIA:
        secundaria[f"{a} - {b}"] = {met: estimar(contraste(a, b, met)[0], cfg["margem_iou"], decide=False)
                                    for met in ("iou_calibrado", "iou_legado", "brier_cal", "ece_cal", "ap")}
    resumo = {}
    for b in sorted({r["braco"] for r in lin_reg}):
        rs = [r for r in lin_reg if r["braco"] == b]
        ap_ev = por_evento(b, "ap")
        resumo[b] = {"repeticoes": len(rs), "ap_por_evento": ap_ev,
                     "ap_retencao": float(np.mean([ap_ev[e] for e in retencao_ev if e in ap_ev])),
                     "ap_transferencia": float(np.mean([ap_ev[e] for e in teste_ev if e in ap_ev])),
                     "iou_calibrado_por_evento": por_evento(b, "iou_calibrado"),
                     "iou_legado_por_evento": por_evento(b, "iou_legado"),
                     "pior_unidade_ap": min(ap_ev.items(), key=lambda x: x[1]) if ap_ev else None,
                     "ece": float(np.mean(list(por_evento(b, "ece").values()))),
                     "ece_cal": float(np.mean(list(por_evento(b, "ece_cal").values()))),
                     "controle_fp_calibrado": float(np.mean([v["fp_calibrado"] for r in rs for v in r["controle"].values()])),
                     "controle_fp_legado": float(np.mean([v["fp_legado"] for r in rs for v in r["controle"].values()])),
                     "classes_finais": dict(__import__("collections").Counter(
                         x["classe"] for r in rs for ev, x in r["final"].items() if ev in unidades)),
                     "custo_medio": {c: float(np.mean([r["custo"][c] for r in rs])) for c in rs[0]["custo"]}}
        if b != "reinicio_acumulado":
            for met in ("ap", "iou_calibrado"):
                esq = defaultdict(list)
                aqs = defaultdict(list)
                for r in rs:
                    for c, v in r[f"esquecimento_{met}"].items():
                        esq[c].append(v)
                    for c, v in r[f"aquisicao_{met}"].items():
                        if v is not None:
                            aqs[c].append(v)
                resumo[b][f"esquecimento_{met}"] = {c: float(np.mean(v)) for c, v in esq.items()}
                resumo[b][f"aquisicao_{met}"] = {c: float(np.mean(v)) for c, v in aqs.items()}
    por_ordem = {}
    for a, b in v3.PRIMARIA:
        por_ordem[f"{a} - {b}"] = {}
        for o in range(len(p["ordens"])):
            pares = {x for x in reps(a) & reps(b) if x[0] == o}
            fa, fb = por_evento(a, "ap", pares), por_evento(b, "ap", pares)
            if fa and fb:
                por_ordem[f"{a} - {b}"][o] = float(np.mean([fa[e]-fb[e] for e in fa if e in fb]))
    return {"versao": v3.VERSAO, "assinatura": ci.hash_json(p), "unidades": unidades, "retencao": retencao_ev,
            "transferencia": teste_ev, "primaria": primaria, "secundaria": secundaria, "resumo": resumo,
            "por_ordem_descritivo": por_ordem, "linhagens": lin_reg, "piso": p["primaria"]["piso_discreto"],
            "promovido": False}


def cartoes(saida, a, p):
    d = Path(saida) / "cartoes"
    d.mkdir(exist_ok=True)
    for r in a["linhagens"]:
        j = r["jobs"][-1]
        ci.gravar_json(d / f"{r['braco']}_o{r['ordem']}_s{r['seed']}.json", {
            "versao": v3.VERSAO, "modelo": f"UNetPequena base {p['config']['base']}, pos-evento duplicado, 4 canais",
            "braco": r["braco"], "especificacao": v3.BRACOS.get(r["braco"], "reinicio com passos acumulados"),
            "ordem_contextos": r["ctx"], "semente": r["seed"], "pesos": f"jobs/{j}/melhor.pt",
            "sha256_peso": sha(Path(saida) / "jobs" / j / "melhor.pt"),
            "alvo": "agua observada nao permanente (Sen1 fora de JRC; GEOID classe 2)",
            "normalizacao": "fixa sem dado", "calibracao": {"T": r["T"], "limiar": r["limiar"], "grupo": "Escocia EMSR698"},
            "desempenho_final_por_evento": r["final"], "controle_alemanha": r["controle"], "custo": r["custo"],
            "uso_pretendido": "pesquisa em desenvolvimento", "promovido": False,
            "fora_do_uso": ["svc02", "escore fisico", "verdade brasileira", "probabilidade territorial", "previsao"]})


def executar(saida=v3.SAIDA):
    a = analisar(saida)
    cartoes(saida, a, dc.ler(Path(saida) / "protocolo.json"))
    ci.gravar_json(Path(saida) / "analise.json", a)
    return a


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("saida", type=Path, nargs="?", default=v3.SAIDA)
    r = executar(ap.parse_args().saida)
    for c, x in r["primaria"].items():
        d = x["decisao_9_unidades"]
        print(c, d.get("estado"), round(d.get("media", 0), 4), d.get("a_favor"))
