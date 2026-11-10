"""Analises pre-registradas do pre-treino v2 e do ciclo autonomo v1, lidas so de artefatos.

Aplica as regras gravadas em cada protocolo (unidade = evento; Holm sobre Wilcoxon
exato; TOST 1-2*alfa/k). Calibracao por modelo no grupo da Escocia quando a medida
existe. Nao treina nem escolhe nada pelo resultado.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import analise_continuidade_v3 as an
import calibracao_decisao as cd
import campanha_independencia_autonomia as ci
import diagnostico_competencia as dc

REPO = Path(__file__).resolve().parents[2]


def _aps(av, grupos=("retencao", "avaliacao")):
    out = {}
    for g in grupos:
        for ev, e in av[g].items():
            a = cd.ap(*cd.arrays(e))
            if a is not None:
                out[ev] = a
    return out


def _iou_cal(av):
    cal = list(av["calibracao"].values())
    T = cd.ajustar_temperatura(cal)
    t = cd.escolher_limiar(cal, T)
    return {ev: cd.medir_evento(e, t, T)["calibrado"]["iou"] for g in ("retencao", "avaliacao") for ev, e in av[g].items()}


def analisar_pretreino(saida=REPO / "local_runs/pretreino-sar/v2"):
    saida = Path(saida)
    p = dc.ler(saida / "protocolo.json")
    concl = dc.ler(saida / "concluido.json")
    cfg = p["config"]
    tabela = defaultdict(lambda: defaultdict(list))
    iou = defaultdict(lambda: defaultdict(list))
    for jid in concl["jobs"]:
        if not jid.startswith("aj_"):
            continue
        av = dc.ler(saida / "jobs" / jid / "avaliacao.json")
        partes = jid.split("_")
        longo = "_longo" in jid
        ini = jid[3:].split("_f")[0].replace("_longo", "")
        f = int(jid.split("_f")[1].split("_")[0])
        chave = f"{ini}{'_longo' if longo else ''}@{f}"
        for ev, a in _aps(av).items():
            tabela[chave][ev].append(a)
        for ev, x in _iou_cal(av).items():
            if x is not None:
                iou[chave][ev].append(x)
    media = {k: {ev: float(np.mean(v)) for ev, v in d.items()} for k, d in tabela.items()}
    k = len(p["primaria"]["contrastes"])
    brutos = [(a, b, {ev: media[a][ev]-media[b][ev] for ev in media.get(a, {}) if ev in media.get(b, {})})
              for a, b in p["primaria"]["contrastes"]]
    aj = an.holm([an.p_exato(list(d.values())) for *_, d in brutos])
    prim = {f"{a} - {b}": an.estimar(d, cfg["margem_ap"], k, cfg["alfa"], aj[i]) for i, (a, b, d) in enumerate(brutos)}
    sec = {}
    for f in cfg["fracoes"]:
        for a, b in (("mae", "aleatorio"), ("mae_speckle", "aleatorio"), ("mae_speckle", "mae")):
            ka, kb = f"{a}@{f}", f"{b}@{f}"
            if ka in media and kb in media:
                sec[f"{ka} - {kb}"] = an.estimar({ev: media[ka][ev]-media[kb][ev] for ev in media[ka] if ev in media[kb]},
                                                 cfg["margem_ap"], decide=False)
    custo = {}
    for jid in concl["jobs"]:
        t = dc.ler(saida / "jobs" / jid / "treino.json")
        custo[jid] = {"passos": t.get("passos") or t.get("custo", {}).get("passos"),
                      "segundos": t.get("segundos") or t.get("custo", {}).get("segundos")}
    r = {"versao": p["versao"], "ap_media_por_braco": {k2: float(np.mean(list(v.values()))) for k2, v in media.items()},
         "ap_por_evento": media,
         "iou_calibrado_media": {k2: float(np.mean([np.mean(v) for v in d.values()])) for k2, d in iou.items()},
         "primaria": prim, "secundaria": sec, "estagio_longo": dc.ler(saida / "decisao_estagio_longo.json"),
         "custo": custo, "regime": "exposicao offline ao pote de treino; nao e aprendizado online causal",
         "promovido": False}
    ci.gravar_json(saida / "analise.json", r)
    return r


def analisar_ciclo(saida=REPO / "local_runs/ciclo-autonomo/v1"):
    saida = Path(saida)
    p = dc.ler(saida / "protocolo.json")
    cfg = p["config"]
    por_pol = defaultdict(lambda: defaultdict(list))
    livros = {}
    for o in range(len(p["ordens"])):
        for nome in p["politicas"]:
            pasta = saida / f"{nome}_o{o}"
            regs = [json.loads(x) for x in (pasta / "livro_razao.jsonl").read_text(encoding="utf-8").splitlines()]
            livros[f"{nome}_o{o}"] = regs
            for ev, a in _aps(dc.ler(pasta / "medida_final.json")).items():
                por_pol[nome][ev].append(a)
    media = {n: {ev: float(np.mean(v)) for ev, v in d.items()} for n, d in por_pol.items()}
    k = len(p["primaria"]["contrastes"])
    brutos = [(a, b, {ev: media[a][ev]-media[b][ev] for ev in media[a] if ev in media[b]}) for a, b in p["primaria"]["contrastes"]]
    aj = an.holm([an.p_exato(list(d.values())) for *_, d in brutos])
    prim = {f"{a} - {b}": an.estimar(d, cfg["margem_ap"], k, cfg["alfa"], aj[i]) for i, (a, b, d) in enumerate(brutos)}
    resumo = {}
    for nome in p["politicas"]:
        regs = [r for k2, v in livros.items() if k2.startswith(nome + "_o") for r in v]
        decisoes = [r["decisao"] for r in regs]
        ganhos = [r["portao"]["aquisicao_candidato"] - r["portao"]["aquisicao_vigente"] for r in regs
                  if r.get("portao") and r["portao"]["aquisicao_vigente"] is not None]
        resumo[nome] = {"aceitas": decisoes.count("aceita"), "recusadas": decisoes.count("recusada"),
                        "sem_admissao": decisoes.count("sem_admissao"),
                        "ganho_aquisicao_portao_mediano": float(np.median(ganhos)) if ganhos else None,
                        "ganho_aquisicao_por_chip": float(np.median(ganhos))/cfg["orcamento_por_contexto"] if ganhos else None,
                        "passos_totais": int(sum(r.get("custo", {}).get("passos", 0) for r in regs)),
                        "passos_em_candidatos_recusados": int(sum(r.get("custo", {}).get("passos", 0) for r in regs
                                                                   if r["decisao"] == "recusada")),
                        "anotacoes": int(sum(len(r["selecionados"]) for r in regs)),
                        "ap_final_media": float(np.mean(list(media[nome].values())))}
    r = {"versao": p["versao"], "primaria": prim, "resumo": resumo, "ap_por_evento": media,
         "nao_e": "regra de selecao declarada; nao e politica aprendida", "promovido": False}
    ci.gravar_json(saida / "analise.json", r)
    return r


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["pretreino"]:
        print(json.dumps(analisar_pretreino()["primaria"], ensure_ascii=False, indent=1)[:3000])
    else:
        print(json.dumps(analisar_ciclo()["primaria"], ensure_ascii=False, indent=1)[:3000])
