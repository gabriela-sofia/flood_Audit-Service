"""Campanha `continuidade_sar_v3`: continuacao, replay_m1, LwF legado x mascarado e hibrido.

Pergunta primaria (pre-registrada antes de qualquer ajuste): com orcamento igual,
o hibrido LwF mascarado + replay de 1 chip por contexto antigo melhora a
discriminacao (AP) do modelo final sobre cada componente, e a correcao da
mascara de distilacao muda o LwF? Unidade = evento (6 reservas de retencao +
3 eventos de avaliacao = 9). Ordens e sementes sao repeticoes subordinadas.

Dados pelo adapter `fontes_sar` (alvo comum de agua observada nao permanente),
normalizacao fixa sem dado, 5 anotacoes por contexto (teto de Madagascar).
Grupos separados: checkpoint = validacao Sen1 (Nigeria, Somalia); calibracao e
limiar = Escocia (EMSR698) inteira; controle de fundo/permanente = Alemanha
(EMSR728) inteira, nunca treinada; decisao = reservas de retencao + avaliacao.
Reservas EMSR868/871/926/927/933 e heldout GEOID nunca entram. Nada e promovido.

v3_1 (11/10/2026): a v3 parou no job 142/318 porque o chip de memoria sorteado
para Madagascar (semente 7) nao tem pixel rotulado visivel depois das mascaras
(2 dos 5 chips de ajuste de Madagascar ficam fora do SAR posterior). Correcao:
a memoria m1 admite, na MESMA permutacao da v3, o primeiro chip com suporte
rotulado. Anotacoes, grupos, bracos e regras de decisao nao mudam. O id do job
so muda onde a memoria efetiva mudou (e nos descendentes); jobs concluidos da
v3 com o mesmo id sao adotados com peso e memoria conferidos.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

import campanha_independencia_autonomia as ci
import diagnostico_competencia as dc
import fontes_sar as fs
import nucleo_sar as nu
from dados_pares_geograficos import materializar_cache, particao, sha

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/manutencao"))

VERSAO = "continuidade_sar_v3_1"
SAIDA = REPO / "local_runs/aprendizado-continuo/sar_v3_1"
ANTERIOR = REPO / "local_runs/aprendizado-continuo/sar_v3"
MUDANCA_V3_1 = ("memoria m1 admite o primeiro chip da mesma permutacao com pixel rotulado visivel; "
                "a v3 sorteava um chip sem suporte e parava")
V1 = REPO / "local_runs/aprendizado-continuo/sar_v1"
PREPARO = REPO / ("local_runs/sar-observacoes-datadas/preparo-retencao/"
                  "303ec29c888715285d235378af7784c84293743b63b4b92e79dd97c3d6fe3320/manifesto.json")
RESERVAS = {"EMSR868", "EMSR871", "EMSR926", "EMSR927", "EMSR933"}
NOVO, VALIDACAO_NOVA, CONTROLE = "EMSR424", "EMSR698", "EMSR728"
CONFIG = {"anotacoes_por_contexto": 5, "passos_por_etapa": 384, "passos_por_epoca": 24, "batch": 16, "base": 4,
          "recorte": 64, "lr": 1e-3, "threads": 2, "modo": "pos", "sementes": [7, 11], "semente_custo": 7,
          "margem_ap": .025, "margem_iou": .025, "alfa": .05, "limiar_legado": .2}
LWF = {"lambda": 1., "T": 2.}
BRACOS = {
    "continuacao": {"memoria": False, "lwf": None},
    "replay_m1": {"memoria": True, "lwf": None},
    "lwf_legado": {"memoria": False, "lwf": {**LWF, "mascarada": False}},
    "lwf_mascarado": {"memoria": False, "lwf": {**LWF, "mascarada": True}},
    "hibrido": {"memoria": True, "lwf": {**LWF, "mascarada": True}},
}
PRIMARIA = [("lwf_mascarado", "continuacao"), ("lwf_mascarado", "lwf_legado"),
            ("hibrido", "lwf_mascarado"), ("hibrido", "replay_m1")]
SECUNDARIA = [("replay_m1", "continuacao"), ("lwf_legado", "continuacao"), ("hibrido", "continuacao"),
              ("reinicio_acumulado", "hibrido")]
PAUSA_A_CADA_S, PAUSA_S = 1200, 35


def ordens(ctx):
    return [ctx[i:] + ctx[:i] for i in range(len(ctx))]


def piso_wilcoxon(n, k, alfa=.05):
    pmin = 2/2**n
    return {"n": n, "familia": k, "p_min": pmin, "holm_pode_rejeitar": k*pmin < alfa,
            "nao_e": "limite discreto ideal; nao e poder estatistico"}


V2_ANALISE = REPO / "local_runs/aprendizado-continuo/sar_v2_1/analise.json"


def dimensionar(v2_analise=V2_ANALISE, n=9, k=len(PRIMARIA), sims=20000, margem=.025):
    """Chance de cada tipo de contraste passar na regra com n eventos, usando efeito e dispersao do v2.

    Simula diferencas por evento ~ Normal(media_v2, dp_v2); conta ganho (Holm no primeiro passo:
    p_exato*k < alfa e IC95 t > margem) e equivalencia (TOST 1-2*alfa/k). Aproximacao declarada.
    """
    from scipy import stats as st
    if not Path(v2_analise).exists():
        return {"estado": "sem_referencia_v2"}
    a = dc.ler(v2_analise)
    ref = {"lwf_menos_continuacao": ("F3_metodos", "lwf - continuacao"),
           "replay_m1_menos_continuacao": ("F1_memoria", "replay_m1 - continuacao"),
           "replay_total_menos_m4": ("F1_memoria", "replay_total - replay_m4")}
    rng = np.random.default_rng(20261011)
    out = {"n": n, "familia": k, "sims": sims, "margem_ap": margem,
           "fonte": {"arquivo": str(v2_analise), "sha256": sha(v2_analise)}}
    for nome, (fam, c) in ref.items():
        d = np.array(list(a["familias"][fam]["contrastes"][c]["ap"]["oito_eventos"]["diferencas_por_evento"].values()))
        mu, dp = float(d.mean()), float(d.std(ddof=1))
        x = rng.normal(mu, dp, (sims, n))
        se = x.std(1, ddof=1)/np.sqrt(n)
        m = x.mean(1)
        lo = m - st.t.ppf(.975, n-1)*se
        q = st.t.ppf(1-(2*.05/k)/2, n-1)
        sinais = (x > 0).sum(1)
        p_aprox = np.array([2*st.binom.sf(s-1, n, .5) if s > n/2 else 1. for s in sinais])
        ganho = (np.minimum(p_aprox, 1)*k < .05) & (lo > margem)
        equiv = (m - q*se > -margem) & (m + q*se < margem)
        out[nome] = {"media_v2": mu, "dp_v2": dp, "p_ganho": float(ganho.mean()), "p_equivalencia": float(equiv.mean()),
                     "nota": "teste de sinais como aproximacao conservadora do Wilcoxon exato"}
    return out


def preparar(v1_dir=V1, preparo=PREPARO, config=None):
    p1 = dc.ler(Path(v1_dir) / "protocolo.json")
    cfg = {**CONFIG, **(config or {})}
    campanha = Path(p1["campanha"])
    inv = dc.ler(campanha / "inventario.json")
    eventos = dc.ler(campanha / "protocolo.json")["ordem_eventos"]
    tr, va, te = particao(inv["pares"], eventos[:3], eventos[3:5])
    pr = dc.ler(preparo)
    if any(c["evento"] in RESERVAS for c in pr["chips"]):
        raise ValueError("preparo contem reserva protegida")
    g = fs.chips_geoid(pr)
    canon = p1["contextos_em_ordem"] + [NOVO]
    anot = {}
    for s in cfg["sementes"]:
        anot[str(s)] = {}
        for i, e in enumerate(p1["contextos_em_ordem"], 1):
            itens = [x for x in tr if x["nome"] in p1["divisoes"][e]["ajuste"]]
            anot[str(s)][e] = ci.escolher(itens, "aleatoria", cfg["anotacoes_por_contexto"], s*100+i)
        anot[str(s)][NOVO] = sorted(pr["divisoes"][NOVO]["ajuste"])[:cfg["anotacoes_por_contexto"]]
    grupos = {
        "validacao_checkpoint": sorted(r["nome"] for r in va),
        "calibracao": sorted(c["nome"] for c in g if c["evento"] == VALIDACAO_NOVA),
        "controle": sorted(c["nome"] for c in g if c["evento"] == CONTROLE),
        "retencao": sorted([n for d in p1["divisoes"].values() for n in d["retencao"]]
                           + pr["divisoes"][NOVO]["retencao"]),
        "avaliacao": sorted(r["nome"] for r in te)}
    nomes_treino = {n for s in anot.values() for c in s.values() for n in c}
    usados_fora = set().union(*[set(v) for k, v in grupos.items()])
    if nomes_treino & usados_fora:
        raise ValueError("chip anotado tambem em grupo de medida")
    seqs = [{"ordem": o, "seed": s, "custo": s == cfg["semente_custo"]}
            for s in cfg["sementes"] for o in range(len(canon))]
    protocolo = {
        "versao": VERSAO, "v1_protocolo_sha256": sha(Path(v1_dir) / "protocolo.json"),
        "preparo_sha256": sha(preparo), "campanha": str(campanha), "inventario": p1["inventario"],
        "config": cfg, "contextos_canonicos": canon, "ordens": ordens(canon), "sequencias": seqs,
        "anotacoes": anot, "grupos": grupos, "bracos": BRACOS, "lwf": LWF,
        "primaria": {"desfecho": "AP aproximada (1000 faixas) do modelo final por evento",
                     "unidades": "9 eventos: 6 reservas de retencao + 3 de avaliacao",
                     "contrastes": [list(c) for c in PRIMARIA], "correcao": "Holm sobre Wilcoxon exato bilateral",
                     "ganho_ou_perda": "p_holm < 0,05 E IC95 t por evento alem de +-margem_ap",
                     "equivalencia": "TOST: IC t de nivel 1-2*alfa/k inteiro dentro de +-margem_ap",
                     "resto": "INDECIDIVEL", "piso_discreto": piso_wilcoxon(9, len(PRIMARIA))},
        "secundaria": {"contrastes": [list(c) for c in SECUNDARIA], "regra": "estimacao com IC, sem decisao",
                       "desfechos": ["IoU legado 0,2", "IoU no limiar calibrado na Escocia", "Brier/NLL/ECE",
                                     "retencao (6) e transferencia (3) separadas", "pior unidade", "esquecimento/BWT",
                                     "aquisicao do contexto novo", "falsos positivos no controle Alemanha", "custo"]},
        "grupos_papel": {"validacao_checkpoint": "so escolhe checkpoint (CE)",
                         "calibracao": "so ajusta temperatura e limiar por modelo",
                         "controle": "so mede falso positivo; sem classe 2",
                         "retencao": "medida final; contextos conhecidos, recorte espacial separado",
                         "avaliacao": "medida final; eventos nunca treinados"},
        "dimensionamento_prospectivo": dimensionar() if config is None else {"estado": "teste"},
        "codigo": {str(f): sha(f) for f in fontes()},
        "regime": "desenvolvimento em dados ja examinados", "promovido": False, "altera_svc02": False,
        "consome_reservas": False, "torch": torch.__version__}
    todos = {r["nome"]: {**r, "fonte": "sen1"} for r in tr + va + te}
    todos.update({c["nome"]: c for c in g})
    suporte = {s: {c: {n: (lambda x: x["positivo"] + x["negativo"])(fs.contagem(todos[n], cfg["modo"])) for n in ns}
                   for c, ns in a.items()} for s, a in anot.items()}
    memoria = admitir_memoria(anot, suporte, canon)
    protocolo.update({"pixels_rotulados_por_anotacao": suporte, "memoria_m1": memoria})
    if config is None:  # so a campanha real adota da v3; testes e fumaca nunca
        protocolo["versao_anterior"] = {"pasta": str(ANTERIOR), "versao": "continuidade_sar_v3", "mudanca": MUDANCA_V3_1}
    return protocolo, inv, todos


def admitir_memoria(anot, suporte, canon):
    """Por semente e contexto: primeiro chip da permutacao m1 com pixel rotulado visivel."""
    mem = {s: {c: next((n for n in _permutacao_m1(ns, int(s), canon.index(c)) if suporte[s][c][n] > 0), None)
               for c, ns in a.items()} for s, a in anot.items()}
    if any(n is None for m in mem.values() for n in m.values()):
        raise ValueError("contexto anotado sem nenhum chip com pixel rotulado")
    return mem


def _permutacao_m1(nomes, seed, i_ctx):
    return [str(n) for n in np.random.default_rng([seed, i_ctx, 17]).permutation(nomes)]


def fontes():
    return [Path(__file__), Path(nu.__file__), Path(fs.__file__), Path(ci.__file__), Path(dc.__file__),
            REPO / "scripts/sar/continuidade_v2.py", REPO / "scripts/sar/dados_pares_geograficos.py",
            REPO / "scripts/sar/modelo_unet.py", REPO / "scripts/neural/proveniencia_de_peso.py",
            REPO / "scripts/sar/calibracao_decisao.py"]


def job_id(c):
    return ci.hash_json(c)[:20]


def planejar(p):
    cfg, por_id, ordem, linhagens = p["config"], {}, [], []

    def reg(j):
        if j["id"] not in por_id:
            por_id[j["id"]] = j
            ordem.append(j["id"])
        return j["id"]
    for s in p["sequencias"]:
        ctx = p["ordens"][s["ordem"]]
        for nome, spec in BRACOS.items():
            pai, ids = None, []
            for k in range(1, len(ctx)+1):
                rel = {"etapa1": True} if k == 1 else spec
                c = {"spec": rel, "seed": s["seed"], "ctx": ctx[:k], "pai": pai, "passos": cfg["passos_por_etapa"]}
                if k > 1 and spec["memoria"] and "memoria_m1" in p:
                    mem = memoria_m1(p, ctx, k, s["seed"])
                    if mem != memoria_v3(p, ctx, k, s["seed"]):
                        c["memoria"] = mem  # id novo so onde a memoria efetiva mudou
                j = {"id": job_id(c), "braco": nome, "spec": spec if k > 1 else {"memoria": False, "lwf": None},
                     "seed": s["seed"], "ordem": s["ordem"], "ctx": ctx, "etapa": k, "pai": pai,
                     "passos": cfg["passos_por_etapa"], "rho": .5, "checkpoint": "validacao", "modo": cfg["modo"]}
                pai = reg(j)
                ids.append(pai)
            linhagens.append({"ordem": s["ordem"], "seed": s["seed"], "braco": nome, "jobs": ids})
        if s["custo"]:
            k = len(ctx)
            c = {"spec": "reinicio_acumulado", "seed": s["seed"], "ctx": ctx, "passos": k*cfg["passos_por_etapa"]}
            j = {"id": job_id(c), "braco": "reinicio_acumulado", "spec": {"memoria": False, "lwf": None, "uniao": True},
                 "seed": s["seed"], "ordem": s["ordem"], "ctx": ctx, "etapa": k, "pai": None,
                 "passos": k*cfg["passos_por_etapa"], "rho": .5, "checkpoint": "validacao", "modo": cfg["modo"]}
            linhagens.append({"ordem": s["ordem"], "seed": s["seed"], "braco": "reinicio_acumulado",
                              "jobs": [reg(j)]})
    return por_id, ordem, linhagens


def memoria_m1(p, ctx, k, seed):
    """Um chip admitido (com pixel rotulado) por contexto antigo; fixado no protocolo."""
    return [p["memoria_m1"][str(seed)][c] for c in ctx[:k-1]]


def memoria_v3(p, ctx, k, seed):
    """Regra da v3: primeiro chip da permutacao, mesmo sem suporte rotulado."""
    canon, anot = p["contextos_canonicos"], p["anotacoes"][str(seed)]
    return [_permutacao_m1(anot[c], seed, canon.index(c))[0] for c in ctx[:k-1]]


def executar_job(saida, p, todos, j, por_id):
    cfg, anot = p["config"], p["anotacoes"][str(j["seed"])]
    ctx, k = j["ctx"], j["etapa"]
    val = [todos[n] for n in p["grupos"]["validacao_checkpoint"]]
    if j["spec"].get("uniao"):
        novas, antigas = [todos[n] for c in ctx for n in anot[c]], []
    else:
        novas = [todos[n] for n in anot[ctx[k-1]]]
        antigas = [todos[n] for n in memoria_m1(p, ctx, k, j["seed"])] \
            if j["spec"]["memoria"] and k > 1 else []
    inicial = None
    if j["pai"]:
        tp = dc.ler(saida / "jobs" / j["pai"] / "treino.json")
        inicial = nu.estado(saida / "jobs" / j["pai"], tp)
    job = {"id": j["id"], "seed": j["seed"], "etapa": k, "passos": j["passos"], "rho": j["rho"],
           "lwf": j["spec"]["lwf"], "checkpoint": j["checkpoint"], "modo": j["modo"]}
    return nu.ajustar(saida / "jobs" / j["id"], job, novas, antigas, val, cfg, inicial=inicial)


def adotar(saida, p, j):
    """Copia o job concluido da v3 com o mesmo id se peso, pai e memoria conferem."""
    va = p.get("versao_anterior")
    if not va:
        return False
    origem = Path(va["pasta"]) / "jobs" / j["id"]
    if not (origem / "treino.json").exists():
        return False
    t = dc.ler(origem / "treino.json")
    esperada = [] if j["spec"].get("uniao") or not j["spec"]["memoria"] or j["etapa"] == 1 \
        else memoria_m1(p, j["ctx"], j["etapa"], j["seed"])
    pai = dc.ler(saida / "jobs" / j["pai"] / "treino.json")["hash_estado"] if j["pai"] else None
    if t["id_job"] != j["id"] or t["pai"] != pai or list(t["memoria"]) != sorted(esperada):
        raise ValueError(f"job da v3 divergente: {j['id']}")
    nu.estado(origem, t)  # confere o hash do peso
    destino = saida / "jobs" / j["id"]
    if destino.exists():
        shutil.rmtree(destino)  # tentativa interrompida de adocao, nunca job concluido
    shutil.copytree(origem, destino, ignore=shutil.ignore_patterns("retomada.pt"))
    ci.gravar_json(destino / "adocao.json", {"origem": str(origem), "versao_origem": va["versao"],
                                            "protocolo_origem_sha256": sha(Path(va["pasta"]) / "protocolo.json"),
                                            "motivo": va["mudanca"]})
    return True


def fila(saida, p, todos, por_trava=True, esperar_min=240.):
    import executar_serializado as serial
    por_id, ordem, _ = planejar(p)
    torch.set_num_threads(p["config"]["threads"])
    torch.use_deterministic_algorithms(True)
    pausa = time.monotonic()
    for jid in ordem:
        if (saida / "jobs" / jid / "treino.json").exists():
            continue
        if adotar(saida, p, por_id[jid]):
            continue
        dono = None
        if por_trava:
            if not serial.energia()["na_tomada"]:
                return {"estado": "aguarda_tomada"}
            dono = serial.tomar([sys.executable, str(Path(__file__)), "job", jid], esperar_min, False)
            if dono is None:
                return {"estado": "trava_ocupada"}
        try:
            executar_job(saida, p, todos, por_id[jid], por_id)
        finally:
            if dono:
                serial.liberar(dono)
        if por_trava and time.monotonic() - pausa > PAUSA_A_CADA_S:
            time.sleep(PAUSA_S)
            pausa = time.monotonic()
    ci.gravar_json(saida / "treino_concluido.json", {"assinatura": ci.hash_json(p), "jobs": ordem})
    return {"estado": "treino_concluido"}


def avaliar(saida, p, todos, por_trava=True, esperar_min=240.):
    """So depois de todo o treino; cada checkpoint medido uma vez em todos os grupos."""
    import executar_serializado as serial
    concl = dc.ler(saida / "treino_concluido.json")
    if concl["assinatura"] != ci.hash_json(p):
        raise ValueError("treino concluido de outro protocolo")
    for f, h in p["codigo"].items():
        if sha(f) != h:
            raise ValueError(f"codigo mudou depois do pre-registro: {f}")
    torch.set_num_threads(p["config"]["threads"])
    pend = [j for j in concl["jobs"] if not (saida / "jobs" / j / "avaliacao.json").exists()]
    for lote in [pend[i:i+15] for i in range(0, len(pend), 15)]:
        dono = serial.tomar([sys.executable, str(Path(__file__)), "avaliar"], esperar_min, False) if por_trava else None
        if por_trava and dono is None:
            return {"estado": "trava_ocupada"}
        try:
            for jid in lote:
                t = dc.ler(saida / "jobs" / jid / "treino.json")
                m = nu.novo_modelo(t["job"]["seed"], p["config"]["base"])
                m.load_state_dict(nu.estado(saida / "jobs" / jid, t)["estado"])
                r = {g: nu.medir(m, [todos[n] for n in nomes], p["config"]["modo"])
                     for g, nomes in p["grupos"].items()}
                ci.gravar_json(saida / "jobs" / jid / "avaliacao.json", r)
                print(json.dumps({"avaliado": jid}), flush=True)
        finally:
            if dono:
                serial.liberar(dono)
    ci.gravar_json(saida / "avaliacao_concluida.json", {"assinatura": ci.hash_json(p)})
    return {"estado": "avaliacao_concluida"}


def executar(saida=SAIDA, config=None, v1_dir=V1, preparo=PREPARO, somente_preparar=False, por_trava=True,
             esperar_min=240.):
    saida = Path(saida).resolve()
    p, inv, todos = preparar(v1_dir, preparo, config)
    saida.mkdir(parents=True, exist_ok=True)
    dc.imutavel(saida / "protocolo.json", p)
    (saida / "codigo").mkdir(exist_ok=True)
    for f, h in p["codigo"].items():
        d = saida / "codigo" / Path(f).name
        if not d.exists():
            shutil.copyfile(f, d)
        if sha(d) != h:
            raise ValueError("snapshot de codigo divergente; nova versao necessaria")
    por_id, ordem, lin = planejar(p)
    if somente_preparar:
        return {"estado": "preparado", "assinatura": ci.hash_json(p), "jobs": len(ordem), "linhagens": len(lin),
                "passos": sum(por_id[j]["passos"] for j in ordem)}
    materializar_cache(inv, Path(p["campanha"]) / "cache")
    if not (saida / "treino_concluido.json").exists():
        e = fila(saida, p, todos, por_trava, esperar_min)
        if e["estado"] != "treino_concluido":
            return e
    if not (saida / "avaliacao_concluida.json").exists():
        e = avaliar(saida, p, todos, por_trava, esperar_min)
        if e["estado"] != "avaliacao_concluida":
            return e
    return {"estado": "pronto_para_analise"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("acao", choices=["preparar", "executar"])
    ap.add_argument("--saida", type=Path, default=SAIDA)
    ap.add_argument("--esperar-min", type=float, default=240.)
    a = ap.parse_args()
    r = executar(a.saida, somente_preparar=a.acao == "preparar", esperar_min=a.esperar_min)
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
