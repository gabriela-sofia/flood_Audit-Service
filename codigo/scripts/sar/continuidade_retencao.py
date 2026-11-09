"""Aprendizado continuo SAR: reinicio, continuacao e continuacao com replay.

Versao `continuidade_sar_v1` (09/10/2026). Contexto = evento conhecido do
papel treino do pacote ativo. Cada contexto tem anotacoes liberadas e uma
reserva espacial de retencao que nunca entra no ajuste. Os bracos partem da
mesma etapa 1, com a mesma semente, capacidade, normalizacao, objetivo e
passos marginais; so muda o que cada etapa herda (pesos, exemplos antigos).

Decisoes usam apenas a validacao: checkpoint por CE e o portao da aquisicao
continuada. Reservas de retencao e eventos de avaliacao so sao inferidos
depois de todos os ajustes. Dados ja examinados: desenvolvimento, nunca
confirmacao inedita. Nada e promovido; escore fisico e svc02 nao mudam.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import campanha_independencia_autonomia as ci
import diagnostico_competencia as dc
import aquisicao_apos_diagnostico as aq
from dados_pares_geograficos import materializar_cache, normalizacao, particao, sha

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/manutencao"))

VERSAO = "continuidade_sar_v1"
CAMPANHA = REPO / "local_runs/sar-independencia/campanha_2026_10_06"
DIAGNOSTICO = REPO / "local_runs/sequencia-aprendizado/2026_10_07/diagnostico"
PACOTE = REPO / ("local_runs/pacote-de-treino/c7a3ba7e9256d042d8417dd0b0b12c2d8ef78619514ca996a943b8b2c6adeba6"
                 "/manifesto.json")
SAIDA = REPO / "local_runs/aprendizado-continuo/sar_v1"

CONFIG = {"anotacoes_por_contexto": 12, "fracao_retencao": 1/3, "adjacencia_larguras": 1.5,
          "passos_por_etapa": 384, "passos_por_epoca": 24, "batch": 16, "base": 4, "recorte": 64,
          "lr": 1e-3, "threads": 2, "sementes": [7, 11, 23], "relevancia": .025, "minimo_eventos": 6}
BRACOS = {
    "reinicio": {"inicio": "aleatorio", "dados": "uniao", "lote": "uniforme", "passos": "S"},
    "continuacao": {"inicio": "etapa_anterior", "dados": "novo", "lote": "uniforme", "passos": "S"},
    "continuacao_replay": {"inicio": "etapa_anterior", "dados": "novo_e_antigo", "lote": "metade_antiga",
                           "passos": "S"},
    "reinicio_passos_acumulados": {"inicio": "aleatorio", "dados": "uniao", "lote": "uniforme",
                                   "passos": "k*S", "etapas": "somente a final"},
}
POLITICAS_E2 = ("entropia", "fixa")
OTIMIZADOR = "Adam novo em cada etapa (lr fixo); momentos nao transportados entre etapas"


# ---------------------------------------------------------------- particao em contextos

def _centros(linhas):
    b = np.array([r["grade"]["bounds"] for r in linhas], dtype=float)
    return (b[:, 0]+b[:, 2])/2, (b[:, 1]+b[:, 3])/2, float(np.mean(b[:, 2]-b[:, 0]))


def adjacencia(linhas, larguras):
    """Chebyshev entre centros: toca, sobrepoe ou diagonal conta como vizinho."""
    x, y, w = _centros(linhas)
    return np.maximum(abs(x[:, None]-x), abs(y[:, None]-y)) <= w*larguras + 1e-12


def componentes(linhas, larguras):
    from scipy.sparse.csgraph import connected_components
    _, rotulos = connected_components(adjacencia(linhas, larguras), directed=False)
    grupos = {}
    for i, g in enumerate(rotulos):
        grupos.setdefault(int(g), []).append(i)
    return list(grupos.values())


def _chave(evento, nomes):
    return hashlib.sha256((evento + "|" + "|".join(sorted(nomes))).encode()).hexdigest()


def dividir_contexto(linhas, fracao, larguras):
    """Reserva de retencao espacialmente separada; regra fixa, sem olhar rotulo.

    Componentes de adjacencia inteiros vao para a reserva em ordem de hash. Se um
    componente concentra mais da metade dos chips (evento em bloco contiguo), a
    reserva e a faixa inicial do eixo mais longo e os vizinhos dela viram tampao,
    fora do ajuste e da reserva.
    """
    if not linhas or len({r["evento"] for r in linhas}) != 1:
        raise ValueError("contexto precisa de um unico evento")
    evento, n = linhas[0]["evento"], len(linhas)
    alvo = math.ceil(n*fracao)
    grupos = componentes(linhas, larguras)
    if max(len(g) for g in grupos) <= n/2:
        grupos.sort(key=lambda g: _chave(evento, [linhas[i]["nome"] for i in g]))
        reserva = []
        for g in grupos:
            if len(reserva) >= alvo:
                break
            reserva += g
        tampao, regra = [], "componentes_inteiros_por_hash"
    else:
        x, y, _ = _centros(linhas)
        eixo = x if np.ptp(x) >= np.ptp(y) else y
        ordem = sorted(range(n), key=lambda i: (eixo[i], linhas[i]["nome"]))
        reserva = ordem[:alvo]
        viz = adjacencia(linhas, larguras)
        tampao = [i for i in range(n) if i not in reserva and viz[i, reserva].any()]
        regra = "faixa_do_eixo_longo_com_tampao"
    ajuste = [i for i in range(n) if i not in reserva and i not in tampao]
    r = {"evento": evento, "regra": regra,
         "ajuste": sorted(linhas[i]["nome"] for i in ajuste),
         "retencao": sorted(linhas[i]["nome"] for i in reserva),
         "tampao": sorted(linhas[i]["nome"] for i in tampao)}
    conferir_separacao(linhas, r, larguras)
    return r


def conferir_separacao(linhas, divisao, larguras):
    por_nome = {r["nome"]: r for r in linhas}
    a = [por_nome[n] for n in divisao["ajuste"]]
    b = [por_nome[n] for n in divisao["retencao"]]
    if not a or not b:
        raise ValueError("ajuste ou retencao vazio")
    if set(divisao["ajuste"]) & set(divisao["retencao"]):
        raise ValueError("chip nos dois papeis")
    viz = adjacencia(a+b, larguras)[:len(a), len(a):]
    if viz.any():
        raise ValueError("ajuste vizinho da reserva de retencao")
    imagens = lambda rs: {r["hashes"][k] for r in rs for k in ("antes", "depois")}
    if imagens(a) & imagens(b):
        raise ValueError("copia de imagem entre ajuste e retencao")


def classes_presentes(linhas):
    sp, _ = dc.suporte(linhas)
    return {"positivo": sum(e["positivo"] for e in sp.values()),
            "negativo": sum(e["negativo"] for e in sp.values())}


# ---------------------------------------------------------------- protocolo

def fontes():
    return [Path(__file__), Path(ci.__file__), Path(dc.__file__), Path(aq.__file__),
            REPO / "scripts/sar/dados_pares_geograficos.py", REPO / "scripts/sar/modelo_unet.py",
            REPO / "scripts/neural/proveniencia_de_peso.py", REPO / "scripts/sar/baseline_otsu.py",
            REPO / "scripts/sar/experimento_pre_pos.py"]


def limiar_congelado(diagnostico):
    card = dc.ler(Path(diagnostico) / "card.json")
    escolha = card["aquisicao"]["escolhido"]
    if not card["aquisicao"]["liberada"] or escolha is None:
        raise ValueError("diagnostico sem competencia: nao ha limiar congelado")
    if dc.BRACOS[escolha["braco"]]["objetivo"] != "ce":
        raise ValueError("objetivo do braco escolhido diverge do protocolo")
    return {"limiar": escolha["limiar"], "braco": escolha["braco"], "modo": card["protocolo"]["config"]["modo"],
            "card_sha256": sha(Path(diagnostico) / "card.json")}


def preparar(campanha=CAMPANHA, diagnostico=DIAGNOSTICO, config=None, pacote=PACOTE):
    campanha = Path(campanha).resolve()
    cfg = {**CONFIG, **(config or {})}
    inv = dc.ler(campanha / "inventario.json")
    p = dc.ler(campanha / "protocolo.json")
    if ci.hash_json(inv) != p["inventario"]:
        raise ValueError("inventario divergente do protocolo da campanha")
    eventos = p["ordem_eventos"]
    tr, va, te = particao(inv["pares"], eventos[:3], eventos[3:5])
    contextos = [e for e in eventos[5:]]
    if sorted(contextos) != sorted({r["evento"] for r in tr}):
        raise ValueError("contextos nao coincidem com o papel treino")
    congelado = limiar_congelado(diagnostico)
    divisoes, suporte_retencao = {}, {}
    materializar_cache(inv, campanha / "cache")
    for e in contextos:
        d = dividir_contexto([r for r in tr if r["evento"] == e], cfg["fracao_retencao"], cfg["adjacencia_larguras"])
        if len(d["ajuste"]) < cfg["anotacoes_por_contexto"]:
            raise ValueError(f"{e}: ajuste menor que o orcamento de anotacoes")
        divisoes[e] = d
        suporte_retencao[e] = classes_presentes([r for r in tr if r["nome"] in d["retencao"]])
    anotacoes = {}
    for seed in cfg["sementes"]:
        anotacoes[str(seed)] = {}
        for k, e in enumerate(contextos, 1):
            itens = [r for r in tr if r["nome"] in divisoes[e]["ajuste"]]
            anotacoes[str(seed)][e] = ci.escolher(itens, "aleatoria", cfg["anotacoes_por_contexto"], seed*100+k)
    pacote_info = None
    if pacote is not None and Path(pacote).exists():
        m = dc.ler(pacote)
        pacote_info = {"manifesto": str(pacote), "sha256": sha(pacote), "hash_resumo": m.get("hash_resumo"),
                       "sar_treino": next(a for a in m["sar"]["apoio"] if a["papel"] == "treino")["eventos"]}
        if sorted(pacote_info["sar_treino"]) != sorted(contextos):
            raise ValueError("contextos divergem do papel treino do pacote ativo")
    protocolo = {
        "versao": VERSAO, "campanha": str(campanha), "inventario": ci.hash_json(inv), "pacote": pacote_info,
        "config": cfg, "contextos_em_ordem": contextos, "ordem": "ordem por hash herdada da campanha de 06/10",
        "validacao": eventos[3:5], "avaliacao_desenvolvimento": eventos[:3],
        "divisoes": divisoes, "suporte_retencao": suporte_retencao, "anotacoes": anotacoes,
        "congelado_da_validacao": congelado, "modo": congelado["modo"], "objetivo": "CE nao ponderada",
        "bracos": BRACOS, "otimizador": OTIMIZADOR,
        "normalizacao": "anotacoes da etapa 1 de cada semente; igual em todos os bracos e etapas",
        "checkpoint": "menor CE de validacao entre as epocas da propria etapa",
        "codigo": {str(f): sha(f) for f in fontes()},
        "hipoteses": hipoteses(cfg),
        "portao_e2": ("continuacao_replay final competente (piso 0,1 no limiar congelado) na validacao em todas "
                      "as sementes e mediana da IoU macro de validacao >= reinicio final - relevancia"),
        "regime": "desenvolvimento em dados conhecidos; reserva de retencao e avaliacao inferidas so no fim",
        "promovido": False, "altera_escore_suscetibilidade": False, "altera_svc02": False,
        "torch": torch.__version__, "numpy": np.__version__}
    return protocolo, inv, (tr, va, te)


def hipoteses(cfg):
    r = cfg["relevancia"]
    return {
        "P1_replay_retem": f"IoU no limiar congelado: replay - continuacao nos contextos antigos, etapa final; relevancia {r}",
        "P2_acumulo_custo_marginal_igual": f"replay - reinicio no modelo final, 5 reservas + 3 eventos de avaliacao; relevancia {r}",
        "P3_alem_de_passos": f"replay - reinicio_passos_acumulados no modelo final, 8 eventos; relevancia {r}",
        "P4_continuar_sem_replay": f"continuacao - reinicio no modelo final, 8 eventos; relevancia {r}",
        "P5_so_passos": f"reinicio_passos_acumulados - reinicio no modelo final, 8 eventos; relevancia {r}",
        "E2_selecao_continuada": f"entropia/fixa - aleatoria (=replay) no modelo final, 8 eventos; relevancia {r}",
        "regra": (f"ganho ou perda sustentados exigem >= {cfg['minimo_eventos']} eventos, Wilcoxon p < 0,05 e IC "
                  "bootstrap por evento alem da relevancia; IC inteiro dentro da relevancia = sem diferenca "
                  "relevante; resto INDECIDIVEL. Sementes agregadas dentro do evento."),
        "primaria": "P2; demais secundarias e exploratorias, sem correcao para multiplas comparacoes",
    }


# ---------------------------------------------------------------- ajuste retomavel

def ajustar(novas, antigas, lote, inicial, passos, validacao, estat, seed, etapa, cfg, pasta, assinatura):
    """Um ajuste de etapa. `inicial` None = pesos aleatorios da semente; senao herda."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    if passos % cfg["passos_por_epoca"]:
        raise ValueError("passos precisam fechar epocas")
    if lote == "metade_antiga" and not antigas:
        raise ValueError("replay sem exemplos antigos")
    if lote == "uniforme" and antigas:
        raise ValueError("lote uniforme recebe um unico conjunto")
    origem = None if inicial is None else inicial["hash_estado"]
    job = ci.hash_json({"assinatura": assinatura, "novas": [r["nome"] for r in novas],
                        "antigas": [r["nome"] for r in antigas], "lote": lote, "inicial": origem,
                        "passos": passos, "seed": seed, "etapa": etapa, "estat": estat})
    ci.semear(seed)
    modelo = ci.UNetPequena(canais_entrada=4, base=cfg["base"])
    if inicial is not None:
        modelo.load_state_dict(inicial["estado"])
    if (pasta / "treino.json").exists():
        r = dc.ler(pasta / "treino.json")
        e = torch.load(pasta / "melhor.pt", weights_only=False, map_location="cpu")
        if r["id_job"] != job or e["id_job"] != job or ci.hash_completo_do_estado(e["modelo"]) != r["hash_estado"]:
            raise ValueError(f"peso ou job divergente em {pasta}")
        modelo.load_state_dict(e["modelo"])
        return modelo, r
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"])
    epocas = passos // cfg["passos_por_epoca"]
    inicio, melhor, historico, melhor_epoca = 0, float("inf"), [], None
    retomada = pasta / "retomada.pt"
    if retomada.exists():
        r = torch.load(retomada, weights_only=False, map_location="cpu")
        if r["id_job"] != job:
            raise ValueError("retomada divergente")
        modelo.load_state_dict(r["modelo"])
        opt.load_state_dict(r["opt"])
        torch.set_rng_state(r["rng"])
        inicio, melhor, historico, melhor_epoca = r["epoca"], r["melhor"], r["historico"], r["melhor_epoca"]
    todas = novas + antigas
    for epoca in range(inicio, epocas):
        modelo.train()
        rng = np.random.default_rng([seed, etapa, epoca])
        perdas, contagem, t0 = [], {"positivo": 0, "negativo": 0, "ignorado": 0, "de_antigas": 0}, time.monotonic()
        for _ in range(cfg["passos_por_epoca"]):
            lote_xy = []
            for i in range(cfg["batch"]):
                fonte = (antigas if i % 2 else novas) if lote == "metade_antiga" else todas
                contagem["de_antigas"] += int(lote == "metade_antiga" and i % 2 == 1)
                lote_xy.append(dc.recorte(fonte, estat, cfg["modo"], rng, cfg["recorte"], {}, "uniforme", i))
            x, y = torch.stack([a for a, _ in lote_xy]), torch.stack([b for _, b in lote_xy])
            for k, valor in (("positivo", 1), ("negativo", 0), ("ignorado", -1)):
                contagem[k] += int((y == valor).sum())
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(modelo(x), y, ignore_index=-1)
            if not torch.isfinite(loss):
                raise ValueError("perda nao finita")
            loss.backward()
            opt.step()
            perdas.append(float(loss.detach()))
        va = ci.avaliar(modelo, validacao, estat, cfg["modo"])
        if va["perda"] < melhor:
            melhor, melhor_epoca = va["perda"], epoca+1
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            ci.gravar_peso(pasta / "melhor.pt", {"modelo": estado, "id_job": job, "estat": estat,
                                                 "base": cfg["base"], "modo": cfg["modo"], "assinatura": assinatura})
        historico.append({"epoca": epoca+1, "passos": cfg["passos_por_epoca"], "pixels_apresentados": contagem,
                          "perda_treino": float(np.mean(perdas)), "perda_validacao": va["perda"],
                          "segundos": time.monotonic()-t0})
        ci.gravar_peso(retomada, {"modelo": modelo.state_dict(), "opt": opt.state_dict(), "rng": torch.get_rng_state(),
                                  "id_job": job, "epoca": epoca+1, "melhor": melhor, "melhor_epoca": melhor_epoca,
                                  "historico": historico})
        print(json.dumps({"job": pasta.name, "epoca": epoca+1, "de": epocas,
                          "perda_validacao": round(va["perda"], 5)}), flush=True)
    e = torch.load(pasta / "melhor.pt", weights_only=False, map_location="cpu")
    modelo.load_state_dict(e["modelo"])
    r = {"id_job": job, "hash_estado": ci.hash_completo_do_estado(e["modelo"]), "seed": seed, "etapa": etapa,
         "inicial": origem, "lote": lote, "novas": [r["nome"] for r in novas], "antigas": [r["nome"] for r in antigas],
         "melhor_epoca": melhor_epoca, "historico": historico, "passos_executados": passos,
         "segundos": sum(h["segundos"] for h in historico), "otimizador": OTIMIZADOR,
         "checkpoint": "menor CE de validacao nesta etapa"}
    ci.gravar_json(pasta / "treino.json", r)
    return modelo, r


def estado_de(pasta, registro):
    e = torch.load(Path(pasta) / "melhor.pt", weights_only=False, map_location="cpu")
    if ci.hash_completo_do_estado(e["modelo"]) != registro["hash_estado"]:
        raise ValueError("estado herdado divergente")
    return {"estado": e["modelo"], "hash_estado": registro["hash_estado"]}


# ---------------------------------------------------------------- execucao

def _validar(job, modelo, va, estat, modo, limiar):
    f = Path(job["pasta"]) / "validacao.json"
    if f.exists():
        m = dc.ler(f)
    else:
        m = dc.medir(modelo, va, estat, modo)
        ci.gravar_json(f, m)
    job["validacao"] = dc.resumo_limiar(m, limiar)
    return m


def _linhagem(nome, seed, etapa1, ctx, anot, por_nome, va, estat, cfg, saida, assinatura, limiar, jobs,
              selecao=None, tr=None, divisoes=None):
    anterior = etapa1
    escolhidas = {ctx[0]: anot[ctx[0]]}
    for k in range(2, len(ctx)+1):
        e = ctx[k-1]
        pasta = saida / f"{nome}_s{seed}_e{k}"
        if selecao is None:
            escolhidas[e] = anot[e]
        else:
            escolhidas[e] = _selecionar(selecao, anterior, e, seed, k, tr, divisoes, estat, cfg, pasta, assinatura)
        novas = [por_nome[n] for n in escolhidas[e]]
        antigas = [por_nome[n] for c in ctx[:k-1] for n in escolhidas[c]]
        braco = BRACOS["continuacao_replay" if selecao else nome]
        if braco["inicio"] == "aleatorio":
            modelo, r = ajustar(novas+antigas, [], "uniforme", None, cfg["passos_por_etapa"], va, estat, seed, k,
                                cfg, pasta, assinatura)
        elif braco["lote"] == "metade_antiga":
            modelo, r = ajustar(novas, antigas, "metade_antiga", estado_de(anterior["pasta"], anterior["treino"]),
                                cfg["passos_por_etapa"], va, estat, seed, k, cfg, pasta, assinatura)
        else:
            modelo, r = ajustar(novas, [], "uniforme", estado_de(anterior["pasta"], anterior["treino"]),
                                cfg["passos_por_etapa"], va, estat, seed, k, cfg, pasta, assinatura)
        job = {"braco": nome, "seed": seed, "etapa": k, "contexto": e, "pasta": str(pasta), "treino": r,
               "pai": anterior["pasta"] if braco["inicio"] != "aleatorio" else None,
               "anotacoes_acumuladas": sum(len(v) for v in escolhidas.values()),
               "buffer_replay_chips": len(antigas) if braco["lote"] == "metade_antiga" else 0}
        _validar(job, modelo, va, estat, cfg["modo"], limiar)
        jobs.append(job)
        anterior = job
    return anterior


def _selecionar(politica, anterior, evento, seed, k, tr, divisoes, estat, cfg, pasta, assinatura):
    f = pasta / "selecao.json"
    if f.exists():
        s = dc.ler(f)
        if s["assinatura"] != assinatura or s["pai"] != anterior["treino"]["hash_estado"]:
            raise ValueError("selecao gravada diverge do pai")
        return s["escolhidos"]
    candidatos = [r for r in tr if r["nome"] in divisoes[evento]["ajuste"]]
    t0 = time.monotonic()
    scores = None
    if politica == "entropia":
        modelo = ci.UNetPequena(canais_entrada=4, base=cfg["base"])
        modelo.load_state_dict(estado_de(anterior["pasta"], anterior["treino"])["estado"])
        sem_rotulo = [{"nome": r["nome"], "evento": r["evento"],
                       "caminhos": {c: r["caminhos"][c] for c in ("antes", "depois")}} for r in candidatos]
        scores = aq.pontuar(modelo, sem_rotulo, estat, cfg["modo"])
    escolhidos = aq.escolher([{"nome": r["nome"], "evento": r["evento"]} for r in candidatos], politica,
                             cfg["anotacoes_por_contexto"], seed*100+k, scores)
    pasta.mkdir(parents=True, exist_ok=True)
    dc.imutavel(f, {"assinatura": assinatura, "politica": politica, "pai": anterior["treino"]["hash_estado"],
                    "escolhidos": escolhidos, "scores": scores, "segundos_selecao": time.monotonic()-t0,
                    "oracle_consultado_na_selecao": False})
    return escolhidos


def portao_e2(jobs, sementes, limiar, relevancia):
    """Usa somente a validacao dos modelos finais."""
    finais = {}
    for j in jobs:
        if j["braco"] in ("reinicio", "continuacao_replay"):
            atual = finais.get((j["braco"], j["seed"]))
            if atual is None or j["etapa"] > atual["etapa"]:
                finais[(j["braco"], j["seed"])] = j
    rep = [finais[("continuacao_replay", s)] for s in sementes]
    rei = [finais[("reinicio", s)] for s in sementes]
    mr = float(np.median([j["validacao"]["macro_iou"] or 0 for j in rep]))
    mi = float(np.median([j["validacao"]["macro_iou"] or 0 for j in rei]))
    competente = all(j["validacao"]["competente"] for j in rep)
    return {"liberado": competente and mr >= mi - relevancia, "replay_competente_todas_sementes": competente,
            "mediana_iou_validacao_replay": mr, "mediana_iou_validacao_reinicio": mi, "limiar": limiar,
            "fonte": "somente validacao"}


def treinar_tudo(saida, protocolo, inv, particoes):
    tr, va, _ = particoes
    cfg = protocolo["config"] | {"modo": protocolo["modo"]}
    assinatura = ci.hash_json(protocolo)
    limiar = protocolo["congelado_da_validacao"]["limiar"]
    ctx = protocolo["contextos_em_ordem"]
    por_nome = {r["nome"]: r for r in tr}
    materializar_cache(inv, Path(protocolo["campanha"]) / "cache")
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    jobs = []
    for seed in cfg["sementes"]:
        anot = protocolo["anotacoes"][str(seed)]
        estat = normalizacao([por_nome[n] for n in anot[ctx[0]]])
        pasta = saida / f"etapa1_s{seed}"
        modelo, r = ajustar([por_nome[n] for n in anot[ctx[0]]], [], "uniforme", None, cfg["passos_por_etapa"],
                            va, estat, seed, 1, cfg, pasta, assinatura)
        etapa1 = {"braco": "etapa1_compartilhada", "seed": seed, "etapa": 1, "contexto": ctx[0], "pasta": str(pasta),
                  "treino": r, "pai": None, "anotacoes_acumuladas": len(anot[ctx[0]]), "buffer_replay_chips": 0,
                  "estat": estat}
        _validar(etapa1, modelo, va, estat, cfg["modo"], limiar)
        jobs.append(etapa1)
        for nome in ("reinicio", "continuacao", "continuacao_replay"):
            _linhagem(nome, seed, etapa1, ctx, anot, por_nome, va, estat, cfg, saida, assinatura, limiar, jobs)
        k = len(ctx)
        pasta = saida / f"reinicio_passos_acumulados_s{seed}_e{k}"
        todas = [por_nome[n] for c in ctx for n in anot[c]]
        modelo, r = ajustar(todas, [], "uniforme", None, k*cfg["passos_por_etapa"], va, estat, seed, k, cfg, pasta,
                            assinatura)
        job = {"braco": "reinicio_passos_acumulados", "seed": seed, "etapa": k, "contexto": ctx[-1],
               "pasta": str(pasta), "treino": r, "pai": None, "anotacoes_acumuladas": len(todas),
               "buffer_replay_chips": 0}
        _validar(job, modelo, va, estat, cfg["modo"], limiar)
        jobs.append(job)
    portao = portao_e2(jobs, cfg["sementes"], limiar, cfg["relevancia"])
    ci.gravar_json(saida / "portao_e2.json", portao)
    if portao["liberado"]:
        for seed in cfg["sementes"]:
            etapa1 = next(j for j in jobs if j["braco"] == "etapa1_compartilhada" and j["seed"] == seed)
            for politica in POLITICAS_E2:
                _linhagem(f"replay_{politica}", seed, etapa1, ctx, protocolo["anotacoes"][str(seed)], por_nome, va,
                          etapa1["estat"], cfg, saida, assinatura, limiar, jobs, selecao=politica, tr=tr,
                          divisoes=protocolo["divisoes"])
    concluido = {"assinatura": assinatura, "jobs": [{k: v for k, v in j.items() if k != "treino"} | {
        "hash_estado": j["treino"]["hash_estado"], "id_job": j["treino"]["id_job"],
        "passos": j["treino"]["passos_executados"], "segundos": j["treino"]["segundos"],
        "melhor_epoca": j["treino"]["melhor_epoca"]} for j in jobs], "portao_e2": portao,
        "codigo": {str(f): sha(f) for f in fontes()}}
    ci.gravar_json(saida / "treino_concluido.json", concluido)
    return concluido


def avaliar_tudo(saida, protocolo, inv, particoes):
    """So depois de todos os ajustes e do portao; nada aqui escolhe."""
    tr, _, te = particoes
    concluido = dc.ler(saida / "treino_concluido.json")
    assinatura = ci.hash_json(protocolo)
    if concluido["assinatura"] != assinatura:
        raise ValueError("treino concluido pertence a outro protocolo")
    for f, h in protocolo["codigo"].items():
        if sha(f) != h:
            raise ValueError(f"codigo mudou depois do pre-registro: {f}; nova versao necessaria")
    materializar_cache(inv, Path(protocolo["campanha"]) / "cache")
    torch.set_num_threads(protocolo["config"]["threads"])
    reservas = [r for r in tr if any(r["nome"] in d["retencao"] for d in protocolo["divisoes"].values())]
    alvos = reservas + te
    estats = {j["seed"]: j["estat"] for j in concluido["jobs"] if j["braco"] == "etapa1_compartilhada"}
    for j in concluido["jobs"]:
        f = Path(j["pasta"]) / "avaliacao.json"
        if f.exists():
            continue
        e = torch.load(Path(j["pasta"]) / "melhor.pt", weights_only=False, map_location="cpu")
        if ci.hash_completo_do_estado(e["modelo"]) != j["hash_estado"]:
            raise ValueError("checkpoint divergente na avaliacao")
        modelo = ci.UNetPequena(canais_entrada=4, base=protocolo["config"]["base"])
        modelo.load_state_dict(e["modelo"])
        m = dc.medir(modelo, alvos, estats[j["seed"]], protocolo["modo"], origem="retencao_e_avaliacao_desenvolvimento")
        ci.gravar_json(f, m)
        print(json.dumps({"avaliado": Path(j["pasta"]).name}), flush=True)
    return concluido


# ---------------------------------------------------------------- analise (pura)

def metricas_evento(e, limiar):
    m = e["limiares"][str(limiar)]
    c = m["contagens"]
    total = sum(c.values())
    return {"iou": m["iou"], "ap": e["ap_aproximada_histograma"], "recall": m["recall"], "precisao": m["precisao"],
            "fracao_predita_agua": (c["tp"]+c["fp"])/total if total else None,
            "tudo_fundo": e["positivos"] > 0 and c["tp"]+c["fp"] == 0, "prevalencia": e["prevalencia"]}


def decidir(difs, relevancia, minimo):
    if not difs:
        return {"estado": "not_testable", "n_eventos": 0}
    from scipy.stats import wilcoxon
    xs = np.array(list(difs.values()), dtype=float)
    p = float(wilcoxon(xs).pvalue) if np.any(xs) else 1.
    rng = np.random.default_rng(20261009)
    ic = np.quantile(rng.choice(xs, size=(5000, len(xs)), replace=True).mean(1), [.025, .975]).tolist()
    if len(xs) < minimo:
        estado = "INDECIDIVEL"
    elif p < .05 and ic[0] > relevancia:
        estado = "ganho_sustentado_no_desenvolvimento"
    elif p < .05 and ic[1] < -relevancia:
        estado = "perda_sustentada_no_desenvolvimento"
    elif ic[0] > -relevancia and ic[1] < relevancia:
        estado = "sem_diferenca_relevante"
    else:
        estado = "INDECIDIVEL"
    return {"diferencas_por_evento": difs, "n_eventos": len(xs), "media": float(xs.mean()),
            "mediana": float(np.median(xs)), "a_favor": f"{int((xs > 0).sum())}/{len(xs)}",
            "ic95_bootstrap_evento": ic, "p_wilcoxon": p, "relevancia": relevancia, "estado": estado,
            "poder": "insuficiente por construcao" if len(xs) < minimo else "minimo atendido"}


def esquecimento(serie, contextos):
    """Chaudhry et al. 2018 (F) e Lopez-Paz & Ranzato 2017 (BWT), por contexto antigo."""
    k = len(contextos)
    f, bwt = {}, {}
    for j, c in enumerate(contextos[:-1], 1):
        vals = [serie[e][c] for e in range(j, k) if serie.get(e, {}).get(c) is not None]
        final, aprendido = serie[k].get(c), serie[j].get(c)
        if vals and final is not None:
            f[c] = max(vals) - final
        if final is not None and aprendido is not None:
            bwt[c] = final - aprendido
    return {"esquecimento_por_contexto": f, "bwt_por_contexto": bwt,
            "esquecimento_medio": float(np.mean(list(f.values()))) if f else None,
            "bwt_medio": float(np.mean(list(bwt.values()))) if bwt else None}


def matriz(jobs, avaliacoes, limiar):
    """valores[braco][seed][etapa][evento] -> metricas; etapa 1 compartilhada entra em cada linhagem."""
    linhagens = sorted({j["braco"] for j in jobs if j["braco"] != "etapa1_compartilhada"})
    m = {}
    for j in jobs:
        alvo = linhagens if j["braco"] == "etapa1_compartilhada" else [j["braco"]]
        for b in alvo:
            if b == "reinicio_passos_acumulados" and j["etapa"] == 1:
                continue
            m.setdefault(b, {}).setdefault(j["seed"], {})[j["etapa"]] = {
                ev: metricas_evento(e, limiar) for ev, e in avaliacoes[j["pasta"]]["por_evento"].items()}
    return m


def media_sementes(m, braco, etapa, metrica):
    eventos = {}
    for seed, por_etapa in m[braco].items():
        for ev, v in por_etapa.get(etapa, {}).items():
            if v[metrica] is not None:
                eventos.setdefault(ev, []).append(v[metrica])
    return {ev: float(np.mean(v)) for ev, v in eventos.items()}


def analisar(protocolo, concluido, avaliacoes):
    cfg, limiar = protocolo["config"], protocolo["congelado_da_validacao"]["limiar"]
    ctx, teste = protocolo["contextos_em_ordem"], protocolo["avaliacao_desenvolvimento"]
    k = len(ctx)
    m = matriz(concluido["jobs"], avaliacoes, limiar)
    rel, minimo = cfg["relevancia"], cfg["minimo_eventos"]
    oito = ctx + teste

    def contraste(a, b, eventos, etapa=k, metrica="iou"):
        va, vb = media_sementes(m, a, etapa, metrica), media_sementes(m, b, etapa, metrica)
        return decidir({e: va[e]-vb[e] for e in eventos if e in va and e in vb}, rel, minimo)

    contrastes = {}
    for metrica in ("iou", "ap"):
        c = {"P1_replay_retem": contraste("continuacao_replay", "continuacao", ctx[:-1], metrica=metrica),
             "P2_acumulo_custo_marginal_igual": contraste("continuacao_replay", "reinicio", oito, metrica=metrica),
             "P3_alem_de_passos": contraste("continuacao_replay", "reinicio_passos_acumulados", oito, metrica=metrica),
             "P4_continuar_sem_replay": contraste("continuacao", "reinicio", oito, metrica=metrica),
             "P5_so_passos": contraste("reinicio_passos_acumulados", "reinicio", oito, metrica=metrica)}
        for p in POLITICAS_E2:
            if f"replay_{p}" in m:
                c[f"E2_{p}_menos_aleatoria"] = contraste(f"replay_{p}", "continuacao_replay", oito, metrica=metrica)
        contrastes[metrica] = c
    retencao = {}
    for b in ("reinicio", "continuacao", "continuacao_replay", *[f"replay_{p}" for p in POLITICAS_E2 if f"replay_{p}" in m]):
        retencao[b] = {}
        for metrica in ("iou", "ap"):
            serie = {e: media_sementes(m, b, e, metrica) for e in range(1, k+1)}
            retencao[b][metrica] = esquecimento(serie, ctx)
            retencao[b][metrica]["contexto_novo_na_sua_etapa"] = {c: serie[i].get(c) for i, c in enumerate(ctx, 1)}
            retencao[b][metrica]["avaliacao_por_etapa"] = {e: {t: serie[e].get(t) for t in teste}
                                                           for e in range(1, k+1)}
            por_seed = {}
            for seed in m[b]:
                s = {e: {ev: v[metrica] for ev, v in m[b][seed].get(e, {}).items()} for e in range(1, k+1)}
                por_seed[seed] = esquecimento(s, ctx)["esquecimento_medio"]
            retencao[b][metrica]["esquecimento_medio_por_semente"] = por_seed
    colapsos = [{"braco": b, "seed": s, "etapa": e, "evento": ev}
                for b, ss in m.items() for s, es in ss.items() for e, evs in es.items()
                for ev, v in evs.items() if v["tudo_fundo"]]
    custos = custos_por_braco(concluido["jobs"], k)
    return {"matriz": m, "contrastes": contrastes, "retencao": retencao, "colapsos_tudo_fundo": colapsos,
            "custos": custos, "limiar": limiar}


def custos_por_braco(jobs, k):
    """Passos marginais, acumulados na linhagem e contidos no modelo final."""
    por_pasta = {j["pasta"]: j for j in jobs}
    r = {}
    for j in jobs:
        if j["etapa"] != k or j["braco"] == "etapa1_compartilhada":
            continue
        contidos, atual = 0, j
        while atual is not None:
            contidos += atual["passos"]
            atual = por_pasta[atual["pai"]] if atual["pai"] else None
        linhagem = [x for x in jobs if x["seed"] == j["seed"] and (x["braco"] == j["braco"] or (
            x["braco"] == "etapa1_compartilhada" and j["braco"] != "reinicio_passos_acumulados"))]
        b = r.setdefault(j["braco"], {"sementes": {}})
        b["sementes"][j["seed"]] = {
            "anotacoes": j["anotacoes_acumuladas"], "buffer_replay_chips": j["buffer_replay_chips"],
            "passos_contidos_no_modelo_final": contidos,
            "passos_executados_na_linhagem": sum(x["passos"] for x in linhagem),
            "segundos_na_linhagem": sum(x["segundos"] for x in linhagem),
            "melhor_epoca_final": j["melhor_epoca"]}
    return r


def cartao_de_modelo(job, protocolo, avaliacao, limiar):
    ctx, teste = protocolo["contextos_em_ordem"], protocolo["avaliacao_desenvolvimento"]
    return {"modelo": f"UNetPequena base {protocolo['config']['base']}, entrada pos-evento duplicada em 4 canais",
            "versao_protocolo": VERSAO, "pesos": str(Path(job["pasta"]) / "melhor.pt"),
            "hash_completo_do_estado": job["hash_estado"], "braco": job["braco"], "semente": job["seed"],
            "etapa": job["etapa"], "contextos_vistos": ctx[:job["etapa"]],
            "objetivo": "CE nao ponderada; agua observada fora da mascara JRC; desconhecido ignorado",
            "dados": "Sen1Floods11 rotulado a mao; eventos do papel treino do pacote ativo; ja examinados",
            "otimizador": OTIMIZADOR, "checkpoint": "menor CE de validacao na etapa",
            "limiar": limiar, "limiar_origem": "validacao de 07/10; nao calibra probabilidade",
            "validacao": job["validacao"],
            "desenvolvimento_por_evento": {ev: metricas_evento(e, limiar)
                                           for ev, e in avaliacao["por_evento"].items() if ev in ctx+teste},
            "uso_pretendido": "pesquisa de aprendizado continuo em desenvolvimento",
            "fora_do_uso": ["servico", "svc02", "escore fisico", "verdade observacional brasileira",
                            "probabilidade territorial", "previsao futura"],
            "promovido": False,
            "limites": ["dez eventos conhecidos no total", "reserva de retencao e do mesmo evento do ajuste",
                        "antes e mediana anual; entrada usada e so pos-evento", "uma ordem de contextos"]}


def consolidar(saida, protocolo):
    concluido = dc.ler(saida / "treino_concluido.json")
    avaliacoes = {j["pasta"]: dc.ler(Path(j["pasta"]) / "avaliacao.json") for j in concluido["jobs"]}
    a = analisar(protocolo, concluido, avaliacoes)
    k = len(protocolo["contextos_em_ordem"])
    cartoes = saida / "cartoes"
    cartoes.mkdir(exist_ok=True)
    for j in concluido["jobs"]:
        if j["etapa"] == k:
            ci.gravar_json(cartoes / f"{Path(j['pasta']).name}.json",
                           cartao_de_modelo(j, protocolo, avaliacoes[j["pasta"]], a["limiar"]))
    pesos = {Path(j["pasta"]).name: {"hash_completo_do_estado": j["hash_estado"],
                                     "arquivo_sha256": sha(Path(j["pasta"]) / "melhor.pt")} for j in concluido["jobs"]}
    r = {"versao": VERSAO, "assinatura": concluido["assinatura"], "protocolo_sha256": sha(saida / "protocolo.json"),
         "portao_e2": concluido["portao_e2"], **{k2: v for k2, v in a.items() if k2 != "matriz"},
         "pesos": pesos, "promovido": False, "altera_escore_suscetibilidade": False, "altera_svc02": False,
         "politica_aprendida": False,
         "nao_e": "nao confirma transferencia, nao promove pesos, nao prova autonomia superior nem previsao futura"}
    ci.gravar_json(saida / "matriz.json", {"matriz": a["matriz"], "limiar": a["limiar"]})
    ci.gravar_json(saida / "resultado.json", r)
    return r


def executar(saida=SAIDA, campanha=CAMPANHA, diagnostico=DIAGNOSTICO, config=None, pacote=PACOTE, somente_preparar=False):
    saida = Path(saida).resolve()
    protocolo, inv, particoes = preparar(campanha, diagnostico, config, pacote)
    saida.mkdir(parents=True, exist_ok=True)
    dc.imutavel(saida / "protocolo.json", protocolo)
    codigo = saida / "codigo"
    codigo.mkdir(exist_ok=True)
    for f, h in protocolo["codigo"].items():
        destino = codigo / Path(f).name
        if not destino.exists():
            shutil.copyfile(f, destino)
        if sha(destino) != h:
            raise ValueError("snapshot de codigo divergente; nova versao necessaria")
    if somente_preparar:
        return {"estado": "preparado", "assinatura": ci.hash_json(protocolo), "treinos_executados": 0}
    if not (saida / "treino_concluido.json").exists():
        treinar_tudo(saida, protocolo, inv, particoes)
    avaliar_tudo(saida, protocolo, inv, particoes)
    return consolidar(saida, protocolo)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--saida", type=Path, default=SAIDA)
    ap.add_argument("--preparar", action="store_true", help="grava protocolo e divisoes; nao treina")
    ap.add_argument("--esperar-min", type=float, default=0.)
    a = ap.parse_args()
    if a.preparar:
        print(json.dumps(executar(a.saida, somente_preparar=True), ensure_ascii=False))
        return 0
    import executar_serializado as serial
    energia = serial.energia()
    if not energia["na_tomada"]:
        print(json.dumps({"estado": "aguarda_tomada", "energia": energia}))
        return 3
    dono = serial.tomar([sys.executable, str(Path(__file__)), "--saida", str(a.saida)], a.esperar_min, False)
    if dono is None:
        print("A maquina ja tem trabalho pesado; nenhuma etapa foi iniciada.")
        return 2
    try:
        r = executar(a.saida)
    finally:
        serial.liberar(dono)
    print(json.dumps({k: r[k] for k in ("versao", "assinatura", "portao_e2")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
