"""Compara entrada temporal por evento e aquisicao ativa sem olhar o oracle.

Esta e uma campanha de desenvolvimento: os rotulos Sen1Floods11 ja foram
examinados no projeto. Reserva por evento nao transforma dados conhecidos
em teste prospectivo virgem, nem autoriza promocao automatica.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import rasterio
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/sar"))
sys.path.insert(0, str(REPO / "scripts/neural"))
from baseline_otsu import contagens, metricas
from dados_pares_geograficos import (entrada, inventariar, normalizacao, particao,
                                    rotulo, sha, materializar_cache)
from experimento_pre_pos import gravar_json, gravar_peso, hash_json
from modelo_unet import UNetPequena, contar_parametros
from proveniencia_de_peso import hash_completo_do_estado, semear


def alvo(r, sensor, janela=None):
    y, v, npv = rotulo(r["caminhos"]["rotulo"], r["caminhos"]["jrc"])
    if janela is not None:
        a, b, s = janela
        y, v, npv = y[a:a+s, b:b+s], v[a:a+s, b:b+s], npv[a:a+s, b:b+s]
    z = y.copy()
    z[~(npv & sensor)] = -1
    return z, v & sensor, npv & sensor


@torch.no_grad()
def avaliar(modelo, linhas, estat, modo):
    modelo.eval()
    eventos, soma = {}, dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
    loss, validos, desconhecidos, total, chips = 0., 0, 0, 0, []
    for r in linhas:
        x, sensor = entrada(r, estat, modo)
        y, v, npv = alvo(r, sensor)
        t = torch.from_numpy(x).unsqueeze(0)
        logits = modelo(t)
        loss += float(nn.functional.cross_entropy(logits, torch.from_numpy(y).unsqueeze(0),
                                                  ignore_index=-1, reduction="sum"))
        validos += int(npv.sum())
        desconhecidos += int((~v).sum())
        total += v.size
        p = logits.argmax(1)[0].numpy() == 1
        verdade = y == 1
        c = contagens(p, verdade, npv)
        e = eventos.setdefault(r["evento"], dict.fromkeys(soma, 0))
        for k in soma:
            soma[k] += c[k]
            e[k] += c[k]
        chips.append({"nome": r["nome"], "evento": r["evento"], "contagens": c, **metricas(c)})
    if not validos:
        raise ValueError("avaliacao sem pixels validos fora de agua permanente")
    por_evento = {e: {"contagens": c, **metricas(c)} for e, c in eventos.items()}
    ious = [c["iou"] for c in por_evento.values() if c["iou"] is not None]
    return {"micro": metricas(soma), "contagens": soma, "por_evento": por_evento,
            "macro_evento_iou": float(np.mean(ious)) if ious else None,
            "eventos_iou_indefinida": len(eventos) - len(ious),
            "perda": loss / validos, "pixels_validos_sem_permanente": validos,
            "pixels_desconhecidos_ou_sensor_invalido": desconhecidos,
            "pixels_totais": total, "por_chip": chips}


def treinar(linhas, validacao, estat, modo, seed, cfg, pasta, assinatura):
    pasta.mkdir(parents=True, exist_ok=True)
    id_job = hash_json({"assinatura": assinatura, "linhas": [r["nome"] for r in linhas],
                        "modo": modo, "seed": seed, "estat": estat, "config": cfg})
    pronto = pasta / "treino.json"
    peso = pasta / "melhor.pt"
    semear(seed)
    modelo = UNetPequena(canais_entrada=4, base=cfg["base"])
    if pronto.exists():
        r = json.loads(pronto.read_text(encoding="utf-8"))
        e = torch.load(peso, weights_only=False, map_location="cpu")
        if r["id_job"] != id_job or e["id_job"] != id_job or hash_completo_do_estado(e["modelo"]) != r["hash_estado"]:
            raise ValueError("resultado de treino divergente")
        modelo.load_state_dict(e["modelo"])
        return modelo, r
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"])
    melhor, historico, inicio, melhor_epoca = float("inf"), [], 0, None
    retomada = pasta / "retomada.pt"
    if retomada.exists():
        e = torch.load(retomada, weights_only=False, map_location="cpu")
        if e["id_job"] != id_job:
            raise ValueError("retomada fora do protocolo")
        modelo.load_state_dict(e["modelo"])
        opt.load_state_dict(e["opt"])
        torch.set_rng_state(e["rng"])
        melhor, historico, inicio, melhor_epoca = e["melhor"], e["historico"], e["epoca"], e["melhor_epoca"]
    for epoca in range(inicio, cfg["epocas"]):
        modelo.train()
        rng = np.random.default_rng(seed * 10000 + epoca)
        t0, perdas = time.monotonic(), []
        for passo in range(cfg["passos_por_epoca"]):
            xs, ys = [], []
            for _ in range(cfg["batch"]):
                for tentativa in range(100):
                    r = linhas[int(rng.integers(len(linhas)))]
                    h, w = r["grade"]["forma"]
                    s = cfg["recorte"]
                    if s > min(h, w):
                        raise ValueError("recorte maior que o chip")
                    a, b = int(rng.integers(h - s + 1)), int(rng.integers(w - s + 1))
                    janela = (a, b, s)
                    xx, sensor = entrada(r, estat, modo, janela)
                    yy, _, _ = alvo(r, sensor, janela)
                    if (yy != -1).any():
                        break
                else:
                    raise ValueError("sem recortes validos; nao omitir passos silenciosamente")
                if rng.random() < .5:
                    xx, yy = xx[:, :, ::-1], yy[:, ::-1]
                if rng.random() < .5:
                    xx, yy = xx[:, ::-1], yy[::-1]
                xs.append(torch.from_numpy(xx.copy()))
                ys.append(torch.from_numpy(yy.copy()))
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(modelo(torch.stack(xs)), torch.stack(ys), ignore_index=-1)
            if not torch.isfinite(loss):
                raise ValueError("perda nao finita")
            loss.backward()
            opt.step()
            perdas.append(float(loss.detach()))
        va = avaliar(modelo, validacao, estat, modo)
        if va["perda"] < melhor:
            melhor, melhor_epoca = va["perda"], epoca + 1
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            gravar_peso(peso, {"modelo": estado, "estat": estat, "modo": modo,
                              "base": cfg["base"], "id_job": id_job,
                              "assinatura": assinatura, "hash_estado": hash_completo_do_estado(estado)})
        historico.append({"epoca": epoca + 1, "perda_treino": float(np.mean(perdas)),
                          "perda_validacao": va["perda"], "iou_validacao": va["macro_evento_iou"],
                          "passos": cfg["passos_por_epoca"], "segundos": time.monotonic() - t0})
        gravar_peso(retomada, {"modelo": modelo.state_dict(), "opt": opt.state_dict(),
                              "rng": torch.get_rng_state(), "id_job": id_job, "epoca": epoca + 1,
                              "historico": historico, "melhor": melhor, "melhor_epoca": melhor_epoca})
        print(json.dumps({"job": pasta.name, "modo": modo, "seed": seed, **historico[-1]}), flush=True)
    e = torch.load(peso, weights_only=False, map_location="cpu")
    modelo.load_state_dict(e["modelo"])
    r = {"id_job": id_job, "seed": seed, "modo": modo, "melhor_epoca": melhor_epoca,
         "hash_estado": e["hash_estado"], "parametros": contar_parametros(modelo),
         "historico": historico, "chips_admitidos": len(linhas), "eventos_treino": sorted({l["evento"] for l in linhas})}
    gravar_json(pronto, r)
    return modelo, r


def escolher(candidatos, politica, quantidade, seed, pontuacoes=None):
    """Recebe somente identidade/evento/score. Nunca recebe rotulo ou erro real."""
    if quantidade <= 0 or quantidade > len(candidatos):
        raise ValueError("orcamento fora do pool")
    fila = sorted(candidatos, key=lambda r: r["nome"])
    if politica == "aleatoria":
        np.random.default_rng(seed).shuffle(fila)
    elif politica == "ativa":
        if pontuacoes is None or set(pontuacoes) != {r["nome"] for r in fila} or not all(np.isfinite(v) for v in pontuacoes.values()):
            raise ValueError("scores incompletos ou nao finitos")
        fila.sort(key=lambda r: (-pontuacoes[r["nome"]], r["nome"]))
    elif politica != "fixa":
        raise ValueError("politica desconhecida")
    if politica != "ativa":
        return [r["nome"] for r in fila[:quantidade]]
    # Diversidade por evento: uma escolha de cada evento por rodada da fila.
    grupos = {}
    for r in fila:
        grupos.setdefault(r["evento"], []).append(r)
    escolhidos = []
    while len(escolhidos) < quantidade:
        for grupo in grupos.values():
            if grupo and len(escolhidos) < quantidade:
                escolhidos.append(grupo.pop(0)["nome"])
    return escolhidos


@torch.no_grad()
def pontuar(modelo, candidatos, estat):
    """Entropia predita com pixels radiometricos validos; nao le Label/JRC."""
    modelo.eval()
    scores = {}
    for r in candidatos:
        x, valido = entrada(r, estat, "pre_pos")
        p = modelo(torch.from_numpy(x).unsqueeze(0)).softmax(1)[0].numpy()
        entropia = -(p * np.log(np.clip(p, 1e-8, 1))).sum(0)
        if not valido.any():
            raise ValueError("candidato sem observacao valida")
        scores[r["nome"]] = float(entropia[valido].mean())
    return scores


def resumo_pareado(difs, relevancia=.025):
    """A unidade e evento. Nao reamostrar pixels ou chamar sementes de eventos."""
    from scipy.stats import wilcoxon
    if not difs:
        return {"estado": "not_testable", "n_eventos": 0}
    xs = np.array(list(difs.values()))
    p = float(wilcoxon(xs).pvalue) if np.any(xs) else 1.
    rng = np.random.default_rng(20261006)
    medias = rng.choice(xs, size=(5000, len(xs)), replace=True).mean(1)
    ic = np.quantile(medias, [.025, .975]).tolist()
    return {"diferencas_por_evento": difs, "n_eventos": len(xs), "media": float(xs.mean()),
            "ic95_bootstrap_evento": ic, "p_wilcoxon": p, "relevancia": relevancia,
            "estado": "ganho_sustentado_no_desenvolvimento" if len(xs) >= 6 and p < .05 and ic[0] > relevancia else "INDECIDIVEL",
            "limite": "bootstrap de poucos eventos e instavel; regime de desenvolvimento, nao teste prospectivo"}


def executar(raiz, saida, cfg):
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    saida.mkdir(parents=True, exist_ok=True)
    inv = inventariar(raiz)
    if inv["recusados"]:
        gravar_json(saida / "quarentena.json", inv)
        raise ValueError("pares recusados; resolver contrato antes de treinar")
    materializar_cache(inv, saida / "cache")
    eventos = sorted(inv["eventos"], key=lambda e: hashlib.sha256(e.encode()).hexdigest())
    if len(eventos) < 6:
        raise ValueError("menos de seis eventos para reserva/validacao/pool")
    protocolo = {"config": cfg, "inventario": hash_json(inv),
                 "codigo": {p.name: sha(p) for p in (Path(__file__), REPO / "scripts/sar/dados_pares_geograficos.py",
                    REPO / "scripts/sar/modelo_unet.py", REPO / "scripts/sar/baseline_otsu.py",
                    REPO / "scripts/sar/experimento_pre_pos.py", REPO / "scripts/neural/proveniencia_de_peso.py",
                    REPO / "scripts/sar/auditar_pares_geograficos.py", REPO / "scripts/sar/adquirir_pares_geograficos.py")},
                 "ordem_eventos": eventos, "primaria": "IoU por evento fora de agua permanente",
                 "regime": inv["regime"], "promovido": False,
                 "autonomia": "replay offline de anotacao; entropia + diversidade vs aleatoria e fixa",
                 "custo": "numero de chips anotados; tempo de inferencia/treino medido separadamente",
                 "torch": torch.__version__, "numpy": np.__version__,
                 "rasterio": rasterio.__version__, "python": sys.version,
                 "relevancia": .025}
    assinatura = hash_json(protocolo)
    f = saida / "protocolo.json"
    if f.exists() and json.loads(f.read_text(encoding="utf-8")) != protocolo:
        raise ValueError("protocolo imutavel; usar nova pasta")
    gravar_json(f, protocolo)
    gravar_json(saida / "inventario.json", inv)
    codigo = saida / "codigo"
    codigo.mkdir(exist_ok=True)
    for nome, esperado in protocolo["codigo"].items():
        origem = REPO / "scripts" / ("neural" if nome == "proveniencia_de_peso.py" else "sar") / nome
        destino = codigo / nome
        if not destino.exists():
            shutil.copyfile(origem, destino)
        if sha(destino) != esperado:
            raise ValueError("snapshot de codigo divergente")
    independencia = []
    for i, evento in enumerate(eventos):
        va_ev = [eventos[(i+1) % len(eventos)], eventos[(i+2) % len(eventos)]]
        tr, va, te = particao(inv["pares"], [evento], va_ev)
        estat = normalizacao(tr)
        for modo in ("pos", "pre_pos", "pos_delta"):
            pasta = saida / f"evento_{i:02d}_{modo}"
            modelo, r = treinar(tr, va, estat, modo, cfg["sementes"][0], cfg, pasta, assinatura)
            resultado = pasta / "avaliacao.json"
            if resultado.exists():
                medido = json.loads(resultado.read_text(encoding="utf-8"))
                if medido["id_job"] != r["id_job"]:
                    raise ValueError("avaliacao fora do job")
            else:
                medido = {**r, "teste": avaliar(modelo, te, estat, modo), "evento_teste": evento,
                          "eventos_validacao": va_ev, "assinatura": assinatura}
                gravar_json(resultado, medido)
            independencia.append(medido)
            print(json.dumps({"concluido": pasta.name, "iou": medido["teste"]["micro"]["iou"]}), flush=True)
    difs = {}
    for e in eventos:
        rs = {r["modo"]: r for r in independencia if r["evento_teste"] == e}
        a, b = rs["pre_pos"]["teste"]["micro"]["iou"], rs["pos"]["teste"]["micro"]["iou"]
        if a is not None and b is not None:
            difs[e] = a - b
    gravar_json(saida / "independencia.json", {"assinatura": assinatura, "resultados": independencia,
                                               "analise": resumo_pareado(difs), "promovido": False})
    # Reserva distinta e fixa para replay. Nunca usada na escolha ou no treino desta politica.
    tr, va, te = particao(inv["pares"], eventos[:3], eventos[3:5])
    autonomia = []
    for seed in cfg["sementes"]:
        nomes_base = escolher(tr, "aleatoria", cfg["lote_anotacao"], seed)
        base = [r for r in tr if r["nome"] in nomes_base]
        estat = normalizacao(base)
        modelo_base, rbase = treinar(base, va, estat, "pre_pos", seed, cfg,
                                    saida / f"base_s{seed}", assinatura)
        # So medido apos concluir TODAS as escolhas da semente.
        pendentes = []
        for politica in ("ativa", "aleatoria", "fixa"):
            escolhidos, modelo, historico_aq = list(nomes_base), modelo_base, []
            for rodada in range(1, cfg["rodadas"] + 1):
                candidatos = [r for r in tr if r["nome"] not in escolhidos]
                t0 = time.monotonic()
                sem_rotulos = [{"nome": r["nome"], "evento": r["evento"],
                                "caminhos": {k: r["caminhos"][k] for k in ("antes", "depois")}}
                               for r in candidatos]
                scores = pontuar(modelo, sem_rotulos, estat) if politica == "ativa" else None
                vistos = [{"nome": r["nome"], "evento": r["evento"]} for r in candidatos]
                novos = escolher(vistos, politica, cfg["lote_anotacao"], seed*100 + rodada, scores)
                historico_aq.append({"rodada": rodada, "nomes": novos, "scores": scores,
                                    "segundos_selecao": time.monotonic()-t0,
                                    "oracle_consultado_na_selecao": False})
                escolhidos += novos
                linhas = [r for r in tr if r["nome"] in escolhidos]
                pasta = saida / f"autonomia_{politica}_s{seed}_r{rodada}"
                modelo, r = treinar(linhas, va, estat, "pre_pos", seed, cfg, pasta, assinatura)
                gravar_json(pasta / "aquisicao.json", {"base": nomes_base, "historico": historico_aq,
                                                        "anotacoes": len(escolhidos), "assinatura": assinatura})
            pendentes.append((politica, modelo, r, list(escolhidos), historico_aq))
        base_teste = avaliar(modelo_base, te, estat, "pre_pos")
        for politica, modelo, r, escolhidos, historico_aq in pendentes:
            resultado = {**r, "politica": politica, "anotacoes": len(escolhidos),
                         "nomes_base": nomes_base, "escolhidos": escolhidos,
                         "historico_aquisicao": historico_aq, "teste": avaliar(modelo, te, estat, "pre_pos"),
                         "teste_base": base_teste, "assinatura": assinatura,
                         "promovido": False, "politica_aprendida": False,
                         "aquisicao": "replay de anotacoes preexistentes, nao coleta prospectiva"}
            autonomia.append(resultado)
            gravar_json(saida / f"autonomia_{politica}_s{seed}.json", resultado)
    comparacoes = {}
    for ref in ("aleatoria", "fixa"):
        ganhos = {}
        for e in eventos[:3]:
            ds = []
            for seed in cfg["sementes"]:
                rs = {r["politica"]: r for r in autonomia if r["seed"] == seed}
                a = rs["ativa"]["teste"]["por_evento"][e]["iou"]
                b = rs[ref]["teste"]["por_evento"][e]["iou"]
                if a is not None and b is not None:
                    ds.append(a-b)
            if ds:
                ganhos[e] = float(np.mean(ds))
        comparacoes[ref] = resumo_pareado(ganhos)
    card = {"assinatura": assinatura, "independencia": resumo_pareado(difs),
            "autonomia": comparacoes, "resultados_autonomia": autonomia,
            "eventos_teste_autonomia": eventos[:3], "promovido": False,
            "altera_escore_suscetibilidade": False, "regime": inv["regime"],
            "fronteira": "o teste desta politica e independente de seus pesos/selecoes, mas nao virgem do desenvolvimento global"}
    gravar_json(saida / "card.json", card)
    return card


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raiz", type=Path, default=REPO / "local_runs/sar-independencia/dados")
    p.add_argument("--saida", type=Path, required=True)
    p.add_argument("--epocas", type=int, default=8)
    p.add_argument("--passos-por-epoca", type=int, default=24)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--base", type=int, default=4)
    p.add_argument("--recorte", type=int, default=64)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--sementes", type=int, nargs="+", default=[7, 11, 23])
    p.add_argument("--rodadas", type=int, default=2)
    p.add_argument("--lote-anotacao", type=int, default=24)
    a = p.parse_args()
    cfg = {k: getattr(a, k) for k in ("epocas", "passos_por_epoca", "batch", "base", "recorte",
                                      "threads", "lr", "sementes", "rodadas", "lote_anotacao")}
    if min(cfg[k] for k in ("epocas", "passos_por_epoca", "batch", "base", "recorte", "threads", "rodadas", "lote_anotacao")) < 1 or a.recorte % 8 or a.lr <= 0 or len(set(a.sementes)) != len(a.sementes):
        p.error("configuracao invalida")
    from auditar_pares_geograficos import auditar
    if not auditar(a.raiz.resolve(), a.saida / "auditoria.json")["admissivel"]:
        raise SystemExit("contrato de dados recusado; treino nao iniciado")
    executar(a.raiz, a.saida, cfg)
