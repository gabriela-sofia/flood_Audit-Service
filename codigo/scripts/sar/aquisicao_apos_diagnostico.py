"""Replay de aquisicao liberado somente por competencia na validacao.

Separa entropia, diversidade e sua combinacao. Mesmo numero de anotacoes e
passos entre politicas. Reinicia os pesos: nao e aprendizado continuo.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import diagnostico_competencia as dc
import campanha_independencia_autonomia as ci
from dados_pares_geograficos import materializar_cache, particao, sha

POLITICAS = ("aleatoria", "fixa", "entropia", "diversidade", "combinada")


def escolher(itens, politica, quantidade, seed, scores=None):
    """A interface da escolha nao aceita caminho de rotulo nem erro observado."""
    if any(set(r) != {"nome", "evento"} for r in itens):
        raise ValueError("seletor recebe apenas identidade e evento")
    if politica not in POLITICAS:
        raise ValueError("politica desconhecida")
    if politica in ("aleatoria", "fixa"):
        return ci.escolher(itens, politica, quantidade, seed)
    if quantidade < 1 or quantidade > len(itens) or len({r["nome"] for r in itens}) != len(itens):
        raise ValueError("orcamento ou identidade invalida")
    fila = sorted(itens, key=lambda r: r["nome"])
    if politica in ("entropia", "combinada"):
        if scores is None or set(scores) != {r["nome"] for r in fila} or not all(np.isfinite(v) for v in scores.values()):
            raise ValueError("scores incompletos")
        fila.sort(key=lambda r: (-scores[r["nome"]], r["nome"]))
        if politica == "entropia":
            return [r["nome"] for r in fila[:quantidade]]
    else:
        np.random.default_rng(seed).shuffle(fila)
    grupos = {}
    for r in fila:
        grupos.setdefault(r["evento"], []).append(r["nome"])
    escolhidos = []
    while len(escolhidos) < quantidade:
        for grupo in grupos.values():
            if grupo and len(escolhidos) < quantidade:
                escolhidos.append(grupo.pop(0))
    return escolhidos


@torch.no_grad()
def pontuar(modelo, candidatos, estat, modo):
    modelo.eval()
    scores = {}
    for r in candidatos:
        if set(r["caminhos"]) != {"antes", "depois"}:
            raise ValueError("pontuador nao recebe oracle")
        x, sensor = dc.entrada(r, estat, modo)
        p = modelo(torch.from_numpy(x).unsqueeze(0)).softmax(1)[0].numpy()
        if not sensor.any():
            raise ValueError("candidato sem observacao valida")
        scores[r["nome"]] = float((-(p*np.log(np.clip(p, 1e-8, 1))).sum(0))[sensor].mean())
    return scores


def conferir(diagnostico):
    card = dc.ler(Path(diagnostico) / "card.json")
    if card["assinatura"] != ci.hash_json(card["protocolo"]):
        raise ValueError("assinatura do diagnostico divergente")
    for arquivo, h in card["protocolo"]["codigo"].items():
        if sha(arquivo) != h:
            raise ValueError("codigo do diagnostico mudou; nova campanha necessaria")
    decisao = dc.selecionar(card["resultados"], card["protocolo"]["config"]["sementes"])
    if decisao != card["aquisicao"]:
        raise ValueError("decisao de competencia divergente")
    return card


def executar(diagnostico, saida):
    diagnostico, saida = Path(diagnostico).resolve(), Path(saida).resolve()
    card = conferir(diagnostico)
    saida.mkdir(parents=True, exist_ok=True)
    if not card["aquisicao"]["liberada"]:
        r = {"estado": "bloqueada_por_competencia", "diagnostico": card["assinatura"],
             "treinos_executados": 0, "rotulos_adquiridos": 0, "teste_avaliado": False,
             "promovido": False, "motivo": "nenhum braco passou o piso em todas as sementes/eventos de validacao"}
        dc.imutavel(saida / "card.json", r)
        return r
    p = card["protocolo"]
    cfg, escolha = p["config"], card["aquisicao"]["escolhido"]
    campanha = Path(p["campanha"])
    inv = dc.ler(campanha / "inventario.json")
    if ci.hash_json(inv) != p["inventario"]:
        raise ValueError("inventario mudou")
    original = dc.ler(campanha / "protocolo.json")
    eventos = original["ordem_eventos"]
    tr, va, te = particao(inv["pares"], eventos[:3], eventos[3:5])
    braco = dc.BRACOS[escolha["braco"]]
    protocolo = {"diagnostico": card["assinatura"], "hash_card_diagnostico": sha(diagnostico / "card.json"),
                 "codigo_aquisicao": sha(Path(__file__)), "politicas": list(POLITICAS),
                 "config": cfg, "escolha_validacao": escolha, "teste": eventos[:3],
                 "orcamento": "mesmas anotacoes e passos; custo de selecao registrado a parte",
                 "regime": "replay em dados conhecidos; nao confirmacao externa",
                 "continuidade": "reinicio; nao mede retencao", "promovido": False}
    dc.imutavel(saida / "protocolo.json", protocolo)
    assinatura = ci.hash_json(protocolo)
    materializar_cache(inv, campanha / "cache")
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    resultados = []
    for seed in cfg["sementes"]:
        base = next(r for r in card["resultados"] if r["seed"] == seed and r["braco_nome"] == escolha["braco"])
        estat, pesos, nomes_base = base["estat"], base["pesos_classe"], base["base"]
        pasta_base = diagnostico / f"{escolha['braco']}_s{seed}"
        estado = torch.load(pasta_base / "melhor.pt", weights_only=False, map_location="cpu")
        if ci.hash_completo_do_estado(estado["modelo"]) != base["hash_estado"]:
            raise ValueError("peso base divergente")
        pendentes = []
        for politica in POLITICAS:
            modelo = ci.UNetPequena(canais_entrada=4, base=cfg["base"])
            modelo.load_state_dict(estado["modelo"])
            escolhidos, historico, medido = list(nomes_base), [], base["validacao"]
            for rodada in range(1, cfg["rodadas"]+1):
                piso = dc.resumo_limiar(medido, escolha["limiar"])
                if not piso["competente"]:
                    r = {"estado": "interrompida_por_competencia", "assinatura": assinatura,
                         "politica": politica, "seed": seed, "rodada": rodada,
                         "motivos": piso["motivos"], "historico": historico,
                         "teste_avaliado": False, "promovido": False,
                         "motivo": "falha preservada; escolhas e pesos parciais continuam nas pastas"}
                    ci.gravar_json(saida / "card.json", r)
                    return r
                candidatos = [r for r in tr if r["nome"] not in escolhidos]
                sem_rotulos = [{"nome": r["nome"], "evento": r["evento"],
                                "caminhos": {k: r["caminhos"][k] for k in ("antes", "depois")}} for r in candidatos]
                t0 = time.monotonic()
                scores = pontuar(modelo, sem_rotulos, estat, cfg["modo"]) if politica in ("entropia", "combinada") else None
                novos = escolher([{"nome": r["nome"], "evento": r["evento"]} for r in candidatos],
                                  politica, cfg["lote_anotacao"], seed*100+rodada, scores)
                h = {"rodada": rodada, "novos": novos, "scores": scores,
                     "segundos_selecao": time.monotonic()-t0, "oracle_consultado_na_selecao": False}
                escolhidos += novos
                historico.append(h)
                linhas = [r for r in tr if r["nome"] in escolhidos]
                pasta = saida / f"{politica}_s{seed}_r{rodada}"
                dc.imutavel(pasta / "selecao.json", {"escolhidos": escolhidos, "novos": novos,
                                                     "assinatura": assinatura})
                modelo, treino = dc.treinar(linhas, va, estat, seed, cfg, braco, pesos, pasta, assinatura)
                medido = dc.medir(modelo, va, estat, cfg["modo"])
                ci.gravar_json(pasta / "validacao.json", medido)
            pendentes.append((politica, modelo, treino, list(escolhidos), historico, medido))
        # O teste so e inferido depois de todas as escolhas da semente.
        for politica, modelo, treino, escolhidos, historico, medido in pendentes:
            teste = dc.medir(modelo, te, estat, cfg["modo"], origem="teste_desenvolvimento")
            r = {"politica": politica, "seed": seed, "treino": treino, "escolhidos": escolhidos,
                 "historico": historico, "validacao": medido, "teste": teste,
                 "competencia_final": dc.resumo_limiar(medido, escolha["limiar"]),
                 "anotacoes": len(escolhidos), "assinatura": assinatura}
            resultados.append(r)
            ci.gravar_json(saida / f"{politica}_s{seed}.json", r)
    comparacoes = {}
    for politica in POLITICAS[1:]:
        difs = {}
        for evento in eventos[:3]:
            ds = []
            for seed in cfg["sementes"]:
                rs = {r["politica"]: r for r in resultados if r["seed"] == seed}
                t = str(escolha["limiar"])
                a = rs[politica]["teste"]["por_evento"][evento]["limiares"][t]["iou"]
                b = rs["aleatoria"]["teste"]["por_evento"][evento]["limiares"][t]["iou"]
                if a is not None and b is not None:
                    ds.append(a-b)
            if ds:
                difs[evento] = float(np.mean(ds))
        comparacoes[politica] = ci.resumo_pareado(difs)
    r = {"estado": "executada", "assinatura": assinatura, "protocolo": protocolo,
         "resultados": resultados, "comparacoes_aleatoria": comparacoes, "promovido": False,
         "altera_escore_suscetibilidade": False, "politica_aprendida": False,
         "limite": "tres eventos conhecidos; sementes nao ampliam numero de eventos; nao prova autonomia geral"}
    ci.gravar_json(saida / "card.json", r)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("diagnostico", type=Path)
    ap.add_argument("--saida", type=Path, required=True)
    a = ap.parse_args()
    r = executar(a.diagnostico, a.saida)
    print(json.dumps({k: v for k, v in r.items() if k not in ("resultados", "protocolo")}, ensure_ascii=False))
