"""Nucleo de treino SAR sobre o adapter Sen1/GEOID, comum as campanhas de 11/10/2026.

Uma etapa de ajuste, retomavel por epoca, com:
  - entrada e alvo do adapter (fontes_sar), normalizacao fixa sem dado;
  - memoria de replay com proporcao rho no lote;
  - distilacao LwF (Li & Hoiem 2016) legada (KL em todos os pixels) ou mascarada
    (KL so no suporte sensor-visivel, inclusive pixel sem rotulo; nunca no NoData);
  - inicializacao herdada, aleatoria ou encoder pre-treinado (opcionalmente congelado);
  - checkpoint pela CE de um grupo de validacao declarado, ou ultima epoca;
  - contabilidade de custo: passos, recortes novos/memoria, pixels supervisionados,
    pixels distilados, passes do professor, bytes de memoria, segundos.
A medida grava histogramas de 1000 faixas por evento e grupo; calibracao e limiar
sao decididos depois, em grupo proprio (calibracao_decisao).
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import campanha_independencia_autonomia as ci
import fontes_sar as fs
from continuidade_v2 import indices_antigos, novo_modelo, carregar_encoder, congelar_encoder, modo_treino

BYTES_CHIP_POS = 2*512*512*4 + 512*512
BINS = 1000


def recortar(linhas, modo, rng, tamanho):
    """Mesmo esquema do v2_1: 100 tentativas aleatorias, depois ancora em pixel rotulado."""
    for _ in range(100):
        r = linhas[int(rng.integers(len(linhas)))]
        h, w = r["grade"]["forma"]
        a, b = int(rng.integers(h-tamanho+1)), int(rng.integers(w-tamanho+1))
        y = fs.alvo(r, modo, (a, b, tamanho))
        if (y != -1).any():
            return _finalizar(r, modo, rng, a, b, tamanho, y, False)
    validos = [(r, c) for r in linhas if len(c := np.argwhere(fs.alvo(r, modo) != -1))]
    if not validos:
        raise ValueError("nenhum pixel rotulado nos chips")
    r, coords = validos[int(rng.integers(len(validos)))]
    h, w = r["grade"]["forma"]
    a0, b0 = coords[int(rng.integers(len(coords)))]
    a = int(rng.integers(max(0, a0-tamanho+1), min(a0, h-tamanho)+1))
    b = int(rng.integers(max(0, b0-tamanho+1), min(b0, w-tamanho)+1))
    return _finalizar(r, modo, rng, a, b, tamanho, fs.alvo(r, modo, (a, b, tamanho)), True)


def _finalizar(r, modo, rng, a, b, t, y, ancorado):
    x, v = fs.entrada(r, modo, (a, b, t))
    fw, fh = rng.random() < .5, rng.random() < .5
    if fw:
        x, y, v = x[:, :, ::-1], y[:, ::-1], v[:, ::-1]
    if fh:
        x, y, v = x[:, ::-1], y[::-1], v[::-1]
    return (torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), torch.from_numpy(v.copy()),
            {"nome": r["nome"], "ancorado": ancorado})


@torch.no_grad()
def perda_validacao(modelo, linhas, modo):
    modelo.eval()
    soma, n = 0., 0
    for r in linhas:
        x, _ = fs.entrada(r, modo)
        y = torch.from_numpy(fs.alvo(r, modo)).unsqueeze(0)
        logits = modelo(torch.from_numpy(x).unsqueeze(0))
        soma += float(nn.functional.cross_entropy(logits, y, ignore_index=-1, reduction="sum"))
        n += int((y != -1).sum())
    return soma / max(n, 1)


@torch.no_grad()
def medir(modelo, linhas, modo):
    """Histogramas por evento; base de AP, limiar, Brier/NLL e confiabilidade."""
    modelo.eval()
    eventos = {}
    for r in linhas:
        x, _ = fs.entrada(r, modo)
        y = fs.alvo(r, modo)
        p = modelo(torch.from_numpy(x).unsqueeze(0)).softmax(1)[0, 1].numpy()
        e = eventos.setdefault(r["evento"], {"positivo": np.zeros(BINS, np.int64), "negativo": np.zeros(BINS, np.int64)})
        idx = np.minimum((p*BINS).astype(int), BINS-1)
        e["positivo"] += np.bincount(idx[y == 1], minlength=BINS)
        e["negativo"] += np.bincount(idx[y == 0], minlength=BINS)
    return {ev: {"histograma": {k: v.tolist() for k, v in e.items()},
                 "positivos": int(e["positivo"].sum()), "negativos": int(e["negativo"].sum())}
            for ev, e in eventos.items()}


def kl_mascarada(saida, professor_logits, T, mascara):
    alvo = (professor_logits/T).softmax(1)
    kl = (alvo * (alvo.clamp_min(1e-8).log() - (saida/T).log_softmax(1))).sum(1)
    if mascara is None:
        return kl.mean()
    m = mascara.float()
    return (kl*m).sum() / m.sum().clamp_min(1.)


def ajustar(pasta, job, novas, antigas, validacao, cfg, inicial=None, encoder=None):
    """job: seed, etapa, passos, rho, lwf {lambda,T,mascarada}|None, checkpoint, modo, congelar_encoder."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    if (pasta / "treino.json").exists():
        t = json.loads((pasta / "treino.json").read_text(encoding="utf-8"))
        e = torch.load(pasta / "melhor.pt", weights_only=False, map_location="cpu")
        if t["id_job"] != job["id"] or ci.hash_completo_do_estado(e["modelo"]) != t["hash_estado"]:
            raise ValueError(f"job divergente em {pasta}")
        m = novo_modelo(job["seed"], cfg["base"])
        m.load_state_dict(e["modelo"])
        return m, t
    modo, seed, k = job["modo"], job["seed"], job["etapa"]
    modelo = novo_modelo(seed, cfg["base"])
    if inicial is not None:
        modelo.load_state_dict(inicial["estado"])
    elif encoder is not None:
        carregar_encoder(modelo, encoder)
    if job.get("congelar_encoder"):
        congelar_encoder(modelo)
    professor = None
    if job.get("lwf") and inicial is not None:
        professor = copy.deepcopy(modelo).eval()
        for p in professor.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in modelo.parameters() if p.requires_grad], lr=cfg["lr"])
    epocas = job["passos"] // cfg["passos_por_epoca"]
    idx_ant = indices_antigos(cfg["batch"], job.get("rho", .5)) if antigas else set()
    inicio, melhor, hist, melhor_ep = 0, float("inf"), [], None
    custo = {"recortes_novos": 0, "recortes_memoria": 0, "pixels_supervisionados": 0, "pixels_distilados": 0,
             "passes_professor": 0, "recortes_ancorados": 0}
    retomada = pasta / "retomada.pt"
    if retomada.exists():
        r = torch.load(retomada, weights_only=False, map_location="cpu")
        if r["id_job"] != job["id"]:
            raise ValueError("retomada divergente")
        modelo.load_state_dict(r["modelo"])
        opt.load_state_dict(r["opt"])
        inicio, melhor, hist, melhor_ep, custo = r["epoca"], r["melhor"], r["historico"], r["melhor_epoca"], r["custo"]
    for epoca in range(inicio, epocas):
        modo_treino(modelo, job.get("congelar_encoder"))
        rng = np.random.default_rng([seed, k, epoca, job.get("sal", 0)])
        t0, perdas = time.monotonic(), []
        for _ in range(cfg["passos_por_epoca"]):
            xs, ys, vs = [], [], []
            for i in range(cfg["batch"]):
                fonte = antigas if i in idx_ant else novas
                x, y, v, info = recortar(fonte, modo, rng, cfg["recorte"])
                xs.append(x), ys.append(y), vs.append(v)
                custo["recortes_memoria" if i in idx_ant else "recortes_novos"] += 1
                custo["recortes_ancorados"] += int(info["ancorado"])
            x, y, v = torch.stack(xs), torch.stack(ys), torch.stack(vs)
            custo["pixels_supervisionados"] += int((y != -1).sum())
            opt.zero_grad(set_to_none=True)
            saida = modelo(x)
            loss = nn.functional.cross_entropy(saida, y, ignore_index=-1)
            if professor is not None:
                with torch.no_grad():
                    z = professor(x)
                custo["passes_professor"] += 1
                masc = v if job["lwf"]["mascarada"] else None
                custo["pixels_distilados"] += int(v.sum()) if masc is not None else int(v.numel())
                T = job["lwf"]["T"]
                loss = loss + job["lwf"]["lambda"] * T * T * kl_mascarada(saida, z, T, masc)
            if not torch.isfinite(loss):
                raise ValueError("perda nao finita")
            loss.backward()
            opt.step()
            perdas.append(float(loss.detach()))
        pv = perda_validacao(modelo, validacao, modo) if validacao else float("nan")
        ultima = epoca+1 == epocas
        if (job["checkpoint"] == "validacao" and pv < melhor) or (job["checkpoint"] == "ultima" and ultima):
            melhor, melhor_ep = (pv if job["checkpoint"] == "validacao" else melhor), epoca+1
            ci.gravar_peso(pasta / "melhor.pt", {"modelo": {a: b.detach().clone() for a, b in modelo.state_dict().items()},
                                                 "id_job": job["id"]})
        hist.append({"epoca": epoca+1, "perda_treino": float(np.mean(perdas)),
                     "perda_validacao": None if pv != pv else pv,
                     "segundos": time.monotonic()-t0})
        ci.gravar_peso(retomada, {"modelo": modelo.state_dict(), "opt": opt.state_dict(), "id_job": job["id"],
                                  "epoca": epoca+1, "melhor": melhor, "melhor_epoca": melhor_ep, "historico": hist,
                                  "custo": custo})
        print(json.dumps({"job": job["id"], "etapa": k, "epoca": epoca+1, "de": epocas,
                          "perda_validacao": None if pv != pv else round(pv, 5)}), flush=True)
    e = torch.load(pasta / "melhor.pt", weights_only=False, map_location="cpu")
    modelo.load_state_dict(e["modelo"])
    mem = sorted({r["nome"] for r in antigas})
    t = {"id_job": job["id"], "hash_estado": ci.hash_completo_do_estado(e["modelo"]), "job": job,
         "pai": None if inicial is None else inicial["hash_estado"], "melhor_epoca": melhor_ep, "historico": hist,
         "novas": [r["nome"] for r in novas], "memoria": mem, "bytes_memoria": len(mem)*BYTES_CHIP_POS,
         "custo": {**custo, "passos": job["passos"], "segundos": sum(h["segundos"] for h in hist)},
         "normalizacao": "fixa, sem dado (fontes_sar.normalizar_fixo)"}
    ci.gravar_json(pasta / "treino.json", t)
    return modelo, t


def estado(pasta, treino):
    e = torch.load(Path(pasta) / "melhor.pt", weights_only=False, map_location="cpu")
    if ci.hash_completo_do_estado(e["modelo"]) != treino["hash_estado"]:
        raise ValueError("estado divergente")
    return {"estado": e["modelo"], "hash_estado": treino["hash_estado"]}
