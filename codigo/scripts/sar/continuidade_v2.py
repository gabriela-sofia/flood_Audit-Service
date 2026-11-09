"""Campanha `continuidade_sar_v2`: memoria, custo, metodos, competencia e pre-treino.

Mesmos contextos, reservas espaciais e anotacoes-base do `continuidade_sar_v1`
(lidos do protocolo v1 com hash conferido). Muda o desenho: cinco ordens em
quadrado latino ciclico, sementes pareadas por ordem e uma familia de bracos
pre-registrada. Cada job e identificado pelo hash do que o determina; bracos
com a mesma etapa recebem o mesmo job, sem copia. A fila toma a trava pesada
por job e a devolve entre jobs. Decisoes (checkpoint) so na validacao;
reservas de retencao e eventos de avaliacao so sao inferidos depois de todos
os ajustes. Nada e promovido; svc02, escore fisico e reservas CEMS intocados.

Fontes dos metodos: Chaudhry et al. 2019 (arXiv:1902.10486), Buzzega et al.
2020 (arXiv:2004.07211), Ash & Adams 2020 (arXiv:1910.08475), Li & Hoiem 2016
(LwF, arXiv:1606.09282), He et al. 2022 (MAE, arXiv:2111.06377).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import campanha_independencia_autonomia as ci
import diagnostico_competencia as dc
import continuidade_retencao as v1
from dados_pares_geograficos import (LIMITES_DB, entrada, imagens, materializar_cache, normalizacao, particao,
                                     recortar_db, sha)
from pretreino_mascarado import DecoderLeve, EncoderUNet, mascarar, perda_mascarada

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/manutencao"))

VERSAO = "continuidade_sar_v2_1"
SAIDA = REPO / "local_runs/aprendizado-continuo/sar_v2_1"
ANTERIOR = REPO / "local_runs/aprendizado-continuo/sar_v2"
MUDANCA_V2_1 = ("v2 parou no job replay_m1 ordem 1 semente 11 etapa 2: memoria de um chip (Ghana_24858) com 484 "
                "pixels rotulados em 262.144; 100 recortes aleatorios sem rotulo. v2_1 mantem as mesmas 100 "
                "tentativas com o mesmo RNG e so depois ancora o recorte num pixel rotulado, contando os casos. "
                "Jobs que o v2 concluiu nunca alcancaram o ramo novo e sao adotados com hash conferido.")
V1 = REPO / "local_runs/aprendizado-continuo/sar_v1"
BYTES_CHIP = 2*512*512*4 + 512*512        # VV/VH pos-evento float32 + rotulo int8
BYTES_LOGITS = 2*512*512*2                # logits float16
SEMENTES_A1 = [7, 11, 23, 31, 43]
SEMENTES_A2 = [53, 61, 71, 83, 97]
PAUSA_A_CADA_S, PAUSA_S = 1200, 35

PADRAO = {"inicio": "herdado", "dados": "replay", "memoria": ["total", None], "rho": .5, "der_alpha": None,
          "lwf": None, "sp": None, "otimizador": "novo", "checkpoint": "validacao", "normalizacao": "etapa1",
          "selecao": "aleatoria", "peso_classe": False, "encoder": "aleatorio", "congelar_encoder": False,
          "passos": "S"}


def braco(**k):
    r = {**PADRAO, **k}
    if set(r) != set(PADRAO):
        raise ValueError(f"campo desconhecido: {set(r) - set(PADRAO)}")
    return r


BRACOS = {
    # F2 referencias de custo e continuidade
    "reinicio": braco(inicio="aleatorio", dados="uniao", memoria=[None, None]),
    "continuacao": braco(dados="novo", memoria=[None, None]),
    "replay_total": braco(),
    "reinicio_passos_acumulados": braco(inicio="aleatorio", dados="uniao", memoria=[None, None], passos="kS"),
    # F1 memoria crescente (m por contexto antigo) e constante (M no total)
    "replay_m1": braco(memoria=["por_contexto_m", 1]),
    "replay_m2": braco(memoria=["por_contexto_m", 2]),
    "replay_m4": braco(memoria=["por_contexto_m", 4]),
    "replay_M12_reservatorio": braco(memoria=["reservatorio_M", 12]),
    "replay_M4_reservatorio": braco(memoria=["reservatorio_M", 4]),
    "replay_M4_por_contexto": braco(memoria=["por_contexto_M", 4]),
    "replay_M12_por_contexto": braco(memoria=["por_contexto_M", 12]),
    "replay_M12_agua": braco(memoria=["agua_M", 12]),
    "replay_M12_representativo": braco(memoria=["representativo_M", 12]),
    "replay_rho25": braco(rho=.25),
    "replay_rho75": braco(rho=.75),
    # F2 sensibilidades
    "replay_otimizador_herdado": braco(otimizador="herdado"),
    "replay_checkpoint_ultima": braco(checkpoint="ultima"),
    "replay_normalizacao_recalculada": braco(normalizacao="recalculada"),
    # F3 metodos
    "der_M12": braco(memoria=["reservatorio_M", 12], der_alpha=.5),
    "der_M8": braco(memoria=["reservatorio_M", 8], der_alpha=.5),
    "lwf": braco(dados="novo", memoria=[None, None], lwf={"lambda": 1., "T": 2.}),
    "sp_replay": braco(sp={"lambda": .6, "ruido": .01}),
    "sp_continuacao": braco(dados="novo", memoria=[None, None], sp={"lambda": .6, "ruido": .01}),
    # F4 competencia e aquisicao
    "replay_ce_ponderada": braco(peso_classe=True),
    "replay_selecao_representativa": braco(selecao="kcentro"),
    # F6 representacoes
    "mae_ajuste": braco(encoder="mae"),
    "mae_congelado": braco(encoder="mae", congelar_encoder=True),
    "aleatorio_congelado": braco(congelar_encoder=True),
}
TIERS = {
    "A": ["reinicio", "continuacao", "replay_total", "replay_m1", "replay_m2", "replay_m4",
          "replay_M12_reservatorio", "der_M8", "lwf", "sp_replay"],
    "B": ["reinicio_passos_acumulados", "der_M12", "replay_M4_reservatorio", "replay_M4_por_contexto",
          "replay_M12_por_contexto", "replay_M12_agua", "replay_M12_representativo", "replay_rho25",
          "replay_rho75", "replay_otimizador_herdado", "replay_checkpoint_ultima",
          "replay_normalizacao_recalculada", "sp_continuacao", "replay_ce_ponderada",
          "replay_selecao_representativa"],
    "C": ["mae_ajuste", "mae_congelado", "aleatorio_congelado"],
}
MAE = {"passos": 1920, "batch": 16, "recorte": 64, "fracao": .75, "bloco": 8, "lr": 1e-3, "semente": 7}


# ---------------------------------------------------------------- desenho

def ordens(contextos):
    """Quadrado latino ciclico: cada contexto ocupa cada posicao uma vez."""
    return [contextos[i:] + contextos[:i] for i in range(len(contextos))]


def sequencias():
    seq = []
    for tier, sementes in (("A1", SEMENTES_A1), ("B", SEMENTES_A1), ("C", SEMENTES_A1), ("A2", SEMENTES_A2)):
        for o, s in enumerate(sementes):
            seq.append({"bloco": tier, "ordem": o, "seed": s,
                        "bracos": TIERS["A" if tier.startswith("A") else tier]})
    return seq


def anotacoes_base(tr, divisoes, canonica, seed, n):
    """Mesma regra do v1: semente*100 + posicao canonica; independe da ordem."""
    r = {}
    for i, e in enumerate(canonica, 1):
        itens = [x for x in tr if x["nome"] in divisoes[e]["ajuste"]]
        r[e] = ci.escolher(itens, "aleatoria", n, seed*100+i)
    return r


def kcentro(vetores, nomes, k):
    """Greedy k-centro (Sener & Savarese 2018): primeiro o mais proximo da media."""
    v = np.asarray(vetores, float)
    if k >= len(nomes):
        return list(nomes)
    centro = v.mean(0)
    escolhidos = [int(np.lexsort((np.array(nomes), ((v-centro)**2).sum(1)))[0])]
    d = ((v - v[escolhidos[0]])**2).sum(1)
    while len(escolhidos) < k:
        candidatos = [i for i in range(len(nomes)) if i not in escolhidos]
        i = max(candidatos, key=lambda j: (d[j], nomes[j]))
        escolhidos.append(i)
        d = np.minimum(d, ((v - v[i])**2).sum(1))
    return [nomes[i] for i in escolhidos]


def descritor(r):
    """Histograma radiometrico pos-evento sem rotulo; base da representatividade."""
    x, v = imagens(r["caminhos"]["antes"], r["caminhos"]["depois"])
    x = recortar_db(x)
    partes = []
    for b in range(2):
        lo, hi = LIMITES_DB[b]
        h, _ = np.histogram(x[b][v], bins=16, range=(lo, hi))
        partes.append(h / max(h.sum(), 1))
    return np.concatenate(partes + [[v.mean()]]).tolist()


def memoria(politica, param, ctx, k, anot, seed, canonica, agua=None, desc=None):
    """Chips disponiveis para replay na etapa k, vindos de ctx[:k-1]."""
    antigos = ctx[:k-1]
    if politica is None or not antigos:
        return []
    if politica == "total":
        return [n for c in antigos for n in anot[c]]
    perm = {c: list(np.random.default_rng([seed, canonica.index(c), 17]).permutation(anot[c])) for c in antigos}
    if politica == "por_contexto_m":
        return [n for c in antigos for n in perm[c][:param]]
    if politica == "reservatorio_M":
        rng, buf, visto = np.random.default_rng([seed, 23]), [], 0
        for c in antigos:
            for n in anot[c]:
                visto += 1
                if len(buf) < param:
                    buf.append(n)
                else:
                    j = int(rng.integers(visto))
                    if j < param:
                        buf[j] = n
        return buf
    cotas = {c: param // len(antigos) + (1 if i < param % len(antigos) else 0) for i, c in enumerate(antigos)}
    r = []
    for c in antigos:
        if politica == "por_contexto_M":
            r += perm[c][:cotas[c]]
        elif politica == "agua_M":
            r += sorted(anot[c], key=lambda n: (-agua[n], n))[:cotas[c]]
        elif politica == "representativo_M":
            r += kcentro([desc[n] for n in anot[c]], list(anot[c]), cotas[c])
        else:
            raise ValueError(f"politica de memoria desconhecida: {politica}")
    return r


def indices_antigos(batch, rho):
    n = int(round(rho*batch))
    return {int((j+.5)*batch/n) for j in range(n)} if n else set()


# ---------------------------------------------------------------- recorte com janela

def recortar(linhas, estat, modo, rng, tamanho):
    """Mesmo consumo de RNG de `dc.recorte` uniforme, devolvendo janela e espelhos."""
    for _ in range(100):
        r = linhas[int(rng.integers(len(linhas)))]
        h, w = r["grade"]["forma"]
        a, b = int(rng.integers(h-tamanho+1)), int(rng.integers(w-tamanho+1))
        x, sensor = entrada(r, estat, modo, (a, b, tamanho))
        y, _, _ = ci.alvo(r, sensor, (a, b, tamanho))
        if (y != -1).any():
            fw, fh = rng.random() < .5, rng.random() < .5
            if fw:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if fh:
                x, y = x[:, ::-1], y[::-1]
            return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), (r["nome"], a, b, fw, fh)
    # v2_1: chip de rotulo esparso. Ancora num pixel rotulado em vez de perder o passo.
    validos = [(r, c) for r in linhas if len(c := rotulados(r))]
    if not validos:
        raise ValueError("nenhum pixel rotulado nos chips; passos nao podem desaparecer")
    r, coords = validos[int(rng.integers(len(validos)))]
    h, w = r["grade"]["forma"]
    a0, b0 = coords[int(rng.integers(len(coords)))]
    a = int(rng.integers(max(0, a0-tamanho+1), min(a0, h-tamanho)+1))
    b = int(rng.integers(max(0, b0-tamanho+1), min(b0, w-tamanho)+1))
    x, sensor = entrada(r, estat, modo, (a, b, tamanho))
    y, _, _ = ci.alvo(r, sensor, (a, b, tamanho))
    fw, fh = rng.random() < .5, rng.random() < .5
    if fw:
        x, y = x[:, :, ::-1], y[:, ::-1]
    if fh:
        x, y = x[:, ::-1], y[::-1]
    return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), (r["nome"], a, b, fw, fh, "ancorado")


_ROTULADOS = {}


def rotulados(r):
    if r["nome"] not in _ROTULADOS:
        _, sensor = imagens(r["caminhos"]["antes"], r["caminhos"]["depois"])
        _, _, mascara = ci.alvo(r, sensor)
        _ROTULADOS[r["nome"]] = np.argwhere(mascara)
    return _ROTULADOS[r["nome"]]


def recortar_logits(z, a, b, t, fw, fh):
    z = z[:, a:a+t, b:b+t]
    if fw:
        z = z[:, :, ::-1]
    if fh:
        z = z[:, ::-1]
    return torch.from_numpy(np.ascontiguousarray(z).astype(np.float32))


# ---------------------------------------------------------------- modelo

def novo_modelo(seed, base):
    ci.semear(seed)
    return ci.UNetPequena(canais_entrada=4, base=base)


def encolher_e_perturbar(modelo, seed, etapa, base, lam, ruido):
    """Ash & Adams 2020: theta <- lam*theta + ruido*theta_aleatorio, so em parametros."""
    estado = torch.get_rng_state()
    try:
        torch.manual_seed(seed*1000 + etapa)
        fresco = ci.UNetPequena(canais_entrada=4, base=base)
    finally:
        torch.set_rng_state(estado)
    with torch.no_grad():
        for (n, p), (_, q) in zip(modelo.named_parameters(), fresco.named_parameters()):
            p.mul_(lam).add_(ruido*q)


def carregar_encoder(modelo, enc_estado):
    """Encoder MAE de 2 canais em U-Net de 4: entrada pos duplicada, pesos divididos ao meio."""
    estado = modelo.state_dict()
    for k, v in enc_estado.items():
        if k not in estado:
            raise ValueError(f"chave do encoder ausente na U-Net: {k}")
        if k == "enc1.0.weight":
            v = torch.cat([v/2, v/2], 1)
        if estado[k].shape != v.shape:
            raise ValueError(f"forma divergente em {k}")
        estado[k] = v.clone()
    modelo.load_state_dict(estado)


def modo_treino(modelo, congelar):
    modelo.train()
    if congelar:
        for nome in ("enc1", "enc2", "enc3"):
            getattr(modelo, nome).eval()


def congelar_encoder(modelo):
    for nome in ("enc1", "enc2", "enc3"):
        for p in getattr(modelo, nome).parameters():
            p.requires_grad_(False)


# ---------------------------------------------------------------- jobs

def job_id(conteudo):
    return ci.hash_json(conteudo)[:20]


def planejar(protocolo, tr):
    """Lista de jobs com dependencias; jobs identicos colapsam pelo id."""
    cfg, canonica = protocolo["config"], protocolo["contextos_canonicos"]
    por_id, ordem_exec = {}, []

    def registrar(j):
        if j["id"] not in por_id:
            por_id[j["id"]] = j
            ordem_exec.append(j["id"])
        return j["id"]

    mae_id = None
    if any("C" == s["bloco"] for s in protocolo["sequencias"]):
        mae_id = registrar({"id": job_id({"tipo": "pretreino", "mae": protocolo["mae"],
                                          "chips": protocolo["chips_pretreino"]}),
                            "tipo": "pretreino", "pai": None, "bloco": "C"})
    linhagens = []
    for s in protocolo["sequencias"]:
        ctx = protocolo["ordens"][s["ordem"]]
        for nome in s["bracos"]:
            spec = BRACOS[nome]
            pai, ids = None, []
            for k in range(1, len(ctx)+1):
                relevante = {c: v for c, v in spec.items()}
                if k == 1:
                    # Etapa 1 e do zero para todos; so importam campos que a afetam.
                    relevante = {c: spec[c] for c in ("checkpoint", "selecao", "peso_classe", "encoder",
                                                      "congelar_encoder")}
                conteudo = {"spec": relevante, "seed": s["seed"], "ctx": ctx[:k], "etapa": k,
                            "pai": pai if (k > 1 and spec["inicio"] == "herdado") else None,
                            "mae": mae_id if spec["encoder"] == "mae" else None,
                            "passos": cfg["passos_por_etapa"]*(k if spec["passos"] == "kS" else 1)}
                j = {"id": job_id(conteudo), "tipo": "ajuste", "braco_spec": spec, "seed": s["seed"],
                     "ordem": s["ordem"], "ctx": ctx, "etapa": k, "pai": conteudo["pai"], "mae": conteudo["mae"],
                     "passos": conteudo["passos"], "bloco": s["bloco"]}
                pai = registrar(j)
                ids.append(pai)
            linhagens.append({"bloco": s["bloco"], "ordem": s["ordem"], "seed": s["seed"], "braco": nome,
                              "jobs": ids})
            if spec["der_alpha"]:
                for jid in ids[:-1]:
                    por_id[jid]["salvar_logits"] = True
    return por_id, ordem_exec, linhagens


def preparar(config=None, v1_dir=V1):
    p1 = dc.ler(Path(v1_dir) / "protocolo.json")
    if sha(Path(v1_dir) / "protocolo.json") != "488d0bc793be7633b1e15cffad3471b10f05b0c6750d0c8be75633b71887b3e5" \
            and config is None:
        raise ValueError("protocolo v1 divergente do registrado")
    campanha = Path(p1["campanha"])
    inv = dc.ler(campanha / "inventario.json")
    if ci.hash_json(inv) != p1["inventario"]:
        raise ValueError("inventario divergente")
    eventos = dc.ler(campanha / "protocolo.json")["ordem_eventos"]
    tr, va, te = particao(inv["pares"], eventos[:3], eventos[3:5])
    cfg = {**p1["config"], "modo": p1["modo"], **(config or {})}
    canonica = p1["contextos_em_ordem"]
    ajuste_ou_tampao = sorted(n for d in p1["divisoes"].values() for n in d["ajuste"] + d["tampao"])
    seqs = sequencias() if not config or "sequencias" not in config else config["sequencias"]
    cfg.pop("sequencias", None)
    mae = {**MAE, **cfg.pop("mae", {})}
    anot = {str(s): anotacoes_base(tr, p1["divisoes"], canonica, s, cfg["anotacoes_por_contexto"])
            for s in sorted({x["seed"] for x in seqs})}
    protocolo = {
        "versao": VERSAO, "v1_protocolo_sha256": sha(Path(v1_dir) / "protocolo.json"), "campanha": str(campanha),
        "versao_anterior": ({"pasta": str(ANTERIOR), "protocolo_sha256": sha(ANTERIOR / "protocolo.json"),
                             "mudanca": MUDANCA_V2_1, "adocao": "jobs concluidos com mesmo id, hash conferido"}
                            if config is None and (ANTERIOR / "protocolo.json").exists() else None),
        "inventario": p1["inventario"], "config": cfg, "contextos_canonicos": canonica,
        "ordens": ordens(canonica), "sequencias": seqs, "divisoes": p1["divisoes"], "anotacoes_base": anot,
        "validacao": eventos[3:5], "avaliacao_desenvolvimento": eventos[:3],
        "limiar": p1["congelado_da_validacao"]["limiar"], "bracos": BRACOS, "tiers": TIERS, "mae": mae,
        "chips_pretreino": ajuste_ou_tampao,
        "pretreino_fonte": "imagens pos-evento sem rotulo de ajuste+tampao dos 5 contextos; nunca reserva, "
                           "validacao ou avaliacao",
        "codigo": {str(f): sha(f) for f in fontes()},
        "familias": familias(), "regras": regras(cfg),
        "regime": "desenvolvimento em dados conhecidos", "promovido": False, "altera_svc02": False,
        "altera_escore_suscetibilidade": False, "consome_reservas_cems": False,
        "torch": torch.__version__, "numpy": np.__version__}
    return protocolo, inv, (tr, va, te)


def fontes():
    return [Path(__file__), Path(v1.__file__), Path(ci.__file__), Path(dc.__file__),
            REPO / "scripts/sar/dados_pares_geograficos.py", REPO / "scripts/sar/modelo_unet.py",
            REPO / "scripts/sar/pretreino_mascarado.py", REPO / "scripts/neural/proveniencia_de_peso.py",
            REPO / "scripts/sar/baseline_otsu.py", REPO / "scripts/sar/experimento_pre_pos.py"]


def familias():
    return {
        "F1_memoria": {"pergunta": "quanta memoria e qual preenchimento preservam aquisicao e retencao?",
                       "contrastes": [["replay_m1", "continuacao"], ["replay_m2", "replay_m1"],
                                      ["replay_m4", "replay_m2"], ["replay_total", "replay_m4"],
                                      ["replay_M12_reservatorio", "replay_m4"],
                                      ["replay_M4_por_contexto", "replay_M4_reservatorio"],
                                      ["replay_M12_por_contexto", "replay_M12_reservatorio"],
                                      ["replay_M12_agua", "replay_M12_por_contexto"],
                                      ["replay_M12_representativo", "replay_M12_por_contexto"],
                                      ["replay_rho25", "replay_total"], ["replay_rho75", "replay_total"]],
                       "hipotese_literatura": "Chaudhry 2019: com < 3 exemplos por classe, buffer balanceado "
                                              "supera reservatorio; com mais, reservatorio e melhor ou igual"},
        "F2_custo": {"pergunta": "o que continuar pesos acrescenta alem de passos, e quanto custa manter?",
                     "contrastes": [["replay_total", "reinicio"], ["continuacao", "reinicio"],
                                    ["reinicio_passos_acumulados", "reinicio"],
                                    ["replay_total", "reinicio_passos_acumulados"],
                                    ["replay_otimizador_herdado", "replay_total"],
                                    ["replay_checkpoint_ultima", "replay_total"],
                                    ["replay_normalizacao_recalculada", "replay_total"]]},
        "F3_metodos": {"pergunta": "distilacao de logits ou warm-start regularizado superam o replay "
                                   "sob orcamento equivalente?",
                       "contrastes": [["der_M12", "replay_M12_reservatorio"], ["der_M8", "replay_M12_reservatorio"],
                                      ["lwf", "continuacao"], ["sp_replay", "replay_total"],
                                      ["sp_continuacao", "continuacao"]]},
        "F4_competencia": {"pergunta": "desequilibrio ou representatividade explicam contextos fracos?",
                           "contrastes": [["replay_ce_ponderada", "replay_total"],
                                          ["replay_selecao_representativa", "replay_total"]]},
        "F6_representacao": {"pergunta": "o encoder MAE transfere para segmentacao com passos e selecao iguais?",
                             "contrastes": [["mae_ajuste", "replay_total"], ["mae_congelado", "aleatorio_congelado"],
                                            ["aleatorio_congelado", "replay_total"]]},
    }


def regras(cfg):
    return {
        "unidade": "evento; ordens e sementes agregadas dentro do evento antes de qualquer teste",
        "desfechos": {"primario": "IoU no limiar congelado do modelo final, 8 eventos (5 reservas de retencao "
                                  "+ 3 de avaliacao), sempre relatado tambem separado",
                      "retencao": "5 reservas de contextos conhecidos (n=5 < 6: estimacao, nao decisao)",
                      "transferencia": "3 eventos nunca treinados (n=3: estimacao, nao decisao)",
                      "secundarios": ["AP", "esquecimento F", "BWT", "aquisicao do contexto novo",
                                      "transferencia por etapa", "pior evento", "custo"]},
        "decisao": (f"ganho/perda: Wilcoxon com Holm dentro da familia < 0,05 E IC95 t por evento alem de "
                    f"+-{cfg['relevancia']}; equivalencia so por TOST (IC90 t inteiro dentro de +-relevancia); "
                    f"n < {cfg['minimo_eventos']} ou resto: INDECIDIVEL. p > 0,05 nunca e equivalencia."),
        "repeticoes": "numero de ordens e sementes fixado antes; nenhuma rodada extra condicionada a resultado",
        "adaptacao": "nenhuma decisao adaptativa le reserva ou avaliacao; checkpoint so por validacao",
    }


# ---------------------------------------------------------------- execucao de um job

class Contexto:
    def __init__(self, saida, protocolo, inv, particoes):
        self.saida, self.p, self.inv = Path(saida), protocolo, inv
        self.tr, self.va, self.te = particoes
        self.por_nome = {r["nome"]: r for r in self.tr}
        self.cfg = protocolo["config"]
        self.assinatura = ci.hash_json(protocolo)
        self._agua, self._desc = {}, {}

    def agua(self, nomes):
        for n in nomes:
            if n not in self._agua:
                r = self.por_nome[n]
                _, sensor = imagens(r["caminhos"]["antes"], r["caminhos"]["depois"])
                y, _, mascara = ci.alvo(r, sensor)
                self._agua[n] = float((y == 1).sum() / max(mascara.sum(), 1))
        return self._agua

    def desc(self, nomes):
        for n in nomes:
            if n not in self._desc:
                self._desc[n] = descritor(self.por_nome[n])
        return self._desc

    def pasta(self, jid):
        return self.saida / "jobs" / jid


def anotacoes_do_job(cx, job, por_id, cache):
    """Anotacoes por contexto ate a etapa do job. Selecao representativa usa so pixels sem rotulo."""
    spec, seed, ctx = job["braco_spec"], job["seed"], job["ctx"]
    base = cx.p["anotacoes_base"][str(seed)]
    if spec["selecao"] == "aleatoria":
        return {c: base[c] for c in ctx}
    chave = (seed, tuple(ctx))
    if chave not in cache:
        r = {}
        for c in ctx:
            pool = sorted(cx.p["divisoes"][c]["ajuste"])
            d = cx.desc(pool)
            r[c] = kcentro([d[n] for n in pool], pool, cx.cfg["anotacoes_por_contexto"])
        cache[chave] = r
    return {c: cache[chave][c] for c in ctx}


def executar_job(cx, job, por_id, cache_anot):
    pasta = cx.pasta(job["id"])
    if (pasta / "treino.json").exists():
        return dc.ler(pasta / "treino.json")
    if job["tipo"] == "pretreino":
        return pretreinar(cx, job, pasta)
    cfg, spec, seed, k, ctx = cx.cfg, job["braco_spec"], job["seed"], job["etapa"], job["ctx"]
    anot = anotacoes_do_job(cx, job, por_id, cache_anot)
    novo = ctx[k-1]
    canonica = cx.p["contextos_canonicos"]
    mem_nomes = memoria(spec["memoria"][0], spec["memoria"][1], ctx, k, anot, seed, canonica,
                        agua=cx.agua([n for c in ctx[:k-1] for n in anot[c]]) if spec["memoria"][0] == "agua_M" else None,
                        desc=cx.desc([n for c in ctx[:k-1] for n in anot[c]])
                        if spec["memoria"][0] == "representativo_M" else None) if spec["dados"] == "replay" else []
    novas = [cx.por_nome[n] for n in anot[novo]]
    if spec["dados"] == "uniao" or k == 1:
        if spec["passos"] == "kS":
            linhas = [cx.por_nome[n] for c in ctx[:k] for n in anot[c]]
        else:
            linhas = novas + [cx.por_nome[n] for c in ctx[:k-1] for n in anot[c]]
        antigas = []
    else:
        linhas = novas
        antigas = [cx.por_nome[n] for n in mem_nomes]
    acumuladas = [cx.por_nome[n] for c in ctx[:k] for n in anot[c]]
    if spec["normalizacao"] == "recalculada" and spec["inicio"] == "herdado":
        estat = normalizacao(acumuladas)
    else:
        estat = normalizacao([cx.por_nome[n] for n in anot[ctx[0]]])
    pai = por_id[job["pai"]] if job["pai"] else None
    modelo = novo_modelo(seed, cfg["base"])
    estado_pai = None
    if pai is not None:
        estado_pai = torch.load(cx.pasta(pai["id"]) / "melhor.pt", weights_only=False, map_location="cpu")
        ref = dc.ler(cx.pasta(pai["id"]) / "treino.json")
        if ci.hash_completo_do_estado(estado_pai["modelo"]) != ref["hash_estado"]:
            raise ValueError("estado herdado divergente")
        modelo.load_state_dict(estado_pai["modelo"])
        if spec["sp"]:
            encolher_e_perturbar(modelo, seed, k, cfg["base"], spec["sp"]["lambda"], spec["sp"]["ruido"])
    elif spec["encoder"] == "mae":
        enc = torch.load(cx.pasta(job["mae"]) / "encoder.pt", weights_only=False, map_location="cpu")
        if ci.hash_completo_do_estado(enc["encoder"]) != dc.ler(cx.pasta(job["mae"]) / "treino.json")["hash_encoder"]:
            raise ValueError("encoder MAE divergente")
        carregar_encoder(modelo, enc["encoder"])
    if spec["congelar_encoder"]:
        congelar_encoder(modelo)
    professor = None
    if spec["lwf"] and pai is not None:
        professor = copy.deepcopy(modelo).eval()
        for p in professor.parameters():
            p.requires_grad_(False)
    logits_mem = {}
    if spec["der_alpha"] and antigas:
        for n in {r["nome"] for r in antigas}:
            origem = _logits_de(cx, por_id, job, n)
            logits_mem[n] = np.load(origem)["z"]
    peso = None
    if spec["peso_classe"]:
        sp_, _ = dc.suporte(linhas + antigas)
        peso = torch.tensor(dc.pesos_classe(sp_), dtype=torch.float32)
    treinaveis = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.Adam(treinaveis, lr=cfg["lr"])
    if spec["otimizador"] == "herdado" and estado_pai is not None and "opt" in estado_pai:
        opt.load_state_dict(estado_pai["opt"])
    id_job = job["id"]
    epocas = job["passos"] // cfg["passos_por_epoca"]
    idx_antigos = indices_antigos(cfg["batch"], spec["rho"]) if antigas else set()
    pasta.mkdir(parents=True, exist_ok=True)
    inicio, melhor, historico, melhor_epoca = 0, float("inf"), [], None
    apresentacoes = {}
    retomada = pasta / "retomada.pt"
    if retomada.exists():
        r = torch.load(retomada, weights_only=False, map_location="cpu")
        if r["id_job"] != id_job:
            raise ValueError("retomada divergente")
        modelo.load_state_dict(r["modelo"])
        opt.load_state_dict(r["opt"])
        torch.set_rng_state(r["rng"])
        inicio, melhor, historico, melhor_epoca, apresentacoes = (r["epoca"], r["melhor"], r["historico"],
                                                                 r["melhor_epoca"], r["apresentacoes"])
    t_inicio = time.monotonic()
    for epoca in range(inicio, epocas):
        modo_treino(modelo, spec["congelar_encoder"])
        rng = np.random.default_rng([seed, k, epoca])
        perdas, t0 = [], time.monotonic()
        contagem = {"positivo": 0, "negativo": 0, "ignorado": 0, "recortes_antigos": 0, "recortes_ancorados": 0}
        for _ in range(cfg["passos_por_epoca"]):
            xs, ys, zs, mem = [], [], [], []
            for i in range(cfg["batch"]):
                fonte = antigas if i in idx_antigos else linhas
                x, y, info = recortar(fonte, estat, cfg["modo"], rng, cfg["recorte"])
                xs.append(x)
                ys.append(y)
                apresentacoes[info[0]] = apresentacoes.get(info[0], 0) + 1
                contagem["recortes_ancorados"] += int(len(info) > 5)
                if i in idx_antigos:
                    contagem["recortes_antigos"] += 1
                    if logits_mem:
                        mem.append(i)
                        zs.append(recortar_logits(logits_mem[info[0]], info[1], info[2], cfg["recorte"], info[3],
                                                  info[4]))
            x, y = torch.stack(xs), torch.stack(ys)
            for c, valor in (("positivo", 1), ("negativo", 0), ("ignorado", -1)):
                contagem[c] += int((y == valor).sum())
            opt.zero_grad(set_to_none=True)
            saida = modelo(x)
            loss = nn.functional.cross_entropy(saida, y, weight=peso, ignore_index=-1)
            if mem:
                loss = loss + spec["der_alpha"] * ((saida[mem] - torch.stack(zs))**2).mean()
            if professor is not None:
                T = spec["lwf"]["T"]
                with torch.no_grad():
                    alvo_p = (professor(x)/T).softmax(1)
                kl = (alvo_p * (alvo_p.clamp_min(1e-8).log() - (saida/T).log_softmax(1))).sum(1).mean()
                loss = loss + spec["lwf"]["lambda"] * T * T * kl
            if not torch.isfinite(loss):
                raise ValueError("perda nao finita")
            loss.backward()
            opt.step()
            perdas.append(float(loss.detach()))
        va = ci.avaliar(modelo, cx.va, estat, cfg["modo"])
        ultima = epoca + 1 == epocas
        if (spec["checkpoint"] == "validacao" and va["perda"] < melhor) or (spec["checkpoint"] == "ultima" and ultima):
            melhor, melhor_epoca = (va["perda"] if spec["checkpoint"] == "validacao" else melhor), epoca+1
            ci.gravar_peso(pasta / "melhor.pt", {"modelo": {c: v.detach().clone() for c, v in modelo.state_dict().items()},
                                                 "opt": copy.deepcopy(opt.state_dict()), "id_job": id_job,
                                                 "estat": estat, "assinatura": cx.assinatura})
        historico.append({"epoca": epoca+1, "pixels_apresentados": contagem, "perda_treino": float(np.mean(perdas)),
                          "perda_validacao": va["perda"], "segundos": time.monotonic()-t0})
        ci.gravar_peso(retomada, {"modelo": modelo.state_dict(), "opt": opt.state_dict(), "rng": torch.get_rng_state(),
                                  "id_job": id_job, "epoca": epoca+1, "melhor": melhor, "melhor_epoca": melhor_epoca,
                                  "historico": historico, "apresentacoes": apresentacoes})
        print(json.dumps({"job": id_job, "etapa": k, "epoca": epoca+1, "de": epocas,
                          "perda_validacao": round(va["perda"], 5)}), flush=True)
    e = torch.load(pasta / "melhor.pt", weights_only=False, map_location="cpu")
    modelo.load_state_dict(e["modelo"])
    validacao = dc.medir(modelo, cx.va, estat, cfg["modo"])
    ci.gravar_json(pasta / "validacao.json", validacao)
    # Logits dos chips recem-anotados, gravados na fronteira do contexto (adaptacao do DER),
    # so nos jobs que alguma linhagem DER herda: guardar em todos custaria ~10 GB sem uso.
    if job.get("salvar_logits"):
        z_dir = pasta / "logits"
        z_dir.mkdir(exist_ok=True)
        modelo.eval()
        with torch.no_grad():
            for r in novas:
                x, _ = entrada(r, estat, cfg["modo"])
                z = modelo(torch.from_numpy(x).unsqueeze(0))[0].numpy().astype(np.float16)
                np.savez_compressed(z_dir / f"{r['nome']}.npz", z=z)
    memoria_bytes = len(mem_nomes)*BYTES_CHIP + (len(mem_nomes)*BYTES_LOGITS if spec["der_alpha"] else 0)
    r = {"id_job": id_job, "hash_estado": ci.hash_completo_do_estado(e["modelo"]), "seed": seed, "etapa": k,
         "ctx": ctx, "pai": job["pai"], "hash_pai": None if pai is None else ci.hash_completo_do_estado(estado_pai["modelo"]),
         "spec": spec, "novas": [x["nome"] for x in novas], "antigas_memoria": mem_nomes,
         "linhas_treino": [x["nome"] for x in linhas], "estat": estat, "melhor_epoca": melhor_epoca,
         "historico": historico, "passos_executados": job["passos"], "apresentacoes_por_chip": apresentacoes,
         "apresentacoes_novas": sum(v for n, v in apresentacoes.items() if n in {x["nome"] for x in novas}),
         "apresentacoes_memoria": sum(v for n, v in apresentacoes.items() if n in set(mem_nomes)),
         "bytes_memoria": memoria_bytes, "anotacoes_acumuladas": len(acumuladas),
         "segundos": sum(h["segundos"] for h in historico), "segundos_parede": time.monotonic()-t_inicio,
         "validacao_resumo": dc.resumo_limiar(validacao, cx.p["limiar"]),
         "checkpoint": spec["checkpoint"], "otimizador": spec["otimizador"],
         "recortes_ancorados": sum(h["pixels_apresentados"].get("recortes_ancorados", 0) for h in historico)}
    ci.gravar_json(pasta / "treino.json", r)
    return r


def _logits_de(cx, por_id, job, nome):
    """Procura, subindo a linhagem, o job em que o chip foi anotado e gravou logits."""
    atual = por_id[job["pai"]]
    while atual is not None:
        f = cx.pasta(atual["id"]) / "logits" / f"{nome}.npz"
        if f.exists():
            return f
        atual = por_id[atual["pai"]] if atual.get("pai") else None
    raise ValueError(f"logits ausentes para {nome}")


def pretreinar(cx, job, pasta):
    """MAE com o encoder da U-Net; so imagens pos-evento de ajuste+tampao, sem rotulo."""
    pasta.mkdir(parents=True, exist_ok=True)
    chips = [cx.por_nome[n] for n in cx.p["chips_pretreino"]]
    MAE = cx.p["mae"]
    ci.semear(MAE["semente"])
    modelo = nn.Module()
    modelo.encoder = EncoderUNet(2, cx.cfg["base"])
    modelo.decoder = DecoderLeve(cx.cfg["base"], 2)
    estat = normalizacao(chips)
    opt = torch.optim.Adam(modelo.parameters(), lr=MAE["lr"])
    gerador = torch.Generator().manual_seed(MAE["semente"])
    rng = np.random.default_rng(MAE["semente"])
    perdas, t0 = [], time.monotonic()
    for passo in range(MAE["passos"]):
        xs, vs = [], []
        for _ in range(MAE["batch"]):
            r = chips[int(rng.integers(len(chips)))]
            h, w = r["grade"]["forma"]
            a, b = int(rng.integers(h-MAE["recorte"]+1)), int(rng.integers(w-MAE["recorte"]+1))
            x, v = entrada(r, estat, "pos", (a, b, MAE["recorte"]))
            xs.append(torch.from_numpy(x[:2].copy()))
            vs.append(torch.from_numpy(v.copy()))
        x, v = torch.stack(xs), torch.stack(vs).unsqueeze(1)
        mascarada, escondido = mascarar(x, MAE["fracao"], MAE["bloco"], gerador)
        opt.zero_grad(set_to_none=True)
        perda = perda_mascarada(modelo.decoder(modelo.encoder(mascarada)[2]), x, escondido, v)
        perda.backward()
        opt.step()
        perdas.append(float(perda.detach()))
        if (passo+1) % 96 == 0:
            print(json.dumps({"job": job["id"], "pretreino_passo": passo+1, "perda": round(float(np.mean(perdas[-96:])), 5)}),
                  flush=True)
    enc = {k: v.detach().clone() for k, v in modelo.encoder.state_dict().items()}
    ci.gravar_peso(pasta / "encoder.pt", {"encoder": enc, "estat": estat, "mae": MAE})
    r = {"id_job": job["id"], "hash_encoder": ci.hash_completo_do_estado(enc), "passos": MAE["passos"],
         "perda_inicial": float(np.mean(perdas[:96])), "perda_final": float(np.mean(perdas[-96:])),
         "segundos": time.monotonic()-t0, "chips": len(chips), "hash_estado": None,
         "nao_e": "reconstrucao menor nao demonstra segmentacao melhor"}
    ci.gravar_json(pasta / "treino.json", r)
    return r


# ---------------------------------------------------------------- fila

def adotar(cx, jid, job, anterior):
    """Copia um job concluido da versao anterior se o id coincide e o peso confere."""
    origem = Path(anterior) / "jobs" / jid
    if not (origem / "treino.json").exists():
        return False
    t = dc.ler(origem / "treino.json")
    if job["tipo"] == "ajuste":
        e = torch.load(origem / "melhor.pt", weights_only=False, map_location="cpu")
        if t["id_job"] != jid or ci.hash_completo_do_estado(e["modelo"]) != t["hash_estado"]:
            raise ValueError(f"job anterior divergente: {jid}")
    else:
        e = torch.load(origem / "encoder.pt", weights_only=False, map_location="cpu")
        if ci.hash_completo_do_estado(e["encoder"]) != t["hash_encoder"]:
            raise ValueError(f"encoder anterior divergente: {jid}")
    destino = cx.pasta(jid)
    shutil.copytree(origem, destino, ignore=shutil.ignore_patterns("retomada.pt"))
    ci.gravar_json(destino / "adocao.json", {"origem": str(origem), "versao_origem": "continuidade_sar_v2",
                                            "protocolo_origem_sha256": sha(Path(anterior) / "protocolo.json"),
                                            "motivo": MUDANCA_V2_1})
    return True


def fila(saida, protocolo, inv, particoes, por_trava=True, esperar_min=240., limite_jobs=None):
    import executar_serializado as serial
    cx = Contexto(saida, protocolo, inv, particoes)
    por_id, ordem_exec, linhagens = planejar(protocolo, cx.tr)
    torch.set_num_threads(cx.cfg["threads"])
    torch.use_deterministic_algorithms(True)
    materializar_cache(inv, Path(protocolo["campanha"]) / "cache")
    feitos, cache_anot, ultima_pausa = 0, {}, time.monotonic()
    for jid in ordem_exec:
        job = por_id[jid]
        if (cx.pasta(jid) / "treino.json").exists():
            continue
        va = protocolo.get("versao_anterior")
        if va and adotar(cx, jid, job, va["pasta"]):
            continue
        if limite_jobs is not None and feitos >= limite_jobs:
            break
        dono = None
        if por_trava:
            if not serial.energia()["na_tomada"]:
                print(json.dumps({"estado": "aguarda_tomada"}), flush=True)
                return {"estado": "aguarda_tomada"}
            dono = serial.tomar([sys.executable, str(Path(__file__)), "job", jid], esperar_min, False)
            if dono is None:
                return {"estado": "trava_ocupada"}
        try:
            executar_job(cx, job, por_id, cache_anot)
        finally:
            if dono:
                serial.liberar(dono)
        feitos += 1
        if por_trava and time.monotonic() - ultima_pausa > PAUSA_A_CADA_S:
            # Quem espera a trava consulta a cada 30 s; sem esta folga a fila a retomaria sempre.
            time.sleep(PAUSA_S)
            ultima_pausa = time.monotonic()
    pendentes = [j for j in ordem_exec if not (cx.pasta(j) / "treino.json").exists()]
    estado = {"estado": "treino_concluido" if not pendentes else "parcial", "pendentes": len(pendentes),
              "jobs": len(ordem_exec), "linhagens": len(linhagens)}
    if not pendentes:
        ci.gravar_json(cx.saida / "treino_concluido.json", {"assinatura": cx.assinatura, "jobs": ordem_exec,
                                                           "linhagens": linhagens,
                                                           "codigo": {str(f): sha(f) for f in fontes()}})
    return estado


def avaliar(saida, protocolo, inv, particoes, por_trava=True, esperar_min=240.):
    """So depois de treino_concluido; mede reservas de retencao e avaliacao uma vez por checkpoint."""
    import executar_serializado as serial
    cx = Contexto(saida, protocolo, inv, particoes)
    concl = dc.ler(cx.saida / "treino_concluido.json")
    if concl["assinatura"] != cx.assinatura:
        raise ValueError("treino concluido de outro protocolo")
    for f, h in protocolo["codigo"].items():
        if sha(f) != h:
            raise ValueError(f"codigo mudou depois do pre-registro: {f}")
    materializar_cache(inv, Path(protocolo["campanha"]) / "cache")
    torch.set_num_threads(cx.cfg["threads"])
    reservas = [r for r in cx.tr if any(r["nome"] in d["retencao"] for d in protocolo["divisoes"].values())]
    alvos = reservas + cx.te
    por_id, _, _ = planejar(protocolo, cx.tr)
    pend = [j for j in concl["jobs"] if por_id[j]["tipo"] == "ajuste"
            and not (cx.pasta(j) / "avaliacao.json").exists()]
    for lote in [pend[i:i+20] for i in range(0, len(pend), 20)]:
        dono = serial.tomar([sys.executable, str(Path(__file__)), "avaliar"], esperar_min, False) if por_trava else None
        if por_trava and dono is None:
            return {"estado": "trava_ocupada"}
        try:
            for jid in lote:
                t = dc.ler(cx.pasta(jid) / "treino.json")
                e = torch.load(cx.pasta(jid) / "melhor.pt", weights_only=False, map_location="cpu")
                if ci.hash_completo_do_estado(e["modelo"]) != t["hash_estado"]:
                    raise ValueError("checkpoint divergente na avaliacao")
                modelo = ci.UNetPequena(canais_entrada=4, base=cx.cfg["base"])
                modelo.load_state_dict(e["modelo"])
                m = dc.medir(modelo, alvos, t["estat"], cx.cfg["modo"], origem="retencao_e_avaliacao_desenvolvimento")
                ci.gravar_json(cx.pasta(jid) / "avaliacao.json", m)
                print(json.dumps({"avaliado": jid}), flush=True)
        finally:
            if dono:
                serial.liberar(dono)
    ci.gravar_json(cx.saida / "avaliacao_concluida.json", {"assinatura": cx.assinatura, "n": len(concl["jobs"])})
    return {"estado": "avaliacao_concluida"}


def executar(saida=SAIDA, config=None, v1_dir=V1, somente_preparar=False, por_trava=True, limite_jobs=None,
             esperar_min=240.):
    saida = Path(saida).resolve()
    protocolo, inv, part = preparar(config, v1_dir)
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
    por_id, ordem_exec, linhagens = planejar(protocolo, part[0])
    if somente_preparar:
        ci.gravar_json(saida / "plano.json", {"jobs": len(ordem_exec), "linhagens": len(linhagens),
                                             "por_bloco": {b: sum(por_id[j]["bloco"] == b for j in ordem_exec)
                                                           for b in ("A1", "B", "C", "A2")}})
        return {"estado": "preparado", "assinatura": ci.hash_json(protocolo), "jobs": len(ordem_exec)}
    if not (saida / "treino_concluido.json").exists():
        e = fila(saida, protocolo, inv, part, por_trava, esperar_min, limite_jobs)
        if e["estado"] != "treino_concluido":
            return e
    if not (saida / "avaliacao_concluida.json").exists():
        e = avaliar(saida, protocolo, inv, part, por_trava, esperar_min)
        if e["estado"] != "avaliacao_concluida":
            return e
    return {"estado": "pronto_para_analise"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("acao", nargs="?", default="executar", choices=["executar", "preparar", "job", "avaliar"])
    ap.add_argument("resto", nargs="*")
    ap.add_argument("--saida", type=Path, default=SAIDA)
    ap.add_argument("--esperar-min", type=float, default=240.)
    a = ap.parse_args()
    if a.acao == "preparar":
        print(json.dumps(executar(a.saida, somente_preparar=True)))
    elif a.acao == "executar":
        print(json.dumps(executar(a.saida, esperar_min=a.esperar_min)))
    else:
        raise SystemExit("acoes job/avaliar sao rotulos da trava, nao entradas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
