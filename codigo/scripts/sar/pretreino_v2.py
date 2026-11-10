"""Pre-treino com normalizacao coerente, objetivo especifico de SAR e eficiencia de rotulos.

Versao `pretreino_sar_v2` (11/10/2026). Pergunta primaria pre-registrada: com
poucos rotulos (1 chip por contexto), um encoder pre-treinado sem rotulo
melhora a discriminacao (AP) sobre a inicializacao aleatoria, com ajuste,
passos, normalizacao e checkpoint iguais? Unidade = evento (5 reservas Sen1 + 3
de avaliacao = 8). Familia primaria: {MAE - aleatorio, MAE_speckle - aleatorio}
na fracao 1/contexto (Holm, k=2; p_min(8)=0,0078).

  - Normalizacao fixa sem dado (fontes_sar) no pre-treino E no ajuste: elimina a
    incompatibilidade do MAE v2 (estatistica do pote x do primeiro contexto).
  - Objetivos: MAE (75% em blocos de 8, reconstrucao do escondido; He et al. 2022)
    e MAE com speckle (perturbacao multiplicativa Gamma(L,1/L) na intensidade
    linear antes do dB, L em {1..4}; reconstrucao do sinal limpo; hipotese
    inspirada em SARMAE, CVPR 2026, sem reproduzir sua arquitetura).
  - Pote do pre-treino: imagens pos-evento de ajuste+tampao dos 5 contextos Sen1.
    Nunca reserva, validacao, avaliacao, GEOID ou reservas CEMS. Exposicao
    OFFLINE a todos os contextos de treino: nao e aprendizado online causal.
  - Fracoes pareadas: 1, 3 e 12 chips por contexto (prefixos da mesma lista).
  - Duracao: o estagio longo (4x passos) so roda se, na VALIDACAO Sen1 e fracao 1,
    algum objetivo superar o aleatorio em AP >= 0,02 na mediana das sementes.
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
from torch import nn

import calibracao_decisao as cd
import campanha_independencia_autonomia as ci
import continuidade_v3 as v3
import diagnostico_competencia as dc
import fontes_sar as fs
import nucleo_sar as nu
from dados_pares_geograficos import LIMITES_DB, materializar_cache, particao, sha
from pretreino_mascarado import DecoderLeve, EncoderUNet, mascarar, perda_mascarada

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/manutencao"))
VERSAO = "pretreino_sar_v2"
SAIDA = REPO / "local_runs/pretreino-sar/v2"
CONFIG = {"sementes": [7, 11, 23], "fracoes": [1, 3, 12], "passos_ajuste": 768, "passos_pretreino": 1920,
          "fator_longo": 4, "limiar_estagio_longo": .02, "batch": 16, "recorte": 64, "lr": 1e-3, "base": 4,
          "passos_por_epoca": 24, "threads": 2, "fracao_mascara": .75, "bloco": 8, "speckle_L": [1, 2, 3, 4],
          "margem_ap": .025, "alfa": .05}
OBJETIVOS = ("mae", "mae_speckle")
INICIOS = ("aleatorio",) + OBJETIVOS


def speckle(x_norm, v, rng):
    """Volta ao dB, aplica Gamma(L,1/L) na intensidade linear, renormaliza. So onde ha leitura."""
    out = x_norm.clone()
    for i in range(x_norm.shape[0]):
        L = int(rng.choice(CONFIG["speckle_L"]))
        g = torch.from_numpy(rng.gamma(L, 1/L, x_norm.shape[1:]).astype(np.float32))
        for b in range(x_norm.shape[1]):
            lo, hi = LIMITES_DB[b % 2]
            db = x_norm[i, b]*(hi-lo)/2 + (lo+hi)/2
            ruido = 10*torch.log10(torch.clamp(10**(db/10)*g[b], min=1e-12))
            out[i, b] = (torch.clamp(ruido, lo, hi) - (lo+hi)/2) / ((hi-lo)/2)
        out[i][:, ~v[i, 0]] = 0
    return out


def pretreinar(pasta, chips, objetivo, seed, passos, cfg):
    pasta = Path(pasta)
    if (pasta / "treino.json").exists():
        return dc.ler(pasta / "treino.json")
    pasta.mkdir(parents=True, exist_ok=True)
    ci.semear(seed)
    enc, dec = EncoderUNet(2, cfg["base"]), DecoderLeve(cfg["base"], 2)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=cfg["lr"])
    ger = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng([seed, 99])
    perdas, t0 = [], time.monotonic()
    for passo in range(passos):
        xs, vs = [], []
        for _ in range(cfg["batch"]):
            r = chips[int(rng.integers(len(chips)))]
            h, w = r["grade"]["forma"]
            a, b = int(rng.integers(h-cfg["recorte"]+1)), int(rng.integers(w-cfg["recorte"]+1))
            x, v = fs.entrada(r, "pos", (a, b, cfg["recorte"]))
            xs.append(torch.from_numpy(x[:2].copy()))
            vs.append(torch.from_numpy(v.copy()))
        x, v = torch.stack(xs), torch.stack(vs).unsqueeze(1)
        entrada = speckle(x, v, rng) if objetivo == "mae_speckle" else x
        mascarada, escondido = mascarar(entrada, cfg["fracao_mascara"], cfg["bloco"], ger)
        saida = dec(enc(mascarada)[2])
        alvo_rec = escondido if objetivo == "mae" else torch.ones_like(escondido)
        loss = perda_mascarada(saida, x, alvo_rec, v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        perdas.append(float(loss.detach()))
        if (passo+1) % 240 == 0:
            print(json.dumps({"pretreino": pasta.name, "passo": passo+1, "perda": round(float(np.mean(perdas[-240:])), 5)}),
                  flush=True)
    estado = {k: v.detach().clone() for k, v in enc.state_dict().items()}
    ci.gravar_peso(pasta / "encoder.pt", {"encoder": estado})
    r = {"objetivo": objetivo, "seed": seed, "passos": passos, "hash_encoder": ci.hash_completo_do_estado(estado),
         "perda_inicial": float(np.mean(perdas[:96])), "perda_final": float(np.mean(perdas[-96:])),
         "segundos": time.monotonic()-t0, "chips_pote": len(chips),
         "nao_e": "reconstrucao menor nao demonstra segmentacao melhor"}
    ci.gravar_json(pasta / "treino.json", r)
    return r


def preparar(config=None, v1_dir=v3.V1, preparo=v3.PREPARO):
    cfg = {**CONFIG, **(config or {})}
    p1 = dc.ler(Path(v1_dir) / "protocolo.json")
    campanha = Path(p1["campanha"])
    inv = dc.ler(campanha / "inventario.json")
    ev = dc.ler(campanha / "protocolo.json")["ordem_eventos"]
    tr, va, te = particao(inv["pares"], ev[:3], ev[3:5])
    pr = dc.ler(preparo)
    g = fs.chips_geoid(pr)
    pote = sorted(n for d in p1["divisoes"].values() for n in d["ajuste"] + d["tampao"])
    anot = {str(s): {e: p1["anotacoes"][str(s)][e] for e in p1["contextos_em_ordem"]} for s in cfg["sementes"]}
    grupos = {"validacao_checkpoint": sorted(r["nome"] for r in va),
              "calibracao": sorted(c["nome"] for c in g if c["evento"] == v3.VALIDACAO_NOVA),
              "controle": sorted(c["nome"] for c in g if c["evento"] == v3.CONTROLE),
              "retencao": sorted(n for d in p1["divisoes"].values() for n in d["retencao"]),
              "avaliacao": sorted(r["nome"] for r in te)}
    if set(pote) & set().union(*map(set, grupos.values())):
        raise ValueError("pote de pre-treino toca grupo de medida")
    p = {"versao": VERSAO, "v1_protocolo_sha256": sha(Path(v1_dir) / "protocolo.json"), "preparo_sha256": sha(preparo),
         "campanha": str(campanha), "config": cfg, "pote_pretreino": pote, "anotacoes": anot, "grupos": grupos,
         "contextos": p1["contextos_em_ordem"], "objetivos": list(OBJETIVOS), "inicios": list(INICIOS),
         "primaria": {"desfecho": "AP por evento do modelo ajustado", "unidades": "8 eventos Sen1 (5 retencao + 3 avaliacao)",
                      "contrastes": [["mae@1", "aleatorio@1"], ["mae_speckle@1", "aleatorio@1"]],
                      "correcao": "Holm, Wilcoxon exato", "piso_discreto": v3.piso_wilcoxon(8, 2),
                      "regra": "ganho/perda: p_holm<0,05 e IC95 t alem de margem; equivalencia por TOST 1-2alfa/k"},
         "secundaria": "fracoes 3 e 12, mae_speckle - mae, curva de eficiencia, custo amortizado, IoU calibrado",
         "estagio_longo": {"condicao": "validacao Sen1, fracao 1: mediana(AP_obj - AP_aleatorio) >= 0,02",
                           "acao": "pre-treino 4x passos, ajuste so na fracao 1, mesmas sementes"},
         "codigo": {str(f): sha(f) for f in fontes()}, "regime": "desenvolvimento; exposicao offline",
         "promovido": False}
    todos = {r["nome"]: {**r, "fonte": "sen1"} for r in tr + va + te}
    todos.update({c["nome"]: c for c in g})
    return p, inv, todos


def fontes():
    return [Path(__file__), Path(nu.__file__), Path(fs.__file__), Path(v3.__file__), Path(cd.__file__),
            REPO / "scripts/sar/pretreino_mascarado.py", REPO / "scripts/sar/modelo_unet.py"]


def jobs(p, longo=False, objetivo_longo=None):
    cfg, out = p["config"], []
    passos_pt = cfg["passos_pretreino"]*(cfg["fator_longo"] if longo else 1)
    objetivos = [objetivo_longo] if longo else OBJETIVOS
    for s in cfg["sementes"]:
        for o in objetivos:
            out.append({"tipo": "pretreino", "id": f"pt_{o}_s{s}_{passos_pt}", "objetivo": o, "seed": s, "passos": passos_pt})
    for f in ([1] if longo else cfg["fracoes"]):
        for s in cfg["sementes"]:
            for ini in ((objetivo_longo,) if longo else INICIOS):
                pt = None if ini == "aleatorio" else f"pt_{ini}_s{s}_{passos_pt}"
                out.append({"tipo": "ajuste", "id": f"aj_{ini}{'_longo' if longo else ''}_f{f}_s{s}", "inicio": ini,
                            "pretreino": pt, "fracao": f, "seed": s})
    return out


def rodar(saida, p, todos, j):
    cfg = p["config"]
    if j["tipo"] == "pretreino":
        return pretreinar(saida / "jobs" / j["id"], [todos[n] for n in p["pote_pretreino"]], j["objetivo"],
                          j["seed"], j["passos"], cfg)
    novas = [todos[n] for e in p["contextos"] for n in p["anotacoes"][str(j["seed"])][e][:j["fracao"]]]
    enc = None
    if j["pretreino"]:
        tp = dc.ler(saida / "jobs" / j["pretreino"] / "treino.json")
        e = torch.load(saida / "jobs" / j["pretreino"] / "encoder.pt", weights_only=False, map_location="cpu")
        if ci.hash_completo_do_estado(e["encoder"]) != tp["hash_encoder"]:
            raise ValueError("encoder divergente")
        enc = e["encoder"]
    job = {"id": j["id"], "seed": j["seed"], "etapa": 1, "passos": cfg["passos_ajuste"], "rho": .5, "lwf": None,
           "checkpoint": "validacao", "modo": "pos", "sal": 7}
    val = [todos[n] for n in p["grupos"]["validacao_checkpoint"]]
    return nu.ajustar(saida / "jobs" / j["id"], job, novas, [], val, {**cfg, "passos_por_epoca": cfg["passos_por_epoca"]},
                      encoder=enc)


def medir(saida, p, todos, jid):
    f = saida / "jobs" / jid / "avaliacao.json"
    if f.exists():
        return dc.ler(f)
    t = dc.ler(saida / "jobs" / jid / "treino.json")
    m = nu.novo_modelo(t["job"]["seed"], p["config"]["base"])
    m.load_state_dict(nu.estado(saida / "jobs" / jid, t)["estado"])
    r = {g: nu.medir(m, [todos[n] for n in nomes], "pos") for g, nomes in p["grupos"].items()}
    ci.gravar_json(f, r)
    return r


def ap_validacao(av):
    return float(np.mean([cd.ap(*cd.arrays(e)) for e in av["validacao_checkpoint"].values()]))


def executar(saida=SAIDA, config=None, por_trava=True, esperar_min=240., somente_preparar=False,
             v1_dir=v3.V1, preparo=v3.PREPARO):
    import executar_serializado as serial
    saida = Path(saida).resolve()
    p, inv, todos = preparar(config, v1_dir, preparo)
    saida.mkdir(parents=True, exist_ok=True)
    dc.imutavel(saida / "protocolo.json", p)
    (saida / "codigo").mkdir(exist_ok=True)
    for f, h in p["codigo"].items():
        d = saida / "codigo" / Path(f).name
        if not d.exists():
            shutil.copyfile(f, d)
        if sha(d) != h:
            raise ValueError("snapshot divergente; nova versao necessaria")
    if somente_preparar:
        return {"estado": "preparado", "jobs": len(jobs(p))}
    materializar_cache(inv, Path(p["campanha"]) / "cache")
    torch.set_num_threads(p["config"]["threads"])
    torch.use_deterministic_algorithms(True)

    def fila(lista):
        for j in lista:
            if (saida / "jobs" / j["id"] / "treino.json").exists():
                continue
            dono = serial.tomar([sys.executable, str(Path(__file__)), j["id"]], esperar_min, False) if por_trava else None
            if por_trava and dono is None:
                return False
            try:
                rodar(saida, p, todos, j)
            finally:
                if dono:
                    serial.liberar(dono)
        return True
    if not fila(jobs(p)):
        return {"estado": "trava_ocupada"}
    # Decisao do estagio longo: so validacao, antes de qualquer reserva ou avaliacao.
    decisao_f = saida / "decisao_estagio_longo.json"
    if not decisao_f.exists():
        val = {}
        for j in jobs(p):
            if j["tipo"] == "ajuste" and j["fracao"] == 1:
                t = dc.ler(saida / "jobs" / j["id"] / "treino.json")
                m = nu.novo_modelo(j["seed"], p["config"]["base"])
                m.load_state_dict(nu.estado(saida / "jobs" / j["id"], t)["estado"])
                val[j["id"]] = ap_validacao({"validacao_checkpoint": nu.medir(
                    m, [todos[n] for n in p["grupos"]["validacao_checkpoint"]], "pos")})
        ganhos = {o: float(np.median([val[f"aj_{o}_f1_s{s}"] - val[f"aj_aleatorio_f1_s{s}"] for s in p["config"]["sementes"]]))
                  for o in OBJETIVOS}
        melhor = max(ganhos, key=ganhos.get)
        ci.gravar_json(decisao_f, {"ap_validacao": val, "ganho_mediano": ganhos,
                                   "roda_longo": ganhos[melhor] >= p["config"]["limiar_estagio_longo"],
                                   "objetivo": melhor, "fonte": "somente validacao Sen1"})
    dec = dc.ler(decisao_f)
    lista = jobs(p) + (jobs(p, True, dec["objetivo"]) if dec["roda_longo"] else [])
    if dec["roda_longo"] and not fila(jobs(p, True, dec["objetivo"])):
        return {"estado": "trava_ocupada"}
    for j in lista:
        if j["tipo"] == "ajuste":
            medir(saida, p, todos, j["id"])
    ci.gravar_json(saida / "concluido.json", {"jobs": [j["id"] for j in lista], "estagio_longo": dec["roda_longo"]})
    return {"estado": "concluido", "estagio_longo": dec["roda_longo"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("acao", choices=["preparar", "executar"])
    a = ap.parse_args()
    print(json.dumps(executar(somente_preparar=a.acao == "preparar")))
