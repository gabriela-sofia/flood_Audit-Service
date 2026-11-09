"""Diagnostica a classe agua antes de liberar aquisicao de exemplos.

So usa treino e validacao de desenvolvimento. Os testes anteriores e o
escore fisico ficam intactos. Competencia minima nao e ganho cientifico.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import campanha_independencia_autonomia as ci
from dados_pares_geograficos import entrada, imagens, materializar_cache, normalizacao, particao, sha

LIMIARES = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
PISO = {"iou": .1, "recall": .1, "precisao": .1}
BRACOS = {
    "controle": {"objetivo": "ce", "amostragem": "uniforme", "multiplicador": 1},
    "ce_ponderada": {"objetivo": "ce_ponderada", "amostragem": "uniforme", "multiplicador": 1},
    "recorte_positivo": {"objetivo": "ce", "amostragem": "metade_positiva", "multiplicador": 1},
    "mais_passos": {"objetivo": "ce", "amostragem": "uniforme", "multiplicador": 4},
}


def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def imutavel(p, valor):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and ler(p) != valor:
        raise ValueError(f"artefato imutavel divergente: {p}")
    ci.gravar_json(p, valor)


def suporte(linhas):
    eventos = {}
    positivos = {}
    for r in linhas:
        _, sensor = imagens(r["caminhos"]["antes"], r["caminhos"]["depois"])
        y, _, mascara = ci.alvo(r, sensor)
        coords = np.argwhere(y == 1)
        if len(coords):
            positivos[r["nome"]] = coords
        e = eventos.setdefault(r["evento"], {"chips": 0, "positivo": 0, "negativo": 0, "ignorado": 0})
        e["chips"] += 1
        e["positivo"] += int((y == 1).sum())
        e["negativo"] += int((y == 0).sum())
        e["ignorado"] += int((~mascara).sum())
    return eventos, positivos


def pesos_classe(eventos):
    n = np.array([sum(e[k] for e in eventos.values()) for k in ("negativo", "positivo")], dtype=float)
    if (n <= 0).any():
        raise ValueError("treino precisa das duas classes observadas")
    # Raiz do inverso, limitada: mudar a perda nao cria observacoes novas.
    w = np.minimum(np.sqrt(n.max() / n), 10.)
    return (w / w.mean()).tolist()


def recorte(linhas, estat, modo, rng, tamanho, positivos, amostragem, indice):
    ancorar = amostragem == "metade_positiva" and indice % 2 == 0
    candidatas = [r for r in linhas if r["nome"] in positivos] if ancorar else linhas
    if not candidatas:
        raise ValueError("amostragem positiva sem positivo admitido")
    for _ in range(100):
        r = candidatas[int(rng.integers(len(candidatas)))]
        h, w = r["grade"]["forma"]
        if tamanho > min(h, w) or tamanho % 8:
            raise ValueError("recorte invalido para a grade")
        if ancorar:
            coords = positivos[r["nome"]]
            a, b = coords[int(rng.integers(len(coords)))]
            a = int(rng.integers(max(0, a-tamanho+1), min(a, h-tamanho)+1))
            b = int(rng.integers(max(0, b-tamanho+1), min(b, w-tamanho)+1))
        else:
            a, b = int(rng.integers(h-tamanho+1)), int(rng.integers(w-tamanho+1))
        janela = (a, b, tamanho)
        x, sensor = entrada(r, estat, modo, janela)
        y, _, _ = ci.alvo(r, sensor, janela)
        if (y != -1).any():
            if rng.random() < .5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if rng.random() < .5:
                x, y = x[:, ::-1], y[::-1]
            return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
    raise ValueError("nenhum recorte valido; passos nao podem desaparecer")


@torch.no_grad()
def medir(modelo, linhas, estat, modo, bins=1000, origem="validacao"):
    """Curvas compactas por evento, sem guardar milhoes de probabilidades."""
    modelo.eval()
    eventos = {}
    loss, validos = 0., 0
    for r in linhas:
        x, sensor = entrada(r, estat, modo)
        y, _, mascara = ci.alvo(r, sensor)
        logits = modelo(torch.from_numpy(x).unsqueeze(0))
        loss += float(nn.functional.cross_entropy(logits, torch.from_numpy(y).unsqueeze(0),
                                                  ignore_index=-1, reduction="sum"))
        validos += int(mascara.sum())
        p = logits.softmax(1)[0, 1].numpy()
        e = eventos.setdefault(r["evento"], {"positivo": np.zeros(bins, np.int64),
                                             "negativo": np.zeros(bins, np.int64),
                                             "contagens": {str(t): dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
                                                           for t in LIMIARES}})
        idx = np.minimum((p * bins).astype(int), bins-1)
        for k, classe in (("positivo", 1), ("negativo", 0)):
            e[k] += np.bincount(idx[mascara & (y == classe)], minlength=bins)
        for t in LIMIARES:
            c = ci.contagens(p >= t, y == 1, mascara)
            for k, v in c.items():
                e["contagens"][str(t)][k] += v
    if not validos:
        raise ValueError("validacao sem suporte")
    saida = {}
    for nome, e in eventos.items():
        pos, neg = e["positivo"], e["negativo"]
        tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
        precisao = tp / np.maximum(tp+fp, 1)
        ap = float((precisao * pos[::-1]).sum() / pos.sum()) if pos.sum() else None
        saida[nome] = {"positivos": int(pos.sum()), "negativos": int(neg.sum()),
                       "ap_aproximada_histograma": ap,
                       "prevalencia": float(pos.sum()/(pos.sum()+neg.sum())),
                       "histograma": {"positivo": pos.tolist(), "negativo": neg.tolist()},
                       "limiares": {t: {"contagens": c, **ci.metricas(c)} for t, c in e["contagens"].items()}}
    return {"por_evento": saida, "perda": loss/validos, "bins": bins,
            "origem": origem, "pixels_validos": validos}


def resumo_limiar(medida, limiar):
    eventos = medida["por_evento"]
    valores = [e["limiares"][str(limiar)]["iou"] for e in eventos.values()]
    motivos = []
    for nome, e in eventos.items():
        m = e["limiares"][str(limiar)]
        if e["positivos"] == 0 or e["negativos"] == 0:
            motivos.append(f"{nome}: falta uma classe na validacao")
        for k, minimo in PISO.items():
            if m.get(k) is None or m[k] < minimo:
                motivos.append(f"{nome}: {k} abaixo do piso operacional {minimo}")
    return {"macro_iou": float(np.mean([v for v in valores if v is not None])) if any(v is not None for v in valores) else None,
            "competente": not motivos, "motivos": motivos}


def treinar(linhas, validacao, estat, seed, cfg, braco, pesos, pasta, assinatura):
    """Treino retomavel, com checkpoint por CE de validacao em todos os bracos."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    job = ci.hash_json({"assinatura": assinatura, "linhas": [r["nome"] for r in linhas],
                        "validacao": [r["nome"] for r in validacao], "estat": estat,
                        "seed": seed, "cfg": cfg, "braco": braco, "pesos": pesos})
    ci.semear(seed)
    modelo = ci.UNetPequena(canais_entrada=4, base=cfg["base"])
    if (pasta / "treino.json").exists():
        r = ler(pasta / "treino.json")
        e = torch.load(pasta / "melhor.pt", weights_only=False, map_location="cpu")
        if r["id_job"] != job or e["id_job"] != job or ci.hash_completo_do_estado(e["modelo"]) != r["hash_estado"]:
            raise ValueError("peso ou job divergente")
        modelo.load_state_dict(e["modelo"])
        return modelo, r
    _, positivos = suporte(linhas)
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"])
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
    weight = torch.tensor(pesos, dtype=torch.float32) if braco["objetivo"] == "ce_ponderada" else None
    for epoca in range(inicio, cfg["epocas"] * braco["multiplicador"]):
        modelo.train()
        rng = np.random.default_rng(seed*10000+epoca)
        perdas, contagem, t0 = [], {"positivo": 0, "negativo": 0, "ignorado": 0}, time.monotonic()
        for _ in range(cfg["passos_por_epoca"]):
            lote = [recorte(linhas, estat, cfg["modo"], rng, cfg["recorte"], positivos,
                            braco["amostragem"], i) for i in range(cfg["batch"])]
            x, y = torch.stack([a for a, _ in lote]), torch.stack([b for _, b in lote])
            for k, valor in (("positivo", 1), ("negativo", 0), ("ignorado", -1)):
                contagem[k] += int((y == valor).sum())
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(modelo(x), y, weight=weight, ignore_index=-1)
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
        h = {"epoca": epoca+1, "passos": cfg["passos_por_epoca"], "pixels_apresentados": contagem,
             "perda_treino": float(np.mean(perdas)), "perda_validacao": va["perda"],
             "iou_validacao": va["macro_evento_iou"], "segundos": time.monotonic()-t0}
        historico.append(h)
        ci.gravar_peso(retomada, {"modelo": modelo.state_dict(), "opt": opt.state_dict(), "rng": torch.get_rng_state(),
                        "id_job": job, "epoca": epoca+1, "melhor": melhor, "melhor_epoca": melhor_epoca, "historico": historico})
        print(json.dumps({"job": pasta.name, **h}), flush=True)
    e = torch.load(pasta / "melhor.pt", weights_only=False, map_location="cpu")
    modelo.load_state_dict(e["modelo"])
    r = {"id_job": job, "hash_estado": ci.hash_completo_do_estado(e["modelo"]), "seed": seed,
         "melhor_epoca": melhor_epoca, "historico": historico, "braco": braco,
         "passos_totais": sum(h["passos"] for h in historico), "checkpoint": "menor CE de validacao"}
    ci.gravar_json(pasta / "treino.json", r)
    return modelo, r


def selecionar(resultados, sementes):
    """Uma regra comum a todas as sementes; piso impede escolher colapso de fundo."""
    candidatos = []
    for braco in BRACOS:
        rs = [r for r in resultados if r["braco_nome"] == braco]
        if sorted(r["seed"] for r in rs) != sorted(sementes):
            raise ValueError("braco incompleto ou semente duplicada")
        for t in LIMIARES:
            resumos = [resumo_limiar(r["validacao"], t) for r in rs]
            candidatos.append({"braco": braco, "limiar": t,
                               "competente": all(r["competente"] for r in resumos),
                               "mediana_macro_iou": float(np.median([r["macro_iou"] or 0 for r in resumos])),
                               "motivos": [m for r in resumos for m in r["motivos"]]})
    aptos = [r for r in candidatos if r["competente"]]
    ordem = {k: i for i, k in enumerate(BRACOS)}
    aptos.sort(key=lambda r: (-r["mediana_macro_iou"], ordem[r["braco"]], abs(r["limiar"]-.5)))
    return {"liberada": bool(aptos), "escolhido": aptos[0] if aptos else None, "candidatos": candidatos,
            "criterio": "todas as sementes e eventos de validacao acima do piso; depois mediana IoU",
            "nao_e": "nao prova ganho, calibracao, competencia universal ou promocao"}


def executar(campanha, saida, configuracao=None, somente_preparar=False):
    campanha, saida = Path(campanha).resolve(), Path(saida).resolve()
    p, inv = ler(campanha / "protocolo.json"), ler(campanha / "inventario.json")
    if ci.hash_json(inv) != p["inventario"]:
        raise ValueError("inventario divergente")
    cfg = {**p["config"], "modo": "pos", **(configuracao or {})}
    eventos = p["ordem_eventos"]
    tr, va, _ = particao(inv["pares"], eventos[:3], eventos[3:5])
    fontes = [Path(__file__), Path(ci.__file__), ci.REPO / "scripts/sar/dados_pares_geograficos.py",
              ci.REPO / "scripts/sar/modelo_unet.py", ci.REPO / "scripts/neural/proveniencia_de_peso.py",
              ci.REPO / "scripts/sar/baseline_otsu.py", ci.REPO / "scripts/sar/experimento_pre_pos.py"]
    protocolo = {"campanha": str(campanha), "inventario": ci.hash_json(inv), "config": cfg,
                 "codigo": {str(f): sha(f) for f in fontes}, "bracos": BRACOS, "piso": PISO,
                 "limiares": LIMIARES, "teste": "nao inferido nem usado para escolher",
                 "validacao": eventos[3:5], "regime": "desenvolvimento em dados conhecidos",
                 "promovido": False, "limite": "uma reserva de validacao; pisos so operacionais"}
    saida.mkdir(parents=True, exist_ok=True)
    imutavel(saida / "protocolo.json", protocolo)
    codigo = saida / "codigo"
    codigo.mkdir(exist_ok=True)
    for f in fontes:
        destino = codigo / f.name
        if not destino.exists():
            shutil.copyfile(f, destino)
        if sha(destino) != protocolo["codigo"][str(f)]:
            raise ValueError("snapshot divergente")
    assinatura = ci.hash_json(protocolo)
    if somente_preparar:
        r = {"estado": "preparado", "assinatura": assinatura, "ajustes_planejados": len(BRACOS)*len(cfg["sementes"]),
             "teste_avaliado": False, "treinos_executados": 0}
        ci.gravar_json(saida / "preparacao.json", r)
        return r
    materializar_cache(inv, campanha / "cache")
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    resultados = []
    for seed in cfg["sementes"]:
        nomes = ci.escolher(tr, "aleatoria", cfg["lote_anotacao"], seed)
        base = [r for r in tr if r["nome"] in nomes]
        estat = normalizacao(base)
        sp, _ = suporte(base)
        sv, _ = suporte(va)
        pesos = pesos_classe(sp)
        for nome, braco in BRACOS.items():
            pasta = saida / f"{nome}_s{seed}"
            modelo, treino = treinar(base, va, estat, seed, cfg, braco, pesos, pasta, assinatura)
            r = {**treino, "braco_nome": nome, "base": nomes, "estat": estat, "pesos_classe": pesos,
                 "suporte_treino": sp, "suporte_validacao": sv,
                 "validacao": medir(modelo, va, estat, cfg["modo"])}
            ci.gravar_json(pasta / "diagnostico.json", r)
            resultados.append(r)
    card = {"assinatura": assinatura, "protocolo": protocolo, "resultados": resultados,
            "aquisicao": selecionar(resultados, cfg["sementes"]), "promovido": False,
            "altera_escore_suscetibilidade": False, "teste_avaliado": False}
    ci.gravar_json(saida / "card.json", card)
    linhas = ["# Diagnostico de competencia SAR", "", "Desenvolvimento; somente treino e validacao.", "",
              "| braco | semente | passos | IoU macro a 0,5 | competencia a 0,5 |", "|---|---:|---:|---:|---|"]
    for r in resultados:
        s = resumo_limiar(r["validacao"], .5)
        linhas.append(f"| {r['braco_nome']} | {r['seed']} | {r['passos_totais']} | {s['macro_iou']} | {s['competente']} |")
    linhas += ["", f"Aquisicao liberada: **{card['aquisicao']['liberada']}**.",
               f"Escolha pela validacao: `{card['aquisicao']['escolhido']}`.", "",
               "Piso de 0,1 em IoU/precisao/recall por evento e semente: apenas operacional, nao validacao cientifica.",
               "AP e distribuicoes dos escores em diagnostico.json; AP aproximada por 1000 intervalos.",
               "Mais passos muda computacao; nao e ganho a orcamento igual.",
               "Limiares mudam a decisao, nao pesos nem calibracao. Nenhum teste foi reaberto."]
    (saida / "relatorio.md").write_text("\n".join(linhas)+"\n", encoding="utf-8")
    return card


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("campanha", type=Path)
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--preparar", action="store_true")
    a = ap.parse_args()
    executar(a.campanha, a.saida, somente_preparar=a.preparar)
