"""Ablacao temporal SAR com U-Net, controle de capacidade e teste separado.

O alvo e inundacao nova na observacao do evento. Sem identificadores de
evento/georreferencia, a avaliacao e exploratoria e nao promove modelos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "sar"))
sys.path.insert(0, str(REPO / "scripts" / "neural"))
from baseline_otsu import contagens, metricas
from dados_pre_pos import (entrada, inventariar, ler_imagem, ler_par,
                          materializar_cache, normalizacao)
from modelo_unet import UNetPequena, contar_parametros
from proveniencia_de_peso import commit_atual, hash_completo_do_estado, semear
from regua import analisar_pareado

MODOS = ("pos", "pre_pos", "pos_delta")


def hash_json(objeto: dict) -> str:
    return hashlib.sha256(json.dumps(objeto, sort_keys=True).encode()).hexdigest()


def gravar_json(caminho: Path, valor: dict) -> None:
    parcial = caminho.with_suffix(".temporario")
    parcial.write_text(json.dumps(valor, indent=2, ensure_ascii=False,
                                  allow_nan=False), encoding="utf-8")
    os.replace(parcial, caminho)


def gravar_peso(caminho: Path, valor: dict) -> None:
    parcial = caminho.with_suffix(".temporario")
    torch.save(valor, parcial)
    os.replace(parcial, caminho)


def lotes(linhas, raiz, estat, modo, tamanho, recorte=None, seed=0):
    """O mesmo sorteio/recorte/espelhamento para todos os bracos pareados."""
    gerador = np.random.default_rng(seed)
    ordem = gerador.permutation(len(linhas)) if recorte else np.arange(len(linhas))
    xs, ys, nomes = [], [], []
    for i in ordem:
        r = linhas[i]
        a, b, y = ler_par(raiz, r["nome"])
        x = entrada(a, b, estat, modo)
        if recorte:
            h, w = y.shape
            if recorte > min(h, w):
                raise ValueError("recorte maior que a imagem")
            t = int(gerador.integers(h - recorte + 1))
            e = int(gerador.integers(w - recorte + 1))
            x, y = x[:, t:t + recorte, e:e + recorte], y[t:t + recorte, e:e + recorte]
            if gerador.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if gerador.random() < 0.5:
                x, y = x[:, ::-1], y[::-1]
        xs.append(torch.from_numpy(x.copy()))
        ys.append(torch.from_numpy(y.copy()))
        nomes.append(r["nome"])
        if len(xs) == tamanho:
            yield torch.stack(xs), torch.stack(ys), nomes
            xs, ys, nomes = [], [], []
    if xs:
        yield torch.stack(xs), torch.stack(ys), nomes


@torch.no_grad()
def avaliar(modelo, linhas, raiz, estat, modo, batch=4):
    """Chip inteiro; contagens sao as mesmas da frente A ja existente."""
    modelo.eval()
    soma = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
    perdas, pixels, chips = 0.0, 0, []
    for x, y, nomes in lotes(linhas, raiz, estat, modo, batch):
        if x.shape[-1] % 8 or x.shape[-2] % 8:
            raise ValueError("grade deve ser multipla de oito")
        logits = modelo(x)
        perdas += float(nn.functional.cross_entropy(logits, y, reduction="sum"))
        pixels += y.numel()
        for nome, p, v in zip(nomes, logits.argmax(1).numpy(), y.numpy()):
            c = contagens(p == 1, v == 1, np.ones(v.shape, dtype=bool))
            for k in soma:
                soma[k] += c[k]
            chips.append({"nome": nome, "contagens": c, **metricas(c)})
    return {"perda": perdas / pixels, "contagens": soma,
            "micro": metricas(soma), "por_chip": chips,
            "macro_iou": media([c["iou"] for c in chips]),
            "chips_iou_indefinida": sum(c["iou"] is None for c in chips)}


def media(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def baseline_mudanca(linhas, raiz, limiar):
    soma = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
    chips = []
    for r in linhas:
        a, b, y = ler_par(raiz, r["nome"])
        d = np.abs(b.astype(float) - a.astype(float)).mean(2) / 255
        c = contagens(d > limiar, y == 1, np.ones(y.shape, dtype=bool))
        for k in soma:
            soma[k] += c[k]
        chips.append({"nome": r["nome"], **metricas(c)})
    return {"limiar": limiar, "micro": metricas(soma), "contagens": soma,
            "macro_iou": media([c["iou"] for c in chips])}


def treinar(raiz, partes, estat, config, modo, seed, saida, assinatura):
    semear(seed)
    modelo = UNetPequena(canais_entrada=6, base=config["base"])
    otimizador = torch.optim.Adam(modelo.parameters(), lr=config["lr"])
    arquivo = saida / f"{modo}_s{seed}.pt"
    ultimo = saida / f"{modo}_s{seed}_retomada.pt"
    inicio, historico, melhor = 0, [], float("inf")
    melhor_estado, melhor_epoca = None, None
    if ultimo.exists():
        estado = torch.load(ultimo, map_location="cpu", weights_only=False)
        if estado["assinatura"] != assinatura:
            raise ValueError("retomada de outra configuracao")
        modelo.load_state_dict(estado["modelo"])
        otimizador.load_state_dict(estado["otimizador"])
        torch.set_rng_state(estado["rng"])
        inicio, historico = estado["epoca"], estado["historico"]
        melhor, melhor_estado = estado["melhor"], estado["melhor_estado"]
        melhor_epoca = estado["melhor_epoca"]
    for epoca in range(inicio, config["epocas"]):
        t0 = time.monotonic()
        modelo.train()
        perda, passos = 0.0, 0
        for x, y, _ in lotes(partes["treino"], raiz, estat, modo, config["batch"],
                             config["recorte"], seed * 10000 + epoca):
            otimizador.zero_grad(set_to_none=True)
            valor = nn.functional.cross_entropy(modelo(x), y)
            if not torch.isfinite(valor):
                raise ValueError("perda nao finita: nao continuar")
            valor.backward()
            otimizador.step()
            perda += float(valor.detach())
            passos += 1
        validacao = avaliar(modelo, partes["validacao"], raiz, estat, modo)
        if validacao["perda"] < melhor:
            melhor = validacao["perda"]
            melhor_epoca = epoca + 1
            melhor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        linha = {"epoca": epoca + 1, "passos": passos, "perda_treino": perda / passos,
                 "perda_validacao": validacao["perda"],
                 "iou_validacao": validacao["micro"]["iou"],
                 "segundos": time.monotonic() - t0}
        historico.append(linha)
        gravar_peso(ultimo, {"assinatura": assinatura, "modelo": modelo.state_dict(),
                            "otimizador": otimizador.state_dict(), "rng": torch.get_rng_state(),
                            "epoca": epoca + 1, "historico": historico, "melhor": melhor,
                            "melhor_estado": melhor_estado, "melhor_epoca": melhor_epoca})
        print(json.dumps({"modo": modo, "seed": seed, **linha}), flush=True)
    modelo.load_state_dict(melhor_estado)
    gravar_peso(arquivo, {"modelo": melhor_estado, "estat": estat, "modo": modo,
                         "base": config["base"], "epoca": melhor_epoca,
                         "assinatura": assinatura,
                         "hash_estado": hash_completo_do_estado(melhor_estado)})
    return modelo, {"seed": seed, "modo": modo, "historico": historico,
                    "melhor_epoca": melhor_epoca, "parametros": contar_parametros(modelo),
                    "peso": str(arquivo), "hash_estado": hash_completo_do_estado(melhor_estado)}


def campanha(raiz: Path, saida: Path, config: dict, mapa: Path | None = None):
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    saida.mkdir(parents=True, exist_ok=True)
    conteudo = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), REPO / "scripts/sar/dados_pre_pos.py",
                          REPO / "scripts/sar/modelo_unet.py",
                          REPO / "scripts/sar/baseline_otsu.py",
                          REPO / "scripts/neural/regua.py",
                          REPO / "scripts/neural/proveniencia_de_peso.py")}
    inv_path = saida / "inventario.json"
    # O inventario e refeito para detectar mudanca nos bytes de qualquer par.
    inv = inventariar(raiz, mapa, progresso=lambda n, total: print(
        json.dumps({"inventariados": n, "total": total}), flush=True))
    materializar_cache(inv, saida / "cache")
    print("Cache de pixels conferido; calculando normalizacao do treino.", flush=True)
    estat = normalizacao(inv["raiz"], inv["pares"])
    protocolo = {"config": config, "dados": hash_json(inv), "normalizacao": estat,
                 "codigo": conteudo, "comparacao": "pre_pos - pos; pos_delta secundaria",
                 "relevancia_iou": 0.025, "selecao": "menor CE na validacao",
                 "capacidade": "mesma UNet de seis canais; pos duplica a data B",
                 "promocao_automatica": False, "regime": "exploratorio",
                 "commit": commit_atual(), "torch": torch.__version__}
    assinatura = hash_json(protocolo)
    protocolo_path = saida / "protocolo.json"
    if protocolo_path.exists():
        anterior = json.loads(protocolo_path.read_text(encoding="utf-8"))
        if anterior != protocolo:
            raise ValueError("rodada imutavel: nova configuracao exige nova pasta")
    else:
        gravar_json(protocolo_path, protocolo)
        gravar_json(inv_path, inv)
    partes = {s: [r for r in inv["pares"] if r["split"] == s]
              for s in ("treino", "validacao", "teste")}
    print(json.dumps({"particoes": {s: len(rs) for s, rs in partes.items()},
                      "assinatura": assinatura}), flush=True)
    # O comparador simples usa SOMENTE validacao para escolher o limiar.
    limiares = [0.025, 0.05, 0.10, 0.20, 0.30]
    bas_path = saida / "baseline.json"
    if bas_path.exists():
        bas = json.loads(bas_path.read_text(encoding="utf-8"))
    else:
        candidatos = [baseline_mudanca(partes["validacao"], inv["raiz"], t) for t in limiares]
        limiar = max(candidatos, key=lambda r: r["micro"]["iou"] or 0)["limiar"]
        bas = {"validacao": candidatos,
               "teste": baseline_mudanca(partes["teste"], inv["raiz"], limiar)}
        gravar_json(bas_path, bas)
    resultados = []
    for seed in config["sementes"]:
        for modo in MODOS:
            resultado_path = saida / f"{modo}_s{seed}.json"
            if resultado_path.exists():
                resultados.append(json.loads(resultado_path.read_text(encoding="utf-8")))
                continue
            modelo, resultado = treinar(inv["raiz"], partes, estat, config, modo,
                                        seed, saida, assinatura)
            # O teste so e medido depois de escolher a epoca, sem selecao por ele.
            resultado["teste"] = avaliar(modelo, partes["teste"], inv["raiz"], estat, modo)
            gravar_json(resultado_path, resultado)
            resultados.append(resultado)
    por_modo = {m: [r for r in resultados if r["modo"] == m] for m in MODOS}
    difs = [por_modo["pre_pos"][i]["teste"]["micro"]["iou"] -
            por_modo["pos"][i]["teste"]["micro"]["iou"] for i in range(len(config["sementes"]))]
    analise = (analisar_pareado(difs, relevancia=0.025) if np.std(difs) > 0 else
               {"n": len(difs), "diferencas": difs, "veredito": "INDECIDIVEL",
                "por_que": "sem variabilidade estimavel entre sementes"})
    card = {"assinatura": assinatura, "regime": "exploratorio",
            "particoes": {s: len(rs) for s, rs in partes.items()},
            "resultados": resultados, "baseline": bas,
            "analise_por_semente": analise,
            "estado_cientifico": "INDECIDIVEL_PARA_GENERALIZACAO",
            "motivo": "sem independencia por evento/espaco verificada; sementes nao substituem eventos",
            "promovido": False, "altera_escore_suscetibilidade": False}
    gravar_json(saida / "card.json", card)
    relatorio(card, saida)
    return card


def relatorio(card, saida):
    linhas = ["# SAR pre/pos: relatorio da rodada", "",
              "Regime exploratorio. Nenhum peso promovido; nenhum escore fisico alterado.",
              "Alvo: inundacao nova anotada pelo S1GFloods, nao previsao futura.",
              "", f"Particoes: `{card['particoes']}`. Assinatura: `{card['assinatura']}`.", "",
              "| Entrada | Semente | Epoca pela validacao | IoU micro teste | IoU macro |",
              "|---|---:|---:|---:|---:|"]
    for r in card["resultados"]:
        linhas.append(f"| {r['modo']} | {r['seed']} | {r['melhor_epoca']} | "
                      f"{r['teste']['micro']['iou']:.6f} | {r['teste']['macro_iou']:.6f} |")
    linhas += ["", f"Baseline de mudanca RGB, limiar pela validacao: `{card['baseline']['teste']}`.",
               "", f"Comparacao pareada por semente: `{card['analise_por_semente']}`.", "",
               "## Limites e uso", "",
               "PNG sem datas, orbitas, identificadores de evento ou georreferencia. A ordem A/B "
               "vem da publicacao; nao foi verificada por metadados orbitais. Copias exatas "
               "ficam juntas, mas proximidade espacial e repeticao de evento nao sao verificaveis.",
               "A particao por hash e tecnica, nao o benchmark oficial nem validacao externa.",
               "Normalizacao somente no treino; recortes somente no treino; avaliacao em chips "
               "inteiros. Tres entradas usam a mesma capacidade e numero de passos.",
               "O controle pos duplica B; pre_pos usa B/A; pos_delta usa B/(B-A). A diferenca "
               "e de canais renderizados, nao razao fisica de retrodifusao.",
               "Pixels 0/255 seguem a anotacao publicada. Nao ha mascara independente de "
               "incerteza/agua permanente; o fundo nao autoriza negativo do corpus territorial.",
               "Sementes medem variabilidade do treino, nao incerteza de generalizacao entre eventos. "
               "Nao comparar diretamente esta IoU com a frente A de agua no Sen1Floods11.",
               "Os pesos antigos de dois canais nao foram adaptados para RGB: essa mudanca "
               "nao teria equivalencia fisica demonstrada. Nenhum ganho de pre-treino e inferido.",
               "O candidato so pode seguir como observacao experimental. Para promocao: metadados "
               "temporais/geometricos, eventos e territorios independentes, relevancia predefinida "
               "e teste protegido sem escolha retrospectiva do vencedor.", "",
               "Fonte: https://github.com/Tamer-Saleh/S1GFlood-Detection", ""]
    (saida / "relatorio.md").write_text("\n".join(linhas), encoding="utf-8")


@torch.no_grad()
def inferir(peso: Path, antes: Path, depois: Path, saida: Path):
    """Inferencia experimental so no dominio PNG RGB do checkpoint."""
    estado = torch.load(peso, map_location="cpu", weights_only=False)
    if estado["hash_estado"] != hash_completo_do_estado(estado["modelo"]):
        raise ValueError("peso modificado")
    a, b = ler_imagem(antes), ler_imagem(depois)
    if a.shape != b.shape or a.shape[0] % 8 or a.shape[1] % 8:
        raise ValueError("par desalinhado ou dimensao incompativel")
    modelo = UNetPequena(canais_entrada=6, base=estado["base"])
    modelo.load_state_dict(estado["modelo"])
    modelo.eval()
    x = torch.from_numpy(entrada(a, b, estado["estat"], estado["modo"]))
    p = modelo(x.unsqueeze(0)).softmax(1)[0, 1].numpy()
    saida.mkdir(parents=True, exist_ok=False)
    np.save(saida / "probabilidade_modelo.npy", p)
    Image.fromarray(((p > 0.5) * 255).astype(np.uint8)).save(saida / "mascara.png")
    gravar_json(saida / "manifesto.json", {"assinatura": estado["assinatura"],
                "hash_estado": estado["hash_estado"], "modo": estado["modo"],
                "antes_sha256": hashlib.sha256(antes.read_bytes()).hexdigest(),
                "depois_sha256": hashlib.sha256(depois.read_bytes()).hexdigest(),
                "uso": "observacao_experimental_nao_admitida",
                "probabilidade_calibrada": False, "negativo_autorizado": False,
                "altera_escore_suscetibilidade": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raiz", type=Path, default=REPO / "local_runs/sar-pre-pos/dados/pares")
    p.add_argument("--saida", type=Path, required=True)
    p.add_argument("--mapa-eventos", type=Path)
    p.add_argument("--epocas", type=int, default=8)
    p.add_argument("--sementes", type=int, nargs="+", default=[7, 11, 23])
    p.add_argument("--base", type=int, default=4)
    p.add_argument("--recorte", type=int, default=64)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--peso", type=Path)
    p.add_argument("--antes", type=Path)
    p.add_argument("--depois", type=Path)
    args = p.parse_args()
    if args.peso:
        if not args.antes or not args.depois:
            p.error("inferir exige --antes e --depois")
        inferir(args.peso, args.antes, args.depois, args.saida)
    else:
        if (min(args.epocas, args.base, args.recorte, args.batch, args.threads) < 1
                or args.recorte % 8 or len(set(args.sementes)) != len(args.sementes)):
            p.error("parametros invalidos ou sementes repetidas")
        cfg = {k: getattr(args, k) for k in
               ("epocas", "sementes", "base", "recorte", "batch", "threads", "lr")}
        campanha(args.raiz, args.saida, cfg, args.mapa_eventos)
