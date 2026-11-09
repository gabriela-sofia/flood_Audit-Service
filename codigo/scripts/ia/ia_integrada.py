"""Ciclo local da IA única: exportar, consultar, avaliar e continuar com replay.

Usa apenas arquivos já existentes. Não abre reservas nem adquire dados. Pesos
legados são importados uma vez, conferidos e convertidos para um state_dict
único, carregável com weights_only=True. Versões novas nunca sobrescrevem pais.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from modelo_integrado import FEATURES, IAIntegrada, REPO, VERSAO
sys.path.insert(0, str(REPO / "scripts/metodo"))
sys.path.insert(0, str(REPO / "scripts/manutencao"))
from proveniencia import agora_utc, sha256_arquivo
import executar_serializado as serial
import ponte_sar_terreno as ponte
from dados_pares_geograficos import imagens, particao
from fontes_sar import normalizar_fixo
from svc02_contrato_inferencia import avaliar_integracao
from memoria_integrada import construir as construir_memoria, recuperar, json_valido

RAIZ = REPO / "local_runs/ia-integrada"
MT = REPO / "local_runs/mecanismos-transferiveis/v3"
AB = REPO / "local_runs/ponte-sar-terreno/v1/ablacao"
CONT = REPO / "local_runs/aprendizado-continuo/sar_v1/protocolo.json"
QUADRO = "mt03_piso_amostral_v1"
LIMITES = ["Ordenação no contraste observado; não é probabilidade anual.",
           "Alagamento pluvial urbano isolado e previsão datada não demonstrados.",
           "Piso físico MT03 depende do quadro amostral: não substituir por piso do raster.",
           "HAND da percepção tem drenagem recortada; terreno não é necessariamente contemporâneo ao evento.",
           "Associação e contribuição do estimador não identificam causa.",
           "Integração de engenharia não confirma generalização ou vantagem autônoma."]


def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def gravar(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fonte(p):
    return {"arquivo": str(Path(p).resolve()), "sha256": sha256_arquivo(Path(p))}


def conferir(f):
    if sha256_arquivo(Path(f["arquivo"])) != f["sha256"]:
        raise ValueError(f"fonte alterada: {f['arquivo']}")


def salvar(modelo, destino, meta):
    """Diretório novo; manifesto é escrito por último. Nunca promover implicitamente."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=False)
    peso = destino / "modelo.pt"
    torch.save(modelo.state_dict(), peso)
    meta = {**meta, "versao": VERSAO, "peso": {"arquivo": "modelo.pt", "sha256": sha256_arquivo(peso)},
            "quando_utc": agora_utc(), "promovido": False, "limites": LIMITES}
    gravar(destino / "manifesto.json", meta)
    return meta


def resolver_pasta(pasta):
    pasta = Path(pasta)
    if pasta.resolve() == RAIZ.resolve():
        estado = ler(RAIZ / "estado.json")
        alvo = (RAIZ / estado["modelo_ativo"]).resolve()
        if RAIZ.resolve() not in alvo.parents:
            raise ValueError("modelo ativo fora da raiz integrada")
        if sha256_arquivo(alvo / "modelo.pt") != estado["peso_sha256"]:
            raise ValueError("modelo ativo diverge do registro")
        pasta = alvo
    return pasta


def carregar(pasta):
    pasta = resolver_pasta(pasta)
    meta = ler(pasta / "manifesto.json")
    if meta["versao"] != VERSAO or meta["peso"]["arquivo"] != "modelo.pt":
        raise ValueError("versão/arquivo de pesos não admitido")
    if sha256_arquivo(pasta / "modelo.pt") != meta["peso"]["sha256"]:
        raise ValueError("peso integrado alterado")
    m = IAIntegrada(meta["base"])
    m.load_state_dict(torch.load(pasta / "modelo.pt", weights_only=True, map_location="cpu"))
    if not all(torch.isfinite(t).all() for t in m.state_dict().values()):
        raise ValueError("estado não finito")
    if not (m.desvio > 0).all():
        raise ValueError("normalização física inválida")
    return m.eval(), meta


def exportar(destino=RAIZ / "v1"):
    """Escolhas fixas: G_terreno e semente 7, sem escolher pelo novo desempenho."""
    fontes = [fonte(MT / n) for n in ("candidatas_congeladas.pkl", "candidatas_congeladas.json")]
    info = ler(MT / "candidatas_congeladas.json")
    if fontes[0]["sha256"] != info["pkl_sha256"]:
        raise ValueError("candidata física não corresponde ao registro")
    # Apenas este pickle local, com hash confrontado ao registro pré-existente.
    with (MT / "candidatas_congeladas.pkl").open("rb") as f:
        antigo = pickle.load(f)["modelos"]["G_terreno"]
    if tuple(antigo.principais) != FEATURES or antigo.interacoes or antigo.territorio:
        raise ValueError("arquitetura física incompatível")
    p_ab = ler(AB / "protocolo.json")
    treino = ler(AB / "combinado_s7/treino.json")
    peso = AB / "combinado_s7/melhor.pt"
    from proveniencia_de_peso import hash_completo_do_estado
    e = torch.load(peso, weights_only=True, map_location="cpu")
    if hash_completo_do_estado(e["modelo"]) != treino["hash_estado"]:
        raise ValueError("peso de percepção não corresponde ao treino")
    m = IAIntegrada(p_ab["config"]["base"])
    m.percepcao.load_state_dict(e["modelo"])
    with torch.no_grad():
        m.relacoes.weight.copy_(torch.tensor(antigo.b)[None])
        for nome, attr in (("media", "mu"), ("desvio", "sd"), ("mediana", "med")):
            getattr(m, nome).copy_(torch.tensor(getattr(antigo.pad, attr).reindex(FEATURES).to_numpy()))
    p1 = ler(CONT)
    fontes += [fonte(p) for p in (peso, AB / "combinado_s7/treino.json", AB / "protocolo.json",
                                 AB / "resultado.json", CONT, Path(p1["campanha"]) / "inventario.json")]
    meta = {"tipo": "uma_ia_duas_tarefas", "base": p_ab["config"]["base"], "fontes": fontes,
            "fisica": {"features": FEATURES, "quadro": QUADRO, "origem": "MT03 G_terreno",
                       "contexto_cientifico": {"populacao": "contraste_observado_mt03",
                          "preditores_calculaveis_sem_rotulo": False,
                          "avaliacao_independente": {"estado": "not_testable"}},
                       "features_brutas": ["hand_m", "elevation_local_m", "ln_sca", "slope_deg"]},
            "percepcao": {"origem": "ablacao SAR-terreno combinado_s7", "canais": 7,
                          "modo": "pos_duplicado", "normalizacao": "fixa",
                          "treino": treino, "ganho_combinado": ler(AB / "resultado.json")["contrastes"]},
            "aprendizado": {"fisica": "parâmetros herdados; congelados nesta atualização",
                            "percepcao": "SAR e terreno ajustados conjuntamente; continuação com replay disponível",
                            "treino_conjunto_entre_tarefas": False},
            "pai": None}
    meta["memoria"] = construir_memoria()
    meta["fontes"] += meta["memoria"]["fontes"]
    meta["codigo"] = [fonte(p) for p in (Path(__file__), Path(__file__).with_name("modelo_integrado.py"),
                          Path(__file__).with_name("memoria_integrada.py"),
                          REPO / "scripts/servico/svc02_contrato_inferencia.py")]
    return salvar(m, destino, meta)


def atributos(d):
    """Transformações MT03 exatas, sem calcular piso a partir da consulta."""
    d = d[["hand_m", "elevation_local_m", "ln_sca", "slope_deg"]].apply(pd.to_numeric)
    return pd.DataFrame({"f_hand": np.log1p(d.hand_m.clip(lower=0)),
                         "f_elev": np.log1p(d.elevation_local_m.clip(lower=0)),
                         "f_sca": d.ln_sca.clip(-5, 25),
                         "f_declive": np.log1p(d.slope_deg.clip(lower=0))})


def inventario(meta):
    p = next((f for f in meta["fontes"] if Path(f["arquivo"]).name == "inventario.json"), None)
    if p is None:
        raise ValueError("modelo sem inventário de cenas admitidas")
    conferir(p)
    return {r["nome"]: {**r, "fonte": "sen1"} for r in ler(p["arquivo"])["pares"]}


def cena(r):
    """Entrada sem abrir rótulo. A máscara replica o suporte do ajuste legado."""
    for k in ("antes", "depois", "jrc"):
        if sha256_arquivo(Path(r["caminhos"][k])).split(":")[-1] != r["hashes"][k]:
            raise ValueError(f"entrada alterada: {r['nome']} {k}")
    x, sensor = imagens(r["caminhos"]["antes"], r["caminhos"]["depois"])
    xs = normalizar_fixo(x)
    xs[2:] = xs[:2]  # Este peso não aprendeu diferença antes/depois.
    xt, vt, prov = ponte.terreno(r)
    v = sensor & vt
    z = np.concatenate([xs, xt]).astype(np.float32)
    z[:, ~v] = 0
    return z, v, prov


def alvo(r, v):
    """Usado só em avaliação/ajuste. Consulta não acessa a verdade de referência."""
    from campanha_independencia_autonomia import alvo as alvo_legado
    for k in ("rotulo", "jrc"):
        if sha256_arquivo(Path(r["caminhos"][k])).split(":")[-1] != r["hashes"][k]:
            raise ValueError("referência observacional alterada")
    y, _, _ = alvo_legado(r, v)
    return y


@torch.no_grad()
def consultar(pasta, pedido):
    m, meta = carregar(pasta)
    contrato = avaliar_integracao(meta, pedido)
    out = {"modelo": meta["peso"]["sha256"], "versao": VERSAO, "contrato": contrato,
           "limites": meta["limites"], "status": "recusado" if not contrato["autorizado"] else "pesquisa"}
    if pedido.get("territorio") or pedido.get("municipio_id"):
        out["contexto"] = recuperar(meta.get("memoria", {}), pedido.get("territorio"), pedido.get("municipio_id"))
    if not contrato["autorizado"]:
        return out
    if pedido.get("pontos"):
        if pedido.get("quadro") != QUADRO:
            raise ValueError("quadro físico ausente/incompatível; não substituir definição de elevação")
        d = pd.DataFrame(pedido["pontos"])
        x = torch.tensor(atributos(d).to_numpy(), dtype=torch.float64)
        r = m(atributos=x)
        out["suscetibilidade"] = {"tipo": "ordem_relativa_no_contraste_observado",
              "valores": r["ordem_relativa"].tolist(),
              "contribuicoes": [dict(zip(FEATURES, c)) for c in r["contribuicoes"].tolist()],
              "ausentes": r["atributos_ausentes"].tolist()}
    if pedido.get("chip"):
        inv = inventario(meta)
        if pedido["chip"] not in inv:
            raise ValueError("chip fora do inventário de desenvolvimento admitido")
        r = inv[pedido["chip"]]
        x, v, prov = cena(r)
        p = m(cena=torch.from_numpy(x)[None])["logits_agua_observada"].softmax(1)[0, 1].numpy()
        import rasterio
        with rasterio.open(r["caminhos"]["jrc"]) as f:
            permitido = v & (f.read(1) == 0) & (f.read_masks(1) > 0)
        out["observacao"] = {"chip": r["nome"], "evento": r["evento"],
                "tipo": "escore_de_agua_na_visada_sem_probabilidade_calibrada",
                "pixels_sensor_terreno_validos": int(v.sum()), "pixels_fora_permanente": int(permitido.sum()),
                "pixels_desconhecidos_ou_permanentes": int((~permitido).sum()),
                "escore_medio_fora_permanente": float(p[permitido].mean()) if permitido.any() else None,
                "procedencia_terreno": prov, "rotulo_usado_na_consulta": False}
    out["explicacao"] = explicar(out)
    return out


def explicar(r):
    partes = ["Análise de desenvolvimento da IA integrada."]
    if "suscetibilidade" in r:
        partes.append("A ordenação física combina altura sobre a drenagem, elevação no quadro MT03, área contribuinte e declive. As contribuições exibidas somam exatamente o escore.")
    if "observacao" in r:
        partes.append("A rede combina radar e terreno para analisar água na visada. Essa observação não altera a ordenação física e não prevê uma inundação futura.")
    partes.append("Os resultados não estabelecem a causa de uma ocorrência nem validade local confirmada.")
    return " ".join(partes)


def protocolo_real(meta):
    for f in meta["fontes"]:
        conferir(f)
    p1 = ler(CONT)
    pacote = p1.get("pacote")
    if not pacote or sha256_arquivo(Path(pacote["manifesto"])).split(":")[-1] != pacote["sha256"].split(":")[-1]:
        raise ValueError("pacote de admissão do treino SAR ausente/alterado")
    registro = ler(pacote["manifesto"])
    if registro.get("regime") != "desenvolvimento" or registro.get("promovido") is not False:
        raise ValueError("pacote não autoriza desenvolvimento")
    inv = inventario(meta)
    ordem = ler(Path(p1["campanha"]) / "protocolo.json")["ordem_eventos"]
    _, val, _ = particao(list(inv.values()), ordem[:3], ordem[3:5])
    grupos = ler(AB / "protocolo.json")["grupos"]
    fit = [n for e in p1["contextos_em_ordem"] for n in p1["anotacoes"]["7"][e]]
    if set(p1["contextos_em_ordem"]) != set(pacote["sar_treino"]):
        raise ValueError("contextos SAR fora do papel de treino admitido")
    if set(fit) & (set(grupos["retencao"]) | set(grupos["avaliacao"]) | {r["nome"] for r in val}):
        raise ValueError("sobreposição entre ajuste e avaliação")
    return p1, inv, val, grupos, fit


@torch.no_grad()
def medir_percepcao(m, linhas):
    from sklearn.metrics import average_precision_score
    # Histogramas idênticos ao protocolo legado; acumulados por evento, sem milhões de valores em RAM.
    h = defaultdict(lambda: [np.zeros(1000, np.int64), np.zeros(1000, np.int64)])
    ce, n = 0., 0
    m.eval()
    for r in linhas:
        x, v, _ = cena(r)
        y = alvo(r, v)
        z = m(cena=torch.from_numpy(x)[None])["logits_agua_observada"]
        yt = torch.from_numpy(y)[None]
        ce += float(torch.nn.functional.cross_entropy(z, yt, ignore_index=-1, reduction="sum"))
        n += int((y != -1).sum())
        p = z.softmax(1)[0, 1].numpy()
        idx = np.minimum((p*1000).astype(int), 999)
        for c in (0, 1):
            h[r["evento"]][c] += np.bincount(idx[y == c], minlength=1000)
    import calibracao_decisao as cd
    eventos = {e: {"ap": cd.ap(hist[1], hist[0]), "positivos": int(hist[1].sum()),
                   "negativos": int(hist[0].sum())} for e, hist in h.items()}
    return {"perda": ce/n if n else None, "pixels_rotulados": n, "por_evento": eventos,
            "chips": len(linhas)}


def avaliar(pasta, saida=None):
    """Todas as nove unidades MT03-B e todos os 222 chips da avaliação SAR pré-existente."""
    from sklearn.metrics import roc_auc_score
    pasta = resolver_pasta(pasta)
    m, meta = carregar(pasta)
    p1, inv, val, grupos, fit = protocolo_real(meta)
    outdir = Path(saida or Path(pasta) / "avaliacao")
    outdir.mkdir(parents=True, exist_ok=True)
    dados = MT / "novas/pontos_avaliacao.parquet"
    referencia = MT / "novas/predicoes.parquet"
    prot = {"modelo": meta["peso"]["sha256"], "dados_fisicos": fonte(dados), "referencia": fonte(referencia),
            "grupos_sar": grupos, "metrica_fisica": "AUC por evento e unidade", "metrica_sar": "AP por evento",
            "natureza": "reprodução e avaliação da integração em desenvolvimento; não confirmação",
            "decisao": "nenhuma seleção de modelo pelo resultado", "reservas_abertas": False}
    prot["codigo"] = [fonte(p) for p in (Path(__file__), Path(__file__).with_name("modelo_integrado.py"))]
    pf = outdir / "protocolo.json"
    if pf.exists() and ler(pf) != prot:
        raise ValueError("avaliação já registrada com outro protocolo")
    gravar(pf, prot)
    d = pd.read_parquet(dados)
    x = atributos(d)
    with torch.no_grad():
        s = m(atributos=torch.tensor(x.to_numpy(), dtype=torch.float64))["ordem_relativa"].numpy()
    ref = pd.read_parquet(referencia)
    if not d.ponto_id.equals(ref.ponto_id):
        raise ValueError("identidades físicas não correspondem à referência")
    erro = float(np.max(np.abs(s-ref.G_terreno.to_numpy())))
    if erro > 1e-10:
        raise ValueError(f"exportação física divergiu: {erro}")
    unidades = []
    for (u, ev), g in d.assign(escore=s).groupby(["unidade", "evento_id"]):
        if len(g) >= 20 and g.classe.nunique() == 2:
            unidades.append({"unidade": str(u), "evento": str(ev), "linhas": len(g),
                             "auc_integrada": float(roc_auc_score(g.classe, g.escore)),
                             "auc_hand": float(roc_auc_score(g.classe, -g.hand_m))})
    medidas = {}
    for g, nomes in grupos.items():
        medidas[g] = medir_percepcao(m, [inv[n] for n in nomes])
        print(json.dumps({"grupo_concluido": g, "chips": len(nomes)}), flush=True)
    # Semente fixa: a referência é a mesma semente, não a média de sementes.
    av_ref = ler(AB / "combinado_s7/avaliacao.json")
    import calibracao_decisao as cd
    erros_ap = [abs(z["ap"] - cd.ap(*cd.arrays(av_ref[g][e])))
                for g, a in medidas.items() for e, z in a["por_evento"].items() if z["ap"] is not None]
    res = {"modelo": meta["peso"]["sha256"], "natureza": prot["natureza"],
           "fisica": {"linhas": len(d), "unidades": unidades, "max_erro_exportacao": erro},
           "percepcao": medidas, "max_erro_ap_exportacao": max(erros_ap, default=0.),
           "paridade_exigida": meta["pai"] is None, "promovido": False, "reservas_abertas": False}
    if meta["pai"] is None and res["max_erro_ap_exportacao"] > 1e-8:
        raise ValueError("percepção integrada não reproduziu o peso de origem")
    gravar(outdir / "resultado.json", res)
    return res


def continuar(pasta, destino, passos=24):
    """Atualização real, limitada e pré-declarada; seleção só pela validação.

    Continua o peso SAR-terreno na última etapa do treino, com replay dos
    outros quatro contextos. Não finge aquisição de novo evento. A física
    permanece idêntica. O candidato é preservado mesmo quando recusado.
    """
    if passos < 1 or passos > 768:
        raise ValueError("orçamento deve estar entre 1 e 768 passos")
    pasta = resolver_pasta(pasta)
    m, meta = carregar(pasta)
    p1, inv, val, grupos, fit = protocolo_real(meta)
    ultimo = p1["contextos_em_ordem"][-1]
    novos = sorted(p1["anotacoes"]["7"][ultimo])
    antigos = sorted(set(fit)-set(novos))
    cfg = {"passos": passos, "batch": 4, "recorte": 64, "lr": 1e-4, "seed": 73,
           "novos": novos, "replay": antigos, "validacao": sorted(r["nome"] for r in val),
           "alvo": "água observada não permanente", "piso": "CE validação candidato <= pai",
           "continua": "pesos; otimizador novo", "fisica_congelada": True,
           "dados_novos": False, "reservas_abertas": False}
    registro = Path(destino).with_name(Path(destino).name + "-protocolo.json")
    if registro.exists() or Path(destino).exists():
        raise FileExistsError("atualização já registrada; use outro destino")
    gravar(registro, {"pai": meta["peso"]["sha256"], "config": cfg})
    antes = medir_percepcao(m, val)
    fisica = {k: t.clone() for k, t in m.state_dict().items() if not k.startswith("percepcao.")}
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    opt = torch.optim.Adam(m.percepcao.parameters(), lr=cfg["lr"])
    hist = []
    m.train()
    for passo in range(passos):
        xs, ys = [], []
        for j in range(cfg["batch"]):
            n = str(rng.choice(novos if j < cfg["batch"]//2 else antigos))
            x, v, _ = cena(inv[n])
            y = alvo(inv[n], v)
            for _ in range(100):
                a, b = rng.integers(0, y.shape[0]-63), rng.integers(0, y.shape[1]-63)
                yy = y[a:a+64, b:b+64]
                if (yy != -1).any():
                    xs.append(torch.from_numpy(x[:, a:a+64, b:b+64].copy()))
                    ys.append(torch.from_numpy(yy.copy()))
                    break
            else:
                raise ValueError("sem recorte observável; passo não omitido")
        opt.zero_grad(set_to_none=True)
        z = m(cena=torch.stack(xs))["logits_agua_observada"]
        perda = torch.nn.functional.cross_entropy(z, torch.stack(ys), ignore_index=-1)
        if not torch.isfinite(perda):
            raise ValueError("perda não finita")
        perda.backward()
        opt.step()
        hist.append(float(perda.detach()))
    depois = medir_percepcao(m.eval(), val)
    if any(not torch.equal(t, m.state_dict()[k]) for k, t in fisica.items()):
        raise ValueError("atualização alterou o núcleo físico congelado")
    decisao = {"aceito_em_desenvolvimento": depois["perda"] <= antes["perda"],
               "pai_validacao": antes, "candidato_validacao": depois,
               "fisica_identica": True, "passos_executados": len(hist), "perdas": hist,
               "ganho_cientifico_confirmado": False}
    # A decisão não promove ou substitui o pai. Ambos são endereçáveis.
    nova = {**meta, "pai": fonte(Path(pasta) / "modelo.pt"), "atualizacao": decisao,
            "fontes": meta["fontes"] + [fonte(registro)], "protocolo_atualizacao": cfg}
    salvar(m, destino, nova)
    return decisao


def campanha(pasta, destino, passos=24):
    """Um ciclo efetivo completo, sob uma única tomada da trava compartilhada."""
    pasta = resolver_pasta(pasta)
    destino = Path(destino)
    if destino.exists():
        raise FileExistsError("campanha já existe; usar nova versão")
    destino.mkdir(parents=True)
    codigo = destino / "codigo"
    codigo.mkdir()
    for p in (Path(__file__), Path(__file__).with_name("modelo_integrado.py"),
              Path(__file__).with_name("memoria_integrada.py"),
              REPO / "scripts/servico/svc02_contrato_inferencia.py"):
        shutil.copyfile(p, codigo / p.name)
    _, meta = carregar(pasta)
    gravar(destino / "protocolo_ciclo.json", {"pai": meta["peso"]["sha256"], "passos": passos,
              "sequencia": ["reproduzir pai", "continuar com replay", "avaliar candidato", "registrar decisão"],
              "criterio": "CE na validação; avaliação posterior não decide aceite", "promovido": False,
              "codigo": [fonte(p) for p in codigo.iterdir()]})
    base = avaliar(pasta, destino / "avaliacao_pai")
    filho = destino / "candidato"
    decisao = continuar(pasta, filho, passos)
    candidato = avaliar(filho, destino / "avaliacao_candidato")
    escolhido = filho if decisao["aceito_em_desenvolvimento"] else Path(pasta)
    _, atual = carregar(escolhido)
    registro = {"modelo_ativo": str(escolhido.resolve().relative_to(RAIZ.resolve())),
                "peso_sha256": atual["peso"]["sha256"], "maturidade": "desenvolvimento",
                "promovido": False, "decisao": str((destino / "decisao.json").resolve())}
    resultado = {"pai": meta["peso"]["sha256"], "candidato": candidato["modelo"],
                 "decisao": decisao, "modelo_ativo": registro, "avaliacao_pai": fonte(destino / "avaliacao_pai/resultado.json"),
                 "avaliacao_candidato": fonte(destino / "avaliacao_candidato/resultado.json"),
                 "reservas_abertas": False, "promovido": False}
    gravar(destino / "decisao.json", resultado)
    gravar(RAIZ / "estado.json", registro)
    # Exemplos reais: seleção pelo identificador, sem escolher pela classe ou desempenho.
    d = pd.read_parquet(MT / "novas/pontos_avaliacao.parquet").sort_values("ponto_id")
    pontos = json_valido(d.head(3)[["hand_m", "elevation_local_m", "ln_sca", "slope_deg"]].to_dict("records"))
    pedido = {"uso": "demonstracao_historica", "quadro": QUADRO, "pontos": pontos,
              "territorio": "CEMS:EMSR662", "municipio_id": "4106902"}
    gravar(destino / "pedido_fisico_real.json", pedido)
    resposta = consultar(escolhido, pedido)
    resposta["origem_pontos"] = {"fonte": fonte(MT / "novas/pontos_avaliacao.parquet"),
                                "ids": d.head(3).ponto_id.tolist(),
                                "contextos": "cartão EMSR662 e município são consultas documentais separadas; não localizam estes pontos"}
    gravar(destino / "resposta_fisica_real.json", resposta)
    _, inv, _, grupos, _ = protocolo_real(atual)
    chip = sorted(grupos["avaliacao"])[0]
    q = {"uso": "demonstracao_historica", "chip": chip}
    gravar(destino / "pedido_imagem_real.json", q)
    gravar(destino / "resposta_imagem_real.json", consultar(escolhido, q))
    # Recusa real pelo mesmo motor: nenhum ajuste do serviço legado.
    pedido["uso"] = "consulta_lugar_novo"
    gravar(destino / "recusa_lugar_novo.json", consultar(escolhido, pedido))
    return resultado


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("acao", choices=("exportar", "consultar", "avaliar", "continuar", "campanha"))
    p.add_argument("--modelo", type=Path, default=RAIZ)
    p.add_argument("--saida", type=Path)
    p.add_argument("--pedido", type=Path)
    p.add_argument("--passos", type=int, default=24)
    p.add_argument("--esperar-min", type=float, default=0.)
    p.add_argument("--intervalo-trava-s", type=float, default=.1)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    dono = None
    try:
        if a.acao in ("avaliar", "continuar", "campanha"):
            if not serial.energia()["na_tomada"]:
                raise RuntimeError("processamento pesado aguarda alimentação externa")
            dono = serial.tomar([sys.executable, __file__, a.acao], a.esperar_min, False,
                               intervalo_s=max(.01, a.intervalo_trava_s))
            if dono is None:
                raise RuntimeError("outro processamento pesado está ativo; repetir após liberar")
        if a.acao == "exportar":
            r = exportar(a.saida or (RAIZ / "v1-completa" if a.modelo == RAIZ else a.modelo))
        elif a.acao == "consultar":
            if a.pedido is None:
                p.error("consultar exige --pedido")
            r = consultar(a.modelo, ler(a.pedido))
            if a.saida:
                gravar(a.saida, r)
        elif a.acao == "avaliar":
            r = avaliar(a.modelo, a.saida)
        elif a.acao == "campanha":
            if a.saida is None:
                p.error("campanha exige --saida")
            r = campanha(a.modelo, a.saida, a.passos)
        else:
            if a.saida is None:
                p.error("continuar exige --saida para uma versão nova")
            r = continuar(a.modelo, a.saida, a.passos)
        print(json.dumps({"acao": a.acao, "concluido": True,
                          "resultado": r if a.acao in ("consultar", "continuar") else str(a.saida or a.modelo)},
                         ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        if dono:
            serial.liberar(dono)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
