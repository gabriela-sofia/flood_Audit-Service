"""IB-04: revisão humana cega e diagnóstico brasileiro sem consumir reservas.

--preparar cria revisão offline dos mesmos 100 pontos, diagnóstico ponderado
e fila por finalidade usando exclusivamente os arquivos existentes.
--humana importa o CSV exportado pela revisão sem alterar a amostra original.
Não admite SAR, não treina modelos e não abre rótulos de eventos reservados.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PILOTO = REPO / "local_runs/instrumento-brasileiro/ib-03a"
SAIDA = REPO / "local_runs/referencia-brasileira/2026-10-11"
CLASSES = ("agua", "seco", "nao_sei")


def sha(p):
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def gravar(p, d):
    """Artefatos novos são imutáveis; mudança exige outra pasta."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if p.exists() and p.read_text(encoding="utf-8") != txt:
        raise ValueError(f"artefato já existe com conteúdo diferente: {p}; use nova pasta")
    p.write_text(txt, encoding="utf-8")


def linhas_csv(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def verificar_piloto(piloto):
    p = Path(piloto)
    cong = ler(p / "anotacao_congelada.json")
    for nome, campo in (("anotacao_ia.json", "anotacao_sha256"), ("chave_estratos.json", "chave_sha256")):
        if sha(p / nome) != cong[campo]:
            raise ValueError(f"hash congelado divergente: {nome}")
    pre = ler(p / "pre_registro.json")
    corpo = {k: v for k, v in pre.items() if k not in ("hash", "quando_utc")}
    h = "sha256:" + hashlib.sha256(json.dumps(corpo, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if h != pre["hash"]:
        raise ValueError("pré-registro alterado")
    if ler(p / "resultado.json")["pre_registro"] != h:
        raise ValueError("resultado não corresponde ao pré-registro")
    return cong


def amostra_fixa(piloto):
    p = Path(piloto)
    linhas = linhas_csv(p / "revisao_humana/amostra_para_anotar.csv")
    ids = [r["codigo"] for r in linhas]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("a revisão exige exatamente os mesmos 100 códigos únicos")
    pontos_lista = ler(p / "pontos_cegos.json")
    pontos = {r["codigo"]: r for r in pontos_lista}
    # Reconstrói o sorteio histórico, em vez de aceitar qualquer lista de 100.
    import numpy as np
    instr = ler(p / "revisao_humana/instrucoes.json")
    fixos = set(np.random.default_rng(instr["semente"]).choice(
        [r["codigo"] for r in pontos_lista], 100, replace=False))
    if set(ids) != fixos:
        raise ValueError("amostra não corresponde ao sorteio histórico congelado")
    for r in linhas:
        q = pontos.get(r["codigo"])
        if q is None or any(abs(float(r[k]) - q[k]) > 1e-6 for k in ("lon", "lat")):
            raise ValueError("código/coordenadas da revisão divergentes")
    return linhas


def avaliar_respostas(linhas, esperados, ia):
    """Kappa nominal nas três classes; parcial nunca valida a referência.

    Identidade/tamanho são verificados antes da concordância. Não sei participa
    da concordância, mas não passa a ser negativo para avaliar o instrumento.
    """
    ids = [r.get("codigo") for r in linhas]
    if len(esperados) != 100 or len(set(esperados)) != 100:
        raise ValueError("amostra fixa deve conter 100 códigos únicos")
    if len(ids) != 100 or len(set(ids)) != 100 or set(ids) != set(esperados):
        raise ValueError("CSV deve conter exatamente os 100 códigos da amostra, sem duplicatas/substituições")
    matriz = {h: {a: 0 for a in CLASSES} for h in CLASSES}
    feitas = []
    for r in linhas:
        cols = [k for k in r if k.startswith("classe")]
        if len(cols) != 1:
            raise ValueError("CSV precisa de uma única coluna classe")
        v = r[cols[0]]
        if v is None:
            raise ValueError("linha CSV incompleta")
        v = v.strip()
        if v and v not in CLASSES:
            raise ValueError(f"classe inválida em {r['codigo']}: {v}")
        if v:
            a = ia[r["codigo"]]["classe"]
            matriz[v][a] += 1
            feitas.append((v, a))
    n = len(feitas)
    if n < 100:
        return {"estado": "aguardando_anotacao_humana", "pontos_anotados": n,
                "pontos_pendentes": 100 - n, "kappa": None, "referencia_ia_validada": False,
                "instrumento_sar_admitido_por_esta_revisao": False}
    po = sum(h == a for h, a in feitas) / n
    pe = sum(sum(matriz[c].values()) * sum(matriz[h][c] for h in CLASSES) for c in CLASSES) / n**2
    kappa = (po - pe) / (1 - pe) if pe < 1 else None
    agua_h = sum(matriz["agua"].values())
    agua_ia = sum(matriz[h]["agua"] for h in CLASSES)
    return {"estado": "avaliado" if kappa is not None else "concordancia_nao_identificavel",
            "pontos_anotados": n, "pontos_pendentes": 0, "kappa": kappa,
            "concorda": po, "matriz_humana_x_ia": matriz,
            "referencia_ia_validada": kappa is not None and kappa >= .8,
            "agua_humana": agua_h, "agua_ia": agua_ia,
            "concordancia_agua_sobre_agua_humana": matriz["agua"]["agua"] / agua_h if agua_h else None,
            "instrumento_sar_admitido_por_esta_revisao": False,
            "limite": "kappa >= 0,8 é a regra congelada da referência por IA neste piloto; não valida SAR, mecanismo ou outro evento"}


def diagnostico(piloto):
    p = Path(piloto)
    verificar_piloto(p)
    chave, ia = ler(p / "chave_estratos.json"), ler(p / "anotacao_ia.json")
    nomes = {"0": "seco", "1": "agua", "2": "ambiguo"}
    N = {nomes[k]: int(v) for k, v in chave["pixels_por_classe_na_regiao"].items()}
    n = Counter(r["estrato"] for r in chave["pontos"])
    tab = {s: Counter() for s in N}
    pesos_agua = []
    for r in chave["pontos"]:
        s, c = r["estrato"], ia[r["codigo"]]["classe"]
        tab[s][c] += 1
        if c == "agua":
            pesos_agua.append(N[s] / n[s])
    W = sum(pesos_agua)
    A = N["agua"] / n["agua"] * tab["agua"]["agua"]
    U = {s: N[s] / n[s] * tab[s]["nao_sei"] for s in N}
    ess = W**2 / sum(w*w for w in pesos_agua) if W else 0
    por_estrato = {s: {"pixels": N[s], "amostrados": n[s], "avaliaveis": n[s] - tab[s]["nao_sei"],
                       "agua_referencia": tab[s]["agua"], "nao_sei": tab[s]["nao_sei"],
                       "peso": N[s] / n[s]} for s in N}
    # Diagnóstico de sensibilidade, não IC nem substituto de C1-C5.
    limites = {"revocacao": [A / (W + sum(U[s] for s in N if s != "agua")),
                              (A + U["agua"]) / (W + U["agua"])],
               "precisao": [tab["agua"]["agua"] / n["agua"],
                             (tab["agua"]["agua"] + tab["agua"]["nao_sei"]) / n["agua"]]}
    # Alocação ilustrativa para detectar omissões: Neyman para proporção de água
    # de referência nos estratos NÃO água. Estimativas do piloto são instáveis.
    es = ("seco", "ambiguo")
    scores = {}
    for s in es:
        m = n[s] - tab[s]["nao_sei"]
        ph = tab[s]["agua"] / m
        scores[s] = N[s] * math.sqrt(ph * (1 - ph))
    alvo = {s: round(800 * scores[s] / sum(scores.values())) for s in es}
    return {"contagem_ia": dict(Counter(a["classe"] for a in ia.values())), "estratos": por_estrato,
            "fracao_desconhecida_ponderada": sum(U.values()) / sum(N.values()),
            "agua_referencia_n": len(pesos_agua), "agua_n_efetivo_kish": ess,
            "fracao_peso_agua_dos_dois_pontos_maiores": sum(sorted(pesos_agua)[-2:]) / W,
            "limites_sensibilidade_desconhecidos": limites,
            "limites_nao_sao_ic": True,
            "populacao": "sobreposição orbital, exclusão JRC >= 50, cena S2; métricas atuais condicionadas à referência anotável",
            "continuacao_proposta_nao_congelada": {
                "agua_instrumento": "censo dos 877 pixels, deduplicado dos 150 existentes; foco em C1/C5",
                "omissoes": {"novos_pontos_exemplo": 800, "alocacao_neyman_ilustrativa": alvo,
                             "regra": "sorteio probabilístico separado em seco/ambíguo; registrar N, n, probabilidades, blocos e recusas"},
                "regras": ["pré-registrar nova versão antes do sorteio; não sobrescrever IB-03a",
                           "não escolher pixels por HAND, escore ou concordância",
                           "não substituir desconhecido por seco; não repor silenciosamente nuvem",
                           "amostra dirigida de água pode diagnosticar erro, mas não estima revocação populacional sem probabilidades",
                           "800 é cenário de orçamento, não garantia de poder; dimensionar precisão por blocos após revisão humana",
                           "mais amostras reduzem incerteza, não corrigem baixa revocação da regra fixa"]}}


def gerar_html(linhas, piloto, saida):
    """Recorta pranchas científicas existentes; não muda pixels ou rótulos."""
    from PIL import Image
    itens = []
    hashes = {}
    for r in linhas:
        k = int(r["codigo"][1:])
        prancha = f"prancha_{k // 16:02d}.png"
        if r["prancha"] != prancha:
            raise ValueError("prancha divergente")
        arquivo = Path(piloto) / prancha
        hashes[prancha] = sha(arquivo)
        with Image.open(arquivo) as im:
            w, h = im.width // 4, im.height // 4
            col, row = k % 4, (k % 16) // 4
            crop = im.crop((col*w, row*h, (col+1)*w, (row+1)*h))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
        itens.append({"codigo": r["codigo"], "imagem": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()})
    payload = json.dumps(itens, ensure_ascii=False).replace("<", "\\u003c")
    interface = Path(__file__).with_name("obs05_revisao_cega.html").read_text(encoding="utf-8")
    pagina = interface.replace("__ITENS__", payload)
    arq = Path(saida) / "revisao_cega.html"
    if arq.exists() and arq.read_text(encoding="utf-8") != pagina:
        raise ValueError("revisão gerada mudou; use nova pasta")
    arq.write_text(pagina, encoding="utf-8")
    return hashes


def fila_brasileira(repo):
    runs = Path(repo) / "local_runs"
    eleg = ler(runs / "candidatos-independentes/elegibilidade_por_pergunta.json")
    catalogo = {c["id"]: c for c in ler(runs / "candidatos-independentes/catalogo.json")}
    protegidos = {c["id"] for c in eleg["candidatos"] if c["brasil"] and c["evento_novo"] and c["independente_local"]}
    dev = []
    for e in linhas_csv(runs / "br-preparo-sem-rotulo/eventos.csv"):
        eid = "GDACS:" + e["eventid"]
        if e["inicio"] >= "2025-01-01" and eid not in protegidos and e["eventid"] != "1103101":
            dev.append({"id": eid, "inicio": e["inicio"], "fim": e["fim"], "estado_catalogo": e["estado"],
                        "estado": "consultar_metadados_s1_s2", "referencia_admitida": False})
    dev.append({"id": "serra03:rs_taquari_jun2025", "inicio": "2025-06-16", "fim": "2025-07-01",
                "estado": "consultar_metadados_s1_s2", "referencia_admitida": False})
    reserva = [{"id": c["id"], "inicio": catalogo.get(c["id"], {}).get("inicio"),
                "independente_local_snapshot": c["independente_local"], "evento_novo_snapshot": c["evento_novo"],
                "estado": "potencial_reserva_instrumento_nao_validado", "elegivel_confirmacao": False}
               for c in eleg["candidatos"] if c["id"] in protegidos]
    return {"desenvolvimento_instrumento": sorted(dev, key=lambda e: (e["inicio"], e["id"])),
            "reservas_brasileiras_potenciais": sorted(reserva, key=lambda e: (e["inicio"] or "", e["id"])),
            "regra_segundo_evento": "mesmo pool/regra IB-03a, excluir Manaus e potenciais reservas; primeiro elegível por início/id, só metadados; registrar seleção separada antes de pixels",
            "limite": "snapshot local; novidade/independência devem ser rechecadas contra versão atual e exames posteriores antes da admissão",
            "urbano_pluvial": {"estado": "not_testable", "acervo_a_reutilizar": ["curitiba", "recife", "sao_paulo"],
                               "limites": ["Curitiba: mecanismo misto não separável", "Recife/SP: ausência de registro não é negativo observado"],
                               "admissao_exige": ["mecanismo independente de relevo/escore: chuva local/drenagem, fluvial ou misto declarados",
                                                   "evidência georreferenciada datada com precisão espacial/temporal documentada",
                                                   "água e seco explicitamente observados na mesma janela/área de vistoria",
                                                   "direitos e cobertura de observação; células invisíveis permanecem desconhecidas",
                                                   "identidade canônica e dependência por episódio, evento, bacia, território e fonte",
                                                   "protocolo próprio antes de avaliação; não transferir automaticamente C1-C5 de água aberta para lâmina urbana"]}}


def preparar(repo=REPO, piloto=PILOTO, saida=SAIDA):
    repo, piloto, saida = Path(repo), Path(piloto), Path(saida)
    cong = verificar_piloto(piloto)
    linhas = amostra_fixa(piloto)
    saida.mkdir(parents=True, exist_ok=True)
    fontes = [piloto / n for n in ("pre_registro.json", "anotacao_congelada.json", "anotacao_ia.json",
              "chave_estratos.json", "pontos_cegos.json", "resultado.json", "selecao.json", "pranchas.json")]
    fontes += [repo / "local_runs" / n for n in ("candidatos-independentes/elegibilidade_por_pergunta.json",
               "candidatos-independentes/catalogo.json", "br-preparo-sem-rotulo/eventos.csv")]
    fontes += [piloto / "revisao_humana/instrucoes.json"]
    manifest = {"tipo": "revisao_cega_ib04", "codigos_fixos": [r["codigo"] for r in linhas],
                "anotacao_congelada": cong, "fontes": {str(p.resolve()): sha(p) for p in fontes},
                "revisao": "mesmos 100 pontos já sorteados; sem estrato/IA/terreno; respostas ficam no navegador até exportar CSV",
                "pranchas": gerar_html(linhas, piloto, saida),
                "reserva_aberta": False, "instrumento_admitido": False}
    gravar(saida / "manifesto_revisao.json", manifest)
    gravar(saida / "diagnostico.json", diagnostico(piloto))
    gravar(saida / "fila_por_finalidade.json", fila_brasileira(repo))
    return {"saida": str(saida), "pontos": len(linhas), "revisao": str(saida / "revisao_cega.html")}


def importar_humana(csv_path, piloto=PILOTO, saida=SAIDA, revisor="", cega=False):
    if not revisor.strip() or not cega:
        raise ValueError("identificar revisor e declarar revisão cega (sem IA/SAR/terreno/escore)")
    piloto, saida = Path(piloto), Path(saida)
    verificar_piloto(piloto)
    manifest = ler(saida / "manifesto_revisao.json")
    for p, h in manifest["fontes"].items():
        if sha(p) != h:
            raise ValueError(f"fonte mudou depois da preparação: {p}")
    for nome, h in manifest["pranchas"].items():
        if sha(piloto / nome) != h:
            raise ValueError(f"prancha mudou depois da preparação: {nome}")
    out = avaliar_respostas(linhas_csv(csv_path), manifest["codigos_fixos"], ler(piloto / "anotacao_ia.json"))
    out.update({"revisor": revisor.strip(), "cegueira_declarada": cega, "respostas_sha256": sha(csv_path),
                "manifesto_sha256": sha(saida / "manifesto_revisao.json"), "avaliador_sha256": sha(__file__),
                "quando_utc": datetime.now(timezone.utc).isoformat(),
                "cegueira": "declaração humana; não certificada por software"})
    dest = saida / "respostas" / sha(csv_path).split(":")[1][:16]
    dest.mkdir(parents=True, exist_ok=True)
    alvo = dest / "respostas.csv"
    if alvo.exists() and alvo.read_bytes() != Path(csv_path).read_bytes():
        raise ValueError("respostas divergentes")
    alvo.write_bytes(Path(csv_path).read_bytes())
    # Uma reimportação idêntica preserva a data da primeira importação.
    if (dest / "concordancia.json").exists():
        anterior = ler(dest / "concordancia.json")
        out["quando_utc"] = anterior["quando_utc"]
    gravar(dest / "concordancia.json", out)
    return out


def selecionar_segundo(saida=SAIDA):
    """Nova seleção de metadados; não abre pixels nem a reserva brasileira."""
    import obs04_instrumento_brasileiro as obs
    saida = Path(saida)
    manifest = ler(saida / "manifesto_revisao.json")
    for p, h in manifest["fontes"].items():
        if sha(p) != h:
            raise ValueError(f"fonte mudou depois da preparação: {p}")
    prot = {r["id"].split(":")[-1] for r in ler(saida / "fila_por_finalidade.json")["reservas_brasileiras_potenciais"]}
    pool = [e for e in obs._pool() if e["eventid"] != "1103101" and e["eventid"] not in prot]
    pre_original = ler(PILOTO / "pre_registro.json")
    pre = {"id": "IB-04-segundo-evento-metadados", "pool": pool,
           "regra_do_evento": pre_original["regra_do_evento"],
           "regra_sobreposicao": {"area_comum_min_km2": obs.AREA_MIN_KM2},
           "ordem": "início, id; primeiro elegível, excluindo Manaus e reservas potenciais",
           "papel": "desenvolvimento do instrumento; não confirmação do modelo",
           "rotulos_abertos": False, "pixels_abertos": False,
           "origem_regra": pre_original["hash"], "codigo_selecao_sha256": sha(obs.__file__)}
    pre["hash"] = "sha256:" + hashlib.sha256(json.dumps(pre, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    dest = saida / "segundo_evento"
    dest.mkdir(parents=True, exist_ok=True)
    gravar(dest / "pre_registro_metadados.json", pre)
    if (dest / "selecao.json").exists():
        r = ler(dest / "selecao.json")
        if r["pre_registro"] != pre["hash"]:
            raise ValueError("seleção existente diverge do pré-registro")
        return r
    return obs.selecionar_evento(eventos=pool, saida=dest, pre=pre)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preparar", action="store_true")
    ap.add_argument("--selecionar-segundo", action="store_true")
    ap.add_argument("--humana", type=Path)
    ap.add_argument("--revisor", default="")
    ap.add_argument("--cega", action="store_true")
    ap.add_argument("--saida", type=Path, default=SAIDA)
    a = ap.parse_args()
    if a.selecionar_segundo:
        out = selecionar_segundo(a.saida)
    elif a.humana:
        out = importar_humana(a.humana, saida=a.saida, revisor=a.revisor, cega=a.cega)
    elif a.preparar:
        out = preparar(saida=a.saida)
    else:
        ap.error("escolha --preparar ou --humana CSV --revisor NOME --cega")
    print(json.dumps(out, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
