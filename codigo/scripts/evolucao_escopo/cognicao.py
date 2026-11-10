"""Reavalia recursos locais de recuperacao no corpus atual, sem substituir o servico."""
from __future__ import annotations
from comum import OUT, RUNS, agora, gravar, ler, rel, sha


def conflito_de_estado(afirmacao: dict, registro: dict):
    """Barreira estruturada: a recuperacao textual nao pode promover o resultado."""
    if afirmacao.get("evidencia_id") != registro.get("evidencia_id"):
        return "identidade_divergente"
    if afirmacao.get("fonte_sha256") != registro.get("fonte_sha256"):
        return "versao_divergente"
    if afirmacao.get("estado") != registro.get("estado"):
        return "estado_divergente"
    return None


def executar():
    import numpy as np
    import pandas as pd
    from avaliacao_de_recuperacao import carregar_chunks, montar_conjunto, medir, recuperador_fts5
    from indice_denso import construir
    from comparar_recuperadores import fundir_rrf
    from claims import texto_tem_injecao, sustenta
    from casos_de_suporte import todos
    p = OUT / "cognicao"
    manifest = RUNS / "rag-corpus/manifest.json"
    before = sha(manifest)
    casos = montar_conjunto()
    chunks = carregar_chunks()
    indice = construir()
    # Cada consulta e vetorizada uma vez e comparada em lote; as classes e o
    # filtro de injecao sao aplicados ANTES do ranking. O indice e o mesmo LSA.
    classes = np.asarray(indice.classes)
    limpo = np.asarray([not texto_tem_injecao(t) for t in indice.conteudos])
    consultas = list(dict.fromkeys((c["consulta"], c["classe_autoridade"]) for c in casos))
    rankings = {}
    for inicio in range(0, len(consultas), 64):
        batch = consultas[inicio:inicio + 64]
        vetores = indice.vetorizar([q for q, _ in batch])
        for j, (q, classe) in enumerate(batch):
            sim = indice.vetores @ vetores[j]
            admissivel = (classes == classe) & limpo
            idx = np.flatnonzero(admissivel)
            ordem = idx[np.argsort(-sim[idx], kind="stable")[:20]]
            rankings[(q, classe)] = [{"chunk_id": indice.chunk_ids[i], "similaridade": float(sim[i])} for i in ordem]
        if inicio % 512 == 0:
            print(f"recuperacao: vetorizadas {min(inicio+64, len(consultas))}/{len(consultas)} consultas", flush=True)

    def denso(q, classe, k):
        return rankings[(q, classe)][:k]

    def hibrido(q, classe, k):
        return fundir_rrf([recuperador_fts5(q, classe, max(k*4, 20)), denso(q, classe, max(k*4, 20))], k)

    medidas = {}
    for nome, func in (("lexical", recuperador_fts5), ("lsa", denso), ("hibrido_rrf", hibrido)):
        print(f"recuperacao: medindo {nome} em {len(casos)} casos", flush=True)
        m = medir(func, casos, 5)
        gravar(p / f"casos_{nome}.csv", pd.DataFrame(m.pop("por_caso")))
        medidas[nome] = m
    # O conjunto adversarial preexistente fornece gabarito manual, sem novo
    # julgamento pelo agente ou necessidade de modelo remoto.
    adversarios = todos()
    linhas = []
    for c in adversarios:
        ok, fracao = sustenta(c["claim"], c["texto_do_trecho"])
        linhas.append({**c, "saida_lexica": ok, "fracao_lexica": fracao, "acertou": ok == c["sustenta"]})
    gravar(p / "casos_suporte.json", {"casos": linhas})
    if sha(manifest) != before:
        raise ValueError("corpus mudou durante a avaliacao; repetir em versao nova")
    out = {"quando_utc": agora(), "corpus_manifesto_sha256": before, "chunks": len(chunks), "consultas": len(casos),
           "metrica": "recuperacao do proprio chunk por consulta sintetica literal/morfologica/lacunar; nao julgamento externo de relevancia",
           "medidas": medidas, "casos_adversariais_suporte": len(linhas),
           "suporte_lexico": {"acertos": sum(c["acertou"] for c in linhas),
                              "falsos_positivos": sum(c["saida_lexica"] and not c["sustenta"] for c in linhas),
                              "falsos_negativos": sum(not c["saida_lexica"] and c["sustenta"] for c in linhas)},
           "metodo": "FTS5 existente, LSA char ngram 3-5/d128/seed20260902, RRF c60; k5; sem selecao por resultado",
           "servico_ativo_alterado": False,
           "limite": "benchmark conhecido construido dos chunks; ganho nele nao demonstra compreensao ou prova da tese fisica"}
    gravar(p / "resultado.json", out)
    return out
