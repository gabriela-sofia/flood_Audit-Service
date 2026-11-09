"""Audita os dados e reservas antes de permitir qualquer treino SAR."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dados_pares_geograficos import REPO, inventariar, particao, sha
from experimento_pre_pos import gravar_json


def auditar(raiz, saida):
    raiz, saida = Path(raiz), Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    aquisicao = json.loads((raiz / "aquisicao.json").read_text(encoding="utf-8"))
    from adquirir_pares_geograficos import ARQUIVOS, hashes
    if set(aquisicao["arquivos"]) != set(ARQUIVOS):
        raise ValueError("aquisicao ainda incompleta")
    for nome, (tamanho, md5) in ARQUIVOS.items():
        arquivo = raiz / nome
        hm, hs = hashes(arquivo)
        if arquivo.stat().st_size != tamanho or hm != md5 or hs != aquisicao["arquivos"][nome]["sha256"]:
            raise ValueError("arquivo publicado divergente")
    if sha(raiz / "Sen1Floods11_Metadata.geojson") != aquisicao["metadados_eventos"]["sha256"]:
        raise ValueError("catalogo divergente da aquisicao")
    inv = inventariar(raiz)
    eventos = sorted(inv["eventos"], key=lambda e: hashlib.sha256(e.encode()).hexdigest())
    cortes, recusas = [], list(inv["recusados"])
    if not recusas:
        for i, evento in enumerate(eventos):
            va = [eventos[(i+1) % len(eventos)], eventos[(i+2) % len(eventos)]]
            try:
                treino, validacao, teste = particao(inv["pares"], [evento], va)
                cortes.append({"teste": evento, "validacao": va,
                               "n_treino": len(treino), "n_validacao": len(validacao),
                               "n_teste": len(teste), "vazamento_detectado": False})
            except ValueError as erro:
                recusas.append({"evento_teste": evento, "motivo": str(erro)})
    resultado = {"inventario": inv, "cortes": cortes, "recusas": recusas,
                 "admissivel": bool(cortes) and not recusas,
                 "treino_executado": False, "ganho_comprovado": False,
                 "promovido": False}
    gravar_json(saida, resultado)
    print(json.dumps({k: resultado[k] for k in ("admissivel", "treino_executado", "ganho_comprovado")}, ensure_ascii=False))
    print(json.dumps({"pares": len(inv["pares"]), "eventos": len(eventos),
                      "cortes": len(cortes), "excluidos_sem_par": len(inv["excluidos_sem_par"]),
                      "recusas": recusas}, ensure_ascii=False))
    return resultado


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raiz", type=Path, default=REPO / "local_runs/sar-independencia/dados")
    p.add_argument("--saida", type=Path, default=REPO / "local_runs/sar-independencia/auditoria.json")
    a = p.parse_args()
    r = auditar(a.raiz.resolve(), a.saida.resolve())
    if not r["admissivel"]:
        raise SystemExit(2)
