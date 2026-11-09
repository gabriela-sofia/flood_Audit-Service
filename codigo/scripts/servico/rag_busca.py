"""
Consulta o corpus RAG local do REV-P.

Devolve TRECHO RECUPERADO, e nao resposta. A diferenca importa: quem responde
e a camada de sintese, e ela so pode citar o que saiu daqui. Nada que este
modulo nao devolva pode ser citado -- inclusive uma referencia que o modelo
"conhece". Ver `rag_verificador_citacao`.
"""

from __future__ import annotations


def _declarar_metadados_de_saida(dados):
    """Metadados de procedencia na copia serializada; nao modifica o objeto de calculo."""
    if not isinstance(dados, dict):
        return dados
    return {"tipo": "registro_rag_busca",
            "gerado_por": 'scripts/servico/rag_busca.py',
            "nao_e": "Registro do produtor rag_busca: declarar a procedencia nao revalida resultados nem autoriza promocao; conservar os estados e limites especificos do artefato.",
            **dados}


import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from claims import VAZIAS, sem_acento, texto_tem_injecao  # noqa: E402
from proveniencia import agora_utc  # noqa: E402
from rag_corpus import CLASSES, MANIFEST, construir  # noqa: E402

OUT = REPO / "local_runs" / "rag-consulta"

CAMPOS = ("chunk_id", "source_id", "source_path", "titulo", "secao",
          "tipo_fonte", "classe_autoridade", "hash_fonte", "hash_chunk",
          "doi", "url")


def _termos(query: str) -> list:
    """Termos de CONTEUDO da pergunta.

    Sem tirar as palavras vazias, "o que e HAND" busca por "que" -- que
    aparece em quase todo chunk do corpus -- e o ranking passa a ser sobre a
    gramatica da pergunta em vez do assunto dela.
    """
    return [t for t in re.findall(r"[A-Za-zÀ-ÿ0-9_]+", sem_acento(query or ""))
            if len(t) >= 3 and t not in VAZIAS]


def _fts_query(query: str) -> str:
    termos = _termos(query)
    return " OR ".join(f'"{t}"' for t in termos) if termos else '""'


def buscar(query: str, limite: int = 5, manifest_path: Path = MANIFEST,
           classe: str | None = None, com_injecao: bool = False) -> dict:
    """Recupera trechos reais do corpus.

    `classe` restringe a autoridade consultada. Nao e filtro de conveniencia:
    e como a regra "classe nao sobrescreve classe" e aplicada na pratica --
    uma pergunta sobre o ESTADO do REV-P consulta `projeto`, uma pergunta
    metodologica consulta `cientifica`.

    `com_injecao=False` (o padrao) NAO devolve trecho que tenta instruir o
    sistema. Ele nao e sanitizado nem reescrito: sai do resultado e aparece em
    `descartados_por_injecao`, para que a auditoria veja o que foi tirado.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        construir(manifest_path.parent)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if classe is not None and classe not in CLASSES:
        raise ValueError(f"classe de autoridade desconhecida: {classe!r}; "
                         f"conhecidas: {list(CLASSES)}")

    db = Path(manifest["db"])
    if not db.exists():                     # manifesto sem indice nao busca
        construir(manifest_path.parent)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        db = Path(manifest["db"])

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    colunas = ", ".join(CAMPOS)
    # Pede folgado e corta depois: o filtro de classe e o descarte por
    # injecao acontecem em Python, e cortar no SQL devolveria menos trechos
    # que o limite sempre que algum fosse descartado.
    bruto = max(limite * 4, 20)
    if manifest["indice"] == "fts5":
        rows = con.execute(
            f"SELECT {colunas}, "
            "snippet(chunks, -1, '[', ']', ' ... ', 32) AS trecho, "
            "conteudo, bm25(chunks) AS rank FROM chunks WHERE chunks MATCH ? "
            "ORDER BY rank LIMIT ?", (_fts_query(query), bruto)).fetchall()
    else:
        termos = _termos(query)
        where = " AND ".join(["lower(conteudo) LIKE ?"] * len(termos)) or "1=1"
        rows = con.execute(
            f"SELECT {colunas}, substr(conteudo, 1, 500) AS trecho, "
            f"conteudo, 0 AS rank FROM chunks WHERE {where} LIMIT ?",
            [f"%{t}%" for t in termos] + [bruto]).fetchall()
    con.close()

    resultados, descartados = [], []
    for row in rows:
        item = dict(row)
        if classe is not None and item["classe_autoridade"] != classe:
            continue
        achado = texto_tem_injecao(item.get("conteudo") or "")
        if achado and not com_injecao:
            descartados.append({**{k: item[k] for k in ("chunk_id", "titulo")},
                                "injecao_detectada": achado})
            continue
        item.pop("conteudo", None)
        resultados.append(item)
        if len(resultados) >= limite:
            break

    return {
        "query": query,
        "classe_consultada": classe,
        "quando_utc": agora_utc(),
        "manifest": str(manifest_path),
        "hash_corpus": manifest.get("hash_corpus"),
        "resultados": resultados,
        "descartados_por_injecao": descartados,
        "politica": ("sem resultado significa nao encontrado no corpus local; "
                     "trecho nao recuperado nao pode ser citado"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--limite", type=int, default=5)
    parser.add_argument("--classe", choices=list(CLASSES), default=None)
    args = parser.parse_args()
    resultado = buscar(args.query, args.limite, classe=args.classe)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ultima_busca.json").write_text(
        json.dumps(_declarar_metadados_de_saida(resultado), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
