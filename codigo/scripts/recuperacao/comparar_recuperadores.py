"""C3 e C6 -- lexical, denso e hibrido no mesmo conjunto.

-------------------------------------------------------------------------
HIBRIDO, E NAO SUBSTITUICAO
-------------------------------------------------------------------------

O FTS5 nao vai ser trocado. Ele acerta 0,997 no literal e no lacunar, e a
cadeia de citacao do REV-P depende de literalidade: DOI, URL, nome proprio e
codigo sao exatamente onde a busca exata ganha de qualquer vetor. Trocar o
lexical pelo denso resolveria a morfologia e quebraria a condicao 3 da cadeia
de citacao -- o identificador aparecer LITERALMENTE na fonte.

Entao o denso ENTRA, e o hibrido e a forma de entrar.

A fusao e por posicao reciproca (RRF): cada lista contribui `1/(c + posicao)`,
e as contribuicoes somam. Escolhida por uma razao pratica e uma cientifica. A
pratica: nao exige que as duas pontuacoes estejam na mesma escala, e um score
de FTS5 e um cosseno nao estao. A cientifica: um documento que aparece em
QUALQUER uma das listas em posicao boa sobe, que e exatamente o comportamento
desejado quando as duas representacoes sao complementares -- como a frente B
mostrou serem, para terreno.

-------------------------------------------------------------------------
A CLASSE DE AUTORIDADE CONTINUA MANDANDO (C4)
-------------------------------------------------------------------------

Os tres recuperadores recebem a mesma `classe` e filtram por ela. Um indice
denso nao pode furar `classe nao sobrescreve classe` por ser denso, e um teste
verifica que nao fura.

-------------------------------------------------------------------------
A MESMA REGUA, PELA TERCEIRA VEZ
-------------------------------------------------------------------------

`medir`, `montar_conjunto` e `NIVEIS` vem importados de
`avaliacao_de_recuperacao`. Frente A importou a metrica do baseline; frente B
importou `avaliar_por_grupo`; aqui e a mesma decisao, pelo mesmo motivo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "recuperacao"))

from avaliacao_de_recuperacao import (  # noqa: E402
    NIVEIS, SAIDA, medir, montar_conjunto, recuperador_fts5,
)
from indice_denso import construir  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

C_RRF = 60          # constante classica do RRF; nao ajustada aos dados


def fundir_rrf(listas: list, k: int, c: int = C_RRF) -> list[dict]:
    """Reciprocal rank fusion. Nao usa score, so posicao."""
    pontos: dict = {}
    origem: dict = {}
    for lista in listas:
        for posicao, item in enumerate(lista, start=1):
            cid = item.get("chunk_id")
            if cid is None:
                continue
            pontos[cid] = pontos.get(cid, 0.0) + 1.0 / (c + posicao)
            origem.setdefault(cid, {})[len(origem.get(cid, {}))] = posicao
    ordenados = sorted(pontos.items(), key=lambda kv: -kv[1])
    return [{"chunk_id": cid, "rrf": round(p, 8)} for cid, p in ordenados[:k]]


def fazer_recuperadores(indice) -> dict:
    def denso(consulta: str, classe: str, k: int) -> list[dict]:
        return indice.buscar(consulta, classe=classe, k=k)

    def hibrido(consulta: str, classe: str, k: int) -> list[dict]:
        # Cada lado devolve mais que `k` antes da fusao: fundir duas listas
        # ja cortadas em k perderia justamente o documento que um dos lados
        # colocou em k+1 e o outro em 1.
        largura = max(k * 4, 20)
        return fundir_rrf([recuperador_fts5(consulta, classe, largura),
                           denso(consulta, classe, largura)], k)

    return {"lexical_fts5": recuperador_fts5, "denso_lsa": denso,
            "hibrido_rrf": hibrido}


def executar(k: int = 5) -> dict:
    casos = montar_conjunto()
    indice = construir()
    medidas = {}
    for nome, recuperador in fazer_recuperadores(indice).items():
        m = medir(recuperador, casos, k)
        medidas[nome] = {c: v for c, v in m.items() if c != "por_caso"}

    base = medidas["lexical_fts5"]
    comparacao = {}
    for nome, m in medidas.items():
        if nome == "lexical_fts5":
            continue
        linha = {"agregado": {
            "recall_em_k": m["agregado"]["recall_em_k"],
            "lexical": base["agregado"]["recall_em_k"],
            "diferenca": round(m["agregado"]["recall_em_k"]
                               - base["agregado"]["recall_em_k"], 6)}}
        for nivel in NIVEIS:
            a, b = m["por_nivel"][nivel], base["por_nivel"][nivel]
            linha[nivel] = {
                "recall_em_k": a["recall_em_k"], "lexical": b["recall_em_k"],
                "diferenca": round(a["recall_em_k"] - b["recall_em_k"], 6)
                if (a["recall_em_k"] is not None
                    and b["recall_em_k"] is not None) else None}
        comparacao[nome] = linha

    resumo = {
        "tipo": "comparacao_de_recuperadores",
        "etapa": "C3/C6",
        "quando_utc": agora_utc(),
        "k": k,
        "casos": len(casos),
        "recuperadores": list(medidas),
        "medidas": medidas,
        "comparacao_com_lexical": comparacao,
        "fusao": f"reciprocal rank fusion, c={C_RRF}, nao ajustado aos dados",
        "nao_e": ("substituicao do FTS5: a cadeia de citacao depende de "
                  "literalidade, e busca exata e onde o lexical ganha de "
                  "qualquer vetor"),
    }
    resumo["hash_resumo"] = sha256_json(
        {c: v for c, v in resumo.items() if c != "quando_utc"})
    return resumo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    # Gravar e o PADRAO. A forma anterior (`--gravar` opcional) criava
    # artefato que envelhece em silencio: em 22/09/2026 o `casos.json` do
    # produto estava desatualizado desde antes da migracao para parquet e um
    # teste passou contra ele por dias. `--gravar` segue aceito para nao
    # quebrar chamador existente.
    parser.add_argument("--gravar", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--nao-gravar", dest="gravar", action="store_false",
                        help="nao escreve artefato; so imprime o resumo")
    args = parser.parse_args()

    resumo = executar(args.k)
    if args.gravar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        destino = SAIDA / f"comparacao_recuperadores_k{args.k}.json"
        destino.write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        resumo["gravado_em"] = str(destino)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
