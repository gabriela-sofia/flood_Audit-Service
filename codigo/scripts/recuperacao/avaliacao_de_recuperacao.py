"""C1 -- o conjunto de avaliacao da recuperacao, e o baseline medido nele.

-------------------------------------------------------------------------
BASELINE ANTES DO MODELO, PELA TERCEIRA VEZ
-------------------------------------------------------------------------

Na frente A o baseline veio dentro do dataset. Na frente B foi construido com
25 estatisticas descritivas. Aqui ele ja existe e esta servindo: o FTS5 do
`rag_busca`. O que falta e medi-lo -- e medir ANTES de existir alternativa,
porque um numero de referencia obtido depois de ver o concorrente ja nasce
escolhido.

-------------------------------------------------------------------------
DE ONDE VEM A CONSULTA, sem anotacao humana e sem circularidade
-------------------------------------------------------------------------

Nao ha conjunto anotado de perguntas com o chunk certo, e escrever um a mao
seria eu decidindo o gabarito e depois medindo contra ele. A saida e a padrao
em recuperacao: a consulta e derivada DO PROPRIO CHUNK, e o acerto e
reencontra-lo.

Tres niveis, e o segundo e o que importa:

  literal      um trecho contiguo, copiado do chunk. Mede o piso: se o indice
               nao acha o texto exato, nao acha nada.

  morfologica  o mesmo trecho com singular e plural trocados. Mede exatamente
               o defeito observado em 2026-09-01, quando "favela comunidade
               urbana" devolveu 0 trechos e "favelas e comunidades urbanas"
               devolveu a fonte certa.

  lacunar      termos de conteudo do chunk, sem ordem e sem as palavras de
               ligacao. Mede a pergunta real de um usuario, que nunca cita o
               documento literalmente.

Isso mede robustez de forma, e nao relevancia semantica -- e a diferenca fica
declarada. Um conjunto que medisse relevancia exigiria julgamento humano, e
julgamento humano nao cabe nesta sessao.

-------------------------------------------------------------------------
A LICAO DA FRENTE B, APLICADA ANTES DE OLHAR O AGREGADO
-------------------------------------------------------------------------

Na frente B o encoder empatou no global e a historia estava toda na
estratificacao por pais. Aqui o agregado tambem pode esconder tudo: o corpus
tem 298 chunks de `projeto` contra 20 de `cientifica`, e um recall@k global
seria dominado por `projeto`.

Entao toda metrica sai por NIVEL e por CLASSE DE AUTORIDADE, e o agregado vem
junto so para completar -- nunca sozinho.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from claims import sem_acento  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402
from rag_busca import buscar  # noqa: E402
from rag_corpus import CLASSES, MANIFEST  # noqa: E402

SAIDA = REPO / "local_runs" / "recuperacao"

LITERAL = "literal"
MORFOLOGICA = "morfologica"
LACUNAR = "lacunar"
NIVEIS = (LITERAL, MORFOLOGICA, LACUNAR)

# Palavras de ligacao saem da consulta lacunar. Lista curta e declarada: uma
# lista grande demais viraria um segundo modelo escondido dentro do conjunto
# de avaliacao.
LIGACAO = {
    "a", "as", "o", "os", "um", "uma", "de", "do", "da", "dos", "das", "e",
    "em", "no", "na", "nos", "nas", "por", "para", "com", "que", "se", "ao",
    "aos", "sem", "sob", "sobre", "entre", "ou", "nao", "mais", "menos", "ja",
    "the", "of", "and", "to", "in", "is", "are", "for",
}

PALAVRA = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]{2,}")


def _palavras(texto: str) -> list[str]:
    return PALAVRA.findall(texto or "")


def _trocar_numero(palavra: str) -> str:
    """Singular <-> plural, do jeito grosseiro que o portugues permite.

    Grosseiro de proposito e declarado como tal: o objetivo nao e conjugar
    certo, e produzir a MESMA classe de variacao que quebrou a busca real --
    "favelas" contra "favela".
    """
    baixa = palavra.lower()
    if baixa.endswith("oes"):
        return palavra[:-3] + "ao"
    if baixa.endswith("ais"):
        return palavra[:-3] + "al"
    if baixa.endswith("ns"):
        return palavra[:-2] + "m"
    if baixa.endswith("s") and len(baixa) > 3:
        return palavra[:-1]
    if baixa.endswith("ao"):
        return palavra[:-2] + "oes"
    if baixa.endswith(("m",)):
        return palavra + "s"
    return palavra + "s"


def consultas_do_chunk(texto: str, n_termos: int = 8,
                       semente: int = 0) -> dict[str, str]:
    """Uma consulta por nivel, deterministica dado o texto.

    Nao usa gerador aleatorio: a posicao do trecho sai de `semente` combinada
    ao tamanho do texto. Consulta que muda a cada execucao produziria um
    baseline que nao se pode comparar com o proximo.
    """
    palavras = _palavras(texto)
    if len(palavras) < n_termos:
        return {}

    inicio = (semente * 7) % max(len(palavras) - n_termos, 1)
    trecho = palavras[inicio:inicio + n_termos]

    conteudo = [p for p in palavras if sem_acento(p.lower()) not in LIGACAO]
    if len(conteudo) < 4:
        return {}
    passo = max(len(conteudo) // 6, 1)
    lacunar = conteudo[::passo][:6]

    return {
        LITERAL: " ".join(trecho),
        MORFOLOGICA: " ".join(
            _trocar_numero(p) if sem_acento(p.lower()) not in LIGACAO else p
            for p in trecho),
        LACUNAR: " ".join(lacunar),
    }


def carregar_chunks(manifest_path: Path = MANIFEST) -> list[dict]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    con = sqlite3.connect(Path(manifest["db"]))
    con.row_factory = sqlite3.Row
    linhas = con.execute(
        "SELECT chunk_id, source_id, classe_autoridade, conteudo FROM chunks"
    ).fetchall()
    con.close()
    return [dict(r) for r in linhas]


def montar_conjunto(manifest_path: Path = MANIFEST,
                    n_termos: int = 8) -> list[dict]:
    """Um caso por chunk e por nivel. O gabarito e o proprio `chunk_id`."""
    casos = []
    for i, chunk in enumerate(carregar_chunks(manifest_path)):
        consultas = consultas_do_chunk(chunk["conteudo"], n_termos, semente=i)
        for nivel, consulta in consultas.items():
            casos.append({
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "classe_autoridade": chunk["classe_autoridade"],
                "nivel": nivel,
                "consulta": consulta,
            })
    return casos


def _posicao(resultados: list, alvo: str) -> int | None:
    for i, r in enumerate(resultados, start=1):
        if r.get("chunk_id") == alvo:
            return i
    return None


def medir(recuperador, casos: list[dict], k: int = 5) -> dict:
    """`recuperador(consulta, classe, k) -> lista de dicts com `chunk_id`.

    Assinatura fechada de proposito: o baseline e qualquer alternativa entram
    pelo mesmo buraco, e a comparacao nao pode depender de quem chama.
    """
    por_caso = []
    for caso in casos:
        resultados = recuperador(caso["consulta"], caso["classe_autoridade"], k)
        pos = _posicao(resultados, caso["chunk_id"])
        por_caso.append({**caso, "posicao": pos,
                         "acertou_em_k": pos is not None,
                         "recuperados": len(resultados)})

    def agregar(subconjunto: list[dict]) -> dict:
        n = len(subconjunto)
        if n == 0:
            return {"n": 0, "recall_em_k": None, "mrr": None,
                    "consultas_sem_resultado": 0}
        acertos = sum(1 for c in subconjunto if c["acertou_em_k"])
        mrr = sum(1.0 / c["posicao"] for c in subconjunto if c["posicao"])
        return {
            "n": n,
            "recall_em_k": round(acertos / n, 6),
            "mrr": round(mrr / n, 6),
            "consultas_sem_resultado": sum(1 for c in subconjunto
                                           if c["recuperados"] == 0),
        }

    return {
        "k": k,
        "agregado": agregar(por_caso),
        "por_nivel": {nivel: agregar([c for c in por_caso
                                      if c["nivel"] == nivel])
                      for nivel in NIVEIS},
        "por_classe": {classe: agregar([c for c in por_caso
                                        if c["classe_autoridade"] == classe])
                       for classe in CLASSES},
        "por_nivel_e_classe": {
            f"{nivel}|{classe}": agregar(
                [c for c in por_caso
                 if c["nivel"] == nivel and c["classe_autoridade"] == classe])
            for nivel in NIVEIS for classe in CLASSES},
        "por_caso": por_caso,
    }


def recuperador_fts5(consulta: str, classe: str, k: int) -> list[dict]:
    """O que esta servindo hoje. Nenhum ajuste, nenhuma reescrita de consulta."""
    try:
        return buscar(consulta, limite=k, classe=classe)["resultados"]
    except Exception:
        # Consulta que o FTS5 rejeita conta como zero resultados, e nao como
        # erro do conjunto: um indice que engasga com a consulta do usuario
        # esta falhando de verdade.
        return []


def executar(k: int = 5) -> dict:
    casos = montar_conjunto()
    medida = medir(recuperador_fts5, casos, k)
    resumo = {
        "tipo": "baseline_recuperacao_fts5",
        "etapa": "C1",
        "quando_utc": agora_utc(),
        "casos": len(casos),
        "niveis": list(NIVEIS),
        "o_que_mede": ("robustez de forma: o indice reencontra o chunk a "
                       "partir de um trecho dele, com e sem variacao"),
        # `nao_e` e o nome canonico da limitacao no projeto, e o coletor le
        # exatamente esse campo. Sob qualquer outro nome a limitacao existia no
        # arquivo e sumia no registro -- o numero viajava e ela ficava.
        "nao_e": ("medida de relevancia semantica: isso exigiria julgamento "
                  "humano, que nao entra num conjunto gerado"),
        **{c: v for c, v in medida.items() if c != "por_caso"},
    }
    resumo["hash_resumo"] = sha256_json(
        {c: v for c, v in resumo.items() if c != "quando_utc"})
    return {"resumo": resumo, "por_caso": medida["por_caso"]}


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

    resultado = executar(args.k)
    if args.gravar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        destino = SAIDA / f"baseline_fts5_k{args.k}.json"
        destino.write_text(
            json.dumps(resultado["resumo"], indent=2, ensure_ascii=False),
            encoding="utf-8")
        resultado["resumo"]["gravado_em"] = str(destino)
    print(json.dumps(resultado["resumo"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
