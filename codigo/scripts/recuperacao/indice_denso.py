"""C2 -- indice denso do corpus, e por que ele NAO e um bi-encoder neural.

-------------------------------------------------------------------------
O QUE ESTA MAQUINA PERMITE HOJE, dito antes de qualquer numero
-------------------------------------------------------------------------

O plano previa um bi-encoder. Ele nao esta disponivel aqui, e as duas razoes
sao verificaveis:

  - o Ollama local esta no ar, mas foi iniciado sem `--embeddings`, e responde
    "This server does not support embeddings" a qualquer chamada de vetor;
  - `transformers` esta instalado e nenhum modelo de sentenca esta em disco;
    baixar um exigiria rede e autorizacao que esta sessao nao tem.

Entao isto E: TF-IDF sobre n-gramas de CARACTERES, reduzido por SVD. Denso,
deterministico, offline, aprendido do proprio corpus. E isto NAO E: um modelo
neural de linguagem. Chamar de "embedding neural" seria a mesma categoria de
erro que o projeto persegue na LLM -- nomear melhor do que e.

-------------------------------------------------------------------------
POR QUE N-GRAMA DE CARACTERE, e nao de palavra
-------------------------------------------------------------------------

O defeito medido em C1 e morfologico: recall@5 de 0,997 no literal contra
0,119 quando singular e plural trocam. TF-IDF de PALAVRAS nao resolveria nada
disso -- "favela" e "favelas" sao dois termos distintos para ele, exatamente
como sao para o FTS5.

N-gramas de caractere resolvem por construcao: "favela" e "favelas"
compartilham quase todos os trigramas e quadrigramas. A variacao que quebra o
indice lexical e justamente a que quase nao muda o vetor de caracteres.

-------------------------------------------------------------------------
A LICAO DA FRENTE B, DE NOVO
-------------------------------------------------------------------------

La, uma representacao aprendida DO DOMINIO superou onde a descricao global
falhava, e falhou onde ela bastava -- e as duas juntas superaram cada uma.
A hipotese aqui e a mesma, e a frente C ja nasce com o hibrido previsto em C3
em vez de descobri-lo depois.

O SVD e ajustado SO no corpus indexado, que e o mesmo universo que sera
consultado. Nao ha vazamento possivel porque nao ha conjunto de teste separado
aqui: o alvo da recuperacao e o proprio chunk, e o indice existe para
encontra-lo.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "recuperacao"))

from avaliacao_de_recuperacao import SAIDA, carregar_chunks  # noqa: E402
from claims import texto_tem_injecao  # noqa: E402
from proveniencia import agora_utc, sha256_json, sha256_texto  # noqa: E402

DIMENSOES = 128
NGRAMA = (3, 5)


@dataclass
class IndiceDenso:
    """Vetores do corpus e o transformador que os produziu."""

    chunk_ids: list
    classes: list
    conteudos: list              # o texto, para o filtro de injecao
    vetores: np.ndarray          # (n, d) ja normalizados em L2
    vetorizador: object
    svd: object
    dimensoes: int
    ultimos_descartados_por_injecao: list = field(default_factory=list)

    def buscar(self, consulta: str, classe: str | None = None,
               k: int = 5, com_injecao: bool = False) -> list[dict]:
        """Vizinhos por cosseno, com as MESMAS duas garantias do `rag_busca`.

        A primeira e o filtro de classe, aplicado depois da similaridade:
        `classe nao sobrescreve classe` e regra de autoridade, e um indice
        denso nao pode furar regra de autoridade por ser denso.

        A segunda e a deteccao de injecao, e ela quase ficou de fora. A versao
        anterior lia os `chunk_id` direto do SQLite e devolvia o vizinho mais
        proximo -- sem passar pelo `texto_tem_injecao` que o `rag_busca` aplica
        na recuperacao. O efeito seria um caminho novo pelo qual um chunk que
        fala em imperativo com o sistema chega ao contexto, contornando uma
        barreira que existe desde a camada cognitiva.

        Recuperador novo herda garantia existente. Nao herdar e o modo de falha
        mais silencioso que uma frente de infraestrutura pode introduzir: nada
        quebra, a metrica melhora, e uma trava deixa de valer.
        """
        v = self.vetorizar([consulta])[0]
        sim = self.vetores @ v
        ordem = np.argsort(-sim)
        saida, descartados = [], []
        for i in ordem:
            if classe is not None and self.classes[i] != classe:
                continue
            achado = texto_tem_injecao(self.conteudos[i] or "")
            if achado and not com_injecao:
                descartados.append({"chunk_id": self.chunk_ids[i],
                                    "injecao_detectada": achado})
                continue
            saida.append({"chunk_id": self.chunk_ids[i],
                          "classe_autoridade": self.classes[i],
                          "similaridade": round(float(sim[i]), 6)})
            if len(saida) >= k:
                break
        self.ultimos_descartados_por_injecao = descartados
        return saida

    def vetorizar(self, textos: list) -> np.ndarray:
        bruto = self.vetorizador.transform(textos)
        reduzido = self.svd.transform(bruto)
        normas = np.linalg.norm(reduzido, axis=1, keepdims=True)
        normas[normas == 0] = 1.0
        return reduzido / normas


def construir(dimensoes: int = DIMENSOES, ngrama: tuple = NGRAMA,
              seed: int = 20260902) -> IndiceDenso:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    chunks = carregar_chunks()
    textos = [c["conteudo"] for c in chunks]

    vetorizador = TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngrama, min_df=2,
        sublinear_tf=True, lowercase=True, strip_accents="unicode")
    matriz = vetorizador.fit_transform(textos)

    # `n_components` nao pode passar do posto da matriz; com corpus pequeno
    # isso acontece, e falhar por isso seria falhar por um detalhe de tamanho.
    d = min(dimensoes, min(matriz.shape) - 1)
    svd = TruncatedSVD(n_components=d, random_state=seed)
    reduzido = svd.fit_transform(matriz)
    normas = np.linalg.norm(reduzido, axis=1, keepdims=True)
    normas[normas == 0] = 1.0

    return IndiceDenso(
        chunk_ids=[c["chunk_id"] for c in chunks],
        classes=[c["classe_autoridade"] for c in chunks],
        conteudos=[c["conteudo"] for c in chunks],
        vetores=reduzido / normas,
        vetorizador=vetorizador, svd=svd, dimensoes=d)


def manifesto(indice: IndiceDenso, seed: int = 20260902) -> dict:
    """Proveniencia do indice. Ele nao tem peso treinado, e tem estado."""
    m = {
        "tipo": "indice_denso_char_ngram_lsa",
        "etapa": "C2",
        "quando_utc": agora_utc(),
        "nao_e": ("bi-encoder neural: e TF-IDF de n-gramas de caractere "
                  "reduzido por SVD, deterministico e offline"),
        "por_que_nao_neural": (
            "o Ollama local foi iniciado sem `--embeddings` e recusa vetores; "
            "nenhum modelo de sentenca esta em disco e baixar exigiria rede"),
        "chunks": len(indice.chunk_ids),
        "dimensoes": indice.dimensoes,
        "ngrama": list(NGRAMA),
        "analisador": "char_wb",
        "seed": seed,
        "variancia_explicada": round(
            float(indice.svd.explained_variance_ratio_.sum()), 6),
        "hash_vetores": sha256_texto(
            "|".join(f"{i}:{round(float(v), 6)}"
                     for i, v in enumerate(indice.vetores[:, 0].tolist()))),
    }
    m["hash_manifesto"] = sha256_json(
        {k: v for k, v in m.items() if k != "quando_utc"})
    return m


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensoes", type=int, default=DIMENSOES)
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

    indice = construir(args.dimensoes)
    m = manifesto(indice)
    if args.gravar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        destino = SAIDA / "indice_denso_manifesto.json"
        destino.write_text(json.dumps(m, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        m["gravado_em"] = str(destino)
    print(json.dumps(m, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
