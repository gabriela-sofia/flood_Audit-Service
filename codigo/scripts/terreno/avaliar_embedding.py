"""B2 -- o embedding medido contra o baseline descritivo, na mesma regua.

-------------------------------------------------------------------------
A MESMA REGUA, DE NOVO LITERALMENTE
-------------------------------------------------------------------------

`avaliar_por_grupo`, `fracao_de_agua`, `pais` e `_spearman` vem importados de
`baseline_descritivo`. A frente A ja tinha aprendido isso: duas copias de uma
regra de metrica divergem, e a divergencia apareceria como "o encoder ganhou"
quando o que mudou foi quem contava.

Fica identico ao baseline:

  o ALVO       fracao de agua do `LabelHand`, so em pixel valido
  o GRUPO      pais, leave-one-country-out
  o ESTIMADOR  ridge fechado, mesmo alpha
  a METRICA    Spearman global e mediano entre paises

O que muda e so a matriz de entrada: 25 estatisticas descritivas de um lado,
`dimensao_embedding` numeros aprendidos do outro.

-------------------------------------------------------------------------
A COMPARACAO E DESIGUAL DE PROPOSITO
-------------------------------------------------------------------------

O embedding tem mais dimensoes que o baseline, e mais dimensoes ajudam um
ridge. Entao o teste esta a favor do encoder -- e se mesmo assim ele nao
superar 0,3821 global / 0,4575 mediano, o resultado negativo e mais forte, e
nao mais fraco.

Um teste que so o modelo novo consegue passar nao prova nada sobre o modelo
novo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))
sys.path.insert(0, str(REPO / "scripts" / "terreno"))

from baseline_descritivo import (  # noqa: E402
    SAIDA, avaliar_por_grupo, fracao_de_agua, pais, vetor_descritivo,
)
from dados_terreno import CAMADAS, nomes_com_terreno  # noqa: E402
from modelo_encoder import AutoencoderMascarado  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402
from treino_encoder import preparar  # noqa: E402


def carregar(rotulo: str = "v1", saida: Path = SAIDA) -> tuple:
    """`(modelo, manifesto)`. O peso nunca vem sozinho."""
    manifesto_path = saida / f"encoder_{rotulo}_manifesto.json"
    peso_path = saida / f"encoder_{rotulo}.pt"
    if not manifesto_path.exists() or not peso_path.exists():
        raise FileNotFoundError(
            f"encoder '{rotulo}' ausente em {saida}; rode o treino primeiro")
    manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))
    modelo = AutoencoderMascarado(canais=len(CAMADAS),
                                  base=manifesto["hiperparametros"]["base"])
    modelo.load_state_dict(torch.load(peso_path, map_location="cpu"))
    modelo.eval()
    return modelo, manifesto


@torch.no_grad()
def embedding_do_chip(modelo, nome: str, estat: dict) -> np.ndarray:
    """Embedding do chip INTEIRO, sem recorte.

    O treino usa recortes de 128; a extracao nao pode, pela mesma razao que a
    avaliacao da frente A roda em chip inteiro -- a unidade da medida tem de
    ser a mesma do baseline, que descreve o chip todo.
    """
    x, _ = preparar(nome, estat)
    return modelo.embedding(x.unsqueeze(0))[0].numpy().astype(np.float64)


def montar_embeddings(modelo, estat: dict,
                      splits: tuple = ("treino", "validacao"),
                      combinar_com_descritivo: bool = False) -> dict:
    """Matriz de embeddings. `combinar_com_descritivo` concatena os dois.

    A combinacao NAO fazia parte do desenho original: ela nasceu do resultado.
    O encoder consertou exatamente os paises em que o baseline falhava e
    quebrou exatamente onde ele ia bem, o que e a assinatura de duas
    representacoes complementares e nao de uma melhor que a outra.

    Testar isso e legitimo e exploratorio, e as duas coisas ficam declaradas:
    a hipotese veio dos dados, e roda no MESMO conjunto de treino e validacao
    em que ela apareceu. O split de teste e o holdout continuam intocados, e e
    so neles que a combinacao poderia virar afirmacao.
    """
    for proibido in ("teste", "bolivia"):
        if proibido in splits:
            raise ValueError(
                f"{proibido!r} nao entra na comparacao com o baseline; ele "
                "existe para medir depois, e olhar antes o gasta")
    X, y, grupos, nomes = [], [], [], []
    for split in splits:
        for nome in nomes_com_terreno(split):
            alvo = fracao_de_agua(nome)
            if alvo is None:
                continue
            vetor = embedding_do_chip(modelo, nome, estat)
            if combinar_com_descritivo:
                vetor = np.concatenate([vetor, vetor_descritivo(nome)])
            X.append(vetor)
            y.append(alvo)
            grupos.append(pais(nome))
            nomes.append(nome)
    return {"X": np.asarray(X), "y": np.asarray(y),
            "grupos": np.asarray(grupos), "nomes": nomes}


def comparar_com_baseline(resultado: dict, saida: Path = SAIDA) -> dict:
    arquivo = saida / "baseline_descritivo.json"
    if not arquivo.exists():
        return {"disponivel": False,
                "por_que": f"baseline ausente: {arquivo}; rode B2-baseline"}
    base = json.loads(arquivo.read_text(encoding="utf-8"))
    comparacao = {"disponivel": True,
                  "baseline": "25 estatisticas descritivas, sem parametro aprendido",
                  "hash_resumo_baseline": base.get("hash_resumo"),
                  "dimensoes_baseline": base.get("dimensoes")}
    for chave in ("spearman_global", "spearman_mediano_entre_grupos"):
        e, b = resultado.get(chave), base.get(chave)
        comparacao[chave] = {
            "encoder": e, "baseline": b,
            "diferenca": round(e - b, 6) if (e is not None and b is not None)
            else None,
        }
    por_pais = {}
    for p, v in resultado.get("spearman_por_grupo", {}).items():
        b = (base.get("spearman_por_grupo") or {}).get(p, {}).get("spearman")
        e = v.get("spearman")
        por_pais[p] = {"encoder": e, "baseline": b,
                       "diferenca": round(e - b, 6)
                       if (e is not None and b is not None) else None}
    comparacao["por_pais"] = por_pais
    comparacao["leitura"] = (
        "diferenca positiva = o embedding ordena melhor que as estatisticas "
        "descritivas; negativa e resultado publicavel, e mais forte porque a "
        "comparacao ja favorece o encoder em numero de dimensoes")
    return comparacao


def executar(rotulo: str = "v1", alpha: float = 1.0,
             combinado: bool = False) -> dict:
    modelo, manifesto = carregar(rotulo)
    dados = montar_embeddings(modelo, manifesto["normalizacao"],
                              combinar_com_descritivo=combinado)
    resultado = avaliar_por_grupo(dados, alpha)
    resumo = {
        "tipo": "avaliacao_embedding_terreno",
        "etapa": "B2",
        "quando_utc": agora_utc(),
        "encoder": rotulo,
        "hash_estado": manifesto["hash_estado"],
        "seed": manifesto["seed"],
        "epocas": manifesto["hiperparametros"]["epocas"],
        "parametros": manifesto["parametros"],
        "dimensoes": int(dados["X"].shape[1]) if len(dados["X"]) else 0,
        "combinado_com_descritivo": combinado,
        "chips": len(dados["nomes"]),
        "usou_rotulo_no_treino": manifesto.get("usou_rotulo"),
        # O encoder e treinado sem rotulo sobre os chips de TREINO de todos os
        # paises; o LOCO deixa um pais fora so da sonda. Entao o pais retirado
        # nunca deu alvo, mas deu TERRENO ao encoder: a avaliacao e
        # transdutiva na representacao e indutiva no alvo. Ate 03/10/2026 isso
        # nao estava escrito em lugar nenhum, e "leave-one-country-out" lia
        # como pais nunca visto.
        "regime": ("transdutivo na representacao: o pre-treino viu, sem "
                   "rotulo, o terreno dos chips de treino do pais que o LOCO "
                   "retira; o alvo desse pais nunca entrou na sonda"),
        **resultado,
        "nao_e": ("previsao de evento: o terreno nao sabe se choveu, e o "
                  "Sen1Floods11 so tem cenas de eventos; a pergunta ja vem "
                  "condicionada a 'dado que choveu forte, a forma ordena?'"),
    }
    resumo["comparacao_com_baseline"] = comparar_com_baseline(resumo)
    resumo["hash_resumo"] = sha256_json(
        {k: v for k, v in resumo.items() if k != "quando_utc"})
    return resumo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotulo", default="v1")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--combinado", action="store_true",
                        help="concatena embedding e descritivo (exploratorio)")
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

    resumo = executar(args.rotulo, args.alpha, args.combinado)
    if args.gravar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        sufixo = "_combinado" if args.combinado else ""
        destino = SAIDA / f"avaliacao_embedding_{args.rotulo}{sufixo}.json"
        destino.write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        resumo["gravado_em"] = str(destino)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
