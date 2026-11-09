"""B2-baseline -- o vetor descritivo que o encoder tera de superar.

-------------------------------------------------------------------------
A LICAO DA FRENTE A, APLICADA ANTES DE TREINAR
-------------------------------------------------------------------------

Na frente A o baseline veio dentro do dataset e por isso era indiscutivel. Aqui
nao ha baseline pronto, entao ele e construido primeiro -- e construido para
ser DIFICIL, nao para ser batido.

O baseline e um vetor de estatisticas descritivas das cinco camadas: media,
desvio, e os percentis 10, 50 e 90, so em pixel valido. Vinte e cinco numeros
sem nenhum parametro aprendido.

Se um encoder auto-supervisionado nao superar isso, ele nao aprendeu
representacao -- ele aprendeu a recomprimir a media. E esse resultado negativo
vale tanto quanto o positivo, do mesmo jeito que os tres A/B nulos do DINOv2
valeram.

-------------------------------------------------------------------------
A TAREFA DE AVALIACAO, e o que ela pode e nao pode dizer
-------------------------------------------------------------------------

Pergunta: a representacao do terreno carrega informacao sobre QUANTA agua foi
observada naquele chip?

O alvo e a fracao de agua do `LabelHand` -- o mesmo rotulo manual da frente A,
o que faz das duas frentes medidas comparaveis. A metrica e Spearman, porque
interessa a ORDEM (este chip acumula mais que aquele), nao o valor.

O que isto NAO e: previsao de evento. O terreno nao sabe se choveu. O que ele
pode carregar e predisposicao -- e o Sen1Floods11 so contem cenas de eventos de
inundacao, entao a pergunta ja vem condicionada a "dado que choveu forte na
regiao, a forma do terreno ordena onde a agua ficou?".

Esse condicionamento e uma limitacao e tambem e o que torna a pergunta
respondivel. Ele fica declarado em toda saida.

-------------------------------------------------------------------------
AGRUPAMENTO POR PAIS
-------------------------------------------------------------------------

A validacao cruzada agrupa por pais, e nao por chip. Chips do mesmo evento
compartilham bacia, data e regime; separa-los aleatoriamente deixaria o
vizinho do treino responder pelo do teste, e o numero subiria sem que nada
denunciasse. Mesma disciplina de `unidade_de_validacao` do resto do projeto.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))
sys.path.insert(0, str(REPO / "scripts" / "terreno"))

from dados_sen1floods11 import ler_chip  # noqa: E402
from dados_terreno import CAMADAS, ler_terreno, nomes_com_terreno  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "terreno-encoder"

ESTATISTICAS = ("media", "desvio", "p10", "p50", "p90")


def vetor_descritivo(nome: str) -> np.ndarray:
    """25 numeros: 5 camadas x 5 estatisticas, so em pixel valido.

    Pixel invalido fora da conta pela mesma razao do `-1` da frente A: a media
    de um vetor que inclui sentinela nao descreve terreno nenhum.
    """
    chip = ler_terreno(nome)
    saida = []
    for i in range(len(CAMADAS)):
        vals = chip.pilha[i][chip.valido]
        if vals.size == 0:
            saida.extend([0.0] * len(ESTATISTICAS))
            continue
        saida.extend([
            float(vals.mean()), float(vals.std()),
            float(np.percentile(vals, 10)),
            float(np.percentile(vals, 50)),
            float(np.percentile(vals, 90)),
        ])
    return np.asarray(saida, dtype=np.float64)


def nomes_das_dimensoes() -> list[str]:
    return [f"{c}_{e}" for c in CAMADAS for e in ESTATISTICAS]


def fracao_de_agua(nome: str) -> float | None:
    """O alvo: quanta agua o rotulo manual viu, so em pixel valido.

    Devolve `None` quando nao ha pixel valido -- e `None` sai da avaliacao em
    vez de virar zero, porque zero aqui significaria "chip seco observado".
    """
    chip = ler_chip(nome)
    n = int(chip.valido.sum())
    if n == 0:
        return None
    return float((chip.agua & chip.valido).sum() / n)


def pais(nome: str) -> str:
    return nome.split("_")[0]


def montar(splits: tuple = ("treino", "validacao")) -> dict:
    """Matriz de features, alvo e grupo. Sem tocar em teste nem no holdout."""
    for proibido in ("teste", "bolivia"):
        if proibido in splits:
            raise ValueError(
                f"{proibido!r} nao entra na construcao do baseline; ele existe "
                "para medir depois, e olhar antes o gasta")
    X, y, grupos, nomes = [], [], [], []
    for split in splits:
        for nome in nomes_com_terreno(split):
            alvo = fracao_de_agua(nome)
            if alvo is None:
                continue
            X.append(vetor_descritivo(nome))
            y.append(alvo)
            grupos.append(pais(nome))
            nomes.append(nome)
    return {"X": np.asarray(X), "y": np.asarray(y),
            "grupos": np.asarray(grupos), "nomes": nomes}


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Correlacao de ordem, sem depender de scipy.

    A checagem de variacao e feita nos VALORES, e nao nos ranks -- e a
    diferenca nao e cosmetica. `argsort` de um vetor constante devolve
    `[0,1,2,...]`, com desvio diferente de zero, e a correlacao sairia 1,0.
    Um modelo que predissesse sempre o mesmo numero seria reportado como
    ordenacao perfeita.
    """
    if len(a) < 3:
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def avaliar_por_grupo(dados: dict, alpha: float = 1.0) -> dict:
    """Leave-one-country-out. Um pais inteiro fora por vez.

    A regressao e ridge fechada em numpy: o baseline nao pode depender de
    escolha de otimizador, ou a comparacao com o encoder passa a medir duas
    coisas ao mesmo tempo.
    """
    X, y, grupos = dados["X"], dados["y"], dados["grupos"]
    unicos = sorted(set(grupos.tolist()))
    preditos = np.full(len(y), np.nan)

    for grupo in unicos:
        treino = grupos != grupo
        teste = ~treino
        if treino.sum() < 5 or teste.sum() == 0:
            continue
        Xt, yt = X[treino], y[treino]
        media, desvio = Xt.mean(axis=0), Xt.std(axis=0)
        desvio[desvio == 0] = 1.0
        Zt = (Xt - media) / desvio
        Zv = (X[teste] - media) / desvio
        # ridge fechado: (Z'Z + aI)^-1 Z'y, com intercepto pela media de y
        A = Zt.T @ Zt + alpha * np.eye(Zt.shape[1])
        w = np.linalg.solve(A, Zt.T @ (yt - yt.mean()))
        preditos[teste] = Zv @ w + yt.mean()

    ok = ~np.isnan(preditos)
    # `preditos` sai junto porque o erro POR CHIP e o que a predicao seletiva
    # precisa, e recalcula-lo fora daqui criaria uma segunda regressao que um
    # dia divergiria desta. Campo aditivo: nenhum numero ja publicado muda.
    por_grupo = {}
    for grupo in unicos:
        m = ok & (grupos == grupo)
        por_grupo[grupo] = {"n": int(m.sum()),
                            "spearman": _spearman(preditos[m], y[m])}
    return {
        "n_avaliados": int(ok.sum()),
        "grupos": len(unicos),
        "preditos_fora_da_dobra": preditos.tolist(),
        "spearman_global": _spearman(preditos[ok], y[ok]),
        "spearman_por_grupo": por_grupo,
        "spearman_mediano_entre_grupos": (
            round(float(np.median([v["spearman"] for v in por_grupo.values()
                                   if v["spearman"] is not None])), 6)
            if any(v["spearman"] is not None for v in por_grupo.values())
            else None),
        "alpha": alpha,
    }


def executar(alpha: float = 1.0) -> dict:
    dados = montar()
    resultado = avaliar_por_grupo(dados, alpha)
    resumo = {
        "tipo": "baseline_descritivo_terreno",
        "etapa": "B2-baseline",
        "quando_utc": agora_utc(),
        "dimensoes": len(nomes_das_dimensoes()),
        "nomes_das_dimensoes": nomes_das_dimensoes(),
        "chips": len(dados["nomes"]),
        "splits_usados": ["treino", "validacao"],
        "alvo": "fracao de agua do LabelHand, so em pixel valido",
        "unidade_de_validacao": "pais (leave-one-country-out)",
        "por_que_pais": ("chips do mesmo evento compartilham bacia, data e "
                         "regime; separa-los ao acaso deixaria o vizinho do "
                         "treino responder pelo do teste"),
        **resultado,
        "nao_e": ("previsao de evento: o terreno nao sabe se choveu. O "
                  "Sen1Floods11 so tem cenas de eventos, entao a pergunta ja "
                  "vem condicionada a 'dado que choveu forte, a forma do "
                  "terreno ordena onde a agua ficou?'"),
    }
    resumo["hash_resumo"] = sha256_json(
        {k: v for k, v in resumo.items() if k != "quando_utc"})
    return resumo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
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

    resumo = executar(args.alpha)
    if args.gravar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        destino = SAIDA / "baseline_descritivo.json"
        destino.write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        resumo["gravado_em"] = str(destino)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
