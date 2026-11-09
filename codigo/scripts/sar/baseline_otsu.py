"""A2 -- o baseline Otsu, medido antes de existir rede.

-------------------------------------------------------------------------
POR QUE ISTO VEM PRIMEIRO
-------------------------------------------------------------------------

Uma rede que publica IoU 0,72 nao diz nada sozinha. Ela so significa alguma
coisa contra um numero anterior, obtido no MESMO conjunto, com a MESMA regra de
mascara. Medir o baseline depois de ver o resultado da rede e como escolher o
comparador sabendo o placar.

E o comparador aqui nao foi escolhido por nos, o que e o que o torna
defensavel: `S1OtsuLabelHand` veio dentro do proprio Sen1Floods11, e e o mesmo
tipo de metodo -- limiar sobre retrodifusao -- que produz o footprint SAR do
REV-P hoje.

-------------------------------------------------------------------------
TRES REGRAS DE MASCARA, e nenhuma delas e detalhe
-------------------------------------------------------------------------

1. Pixel com rotulo `-1` NAO entra na metrica. Ele nao e nao-agua: e pixel que
   o rotulador humano nao conseguiu decidir. No corpus real sao 13,63% dos
   pixels, e conta-los como negativo acertado inflaria toda metrica que tem
   verdadeiro-negativo no denominador.

2. Pixel onde o Otsu nao prediz (`-1`) tambem sai. Nao se credita nem se pune
   um metodo por onde ele se absteve -- mas o quanto ele se absteve e
   reportado, porque abster-se muito e uma forma de nao errar.

3. A metrica e calculada DUAS vezes: no valido inteiro, e so onde a agua NAO e
   permanente (`JRCWaterHand == 0`). A segunda e a que interessa. Um metodo de
   limiar acha rio com facilidade, e rio estava la antes da enchente: acertar
   agua permanente e acertar a pergunta errada.

-------------------------------------------------------------------------
MICRO E MACRO
-------------------------------------------------------------------------

`micro` agrega todos os pixels de todos os chips e depois divide -- e por isso
e dominada pelos chips com muita agua. `macro` calcula por chip e tira a media
-- e por isso da o mesmo peso a um chip com 40% de agua e a um com 0,1%.

As duas sao publicadas. Reportar so a que ficou melhor e a forma mais comum de
mentir com metrica de segmentacao.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))

from dados_sen1floods11 import (  # noqa: E402
    AGUA, SEM_DADO, ler_chip, ler_jrc, ler_otsu, nomes_do_split,
    splits_sao_disjuntos,
)
from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "sar-inundacao"


def contagens(pred: np.ndarray, verdade: np.ndarray,
              valido: np.ndarray) -> dict:
    """TP/FP/FN/TN restritos a `valido`. Fora dele, nada e contado."""
    p = pred & valido
    v = verdade & valido
    return {
        "tp": int((p & v).sum()),
        "fp": int((p & ~v & valido).sum()),
        "fn": int((~p & v & valido).sum()),
        "tn": int((~p & ~v & valido).sum()),
    }


def metricas(c: dict) -> dict:
    """IoU, F1, precisao e recall a partir das contagens.

    Denominador zero devolve `None`, e nao 0.0. Um chip sem nenhum pixel de
    agua verdadeira nem predita tem IoU indefinido -- chamar isso de zero
    puniria o metodo por acertar, e chamar de um o premiaria por nada.
    """
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    uniao = tp + fp + fn
    return {
        "iou": round(tp / uniao, 6) if uniao else None,
        "f1": round(2 * tp / (2 * tp + fp + fn), 6) if (2 * tp + fp + fn) else None,
        "precisao": round(tp / (tp + fp), 6) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 6) if (tp + fn) else None,
    }


def _media(valores: list) -> float | None:
    limpos = [v for v in valores if v is not None]
    return round(float(np.mean(limpos)), 6) if limpos else None


def avaliar_split(split: str = "teste") -> dict:
    nomes = nomes_do_split(split)
    por_chip = []
    soma = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    soma_sem_perm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    pixels_totais = 0
    fora_por_rotulo = 0
    fora_por_abstencao = 0
    pixels_agua_permanente = 0

    for nome in nomes:
        chip = ler_chip(nome)
        otsu = ler_otsu(nome)
        jrc = ler_jrc(nome)

        prediz = otsu != SEM_DADO
        valido = chip.valido & prediz
        pred = otsu == AGUA
        verdade = chip.agua

        pixels_totais += chip.rotulo.size
        fora_por_rotulo += int((~chip.valido).sum())
        fora_por_abstencao += int((chip.valido & ~prediz).sum())
        pixels_agua_permanente += int((jrc == 1).sum())

        c = contagens(pred, verdade, valido)
        # A pergunta que importa: e inundacao, ou e o rio que sempre esteve la?
        c_sem_perm = contagens(pred, verdade, valido & (jrc == 0))
        for k in soma:
            soma[k] += c[k]
            soma_sem_perm[k] += c_sem_perm[k]

        por_chip.append({
            "nome": nome,
            "pixels_validos": int(valido.sum()),
            "agua_verdadeira": int((verdade & valido).sum()),
            "agua_predita": int((pred & valido).sum()),
            "agua_permanente_jrc": int((jrc == 1).sum()),
            **c,
            **metricas(c),
            **{f"{k}_sem_permanente": v
               for k, v in metricas(c_sem_perm).items()},
        })

    resumo = {
        "tipo": "baseline_otsu_sen1floods11",
        "etapa": "A2",
        "quando_utc": agora_utc(),
        "split": split,
        "chips": len(nomes),
        "splits_disjuntos": splits_sao_disjuntos()["disjuntos"],
        "mascara": {
            "pixels_totais": pixels_totais,
            "fora_por_rotulo_sem_dado": fora_por_rotulo,
            "fora_por_abstencao_do_otsu": fora_por_abstencao,
            "fracao_fora_por_rotulo": round(fora_por_rotulo / pixels_totais, 6),
            "fracao_fora_por_abstencao": round(
                fora_por_abstencao / pixels_totais, 6),
            "pixels_agua_permanente_jrc": pixels_agua_permanente,
            "por_que": ("rotulo -1 nao e nao-agua; abstencao do baseline nao "
                        "credita nem pune"),
        },
        "micro": {**soma, **metricas(soma)},
        "micro_sem_agua_permanente": {**soma_sem_perm,
                                      **metricas(soma_sem_perm)},
        "macro": {
            "iou": _media([c["iou"] for c in por_chip]),
            "f1": _media([c["f1"] for c in por_chip]),
            "precisao": _media([c["precisao"] for c in por_chip]),
            "recall": _media([c["recall"] for c in por_chip]),
        },
        "macro_sem_agua_permanente": {
            "iou": _media([c["iou_sem_permanente"] for c in por_chip]),
            "f1": _media([c["f1_sem_permanente"] for c in por_chip]),
            "precisao": _media([c["precisao_sem_permanente"] for c in por_chip]),
            "recall": _media([c["recall_sem_permanente"] for c in por_chip]),
        },
        "chips_com_iou_indefinido": sum(1 for c in por_chip
                                        if c["iou"] is None),
        "nao_e": ("este numero e o piso do metodo de limiar neste conjunto; "
                  "ele nao autoriza nenhuma afirmacao sobre suscetibilidade"),
    }
    resumo["hash_resumo"] = sha256_json(
        {k: v for k, v in resumo.items() if k != "quando_utc"})
    return {"resumo": resumo, "por_chip": por_chip}


def gravar(resultado: dict, saida: Path = SAIDA) -> Path:
    saida.mkdir(parents=True, exist_ok=True)
    split = resultado["resumo"]["split"]
    destino = saida / f"baseline_otsu_{split}.json"
    destino.write_text(
        json.dumps(resultado["resumo"], indent=2, ensure_ascii=False),
        encoding="utf-8")
    csv_path = saida / f"baseline_otsu_{split}_por_chip.csv"
    campos = list(resultado["por_chip"][0])
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultado["por_chip"])
    return destino


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="teste")
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

    resultado = avaliar_split(args.split)
    if args.gravar:
        destino = gravar(resultado)
        resultado["resumo"]["gravado_em"] = str(destino)
    print(json.dumps(resultado["resumo"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
