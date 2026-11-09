"""A5 -- a rede medida contra o baseline, na mesma regua.

-------------------------------------------------------------------------
A MESMA REGUA, LITERALMENTE
-------------------------------------------------------------------------

`contagens` e `metricas` sao importadas de `baseline_otsu`, e nao
reimplementadas. Duas copias de uma regra de metrica divergem -- e a
divergencia apareceria como "a rede ganhou", quando o que mudou foi quem
contava. Vale aqui a mesma decisao que `rag_verificador_citacao` tomou ao
importar `sustenta` de `claims`.

Tres coisas ficam identicas ao A2 por construcao:

  a MASCARA      pixel com rotulo `-1` fora; e aqui tambem sai o pixel sem
                 leitura de S1, porque a rede nao teve entrada para ele.
  a UNIDADE      chip inteiro de 512x512, nunca recorte. O treino usa recorte;
                 a avaliacao nao pode, ou os numeros deixam de ser comparaveis.
  os DOIS CORTES com e sem agua permanente do JRC. O segundo e o que interessa:
                 no baseline, 47% dos verdadeiros-positivos eram rio e lago
                 anteriores a enchente.

-------------------------------------------------------------------------
O QUE ESTE SCRIPT PODE CONCLUIR
-------------------------------------------------------------------------

Que a rede ganhou ou perdeu do limiar NESTE conjunto, com este split. Nao que
ela funciona em Recife, Curitiba ou Petropolis -- o Sen1Floods11 tem sua
propria distribuicao geografica, e a queda fora de regiao e a etapa A6, com a
Bolivia que continua trancada ate la.

Perder para o Otsu e resultado publicavel. A frente A entrega um numero contra
um baseline declarado, incluindo o caso em que ele e desfavoravel.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))

from baseline_otsu import SAIDA, contagens, metricas  # noqa: E402
from dados_sen1floods11 import ler_chip, ler_jrc, nomes_do_split  # noqa: E402
from modelo_unet import UNetPequena  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402
from treino import preparar  # noqa: E402

CLASSE_AGUA = 1


def carregar(rotulo: str = "v1", saida: Path = SAIDA) -> tuple:
    """Devolve `(modelo, manifesto)` -- o peso e o que se sabe sobre ele.

    O manifesto vem junto de proposito: avaliar um checkpoint sem saber com que
    seed, split e normalizacao ele foi feito produz um numero orfao.
    """
    manifesto_path = saida / f"unet_{rotulo}_manifesto.json"
    peso_path = saida / f"unet_{rotulo}.pt"
    if not manifesto_path.exists() or not peso_path.exists():
        raise FileNotFoundError(
            f"checkpoint '{rotulo}' ausente em {saida}; rode o treino primeiro")
    manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))
    modelo = UNetPequena(canais_entrada=2, classes=2,
                         base=manifesto["hiperparametros"]["base"])
    modelo.load_state_dict(torch.load(peso_path, map_location="cpu"))
    modelo.eval()
    return modelo, manifesto


@torch.no_grad()
def prever(modelo, nome: str, estat: dict) -> tuple[np.ndarray, np.ndarray]:
    """Devolve `(agua_predita, confianca)` para o chip INTEIRO."""
    x, _ = preparar(nome, estat)
    logits = modelo(x.unsqueeze(0))[0]
    prob = torch.softmax(logits, dim=0)[CLASSE_AGUA].numpy()
    return prob >= 0.5, prob


MEDIA_DO_CHIP = "media_do_chip"
MARGEM_NA_PREDICAO = "margem_na_predicao"
MODOS_CONFIANCA = (MEDIA_DO_CHIP, MARGEM_NA_PREDICAO)


def confianca(prob: np.ndarray, valido: np.ndarray, pred: np.ndarray,
              modo: str = MARGEM_NA_PREDICAO) -> float | None:
    """Quao decidida esta a rede neste chip.

    `media_do_chip` foi a primeira versao, e ELA NAO DISCRIMINA -- fica aqui
    porque o resultado negativo e informacao. A medida e a margem media
    `|p-0,5|*2` sobre TODOS os pixels validos, e ~87% deles sao fundo obvio
    que a rede acerta com probabilidade quase zero. O fundo domina a media, e
    um chip dificil sai com quase a mesma certeza de um facil: na calibracao
    real, todos os limiares de 0,05 a 0,50 recusaram os MESMOS tres chips.

    `margem_na_predicao` mede a margem so onde a rede SE COMPROMETE -- os
    pixels em que ela prediz agua. E ali que a duvida importa: predizer agua
    com p=0,52 e diferente de predizer com p=0,99, e essa diferenca some
    quando o fundo entra na conta.

    Chip sem nenhum pixel predito como agua devolve `None`, e `None` NAO e
    recusa. A rede tem posicao ali -- ela diz que nao ha agua --, e transformar
    isso em `nao_identificavel` confundiria "nao sei" com "nao ha", que e
    justamente a distincao que a frente inteira existe para manter.
    """
    if not valido.any():
        return None
    if modo == MEDIA_DO_CHIP:
        return float(np.abs(prob[valido] - 0.5).mean() * 2)
    if modo == MARGEM_NA_PREDICAO:
        alvo = pred & valido
        if not alvo.any():
            return None
        return float(np.abs(prob[alvo] - 0.5).mean() * 2)
    raise ValueError(f"modo de confianca desconhecido: {modo!r}; "
                     f"conhecidos: {list(MODOS_CONFIANCA)}")


def avaliar(rotulo: str = "v1", split: str = "teste",
            limiar_confianca: float | None = None,
            modo_confianca: str = MARGEM_NA_PREDICAO) -> dict:
    modelo, manifesto = carregar(rotulo)
    estat = manifesto["normalizacao"]
    nomes = nomes_do_split(split)

    por_chip = []
    soma = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    soma_sem_perm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    recusados = []

    for nome in nomes:
        chip = ler_chip(nome)
        jrc = ler_jrc(nome)
        pred, prob = prever(modelo, nome, estat)

        # a mesma mascara do A2, mais o pixel sem leitura de S1: a rede nao
        # teve entrada para ele, entao nao se credita nem se pune.
        com_leitura = np.isfinite(chip.s1).all(axis=0)
        valido = chip.valido & com_leitura
        verdade = chip.agua

        # A CERTEZA E CALCULADA SEMPRE, e nao so quando ha limiar.
        #
        # Enquanto ela so existia dentro do `if`, a unica coisa que se podia
        # perguntar sobre a confianca era "quantos chips ela recusa neste
        # corte?". A pergunta que decide se a recusa vale alguma coisa e outra:
        # a confianca ORDENA os chips melhor que o acaso? Essa exige o par
        # (certeza, erro) de TODO chip avaliado, e nao so dos recusados.
        #
        # Sem isto, a A7 escolhia limiar por orcamento de cobertura e relatava
        # o ganho de IoU como consequencia -- e com o grid grosso o ponto
        # escolhido recusava 1 de 88 chips, removendo praticamente nada. O
        # numero parecia dizer "a confianca nao separa", quando na verdade
        # ninguem tinha medido a separacao.
        certeza = confianca(prob, valido, pred, modo_confianca)

        # A7 em forma de opcao: um chip cuja predicao e toda morna nao vira
        # "sem agua" -- vira `nao_identificavel`, e sai da metrica declarado.
        if limiar_confianca is not None:
            if certeza is not None and certeza < limiar_confianca:
                recusados.append({"nome": nome,
                                  "certeza": round(certeza, 6),
                                  "modo": modo_confianca})
                continue

        c = contagens(pred, verdade, valido)
        c_sem_perm = contagens(pred, verdade, valido & (jrc == 0))
        for k in soma:
            soma[k] += c[k]
            soma_sem_perm[k] += c_sem_perm[k]

        por_chip.append({
            "nome": nome,
            "certeza": None if certeza is None else round(certeza, 6),
            "modo_confianca": modo_confianca,
            "pixels_validos": int(valido.sum()),
            "agua_verdadeira": int((verdade & valido).sum()),
            "agua_predita": int((pred & valido).sum()),
            **c,
            **metricas(c),
            **{f"{k}_sem_permanente": v for k, v in metricas(c_sem_perm).items()},
        })

    def _media(chave):
        vals = [c[chave] for c in por_chip if c[chave] is not None]
        return round(float(np.mean(vals)), 6) if vals else None

    resumo = {
        "tipo": "avaliacao_unet_sen1floods11",
        "etapa": "A5",
        "quando_utc": agora_utc(),
        "checkpoint": rotulo,
        "hash_estado": manifesto["hash_estado"],
        "seed": manifesto["seed"],
        "epocas_treinadas": manifesto["hiperparametros"]["epocas"],
        "parametros": manifesto["parametros"],
        "split": split,
        "chips_avaliados": len(por_chip),
        "chips_recusados": len(recusados),
        "limiar_confianca": limiar_confianca,
        "modo_confianca": modo_confianca,
        "recusados": recusados,
        "micro": {**soma, **metricas(soma)},
        "micro_sem_agua_permanente": {**soma_sem_perm, **metricas(soma_sem_perm)},
        "macro": {k: _media(k) for k in ("iou", "f1", "precisao", "recall")},
        "macro_sem_agua_permanente": {
            k: _media(f"{k}_sem_permanente")
            for k in ("iou", "f1", "precisao", "recall")},
        "nao_e": ("resultado neste conjunto, com este split; nao e afirmacao "
                  "sobre Recife, Curitiba ou Petropolis, e nao desloca escore "
                  "de suscetibilidade"),
    }
    return {"resumo": resumo, "por_chip": por_chip}


def comparar_com_baseline(resumo: dict, split: str = "teste",
                          saida: Path = SAIDA) -> dict:
    """A diferenca publicada com o sinal que der."""
    arquivo = saida / f"baseline_otsu_{split}.json"
    if not arquivo.exists():
        return {"disponivel": False,
                "por_que": f"baseline ausente: {arquivo}; rode a etapa A2"}
    base = json.loads(arquivo.read_text(encoding="utf-8"))
    comparacao = {"disponivel": True, "baseline": "otsu_do_dataset",
                  "hash_resumo_baseline": base.get("hash_resumo")}
    for bloco in ("micro", "micro_sem_agua_permanente",
                  "macro", "macro_sem_agua_permanente"):
        linha = {}
        for m in ("iou", "f1", "precisao", "recall"):
            r, b = resumo[bloco].get(m), base[bloco].get(m)
            linha[m] = {
                "rede": r, "otsu": b,
                "diferenca": round(r - b, 6) if (r is not None and b is not None)
                else None,
            }
        comparacao[bloco] = linha
    return comparacao


def gravar(resultado: dict, comparacao: dict, saida: Path = SAIDA) -> Path:
    saida.mkdir(parents=True, exist_ok=True)
    rotulo = resultado["resumo"]["checkpoint"]
    split = resultado["resumo"]["split"]
    payload = {**resultado["resumo"], "comparacao_com_baseline": comparacao}
    payload["hash_resumo"] = sha256_json(
        {k: v for k, v in payload.items() if k != "quando_utc"})
    destino = saida / f"avaliacao_unet_{rotulo}_{split}.json"
    destino.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    if resultado["por_chip"]:
        csv_path = saida / f"avaliacao_unet_{rotulo}_{split}_por_chip.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=list(resultado["por_chip"][0]))
            escritor.writeheader()
            escritor.writerows(resultado["por_chip"])
    return destino


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotulo", default="v1")
    parser.add_argument("--split", default="teste")
    parser.add_argument("--limiar-confianca", type=float, default=None)
    parser.add_argument("--modo-confianca", default=MARGEM_NA_PREDICAO,
                        choices=list(MODOS_CONFIANCA))
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

    resultado = avaliar(args.rotulo, args.split, args.limiar_confianca,
                        args.modo_confianca)
    comparacao = comparar_com_baseline(resultado["resumo"], args.split)
    if args.gravar:
        destino = gravar(resultado, comparacao)
        resultado["resumo"]["gravado_em"] = str(destino)
    print(json.dumps({**resultado["resumo"],
                      "comparacao_com_baseline": comparacao},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
