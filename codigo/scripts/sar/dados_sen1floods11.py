"""Leitor e contrato de dado do Sen1Floods11 para a frente A do REV-P.

-------------------------------------------------------------------------
O TRES ESTADOS, e por que o terceiro nao pode virar o segundo
-------------------------------------------------------------------------

`LabelHand` tem tres valores, e nao dois:

    -1  sem dado      o rotulador humano NAO conseguiu decidir aquele pixel
     0  nao-agua      afirmacao de observacao: olhou e nao havia agua
     1  agua

Tratar `-1` como `0` seria repetir, dentro da rede, exatamente o erro que o
REV-P inteiro combate: ausencia de registro virando registro de ausencia. E
aqui isso nao e um detalhe de borda -- no corpus real, `-1` e 13,63% dos
pixels. Um sexto da classe negativa seria inventado.

Por isso NENHUMA funcao deste modulo devolve rotulo binarizado sozinho. Toda
leitura devolve o rotulo E a mascara de validade, e quem for treinar ou medir
tem de dizer o que faz com ela. A `ignore_index` do treino e essa mascara; a
metrica so olha pixel valido.

-------------------------------------------------------------------------
OS CINCO PRODUTOS
-------------------------------------------------------------------------

    S1Hand            2 bandas VV e VH, float32, em dB
    LabelHand         rotulo manual por pixel, os tres estados acima
    S1OtsuLabelHand   o BASELINE: limiar de Otsu sobre o proprio S1
    JRCWaterHand      agua permanente (JRC), separa inundacao de rio e lago
    S2Hand            optico, nao usado nesta frente

`S1OtsuLabelHand` merece nota: ele e o mesmo tipo de metodo que produz o
footprint SAR do REV-P hoje, e veio junto com o dataset. O comparador da
frente A ja existia -- nao foi escolhido por mim, o que e exatamente o que o
torna defensavel.

`JRCWaterHand` resolve a confusao que mais importa: o `ds03` ja distingue
classe 9 (agua permanente) de classe 1 (inundacao), e sem essa mascara uma
rede aprenderia a achar rio e chamaria isso de acerto.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao treina, nao promove negativo para o dataset SUSC e nao extrai feature. A
proveniencia do dado (`local_runs/n1a-sen1floods11/PROVENIENCIA.md`) proibe as
duas ultimas coisas explicitamente, e a proibicao continua valendo depois de
existir modelo.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from proveniencia import sha256_arquivo  # noqa: E402

RAIZ = REPO / "local_runs" / "n1a-sen1floods11" / "v1.1"
DADOS = RAIZ / "data" / "flood_events" / "HandLabeled"
SPLITS = RAIZ / "splits" / "flood_handlabeled"

SEM_DADO = -1
NAO_AGUA = 0
AGUA = 1
ESTADOS = (SEM_DADO, NAO_AGUA, AGUA)

# produto -> (subdiretorio, sufixo do arquivo)
PRODUTOS = {
    "s1": ("S1Hand", "S1Hand"),
    "rotulo": ("LabelHand", "LabelHand"),
    "otsu": ("S1OtsuLabelHand", "S1OtsuLabelHand"),
    "jrc": ("JRCWaterHand", "JRCWaterHand"),
    "s2": ("S2Hand", "S2Hand"),
}

# Os splits sao os OFICIAIS do Sen1Floods11. Nao sao reamostrados: um split
# proprio sobre 446 chips daria para escolher, sem querer, o conjunto em que o
# modelo vai bem. `bolivia` e o holdout fora de regiao e fica trancado ate a
# etapa A6.
SPLITS_OFICIAIS = {
    "treino": "flood_train_data.csv",
    "validacao": "flood_valid_data.csv",
    "teste": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}

LADO = 512


class ContratoDeDadoViolado(Exception):
    """Falha dura. Um chip fora do contrato nao entra silenciosamente."""


@dataclass(frozen=True)
class Chip:
    """Um chip lido, com a mascara de validade sempre junto do rotulo."""

    nome: str
    s1: np.ndarray               # (2, 512, 512) float32, dB
    rotulo: np.ndarray           # (512, 512) int16, com -1
    valido: np.ndarray           # (512, 512) bool -- rotulo != SEM_DADO

    @property
    def agua(self) -> np.ndarray:
        """Binario de agua. So faz sentido lido junto de `valido`."""
        return self.rotulo == AGUA

    @property
    def fracao_valida(self) -> float:
        return float(self.valido.mean())

    @property
    def fracao_agua_no_valido(self) -> float:
        n = int(self.valido.sum())
        return float((self.agua & self.valido).sum() / n) if n else 0.0


def caminho(nome: str, produto: str) -> Path:
    if produto not in PRODUTOS:
        raise ValueError(f"produto desconhecido: {produto!r}; "
                         f"conhecidos: {sorted(PRODUTOS)}")
    sub, sufixo = PRODUTOS[produto]
    return DADOS / sub / f"{nome}_{sufixo}.tif"


def _ler_raster(caminho_arquivo: Path, bandas_esperadas: int) -> np.ndarray:
    if not caminho_arquivo.exists():
        raise ContratoDeDadoViolado(f"ausente: {caminho_arquivo}")
    with rasterio.open(caminho_arquivo) as fonte:
        if fonte.count != bandas_esperadas:
            raise ContratoDeDadoViolado(
                f"{caminho_arquivo.name}: {fonte.count} bandas, "
                f"esperado {bandas_esperadas}")
        if (fonte.height, fonte.width) != (LADO, LADO):
            raise ContratoDeDadoViolado(
                f"{caminho_arquivo.name}: {fonte.height}x{fonte.width}, "
                f"esperado {LADO}x{LADO}")
        if str(fonte.crs) != "EPSG:4326":
            raise ContratoDeDadoViolado(
                f"{caminho_arquivo.name}: CRS {fonte.crs}, esperado EPSG:4326")
        return fonte.read()


def ler_rotulo(nome: str) -> tuple[np.ndarray, np.ndarray]:
    """Devolve `(rotulo, valido)`. Nunca so o rotulo.

    A assinatura e de proposito: quem chamar tem de receber a mascara na mao e
    decidir o que faz com ela. Uma funcao que devolvesse so o rotulo tornaria
    `rotulo == 1` a leitura obvia -- e essa leitura joga `-1` e `0` no mesmo
    balde sem ninguem notar.
    """
    bruto = _ler_raster(caminho(nome, "rotulo"), 1)[0].astype(np.int16)
    fora = set(np.unique(bruto).tolist()) - set(ESTADOS)
    if fora:
        raise ContratoDeDadoViolado(
            f"{nome}: rotulo com valores fora do contrato: {sorted(fora)}")
    return bruto, bruto != SEM_DADO


def ler_s1(nome: str) -> np.ndarray:
    """VV e VH em dB. NaN e perda de sinal e nao pode virar zero.

    Zero dB nao e "sem medida": e uma retrodifusao alta e plausivel. Preencher
    NaN com zero criaria superficie brilhante onde nao houve leitura, que e o
    inverso do que a rede precisa aprender sobre agua (que e escura no SAR).
    """
    return _ler_raster(caminho(nome, "s1"), 2).astype(np.float32)


def ler_otsu(nome: str) -> np.ndarray:
    """Baseline. Vem do dataset, nao foi ajustado por nos."""
    return _ler_raster(caminho(nome, "otsu"), 1)[0].astype(np.int16)


def ler_jrc(nome: str) -> np.ndarray:
    """Agua permanente. Separa inundacao de corpo d'agua que sempre esteve la."""
    return _ler_raster(caminho(nome, "jrc"), 1)[0].astype(np.uint8)


def ler_chip(nome: str) -> Chip:
    rotulo, valido = ler_rotulo(nome)
    return Chip(nome=nome, s1=ler_s1(nome), rotulo=rotulo, valido=valido)


def nomes_do_split(split: str) -> list[str]:
    """Nomes de chip do split OFICIAL, na ordem do arquivo."""
    if split not in SPLITS_OFICIAIS:
        raise ValueError(f"split desconhecido: {split!r}; "
                         f"conhecidos: {sorted(SPLITS_OFICIAIS)}")
    arquivo = SPLITS / SPLITS_OFICIAIS[split]
    if not arquivo.exists():
        raise ContratoDeDadoViolado(f"split ausente: {arquivo}")
    nomes = []
    with arquivo.open(encoding="utf-8", newline="") as f:
        for linha in csv.reader(f):
            if not linha or not linha[0].strip():
                continue
            nomes.append(linha[0].strip().replace("_S1Hand.tif", ""))
    return nomes


def splits_sao_disjuntos() -> dict:
    """Nenhum chip pode aparecer em dois splits.

    Verificado em vez de assumido: o vazamento entre treino e teste e o erro
    que mais facilmente produz um numero bonito e falso, e ele nao aparece em
    nenhuma metrica -- so numa checagem como esta.
    """
    conjuntos = {s: set(nomes_do_split(s)) for s in SPLITS_OFICIAIS}
    colisoes = {}
    nomes_ordenados = sorted(conjuntos)
    for i, a in enumerate(nomes_ordenados):
        for b in nomes_ordenados[i + 1:]:
            comum = conjuntos[a] & conjuntos[b]
            if comum:
                colisoes[f"{a}|{b}"] = sorted(comum)
    return {
        "tamanhos": {s: len(c) for s, c in conjuntos.items()},
        "colisoes": colisoes,
        "disjuntos": not colisoes,
    }


def auditar_chip(nome: str) -> dict:
    """Retrato verificavel de um chip, com hash de cada produto usado."""
    chip = ler_chip(nome)
    otsu = ler_otsu(nome)
    jrc = ler_jrc(nome)
    return {
        "nome": nome,
        "fracao_valida": round(chip.fracao_valida, 6),
        "fracao_agua_no_valido": round(chip.fracao_agua_no_valido, 6),
        "pixels_sem_dado": int((~chip.valido).sum()),
        "s1_nan": int(np.isnan(chip.s1).sum()),
        "s1_min_db": round(float(np.nanmin(chip.s1)), 4),
        "s1_max_db": round(float(np.nanmax(chip.s1)), 4),
        "otsu_agua": int((otsu == AGUA).sum()),
        "jrc_agua_permanente": int((jrc == 1).sum()),
        "hash_s1": sha256_arquivo(caminho(nome, "s1")),
        "hash_rotulo": sha256_arquivo(caminho(nome, "rotulo")),
        "hash_otsu": sha256_arquivo(caminho(nome, "otsu")),
    }


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="teste",
                        choices=sorted(SPLITS_OFICIAIS))
    parser.add_argument("--chip", default=None)
    args = parser.parse_args()

    if args.chip:
        print(json.dumps(auditar_chip(args.chip), indent=2, ensure_ascii=False))
        return 0

    disjunto = splits_sao_disjuntos()
    nomes = nomes_do_split(args.split)
    validas, aguas, sem_agua = [], [], 0
    for nome in nomes:
        chip = ler_chip(nome)
        validas.append(chip.fracao_valida)
        aguas.append(chip.fracao_agua_no_valido)
        if not (chip.agua & chip.valido).any():
            sem_agua += 1
    print(json.dumps({
        "split": args.split,
        "chips": len(nomes),
        "splits": disjunto,
        "fracao_valida_media": round(float(np.mean(validas)), 6),
        "fracao_agua_no_valido_media": round(float(np.mean(aguas)), 6),
        "chips_sem_nenhum_pixel_de_agua": sem_agua,
        "nao_e": ("chip sem pixel de agua e negativo POR OBSERVACAO, e nao "
                  "chip vazio"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
