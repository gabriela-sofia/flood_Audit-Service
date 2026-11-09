"""A4 -- treino da U-Net da frente A, com proveniencia de peso.

-------------------------------------------------------------------------
AS TRES DECISOES QUE MUDAM O RESULTADO, e por que estao aqui e nao no laco
-------------------------------------------------------------------------

1. NORMALIZACAO SO COM O TREINO. Media e desvio de VV/VH sao calculados
   exclusivamente no split de treino e gravados no checkpoint. Calcular sobre
   o conjunto todo seria vazamento -- o modelo saberia algo sobre a
   distribuicao do teste antes de ver o teste, e o vazamento nao aparece em
   metrica nenhuma.

2. NaN NAO E ZERO, E NAO E PREDIZIVEL. `S1Hand` traz NaN onde nao houve
   leitura. NaN propaga pela convolucao e destroi o gradiente, entao ele
   precisa virar numero -- mas o pixel tambem nao pode ser aprendido nem
   cobrado. A regra e dupla: o valor vira 0 DEPOIS da normalizacao (a media do
   treino, o ponto mais neutro que existe) e o pixel entra na mascara de
   ignore, junto com o `-1` do rotulo. Preencher sem ignorar ensinaria a rede
   que ausencia de leitura parece agua media.

3. RECORTE ALEATORIO NO TREINO, CHIP INTEIRO NA AVALIACAO. 512x512 em CPU e
   caro; recortes de 256 no treino cabem no orcamento. Mas avaliar em recorte
   mudaria a pergunta -- a metrica do baseline foi medida no chip inteiro, e
   comparar numeros medidos em unidades diferentes e a forma mais silenciosa
   de vencer uma comparacao.

-------------------------------------------------------------------------
PROVENIENCIA DE PESO
-------------------------------------------------------------------------

Peso que muda exige responder "mudou?" sem reexecutar. O checkpoint carrega
hash do estado, seed, split, normalizacao, contagem de parametros, versao do
torch e o commit do repositorio. Mesma regra do resto do projeto: o hash nao
cobre o instante do registro.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))

from dados_sen1floods11 import (  # noqa: E402
    SEM_DADO, ler_chip, nomes_do_split,
)
from modelo_unet import UNetPequena, contar_parametros  # noqa: E402
from proveniencia import agora_utc  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "neural"))
from proveniencia_de_peso import (  # noqa: E402
    commit_atual, hash_completo_do_estado, hash_do_estado, selar, semear,
)

SAIDA = REPO / "local_runs" / "sar-inundacao"
IGNORAR = -1          # indice de ignore da CrossEntropyLoss


def estatisticas_do_treino(nomes: list[str]) -> dict:
    """Media e desvio por banda, calculados SO no treino.

    NaN fica de fora da estatistica: incluir seria deixar a ausencia de leitura
    deslocar o centro da distribuicao.
    """
    somas = np.zeros(2, dtype=np.float64)
    somas_q = np.zeros(2, dtype=np.float64)
    contagem = np.zeros(2, dtype=np.int64)
    for nome in nomes:
        s1 = ler_chip(nome).s1
        for b in range(2):
            banda = s1[b]
            fin = np.isfinite(banda)
            somas[b] += float(banda[fin].sum())
            somas_q[b] += float((banda[fin].astype(np.float64) ** 2).sum())
            contagem[b] += int(fin.sum())
    media = somas / contagem
    var = somas_q / contagem - media ** 2
    return {"media": [round(float(v), 6) for v in media],
            "desvio": [round(float(v), 6) for v in np.sqrt(np.maximum(var, 1e-12))],
            "pixels_finitos": [int(v) for v in contagem]}


def preparar(nome: str, estat: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Devolve `(x, alvo)` com o alvo ja carregando o ignore.

    O alvo sai com `-1` em dois casos: rotulo sem dado, e pixel sem leitura de
    S1. Os dois sao `IGNORAR` para a perda, e por motivos diferentes que
    convergem na mesma decisao -- num nao se sabe a verdade, no outro nao se
    tem entrada.
    """
    chip = ler_chip(nome)
    s1 = chip.s1.astype(np.float32)
    sem_leitura = ~np.isfinite(s1).all(axis=0)

    media = np.array(estat["media"], dtype=np.float32).reshape(2, 1, 1)
    desvio = np.array(estat["desvio"], dtype=np.float32).reshape(2, 1, 1)
    x = (np.nan_to_num(s1, nan=0.0) - media) / desvio
    # o pixel sem leitura vira exatamente a media do treino, que apos a
    # normalizacao e zero -- o ponto mais neutro disponivel.
    x[:, sem_leitura] = 0.0

    alvo = chip.rotulo.astype(np.int64)
    alvo[chip.rotulo == SEM_DADO] = IGNORAR
    alvo[sem_leitura] = IGNORAR
    return torch.from_numpy(x), torch.from_numpy(alvo)


def _recorte(x: torch.Tensor, alvo: torch.Tensor, lado: int,
             gerador: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    _, h, w = x.shape
    if lado >= h:
        return x, alvo
    i = int(torch.randint(0, h - lado + 1, (1,), generator=gerador).item())
    j = int(torch.randint(0, w - lado + 1, (1,), generator=gerador).item())
    return x[:, i:i + lado, j:j + lado], alvo[i:i + lado, j:j + lado]


@torch.no_grad()
def perda_media(modelo: nn.Module, nomes: list[str], estat: dict,
                criterio: nn.Module) -> float:
    """Avaliacao em CHIP INTEIRO -- a mesma unidade em que o baseline foi medido."""
    modelo.eval()
    total, n = 0.0, 0
    for nome in nomes:
        x, alvo = preparar(nome, estat)
        saida = modelo(x.unsqueeze(0))
        perda = criterio(saida, alvo.unsqueeze(0))
        if torch.isfinite(perda):
            total += float(perda.detach())
            n += 1
    return round(total / n, 6) if n else float("nan")


SPLITS_QUE_O_PRETREINO_NAO_PODE_VER = frozenset({"validacao", "teste",
                                                 "bolivia"})


def verificar_contrato_do_encoder(peso: Path, estado: dict,
                                  estat_do_fino: dict) -> dict:
    """O que o encoder traz consigo, conferido contra o peso que foi carregado.

    Ate 03/10/2026 o manifesto do ajuste fino guardava so o CAMINHO do encoder.
    A restricao cientifica -- sem rotulo, sem cena de validacao ou teste, de
    qual linhagem saiu -- morava num arquivo ao lado que ninguem conferia, e um
    `.pt` trocado no disco viraria outra corrida com o mesmo nome.

    Falha dura so em CONTRADICAO positiva: hash que nao bate, rotulo usado,
    split proibido no pre-treino. Manifesto ausente fica declarado como nao
    verificado, porque encoder sem manifesto e o caso dos testes de forma.

    A deriva de normalizacao vai junto: o encoder aprendeu sobre entradas
    padronizadas pela estatistica do POTE dele, e o ajuste fino padroniza pela
    do split de treino inteiro. A diferenca nao e corrigida aqui -- ela e
    registrada, porque e uma das coisas que mudam entre pre-treino e uso.
    """
    ao_lado = peso.with_name(peso.stem + "_manifesto.json")
    if not ao_lado.exists():
        return {"manifesto": None, "verificado": False,
                "motivo": "encoder sem manifesto ao lado; origem nao auditavel"}
    meta = json.loads(ao_lado.read_text(encoding="utf-8"))
    hash_carregado = hash_do_estado(estado)
    hash_declarado = meta.get("hash_estado")
    if hash_declarado and hash_declarado != hash_carregado:
        raise ValueError(
            f"contrato do encoder quebrado: o manifesto declara "
            f"{hash_declarado} e o peso carregado e {hash_carregado}")
    completo = meta.get("hash_estado_completo")
    if completo and completo != hash_completo_do_estado(estado):
        raise ValueError("contrato do encoder quebrado: o hash completo do peso nao bate "
                         "com o manifesto, embora os 8 primeiros valores possam bater")
    if meta.get("usou_rotulo") is True:
        raise ValueError("contrato do encoder quebrado: o pre-treino declara "
                         "que usou rotulo, e entao nao e pre-treino sem rotulo")
    splits = set(meta.get("splits_do_pretreino") or [])
    proibidos = splits & SPLITS_QUE_O_PRETREINO_NAO_PODE_VER
    if proibidos:
        raise ValueError(
            f"contrato do encoder quebrado: o pre-treino viu {sorted(proibidos)}, "
            f"onde o ajuste fino seleciona epoca ou mede")
    norm = meta.get("normalizacao")
    deriva = None
    if norm and estat_do_fino:
        deriva = {
            "media": [round(f - p, 6) for f, p in
                      zip(estat_do_fino["media"], norm["media"])],
            "desvio": [round(f - p, 6) for f, p in
                       zip(estat_do_fino["desvio"], norm["desvio"])],
        }
    return {
        "manifesto": str(ao_lado),
        "verificado": True,
        "hash_estado": hash_carregado,
        "tipo_da_origem": meta.get("tipo_da_origem") or meta.get("tipo"),
        "usou_rotulo": meta.get("usou_rotulo"),
        "splits_do_pretreino": sorted(splits) or None,
        "passos_do_pretreino": (meta.get("passos_acumulados")
                                or meta.get("passos_totais")),
        "rodada": meta.get("rodada"),
        "linhagem": meta.get("linhagem"),
        "regime": ("indutivo: o pre-treino nao viu cena de validacao, teste "
                   "nem do holdout" if splits else
                   "nao declarado pelo manifesto do encoder"),
        "deriva_de_normalizacao_fino_menos_pretreino": deriva,
    }


def treinar(epocas: int = 2, base: int = 16, lado: int = 256,
            lote: int = 2, taxa: float = 1e-3, seed: int = 20260901,
            limite_treino: int | None = None,
            limite_validacao: int | None = None,
            threads: int | None = None,
            encoder_pretreinado: Path | None = None,
            selecionar_por_validacao: bool = True) -> dict:
    if threads:
        torch.set_num_threads(threads)
    semear(seed)

    nomes_treino = nomes_do_split("treino")[:limite_treino]
    nomes_valid = nomes_do_split("validacao")[:limite_validacao]

    inicio_estat = time.time()
    estat = estatisticas_do_treino(nomes_treino)
    segundos_estat = round(time.time() - inicio_estat, 2)

    modelo = UNetPequena(canais_entrada=2, classes=2, base=base)

    # AJUSTE FINO A PARTIR DO PRE-TREINO MASCARADO.
    #
    # `encoder_pretreinado` carrega os pesos de `enc1`, `enc2` e `enc3` que
    # sairam de `pretreino_mascarado`, que NAO leu rotulo nenhum. O resto da
    # rede continua com a mesma inicializacao aleatoria da semente -- so o
    # ponto de partida do encoder muda, e e essa a variavel do experimento.
    #
    # A carga e ESTRITA: uma chave que nao encaixa significa que o encoder
    # pre-treinado nao e o desta rede, e carregar o que der faria a corrida
    # medir uma arquitetura hibrida sem que nada denunciasse.
    pretreino = None
    contrato_do_encoder = None
    if encoder_pretreinado:
        estado = torch.load(encoder_pretreinado, map_location="cpu")
        alvo = {k: v for k, v in modelo.state_dict().items()
                if k.split(".")[0] in ("enc1", "enc2", "enc3")}
        faltando = set(alvo) - set(estado)
        sobrando = set(estado) - set(alvo) - {"pool"}
        if faltando or any(not s.startswith("pool") for s in sobrando):
            raise ValueError(
                f"encoder pre-treinado nao encaixa nesta rede; faltando="
                f"{sorted(faltando)[:3]} sobrando={sorted(sobrando)[:3]}")
        for chave, valor in estado.items():
            if chave in alvo and alvo[chave].shape != valor.shape:
                raise ValueError(f"forma incompativel em {chave}: "
                                 f"{alvo[chave].shape} vs {valor.shape}")
        modelo.load_state_dict(estado, strict=False)
        pretreino = str(encoder_pretreinado)
        contrato_do_encoder = verificar_contrato_do_encoder(
            Path(encoder_pretreinado), estado, estat)
    criterio = nn.CrossEntropyLoss(ignore_index=IGNORAR)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa)
    gerador = torch.Generator().manual_seed(seed)

    historico = []
    melhor = {"perda": None, "epoca": None, "estado": None}
    inicio = time.time()
    for epoca in range(1, epocas + 1):
        modelo.train()
        ordem = torch.randperm(len(nomes_treino), generator=gerador).tolist()
        acumulada, passos = 0.0, 0
        for k in range(0, len(ordem), lote):
            xs, alvos = [], []
            for idx in ordem[k:k + lote]:
                x, alvo = preparar(nomes_treino[idx], estat)
                x, alvo = _recorte(x, alvo, lado, gerador)
                xs.append(x)
                alvos.append(alvo)
            entrada = torch.stack(xs)
            alvo = torch.stack(alvos)
            # Um recorte pode cair inteiro em area ignorada; a perda vira NaN e
            # o passo e pulado. Pular e correto: nao ha nada para aprender ali.
            if (alvo != IGNORAR).sum() == 0:
                continue
            otimizador.zero_grad()
            perda = criterio(modelo(entrada), alvo)
            if not torch.isfinite(perda):
                continue
            perda.backward()
            otimizador.step()
            acumulada += float(perda.detach())
            passos += 1
        registro = {
            "epoca": epoca,
            "perda_treino": round(acumulada / passos, 6) if passos else None,
            "passos": passos,
            "perda_validacao": perda_media(modelo, nomes_valid, estat, criterio),
            "segundos_acumulados": round(time.time() - inicio, 2),
        }
        historico.append(registro)
        # SELECAO DE EPOCA PELA VALIDACAO, e por que ela deixou de ser opcional.
        #
        # O peso gravado era o da ULTIMA epoca, e a perda de validacao oscila:
        # numa corrida de 30 epocas ela passou por 0,151 na epoca 25 e voltou a
        # 0,196 na 30. Gravar a ultima significa gravar o sorteio daquela
        # epoca -- e foi assim que 30 epocas mediram PIOR que 12, invertendo a
        # leitura sobre o teto.
        #
        # A selecao usa `validacao`, que e split separado do `teste` em que a
        # medida final acontece. Escolher pelo teste seria escolher a epoca que
        # vai bem no juiz.
        atual = registro["perda_validacao"]
        if (selecionar_por_validacao and atual is not None
                and math.isfinite(atual)
                and (melhor["perda"] is None or atual < melhor["perda"])):
            melhor.update(
                perda=atual, epoca=epoca,
                estado={c: v.detach().clone()
                        for c, v in modelo.state_dict().items()})
        # Progresso vai para stderr: stdout carrega o manifesto JSON, e
        # misturar as duas coisas obrigaria quem consome a saida a filtrar.
        # Uma corrida de horas sem sinal de vida e indistinguivel de uma
        # travada.
        print(f"epoca {epoca}/{epocas} "
              f"treino={registro['perda_treino']} "
              f"validacao={registro['perda_validacao']} "
              f"({registro['segundos_acumulados']}s)",
              file=sys.stderr, flush=True)

    estado = (melhor["estado"] if melhor["estado"] is not None
              else modelo.state_dict())
    hash_estado = hash_do_estado(estado)

    manifesto = {
        "tipo": "unet_pequena_sen1floods11",
        "etapa": "A4",
        "quando_utc": agora_utc(),
        "commit": commit_atual(),
        "torch": torch.__version__,
        "dispositivo": "cpu",
        "seed": seed,
        "encoder_pretreinado": pretreino,
        "epoca_escolhida": melhor["epoca"],
        "perda_validacao_da_escolhida": melhor["perda"],
        "criterio_de_escolha": (
            "menor perda de validacao entre as epocas; o split de validacao e "
            "separado do teste, onde a medida final acontece"
            if selecionar_por_validacao else "ultima epoca"),
        "inicializacao_do_encoder": ("pre-treino mascarado sem rotulo"
                                     if pretreino else "aleatoria"),
        "contrato_do_encoder": contrato_do_encoder,
        "hiperparametros": {"epocas": epocas, "base": base, "lado": lado,
                            "lote": lote, "taxa": taxa,
                            "threads": threads or torch.get_num_threads()},
        "split": {"treino": len(nomes_treino), "validacao": len(nomes_valid),
                  "limite_treino": limite_treino,
                  "limite_validacao": limite_validacao},
        "parametros": contar_parametros(modelo),
        "normalizacao": estat,
        "segundos_estatisticas": segundos_estat,
        "historico": historico,
        "hash_estado": hash_estado,
        "hash_estado_completo": hash_completo_do_estado(estado),
        "nao_e": ("checkpoint de fumaca nao e modelo treinado; ele prova que a "
                  "esteira roda e reproduz, nao que a rede aprendeu"),
    }
    selar(manifesto)
    return {"manifesto": manifesto, "estado": estado}



def gravar(resultado: dict, rotulo: str, saida: Path = SAIDA) -> Path:
    saida.mkdir(parents=True, exist_ok=True)
    torch.save(resultado["estado"], saida / f"unet_{rotulo}.pt")
    destino = saida / f"unet_{rotulo}_manifesto.json"
    destino.write_text(
        json.dumps(resultado["manifesto"], indent=2, ensure_ascii=False),
        encoding="utf-8")
    return destino


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=int, default=2)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--lado", type=int, default=256)
    parser.add_argument("--lote", type=int, default=2)
    parser.add_argument("--taxa", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--limite-treino", type=int, default=None)
    parser.add_argument("--limite-validacao", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--encoder-pretreinado", type=Path, default=None,
                        help="pesos de scripts/sar/pretreino_mascarado.py")
    parser.add_argument("--rotulo", default="fumaca")
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

    resultado = treinar(
        epocas=args.epocas, base=args.base, lado=args.lado, lote=args.lote,
        taxa=args.taxa, seed=args.seed, limite_treino=args.limite_treino,
        limite_validacao=args.limite_validacao, threads=args.threads,
        encoder_pretreinado=args.encoder_pretreinado)
    if args.gravar:
        destino = gravar(resultado, args.rotulo)
        resultado["manifesto"]["gravado_em"] = str(destino)
    print(json.dumps(resultado["manifesto"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
