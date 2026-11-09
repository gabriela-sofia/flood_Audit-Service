"""As tres pecas ligadas: rodadas que acumulam, sob uma catraca que nao desce.

-------------------------------------------------------------------------
A PERGUNTA
-------------------------------------------------------------------------

"A cada rodada ele aprende mais?" Ate agora essa pergunta era indecidivel no
REV-P por um motivo mecanico: nao existia rodada seguinte. Toda corrida
comecava do zero, e uma sequencia de sorteios independentes nao tem tendencia
para medir.

Aqui a pergunta fica decidivel:

  peca A   `pretreino_mascarado --retomar` faz a rodada N+1 continuar de onde a
           N parou, com prova de equivalencia tensor a tensor.
  peca C   `regua.analisar_pareado` diz se a diferenca entre rodadas sai do
           ruido, e devolve INDECIDIVEL quando nao sai.
  peca B   `catraca.disputar` so troca o campeao quando sai, e grava a recusa
           quando nao sai.

-------------------------------------------------------------------------
TRES SEMENTES, TRES LINHAGENS, e por que nao uma so
-------------------------------------------------------------------------

Uma linhagem unica daria um numero por rodada, e um numero por rodada nao tem
com que ser pareado -- a catraca ficaria sem o teste que a sustenta. Cada
semente carrega a propria linhagem, e a disputa compara rodada N+1 contra
rodada N NA MESMA SEMENTE. E o mesmo desenho pareado que a frente A ja usava,
agora aplicado ao eixo do tempo em vez do eixo da inicializacao.

-------------------------------------------------------------------------
A RODADA RECUSADA E O UNICO PONTO SUTIL DO ARQUIVO
-------------------------------------------------------------------------

Quando a catraca recusa, o peso volta para o do campeao. Se o gerador de
aleatoriedade voltasse junto, a rodada seguinte partiria do mesmo estado com o
mesmo fluxo de sorteios e produziria BIT A BIT o mesmo resultado -- e seria
recusada para sempre. O experimento entraria num laco que pareceria "o modelo
parou de aprender" quando seria "o experimento parou de tentar".

Entao a catraca protege o PESO e nao o fluxo: apos uma recusa, a proxima rodada
retoma o modelo e o otimizador do campeao e o gerador de onde a tentativa
recusada deixou. O peso nunca regride, e a tentativa seguinte ve outro sorteio
de recortes.

-------------------------------------------------------------------------
O QUE ESTE EXPERIMENTO NAO RESPONDE
-------------------------------------------------------------------------

Se o ganho TRANSFERE. A regua aqui e a perda de reconstrucao em chips retidos:
ela mede se a rodada aprendeu a tarefa dela, em dado que o gradiente nunca
tocou. Se isso vira agua melhor detectada em SAR e pergunta do ajuste fino, e
tem arena propria -- `encoder_sar__iou_sem_permanente__validacao` -- que custa
horas de CPU por rodada em vez de minutos.

E o pote aqui e FIXO. Rodada com pote fixo e, no fundo, mais epocas: ela
responde "insistencia acumula?", nao "dado novo acumula?". A segunda exige pote
crescente com orcamento de passos controlado, porque senao o ganho de mais dado
fica confundido com o de mais passos -- que e exatamente o erro ja cometido e
registrado em `63e74fe`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))
sys.path.insert(0, str(REPO / "scripts" / "neural"))

import pretreino_mascarado as pm  # noqa: E402
from catraca import (  # noqa: E402
    PROMOCOES, SAIDA as SAIDA_CATRACA, arena, campeao, curva_do_campeao,
    disputar, historico,
)
from proveniencia import agora_utc  # noqa: E402
from proveniencia_de_peso import commit_atual, selar  # noqa: E402

ESTADOS = SAIDA_CATRACA / "estados"
SEMENTES = (7, 99, 20260903)


def _estado_da_proxima(do_campeao: dict, da_tentativa: dict) -> dict:
    """Peso do campeao, fluxo de aleatoriedade de onde a tentativa parou.

    A assimetria e deliberada e esta explicada no cabecalho: proteger o fluxo
    junto com o peso faria a tentativa recusada se repetir identica para
    sempre.
    """
    novo = dict(do_campeao)
    novo["gerador"] = da_tentativa["gerador"]
    novo["rng_global"] = da_tentativa["rng_global"]
    return novo


def rodar(rodadas: int = 6, epocas_por_rodada: int = 1,
          sementes: tuple = SEMENTES, base: int = 16, lote: int = 8,
          taxa: float = 1e-3, recortes_por_chip: int = 16,
          threads: int | None = None, saida: Path = SAIDA_CATRACA,
          plano_de_rodadas: list[dict] | None = None,
          nome_da_arena: str = "pretreino_sar",
          rotulo_das_rodadas: str = "rodada",
          sobrescrever: bool = False) -> dict:
    """`plano_de_rodadas` deixa o pote MUDAR entre rodadas.

    Cada entrada e `{"chips": [...], "recortes_por_chip": n}` e substitui o
    pote fixo naquela rodada. Existe para a pergunta que o pote fixo nao pode
    responder -- "dado novo acumula?" --, e o `recortes_por_chip` vem junto
    porque sem ele um pote maior daria mais passos, e o ganho de mais dado
    ficaria confundido com o de mais passos.

    O conjunto retido continua sendo o mesmo em todas as rodadas, venha o pote
    de onde vier.
    """
    ESTADOS.mkdir(parents=True, exist_ok=True)

    # O corte do pote e feito UMA VEZ. Refazer a cada rodada mudaria o conjunto
    # retido junto com o pote, e a curva compararia medidas de conjuntos
    # diferentes -- que e o defeito que este arquivo inteiro existe para nao
    # ter.
    todos = pm.nomes_do_split("treino")
    treino, retidos = pm.dividir_o_pote(todos, pm.CHIPS_RETIDOS)
    if plano_de_rodadas is not None and len(plano_de_rodadas) != rodadas:
        raise ValueError(
            f"plano com {len(plano_de_rodadas)} entradas para {rodadas} "
            f"rodadas; uma rodada sem pote declarado herdaria o da anterior "
            f"em silencio")

    a = arena(nome_da_arena, "perda_retida", "retido_do_treino",
              maior_e_melhor=False)

    # UMA CORRIDA POR ARENA E POR ROTULO, e por que isso deixou de ser
    # implicito.
    #
    # A arena vive em disco entre corridas; os pesos de uma corrida nova
    # comecam do zero. Em 13/09/2026 uma corrida nova entrou nas arenas de
    # 04/09: a rodada 1 dela, que partiu do zero, foi julgada contra o campeao
    # da corrida antiga, recusada, e por isso a rodada 2 partiu do zero de
    # novo -- nenhuma rodada podia acumular. Na mesma corrida os arquivos
    # `tentativa_*_r{r}_s{s}.pt` foram sobrescritos: `tentativa_pote_crescente
    # _r2_s7.pt` hoje guarda uma rodada 1 de 467 passos, e nao a rodada 2 de
    # 04/09. A linhagem gravada nos manifestos aponta para hashes cujo arquivo
    # ja nao existe.
    alvos = [ESTADOS / f"tentativa_{rotulo_das_rodadas}_r{r}_s{s}.pt"
             for r in range(1, rodadas + 1) for s in sementes]
    alvos += [ESTADOS / f"campeao_{rotulo_das_rodadas}_s{s}.pt"
              for s in sementes]
    ocupados = [p.name for p in alvos if p.exists()]
    if not sobrescrever and (campeao(a, saida) is not None or ocupados):
        raise RuntimeError(
            f"a arena '{a['id']}' ou os estados '{rotulo_das_rodadas}' ja "
            f"pertencem a outra corrida ({len(ocupados)} arquivos); uma "
            f"corrida nova aqui seria julgada contra o campeao antigo e "
            f"apagaria a linhagem dele. Use outro nome de arena/rotulo, ou "
            f"`sobrescrever=True` sabendo disso")

    estados_correntes: dict[int, Path | None] = {s: None for s in sementes}
    linha = []
    inicio = time.time()

    for r in range(1, rodadas + 1):
        medidas, manifestos, tentativas = {}, {}, {}
        do_plano = plano_de_rodadas[r - 1] if plano_de_rodadas else {}
        pote_da_rodada = do_plano.get("chips", treino)
        recortes_da_rodada = do_plano.get("recortes_por_chip",
                                          recortes_por_chip)
        for s in sementes:
            resultado = pm.treinar(
                epocas_alvo=epocas_por_rodada, seed=s, base=base, lote=lote,
                taxa=taxa, recortes_por_chip=recortes_da_rodada,
                threads=threads, chips=pote_da_rodada, chips_retidos=retidos,
                retomar=estados_correntes[s])
            caminho = ESTADOS / f"tentativa_{rotulo_das_rodadas}_r{r}_s{s}.pt"
            torch.save(resultado["continuidade"], caminho)
            tentativas[s] = caminho
            medidas[str(s)] = resultado["manifesto"]["perda_retida_final"]
            manifestos[s] = {
                "hash_estado": resultado["manifesto"]["hash_estado"],
                "peso_pai": resultado["manifesto"]["peso_pai"],
                "epocas_acumuladas": resultado["manifesto"][
                    "epocas_acumuladas"],
                "perda_treino": resultado["manifesto"]["historico"][-1][
                    "perda_treino"],
            }
            print(f"[rodada {r}] semente {s}: retida="
                  f"{medidas[str(s)]} treino="
                  f"{manifestos[s]['perda_treino']} "
                  f"({round(time.time() - inicio)}s)", flush=True)

        veredito = disputar(
            a, f"{rotulo_das_rodadas}_{r}", medidas,
            proveniencia={"por_semente": {str(k): v
                                          for k, v in manifestos.items()},
                          "epocas_por_rodada": epocas_por_rodada,
                          "chips_no_pote": len(pote_da_rodada),
                          "recortes_por_chip": recortes_da_rodada,
                          "passos_por_epoca": (len(pote_da_rodada)
                                               * recortes_da_rodada // lote),
                          "chips_retidos": len(retidos)},
            saida=saida)

        promovida = veredito["veredito"] in PROMOCOES
        for s in sementes:
            if promovida:
                destino = ESTADOS / f"campeao_{rotulo_das_rodadas}_s{s}.pt"
                torch.save(torch.load(tentativas[s], map_location="cpu",
                                      weights_only=False), destino)
                estados_correntes[s] = destino
            elif estados_correntes[s] is not None:
                atual = torch.load(estados_correntes[s], map_location="cpu",
                                   weights_only=False)
                recusada = torch.load(tentativas[s], map_location="cpu",
                                      weights_only=False)
                torch.save(_estado_da_proxima(atual, recusada),
                           estados_correntes[s])

        linha.append({
            "rodada": r,
            "chips_no_pote": len(pote_da_rodada),
            "recortes_por_chip": recortes_da_rodada,
            "passos_por_epoca": len(pote_da_rodada) * recortes_da_rodada // lote,
            "medidas": medidas,
            "media": veredito["media"],
            "veredito": veredito["veredito"],
            "por_que": veredito["por_que"],
            "ganho_pareado": (veredito["analise"]["media"]
                              if veredito["analise"] else None),
            "ic95_do_ganho": (veredito["analise"]["ic95"]
                              if veredito["analise"] else None),
            "mde": (veredito["analise"]["mde"] if veredito["analise"]
                    else None),
        })
        print(f"[rodada {r}] {veredito['veredito']}: {veredito['por_que']}",
              flush=True)

    curva = curva_do_campeao(a, saida)
    promovidas = [p for p in linha if p["veredito"] in PROMOCOES]
    resumo = {
        "tipo": "experimento_acumulacao_neural",
        "quando_utc": agora_utc(),
        "commit": commit_atual(),
        "arena": a,
        "pergunta": ("com retomada e catraca, a rodada N+1 fica melhor que a "
                     "N em dado retido -- e quando ela nao fica, isso aparece?"),
        "rodadas": rodadas,
        "epocas_por_rodada": epocas_por_rodada,
        "sementes": list(sementes),
        "chips_no_pote": len(treino),
        "chips_retidos": len(retidos),
        "plano_de_rodadas": ([{"chips": len(p["chips"]),
                               "recortes_por_chip": p["recortes_por_chip"]}
                              for p in plano_de_rodadas]
                             if plano_de_rodadas else None),
        "linha_do_tempo": linha,
        "curva_do_campeao": curva,
        "rodadas_promovidas": len(promovidas),
        "rodadas_recusadas": rodadas - len(promovidas),
        "campeao_final": campeao(a, saida),
        "disputas_registradas": len(historico(a, saida)),
        "segundos": round(time.time() - inicio, 1),
        "nao_responde": ("se o ganho transfere para deteccao de agua; a regua "
                         "aqui e reconstrucao mascarada, e transferencia e "
                         "pergunta do ajuste fino"),
        "nao_e": ("evidencia de que mais DADO acumula: o pote e fixo, entao a "
                  "rodada e mais epoca. Pote crescente exige controle de "
                  "orcamento de passos, senao mais dado fica confundido com "
                  "mais passos"),
    }
    selar(resumo)
    return resumo


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rodadas", type=int, default=6)
    p.add_argument("--epocas-por-rodada", type=int, default=1)
    p.add_argument("--sementes", default=",".join(str(s) for s in SEMENTES))
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--recortes-por-chip", type=int, default=16)
    p.add_argument("--saida", type=Path,
                   default=SAIDA_CATRACA / "experimento_acumulacao.json")
    args = p.parse_args()

    resumo = rodar(rodadas=args.rodadas,
                   epocas_por_rodada=args.epocas_por_rodada,
                   sementes=tuple(int(s) for s in args.sementes.split(",")),
                   base=args.base, threads=args.threads,
                   recortes_por_chip=args.recortes_por_chip)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(json.dumps({k: v for k, v in resumo.items()
                      if k not in ("campeao_final",)},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
