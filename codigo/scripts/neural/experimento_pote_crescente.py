"""A pergunta que o pote fixo nao podia responder: dado NOVO acumula?

-------------------------------------------------------------------------
POR QUE ESTE EXPERIMENTO, E POR QUE AGORA
-------------------------------------------------------------------------

`experimento_acumulacao` rodou com pote fixo e a catraca recusou cinco rodadas
seguidas -- corretamente, porque nao havia ganho: quinze replicas de "uma epoca
a partir daquele estado" deram media +0,0034 com desvio 0,119.

Esse resultado nao e surpreendente, e a leitura correta dele NAO e "o modelo
nao aprende". Com pote fixo, uma rodada e simplesmente mais uma epoca sobre
dado que ja foi visto seis vezes. Um encoder pequeno sobre 232 chips satura, e
a catraca reportou a saturacao em vez de esconde-la.

A promessa de autonomia da frente nunca foi essa. Ela e: *cada chip novo que
chegar entra sozinho no pre-treino, porque a tarefa e reconstruir o que foi
escondido*. Este arquivo testa exatamente isso.

-------------------------------------------------------------------------
O CONTROLE, e por que sem ele o experimento nao vale nada
-------------------------------------------------------------------------

Pote maior significa mais recortes, e mais recortes significam mais passos. Sem
controle, o braco que recebe dado novo tambem recebe mais gradiente, e o ganho
de DADO fica indistinguivel do ganho de PASSOS. Esse erro ja foi cometido nesta
frente e esta registrado no commit `63e74fe` ("parte do ganho que anunciei era
de mais passos, e nao de mais dado").

O controle aqui e o orcamento de passos, igualado por construcao: cada rodada
usa `recortes_por_chip = ALVO_DE_RECORTES / tamanho_do_pote`, entao os dois
bracos dao ~464 passos por epoca independentemente de quantos chips tem dentro.

    braco CRESCENTE   93 -> 139 -> 186 -> 232 chips, mesmos passos por rodada
    braco FIXO        93 chips nas quatro rodadas, mesmos passos por rodada

A unica coisa que difere entre eles e a DIVERSIDADE do que se ve, e e ela que o
experimento quer medir.

-------------------------------------------------------------------------
O QUE FOI DECLARADO ANTES DE RODAR
-------------------------------------------------------------------------

Com 6 sementes e o desvio ja medido para esta regua (~0,06 apos a troca do
instrumento de BatchNorm), o menor efeito detectavel e da ordem de 0,085. Isso
esta escrito aqui ANTES do resultado de proposito: se o efeito vier menor que
isso, a resposta correta e INDECIDIVEL, e nao "dado novo nao ajuda".

Foi exatamente essa confusao que encerrou a linha das AOIs por um ano de
trabalho -- e a regua existe para que ela nao se repita.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))
sys.path.insert(0, str(REPO / "scripts" / "neural"))

import experimento_acumulacao as ea  # noqa: E402
import pretreino_mascarado as pm  # noqa: E402
from catraca import SAIDA as SAIDA_CATRACA  # noqa: E402
from proveniencia import agora_utc  # noqa: E402
from proveniencia_de_peso import commit_atual, selar  # noqa: E402
from regua import analisar_pareado, mde  # noqa: E402

# 232 chips x 16 recortes: o orcamento da corrida anterior, mantido para que os
# numeros sejam comparaveis com o que ja foi medido.
ALVO_DE_RECORTES = 3712
SEMENTES = (7, 99, 20260903, 11, 2026, 404)
FRACOES = (0.40, 0.60, 0.80, 1.00)

# Declarado ANTES de rodar. Ver o cabecalho.
DESVIO_ESPERADO = 0.06


def _plano(treino: list[str], fracoes, crescente: bool) -> list[dict]:
    """Os potes de cada rodada, com o numero de recortes que iguala os passos.

    No braco fixo o pote e sempre o da primeira fracao -- e o `recortes_por_chip`
    e o mesmo daquela fracao, e nao o do braco crescente. Igualar os passos
    exige que cada braco calcule o seu.
    """
    plano = []
    for f in fracoes:
        quantos = max(1, round(len(treino) * (f if crescente else fracoes[0])))
        plano.append({
            "chips": treino[:quantos],
            "recortes_por_chip": max(1, round(ALVO_DE_RECORTES / quantos)),
        })
    return plano


def rodar(sementes: tuple = SEMENTES, fracoes: tuple = FRACOES,
          threads: int | None = None, saida: Path = SAIDA_CATRACA) -> dict:
    todos = pm.nomes_do_split("treino")
    treino, retidos = pm.dividir_o_pote(todos, pm.CHIPS_RETIDOS)

    bracos = {}
    for nome, crescente in (("crescente", True), ("fixo", False)):
        plano = _plano(treino, fracoes, crescente)
        print(f"=== braco {nome}: "
              f"{[(len(p['chips']), p['recortes_por_chip']) for p in plano]}",
              flush=True)
        bracos[nome] = ea.rodar(
            rodadas=len(plano), epocas_por_rodada=1, sementes=sementes,
            threads=threads, saida=saida, plano_de_rodadas=plano,
            nome_da_arena=f"pretreino_sar_pote_{nome}",
            rotulo_das_rodadas=f"pote_{nome}")

    # A comparacao que interessa: mesma semente, mesma rodada, bracos
    # diferentes. Pareada, porque a variancia entre sementes e maior que o
    # efeito procurado.
    ultima_c = bracos["crescente"]["linha_do_tempo"][-1]["medidas"]
    ultima_f = bracos["fixo"]["linha_do_tempo"][-1]["medidas"]
    # perda: o crescente ganha quando fica MENOR, logo fixo - crescente.
    ganhos = [ultima_f[s] - ultima_c[s] for s in sorted(ultima_c)]
    comparacao = analisar_pareado(ganhos)

    passos = {n: [p["passos_por_epoca"] for p in b["linha_do_tempo"]]
              for n, b in bracos.items()}
    resumo = {
        "tipo": "experimento_pote_crescente",
        "quando_utc": agora_utc(),
        "commit": commit_atual(),
        "pergunta": "dado novo acumula, com o orcamento de passos controlado?",
        "sementes": list(sementes),
        "fracoes": list(fracoes),
        "chips_por_rodada": {
            n: [p["chips_no_pote"] for p in b["linha_do_tempo"]]
            for n, b in bracos.items()},
        "passos_por_epoca": passos,
        "controle_de_passos_ok": max(max(v) - min(v) for v in passos.values())
                                 <= 10,
        "mde_declarado_antes": round(mde(DESVIO_ESPERADO, len(sementes)), 6),
        "desvio_esperado_declarado_antes": DESVIO_ESPERADO,
        "bracos": {n: {"linha_do_tempo": b["linha_do_tempo"],
                       "curva_do_campeao": b["curva_do_campeao"],
                       "rodadas_promovidas": b["rodadas_promovidas"],
                       "rodadas_recusadas": b["rodadas_recusadas"]}
                   for n, b in bracos.items()},
        "comparacao_final": comparacao,
        "veredito": comparacao["veredito"],
        "nao_e": ("evidencia sobre deteccao de agua: a regua e reconstrucao "
                  "mascarada em chip retido, e transferencia e pergunta do "
                  "ajuste fino"),
        "nao_responde": ("se o ganho continua com potes maiores que 232 chips; "
                         "nao ha mais S1 sem rotulo disponivel sem gastar os "
                         "splits em que o ajuste fino mede"),
    }
    selar(resumo)
    return resumo


def redecidir(caminho: Path) -> dict:
    """Reaplica a regra corrente as medidas ja gravadas, sem retreinar nada.

    Existe porque a corrida de 2026-09-04 rodou com o processo carregado ANTES
    de o teste de postos virar primario, e reexecutar 48 epocas de CPU para
    trocar um veredito seria gastar quatro horas para nao mudar dado nenhum. As
    medidas por semente sao as mesmas; so a regra que as le mudou.

    O arquivo de saida diz explicitamente o que e confirmatorio e o que nao e.
    A comparacao entre bracos foi pre-registrada em
    `docs/PRE_REGISTRO_TESTE_DE_POSTOS_2026_09_04.md` antes de o braco `fixo`
    existir; as decisoes rodada-a-rodada DENTRO de cada braco nao foram, e vao
    marcadas como releitura.
    """
    d = json.loads(caminho.read_text(encoding="utf-8"))
    linhas = {n: b["linha_do_tempo"] for n, b in d["bracos"].items()}

    dentro = {}
    for nome, linha in linhas.items():
        campeao_medidas = linha[0]["medidas"]
        dentro[nome] = [
            {"rodada": p["rodada"],
             "chips_no_pote": p["chips_no_pote"],
             "contra": linha[0]["rodada"],
             "analise": analisar_pareado(
                 [campeao_medidas[s] - p["medidas"][s]
                  for s in sorted(p["medidas"])])}
            for p in linha[1:]
        ]

    c = linhas["crescente"][-1]["medidas"]
    f = linhas["fixo"][-1]["medidas"]
    confirmatorio = analisar_pareado([f[s] - c[s] for s in sorted(c)])

    resumo = {
        "tipo": "redecisao_do_pote_crescente",
        "quando_utc": agora_utc(),
        "commit": commit_atual(),
        "origem": str(caminho),
        "regra": ("teste de postos pareado como primario; ver "
                  "docs/PRE_REGISTRO_TESTE_DE_POSTOS_2026_09_04.md"),
        "confirmatorio": {
            "pergunta": ("com o orcamento de passos igualado, o pote que cresce "
                         "termina melhor que o pote fixo?"),
            "pre_registrado": True,
            "mde_declarado_antes": d["mde_declarado_antes"],
            "passos_por_epoca": d["passos_por_epoca"],
            "controle_de_passos_ok": d["controle_de_passos_ok"],
            "ganho_por_semente": {s: round(f[s] - c[s], 6) for s in sorted(c)},
            "analise": confirmatorio,
            "veredito": confirmatorio["veredito"],
        },
        "releitura_dentro_de_cada_braco": {
            "pre_registrado": False,
            "aviso": ("a regra foi fixada depois de ver o braco crescente; "
                      "estes vereditos sao releitura e nao evidencia nova"),
            "bracos": dentro,
        },
        "nao_e": ("evidencia sobre deteccao de agua: a regua e reconstrucao "
                  "mascarada em chip retido"),
    }
    selar(resumo)
    return resumo


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--sementes", default=",".join(str(s) for s in SEMENTES),
                   help="sementes pareadas separadas por virgula")
    p.add_argument("--redecidir", type=Path, default=None,
                   help="reaplica a regra corrente a um JSON ja gravado")
    p.add_argument("--saida", type=Path,
                   default=SAIDA_CATRACA / "experimento_pote_crescente.json")
    args = p.parse_args()

    if args.redecidir is not None:
        r = redecidir(args.redecidir)
        destino = args.redecidir.with_name(
            args.redecidir.stem + "_redecidido.json")
        destino.write_text(json.dumps(r, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        conf = r["confirmatorio"]["analise"]
        print(f"CONFIRMATORIO crescente vs fixo: {conf['veredito']} "
              f"({conf['sementes_no_mesmo_sentido']}, p_postos "
              f"{conf['p_postos']}, mediana {conf['mediana']}, "
              f"mde declarado {r['confirmatorio']['mde_declarado_antes']})")
        for nome, pontos in r["releitura_dentro_de_cada_braco"]["bracos"].items():
            print(f"\nreleitura {nome} (contra a rodada 1):")
            for ponto in pontos:
                a = ponto["analise"]
                print(f"  r{ponto['rodada']} chips={ponto['chips_no_pote']:>3} "
                      f"{a['veredito']:<18} p_postos={a['p_postos']} "
                      f"mediana={a['mediana']}")
        print(f"\ngravado em {destino}")
        return 0

    sementes = tuple(int(s) for s in args.sementes.split(",") if s.strip())
    resumo = rodar(sementes=sementes, threads=args.threads,
                   saida=args.saida.parent)
    args.saida.write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    for nome, b in resumo["bracos"].items():
        print(f"\nbraco {nome}: promovidas {b['rodadas_promovidas']} "
              f"recusadas {b['rodadas_recusadas']}")
        for ponto in b["linha_do_tempo"]:
            print(f"  r{ponto['rodada']} chips={ponto['chips_no_pote']} "
                  f"passos={ponto['passos_por_epoca']} "
                  f"media={ponto['media']} {ponto['veredito']}")
    c = resumo["comparacao_final"]
    print(f"\ncrescente vs fixo na ultima rodada: ganho {c['media']} "
          f"ic95 {c['ic95']} mde {c['mde']} -> {c['veredito']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
