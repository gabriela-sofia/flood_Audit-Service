"""D1 -- o metodo atual medido, e o juiz neural que so pode restringir.

-------------------------------------------------------------------------
O JUIZ NAO PODE APROVAR, SO DERRUBAR
-------------------------------------------------------------------------

Usar um modelo de linguagem para verificar o que um modelo de linguagem
escreveu e circular por natureza. O desenho que quebra a circularidade e este:

    publica  <=>  a regra lexica aprova  E  o juiz nao discorda

O juiz nunca ve o payload, nunca ve o que foi gerado e nunca sabe qual e a
resposta desejada. Ele ve um par `(claim, trecho)` e responde se o trecho
sustenta o claim. E, sobretudo, ele so pode SUBTRAIR: um claim que a regra
lexica reprovou continua reprovado, digam o que disserem os pesos.

Isso torna o pior caso do juiz inofensivo. Se ele alucinar "sustenta" para
tudo, o sistema volta a ser exatamente o que era antes desta frente. Se ele
alucinar "nao sustenta" para tudo, o sistema fica mais conservador -- publica
menos, nunca publica errado.

A frente C reforcou por que isso importa: la, fundir duas listas quando uma
estava errada produziu resultado PIOR que usar so a certa. Conjuncao nao tem
esse problema, porque nao mistura -- ela exige acordo.

-------------------------------------------------------------------------
O CUSTO, e ele e real
-------------------------------------------------------------------------

Conjuncao so aperta. Ela nao conserta falso NEGATIVO -- a fonte certa que a
regra lexica reprova por falta de sobreposicao continua reprovada, e o juiz
nao tem como salva-la. Isso e uma escolha declarada: no REV-P, publicar de
menos e recuperavel e publicar de mais nao e.

A familia `sustenta_parafrase` existe justamente para medir esse custo em vez
de deixa-lo implicito.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "verificacao"))

from casos_de_suporte import FAMILIAS, todos  # noqa: E402
from claims import sustenta  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "verificacao-suporte"

PROMPT_JUIZ = (
    "Voce recebe um TRECHO de um documento e uma AFIRMACAO. Responda apenas "
    "se o trecho, sozinho, sustenta a afirmacao.\n"
    "Responda SUSTENTA se tudo o que a afirmacao diz esta no trecho.\n"
    "Responda NAO_SUSTENTA se a afirmacao contradiz o trecho, inverte um "
    "sentido, troca um numero, generaliza alem do que o trecho permite, ou "
    "fala de outro assunto.\n"
    "Na duvida, responda NAO_SUSTENTA.\n"
    "Responda com uma unica palavra: SUSTENTA ou NAO_SUSTENTA."
)


def julgar_local(claim: str, trecho: str, url: str, modelo: str,
                 tempo_limite: float = 60.0) -> tuple[bool | None, str]:
    """Pergunta ao motor local. Devolve `(discorda, bruto)`.

    `None` significa indisponivel -- e indisponivel NAO e discordancia. Um juiz
    fora do ar nao pode derrubar frase nenhuma, pela mesma razao que a
    proveniencia ausente nao derruba a resposta: e informacao sobre o
    resultado, nao condicao para ele.
    """
    import urllib.error
    import urllib.request

    corpo = json.dumps({
        "model": modelo,
        "messages": [
            {"role": "system", "content": PROMPT_JUIZ},
            {"role": "user",
             "content": f"TRECHO:\n{trecho}\n\nAFIRMACAO:\n{claim}"},
        ],
        "temperature": 0,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=corpo, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=tempo_limite) as r:
            dados = json.loads(r.read().decode("utf-8"))
        texto = (dados["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

    plana = texto.upper().replace("-", "_")
    if "NAO_SUSTENTA" in plana or "NÃO_SUSTENTA" in plana:
        return True, texto
    if "SUSTENTA" in plana:
        return False, texto
    # resposta fora do vocabulario nao vira voto: o juiz nao respondeu.
    return None, texto


def avaliar(usar_juiz: bool = False, url: str | None = None,
            modelo: str | None = None) -> dict:
    url = url or os.environ.get(
        "LOCAL_LLM_URL", "http://127.0.0.1:11434/v1/chat/completions")
    modelo = modelo or os.environ.get("LOCAL_LLM_MODEL", "qwen2.5:3b-instruct")

    por_caso = []
    latencias = []
    for caso in todos():
        aprovou_lexico, fracao = sustenta(caso["claim"],
                                          caso["texto_do_trecho"])
        discorda, bruto = None, ""
        if usar_juiz:
            t0 = time.time()
            discorda, bruto = julgar_local(caso["claim"],
                                           caso["texto_do_trecho"],
                                           url, modelo)
            latencias.append(round(time.time() - t0, 3))

        # A conjuncao: o juiz so subtrai.
        conjuncao = bool(aprovou_lexico) and not (discorda is True)

        por_caso.append({
            "id": caso["id"], "familia": caso["familia"],
            "esperado": caso["sustenta"],
            "lexico": bool(aprovou_lexico), "fracao_lexica": fracao,
            "juiz_discorda": discorda,
            "juiz_bruto": bruto[:80],
            "conjuncao": conjuncao,
            "lexico_correto": bool(aprovou_lexico) == caso["sustenta"],
            "conjuncao_correta": conjuncao == caso["sustenta"],
        })

    def resumir(sub: list, chave: str) -> dict:
        n = len(sub)
        if n == 0:
            return {"n": 0, "acertos": 0, "acuracia": None,
                    "falsos_positivos": 0, "falsos_negativos": 0}
        acertos = sum(1 for c in sub if c[f"{chave}_correto" if chave == "lexico"
                                         else "conjuncao_correta"])
        decisao = "lexico" if chave == "lexico" else "conjuncao"
        fp = sum(1 for c in sub if c[decisao] and not c["esperado"])
        fn = sum(1 for c in sub if not c[decisao] and c["esperado"])
        return {"n": n, "acertos": acertos, "acuracia": round(acertos / n, 4),
                "falsos_positivos": fp, "falsos_negativos": fn}

    resumo = {
        "tipo": "medida_de_suporte",
        "etapa": "D1" if not usar_juiz else "D1/D3",
        "quando_utc": agora_utc(),
        "casos": len(por_caso),
        "juiz_usado": usar_juiz,
        "modelo_do_juiz": modelo if usar_juiz else None,
        "juiz_indisponivel": sum(1 for c in por_caso
                                 if c["juiz_discorda"] is None) if usar_juiz else None,
        "latencia_mediana_s": (
            round(sorted(latencias)[len(latencias) // 2], 3)
            if latencias else None),
        "lexico": {
            "geral": resumir(por_caso, "lexico"),
            "por_familia": {f: resumir([c for c in por_caso
                                        if c["familia"] == f], "lexico")
                            for f in FAMILIAS},
        },
        "regra": ("publica <=> a regra lexica aprova E o juiz nao discorda; o "
                  "juiz so subtrai"),
        "custo_declarado": ("conjuncao nao conserta falso negativo: a fonte "
                            "certa que o lexico reprova continua reprovada"),
        "nao_e": ("medida de verdade da afirmacao: o juiz decide se o trecho "
                  "SUSTENTA a frase, e uma frase falsa bem sustentada por uma "
                  "fonte errada passa igual"),
    }
    if usar_juiz:
        resumo["conjuncao"] = {
            "geral": resumir(por_caso, "conjuncao"),
            "por_familia": {f: resumir([c for c in por_caso
                                        if c["familia"] == f], "conjuncao")
                            for f in FAMILIAS},
        }
    resumo["hash_resumo"] = sha256_json(
        {k: v for k, v in resumo.items()
         if k not in ("quando_utc", "latencia_mediana_s")})
    return {"resumo": resumo, "por_caso": por_caso}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--juiz", action="store_true",
                        help="consulta o motor local (D3)")
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

    resultado = avaliar(usar_juiz=args.juiz)
    if args.gravar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        sufixo = "_com_juiz" if args.juiz else "_lexico"
        destino = SAIDA / f"medida_de_suporte{sufixo}.json"
        destino.write_text(
            json.dumps({**resultado["resumo"],
                        "por_caso": resultado["por_caso"]},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        resultado["resumo"]["gravado_em"] = str(destino)
    print(json.dumps(resultado["resumo"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
