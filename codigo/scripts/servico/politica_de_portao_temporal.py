"""As tres politicas possiveis para G6, executadas em sombra sobre o produto.

POR QUE ESTE MODULO EXISTE, e por que ele nao liga portao nenhum:

`CONTRATO_V2_CANDIDATO_TEMPORAL.md` mediu que ligar G6 hoje faria as regioes
servidas pararem de emitir escore, e deixou tres saidas admissiveis sem
escolher. Escolher exige ver o efeito das TRES sobre as MESMAS requisicoes, e
isso ninguem tinha rodado: o shadow anterior comparou `v1` contra uma `v2`
so.

Aqui as tres sao funcoes puras sobre `(maturidade, estado de suporte)`, e a
comparacao roda sobre as requisicoes canonicas reais do `svc02`. Nenhuma
delas e aplicada: o modulo devolve tabela e diferenca, e a decisao continua
sendo humana e versionada.

-------------------------------------------------------------------------
A CADEIA QUE NENHUMA DAS TRES PODE COLAPSAR
-------------------------------------------------------------------------

    evidencia -> avaliacao -> suporte -> POLITICA -> decisao

`suporte_cientifico` para em `suporte`. O que este arquivo acrescenta e
exatamente o degrau seguinte, isolado -- para que trocar de politica nao exija
tocar em quem avalia, e para que a avaliacao nao mude de valor porque a
politica mudou.

-------------------------------------------------------------------------
AS TRES
-------------------------------------------------------------------------

  exibir          G6 avalia e NUNCA decide. E o comportamento `v1` de hoje,
                  com a diferenca de que agora esta escrito como politica em
                  vez de ser o efeito de um booleano desligado.

  portao          G6 decide sempre: qualquer estado diferente de `supported`
                  recusa. E a `v2` do documento candidato.

  por_maturidade  G6 decide ONDE O PRODUTO AFIRMA VALIDADE LOCAL. A ideia e
                  que a recusa temporal so acrescenta informacao quando ela
                  CONTRADIZ o que a maturidade ja promete:

                    `validado` / `mvp_local`  -- o produto diz que a regiao tem
                        modelo proprio ajustado nela. Dizer isso sem suporte
                        temporal e afirmar validade que ninguem demonstrou:
                        aqui G6 recusa.

                    `transferencia_*`  -- o produto ja declara que o modelo nao
                        viu a regiao. A recusa temporal nao acrescenta cautela
                        nova; ela so apaga o escore, e o estado de suporte
                        viaja como limitacao declarada.

                  Ela NAO e um meio-termo de conveniencia: e a unica das tres
                  em que a consequencia da recusa depende do que o produto
                  esta afirmando.

Uso:
    python scripts/servico/politica_de_portao_temporal.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))

import suporte_cientifico as sup  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

# O contrato NAO e importado aqui, e a razao e estrutural: desde a v2 e o
# `svc02` que chama `decidir`, entao importa-lo no topo fecharia um ciclo. A
# separacao que o modulo afirma no cabecalho -- a politica e o degrau seguinte
# a avaliacao, isolado dela -- passa a valer tambem no grafo de imports: as
# tres politicas sao funcoes puras de `(maturidade, estado)` e nao precisam de
# contrato nenhum para existir. Quem precisa e `sombra`, que RODA requisicoes,
# e ela importa onde usa.

OUT = REPO / "local_runs" / "politica-portao-temporal"

EXIBIR = "exibir"
PORTAO = "portao"
POR_MATURIDADE = "por_maturidade"
POLITICAS = (EXIBIR, PORTAO, POR_MATURIDADE)

# Maturidades em que o produto AFIRMA validade local. A lista e curta de
# proposito: ela e o criterio inteiro da terceira politica.
AFIRMA_VALIDADE_LOCAL = ("validado", "mvp_local")

DESCRICAO = {
    EXIBIR: ("G6 avalia e nunca decide; o estado de suporte viaja no payload "
             "como informacao"),
    PORTAO: ("G6 decide sempre: estado diferente de `supported` recusa a "
             "requisicao"),
    POR_MATURIDADE: ("G6 decide onde a maturidade afirma validade local; onde "
                     "o produto ja declara transferencia, o estado de suporte "
                     "vira limitacao declarada"),
}

CUSTO = {
    EXIBIR: ("o painel que promete 'expor validade por regiao' precisa dizer "
             "que expoe, e nao que exige"),
    PORTAO: ("a demonstracao de ponta a ponta deixa de emitir escore em todas "
             "as regioes servidas e vira demonstracao de recusa auditavel"),
    POR_MATURIDADE: ("duas regras de recusa coexistem, e a documentacao "
                     "precisa explicar por que a mesma evidencia tem "
                     "consequencia diferente conforme a maturidade"),
}


def decidir(politica: str, maturidade: str | None, estado: str | None) -> dict:
    """A decisao de UMA politica, sem tocar em escore nem em avaliacao.

    `estado is None` e ausencia de avaliacao, e nao `not_testable`. A politica
    conservadora trata ausencia como nao demonstrado -- e registra que a razao
    e ausencia, porque confundir os dois faria a falta da matriz parecer
    resultado cientifico.
    """
    if politica not in POLITICAS:
        raise ValueError(f"politica desconhecida: {politica!r}")
    demonstrado = bool(estado) and sup.suporte_demonstrado(estado)
    razao_ausencia = estado is None

    if politica == EXIBIR:
        recusa = False
    elif politica == PORTAO:
        recusa = not demonstrado
    else:
        recusa = (not demonstrado
                  and str(maturidade) in AFIRMA_VALIDADE_LOCAL)

    return {
        "politica": politica,
        "maturidade": maturidade,
        "estado_de_suporte": estado,
        "suporte_demonstrado": demonstrado,
        "avaliacao_ausente": razao_ausencia,
        "recusa": recusa,
        "gate_que_nao_fechou": "suporte_temporal" if recusa else None,
        "motivo": (
            "suporte temporal nao demonstrado e a maturidade afirma validade "
            "local: emitir escore seria afirmar validade que ninguem mediu"
            if recusa and politica == POR_MATURIDADE else
            "suporte temporal nao demonstrado" if recusa else
            "a politica nao usa o suporte temporal para decidir"
            if politica == EXIBIR else
            "suporte temporal demonstrado" if demonstrado else
            "o produto ja declara transferencia: a recusa temporal nao "
            "acrescenta cautela, e o estado viaja como limitacao"),
    }


def _estado_do_payload(resposta: dict) -> str | None:
    bloco = resposta.get("suporte_cientifico") or {}
    temporal = bloco.get("temporal") if isinstance(bloco, dict) else None
    if isinstance(temporal, dict):
        return temporal.get("support")
    return bloco.get("support") if isinstance(bloco, dict) else None


def sombra(requisicoes=None, modelos=None) -> dict:
    """As tres politicas sobre as MESMAS requisicoes reais do produto."""
    from svc02_contrato_inferencia import (  # noqa: PLC0415
        POLITICA_PORTAO_TEMPORAL, VERSAO_CONTRATO,
        _requisicoes_de_demonstracao, caixas_das_regioes,
        carregar_matriz_cientifica, carregar_modelos, inferir,
        regioes_com_referencia_local,
    )

    modelos = carregar_modelos() if modelos is None else modelos
    if not modelos:
        return {"executado": False, "motivo": "modelos servidos ausentes"}
    caixas = caixas_das_regioes()
    com_ref = regioes_com_referencia_local()
    matriz = carregar_matriz_cientifica()
    requisicoes = (list(_requisicoes_de_demonstracao(modelos))
                   if requisicoes is None else requisicoes)

    casos = []
    for nome, req in requisicoes:
        r = inferir(req, modelos, caixas, com_ref, matriz)
        estado = _estado_do_payload(r)
        maturidade = r.get("maturidade")
        # G6 so e avaliado depois que G1-G5 fecham. Uma requisicao que ja
        # recusou antes NAO pode mudar de portao por causa da politica -- e
        # esse e o teste de que a ordem da escada se preserva.
        ja_recusou = r.get("status") != "ok"
        por_politica = {}
        for p in POLITICAS:
            d = decidir(p, maturidade, estado)
            if ja_recusou:
                d = {**d, "recusa": False,
                     "gate_que_nao_fechou": None,
                     "motivo": (f"a requisicao ja parou em "
                                f"'{r.get('gate_que_nao_fechou')}', antes de G6")}
            por_politica[p] = d
        casos.append({
            "caso": nome,
            "regiao": r.get("regiao"),
            "status_v1": r.get("status"),
            "gate_v1": r.get("gate_que_nao_fechou"),
            "maturidade": maturidade,
            "estado_de_suporte": estado,
            "emite_escore_hoje": r.get("escore") is not None,
            "por_politica": por_politica,
        })

    resumo = {}
    for p in POLITICAS:
        recusadas = [c["caso"] for c in casos if c["por_politica"][p]["recusa"]]
        emitem = [c["caso"] for c in casos
                  if c["emite_escore_hoje"] and not c["por_politica"][p]["recusa"]]
        resumo[p] = {
            "descricao": DESCRICAO[p],
            "custo_declarado": CUSTO[p],
            "requisicoes_que_passariam_a_recusar": recusadas,
            "n_recusas_novas": len(recusadas),
            "regioes_que_continuariam_emitindo_escore": sorted(
                {c["regiao"] for c in casos
                 if c["emite_escore_hoje"] and not c["por_politica"][p]["recusa"]}),
            "n_requisicoes_que_seguem_emitindo": len(emitem),
        }

    ordem_preservada = all(
        not c["por_politica"][p]["recusa"]
        for c in casos if c["status_v1"] != "ok" for p in POLITICAS)

    return {
        "executado": True,
        "artefato": "politica_de_portao_temporal",
        "schema_version": "1.0.0",
        "gerado_em": agora_utc(),
        "contrato_servido": VERSAO_CONTRATO,
        "politica_em_vigor": POLITICA_PORTAO_TEMPORAL,
        "cadeia": "evidencia -> avaliacao -> suporte -> politica -> decisao",
        "politicas": {p: {"descricao": DESCRICAO[p], "custo": CUSTO[p]}
                      for p in POLITICAS},
        "criterio_da_terceira": {
            "maturidades_que_afirmam_validade_local": list(AFIRMA_VALIDADE_LOCAL),
            "por_que": ("a recusa temporal so acrescenta informacao quando "
                        "CONTRADIZ o que a maturidade ja promete"),
        },
        "casos": casos,
        "resumo_por_politica": resumo,
        "ordem_da_escada_preservada": ordem_preservada,
        "o_que_a_ordem_preservada_significa": (
            "nenhuma requisicao que ja recusava antes de G6 muda de portao em "
            "nenhuma das tres politicas: G6 so pode agir depois de G1-G5"),
        "nao_e": ("nao liga portao, nao sobe versao de contrato e nao altera "
                  "escore: e a comparacao que a decisao exige"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="imprime o artefato")
    ns = ap.parse_args()

    r = sombra()
    if not r.get("executado"):
        print(r.get("motivo"))
        return 2
    r["hash"] = sha256_json({k: v for k, v in r.items() if k != "gerado_em"})
    OUT.mkdir(parents=True, exist_ok=True)
    destino = OUT / "sombra_das_tres_politicas.json"
    destino.write_text(json.dumps(r, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/servico/politica_de_portao_temporal.py\n",
        encoding="utf-8")

    if ns.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0

    print("===SOMBRA DAS TRES POLITICAS DE G6===")
    print(f"contrato servido: {r['contrato_servido']} · "
          f"portao ativo hoje: {r['portao_temporal_ativo_hoje']}")
    largura = max(len(c["caso"]) for c in r["casos"])
    print(f"\n{'caso':{largura}s} {'regiao':12s} {'maturidade':34s} "
          f"{'suporte':14s} " + " ".join(f"{p:>14s}" for p in POLITICAS))
    for c in r["casos"]:
        marcas = " ".join(
            f"{('RECUSA' if c['por_politica'][p]['recusa'] else ('escore' if c['emite_escore_hoje'] else '-')):>14s}"
            for p in POLITICAS)
        print(f"{c['caso']:{largura}s} {str(c['regiao'])[:12]:12s} "
              f"{str(c['maturidade'])[:34]:34s} "
              f"{str(c['estado_de_suporte'])[:14]:14s} {marcas}")
    print("\nresumo")
    for p in POLITICAS:
        s = r["resumo_por_politica"][p]
        print(f"  {p:16s} recusas novas={s['n_recusas_novas']:2d} · "
              f"seguem emitindo={s['n_requisicoes_que_seguem_emitindo']:2d} "
              f"{s['regioes_que_continuariam_emitindo_escore']}")
    print(f"\nordem da escada preservada nas tres: {r['ordem_da_escada_preservada']}")
    print(f"GRAVADO={destino}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
