"""A regra do referencial local, medida em vez de enunciada.

POR QUE ESTA AUDITORIA EXISTE:

o projeto ja cometeu quatro vezes a mesma classe de erro, e as quatro estao
documentadas -- tres no docstring do `ds03` e a quarta em
`metodo_aplicacao_sem_rotulo_local_v1.md`. A raiz e sempre a mesma:

    uma variavel fisica so e comparavel entre regioes se for medida contra um
    referencial do proprio fenomeno, e nao contra um referencial externo.

HAND mede altura acima da drenagem mais proxima; TWI e adimensional. As duas
atravessam regiao porque a pergunta que fazem e sobre o lugar, respondida em
unidades do proprio lugar. Elevacao ao nivel do mar nao e: o mar nao participa
do fenomeno de inundacao urbana em Curitiba, a 900 metros de altitude.

-------------------------------------------------------------------------
O QUE ESTA AUDITORIA ENCONTROU
-------------------------------------------------------------------------

A correcao ja existe. `elevation_rel_m` foi criada no `ds04` como REGRA
PERMANENTE do contrato -- o `ds03` a descreve como "a leitura que pode ser
comparada ou agrupada ENTRE fontes", aplicada "para todo redutor, presente ou
futuro". Ela esta na tabela unica, preenchida.

E nenhum modelo servido a usa. O `svc01` mantem `elevation_m` em `TOPO` e
declara a limitacao em texto -- "altitude absoluta, nao grandeza causal
comparavel entre cidades de altitude de base diferente" --, delegando ao portao
de dominio do `SVC-02` medir o estrago por requisicao.

Medido na transferencia que o projeto de fato serve (pool sem Curitiba ->
Curitiba):

    elevation_m       0,0% dos pontos no dominio do treino, 7,01 desvios
    elevation_rel_m  94,5% dos pontos no dominio do treino, 0,72 desvio

A variavel que o contrato marcou como nao-comparavel e a unica das quatro que
quebra. As outras tres -- hand_m, twi_dinf, slope_deg -- ficam entre 96% e 98%,
e sao exatamente as relativas ou adimensionais. A regra nao e boa pratica
importada: ela preve quais variaveis quebram, e acerta.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao troca variavel de modelo servido, nao reajusta e nao muda o que o produto
responde -- isso e decisao de maturidade e de versao do contrato. Ele mede o
custo de manter o referencial errado e o ganho disponivel, para que a troca
seja decidida com numero.

Uso:
    python scripts/treino/aud_desenho04_referencial_local.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_pool_fluvial_v2.csv"
OUT = REPO / "local_runs" / "aud-desenho-04"

# Pares (referencial externo -> referencial do fenomeno) que o contrato ja
# oferece. A chave e a variavel em uso; o valor e a correcao disponivel.
PARES_DE_REFERENCIAL = {"elevation_m": "elevation_rel_m"}

# Variaveis do vetor de terreno servido, na ordem do `svc01.TOPO`.
TOPO = ("elevation_m", "slope_deg", "hand_m", "twi_dinf")

# Faixa do treino usada como dominio, e piso de cobertura abaixo do qual a
# transferencia deixa de ser sustentavel. O 5-95% e o mesmo criterio do portao
# G5 do SVC-02; o piso e declarado aqui e nao estimado.
PERCENTIS_DO_DOMINIO = (5, 95)
PCT_MINIMO_DE_COBERTURA = 80.0


def cobertura_de_dominio(treino: pd.Series, alvo: pd.Series) -> dict:
    """Quanto da regiao alvo cai dentro do que o treino viu."""
    t = treino.dropna().to_numpy(dtype=float)
    a = alvo.dropna().to_numpy(dtype=float)
    if not len(t) or not len(a):
        return {"avaliado": False, "motivo": "sem dado em treino ou alvo"}
    lo, hi = np.percentile(t, PERCENTIS_DO_DOMINIO)
    dentro = float(((a >= lo) & (a <= hi)).mean()) * 100
    # Diferenca padronizada de media (Cohen), o mesmo indicador que
    # `metodo_aplicacao_sem_rotulo_local` usou para achar o problema.
    sp = float(np.sqrt((t.var() + a.var()) / 2))
    return {
        "avaliado": True,
        "faixa_treino": [round(float(lo), 2), round(float(hi), 2)],
        "pct_alvo_no_dominio": round(dentro, 1),
        "diferenca_padronizada": round(abs(t.mean() - a.mean()) / sp, 2)
        if sp else None,
        "sustenta_transferencia": dentro >= PCT_MINIMO_DE_COBERTURA,
    }


def auditar_transferencia(pool: pd.DataFrame, alvo: str,
                          relevo: str = "PLANO_OU_ONDULADO") -> dict:
    """Cada variavel do vetor, mais as correcoes disponiveis."""
    a = pool[pool.fonte == alvo]
    t = pool[(pool.fonte != alvo) & (pool.classe_relevo == relevo)]
    if not len(a) or not len(t):
        return {"avaliado": False, "motivo": f"sem treino ou alvo para {alvo}"}

    variaveis = list(TOPO) + [v for v in PARES_DE_REFERENCIAL.values()
                              if v in pool.columns]
    por_variavel = {v: cobertura_de_dominio(t[v], a[v])
                    for v in variaveis if v in pool.columns}

    ganhos = {}
    for absoluta, relativa in PARES_DE_REFERENCIAL.items():
        ea, er = por_variavel.get(absoluta), por_variavel.get(relativa)
        if not (ea and er and ea.get("avaliado") and er.get("avaliado")):
            continue
        ganhos[absoluta] = {
            "correcao_disponivel": relativa,
            "cobertura_atual": ea["pct_alvo_no_dominio"],
            "cobertura_corrigida": er["pct_alvo_no_dominio"],
            "ganho_em_pontos_percentuais": round(
                er["pct_alvo_no_dominio"] - ea["pct_alvo_no_dominio"], 1),
            "em_uso_no_modelo_servido": absoluta,
            "veredito": ("corrigivel_e_nao_corrigida"
                         if not ea["sustenta_transferencia"]
                         and er["sustenta_transferencia"] else "sem_ganho_claro"),
        }
    quebram = sorted(v for v, e in por_variavel.items()
                     if e.get("avaliado") and not e["sustenta_transferencia"])
    return {
        "avaliado": True, "alvo": alvo, "n_alvo": int(len(a)),
        "n_treino": int(len(t)), "por_variavel": por_variavel,
        "variaveis_que_quebram": quebram,
        "correcoes_disponiveis": ganhos,
    }


def main() -> int:
    if not POOL.exists():
        print(f"pool ausente: {POOL}")
        return 2
    pool = pd.read_csv(POOL, low_memory=False)
    r = auditar_transferencia(pool, "curitiba")
    if not r["avaliado"]:
        print(r["motivo"])
        return 1

    relatorio = {
        "artefato": "aud_desenho04_referencial_local",
        "schema_version": "1.0.0",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "regra": ("uma variavel fisica so e comparavel entre regioes se for "
                  "medida contra um referencial do proprio fenomeno"),
        "criterios": {
            "percentis_do_dominio": list(PERCENTIS_DO_DOMINIO),
            "pct_minimo_de_cobertura": PCT_MINIMO_DE_COBERTURA,
        },
        **r,
        "o_que_o_projeto_ja_declara": {
            "ds03": ("elevation_rel_m e regra permanente do contrato, 'a "
                     "leitura que pode ser comparada ou agrupada ENTRE fontes'"),
            "svc01": ("mantem elevation_m em TOPO e declara a limitacao em "
                      "texto, delegando ao portao G5 medir por requisicao"),
            "lacuna": ("a correcao existe na tabela, e declarada permanente e "
                       "nao chegou ao vetor do modelo servido"),
        },
        "nao_e": ("nao troca variavel de modelo servido, nao reajusta e nao "
                  "muda o que o produto responde"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destino = OUT / "referencial_local.json"
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/treino/aud_desenho04_referencial_local.py\n",
        encoding="utf-8")

    print(f"transferencia pool -> {r['alvo']} "
          f"(treino n={r['n_treino']:,}, alvo n={r['n_alvo']:,})\n")
    print(f"{'variavel':20s} {'faixa treino':>24s} {'% no dominio':>13s} "
          f"{'dif padr':>9s}  sustenta")
    for v, e in r["por_variavel"].items():
        if not e.get("avaliado"):
            continue
        faixa = f"[{e['faixa_treino'][0]}, {e['faixa_treino'][1]}]"
        print(f"{v:20s} {faixa:>24s} {e['pct_alvo_no_dominio']:12.1f}% "
              f"{e['diferenca_padronizada']:9.2f}  "
              f"{'sim' if e['sustenta_transferencia'] else 'NAO'}")
    for v, g in r["correcoes_disponiveis"].items():
        print(f"\n{v} -> {g['correcao_disponivel']}: "
              f"{g['cobertura_atual']}% para {g['cobertura_corrigida']}% "
              f"(+{g['ganho_em_pontos_percentuais']} p.p.) — {g['veredito']}")
    print(f"\nRELATORIO={destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
