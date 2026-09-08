"""O teste da segunda cidade: a lei do TWI se repete em Chicago?

O QUE ESTA SENDO TESTADO, e por que ele decide o projeto:

Em Nova York, dentro do mesmo dia e com a chuva a 1 km, mediu-se

    twi_dinf          +0,0390  IC95 [+0,0317; +0,0472]
    twi_dinf x chuva  +0,0397  IC95 [+0,0221; +0,0560]
    hand_m            nulo, e hand_m x chuva tambem

Enquanto isso existir num lugar so, nao ha como distinguir lei fisica de
particularidade de Nova York -- e transferir para Recife seria trocar fisica por
geografia. Chicago responde exatamente essa pergunta.

O ESTIMADOR E IMPORTADO, NAO REESCRITO. `centrar`, `estratos_utilizaveis`,
`com_interacoes`, `ajustar` e `ic_por_estrato` vem de
`terreno_sob_gatilho_1km.py`. Se as duas cidades passassem por implementacoes
diferentes, uma diferenca de resultado poderia vir do codigo em vez do
territorio, e nao haveria como saber qual.

-------------------------------------------------------------------------
O QUE CHICAGO PODE E O QUE ELA NAO PODE RESPONDER
-------------------------------------------------------------------------

PODE responder sobre o TWI: a cidade tem espalhamento real de indice de umidade
(p5 7,70, mediana 11,03, p95 16,84).

NAO PODE responder sobre o HAND. Chicago e plana ao ponto de o HAND mediano ser
0,057 m, com p95 em 6,24 m -- quase toda a cidade esta no nivel da drenagem. Um
nulo de HAND aqui NAO confirma o nulo de Nova York: seria nulo por falta de
contraste, que e outra coisa. Isso vai declarado no resultado, e nao lido como
replicacao.

E os positivos aqui sao dois tipos com significado fisico diferente -- agua na
via e agua no porao. Em Nova York a via respondeu mais que o refluxo (1,96
contra 1,40), que e a ordem que a fisica manda. O mesmo corte e medido aqui.

Uso:
    python scripts/chuva/terreno_sob_gatilho_chicago.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from chuva.terreno_sob_gatilho_1km import (  # noqa: E402
    CHUVA, MINIMO_POR_ESTRATO, SEMENTE, SINAL_EXIGIDO, TERRENO, ajustar,
    centrar, com_interacoes, estratos_utilizaveis, ic_por_estrato,
    variacao_interna,
)

PONTOS = REPO / "local_runs" / "aq-chicago-03" / "chicago_pontos_com_terreno.csv"
MRMS = REPO / "local_runs" / "chuva-mrms-chicago" / "mrms_por_ponto_chicago.csv"
NYC = REPO / "local_runs" / "nyc-terreno-sob-gatilho" / "terreno_sob_gatilho_1km.json"
OUT = REPO / "local_runs" / "chicago-teste-da-lei"

# Chicago e plana demais para que um nulo de HAND signifique alguma coisa.
HAND_SEM_CONTRASTE_P95_M = 10.0


def carregar() -> pd.DataFrame:
    if not PONTOS.exists():
        raise SystemExit(f"ausente: {PONTOS}")
    if not MRMS.exists():
        raise SystemExit(
            f"ausente: {MRMS}. Rode antes: "
            "python scripts/chuva/adquirir_mrms_chicago.py --online")
    d = pd.read_csv(PONTOS)
    chuva = pd.read_csv(MRMS).drop_duplicates(["ponto_id", "dia"])
    d = d.merge(chuva, on=["ponto_id", "dia"], how="inner")
    return d.dropna(subset=TERRENO + [CHUVA]).copy()


def medir(s: pd.DataFrame, rng, rotulo: str) -> dict:
    z, nomes = com_interacoes(s)
    beta = ajustar(centrar(z, nomes), z.classe.to_numpy())
    ic = ic_por_estrato(z, nomes, rng)
    saida = {}
    print(f"\n--- {rotulo}: {len(s):,} pontos, "
          f"{int(s.classe.sum()):,} positivos, {s.dia.nunique():,} dias")
    for f, b in zip(nomes, beta):
        lo, hi = ic[f]
        exigido = SINAL_EXIGIDO.get(f)
        bate = exigido is None or np.sign(b) == exigido
        exclui = (lo > 0) or (hi < 0)
        saida[f] = {"coef": round(float(b), 4), "ic95": [lo, hi],
                    "ic95_exclui_zero": bool(exclui),
                    "sinal_esperado": exigido,
                    "sinal_bate": bool(bate),
                    "aprendido": bool(bate and exclui)}
        print(f"  {f:20s} {b:+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  "
              f"{'APRENDIDO' if bate and exclui else 'nao sustentado'}")
    return saida


def comparar_com_nyc(chicago: dict) -> dict:
    """A lei se repete? Compara sinal e sobreposicao de intervalo."""
    if not NYC.exists():
        return {"comparavel": False, "motivo": f"ausente: {NYC}"}
    ny = json.loads(NYC.read_text(encoding="utf-8")).get("efeitos", {})
    saida = {}
    for f, c in chicago.items():
        n = ny.get(f)
        if not n:
            continue
        mesmo_sinal = np.sign(c["coef"]) == np.sign(n["coef"])
        lo = max(c["ic95"][0], n["ic95"][0])
        hi = min(c["ic95"][1], n["ic95"][1])
        saida[f] = {
            "nyc": {"coef": n["coef"], "ic95": n["ic95"],
                    "aprendido": n.get("aprendido")},
            "chicago": {"coef": c["coef"], "ic95": c["ic95"],
                        "aprendido": c["aprendido"]},
            "mesmo_sinal": bool(mesmo_sinal),
            "intervalos_se_sobrepoem": bool(lo <= hi),
            "aprendido_nas_duas": bool(c["aprendido"] and n.get("aprendido")),
        }
    return saida


def main() -> int:
    rng = np.random.default_rng(SEMENTE)
    print("=== A LEI DO TWI SE REPETE EM CHICAGO? ===")
    bruto = carregar()
    print(f"chamados com terreno e chuva: {len(bruto):,}")

    s = estratos_utilizaveis(bruto)
    print(f"dias informativos (as duas classes, >= {MINIMO_POR_ESTRATO} pontos): "
          f"{s.dia.nunique():,}")

    hand_p95 = float(s.hand_m.quantile(0.95))
    sem_contraste = hand_p95 < HAND_SEM_CONTRASTE_P95_M
    intra = variacao_interna(s, TERRENO + [CHUVA])
    print("\nfracao da variancia que sobra DENTRO do dia:")
    for f, v in intra.items():
        print(f"  {f:20s} {v}")

    efeitos = medir(s, rng, "cidade inteira")

    por_tipo = {}
    for tipo in sorted(s.loc[s.classe == 1, "sr_type"].dropna().unique()):
        parte = s[(s.classe == 0) | (s.sr_type == tipo)]
        parte = estratos_utilizaveis(parte)
        if parte.classe.nunique() < 2 or parte.dia.nunique() < 30:
            continue
        por_tipo[tipo] = medir(parte, rng, f"so {tipo}")

    comparacao = comparar_com_nyc(efeitos)
    print("\n=== A LEI SE REPETE? ===")
    for f, c in comparacao.items() if isinstance(comparacao, dict) else []:
        if not isinstance(c, dict) or "nyc" not in c:
            continue
        print(f"  {f:20s} NY {c['nyc']['coef']:+.4f} | CHI "
              f"{c['chicago']['coef']:+.4f} | mesmo sinal="
              f"{c['mesmo_sinal']} | aprendido nas duas={c['aprendido_nas_duas']}")

    OUT.mkdir(parents=True, exist_ok=True)
    resultado = {
        "tipo": "teste_da_lei_segunda_cidade",
        "gerado_por": "scripts/chuva/terreno_sob_gatilho_chicago.py",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cidade": "chicago",
        "pergunta": ("o efeito do TWI e sua interacao com a chuva, medidos em "
                     "Nova York, se repetem em outra cidade?"),
        "estimador_importado_de": "scripts/chuva/terreno_sob_gatilho_1km.py",
        "amostra": {
            "pontos": int(len(s)),
            "positivos": int(s.classe.sum()),
            "dias_informativos": int(s.dia.nunique()),
            "fracao_intra_dia": intra,
        },
        "efeitos": efeitos,
        "por_tipo_de_chamado": por_tipo,
        "comparacao_com_nyc": comparacao,
        "hand_sem_contraste": {
            "p95_do_hand_m": round(hand_p95, 3),
            "limiar_declarado_m": HAND_SEM_CONTRASTE_P95_M,
            "sem_contraste": bool(sem_contraste),
            "consequencia": (
                "nulo de HAND aqui NAO confirma o nulo de Nova York: seria nulo "
                "por falta de variacao, que e outra coisa" if sem_contraste
                else "ha contraste de HAND suficiente para a leitura valer"),
        },
        "nao_e": (
            "nao promove variavel, nao ajusta modelo servido e nao autoriza "
            "sozinho a transferencia para regiao brasileira: mede se o efeito "
            "reaparece numa segunda cidade dos Estados Unidos"),
    }
    (OUT / "teste_da_lei_chicago.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/chuva/terreno_sob_gatilho_chicago.py\n", encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
