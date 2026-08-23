"""
NIVEL 1 / A -- Sen1Floods11.

O que e: 4.831 chips 512x512 a 10 m, cobrindo 120.406 km2, 11 eventos de
inundacao, 6 continentes, todos os 14 biomas. Destes, 446 sao ROTULADOS A MAO
(252 treino / 89 validacao / 90 teste); o restante foi rotulado por algoritmo
e serve como supervisao fraca.

POR QUE ESTA BASE RESOLVE A LACUNA L1:
o rotulo manual tem tres estados por pixel -- agua, nao-agua, e sem-dado --
e nao dois. Isso significa que "nao-agua" aqui e uma AFIRMACAO de que o pixel
foi observado e estava seco, e nao a ausencia de uma afirmacao. E exatamente
a diferenca entre o negativo do Protocolo C (ausencia de registro) e negativo
por observacao.

Estrutura relevante do bucket:
  v1.1/data/flood_events/HandLabeled/    <- rotulo manual (o que interessa)
      LabelHand/        rotulo 3 estados
      S1Hand/           Sentinel-1 VV/VH correspondente
      S2Hand/           Sentinel-2 correspondente
      JRCWaterHand/     agua permanente JRC (permite isolar agua de cheia)
  v1.1/data/flood_events/WeaklyLabeled/  <- supervisao fraca (opcional)
  v1.1/splits/                           <- particoes oficiais treino/val/teste

Fonte  : Cloud to Street (bucket publico gs://sen1floods11)
Licenca: CC BY 4.0
Ref    : Bonafilia et al., CVPRW 2020

NAO faz: nenhuma promocao a negativo, nenhuma extracao de variavel, nenhum
treino. So adquire e inventaria.

Uso:
    python scripts/externo/n1a_sen1floods11.py            # so rotulo manual
    python scripts/externo/n1a_sen1floods11.py --tudo     # inclui weak labels
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict

from n1_comum import (
    BASE_OUT, baixar, escrever_proveniencia, gb, listar_bucket, salvar_json,
)

BUCKET = "sen1floods11"
OUTDIR = BASE_OUT / "n1a-sen1floods11"

PREFIXOS_ESSENCIAIS = [
    "v1.1/data/flood_events/HandLabeled/",
    "v1.1/splits/",
    "v1.1/catalog/",
]
PREFIXO_OPCIONAL = "v1.1/data/flood_events/WeaklyLabeled/"


def main() -> int:
    tudo = "--tudo" in sys.argv
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("===N1A_SEN1FLOODS11===")

    itens: list[dict] = []
    for pref in PREFIXOS_ESSENCIAIS:
        lote = listar_bucket(BUCKET, pref)
        print(f"PREFIXO {pref} n={len(lote)} {gb(sum(i['size'] for i in lote))}GB")
        itens += lote
    if tudo:
        lote = listar_bucket(BUCKET, PREFIXO_OPCIONAL)
        print(f"PREFIXO {PREFIXO_OPCIONAL} n={len(lote)} {gb(sum(i['size'] for i in lote))}GB")
        itens += lote

    if not itens:
        print("INVENTARIO=VAZIO (bucket inacessivel?)")
        print("===END===")
        return 1

    total = sum(i["size"] for i in itens)
    print(f"INVENTARIO n={len(itens)} total={gb(total)}GB")

    # composicao por subpasta, para o inventario ficar auditavel
    por_pasta: Counter = Counter()
    for i in itens:
        partes = i["name"].split("/")
        chave = "/".join(partes[:5])
        por_pasta[chave] += 1
    for k, v in por_pasta.most_common(12):
        print(f"  {k} n={v}")

    salvar_json(OUTDIR / "inventario.json", {"n": len(itens), "bytes": total, "itens": itens})

    # --- download ---
    ok, falhou = 0, 0
    for n, i in enumerate(itens, 1):
        destino = OUTDIR / i["name"]
        if baixar(i["url"], destino, i["size"]):
            ok += 1
        else:
            falhou += 1
        if n % 200 == 0:
            print(f"  progresso {n}/{len(itens)} ok={ok} falhou={falhou}", file=sys.stderr)

    # --- verificacao de integridade ---
    presentes = [i for i in itens if (OUTDIR / i["name"]).exists()]
    bytes_disco = sum((OUTDIR / i["name"]).stat().st_size for i in presentes)
    cobertura = len(presentes) / len(itens) if itens else 0

    # quantos chips de rotulo manual chegaram (o ativo central)
    labelhand = [i for i in presentes if "/LabelHand/" in i["name"]]

    escrever_proveniencia(
        OUTDIR,
        titulo="Sen1Floods11",
        fonte="Cloud to Street (bucket publico Google Cloud Storage)",
        identificador="gs://sen1floods11 (v1.1)",
        licenca="CC BY 4.0",
        natureza="OBSERVADO. Rotulo manual por pixel com tres estados "
                 "(agua / nao-agua / sem-dado). A classe nao-agua e afirmacao "
                 "de observacao, nao ausencia de registro.",
        uso_pretendido="Fonte candidata de NEGATIVO POR OBSERVACAO para a "
                       "lacuna L1; referencia de validacao de deteccao.",
        uso_proibido="Promover a negativo do dataset SUSC sem tarefa propria "
                     "de adjudicacao; usar como variavel; misturar chips de "
                     "rotulo fraco com rotulo manual sem marcar a procedencia.",
        extra=f"- Chips com rotulo manual baixados: {len(labelhand)}\n"
              f"- Rotulo fraco incluido: {tudo}\n",
    )

    print(f"BAIXADOS={len(presentes)}/{len(itens)} cobertura={cobertura:.3f} disco={gb(bytes_disco)}GB")
    print(f"LABELHAND_CHIPS={len(labelhand)}")
    print(f"FALHAS={falhou}")
    print("===END===")
    return 0 if cobertura >= 0.99 else 2


if __name__ == "__main__":
    raise SystemExit(main())
