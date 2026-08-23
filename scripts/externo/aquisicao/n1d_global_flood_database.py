"""
NIVEL 1 / D -- Global Flood Database (GFD) v1.

O que e: 913 eventos de inundacao mapeados por MODIS entre 2000 e 2018, em
169 paises, a partir de 12.719 cenas Terra/Aqua. Distribuicao por continente:
Asia 398, Americas 223, Africa 143, Europa 92, Oceania 57.

POR QUE ESTA BASE RESOLVE A LACUNA L1 EM ESCALA GLOBAL:
alem da banda `flooded` (extensao maxima da cheia), cada evento traz
`clear_views` -- numero de dias em que o pixel foi observado sem nuvem entre
o inicio e o fim do evento -- e `clear_perc`. Com isso:

    clear_views > 0  E  flooded = 0   =>   pixel OBSERVADO e NAO inundado

Ou seja, o negativo por observacao que o Protocolo C tenta construir na mao,
aqui existe em 913 eventos, com data, de forma programatica. A banda
`jrc_perm_water` ainda permite separar agua permanente de agua de cheia, o
que evita o erro classico de contar rio como inundacao.

LIMITACAO QUE PRECISA FICAR REGISTRADA: a classificacao MODIS e de 250 m.
Para as features do v12, que sao de 30 m ou melhores, isso e grosseiro.
O uso adequado e como fonte de NEGATIVO em escala e como evidencia de
observacao, nao como fonte de positivo fino em area urbana. O UFO (Nivel 1/B)
existe justamente para cobrir essa lacuna de escala.

LICENCA: CC BY-NC 4.0 (nao comercial). Serve para TCC e pesquisa; NAO serve
para o produto/API comercial do REV-P. Isso precisa estar claro antes de
qualquer uso, e por isso esta escrito tambem na proveniencia.

Fonte  : Cloud to Street / Dartmouth Flood Observatory
Bucket : gs://gfd_v1_4 (publico)
Ref    : Tellman et al., Nature 2021, doi:10.1038/s41586-021-03695-w

Uso:
    python scripts/externo/n1d_global_flood_database.py            # inventario
    python scripts/externo/n1d_global_flood_database.py --baixar   # baixa tudo
"""

from __future__ import annotations

import sys
from collections import Counter

from n1_comum import (
    BASE_OUT, baixar, escrever_proveniencia, gb, listar_bucket, salvar_json,
)

BUCKET = "gfd_v1_4"
OUTDIR = BASE_OUT / "n1d-gfd"


def main() -> int:
    fazer_download = "--baixar" in sys.argv
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("===N1D_GFD===")

    itens = listar_bucket(BUCKET)
    if not itens:
        print("INVENTARIO=VAZIO (bucket inacessivel ou nao publico)")
        print("ACAO=alternativa e Earth Engine (GLOBAL_FLOOD_DB/MODIS_EVENTS/V1)")
        print("===END===")
        return 1

    total = sum(i["size"] for i in itens)
    print(f"INVENTARIO n={len(itens)} total={gb(total)}GB")

    ext: Counter = Counter()
    pastas: Counter = Counter()
    for i in itens:
        nome = i["name"]
        ext["." + nome.rsplit(".", 1)[-1] if "." in nome else "sem_ext"] += 1
        pastas[nome.split("/")[0] if "/" in nome else "(raiz)"] += 1
    print("EXTENSOES=" + "; ".join(f"{k}:{v}" for k, v in ext.most_common(8)))
    print("PASTAS=" + "; ".join(f"{k}:{v}" for k, v in pastas.most_common(8)))
    print("EXEMPLOS=" + "; ".join(i["name"] for i in itens[:3]))

    salvar_json(OUTDIR / "inventario.json", {"n": len(itens), "bytes": total, "itens": itens})

    escrever_proveniencia(
        OUTDIR,
        titulo="Global Flood Database v1 (2000-2018)",
        fonte="Cloud to Street / Dartmouth Flood Observatory",
        identificador="gs://gfd_v1_4 ; doi:10.1038/s41586-021-03695-w",
        licenca="CC BY-NC 4.0 -- NAO COMERCIAL. Uso academico/TCC permitido; "
                "uso no produto ou API comercial do REV-P NAO permitido.",
        natureza="OBSERVADO, 250 m. Banda `flooded` = extensao maxima; "
                 "`clear_views`/`clear_perc` = dias de observacao sem nuvem; "
                 "`jrc_perm_water` = agua permanente.",
        uso_pretendido="NEGATIVO POR OBSERVACAO em escala global "
                       "(clear_views>0 e flooded=0); evidencia de que a area "
                       "foi observada na janela do evento.",
        uso_proibido="Usar como positivo fino em area urbana (250 m e grosseiro "
                     "demais); confundir agua permanente com agua de cheia; "
                     "qualquer uso comercial (licenca NC).",
        extra=f"- Objetos inventariados: {len(itens)} ({gb(total)}GB)\n"
              f"- Download completo executado: {fazer_download}\n",
    )

    if not fazer_download:
        print("MODO=INVENTARIO (rode com --baixar para trazer os arquivos)")
        print("===END===")
        return 0

    ok, falhou = 0, 0
    for n, i in enumerate(itens, 1):
        if baixar(i["url"], OUTDIR / i["name"], i["size"]):
            ok += 1
        else:
            falhou += 1
        if n % 100 == 0:
            print(f"  progresso {n}/{len(itens)} ok={ok}", file=sys.stderr)

    presentes = [i for i in itens if (OUTDIR / i["name"]).exists()]
    disco = sum((OUTDIR / i["name"]).stat().st_size for i in presentes)
    print(f"BAIXADOS={len(presentes)}/{len(itens)} disco={gb(disco)}GB FALHAS={falhou}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
