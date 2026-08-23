"""
NIVEL 1 / B -- Urban Flood Observations (UFO).

O que e: 215 chips 1024x1024 a 3 m (PlanetScope, 4 bandas), de 14 eventos de
inundacao entre 2017 e 2021, cobrindo drivers pluvial, fluvial e mare de
tempestade, em contextos URBANOS diversos. Rotulo manual em duas classes:
`inundated` e `non-inundated`.

POR QUE ESTA BASE IMPORTA MAIS QUE O TAMANHO DELA SUGERE:
e a unica fonte do Nivel 1 desenhada especificamente para inundacao URBANA,
que e o fenomeno do REV-P. As demais (Sen1Floods11, GFD) sao dominadas por
cheia fluvial em area aberta. Alem disso, a resolucao de 3 m e a unica
compativel com a escala de quarteirao -- a 250 m do GFD, um bairro inteiro
vira um pixel.

Resultado de referencia publicado: modelo treinado no UFO com validacao
leave-one-event-out atinge mIoU 77,3%. Os produtos globais de agua avaliados
contra ele ficaram bem abaixo (NASA IMPACT/Sentinel-1: 44,1%; Dynamic World:
48,1%). Isso e um alerta metodologico direto para o REV-P: produto global de
agua superficial NAO substitui rotulo em contexto urbano.

Fonte  : Zenodo, DOI 10.5281/zenodo.15238469
Codigo : https://github.com/Tellman-lab/urbanFloodObservations
Licenca: CC BY 4.0

NAO faz: nenhuma promocao a negativo, nenhuma extracao de variavel, nenhum
treino.

Uso:
    python scripts/externo/n1b_ufo_zenodo.py
"""

from __future__ import annotations

from n1_comum import (
    BASE_OUT, baixar, escrever_proveniencia, gb, get_json, salvar_json,
)

# DOI principal e alternativo relatados na literatura -- tenta os dois
RECORD_IDS = ["15238469", "19698577"]
OUTDIR = BASE_OUT / "n1b-ufo"


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("===N1B_UFO===")

    registro = None
    usado = None
    for rid in RECORD_IDS:
        payload = get_json(f"https://zenodo.org/api/records/{rid}")
        if payload and payload.get("files"):
            registro, usado = payload, rid
            break
        print(f"  record {rid} sem arquivos ou inacessivel")

    if not registro:
        print("REGISTRO=NAO_ENCONTRADO")
        print("===END===")
        return 1

    titulo = registro.get("metadata", {}).get("title", "?")
    licenca = (
        registro.get("metadata", {}).get("license", {}).get("id")
        or registro.get("metadata", {}).get("license", {}).get("name")
        or "ver metadata"
    )
    arquivos = registro.get("files", [])
    total = sum(f.get("size", 0) for f in arquivos)

    print(f"RECORD={usado}")
    print(f"TITULO={titulo[:90]}")
    print(f"LICENCA={licenca}")
    print(f"ARQUIVOS n={len(arquivos)} total={gb(total)}GB")
    for f in arquivos[:15]:
        print(f"  {f.get('key')} {gb(f.get('size', 0))}GB")

    salvar_json(OUTDIR / "zenodo_record.json", registro)

    ok, falhou = 0, 0
    for f in arquivos:
        url = (f.get("links") or {}).get("self")
        nome = f.get("key")
        if not url or not nome:
            falhou += 1
            continue
        if baixar(url, OUTDIR / nome, f.get("size")):
            ok += 1
        else:
            falhou += 1
            print(f"  FALHOU {nome}")

    presentes = [f for f in arquivos if (OUTDIR / str(f.get("key"))).exists()]
    disco = sum((OUTDIR / str(f["key"])).stat().st_size for f in presentes)

    escrever_proveniencia(
        OUTDIR,
        titulo="Urban Flood Observations (UFO)",
        fonte="Zenodo / Tellman Lab",
        identificador=f"DOI 10.5281/zenodo.{usado}",
        licenca=str(licenca),
        natureza="OBSERVADO. Rotulo manual em duas classes, `inundated` e "
                 "`non-inundated`, sobre imagem PlanetScope de 3 m. A classe "
                 "non-inundated e afirmacao de observacao.",
        uso_pretendido="Fonte candidata de NEGATIVO POR OBSERVACAO em contexto "
                       "URBANO (analogo mais proximo de Recife); referencia "
                       "para avaliar se produto global de agua e suficiente.",
        uso_proibido="Promover a negativo do dataset SUSC sem adjudicacao "
                     "propria; usar como variavel; extrapolar rotulo de 3 m "
                     "para outra resolucao sem documentar a reamostragem.",
        extra="- Codigo de referencia: https://github.com/Tellman-lab/urbanFloodObservations\n"
              "- Alerta metodologico registrado: produtos globais de agua "
              "(Dynamic World, NASA IMPACT) tiveram IoU 44-48% contra este "
              "rotulo urbano, contra 77,3% de modelo treinado nele.\n",
    )

    print(f"BAIXADOS={len(presentes)}/{len(arquivos)} disco={gb(disco)}GB FALHAS={falhou}")
    print("===END===")
    return 0 if len(presentes) == len(arquivos) else 2


if __name__ == "__main__":
    raise SystemExit(main())
