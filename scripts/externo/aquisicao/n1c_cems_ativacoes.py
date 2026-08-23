"""
NIVEL 1 / C -- Copernicus EMS Rapid Mapping.

O que e: servico da Comissao Europeia que, a cada emergencia ativada, publica
mapeamento por satelite com produtos vetoriais. Mais de 490 ativacoes so ate
2020, mundiais, gratuitas.

POR QUE E A FONTE MAIS FORTE DO NIVEL 1 PARA A LACUNA L1:
cada ativacao publica a AREA DE INTERESSE (AOI) que foi efetivamente
analisada, junto com a extensao de inundacao observada dentro dela. Isso
produz o negativo mais defensavel que existe nesta linha de trabalho:

    dentro da AOI  E  fora do poligono de inundacao  =  observado e nao inundado

Nao e inferencia, nao e ausencia de registro, e o recorte que o analista
declarou ter examinado. E exatamente a estrutura que o Protocolo C tenta
reconstruir manualmente no Brasil, e que aqui vem pronta.

Este script faz a etapa 1: inventariar TODAS as ativacoes, identificar as de
inundacao, e registrar quais intersectam a AOI do piloto (noroeste da
Inglaterra, EXT-UK-03) e quais cobrem o Brasil. O download dos pacotes
vetoriais por ativacao e etapa seguinte e separada, porque depende de decidir
o escopo -- baixar os vetores de 490 ativacoes sem criterio violaria a regra
de usar o minimo de dado por etapa.

Fonte  : Copernicus Emergency Management Service (CEMS) Rapid Mapping
API    : rapidmapping.emergency.copernicus.eu/backend/dashboard-api/
Licenca: dados Copernicus, uso livre com atribuicao

NAO faz: nenhum download de vetor ainda, nenhuma promocao a negativo.

Uso:
    python scripts/externo/n1c_cems_ativacoes.py
"""

from __future__ import annotations

import re
from collections import Counter

from n1_comum import BASE_OUT, escrever_proveniencia, get_json, salvar_json

OUTDIR = BASE_OUT / "n1c-cems"

API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations-info/"

# AOI do piloto (EXT-UK-03), em WGS84 aproximado, e o Brasil
AOI_UK = (-3.05, 52.80, -1.70, 54.20)   # lon_min, lat_min, lon_max, lat_max
BBOX_UK_PAIS = (-8.2, 49.8, 2.1, 61.0)
BBOX_BR = (-74.0, -34.0, -34.0, 5.3)

PALAVRAS_FLOOD = ("flood", "inondation", "alluvione", "inundac", "storm", "cyclone", "typhoon")

# --------------------------------------------------------------------------
# CORRECAO 2026-08-07: a primeira versao deste script falhou em silencio.
# Dois erros, ambos por eu ter suposto o formato em vez de olhar:
#   1. A API e PAGINADA (`count`/`next`/`previous`/`results`, 10 por pagina).
#      A versao anterior leu so a primeira pagina -- 10 de 246 ativacoes.
#   2. Os campos reais sao `code`, `category`, `countries` (LISTA) e
#      `centroid` (WKT "POINT (lon lat)"). Nao existem `lat`/`lon` soltos,
#      que era o que a versao anterior procurava -- por isso os tres arquivos
#      de saida sairam vazios sem nenhum erro visivel.
# --------------------------------------------------------------------------


def parse_centroid(wkt) -> tuple[float, float] | tuple[None, None]:
    """Extrai (lon, lat) de 'POINT (lon lat)'."""
    if not wkt or not isinstance(wkt, str):
        return None, None
    m = re.search(r"POINT\s*\(\s*([-\d.eE]+)\s+([-\d.eE]+)\s*\)", wkt)
    if not m:
        return None, None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None, None


def dentro(lon, lat, bbox) -> bool:
    if lon is None or lat is None:
        return False
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def campo(d: dict, *nomes):
    for n in nomes:
        if n in d and d[n] not in (None, ""):
            return d[n]
    return None


def buscar_tudo() -> list[dict]:
    """Percorre todas as paginas seguindo `next`."""
    registros: list[dict] = []
    url: str | None = API
    params: dict | None = {"limit": 200}
    paginas = 0
    while url and paginas < 100:
        payload = get_json(url, params)
        params = None
        if not payload:
            break
        registros += payload.get("results", []) or []
        paginas += 1
        url = payload.get("next")
    print(f"PAGINAS_LIDAS={paginas}")
    return registros


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("===N1C_CEMS===")

    registros = buscar_tudo()
    if not registros:
        print("API=INACESSIVEL")
        print("===END===")
        return 1

    print(f"ATIVACOES={len(registros)}")
    salvar_json(OUTDIR / "activations_raw.json", registros)
    print("CAMPOS=" + ", ".join(list(registros[0].keys())))

    codigos = [str(campo(r, "code") or "") for r in registros]
    nums = sorted(int(c[4:]) for c in codigos if c.startswith("EMSR") and c[4:].isdigit())
    if nums:
        print(f"FAIXA_EMSR=EMSR{nums[0]} a EMSR{nums[-1]} (n={len(nums)})")

    tipos: Counter = Counter()
    paises: Counter = Counter()
    floods, no_uk_aoi, no_uk_pais, no_br = [], [], [], []
    for r in registros:
        if not isinstance(r, dict):
            continue
        cat = str(campo(r, "category") or "?")
        tipos[cat] += 1
        for c in (r.get("countries") or []):
            paises[str(c)] += 1
        nome = str(campo(r, "name") or "").lower()
        eh_flood = any(p in cat.lower() or p in nome for p in PALAVRAS_FLOOD)
        lon, lat = parse_centroid(r.get("centroid"))
        r["_lon"], r["_lat"] = lon, lat
        if eh_flood:
            floods.append(r)
            if dentro(lon, lat, AOI_UK):
                no_uk_aoi.append(r)
            if dentro(lon, lat, BBOX_UK_PAIS):
                no_uk_pais.append(r)
            if dentro(lon, lat, BBOX_BR):
                no_br.append(r)

    print("CATEGORIAS=" + "; ".join(f"{k}:{v}" for k, v in tipos.most_common(8)))
    print("PAISES_TOP=" + "; ".join(f"{k}:{v}" for k, v in paises.most_common(8)))
    print(f"FLOOD_LIKE={len(floods)}")
    print(f"NA_AOI_UK={len(no_uk_aoi)} NO_REINO_UNIDO={len(no_uk_pais)} NO_BRASIL={len(no_br)}")
    print(f"AOIS_TOTAIS_EM_FLOOD={sum(int(r.get('n_aois') or 0) for r in floods)}")

    for nome, lista in (("uk_aoi", no_uk_aoi), ("uk_pais", no_uk_pais),
                        ("brasil", no_br), ("flood", floods)):
        salvar_json(OUTDIR / f"activations_{nome}.json", lista)

    for rot, lista in (("UK", no_uk_pais), ("BR", no_br)):
        for r in lista[:12]:
            print(f"  {rot} {campo(r, 'code')} | {str(campo(r, 'name'))[:60]} "
                  f"| {str(campo(r, 'eventTime'))[:10]} | aois={r.get('n_aois')}")

    escrever_proveniencia(
        OUTDIR,
        titulo="Copernicus EMS Rapid Mapping - inventario de ativacoes",
        fonte="Copernicus Emergency Management Service (CEMS)",
        identificador=str(usado),
        licenca="Dados Copernicus - uso livre com atribuicao a European Union",
        natureza="OBSERVADO. Cada ativacao declara a Area de Interesse "
                 "efetivamente analisada e a extensao de inundacao detectada "
                 "dentro dela.",
        uso_pretendido="Base do NEGATIVO POR OBSERVACAO: area dentro da AOI e "
                       "fora do poligono de inundacao foi analisada e nao "
                       "estava inundada.",
        uso_proibido="Tratar AOI como area inundada; usar sem baixar o produto "
                     "vetorial correspondente; promover a negativo antes da "
                     "tarefa de adjudicacao.",
        extra="- Esta etapa inventaria ativacoes apenas. O download dos "
              "pacotes vetoriais por ativacao e etapa separada, com escopo "
              "decidido a partir deste inventario.\n",
    )
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
