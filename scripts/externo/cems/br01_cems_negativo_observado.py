"""
BR-01 -- Negativo por observacao no Brasil, a partir do Copernicus EMS EMSR720.

*** ESTA E A ETAPA MAIS IMPORTANTE DA FRENTE EXTERNA ATE AQUI. ***

O REV-P nunca teve negativo formal. O gate `C4_BLOCKED_NO_FORMAL_NEGATIVES`
existe porque todo negativo do Protocolo C foi construido por AUSENCIA de
registro -- "nao ha noticia de enchente aqui, logo aqui nao inundou". A
propria Environment Agency, no caso ingles, documenta por que isso e fragil:
ausencia de registro nao e evidencia de ausencia de evento.

O EMSR720 (Flood in Rio Grande Do Sul State, Brazil, 2024-05-02) quebra esse
impasse, e no Brasil, que e onde o problema realmente esta. A estrutura do
produto Copernicus EMS da, no proprio dado, a afirmacao que faltava:

    areaOfInterestA   poligono que o analista DECLAROU ter examinado
    imageFootprintA   o que a imagem de satelite efetivamente cobriu
    observedEventA    a inundacao DETECTADA dentro daquela area

Disso decorre, sem inferencia:

    (AOI ∩ footprint) \\ observedEvent  =  area OBSERVADA e NAO INUNDADA

Nao e ausencia de registro. E registro de ausencia. Essa distincao e a
diferenca entre um negativo defensavel e um negativo que a banca derruba.

O CRUZAMENTO COM O FOOTPRINT NAO E PRECIOSISMO: a AOI e o recorte pedido, mas
o que garante observacao e a imagem. Area dentro da AOI e fora do footprint
nao foi vista, e nao pode virar negativo. Por isso a intersecao.

ENTRADA: EMSR720_products.zip (15 pacotes unicos, 5 AOIs, produtos DEL/GRA).
Baixado manualmente do portal em 2026-08-07 -- a API publica do CEMS nao expoe
link direto para os pacotes vetoriais, apenas metadados de ativacao.

NAO faz: nao amostra ponto, nao extrai feature, nao promove negativo ao
dataset SUSC, nao treina. Reduz o produto a geometrias de decisao e mede.

Uso:
    python scripts/externo/br01_cems_negativo_observado.py --zip <caminho>
"""

from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "local_runs" / "br-01-cems-negativo"
CRS_METRICO = 31982   # SIRGAS 2000 / UTM 22S -- cobre o Rio Grande do Sul

CAMADAS = ["areaOfInterestA", "observedEventA", "imageFootprintA",
           "floodDepthA", "maximumFloodExtentA"]


def extrair(zip_externo: Path, destino: Path) -> dict:
    """Desaninha os pacotes e agrupa os shapefiles por tipo de camada."""
    destino.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(zip_externo)
    pacotes = sorted(set(z.namelist()))
    por_camada: dict[str, list] = defaultdict(list)

    for pacote in pacotes:
        interno = zipfile.ZipFile(io.BytesIO(z.read(pacote)))
        pasta = destino / Path(pacote).stem
        pasta.mkdir(parents=True, exist_ok=True)
        interno.extractall(pasta)
        for shp in pasta.rglob("*.shp"):
            for chave in CAMADAS:
                if chave in shp.name:
                    por_camada[chave].append(shp)
                    break
    return {"pacotes": len(pacotes), "por_camada": por_camada}


def juntar(caminhos: list[Path]) -> gpd.GeoDataFrame | None:
    partes = []
    for p in caminhos:
        try:
            g = gpd.read_file(p)
            if len(g) == 0:
                continue
            g["_arquivo"] = p.name
            partes.append(g.to_crs(CRS_METRICO))
        except Exception as exc:  # noqa: BLE001
            print(f"   AVISO falha lendo {p.name}: {type(exc).__name__}")
    if not partes:
        return None
    return gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=CRS_METRICO)


def main() -> int:
    args = sys.argv[1:]
    caminho_zip = Path(args[args.index("--zip") + 1]) if "--zip" in args else None
    if not caminho_zip or not caminho_zip.exists():
        print("uso: --zip <EMSR720_products.zip>")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===BR-01_CEMS_NEGATIVO===")

    info = extrair(caminho_zip, OUT / "extraido")
    print(f"PACOTES={info['pacotes']}")
    for k, v in info["por_camada"].items():
        print(f"  camada {k}: {len(v)} shapefiles")

    camadas = {}
    for chave, caminhos in info["por_camada"].items():
        g = juntar(caminhos)
        if g is None:
            continue
        camadas[chave] = g
        area = g.geometry.area.sum() / 1e6
        print(f"{chave}: feicoes={len(g)} area_total={area:.1f}km2 crs={g.crs.to_epsg()}")

    aoi = camadas.get("areaOfInterestA")
    obs = camadas.get("observedEventA")
    fp = camadas.get("imageFootprintA")
    if aoi is None or obs is None:
        print("ABORTADO: faltam areaOfInterestA ou observedEventA")
        return 2

    # dissolucao -- as AOIs se sobrepoem entre produtos DEL e GRA
    u_aoi = aoi.geometry.union_all()
    u_obs = obs.geometry.union_all()
    u_fp = fp.geometry.union_all() if fp is not None else None

    a_aoi = u_aoi.area / 1e6
    a_obs = u_obs.area / 1e6
    print("---GEOMETRIAS DISSOLVIDAS---")
    print(f"AOI_TOTAL={a_aoi:.1f}km2")
    print(f"OBSERVADO_INUNDADO={a_obs:.1f}km2 ({100*a_obs/a_aoi:.1f}% da AOI)")

    # area efetivamente observada = AOI ∩ footprint
    observada = u_aoi.intersection(u_fp) if u_fp is not None else u_aoi
    a_observada = observada.area / 1e6
    if u_fp is not None:
        print(f"FOOTPRINT_TOTAL={u_fp.area/1e6:.1f}km2")
        print(f"AOI_EFETIVAMENTE_OBSERVADA={a_observada:.1f}km2 "
              f"({100*a_observada/a_aoi:.1f}% da AOI)")
        nao_vista = a_aoi - a_observada
        print(f"AOI_NAO_COBERTA_POR_IMAGEM={nao_vista:.1f}km2 -- NAO pode virar negativo")

    # ----------------------------------------------------------------------
    # SEPARACAO DE MECANISMO -- descoberta de 2026-08-07.
    # O `observedEventA` do EMSR720 traz `event_type` com DOIS valores:
    #   5-Flood         / obj_desc 'Riverine flood'
    #   6-Mass Movement / obj_desc 'Landslide'
    # Ou seja, o produto Copernicus separa enchente de deslizamento no proprio
    # poligono, no Brasil. E exatamente a informacao cuja ausencia travou
    # Petropolis ("mistura enchente/deslizamento nao separada").
    #
    # Isso obriga a refinar o negativo. Area de deslizamento NAO e inundacao,
    # mas tambem NAO e "nao inundada" num sentido util: e area de outro perigo,
    # com terreno alterado. Trata-la como negativo introduziria confusao entre
    # dois processos distintos. Por isso ela e removida das duas classes.
    # ----------------------------------------------------------------------
    tipo = obs.get("event_type")
    if tipo is not None:
        flood = obs[obs["event_type"].astype(str).str.contains("Flood", na=False)]
        massa = obs[obs["event_type"].astype(str).str.contains("Mass", na=False)]
        print("---SEPARACAO_DE_MECANISMO---")
        tab = obs.groupby(["event_type", "obj_desc"]).agg(
            n=("geometry", "size"),
            km2=("geometry", lambda s: round(s.area.sum() / 1e6, 2)))
        for (et, od), r in tab.iterrows():
            print(f"  {et} / {od}: {int(r.n)} feicoes, {r.km2}km2")
        u_flood = flood.geometry.union_all() if len(flood) else None
        u_massa = massa.geometry.union_all() if len(massa) else None
        if u_flood is not None:
            print(f"INUNDACAO_DISSOLVIDA={u_flood.area/1e6:.2f}km2")
        if u_massa is not None:
            print(f"DESLIZAMENTO_DISSOLVIDO={u_massa.area/1e6:.2f}km2 "
                  f"-- removido de ambas as classes")
    else:
        u_flood, u_massa = u_obs, None

    positivo = u_flood if u_flood is not None else u_obs
    negativo = observada.difference(positivo)
    if u_massa is not None:
        negativo = negativo.difference(u_massa)
    a_neg = negativo.area / 1e6
    a_pos = positivo.area / 1e6
    print("---RESULTADO---")
    print(f"POSITIVO_INUNDACAO={a_pos:.2f}km2")
    print(f"NEGATIVO_OBSERVADO_ESTRITO={a_neg:.2f}km2 "
          f"({100*a_neg/a_observada:.1f}% do observado)")
    print(f"RAZAO_NEG_POS={a_neg/a_pos:.2f}")

    for nome, geom in (("aoi", u_aoi), ("observado_inundado", u_obs),
                       ("positivo_inundacao", positivo),
                       ("area_observada", observada), ("negativo_observado", negativo)):
        gpd.GeoDataFrame(geometry=[geom], crs=CRS_METRICO).to_parquet(
            OUT / f"{nome}.parquet", index=False)

    # atributos uteis do observedEvent (data, fonte, tipo)
    cols = [c for c in obs.columns if c.lower() in
            ("event_type", "obj_desc", "source_nam", "src_date", "notation", "obj_type")]
    if cols:
        print("---ATRIBUTOS observedEventA---")
        for c in cols:
            vals = obs[c].dropna().astype(str).unique()[:5]
            print(f"  {c}: {list(vals)}")

    (OUT / "resumo.json").write_text(json.dumps({
        "ativacao": "EMSR720",
        "evento": "Flood in Rio Grande do Sul State, Brazil",
        "data_evento": "2024-05-02",
        "crs_metrico": CRS_METRICO,
        "pacotes": info["pacotes"],
        "area_aoi_km2": round(a_aoi, 1),
        "area_observada_km2": round(a_observada, 1),
        "area_observedEvent_total_km2": round(a_obs, 1),
        "area_positivo_inundacao_km2": round(a_pos, 2),
        "area_negativo_observado_km2": round(a_neg, 2),
        "razao_neg_pos": round(a_neg / a_pos, 2),
        "definicao_negativo": (
            "(areaOfInterest INTERSECAO imageFootprint) MENOS inundacao MENOS "
            "deslizamento. A intersecao com o footprint e obrigatoria: area da "
            "AOI que a imagem nao cobriu nao foi observada e nao pode virar "
            "negativo. O deslizamento sai das DUAS classes: nao e inundacao, "
            "mas tambem nao e negativo util -- e outro perigo, com terreno "
            "alterado."
        ),
        "separacao_de_mecanismo": (
            "O `event_type` do observedEventA distingue '5-Flood/Riverine flood' "
            "de '6-Mass Movement/Landslide' na origem. E a informacao cuja "
            "ausencia mantem Petropolis bloqueado."
        ),
        "status": ("CANDIDATO. Nenhum ponto amostrado, nenhum negativo promovido "
                   "ao dataset SUSC, gate C4 inalterado."),
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"TEMPO={round(time.time()-t0,1)}s")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
