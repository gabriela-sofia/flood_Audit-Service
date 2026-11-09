"""Pacote areal de uma ativacao Copernicus EMS: positivo e negativo OBSERVADO.

POR QUE ESTE SCRIPT EXISTE:

`local_runs/br-01-cems-negativo/` continha o ativo mais forte do projeto -- a
unica evidencia areal brasileira, com negativo observado -- e NENHUM script do
repositorio o produzia. Duas consequencias, as duas medidas:

  1. o pacote nao era reproduzivel;
  2. o coletor de `coletor_de_conhecimento.py` so verifica um artefato cujo
     script gerador esta no repositorio, entao o pacote nao estava entre os 57
     registros da memoria cientifica. O melhor dado do projeto existia em disco
     e nao existia para a IA.

Este modulo fecha as duas, e generaliza: a mesma rotina roda para qualquer
ativacao, o que abre as 86 ativacoes de cheia do catalogo.

-------------------------------------------------------------------------
A DEFINICAO DE NEGATIVO, que e o ponto todo
-------------------------------------------------------------------------

    area_observada  = uniao(areaOfInterestA) INTERSECAO uniao(imageFootprintA)

A intersecao com o footprint e OBRIGATORIA. Area da AOI que a imagem nao cobriu
nao foi observada, e nao pode virar negativo -- seria transformar ausencia de
observacao em ausencia de agua, que e exatamente o erro que o projeto combate
nos chamados das centrais.

    positivo    = observedEventA com event_type 5-Flood, dentro da area observada
    deslizamento= observedEventA com event_type 6-Mass Movement
    negativo    = area_observada MENOS inundacao MENOS deslizamento

O deslizamento sai das DUAS classes: nao e inundacao, e tambem nao e negativo
util, porque o terreno ali foi alterado pelo proprio evento. E a separacao vem
da origem, no campo `event_type` do produto -- nao de heuristica nossa. E a
informacao cuja ausencia mantem Petropolis bloqueado.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao amostra ponto, nao promove negativo a dataset, nao toca em gate e nao
escreve em diretorio versionado. Produz o pacote areal e o declara.

Uso:
    python scripts/geo/cems01_pacote_areal.py --ativacao EMSR720 \
        --extraido local_runs/br-01-cems-negativo/extraido --crs 31982
    python scripts/geo/cems01_pacote_areal.py --ativacao EMSR734
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import fiona
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import CRS, Transformer
from shapely import make_valid, union_all, wkb
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
PACOTES = RUNS / "cems-05-pacotes"
CATALOGO = RUNS / "cems-04-fila" / "ativacoes_meta.json"
OUT = RUNS / "cems-areal"

# Camadas do produto EMS que este pacote consome. O resto do produto
# (transporte, edificado, instalacoes) descreve exposicao, nao observacao.
CAMADA_AOI = "areaOfInterestA"
CAMADA_FOOTPRINT = "imageFootprintA"
CAMADA_EVENTO = "observedEventA"

# Prefixos do `event_type` na origem. Nao sao heuristica: sao o vocabulario do
# proprio produto EMS.
TIPO_INUNDACAO = "5-Flood"
TIPO_DESLIZAMENTO = "6-Mass Movement"

REFERENCIA = {
    "pacote": "local_runs/br-01-cems-negativo",
    "por_que": ("pacote gerado antes deste script existir; serve de prova de "
                "reproducao para EMSR720 e nao e insumo de nada"),
}


def crs_metrico_de(lon: float, lat: float) -> int:
    """UTM WGS84 da zona do centroide. Regra fixa, para nao escolher a dedo."""
    zona = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zona


def _sanear(g):
    """Geometria invalida vira valida antes de qualquer operacao de conjunto."""
    return g if g.is_valid else make_valid(g)


def ler_camada(extraido: Path, sufixo: str, destino: int) -> tuple[list, list]:
    """Le todas as ocorrencias de uma camada, reprojetadas para o CRS metrico.

    Devolve (feicoes, ilegiveis). Cada produto traz sua propria copia da camada,
    e a uniao entre produtos e feita por quem chama.

    `SHAPE_RESTORE_SHX` existe porque produtos do EMS chegam sem o indice `.shx`
    -- EMSR702_AOI03 e um deles. O GDAL reconstroi o indice a partir do `.shp`,
    que e informacao derivada e nao dado novo.

    Arquivo que ainda assim nao abrir NAO e ignorado em silencio: ele volta na
    lista de ilegiveis e o pacote inteiro passa a se declarar INCOMPLETO. Um
    footprint perdido encolhe a area observada, e area observada menor com
    positivo intacto e um pacote que parece melhor do que e.
    """
    saida, ilegiveis = [], []
    with fiona.Env(SHAPE_RESTORE_SHX="YES"):
        for shp in sorted(extraido.rglob(f"*_{sufixo}_v*.shp")):
            try:
                with fiona.open(shp) as fonte:
                    origem = CRS.from_user_input(fonte.crs)
                    t = Transformer.from_crs(origem, CRS.from_epsg(destino),
                                             always_xy=True).transform
                    for feicao in fonte:
                        g = shape(feicao["geometry"])
                        if g.is_empty:
                            continue
                        saida.append((_sanear(shapely_transform(t, g)),
                                      dict(feicao["properties"])))
            except Exception as e:                               # noqa: BLE001
                ilegiveis.append({"arquivo": shp.name,
                                  "erro": type(e).__name__,
                                  "detalhe": str(e)[:160]})
    return saida, ilegiveis


def montar_camadas(aois: list[tuple], footprints: list[tuple],
                   eventos: list[tuple]) -> dict:
    """A algebra de conjuntos do pacote, separada do IO para poder ser testada.

    Recebe listas de (geometria, propriedades) ja no mesmo CRS metrico e devolve
    as camadas dissolvidas. Toda a definicao de negativo vive aqui.
    """
    if not aois:
        raise ValueError("nenhuma area de interesse")
    if not footprints:
        raise ValueError(
            "nenhum footprint de imagem: sem area observada nenhum negativo "
            "pode ser afirmado")

    aoi = _sanear(union_all([g for g, _ in aois]))
    footprint = _sanear(union_all([g for g, _ in footprints]))
    area_observada = _sanear(aoi.intersection(footprint))

    def uniao_por_tipo(prefixo: str | None):
        sel = [g for g, p in eventos
               if prefixo is None
               or str(p.get("event_type", "")).startswith(prefixo)]
        if not sel:
            return None
        return _sanear(union_all(sel)).intersection(area_observada)

    observado_inundado = uniao_por_tipo(None)
    positivo = uniao_por_tipo(TIPO_INUNDACAO)
    deslizamento = uniao_por_tipo(TIPO_DESLIZAMENTO)

    negativo = area_observada
    for retirar in (observado_inundado, deslizamento):
        if retirar is not None and not retirar.is_empty:
            negativo = _sanear(negativo.difference(retirar))

    return {"aoi": aoi, "area_observada": area_observada,
            "observado_inundado": observado_inundado,
            "positivo_inundacao": positivo, "deslizamento": deslizamento,
            "negativo_observado": negativo}


def gravar(caminho: Path, geometria, epsg: int) -> str:
    """Grava uma geometria dissolvida como parquet WKB, com o CRS no metadado."""
    binario = wkb.dumps(geometria)
    tabela = pa.table({"geometry": pa.array([binario], type=pa.binary())})
    tabela = tabela.replace_schema_metadata({
        "geo": json.dumps({"primary_column": "geometry",
                           "columns": {"geometry": {"encoding": "WKB",
                                                    "crs": f"EPSG:{epsg}"}}}),
    })
    pq.write_table(tabela, caminho)
    return hashlib.sha256(binario).hexdigest()


def area_km2(g) -> float:
    return round(g.area / 1e6, 4)


def extrair_se_preciso(ativacao: str, extraido: Path | None) -> Path:
    if extraido is not None:
        return extraido
    destino = OUT / ativacao / "extraido"
    if destino.exists() and any(destino.iterdir()):
        return destino
    zipado = PACOTES / f"{ativacao}_products.zip"
    if not zipado.exists():
        raise FileNotFoundError(
            f"pacote ausente: {zipado}. Baixe a ativacao antes, ou aponte "
            "--extraido para um diretorio ja extraido")
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zipado) as z:
        z.extractall(destino)
    # O pacote do EMS e um zip de zips: um por produto (AOI x tipo x versao).
    # Sem descer esse nivel, nenhum shapefile aparece e o script recusaria por
    # "nenhuma camada", que e a mensagem certa para o motivo errado.
    while True:
        internos = [z for z in destino.rglob("*.zip")]
        if not internos:
            break
        for interno in internos:
            alvo = interno.with_suffix("")
            alvo.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(interno) as z:
                    z.extractall(alvo)
            except zipfile.BadZipFile:
                print(f"  aviso: {interno.name} nao e zip valido, ignorado")
            interno.unlink()
    return destino


def meta_da_ativacao(ativacao: str) -> dict:
    if not CATALOGO.exists():
        return {}
    return json.loads(CATALOGO.read_text(encoding="utf-8")).get(ativacao, {})


def construir(ativacao: str, extraido: Path, epsg: int | None) -> dict:
    inicio = time.time()
    meta = meta_da_ativacao(ativacao)

    if epsg is None:
        amostra, _ = ler_camada(extraido, CAMADA_AOI, 4326)
        if not amostra:
            raise ValueError(f"nenhuma camada {CAMADA_AOI} em {extraido}")
        c = union_all([g for g, _ in amostra]).centroid
        epsg = crs_metrico_de(c.x, c.y)

    aois, falhas_aoi = ler_camada(extraido, CAMADA_AOI, epsg)
    footprints, falhas_fp = ler_camada(extraido, CAMADA_FOOTPRINT, epsg)
    eventos, falhas_ev = ler_camada(extraido, CAMADA_EVENTO, epsg)
    ilegiveis = falhas_aoi + falhas_fp + falhas_ev
    if not aois:
        raise ValueError(f"nenhuma AOI em {extraido}")
    if not footprints:
        raise ValueError(
            f"nenhum {CAMADA_FOOTPRINT} em {extraido}: sem footprint nao existe "
            "area observada, e nenhum negativo pode ser afirmado")

    camadas = montar_camadas(aois, footprints, eventos)
    aoi = camadas["aoi"]
    area_observada = camadas["area_observada"]
    observado_inundado = camadas["observado_inundado"]
    positivo = camadas["positivo_inundacao"]
    deslizamento = camadas["deslizamento"]

    destino = OUT / ativacao
    destino.mkdir(parents=True, exist_ok=True)
    hashes, areas = {}, {}
    for nome, g in camadas.items():
        if g is None or g.is_empty:
            areas[nome] = 0.0
            continue
        hashes[nome] = gravar(destino / f"{nome}.parquet", g, epsg)
        areas[nome] = area_km2(g)

    tipos = {}
    for _, p in eventos:
        t = str(p.get("event_type", "") or "sem_event_type")
        tipos[t] = tipos.get(t, 0) + 1

    return {
        "tipo": "pacote_areal_cems",
        "gerado_por": "scripts/geo/cems01_pacote_areal.py",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ativacao": ativacao,
        "evento": meta.get("name"),
        "paises": meta.get("countries"),
        "data_evento": (meta.get("eventTime") or "")[:10] or None,
        "categoria": meta.get("category"),
        "crs_metrico": epsg,
        "produtos_lidos": {
            CAMADA_AOI: len(aois),
            CAMADA_FOOTPRINT: len(footprints),
            CAMADA_EVENTO: len(eventos),
        },
        "event_type_na_origem": tipos,
        "areas_km2": areas,
        "razao_neg_pos": (round(areas["negativo_observado"]
                                / areas["positivo_inundacao"], 2)
                          if areas.get("positivo_inundacao") else None),
        "sha256_por_camada": hashes,
        "definicao_negativo": (
            "(areaOfInterestA INTERSECAO imageFootprintA) MENOS inundacao MENOS "
            "deslizamento. A intersecao com o footprint e obrigatoria: area da "
            "AOI que a imagem nao cobriu nao foi observada e nao pode virar "
            "negativo. O deslizamento sai das DUAS classes -- nao e inundacao, "
            "e tambem nao e negativo util, porque o terreno ali foi alterado."),
        "separacao_de_mecanismo": (
            f"`event_type` do {CAMADA_EVENTO} distingue '{TIPO_INUNDACAO}' de "
            f"'{TIPO_DESLIZAMENTO}' na origem, e nao por heuristica nossa"),
        "camadas_ilegiveis": ilegiveis,
        "status": (("CANDIDATO: nenhum ponto amostrado, nenhum negativo promovido "
                    "ao dataset, nenhum gate alterado") if not ilegiveis else
                   (f"INCOMPLETO: {len(ilegiveis)} camada(s) nao abriram; a area "
                    "observada pode estar subestimada e o pacote nao deve ser "
                    "comparado com outro sem essa ressalva")),
        "nao_e": (
            "nao e mapa de suscetibilidade e nao e rotulo promovido: e a extensao "
            "de agua delineada por analista sobre a area que a imagem observou, "
            "num unico evento. Delineacao de satelite ve lamina de area grande e "
            "nao ve alagamento urbano raso sob dossel, predio ou viaduto"),
        "referencia_de_reproducao": REFERENCIA,
        "segundos": round(time.time() - inicio, 1),
    }


def conferir_reproducao(resumo: dict) -> dict | None:
    """Compara com o pacote antigo de EMSR720, que este script veio substituir."""
    antigo = RUNS / "br-01-cems-negativo"
    if resumo["ativacao"] != "EMSR720" or not antigo.exists():
        return None
    conferencia = {}
    for nome in ("aoi", "area_observada", "observado_inundado",
                 "positivo_inundacao", "negativo_observado"):
        arquivo = antigo / f"{nome}.parquet"
        if not arquivo.exists():
            continue
        g = wkb.loads(pq.read_table(arquivo).column("geometry")[0].as_py())
        esperado, obtido = area_km2(g), resumo["areas_km2"].get(nome)
        conferencia[nome] = {
            "km2_do_pacote_antigo": esperado,
            "km2_reproduzido": obtido,
            "diferenca_km2": round(abs(esperado - (obtido or 0.0)), 4),
        }
    conferencia["criterio"] = "diferenca <= 0,01 km2 em cada camada"
    conferencia["reproduz"] = all(
        v["diferenca_km2"] <= 0.01 for v in conferencia.values()
        if isinstance(v, dict))
    return conferencia


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ativacao", required=True)
    p.add_argument("--extraido", default=None,
                   help="diretorio ja extraido; sem isto, extrai de cems-05-pacotes")
    p.add_argument("--crs", type=int, default=None,
                   help="EPSG metrico; sem isto, UTM WGS84 da zona do centroide")
    ns = p.parse_args()

    extraido = Path(ns.extraido) if ns.extraido else None
    if extraido is not None and not extraido.is_absolute():
        extraido = REPO / extraido
    try:
        extraido = extrair_se_preciso(ns.ativacao, extraido)
    except FileNotFoundError as e:
        print(f"ABORTADO: {e}")
        return 2

    print(f"=== PACOTE AREAL {ns.ativacao} ===")
    resumo = construir(ns.ativacao, extraido, ns.crs)
    print(f"evento: {resumo['evento']} ({resumo['data_evento']})")
    print(f"CRS metrico: EPSG:{resumo['crs_metrico']}  "
          f"produtos: {resumo['produtos_lidos']}")
    print(f"event_type na origem: {resumo['event_type_na_origem']}")
    for nome, v in resumo["areas_km2"].items():
        print(f"  {nome:20s} {v:10.4f} km2")
    print(f"razao negativo/positivo: {resumo['razao_neg_pos']}")

    conf = conferir_reproducao(resumo)
    if conf:
        resumo["conferencia_de_reproducao"] = conf
        print("\nCONFERENCIA CONTRA O PACOTE ANTERIOR:")
        for nome, v in conf.items():
            if isinstance(v, dict):
                print(f"  {nome:20s} antigo={v['km2_do_pacote_antigo']:10.4f}  "
                      f"novo={v['km2_reproduzido']:10.4f}  "
                      f"dif={v['diferenca_km2']:.4f}")
        print(f"  reproduz: {conf['reproduz']}")

    destino = OUT / ns.ativacao
    (destino / "resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/geo/cems01_pacote_areal.py --ativacao EMSR720 "
        "--extraido local_runs/br-01-cems-negativo/extraido --crs 31982\n",
        encoding="utf-8")
    print(f"\nGRAVADO={destino}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
