"""TC01: o territorio consultavel -- grade regular sobre uma delimitacao que NAO depende da inundacao.

-------------------------------------------------------------------------
A LACUNA
-------------------------------------------------------------------------

O GFD do projeto amostra positivo no interior da mancha e negativo na faixa de
2 km em volta dela (`gfd01.classes_do_rotulo`): a extensao da amostra e
construida a partir do desfecho, e um lugar consultado ANTES do evento nao tem
essa faixa. O CEMS amostra a AOI que o analista desenhou em torno da ativacao.
Nenhuma fonte do projeto responde "num lugar delimitado antes, onde a agua
chega quando ha cheia" (`verdade_observacional.POPULACOES_ALVO`). A extensao de
onde vem o fundo muda pesos e desempenho (VanDerWal et al. 2009, Ecol. Model.
220: 589-594; Phillips et al. 2009, Ecol. Appl. 19: 181-197).

-------------------------------------------------------------------------
O DESENHO
-------------------------------------------------------------------------

  delimitacao  extensao do MDT do tile de 1 grau ja derivado (ter01): grade fixa,
               escolhida sem olhar a mancha
  amostra      grade regular de PASSO_M alinhada a projecao do MODIS do evento
               (deterministica, sem semente)
  estado       positivo: flooded = 1 e fora da agua permanente do GFD
               negativo_observado: clear_views > 0, flooded = 0, fora da agua
                 permanente -- visto seco pelo sensor durante o evento
               sem_visada: clear_views = 0 (NAO e negativo)
               agua_permanente: jrc_perm_water = 1 (sai do binario)
               fora_da_cena: banda mascarada (fora do footprint do evento ou
                 fora do dominio de declive do GFD)
  terreno      hand, twi, slope, elevation do ter01 + posicao no ponto (v2),
               a mesma funcao da consulta

Piloto em eventos JA conhecidos do desenvolvimento (ds06): nao consome reserva.
A distancia a borda da mancha NAO e usada para escolher ponto; fica gravada a
classe de cada ponto e quem quiser margem de incerteza a aplica declarando.

nao_e: nao e probabilidade de ocorrencia (o tile ainda foi escolhido porque
houve evento) nem rotulo de serra; MODIS 250 m nao ve cheia curta nem urbana.

Uso:
    python scripts/geo/tc01_territorio_consultavel.py --piloto
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("servico", "suscetibilidade", "metodo", "geo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

RUNS = REPO / "local_runs"
TER01 = RUNS / "ter-01-cadeia-harmonizada"
SAIDA = RUNS / "tc01-territorio-consultavel"
COLECAO = "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1"
LICENCA = "Global Flood Database v1 (Tellman et al. 2021, Nature 596: 80-86), CC BY 4.0 -- uso com atribuicao"
PASSO_M = 1000.0
PILOTO = {"GFD:1939": 3, "GFD:4322": 3, "GFD:3782": 3}      # evento -> tiles com mais pontos no ds06
CAMADAS = {"elevation_m": "dem_filled.tif", "hand_m": "hand_dinf.tif", "twi_dinf": "twi_dinf.tif",
           "slope_deg": "slope_deg_wbt.tif"}
ESTADOS_QUE_ENSINAM = ("positivo", "negativo_observado")


def estado(flooded: float, clear_views: float, perm: float) -> str:
    """A regra de classe do territorio, sem distancia a mancha."""
    if any(v is None or (isinstance(v, float) and np.isnan(v)) or v < 0 for v in (flooded, clear_views, perm)):
        return "fora_da_cena"
    if perm == 1:
        return "agua_permanente"
    if flooded == 1:
        return "positivo"
    if clear_views > 0:
        return "negativo_observado"
    return "sem_visada"


def tiles_do_piloto() -> list[tuple[str, str, Path]]:
    from ds06_dataset_de_treino import versao_atual
    d = pd.read_parquet(versao_atual() / "pontos.parquet", columns=["fonte", "evento_id", "grupo_validacao"])
    g = d[d.fonte == "gfd"]
    out = []
    for ev, n in PILOTO.items():
        cont = g[g.evento_id == ev].grupo_validacao.value_counts()
        escolhidos = 0
        for tile in cont.index:
            arq = TER01 / ("gfd_tile_" + tile.split("_", 1)[1]) / "dem_filled.tif"
            if arq.exists():
                out.append((ev, tile, arq))
                escolhidos += 1
            if escolhidos == n:
                break
    return out


SEMENTE = 20261008
TILES_POR_EVENTO = 2


def tiles_sorteados(semente: int = SEMENTE, por_evento: int = TILES_POR_EVENTO) -> list[tuple[str, str, Path]]:
    """Todos os eventos GFD do ds06 (desenvolvimento); tiles sorteados entre os que o evento toca e tem MDT.

    O sorteio NAO usa quantos pontos o tile tem (o piloto usou, e isso escolhe onde inundou mais).
    """
    from ds06_dataset_de_treino import versao_atual
    d = pd.read_parquet(versao_atual() / "pontos.parquet", columns=["fonte", "evento_id", "grupo_validacao"])
    g = d[d.fonte == "gfd"].drop_duplicates(["evento_id", "grupo_validacao"])
    rng = np.random.default_rng(semente)
    out = []
    for ev in sorted(g.evento_id.unique()):
        tiles = sorted(t for t in g[g.evento_id == ev].grupo_validacao
                       if (TER01 / ("gfd_tile_" + t.split("_", 1)[1]) / "dem_filled.tif").exists())
        for t in rng.permutation(tiles)[:por_evento]:
            out.append((ev, t, TER01 / ("gfd_tile_" + t.split("_", 1)[1]) / "dem_filled.tif"))
    return out


def regiao(arq: Path) -> list[float]:
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(arq) as r:
        return list(transform_bounds(r.crs, "EPSG:4326", *r.bounds))


def amostrar_ee(ev: str, caixa: list[float], passo_m: float = PASSO_M) -> pd.DataFrame:
    import ee
    ee.Initialize()
    img = ee.Image(ee.ImageCollection(COLECAO).filter(ee.Filter.eq("id", int(ev.split(":")[1]))).first())
    bandas = img.select(["flooded", "clear_views", "jrc_perm_water"]).unmask(-1)
    reg = ee.Geometry.Rectangle(caixa, proj="EPSG:4326", geodesic=False)
    proj = img.select("flooded").projection().atScale(passo_m)
    amostra = bandas.addBands(ee.Image.pixelLonLat()).sample(region=reg, projection=proj, geometries=False,
                                                             dropNulls=False, tileScale=4)
    url = amostra.getDownloadURL(filetype="CSV", selectors=["flooded", "clear_views", "jrc_perm_water",
                                                            "latitude", "longitude"])
    with urllib.request.urlopen(url, timeout=900) as r:
        linhas = list(csv.DictReader(io.StringIO(r.read().decode("utf-8", errors="replace"))))
    f = lambda x: float(x) if x not in ("", None) else np.nan  # noqa: E731
    t = pd.DataFrame({"lat": [f(x["latitude"]) for x in linhas], "lon": [f(x["longitude"]) for x in linhas],
                      "flooded": [f(x["flooded"]) for x in linhas], "clear_views": [f(x["clear_views"]) for x in linhas],
                      "jrc_perm_water": [f(x["jrc_perm_water"]) for x in linhas]})
    t["estado"] = [estado(a, b, c) for a, b, c in zip(t.flooded, t.clear_views, t.jrc_perm_water)]
    return t


def terreno(t: pd.DataFrame, pasta: Path) -> pd.DataFrame:
    import rasterio
    from rasterio.warp import transform
    import posicao_no_ponto as pp
    out = t.copy()
    for col, nome in CAMADAS.items():
        with rasterio.open(pasta / nome) as r:
            xs, ys = transform("EPSG:4326", r.crs, t.lon.tolist(), t.lat.tolist())
            v = np.array([x[0] for x in r.sample(zip(xs, ys), masked=True)], dtype=object)
            out[col] = [float(x) if x is not np.ma.masked and not np.ma.is_masked(x) else np.nan for x in v]
    pos, _ = pp.derivar_unidade(pasta / "dem_filled.tif", t.assign(ponto_id=np.arange(len(t))))
    for c in pp.COLUNAS:
        out[c] = pos[c].to_numpy()
    return out


def _um(item):
    ev, tile, arq = item
    try:
        caixa = regiao(arq)
        return item, caixa, amostrar_ee(ev, caixa), None
    except Exception as e:                                       # noqa: BLE001
        return item, None, None, e


def piloto(fase: str = "piloto") -> dict:
    from concurrent.futures import ThreadPoolExecutor
    from proveniencia import agora_utc, sha256_json
    partes, registro, falhas = [], [], []
    inicio = time.time()
    lista = tiles_do_piloto() if fase == "piloto" else tiles_sorteados()
    with ThreadPoolExecutor(max_workers=1 if fase == "piloto" else 4) as pool:
        baixados = list(pool.map(_um, lista))
    for (ev, tile, arq), caixa, t, erro in baixados:
        if erro is not None:
            falhas.append({"evento": ev, "unidade": tile, "erro": type(erro).__name__, "detalhe": str(erro)[:160]})
            continue
        t = terreno(t, arq.parent)
        t.insert(0, "evento_id", ev)
        t.insert(1, "unidade", tile)
        t["ponto_id"] = [f"TC01_{ev.split(':')[1]}_{tile}_{i:06d}" for i in range(len(t))]
        partes.append(t)
        registro.append({"evento": ev, "unidade": tile, "caixa_wgs84": [round(x, 5) for x in caixa],
                         "pontos": int(len(t)), "por_estado": t.estado.value_counts().to_dict()})
        print(ev, tile, len(t), t.estado.value_counts().to_dict(), round(time.time() - inicio), "s", flush=True)
    d = pd.concat(partes, ignore_index=True)
    destino = SAIDA / fase
    destino.mkdir(parents=True, exist_ok=True)
    d.to_parquet(destino / "pontos.parquet", index=False)
    ensina = d[d.estado.isin(ESTADOS_QUE_ENSINAM)]
    term = np.isfinite(ensina[list(CAMADAS)].to_numpy(float)).all(axis=1)
    man = {"tipo": "tc01_territorio_consultavel", "fase": fase, "quando_utc": agora_utc(),
           "selecao_de_tiles": ("piloto: os de mais pontos no ds06 (DEPENDE do desfecho, so para auditar a cadeia)"
                                if fase == "piloto" else f"sorteio com semente {SEMENTE}, {TILES_POR_EVENTO} por evento, "
                                "entre os tiles com MDT que o evento toca; nao usa contagem de pontos"),
           "falhas": falhas,
           "colecao": COLECAO, "licenca": LICENCA, "passo_m": PASSO_M,
           "delimitacao": "extensao do MDT do tile ter01 (grade de 1 grau), independente da mancha",
           "regime": "desenvolvimento: eventos ja no ds06; nao toca reserva",
           "registro": registro, "registro_sha256": sha256_json(registro),
           "por_estado": d.estado.value_counts().to_dict(),
           "admissao": {"ensinam": int(len(ensina)), "com_terreno_finito": int(term.sum()),
                        "positivos": int((ensina.estado == "positivo").sum()),
                        "fracao_positiva": round(float((ensina.estado == "positivo").mean()), 4),
                        "regra": "positivo e negativo_observado; sem_visada, agua_permanente e fora_da_cena saem"},
           "segundos": round(time.time() - inicio),
           "nao_e": "probabilidade de ocorrencia; o tile foi escolhido porque houve evento"}
    (destino / "manifesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=float),
                                            encoding="utf-8")
    return man


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--piloto", action="store_true")
    ap.add_argument("--desenvolvimento", action="store_true", help="69 eventos do ds06, tiles sorteados")
    a = ap.parse_args()
    if a.piloto == a.desenvolvimento:
        ap.error("escolha --piloto ou --desenvolvimento")
    if a.desenvolvimento and not (SAIDA / "piloto" / "manifesto.json").exists():
        ap.error("a ampliacao depende da auditoria do piloto")
    m = piloto("piloto" if a.piloto else "desenvolvimento")
    print(json.dumps({k: m[k] for k in ("por_estado", "admissao", "segundos")}, ensure_ascii=False))
    print(f"GRAVADO={SAIDA / ('piloto' if a.piloto else 'desenvolvimento')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
