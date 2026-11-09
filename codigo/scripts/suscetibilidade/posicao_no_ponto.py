"""Posicao topografica no ponto, a 30 m, com a MESMA funcao no lote e na consulta de um lugar novo.

-------------------------------------------------------------------------
O QUE A v1 (`posicao_topografica.py`) ERRAVA
-------------------------------------------------------------------------

  suporte   a elevacao do ponto vinha do MDT a 30 m e o minimo da janela vinha
            do MDT agregado a 240 m (media): 0,7% dos pos_min_1km sairam
            negativos, o que e impossivel para "acima do minimo da janela"
  borda     janela com metade valida na borda do raster entrava; numa consulta
            de lugar novo o MDT cobre a janela inteira e o valor seria outro
  consulta  nao havia funcao de consulta: lote e inferencia podiam divergir

-------------------------------------------------------------------------
v2
-------------------------------------------------------------------------

  pos_min_{r}  z(ponto) - min(z na janela), tudo a 30 m (>= 0 por construcao)
  dev_{r}      (z - media da janela) / desvio da janela (De Reu et al. 2013,
               Geomorphology 186: 39-49)
  janela       quadrada, lado 2k+1 celulas, k = round(r / 30), r = 1/2/4 km
  borda        janela que sai do raster -> NaN (o que esta fora e desconhecido)
  NoData       mar e vazio do MDT sao propriedade do MDT, iguais no treino e na
               consulta: aceitos ate metade da janela
  consulta     `no_ponto(arq, lon, lat)` le so a janela e chama a mesma funcao

Por unidade grava tambem o quantil 1% do MDT inteiro (`piso_mdt_q01`), o piso
que nao depende de quais pontos foram amostrados -- usado na auditoria do A16.

nao_e: nao e rotulo nem observacao de agua. O MDT e `dem_filled` (depressoes
preenchidas), o mesmo de onde vem `elevation_m`; FABDEM/GLO-30 e atual e o
evento pode ser antigo.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("servico", "suscetibilidade", "metodo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

RUNS = REPO / "local_runs"
TER01 = RUNS / "ter-01-cadeia-harmonizada"
VERSAO = "v2"
SAIDA = RUNS / "posicao-topografica" / VERSAO
SUPORTE_M = 30.0
RAIOS_M = (1000, 2000, 4000)
FRACAO_VALIDA_MIN = 0.5
COLUNAS = tuple(f"{p}_{r // 1000}km" for r in RAIOS_M for p in ("pos_min", "dev"))


def raio_em_celulas(r: float, px: float = SUPORTE_M) -> int:
    return max(1, int(round(r / px)))


def descritores(z: np.ndarray, lin: int, col: int, px: float = SUPORTE_M) -> dict:
    """Descritores de um ponto a partir do MDT em memoria. A mesma funcao serve ao lote e a consulta."""
    out = {c: np.nan for c in COLUNAS}
    h, w = z.shape
    if not (0 <= lin < h and 0 <= col < w):
        return out
    z0 = z[lin, col]
    if not np.isfinite(z0):
        return out
    for r in RAIOS_M:
        k = raio_em_celulas(r, px)
        if lin - k < 0 or col - k < 0 or lin + k >= h or col + k >= w:
            continue                                   # janela sai do raster: desconhecido, nao truncado
        jan = z[lin - k:lin + k + 1, col - k:col + k + 1]
        ok = np.isfinite(jan)
        if ok.mean() < FRACAO_VALIDA_MIN:
            continue
        v = jan if ok.all() else jan[ok]
        sd = float(v.std())
        out[f"pos_min_{r // 1000}km"] = float(z0 - v.min())
        out[f"dev_{r // 1000}km"] = float((z0 - v.mean()) / sd) if sd > 0 else np.nan
    return out


def _ler(arq: Path):
    import rasterio
    with rasterio.open(arq) as r:
        z = r.read(1, masked=True).astype("float32").filled(np.nan)
        return z, r.transform, r.crs, abs(r.transform.a)


def no_ponto(arq: Path, lon: float, lat: float) -> dict:
    """Consulta de um lugar: le so a janela do maior raio em torno do ponto e aplica `descritores`."""
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import Window
    with rasterio.open(arq) as r:
        px = abs(r.transform.a)
        k = raio_em_celulas(max(RAIOS_M), px)
        xs, ys = transform("EPSG:4326", r.crs, [lon], [lat])
        lin, col = rasterio.transform.rowcol(r.transform, xs[0], ys[0])
        janela = Window(col - k, lin - k, 2 * k + 1, 2 * k + 1)
        # boundless: o que sai do raster vira NaN e a regra de borda o reconhece como desconhecido
        z = r.read(1, window=janela, boundless=True, masked=True).astype("float32").filled(np.nan)
        fora = np.zeros_like(z, dtype=bool)
        lins = np.arange(lin - k, lin + k + 1)[:, None]
        cols = np.arange(col - k, col + k + 1)[None, :]
        fora |= (lins < 0) | (lins >= r.height) | (cols < 0) | (cols >= r.width)
    if fora.any():
        # replica a regra de borda: raios cuja janela toca fora do raster ficam NaN
        out = {c: np.nan for c in COLUNAS}
        for rr in RAIOS_M:
            kk = raio_em_celulas(rr, px)
            sub = fora[k - kk:k + kk + 1, k - kk:k + kk + 1]
            if not sub.any():
                d = descritores(z[k - kk:k + kk + 1, k - kk:k + kk + 1], kk, kk, px)
                for p in ("pos_min", "dev"):
                    out[f"{p}_{rr // 1000}km"] = d[f"{p}_{rr // 1000}km"]
        return out
    return descritores(z, k, k, px)


def derivar_unidade(arq: Path, pts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import rasterio
    from rasterio.warp import transform
    z, t, crs, px = _ler(arq)
    xs, ys = transform("EPSG:4326", crs, pts.lon.tolist(), pts.lat.tolist())
    lin, col = rasterio.transform.rowcol(t, xs, ys)
    linhas = [descritores(z, int(a), int(b), px) for a, b in zip(lin, col)]
    out = pd.DataFrame(linhas, columns=list(COLUNAS))
    out.insert(0, "ponto_id", pts.ponto_id.to_numpy())
    # o valor do ponto no MDT, para provar que e o mesmo de elevation_m
    h, w = z.shape
    lin, col = np.asarray(lin), np.asarray(col)
    dentro = (lin >= 0) & (lin < h) & (col >= 0) & (col < w)
    zp = np.full(len(pts), np.nan)
    zp[dentro] = z[lin[dentro], col[dentro]]
    out["z_mdt_no_ponto"] = zp
    valido = z[np.isfinite(z)]
    unidade = {"celulas_validas": int(valido.size), "fracao_valida": round(float(valido.size / z.size), 4),
               "piso_mdt_q01": float(np.quantile(valido, 0.01)) if valido.size else np.nan,
               "piso_mdt_min": float(valido.min()) if valido.size else np.nan, "px_m": px, "crs": str(crs)}
    return out, unidade


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def pasta_da_unidade(fonte: str, grupo: str) -> Path:
    """Diretorio de terreno da unidade. CEMS aceita a raiz da ativacao, como `cems03.terreno_da_aoi`:
    o EMSR720 tem UM terreno para as cinco AOIs (a v1 procurava so por AOI e perdeu 10.792 pontos)."""
    if fonte == "gfd":
        return TER01 / ("gfd_tile_" + grupo.split("_", 1)[1])
    por_aoi = TER01 / grupo
    if (por_aoi / "dem_filled.tif").exists():
        return por_aoi
    return TER01 / grupo.split("_AOI", 1)[0]


def derivar(fontes=("gfd", "cems")) -> dict:
    from ds06_dataset_de_treino import versao_atual
    from proveniencia import agora_utc, sha256_json
    origem = versao_atual()
    d = pd.read_parquet(origem / "pontos.parquet",
                        columns=["ponto_id", "fonte", "grupo_validacao", "lat", "lon", "elevation_m"])
    d = d[d.fonte.isin(fontes)]
    partes, unidades, sem_mdt = [], [], []
    inicio = time.time()
    for i, ((f, g), pts) in enumerate(d.groupby(["fonte", "grupo_validacao"])):
        arq = pasta_da_unidade(f, g) / "dem_filled.tif"
        if not arq.exists():
            sem_mdt.append({"fonte": f, "unidade": g, "pontos": int(len(pts))})
            continue
        t, u = derivar_unidade(arq, pts)
        t["elevation_m_ds06"] = pts.elevation_m.to_numpy(float)
        partes.append(t)
        man = arq.parent / "run_manifest.json"
        unidades.append({"fonte": f, "unidade": g, "mdt": str(arq.relative_to(REPO)), "bytes": arq.stat().st_size,
                         "run_manifest_sha256": _sha(man) if man.exists() else None, **u})
        if i % 50 == 0:
            print(i, round(time.time() - inicio), "s", flush=True)
    t = pd.concat(partes, ignore_index=True)
    dif = (t.z_mdt_no_ponto - t.elevation_m_ds06).abs()
    SAIDA.mkdir(parents=True, exist_ok=True)
    alvo = SAIDA / "pontos.parquet"
    t.drop(columns=["elevation_m_ds06"]).to_parquet(alvo, index=False)
    pd.DataFrame(unidades).to_parquet(SAIDA / "unidades.parquet", index=False)
    man = {"tipo": "posicao_no_ponto", "versao": VERSAO, "quando_utc": agora_utc(),
           "dataset": origem.name, "pontos_ds06_sha256": _sha(origem / "pontos.parquet"),
           "codigo_sha256": _sha(Path(__file__)), "saida": str(alvo.relative_to(REPO)), "saida_sha256": _sha(alvo),
           "linhas": int(len(t)), "suporte_m": SUPORTE_M, "raios_m": RAIOS_M, "fracao_valida_min": FRACAO_VALIDA_MIN,
           "regra_de_borda": "janela que sai do raster -> NaN",
           "nan_por_coluna": {c: round(float(t[c].isna().mean()), 4) for c in COLUNAS},
           "pos_min_negativo": {c: int((t[c] < 0).sum()) for c in COLUNAS if c.startswith("pos_min")},
           "z_mdt_igual_a_elevation_m": {"mediana_abs_m": round(float(dif.median()), 4),
                                         "p99_abs_m": round(float(dif.quantile(0.99)), 4)},
           "unidades": len(unidades), "sem_mdt": sem_mdt, "unidades_sha256": sha256_json(unidades),
           "segundos": round(time.time() - inicio),
           "nao_e": "descritor de terreno sem rotulo; nao e observacao de agua"}
    (SAIDA / "manifesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return man


def juntar(d: pd.DataFrame) -> pd.DataFrame:
    """Junta os descritores v2 ao ds06; falha se foram derivados sobre outro arquivo de pontos."""
    from ds06_dataset_de_treino import versao_atual
    man = json.loads((SAIDA / "manifesto.json").read_text(encoding="utf-8"))
    if man["pontos_ds06_sha256"] != _sha(versao_atual() / "pontos.parquet"):
        raise RuntimeError("posicao no ponto derivada sobre outro ds06; refazer")
    t = pd.read_parquet(SAIDA / "pontos.parquet", columns=["ponto_id", *COLUNAS])
    return d.merge(t, on="ponto_id", how="left", validate="many_to_one")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    m = derivar()
    print(json.dumps({k: m[k] for k in ("linhas", "unidades", "nan_por_coluna", "pos_min_negativo",
                                        "z_mdt_igual_a_elevation_m", "segundos")}, ensure_ascii=False))
    print(f"GRAVADO={SAIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
