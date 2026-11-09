"""Posicao topografica relativa so do DEM, em varias escalas: o substituto de `elevation_local_m` sem rotulo.

-------------------------------------------------------------------------
O DEFEITO QUE ISTO RESOLVE
-------------------------------------------------------------------------

`elevation_local_m` (ds06 `referencia_local`) e a elevacao acima do quantil 1%
dos PONTOS AMOSTRADOS na unidade (tile de 1 grau no GFD, AOI do analista no
CEMS). Os pontos sao escolhidos pelo rotulo (positivo no interior da mancha,
negativo na faixa de 2 km), entao o piso depende do desenho amostral e do
rotulo, e nao e o mesmo conceito entre fontes. Para um lugar NOVO, sem amostra
rotulada, a variavel nao se calcula do mesmo jeito: a resposta para lugar novo
e os testes confirmatorios a calcularam sobre os pontos rotulados do proprio
teste. E o caso de vazamento de Kaufman et al. (2012, ACM TKDD 6(4)): informacao
do alvo que nao estaria disponivel na predicao.

-------------------------------------------------------------------------
O SUBSTITUTO
-------------------------------------------------------------------------

Descritores de posicao topografica calculados SO do MDT, em janela fixa,
como o TPI de Weiss (2001) e o DEV de De Reu et al. (2013, Geomorphology 186):

  pos_min_{r}  elevacao do ponto (30 m) menos o MINIMO do MDT na janela de raio r
               -- o mesmo conceito de "acima do piso do lugar", sem amostra
  dev_{r}      (elevacao - media da janela) / desvio da janela -- De Reu 2013
               mostra que o DEV classifica melhor que o TPI em paisagem
               heterogenea

  r = 1, 2, 4 km; MDT agregado a 240 m (8 x 30 m, media) antes da janela;
  janela quadrada de lado 2k+1 celulas; celula com menos de metade da janela
  valida (borda do raster, nodata) fica NaN -- nunca zero.

Calculavel para qualquer lugar com MDT, com ou sem rotulo.

nao_e: nao e rotulo nem observacao de agua; nao corrige o MDT (FABDEM e atual,
o evento pode ser antigo); a janela truncada na borda do tile vira NaN e sai
da populacao do experimento, declarado.

Uso:
    python scripts/suscetibilidade/posicao_topografica.py
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
VERSAO = "v1"
SAIDA = RUNS / "posicao-topografica" / VERSAO
AGREGACAO = 8                    # 8 x 30 m = 240 m
RAIOS_M = (1000, 2000, 4000)
FRACAO_VALIDA_MIN = 0.5


def pasta_da_unidade(fonte: str, grupo: str) -> Path:
    return TER01 / (("gfd_tile_" + grupo.split("_", 1)[1]) if fonte == "gfd" else grupo)


def campos(z: np.ndarray, px: float) -> dict:
    """Minimo, media, desvio e fracao valida da janela, por raio, no MDT agregado."""
    from scipy.ndimage import minimum_filter, uniform_filter
    valido = np.isfinite(z)
    zi = np.where(valido, z, np.inf)
    z0 = np.where(valido, z, 0.0)
    out = {}
    for r in RAIOS_M:
        k = max(1, int(round(r / px)))
        lado = 2 * k + 1
        frac = uniform_filter(valido.astype(float), lado, mode="constant", cval=0.0)
        soma = uniform_filter(z0, lado, mode="constant", cval=0.0)
        soma2 = uniform_filter(z0 ** 2, lado, mode="constant", cval=0.0)
        n = np.where(frac > 0, frac, np.nan)
        media = soma / n
        desvio = np.sqrt(np.maximum(soma2 / n - media ** 2, 0.0))
        minimo = minimum_filter(zi, lado, mode="constant", cval=np.inf)
        ruim = frac < FRACAO_VALIDA_MIN
        out[r] = {"min": np.where(ruim | ~np.isfinite(minimo), np.nan, minimo),
                  "media": np.where(ruim, np.nan, media), "desvio": np.where(ruim, np.nan, desvio)}
    return out


def derivar_unidade(arq: Path, pts: pd.DataFrame) -> pd.DataFrame:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform
    with rasterio.open(arq) as r:
        h, w = r.height // AGREGACAO, r.width // AGREGACAO
        z = r.read(1, out_shape=(h, w), resampling=Resampling.average, masked=True).astype("float64")
        z = z.filled(np.nan)
        t = r.transform * r.transform.scale(r.width / w, r.height / h)
        px = abs(t.a)
        xs, ys = transform("EPSG:4326", r.crs, pts.lon.tolist(), pts.lat.tolist())
    lin, col = rasterio.transform.rowcol(t, xs, ys)
    lin, col = np.asarray(lin), np.asarray(col)
    dentro = (lin >= 0) & (lin < h) & (col >= 0) & (col < w)
    c = campos(z, px)
    out = pd.DataFrame({"ponto_id": pts.ponto_id.to_numpy()})
    e = pts.elevation_m.to_numpy(float)
    for r, f in c.items():
        km = r // 1000
        mn, md, dv = (np.full(len(pts), np.nan) for _ in range(3))
        mn[dentro] = f["min"][lin[dentro], col[dentro]]
        md[dentro] = f["media"][lin[dentro], col[dentro]]
        dv[dentro] = f["desvio"][lin[dentro], col[dentro]]
        out[f"pos_min_{km}km"] = e - mn
        out[f"dev_{km}km"] = np.where(dv > 0, (e - md) / dv, np.nan)
    return out


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def derivar(fontes=("gfd", "cems")) -> dict:
    from ds06_dataset_de_treino import versao_atual
    from proveniencia import agora_utc, sha256_json
    origem = versao_atual()
    d = pd.read_parquet(origem / "pontos.parquet",
                        columns=["ponto_id", "fonte", "grupo_validacao", "lat", "lon", "elevation_m"])
    d = d[d.fonte.isin(fontes)]
    partes, inventario, sem_mdt = [], [], []
    inicio = time.time()
    for (f, g), pts in d.groupby(["fonte", "grupo_validacao"]):
        pasta = pasta_da_unidade(f, g)
        arq = pasta / "dem_filled.tif"
        if not arq.exists():
            sem_mdt.append({"fonte": f, "unidade": g, "pontos": int(len(pts))})
            continue
        partes.append(derivar_unidade(arq, pts))
        st = arq.stat()
        man = pasta / "run_manifest.json"
        inventario.append({"unidade": g, "mdt": str(arq.relative_to(REPO)), "bytes": st.st_size,
                           "run_manifest_sha256": _sha(man) if man.exists() else None})
    t = pd.concat(partes, ignore_index=True)
    SAIDA.mkdir(parents=True, exist_ok=True)
    alvo = SAIDA / "pontos.parquet"
    t.to_parquet(alvo, index=False)
    man = {"tipo": "posicao_topografica", "versao": VERSAO, "quando_utc": agora_utc(),
           "dataset": origem.name, "pontos_ds06_sha256": _sha(origem / "pontos.parquet"),
           "codigo_sha256": _sha(Path(__file__)), "saida": str(alvo.relative_to(REPO)), "saida_sha256": _sha(alvo),
           "linhas": int(len(t)), "agregacao_m": AGREGACAO * 30, "raios_m": RAIOS_M,
           "fracao_valida_min": FRACAO_VALIDA_MIN,
           "nan_por_coluna": {c: round(float(t[c].isna().mean()), 4) for c in t.columns if c != "ponto_id"},
           "unidades": len(inventario), "sem_mdt": sem_mdt,
           "inventario_sha256": sha256_json(inventario),
           "hash_dos_mdt": ("conteudo dos rasters nao rehasheado (~40 GB): cada um ligado pelo hash do run_manifest "
                            "da derivacao e pelo tamanho"),
           "segundos": round(time.time() - inicio),
           "nao_e": "descritor de terreno sem rotulo; nao e observacao de agua"}
    (SAIDA / "manifesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    (SAIDA / "inventario_mdt.json").write_text(json.dumps(inventario, ensure_ascii=False, indent=0),
                                                encoding="utf-8")
    return man


def juntar(d: pd.DataFrame) -> pd.DataFrame:
    """Junta os descritores ao ds06; falha se foram derivados sobre outro arquivo de pontos."""
    from ds06_dataset_de_treino import versao_atual
    man = json.loads((SAIDA / "manifesto.json").read_text(encoding="utf-8"))
    if man["pontos_ds06_sha256"] != _sha(versao_atual() / "pontos.parquet"):
        raise RuntimeError("posicao topografica derivada sobre outro ds06; refazer")
    return d.merge(pd.read_parquet(SAIDA / "pontos.parquet"), on="ponto_id", how="left", validate="many_to_one")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    m = derivar()
    print(json.dumps({k: m[k] for k in ("linhas", "unidades", "nan_por_coluna", "segundos")}, ensure_ascii=False))
    print(f"GRAVADO={SAIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
