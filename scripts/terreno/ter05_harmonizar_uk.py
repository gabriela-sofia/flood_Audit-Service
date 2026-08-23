"""
TER-05 -- Traz o piloto ingles para a cadeia harmonizada.

POR QUE O UK FICOU DE FORA ATE AGORA, e por que isso e caro:

o `ter02` casa ponto e raster pelo `grupo_cv`, e no CEMS o grupo E a AOI
(`EMSR847_AOI01`). No piloto ingles o grupo e o EVENTO (`EV_1.0`), sao 401
deles, e nenhum tem raster com esse nome. O UK entao nunca entrou na
sobreposicao harmonizada -- nao por decisao, por incompatibilidade de chave.

A consequencia so ficou visivel ao montar a tabela unica: o modelo fluvial
multirregiao rodou sem os 7.476 pontos ingleses, que sao justamente a fonte do
holdout temporal de 25 anos (nove folds, AUC medio 0,7686). A unica fonte com
validacao prospectiva estava fora do conjunto que ela deveria validar.

O QUE ESTE SCRIPT FAZ:

amostra as quatro variaveis de terreno dos pontos do UK no raster derivado
pelo `ter01` para a AOI unica do piloto, compara contra a cadeia atual e grava
no MESMO formato que o `ter02` produz -- `ponto_id` mais colunas `_wbt` --
para que o `ds04` sobreponha sem saber que a origem e outra.

O QUE ELE NAO FAZ: nao deriva raster (isso e ter01), nao treina, e nao decide
se a troca melhora o modelo. Mede a troca e registra.

Pre-requisito:
    python scripts/terreno/ter01_cadeia_harmonizada.py \\
        --regiao uk_noroeste_harmonizado --bbox -2.7625,53.0466,-2.0020,53.9460

Uso:
    python scripts/terreno/ter05_harmonizar_uk.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
DERIV = RUNS / "ter-01-cadeia-harmonizada" / "uk_noroeste_harmonizado"
DATASET = RUNS / "ds-01-multirregiao" / "dataset_multirregiao_v1.csv"
OUT = RUNS / "ter-02-comparacao"          # mesmo diretorio: o ds04 le de la
REGIAO = "UK_noroeste"

MAPA = {"hand_m": "hand_dinf", "twi_dinf": "twi_dinf",
        "slope_deg": "slope_deg_wbt", "elevation_m": "dem_filled"}


def amostrar(raster: Path, lons, lats) -> np.ndarray:
    """Mesma funcao de amostragem do ter02, com o mesmo tratamento de nodata.

    -9999 (cadeia antiga) e -32768 (WhiteboxTools) sao finitos e passariam
    direto para a coluna de variavel se so se filtrasse valor nao finito.
    """
    import rasterio
    from rasterio.warp import transform as wtr

    warnings.filterwarnings("ignore")
    with rasterio.open(raster) as r:
        xs, ys = wtr("EPSG:4326", r.crs, list(lons), list(lats))
        v = np.array([x[0] for x in r.sample(zip(xs, ys))], dtype="float64")
        nd = r.nodata
    if nd is not None:
        v[np.isclose(v, nd)] = np.nan
    v[~np.isfinite(v)] = np.nan
    v[v < -1000] = np.nan
    return v


def main() -> int:
    print("===TER05_HARMONIZAR_UK===")
    if not (DERIV / "run_manifest.json").exists():
        print(f"ABORTADO: derivacao ausente em {DERIV}.")
        print("Rode o ter01 com --regiao uk_noroeste_harmonizado e o bbox do piloto.")
        return 1
    if not DATASET.exists():
        print(f"ABORTADO: {DATASET} ausente. Rode ds01.")
        return 1

    man = json.loads((DERIV / "run_manifest.json").read_text(encoding="utf-8"))
    print(f"derivacao: {man['crs']} px={man['pixel_size'][0]}m "
          f"canal={man['stream_area_km2']} km2 "
          f"(p{man['stream_percentile_equivalente']:.1f}) shape={man['shape']}")

    df = pd.read_csv(DATASET, low_memory=False)
    d = df[(df.regiao == REGIAO) & df.classe.isin([0, 1])].copy()
    print(f"PONTOS={len(d):,} em {d.grupo_cv.nunique()} eventos")

    linhas = []
    for col, camada in MAPA.items():
        r = DERIV / f"{camada}.tif"
        if not r.exists():
            print(f"   FALTA raster {camada}")
            d[f"{col}_wbt"] = np.nan
            continue
        novo = amostrar(r, d.lon, d.lat)
        d[f"{col}_global"] = d[col]
        d[f"{col}_wbt"] = novo

        a = d[f"{col}_global"].to_numpy(dtype="float64")
        ok = np.isfinite(a) & np.isfinite(novo)
        if ok.sum() < 5:
            continue
        cl = d.classe.to_numpy()
        kp, kn = (cl == 1) & ok, (cl == 0) & ok
        pear = float(np.corrcoef(a[ok], novo[ok])[0, 1])
        ma, mn = float(np.median(a[ok])), float(np.median(novo[ok]))
        ca = float(np.median(a[kn]) - np.median(a[kp])) if kp.any() and kn.any() else None
        cn = float(np.median(novo[kn]) - np.median(novo[kp])) if kp.any() and kn.any() else None
        linhas.append({"regiao": REGIAO, "variavel": col, "n": int(ok.sum()),
                       "pearson": round(pear, 4),
                       "mediana_global": round(ma, 3), "mediana_wbt": round(mn, 3),
                       "razao": round(ma / mn, 3) if mn else None,
                       "contraste_global": round(ca, 3) if ca is not None else None,
                       "contraste_wbt": round(cn, 3) if cn is not None else None,
                       "fora_do_raster": int((~np.isfinite(novo)).sum())})
        print(f"   {col:14s} pearson={pear:6.3f}  med {ma:9.2f} -> {mn:8.2f}  "
              f"contraste {str(ca):>9s} -> {str(cn):>8s}  "
              f"fora={int((~np.isfinite(novo)).sum())}")

    cols_wbt = [f"{c}_wbt" for c in MAPA]
    completo = d[cols_wbt].notna().all(axis=1)
    print(f"\nCOM_AS_QUATRO={int(completo.sum()):,}  "
          f"INCOMPLETO={int((~completo).sum()):,} "
          f"(ponto fora do recorte vira nulo, nao valor)")

    saida = OUT / "dataset_harmonizado_uk.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    d[["ponto_id", "grupo_cv", "classe", *cols_wbt]].to_csv(saida, index=False)
    pd.DataFrame(linhas).to_csv(OUT / "comparacao_uk.csv", index=False)
    (OUT / "resumo_uk.json").write_text(json.dumps({
        "regiao": REGIAO,
        "pontos": int(len(d)), "com_as_quatro": int(completo.sum()),
        "eventos": int(d.grupo_cv.nunique()),
        "manifesto": {k: man[k] for k in
                      ("crs", "pixel_size", "stream_area_km2",
                       "drenagem_limiar_celulas", "stream_percentile_equivalente")},
        "por_feature": linhas,
        "nota": ("o piloto ingles e uma AOI unica de ~100x50 km com 401 eventos "
                 "dentro; o grupo de validacao continua sendo o evento, nao a AOI"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"GRAVADO={saida}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
