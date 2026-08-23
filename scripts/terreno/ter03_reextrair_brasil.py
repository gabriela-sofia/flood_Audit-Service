"""
TER-03 -- Re-extrai o terreno dos pontos brasileiros na cadeia harmonizada.

POR QUE:
os 278 pontos de Recife e as 1.680 linhas / 1.471 unidades de Curitiba carregam
`hand_m_dinf` e `twi_dinf` derivados com limiar de canal em PERCENTIL, que nao
transfere entre regioes (ver ext_hand_incomparavel_entre_regioes_v1.md):

    Recife      0,1123 km2 de area contribuinte
    Curitiba    0,0847 km2
    Petropolis  0,7507 km2      <- 6,7x mais esparso que Recife

Todos rotulados "percentil 98". Sao tres definicoes de canal, logo tres
variaveis diferentes com o mesmo nome. Enquanto isso nao for refeito, qualquer
juncao com o conjunto externo herda o erro.

O QUE ESTE SCRIPT FAZ, e o que NAO faz:
re-amostra APENAS as quatro variaveis de terreno (elevacao, declividade, HAND,
TWI) dos rasters do `ter01`. Chuva e uso do solo ficam intactos -- nao sao
derivados de DEM e nao tem o problema. As colunas originais sao preservadas com
sufixo `_original`; nada e sobrescrito no arquivo de origem.

DUAS DIFERENCAS QUE ESTE SCRIPT NAO RESOLVE, e que precisam ficar declaradas:

  1. RESOLUCAO. Recife e Curitiba foram derivados de MDT de 10 m; a cadeia
     harmonizada roda a 30 m, porque 30 m e o unico que existe nas regioes
     externas. Medido em Recife com limiar identico, a mudanca de 10 m para
     30 m altera o HAND por fator 1,60. O ganho e comparabilidade; o custo e
     detalhe. Esta e uma troca declarada, nao um efeito colateral.

  2. FONTE DE CHUVA. Recife usa CHIRPS; Curitiba usa Open-Meteo ERA5-Land,
     apesar de as colunas terem sufixo `_chirps`. O campo `rain_data_source`
     registra a diferenca. Isso NAO e resolvido aqui e impede tratar as duas
     chuvas como a mesma variavel num modelo conjunto.

NAO faz: nao treina, nao altera arquivo de origem, nao mexe em chuva nem em
uso do solo, nao decide qual versao e a boa.

Uso:
    python scripts/terreno/ter03_reextrair_brasil.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROJETO = REPO.parent / "PROJETO"
RUNS = REPO / "local_runs"
DERIV = RUNS / "ter-01-cadeia-harmonizada"
OUT = RUNS / "ter-03-brasil-harmonizado"

# coluna no dataset -> raster do ter01
MAPA = {"elevation_m": "dem_filled", "slope_deg": "slope_deg_wbt",
        "hand_m_dinf": "hand_dinf", "twi_dinf": "twi_dinf"}

FONTES = {
    "recife": {
        "tabela": PROJETO / "local_runs" / "recife_modelo_v12_extracao_final" / "dataset_v12_final.csv",
        "deriv": DERIV / "recife_harmonizado",
        "grupo": "ponto_id",
        "esperado": 278,
    },
    "curitiba": {
        "tabela": (REPO / "outputs_public" / "data"
                   / "susc_20k_siac156_curitiba_flood_candidates" / "registries"
                   / "v20n_dataset_curitiba_variaveis_v2.csv"),
        "deriv": DERIV / "curitiba_harmonizado",
        "grupo": "chave_unidade_observacao",
        "esperado": 1680,
    },
}


def amostrar(raster: Path, lons, lats) -> np.ndarray:
    """Respeita o nodata declarado. -9999 e -32768 sao FINITOS e passariam."""
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
    OUT.mkdir(parents=True, exist_ok=True)
    print("===TER03_REEXTRAIR_BRASIL===")
    relatorio = {}

    for nome, cfg in FONTES.items():
        tab, deriv = cfg["tabela"], cfg["deriv"]
        if not tab.exists():
            print(f"[{nome}] ABORTADO: tabela ausente -> {tab}")
            return 1
        if not (deriv / "run_manifest.json").exists():
            print(f"[{nome}] ABORTADO: derivacao ausente -> {deriv}. Rode ter01.")
            return 1

        d = pd.read_csv(tab, low_memory=False)
        if len(d) != cfg["esperado"]:
            print(f"[{nome}] AVISO: {len(d)} linhas, esperado {cfg['esperado']}. "
                  f"Prossigo, mas confira a fonte.")
        grupos = d[cfg["grupo"]].nunique()
        print(f"\n[{nome}] linhas={len(d):,} grupos({cfg['grupo']})={grupos:,} "
              f"pos={int((d.rotulo == 1).sum()):,} neg={int((d.rotulo == 0).sum()):,}")

        man = json.loads((deriv / "run_manifest.json").read_text(encoding="utf-8"))
        print(f"   derivacao: {man['crs']} px={man['pixel_size'][0]}m "
              f"canal={man['drenagem_limiar_celulas']:.1f} cel "
              f"({man['stream_area_km2']} km2, p{man['stream_percentile_equivalente']:.1f})")

        linhas = []
        for col, camada in MAPA.items():
            r = deriv / f"{camada}.tif"
            if not r.exists():
                print(f"   FALTA raster {camada}")
                continue
            novo = amostrar(r, d.lon, d.lat)
            d[f"{col}_original"] = d[col] if col in d.columns else np.nan
            d[f"{col}_wbt30"] = novo

            a = d[f"{col}_original"].to_numpy(dtype="float64")
            ok = np.isfinite(a) & np.isfinite(novo)
            if ok.sum() < 5:
                continue
            pear = float(np.corrcoef(a[ok], novo[ok])[0, 1])
            ma, mn = float(np.median(a[ok])), float(np.median(novo[ok]))
            lab = d.rotulo.to_numpy()
            kp, kn = (lab == 1) & ok, (lab == 0) & ok
            ca = float(np.median(a[kn]) - np.median(a[kp])) if kp.any() and kn.any() else None
            cn = float(np.median(novo[kn]) - np.median(novo[kp])) if kp.any() and kn.any() else None
            linhas.append({"regiao": nome, "variavel": col, "n": int(ok.sum()),
                           "pearson": round(pear, 4),
                           "mediana_original": round(ma, 3), "mediana_wbt30": round(mn, 3),
                           "razao": round(ma / mn, 3) if mn else None,
                           "contraste_original": round(ca, 3) if ca is not None else None,
                           "contraste_wbt30": round(cn, 3) if cn is not None else None,
                           "perdidos_nodata": int((~np.isfinite(novo)).sum())})
            print(f"   {col:14s} pearson={pear:6.3f}  med {ma:9.2f} -> {mn:8.2f}  "
                  f"contraste {str(ca):>9s} -> {str(cn):>8s}  "
                  f"nodata={int((~np.isfinite(novo)).sum())}")

        d.to_csv(OUT / f"{nome}_harmonizado.csv", index=False)
        relatorio[nome] = {"linhas": int(len(d)), "grupos": int(grupos),
                           "grupo_col": cfg["grupo"],
                           "manifesto": {k: man[k] for k in
                                         ("crs", "pixel_size", "stream_area_km2",
                                          "drenagem_limiar_celulas",
                                          "stream_percentile_equivalente")},
                           "features": linhas}

    (OUT / "comparacao.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    todas = [r for v in relatorio.values() for r in v["features"]]
    pd.DataFrame(todas).to_csv(OUT / "comparacao_por_feature.csv", index=False)
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
