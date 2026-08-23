"""
TER-02 -- Re-extrai as features nos pontos ja amostrados e compara as cadeias.

O QUE ESTE SCRIPT PRODUZ, e por que nesta ordem:

  1. amostragem  para cada ponto do dataset que caia numa AOI com derivacao
                 harmonizada (ter01), le hand/twi/slope/elev dos rasters
                 WhiteboxTools e grava ao lado dos valores atuais.
  2. comparacao  por AOI e no agregado: pearson, mediana, contraste
                 negativo-positivo, nos dois instrumentos. E o equivalente,
                 no nivel do ponto, do compare_*.json do recife_validation.
  3. dataset     `dataset_serra_harmonizado.csv`, com as colunas de feature
                 substituidas pelas harmonizadas e as antigas preservadas com
                 sufixo `_global`. Nada e sobrescrito no dataset original.

POR QUE COMPARAR NO PONTO E NAO SO NO RASTER:
o raster inteiro inclui encosta alta, topo e area sem interesse. O modelo so
ve os pontos amostrados. Uma divergencia grande no raster pode ser irrelevante
se ocorrer onde nao ha ponto -- e uma divergencia pequena no raster pode ser
decisiva se estiver concentrada no fundo de vale, que e onde estao os
positivos. A comparacao que importa e na amostra.

O QUE JA SE SABE, medido em 12/08/2026 na EMSR789_AOI06:

    HAND                      mediana   positivo  negativo  contraste
    percentil 98 (errado)       70,69      16,97    155,95   +138,98
    area 0,1123 km2 (correto)   38,99      13,34     84,54    +71,20
    produto global (atual)      36,36       9,25     72,97    +63,72

    pearson contra o global:  0,953 com area fixa / 0,815 com percentil

O limiar em area contribuinte trouxe a cadeia propria para 0,953 de acordo com
o produto global -- duas implementacoes independentes chegando ao mesmo lugar.
Isso NAO significa que as duas sejam intercambiaveis: significa que a cadeia
propria esta correta e pode agora ser aplicada onde o produto global nao serve
(resolucao fina, vale confinado, e Petropolis, que ja tem derivacao WBT).

NAO faz: nao treina, nao altera o dataset original, nao escolhe qual cadeia e
"melhor" por AUC -- a escolha e por comparabilidade com Petropolis.

POR QUE EXISTE `--todas`, adotado em 13/08/2026:

ate aqui este script so olhava as 22 AOIs INGREMES, porque a pergunta era a
analogia com Petropolis. A consequencia passou despercebida: as 97 AOIs de
planicie ficavam derivadas pelo ter01 e invisiveis aqui, entao o modelo
fluvial multirregiao rodou com 5.834 pontos -- CEMS ingreme mais Curitiba -- e
o resto do CEMS continuou em cadeia global, isto e, FORA do pool.

Isso importa para uma pergunta que esta em aberto: o leave-one-source-out
mostrou que o modelo treinado so nas 21 AOIs tropicais aplicado a Curitiba fica
em 0,4880, nivel de acaso. Nao da para decidir entre amostra pequena e
incompatibilidade real enquanto o lado planicie do conjunto nao estiver na
mesma cadeia. Harmonizar as 97 e o que separa as duas hipoteses.

O comportamento padrao NAO muda: sem `--todas`, este script produz exatamente
os mesmos tres arquivos de antes, com os mesmos numeros. `--todas` escreve num
conjunto proprio de arquivos, com sufixo `_todas`, e acrescenta a coluna
`classe_relevo` para que serra e planicie continuem distinguiveis depois de
juntas.

Uso:
    python scripts/terreno/ter02_reextrair_e_comparar.py
    python scripts/terreno/ter02_reextrair_e_comparar.py --todas
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
DERIV = RUNS / "ter-01-cadeia-harmonizada"
DATASET = RUNS / "ds-01-multirregiao" / "dataset_multirregiao_v1.csv"
RELEVO = RUNS / "cems-02-analogos-v2" / "verificacao_relevo_por_aoi.csv"
OUT = RUNS / "ter-02-comparacao"

# coluna do dataset -> raster produzido pelo ter01
MAPA = {"hand_m": "hand_dinf", "twi_dinf": "twi_dinf",
        "slope_deg": "slope_deg_wbt", "elevation_m": "dem_filled"}


def amostrar(raster: Path, lons, lats):
    """Amostra respeitando o nodata DECLARADO, com rede de seguranca.

    Bug corrigido em 12/08/2026: a primeira versao so filtrava valores nao
    finitos. O WhiteboxTools grava nodata como -32768 e a cadeia antiga usa
    -9999 -- ambos sao numeros finitos e passariam direto para a coluna de
    feature. Um ponto que caisse fora do dominio viraria um HAND de -32768,
    que nao e detectavel por inspecao de media e destruiria o ajuste.

    No dataset atual a contaminacao foi ZERO (todos os pontos caem dentro da
    AOI, e o ter01 usa margem de 0,02 grau). A correcao e para que continue
    zero quando a AOI mudar.
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
    v[v < -1000] = np.nan      # rede de seguranca: nodata nao declarado
    return v


def main() -> int:
    todas = "--todas" in sys.argv[1:]
    sufixo = "_todas" if todas else ""
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"===TER02_REEXTRAIR=== escopo={'TODAS_AS_AOIS' if todas else 'INGREMES'}")

    for p in (DATASET, RELEVO, DERIV):
        if not p.exists():
            print(f"ABORTADO: {p} ausente.")
            return 1

    df = pd.read_csv(DATASET, low_memory=False)
    rel = pd.read_csv(RELEVO)
    ingremes = set(rel.loc[rel.analogo_de_petropolis, "aoi"].astype(str))
    relevo_de = dict(zip(rel["aoi"].astype(str), rel["classe_relevo"].astype(str)))
    alvo = set(relevo_de) if todas else ingremes

    derivadas = {d.name for d in DERIV.iterdir()
                 if d.is_dir() and (d / "run_manifest.json").exists()}
    usaveis = sorted(alvo & derivadas)
    faltando = sorted(alvo - derivadas)
    print(f"AOIS_ALVO={len(alvo)} DERIVADAS={len(usaveis)} FALTANDO={len(faltando)}")
    for f in faltando:
        print(f"   SEM_DERIVACAO {f}")
    if not usaveis:
        print("ABORTADO: nenhuma AOI alvo com derivacao. Rode o ter01 antes.")
        return 1

    # Sob --todas re-extrai TODAS as classes, nao so 0 e 1. Agua permanente e
    # movimento de massa nao entram em treino, mas ficavam em cadeia global sem
    # que isso fosse decisao -- e numa base de conhecimento de resolucao unica,
    # classe de auditoria em outro instrumento e uma inconsistencia gratuita.
    # O contraste positivo-negativo continua sendo calculado so em 0 e 1.
    classes = df.classe.isin([0, 1, 2, 9]) if todas else df.classe.isin([0, 1])
    base = df[df.grupo_cv.isin(usaveis) & classes].copy()
    print(f"PONTOS={len(base):,} em {base.grupo_cv.nunique()} AOIs "
          f"(classes {sorted(base.classe.unique())})")

    for col in MAPA:
        base[f"{col}_global"] = base[col]
        base[f"{col}_wbt"] = np.nan

    linhas_cmp = []
    for aoi in usaveis:
        d = DERIV / aoi
        m = base.grupo_cv == aoi
        s = base.loc[m]
        for col, camada in MAPA.items():
            r = d / f"{camada}.tif"
            if not r.exists():
                continue
            base.loc[m, f"{col}_wbt"] = amostrar(r, s.lon, s.lat)

        for col in MAPA:
            a = base.loc[m, f"{col}_global"].to_numpy(dtype="float64")
            b = base.loc[m, f"{col}_wbt"].to_numpy(dtype="float64")
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 5:
                continue
            cl = base.loc[m, "classe"].to_numpy()
            kp, kn = (cl == 1) & ok, (cl == 0) & ok
            linhas_cmp.append({
                "aoi": aoi, "feature": col, "n": int(ok.sum()),
                "pearson": round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 4),
                "mediana_global": round(float(np.median(a[ok])), 3),
                "mediana_wbt": round(float(np.median(b[ok])), 3),
                "contraste_global": (round(float(np.median(a[kn]) - np.median(a[kp])), 3)
                                     if kp.any() and kn.any() else None),
                "contraste_wbt": (round(float(np.median(b[kn]) - np.median(b[kp])), 3)
                                  if kp.any() and kn.any() else None),
            })

    cmp = pd.DataFrame(linhas_cmp)
    if not cmp.empty:
        cmp["classe_relevo"] = cmp["aoi"].map(relevo_de).fillna("NAO_CLASSIFICADO")
    cmp.to_csv(OUT / f"comparacao_por_aoi{sufixo}.csv", index=False)

    print("\n--- AGREGADO POR FEATURE (mediana das AOIs) ---")
    print(f"{'feature':12s} {'pearson':>8s} {'med_global':>11s} {'med_wbt':>9s} "
          f"{'contr_global':>13s} {'contr_wbt':>10s}")
    resumo = {}
    for col in MAPA:
        c = cmp[cmp.feature == col]
        if c.empty:
            continue
        r = {"pearson_mediano": float(c.pearson.median()),
             "mediana_global": float(c.mediana_global.median()),
             "mediana_wbt": float(c.mediana_wbt.median()),
             "contraste_global": float(c.contraste_global.median(skipna=True)),
             "contraste_wbt": float(c.contraste_wbt.median(skipna=True)),
             "n_aois": int(len(c))}
        resumo[col] = r
        print(f"{col:12s} {r['pearson_mediano']:8.3f} {r['mediana_global']:11.2f} "
              f"{r['mediana_wbt']:9.2f} {r['contraste_global']:13.2f} "
              f"{r['contraste_wbt']:10.2f}")

    # dataset harmonizado: feature = WBT, global preservado ao lado
    harm = base.copy()
    for col in MAPA:
        harm[col] = harm[f"{col}_wbt"]
    harm["classe_relevo"] = harm.grupo_cv.map(relevo_de).fillna("NAO_CLASSIFICADO")
    antes = len(harm)
    harm = harm.dropna(subset=list(MAPA))
    print(f"\nDATASET_HARMONIZADO n={len(harm):,} "
          f"(descartados {antes-len(harm):,} sem valor WBT) "
          f"grupos={harm.grupo_cv.nunique()}")
    if todas:
        print("  por classe de relevo: "
              f"{harm.classe_relevo.value_counts().to_dict()}")
    nome = ("dataset_harmonizado_todas.csv" if todas
            else "dataset_serra_harmonizado.csv")
    harm.to_csv(OUT / nome, index=False)

    (OUT / f"resumo{sufixo}.json").write_text(json.dumps({
        "escopo": "todas_as_aois" if todas else "somente_ingremes",
        "aois_alvo": len(alvo), "aois_ingremes": len(ingremes),
        "aois_derivadas": len(usaveis),
        "aois_sem_derivacao": faltando, "pontos": int(len(harm)),
        "grupos": int(harm.grupo_cv.nunique()),
        "pontos_por_classe_relevo": {k: int(v) for k, v in
                                     harm.classe_relevo.value_counts().items()},
        "por_feature": resumo,
        "nota": ("comparacao no NIVEL DO PONTO amostrado, nao do raster inteiro: "
                 "o modelo so ve os pontos"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"GRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
