"""
TER-06 -- Traz o Sen1Floods11 e o UFO para a cadeia unica de 30 m.

POR QUE ESTAS DUAS FONTES FICARAM DE FORA ATE AGORA:

o `ds01` declarou, e a declaracao continua correta para a epoca: elas entravam
com as colunas de feature vazias de proposito, porque preencher exigiria
adquirir raster para 661 locais espalhados em seis continentes -- "custo alto
para um uso que ainda nao foi decidido".

O uso foi decidido. A base de conhecimento passa a ter UMA cadeia e UMA
resolucao, 30 m, em todo o projeto. Com isso o custo deixa de ser especulativo
e vira uma conta: 661 chips de 3 a 5 km, 82 tiles DEM distintos, porque os
chips se agrupam geograficamente muito mais do que a dispersao continental
sugere.

O QUE ELAS GANHAM:
hoje tem `elevation_m` e `hand_m` vindos do PRODUTO GLOBAL, e `slope_deg` e
`twi_dinf` vazios -- derivadas direcionais exigem grade local reprojetada. A
cadeia propria entrega as quatro na mesma definicao de canal (0,1123 km2) que
todas as outras regioes.

O LIMITE QUE ESTA ESCALA IMPOE, e como ele e tratado:

HAND e a altura acima do canal mais proximo. Numa janela de ~9 km, o canal mais
proximo de verdade pode estar FORA da janela, e ai a cadeia encontra um canal
que nao e o relevante e devolve um HAND menor do que o real. Isso nao e
hipotetico em planicie de inundacao, que e exatamente onde estas fontes
amostram.

O `ter01` ja tem a guarda: rede com menos de 50 celulas de canal e reportada
como REDE DEGENERADA. Este script LE essa condicao e marca o chip, e os pontos
de chip degenerado saem com HAND nulo declarado em vez de um numero plausivel e
errado. Preferir o nulo aqui e o mesmo criterio que o projeto usa em todo o
resto: ausencia declarada nunca vira estimativa.

O QUE NAO MUDA: o `grupo_cv` continua sendo o que era -- pais no Sen1Floods11,
chip no UFO. A granularidade da derivacao de terreno e uma coisa; a unidade de
validacao cruzada e outra, e trocar a segunda seria afrouxar o bloqueio
espacial de 11 grupos para 446. Decisao separada, nao tomada aqui.

NAO faz: nao treina, nao promove rotulo, nao altera a fonte de origem, e nao
reclassifica o UFO -- ele continua MISTO_NAO_SEPARAVEL e fora dos modelos por
mecanismo.

Uso:
    python scripts/terreno/ter06_harmonizar_chips_nivel1.py
    python scripts/terreno/ter06_harmonizar_chips_nivel1.py --teto 180
    python scripts/terreno/ter06_harmonizar_chips_nivel1.py --so-amostrar
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
TER01 = REPO / "scripts" / "terreno" / "ter01_cadeia_harmonizada.py"
DERIV = RUNS / "ter-01-cadeia-harmonizada"
PONTOS = RUNS / "n1g-pontos-com-features" / "nivel1_pontos_com_features_v1.csv"
OUT = RUNS / "ter-02-comparacao"          # mesmo diretorio: o ds04 le de la

MAPA = {"hand_m": "hand_dinf", "twi_dinf": "twi_dinf",
        "slope_deg": "slope_deg_wbt", "elevation_m": "dem_filled"}

# abaixo disto a rede de drenagem da janela nao sustenta HAND. O ter01 usa o
# mesmo numero para emitir REDE DEGENERADA.
CANAL_MINIMO_CELULAS = 50

PREFIXO = "n1chip_"


def rotulo(chip: str) -> str:
    """Nome de pasta seguro: chip do UFO tem sufixos com ponto e hifen."""
    limpo = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(chip))
    return f"{PREFIXO}{limpo}"


def bboxes(d: pd.DataFrame) -> dict[str, tuple]:
    """Um bbox por chip, a partir dos proprios pontos amostrados."""
    g = d.groupby("chip").agg(minx=("lon", "min"), maxx=("lon", "max"),
                              miny=("lat", "min"), maxy=("lat", "max"))
    return {str(c): (r.minx, r.miny, r.maxx, r.maxy) for c, r in g.iterrows()}


def chips_incompletos(saida: Path) -> set[str]:
    """Chips onde algum ponto ficou sem alguma das quatro variaveis.

    Duas causas, e a mesma correcao serve para as duas: derivacao que falhou,
    e ponto perto da borda do recorte onde o roteamento de fluxo nao alcanca
    canal nenhum -- ai o HAND sai nodata mesmo com elevacao valida. Alargar a
    folga e correcao fisica, nao contorno: o canal existe, estava fora da
    janela.
    """
    if not saida.exists():
        return set()
    d = pd.read_csv(saida, low_memory=False)
    cols = [f"{c}_wbt" for c in MAPA]
    ruim = d[d[cols].isna().any(axis=1)]
    return set(ruim["chip"].astype(str).unique())


def derivar_lote(alvos: dict[str, tuple], teto_s: int, margem: float | None = None,
                 refazer: set[str] | None = None) -> dict:
    """Um subprocesso por chip, com teto. Mesmo padrao do lote do ter01.

    Uma janela patologica -- ilha, delta plano, recorte todo em agua -- nao
    pode custar o lote inteiro. Ja aconteceu duas vezes com as AOIs do CEMS.
    """
    refazer = refazer or set()
    ok, estourou, falhou = [], [], []
    for i, (chip, bb) in enumerate(sorted(alvos.items()), 1):
        lab = rotulo(chip)
        pasta = DERIV / lab
        man = pasta / "run_manifest.json"
        if man.exists() and chip not in refazer:
            ok.append(chip)
            continue
        if chip in refazer and pasta.exists():
            # apaga a derivacao antiga: o ter01 reaproveita o que ja existe, e
            # refazer com folga maior exige que ele recomece do DEM
            shutil.rmtree(pasta, ignore_errors=True)
        extra = ["--margem", str(margem)] if margem is not None else []
        try:
            r = subprocess.run(
                [sys.executable, "-u", str(TER01), "--regiao", lab,
                 "--bbox", ",".join(f"{v:.6f}" for v in bb), *extra],
                capture_output=True, text=True, timeout=teto_s)
            if (DERIV / lab / "run_manifest.json").exists():
                ok.append(chip)
            else:
                falhou.append(chip)
                saida = (r.stdout or "").strip().splitlines()
                print(f"  [{chip}] FALHOU: "
                      f"{saida[-1] if saida else (r.stderr or '')[:110]}", flush=True)
        except subprocess.TimeoutExpired:
            estourou.append(chip)
            print(f"  [{chip}] ESTOUROU {teto_s}s -- pulado", flush=True)
        if i % 50 == 0:
            print(f"  ... {i}/{len(alvos)} (ok={len(ok)} "
                  f"estourou={len(estourou)} falhou={len(falhou)})", flush=True)
    return {"derivados": ok, "estouraram": estourou, "falharam": falhou}


def amostrar(raster: Path, lons, lats) -> np.ndarray:
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
    args = sys.argv[1:]
    teto = int(args[args.index("--teto") + 1]) if "--teto" in args else 120
    OUT.mkdir(parents=True, exist_ok=True)
    print("===TER06_HARMONIZAR_CHIPS_NIVEL1===")

    if not PONTOS.exists():
        print(f"ABORTADO: {PONTOS} ausente. Rode n1g.")
        return 1
    d = pd.read_csv(PONTOS, low_memory=False)
    d["chip"] = d["chip"].astype(str)
    alvos = bboxes(d)
    print(f"PONTOS={len(d):,} CHIPS={len(alvos)} "
          f"fontes={sorted(d.fonte.unique())}")

    margem = float(args[args.index("--margem") + 1]) if "--margem" in args else None
    refazer: set[str] = set()
    if "--refazer-faltantes" in args:
        refazer = chips_incompletos(OUT / "dataset_harmonizado_nivel1.csv")
        print(f"--- refazendo {len(refazer)} chips com cobertura incompleta "
              f"(margem {margem or 'padrao'}) ---")

    if "--so-amostrar" not in args:
        print(f"--- derivando (teto {teto}s por chip) ---")
        lote = derivar_lote(alvos, teto, margem, refazer)
        print(f"DERIVADOS={len(lote['derivados'])}/{len(alvos)} "
              f"estouraram={len(lote['estouraram'])} "
              f"falharam={len(lote['falharam'])}")
    else:
        lote = {"derivados": [c for c in alvos
                              if (DERIV / rotulo(c) / "run_manifest.json").exists()],
                "estouraram": [], "falharam": []}
        print(f"so-amostrar: {len(lote['derivados'])} chips ja derivados")

    # ---- amostragem, chip a chip ----
    for col in MAPA:
        d[f"{col}_wbt"] = np.nan
    degenerados, sem_deriv = [], []

    for chip in sorted(alvos):
        pasta = DERIV / rotulo(chip)
        man_p = pasta / "run_manifest.json"
        if not man_p.exists():
            sem_deriv.append(chip)
            continue
        man = json.loads(man_p.read_text(encoding="utf-8"))
        m = (d.chip == chip).to_numpy()
        s = d.loc[m]

        # HAND so e confiavel se a janela tem rede de drenagem. Abaixo do
        # minimo o valor sai nulo declarado, e nao um numero plausivel.
        canais = man.get("n_celulas_canal")
        if canais is None:
            raise SystemExit(
                f"{man_p} sem 'n_celulas_canal': a guarda de rede degenerada "
                "nao pode ser avaliada, e passar HAND sem ela seria pior que "
                "abortar. Verifique a versao do ter01 que gerou este manifesto.")
        rede_degenerada = canais < CANAL_MINIMO_CELULAS
        if rede_degenerada:
            degenerados.append({"chip": chip, "canais": int(canais)})

        for col, camada in MAPA.items():
            r = pasta / f"{camada}.tif"
            if not r.exists():
                continue
            v = amostrar(r, s.lon, s.lat)
            if rede_degenerada and col == "hand_m":
                v[:] = np.nan
            d.loc[m, f"{col}_wbt"] = v

    cols_wbt = [f"{c}_wbt" for c in MAPA]
    completo = d[cols_wbt].notna().all(axis=1)
    print(f"\nCOM_AS_QUATRO={int(completo.sum()):,} de {len(d):,}")
    print(f"  chips sem derivacao: {len(sem_deriv)}")
    print(f"  chips com REDE DEGENERADA (HAND anulado): {len(degenerados)}")
    for f, s in d.groupby("fonte"):
        c = s[cols_wbt].notna().all(axis=1)
        print(f"  {f:34s} {int(c.sum()):>6,} / {len(s):>6,}")

    # ---- comparacao contra o produto global, onde havia valor ----
    linhas = []
    for col in ("elevation_m", "hand_m"):
        a = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype="float64")
        b = d[f"{col}_wbt"].to_numpy(dtype="float64")
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 5:
            continue
        linhas.append({"feature": col, "n": int(ok.sum()),
                       "pearson": round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 4),
                       "mediana_global": round(float(np.median(a[ok])), 3),
                       "mediana_wbt": round(float(np.median(b[ok])), 3)})
        print(f"  {col:14s} pearson={linhas[-1]['pearson']:6.3f}  "
              f"med {linhas[-1]['mediana_global']:8.2f} -> "
              f"{linhas[-1]['mediana_wbt']:8.2f}")
    print("  slope_deg e twi_dinf: nao havia valor anterior para comparar "
          "(estavam vazios por decisao do ds01)")

    saida = OUT / "dataset_harmonizado_nivel1.csv"
    d[["ponto_id", "chip", "fonte", "classe_obs", *cols_wbt]].to_csv(
        saida, index=False)
    (OUT / "resumo_nivel1.json").write_text(json.dumps({
        "pontos": int(len(d)), "chips": len(alvos),
        "com_as_quatro": int(completo.sum()),
        "chips_sem_derivacao": sem_deriv,
        "chips_rede_degenerada": degenerados,
        "canal_minimo_celulas": CANAL_MINIMO_CELULAS,
        "por_fonte": {f: {"n": int(len(s)),
                          "com_as_quatro": int(s[cols_wbt].notna().all(axis=1).sum())}
                      for f, s in d.groupby("fonte")},
        "comparacao": linhas,
        "lote": {k: len(v) for k, v in lote.items()},
        "limite_declarado": (
            "HAND numa janela de ~9 km pode encontrar um canal que nao e o "
            "relevante se o canal real estiver fora do recorte. Chip com rede "
            "abaixo de 50 celulas sai com HAND nulo; os demais carregam o "
            "limite sem que ele seja mensuravel ponto a ponto"),
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"GRAVADO={saida}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
