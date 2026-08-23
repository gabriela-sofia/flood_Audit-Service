"""
NIVEL 1 / F -- Reduz Sen1Floods11 e UFO a tabelas de pontos.

Implementa o `docs/metodologia_cientifica/ext_contrato_reducao_a_pontos_v1.md`.
Depois desta etapa nenhum chip precisa ser reaberto pelo pipeline (regra R4):
dezenas de milhares de imagens viram uma tabela de alguns megabytes.

O QUE TORNA ESTAS DUAS FONTES DIFERENTES DE TUDO QUE O PROJETO TINHA:
o rotulo delas afirma NAO-INUNDACAO, em vez de apenas nao afirmar inundacao.

  Sen1Floods11  LabelHand tem TRES estados por pixel:
                   1 = agua,  0 = nao-agua,  -1 = sem dado
                O `0` e observacao de pixel seco; o `-1` e ausencia de
                observacao. Colapsar os dois destruiria o motivo de usar a
                base (regra R1 do contrato).

  UFO           mascara `inundated` / `non-inundated` em contexto URBANO a
                3 m -- a unica das quatro fontes desenhada para o fenomeno
                que o REV-P estuda.

REGRA R3 -- agua permanente sai antes.
Rio e lago nao sao inundacao. O Sen1Floods11 distribui `JRCWaterHand` com a
agua permanente do JRC justamente para isso; usar essa camada nao e opcional.
Pixel marcado como agua no rotulo E agua permanente no JRC vira
`AGUA_PERMANENTE`, nao `INUNDADO`.

REGRA R2 -- resolucao nativa fica registrada por linha, porque 10 m do
Sen1Floods11 e 3 m do UFO nao significam a mesma coisa.

AMOSTRAGEM: nao se guarda pixel a pixel. 446 chips de 512x512 dariam 117
milhoes de linhas para representar 11 eventos -- pseudorreplicacao em estado
puro. Amostra-se um teto por chip, estratificado por classe, e o `evento_id`
acompanha cada linha para que a agregacao por evento continue possivel.

NAO faz: nao promove negativo, nao extrai variavel, nao treina.

Uso:
    python scripts/externo/n1f_reduzir_a_pontos.py
"""

from __future__ import annotations

import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
S1F = RUNS / "n1a-sen1floods11" / "v1.1" / "data" / "flood_events" / "HandLabeled"
UFO_ZIP = RUNS / "n1b-ufo" / "ufo-dataset.zip"
OUT = RUNS / "n1f-pontos"

POR_CHIP = 120          # teto de pontos por chip, estratificado por classe
SEMENTE = 20260807

COLUNAS = ["ponto_id", "lat", "lon", "classe_obs", "data_evento", "fonte",
           "evento_id", "resolucao_m", "confianca", "licenca"]


def amostrar_chip(rotulo: np.ndarray, transform, crs, rng, permanente=None):
    """Sorteia pixels com teto por classe e devolve (linha, coluna, classe)."""
    mapa = {1: "INUNDADO", 0: "NAO_INUNDADO_OBSERVADO", -1: "SEM_DADO"}
    saidas = []
    for valor, nome in mapa.items():
        sel = np.nonzero(rotulo == valor)
        n = len(sel[0])
        if n == 0:
            continue
        cota = max(1, POR_CHIP // 3)
        idx = rng.choice(n, size=min(cota, n), replace=False)
        r, c = sel[0][idx], sel[1][idx]
        classes = np.full(len(r), nome, dtype=object)
        if valor == 1 and permanente is not None:
            # R3: agua que ja era agua nao e inundacao
            eh_perm = permanente[r, c] > 0
            classes[eh_perm] = "AGUA_PERMANENTE"
        saidas.append((r, c, classes))
    return saidas


def reduzir_sen1floods11(rng) -> pd.DataFrame:
    import rasterio
    from rasterio.warp import transform as warp_transform

    rotulos = sorted((S1F / "LabelHand").glob("*.tif"))
    print(f"[sen1floods11] chips={len(rotulos)}")
    if not rotulos:
        return pd.DataFrame(columns=COLUNAS)

    linhas = []
    faltou_jrc = 0
    for i, caminho in enumerate(rotulos):
        with rasterio.open(caminho) as src:
            rot = src.read(1)
            tr, crs = src.transform, src.crs

        jrc_path = S1F / "JRCWaterHand" / caminho.name.replace("LabelHand", "JRCWaterHand")
        perm = None
        if jrc_path.exists():
            with rasterio.open(jrc_path) as s2:
                perm = s2.read(1)
        else:
            faltou_jrc += 1

        # o nome do chip e "<Regiao>_<id>_LabelHand.tif": a regiao e o evento
        partes = caminho.stem.split("_")
        evento = partes[0]

        for r, c, classes in amostrar_chip(rot, tr, crs, rng, perm):
            xs, ys = rasterio.transform.xy(tr, r, c)
            lon, lat = warp_transform(crs, "EPSG:4326", np.atleast_1d(xs),
                                      np.atleast_1d(ys))
            linhas.append(pd.DataFrame({
                "lat": lat, "lon": lon, "classe_obs": classes,
                "evento_id": evento, "chip": caminho.stem,
            }))
        if (i + 1) % 150 == 0:
            print(f"   {i+1}/{len(rotulos)} chips", flush=True)

    df = pd.concat(linhas, ignore_index=True)
    df["data_evento"] = pd.NaT   # o chip nao carrega a data; o evento sim, por nome
    df["fonte"] = "sen1floods11_handlabeled"
    df["resolucao_m"] = 10
    df["confianca"] = "manual"
    df["licenca"] = "CC BY 4.0"
    if faltou_jrc:
        print(f"   AVISO {faltou_jrc} chips sem JRCWaterHand -- R3 nao aplicada neles")
    return df


def reduzir_ufo(rng) -> pd.DataFrame:
    if not UFO_ZIP.exists():
        print("[ufo] zip ausente")
        return pd.DataFrame(columns=COLUNAS)

    import rasterio

    z = zipfile.ZipFile(UFO_ZIP)
    nomes = z.namelist()
    rotulos = [n for n in nomes
               if n.lower().endswith((".tif", ".tiff"))
               and any(k in n.lower() for k in ("rotulo", "mask", "annot"))]
    print(f"[ufo] arquivos={len(nomes)} candidatos_a_rotulo={len(rotulos)}")
    if not rotulos:
        amostra = [n for n in nomes if n.lower().endswith((".tif", ".tiff"))][:5]
        print(f"   estrutura nao reconhecida; exemplos: {amostra}")
        return pd.DataFrame(columns=COLUNAS)

    linhas = []
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        for i, nome in enumerate(rotulos):
            destino = Path(td) / Path(nome).name
            destino.write_bytes(z.read(nome))
            try:
                with rasterio.open(destino) as src:
                    rot = src.read(1)
                    tr, crs = src.transform, src.crs
            except Exception:  # noqa: BLE001
                continue
            # UFO: 1 = inundado, 0 = nao inundado
            for valor, classe in ((1, "INUNDADO"), (0, "NAO_INUNDADO_OBSERVADO")):
                sel = np.nonzero(rot == valor)
                if len(sel[0]) == 0:
                    continue
                idx = rng.choice(len(sel[0]),
                                 size=min(POR_CHIP // 2, len(sel[0])), replace=False)
                r, c = sel[0][idx], sel[1][idx]
                xs, ys = rasterio.transform.xy(tr, r, c)
                from rasterio.warp import transform as warp_transform
                lon, lat = warp_transform(crs, "EPSG:4326",
                                          np.atleast_1d(xs), np.atleast_1d(ys))
                linhas.append(pd.DataFrame({
                    "lat": lat, "lon": lon, "classe_obs": classe,
                    "evento_id": Path(nome).stem, "chip": Path(nome).stem,
                }))
            if (i + 1) % 50 == 0:
                print(f"   {i+1}/{len(rotulos)}", flush=True)

    if not linhas:
        return pd.DataFrame(columns=COLUNAS)
    df = pd.concat(linhas, ignore_index=True)
    df["data_evento"] = pd.NaT
    df["fonte"] = "ufo_urban_flood_observations"
    df["resolucao_m"] = 3
    df["confianca"] = "manual"
    df["licenca"] = "CC BY 4.0"
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()
    print("===N1F_REDUCAO_A_PONTOS===")

    partes = []
    for fn in (reduzir_sen1floods11, reduzir_ufo):
        try:
            d = fn(rng)
            if len(d):
                partes.append(d)
        except Exception as exc:  # noqa: BLE001
            print(f"   FALHA {fn.__name__}: {type(exc).__name__}: {str(exc)[:120]}")

    if not partes:
        print("NENHUMA FONTE REDUZIDA")
        print("===END===")
        return 1

    df = pd.concat(partes, ignore_index=True)
    df["ponto_id"] = [f"N1_{i:07d}" for i in range(len(df))]
    df = df[COLUNAS + ["chip"]]
    saida = OUT / "nivel1_pontos_v1.csv"
    df.to_csv(saida, index=False)

    print("---RESUMO---")
    print(f"N_TOTAL={len(df):,}")
    for fonte, sub in df.groupby("fonte"):
        cont = Counter(sub["classe_obs"])
        print(f"[{fonte}] n={len(sub):,} eventos={sub.evento_id.nunique()} "
              f"chips={sub.chip.nunique()} res={sub.resolucao_m.iloc[0]}m")
        for k, v in cont.most_common():
            print(f"    {k}: {v:,} ({100*v/len(sub):.1f}%)")

    neg = int((df.classe_obs == "NAO_INUNDADO_OBSERVADO").sum())
    pos = int((df.classe_obs == "INUNDADO").sum())
    print(f"NEGATIVO_POR_OBSERVACAO={neg:,}  INUNDADO={pos:,}  "
          f"razao={neg/max(pos,1):.2f}")

    (OUT / "resumo.json").write_text(json.dumps({
        "n_total": int(len(df)),
        "por_fonte": {f: int(len(s)) for f, s in df.groupby("fonte")},
        "por_classe": {k: int(v) for k, v in Counter(df.classe_obs).items()},
        "por_chip_teto": POR_CHIP,
        "semente": SEMENTE,
        "contrato": "docs/metodologia_cientifica/ext_contrato_reducao_a_pontos_v1.md",
        "aviso": ("Pontos CANDIDATOS. `NAO_INUNDADO_OBSERVADO` nao foi promovido "
                  "a negativo do dataset SUSC. `SEM_DADO` e classe propria e nao "
                  "pode ser confundida com negativo."),
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SAIDA={saida.name} TEMPO={round(time.time()-t0,1)}s")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
