"""Pontos, terreno e linhas no formato do ds06 para os eventos GFD em quarentena.

Tres passos, os mesmos da cadeia dos eventos conhecidos, sem passar por ela:

  pontos   `gfd01.classes_do_rotulo` + `stratifiedSample` no tile escolhido
           (passo 250 m, semente do gfd01), `PONTOS_POR_CLASSE_NO_TILE` por classe
  terreno  `cadeia_em_lote` sobre os tiles (FABDEM + WhiteboxTools, mesma
           cadeia), e `gfd02._amostrar_no_tile` nos pontos
  linhas   `espelho_do_ds06`: as regras que o ds04/ds05/ds06 aplicam ao GFD --
           agua permanente sai, relevo por janela do ponto (ou do tile),
           relevo nao resolvido sai, terreno nao finito sai, elevacao local
           contra o piso (quantil 1%) do proprio tile com >= 30 pontos

O ESPELHO E PROVADO, nao suposto: `--provar-espelho` aplica-o aos pontos GFD
conhecidos e compara com o ds06 vigente (mesmos pontos admitidos, mesma
classe de relevo, mesmas variaveis). Sem a prova passar, `--linhas` recusa.

Nada daqui entra em ds03-ds06: a saida e `gfd-quarentena/pontos_quarentena.parquet`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("servico", "geo", "suscetibilidade", "terreno", "metodo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

from proveniencia import agora_utc  # noqa: E402

Q = REPO / "local_runs" / "gfd-quarentena"
TER = REPO / "local_runs" / "ter-01-cadeia-harmonizada"
PONTOS_POR_CLASSE_NO_TILE = 1300   # ~8000 por evento no gfd01, dividido em ate 3 tiles


def _rotulo(lat: int, lon: int) -> str:
    from cadeia_em_lote import _rotulo_do_tile
    return _rotulo_do_tile(lat, lon)


def espelho_do_ds06(p: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """As regras do ds04/ds05/ds06 para o GFD, sobre a saida do gfd02."""
    from ds04_reduzir_por_fonte import FRAC_GT15_MIN, RELEVO_LOCAL_MIN_M, relevo_dos_pontos
    from ds06_dataset_de_treino import MINIMO_PONTOS_NO_PISO, QUANTIL_DO_PISO
    d = p.copy()
    d["classe"] = (d.classe_obs == "INUNDACAO").astype(int)
    d["classe_relevo"] = "NAO_CLASSIFICADO"
    for tile, sub in d.groupby("grupo_cv"):
        d.loc[d.grupo_cv == tile, "classe_relevo"] = relevo_dos_pontos(sub.elevation_m, sub.slope_deg)[0]
    medida = d.relevo_local_janela_m.notna() & d.frac_gt15_janela.notna()
    ingreme = (d.relevo_local_janela_m >= RELEVO_LOCAL_MIN_M) & (d.frac_gt15_janela >= FRAC_GT15_MIN)
    d.loc[medida, "classe_relevo"] = np.where(ingreme[medida], "INGREME", "PLANO_OU_ONDULADO")
    saiu = {"papel:agua_permanente": int((d.agua_permanente_jrc == 1).sum())}
    d = d[d.agua_permanente_jrc != 1]
    finito = np.isfinite(d[["hand_m", "twi_dinf", "slope_deg", "elevation_m"]].to_numpy(float)).all(axis=1)
    saiu["terreno_nao_finito"] = int((~finito).sum())
    d = d[finito]
    import contrato_do_fenomeno as cf
    resolvido = np.array([cf.relevo_resolvido("gfd", r) for r in d.classe_relevo])
    saiu["relevo_nao_resolvido_pelo_instrumento"] = int((~resolvido).sum())
    d = d[resolvido].copy()
    g = d.groupby("grupo_cv").elevation_m
    piso = g.transform(lambda s: s.quantile(QUANTIL_DO_PISO))
    d["elevation_local_m"] = (d.elevation_m - piso).where(g.transform("size") >= MINIMO_PONTOS_NO_PISO)
    d["evento_id"] = "GFD:" + d.evento_gfd.astype(int).astype(str)
    d["grupo_validacao"] = d.grupo_cv
    d["fonte"] = "gfd"
    d["mecanismo"] = "FLUVIAL_ENXURRADA"
    return d.reset_index(drop=True), saiu


def provar_espelho() -> dict:
    """Aplica o espelho aos pontos GFD conhecidos e compara com o ds06 vigente."""
    from ds06_dataset_de_treino import versao_atual
    p = pd.read_csv(REPO / "local_runs" / "gfd-brasil" / "pontos_gfd_com_terreno.csv").drop_duplicates("ponto_id")
    e, saiu = espelho_do_ds06(p)
    v = pd.read_parquet(versao_atual() / "pontos.parquet",
                        columns=["ponto_id", "fonte", "classe", "classe_relevo", "hand_m", "twi_dinf", "slope_deg",
                                 "elevation_local_m", "evento_id", "grupo_validacao"])
    v = v[v.fonte == "gfd"]
    so_espelho, so_ds06 = set(e.ponto_id) - set(v.ponto_id), set(v.ponto_id) - set(e.ponto_id)
    m = e.merge(v, on="ponto_id", suffixes=("_e", "_v"))
    dif = {c: float(np.nanmax(np.abs(m[f"{c}_e"].astype(float) - m[f"{c}_v"].astype(float))))
           for c in ("hand_m", "twi_dinf", "slope_deg", "elevation_local_m")}
    r = {"pontos_espelho": len(e), "pontos_ds06": len(v), "so_no_espelho": len(so_espelho), "so_no_ds06": len(so_ds06),
         "classe_igual": float((m.classe_e == m.classe_v).mean()),
         "relevo_igual": float((m.classe_relevo_e == m.classe_relevo_v).mean()),
         "evento_igual": float((m.evento_id_e == m.evento_id_v).mean()),
         "grupo_igual": float((m.grupo_validacao_e == m.grupo_validacao_v).mean()),
         "diferenca_maxima": dif, "saiu_no_espelho": saiu}
    r["passou"] = (r["so_no_espelho"] == 0 and r["so_no_ds06"] == 0 and min(r["classe_igual"], r["relevo_igual"],
                   r["evento_igual"], r["grupo_igual"]) == 1.0 and all(x < 1e-6 for x in dif.values()))
    (Q / "prova_do_espelho.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def amostrar_pontos() -> int:
    import csv
    import io
    import urllib.request
    import ee
    from gfd01_eventos_do_brasil import COLECAO, FAIXA_M, PASSO_M, classes_do_rotulo
    ee.Initialize()
    sel = json.loads((Q / "selecao.json").read_text(encoding="utf-8"))
    saida = Q / "pontos"
    saida.mkdir(parents=True, exist_ok=True)
    for ev in sel["eventos"]:
        img = ee.Image(ee.ImageCollection(COLECAO).filter(ee.Filter.eq("id", ev["id"])).first())
        esc = img.select("flooded").projection().nominalScale().getInfo()
        pos, neg = classes_do_rotulo(img, esc, FAIXA_M)
        rot = ee.Image(0).where(neg, 1).where(pos, 2).rename("classe_gfd").selfMask()
        for t in ev["tiles"]:
            alvo = saida / f"pontos_gfd{ev['id']}_{t['tile']}.csv"
            if alvo.exists():
                continue
            geo = ee.Geometry.Rectangle(t["tile_1grau"])
            am = rot.addBands(ee.Image.pixelLonLat()).stratifiedSample(
                numPoints=PONTOS_POR_CLASSE_NO_TILE, classBand="classe_gfd", region=geo,
                scale=max(PASSO_M, esc), projection=img.select("flooded").projection(),
                seed=20260911, geometries=False, tileScale=4)
            url = am.getDownloadURL(filetype="CSV", selectors=["classe_gfd", "latitude", "longitude"])
            with urllib.request.urlopen(url, timeout=900) as r:
                linhas = list(csv.DictReader(io.StringIO(r.read().decode("utf-8", errors="replace"))))
            la, lo = map(int, t["tile"].split("_"))
            pts = [{"ponto_id": f"GFDQ{ev['id']}_{t['tile']}_{i:05d}",
                    "classe_obs": "INUNDACAO" if int(float(x["classe_gfd"])) == 2 else "NAO_INUNDADO_OBSERVADO",
                    "lat": float(x["latitude"]), "lon": float(x["longitude"]), "evento_gfd": ev["id"],
                    "grupo_cv": "GFD_" + _rotulo(la, lo), "tile": "gfd_tile_" + _rotulo(la, lo)}
                   for i, x in enumerate(linhas) if x.get("latitude")]
            pd.DataFrame(pts).to_csv(alvo, index=False)
            print(f"  GFD:{ev['id']} {t['tile']}: {len(pts)} pontos", flush=True)
    return 0


def janelas() -> Path:
    sel = json.loads((Q / "selecao.json").read_text(encoding="utf-8"))
    js = []
    for ev in sel["eventos"]:
        for t in ev["tiles"]:
            la, lo = map(int, t["tile"].split("_"))
            js.append({"nome": "gfd_tile_" + _rotulo(la, lo), "tile_1grau": t["tile_1grau"]})
    alvo = Q / "janelas.json"
    alvo.write_text(json.dumps(js, indent=1), encoding="utf-8")
    return alvo


def linhas() -> dict:
    from gfd02_amostrar_terreno import _amostrar_no_tile
    prova = json.loads((Q / "prova_do_espelho.json").read_text(encoding="utf-8")) if (Q / "prova_do_espelho.json").exists() else {}
    if not prova.get("passou"):
        raise SystemExit("o espelho do ds06 nao foi provado contra os pontos conhecidos: rode --provar-espelho")
    p = pd.concat([pd.read_csv(f) for f in sorted((Q / "pontos").glob("pontos_gfd*.csv"))], ignore_index=True)
    partes, sem_terreno = [], []
    for tile, sub in p.groupby("tile"):
        if not (TER / tile / "dem_filled.tif").exists():
            sem_terreno.append(tile)
            continue
        s = _amostrar_no_tile(sub, TER / tile)
        partes.append(sub.reset_index(drop=True).assign(**{k: v for k, v in s.items()}))
    bruto = pd.concat(partes, ignore_index=True)
    e, saiu = espelho_do_ds06(bruto)
    e.to_parquet(Q / "pontos_quarentena.parquet", index=False)
    r = {"tipo": "pontos_quarentena", "quando_utc": agora_utc(), "pontos_amostrados": int(len(p)),
         "tiles_sem_terreno": sem_terreno, "linhas": int(len(e)), "saiu": saiu,
         "por_relevo": e.classe_relevo.value_counts().to_dict(), "eventos": int(e.evento_id.nunique()),
         "nao_e": "nao entra em ds03-ds06 antes do gfdq03; sem rotulo visto por nenhum modelo"}
    (Q / "linhas_manifesto.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provar-espelho", action="store_true")
    ap.add_argument("--pontos", action="store_true")
    ap.add_argument("--janelas", action="store_true")
    ap.add_argument("--linhas", action="store_true")
    a = ap.parse_args()
    if a.provar_espelho:
        print(json.dumps(provar_espelho(), ensure_ascii=False, indent=1))
    if a.pontos:
        amostrar_pontos()
    if a.janelas:
        print(janelas())
    if a.linhas:
        print(json.dumps(linhas(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
