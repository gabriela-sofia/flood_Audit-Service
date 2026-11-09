"""Dominio aquatico no acervo: terra, mar, agua permanente, recorrente e desconhecido, por referencias declaradas.

-------------------------------------------------------------------------
A LACUNA
-------------------------------------------------------------------------

Medido em 09/10/2026: a 30 km da costa de Recife o `dem_filled` do ter-01 tem
6,9 m e HAND 0 -- mar aparece como planicie suscetivel. Os tiles GFD nao tem
mascara de agua (so 832 das 1.649 unidades tem `agua_permanente_jrc.tif`), e o
JRC sozinho nao serve de mascara de oceano: o produto mascara onde o Landsat nao
foi adquirido (oceano aberto) e o guia diz 255 = sem dado.

-------------------------------------------------------------------------
AS REFERENCIAS (documentacao oficial conferida em 09/10/2026)
-------------------------------------------------------------------------

  terra-mar   Copernicus DEM GLO-30, camada auxiliar Water Body Mask (WBM),
              release 2021 no bucket publico da AWS: 0 sem agua, 1 oceano,
              2 lago, 3 rio (Product Handbook v5.0, tabela 8). E a MESMA linhagem
              do FABDEM do ter-01 (GLO-30 corrigido), adquirida pelo TanDEM-X em
              2011-2015. Oceano aberto nao tem tile (readme do bucket): tile
              inexistente vira `mar_aberto_sem_tile`, inferencia declarada.
  permanencia JRC Global Surface Water v1.4 (1984-2021), `occurrence` 0-100 com
              255 = sem dado; o projeto ja usa v1.4 e o corte 50 (calibrado contra
              o JRCWaterHand). `seasonality` (meses com agua em 2021, 0-12) entra
              como atributo, nao como classe: e um ano so.
  versao      v1.5 (1984-2024) existe; nao e usada para manter o corte 50 e os
              rotulos historicos comparaveis. Trocar e decisao de versao.

-------------------------------------------------------------------------
A REGRA (fixada antes do piloto; mudar exige nova versao)
-------------------------------------------------------------------------

  WBM sem tile                         -> mar_aberto_sem_tile      (fora do dominio)
  WBM 1                                -> mar                      (fora)
  WBM 2/3 e JRC >= 50                  -> agua_permanente          (fora)
  WBM 2/3 e JRC < 50                   -> agua_editada_nao_permanente (ambiguo; dentro com ressalva)
  WBM 0 e JRC >= 50                    -> agua_permanente          (fora)
  WBM 0 e 10 <= JRC < 50               -> agua_recorrente          (DENTRO: varzea e suscetivel)
  WBM 0 e 1 <= JRC < 10                -> agua_episodica           (dentro)
  WBM 0 e JRC = 0                      -> terra_sem_agua_observada (dentro; nao e "seco garantido")
  WBM 0 e JRC = 255 / ilegivel         -> terra_agua_desconhecida  (dentro com ressalva)
  WBM ilegivel                         -> desconhecido             (nunca vira seco)

Agua recorrente NAO e erro nem exclusao (verdade_observacional.ALVOS); NoData nao
vira seco; ausencia de observacao nao vira negativo.

nao_e: nao e rotulo de inundacao, nao reescreve estados do TC01 nem do GFD.

Uso:
    python scripts/geo/dominio_aquatico.py --piloto
    python scripts/geo/dominio_aquatico.py --ponto -34.60 -8.05
    python scripts/geo/dominio_aquatico.py --amostrar <pontos.parquet>      # tabela por ponto (aquisicao leve)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
SAIDA = RUNS / "dominio-aquatico"
CACHE_WBM = SAIDA / "cache" / "wbm"
VERSAO_REGRA = "v1"
CORTE_PERMANENTE = 50          # verdade_observacional / agua_permanente_jrc: calibrado em 2026
CORTE_RECORRENTE = 10          # fixado em 06/10/2026 antes de leitura estratificada
WBM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}_DEM/AUXFILES/{n}_WBM.tif"
JRC_URL = "https://storage.googleapis.com/global-surface-water/downloads2021/{camada}/{camada}_{tile}v1_4_2021.tif"
REFERENCIAS = {
    "wbm": {"produto": "Copernicus DEM GLO-30 Water Body Mask, release 2021 (AWS Open Data)",
            "classes": {0: "sem_agua", 1: "oceano", 2: "lago", 3: "rio"},
            "fonte_classes": "Copernicus DEM Product Handbook v5.0 (29/11/2022), tabela 8",
            "periodo": "TanDEM-X 2011-2015", "resolucao": "1 arc-seg (~30 m)",
            "ausencia": "oceano aberto nao tem tile (readme do bucket)",
            "licenca": ("licenca gratuita Copernicus DEM; aviso: (c) DLR e.V. 2010-2014 and (c) Airbus Defence and "
                        "Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights "
                        "reserved; doi:10.5270/ESA-c5d3d65")},
    "jrc": {"produto": "JRC Global Surface Water v1.4 (Pekel et al. 2016, doi:10.1038/nature20584)",
            "camadas": {"occurrence": "0-100 %, 255 sem dado, 1984-2021",
                        "seasonality": "0-12 meses com agua em 2021, 255 sem dado"},
            "fonte_valores": "JRC GSW Data Users Guide (occurrence e seasonality)",
            "licenca": "gratuito, sem restricao de uso; citar Pekel et al. 2016; 'Source: EC JRC/Google'",
            "nao_e": "nao e mascara de oceano: oceano aberto nao tem aquisicao Landsat"},
}
FORA = {"mar_aberto_sem_tile", "mar", "agua_permanente"}
RESSALVA = {"agua_editada_nao_permanente", "terra_agua_desconhecida", "desconhecido"}

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def _sha_json(o) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                                                 default=str).encode("utf-8")).hexdigest()


def _agora() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def nome_copernicus(lon: float, lat: float) -> str:
    """Tile de 1 grau do GLO-30 pelo canto SUDOESTE: (-34.6, -8.05) -> ..._S09_00_W035_00."""
    la, lo = math.floor(lat), math.floor(lon)
    return (f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_"
            f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00")


def tile_jrc(lon: float, lat: float) -> str:
    """Tile de 10 graus do JRC pelo canto NOROESTE (mesma convencao de agua_permanente_jrc.tile_de)."""
    oeste, norte = math.floor(lon / 10) * 10, math.ceil(lat / 10) * 10
    return f"{abs(oeste)}{'W' if oeste < 0 else 'E'}_{abs(norte)}{'S' if norte < 0 else 'N'}"


def wbm_local(nome: str, cache: Path = CACHE_WBM) -> tuple[Path | None, dict]:
    """Tile WBM em cache local (baixa uma vez, ~80 KB). 404 fica registrado como `sem_tile`."""
    cache.mkdir(parents=True, exist_ok=True)
    arq, reg = cache / f"{nome}_WBM.tif", cache / f"{nome}_WBM.json"
    if reg.exists():
        r = json.loads(reg.read_text(encoding="utf-8"))
        return (arq if r["estado"] == "baixado" and arq.exists() else None), r
    url = WBM_URL.format(n=nome)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            dados = resp.read()
        arq.write_bytes(dados)
        r = {"estado": "baixado", "url": url, "bytes": len(dados), "sha256": _sha(arq), "quando_utc": _agora()}
    except urllib.error.HTTPError as e:
        if e.code not in (403, 404):
            raise
        r = {"estado": "sem_tile", "url": url, "http": e.code, "quando_utc": _agora(),
             "leitura": "bucket nao tem tile: oceano aberto segundo o readme"}
    reg.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    return (arq if r["estado"] == "baixado" else None), r


def _amostrar(caminho: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    import rasterio
    with rasterio.open(caminho) as r:
        v = np.array([x[0] for x in r.sample(zip(lon, lat))], dtype=float)
    return v


def classificar(wbm, occ) -> np.ndarray:
    """A regra v1, vetorizada. `wbm` = -1 sem tile, NaN ilegivel; `occ` NaN ilegivel, 255 sem dado."""
    wbm, occ = np.asarray(wbm, float), np.asarray(occ, float)
    occ_ok = np.isfinite(occ) & (occ != 255)
    agua_editada = np.isin(wbm, (2, 3))
    return np.select(
        [~np.isfinite(wbm), wbm == -1, wbm == 1,
         agua_editada & occ_ok & (occ >= CORTE_PERMANENTE), agua_editada,
         (wbm == 0) & ~occ_ok, (wbm == 0) & (occ >= CORTE_PERMANENTE),
         (wbm == 0) & (occ >= CORTE_RECORRENTE), (wbm == 0) & (occ >= 1), (wbm == 0) & (occ == 0)],
        ["desconhecido", "mar_aberto_sem_tile", "mar",
         "agua_permanente", "agua_editada_nao_permanente",
         "terra_agua_desconhecida", "agua_permanente",
         "agua_recorrente", "agua_episodica", "terra_sem_agua_observada"],
        default="desconhecido")


def dominio(estado) -> np.ndarray:
    e = np.asarray(estado, dtype=object)
    return np.where(np.isin(e, list(FORA)), "fora_do_dominio_terrestre",
                    np.where(np.isin(e, list(RESSALVA)), "dentro_com_ressalva", "dentro"))


def amostrar(lon, lat, jrc: bool = True, progresso: bool = False) -> pd.DataFrame:
    """Valores das referencias em cada ponto, agrupados por tile (WBM 1 grau; JRC por janela remota)."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    n = len(lon)
    wbm, occ, saz = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    tiles = pd.Series([nome_copernicus(a, b) for a, b in zip(lon, lat)])
    registro = {"wbm_baixados": 0, "wbm_sem_tile": 0, "jrc_falhas": 0}
    inicio = time.time()
    for k, (nome, idx) in enumerate(tiles.groupby(tiles).groups.items()):
        idx = np.asarray(idx)
        arq, reg = wbm_local(nome)
        if reg["estado"] == "sem_tile":
            wbm[idx] = -1
            registro["wbm_sem_tile"] += 1
        else:
            wbm[idx] = _amostrar(str(arq), lon[idx], lat[idx])
            registro["wbm_baixados"] += 1
        if progresso and k % 20 == 0:
            print("wbm", k, nome, len(idx), round(time.time() - inicio), "s", flush=True)
    if jrc:
        tj = pd.Series([tile_jrc(a, b) for a, b in zip(lon, lat)])
        registro["jrc_bytes_transitorios"] = 0
        for tile, idx in tj.groupby(tj).groups.items():
            idx = np.asarray(idx)
            for camada, alvo in (("occurrence", occ), ("seasonality", saz)):
                url = JRC_URL.format(camada=camada, tile=tile)
                try:
                    if len(idx) > 5000:
                        # muitos pontos: o tile inteiro uma vez e mais barato que milhares de leituras remotas;
                        # amostra e apaga (contrato de aquisicao leve: o que fica e a tabela por ponto)
                        tmp = SAIDA / "tmp" / Path(url).name
                        tmp.parent.mkdir(parents=True, exist_ok=True)
                        urllib.request.urlretrieve(url, tmp)
                        registro["jrc_bytes_transitorios"] += tmp.stat().st_size
                        alvo[idx] = _amostrar(str(tmp), lon[idx], lat[idx])
                        tmp.unlink()
                    else:
                        alvo[idx] = _amostrar("/vsicurl/" + url, lon[idx], lat[idx])
                except Exception:  # noqa: BLE001 -- leitura que falha vira desconhecido, com contagem
                    registro["jrc_falhas"] += 1
            if progresso:
                print("jrc", tile, len(idx), round(time.time() - inicio), "s", flush=True)
    estado = classificar(wbm, occ)
    out = pd.DataFrame({"wbm": wbm, "jrc_occurrence": occ, "jrc_seasonality_2021": saz,
                        "estado_aquatico": estado, "dominio_aquatico": dominio(estado),
                        "sazonal_em_2021": np.isfinite(saz) & (saz >= 1) & (saz <= 11)})
    out.attrs["registro"] = registro
    return out


# ---------------------------------------------------------------------------
# Piloto pre-definido
# ---------------------------------------------------------------------------

PILOTO = [  # (id, grupo, lon, lat, esperado pela geografia, por que) -- escrito antes de qualquer leitura
    ("costa_mar_recife", "costa", -34.600, -8.050, "mar_aberto_sem_tile|mar", "Atlantico a ~30 km da costa"),
    ("costa_praia_recife", "costa", -34.890, -8.110, "terra|mar", "faixa de praia de Boa Viagem: borda terra-mar"),
    ("costa_barra_rio_grande", "costa", -52.070, -32.180, "mar|agua_permanente", "barra da Lagoa dos Patos"),
    ("rio_amazonas_obidos", "rio", -55.520, -1.900, "agua_permanente", "canal do Amazonas em Obidos"),
    ("rio_parana_corrientes", "rio", -58.830, -27.450, "agua_permanente", "canal do Parana em Corrientes"),
    ("rio_guaiba", "rio", -51.270, -30.100, "agua_permanente", "Lago Guaiba, Porto Alegre"),
    ("lago_itaipu", "lago", -54.450, -24.900, "agua_permanente", "reservatorio de Itaipu (1982)"),
    ("lago_mirim", "lago", -52.900, -32.850, "agua_permanente", "Lagoa Mirim"),
    ("lago_patos_centro", "lago", -51.300, -31.100, "agua_permanente", "centro da Lagoa dos Patos"),
    ("sazonal_pantanal", "sazonal", -57.300, -18.600, "agua_recorrente|agua_episodica|terra",
     "planicie do Pantanal (Paraguai/Taquari)"),
    ("sazonal_varzea_santarem", "sazonal", -54.900, -2.300, "agua_recorrente|agua_episodica|agua_permanente",
     "varzea do Amazonas a jusante de Santarem"),
    ("sazonal_bananal", "sazonal", -50.300, -11.000, "agua_recorrente|agua_episodica|terra",
     "planicie do Araguaia na Ilha do Bananal"),
    ("terra_planalto_brasilia", "terra", -47.950, -15.700, "terra", "planalto central"),
    ("terra_mato_grosso", "terra", -55.500, -13.000, "terra", "chapada agricola"),
    ("terra_serra_petropolis", "terra", -43.180, -22.500, "terra", "serra do Mar, Petropolis"),
]


def pre_registrar_piloto(destino: Path = SAIDA / "piloto") -> dict:
    destino.mkdir(parents=True, exist_ok=True)
    arq = destino / "pre_registro.json"
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    corpo = {"regra": VERSAO_REGRA, "cortes": {"permanente": CORTE_PERMANENTE, "recorrente": CORTE_RECORRENTE},
             "pontos": [dict(zip(("id", "grupo", "lon", "lat", "esperado", "por_que"), p)) for p in PILOTO],
             "referencias": REFERENCIAS,
             "criterio_de_validacao": {
                 "leitura": "toda referencia lida ou ausencia explicada (sem tile / 255); nenhuma excecao muda",
                 "coerencia": "ponto de mar aberto nao pode sair terra; ponto de terra firme nao pode sair mar",
                 "sem_ajuste": "discordancia com o esperado e registrada e investigada; a regra nao e mudada pelo piloto",
                 "custo": "bytes e segundos por tile medidos para estimar a ampliacao"},
             "esperado_e_hipotese": "o esperado vem da geografia; coordenada aproximada pode cair na margem"}
    corpo["hash"] = _sha_json(corpo)
    corpo["quando_utc"] = _agora()
    arq.write_text(json.dumps(corpo, ensure_ascii=False, indent=1), encoding="utf-8")
    return corpo


def _bate(estado: str, esperado: str) -> bool:
    for e in esperado.split("|"):
        if e == "terra" and estado.startswith("terra"):
            return True
        if e == estado:
            return True
    return False


def executar_piloto(destino: Path = SAIDA / "piloto") -> dict:
    pre = pre_registrar_piloto(destino)
    pts = pd.DataFrame(pre["pontos"])
    t0 = time.time()
    a = amostrar(pts.lon.to_numpy(), pts.lat.to_numpy())
    seg = time.time() - t0
    d = pd.concat([pts, a], axis=1)
    d["bate_com_esperado"] = [_bate(e, x) for e, x in zip(d.estado_aquatico, d.esperado)]
    # o que o terreno do acervo diz no mesmo ponto: o mar tinha valor no dem_filled
    sys.path.insert(0, str(REPO / "scripts" / "geo"))
    import acervo_espacial as ae
    ac = ae.Acervo()
    ter = []
    for r in d.itertuples(index=False):
        q = ac.consultar_ponto(float(r.lon), float(r.lat), vizinhanca=False)
        ter.append({"cobertura_terreno": q["cobertura"],
                    "unidade": q["escolhida"]["unidade"] if q["escolhida"] else None,
                    "elevation_m": q["atributos"]["elevation_m"]["valor"], "hand_m": q["atributos"]["hand_m"]["valor"]})
    d = pd.concat([d, pd.DataFrame(ter)], axis=1)
    bytes_wbm = sum(json.loads(p.read_text(encoding="utf-8")).get("bytes", 0) for p in CACHE_WBM.glob("*.json"))
    coerente = not (((d.grupo == "costa") & (d.id == "costa_mar_recife") & d.estado_aquatico.str.startswith("terra"))
                    | ((d.grupo == "terra") & d.estado_aquatico.isin(["mar", "mar_aberto_sem_tile"]))).any()
    res = {"pre_registro": pre["hash"], "segundos": round(seg, 1), "registro": a.attrs["registro"],
           "bytes_wbm_em_cache": bytes_wbm, "coerente": bool(coerente),
           "bate_com_esperado": int(d.bate_com_esperado.sum()), "pontos": int(len(d)),
           "desconhecidos": int((d.estado_aquatico == "desconhecido").sum()),
           "mar_com_terreno_finito": int((d.estado_aquatico.isin(["mar", "mar_aberto_sem_tile"])
                                          & d.elevation_m.notna()).sum()),
           "pontos_detalhe": d.drop(columns=["por_que"]).to_dict("records"), "quando_utc": _agora()}
    res["validado"] = bool(res["coerente"] and res["desconhecidos"] == 0)
    (destino / "resultado.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                            encoding="utf-8")
    return res


def amostrar_tabela(pontos: Path, saida: Path) -> Path:
    """Aquisicao leve: referencias amostradas nos pontos -> parquet com manifesto; tiles WBM ficam em cache."""
    d = pd.read_parquet(pontos, columns=["ponto_id", "lon", "lat"])
    t0 = time.time()
    a = amostrar(d.lon.to_numpy(), d.lat.to_numpy(), progresso=True)
    out = pd.concat([d[["ponto_id"]], a], axis=1)
    saida.mkdir(parents=True, exist_ok=True)
    arq = saida / "dominio_por_ponto.parquet"
    out.to_parquet(arq, index=False)
    man = {"tipo": "dominio_aquatico_por_ponto", "regra": VERSAO_REGRA, "referencias": REFERENCIAS,
           "pontos_origem": str(pontos.relative_to(REPO)), "pontos_origem_sha256": _sha(pontos),
           "saida": arq.name, "saida_sha256": _sha(arq), "linhas": int(len(out)),
           "por_estado": out.estado_aquatico.value_counts().to_dict(), "registro": a.attrs["registro"],
           "segundos": round(time.time() - t0), "codigo_sha256": _sha(Path(__file__)), "quando_utc": _agora(),
           "nao_e": "nao reescreve estado observacional nem rotulo historico"}
    (saida / "manifesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return saida / "manifesto.json"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--piloto", action="store_true")
    ap.add_argument("--ponto", nargs=2, type=float)
    ap.add_argument("--amostrar", type=Path)
    ap.add_argument("--saida", type=Path)
    a = ap.parse_args()
    if a.piloto:
        r = executar_piloto()
        print(json.dumps({k: v for k, v in r.items() if k != "pontos_detalhe"}, ensure_ascii=False, indent=1))
        for p in r["pontos_detalhe"]:
            print(p["id"], p["estado_aquatico"], "| esperado", p["esperado"], "| wbm", p["wbm"], "occ",
                  p["jrc_occurrence"], "saz", p["jrc_seasonality_2021"], "| dem", p["elevation_m"])
    if a.ponto:
        print(amostrar([a.ponto[0]], [a.ponto[1]]).iloc[0].to_dict())
    if a.amostrar:
        print(f"GRAVADO={amostrar_tabela(a.amostrar, a.saida or SAIDA / 'tc01')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
