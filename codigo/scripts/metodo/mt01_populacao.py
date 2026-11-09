"""MT01, passo 1: o que e testavel -- populacao evento x territorio x processo x fonte x qualidade.

Mecanismos transferiveis (10/10/2026). Antes de qualquer ajuste, reconstroi a
populacao do ds06 v7 com o que cada linha permite dizer:

  processo   CEMS traz o processo POR POLIGONO observado (Riverine/Flash/Coastal
             flood, `obj_desc` dos produtos rapid mapping); o positivo herda o
             processo do poligono que o contem. GFD (MODIS 250 m), Sen1Floods11
             e UFO so tem o processo do EVENTO, nunca do ponto -- ficam como
             atribuicao por evento, declarada, e nao como processo do ponto.
  papel      positivo, negativo observado, exclusao qualificada; a ausencia de
             registro (centrais 156/311) e contada no ds05, nunca vira negativo.
  qualidade  instrumento, resolucao nativa, procedencia da data, familia do MDT.
  bacia      HydroBASINS v1c niveis 6/8/12 em SA, EU, AF, AS, NA, AU.
  contexto   descritores do territorio SEM rotulo: quantis de HAND, declive e
             TWI do raster inteiro da unidade de terreno (nao dos pontos
             amostrados, cuja amostragem depende da classe).

Blocos de processo (proxies declarados, nunca medidas diretas):
  acumulacao/extravasamento  hand_m, elevation_local_m
  concentracao/conexao       ln_sca (= twi + ln tan declive), slope_deg
  producao de escoamento     construido_frac, areia_pct_0_30 (rain_max_24h so como modificador
                             parcial: ERA5-Land ~9 km e quase constante dentro do evento)
  restricao da saida         ausente (galeria, mare, remanso, operacao): nao identificavel

nao_e: nao treina, nao separa treino/teste, nao le reserva protegida (EMSR868/871
nao sao abertas), nao reescreve rotulo.

Uso:
    python scripts/metodo/mt01_populacao.py montar
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "geo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402
import mt01_bacias_globais as bg  # noqa: E402

VERSAO = "mt01_populacao_v1"
RUNS = REPO / "local_runs"
DS06 = RUNS / "ds-06-dataset-de-treino" / "v7" / "pontos.parquet"
DS05 = RUNS / "ds-05-tabela-unica" / "tabela_unica_v2.parquet"
CEMS_ZIPS = RUNS / "cems-05-pacotes"
ACERVO = RUNS / "acervo-espacial" / "unidades.parquet"
SAIDA = RUNS / "mecanismos-transferiveis" / "v1"
PROTEGIDAS = ("EMSR868", "EMSR871")  # reservas brutas do OBS03: nunca abertas aqui

FONTES_CONTRASTE = ("gfd", "cems", "sen1floods11", "ufo")
COLUNAS = ["ponto_id", "fonte", "mecanismo", "papel", "classe", "qualidade_negativo", "territorio", "classe_relevo",
           "unidade_espacial", "evento_id", "evento_tipo", "janela_inicio", "janela_fim", "precisao_data", "lat",
           "lon", "resolucao_nativa_m", "procedencia", "hand_m", "twi_dinf", "slope_deg", "elevation_local_m",
           "rain_max_24h", "construido_frac", "areia_pct_0_30", "argila_pct_0_30", "worldcover_2021"]

BLOCOS = {
    "acumulacao": ["hand_m", "elevation_local_m"],
    "conexao": ["ln_sca", "slope_deg"],
    "producao": ["construido_frac", "areia_pct_0_30"],
}
AUSENTES = {
    "restricao_da_saida": "galerias, bueiros, obstrucao, mare, remanso e operacao nao existem no acervo de pontos",
    "forcamento_no_ponto": "rain_max_24h e ERA5-Land ~9 km: quase constante dentro do evento; GFD/centrais sem chuva",
    "umidade_antecedente": "rain_decay_index so em parte das fontes; nao entra no contraste dentro do evento",
    "capacidade_do_canal": "HAND mede altura sobre a drenagem representada, nao nivel nem capacidade",
}

INSTRUMENTO = {"gfd": "MODIS 250 m (DFO/GFD), optico, nuvem e cheia rapida invisiveis",
               "cems": "CEMS rapid mapping (S1/optico, 10-30 m), delimitacao por analista",
               "sen1floods11": "Sen1Floods11 rotulo manual sobre S1/S2 10 m",
               "ufo": "UFO, rotulo urbano em imagem de 3 m",
               "uk": "contornos registrados EA, negativo por exclusao"}
PROCESSO_EVENTO = {"gfd": "fluvial_grande_escala_evento", "sen1floods11": "fluvial_monsonico_evento",
                   "ufo": "urbano_misto_evento", "uk": "fluvial_registrado_evento"}
CEMS_PROC = {"Riverine flood": "fluvial", "Flash flood": "enxurrada", "Coastal flood": "costeira"}


def escrever(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".part")
    t.write_text(json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False, default=str) + "\n",
                 encoding="utf-8")
    t.replace(p)


# ------------------------------------------------------------- processo CEMS

def poligonos_cems(codigo: str) -> list[tuple[str, object]]:
    """(processo, geometria) dos `observedEvent` de uma ativacao; recusa reserva protegida."""
    from shapely.geometry import shape
    if codigo in PROTEGIDAS:
        raise PermissionError(f"{codigo} e reserva protegida")
    arq = CEMS_ZIPS / f"{codigo}_products.zip"
    if not arq.exists():
        return []
    out = []

    def ler(nome, dados):
        if "observedEvent" in nome and nome.endswith(".json"):
            for f in json.loads(dados)["features"]:
                p = f["properties"]
                if p.get("event_type", "").startswith("5-") and f.get("geometry"):
                    out.append((CEMS_PROC.get(p.get("obj_desc"), "outro_alagamento"), shape(f["geometry"])))

    with zipfile.ZipFile(arq) as z:
        for n in z.namelist():
            if n.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(z.read(n))) as interno:
                    for m in interno.namelist():
                        ler(m, interno.read(m) if "observedEvent" in m and m.endswith(".json") else b"")
            else:
                ler(n, z.read(n) if "observedEvent" in n and n.endswith(".json") else b"")
    return out


def processo_cems(d: pd.DataFrame, tolerancia_graus: float = 2e-4) -> pd.Series:
    """Processo do poligono observado que contem o ponto (+-~20 m). Varios processos -> `misto`."""
    import shapely
    proc = pd.Series("nao_aplicavel", index=d.index, dtype=object)
    for ev, g in d[d.fonte == "cems"].groupby("evento_id"):
        codigo = ev.split(":")[1]
        pol = poligonos_cems(codigo)
        pos = g[g.classe == 1]
        if not pol:
            proc.loc[pos.index] = "sem_poligono_no_acervo"
            continue
        nomes = np.array([p for p, _ in pol])
        geoms = np.array([gg for _, gg in pol], dtype=object)
        arv = shapely.STRtree(geoms)
        pts = shapely.buffer(shapely.points(pos.lon.to_numpy(), pos.lat.to_numpy()), tolerancia_graus)
        i_pt, i_pol = arv.query(pts, predicate="intersects")
        achados = pd.DataFrame({"i": i_pt, "p": nomes[i_pol]}).drop_duplicates()
        por_ponto = achados.groupby("i").p.agg(lambda s: s.iloc[0] if len(s) == 1 else "misto")
        r = np.full(len(pos), "nao_atribuido", dtype=object)
        r[por_ponto.index.to_numpy()] = por_ponto.to_numpy()
        proc.loc[pos.index] = r
    return proc


# ------------------------------------------------------------- contexto sem rotulo

def unidade_de_terreno(unidade_espacial: str, nomes: set) -> str | None:
    fonte, resto = unidade_espacial.split("|", 1)
    candidatos = [resto, "gfd_tile_" + resto.replace("GFD_", ""), "n1chip_" + resto]
    if fonte == "cems":
        candidatos.append(resto.split("_")[0])
    for c in candidatos:
        if c in nomes:
            return c
    return None


def contexto_das_unidades(unidades: pd.DataFrame, passo: int = 4) -> pd.DataFrame:
    """Quantis do raster INTEIRO de cada unidade (decimado), sem olhar ponto nem rotulo."""
    import rasterio
    from rasterio.enums import Resampling
    linhas = []
    for r in unidades.itertuples(index=False):
        reg = {"unidade_terreno": r.unidade, "familia_mdt": r.familia_mdt, "compativel": bool(r.compativel)}
        for arq, nome in (("hand_dinf.tif", "hand"), ("slope_deg_wbt.tif", "declive"), ("twi_dinf.tif", "twi")):
            try:
                with rasterio.open(Path(r.caminho) / arq) as s:
                    a = s.read(1, out_shape=(max(1, s.height // passo), max(1, s.width // passo)),
                               resampling=Resampling.nearest).astype(float)
                    if s.nodata is not None:
                        a[a == s.nodata] = np.nan
                    a[a < -1e4] = np.nan
                v = a[np.isfinite(a)]
                for q in (10, 50, 90):
                    reg[f"{nome}_p{q}"] = float(np.percentile(v, q)) if v.size else np.nan
                reg[f"{nome}_celulas"] = int(v.size)
            except Exception as e:  # registra, nao inventa
                reg[f"{nome}_erro"] = str(e)[:120]
        linhas.append(reg)
    return pd.DataFrame(linhas)


# ------------------------------------------------------------- bacias

def localizar_bacias(lon: np.ndarray, lat: np.ndarray) -> pd.DataFrame:
    import bacias_hydrobasins as hb
    bg.registrar_regioes()
    b = hb.Bacias()
    l6 = b.localizar(lon, lat, 6)
    l8 = b.localizar(lon, lat, 8)
    l12 = b.localizar(lon, lat, 12)
    return pd.DataFrame({"MAIN_BAS": l6.MAIN_BAS.to_numpy(), "estado_bacia": l6.estado_bacia.to_numpy(),
                         "HYBAS_L6": l6.HYBAS_ID.to_numpy(), "HYBAS_L8": l8.HYBAS_ID.to_numpy(),
                         "MAIN_BAS_L8": l8.MAIN_BAS.to_numpy(), "UP_AREA_L12": l12.UP_AREA.to_numpy(),
                         "SUB_AREA_L12": l12.SUB_AREA.to_numpy(), "COAST_L12": l12.COAST.to_numpy(),
                         "ENDO_L12": l12.ENDO.to_numpy()}), b.man["hash_resumo"]


# ------------------------------------------------------------- montagem

def derivar(d: pd.DataFrame) -> pd.DataFrame:
    """Area contribuinte especifica de volta do TWI: ln SCA = TWI + ln tan(declive). Sem rotulo."""
    tg = np.tan(np.radians(d.slope_deg.clip(lower=0.01)))
    d["ln_sca"] = d.twi_dinf + np.log(tg)
    return d


def montar(saida: Path = SAIDA) -> dict:
    t0 = time.time()
    saida.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(DS06, columns=COLUNAS)
    d = derivar(d)
    d["processo_ponto"] = "nao_aplicavel"
    d.loc[d.classe == 1, "processo_ponto"] = "nao_separavel_no_ponto"
    d["processo_evento"] = d.fonte.map(PROCESSO_EVENTO).fillna("pluvial_urbano_registro")
    proc = processo_cems(d)
    d.loc[d.fonte == "cems", "processo_ponto"] = proc[d.fonte == "cems"]
    d["instrumento"] = d.fonte.map(INSTRUMENTO).fillna("central de chamados (presenca; negativo = ausencia)")

    u = pd.read_parquet(ACERVO)
    nomes = set(u.unidade)
    mapa = {x: unidade_de_terreno(x, nomes) for x in d.unidade_espacial.dropna().unique()}
    d["unidade_terreno"] = d.unidade_espacial.map(mapa)
    usadas = u[u.unidade.isin(set(mapa.values()) - {None})]
    arq_ctx = saida / "contexto_unidades.parquet"
    if not arq_ctx.exists():
        contexto_das_unidades(usadas).to_parquet(arq_ctx, index=False)
    ctx = pd.read_parquet(arq_ctx)
    d = d.merge(ctx[["unidade_terreno", "familia_mdt"]], on="unidade_terreno", how="left")

    coords = d[["lon", "lat"]].round(5).drop_duplicates().reset_index(drop=True)
    bac, hash_bacias = localizar_bacias(coords.lon.to_numpy(), coords.lat.to_numpy())
    coords = pd.concat([coords, bac], axis=1)
    d = d.assign(_lon=d.lon.round(5), _lat=d.lat.round(5)).merge(
        coords.rename(columns={"lon": "_lon", "lat": "_lat"}), on=["_lon", "_lat"], how="left").drop(
        columns=["_lon", "_lat"])
    d["ln_up_area_l12"] = np.log(d.UP_AREA_L12.where(d.UP_AREA_L12 > 0))

    d = d.sort_values("ponto_id", kind="stable").reset_index(drop=True)
    arq = saida / "populacao.parquet"
    d.to_parquet(arq, index=False)
    inv = inventario(d)
    man = {"tipo": "mt01_populacao", "versao": VERSAO, "quando_utc": agora_utc(),
           "entradas": {str(p.relative_to(REPO)): sha256_arquivo(p) for p in (DS06, ACERVO)},
           "cems_zips": {p.name: sha256_arquivo(p) for p in sorted(CEMS_ZIPS.glob("EMSR*_products.zip"))
                         if not p.name.startswith(PROTEGIDAS)},
           "reservas_protegidas_nao_abertas": list(PROTEGIDAS),
           "bacias_hash_resumo": hash_bacias,
           "codigo": {str(Path(f).relative_to(REPO)): sha256_arquivo(Path(f)) for f in (__file__, bg.__file__)},
           "saidas": {"populacao.parquet": sha256_arquivo(arq), "contexto_unidades.parquet": sha256_arquivo(arq_ctx)},
           "blocos": BLOCOS, "ausentes": AUSENTES, "segundos": round(time.time() - t0, 1),
           "nao_e": "nao treina; nao separa; populacao de desenvolvimento ja examinada em ciclos anteriores"}
    escrever(saida / "manifesto_populacao.json", man)
    escrever(saida / "inventario.json", inv)
    return {"manifesto": man, "inventario": inv}


def inventario(d: pd.DataFrame) -> dict:
    """Contagens por evento x territorio x processo x fonte x qualidade; linhas nao sao unidades."""
    inv = {"linhas": len(d)}
    inv["por_fonte_papel"] = {f: g.papel.value_counts().to_dict() for f, g in d.groupby("fonte")}
    ev = d[d.evento_id.notna()]
    duas = ev.groupby("evento_id").classe.nunique()
    inv["eventos_com_duas_classes"] = {f: int(duas[duas == 2].index.isin(g.evento_id.unique()).sum())
                                       for f, g in ev.groupby("fonte")}
    inv["processo_ponto_positivos"] = {f: g[g.classe == 1].processo_ponto.value_counts().to_dict()
                                       for f, g in d.groupby("fonte")}
    cems = d[(d.fonte == "cems") & (d.classe == 1)]
    inv["cems_processo_por_evento"] = {e: g.processo_ponto.value_counts().to_dict()
                                       for e, g in cems.groupby("evento_id")}
    inv["qualidade"] = {f: {"resolucao_m": sorted(g.resolucao_nativa_m.dropna().unique().tolist()),
                            "precisao_data": g.precisao_data.value_counts().to_dict(),
                            "familia_mdt": g.familia_mdt.fillna("sem_unidade").value_counts().to_dict(),
                            "instrumento": INSTRUMENTO.get(f, "central de chamados")}
                        for f, g in d.groupby("fonte")}
    inv["bacia"] = d.estado_bacia.fillna("nao_localizado").value_counts().to_dict()
    inv["cobertura_atributos"] = {f: {c: round(float(g[c].notna().mean()), 4)
                                      for c in sum(BLOCOS.values(), []) + ["rain_max_24h", "ln_up_area_l12"]}
                                  for f, g in d.groupby("fonte")}
    a = pd.read_parquet(DS05, columns=["fonte", "nivel_negativo", "classe_nome", "mecanismo"])
    inv["ds05_ausencia_e_exclusoes"] = {f: g.nivel_negativo.value_counts().to_dict() for f, g in a.groupby("fonte")}
    inv["testavel"] = {
        "contraste_com_negativo_observado": sorted(d[d.papel == "negativo_observado"].fonte.unique().tolist()),
        "so_presenca": sorted(set(d.fonte) - set(d[d.papel == "negativo_observado"].fonte)),
        "processo_no_ponto": "so CEMS (poligono observado com obj_desc)",
        "pluvial_urbano": ("not_testable como suscetibilidade: centrais 156/311 tem presenca e ausencia de registro "
                           "(target-group background, Phillips et al. 2009), nenhum negativo observado no ds06"),
    }
    return inv


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("acao", choices=("montar",))
    ap.add_argument("--saida", type=Path, default=SAIDA)
    a = ap.parse_args()
    r = montar(a.saida)
    print(json.dumps({k: r["inventario"][k] for k in ("linhas", "eventos_com_duas_classes",
                                                      "processo_ponto_positivos", "bacia")},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
