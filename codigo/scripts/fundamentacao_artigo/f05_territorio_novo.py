"""F05: confirmacao pre-registrada da funcao transferida (Q_sel_ext) num territorio brasileiro nunca examinado.

As cinco coortes do F03 (RMPA, Manaus, Curitiba, Recife, RMR) ja foram examinadas pelo projeto. Aqui a funcao
congelada em `v9_evolucao_abrangente/p0_regularizacao/modelos.pkl` (ajustada so fora do Brasil) e aplicada, sem
nenhum reajuste, a um episodio fluvial urbano com FCU escolhido por regra fixada ANTES de ler qualquer pixel de agua.
Modelo de execucao: `scripts/vulnerabilizados/tv09_manaus.py` (confirmacao de Manaus).

Subcomandos (ambiente entre parenteses):
  triagem     (revp-susc)  regra congelada; picos ANA; cenas S2 julgadas so pelo SCL (fracao de agua nao e gravada)
  grade       (revp-susc)  pontos de 60 m nos setores urbanos do municipio escolhido; terreno e ponte fisica do tile
  cobertura   (py312)      GHSL e SoilGrids (Earth Engine), mesma rota do v8_02b
  protocolo   (py312)      congela hipotese, referencia e criterios antes de predicao e referencia
  predicoes   (py312)      rede bloqueada; Q_sel_ext, HAND e menor declive
  referencia  (revp-susc)  agua Sentinel-2 do evento contra aguas baixas; nao le predicoes
  avaliacao   (py312)      avaliacao unica; IC95% por 2.000 reamostragens de comunidades
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
AQUI = Path(__file__).resolve().parent
for p in ("vulnerabilizados", "metodo", "geo"):
    sys.path.insert(0, str(RAIZ / "scripts" / p))
from comum_tv import (AucPonderavel, agora, bloquear_rede, gravar, intervalo, ler, pesos_bootstrap,  # noqa: E402
                      rel, rotulo_referencia, sha)

BASE = RAIZ / "local_runs/fundamentacao-artigo-20261028/territorio_novo"
# versao 1 = primeiro admissivel (Porto Velho 2019, artefatos na raiz); versao k = k-esimo admissivel na MESMA ordem da regra
VERSAO = int(sys.argv[2]) if len(sys.argv) > 2 else 1
T = BASE if VERSAO == 1 else BASE / f"v{VERSAO}"
ORIGEM = {2: {"motivo": "v1 (Porto Velho, Madeira, 08/03/2019) terminou nao_testavel: 14 positivos em FCU, abaixo do piso de 20",
              "regra_da_escolha": "proximo admissivel na ordem ja congelada da triagem (comunidades FCU no tile, depois cota); "
                                  "decidido antes de ler qualquer pixel de agua deste episodio",
              "criterios": "identicos aos da v1 (piso, bootstrap, semente, comparadores, leitura)",
              "limitacao_declarada": "mesmo ano hidrologico (2021) da coorte de Manaus ja examinada; Itacoatiara fica no Amazonas, "
                                     "~175 km a jusante, com territorio, setores e cenas Sentinel-2 distintos",
              "codigo_v1": "territorio_novo/f05_territorio_novo_v1_congelado.py (hash identico ao do protocolo v1)"}}
TER = RAIZ / "local_runs/ter-01-cadeia-harmonizada"
MODELOS = RAIZ / "local_runs/transferencia-vulnerabilizados/v9_evolucao_abrangente/p0_regularizacao/modelos.pkl"
SETORES = {"AM": RAIZ / "local_runs/social-fontes-complementares/2026-10-12/AM_setores_CD2022.zip",
           "RO": RAIZ / "local_runs/transferencia-vulnerabilizados/fontes/setores_2022/RO_setores_CD2022.zip",
           "PA": BASE / "fontes/PA_setores_CD2022.zip"}
ANA = ("http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica?codEstacao={c}"
       "&dataInicio=01/01/{a0}&dataFim=31/12/{a1}&tipoDados=1&nivelConsistencia=")
STAC = "https://earth-search.aws.element84.com/v1/search"
CLASSES = ["fcu", "baixa_renda", "comparador_urbano", "urbano_sem_renda"]
VALIDO_SCL = [2, 4, 5, 6, 7]
N_BOOT, SEMENTE, NIVEL = 2000, 20261028, 0.95

# Cidades fluviais com >= 10 FCU, estacao fluviometrica ANA na sede e tile de terreno FABDEM/WBT no acervo.
CANDIDATOS = {
    "porto_velho": {"uf": "RO", "cd_mun": "1100205", "estacao": 15400000, "rio": "Madeira", "tile": "gfd_tile_S09W064"},
    "santarem": {"uf": "PA", "cd_mun": "1506807", "estacao": 17900000, "rio": "Tapajos/Amazonas", "tile": "gfd_tile_S03W055"},
    "maraba": {"uf": "PA", "cd_mun": "1504208", "estacao": 29050000, "rio": "Tocantins", "tile": "gfd_tile_S06W050"},
    "itacoatiara": {"uf": "AM", "cd_mun": "1301902", "estacao": 16030000, "rio": "Amazonas", "tile": "gfd_tile_S04W059"},
    "parintins": {"uf": "AM", "cd_mun": "1303403", "estacao": 16350002, "rio": "Amazonas", "tile": "gfd_tile_S03W057"},
    "tefe": {"uf": "AM", "cd_mun": "1304203", "estacao": 12900001, "rio": "Tefe", "tile": "gfd_tile_S04W065"},
    "coari": {"uf": "AM", "cd_mun": "1301209", "estacao": 13150003, "rio": "Solimoes", "tile": "gfd_tile_S05W064"},
}

REGRA_TRIAGEM = {
    "versao": "f05-triagem-1",
    "candidatos": CANDIDATOS,
    "fora_por_ja_examinados": ["Manaus (2021; 2022 reprovado na triagem v6)", "Iranduba/Manacapuru", "RMPA", "Curitiba",
                               "Recife", "RMR", "Blumenau 2023 (reprovado na triagem v6, nao reabrir)"],
    "fora_por_terreno": "Rio Branco (sem tile de terreno no acervo)",
    "episodios": "tres maiores maximos anuais de cota da estacao entre 2017 e 2026 (serie ANA, consistida quando houver)",
    "cena_evento": "S2 L2A em [pico-2, pico+10] dias; fracao valida do SCL (classes 2,4,5,6,7) na caixa das FCU >= 0,80; "
                   "a mais proxima do pico (empate: maior fracao valida); a segunda admissivel vira sensibilidade",
    "cena_aguas_baixas": "minimo de cota nos 300 dias antes do pico; S2 L2A em [minimo-30, minimo+30] com fracao valida >= 0,80; "
                         "a mais proxima do minimo; pre-filtro por eo:cloud_cover <= 60 (metadado, nao pixel)",
    "caixa_fcu": "envoltorio dos setores FCU do municipio com centroide dentro do quadrado de 1 grau do tile",
    "escolha": "entre episodios com as duas cenas: mais comunidades FCU dentro do tile; empate pela maior cota",
    "o_que_se_le": "so SCL; a fracao da classe agua (6) nao e gravada nem impressa",
}


def gt(p, valor):
    return gravar(p, valor, raiz=BASE)


def congelar(p, valor: dict) -> dict:
    p = Path(p)
    if p.exists():
        existente = ler(p)
        for k, v in valor.items():
            if json.loads(json.dumps(existente.get(k), default=str)) != json.loads(json.dumps(v, default=str)):
                raise ValueError(f"congelado diverge em {k}: use outra versao")
        return existente
    gt(p, {**valor, "quando_utc": agora()})
    return ler(p)


def caixa_tile(tile: str):
    """Quadrado de 1 grau pelo nome (canto sudoeste), como o nucleo por latitude do tv09."""
    n = tile.split("_")[-1]
    la = -int(n[1:3]) if n[0] == "S" else int(n[1:3])
    lo = -int(n[4:7]) if n[3] == "W" else int(n[4:7])
    return lo, la, lo + 1, la + 1


# ------------------------------------------------------------------ triagem (revp-susc)
def serie_ana(estacao: int, a0: int, a1: int) -> pd.DataFrame:
    import xml.etree.ElementTree as ET
    with urllib.request.urlopen(ANA.format(c=estacao, a0=a0, a1=a1), timeout=180) as r:
        raiz = ET.fromstring(r.read())
    linhas = []
    for s in raiz.iter("SerieHistorica"):
        d = {e.tag: e.text for e in s}
        mes = pd.Timestamp(d["DataHora"][:10])
        for k in range(1, 32):
            v = d.get(f"Cota{k:02d}")
            if v is None or k > mes.days_in_month:
                continue
            linhas.append((mes.replace(day=k), int(d["NivelConsistencia"]), float(v)))
    df = pd.DataFrame(linhas, columns=["dia", "consistencia", "cota_cm"])
    return df.sort_values("consistencia").drop_duplicates("dia", keep="last").sort_values("dia").reset_index(drop=True)


def busca_s2(bbox, d0: date, d1: date):
    corpo = {"collections": ["sentinel-2-l2a"], "bbox": list(bbox), "limit": 200,
             "datetime": f"{d0.isoformat()}T00:00:00Z/{d1.isoformat()}T23:59:59Z"}
    req = urllib.request.Request(STAC, data=json.dumps(corpo).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["features"]


def fracoes_scl(href: str, bbox) -> dict:
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    with rasterio.open("/vsicurl/" + href) as s:
        b = transform_bounds("EPSG:4326", s.crs, *bbox)
        w = from_bounds(*b, transform=s.transform).round_offsets().round_lengths()
        a = s.read(1, window=w, boundless=True, fill_value=0)
    return {"valida": float(np.isin(a, VALIDO_SCL).mean()), "nuvem": float(np.isin(a, [8, 9, 10]).mean()),
            "sombra": float((a == 3).mean()), "sem_dado": float((a == 0).mean())}


def cenas(bbox, d0, d1, alvo: date, teto_nuvem=None):
    out = []
    for f in busca_s2(bbox, d0, d1):
        nuvem = f["properties"].get("eo:cloud_cover")
        if teto_nuvem is not None and nuvem is not None and float(nuvem) > teto_nuvem:
            continue
        dia = date.fromisoformat(f["properties"]["datetime"][:10])
        fr = fracoes_scl(f["assets"]["scl"]["href"], bbox)
        out.append({"item": f["id"], "dia": dia.isoformat(), "distancia_dias": abs((dia - alvo).days),
                    "nuvem_tile": None if nuvem is None else float(nuvem), **fr,
                    "baseline": f["properties"].get("s2:processing_baseline"),
                    "boa_offset_aplicado": f["properties"].get("earthsearch:boa_offset_applied")})
    return sorted(out, key=lambda c: (c["distancia_dias"], -c["valida"]))


def malha_municipio(uf: str, cd_mun: str):
    import zipfile
    import geopandas as gpd
    z = SETORES[uf]
    shp = [n for n in zipfile.ZipFile(z).namelist() if n.endswith(".shp")][0]
    g = gpd.read_file(f"/vsizip/{z.as_posix()}/{shp}", columns=["CD_SETOR", "CD_MUN"])
    return g[g.CD_MUN.astype(str).eq(cd_mun)].reset_index(drop=True)


def triagem():
    from tv01_universo_escassez import universo_nacional
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    regra = congelar(BASE / "triagem/regra.json", REGRA_TRIAGEM)
    nac = universo_nacional()
    linhas, series = [], {}
    for nome, c in CANDIDATOS.items():
        st = malha_municipio(c["uf"], c["cd_mun"]).merge(nac[["CD_SETOR", "CD_FCU", "classe_territorial"]], on="CD_SETOR")
        x0, y0, x1, y1 = caixa_tile(c["tile"])
        cen = st.to_crs(4326).geometry.representative_point()
        fcu = st[st.classe_territorial.eq("fcu") & cen.x.between(x0, x1).to_numpy() & cen.y.between(y0, y1).to_numpy()]
        info = {"candidato": nome, **c, "fcu_municipio": int(st[st.classe_territorial.eq("fcu")].CD_FCU.nunique()),
                "fcu_no_tile": int(fcu.CD_FCU.nunique()), "terreno": (TER / c["tile"] / "hand_dinf.tif").exists()}
        if info["fcu_no_tile"] == 0 or not info["terreno"]:
            linhas.append({**info, "estado": "sem_fcu_ou_terreno"})
            continue
        bbox = tuple(float(v) for v in fcu.to_crs(4326).total_bounds)
        s = serie_ana(c["estacao"], 2016, 2026)
        series[nome] = s
        gt(BASE / f"triagem/cota_{nome}.csv", s)
        anual = s[s.dia.dt.year >= 2017].assign(ano=lambda q: q.dia.dt.year)
        picos = anual.loc[anual.groupby("ano").cota_cm.idxmax()].nlargest(3, "cota_cm")
        for _, pk in picos.iterrows():
            pico = pk.dia.date()
            antes = s[(s.dia >= pd.Timestamp(pico - timedelta(days=300))) & (s.dia <= pd.Timestamp(pico))]
            minimo = antes.loc[antes.cota_cm.idxmin()].dia.date()
            ev = cenas(bbox, pico - timedelta(days=2), pico + timedelta(days=10), pico)
            ev_ok = [e for e in ev if e["valida"] >= .80]
            pre = cenas(bbox, minimo - timedelta(days=30), minimo + timedelta(days=30), minimo, teto_nuvem=60)
            pre_ok = [e for e in pre if e["valida"] >= .80]
            linhas.append({**info, "pico": pico.isoformat(), "cota_pico_cm": float(pk.cota_cm),
                           "consistencia_pico": int(pk.consistencia), "minimo_antes": minimo.isoformat(),
                           "caixa_fcu": bbox, "cenas_evento": ev, "cenas_pre": pre,
                           "evento": ev_ok[0] if ev_ok else None, "evento_alt": ev_ok[1] if len(ev_ok) > 1 else None,
                           "pre": pre_ok[0] if pre_ok else None,
                           "estado": "admissivel" if ev_ok and pre_ok else ("sem_cena_evento" if not ev_ok else "sem_cena_pre")})
            print(nome, pico, linhas[-1]["estado"], [round(e["valida"], 2) for e in ev], flush=True)
    adm = [r for r in linhas if r["estado"] == "admissivel"]
    escolha = max(adm, key=lambda r: (r["fcu_no_tile"], r["cota_pico_cm"])) if adm else None
    gt(BASE / "triagem/triagem.json", {"regra_sha256": sha(BASE / "triagem/regra.json"), "regra_quando_utc": regra["quando_utc"],
                                    "episodios": linhas, "admissiveis": [(r["candidato"], r["pico"]) for r in adm],
                                    "escolha": None if escolha is None else {k: escolha[k] for k in
                                                                            ["candidato", "uf", "cd_mun", "tile", "rio", "estacao", "pico",
                                                                             "cota_pico_cm", "minimo_antes", "fcu_no_tile", "caixa_fcu",
                                                                             "evento", "evento_alt", "pre"]},
                                    "bloqueio": None if escolha else "nenhum episodio com cena de evento e de aguas baixas admissiveis",
                                    "quando_utc": agora()})
    print("escolha:", None if escolha is None else (escolha["candidato"], escolha["pico"], escolha["evento"]["item"]))


# ------------------------------------------------------------------ grade (revp-susc)
def escolha() -> dict:
    t = ler(BASE / "triagem/triagem.json")
    adm = sorted([r for r in t["episodios"] if r["estado"] == "admissivel"],
                 key=lambda r: (r["fcu_no_tile"], r["cota_pico_cm"]), reverse=True)
    if len(adm) < VERSAO:
        raise SystemExit("triagem sem episodio admissivel para esta versao")
    e = adm[VERSAO - 1]
    if VERSAO == 1 and (e["candidato"], e["pico"]) != (t["escolha"]["candidato"], t["escolha"]["pico"]):
        raise SystemExit("ordenacao diverge da escolha registrada na triagem")
    return e


def grade():
    import rasterio
    from rasterio import features
    from rasterio.warp import transform as wtransform
    from rasterio.windows import Window, from_bounds
    from mt02_atributos import distancias_da_unidade
    from mt11_confirmatorio_transdutivo import declive_p50
    from tv01_universo_escassez import universo_nacional
    e = escolha()
    base = TER / e["tile"]
    nac = universo_nacional()
    st = malha_municipio(e["uf"], e["cd_mun"]).merge(nac, on="CD_SETOR", how="left", suffixes=("", "_nac"))
    st = st[st.classe_territorial.isin(CLASSES)].reset_index(drop=True)
    with rasterio.open(base / "hand_dinf.tif") as s:
        crs = s.crs.to_string()
        st = st.to_crs(crs)
        st["idx"] = np.arange(1, len(st) + 1)
        w = from_bounds(*st.total_bounds, transform=s.transform).round_offsets().round_lengths()
        w = w.intersection(Window(0, 0, s.width, s.height))
        tr, hand, nod = s.window_transform(w), s.read(1, window=w), s.nodata
    idx = features.rasterize(zip(st.geometry, st.idx), out_shape=hand.shape, transform=tr, fill=0, dtype="int32")
    lin, col = np.indices(hand.shape)
    par = ((lin + int(w.row_off)) % 2 == 0) & ((col + int(w.col_off)) % 2 == 0) & (idx > 0)
    xs, ys = (np.asarray(v) for v in rasterio.transform.xy(tr, lin[par], col[par]))
    lon, lat = (np.asarray(v) for v in wtransform(crs, "EPSG:4326", xs, ys))
    x0, y0, x1, y1 = caixa_tile(e["tile"])
    nuc = (lon >= x0) & (lon < x1) & (lat >= y0) & (lat < y1)
    g = pd.DataFrame({"tile": e["tile"], "x": xs[nuc], "y": ys[nuc], "lon": lon[nuc], "lat": lat[nuc],
                      "idx": idx[par][nuc], "hand_m": hand[par][nuc].astype("float64")})
    g.loc[g.hand_m == nod, "hand_m"] = np.nan
    for nome, arq in [("slope_deg", "slope_deg_wbt.tif"), ("twi_dinf", "twi_dinf.tif")]:
        with rasterio.open(base / arq) as s:
            a = s.read(1, window=w)
            g[nome] = a[par][nuc].astype("float64")
            g.loc[g[nome] == s.nodata, nome] = np.nan
    for c in ["hand_m", "slope_deg", "twi_dinf"]:
        g.loc[(g[c] < -1000) | ~np.isfinite(g[c]), c] = np.nan
    g = g.merge(pd.DataFrame(st.drop(columns="geometry"))[["idx", "CD_SETOR", "CD_FCU", "NM_FCU", "classe_territorial",
                                                           "renda_resp", "pessoas"]], on="idx").drop(columns="idx")
    g["terreno_valido"] = g[["hand_m", "slope_deg", "twi_dinf"]].notna().all(axis=1)
    # ponte fisica: mesmas definicoes do v8_02 (contrato do Q)
    g["ln_sca"] = g.twi_dinf + np.log(np.tan(np.radians(np.clip(g.slope_deg, .01, None))))
    d1, d2, lado = distancias_da_unidade(base, g.lon.to_numpy(), g.lat.to_numpy())
    g["dist_drenagem_m"], g["dist_canal10_m"], g["declive_p50"] = d1, d2, declive_p50(base)
    g.insert(0, "ponto_id", [f"TN_{i:07d}" for i in range(len(g))])
    gt(T / "grade.parquet", g)
    fcu = g[g.classe_territorial.eq("fcu")]
    gt(T / "grade.json", {"episodio": e["candidato"] + "_" + e["pico"], "pontos": int(len(g)),
                          "por_classe": g.classe_territorial.value_counts().to_dict(),
                          "terreno_valido": g.groupby("classe_territorial").terreno_valido.mean().to_dict(),
                          "comunidades_fcu_universo": int(st[st.classe_territorial.eq("fcu")].CD_FCU.nunique()),
                          "comunidades_fcu_com_ponto": int(fcu.CD_FCU.nunique()), "lado_unidade_km": lado,
                          "sha_grade": sha(T / "grade.parquet"), "quando_utc": agora()})
    print(len(g), g.classe_territorial.value_counts().to_dict(), fcu.CD_FCU.nunique())


# ------------------------------------------------------------------ cobertura (py312)
def cobertura():
    import ee
    import cob01_cobertura_e_solo as cob
    from v8_02b_cobertura_solo import amostrar, baixar
    ee.Initialize()
    e = escolha()
    ano = int(e["pico"][:4])
    g = pd.read_parquet(T / "grade.parquet", columns=["ponto_id", "lon", "lat"])
    ep = cob.anos_das_camadas(ano)["ghsl"]
    caixa = [g.lon.min() - .01, g.lat.min() - .01, g.lon.max() + .01, g.lat.max() + .01]
    ghsl = ee.ImageCollection(cob.GHSL).filter(ee.Filter.eq("system:index", str(ep))).first().select("built_surface").divide(cob.AREA_CELULA_GHSL_M2)
    areia = ee.Image(cob.SOILGRIDS["areia"]).select([f"sand_{c}_mean" for c in cob.CAMADAS_SOLO]).reduce(ee.Reducer.mean()).divide(10)
    arq_c, arq_a = T / f"cobertura/ghsl{ep}.tif", T / "cobertura/areia.tif"
    arq_c.parent.mkdir(parents=True, exist_ok=True)
    if not arq_c.exists():
        baixar(ee, ghsl, caixa, arq_c)
    if not arq_a.exists():
        baixar(ee, areia, caixa, arq_a)
    out = g[["ponto_id"]].assign(construido_v2=amostrar(arq_c, g.lon.to_numpy(), g.lat.to_numpy()),
                                 areia_v2=amostrar(arq_a, g.lon.to_numpy(), g.lat.to_numpy()), ghsl_epoca=ep)
    gt(T / "cobertura.parquet", out)
    gt(T / "cobertura.json", {"ghsl_epoca": ep, "linhas": len(out), "finitos": {c: int(out[c].notna().sum()) for c in ["construido_v2", "areia_v2"]},
                              "rasters_sha256": {arq_c.name: sha(arq_c), arq_a.name: sha(arq_a)}, "quando_utc": agora()})
    print(ep, {c: int(out[c].notna().sum()) for c in ["construido_v2", "areia_v2"]})


# ------------------------------------------------------------------ protocolo (py312)
def protocolo():
    e = escolha()
    cod = {rel(p): sha(p) for p in [AQUI / "f05_territorio_novo.py", RAIZ / "scripts/vulnerabilizados/comum_tv.py",
                                    RAIZ / "scripts/metodo/mt05_transferencia_sem_amostra.py"]}
    p = congelar(T / "protocolo.json", {
        "versao": f"f05-territorio-novo-{VERSAO}",
        **({"origem_da_versao": ORIGEM[VERSAO], "ordem_na_triagem": VERSAO} if VERSAO > 1 else {}),
        "episodio": {k: e[k] for k in ["candidato", "uf", "cd_mun", "rio", "estacao", "pico", "cota_pico_cm", "tile"]},
        "natureza": "territorio e episodio nunca examinados pelo projeto; escolhidos pela triagem congelada (so SCL, sem agua)",
        "funcao": {"arquivo": rel(MODELOS), "sha256": sha(MODELOS), "chave": "Q_sel_ext",
                   "ajuste": "so componentes fora_BR; nenhum reajuste, nenhum limiar escolhido no destino"},
        "hipotese": "H-F05: em pontos FCU do territorio novo, Q_sel_ext ordena a agua de cheia observada melhor que o HAND",
        "comparadores": {"A0_hand": "-hand_m", "menos_declive": "-slope_deg"},
        "populacao": {"universo": "setores urbanos do municipio nas classes " + ",".join(CLASSES),
                      "grade": "celulas de 30 m com linha e coluna pares (60 m) no quadrado de 1 grau do tile",
                      "teste": "pontos FCU com terreno valido e estado positivo/negativo"},
        "referencia": {"evento": e["evento"]["item"], "pre_aguas_baixas": e["pre"]["item"],
                       "evento_alt": None if e["evento_alt"] is None else e["evento_alt"]["item"],
                       "agua": "MNDWI > 0 em 20 m (B03 media 2x2; B11)", "valido_scl": VALIDO_SCL,
                       "reflectancia": "se baseline >= 04.00 e o item nao declara offset aplicado, subtrai 1000 de B03/B08/B11",
                       "permanente": "agua valida na cena de aguas baixas (sem JRC no tile)",
                       "estados": "rotulo_referencia() de comum_tv", "ligacao": "pixel de 20 m que contem o ponto",
                       "observacao_de_regime": "positivo inclui alagamento sazonal acima da agua baixa; nao e anomalia sobre a cheia media"},
        "sensibilidades_planejadas": {"mndwi_0_2": "limiar 0,2", "ndwi_b08": "NDWI B03/B08 > 0",
                                      "cena_alternativa": "segunda cena admissivel da triagem, se houver (pode ser tile MGRS vizinho da mesma passagem, nao outra data)"},
        "criterios": {"nivel_intervalo": NIVEL, "bootstrap": {"B": N_BOOT, "semente": SEMENTE, "grupo": "comunidade CD_FCU"},
                      "primario": "Q_sel_ext - A0_hand em FCU: sustentado se limite inferior do IC95% > 0; contrario se superior < 0; senao inconclusivo",
                      "secundarios": ["AUC(Q_sel_ext) em FCU: limite inferior > 0,5", "Q_sel_ext - menos_declive (mesma leitura)"],
                      "nao_testavel": "< 8 comunidades com ponto conhecido ou < 20 positivos/negativos em FCU"},
        "parada": "avaliacao unica; resultado e registrado qualquer que seja; sem segunda escolha de territorio",
        "codigo": cod, "sha_grade": sha(T / "grade.parquet"), "sha_cobertura": sha(T / "cobertura.parquet"),
        "sha_triagem": sha(BASE / "triagem/triagem.json")})
    print(p["quando_utc"], sha(T / "protocolo.json"))


# ------------------------------------------------------------------ predicoes (py312)
def tabela() -> pd.DataFrame:
    g = pd.read_parquet(T / "grade.parquet").merge(pd.read_parquet(T / "cobertura.parquet"), on="ponto_id")
    return g[g.terreno_valido].reset_index(drop=True)


def predicoes():
    import mt05_transferencia_sem_amostra as mt
    bloquear_rede()
    prot = ler(T / "protocolo.json")
    if (T / "referencia.parquet").exists():
        raise SystemExit("referencia ja existe")
    for arq, h in prot["codigo"].items():
        if sha(RAIZ / arq) != h:
            raise SystemExit(f"codigo alterado depois do congelamento: {arq}")
    if sha(MODELOS) != prot["funcao"]["sha256"]:
        raise SystemExit("modelos.pkl diverge do protocolo")
    with MODELOS.open("rb") as f:
        Q = pickle.load(f)["Q_sel_ext"]
    d = tabela()
    x = mt.caracteristicas(d)
    saida = d[["ponto_id"]].assign(Q_sel_ext=Q.escore(x), A0_hand=-d.hand_m.to_numpy(), menos_declive=-d.slope_deg.to_numpy())
    gt(T / "predicoes.parquet", saida)
    gt(T / "predicoes_registro.json", {"sha_predicoes": sha(T / "predicoes.parquet"), "pontos": len(saida),
                                       "faltantes_entrada": {c: int(x[c].isna().sum()) for c in x},
                                       "protocolo_sha256": sha(T / "protocolo.json"), "quando_utc": agora()})
    print(len(saida))


# ------------------------------------------------------------------ referencia (revp-susc)
def refletancia(a, item_info):
    a = a.astype("float64")
    b = item_info.get("baseline")
    if b is not None and float(b) >= 4 and item_info.get("boa_offset_aplicado") is not True:
        a = a - 1000.0
    return a


def referencia():
    import rasterio
    from rasterio.warp import transform as wtransform
    from tv06_referencia_rmpa import amostrar, indices, ler_janela
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    prot = ler(T / "protocolo.json")
    sha_pred = ler(T / "predicoes_registro.json")["sha_predicoes"]
    e = escolha()
    g = pd.read_parquet(T / "grade.parquet", columns=["ponto_id", "lon", "lat"])
    itens = {"evento": e["evento"], "pre": e["pre"]}
    if e["evento_alt"] is not None:
        itens["evento_alt"] = e["evento_alt"]
    amostras, fontes = {}, []
    for chave, info in itens.items():
        corpo = {"collections": ["sentinel-2-l2a"], "ids": [info["item"]]}
        req = urllib.request.Request(STAC, data=json.dumps(corpo).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            item = json.load(r)["features"][0]
        crs = f"EPSG:{item['properties']['proj:epsg']}"
        xs, ys = (np.asarray(v) for v in wtransform("EPSG:4326", crs, g.lon.to_numpy(), g.lat.to_numpy()))
        b = (xs.min() - 200, ys.min() - 200, xs.max() + 200, ys.max() + 200)
        cols = {}
        for banda, chave_asset in [("SCL", "scl"), ("B03", "green"), ("B11", "swir16"), ("B08", "nir")]:
            u = item["assets"][chave_asset]["href"]
            a, tr, crs_lido = ler_janela("/vsicurl/" + u, b, 20.0)
            if crs_lido != crs:
                raise ValueError(crs_lido)
            v = amostrar(a, tr, xs, ys, 0)
            cols[banda] = v if banda == "SCL" else refletancia(v, info)
            fontes.append(u)
        m, n = indices(cols["B03"], cols["B11"], cols["B08"])
        amostras[chave] = {"valido": np.isin(cols["SCL"], prot["referencia"]["valido_scl"]), "com_dado": cols["SCL"] != 0,
                           "mndwi": m, "ndwi": n, "scl": cols["SCL"].astype(int)}
        print(chave, flush=True)
    ev, pre = amostras["evento"], amostras["pre"]
    sem_jrc = np.zeros(len(g), bool)
    ref = g[["ponto_id"]].copy()
    ref["valido_evento"], ref["valido_pre"] = ev["valido"], pre["valido"]
    ref["scl_evento"], ref["mndwi_evento"], ref["mndwi_pre"] = ev["scl"], ev["mndwi"], pre["mndwi"]
    testes = [("primaria", ev, ev["mndwi"] > 0, pre["mndwi"] > 0), ("mndwi_0_2", ev, ev["mndwi"] > .2, pre["mndwi"] > .2),
              ("ndwi_b08", ev, ev["ndwi"] > 0, pre["ndwi"] > 0)]
    if "evento_alt" in amostras:
        alt = amostras["evento_alt"]
        testes.append(("cena_alternativa", alt, alt["mndwi"] > 0, pre["mndwi"] > 0))
    for nome, ee_, ag, ap in testes:
        ref[f"estado_{nome}"] = rotulo_referencia(ag, ap, sem_jrc, ee_["valido"], pre["valido"])
    gt(T / "referencia.parquet", ref)
    gt(T / "referencia.json", {"sha_predicoes_ja_registradas": sha_pred, "leu_predicoes": False, "fontes": fontes,
                               "estados": {c: ref[c].value_counts().to_dict() for c in ref if c.startswith("estado_")},
                               "sha_referencia": sha(T / "referencia.parquet"), "quando_utc": agora()})
    print({c: ref[c].value_counts().to_dict() for c in ref if c.startswith("estado_")})


# ------------------------------------------------------------------ avaliacao (py312)
def contraste(p: pd.DataFrame) -> dict:
    """Mesma logica do F03: AUC ponderavel e pesos de reamostragem de comunidades inteiras."""
    cols = ["Q_sel_ext", "A0_hand", "menos_declive"]
    fun = {k: AucPonderavel(p.y, p[k]) for k in cols}
    pontual = {k: fun[k](np.ones(len(p))) for k in cols}
    reps = np.array([[fun[k](w) for k in cols] for w in pesos_bootstrap(p.grupo, N_BOOT, SEMENTE)])
    res = {"pontos": int(len(p)), "positivos": int(p.y.sum()), "negativos": int((1 - p.y).sum()),
           "comunidades": int(p.grupo.nunique()), "comunidades_com_positivo": int(p[p.y == 1].grupo.nunique()),
           "prevalencia": float(p.y.mean()), "auc": {k: float(v) for k, v in pontual.items()},
           "ic95_auc_Q": intervalo(reps[:, 0], NIVEL), "ic95_auc_hand": intervalo(reps[:, 1], NIVEL)}
    for ref, j in (("A0_hand", 1), ("menos_declive", 2)):
        ic = intervalo(reps[:, 0] - reps[:, j], NIVEL)
        res[f"Q_menos_{ref}"] = {"delta": float(pontual["Q_sel_ext"] - pontual[ref]), "ic95": ic,
                                "leitura": "sustentado" if ic and ic[0] > 0 else ("contrario" if ic and ic[1] < 0 else "inconclusivo")}
    return res


def avaliacao():
    bloquear_rede()
    reg = ler(T / "predicoes_registro.json")
    if sha(T / "predicoes.parquet") != reg["sha_predicoes"] or ler(T / "referencia.json")["sha_predicoes_ja_registradas"] != reg["sha_predicoes"]:
        raise SystemExit("ordem predicao -> referencia nao confere")
    if (T / "resultado.json").exists():
        raise SystemExit("avaliacao unica ja executada")
    prot = ler(T / "protocolo.json")
    d = tabela().merge(pd.read_parquet(T / "predicoes.parquet"), on="ponto_id").merge(pd.read_parquet(T / "referencia.parquet"), on="ponto_id")
    d["grupo"] = np.where(d.classe_territorial.eq("fcu"), "FCU:" + d.CD_FCU.astype(str), "SET:" + d.CD_SETOR.astype(str))

    def teste(estado: str, classe="fcu"):
        q = d[d.classe_territorial.eq(classe) & d[estado].isin(["positivo", "negativo"])].copy()
        q["y"] = q[estado].eq("positivo").astype(int)
        if q.grupo.nunique() < 8 or q.y.sum() < 20 or (1 - q.y).sum() < 20:
            return {"estado": "nao_testavel", "comunidades": int(q.grupo.nunique()), "positivos": int(q.y.sum()),
                    "negativos": int((1 - q.y).sum())}
        return {"estado": "avaliado", **contraste(q.reset_index(drop=True))}

    principal = teste("estado_primaria")
    sens = {k: teste(f"estado_{k}") for k in ["mndwi_0_2", "ndwi_b08", "cena_alternativa"] if f"estado_{k}" in d}
    comp = teste("estado_primaria", "comparador_urbano")
    fcu = d[d.classe_territorial.eq("fcu")]
    gt(T / "resultado.json", {
        "episodio": prot["episodio"], "regra": prot["criterios"]["primario"], "nivel_intervalo": NIVEL,
        "bootstrap": {"B": N_BOOT, "semente": SEMENTE, "grupo": "comunidade FCU"},
        "referencia": "Sentinel-2 MNDWI > 0 na cena do evento contra a cena de aguas baixas (rotulo_referencia)",
        "fcu_primario": principal, "sensibilidades_fcu": sens, "comparador_urbano_descritivo": comp,
        "estados_fcu": fcu.estado_primaria.value_counts().to_dict(),
        "leitura_primaria": principal.get("Q_menos_A0_hand", {}).get("leitura", principal["estado"]),
        "protocolo_sha256": sha(T / "protocolo.json"), "predicoes_sha256": sha(T / "predicoes.parquet"),
        "referencia_sha256": sha(T / "referencia.parquet"), "quando_utc": agora()})
    print(json.dumps(principal, default=float, indent=1))


def descritivo():
    """Depois da avaliacao unica: contraste na referencia primaria apesar do piso nao atingido. Nao e confirmacao."""
    bloquear_rede()
    r = ler(T / "resultado.json")
    if r["predicoes_sha256"] != sha(T / "predicoes.parquet") or r["referencia_sha256"] != sha(T / "referencia.parquet"):
        raise SystemExit("artefatos mudaram depois da avaliacao")
    d = tabela().merge(pd.read_parquet(T / "predicoes.parquet"), on="ponto_id").merge(pd.read_parquet(T / "referencia.parquet"), on="ponto_id")
    q = d[d.classe_territorial.eq("fcu") & d.estado_primaria.isin(["positivo", "negativo"])].reset_index(drop=True)
    q = q.assign(y=q.estado_primaria.eq("positivo").astype(int), grupo="FCU:" + q.CD_FCU.astype(str))
    gt(T / "descritivo_pos_avaliacao.json", {
        "status": "descritivo_pos_avaliacao; o protocolo classificou o teste primario como nao_testavel (< 20 positivos em FCU); nao e confirmacao",
        "fcu_referencia_primaria": contraste(q), "resultado_sha256": sha(T / "resultado.json"), "quando_utc": agora()})
    print(json.dumps(contraste(q), default=float, indent=1))


if __name__ == "__main__":
    {"triagem": triagem, "grade": grade, "cobertura": cobertura, "protocolo": protocolo, "predicoes": predicoes,
     "referencia": referencia, "avaliacao": avaliacao, "descritivo": descritivo}[sys.argv[1]]()
