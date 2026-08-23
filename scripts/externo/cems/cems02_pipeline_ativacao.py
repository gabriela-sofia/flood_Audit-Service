"""
CEMS-02 -- Pipeline generico: de um zip de ativacao a uma tabela de pontos.

POR QUE ESTE SCRIPT E O PRODUTO, e nao mais um script:
baixar ativacao e barato; o caro e transformar cada uma num dataset. Cinco
ativacoes analogas as regioes do projeto somam cerca de 90 AOIs espalhadas
pelo mundo, cada uma exigindo DEM, HAND, cobertura do solo, chuva, reprojecao
para o UTM local e calculo de TWI. Feito a mao, uma por uma, seria inviavel.
Aqui vira um comando por ativacao.

O QUE ELE ENSINA AO MODELO, que e o objetivo declarado:
cada ponto sai rotulado com a classe que o proprio produto Copernicus declara,
nao com inferencia nossa:

    INUNDACAO              observedEventA com event_type '5-Flood'
    MOVIMENTO_DE_MASSA     observedEventA com event_type '6-Mass Movement'
    NAO_INUNDADO_OBSERVADO (AOI ∩ footprint) menos as duas anteriores
    AGUA_PERMANENTE        JRC Global Surface Water > 50%

O movimento de massa e classe PROPRIA, nao negativo e nao positivo. Foi essa
separacao, ausente nos dados brasileiros, que manteve Petropolis bloqueado.

CADEIA, na ordem:
  1. desaninha o zip e le as camadas vetoriais
  2. deriva a AOI e escolhe o UTM local automaticamente
  3. baixa so os tiles globais que cobrem a AOI (DEM, HAND, WorldCover, GSW)
  4. reprojeta para o UTM local a 30 m e deriva slope e TWI D-infinity
  5. amostra pontos por classe, com teto por AOI (anti-pseudorreplicacao)
  6. le a janela de chuva CHIRPS v3 para a data do evento e os 14 dias antes
  7. grava tabela com o mesmo esquema do piloto ingles

REGRAS HERDADAS, que nao sao renegociaveis aqui:
  - agua permanente sai antes (R3 do contrato de reducao a pontos)
  - unidade de analise e o EVENTO/AOI, nao o pixel (U1-U3 da adjudicacao)
  - resolucao nativa fica registrada por linha (R2)
  - nada e promovido a rotulo definitivo: a coluna se chama `classe_obs`

NAO faz: nao treina, nao avalia, nao junta com o dataset brasileiro.

Uso:
    python scripts/externo/cems02_pipeline_ativacao.py --zip <arquivo.zip> --codigo EMSR851
    python scripts/externo/cems02_pipeline_ativacao.py --zip <...> --codigo <...> --so-geometria
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import sys
import time
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
CACHE = RUNS / "cache-global"      # tiles globais compartilhados entre ativacoes
RES = 30.0
# Teto por AOI E por classe. Ate 09/08/2026 era por ATIVACAO, o que produzia
# 300 pontos por classe para a ativacao inteira e, pior, um unico `grupo_cv`
# para todas as AOIs. Com 5 ativacoes isso dava 5 grupos -- EPV de 1,67, abaixo
# do minimo de 10 da regra do projeto, tornando o modelo ininterpretavel.
# Agora cada AOI e um grupo proprio: as mesmas 5 ativacoes rendem entre 25 e 40
# grupos, sem adquirir nenhum dado novo.
MAX_POR_AOI = 120
DIAS_ANTES = 14
AGUA_LIMIAR = 50
SEMENTE = 20260807

CAMADAS = ["areaOfInterestA", "observedEventA", "imageFootprintA"]


# ----------------------------------------------------------------- utilitarios

def utm_local(lon: float, lat: float) -> int:
    """EPSG do UTM que contem o ponto. Trabalhar em graus deformaria slope/TWI."""
    zona = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zona


def baixar(url: str, destino: Path) -> bool:
    import requests

    if destino.exists() and destino.stat().st_size > 0:
        return True
    destino.parent.mkdir(parents=True, exist_ok=True)
    for tentativa in range(4):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                tmp = destino.with_suffix(destino.suffix + ".part")
                with tmp.open("wb") as fh:
                    for ch in r.iter_content(chunk_size=1 << 20):
                        fh.write(ch)
                tmp.replace(destino)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2 ** tentativa)
    return False


def tiles_para(bbox: tuple[float, float, float, float]) -> dict[str, list]:
    """Monta a lista de tiles globais que cobrem o bbox (lon0, lat0, lon1, lat1)."""
    lon0, lat0, lon1, lat1 = bbox
    alvos: dict[str, list] = defaultdict(list)

    for la in range(math.floor(lat0), math.floor(lat1) + 1):
        for lo in range(math.floor(lon0), math.floor(lon1) + 1):
            ns, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
            base = f"{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00"
            t = f"Copernicus_DSM_COG_10_{base}_DEM"
            alvos["dem"].append((
                f"https://copernicus-dem-30m.s3.amazonaws.com/{t}/{t}.tif",
                CACHE / "dem" / f"{t}.tif"))
            h = f"Copernicus_DSM_COG_10_{base}_HAND"
            alvos["hand"].append((
                f"https://glo-30-hand.s3.amazonaws.com/v1/2021/{h}.tif",
                CACHE / "hand" / f"{h}.tif"))

    for la in range(int(math.floor(lat0 / 3) * 3), int(math.floor(lat1 / 3) * 3) + 1, 3):
        for lo in range(int(math.floor(lon0 / 3) * 3), int(math.floor(lon1 / 3) * 3) + 1, 3):
            ns, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
            t = f"{ns}{abs(la):02d}{ew}{abs(lo):03d}"
            alvos["worldcover"].append((
                "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                f"ESA_WorldCover_10m_2021_v200_{t}_Map.tif",
                CACHE / "worldcover" / f"{t}.tif"))

    lo10 = int(math.floor(lon0 / 10) * 10)
    la10 = int(math.ceil(lat1 / 10) * 10)
    ew = "W" if lo10 < 0 else "E"
    ns = "N" if la10 >= 0 else "S"
    nome = f"occurrence_{abs(lo10)}{ew}_{abs(la10)}{ns}"
    alvos["gsw"].append((
        "https://storage.googleapis.com/global-surface-water/downloads2021/"
        f"occurrence/{nome}v1_4_2021.tif", CACHE / "gsw" / f"{nome}.tif"))
    return alvos


def codigo_aoi(nome: str) -> str:
    """
    Extrai o identificador da AOI do nome do pacote.

    Os pacotes seguem `EMSR720_AOI01_DEL_PRODUCT_v1`. A AOI e a unidade de
    analise correta: pontos da mesma AOI compartilham evento, data, imagem e
    analista, e nao sao independentes entre si. Pontos de AOIs diferentes da
    mesma ativacao sao muito mais independentes -- podem estar a centenas de
    quilometros um do outro.
    """
    m = re.search(r"_(AOI\d+)_", nome)
    return m.group(1) if m else "AOI00"


def extrair(zip_externo: Path, destino: Path) -> dict[str, list[tuple[Path, str]]]:
    destino.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(zip_externo)
    por_camada: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for pacote in sorted(set(z.namelist())):
        if not pacote.lower().endswith(".zip"):
            continue
        interno = zipfile.ZipFile(io.BytesIO(z.read(pacote)))
        pasta = destino / Path(pacote).stem
        pasta.mkdir(parents=True, exist_ok=True)
        interno.extractall(pasta)
        aoi = codigo_aoi(Path(pacote).stem)
        for shp in pasta.rglob("*.shp"):
            for chave in CAMADAS:
                if chave in shp.name:
                    por_camada[chave].append((shp, aoi))
    return por_camada


def juntar(caminhos, crs_alvo: int):
    """Aceita lista de Path ou de (Path, codigo_aoi); anexa coluna `_aoi`."""
    import geopandas as gpd

    partes = []
    for item in caminhos:
        p, aoi = item if isinstance(item, tuple) else (item, "AOI00")
        try:
            g = gpd.read_file(p)
            if len(g):
                g = g.to_crs(crs_alvo)
                g["_aoi"] = aoi
                partes.append(g)
        except Exception:  # noqa: BLE001
            continue
    if not partes:
        return None
    return gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=crs_alvo)


# -------------------------------------------------------------------- rasters

MAX_CELULAS = 60_000_000   # ~240 MB por camada em float32


def preparar_rasters(bbox_ll, crs_metrico: int, bounds_m, outdir: Path) -> dict:
    """
    Baixa tiles, mosaica, reprojeta ao UTM local e deriva slope e TWI.

    IMPORTANTE: o bbox recebido deve corresponder a UMA AOI, nao ao conjunto
    de AOIs da ativacao. Ate 09/08/2026 esta funcao recebia o envelope de
    todas as AOIs juntas; no EMSR857 (Mocambique) as AOIs se espalham por
    4,3 x 7,2 graus, o que produziu uma grade de 26.558 x 14.726 = 391 milhoes
    de celulas quase todas vazias, e um MemoryError pedindo 16,7 GiB.
    Uma grade por AOI mantem cada matriz pequena e ainda coincide com a
    unidade de agrupamento adotada na validacao.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    from rasterio.warp import reproject

    alvos = tiles_para(bbox_ll)
    for chave, itens in alvos.items():
        ok = sum(1 for u, d in itens if baixar(u, d))
        print(f"   tiles {chave}: {ok}/{len(itens)}")

    x0, y0, x1, y1 = bounds_m
    x0, y0 = math.floor(x0 / RES) * RES - 5 * RES, math.floor(y0 / RES) * RES - 5 * RES
    x1, y1 = math.ceil(x1 / RES) * RES + 5 * RES, math.ceil(y1 / RES) * RES + 5 * RES
    largura, altura = int((x1 - x0) / RES), int((y1 - y0) / RES)
    if largura * altura > MAX_CELULAS:
        raise MemoryError(
            f"grade {altura}x{largura} = {largura*altura:,} celulas excede o "
            f"teto de {MAX_CELULAS:,}. O bbox recebido provavelmente cobre "
            f"varias AOIs -- processe uma AOI por vez.")
    transform = rasterio.transform.from_origin(x0, y1, RES, RES)
    print(f"   grade={altura}x{largura} crs=EPSG:{crs_metrico}")

    saida = {}
    for chave, reamostra, dtype in (("dem", Resampling.bilinear, "float32"),
                                    ("hand", Resampling.bilinear, "float32"),
                                    ("worldcover", Resampling.nearest, "float32"),
                                    ("gsw", Resampling.bilinear, "float32")):
        arquivos = [d for _, d in alvos[chave] if d.exists()]
        if not arquivos:
            saida[chave] = None
            continue
        fontes = [rasterio.open(a) for a in arquivos]
        try:
            arr, tr = merge(fontes, bounds=bbox_ll)
            perfil = fontes[0].profile.copy()
        finally:
            for f in fontes:
                f.close()
        destino = np.full((altura, largura), np.nan, dtype="float32")
        reproject(source=arr[0].astype("float32"), destination=destino,
                  src_transform=tr, src_crs=perfil["crs"],
                  dst_transform=transform, dst_crs=f"EPSG:{crs_metrico}",
                  resampling=reamostra, src_nodata=perfil.get("nodata"),
                  dst_nodata=np.nan)
        saida[chave] = destino

    z = saida.get("dem")
    if z is not None:
        dzdy, dzdx = np.gradient(z, RES, RES)
        saida["slope"] = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype("float32")
        try:
            import pyproj
            from pysheds.grid import Grid
            from pysheds.view import Raster, ViewFinder

            vf = ViewFinder(shape=z.shape, affine=transform,
                            crs=pyproj.Proj(f"EPSG:{crs_metrico}", preserve_units=True),
                            nodata=np.nan)
            dem = Raster(np.nan_to_num(z, nan=float(np.nanmin(z))).astype("float64"),
                         viewfinder=vf)
            grid = Grid.from_raster(dem)
            d = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
            acc = grid.accumulation(grid.flowdir(d, routing="dinf"), routing="dinf")
            a = (np.asarray(acc, dtype="float64") + 1.0) * RES
            tanb = np.tan(np.radians(np.clip(saida["slope"], 0.05, None)))
            saida["twi"] = np.log(a / tanb).astype("float32")
        except Exception as exc:  # noqa: BLE001
            # NAO engolir: em 09/08/2026 este except transformou a ausencia do
            # `pysheds` em `twi_dinf` NaN nas SEIS ativacoes processadas, sem
            # nenhum aviso visivel. Falha silenciosa em feature e pior que
            # parada: produz dataset com forma correta e conteudo faltando.
            raise RuntimeError(
                f"TWI nao pode ser calculado ({type(exc).__name__}: {exc}). "
                "Verifique se `pysheds` esta instalado no ambiente: "
                "pip install pysheds"
            ) from exc
    saida["_transform"] = transform
    saida["_forma"] = (altura, largura)
    return saida


# ------------------------------------------------------------------ principal

def main() -> int:
    args = sys.argv[1:]
    if "--zip" not in args or "--codigo" not in args:
        print("uso: --zip <arquivo.zip> --codigo <EMSRxxx> [--so-geometria]")
        return 1
    caminho_zip = Path(args[args.index("--zip") + 1])
    codigo = args[args.index("--codigo") + 1]
    so_geometria = "--so-geometria" in args
    # limpeza e o padrao: o `extraido/` chega a alguns GB por ativacao e nao e
    # reaberto depois que os parquets e a tabela de pontos existem. O zip
    # original continua sendo a evidencia de proveniencia -- ele fica onde
    # estiver, fora do repositorio.
    manter_extraido = "--manter-extraido" in args

    import geopandas as gpd
    from rasterio.features import rasterize

    OUT = RUNS / f"cems-{codigo.lower()}"
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()
    print(f"===CEMS02_{codigo}===")

    camadas_p = extrair(caminho_zip, OUT / "extraido")
    for k, v in camadas_p.items():
        print(f"  {k}: {len(v)} shapefiles")
    if "areaOfInterestA" not in camadas_p or "observedEventA" not in camadas_p:
        print("ABORTADO: faltam camadas essenciais")
        return 2

    # CRS metrico escolhido pelo centro da AOI
    aoi_ll = juntar(camadas_p["areaOfInterestA"], 4326)
    c = aoi_ll.geometry.union_all().centroid
    crs_m = utm_local(c.x, c.y)
    bbox_ll = tuple(aoi_ll.total_bounds)
    print(f"CENTRO=({c.y:.3f}, {c.x:.3f}) UTM_LOCAL=EPSG:{crs_m}")
    print(f"BBOX_LL={tuple(round(v,4) for v in bbox_ll)}")

    aoi = juntar(camadas_p["areaOfInterestA"], crs_m)
    obs = juntar(camadas_p["observedEventA"], crs_m)
    fp = juntar(camadas_p.get("imageFootprintA", []), crs_m)

    u_aoi = aoi.geometry.union_all()
    observada = u_aoi.intersection(fp.geometry.union_all()) if fp is not None else u_aoi

    tipos = obs["event_type"].astype(str) if "event_type" in obs.columns else None
    if tipos is not None:
        flood = obs[tipos.str.contains("Flood", na=False)]
        massa = obs[tipos.str.contains("Mass", na=False)]
    else:
        flood, massa = obs, obs.iloc[0:0]
    u_flood = flood.geometry.union_all() if len(flood) else None
    u_massa = massa.geometry.union_all() if len(massa) else None

    negativo = observada.difference(u_flood) if u_flood is not None else observada
    if u_massa is not None:
        negativo = negativo.difference(u_massa)

    km2 = lambda g: (g.area / 1e6) if g is not None else 0.0  # noqa: E731
    print("---GEOMETRIA---")
    print(f"AOI={km2(u_aoi):.1f}km2 OBSERVADA={km2(observada):.1f}km2")
    print(f"INUNDACAO={km2(u_flood):.2f}km2 ({len(flood)} feicoes)")
    print(f"MASSA={km2(u_massa):.2f}km2 ({len(massa)} feicoes)")
    print(f"NEGATIVO_OBSERVADO={km2(negativo):.2f}km2")

    for nome, g in (("aoi", u_aoi), ("area_observada", observada),
                    ("inundacao", u_flood), ("massa", u_massa),
                    ("negativo_observado", negativo)):
        if g is not None:
            gpd.GeoDataFrame(geometry=[g], crs=crs_m).to_parquet(
                OUT / f"{nome}.parquet", index=False)

    if so_geometria:
        print("MODO=SO_GEOMETRIA (sem rasters, sem pontos)")
        print("===END===")
        return 0

    # ------------------------------------------------------------------
    # UMA GRADE POR AOI. A versao anterior montava uma grade unica para o
    # envelope de todas as AOIs; no EMSR857 isso pediu 16,7 GiB. Aqui cada
    # AOI e processada isoladamente, o que mantem as matrizes pequenas e
    # coincide com a unidade de agrupamento da validacao.
    # ------------------------------------------------------------------
    print("---RASTERS POR AOI---")
    codigos_aoi = sorted(aoi["_aoi"].unique()) if "_aoi" in aoi.columns else ["AOI00"]
    print(f"  AOIs a processar: {len(codigos_aoi)}")

    from pyproj import Transformer
    para_ll = Transformer.from_crs(crs_m, 4326, always_xy=True)
    linhas, falhas_aoi = [], []

    for n_aoi, cod in enumerate(codigos_aoi, 1):
        sub_aoi = aoi[aoi["_aoi"] == cod] if "_aoi" in aoi.columns else aoi
        g_aoi = sub_aoi.geometry.union_all()
        obs_aoi = observada.intersection(g_aoi)
        if obs_aoi.is_empty:
            continue
        b = obs_aoi.bounds
        lo0, la0 = para_ll.transform(b[0], b[1])
        lo1, la1 = para_ll.transform(b[2], b[3])
        margem = 0.05
        bbox_aoi = (lo0 - margem, la0 - margem, lo1 + margem, la1 + margem)

        try:
            r = preparar_rasters(bbox_aoi, crs_m, obs_aoi.bounds, OUT)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{cod}] FALHOU {type(exc).__name__}: {str(exc)[:110]}")
            falhas_aoi.append(f"{cod}: {type(exc).__name__}")
            continue

        transform, forma = r["_transform"], r["_forma"]

        def mascara(geom, forma=forma, transform=transform):
            if geom is None or geom.is_empty:
                return np.zeros(forma, dtype="uint8")
            gs = geom.geoms if hasattr(geom, "geoms") else [geom]
            return rasterize(((g, 1) for g in gs if not g.is_empty),
                             out_shape=forma, transform=transform, fill=0,
                             dtype="uint8", all_touched=True)

        m_obs = mascara(obs_aoi)
        m_flood = mascara(u_flood.intersection(g_aoi) if u_flood is not None else None)
        m_massa = mascara(u_massa.intersection(g_aoi) if u_massa is not None else None)
        agua = (r["gsw"] > AGUA_LIMIAR) if r.get("gsw") is not None else np.zeros(forma, bool)

        classe = np.zeros(forma, dtype="uint8")
        classe[m_obs > 0] = 3
        classe[m_massa > 0] = 2
        classe[m_flood > 0] = 1
        classe[agua & (classe != 1)] = 4
        nomes = {1: "INUNDACAO", 2: "MOVIMENTO_DE_MASSA",
                 3: "NAO_INUNDADO_OBSERVADO", 4: "AGUA_PERMANENTE"}

        tr_ll = Transformer.from_crs(crs_m, 4326, always_xy=True)
        for k, nome in nomes.items():
            sel = np.nonzero(classe == k)
            n = len(sel[0])
            if n == 0:
                continue
            idx = rng.choice(n, size=min(MAX_POR_AOI, n), replace=False)
            rr, cc = sel[0][idx], sel[1][idx]
            x_m = transform.c + (cc + 0.5) * RES
            y_m = transform.f - (rr + 0.5) * RES
            lon, lat = tr_ll.transform(x_m, y_m)
            bloco = pd.DataFrame({
                "classe_obs": nome, "aoi_id": cod,
                "x_m": x_m, "y_m": y_m, "lon": lon, "lat": lat,
            })
            for nome_col, chave in (("elevation_m", "dem"), ("hand_m", "hand"),
                                    ("slope_deg", "slope"), ("twi_dinf", "twi"),
                                    ("worldcover", "worldcover")):
                arr = r.get(chave)
                bloco[nome_col] = arr[rr, cc] if arr is not None else np.nan
            linhas.append(bloco)
        if n_aoi % 5 == 0:
            print(f"  {n_aoi}/{len(codigos_aoi)} AOIs processadas", flush=True)

    if not linhas:
        print("ABORTADO: nenhuma AOI produziu pontos")
        return 3
    df = pd.concat(linhas, ignore_index=True)
    if falhas_aoi:
        print(f"  AOIs_COM_FALHA={len(falhas_aoi)}: {'; '.join(falhas_aoi[:6])}")
    for k, sub in df.groupby("classe_obs"):
        print(f"  {k}: {len(sub)} pontos em {sub.aoi_id.nunique()} AOIs")

    df["ativacao"] = codigo
    # grupo de validacao cruzada = ativacao + AOI. Este e o conserto central:
    # antes toda a ativacao era um grupo so.
    df["grupo_cv"] = codigo + "_" + df["aoi_id"].astype(str)
    df["crs_metrico"] = crs_m
    df["resolucao_m"] = RES
    df["fonte"] = "copernicus_ems_rapid_mapping"
    df["licenca"] = "Copernicus - uso livre com atribuicao"
    df["ponto_id"] = [f"{codigo}_{i:06d}" for i in range(len(df))]

    # --- verificacao antes de gravar: feature vazia e erro, nao aviso ---
    problemas = []
    for f in ("elevation_m", "slope_deg", "hand_m", "twi_dinf"):
        n_ok = int(df[f].notna().sum())
        if n_ok == 0:
            problemas.append(f"{f} inteiramente vazia")
        elif n_ok < 0.5 * len(df):
            problemas.append(f"{f} com so {n_ok}/{len(df)} preenchidos")
    if problemas:
        print("ABORTADO -- verificacao de features falhou:")
        for p_ in problemas:
            print(f"   {p_}")
        return 4

    saida = OUT / f"pontos_{codigo.lower()}.csv"
    df.to_csv(saida, index=False)
    print("---AMOSTRA---")
    print(f"N={len(df):,} AOIs={df['aoi_id'].nunique()} grupos_cv={df['grupo_cv'].nunique()}")
    for k, v in df["classe_obs"].value_counts().items():
        print(f"  {k}: {v}")
    print("---MEDIANAS POR CLASSE---")
    for f in ("elevation_m", "slope_deg", "hand_m", "twi_dinf"):
        med = df.groupby("classe_obs")[f].median().round(2).to_dict()
        print(f"  {f}: {med}")

    # --- limpeza: so o util fica no repositorio ---
    if not manter_extraido:
        import shutil
        alvo = OUT / "extraido"
        antes = sum(f.stat().st_size for f in alvo.rglob("*") if f.is_file()) / 1024**2
        try:
            shutil.rmtree(alvo)
            print(f"LIMPEZA=removido extraido/ ({antes:.0f}MB)")
        except Exception as exc:  # noqa: BLE001
            print(f"LIMPEZA=falhou {type(exc).__name__} -- remova {alvo} a mao")

    (OUT / "PROVENIENCIA.md").write_text(
        f"# Proveniencia - Copernicus EMS {codigo}\n"
        f"- Fonte: Copernicus Emergency Management Service, Rapid Mapping\n"
        f"- Ativacao: {codigo}\n"
        f"- Pacote de origem: {caminho_zip.name}\n"
        f"- Data de processamento: {time.strftime('%Y-%m-%d %H:%M')}\n"
        f"- Licenca: Copernicus, uso livre com atribuicao a European Union\n"
        f"- CRS metrico: EPSG:{crs_m} (UTM local, escolhido pelo centro da AOI)\n"
        f"- Natureza: OBSERVADO. `areaOfInterest` declara o que foi analisado;\n"
        f"  `observedEvent` declara o que foi detectado dentro dela.\n"
        f"- Negativo: (AOI INTERSECAO footprint) MENOS inundacao MENOS massa.\n"
        f"- Movimento de massa e CLASSE PROPRIA: nem positivo, nem negativo.\n"
        f"- Os shapefiles extraidos foram removidos apos o processamento; o zip\n"
        f"  original permanece como evidencia de aquisicao.\n"
        f"- Status: CANDIDATO. Nenhum rotulo promovido, nenhum modelo treinado.\n",
        encoding="utf-8")

    (OUT / "resumo.json").write_text(json.dumps({
        "ativacao": codigo, "crs_metrico": crs_m, "bbox_ll": list(bbox_ll),
        "area_aoi_km2": round(km2(u_aoi), 2),
        "area_observada_km2": round(km2(observada), 2),
        "area_inundacao_km2": round(km2(u_flood), 2),
        "area_massa_km2": round(km2(u_massa), 2),
        "area_negativo_km2": round(km2(negativo), 2),
        "n_pontos": int(len(df)),
        "por_classe": {k: int(v) for k, v in df["classe_obs"].value_counts().items()},
        "n_aois": int(df["aoi_id"].nunique()),
        "n_grupos_cv": int(df["grupo_cv"].nunique()),
        "por_aoi": {k: int(v) for k, v in df["aoi_id"].value_counts().items()},
        "max_por_aoi_por_classe": MAX_POR_AOI,
        "aviso": ("`classe_obs` e a classe DECLARADA pelo produto Copernicus, "
                  "nao rotulo promovido. Movimento de massa e classe propria: "
                  "nem positivo de inundacao, nem negativo."),
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SAIDA={saida.name} TEMPO={round(time.time()-t0,1)}s")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
