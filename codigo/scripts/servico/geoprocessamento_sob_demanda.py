"""
Ingestao sob demanda: camadas para coordenada fora da grade das tres regioes.

ONDE ISTO ENCAIXA:

`resolver_camadas.py` preenche `camadas` pela celula da grade de 120 m das tres
regioes conhecidas. Ponto fora de toda celula servivel volta com
`pontos_fora_de_toda_celula_servivel > 0`, e ate aqui o contrato so podia
recusar. Este modulo e o caminho (c) que aquele arquivo deixou de fora
conscientemente -- e a fronteira ja tinha sido desenhada para isto: o
`resolver()` nao muda, o `svc05` chama este modulo SO quando o resolvedor
reporta ponto de fora.

-------------------------------------------------------------------------
A CADEIA TEM DE SER A MESMA, e isso nao e detalhe
-------------------------------------------------------------------------

WhiteboxTools, D-infinity, 30 m, limiar de canal 0,1123 km2. Chuva por
Open-Meteo/ERA5-Land com a mesma formula de decaimento.

Derivar por outra cadeia produziria um numero com o nome certo e o significado
errado -- e o projeto ja pagou por isso duas vezes: o cartao de resposta exibia
HAND de 10 m ao lado de escore calculado com HAND de 30 m, e antes disso a
coluna `hand_m` chegou a misturar produto global com cadeia propria. Aqui seria
pior que nas duas, porque entraria direto no escore de uma area que ninguem
conferiu.

Por isso o modulo declara `cadeia` em toda saida, e ha teste que falha se a
constante divergir da que o SVC-03 usa.

-------------------------------------------------------------------------
EGRESSO E OPT-IN, e o dry-run e o padrao
-------------------------------------------------------------------------

O repositorio e fail-closed por regra e nenhum teste toca a rede. Sem
`--online` (ou `online=True`), este modulo NAO baixa nada: ele monta o plano de
aquisicao e devolve o que baixaria, com URL, tamanho estimado e janela de
chuva. E util assim -- da para auditar o plano antes de gastar rede.

-------------------------------------------------------------------------
O QUE ELE NAO RESOLVE, e precisa estar escrito
-------------------------------------------------------------------------

Ter as camadas nao torna a area servivel. Depois de derivar, o ponto ainda
passa pelos cinco portoes, e o G5 (dominio coberto) e o que decide: uma AOI em
terreno que o modelo nunca viu recebe `insufficient_data` com a distancia de
dominio declarada, exatamente como Curitiba recebe hoje na elevacao.

Este modulo amplia de onde vem o dado. Ele nao amplia sobre o que o modelo pode
falar -- isso continua sendo decisao do portao, e e assim que tem de ser.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts"))

from chuva.contrato_chuva import (  # noqa: E402
    DECAY_K, LOOKBACK_DAYS, METHOD_VERSION, PRODUCT_ID, PROVIDER_ID,
    PROVIDER_URL, SPATIAL_RESOLUTION_DEGREES, TIMEZONE, requested_window,
)
from chuva.providers import OpenMeteoEra5LandProvider, RainRequest  # noqa: E402

RUNS = REPO / "local_runs"
OUT = RUNS / "ing-01-sob-demanda"
CACHE = RUNS / "cache-sob-demanda"

# A cadeia canonica. Duplicada do SVC-03 de proposito -- importar aquele modulo
# traria rasterio e leitura de raster para dentro de um modulo que precisa
# poder rodar em dry-run. Ha teste que falha se as duas divergirem.
CADEIA = "WhiteboxTools 30 m, limiar de canal 0,1123 km2"
LIMIAR_CANAL_KM2 = 0.1123
RESOLUCAO_M = 30

# Copernicus DEM GLO-30 pelo bucket ABERTO da AWS, e nao pelo Planetary
# Computer: o PC exige assinatura SAS (`planetary_computer.sign`), e sem ela os
# assets respondem HTTP 409. Testado -- foi assim que a primeira versao falhou.
# O bucket aberto nao pede credencial nenhuma.
DEM_COLECAO = "Copernicus DEM GLO-30 (bucket aberto da AWS)"
DEM_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"

# ATENCAO, e isto precisa viajar com o numero: o GLO-30 e um DSM, nao um DTM.
# Ele inclui dossel e edificacao. O proprio projeto ja rejeitou o GLO-30 para
# Petropolis por essa razao e usou FABDEM no lugar (SUSC-20G-2). Derivar HAND
# de DSM em terreno vegetado ou urbano denso desloca a altura acima da
# drenagem para cima, e o escore sai otimista.
#
# Por isso toda saida deste modulo carrega `alerta_dsm`, e a AOI derivada aqui
# NAO e equivalente a uma regiao cujo terreno veio de MDT. E dado para
# explorar area nova, nao para promover a servida.
ALERTA_DSM = (
    "GLO-30 e DSM (inclui dossel e edificacao), nao MDT. O projeto ja o "
    "rejeitou em Petropolis por isso e usou FABDEM. HAND derivado aqui tende a "
    "ficar alto em area vegetada ou urbana densa, e o escore, otimista")

# Importados do contrato unico; aliases preservam compatibilidade publica.
CHUVA_URL = PROVIDER_URL
LOOKBACK_DIAS = LOOKBACK_DAYS

# Um pedido sem limite baixa o planeta. A AOI e recortada antes de qualquer
# download, e AOI grande demais e recusada em vez de fatiada em silencio.
MAX_LADO_KM = 25.0
MARGEM_KM = 3.0   # bacia a montante nao cabe na propria caixa do ponto


def _graus_por_km(lat: float) -> tuple[float, float]:
    return 1 / 110.574, 1 / (111.320 * max(math.cos(math.radians(lat)), 1e-6))


def caixa_da_aoi(pontos: list[dict], margem_km: float = MARGEM_KM) -> dict:
    """Caixa envolvente com margem.

    A margem nao e folga arbitraria: HAND e TWI dependem da area contribuinte
    A MONTANTE, que quase nunca cabe na caixa dos pontos. Recortar justo daria
    um HAND calculado sobre uma bacia truncada -- numero plausivel e errado,
    que e o pior tipo.
    """
    lats = [float(p["lat"]) for p in pontos]
    lons = [float(p["lon"]) for p in pontos]
    dlat, dlon = _graus_por_km(sum(lats) / len(lats))
    return {
        "lat_min": min(lats) - margem_km * dlat,
        "lat_max": max(lats) + margem_km * dlat,
        "lon_min": min(lons) - margem_km * dlon,
        "lon_max": max(lons) + margem_km * dlon,
        "margem_km": margem_km,
        "por_que_margem": ("HAND e TWI dependem da area contribuinte a "
                           "montante, que nao cabe na caixa dos pontos"),
    }


def _lado_km(caixa: dict) -> tuple[float, float]:
    lat_med = (caixa["lat_min"] + caixa["lat_max"]) / 2
    dlat, dlon = _graus_por_km(lat_med)
    return ((caixa["lat_max"] - caixa["lat_min"]) / dlat,
            (caixa["lon_max"] - caixa["lon_min"]) / dlon)


def planejar(pontos: list[dict], data_evento: str | None = None) -> dict:
    """O que seria baixado. Nao baixa nada -- e o dry-run."""
    caixa = caixa_da_aoi(pontos)
    alt_km, larg_km = _lado_km(caixa)
    cabe = alt_km <= MAX_LADO_KM and larg_km <= MAX_LADO_KM

    referencia = date.fromisoformat(data_evento) if data_evento else date.today()
    inicio_s, fim_s = requested_window(referencia)
    lat_c = (caixa["lat_min"] + caixa["lat_max"]) / 2
    lon_c = (caixa["lon_min"] + caixa["lon_max"]) / 2

    plano = {
        "cadeia": CADEIA,
        "resolucao_m": RESOLUCAO_M,
        "limiar_de_canal_km2": LIMIAR_CANAL_KM2,
        "n_pontos": len(pontos),
        "caixa": caixa,
        "lado_km": {"altura": round(alt_km, 2), "largura": round(larg_km, 2)},
        "cabe_no_limite": cabe,
        "limite_lado_km": MAX_LADO_KM,
        "terreno": {
            "fonte": DEM_COLECAO,
            "bucket": DEM_BUCKET,
            "tiles": _urls_dos_tiles(caixa),
            "crs_de_derivacao": epsg_utm(
                (caixa["lat_min"] + caixa["lat_max"]) / 2,
                (caixa["lon_min"] + caixa["lon_max"]) / 2),
            "alerta_dsm": ALERTA_DSM,
            "deriva": ["elevation_m", "slope_deg", "hand_m", "twi_dinf"],
            "ferramenta": "WhiteboxTools (mesma do ter01/svc03)",
        },
        "chuva": {
            "provider": PROVIDER_ID,
            "product": PRODUCT_ID,
            "fonte": "Open-Meteo / ERA5-Land archive",
            "url": CHUVA_URL,
            "ponto": {"lat": round(lat_c, 5), "lon": round(lon_c, 5)},
            "janela": {"inicio": inicio_s, "fim": fim_s,
                       "data_referencia": referencia.isoformat(),
                       "dia_do_evento_incluido": False},
            "decay_k": DECAY_K,
            "timezone": TIMEZONE,
            "spatial_resolution_degrees": SPATIAL_RESOLUTION_DEGREES,
            "method_version": METHOD_VERSION,
            "deriva": ["rain_max_24h", "rain_decay_index"],
            "nota": ("um ponto so para a AOI inteira: a celula do produto tem "
                     "~11 km e a chuva nao varia na escala da AOI"),
        },
        "executado": False,
        "por_que_nao_executado": (
            "dry-run e o padrao; o repositorio e fail-closed e nenhum teste "
            "toca a rede. Passe --online (ou online=True) para adquirir"),
    }
    if not cabe:
        plano["recusado"] = (
            f"AOI de {alt_km:.1f} x {larg_km:.1f} km acima do limite de "
            f"{MAX_LADO_KM} km. Fatiar em silencio faria a derivacao rodar "
            "sobre bacias truncadas")
    return plano


# ----------------------------------------------------------- aquisicao real

def _buscar_chuva(lat: float, lon: float, inicio: str, fim: str) -> dict:
    """Acquire/cache first, then derive under the versioned rain contract."""
    provider = OpenMeteoEra5LandProvider(CACHE / "chuva")
    acquisition = provider.acquire(RainRequest(
        latitude=float(lat), longitude=float(lon), start_date=inicio,
        end_date=fim, timezone=TIMEZONE), online=True)
    return provider.derive(acquisition)


def epsg_utm(lat: float, lon: float) -> str:
    """UTM da AOI. Derivar em grau distorce declividade e area contribuinte.

    O registro de auditoria da cadeia mostra as regioes derivadas em
    EPSG:327xx / 319xx a 30 m -- Recife em 32725, Curitiba em 32722. Rodar o
    WhiteboxTools sobre coordenada geografica daria declividade em unidade que
    nao e grau de terreno e area contribuinte em grau quadrado. O numero sairia
    com o nome certo e o significado errado.
    """
    zona = int((lon + 180) // 6) + 1
    return f"EPSG:{(32700 if lat < 0 else 32600) + zona}"


def _urls_dos_tiles(caixa: dict) -> list[str]:
    """Tiles de 1 grau que cobrem a caixa, no padrao de nome do bucket."""
    urls = []
    for la in range(math.floor(caixa["lat_min"]), math.floor(caixa["lat_max"]) + 1):
        for lo in range(math.floor(caixa["lon_min"]),
                        math.floor(caixa["lon_max"]) + 1):
            ns, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
            nome = (f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_"
                    f"{ew}{abs(lo):03d}_00_DEM")
            urls.append(f"{DEM_BUCKET}/{nome}/{nome}.tif")
    return urls


def _baixar_dem(caixa: dict, destino: Path) -> tuple[Path, dict]:
    """Recorta o GLO-30 pela caixa e REPROJETA para UTM a 30 m."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    destino.mkdir(parents=True, exist_ok=True)
    alvo = destino / "dem_utm30.tif"
    urls = _urls_dos_tiles(caixa)
    if alvo.exists():
        return alvo, {"tiles": urls, "reaproveitado_do_cache": True}

    bbox = (caixa["lon_min"], caixa["lat_min"],
            caixa["lon_max"], caixa["lat_max"])
    abertos, falhas = [], []
    for u in urls:
        try:
            abertos.append(rasterio.open(u))
        except Exception as e:
            falhas.append({"url": u, "erro": f"{type(e).__name__}: {e}"[:120]})
    if not abertos:
        raise RuntimeError(f"nenhum tile GLO-30 acessivel para {bbox}: {falhas}")
    try:
        arranjo, transform = merge(abertos, bounds=bbox)
        perfil = abertos[0].profile
        crs_origem = abertos[0].crs
    finally:
        for a in abertos:
            a.close()

    destino_crs = epsg_utm((caixa["lat_min"] + caixa["lat_max"]) / 2,
                           (caixa["lon_min"] + caixa["lon_max"]) / 2)
    largura, altura = arranjo.shape[2], arranjo.shape[1]
    t_dst, w_dst, h_dst = calculate_default_transform(
        crs_origem, destino_crs, largura, altura,
        *rasterio.transform.array_bounds(altura, largura, transform),
        resolution=(RESOLUCAO_M, RESOLUCAO_M))

    perfil.update(driver="GTiff", crs=destino_crs, transform=t_dst,
                  width=w_dst, height=h_dst, count=1, dtype="float32")
    with rasterio.open(alvo, "w", **perfil) as dst:
        reproject(source=arranjo[0], destination=rasterio.band(dst, 1),
                  src_transform=transform, src_crs=crs_origem,
                  dst_transform=t_dst, dst_crs=destino_crs,
                  resampling=Resampling.bilinear)
    return alvo, {"tiles": urls, "tiles_lidos": len(urls) - len(falhas),
                  "falhas": falhas, "crs": destino_crs,
                  "resolucao_m": RESOLUCAO_M,
                  "reaproveitado_do_cache": False}


_MAX_PROCS_PINADO = False

# Mesmos parametros do `terreno_invariantes`: tres tentativas contra um panico
# de ~2,5% por chamada, e espera curta para a saida ficar visivel.
TENTATIVAS_POR_FERRAMENTA = 3
ESPERAS_PELA_SAIDA = 30
INTERVALO_DE_ESPERA_S = 0.1


def _derivar_terreno(dem: Path, destino: Path) -> dict:
    """HAND, TWI, declividade pela cadeia canonica -- WhiteboxTools D-infinity."""
    import whitebox

    wbt = whitebox.WhiteboxTools()
    # A MESMA disciplina do `terreno_invariantes`, e aqui ela pesa mais: este e
    # o caminho que serve AOI nova.
    #
    #   `run_tool` devolve 0 SEMPRE -- drena o stdout e retorna, sem
    #   `proc.wait()`. Com `verbose = False` o callback nem e chamado, entao a
    #   mensagem de erro do binario era descartada. Chamar sete ferramentas sem
    #   conferir nada significava servir HAND de uma cadeia que pode ter
    #   quebrado no meio, sem sinal nenhum.
    #
    # O `FillDepressions` do WBT 2.4.0 entra em panico em ~2,5% das chamadas
    # sobre entrada identica (`fill_depressions.rs:368`, "Error unwrapping
    # 'output'"). Numa cadeia de sete ferramentas isso e ~16% de chance de
    # alguma ingestao quebrar.
    saida_do_wbt: list[str] = []
    repeticoes: list[dict] = []
    wbt.verbose = True
    wbt.default_callback = saida_do_wbt.append
    # UMA vez por processo: `set_max_procs` lanca o executavel de novo e faz
    # `os.chdir` global, e o valor fica gravado no proprio WhiteboxTools.
    global _MAX_PROCS_PINADO
    if not _MAX_PROCS_PINADO:
        wbt.set_max_procs(1)
        _MAX_PROCS_PINADO = True
    wbt.set_working_dir(str(destino))

    def rodar(nome: str, saida: Path, *args, **kwargs) -> Path:
        for tentativa in range(1, TENTATIVAS_POR_FERRAMENTA + 1):
            saida_do_wbt.clear()
            if saida.exists():
                saida.unlink()
            getattr(wbt, nome)(*args, **kwargs)
            for _ in range(ESPERAS_PELA_SAIDA):
                if saida.exists():
                    if tentativa > 1:
                        repeticoes.append({"ferramenta": nome,
                                           "tentativas": tentativa})
                    return saida
                time.sleep(INTERVALO_DE_ESPERA_S)
        relato = " | ".join(l for l in saida_do_wbt if l.strip())[-1200:]
        raise RuntimeError(
            f"WhiteboxTools.{nome} nao produziu {saida.name} em "
            f"{TENTATIVAS_POR_FERRAMENTA} tentativas. "
            f"stdout do binario: {relato or '(vazio)'}")

    preenchido = rodar("fill_depressions", destino / "dem_filled.tif",
                       str(dem), str(destino / "dem_filled.tif"))
    declividade = rodar("slope", destino / "slope_deg_wbt.tif",
                        str(preenchido), str(destino / "slope_deg_wbt.tif"),
                        units="degrees")
    facc = rodar("d_inf_flow_accumulation", destino / "facc_dinf.tif",
                 str(preenchido), str(destino / "facc_dinf.tif"),
                 out_type="specific contributing area")
    rodar("d_inf_pointer", destino / "pointer_dinf.tif", str(preenchido),
          str(destino / "pointer_dinf.tif"))
    rodar("wetness_index", destino / "twi_dinf.tif", str(facc),
          str(declividade), str(destino / "twi_dinf.tif"))
    # canal pelo mesmo limiar de area contribuinte da cadeia
    streams = rodar("extract_streams", destino / "streams.tif", str(facc),
                    str(destino / "streams.tif"),
                    threshold=LIMIAR_CANAL_KM2 * 1e6)
    rodar("elevation_above_stream", destino / "hand_dinf.tif",
          str(preenchido), str(streams), str(destino / "hand_dinf.tif"))
    return {
        "repeticoes_do_binario": repeticoes,
        "elevation_m": str(preenchido),
        "slope_deg": str(destino / "slope_deg_wbt.tif"),
        "hand_m": str(destino / "hand_dinf.tif"),
        "twi_dinf": str(destino / "twi_dinf.tif"),
    }


def _amostrar(rasters: dict, pontos: list[dict]) -> list[dict]:
    """Amostra os rasters NO CRS DELES, e nao em grau.

    A primeira versao passava lon/lat direto para `src.sample()` sobre raster
    em UTM. As coordenadas caiam fora da extensao e todas as quatro variaveis
    voltavam `null` -- silenciosamente, porque amostra fora do raster e nodata
    e nao erro. E o mesmo tipo de descuido que o cabecalho deste modulo
    descreve: numero com o nome certo e o significado errado, so que aqui o
    valor nem chegava a existir.
    """
    import rasterio
    from rasterio.warp import transform as transformar

    saida: list[dict] = [{} for _ in pontos]
    lons = [float(p["lon"]) for p in pontos]
    lats = [float(p["lat"]) for p in pontos]
    for var, caminho in rasters.items():
        with rasterio.open(caminho) as src:
            xs, ys = transformar("EPSG:4326", src.crs, lons, lats)
            for i, v in enumerate(src.sample(zip(xs, ys))):
                valor = float(v[0])
                fora = (src.nodata is not None and valor == src.nodata)
                saida[i][var] = None if fora else round(valor, 6)
    return saida


def adquirir(pontos: list[dict], data_evento: str | None = None,
             online: bool = False, destino: Path = CACHE) -> dict:
    """Devolve o plano e, com `online=True`, as camadas derivadas."""
    plano = planejar(pontos, data_evento)
    if plano.get("recusado") or not online:
        return plano

    chave = f"{plano['caixa']['lat_min']:.3f}_{plano['caixa']['lon_min']:.3f}"
    pasta = destino / chave
    try:
        dem, proc = _baixar_dem(plano["caixa"], pasta)
        plano["procedencia_do_dem"] = proc
        rasters = _derivar_terreno(dem, pasta)
        camadas = _amostrar(rasters, pontos)
        chuva = _buscar_chuva(plano["chuva"]["ponto"]["lat"],
                              plano["chuva"]["ponto"]["lon"],
                              plano["chuva"]["janela"]["inicio"],
                              plano["chuva"]["janela"]["fim"])
        for c in camadas:
            c.update(chuva)
        plano.update({"executado": True, "camadas": camadas,
                      "cache": str(pasta), "chuva_derivada": chuva,
                      "alerta_dsm": ALERTA_DSM})
        plano.pop("por_que_nao_executado", None)
    except Exception as e:
        plano.update({"executado": False,
                      "erro": f"{type(e).__name__}: {e}",
                      "por_que_nao_executado": "a aquisicao falhou; ver `erro`"})
    return plano


def main() -> int:
    online = "--online" in sys.argv
    lat = lon = None
    if "--ponto" in sys.argv:
        bruto = sys.argv[sys.argv.index("--ponto") + 1]
        lat, lon = (float(x) for x in bruto.split(","))
    if lat is None:
        print("uso: --ponto LAT,LON [--data AAAA-MM-DD] [--online]")
        return 2
    data = (sys.argv[sys.argv.index("--data") + 1]
            if "--data" in sys.argv else None)

    print("===INGESTAO SOB DEMANDA===")
    print(f"modo: {'ONLINE (baixa dado)' if online else 'DRY-RUN (nao baixa)'}")
    r = adquirir([{"lat": lat, "lon": lon}], data, online=online)
    print(json.dumps(r, indent=2, ensure_ascii=False)[:2600])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ultimo_plano.json").write_text(
        json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGRAVADO={OUT}/ultimo_plano.json")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
