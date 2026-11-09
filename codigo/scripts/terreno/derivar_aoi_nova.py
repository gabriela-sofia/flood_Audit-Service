"""Etapa 2.2 -- uma AOI nova vira terreno pela cadeia canonica, e por nenhuma outra.

-------------------------------------------------------------------------
QUATRO CAMADAS VEM DO SERVICO, E A QUINTA NAO PODE VIR
-------------------------------------------------------------------------

`_derivar_terreno` do `geoprocessamento_sob_demanda` e o MESMO codigo que o
servico usa para atender AOI sob demanda, e ele e reusado aqui ate onde da.
Derivando o mesmo DEM pelos dois caminhos e comparando com o que o `ter-01`
guardou:

    dem_filled          r = 1,000000   diferenca maxima 0
    slope_deg_wbt       r = 1,000000   diferenca maxima 0
    dinf_sca            r = 1,000000   ruido de ponto flutuante
    dinf_accum_cells    r = 1,000000   ruido de ponto flutuante
    hand_dinf           r = 0,928892   diferenca maxima 166 m

O HAND diverge porque o canal e extraido de outro jeito, e a divergencia ja
esta identificada no projeto -- `terreno_invariantes` a separa em MODO_SERVICO
e MODO_VALIDADO. Entao o canal e o HAND sao derivados aqui pela convencao do
`ter-01`, com o motivo escrito em `_canal_e_hand_do_ter01`, e o resto vem do
servico sem copia.

Este modulo acrescenta ainda o que falta entre uma coisa e outra: a reprojecao
para metros a 30 m, a camada `dinf_accum_cells` e o `run_manifest.json` no
formato que o `ter-01` grava -- sem ele `cadeia_declarada` recusa o diretorio,
e recusar e o comportamento certo.

-------------------------------------------------------------------------
POR QUE 30 m, E NAO A RESOLUCAO NATIVA
-------------------------------------------------------------------------

O limiar de canal de 0,1123 km2 vira um numero de CELULAS, e esse numero
depende do tamanho da celula. Rodar a cadeia a 1 m extrairia uma rede de
drenagem completamente diferente com o mesmo parametro escrito -- que e a
forma exata do erro que o `EMSR789_AOI06` ja tem em disco.

A agregacao usa MEDIA, que preserva o nivel medio do terreno na celula. Vizinho
mais proximo herdaria o valor de um pixel qualquer.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao baixa DEM. Ele recebe um raster que ja esta em disco, porque a fronteira
entre "derivar do que existe" e "sair buscando dado novo na rede" e uma
fronteira de licenca e de procedencia, e ela nao se cruza por conveniencia de
um script de treino.

Nao treina e nao promove: o terreno derivado entra no conjunto sem rotulo, e
quem decide retreinar e o gatilho, que so PROPOE.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "sar"))
sys.path.insert(0, str(REPO / "scripts" / "terreno"))

from dados_terreno import (  # noqa: E402
    CAMADAS, FORMULA_TWI_CANONICA, LIMIAR_CANAL_CANONICO_KM2, RAIZ,
    RESOLUCAO_CANONICA_M, cadeia_declarada,
)
from geoprocessamento_sob_demanda import _derivar_terreno  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402


# Quantas meias-voltas esperar por uma saida do WhiteboxTools.
ESPERAS_PELA_SAIDA = 120


class DerivacaoRecusada(Exception):
    """Falha dura. Terreno fora da cadeia nao entra no treino em silencio."""


def _utm_de(lon: float, lat: float) -> int:
    """EPSG do fuso UTM que contem o ponto. Norte 326xx, sul 327xx."""
    fuso = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + fuso


def _fonte_em_mosaico(src, caixa, mosaico):
    """Le, na grade da propria fonte, os recortes vizinhos que cobrem `caixa` (graus).

    Grade canonica v2 (10/10/2026): o retangulo UTM de um tile tem cantos fora da
    sua caixa em graus; sem vizinhos, esses cantos viravam NoData so naquele tile
    e as celulas de borda saiam da media de uma fracao do pixel (A2 da v1). Os
    recortes do acervo sao cortes da mesma grade FABDEM de 1"; recorte fora dessa
    grade e recusado, nao reamostrado.
    """
    import math
    import rasterio
    from rasterio.merge import merge
    rx, ry = src.res
    o = src.bounds.left + math.floor((caixa[0] - src.bounds.left) / rx) * rx
    n = src.bounds.top + math.ceil((caixa[3] - src.bounds.top) / ry) * ry
    l = o + math.ceil((caixa[2] - o) / rx) * rx
    s_ = n - math.ceil((n - caixa[1]) / ry) * ry
    usados, recusados = [], []
    for arq in mosaico:
        with rasterio.open(arq) as f:
            b = f.bounds
            if b.right <= o or b.left >= l or b.top <= s_ or b.bottom >= n:
                continue
            dx, dy = (b.left - src.bounds.left) / rx, (src.bounds.top - b.top) / ry
            # resolucao por tolerancia: recortes da mesma grade diferem em 1e-19 grau
            # no float (S04 vs S03, 10/10/2026), e igualdade exata os recusava
            mesma_res = all(abs(a - b) <= 1e-9 * abs(b) for a, b in zip(f.res, src.res))
            if (not mesma_res or f.crs != src.crs or abs(dx - round(dx)) > 1e-6
                    or abs(dy - round(dy)) > 1e-6):
                recusados.append(Path(arq).name)
                continue
        usados.append(arq)
    # a propria fonte primeiro: onde recortes se sobrepoem, valem os seus pixels
    arr, tr = merge([str(src.name), *[str(a) for a in usados if Path(a).resolve() != Path(src.name).resolve()]],
                    bounds=(o, s_, l, n), res=src.res, nodata=src.nodata, method="first")
    return arr[0], tr, {"recortes_usados": sorted({Path(a).name for a in usados} | {Path(src.name).name}),
                        "recortes_fora_da_grade": recusados, "caixa_graus": [o, s_, l, n]}


def reprojetar_para_30m(dem: Path, destino: Path,
                        epsg_metrico: int | None = None,
                        alinhar_m: float | None = None,
                        mosaico: list[Path] | None = None) -> dict:
    """DEM em metros, a 30 m, agregado por media.

    `alinhar_m` (grade canonica, 10/10/2026): a origem vira multiplo de
    `alinhar_m` no fuso, e dois recortes do mesmo fuso passam a ter celulas
    coincidentes. Sem ele, a origem vem da caixa do recorte (comportamento
    historico do ter-01), e recortes vizinhos ficam deslocados por fracao de
    celula -- medido em 100% dos pares do ter-01 em 09/10/2026.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    with rasterio.open(dem) as src:
        if epsg_metrico is None:
            oeste, sul, leste, norte = src.bounds
            if src.crs and src.crs.is_projected:
                from rasterio.warp import transform_bounds
                oeste, sul, leste, norte = transform_bounds(
                    src.crs, "EPSG:4326", oeste, sul, leste, norte)
            epsg_metrico = _utm_de((oeste + leste) / 2, (sul + norte) / 2)
        alvo = f"EPSG:{epsg_metrico}"
        transformacao, largura, altura = calculate_default_transform(
            src.crs, alvo, src.width, src.height, *src.bounds,
            resolution=RESOLUCAO_CANONICA_M)
        if alinhar_m:
            import math
            from rasterio.transform import from_origin
            x0, y1 = transformacao.c, transformacao.f
            x1 = x0 + transformacao.a * largura
            y0 = y1 + transformacao.e * altura
            x0a = math.floor(x0 / alinhar_m) * alinhar_m
            y1a = math.ceil(y1 / alinhar_m) * alinhar_m
            largura = int(math.ceil((x1 - x0a) / RESOLUCAO_CANONICA_M))
            altura = int(math.ceil((y1a - y0) / RESOLUCAO_CANONICA_M))
            transformacao = from_origin(x0a, y1a, RESOLUCAO_CANONICA_M,
                                        RESOLUCAO_CANONICA_M)
        perfil = src.profile.copy()
        perfil.update(crs=alvo, transform=transformacao, width=largura,
                      height=altura, dtype="float32", count=1,
                      nodata=-32768.0, compress="deflate")
        saida = np.full((altura, largura), -32768.0, dtype="float32")
        fonte, fonte_tr, info_mosaico = rasterio.band(src, 1), src.transform, None
        if mosaico:
            from rasterio.warp import transform_bounds
            x1 = transformacao.c + transformacao.a * largura
            y0 = transformacao.f + transformacao.e * altura
            caixa = transform_bounds(alvo, src.crs, transformacao.c, y0, x1,
                                     transformacao.f, densify_pts=101)
            margem = 4 * abs(src.res[0])
            caixa = (caixa[0] - margem, caixa[1] - margem,
                     caixa[2] + margem, caixa[3] + margem)
            fonte, fonte_tr, info_mosaico = _fonte_em_mosaico(src, caixa,
                                                               mosaico)
        reproject(source=fonte, destination=saida,
                  src_transform=fonte_tr, src_crs=src.crs,
                  **({"src_nodata": src.nodata} if mosaico else {}),
                  dst_transform=transformacao, dst_crs=alvo,
                  dst_nodata=-32768.0,
                  resampling=Resampling.average,
                  # grade alinhada pede transformacao exata: o aproximado do
                  # GDAL (0,125 px por bloco) deixava 2% das celulas comuns de
                  # dois recortes diferentes (medido em 10/10/2026)
                  **({"tolerance": 0} if alinhar_m else {}))

    destino.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destino, "w", **perfil) as fora:
        fora.write(saida, 1)
    return {"crs": alvo, "shape": [int(altura), int(largura)],
            "pixel_size": [RESOLUCAO_CANONICA_M, RESOLUCAO_CANONICA_M],
            "resampling_dem": "average",
            "origem_alinhada_m": alinhar_m,
            "origem": [float(transformacao.c), float(transformacao.f)],
            "mosaico_de_entrada": info_mosaico,
            "celulas_sem_valor": int((saida == -32768.0).sum())}


def _esperar(alvo: Path, wbt, chamada, *args, **kwargs) -> Path:
    """Chama a ferramenta e espera o arquivo aparecer.

    `run_tool` do WhiteboxTools devolve 0 SEMPRE -- drena o stdout e retorna,
    sem `proc.wait()`. Conferir o codigo de retorno nao diz nada; conferir se a
    saida existe, diz.
    """
    if alvo.exists():
        alvo.unlink()
    getattr(wbt, chamada)(*args, **kwargs)
    for _ in range(ESPERAS_PELA_SAIDA):
        if alvo.exists():
            return alvo
        time.sleep(0.5)
    raise DerivacaoRecusada(f"WhiteboxTools.{chamada} nao produziu {alvo.name}")


def _canal_e_hand_do_ter01(dem_filled: Path, pasta: Path,
                           resolucao_m: float) -> dict:
    """Acumulacao em celulas, canal por AREA convertida em celulas, e HAND.

    ESTA PARTE NAO PODE VIR DE `_derivar_terreno`, e a razao foi medida.

    A cadeia do servico aplica `threshold = 0,1123 x 1e6` sobre a SCA -- que
    sai em METROS, nao em metros quadrados. Derivando o mesmo DEM pelos dois
    caminhos, `dem_filled`, `slope_deg_wbt`, `dinf_sca` e `dinf_accum_cells`
    saem BIT A BIT iguais (r = 1,000000), e o `hand_dinf` sai com r = 0,9289 e
    diferenca de ate 166 m. E a mesma divergencia que o `terreno_invariantes`
    ja separa em MODO_SERVICO e MODO_VALIDADO.

    O `ter-01` -- que derivou os 784 diretorios que o encoder le -- converteu a
    area em celulas: 0,1123 km2 / 900 m2 = 124,78 celulas, o
    `stream_threshold_cells` que cada `run_manifest.json` carrega. Reproduzir
    isso e obrigatorio: HAND derivado pela outra convencao entraria no mesmo
    canal do encoder com o nome certo e o significado diferente.
    """
    import whitebox

    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    wbt.set_working_dir(str(pasta))

    celulas = pasta / "dinf_accum_cells.tif"
    _esperar(celulas, wbt, "d_inf_flow_accumulation", str(dem_filled),
             str(celulas), out_type="cells")

    limiar_celulas = (LIMIAR_CANAL_CANONICO_KM2 * 1e6) / (resolucao_m ** 2)
    streams = pasta / "streams_dinf.tif"
    _esperar(streams, wbt, "extract_streams", str(celulas), str(streams),
             threshold=limiar_celulas)
    hand = pasta / "hand_dinf.tif"
    _esperar(hand, wbt, "elevation_above_stream", str(dem_filled),
             str(streams), str(hand))
    return {"stream_threshold_cells": limiar_celulas,
            "entrada_do_canal": "dinf_accum_cells.tif"}


def _enxugar(destino: Path, nome: str) -> dict:
    """Comprime o que fica e descarta o que ninguem le.

    -------------------------------------------------------------------------
    POR QUE ISTO E PARTE DA DERIVACAO, E NAO UMA FAXINA DEPOIS
    -------------------------------------------------------------------------

    O WhiteboxTools grava sem compressao e em float64: medido em 21/09/2026, um
    tile de 1 grau nasce com 943 MB, e `local_runs` chegou a 418 GB com 91 GB
    livres no disco. Deixar isso para uma faxina periodica significa que o
    problema volta a cada tile derivado -- e a fila tem centenas deles.

    A compressao e SEM PERDA e conferida celula a celula antes de substituir o
    arquivo: `deflate` com predictor, os mesmos valores. Medido: 943 MB -> 528 MB,
    com `streams_dinf` a 2,2% do tamanho.

    O descarte e das camadas que NENHUM consumidor le -- `pointer_dinf`,
    `streams` e o DEM reprojetado de entrada --, e a cadeia as refaz sem rede.
    `dinf_sca` e `dinf_accum_cells` FICAM: sao dois dos cinco canais do encoder.
    """
    sys.path.insert(0, str(REPO / "scripts" / "manutencao"))
    from pegada_de_disco import _comprimir_um

    descartadas, antes, depois, comprimidas = [], 0, 0, 0
    for arq in sorted(destino.glob("*.tif")):
        if arq.name in (f"{nome}_dem_30m.tif", "pointer_dinf.tif",
                        "streams.tif"):
            descartadas.append(arq.name)
            arq.unlink()
            continue
        r = _comprimir_um(arq)
        antes += r["antes"]
        depois += r["depois"]
        comprimidas += r["estado"] == "comprimido"
    return {"camadas_comprimidas": comprimidas,
            "bytes_antes": antes, "bytes_depois": depois,
            "descartadas": descartadas,
            "por_que": ("compressao sem perda conferida celula a celula; "
                        "descarte so de camada sem consumidor, que a cadeia "
                        "refaz a partir do que fica"),
            "nao_e": ("nao muda valor de celula e nao toca dinf_sca nem "
                      "dinf_accum_cells, que sao canais do encoder")}


def derivar(dem: Path, nome: str, raiz: Path = RAIZ,
            epsg_metrico: int | None = None, enxugar: bool = True,
            alinhar_m: float | None = None,
            mosaico: list[Path] | None = None) -> dict:
    """DEM em disco -> diretorio no formato do `ter-01`, ou excecao."""
    dem = Path(dem)
    if not dem.exists():
        raise DerivacaoRecusada(f"DEM ausente: {dem}")
    destino = Path(raiz) / nome
    destino.mkdir(parents=True, exist_ok=True)

    dem_30m = destino / f"{nome}_dem_30m.tif"
    inicio = time.time()
    geo = reprojetar_para_30m(dem, dem_30m, epsg_metrico, alinhar_m, mosaico)
    saidas = _derivar_terreno(dem_30m, destino)

    # `facc_dinf` E a area contribuinte especifica; o `ter-01` a chama de
    # `dinf_sca`. Renomear e mais honesto que derivar duas vezes: seria a mesma
    # conta com dois nomes, e um dia elas divergiriam.
    facc = destino / "facc_dinf.tif"
    sca = destino / "dinf_sca.tif"
    if facc.exists() and not sca.exists():
        facc.replace(sca)
    # sobrescreve `streams`/`hand_dinf` da cadeia do servico -- ver o docstring
    # de `_canal_e_hand_do_ter01`, e a divergencia de 166 m que ele mede.
    canal = _canal_e_hand_do_ter01(destino / "dem_filled.tif", destino,
                                   RESOLUCAO_CANONICA_M)

    faltando = [c for c in CAMADAS if not (destino / f"{c}.tif").exists()]
    if faltando:
        raise DerivacaoRecusada(
            f"{nome}: cadeia rodou e faltam camadas {faltando}")

    manifesto = {
        "region_label": nome,
        "dtm_input": str(dem),
        "outdir": str(destino),
        "crs": geo["crs"],
        "shape": geo["shape"],
        "pixel_size": geo["pixel_size"],
        "stream_area_km2": LIMIAR_CANAL_CANONICO_KM2,
        "stream_threshold_cells": canal["stream_threshold_cells"],
        "entrada_do_canal": canal["entrada_do_canal"],
        "twi_formula": FORMULA_TWI_CANONICA,
        "resampling_dem": geo["resampling_dem"],
        "origem_alinhada_m": geo["origem_alinhada_m"],
        "origem": geo["origem"],
        "mosaico_de_entrada": geo["mosaico_de_entrada"],
        "celulas_sem_valor_no_dem_30m": geo["celulas_sem_valor"],
        "wbt_version": _versao_wbt(),
        "elapsed_s": round(time.time() - inicio, 1),
        "python": platform.python_version(),
        "platform": sys.platform,
        "repeticoes_do_binario": saidas.get("repeticoes_do_binario"),
        "quando_utc": agora_utc(),
        "tipo": "derivacao_de_terreno_aoi_nova",
        "nao_e": ("observacao de agua nem rotulo: descreve a forma do lugar, "
                  "nao o que aconteceu nele"),
    }
    (destino / "run_manifest.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")

    # A propria checagem que o treino faz. Derivar e nao ser aceito e um modo
    # de falha silencioso: o diretorio existiria e o treino o ignoraria.
    contrato = cadeia_declarada(nome, destino)
    if not contrato["no_contrato"]:
        raise DerivacaoRecusada(
            f"{nome}: derivado e recusado pelo contrato -- {contrato['por_que']}")

    # Depois do contrato, nunca antes: enxugar um diretorio que a proxima linha
    # recusaria seria trabalho jogado fora, e apagar entrada de uma derivacao
    # invalida tiraria a chance de refaze-la sem rede.
    if enxugar:
        manifesto["enxugado"] = _enxugar(destino, nome)
        (destino / "run_manifest.json").write_text(
            json.dumps(manifesto, indent=2, ensure_ascii=False),
            encoding="utf-8")

    manifesto["hash_manifesto"] = sha256_json(
        {k: v for k, v in manifesto.items()
         if k not in ("quando_utc", "elapsed_s")})
    return manifesto


def _versao_wbt() -> str:
    try:
        import whitebox
        return (whitebox.WhiteboxTools().version() or "").splitlines()[0]
    except Exception:
        return "desconhecida"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--nome", required=True)
    parser.add_argument("--epsg", type=int, default=None,
                        help="CRS metrico; padrao e o fuso UTM da AOI")
    args = parser.parse_args()
    print(json.dumps(derivar(args.dem, args.nome, epsg_metrico=args.epsg),
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
