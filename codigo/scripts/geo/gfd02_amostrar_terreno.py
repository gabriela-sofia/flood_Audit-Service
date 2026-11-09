"""O terreno da cadeia canonica sob os pontos do Global Flood Database.

-------------------------------------------------------------------------
O QUE ESTE ARQUIVO DESTRAVA
-------------------------------------------------------------------------

`gfd01` entregou 149.179 pontos brasileiros em que `clear_views > 0` -- ou
seja, o MODIS OLHOU aquele lugar -- e o `ds05` os recusa por inteiro com
`saiu__B_cadeia_global_nao_comparavel`, corretamente: eles chegaram com
`lat`/`lon` e nada de terreno.

Isso importa porque negativo OBSERVADO e o ativo escasso do projeto. As
centrais de chamado dao 373.224 pontos de negativo AUSENTE, que e fundo, e com
fundo a prevalencia nao e identificavel (`aud_desenho02`: Recife 98,4% de
fundo). O `EMSR720` e a unica ativacao brasileira e da 216,55 km2. O GFD da
757.754 km2 na faixa.

-------------------------------------------------------------------------
O QUE MUDA DE ESCALA, E ISSO E DECLARADO E NAO ESCONDIDO
-------------------------------------------------------------------------

O rotulo do GFD e de 250 m e a cadeia e de 30 m. Amostrar terreno fino num
rotulo grosso reintroduz erro de posicao do rotulo -- exatamente o defeito
diagnosticado no pluvial, onde o oraculo de lugar dava 0,95 e o terreno 0,56.

Duas defesas, e as duas sao verificaveis:

  1. `gfd01` ja empurrou todo ponto 250 m para DENTRO da sua classe, por
     `fastDistanceTransform`. Nenhum ponto esta na borda entre inundado e
     observado-seco.
  2. cada ponto sai com `resolucao_m = 250`, e nao com 30. Quem modelar sabe
     que a etiqueta e grossa mesmo que a feature seja fina.

-------------------------------------------------------------------------
O GRUPO DE VALIDACAO E O TILE, e nao o evento
-------------------------------------------------------------------------

`gfd01` gravou `grupo_cv = GFD<evento>`. Um evento do GFD atravessa mil
quilometros: agrupar por evento poria numa unica dobra pontos de bacias
inteiramente diferentes, e a `RAIZ_DO_FRACASSO_PLUVIAL` mediu o custo do
oposto -- agrupamento fino demais fez Recife cair de 0,5919 para 0,5191 quando
o grupo passou de 120 m para 5 km, porque o que parecia sinal era memoria de
lugar.

Aqui o grupo passa a ser o TILE de 1 grau (~111 km), que e tambem a unidade de
derivacao. O evento continua gravado em `grupo_evento_gfd`, entao a escolha e
reversivel por quem discordar dela.

-------------------------------------------------------------------------
AGUA PERMANENTE E RECONFERIDA A 30 m
-------------------------------------------------------------------------

`gfd01` ja excluiu `jrc_perm_water` das duas classes, a 250 m. A mascara JRC
alinhada ao terreno e de 30 m e ve o leito que a de 250 m alisou. Ponto que
cair sobre ela vira classe PROPRIA -- `AGUA_PERMANENTE` --, que e a mesma
regra do `cems03`: nao e positivo, porque a agua ja estava la, e nao e
negativo, porque e agua.

No RS isso nao foi detalhe: com o leito do rio dentro do negativo o
`twi_dinf` INVERTEU de +0,4408 para -0,1727.

-------------------------------------------------------------------------
A TRUNCAGEM DE BACIA VIAJA COM O PONTO
-------------------------------------------------------------------------

`cadeia_em_lote` deriva com margem de bacia, e o ponto so e aceito no tile
interno. Mesmo assim o talvegue principal segue truncado, entao cada ponto
leva `dist_borda_derivacao_m`. Quem quiser um corte mais duro corta por ela em
vez de confiar num numero unico -- que e o oposto do que o projeto fez quando
aceitou o HAND de Valparaiso sem olhar de onde vinha a area contribuinte.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao deriva terreno, nao baixa nada, nao decide admissao e nao promove modelo:
le raster que o `cadeia_em_lote` derivou e escreve as duas tabelas que o
`ds01` e o `ds04` consomem.

Uso:
    python scripts/geo/gfd02_amostrar_terreno.py
    python scripts/geo/gfd02_amostrar_terreno.py --gravar
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "geo"))

from agua_permanente_jrc import LIMIAR_CALIBRADO  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

RUNS = REPO / "local_runs"
GFD = RUNS / "gfd-brasil"
TER01 = RUNS / "ter-01-cadeia-harmonizada"
SAIDA_HARM = RUNS / "ter-02-comparacao"
ARQUIVO_HARM = "dataset_harmonizado_gfd.csv"
ARQUIVO_PONTOS = "pontos_gfd_com_terreno.csv"

CAMADAS_TERRENO = {
    "elevation_m": "dem_filled.tif",
    "hand_m": "hand_dinf.tif",
    "slope_deg": "slope_deg_wbt.tif",
    "twi_dinf": "twi_dinf.tif",
}
CANAL = "streams_dinf.tif"
AGUA_PERMANENTE = "agua_permanente_jrc.tif"

# A OCORRENCIA CRUA VIAJA COM O PONTO, e nao so o binario.
#
# Medido em 12/09/2026 nos 9.604 pontos dos dois tiles do alto Parana: a classe
# INUNDACAO do GFD tem `occurrence` com mediana 79 (p10 72, p90 83) e apenas
# 5,7% acima de 90. Nao e reservatorio -- agua aberta daria 95-100 -- e a
# varzea do Parana, terreno com agua em ~3 de cada 4 observacoes. A classe
# NAO_INUNDADO_OBSERVADO tem mediana 0.
#
# O corte de 50 do projeto foi calibrado contra o `JRCWaterHand` do
# Sen1Floods11 (IoU 0,8616) e chama isso de permanente; o GFD usa definicao
# mais estrita e preserva. As duas sao defensaveis e respondem perguntas
# diferentes, entao gravar SO o binario congelaria a escolha de uma delas
# dentro do dado. Com a coluna continua qualquer corte e reversivel.

VARIAVEIS = ["elevation_m", "slope_deg", "hand_m", "twi_dinf"]
RESOLUCAO_ESPERADA_M = 30.0
LIMIAR_ESPERADO_KM2 = 0.1123
RESOLUCAO_DO_ROTULO_M = 250

FONTE = "global_flood_database"
LICENCA = ("Global Flood Database v1 (Tellman et al. 2021), CC BY 4.0 "
           "-- uso com atribuicao")


def _rotulo_do_tile(lat: int, lon: int) -> str:
    ns = f"{'S' if lat < 0 else 'N'}{abs(lat):02d}"
    ew = f"{'W' if lon < 0 else 'E'}{abs(lon):03d}"
    return f"{ns}{ew}"


def conferir_cadeia(terreno: Path) -> dict:
    """A cadeia do diretorio e a canonica, ou o tile e recusado.

    Mesma conferencia do `ter02`, e pela mesma razao: sem ela o modulo
    carimbaria `wbt30` em qualquer coisa, e existe em disco diretorio derivado
    por outro criterio de canal (`EMSR789_AOI06`).
    """
    manifesto = terreno / "run_manifest.json"
    if not manifesto.exists():
        raise ValueError(f"sem run_manifest.json em {terreno.name}")
    d = json.loads(manifesto.read_text(encoding="utf-8"))
    px = d.get("pixel_size") or [None, None]
    area = d.get("stream_area_km2")
    if float(px[0]) != RESOLUCAO_ESPERADA_M or float(area) != LIMIAR_ESPERADO_KM2:
        raise ValueError(f"cadeia fora do contrato em {terreno.name}: "
                         f"pixel={px} area={area}")
    return {"crs": d.get("crs"), "pixel_size": px, "stream_area_km2": area,
            "twi_formula": d.get("twi_formula"),
            "wbt_version": d.get("wbt_version")}


def _ocorrencia_jrc(x, y, crs):
    """`occurrence` bruta do JRC nos pontos, em porcentagem.

    Lida do produto de 10x10 graus por `/vsicurl`, no mesmo caminho que a
    `agua_permanente_jrc` usa para montar a mascara binaria. O codigo de
    ausencia do produto (acima de 100) vira NaN em vez de virar agua.
    """
    import numpy as np
    import rasterio
    from pyproj import Transformer

    from agua_permanente_jrc import url_de

    para_4326 = Transformer.from_crs(crs, 4326, always_xy=True)
    lon, lat = para_4326.transform(x, y)
    try:
        with rasterio.open(url_de(float(np.mean(lon)), float(np.mean(lat)))) as s:
            v = np.array([p[0] for p in s.sample(list(zip(lon, lat)))],
                         dtype="float64")
    except Exception as e:                          # noqa: BLE001
        # Sem rede o binario alinhado ao terreno ainda existe; a coluna
        # continua fica NaN e o manifesto diz por que, em vez de zero.
        print(f"      AVISO: occurrence do JRC indisponivel ({e})")
        return np.full(len(x), np.nan)
    v[v > 100] = np.nan
    return v


# O RELEVO E MEDIDO NUMA JANELA EM VOLTA DE CADA PONTO, e nao por tile.
#
# O `ds04` classificava o relevo do GFD por TILE de 1 grau, sobre os proprios
# pontos amostrados -- e os pontos saem da faixa de 2 km em volta da cheia,
# que e planicie por construcao. Em 30/09/2026 nenhum dos 417 tiles dava
# INGREME, e a serra brasileira nao existia na tabela. Classificar pelo
# declive DO PONTO seria o erro oposto: numa serra a cheia acontece no fundo
# do vale, que e plano, e o ponto chamaria de planicie exatamente o vale de
# serra.
#
# A janela tem o lado MEDIANO das 268 AOIs do CEMS (13,7 km, medido em
# 30/09/2026) -- a escala em que o criterio `relevo >= 400 m E frac_gt15 >=
# 0,25` foi calibrado (`janela_raster_cems03`). O passo de 150 m dentro dela
# so reduz custo: p2/p98 e fracao nao mudam com subamostragem regular.
JANELA_RELEVO_M = 13_700.0
PASSO_DA_JANELA_PX = 5
# Nomeadas aqui porque a saida escolhe coluna por lista: na primeira execucao
# (30/09/2026) a janela foi medida em 376 mil pontos e descartada na escrita,
# e o ds04 recebeu "gfd02 sem janela de relevo".
COLUNAS_DE_RELEVO = ("relevo_local_janela_m", "frac_gt15_janela")


def _relevo_na_janela(dem, slope, valido, lin, col, px: float):
    """p2-p98 da elevacao e fracao de declive > 15 graus na janela do ponto."""
    import numpy as np

    s = PASSO_DA_JANELA_PX
    e = np.where(valido, dem, np.nan)[::s, ::s].astype("float64")
    sl = np.where(valido & np.isfinite(slope), slope, np.nan)[::s, ::s]
    h = int(round(JANELA_RELEVO_M / 2 / (px * s)))
    minimo = 0.5 * (2 * h + 1) ** 2
    relevo = np.full(len(lin), np.nan)
    frac = np.full(len(lin), np.nan)
    memo: dict = {}
    for i, (r, c) in enumerate(zip(np.asarray(lin) // s, np.asarray(col) // s)):
        chave = (int(r), int(c))
        if chave not in memo:
            we = e[max(r - h, 0):r + h + 1, max(c - h, 0):c + h + 1]
            ws = sl[max(r - h, 0):r + h + 1, max(c - h, 0):c + h + 1]
            ve, vs = we[np.isfinite(we)], ws[np.isfinite(ws)]
            # Janela com menos de metade de dado valido (borda da derivacao,
            # mar) nao classifica: vira NAO_CLASSIFICADO no ds04.
            memo[chave] = ((float(np.percentile(ve, 98) - np.percentile(ve, 2)),
                            float((vs > 15).mean()))
                           if ve.size >= minimo and vs.size >= minimo
                           else (np.nan, np.nan))
        relevo[i], frac[i] = memo[chave]
    return relevo.round(1), frac.round(4)


def _amostrar_no_tile(d, terreno: Path):
    """Valores das camadas nos pontos do tile, no CRS do proprio raster."""
    import numpy as np
    import rasterio
    from pyproj import Transformer

    from scipy import ndimage

    # A DISTANCIA E ATE O DADO VALIDO, e nao ate a moldura do raster.
    #
    # A moldura em UTM e o retangulo que ENVOLVE o retangulo em lat/lon
    # reprojetado, entao os cantos ficam com nodata. Medir contra a moldura
    # daria ao ponto de canto uma folga que ele nao tem, e a folga e exatamente
    # o que esta sendo auditado: onde a area contribuinte foi truncada.
    with rasterio.open(terreno / CAMADAS_TERRENO["elevation_m"]) as src:
        crs, limites = src.crs, src.bounds
        dem = src.read(1)
        valido = (dem != src.nodata) if src.nodata is not None else ~np.isnan(dem)
        px = abs(src.transform.a)
        inverso = ~src.transform
    # `distance_transform_edt` mede a distancia de cada celula valida ate a
    # celula invalida mais proxima; a borda externa entra como invalida.
    moldurado = np.zeros((valido.shape[0] + 2, valido.shape[1] + 2), bool)
    moldurado[1:-1, 1:-1] = valido
    distancia = ndimage.distance_transform_edt(moldurado)[1:-1, 1:-1] * px

    tr = Transformer.from_crs(4326, crs, always_xy=True)
    x, y = tr.transform(d.lon.to_numpy(), d.lat.to_numpy())
    coords = list(zip(x, y))

    saida = {"x_m": x, "y_m": y, "crs_metrico": crs.to_epsg()}
    for nome, arquivo in CAMADAS_TERRENO.items():
        with rasterio.open(terreno / arquivo) as src:
            v = np.array([p[0] for p in src.sample(coords)], dtype="float64")
            if src.nodata is not None:
                v[v == src.nodata] = np.nan
            saida[nome] = v
    saida["jrc_occurrence"] = _ocorrencia_jrc(x, y, crs)
    saida["fonte_agua_permanente"] = np.full(len(d), "mascara_30m", dtype=object)
    for chave, arquivo in (("sobre_canal_modelado", CANAL),
                           ("agua_permanente_jrc", AGUA_PERMANENTE)):
        alvo = terreno / arquivo
        if not alvo.exists() and chave == "agua_permanente_jrc":
            # SEM MASCARA, A REGRA E A MESMA, LIDA NO PONTO.
            #
            # So 24 dos 228 tiles do lote tinham a mascara, e aqui ela virava
            # zero em silencio: em 18/09/2026, 11.613 de 22.574 positivos dos
            # 204 tiles sem mascara (51,4%) tinham occurrence >= 50 -- leito
            # de rio dentro do positivo. A occurrence pontual e o mesmo
            # produto e o mesmo limiar calibrado; onde as duas existem elas
            # concordam em 99,61% dos pontos, e o manifesto remede isso.
            occ = saida["jrc_occurrence"]
            lida = np.isfinite(occ)
            saida[chave] = np.where(
                lida, (occ >= LIMIAR_CALIBRADO) & (occ <= 100), False
            ).astype(int)
            saida["fonte_agua_permanente"] = np.where(
                lida, "ocorrencia_pontual", "indisponivel")
            continue
        if not alvo.exists():
            saida[chave] = np.zeros(len(d), dtype=int)
            continue
        with rasterio.open(alvo) as src:
            c = np.array([p[0] for p in src.sample(coords)], dtype="float64")
            nd = src.nodata
            saida[chave] = np.where(
                np.isnan(c) | ((c == nd) if nd is not None else False), 0,
                (c > 0).astype(int))

    # Distancia ate a borda do dado valido da DERIVACAO, que e maior que o
    # tile por causa da margem de bacia. Ponto fora do raster recebe 0.
    col, lin = inverso * (x, y)
    col = np.clip(col.astype(int), 0, distancia.shape[1] - 1)
    lin = np.clip(lin.astype(int), 0, distancia.shape[0] - 1)
    dentro = ((x >= limites.left) & (x <= limites.right)
              & (y >= limites.bottom) & (y <= limites.top))
    saida["dist_borda_derivacao_m"] = np.where(
        dentro, distancia[lin, col], 0.0).round(1)
    saida["fora_do_raster"] = (~dentro).astype(int)
    with rasterio.open(terreno / CAMADAS_TERRENO["slope_deg"]) as src:
        slope = src.read(1).astype("float64")
        if src.nodata is not None:
            slope[slope == src.nodata] = np.nan
    relevo, frac = _relevo_na_janela(dem, slope, valido, lin, col, px)
    saida["relevo_local_janela_m"] = np.where(dentro, relevo, np.nan)
    saida["frac_gt15_janela"] = np.where(dentro, frac, np.nan)
    return saida


def construir(gravar: bool = False) -> dict:
    import numpy as np
    import pandas as pd

    arquivos = sorted(GFD.glob("pontos_gfd*.csv"))
    arquivos = [a for a in arquivos if a.name != ARQUIVO_PONTOS]
    if not arquivos:
        raise SystemExit(f"sem pontos GFD em {GFD}")
    d = pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True)
    d["tlat"] = d.lat.apply(math.floor).astype(int)
    d["tlon"] = d.lon.apply(math.floor).astype(int)
    d["tile"] = [_rotulo_do_tile(a, b) for a, b in zip(d.tlat, d.tlon)]

    tiles_pedidos = sorted(d.tile.unique())
    partes, cadeias, recusados = [], {}, {}
    for tile in tiles_pedidos:
        pasta = TER01 / f"gfd_tile_{tile}"
        if not (pasta / "run_manifest.json").exists():
            recusados[tile] = "terreno nao derivado"
            continue
        try:
            cadeias[tile] = conferir_cadeia(pasta)
        except ValueError as e:
            recusados[tile] = str(e)
            continue
        sub = d[d.tile == tile].copy()
        vals = _amostrar_no_tile(sub, pasta)
        for k, v in vals.items():
            sub[k] = v
        partes.append(sub)
        print(f"  {tile}: {len(sub):,} pontos "
              f"| {int(np.isfinite(vals['hand_m']).sum()):,} com HAND "
              f"| agua permanente 30 m em {int(vals['agua_permanente_jrc'].sum()):,}")

    if not partes:
        raise SystemExit(
            "nenhum tile com terreno derivado; rode antes "
            "`python scripts/terreno/cadeia_em_lote.py --tiles-gfd N --baixar`")
    t = pd.concat(partes, ignore_index=True)

    # A classe do poligono e gravada ANTES da reconferencia, como no `cems03`:
    # depois dela a origem do ponto reclassificado se perderia.
    t["classe_do_poligono"] = t.classe_obs
    # A reconferencia a 30 m: o ponto sobre o leito vira classe PROPRIA.
    reclassificados = int(((t.agua_permanente_jrc == 1)
                           & (t.classe_obs != "AGUA_PERMANENTE")).sum())
    reclassificados_por_fonte = {
        f: int(((t.agua_permanente_jrc == 1)
                & (t.classe_obs != "AGUA_PERMANENTE")
                & (t.fonte_agua_permanente == f)).sum())
        for f in sorted(t.fonte_agua_permanente.unique())}
    com_mascara = t[(t.fonte_agua_permanente == "mascara_30m")
                    & t.jrc_occurrence.notna()]
    concordancia = (round(float(
        ((com_mascara.agua_permanente_jrc == 1)
         == ((com_mascara.jrc_occurrence >= LIMIAR_CALIBRADO)
             & (com_mascara.jrc_occurrence <= 100))).mean()), 4)
        if len(com_mascara) else None)
    t.loc[t.agua_permanente_jrc == 1, "classe_obs"] = "AGUA_PERMANENTE"

    completo = t[VARIAVEIS].notna().all(axis=1)
    t["grupo_evento_gfd"] = t.grupo_cv
    t["grupo_cv"] = "GFD_" + t.tile
    t["ativacao"] = "GFD"
    t["aoi_id"] = t.tile
    t["resolucao_m"] = RESOLUCAO_DO_ROTULO_M
    t["fonte"] = FONTE
    t["licenca"] = LICENCA
    t["worldcover"] = np.nan

    colunas = ["classe_obs", "aoi_id", "x_m", "y_m", "lon", "lat",
               "elevation_m", "hand_m", "slope_deg", "twi_dinf", "worldcover",
               "ativacao", "grupo_cv", "crs_metrico", "resolucao_m", "fonte",
               "licenca", "sobre_canal_modelado", "agua_permanente_jrc",
               "classe_do_poligono", "ponto_id", "evento_gfd",
               "grupo_evento_gfd", "tile", "dist_borda_derivacao_m",
               "fora_do_raster", "jrc_occurrence", "fonte_agua_permanente",
               *COLUNAS_DE_RELEVO]
    pontos = t[colunas]

    harm = pd.DataFrame({"ponto_id": t.ponto_id})
    for v in VARIAVEIS:
        harm[f"{v}_wbt"] = t[v]
    harm["regiao"] = "gfd_" + t.tile
    harm["grupo_cv"] = t.grupo_cv

    manifesto = {
        "tipo": "amostragem_de_terreno_dos_pontos_gfd",
        "quando_utc": agora_utc(),
        "tiles_pedidos": len(tiles_pedidos),
        "tiles_com_terreno": len(cadeias),
        "tiles_recusados": recusados,
        "pontos_no_lote_gfd": int(len(d)),
        "pontos_amostrados": int(len(t)),
        "com_as_quatro_variaveis": int(completo.sum()),
        "sem_alguma": int((~completo).sum()),
        "por_classe": {k: int(v) for k, v in t.classe_obs.value_counts().items()},
        "reclassificados_como_agua_permanente_a_30m": reclassificados,
        "agua_permanente_por_fonte": {
            "pontos": {k: int(v) for k, v
                       in t.fonte_agua_permanente.value_counts().items()},
            "reclassificados": reclassificados_por_fonte,
            "tiles_com_mascara_30m": int(t.loc[
                t.fonte_agua_permanente == "mascara_30m", "tile"].nunique()),
            "concordancia_mascara_x_ocorrencia_pontual": concordancia,
            "limiar_occurrence": LIMIAR_CALIBRADO,
            "regra": ("sem a mascara alinhada ao tile, o ponto com occurrence "
                      ">= limiar calibrado vira AGUA_PERMANENTE, a mesma regra "
                      "da mascara lida no ponto; sem occurrence o ponto fica "
                      "'indisponivel' e nao e reclassificado"),
        },
        "ocorrencia_jrc_por_classe": {
            classe: {
                "n": int(sub.jrc_occurrence.notna().sum()),
                "mediana": (round(float(sub.jrc_occurrence.median()), 1)
                            if sub.jrc_occurrence.notna().any() else None),
                "p10": (round(float(sub.jrc_occurrence.quantile(0.10)), 1)
                        if sub.jrc_occurrence.notna().any() else None),
                "p90": (round(float(sub.jrc_occurrence.quantile(0.90)), 1)
                        if sub.jrc_occurrence.notna().any() else None),
                "frac_acima_de_90": (
                    round(float((sub.jrc_occurrence >= 90).mean()), 4)
                    if sub.jrc_occurrence.notna().any() else None),
            }
            for classe, sub in t.groupby("classe_do_poligono")},
        "o_que_a_ocorrencia_decide": (
            "positivo com occurrence entre 70 e 90 nao e reservatorio (agua "
            "aberta daria 95-100) nem evento: e varzea, terreno com agua em 3 "
            "de cada 4 observacoes. Pedir ao terreno que preveja isso ensina "
            "'a varzea e a varzea' -- AUC alta e sem significado, a mesma "
            "armadilha do oraculo de lugar do pluvial. A coluna continua fica "
            "gravada para que o corte seja decisao explicita e reversivel"),
        "por_que_reclassificar": (
            "gfd01 excluiu jrc_perm_water a 250 m; a mascara alinhada ao "
            "terreno e de 30 m e ve o leito que a de 250 m alisou. No RS o "
            "leito dentro do negativo INVERTEU o twi_dinf de +0,4408 para "
            "-0,1727"),
        "resolucao_do_rotulo_m": RESOLUCAO_DO_ROTULO_M,
        "por_que_o_rotulo_fica_250": (
            "a feature e de 30 m e a etiqueta e de 250 m; gravar 30 aqui "
            "esconderia o erro de posicao do rotulo, que e o defeito "
            "diagnosticado no pluvial"),
        "grupo_cv": (
            "o TILE de 1 grau, e nao o evento: um evento do GFD atravessa mil "
            "quilometros e poria bacias diferentes na mesma dobra. O evento "
            "fica em grupo_evento_gfd, entao a escolha e reversivel"),
        "pontos_fora_do_raster": int(t.fora_do_raster.sum()),
        "dist_borda_derivacao_m": {
            "minima": float(t.dist_borda_derivacao_m.min()),
            "mediana": float(t.dist_borda_derivacao_m.median()),
            "medida_contra": (
                "a borda do DADO VALIDO, por distance_transform_edt, e nao a "
                "moldura do raster: a moldura em UTM envolve o retangulo em "
                "lat/lon reprojetado e deixa nodata nos cantos"),
            "por_que": (
                "a margem de bacia do cadeia_em_lote nao resolve talvegue "
                "principal; a distancia viaja com o ponto para que o corte "
                "seja de quem modela"),
        },
        "cadeias": cadeias,
        "conferencia": (f"run_manifest.json de cada tile precisa declarar "
                        f"{RESOLUCAO_ESPERADA_M} m e {LIMIAR_ESPERADO_KM2} "
                        "km2, senao o tile e recusado"),
        "licenca": LICENCA,
        "nao_e": (
            "nao deriva terreno, nao baixa nada, nao decide admissao e nao "
            "promove modelo: le raster que o cadeia_em_lote derivou e escreve "
            "as tabelas que o ds01 e o ds04 consomem"),
    }
    manifesto["hash_resumo"] = sha256_json(
        {k: v for k, v in manifesto.items() if k != "quando_utc"})

    if gravar:
        SAIDA_HARM.mkdir(parents=True, exist_ok=True)
        pontos.to_csv(GFD / ARQUIVO_PONTOS, index=False)
        harm.to_csv(SAIDA_HARM / ARQUIVO_HARM, index=False)
        (GFD / "resumo_terreno.json").write_text(
            json.dumps(manifesto, indent=2, ensure_ascii=False),
            encoding="utf-8")
        manifesto["arquivos"] = [str(GFD / ARQUIVO_PONTOS),
                                 str(SAIDA_HARM / ARQUIVO_HARM)]
    return manifesto


def main() -> int:
    p = argparse.ArgumentParser()
    # Gravar e o PADRAO. A forma anterior (`--gravar` opcional) criava
    # artefato que envelhece em silencio: em 22/09/2026 o `casos.json` do
    # produto estava desatualizado desde antes da migracao para parquet e um
    # teste passou contra ele por dias. `--gravar` segue aceito para nao
    # quebrar chamador existente.
    p.add_argument("--gravar", action="store_true", default=True,
                   help=argparse.SUPPRESS)
    p.add_argument("--nao-gravar", dest="gravar", action="store_false",
                   help="nao escreve artefato; so imprime o resumo")
    a = p.parse_args()

    m = construir(a.gravar)
    print(f"\ntiles com terreno : {m['tiles_com_terreno']}/{m['tiles_pedidos']}")
    print(f"pontos amostrados : {m['pontos_amostrados']:,} de "
          f"{m['pontos_no_lote_gfd']:,} no lote")
    print(f"com as 4 variaveis: {m['com_as_quatro_variaveis']:,} "
          f"(sem alguma: {m['sem_alguma']:,})")
    print(f"por classe        : {m['por_classe']}")
    print(f"reclassificados   : {m['reclassificados_como_agua_permanente_a_30m']:,} "
          "viraram AGUA_PERMANENTE "
          f"{m['agua_permanente_por_fonte']['reclassificados']} "
          f"(concordancia mascara x ponto: "
          f"{m['agua_permanente_por_fonte']['concordancia_mascara_x_ocorrencia_pontual']})")
    print("ocorrencia JRC    : "
          + " | ".join(
              f"{c}: mediana {v['mediana']} p90 {v['p90']} "
              f"(>=90 em {v['frac_acima_de_90']:.1%})"
              for c, v in sorted(m["ocorrencia_jrc_por_classe"].items())
              if v["mediana"] is not None))
    print(f"dist da borda     : min {m['dist_borda_derivacao_m']['minima']:,.0f} m "
          f"mediana {m['dist_borda_derivacao_m']['mediana']:,.0f} m "
          f"| fora do raster: {m['pontos_fora_do_raster']:,}")
    if not a.gravar:
        print("\nNADA GRAVADO. Use --gravar.")
    else:
        for f in m["arquivos"]:
            print(f"GRAVADO={f}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
