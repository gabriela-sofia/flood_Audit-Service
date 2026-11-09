"""COB-01 -- Cobertura, area construida e solo sob cada ponto do dataset de treino.

-------------------------------------------------------------------------
POR QUE ISTO EXISTE
-------------------------------------------------------------------------

O dataset de treino tem o terreno de cada ponto e nada do que esta em cima
dele ou embaixo dele. A coluna `worldcover` dos pontos GFD existe e esta
vazia. Sem cobertura e sem solo, duas perguntas ficam impossiveis: se a
ocupacao do lugar muda a relacao terreno x inundacao, e se a textura do solo
explica o que o terreno nao explica.

-------------------------------------------------------------------------
A REGRA DO TEMPO
-------------------------------------------------------------------------

Cobertura muda; um mapa de 2021 aplicado a uma cheia de 2005 descreve outro
lugar. Por isso cada camada e lida no ANO DO EVENTO do ponto (inicio da
janela), e o ano efetivamente usado fica gravado ao lado do valor:

    mapbiomas_classe     MapBiomas colecao 9, 30 m, 1985-2023, so Brasil
    modis_lc             MODIS MCD12Q1 LC_Type1 (IGBP), 500 m, 2001-2023
    construido_frac      GHSL built surface / area da celula, 100 m, epoca de
                         5 anos imediatamente anterior ou igual ao evento
    worldcover_2021      ESA WorldCover 10 m: ESTADO DE 2021, declarado
    argila/areia_pct     SoilGrids 250 m, media de 0 a 30 cm: solo tratado como
                         estavel na escala do projeto, declarado

Ano fora da cobertura da camada usa o extremo mais proximo e grava
`*_ano` diferente do ano do evento: quem le sabe que o valor e emprestado.
Ponto sem data nao recebe camada temporal.

-------------------------------------------------------------------------
O QUE ISTO NAO FAZ
-------------------------------------------------------------------------

Nao classifica mecanismo, nao decide rotulo e nao entra em ajuste sozinho:
grava uma tabela por `ponto_id` que o ds06 junta ao dataset. Nao preenche o
que a camada mascara (fora do Brasil o MapBiomas fica nulo).

Uso:
    python scripts/geo/cob01_cobertura_e_solo.py            # coleta, retomando lotes
    python scripts/geo/cob01_cobertura_e_solo.py --juntar   # so consolida os lotes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("suscetibilidade", "servico"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

from proveniencia import agora_utc, sha256_json, sha256_arquivo  # noqa: E402

SAIDA = REPO / "local_runs" / "cob-01-cobertura-e-solo"
LOTES = SAIDA / "lotes"

MAPBIOMAS = ("projects/mapbiomas-public/assets/brazil/lulc/collection9/"
             "mapbiomas_collection90_integration_v1")
MAPBIOMAS_ANOS = (1985, 2023)
MODIS = "MODIS/061/MCD12Q1"
MODIS_ANOS = (2001, 2023)
GHSL = "JRC/GHSL/P2023A/GHS_BUILT_S"
GHSL_EPOCAS = tuple(range(1975, 2021, 5))   # 2025/2030 sao projecoes, nao observacao
WORLDCOVER = "ESA/WorldCover/v200/2021"
SOILGRIDS = {"argila": "projects/soilgrids-isric/clay_mean",
             "areia": "projects/soilgrids-isric/sand_mean"}
CAMADAS_SOLO = ("0-5cm", "5-15cm", "15-30cm")
AREA_CELULA_GHSL_M2 = 100 * 100

COLUNAS = ("mapbiomas_classe", "modis_lc", "construido_frac", "worldcover_2021",
           "argila_pct_0_30", "areia_pct_0_30")

TAMANHO_DO_LOTE = 4000
TRABALHADORES = 6
TENTATIVAS = 4


def ano_de_referencia(janela_inicio: pd.Series) -> pd.Series:
    data = pd.to_datetime(janela_inicio, errors="coerce", utc=True)
    return data.dt.year.astype("Int64")


def anos_das_camadas(ano: int | None) -> dict:
    """Ano efetivamente lido em cada camada temporal."""
    if ano is None:
        return {"mapbiomas": None, "modis": None, "ghsl": None}
    return {"mapbiomas": int(np.clip(ano, *MAPBIOMAS_ANOS)),
            "modis": int(np.clip(ano, *MODIS_ANOS)),
            "ghsl": max([e for e in GHSL_EPOCAS if e <= ano] or [GHSL_EPOCAS[0]])}


def imagem(ee, ano: int | None):
    """Pilha de bandas para um ano: temporais (se houver ano) + estaticas."""
    img = ee.Image(WORLDCOVER).select("Map").rename("worldcover_2021")
    for nome, asset in SOILGRIDS.items():
        bandas = [f"{asset.split('/')[-1].replace('_mean', '')}_{c}_mean" for c in CAMADAS_SOLO]
        # g/kg -> %: o SoilGrids grava textura em g/kg
        img = img.addBands(ee.Image(asset).select(bandas).reduce(ee.Reducer.mean())
                           .divide(10).rename(f"{nome}_pct_0_30"))
    a = anos_das_camadas(ano)
    if ano is not None:
        img = img.addBands(ee.Image(MAPBIOMAS).select(f"classification_{a['mapbiomas']}")
                           .rename("mapbiomas_classe"))
        img = img.addBands(ee.ImageCollection(MODIS).filter(
            ee.Filter.calendarRange(a["modis"], a["modis"], "year")).first()
            .select("LC_Type1").rename("modis_lc"))
        img = img.addBands(ee.ImageCollection(GHSL).filter(
            ee.Filter.eq("system:index", str(a["ghsl"]))).first()
            .select("built_surface").divide(AREA_CELULA_GHSL_M2).rename("construido_frac"))
    return img


def planejar(pontos: pd.DataFrame) -> list[dict]:
    """Lotes deterministicos: por ano, ponto_id ordenado, tamanho fixo."""
    p = pontos[["ponto_id", "lat", "lon"]].copy()
    p["ano"] = ano_de_referencia(pontos["janela_inicio"])
    lotes = []
    for ano, s in p.groupby(p.ano.fillna(-1).astype(int)):
        s = s.sort_values("ponto_id")
        for i in range(0, len(s), TAMANHO_DO_LOTE):
            parte = s.iloc[i:i + TAMANHO_DO_LOTE]
            lotes.append({"nome": f"{'sem_ano' if ano < 0 else ano}_{i // TAMANHO_DO_LOTE:04d}",
                          "ano": None if ano < 0 else int(ano),
                          "pontos": parte[["ponto_id", "lat", "lon"]]})
    return lotes


def identidade_lote(lote: dict) -> str:
    return sha256_json({"ano": lote["ano"], "receita": {
                              "mapbiomas": [MAPBIOMAS, MAPBIOMAS_ANOS],
                              "modis": [MODIS, MODIS_ANOS], "ghsl": [GHSL, GHSL_EPOCAS],
                              "worldcover": WORLDCOVER, "solo": SOILGRIDS,
                              "camadas_solo": CAMADAS_SOLO, "area_ghsl_m2": AREA_CELULA_GHSL_M2,
                              "escala_m": 30, "agregacao_solo": "media_gkg_div10",
                              "versao": 1},
                              "pontos": [[str(pid), float(lat), float(lon)] for pid, lat, lon in
                                         lote["pontos"][["ponto_id", "lat", "lon"]].itertuples(index=False, name=None)]})


def caminho_lote(lote: dict) -> Path:
    return LOTES / f"{lote['nome']}_{identidade_lote(lote).split(':')[-1][:16]}.parquet"


def coletar_lote(ee, lote: dict) -> Path:
    # O nome ordinal pode repetir depois de trocar dataset, datas ou amostra.
    # A identidade inclui ano, IDs E coordenadas; material antigo nao e apagado.
    assinatura = identidade_lote(lote)
    destino = caminho_lote(lote)
    entrada = destino.with_suffix(".entrada.json")
    if destino.exists():
        meta = json.loads(entrada.read_text(encoding="utf-8")) if entrada.exists() else {}
        if meta.get("assinatura") != assinatura:
            raise RuntimeError("cache sem identidade verificavel; nao reutilizado")
        if meta.get("dados_sha256") != sha256_arquivo(destino):
            raise RuntimeError("hash do cache diverge; nao reutilizado")
        d = pd.read_parquet(destino, columns=["ponto_id", "ano_evento"])
        ids = lote["pontos"].ponto_id.astype(str)
        if len(d) != len(ids) or d.ponto_id.duplicated().any() or set(d.ponto_id.astype(str)) != set(ids):
            raise RuntimeError("IDs do lote gravado divergem do pedido")
        if not (d.ano_evento.isna().all() if lote["ano"] is None else (d.ano_evento == lote["ano"]).all()):
            raise RuntimeError("ano do lote gravado diverge do pedido")
        return destino
    feats = [ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"ponto_id": pid})
             for pid, lat, lon in lote["pontos"].itertuples(index=False)]
    fc = ee.FeatureCollection(feats)
    erro = None
    for tentativa in range(TENTATIVAS):
        try:
            r = imagem(ee, lote["ano"]).reduceRegions(
                collection=fc, reducer=ee.Reducer.first(), scale=30).getInfo()
            linhas = [f["properties"] for f in r["features"]]
            d = pd.DataFrame(linhas)
            if "ponto_id" not in d or len(d) != len(lote["pontos"]):
                raise RuntimeError("resposta incompleta do Earth Engine")
            if d.ponto_id.duplicated().any() or set(d.ponto_id.astype(str)) != set(lote["pontos"].ponto_id.astype(str)):
                raise RuntimeError("IDs devolvidos pelo Earth Engine divergem do pedido")
            # camada mascarada volta como ausente ou None: vira NaN declarado
            for c in COLUNAS:
                d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
            anos = anos_das_camadas(lote["ano"])
            d["ano_evento"] = lote["ano"]
            for camada, ano in anos.items():
                d[f"{camada}_ano"] = ano
            parcial = destino.with_suffix(".parcial")
            d.to_parquet(parcial, index=False)
            entrada.write_text(json.dumps({"assinatura": assinatura, "nome_logico": lote["nome"],
                                           "ano": lote["ano"], "pontos": len(d),
                                           "dados_sha256": sha256_arquivo(parcial)}, ensure_ascii=False), encoding="utf-8")
            parcial.replace(destino)
            return destino
        except Exception as e:  # noqa: BLE001 - rede e cota do servidor
            erro = e
            time.sleep(15 * 2 ** tentativa)
    raise RuntimeError(f"{lote['nome']}: {TENTATIVAS} tentativas falharam: {erro}")


def juntar(lotes: list[dict] | None = None) -> tuple[pd.DataFrame, dict]:
    # O main fornece a lista da versao atual: nao concatena caches de outras
    # datas/amostras nem os ordinais historicos preservados no mesmo diretorio.
    arquivos = [caminho_lote(l) for l in lotes] if lotes is not None else sorted(LOTES.glob("*.parquet"))
    partes = [pd.read_parquet(p) for p in arquivos]
    d = pd.concat(partes, ignore_index=True)
    if d.ponto_id.duplicated().any():
        raise ValueError("ponto coletado em mais de um lote")
    colunas = list(COLUNAS)
    # Emprestado e o ano FORA da cobertura da camada. A epoca do GHSL anterior
    # ao evento e a regra dele, e nao emprestimo: so conta antes de 1975 ou
    # depois da ultima epoca observada mais 4 anos.
    ano = d.ano_evento.astype("float")
    cobertura = {"mapbiomas": MAPBIOMAS_ANOS, "modis": MODIS_ANOS,
                 "ghsl": (GHSL_EPOCAS[0], GHSL_EPOCAS[-1] + 4)}
    emprestado = {c: int(((ano < lo) | (ano > hi)).sum()) for c, (lo, hi) in cobertura.items()}
    resumo = {"pontos": len(d), "lotes": len(partes),
              "preenchimento": {c: round(float(d[c].notna().mean()), 4) for c in colunas},
              "ano_emprestado_do_extremo": emprestado,
              "sem_ano": int(d.ano_evento.isna().sum())}
    return d, resumo


def main() -> int:
    global SAIDA, LOTES
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--juntar", action="store_true", help="so consolida os lotes ja coletados")
    ap.add_argument("--trabalhadores", type=int, default=TRABALHADORES)
    args = ap.parse_args()
    from ds06_dataset_de_treino import versao_atual

    origem = versao_atual()
    # Novas consolidacoes ficam versionadas; o cob01 historico alimentou
    # experimentos congelados e nao e sobrescrito pela ativacao do v8.
    if int(origem.name[1:]) >= 8:
        SAIDA = SAIDA / "por_dataset" / origem.name
        LOTES = SAIDA / "lotes"
        SAIDA.mkdir(parents=True, exist_ok=True)
    pontos = pd.read_parquet(origem / "pontos.parquet",
                             columns=["ponto_id", "lat", "lon", "janela_inicio"])
    LOTES.mkdir(parents=True, exist_ok=True)
    lotes = planejar(pontos)
    if not args.juntar:
        import ee
        ee.Initialize()
        falta = [l for l in lotes if not caminho_lote(l).exists()]
        print(f"{len(lotes)} lotes, {len(falta)} a coletar", flush=True)
        falhas = []
        with ThreadPoolExecutor(max_workers=args.trabalhadores) as pool:
            futuros = {pool.submit(coletar_lote, ee, l): l["nome"] for l in falta}
            for i, f in enumerate(as_completed(futuros), start=1):
                try:
                    f.result()
                    print(f"  {i}/{len(falta)} {futuros[f]} ok", flush=True)
                except Exception as e:  # noqa: BLE001
                    falhas.append({"lote": futuros[f], "erro": str(e)[:200]})
                    print(f"  {i}/{len(falta)} {futuros[f]} FALHA {e}", flush=True)
        if falhas:
            (SAIDA / "falhas.json").write_text(json.dumps(falhas, indent=2), encoding="utf-8")
            print(f"{len(falhas)} lotes falharam; rode de novo para retomar", flush=True)
            return 1
    d, resumo = juntar(lotes)
    faltando = set(pontos.ponto_id) - set(d.ponto_id)
    d.to_parquet(SAIDA / "cobertura_e_solo.parquet", index=False)
    manifesto = {
        "tipo": "cobertura_e_solo_nos_pontos", "quando_utc": agora_utc(),
        "dataset_de_origem": origem.name, "pontos_sem_coleta": len(faltando),
        "camadas": {"mapbiomas": MAPBIOMAS, "modis": MODIS, "ghsl": GHSL,
                    "worldcover": WORLDCOVER, "soilgrids": SOILGRIDS},
        "regra_do_tempo": ("camada temporal lida no ano do inicio da janela do evento; "
                           "fora da cobertura, extremo mais proximo com o ano gravado"),
        "estaticas": {"worldcover_2021": "estado de 2021", "solo": "estavel na escala do projeto"},
        **resumo,
        "nao_e": ("nao e rotulo nem variavel de treino por si: e o contexto de cobertura e "
                  "solo sob cada ponto, com o ano de cada camada"),
    }
    manifesto["hash_resumo"] = sha256_json({k: v for k, v in manifesto.items() if k != "quando_utc"})
    (SAIDA / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps({k: manifesto[k] for k in ("pontos", "pontos_sem_coleta", "preenchimento",
                                                 "ano_emprestado_do_extremo")}, indent=1))
    return 0 if not faltando else 1


if __name__ == "__main__":
    raise SystemExit(main())
