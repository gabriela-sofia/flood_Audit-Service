"""
Resolve as `camadas` de um ponto a partir da grade de 120 m ja calculada.

O PROBLEMA QUE ESTE MODULO RESOLVE:

`inferir()` espera receber `camadas` prontas -- HAND, TWI, elevacao,
declividade, chuva. Ele nao sabe calcular feature a partir de lat/lon. Sem
alguem para preencher isso, qualquer coordenada nova cai em `insufficient_data`
no portao G4, sempre, e o transporte HTTP nao cumpre a promessa do esboco de
telas ("voce entrega uma geometria, ele devolve").

-------------------------------------------------------------------------
A ESCOLHA: celula mais proxima da grade do SVC-03, e nao geoprocessamento
-------------------------------------------------------------------------

Havia tres caminhos. O escolhido e o (b): buscar a celula mais proxima na
grade de 120 m que o SVC-03 ja gerou.

  (a) API fina, exigindo `camadas` no corpo. Continua valendo -- ver
      `resolver()`, que so age quando a requisicao NAO trouxe as camadas. Mas
      sozinho nao cumpre a promessa de "so a geometria".
  (b) ESCOLHIDO. Lookup espacial sobre `svc-03-grade/grade_*.csv`.
  (c) Reviver o geoprocessamento sob demanda do prototipo de julho: amostrar o
      raster e chamar a API de chuva ao vivo para qualquer coordenada.

Por que (b) e nao (c), alem do esforco:

o terreno da grade vem da MESMA cadeia canonica em que o modelo foi ajustado
-- WhiteboxTools 30 m, limiar de canal 0,1123 km2. Recalcular por requisicao
abriria a porta para servir um valor derivado por outra cadeia, que e
exatamente o defeito que o cartao de resposta teve: HAND de 10 m exibido ao
lado de escore calculado com HAND de 30 m. Aqui o risco seria pior, porque
entraria no escore em vez de so na figura.

A grade tambem ja foi conferida contra o contrato: o SVC-03 sorteia 25 celulas
por regiao e exige que `inferir()` reproduza o valor da grade (25/25 nas tres).
Entao o que este modulo devolve ja tem prova de que bate com o servico.

O (c) continua sendo o caminho para cumprir a promessa literalmente, em
coordenada arbitraria fora da cobertura. A fronteira deste modulo foi desenhada
para que ele possa ser trocado sem tocar no transporte.

-------------------------------------------------------------------------
O TETO DE DISTANCIA, e por que ele nao e opcional
-------------------------------------------------------------------------

Vizinho mais proximo SEM teto responde qualquer coisa. Uma coordenada no meio
do Atlantico gruda na celula costeira de Recife mais proxima e recebe escore
como se fosse terra firme. Isso e escore por analogia -- precisamente o que os
portoes G2 e G3 existem para impedir, reintroduzido por baixo deles.

O teto e METADE DA DIAGONAL da celula: o ponto tem de cair DENTRO de alguma
celula servida, nao apenas perto dela. Celula que a grade recusou (NaN, fora do
dominio) nao entra na busca -- e nao ser servida ali e resposta, nao lacuna.

-------------------------------------------------------------------------
A CHUVA CONTINUA SENDO CENARIO
-------------------------------------------------------------------------

A grade nao tem coluna de chuva de proposito: na escala de ~11 km em que ela e
medida, Recife inteiro cabe em quatro celulas, entao ela desloca o escore sem
mudar o ordenamento (`ext_chuva_estado_do_projeto_v1.md`, e a regra do E5).

Quem chama declara o cenario pelo nome. **Nao ha cenario padrao**: se o modelo
da regiao exige chuva e a requisicao nao declarou nada, este modulo devolve o
que falta e o contrato recusa. Escolher a mediana em silencio faria o escore
depender de um valor que nao aparece em lugar nenhum da resposta.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts"))

from chuva.contrato_chuva import (  # noqa: E402
    SCENARIOS, scenario_evidence, scenario_values,
)

RUNS = REPO / "local_runs"
GRADE = RUNS / "svc-03-grade"

# Vem do SVC-03: PASSO=4 sobre raster de 30 m. Duplicado aqui de proposito --
# importar o svc03 traria rasterio e a leitura de raster para dentro de um
# modulo que so precisa ler CSV.
RESOLUCAO_M = 120.0
RAIO_MAXIMO_M = RESOLUCAO_M * math.sqrt(2) / 2   # metade da diagonal: ~84,85 m

# Fonte unica importada tambem pelo SVC-03. Nenhum cenario e padrao.
CENARIOS_CHUVA = {nome: scenario_values(nome) for nome in SCENARIOS}

METROS_POR_GRAU_LAT = 111_320.0


class GradeDaRegiao:
    """Indice espacial sobre as celulas SERVIVEIS de uma regiao.

    Celula com escore NaN foi recusada pelo portao de dominio na hora de gerar
    a grade. Ela e excluida do indice: o modelo se cala ali, e o vizinho mais
    proximo nao pode falar no lugar dele.
    """

    def __init__(self, regiao: str, caminho: Path):
        import pandas as pd
        from scipy.spatial import cKDTree

        d = pd.read_csv(caminho)
        colunas_escore = [c for c in d.columns if c.startswith("escore")]
        if colunas_escore:
            # basta um cenario ter escore para a celula estar dentro do dominio
            # do terreno; a recusa por chuva e por cenario, nao por celula
            d = d[d[colunas_escore].notna().any(axis=1)]
        self.regiao = regiao
        self.camadas = [c for c in d.columns
                        if c not in ("lat", "lon")
                        and not c.startswith(("escore", "ic95"))]
        self.lat = d["lat"].to_numpy(dtype=float)
        self.lon = d["lon"].to_numpy(dtype=float)
        self.valores = {c: d[c].to_numpy(dtype=float) for c in self.camadas}

        # Projecao equirretangular local: para distancias de ~100 m perto do
        # centro da regiao o erro e desprezivel, e evita trazer pyproj para
        # dentro do caminho de uma requisicao.
        self.lat0 = float(np.mean(self.lat))
        self.escala_lon = math.cos(math.radians(self.lat0)) * METROS_POR_GRAU_LAT
        self.arvore = cKDTree(np.column_stack([
            self.lat * METROS_POR_GRAU_LAT, self.lon * self.escala_lon]))

    def __len__(self) -> int:
        return len(self.lat)

    def mais_proxima(self, lat: float, lon: float) -> tuple[dict | None, float]:
        """Camadas da celula que contem o ponto, ou None se nenhuma contem."""
        alvo = [lat * METROS_POR_GRAU_LAT, lon * self.escala_lon]
        dist, i = self.arvore.query(alvo, k=1)
        if not np.isfinite(dist) or dist > RAIO_MAXIMO_M:
            return None, float(dist)
        return ({c: float(self.valores[c][i]) for c in self.camadas},
                float(dist))


TER01 = RUNS / "ter-01-cadeia-harmonizada"
INDICE_DE_TILES = GRADE / "indice_de_tiles.json"
# Caixa do Brasil com folga: so para nao indexar as ~700 AOIs estrangeiras do
# CEMS, que nenhuma regiao do produto consulta.
CAIXA_BRASIL = (-74.5, -34.5, -28.5, 6.0)   # lon_min, lat_min, lon_max, lat_max
RASTER_DA_CAMADA = {
    "elevation_m": "dem_filled.tif",
    "slope_deg": "slope_deg_wbt.tif",
    "hand_m": "hand_dinf.tif",
    "twi_dinf": "twi_dinf.tif",
}


class CoberturaPorTiles:
    """A mesma pergunta que `GradeDaRegiao` responde, lida direto dos rasters.

    -------------------------------------------------------------------------
    POR QUE NAO UMA GRADE CSV, COMO NAS CIDADES
    -------------------------------------------------------------------------

    A grade de 120 m de uma cidade tem ate ~170 mil celulas e cabe numa
    KD-tree carregada no startup. A do Centro-Oeste teria ~84 milhoes: a
    mesma arquitetura nao atravessa a mudanca de escala, e forcar CSV aqui
    seria gerar dezenas de gigabytes para consultar meia duzia de pontos.

    E a grade nunca foi a fonte -- ela e uma SUBAMOSTRA (1 pixel a cada 4) dos
    quatro rasters que a cadeia de terreno ja gravou em `ter-01`. Ler o pixel
    do raster e a mesma cadeia (WhiteboxTools 30 m, limiar 0,1123 km2), sem a
    subamostragem. Nao e geoprocessamento sob demanda: nada e derivado na hora
    da requisicao; so se le o que a cadeia ja derivou.

    -------------------------------------------------------------------------
    O TILE MAIS CENTRAL, e por que
    -------------------------------------------------------------------------

    Os tiles do GFD foram derivados com 0,1 grau de margem e se sobrepoem.
    HAND e area de contribuicao sao piores perto da borda da janela, porque a
    bacia que drena para o ponto pode estar cortada. Quando mais de um tile
    contem o ponto, vence aquele cujo centro esta mais perto dele -- e o que
    poe o ponto o mais longe possivel de uma borda.

    Pixel com valor ausente (oceano, borda sem dado) nao e zero: devolve
    `None`, e o G4 do contrato recusa, exatamente como a celula recusada da
    grade.
    """

    modo = "pixel_30m_do_tile_derivado"

    def __init__(self, raiz: Path = None, cache: Path = None):
        import threading
        self.raiz = TER01 if raiz is None else raiz
        self.tiles = self._indexar(self.raiz,
                                   INDICE_DE_TILES if cache is None else cache)
        self._abertos: dict = {}
        self._transformadores: dict = {}
        self._trava = threading.Lock()

    @staticmethod
    def _indexar(raiz: Path, cache: Path) -> list:
        """Caixa WGS84 de cada diretorio completo, com cache em disco.

        Abrir mil rasters no startup custava 18 s. O cache guarda a caixa por
        diretorio e so abre os que ele ainda nao viu -- que e o caso normal
        enquanto a fila de terreno continua acrescentando tiles.
        """
        import json as _json
        if not raiz.exists():
            return []
        visto = {}
        if cache.exists():
            try:
                visto = _json.loads(cache.read_text(encoding="utf-8"))
            except ValueError:
                visto = {}
        mudou = False
        tiles = []
        for d in sorted(raiz.iterdir()):
            if not d.is_dir() or not (d / "run_manifest.json").exists():
                continue
            if not all((d / a).exists() for a in RASTER_DA_CAMADA.values()):
                continue
            cx = visto.get(d.name)
            if cx is None:
                try:
                    import rasterio
                    from rasterio.warp import transform_bounds
                    with rasterio.open(d / RASTER_DA_CAMADA["hand_m"]) as src:
                        cx = list(transform_bounds(src.crs, "EPSG:4326",
                                                   *src.bounds))
                except Exception:
                    continue
                visto[d.name] = cx
                mudou = True
            lon0, lat0, lon1, lat1 = cx
            b = CAIXA_BRASIL
            if lon0 < b[2] and lon1 > b[0] and lat0 < b[3] and lat1 > b[1]:
                tiles.append((d.name, cx))
        if mudou:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(_json.dumps(visto), encoding="utf-8")
            except OSError:
                pass
        return tiles

    def __len__(self) -> int:
        return len(self.tiles)

    def _raster(self, nome: str, arquivo: str):
        import rasterio
        chave = (nome, arquivo)
        if chave not in self._abertos:
            if len(self._abertos) > 64:
                for src in self._abertos.values():
                    src.close()
                self._abertos.clear()
            self._abertos[chave] = rasterio.open(self.raiz / nome / arquivo)
        return self._abertos[chave]

    def _para_crs(self, crs, lon: float, lat: float):
        from pyproj import Transformer
        k = str(crs)
        if k not in self._transformadores:
            self._transformadores[k] = Transformer.from_crs(
                "EPSG:4326", crs, always_xy=True)
        return self._transformadores[k].transform(lon, lat)

    def mais_proxima(self, lat: float, lon: float) -> tuple:
        candidatos = [(n, cx) for n, cx in self.tiles
                      if cx[0] <= lon <= cx[2] and cx[1] <= lat <= cx[3]]
        candidatos.sort(key=lambda t: ((t[1][0] + t[1][2]) / 2 - lon) ** 2
                        + ((t[1][1] + t[1][3]) / 2 - lat) ** 2)
        with self._trava:
            for nome, _ in candidatos:
                valores, dist = {}, float("inf")
                completo = True
                for camada, arquivo in RASTER_DA_CAMADA.items():
                    src = self._raster(nome, arquivo)
                    x, y = self._para_crs(src.crs, lon, lat)
                    linha, coluna = src.index(x, y)
                    if not (0 <= linha < src.height and 0 <= coluna < src.width):
                        completo = False
                        break
                    v = next(src.sample([(x, y)], masked=True))[0]
                    if np.ma.is_masked(v) or not np.isfinite(float(v)):
                        completo = False
                        break
                    valores[camada] = float(v)
                    cx_, cy_ = src.xy(linha, coluna)
                    dist = math.hypot(cx_ - x, cy_ - y)
                if completo:
                    return valores, dist
        return None, float("inf")


def carregar_grades(diretorio: Path = GRADE) -> dict:
    """Le as grades uma vez. Chamado no startup, nunca por requisicao.

    As cidades tem grade CSV. As macrorregioes, que nenhuma grade CSV comporta,
    recebem a `CoberturaPorTiles` -- uma instancia so, compartilhada, porque a
    busca e por ponto e a regiao ja foi decidida pelo G2 antes de chegar aqui.

    Sem rasterio ou sem `ter-01` no disco, as cidades continuam servidas e as
    macrorregioes recusam em G4 -- degradacao declarada no log de startup, e
    nao excecao que derrubaria o servico inteiro.
    """
    grades = {}
    if not diretorio.exists():
        return grades
    for p in sorted(diretorio.glob("grade_*.csv")):
        regiao = p.stem.replace("grade_", "")
        grades[regiao] = GradeDaRegiao(regiao, p)
    try:
        import regioes as _reg
        macro = [r for r, c in _reg.SEMENTES.items()
                 if c.get("tipo") == "macrorregiao" and r not in grades]
        if macro:
            cobertura = CoberturaPorTiles()
            if len(cobertura):
                for r in macro:
                    grades[r] = cobertura
    except Exception as e:
        print(f"[resolver_camadas] cobertura por tiles indisponivel: "
              f"{type(e).__name__}: {e}")
    return grades


# Espelha `DERIVADAS_RELATIVAS` do SVC-02: a chave e a variavel do vetor, o
# valor e a coluna que a grade de fato tem.
ORIGEM_DAS_DERIVADAS = {"elevation_rel_m": "elevation_m"}


def resolver(requisicao: dict, modelo: dict, grades: dict,
             regiao: str) -> tuple[dict, dict]:
    """Preenche as camadas que faltam. Devolve (requisicao, procedencia).

    Nao substitui o que a requisicao ja trouxe: camada declarada explicitamente
    ganha da grade sempre. Quem chama pode ter dado melhor que o nosso, e o
    contrato nao e dono da fonte -- e dono do portao.
    """
    feats = modelo["features"]
    faltam_chuva = [f for f in feats if f.startswith("rain_")]
    cenario_nome = (requisicao.get("cenario_de_chuva") or "").strip()
    cenario = CENARIOS_CHUVA.get(cenario_nome, {})
    evidencia_cenario = scenario_evidence(
        cenario_nome, modelo.get("faixa_dominio_5_95")) if cenario_nome else None

    pontos = (requisicao.get("geometria") or {}).get("pontos") or []
    resolvidos, distancias, fora = 0, [], 0
    novos = []
    for p in pontos:
        camadas = dict(p.get("camadas") or {})
        ausentes = [f for f in feats if camadas.get(f) is None]
        # A grade guarda o que se le de um raster: `elevation_m`. O vetor usa
        # `elevation_rel_m`, a mesma elevacao contra o P1 da regiao -- e a
        # grade nao tem essa coluna. Por isso buscamos tambem a ORIGEM de cada
        # derivada; a subtracao e do `SVC-02`, que tem o baseline da regiao.
        # E a origem so entra se a propria derivada faltar: ponto que ja trouxe
        # a relativa nao precisa da absoluta, e ir busca-la marcaria valor
        # preenchido onde nao houve resolucao nenhuma.
        ausentes += [o for f, o in ORIGEM_DAS_DERIVADAS.items()
                     if f in ausentes and o not in ausentes
                     and camadas.get(o) is None]
        if ausentes and regiao in grades:
            da_grade, dist = grades[regiao].mais_proxima(
                float(p["lat"]), float(p["lon"]))
            if da_grade is None:
                fora += 1
            else:
                distancias.append(dist)
                for f in ausentes:
                    if f in da_grade:
                        camadas[f] = da_grade[f]
                        resolvidos += 1
        for f in feats:
            if camadas.get(f) is None and f in cenario:
                camadas[f] = cenario[f]
        novos.append({**p, "camadas": camadas})

    saida = {**requisicao,
             "geometria": {**requisicao["geometria"], "pontos": novos}}
    procedencia = {
        "modo": getattr(grades.get(regiao), "modo",
                        "celula_mais_proxima_da_grade_120m"),
        "valores_preenchidos": resolvidos,
        "pontos_fora_de_toda_celula_servivel": fora,
        "raio_maximo_m": round(RAIO_MAXIMO_M, 2),
        "distancia_mediana_m": (round(float(np.median(distancias)), 2)
                                if distancias else None),
        "cenario_de_chuva": cenario_nome or None,
        "rain_evidence": ({
            "used": bool(faltam_chuva and cenario),
            "provider": "open_meteo",
            "product": "era5_land",
            **evidencia_cenario,
            "limitations": [
                "scenario forcing is not a point measurement",
                "Recife temporal adherence is low; rain may identify event dates",
            ],
        } if evidencia_cenario else {
            "used": False,
            "reason": ("regional model does not use rain" if not faltam_chuva
                       else "rain scenario was not supplied"),
        }),
        "cenarios_disponiveis": list(CENARIOS_CHUVA),
        "chuva_exigida_pelo_modelo": faltam_chuva,
        "cadeia_do_terreno": ("WhiteboxTools 30 m, limiar de canal 0,1123 km2 "
                              "-- a mesma do ajuste"),
        "nao_e": ("nao e geoprocessamento sob demanda: fora da cobertura da "
                  "grade o contrato recusa, e nao extrapola"),
    }
    # O `detalhe` da recusa dira apenas que `rain_max_24h` esta ausente, o que
    # e verdade mas nao ajuda quem chamou. Dizer o que fazer e trabalho de
    # transporte, nao de contrato -- entao a dica vive aqui, ao lado do
    # motivo, e nao dentro da resposta do portao.
    if faltam_chuva and not cenario:
        procedencia["dica"] = (
            f"o modelo desta regiao usa {faltam_chuva}; declare "
            f"`cenario_de_chuva` (um de {list(CENARIOS_CHUVA)}) ou envie os "
            "valores em `camadas`. Nao ha cenario padrao de proposito")
    if evidencia_cenario and evidencia_cenario.get("extrapolation"):
        procedencia["cenario_extrapolativo"] = True
        procedencia["dica_extrapolacao"] = (
            "o cenario ultrapassa o dominio p5-p95 do ajuste; recusas de G5 "
            "devem ser atribuidas explicitamente ao forcing extrapolativo")
    if fora:
        procedencia["dica"] = (
            f"{fora} ponto(s) fora de toda celula servivel da grade "
            f"(raio de {RAIO_MAXIMO_M:.2f} m). Ali o modelo se cala: a celula "
            "foi recusada pelo portao de dominio ou esta fora do territorio")
    return saida, procedencia
