"""Pilha Sentinel-1 do MESMO lugar em varias datas, sobre AOIs CEMS com pontos fisicos.

-------------------------------------------------------------------------
POR QUE ESTA AQUISICAO EXISTE
-------------------------------------------------------------------------

A linhagem `aprendizado_conjunto_v1` (16/10/2026) mostrou que compartilhar o
encoder fisico com a segmentacao SAR do Sen1Floods11 muda o encoder e nao muda
a ordenacao fisica (+0,0015 de AUC, IC97,5% [-0,0018; +0,0045]). O diagnostico
foi que a supervisao SAR vinha dos MESMOS rotulos dos pontos fisicos e so
cobria 10 componentes. E o grafo causal mostrou que nenhuma relacao
terreno -> inundacao fecha na base, porque todo registro existe por causa da
inundacao (selecao S=1).

O que falta nas duas frentes e a mesma coisa: o mesmo lugar, observado em
varias datas, com e sem evento, por um instrumento que nao depende do rotulo.
Este modulo produz isso e nada alem disso:

    (a) frequencia de agua observada por pixel, fora das janelas de evento,
        com agua permanente JRC separada (nunca somada a frequencia);
    (b) o quadro pixel x data (deteccao SAR por data admitida), sobre o qual
        um desenho dentro do lugar podera, depois, usar chuva como exposicao.

-------------------------------------------------------------------------
O QUE CONTA COMO UNIDADE
-------------------------------------------------------------------------

Baixar nao e admitir. Cada linha do manifesto diz se a data foi SOLICITADA,
BAIXADA e ADMITIDA, e por que nao. A unidade inferencial continua sendo o
componente de dependencia (`unidade` da base conjunta); bloco, pixel e data
sao repeticoes dentro dele.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao abre as reservas (EMSR868/871 e as do `confirmacao/reserva.json`), nao
treina, nao promove modelo, nao toca o servico e nao le rotulo de evento para
decidir deteccao: o limiar de agua e calibrado contra o JRC, fora das janelas
de evento.

Fases (cada uma grava antes de a seguinte comecar):
    python scripts/sar/pilha_s1_mesmo_lugar.py protocolo
    python scripts/sar/pilha_s1_mesmo_lugar.py blocos
    python scripts/sar/pilha_s1_mesmo_lugar.py catalogo
    python scripts/manutencao/executar_serializado.py -- python scripts/sar/pilha_s1_mesmo_lugar.py baixar
    python scripts/manutencao/executar_serializado.py -- python scripts/sar/pilha_s1_mesmo_lugar.py derivar
    python scripts/sar/pilha_s1_mesmo_lugar.py relatorio
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "pilha-s1-mesmo-lugar" / "v1"
PONTOS = REPO / "local_runs" / "aprendizado-conjunto" / "v1" / "base" / "pontos.parquet"
TERRENO = REPO / "local_runs" / "ter-01-cadeia-harmonizada"
RESERVA = REPO / "local_runs" / "mecanismos-transferiveis" / "confirmacao" / "reserva.json"
JRC_CALIBRACAO = REPO / "local_runs" / "agua-permanente-jrc" / "calibracao.json"

COLECAO = "COPERNICUS/S1_GRD"
RESERVAS_FIXAS = ("EMSR868", "EMSR871")

# ------------------------------------------------------------------ protocolo
# Tudo que decide admissao mora aqui e entra no hash do protocolo, gravado
# ANTES do catalogo. Mudar um numero depois e protocolo novo, nao ajuste.

BLOCO_PX = 256                 # 256 x 30 m = 7,68 km, na grade ter-01 da AOI
PIXEL_M = 30.0
MAX_BLOCOS_POR_COMPONENTE = 8
MAX_BLOCOS_TOTAL = 220
DIAS_ANTES = 365
DIAS_DEPOIS = 180
JANELA_EVENTO = (-7, 21)       # dias em torno de [janela_inicio, janela_fim]
TRANSICAO_DIAS = 60            # entre janela de evento e "fora"; nao entra em (a)
COBERTURA_MINIMA = 0.90        # pegada (pre) e pixel valido (pos)
MAX_DATAS_POR_BLOCO = 60
DATAS_POR_PEDIDO = 15
TETO_PEDIDOS = 2000
TETO_GB = 4.0
VV_MEDIANA_ADMISSIVEL = (-25.0, -3.0)
LAT_NEVE = 40.0                # |lat| acima disso, meses de neve sao recusados
MESES_NEVE_NORTE = (12, 1, 2, 3)
MESES_NEVE_SUL = (6, 7, 8, 9)
ARVORE_MAX = 0.5               # fracao WorldCover classe 10 na celula de 30 m
CONSTRUIDO_MAX = 0.5           # fracao WorldCover classe 50
DECLIVE_MAX_GRAUS = 15.0       # sombra/inversao de visada (aproximacao)
JRC_PERMANENTE = 50            # occurrence >= 50: calibrado em 10/09 contra Sen1Floods11
JRC_AGUA_CALIBRACAO = 90       # nucleo certo de agua para calibrar o limiar
LIMIARES_VV = [round(-26.0 + 0.5 * i, 1) for i in range(29)]   # -26 .. -12 dB
MIN_PIXELS_CLASSE_CALIBRACAO = 200
MIN_DATAS_FORA = 10            # bloco entra em (a) com >= 10 datas fora de evento
MIN_PIXELS_OBSERVAVEIS = 1000
NODATA = -32768


def protocolo() -> dict:
    jrc = json.loads(JRC_CALIBRACAO.read_text(encoding="utf-8"))
    assert jrc["limiar_escolhido"] == JRC_PERMANENTE, "calibracao JRC mudou"
    return {
        "tipo": "protocolo_aquisicao_s1_multidata",
        "nao_e": ("nao e rotulo de inundacao, nao e reserva confirmatoria, nao e "
                  "probabilidade territorial; frequencia de deteccao SAR != suscetibilidade"),
        "versao": "pilha_s1_mesmo_lugar_v1",
        "objetivo": {
            "a": "frequencia de agua observada por pixel fora das janelas de evento; JRC permanente separado",
            "b": "quadro pixel x data (deteccao por data admitida) para desenho dentro do lugar posterior",
        },
        "fonte": {
            "colecao": COLECAO,
            "acesso": "Google Earth Engine, ee.data.computePixels (credencial local, nada gravado no repo)",
            "processamento_da_colecao": ("GRD IW; ruido termico removido, calibracao radiometrica, "
                                         "correcao de terreno Range-Doppler (SRTM/ASTER), sigma0 em dB"),
            "processamento_aqui": ("mosaico por dia e orbita relativa; dB->linear; media 3x3 (10->30 m, "
                                   "reduceResolution) na grade ter-01 da AOI; linear->dB; int16 = dB*100"),
            "alternativas": {
                "ASF": "mesma cena GRD; exige credencial EOSDIS (nao configurada)",
                "CDSE": "mesma cena GRD; exige conta Copernicus Data Space (nao configurada)",
                "Planetary Computer sentinel-1-rtc": "gamma0 RTC CC-BY-4.0, sem credencial; processamento diferente do S1_GRD do Sen1Floods11",
            },
            "por_que_ee": ("mesma colecao do Sen1Floods11 e do adquirir_s1_para_aoi.py; recorte e multilook no "
                           "servidor, entao so os blocos descem"),
            "estado_da_cota": "projeto EE em modo restrito do nivel nao comercial (aviso em 17/10/2026); pedidos interativos respondem",
        },
        "licenca": {
            "sentinel_1": ("Copernicus Sentinel data, acesso livre e aberto (Regulamento Delegado UE 1159/2013; "
                           "Legal Notice on the use of Copernicus Sentinel Data and Service Information); "
                           "atribuicao 'contains modified Copernicus Sentinel data [ano]'"),
            "jrc_gsw": "JRC Global Surface Water v1.4 (Pekel et al. 2016), CC BY 4.0",
            "worldcover": "ESA WorldCover 10 m 2021 v200, CC BY 4.0",
            "earth_engine": "termos de uso nao comercial do Earth Engine (pesquisa)",
        },
        "unidades": {
            "inferencial": "componente de dependencia (`unidade` da base conjunta CEMS: 26)",
            "eventos": "evento_id CEMS (32)",
            "repeticoes": "bloco, pixel e data sao repeticoes dentro do componente",
            "download_nao_e_admissao": True,
        },
        "selecao_de_blocos": {
            "grade": f"ter-01 da AOI ({PIXEL_M:.0f} m, CRS UTM da AOI), blocos de {BLOCO_PX} px recortados na borda",
            "candidatos": "blocos que contem >= 1 ponto fisico CEMS",
            "ordem": "com positivo e negativo primeiro, depois mais pontos; rodizio entre AOIs do componente",
            "max_por_componente": MAX_BLOCOS_POR_COMPONENTE,
            "max_total": MAX_BLOCOS_TOTAL,
            "nao_usa": "valor de terreno nem rotulo alem da presenca de ponto no bloco",
        },
        "periodo": {"dias_antes": DIAS_ANTES, "dias_depois": DIAS_DEPOIS,
                    "teto": "ontem (UTC)"},
        "janelas": {
            "evento": f"[janela_inicio{JANELA_EVENTO[0]} d, janela_fim+{JANELA_EVENTO[1]} d] de CADA evento do componente",
            "transicao": f"ate {TRANSICAO_DIAS} d alem da janela de evento; fica no quadro, nao entra em (a)",
            "fora": "o resto do periodo; so estas datas entram na frequencia (a)",
            "por_que_todos_os_eventos_do_componente": "componente com eventos em datas diferentes (EMSR772/779...) nao pode ter um deles contado como 'seco'",
        },
        "admissao_pre_download": {
            "colecao": "IW, VV e VH presentes",
            "orbita": "uma orbita relativa e uma passagem por bloco (a com mais datas cobertas; empate: mais datas na janela de evento, depois menor numero)",
            "cobertura_pegada": f">= {COBERTURA_MINIMA} do bloco pela uniao das pegadas do dia",
            "max_datas": MAX_DATAS_POR_BLOCO,
        },
        "admissao_pos_download": {
            "grade": "array devolvido com a forma pedida; grade imposta no pedido (affineTransform + crsCode)",
            "pixel_valido": f">= {COBERTURA_MINIMA} do bloco com VV e VH validos",
            "radiometria": f"mediana VV no bloco em {list(VV_MEDIANA_ADMISSIVEL)} dB",
            "neve": f"|lat| > {LAT_NEVE}: meses {MESES_NEVE_NORTE} (norte) / {MESES_NEVE_SUL} (sul) recusados (neve umida imita agua)",
            "data_repetida": "recusada",
        },
        "observabilidade_por_pixel": {
            "arvore": f"fracao WorldCover classe 10 > {ARVORE_MAX} -> nao observavel",
            "construido": f"fracao WorldCover classe 50 > {CONSTRUIDO_MAX} -> nao observavel",
            "declive": f"slope_deg_wbt ter-01 > {DECLIVE_MAX_GRAUS} graus -> nao observavel (aproximacao de sombra/inversao)",
            "permanente": f"JRC occurrence >= {JRC_PERMANENTE} -> classe separada, fora da frequencia",
        },
        "deteccao_de_agua": {
            "regra": "VV (30 m, dB) <= limiar unico global",
            "calibracao": (f"limiar em {LIMIARES_VV[0]}..{LIMIARES_VV[-1]} dB (passo 0,5) que maximiza a acuracia "
                           f"balanceada media por componente entre JRC occurrence >= {JRC_AGUA_CALIBRACAO} (agua) e "
                           "occurrence == 0 com max_extent == 0 (terra), so em pixels observaveis e datas FORA de evento"),
            "nao_usa": "rotulo CEMS, pontos fisicos, HAND ou qualquer variavel de terreno",
            "sensibilidade": "limiar +-2 dB reportado; VV e VH brutos ficam guardados para outra regra",
            "vies_conhecido": "cheia real fora de janela em pixel JRC==0 conta como falso positivo na calibracao (puxa o limiar para baixo)",
        },
        "cobertura": {
            "bloco_em_a": f">= {MIN_DATAS_FORA} datas admitidas fora de evento e >= {MIN_PIXELS_OBSERVAVEIS} pixels observaveis nao permanentes",
            "bloco_em_b": "bloco em (a) e >= 1 data admitida na janela de evento",
            "componente_coberto": "componente com >= 1 bloco coberto",
        },
        "teto": {"blocos": MAX_BLOCOS_TOTAL, "pedidos_computePixels": TETO_PEDIDOS,
                 "gb_em_disco": TETO_GB, "datas_por_pedido": DATAS_POR_PEDIDO,
                 "cota_ee": "para no primeiro erro de cota persistente (3 tentativas) e registra"},
        "reservas": {"fixas": list(RESERVAS_FIXAS), "arquivo": str(RESERVA.relative_to(REPO)),
                     "regra": "qualquer AOI/evento reservado aborta a selecao"},
        "regime": "desenvolvimento retrospectivo: AOIs e eventos ja examinados; a pilha e observacao nova",
    }


def reservadas() -> set[str]:
    r = json.loads(RESERVA.read_text(encoding="utf-8"))
    return set(RESERVAS_FIXAS) | set(r.get("reservadas", {})) | set(r.get("intocadas_outro_protocolo", []))


def _gravar(nome: str, obj) -> Path:
    SAIDA.mkdir(parents=True, exist_ok=True)
    f = SAIDA / nome
    f.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return f


def _ler(nome: str):
    return json.loads((SAIDA / nome).read_text(encoding="utf-8"))


def cmd_protocolo(_a) -> None:
    f = SAIDA / "protocolo.json"
    p = protocolo()
    if f.exists():
        antigo = _ler("protocolo.json")
        if antigo["sha256"] != sha256_json(p):
            raise SystemExit("protocolo.json ja existe com outro conteudo: versao nova, nao sobrescrita")
        print("protocolo inalterado", antigo["sha256"])
        return
    _gravar("protocolo.json", {**p, "quando_utc": agora_utc(), "sha256": sha256_json(p)})
    print("protocolo gravado", sha256_json(p))


def _exigir_protocolo() -> dict:
    p = _ler("protocolo.json")
    if p["sha256"] != sha256_json(protocolo()):
        raise SystemExit("o codigo divergiu do protocolo congelado")
    return p


# ------------------------------------------------------------------ janelas

def classificar_data(d: date, eventos: list[tuple[date, date]]) -> str:
    """evento / transicao / fora, contra TODAS as janelas do componente."""
    rotulo = "fora"
    for j0, j1 in eventos:
        a, b = j0 + timedelta(days=JANELA_EVENTO[0]), j1 + timedelta(days=JANELA_EVENTO[1])
        if a <= d <= b:
            return "evento"
        if a - timedelta(days=TRANSICAO_DIAS) <= d <= b + timedelta(days=TRANSICAO_DIAS):
            rotulo = "transicao"
    return rotulo


def mes_de_neve(d: date, lat: float) -> bool:
    if abs(lat) <= LAT_NEVE:
        return False
    return d.month in (MESES_NEVE_NORTE if lat > 0 else MESES_NEVE_SUL)


def admitir_data(valido_frac: float, vv_mediana: float, d: date, lat: float,
                 ja_vista: bool) -> str | None:
    """None = admitida; senao o motivo. Ordem fixa, primeiro motivo vence."""
    if ja_vista:
        return "data_repetida"
    if not np.isfinite(valido_frac) or valido_frac < COBERTURA_MINIMA:
        return "pixel_valido_insuficiente"
    if not (VV_MEDIANA_ADMISSIVEL[0] <= vv_mediana <= VV_MEDIANA_ADMISSIVEL[1]):
        return "radiometria_anomala"
    if mes_de_neve(d, lat):
        return "estacao_de_neve"
    return None


# ------------------------------------------------------------------ blocos

def _pontos_cems() -> pd.DataFrame:
    p = pd.read_parquet(PONTOS)
    c = p[p.fonte == "cems"].copy()
    proibidas = reservadas()
    for col in ("unidade", "evento_id", "unidade_terreno"):
        ids = c[col].astype(str).str.replace("CEMS:", "", regex=False).str.split("_").str[0]
        hit = sorted(set(ids) & proibidas)
        if hit:
            raise SystemExit(f"reserva na base de pontos ({col}): {hit}")
    return c


def selecionar_blocos(c: pd.DataFrame) -> tuple[list[dict], int]:
    import rasterio
    from pyproj import Transformer

    cand = []
    for aoi, g in c.groupby("unidade_terreno"):
        with rasterio.open(TERRENO / aoi / "dem_filled.tif") as s:
            tr = Transformer.from_crs("EPSG:4326", s.crs, always_xy=True)
            x, y = tr.transform(g.lon.values, g.lat.values)
            r, k = rasterio.transform.rowcol(s.transform, x, y)
            r, k = np.asarray(r), np.asarray(k)
            br, bc = r // BLOCO_PX, k // BLOCO_PX
            for (i, j) in sorted(set(zip(br.tolist(), bc.tolist()))):
                m = (br == i) & (bc == j)
                pos = int(g.classe.values[m].sum())
                cand.append({
                    "aoi": aoi, "componente": g.unidade.iloc[0], "evento_id": g.evento_id.iloc[0],
                    "br": int(i), "bc": int(j), "n_pontos": int(m.sum()), "n_pos": pos,
                    "n_neg": int(m.sum()) - pos, "ambos": bool(pos > 0 and m.sum() > pos),
                    "linha0": int(i * BLOCO_PX), "coluna0": int(j * BLOCO_PX),
                    "altura": int(min(BLOCO_PX, s.height - i * BLOCO_PX)),
                    "largura": int(min(BLOCO_PX, s.width - j * BLOCO_PX)),
                    "crs": str(s.crs), "transform_aoi": list(s.transform)[:6],
                    "lat": float(g.lat.values[m].mean()),
                })
    d = pd.DataFrame(cand)
    d = d[(d.altura > 0) & (d.largura > 0)]
    escolhidos = []
    for comp, g in d.groupby("componente"):
        filas = {a: h.sort_values(["ambos", "n_pontos", "br", "bc"], ascending=[False, False, True, True])
                 .to_dict("records") for a, h in g.groupby("aoi")}
        ordem_aoi = sorted(filas, key=lambda a: (-max(r["ambos"] for r in filas[a]),
                                                 -sum(r["n_pontos"] for r in filas[a]), a))
        pegos = []
        while len(pegos) < MAX_BLOCOS_POR_COMPONENTE and any(filas.values()):
            for a in ordem_aoi:
                if filas[a] and len(pegos) < MAX_BLOCOS_POR_COMPONENTE:
                    pegos.append(filas[a].pop(0))
        escolhidos += pegos
    escolhidos = escolhidos[:MAX_BLOCOS_TOTAL]
    for b in escolhidos:
        t = b["transform_aoi"]
        b["x0"] = t[2] + b["coluna0"] * t[0]
        b["y0"] = t[5] + b["linha0"] * t[4]
        b["bloco"] = f"{b['aoi']}_b{b['br']:03d}_{b['bc']:03d}"
    return escolhidos, len(d)


def cmd_blocos(_a) -> None:
    _exigir_protocolo()
    c = _pontos_cems()
    eventos = (c.groupby(["unidade", "evento_id"])
               .agg(j0=("janela_inicio", "min"), j1=("janela_fim", "max"),
                    precisao=("precisao_data", lambda s: sorted(set(s))))
               .reset_index())
    blocos, n_cand = selecionar_blocos(c)
    janelas = {u: [[r.j0, r.j1, r.evento_id] for r in g.itertuples()] for u, g in eventos.groupby("unidade")}
    _gravar("blocos.json", {
        "tipo": "selecao_de_blocos_s1_multidata", "quando_utc": agora_utc(),
        "nao_e": "nao e amostra aleatoria do territorio: blocos onde ja ha ponto fisico CEMS",
        "candidatos": n_cand, "selecionados": len(blocos),
        "componentes_com_bloco": len({b["componente"] for b in blocos}),
        "componentes_na_base": int(c.unidade.nunique()), "eventos_na_base": int(c.evento_id.nunique()),
        "janelas_por_componente": janelas,
        "eventos": eventos.assign(precisao=eventos.precisao.map(list)).to_dict("records"),
        "blocos": blocos,
    })
    print(len(blocos), "blocos de", n_cand, "candidatos;", len({b['componente'] for b in blocos}), "componentes")


# ------------------------------------------------------------------ catalogo

def _ee():
    import warnings
    warnings.filterwarnings("ignore", message=".*restricted mode.*")
    import ee
    ee.Initialize()
    return ee


def _bloco_4326(b: dict):
    from pyproj import Transformer
    from shapely.geometry import Polygon

    tr = Transformer.from_crs(b["crs"], "EPSG:4326", always_xy=True)
    x0, y0 = b["x0"], b["y0"]
    x1, y1 = x0 + b["largura"] * PIXEL_M, y0 - b["altura"] * PIXEL_M
    xs = np.r_[np.linspace(x0, x1, 9), np.full(9, x1), np.linspace(x1, x0, 9), np.full(9, x0)]
    ys = np.r_[np.full(9, y0), np.linspace(y0, y1, 9), np.full(9, y1), np.linspace(y1, y0, 9)]
    lo, la = tr.transform(xs, ys)
    return Polygon(zip(lo, la))


def periodo_do_bloco(b: dict, janelas: list) -> tuple[date, date]:
    proprio = [w for w in janelas if w[2] == b["evento_id"]][0]
    ini = date.fromisoformat(proprio[0]) - timedelta(days=DIAS_ANTES)
    fim = max(date.fromisoformat(w[1]) for w in janelas) + timedelta(days=DIAS_DEPOIS)
    return ini, min(fim, datetime.now(timezone.utc).date() - timedelta(days=1))


def catalogar_bloco(ee, b: dict, janelas: list) -> dict:
    from shapely.geometry import Polygon, shape
    from shapely.ops import unary_union

    poli = _bloco_4326(b)
    ini, fim = periodo_do_bloco(b, janelas)
    col = (ee.ImageCollection(COLECAO)
           .filterBounds(ee.Geometry.Polygon([list(poli.exterior.coords)]))
           .filterDate(ini.isoformat(), (fim + timedelta(days=1)).isoformat())
           .filter(ee.Filter.eq("instrumentMode", "IW")))
    info = col.getInfo()["features"]
    eventos = [(date.fromisoformat(w[0]), date.fromisoformat(w[1])) for w in janelas]
    grupos: dict = {}
    recusas = []
    for f in info:
        p = f["properties"]
        quando = datetime.fromtimestamp(p["system:time_start"] / 1000, timezone.utc)
        pol = p.get("transmitterReceiverPolarisation", [])
        reg = {"id": f["id"], "quando_utc": quando.isoformat(timespec="seconds"),
               "orbita": p.get("relativeOrbitNumber_start"), "passagem": p.get("orbitProperties_pass"),
               "plataforma": p.get("platform_number")}
        if not ("VV" in pol and "VH" in pol):
            recusas.append({**reg, "motivo": "polarizacao_sem_vv_vh"})
            continue
        chave = (reg["orbita"], reg["passagem"], quando.date().isoformat())
        grupos.setdefault(chave, {"cenas": [], "geo": []})
        grupos[chave]["cenas"].append(reg)
        # A pegada vem em `system:footprint` (LinearRing); `getInfo` de colecao
        # nao traz `geometry` por imagem.
        pe = p["system:footprint"]
        grupos[chave]["geo"].append(Polygon(pe["coordinates"]) if pe["type"] == "LinearRing" else shape(pe))
    datas = []
    for (orb, pas, d), g in grupos.items():
        cob = unary_union(g["geo"]).intersection(poli).area / poli.area
        dd = date.fromisoformat(d)
        datas.append({"data": d, "orbita": orb, "passagem": pas, "cobertura_pegada": round(float(cob), 4),
                      "janela": classificar_data(dd, eventos), "cenas": [c["id"] for c in g["cenas"]],
                      "quando_utc": min(c["quando_utc"] for c in g["cenas"]),
                      "plataformas": sorted({str(c["plataforma"]) for c in g["cenas"]})})
    df = pd.DataFrame(datas)
    escolha = None
    if len(df):
        ok = df[df.cobertura_pegada >= COBERTURA_MINIMA]
        if len(ok):
            r = (ok.groupby(["orbita", "passagem"])
                 .agg(n=("data", "size"), n_ev=("janela", lambda s: int((s == "evento").sum())))
                 .reset_index().sort_values(["n", "n_ev", "orbita"], ascending=[False, False, True]))
            escolha = {"orbita": int(r.iloc[0].orbita), "passagem": r.iloc[0].passagem,
                       "alternativas": r.to_dict("records")}
    for d in datas:
        if escolha is None or (d["orbita"], d["passagem"]) != (escolha["orbita"], escolha["passagem"]):
            d["pre"] = "orbita_diferente" if escolha else "sem_orbita_admissivel"
        elif d["cobertura_pegada"] < COBERTURA_MINIMA:
            d["pre"] = "cobertura_pegada_insuficiente"
        else:
            d["pre"] = "solicitar"
    sol = sorted([d for d in datas if d["pre"] == "solicitar"], key=lambda d: d["data"])
    if len(sol) > MAX_DATAS_POR_BLOCO:
        # Corte deterministico que preserva as datas de evento: descarta "fora"
        # mais antigas primeiro.
        sobra = len(sol) - MAX_DATAS_POR_BLOCO
        for d in sol:
            if sobra and d["janela"] == "fora":
                d["pre"] = "acima_do_teto_de_datas"
                sobra -= 1
    return {"tipo": "catalogo_s1_por_bloco",
            "nao_e": "metadados de cena e pegada; nao e cena baixada nem admitida",
            "bloco": b["bloco"], "periodo": [ini.isoformat(), fim.isoformat()],
            "n_cenas": len(info), "escolha": escolha, "datas": sorted(datas, key=lambda d: (d["data"], d["orbita"])),
            "recusas_de_cena": recusas}


def cmd_catalogo(_a) -> None:
    _exigir_protocolo()
    B = _ler("blocos.json")
    ee = _ee()
    pasta = SAIDA / "catalogo"
    pasta.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(B["blocos"]):
        f = pasta / f"{b['bloco']}.json"
        if f.exists():
            continue
        for tentativa in range(3):
            try:
                r = catalogar_bloco(ee, b, B["janelas_por_componente"][b["componente"]])
                break
            except Exception as e:  # noqa: BLE001 - rede/EE; registra e tenta de novo
                print("catalogo falhou", b["bloco"], tentativa, e)
                time.sleep(10 * (tentativa + 1))
        else:
            raise SystemExit(f"catalogo nao respondeu para {b['bloco']}")
        f.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(i, b["bloco"], r["n_cenas"], "cenas", r["escolha"] and (r["escolha"]["orbita"], r["escolha"]["passagem"]),
              sum(d["pre"] == "solicitar" for d in r["datas"]), "solicitar")
    cmd_contagem_projetada(B)


def cmd_contagem_projetada(B: dict) -> None:
    """Quantos componentes PODEM ser cobertos, antes de baixar um pixel."""
    linhas = []
    for b in B["blocos"]:
        r = json.loads((SAIDA / "catalogo" / f"{b['bloco']}.json").read_text(encoding="utf-8"))
        sol = [d for d in r["datas"] if d["pre"] == "solicitar"]
        linhas.append({"bloco": b["bloco"], "componente": b["componente"], "evento_id": b["evento_id"],
                       "n_solicitar": len(sol), "n_fora": sum(d["janela"] == "fora" for d in sol),
                       "n_evento": sum(d["janela"] == "evento" for d in sol)})
    d = pd.DataFrame(linhas)
    d["pode_a"] = d.n_fora >= MIN_DATAS_FORA
    d["pode_b"] = d.pode_a & (d.n_evento >= 1)
    pedidos = int(np.ceil(d.n_solicitar / DATAS_POR_PEDIDO).sum() + len(d))
    gb = float(d.n_solicitar.sum() * BLOCO_PX * BLOCO_PX * 2 * 2 / 1e9)
    out = {
        "tipo": "contagem_projetada_pre_download", "quando_utc": agora_utc(),
        "nao_e": "nao e cobertura admitida: so pegada e orbita, antes de qualquer pixel",
        "blocos": len(d), "datas_a_solicitar": int(d.n_solicitar.sum()),
        "pedidos_previstos": pedidos, "gb_previstos_brutos": round(gb, 3),
        "dentro_do_teto": bool(pedidos <= TETO_PEDIDOS and gb <= TETO_GB),
        "componentes_que_podem_a": int(d[d.pode_a].componente.nunique()),
        "componentes_que_podem_b": int(d[d.pode_b].componente.nunique()),
        "eventos_que_podem_b": int(d[d.pode_b].evento_id.nunique()),
        "por_componente": d.groupby("componente").agg(blocos=("bloco", "size"), datas=("n_solicitar", "sum"),
                                                      fora=("n_fora", "sum"), evento=("n_evento", "sum"),
                                                      blocos_a=("pode_a", "sum"), blocos_b=("pode_b", "sum"))
        .reset_index().to_dict("records"),
    }
    _gravar("contagem_projetada.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "por_componente"}, ensure_ascii=False))


# ------------------------------------------------------------------ baixar

def _grade_pedido(b: dict) -> dict:
    return {"dimensions": {"width": b["largura"], "height": b["altura"]},
            "affineTransform": {"scaleX": PIXEL_M, "shearX": 0, "translateX": b["x0"],
                                "shearY": 0, "scaleY": -PIXEL_M, "translateY": b["y0"]},
            "crsCode": b["crs"]}


def _multilook(ee, img, b: dict):
    """dB -> linear, media de 10 m para a celula de 30 m, linear -> dB."""
    lin = ee.Image(10).pow(img.divide(10))
    return (lin.setDefaultProjection(crs=b["crs"], scale=10)
            .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=64)
            .reproject(crs=b["crs"], crsTransform=[PIXEL_M, 0, b["x0"], 0, -PIXEL_M, b["y0"]])
            .log10().multiply(10))


def _pedido(ee, expr, b: dict) -> np.ndarray:
    req = {"expression": expr, "fileFormat": "NPY", "grid": _grade_pedido(b)}
    ultimo = None
    for tentativa in range(3):
        try:
            return np.load(io.BytesIO(ee.data.computePixels(req)))
        except Exception as e:  # noqa: BLE001
            ultimo = e
            msg = str(e).lower()
            time.sleep(20 * (tentativa + 1) if ("quota" in msg or "429" in msg or "too many" in msg) else 5)
    raise RuntimeError(f"computePixels falhou 3 vezes: {ultimo}")


def baixar_bloco(ee, b: dict, cat: dict) -> dict:
    sol = [d for d in cat["datas"] if d["pre"] == "solicitar"]

    def do_dia(d: dict, bandas: list):
        # As cenas EXATAS que o catalogo admitiu (VV+VH, orbita escolhida).
        # Filtrar a colecao por orbita e dia deixava entrar cena HH/HV (1SDH)
        # do mesmo dia, e o mosaico levantava "no band named VV".
        return ee.ImageCollection([ee.Image(i) for i in d["cenas"]]).select(bandas).mosaic()
    vv = np.full((len(sol), b["altura"], b["largura"]), NODATA, np.int16)
    vh = vv.copy()
    pedidos = 0
    for k in range(0, len(sol), DATAS_POR_PEDIDO):
        parte = sol[k:k + DATAS_POR_PEDIDO]
        bandas = []
        for i, d in enumerate(parte):
            m = do_dia(d, ["VV", "VH"])
            bandas.append(_multilook(ee, m, b).multiply(100).round().toInt16()
                          .unmask(NODATA).rename([f"VV_{i:02d}", f"VH_{i:02d}"]))
        arr = _pedido(ee, ee.Image.cat(bandas), b)
        pedidos += 1
        if arr.shape != (b["altura"], b["largura"]):
            raise RuntimeError(f"grade devolvida {arr.shape} != pedida")
        for i in range(len(parte)):
            vv[k + i] = arr[f"VV_{i:02d}"]
            vh[k + i] = arr[f"VH_{i:02d}"]
    # Estaticos: WorldCover, JRC, angulo de incidencia de uma data da orbita.
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    arvore = wc.eq(10).toFloat().reduceResolution(ee.Reducer.mean(), maxPixels=64)
    constr = wc.eq(50).toFloat().reduceResolution(ee.Reducer.mean(), maxPixels=64)
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    ang = do_dia(sol[0], ["angle"])
    est = ee.Image.cat([arvore.rename("arvore"), constr.rename("construido"),
                        jrc.select("occurrence").unmask(0).rename("jrc_occ"),
                        jrc.select("max_extent").unmask(0).rename("jrc_max"),
                        ang.rename("angulo")]).reproject(
        crs=b["crs"], crsTransform=[PIXEL_M, 0, b["x0"], 0, -PIXEL_M, b["y0"]]).toFloat().unmask(-1)
    s = _pedido(ee, est, b)
    pedidos += 1
    pasta = SAIDA / "pilhas"
    pasta.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pasta / f"{b['bloco']}.npz", vv=vv, vh=vh,
                        datas=np.array([d["data"] for d in sol]),
                        **{k: s[k].astype(np.float32) for k in ("arvore", "construido", "jrc_occ", "jrc_max", "angulo")})
    f = pasta / f"{b['bloco']}.npz"
    return {"bloco": b["bloco"], "pedidos": pedidos, "datas": len(sol), "bytes": f.stat().st_size,
            "sha256": sha256_arquivo(f), "quando_utc": agora_utc()}


def cmd_baixar(a) -> None:
    _exigir_protocolo()
    proj = _ler("contagem_projetada.json")
    if not proj["dentro_do_teto"]:
        raise SystemExit("contagem projetada estoura o teto; nada baixado")
    B = _ler("blocos.json")
    ee = _ee()
    reg = SAIDA / "downloads.jsonl"
    feitos = set()
    pedidos = 0
    if reg.exists():
        for ln in reg.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            if r.get("bytes"):
                feitos.add(r["bloco"])
            pedidos += r.get("pedidos", 0)
    fila = []
    for b in B["blocos"]:
        cat = json.loads((SAIDA / "catalogo" / f"{b['bloco']}.json").read_text(encoding="utf-8"))
        if b["bloco"] in feitos or not cat["escolha"] or not any(d["pre"] == "solicitar" for d in cat["datas"]):
            continue
        fila.append((b, cat))
    if a.limite:
        fila = fila[:a.limite]
    print(len(fila), "blocos na fila;", pedidos, "pedidos ja feitos")

    def um(bc):
        b, cat = bc
        t = time.time()
        try:
            r = baixar_bloco(ee, b, cat)
        except Exception as e:  # noqa: BLE001 - erro vira linha no registro, nao some
            r = {"bloco": b["bloco"], "erro": str(e)[:500], "pedidos": 0, "quando_utc": agora_utc()}
        r["segundos"] = round(time.time() - t, 1)
        return r

    erros_seguidos = 0
    with ThreadPoolExecutor(max_workers=a.paralelo) as ex, reg.open("a", encoding="utf-8") as out:
        futuros = [ex.submit(um, bc) for bc in fila]
        for fu in as_completed(futuros):
            r = fu.result()
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            pedidos += r.get("pedidos", 0)
            print(r["bloco"], r.get("datas"), r.get("segundos"), r.get("erro", "ok")[:120], flush=True)
            erros_seguidos = erros_seguidos + 1 if "erro" in r else 0
            disco = sum(f.stat().st_size for f in (SAIDA / "pilhas").glob("*.npz")) / 1e9
            if pedidos > TETO_PEDIDOS or disco > TETO_GB or erros_seguidos >= 5:
                print("TETO OU ERROS: parando", pedidos, round(disco, 3), erros_seguidos)
                for f in futuros:
                    f.cancel()
                break


# ------------------------------------------------------------------ derivar

def observabilidade(arvore, construido, declive, jrc_occ) -> dict:
    valido = (arvore >= 0) & np.isfinite(declive)
    perm = valido & (jrc_occ >= JRC_PERMANENTE)
    obs = valido & ~perm & (arvore <= ARVORE_MAX) & (construido <= CONSTRUIDO_MAX) & (declive <= DECLIVE_MAX_GRAUS)
    return {"observavel": obs, "permanente": perm, "valido": valido}


def calibrar_limiar(por_componente: dict) -> dict:
    """por_componente[c] = (vv_agua_db, vv_terra_db) em pixels x datas fora de evento.

    Acuracia balanceada media entre componentes (peso igual por componente).
    """
    curva = {}
    usados = {c: v for c, v in por_componente.items()
              if len(v[0]) >= MIN_PIXELS_CLASSE_CALIBRACAO and len(v[1]) >= MIN_PIXELS_CLASSE_CALIBRACAO}
    for t in LIMIARES_VV:
        ba = [0.5 * ((a <= t).mean() + (b > t).mean()) for a, b in usados.values()]
        curva[t] = float(np.mean(ba)) if ba else float("nan")
    validos = {t: v for t, v in curva.items() if np.isfinite(v)}
    if not validos:
        return {"limiar_db": None, "curva": curva, "componentes": 0}
    melhor = max(validos, key=lambda t: (validos[t], -abs(t + 18)))
    return {"limiar_db": melhor, "ba": validos[melhor], "curva": curva, "componentes": len(usados),
            "componentes_usados": sorted(usados)}


def _carregar_bloco(b: dict) -> dict | None:
    import rasterio
    from rasterio.windows import Window

    f = SAIDA / "pilhas" / f"{b['bloco']}.npz"
    if not f.exists():
        return None
    z = dict(np.load(f))
    with rasterio.open(TERRENO / b["aoi"] / "slope_deg_wbt.tif") as s:
        dec = s.read(1, window=Window(b["coluna0"], b["linha0"], b["largura"], b["altura"])).astype(np.float32)
        if s.nodata is not None:
            dec[dec == s.nodata] = np.nan
    z["declive"] = dec
    return z


def admissao_do_bloco(b: dict, cat: dict, z: dict | None) -> list[dict]:
    linhas = []
    vistos = set()
    solicitadas = {}
    if z is not None:
        for i, d in enumerate(z["datas"].tolist()):
            solicitadas[d] = i
    for d in cat["datas"]:
        base = {"bloco": b["bloco"], "aoi": b["aoi"], "componente": b["componente"], "evento_id": b["evento_id"],
                "data": d["data"], "quando_utc": d["quando_utc"], "orbita": d["orbita"], "passagem": d["passagem"],
                "janela": d["janela"], "cobertura_pegada": d["cobertura_pegada"], "n_cenas": len(d["cenas"]),
                "solicitada": d["pre"] == "solicitar", "baixada": False, "admitida": False,
                "motivo": d["pre"] if d["pre"] != "solicitar" else None,
                "valido_frac": np.nan, "vv_mediana_db": np.nan}
        if d["pre"] == "solicitar":
            if z is None or d["data"] not in solicitadas:
                base["motivo"] = "nao_baixada"
            else:
                i = solicitadas[d["data"]]
                vv, vh = z["vv"][i], z["vh"][i]
                ok = (vv != NODATA) & (vh != NODATA)
                base["baixada"] = True
                base["valido_frac"] = float(ok.mean())
                base["vv_mediana_db"] = float(np.median(vv[ok]) / 100) if ok.any() else np.nan
                motivo = admitir_data(base["valido_frac"], base["vv_mediana_db"], date.fromisoformat(d["data"]),
                                      b["lat"], d["data"] in vistos)
                base["motivo"] = motivo
                base["admitida"] = motivo is None
                vistos.add(d["data"])
        linhas.append(base)
    return linhas


def cmd_derivar(_a) -> None:
    p = _exigir_protocolo()
    B = _ler("blocos.json")
    linhas, dados = [], {}
    for b in B["blocos"]:
        cat = json.loads((SAIDA / "catalogo" / f"{b['bloco']}.json").read_text(encoding="utf-8"))
        z = _carregar_bloco(b)
        linhas += admissao_do_bloco(b, cat, z)
        if z is not None:
            dados[b["bloco"]] = b      # um bloco por vez na memoria: recarrega abaixo
    man = pd.DataFrame(linhas)

    # 1) calibracao do limiar contra o JRC, so datas FORA admitidas
    amostra: dict = {}
    rng = np.random.default_rng(20261017)
    for nome, b in dados.items():
        m = man[(man.bloco == nome) & man.admitida & (man.janela == "fora")]
        if m.empty:
            continue
        z = _carregar_bloco(b)
        idx = [int(np.where(z["datas"] == d)[0][0]) for d in m.data]
        ob = observabilidade(z["arvore"], z["construido"], z["declive"], z["jrc_occ"])
        agua = (z["jrc_occ"] >= JRC_AGUA_CALIBRACAO) & (z["arvore"] <= ARVORE_MAX) & \
               (z["construido"] <= CONSTRUIDO_MAX) & (z["declive"] <= DECLIVE_MAX_GRAUS)
        terra = ob["observavel"] & (z["jrc_occ"] == 0) & (z["jrc_max"] == 0)
        vv = z["vv"][idx]
        va = vv[:, agua]
        vt = vv[:, terra]
        va = va[va != NODATA] / 100.0
        vt = vt[vt != NODATA] / 100.0
        # Teto por bloco para nenhum bloco grande dominar o componente.
        va = rng.choice(va, min(len(va), 20000), replace=False) if len(va) else va
        vt = rng.choice(vt, min(len(vt), 20000), replace=False) if len(vt) else vt
        a0, t0 = amostra.get(b["componente"], (np.empty(0), np.empty(0)))
        amostra[b["componente"]] = (np.r_[a0, va], np.r_[t0, vt])
    cal = calibrar_limiar(amostra)
    _gravar("calibracao_agua.json", {
        "tipo": "calibracao_limiar_vv_contra_jrc", "quando_utc": agora_utc(), "protocolo": p["sha256"],
        "nao_e": "nao e validacao de deteccao de cheia: separa agua permanente de terra seca JRC fora de evento",
        **cal,
        "n_por_componente": {c: [int(len(v[0])), int(len(v[1]))] for c, v in amostra.items()},
    })
    thr = cal["limiar_db"]
    if thr is None:
        raise SystemExit("sem componente com as duas classes: limiar nao calibravel")

    # 2) frequencia (a) e quadro (b) por bloco
    pf = SAIDA / "frequencia"
    pf.mkdir(exist_ok=True)
    resumo = []
    for nome, b in dados.items():
        m = man[(man.bloco == nome) & man.admitida]
        z = _carregar_bloco(b)
        idx = np.array([int(np.where(z["datas"] == d)[0][0]) for d in m.data], int)
        jan = m.janela.values
        ob = observabilidade(z["arvore"], z["construido"], z["declive"], z["jrc_occ"])
        vv = z["vv"][idx]
        val = vv != NODATA
        det = np.where(val, (vv <= thr * 100).astype(np.uint8), np.uint8(255))
        fora = jan == "fora"
        n_fora = val[fora].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            freq = np.where(n_fora > 0, (det[fora] == 1).sum(0) / n_fora, np.nan).astype(np.float32)
            freq_m2 = np.where(n_fora > 0, ((vv[fora] <= (thr - 2) * 100) & val[fora]).sum(0) / n_fora, np.nan)
            freq_p2 = np.where(n_fora > 0, ((vv[fora] <= (thr + 2) * 100) & val[fora]).sum(0) / n_fora, np.nan)
        classe = np.zeros(freq.shape, np.uint8)          # 0 = nao observavel
        classe[ob["observavel"]] = 1                       # 1 = observavel
        classe[ob["permanente"]] = 2                       # 2 = permanente JRC
        freq_a = np.where(classe == 1, freq, np.nan).astype(np.float32)
        np.savez_compressed(pf / f"{nome}.npz", freq_fora=freq_a, n_fora=n_fora.astype(np.int16),
                            freq_fora_menos2db=np.where(classe == 1, freq_m2, np.nan).astype(np.float32),
                            freq_fora_mais2db=np.where(classe == 1, freq_p2, np.nan).astype(np.float32),
                            classe=classe, deteccao=det, datas=m.data.values.astype(str),
                            janela=jan.astype(str), quando_utc=m.quando_utc.values.astype(str),
                            limiar_db=np.float32(thr), x0=b["x0"], y0=b["y0"], crs=b["crs"])
        nobs = int((classe == 1).sum())
        persist = int(((freq_a > 0.9) & (n_fora >= MIN_DATAS_FORA)).sum())
        resumo.append({"bloco": nome, "componente": b["componente"], "evento_id": b["evento_id"], "aoi": b["aoi"],
                       "datas_admitidas": int(len(idx)), "datas_fora": int(fora.sum()),
                       "datas_evento": int((jan == "evento").sum()), "datas_transicao": int((jan == "transicao").sum()),
                       "pixels": int(classe.size), "observaveis": nobs, "permanentes": int((classe == 2).sum()),
                       "nao_observaveis": int((classe == 0).sum()),
                       "freq_media_observavel": float(np.nanmean(freq_a)) if nobs else np.nan,
                       "frac_pixels_alguma_agua": float(np.nanmean(freq_a > 0)) if nobs else np.nan,
                       "baixo_retro_persistente": persist,
                       "n_pontos": b["n_pontos"], "n_pos": b["n_pos"], "n_neg": b["n_neg"]})
    r = pd.DataFrame(resumo)
    r["em_a"] = (r.datas_fora >= MIN_DATAS_FORA) & (r.observaveis >= MIN_PIXELS_OBSERVAVEIS)
    r["em_b"] = r.em_a & (r.datas_evento >= 1)
    man.to_parquet(SAIDA / "manifesto_admissao.parquet", index=False)
    man.to_csv(SAIDA / "manifesto_admissao.csv", index=False)
    r.to_csv(SAIDA / "blocos_derivados.csv", index=False)
    print("limiar", thr, "BA", cal.get("ba"), "blocos", len(r), "em_a", int(r.em_a.sum()), "em_b", int(r.em_b.sum()))


# ------------------------------------------------------------------ emenda E1

E1_DELTA_DB = 3.0
E1_QUANTIL_REFERENCIA = 75


def emenda_e1() -> dict:
    return {
        "id": "E1",
        "tipo": "emenda_regra_de_deteccao",
        "nao_e": "nao substitui o resultado v1 registrado; os dois ficam lado a lado",
        "motivo": ("na derivacao v1 a frequencia media fora de evento passou de 0,4-0,6 em componentes "
                   "aridos/saelianos (EMSR753, EMSR750, EMSR772): solo seco e areia tem VV abaixo de "
                   "-14,5 dB sem agua. Visto no resumo de frequencia por bloco, sem rotulo CEMS."),
        "regra": (f"agua na data t = VV_t <= limiar v1 E VV_t <= P{E1_QUANTIL_REFERENCIA}(VV do pixel nas datas "
                  f"fora admitidas) - {E1_DELTA_DB} dB; pixel com < {MIN_DATAS_FORA} datas fora fica sem referencia"),
        "consequencia_declarada": (f"pixel alagado em mais de {100 - E1_QUANTIL_REFERENCIA}% das datas fora perde "
                                   "deteccao (a referencia dele ja e agua); agua permanente JRC continua separada"),
        "checagem": ("so contra o JRC: frequencia media em terra JRC occurrence==0 e max_extent==0 (deve cair) e em "
                     "agua sazonal JRC 10-49 (deve ficar acima da terra), por componente; nenhum rotulo CEMS"),
        "registrada_antes_do_resultado": True,
    }


def deteccao_e1(vv: np.ndarray, fora: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """vv int16 dB*100 [t,h,w]; devolve (deteccao uint8 com 255 = sem leitura, referencia P75 em dB)."""
    val = vv != NODATA
    vf = np.where(val[fora], vv[fora] / 100.0, np.nan)
    n = val[fora].sum(0)
    with np.errstate(all="ignore"):
        ref = (np.nanpercentile(vf, E1_QUANTIL_REFERENCIA, axis=0) if fora.any()
               else np.full(vv.shape[1:], np.nan))
    ref = np.where(n >= MIN_DATAS_FORA, ref, np.nan)
    v = vv / 100.0
    agua = (v <= thr) & (v <= ref[None] - E1_DELTA_DB)
    det = np.where(val & np.isfinite(ref)[None], agua.astype(np.uint8), np.uint8(255))
    return det, ref.astype(np.float32)


def _media(a: np.ndarray, msk: np.ndarray) -> tuple[float, int]:
    x = a[msk & np.isfinite(a)]
    return (float(x.mean()), int(x.size)) if x.size else (np.nan, 0)


def cmd_emenda1(_a) -> None:
    _exigir_protocolo()
    e = emenda_e1()
    if not (SAIDA / "emendas.json").exists():
        _gravar("emendas.json", {"tipo": "emendas_ao_protocolo", "nao_e": "nao altera o protocolo congelado",
                                 "emendas": [{**e, "quando_utc": agora_utc(), "sha256": sha256_json(e)}]})
    elif _ler("emendas.json")["emendas"][0]["sha256"] != sha256_json(e):
        raise SystemExit("emenda E1 divergiu do registro")
    B = _ler("blocos.json")
    man = pd.read_parquet(SAIDA / "manifesto_admissao.parquet")
    thr = _ler("calibracao_agua.json")["limiar_db"]
    pf = SAIDA / "frequencia_e1"
    pf.mkdir(exist_ok=True)
    linhas = []
    for b in B["blocos"]:
        z = _carregar_bloco(b)
        m = man[(man.bloco == b["bloco"]) & man.admitida]
        if z is None or m.empty:
            continue
        idx = np.array([int(np.where(z["datas"] == d)[0][0]) for d in m.data], int)
        jan = m.janela.values
        fora = jan == "fora"
        vv = z["vv"][idx]
        det, ref = deteccao_e1(vv, fora, thr)
        ob = observabilidade(z["arvore"], z["construido"], z["declive"], z["jrc_occ"])
        val = vv != NODATA
        nf = (det[fora] != 255).sum(0)
        nv = val[fora].sum(0)
        with np.errstate(all="ignore"):
            f_e1 = np.where(nf > 0, (det[fora] == 1).sum(0) / nf, np.nan)
            f_v1 = np.where(nv > 0, ((vv[fora] <= thr * 100) & val[fora]).sum(0) / nv, np.nan)
        obs = ob["observavel"]
        f_e1a = np.where(obs, f_e1, np.nan).astype(np.float32)
        np.savez_compressed(pf / f"{b['bloco']}.npz", freq_fora=f_e1a, referencia_p75_db=ref,
                            deteccao=det, datas=m.data.values.astype(str), janela=jan.astype(str),
                            limiar_db=np.float32(thr), delta_db=np.float32(E1_DELTA_DB))
        terra = obs & (z["jrc_occ"] == 0) & (z["jrc_max"] == 0)
        sazonal = obs & (z["jrc_occ"] >= 10) & (z["jrc_occ"] < JRC_PERMANENTE)
        (t1, nt), (t2, _) = _media(f_v1, terra), _media(f_e1, terra)
        (s1, ns), (s2, _) = _media(f_v1, sazonal), _media(f_e1, sazonal)
        linhas.append({"bloco": b["bloco"], "componente": b["componente"], "n_terra": nt, "n_sazonal": ns,
                       "terra_v1": t1, "terra_e1": t2, "sazonal_v1": s1, "sazonal_e1": s2,
                       "obs_v1": _media(f_v1, obs)[0], "obs_e1": _media(f_e1a, obs)[0]})
    d = pd.DataFrame(linhas)
    d.to_csv(SAIDA / "emenda_e1_por_bloco.csv", index=False)

    def por_comp(col: str, peso: str) -> pd.Series:
        h = d[(d[peso] > 0) & d[col].notna()]
        return h.groupby("componente").apply(lambda g: float(np.average(g[col], weights=g[peso])),
                                             include_groups=False)
    c = pd.DataFrame({"terra_v1": por_comp("terra_v1", "n_terra"), "terra_e1": por_comp("terra_e1", "n_terra"),
                      "sazonal_v1": por_comp("sazonal_v1", "n_sazonal"),
                      "sazonal_e1": por_comp("sazonal_e1", "n_sazonal")})
    ambos = c.dropna()
    out = {"tipo": "checagem_emenda_e1_contra_jrc", "quando_utc": agora_utc(),
           "nao_e": "nao e validacao contra cheia observada; JRC e referencia de agua, nao de inundacao",
           "limiar_db": thr, "delta_db": E1_DELTA_DB,
           "terra_jrc0_media_componentes": {"v1": float(c.terra_v1.mean()), "e1": float(c.terra_e1.mean())},
           "terra_jrc0_max_componente": {"v1": float(c.terra_v1.max()), "e1": float(c.terra_e1.max())},
           "componentes_com_sazonal": int(len(ambos)),
           "sazonal_menos_terra": {"v1": float((ambos.sazonal_v1 - ambos.terra_v1).mean()),
                                   "e1": float((ambos.sazonal_e1 - ambos.terra_e1).mean())},
           "componentes_sazonal_acima_da_terra": {"v1": int((ambos.sazonal_v1 > ambos.terra_v1).sum()),
                                                  "e1": int((ambos.sazonal_e1 > ambos.terra_e1).sum())},
           "por_componente": c.reset_index().rename(columns={"index": "componente"}).to_dict("records")}
    _gravar("emenda_e1_checagem.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "por_componente"}, ensure_ascii=False))
    print(c.round(3).to_string())


QUALIDADE_TERRA_MAX = 0.10
QUALIDADE_CONTRASTE_MIN = 0.05


def qualidade_da_referencia() -> dict | None:
    """Selo POST HOC por componente, so contra o JRC: a frequencia serve como referencia?

    Utilizavel = deteccao media em terra JRC==0 <= 0,10 e agua sazonal JRC 10-49
    pelo menos 0,05 acima dela. Definido depois de ver a checagem da E1; e
    filtro de qualidade declarado, nunca resultado.
    """
    f = SAIDA / "emenda_e1_checagem.json"
    if not f.exists():
        return None
    c = pd.DataFrame(json.loads(f.read_text(encoding="utf-8"))["por_componente"])
    out = {"regra": f"terra_jrc0 <= {QUALIDADE_TERRA_MAX} e sazonal - terra >= {QUALIDADE_CONTRASTE_MIN}",
           "post_hoc": True}
    for regra in ("v1", "e1"):
        ok = (c[f"terra_{regra}"] <= QUALIDADE_TERRA_MAX) & \
             (c[f"sazonal_{regra}"] - c[f"terra_{regra}"] >= QUALIDADE_CONTRASTE_MIN)
        out[regra] = sorted(c[ok].componente)
    out["alguma"] = sorted(set(out["v1"]) | set(out["e1"]))
    out["nenhuma"] = sorted(set(c.componente) - set(out["alguma"]))
    out["contagem"] = {k: len(out[k]) for k in ("v1", "e1", "alguma", "nenhuma")}
    return out


# ------------------------------------------------------------------ relatorio

def cmd_relatorio(_a) -> None:
    p = _exigir_protocolo()
    B = _ler("blocos.json")
    man = pd.read_parquet(SAIDA / "manifesto_admissao.parquet")
    r = pd.read_csv(SAIDA / "blocos_derivados.csv")
    cal = _ler("calibracao_agua.json")
    proj = _ler("contagem_projetada.json")
    downloads = [json.loads(x) for x in (SAIDA / "downloads.jsonl").read_text(encoding="utf-8").splitlines()]
    todos_comp = sorted({e["unidade"] for e in B["eventos"]})
    todos_ev = sorted({e["evento_id"] for e in B["eventos"]})
    comp_bloco = sorted({b["componente"] for b in B["blocos"]})
    a_comp = sorted(r[r.em_a].componente.unique())
    b_comp = sorted(r[r.em_b].componente.unique())
    # Evento coberto em (b): ha data admitida dentro da janela DO PROPRIO evento
    # em bloco da AOI dele. Componentes com varios eventos so contam os seus.
    jan_ev = {e["evento_id"]: (date.fromisoformat(e["j0"]), date.fromisoformat(e["j1"])) for e in B["eventos"]}
    man["janela_proprio_evento"] = [classificar_data(date.fromisoformat(d), [jan_ev[e]]) == "evento"
                                    for d, e in zip(man.data, man.evento_id)]
    proprio = man[man.admitida & man.janela_proprio_evento].groupby("bloco").size()
    r["datas_proprio_evento"] = r.bloco.map(proprio).fillna(0).astype(int)
    ev_b = sorted(r[r.em_a & (r.datas_proprio_evento >= 1)].evento_id.unique())
    aoi_data = (man.groupby(["aoi", "data", "janela"])
                .agg(blocos=("bloco", "nunique"), solicitada=("solicitada", "any"),
                     baixada=("baixada", "any"), admitida=("admitida", "any"),
                     motivos=("motivo", lambda s: ";".join(sorted({str(x) for x in s if x is not None and x == x}))))
                .reset_index())
    aoi_data.to_csv(SAIDA / "manifesto_aoi_x_data.csv", index=False)
    motivos = man[~man.admitida].motivo.value_counts().to_dict()
    sem_bloco = sorted(set(todos_comp) - set(comp_bloco))
    recusados_comp = sorted(set(comp_bloco) - set(a_comp))
    por_comp = []
    for c in todos_comp:
        rc = r[r.componente == c]
        mc = man[man.componente == c]
        por_comp.append({
            "componente": c, "blocos": int(len(rc)), "blocos_em_a": int(rc.em_a.sum()), "blocos_em_b": int(rc.em_b.sum()),
            "datas_catalogadas": int(mc[["bloco", "data"]].drop_duplicates().shape[0]),
            "solicitadas": int(mc.solicitada.sum()), "baixadas": int(mc.baixada.sum()), "admitidas": int(mc.admitida.sum()),
            "admitidas_fora": int((mc.admitida & (mc.janela == "fora")).sum()),
            "admitidas_evento": int((mc.admitida & (mc.janela == "evento")).sum()),
            "motivo_principal_de_recusa": (mc[~mc.admitida].motivo.value_counts().index[0]
                                           if (~mc.admitida).any() else None),
            "freq_media_observavel": float(np.average(rc.freq_media_observavel.fillna(0), weights=rc.observaveis))
            if rc.observaveis.sum() else None,
        })
    out = {
        "tipo": "relatorio_aquisicao_s1_multidata", "quando_utc": agora_utc(), "protocolo": p["sha256"],
        "nao_e": ("nao e rotulo, nao e suscetibilidade, nao e teste de P1 nem desenho causal: e o insumo "
                  "admitido para eles; reservas nao abertas, nada promovido, servico intocado"),
        "projetado_antes_do_download": {k: proj[k] for k in ("blocos", "datas_a_solicitar", "pedidos_previstos",
                                                             "gb_previstos_brutos", "componentes_que_podem_a",
                                                             "componentes_que_podem_b", "eventos_que_podem_b")},
        "downloads": {"blocos_baixados": sum(1 for d in downloads if d.get("bytes")),
                      "blocos_com_erro": sorted({d["bloco"] for d in downloads if "erro" in d} -
                                                {d["bloco"] for d in downloads if d.get("bytes")}),
                      "pedidos": int(sum(d.get("pedidos", 0) for d in downloads)),
                      "gb_em_disco": round(sum(d.get("bytes", 0) for d in downloads) / 1e9, 3),
                      "segundos": round(sum(d.get("segundos", 0) for d in downloads), 1)},
        "datas": {"catalogadas_bloco_x_data": int(len(man)), "solicitadas": int(man.solicitada.sum()),
                  "baixadas": int(man.baixada.sum()), "admitidas": int(man.admitida.sum()),
                  "admitidas_por_janela": man[man.admitida].janela.value_counts().to_dict(),
                  "recusas_por_motivo": motivos,
                  "aoi_x_data": {"linhas": int(len(aoi_data)), "admitidas": int(aoi_data.admitida.sum())}},
        "calibracao": {k: cal.get(k) for k in ("limiar_db", "ba", "componentes")},
        "cobertura": {
            "componentes_na_base": len(todos_comp), "eventos_na_base": len(todos_ev),
            "componentes_com_bloco": len(comp_bloco),
            "componentes_cobertos_a": len(a_comp), "componentes_cobertos_b": len(b_comp),
            "eventos_cobertos_b": len(ev_b),
            "eventos_com_bloco_em_a": int(r[r.em_a].evento_id.nunique()),
            "blocos_em_a": int(r.em_a.sum()), "blocos_em_b": int(r.em_b.sum()),
            "pixels_observaveis_em_a": int(r[r.em_a].observaveis.sum()),
            "pontos_fisicos_em_blocos_a": int(r[r.em_a].n_pontos.sum()),
            "lista_a": a_comp, "lista_b": b_comp, "eventos_b": ev_b,
            "sem_bloco": sem_bloco, "com_bloco_mas_fora_de_a": recusados_comp,
        },
        "por_componente": por_comp,
        "qualidade_da_referencia": qualidade_da_referencia(),
        "arquivos": {f.name: sha256_arquivo(f) for f in sorted(SAIDA.glob("*.*")) if f.suffix in (".json", ".csv", ".parquet")
                     and f.name != "relatorio.json"},
    }
    _gravar("relatorio.json", out)
    print(json.dumps({k: out[k] for k in ("downloads", "datas", "calibracao")}, ensure_ascii=False, default=str))
    print(json.dumps({k: v for k, v in out["cobertura"].items() if not k.startswith("lista")}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="fase", required=True)
    for nome in ("protocolo", "blocos", "catalogo", "derivar", "emenda1", "relatorio"):
        sub.add_parser(nome)
    pb = sub.add_parser("baixar")
    pb.add_argument("--paralelo", type=int, default=3)
    pb.add_argument("--limite", type=int, default=0)
    a = ap.parse_args()
    {"protocolo": cmd_protocolo, "blocos": cmd_blocos, "catalogo": cmd_catalogo, "baixar": cmd_baixar,
     "derivar": cmd_derivar, "emenda1": cmd_emenda1, "relatorio": cmd_relatorio}[a.fase](a)


if __name__ == "__main__":
    main()
