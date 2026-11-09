"""Resolvedor comum por lugar e tempo: o que o acervo LOCAL tem para um polígono (e, se dado, um evento/intervalo).

POR QUE EXISTE. A resposta integrada consultava dois catálogos fixos (inventário de chips SAR herdado e terreno da
rodada 3) e concluía "não há SAR nem terreno nesta AOI" quando o arquivo existia em outro lugar do disco
(auditoria `local_runs/correspondencia-local/20261019_v2`: 67 SARs próprios por AOI em `ter-01`). Este módulo
procura em TODAS as fontes locais declaradas abaixo e devolve candidatos com evidência, não só "achou/não achou".

FONTES PESQUISADAS (a cobertura da busca vai na resposta):
  terreno            índice `acervo-espacial/unidades.parquet` (ter-01) + diretórios de ter-01 fora do índice (cabeçalho
                     lido na hora) + terreno da rodada 3 (unidades do Q de pesquisa)
  sar_proprio        `ter-01/*/s1_grd_manifesto.json` (cena, UTC, órbita, polarizações, unidade, grade)
  sar_chip           inventário de pares do pacote de pesquisa v2 (antes/depois/JRC; Sen1Floods11)
  referencia_cems    `cems-06-por-aoi-e-data` (produto por AOI e data: AOI, área observada, positivo) + metadados da
                     ativação (`cems-04-fila/ativacoes_meta.json`) + pegadas (`acervo-espacial/pegadas.parquet`)
  historico_f        catálogo admitido do pacote v2 (F; identidade e coordenada exatas)
  contexto_documental pontos da tabela única (ds-05) com data e fonte dentro da AOI — registro de ocorrência/negativo,
                     NÃO frequência F
  social             pacote social por polígono (social07)
  regiao_servida     caixas do contrato (svc02) + malha IBGE para macrorregiões
  reservas           reservas-observacionais e papéis do grafo comum: o conteúdo NÃO é lido

CONFIRMAÇÃO. Caixa sobreposta só gera candidato. Para raster, o polígono é levado ao CRS do arquivo e contam-se os
pixels VÁLIDOS dentro dele (finito, diferente de NoData); a fração válida é sobre a área da AOI (EPSG:6933). Para
vetor, interseção real de polígonos.

SELEÇÃO. Determinística e declarada, nunca por rótulo ou desempenho: maior fração válida da AOI; empate → (com evento)
menor distância em dias à data do evento → nome. Todos os candidatos voltam, com o motivo de terem ou não sido escolhidos.

ESTADOS por candidato e uso: presente_ligado | presente_nao_ligado | presente_incompativel | reservado |
nao_localizado. Nenhum estado vira "ausência de água" ou "impossível".
"""
from __future__ import annotations

import json
import math
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
R = REPO / "local_runs"
TER01 = R / "ter-01-cadeia-harmonizada"
VERSAO = "resolvedor_acervo_v2_selecao_exata"
MAX_PIXELS_NATIVOS = 4_000_000
ESTADOS = ("presente_ligado", "presente_nao_ligado", "presente_incompativel", "reservado", "nao_localizado")


# ------------------------------------------------------------------ geometria e cobertura real

def poligono(aoi) -> "object":
    """bbox {lat_min,lon_min,lat_max,lon_max} ou GeoJSON Polygon/MultiPolygon (EPSG:4326) → shapely."""
    from shapely.geometry import box, shape
    if isinstance(aoi, dict) and "type" in aoi:
        g = shape(aoi)
    else:
        g = box(float(aoi["lon_min"]), float(aoi["lat_min"]), float(aoi["lon_max"]), float(aoi["lat_max"]))
    if g.is_empty or not g.is_valid:
        raise ValueError("AOI vazia ou invalida")
    return g


def _area_km2(g4326) -> float:
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "social"))
    from social_projecao import reprojetar
    return reprojetar([g4326], 4326, 6933)[0].area / 1e6


def cobertura_raster(arquivo: Path, g4326, banda: int = 1) -> dict:
    """Fração da AOI coberta por pixels válidos deste raster (lida dos pixels, não do cabeçalho)."""
    import rasterio
    from pyproj import Transformer
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds
    from shapely.geometry import box
    from shapely.ops import transform as tr_geom
    arquivo = Path(arquivo)
    if not arquivo.exists():
        return {"arquivo": str(arquivo), "existe": False}
    with rasterio.open(arquivo) as r:
        t = Transformer.from_crs(4326, r.crs, always_xy=True)
        g = tr_geom(t.transform, g4326)
        inter = g.intersection(box(*r.bounds))
        base = {"arquivo": _rel(arquivo), "existe": True, "crs": str(r.crs), "res": [abs(r.res[0]), abs(r.res[1])],
                "dimensoes": [r.height, r.width], "nodata": "NaN" if r.nodata is not None and r.nodata != r.nodata else r.nodata, "bandas": r.count, "unidade_crs_metrica": r.crs.is_projected}
        if inter.is_empty or inter.area == 0:
            return {**base, "fracao_coberta_pela_grade": 0.0, "fracao_valida_da_aoi": 0.0, "pixels_validos": 0}
        w = from_bounds(*inter.bounds, transform=r.transform).round_offsets().round_lengths()
        w = w.intersection(rasterio.windows.Window(0, 0, r.width, r.height))
        # Janela enorme (AOI do tamanho de um pais sobre tile GFD) e lida DECIMADA (vizinho mais proximo) e o fator
        # vai declarado: a fracao valida e estimativa nessa grade, nao contagem nativa.
        fator = max(1, int(math.ceil(math.sqrt(w.width * w.height / MAX_PIXELS_NATIVOS))))
        forma = (max(1, int(w.height // fator)), max(1, int(w.width // fator)))
        a = r.read(banda, window=w, out_shape=forma, masked=True).astype('float64').filled(np.nan)
        tw = r.window_transform(w) * rasterio.Affine.scale(w.width / forma[1], w.height / forma[0])
        dentro = geometry_mask([inter], out_shape=a.shape, transform=tw, invert=True)
        valido = dentro & np.isfinite(a)
        if r.nodata is not None and not (isinstance(r.nodata, float) and math.isnan(r.nodata)):
            valido &= a != r.nodata
    t_inv = Transformer.from_crs(r.crs, 4326, always_xy=True)
    frac_grade = _area_km2(tr_geom(t_inv.transform, inter)) / max(_area_km2(g4326), 1e-12)
    frac_valida = frac_grade * (valido.sum() / max(dentro.sum(), 1))
    return {**base, "fracao_coberta_pela_grade": round(float(min(frac_grade, 1.0)), 6),
            "pixels_no_poligono": int(dentro.sum()), "pixels_validos": int(valido.sum()), "fator_de_decimacao": fator,
            "fracao_valida_da_aoi": round(float(min(frac_valida, 1.0)), 6)}


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def _ler(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _dias(a: str | None, b: str | None) -> int | None:
    try:
        return abs((date.fromisoformat(str(a)[:10]) - date.fromisoformat(str(b)[:10])).days)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ catálogos (cacheados por processo)

@lru_cache(maxsize=1)
def _unidades():
    import pandas as pd
    u = pd.read_parquet(R / "acervo-espacial/unidades.parquet")
    u["bbox_l"] = u.bbox.map(json.loads)
    u["no_indice"] = True
    fora = sorted(set(p.name for p in TER01.iterdir() if p.is_dir()) - set(u.unidade))
    extras = []
    for nome in fora:   # diretórios novos: cabeçalho lido na hora, sem metadados de cadeia do índice
        arq = TER01 / nome / "hand_dinf.tif"
        if not arq.exists():
            continue
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(arq) as r:
            bb = list(transform_bounds(r.crs, 4326, *r.bounds))
        extras.append({"unidade": nome, "tipo": "fora_do_indice", "caminho": _rel(TER01 / nome), "bbox_l": bb, "no_indice": False,
                       "cadeia_no_contrato": None, "familia_mdt": None, "compativel": None, "stream_area_km2": None})
    if extras:
        u = pd.concat([u, pd.DataFrame(extras)], ignore_index=True)
    return u, len(fora)


@lru_cache(maxsize=1)
def _sar_proprio() -> list[dict]:
    out = []
    for m in sorted(TER01.glob("*/s1_grd_manifesto.json")):
        d = _ler(m)
        # Data EXATA = aquisicao UTC das cenas mosaicadas. `data_da_imagem` do manifesto e a data PEDIDA ao catalogo
        # (achado de 19/10/2026: EMSR754 pedida 14/09, cena de 15/09 11:06 UTC; Petropolis pedida 05/04, cenas de 03/04
        # e 09/04). Mosaico de mais de um dia nao tem data unica: fica None e `mosaico_multidata`.
        aquis = datas_aquisicao_utc(d.get("cenas_mosaicadas") or [])
        out.append({"unidade": m.parent.name, "manifesto": _rel(m), "arquivo": _rel(m.parent / "s1_grd.tif"),
                    "data_da_imagem": aquis[0] if len(aquis) == 1 else None, "datas_de_aquisicao_utc": aquis,
                    "mosaico_multidata": len(aquis) > 1, "data_pedida_no_manifesto": d.get("data_da_imagem"),
                    "tolerancia_dias_do_pedido": d.get("tolerancia_dias"),
                    "colecao": d.get("colecao"), "unidade_radiometrica": d.get("unidade"),
                    "bandas": d.get("bandas"), "composicao": d.get("composicao"),
                    "cenas": [{k: c.get(k) for k in ("id", "quando_utc", "passagem", "orbita_relativa", "polarizacoes")}
                              for c in d.get("cenas_mosaicadas") or []],
                    "cobertura_declarada": d.get("cobertura_da_aoi"), "alinhado_a_grade_do_terreno": d.get("alinhado_a_grade_do_terreno"),
                    "grade": d.get("grade_do_terreno")})
    return out


def datas_aquisicao_utc(cenas):
    from datetime import datetime, timezone
    datas=[]
    for c in cenas:
        valor=c.get('quando_utc')
        if not valor:raise ValueError('cena sem instante de aquisicao')
        dt=datetime.fromisoformat(valor.replace('Z','+00:00'))
        if dt.tzinfo is None:raise ValueError('aquisicao sem fuso; UTC nao pode ser presumido')
        datas.append(dt.astimezone(timezone.utc).date().isoformat())
    return sorted(set(datas))


@lru_cache(maxsize=1)
def _chips_sar() -> dict:
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "ia"))
    import ia_integrada as anterior
    import ia_integrada_v2 as v2
    _, meta = v2.carregar()
    return anterior.inventario(meta["pai_percepcao_manifesto"])


@lru_cache(maxsize=1)
def _cems06():
    import pandas as pd
    return pd.read_csv(R / "cems-06-por-aoi-e-data/catalogo.csv")


@lru_cache(maxsize=1)
def _ativacoes() -> dict:
    return _ler(R / "cems-04-fila/ativacoes_meta.json")


@lru_cache(maxsize=1)
def _reservas() -> set:
    res = {p.parent.name for p in (R / "reservas-observacionais").glob("*/*/produto.zip")}
    nos = R / "evolucao-acervo-local/v2/grafo_comum/grafo/nos.parquet"
    if nos.exists():
        import pandas as pd
        n = pd.read_parquet(nos)
        res |= set(n[n.papeis.astype(str).str.contains("reserva")].evento)
    return res


@lru_cache(maxsize=1)
def _rodada3() -> dict:
    import rasterio
    from rasterio.warp import transform_bounds
    raiz = R / "confirmacao-rodada3/v1/confirmacao/terreno"
    out = {}
    for d in sorted(p for p in raiz.iterdir() if (p / "run_manifest.json").is_file()):
        with rasterio.open(d / "hand_dinf.tif") as s:
            out[d.name] = list(transform_bounds(s.crs, 4326, *s.bounds))
    return out


def _bbox_cruza(bb, g) -> bool:
    from shapely.geometry import box
    return box(*bb).intersects(g)


# ------------------------------------------------------------------ modalidades

def _escolher(cands: list[dict], evento_data: str | None) -> dict | None:
    ok = [c for c in cands if c.get("estado") == "presente_ligado" and (c.get("cobertura") or {}).get("fracao_valida_da_aoi", 0) > 0]
    if not ok:
        return None
    return sorted(ok, key=lambda c: (-(c["cobertura"]["fracao_valida_da_aoi"]),
                                     _dias(c.get("data"), evento_data) if evento_data and c.get("data") else 10**6,
                                     c.get("id", "")))[0]


def terreno(g) -> dict:
    u, fora = _unidades()
    sel = u[u.bbox_l.map(lambda bb: _bbox_cruza(bb, g))]
    cands = []
    for _, x in sel.iterrows():
        cob = cobertura_raster(REPO / x.caminho / "hand_dinf.tif", g)
        compat = x.get("compativel")
        estado = ("presente_ligado" if compat is True else "presente_incompativel" if compat is False else "presente_nao_ligado")
        cands.append({"id": x.unidade, "tipo": x.tipo, "estado": estado, "cobertura": cob,
                      "cadeia": {"no_contrato": x.get("cadeia_no_contrato"), "familia_mdt": x.get("familia_mdt"),
                                 "limiar_canal_km2": x.get("stream_area_km2"), "twi": x.get("twi_formula"),
                                 "reamostragem_mdt": x.get("reamostragem_mdt"), "licenca_mdt": x.get("mdt_licenca") if "mdt_licenca" in x else None,
                                 "no_indice": bool(x.no_indice)},
                      "motivo": (None if estado == "presente_ligado" else
                                 "familia de MDT ou cadeia fora do contrato (nao misturada)" if estado == "presente_incompativel" else
                                 "diretorio fora do indice acervo-espacial: cadeia nao conferida")})
    for nome, bb in _rodada3().items():
        if _bbox_cruza(bb, g):
            cob = cobertura_raster(R / "confirmacao-rodada3/v1/confirmacao/terreno" / nome / "hand_dinf.tif", g)
            cands.append({"id": f"rodada3:{nome}", "tipo": "unidade_pesquisa_q", "estado": "presente_ligado", "cobertura": cob,
                          "uso": "Q de pesquisa por coordenada (consulta_territorial_v2)"})
    escolhido = _escolher([c for c in cands if not c["id"].startswith("rodada3:")], None)
    return {"candidatos": cands, "selecionado": escolhido and escolhido["id"], "fora_do_indice_no_disco": fora,
            "regra": "maior fracao valida de HAND na AOI, so cadeia compativel; rodada3 listada a parte (uso: Q de pesquisa)"}


def _pedido_sar_compativel(c,aquisicao_data=None,sar_unidade=None,intervalo=None):
    compativel = not sar_unidade or c.get('id') == sar_unidade
    if aquisicao_data:
        compativel = compativel and not c.get('mosaico_multidata',False) and c.get('data') == aquisicao_data
    if intervalo:
        datas=c.get('datas_de_aquisicao_utc') or ([c['data']] if c.get('data') else [])
        compativel = compativel and bool(datas) and all(intervalo[0]<=d<=intervalo[1] for d in datas)
    return bool(compativel)


def selecionar_sar_exato(candidatos, evento_data=None, aquisicao_data=None, sar_unidade=None, intervalo=None):
    """Filtro pedido ANTES do ranking: nunca troca arquivo/data por maior cobertura."""
    elegiveis = []
    for c in candidatos:
        compativel = _pedido_sar_compativel(c,aquisicao_data,sar_unidade,intervalo)
        c["compativel_com_pedido_exato"] = bool(compativel)
        if compativel:
            elegiveis.append(c)
    return _escolher(elegiveis, evento_data)


def sar_proprio(g, evento_data: str | None, aquisicao_data=None, sar_unidade=None, intervalo=None) -> dict:
    cands = []
    for s in _sar_proprio():
        u, _ = _unidades()
        linha = u[u.unidade == s["unidade"]]
        bb = linha.bbox_l.iloc[0] if len(linha) else None
        if bb is None or not _bbox_cruza(bb, g):
            continue
        reservado = s["unidade"].split("_")[0] in _reservas()
        compativel=_pedido_sar_compativel({'id':s['unidade'],'data':s['data_da_imagem'],**s},aquisicao_data,sar_unidade,intervalo)
        if reservado:
            cob,estado={'estado':'nao_lida_reserva'},'reservado'
        elif not compativel:
            cob,estado={'estado':'nao_lida_fora_do_pedido'},'nao_selecionado_pedido_exato'
        else:
            cob,estado=cobertura_raster(REPO / s["arquivo"],g,banda=1),'presente_ligado'
        cands.append({**s, "id": s["unidade"], "data": s["data_da_imagem"], "cobertura": cob,
                      "estado": estado,
                      "usos": {"dado_bruto_sar": estado,
                               "percepcao_peso_pesquisa_v2": ("presente_incompativel: o peso foi ajustado a 10 m em chips 512x512 com terreno "
                                                              "levado a grade do chip; este SAR esta a 30 m na grade do terreno (arena diferente, "
                                                              "sem avaliacao de compatibilidade)"),
                               "mudanca_antes_depois": "presente_incompativel: uma unica data; sem imagem anterior registrada"},
                      "papel_temporal": _papel_temporal(s, evento_data)})
    sel = selecionar_sar_exato(cands, evento_data, aquisicao_data, sar_unidade, intervalo)
    return {"candidatos": cands, "selecionado": (sel or {}).get("id"),
            "pedido_exato": {"aquisicao_data": aquisicao_data, "sar_unidade": sar_unidade, "intervalo": intervalo, "aplica_a": "sar_proprio"},
            "motivo_sem_selecao": "nenhum candidato valido compativel com arquivo/dia UTC/intervalo pedidos; sem substituicao" if not sel else None,
            "regra": "maior fracao valida de VV na AOI; empate: dias ate o evento; depois nome",
            "fonte_da_busca": "ter-01/*/s1_grd_manifesto.json"}


def _papel_temporal(s: dict, evento_data: str | None) -> dict:
    code = s["unidade"].split("_")[0]
    meta = _ativacoes().get(code)
    ev = (meta or {}).get("eventTime", "")[:10] or evento_data
    c = _cems06()
    aoi = s["unidade"].split("_")[1] if "_" in s["unidade"] and code.startswith("EMSR") else None
    prods = c[(c.ativacao == code) & ((c.aoi == aoi) if aoi else True)] if code.startswith("EMSR") else c.iloc[0:0]
    prods = prods.assign(produto=prods.aoi + "_" + prods.produto)
    datas = s.get("datas_de_aquisicao_utc") or ([s["data_da_imagem"]] if s.get("data_da_imagem") else [])
    dist = {x.produto: min(abs((date.fromisoformat(x.data_pos_evento) - date.fromisoformat(a)).days) for a in datas)
            for x in prods.itertuples()} if datas else {}
    if ev and datas:
        dd = [(date.fromisoformat(a) - date.fromisoformat(ev)).days for a in datas]
        d = dd[0] if len(dd) == 1 else dd
        papel = "pos_evento" if min(dd) >= 0 else "pre_evento" if max(dd) < 0 else "mosaico_atravessa_o_evento"
    elif ev:
        d, papel = None, "sem_data_de_aquisicao_registrada"
    else:
        d, papel = None, "sem_evento_associado_no_acervo"
    return {"data_da_observacao": s.get("data_da_imagem"), "datas_de_aquisicao_utc": datas,
            "mosaico_multidata": len(datas) > 1, "data_pedida_no_manifesto": s.get("data_pedida_no_manifesto"),
            "data_do_evento": ev or None, "dias_apos_evento": d, "papel": papel,
            "evento_fonte": "cems-04-fila/ativacoes_meta.json" if meta else ("pedido" if evento_data else None),
            "produtos_cems_mesma_data": sorted(k for k, v in dist.items() if v == 0),
            "produtos_cems_a_um_dia": sorted(k for k, v in dist.items() if v == 1),
            "nota": ("datas = aquisicao UTC das cenas, nunca a data pedida; mesma data exige o mesmo dia UTC e +-1 dia fica "
                     "em campo proprio; sem evento associado a cena e apenas uma observacao datada")}


def atualizar_datas_sar(mod: dict, evento_data: str | None) -> dict:
    """Refaz os campos de DATA dos candidatos de SAR proprio (sem reabrir raster) a partir do catalogo atual:
    usado para resultados guardados antes da correcao de 19/10 (data pedida -> data de aquisicao)."""
    cat = {x["unidade"]: x for x in _sar_proprio()}
    cands = []
    for c in mod.get("candidatos", []):
        s = cat.get(c["id"])
        if s is None:
            cands.append(c)
            continue
        cands.append({**c, **{k: s[k] for k in ("data_da_imagem", "datas_de_aquisicao_utc", "mosaico_multidata",
                                                "data_pedida_no_manifesto", "tolerancia_dias_do_pedido")},
                      "data": s["data_da_imagem"], "papel_temporal": _papel_temporal(s, evento_data)})
    return {**mod, "candidatos": cands}


def sar_chip(g, evento_data: str | None, intervalo=None) -> dict:
    cands = []
    for nome, c in _chips_sar().items():
        data=c.get('data_pos_catalogo')
        if intervalo and (not data or not intervalo[0] <= data[:10] <= intervalo[1]):
            continue
        bb = c["grade"]["bounds"]
        if not _bbox_cruza(bb, g):
            continue
        cob = cobertura_raster(Path(c["caminhos"]["depois"]), g, banda=1)
        cands.append({"id": nome, "evento": c.get("evento"), "data": c.get("data_pos_catalogo"), "cobertura": cob,
                      "estado": "presente_ligado", "usos": {"percepcao_peso_pesquisa_v2": "presente_ligado (arena do peso)"}})
    return {"candidatos": cands, "selecionado": (_escolher(cands, evento_data) or {}).get("id"),
            "regra": "maior fracao valida da cena posterior na AOI; empate: dias ate o evento; depois nome"}


@lru_cache(maxsize=1)
def _cems06_poligonos() -> list[dict]:
    """Poligono REAL da AOI de cada produto cems-06, lido uma vez e levado a EPSG:4326 (filtro antes do recorte)."""
    import pyarrow.parquet as pq
    from pyproj import Transformer
    from shapely import wkb
    from shapely.ops import transform as tr_geom
    out = []
    for _, x in _cems06().iterrows():
        pasta = R / "cems-06-por-aoi-e-data" / x.ativacao / f"{x.aoi}_{x.produto}"
        if not (pasta / "aoi.parquet").exists():
            continue
        p_epsg = wkb.loads(pq.read_table(pasta / "aoi.parquet").column("geometry")[0].as_py())
        t = Transformer.from_crs(int(x.epsg), 4326, always_xy=True)
        out.append({"linha": x.to_dict(), "pasta": pasta, "epsg": int(x.epsg), "aoi_4326": tr_geom(t.transform, p_epsg)})
    return out


def referencia_cems(g, evento_data: str | None, intervalo=None) -> dict:
    """Produtos CEMS por AOI e data cuja AOI REAL (poligono) cruza a consulta; areas recortadas na consulta."""
    import pyarrow.parquet as pq
    from pyproj import Transformer
    from shapely import wkb
    from shapely.ops import transform as tr_geom
    cands = []
    for c in _cems06_poligonos():
        data=c['linha'].get('data_pos_evento')
        if intervalo and (not isinstance(data,str) or not intervalo[0] <= data[:10] <= intervalo[1]):
            continue
        if not c["aoi_4326"].intersects(g):
            continue
        x, pasta = c["linha"], c["pasta"]
        if x['ativacao'] in _reservas():
            cands.append({'id':f"{x['ativacao']}_{x['aoi']}_{x['produto']}",
                          'ativacao':x['ativacao'],'aoi':x['aoi'],'produto':x['produto'],
                          'data':x.get('data_pos_evento'),'estado':'reservado','conteudo':'nao_lido'})
            continue
        gq = tr_geom(Transformer.from_crs(4326, c["epsg"], always_xy=True).transform, g)

        def _area(nome):
            f = pasta / f"{nome}.parquet"
            if not f.exists():
                return 0.0
            return wkb.loads(pq.read_table(f).column("geometry")[0].as_py()).intersection(gq).area / 1e6
        reservado = x["ativacao"] in _reservas()
        cands.append({"id": f"{x['ativacao']}_{x['aoi']}_{x['produto']}", "ativacao": x["ativacao"], "aoi": x["aoi"], "produto": x["produto"],
                      "tipo_produto": "delineacao" if "DEL" in x["produto"] else "graduacao" if "GRA" in x["produto"] else "outro",
                      "data": x["data_pos_evento"], "sensor": x["sensor"], "resolucao_sensor": x["resolucao_sensor"],
                      "km2_observada_na_consulta": round(_area("area_observada"), 4), "km2_positivo_na_consulta": round(_area("positivo_inundacao"), 4),
                      "km2_negativo_na_consulta": round(_area("negativo_observado"), 4),
                      "estado": "reservado" if reservado else "presente_ligado", "pasta": _rel(pasta),
                      "evento": (_ativacoes().get(x["ativacao"]) or {}).get("eventTime", "")[:10] or None,
                      "papel": "referencia observada independente (delineacao de analista); nao e entrada do modelo"})
    cands = sorted(cands, key=lambda k: (k["data"] or "", k["id"]))
    return {"candidatos": cands, "datas": sorted({k["data"] for k in cands if k["data"]}),
            "fonte_da_busca": "cems-06-por-aoi-e-data/catalogo.csv + aoi.parquet de cada produto (poligono real)"}


def historico_f(g) -> dict:
    import pandas as pd
    cat = pd.read_parquet(R / "ia-integrada-v2/pacote_v1/historico_admitido.parquet")
    b = g.bounds
    d = cat[cat.lon.between(b[0], b[2]) & cat.lat.between(b[1], b[3])]
    from shapely import points
    if len(d):
        d = d[g.contains(points(d.lon.to_numpy(), d.lat.to_numpy()))]
    return {"estado": "presente_ligado" if len(d) else "nao_localizado", "pontos": int(len(d)),
            "eventos": sorted(d.evento_id.unique().tolist())[:20],
            "fonte_da_busca": "ia-integrada-v2/pacote_v1/historico_admitido.parquet (28.875 pontos, GFD)",
            "nao_e": "o F so existe para pontos admitidos com identidade e coordenada exatas; ausencia aqui nao e ausencia de historia"}


@lru_cache(maxsize=1)
def _gfd_brasil():
    """Pontos observados do Global Flood Database em eventos que tocam o Brasil (MODIS 250 m, 2000-2018)."""
    import re
    import pandas as pd
    pasta = R / "gfd-brasil"
    # So os arquivos POR EVENTO: `pontos_gfd_com_terreno.csv` repete os mesmos pontos com terreno e dobrava a contagem.
    arqs = sorted(pasta.glob("pontos_gfd[0-9]*.csv"))
    if not arqs:
        return None, {}
    d = pd.concat([pd.read_csv(a, usecols=["ponto_id", "classe_obs", "lat", "lon", "evento_gfd"]) for a in arqs], ignore_index=True)
    janelas = {}
    cat = pasta / "catalogo.json"
    if cat.exists():
        for e in _ler(cat).get("eventos", []):
            m = re.search(r"From_(\d{8})_to_(\d{8})", e.get("id_ee", ""))
            if m:
                a, b = m.groups()
                janelas[int(e["id"])] = (f"{a[:4]}-{a[4:6]}-{a[6:]}", f"{b[:4]}-{b[4:6]}-{b[6:]}")
    inv = R / "n1d-gfd/inventario.json"          # segunda fonte local de janelas (913 identificadores DFO com datas)
    if inv.exists():
        for ev, a, b in re.findall(r"DFO_(\d+)_From_(\d{8})_to_(\d{8})", inv.read_text(encoding="utf-8")):
            janelas.setdefault(int(ev), (f"{a[:4]}-{a[4:6]}-{a[6:]}", f"{b[:4]}-{b[4:6]}-{b[6:]}"))
    return d, janelas


def historico_observado_gfd(g) -> dict:
    """Observacao ANTERIOR por evento (GFD), em campo proprio: nao e F e nao entra no escore."""
    from shapely import points
    d, janelas = _gfd_brasil()
    if d is None:
        return {"estado": "nao_localizado", "fonte_da_busca": "local_runs/gfd-brasil/pontos_gfd<id>.csv"}
    b = g.bounds
    s = d[d.lon.between(b[0], b[2]) & d.lat.between(b[1], b[3])]
    if len(s):
        s = s[g.contains(points(s.lon.to_numpy(), s.lat.to_numpy()))]
    eventos = []
    for ev, q in s.groupby("evento_gfd"):
        j = janelas.get(int(ev))
        eventos.append({"evento": f"GFD:{int(ev)}", "inicio": j[0] if j else None, "fim": j[1] if j else None,
                        "data_registrada_localmente": bool(j), "pontos_inundados": int((q.classe_obs == "INUNDACAO").sum()),
                        "pontos_nao_inundados": int((q.classe_obs != "INUNDACAO").sum())})
    com_cheia = [e for e in eventos if e["pontos_inundados"] > 0]
    return {"estado": "presente_ligado" if eventos else "nao_localizado", "eventos": eventos,
            "eventos_com_ponto_inundado_na_aoi": len(com_cheia), "eventos_com_ponto_amostrado_na_aoi": len(eventos),
            "eventos_no_acervo": int(d.evento_gfd.nunique()),
            "licenca": "Global Flood Database v1 (Tellman et al. 2021), CC BY 4.0 -- declarada em local_runs/gfd-brasil/catalogo.json",
            "fonte_da_busca": "local_runs/gfd-brasil/pontos_gfd<id>.csv (eventos GFD que tocam o Brasil)",
            "nao_e": ("F admitido nem frequencia completa: os pontos sao AMOSTRA (passo 250 m numa faixa de 2 km em torno do "
                      "inundado de cada evento); ausencia de ponto na AOI nao e ausencia de cheia")}


def contexto_documental(g, intervalo: tuple | None = None) -> dict:
    """Registros datados da tabela unica dentro da AOI (ocorrencias, negativos observados), por fonte. Nao e F."""
    import pyarrow.dataset as ds
    from shapely import points
    b = g.bounds
    tab = ds.dataset(R / "ds-05-tabela-unica/tabela_unica_v2.parquet")
    f = (ds.field("lon") >= b[0]) & (ds.field("lon") <= b[2]) & (ds.field("lat") >= b[1]) & (ds.field("lat") <= b[3])
    d = tab.to_table(filter=f, columns=["fonte", "classe", "classe_nome", "data_evento", "precisao_data", "mecanismo",
                                       "procedencia", "admitido", "lat", "lon", "nivel_negativo"]).to_pandas()
    if len(d):
        d = d[g.contains(points(d.lon.to_numpy(), d.lat.to_numpy()))]
    if intervalo:
        d = d[(d.data_evento.astype(str) >= intervalo[0]) & (d.data_evento.astype(str) <= intervalo[1])]
    por = {}
    for fonte, s in d.groupby("fonte"):
        pos = s[s.classe == 1]
        datas = sorted(pos.data_evento.dropna().astype(str).str[:10].unique())
        por[fonte] = {"registros": int(len(s)), "positivos": int(len(pos)), "negativos": int((s.classe == 0).sum()),
                      "datas_com_positivo": len(datas), "primeira": datas[0] if datas else None, "ultima": datas[-1] if datas else None,
                      "precisao_data": s.precisao_data.astype(str).value_counts().to_dict(),
                      "procedencia": s.procedencia.astype(str).value_counts().head(3).to_dict(),
                      "ultimas_datas": datas[-8:]}
    return separar_documental({"registros": int(len(d)), "por_fonte": por,
                               "fonte_da_busca": "ds-05-tabela-unica/tabela_unica_v2.parquet (lat/lon dentro da AOI)"})


# Fontes da tabela unica que NAO sao registro documental: sao pontos amostrados de uma observacao que o resolvedor ja
# liga em outra modalidade. Conta-las como "documental" contaria a mesma referencia duas vezes (achado de 19/10/2026:
# 63 das 69 AOIs do inventario so tinham `cems` como "documental").
FONTES_DERIVADAS_DE_OBSERVACAO = {
    "cems": "referencia_cems (pontos amostrados dos produtos CEMS da ativacao)",
    "gfd": "historico_observado_gfd (pontos amostrados do MODIS por evento)",
    "sen1floods11": "sar_chip (rotulo manual sobre o chip Sentinel-1)",
    "ufo": "observacao urbana rotulada por chip (UFO)",
}


def separar_documental(doc: dict) -> dict:
    """Separa registro documental (centrais 156/311, defesa civil, contornos oficiais) de ponto derivado de observacao
    ja contada. Idempotente: aceita o formato antigo (so `por_fonte`) e devolve o novo."""
    por = doc.get("por_fonte") or {}
    docs = {f: v for f, v in por.items() if f not in FONTES_DERIVADAS_DE_OBSERVACAO}
    deriv = {**(doc.get("derivados_de_observacao") or {}),
             **{f: {"registros": v.get("registros"), "ja_contado_em": FONTES_DERIVADAS_DE_OBSERVACAO[f]}
                for f, v in por.items() if f in FONTES_DERIVADAS_DE_OBSERVACAO}}
    n = sum(v.get("registros", 0) for v in docs.values())
    return {"estado": "presente_ligado" if n else "nao_localizado", "registros": int(n), "por_fonte": docs,
            "derivados_de_observacao": deriv,
            "fonte_da_busca": doc.get("fonte_da_busca", "ds-05-tabela-unica/tabela_unica_v2.parquet (lat/lon dentro da AOI)"),
            "nao_e": ("frequencia F nem observacao de satelite: e registro documental por fonte (central de chamados, defesa "
                      "civil, contorno oficial), com a precisao de data declarada; pontos derivados de CEMS/GFD/chips ficam em "
                      "`derivados_de_observacao` e nao contam como modalidade propria")}


def social(g) -> dict:
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "social"))
    import social07_servico_medidas as s7
    b = g.bounds
    aoi = {"lon_min": b[0], "lat_min": b[1], "lon_max": b[2], "lat_max": b[3]}
    out = {}
    for regiao in ("curitiba", "rs_emsr720", "am_piloto"):
        try:
            m = s7.consultar(regiao, aoi)
            if m["setores_intersectados"]:
                out[regiao] = {"estado": "presente_ligado", "setores": m["setores_intersectados"]}
        except (ValueError, KeyError) as e:
            out[regiao] = {"estado": "erro", "motivo": str(e)}
    return {"regioes": out, "estado": "presente_ligado" if any(v.get("estado") == "presente_ligado" for v in out.values()) else "nao_localizado",
            "fonte_da_busca": "social-consolidacao/2026-10-12-v3 (poligonos IBGE 2022)",
            "nota": "consulta por caixa envolvente do poligono; as medidas da resposta usam o poligono do setor"}


def _sem_nao_finito(o):
    """Metadado ausente vira None: NaN de coluna vazia do indice (twi, licenca_mdt...) nao e valor e quebra o JSON
    estrito da rota HTTP. O nodata NaN de raster ja sai declarado como a string "NaN"."""
    if isinstance(o, dict):
        return {k: _sem_nao_finito(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sem_nao_finito(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def resolver(aoi, evento_data: str | None = None, intervalo: tuple | None = None, modalidades: tuple | None = None,
             aquisicao_data: str | None = None, sar_unidade: str | None = None) -> dict:
    g = poligono(aoi)
    todas = {"terreno": lambda: terreno(g), "sar_proprio": lambda: sar_proprio(g, evento_data, aquisicao_data, sar_unidade, intervalo),
             "sar_chip": lambda: sar_chip(g, evento_data, intervalo), "referencia_cems": lambda: referencia_cems(g, evento_data, intervalo),
             "historico_f": lambda: historico_f(g), "historico_observado_gfd": lambda: historico_observado_gfd(g),
             "contexto_documental": lambda: contexto_documental(g, intervalo),
             "social": lambda: social(g)}
    usar = modalidades or tuple(todas)
    res = {}
    for k in usar:
        try:
            res[k] = todas[k]()
        except Exception as e:  # noqa: BLE001 -- falha de uma fonte vira estado declarado, nao ausencia silenciosa
            res[k] = {"estado": "erro_na_busca", "erro": f"{type(e).__name__}: {e}"}
    estados = {}
    for k, v in res.items():
        for c in v.get("candidatos", []):
            estados[c.get("estado")] = estados.get(c.get("estado"), 0) + 1
    return {"versao": VERSAO, "aoi_km2": round(_area_km2(g), 4), "aoi_bounds": list(g.bounds), "evento_data": evento_data,
            "intervalo": list(intervalo) if intervalo else None,
            "intervalo_aplicado_a": ["sar_proprio (aquisicoes UTC integralmente dentro)", "sar_chip (data posterior do catalogo)", "referencia_cems (data do produto)", "contexto_documental"],
            "historico_temporal": "F e observacoes anteriores conservam o catalogo admitido; nao sao recalculados pelo intervalo pedido",
            "modalidades": _sem_nao_finito(res), "estados_dos_candidatos": estados,
            "cobertura_da_busca": {"ter01_no_indice": int(_unidades()[0].no_indice.sum()), "ter01_fora_do_indice": _unidades()[1],
                                   "sar_proprio_manifestos": len(_sar_proprio()), "reservas_conhecidas": len(_reservas())},
            "nao_e": "validacao de pixels de referencia, inferencia ou prova de coincidencia de evento; e correspondencia de acervo"}
