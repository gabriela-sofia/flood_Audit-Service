"""Resposta integrada v1 — uma jornada: AOI → execução local → resposta → explicação → fontes → social → artefato.

Orquestra componentes que já existem, sem recalcular nenhum deles e sem misturar autoridades:
  física      UMA autoridade por resposta, declarada em `autoridade_fisica`:
                contrato_servido  → `svc05.rota_inferencia` (SVC-02: portões, IC, model card, explicação verificada)
                pesquisa_q_v2     → `ia_integrada_v2.consultar` / `consulta_territorial_v2` (Q por ponto, histórico
                                    admitido, imputação explícita; promovido=false)
              escolhida pela geografia da AOI (região servida tem prioridade; o pedido pode exigir a de pesquisa).
              A outra, quando também se aplica, aparece só como `disponivel_nao_usada`, nunca somada.
  histórico   só do catálogo admitido do pacote v2 (pontos GFD com identidade e coordenada exatas); fora dele,
              `indisponivel` com motivo — nunca inventado
  SAR         observação do chip do inventário admitido que intersecta a AOI (rótulo nunca lido); senão `nao_aplicavel`
  suporte     domínio/cobertura/imputação da autoridade + motivos de ausência
  explicação  a da autoridade (itens + verificador de claims; ou contribuições que somam Q, conferidas)
  fontes      procedência (hashes) + RAG local por classe: `cientifica` (literatura) separada de `projeto`
              (estado do REV-P); corpus que não cobre a pergunta é declarado, não preenchido
  social      medidas por POLÍGONO (`social07`, setor censitário, unidade e denominador) em toda região social
              que a AOI intersecta; eixo v0 (bbox declarado) em Curitiba; sem fonte → `sem_fonte_social` (não é
              baixa vulnerabilidade). O bloco físico tem o mesmo hash antes e depois da camada social.
  artefatos   cartão e relatório são PROJEÇÕES do payload: cada campo leva o caminho de onde saiu e o hash do
              payload; `verificar_projecao` reextrai e compara (nada é recalculado).
"""
from __future__ import annotations

import hashlib
import json
import math
import types
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VERSAO = "resposta_integrada_v2"   # v1 continua servida em /integrada/v1
MAX_PIXELS_RESUMO_SAR = 1_000_000
PERGUNTA_PADRAO = "Quais variaveis explicam este escore e quais sao os limites da resposta?"
PACOTE_PESQUISA = REPO / "local_runs/ia-integrada-v2/pacote_v1"   # integridade conferida por hash a cada consulta


def sha_json(o) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _aoi_valida(aoi: dict) -> str | None:
    try:
        v = [float(aoi[k]) for k in ("lat_min", "lon_min", "lat_max", "lon_max")]
    except (KeyError, TypeError, ValueError):
        return "AOI precisa de lat_min, lon_min, lat_max, lon_max numericos"
    if not all(math.isfinite(x) for x in v):
        return "AOI com coordenada nao finita"
    if not (-90 <= v[0] < v[2] <= 90 and -180 <= v[1] < v[3] <= 180):
        return "AOI invertida ou fora de EPSG:4326"
    return None


def _grade_de_pontos(aoi: dict, n: int) -> list[dict]:
    la = [aoi["lat_min"] + (aoi["lat_max"] - aoi["lat_min"]) * (i + .5) / n for i in range(n)]
    lo = [aoi["lon_min"] + (aoi["lon_max"] - aoi["lon_min"]) * (j + .5) / n for j in range(n)]
    return [{"lat": a, "lon": b} for a in la for b in lo]


def _caixa(aoi):
    from shapely.geometry import box
    return box(aoi["lon_min"], aoi["lat_min"], aoi["lon_max"], aoi["lat_max"])


# ------------------------------------------------------------------ geografia do acervo (o que se aplica a esta AOI)

@lru_cache(maxsize=1)
def _catalogo_historico():
    import pandas as pd
    return pd.read_parquet(REPO / "local_runs/ia-integrada-v2/pacote_v1/historico_admitido.parquet")


@lru_cache(maxsize=1)
def _chips_sar() -> dict:
    import ia_integrada as anterior
    import ia_integrada_v2 as v2
    _, meta = v2.carregar()
    return {k: c["grade"]["bounds"] for k, c in anterior.inventario(meta["pai_percepcao_manifesto"]).items()}


MACRORREGIOES = {"norte", "nordeste", "centro_oeste", "sudeste", "sul"}


def aplicabilidade(aoi: dict, evento_data: str | None = None, intervalo: tuple | None = None,
                   aquisicao_data: str | None = None, sar_unidade: str | None = None) -> dict:
    import svc05_transporte_http as s5
    g = _caixa(aoi)
    if not s5.servico.pronto:
        s5.servico.carregar()
    regiao = s5._deduzir_regiao(_grade_de_pontos(aoi, 3))
    fora_do_territorio = None
    if not regiao:
        # O G2 v3 ja recusa ponto fora do territorio da macrorregiao; aqui so se registra o motivo dado pelo contrato.
        from svc02_contrato_inferencia import g2_regiao
        _, falha, _ = g2_regiao({}, _grade_de_pontos(aoi, 3), s5.servico.caixas)
        detalhe = (falha or {}).get("detalhe") or ""
        if "territorio" in detalhe:
            caixa = next((n for n, c in s5.servico.caixas.items() if n in MACRORREGIOES
                          and c["lat_min"] <= (aoi["lat_min"] + aoi["lat_max"]) / 2 <= c["lat_max"]
                          and c["lon_min"] <= (aoi["lon_min"] + aoi["lon_max"]) / 2 <= c["lon_max"]), None)
            fora_do_territorio = {"regiao_por_caixa": caixa, "motivo": detalhe,
                                  "regra": "territorio IBGE da macrorregiao (contrato v3, G2)"}
    if regiao in MACRORREGIOES:
        # Mesmo portao do contrato v3, aplicado ao POLIGONO inteiro da AOI (a requisicao ao contrato so leva a grade).
        from svc02_contrato_inferencia import territorio_da_macrorregiao
        ok, motivo = territorio_da_macrorregiao(regiao, _grade_de_pontos(aoi, 5), geometria=_caixa(aoi))
        if not ok:
            fora_do_territorio = {"regiao_por_caixa": regiao, "motivo": motivo,
                                  "regra": "territorio IBGE da macrorregiao sobre o poligono inteiro da AOI (contrato v3, G2)"}
            regiao = None
    cat = _catalogo_historico()
    dentro = cat[cat.lon.between(aoi["lon_min"], aoi["lon_max"]) & cat.lat.between(aoi["lat_min"], aoi["lat_max"])]
    # Correspondencia pelo resolvedor comum: caixa so pre-seleciona; entra quem tem pixel valido dentro da AOI.
    import resolvedor_acervo as ra
    corr = ra.resolver(aoi, evento_data=evento_data, intervalo=intervalo,
                       aquisicao_data=aquisicao_data, sar_unidade=sar_unidade,
                       modalidades=("terreno", "sar_proprio", "sar_chip", "referencia_cems", "historico_f", "historico_observado_gfd",
                                    "contexto_documental"))
    m = corr["modalidades"]
    valido = lambda c: (c.get("cobertura") or {}).get("fracao_valida_da_aoi", 0) > 0  # noqa: E731
    unidades = [c["id"].split(":", 1)[1] for c in m["terreno"].get("candidatos", []) if c["id"].startswith("rodada3:") and valido(c)]
    sel = m["sar_chip"].get("selecionado")
    chips = ([sel] if sel else []) + sorted(c["id"] for c in m["sar_chip"].get("candidatos", []) if valido(c) and c["id"] != sel)
    return {"regiao_servida": regiao or None, "fora_do_territorio_do_modelo": fora_do_territorio,
            "pontos_do_catalogo_historico": int(len(dentro)),
            "eventos_do_catalogo": sorted(dentro.evento_id.unique().tolist())[:10], "unidades_de_terreno_pesquisa": unidades,
            "chips_sar": chips, "correspondencia": corr}


# ------------------------------------------------------------------ autoridades físicas

def _contrato_servido(aoi: dict, pedido: dict) -> dict:
    import svc05_transporte_http as s5
    req = s5.RequisicaoInferencia(**{
        "geometria": {"tipo": "aoi", "crs": "EPSG:4326", "pontos": _grade_de_pontos(aoi, int(pedido.get("lado_da_grade", 5)))},
        "pontos_sem_cobertura": "descartar", "sintese": True,
        **({"cenario_de_chuva": pedido["cenario_de_chuva"]} if pedido.get("cenario_de_chuva") else {})})
    r = s5.rota_inferencia(req, types.SimpleNamespace(client=None))
    return {"requisicao": req.model_dump(exclude_none=True), "resposta": r}


def _pesquisa(aoi: dict, ap: dict, pedido: dict) -> dict:
    import ia_integrada_v2 as ia
    if ap["pontos_do_catalogo_historico"]:
        import pandas as pd
        cat = _catalogo_historico()
        dentro = cat[cat.lon.between(aoi["lon_min"], aoi["lon_max"]) & cat.lat.between(aoi["lat_min"], aoi["lat_max"])]
        dentro = dentro.sort_values("ponto_id").head(int(pedido.get("max_pontos_pesquisa", 200)))
        esc = pd.read_parquet(REPO / "local_runs/mt11-transdutivo/escores_emendado.parquet").set_index("ponto_id")
        pontos = [{**{c: (None if pd.isna(esc.loc[p, c]) else float(esc.loc[p, c])) for c in ia.BRUTAS},
                   "ponto_id": p, "lat": float(la), "lon": float(lo)}
                  for p, la, lo in zip(dentro.ponto_id, dentro.lat, dentro.lon) if p in esc.index]
        r = ia.consultar({"pontos": pontos, "usar_historico": pedido.get("usar_historico", True)}, PACOTE_PESQUISA)
        return {"modo": "pontos_do_catalogo_admitido", "atributos": "pre-computados do acervo MT11 (escores_emendado), nao extraidos do raster nesta chamada",
                "resposta": r}
    unidade = ap["unidades_de_terreno_pesquisa"][0]
    import consulta_territorial_v2 as ct
    r = ct.consultar_coordenadas({"unidade": unidade, "coordenadas": _grade_de_pontos(aoi, int(pedido.get("lado_da_grade", 5)))},
                                 PACOTE_PESQUISA)
    return {"modo": "coordenadas_no_raster_da_unidade", "unidade": unidade, "resposta": r}


def _escolher_autoridade(ap: dict, pedido: dict) -> tuple[str | None, str, str | None]:
    pesquisa_ok = bool(ap["pontos_do_catalogo_historico"] or ap["unidades_de_terreno_pesquisa"])
    servido_ok = bool(ap["regiao_servida"])
    pedida = pedido.get("autoridade_fisica", "auto")
    if pedida == "pesquisa_q_v2":
        return ("pesquisa_q_v2", "pedida explicitamente", "contrato_servido" if servido_ok else None) if pesquisa_ok else \
               (None, "pesquisa pedida, mas a AOI nao toca unidade de terreno nem catalogo admitido da pesquisa", None)
    if pedida == "contrato_servido":
        return ("contrato_servido", "pedida explicitamente", "pesquisa_q_v2" if pesquisa_ok else None) if servido_ok else \
               (None, "contrato pedido, mas a AOI nao cai em regiao servida", None)
    if servido_ok:
        return "contrato_servido", f"AOI na regiao servida '{ap['regiao_servida']}' (prioridade do contrato)", "pesquisa_q_v2" if pesquisa_ok else None
    if pesquisa_ok:
        return "pesquisa_q_v2", "AOI fora das regioes servidas e dentro do acervo admitido da pesquisa", None
    return None, "AOI fora das regioes servidas e do acervo admitido da pesquisa", None


# ------------------------------------------------------------------ blocos

def _bloco_fisico(autoridade, execucao) -> dict:
    if autoridade == "contrato_servido":
        r = execucao["resposta"]
        mc = r.get("model_card") or {}
        return {"autoridade": "contrato_servido", "estimador": mc.get("modelo"), "versao_contrato": r.get("contrato"),
                "status": r.get("status"), "regiao": r.get("regiao"), "maturidade": r.get("maturidade"),
                "escore": r.get("escore"), "ic95": r.get("ic95"), "unidade_de_resposta": r.get("unidade_de_resposta"),
                "gate_que_nao_fechou": r.get("gate_que_nao_fechou"), "detalhe": r.get("detalhe"),
                "afirmacao_autorizada": r.get("afirmacao_autorizada"), "n_pontos": r.get("n_pontos"),
                "observado_vs_imputado": {"modo": (r.get("procedencia_das_camadas") or {}).get("modo"),
                                          "valores_preenchidos_pela_grade": (r.get("procedencia_das_camadas") or {}).get("valores_preenchidos"),
                                          "pontos_fora_de_toda_celula": (r.get("procedencia_das_camadas") or {}).get("pontos_fora_de_toda_celula_servivel"),
                                          "imputacao_pela_mediana": False,
                                          "leitura": "camadas observadas na grade derivada de 120 m; ponto sem celula e descartado e declarado, nunca imputado"}}
    r = execucao["resposta"]
    ok = [p for p in r.get("pontos", []) if p.get("status") == "pesquisa"]
    qs = [p["Q_conexao"] for p in ok]
    return {"autoridade": "pesquisa_q_v2", "estimador": "IAIntegradaV2 Q_conexao", "versao": r.get("versao"),
            "modelo_sha256": r.get("modelo_sha256"), "status": r.get("status"), "maturidade": r.get("maturidade"),
            "promovido": r.get("promovido"), "probabilidade_calibrada": r.get("probabilidade_calibrada"),
            "modo": execucao.get("modo"), "pontos": len(r.get("pontos", [])), "respondidos": len(ok),
            "Q_conexao_mediana": sorted(qs)[len(qs) // 2] if qs else None,
            "Q_conexao_min_max": [min(qs), max(qs)] if qs else None,
            "observado_vs_imputado": {"pontos_com_imputacao": sum(bool(p.get("atributos_imputados")) for p in ok),
                                      "atributos_imputados": sorted({a for p in ok for a in p.get("atributos_imputados", [])}),
                                      "faixa_por_imputacao": "por ponto em `execucao.resposta.pontos[].imputacao` (nao e IC)"},
            "recusados": [{"ponto_id": p.get("ponto_id"), "motivo": p.get("motivo")} for p in r.get("pontos", []) if p.get("status") == "recusado"][:10],
            "nota": "escore de pesquisa; nao e probabilidade calibrada nem o escore do contrato servido"}


def _bloco_historico(autoridade, execucao, ap) -> dict:
    if autoridade == "pesquisa_q_v2":
        ok = [p for p in execucao["resposta"].get("pontos", []) if p.get("status") == "pesquisa"]
        est = {}
        for p in ok:
            k = p["historico"].get("motivo", p["historico"]["estado"])
            est[k] = est.get(k, 0) + 1
        adm = [p for p in ok if p["historico"]["estado"] == "admitido_no_catalogo"]
        return {"estado": "admitido" if adm else "indisponivel", "por_ponto": est,
                "eventos": sorted({p["historico"].get("evento") for p in adm if p["historico"].get("evento")}),
                "F_transf_mediana": sorted(p["F_transf"] for p in adm)[len(adm) // 2] if adm else None,
                "fonte": "local_runs/ia-integrada-v2/pacote_v1/historico_admitido.parquet (identidade e coordenada exatas)"}
    return {"estado": "indisponivel", "pontos_do_catalogo_na_aoi": ap["pontos_do_catalogo_historico"],
            "motivo": ("o catalogo admitido (28.875 pontos de eventos GFD, nenhum no Brasil) nao cobre esta AOI; "
                       "historico nao e inventado nem interpolado"),
            "nao_e": "ausencia de inundacao no passado"}


def _bloco_sar(ap) -> dict:
    """PERCEPCAO calculada (cabeca de pesquisa v2). So roda em chip da arena do peso; nada aqui e dado bruto."""
    if not ap["chips_sar"]:
        proprio = [c for c in ((ap.get("correspondencia") or {}).get("modalidades", {}).get("sar_proprio", {}).get("candidatos") or [])
                   if (c.get("cobertura") or {}).get("fracao_valida_da_aoi", 0) > 0]
        if proprio:
            return {"estado": "nao_executada", "motivo": proprio[0]["usos"]["percepcao_peso_pesquisa_v2"],
                    "sar_presente": [c["id"] for c in proprio]}
        return {"estado": "nao_aplicavel", "motivo": "nenhum chip do inventario admitido (418, Sen1Floods11) com pixel valido na AOI"}
    import ia_integrada_v2 as ia
    chip = ap["chips_sar"][0]
    r = ia.consultar({"chip": chip}, PACOTE_PESQUISA)
    return {"estado": "observado", "chip": chip, "chips_intersectados": len(ap["chips_sar"]), **(r.get("observacao") or {}),
            "selecao": "maior fracao valida da cena na AOI (resolvedor); empate por data e nome",
            "autoridade": "cabeca de percepcao do pacote de pesquisa v2 (separada do escore fisico)"}


def _estatisticas_sar(arquivo: Path, aoi: dict) -> dict:
    """Resumo do DADO BRUTO (dB) dentro do poligono: nao e deteccao de agua."""
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds
    from shapely.geometry import box
    from shapely.ops import transform as tr_geom
    with rasterio.open(arquivo) as r:
        g = tr_geom(Transformer.from_crs(4326, r.crs, always_xy=True).transform, _caixa(aoi)).intersection(box(*r.bounds))
        if g.is_empty:
            return {"pixels_validos": 0}
        w = from_bounds(*g.bounds, transform=r.transform).round_offsets().round_lengths().intersection(Window(0, 0, r.width, r.height))
        fator=max(1,int(math.ceil(math.sqrt(w.width*w.height/MAX_PIXELS_RESUMO_SAR))))
        forma=(max(1,int(w.height//fator)),max(1,int(w.width//fator)))
        a = r.read(window=w, out_shape=(r.count,*forma), masked=True).astype("float64").filled(np.nan)
        tw=r.window_transform(w)*rasterio.Affine.scale(w.width/forma[1],w.height/forma[0])
        dentro = geometry_mask([g], out_shape=a.shape[1:], transform=tw, invert=True)
        nomes = list(r.descriptions) if any(r.descriptions) else [f"banda{i + 1}" for i in range(r.count)]
    validos = dentro & np.isfinite(a).all(axis=0)
    out = {"pixels_validos": int(validos.sum()), "mascara": "intersecao de todas as bandas, mascaras internas e NoData",
           "fator_decimacao":fator,"contagem": "nativa" if fator==1 else "grade_amostrada_vizinho_mais_proximo",
           "percentis": "nativos" if fator==1 else "estimativas na grade amostrada; nao contagem nativa"}
    for i, n in enumerate(nomes):
        v = a[i][validos]
        if v.size:
            out[n] = {"p10": round(float(np.percentile(v, 10)), 3), "mediana": round(float(np.median(v)), 3),
                      "p90": round(float(np.percentile(v, 90)), 3)}
    return out


def _bloco_sar_bruto(ap, aoi) -> dict:
    sp = (ap.get("correspondencia") or {}).get("modalidades", {}).get("sar_proprio", {})
    if sp.get('estado')=='erro_na_busca':
        return {'estado':'erro_na_busca','erro':sp.get('erro'),'nao_e':'ausencia de SAR ou agua; consulta de fonte falhou'}
    cands = [c for c in sp.get("candidatos") or [] if (c.get("cobertura") or {}).get("fracao_valida_da_aoi", 0) > 0]
    if not cands:
        catalogados=sp.get('candidatos') or []
        if catalogados:
            reservado=any(c.get('estado')=='reservado' and c.get('compativel_com_pedido_exato',True) for c in catalogados)
            return {'estado':'reservado' if reservado else 'sem_selecao_compativel','pedido_exato':sp.get('pedido_exato'),
                    'motivo':sp.get('motivo_sem_selecao'),'candidatos_catalogados':[c['id'] for c in catalogados],
                    'nao_e':'ausencia de SAR ou de agua; nenhum resumo de cena substituta foi produzido'}
        return {"estado": "nao_localizado", "fonte_da_busca": sp.get("fonte_da_busca"),
                "nao_e": "ausencia de agua nem impossibilidade de observar: nenhum SAR proprio de AOI com pixel valido aqui"}
    if not sp.get("selecionado"):
        reservados = [x for x in cands if x.get("estado") == "reservado" and x.get("compativel_com_pedido_exato", True)]
        return {"estado": "reservado" if reservados else "sem_selecao_compativel",
                "pedido_exato": sp.get("pedido_exato"), "motivo": sp.get("motivo_sem_selecao"),
                "candidatos_presentes": [x["id"] for x in cands],
                "nao_e": "ausencia de SAR no acervo nem ausencia de agua; nenhum resumo de cena substituta foi produzido"}
    c = next(x for x in cands if x["id"] == sp["selecionado"])
    if c["estado"] == "reservado":
        return {"estado": "reservado", "id": c["id"], "nao_e": "conteudo nao lido"}
    return {"estado": "presente_ligado", "id": c["id"], "arquivo": c["arquivo"], "manifesto": c["manifesto"], "colecao": c["colecao"],
            "data_da_imagem": c["data_da_imagem"], "datas_de_aquisicao_utc": c.get("datas_de_aquisicao_utc"),
            "mosaico_multidata": c.get("mosaico_multidata"), "data_pedida_no_manifesto": c.get("data_pedida_no_manifesto"),
            "cenas": c["cenas"], "unidade_radiometrica": c["unidade_radiometrica"],
            "bandas": c["bandas"], "resolucao_m": (c["cobertura"].get("res") or [None])[0], "crs": c["cobertura"].get("crs"),
            "fracao_valida_da_aoi": c["cobertura"]["fracao_valida_da_aoi"], "papel_temporal": c["papel_temporal"], "usos": c["usos"],
            "estatisticas_db_na_aoi": _estatisticas_sar(REPO / c["arquivo"], aoi), "outros_candidatos": [x["id"] for x in cands if x is not c],
            "selecao": sp.get("regra"), "nao_e": "deteccao de agua, percepcao calculada nem referencia independente"}


def _bloco_referencia(ap, sar_bruto) -> dict:
    rc = (ap.get("correspondencia") or {}).get("modalidades", {}).get("referencia_cems", {})
    if rc.get('estado')=='erro_na_busca':
        return {'estado':'erro_na_busca','erro':rc.get('erro'),'nao_e':'ausencia de referencia ou agua; consulta de fonte falhou'}
    cands = [c for c in rc.get("candidatos") or [] if c.get("km2_observada_na_consulta", 0) > 0]
    if not cands:
        if any(c.get('estado')=='reservado' for c in rc.get('candidatos',[])):
            return {'estado':'reservado','conteudo':'nao_lido','nao_e':'ausencia de referencia ou agua'}
        return {"estado": "nao_localizado", "fonte_da_busca": rc.get("fonte_da_busca"),
                "nao_e": "ausencia de cheia: nenhum produto CEMS observado cobre esta AOI no acervo local"}
    # Coincidencia pela data de AQUISICAO das cenas (dia UTC), nunca pela data pedida do manifesto; +-1 dia fica a parte.
    from datetime import date
    datas_sar = (sar_bruto or {}).get("datas_de_aquisicao_utc") or []
    perto = lambda c, n: any(abs((date.fromisoformat(c["data"]) - date.fromisoformat(a)).days) == n for a in datas_sar)  # noqa: E731
    return {"estado": "presente_ligado", "produtos": [{k: c[k] for k in ("id", "tipo_produto", "data", "sensor", "km2_observada_na_consulta",
                                                                      "km2_positivo_na_consulta", "evento", "estado")} for c in cands],
            "datas": sorted({c["data"] for c in cands}), "mesma_data_do_sar_bruto": [c["id"] for c in cands if perto(c, 0)],
            "a_um_dia_do_sar_bruto": [c["id"] for c in cands if perto(c, 1) and not perto(c, 0)],
            "papel": "referencia observada independente (delineacao de analista); nao entra no escore nem na percepcao",
            "fonte_da_busca": rc.get("fonte_da_busca")}


def _bloco_contexto_historico(ap, historico_f) -> dict:
    doc = (ap.get("correspondencia") or {}).get("modalidades", {}).get("contexto_documental", {})
    gfd = (ap.get("correspondencia") or {}).get("modalidades", {}).get("historico_observado_gfd", {})
    return {"f_admitido": historico_f, "observacao_anterior_gfd": gfd, "documental": doc,
            "regra": ("F so vem do catalogo admitido (identidade, coordenada e numero minimo de datas); registros documentais entram "
                      "em campo proprio, com fonte e precisao de data, e nunca viram F nem zero")}


def _bloco_suporte(autoridade, execucao, motivos) -> dict:
    if autoridade == "contrato_servido":
        r = execucao["resposta"]
        return {"dominio": r.get("dominio"), "suporte_cientifico": r.get("suporte_cientifico"),
                "cobertura_da_area": r.get("cobertura_da_area"), "motivos_de_ausencia": motivos}
    if autoridade == "pesquisa_q_v2":
        r = execucao["resposta"]
        cob = {}
        for p in r.get("pontos", []):
            k = p.get("cobertura_terreno", "nao_informado")
            cob[k] = cob.get(k, 0) + 1
        return {"cobertura_terreno": cob, "confianca_local": "nao calibrada", "limites": r.get("limites"),
                "seletor_de_suporte": "nao ha regra de abstencao validada (frente C: nao distinguivel do acaso)",
                "motivos_de_ausencia": motivos}
    return {"motivos_de_ausencia": motivos}


def _bloco_explicacao(autoridade, execucao) -> dict:
    if autoridade == "contrato_servido":
        r = execucao["resposta"]
        e = r.get("explicacao") or {}
        a = r.get("auditoria_sintese") or {}
        return {"texto": e.get("texto"), "itens": e.get("itens"), "sintese": r.get("sintese"),
                "verificador": {"motor": a.get("motor"), "e_llm": a.get("e_llm"), "aprovadas": a.get("aprovadas"), "reprovadas": a.get("reprovadas")},
                "fiel_ao_calculo": "itens vem do mesmo payload que gerou o escore; frase so publica claim verificado"}
    if autoridade == "pesquisa_q_v2":
        r = execucao["resposta"]
        ok = [p for p in r.get("pontos", []) if p.get("status") == "pesquisa"]
        resid = max((abs(sum(p["contribuicoes_q"].values()) - p["Q_conexao"]) for p in ok), default=None)
        media = {}
        for p in ok:
            for k, v in p["contribuicoes_q"].items():
                media[k] = media.get(k, 0.0) + v / len(ok)
        return {"texto": r.get("explicacao"), "contribuicoes_medias_Q": dict(sorted(media.items(), key=lambda kv: -abs(kv[1]))[:8]),
                "residuo_max_soma_contribuicoes_menos_Q": resid, "fiel_ao_calculo": resid is not None and resid < 1e-9,
                "nao_e": "atribuicao causal"}
    return {"estado": "sem_escore", "motivo": "nao ha autoridade fisica para esta AOI"}


def _bloco_fontes(autoridade, execucao, pergunta) -> dict:
    out = {"pergunta": pergunta}
    if autoridade == "contrato_servido":
        from consulta_cognitiva import responder
        r = execucao["resposta"]
        out["proveniencia"] = r.get("proveniencia")
        for modo in ("cientifico", "proveniencia"):
            c = responder(pergunta, r, modo=modo)
            out[modo] = {"classe": c.get("classe_de_autoridade_consultada"), "fontes": c.get("fontes"),
                         "descartados_por_injecao": len(c.get("descartados_por_injecao") or []),
                         "cobertura": "coberta" if c.get("fontes") else "corpus nao cobre a pergunta nesta classe"}
    else:
        from rag_busca import buscar
        if autoridade == "pesquisa_q_v2":
            r = execucao["resposta"]
            out["proveniencia"] = {"modelo_sha256": r.get("modelo_sha256"),
                                   "territorial": (r.get("procedencia_territorial") or {}).get("fontes")}
        for classe, nome in (("cientifica", "cientifico"), ("projeto", "proveniencia")):
            b = buscar(pergunta, limite=5, classe=classe)
            fs = [{k: t.get(k) for k in ("chunk_id", "source_id", "titulo", "classe_autoridade", "hash_chunk", "doi", "url")}
                  for t in b.get("trechos") or b.get("resultados") or []]
            out[nome] = {"classe": classe, "fontes": fs, "cobertura": "coberta" if fs else "corpus nao cobre a pergunta nesta classe"}
    out["literatura_por_variavel"] = _literatura_por_variavel(autoridade, execucao)
    out["distincao"] = "classe cientifica = literatura curada com DOI/URL; classe projeto = estado do REV-P (nao e literatura)"
    return out


TERMOS_DE_VARIAVEL = {"hand": "HAND altura acima da drenagem", "twi": "indice topografico umidade", "slope": "declividade",
                      "declive": "declividade", "sca": "area de contribuicao", "dist": "drenagem", "elevation": "elevacao",
                      "construido": "construido", "areia": "solo areia"}


def _literatura_por_variavel(autoridade, execucao) -> list[dict]:
    """Para cada variavel que a explicacao cita, o trecho de LITERATURA (classe cientifica) que a descreve, se houver."""
    from rag_busca import buscar
    if autoridade == "contrato_servido":
        nomes = [(i["variavel"], i.get("nome") or i["variavel"]) for i in ((execucao["resposta"].get("explicacao") or {}).get("itens") or [])]
    elif autoridade == "pesquisa_q_v2":
        ok = [p for p in execucao["resposta"].get("pontos", []) if p.get("status") == "pesquisa"]
        termos = sorted({t for p in ok for t in p["contribuicoes_q"]})
        nomes = []
        for t in termos:
            q = next((v for k, v in TERMOS_DE_VARIAVEL.items() if k in t.lower()), None)
            if q and q not in [n for _, n in nomes]:
                nomes.append((t, q))
    else:
        return []
    out = []
    for var, consulta in nomes:
        b = buscar(consulta, limite=2, classe="cientifica")
        hits = [{"titulo": r.get("titulo"), "doi": r.get("doi"), "url": r.get("url"), "hash_chunk": r.get("hash_chunk")}
                for r in b.get("resultados") or []]
        out.append({"variavel": var, "consulta": consulta, "literatura": hits,
                    "estado": "coberta" if hits else "sem_referencia_no_corpus_cientifico_local"})
    return out


def _bloco_social(aoi, fisica_payload) -> dict:
    import social07_servico_medidas as s7
    import svc05_transporte_http as s5
    h_antes = sha_json(fisica_payload)
    regioes = {}
    for regiao in ("curitiba", "rs_emsr720", "am_piloto"):
        try:
            m = s7.consultar(regiao, aoi)
        except (ValueError, KeyError, FileNotFoundError) as e:
            regioes[regiao] = {"estado": "erro_estruturado", "motivo": f"{type(e).__name__}: {e}"}
            continue
        if m["setores_intersectados"]:
            ctx = (m["exposicao"].get("contexto_aoi") or {})
            regioes[regiao] = {"estado": "medida", "unidade": "setor censitario IBGE 2022 (poligono oficial)",
                               "setores_intersectados": m["setores_intersectados"],
                               "denominadores": "componentes = razao das somas (numerador/denominador) dos setores intersectados",
                               "vulnerabilidade": m["vulnerabilidade"], "exposicao_limites": ctx.get("limites_localizacao"),
                               "exposicao_estudo_regional": (m["exposicao"].get("estudo_exploratorio_regional") or {}).get("estado"),
                               "limites": m["limites"], "fontes_manifesto_sha256": m["fontes_manifesto_sha256"]}
    v0 = None
    if "curitiba" in regioes and s5.servico.unidades_sociais.get("curitiba"):
        v0 = s5.avaliar_aoi_social(aoi, s5.servico.unidades_sociais["curitiba"], fisica_payload, cobertura_minima=.25)
        v0 = {"contrato": v0.get("contrato"), "status": v0.get("status"), "eixos": v0.get("eixos"),
              "geometria": "bbox do setor (piloto v1) -- aproximacao declarada; as medidas acima usam poligono"}
    estado = "medida" if regioes and any(r["estado"] == "medida" for r in regioes.values()) else "sem_fonte_social"
    return {"estado": estado, "regioes": regioes, "eixo_v0_curitiba": v0,
            "fisico_inalterado": sha_json(fisica_payload) == h_antes, "hash_fisico": h_antes,
            **({"nao_e": "baixa vulnerabilidade social; a fonte nao cobre a AOI"} if estado == "sem_fonte_social" else {}),
            "nao_altera": "escore fisico, IC, maturidade e portoes"}


# ------------------------------------------------------------------ orquestração

def responder(pedido: dict, versao: str = "v1") -> dict:
    aoi = pedido.get("aoi") or {}
    erro = _aoi_valida(aoi)
    pergunta = pedido.get("pergunta") or PERGUNTA_PADRAO
    base = {"versao": f"resposta_integrada_{versao}", "pedido": {k: v for k, v in pedido.items()}, "pergunta": pergunta}
    if erro:
        p = {**base, "status": "recusado", "motivo": erro, "autoridade_fisica": None, "fisico": None,
             "recusas": [{"etapa": "validacao_da_aoi", "motivo": erro}]}
        p["hash_payload"] = sha_json({k: v for k, v in p.items() if k != "hash_payload"})
        return p
    aoi = {k: float(aoi[k]) for k in ("lat_min", "lon_min", "lat_max", "lon_max")}
    for nome,padrao,maximo in (("lado_da_grade",5,50),("max_pontos_pesquisa",200,5000)):
        valor=pedido.get(nome,padrao)
        if isinstance(valor,bool) or not isinstance(valor,int) or not 1<=valor<=maximo:
            p={**base,"status":"recusado","motivo":f"{nome} deve ser inteiro entre 1 e {maximo}",
               "autoridade_fisica":None,"fisico":None,"recusas":[{"etapa":"limite_do_pedido","motivo":nome}]}
            p['hash_payload']=sha_json(p)
            return p
    if pedido.get("uso") in ("previsao_de_evento", "uso_operacional", "laudo"):
        p = {**base, "status": "recusado", "motivo": f"uso '{pedido['uso']}' fora do escopo: o REV-P nao preve evento nem emite laudo",
             "autoridade_fisica": None, "fisico": None, "recusas": [{"etapa": "escopo", "motivo": pedido["uso"]}]}
        p["hash_payload"] = sha_json({k: v for k, v in p.items() if k != "hash_payload"})
        return p
    intervalo = tuple(pedido["intervalo"]) if pedido.get("intervalo") else None
    erro_temporal = None
    try:
        from datetime import date
        for k in ("evento_data", "aquisicao_data"):
            if pedido.get(k):
                date.fromisoformat(pedido[k])
        if intervalo and (len(intervalo) != 2 or date.fromisoformat(intervalo[0]) > date.fromisoformat(intervalo[1])):
            raise ValueError("intervalo precisa de duas datas em ordem")
    except (TypeError, ValueError) as e:
        erro_temporal = f"data ou intervalo invalido: {e}"
    if erro_temporal:
        p = {**base, "status": "recusado", "motivo": erro_temporal, "autoridade_fisica": None, "fisico": None,
             "recusas": [{"etapa": "validacao_temporal", "motivo": erro_temporal}]}
        p["hash_payload"] = sha_json(p)
        return p
    ap = aplicabilidade(aoi, pedido.get("evento_data"), intervalo, pedido.get("aquisicao_data"), pedido.get("sar_unidade"))
    autoridade, porque, outra = _escolher_autoridade(ap, pedido)
    motivos, recusas = [], []
    execucao = None
    if autoridade == "contrato_servido":
        execucao = _contrato_servido(aoi, pedido)
        if execucao["resposta"].get("status") != "ok":
            recusas.append({"etapa": "contrato_servido", "gate": execucao["resposta"].get("gate_que_nao_fechou"),
                            "motivo": execucao["resposta"].get("detalhe")})
    elif autoridade == "pesquisa_q_v2":
        try:
            execucao = _pesquisa(aoi, ap, pedido)
        except (ValueError, OSError, KeyError) as e:
            recusas.append({"etapa": "pesquisa_q_v2", "motivo": f"{type(e).__name__}: {e}"})
            autoridade = None
    else:
        motivos.append(porque)
    if ap.get("fora_do_territorio_do_modelo"):
        motivos.append(ap["fora_do_territorio_do_modelo"]["motivo"])
    fisico = _bloco_fisico(autoridade, execucao) if execucao else None
    try:
        sar = _bloco_sar(ap)
    except (ValueError, OSError, KeyError) as e:
        sar = {"estado": "erro_estruturado", "motivo": f"{type(e).__name__}: {e}"}
    payload = {
        **base, "status": ("ok" if fisico and fisico.get("status") in ("ok", "pesquisa") else
                           "parcial" if sar.get("estado") == "observado" or (versao == "v2" and _tem_observacao(ap)) else "recusado"),
        "aoi": aoi, "aplicabilidade": ({k: v for k, v in ap.items() if k != "correspondencia"} if versao == "v1" else ap),
        "autoridade_fisica": {"usada": autoridade, "porque": porque, "disponivel_nao_usada": outra,
                              "regra": "uma autoridade por resposta; nunca combinadas"},
        "execucao": execucao, "fisico": fisico,
        "historico": _bloco_historico(autoridade, execucao, ap) if execucao else _bloco_historico(None, None, ap),
        "percepcao_sar": sar, "suporte": _bloco_suporte(autoridade, execucao, motivos),
        **(_blocos_v2(ap, aoi, autoridade, execucao) if versao == "v2" else {}),
        "explicacao": _bloco_explicacao(autoridade, execucao),
        "fontes": _bloco_fontes(autoridade, execucao, pergunta),
        "social": _bloco_social(aoi, fisico),
        "limitacoes": ((execucao or {}).get("resposta") or {}).get("limitacoes") or ((execucao or {}).get("resposta") or {}).get("limites") or [],
        "recusas": recusas,
        "nao_e": "laudo, previsao de evento, probabilidade calibrada ou atribuicao causal"}
    if payload["status"] != "ok":
        payload["motivo"] = "; ".join(motivos + [str(r.get("motivo")) for r in recusas]) or None
    payload["hash_payload"] = sha_json({k: v for k, v in payload.items() if k != "hash_payload"})
    return payload


def _tem_observacao(ap) -> bool:
    m = (ap.get("correspondencia") or {}).get("modalidades", {})
    return any((c.get("cobertura") or {}).get("fracao_valida_da_aoi", 0) > 0 for c in m.get("sar_proprio", {}).get("candidatos", [])) or \
        any(c.get("km2_observada_na_consulta", 0) > 0 for c in m.get("referencia_cems", {}).get("candidatos", []))


def _blocos_v2(ap, aoi, autoridade, execucao) -> dict:
    hist = _bloco_historico(autoridade, execucao, ap) if execucao else _bloco_historico(None, None, ap)
    bruto = _bloco_sar_bruto(ap, aoi)
    return {"sar_bruto": bruto, "referencia_observada": _bloco_referencia(ap, bruto), "contexto_historico": _bloco_contexto_historico(ap, hist)}


# ------------------------------------------------------------------ artefatos (projeções)

CAMPOS_DO_CARTAO = [
    ("Status", "status"), ("Autoridade fisica", "autoridade_fisica.usada"), ("Estimador", "fisico.estimador"),
    ("Versao do contrato", "fisico.versao_contrato"), ("Versao da pesquisa", "fisico.versao"), ("Escore", "fisico.escore"),
    ("IC95", "fisico.ic95"), ("Q mediano", "fisico.Q_conexao_mediana"), ("Maturidade", "fisico.maturidade"),
    ("Imputacao", "fisico.observado_vs_imputado"), ("Historico", "historico.estado"), ("SAR", "percepcao_sar.estado"),
    ("Social", "social.estado"), ("Fisico inalterado pelo social", "social.fisico_inalterado"),
    ("Explicacao", "explicacao.texto"), ("Fontes cientificas", "fontes.cientifico.cobertura"),
    ("Fontes do projeto", "fontes.proveniencia.cobertura"), ("SAR bruto (estado)", "sar_bruto.estado"),
    ("SAR bruto (data)", "sar_bruto.data_da_imagem"), ("SAR bruto (papel temporal)", "sar_bruto.papel_temporal.papel"),
    ("SAR bruto (cobertura valida)", "sar_bruto.fracao_valida_da_aoi"), ("Referencia observada", "referencia_observada.estado"),
    ("Referencia (datas)", "referencia_observada.datas"), ("Referencia na data do SAR", "referencia_observada.mesma_data_do_sar_bruto"),
    ("Contexto documental", "contexto_historico.documental.estado"),
    ("Observacao anterior (GFD)", "contexto_historico.observacao_anterior_gfd.eventos_com_ponto_inundado_na_aoi"), ("Recusas", "recusas"), ("Motivo", "motivo")]


def _pegar(p: dict, caminho: str):
    o = p
    for k in caminho.split("."):
        if not isinstance(o, dict) or k not in o:
            return None
        o = o[k]
    return o


def cartao(payload: dict) -> dict:
    return {"artefato": "cartao_resposta_integrada", "versao": payload.get("versao", VERSAO), "hash_payload": payload["hash_payload"],
            "campos": [{"rotulo": r, "caminho": c, "valor": _pegar(payload, c)} for r, c in CAMPOS_DO_CARTAO
                       if _pegar(payload, c) is not None]}


def relatorio_md(payload: dict) -> str:
    c = cartao(payload)
    linhas = [f"# Resposta integrada — {payload.get('status')}", "", f"Versao `{payload.get('versao', VERSAO)}` · hash do payload `{payload['hash_payload']}`", "",
              "| campo | valor | origem no payload |", "|---|---|---|"]
    for f in c["campos"]:
        v = json.dumps(f["valor"], ensure_ascii=False, default=str)
        linhas.append(f"| {f['rotulo']} | {v[:300].replace('|', '/')} | `{f['caminho']}` |")
    soc = payload.get("social") or {}
    for reg, m in (soc.get("regioes") or {}).items():
        if m.get("estado") == "medida":
            linhas += ["", f"## Social — {reg}", f"Unidade: {m['unidade']}; setores: {m['setores_intersectados']}; {m['denominadores']}."]
    sb = payload.get("sar_bruto") or {}
    if sb.get("estado") == "presente_ligado":
        cena = (sb.get("cenas") or [{}])[0]
        linhas += ["", "## SAR bruto (dado, nao percepcao)",
                   f"{sb['colecao']} adquirido em {', '.join(sb.get('datas_de_aquisicao_utc') or []) or 'data nao registrada'}"
                   f"{' (MOSAICO DE MAIS DE UM DIA)' if sb.get('mosaico_multidata') else ''}; data pedida {sb.get('data_pedida_no_manifesto')} "
                   f"({cena.get('quando_utc')}, {cena.get('passagem')}, orbita {cena.get('orbita_relativa')}), "
                   f"{sb['unidade_radiometrica']}, {sb.get('resolucao_m')} m, {sb.get('crs')}; cobertura valida {sb['fracao_valida_da_aoi']}; "
                   f"papel temporal: {sb['papel_temporal']['papel']} (evento {sb['papel_temporal']['data_do_evento']})."]
    ro = payload.get("referencia_observada") or {}
    if ro.get("estado") == "presente_ligado":
        linhas += ["", "## Referencia observada independente (CEMS)",
                   *[f"- {x['id']} ({x['tipo_produto']}, {x['data']}, {x['sensor']}): {x['km2_positivo_na_consulta']} km2 positivos de "
                     f"{x['km2_observada_na_consulta']} km2 observados na AOI" for x in ro["produtos"]]]
    linhas += ["", "## Limitacoes", *[f"- {x}" for x in (payload.get("limitacoes") or [])], "", f"Nao e: {payload.get('nao_e')}."]
    return "\n".join(linhas) + "\n"


def verificar_projecao(payload: dict, cartao_: dict, relatorio: str) -> dict:
    """Reextrai cada campo do cartão do payload e confere hash: o artefato não recalcula nem diverge."""
    recalculado = sha_json({k: v for k, v in payload.items() if k != "hash_payload"})
    divergentes = [f["caminho"] for f in cartao_["campos"] if _pegar(payload, f["caminho"]) != f["valor"]]
    return {"hash_confere": recalculado == payload["hash_payload"] == cartao_["hash_payload"],
            "hash_no_relatorio": payload["hash_payload"] in relatorio, "campos_divergentes": divergentes,
            "ok": recalculado == payload["hash_payload"] == cartao_["hash_payload"] and not divergentes and payload["hash_payload"] in relatorio}


def criar_router():
    from fastapi import APIRouter
    from pydantic import BaseModel, ConfigDict, Field

    class PedidoIntegrado(BaseModel):
        model_config = ConfigDict(extra="forbid")
        aoi: dict[str, float]
        pergunta: str | None = None
        autoridade_fisica: str = "auto"
        cenario_de_chuva: str | None = None
        lado_da_grade: int = Field(default=5,ge=1,le=50)
        max_pontos_pesquisa: int = Field(default=200,ge=1,le=5000)
        usar_historico: bool = True
        uso: str | None = None
        evento_data: str | None = None
        aquisicao_data: str | None = None
        sar_unidade: str | None = None
        intervalo: list[str] | None = None

    router = APIRouter(tags=["resposta integrada"])

    def _rota(versao):
        def consulta(req):
            p = responder(req.model_dump(exclude_none=True), versao=versao)
            c = cartao(p)
            md = relatorio_md(p)
            return {"payload": p, "cartao": c, "relatorio_md": md, "verificacao": verificar_projecao(p, c, md)}
        consulta.__annotations__["req"] = PedidoIntegrado
        return consulta
    router.post("/integrada/v1/consulta")(_rota("v1"))
    router.post("/integrada/v2/consulta")(_rota("v2"))
    @router.get("/integrada/v2/evidencias")
    def evidencias():
        from evidencias_escopo import consultar
        return consultar()
    return router
