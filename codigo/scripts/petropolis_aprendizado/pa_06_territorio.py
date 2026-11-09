"""Resposta territorial candidata para as 42 FCU de Petropolis.

Junta, sem somar em indice: agua (modelo servido svc-03, congelado), encosta
(observado 2022, relatos historicos, contexto tecnico 2025 e inferencia antes/
depois da atualizacao local), conexao (bacia D8 a montante de cada FCU),
contexto social (Censo 2022) e o que falta observar. Cada componente guarda a
propria natureza: observado, contexto_tecnico, inferido, desconhecido, nao_avaliado.

Tambem mede o que a ferramenta passou a informar sobre as comunidades frente a
ficha anterior (estudo multiperigos v1): cobertura de informacao, nao reducao de risco.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pa_comum import (CHARTER, EPSG_PETROPOLIS, ESTUDO, OUT, SVC03_RES, SVC03_TIF, TERRENO, Etapa,
                      gravar_json, ler_json, rel, sha256)
from pa_03_referencias import ufrj_gdf, _raster, _rc

RES = OUT / "resultados"
TER = OUT / "territorio"
PESOS = OUT / "pesos"
CASOS = ["33039060003", "33039060014", "33039060021", "33039060022", "33039060052"]
TOPO = 0.8          # "alto" = quintil superior do dominio municipal; fixado no protocolo como orcamento de 20%
RAIO_CONEXAO_M = 1000.0


def _escore(pilha, coef, inter, mu, sd, F):
    X = (pilha - np.asarray([mu[c] for c in F])[:, None, None]) / np.asarray([sd[c] for c in F])[:, None, None]
    return inter + np.tensordot(np.asarray(coef), X, axes=1)


def _percentil(v, valido):
    out = np.full(v.shape, np.nan, "float32")
    x = v[valido]
    ordem = np.argsort(x, kind="mergesort")
    rk = np.empty(len(x))
    rk[ordem] = np.arange(len(x))
    out[valido] = (rk + 0.5) / len(x)
    return out


def _gravar_raster(perfil, arr, nome):
    import rasterio
    p = dict(perfil)
    p.update(count=1, dtype=arr.dtype.name, nodata=(np.nan if arr.dtype.kind == "f" else 0), compress="deflate")
    with rasterio.open(TER / nome, "w", **p) as d:
        d.write(arr, 1)
    return TER / nome


def bacias(fcu_ras, perfil):
    import whitebox
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    pts = _gravar_raster(perfil, fcu_ras.astype("int32"), "fcu_pour_points.tif")
    saida = TER / "bacia_montante_fcu.tif"
    if saida.exists():
        saida.unlink()
    wbt.watershed(str(TERRENO / "petropolis/d8_pointer_esri.tif"), str(pts), str(saida), esri_pntr=True)
    import rasterio
    with rasterio.open(saida) as r:
        b = r.read(1)
        nd = r.nodata
    b = np.where((b == nd) | (b < 0), 0, b).astype("int32")
    return b


def main() -> int:
    import geopandas as gpd
    import rasterio
    from pyproj import Transformer

    e = Etapa("pa_06_territorio")
    TER.mkdir(parents=True, exist_ok=True)
    prot = ler_json(OUT / "protocolo/protocolo_v1.json")
    F = prot["atributos"]
    pad = ler_json(PESOS / "padronizacao.json")
    mu, sd = pad["media"], pad["desvio"]
    a1 = ler_json(PESOS / "A1_log.json")
    aj = pd.read_csv(RES / "ajustes_locais.csv")
    a3 = aj[(aj.braco == "A3_log_l25") & (aj.B.astype(str) == "todos")]
    coef_a3 = a3[[f"coef_{c}" for c in F]].mean().values
    inter_a3 = float(a3.intercepto.mean())
    contr = pd.read_csv(RES / "contrastes_T1.csv")
    rt = ler_json(RES / "resultado_treino.json")
    q = pd.DataFrame(rt["D4_dominio"]["P1_P99_doador"])

    def rot(nome, B):
        x = contr[(contr.contraste == nome) & (contr.B.astype(str) == str(B))]
        return (x.iloc[0].to_dict() if len(x) else None)
    c1, c2 = rot("C1_transferencia", 0), rot("C2_aprendizado_local", "todos")
    estado_antes = "inferido_avaliado_no_episodio_2022" if c1 and c1["rotulo_regra"] == "sustentado" else "inferido_nao_validado"
    estado_depois = "inferido_avaliado_no_episodio_2022" if c2 and c2["rotulo_regra"] == "sustentado" else "inferido_nao_validado"
    recusa_hist = bool(c2 and c2["rotulo_regra"] == "contrario")
    e1 = ler_json(PESOS / "emenda1/E1_implantacao_A3_log_l25.json")
    ce1 = pd.read_csv(RES / "emenda1/contrastes_E1.csv")
    ce1_c2 = ce1[(ce1.contraste == "E1_C2") & (ce1.K.astype(str) == "todas")].iloc[0].to_dict()
    ce1_rv = ce1[(ce1.contraste == "E1_A3_vs_regra") & (ce1.K.astype(str) == "todas")].iloc[0].to_dict()
    estado_e1 = ("inferido_desenvolvimento_cv_bairro_2022" if ce1_c2["rotulo_regra"] == "sustentado"
                 else "inferido_nao_validado")

    with rasterio.open(TERRENO / "petropolis/atributos_encosta.tif") as r:
        pilha = r.read().astype("float64")
        perfil = r.profile
        transf = r.transform
        rr = r
        valido = np.isfinite(pilha).all(0)
        antes = _escore(pilha, a1["coef"], a1["intercepto"], mu, sd, F)
        depois = _escore(pilha, coef_a3, inter_a3, mu, sd, F)
        dentro = np.ones(valido.shape, bool)
        for i, c in enumerate(F):
            dentro &= (pilha[i] >= q.loc["0.01", c]) & (pilha[i] <= q.loc["0.99", c]) if "0.01" in q.index else \
                      (pilha[i] >= q.loc[0.01, c]) & (pilha[i] <= q.loc[0.99, c])
        dentro &= valido
        p_antes, p_depois = _percentil(antes, valido), _percentil(depois, valido)
        p_e1 = _percentil(_escore(pilha, e1["coef"], e1["intercepto"], mu, sd, F), valido)
        p_declive = _percentil(pilha[0], valido)
        fcu = gpd.read_file(ESTUDO / "fcu_42_geometrias.geojson").to_crs(EPSG_PETROPOLIS)
        fcu["k"] = np.arange(1, len(fcu) + 1)
        fcu_ras = _raster(rr, list(fcu.geometry), valores=list(fcu.k))
    arq_a = _gravar_raster(perfil, p_antes, "encosta_percentil_antes_A1.tif")
    arq_d = _gravar_raster(perfil, p_depois, "encosta_percentil_depois_A3_todos.tif")
    arq_e1 = _gravar_raster(perfil, p_e1, "encosta_percentil_E1_cicatrizes2022.tif")
    bac = bacias(fcu_ras, perfil)

    # charter 2022 e relatos historicos
    ch = gpd.read_file(CHARTER).to_crs(EPSG_PETROPOLIS).reset_index(drop=True)
    ch_d = _raster(rr, list(ch[ch.tipoevento.eq("Deslizamento")].geometry), all_touched=True).astype(bool)
    ch_f = _raster(rr, list(ch[ch.tipoevento.eq("Fluxo de detritos")].geometry), all_touched=True).astype(bool)
    lab_ch = _raster(rr, list(ch.geometry), valores=list(np.arange(1, len(ch) + 1)), all_touched=True)
    uf = ufrj_gdf()
    uf = uf[uf.localizado & uf.tipo_de_ev.eq("deslizamento")]
    fichas = {f["CD_FCU"]: f for f in json.loads((ESTUDO / "fichas_42_comunidades.json").read_text(encoding="utf-8"))}
    ev = pd.read_csv(ESTUDO / "evidencia_fcu_por_processo.csv", dtype={"CD_FCU": str})
    zon = pd.read_csv(ESTUDO / "fcu_zoneamento_municipal_2025.csv", dtype={"CD_FCU": str})
    st = pd.read_csv(ESTUDO / "setores_petropolis_metadados.csv", dtype={"CD_FCU": str, "CD_SETOR": str})
    st = st[st.CD_FCU != "."]
    svc = ler_json(SVC03_RES)
    svc_p = [x for x in svc["regioes"] if x["regiao"] == "petropolis"][0]

    with rasterio.open(SVC03_TIF) as ra:
        agua_arr = ra.read(1)
        tr = Transformer.from_crs(EPSG_PETROPOLIS, ra.crs, always_xy=True)
        ta = ra.transform

    linhas, cands = [], []
    for _, f in fcu.iterrows():
        cod = f.CD_FCU
        m = fcu_ras == f.k
        mv = m & valido
        n = int(m.sum())
        rows, cols = np.nonzero(mv)
        xs, ys = rasterio.transform.xy(transf, rows, cols)
        ax, ay = tr.transform(np.asarray(xs), np.asarray(ys))
        ac = np.floor((ax - ta.c) / ta.a).astype(int)
        al = np.floor((ay - ta.f) / ta.e).astype(int)
        okk = (al >= 0) & (al < agua_arr.shape[0]) & (ac >= 0) & (ac < agua_arr.shape[1])
        agua_v = np.full(len(rows), np.nan)
        agua_v[okk] = agua_arr[al[okk], ac[okk]]
        geom = f.geometry
        h_in = uf[uf.within(geom)]
        h_100 = uf[uf.within(geom.buffer(100))]
        up_total = (bac == f.k) & ~m & valido
        # escala de encosta: so o que drena para a FCU a ate 1 km dela; a bacia inteira
        # de um fundo de vale (ex.: Pedro do Rio no Piabanha) e escala de rio, nao de encosta
        from scipy.ndimage import distance_transform_edt
        l0, l1 = max(rows.min() - 40, 0) if len(rows) else 0, min(rows.max() + 41, m.shape[0]) if len(rows) else 0
        c0, c1 = max(cols.min() - 40, 0) if len(cols) else 0, min(cols.max() + 41, m.shape[1]) if len(cols) else 0
        perto = np.zeros_like(m)
        if len(rows):
            perto[l0:l1, c0:c1] = distance_transform_edt(~m[l0:l1, c0:c1]) * 30.0 <= RAIO_CONEXAO_M
        up = up_total & perto
        up_alto = up & (p_e1 >= TOPO)
        ch_up = set(np.unique(lab_ch[up & (ch_d | ch_f)])) - {0}
        evf = ev[ev.CD_FCU.eq(cod)]
        zf = zon[zon.CD_FCU.eq(cod)]
        sf = st[st.CD_FCU.eq(cod)]
        renda = sf.dropna(subset=["renda_resp"])
        ficha = fichas[cod]
        desl = evf[evf.processo.eq("Deslizamento")]
        flux = evf[evf.processo.eq("Fluxo de detritos")]
        linha = {
            "CD_FCU": cod, "NM_FCU": f.NM_FCU, "area_m2": float(geom.area), "celulas_30m": n,
            "populacao_censo2022": ficha["populacao_total_fcu_censo2022"],
            "domicilios_censo2022": ficha["domicilios_total_fcu_censo2022"],
            "renda_resp_media_ponderada": (float(np.average(renda.renda_resp, weights=renda.pessoas)) if len(renda) and renda.pessoas.sum() > 0 else None),
            "setores": int(len(sf)), "setores_com_renda": int(len(renda)),
            "charter2022_deslizamentos": int(desl.poligonos.sum()) if len(desl) else 0,
            "charter2022_area_desliz_m2": float(desl.area_uniao_m2.sum()) if len(desl) else 0.0,
            "charter2022_area_desliz_erosao10m_m2": float(desl.area_erosao_interna_10m_m2.sum()) if len(desl) else 0.0,
            "charter2022_area_desliz_erosao30m_m2": float(desl.area_erosao_interna_30m_m2.sum()) if len(desl) else 0.0,
            "charter2022_fluxos_detritos": int(flux.poligonos.sum()) if len(flux) else 0,
            "ufrj_relatos_dentro": int(len(h_in)), "ufrj_relatos_100m": int(len(h_100)),
            "ufrj_anos": (f"{h_100.data_dt.min():%Y}-{h_100.data_dt.max():%Y}" if len(h_100) and h_100.data_dt.notna().any() else None),
            "zoneamento2025_fracao_area": float(zf.fracao_area_fcu_em_zoneamento.iloc[0]) if len(zf) else 0.0,
            "zoneamento2025_classes": (zf.classes_presentes.iloc[0] if len(zf) else "[]"),
            "agua_escore_medio": float(np.nanmean(agua_v)) if np.isfinite(agua_v).any() else None,
            "agua_escore_max": float(np.nanmax(agua_v)) if np.isfinite(agua_v).any() else None,
            "agua_cobertura": float(np.isfinite(agua_v).mean()) if len(agua_v) else 0.0,
            "declive_percentil_mediano": float(np.nanmedian(p_declive[mv])) if mv.any() else None,
            "encosta_antes_pct_mediano": float(np.nanmedian(p_antes[mv])) if mv.any() else None,
            "encosta_antes_frac_quintil_sup": float(np.mean(p_antes[mv] >= TOPO)) if mv.any() else None,
            "encosta_depois_pct_mediano": float(np.nanmedian(p_depois[mv])) if mv.any() else None,
            "encosta_depois_frac_quintil_sup": float(np.mean(p_depois[mv] >= TOPO)) if mv.any() else None,
            "encosta_E1_pct_mediano": float(np.nanmedian(p_e1[mv])) if mv.any() else None,
            "encosta_E1_frac_quintil_sup": float(np.mean(p_e1[mv] >= TOPO)) if mv.any() else None,
            "dominio_frac_dentro_doador": float(dentro[mv].mean()) if mv.any() else 0.0,
            "montante_bacia_total_km2": float(up_total.sum() * 900 / 1e6),
            "montante_area_km2": float(up.sum() * 900 / 1e6),
            "montante_quintil_sup_km2": float(up_alto.sum() * 900 / 1e6),
            "montante_feicoes_charter2022": len(ch_up),
        }
        linha["encosta_delta_pct_mediano"] = (linha["encosta_depois_pct_mediano"] - linha["encosta_antes_pct_mediano"]
                                              if mv.any() else None)
        linhas.append(linha)

        obs = []
        if linha["charter2022_deslizamentos"]:
            # cicatrizes sao estreitas e alongadas: 30 m de erosao apaga quase todas; o criterio de
            # borda e a erosao de 10 m (erro posicional plausivel do mapeamento); 30 m fica so relatado
            e10, e30 = linha["charter2022_area_desliz_erosao10m_m2"], linha["charter2022_area_desliz_erosao30m_m2"]
            borda = e10 < 10.0
            obs.append({"tipo": "cicatriz_2022_INPE_Charter", "estado": "observado",
                        "poligonos": linha["charter2022_deslizamentos"], "area_m2": round(linha["charter2022_area_desliz_m2"], 1),
                        "area_apos_erosao_10m_m2": round(e10, 1), "area_apos_erosao_30m_m2": round(e30, 1),
                        "sensibilidade_borda": (f"intersecao quase some com erosao interna de 10 m ({e10:.0f} m2): apoio fraco, possivel efeito de borda"
                                                if borda else f"persiste a erosao interna de 10 m ({e10:.0f} m2)")})
        if linha["charter2022_fluxos_detritos"]:
            obs.append({"tipo": "fluxo_detritos_2022_INPE_Charter", "estado": "observado",
                        "poligonos": linha["charter2022_fluxos_detritos"]})
        if linha["ufrj_relatos_100m"]:
            obs.append({"tipo": "relato_historico_UFRJ_GEOCART", "estado": "observado",
                        "relatos_dentro": linha["ufrj_relatos_dentro"], "relatos_ate_100m": linha["ufrj_relatos_100m"],
                        "periodo": linha["ufrj_anos"],
                        "natureza": "relato jornalistico de dano geocodificado por endereco; posicao aproximada"})
        desconhecido = []
        if not obs:
            desconhecido.append("nenhum registro observado de encosta nas fontes disponiveis; ausencia de registro nao e ausencia do fenomeno")
        if linha["dominio_frac_dentro_doador"] < 0.8:
            desconhecido.append(f"{1 - linha['dominio_frac_dentro_doador']:.0%} das celulas tem algum atributo fora do P1-P99 do doador: inferencia em extrapolacao")
        necessaria = ["referencia local de inundacao com data, extensao e area inspecionada (agua sem referencia local)"]
        if not obs:
            necessaria.append("inspecao de encosta com registro de area vistoriada (presencas e ausencias observadas)")
        else:
            necessaria.append("delimitar ruptura, percurso e deposito e a data de cada feicao")
        if linha["montante_quintil_sup_km2"] > 0.05:
            necessaria.append("mapear percursos a montante e cronologia entre ruptura e chegada a comunidade")
        if linha["setores_com_renda"] < linha["setores"]:
            necessaria.append("renda ausente em parte dos setores")
        cands.append({
            "CD_FCU": cod, "NM_FCU": f.NM_FCU,
            "agua": {"estado": "inferido", "maturidade": svc_p["maturidade"], "modelo": svc_p["modelo"],
                     "escore_medio": linha["agua_escore_medio"], "escore_max": linha["agua_escore_max"],
                     "cobertura": linha["agua_cobertura"], "referencia_local": "ausente",
                     "nota": "escore de agua baixo NAO autoriza declarar seguranca contra movimentos de massa"},
            "encosta": {
                "observado": obs,
                "contexto_tecnico": {"estado": "contexto_tecnico", "fonte": "SIG municipal areas_de_risco_2025_def_civil",
                                     "fracao_area": linha["zoneamento2025_fracao_area"], "classes": linha["zoneamento2025_classes"],
                                     "nota": "zoneamento posterior a 2022; nao e ocorrencia observada"},
                "inferido": {"antes_transferido_A1": {"estado": estado_antes, "percentil_mediano": linha["encosta_antes_pct_mediano"],
                                                      "fracao_quintil_superior": linha["encosta_antes_frac_quintil_sup"]},
                             "atualizado_relatos_historicos_A3": {
                                 "estado": estado_depois, "recusado_pela_avaliacao": recusa_hist,
                                 "percentil_mediano": linha["encosta_depois_pct_mediano"],
                                 "fracao_quintil_superior": linha["encosta_depois_frac_quintil_sup"],
                                 "motivo": ("atualizacao piorou a ordenacao das cicatrizes de 2022 (C2 contrario): relatos "
                                            "marcam o receptor do dano, nao a ruptura; a ferramenta nao usa este estado")
                                 if recusa_hist else None},
                             "atualizado_cicatrizes_2022_E1": {
                                 "estado": estado_e1, "percentil_mediano": linha["encosta_E1_pct_mediano"],
                                 "fracao_quintil_superior": linha["encosta_E1_frac_quintil_sup"],
                                 "avaliacao": "emenda 1, validacao cruzada por bairro dentro do episodio de 2022 (desenvolvimento)",
                                 "E1_C2_delta": ce1_c2["delta"], "E1_C2_rotulo": ce1_c2["rotulo_regra"],
                                 "E1_vs_regra_delta": ce1_rv["delta"], "E1_vs_regra_rotulo": ce1_rv["rotulo_regra"]},
                             "dominio_frac_dentro_doador": linha["dominio_frac_dentro_doador"],
                             "natureza": "ordenacao relativa ao municipio por terreno antecedente; nao probabilidade nem previsao de ruptura"},
                "desconhecido": desconhecido},
            "conexao": {"estado": "inferido",
                        "metodo": "celulas que drenam para a FCU por D8 (FABDEM 30 m) a ate 1 km dela; quintil superior pelo modelo E1",
                        "area_montante_km2": round(linha["montante_area_km2"], 3),
                        "bacia_montante_total_km2": round(linha["montante_bacia_total_km2"], 3),
                        "area_montante_quintil_superior_encosta_km2": round(linha["montante_quintil_sup_km2"], 3),
                        "feicoes_charter2022_a_montante": linha["montante_feicoes_charter2022"],
                        "nota": "percurso topografico a investigar; nao demonstra volume, velocidade, alcance ou cascata"},
            "contexto_social": {"estado": "observado", "populacao_censo2022": linha["populacao_censo2022"],
                                "domicilios_censo2022": linha["domicilios_censo2022"],
                                "renda_resp_media_ponderada": linha["renda_resp_media_ponderada"],
                                "cobertura_renda": f"{linha['setores_com_renda']}/{linha['setores']} setores",
                                "nota": "universo censitario da comunidade em 2022; nao pessoas atingidas"},
            "fontes": ["INPE/Charter 0864", "UFRJ/GEOCART eventos", "SIG municipal 2025", "svc-03 fluvial_serra",
                       "FABDEM V1-2", "Dille et al. 2026 (doador)", "IBGE Censo 2022 FCU"],
            "limites": ["teste de encostas em um episodio (fev/2022) e no suporte central de 11,9 km2",
                        "fundo de teste nao rotulado: sem especificidade ou calibracao fisica",
                        "agua: transferencia sem referencia local", "cascata nao demonstrada"],
            "observacao_necessaria": necessaria,
        })

    tab = pd.DataFrame(linhas)
    tab.to_csv(TER / "tabela_42_fcu.csv", index=False, encoding="utf-8-sig")
    gravar_json(TER / "candidata_resposta_territorial.json",
                {"contrato": ["agua", "encosta", "conexao", "contexto_social", "fontes", "limites", "observacao_necessaria"],
                 "estados": ["observado", "inferido", "contexto_tecnico", "desconhecido", "nao_avaliado"],
                 "estado_encosta_antes": estado_antes, "estado_encosta_depois": estado_depois,
                 "atualizacao_historica_recusada": recusa_hist, "estado_encosta_E1": estado_e1,
                 "e1_c2": ce1_c2, "e1_vs_regra": ce1_rv,
                 "regra_estado": prot["resposta_territorial"]["estado"],
                 "c1": c1, "c2_todos": c2, "comunidades": cands})

    # comparacao com a ficha anterior: informacao disponivel por comunidade
    antes_info = {"encosta_observada": int(sum(bool(fichas[c]["processos_com_sobreposicao"]) for c in fichas)),
                  "contexto_tecnico": int(sum(fichas[c].get("contexto_municipal_2025", {}).get("feicoes_intersectadas", 0) > 0 for c in fichas)),
                  "encosta_inferida": 0, "conexao": 0, "agua_escore_explicito": 0, "observacao_necessaria_especifica": 0}
    depois_info = {"encosta_observada": int(((tab.charter2022_deslizamentos > 0) | (tab.ufrj_relatos_100m > 0)).sum()),
                   "encosta_observada_so_por_relato_historico": int(((tab.charter2022_deslizamentos == 0) & (tab.ufrj_relatos_100m > 0)).sum()),
                   "contexto_tecnico": int((tab.zoneamento2025_fracao_area > 0).sum()),
                   "encosta_inferida_dentro_dominio_80pct": int((tab.dominio_frac_dentro_doador >= 0.8).sum()),
                   "encosta_inferida_total": int(tab.encosta_depois_pct_mediano.notna().sum()),
                   "conexao_com_montante_quintil_sup": int((tab.montante_quintil_sup_km2 > 0).sum()),
                   "agua_escore_explicito": int(tab.agua_escore_medio.notna().sum()),
                   "sem_nenhum_registro_observado": int(((tab.charter2022_deslizamentos == 0) & (tab.ufrj_relatos_100m == 0)).sum())}
    pop = {"populacao_em_fcu_com_encosta_observada_antes": int(tab[tab.charter2022_deslizamentos > 0].populacao_censo2022.sum()),
           "populacao_em_fcu_com_encosta_observada_depois": int(tab[(tab.charter2022_deslizamentos > 0) | (tab.ufrj_relatos_100m > 0)].populacao_censo2022.sum()),
           "populacao_universo": int(tab.populacao_censo2022.sum())}
    gravar_json(TER / "informacao_antes_depois.json", {"antes_ficha_estudo_v1": antes_info, "depois_v1": depois_info,
                                                       "populacao_universo_censitario": pop,
                                                       "nota": "cobertura de informacao por comunidade; nao reducao de risco nem pessoas atingidas"})
    e.entrada(RES / "ajustes_locais.csv", PESOS / "A1_log.json", ESTUDO / "fcu_42_geometrias.geojson", SVC03_TIF)
    e.saida(*TER.glob("*"))
    e.fechar(estado_antes=estado_antes, estado_depois=estado_depois, estado_e1=estado_e1,
             rasters=[rel(arq_a), rel(arq_d), rel(arq_e1)], sha_tabela=sha256(TER / "tabela_42_fcu.csv"))
    print(tab[tab.CD_FCU.isin(CASOS)].T.to_string())
    print(json.dumps({"antes": antes_info, "depois": depois_info, "pop": pop}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
