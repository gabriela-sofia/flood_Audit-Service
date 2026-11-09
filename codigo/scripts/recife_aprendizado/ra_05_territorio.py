"""Resposta territorial candidata para as 295 FCU de Recife e casos (3 pilotos + Soledade).

Contrato: terreno, captacao, rede, saida, agua_observada, inferencia_aprendida, contexto_social,
fontes, limites, proxima_observacao. Cada item guarda sua natureza: observado, inferido,
contexto_tecnico, desconhecido, nao_avaliado. Nenhum escore e somado num indice.

Denominador: 295 FCU. Coberta = tem ao menos uma rua da populacao do experimento; avaliada =
tem rua de bairro de TESTE com relato (onde o resultado fora do ajuste pode ser lido).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ra_comum import EPSG, ESTUDO, OUT, ROOT, Etapa, gravar_json, ler_json, sha256

RES = OUT / "resultados"
TER = OUT / "territorio"
PILOTOS = {"26116060588": "Chico Mendes - Linha Nova", "26116060627": "Encanta Moça",
           "26116060831": "Estrada dos Remédios - Rozenblit I"}
UNIV = ROOT / "local_runs/transferencia-vulnerabilizados/v1/universo/fcu_nacional.csv"


def pct_cidade(v, base):
    b = np.sort(base[np.isfinite(base)])
    return (np.searchsorted(b, v, "left") + np.searchsorted(b, v, "right")) / 2 / len(b)


def main() -> int:
    import geopandas as gpd
    e = Etapa("ra_05_territorio")
    TER.mkdir(parents=True, exist_ok=True)
    prot = ler_json(OUT / "protocolo/protocolo_v1.json")
    t = pd.read_parquet(OUT / "atributos/atributos_objetos.parquet")
    p = pd.read_parquet(RES / "pred_ruas.parquet")
    c = pd.read_csv(RES / "contrastes.csv")
    obj = gpd.read_file(OUT / "referencias/objetos_rua_bairro.gpkg")
    rel_ = pd.read_parquet(OUT / "referencias/relatos_sedec_geocodificados.parquet")
    fcu = gpd.read_file(ESTUDO / "fcu_recife_epsg31985.geojson").to_crs(EPSG)
    cob = pd.read_csv(ESTUDO / "fcu_cobertura_drenagem.csv", dtype={"CD_FCU": str})
    univ = pd.read_csv(UNIV, dtype={"CD_FCU": str, "CD_MUN": str})
    univ = univ[univ.CD_MUN.eq("2611606")][["CD_FCU", "pessoas", "dom_ocupados"]]

    def lab(nome, B):
        x = c[(c.contraste == nome) & (c.B.astype(str) == str(B))]
        return x.iloc[0].to_dict() if len(x) else None
    c1, c5, c6 = lab("C1_transferencia", 0), lab("C5_contexto_urbano", "todas"), lab("C6_PE3D", "todas_pe3d")
    c2 = lab("C2_aprendizado_local", "todas")
    est_antes = "inferido_avaliado_fora_do_ajuste" if c1["rotulo_regra"] == "sustentado" else "inferido_nao_validado"
    est_depois = "inferido_avaliado_fora_do_ajuste" if c5["rotulo_regra"] == "sustentado" else "inferido_nao_validado"
    # emendas 1-2 (desenvolvimento pos-teste): o ganho urbano e de PE3D nao sobrevive ao controle de comprimento
    ce1 = pd.read_csv(RES / "emenda1/contrastes_E1.csv").set_index("contraste")
    ce2 = pd.read_csv(RES / "emenda2/contrastes_E2.csv").set_index("contraste")
    alerta = {"confundidor": "comprimento da rua (oportunidade de relato por endereco)",
              "urbano_controlado": ce1.loc["E1c_urbano_controlado_comprimento", ["delta", "ic95_lo", "ic95_hi", "rotulo_regra"]].to_dict(),
              "pe3d_controlado": ce2.loc["E2a_PE3D_controlado", ["delta", "ic95_lo", "ic95_hi", "rotulo_regra"]].to_dict(),
              "origem": "emendas 1 e 2, desenvolvimento pos-teste"}
    if alerta["urbano_controlado"]["rotulo_regra"] != "sustentado":
        est_depois = "inferido_ganho_nao_sustentado_sob_controle_de_comprimento"
    est_pe3d = ("inferido_ganho_nao_sustentado_sob_controle_de_comprimento"
                if alerta["pe3d_controlado"]["rotulo_regra"] != "sustentado" else "inferido_avaliado_fora_do_ajuste_subconjunto")

    d = t.merge(p[["id_obj", "A1_Q", "A4_l25__Btodas__s0", "A5_l25__Btodas__s0"]], on="id_obj")
    d["pct_antes"] = pct_cidade(d.A1_Q.values, d.A1_Q.values)
    d["pct_depois"] = pct_cidade(d["A4_l25__Btodas__s0"].values, d["A4_l25__Btodas__s0"].values)
    pe = d.tem_pe3d.values
    d["pct_pe3d"] = np.nan
    d.loc[pe, "pct_pe3d"] = pct_cidade(d.loc[pe, "A5_l25__Btodas__s0"].values, d.loc[pe, "A5_l25__Btodas__s0"].values)
    g = obj[["id_obj", "geometry"]].merge(d, on="id_obj")
    j = gpd.sjoin(g, fcu[["CD_FCU", "NM_FCU", "geometry"]], predicate="intersects")

    linhas, cands = [], []
    for _, f in fcu.iterrows():
        sub = j[j.CD_FCU == f.CD_FCU]
        cb = cob[cob.CD_FCU == f.CD_FCU]
        un = univ[univ.CD_FCU == f.CD_FCU]
        avaliada = bool(len(sub) and ((sub.papel == "teste") & (sub.y == 1)).any())
        lin = {"CD_FCU": f.CD_FCU, "NM_FCU": f.NM_FCU,
               "pessoas_censo2022": int(un.pessoas.iloc[0]) if len(un) else None,
               "domicilios_censo2022": int(un.dom_ocupados.iloc[0]) if len(un) else None,
               "fracao_moldura_pe3d": float(cb.fracao_moldura_pe3d.iloc[0]) if len(cb) else None,
               "captacoes_BL_CG_cadastro": int(cb.captacoes_BL_CG.iloc[0]) if len(cb) else None,
               "ruas_na_populacao": int(len(sub)), "ruas_com_relato": int(sub.y.sum()) if len(sub) else 0,
               "relatos_geocodificados": int(sub.n_relatos.sum()) if len(sub) else 0,
               "papel": ",".join(sorted(sub.papel.unique())) if len(sub) else None,
               "coberta": bool(len(sub)), "avaliada_fora_do_ajuste": avaliada}
        if len(sub):
            lin.update({"q_hand_medio": float(np.expm1(sub.f_hand).mean()),
                        "entradas_por_100m_medio": float(sub.dren_entradas_100m.mean()),
                        "dist_rede_m_media": float(sub.dren_dist_rede_m.mean()),
                        "dist_canal_m_min": float(sub.saida_dist_canal_m.min()),
                        "pe3d_rel50_medio": float(sub.pe3d_rel50.mean()) if sub.tem_pe3d.any() else None,
                        "pe3d_depr_max": float(sub.pe3d_depr_max.max()) if sub.tem_pe3d.any() else None,
                        "pct_antes_mediano": float(sub.pct_antes.median()), "pct_depois_mediano": float(sub.pct_depois.median()),
                        "pct_pe3d_mediano": float(sub.pct_pe3d.median()) if sub.tem_pe3d.any() else None})
            if avaliada:
                pos = sub[(sub.y == 1) & (sub.papel == "teste")]
                lin["relato_teste_pct_antes_medio"] = float(pos.pct_antes.mean())
                lin["relato_teste_pct_depois_medio"] = float(pos.pct_depois.mean())
                lin["relato_teste_no_quintil_sup_antes"] = int((pos.pct_antes >= 0.8).sum())
                lin["relato_teste_no_quintil_sup_depois"] = int((pos.pct_depois >= 0.8).sum())
        linhas.append(lin)
        prox = []
        if not len(sub):
            prox.append("fora da populacao do experimento: bairro sem relato geocodificado ate a rua")
        if lin["ruas_com_relato"] == 0:
            prox.append("relato de agua datado e localizado na rua (ausencia de chamado nao e rua seca)")
        prox += ["condicao de saida observada (nivel de canal/mare) no episodio", "estado operacional da captacao e da rede (cadastro nao informa)"]
        if not (lin["fracao_moldura_pe3d"] or 0) >= 0.95:
            prox.append("terreno PE3D nativo nesta area (fora das 8 folhas adquiridas)")
        cands.append({
            "CD_FCU": f.CD_FCU, "NM_FCU": f.NM_FCU,
            "terreno": {"estado": "observado" if len(sub) else "nao_avaliado",
                        "hand_30m_medio_m": lin.get("q_hand_medio"), "pe3d_rua_menos_entorno_50m_m": lin.get("pe3d_rel50_medio"),
                        "pe3d_depressao_max_m": lin.get("pe3d_depr_max"),
                        "nota": "PE3D 1 m agregado a 2 m; datum vertical nao reconciliado; cota nao e lamina d'agua"},
            "captacao": {"estado": "contexto_tecnico", "entradas_BL_CG_cadastro": lin["captacoes_BL_CG_cadastro"],
                         "entradas_por_100m_de_rua": lin.get("entradas_por_100m_medio"),
                         "nota": "existencia cadastral; nao funcionamento, limpeza ou capacidade"},
            "rede": {"estado": "contexto_tecnico", "dist_rede_m_media": lin.get("dist_rede_m_media"),
                     "nota": "grafo geometrico candidato; cotas e secoes insuficientes: capacidade desconhecida"},
            "saida": {"estado": "contexto_tecnico", "dist_canal_ou_rio_m": lin.get("dist_canal_m_min"),
                      "nota": "proximidade topografica a canal/rio; nivel e mare nao observados"},
            "agua_observada": {"estado": "observado" if lin["ruas_com_relato"] else "desconhecido",
                               "ruas_com_relato_sedec": lin["ruas_com_relato"], "relatos": lin["relatos_geocodificados"],
                               "nota": "registro administrativo de agua geocodificado ate a rua; ausencia de relato nao e ausencia de alagamento"},
            "inferencia_aprendida": {"antes_Q_transferido": {"estado": est_antes, "percentil_mediano_ruas": lin.get("pct_antes_mediano")},
                                     "depois_Q_mais_drenagem_saida": {"estado": est_depois, "percentil_mediano_ruas": lin.get("pct_depois_mediano")},
                                     "com_PE3D": {"estado": est_pe3d, "percentil_mediano_ruas": lin.get("pct_pe3d_mediano")},
                                     "alerta_comprimento": alerta,
                                     "avaliada_fora_do_ajuste": avaliada,
                                     "natureza": "priorizacao de ruas com registro de alagamento; nao probabilidade nem profundidade"},
            "contexto_social": {"estado": "observado", "pessoas_censo2022": lin["pessoas_censo2022"],
                                "nota": "universo censitario da comunidade; nao pessoas atingidas"},
            "fontes": ["SEDEC registros 2014-2024", "logradouros/bairros ESIG", "PE3D MDT 1 m", "EMLURB micro/macrodrenagem",
                       "nucleo Q (v9/v10)", "IBGE Censo 2022 FCU"],
            "limites": ["presenca-fundo; sem negativos observados", "relatos concentrados em 2014-2016 e 2023-2024; episodio de 2022 sem endereco",
                        "teste espacial por bairro; rede atravessa bairros", "chuva e mare nao avaliadas"],
            "proxima_observacao": prox})
    tab = pd.DataFrame(linhas)
    tab.to_csv(TER / "tabela_295_fcu.csv", index=False, encoding="utf-8-sig")
    gravar_json(TER / "candidata_resposta_territorial.json",
                {"contrato": ["terreno", "captacao", "rede", "saida", "agua_observada", "inferencia_aprendida", "contexto_social",
                              "fontes", "limites", "proxima_observacao"],
                 "estados": ["observado", "inferido", "contexto_tecnico", "desconhecido", "nao_avaliado"],
                 "c1": c1, "c2": c2, "c5": c5, "c6": c6, "alerta_comprimento": alerta,
                 "estado_antes": est_antes, "estado_depois": est_depois, "estado_pe3d": est_pe3d, "comunidades": cands})

    cobertura = {"fcu_total": int(len(tab)), "fcu_cobertas": int(tab.coberta.sum()),
                 "fcu_com_rua_com_relato": int((tab.ruas_com_relato > 0).sum()),
                 "fcu_avaliadas_fora_do_ajuste": int(tab.avaliada_fora_do_ajuste.sum()),
                 "fcu_moldura_pe3d_95": int((tab.fracao_moldura_pe3d.fillna(0) >= 0.95).sum()),
                 "pessoas_total": int(tab.pessoas_censo2022.fillna(0).sum()),
                 "pessoas_em_fcu_cobertas": int(tab[tab.coberta].pessoas_censo2022.fillna(0).sum()),
                 "pessoas_em_fcu_avaliadas": int(tab[tab.avaliada_fora_do_ajuste].pessoas_censo2022.fillna(0).sum()),
                 "nota": "cobertura de informacao e de avaliacao; nao reducao de risco"}
    gravar_json(TER / "cobertura_295.json", cobertura)

    # casos: pilotos e Soledade
    sol = obj[obj.NLOGRACONC.str.contains("JOAO DE BARROS|JOAQUIM FELIPE", na=False)]
    casos = {"pilotos": tab[tab.CD_FCU.isin(PILOTOS)].to_dict("records")}
    sol_d = d[d.id_obj.isin(sol.id_obj)]
    casos["soledade"] = {"ruas_encontradas_na_populacao": sol_d[["id_obj", "NLOGRACONC", "EBAIRRNOME", "papel", "y", "n_relatos",
                                                                "pct_antes", "pct_depois", "pct_pe3d", "tem_pe3d",
                                                                "saida_dist_canal_m", "pe3d_rel50", "pe3d_depr_max"]].to_dict("records"),
                         "referencia_publicada": pd.read_csv(ESTUDO / "referencia_publicada_soledade.csv").to_dict("records"),
                         "nota": ("medidas de um ponto (2013-2014) transcritas de PDF; nao usadas no ajuste nem como rotulo; "
                                  "conferencia visual da tabela pendente")}
    gravar_json(TER / "casos.json", casos)
    e.saida(*TER.glob("*"))
    e.fechar(**cobertura)
    print(json.dumps(cobertura, ensure_ascii=False, indent=1))
    print(pd.DataFrame(casos["pilotos"]).T.to_string())
    print(json.dumps(casos["soledade"]["ruas_encontradas_na_populacao"], ensure_ascii=False, indent=0, default=str)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
