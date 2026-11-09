"""Segunda confirmacao: Manaus, cheia recorde do Rio Negro (pico 30,02 m em 16/06/2021). Evento e territorio novos.

Por que existe: na RMPA o candidato primario (doadores internacionais) perdeu para a regra HAND e a
sensibilidade planejada com doadores fluviais sul-americanos ganhou do HAND. Isso e uma hipotese nova
(H-T'), gerada pelos dados da RMPA; ela so pode ser confirmada fora deles. A RMPA nao e reaberta.

Subcomandos (ambiente entre parenteses):
  grade       (revp-susc)  pontos de 60 m nos setores da Concentracao Urbana de Manaus, terreno, classe
  protocolo   (py312)      congela H-T' e as replicacoes antes de predicao e referencia
  predicoes   (py312)      rede bloqueada; pools internacional e sul-americano; hash
  referencia  (revp-susc)  agua Sentinel-2 21/06/2021 contra aguas baixas 12/10/2020; nao le predicoes
  avaliacao   (py312)      avaliacao unica
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum_tv import OUT, RUNS, agora, bloquear_rede, congelar, gravar, ler, rel, rotulo_referencia, sha

M = OUT / "confirmacao_manaus"
AQUI = Path(__file__).resolve().parent
TILES = {"gfd_tile_S03W060": (-3.0, -2.0), "gfd_tile_S04W060": (-4.0, -3.0)}
CRS_TERRENO = "EPSG:32721"
CLASSES = ["fcu", "baixa_renda", "comparador_urbano", "urbano_sem_renda"]
S2 = {"evento": "S2A_20MRB_20210621_0_L2A", "pre": "S2B_20MRB_20201012_0_L2A", "evento_alt": "S2B_20MRB_20210626_0_L2A"}
NIVEL = 1 - 0.05 / 2

PROTOCOLO = {
    "versao": "tv-confirmacao-manaus-v1",
    "hipotese_nova": ("H-T': sem rotulo de Manaus, a logistica de terreno ajustada em doadores fluviais sul-americanos "
                      "(pool GFD/EMSR720 fora do MAIN_BAS do destino e a mais de 50 km) ordena a agua de cheia dentro de "
                      "FCU melhor que a regra HAND"),
    "origem_da_hipotese": "sensibilidade planejada 'retencao_local' da confirmacao RMPA (+0,0285); por isso so vale fora da RMPA",
    "replicacoes": {"H_T_primario_RMPA": "mesmo teste com o pool internacional (componentes CEMS fora do componente sul-americano)",
                    "H_C": "nao testavel: um unico municipio na Concentracao Urbana de Manaus (desenho por municipio da RMPA)"},
    "natureza": "evento novo (2021 nao esta no acervo) e territorio novo; referencia nunca examinada",
    "populacao": {"universo": "setores com NM_CONCURB == 'Manaus' e classes " + ",".join(CLASSES),
                  "grade": "celulas de 30 m com linha e coluna pares (60 m) nos tiles gfd_tile_S03W060/S04W060 (nucleo por latitude)",
                  "destino_de_selecao": "o municipio de Manaus (descritor sem classe)"},
    "referencia_primaria": {"evento": S2["evento"], "por_que": "primeira passagem apos o pico (16/06/2021) com >= 80% valido na cidade",
                            "pre_aguas_baixas": S2["pre"], "por_que_pre": "ultima cena com nuvem de tile < 20% entre 15/09 e 31/12/2020",
                            "agua": "MNDWI > 0 em 20 m (B03 media 2x2)", "valido_scl": [2, 4, 5, 6, 7],
                            "permanente": "agua valida na cena de aguas baixas (nao ha camada JRC nesses tiles)",
                            "estados": "rotulo_referencia() da linhagem", "ligacao": "pixel de 20 m (EPSG:32720) que contem o ponto",
                            "observacao_de_regime": "positivo inclui alagamento sazonal do pulso acima da agua baixa; nao e anomalia sobre a cheia media"},
    "sensibilidades_planejadas": {"mndwi_0_2": "limiar 0,2", "ndwi_b08": "NDWI B03/B08 > 0",
                                  "data_alternativa": S2["evento_alt"]},
    "modelo": {"igual_a": "confirmacao RMPA (preditores, estimador, K, sementes, orcamento, media das sementes)",
               "retencao": "eventos com ponto no MAIN_BAS do destino e pontos a menos de 50 km saem dos dois pools",
               "pool_internacional": "componentes primarios != 'EMSR720' (unidade = componente)",
               "pool_sulamericano": "eventos do componente 'EMSR720' (GFD sul-americano + EMSR720; unidade = evento)",
               "candidato_H_T_linha": "geral do pool sul-americano", "candidato_replicacao": "geral do pool internacional",
               "recusa_dominio": "p01-p99 do pool do candidato"},
    "criterios": {"nivel_intervalo": NIVEL, "bootstrap": {"B": 2000, "semente": 20261020, "grupo": "comunidade CD_FCU"},
                  "T1": "AUC(candidato) em FCU: limite inferior > 0,60",
                  "T2": "AUC(candidato) - AUC(HAND) em FCU: inferior > 0 (superioridade); > 0,01 ganho relevante",
                  "decisao": "sustentada se T1 e T2; nao_sustentada se superior(T1) < 0,60 ou superior(T2) < 0; senao inconclusiva",
                  "nao_testavel": "< 8 comunidades com ponto conhecido ou < 20 positivos/negativos em FCU",
                  "grupos": "FCU vs comparador_urbano, absoluto em cada grupo e diferenca (descritivo)"},
    "parada": "avaliacao unica; a RMPA permanece com seu resultado",
}


# ------------------------------------------------------------------ grade (revp-susc)
def grade():
    import rasterio
    from rasterio import features
    from rasterio.warp import transform as wtransform
    from rasterio.windows import Window, from_bounds
    from tv_geo import ler_malha
    from tv01_universo_escassez import universo_nacional
    nac = universo_nacional()
    st = ler_malha("AM").merge(nac, on="CD_SETOR", how="left")
    st = st[st.NM_CONCURB.eq("Manaus") & st.classe_territorial.isin(CLASSES)].reset_index(drop=True).to_crs(CRS_TERRENO)
    st["idx"] = np.arange(1, len(st) + 1)
    partes = []
    for tile, (la0, la1) in TILES.items():
        base = RUNS / "ter-01-cadeia-harmonizada" / tile
        with rasterio.open(base / "hand_dinf.tif") as s:
            if s.crs.to_string() != CRS_TERRENO:
                raise ValueError("CRS do terreno inesperado")
            w = from_bounds(*st.total_bounds, transform=s.transform).round_offsets().round_lengths()
            w = w.intersection(Window(0, 0, s.width, s.height))
            tr, hand, nod = s.window_transform(w), s.read(1, window=w), s.nodata
        idx = features.rasterize(zip(st.geometry, st.idx), out_shape=hand.shape, transform=tr, fill=0, dtype="int32")
        lin, col = np.indices(hand.shape)
        r0, c0 = int(w.row_off), int(w.col_off)
        par = ((lin + r0) % 2 == 0) & ((col + c0) % 2 == 0) & (idx > 0)
        xs, ys = (np.asarray(v) for v in rasterio.transform.xy(tr, lin[par], col[par]))
        lon, lat = (np.asarray(v) for v in wtransform(CRS_TERRENO, "EPSG:4326", xs, ys))
        nuc = (lat >= la0) & (lat < la1)
        df = pd.DataFrame({"tile": tile, "x": xs[nuc], "y": ys[nuc], "lon": lon[nuc], "lat": lat[nuc],
                           "idx": idx[par][nuc], "hand_m": hand[par][nuc].astype("float64")})
        df.loc[df.hand_m == nod, "hand_m"] = np.nan
        for nome, arq in [("slope_deg", "slope_deg_wbt.tif"), ("twi_dinf", "twi_dinf.tif")]:
            with rasterio.open(base / arq) as s:
                a = s.read(1, window=w)
                df[nome] = a[par][nuc].astype("float64")
                df.loc[df[nome] == s.nodata, nome] = np.nan
        partes.append(df)
    g = pd.concat(partes, ignore_index=True).merge(
        pd.DataFrame(st.drop(columns="geometry"))[["idx", "CD_SETOR", "CD_MUN", "NM_MUN", "CD_FCU", "NM_FCU",
                                                   "classe_territorial", "renda_resp", "pessoas"]], on="idx").drop(columns="idx")
    for c in ["hand_m", "slope_deg", "twi_dinf"]:
        g.loc[(g[c] < -1000) | ~np.isfinite(g[c]), c] = np.nan
    g["terreno_valido"] = g[["hand_m", "slope_deg", "twi_dinf"]].notna().all(axis=1)
    g.insert(0, "ponto_id", [f"MAO_{i:07d}" for i in range(len(g))])
    gravar(M / "grade.parquet", g)
    gravar(M / "setores_universo.csv", pd.DataFrame(st.drop(columns="geometry"))[["CD_SETOR", "CD_FCU", "classe_territorial", "pessoas"]])
    gravar(M / "grade.json", {"pontos": int(len(g)), "por_classe": g.classe_territorial.value_counts().to_dict(),
                              "terreno_valido": g.groupby("classe_territorial").terreno_valido.mean().to_dict(),
                              "comunidades_fcu_universo": int(st[st.classe_territorial.eq("fcu")].CD_FCU.nunique()),
                              "comunidades_fcu_com_ponto": int(g[g.classe_territorial.eq("fcu")].CD_FCU.nunique()),
                              "sha_grade": sha(M / "grade.parquet")})
    print(len(g), g.classe_territorial.value_counts().to_dict())


# ------------------------------------------------------------------ protocolo (py312)
def protocolo():
    codigo = {rel(p): sha(p) for p in [AQUI / "comum_tv.py", AQUI / "tv_transferencia.py", AQUI / "tv09_manaus.py"]}
    p = congelar(M / "protocolo.json", {**PROTOCOLO, "codigo": codigo, "sha_grade": sha(M / "grade.parquet")})
    print(p["quando_utc"], sha(M / "protocolo.json"))


# ------------------------------------------------------------------ predicoes (py312)
def pools(doadores, alvo):
    import tv_transferencia as tt
    d, info = tt.reter(doadores, "local", alvo.lon.to_numpy(), alvo.lat.to_numpy(), episodio="NENHUM_EVENTO_NOVO")
    out = {}
    for nome, filtro, unidade in [("internacional", d.componente.ne("EMSR720"), "componente"),
                                  ("sulamericano", d.componente.eq("EMSR720"), "evento")]:
        q = d[filtro].copy()
        q["unidade_doadora"] = q[unidade]
        c = q.groupby(["unidade_doadora", "classe"]).size().unstack(fill_value=0)
        ok = c[(c.get(0, 0) >= tt.MIN_POR_CLASSE_DOADOR) & (c.get(1, 0) >= tt.MIN_POR_CLASSE_DOADOR)].index
        out[nome] = q[q.unidade_doadora.isin(ok)].reset_index(drop=True)
        info[f"unidades_{nome}"] = int(len(ok))
    return out, info


def predicoes():
    import tv_transferencia as tt
    bloquear_rede()
    prot = ler(M / "protocolo.json")
    if (M / "referencia.parquet").exists():
        raise SystemExit("referencia ja existe")
    for arq, h in prot["codigo"].items():
        if sha(Path(__file__).resolve().parents[2] / arq) != h:
            raise SystemExit(f"codigo alterado depois do congelamento: {arq}")
    g = pd.read_parquet(M / "grade.parquet")
    alvo = g[g.terreno_valido].reset_index(drop=True)
    ps, info = pools(tt.carregar_doadores(), alvo)
    saida = alvo[["ponto_id"]].copy()
    saida["regra_hand"] = -alvo.hand_m.to_numpy()
    reg = []
    ad, x = tt.descritor(alvo), tt.matriz(alvo)
    for nome, d in ps.items():
        dom = tt.dominio(d)
        info[f"dominio_{nome}"] = dom
        saida[f"dominio_{nome}"] = tt.dentro_do_dominio(alvo, dom)
        desc = tt.descritores_doadores(d)
        acum = {b: np.zeros(len(alvo)) for b in tt.BRACOS}
        for semente in tt.SEMENTES:
            conj = tt.selecionar(desc, ad, float(alvo.lon.mean()), float(alvo.lat.mean()), semente)
            modelos, r = tt.ajustar_bracos(d, conj, semente)
            reg.append({"pool": nome, "semente": semente, "bracos": r})
            for b, m in modelos.items():
                acum[b] += m.decision_function(x) / len(tt.SEMENTES)
        for b in tt.BRACOS:
            saida[f"{b}_{nome}"] = acum[b]
        print(nome, info[f"unidades_{nome}"], flush=True)
    gravar(M / "predicoes.parquet", saida)
    gravar(M / "predicoes_registro.json", {"sha_predicoes": sha(M / "predicoes.parquet"), "retencao": info,
                                           "ajustes": reg, "protocolo_sha256": sha(M / "protocolo.json"), "quando_utc": agora()})


# ------------------------------------------------------------------ referencia (revp-susc)
def referencia():
    import os
    import rasterio
    from rasterio.warp import transform as wtransform
    from tv06_referencia_rmpa import amostrar, indices, ler_janela
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    prot = ler(M / "protocolo.json")
    sha_pred = ler(M / "predicoes_registro.json")["sha_predicoes"]
    g = pd.read_parquet(M / "grade.parquet", columns=["ponto_id", "lon", "lat"])
    xs, ys = (np.asarray(v) for v in wtransform("EPSG:4326", "EPSG:32720", g.lon.to_numpy(), g.lat.to_numpy()))
    b = (xs.min() - 200, ys.min() - 200, xs.max() + 200, ys.max() + 200)
    base = "/vsicurl/https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/20/M/RB/{a}/{m}/{item}/{banda}.tif"
    amostras, fontes = {}, []
    for chave, item in S2.items():
        dia = item.split("_")[2]
        cols = {}
        for banda in ["SCL", "B03", "B11", "B08"]:
            u = base.format(a=dia[:4], m=int(dia[4:6]), item=item, banda=banda)
            a, tr, crs = ler_janela(u, b, 20.0)
            if crs != "EPSG:32720":
                raise ValueError(crs)
            cols[banda] = amostrar(a, tr, xs, ys, 0)
            fontes.append(u.replace("/vsicurl/", ""))
        m, n = indices(cols["B03"], cols["B11"], cols["B08"])
        amostras[chave] = {"valido": np.isin(cols["SCL"], prot["referencia_primaria"]["valido_scl"]),
                           "com_dado": cols["SCL"] != 0, "mndwi": m, "ndwi": n, "scl": cols["SCL"].astype(int)}
        print(chave, flush=True)
    ev, pre, alt = amostras["evento"], amostras["pre"], amostras["evento_alt"]
    sem_jrc = np.zeros(len(g), bool)
    ref = g[["ponto_id"]].copy()
    ref["oportunidade_s2"], ref["valido_evento"], ref["valido_pre"] = ev["com_dado"], ev["valido"], pre["valido"]
    ref["scl_evento"], ref["mndwi_evento"], ref["mndwi_pre"] = ev["scl"], ev["mndwi"], pre["mndwi"]
    for nome, e, ag, ap in [("primaria", ev, ev["mndwi"] > 0, pre["mndwi"] > 0),
                            ("mndwi_0_2", ev, ev["mndwi"] > .2, pre["mndwi"] > .2),
                            ("ndwi_b08", ev, ev["ndwi"] > 0, pre["ndwi"] > 0),
                            ("data_alternativa", alt, alt["mndwi"] > 0, pre["mndwi"] > 0)]:
        ref[f"estado_{nome}"] = rotulo_referencia(ag, ap, sem_jrc, e["valido"], pre["valido"])
    gravar(M / "referencia.parquet", ref)
    gravar(M / "referencia.json", {"sha_predicoes_ja_registradas": sha_pred, "leu_predicoes": False, "fontes": fontes,
                                   "estados": {c: ref[c].value_counts().to_dict() for c in ref if c.startswith("estado_")},
                                   "sha_referencia": sha(M / "referencia.parquet"), "quando_utc": agora()})
    print({c: ref[c].value_counts().to_dict() for c in ref if c.startswith("estado_")})


# ------------------------------------------------------------------ avaliacao (py312)
def avaliacao():
    from comum_tv import ap_score, auc_mann_whitney, bootstrap_media, intervalo
    from tv07_avaliacao_rmpa import estimar_auc
    bloquear_rede()
    reg = ler(M / "predicoes_registro.json")
    if sha(M / "predicoes.parquet") != reg["sha_predicoes"] or ler(M / "referencia.json")["sha_predicoes_ja_registradas"] != reg["sha_predicoes"]:
        raise SystemExit("ordem predicao -> referencia nao confere")
    g = pd.read_parquet(M / "grade.parquet")
    d = g.merge(pd.read_parquet(M / "predicoes.parquet"), on="ponto_id", how="left").merge(
        pd.read_parquet(M / "referencia.parquet"), on="ponto_id", how="left")
    d["grupo"] = np.where(d.classe_territorial.eq("fcu"), "FCU:" + d.CD_FCU.astype(str), "SET:" + d.CD_SETOR.astype(str))

    def teste(pool, estado="estado_primaria", aceito_dom=True):
        dom = d[f"dominio_{pool}"].fillna(False).astype(bool) if aceito_dom else True
        q = d[d.terreno_valido & dom & d[estado].isin(["positivo", "negativo"])].copy()
        q["y"] = q[estado].eq("positivo").astype(int)
        f = q[q.classe_territorial.eq("fcu")]
        if f.CD_FCU.nunique() < 8 or f.y.sum() < 20 or (1 - f.y).sum() < 20:
            return {"estado": "nao_testavel", "comunidades": int(f.CD_FCU.nunique()), "positivos": int(f.y.sum())}
        e = estimar_auc(f, [f"geral_{pool}", "regra_hand"], "CD_FCU", NIVEL)
        lo1, hi1 = e["auc"]["intervalo"]
        lo2, hi2 = e["diferenca"]["intervalo"]
        est = "sustentada" if lo1 > .6 and lo2 > 0 else "nao_sustentada" if hi1 < .6 or hi2 < 0 else "inconclusiva"
        grupos = {}
        for i, cl in enumerate(["fcu", "baixa_renda", "comparador_urbano"]):
            s = q[q.classe_territorial.eq(cl)]
            if s.y.sum() >= 20 and (1 - s.y).sum() >= 20:
                ee = estimar_auc(s, [f"geral_{pool}", "regra_hand"], "grupo", NIVEL, 20261020 + i)
                grupos[cl] = {"geral": ee["auc"], "geral_menos_hand": ee["diferenca"],
                              "auc_hand": auc_mann_whitney(s.y, s.regra_hand), "prevalencia": float(s.y.mean()),
                              "ap_geral": ap_score(s.y, s[f"geral_{pool}"]), "ap_hand": ap_score(s.y, s.regra_hand)}
        return {"estado": est, "T1": e["auc"], "T2": e["diferenca"], "T2_ganho_relevante": bool(lo2 > .01),
                "auc_hand": auc_mann_whitney(f.y, f.regra_hand), "prevalencia": float(f.y.mean()), "por_grupo": grupos}

    principal = teste("sulamericano")
    replic = teste("internacional")
    sens = {k: teste("sulamericano", f"estado_{k}") for k in ["mndwi_0_2", "ndwi_b08", "data_alternativa"]}
    sens["sem_recusa_dominio"] = teste("sulamericano", aceito_dom=False)
    sens_rep = {k: teste("internacional", f"estado_{k}") for k in ["mndwi_0_2", "ndwi_b08", "data_alternativa"]}
    exp = {}
    for pool in ["sulamericano", "internacional"]:
        q = d[d.terreno_valido & d[f"dominio_{pool}"].fillna(False).astype(bool) & d.estado_primaria.isin(["positivo", "negativo"])]
        f = q[q.classe_territorial.eq("fcu")]
        y = f.estado_primaria.eq("positivo").astype(int)
        exp[pool] = {b: auc_mann_whitney(y, f[f"{b}_{pool}"]) for b in ["geral", "contexto", "proximidade", "aleatorio"]}
    opp = {}
    boots = {}
    for i, cl in enumerate(["fcu", "baixa_renda", "comparador_urbano"]):
        s = d[d.classe_territorial.eq(cl)]
        boots[cl] = bootstrap_media(s, "grupo", "valido_evento", 2000, 20261030 + i)
        opp[cl] = {"fracao_valida": float(s.valido_evento.mean()), "intervalo": intervalo(boots[cl], NIVEL)}
    opp["fcu_menos_comparador"] = intervalo(boots["fcu"] - boots["comparador_urbano"], NIVEL)
    rec = {cl: float((~(d.terreno_valido & d.dominio_sulamericano.fillna(False).astype(bool)))[d.classe_territorial.eq(cl)].mean())
           for cl in ["fcu", "baixa_renda", "comparador_urbano"]}
    gravar(M / "avaliacao/resultado.json", {
        "quando_utc": agora(), "nivel_intervalo": NIVEL, "H_T_linha": principal, "replicacao_H_T_primario": replic,
        "sensibilidades_H_T_linha": sens, "sensibilidades_replicacao": sens_rep,
        "exploratorio_bracos_em_fcu": exp, "H_C": "nao_testavel (um municipio)", "oportunidade_s2": opp,
        "recusa_por_classe_pool_sulamericano": rec,
        "estados": {"H_T_linha": principal["estado"], "replicacao_H_T": replic["estado"]},
        "protocolo_sha256": sha(M / "protocolo.json"), "predicoes_sha256": sha(M / "predicoes.parquet"),
        "referencia_sha256": sha(M / "referencia.parquet")})
    print("H-T'", principal["estado"], "replicacao", replic["estado"])


if __name__ == "__main__":
    {"grade": grade, "protocolo": protocolo, "predicoes": predicoes, "referencia": referencia, "avaliacao": avaliacao}[sys.argv[1]]()
