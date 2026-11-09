"""Avaliacao unica da confirmacao na RMPA e conclusao separada de H-T, H-C, H-D e H-A.

Python 3.12. Implementa exatamente `confirmacao_rmpa/protocolo.json`; o que nao esta la sai
marcado como posterior. Le predicoes so depois de conferir que o hash registrado antes da
referencia confere com o arquivo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum_tv import (OUT, agora, ap_score, auc_mann_whitney, bloquear_rede, bootstrap_auc, bootstrap_media,
                      bootstrap_por_grupo, gravar, intervalo, ler, sha)

C = OUT / "confirmacao_rmpa"
B, SEMENTE = 2000, 20261020


def montar() -> pd.DataFrame:
    reg = ler(C / "predicoes_registro.json")
    if sha(C / "predicoes.parquet") != reg["sha_predicoes"]:
        raise SystemExit("predicoes alteradas depois do registro")
    if ler(C / "referencia.json")["sha_predicoes_ja_registradas"] != reg["sha_predicoes"]:
        raise SystemExit("referencia nao foi construida depois destas predicoes")
    g = pd.read_parquet(C / "grade.parquet", columns=["ponto_id", "CD_SETOR", "CD_MUN", "NM_MUN", "CD_FCU", "NM_FCU",
                                                       "classe_territorial", "terreno_valido", "hand_m", "lon", "lat"])
    p = pd.read_parquet(C / "predicoes.parquet")
    r = pd.read_parquet(C / "referencia.parquet")
    d = g.merge(p, on="ponto_id", how="left").merge(r, on="ponto_id", how="left")
    d["aceito"] = d.terreno_valido & d.dominio_estrita.fillna(False).astype(bool)
    d["aceito_local"] = d.terreno_valido & d.dominio_local.fillna(False).astype(bool)
    d["grupo"] = np.where(d.classe_territorial.eq("fcu"), "FCU:" + d.CD_FCU.astype(str), "SET:" + d.CD_SETOR.astype(str))
    return d


def conhecidos(d: pd.DataFrame, estado: str = "estado_primaria", aceito: str = "aceito") -> pd.DataFrame:
    q = d[d[aceito] & d[estado].isin(["positivo", "negativo"])].copy()
    q["y"] = q[estado].eq("positivo").astype(int)
    return q


def estimar_auc(q: pd.DataFrame, cols: list[str], grupo: str, nivel: float, semente: int = SEMENTE) -> dict:
    """AUC da primeira coluna e, se houver segunda, diferenca pareada (primeira - segunda) sob as mesmas reamostragens."""
    bs = bootstrap_auc(q, grupo, cols, B, semente)
    base = {"grupos": int(q[grupo].nunique()), "n": int(len(q)), "positivos": int(q.y.sum()), "negativos": int((1 - q.y).sum())}
    a = auc_mann_whitney(q.y, q[cols[0]])
    out = {"auc": {"estimativa": a, "intervalo": intervalo(bs[:, 0], nivel), **base}, "_boot": bs}
    if len(cols) == 2:
        b = auc_mann_whitney(q.y, q[cols[1]])
        out["diferenca"] = {"estimativa": None if a is None or b is None else a - b,
                            "intervalo": intervalo(bs[:, 0] - bs[:, 1], nivel), **base}
    return out


def h_t(d, nivel, estado="estado_primaria", candidato="geral_estrita", aceito="aceito") -> dict:
    q = conhecidos(d, estado, aceito)
    f = q[q.classe_territorial.eq("fcu")]
    com = f.groupby("CD_FCU").size()
    if len(com) < 8 or f.y.sum() < 20 or (1 - f.y).sum() < 20:
        return {"estado": "nao_testavel", "comunidades": int(len(com)), "positivos": int(f.y.sum())}
    e = estimar_auc(f, [candidato, "regra_hand"], "CD_FCU", nivel)
    t1, t2 = e["auc"], e["diferenca"]
    lo1, hi1 = t1["intervalo"]
    lo2, hi2 = t2["intervalo"]
    if lo1 > .60 and lo2 > 0:
        est = "sustentada"
    elif hi1 < .60 or hi2 < 0:
        est = "nao_sustentada"
    else:
        est = "inconclusiva"
    return {"estado": est, "T1_auc_candidato": t1, "T2_candidato_menos_hand": t2,
            "T2_ganho_relevante": bool(lo2 > .01), "T1_sustentado": bool(lo1 > .60),
            "T2_superioridade": bool(lo2 > 0), "auc_hand": auc_mann_whitney(f.y, f.regra_hand),
            "ap_candidato": ap_score(f.y, f[candidato]), "prevalencia": float(f.y.mean())}


def por_grupo_absoluto(d, nivel) -> dict:
    q = conhecidos(d)
    out, boots = {}, {}
    for i, cl in enumerate(["fcu", "baixa_renda", "comparador_urbano"]):
        s = q[q.classe_territorial.eq(cl)]
        e = estimar_auc(s, ["geral_estrita", "regra_hand"], "grupo", nivel, SEMENTE + i)
        boots[cl] = e["_boot"]
        out[cl] = {"geral": e["auc"], "geral_menos_hand": e["diferenca"],
                   "hand": {"estimativa": auc_mann_whitney(s.y, s.regra_hand), "intervalo": intervalo(e["_boot"][:, 1], nivel)},
                   "prevalencia": float(s.y.mean()), "ap_geral": ap_score(s.y, s.geral_estrita), "ap_hand": ap_score(s.y, s.regra_hand)}
    # FCU - comparador: bootstraps independentes (grupos distintos)
    for j, nome in enumerate(["geral", "hand"]):
        est = out["fcu"][nome]["estimativa"] if nome == "hand" else out["fcu"]["geral"]["estimativa"]
        cmp = out["comparador_urbano"][nome]["estimativa"] if nome == "hand" else out["comparador_urbano"]["geral"]["estimativa"]
        out[f"fcu_menos_comparador_{nome}"] = {"estimativa": est - cmp,
                                               "intervalo": intervalo(boots["fcu"][:, j] - boots["comparador_urbano"][:, j], nivel)}
    return out


def h_c(d, nivel) -> dict:
    q = conhecidos(d)
    linhas = []
    for mun, s in q.groupby("CD_MUN"):
        if s.y.sum() < 20 or (1 - s.y).sum() < 20:
            continue
        lin = {"CD_MUN": mun, "NM_MUN": s.NM_MUN.iloc[0], "n": len(s), "pos": int(s.y.sum())}
        for b in ["geral", "contexto", "proximidade", "aleatorio"]:
            lin[b] = auc_mann_whitney(s.y, s[f"{b}_estrita"])
        lin["hand"] = auc_mann_whitney(s.y, s.regra_hand)
        f = s[s.classe_territorial.eq("fcu")]
        if f.y.sum() >= 5 and (1 - f.y).sum() >= 5:
            for b in ["geral", "contexto", "proximidade", "aleatorio"]:
                lin[f"fcu_{b}"] = auc_mann_whitney(f.y, f[f"{b}_estrita"])
        linhas.append(lin)
    u = pd.DataFrame(linhas)
    if len(u) < 8:
        return {"estado": "nao_testavel", "municipios": len(u), "tabela": u.to_dict(orient="records")}
    rng = np.random.default_rng(SEMENTE)
    contrastes, todos_pos, algum_neg = {}, True, False
    for b in ["geral", "proximidade", "aleatorio"]:
        dif = (u.contexto - u[b]).to_numpy()
        bs = np.array([dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(B)])
        lo, hi = intervalo(bs, nivel)
        contrastes[f"contexto-{b}"] = {"media": float(dif.mean()), "intervalo": [lo, hi], "municipios": len(dif),
                                       "a_favor": int((dif > 0).sum())}
        todos_pos &= lo > 0
        algum_neg |= hi < 0
    sec = {}
    uf = u.dropna(subset=["fcu_contexto"]) if "fcu_contexto" in u else u.iloc[0:0]
    for b in ["geral", "proximidade", "aleatorio"]:
        if len(uf):
            dd = (uf.fcu_contexto - uf[f"fcu_{b}"])
            sec[f"contexto-{b}"] = {"media": float(dd.mean()), "municipios": int(len(dd)), "a_favor": int((dd > 0).sum())}
    return {"estado": "sustentada" if todos_pos else "nao_sustentada" if algum_neg else "inconclusiva",
            "contrastes": contrastes, "secundario_fcu_descritivo": sec, "tabela": u.to_dict(orient="records")}


def oportunidade(d, nivel) -> dict:
    """D2: fracao de pontos com observacao S2; FCU por comunidade, demais por setor."""
    out = {}
    for col in ["oportunidade_s2", "valido_evento"]:
        res, boots = {}, {}
        for i, cl in enumerate(["fcu", "baixa_renda", "comparador_urbano"]):
            s = d[d.classe_territorial.eq(cl)]
            boots[cl] = bootstrap_media(s, "grupo", col, B, SEMENTE + i)
            res[cl] = {"fracao": float(s[col].astype(float).mean()), "intervalo": intervalo(boots[cl], nivel), "pontos": int(len(s))}
        res["fcu_menos_comparador"] = {"estimativa": res["fcu"]["fracao"] - res["comparador_urbano"]["fracao"],
                                       "intervalo": intervalo(boots["fcu"] - boots["comparador_urbano"], nivel)}
        out[col] = res
    return out


def disponibilidade_acervo(nivel) -> dict:
    """D1: setores com ponto fisico do acervo por classe nas 7 UFs; bootstrap por municipio."""
    s = pd.read_parquet(OUT / "universo/setores_geometria_local.parquet",
                        columns=["CD_SETOR", "CD_MUN", "UF", "classe_territorial", "fisica", "administrativa", "pessoas"])
    s = s[s.classe_territorial.isin(["fcu", "comparador_urbano", "baixa_renda"])]
    mun = s.groupby("CD_MUN").classe_territorial.apply(lambda c: {"fcu", "comparador_urbano"} <= set(c))
    s = s[s.CD_MUN.isin(mun[mun].index)].assign(com=lambda x: (x.fisica > 0).astype(float),
                                                 adm=lambda x: (x.administrativa > 0).astype(float))

    def dif(q, col="com"):
        a = q[q.classe_territorial.eq("fcu")][col].mean()
        b = q[q.classe_territorial.eq("comparador_urbano")][col].mean()
        return float(a - b)
    out = {}
    for col in ["com", "adm"]:
        bs = bootstrap_por_grupo(s, "CD_MUN", lambda q: dif(q, col), B, SEMENTE)
        out["fisica" if col == "com" else "administrativa"] = {
            "fcu": float(s[s.classe_territorial.eq("fcu")][col].mean()),
            "baixa_renda": float(s[s.classe_territorial.eq("baixa_renda")][col].mean()),
            "comparador_urbano": float(s[s.classe_territorial.eq("comparador_urbano")][col].mean()),
            "fcu_menos_comparador": dif(s, col), "intervalo": intervalo(bs, nivel),
            "municipios": int(s.CD_MUN.nunique()), "setores": int(len(s))}
    return out


def h_d(d, nivel) -> dict:
    d1 = disponibilidade_acervo(nivel)
    d2 = oportunidade(d, nivel)
    e = d[d.estado_emsn194.isin(["positivo"]) & d.valido_evento]
    d3 = {cl: {"pontos_na_extensao_emsn": int(len(s)), "detectados_s2": float(s.estado_primaria.eq("positivo").mean()) if len(s) else None}
          for cl, s in e.groupby("classe_territorial")}
    lo1, hi1 = d1["fisica"]["intervalo"]
    lo2, hi2 = d2["valido_evento"]["fcu_menos_comparador"]["intervalo"]
    desfavor = hi1 < 0 or hi2 < 0
    dentro = (-.05 <= lo1 and hi1 <= .05) and (-.05 <= lo2 and hi2 <= .05)
    return {"estado": "sustentada_associativa" if desfavor else "nao_sustentada" if dentro else "inconclusiva",
            "D1_disponibilidade_acervo": d1, "D2_oportunidade_s2": d2,
            "D3_deteccao_optica_dentro_da_extensao_emsn": d3,
            "D4_registro": {"estado": "nao_testavel", "motivo": "sem registro administrativo geocodificado da RMPA no acervo"},
            "leitura": "associacao delimitada; nao demonstra que segregacao ou atencao publica causaram a diferenca"}


def h_a(d, ht, nivel) -> dict:
    uni = ler(OUT / "universo/resultado.json")["funil_fcu"]
    q = d.copy()
    q["recusa"] = ~q.aceito
    boots = {}
    rec = {}
    for i, cl in enumerate(["fcu", "comparador_urbano", "baixa_renda"]):
        s = q[q.classe_territorial.eq(cl)]
        boots[cl] = bootstrap_media(s, "grupo", "recusa", B, SEMENTE + 10 + i)
        rec[cl] = float(s.recusa.mean())
    dif = intervalo(boots["fcu"] - boots["comparador_urbano"], nivel)
    a3 = dif[1] <= .05
    # captura por orcamento de area: topo do escore entre os pontos aceitos do municipio (sem rotulo)
    k = conhecidos(d)
    aceitos = d[d.aceito].copy()
    aceitos["pct"] = aceitos.groupby("CD_MUN").geral_estrita.rank(pct=True, ascending=False)
    k = k.merge(aceitos[["ponto_id", "pct"]], on="ponto_id")
    orc = {}
    for cl in ["fcu", "baixa_renda", "comparador_urbano"]:
        s = k[k.classe_territorial.eq(cl)]
        orc[cl] = {}
        for frac in (.1, .2, .3):
            alerta = s.pct <= frac
            tp = int((alerta & s.y.eq(1)).sum())
            pos = int(s.y.sum())
            orc[cl][f"{int(frac*100)}%"] = {"captura": tp / pos if pos else None, "falso_negativo": 1 - tp / pos if pos else None,
                                            "precisao": tp / int(alerta.sum()) if alerta.sum() else None,
                                            "prevalencia": float(s.y.mean()), "fracao_dos_pontos_em_alerta": float(alerta.mean())}
    rmpa = {"comunidades_universo": int(pd.read_csv(C / "setores_universo.csv").query("classe_territorial=='fcu'").CD_FCU.nunique()),
            "comunidades_com_ponto": int(d[d.classe_territorial.eq("fcu")].CD_FCU.nunique()),
            "comunidades_com_resposta": int(d[d.classe_territorial.eq("fcu") & d.aceito].CD_FCU.nunique()),
            "comunidades_avaliaveis": int(k[k.classe_territorial.eq("fcu")].CD_FCU.nunique()),
            "comunidades_com_duas_classes": int((k[k.classe_territorial.eq("fcu")].groupby("CD_FCU").y.agg(["min", "max"]).pipe(lambda z: (z["min"] == 0) & (z["max"] == 1))).sum())}
    t1 = ht.get("T1_sustentado", False)
    return {"estado_no_dominio_avaliado": "sustentada" if (t1 and a3) else ("nao_testavel" if ht["estado"] == "nao_testavel" else
                                                                            "nao_sustentada" if ht.get("T1_auc_candidato", {}).get("intervalo", [0, 1])[1] < .60 else "inconclusiva"),
            "estado_universo": "nao_testavel_fora_do_episodio_e_mecanismo_avaliados",
            "A1_universo": uni, "A1_rmpa": rmpa, "A2_orcamentos_de_area": orc,
            "A3_recusa": {"por_classe": rec, "fcu_menos_comparador_intervalo": dif, "aceitavel": bool(a3)},
            "ausencia_de_modelo_local": "indisponibilidade, nao AUC zero"}


def por_comunidade(d) -> pd.DataFrame:
    k = conhecidos(d)
    f = d[d.classe_territorial.eq("fcu")]
    aceitos = d[d.aceito].copy()
    aceitos["pct"] = aceitos.groupby("CD_MUN").geral_estrita.rank(pct=True, ascending=False)
    f = f.merge(aceitos[["ponto_id", "pct"]], on="ponto_id", how="left")
    linhas = []
    for cfcu, s in f.groupby("CD_FCU"):
        kk = s[s.aceito & s.estado_primaria.isin(["positivo", "negativo"])]
        y = kk.estado_primaria.eq("positivo").astype(int)
        alerta = kk.pct <= .2
        linhas.append({"CD_FCU": cfcu, "NM_FCU": s.NM_FCU.iloc[0], "NM_MUN": s.NM_MUN.iloc[0], "pontos": len(s),
                       "recusados": int((~s.aceito).sum()), "sem_oportunidade": int((~s.oportunidade_s2).sum()),
                       "desconhecido_ou_permanente": int(s.estado_primaria.isin(["desconhecido", "permanente", "pre_desconhecido"]).sum()),
                       "conhecidos": len(kk), "positivos": int(y.sum()), "fracao_inundada_obs": float(y.mean()) if len(kk) else None,
                       "emsn_positivos": int(s.estado_emsn194.eq("positivo").sum()),
                       "escore_geral_medio": float(s.geral_estrita.mean()) if s.geral_estrita.notna().any() else None,
                       "hand_mediano": float(s.hand_m.median()),
                       "auc_geral": auc_mann_whitney(y, kk.geral_estrita) if y.sum() >= 5 and (1 - y).sum() >= 5 else None,
                       "auc_hand": auc_mann_whitney(y, kk.regra_hand) if y.sum() >= 5 and (1 - y).sum() >= 5 else None,
                       "acertos_alerta20": int((alerta & y.eq(1)).sum()), "omissoes_alerta20": int((~alerta & y.eq(1)).sum()),
                       "falsos_alarmes_alerta20": int((alerta & y.eq(0)).sum()),
                       "lon": float(s.lon.mean()), "lat": float(s.lat.mean())})
    return pd.DataFrame(linhas)


def main():
    bloquear_rede()
    t0 = agora()
    prot = ler(C / "protocolo.json")
    nivel = prot["metricas_e_criterios"]["nivel_intervalo"]
    d = montar()
    ht = h_t(d, nivel)
    print("H-T", ht["estado"], flush=True)
    sens = {"mndwi_0_2": h_t(d, nivel, "estado_mndwi_0_2"), "ndwi_b08": h_t(d, nivel, "estado_ndwi_b08"),
            "emsn194_dependente_fabdem": h_t(d, nivel, "estado_emsn194"),
            "retencao_local": h_t(d, nivel, candidato="geral_local", aceito="aceito_local"),
            "todos_com_terreno": h_t(d, nivel, aceito="terreno_valido")}
    print("sensibilidades", {k: v["estado"] for k, v in sens.items()}, flush=True)
    grupos = por_grupo_absoluto(d, nivel)
    hc = h_c(d, nivel)
    print("H-C", hc["estado"], flush=True)
    hd = h_d(d, nivel)
    print("H-D", hd["estado"], flush=True)
    ha = h_a(d, ht, nivel)
    print("H-A", ha["estado_no_dominio_avaliado"], flush=True)
    com = por_comunidade(d)
    gravar(C / "avaliacao/por_comunidade.csv", com)
    gravar(C / "avaliacao/por_municipio_hc.csv", pd.DataFrame(hc["tabela"]))
    gravar(C / "avaliacao/resultado.json", {
        "inicio_utc": t0, "fim_utc": agora(), "protocolo_sha256": sha(C / "protocolo.json"),
        "predicoes_sha256": sha(C / "predicoes.parquet"), "referencia_sha256": sha(C / "referencia.parquet"),
        "nivel_intervalo": nivel, "H_T": ht, "H_T_sensibilidades_planejadas": sens, "H_T_por_grupo": grupos,
        "H_C": {k: v for k, v in hc.items() if k != "tabela"}, "H_D": hd, "H_A": ha,
        "estados": {"H_T": ht["estado"], "H_C": hc["estado"], "H_D": hd["estado"],
                    "H_A": ha["estado_no_dominio_avaliado"], "H_A_universo": ha["estado_universo"]},
    })


if __name__ == "__main__":
    main()
