"""MT01, leituras POS-HOC (rotuladas assim; nao entram nos vereditos pre-registrados).

  familia_mdt    AUC por familia do MDT do destino (o controle e pre-registrado; o recorte por
                 unidade e calculado aqui porque o registro do destino nao guarda a familia)
  peso_suporte   media das unidades ponderada por min(positivos, negativos)^0,5 do teste:
                 unidades com 3 positivos pesam igual as de milhares no veredito; isto mostra o efeito
  secundarias    CORRECAO de defeito: `ap` e `captura_topo` do executor nao tratavam empate (a regra HAND tem
                 muitos HAND = 0 exatos, e a ordem de entrada decidia o empate). Recalculadas das previsoes
                 gravadas: AP pelo scikit-learn (limiares por valor distinto) e captura com o grupo empatado no
                 corte entrando pela fracao. Os valores originais ficam preservados nos registros dos destinos.
  heterogeneidade  G ajustado DENTRO de cada unidade com suporte (>= 2 eventos ou >= 200 por classe): os
                 coeficientes variam entre territorios? (transferir mecanismo nao exige coeficiente identico)
  teto_local     unidades com >= 5 eventos: G ajustado com o resto do mundo MAIS 4/5 dos eventos do proprio
                 territorio (dobras por evento), avaliado no 1/5 restante; contra G sem rotulo local nos mesmos
                 eventos. Eventos do mesmo territorio dividem lugar: e exatamente o cenario "com dado local".

Uso:
    python scripts/metodo/mt01_pos_hoc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc  # noqa: E402
import mt01_populacao as mp  # noqa: E402
import mt01_relatorio as rel  # noqa: E402

PRINCIPAIS = ("A0_regra_hand", "A1_regra_composta", "G", "C", "H", "G_sem_producao", "G_sem_acumulacao",
              "G_sem_conexao")


def main(pasta: str = "destinos") -> dict:
    sys.stdout.reconfigure(encoding="utf-8")
    destinos = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((mp.SAIDA / pasta).glob("*.json"))]
    ok = [r for r in destinos if r["estado"] == "executado"]
    m = rel.medidas(ok)
    u = rel.por_unidade(m)
    pop = pd.read_parquet(mp.SAIDA / "populacao.parquet", columns=["evento_id", "familia_mdt", "papel"])
    un = pd.read_parquet(mp.SAIDA / "unidades.parquet")
    un = un[un.desenho == "primario"]
    fam = pop[pop.papel.isin(["positivo_episodico", "negativo_observado"])].merge(un, on="evento_id")
    fam = fam.groupby("unidade").familia_mdt.agg(lambda s: s.fillna("sem").value_counts().idxmax())
    out = {"tipo": "mt01_pos_hoc", "quando_utc": agora_utc(), "pasta": pasta,
           "aviso": "leituras posteriores; nao alteram protocolo nem vereditos"}
    uf = u.assign(familia=u.index.map(fam))
    out["por_familia_mdt"] = {f: {"unidades": int(len(g)), **{c: round(float(g[c].mean()), 4)
                                                               for c in PRINCIPAIS if c in g}}
                              for f, g in uf.groupby("familia")}
    tam = {r["destino"]: min(r["teste"]["positivos"], r["teste"]["linhas"] - r["teste"]["positivos"]) for r in ok}
    w = pd.Series(tam).reindex(u.index) ** .5
    out["media_ponderada_por_suporte"] = {c: round(float(np.average(u[c].dropna(), weights=w[u[c].notna()])), 4)
                                          for c in PRINCIPAIS if c in u}
    out["media_igual"] = {c: round(float(u[c].mean()), 4) for c in PRINCIPAIS if c in u}
    dd = (u["G"] - u["A0_regra_hand"])
    out["G_menos_A0_por_tamanho"] = {
        "unidades_com_menos_de_50_positivos": round(float(dd[pd.Series(tam).reindex(u.index) < 50].mean()), 4),
        "unidades_com_50_ou_mais": round(float(dd[pd.Series(tam).reindex(u.index) >= 50].mean()), 4)}
    out["secundarias_corrigidas"] = secundarias(pasta)
    if pasta == "destinos":
        out["heterogeneidade"] = heterogeneidade()
        out["teto_local"] = teto_local(ok)
    mp.escrever(mp.SAIDA / f"pos_hoc_{pasta}.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def captura(y: np.ndarray, s: np.ndarray, f: float) -> float:
    """Fracao dos positivos no topo f do quadro, com o grupo empatado no corte entrando pela fracao."""
    o = np.argsort(-s, kind="stable")
    s, y = s[o], y[o]
    k = int(np.floor(f * len(s)))
    if k == 0 or y.sum() == 0:
        return float("nan")
    corte = s[k - 1]
    acima = s > corte
    emp = s == corte
    vagas = k - acima.sum()
    return float((y[acima].sum() + y[emp].sum() * vagas / emp.sum()) / y.sum())


def secundarias(pasta: str) -> dict:
    from sklearn.metrics import average_precision_score
    linhas = []
    for arq in sorted((mp.SAIDA / pasta).glob("*.parquet")):
        p = pd.read_parquet(arq)
        dest = json.loads(arq.with_suffix(".json").read_text(encoding="utf-8"))["destino"]
        for ev, g in p.groupby("evento_id"):
            y = g.classe.to_numpy()
            for m in ("A0_regra_hand", "A1_regra_composta", "G", "C", "H"):
                s = g[m].to_numpy()
                linhas.append({"destino": dest, "modelo": m, "evento_id": ev,
                               "ap_sobre_prevalencia": average_precision_score(y, s) / y.mean(),
                               "captura_topo_10": captura(y, s, .1), "captura_topo_20": captura(y, s, .2)})
    t = pd.DataFrame(linhas)
    u = t.groupby(["destino", "modelo", "evento_id"]).mean().groupby(["destino", "modelo"]).mean()
    return {met: {m: round(float(g[met].mean()), 4) for m, g in u.groupby(level="modelo")}
            for met in ("ap_sobre_prevalencia", "captura_topo_10", "captura_topo_20")}


def heterogeneidade(min_classe: int = 200) -> dict:
    import mt01_experimento as ex
    d = ex.carregar()
    x = ex.caracteristicas(d)
    sem = ex.CONFIG["semente"]
    efeitos = sum(ex.PRINCIPAIS.values(), [])
    linhas = []
    for un, g in d.groupby("unidade"):
        a = ex.amostrar(g, ex.CONFIG["teto_ajuste_por_evento_classe"], sem)
        if min((a.classe == 1).sum(), (a.classe == 0).sum()) < min_classe:
            continue
        m = ex.Logistica(efeitos, lam_p=1e-3).fit(x.loc[a.index], a.classe.to_numpy(float), a.evento_id.to_numpy())
        # padronizacao comum: coeficiente por desvio do POOL, para comparar territorios na mesma escala
        escala = x.loc[d.index, efeitos].std()
        b = m.b / m.pad.sd[efeitos].to_numpy() * escala.to_numpy()
        linhas.append({"unidade": un, "eventos": int(a.evento_id.nunique()), **dict(zip(efeitos, b))})
    t = pd.DataFrame(linhas)
    return {"unidades": int(len(t)), "por_efeito": {
        c: {"mediana": round(float(t[c].median()), 3), "p10": round(float(t[c].quantile(.1)), 3),
            "p90": round(float(t[c].quantile(.9)), 3), "negativos": int((t[c] < 0).sum()),
            "positivos": int((t[c] > 0).sum())} for c in efeitos},
        "por_unidade": t.round(3).to_dict("records")}


def teto_local(ok: list[dict], dobras: int = 5, minimo_eventos: int = 5) -> dict:
    import mt01_experimento as ex
    d = ex.carregar()
    x = ex.caracteristicas(d)
    sem = ex.CONFIG["semente"]
    res = {}
    for r in ok:
        dest = r["destino"]
        evs = np.array(sorted(d[d.unidade == dest].evento_id.unique()))
        if len(evs) < minimo_eventos:
            continue
        dobra = dict(zip(evs, np.random.default_rng(sem).permutation(len(evs)) % dobras))
        fora = ex.amostrar(d[d.unidade != dest], ex.CONFIG["teto_ajuste_por_evento_classe"], sem)
        dentro = d[d.unidade == dest]
        linhas = []
        for k in range(dobras):
            tr_local = ex.amostrar(dentro[dentro.evento_id.map(dobra) != k], ex.CONFIG["teto_ajuste_por_evento_classe"], sem)
            te = ex.amostrar(dentro[dentro.evento_id.map(dobra) == k], ex.CONFIG["teto_teste_por_evento_classe"], sem)
            tr = pd.concat([fora, tr_local])
            escores = {"A0_regra_hand": -te.hand_m.to_numpy()}
            for nome, base in (("G_sem_local", fora), ("G_com_local", tr), ("G_so_local", tr_local)):
                if base.classe.nunique() == 2:
                    escores[nome] = ex.ajustar("G", x.loc[base.index], base, r["lam"]).escore(x.loc[te.index])
            for nome, s in escores.items():
                for ev, g in te.assign(_s=s).groupby("evento_id"):
                    if g.classe.nunique() == 2 and len(g) >= ex.CONFIG["minimo_teste_evento"]:
                        linhas.append({"modelo": nome, "evento_id": ev, "auc": ex.auc(g.classe.to_numpy(), g._s.to_numpy())})
        t = pd.DataFrame(linhas)
        comuns = set.intersection(*[set(g.evento_id) for _, g in t.groupby("modelo")])
        t = t[t.evento_id.isin(comuns)]
        res[dest] = {"eventos": len(comuns), **{m_: round(float(g.auc.mean()), 4) for m_, g in t.groupby("modelo")}}
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "destinos")
