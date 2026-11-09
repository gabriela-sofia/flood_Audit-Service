"""MT01, passo 7: leitura dos destinos executados, contrastes por unidade e verificacao independente.

Toda inferencia e por UNIDADE de retencao (componente), nunca por linha ou evento:
media igual dos eventos dentro da unidade, depois das unidades; IC por bootstrap
de unidades. Com menos de 8 unidades o veredito e `not_testable`, como no TT01.

A verificacao independente recalcula a AUC das previsoes gravadas com o
`roc_auc_score` do scikit-learn (o executor usa soma de postos propria), confere
hashes e reconstroi, a partir da populacao, que nenhuma unidade compartilha
evento, lugar ou sub-bacia nivel 8 com outra.

Uso:
    python scripts/metodo/mt01_relatorio.py analisar
    python scripts/metodo/mt01_relatorio.py verificar
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402
import mt01_experimento as ex  # noqa: E402
import mt01_populacao as mp  # noqa: E402

SAIDA = mp.SAIDA
CONTRASTES = [
    ("G", "A0_regra_hand", "H1", "superioridade"), ("G", "A1_regra_composta", "H1", "superioridade"),
    ("A1_regra_composta", "A0_regra_hand", "regra", "superioridade"),
    ("C", "G", "H2", "nao_inferioridade"), ("H", "G", "H2", "nao_inferioridade"), ("H", "C", "H2", "nao_inferioridade"),
    ("G_sem_acumulacao", "G", "H3", "perda"), ("G_sem_conexao", "G", "H3", "perda"),
    ("G_sem_producao", "G", "H3", "perda"), ("G_sem_mesma_fonte", "G", "H4", "perda"),
    ("D_contexto", "D_proximo", "doadores", "superioridade"), ("D_contexto", "D_aleatorio", "doadores", "superioridade"),
    ("D_proximo", "D_aleatorio", "doadores", "superioridade"), ("G", "D_aleatorio", "doadores", "superioridade"),
]


def limpo(o):
    """NaN -> None (JSON estrito); chaves nao-texto viram texto."""
    if isinstance(o, dict):
        return {str(k): limpo(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpo(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def carregar_destinos(pasta: str = "destinos") -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((SAIDA / pasta).glob("*.json"))]


def sensibilidade(nome: str) -> dict:
    """Mesmos contrastes H1/H2 na pasta da sensibilidade pre-registrada (so bracos A0/A1/G/C/H)."""
    pre = json.loads((SAIDA / "pre_registro.json").read_text(encoding="utf-8"))
    cfg = pre["config"]
    ok = [r for r in carregar_destinos(f"destinos_{nome}") if r["estado"] == "executado"]
    u = por_unidade(medidas(ok))
    res = {"tipo": "mt01_sensibilidade", "sensibilidade": nome, "regra": cfg["sensibilidades"][nome],
           "quando_utc": agora_utc(), "unidades": len(ok),
           "auc_media_unidades": {c: float(u[c].mean()) for c in u.columns},
           "contrastes": {f"{a}-{b}": contraste(u, a, b, t, cfg["margem_auc"], cfg["bootstrap"])
                          for a, b, _, t in CONTRASTES[:6]}}
    if nome == "jrc":
        res["entrada_ocorrencia_jrc"] = sha256_arquivo(SAIDA / "ocorrencia_jrc.parquet")
    res = limpo(res)
    mp.escrever(SAIDA / f"sensibilidade_{nome}.json", res)
    return res


def medidas(destinos: list[dict]) -> pd.DataFrame:
    m = pd.DataFrame([x for r in destinos if r["estado"] == "executado" for x in r["medidas"]])
    m["modelo"] = m.modelo.str.replace(r"^(D_aleatorio|curva_U\d+)_\d+$", r"\1", regex=True)
    return m


def por_unidade(m: pd.DataFrame, metrica: str = "auc") -> pd.DataFrame:
    """Eventos com peso igual na unidade; sementes de sorteio fazem media antes."""
    base = m[m.processo.isna()] if "processo" in m else m
    e = base.groupby(["destino", "modelo", "evento_id"])[metrica].mean().reset_index()
    return e.groupby(["destino", "modelo"])[metrica].mean().unstack()


def bootstrap(v: np.ndarray, reps: int, semente: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(semente)
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(reps)])
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))


def contraste(u: pd.DataFrame, a: str, b: str, tipo: str, margem: float, reps: int) -> dict:
    if a not in u or b not in u:
        return {"estado": "braco_ausente"}
    dd = (u[a] - u[b]).dropna()
    r = {"unidades": int(len(dd)), "delta_medio": float(dd.mean()) if len(dd) else None,
         "delta_mediano": float(dd.median()) if len(dd) else None,
         "unidades_a_favor": int((dd > 0).sum()), "acima_da_margem": int((dd > margem).sum()),
         "abaixo_de_menos_margem": int((dd < -margem).sum())}
    if len(dd) < 8:
        return {**r, "veredito": "not_testable", "motivo": "menos de 8 unidades com os dois bracos"}
    lo, hi = bootstrap(dd.to_numpy(), reps)
    r["ic95"] = [lo, hi]
    if tipo == "superioridade":
        r["veredito"] = ("sustentado" if r["delta_medio"] >= margem and lo > 0 else
                         "nao_sustentado" if hi < margem else "indecidivel")
    elif tipo == "nao_inferioridade":
        r["veredito"] = ("superior" if r["delta_medio"] >= margem and lo > 0 else
                         "nao_inferior" if lo > -margem else
                         "inferior" if hi < -margem else "indecidivel")
    else:  # perda: a = modelo sem algo; perda relevante se delta <= -margem com IC < 0
        r["veredito"] = ("perda_relevante" if r["delta_medio"] <= -margem and hi < 0 else
                         "sem_perda_relevante" if lo > -margem else "indecidivel")
    return r


def analisar() -> dict:
    pre = json.loads((SAIDA / "pre_registro.json").read_text(encoding="utf-8"))
    cfg = pre["config"]
    destinos = carregar_destinos()
    estados = pd.Series([r["estado"] for r in destinos]).value_counts().to_dict()
    ok = [r for r in destinos if r["estado"] == "executado"]
    m = medidas(destinos)
    u = por_unidade(m)
    fonte_dest = {r["destino"]: "+".join(r["teste"]["fontes"]) for r in ok}
    res = {"tipo": "mt01_resultado", "quando_utc": agora_utc(), "pre_registro": pre["hash_resumo"],
           "estados": estados, "unidades_executadas": len(ok),
           "cobertura": {"eventos_teste": int(sum(r["teste"]["eventos"] for r in ok)),
                         "linhas_teste": int(sum(r["teste"]["linhas"] for r in ok)),
                         "positivos_teste": int(sum(r["teste"]["positivos"] for r in ok)),
                         "unidades_pre_registradas": len(pre["unidades_de_retencao"])}}
    res["auc_por_modelo"] = {c: {"media_unidades": float(u[c].mean()), "mediana": float(u[c].median()),
                                 "pior_unidade": float(u[c].min()), "decil_inferior": float(u[c].quantile(.1)),
                                 "unidades": int(u[c].notna().sum())} for c in u.columns}
    for met in ("ap_sobre_prevalencia", "captura_topo_10", "captura_topo_20"):
        um = por_unidade(m, met)
        res[f"{met}_por_modelo"] = {c: float(um[c].mean()) for c in um.columns}
    res["contrastes"] = {f"{a}-{b}": {"hipotese": h, "tipo": t,
                                      **contraste(u, a, b, t, cfg["margem_auc"], cfg["bootstrap"])}
                         for a, b, h, t in CONTRASTES}
    uf = u.assign(fonte=u.index.map(fonte_dest))
    res["por_fonte_do_destino"] = {f: {"unidades": int(len(g)), **{c: float(g[c].mean()) for c in
                                       ("A0_regra_hand", "A1_regra_composta", "G", "C", "H", "G_sem_mesma_fonte")
                                       if c in g}} for f, g in uf.groupby("fonte")}
    res["por_unidade"] = uf.round(4).reset_index().to_dict("records")
    # processo no ponto (CEMS): mesma referencia de negativos, positivos por processo
    if "processo" in m:
        pr = m[m.processo.notna()]
        e = pr.groupby(["modelo", "processo", "destino", "evento_id"]).auc.mean().reset_index()
        pu = e.groupby(["modelo", "processo", "destino"]).auc.mean().reset_index()
        res["por_processo_cems"] = {f"{mo}|{p}": {"unidades": int(len(g)), "eventos": int(
            len(e[(e.modelo == mo) & (e.processo == p)])), "auc_media_unidades": float(g.auc.mean())}
                                    for (mo, p), g in pu.groupby(["modelo", "processo"])
                                    if mo in ("A0_regra_hand", "A1_regra_composta", "G", "C", "H")}
    # poucos rotulos (braco separado)
    pr = pd.DataFrame([x for r in ok for x in r["poucos_rotulos"]])
    if len(pr):
        e = pr.groupby(["n", "modelo", "destino", "evento_id"]).auc.mean().reset_index()
        pu = e.groupby(["n", "modelo", "destino"]).auc.mean().unstack("modelo")
        res["poucos_rotulos"] = {int(n): {"unidades": int(len(g)), **{c: float(g[c].mean()) for c in g.columns},
                                          "adaptado_menos_zero": float((g.H_adaptado - g.H_zero).mean()),
                                          "so_local_menos_zero": float((g.so_local - g.H_zero).mean())}
                                 for n, g in pu.groupby(level="n")}
    # suporte dos doadores realizado
    sup = pd.DataFrame([{"braco": k.rsplit("_", 1)[0] if k.startswith("D_aleatorio") else k, **v}
                        for r in ok for k, v in r["suporte_doadores"].items()])
    if len(sup):
        res["suporte_doadores"] = sup.groupby("braco")[["eventos", "positivos", "negativos"]].median().to_dict("index")
    # estabilidade dos coeficientes e comportamento sintetico
    coefs = pd.DataFrame([{"destino": r["destino"], "modelo": k, **v["b"]} for r in ok
                          for k, v in r["coeficientes"].items()])
    res["sinais_G"] = {c: {"positivo": int((coefs[coefs.modelo == "G"][c] > 0).sum()),
                           "negativo": int((coefs[coefs.modelo == "G"][c] < 0).sum())}
                       for c in coefs.columns if c.startswith(("f_", "sem_"))}
    inter = pd.DataFrame([{"destino": r["destino"], **r["coeficientes"]["C"]["g"]} for r in ok])
    res["interacoes_C"] = {c: {"mediana": float(inter[c].median()), "positivo": int((inter[c] > 0).sum()),
                               "negativo": int((inter[c] < 0).sum())} for c in inter.columns if c != "destino"}
    comp = {}
    for r in ok:
        for f, v in r["comportamento"].items():
            for z in ("z_rio", "z_relevo"):
                if z in v:
                    comp.setdefault(f"{f}|{z}", []).append(v[z])
    res["comportamento_sintetico_C"] = {k: np.nanmean(np.array(v, float), axis=0).round(4).tolist()
                                        for k, v in comp.items()}
    res["lam_escolhidos"] = pd.Series([json.dumps(r["lam"], sort_keys=True) for r in ok]).value_counts().head(10).to_dict()
    res["curva_territorios"] = {c: float(u[c].mean()) for c in u.columns if c.startswith("curva_")}
    res = limpo(res)
    res["hash_resumo"] = sha256_json({k: v for k, v in res.items() if k != "quando_utc"})
    mp.escrever(SAIDA / "resultado.json", res)
    return res


def verificar() -> dict:
    from sklearn.metrics import roc_auc_score
    pre = ex.verificar_fontes()
    destinos = carregar_destinos()
    d = pd.read_parquet(SAIDA / "populacao.parquet",
                        columns=["evento_id", "unidade_espacial", "HYBAS_L8", "fonte", "papel"])
    un = pd.read_parquet(SAIDA / "unidades.parquet")
    un = un[un.desenho == pre["config"]["desenho"]]
    d = d.merge(un[["evento_id", "unidade"]], on="evento_id", how="inner")
    cruz = {}
    for c in ("evento_id", "unidade_espacial", "HYBAS_L8"):
        x = d[d[c].notna() & (d[c] != 0)].groupby(c).unidade.nunique()
        cruz[c] = int((x > 1).sum())
    maior, n, hashes = 0.0, 0, 0
    for r in destinos:
        if r["estado"] != "executado":
            continue
        arq = SAIDA / "destinos" / f"{sha256_json(r['destino'])[7:23]}.parquet"
        hashes += int(sha256_arquivo(arq) == r["predicoes_sha256"])
        p = pd.read_parquet(arq)
        if (r["pre_registro"] != pre["hash_resumo"]):
            raise RuntimeError("destino executado sob outro pre-registro")
        med = {(x["modelo"], x["evento_id"]): x["auc"] for x in r["medidas"] if "processo" not in x}
        for (ev), g in p.groupby("evento_id"):
            for mod in ("A0_regra_hand", "A1_regra_composta", "G", "C", "H"):
                a = roc_auc_score(g.classe, g[mod])
                maior = max(maior, abs(a - med[(mod, ev)]))
                n += 1
    out = {"quando_utc": agora_utc(), "auc_recalculadas": n, "divergencia_maxima": maior,
           "predicoes_com_hash_conferido": hashes, "destinos": len(destinos),
           "unidades_que_compartilham": cruz,
           "nao_e": "verifica implementacao e separacao; nao valida a tese"}
    mp.escrever(SAIDA / "verificacao_independente.json", out)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("acao", choices=("analisar", "verificar", "sensibilidade"))
    ap.add_argument("--nome", choices=("jrc", "exutorio"))
    a = ap.parse_args()
    if a.acao == "sensibilidade":
        print(json.dumps(sensibilidade(a.nome), ensure_ascii=False, indent=1))
        return 0
    r = analisar() if a.acao == "analisar" else verificar()
    print(json.dumps(r if a.acao == "verificar" else {k: r[k] for k in ("estados", "auc_por_modelo", "contrastes")},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
