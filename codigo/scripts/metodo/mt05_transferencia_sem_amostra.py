"""MT05: retreinos sem elevacao dependente da amostra, em desenvolvimento.

Retem componentes inteiros do MT03. As penalidades sao as escolhidas nas
dobras internas do respectivo treino MT03, nunca no destino. Nao seleciona
nova candidata usando a reserva confirmatoria ja aberta.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "2")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/metodo"))
import mt01_experimento as e1
import mt01_relatorio as rel
import mt02_experimento as e2
import mt03_experimento as e3
from proveniencia import agora_utc, sha256_arquivo, sha256_json

SAIDA = REPO / "local_runs/mecanismos-transferiveis/v5/sem_amostra_v1"
BASE = ["f_hand", "f_sca", "f_declive", "f_construido", "f_areia"]
BRACOS = {
    "Q_base": BASE,
    "Q_conexao": BASE + e2.CONEXAO,
    "Q_terreno": ["f_hand", "f_sca", "f_declive"],
    "Q_sem_acumulacao": [c for c in BASE if c != "f_hand"] + e2.CONEXAO,
    "Q_sem_conexao": ["f_hand", "f_construido", "f_areia"],
}
INTERACOES = {n: e2.INTER_CONEXAO for n in ("Q_conexao", "Q_sem_acumulacao")}


def escrever(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rel.limpo(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def ler(p):
    return json.loads(p.read_text(encoding="utf-8"))


def caracteristicas(d):
    """Entradas locais/rasters; nao acessa elevacao relativa nem classe/evento."""
    x = pd.DataFrame(index=d.index)
    x["f_hand"] = np.log1p(d.hand_m.clip(lower=0))
    x["f_sca"] = d.ln_sca.clip(-5, 25)
    x["f_declive"] = np.log1p(d.slope_deg.clip(lower=0))
    x["f_construido"] = d.construido_v2
    x["f_areia"] = d.areia_v2 / 100
    x["f_dist_dren"] = np.log1p(d.dist_drenagem_m)
    x["f_dist_canal"] = np.log1p(d.dist_canal10_m)
    x["z_relevo"] = np.log1p(d.declive_p50)
    return x


def checar_retencao(tr, te):
    for c in ("evento_id", "unidade", "unidade_espacial", "HYBAS_L8"):
        a, b = set(tr[c].dropna()), set(te[c].dropna())
        if c == "HYBAS_L8":
            a.discard(0); b.discard(0)
        if a & b:
            raise ValueError(f"treino/destino compartilham {c}: {sorted(a & b)[:3]}")
    return True


def protocolo(d):
    fontes = [*e3.fontes(), e3.SAIDA / "pre_registro_mt03.json",
              REPO / "scripts/metodo/mt05_transferencia_sem_amostra.py",
              REPO / "scripts/metodo/mt01_experimento.py",
              REPO / "scripts/metodo/mt02_experimento.py",
              REPO / "scripts/metodo/mt03_experimento.py"]
    fontes += sorted((e3.SAIDA / "destinos").glob("*.json"))
    fontes += sorted((e3.SAIDA / "destinos").glob("*.parquet"))
    return {
        "tipo": "MT05_protocolo_de_desenvolvimento_antes_dos_retreinos",
        "bracos": BRACOS, "interacoes": INTERACOES,
        "populacao": {"linhas": len(d), "unidades": d.unidade.nunique(), "eventos": d.evento_id.nunique()},
        "unidades": sorted(d.unidade.unique()),
        "fontes": {str(p.relative_to(REPO)): sha256_arquivo(p) for p in set(fontes)},
        "amostra": "exatamente MT03-A: mesmas linhas, tetos 400/2000 por evento-classe, semente20261010",
        "penalidades": "por destino, MT03-A ja selecionou p/i em dobras internas sem destino; herdadas e iguais entre bracos, sem nova otimizacao",
        "avaliacao": "evento com duas classes e >=20 pontos; media igual de eventos por componente; bootstrap de componentes 2000/seed7; minimo8; margemAUC0,01",
        "comparacoes": ["Q_base-HAND", "Q_base-G_v2", "Q_conexao-G_conexao", "Q_conexao-Q_base", "Q_sem_acumulacao-Q_conexao", "Q_sem_conexao-Q_conexao", "Q_terreno-Q_base"],
        "recortes": ["todos", "processo:fluvial", "processo:enxurrada"],
        "secundarias": "AP/prevalencia e captura10/20 com empate fracionado; descritivas, sem alegacao confirmatoria",
        "limites": ["desenvolvimento retrospectivo; territorios e falhas ja conhecidos", "populacao comum MT03 originalmente filtrada por disponibilidade de elevation_local_m; nao demonstra cobertura fora dela", "nao e novo teste confirmatorio nem causalidade", "desempenho e ordenacao dentro do evento, nao probabilidade calibrada", "sem controle conjunto de multiplicidade; contrastes exploratorios", "proxies nao demonstram invariancia causal", "nao acrescenta rotulos brasileiros independentes"],
        "reservas": "nenhum dado confirmatorio MT04, OBS03, MT02 ou SAR e carregado no treino/avaliacao MT05",
        "modelo_promovido": False,
    }


def pre_registrar():
    d = e3.carregar()
    p = protocolo(d)
    dest = SAIDA / "protocolo.json"
    if dest.exists() and ler(dest) != rel.limpo(p):
        raise ValueError("entradas/protocolo mudaram; requer outra versao")
    if not dest.exists():
        escrever(dest, p)
        escrever(SAIDA / "registro_inicio.json", {"quando_utc": agora_utc(), "protocolo_sha256": sha256_arquivo(dest), "resultados_MT05_lidos": False})
    return d, p


def medir(te, scores, destino):
    rows = []
    for ev, g in te.groupby("evento_id"):
        recortes = {"todos": g}
        if (g.fonte == "cems").all():
            for proc in ("fluvial", "enxurrada"):
                sub = g[(g.classe == 0) | (g.processo_ponto == proc)]
                if (sub.classe == 1).sum() >= 10 and (sub.classe == 0).sum() >= 10:
                    recortes["processo:" + proc] = sub
        for rec, sub in recortes.items():
            y = sub.classe.to_numpy()
            for nome, score in scores.items():
                s = pd.Series(score, index=te.index).loc[sub.index].to_numpy()
                if not np.isfinite(s).all():
                    raise ValueError("escore nao finito")
                auc = roc_auc_score(y, s)
                if abs(auc - e1.auc(y, s)) > 1e-12:
                    raise ValueError("AUC independente divergente")
                rows.append({"destino": destino, "evento_id": ev, "recorte": rec, "modelo": nome,
                             "fonte": sub.fonte.iloc[0], "n": len(sub), "positivos": int(y.sum()),
                             "auc": auc, "ap_sobre_prevalencia": average_precision_score(y, s) / y.mean(),
                             "captura_topo_10": e2.captura(y, s, .1), "captura_topo_20": e2.captura(y, s, .2)})
    return rows


def executar():
    d, p = pre_registrar()
    x = caracteristicas(d)
    for k, destino in enumerate(p["unidades"]):
        arq = SAIDA / "destinos" / (sha256_json(destino)[7:23] + ".json")
        if arq.exists():
            r = ler(arq)
            if r["protocolo_sha256"] != sha256_arquivo(SAIDA / "protocolo.json") or r["predicoes_sha256"] != sha256_arquivo(arq.with_suffix(".parquet")):
                raise ValueError("checkpoint divergente")
            continue
        t0 = time.monotonic()
        alvo = d.unidade == destino
        checar_retencao(d[~alvo], d[alvo])
        tr = e1.amostrar(d[~alvo], e3.CONFIG["teto_ajuste_por_evento_classe"], e3.CONFIG["semente"])
        te = e1.amostrar(d[alvo], e3.CONFIG["teto_teste_por_evento_classe"], e3.CONFIG["semente"])
        ok = te.groupby("evento_id").filter(lambda g: len(g) >= 20 and g.classe.nunique() == 2)
        te = te[te.evento_id.isin(ok.evento_id.unique())]
        old = e3.SAIDA / "destinos" / arq.name
        prev, ref = ler(old), pd.read_parquet(old.with_suffix(".parquet"))
        if prev["destino"] != destino or not np.array_equal(te.ponto_id.to_numpy(), ref.ponto_id.to_numpy()):
            raise ValueError("populacao/ordem diverge do MT03")
        if not np.array_equal(te.classe.to_numpy(), ref.classe.to_numpy()):
            raise ValueError("rotulos divergentes")
        scores = {n: ref[n].to_numpy() for n in ("A0_regra_hand", "G_v2", "G_conexao")}
        coef = {}
        for nome, cols in BRACOS.items():
            mod = e1.Logistica(cols, INTERACOES.get(nome, ()), lam_p=prev["lam"]["p"], lam_i=prev["lam"]["i"])
            mod.fit(x.loc[tr.index], tr.classe.to_numpy(float), tr.evento_id.to_numpy())
            if not mod.convergiu:
                raise ValueError("otimizacao nao convergiu: " + nome)
            scores[nome] = mod.escore(x.loc[te.index])
            coef[nome] = {"principais": dict(zip(cols, mod.b.tolist())),
                          "interacoes": dict(zip([a + ":" + b for a, b in mod.interacoes], mod.g.tolist())),
                          "convergiu": mod.convergiu, "iteracoes": mod.iteracoes}
        pred = te[["ponto_id", "evento_id", "classe", "fonte", "processo_ponto"]].copy()
        for n, s in scores.items():
            pred[n] = s
        arq.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(arq.with_suffix(".parquet"), index=False)
        r = {"destino": destino, "estado": "executado", "lam_herdada": prev["lam"],
             "treino": {"linhas": len(tr), "unidades": tr.unidade.nunique(), "eventos": sorted(tr.evento_id.unique())},
             "teste": {"linhas": len(te), "eventos": sorted(te.evento_id.unique()), "positivos": int(te.classe.sum())},
             "retencao_sem_sobreposicao": True, "coeficientes": coef, "medidas": medir(te, scores, destino),
             "protocolo_sha256": sha256_arquivo(SAIDA / "protocolo.json"),
             "predicoes_sha256": sha256_arquivo(arq.with_suffix(".parquet")), "segundos": time.monotonic() - t0}
        escrever(arq, r)
        print(f"{k+1}/{len(p['unidades'])} {destino}: {r['segundos']:.1f}s", flush=True)
    return analisar()


def analisar():
    pre = ler(SAIDA / "protocolo.json")
    ds = [ler(f) for f in sorted((SAIDA / "destinos").glob("*.json"))]
    if sorted(r["destino"] for r in ds) != pre["unidades"]:
        raise ValueError("execucao incompleta; nao consolidar parcialmente")
    m = pd.DataFrame([a for r in ds for a in r["medidas"]])
    m.to_parquet(SAIDA / "medidas.parquet", index=False)
    pares = [("Q_base", "A0_regra_hand", "superioridade"), ("Q_base", "G_v2", "nao_inferioridade"),
             ("Q_conexao", "G_conexao", "nao_inferioridade"), ("Q_conexao", "Q_base", "superioridade"),
             ("Q_sem_acumulacao", "Q_conexao", "perda"), ("Q_sem_conexao", "Q_conexao", "perda"),
             ("Q_terreno", "Q_base", "nao_inferioridade")]
    resultados = {}
    for rec, g in m.groupby("recorte"):
        e = g.groupby(["destino", "evento_id", "modelo"]).auc.mean().reset_index()
        u = e.groupby(["destino", "modelo"]).auc.mean().unstack()
        resultados[rec] = {"unidades": len(u), "auc_media": u.mean().to_dict(),
                           "por_unidade": u.reset_index().to_dict("records"),
                           "contrastes_exploratorios": {a + "-" + b: rel.contraste(u, a, b, tipo, .01, 2000) for a, b, tipo in pares},
                           "secundarias_descritivas": {met: g.groupby(["destino", "evento_id", "modelo"])[met].mean().groupby(["destino", "modelo"]).mean().groupby("modelo").mean().to_dict() for met in ("ap_sobre_prevalencia", "captura_topo_10", "captura_topo_20")}}
    r = {"tipo": "MT05_retreinos_sem_amostra_resultado", "quando_utc": agora_utc(),
         "protocolo_sha256": sha256_arquivo(SAIDA / "protocolo.json"), "ajustes_executados": len(ds) * len(BRACOS),
         "unidades": len(ds), "eventos": m.evento_id.nunique(), "recortes": resultados,
         "modelo_promovido": False, "regime": "desenvolvimento retrospectivo, nao confirmacao independente", "limites": pre["limites"]}
    escrever(SAIDA / "resultado.json", r)
    return {"ajustes": r["ajustes_executados"], "unidades": r["unidades"], "arquivo": str(SAIDA / "resultado.json")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["preparar", "executar", "analisar"])
    a = ap.parse_args()
    print(pre_registrar()[1]["populacao"] if a.acao == "preparar" else executar() if a.acao == "executar" else analisar())
