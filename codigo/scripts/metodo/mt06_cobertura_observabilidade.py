"""MT06: contribuição de cobertura/solo versus disponibilidade de observação.

Experimento novo no desenvolvimento MT05. Retém componentes inteiros, retreina
oito braços e preserva as máscaras ao romper valores de cobertura no treino.
Não lê reservas nem dados ópticos/forçamento da rodada 2. Não altera o serviço.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "2")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/metodo"))
import mt05_transferencia_sem_amostra as mt
from proveniencia import agora_utc, sha256_arquivo, sha256_json

SAIDA = REPO / "local_runs/mecanismos-transferiveis/v6/cobertura_observabilidade_v1_1"
FULL = list(mt.BRACOS["Q_conexao"])
VALUES = ["f_construido", "f_areia"]
PHYS = [c for c in FULL if c not in VALUES]
MASKS = ["ausente_construido", "ausente_areia"]
INTER = list(mt.e2.INTER_CONEXAO)
BRACOS = {
    "fisica": PHYS,
    "fisica_construido": PHYS + ["f_construido"],
    "fisica_areia": PHYS + ["f_areia"],
    "fisica_cobertura": FULL,
    "fisica_mascaras": PHYS + MASKS,
    "fisica_valores_mascaras": FULL + MASKS,
    "mascaras_so": MASKS,
    "valores_rompidos_mascaras_preservadas": FULL + MASKS,
}
PRIMARIAS = {
    "P1_cobertura": ("fisica_cobertura", "fisica"),
    "P2_construido_condicional": ("fisica_cobertura", "fisica_areia"),
    "P3_areia_condicional": ("fisica_cobertura", "fisica_construido"),
}
SECUNDARIAS = {
    **PRIMARIAS,
    "mascaras_sobre_fisica": ("fisica_mascaras", "fisica"),
    "mascaras_sobre_valores": ("fisica_valores_mascaras", "fisica_cobertura"),
    "valores_sobre_mascaras": ("fisica_valores_mascaras", "fisica_mascaras"),
    "alinhamento_dos_valores": ("fisica_valores_mascaras", "valores_rompidos_mascaras_preservadas"),
}
CFG = {"semente_amostra": 20261010, "semente_permutacao": 20261017,
       "treino_por_evento_classe": 400, "teste_por_evento_classe": 2000,
       "bootstrap": 2000, "semente_bootstrap": 7, "minimo_componentes": 8,
       "margem_auc": .01, "nivel_primarias": 1 - .05 / 3,
       "construido_aberto_max": .1, "construido_denso_min": .5,
       "minimo_classe_no_estrato": 10, "limite_disco_novo_bytes": 1024**3,
       "elegibilidade_todos": "mesma MT05: >=20 pontos e duas classes; piso10 somente em estratos",
       "threads_blas": 2}


def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def escrever(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_text(json.dumps(mt.rel.limpo(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, p)


def caracteristicas(d):
    x = mt.caracteristicas(d)
    for col, missing in zip(VALUES, MASKS):
        x[missing] = x[col].isna().astype(float)
    return x


def romper_valores(x, eventos, seed):
    """Permuta valores FINITOS dentro do evento; mantém ausência e atributos físicos."""
    out = x.copy(deep=True)
    rng = np.random.default_rng(seed)
    for _, ids in pd.Series(eventos, index=x.index).groupby(pd.Series(eventos, index=x.index), sort=True).groups.items():
        for c in VALUES:
            idx = x.loc[ids, c].dropna().index
            out.loc[idx, c] = rng.permutation(x.loc[idx, c].to_numpy())
    for c in VALUES:
        if not x[c].isna().equals(out[c].isna()):
            raise ValueError("permutação mudou a máscara")
    return out


def estratos(d):
    yield "todos", d
    if (d.fonte == "cems").all():
        for proc in ("fluvial", "enxurrada"):
            yield "processo:" + proc, d[(d.classe == 0) | (d.processo_ponto == proc)]
    built = d.f_construido
    yield "ocupacao:aberto", d[built < CFG["construido_aberto_max"]]
    yield "ocupacao:intermediario", d[(built >= CFG["construido_aberto_max"]) & (built < CFG["construido_denso_min"])]
    yield "ocupacao:denso", d[built >= CFG["construido_denso_min"]]
    yield "solo:observado", d[d.ausente_areia == 0]
    yield "solo:ausente", d[d.ausente_areia == 1]


def medir(pred):
    rows = []
    for ev, event in pred.groupby("evento_id"):
        for rec, g in estratos(event):
            if not elegivel(g, rec):
                continue
            y = g.classe.to_numpy()
            for name in BRACOS:
                score = g[name].to_numpy()
                if not np.isfinite(score).all():
                    raise ValueError("escore não finito")
                rows.append({"destino": str(g.unidade.iloc[0]), "evento_id": ev, "fonte": str(g.fonte.iloc[0]),
                             "recorte": rec, "modelo": name, "n": len(g), "positivos": int(y.sum()),
                             "auc": float(roc_auc_score(y, score)),
                             "ap_sobre_prevalencia": float(average_precision_score(y, score) / y.mean())})
    return rows


def elegivel(g, rec):
    counts = g.classe.value_counts()
    minimum = 1 if rec == "todos" else CFG["minimo_classe_no_estrato"]
    return len(g) >= 20 and min(counts.get(0, 0), counts.get(1, 0)) >= minimum


def contraste(u, a, b, nivel=.95):
    if a not in u or b not in u:
        return {"veredito": "not_testable", "motivo": "braço/recorte ausente", "n": 0}
    d = (u[a] - u[b]).dropna()
    r = {"n": len(d), "media": float(d.mean()) if len(d) else None,
         "a_favor": int((d > 0).sum()), "por_componente": d.to_dict()}
    if len(d) < CFG["minimo_componentes"]:
        return {**r, "veredito": "not_testable", "motivo": "piso histórico de componentes; estimativa descritiva"}
    v = d.to_numpy()
    boot = np.random.default_rng(CFG["semente_bootstrap"]).choice(v, (CFG["bootstrap"], len(v)), replace=True).mean(1)
    q = (1 - nivel) / 2
    lo, hi = np.quantile(boot, [q, 1-q])
    verdict = "sustentado" if r["media"] >= CFG["margem_auc"] and lo > 0 else (
        "nao_sustentado" if hi < CFG["margem_auc"] else "indecidivel")
    return {**r, "ic": [float(lo), float(hi)], "nivel_ic": nivel, "veredito": verdict,
            "inferior_a_zero": bool(hi < 0)}


def preparar():
    arq = SAIDA / "protocolo.json"
    if arq.exists():
        p = ler(arq)
        conferir(p)
        return p
    d = mt.e3.carregar()
    x = caracteristicas(d)
    meta = d[["ponto_id", "evento_id", "unidade", "unidade_espacial", "HYBAS_L8", "classe", "fonte", "processo_ponto"]].copy()
    base = pd.concat([meta, x], axis=1)
    SAIDA.mkdir(parents=True, exist_ok=True)
    base.to_parquet(SAIDA / "base.parquet", index=False)
    olds = {}
    for un in sorted(d.unidade.unique()):
        name = sha256_json(un)[7:23]
        old = mt.e3.SAIDA / "destinos" / f"{name}.json"
        ref = mt.SAIDA / "destinos" / f"{name}.json"
        info = ler(old)
        if info["destino"] != un:
            raise ValueError("identidade MT03 divergente")
        olds[un] = {"lam": info["lam"], "mt03_json": str(old.relative_to(REPO)), "mt03_sha256": sha256_arquivo(old),
                    "mt05_predicao": str(ref.with_suffix('.parquet').relative_to(REPO)),
                    "mt05_predicao_sha256": sha256_arquivo(ref.with_suffix('.parquet'))}
    codes = [Path(__file__), Path(mt.__file__), Path(mt.e1.__file__), Path(mt.e2.__file__), Path(mt.e3.__file__)]
    p = {"tipo": "MT06_protocolo_antes_dos_resultados", "quando_utc": agora_utc(), "config": CFG,
         "sucessao": "v1 só teve piloto de custo e estratificação legada; v1.1 preserva a população MT05 na primária antes dos novos treinos",
         "questao": "cobertura e solo acrescentam ordenação transferível além da física ou capturam ausência/seleção?",
         "bracos": BRACOS, "interacoes_fisicas": INTER, "primarias": PRIMARIAS,
         "secundarias": SECUNDARIAS, "unidades": sorted(d.unidade.unique()), "eventos": int(d.evento_id.nunique()),
         "linhas": len(d), "ajustes_previstos": len(BRACOS) * d.unidade.nunique(),
         "base_sha256": sha256_arquivo(SAIDA / "base.parquet"),
         "base_mt05_protocolo_sha256": sha256_arquivo(mt.SAIDA / "protocolo.json"),
         "codigo_sha256": {str(c.relative_to(REPO)): sha256_arquivo(c) for c in codes}, "destinos": olds,
         "penalidades": "herdadas do MT03 de cada treino; mesmas entre braços; não reotimizadas por braço",
         "primarias_familia": "três contrastes todos os componentes; IC de 1-0,05/3 por Bonferroni; margem +0,01 de AUC e IC exclui zero",
         "recortes_secundarios": "fluvial/enxurrada CEMS; construído <0,1 / [0,1;0,5) / >=0,5; solo observado/ausente; fontes separadas",
         "unidade_inferencial": "média igual de eventos no componente; fonte não cria réplicas; GFD tem um componente",
         "controle": "valores de construído/areia permutados separadamente dentro do evento de TREINO; máscara, físico e teste preservados",
         "regime": "desenvolvimento retrospectivo; unidades e falhas conhecidas; nenhum dado novo independente",
         "limites": ["predição de rótulos observados; não isola causa física nem erro de instrumento sem referência externa",
                    "mesma seleção histórica MT03; elevação amostral não entra mas seu filtro histórico permanece",
                    "secundárias exploratórias, sem confirmação independente",
                    "por_fonte é estratificação do teste territorial; não é retenção da fonte inteira",
                    "construído não classifica sozinho o processo urbano/pluvial",
                    "orçamento e dependência entre treinos limitam inferência; nenhum modelo promovido"],
         "reservas_lidas": False, "modelo_promovido": False}
    escrever(arq, p)
    escrever(SAIDA / "registro_inicio.json", {"quando_utc": agora_utc(), "protocolo_sha256": sha256_arquivo(arq), "resultados_mt06_lidos": False})
    return p


def conferir(p):
    if p["config"] != CFG or p["bracos"] != BRACOS:
        raise ValueError("configuração mudou; versionar")
    if sha256_arquivo(SAIDA / "base.parquet") != p["base_sha256"]:
        raise ValueError("base mudou")
    for file, h in p["codigo_sha256"].items():
        if sha256_arquivo(REPO / file) != h:
            raise ValueError("código mudou após congelamento: " + file)


def selecionar(d, un):
    tr0, te0 = d[d.unidade != un], d[d.unidade == un]
    mt.checar_retencao(tr0, te0)
    tr = mt.e1.amostrar(tr0, CFG["treino_por_evento_classe"], CFG["semente_amostra"])
    te = mt.e1.amostrar(te0, CFG["teste_por_evento_classe"], CFG["semente_amostra"])
    te = te.groupby("evento_id").filter(lambda g: len(g) >= 20 and g.classe.nunique() == 2)
    return tr, te


def ajuste(nome, tr, te, lam, xtr=None):
    x = tr if xtr is None else xtr
    inter = () if nome == "mascaras_so" else INTER
    cols = sorted(set(BRACOS[nome]) | {c for pair in inter for c in pair})
    m = mt.e1.Logistica(BRACOS[nome], inter,
                       lam_p=lam["p"], lam_i=lam["i"])
    t0 = time.monotonic()
    m.fit(x[cols], tr.classe.to_numpy(float), tr.evento_id.to_numpy())
    if not m.convergiu:
        raise ValueError("ajuste não convergiu: " + nome)
    score = m.escore(te[cols])
    weights = {"principais": dict(zip(m.principais, m.b.tolist())),
               "interacoes": [{"a": a, "b": b, "peso": float(w)} for (a, b), w in zip(m.interacoes, m.g)],
               "padronizacao": {"media": m.pad.mu.to_dict(), "desvio": m.pad.sd.to_dict(), "mediana": m.pad.med.to_dict()},
               "iteracoes": m.iteracoes, "segundos_relogio": time.monotonic() - t0}
    return score, weights


def piloto():
    p = preparar()
    d = pd.read_parquet(SAIDA / "base.parquet")
    un = p["unidades"][0]
    tr, te = selecionar(d, un)
    _, weights = ajuste("fisica_cobertura", tr, te, p["destinos"][un]["lam"])
    r = {"tipo": "piloto_custo_somente_treino_e_execucao", "destino_por_ordem": un,
         "segundos_um_ajuste": weights["segundos_relogio"], "projecao_horas": weights["segundos_relogio"] * p["ajustes_previstos"] / 3600,
         "linhas_treino": len(tr), "nenhuma_metrica_de_teste_lida": True,
         "limites_operacionais": "um processo, dois threads BLAS, menos de 1 GiB de novas saídas; sem rede"}
    escrever(SAIDA / "piloto.json", r)
    return r


def executar():
    p = preparar()
    conferir(p)
    d = pd.read_parquet(SAIDA / "base.parquet")
    p_hash = sha256_arquivo(SAIDA / "protocolo.json")
    for k, un in enumerate(p["unidades"]):
        path = SAIDA / "destinos" / (sha256_json(un)[7:23] + ".json")
        if path.exists():
            old = ler(path)
            if old["protocolo_sha256"] != p_hash or old["predicoes_sha256"] != sha256_arquivo(path.with_suffix('.parquet')):
                raise ValueError("retomada com checkpoint divergente")
            continue
        tr, te = selecionar(d, un)
        parent = p["destinos"][un]
        if sha256_arquivo(REPO / parent["mt03_json"]) != parent["mt03_sha256"]:
            raise ValueError("seleção de penalidades de origem mudou")
        refp = REPO / parent["mt05_predicao"]
        if sha256_arquivo(refp) != parent["mt05_predicao_sha256"]:
            raise ValueError("predição MT05 mudou")
        ref = pd.read_parquet(refp, columns=["ponto_id", "classe", "Q_conexao"])
        if not np.array_equal(ref.ponto_id, te.ponto_id) or not np.array_equal(ref.classe, te.classe):
            raise ValueError("teste diverge dos pontos MT05")
        pred = te[["ponto_id", "evento_id", "unidade", "classe", "fonte", "processo_ponto", "f_construido", "ausente_areia"]].copy()
        broken = romper_valores(tr, tr.evento_id.to_numpy(), CFG["semente_permutacao"])
        weights = {}
        for name in BRACOS:
            s, weights[name] = ajuste(name, tr, te, parent["lam"], broken if name.startswith("valores_rompidos") else None)
            pred[name] = s
        parity = float(np.max(np.abs(pred.fisica_cobertura.to_numpy() - ref.Q_conexao.to_numpy())))
        if parity > 1e-10:
            raise ValueError("braço completo não reproduz Q_conexao: " + str(parity))
        path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(path.with_suffix('.parquet'), index=False)
        r = {"destino": un, "protocolo_sha256": p_hash, "predicoes_sha256": sha256_arquivo(path.with_suffix('.parquet')),
             "treino_linhas": len(tr), "treino_componentes": tr.unidade.nunique(), "teste_linhas": len(te),
             "paridade_max_Q_conexao": parity, "pesos": weights, "medidas": medir(pred), "lam": parent["lam"]}
        escrever(path, r)
        print(json.dumps({"destino": un, "concluidos": k+1, "total": len(p["unidades"]),
                          "segundos": sum(w["segundos_relogio"] for w in weights.values())}), flush=True)
    return analisar()


def analisar():
    p = ler(SAIDA / "protocolo.json")
    conferir(p)
    rows = [ler(f) for f in sorted((SAIDA / "destinos").glob("*.json"))]
    if sorted(r["destino"] for r in rows) != p["unidades"]:
        raise ValueError("campanha incompleta; análise final recusada")
    m = pd.DataFrame([v for r in rows for v in r["medidas"]])
    m.to_parquet(SAIDA / "medidas.parquet", index=False)
    all_u = m[m.recorte == "todos"].groupby(["destino", "modelo"]).auc.mean().unstack()
    prim = {name: contraste(all_u, a, b, CFG["nivel_primarias"]) for name, (a, b) in PRIMARIAS.items()}
    sec = {}
    for rec, g in m.groupby("recorte"):
        u = g.groupby(["destino", "modelo"]).auc.mean().unstack()
        sec[rec] = {"componentes": len(u), "auc_media": u.mean().to_dict(),
                    "contrastes": {name: contraste(u, a, b) for name, (a, b) in SECUNDARIAS.items()}}
    fontes = {}
    for source, g in m[m.recorte == "todos"].groupby("fonte"):
        u = g.groupby(["destino", "modelo"]).auc.mean().unstack()
        fontes[source] = {"componentes": len(u), "contrastes": {name: contraste(u, a, b) for name, (a, b) in SECUNDARIAS.items()}}
    r = {"tipo": "MT06_resultado", "quando_utc": agora_utc(), "protocolo_sha256": sha256_arquivo(SAIDA / "protocolo.json"),
         "ajustes": len(rows)*len(BRACOS), "componentes": len(rows), "eventos": m.evento_id.nunique(),
         "primarias": prim, "secundarias": sec, "por_fonte": fontes,
         "paridade_Q_conexao_max": max(x["paridade_max_Q_conexao"] for x in rows),
         "segundos_relogio_somados": sum(w["segundos_relogio"] for row in rows for w in row["pesos"].values()),
         "regime": p["regime"], "limites": p["limites"], "modelo_promovido": False}
    escrever(SAIDA / "resultado.json", r)
    return {"ajustes": r["ajustes"], "primarias": {k: {q: v.get(q) for q in ('n', 'media', 'ic', 'veredito')} for k, v in prim.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["preparar", "piloto", "executar", "analisar"])
    args = ap.parse_args()
    print(json.dumps({"preparar": preparar, "piloto": piloto, "executar": executar, "analisar": analisar}[args.acao](),
                     ensure_ascii=False, default=str))
