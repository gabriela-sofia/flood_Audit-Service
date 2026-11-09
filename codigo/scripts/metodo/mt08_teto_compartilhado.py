"""MT08: teto de informação do aprendizado físico compartilhado.

Pergunta: quanto uma função QUALQUER das mesmas entradas pré-evento (terreno,
conexão, cobertura) consegue, no máximo, melhorar a ordenação física nos 43
componentes retidos do MT05/MT06?

Por que isso fecha a pergunta do aprendizado conjunto: em todas as campanhas
conjuntas (v1, frequência, temporal, O3 e a projeção de gradientes v4) a cabeça
física só recebe uma função dessas entradas. A tarefa auxiliar pode mudar QUAL
função é aprendida, mas não acrescenta entrada. Logo o ganho físico de qualquer
variante sobre a parada é limitado por quanto a melhor função das entradas ordena
no próprio destino. Este experimento mede esse limite, em vez de testar mais uma
variante.

Braços, todos avaliados nos MESMOS pontos de teste do MT06 (mesmo sorteio):
  Q_conexao   referência; predição congelada do MT06 (= MT05)
  T_gbm       árvores com reforço treinadas só nos outros componentes: o que mais
              flexibilidade extrai SEM informação do destino
  P_*         ajuste conjunto de todos os componentes MAIS dobras do próprio
              destino (validação cruzada dentro do destino): a melhor função
              compartilhada que já viu rótulos deste lugar
  L_*         ajuste só no destino (validação cruzada): a melhor função
              específica do lugar com estas entradas
Dobras do destino em dois esquemas: aleatória estratificada por evento e classe
(vizinhos espaciais nas duas partes: teto otimista, conservador para a afirmação
de teto baixo) e blocos espaciais de 0,05 grau (sensibilidade).

Leitura pré-registrada (antes dos ajustes):
  teto_local  = max(P_*, L_*) - Q_conexao   por componente; limite absoluto
  teto_global = max(P_*) - Q_conexao         limite do que é compartilhável
  Se o limite superior do IC de teto_global < +0,01, nenhuma representação
  compartilhada destas entradas entrega o ganho físico relevante, qualquer que
  seja a tarefa auxiliar ou o otimizador. Se teto_local também < +0,01, as
  entradas estão no teto: só entrada/observação nova move a física.

Regime: desenvolvimento retrospectivo nos mesmos 43 componentes; não lê reservas,
não promove modelo, não altera os módulos congelados do MT06/MT07.

Uso (sempre pela trava):
    python scripts/manutencao/executar_serializado.py -- python scripts/metodo/mt08_teto_compartilhado.py executar
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import mt06_cobertura_observabilidade as e
from verificar_fechamento_independente import auc_pares

SAIDA = e.REPO / "local_runs/mecanismos-transferiveis/v8/teto_compartilhado_v1"
COORDENADAS = SAIDA / "coordenadas.parquet"
COLUNAS_CONTEXTO = ["ponto_id", "lat", "lon", "instrumento", "worldcover_2021", "mecanismo",
                    "evento_tipo", "processo_evento", "hand_m", "dist_canal10_m", "slope_deg"]
GBM_COLS = sorted(set(e.FULL) | {"z_relevo"})
LIN_COLS = sorted(set(e.FULL) | {c for pair in e.INTER for c in pair})
ESQUEMAS = ("aleatorio", "bloco")
BRACOS = ["Q_conexao", "T_gbm"] + [f"{t}_{m}_{s}" for s in ESQUEMAS for t in ("P", "L") for m in ("lin", "gbm")]
CFG = {"dobras": 5, "semente_dobras": 20261017, "grau_bloco": .05,
       "gbm": {"max_iter": 200, "learning_rate": .1, "max_leaf_nodes": 31, "min_samples_leaf": 40,
               "l2_regularization": 1., "early_stopping": False, "random_state": 0},
       "bootstrap": 2000, "semente_bootstrap": 7, "nivel": .975, "margem_auc": .01,
       "minimo_componentes": 8, "threads_blas": 2}
LEITURA = {
    "teto_abaixo_da_margem": "limite superior do IC < margem: nenhuma função destas entradas desse tipo entrega o ganho relevante",
    "folga_sustentada": "média >= margem e limite inferior > 0: existe folga que um método melhor poderia capturar",
    "indecidivel": "IC cruza a margem",
}


# ------------------------------------------------------------------ dobras e pesos

def dobras_aleatorias(te: pd.DataFrame, k: int, semente: int) -> np.ndarray:
    """Estratificada por evento x classe: cada estrato espalha-se pelas k dobras."""
    rng = np.random.default_rng(semente)
    f = np.empty(len(te), int)
    pos = {i: n for n, i in enumerate(te.index)}
    for _, g in te.groupby(["evento_id", "classe"], sort=True):
        idx = np.array([pos[i] for i in g.index])
        f[idx[rng.permutation(len(idx))]] = np.arange(len(idx)) % k
    return f


def dobras_bloco(lat: np.ndarray, lon: np.ndarray, k: int, semente: int, grau: float) -> np.ndarray:
    """Blocos espaciais (independentes do evento) sorteados para as dobras."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("coordenada ausente no teste")
    chave = pd.Series(list(zip(np.floor(lat / grau).astype(int), np.floor(lon / grau).astype(int))))
    blocos, inv = np.unique(chave.astype(str), return_inverse=True)
    ordem = np.random.default_rng(semente).permutation(len(blocos))
    return (ordem % k)[inv]


def pesos_evento_classe(y, evento) -> np.ndarray:
    """O mesmo objetivo da Logística MT01: cada evento pesa 1, cada classe metade; média 1."""
    y = np.asarray(y, float)
    _, ev = np.unique(np.asarray(evento), return_inverse=True)
    npos = np.bincount(ev, weights=y)
    nneg = np.bincount(ev, weights=1 - y)
    w = np.where(y == 1, .5 / np.maximum(npos[ev], 1), .5 / np.maximum(nneg[ev], 1))
    return w * len(w) / w.sum()


# ------------------------------------------------------------------ ajustes

def ajustar_gbm(tr: pd.DataFrame, cfg=None):
    m = HistGradientBoostingClassifier(**(cfg or CFG["gbm"]))
    m.fit(tr[GBM_COLS].to_numpy(float), tr.classe.to_numpy(int),
          sample_weight=pesos_evento_classe(tr.classe, tr.evento_id))
    return lambda x: m.predict_proba(x[GBM_COLS].to_numpy(float))[:, 1]


def ajustar_lin(tr: pd.DataFrame, lam: dict):
    m = e.mt.e1.Logistica(e.FULL, e.INTER, lam_p=lam["p"], lam_i=lam["i"])
    m.fit(tr[LIN_COLS], tr.classe.to_numpy(float), tr.evento_id.to_numpy())
    if not m.convergiu:
        raise ValueError("logística não convergiu")
    return lambda x: m.escore(x[LIN_COLS])


def validacao_cruzada(tr, te, dobra, lam, modelo, com_outros):
    """Escore fora da dobra para cada ponto do destino."""
    s = np.full(len(te), np.nan)
    for k in np.unique(dobra):
        fit = te[dobra != k] if not com_outros else pd.concat([tr, te[dobra != k]])
        if fit.classe.nunique() < 2:
            raise ValueError("dobra de treino sem as duas classes")
        f = ajustar_gbm(fit) if modelo == "gbm" else ajustar_lin(fit, lam)
        s[dobra == k] = f(te[dobra == k])
    if not np.isfinite(s).all():
        raise ValueError("escore não finito")
    return s


# ------------------------------------------------------------------ protocolo

def preparar():
    arq = SAIDA / "protocolo.json"
    if arq.exists():
        p = e.ler(arq)
        conferir(p)
        return p
    pai = e.ler(e.SAIDA / "protocolo.json")
    if not COORDENADAS.exists():
        d = e.mt.e3.carregar()
        SAIDA.mkdir(parents=True, exist_ok=True)
        d[COLUNAS_CONTEXTO].to_parquet(COORDENADAS, index=False)
    v4 = e.REPO / "local_runs/aprendizado-conjunto/v4_prioridade_fisica"
    codigos = [Path(__file__), Path(e.__file__), Path(e.mt.__file__), Path(e.mt.e1.__file__)]
    p = {"tipo": "MT08_protocolo_antes_dos_ajustes", "quando_utc": e.agora_utc(), "config": CFG,
         "questao": "qual o maior ganho físico que QUALQUER função das mesmas entradas pode dar nos 43 componentes?",
         "parent_mt06_sha256": e.sha256_arquivo(e.SAIDA / "protocolo.json"),
         "base_sha256": pai["base_sha256"], "coordenadas_sha256": e.sha256_arquivo(COORDENADAS),
         "unidades": pai["unidades"], "destinos": pai["destinos"], "bracos": BRACOS,
         "entradas_gbm": GBM_COLS, "entradas_lin": LIN_COLS, "interacoes": e.INTER,
         "codigo_sha256": {str(c.relative_to(e.REPO)): e.sha256_arquivo(c) for c in codigos},
         "primarias": {"teto_global_aleatorio": "max(P_lin, P_gbm) - Q_conexao, dobras aleatórias",
                       "teto_local_aleatorio": "max(P_lin, P_gbm, L_lin, L_gbm) - Q_conexao, dobras aleatórias"},
         "secundarias": "mesmos tetos com dobras em bloco; cada braço isolado; T_gbm - Q_conexao; recortes MT06",
         "leitura": LEITURA,
         "otimismo_declarado": "max por componente escolhido depois de ver o teste e dobras aleatórias com vizinhos: "
                               "superestimam o teto; por isso um teto BAIXO é afirmação conservadora e um teto ALTO não é",
         "predicao_registrada": "se teto_global tiver limite superior < +0,01, toda variante conjunta sobre estas "
                                "entradas (inclusive v4_prioridade_fisica, em execução) deve ficar sem ganho relevante",
         "v4_prioridade_fisica_resultado_existia": (v4 / "resultado.json").exists() or (v4 / "analise.json").exists(),
         "regime": "desenvolvimento retrospectivo; limite de informação das entradas, não causalidade nem confirmação",
         "limites": ["teto é da FAMÍLIA de funções testada (logística Q_conexao e árvores com reforço); "
                     "uma família mais rica poderia, em princípio, achar mais",
                     "AUC dentro do evento; não mede calibração nem probabilidade territorial",
                     "mesma população histórica MT03/MT05; componentes conhecidos"],
         "reservas_lidas": False, "modelo_promovido": False}
    e.escrever(arq, p)
    e.escrever(SAIDA / "registro_inicio.json", {"quando_utc": e.agora_utc(), "protocolo_sha256": e.sha256_arquivo(arq),
                                                 "resultados_mt08_lidos": False})
    return p


def conferir(p):
    if p["config"] != e.mt.rel.limpo(CFG) or p["bracos"] != BRACOS:
        raise ValueError("configuração mudou; versionar")
    if e.sha256_arquivo(e.SAIDA / "base.parquet") != p["base_sha256"]:
        raise ValueError("base mudou")
    if e.sha256_arquivo(COORDENADAS) != p["coordenadas_sha256"]:
        raise ValueError("coordenadas mudaram")
    for f, h in p["codigo_sha256"].items():
        if e.sha256_arquivo(e.REPO / f) != h:
            raise ValueError("código mudou após congelamento: " + f)


# ------------------------------------------------------------------ execução

def um_destino(d, coords, un, parent):
    tr, te = e.selecionar(d, un)
    ref = pd.read_parquet(e.REPO / parent["mt05_predicao"], columns=["ponto_id", "classe", "Q_conexao"])
    if e.sha256_arquivo(e.REPO / parent["mt05_predicao"]) != parent["mt05_predicao_sha256"]:
        raise ValueError("referência Q mudou")
    if not np.array_equal(te.ponto_id, ref.ponto_id) or not np.array_equal(te.classe, ref.classe):
        raise ValueError("pontos de teste divergem do MT06")
    c = coords.loc[te.ponto_id]
    folds = {"aleatorio": dobras_aleatorias(te, CFG["dobras"], CFG["semente_dobras"]),
             "bloco": dobras_bloco(c.lat.to_numpy(), c.lon.to_numpy(), CFG["dobras"], CFG["semente_dobras"], CFG["grau_bloco"])}
    pred = te[["ponto_id", "evento_id", "unidade", "classe", "fonte", "processo_ponto", "f_construido", "ausente_areia"]].copy()
    pred["Q_conexao"] = ref.Q_conexao.to_numpy()
    seg = {}
    t0 = time.monotonic()
    pred["T_gbm"] = ajustar_gbm(tr)(te)
    seg["T_gbm"] = time.monotonic() - t0
    for s, dobra in folds.items():
        pred[f"dobra_{s}"] = dobra
        for tipo, outros in (("P", True), ("L", False)):
            for modelo in ("lin", "gbm"):
                t0 = time.monotonic()
                pred[f"{tipo}_{modelo}_{s}"] = validacao_cruzada(tr, te, dobra, parent["lam"], modelo, outros)
                seg[f"{tipo}_{modelo}_{s}"] = time.monotonic() - t0
    return pred, seg, {"treino_linhas": len(tr), "teste_linhas": len(te),
                       "pontos_por_dobra_bloco": np.bincount(folds["bloco"], minlength=CFG["dobras"]).tolist()}


def medir(pred):
    rows = []
    for ev, event in pred.groupby("evento_id"):
        for rec, g in e.estratos(event):
            if not e.elegivel(g, rec):
                continue
            y = g.classe.to_numpy()
            for name in BRACOS:
                rows.append({"destino": str(g.unidade.iloc[0]), "evento_id": ev, "recorte": rec, "modelo": name,
                             "n": len(g), "auc": float(e.mt.e1.auc(y, g[name].to_numpy()))})
    return rows


def executar():
    p = preparar()
    d = pd.read_parquet(e.SAIDA / "base.parquet")
    coords = pd.read_parquet(COORDENADAS).set_index("ponto_id")
    h = e.sha256_arquivo(SAIDA / "protocolo.json")
    for i, un in enumerate(p["unidades"]):
        path = SAIDA / "destinos" / (e.sha256_json(un)[7:23] + ".json")
        if path.exists():
            ck = e.ler(path)
            if ck["protocolo_sha256"] != h or ck["predicoes_sha256"] != e.sha256_arquivo(path.with_suffix(".parquet")):
                raise ValueError("checkpoint divergente")
            continue
        pred, seg, info = um_destino(d, coords, un, p["destinos"][un])
        path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(path.with_suffix(".parquet"), index=False)
        e.escrever(path, {"destino": un, "protocolo_sha256": h,
                         "predicoes_sha256": e.sha256_arquivo(path.with_suffix(".parquet")),
                         "medidas": medir(pred), "segundos": seg, **info})
        print(json.dumps({"MT08": f"{i+1}/{len(p['unidades'])}", "destino": un, "segundos": round(sum(seg.values()), 1)}), flush=True)
    return analisar()


# ------------------------------------------------------------------ análise

def contraste_teto(dif: pd.Series) -> dict:
    d = dif.dropna()
    r = {"n": len(d), "media": float(d.mean()) if len(d) else None, "mediana": float(d.median()) if len(d) else None,
         "componentes_acima_da_margem": int((d >= CFG["margem_auc"]).sum()), "por_componente": d.to_dict()}
    if len(d) < CFG["minimo_componentes"]:
        return {**r, "veredito": "not_testable"}
    boot = np.random.default_rng(CFG["semente_bootstrap"]).choice(d.to_numpy(), (CFG["bootstrap"], len(d))).mean(1)
    q = (1 - CFG["nivel"]) / 2
    lo, hi = np.quantile(boot, [q, 1 - q])
    v = "teto_abaixo_da_margem" if hi < CFG["margem_auc"] else (
        "folga_sustentada" if r["media"] >= CFG["margem_auc"] and lo > 0 else "indecidivel")
    return {**r, "ic": [float(lo), float(hi)], "nivel_ic": CFG["nivel"], "veredito": v}


def tetos(u: pd.DataFrame) -> dict:
    """u: componente x braço (AUC média igual de eventos). Devolve os contrastes nomeados."""
    out = {}
    for s in ESQUEMAS:
        P = [f"P_lin_{s}", f"P_gbm_{s}"]
        L = [f"L_lin_{s}", f"L_gbm_{s}"]
        out[f"teto_global_{s}"] = contraste_teto(u[P].max(axis=1) - u.Q_conexao)
        out[f"teto_local_{s}"] = contraste_teto(u[P + L].max(axis=1) - u.Q_conexao)
        for b in P + L:
            out[f"{b}-Q_conexao"] = contraste_teto(u[b] - u.Q_conexao)
    out["T_gbm-Q_conexao"] = contraste_teto(u.T_gbm - u.Q_conexao)
    return out


def analisar():
    p = preparar()
    rows = [e.ler(f) for f in sorted((SAIDA / "destinos").glob("*.json"))]
    if sorted(r["destino"] for r in rows) != p["unidades"]:
        raise ValueError("campanha incompleta; análise recusada")
    tab = pd.DataFrame([m for r in rows for m in r["medidas"]])
    tab.to_parquet(SAIDA / "medidas.parquet", index=False)
    max_dif = 0.
    for r in rows:
        path = SAIDA / "destinos" / (e.sha256_json(r["destino"])[7:23] + ".parquet")
        if e.sha256_arquivo(path) != r["predicoes_sha256"]:
            raise ValueError("hash divergente")
        pred = pd.read_parquet(path)
        groups = {ev: g for ev, g in pred.groupby("evento_id")}
        for m in r["medidas"]:
            sub = dict(e.estratos(groups[m["evento_id"]]))[m["recorte"]]
            max_dif = max(max_dif, abs(auc_pares(sub.classe.to_numpy(), sub[m["modelo"]].to_numpy()) - m["auc"]))
    if max_dif > 1e-12:
        raise ValueError("AUC independente diverge")
    recs = {}
    for rec, g in tab.groupby("recorte"):
        u = g.groupby(["destino", "modelo"]).auc.mean().unstack()
        recs[rec] = {"componentes": len(u), "auc_media": u.mean().to_dict(), "contrastes": tetos(u)}
    todos = recs["todos"]["contrastes"]
    result = {"tipo": "MT08_resultado", "quando_utc": e.agora_utc(), "componentes": len(rows),
              "ajustes": len(rows) * (1 + len(ESQUEMAS) * 4 * CFG["dobras"]),
              "primarias": {k: todos[k] for k in ("teto_global_aleatorio", "teto_local_aleatorio")},
              "recortes": recs, "max_dif_auc_independente": max_dif,
              "segundos_relogio_somados": sum(sum(r["segundos"].values()) for r in rows),
              "leitura": LEITURA, "regime": p["regime"], "limites": p["limites"], "modelo_promovido": False}
    e.escrever(SAIDA / "resultado.json", result)
    return {"ajustes": result["ajustes"],
            "primarias": {k: {q: v.get(q) for q in ("n", "media", "ic", "veredito", "componentes_acima_da_margem")}
                          for k, v in result["primarias"].items()}}


def piloto():
    """Custo de um destino pequeno e de um grande; nenhuma métrica de teste é lida."""
    p = preparar()
    d = pd.read_parquet(e.SAIDA / "base.parquet")
    coords = pd.read_parquet(COORDENADAS).set_index("ponto_id")
    out = {}
    for un in ("CEMS:EMSR778",):
        t0 = time.monotonic()
        _, seg, info = um_destino(d, coords, un, p["destinos"][un])
        out[un] = {"segundos": seg, "total": time.monotonic() - t0, **info}
    out["nenhuma_metrica_de_teste_lida"] = True
    e.escrever(SAIDA / "piloto.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["preparar", "piloto", "executar", "analisar"])
    a = ap.parse_args()
    print(json.dumps({"preparar": preparar, "piloto": piloto, "executar": executar, "analisar": analisar}[a.acao](),
                     ensure_ascii=False, default=str))
