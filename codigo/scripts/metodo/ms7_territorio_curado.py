"""MS7: quadro territorial curado, treino por evento e lugar, sem piso amostral.

Experimento novo de desenvolvimento: nao substitui MS6, nao abre reserva e
nao promove pesos. AUC dentro de evento e bootstrap de grupos conectados
por tile respondem perguntas diferentes; ambos ficam registrados.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for nome in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(nome, "1")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
for pasta in ("servico", "metodo", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
import generalizacao_por_reservas as gen  # noqa: E402
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402

BASE = ("hand_m", "twi_dinf", "slope_deg")
VETOR = BASE + ("pos_min_2km",)
SEMENTE = 20261008
BRACOS = {"T0_hand": ("regra", BASE, None),
          "T1_base_vizinhanca": ("logistica", BASE, "vizinhanca"),
          "T2_base_territorio": ("logistica", BASE, "territorio"),
          "T3_pos_vizinhanca": ("logistica", VETOR, "vizinhanca"),
          "T4_pos_territorio": ("logistica", VETOR, "territorio"),
          "T5_gbm_territorio": ("gbm", VETOR, "territorio")}
COMPARACOES = (("Q1", "T2_base_territorio", "T1_base_vizinhanca"),
               ("Q2", "T4_pos_territorio", "T3_pos_vizinhanca"),
               ("Q3", "T4_pos_territorio", "T2_base_territorio"),
               ("Q4", "T2_base_territorio", "T0_hand"),
               ("Q5", "T5_gbm_territorio", "T4_pos_territorio"))


def tile_geografico(lon, lat) -> np.ndarray:
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()
            and (np.abs(lon) < 180).all() and (np.abs(lat) < 90).all()):
        raise ValueError("coordenada invalida para tile geografico")
    return ((np.floor(lat) + 90) * 360 + np.floor(lon) + 180).astype(np.int32)


def ler_territorio(arq: Path, eventos: list[str], lote: int = 32768) -> tuple[dict, dict]:
    """Le so o vetor da tarefa. Estados desconhecidos nunca entram como classe 0."""
    partes = {k: [] for k in ("X", "y", "evento", "tile")}
    codigos = {e: i for i, e in enumerate(eventos)}
    contagens = {"linhas": 0, "observadas": 0, "elegiveis": 0, "perda_por_classe": {"0": 0, "1": 0}}
    for batch in pq.ParquetFile(arq).iter_batches(batch_size=lote, columns=["evento_id", "estado", "lon", "lat", *VETOR]):
        d = batch.to_pandas()
        contagens["linhas"] += len(d)
        d = d[d.estado.isin(("positivo", "negativo_observado"))]
        contagens["observadas"] += len(d)
        ok = np.isfinite(d[list(VETOR)].to_numpy(float)).all(1)
        for classe, estado in (("0", "negativo_observado"), ("1", "positivo")):
            contagens["perda_por_classe"][classe] += int((~ok & (d.estado == estado)).sum())
        d = d[ok]
        if not set(d.evento_id) <= set(codigos):
            raise ValueError("evento territorial nao consta do desenvolvimento conhecido")
        partes["X"].append(d[list(VETOR)].to_numpy(np.float32))
        partes["y"].append((d.estado == "positivo").to_numpy(np.uint8))
        partes["evento"].append(d.evento_id.map(codigos).to_numpy(np.int16))
        partes["tile"].append(tile_geografico(d.lon, d.lat))
        contagens["elegiveis"] += len(d)
    if not contagens["elegiveis"]:
        raise ValueError("territorio sem observacoes elegiveis")
    return {k: np.concatenate(v) for k, v in partes.items()}, contagens


def separar(quadros: dict, teste_eventos: np.ndarray) -> tuple[dict, np.ndarray]:
    te = np.isin(quadros["territorio"]["evento"], teste_eventos)
    tiles = np.unique(quadros["territorio"]["tile"][te])
    tr = {q: np.flatnonzero(~np.isin(d["evento"], teste_eventos) & ~np.isin(d["tile"], tiles))
          for q, d in quadros.items()}
    return tr, np.flatnonzero(te)


def ajustar(tipo: str, vetor: tuple, X: np.ndarray, y: np.ndarray):
    colunas = [VETOR.index(v) for v in vetor]
    if tipo == "regra":
        return lambda x: -x[:, 0], {"regra": "-hand_m"}
    if len(np.unique(y)) != 2:
        raise ValueError("treino sem duas classes depois do corte")
    x = X[:, colunas].astype(float)
    if tipo == "gbm":
        m = HistGradientBoostingClassifier(random_state=SEMENTE, **gen.GBM).fit(x, y)
        return lambda z: m.predict_proba(z[:, colunas])[:, 1], {"tipo": "gbm", "parametros": gen.GBM}
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1
    b = gen.logistica_com_intercepto((x - mu) / sd, y.astype(float))
    if b is None:
        raise RuntimeError("logistica sem ajuste finito")
    return (lambda z: expit(b[0] + ((z[:, colunas] - mu) / sd) @ b[1:]),
            {"tipo": "logistica", "vetor": list(vetor), "media": mu.tolist(), "desvio": sd.tolist(),
             "intercepto": float(b[0]), "coeficientes": b[1:].tolist(),
             "nao_e": "coeficiente condicional do estimador, nao efeito causal"})


def grupos_conectados(evento: np.ndarray, tile: np.ndarray) -> dict[int, int]:
    """Eventos que compartilham lugar sao uma unidade no bootstrap conservador."""
    eventos = np.unique(evento).tolist()
    pai = {e: e for e in eventos}
    def raiz(e):
        while pai[e] != e:
            pai[e] = pai[pai[e]]
            e = pai[e]
        return e
    donos = {}
    for e in eventos:
        for t in np.unique(tile[evento == e]):
            if int(t) in donos:
                pai[raiz(e)] = raiz(donos[int(t)])
            else:
                donos[int(t)] = e
    return {e: raiz(e) for e in eventos}


def comparar_eventos(a: dict, b: dict, grupos: dict, rng) -> dict:
    comuns = sorted(set(a) & set(b))
    dif = np.array([a[e] - b[e] for e in comuns])
    if not len(dif):
        return {"eventos_pareados": 0, "veredito": "not_testable"}
    medias = rng.choice(dif, (5000, len(dif))).mean(1)
    por_grupo = {}
    for e, valor in zip(comuns, dif):
        por_grupo.setdefault(grupos[e], []).append(float(valor))
    blocos = list(por_grupo.values())
    medias_bloco = []
    for _ in range(5000):
        sorteio = rng.integers(len(blocos), size=len(blocos))
        medias_bloco.append(float(np.mean([v for i in sorteio for v in blocos[i]])))
    r = {"eventos_pareados": len(comuns), "eventos_a_favor": int((dif > 0).sum()),
         "dentro_do_evento_mediana": round(float(np.median(dif)), 4),
         "ic95_media_dentro": np.percentile(medias, [2.5, 97.5]).round(4).tolist(),
         "grupos_espaciais_conectados": len(blocos),
         "ic95_media_por_grupo": np.percentile(medias_bloco, [2.5, 97.5]).round(4).tolist()}
    if np.any(dif != 0):
        r["p_wilcoxon"] = float(wilcoxon(dif).pvalue)
    lo, hi = r["ic95_media_por_grupo"]
    r["veredito"] = ("not_testable" if len(blocos) < 8 else "ganho" if lo > 0.01 else
                     "perda" if hi < -0.01 else "sem_diferenca_relevante" if -0.01 <= lo and hi <= 0.01
                     else "indecidivel")
    r["limite"] = "exploratorio; bootstrap agrupado nao torna eventos conhecidos em confirmacao"
    return r


def executar(manifesto: Path, saida: Path) -> dict:
    import fechamento_do_framework as fech
    import posicao_no_ponto as pp
    from ds06_dataset_de_treino import versao_atual
    manifesto, saida = Path(manifesto).resolve(), Path(saida).resolve()
    cur = json.loads(manifesto.read_text(encoding="utf-8"))
    pontos = manifesto.parent / "pontos.parquet"
    if cur.get("pontos_sha256") != sha256_arquivo(pontos):
        raise RuntimeError("curadoria divergente")
    posman = json.loads((pp.SAIDA / "manifesto.json").read_text(encoding="utf-8"))
    if fech.verificar_posicao(posman, REPO)["estado"] != "verificado":
        raise RuntimeError("posicao no ponto sem linhagem atual")
    origens = {"curadoria": sha256_arquivo(manifesto), "pontos_curados": sha256_arquivo(pontos),
               "ds06": sha256_arquivo(versao_atual() / "pontos.parquet"),
               "posicao": sha256_arquivo(pp.SAIDA / "pontos.parquet"),
               "codigo": sha256_arquivo(Path(__file__)), "estimador": sha256_arquivo(Path(gen.__file__))}
    criterios = {"origens": origens, "regime": "exploratorio", "vetor": list(VETOR),
                 "primario": "Q2 no territorio curado", "comparacoes": COMPARACOES,
                 "dobras": "5 por evento a1; retirar do treino todo tile geografico do teste",
                 "orcamento": "mesmo numero de linhas por origem/dobra, minimo dos elegiveis, teto 60000",
                 "incerteza": "5000 bootstraps por evento e por componente conectado evento/tile",
                 "regra": ">=8 grupos; IC da media por grupo inteiramente alem de +/-0.01 de AUC",
                 "visto_antes": "MS3/MS4/MS5, falha da replica RFD05 e 19255 repeticoes TC01; MS6 sem resultado",
                 "semente": SEMENTE, "bracos": BRACOS}
    criterios = json.loads(json.dumps(criterios))
    destino = saida / sha256_json(criterios).split(":")[1]
    destino.mkdir(parents=True, exist_ok=True)
    pre = destino / "pre_registro.json"
    if pre.exists():
        if json.loads(pre.read_text(encoding="utf-8"))["criterios"] != criterios:
            raise RuntimeError("protocolo mudou")
    else:
        pre.write_text(json.dumps({"hash": sha256_json(criterios), "criterios": criterios,
                                   "quando_utc": agora_utc()}, indent=2) + "\n", encoding="utf-8")
    if (destino / "resultado.json").exists():
        return json.loads((destino / "resultado.json").read_text(encoding="utf-8"))
    inicio = time.time()
    d, _, filtro = gen.carregar(BASE)
    v = pp.juntar(d[d.fonte == "gfd"])
    v = v[np.isfinite(v[list(VETOR)].to_numpy(float)).all(1)]
    eventos = sorted(v.evento_id.unique())
    cod = {e: i for i, e in enumerate(eventos)}
    viz = {"X": v[list(VETOR)].to_numpy(np.float32), "y": v.classe.to_numpy(np.uint8),
           "evento": v.evento_id.map(cod).to_numpy(np.int16), "tile": tile_geografico(v.lon, v.lat)}
    del d, v
    ter, perda = ler_territorio(pontos, eventos)
    quadros = {"territorio": ter, "vizinhanca": viz}
    atribuicao = gen.dobras(eventos, 5, "a1")
    rng = np.random.default_rng(SEMENTE)
    pred = {k: np.full(len(ter["y"]), np.nan, np.float32) for k in BRACOS}
    apoio, modelos = [], {}
    for f in range(5):
        teste_ev = np.array([cod[e] for e in eventos if atribuicao[e] == f])
        indices, teste = separar(quadros, teste_ev)
        n = min(60000, *(len(i) for i in indices.values()))
        if n < 20 or not len(teste):
            raise RuntimeError("suporte insuficiente apos exclusao espacial")
        amostras = {q: rng.choice(i, n, replace=False) for q, i in indices.items()}
        apoio.append({"dobra": f, "linhas_teste": len(teste), "linhas_ajuste_por_origem": n,
                      "eventos_teste": [eventos[e] for e in teste_ev], "por_origem": {
                          q: {"disponiveis": len(i), "positivos_no_ajuste": int(quadros[q]["y"][amostras[q]].sum()),
                              "tiles_compartilhados": len(set(quadros[q]["tile"][i]) & set(ter["tile"][teste])),
                              "eventos_compartilhados": len(set(quadros[q]["evento"][i]) & set(teste_ev))}
                          for q, i in indices.items()}})
        for nome, (tipo, vetor, origem) in BRACOS.items():
            q = quadros[origem] if origem else viz
            tr = amostras[origem] if origem else np.arange(len(viz["y"]))
            fn, card = ajustar(tipo, vetor, q["X"][tr], q["y"][tr])
            modelos[f"{f}/{nome}"] = card
            for pos in range(0, len(teste), 32768):
                idx = teste[pos:pos + 32768]
                pred[nome][idx] = fn(ter["X"][idx])
        print(f"dobra={f} linhas_ajuste={n} teste={len(teste)} segundos={round(time.time()-inicio)}", flush=True)
    medidas, aucs = {}, {}
    for nome, p in pred.items():
        if not np.isfinite(p).all():
            raise RuntimeError("previsoes fora de dobra incompletas")
        pe = {}
        for e in np.unique(ter["evento"]):
            m = ter["evento"] == e
            y = ter["y"][m]
            if len(y) >= 20 and len(np.unique(y)) == 2:
                pe[int(e)] = float(roc_auc_score(y, p[m]))
        aucs[nome] = pe
        medidas[nome] = {"auc_agrupada": float(roc_auc_score(ter["y"], p)),
                         "average_precision": float(average_precision_score(ter["y"], p)),
                         "auc_mediana_dentro": float(np.median(list(pe.values()))), "eventos_com_auc": len(pe),
                         "auc_por_evento": {eventos[e]: a for e, a in pe.items()},
                         "brier": None if nome == "T0_hand" else float(brier_score_loss(ter["y"], p))}
    grupos = grupos_conectados(ter["evento"], ter["tile"])
    comparacoes = {cid: {"a": a, "b": b, **comparar_eventos(aucs[a], aucs[b], grupos, rng)}
                   for cid, a, b in COMPARACOES}
    if origens["pontos_curados"] != sha256_arquivo(pontos):
        raise RuntimeError("dados mudaram durante experimento")
    r = {"tipo": "ms7_territorio_curado", "regime": "exploratorio", "pre_registro": sha256_json(criterios),
         "origens": origens, "perdas": perda, "populacao_vizinhanca": filtro,
         "suporte_por_dobra": apoio, "medidas": medidas, "comparacoes": comparacoes,
         "modelos_de_desenvolvimento": modelos, "promovido": False, "rotulos_novos_lidos": False,
         "segundos": round(time.time()-inicio), "destino": str(destino.relative_to(REPO)),
         "nao_e": "nao e confirmacao independente, risco, probabilidade de ocorrencia ou eficacia causal"}
    r["hash_resumo"] = sha256_json(r)
    r["quando_utc"] = agora_utc()
    (destino / "resultado.json").write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(destino / "predicoes.npz", evento=ter["evento"], classe=ter["y"], **pred)
    print(f"GRAVADO={destino / 'resultado.json'}", flush=True)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--curadoria", type=Path, required=True)
    ap.add_argument("--saida", type=Path, required=True)
    args = ap.parse_args()
    r = executar(args.curadoria, args.saida)
    print(json.dumps({k: v["veredito"] for k, v in r["comparacoes"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
