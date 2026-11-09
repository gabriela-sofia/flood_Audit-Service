"""MT03: conexao com dominio a montante suficiente (A) e candidatas congeladas em unidades novas (B).

A -- DESENVOLVIMENTO nas mesmas 43 unidades do MT01/MT02 (eventos ja lidos), mesmo estimador, mesmas
     amostras, base ds06 v8 (= cobertura v2 do MT02). Bracos:
       G_v2          terreno + construido + areia (base)
       G_conexao     G_v2 + distancia a drenagem fina e a canal >= 10 km2 do RASTER LOCAL (MT02; trunca no recorte)
       G_rios10      G_v2 + distancia a rio HydroRIVERS com area a montante >= 10 km2 (+ interacao com relevo)
       G_rios100/1000/10000  mesmo formato, outros limiares: SENSIBILIDADE, nao selecao
       G_rios_porte  G_v2 + rio >= 10 e rio >= 1000 km2 (+ interacoes com relevo)
       G_upa         G_v2 + area a montante HydroSHEDS 15" no ponto (maximo 3x3)
     Recortes: todos, processo fluvial, processo enxurrada (pontos CEMS por obj_desc), estratos: unidade com
     recorte truncado (maior area local < 10% do maior rio HydroRIVERS da caixa; sem rotulo), plano.
     Metricas: AUC (primaria), AP/prevalencia, captura nos 10% e 20% de cima (orcamento de inspecao).

B -- candidatas ajustadas nas 43 unidades e CONGELADAS antes de ler pontos novos; avaliadas nas unidades
     CEMS de desenvolvimento da particao (triagem_cems/particao.json). Nao e confirmacao: a reserva fica
     fechada. Serve para (i) a conexao na enxurrada com mais unidades (9 antigas por deixa-um-de-fora + novas),
     (ii) validar a indicacao de suporte congelada em unidades que nao existiam no ajuste.

O ganho fluvial do MT02 e candidato: aqui ele e REPLICADO com outra representacao nas mesmas unidades
(desenvolvimento), sem escolher recorte depois de ver.

Uso:
    python scripts/metodo/mt03_experimento.py pre_registrar
    python scripts/metodo/mt03_experimento.py executar        (A)
    python scripts/metodo/mt03_experimento.py analisar        (A)
    python scripts/metodo/mt03_experimento.py congelar        (B: ajusta e grava candidatas)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

for nome in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(nome, "2")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402
import mt01_experimento as e1  # noqa: E402
import mt01_relatorio as rel  # noqa: E402
import mt02_experimento as e2  # noqa: E402
import mt03_ds06_v8 as v8  # noqa: E402

VERSAO = "mt03_experimento_v1"
SAIDA = e2.SAIDA.parent / "v3"
HS = SAIDA / "conexao" / "atributos_hydrosheds_mt01.parquet"
TRUNC = SAIDA / "conexao" / "truncamento_por_unidade.csv"
BASE = e2.TERRENO + ["f_construido", "f_areia"]
LIM = (10, 100, 1000, 10000)
BRACOS = {"G_v2": BASE, "G_conexao": e2.BRACOS["G_conexao"],
          **{f"G_rios{t}": BASE + [f"f_rio{t}"] for t in LIM},
          "G_rios_porte": BASE + ["f_rio10", "f_rio1000"], "G_upa": BASE + ["f_upa"]}
INTERACOES = {"G_conexao": e2.INTER_CONEXAO, **{f"G_rios{t}": [(f"f_rio{t}", "z_relevo")] for t in LIM},
              "G_rios_porte": [("f_rio10", "z_relevo"), ("f_rio1000", "z_relevo")]}
CONFIG = {**e2.CONFIG, "braco_lam_interacao": "G_rios10",
          "truncado_regra": ("razao mediana da unidade (maior area local / maior rio HydroRIVERS da caixa, por recorte) "
                             "abaixo da MEDIANA entre as 43 unidades; corte pela distribuicao sem rotulo, para estratos "
                             "equilibrados -- o corte de 10% poria quase todas as unidades no mesmo estrato")}
HIPOTESES = {
    "H3a_fluvial": ("G_rios10 - G_v2 > 0 (IC por unidade) no processo fluvial: o ganho do MT02 replica com rede "
                    "nao truncada; sinal esperado: mais perto do rio -> mais inundado"),
    "H3b_representacao": "G_rios10 - G_conexao >= -0,01 no geral e > 0 nas unidades truncadas",
    "H3c_enxurrada": ("G_rios10 - G_v2 na enxurrada: com 9 unidades espera-se INDECIDIVEL; decisao so com as "
                      "unidades novas de desenvolvimento somadas (B)"),
    "H3d_orcamento": "captura nos 10% de cima: G_rios10 - G_v2 > 0 no fluvial (o MT02 nao teve ganho no topo)",
    "limiar": "G_rios100/1000/10000 e G_rios_porte sao sensibilidade; nenhum limiar e escolhido pelo resultado",
    "upa": "G_upa - G_v2: area a montante no ponto, sem hipotese de sinal por processo (descritivo)",
}


def carregar() -> pd.DataFrame:
    d = e2.carregar()
    # migracao para o ds06 v8: as colunas de cobertura vem da base v8 e tem de bater com as do MT02
    b8 = pd.read_parquet(v8.BASE_V8 / "v8" / "pontos.parquet", columns=["ponto_id", "construido_frac", "areia_pct_0_30"])
    d = d.merge(b8.rename(columns={"construido_frac": "construido_v8", "areia_pct_0_30": "areia_v8"}),
                on="ponto_id", how="left")
    for a, b in (("construido_v2", "construido_v8"), ("areia_v2", "areia_v8")):
        if not ((d[a] == d[b]) | (d[a].isna() & d[b].isna())).all():
            raise RuntimeError(f"ds06 v8 diverge da cobertura do MT02 em {a}")
    h = pd.read_parquet(HS)
    d = d.merge(h, on="ponto_id", how="left")
    t = pd.read_csv(TRUNC)[["unidade_terreno", "razao_area_local"]]
    d = d.merge(t, on="unidade_terreno", how="left")
    return d.sort_values("ponto_id", kind="stable").reset_index(drop=True)


def caracteristicas(d: pd.DataFrame) -> pd.DataFrame:
    x = e2.caracteristicas(d)
    for t in LIM:
        x[f"f_rio{t}"] = np.log1p(d[f"dist_rio_{t}_m"])
    x["f_upa"] = np.log1p(d.upa_15s_max3_km2.clip(lower=0))
    return x


def ajustar(nome, x, d, lam):
    mod = e1.Logistica(BRACOS[nome], INTERACOES.get(nome, ()), lam_p=lam["p"], lam_i=lam["i"])
    return mod.fit(x, d.classe.to_numpy(float), d.evento_id.to_numpy())


def selecionar(d, x, semente) -> dict:
    unidades = np.array(sorted(d.unidade.unique()))
    dobra = dict(zip(unidades, np.random.default_rng(semente).permutation(len(unidades)) % CONFIG["dobras_internas"]))
    fd = d.unidade.map(dobra).to_numpy()

    def nota(nome, lam):
        v = []
        for k in range(CONFIG["dobras_internas"]):
            tr, te = fd != k, fd == k
            m = ajustar(nome, x[tr], d[tr], lam)
            v.append(e1.auc_media_por_unidade(d[te], m.escore(x[te])))
        return float(np.nanmean(v))

    lam, notas = {"p": .001, "i": 1.}, {}
    lam["p"] = max(CONFIG["grade_lam_principal"], key=lambda l: notas.setdefault(f"G_v2:{l}", nota("G_v2", {**lam, "p": l})))
    lam["i"] = max(CONFIG["grade_lam_interacao"],
                   key=lambda l: notas.setdefault(f"G_rios10:{l}", nota("G_rios10", {**lam, "i": l})))
    return {"lam": lam, "notas": notas}


def executar_destino(destino, d, x) -> dict:
    sem = CONFIG["semente"]
    alvo = d.unidade == destino
    tr = e1.amostrar(d[~alvo], CONFIG["teto_ajuste_por_evento_classe"], sem)
    te = e1.amostrar(d[alvo], CONFIG["teto_teste_por_evento_classe"], sem)
    ok = te.groupby("evento_id").filter(lambda g: len(g) >= CONFIG["minimo_teste_evento"] and g.classe.nunique() == 2)
    te = te[te.evento_id.isin(ok.evento_id.unique())]
    if te.empty:
        return {"destino": destino, "estado": "sem_evento_avaliavel"}
    xtr, xte = x.loc[tr.index], x.loc[te.index]
    sel = selecionar(tr, xtr, sem)
    escores = {"A0_regra_hand": -te.hand_m.to_numpy()}
    coef = {}
    for nome in BRACOS:
        m = ajustar(nome, xtr, tr, sel["lam"])
        escores[nome] = m.escore(xte)
        coef[nome] = {"b": dict(zip(m.principais, np.round(m.b, 6))),
                      "g": dict(zip([f"{a}:{b}" for a, b in m.interacoes], np.round(m.g, 6)))}
    pred = te[["ponto_id", "evento_id", "classe", "fonte", "processo_ponto"]].copy()
    for k, v in escores.items():
        pred[k] = v
    return {"destino": destino, "estado": "executado", "lam": sel["lam"], "notas": sel["notas"],
            "teste": {"eventos": int(te.evento_id.nunique()), "linhas": int(len(te)), "positivos": int(te.classe.sum())},
            "coeficientes": coef, "medidas": e2.medir(te, escores, destino), "_pred": pred}


def truncadas(d: pd.DataFrame) -> list:
    r = d.groupby(["unidade", "unidade_terreno"]).razao_area_local.first().groupby("unidade").median()
    return sorted(r[r < r.median()].index)


def fontes() -> list[Path]:
    return [HS, TRUNC, v8.BASE_V8 / "v8" / "pontos.parquet", Path(__file__), Path(e2.__file__), Path(e1.__file__),
            REPO / "scripts/metodo/mt03_conexao_hydrosheds.py", e2.ATRIB, e2.COB2]


def pre_registrar() -> dict:
    arq = SAIDA / "pre_registro_mt03.json"
    if arq.exists():
        raise FileExistsError(arq)
    d = carregar()
    trunc = pd.read_csv(TRUNC)
    pre2 = json.loads((e2.SAIDA / "pre_registro.json").read_text(encoding="utf-8"))
    pre = {"tipo": "mt03_pre_registro", "versao": VERSAO, "quando_utc": agora_utc(),
           "natureza": {"A": "desenvolvimento sobre eventos ja lidos (MT01/MT02)",
                        "B": "candidatas congeladas avaliadas em unidades novas de DESENVOLVIMENTO; reserva fechada"},
           "base": "ds06 v8 (ds-06-dataset-de-treino-cobertura-v2/v8), conferida contra a cobertura v2 do MT02",
           "populacao": {"linhas": int(len(d)), "unidades": int(d.unidade.nunique())},
           "unidades": sorted(d.unidade.unique()), "bracos": BRACOS,
           "interacoes": {k: [f"{a}:{b}" for a, b in v] for k, v in INTERACOES.items()},
           "config": CONFIG, "hipoteses": HIPOTESES,
           "estratos": {"plano": pre2["estratos"]["plano"], "truncado": truncadas(d),
                        "truncado_regra": CONFIG["truncado_regra"]},
           "truncamento_medido_antes": json.loads((SAIDA / "conexao" / "truncamento.json").read_text(encoding="utf-8")),
           "contrastes": [["G_rios10", "G_v2"], ["G_rios10", "G_conexao"], ["G_conexao", "G_v2"], ["G_rios_porte", "G_v2"],
                          ["G_upa", "G_v2"], ["G_rios100", "G_v2"], ["G_rios1000", "G_v2"], ["G_rios10000", "G_v2"],
                          ["G_v2", "A0_regra_hand"]],
           "metricas": "AUC por evento -> unidade -> media; IC bootstrap de unidades; margem 0,01; AP e captura secundarias",
           "B": {"candidatas": ["A0_regra_hand", "G_v2", "G_terreno", "G_conexao", "G_rios10"],
                 "ajuste": "todas as 43 unidades, teto 400 por evento x classe, penalidade por dobras de territorio",
                 "avaliacao": "unidades de desenvolvimento da particao triagem_cems (hash na particao)",
                 "enxurrada": ("conexao na enxurrada julgada com as 9 antigas (A, deixa-um-de-fora) + novas com "
                               "enxurrada (B); superioridade so com >= 8 unidades e IC acima de 0"),
                 "suporte": "candidata v3/suporte/candidata_congelada.json aplicada aos descritores das novas"},
           "fontes": {str(p.relative_to(REPO)): sha256_arquivo(p) for p in fontes()}}
    pre["hash_resumo"] = sha256_json({k: v for k, v in pre.items() if k != "quando_utc"})
    arq.write_text(json.dumps(pre, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return pre


def executar() -> None:
    pre = json.loads((SAIDA / "pre_registro_mt03.json").read_text(encoding="utf-8"))
    for n, h in pre["fontes"].items():
        if sha256_arquivo(REPO / n) != h:
            raise RuntimeError(f"alterado depois do pre-registro: {n}")
    d = carregar()
    x = caracteristicas(d)
    pasta = SAIDA / "destinos"
    pasta.mkdir(exist_ok=True)
    for k, dest in enumerate(pre["unidades"]):
        arq = pasta / f"{sha256_json(dest)[7:23]}.json"
        if arq.exists():
            continue
        t0 = time.time()
        r = executar_destino(dest, d, x)
        pred = r.pop("_pred", None)
        if pred is not None:
            pred.to_parquet(arq.with_suffix(".parquet"), index=False)
            r["predicoes_sha256"] = sha256_arquivo(arq.with_suffix(".parquet"))
        r.update({"segundos": round(time.time() - t0, 1), "pre_registro": pre["hash_resumo"]})
        arq.write_text(json.dumps(rel.limpo(r), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(k, dest, r["estado"], r["segundos"], flush=True)


def analisar() -> dict:
    pre = json.loads((SAIDA / "pre_registro_mt03.json").read_text(encoding="utf-8"))
    ok = [r for r in rel.carregar_destinos("../v3/destinos") if r["estado"] == "executado"]
    m = rel.medidas(ok)
    out = {"tipo": "mt03_resultado_A", "quando_utc": agora_utc(), "pre_registro": pre["hash_resumo"],
           "unidades_executadas": len(ok)}
    estratos = {"todos": None, "plano": pre["estratos"]["plano"], "truncado": pre["estratos"]["truncado"]}
    estratos["nao_truncado"] = sorted(set(pre["unidades"]) - set(pre["estratos"]["truncado"]))
    blocos = {}
    for nome, unid in estratos.items():
        mm = m[m.recorte == "todos"]
        if unid is not None:
            mm = mm[mm.destino.isin(unid)]
        blocos[nome] = mm
    for proc in ("fluvial", "enxurrada", "costeira"):
        blocos[f"processo:{proc}"] = m[m.recorte == f"processo:{proc}"]
    for nome, mm in blocos.items():
        u = rel.por_unidade(mm.drop(columns="recorte"))
        out[nome] = {"unidades": int(len(u)), "media": u.mean().round(4).to_dict(),
                     "contrastes": {f"{a}-{b}": rel.contraste(u, a, b, "superioridade", CONFIG["margem_auc"],
                                                              CONFIG["bootstrap"]) for a, b in pre["contrastes"]
                                    if a in u and b in u and u[[a, b]].dropna().shape[0] >= 3}}
    secundarias = metricas_secundarias(ok)
    for (rec, met), g in secundarias.groupby(["recorte", "metrica"]):
        u = g.groupby(["destino", "modelo", "evento_id"]).valor.mean().groupby(["destino", "modelo"]).mean().unstack()
        out[f"{met}|{rec}"] = {"unidades": int(len(u)), "media": u.mean().round(4).to_dict(),
                               **{f"{a}-{b}": rel.contraste(u, a, b, "superioridade", 0, CONFIG["bootstrap"])
                                  for a, b in (("G_rios10", "G_v2"), ("G_conexao", "G_v2"), ("G_rios10", "G_conexao"))
                                  if a in u and b in u and len(u) >= 3}}
    out["coeficientes"] = {}
    for nome in ("G_rios10", "G_rios_porte", "G_upa", "G_conexao"):
        cf = pd.DataFrame([{**r["coeficientes"][nome]["b"], **r["coeficientes"][nome]["g"]} for r in ok])
        out["coeficientes"][nome] = {c: {"mediana": float(cf[c].median()), "positivos": int((cf[c] > 0).sum()),
                                         "n": int(cf[c].notna().sum())} for c in cf}
    out["por_unidade"] = rel.por_unidade(m[m.recorte == "todos"].drop(columns="recorte")).round(4).reset_index().to_dict("records")
    out = rel.limpo(out)
    out["hash_resumo"] = sha256_json({k: v for k, v in out.items() if k != "quando_utc"})
    (SAIDA / "resultado_A.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def metricas_secundarias(ok: list) -> pd.DataFrame:
    """AP/prevalencia e captura no topo, no evento inteiro e no recorte de processo (positivos do processo
    contra todos os negativos do evento), recalculadas das previsoes gravadas; empate por limiar/fracao."""
    from sklearn.metrics import average_precision_score
    linhas = []
    for r in ok:
        arq = SAIDA / "destinos" / f"{sha256_json(r['destino'])[7:23]}.parquet"
        p = pd.read_parquet(arq)
        for ev, g in p.groupby("evento_id"):
            recortes = {"todos": g}
            if (g.fonte == "cems").all():
                for proc in ("fluvial", "enxurrada"):
                    sub = g[(g.classe == 0) | (g.processo_ponto == proc)]
                    if (sub.classe == 1).sum() >= 10 and (sub.classe == 0).sum() >= 10:
                        recortes[f"processo:{proc}"] = sub
            for rec, sub in recortes.items():
                y = sub.classe.to_numpy()
                for mod in ("A0_regra_hand", "G_v2", "G_conexao", "G_rios10", "G_rios_porte", "G_upa"):
                    s_ = sub[mod].to_numpy()
                    linhas += [{"destino": r["destino"], "evento_id": ev, "recorte": rec, "modelo": mod,
                                "metrica": "ap_sobre_prevalencia", "valor": average_precision_score(y, s_) / y.mean()},
                               {"destino": r["destino"], "evento_id": ev, "recorte": rec, "modelo": mod,
                                "metrica": "captura_topo_10", "valor": e2.captura(y, s_, .1)},
                               {"destino": r["destino"], "evento_id": ev, "recorte": rec, "modelo": mod,
                                "metrica": "captura_topo_20", "valor": e2.captura(y, s_, .2)}]
    return pd.DataFrame(linhas)


# ------------------------------------------------------------- B: candidatas congeladas

CANDIDATAS = ["G_v2", "G_terreno", "G_conexao", "G_rios10"]


def congelar() -> dict:
    arq = SAIDA / "candidatas_congeladas.pkl"
    if arq.exists():
        raise FileExistsError(arq)
    pre = json.loads((SAIDA / "pre_registro_mt03.json").read_text(encoding="utf-8"))
    d = carregar()
    x = caracteristicas(d)
    sem = CONFIG["semente"]
    tr = e1.amostrar(d, CONFIG["teto_ajuste_por_evento_classe"], sem)
    xtr = x.loc[tr.index]
    sel = selecionar(tr, xtr, sem)
    modelos = {}
    for nome in CANDIDATAS:
        cols, inter = (e2.BRACOS["G_terreno"], ()) if nome == "G_terreno" else (BRACOS[nome], INTERACOES.get(nome, ()))
        modelos[nome] = e1.Logistica(cols, inter, lam_p=sel["lam"]["p"], lam_i=sel["lam"]["i"]).fit(
            xtr, tr.classe.to_numpy(float), tr.evento_id.to_numpy())
    with arq.open("wb") as f:
        pickle.dump({"modelos": modelos, "lam": sel["lam"], "notas": sel["notas"]}, f)
    meta = {"tipo": "mt03_candidatas_congeladas", "quando_utc": agora_utc(), "pre_registro": pre["hash_resumo"],
            "lam": sel["lam"], "linhas_ajuste": int(len(tr)), "unidades": int(tr.unidade.nunique()),
            "coeficientes": {n: {"b": dict(zip(m.principais, np.round(m.b, 6))),
                                 "g": dict(zip([f"{a}:{b}" for a, b in m.interacoes], np.round(m.g, 6)))}
                             for n, m in modelos.items()},
            "pkl_sha256": sha256_arquivo(arq),
            "nao_e": "candidatas de desenvolvimento; nenhuma servida; avaliadas so em unidades de desenvolvimento"}
    (SAIDA / "candidatas_congeladas.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float),
                                                      encoding="utf-8")
    return meta


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=("pre_registrar", "executar", "analisar", "congelar"))
    a = ap.parse_args()
    r = {"pre_registrar": pre_registrar, "executar": executar, "analisar": analisar, "congelar": congelar}[a.acao]()
    if r:
        print(json.dumps({k: v for k, v in r.items() if k not in ("fontes", "unidades", "por_unidade")},
                         ensure_ascii=False, indent=1, default=str)[:7000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
