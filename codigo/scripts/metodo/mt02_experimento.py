"""MT02: base corrigida, blocos de conexao e de saida, e indicacao de suporte -- mesmas unidades e pontos do MT01.

Desenvolvimento sobre eventos ja lidos. Mesmas 43 unidades, mesmos pontos, mesmas sementes e
mesmo estimador (`mt01_experimento.Logistica`). O que muda entre bracos e so a matriz de atributos:

  A0_regra_hand        -HAND
  G_v1                 G do MT01 (cobertura do cob01, com o defeito de versao)
  G_v2                 G com a cobertura v2 (cob-02-cobertura-v7)
  G_terreno            so acumulacao + conexao do MT01 (invariante a correcao: controle)
  G_v2_sem_construido  ablacao
  G_v2_sem_solo        ablacao
  G_v2_disp            G_v2 + indicador de mascara do solo: SONDA de atalho, nunca candidata
  G_conexao            G_v2 + distancia a drenagem e a canal >= 10 km2 + interacoes com relevo
  G_saida              G_v2 + distancia ao mar pelo rio + cota absoluta + interacoes com proximidade do mar
  G_completo           G_v2 + conexao + saida

Avaliacao: AUC dentro do evento (postos medios), AP (scikit-learn, empate por limiar) e captura no
topo com empate fracionado; por processo no ponto (CEMS); no subconjunto com solo observado; nos
eventos com linhas recoletadas. Estratos de dificuldade (pre-registrados AQUI, vindos do pos-hoc do
MT01, portanto desenvolvimento): territorio plano (HAND p90 do raster < 10 m) e processo enxurrada.

Nada e promovido. Uso:
    python scripts/metodo/mt02_experimento.py pre_registrar
    python scripts/metodo/mt02_experimento.py executar
"""
from __future__ import annotations

import argparse
import json
import os
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
import mt01_populacao as mp  # noqa: E402

VERSAO = "mt02_experimento_v1"
SAIDA = mp.RUNS / "mecanismos-transferiveis" / "v2"
COB2 = mp.RUNS / "cob-02-cobertura-v7" / "cobertura_e_solo_v2.parquet"
ATRIB = SAIDA / "atributos_conexao_saida.parquet"
TERRENO = ["f_hand", "f_elev", "f_sca", "f_declive"]
CONEXAO = ["f_dist_dren", "f_dist_canal"]
SAIDA_ATR = ["f_mar", "f_cota"]
INTER_CONEXAO = [("f_dist_dren", "z_relevo"), ("f_dist_canal", "z_relevo"), ("f_sca", "z_relevo")]
INTER_SAIDA = [("f_hand", "z_mar"), ("f_cota", "z_mar")]
BRACOS = {
    "G_v1": TERRENO + ["f_construido_v1", "f_areia_v1"],
    "G_v2": TERRENO + ["f_construido", "f_areia"],
    "G_terreno": TERRENO,
    "G_v2_sem_construido": TERRENO + ["f_areia"],
    "G_v2_sem_solo": TERRENO + ["f_construido"],
    "G_v2_disp": TERRENO + ["f_construido", "f_areia", "solo_mascarado"],
    "G_conexao": TERRENO + ["f_construido", "f_areia"] + CONEXAO,
    "G_saida": TERRENO + ["f_construido", "f_areia"] + SAIDA_ATR,
    "G_completo": TERRENO + ["f_construido", "f_areia"] + CONEXAO + SAIDA_ATR,
}
INTERACOES = {"G_conexao": INTER_CONEXAO, "G_saida": INTER_SAIDA, "G_completo": INTER_CONEXAO + INTER_SAIDA}
CONFIG = {**{k: e1.CONFIG[k] for k in ("semente", "teto_ajuste_por_evento_classe", "teto_teste_por_evento_classe",
                                        "minimo_teste_evento", "dobras_internas", "margem_auc", "bootstrap")},
          "grade_lam_principal": [0.0001, 0.001, 0.01, 0.1], "grade_lam_interacao": [0.1, 1.0, 10.0, 100.0],
          "plano_hand_p90_m": 10.0, "endorreico_dist_mar_km": 5000.0}
HIPOTESES = {
    "correcao": "G_v2 - G_v1: efeito da correcao de versao; esperado pequeno no geral e maior nos eventos recoletados",
    "representacao": "G_v2 - G_terreno: cobertura/solo nao identificam producao; e ganho preditivo, nao mecanismo",
    "ablacoes": "G_v2_sem_construido e G_v2_sem_solo contra G_v2",
    "sonda": "G_v2_disp - G_v2 > 0 mede atalho de disponibilidade; nunca vira candidata",
    "conexao": ("G_conexao - G_v2 > 0 na enxurrada (AUC por processo) e no geral nao-inferior; sinal esperado: "
                "mais longe da drenagem -> menos inundado, mais forte em relevo alto"),
    "saida": ("G_saida - G_v2 > 0 nos territorios planos; sinal esperado: perto do mar pelo rio, cota absoluta "
              "baixa pesa mais (f_cota:z_mar negativo); no geral nao-inferior"),
    "preservacao": "nos territorios nao planos, G_completo - G_v2 >= -0,01 (IC)",
}


def carregar() -> pd.DataFrame:
    d = e1.carregar()  # mesmas linhas, unidades e contexto do MT01
    c = pd.read_parquet(COB2, columns=["ponto_id", "construido_frac", "areia_pct_0_30", "motivo_areia_pct_0_30",
                                        "origem"])
    c = c.rename(columns={"construido_frac": "construido_v2", "areia_pct_0_30": "areia_v2"})
    a = pd.read_parquet(ATRIB)
    d = d.merge(c, on="ponto_id", how="left").merge(a, on="ponto_id", how="left")
    return d.sort_values("ponto_id", kind="stable").reset_index(drop=True)


def caracteristicas(d: pd.DataFrame) -> pd.DataFrame:
    x = e1.caracteristicas(d)
    x = x.rename(columns={"f_construido": "f_construido_v1", "f_areia": "f_areia_v1"})
    x["f_construido"] = d.construido_v2
    x["f_areia"] = d.areia_v2 / 100
    x["solo_mascarado"] = (d.motivo_areia_pct_0_30 == "mascara_da_camada").astype(float)
    x["f_dist_dren"] = np.log1p(d.dist_drenagem_m)
    x["f_dist_canal"] = np.log1p(d.dist_canal10_m)
    mar = d.dist_mar_rio_km.where(~d.sumidouro_endorreico.fillna(False).astype(bool), CONFIG["endorreico_dist_mar_km"])
    x["f_mar"] = np.log1p(mar)
    x["z_mar"] = -np.log1p(mar)  # proximidade do mar pelo rio (modificador)
    x["f_cota"] = np.log1p(d.elevation_m.clip(lower=0))
    return x


def ajustar(nome, x, d, lam):
    mod = e1.Logistica(BRACOS[nome], INTERACOES.get(nome, ()), lam_p=lam["p"], lam_i=lam["i"])
    return mod.fit(x, d.classe.to_numpy(float), d.evento_id.to_numpy())


def selecionar(d, x, semente) -> dict:
    """Penalidade principal por G_v2; de interacao por G_completo. Dobras internas por territorio de treino."""
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
                   key=lambda l: notas.setdefault(f"G_completo:{l}", nota("G_completo", {**lam, "i": l})))
    return {"lam": lam, "notas": notas}


def captura(y, s, f):
    o = np.argsort(-s, kind="stable")
    s, y = s[o], y[o]
    k = int(np.floor(f * len(s)))
    if k == 0 or y.sum() == 0:
        return float("nan")
    corte = s[k - 1]
    acima, emp = s > corte, s == corte
    return float((y[acima].sum() + y[emp].sum() * (k - acima.sum()) / emp.sum()) / y.sum())


def medir(te: pd.DataFrame, escores: dict, destino: str) -> list:
    from sklearn.metrics import average_precision_score
    out = []
    for nome, s in escores.items():
        t = te.assign(_s=s)
        for ev, g in t.groupby("evento_id"):
            y, sc = g.classe.to_numpy(), g._s.to_numpy()
            base = {"destino": destino, "modelo": nome, "evento_id": ev, "fonte": g.fonte.iloc[0]}
            out.append({**base, "recorte": "todos", "auc": e1.auc(y, sc),
                        "ap_sobre_prevalencia": average_precision_score(y, sc) / y.mean(),
                        "captura_topo_10": captura(y, sc, .1), "captura_topo_20": captura(y, sc, .2)})
            ok = g[g.motivo_areia_pct_0_30 == "valor"]
            if ok.classe.nunique() == 2 and len(ok) >= 20:
                out.append({**base, "recorte": "solo_observado", "auc": e1.auc(ok.classe.to_numpy(), ok._s.to_numpy())})
            rec = g[(g.origem == "recoleta_v2") | (g.classe == 1)]
            if (g.origem == "recoleta_v2").any() and rec.classe.nunique() == 2:
                out.append({**base, "recorte": "negativos_recoletados", "auc": e1.auc(rec.classe.to_numpy(), rec._s.to_numpy())})
        cems = t[t.fonte == "cems"]
        for (ev, proc), g in cems[cems.classe == 1].groupby(["evento_id", "processo_ponto"]):
            neg = cems[(cems.evento_id == ev) & (cems.classe == 0)]
            if len(g) >= 10 and len(neg) >= 10:
                yy = np.r_[np.ones(len(g)), np.zeros(len(neg))]
                out.append({"destino": destino, "modelo": nome, "evento_id": ev, "fonte": "cems",
                            "recorte": f"processo:{proc}", "auc": e1.auc(yy, np.r_[g._s, neg._s])})
    return out


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
            "teste": {"eventos": int(te.evento_id.nunique()), "linhas": int(len(te)),
                      "positivos": int(te.classe.sum()), "fontes": sorted(set(te.fonte))},
            "coeficientes": coef, "medidas": medir(te, escores, destino), "_pred": pred}


def fontes() -> list[Path]:
    return [COB2, ATRIB, mp.SAIDA / "populacao.parquet", mp.SAIDA / "unidades.parquet",
            mp.SAIDA / "contexto_unidades.parquet", Path(__file__), Path(e1.__file__), Path(mp.__file__),
            REPO / "scripts/metodo/mt02_cobertura_v2.py", REPO / "scripts/metodo/mt02_atributos.py"]


def pre_registrar() -> dict:
    arq = SAIDA / "pre_registro.json"
    if arq.exists():
        raise FileExistsError("pre-registro ja existe")
    d = carregar()
    planos = d.groupby("unidade").hand_p90.median()
    pre = {"tipo": "mt02_pre_registro", "versao": VERSAO, "quando_utc": agora_utc(),
           "natureza": "desenvolvimento sobre eventos ja lidos (MT01); nao e confirmacao",
           "populacao": {"linhas": int(len(d)), "eventos": int(d.evento_id.nunique()), "unidades": int(d.unidade.nunique()),
                         "recoletados": int((d.origem == "recoleta_v2").sum())},
           "unidades": sorted(d.unidade.unique()), "bracos": BRACOS,
           "interacoes": {k: [f"{a}:{b}" for a, b in v] for k, v in INTERACOES.items()},
           "config": CONFIG, "hipoteses": HIPOTESES,
           "estratos": {"plano": sorted(planos[planos < CONFIG["plano_hand_p90_m"]].index),
                        "plano_origem": "corte de 10 m escolhido no pos-hoc do MT01: estrato de desenvolvimento, nao regra",
                        "enxurrada": "positivos CEMS com obj_desc Flash flood contra os negativos do evento"},
           "producao": ("nao identificavel: forcamento quase constante dentro do evento; cobertura e solo isolados "
                        "nao identificam producao de escoamento"),
           "contrastes": [["G_v2", "G_v1"], ["G_v2", "G_terreno"], ["G_v2_sem_construido", "G_v2"],
                          ["G_v2_sem_solo", "G_v2"], ["G_v2_disp", "G_v2"], ["G_conexao", "G_v2"],
                          ["G_saida", "G_v2"], ["G_completo", "G_v2"], ["G_v2", "A0_regra_hand"]],
           "metricas": "AUC por evento -> unidade -> media; IC bootstrap de unidades; margem 0,01; minimo 8 unidades",
           "fontes": {str(p.relative_to(REPO)): sha256_arquivo(p) for p in fontes()}}
    pre["hash_resumo"] = sha256_json({k: v for k, v in pre.items() if k != "quando_utc"})
    mp.escrever(arq, pre)
    return pre


def executar() -> None:
    pre = json.loads((SAIDA / "pre_registro.json").read_text(encoding="utf-8"))
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
        mp.escrever(arq, r)
        print(k, dest, r["estado"], r["segundos"], flush=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("acao", choices=("pre_registrar", "executar"))
    a = ap.parse_args()
    if a.acao == "pre_registrar":
        p = pre_registrar()
        print(json.dumps(p["populacao"]), p["hash_resumo"], p["estratos"]["plano"])
    else:
        executar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
