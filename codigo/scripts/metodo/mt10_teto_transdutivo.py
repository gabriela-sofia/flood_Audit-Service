"""MT10: a observação SAR do próprio destino, sem rótulo, carrega a folga que a física não alcança?

Contexto (MT08, registrado antes): o que é compartilhável entre lugares já está no Q_conexao
(logística global que viu o destino: +0,0001 em blocos; árvores só com outros lugares: -0,003).
A folga que existe só aparece com informação do próprio destino. Em todas as campanhas
conjuntas a tarefa auxiliar foi retida junto com o componente, então nenhuma delas podia
levar essa informação. Resta uma via: supervisão SEM rótulo de inundação observada NO
destino (desenho transdutivo). Antes de treinar uma rede assim, este teste mede o limite
de informação dessa via: a frequência de água SAR do pixel fora das janelas de evento
(pilha S1, `freq_fora`), que é exatamente o que uma tarefa auxiliar no destino ensinaria.

Braços (mesmos pontos de teste do MT06 que caem em bloco da pilha com pixel observável):
  Q          Q_conexao congelado (MT06)
  F_transf   logística [Q, F, Q x F] com intercepto por evento, ajustada nos OUTROS
             componentes e aplicada ao destino: usa a observação do destino, nunca o rótulo
  F_so       só F, sinal e escala aprendidos nos outros componentes
  F_local    [Q, F, Q x F] ajustada dentro do destino, dobras em bloco de 0,05 grau (teto)
Primária: F_transf - Q, 20 componentes utilizáveis (regra E1/v1 registrada em 17/10 na
pilha), IC95% por componente, margem +0,01. Secundárias: pontos sem água JRC (occ = 0),
F_so - Q, F_local - Q.

Leitura pré-registrada:
  sustentado      a observação do destino tem a informação específica do lugar; o
                  benefício que o aprendizado conjunto procurava existe, mas é da
                  OBSERVAÇÃO DO DESTINO, não do compartilhamento entre lugares
  nao_sustentado  nem a história SAR do destino a carrega: o benefício físico conjunto
                  com as observações disponíveis fica fechado negativamente
  Ressalva fixada antes: o rótulo CEMS é delimitado por analista sobre S1/óptico; ganho de
  F pode ser erro compartilhado de instrumento e não física. Por isso a secundária sem
  água JRC e a leitura por família ficam declaradas.

Uso:
    python scripts/manutencao/executar_serializado.py -- python scripts/metodo/mt10_teto_transdutivo.py executar
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

import mt06_cobertura_observabilidade as e
import mt08_teto_compartilhado as t8

sys.path.insert(0, str(e.REPO / "scripts/sar"))
import pilha_s1_mesmo_lugar as pilha  # noqa: E402

SAIDA = e.REPO / "local_runs/mecanismos-transferiveis/v10/teto_transdutivo_v1"
BRACOS = ["Q", "F_transf", "F_so", "F_local"]
CFG = {"lam_p": .01, "lam_i": .01, "n_fora_min": 10, "dobras": 5, "semente_dobras": 20261017, "grau_bloco": .05,
       "bootstrap": 2000, "semente_bootstrap": 7, "nivel": .95, "margem_auc": .01, "minimo_componentes": 8,
       "pixel_m": 30.}


def regra_por_componente():
    q = pilha.qualidade_da_referencia()
    return {c: ("e1" if c in q["e1"] else "v1") for c in q["alguma"]}, sorted(q["nenhuma"]) if "nenhuma" in q else []


def pixel(lat, lon, b, transf):
    x, y = transf.transform(lon, lat)
    col = np.floor((x - b["x0"]) / CFG["pixel_m"]).astype(int)
    lin = np.floor((b["y0"] - y) / CFG["pixel_m"]).astype(int)
    ok = (lin >= 0) & (lin < b["altura"]) & (col >= 0) & (col < b["largura"])
    return lin, col, ok


def extrair():
    """F em cada ponto de teste CEMS do MT06 que cai num bloco da pilha."""
    regra, _ = regra_por_componente()
    coords = pd.read_parquet(t8.COORDENADAS).set_index("ponto_id")
    blocos = e.ler(pilha.SAIDA / "blocos.json")["blocos"]
    pai = e.ler(e.SAIDA / "protocolo.json")
    rows = []
    for un in pai["unidades"]:
        if not un.startswith("CEMS:"):
            continue
        pred = pd.read_parquet(e.REPO / pai["destinos"][un]["mt05_predicao"], columns=["ponto_id", "evento_id", "classe", "Q_conexao"])
        c = coords.loc[pred.ponto_id]
        lat, lon = c.lat.to_numpy(), c.lon.to_numpy()
        achado = np.zeros(len(pred), bool)
        for b in (b for b in blocos if b["componente"] == un):
            transf = Transformer.from_crs("EPSG:4326", b["crs"], always_xy=True)
            lin, col, ok = pixel(lat, lon, b, transf)
            ok &= ~achado
            if not ok.any():
                continue
            fv1 = np.load(pilha.SAIDA / "frequencia" / f"{b['bloco']}.npz", allow_pickle=True)
            if abs(float(fv1["x0"]) - b["x0"]) > 1e-6 or abs(float(fv1["y0"]) - b["y0"]) > 1e-6:
                raise ValueError("origem do bloco diverge da frequência: " + b["bloco"])
            pil = np.load(pilha.SAIDA / "pilhas" / f"{b['bloco']}.npz", allow_pickle=True)
            fe1 = np.load(pilha.SAIDA / "frequencia_e1" / f"{b['bloco']}.npz", allow_pickle=True)
            i = np.flatnonzero(ok)
            for k, r_, c_ in zip(i, lin[i], col[i]):
                rows.append({"ponto_id": pred.ponto_id.iloc[k], "evento_id": pred.evento_id.iloc[k], "unidade": un,
                             "classe": int(pred.classe.iloc[k]), "Q": float(pred.Q_conexao.iloc[k]),
                             "lat": lat[k], "lon": lon[k], "bloco": b["bloco"],
                             "classe_pixel": int(fv1["classe"][r_, c_]), "n_fora": int(fv1["n_fora"][r_, c_]),
                             "F_v1": float(fv1["freq_fora"][r_, c_]), "F_e1": float(fe1["freq_fora"][r_, c_]),
                             "jrc_occ": float(pil["jrc_occ"][r_, c_])})
            achado[i] = True
    t = pd.DataFrame(rows)
    t["regra"] = t.unidade.map(regra)
    t["F"] = np.where(t.regra == "e1", t.F_e1, t.F_v1)
    t["utilizavel"] = t.regra.notna()
    return t


def preparar():
    arq = SAIDA / "protocolo.json"
    base = SAIDA / "pontos.parquet"
    codigos = [Path(__file__), Path(t8.__file__), Path(e.__file__), Path(pilha.__file__)]
    if not base.exists():
        SAIDA.mkdir(parents=True, exist_ok=True)
        extrair().to_parquet(base, index=False)
    p = {"tipo": "MT10_protocolo_antes_dos_ajustes", "config": CFG, "bracos": BRACOS,
         "pontos_sha256": e.sha256_arquivo(base), "mt08_protocolo_sha256": e.sha256_arquivo(t8.SAIDA / "protocolo.json"),
         "codigo_sha256": {str(c.relative_to(e.REPO)): e.sha256_arquivo(c) for c in codigos},
         "populacao_primaria": "componentes com referência SAR utilizável pela regra de 17/10 (E1 onde passa, senão v1); "
                               "pixel observável (classe 1) e n_fora >= 10",
         "primaria": "F_transf - Q", "leitura": "ver docstring", "motivacao_mt08_lida": True,
         "regime": "desenvolvimento retrospectivo; limite de informação da observação do destino, não causalidade",
         "reservas_lidas": False, "modelo_promovido": False}
    if arq.exists():
        old = e.ler(arq)
        if {k: v for k, v in old.items() if k != "quando_utc"} != e.mt.rel.limpo(p):
            raise ValueError("protocolo mudou; versionar")
        return old
    p["quando_utc"] = e.agora_utc()
    e.escrever(arq, p)
    return p


def _modelo(tr, cols, inter):
    m = e.mt.e1.Logistica(cols, inter, lam_p=CFG["lam_p"], lam_i=CFG["lam_i"])
    m.fit(tr[cols], tr.classe.to_numpy(float), tr.evento_id.to_numpy())
    if not m.convergiu:
        raise ValueError("não convergiu")
    return m


def escores(t):
    out = []
    for un in sorted(t.unidade.unique()):
        tr, te = t[t.unidade != un], t[t.unidade == un].copy()
        te["F_transf"] = _modelo(tr, ["Q", "F"], [("Q", "F")]).escore(te[["Q", "F"]])
        te["F_so"] = _modelo(tr, ["F"], []).escore(te[["F"]])
        dobra = t8.dobras_bloco(te.lat.to_numpy(), te.lon.to_numpy(), CFG["dobras"], CFG["semente_dobras"], CFG["grau_bloco"])
        s = np.full(len(te), np.nan)
        for k in np.unique(dobra):
            fit = te[dobra != k]
            if fit.classe.nunique() < 2:
                continue
            s[dobra == k] = _modelo(fit, ["Q", "F"], [("Q", "F")]).escore(te[dobra == k][["Q", "F"]])
        te["F_local"] = np.where(np.isfinite(s), s, te.Q)  # dobra sem as duas classes: cai na referência
        out.append(te)
    return pd.concat(out)


def contraste(u, a, b):
    d = (u[a] - u[b]).dropna()
    r = {"n": len(d), "media": float(d.mean()) if len(d) else None, "a_favor": int((d > 0).sum()), "por_componente": d.to_dict()}
    if len(d) < CFG["minimo_componentes"]:
        return {**r, "veredito": "not_testable"}
    boot = np.random.default_rng(CFG["semente_bootstrap"]).choice(d.to_numpy(), (CFG["bootstrap"], len(d))).mean(1)
    lo, hi = np.quantile(boot, [(1 - CFG["nivel"]) / 2, 1 - (1 - CFG["nivel"]) / 2])
    v = "sustentado" if r["media"] >= CFG["margem_auc"] and lo > 0 else ("nao_sustentado" if hi < CFG["margem_auc"] else "indecidivel")
    return {**r, "ic": [float(lo), float(hi)], "veredito": v}


def medir(s, rotulo):
    rows = []
    for (un, ev), g in s.groupby(["unidade", "evento_id"]):
        if len(g) < 20 or g.classe.nunique() < 2:
            continue
        for b in BRACOS:
            rows.append({"destino": un, "evento_id": ev, "recorte": rotulo, "modelo": b,
                         "auc": e.mt.e1.auc(g.classe.to_numpy(), g[b].to_numpy())})
    return rows


def executar():
    p = preparar()
    t = pd.read_parquet(SAIDA / "pontos.parquet")
    if e.sha256_arquivo(SAIDA / "pontos.parquet") != p["pontos_sha256"]:
        raise ValueError("pontos mudaram")
    t = t[(t.classe_pixel == 1) & (t.n_fora >= CFG["n_fora_min"]) & t.F.notna()]
    recortes = {"utilizaveis": t[t.utilizavel], "utilizaveis_sem_jrc": t[t.utilizavel & (t.jrc_occ == 0)]}
    v1 = t.assign(F=t.F_v1)
    recortes["todos_v1"] = v1
    med, cobertura = [], {}
    for nome, sub in recortes.items():
        cobertura[nome] = {"pontos": len(sub), "componentes": int(sub.unidade.nunique()), "eventos": int(sub.evento_id.nunique())}
        s = escores(sub)
        s.to_parquet(SAIDA / f"escores_{nome}.parquet", index=False)
        med += medir(s, nome)
    tab = pd.DataFrame(med)
    tab.to_parquet(SAIDA / "medidas.parquet", index=False)
    res = {}
    for rec, g in tab.groupby("recorte"):
        u = g.groupby(["destino", "modelo"]).auc.mean().unstack()
        res[rec] = {"componentes": len(u), "auc_media": u.mean().to_dict(),
                    "contrastes": {f"{a}-Q": contraste(u, a, "Q") for a in BRACOS[1:]}}
    r = {"tipo": "MT10_resultado", "quando_utc": e.agora_utc(), "protocolo_sha256": e.sha256_arquivo(SAIDA / "protocolo.json"),
         "cobertura": cobertura, "primaria": res["utilizaveis"]["contrastes"]["F_transf-Q"], "recortes": res,
         "regime": p["regime"], "modelo_promovido": False}
    e.escrever(SAIDA / "resultado.json", r)
    return {"cobertura": cobertura, "primaria": {k: r["primaria"].get(k) for k in ("n", "media", "ic", "veredito")},
            "secundarias": {rec: {k: (v["media"], v.get("ic"), v["veredito"]) for k, v in x["contrastes"].items()} for rec, x in res.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["preparar", "executar"])
    a = ap.parse_args()
    print(json.dumps({"preparar": preparar, "executar": executar}[a.acao](), ensure_ascii=False, default=str))
