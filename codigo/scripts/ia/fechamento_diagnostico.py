"""Etapa A do fechamento: o que o acervo temporal e os ajustes salvos dizem antes de treinar de novo.

Linhagem `fechamento_aprendizado_transferencia_v1` (17/10/2026). Le artefatos ja gravados;
nao treina e nao le reserva. Quatro diagnosticos:

  1 alinhamento   grade da pilha x grade ter-01; datas de produto CEMS x passagens S1 da pilha
                  (rotulo potencialmente derivado da mesma cena)
  2 informacao    quantos pixels mudam de estado; quanto da deteccao e da aquisicao (bloco x data),
                  do lugar (pixel) e da interacao lugar x data, que e a unica parte que diz
                  "onde aparece agua quando"
  3 otimizacao    perdas, normas e cosseno dos gradientes, diferenca de ordenacao fisica e
                  desempenho auxiliar contra preditores triviais, a partir de frequencia_v1_1
  4 o sintetico e o sobreajuste ficam em `fechamento_sintetico.py`

Uso:
    python scripts/ia/fechamento_diagnostico.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for s in ("ia", "sar", "servico", "metodo", "neural"):
    sys.path.insert(0, str(REPO / "scripts" / s))
import conjunta_campanha as cc  # noqa: E402
from proveniencia import agora_utc, sha256_arquivo  # noqa: E402

SAIDA = REPO / "local_runs" / "fechamento-aprendizado-transferencia" / "v1"
PILHA = REPO / "local_runs" / "pilha-s1-mesmo-lugar" / "v1"
FREQ = REPO / "local_runs" / "aprendizado-conjunto" / "frequencia_v1_1"
QUADRO = REPO / "local_runs" / "chuva-deteccao-mesmo-lugar" / "v2_era5" / "quadro.parquet"


_COMP = {}


def _comp_do_bloco() -> dict:
    """bloco -> componente de dependencia (o prefixo do nome e o evento, nao o componente)."""
    if not _COMP:
        _COMP.update({b["bloco"]: b["componente"] for b in cc.ler(PILHA / "blocos.json")["blocos"]})
    return _COMP


def alinhamento() -> dict:
    blocos = cc.ler(PILHA / "blocos.json")["blocos"]
    erro = []
    for b in blocos:
        t = b["transform_aoi"]
        erro.append(max(abs(b["x0"] - (t[2] + b["coluna0"] * t[0])), abs(b["y0"] - (t[5] + b["linha0"] * t[4]))))
    pts = pd.read_parquet(cc.cb.BASE / "pontos.parquet", columns=["fonte", "evento_id", "janela_inicio", "janela_fim", "precisao_data"])
    cems = pts[pts.fonte == "cems"].groupby("evento_id").agg(ini=("janela_inicio", "min"), fim=("janela_fim", "max"),
                                                             precisao=("precisao_data", "first"))
    man = pd.read_parquet(PILHA / "manifesto_admissao.parquet")
    adm = man[man.admitida.astype(bool)]
    mesmas = []
    for ev, r in cems.iterrows():
        a = adm[(adm.evento_id == ev) | (adm.componente == ev)]
        datas = set(pd.to_datetime(a["data"]).dt.date.astype(str))
        d0, d1 = pd.Timestamp(r.ini).date(), pd.Timestamp(r.fim).date()
        dias = {str(d) for d in pd.date_range(d0, d1, freq="D").date}
        mesmas.append({"evento": ev, "data_produto": f"{d0}..{d1}", "precisao": r.precisao,
                       "passagens_pilha_na_data_do_produto": sorted(datas & dias)})
    n_mesma = sum(bool(m["passagens_pilha_na_data_do_produto"]) for m in mesmas)
    return {"blocos": len(blocos), "erro_max_origem_bloco_m": float(max(erro)),
            "eventos_cems": len(mesmas), "eventos_com_passagem_pilha_na_data_do_produto": n_mesma,
            "por_evento": mesmas,
            "leitura": ("CEMS mapeia com S1 e/ou optico; quando a pilha tem passagem no mesmo dia do produto, o rotulo fisico "
                        "e a deteccao SAR da pilha podem vir da mesma cena e nao sao observacoes independentes")}


def informacao(max_blocos: int | None = None) -> dict:
    quadro = pd.read_parquet(QUADRO)
    linhas, cont = [], {"pixels": 0, "nunca": 0, "sempre": 0, "muda": 0, "transicoes_2mais": 0}
    blocos = sorted(p.stem for p in (PILHA / "frequencia_e1").glob("*.npz"))[:max_blocos]
    for nome in blocos:
        with np.load(PILHA / "frequencia" / f"{nome}.npz", allow_pickle=True) as z:
            classe, datas = z["classe"], z["datas"]
        with np.load(PILHA / "frequencia_e1" / f"{nome}.npz", allow_pickle=True) as z:
            det = z["deteccao"]
        ok = (classe == 1) & (det != 255).all(0)
        if ok.sum() < 1000 or len(datas) < 10:
            continue
        d = det[:, ok].astype(np.float64)  # [datas, pixels]
        f = d.mean(0)
        cont["pixels"] += d.shape[1]
        cont["nunca"] += int((f == 0).sum()); cont["sempre"] += int((f >= .95).sum())
        cont["muda"] += int(((f > 0) & (f < .95)).sum())
        cont["transicoes_2mais"] += int((np.abs(np.diff(d, axis=0)).sum(0) >= 2).sum())
        m = d.mean(); vt = d.var()
        if vt == 0:
            continue
        aq = d.mean(1, keepdims=True); lu = d.mean(0, keepdims=True)
        inter = d - aq - lu + m
        q = quadro[quadro.bloco == nome].set_index("data")
        ds = [str(x) for x in datas]
        p72 = q.reindex(ds).p72.to_numpy(float) if not q.empty else np.full(len(ds), np.nan)
        taxa = d.mean(1)
        mes = pd.to_datetime(pd.Series(ds)).dt.month.to_numpy()
        sazonal = pd.Series(taxa).groupby(mes).transform("mean").to_numpy()
        linhas.append({"bloco": nome, "componente": _comp_do_bloco()[nome], "pixels": d.shape[1], "datas": d.shape[0],
                       "frac_var_aquisicao": float(((aq - m) ** 2).mean() / vt),
                       "frac_var_lugar": float(((lu - m) ** 2).mean() / vt),
                       "frac_var_lugar_x_data": float((inter ** 2).mean() / vt),
                       "frac_var_aquisicao_explicada_por_mes": float(1 - np.var(taxa - sazonal) / max(np.var(taxa), 1e-12)),
                       "corr_taxa_p72": float(pd.Series(taxa).corr(pd.Series(p72))) if np.isfinite(p72).sum() > 5 else None})
    t = pd.DataFrame(linhas)
    por_comp = t.groupby("componente")[["frac_var_aquisicao", "frac_var_lugar", "frac_var_lugar_x_data",
                                        "frac_var_aquisicao_explicada_por_mes", "corr_taxa_p72"]].median()
    return {"blocos": len(t), "contagem_pixels_E1": cont,
            "fracao_pixels": {k: cont[k] / max(cont["pixels"], 1) for k in ("nunca", "sempre", "muda", "transicoes_2mais")},
            "mediana_entre_componentes": {k: float(v) for k, v in por_comp.median().items()},
            "por_componente": por_comp.round(4).to_dict("index"),
            "leitura": ("frac_var_lugar e o que a frequencia agregada captura; frac_var_lugar_x_data e a informacao de "
                        "estado/transicao condicionada a aquisicao, a unica que pode ensinar resposta a forcamento; "
                        "frac_var_aquisicao e da cena (estacao, umidade geral, calibracao) e nao e do lugar")}


def confiabilidade(max_blocos: int | None = None) -> dict:
    """Frequencia do pixel em datas pares x impares (fora de evento): sinal de lugar ou oscilacao?
    Tambem em suporte 3x3 (90 m) e com estado persistente (deteccao em duas passagens seguidas)."""
    out = []
    blocos = sorted(p.stem for p in (PILHA / "frequencia_e1").glob("*.npz"))[:max_blocos]
    for nome in blocos:
        with np.load(PILHA / "frequencia" / f"{nome}.npz", allow_pickle=True) as z:
            classe, janela, dv1 = z["classe"], z["janela"], z["deteccao"]
        with np.load(PILHA / "frequencia_e1" / f"{nome}.npz", allow_pickle=True) as z:
            de1 = z["deteccao"]
        fora = np.array([str(j) == "fora" for j in janela])
        if fora.sum() < 12:
            continue
        r = {"bloco": nome, "componente": _comp_do_bloco()[nome]}
        for regra, det in (("v1", dv1), ("e1", de1)):
            d = det[fora].astype(np.float32)
            d[det[fora] == 255] = np.nan
            ok = (classe == 1) & np.isfinite(d).all(0)
            if ok.sum() < 1000:
                continue
            par, impar = np.nanmean(d[0::2], 0), np.nanmean(d[1::2], 0)
            r[f"{regra}_r_30m"] = float(np.corrcoef(par[ok], impar[ok])[0, 1]) if par[ok].std() > 0 and impar[ok].std() > 0 else None
            # 90 m: media 3x3 de pixels observaveis
            def agreg(a):
                h, w = (a.shape[0] // 3) * 3, (a.shape[1] // 3) * 3
                aa = np.where(ok, a, np.nan)[:h, :w].reshape(h // 3, 3, w // 3, 3)
                return np.nanmean(aa, (1, 3))
            with np.errstate(invalid="ignore"):
                p9, i9 = agreg(par), agreg(impar)
            k = np.isfinite(p9) & np.isfinite(i9)
            r[f"{regra}_r_90m"] = float(np.corrcoef(p9[k], i9[k])[0, 1]) if k.sum() > 50 and p9[k].std() > 0 else None
            pers = (d[1:] == 1) & (d[:-1] == 1)
            fp, fi = pers[0::2].mean(0), pers[1::2].mean(0)
            r[f"{regra}_r_persistente_30m"] = float(np.corrcoef(fp[ok], fi[ok])[0, 1]) if fp[ok].std() > 0 and fi[ok].std() > 0 else None
        out.append(r)
    t = pd.DataFrame(out)
    comp = t.groupby("componente").median(numeric_only=True)
    return {"blocos": len(t), "mediana_entre_componentes": {k: float(v) for k, v in comp.median().items()},
            "por_componente": comp.round(3).to_dict("index"),
            "leitura": "r = confiabilidade meia-datas da frequencia; baixa a 30 m e alta a 90 m indica sinal de lugar em suporte maior"}


def predicibilidade(versao: str = "v1") -> dict:
    """O terreno do encoder explica, FORA do componente, algum alvo temporal? Deixa um componente de fora.

    Alvos por pixel (pixels ja amostrados em frequencia_v1_1/base, 20 componentes):
      frequencia_fora   deteccao media fora de evento (regra do componente)
      resposta_chuva    deteccao nas datas fora com p72 no quartil superior do bloco menos a do inferior
      deteccao_evento   deteccao media nas datas da janela de evento
    Sem rotulo fisico; mede se ha informacao transferivel para o encoder aprender."""
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingRegressor
    aux = pd.read_parquet(FREQ / "base" / "auxiliar.parquet")
    quadro = pd.read_parquet(QUADRO)
    alvos = []
    for (bloco, regra), g in aux.groupby(["bloco", "regra"]):
        pasta = "frequencia_e1" if regra == "e1" else "frequencia"
        with np.load(PILHA / pasta / f"{bloco}.npz", allow_pickle=True) as z:
            det, datas, janela = z["deteccao"], [str(x) for x in z["datas"]], [str(x) for x in z["janela"]]
        d = det.reshape(len(datas), -1)[:, g.pixel.to_numpy()].astype(np.float32)
        d[d == 255] = np.nan
        q = quadro[quadro.bloco == bloco].set_index("data").reindex(datas)
        fora = np.array([j == "fora" for j in janela]); ev = np.array([j == "evento" for j in janela])
        p = q.p72.to_numpy(float)
        lo, hi = np.nanquantile(p[fora], [.25, .75]) if np.isfinite(p[fora]).sum() >= 8 else (np.nan, np.nan)
        molhado, seco = fora & (p >= hi), fora & (p <= lo)
        with np.errstate(invalid="ignore"):
            resp = np.nanmean(d[molhado], 0) - np.nanmean(d[seco], 0) if molhado.sum() >= 2 and seco.sum() >= 2 else np.full(len(g), np.nan)
            dev = np.nanmean(d[ev], 0) if ev.sum() else np.full(len(g), np.nan)
        alvos.append(pd.DataFrame({"idx": g.index, "resposta_chuva": resp, "deteccao_evento": dev}))
    a = aux.join(pd.concat(alvos).set_index("idx"))
    a = a.rename(columns={"frequencia": "frequencia_fora"})
    X = a[list(cc.cb.TERRENO)].to_numpy(np.float32)
    res = {}
    for alvo in ("frequencia_fora", "resposta_chuva", "deteccao_evento"):
        linhas = []
        for comp in sorted(a.componente.unique()):
            tr = (a.componente != comp).to_numpy() & np.isfinite(a[alvo].to_numpy())
            te = (a.componente == comp).to_numpy() & np.isfinite(a[alvo].to_numpy())
            if te.sum() < 200 or tr.sum() < 2000:
                continue
            rng = np.random.default_rng(7)
            itr = rng.choice(np.flatnonzero(tr), min(60000, tr.sum()), replace=False)
            m = HistGradientBoostingRegressor(max_iter=150, random_state=7).fit(X[itr], a[alvo].to_numpy()[itr])
            y, pr = a[alvo].to_numpy()[te], m.predict(X[te])
            mse0 = float(((y - a[alvo].to_numpy()[itr].mean()) ** 2).mean())
            mse = float(((y - pr) ** 2).mean())
            if versao == "v1":
                linhas.append({"componente": comp, "r2_fora_do_componente": 1 - mse / max(mse0, 1e-12),
                               "spearman_fora_do_componente": float(spearmanr(y, pr)[0]), "pixels": int(te.sum())})
            else:  # v1.1: nomes corretos (adendo 17/10/2026); o valor de v1 e o skill frente a media do TREINO
                from sklearn.metrics import r2_score
                linhas.append({"componente": comp, "skill_mse_vs_media_treino": 1 - mse / max(mse0, 1e-12),
                               "r2_usual_media_do_destino": float(r2_score(y, pr)), "mse": mse,
                               "mae": float(np.abs(y - pr).mean()), "mse_media_treino": mse0,
                               "spearman_fora_do_componente": float(spearmanr(y, pr)[0]), "pixels": int(te.sum())})
        t = pd.DataFrame(linhas)
        if versao == "v1":
            res[alvo] = {"componentes": len(t), "mediana_r2": float(t.r2_fora_do_componente.median()),
                         "mediana_spearman": float(t.spearman_fora_do_componente.median()),
                         "componentes_r2_positivo": int((t.r2_fora_do_componente > 0).sum()),
                         "por_componente": t.round(4).to_dict("records")}
        else:
            res[alvo] = {"componentes": len(t),
                         **{f"mediana_{c}": float(t[c].median()) for c in ("skill_mse_vs_media_treino", "r2_usual_media_do_destino",
                                                                           "mse", "mae", "spearman_fora_do_componente")},
                         "componentes_skill_positivo": int((t.skill_mse_vs_media_treino > 0).sum()),
                         "componentes_r2_usual_positivo": int((t.r2_usual_media_do_destino > 0).sum()),
                         "por_componente": t.round(5).to_dict("records")}
    res["leitura"] = (("R2 fora do componente <= 0 significa que o terreno do encoder nao transporta o alvo entre lugares: "
                       "uma cabeca auxiliar sobre o mesmo encoder nao tem informacao transferivel a acrescentar") if versao == "v1" else
                      ("skill = 1 - MSE/MSE(media do treino); r2_usual usa a media do proprio destino (sklearn). Nenhum dos dois e "
                       "condicao necessaria nem suficiente para ganho multitarefa; sao evidencia compativel com limite de alvo, "
                       "atributos, populacao, objetivo ou otimizacao, nao prova de que o unico limite e o alvo"))
    return res


def executar_v1_1() -> dict:
    """Recalcula so a previsibilidade com nomes corretos; o arquivo v1 fica intacto."""
    r = {"tipo": "diagnostico_A_v1_1_previsibilidade", "quando_utc": agora_utc(),
         "corrige": "diagnostico_A.json (v1) chamava de R2 um skill de MSE frente a media do TREINO",
         "v1_sha256": sha256_arquivo(SAIDA / "diagnostico" / "diagnostico_A.json"),
         "predicibilidade_transferivel": predicibilidade("v1_1")}
    cc.escrever(SAIDA / "diagnostico" / "diagnostico_A_v1_1.json", r)
    return r


def otimizacao() -> dict:
    rs = [cc.ler(f) for f in sorted((FREQ / "campanha").glob("*/*/resultado.json"))]
    rs = [r for r in rs if not r["rompido"]]
    g = pd.DataFrame([{"modo": r["modo"], "passo": x["passo"], "norma_fisica": x["norma_fisica"],
                       "norma_auxiliar": x["norma_auxiliar"], "cosseno": x["cosseno"]} for r in rs for x in r["gradientes"]])
    conj = g[g.modo == "conjunto"].groupby("passo")[["norma_fisica", "norma_auxiliar", "cosseno"]].median()
    h = pd.DataFrame([{"modo": r["modo"], **r["historico"][i]} for r in rs for i in (0, 99, 249, 499)])
    perdas = h.groupby(["modo", "passo"])[["fisica", "auxiliar"]].median().round(4)
    # auxiliar contra preditores triviais
    aux = pd.read_parquet(FREQ / "base" / "auxiliar.parquet", columns=["componente", "bloco", "frequencia"])
    triv = []
    for r in rs:
        if r["modo"] != "conjunto" or r["semente"] != 7:
            continue
        tr = aux[aux.componente.isin(r["aux_componentes_treino"])]
        for a in r["aux_avaliacao"]:
            te = aux[aux.componente == a["componente"]]
            triv.append({"componente": a["componente"], "mse_conjunto": a["mse"],
                         "mse_media_global_treino": float(((te.frequencia - tr.frequencia.mean()) ** 2).mean()),
                         "mse_media_do_bloco_oraculo": float(((te.frequencia - te.groupby("bloco").frequencia.transform("mean")) ** 2).mean()),
                         "variancia_frequencia": float(te.frequencia.var())})
    tv = pd.DataFrame(triv)
    # ordenacao fisica conjunto x parada
    from scipy.stats import spearmanr
    rho = []
    for d in sorted({r["destino"] for r in rs}):
        for s in (7, 11, 23):
            a, b = FREQ / "campanha" / d / f"conjunto_s{s}" / "predicoes.parquet", FREQ / "campanha" / d / f"parada_s{s}" / "predicoes.parquet"
            if a.exists() and b.exists():
                pa, pb = pd.read_parquet(a), pd.read_parquet(b)
                rho.append(float(spearmanr(pa.escore, pb.escore)[0]))
    return {"execucoes": len(rs),
            "gradiente_conjunto_mediana_por_passo": conj.round(4).reset_index().to_dict("records"),
            "perdas_mediana": {f"{k[0]}|{k[1]}": v for k, v in perdas.to_dict("index").items()},
            "auxiliar_contra_triviais_mediana": {k: float(v) for k, v in tv.drop(columns="componente").median().items()},
            "auxiliar_por_componente": tv.round(5).to_dict("records"),
            "spearman_fisica_conjunto_parada": {"mediana": float(np.median(rho)), "min": float(np.min(rho)), "n": len(rho)},
            "leitura": ("cosseno negativo crescente = a auxiliar puxa o encoder contra a fisica no fim do treino; "
                        "mse_conjunto perto de mse_media_do_bloco = a auxiliar aprende sobretudo o nivel do bloco (lugar agregado)")}


def executar() -> dict:
    SAIDA.mkdir(parents=True, exist_ok=True)
    r = {"tipo": "diagnostico_etapa_A", "versao": "fechamento_aprendizado_transferencia_v1", "quando_utc": agora_utc(),
         "nao_e": "diagnostico de artefatos ja gravados; nao e resultado novo de hipotese",
         "fontes": {str(p.relative_to(REPO)): sha256_arquivo(p) for p in (
             PILHA / "blocos.json", PILHA / "manifesto_admissao.parquet", QUADRO, FREQ / "resultado.json",
             FREQ / "base" / "manifesto.json")},
         "alinhamento": alinhamento(), "informacao_temporal": informacao(), "confiabilidade_temporal": confiabilidade(),
         "predicibilidade_transferivel": predicibilidade(), "otimizacao_frequencia_v1_1": otimizacao()}
    cc.escrever(SAIDA / "diagnostico" / "diagnostico_A.json", r)
    return r


if __name__ == "__main__":
    if "--v1_1" in sys.argv:
        r = executar_v1_1()["predicibilidade_transferivel"]
        print(json.dumps({k: {x: v[x] for x in v if x.startswith(("mediana", "componentes"))} for k, v in r.items() if isinstance(v, dict)}, indent=1))
        sys.exit(0)
    r = executar()
    print(json.dumps({k: r[k] for k in ("alinhamento",)}, default=str)[:600])
    print(json.dumps(r["informacao_temporal"]["fracao_pixels"]), json.dumps(r["informacao_temporal"]["mediana_entre_componentes"]))
    print(json.dumps(r["otimizacao_frequencia_v1_1"]["auxiliar_contra_triviais_mediana"]), r["otimizacao_frequencia_v1_1"]["spearman_fisica_conjunto_parada"])
