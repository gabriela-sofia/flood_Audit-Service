"""MT11, fechamento (18/10/2026): auditoria recebida, influência completa, população efetiva e nomes das leituras.

Linhagem própria `local_runs/mt11-fechamento/v1`. Não treina, não baixa, não recalcula atributos nem reescreve
artefatos do MT11: lê `escores_emendado.parquet`, `completude.csv` e o resultado congelado, e usa as funções do
próprio MT11 (`mt11_confirmatorio_transdutivo.medir/contraste`) para reproduzir os nove contrastes.

  auditoria   hashes dos artefatos auditados pela coordenadora; nove contrastes reproduzidos contra o resultado
  influencia  retirada de cada componente, nas 9 combinações (leitura x hipótese), com a MESMA reamostragem;
              diagnóstico posterior ao resultado — não altera a decisão registrada nem autoriza excluir unidade
  populacao   funil de pontos por leitura: solicitados, obtidos, após o espelho, materializados na leitura,
              em eventos válidos para Q, observáveis para HF, em eventos válidos para HF, eventos e componentes
  adendo      nomes públicos inequívocos das leituras, proveniência efetiva (atributos_e8) e quais emendas
              alteraram população ou unidade

Uso:
    python scripts/ia/mt11_fechamento.py auditoria|influencia|populacao|adendo|tudo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for s in ("ia", "metodo", "servico", "geo", "suscetibilidade", "terreno", "sar"):
    sys.path.insert(0, str(REPO / "scripts" / s))
from proveniencia import agora_utc, sha256_arquivo  # noqa: E402

MT11 = REPO / "local_runs" / "mt11-transdutivo"
COORD = REPO / "local_runs" / "coordenacao-mt11" / "v2"
SAIDA = REPO / "local_runs" / "mt11-fechamento" / "v1"
HIPOTESES = {"HQ1": ("Q_conexao", "A0_regra_hand"), "HQ2": ("Q_conexao", "Q_base"), "HF": ("F_transf", "Q_conexao")}
NOMES_PUBLICOS = {
    "congelada_original": "agrupamento_original_com_atributos_e8",
    "emendada_primaria": "grafo_primario_com_atributos_e8",
    "sensibilidade_exutorio": "grafo_exutorio_com_atributos_e8",
}


def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def gravar(p, o):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _m11():
    import mt11_confirmatorio_transdutivo as m11
    return m11


def dados():
    """Escores, máscara de observabilidade do HF e mapas de unidade, exatamente como `mt11_emenda.avaliar`."""
    m11 = _m11()
    s = pd.read_parquet(MT11 / "escores_emendado.parquet")
    obs = ((s.arvore <= m11.CFG["arvore_max"]) & (s.construido_wc <= m11.CFG["construido_max"])
           & (s.slope_deg <= m11.CFG["declive_max_graus"]) & (s.jrc_occ_s1 < m11.CFG["jrc_permanente"])
           & (s.n_pre >= m11.CFG["min_datas_pre"]) & s.F.notna())
    t = pd.read_csv(MT11 / "completude.csv").set_index("evento_id")
    mapas = {"congelada_original": t.unidade_centro,
             "emendada_primaria": t.componente_primario.where(~t.ligado_primario.astype(bool)),
             "sensibilidade_exutorio": t.componente_exutorio.where(~t.ligado_exutorio.astype(bool))}
    return s, obs, t, mapas


def deltas(s, obs, mapa) -> tuple[dict, dict]:
    """Delta por componente das três hipóteses e as tabelas de medida (uma leitura)."""
    m11 = _m11()
    sub = s.assign(unidade=s.evento_id.map(mapa)).dropna(subset=["unidade"])
    sub_hf = sub[obs.loc[sub.index]]
    mq = m11.medir(sub, ["A0_regra_hand", "Q_base", "Q_conexao"])
    mf = m11.medir(sub_hf, ["Q_conexao", "F_transf"])
    uq = mq.groupby(["unidade", "modelo"]).auc.mean().unstack()
    uf = mf.groupby(["unidade", "modelo"]).auc.mean().unstack()
    d = {"HQ1": (uq.Q_conexao - uq.A0_regra_hand).dropna(), "HQ2": (uq.Q_conexao - uq.Q_base).dropna(),
         "HF": (uf.F_transf - uf.Q_conexao).dropna()}
    return d, {"sub": sub, "sub_hf": sub_hf, "mq": mq, "mf": mf}


# ------------------------------------------------------------------ auditoria

def auditoria() -> dict:
    m11 = _m11()
    reg = ler(COORD / "registro_revisao.json")
    hashes = {p: {"registrado": h, "atual": sha256_arquivo(REPO / p) if (REPO / p).exists() else None}
              for p, h in reg.items()}
    for v in hashes.values():
        v["confere"] = v["registrado"] == v["atual"]
    original = ler(MT11 / "resultado_emendado.json")
    s, obs, _, mapas = dados()
    rep = {}
    for leitura, mapa in mapas.items():
        d, _ = deltas(s, obs, mapa)
        for h, dd in d.items():
            c = m11.contraste(dd)
            o = original["leituras"][leitura][h]
            rep[f"{leitura}|{h}"] = {"n": c["n"], "media": c["media"], "ic": c.get("ic"), "veredito": c["veredito"],
                                     "a_favor": c["a_favor"], "dif_media": abs(c["media"] - o["media"]),
                                     "dif_ic": float(np.max(np.abs(np.array(c["ic"]) - o["ic"]))),
                                     "n_confere": c["n"] == o["n"], "veredito_confere": c["veredito"] == o["veredito"]}
    aud = ler(COORD / "auditoria_resultado_mt11.json")
    r = {"tipo": "mt11_fechamento_auditoria_recebida", "quando_utc": agora_utc(), "hashes": hashes,
         "todos_hashes_conferem": all(v["confere"] for v in hashes.values()),
         "contrastes_reproduzidos": rep,
         "max_dif_media": max(v["dif_media"] for v in rep.values()), "max_dif_ic": max(v["dif_ic"] for v in rep.values()),
         "auditoria_coordenadora_hash": sha256_arquivo(COORD / "auditoria_resultado_mt11.json"),
         "coordenadora_ic_reproduzido": all(
             np.allclose(aud["leituras_recalculadas"][l][h]["ic"], rep[f"{l}|{h}"]["ic"], atol=1e-15, rtol=0)
             for l in mapas for h in HIPOTESES),
         "recalculo_de_atributos": False, "download": False, "treino": False}
    gravar(SAIDA / "auditoria_recebida.json", r)
    return {k: r[k] for k in ("todos_hashes_conferem", "max_dif_media", "max_dif_ic", "coordenadora_ic_reproduzido")}


# ------------------------------------------------------------------ influência

def influencia() -> dict:
    m11 = _m11()
    s, obs, _, mapas = dados()
    linhas = []
    for leitura, mapa in mapas.items():
        d, _ = deltas(s, obs, mapa)
        for h, dd in d.items():
            base = m11.contraste(dd)
            for u, v in dd.sort_values(ascending=False).items():
                c = m11.contraste(dd.drop(u))
                linhas.append({"leitura": leitura, "nome_publico": NOMES_PUBLICOS[leitura], "hipotese": h,
                               "unidade_retirada": u, "delta_retirado": float(v), "n_restante": c["n"],
                               "media_restante": c["media"], "ic_inferior": (c.get("ic") or [None, None])[0],
                               "ic_superior": (c.get("ic") or [None, None])[1], "veredito_restante": c["veredito"],
                               "regra_projeto_restante": c["veredito"] == "sustentado",
                               "limite_inferior_acima_da_margem": bool(c.get("ic") and c["ic"][0] > m11.CFG["margem_auc"]),
                               "veredito_original": base["veredito"], "n_original": base["n"],
                               "media_original": base["media"], "ic_original": base.get("ic")})
    t = pd.DataFrame(linhas)
    t.to_csv(SAIDA / "influencia_completa.csv", index=False)
    resumo = {}
    for (l, h), g in t.groupby(["leitura", "hipotese"]):
        resumo[f"{l}|{h}"] = {"veredito_original": g.veredito_original.iloc[0], "n": int(g.n_original.iloc[0]),
                              "retiradas": len(g), "retiradas_que_mantem_regra": int(g.regra_projeto_restante.sum()),
                              "media_restante_min": float(g.media_restante.min()), "media_restante_max": float(g.media_restante.max()),
                              "ic_inferior_min": float(g.ic_inferior.min()), "ic_inferior_max": float(g.ic_inferior.max())}
    alvo = t[(t.leitura == "emendada_primaria") & (t.hipotese == "HF") & (t.unidade_retirada == "MT11:GFD:4355")].iloc[0]
    coord = pd.read_csv(COORD / "influencia_hf.csv").set_index("unidade_retirada")
    hf = t[(t.leitura == "emendada_primaria") & (t.hipotese == "HF")].set_index("unidade_retirada")
    r = {"tipo": "mt11_influencia_completa", "quando_utc": agora_utc(), "arquivo": sha256_arquivo(SAIDA / "influencia_completa.csv"),
         "natureza": "diagnóstico posterior ao resultado; mesma reamostragem (semente e nível do protocolo); não altera decisão "
                     "registrada nem autoriza excluir unidade pelo tamanho do efeito; nenhuma retirada é promovida",
         "resumo": resumo,
         "HF_primario_sem_GFD4355": {"n": int(alvo.n_restante), "media": float(alvo.media_restante),
                                     "ic98_33": [float(alvo.ic_inferior), float(alvo.ic_superior)],
                                     "veredito": alvo.veredito_restante},
         "confere_coordenadora": {"unidades": sorted(coord.index) == sorted(hf.index),
                                  "max_dif_media": float((coord.media_restante - hf.media_restante.reindex(coord.index)).abs().max()),
                                  "max_dif_ic": float(max((coord.ic_inferior - hf.ic_inferior.reindex(coord.index)).abs().max(),
                                                          (coord.ic_superior - hf.ic_superior.reindex(coord.index)).abs().max()))}}
    gravar(SAIDA / "influencia.json", r)
    return {"HF_sem_4355": r["HF_primario_sem_GFD4355"], "confere": r["confere_coordenadora"],
            "resumo": {k: (v["retiradas_que_mantem_regra"], v["retiradas"]) for k, v in resumo.items()}}


# ------------------------------------------------------------------ população

def populacao() -> dict:
    s, obs, t, mapas = dados()
    m11 = _m11()
    base = {"solicitados": int(2 * t.solicitados_por_classe.sum()),
            "obtidos": int(t.obtidos_inundacao.sum() + t.obtidos_seco.sum()),
            "obtidos_inundacao": int(t.obtidos_inundacao.sum()), "obtidos_seco": int(t.obtidos_seco.sum()),
            "apos_espelho_ds06": int(t.linhas_pos_espelho.sum()), "linhas_escores": len(s),
            "eventos_com_pontos": int(t.linhas_pos_espelho.gt(0).sum()),
            "eventos_qualidade_sar_aprovada": int(t.regra.notna().sum()),
            "observaveis_hf_todos_eventos": int(obs.sum())}
    if base["apos_espelho_ds06"] != len(s):
        raise SystemExit("escores e completude divergem")
    linhas = []
    for leitura, mapa in mapas.items():
        d, x = deltas(s, obs, mapa)
        sub, sub_hf, mq, mf = x["sub"], x["sub_hf"], x["mq"], x["mf"]
        ev_q = set(mq.evento_id)
        ev_f = set(mf.evento_id)
        # pareamento do HF: Q e F nos mesmos pontos (mesmo subconjunto, medidos juntos)
        par = mf.pivot_table(index="evento_id", columns="modelo", values="n")
        linhas.append({
            "leitura": leitura, "nome_publico": NOMES_PUBLICOS[leitura], "atributos": "atributos_e8",
            "eventos_na_leitura": int(sub.evento_id.nunique()), "pontos_materializados_na_leitura": len(sub),
            "eventos_validos_Q": len(ev_q), "pontos_eventos_validos_Q": int(sub[sub.evento_id.isin(ev_q)].shape[0]),
            "eventos_fora_Q": sorted(set(sub.evento_id) - ev_q),
            "componentes_Q": int(len(d["HQ1"])),
            "pontos_observaveis_HF": len(sub_hf), "eventos_com_pontos_observaveis": int(sub_hf.evento_id.nunique()),
            "eventos_validos_HF": len(ev_f), "pontos_eventos_validos_HF": int(sub_hf[sub_hf.evento_id.isin(ev_f)].shape[0]),
            "componentes_HF": int(len(d["HF"])),
            "HF_Q_e_F_mesmos_pontos": bool(len(par) == 0 or (par["Q_conexao"] == par["F_transf"]).all()),
            "minimo_pontos_evento": m11.CFG["minimo_pontos_evento"]})
    p = pd.DataFrame(linhas)
    p.to_csv(SAIDA / "populacao_efetiva.csv", index=False)
    emend = t[t.ligado_primario.astype(bool) == False]  # noqa: E712
    r = {"tipo": "mt11_populacao_efetiva", "quando_utc": agora_utc(), "funil_comum": base, "por_leitura": linhas,
         "hf_primario": {"eventos_qualidade_antes_do_grafo": base["eventos_qualidade_sar_aprovada"],
                         "eventos_qualidade_livres_no_primario": int(emend.regra.notna().sum()),
                         "nota": "22 eventos passam a qualidade SAR antes da seleção pelo grafo; o contraste primário usa os "
                                 "17 válidos e livres, em 16 componentes"},
         "arquivo": sha256_arquivo(SAIDA / "populacao_efetiva.csv"),
         "regra": "campo 'pontos' do resultado congelado = linhas materializadas na leitura (inclui evento com < 20 pontos); "
                  "pontos efetivos das métricas = só eventos válidos (>= 20 pontos e duas classes)"}
    gravar(SAIDA / "populacao_efetiva.json", r)
    return {"funil": base, "por_leitura": [{k: v for k, v in x.items() if k in (
        "leitura", "pontos_materializados_na_leitura", "pontos_eventos_validos_Q", "eventos_fora_Q", "eventos_validos_Q",
        "componentes_Q", "pontos_eventos_validos_HF", "eventos_validos_HF", "componentes_HF", "HF_Q_e_F_mesmos_pontos")} for x in linhas]}


# ------------------------------------------------------------------ reparação posterior ao resultado

def reparo() -> dict:
    """Mesmo critério científico (grafo MT01, desenho primário e exutório) com cobertura completa das referências.

    Posterior ao resultado: NÃO é pré-registro e não substitui `resultado_emendado.json`. Só muda o que a cobertura
    completa muda (eventos ligados e agrupamento); medida, bootstrap, margem e decisão são os do protocolo.
    """
    m11 = _m11()
    conf = ler(SAIDA / "cobertura" / "confronto.json")
    s, obs, t, mapas = dados()
    novos = {}
    for desenho, chave in (("primario", "emendada_primaria"), ("exutorio", "sensibilidade_exutorio")):
        ev = {e["evento_id"].replace("MT11:", ""): e for e in conf["desenhos"][desenho]["eventos"]}
        novos[chave] = pd.Series({k: (None if e["ligado_completo"] else e["componente"]) for k, e in ev.items()}).reindex(t.index)
    res, comp = {}, {}
    for leitura, mapa in novos.items():
        d, _ = deltas(s, obs, mapa)
        antigo, _ = deltas(s, obs, mapas[leitura])
        res[leitura] = {}
        for h, dd in d.items():
            c = m11.contraste(dd)
            o = m11.contraste(antigo[h])
            res[leitura][h] = {"n": c["n"], "media": c["media"], "ic": c.get("ic"), "veredito": c["veredito"],
                               "a_favor": c["a_favor"], "original": {"n": o["n"], "media": o["media"], "ic": o.get("ic"),
                                                                     "veredito": o["veredito"]},
                               "limite_inferior_acima_da_margem": bool(c.get("ic") and c["ic"][0] > m11.CFG["margem_auc"])}
        a, b = mapas[leitura].dropna(), mapa.dropna()
        comp[leitura] = {"saem": sorted(set(a.index) - set(b.index)), "entram": sorted(set(b.index) - set(a.index)),
                         "componentes_original": int(a.nunique()), "componentes_reparo": int(b.nunique()),
                         "agrupamento_igual_nos_comuns": bool(
                             pd.Series(a.loc[a.index.intersection(b.index)]).groupby(lambda x: b[x]).nunique().eq(1).all())}
    r = {"tipo": "mt11_reparo_posterior_ao_resultado", "quando_utc": agora_utc(),
         "natureza": "reparação de independência posterior ao resultado; não é pré-registro; resultado original preservado",
         "motivo": "cobertura completa das referências (todos os pontos CEMS MT03/MT04, ds06, quarentena; polígonos neutros) "
                   "revela vínculo que a amostra de 40 pontos por unidade não via",
         "confronto": sha256_arquivo(SAIDA / "cobertura" / "confronto.json"),
         "mudancas_de_eventos": comp, "contrastes": res,
         "resultado_original": sha256_arquivo(MT11 / "resultado_emendado.json")}
    gravar(SAIDA / "reparo_posterior.json", r)
    return {"mudancas": comp, "contrastes": {l: {h: {k: v[k] for k in ("n", "media", "ic", "veredito")} for h, v in x.items()}
                                             for l, x in res.items()}}


GRAFO_3B = REPO / "local_runs" / "confirmacao-rodada3" / "v2" / "grafo"
SAIDA_V2 = REPO / "local_runs" / "mt11-fechamento" / "v2"


def reparo_3b() -> dict:
    """Segunda reparação posterior: componentes e ligações do grafo reparado da rodada 3B (cobertura ainda mais completa:
    AOIs neutras locais no lugar de 36 centroides). Mesma regra do MT11: evento ligado a desenvolvimento, examinado,
    reserva ou confirmação MT04 sai; candidatos da rodada 3 e outros eventos MT11 não excluem. Não é pré-registro."""
    m11 = _m11()
    n = pd.read_parquet(GRAFO_3B / "nos.parquet")
    excl = n.papeis.str.contains("desenvolvimento|examinado|reserva|confirmacao_MT04")
    m = n[n.papeis.str.contains("confirmacao_MT11")].set_index("evento")
    s, obs, t, mapas = dados()
    novos, mudancas = {}, {}
    for chave, col in (("emendada_primaria", "componente_primario"), ("sensibilidade_exutorio", "componente_exutorio")):
        ligado = m[col].isin(set(n.loc[excl, col]))
        novos[chave] = m[col].where(~ligado).reindex(t.index)
        a = mapas[chave].dropna()
        b = novos[chave].dropna()
        mudancas[chave] = {"saem": sorted(set(a.index) - set(b.index)), "entram": sorted(set(b.index) - set(a.index)),
                           "componentes_original": int(a.nunique()), "componentes_3b": int(b.nunique())}
    res = {}
    for leitura, mapa in novos.items():
        d, _ = deltas(s, obs, mapa)
        antigo, _ = deltas(s, obs, mapas[leitura])
        res[leitura] = {h: {**{k: m11.contraste(dd).get(k) for k in ("n", "media", "ic", "veredito", "a_favor")},
                            "original": {k: m11.contraste(antigo[h]).get(k) for k in ("n", "media", "ic", "veredito")}}
                        for h, dd in d.items()}
    a = pd.read_parquet(GRAFO_3B / "arestas.parquet")
    p = a[a.desenho == "primario"]
    ev_novos = [e for e in mudancas["emendada_primaria"]["saem"]]
    evid = {}
    for e in ev_novos:
        ch = set(p[p.evento == e].chave)
        evid[e] = p[p.chave.isin(ch)].groupby(["causa", "chave"]).evento.apply(lambda x: sorted(set(x))).reset_index().to_dict("records")
    r = {"tipo": "mt11_reparo_posterior_grafo_3b", "quando_utc": agora_utc(),
         "natureza": "segunda reparação de independência posterior ao resultado, pelo mesmo critério; não é pré-registro; resultado "
                     "original e primeiro reparo preservados; nenhuma exclusão por efeito",
         "grafo_3b": {f: sha256_arquivo(GRAFO_3B / f) for f in ("nos.parquet", "arestas.parquet", "componentes.parquet", "grafo.json")},
         "mudancas_de_eventos": mudancas, "evidencia_eventos_que_saem_primario": evid, "contrastes": res,
         "resultado_original": sha256_arquivo(MT11 / "resultado_emendado.json")}
    gravar(SAIDA_V2 / "reparo_grafo_3b.json", r)
    return {"mudancas": mudancas, "contrastes": {l: {h: {k: v[k] for k in ("n", "media", "ic", "veredito")} for h, v in x.items()}
                                                 for l, x in res.items()}}


# ------------------------------------------------------------------ matriz de evidências

def _fmt(x):
    return f"{x:+.4f}"


def matriz() -> dict:
    """Matriz por hipótese, montada dos artefatos (MT11 reproduzido, influência, reparo, rodada 3, MT08-MT10)."""
    aud = ler(SAIDA / "auditoria_recebida.json")["contrastes_reproduzidos"]
    inf = ler(SAIDA / "influencia.json")["resumo"]
    rep = ler(SAIDA / "reparo_posterior.json")["contrastes"]
    rep3 = ler(SAIDA_V2 / "reparo_grafo_3b.json")["contrastes"] if (SAIDA_V2 / "reparo_grafo_3b.json").exists() else None
    r3 = ler(REPO / "local_runs/confirmacao-rodada3/v1/confirmacao/resultado.json")
    mt10 = ler(REPO / "local_runs/mecanismos-transferiveis/v10/teto_transdutivo_v1/resultado.json")
    mt10_pre = ler(REPO / "local_runs/mecanismos-transferiveis/v10/teto_transdutivo_v1/sensibilidade_pre_evento.json")

    def ic(c):
        return f"[{_fmt(c['ic'][0])}; {_fmt(c['ic'][1])}]"

    def bloco(h):
        p, e = aud[f"emendada_primaria|{h}"], aud[f"sensibilidade_exutorio|{h}"]
        rp = rep["emendada_primaria"][h]
        i = inf[f"emendada_primaria|{h}"]
        ie = inf[f"sensibilidade_exutorio|{h}"]
        return (f"primário {_fmt(p['media'])} {ic(p)} ({p['n']} comp., {p['a_favor']} a favor; {p['veredito']})",
                f"exutório {_fmt(e['media'])} {ic(e)} ({e['n']} comp.; {e['veredito']}); reparo 1 (cobertura completa): primário "
                f"{_fmt(rp['media'])} {ic(rp)} ({rp['n']} comp.; {rp['veredito']})"
                + (f"; reparo 2 (grafo 3B): primário {_fmt(rep3['emendada_primaria'][h]['media'])} {ic(rep3['emendada_primaria'][h])} "
                   f"({rep3['emendada_primaria'][h]['n']} comp.; {rep3['emendada_primaria'][h]['veredito']})" if rep3 else ""),
                f"retiradas que mantêm a regra: primário {i['retiradas_que_mantem_regra']}/{i['retiradas']} (média restante "
                f"{_fmt(i['media_restante_min'])} a {_fmt(i['media_restante_max'])}; limite inf. {_fmt(i['ic_inferior_min'])} a "
                f"{_fmt(i['ic_inferior_max'])}); exutório {ie['retiradas_que_mantem_regra']}/{ie['retiradas']}")
    linhas = []
    for h, hip, mec, frase, lacuna in (
        ("HQ1", "Q_conexao supera a regra de HAND", "física congelada (HAND, declive, SCA, cobertura, conexão) contra o proxy único HAND",
         "Os pesos físicos congelados superaram a regra de HAND na avaliação externa retrospectiva, nos dois desenhos; o ganho "
         "excede +0,01 apenas no desenho primário, onde o limite inferior passa a margem.",
         "replicação com grafo comum reparado e em regiões fora da amostra GFD 2015-2018"),
        ("HQ2", "a conexão acrescenta discriminação à base física", "distância à drenagem/canal e interações com relevo",
         "A conexão acrescentou discriminação no desenho primário; no exutório o resultado é inconclusivo. A rodada 3 é "
         "evidência adicional, pendente da reparação de independência (3B).",
         "síntese conjunta MT11 + rodada 3 só com o mesmo grafo e família de testes"),
        ("HF", "o histórico SAR do destino acrescenta discriminação ao Q", "frequência de água S1 anterior ao evento no próprio pixel",
         "Na leitura registrada, a frequência SAR anterior ao evento elevou a AUC média por componente no subconjunto observável; "
         "após a reparação de independência posterior (grafo 3B), o componente de maior ganho sai por ligação a evento examinado "
         "e o acréscimo fica positivo porém inconclusivo; também inconclusivo no exutório.",
         "replicação externa do HF e de sua heterogeneidade; máscaras datadas para avaliação integralmente prévia")):
        a, b, c = bloco(h)
        linhas.append({"hipotese": f"{h}: {hip}", "mecanismo_ou_proxy": mec, "resultado": a, "populacao":
                       "MT11 GFD 2015-2018, desenho primário emendado; HF só pixels observáveis com referência SAR aprovada",
                       "unidade": "componente do grafo (MT01: evento, L8, onda 30 d, chuva 500 km/15 d)",
                       "nivel_intervalo": "IC98,33% bootstrap de componentes (2.000, semente do protocolo)", "sensibilidade": b,
                       "influencia": c, "pressupostos": "rótulo MODIS GFD; máscaras auxiliares de 2021 (JRC, WorldCover); "
                       "agrupamento pelo grafo; margem +0,01 do projeto", "frase_permitida": frase, "proxima_lacuna": lacuna})
    p = r3["primaria"]
    r3b = ler(REPO / "local_runs/confirmacao-rodada3/v2/reanalise/resultado_reparo.json")["primaria"]
    linhas.append({"hipotese": "Q_conexao - Q_base fluvial, rodada 3 (CEMS 2019-2021)", "mecanismo_ou_proxy": "conexão nos pesos Q",
                   "resultado": f"v1 {_fmt(p['delta_medio'])} [{_fmt(p['ic'][0])}; {_fmt(p['ic'][1])}] IC95 ({p['unidades']} comp.; {p['veredito']}); "
                                f"reparado 3B {_fmt(r3b['delta_medio'])} [{_fmt(r3b['ic'][0])}; {_fmt(r3b['ic'][1])}] ({r3b['unidades']} comp.; {r3b['veredito']}, no mínimo do protocolo)",
                   "populacao": "positivos Riverine flood x negativos observados CEMS", "unidade": "componente do grafo da rodada 3 (v1)",
                   "nivel_intervalo": "IC95 bootstrap de componentes", "sensibilidade": "exutório 9 comp. "
                   f"{_fmt(r3['sensibilidade_exutorio']['primaria']['delta_medio'])}",
                   "influencia": "sem EMSR456 +0,0161 [+0,0056; +0,0280] (descritivo)",
                   "pressupostos": "grafo v1 tinha vínculos não detectados; reparo 3B posterior ao resultado, adendo 2 escrito após ler a execução 2",
                   "frase_permitida": "Evidência adicional estreita (8 componentes CEMS europeus livres); não somar a MT11 nem tratar como réplica independente.",
                   "proxima_lacuna": "outro universo de casos; grafo comum MT11 + rodada 3 (GFD:4597-EMSR465 ligados)"})
    linhas.append({"hipotese": "histórico SAR no desenvolvimento (MT10)", "mecanismo_ou_proxy": "freq_fora S1 do pixel",
                   "resultado": "primária +0,029 [+0,001; +0,060] (18 comp. CEMS de desenvolvimento)",
                   "populacao": "CEMS de desenvolvimento, rótulo compartilha S1", "unidade": "componente MT05",
                   "nivel_intervalo": "IC do protocolo MT10", "sensibilidade": "só datas anteriores (pós-hoc): +0,014 [−0,001; +0,031]",
                   "influencia": "não calculada", "pressupostos": "datas posteriores ao evento na primária",
                   "frase_permitida": "Desenvolvimento retrospectivo; motivou o MT11, não o confirma.",
                   "proxima_lacuna": "—", "_fontes": [sha256_arquivo(REPO / "local_runs/mecanismos-transferiveis/v10/teto_transdutivo_v1/resultado.json"),
                                                     sha256_arquivo(REPO / "local_runs/mecanismos-transferiveis/v10/teto_transdutivo_v1/sensibilidade_pre_evento.json")]})
    linhas.append({"hipotese": "aprendizado neural compartilhado melhora a física", "mecanismo_ou_proxy": "encoder conjunto, tarefa auxiliar",
                   "resultado": "v4-v6 e MT07 abaixo de +0,01 (v6 +0,0068 [+0,0042; +0,0091]); MT08: famílias testadas sem ganho relevante",
                   "populacao": "43 componentes de desenvolvimento", "unidade": "componente", "nivel_intervalo": "97,5-98,75%",
                   "sensibilidade": "—", "influencia": "—", "pressupostos": "variantes e famílias testadas",
                   "frase_permitida": "Não demonstrado nas variantes testadas; comparar F_transf com Q não testa essa pergunta.",
                   "proxima_lacuna": "representação compartilhada com ablação e controle rompido/permutado"})
    linhas.append({"hipotese": "identificação da causa (HAND, canal)", "mecanismo_ou_proxy": "relevo relativo e distância ao canal",
                   "resultado": "MT09: direção estável dentro do evento, pares a 3 km e 4 famílias de instrumento",
                   "populacao": "331.492 pontos, 39 componentes", "unidade": "componente (meta-análise)",
                   "nivel_intervalo": "IC95 e intervalo de predição", "sensibilidade": "pares a 1 km; amostragem por classe",
                   "influencia": "—", "pressupostos": "sem confundidor local residual; rótulo não diferencial",
                   "frase_permitida": "Evidência observacional convergente sob pressupostos declarados; não identifica efeito causal.",
                   "proxima_lacuna": "desenho com variação exógena ou dados de lâmina/chuva apropriados"})
    t = pd.DataFrame([{k: v for k, v in x.items() if not k.startswith("_")} for x in linhas])
    t.to_csv(SAIDA / "matriz_evidencias.csv", index=False)
    _ = (mt10, mt10_pre)
    return {"linhas": len(t), "arquivo": sha256_arquivo(SAIDA / "matriz_evidencias.csv")}


# ------------------------------------------------------------------ adendo (nomes e proveniência)

def adendo() -> dict:
    emendas = ler(MT11 / "emendas.json")
    pop = ler(SAIDA / "populacao_efetiva.json")
    r = {"tipo": "mt11_adendo_fechamento", "versao": 1, "quando_utc": agora_utc(),
         "resultado_original": {"arquivo": "local_runs/mt11-transdutivo/resultado_emendado.json",
                                "sha256": sha256_arquivo(MT11 / "resultado_emendado.json"), "alterado": False},
         "nomes_das_leituras": {
             antiga: {"nome_publico": nova, "chave_preservada": antiga,
                      "agrupamento": {"congelada_original": "bacia do centro do tile (nível 8), como no protocolo de 13:47",
                                      "emendada_primaria": "componentes do grafo comum, desenho primário, sem eventos ligados",
                                      "sensibilidade_exutorio": "componentes do grafo comum, desenho exutório, sem eventos ligados"}[antiga],
                      "atributos": "atributos_e8 (emenda 8: S1 pela órbita que cobre os pontos)"}
             for antiga, nova in NOMES_PUBLICOS.items()},
         "o_que_congelada_original_nao_e": "não é execução integral do protocolo original: aplica o agrupamento original aos "
                                           "atributos emendados; o HF dessa leitura usa a órbita/população da emenda 8; a extração "
                                           "original existe só para 3 eventos (atributos/)",
         "emendas_que_alteram_populacao_ou_unidade": {"3": "unidade e população (grafo comum, níveis L6/L8 corretos, exclusão de ligados)",
                                                      "8": "população do HF (pontos na pegada da órbita escolhida pela cobertura)",
                                                      "9": "evento sem linhas tratado como exclusão técnica (não altera unidade)"},
         "emendas_registradas": len(emendas.get("emendas", emendas)) if isinstance(emendas, (dict, list)) else None,
         "campo_pontos": {"significado_no_resultado": "linhas materializadas na leitura",
                          "efetivos": {x["leitura"]: {"materializados": x["pontos_materializados_na_leitura"],
                                                      "efetivos_Q": x["pontos_eventos_validos_Q"],
                                                      "efetivos_HF": x["pontos_eventos_validos_HF"]} for x in pop["por_leitura"]},
                          "contrastes_alterados": False},
         "frase_corrigida_influencia": "A decisão primária registrada (HF sustentado na leitura do grafo primário) não muda "
                                       "retroativamente. Isso não significa que a conclusão resista à retirada de uma unidade: "
                                       "sem MT11:GFD:4355 a média segue positiva, mas o limite inferior do IC98,33% fica abaixo de zero."}
    gravar(SAIDA / "adendo_v1.json", r)
    return {k: r[k] for k in ("nomes_das_leituras", "campo_pontos")}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["auditoria", "influencia", "populacao", "reparo", "reparo_3b", "matriz", "adendo", "tudo"])
    a = ap.parse_args()
    passos = ["auditoria", "influencia", "populacao", "reparo", "matriz", "adendo"] if a.acao == "tudo" else [a.acao]
    for p in passos:
        print("==", p)
        print(json.dumps(globals()[p](), ensure_ascii=False, indent=1, default=str)[:5000])
