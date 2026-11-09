"""MT11, emendas de 17/10/2026 registradas ANTES de qualquer escore ou AUC.

O protocolo de 13:47:03 UTC e o código `mt11_confirmatorio_transdutivo.py` ficam intactos (hashes
conferidos). Este módulo acrescenta, sem reescrever o congelamento:

  validar        confere TODOS os hashes registrados (código, candidatos, poder, pesos Q, F JSON e
                 pickle) e os de cada emenda; qualquer divergência recusa a execução
  completude     tabela por evento, sem escore nem AUC: pontos por classe, terreno íntegro, atributos,
                 perdas do espelho, S1 (órbita, datas, admitidas, regra), populações Q e HF, e o estado
                 concluído | exclusão técnica prevista | pendência bloqueante
  distribuicao_q quanto muda o F_transf ao trocar o Q fora do componente pelo Q do modelo único, só
                 em dados conhecidos (MT10), sem recalibrar
  avaliar        uma vez: três leituras lado a lado — `congelada_original` (unidade do centro do tile,
                 como registrado), `emendada_primaria` (componentes do grafo comum, sem eventos ligados a
                 desenvolvimento/examinados) e `sensibilidade_exutorio` (MAIN_BAS em qualquer data)
Funções de escore, medida e contraste são as ORIGINAIS (importadas, não copiadas).
Uso: python scripts/metodo/mt11_emenda.py registrar|validar|completude|distribuicao_q|avaliar
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for s in ("metodo", "ia", "geo", "servico", "suscetibilidade", "terreno", "sar"):
    sys.path.insert(0, str(REPO / "scripts" / s))
import mt11_confirmatorio_transdutivo as m11  # noqa: E402
import registro_aberturas as reg  # noqa: E402
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402

SAIDA = m11.SAIDA
EMENDAS = SAIDA / "emendas.json"
GRAFO = SAIDA / "independencia" / "grafo.json"
COMPLETUDE = SAIDA / "completude.json"
ATRIBUTOS = SAIDA / "atributos_e8"      # emenda 8: S1 pela órbita que cobre os pontos; `atributos/` original preservado
AVALIACAO = "mt11_confirmatorio_transdutivo"
RASTERS = ("dem_filled.tif", "hand_dinf.tif", "slope_deg_wbt.tif", "twi_dinf.tif", "streams_dinf.tif", "dinf_accum_cells.tif")
CODIGO_EMENDA = ["scripts/metodo/mt11_emenda.py", "scripts/metodo/registro_aberturas.py", "scripts/metodo/mt11_independencia.py",
                 "scripts/metodo/mt11_fabdem_transporte.py", "scripts/metodo/mt11_terreno_executar.py",
                 "scripts/ia/rodada3_fabdem_gee.py", "scripts/metodo/mt01_independencia.py", "scripts/metodo/mt01_populacao.py"]
ATRIBUTOS_TEMPO = [
    {"atributo": "rótulo GFD (MODIS)", "referencia_temporal": "janela do evento", "funcao": "desfecho", "disponivel_antes_do_evento": False},
    {"atributo": "S1 VV (F_v1, F_e1)", "referencia_temporal": "[início-425 d, início-60 d]", "funcao": "entrada de F", "disponivel_antes_do_evento": True},
    {"atributo": "FABDEM V1-2 e derivados (HAND, TWI, declive, distâncias, declive_p50)", "referencia_temporal": "Copernicus GLO-30 (aquisição 2011-2015) e edição 2021-2022",
     "funcao": "entradas do Q", "disponivel_antes_do_evento": "parcial: produto publicado depois; forma do terreno tratada como estática"},
    {"atributo": "GHSL built_surface (construido_frac)", "referencia_temporal": "época <= ano do evento (2015)", "funcao": "entrada do Q", "disponivel_antes_do_evento": True},
    {"atributo": "SoilGrids areia 0-30 cm", "referencia_temporal": "estática (publicação 2020)", "funcao": "entrada do Q", "disponivel_antes_do_evento": "estática; publicada depois"},
    {"atributo": "ESA WorldCover v200", "referencia_temporal": "2021", "funcao": "observabilidade HF (árvore, construído) e qualidade",
     "disponivel_antes_do_evento": False},
    {"atributo": "JRC GSW 1.4 occurrence", "referencia_temporal": "1984-2021", "funcao": "água permanente (espelho do ds06), observabilidade HF, qualidade",
     "disponivel_antes_do_evento": False},
]


def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def escrever(p, obj):
    m11.escrever(p, obj)


# ------------------------------------------------------------------ emendas

def _hashes(caminhos):
    return {c: sha256_arquivo(REPO / c) for c in caminhos}


def registrar() -> dict:
    """Grava as emendas uma vez; recusa reescrever uma emenda já registrada com conteúdo diferente."""
    sel = ler(SAIDA / "selecao.json")
    g = ler(GRAFO)
    artefatos = {str(p.relative_to(REPO)): sha256_arquivo(p) for p in
                 (GRAFO, SAIDA / "independencia" / "nos.parquet", SAIDA / "fabdem_transporte.json",
                  SAIDA / "fabdem_transporte_regional.json", SAIDA / "fabdem_caminho_gravacao.json", SAIDA / "gfd_datas.json")}
    ja_aberto = {"area_do_rotulo_consultada": [e["evento_id"] for e in sel["examinados"]],
                 "pontos_amostrados": [e["evento_id"] for e in sel["eventos"]],
                 "escores_ou_auc_lidos": [], "quando_selecao_utc": sel["quando_utc"]}
    emendas = [
        {"n": 1, "tipo": "correcao_de_implementacao",
         "defeito": "conferir_registro() só checava código e candidatos; avaliar() não conferia pesos Q/F, descartava em silêncio "
                    "evento sem atributo e gravava no ledger só ao fim, sem fase; a seleção consultou o rótulo de 46 eventos sem registro",
         "correcao": "validar() confere todos os hashes; completude explícita antes dos escores; registro por evento e fase "
                     "(registro_aberturas.py) com conciliação dos 46 eventos; avaliar() original NÃO será chamado",
         "efeito_na_populacao": "nenhum"},
        {"n": 2, "tipo": "transporte_de_dado",
         "defeito": "data.bris.ac.uk recusa conexão desde 14:05 UTC; a cadeia rotulou 3 janelas como 'oceano'",
         "correcao": "mesmo produto FABDEM V1-2 pelo espelho GEE projects/sat-io/open-datasets/FABDEM, só quando a origem não responde, "
                     "com rodada3_fabdem_gee.baixar gravando em temporário e publicando após validação; cadeia_em_lote, grade e derivação inalteradas",
         "evidencia": "100 recortes de 1024x1024 bit-idênticos em 6 regiões (40 + 60) e 5 janelas regravadas idênticas pixel a pixel, origem "
                      "com diferença <= 1e-10 célula (ponto flutuante); a afirmação anterior '~45% dos tiles diferem' veio de memória de "
                      "01/10 (18 de 40 arquivos) sem artefato localizado e não se reproduz",
         "alcance": "janelas conferidas; não certifica tile não conferido", "efeito_na_populacao": "nenhum"},
        {"n": 3, "tipo": "mudanca_de_populacao_e_unidade",
         "defeito": "MAIN_BAS lido do nível 8 enquanto MT01/rodada 3 usam o nível 6 (mt01_populacao.localizar_bacias): a conferência "
                    "contra o MT era vazia por construção; unidade pelo centro do tile ignora a área amostrada",
         "correcao": "grafo comum (mt01_independencia.grafo) sobre pontos efetivamente amostrados, desenvolvimento, examinados, "
                     "quarentena GFD, ds06 e rodada 3",
         "analises": {"congelada_original": "unidade = bacia do centro do tile (nível 8), todos os eventos avaliáveis — como registrado",
                      "emendada_primaria": "componentes do desenho primário; exclui eventos ligados a desenvolvimento/examinados",
                      "sensibilidade_exutorio": "componentes do desenho exutório; exclui ligados nesse desenho"},
         "contagens": {d: {k: g["desenhos"][d][k] for k in ("componentes_mt11", "componentes_mt11_livres", "eventos_livres")}
                       for d in ("primario", "exutorio")},
         "eventos_excluidos_primario": g["desenhos"]["primario"]["eventos_ligados_a_examinado"],
         "eventos_excluidos_exutorio": g["desenhos"]["exutorio"]["eventos_ligados_a_examinado"],
         "unidades_iniciais_pelo_centro": len({e["unidade"] for e in sel["eventos"]}),
         "efeito_na_populacao": "a análise emendada usa outra população e outras unidades; a congelada é reportada ao lado"},
        {"n": 4, "tipo": "escopo_temporal", "tabela": ATRIBUTOS_TEMPO,
         "leitura": "análise retrospectiva: F só com S1 anterior ao evento, máscaras auxiliares posteriores (WorldCover 2021, JRC 1984-2021)",
         "analise_estritamente_previa": "não executada nesta rodada: exigiria reconstruir observabilidade e qualidade com dados datados "
                                        "(p.ex. JRC MonthlyHistory até o evento, cobertura anual) e um congelamento próprio; fica como lacuna",
         "efeito_na_populacao": "nenhum"},
        {"n": 5, "tipo": "declaracao_de_distribuicao",
         "texto": "F_transf foi ajustado com Q fora do componente (MT06) e é aplicado com Q do modelo único; a diferença é medida só "
                  "em dados conhecidos por distribuicao_q(); nenhuma recalibração com rótulos novos",
         "efeito_na_populacao": "nenhum"},
        {"n": 6, "tipo": "correcao_de_documentacao",
         "texto": "datas_por_orbita da pré-triagem é datas distintas / órbitas distintas no centróide, não a contagem da órbita usada; "
                  "a lista de candidatos fica como está e a órbita escolhida, datas e admitidas por evento vão para a completude",
         "efeito_na_populacao": "nenhum"},
        {"n": 7, "tipo": "interpretacao",
         "texto": "a regra congelada (média >= +0,01 e limite inferior > 0) sustenta ganho positivo com média estimada acima da margem; "
                  "dizer que o ganho excede +0,01 exigiria limite inferior > +0,01. Regra inalterada",
         "efeito_na_populacao": "nenhum"},
    ]
    doc = {"tipo": "mt11_emendas", "protocolo_original_sha256": sha256_arquivo(SAIDA / "protocolo.json"),
           "registradas_antes_de_escores": True, "ja_aberto_antes_das_emendas": ja_aberto,
           "codigo_sha256": _hashes(CODIGO_EMENDA), "artefatos_sha256": artefatos, "emendas": emendas}
    if EMENDAS.exists():
        old = ler(EMENDAS)
        if {k: v for k, v in old.items() if k != "quando_utc"} != m11.mt.rel.limpo(doc):
            raise ValueError("emendas já registradas com outro conteúdo; acrescente uma emenda nova, não reescreva")
        return old
    doc["quando_utc"] = agora_utc()
    escrever(EMENDAS, doc)
    return {"emendas": len(emendas), "quando_utc": doc["quando_utc"]}


def registrar_adicional(emenda: dict, codigo: list[str], artefatos: list[Path] = ()) -> dict:
    """Acrescenta uma emenda posterior com o retrato do código vigente; nunca reescreve as anteriores."""
    e = ler(EMENDAS)
    e.setdefault("adicionais", [])
    nova = {**emenda, "codigo_sha256": _hashes(codigo),
            "artefatos_sha256": {str(Path(a).relative_to(REPO)): sha256_arquivo(a) for a in artefatos}}
    for a in e["adicionais"]:
        if a["n"] == emenda["n"]:
            if {k: v for k, v in a.items() if k != "quando_utc"} != m11.mt.rel.limpo(nova):
                raise ValueError(f"emenda {emenda['n']} já registrada com outro conteúdo")
            return a
    nova["quando_utc"] = agora_utc()
    e["adicionais"].append(nova)
    escrever(EMENDAS, e)
    return nova


def registrar_emenda8() -> dict:
    copia = sorted((SAIDA / "codigo_emendas").glob("mt11_emenda_*.py"))
    return registrar_adicional({
        "n": 8, "tipo": "correcao_de_implementacao",
        "defeito": "_pilha_s1 escolhe a órbita com mais datas que TOCAM o tile; num tile de 1 grau ela cobria 2% (GFD:4483) e 9% "
                   "(GFD:4434) dos pontos e a admissão (>= 90% dos pontos com leitura) recusou todas as datas dos 3 primeiros eventos",
        "o_que_ja_foi_aberto": "atributos de 3 eventos extraídos com a regra original (GFD:4483, 4434, 4665), preservados em atributos/; "
                               "nenhum escore ou AUC calculado",
        "correcao": "órbita que cobre mais pontos pela pegada das cenas; admissão sobre os pontos da pegada; fora dela F ausente; "
                    "qualidade nos pixels aleatórios da mesma pegada (mt11_emenda_s1.py); demais atributos com as funções originais",
        "efeito_na_populacao": "população HF passa a ser os pontos na pegada da órbita escolhida; Q inalterado",
        "versao_anterior_do_modulo": [str(c.relative_to(REPO)) for c in copia]},
        CODIGO_EMENDA + ["scripts/metodo/mt11_emenda_s1.py"], copia)


# ------------------------------------------------------------------ validar

def validar() -> dict:
    erros = []
    p = ler(SAIDA / "protocolo.json")
    for f, h in p["codigo_sha256"].items():
        if sha256_arquivo(REPO / f) != h:
            erros.append(f"código do pré-registro mudou: {f}")
    for chave, arq in (("candidatos_csv_sha256", SAIDA / "candidatos.csv"), ("poder_sha256", SAIDA / "poder.json"),
                       ("f_transf_sha256", SAIDA / "f_transf.json"), ("q_pkl_sha256", m11.Q_PKL)):
        if sha256_arquivo(arq) != p[chave]:
            erros.append(f"artefato divergente: {chave}")
    if sha256_arquivo(SAIDA / "f_transf.pkl") != ler(SAIDA / "f_transf.json")["pkl_sha256"]:
        erros.append("pickle do F_transf divergente")
    if not EMENDAS.exists():
        erros.append("emendas ausentes")
    else:
        e = ler(EMENDAS)
        if e["protocolo_original_sha256"] != sha256_arquivo(SAIDA / "protocolo.json"):
            erros.append("protocolo diverge do registrado nas emendas")
        vigente = {**e["codigo_sha256"], **e["artefatos_sha256"]}
        for a in e.get("adicionais", []):           # cada emenda posterior traz o retrato do código que passou a valer
            vigente.update(a.get("codigo_sha256", {}))
            vigente.update(a.get("artefatos_sha256", {}))
        for f, h in vigente.items():
            if sha256_arquivo(REPO / f) != h:
                erros.append(f"emenda: arquivo mudou após o registro: {f}")
    if erros:
        raise ValueError("integridade recusada: " + "; ".join(erros))
    return {"integridade": "ok", "conferidos": len(p["codigo_sha256"]) + 5 + len(vigente),
            "emendas_adicionais": [a["n"] for a in ler(EMENDAS).get("adicionais", [])]}


# ------------------------------------------------------------------ completude

def terreno_ok(pasta: Path) -> dict:
    import rasterio
    if not (pasta / "run_manifest.json").exists():
        return {"estado": "ausente", "motivo": "sem run_manifest.json"}
    faltam = [r for r in RASTERS if not (pasta / r).exists()]
    if faltam:
        return {"estado": "parcial", "motivo": "faltam " + ",".join(faltam)}
    geo, frac = set(), 0.
    try:
        for r in RASTERS:
            with rasterio.open(pasta / r) as s:
                geo.add((s.crs.to_string(), s.shape, tuple(round(v, 6) for v in tuple(s.transform)[:6])))
                if r == "dem_filled.tif":
                    a = s.read(1, out_shape=(max(1, s.height // 8), max(1, s.width // 8)))
                    v = (a != s.nodata) if s.nodata is not None else np.isfinite(a)
                    frac = float(v.mean())
    except Exception as e:  # noqa: BLE001
        return {"estado": "parcial", "motivo": f"raster ilegível: {type(e).__name__}"}
    if len(geo) != 1:
        return {"estado": "parcial", "motivo": "grades divergentes entre derivados"}
    if frac < .5:
        return {"estado": "parcial", "motivo": f"DEM válido em {frac:.2f} da janela"}
    return {"estado": "completo", "fracao_dem_valido": frac}


def estado_do_evento(d: pd.DataFrame, meta: dict) -> dict:
    """Populações Q e HF pelas regras pré-registradas; conta classes, nunca calcula escore."""
    s1 = meta["s1"]
    regra = s1.get("regra")
    F = np.where(regra == "e1", d.F_e1, np.where(regra == "v1", d.F_v1, np.nan)) if regra else np.full(len(d), np.nan)
    obs = ((d.arvore <= m11.CFG["arvore_max"]) & (d.construido_wc <= m11.CFG["construido_max"])
           & (d.slope_deg <= m11.CFG["declive_max_graus"]) & (d.jrc_occ_s1 < m11.CFG["jrc_permanente"])
           & (d.n_pre >= m11.CFG["min_datas_pre"]) & np.isfinite(F)).to_numpy()
    q_ok = len(d) >= m11.CFG["minimo_pontos_evento"] and d.classe.nunique() == 2
    hf = d[obs]
    hf_ok = len(hf) >= m11.CFG["minimo_pontos_evento"] and hf.classe.nunique() == 2
    motivos = []
    if not q_ok:
        motivos.append("Q: < 20 pontos ou uma classe após o espelho")
    if not hf_ok:
        motivos.append("HF: " + ("referência SAR reprovada na qualidade ou sem S1" if not regra else "< 20 pontos observáveis ou uma classe"))
    return {"linhas_pos_espelho": len(d), "perdas_espelho": meta.get("saiu_no_espelho"), "orbita": s1.get("orbita"),
            "datas_s1": s1.get("datas"), "datas_admitidas": s1.get("datas_admitidas"), "regra": regra,
            "qualidade_v1": (s1.get("qualidade") or {}).get("v1", {}).get("passa"),
            "qualidade_e1": (s1.get("qualidade") or {}).get("e1", {}).get("passa"),
            "fracao_F_preenchido": float(np.isfinite(F).mean()) if len(d) else 0., "fracao_n_pre_ge_10": float((d.n_pre >= 10).mean()) if len(d) else 0.,
            "pontos_Q": len(d), "classes_Q": int(d.classe.nunique()), "pontos_HF": len(hf), "classes_HF": int(hf.classe.nunique()),
            "avaliavel_Q": bool(q_ok), "avaliavel_HF": bool(hf_ok),
            "estado": "concluido" if not motivos else "exclusao_tecnica_prevista", "motivo": "; ".join(motivos) or None}


def exigir_completude(c: dict) -> None:
    if c.get("pendencias"):
        raise SystemExit(f"pendências bloqueantes: {len(c['pendencias'])}; avaliação recusada")


def completude() -> dict:
    validar()
    sel = ler(SAIDA / "selecao.json")
    g = ler(GRAFO)
    comp = {d: {e["evento_id"][5:]: e for e in g["desenhos"][d]["eventos"]} for d in ("primario", "exutorio")}
    linhas = []
    for ev in sel["eventos"]:
        t = ev["tiles"][0]
        csvp = SAIDA / "pontos" / f"pontos_gfd{ev['id']}_{t['tile']}.csv"
        p = pd.read_csv(csvp)
        tile = p.tile.iloc[0]
        ter = terreno_ok(m11.TER / tile)
        r = {"evento_id": ev["evento_id"], "pais": ev["pais"], "tile": tile, "unidade_centro": ev["unidade"],
             "componente_primario": comp["primario"][ev["evento_id"]]["componente"],
             "ligado_primario": comp["primario"][ev["evento_id"]]["ligado_a_desenvolvimento_ou_examinado"],
             "componente_exutorio": comp["exutorio"][ev["evento_id"]]["componente"],
             "ligado_exutorio": comp["exutorio"][ev["evento_id"]]["ligado_a_desenvolvimento_ou_examinado"],
             "solicitados_por_classe": m11.CFG["pontos_por_classe_no_tile"],
             "obtidos_inundacao": int((p.classe_obs == "INUNDACAO").sum()), "obtidos_seco": int((p.classe_obs != "INUNDACAO").sum()),
             "terreno": ter["estado"], "terreno_motivo": ter.get("motivo")}
        arq = ATRIBUTOS / f"{ev['evento_id'].replace(':', '_')}.parquet"
        if arq.exists() and arq.with_suffix(".json").exists():
            meta = ler(arq.with_suffix(".json"))
            if meta["sha256"] != sha256_arquivo(arq):
                r.update(estado="pendencia_bloqueante", motivo="atributos com hash divergente")
                linhas.append(r)
                continue
            r.update(estado_do_evento(pd.read_parquet(arq), meta))

        elif ter["estado"] != "completo":
            r.update(estado="pendencia_bloqueante", motivo=f"terreno {ter['estado']}: {ter.get('motivo')}")
        else:
            r.update(estado="pendencia_bloqueante", motivo="atributos não extraídos")
        linhas.append(r)
    t = pd.DataFrame(linhas)
    t.to_csv(SAIDA / "completude.csv", index=False)
    res = {"tipo": "mt11_completude_sem_escores", "quando_utc": agora_utc(), "emendas_sha256": sha256_arquivo(EMENDAS),
           "eventos": len(t), "por_estado": t.estado.value_counts().to_dict(),
           "pendencias": t[t.estado == "pendencia_bloqueante"][["evento_id", "motivo"]].to_dict("records"),
           "csv_sha256": sha256_arquivo(SAIDA / "completude.csv"), "nenhum_escore_calculado": True}
    escrever(COMPLETUDE, res)
    return {k: res[k] for k in ("eventos", "por_estado", "pendencias")}


# ------------------------------------------------------------------ distribuição de Q (conhecido)

def distribuicao_q() -> dict:
    """Material conhecido (MT10 pré-evento): Q fora do componente x Q do modelo único, e o efeito no F_transf."""
    import mt06_cobertura_observabilidade as e6
    import mt10_sensibilidade_pre_evento as s10
    t, _ = s10.tabela_pre_evento()
    base = pd.read_parquet(e6.SAIDA / "base.parquet")
    x = base.set_index("ponto_id").loc[t.ponto_id]
    with m11.Q_PKL.open("rb") as f:
        q = pickle.load(f)["Q_conexao"].escore(x[[c for c in x.columns if c.startswith("f_") or c == "z_relevo"]].reset_index(drop=True))
    t = t.assign(Q_full=q)
    fexp = ler(SAIDA / "f_transf.json")["exportado"]
    t["Fq_oof"] = m11.escore_exportado(pd.DataFrame({"Q": t.Q.to_numpy(), "F": t.F.to_numpy()}), fexp)
    t["Fq_full"] = m11.escore_exportado(pd.DataFrame({"Q": t.Q_full.to_numpy(), "F": t.F.to_numpy()}), fexp)
    from scipy.stats import spearmanr
    rho_ev = [spearmanr(g.Fq_oof, g.Fq_full).statistic for _, g in t.groupby("evento_id") if len(g) > 5]
    auc = lambda c: np.mean([m11.mt.e1.auc(g.classe.to_numpy(), g[c].to_numpy()) for _, g in t.groupby("evento_id")  # noqa: E731
                             if g.classe.nunique() == 2])
    r = {"tipo": "mt11_distribuicao_q_conhecido", "quando_utc": agora_utc(), "pontos": len(t), "eventos": int(t.evento_id.nunique()),
         "Q_oof": {"media": float(t.Q.mean()), "dp": float(t.Q.std())}, "Q_full": {"media": float(t.Q_full.mean()), "dp": float(t.Q_full.std())},
         "pearson_Q": float(np.corrcoef(t.Q, t.Q_full)[0, 1]), "spearman_F_transf_mediana_por_evento": float(np.median(rho_ev)),
         "auc_media_eventos_conhecidos": {"F_transf_com_Q_oof": float(auc("Fq_oof")), "F_transf_com_Q_full": float(auc("Fq_full")),
                                          "Q_oof": float(auc("Q")), "Q_full": float(auc("Q_full"))},
         "nao_e": "dados de desenvolvimento conhecidos; Q_full viu esses pontos no ajuste (otimista); nada recalibrado"}
    escrever(SAIDA / "distribuicao_q.json", r)
    return r


# ------------------------------------------------------------------ avaliar (uma vez)

def avaliar() -> dict:
    validar()
    if (SAIDA / "resultado_emendado.json").exists():
        return ler(SAIDA / "resultado_emendado.json")
    c = ler(COMPLETUDE)
    exigir_completude(c)
    tab = pd.read_csv(SAIDA / "completude.csv")
    if sha256_arquivo(SAIDA / "completude.csv") != c["csv_sha256"]:
        raise ValueError("completude mudou depois de fechada")
    psha = sha256_arquivo(SAIDA / "protocolo.json")
    eventos = tab[tab.estado != "pendencia_bloqueante"].evento_id.tolist()
    reg.registrar(eventos, "gfd", "escores_avaliados", "intencao", AVALIACAO, psha)
    f_exp = ler(SAIDA / "f_transf.json")["exportado"]
    partes = []
    for ev in eventos:
        arq = ATRIBUTOS / f"{ev.replace(':', '_')}.parquet"
        d = pd.read_parquet(arq)
        d["regra_unidade"] = ler(arq.with_suffix(".json"))["s1"].get("regra")
        partes.append(d)
    s = m11.escores(pd.concat(partes, ignore_index=True), f_exp)
    s.to_parquet(SAIDA / "escores_emendado.parquet", index=False)
    obs = ((s.arvore <= m11.CFG["arvore_max"]) & (s.construido_wc <= m11.CFG["construido_max"]) & (s.slope_deg <= m11.CFG["declive_max_graus"])
           & (s.jrc_occ_s1 < m11.CFG["jrc_permanente"]) & (s.n_pre >= m11.CFG["min_datas_pre"]) & s.F.notna())
    t = tab.set_index("evento_id")
    leituras = {"congelada_original": t.unidade_centro,
                "emendada_primaria": t.componente_primario.where(~t.ligado_primario.astype(bool)),
                "sensibilidade_exutorio": t.componente_exutorio.where(~t.ligado_exutorio.astype(bool))}
    res = {}
    for nome, mapa in leituras.items():
        sub = s.assign(unidade=s.evento_id.map(mapa)).dropna(subset=["unidade"])
        sub_hf = sub[obs.loc[sub.index]]
        mq = m11.medir(sub, ["A0_regra_hand", "Q_base", "Q_conexao"])
        mf = m11.medir(sub_hf, ["Q_conexao", "F_transf"])
        uq = mq.groupby(["unidade", "modelo"]).auc.mean().unstack()
        uf = mf.groupby(["unidade", "modelo"]).auc.mean().unstack() if len(mf) else pd.DataFrame(columns=["Q_conexao", "F_transf"])
        res[nome] = {
            "HQ1": {**m11.contraste(uq.Q_conexao - uq.A0_regra_hand), "eventos": int(mq.evento_id.nunique()), "pontos": int(len(sub))},
            "HQ2": {**m11.contraste(uq.Q_conexao - uq.Q_base), "eventos": int(mq.evento_id.nunique()), "pontos": int(len(sub))},
            "HF": {**(m11.contraste(uf.F_transf - uf.Q_conexao) if len(uf) else {"n": 0, "veredito": "not_testable"}),
                   "eventos": int(mf.evento_id.nunique()) if len(mf) else 0, "pontos": int(len(sub_hf))},
            "auc_media_unidades": {"Q": uq.mean().to_dict(), "HF": uf.mean().to_dict() if len(uf) else {}}}
    r = {"tipo": "MT11_resultado_emendado", "quando_utc": agora_utc(), "protocolo_sha256": psha,
         "emendas_sha256": sha256_arquivo(EMENDAS), "completude_sha256": sha256_arquivo(COMPLETUDE),
         "leituras": res, "decisao_congelada": "sustentado: média >= +0,01 e limite inferior > 0; ganho_excluido: limite superior < +0,01; "
                                               "inferior: limite superior < 0; IC 98,33% por unidade",
         "eventos_avaliados": eventos, "modelo_promovido": False}
    escrever(SAIDA / "resultado_emendado.json", r)
    reg.registrar(eventos, "gfd", "escores_avaliados", "concluido", AVALIACAO, psha)
    return {n: {h: {k: v.get(k) for k in ("n", "media", "ic", "veredito", "a_favor", "eventos", "pontos")} for h, v in x.items()
                if h != "auc_media_unidades"} for n, x in res.items()}


def conciliar_registro() -> dict:
    """Registra no ledger o que já foi aberto: 46 eventos com área do rótulo consultada e 45 com pontos amostrados."""
    sel = ler(SAIDA / "selecao.json")
    psha = sha256_arquivo(SAIDA / "protocolo.json")
    a = reg.registrar([e["evento_id"] for e in sel["examinados"]], "gfd", "rotulo_consultado_na_selecao", "concluido", AVALIACAO,
                      psha, quando_original_utc=sel["quando_utc"], detalhe={"o_que": "área por classe nos tiles de 1 grau"})
    pts = sorted((SAIDA / "pontos").glob("*.csv"))
    b = reg.registrar([e["evento_id"] for e in sel["eventos"]], "gfd", "pontos_amostrados", "concluido", AVALIACAO, psha,
                      quando_original_utc=max(pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC") for p in pts).isoformat(),
                      detalhe={"o_que": "stratifiedSample 1300 por classe, classe do ponto conhecida"})
    c = reg.registrar(["GFD:3452"], "gfd", "atributos_extraidos_fumaca", "concluido", AVALIACAO, psha,
                      quando_original_utc="2026-10-17T13:40:00Z",
                      detalhe={"o_que": "fumaça em evento já examinado (gfdq03), sem escore", "aproximado": True})
    return {"selecao": a, "pontos": b, "fumaca": c}


if __name__ == "__main__":
    fases = {"registrar": registrar, "registrar_emenda8": registrar_emenda8, "validar": validar, "conciliar": conciliar_registro, "completude": completude,
             "distribuicao_q": distribuicao_q, "avaliar": avaliar}
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fase", choices=list(fases))
    print(json.dumps(fases[ap.parse_args().fase](), ensure_ascii=False, default=str))
