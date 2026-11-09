"""v10 Frente V: lacuna observacional nas comunidades com geocodificacao, chamados e IBGE, na precisao real das fontes.

Recife (episodio de 28/05/2022; referencia S2 de 01/06/2022, v6):
  - grade v6 x setores IBGE 2022 (atributo de bairro do DBF) -> por BAIRRO: pontos, pessoas, fracao observavel,
    positivos S2 entre observaveis, escore medio do nucleo limpo (Qsep), fracao FCU;
  - SEDEC 2022 (v3): relatos de AGUA (categoria `relato_administrativo_agua`), sem duplicata de conteudo, na janela
    28/05-07/06; localizacao = BAIRRO declarado (precisao de bairro, nao de endereco); 15 tem vinculo nominal com FCU;
  - os relatos nao ensinaram nenhum modelo; a janela SEDEC foi usada na v6 para escolher a data da cena (dependencia
    de data declarada, nao de local);
  - triangulacao: Spearman entre relatos por bairro e (positivos S2, escore Q); bairros com relato e sem positivo S2
    observavel = parcela de presenca relatada nao identificada pela referencia;
  - pontos geocodificados harmonizados (precisao de geocodificador, raios 30/100/250 m como sensibilidade, v7) na
    janela: estado S2 dos pontos de grade no mesmo setor.
Curitiba (episodio de novembro/2023; S2 de 05/11/2023): chamados geocodificados 28/10-12/11/2023 por comunidade FCU
vs positivos/negativos/desconhecidos da referencia FCU.
Por comunidade FCU (cinco coortes): referencia utilizavel, desconhecido, positivos, consultas da Frente II (IDs e estados
revelados), carga de encaminhamento da Frente IV. Ausencia de chamado NUNCA e seco; coincidencia de escore e chamado
NAO e acuracia.
"""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_comum import AUD, TV, V6, V7, V8, V10, agora, congelar10, etapa, g10, nucleo_limpo

D = V10 / "v_social"
JANELA_PE = ("2022-05-28", "2022-06-07")
JANELA_PR = ("2023-10-28", "2023-11-12")


def bairros_pe():
    import shapefile
    z = zipfile.ZipFile(TV / "fontes/setores_2022/PE_setores_CD2022.zip")
    r = shapefile.Reader(dbf=io.BytesIO(z.read("PE_setores_CD2022.dbf")), encoding="utf-8")
    campos = [f[0] for f in r.fields[1:]]
    df = pd.DataFrame(r.records(), columns=campos)
    return df[[c for c in ["CD_SETOR", "CD_BAIRRO", "NM_BAIRRO", "CD_MUN"] if c in df]]


def norm(s):
    import unicodedata
    return s.fillna("").map(lambda t: unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().upper().strip())


def pps_preparar():
    """Plano de analise congelado ANTES de qualquer resposta + estimativas de desenho do que JA e conhecido (estado S2 da
    chave): parcela da populacao por estrato sem referencia (desconhecido/permanente) -- e exatamente onde a revisao
    humana teria de responder. Nenhuma resposta e simulada; a chave e conferida pelo sha e nao e copiada."""
    from v8_07_amostra_probabilistica import estimar
    from v10_comum import sha_hex
    P = V8 / "revisao_probabilistica_v2"
    est = json.loads((P / "estado.json").read_text(encoding="utf-8"))
    plano = congelar10(D / "plano_analise_pps.json", {
        "versao": "v10-pps-1", "pacote": "v8 revisao_probabilistica_v2", "sha_chave_esperado": est["sha_chave"],
        "estimandos": ["prevalencia de agua visivel por estrato (razao HT, EP entre setores com reposicao no 1o estagio)",
                       "P(agua visivel | S2 desconhecido) por estrato: a lacuna observacional estimada",
                       "concordancia revisor x S2 onde S2 tem estado", "AUC do nucleo limpo contra a resposta do revisor (nao observavel excluido e contado)"],
        "regras": ["nao observavel nunca vira negativo", "resposta so entra com fonte_imagem, data_imagem e resolucao_m preenchidas",
                   "a chave so e lida na analise, depois de o formulario fechado ter sha registrado"],
        "execucao": "somente se respostas_recebidas > 0"})
    k = pd.read_parquet(P / "chave_selada.parquet")
    out = {"plano_quando": plano["quando_utc"], "sha_chave_confere": sha_hex(P / "chave_selada.parquet") == est["sha_chave"],
           "respostas_recebidas": int(est["respostas_recebidas"]), "analise_com_respostas": "nao_executada_sem_respostas"}
    k = k.assign(sem_referencia=k.estado_primaria.isin(["desconhecido", "pre_desconhecido", "permanente"]).astype(float),
                 positivo_s2=k.estado_primaria.eq("positivo").astype(float))
    linhas = []
    for e, q in k.groupby("estrato"):
        for var in ["sem_referencia", "positivo_s2"]:
            p, ep = estimar(q, var)
            linhas.append({"estrato": e, "indicador": var, "estimativa": p, "ep": ep, "pontos_amostra": len(q), "setores": q.CD_SETOR.nunique()})
    g10(D / "pps_estimativas_s2_por_estrato.csv", pd.DataFrame(linhas))
    out["estimativas_s2"] = linhas
    return out


def main():
    from scipy.stats import spearmanr
    from v8_04_campanha import caracteristicas, coorte
    congelar10(D / "protocolo.json", {"versao": "v10-v-1", "janela_pe": JANELA_PE, "janela_pr": JANELA_PR, "precisao_sedec": "bairro",
                                      "regra": "ausencia de chamado nao e seco; coincidencia nao e acuracia"})
    t0 = time.time()
    Q = nucleo_limpo()
    # ---------------- Recife por bairro
    g = pd.read_parquet(V6 / "recife_avaliacao/grade.parquet")
    ref = pd.read_parquet(V6 / "recife_avaliacao/referencia.parquet", columns=["ponto_id", "estado_E2_primario"])
    g = g.merge(ref, on="ponto_id", how="left")
    bp = bairros_pe()
    g = g.merge(bp[["CD_SETOR", "NM_BAIRRO"]], on="CD_SETOR", how="left")
    dq = coorte("recife")
    sc = pd.Series(Q.escore(caracteristicas(dq)), index=dq.ponto_id)
    g["escore_Q"] = g.ponto_id.map(sc)
    e = g.estado_E2_primario.fillna("sem_estado")
    g = g.assign(observavel=e.isin(["positivo", "negativo"]), pos=e.eq("positivo"), fcu=g.classe_territorial.eq("fcu"), bairro=norm(g.NM_BAIRRO))
    pb = g.groupby("bairro").agg(pontos=("ponto_id", "size"), observavel=("observavel", "mean"), positivos=("pos", "sum"),
                                 fracao_fcu=("fcu", "mean"), escore_Q_medio=("escore_Q", "mean")).reset_index()
    pb["pos_por_observavel"] = pb.positivos / (pb.pontos * pb.observavel).replace(0, np.nan)
    sed = pd.read_parquet(TV / "v3/recife/registros_2022.parquet")
    agua = sed[sed.categoria.eq("relato_administrativo_agua") & ~sed.duplicado_conteudo.fillna(False)]
    dt = pd.to_datetime(agua.data_registro, errors="coerce")
    jan = agua[(dt >= JANELA_PE[0]) & (dt <= JANELA_PE[1])].assign(bairro=lambda f: norm(f.bairro))
    rel = jan.groupby("bairro").size().rename("relatos_janela").reset_index()
    tb = pb.merge(rel, on="bairro", how="outer")
    tb["relatos_janela"] = tb.relatos_janela.fillna(0)
    casados = tb[tb.pontos.notna()]
    sem_bairro_grade = int(tb[tb.pontos.isna()].relatos_janela.sum())
    rho_pos = spearmanr(casados.relatos_janela, casados.pos_por_observavel, nan_policy="omit")
    rho_q = spearmanr(casados.relatos_janela, casados.escore_Q_medio, nan_policy="omit")
    taxa = casados.relatos_janela / casados.pontos * 1000  # relatos por 1.000 pontos da grade: separa tamanho de intensidade
    rho_pos_t = spearmanr(taxa, casados.pos_por_observavel, nan_policy="omit")
    rho_q_t = spearmanr(taxa, casados.escore_Q_medio, nan_policy="omit")
    nao_ident = casados[(casados.relatos_janela > 0) & (casados.positivos.fillna(0) == 0)]
    g10(D / "recife_por_bairro.csv", tb)
    # geocodificados harmonizados na janela
    geo = pd.read_parquet(AUD / "geocodificacao_brasileira_contextualizada.parquet")
    gd = pd.to_datetime(geo.data_declarada, errors="coerce")
    gpe = geo[(geo.UF == "PE") & (gd >= JANELA_PE[0]) & (gd <= JANELA_PE[1])]
    gs = []
    for _, r in gpe.iterrows():
        m = g.CD_SETOR.eq(str(r.ibge_CD_SETOR))
        gs.append({"id_origem": r.id_origem, "data": str(r.data_declarada)[:10], "bairro": r.ibge_NM_BAIRRO, "fcu": r.ibge_CD_FCU, "pontos_setor": int(m.sum()),
                   "positivos_setor": int(g[m].pos.sum()), "observaveis_setor": int(g[m].observavel.sum()), "sensivel_borda_30m": bool(r.sensivel_borda_30m)})
    g10(D / "recife_geocodificados_janela.csv", pd.DataFrame(gs))
    # ---------------- Curitiba: chamados geocodificados x FCU
    gpr = geo[(geo.UF == "PR") & (gd >= JANELA_PR[0]) & (gd <= JANELA_PR[1])]
    cc = pd.read_parquet(V8 / "contrato/contrato_coortes_brasileiras.parquet", columns=["ponto_id", "coorte", "CD_FCU", "classe_territorial", "estado"]).query("coorte == 'curitiba'")
    fcu_ref = cc[cc.classe_territorial.eq("fcu")].groupby("CD_FCU").estado.value_counts().unstack(fill_value=0).reset_index()
    ch = gpr[gpr.ibge_CD_FCU.notna()].groupby("ibge_CD_FCU").size().rename("chamados_janela").reset_index().rename(columns={"ibge_CD_FCU": "CD_FCU"})
    ch["CD_FCU"] = ch.CD_FCU.astype(str)
    fcu_ref["CD_FCU"] = fcu_ref.CD_FCU.astype(str)
    cur = fcu_ref.merge(ch, on="CD_FCU", how="outer").fillna({"chamados_janela": 0})
    g10(D / "curitiba_fcu_chamados.csv", cur)
    # ---------------- tabela por comunidade FCU (cinco coortes)
    con = pd.read_parquet(V8 / "contrato/contrato_coortes_brasileiras.parquet", columns=["ponto_id", "coorte", "CD_FCU", "NM_FCU", "classe_territorial", "estado"])
    con = con[con.classe_territorial.eq("fcu")]
    es = con.estado.fillna("sem_estado")
    tab = con.assign(observavel=es.isin(["positivo", "negativo"]), desconhecido=es.isin(["desconhecido", "pre_desconhecido"]), pos=es.eq("positivo"),
                     neg=es.eq("negativo")).groupby(["coorte", "CD_FCU", "NM_FCU"]).agg(pontos=("ponto_id", "size"), observavel=("observavel", "mean"),
                                                                                       desconhecido=("desconhecido", "mean"), positivos=("pos", "sum"),
                                                                                       negativos=("neg", "sum")).reset_index()
    tab["grupo"] = "FCU:" + tab.CD_FCU.astype(str)
    cq = V10 / "ii_ciclo/consultas.parquet"
    if cq.exists():
        q = pd.read_parquet(cq, columns=["coorte", "grupo", "estrategia", "semente", "estado_revelado"])
        q = q[q.estrategia.eq("margem_v9_gate")]
        qq = q.groupby(["coorte", "grupo"]).agg(consultas_margem_6sementes=("estado_revelado", "size"),
                                                positivos_revelados=("estado_revelado", lambda s: int((s == "positivo").sum())),
                                                sem_rotulo_revelado=("estado_revelado", lambda s: int((~s.isin(["positivo", "negativo"])).sum()))).reset_index()
        tab = tab.merge(qq, on=["coorte", "grupo"], how="left")
    cp = V10 / "iv_competencia/custo_por_comunidade_orc05.csv"
    if cp.exists():
        c4 = pd.read_csv(cp)
        c4 = c4[c4.metodo.eq("margem")].rename(columns={"destino": "coorte"})[["coorte", "grupo", "carga", "positivos_aceitos_baixos", "positivos_encaminhados", "falsos_alertas"]]
        tab = tab.merge(c4, on=["coorte", "grupo"], how="left")
    tab["situacao"] = np.select([tab.positivos.ge(5) & tab.negativos.ge(5), tab.positivos.gt(0), tab.observavel.gt(0)],
                                ["avaliavel_5x5", "positivo_raro", "so_negativos_observados"], "sem_referencia")
    g10(D / "por_comunidade_fcu.csv", tab)
    resumo = tab.groupby(["coorte", "situacao"]).agg(comunidades=("grupo", "size"), pontos=("pontos", "sum")).reset_index()
    g10(D / "resumo_situacao.csv", resumo)
    # ---------------- PPS v2 (RMPA): analise preparada; a chave nunca vai ao formulario e respostas sao 0
    pps = pps_preparar()
    g10(D / "resultado.json", {"quando_utc": agora(), "segundos": time.time() - t0, "pps": pps,
                               "recife": {"relatos_agua_janela": int(len(jan)), "relatos_sem_bairro_na_grade": sem_bairro_grade, "bairros_com_relato": int((casados.relatos_janela > 0).sum()),
                                          "spearman_relatos_x_pos_s2": [float(rho_pos.statistic), float(rho_pos.pvalue)],
                                          "spearman_relatos_x_escore_Q": [float(rho_q.statistic), float(rho_q.pvalue)],
                                          "spearman_taxa_x_pos_s2": [float(rho_pos_t.statistic), float(rho_pos_t.pvalue)],
                                          "spearman_taxa_x_escore_Q": [float(rho_q_t.statistic), float(rho_q_t.pvalue)],
                                          "bairros_na_grade": int(len(casados)),
                                          "bairros_relato_sem_positivo_s2": int(len(nao_ident)), "geocodificados_janela": len(gs)},
                               "curitiba": {"chamados_geocodificados_janela": int(len(gpr)), "em_fcu": int(gpr.ibge_CD_FCU.notna().sum()),
                                            "fcu_com_chamado": int((cur.chamados_janela > 0).sum()),
                                            "fcu_com_chamado_sem_positivo_s2": int(((cur.chamados_janela > 0) & (cur.get("positivo", 0) == 0)).sum())}})
    etapa("v_social", segundos=time.time() - t0)
    print(resumo.to_string())


if __name__ == "__main__":
    main()
