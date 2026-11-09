"""Figuras e tabelas do artigo, geradas so dos artefatos reais da linhagem.

Cada figura sai em PNG e SVG e tem legenda em figuras/legendas.md com fonte,
populacao, data, unidade, procedimento e limitacao. Nenhum valor simulado.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pa_comum import CHARTER, EPSG_PETROPOLIS, ESTUDO, OUT, TERRENO, Etapa, gravar_json, ler_json  # noqa: E402

FIG = OUT / "figuras"
TAB = OUT / "tabelas"
RES = OUT / "resultados"
TER = OUT / "territorio"
C = {"azul": "#2a78d6", "laranja": "#eb6834", "aqua": "#1baf7a", "amarelo": "#eda100", "magenta": "#e87ba4",
     "verde": "#008300", "violeta": "#6250d6", "vermelho": "#e34948", "cinza": "#8a8986", "tinta": "#0b0b0b",
     "tinta2": "#52514e", "grade": "#e4e3df"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": C["tinta2"], "axes.labelcolor": C["tinta"],
                     "xtick.color": C["tinta2"], "ytick.color": C["tinta2"], "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": C["grade"], "grid.linewidth": 0.6,
                     "svg.fonttype": "none", "figure.dpi": 150})
LEG: list[str] = []


def salvar(fig, nome, legenda: dict):
    fig.savefig(FIG / f"{nome}.png", bbox_inches="tight", dpi=200)
    fig.savefig(FIG / f"{nome}.svg", bbox_inches="tight")
    plt.close(fig)
    LEG.append(f"## {nome}\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in legenda.items()) + "\n")


def fig_mapa():
    import geopandas as gpd
    import rasterio
    from pa_03_referencias import ufrj_gdf
    with rasterio.open(TERRENO / "petropolis/atributos_encosta.tif") as r:
        decl = r.read(1)
        ext = [r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top]
    fcu = gpd.read_file(ESTUDO / "fcu_42_geometrias.geojson").to_crs(EPSG_PETROPOLIS)
    ch = gpd.read_file(CHARTER).to_crs(EPSG_PETROPOLIS)
    uf = ufrj_gdf()
    uf = uf[uf.localizado & uf.tipo_de_ev.eq("deslizamento")]
    hull = ch.union_all().convex_hull
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})
    for a in ax:
        a.imshow(decl, extent=ext, cmap="Greys", vmin=0, vmax=50, alpha=0.55, interpolation="nearest")
        a.grid(False)
        a.set_xticks([]); a.set_yticks([])
    a = ax[0]
    uf.plot(ax=a, markersize=3, color=C["violeta"], alpha=0.6, label="relato historico UFRJ/GEOCART (1933-2015)")
    fcu.boundary.plot(ax=a, color=C["tinta"], linewidth=0.7)
    gpd.GeoSeries([hull], crs=EPSG_PETROPOLIS).boundary.plot(ax=a, color=C["vermelho"], linewidth=1.2)
    gpd.GeoSeries([hull.buffer(300)], crs=EPSG_PETROPOLIS).boundary.plot(ax=a, color=C["vermelho"], linewidth=0.6, linestyle="--")
    minx, miny, maxx, maxy = fcu.total_bounds
    a.set_xlim(minx - 3000, maxx + 3000); a.set_ylim(miny - 3000, maxy + 3000)
    a.set_title("(a) municipio: comunidades (contorno), relatos historicos e suporte do teste de 2022", fontsize=8.5, loc="left")
    a = ax[1]
    xmin, ymin, xmax, ymax = hull.buffer(400).bounds
    ch[ch.tipoevento.eq("Deslizamento")].plot(ax=a, color=C["vermelho"], edgecolor="white", linewidth=0.3)
    ch[ch.tipoevento.eq("Fluxo de detritos")].plot(ax=a, color=C["laranja"], edgecolor="white", linewidth=0.3)
    ch[ch.tipoevento.str.startswith("Inunda")].plot(ax=a, color=C["azul"], edgecolor="white", linewidth=0.3)
    fcu.boundary.plot(ax=a, color=C["tinta"], linewidth=0.9)
    uf.plot(ax=a, markersize=4, color=C["violeta"], alpha=0.7)
    gpd.GeoSeries([hull], crs=EPSG_PETROPOLIS).boundary.plot(ax=a, color=C["vermelho"], linewidth=1.0)
    for cod, nome in [("33039060003", "Morro da Oficina"), ("33039060014", "V. S. Francisco A"),
                      ("33039060021", "Sgt. Boening"), ("33039060022", "Paraiso A"), ("33039060052", "Paraiso B")]:
        g = fcu[fcu.CD_FCU == cod]
        if len(g):
            p = g.geometry.iloc[0].representative_point()
            if xmin < p.x < xmax and ymin < p.y < ymax:
                a.annotate(nome, (p.x, p.y), fontsize=7, color=C["tinta"], xytext=(4, 4), textcoords="offset points")
    a.set_xlim(xmin, xmax); a.set_ylim(ymin, ymax)
    a.set_title("(b) suporte do teste: cicatrizes 2022 (vermelho), fluxos (laranja), inundacao (azul)", fontsize=8.5, loc="left")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hand = [Patch(color=C["vermelho"], label="deslizamento 2022 (INPE/Charter)"),
            Patch(color=C["laranja"], label="fluxo de detritos 2022"), Patch(color=C["azul"], label="inundacao 2022"),
            Line2D([], [], marker="o", ls="", color=C["violeta"], ms=3, label="relato historico de deslizamento"),
            Line2D([], [], color=C["tinta"], lw=0.8, label="FCU (Censo 2022)"),
            Line2D([], [], color=C["vermelho"], lw=1.2, label="suporte do teste"),
            Line2D([], [], color=C["vermelho"], lw=0.6, ls="--", label="exclusao +300 m (sem atualizacao)")]
    fig.legend(handles=hand, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.subplots_adjust(bottom=0.12)
    salvar(fig, "fig1_processos_comunidades", {
        "fonte": "INPE/Charter 0864; UFRJ/GEOCART via SIG municipal; IBGE FCU 2022; FABDEM V1-2 (fundo: declividade)",
        "populacao": "42 FCU de Petropolis; 81 deslizamentos, 3 fluxos, 1 inundacao (2022); 1.379 relatos localizados",
        "data": "episodio fev/2022; relatos 1933-2015; FCU 2022", "unidade": "EPSG:31983, fundo em graus de declividade",
        "procedimento": "sobreposicao cartografica; suporte = envoltoria convexa das 85 feicoes",
        "limitacao": "relatos tem posicao de endereco; exterior dos poligonos nao e ausencia; FCU posterior ao episodio"})


def fig_origem_destino():
    m = pd.read_csv(RES / "metricas_bracos.csv")
    rt = ler_json(RES / "resultado_treino.json")
    z = m[m.semente == -1].set_index("braco")
    bracos = ["A0_regra", "A1_log", "A1_hgb"]
    rot = {"A0_regra": "regra: declive", "A1_log": "Idai logistica", "A1_hgb": "Idai boosting"}
    cor = {"A0_regra": C["cinza"], "A1_log": C["azul"], "A1_hgb": C["laranja"]}
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    paineis = [("AUC_idai_reserva", "(a) origem: Idai, 13 blocos retidos\nfonte vs ausencia mapeada"),
               ("AUC_obj", "(b) destino: cicatrizes 2022\nAUC por objeto, presenca-fundo"),
               ("AUC_T_hist", "(c) destino: relatos historicos\npresenca-fundo de relato")]
    for a, (col, tit) in zip(ax, paineis):
        for i, b in enumerate(bracos):
            v = z.loc[b, col]
            a.plot([v], [i], "o", ms=8, color=cor[b], markeredgecolor="white", markeredgewidth=1.5)
            if col == "AUC_obj":
                lo, hi = rt["bootstrap_resumo"][b]["ic95"]
                a.plot([lo, hi], [i, i], color=cor[b], lw=2)
            a.annotate(f"{v:.3f}", (v, i), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.5, color=C["tinta"])
        a.axvline(0.5, color=C["tinta2"], lw=0.8, ls=":")
        a.set_xlim(0.3, 0.95)
        a.set_title(tit, fontsize=8.5, loc="left")
        a.set_xlabel("AUC (0,5 = acaso)")
    ax[0].set_yticks(range(3)); ax[0].set_yticklabels([rot[b] for b in bracos]); ax[0].set_ylim(-0.6, 2.6)
    salvar(fig, "fig2_origem_destino", {
        "fonte": "resultados/metricas_bracos.csv; bootstrap em resultado_treino.json",
        "populacao": "Idai reserva: 2.111 celulas-fonte e 11.671 ausencias mapeadas; T1: 81 cicatrizes, 12.170 celulas de fundo; T_hist: 622 sitios",
        "data": "Idai mar/2019; Petropolis fev/2022; relatos 1933-2015", "unidade": "AUC; em (b) por objeto (cicatriz)",
        "procedimento": "modelos ajustados so no Idai, aplicados sem rotulo local; IC95 em (b) por bootstrap de cicatrizes e blocos de 500 m",
        "limitacao": "(b) e (c) sao presenca-fundo: nao ha especificidade; (c) abaixo de 0,5 reflete a posicao dos relatos (receptor), nao falha de terreno"})


def fig_orcamento():
    m = pd.read_csv(RES / "metricas_bracos.csv")
    z = m[m.semente == -1].set_index("braco")
    ret = pd.read_csv(RES / "retencao_idai.csv")
    e1 = pd.read_csv(RES / "emenda1/metricas_E1.csv")
    e1i = e1.set_index("vetor")
    re1 = pd.read_csv(RES / "emenda1/retencao_E1.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    series = [("A3_log_l25", "externo + local (logistica, lambda 25)", C["azul"]),
              ("A2_log_l25", "so local (logistica, lambda 25)", C["laranja"]),
              ("A3perm_log_l25", "controle: prior permutado", C["magenta"]),
              ("A3_hgb", "externo + local (boosting)", C["aqua"]), ("A2_hgb", "so local (boosting)", C["amarelo"])]
    Bs, Bl = ["10", "30", "100", "300", "todos"], ["10", "30", "100", "300", "402\n(todos)"]
    Ks, Kl = ["5", "15", "todas"], ["5", "15", "todas\n(59-71)"]

    def ref(a, valores, xmax):
        for nome, rot, cor, dy in [("A1_log", "Idai sem rotulo local", C["violeta"], 4), ("A0_regra", "regra de declive", C["cinza"], -10)]:
            a.axhline(valores[nome], color=cor, lw=1.2, ls="--")
            a.annotate(rot, (xmax, valores[nome]), xytext=(0, dy), textcoords="offset points", ha="right", fontsize=7, color=C["tinta2"])
        a.axhline(0.5, color=C["tinta2"], lw=0.7, ls=":")

    a = ax[0]
    for nome, rot, cor in series:
        g = m[m.braco == nome].groupby(m.B.astype(str)).AUC_obj.agg(["mean", "min", "max"]).loc[Bs]
        x = np.arange(len(Bs))
        a.fill_between(x, g["min"], g["max"], color=cor, alpha=0.15, lw=0)
        a.plot(x, g["mean"], "-o", color=cor, lw=2, ms=4, label=rot)
    ref(a, z.AUC_obj, len(Bs) - 1)
    a.set_xticks(range(len(Bs))); a.set_xticklabels(Bl)
    a.set_xlabel("sitios historicos usados na atualizacao (B)"); a.set_ylabel("AUC por objeto, cicatrizes 2022")
    a.set_title("(a) atualizacao com relatos historicos\nprotocolo v1, teste 2022 retido", fontsize=8.5, loc="left")
    a = ax[1]
    for nome, rot, cor in series:
        x = np.arange(len(Ks))
        v = [e1[e1.vetor.str.startswith(f"{nome}__K{k}__")].AUC_obj_oof for k in Ks]
        a.fill_between(x, [vv.min() for vv in v], [vv.max() for vv in v], color=cor, alpha=0.15, lw=0)
        a.plot(x, [vv.mean() for vv in v], "-o", color=cor, lw=2, ms=4)
    ref(a, e1i.AUC_obj_oof, len(Ks) - 1)
    a.set_xticks(range(len(Ks))); a.set_xticklabels(Kl)
    a.set_xlabel("cicatrizes de outros bairros usadas na atualizacao")
    a.set_title("(b) atualizacao com cicatrizes 2022\nemenda 1, avaliada no bairro retido", fontsize=8.5, loc="left")
    a = ax[2]
    for nome, cor, rot in [("A3_log_l25", C["azul"], "logistica: relatos (B)"), ("A3_hgb", C["aqua"], "boosting: relatos (B)")]:
        g = ret[ret.braco == nome].set_index(ret[ret.braco == nome].B.astype(str)).loc[Bs]
        a.plot(np.arange(len(Bs)), g.AUC_reserva_media, "-o", color=cor, lw=2, ms=4, label=rot)
    for nome, cor, rot in [("A3_log_l25", C["violeta"], "logistica: cicatrizes (K)"), ("A3_hgb", C["verde"], "boosting: cicatrizes (K)")]:
        g = re1[re1.braco == nome].set_index(re1[re1.braco == nome].K.astype(str)).loc[Ks]
        a.plot([0, 1, 2], g.AUC_reserva_media, "-s", color=cor, lw=2, ms=4, label=rot)
    a.axhline(ret.AUC_referencia.iloc[0], color=C["tinta2"], lw=1, ls="--")
    a.annotate("Idai antes da atualizacao", (4, ret.AUC_referencia.iloc[0]), xytext=(0, 4), textcoords="offset points", ha="right", fontsize=7, color=C["tinta2"])
    a.set_xticks(range(5)); a.set_xticklabels(["B10 / K5", "B30 / K15", "B100 / K todas", "B300", "B402"], fontsize=7)
    a.set_ylabel("AUC no Idai retido (13 blocos)")
    a.set_title("(c) retencao do conhecimento de origem", fontsize=8.5, loc="left")
    a.legend(fontsize=6.5, frameon=False, loc="lower left")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig.subplots_adjust(bottom=0.24, wspace=0.28)
    salvar(fig, "fig3_antes_depois_orcamento_retencao", {
        "fonte": "resultados/metricas_bracos.csv, retencao_idai.csv, emenda1/metricas_E1.csv, emenda1/retencao_E1.csv",
        "populacao": "81 cicatrizes de 2022; pool de 402 sitios historicos; 5 dobras por bairro; Idai 13 blocos retidos",
        "data": "relatos 1933-2015; episodio fev/2022; Idai 2019", "unidade": "AUC por objeto (a,b), AUC celular (c)",
        "procedimento": "linha = media de 10 sementes; faixa = minimo-maximo das sementes (estabilidade computacional, nao episodios); eixo x categorico",
        "limitacao": "(a) e protocolo v1 congelado; (b) e emenda pos-v1, validacao cruzada dentro do episodio (desenvolvimento); em (c) B e K nao sao a mesma unidade"})


def fig_blocos_coef():
    ab = pd.read_csv(RES / "ablacoes.csv")
    m = pd.read_csv(RES / "metricas_bracos.csv")
    z = m[m.semente == -1].set_index("braco")
    d5 = pd.read_csv(RES / "emenda1/D5_receptor_T_hist.csv")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    a = ax[0]
    cols = [("AUC_idai_reserva", "Idai retido", C["azul"]), ("AUC_obj", "cicatrizes 2022", C["laranja"]),
            ("AUC_T_hist", "relatos historicos", C["aqua"])]
    y = np.arange(len(ab))
    for j, (col, rot, cor) in enumerate(cols):
        d = ab[col] - z.loc["A1_log", col]
        a.barh(y + (j - 1) * 0.26, d, height=0.24, color=cor, label=rot)
    a.axvline(0, color=C["tinta2"], lw=0.8)
    a.set_yticks(y); a.set_yticklabels([f"sem {b}" for b in ab.bloco_removido])
    a.set_xlabel("Delta AUC ao remover o bloco (A1 logistica)")
    a.set_title("(a) contribuicao dos blocos fisicos", fontsize=8.5, loc="left")
    a.legend(fontsize=6.5, frameon=False)
    a = ax[1]
    F = ler_json(OUT / "protocolo/protocolo_v1.json")["atributos"]
    a1 = ler_json(OUT / "pesos/A1_log.json")
    aj = pd.read_csv(RES / "ajustes_locais.csv")
    hist = aj[(aj.braco == "A2_log_l25") & (aj.B.astype(str) == "todos")][[f"coef_{c}" for c in F]].mean().values
    e1 = ler_json(OUT / "pesos/emenda1/E1_implantacao_A3_log_l25.json")
    aje1 = pd.read_csv(RES / "emenda1/ajustes_E1.csv")
    e1loc = aje1[(aje1.braco == "A2_log_l25") & (aje1.K.astype(str) == "todas")][[f"coef_{c}" for c in F]].mean().values
    yy = np.arange(len(F))
    for j, (v, rot, cor) in enumerate([(a1["coef"], "Idai (origem)", C["azul"]),
                                       (hist, "so relatos historicos", C["laranja"]),
                                       (e1loc, "so cicatrizes 2022 (media das dobras)", C["aqua"]),
                                       (e1["coef"], "Idai + cicatrizes 2022", C["violeta"])]):
        a.barh(yy + (j - 1.5) * 0.2, v, height=0.18, color=cor, label=rot)
    a.axvline(0, color=C["tinta2"], lw=0.8)
    a.set_yticks(yy); a.set_yticklabels(F)
    a.set_xlabel("coeficiente padronizado (escala Idai)")
    a.set_title("(b) o que cada fonte ensinou ao estimador", fontsize=8.5, loc="left")
    a.legend(fontsize=6.2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    a = ax[2]
    est = ["todos", "preciso", "aproximado"]
    for j, (sc, rot, cor) in enumerate([("A1_cel", "escore na celula do relato", C["laranja"]),
                                         ("A1_montante7", "maximo a montante (7 passos D8)", C["azul"])]):
        v = [d5[(d5.estrato == e) & (d5.escore == sc)].AUC_T_hist.iloc[0] for e in est]
        a.bar(np.arange(3) + (j - 0.5) * 0.36, v, width=0.34, color=cor, label=rot)
        for i, vv in enumerate(v):
            a.annotate(f"{vv:.2f}", (i + (j - 0.5) * 0.36, vv), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
    n = [d5[(d5.estrato == e)].sitios.iloc[0] for e in est]
    a.set_xticks(range(3)); a.set_xticklabels([f"{e}\n(n={k})" for e, k in zip(est, n)])
    a.axhline(0.5, color=C["tinta2"], lw=0.8, ls=":")
    a.set_ylim(0, 0.9); a.set_ylabel("AUC relatos x dominio de relato")
    a.set_title("(c) relatos como receptores (D5)", fontsize=8.5, loc="left")
    a.legend(fontsize=6.5, frameon=False, loc="upper left")
    salvar(fig, "fig4_blocos_coeficientes_receptor", {
        "fonte": "resultados/ablacoes.csv; pesos/*.json; ajustes_locais.csv; emenda1/ajustes_E1.csv; emenda1/D5_receptor_T_hist.csv",
        "populacao": "como fig2; (c) 622 sitios historicos, estratificados pela precisao do geocodigo",
        "data": "como fig2", "unidade": "Delta AUC; coeficiente por desvio-padrao do ajuste Idai; AUC",
        "procedimento": "(a) A1 reajustado sem cada bloco; (b) coeficientes salvos; (c) escore A1 propagado a montante por D8",
        "limitacao": "coeficientes descrevem o estimador sob este desenho, nao pesos causais; (c) apoia a leitura de receptor, nao prova a cadeia de cada relato"})


def fig_comunidades():
    t = pd.read_csv(TER / "tabela_42_fcu.csv", dtype={"CD_FCU": str})
    t = t.sort_values("encosta_E1_pct_mediano").reset_index(drop=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 9), sharey=True, gridspec_kw={"width_ratios": [1.6, 0.8, 0.8]})
    y = np.arange(len(t))
    a = ax[0]
    for i, r in t.iterrows():
        a.plot([r.encosta_antes_pct_mediano, r.encosta_E1_pct_mediano], [i, i], color=C["grade"], lw=2, zorder=1)
    a.scatter(t.encosta_antes_pct_mediano, y, s=22, color=C["cinza"], label="antes: Idai sem rotulo local", zorder=2)
    a.scatter(t.encosta_E1_pct_mediano, y, s=26, color=C["azul"], label="depois: Idai + cicatrizes 2022 (E1)", zorder=3)
    a.scatter(t.encosta_depois_pct_mediano, y, s=14, marker="x", color=C["laranja"], label="relatos historicos (recusado)", zorder=3)
    obs = t.charter2022_deslizamentos > 0
    hist = t.ufrj_relatos_100m > 0
    a.scatter(np.full(obs.sum(), 1.03), y[obs], marker="s", s=22, color=C["vermelho"], label="cicatriz 2022 observada", clip_on=False)
    a.scatter(np.full(hist.sum(), 1.07), y[hist], marker="^", s=22, color=C["violeta"], label="relato historico ate 100 m", clip_on=False)
    a.set_yticks(y); a.set_yticklabels([f"{n[:34]} ({p})" for n, p in zip(t.NM_FCU, t.populacao_censo2022)], fontsize=6.5)
    a.set_xlim(0, 1.1); a.set_xlabel("percentil mediano de suscetibilidade de encosta no municipio")
    a.set_title("(a) encosta: inferencia antes/depois e registros observados", fontsize=8.5, loc="left")
    a.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    a = ax[1]
    a.barh(y, t.agua_escore_medio.fillna(0), color=C["aqua"], height=0.6)
    a.set_xlabel("escore de agua medio (svc-03)")
    a.set_title("(b) agua: transferencia\nsem referencia local", fontsize=8.5, loc="left")
    a = ax[2]
    a.barh(y, t.montante_quintil_sup_km2, color=C["amarelo"], height=0.6)
    a.set_xlabel("km2 que drenam para a FCU (ate 1 km)\nno quintil superior de encosta (E1)")
    a.set_title("(c) conexao: percurso\na investigar", fontsize=8.5, loc="left")
    salvar(fig, "fig5_comunidades_42", {
        "fonte": "territorio/tabela_42_fcu.csv (gerada de pesos salvos, svc-03 e fontes do ledger)",
        "populacao": "42 FCU; entre parenteses a populacao do Censo 2022 (universo da comunidade, nao atingidos)",
        "data": "terreno FABDEM; cicatrizes 2022; relatos 1933-2015; Censo 2022",
        "unidade": "percentil no dominio municipal; escore de agua adimensional; km2",
        "procedimento": "percentil mediano das celulas de 30 m da FCU; area que drena para a FCU por D8 (WhiteboxTools) a ate 1 km dela",
        "limitacao": "ordenacao relativa, nao probabilidade; agua e conexao sem validacao local; ausencia de marcador nao e ausencia do fenomeno"})


def fig_cobertura():
    t = pd.read_csv(TER / "tabela_42_fcu.csv", dtype={"CD_FCU": str})
    info = ler_json(TER / "informacao_antes_depois.json")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    a = ax[0]
    itens = [("encosta observada", info["antes_ficha_estudo_v1"]["encosta_observada"], info["depois_v1"]["encosta_observada"]),
             ("contexto tecnico 2025", info["antes_ficha_estudo_v1"]["contexto_tecnico"], info["depois_v1"]["contexto_tecnico"]),
             ("encosta inferida (dominio >=80%)", 0, info["depois_v1"]["encosta_inferida_dentro_dominio_80pct"]),
             ("conexao a montante", 0, info["depois_v1"]["conexao_com_montante_quintil_sup"]),
             ("escore de agua explicito", 0, info["depois_v1"]["agua_escore_explicito"]),
             ("sem nenhum registro observado", 42 - info["antes_ficha_estudo_v1"]["encosta_observada"], info["depois_v1"]["sem_nenhum_registro_observado"])]
    y = np.arange(len(itens))
    a.barh(y + 0.18, [i[1] for i in itens], height=0.34, color=C["cinza"], label="ficha anterior (estudo v1)")
    a.barh(y - 0.18, [i[2] for i in itens], height=0.34, color=C["azul"], label="esta campanha")
    for yy, (_, b, d) in zip(y, itens):
        a.annotate(str(b), (b, yy + 0.18), xytext=(2, 0), textcoords="offset points", va="center", fontsize=7)
        a.annotate(str(d), (d, yy - 0.18), xytext=(2, 0), textcoords="offset points", va="center", fontsize=7)
    a.set_yticks(y); a.set_yticklabels([i[0] for i in itens]); a.set_xlim(0, 46)
    a.set_xlabel("numero de FCU (de 42)")
    a.set_title("(a) informacao disponivel por comunidade", fontsize=8.5, loc="left")
    a.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    a = ax[1]
    v = np.sort(t.dominio_frac_dentro_doador.values)
    a.plot(np.arange(1, len(v) + 1), v, "-o", color=C["violeta"], ms=3, lw=1.5)
    a.axhline(0.8, color=C["tinta2"], lw=0.8, ls="--")
    a.set_ylim(0, 1.02); a.set_xlabel("FCU ordenadas"); a.set_ylabel("fracao das celulas dentro do P1-P99 do Idai")
    a.set_title("(b) dominio do doador: onde a inferencia extrapola", fontsize=8.5, loc="left")
    salvar(fig, "fig6_cobertura_desconhecido", {
        "fonte": "territorio/informacao_antes_depois.json; tabela_42_fcu.csv; fichas do estudo multiperigos v1",
        "populacao": "42 FCU", "data": "2026-10-24", "unidade": "contagem de FCU; fracao de celulas",
        "procedimento": "contagem de componentes presentes na ficha anterior e na candidata atual; dominio = todos os 7 atributos entre P1 e P99 do ajuste Idai",
        "limitacao": "mais informacao nao e menos risco; 'sem registro observado' continua desconhecido, nao seguro"})


def tabelas():
    TAB.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(RES / "metricas_bracos.csv")
    g = (m.groupby(["braco", m.B.astype(str)]).agg(AUC_obj=("AUC_obj", "mean"), AUC_obj_min=("AUC_obj", "min"),
                                                   AUC_obj_max=("AUC_obj", "max"), AP_cel=("AP_cel", "mean"),
                                                   rec_cic_20=("rec_cic_20", "mean"), AUC_idai_reserva=("AUC_idai_reserva", "mean"),
                                                   AUC_T_hist=("AUC_T_hist", "mean"), sementes=("AUC_obj", "size")).reset_index())
    g.to_csv(TAB / "tabela_T1_bracos_orcamentos.csv", index=False)
    c1 = pd.read_csv(RES / "contrastes_T1.csv")
    c2 = pd.read_csv(RES / "emenda1/contrastes_E1.csv")
    pr = pd.concat([c1[c1.primario].assign(origem="protocolo v1 (relatos historicos)"),
                    c2[c2.primario | (c2.contraste == "E1_transferencia_dentro_dobra") | (c2.contraste == "E1_A3_vs_regra")].assign(origem="emenda 1 (cicatrizes 2022, CV bairro)")])
    pr.to_csv(TAB / "tabela_contrastes_primarios.csv", index=False)
    t = pd.read_csv(TER / "tabela_42_fcu.csv", dtype={"CD_FCU": str})
    t.to_csv(TAB / "tabela_42_fcu.csv", index=False, encoding="utf-8-sig")
    pz = pd.read_csv(RES / "emenda1/por_comunidade_zona_teste.csv", dtype={"CD_FCU": str})
    pz.to_csv(TAB / "tabela_comunidades_zona_teste.csv", index=False, encoding="utf-8-sig")
    return {"tabelas": sorted(p.name for p in TAB.glob("*.csv"))}


def main() -> int:
    e = Etapa("pa_07_figuras")
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_mapa, fig_origem_destino, fig_orcamento, fig_blocos_coef, fig_comunidades, fig_cobertura):
        print("[fig]", f.__name__, flush=True)
        f()
    (FIG / "legendas.md").write_text("# Legendas das figuras\n\n" + "\n".join(LEG), encoding="utf-8")
    t = tabelas()
    gravar_json(TAB / "indice.json", t)
    e.saida(*FIG.glob("*"), *TAB.glob("*"))
    e.fechar(figuras=len(LEG))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
