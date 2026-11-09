"""Figuras e tabelas do artigo (Recife), geradas so dos artefatos da linhagem. PNG + SVG + legendas."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ra_comum import DREN, EPSG, ESTUDO, MACRO, OUT, Etapa, gravar_json, ler_json  # noqa: E402

FIG, TAB, RES, TER = OUT / "figuras", OUT / "tabelas", OUT / "resultados", OUT / "territorio"
C = {"azul": "#2a78d6", "laranja": "#eb6834", "aqua": "#1baf7a", "amarelo": "#eda100", "magenta": "#e87ba4",
     "verde": "#008300", "violeta": "#6250d6", "vermelho": "#e34948", "cinza": "#8a8986", "tinta": "#0b0b0b",
     "tinta2": "#52514e", "grade": "#e4e3df"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": C["tinta2"], "axes.labelcolor": C["tinta"], "xtick.color": C["tinta2"],
                     "ytick.color": C["tinta2"], "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": C["grade"], "grid.linewidth": 0.6, "svg.fonttype": "none", "figure.dpi": 150})
LEG: list[str] = []
NOMES = {"A0_regra": "regra: HAND 30 m", "A1_Q": "Q transferido (sem rotulo)", "A2_q_l25": "so local, atributos Q",
         "A3_q_l25": "Q + atualizacao local", "A3_q_l1": "Q + atualizacao local (lambda 1)", "A3perm_l25": "controle: prior permutado", "A2u_l25": "so local + drenagem/saida",
         "A4_l25": "Q atualizado + drenagem/saida", "A4_hgb": "boosting: Q + drenagem/saida"}


def salvar(fig, nome, leg):
    fig.savefig(FIG / f"{nome}.png", bbox_inches="tight", dpi=200)
    fig.savefig(FIG / f"{nome}.svg", bbox_inches="tight")
    plt.close(fig)
    LEG.append(f"## {nome}\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in leg.items()) + "\n")


def fig1_caso_processo():
    import geopandas as gpd
    import rasterio
    obj = gpd.read_file(OUT / "referencias/objetos_rua_bairro.gpkg")
    t = pd.read_parquet(OUT / "atributos/atributos_objetos.parquet")
    p = pd.read_parquet(RES / "pred_ruas.parquet")
    aj = pd.read_csv(RES / "ajustes.csv")
    fcu = gpd.read_file(ESTUDO / "fcu_recife_epsg31985.geojson").to_crs(EPSG)
    alvo = fcu[fcu.CD_FCU == "26116060588"]                       # piloto fixado no estudo (Chico Mendes - Linha Nova)
    caixa = alvo.buffer(350).total_bounds
    from shapely.geometry import box
    bx = box(*caixa)
    el = gpd.read_file(DREN / "layer_00_elementos_da_rede_epsg31985.gpkg", bbox=tuple(caixa), columns=["TIPO"])
    rede = gpd.read_file(DREN / "layer_01_rede_epsg31985.gpkg", bbox=tuple(caixa), columns=["OBJECTID"])
    can = pd.concat([gpd.read_file(MACRO / "layer_02_canais_epsg31985.gpkg", bbox=tuple(caixa), columns=["NOME"]),
                     gpd.read_file(MACRO / "layer_03_rios_e_riachos_epsg31985.gpkg", bbox=tuple(caixa), columns=["NOME"])])
    with rasterio.open(OUT / "atributos/pe3d_mdt_2m_original.tif") as r:
        from rasterio.windows import from_bounds
        w = from_bounds(*caixa, transform=r.transform)
        z = r.read(1, window=w).astype(float)
    z[z <= -9000] = np.nan
    d = obj.merge(t[["id_obj", "y"]], on=["id_obj", "y"]).merge(p[["id_obj", "A4_l25__Btodas__s0"]], on="id_obj")
    d = d[d.intersects(bx)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    a.imshow(z, extent=[caixa[0], caixa[2], caixa[1], caixa[3]], cmap="Greys", alpha=0.6)
    a.grid(False); a.set_xticks([]); a.set_yticks([])
    rede.plot(ax=a, color=C["azul"], linewidth=0.5, alpha=0.7)
    can.plot(ax=a, color=C["aqua"], linewidth=2.0)
    ent = el[el.TIPO.astype(str).str.strip().isin(["BL", "CG"])]
    ent.plot(ax=a, color=C["amarelo"], markersize=3)
    d.plot(ax=a, column="A4_l25__Btodas__s0", cmap="Oranges", linewidth=1.6)
    d[d.y == 1].plot(ax=a, color=C["vermelho"], linewidth=3.2)
    alvo.boundary.plot(ax=a, color=C["tinta"], linewidth=1.2)
    a.set_xlim(caixa[0], caixa[2]); a.set_ylim(caixa[1], caixa[3])
    a.set_title("(a) Chico Mendes - Linha Nova: terreno PE3D, entradas, rede, canal e ruas", fontsize=8.5, loc="left")
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], color=C["vermelho"], lw=3, label="rua com relato SEDEC"),
                      Line2D([], [], color=C["laranja"], lw=1.6, label="demais ruas (escore apos atualizacao)"),
                      Line2D([], [], color=C["azul"], lw=0.8, label="rede cadastrada"),
                      Line2D([], [], color=C["aqua"], lw=2, label="canal / rio"),
                      Line2D([], [], marker="o", ls="", color=C["amarelo"], ms=3, label="entrada BL/CG")],
             fontsize=6.5, loc="lower left", frameon=True)
    a = ax[1]
    pos = d[d.y == 1].id_obj
    if len(pos):
        oid = pos.iloc[0]
        row = t[t.id_obj == oid].iloc[0]
        pad = ler_json(OUT / "pesos/padronizacao.json")
        prot = ler_json(OUT / "protocolo/protocolo_v1.json")
        QF = prot["blocos"]["Q30"]
        U = prot["blocos"]["DREN"] + prot["blocos"]["SAIDA"]
        zq = [(np.nan_to_num(row[c], nan=pad["Q30"]["med"][c]) - pad["Q30"]["mu"][c]) / pad["Q30"]["sd"][c] for c in QF]
        zu = [(np.nan_to_num(row[c], nan=pad["urbanos"]["med"][c]) - pad["urbanos"]["mu"][c]) / pad["urbanos"]["sd"][c] for c in U]
        bq = ler_json(OUT / "atributos/nucleo_q_congelado.json")["b"]
        a4 = aj[(aj.braco == "A4_l25") & (aj.B.astype(str) == "todas")].iloc[0]
        nomes = QF + U
        antes = np.r_[np.asarray(bq) * zq, np.zeros(len(U))]
        depois = np.asarray([a4[f"coef_{c}"] for c in nomes]) * np.r_[zq, zu]
        yy = np.arange(len(nomes))
        a.barh(yy + 0.2, antes, height=0.38, color=C["cinza"], label="antes: Q transferido")
        a.barh(yy - 0.2, depois, height=0.38, color=C["azul"], label="depois: Q atualizado + drenagem/saida")
        a.axvline(0, color=C["tinta2"], lw=0.8)
        a.set_yticks(yy); a.set_yticklabels(nomes, fontsize=7)
        a.set_xlabel("contribuicao ao escore (coeficiente x atributo padronizado)")
        a.set_title(f"(b) por que o escore da rua relatada mudou ({row.NLOGRACONC.title()[:28]})", fontsize=8.5, loc="left")
        a.legend(fontsize=6.5, frameon=False, loc="lower right")
    salvar(fig, "fig1_processo_caso_real", {
        "fonte": "PE3D 1 m (agregado a 2 m), EMLURB micro/macrodrenagem, logradouros ESIG, SEDEC, pesos salvos",
        "populacao": "piloto Chico Mendes - Linha Nova (escolhido no estudo por cobertura, sem consultar alagamento)",
        "data": "PE3D 2016 (cabecalho do ZIP); cadastro atual; relatos 2014-2024", "unidade": "m (cota), contribuicao adimensional",
        "procedimento": "contribuicoes = coeficiente x atributo padronizado; antes = Q congelado, depois = A4 (B=todas)",
        "limitacao": "o piloto esta em bairro de atualizacao: mostra o estimador, nao resultado fora do ajuste; cadastro nao e funcionamento"})


def fig2_mapa_real():
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box
    obj = gpd.read_file(OUT / "referencias/objetos_rua_bairro.gpkg")
    fcu = gpd.read_file(ESTUDO / "fcu_recife_epsg31985.geojson").to_crs(EPSG)
    from ra_comum import PE3D_ZIPS
    folhas = []
    for z in sorted(PE3D_ZIPS.glob("*.zip")):
        with rasterio.open(f"/vsizip/{z.as_posix()}/{z.stem}.tif") as r:
            folhas.append(box(*r.bounds))
    can = gpd.read_file(MACRO / "layer_02_canais_epsg31985.gpkg", columns=["NOME"])
    rio = gpd.read_file(MACRO / "layer_03_rios_e_riachos_epsg31985.gpkg", columns=["NOME"])
    fig, a = plt.subplots(figsize=(8, 9))
    a.grid(False); a.set_xticks([]); a.set_yticks([])
    obj[obj.papel == "atualizacao"].plot(ax=a, color=C["cinza"], linewidth=0.3, alpha=0.6)
    obj[obj.papel == "teste"].plot(ax=a, color=C["violeta"], linewidth=0.3, alpha=0.6)
    fcu.boundary.plot(ax=a, color=C["tinta"], linewidth=0.35)
    gpd.GeoSeries(folhas, crs=EPSG).boundary.plot(ax=a, color=C["laranja"], linewidth=1.1)
    can.plot(ax=a, color=C["aqua"], linewidth=0.8); rio.plot(ax=a, color=C["aqua"], linewidth=1.2)
    obj[obj.y == 1].plot(ax=a, color=C["vermelho"], linewidth=2.2)
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], color=C["vermelho"], lw=2.2, label="rua com relato SEDEC (96)"),
                      Line2D([], [], color=C["cinza"], lw=1, label="ruas de bairros de atualizacao"),
                      Line2D([], [], color=C["violeta"], lw=1, label="ruas de bairros de teste"),
                      Line2D([], [], color=C["laranja"], lw=1.1, label="8 folhas PE3D 1 m"),
                      Line2D([], [], color=C["aqua"], lw=1, label="canais e rios (EMLURB)"),
                      Line2D([], [], color=C["tinta"], lw=0.5, label="FCU (295)")], fontsize=7, loc="lower right")
    ref = ler_json(OUT / "referencias/referencias.json")
    a.set_title("Populacao do experimento, particao por bairro, PE3D e referencia. Relatos sem rua: "
                f"{ref['relatos_por_estado'].get('sem_endereco', 0)} sem endereco, "
                f"{ref['relatos_por_estado'].get('nao_geocodificado', 0)} nao geocodificados (desconhecido)", fontsize=8, loc="left")
    salvar(fig, "fig2_mapa_referencia_particoes", {
        "fonte": "objetos rua-bairro (ESIG), relatos SEDEC geocodificados, folhas PE3D, EMLURB macrodrenagem, FCU 2022",
        "populacao": "5.477 ruas de 38 bairros (19 de atualizacao, 19 de teste); 96 ruas com relato",
        "data": "relatos 2014-2024", "unidade": "EPSG:31985", "procedimento": "geocodificacao por nome de rua no bairro do relato",
        "limitacao": "1.228 relatos sem rua localizavel ficam fora (inclusive todo o episodio de 2022); ruas sem relato sao fundo, nao secas"})


def fig3_orcamento_retencao():
    m = pd.read_csv(RES / "metricas_bracos.csv")
    m = m[m.subconjunto == "cidade"]
    z = m[m.semente == -1].set_index("braco")
    ret = pd.read_csv(RES / "retencao_origem.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    a = ax[0]
    Bs, Bl = ["10", "25", "todas"], ["10", "25", "48 (todas)"]
    for nome, cor in [("A3_q_l25", C["azul"]), ("A2_q_l25", C["laranja"]), ("A3perm_l25", C["magenta"]),
                      ("A4_l25", C["violeta"]), ("A2u_l25", C["amarelo"]), ("A4_hgb", C["aqua"])]:
        g = m[m.braco == nome].groupby(m.B.astype(str)).AUC_bairro.agg(["mean", "min", "max"]).loc[Bs]
        x = np.arange(3)
        a.fill_between(x, g["min"], g["max"], color=cor, alpha=0.15, lw=0)
        a.plot(x, g["mean"], "-o", color=cor, lw=2, ms=4, label=NOMES[nome])
    for nome, cor, dy in [("A1_Q", C["tinta"], 4), ("A0_regra", C["cinza"], -10)]:
        a.axhline(z.loc[nome, "AUC_bairro"], color=cor, lw=1.2, ls="--")
        a.annotate(NOMES[nome], (2, z.loc[nome, "AUC_bairro"]), xytext=(0, dy), textcoords="offset points", ha="right", fontsize=7, color=C["tinta2"])
    a.axhline(0.5, color=C["tinta2"], lw=0.7, ls=":")
    a.set_xticks(range(3)); a.set_xticklabels(Bl)
    a.set_xlabel("ruas com relato usadas na atualizacao (bairros de atualizacao)")
    a.set_ylabel("AUC dentro do bairro, 19 bairros de teste")
    a.set_title("(a) estado externo, atualizacao e teste retido", fontsize=8.5, loc="left")
    a.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)
    a = ax[1]
    base = ret[ret.modelo == "Q_congelado_efeitos_principais"].AUC_retencao.iloc[0]
    for nome, cor in [("A3_q_l25", C["azul"]), ("A3_q_l1", C["laranja"]), ("A3perm_l25", C["magenta"]), ("A4_l25", C["violeta"])]:
        g = ret[ret.modelo == nome].groupby(ret.B.astype(str)).AUC_retencao.mean().reindex(Bs)
        a.plot(range(3), g.values, "-o", color=cor, lw=2, ms=4, label=NOMES.get(nome, nome) + (" (parcela Q)" if nome == "A4_l25" else ""))
    a.axhline(base, color=C["tinta2"], lw=1, ls="--")
    a.annotate("Q antes da atualizacao", (2, base), xytext=(0, 4), textcoords="offset points", ha="right", fontsize=7, color=C["tinta2"])
    a.set_xticks(range(3)); a.set_xticklabels(Bl); a.set_ylabel("AUC nas 6 unidades de retencao (origem)")
    a.set_title("(b) retencao na origem limpa", fontsize=8.5, loc="left")
    a.legend(fontsize=6.5, frameon=False, loc="lower right")
    salvar(fig, "fig3_orcamento_teste_retencao", {
        "fonte": "resultados/metricas_bracos.csv, retencao_origem.csv", "populacao": "48 ruas com relato e 2.177 de fundo em 19 bairros de teste; 6 unidades de retencao v10",
        "data": "relatos 2014-2024", "unidade": "AUC dentro do bairro (a); AUC dentro do evento (b)",
        "procedimento": "media de 10 sementes; faixa = min-max das sementes (estabilidade computacional, nao bairros)",
        "limitacao": "presenca-fundo; poucas positivas por orcamento; retencao de A4 so pela parcela Q"})


def fig4_blocos():
    c = pd.read_csv(RES / "contrastes.csv")
    m = pd.read_csv(RES / "metricas_bracos.csv")
    linhas = [("A1_Q", 0, "cidade", "Q transferido"), ("A3_q_l25", "todas", "cidade", "Q + atualizacao"),
              ("A4_sem_DREN", "todas", "cidade", "+ saida"), ("A4_sem_SAIDA", "todas", "cidade", "+ drenagem"),
              ("A4_l25", "todas", "cidade", "+ drenagem + saida"),
              ("A1_Q", 0, "pe3d", "Q transferido [PE3D]"), ("A0_pe3d", 0, "pe3d", "regra PE3D: rua abaixo do entorno [PE3D]"),
              ("A4p_l25", "todas_pe3d", "pe3d", "Q atualizado + drenagem + saida [PE3D]"),
              ("A5_so_PE3D_l25", "todas_pe3d", "pe3d", "Q atualizado + PE3D [PE3D]"),
              ("A5_l25", "todas_pe3d", "pe3d", "+ drenagem + saida + PE3D [PE3D]")]
    vals = []
    for b, B, sub, rot in linhas:
        x = m[(m.braco == b) & (m.B.astype(str) == str(B)) & (m.subconjunto == sub)]
        vals.append((rot, x.AUC_bairro.mean(), sub))
    fig, a = plt.subplots(figsize=(8.5, 4.4))
    y = np.arange(len(vals))[::-1]
    for yy, (rot, v, sub) in zip(y, vals):
        a.barh(yy, v - 0.5, left=0.5, color=C["azul"] if sub == "cidade" else C["violeta"], height=0.6)
        a.annotate(f"{v:.3f}", (v, yy), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)
    a.set_yticks(y); a.set_yticklabels([v[0] for v in vals], fontsize=7.5)
    a.axvline(0.5, color=C["tinta2"], lw=0.8, ls=":")
    a.set_xlabel("AUC dentro do bairro no teste (0,5 = acaso)")
    c6 = c[c.contraste == "C6_PE3D"].iloc[0]
    c5 = c[(c.contraste == "C5_contexto_urbano") & (c.B.astype(str) == "todas")].iloc[0]
    a.set_title(f"Efeito dos blocos no mesmo suporte. C5 drenagem+saida {c5.delta:+.3f} [{c5.ic95_lo:+.3f}; {c5.ic95_hi:+.3f}]; "
                f"C6 PE3D {c6.delta:+.3f} [{c6.ic95_lo:+.3f}; {c6.ic95_hi:+.3f}]", fontsize=8, loc="left")
    salvar(fig, "fig4_efeito_blocos", {
        "fonte": "resultados/metricas_bracos.csv, contrastes.csv", "populacao": "teste: 48 positivas (cidade); subconjunto PE3D do teste em roxo",
        "data": "relatos 2014-2024", "unidade": "AUC dentro do bairro",
        "procedimento": "mesmas ruas e mesmo orcamento (B=todas, lambda 25); IC por bootstrap de bairros (descritivo)",
        "limitacao": "chuva e saida observada nao avaliadas; densidade de entradas pode refletir resposta a problemas anteriores"})


def fig5_casos():
    t = pd.read_csv(TER / "tabela_295_fcu.csv", dtype={"CD_FCU": str})
    v = t[t.avaliada_fora_do_ajuste].copy()
    v["delta"] = v.relato_teste_pct_depois_medio - v.relato_teste_pct_antes_medio
    v = v.sort_values("delta").reset_index(drop=True)
    fig, a = plt.subplots(figsize=(8.5, 8))
    y = np.arange(len(v))
    for i, r in v.iterrows():
        cor = C["azul"] if r.delta > 0.05 else (C["vermelho"] if r.delta < -0.05 else C["cinza"])
        a.plot([r.relato_teste_pct_antes_medio, r.relato_teste_pct_depois_medio], [i, i], color=cor, lw=2)
    a.scatter(v.relato_teste_pct_antes_medio, y, color=C["cinza"], s=18, zorder=3, label="antes: Q transferido")
    a.scatter(v.relato_teste_pct_depois_medio, y, color=C["tinta"], s=18, marker="D", zorder=3, label="depois: Q + drenagem/saida")
    a.set_yticks(y); a.set_yticklabels([f"{n[:36]} ({int(p) if pd.notna(p) else '?'})" for n, p in zip(v.NM_FCU, v.pessoas_censo2022)], fontsize=6.5)
    a.set_xlim(0, 1); a.set_xlabel("percentil (na cidade) das ruas com relato da comunidade")
    a.set_title(f"Comunidades avaliadas fora do ajuste ({len(v)} de 295): azul = ganho > 0,05, vermelho = perda > 0,05", fontsize=8.5, loc="left")
    a.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
    salvar(fig, "fig5_casos_ganho_falha", {
        "fonte": "territorio/tabela_295_fcu.csv", "populacao": "FCU com rua de bairro de teste com relato; entre parenteses pessoas do Censo 2022 (universo)",
        "data": "relatos 2014-2024", "unidade": "percentil na populacao de ruas", "procedimento": "regra declarada: todas as FCU avaliadas, sem selecao por ganho",
        "limitacao": ("poucas ruas com relato por comunidade e varias FCU compartilham a mesma avenida relatada (nao sao casos "
                      "independentes); o 'depois' (A4) tem ganho explicado pelo comprimento da rua nas emendas 1-2; "
                      "o percentil e da cidade, a metrica primaria e dentro do bairro")})


def fig6_cobertura():
    cob = ler_json(TER / "cobertura_295.json")
    fig, a = plt.subplots(figsize=(8, 3.2))
    itens = [("FCU em Recife", cob["fcu_total"]), ("cobertas pela populacao do experimento", cob["fcu_cobertas"]),
             ("com 95% da area nas folhas PE3D", cob["fcu_moldura_pe3d_95"]), ("com rua relatada (SEDEC)", cob["fcu_com_rua_com_relato"]),
             ("avaliadas fora do ajuste (bairros de teste)", cob["fcu_avaliadas_fora_do_ajuste"])]
    y = np.arange(len(itens))[::-1]
    a.barh(y, [i[1] for i in itens], color=[C["cinza"], C["azul"], C["laranja"], C["vermelho"], C["violeta"]], height=0.6)
    for yy, (_, v) in zip(y, itens):
        a.annotate(str(v), (v, yy), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    a.set_yticks(y); a.set_yticklabels([i[0] for i in itens]); a.set_xlabel("numero de FCU")
    a.set_title(f"Cobertura: pessoas em FCU avaliadas {cob['pessoas_em_fcu_avaliadas']:,} de {cob['pessoas_total']:,} (universo censitario)".replace(",", "."),
                fontsize=8.5, loc="left")
    salvar(fig, "fig6_cobertura_295", {
        "fonte": "territorio/cobertura_295.json", "populacao": "295 FCU de Recife (Censo 2022)", "data": "2026-10-24",
        "unidade": "contagem de FCU", "procedimento": "denominador preservado; cobertura = ha rua da populacao; avaliada = rua de teste com relato",
        "limitacao": "cobertura de informacao nao e reducao de risco; FCU fora da populacao ficam desconhecidas, nao seguras"})


def fig7_comprimento():
    e1 = pd.read_csv(RES / "emenda1/metricas_E1.csv").set_index("vetor")
    e2 = pd.read_csv(RES / "emenda2/metricas_E2.csv").set_index("vetor")
    m = pd.read_csv(RES / "metricas_bracos.csv")
    hand = m[(m.braco == "A0_regra") & (m.subconjunto == "cidade")].AUC_bairro.iloc[0]
    itens = [("regra HAND 30 m", hand, "cidade"), ("Q transferido", e1.loc["A1_Q", "AUC_bairro"], "cidade"),
             ("so comprimento da rua", e1.loc["regra_comprimento", "AUC_bairro"], "cidade"),
             ("Q atualizado + comprimento", e1.loc["A3_comp", "AUC_bairro"], "cidade"),
             ("+ drenagem/saida (protocolo, sem comprimento)", e1.loc["A4", "AUC_bairro"], "cidade"),
             ("+ drenagem/saida + comprimento", e1.loc["A4_comp", "AUC_bairro"], "cidade"),
             ("+ saida pelo centroide (sem comprimento)", e1.loc["A4c", "AUC_bairro"], "cidade"),
             ("[PE3D] Q + comprimento", e2.loc["A3_comp", "AUC_bairro"], "pe3d"),
             ("[PE3D] + drenagem/saida + comprimento", e2.loc["A4p_comp", "AUC_bairro"], "pe3d"),
             ("[PE3D] + PE3D + comprimento", e2.loc["A5_comp", "AUC_bairro"], "pe3d")]
    fig, a = plt.subplots(figsize=(8.5, 4.4))
    y = np.arange(len(itens))[::-1]
    for yy, (rot, v, sub) in zip(y, itens):
        a.barh(yy, v - 0.5, left=0.5, height=0.6, color=C["laranja"] if "comprimento" in rot and "sem" not in rot else
               (C["violeta"] if sub == "pe3d" else C["azul"]))
        a.annotate(f"{v:.3f}", (v, yy), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)
    a.set_yticks(y); a.set_yticklabels([i[0] for i in itens], fontsize=7.5)
    a.axvline(0.5, color=C["tinta2"], lw=0.8, ls=":")
    a.set_xlabel("AUC dentro do bairro no teste")
    a.set_title("Emendas 1-2 (pos-teste): comprimento da rua explica o ganho urbano e de PE3D", fontsize=8.5, loc="left")
    salvar(fig, "fig7_confundidor_comprimento", {
        "fonte": "resultados/emenda1 e emenda2", "populacao": "teste (48 positivas); subconjunto PE3D com 18 positivas de teste e 20 de ajuste",
        "data": "relatos 2014-2024", "unidade": "AUC dentro do bairro",
        "procedimento": "diagnosticos congelados apos ver o protocolo v1; mesmas particoes e lambda 25",
        "limitacao": "desenvolvimento pos-teste; comprimento mede oportunidade de registro, nao processo fisico"})


def tabelas():
    TAB.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(RES / "metricas_bracos.csv")
    m.groupby(["subconjunto", "braco", m.B.astype(str)]).agg(AUC_bairro=("AUC_bairro", "mean"), AUC_min=("AUC_bairro", "min"),
                                                              AUC_max=("AUC_bairro", "max"), AP=("AP", "mean"),
                                                              rec_q20=("rec_q20_bairro", "mean"), sementes=("AUC_bairro", "size")).reset_index().to_csv(TAB / "tabela_bracos.csv", index=False)
    c = pd.read_csv(RES / "contrastes.csv")
    c.to_csv(TAB / "tabela_contrastes.csv", index=False)
    pd.read_csv(RES / "retencao_origem.csv").to_csv(TAB / "tabela_retencao.csv", index=False)
    pd.read_csv(TER / "tabela_295_fcu.csv", dtype={"CD_FCU": str}).to_csv(TAB / "tabela_295_fcu.csv", index=False, encoding="utf-8-sig")
    ref = ler_json(OUT / "referencias/referencias.json")
    pd.DataFrame([{"item": "relatos SEDEC de agua (2014-2024)", "n": ref["relatos_total"]},
                  {"item": "relatos geocodificados ate a rua", "n": ref["relatos_geocodificados_total"]},
                  {"item": "ruas na populacao", "n": ref["objetos_populacao"]}, {"item": "ruas com relato", "n": ref["objetos_positivos"]},
                  {"item": "bairros", "n": ref["bairros_populacao"]}, {"item": "ruas na cobertura PE3D", "n": ref["objetos_populacao_em_pe3d"]},
                  {"item": "ruas com relato na cobertura PE3D", "n": ref["positivos_em_pe3d"]}]).to_csv(TAB / "tabela_populacao.csv", index=False)


def main() -> int:
    e = Etapa("ra_06_figuras")
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig1_caso_processo, fig2_mapa_real, fig3_orcamento_retencao, fig4_blocos, fig5_casos, fig6_cobertura, fig7_comprimento):
        print("[fig]", f.__name__, flush=True)
        f()
    (FIG / "legendas.md").write_text("# Legendas das figuras — Recife\n\n" + "\n".join(LEG), encoding="utf-8")
    tabelas()
    e.saida(*FIG.glob("*"), *TAB.glob("*"))
    e.fechar(figuras=len(LEG))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
