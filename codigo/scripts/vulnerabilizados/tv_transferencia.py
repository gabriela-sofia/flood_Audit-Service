"""Nucleo da transferencia sem rotulo do destino: pool de doadores, retencao, selecao e ajuste.

Reaproveita o desenho de `scripts/evolucao_escopo/contexto_local.py` (logistica C=1 balanceada,
StandardScaler so no treino, orcamento pareado por classe, K doadores) com tres mudancas declaradas:
- doadores vem do ds-06 v8 (GFD e CEMS, cadeia FABDEM/WBT 30 m), agrupados pelo componente primario
  do grafo comum EAL2; a retencao estrita retira o componente inteiro do destino;
- preditores: log1p(HAND), log1p(declive), TWI D-infinito (as colunas que existem no ds-06);
- descritor de contexto: log1p(p90 HAND) e log1p(mediana do declive) dos pontos SEM usar classe,
  no doador e no destino (selecao transdutiva quanto a entradas, indutiva quanto a rotulos).
Nenhuma funcao daqui recebe rotulo do destino.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from comum_tv import RUNS

DS06 = RUNS / "ds-06-dataset-de-treino/v8/pontos.parquet"
GRAFO = RUNS / "evolucao-acervo-local/v2/grafo_comum/grafo"
PREDITORES = ["hand_m", "slope_deg", "twi_dinf"]
MIN_POR_CLASSE_DOADOR = 20
TETO_POR_CLASSE = 1000
K = 6
SEMENTES = [20261020, 20261021, 20261022]
BRACOS = ["geral", "contexto", "proximidade", "aleatorio"]


def matriz(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.log1p(df.hand_m.clip(lower=0)), np.log1p(df.slope_deg.clip(lower=0)),
                            df.twi_dinf.clip(0, 40)])


def descritor(df: pd.DataFrame) -> pd.Series:
    """Contexto de terreno sem classe: p90 de HAND e mediana do declive, em log1p."""
    return pd.Series({"hand": float(np.log1p(max(np.quantile(df.hand_m.clip(lower=0), .9), 0))),
                      "slope": float(np.log1p(max(np.median(df.slope_deg.clip(lower=0)), 0)))})


def carregar_doadores() -> pd.DataFrame:
    d = pd.read_parquet(DS06, columns=["fonte", "evento_id", "classe", "lat", "lon", *PREDITORES])
    d = d[d.fonte.isin(["gfd", "cems"]) & d.classe.isin([0, 1])].copy()
    d = d[np.isfinite(d[PREDITORES]).all(axis=1) & (d.hand_m > -1000)].reset_index(drop=True)
    d["evento"] = d.evento_id.str.replace("CEMS:", "", regex=False)
    comp = pd.read_parquet(GRAFO / "componentes.parquet")
    comp = comp[comp.desenho.eq("primario")].set_index("evento").componente
    d["componente"] = d.evento.map(comp)
    if d.componente.isna().any():
        raise ValueError("evento doador fora do grafo comum")
    return d


def bacias_dos_eventos() -> pd.DataFrame:
    m = pd.read_parquet(GRAFO / "membros_bacia.parquet", columns=["evento", "nivel", "bacia"])
    return m[m.nivel.eq("MAIN_BAS")].drop_duplicates()


def main_bas_de_pontos(lon: np.ndarray, lat: np.ndarray) -> set[int]:
    """MAIN_BAS do HydroBASINS nivel 6 (America do Sul) que contem os pontos do destino."""
    import shapely
    t = pd.read_parquet(RUNS / "hydrobasins/v1c/tabelas/nivel_06.parquet",
                        columns=["HYBAS_ID", "MAIN_BAS", "lon_min", "lat_min", "lon_max", "lat_max", "wkb"])
    t = t[(t.lon_max >= lon.min()) & (t.lon_min <= lon.max()) & (t.lat_max >= lat.min()) & (t.lat_min <= lat.max())]
    geoms = shapely.from_wkb(t.wkb.to_numpy())
    pts = shapely.points(lon, lat)
    achados = set()
    for g, mb in zip(geoms, t.MAIN_BAS):
        if shapely.intersects(g, pts).any():
            achados.add(int(mb))
    return achados


def reter(doadores: pd.DataFrame, modo: str, alvo_lon: np.ndarray, alvo_lat: np.ndarray,
          episodio: str, raio_km: float = 50.0) -> tuple[pd.DataFrame, dict]:
    """Retira do pool tudo o que depende do destino.

    estrita: componente primario do episodio do destino inteiro (dependencia transitiva);
    local: eventos do mesmo episodio, eventos com ponto no MAIN_BAS do destino e pontos a menos
    de `raio_km` do destino. Em ambos, unidades sem `MIN_POR_CLASSE_DOADOR` por classe saem.
    """
    mbs = main_bas_de_pontos(alvo_lon, alvo_lat)
    bac = bacias_dos_eventos()
    ev_mesma_bacia = set(bac[bac.bacia.isin(mbs)].evento)
    comp_episodio = set(doadores.loc[doadores.evento.eq(episodio), "componente"])
    if not comp_episodio:
        g = pd.read_parquet(GRAFO / "componentes.parquet")
        comp_episodio = set(g[(g.desenho == "primario") & g.evento.eq(episodio)].componente)
    if modo == "estrita":
        fora = doadores.componente.isin(comp_episodio) | doadores.evento.isin(ev_mesma_bacia)
        unidade = "componente"
    elif modo == "local":
        fora = doadores.evento.eq(episodio) | doadores.evento.isin(ev_mesma_bacia)
        unidade = "evento"
    else:
        raise ValueError(modo)
    d = doadores[~fora].copy()
    # distancia ao destino (haversine ao ponto mais proximo de uma grade grosseira do destino)
    alvo = np.column_stack([alvo_lon, alvo_lat])
    if len(alvo) > 500:  # 500 pontos do destino bastam para um raio de dezenas de km
        alvo = alvo[np.linspace(0, len(alvo) - 1, 500).astype(int)]
    lat1, lon1 = np.radians(d.lat.to_numpy())[:, None], np.radians(d.lon.to_numpy())[:, None]
    lat2, lon2 = np.radians(alvo[:, 1])[None, :], np.radians(alvo[:, 0])[None, :]
    perto = np.zeros(len(d), bool)
    for i in range(0, len(d), 4000):
        a = (np.sin((lat2 - lat1[i:i + 4000]) / 2) ** 2 + np.cos(lat1[i:i + 4000]) * np.cos(lat2)
             * np.sin((lon2 - lon1[i:i + 4000]) / 2) ** 2)
        perto[i:i + 4000] = (2 * 6371 * np.arcsin(np.sqrt(a))).min(axis=1) < raio_km
    d = d[~perto].copy()
    d["unidade_doadora"] = d[unidade]
    cont = d.groupby(["unidade_doadora", "classe"]).size().unstack(fill_value=0)
    ok = cont[(cont.get(0, 0) >= MIN_POR_CLASSE_DOADOR) & (cont.get(1, 0) >= MIN_POR_CLASSE_DOADOR)].index
    d = d[d.unidade_doadora.isin(ok)].reset_index(drop=True)
    info = {"modo": modo, "main_bas_destino": sorted(mbs), "componentes_do_episodio": sorted(comp_episodio),
            "eventos_mesma_bacia_retirados": len(ev_mesma_bacia), "pontos_a_menos_de_km": int(perto.sum()),
            "raio_km": raio_km, "unidades_doadoras": int(d.unidade_doadora.nunique()),
            "unidades_sem_duas_classes_suficientes": int(len(cont) - len(ok)),
            "pontos_doadores": int(len(d))}
    return d, info


def descritores_doadores(d: pd.DataFrame) -> pd.DataFrame:
    desc = d.groupby("unidade_doadora").apply(descritor, include_groups=False)
    cent = d.groupby("unidade_doadora")[["lat", "lon"]].mean()
    return desc.join(cent)


def selecionar(desc: pd.DataFrame, alvo_desc: pd.Series, alvo_lon: float, alvo_lat: float,
               semente: int) -> dict[str, list]:
    """Conjuntos de doadores por braco; nenhum usa rotulo do destino."""
    unidades = sorted(desc.index)
    z = desc[["hand", "slope"]]
    sd = z.std(ddof=0).replace(0, 1)
    dist_ctx = (((z - alvo_desc[["hand", "slope"]]) / sd) ** 2).sum(axis=1)
    contexto = list(dist_ctx.sort_values(kind="stable").index[:K])
    la1, lo1 = np.radians(alvo_lat), np.radians(alvo_lon)
    la2, lo2 = np.radians(desc.lat), np.radians(desc.lon)
    km = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    proximidade = list(pd.Series(km, index=desc.index).sort_values(kind="stable").index[:K])
    rng = np.random.default_rng(semente)
    aleatorio = list(rng.choice(unidades, size=min(K, len(unidades)), replace=False))
    return {"geral": unidades, "contexto": contexto, "proximidade": proximidade, "aleatorio": aleatorio}


def ajustar_bracos(d: pd.DataFrame, conjuntos: dict[str, list], semente: int):
    """Amostra ate TETO_POR_CLASSE por unidade/classe; iguala o total por classe entre bracos; ajusta."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(semente)
    amostras = {}
    for nome, us in conjuntos.items():
        sub = d[d.unidade_doadora.isin(us)]
        idx = []
        for (_, _), q in sub.groupby(["unidade_doadora", "classe"]):
            idx.append(rng.choice(q.index.to_numpy(), size=min(TETO_POR_CLASSE, len(q)), replace=False))
        amostras[nome] = np.concatenate(idx)
    totais = {cl: min(int((d.loc[a, "classe"] == cl).sum()) for a in amostras.values()) for cl in (0, 1)}
    if min(totais.values()) == 0:
        raise ValueError("orcamento comum sem uma classe")
    modelos, registro = {}, {}
    for nome, a in amostras.items():
        cls = d.loc[a, "classe"].to_numpy()
        sel = np.concatenate([rng.choice(a[cls == cl], totais[cl], replace=False) for cl in (0, 1)])
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=300,
                                                               random_state=semente))
        m.fit(matriz(d.loc[sel]), d.loc[sel, "classe"])
        modelos[nome] = m
        registro[nome] = {"unidades": [str(u) for u in conjuntos[nome]], "pontos_por_classe": totais,
                          "coef_padronizado": [float(c) for c in m[-1].coef_[0]],
                          "intercepto": float(m[-1].intercept_[0])}
    return modelos, registro


def dominio(d: pd.DataFrame) -> dict:
    """Caixa de aplicabilidade fixa: percentis 1-99 de cada preditor transformado no pool doador."""
    x = matriz(d)
    return {"p01": [float(v) for v in np.quantile(x, .01, axis=0)], "p99": [float(v) for v in np.quantile(x, .99, axis=0)]}


def dentro_do_dominio(df: pd.DataFrame, dom: dict) -> np.ndarray:
    x = matriz(df)
    return ((x >= np.array(dom["p01"])) & (x <= np.array(dom["p99"]))).all(axis=1)
