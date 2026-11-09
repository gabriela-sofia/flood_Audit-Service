"""MT01, passo 2: desenho de independencia -- por que o TT01 virou um componente e o que separa o MT01.

Duas perguntas diferentes, que o TT01 respondia com uma so regra:

  1. Diagnostico do TT01 (populacao GFD congelada, sem reajustar nada).
     Quantas ligacoes de cada tipo existem, quais eventos fazem ponte, e quanto
     do componente unico e LUGAR observado de novo, COVARIAVEL deterministica
     compartilhada (raster de terreno), BACIA (MAIN_BAS) ou EVENTO.
  2. Grafo de dependencia do MT01 (populacao multi-fonte). Ligacoes:
       L_evento  mesmo evento (mesma tempestade/cheia catalogada)
       L_lugar   mesma sub-bacia nivel 8 ou mesma unidade espacial (lugar repetido)
       L_onda    mesmo MAIN_BAS (nivel 6) com janelas a menos de 30 dias (mesma onda de cheia)
       L_chuva   pegadas a menos de 500 km com janelas a menos de 15 dias (mesmo sistema de chuva)
     Desenho primario = uniao das quatro. Sensibilidade conservadora: MAIN_BAS
     em qualquer data (mesmo exutorio). Sensibilidade liberal: so evento e lugar.

Compartilhar o raster de onde saem HAND/TWI nao e compartilhar rotulo: o valor da
covariavel e funcao deterministica do MDT, igual para qualquer evento. Lugar
repetido e outra coisa -- o mesmo pixel observado em dois eventos -- e entra na
dependencia. Nenhum componente e dissolvido para ganhar unidades.

Uso:
    python scripts/metodo/mt01_independencia.py diagnosticar_tt01
    python scripts/metodo/mt01_independencia.py unidades
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc, sha256_arquivo  # noqa: E402
import mt01_populacao as mp  # noqa: E402

VERSAO = "mt01_independencia_v1"
TT01 = mp.RUNS / "teste-transferencia-territorial" / "v2"
REGRA = {"onda_dias": 30, "chuva_km": 500, "chuva_dias": 15}


class Uniao:
    def __init__(self, itens):
        self.pai = {i: i for i in sorted(itens)}

    def raiz(self, i):
        while self.pai[i] != i:
            self.pai[i] = self.pai[self.pai[i]]
            i = self.pai[i]
        return i

    def unir(self, a, b):
        a, b = self.raiz(a), self.raiz(b)
        if a != b:
            self.pai[max(a, b)] = min(a, b)

    def grupos(self) -> dict:
        return {i: self.raiz(i) for i in self.pai}


def ligar_por_valor(u: Uniao, pares: pd.DataFrame, coluna: str) -> int:
    """Une eventos que compartilham um valor; devolve quantas ligacoes novas ocorreram."""
    antes = len(set(u.grupos().values()))
    for _, g in pares[["evento_id", coluna]].dropna().drop_duplicates().groupby(coluna):
        ev = g.evento_id.tolist()
        for e in ev[1:]:
            u.unir(ev[0], e)
    return antes - len(set(u.grupos().values()))


def n_componentes(pares: pd.DataFrame, colunas, excluir=()) -> int:
    p = pares[~pares.evento_id.isin(excluir)]
    u = Uniao(p.evento_id.unique())
    for c in colunas:
        p2 = p[~p[c].isin([0, ""])] if p[c].dtype.kind in "iuf" or p[c].dtype == object else p
        ligar_por_valor(u, p2, c)
    return len(set(u.grupos().values()))


# ------------------------------------------------------------- TT01

def diagnosticar_tt01() -> dict:
    pop = pd.read_parquet(TT01 / "populacao.parquet",
                          columns=["evento_id", "lon", "lat", "unidade_terreno", "y", "tile", "MAIN_BAS"])
    pares = pop[["evento_id", "tile", "MAIN_BAS", "unidade_terreno"]].drop_duplicates()
    graus_ev = pares.groupby("evento_id").tile.nunique()
    graus_tile = pares.groupby("tile").evento_id.nunique()
    r = {"eventos": int(pares.evento_id.nunique()), "tiles": int(pares.tile.nunique()),
         "tiles_por_evento": {"mediana": float(graus_ev.median()), "max": int(graus_ev.max()),
                              "eventos_com_mais_de_10_tiles": int((graus_ev > 10).sum())},
         "eventos_por_tile": {"mediana": float(graus_tile.median()), "max": int(graus_tile.max()),
                              "tiles_em_mais_de_1_evento": int((graus_tile > 1).sum())}}
    r["componentes"] = {
        "so_tile": n_componentes(pares, ["tile"]),
        "so_MAIN_BAS": n_componentes(pares, ["MAIN_BAS"]),
        "so_raster": n_componentes(pares, ["unidade_terreno"]),
        "tile_MAIN_BAS_raster": n_componentes(pares, ["tile", "MAIN_BAS", "unidade_terreno"]),
    }
    # pontes: retirar eventos de maior grau, um a um, e contar componentes por tile
    ordem = graus_ev.sort_values(ascending=False, kind="stable").index.tolist()
    pontes = []
    for k in (0, 1, 2, 3, 5, 10, 20):
        fora = ordem[:k]
        pontes.append({"retirados": k, "componentes_tile": n_componentes(pares, ["tile"], fora),
                       "componentes_tile_bacia": n_componentes(pares, ["tile", "MAIN_BAS"], fora),
                       "eventos_restantes": r["eventos"] - k})
    r["pontes_por_grau"] = pontes
    r["eventos_de_maior_grau"] = {e: int(graus_ev[e]) for e in ordem[:5]}
    # lugar repetido: a mesma coordenada (arredondada a ~250 m, a celula do rotulo) em mais de um evento
    cel = pop.assign(cx=(pop.lon / 0.0025).round().astype(int), cy=(pop.lat / 0.0025).round().astype(int))
    por_cel = cel.groupby(["cx", "cy"]).agg(eventos=("evento_id", "nunique"), y_min=("y", "min"), y_max=("y", "max"))
    rep = por_cel[por_cel.eventos > 1]
    r["lugar_repetido"] = {"celulas_250m": int(len(por_cel)), "celulas_em_mais_de_1_evento": int(len(rep)),
                           "fracao": round(len(rep) / len(por_cel), 4),
                           "celulas_repetidas_com_rotulos_diferentes": int((rep.y_min != rep.y_max).sum())}
    sistemas = pares.groupby("MAIN_BAS").evento_id.nunique().sort_values(ascending=False)
    r["MAIN_BAS"] = {"sistemas": int(len(sistemas)), "eventos_no_maior_sistema": int(sistemas.iloc[0]),
                     "sistemas_com_mais_de_10_eventos": int((sistemas > 10).sum())}
    r["leitura"] = (
        "O componente unico nao vem de rotulo compartilhado. Vem de (a) eventos GFD que cobrem muitos tiles "
        "e de tiles revisitados por varios eventos -- lugar observado de novo -- e (b) do MAIN_BAS, que poe no "
        "mesmo sistema eventos de anos diferentes. O raster de terreno compartilhado e covariavel deterministica.")
    r["entradas"] = {"populacao.parquet": sha256_arquivo(TT01 / "populacao.parquet")}
    return r


# ------------------------------------------------------------- MT01

def eventos_da_populacao(d: pd.DataFrame) -> pd.DataFrame:
    e = d[d.evento_id.notna()].copy()
    for c in ("janela_inicio", "janela_fim"):
        e[c] = pd.to_datetime(e[c], errors="coerce", utc=True).dt.tz_localize(None)
    g = e.groupby("evento_id").agg(fonte=("fonte", "first"), inicio=("janela_inicio", "min"),
                                   fim=("janela_fim", "max"), lat=("lat", "median"), lon=("lon", "median"),
                                   linhas=("classe", "size"), positivos=("classe", "sum"))
    return g.reset_index()


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def grafo(d: pd.DataFrame, desenho: str = "primario", regra: dict = REGRA) -> tuple[dict, dict]:
    """Componentes de eventos. Usa identidade, lugar, bacia, data e coordenada -- nunca a classe."""
    d = d[d.evento_id.notna()]
    ev = eventos_da_populacao(d)
    u = Uniao(ev.evento_id)
    lig = {}
    lugar = d[["evento_id", "unidade_espacial", "HYBAS_L8"]].drop_duplicates()
    lig["lugar_unidade"] = ligar_por_valor(u, lugar, "unidade_espacial")
    lig["lugar_subbacia_l8"] = ligar_por_valor(u, lugar[lugar.HYBAS_L8 > 0], "HYBAS_L8")
    if desenho == "liberal":
        return u.grupos(), lig
    sist = d[d.MAIN_BAS > 0][["evento_id", "MAIN_BAS"]].drop_duplicates().merge(ev[["evento_id", "inicio", "fim"]])
    if desenho == "exutorio":
        lig["mesmo_exutorio"] = ligar_por_valor(u, sist, "MAIN_BAS")
        return u.grupos(), lig
    antes = len(set(u.grupos().values()))
    folga = pd.Timedelta(days=regra["onda_dias"])
    for _, g in sist.groupby("MAIN_BAS"):
        g = g.drop_duplicates("evento_id").sort_values("inicio").reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.iloc[i], g.iloc[j]
                if pd.isna(a.inicio) or pd.isna(b.inicio) or (b.inicio - a.fim) <= folga:
                    u.unir(a.evento_id, b.evento_id)
    lig["onda_mesmo_main_bas"] = antes - len(set(u.grupos().values()))
    antes = len(set(u.grupos().values()))
    folga = pd.Timedelta(days=regra["chuva_dias"])
    for i in range(len(ev)):
        for j in range(i + 1, len(ev)):
            a, b = ev.iloc[i], ev.iloc[j]
            if haversine_km(a.lat, a.lon, b.lat, b.lon) > regra["chuva_km"]:
                continue
            perto_no_tempo = (pd.isna(a.inicio) or pd.isna(b.inicio) or
                              (max(a.inicio, b.inicio) - min(a.fim, b.fim)) <= folga)
            if perto_no_tempo:
                u.unir(a.evento_id, b.evento_id)
    lig["mesmo_sistema_de_chuva"] = antes - len(set(u.grupos().values()))
    return u.grupos(), lig


def unidades(pop: Path = mp.SAIDA / "populacao.parquet") -> dict:
    d = pd.read_parquet(pop)
    d = d[d.fonte.isin(mp.FONTES_CONTRASTE) & d.papel.isin(["positivo_episodico", "negativo_observado"])]
    duas = d.groupby("evento_id").classe.nunique()
    d = d[d.evento_id.isin(duas[duas == 2].index)]
    out = {"versao": VERSAO, "quando_utc": agora_utc(), "regra": REGRA, "desenhos": {}}
    tabelas = []
    for desenho in ("primario", "exutorio", "liberal"):
        comp, lig = grafo(d, desenho)
        t = d.assign(unidade=d.evento_id.map(comp)).groupby("unidade").agg(
            eventos=("evento_id", "nunique"), linhas=("classe", "size"), positivos=("classe", "sum"),
            fontes=("fonte", lambda s: "+".join(sorted(s.unique()))),
            lugares=("unidade_espacial", "nunique"))
        out["desenhos"][desenho] = {
            "componentes": int(len(t)), "ligacoes_que_uniram": lig,
            "maior_componente_eventos": int(t.eventos.max()),
            "componentes_com_fontes_misturadas": int(t.fontes.str.contains(r"\+").sum()),
            "componentes_por_fonte": t.fontes.value_counts().to_dict()}
        tabelas.append(pd.DataFrame({"evento_id": list(comp), "desenho": desenho, "unidade": list(comp.values())}))
    arq = mp.SAIDA / "unidades.parquet"
    pd.concat(tabelas).to_parquet(arq, index=False)
    out["saidas"] = {"unidades.parquet": sha256_arquivo(arq)}
    out["entradas"] = {"populacao.parquet": sha256_arquivo(pop)}
    out["codigo"] = {str(Path(__file__).relative_to(REPO)): sha256_arquivo(Path(__file__))}
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("acao", choices=("diagnosticar_tt01", "unidades"))
    a = ap.parse_args()
    if a.acao == "diagnosticar_tt01":
        r = diagnosticar_tt01()
        mp.escrever(mp.SAIDA / "diagnostico_tt01.json", {"versao": VERSAO, "quando_utc": agora_utc(), **r})
    else:
        r = unidades()
        mp.escrever(mp.SAIDA / "independencia.json", r)
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
