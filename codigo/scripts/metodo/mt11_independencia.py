"""MT11, emenda: grafo comum de dependência na área EFETIVAMENTE amostrada, antes de qualquer escore.

O pré-registro chamou de unidade a bacia principal do CENTRO do tile, com `MAIN_BAS` lido da tabela
HydroBASINS de nível 8. O MT01 e a rodada 3 usam `mt01_populacao.localizar_bacias`: `MAIN_BAS` do
nível 6 e sub-bacia do nível 8. Como o identificador do HydroBASINS carrega o nível, a conferência
"nenhum candidato divide bacia com o MT" comparava IDs de níveis diferentes e nunca podia achar
coincidência. Aqui tudo passa pelo localizador canônico, nos pontos de cada evento.

Nós do grafo (evento -> pontos com lat/lon, unidade espacial, L8, MAIN_BAS L6, janela):
  desenvolvimento e examinados   `rodada3_confirmacao.populacao_examinada` (população MT01 + CEMS
                                 examinados do MT03/MT04 + códigos do ledger da rodada 3)
  ds06 inteiro                   até 40 pontos por (evento, unidade espacial) — inclui UK, Chicago,
                                 NYC e GFD conhecidos
  quarentena GFD (gfdq03)        16 eventos examinados em 03/10
  rodada 3 selecionados          31 ativações CEMS 2019-2021: pontos da camada NEUTRA de AOI dos
                                 pacotes (mesma extração da seleção dela; nenhum rótulo lido)
  MT11                           45 eventos: os pontos amostrados (posição, nunca a classe)
Regras: `mt01_independencia.grafo` — primário (evento, L8/unidade, onda MAIN_BAS 30 d, chuva 500 km
15 d) e sensibilidade conservadora `exutorio` (MAIN_BAS em qualquer data).

Saídas em `local_runs/mt11-transdutivo/independencia/`: nós, ligações por evento MT11, componentes,
unidades finais e exclusões. Nenhum escore, AUC ou classe de ponto é usado.
Uso: python scripts/metodo/mt11_independencia.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for s in ("ia", "metodo", "servico", "geo", "suscetibilidade", "terreno", "sar"):
    sys.path.insert(0, str(REPO / "scripts" / s))
import mt01_independencia as ind  # noqa: E402
import mt01_populacao as mp  # noqa: E402
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402

MT11 = REPO / "local_runs" / "mt11-transdutivo"
SAIDA = MT11 / "independencia"
GFD_DATAS = MT11 / "gfd_datas.json"
COLS = ["evento_id", "fonte", "classe", "janela_inicio", "janela_fim", "lat", "lon", "unidade_espacial", "HYBAS_L8", "MAIN_BAS"]


def bacias(d: pd.DataFrame) -> pd.DataFrame:
    b, _ = mp.localizar_bacias(d.lon.to_numpy(), d.lat.to_numpy())
    d = d.reset_index(drop=True).copy()
    d["MAIN_BAS"], d["HYBAS_L8"] = b.MAIN_BAS.to_numpy(), b.HYBAS_L8.to_numpy()
    return d


def datas_gfd() -> dict:
    if not GFD_DATAS.exists():
        import ee
        ee.Initialize()
        col = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")
        ids, t0, t1 = (col.aggregate_array(p).getInfo() for p in ("id", "system:time_start", "system:time_end"))
        import datetime as dt
        f = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.UTC).date().isoformat()  # noqa: E731
        GFD_DATAS.write_text(json.dumps({f"GFD:{i}": [f(a), f(b)] for i, a, b in zip(ids, t0, t1)}), encoding="utf-8")
    return json.loads(GFD_DATAS.read_text(encoding="utf-8"))


def no_mt11(datas: dict) -> pd.DataFrame:
    sel = json.loads((MT11 / "selecao.json").read_text(encoding="utf-8"))
    partes = []
    for ev in sel["eventos"]:
        t = ev["tiles"][0]
        p = pd.read_csv(MT11 / "pontos" / f"pontos_gfd{ev['id']}_{t['tile']}.csv", usecols=["lat", "lon", "tile"])
        i, f = datas[ev["evento_id"]]
        partes.append(p.assign(evento_id="MT11:" + ev["evento_id"], fonte="mt11", classe=0, janela_inicio=i,
                               janela_fim=f, unidade_espacial=p.tile).drop(columns="tile"))
    return bacias(pd.concat(partes, ignore_index=True))


def no_quarentena(datas: dict) -> pd.DataFrame:
    q = pd.read_parquet(REPO / "local_runs/gfd-quarentena/pontos_quarentena.parquet", columns=["evento_id", "lat", "lon", "tile"])
    q = q.sample(frac=1, random_state=0).groupby("evento_id").head(400)
    q = q.assign(fonte="gfd_quarentena", classe=0, janela_inicio=q.evento_id.map(lambda e: datas[e][0]),
                 janela_fim=q.evento_id.map(lambda e: datas[e][1]), unidade_espacial=q.tile).drop(columns="tile")
    q["evento_id"] = "GFDQ:" + q.evento_id
    return bacias(q)


def no_ds06() -> pd.DataFrame:
    from ds06_dataset_de_treino import versao_atual
    d = pd.read_parquet(versao_atual() / "pontos.parquet",
                        columns=["fonte", "evento_id", "janela_inicio", "janela_fim", "lat", "lon", "grupo_validacao"])
    d = d.dropna(subset=["evento_id", "lat", "lon"])
    d = d.sample(frac=1, random_state=0).groupby(["evento_id", "grupo_validacao"]).head(40)
    d = d.assign(classe=0, unidade_espacial=d.grupo_validacao.astype(str)).drop(columns="grupo_validacao")
    d["evento_id"] = "DS06:" + d.evento_id.astype(str)
    return bacias(d)


def no_rodada3() -> pd.DataFrame:
    import rodada3_confirmacao as r3
    import mt03_triagem_cems as tri
    from shapely import to_wkt, union_all
    c3 = r3.C3
    sel = json.loads((c3 / "selecao.json").read_text(encoding="utf-8"))
    cat = {a["code"]: a for a in json.loads((r3.led.V4 / "triagem_historica" / "catalogo.json").read_text(encoding="utf-8"))["ativacoes"]}
    linhas = []
    for s in sel["selecao"]:
        if s["estado"] != "selecionado_para_rotulo":
            continue
        inicio = (cat.get(s["code"]) or {}).get("activationTime", "")[:10] or None
        for a in s["aois"]:
            with tempfile.TemporaryDirectory(prefix="mt11_r3_") as tmp:
                r3._extrair(REPO / a["arquivo"], Path(tmp))
                g, _, _ = r3._camada(Path(tmp), r3.AOI_RE, 4326)
            if not g:
                continue
            for lon, lat in tri._pontos_da_aoi(to_wkt(union_all(g))):
                linhas.append({"evento_id": "R3:" + s["code"], "fonte": "rodada3", "classe": 0, "janela_inicio": inicio,
                               "janela_fim": a.get("data_cena_pos") or inicio, "lon": lon, "lat": lat,
                               "unidade_espacial": f"{s['code']}_AOI{a['aoi']:02d}"})
    return bacias(pd.DataFrame(linhas))


def no_examinados() -> pd.DataFrame:
    import rodada3_confirmacao as r3
    cat = {a["code"]: a for a in json.loads((r3.led.V4 / "triagem_historica" / "catalogo.json").read_text(encoding="utf-8"))["ativacoes"]}
    e = r3.populacao_examinada(cat)
    e["evento_id"] = "EXAM:" + e.evento_id.astype(str).str.replace("^EXAM:", "", regex=True)
    return e


def classificar_mt11(juntos: pd.DataFrame, desenho: str) -> dict:
    """Componentes do grafo e ligações de cada evento MT11 a nós de outras origens (sem classe, sem escore)."""
    comp, lig = ind.grafo(juntos, desenho)
    s = pd.Series(comp)
    eventos = []
    for ev in sorted(e for e in juntos.evento_id.unique() if str(e).startswith("MT11:")):
        membros = s[s == comp[ev]].index
        outros = sorted(m for m in membros if not m.startswith("MT11:"))
        eventos.append({"evento_id": ev, "componente": comp[ev], "mt11_no_componente": sorted(m for m in membros if m.startswith("MT11:")),
                        "ligado_a": outros[:12], "n_ligados": len(outros),
                        "ligado_a_desenvolvimento_ou_examinado": any(o.startswith(("EXAM:", "DS06:", "GFDQ:")) for o in outros),
                        "ligado_a_rodada3": any(o.startswith("R3:") for o in outros)})
    livres = [e for e in eventos if not e["ligado_a_desenvolvimento_ou_examinado"]]
    return {"ligacoes": lig, "eventos": eventos, "componentes_mt11": len({e["componente"] for e in eventos}),
            "componentes_mt11_livres": len({e["componente"] for e in livres}), "eventos_livres": len(livres),
            "eventos_ligados_a_examinado": [e["evento_id"] for e in eventos if e["ligado_a_desenvolvimento_ou_examinado"]],
            "eventos_ligados_a_rodada3": [e["evento_id"] for e in eventos if e["ligado_a_rodada3"]]}


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    datas = datas_gfd()
    partes = {"mt11": no_mt11(datas), "quarentena": no_quarentena(datas), "ds06": no_ds06(),
              "rodada3": no_rodada3(), "examinados": no_examinados()}
    juntos = pd.concat([p.reindex(columns=COLS) for p in partes.values()], ignore_index=True)
    for c in ("HYBAS_L8", "MAIN_BAS"):
        juntos[c] = pd.to_numeric(juntos[c], errors="coerce").fillna(0).astype("int64")
    juntos.to_parquet(SAIDA / "nos.parquet", index=False)
    m11 = partes["mt11"]
    sel = json.loads((MT11 / "selecao.json").read_text(encoding="utf-8"))
    centro = {"MT11:" + e["evento_id"]: e["unidade"] for e in sel["eventos"]}
    cobertura = []
    for ev, g in m11.groupby("evento_id"):
        mb = g.MAIN_BAS[g.MAIN_BAS > 0].value_counts()
        cobertura.append({"evento_id": ev, "unidade_pre_registro_centro_nivel8": centro[ev], "pontos": len(g),
                          "sem_bacia": int((g.MAIN_BAS <= 0).sum()), "main_bas_l6": mb.to_dict(),
                          "main_bas_dominante": int(mb.index[0]) if len(mb) else None,
                          "fracao_dominante": float(mb.iloc[0] / mb.sum()) if len(mb) else None,
                          "subbacias_l8": int(g.HYBAS_L8[g.HYBAS_L8 > 0].nunique())})
    res = {"tipo": "mt11_independencia_grafo_comum", "quando_utc": agora_utc(), "codigo_sha256": sha256_arquivo(Path(__file__)),
           "nos_sha256": sha256_arquivo(SAIDA / "nos.parquet"), "regra": ind.REGRA,
           "nos_por_parte": {k: {"eventos": int(v.evento_id.nunique()), "pontos": len(v)} for k, v in partes.items()},
           "cobertura_mt11": cobertura, "desenhos": {}}
    for desenho in ("primario", "exutorio"):
        res["desenhos"][desenho] = classificar_mt11(juntos, desenho)
    res["hash"] = sha256_json({k: v for k, v in res.items() if k != "quando_utc"})
    (SAIDA / "grafo.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({d: {k: v for k, v in x.items() if k not in ("eventos",)} for d, x in res["desenhos"].items()},
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
