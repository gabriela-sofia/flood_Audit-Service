"""Versao de desenvolvimento contextualizada: o TC01 curado lido pelo acervo espacial, sem tocar no pacote congelado.

-------------------------------------------------------------------------
O QUE MUDA EM RELACAO AO PACOTE CONGELADO
-------------------------------------------------------------------------

O pacote (`pacote_de_treino`) leu o terreno de cada ponto no tile do EVENTO que o
amostrou. Perto da borda desse tile a janela de 2 km saia do raster e o ponto
ficava sem `pos_min_2km` -- mesmo quando outro tile GFD, da mesma cadeia e do
mesmo FABDEM, cobria a janela inteira. Aqui cada ponto passa por
`acervo_espacial.contextualizar`: a mesma ordem e a mesma funcao da consulta de
um lugar novo. Nada e copiado de vizinho; o que nenhuma unidade compativel cobre
continua desconhecido, com o motivo.

-------------------------------------------------------------------------
TRES FALTAS QUE NAO SE CONFUNDEM
-------------------------------------------------------------------------

  atributo      o descritor nao pode ser calculado no ponto (janela fora do
                raster, MDT sem dado). Derivacao/escolha de unidade resolve.
  observacao    o sensor nao viu (fora_da_cena) ou viu agua permanente. Nenhuma
                derivacao resolve; o ponto fica com estado e sem classe.
  independencia o ponto esta num tile/raster/componente que outro papel usa.
                Mais atributo nao resolve; so outro lugar ou outro evento.

Cada linha leva o estado observacional ORIGINAL do TC01, o papel do evento no
pacote congelado, o componente espacial e os valores do lote antigo, para que a
diferenca seja auditavel linha a linha.

nao_e: nao e teste, nao promove modelo, nao muda o escore servido e nao altera o
pacote congelado (o executor `treinar_pacote` continua lendo so ele).

Uso:
    python scripts/manutencao/executar_serializado.py -- python scripts/metodo/contexto_de_desenvolvimento.py --materializar
    python scripts/metodo/contexto_de_desenvolvimento.py --piloto GFD_S17W056
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _pasta in ("geo", "metodo", "servico", "suscetibilidade", "terreno"):
    sys.path.insert(0, str(REPO / "scripts" / _pasta))

import acervo_espacial as ae  # noqa: E402
import posicao_no_ponto as pp  # noqa: E402

RUNS = REPO / "local_runs"
SAIDA = RUNS / "contexto-desenvolvimento"
VERSAO = "v1"
TERRENO_LOTE = ("elevation_m", "hand_m", "twi_dinf", "slope_deg", *pp.COLUNAS)
VETOR = ("hand_m", "twi_dinf", "slope_deg", "pos_min_2km")
OBSERVADOS = ("positivo", "negativo_observado")
INSTRUMENTO = "MODIS Terra+Aqua, compostos de 3 dias, 250 m (GFD v1, Tellman et al. 2021)"
POPULACAO = "territorio_consultavel (TC01: grade de 1 km sobre o tile de 1 grau, delimitado sem a mancha)"


def _sha(p: Path) -> str:
    return ae.impressao(Path(p), completo=True)["hash"]


def fontes() -> dict:
    curado = sorted(RUNS.glob("tc01-curado/v1/*/pontos.parquet"), key=lambda p: p.stat().st_mtime)[-1]
    pacote = sorted(RUNS.glob("pacote-de-treino/*/manifesto.json"), key=lambda p: p.stat().st_mtime)[-1]
    return {"curado": curado, "pacote": pacote, "acervo": ae.SAIDA / "manifesto.json",
            "ds06": RUNS / "ds-06-dataset-de-treino" / "v7" / "pontos.parquet",
            "anotacoes": RUNS / "verdade-observacional" / "anotacoes_ds06_v7.parquet", "encoder": ae.ENCODER}


def carregar_base(curado: Path, unidades: list[str] | None = None) -> pd.DataFrame:
    cols = ["ponto_id", "evento_id", "unidade", "lon", "lat", "estado", "flooded", "clear_views", "jrc_perm_water",
            *TERRENO_LOTE]
    d = pd.read_parquet(curado, columns=cols)
    if unidades:
        d = d[d.unidade.isin(unidades)]
    return d.rename(columns={c: f"{c}_lote" for c in TERRENO_LOTE}).reset_index(drop=True)


def tabela_de_eventos(f: dict, eventos: list[str]) -> pd.DataFrame:
    """Janela, mecanismo e identidade canonica por evento. Nao le classe nem regime hidrico."""
    d = pd.read_parquet(f["ds06"], columns=["ponto_id", "fonte", "evento_id", "mecanismo", "janela_inicio",
                                             "janela_fim", "precisao_data", "resolucao_nativa_m"])
    d = d[d.evento_id.isin(eventos)]
    can = pd.read_parquet(f["anotacoes"], columns=["ponto_id", "evento_canonico"])
    d = d.merge(can, on="ponto_id", how="left")
    e = (d.groupby("evento_id")
         .agg(fonte=("fonte", "first"), mecanismo=("mecanismo", "first"), janela_inicio=("janela_inicio", "first"),
              janela_fim=("janela_fim", "first"), precisao_data=("precisao_data", "first"),
              resolucao_rotulo_m=("resolucao_nativa_m", "first"),
              evento_canonico=("evento_canonico", lambda s: ";".join(sorted(set(s.dropna().astype(str))))),
              janelas_distintas=("janela_inicio", "nunique"))
         .reset_index())
    if (e.janelas_distintas > 1).any():
        raise ValueError("evento com mais de uma janela no ds06; identidade ambigua")
    e["instrumento"] = INSTRUMENTO
    return e.drop(columns="janelas_distintas")


def _classe(estado: pd.Series) -> pd.Series:
    return estado.map({"positivo": 1, "negativo_observado": 0}).astype("Int8")


def anotar(base: pd.DataFrame, ctx: pd.DataFrame, eventos: pd.DataFrame, papeis: dict) -> pd.DataFrame:
    from ms7_territorio_curado import tile_geografico
    if not (base.ponto_id.to_numpy() == ctx.ponto_id.to_numpy()).all():
        raise RuntimeError("contexto fora de ordem")
    d = pd.concat([base, ctx.drop(columns="ponto_id")], axis=1)
    d["classe"] = _classe(d.estado)
    d["papel_pacote"] = d.evento_id.map(papeis).fillna("evento_fora_do_pacote")
    d["tile_geografico"] = tile_geografico(d.lon, d.lat)
    d["unidade_lote"] = "gfd_tile_" + d.unidade.str.split("_", n=1).str[1]
    d["mudou_de_unidade"] = d.unidade_terreno.notna() & (d.unidade_terreno != d.unidade_lote)
    finito_lote = np.isfinite(d[[f"{c}_lote" for c in VETOR]].to_numpy(float)).all(1)
    finito_novo = np.isfinite(d[list(VETOR)].to_numpy(float)).all(1)
    d["vetor_finito_lote"] = finito_lote
    d["vetor_finito"] = finito_novo
    d["origem_do_vetor"] = np.select(
        [finito_novo & finito_lote & ~d.mudou_de_unidade, finito_novo & finito_lote & d.mudou_de_unidade,
         finito_novo & ~finito_lote, ~finito_novo],
        ["mesma_unidade_do_lote", "unidade_mais_central", "recuperado_por_unidade_compativel", "desconhecido"],
        default="desconhecido")
    for c in ATRIBUTOS_ESTADO:
        d[f"estado_{c}"] = np.where(np.isfinite(d[c].to_numpy(float)),
                                    "medido" if c == "elevation_m" else "derivado", "desconhecido")
    d["estado_vizinhanca"] = np.where(np.isfinite(d[list(ae.COLUNAS_VIZINHANCA)].to_numpy(float)).all(1),
                                      "derivado", "desconhecido")
    d["imputado"] = False                    # nenhuma coluna e imputada nesta versao
    d["falta"] = np.select(
        [~d.estado.isin(OBSERVADOS), ~finito_novo],
        ["observacao", "atributo"], default="nenhuma")
    d = d.merge(eventos, on="evento_id", how="left", validate="many_to_one")
    d["populacao"] = POPULACAO
    return d


ATRIBUTOS_ESTADO = ("elevation_m", "hand_m", "twi_dinf", "slope_deg", "pos_min_2km")


def componentes(d: pd.DataFrame) -> pd.Series:
    """Componente evento-tile (o do MS7) recalculado sobre as linhas que ensinam nesta versao."""
    from ms7_territorio_curado import grupos_conectados
    ens = d[d.estado.isin(OBSERVADOS) & d.vetor_finito]
    eventos = sorted(ens.evento_id.unique())
    cod = {e: i for i, e in enumerate(eventos)}
    g = grupos_conectados(ens.evento_id.map(cod).to_numpy(), ens.tile_geografico.to_numpy())
    return d.evento_id.map({e: int(g[cod[e]]) for e in eventos}).astype("Int32")


def dependencia_entre_papeis(d: pd.DataFrame) -> dict:
    """Quao perto o que nao e treino esta do treino: tile, raster de terreno, distancia e janelas."""
    from sklearn.neighbors import BallTree
    ens = d[d.estado.isin(OBSERVADOS) & d.vetor_finito & d.papel_pacote.isin(("treino", "validacao",
                                                                              "avaliacao_desenvolvimento"))]
    tr = ens[ens.papel_pacote == "treino"]
    if tr.empty:
        return {"estado": "sem linhas de treino neste recorte"}
    arvore =BallTree(np.radians(tr[["lat", "lon"]].to_numpy()), metric="haversine")
    tiles_treino = set(tr.tile_geografico)
    rasters_treino = set(tr.unidade_terreno)
    out = {}
    for papel in ("validacao", "avaliacao_desenvolvimento"):
        p = ens[ens.papel_pacote == papel]
        if p.empty:
            continue
        dist, _ = arvore.query(np.radians(p[["lat", "lon"]].to_numpy()), k=1)
        m = dist[:, 0] * 6_371_000.0
        out[papel] = {
            "linhas": int(len(p)),
            "tile_geografico_compartilhado_com_treino": int(p.tile_geografico.isin(tiles_treino).sum()),
            "raster_de_terreno_compartilhado_com_treino": int(p.unidade_terreno.isin(rasters_treino).sum()),
            "distancia_ao_treino_m": {"min": round(float(m.min()), 1), "p05": round(float(np.quantile(m, 0.05)), 1),
                                      "mediana": round(float(np.median(m)), 1)},
            "fracao_a_menos_de_2km": round(float((m < 2000).mean()), 4),
            "fracao_a_menos_de_3_84km": round(float((m < 3840).mean()), 4),
            "leitura": ("janela de pos_min_2km (2 km) ou da vizinhanca (3,84 km) que alcanca ponto do treino "
                        "compartilha relevo com ele; nao e vazamento de rotulo, e dependencia espacial")}
    return out


def duplicatas(d: pd.DataFrame) -> dict:
    chave = d.lon.round(7).astype(str) + "," + d.lat.round(7).astype(str)
    por_evento = int(pd.Series(chave + "|" + d.evento_id).duplicated().sum())
    ens = d[d.estado.isin(OBSERVADOS)]
    ch = chave[ens.index]
    entre_eventos = ens.groupby(ch).evento_id.nunique()
    lugares_multi = entre_eventos[entre_eventos > 1].index
    papeis_por_lugar = ens[ch.isin(lugares_multi)].groupby(ch[ch.isin(lugares_multi)]).papel_pacote.nunique()
    return {"coordenada_repetida_no_mesmo_evento": por_evento,
            "lugares_observados_em_mais_de_um_evento": int(len(lugares_multi)),
            "linhas_nesses_lugares": int(ch.isin(lugares_multi).sum()),
            "lugares_em_mais_de_um_papel": int((papeis_por_lugar > 1).sum()),
            "leitura": "mesmo lugar em eventos diferentes e repeticao legitima; em papeis diferentes seria vazamento"}


def perdas(d: pd.DataFrame) -> dict:
    obs = d[d.estado.isin(OBSERVADOS)]
    out = {"por_estado": d.estado.value_counts().to_dict(),
           "falta": d.falta.value_counts().to_dict(),
           "observados": int(len(obs)),
           "vetor_finito_lote": int(obs.vetor_finito_lote.sum()),
           "vetor_finito": int(obs.vetor_finito.sum()),
           "origem_do_vetor": obs.origem_do_vetor.value_counts().to_dict(),
           "perdeu_o_que_o_lote_tinha": int((obs.vetor_finito_lote & ~obs.vetor_finito).sum())}
    out["por_classe"] = {}
    for k, g in obs.groupby("estado"):
        out["por_classe"][k] = {"linhas": int(len(g)), "finito_lote": int(g.vetor_finito_lote.sum()),
                                "finito": int(g.vetor_finito.sum()),
                                "fracao_perdida_lote": round(float(1 - g.vetor_finito_lote.mean()), 4),
                                "fracao_perdida": round(float(1 - g.vetor_finito.mean()), 4)}
    out["motivo_ausencia"] = obs.loc[~obs.vetor_finito, "motivo_ausencia"].value_counts(dropna=False).to_dict()
    out["por_papel"] = {p: {"linhas": int(len(g)), "finito_lote": int(g.vetor_finito_lote.sum()),
                            "finito": int(g.vetor_finito.sum()), "positivos": int((g.estado == "positivo").sum()),
                            "positivos_finitos": int(((g.estado == "positivo") & g.vetor_finito).sum())}
                        for p, g in obs.groupby("papel_pacote")}
    out["por_contexto"] = {f"{'nucleo' if k else 'margem'}": {"linhas": int(len(g)),
                                                              "finito": int(g.vetor_finito.sum())}
                           for k, g in obs.groupby(obs.no_nucleo.fillna(False).astype(bool))}
    out["por_familia_de_unidade"] = obs.unidade_compativel.map({True: "compativel", False: "incompativel"}).fillna(
        "sem_unidade").value_counts().to_dict()
    return out


def equivalencia_com_o_lote(d: pd.DataFrame) -> dict:
    """Onde a unidade e a mesma do lote, o valor tem de ser o mesmo (dev: tolerancia do float32 do lote)."""
    m = (d.unidade_terreno == d.unidade_lote)
    out = {"linhas_mesma_unidade": int(m.sum())}
    for c in TERRENO_LOTE:
        a, b = d.loc[m, c].to_numpy(float), d.loc[m, f"{c}_lote"].to_numpy(float)
        ambos = np.isfinite(a) & np.isfinite(b)
        so_um = np.isfinite(a) ^ np.isfinite(b)
        dif = np.abs(a[ambos] - b[ambos])
        out[c] = {"ambos_finitos": int(ambos.sum()), "finitude_diverge": int(so_um.sum()),
                  "iguais_exatos": int((dif == 0).sum()), "dif_max": float(dif.max()) if len(dif) else None}
    mud = d[d.mudou_de_unidade & np.isfinite(d.hand_m) & np.isfinite(d.hand_m_lote)]
    out["mudou_de_unidade"] = {
        "linhas": int(d.mudou_de_unidade.sum()),
        "dif_abs_hand_m": {"mediana": float(np.median(np.abs(mud.hand_m - mud.hand_m_lote))) if len(mud) else None,
                           "p95": float(np.quantile(np.abs(mud.hand_m - mud.hand_m_lote), 0.95)) if len(mud) else None},
        "dif_abs_elevation_m_p95": float(np.quantile(np.abs(mud.elevation_m - mud.elevation_m_lote), 0.95))
        if len(mud) else None,
        "leitura": ("HAND de dois tiles do mesmo FABDEM diverge quando a drenagem perto da borda do tile do lote "
                    "foi cortada; a regra escolhe o tile em que o ponto esta mais longe da borda")}
    return out


def unidades_de_contexto(d: pd.DataFrame, acervo: ae.Acervo) -> pd.DataFrame:
    """Contexto por raster de terreno sem rotulo: relevo de TODOS os pontos (inclusive desconhecidos)."""
    g = d[d.unidade_terreno.notna()].groupby("unidade_terreno")
    t = g.agg(pontos=("ponto_id", "size"), eventos=("evento_id", "nunique"),
              fracao_no_nucleo=("no_nucleo", lambda s: float(s.fillna(False).astype(bool).mean())),
              hand_p10=("hand_m", lambda s: s.quantile(0.1)), hand_mediana=("hand_m", "median"),
              hand_p90=("hand_m", lambda s: s.quantile(0.9)), declive_mediano=("slope_deg", "median"),
              twi_mediano=("twi_dinf", "median"), pos_min_2km_mediano=("pos_min_2km", "median")).reset_index()
    u = acervo.todas.set_index("unidade")
    for c in ("tipo", "familia_mdt", "mdt_origem", "cadeia_no_contrato", "crs", "impressao_unidade"):
        t[c] = t.unidade_terreno.map(u[c])
    t["nao_e"] = "caracterizacao do lugar sem rotulo; nao preve desempenho de modelo"
    return t


def materializar(saida: Path = SAIDA, unidades: list[str] | None = None, vizinhanca: bool = True,
                 progresso: bool = True, processos: int = 1) -> Path:
    from proveniencia import sha256_json
    f = fontes()
    acervo = ae.Acervo()
    ver = ae.verificar()
    if not ver["tabelas_integras"]:
        raise RuntimeError(f"indice do acervo divergente: {ver['tabelas_divergentes']}")
    pac = json.loads(Path(f["pacote"]).read_text(encoding="utf-8"))
    papeis = pac["fisico"]["papeis_por_evento"]
    criterios = {"versao": VERSAO, "curado": _sha(f["curado"]), "pacote": pac["hash_resumo"],
                 "acervo": acervo.man["hash_resumo"], "codigo_acervo": _sha(Path(ae.__file__)),
                 "codigo": _sha(Path(__file__)), "codigo_posicao": _sha(Path(pp.__file__)),
                 "encoder_normalizacao": _sha(f["encoder"]), "unidades": sorted(unidades) if unidades else "todas",
                 "vizinhanca": vizinhanca}
    destino = saida / VERSAO / sha256_json(criterios).split(":")[-1][:16]
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "criterios.json").write_text(json.dumps(criterios, ensure_ascii=False, indent=1), encoding="utf-8")
    inicio = time.time()
    base = carregar_base(f["curado"], unidades)
    # Contexto completo onde ha observacao. Linha sem observacao fica preservada com o estado e sem contexto
    # calculado nesta versao (custo: ~10 ms/ponto); a mesma funcao o calcula sob demanda.
    observado = base.estado.isin(OBSERVADOS).to_numpy()
    calc = ae.contextualizar(base.loc[observado, ["ponto_id", "lon", "lat"]].reset_index(drop=True), acervo,
                             vizinhanca=vizinhanca, em_memoria=True, checkpoint=destino / "parciais",
                             progresso=progresso, processos=processos)
    ctx = pd.DataFrame({"ponto_id": base.ponto_id}).merge(calc, on="ponto_id", how="left", validate="one_to_one")
    ctx["contexto_calculado"] = observado
    ctx.loc[~observado, "motivo_ausencia"] = "contexto_nao_calculado: estado sem observacao"
    eventos = tabela_de_eventos(f, sorted(base.evento_id.unique()))
    d = anotar(base, ctx, eventos, papeis)
    d["componente"] = componentes(d)
    # Um componente com eventos de papeis diferentes significa que um ponto recuperado juntou lugares
    # que o pacote mantinha separados: a linha fica marcada, nao reatribuida.
    papeis_por_comp = d[d.componente.notna()].groupby("componente").papel_pacote.agg(lambda s: tuple(sorted(set(s))))
    conflito = papeis_por_comp[papeis_por_comp.map(len) > 1]
    d["conflito_de_papel"] = d.componente.isin(conflito.index)
    arq = destino / "pontos_contexto.parquet"
    d.to_parquet(arq.with_suffix(".part"), index=False, compression="zstd")
    arq.with_suffix(".part").replace(arq)
    eventos.to_parquet(destino / "eventos.parquet", index=False)
    uc = unidades_de_contexto(d, acervo)
    uc.to_parquet(destino / "unidades_contexto.parquet", index=False)
    man = {"tipo": "contexto_de_desenvolvimento", "versao": VERSAO, "regime": "desenvolvimento",
           "promovido": False, "pacote_congelado_alterado": False, "criterios": criterios,
           "entradas": {str(Path(p).relative_to(REPO)): _sha(p) for p in f.values()},
           "saidas": {p.name: _sha(p) for p in (arq, destino / "eventos.parquet", destino / "unidades_contexto.parquet")},
           "linhas": int(len(d)), "perdas": perdas(d), "equivalencia_com_o_lote": equivalencia_com_o_lote(d),
           "duplicatas": duplicatas(d), "dependencia_entre_papeis": dependencia_entre_papeis(d),
           "componentes": {"n": int(d.componente.nunique()), "com_papeis_misturados": int(len(conflito)),
                           "linhas_em_conflito": int(d.conflito_de_papel.sum()),
                           "conflitos": {str(k): list(v) for k, v in conflito.items()}},
           "unidades_de_terreno_usadas": int(d.unidade_terreno.nunique()),
           "regra_de_unidade": "acervo_espacial: compativel > mais longe da borda > nome; primeira com nucleo finito",
           "vizinhanca": {"definicao": "N1b/V1/MS4: 128 px x 5 camadas, 25 estatisticas",
                          "diferencas": ["janela que sai do raster e desconhecida (N1: >= 50% valido bastava)",
                                         "float32 sem o arredondamento float16 do cache do N1"],
                          "normalizacao": str(Path(f["encoder"]).relative_to(REPO))},
           "unidade_hidrologica": "nao ligada: HydroBASINS nao esta no acervo (lacuna declarada)",
           "segundos": round(time.time() - inicio),
           "nao_e": ("nao e teste nem confirmacao; recuperar atributo nao cria observacao nem independencia; "
                     "o pacote congelado e o executor dele nao mudam")}
    man["hash_resumo"] = sha256_json({k: v for k, v in man.items() if k != "segundos"})
    man["quando_utc"] = ae._agora()
    (destino / "manifesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=ae._json_seguro),
                                            encoding="utf-8")
    return destino / "manifesto.json"


def carregar(manifesto: Path, vetor=VETOR, populacao: str = "pacote", colunas_extra=()) -> pd.DataFrame:
    """Vista de treino da versao contextualizada, no formato do pacote (ponto_id, evento_id, classe, papel, vetor).

    populacao="pacote"   so as linhas que o pacote congelado tinha (mesma populacao, valores relidos)
    populacao="ampliada" mais as recuperadas por unidade compativel; conflito de papel fica de fora
    Desconhecido nunca vira classe 0; papel e o do evento no pacote congelado.
    """
    man = json.loads(Path(manifesto).read_text(encoding="utf-8"))
    arq = Path(manifesto).parent / "pontos_contexto.parquet"
    if _sha(arq) != man["saidas"]["pontos_contexto.parquet"]:
        raise ValueError("versao contextualizada divergente do manifesto")
    cols = ["ponto_id", "evento_id", "estado", "classe", "papel_pacote", "vetor_finito_lote", "conflito_de_papel",
            "origem_do_vetor", "unidade_terreno", *vetor, *colunas_extra]
    d = pd.read_parquet(arq, columns=list(dict.fromkeys(cols)))
    d = d[d.estado.isin(OBSERVADOS) & np.isfinite(d[list(vetor)].to_numpy(float)).all(1)]
    d = d[d.papel_pacote.isin(("treino", "validacao", "avaliacao_desenvolvimento")) & ~d.conflito_de_papel]
    if populacao == "pacote":
        d = d[d.vetor_finito_lote]
    elif populacao != "ampliada":
        raise ValueError("populacao deve ser 'pacote' ou 'ampliada'")
    return d.rename(columns={"papel_pacote": "papel"}).assign(classe=lambda x: x.classe.astype("uint8"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--materializar", action="store_true")
    ap.add_argument("--piloto", nargs="+", metavar="UNIDADE_TC01")
    ap.add_argument("--sem-vizinhanca", action="store_true")
    ap.add_argument("--processos", type=int, default=1, help="unidades em paralelo; ~1 GB de RAM por processo")
    a = ap.parse_args()
    if not (a.materializar or a.piloto):
        ap.error("escolha --materializar ou --piloto")
    m = materializar(SAIDA / ("piloto" if a.piloto else ""), unidades=a.piloto, vizinhanca=not a.sem_vizinhanca,
                     processos=a.processos)
    r = json.loads(m.read_text(encoding="utf-8"))
    print(json.dumps({k: r[k] for k in ("linhas", "perdas", "componentes", "segundos")}, ensure_ascii=False,
                     default=ae._json_seguro)[:4000])
    print(f"GRAVADO={m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
