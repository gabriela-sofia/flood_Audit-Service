"""
DS-06 -- O dataset de treino: a tabela de pontos que ensina o que e inundacao.

POR QUE ESTE ARQUIVO EXISTE:

o ds05 consolida tudo o que foi reduzido, inclusive o que nao ensina nada
sobre inundacao -- ausencia de registro, agua permanente, movimento de massa,
pixel sem rotulo. Isso e correto para auditoria e errado para treino: cada
consumidor filtrava por conta propria, e em 02/10/2026 tres filtros
diferentes conviviam no projeto (ver `ds03.papel_evidencial`).

Este passo produz UMA tabela de pontos para aprendizado, em que cada linha:

  - ensina inundacao pelo contrato do fenomeno (positivo, negativo observado
    ou negativo por exclusao);
  - carrega a identidade do EVENTO que a produziu e a janela dele, quando a
    fonte tem evento catalogado -- e diz quando nao tem;
  - carrega o grupo de validacao e a unidade ESPACIAL de referencia;
  - carrega `elevation_local_m`, a elevacao contra o piso da propria unidade
    espacial.

POR QUE ELEVACAO LOCAL E NAO SO `elevation_rel_m`:

`elevation_rel_m` desconta o piso da FONTE. Para o GFD, que cobre o Brasil
inteiro, isso e um piso nacional, e no Norte a elevacao "relativa" passou a
medir em que bacia o ponto esta: o sinal saiu +0,35, contra a fisica. Contra o
piso do proprio tile o sinal vai a -1,58 (-0,84 ja condicionado ao HAND).
A inversao era de referencia, nao de fenomeno. `elevation_rel_m` continua na
tabela, inalterada, para quem precisa da leitura regional.

O QUE ESTE ARQUIVO NAO FAZ:

nao treina, nao escolhe vetor, nao separa treino e teste, e nao imputa nada.
Linha que sai, sai com motivo em `excluidos.parquet`. Solo e cobertura nao
existem no dado local (a coluna `worldcover` dos pontos GFD esta vazia): ficam
declarados como eixos ausentes, nunca preenchidos.

Uso:
    python scripts/suscetibilidade/ds06_dataset_de_treino.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("suscetibilidade", "metodo", "servico", "verificacao", "geo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

import ds03_esquema_alvo as ds03  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

VERSAO = "v7"  # v1 antes do lote; v2 lote; v3 escala x relevo, Curitiba MISTO; v4 cobertura e solo; v5 escala na admissao; v6 data da cena do CEMS; v7 zero do UFO sem chao visivel vira SEM_DADO
RUNS = REPO / "local_runs"
TABELA = RUNS / "ds-05-tabela-unica" / "tabela_unica_v2.parquet"
GFD = RUNS / "gfd-brasil"
LISTA_AMPLIADA = RUNS / "br-02-negativo-nacional" / "eventos_que_tocam_brasil.csv"
SAIDA = RUNS / "ds-06-dataset-de-treino"
DOC = REPO / "docs" / "DATASET_DE_TREINO.md"
COBERTURA = RUNS / "cob-01-cobertura-e-solo" / "cobertura_e_solo.parquet"
COLUNAS_DE_COBERTURA = ["mapbiomas_classe", "mapbiomas_ano", "modis_lc", "modis_ano",
                        "construido_frac", "ghsl_ano", "worldcover_2021",
                        "argila_pct_0_30", "areia_pct_0_30"]

TERRENO = ("hand_m", "twi_dinf", "slope_deg", "elevation_m")

# Unidade espacial em que o piso local de elevacao e medido. Onde o grupo de
# validacao ja e um lugar (tile, AOI, chip), ele e a unidade; onde o grupo e
# um dia ou um evento sem recorte proprio, a unidade e uma celula de grade.
UNIDADE_ESPACIAL = {"gfd": "grupo_cv", "cems": "grupo_cv", "sen1floods11": "aoi",
                    "ufo": "aoi"}
CELULA_GRAUS = 0.1
MINIMO_PONTOS_NO_PISO = 30
QUANTIL_DO_PISO = 0.01

# Eventos catalogados sao cheias identificadas por um produtor externo; o
# registro cotidiano e uma ocorrencia por dia de uma central ou defesa civil.
TIPO_DE_EVENTO = {"gfd": "evento_catalogado", "cems": "evento_catalogado",
                  "sen1floods11": "evento_catalogado", "uk": "evento_catalogado",
                  "ufo": "cena_datada", "curitiba": "registro_cotidiano",
                  "recife": "registro_cotidiano", "sao_paulo": "registro_cotidiano",
                  "nyc": "registro_cotidiano", "chicago": "registro_cotidiano"}

EIXOS_AUSENTES = {
    "grupo_hidrologico_do_solo": ("HYSOGs250m nao esta no catalogo do Earth Engine; a "
                                  "textura (SoilGrids) entra no lugar, declarada como textura"),
}
EIXOS_SEM_COLETA = {
    "solo": "rode scripts/geo/cob01_cobertura_e_solo.py (SoilGrids)",
    "cobertura_do_solo": "rode scripts/geo/cob01_cobertura_e_solo.py (MapBiomas, MODIS, WorldCover, GHSL)",
}


def versao_atual(base: Path = None) -> Path:
    """Diretorio da versao mais recente com manifesto: o que os consumidores leem."""
    base = base or SAIDA
    versoes = sorted((p.parent for p in base.glob("v*/manifesto.json")),
                     key=lambda d: int(d.name[1:]))
    if not versoes:
        raise FileNotFoundError(f"nenhum dataset construido em {base}")
    return versoes[-1]


def comparar(atual: dict, anterior: dict | None) -> dict | None:
    """Ganho de EVIDENCIA entre versoes: grupos e eventos, nao so linhas."""
    if not anterior:
        return None
    ctx = lambda m: {c["contexto"]: c for c in m["contextos"]}  # noqa: E731
    a, b = ctx(anterior), ctx(atual)
    mudancas = {}
    for nome in sorted(set(a) | set(b)):
        x, y = a.get(nome), b.get(nome)
        g0 = x["grupos_por_classe"] if x else {"0": 0, "1": 0}
        g1 = y["grupos_por_classe"] if y else {"0": 0, "1": 0}
        e0 = x["eventos_por_classe"] if x else {"0": 0, "1": 0}
        e1 = y["eventos_por_classe"] if y else {"0": 0, "1": 0}
        if g0 != g1 or e0 != e1:
            mudancas[nome] = {"grupos_antes": g0, "grupos_depois": g1,
                              "eventos_antes": e0, "eventos_depois": e1,
                              "linhas": [x["linhas"] if x else 0, y["linhas"] if y else 0],
                              "pronto": [bool(x and x["pronto_para_treino"]),
                                         bool(y and y["pronto_para_treino"])]}
    return {"versao_anterior": anterior.get("versao"),
            "linhas": [anterior["linhas"], atual["linhas"]],
            "eventos_catalogados": [anterior["eventos"]["por_tipo"].get("evento_catalogado", {}).get("eventos"),
                                    atual["eventos"]["por_tipo"].get("evento_catalogado", {}).get("eventos")],
            "contextos_prontos": [len(anterior["contextos_prontos_para_treino"]),
                                  len(atual["contextos_prontos_para_treino"])],
            "novos_prontos": sorted(set(atual["contextos_prontos_para_treino"])
                                    - set(anterior["contextos_prontos_para_treino"])),
            "contextos_que_mudaram": mudancas}


def hash_arquivo(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return "sha256:" + h.hexdigest()


def janelas_gfd(pontos: Path, catalogo: Path, lista: Path) -> pd.DataFrame:
    """ponto_id -> evento DFO e janela do asset, reconciliada com a lista ampliada."""
    import cortes_por_evento as cortes

    eventos = cortes.reconciliar(json.loads(catalogo.read_text(encoding="utf-8")),
                                 pd.read_csv(lista).to_dict("records"))
    g = pd.read_csv(pontos, usecols=["ponto_id", "evento_gfd"])
    g["evento_id"] = g.evento_gfd.map(lambda x: f"GFD:{int(x)}" if pd.notna(x) else None)
    g["janela_inicio"] = g.evento_id.map(lambda e: eventos.get(e, {}).get("inicio"))
    g["janela_fim"] = g.evento_id.map(lambda e: eventos.get(e, {}).get("fim"))
    return g.drop(columns="evento_gfd")


def identidade_de_evento(d: pd.DataFrame, gfd: pd.DataFrame | None) -> pd.DataFrame:
    """Evento e janela por linha, a partir do identificador que cada fonte ja traz."""
    ev = pd.Series([None] * len(d), index=d.index, dtype=object)
    f = d.fonte
    ev[f == "cems"] = d.loc[f == "cems", "aoi"].str.split("_").str[0].radd("CEMS:")
    ev[f == "sen1floods11"] = d.loc[f == "sen1floods11", "grupo_cv"].str.replace(
        "N1_", "", regex=False).radd("S1F11:")
    ev[f == "uk"] = d.loc[f == "uk", "grupo_cv"].radd("UK:")
    ev[f == "ufo"] = d.loc[f == "ufo", "aoi"].str.split("_").str[:2].str.join("_").radd("UFO:")
    data = pd.to_datetime(d.data_evento.astype("string"), format="mixed", errors="coerce", utc=True)
    dia = data.dt.strftime("%Y-%m-%d")
    out = pd.DataFrame({"evento_id": ev, "janela_inicio": dia, "janela_fim": dia}, index=d.index)
    if gfd is not None:
        m = d.loc[f == "gfd", ["ponto_id"]].merge(gfd, on="ponto_id", how="left",
                                                   validate="one_to_one")
        m.index = d.index[f == "gfd"]
        out.loc[m.index, ["evento_id", "janela_inicio", "janela_fim"]] = m[
            ["evento_id", "janela_inicio", "janela_fim"]].to_numpy()
    out["evento_tipo"] = f.map(TIPO_DE_EVENTO).fillna("nao_identificado")
    return out


def referencia_local(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(unidade_espacial, elevation_local_m). Unidade pequena demais fica NaN."""
    celula = (f"c{CELULA_GRAUS}_" + (d.lat / CELULA_GRAUS).round().astype("Int64").astype(str)
              + "_" + (d.lon / CELULA_GRAUS).round().astype("Int64").astype(str))
    unidade = celula.copy()
    for fonte, coluna in UNIDADE_ESPACIAL.items():
        m = d.fonte == fonte
        unidade[m] = d.loc[m, coluna].astype(str)
    unidade = d.fonte + "|" + unidade
    grupos = d.groupby(unidade).elevation_m
    piso = grupos.transform(lambda s: s.quantile(QUANTIL_DO_PISO))
    n = grupos.transform("size")
    local = (d.elevation_m - piso).where(n >= MINIMO_PONTOS_NO_PISO)
    return unidade, local


def construir(tabela: pd.DataFrame, gfd: pd.DataFrame | None,
              cobertura_e_solo: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import contrato_do_fenomeno as cf

    d = tabela.copy()
    d["papel"] = ds03.papeis_evidenciais(d)
    motivo = pd.Series([""] * len(d), index=d.index, dtype=object)

    def sair(mascara, nome):
        livre = mascara & (motivo == "")
        motivo[livre] = nome

    # O papel vem primeiro: "agua permanente" diz mais que "nao admitido".
    nao_ensina = ~d.papel.isin(ds03.PAPEIS_QUE_ENSINAM)
    motivo[nao_ensina] = "papel:" + d.loc[nao_ensina, "papel"]
    rejeitado = ~d.admitido.astype(bool) & (motivo == "")
    motivo[rejeitado] = "ds05:" + d.loc[rejeitado, "motivo_rejeicao"].fillna("sem_motivo")
    sair(~np.isfinite(d[list(TERRENO)].to_numpy(dtype=float)).all(axis=1), "terreno_nao_finito")
    sair(pd.Series([not cf.relevo_resolvido(f, r) for f, r in zip(d.fonte, d.classe_relevo)],
                   index=d.index), "escala:relevo_nao_resolvido_pelo_instrumento")
    excluidos = d.loc[motivo != "", ["ponto_id", "fonte", "classe", "nivel_negativo", "papel"]].assign(
        motivo=motivo[motivo != ""])

    p = d.loc[motivo == ""].copy()
    p = pd.concat([p, identidade_de_evento(p, gfd)], axis=1)
    p["unidade_espacial"], p["elevation_local_m"] = referencia_local(p)
    import cobertura_por_contexto as cobertura
    p["territorio"] = cobertura._territorio(p)
    p["mecanismo_justificado"] = p.fonte.map(
        lambda f: cf.INSTRUMENTOS.get(f, {}).get("mecanismo_justificado", False))
    p["contexto"] = p.territorio + "|" + p.mecanismo + "|" + p.classe_relevo
    p = p.rename(columns={"grupo_cv": "grupo_validacao", "nivel_negativo": "qualidade_negativo"})
    colunas = ["ponto_id", "fonte", "mecanismo", "mecanismo_justificado", "papel", "classe",
               "qualidade_negativo", "territorio", "classe_relevo", "contexto",
               "grupo_validacao", "unidade_agrupamento", "unidade_espacial",
               "evento_id", "evento_tipo", "janela_inicio", "janela_fim",
               "precisao_data", "lat", "lon", "resolucao_nativa_m", "procedencia",
               "hand_m", "twi_dinf", "slope_deg", "elevation_m", "elevation_rel_m",
               "elevation_local_m", "rain_max_24h", "rain_decay_index", "fonte_chuva"]
    p = p[colunas].sort_values("ponto_id").reset_index(drop=True)
    if cobertura_e_solo is not None:
        if cobertura_e_solo.ponto_id.duplicated().any():
            raise ValueError("cobertura com ponto_id repetido")
        p = p.merge(cobertura_e_solo[["ponto_id", *COLUNAS_DE_COBERTURA]], on="ponto_id",
                    how="left", validate="one_to_one")
    return p, excluidos.reset_index(drop=True), validar(p)


def validar(p: pd.DataFrame) -> dict:
    """Invariantes do dataset. Qualquer falha impede a gravacao."""
    import contrato_do_fenomeno as cf

    checagens = {
        "ponto_id_unico": not p.ponto_id.duplicated().any(),
        "so_papel_que_ensina": p.papel.isin(ds03.PAPEIS_QUE_ENSINAM).all(),
        "classe_binaria": p.classe.isin([0, 1]).all(),
        "negativo_nunca_ausencia": not ((p.classe == 0) & (p.qualidade_negativo == "ausencia")).any(),
        "terreno_finito": np.isfinite(p[list(TERRENO)].to_numpy(dtype=float)).all(),
        "grupo_de_validacao_presente": p.grupo_validacao.notna().all(),
        "mecanismo_do_instrumento": (p.mecanismo == p.fonte.map(
            lambda f: cf.INSTRUMENTOS[f]["mecanismo"])).all(),
        "evento_catalogado_tem_identidade": p.loc[
            p.evento_tipo == "evento_catalogado", "evento_id"].notna().all(),
        "gfd_tem_janela": p.loc[p.fonte == "gfd", "janela_inicio"].notna().all(),
    }
    return {k: bool(v) for k, v in checagens.items()}


def _grupos_por_classe(s: pd.DataFrame) -> dict:
    return {str(c): int(s.loc[s.classe == c, "grupo_validacao"].nunique()) for c in (0, 1)}


def manifesto(p: pd.DataFrame, excluidos: pd.DataFrame, checagens: dict, entradas: dict) -> dict:
    import contrato_do_fenomeno as cf

    contextos = []
    for nome, s in p.groupby("contexto"):
        g = _grupos_por_classe(s)
        contextos.append({
            "contexto": nome, "linhas": len(s), "grupos_por_classe": g,
            "eventos_por_classe": {str(c): int(s.loc[s.classe == c, "evento_id"].nunique())
                                   for c in (0, 1)},
            "fontes": sorted(s.fonte.unique().tolist()),
            "pronto_para_treino": min(g.values()) >= cf.MINIMO_GRUPOS,
        })
    negativos = p[p.classe == 0]
    return {
        "tipo": "dataset_de_treino", "versao": VERSAO, "entradas": entradas,
        "linhas": len(p), "excluidos": len(excluidos),
        "excluidos_por_motivo": {str(k): int(v) for k, v in excluidos.motivo.value_counts().items()},
        "por_mecanismo_e_papel": {
            f"{m}|{pa}": int(n) for (m, pa), n in p.groupby(["mecanismo", "papel"]).size().items()},
        "eventos": {
            "por_tipo": {t: {"linhas": len(s), "eventos": int(s.evento_id.nunique()),
                             "dias": int(s.janela_inicio.nunique())}
                         for t, s in p.groupby("evento_tipo")},
            "negativos_por_tipo_de_evento": {str(k): int(v) for k, v in
                                             negativos.evento_tipo.value_counts().items()},
            "negativos_em_evento_catalogado": int((negativos.evento_tipo == "evento_catalogado").sum()),
        },
        "temporalidade": {str(k): int(v) for k, v in p.precisao_data.value_counts().items()},
        "elevacao_local": {"unidades": int(p.unidade_espacial.nunique()),
                           "sem_piso_por_unidade_pequena": int(p.elevation_local_m.isna().sum()),
                           "quantil_do_piso": QUANTIL_DO_PISO,
                           "minimo_de_pontos": MINIMO_PONTOS_NO_PISO},
        "contextos": contextos,
        "contextos_prontos_para_treino": sorted(c["contexto"] for c in contextos
                                                if c["pronto_para_treino"]),
        "mecanismo_sem_justificativa": sorted(p.loc[~p.mecanismo_justificado, "fonte"].unique()),
        "eixos_ausentes": {**EIXOS_AUSENTES,
                           **({} if "construido_frac" in p else EIXOS_SEM_COLETA)},
        "cobertura_e_solo": ({c: round(float(p[c].notna().mean()), 4)
                              for c in COLUNAS_DE_COBERTURA if not c.endswith("_ano")}
                             if "construido_frac" in p else None),
        "checagens": checagens,
        "nao_e": ("nao e split de treino e teste, nao escolhe vetor e nao imputa: e a "
                  "tabela de pontos que ensina inundacao, com evento, grupo e procedencia"),
    }


def markdown(m: dict) -> str:
    n = lambda x: f"{x:,}".replace(",", ".")  # noqa: E731
    L = ["# Dataset de treino", "",
         f"**Gerado de** `scripts/suscetibilidade/ds06_dataset_de_treino.py` ({m['versao']}) em "
         f"{m['quando_utc']}. **Não editar à mão.**", "", f"**Não é.** {m['nao_e']}.", "",
         f"{n(m['linhas'])} pontos que ensinam inundação; {n(m['excluidos'])} linhas da tabela "
         "única ficaram fora, cada uma com motivo.", "",
         "| motivo de exclusão | linhas |", "|---|---:|"]
    L += [f"| `{k}` | {n(v)} |" for k, v in m["excluidos_por_motivo"].items()]
    L += ["", "## O que cada mecanismo tem", "", "| mecanismo · papel | linhas |", "|---|---:|"]
    L += [f"| {k.replace('|', ' · ')} | {n(v)} |" for k, v in m["por_mecanismo_e_papel"].items()]
    e = m["eventos"]
    L += ["", "## Eventos", "",
          "| tipo | linhas | eventos distintos | dias distintos |", "|---|---:|---:|---:|"]
    L += [f"| {k} | {n(v['linhas'])} | {n(v['eventos'])} | {n(v['dias'])} |"
          for k, v in e["por_tipo"].items()]
    L += ["", "Registro cotidiano não tem evento: é uma ocorrência por dia de uma central ou "
          "da defesa civil, e o dia é a sua unidade."]
    L += ["", f"Negativos em evento catalogado: **{n(e['negativos_em_evento_catalogado'])}** de "
          f"{n(sum(e['negativos_por_tipo_de_evento'].values()))}.", "",
          "## Contextos", "",
          "| contexto | linhas | grupos − / + | eventos − / + | fontes | pronto |",
          "|---|---:|---|---|---|---|"]
    for c in sorted(m["contextos"], key=lambda c: -c["linhas"]):
        g, ev = c["grupos_por_classe"], c["eventos_por_classe"]
        L.append(f"| {c['contexto']} | {n(c['linhas'])} | {g['0']} / {g['1']} | {ev['0']} / {ev['1']} | "
                 f"{', '.join(c['fontes'])} | {'sim' if c['pronto_para_treino'] else 'não'} |")
    L += ["", "## Temporalidade da data", "", "| precisão | linhas |", "|---|---:|"]
    L += [f"| `{k}` | {n(v)} |" for k, v in m["temporalidade"].items()]
    c = m.get("comparacao_com_versao_anterior")
    if c:
        L += ["", f"## Ganho em relação à {c['versao_anterior']}", "",
              f"Linhas {n(c['linhas'][0])} → {n(c['linhas'][1])}; eventos catalogados "
              f"{c['eventos_catalogados'][0]} → {c['eventos_catalogados'][1]}; contextos prontos "
              f"{c['contextos_prontos'][0]} → {c['contextos_prontos'][1]}.", ""]
        if c["novos_prontos"]:
            L += ["Passaram a ter grupos suficientes: " + ", ".join(f"`{x}`" for x in c["novos_prontos"]) + ".", ""]
        L += ["| contexto | grupos − / + antes | depois | eventos − / + antes | depois |",
              "|---|---|---|---|---|"]
        for nome, d in c["contextos_que_mudaram"].items():
            ga, gd, ea, ed = d["grupos_antes"], d["grupos_depois"], d["eventos_antes"], d["eventos_depois"]
            L.append(f"| {nome} | {ga['0']} / {ga['1']} | {gd['0']} / {gd['1']} | "
                     f"{ea['0']} / {ea['1']} | {ed['0']} / {ed['1']} |")
    if m.get("cobertura_e_solo"):
        L += ["", "## Cobertura e solo (fração preenchida)", "",
              "| camada | preenchido |", "|---|---:|"]
        L += [f"| `{k}` | {v:.1%} |".replace(".", ",") for k, v in m["cobertura_e_solo"].items()]
        L += ["", "MapBiomas só cobre o Brasil; SoilGrids mascara área urbana. Camada temporal "
              "lida no ano do evento; o ano usado fica em `*_ano`."]
    L += ["", "## Eixos ausentes", ""]
    L += [f"- **{k}**: {v}." for k, v in m["eixos_ausentes"].items()]
    L += ["", "## Checagens", ""]
    L += [f"- {'✔' if v else '✘'} `{k}`" for k, v in m["checagens"].items()]
    L += ["", f"`{m['hash_resumo']}`", ""]
    return "\n".join(L)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "metodo"))
    import rastreio_de_impacto as _ri
    _entrada, _saida = _ri.ganchos("dataset_de_treino", "ds06")
    _entrada()
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--saida", type=Path, default=SAIDA)
    ap.add_argument("--doc", type=Path, default=DOC)
    ap.add_argument("--reverificar", action="store_true",
                    help=("reconstroi em memoria e, se o conteudo for IDENTICO ao da versao vigente, "
                          "so recarimba no rastreio; se diferir, falha pedindo nova VERSAO"))
    args = ap.parse_args()
    pontos_gfd = GFD / "pontos_gfd_com_terreno.csv"
    entradas = {"tabela_unica": hash_arquivo(TABELA), "pontos_gfd": hash_arquivo(pontos_gfd),
                "catalogo_gfd": hash_arquivo(GFD / "catalogo.json"),
                "lista_ampliada": hash_arquivo(LISTA_AMPLIADA)}
    tabela = pd.read_parquet(TABELA)
    gfd = janelas_gfd(pontos_gfd, GFD / "catalogo.json", LISTA_AMPLIADA)
    if hash_arquivo(pontos_gfd) != entradas["pontos_gfd"]:
        raise RuntimeError("pontos GFD mudaram durante a leitura: rode de novo")
    cobertura = pd.read_parquet(COBERTURA) if COBERTURA.exists() else None
    if cobertura is not None:
        entradas["cobertura_e_solo"] = hash_arquivo(COBERTURA)
    p, excluidos, checagens = construir(tabela, gfd, cobertura)
    m = manifesto(p, excluidos, checagens, entradas)
    if args.reverificar:
        vigente = pd.read_parquet(versao_atual(args.saida) / "pontos.parquet")
        igual = (list(vigente.columns) == list(p.columns) and len(vigente) == len(p)
                 and pd.util.hash_pandas_object(vigente, index=False).equals(
                     pd.util.hash_pandas_object(p.reset_index(drop=True), index=False)))
        if not igual:
            raise SystemExit("conteudo difere da versao vigente: suba VERSAO e reconstrua")
        print(f"reverificado: {versao_atual(args.saida).name} identico ao reconstruido")
        _saida()
        return 0
    try:
        anterior = json.loads((versao_atual(args.saida) / "manifesto.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        anterior = None
    if anterior and anterior.get("versao") == VERSAO:
        raise SystemExit(f"{VERSAO} ja existe: suba VERSAO para nao sobrescrever a rodada anterior")
    m["comparacao_com_versao_anterior"] = comparar(m, anterior)
    if not all(checagens.values()):
        falhas = [k for k, v in checagens.items() if not v]
        raise SystemExit(f"dataset reprovado nas checagens: {falhas}")
    destino = args.saida / VERSAO
    destino.mkdir(parents=True, exist_ok=True)
    p.to_parquet(destino / "pontos.parquet", index=False)
    excluidos.to_parquet(destino / "excluidos.parquet", index=False)
    m["saidas"] = {"pontos": hash_arquivo(destino / "pontos.parquet"),
                   "excluidos": hash_arquivo(destino / "excluidos.parquet")}
    m["hash_resumo"] = sha256_json(m)
    m["quando_utc"] = agora_utc()
    (destino / "manifesto.json").write_text(json.dumps(m, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    args.doc.write_text(markdown(m), encoding="utf-8")
    print(json.dumps({"linhas": m["linhas"], "prontos": m["contextos_prontos_para_treino"],
                      "checagens": m["checagens"]}, ensure_ascii=False, indent=1))
    _saida()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
