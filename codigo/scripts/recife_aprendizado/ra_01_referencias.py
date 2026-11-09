"""Referencia de alagamento urbano: relatos SEDEC de agua geocodificados ate a RUA dentro do BAIRRO.

Unidade = objeto rua-bairro: segmentos oficiais de um logradouro (CLOGRACODI) recortados
pelo poligono do bairro. Presenca = ao menos um relato administrativo de agua da SEDEC
(ocorrencia "Alagamentos"/"Imoveis alagados", 2014-2024) cujo endereco casa com o nome
da rua no bairro do relato. Fundo = demais ruas dos mesmos bairros: NAO rotulado, nunca seco.

Populacao: todas as ruas dos bairros com ao menos um relato geocodificado (cidade). Ruas com
>= 90% do comprimento dentro das 8 folhas PE3D formam o subconjunto do bloco PE3D. A precisao e de
rua no bairro: o relato nao e espalhado por pixels nem por bueiros.

Particao: bairros inteiros sorteados (semente fixa) entre atualizacao e teste, equilibrando
o numero de ruas com relato. A rede de drenagem atravessa bairros: dependencia declarada.
"""
from __future__ import annotations

import csv
import difflib
import glob
import io
import re
import unicodedata

import numpy as np
import pandas as pd

from ra_comum import END, EPSG, ESTUDO, OUT, PE3D_ZIPS, SEDEC_RAW, SEMENTE, V3R, Etapa, gravar_json, rel, sha256

REF = OUT / "referencias"
TIPOS = {"R": "RUA", "R.": "RUA", "AV": "AVENIDA", "AV.": "AVENIDA", "TV": "TRAVESSA", "TV.": "TRAVESSA",
         "TR": "TRAVESSA", "EST": "ESTRADA", "PC": "PRACA", "PCA": "PRACA", "VL": "VILA", "CJ": "CONJUNTO",
         "LD": "LADEIRA", "CRG": "CORREGO", "1TR": "1A TRAVESSA", "2TR": "2A TRAVESSA"}
PREFIXOS = ("RUA", "AVENIDA", "TRAVESSA", "ESTRADA", "PRACA", "VILA", "CONJUNTO", "LADEIRA", "CORREGO",
            "BECO", "ALAMEDA", "LARGO", "CAIS", "CANAL", "1A TRAVESSA", "2A TRAVESSA", "3A TRAVESSA",
            "4A TRAVESSA", "5A TRAVESSA", "VIA", "RODOVIA", "PRIMEIRA TRAVESSA", "SEGUNDA TRAVESSA")
ABREV = {"PROF": "PROFESSOR", "PROFA": "PROFESSORA", "DEP": "DEPUTADO", "CEL": "CORONEL", "MIN": "MINISTRO",
         "MAL": "MARECHAL", "DR": "DOUTOR", "DRA": "DOUTORA", "PE": "PADRE", "STA": "SANTA", "STO": "SANTO",
         "GAL": "GENERAL", "GEN": "GENERAL", "ENG": "ENGENHEIRO", "DES": "DESEMBARGADOR", "PRES": "PRESIDENTE",
         "GOV": "GOVERNADOR", "SEN": "SENADOR", "CAP": "CAPITAO", "TEN": "TENENTE", "SGT": "SARGENTO",
         "MONS": "MONSENHOR", "VER": "VEREADOR", "N": "NOSSA", "SRA": "SENHORA", "S": "SAO"}
PARTICULAS = {"DA", "DE", "DO", "DAS", "DOS", "E"}
CORTES = (" COMUNIDADE ", " COM ", " ESQUINA ", " EM BAIXO ", " EMBAIXO ", " PROXIMO ", " PROX ", " LOTEAMENTO ",
          " ESCOLA ", " COLEGIO ", " ENTRE ", " NO ", " NA ", " APOS ", " ATRAS ")
LIMIAR_FUZZY = 0.90
FRAC_COBERTURA = 0.90


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = [ABREV.get(t, TIPOS.get(t, t)) for t in s.split()]
    s = " ".join(toks)
    s = re.sub(r"^(\d+A TRAVESSA) (DA )?RUA (E )?", r"\1 ", s)     # "1A TRAVESSA DA RUA X" -> "1A TRAVESSA X"
    for c in CORTES:                                                  # descarta o que vem depois de um marcador
        i = s.find(c)
        if i > 0:
            s = s[:i]
    return s.strip()


def nucleo(s: str) -> frozenset:
    """Tokens do nome sem tipo e sem particulas, para casamento por contencao."""
    return frozenset(t for t in sem_tipo(s).split() if t not in PARTICULAS)


def sem_tipo(s: str) -> str:
    for p in sorted(PREFIXOS, key=len, reverse=True):
        if s.startswith(p + " "):
            return s[len(p) + 1:]
    return s


def norm_bairro(s: str) -> str:
    return norm(s).replace(" DE ", " ").replace(" DO ", " ").replace(" DA ", " ").replace(" DOS ", " ").replace(" DAS ", " ")


def relatos():
    r = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(V3R / "registros_*.parquet")))])
    a = r[r.categoria.eq("relato_administrativo_agua")].copy()
    linhas = []
    for arq, g in a.groupby("arquivo"):
        bruto = (SEDEC_RAW / arq).read_bytes()
        try:
            txt = bruto.decode("utf-8")
        except UnicodeDecodeError:
            txt = bruto.decode("latin-1")
        rows = list(csv.reader(io.StringIO(txt), delimiter=";"))
        ie = [i for i, h in enumerate(rows[0]) if "ndere" in h][0]
        for _, x in g.iterrows():
            row = rows[int(x.linha_csv) - 1]
            linhas.append({"registro_sha256": x.registro_sha256, "arquivo": arq, "linha_csv": int(x.linha_csv),
                           "ano": int(x.ano_arquivo), "data_registro": x.data_registro, "data_acao": x.data_acao,
                           "ocorrencia": x.ocorrencia, "solicitacao": x.solicitacao, "bairro": x.bairro,
                           "localidade": x.localidade, "duplicado_conteudo": bool(x.duplicado_conteudo),
                           "endereco_bruto": row[ie] if len(row) > ie else None})
    d = pd.DataFrame(linhas)
    d["rua_texto"] = d.endereco_bruto.fillna("").str.split(",").str[0].str.strip()
    d["rua_norm"] = d.rua_texto.map(norm)
    d["e_canal"] = d.rua_norm.str.startswith("CANAL")
    return d


def folhas_pe3d():
    import rasterio
    from shapely.geometry import box
    out = []
    for z in sorted(PE3D_ZIPS.glob("*.zip")):
        tif = z.stem + ".tif"
        with rasterio.open(f"/vsizip/{z.as_posix()}/{tif}") as r:
            out.append({"folha": z.stem, "zip": rel(z), "sha256_zip": sha256(z), "crs": str(r.crs),
                        "res": r.res, "forma": [r.height, r.width], "nodata": r.nodata,
                        "geometry": box(*r.bounds)})
    return out


def main() -> int:
    import geopandas as gpd
    from shapely.ops import unary_union

    e = Etapa("ra_01_referencias")
    REF.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)
    fol = folhas_pe3d()
    fg = gpd.GeoDataFrame(fol, crs=EPSG)
    cobertura = unary_union(list(fg.geometry))
    fg.drop(columns="geometry").to_csv(REF / "folhas_pe3d.csv", index=False)

    bairros = gpd.read_file(END / "bairros_do_recife.gpkg").to_crs(EPSG)
    bairros["bnorm"] = bairros.EBAIRRNOME.map(norm_bairro)
    ruas = gpd.read_file(END / "trechos_de_logradouros.gpkg").to_crs(EPSG)
    ruas["rnorm"] = ruas.NLOGRACONC.map(norm)
    ruas["rsem"] = ruas.rnorm.map(sem_tipo)
    e.entrada(END / "bairros_do_recife.gpkg", END / "trechos_de_logradouros.gpkg")

    # objetos rua-bairro: segmentos do logradouro recortados por cada bairro
    inter = gpd.overlay(ruas[["CLOGRACODI", "NLOGRACONC", "rnorm", "rsem", "geometry"]],
                        bairros[["CBAIRRCODI", "EBAIRRNOME", "bnorm", "geometry"]], how="intersection", keep_geom_type=True)
    obj = inter.dissolve(by=["CLOGRACODI", "CBAIRRCODI"], aggfunc="first").reset_index()
    obj["comprimento_m"] = obj.length
    obj = obj[obj.comprimento_m >= 20].copy()
    obj["frac_pe3d"] = obj.geometry.intersection(cobertura).length / obj.comprimento_m
    obj["id_obj"] = "RB_" + obj.CLOGRACODI.astype(int).astype(str) + "_" + obj.CBAIRRCODI.astype(int).astype(str)

    # geocodificacao dos relatos
    rel_ = relatos()
    mapa_b = {b: c for b, c in zip(bairros.bnorm, bairros.CBAIRRCODI)}
    rel_["bnorm"] = rel_.bairro.map(norm_bairro)
    rel_["CBAIRRCODI"] = rel_.bnorm.map(mapa_b)
    casados = []
    for i, x in rel_.iterrows():
        estado, alvo, metodo, escore = "nao_geocodificado", None, None, None
        if pd.isna(x.CBAIRRCODI):
            estado = "bairro_nao_casado"
        elif x.e_canal:
            estado = "canal_nao_e_rua"
        elif not x.rua_norm:
            estado = "sem_endereco"
        else:
            cand = obj[obj.CBAIRRCODI == x.CBAIRRCODI]
            exato = cand[cand.rnorm == x.rua_norm]
            if len(exato) == 0:
                exato = cand[cand.rsem == sem_tipo(x.rua_norm)]
                metodo = "nome_sem_tipo" if len(exato) else None
            else:
                metodo = "nome_exato"
            if len(exato) == 0 and len(cand):
                nr = nucleo(x.rua_norm)
                if len(nr) >= 1:
                    cont = cand[cand.rnorm.map(lambda s: nr <= nucleo(s))]
                    if cont.CLOGRACODI.nunique() == 1:
                        exato, metodo = cont.iloc[[0]], "contido_unico"
            if len(exato) == 0 and len(cand):
                sc = cand.rnorm.map(lambda s: difflib.SequenceMatcher(None, s, x.rua_norm).ratio())
                if sc.max() >= LIMIAR_FUZZY and (sc >= LIMIAR_FUZZY).sum() == 1:
                    exato, metodo, escore = cand[sc == sc.max()], "aproximado_unico", float(sc.max())
            if len(exato) == 1:
                estado, alvo = "geocodificado_rua_bairro", exato.id_obj.iloc[0]
            elif len(exato) > 1:
                estado = "ambiguo_varios_logradouros"
        casados.append({"estado_geocodigo": estado, "id_obj": alvo, "metodo": metodo, "escore_fuzzy": escore})
    rel_ = pd.concat([rel_.reset_index(drop=True), pd.DataFrame(casados)], axis=1)
    rel_.to_parquet(REF / "relatos_sedec_geocodificados.parquet", index=False)

    # populacao: ruas na cobertura PE3D, bairros com relato geocodificado na cobertura
    obj["em_pe3d"] = obj.frac_pe3d >= FRAC_COBERTURA
    obj_cob = obj[obj.em_pe3d]
    geo = rel_[rel_.estado_geocodigo.eq("geocodificado_rua_bairro")]
    geo_cob = geo[geo.id_obj.isin(obj_cob.id_obj)]
    bairros_pop = sorted(geo.CBAIRRCODI.unique())
    pop = obj[obj.CBAIRRCODI.isin(bairros_pop)].copy()
    cont = geo.groupby("id_obj").agg(n_relatos=("registro_sha256", "size"),
                                         n_relatos_sem_duplicado=("duplicado_conteudo", lambda s: int((~s).sum())),
                                         anos=("ano", lambda s: ",".join(map(str, sorted(set(s))))),
                                         n_2022=("ano", lambda s: int((s == 2022).sum())))
    pop = pop.merge(cont, left_on="id_obj", right_index=True, how="left")
    pop["n_relatos"] = pop.n_relatos.fillna(0).astype(int)
    pop["y"] = (pop.n_relatos > 0).astype(int)

    # particao por bairro
    pb = pop.groupby("CBAIRRCODI").y.sum().sort_values(ascending=False)
    ordem = list(pb.index)
    rng.shuffle(ordem)
    ordem = sorted(ordem, key=lambda b: -pb[b])          # gulosa por positivos, desempate pelo sorteio
    papel, soma = {}, {"atualizacao": 0, "teste": 0}
    for b in ordem:
        alvo = "atualizacao" if soma["atualizacao"] <= soma["teste"] else "teste"
        papel[b] = alvo
        soma[alvo] += pb[b]
    pop["papel"] = pop.CBAIRRCODI.map(papel)
    pop = gpd.GeoDataFrame(pop, geometry="geometry", crs=EPSG)
    pop[["id_obj", "CLOGRACODI", "NLOGRACONC", "CBAIRRCODI", "EBAIRRNOME", "comprimento_m", "frac_pe3d", "em_pe3d",
         "n_relatos", "n_relatos_sem_duplicado", "anos", "n_2022", "y", "papel", "geometry"]].to_file(
        REF / "objetos_rua_bairro.gpkg", driver="GPKG")

    ledger = pd.DataFrame([
        {"fonte": "SEDEC registro de atendimentos (relatos de agua)", "arquivo": rel(SEDEC_RAW), "n": int(len(rel_)),
         "natureza": "registro administrativo de presenca de agua (alagamento/imovel alagado), data do registro, sem hora",
         "localizacao": "nome de rua (numero mascarado) + bairro", "papel": "presenca (rua-bairro)",
         "nao_e": "ausencia de chamado nao e rua seca; nao e profundidade nem extensao"},
        {"fonte": "Base oficial de logradouros e bairros (ESIG)", "arquivo": rel(END), "n": int(len(ruas)),
         "natureza": "geometria de referencia", "localizacao": "linha de logradouro", "papel": "unidade e geocodificador", "nao_e": "observacao de agua"},
        {"fonte": "PE3D MDT 1 m (8 folhas)", "arquivo": rel(PE3D_ZIPS), "n": len(fol), "natureza": "terreno nativo",
         "localizacao": "raster 1 m EPSG:31985", "papel": "bloco PE3D", "nao_e": "agua; datum vertical a reconciliar"},
        {"fonte": "EMLURB microdrenagem (elementos, rede, canaleta)", "arquivo": rel(OUT.parents[2] / "PROJETO"), "n": 175630,
         "natureza": "cadastro tecnico (fotografia atual)", "localizacao": "pontos/linhas", "papel": "bloco drenagem",
         "nao_e": "capacidade, estado ou funcionamento; densidade pode refletir resposta a problemas anteriores"},
        {"fonte": "EMLURB macrodrenagem (canais, rios)", "arquivo": "PROJETO/data/processed/recife/esig_emlurb_macrodrenagem", "n": 111,
         "natureza": "cadastro tecnico", "localizacao": "linhas", "papel": "bloco saida (topografico)", "nao_e": "nivel observado"},
        {"fonte": "Sentinel-2 E2 (v6) 01/06/2022", "arquivo": "local_runs/transferencia-vulnerabilizados/v6_evidencia_robusta/recife_avaliacao/referencia.parquet",
         "n": 448, "natureza": "agua persistente 4 dias apos o evento", "localizacao": "celula da grade v6", "papel": "nao usado como rotulo de alagamento de rua",
         "nao_e": "rua seca no pico (negativo orbital nao identifica alagamento rapido)"},
        {"fonte": "Silva Junior et al. 2017 (Soledade)", "arquivo": rel(ESTUDO / "referencia_publicada_soledade.csv"), "n": 5,
         "natureza": "alturas in loco 2013/2014 em um ponto; candidato documental", "localizacao": "cruzamento", "papel": "caso tecnico, nao rotulo de treino",
         "nao_e": "cinco sitios ou eventos independentes"},
        {"fonte": "EMLURB 156 (2012-2026)", "arquivo": "nao baixado", "n": None, "natureza": "pedidos de servico (manutencao)",
         "localizacao": "rua (lat/lon por logradouro a partir de 2019)", "papel": "nao usado: pedido/servico nao e observacao de agua",
         "nao_e": "rotulo fisico"},
        {"fonte": "Chuva subdiaria APAC/Cemaden/INMET; mare/nivel", "arquivo": "nao adquirido", "n": None,
         "natureza": "forcamento e saida", "localizacao": "estacao", "papel": "nao usado: tarefa estatica multi-ano; Cemaden com CAPTCHA",
         "nao_e": "-"},
    ])
    ledger.to_csv(REF / "ledger_fontes.csv", index=False, encoding="utf-8-sig")

    resumo = {
        "relatos_total": int(len(rel_)), "relatos_por_estado": rel_.estado_geocodigo.value_counts().to_dict(),
        "geocodificados_por_metodo": geo.metodo.value_counts().to_dict(),
        "relatos_geocodificados_na_cobertura": int(len(geo_cob)),
        "objetos_total_cidade": int(len(obj)), "objetos_na_cobertura": int(len(obj_cob)),
        "bairros_populacao": len(bairros_pop), "objetos_populacao": int(len(pop)),
        "objetos_positivos": int(pop.y.sum()), "prevalencia_objetos": float(pop.y.mean()),
        "objetos_populacao_em_pe3d": int(pop.em_pe3d.sum()), "positivos_em_pe3d": int(pop[pop.em_pe3d].y.sum()),
        "relatos_geocodificados_total": int(len(geo)),
        "por_papel": pop.groupby("papel").y.agg(["sum", "count"]).to_dict(),
        "bairros_por_papel": pd.Series(papel).value_counts().to_dict(),
        "anos_relatos_cobertura": geo_cob.ano.value_counts().sort_index().to_dict(),
        "cobertura_pe3d_km2": float(cobertura.area / 1e6),
        "parametros": {"limiar_fuzzy": LIMIAR_FUZZY, "frac_cobertura": FRAC_COBERTURA, "semente": SEMENTE},
        "hashes": {p.name: sha256(p) for p in sorted(REF.glob("*")) if p.suffix in (".parquet", ".gpkg", ".csv")},
    }
    gravar_json(REF / "referencias.json", resumo)
    e.saida(*REF.glob("*"))
    e.fechar(**{k: v for k, v in resumo.items() if k != "hashes"})
    import json
    print(json.dumps({k: v for k, v in resumo.items() if k != "hashes"}, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
