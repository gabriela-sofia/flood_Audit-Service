"""Etapas 1-2: universo de territorios vulnerabilizados e escassez de observacao, antes de qualquer resultado.

Definicoes fixadas aqui, antes de olhar predicao ou referencia de avaliacao:
- FCU: setor com CD_TIPO == '1' (Favela e Comunidade Urbana, Censo 2022); a comunidade e o CD_FCU.
  E categoria territorial, nao mecanismo hidrologico nem definicao de toda vulnerabilidade.
- baixa_renda: setor urbano nao especial (CD_TIPO == '0') com mediana do rendimento do responsavel
  com rendimento (V06006) <= R$1.212 (salario minimo de 2022). Indicador unico e publico; nao
  classifica pessoas. Setor sem V06006 fica 'sem_dado', nunca e imputado.
- comparador: setor urbano nao especial com V06006 > R$1.212.
Outros tipos especiais (indigena, quilombola, etc.) sao contados no universo e ficam fora da
avaliacao fisica urbana, com motivo registrado.

Escassez separa: falta de dado para treinar, para avaliar, de insumo fisico e de registro
administrativo. Ausencia de registro nao e ausencia de inundacao.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum_tv import (AGREGADO_BASICO, AGREGADO_RENDA, OUT, RUNS, SALARIO_MINIMO_2022, SETORES, agora, gravar,
                      rel, sha)

TIPOS = {"0": "nao_especial", "1": "favela_comunidade_urbana"}
FONTE_TIPO = {"gfd": "fisica_orbital_modis_250m", "cems": "fisica_orbital_delineada_analista",
              "sen1floods11": "fisica_orbital_rotulo_manual", "recife": "administrativa_sedec",
              "curitiba": "administrativa_156", "sao_paulo": "administrativa_sp156"}
MODELO_LOCAL = {  # municipio -> situacao do modelo ajustado com dado do proprio lugar (svc-01-modelos)
    "2611606": "ajustado_local_registro_administrativo_recife_269_pontos",
    "3550308": "ajustado_local_retirado_rotulo_sp156_reprovado",
}


def ler_csv_zip(p: Path, colunas: list[str]) -> pd.DataFrame:
    z = zipfile.ZipFile(p)
    nome = [n for n in z.namelist() if n.endswith(".csv")][0]
    with z.open(nome) as f:
        return pd.read_csv(io.TextIOWrapper(f, encoding="latin1"), sep=";", usecols=colunas, dtype=str)


def universo_nacional() -> pd.DataFrame:
    b = ler_csv_zip(AGREGADO_BASICO, ["CD_SETOR", "SITUACAO", "CD_TIPO", "AREA_KM2", "CD_UF", "NM_UF", "CD_MUN",
                                      "NM_MUN", "CD_FCU", "NM_FCU", "CD_CONCURB", "NM_CONCURB", "v0001", "v0007"])
    r = ler_csv_zip(AGREGADO_RENDA, ["CD_SETOR", "V06006"])
    b = b.merge(r, on="CD_SETOR", how="left")
    b["pessoas"] = pd.to_numeric(b.v0001, errors="coerce")
    b["dom_ocupados"] = pd.to_numeric(b.v0007, errors="coerce")
    b["area_km2"] = pd.to_numeric(b.AREA_KM2.str.replace(",", ".", regex=False), errors="coerce")
    b["renda_resp"] = pd.to_numeric(b.V06006.str.replace(",", ".", regex=False), errors="coerce")
    b["classe_territorial"] = classe_territorial(b.CD_TIPO, b.SITUACAO, b.renda_resp)
    return b.drop(columns=["v0001", "v0007", "AREA_KM2", "V06006"])


def classe_territorial(tipo: pd.Series, situacao: pd.Series, renda: pd.Series) -> pd.Series:
    urbano = situacao.astype(str).str.lower().str.startswith("urbana")
    out = pd.Series("outro_tipo_especial", index=tipo.index, dtype=object)
    out[tipo.eq("1")] = "fcu"
    nao_esp = tipo.eq("0")
    out[nao_esp & ~urbano] = "rural_nao_especial"
    out[nao_esp & urbano & renda.isna()] = "urbano_sem_renda"
    out[nao_esp & urbano & (renda <= SALARIO_MINIMO_2022)] = "baixa_renda"
    out[nao_esp & urbano & (renda > SALARIO_MINIMO_2022)] = "comparador_urbano"
    return out


def ler_setores_uf(uf: str):
    import geopandas as gpd
    p = SETORES[uf]
    shp = [n for n in zipfile.ZipFile(p).namelist() if n.endswith(".shp")][0]
    g = gpd.read_file(f"/vsizip/{p.as_posix()}/{shp}", columns=["CD_SETOR"])
    return g


def main():
    import geopandas as gpd
    t0 = agora()
    nac = universo_nacional()
    fcu_set = nac[nac.classe_territorial.eq("fcu")]
    fcu = (fcu_set.groupby("CD_FCU")
           .agg(NM_FCU=("NM_FCU", "first"), CD_UF=("CD_UF", "first"), NM_UF=("NM_UF", "first"),
                CD_MUN=("CD_MUN", "first"), NM_MUN=("NM_MUN", "first"), municipios=("CD_MUN", "nunique"),
                setores=("CD_SETOR", "size"), pessoas=("pessoas", "sum"), dom_ocupados=("dom_ocupados", "sum"),
                area_km2=("area_km2", "sum"), NM_CONCURB=("NM_CONCURB", "first"))
           .reset_index())
    ufs_sigla = {"41": "PR", "43": "RS", "13": "AM", "26": "PE", "33": "RJ", "35": "SP", "11": "RO"}
    fcu["geometria_local"] = fcu.CD_UF.isin(ufs_sigla)
    gravar(OUT / "universo/fcu_nacional.csv", fcu)
    resumo_uf = (nac.groupby(["CD_UF", "NM_UF", "classe_territorial"]).agg(setores=("CD_SETOR", "size"),
                 pessoas=("pessoas", "sum")).reset_index())
    gravar(OUT / "universo/classes_por_uf.csv", resumo_uf)

    # ---- geometria nas UFs com malha local, referencias do acervo e insumo fisico
    tab = pd.read_parquet(RUNS / "ds-05-tabela-unica/tabela_unica_v2.parquet",
                          columns=["fonte", "aoi", "classe_nome", "lat", "lon", "data_evento", "admitido"])
    tab = tab[tab.lon.between(-75, -34) & tab.lat.between(-34, 6) & tab.fonte.isin(FONTE_TIPO)]
    pts = gpd.GeoDataFrame(tab, geometry=gpd.points_from_xy(tab.lon, tab.lat), crs=4326)
    unidades = pd.read_parquet(RUNS / "acervo-espacial/unidades.parquet",
                               columns=["unidade", "tipo", "bbox", "compativel", "familia_mdt"])
    unidades = unidades[unidades.compativel.astype(bool) & unidades.familia_mdt.eq("fabdem_average")]
    caixas = np.array([[float(v) for v in (json.loads(b) if isinstance(b, str) else b)] for b in unidades.bbox])
    cems = []
    for shp in sorted((RUNS / "br-01-cems-negativo/extraido").glob("*/*_areaOfInterestA_*.shp")):
        cems.append(gpd.read_file(shp).to_crs(4326).union_all())
    cems_aoi = gpd.GeoSeries(cems, crs=4326).union_all() if cems else None
    linhas, por_setor = [], []
    for uf, cod in [(v, k) for k, v in ufs_sigla.items()]:
        g = ler_setores_uf(uf).merge(nac[["CD_SETOR", "CD_FCU", "CD_MUN", "NM_MUN", "classe_territorial", "pessoas",
                                          "renda_resp", "NM_CONCURB"]], on="CD_SETOR", how="left")
        g = g.to_crs(4326)
        g["classe_territorial"] = g.classe_territorial.fillna("sem_agregado")
        j = gpd.sjoin(pts, g[["CD_SETOR", "geometry"]], predicate="within", how="inner")
        cont = (j.assign(tipo_ref=j.fonte.map(FONTE_TIPO), pos=j.classe_nome.eq("INUNDACAO"),
                         neg=j.classe_nome.eq("NAO_INUNDADO"))
                .groupby("CD_SETOR").agg(pontos=("fonte", "size"), pos=("pos", "sum"), neg=("neg", "sum"),
                                         fisica=("tipo_ref", lambda s: s.str.startswith("fisica").sum()),
                                         administrativa=("tipo_ref", lambda s: s.str.startswith("administrativa").sum()),
                                         fontes=("fonte", lambda s: ";".join(sorted(set(s))))))
        jf = j[j.fonte.map(FONTE_TIPO).str.startswith("fisica")]
        cont["fisica_pos"] = jf[jf.classe_nome.eq("INUNDACAO")].groupby("CD_SETOR").size()
        cont["fisica_neg"] = jf[jf.classe_nome.eq("NAO_INUNDADO")].groupby("CD_SETOR").size()
        g = g.merge(cont.reset_index(), on="CD_SETOR", how="left")
        for c in ["pontos", "pos", "neg", "fisica", "fisica_pos", "fisica_neg", "administrativa"]:
            g[c] = g[c].fillna(0).astype(int)
        c = g.geometry.representative_point()
        dentro = (caixas[:, 0][None, :] <= c.x.to_numpy()[:, None]) & (c.x.to_numpy()[:, None] <= caixas[:, 2][None, :]) \
            & (caixas[:, 1][None, :] <= c.y.to_numpy()[:, None]) & (c.y.to_numpy()[:, None] <= caixas[:, 3][None, :])
        g["terreno_fabdem"] = dentro.any(axis=1)
        g["cems_aoi"] = g.intersects(cems_aoi) if cems_aoi is not None else False
        g["lon"], g["lat"] = c.x, c.y
        g["UF"] = uf
        por_setor.append(pd.DataFrame(g.drop(columns="geometry")))
        f = g[g.classe_territorial.eq("fcu")]
        if len(f):
            agg = f.groupby("CD_FCU").agg(UF=("UF", "first"), CD_MUN=("CD_MUN", "first"), NM_MUN=("NM_MUN", "first"),
                                          pessoas=("pessoas", "sum"), setores=("CD_SETOR", "size"),
                                          pontos=("pontos", "sum"), fisica=("fisica", "sum"),
                                          fisica_pos=("fisica_pos", "sum"), fisica_neg=("fisica_neg", "sum"),
                                          administrativa=("administrativa", "sum"),
                                          terreno_fabdem=("terreno_fabdem", "all"), cems_aoi=("cems_aoi", "any"),
                                          lon=("lon", "mean"), lat=("lat", "mean"))
            linhas.append(agg.reset_index())
        print(uf, len(g), "setores;", len(f), "setores FCU", flush=True)
    setores = pd.concat(por_setor, ignore_index=True)
    gravar(OUT / "universo/setores_geometria_local.parquet", setores)
    led = pd.concat(linhas, ignore_index=True).merge(fcu[["CD_FCU", "NM_FCU", "area_km2", "NM_CONCURB"]], on="CD_FCU", how="left")
    led["modelo_local"] = led.CD_MUN.map(MODELO_LOCAL).fillna("inexistente")
    led["falta_dado_treinar"] = led.fisica_pos.eq(0)
    led["falta_dado_avaliar"] = (led.fisica_pos.eq(0) | led.fisica_neg.eq(0))
    led["falta_insumo_fisico"] = ~led.terreno_fabdem
    fonte_adm_mun = setores[setores.administrativa > 0].CD_MUN.unique()
    led["registro_administrativo"] = np.where(led.administrativa > 0, "com_registro",
                                              np.where(led.CD_MUN.isin(fonte_adm_mun), "fonte_existe_sem_registro",
                                                       "fonte_inexistente_no_acervo"))
    gravar(OUT / "universo/ledger_fcu_geometria_local.csv", led)

    # comparacao de disponibilidade de referencia por classe territorial (denominadores: setores, pessoas, area)
    urb = setores[setores.classe_territorial.isin(["fcu", "baixa_renda", "comparador_urbano"])]
    disp = (urb.assign(com_fisica=urb.fisica > 0, com_fisica_duas=(urb.fisica_pos > 0) & (urb.fisica_neg > 0),
                       com_adm=urb.administrativa > 0)
            .groupby(["UF", "classe_territorial"])
            .agg(setores=("CD_SETOR", "size"), pessoas=("pessoas", "sum"), com_fisica=("com_fisica", "sum"),
                 com_fisica_duas_classes=("com_fisica_duas", "sum"), com_administrativa=("com_adm", "sum"),
                 pontos_fisicos=("fisica", "sum"), pontos_administrativos=("administrativa", "sum"),
                 terreno=("terreno_fabdem", "sum"), cems_aoi=("cems_aoi", "sum")).reset_index())
    gravar(OUT / "universo/disponibilidade_por_classe.csv", disp)

    funil = {
        "fcu_nacional": int(len(fcu)), "pessoas_em_fcu_nacional": float(fcu.pessoas.sum()),
        "setores_fcu_nacional": int(len(fcu_set)),
        "fcu_com_geometria_local": int(fcu.geometria_local.sum()),
        "fcu_ledger": int(len(led)),
        "fcu_com_algum_ponto_do_acervo": int((led.pontos > 0).sum()),
        "fcu_com_ponto_fisico": int((led.fisica > 0).sum()),
        "fcu_com_fisico_positivo": int((led.fisica_pos > 0).sum()),
        "fcu_com_fisico_duas_classes": int(((led.fisica_pos > 0) & (led.fisica_neg > 0)).sum()),
        "fcu_com_registro_administrativo": int((led.administrativa > 0).sum()),
        "fcu_em_aoi_cems": int(led.cems_aoi.sum()),
        "fcu_com_terreno_fabdem": int(led.terreno_fabdem.sum()),
        "fcu_com_modelo_local_ajustado": int(led.modelo_local.ne("inexistente").sum()),
    }
    fontes = {k: {"arquivo": rel(v), "sha256": sha(v)} for k, v in SETORES.items()}
    fontes.update({"agregado_basico": {"arquivo": rel(AGREGADO_BASICO), "sha256": sha(AGREGADO_BASICO)},
                   "agregado_renda": {"arquivo": rel(AGREGADO_RENDA), "sha256": sha(AGREGADO_RENDA)},
                   "tabela_unica": {"arquivo": "local_runs/ds-05-tabela-unica/tabela_unica_v2.parquet",
                                    "sha256": sha(RUNS / "ds-05-tabela-unica/tabela_unica_v2.parquet")}})
    gravar(OUT / "universo/resultado.json", {
        "inicio_utc": t0, "fim_utc": agora(), "funil_fcu": funil, "fontes": fontes,
        "definicoes": {"fcu": "CD_TIPO=='1' (Censo 2022); comunidade = CD_FCU",
                       "baixa_renda": f"urbano, CD_TIPO=='0', V06006 <= {SALARIO_MINIMO_2022}",
                       "comparador_urbano": f"urbano, CD_TIPO=='0', V06006 > {SALARIO_MINIMO_2022}",
                       "tipos_de_referencia": FONTE_TIPO,
                       "falta_dado_treinar": "nenhum positivo fisico do acervo dentro da comunidade",
                       "falta_dado_avaliar": "faltam positivo ou negativo fisico dentro da comunidade",
                       "falta_insumo_fisico": "centro de algum setor fora de unidade FABDEM compativel do acervo",
                       "registro_administrativo": "contagem de pontos SEDEC/156/SP156 na comunidade; distingue fonte inexistente"},
        "limites": ["pontos do acervo sao amostras esparsas (GFD 250 m, CEMS 150 m), nao cobertura completa",
                    "geometria so nas UFs PR, RS, AM, PE, RJ, SP, RO; demais UFs entram so no universo por agregado",
                    "agregados do Censo 2022; a data da observacao fisica e diferente"],
    })
    print(json.dumps(funil, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
