"""
DS-01 -- Monta o dataset multirregiao a partir das fontes ja processadas.

PROBLEMA QUE RESOLVE, e a armadilha que evita:
o projeto passou a ter fontes de procedencia diferente -- outline historico
ingles, poligono observado Copernicus, ponto rotulado a mao do Sen1Floods11 e
do UFO. Empilhar tudo numa tabela so, sem marcar de onde veio cada linha,
produziria um dataset onde a mesma coluna `label` significa coisas
diferentes: num caso "houve registro de inundacao", noutro "o analista viu
inundacao", noutro "um humano marcou o pixel". Misturar isso sem carimbo e o
caminho mais curto para um resultado indefensavel.

Por isso toda linha carrega, obrigatoriamente:
    regiao          UK / RS / analogo (com o codigo da ativacao)
    fonte_label     como o rotulo foi produzido
    tipo_negativo   por_observacao | por_exclusao_qualificada | ausente
    grupo_cv        unidade de agrupamento (evento, AOI ou bloco espacial)
    resolucao_m     resolucao nativa da observacao

HIERARQUIA DE NEGATIVO, declarada e nao negociavel:

  por_observacao            area declarada como analisada e sem inundacao
                            detectada (Copernicus EMS AOI ∩ footprint,
                            Sen1Floods11 classe 0, UFO non-inundated).
                            E o unico que constitui registro de ausencia.

  por_exclusao_qualificada  fora de outline registrado E fora de zona
                            modelada E com afastamento minimo E pareado por
                            cobertura (criterio N1-N4 do piloto ingles).
                            Melhor que ausencia pura, mas ainda e inferencia.

  ausente                   o negativo do Protocolo C brasileiro. Nao entra.

Modelos que misturarem os dois primeiros DEVEM reportar a proporcao de cada
tipo. Um AUC obtido majoritariamente sobre negativo por exclusao nao pode ser
apresentado como se fosse validado por observacao.

CLASSES: o dataset e multiclasse por construcao, nao binario.
    1 INUNDACAO
    2 MOVIMENTO_DE_MASSA   -- classe propria; nem positivo, nem negativo
    0 NAO_INUNDADO
    9 AGUA_PERMANENTE      -- excluida do treino, mantida para auditoria

Tratar movimento de massa como negativo ensinaria ao modelo que encosta
cicatrizada e lugar seguro. E o oposto da verdade, e e o erro que os dados
brasileiros nao permitem evitar.

NAO faz: nao treina, nao avalia, nao promove nada a rotulo definitivo.

Uso:
    python scripts/treino/ds01_montar_multirregiao.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
OUT = RUNS / "ds-01-multirregiao"

ESQUEMA = ["ponto_id", "regiao", "classe", "classe_nome", "grupo_cv",
           "fonte_label", "tipo_negativo", "resolucao_m", "lat", "lon",
           "data_evento", "elevation_m", "slope_deg", "hand_m", "twi_dinf",
           "worldcover", "rain_max_24h", "rain_decay_index_api",
           "fonte_chuva"]

# `INUNDADO` vem do Nivel 1 e `INUNDACAO` do CEMS -- vocabularios diferentes
# para a mesma coisa. Normalizar aqui evita que 25.886 pontos virem NaN em
# silencio, que foi o que aconteceu na primeira execucao.
MAPA_CLASSE = {"INUNDACAO": 1, "INUNDADO": 1, "MOVIMENTO_DE_MASSA": 2,
               "NAO_INUNDADO_OBSERVADO": 0, "NAO_INUNDADO": 0,
               "AGUA_PERMANENTE": 9, "SEM_DADO": -1}


def vazio() -> pd.DataFrame:
    return pd.DataFrame(columns=ESQUEMA)


def carregar_chamados() -> pd.DataFrame:
    """Centrais de atendimento ao cidadao promovidas por `aq_chamados03`.

    Nao ha leitura nem decisao aqui de proposito: o promotor ja entregou o
    esquema deste modulo, com rotulo, grupo e terreno pela cadeia canonica.
    Se ele nao rodou, a pasta esta vazia e o dataset segue sem estas fontes --
    e o mesmo comportamento das outras, que somem quando a entrada falta.

    O negativo delas e `ausente`: chamado de outro assunto da oportunidade de
    observacao, mas ninguem foi ver se havia agua. E o mesmo nivel de Curitiba,
    e quem decide o que fazer com ele e o `ds05`.
    """
    pasta = RUNS / "aq-chamados-03"
    arquivos = sorted(pasta.glob("*_ds01.csv")) if pasta.exists() else []
    if not arquivos:
        return vazio()
    partes = [pd.read_csv(a, low_memory=False) for a in arquivos]
    d = pd.concat(partes, ignore_index=True)
    faltando = [c for c in ESQUEMA if c not in d.columns]
    if faltando:
        print(f"  chamados: esquema incompleto, faltam {faltando} -- ignorado")
        return vazio()
    print(f"  chamados: {len(d):,} de {len(arquivos)} fonte(s) "
          f"({', '.join(sorted(d.regiao.unique()))})")
    return d[ESQUEMA]


def carregar_uk() -> pd.DataFrame:
    """Piloto ingles. Negativo por EXCLUSAO QUALIFICADA (criterio N1-N4)."""
    com_chuva = RUNS / "ft-uk-04-chuva" / "amostra_candidata_uk_v2_com_chuva.csv"
    sem_chuva = RUNS / "ft-uk-03-amostra" / "amostra_candidata_uk_v1.csv"
    # A amostra v1 e a fonte estrutural. A v2 e apenas uma visao derivada e
    # pode ter sido produzida por um metodo historico; usa-la como entrada
    # fazia a primeira e a segunda execucoes divergirem. Sobrepor chuva por
    # ponto preserva a estrutura e deixa ausente onde nao ha data observada.
    caminho = sem_chuva
    if not caminho.exists():
        return vazio()
    d = pd.read_csv(caminho)
    if com_chuva.exists():
        rain = pd.read_csv(com_chuva, low_memory=False)
        rain_cols = [c for c in ("ponto_id", "rain_max_24h",
                                  "rain_decay_index", "fonte_chuva")
                     if c in rain.columns]
        d = d.drop(columns=[c for c in rain_cols if c != "ponto_id"],
                   errors="ignore").merge(
                       rain[rain_cols].drop_duplicates("ponto_id"),
                       on="ponto_id", how="left", validate="one_to_one")
    out = pd.DataFrame({
        "ponto_id": d["ponto_id"], "regiao": "UK_noroeste",
        "classe": d["label"].map({1: 1, 0: 0}),
        "classe_nome": d["label"].map({1: "INUNDACAO", 0: "NAO_INUNDADO"}),
        "grupo_cv": d["grupo_cv"],
        "fonte_label": d["fonte_label"],
        "tipo_negativo": np.where(d["label"] == 0, "por_exclusao_qualificada", ""),
        "resolucao_m": 30, "lat": d["lat"], "lon": d["lon"],
        "data_evento": d.get("start_date"),
        "elevation_m": d["elevation_m"], "slope_deg": d["slope_deg"],
        "hand_m": d["hand_m"], "twi_dinf": d["twi_dinf"],
        "worldcover": d["worldcover"],
    })
    max_col = "rain_max_24h" if "rain_max_24h" in d.columns else "rain_max_24h_rnl"
    decay_col = ("rain_decay_index" if "rain_decay_index" in d.columns
                 else "rain_decay_index_api_rnl")
    out["rain_max_24h"] = d[max_col] if max_col in d.columns else np.nan
    out["rain_decay_index_api"] = d[decay_col] if decay_col in d.columns else np.nan
    if "fonte_chuva" in d.columns:
        out["fonte_chuva"] = d["fonte_chuva"].fillna("ausente")
    else:
        out["fonte_chuva"] = np.where(
            out["rain_max_24h"].notna(), "open_meteo_era5_land", "ausente")
    print(f"[UK] n={len(out):,} (chuva={'sim' if com_chuva.exists() else 'NAO'})")
    return out


def carregar_cems() -> pd.DataFrame:
    """Toda pasta cems-emsr*/pontos_*.csv. Negativo POR OBSERVACAO."""
    partes = []
    for pasta in sorted(RUNS.glob("cems-emsr*")):
        for csv in pasta.glob("pontos_*.csv"):
            d = pd.read_csv(csv)
            codigo = str(d["ativacao"].iloc[0]) if "ativacao" in d.columns else pasta.name
            out = pd.DataFrame({
                "ponto_id": d["ponto_id"], "regiao": f"analogo_{codigo}",
                "classe": d["classe_obs"].map(MAPA_CLASSE),
                "classe_nome": d["classe_obs"],
                # grupo = ativacao + AOI, vindo do proprio cems02. Ate
                # 09/08/2026 era so o codigo da ativacao, o que dava 5 grupos
                # no total e EPV de 1,67 -- abaixo do minimo do projeto.
                "grupo_cv": (d["grupo_cv"] if "grupo_cv" in d.columns
                             else f"AOI_{codigo}"),
                "fonte_label": "copernicus_ems_observado",
                "tipo_negativo": np.where(
                    d["classe_obs"] == "NAO_INUNDADO_OBSERVADO", "por_observacao", ""),
                "resolucao_m": d.get("resolucao_m", 30),
                "lat": d["lat"], "lon": d["lon"], "data_evento": pd.NaT,
                "elevation_m": d.get("elevation_m"), "slope_deg": d.get("slope_deg"),
                "hand_m": d.get("hand_m"), "twi_dinf": d.get("twi_dinf"),
                "worldcover": d.get("worldcover"),
                "rain_max_24h": np.nan, "rain_decay_index_api": np.nan,
                "fonte_chuva": "ausente",
            })
            partes.append(out)
            print(f"[CEMS {codigo}] n={len(out):,}")
    return pd.concat(partes, ignore_index=True) if partes else vazio()


def carregar_nivel1() -> pd.DataFrame:
    """
    Sen1Floods11 e UFO. Negativo POR OBSERVACAO, mas SEM features.

    Entram no dataset com as colunas de feature vazias de proposito: eles
    servem para auditar o criterio de negativo por concordancia espacial, nao
    para treinar. Preencher feature aqui exigiria adquirir raster para 11+215
    locais no mundo -- custo alto para um uso que ainda nao foi decidido.
    """
    # prefere a versao COM features (n1g). O n1f e o fallback historico.
    com_feat = RUNS / "n1g-pontos-com-features" / "nivel1_pontos_com_features_v1.csv"
    sem_feat = RUNS / "n1f-pontos" / "nivel1_pontos_v1.csv"
    caminho = com_feat if com_feat.exists() else sem_feat
    if not caminho.exists():
        return vazio()
    d = pd.read_csv(caminho)
    tem_feat = "elevation_m" in d.columns and d["elevation_m"].notna().any()
    out = pd.DataFrame({
        "ponto_id": d["ponto_id"], "regiao": "nivel1_" + d["fonte"].astype(str),
        "classe": d["classe_obs"].map(MAPA_CLASSE),
        "classe_nome": d["classe_obs"],
        "grupo_cv": "N1_" + d["evento_id"].astype(str),
        "fonte_label": d["fonte"], "tipo_negativo": np.where(
            d["classe_obs"] == "NAO_INUNDADO_OBSERVADO", "por_observacao", ""),
        "resolucao_m": d["resolucao_m"], "lat": d["lat"], "lon": d["lon"],
        "data_evento": pd.NaT,
        # slope e TWI ficam NaN mesmo no n1g: sao derivadas direcionais e
        # exigem grade local reprojetada, que nao foi calculada para os 661
        # chips espalhados em 6 continentes. Declarado, nao inventado.
        "elevation_m": d["elevation_m"] if tem_feat else np.nan,
        "slope_deg": np.nan,
        "hand_m": d["hand_m"] if tem_feat else np.nan,
        "twi_dinf": np.nan,
        "worldcover": d["worldcover"] if tem_feat else np.nan,
        "rain_max_24h": np.nan, "rain_decay_index_api": np.nan,
        "fonte_chuva": "ausente",
    })
    print(f"[NIVEL1] n={len(out):,} "
          f"({'COM elevation/hand/worldcover' if tem_feat else 'sem features'}; "
          f"slope e twi ausentes por decisao)")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===DS-01_MULTIRREGIAO===")

    df = pd.concat([carregar_uk(), carregar_cems(), carregar_nivel1(),
                    carregar_chamados()], ignore_index=True)
    if df.empty:
        print("NENHUMA FONTE DISPONIVEL")
        return 1
    df = df[ESQUEMA]
    df.to_csv(OUT / "dataset_multirregiao_v1.csv", index=False)

    print("---COMPOSICAO---")
    print(f"N_TOTAL={len(df):,}")
    for r, s in df.groupby("regiao"):
        cls = s["classe_nome"].value_counts().to_dict()
        print(f"  {r}: n={len(s):,} {cls}")
    print("---TIPO DE NEGATIVO---")
    neg = df[df["classe"] == 0]
    for t, s in neg.groupby("tipo_negativo"):
        print(f"  {t or '(vazio)'}: {len(s):,} ({100*len(s)/max(len(neg),1):.1f}% dos negativos)")

    # subconjunto treinavel: tem feature e classe util
    treinavel = df[(df["classe"].isin([0, 1, 2]))
                   & df["elevation_m"].notna() & df["hand_m"].notna()]
    print("---SUBCONJUNTO TREINAVEL (com features)---")
    print(f"N={len(treinavel):,} grupos={treinavel.grupo_cv.nunique()}")
    print("  " + "; ".join(f"{k}:{v}" for k, v in
                           treinavel["classe_nome"].value_counts().items()))
    com_chuva = int(treinavel["rain_max_24h"].notna().sum())
    print(f"  com_chuva={com_chuva:,} sem_chuva={len(treinavel)-com_chuva:,}")
    treinavel.to_csv(OUT / "dataset_treinavel_v1.csv", index=False)

    (OUT / "resumo.json").write_text(json.dumps({
        "n_total": int(len(df)), "n_treinavel": int(len(treinavel)),
        "por_regiao": {k: int(v) for k, v in df["regiao"].value_counts().items()},
        "por_classe": {k: int(v) for k, v in df["classe_nome"].value_counts().items()},
        "tipo_negativo": {k or "vazio": int(v) for k, v in
                          neg["tipo_negativo"].value_counts().items()},
        "hierarquia_negativo": [
            "por_observacao > por_exclusao_qualificada > ausente(nao entra)"],
        "aviso": ("Modelo que misturar tipos de negativo DEVE reportar a "
                  "proporcao de cada um. Movimento de massa e classe propria."),
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
