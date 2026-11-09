"""N1: o embedding do encoder de terreno congelado acrescenta ao HAND+TWI em evento nao visto?

Mesma cadeia e escala do treino do encoder: as 5 camadas wbt30 (30 m, UTM)
de `ter-01-cadeia-harmonizada`, janela de 128 px (3,84 km) centrada no ponto
-- o tamanho do recorte do treino --, normalizada com media e desvio do
MANIFESTO do encoder, pixel invalido = 0 (a media), como `preparar` faz.

Checkpoint: entre os encoders B2 por chip, o de menor perda de VALIDACAO na
ultima epoca registrada. Escolha feita antes de qualquer medida downstream.

Exposicao: o encoder foi treinado em 03/09 com chips S1F11 e AOIs derivadas;
nenhum tile `gfd_tile_*` estava em `ter-01` antes dessa data -- conferido pela
data de criacao das pastas e gravado no card.

Amostra: pontos GFD de planicie autorizados para avaliacao_espacial; ate
POR_EVENTO pontos por evento (metade por classe quando houver), semente fixa.
Divisao: 5 dobras agrupadas por evento; PCA do embedding ajustada so no
treino da dobra. Tres modelos nas MESMAS linhas: terreno (hand+twi),
embedding (PCA), terreno+embedding.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
TER = RUNS / "ter-01-cadeia-harmonizada"
ENC = RUNS / "terreno-encoder"
CAMADAS = ["dem_filled", "slope_deg_wbt", "hand_dinf", "dinf_sca", "dinf_accum_cells"]
LADO = 128
POR_EVENTO = 300
COMPONENTES = 16
REF = ["hand_m", "twi_dinf"]

import sys as _sys
_sys.path.insert(0, str(REPO / "scripts" / "terreno"))
from dados_terreno import CAMADAS as _CAMADAS_TREINO, CAMADAS_EM_LOG, SEM_DADO  # noqa: E402
assert tuple(CAMADAS) == tuple(_CAMADAS_TREINO), "ordem de canais difere do treino"


def escolher_checkpoint() -> tuple[str, dict]:
    cands = []
    for m in sorted(ENC.glob("encoder_amostra_chip_s*_manifesto.json")):
        man = json.loads(m.read_text(encoding="utf-8"))
        vals = [h["perda_validacao"] for h in man["historico"] if h.get("perda_validacao")]
        if vals:
            cands.append((vals[-1], m.name.replace("encoder_", "").replace("_manifesto.json", ""), man))
    cands.sort(key=lambda x: x[0])
    v, rot, man = cands[0]
    return rot, {"rotulo": rot, "perda_validacao_ultima_epoca": v,
                 "candidatos": {c[1]: c[0] for c in cands}}


def exposicao_gfd(manifesto: dict) -> dict:
    from datetime import datetime, timezone
    treino = datetime.fromisoformat(manifesto["quando_utc"].replace("Z", "+00:00"))
    datas = [datetime.fromtimestamp(os.path.getctime(d), tz=timezone.utc) for d in TER.glob("gfd_tile_*")]
    antes = sum(1 for x in datas if x < treino)
    return {"encoder_treinado_em": manifesto["quando_utc"], "tiles_gfd": len(datas),
            "tiles_gfd_criados_antes_do_treino": antes, "regime": "indutivo" if antes == 0 else "transdutivo_parcial"}


def janela(tile_dir: Path, lat: float, lon: float, abertos: dict):
    from rasterio.windows import Window
    from rasterio.warp import transform
    pilha, valido = [], None
    for c in CAMADAS:
        r = abertos[(tile_dir, c)]
        xs, ys = transform("EPSG:4326", r.crs, [lon], [lat])
        lin, col = r.index(xs[0], ys[0])
        w = Window(col - LADO // 2, lin - LADO // 2, LADO, LADO)
        a = r.read(1, window=w, boundless=True, fill_value=np.nan).astype(np.float32)
        if r.nodata is not None:
            a[a == r.nodata] = np.nan
        a[a == SEM_DADO] = np.nan
        pilha.append(a)
        v = np.isfinite(a)
        valido = v if valido is None else (valido & v)
    pilha = np.stack(pilha)
    # mesma transformacao de `dados_terreno.ler_terreno`: sem o log1p o encoder
    # recebe sca/acumulacao em escala bruta, fora de tudo que viu no treino
    for i, c in enumerate(CAMADAS):
        if c in CAMADAS_EM_LOG:
            pilha[i] = np.log1p(np.where(valido & (pilha[i] > 0), pilha[i], 0.0))
    return pilha, valido


def embeddings(pts: pd.DataFrame, rotulo: str, cache: Path | None = None) -> np.ndarray:
    """Embedding por ponto; com `cache`, grava tambem a janela normalizada (float16).

    O cache serve ao N2: avaliar cada checkpoint novo sem reabrir centenas de
    rasters. E so implementacao -- o protocolo do N1 nao muda com ele.
    """
    import rasterio
    import sys
    import torch
    sys.path.insert(0, str(REPO / "scripts" / "terreno"))
    from avaliar_embedding import carregar
    modelo, man = carregar(rotulo, ENC)
    media = np.asarray(man["normalizacao"]["media"], np.float32).reshape(-1, 1, 1)
    desvio = np.asarray(man["normalizacao"]["desvio"], np.float32).reshape(-1, 1, 1)
    out = np.full((len(pts), modelo.embedding(torch.zeros(1, len(CAMADAS), LADO, LADO)).shape[1]), np.nan)
    pts = pts.reset_index(drop=True)
    jan = (np.lib.format.open_memmap(cache, mode="w+", dtype=np.float16,
                                      shape=(len(pts), len(CAMADAS), LADO, LADO)) if cache else None)
    for tile, g in pts.groupby("tile_dir"):
        d = TER / tile
        if not all((d / f"{c}.tif").exists() for c in CAMADAS):
            continue
        abertos = {(d, c): rasterio.open(d / f"{c}.tif") for c in CAMADAS}
        try:
            lote, onde = [], []
            for i, r in g.iterrows():
                x, v = janela(d, r.lat, r.lon, abertos)
                if v.mean() < 0.5:
                    continue
                x = (x - media) / desvio
                x[:, ~v] = 0.0
                if jan is not None:
                    if np.abs(x).max() > 6e4:
                        raise ValueError(f"janela fora do alcance de float16 em {tile}: {np.abs(x).max():.3g}")
                    jan[i] = x.astype(np.float16)
                lote.append(x)
                onde.append(i)
                if len(lote) == 64:
                    with torch.no_grad():
                        out[onde] = modelo.embedding(torch.from_numpy(np.stack(lote))).numpy()
                    lote, onde = [], []
            if lote:
                with torch.no_grad():
                    out[onde] = modelo.embedding(torch.from_numpy(np.stack(lote))).numpy()
        finally:
            for r in abertos.values():
                r.close()
    if jan is not None:
        jan.flush()
    return out


def avaliar(d: pd.DataFrame, E: np.ndarray, semente: int) -> dict:
    rng = np.random.default_rng(semente)
    u = np.asarray(d.evento_id.unique(), dtype=object)
    rng.shuffle(u)
    dobra = d.evento_id.map({e: i % 5 for i, e in enumerate(u)}).to_numpy()
    y = d.classe.to_numpy()
    prev = {k: np.zeros(len(d)) for k in ("terreno", "embedding", "terreno_mais_embedding")}
    for k in range(5):
        te, tr = dobra == k, dobra != k
        pca = PCA(COMPONENTES, random_state=semente).fit(E[tr])
        Z = pca.transform(E)
        T = d[REF].to_numpy(float)
        for nome, X in (("terreno", T), ("embedding", Z), ("terreno_mais_embedding", np.column_stack([T, Z]))):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            m = LogisticRegression(C=1.0, max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
            prev[nome][te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    evs = d.evento_id.to_numpy()
    idx = {e: np.flatnonzero(evs == e) for e in np.unique(evs)}

    def dentro(pv):
        a = [roc_auc_score(y[i], pv[i]) for i in idx.values() if len(np.unique(y[i])) == 2]
        return float(np.median(a)) if a else None
    met = {n: {"auc_agrupada": round(float(roc_auc_score(y, v)), 4), "auc_mediana_dentro_do_evento": round(dentro(v), 4),
               "brier": round(float(brier_score_loss(y, v)), 4)} for n, v in prev.items()}
    difs = {"embedding": [], "terreno_mais_embedding": []}
    for _ in range(500):
        i = np.concatenate([idx[e] for e in rng.choice(list(idx), len(idx))])
        if len(np.unique(y[i])) == 2:
            for n in difs:
                difs[n].append(roc_auc_score(y[i], prev[n][i]) - roc_auc_score(y[i], prev["terreno"][i]))
    ic = {n: np.percentile(v, [2.5, 97.5]).round(4).tolist() for n, v in difs.items()}
    return {"metricas": met, "diferenca_contra_terreno_ic95_por_evento": ic, "eventos": int(len(idx)),
            "linhas": int(len(d)), "positivos": int(y.sum())}


def amostrar(g: pd.DataFrame, semente: int) -> pd.DataFrame:
    """Ate POR_EVENTO//2 linhas por evento x classe, com TODAS as colunas.

    O `apply` do pandas atual descarta as colunas de agrupamento; por isso a
    amostra devolve so os indices e as linhas saem de `g` inteiro.
    """
    idx = [i for _, s in g.groupby(["evento_id", "classe"], sort=True)
           for i in s.sample(min(len(s), POR_EVENTO // 2), random_state=semente).index]
    return g.loc[idx]


def executar(p: dict, saida: Path) -> dict:
    import admissao_cientifica as adm
    from ds06_dataset_de_treino import versao_atual
    semente = p["sementes"][0]
    t = pd.read_parquet(versao_atual() / "pontos.parquet",
                        columns=["ponto_id", "fonte", "mecanismo", "papel", "classe", "classe_relevo", "evento_id",
                                 "grupo_validacao", "qualidade_negativo", "precisao_data", "janela_inicio",
                                 "lat", "lon", "hand_m", "twi_dinf", "slope_deg", "elevation_m"])
    g = t[(t.fonte == "gfd") & (t.classe_relevo == "PLANO_OU_ONDULADO") & (t.mecanismo == "FLUVIAL_ENXURRADA")]
    g, barr = adm.filtrar(g, "avaliacao_espacial", REF)
    amostra = amostrar(g, semente)
    amostra = amostra.assign(tile_dir="gfd_tile_" + amostra.grupo_validacao.str.replace("GFD_", "", regex=False))
    rotulo, escolha = escolher_checkpoint()
    man = json.loads((ENC / f"encoder_{rotulo}_manifesto.json").read_text(encoding="utf-8"))
    E = embeddings(amostra, rotulo, cache=saida / "janelas_f16.npy")
    ok = np.isfinite(E).all(axis=1)
    amostra = amostra.reset_index(drop=True).assign(indice_janela=np.arange(len(amostra)), janela_valida=ok)
    amostra[["ponto_id", "evento_id", "classe", "tile_dir", "lat", "lon", "hand_m", "twi_dinf", "indice_janela",
             "janela_valida"]].to_parquet(saida / "amostra.parquet", index=False)
    d, E = amostra[ok], E[ok]
    np.save(saida / "embeddings.npy", E)
    r = avaliar(d.reset_index(drop=True), E, semente)
    ic = r["diferenca_contra_terreno_ic95_por_evento"]["terreno_mais_embedding"]
    return {"tipo": "encoder_por_ponto", "pergunta": p["pergunta"], "checkpoint": escolha,
            "exposicao_do_pretreino": exposicao_gfd(man), "barrado_na_admissao": barr,
            "amostra": {"pontos_sorteados": int(len(amostra)), "com_janela_valida": int(ok.sum()),
                        "por_evento_max": POR_EVENTO, "janela_px": LADO, "componentes_pca": COMPONENTES},
            **r,
            "veredito": ("acrescimo" if ic[0] > 0 else "perda" if ic[1] < 0 else "sem_ganho_demonstrado"),
            "interpretacao_permitida": ("representacao congelada avaliada em tarefa explicita; ganho so com IC por "
                                        "evento fora de zero; reducao de perda de reconstrucao nao entra aqui")}
