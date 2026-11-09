"""F1: com negativo OBSERVADO do FloodNet, o que chuva e terreno explicam no pluvial de Nova York?

Entrada: janelas sensor x dia da auditoria (`fn02`), so `lamina` e `sem_lamina`
(sensor funcionando; desconhecido e nao_observado ficam fora). Chuva: MRMS
24 h (00Z-00Z, o mesmo dia UTC da janela) no pixel de 0,01 grau do sensor.
Terreno: HAND, TWI e declive do MDT de 30 m de Nova York no ponto do sensor.

  validacao  dia de evento do catalogo publico (sensor mais proximo a <= 50 m)
             aparece como `lamina` na janela daquele sensor e dia?
  forcamento DENTRO do sensor: AUC da chuva para lamina x sem_lamina, por sensor
             com as duas classes; mediana e IC por bootstrap de sensores
  terreno    ENTRE sensores no MESMO dia chuvoso (MRMS >= 20 mm no sensor):
             AUC de -HAND e de TWI por dia com as duas classes; media e IC por
             bootstrap de dias; AUC da chuva no mesmo estrato como controle

Declarado antes: o FloodNet instala onde ja houve alagamento, entao o contraste
entre sensores e entre lugares ja propensos -- diz se o terreno ordena DENTRO
desse conjunto, nao na cidade. `sem_lamina` e "nada acima de 10 mm no ponto do
sensor naquele dia", nao seco absoluto. Mecanismo: so sensores `pluvial`;
rompimento de rede chega igual a chuva.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
AUD = REPO / "local_runs" / "floodnet-03-auditoria" / "janelas_sensor_dia.parquet"
MRMS = REPO / "local_runs" / "chuva-mrms-nyc" / "janelas"
MDT = REPO / "local_runs" / "mdt-cidades" / "nyc"
CATALOGO = REPO / "local_runs" / "floodnet-01-catalogo-publico"
CHUVA_DIA_MM = 20.0


def chuva(d: pd.DataFrame) -> np.ndarray:
    out = np.full(len(d), np.nan)
    for dia, idx in d.groupby(d.dia.dt.strftime("%Y-%m-%d")).indices.items():
        f = MRMS / f"{dia}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        a, _, c, _, e, fy = z["transform"]
        col = ((d.lon.to_numpy()[idx] - c) / a).astype(int)
        lin = ((d.lat.to_numpy()[idx] - fy) / e).astype(int)
        ok = (lin >= 0) & (lin < z["z"].shape[0]) & (col >= 0) & (col < z["z"].shape[1])
        v = np.full(len(idx), np.nan)
        v[ok] = z["z"][lin[ok], col[ok]]
        out[idx] = v
    return out


def terreno(d: pd.DataFrame) -> dict:
    import rasterio
    from rasterio.warp import transform
    out = {}
    for nome, arq in (("hand_m", "nyc_mdt_30m_hand.tif"), ("twi", "nyc_mdt_30m_twi.tif"),
                      ("slope_deg", "nyc_mdt_30m_slope.tif")):
        with rasterio.open(MDT / arq) as r:
            xs, ys = transform("EPSG:4326", r.crs, d.lon.tolist(), d.lat.tolist())
            v = np.array([p[0] for p in r.sample(zip(xs, ys))], dtype=float)
            if r.nodata is not None:
                v[v == r.nodata] = np.nan
        out[nome] = v
    return out


def validar_com_catalogo(j: pd.DataFrame) -> dict:
    from sklearn.neighbors import BallTree
    ev = pd.read_csv(CATALOGO / "eventos.csv", usecols=["sensor_id", "flood_start_time"])
    se = pd.read_csv(CATALOGO / "sensores.csv", usecols=["sensor_id", "latitude", "longitude"])
    ev = ev.merge(se, on="sensor_id").dropna()
    ev["dia"] = pd.to_datetime(ev.flood_start_time, utc=True, format="mixed").dt.floor("D")
    s = j.groupby("deployment_id")[["lat", "lon"]].first().dropna()
    arv = BallTree(np.radians(s[["lat", "lon"]].to_numpy()), metric="haversine")
    dist, i = arv.query(np.radians(ev[["latitude", "longitude"]].to_numpy()), k=1)
    ev["deployment_id"] = s.index.to_numpy()[i[:, 0]]
    ev = ev[dist[:, 0] * 6371000 <= 50]
    m = ev.merge(j[["deployment_id", "dia", "classe"]], on=["deployment_id", "dia"], how="left")
    m = m[m.dia <= j.dia.max()]
    return {"eventos_do_catalogo_no_periodo": int(len(m)),
            "classe_da_janela": m.classe.fillna("sem_janela").value_counts().to_dict()}


def bootstrap_media(v: np.ndarray, rng, reps: int = 2000) -> list:
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(reps)], [2.5, 97.5]).round(4).tolist()


def executar(p: dict, saida: Path) -> dict:
    rng = np.random.default_rng(p["sementes"][0])
    j = pd.read_parquet(AUD)
    j["dia"] = pd.to_datetime(j.dia, utc=True)
    valid = validar_com_catalogo(j)
    d = j[j.classe.isin(["lamina", "sem_lamina"])].reset_index(drop=True)
    d["y"] = (d.classe == "lamina").astype(int)
    d["chuva_mm"] = chuva(d)
    sens = d.groupby("deployment_id")[["lat", "lon"]].first().reset_index()
    for k, v in terreno(sens).items():
        sens[k] = v
    d = d.merge(sens.drop(columns=["lat", "lon"]), on="deployment_id").dropna(subset=["chuva_mm"])
    # forcamento dentro do sensor
    aucs = [roc_auc_score(g.y, g.chuva_mm) for _, g in d.groupby("deployment_id") if g.y.nunique() == 2 and g.y.sum() >= 3]
    forc = {"sensores": len(aucs), "auc_mediana": round(float(np.median(aucs)), 4) if aucs else None,
            "auc_media": round(float(np.mean(aucs)), 4) if aucs else None,
            "ic95_media": bootstrap_media(np.array(aucs), rng) if len(aucs) >= 5 else None}
    # terreno entre sensores no mesmo dia chuvoso
    ch = d[d.chuva_mm >= CHUVA_DIA_MM].dropna(subset=["hand_m", "twi"])
    por_dia = {"hand": [], "twi": [], "chuva": []}
    for _, g in ch.groupby("dia"):
        if g.y.nunique() == 2 and len(g) >= 5:
            por_dia["hand"].append(roc_auc_score(g.y, -g.hand_m))
            por_dia["twi"].append(roc_auc_score(g.y, g.twi))
            por_dia["chuva"].append(roc_auc_score(g.y, g.chuva_mm))
    terr = {k: {"dias": len(v), "auc_media": round(float(np.mean(v)), 4) if v else None,
                "ic95_media_por_dia": bootstrap_media(np.array(v), rng) if len(v) >= 5 else None}
            for k, v in por_dia.items()}
    return {"tipo": "floodnet_negativo_observado", "pergunta": p["pergunta"],
            "janelas_lamina": int(d.y.sum()), "janelas_sem_lamina": int((d.y == 0).sum()),
            "sensores": int(d.deployment_id.nunique()), "dias": int(d.dia.nunique()),
            "validacao_com_catalogo": valid, "forcamento_dentro_do_sensor": forc,
            "terreno_entre_sensores_mesmo_dia_chuvoso": {"limiar_mm": CHUVA_DIA_MM, **terr},
            "veredito": "diagnostico",
            "interpretacao_permitida": ("primeiro uso de negativo pluvial observado; sensores instalados onde ja "
                                        "alagou: contraste entre lugares propensos, nao representa a cidade; "
                                        "Nova York nao valida o Brasil")}
