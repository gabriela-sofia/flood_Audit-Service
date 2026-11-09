"""Ocorrencia de agua JRC v1.4 nos pontos da populacao MT01 (contrato gfd02: alvo exige occurrence < 50).

O ds06 v7 ainda nao aplica o corte do contrato (o `alvo_v2` de 10/10 so existe
para a populacao TC01, com outros IDs). Aqui a ocorrencia e lida no ponto, do
mesmo produto e pelo mesmo caminho remoto que `agua_permanente_jrc.url_de`
usa, para uma SENSIBILIDADE pre-registrada: o contraste fica de pe sem agua
frequente? Nao reescreve rotulo nem populacao. Codigo > 100 vira NaN.

Uso:
    python scripts/metodo/mt01_ocorrencia_jrc.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "geo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc, sha256_arquivo  # noqa: E402
import mt01_populacao as mp  # noqa: E402

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_CACHEMAX", "512")


def main() -> int:
    import rasterio
    from agua_permanente_jrc import tile_de, url_de
    sys.stdout.reconfigure(encoding="utf-8")
    saida = mp.SAIDA / "ocorrencia_jrc.parquet"
    d = pd.read_parquet(mp.SAIDA / "populacao.parquet", columns=["ponto_id", "fonte", "papel", "lon", "lat"])
    d = d[d.fonte.isin(mp.FONTES_CONTRASTE)]
    d["tile"] = [tile_de(lo, la) for lo, la in zip(d.lon, d.lat)]
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0 = time.time()
    feitos = pd.read_parquet(saida) if saida.exists() else pd.DataFrame(columns=["ponto_id", "occurrence", "erro"])
    falta = d[~d.ponto_id.isin(feitos.ponto_id)].sort_values(["tile", "lat", "lon"])
    # blocos de 1.500 pontos vizinhos: os tiles sul-americanos concentram centenas de milhares de pontos
    blocos = [g for _, t in falta.groupby("tile") for _, g in t.groupby(np.arange(len(t)) // 1500)]

    def ler(g):
        try:  # cada thread abre o proprio dataset
            with rasterio.open(url_de(float(g.lon.iloc[0]), float(g.lat.iloc[0]))) as s:
                v = np.array([p[0] for p in s.sample(list(zip(g.lon, g.lat)))], dtype="float64")
            v[v > 100] = np.nan
            erro = None
        except Exception as e:  # sem rede: fica NaN e o motivo, nunca zero
            v, erro = np.full(len(g), np.nan), str(e)[:120]
        return pd.DataFrame({"ponto_id": g.ponto_id.to_numpy(), "occurrence": v, "erro": erro})

    partes = []
    with ThreadPoolExecutor(8) as pool:
        for k, f in enumerate(as_completed([pool.submit(ler, g) for g in blocos])):
            partes.append(f.result())
            if k % 20 == 19:
                print(k + 1, "/", len(blocos), round(time.time() - t0), flush=True)
                pd.concat([feitos, *partes]).to_parquet(saida, index=False)
    out = pd.concat([feitos, *partes]) if partes else feitos
    out.to_parquet(saida, index=False)
    mp.escrever(mp.SAIDA / "ocorrencia_jrc.json", {
        "quando_utc": agora_utc(), "pontos": int(len(out)), "nan": int(out.occurrence.isna().sum()),
        "produto": "JRC GSW v1.4 occurrence (1984-2021), leitura remota por agua_permanente_jrc.url_de",
        "sha256": sha256_arquivo(saida), "nao_e": "nao reescreve rotulo; sensibilidade do contrato gfd02"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
