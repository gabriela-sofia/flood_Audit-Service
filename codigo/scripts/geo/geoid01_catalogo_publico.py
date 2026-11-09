"""GEOID-Flood: adquirir catalogos e verificar identidade antes de baixar rotulos.

O split publicado nao e aceito como prova de independencia no REV-P. Este
comando nao baixa shards, nao consulta proporcao positiva e nao libera teste.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/servico"))
sys.path.insert(0, str(REPO / "scripts/suscetibilidade"))
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402

BASE = "https://huggingface.co"
DATASET = "links-ads/geoid-flood"
ARQUIVOS = ("README.md", "geoid-flood/tile_catalog.parquet", "geoid-flood-heldout/tile_catalog.parquet")
PROIBIDAS = ("positive_proportion", "flooded_proportion", "invalid_pixel_frac", "is_valid")


def baixar(url: str, destino: Path, limite: int = 32 * 1024 * 1024) -> dict:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        if destino.stat().st_size > limite:
            raise ValueError("cache excedeu limite do catalogo")
        return {"url": url, "bytes": destino.stat().st_size, "sha256": sha256_arquivo(destino)}
    temporario = destino.with_suffix(destino.suffix + ".part")
    for tentativa in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "REV-P-catalog-audit/1.0"})
            total = 0
            with urllib.request.urlopen(req, timeout=90) as resp, temporario.open("wb") as f:
                while True:
                    b = resp.read(1 << 20)
                    if not b:
                        break
                    total += len(b)
                    if total > limite:
                        raise ValueError("arquivo excedeu limite do catalogo; nao baixar shard")
                    f.write(b)
            temporario.replace(destino)
            return {"url": url, "bytes": total, "sha256": sha256_arquivo(destino)}
        except (OSError, TimeoutError):
            if tentativa == 2:
                raise
            time.sleep(2 ** tentativa)
    raise RuntimeError("download nao concluido")


def ativacao(texto: str) -> str | None:
    m = re.search(r"(?<![A-Za-z0-9])EMSR\d+", str(texto), flags=re.IGNORECASE)
    return m.group().upper() if m else None


def resumir_catalogo(arq: Path, conhecidos: set[str]) -> dict:
    nomes = pq.ParquetFile(arq).schema_arrow.names
    # Ler somente identidade, localizacao e datas; nao usar qualidade derivada
    # do alvo para procurar uma reserva com resultado favoravel.
    permitidas = [n for n in nomes if n not in PROIBIDAS and
                  (any(t in n.lower() for t in ("event", "aoi", "activation", "country", "countries", "time", "date", "crs", "geometry", "tile_id")))]
    d = pd.read_parquet(arq, columns=permitidas)
    identidade = [n for n in permitidas if any(t in n.lower() for t in ("event", "activation", "aoi", "tile_id"))]
    if not identidade:
        return {"estado": "schema_nao_reconhecido", "colunas": nomes, "rotulos_lidos": False}
    eventos = pd.Series([next((ativacao(row[n]) for n in identidade if ativacao(row[n])), None)
                        for row in d[identidade].to_dict("records")], index=d.index)
    country = [n for n in permitidas if "countr" in n.lower()]
    brasil = pd.Series(False, index=d.index)
    for n in country:
        brasil |= d[n].astype(str).str.contains(r"Brazil|Brasil|\bBRA\b|\bBR\b", case=False, regex=True)
    evs = sorted(set(eventos.dropna()))
    br = sorted(set(eventos[brasil].dropna()))
    pares = d.assign(_evento=eventos).dropna(subset=["_evento"])
    return {"estado": "catalogado", "tiles": int(len(d)), "ativacoes": evs, "ativacoes_brasil": br,
            "ativacoes_ja_no_repositorio": sorted(set(evs) & conhecidos),
            "brasil_sem_ativacao_local": sorted(set(br) - conhecidos),
            "colunas_lidas": permitidas, "colunas_de_rotulo_nao_lidas": [n for n in nomes if n in PROIBIDAS],
            "tiles_por_ativacao": {str(k): int(v) for k, v in pares._evento.value_counts().items()},
            "limite": "ausencia de ativacao local nao prova evento novo entre fontes ou territorio independente",
            "rotulos_lidos": False}


def executar(saida: Path) -> dict:
    api = BASE + "/api/datasets/" + DATASET
    with urllib.request.urlopen(api, timeout=60) as r:
        info = json.load(r)
    revisao = info.get("sha")
    if not revisao or not re.fullmatch(r"[0-9a-f]{40}", revisao):
        raise ValueError("revisao remota invalida")
    destino = Path(saida).resolve() / revisao
    downloads, falhas = {}, {}
    existentes = {x["rfilename"] for x in info.get("siblings", [])}
    for nome in ARQUIVOS:
        if nome not in existentes:
            falhas[nome] = "arquivo nao publicado nesta revisao"
            continue
        try:
            downloads[nome] = baixar(BASE + f"/datasets/{DATASET}/resolve/{revisao}/{nome}", destino / nome)
        except (OSError, ValueError) as e:
            falhas[nome] = str(e)
    conhecidos = {ativacao(p.name) for p in (REPO / "local_runs").iterdir()}
    conhecidos.discard(None)
    # Ativacoes usadas fora de diretorios de mesmo nome tambem contam.
    ds06 = REPO / "local_runs/ds-06-dataset-de-treino"
    catalogos = sorted(ds06.glob("v*/pontos.parquet"))
    if catalogos:
        from ds06_dataset_de_treino import versao_atual
        d = pd.read_parquet(versao_atual() / "pontos.parquet", columns=["evento_id"])
        conhecidos.update(e for v in d.evento_id.unique() if (e := ativacao(v)))
    resumo = {n: resumir_catalogo(destino / n, conhecidos) for n in downloads if n.endswith(".parquet")}
    r = {"tipo": "geoid01_catalogo_publico", "dataset": DATASET, "revisao": revisao,
         "fonte": "https://huggingface.co/datasets/links-ads/geoid-flood", "paper": "https://arxiv.org/abs/2608.02315",
         "licenca_declarada": "CC BY 4.0; DEM com licenca propria WorldDEM-30",
         "downloads": downloads, "falhas": falhas, "catalogos": resumo,
         "shards_baixados": False, "pixels_de_rotulo_lidos": False, "promovido": False,
         "proxima_etapa": "confrontar ativacao, mecanismo, periodo, bacia e footprint com conhecidos/examinados; congelar uso antes dos shards"}
    r["hash_resumo"] = sha256_json(r)
    r["quando_utc"] = agora_utc()
    destino.mkdir(parents=True, exist_ok=True)
    # O inventario local pode evoluir sem que a revisao remota mude.
    snapshot = destino / "auditorias" / r["hash_resumo"].split(":")[-1]
    snapshot.mkdir(parents=True, exist_ok=True)
    arq = snapshot / "manifesto.json"
    if arq.exists():
        antigo = json.loads(arq.read_text(encoding="utf-8"))
        if {k: v for k, v in antigo.items() if k != "quando_utc"} != {k: v for k, v in r.items() if k != "quando_utc"}:
            raise RuntimeError("snapshot existente divergente")
        r = antigo
    else:
        arq.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"GRAVADO={arq}")
    print(json.dumps({n: {k: v for k, v in s.items() if k in ("tiles", "estado", "ativacoes_brasil", "brasil_sem_ativacao_local")}
                      for n, s in resumo.items()}))
    return r


def main() -> int:
    sys.path.insert(0, str(REPO / "scripts/suscetibilidade"))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--saida", type=Path, required=True)
    args = ap.parse_args()
    r = executar(args.saida)
    return int(bool(r["falhas"]))


if __name__ == "__main__":
    raise SystemExit(main())
