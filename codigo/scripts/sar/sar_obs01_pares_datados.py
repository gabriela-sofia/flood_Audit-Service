"""Adquire pares SAR datados sem consumir reservas ou alterar treinos ativos.

O catalogo define identidade; os nomes publicados definem as datas SAR.
A data da delineacao anterior nao e a data da imagem anterior.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import tarfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path, PurePosixPath

import pandas as pd
import pyarrow.parquet as pq
from shapely import wkb

REPO = Path(__file__).resolve().parents[2]
for sub in ("geo", "metodo", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / sub))
from obs01_aquisicao_dirigida import baixar, gravar, sha256_arquivo, sha256_json  # noqa: E402
from ds06_dataset_de_treino import versao_atual  # noqa: E402
import verdade_observacional as vo  # noqa: E402

REVISAO = "868407460bf3db492f50730a57585916baa71dc6"
BASE = f"https://huggingface.co/datasets/links-ads/geoid-flood/resolve/{REVISAO}/"
RESERVADAS = {"EMSR868", "EMSR871"}
COLUNAS = ["event_id", "tile_id", "bbox_id", "geometry", "utm_crs", "event_time",
           "delineation_time_pre", "delineation_time_post", "countries", "split"]
PADRAO = re.compile(r"^(EMSR\d+-\d+-\d+)_s1grd_(pre|post)_(\d{8}T\d{6})\.tif$")


def datas_validas(antes, evento, depois):
    a, e, d = (pd.Timestamp(x) for x in (antes, evento, depois))
    if any(pd.isna(x) for x in (a, e, d)):
        return False
    return a < e <= d and (e-a).total_seconds() <= 90*86400 and (d-e).total_seconds() <= 14*86400


def membros_seguros(shard):
    if not shard["path"].startswith("geoid-flood/shards/train/"):
        raise ValueError("somente shards de desenvolvimento; heldout proibido")
    for nome in shard["members"]:
        p = PurePosixPath(nome)
        if p.is_absolute() or ".." in p.parts or "\\" in nome:
            raise ValueError("membro inseguro")
        if any(nome.startswith(x + "-") for x in RESERVADAS):
            raise ValueError("shard contem reserva protegida")


def referencias():
    ds = versao_atual() / "pontos.parquet"
    conhecidos = REPO / "local_runs/reservas-examinadas/conhecidos_2026_10_03.json"
    examinados = REPO / "local_runs/reservas-examinadas/examinados_apos_congelamento.jsonl"
    fontes = {str(p.resolve()): sha256_arquivo(p) for p in (ds, conhecidos, examinados)}
    partes = [vo.janelas_de_eventos(b.to_pandas()) for b in pq.ParquetFile(ds).iter_batches(
        batch_size=65536, columns=["fonte", "evento_id", "janela_inicio", "janela_fim", "lat", "lon"])]
    r = pd.concat(partes, ignore_index=True).groupby(["fonte", "evento_id"], as_index=False).agg(
        inicio=("inicio", "min"), fim=("fim", "max"), lat_min=("lat_min", "min"),
        lat_max=("lat_max", "max"), lon_min=("lon_min", "min"), lon_max=("lon_max", "max"))
    r = pd.concat([r, vo.consultados()], ignore_index=True)
    ids = {x.split(":", 1)[-1] for vs in json.loads(conhecidos.read_text(encoding="utf-8"))["por_fonte"].values() for x in vs}
    ids.update(p.name.split("_", 1)[0] for p in (REPO / "local_runs/cems-05-pacotes").glob("*_products.zip"))
    ids.update(str(x).split(":", 1)[-1] for x in r.evento_id)
    return r, ids | RESERVADAS, fontes


def selecionar(catalogo, indice, refs, conhecidos, eventos=4, tiles=4):
    metadados = {}
    for row in catalogo.to_dict("records"):
        chave = f"{row['event_id']}-{row['tile_id']}-{row['bbox_id']}"
        if chave in metadados:
            raise ValueError("tile com identidade duplicada")
        metadados[chave] = row
    pares = {}
    # Os primeiros membros reduzem trafego de cabecalhos, sem olhar rotulos.
    for s in indice["shards"]:
        if not s["path"].startswith("geoid-flood/shards/train/s1grd/"):
            continue
        membros_seguros(s)
        for ordem, nome in enumerate(s["members"]):
            m = PADRAO.fullmatch(PurePosixPath(nome).name)
            if m and ordem < 40:
                chave, fase, data = m.groups()
                pares.setdefault(chave, {})[fase] = {"shard": s["path"], "membro": nome,
                    "data": datetime.strptime(data, "%Y%m%dT%H%M%S").isoformat(), "ordem": ordem}
    elegiveis = []
    for chave, imagens in pares.items():
        row = metadados.get(chave)
        if set(imagens) != {"pre", "post"} or row is None or row["split"] != "train":
            continue
        if row["event_id"] in conhecidos or pd.Timestamp(row["event_time"]).year < 2020:
            continue
        if not datas_validas(imagens["pre"]["data"], row["event_time"], imagens["post"]["data"]):
            continue
        distancia = abs((pd.Timestamp(imagens["post"]["data"])-pd.Timestamp(row["delineation_time_post"])).total_seconds())
        if not pd.notna(distancia) or distancia > 24*3600:
            continue
        bbox = wkb.loads(row["geometry"]).bounds
        janela = pd.DataFrame([{"fonte": "geoid_sar_novo", "evento_id": "CEMS:" + row["event_id"],
            "inicio": pd.Timestamp(row["event_time"][:10]), "fim": pd.Timestamp(imagens["post"]["data"][:10]),
            "lon_min": bbox[0], "lat_min": bbox[1], "lon_max": bbox[2], "lat_max": bbox[3]}])
        if vo.equivalencias(janela, refs):
            continue
        alvos = {}
        for camada in ("label", "validity"):
            nomes = [(s["path"], n) for s in indice["shards"]
                     if s["path"].startswith(f"geoid-flood/shards/train/{camada}/")
                     for n in s["members"] if PurePosixPath(n).name.startswith(chave + "_")]
            if len(nomes) != 1:
                break
            alvos[camada] = {"shard": nomes[0][0], "membro": nomes[0][1]}
        if len(alvos) != 2:
            continue
        elegiveis.append({"chave": chave, "evento": row["event_id"], "pais": row["countries"],
            "evento_data": row["event_time"], "bbox": list(bbox), "utm_crs": row["utm_crs"],
            "referencia_pre_data": row["delineation_time_pre"], "rotulo_pos_data": row["delineation_time_post"],
            "arquivos": imagens | alvos,
            "custo_ordem": max(x["ordem"] for x in imagens.values())})
    ordem_eventos = sorted({x["evento"] for x in elegiveis}, key=lambda e: (
        min(x["custo_ordem"] for x in elegiveis if x["evento"] == e), sha256_json(e)))
    escolhidos = []
    for e in ordem_eventos:
        linhas = sorted((x for x in elegiveis if x["evento"] == e), key=lambda x: (x["custo_ordem"], x["chave"]))[:tiles]
        if len(linhas) < tiles:
            continue
        # Ativacoes proximas no mesmo episodio nao aumentam a contagem.
        if escolhidos:
            a = [dict(fonte="novo", evento_id=x["evento"], inicio=pd.Timestamp(x["evento_data"][:10]),
                      fim=pd.Timestamp(x["arquivos"]["post"]["data"][:10]),
                      lon_min=x["bbox"][0], lat_min=x["bbox"][1], lon_max=x["bbox"][2], lat_max=x["bbox"][3]) for x in escolhidos]
            b = [dict(fonte="candidato", evento_id=x["evento"], inicio=pd.Timestamp(x["evento_data"][:10]),
                      fim=pd.Timestamp(x["arquivos"]["post"]["data"][:10]),
                      lon_min=x["bbox"][0], lat_min=x["bbox"][1], lon_max=x["bbox"][2], lat_max=x["bbox"][3]) for x in linhas]
            if vo.equivalencias(pd.DataFrame(b), pd.DataFrame(a)):
                continue
        escolhidos.extend(linhas)
        if len({x["evento"] for x in escolhidos}) == eventos:
            break
    return escolhidos


class LeitorHTTP(io.RawIOBase):
    """Tar seekavel: so transfere o intervalo pedido e confere Content-Range."""
    def __init__(self, url, tamanho, teto=512*1024**2, orcamento=None):
        self.url, self.tamanho, self.teto = url, tamanho, teto
        self.pos = self.bytes = self.pedidos = 0
        self.orcamento = orcamento if orcamento is not None else {"solicitados": 0}

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, pos, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("origem de seek invalida")
        destino = pos + (self.pos if whence == 1 else self.tamanho if whence == 2 else 0)
        if not 0 <= destino <= self.tamanho:
            raise ValueError("seek fora do shard")
        self.pos = destino
        return destino

    def read(self, tamanho=-1):
        if tamanho < 0 or tamanho > 16*1024**2:
            raise ValueError("leitura sem limite ou grande demais")
        tamanho = min(tamanho, self.tamanho-self.pos)
        if tamanho == 0:
            return b""
        ini, fim = self.pos, self.pos+tamanho-1
        url = self.url + f"?revp_range={ini}-{fim}"
        for tentativa in range(3):
            if self.orcamento["solicitados"] + tamanho > self.teto:
                raise ValueError("orcamento de rede esgotado")
            # Reservar tambem tentativas interrompidas, cujo trafego e incerto.
            self.orcamento["solicitados"] += tamanho
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={ini}-{fim}",
                    "Accept-Encoding": "identity", "User-Agent": "REV-P-SAR-observation/1.0"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    if r.status != 206 or r.headers.get("Content-Range") != f"bytes {ini}-{fim}/{self.tamanho}":
                        raise ValueError("servidor ignorou ou alterou Range")
                    b = r.read(tamanho+1)
                    self.bytes += len(b)
                    self.pedidos += 1
                    if len(b) != tamanho:
                        raise ValueError("intervalo truncado ou excedido")
                self.pos += tamanho
                return b
            except OSError:
                if tentativa == 2:
                    raise
                time.sleep(2**tentativa)
        raise RuntimeError("leitura HTTP nao concluida")


def extrair_selecionados(shard, nomes, destino, teto, orcamento=None):
    membros_seguros(shard)
    if not set(nomes).issubset(shard["members"]):
        raise ValueError("arquivo ausente do indice congelado")
    orcamento = orcamento if orcamento is not None else {"solicitados": 0}
    integral = shard["size"] <= 96*1024**2 and "/s1grd/" not in shard["path"]
    # Rotulos compactos: um download com checksum evita milhares de requests.
    if integral:
        bruto = destino.parent / "shards-compactos" / PurePosixPath(shard["path"]).name
        novo = not bruto.exists()
        if novo:
            if orcamento["solicitados"] + shard["size"] > teto:
                raise ValueError("orcamento de rede esgotado")
            orcamento["solicitados"] += shard["size"]
        registro = baixar(BASE + shard["path"], bruto, shard["size"], tamanho=shard["size"])
        if registro["sha256"].split(":")[-1] != shard["sha256"].split(":")[-1]:
            raise ValueError("SHA256 publicado do shard divergente")
        leitor = bruto.open("rb")
        bytes_rede, pedidos = (shard["size"] if novo else 0), 0
    else:
        leitor = LeitorHTTP(BASE + shard["path"], shard["size"], teto, orcamento)
        bytes_rede, pedidos = 0, 0
    encontrados = []
    with leitor, tarfile.open(fileobj=leitor, mode="r:") as t:
        for membro in t:
            if membro.name not in nomes:
                continue
            if not membro.isfile() or not 0 < membro.size <= 16*1024**2:
                raise ValueError("membro invalido ou fora do limite")
            alvo = (destino / membro.name).resolve()
            if not alvo.is_relative_to(destino.resolve()):
                raise ValueError("membro sai da pasta de aquisicao")
            meta = alvo.with_suffix(alvo.suffix + ".json")
            if alvo.exists():
                r = json.loads(meta.read_text(encoding="utf-8"))
                if r["membro"] != membro.name or r["shard"] != shard["path"] or r["bytes"] != alvo.stat().st_size or sha256_arquivo(alvo) != r["sha256"]:
                    raise ValueError("cache de membro divergente")
            else:
                alvo.parent.mkdir(parents=True, exist_ok=True)
                parcial = alvo.with_suffix(alvo.suffix + ".part")
                try:
                    with t.extractfile(membro) as entrada, parcial.open("wb") as saida:
                        while b := entrada.read(1024*1024):
                            saida.write(b)
                    if parcial.stat().st_size != membro.size:
                        raise ValueError("membro truncado")
                    r = {"arquivo": str(alvo), "membro": membro.name, "shard": shard["path"],
                         "bytes": membro.size, "offset_tar": membro.offset_data,
                         "sha256": sha256_arquivo(parcial), "revisao": REVISAO,
                         "sha256_shard_publicado": shard["sha256"], "shard_integral_verificado": integral,
                         "integridade": "SHA256 integral publicado e SHA256 local do membro" if integral else
                             "HTTPS, revisao fixa, intervalo e cabecalho TAR conferidos; SHA256 local do membro"}
                    gravar(meta, r)
                    parcial.replace(alvo)
                except BaseException:
                    parcial.unlink(missing_ok=True)
                    raise
            encontrados.append(r)
            if len(encontrados) == len(nomes):
                break
    if len(encontrados) != len(nomes):
        raise ValueError("indice e conteudo do shard divergem")
    return encontrados, bytes_rede if integral else leitor.bytes, pedidos if integral else leitor.pedidos


def executar(saida, adquirir=False, eventos=4, tiles=4, plano=None):
    raiz = Path(saida).resolve()
    if plano:
        p = json.loads(Path(plano).read_text(encoding="utf-8"))
        pasta = Path(plano).resolve().parent
        if p["codigo_hash"] != sha256_arquivo(Path(__file__)):
            raise ValueError("codigo diverge do plano; nova versao obrigatoria")
        if pasta.name != sha256_json(p).split(":")[-1] or p["tipo"] != "plano_sar_datado":
            raise ValueError("plano alterado")
    else:
        arq = REPO / f"local_runs/geoid-catalogo-publico/{REVISAO}/geoid-flood/tile_catalog.parquet"
        registro = baixar(BASE + "shard_index.json.gz", raiz / REVISAO / "shard_index.json.gz", 4*1024**2)
        with gzip.open(registro["arquivo"], "rt", encoding="utf-8") as f:
            conteudo = f.read(64*1024**2+1)
        if len(conteudo) > 64*1024**2:
            raise ValueError("indice descompactado excede limite")
        indice = json.loads(conteudo)
        refs, conhecidos, fontes = referencias()
        fontes[str(arq.resolve())] = sha256_arquivo(arq)
        fontes[registro["arquivo"]] = registro["sha256"]
        escolhidos = selecionar(pd.read_parquet(arq, columns=COLUNAS), indice, refs, conhecidos, eventos, tiles)
        s = {x["arquivos"][k]["shard"] for x in escolhidos for k in x["arquivos"]}
        p = {"tipo": "plano_sar_datado", "versao": 2, "revisao": REVISAO,
             "codigo_hash": sha256_arquivo(Path(__file__)), "referencias": fontes, "pares": escolhidos,
             "shards": [x for x in indice["shards"] if x["path"] in s],
             "selecao": "train, eventos desde 2020 sem identidade/equivalencia conhecida; datas SAR nos nomes; 90 dias antes/14 depois; rotulo e SAR posterior separados por no maximo 24h; membros iniciais por custo, sem ler qualidade ou rotulos",
             "eventos_solicitados": eventos, "tiles_por_evento": tiles, "reservas_excluidas": sorted(RESERVADAS),
             "papel": "novos contextos de desenvolvimento; nao confirmacao independente do nucleo fisico",
             "licenca": "CC-BY-4.0, modified Copernicus Sentinel-1/CEMS; nao adquirir DEM",
             "limite_rede_bytes": 512*1024**2, "promovido": False, "rotulos_lidos": False}
        pasta = raiz / sha256_json(p).split(":")[-1]
        gravar(pasta / "plano.json", p)
    if any(sha256_arquivo(a) != h for a, h in p["referencias"].items()):
        raise ValueError("referencias mudaram: nao usar plano antigo")
    print(f"PLANO={pasta / 'plano.json'}", flush=True)
    print(json.dumps({"eventos": sorted({x["evento"] for x in p["pares"]}), "pares": len(p["pares"])}), flush=True)
    if not adquirir:
        return p
    arquivos, falhas, trafego, pedidos = [], [], 0, 0
    orcamento = {"solicitados": 0}
    for shard in p["shards"]:
        nomes = {a["membro"] for x in p["pares"] for a in x["arquivos"].values() if a["shard"] == shard["path"]}
        try:
            r, b, n = extrair_selecionados(shard, nomes, pasta / "dados", p["limite_rede_bytes"], orcamento)
            arquivos.extend(r)
            trafego += b
            pedidos += n
            print(json.dumps({"shard": shard["path"], "arquivos": len(r), "bytes_transferidos": b}), flush=True)
        except (OSError, ValueError, tarfile.TarError) as exc:
            falhas.append({"shard": shard["path"], "erro": str(exc)})
    if any(sha256_arquivo(a) != h for a, h in p["referencias"].items()):
        raise ValueError("inventario mudou durante a aquisicao")
    r = {"tipo": "aquisicao_sar_datada", "plano_sha256": sha256_arquivo(pasta / "plano.json"),
         "arquivos": arquivos, "falhas": falhas, "bytes_transferidos": trafego, "pedidos_range": pedidos,
         "bytes_reservados_incluindo_tentativas": orcamento["solicitados"],
         "reservas_consumidas": False, "admitido_no_treino": False, "promovido": False}
    alvo = pasta / "aquisicoes" / sha256_json(r).split(":")[-1] / "manifesto.json"
    gravar(alvo, r)
    print(f"AQUISICAO={alvo}", flush=True)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--saida", type=Path, default=REPO / "local_runs/sar-observacoes-datadas")
    ap.add_argument("--eventos", type=int, default=4)
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--plano", type=Path)
    ap.add_argument("--baixar", action="store_true")
    a = ap.parse_args()
    if a.eventos < 1 or a.tiles < 1:
        ap.error("eventos e tiles devem ser positivos")
    r = executar(a.saida, a.baixar, a.eventos, a.tiles, a.plano)
    raise SystemExit(int(bool(r.get("falhas"))))
