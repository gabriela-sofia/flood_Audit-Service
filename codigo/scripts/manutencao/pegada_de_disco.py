"""A pegada em disco da cadeia de terreno, medida, e o que dela e recuperavel.

-------------------------------------------------------------------------
POR QUE ESTE ARQUIVO EXISTE
-------------------------------------------------------------------------

Medido em 21/09/2026: `local_runs` tem 389,67 GB em `ter-01-cadeia-harmonizada`,
contra 91,6 GB livres no disco. O WhiteboxTools grava TODA saida sem compressao e
em float64 -- um tile de 1 grau ocupa 943 MB --, enquanto o DEM de entrada, que o
proprio projeto grava, usa deflate.

Isso nao e so disco. A pasta `Documents` esta espelhada pelo Google Drive (backup
de pasta, conta de 15 GB), entao cada raster novo entra numa fila de upload que
nunca vai caber, e o Drive fica varrendo centenas de gigabytes enquanto a cadeia
roda. Foi o que transformou "rodar tudo junto" em problema de maquina.

-------------------------------------------------------------------------
AS DUAS ECONOMIAS, E ELAS NAO SAO A MESMA COISA
-------------------------------------------------------------------------

  COMPRIMIR    e SEM PERDA e reversivel: os valores continuam identicos, celula a
               celula, e nenhum numero do projeto muda. Medido num tile: 943 MB
               -> 528 MB, com `streams_dinf` caindo a 2,2% do tamanho.

  APAGAR       e decisao humana, e so vale para camada que NINGUEM consome e que
               a propria cadeia sabe refazer. Aqui isso e `pointer_dinf.tif`,
               `streams.tif` e `<nome>_dem_30m.tif`.

`dinf_sca` e `dinf_accum_cells` NAO entram na lista de apagaveis, e essa e a
diferenca que importa: eles sao dois dos cinco canais que o encoder de terreno le
(`dados_terreno.CAMADAS`). Apaga-los economizaria 93 GB e desligaria a frente de
aprendizado que existe justamente para que o dado bruto possa sair um dia.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao apaga nada sem `--apagar-intermediarias`, nao muda valor de celula, nao mexe
em manifesto de cadeia e nao decide politica de sincronizacao de nuvem: ele mede,
comprime sob pedido e diz o que o apagamento custaria para refazer.

Uso:
    python scripts/manutencao/pegada_de_disco.py
    python scripts/manutencao/pegada_de_disco.py --comprimir --limite-gb 20
    python scripts/manutencao/pegada_de_disco.py --apagar-intermediarias
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from proveniencia import agora_utc, sha256_json  # noqa: E402

TER01 = REPO / "local_runs" / "ter-01-cadeia-harmonizada"
SAIDA = REPO / "local_runs" / "manutencao-pegada"

# Quem consome cada camada. Levantado por varredura em `scripts/` e `tests/` em
# 21/09/2026; a coluna existe para que apagar deixe de ser palpite.
CONSUMIDAS = {
    "dem_filled.tif": "gfd02, cems03, resolver_camadas, svc03, encoder (canal)",
    "hand_dinf.tif": "gfd02, cems03, resolver_camadas, svc03, lentes, encoder",
    "slope_deg_wbt.tif": "gfd02, cems03, resolver_camadas, svc03, encoder",
    "twi_dinf.tif": "gfd02, cems03, svc01..svc04, br04 e a tabela unica inteira",
    "streams_dinf.tif": "gfd02/cems03 (`sobre_canal_modelado`), lentes, svc08",
    "dinf_sca.tif": "encoder de terreno (canal, em log) e substrato_multimodal",
    "dinf_accum_cells.tif": "encoder de terreno (canal, em log)",
    "agua_permanente_jrc.tif": "gfd02, cems03 (agua permanente sai das 2 classes)",
    "s1_grd.tif": "substrato multimodal SAR",
}

# Camada sem consumidor, que a cadeia refaz a partir do que fica em disco.
INTERMEDIARIAS = {
    "pointer_dinf.tif": (
        "ponteiro D-infinity: so quem o produz o cita; `derivar_aoi_nova` o "
        "refaz do `dem_filled` com `d_inf_pointer` (sem rede)"),
    "streams.tif": (
        "canal por limiar de celulas antes do `streams_dinf`; refeito por "
        "`extract_streams` sobre o `dinf_accum_cells` (sem rede)"),
    "_dem_30m.tif": (
        "DEM reprojetado a 30 m, entrada do preenchimento; refeito a partir do "
        "FABDEM em `local_runs/mdt-fabdem` (sem rede, se o FABDEM ficar)"),
}

JA_COMPRIMIDO = ("deflate", "lzw", "zstd", "packbits")

# Arquivo intermediario que NAO se apaga, e a razao e um teste.
# `test_aquisicao_de_dado_terreno` reproduz a cadeia inteira a partir deste DEM
# e compara camada a camada com o que esta no disco. Sem ele o teste PULA -- e
# pulo permanente e silencioso e o que o proprio conftest chama de pior que
# teste ausente. Sao 18 MB para manter a prova de reproducao de pe.
PRESERVAR = {"EMSR734_AOI02_dem_30m.tif"}


def _classe(nome: str) -> str:
    if nome.endswith("_dem_30m.tif"):
        return "intermediaria"
    if nome in INTERMEDIARIAS:
        return "intermediaria"
    if nome in CONSUMIDAS:
        return "consumida"
    return "outra"


def medir() -> dict:
    """Bytes por camada e por classe, sem abrir raster nenhum."""
    por_camada: dict = {}
    for pasta in sorted(p for p in TER01.iterdir() if p.is_dir()):
        for arq in pasta.glob("*.tif"):
            chave = ("_dem_30m.tif" if arq.name.endswith("_dem_30m.tif")
                     else arq.name)
            d = por_camada.setdefault(
                chave, {"bytes": 0, "arquivos": 0, "classe": _classe(chave)})
            d["bytes"] += arq.stat().st_size
            d["arquivos"] += 1
    total = sum(d["bytes"] for d in por_camada.values())
    por_classe: dict = {}
    for chave, d in por_camada.items():
        c = por_classe.setdefault(d["classe"], {"bytes": 0, "arquivos": 0})
        c["bytes"] += d["bytes"]
        c["arquivos"] += d["arquivos"]
    return {"por_camada": por_camada, "por_classe": por_classe, "total": total}


def _comprimir_um(arq: Path) -> dict:
    """Reescreve com deflate, conferindo que a matriz continua identica.

    O arquivo so e substituido depois de lido de volta e comparado. Se a
    comparacao falhar, o temporario e descartado e o original fica intacto --
    perder 40% de disco nao vale um pixel diferente.
    """
    import numpy as np
    import rasterio

    with rasterio.open(arq) as src:
        if str(src.profile.get("compress", "")).lower() in JA_COMPRIMIDO:
            return {"estado": "ja_comprimido", "antes": arq.stat().st_size,
                    "depois": arq.stat().st_size}
        dados = src.read(1)
        perfil = src.profile.copy()

    perfil.update(compress="deflate", zlevel=6, tiled=True,
                  blockxsize=256, blockysize=256,
                  predictor=3 if str(perfil["dtype"]).startswith("float") else 2)
    temporario = arq.with_suffix(".tif.comprimindo")
    with rasterio.open(temporario, "w", **perfil) as fora:
        fora.write(dados, 1)
    with rasterio.open(temporario) as conferindo:
        devolta = conferindo.read(1)

    iguais = np.array_equal(dados, devolta, equal_nan=True)
    if not iguais:
        temporario.unlink(missing_ok=True)
        return {"estado": "recusado_valor_mudou", "antes": arq.stat().st_size,
                "depois": arq.stat().st_size}
    antes = arq.stat().st_size
    depois = temporario.stat().st_size
    os.replace(temporario, arq)
    return {"estado": "comprimido", "antes": antes, "depois": depois}


def comprimir(limite_gb: float | None, so_camadas: list | None) -> dict:
    alvos = []
    for pasta in sorted(p for p in TER01.iterdir() if p.is_dir()):
        for arq in sorted(pasta.glob("*.tif")):
            chave = ("_dem_30m.tif" if arq.name.endswith("_dem_30m.tif")
                     else arq.name)
            if so_camadas and chave not in so_camadas:
                continue
            alvos.append((arq, chave))
    # Maiores primeiro: com orcamento de tempo curto, o ganho vem antes.
    alvos.sort(key=lambda t: t[0].stat().st_size, reverse=True)

    registro = {"comprimido": 0, "ja_comprimido": 0, "recusado_valor_mudou": 0,
                "erro": 0}
    antes = depois = 0
    lidos = 0
    t0 = time.time()
    for arq, chave in alvos:
        if limite_gb and antes / 1e9 >= limite_gb:
            break
        try:
            r = _comprimir_um(arq)
        except Exception as e:                                   # noqa: BLE001
            registro["erro"] += 1
            print(f"  [erro] {arq.parent.name}/{arq.name}: {type(e).__name__}",
                  flush=True)
            continue
        registro[r["estado"]] += 1
        # O orcamento conta so o que foi REESCRITO. Contando o que ja estava
        # comprimido, uma fatia gastava a cota sem comprimir nada e o laco de
        # fora concluia "nada mais a comprimir" -- aconteceu em 21/09/2026.
        if r["estado"] == "comprimido":
            antes += r["antes"]
            depois += r["depois"]
            lidos += 1
        if lidos % 25 == 0:
            print(f"  {lidos}/{len(alvos)} | {antes/1e9:.1f} GB lidos -> "
                  f"{depois/1e9:.1f} GB | {time.time()-t0:.0f}s", flush=True)
    return {"arquivos_vistos": lidos, "por_estado": registro,
            "bytes_antes": antes, "bytes_depois": depois,
            "segundos": round(time.time() - t0, 1)}


def apagar_intermediarias(confirmar: bool) -> dict:
    alvos = []
    for pasta in sorted(p for p in TER01.iterdir() if p.is_dir()):
        for arq in pasta.glob("*.tif"):
            if arq.name in PRESERVAR:
                continue
            if _classe("_dem_30m.tif" if arq.name.endswith("_dem_30m.tif")
                       else arq.name) == "intermediaria":
                alvos.append(arq)
    bytes_ = sum(a.stat().st_size for a in alvos)
    if not confirmar:
        return {"apagados": 0, "arquivos": len(alvos), "bytes": bytes_,
                "estado": "DRY RUN -- use --apagar-intermediarias --confirmar"}
    n = 0
    for a in alvos:
        a.unlink()
        n += 1
    return {"apagados": n, "bytes": bytes_, "estado": "apagado"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--comprimir", action="store_true")
    p.add_argument("--limite-gb", type=float, default=None,
                   help="para depois de ler este tanto de dado")
    p.add_argument("--camada", nargs="*",
                   help="comprime so estas camadas (ex.: streams_dinf.tif)")
    p.add_argument("--apagar-intermediarias", action="store_true")
    p.add_argument("--confirmar", action="store_true")
    a = p.parse_args()

    if not TER01.exists():
        print(f"ABORTADO: {TER01} ausente")
        return 1

    m = medir()
    print("=== PEGADA DA CADEIA DE TERRENO ===")
    for chave, d in sorted(m["por_camada"].items(),
                           key=lambda kv: -kv[1]["bytes"]):
        print(f"  {chave:26s} {d['bytes']/1e9:7.2f} GB  {d['arquivos']:5d} "
              f"arq  {d['classe']}")
    print(f"  {'TOTAL':26s} {m['total']/1e9:7.2f} GB")
    for classe, d in sorted(m["por_classe"].items()):
        print(f"    {classe:24s} {d['bytes']/1e9:7.2f} GB")

    saida = {"tipo": "pegada_de_disco_da_cadeia_de_terreno",
             "quando_utc": agora_utc(), "medida": m,
             "consumidas": CONSUMIDAS, "intermediarias": INTERMEDIARIAS,
             "nao_e": ("nao decide apagar, nao muda valor de celula e nao "
                       "configura sincronizacao de nuvem: mede, comprime sem "
                       "perda sob pedido e declara o custo de refazer")}

    if a.comprimir:
        print("\n=== COMPRIMINDO (sem perda, com conferencia) ===")
        saida["compressao"] = comprimir(a.limite_gb, a.camada)
        r = saida["compressao"]
        if r["bytes_antes"]:
            print(f"  {r['bytes_antes']/1e9:.2f} GB -> "
                  f"{r['bytes_depois']/1e9:.2f} GB "
                  f"({100*r['bytes_depois']/r['bytes_antes']:.1f}%) em "
                  f"{r['segundos']:.0f}s | {r['por_estado']}")

    if a.apagar_intermediarias:
        print("\n=== INTERMEDIARIAS ===")
        saida["apagamento"] = apagar_intermediarias(a.confirmar)
        print(f"  {saida['apagamento']}")

    saida["hash_resumo"] = sha256_json(
        {k: v for k, v in saida.items() if k != "quando_utc"})
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "pegada.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGRAVADO={SAIDA / 'pegada.json'}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
