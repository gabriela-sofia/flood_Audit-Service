"""Consulta o S2ID (Danos Informados) por UF e intervalo de anos, isolando um município
pelo **código IBGE**, e não pelo nome.

Por que pelo código: o CSV do S2ID vem em latin-1 com o nome acentuado ("Petrópolis"). Casar
por substring sem acento ("Petropolis") devolve **zero** silenciosamente — falso negativo que
passa por "não houve evento". O código IBGE aparece no campo `Protocolo`
(`RJ-F-3303906-13120-20250405`) e não tem esse problema.

Fonte pública, sem login. Adaptado de `s2id_danos_uf.py` (linhagem Petrópolis/COBRADE), que
consultava um ano por vez: janelas maiores que 1 ano com as 65 tipologias fazem o servidor
devolver HTML em vez de CSV, então cada ano continua sendo uma requisição.

Uso:
    python baixar_s2id_municipio.py --uf PR --ibge 4106902 --de 2023 --ate 2026 --outdir <dir>
"""

from __future__ import annotations

import argparse
import csv
import http.cookiejar
import re
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://s2id.mi.gov.br"
PAGE = BASE + "/paginas/relatorios/index.xhtml"

IBGE_CONHECIDOS = {"3303906": "Petropolis", "4106902": "Curitiba", "2611606": "Recife"}


def baixar_ano(uf: str, ano: int, timeout: int = 300) -> bytes:
    """Devolve o CSV bruto de Danos Informados da UF no ano, com as 65 tipologias COBRADE."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", "revp-s2id-probe/2")]

    with opener.open(PAGE, timeout=120) as resp:
        html = resp.read().decode("latin-1")
    action = re.search(r'<form id="form"[^>]*action="([^"]+)"', html).group(1)
    view_state = re.search(r'name="javax\.faces\.ViewState"[^>]*value="([^"]+)"', html).group(1)
    cobrades = re.findall(r'name="abas:sanfonas:cobrades3" id="[^"]+" value="(\d+)"', html)
    if not cobrades:
        raise RuntimeError("nenhuma tipologia COBRADE encontrada na página — layout mudou")

    pairs = [("form", "form"),
             ("abas:sanfonas:j_idt74_input", f"01/01/{ano}"),
             ("abas:sanfonas:dt_final_danos_input", f"31/12/{ano}")]
    pairs += [("abas:sanfonas:cobrades3", c) for c in cobrades]
    pairs += [("abas:sanfonas:estadosDanosInformados", uf),
              ("abas:sanfonas_active", "0"), ("abas_active", "0"),
              ("abas:sanfonas:j_idt95", "abas:sanfonas:j_idt95"),
              ("javax.faces.ViewState", view_state)]

    body = urllib.parse.urlencode(pairs, encoding="latin-1").encode("latin-1")
    req = urllib.request.Request(
        BASE + action, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
    )
    with opener.open(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type") or ""
        raw = resp.read()
    if "csv" not in content_type:
        raise RuntimeError(
            f"S2ID devolveu {content_type!r} ({len(raw)} bytes) em vez de CSV para {uf}/{ano} — "
            "sintoma conhecido de janela grande demais"
        )
    return raw


def registros(raw: bytes) -> list[dict]:
    linhas = raw.decode("latin-1").splitlines()
    cabecalho = linhas[4].split(";")
    return [dict(zip(cabecalho, ln.split(";"))) for ln in linhas[5:] if ln.count(";") > 5]


def filtrar_municipio(regs: list[dict], ibge: str) -> list[dict]:
    return [r for r in regs if ibge in r.get("Protocolo", "")]


def data_do_protocolo(protocolo: str) -> str:
    """`RJ-F-3303906-13120-20250405` -> `2025-04-05`."""
    bruto = protocolo[-8:]
    return f"{bruto[:4]}-{bruto[4:6]}-{bruto[6:]}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uf", required=True)
    p.add_argument("--ibge", required=True, help="código IBGE do município (7 dígitos)")
    p.add_argument("--de", type=int, required=True)
    p.add_argument("--ate", type=int, required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nome = IBGE_CONHECIDOS.get(args.ibge, args.ibge)
    encontrados: list[dict] = []

    for ano in range(args.de, args.ate + 1):
        raw = baixar_ano(args.uf, ano)
        (outdir / f"s2id_{args.uf}_{ano}.csv").write_bytes(raw)
        todos = registros(raw)
        meus = filtrar_municipio(todos, args.ibge)
        print(f"{args.uf} {ano}: {len(todos):>4} registros na UF | {nome}: {len(meus)}")
        for r in meus:
            r["_data_evento"] = data_do_protocolo(r["Protocolo"])
        encontrados.extend(meus)

    encontrados.sort(key=lambda r: r["_data_evento"])
    destino = outdir / f"s2id_{nome.lower()}_{args.de}_{args.ate}.csv"
    if encontrados:
        with destino.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(encontrados[0]))
            writer.writeheader()
            writer.writerows(encontrados)
    print(f"\nTOTAL {nome} {args.de}-{args.ate}: {len(encontrados)} registros -> {destino}")
    for r in encontrados:
        print(f"  {r['_data_evento']}  {r['COBRADE'][:55]:<55} {r['Status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
