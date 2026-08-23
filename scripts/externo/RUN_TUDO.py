"""
RUN_TUDO -- orquestrador da frente externa. Um comando, cadeia inteira.

Executa na ordem de dependencia, pula o que ja esta pronto, e nao interrompe a
cadeia quando uma etapa depende de dado ausente -- so registra e segue. Isso
importa porque a frente tem etapas que precisam de rede (download) e etapas
que nao (analise), e travar tudo por causa de um download pendente ja custou
horas neste projeto.

CADEIA:
  aquisicao      ext_uk01 (outlines) -> ext_uk04 (flood zones)
                 aq_mestre (DEM/HAND/WorldCover/GSW)
                 n1_rodar_tudo (Nivel 1) -> aq_rs01 (features do RS)
  organizacao    ext_uk07 (GeoParquet)
  analise        ext_uk02/03 (AOI) -> ext_uk05 (contabilidade)
                 ft_uk01 (rasters) -> ft_uk02 (mascaras) -> ft_uk03 (amostra)
                 ft_uk04 (chuva) -> n1f (reducao a pontos)
  modelo         ds01 (dataset) -> mod_uk00 (fumaca) -> mod_uk01 (Firth)

Uso:
    python scripts/externo/RUN_TUDO.py --etapas analise,modelo
    python scripts/externo/RUN_TUDO.py --tudo
    python scripts/externo/RUN_TUDO.py --status
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
RUNS = REPO / "local_runs"
UK = AQUI / "uk"
AQ = AQUI / "aquisicao"
PT = AQUI / "pontos"
CEMS = AQUI / "cems"

# (grupo, script, args, artefato que comprova conclusao, precisa_de_rede)
CADEIA = [
    ("aquisicao", UK / "ext_uk01_baixar_recorded_flood_outlines.py", [],
     RUNS / "ext-uk-01-aquisicao" / "inventario.json", True),
    ("aquisicao", UK / "ext_uk04_zonas_inundacao_aoi.py", [],
     RUNS / "ext-uk-04-flood-zones" / "resumo.json", True),
    ("aquisicao", AQ / "aq_mestre.py", ["--baixar"],
     RUNS / "aq-features" / "download.json", True),
    ("aquisicao", AQ / "n1_rodar_tudo.py", [],
     RUNS / "n1-relatorio-consolidado.md", True),
    ("aquisicao", PT / "aq_rs01_variaveis.py", ["--baixar"],
     RUNS / "aq-rs-features" / "PROVENIENCIA.md", True),

    ("organizacao", UK / "ext_uk07_converter_geoparquet.py", [],
     RUNS / "geostore" / "outlines_elegiveis.parquet", False),

    ("analise", UK / "ext_uk02_selecionar_area_piloto.py", [],
     RUNS / "ext-uk-02-area-piloto" / "selecao.json", False),
    ("analise", UK / "ext_uk03_dimensionar_aoi.py", [],
     RUNS / "ext-uk-03-aoi" / "dimensionamento.json", False),
    ("analise", UK / "ext_uk05_contabilidade_negativo.py", [],
     RUNS / "ext-uk-05-contabilidade" / "contabilidade.json", False),
    ("analise", UK / "ft_uk01_preparar_rasters_aoi.py", [],
     RUNS / "ft-uk-01-rasters" / "resumo.json", False),
    ("analise", UK / "ft_uk02_mascaras_decisao.py", [],
     RUNS / "ft-uk-02-mascaras" / "resumo.json", False),
    ("analise", UK / "ft_uk03_amostrar_e_extrair.py", [],
     RUNS / "ft-uk-03-amostra" / "resumo.json", False),
    ("analise", UK / "ft_uk04_variaveis_chuva.py", [],
     RUNS / "ft-uk-04-chuva" / "resumo.json", False),
    ("analise", PT / "n1f_reduzir_a_pontos.py", [],
     RUNS / "n1f-pontos" / "resumo.json", False),
    ("analise", PT / "n1g_variaveis_nos_pontos.py", [],
     RUNS / "n1g-features" / "resumo.json", False),
]


def status() -> None:
    print("===STATUS_DA_CADEIA===")
    grupo_atual = None
    for grupo, script, _a, artefato, rede in CADEIA:
        if grupo != grupo_atual:
            print(f"--- {grupo.upper()} ---")
            grupo_atual = grupo
        # ATE 09/08/2026 este status so checava existencia de arquivo, e por
        # isso marcava OK para etapas que tinham parado no meio (CHIRPS com
        # 28 de 97 meses, EMSR857 sem tabela de pontos). Agora checa conteudo.
        pronto = artefato.exists() and artefato.stat().st_size > 0
        nota = ""
        if pronto and artefato.suffix == ".json":
            try:
                d = json.loads(artefato.read_text(encoding="utf-8"))
                for chave, esperado in (("meses", None), ("n_pontos", None)):
                    if chave in d and not d[chave]:
                        pronto = False
                rel = d.get("relatorio") or {}
                for fl, info in (rel.items() if isinstance(rel, dict) else []):
                    if isinstance(info, dict) and "meses_completos" in info:
                        c, alvo = info["meses_completos"], info.get("meses_alvo", 0)
                        if alvo and c < alvo:
                            pronto = False
                            nota = f"PARCIAL {c}/{alvo}"
            except Exception:  # noqa: BLE001
                pass
        marca = "OK  " if pronto else ("REDE" if rede else "PEND")
        print(f"  {marca}  {script.name:48s} "
              f"{nota or ('' if pronto else '-> ' + artefato.name)}")
    print("===END===")


def main() -> int:
    args = sys.argv[1:]
    if "--status" in args or not args:
        status()
        return 0

    grupos = {"aquisicao", "organizacao", "analise", "modelo"}
    if "--etapas" in args:
        grupos = set(args[args.index("--etapas") + 1].split(","))

    print("===RUN_TUDO===")
    resultados = []
    for grupo, script, extra, artefato, rede in CADEIA:
        if grupo not in grupos:
            continue
        if artefato.exists():
            print(f"[PULADO] {script.name} (artefato ja existe)")
            resultados.append((script.name, "pulado"))
            continue
        if not script.exists():
            print(f"[AUSENTE] {script.name}")
            resultados.append((script.name, "ausente"))
            continue
        print(f"[RODANDO] {script.name} {' '.join(extra)}", flush=True)
        t0 = time.time()
        p = subprocess.run([sys.executable, str(script)] + extra,
                           cwd=str(REPO), capture_output=True, text=True)
        dt = (time.time() - t0) / 60
        cauda = "\n".join((p.stdout or "").splitlines()[-6:])
        print(cauda)
        estado = "ok" if p.returncode == 0 else f"falha({p.returncode})"
        print(f"[{estado}] {script.name} em {dt:.1f}min")
        if p.returncode != 0 and p.stderr:
            print("  " + (p.stderr.strip().splitlines() or ["?"])[-1][:160])
        resultados.append((script.name, estado))

    print("---RESUMO---")
    for nome, estado in resultados:
        print(f"  {estado:12s} {nome}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
