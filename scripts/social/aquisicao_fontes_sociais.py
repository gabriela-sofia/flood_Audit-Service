"""Aquisicao auditavel das fontes sociais oficiais do REV-P.

Por padrao o script roda offline: apenas verifica se os arquivos existem e
grava manifesto. Com `--online`, baixa os arquivos publicos oficiais declarados
em `FONTES_SOCIAIS_OFICIAIS`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "local_runs" / "camada-social-fontes" / "ibge"

FONTES_SOCIAIS_OFICIAIS = [
    {
        "fonte_id": "malha_setores_pr_2022",
        "arquivo": "PR_setores_CD2022.zip",
        "url": "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_de_setores_censitarios__divisoes_intramunicipais/censo_2022/setores/shp/UF/PR_setores_CD2022.zip",
        "escopo": "piloto_curitiba",
        "dominios": [
            "geometria_setorial",
            "tipologia_territorial_vulnerabilidade_ibge",
        ],
    },
    {
        "fonte_id": "agregados_basicos_setor_br_20260520",
        "arquivo": "Agregados_por_setores_basico_BR_20260520.zip",
        "url": "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios/Agregados_por_Setor_csv/Agregados_por_setores_basico_BR_20260520.zip",
        "escopo": "brasil_filtrado_por_municipio",
        "dominios": ["exposicao_populacional", "moradia_infraestrutura",
                     "qualidade_dado_censitario"],
    },
    {
        "fonte_id": "renda_responsavel_setor_br_20260508",
        "arquivo": "Agregados_por_setores_renda_responsavel_BR_20260508_csv.zip",
        "url": "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios_Rendimento_do_Responsavel/Agregados_por_setores_renda_responsavel_BR_20260508_csv.zip",
        "escopo": "brasil_filtrado_por_municipio",
        "dominios": ["condicao_socioeconomica"],
    },
    {
        "fonte_id": "dicionario_basico_setores_20260520",
        "arquivo": "dicionario_de_dados_agregados_por_setores_censitarios_20260520.xlsx",
        "url": "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios/dicionario_de_dados_agregados_por_setores_censitarios_20260520.xlsx",
        "escopo": "metadados",
        "dominios": ["proveniencia"],
    },
    {
        "fonte_id": "dicionario_renda_responsavel_20260508",
        "arquivo": "dicionario_de_dados_renda_responsavel_20260508.xlsx",
        "url": "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios_Rendimento_do_Responsavel/dicionario_de_dados_renda_responsavel_20260508.xlsx",
        "escopo": "metadados",
        "dominios": ["proveniencia"],
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def baixar(url: str, destino: Path, timeout: int = 180) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".tmp")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "REV-P social layer audit/2026-09-01"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(destino)


def executar(output_dir: Path = OUT, online: bool = False,
             force: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fonte in FONTES_SOCIAIS_OFICIAIS:
        destino = output_dir / fonte["arquivo"]
        baixado_agora = False
        erro = None
        if online and (force or not destino.exists()):
            try:
                baixar(fonte["url"], destino)
                baixado_agora = True
                time.sleep(0.2)
            except Exception as e:  # pragma: no cover - depende de rede
                erro = f"{type(e).__name__}: {e}"
        rows.append({
            **fonte,
            "existe": destino.exists(),
            "bytes": destino.stat().st_size if destino.exists() else None,
            "sha256": sha256(destino) if destino.exists() else None,
            "baixado_agora": baixado_agora,
            "erro": erro,
        })
    manifesto = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "modo": "online" if online else "offline",
        "output_dir": str(output_dir),
        "fontes": rows,
        "status": "ok" if all(r["existe"] and not r["erro"] for r in rows)
        else "incompleto",
        "nota_metodologica": (
            "baixar fonte nao valida vulnerabilidade; apenas cria provenance "
            "para integracao territorial posterior"),
    }
    (output_dir / "manifesto_aquisicao_fontes_sociais.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "commands.txt").write_text(
        "python scripts/social/aquisicao_fontes_sociais.py\n"
        "python scripts/social/aquisicao_fontes_sociais.py --online\n",
        encoding="utf-8")
    return manifesto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa ou verifica fontes sociais oficiais do REV-P.")
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--online", action="store_true",
                        help="baixa arquivos ausentes das URLs oficiais")
    parser.add_argument("--force", action="store_true",
                        help="baixa novamente mesmo se o arquivo ja existir")
    return parser.parse_args()


def main() -> int:
    ns = parse_args()
    manifesto = executar(Path(ns.output_dir), online=ns.online, force=ns.force)
    print("===AQUISICAO FONTES SOCIAIS===")
    print(f"modo={manifesto['modo']}")
    print(f"status={manifesto['status']}")
    for f in manifesto["fontes"]:
        print(f"{f['fonte_id']}: existe={f['existe']} bytes={f['bytes']}")
    print(f"GRAVADO={Path(ns.output_dir)}")
    print("===END===")
    return 0 if manifesto["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
