"""Monta o dataset final de pontos positivos de Curitiba a partir do SIAC 156 -- mesma regra de
aceitação do dataset v12 de Recife (`dataset_eventos_variaveis_v12_final.csv`): geocodificação
strong OU medium entra em bloco como positivo, sem segunda camada de verificação manual
individual (essa é a regra real usada em Recife -- ver
`PROJETO/local_runs/treino_exploratorio_diagnostico_v2/_scratch/stage1_build_points_v2.py:124`,
comentário "Recife POSITIVES: same geocoded points as v1 (unchanged)").

Cada linha carrega proveniência completa (`qa_record_id`, `ano_fonte`, `data_criacao`,
`assunto`/`subdivisao`, `nivel_confianca`, `nominatim_nome_exibicao`, `nominatim_osm_id`) e um
campo explícito `occurrence_phenomenon` confirmando exclusão de deslizamento (mesma prática do
GeoJSON final de Recife).

`failed` nunca entra -- nenhuma coordenada fabricada.

Uso:
    python montar_dataset_final.py --geocoded geocodificado_full.csv --out dataset_positivos_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_dataset(geocoded_rows: list[dict]) -> list[dict]:
    out = []
    for row in geocoded_rows:
        if row.get("nivel_confianca") not in ("strong", "medium"):
            continue  # failed -- nao fabrica coordenada
        out.append(
            {
                "regiao": "Curitiba",
                "rotulo": 1,
                "ponto_id": f"CUR_SIAC156_{row['registro_qa_id']}",
                "lat": row["lat"],
                "lon": row["lon"],
                "data_evento": row["data_criacao"],
                "ano_fonte": row["ano_fonte"],
                "assunto": row["assunto"],
                "subdivisao": row["subdivisao"],
                "logradouro": row["logradouro"],
                "bairro": row["bairro"],
                "nivel_confianca": row["nivel_confianca"],
                "nominatim_nome_exibicao": row["nominatim_nome_exibicao"],
                "nominatim_osm_id": row["osm_id"],
                "n_grupo_duplicata": row["n_grupo_duplicata"],
                "fenomeno_ocorrencia": "flood (alagamento/enchente) - landslide excluded by category filter (SIAC 156 Assunto/Subdivisao is a distinct structured category, not free text)",
                "papel_fonte_dataset": "siac156_curitiba_v1",
                "uso_permitido": "Ponto positivo administrativo geocodificado (mesmo criterio do v12 Recife). Nao e ground truth cartografico individual -- confianca strong = correspondencia de rua (geometria, nao numero exato); medium = fallback bairro-centroide.",
                "uso_proibido": "Nao usar como escore, threshold ou variavel derivada do rotulo. Nao promover a 'validado'/'confirmado' sem revisao do REV-P.",
            }
        )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geocoded", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out")
    args = p.parse_args(argv)

    with open(args.geocoded, encoding="utf-8", newline="") as f:
        geocoded_rows = list(csv.DictReader(f))

    out_rows = build_dataset(geocoded_rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    from collections import Counter
    by_year = Counter(r["ano_fonte"] for r in out_rows)
    by_tier = Counter(r["nivel_confianca"] for r in out_rows)
    summary = {
        "total_positivos": len(out_rows),
        "por_ano": dict(sorted(by_year.items())),
        "por_confianca": dict(by_tier),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
