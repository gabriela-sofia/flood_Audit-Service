"""Monta o dataset final de pontos positivos de Curitiba a partir do SIAC 156 -- mesma regra de
aceitação do dataset v12 de Recife (`dataset_eventos_features_v12_final.csv`): geocodificação
strong OU medium entra em bloco como positivo, sem segunda camada de verificação manual
individual (essa é a regra real usada em Recife -- ver
`PROJETO/local_runs/treino_exploratorio_diagnostico_v2/_scratch/stage1_build_points_v2.py:124`,
comentário "Recife POSITIVES: same geocoded points as v1 (unchanged)").

Cada linha carrega proveniência completa (`qa_record_id`, `source_year`, `data_criacao`,
`assunto`/`subdivisao`, `confidence_tier`, `nominatim_display_name`, `nominatim_osm_id`) e um
campo explícito `occurrence_phenomenon` confirmando exclusão de deslizamento (mesma prática do
GeoJSON final de Recife).

`failed` nunca entra -- nenhuma coordenada fabricada.

Uso:
    python build_final_dataset.py --geocoded geocodificado_full.csv --out dataset_positivos_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_dataset(geocoded_rows: list[dict]) -> list[dict]:
    out = []
    for row in geocoded_rows:
        if row.get("confidence_tier") not in ("strong", "medium"):
            continue  # failed -- nao fabrica coordenada
        out.append(
            {
                "region": "Curitiba",
                "label": 1,
                "point_id": f"CUR_SIAC156_{row['qa_record_id']}",
                "lat": row["lat"],
                "lon": row["lon"],
                "event_date": row["data_criacao"],
                "source_year": row["source_year"],
                "assunto": row["assunto"],
                "subdivisao": row["subdivisao"],
                "logradouro": row["logradouro"],
                "bairro": row["bairro"],
                "confidence_tier": row["confidence_tier"],
                "nominatim_display_name": row["nominatim_display_name"],
                "nominatim_osm_id": row["osm_id"],
                "duplicate_group_count": row["duplicate_group_count"],
                "occurrence_phenomenon": "flood (alagamento/enchente) - landslide excluded by category filter (SIAC 156 Assunto/Subdivisao is a distinct structured category, not free text)",
                "dataset_role_source": "siac156_curitiba_v1",
                "allowed_use": "Ponto positivo administrativo geocodificado (mesmo criterio do v12 Recife). Nao e ground truth cartografico individual -- confianca strong = correspondencia de rua (geometria, nao numero exato); medium = fallback bairro-centroide.",
                "prohibited_use": "Nao usar como score, threshold ou feature derivada do label. Nao promover a 'validado'/'confirmado' sem revisao do REV-P.",
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
    by_year = Counter(r["source_year"] for r in out_rows)
    by_tier = Counter(r["confidence_tier"] for r in out_rows)
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
