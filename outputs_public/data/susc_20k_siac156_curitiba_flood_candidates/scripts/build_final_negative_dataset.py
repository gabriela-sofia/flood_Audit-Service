"""SUSC-20N -- monta o dataset final de pontos NEGATIVOS de Curitiba a partir dos candidatos
geocodificados, aplicando a checagem de colisao espacial contra os positivos.

Existe porque o `build_final_dataset.py` so monta positivos (label=1); a etapa equivalente do
SUSC-20K3 foi executada mas nao ficou versionada como script. As regras aqui sao as
documentadas em `susc_20k2_k3_replica_completa_metodo_recife_report.md` secao 7.3:

  - aceite em bloco `strong` OU `medium` (`failed` nunca entra -- coordenada nunca e fabricada);
  - remocao por colisao espacial: candidato a menos de 30 m de QUALQUER positivo e descartado
    (mesma rua tem reclamacao de enchente E de outra categoria -> o ponto nao e negativo limpo).
    Raio de 30 m, mesmo criterio de Recife;
  - `negative_source_type = siac156_curitiba_non_hydrological_category_v1`;
  - `event_date` = data real da ocorrencia nao-hidrologica do proprio registro, nunca sintetica.

Uniao com uma geracao anterior (`--existentes`): linhas ja presentes no dataset negativo
anterior sao carregadas VERBATIM (mesmo `point_id`, mesma proveniencia) e candidatos novos que
repitam a chave (logradouro, bairro, data) sao descartados. Isso preserva continuidade de id
entre rodadas -- necessario porque a amostragem deterministica de
`mine_negative_candidates_siac156.py` usa `stable_rank(str(csv_path), ...)`, que depende do
caminho passado na linha de comando e portanto NAO reproduz a mesma amostra entre sessoes com
diretorios diferentes. Duas geracoes = uniao de dois sorteios do mesmo pool sob os mesmos
criterios, nao um superconjunto.

Uso:
    python build_final_negative_dataset.py --geocoded geocodificados.csv \
        --positivos dataset_positivos.csv --existentes dataset_negativos_v1.csv \
        --out dataset_negativos_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

COLLISION_RADIUS_M = 30.0
NEGATIVE_SOURCE_TYPE = "siac156_curitiba_non_hydrological_category_v1"
ALLOWED_USE = ("Ponto negativo administrativo geocodificado, pareado por bairro com positivo "
               "real, sem colisao <30m. Mesma regra causal do v12 Recife.")
PROHIBITED_USE = ("Nao usar como score, threshold ou feature derivada do label. Ausencia de "
                  "registro NAO foi usada como criterio -- presenca de outro fenomeno real.")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def negative_point_id(logradouro: str, bairro: str, data_criacao: str) -> str:
    """Id derivado da chave do registro. O esquema exato da geracao anterior (SUSC-20K3) nao
    ficou documentado e nao e recuperavel do hash; ids antigos sao preservados verbatim pela
    uniao, e so pontos NOVOS recebem id por este esquema."""
    key = f"{logradouro}|{bairro}|{data_criacao}"
    return "CUR_SIAC156_NEG_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def collides_with_positive(lat: float, lon: float, positives: list[tuple[float, float]]) -> bool:
    # filtro grosseiro por caixa antes do haversine (30 m ~ 0.00027 grau de latitude)
    d = 0.0005
    for plat, plon in positives:
        if abs(plat - lat) > d or abs(plon - lon) > d:
            continue
        if haversine_m(lat, lon, plat, plon) < COLLISION_RADIUS_M:
            return True
    return False


def build(geocoded: list[dict], positives: list[tuple[float, float]],
          existentes: list[dict] | None) -> tuple[list[dict], dict]:
    out: list[dict] = []
    vistos: set[tuple[str, str, str]] = set()

    if existentes:
        for row in existentes:
            out.append(dict(row))
            vistos.add((row["logradouro"], row["bairro"], row["event_date"]))

    n_failed = n_colisao = n_dup = 0
    for row in geocoded:
        if row.get("confidence_tier") not in ("strong", "medium"):
            n_failed += 1
            continue
        chave = (row["logradouro"], row["bairro"], row["data_criacao"])
        if chave in vistos:
            n_dup += 1
            continue
        lat, lon = float(row["lat"]), float(row["lon"])
        if collides_with_positive(lat, lon, positives):
            n_colisao += 1
            continue
        vistos.add(chave)
        out.append({
            "region": "Curitiba",
            "label": 0,
            "point_id": negative_point_id(row["logradouro"], row["bairro"], row["data_criacao"]),
            "lat": lat,
            "lon": lon,
            "event_date": row["data_criacao"],
            "source_year": row["source_year"],
            "assunto": row["assunto"],
            "subdivisao": row["subdivisao"],
            "logradouro": row["logradouro"],
            "bairro": row["bairro"],
            "confidence_tier": row["confidence_tier"],
            "nominatim_display_name": row["nominatim_display_name"],
            "nominatim_osm_id": row["osm_id"],
            "negative_source_type": NEGATIVE_SOURCE_TYPE,
            "occurrence_phenomenon": (f"non-flood ({row['assunto']}/{row['subdivisao']}) - causally "
                                      "independent of rain, see negative_categories_curitiba.py"),
            "dataset_role_source": "siac156_curitiba_negative_v1",
            "allowed_use": ALLOWED_USE,
            "prohibited_use": PROHIBITED_USE,
        })

    resumo = {
        "n_existentes_carregados": len(existentes) if existentes else 0,
        "n_candidatos_geocodificados": len(geocoded),
        "n_descartados_failed": n_failed,
        "n_descartados_duplicata_de_existente": n_dup,
        "n_descartados_colisao_30m": n_colisao,
        "n_novos_aceitos": len(out) - (len(existentes) if existentes else 0),
        "total_final": len(out),
        "por_confianca": dict(Counter(r["confidence_tier"] for r in out)),
        "por_ano": dict(sorted(Counter(str(r["source_year"]) for r in out).items())),
        "raio_colisao_m": COLLISION_RADIUS_M,
    }
    return out, resumo


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geocoded", required=True)
    p.add_argument("--positivos", required=True)
    p.add_argument("--existentes", default=None, help="dataset negativo de uma geracao anterior")
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out")
    args = p.parse_args(argv)

    with open(args.geocoded, encoding="utf-8", newline="") as f:
        geocoded = list(csv.DictReader(f))
    with open(args.positivos, encoding="utf-8", newline="") as f:
        positives = [(float(r["lat"]), float(r["lon"])) for r in csv.DictReader(f)]
    existentes = None
    if args.existentes:
        with open(args.existentes, encoding="utf-8", newline="") as f:
            existentes = list(csv.DictReader(f))

    rows, resumo = build(geocoded, positives, existentes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
