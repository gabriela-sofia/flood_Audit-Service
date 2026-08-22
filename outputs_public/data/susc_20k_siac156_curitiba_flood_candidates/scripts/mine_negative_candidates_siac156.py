"""Minera candidatos a ponto negativo do SIAC 156 (Curitiba) -- espelha
`build_recife_negative_candidates.py` + a correção de pareamento geográfico do v9
(`build_v9_bairro_matched_new_negatives.py`) aplicada desde o início, não como remendo
posterior (o v8 de Recife cometeu esse erro e teve que ser corrigido depois -- aqui já nasce
pareado).

Segue as 5 condições de
`docs/metodologia_cientifica/revp_criterio_ponto_negativo_recife_e_replicacao_curitiba_petropolis.md`:
  1. presença de outro fenômeno real (categoria do `negative_categories_curitiba.py`, nunca
     ausência de registro de enchente)
  2. categoria causalmente independente de chuva (documentada por item, não herdada de Recife)
  3. coordenada auditável (geocodificação Nominatim strong/medium -- feita em etapa separada,
     `geocode_nominatim.py`, já validada nesta rodada)
  4. data real da ocorrência (campo `DataCriacao` da fonte, nunca sintética)
  5. pareamento geográfico: bairro do negativo tem que ter positivo real (aplicado aqui já na
     mineração, usando o dataset de positivos fechado desta rodada)

Amostragem determinística por ano (mesmo espírito do v8/v9 de Recife: hash estável, não
depende de semente aleatória) -- evita reprocessar um arquivo de 700k+ linhas inteiro sem
limite quando o pool de candidatos válidos é muito maior que o necessário.

Correção SUSC-20N: `rank_key` usava `str(csv_path)` no hash, então a amostra dependia do
caminho de execução (mesmo arquivo, diretório diferente = amostra diferente) -- descoberto
quando a rodada de reforço de negativo (--max-por-ano 120) não reproduziu como superconjunto
da rodada anterior (--max-por-ano 40). Corrigido para usar `source_year` (derivado do próprio
registro) em vez do path; a amostra agora é estável entre máquinas e diretórios de execução.
Ver `test_rank_key_is_independent_of_csv_path` e
`susc_20n_reforco_negativos_retest_epv_report.md` seção 2.

Uso:
    python mine_negative_candidates_siac156.py --csv ano1.csv ano2.csv ... \
        --positivos dataset_positivos_curitiba_v1.csv --out negativos_brutos.csv \
        --max-por-ano 400
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from negative_categories_curitiba import is_curitiba_negative_category  # noqa: E402


def normalize(text: str | None) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return ascii_only.upper().strip()


def stable_rank(*parts: str) -> int:
    """Hash estável -- usado pra ordenação determinística sem depender de seed aleatória."""
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


def load_positive_bairros(positivos_path: Path | str) -> set[str]:
    with open(positivos_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {normalize(r["bairro"]) for r in rows}


def mine_year_file(csv_path: Path | str, positive_bairros: set[str], max_por_ano: int | None) -> list[dict]:
    source_year = None
    candidates = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            assunto = row.get("Assunto")
            subdivisao = row.get("Subdivisao")
            if not is_curitiba_negative_category(assunto, subdivisao):
                continue
            bairro = row.get("Bairro") or ""
            if normalize(bairro) not in positive_bairros:
                continue  # condicao 5: pareamento geografico
            logradouro = row.get("Logradouro") or ""
            data_criacao = row.get("DataCriacao") or ""
            if not logradouro.strip() or not bairro.strip():
                continue
            if source_year is None:
                source_year = data_criacao[-4:] if len(data_criacao) >= 4 else ""
            candidates.append(
                {
                    "source_year": source_year,
                    "data_criacao": data_criacao,
                    "assunto": assunto,
                    "subdivisao": subdivisao,
                    "situacao": row.get("Situacao"),
                    "logradouro": logradouro,
                    "bairro": bairro,
                    "regional": row.get("Regional"),
                    "origem": row.get("Origem"),
                    # SUSC-20N corrigiu um bug de reprodutibilidade: a versao anterior usava
                    # str(csv_path) aqui, entao a amostra dependia do caminho passado na linha
                    # de comando (mesmo arquivo, diretorio diferente = amostra diferente). Chave
                    # agora e so o conteudo do registro (source_year, nao o path), estavel entre
                    # maquinas e sessoes.
                    "rank_key": stable_rank(source_year, logradouro, bairro, data_criacao),
                }
            )
    candidates.sort(key=lambda r: r["rank_key"])
    if max_por_ano is not None:
        candidates = candidates[:max_por_ano]
    return candidates


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--positivos", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-por-ano", type=int, default=None)
    p.add_argument("--summary-out")
    args = p.parse_args(argv)

    positive_bairros = load_positive_bairros(args.positivos)

    all_rows = []
    for csv_path in args.csv:
        rows = mine_year_file(csv_path, positive_bairros, args.max_por_ano)
        all_rows.extend(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_year", "data_criacao", "assunto", "subdivisao", "situacao", "logradouro", "bairro", "regional", "origem", "rank_key"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    from collections import Counter
    by_year = Counter(r["source_year"] for r in all_rows)
    summary = {"total": len(all_rows), "por_ano": dict(sorted(by_year.items())), "bairros_positivos_usados_no_pareamento": len(positive_bairros)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
