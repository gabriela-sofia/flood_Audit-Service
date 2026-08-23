"""QA/deduplicação dos registros hidrológicos do SIAC 156 -- Etapa 2, espelha
`PROJETO/scripts/external_validation/qa_recife_seced_geocoding_ready_records_v1.py` (SEDEC
Recife), adaptada ao schema mais limpo do SIAC 156 (categoria já vem estruturada em
`Assunto`/`Subdivisao`, sem mistura de texto livre; `Logradouro` sem número de casa por
desenho do sistema, não por mascaramento -- diferença real registrada no relatório).

Entrada: CSV bruto de `extrair_reclamacoes_enchente_siac156.py` filtrado por toda a base (não só
janelas de evento) -- ver `mine_all_years_siac156.py`.

Critérios (mesmo espírito do Recife, adaptado):
  - `qa_record_id`: hash estável de (source_file, row_index, logradouro, bairro) -- rastreável.
  - `duplicate_group_count`: registros com mesma chave (logradouro normalizado + bairro + data +
    assunto+subdivisao) viram grupo de duplicata -- mesmo pedido reenviado ou re-triado pelo
    sistema.
  - `bairro_reconhecido`: contra a lista real de bairros observados no ano inteiro (não só nas
    linhas de enchente) -- 75 bairros distintos no ano de 2025, que bate com a contagem oficial
    de bairros de Curitiba.
  - `rua_generica_ou_vazia`: string de logradouro vazia, só dígitos, ou comprimento <= 3.
  - `decisao_qa`: `ready_for_geocoding` (bairro reconhecido, rua não genérica, endereço
    presente) | `needs_manual_cleanup` (bairro fora da lista OU rua genérica/curta) |
    `reject` (sem logradouro nem bairro -- não geocodificável de jeito nenhum).

Uso:
    python qa_deduplicacao_siac156.py --raw candidatos_todos_anos.csv --bairros-referencia bairros.json \
        --out qa_registro.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def normalize(text: str | None) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return ascii_only.upper().strip()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


GENERIC_STREET_RE = re.compile(r"^(RUA|AVENIDA|AV|TRAVESSA|ALAMEDA)?\s*\.?\s*$")


def is_generic_or_empty_street(logradouro: str) -> bool:
    norm = normalize(logradouro)
    if not norm:
        return True
    if len(norm) <= 3:
        return True
    if GENERIC_STREET_RE.match(norm):
        return True
    if norm.isdigit():
        return True
    return False


def decisao_qa(bairro_reconhecido: bool, rua_generica: bool, tem_logradouro: bool, tem_bairro: bool) -> tuple[str, list[str]]:
    issues = []
    if not tem_logradouro and not tem_bairro:
        issues.append("sem_logradouro_e_sem_bairro")
        return "reject", issues
    if not bairro_reconhecido:
        issues.append("bairro_fora_da_lista_de_referencia")
    if rua_generica:
        issues.append("rua_generica_ou_vazia")
    if issues:
        return "needs_manual_cleanup", issues
    return "pronto_para_geocodificar", []


def run_qa(rows: list[dict], bairros_referencia: set[str]) -> list[dict]:
    # conta grupos de duplicata primeiro (precisa de duas passadas: uma pra contar, outra pra anotar)
    dup_counts: dict[str, int] = {}
    keys = []
    for row in rows:
        key = "|".join(
            [
                normalize(row.get("Logradouro")),
                normalize(row.get("Bairro")),
                (row.get("DataCriacao") or "").strip(),
                normalize(row.get("Assunto")) + "_" + normalize(row.get("Subdivisao")),
            ]
        )
        keys.append(key)
        dup_counts[key] = dup_counts.get(key, 0) + 1

    out = []
    for row, key in zip(rows, keys):
        logradouro = (row.get("Logradouro") or "").strip()
        bairro = (row.get("Bairro") or "").strip()
        bairro_reconhecido = normalize(bairro) in bairros_referencia
        rua_generica = is_generic_or_empty_street(logradouro)
        decision, issues = decisao_qa(bairro_reconhecido, rua_generica, bool(logradouro), bool(bairro))

        qa_id = stable_hash(row.get("ano_fonte", ""), logradouro, bairro, row.get("DataCriacao", ""))
        out.append(
            {
                "registro_qa_id": qa_id,
                "ano_fonte": row.get("ano_fonte"),
                "data_criacao": row.get("DataCriacao"),
                "assunto": row.get("Assunto"),
                "subdivisao": row.get("Subdivisao"),
                "situacao": row.get("Situacao"),
                "logradouro": logradouro,
                "bairro": bairro,
                "regional": row.get("Regional"),
                "origem": row.get("Origem"),
                "bairro_reconhecido": bairro_reconhecido,
                "rua_generica_ou_vazia": rua_generica,
                "n_grupo_duplicata": dup_counts[key],
                "decisao_qa": decision,
                "problemas_qa": ";".join(issues) if issues else "none",
                "uso_permitido": "Candidato a geocodificacao; nao e coordenada, rotulo ou evidencia de treino.",
                "uso_proibido": "Nao usar como ground truth, escore ou variavel derivada do rotulo.",
            }
        )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True)
    p.add_argument("--bairros-referencia", required=True, help="JSON com lista de bairros validos (maiusculo, sem acento)")
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out")
    args = p.parse_args(argv)

    with open(args.raw, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    bairros_referencia = {normalize(b) for b in json.loads(Path(args.bairros_referencia).read_text(encoding="utf-8"))}

    out_rows = run_qa(rows, bairros_referencia)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    from collections import Counter
    decisions = Counter(r["decisao_qa"] for r in out_rows)
    summary = {"total": len(out_rows), "por_decisao": dict(decisions)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
