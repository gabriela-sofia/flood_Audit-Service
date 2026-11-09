"""Quais eventos ja foram olhados como teste, e se ainda existe reserva intacta.

-------------------------------------------------------------------------
POR QUE ISTO EXISTE
-------------------------------------------------------------------------

Em 01/10/2026 `generalizacao_protegida` lacrou 13 eventos GFD como teste
final ("20% por hash da identidade; sem avaliacao nesta rodada"). Em 02/10
`curva_de_aprendizado` e `avaliacao_politica_retrospectiva` reservaram "25% dos
eventos por hash da identidade" -- a MESMA ordenacao. 20% esta contido em 25%:
os 13 eventos lacrados viraram teste das duas, e os numeros sairam no
fechamento como "teste protegido". Nada no codigo cruzava uma reserva com a
outra.

E antes disso `sinais_marginais` ja tinha medido as relacoes sobre a tabela
inteira, lacre incluido: o vetor das hipoteses sustentadas foi escolhido
olhando todos os eventos.

Consequencia: nao ha reserva intacta no ds06. Criar uma particao nova agora
nao a torna intacta -- os rotulos ja informaram escolhas. A unica reserva que
pode ser confirmatoria e PROSPECTIVA: evento que nao estava na base quando
o conjunto conhecido foi congelado.

Este modulo:
  - reconstroi, a partir dos artefatos, os eventos que cada avaliacao usou
    como teste;
  - congela (uma vez, imutavel) a lista de eventos conhecidos;
  - classifica qualquer teste novo: `exploratorio` se toca evento conhecido,
    `confirmatorio_prospectivo` so se todos os eventos forem posteriores.

nao_e: nao reabre o lacre nem o declara valido. Registra que ele foi aberto.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

from proveniencia import agora_utc, sha256_json  # noqa: E402

RUNS = REPO / "local_runs"
SAIDA = RUNS / "reservas-examinadas"
CONHECIDOS = SAIDA / "conhecidos_2026_10_03.json"
LACRE_R2 = RUNS / "generalizacao-protegida-2026-10-01-r2" / "protocolo.json"
CURVA = RUNS / "curva-de-aprendizado" / "curva.json"
POLITICA = RUNS / "politica-retrospectiva" / "politica.json"

EXPLORATORIO = "exploratorio"
CONFIRMATORIO = "confirmatorio_prospectivo"


def _j(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def testes_registrados() -> list[dict]:
    """Cada avaliacao que leu rotulo de evento como teste, com quando e quais."""
    out = []
    lac = _j(LACRE_R2)
    if lac:
        out.append({"avaliacao": "generalizacao_protegida_r2:lacre",
                    "quando_utc": lac["quando_utc"], "papel": "lacrado (nao avaliado ali)",
                    "eventos": sorted(lac["final_protegido"])})
    c = _j(CURVA)
    if c:
        for nome, e in c.get("estudos", {}).items():
            if e.get("eventos_de_teste"):
                out.append({"avaliacao": f"curva_de_aprendizado:{nome}",
                            "quando_utc": c.get("quando_utc"), "papel": "teste",
                            "eventos": sorted(e["eventos_de_teste"])})
    p = _j(POLITICA)
    if p:
        out.append({"avaliacao": "politica_retrospectiva", "quando_utc": p.get("quando_utc"),
                    "papel": "teste", "eventos": sorted(p["teste_protegido"])})
    return out


def lacre_aberto() -> dict:
    regs = testes_registrados()
    lac = next((r for r in regs if r["avaliacao"].endswith(":lacre")), None)
    if not lac:
        return {"estado": "sem_lacre_registrado"}
    abertos = {}
    for r in regs:
        if r is lac or r["papel"] != "teste":
            continue
        comum = sorted(set(r["eventos"]) & set(lac["eventos"]))
        if comum:
            abertos[r["avaliacao"]] = comum
    return {"estado": "aberto" if abertos else "intacto", "lacrados": len(lac["eventos"]),
            "aberto_por": abertos,
            "causa": ("a mesma ordenacao por sha256 do evento foi usada para o lacre (20%) e "
                      "para os testes de 02/10 (25%): o primeiro esta contido no segundo")}


def eventos_atuais(fontes=("gfd", "cems", "sen1floods11", "uk", "ufo")) -> dict:
    from ds06_dataset_de_treino import versao_atual
    import pandas as pd
    origem = versao_atual()
    d = pd.read_parquet(origem / "pontos.parquet", columns=["fonte", "evento_id"])
    d = d.dropna(subset=["evento_id"])
    return {"dataset": origem.name,
            "por_fonte": {f: sorted(d.loc[d.fonte == f, "evento_id"].astype(str).unique())
                          for f in fontes}}


def congelar(destino: Path = CONHECIDOS) -> dict:
    """Uma vez so. Congelar de novo depois de ver um evento novo o tornaria conhecido."""
    if destino.exists():
        return json.loads(destino.read_text(encoding="utf-8"))
    atual = eventos_atuais()
    r = {"tipo": "eventos_conhecidos_congelados", "quando_utc": agora_utc(), **atual,
         "regra": ("evento listado aqui ja pode ter informado escolha (vetor, politica, "
                   "limiar); teste que toque qualquer um deles e exploratorio"),
         "nao_e": "nao e particao de treino/teste: e a fronteira do que pode ser confirmatorio"}
    r["hash"] = sha256_json({k: v for k, v in r.items() if k != "quando_utc"})
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


EXAMINADOS_DEPOIS = SAIDA / "examinados_apos_congelamento.jsonl"


def registrar_exame(eventos, fonte: str, avaliacao: str, destino: Path = EXAMINADOS_DEPOIS) -> None:
    """Append-only: evento lido como teste DEPOIS do congelamento deixa de poder confirmar.

    O conjunto congelado e imutavel de proposito; sem este registro, um segundo
    teste "confirmatorio" sobre os mesmos eventos novos passaria pelo `regime`.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"quando_utc": agora_utc(), "fonte": fonte, "avaliacao": avaliacao,
                            "eventos": sorted(map(str, eventos))}, ensure_ascii=False) + "\n")


def examinados_depois(fonte: str, origem: Path = EXAMINADOS_DEPOIS) -> set:
    if not origem.exists():
        return set()
    out = set()
    for linha in origem.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            r = json.loads(linha)
            if r["fonte"] == fonte:
                out |= set(r["eventos"])
    return out


def equivalentes_conhecidos(janelas, conhecidos: dict | None = None, base=None, consultas=None) -> dict:
    """Eventos de teste que podem ser o MESMO evento de um conhecido ou consultado, em QUALQUER fonte.

    `janelas`: DataFrame com evento_id, inicio, fim, lat_min, lat_max, lon_min, lon_max
    (o formato de `verdade_observacional.janelas_de_eventos`). Comparar so o ID dentro
    da fonte deixava passar GFD:4577 = S1F11:Bolivia e os eventos do pacote de serra.
    """
    import pandas as pd
    import verdade_observacional as vo
    conhecidos = conhecidos if conhecidos is not None else _j(CONHECIDOS)
    if base is None:
        from ds06_dataset_de_treino import versao_atual
        d = pd.read_parquet(versao_atual() / "pontos.parquet",
                            columns=["fonte", "evento_id", "janela_inicio", "janela_fim", "lat", "lon"])
        base = vo.janelas_de_eventos(d.dropna(subset=["evento_id"]))
    sabidos = {e for v in (conhecidos or {}).get("por_fonte", {}).values() for e in v}
    base = base[base.evento_id.astype(str).isin(sabidos)]
    consultas = vo.consultados() if consultas is None else consultas
    datadas = consultas.dropna(subset=["inicio"]) if len(consultas) else consultas
    teste = janelas.assign(fonte="teste")
    out = {"conhecidos": vo.equivalencias(teste, base, mesma_fonte=True),
           "consultados": vo.equivalencias(teste, datadas, mesma_fonte=True) if len(datadas) else []}
    ids_lidos = set(consultas.evento_id) if len(consultas) else set()
    out["consultados"] += [{"a": f"teste|{e}", "b": f"leitura_parcial|{e}"}
                           for e in sorted(set(janelas.evento_id.astype(str)) & ids_lidos)]
    return out


def regime(eventos_de_teste, fonte: str, conhecidos: dict | None = None,
           examinados: set | None = None, janelas=None, base=None, consultas=None) -> dict:
    """`janelas` (opcional) liga a identidade entre fontes; sem ela o confirmatorio sai com a ressalva."""
    conhecidos = conhecidos if conhecidos is not None else _j(CONHECIDOS)
    if not conhecidos:
        return {"regime": EXPLORATORIO, "motivo": "conjunto conhecido nao congelado"}
    eq = None
    if janelas is not None:
        eq = equivalentes_conhecidos(janelas, conhecidos, base, consultas)
        if eq["conhecidos"]:
            return {"regime": EXPLORATORIO, "equivalentes_a_conhecidos": eq["conhecidos"],
                    "motivo": ("evento de teste coincide no tempo e no espaco com evento conhecido de "
                               "outra fonte ou outro ID: e o mesmo evento ate prova em contrario")}
    examinados = examinados if examinados is not None else examinados_depois(fonte)
    ja_lidos = sorted(set(map(str, eventos_de_teste)) & examinados)
    if ja_lidos:
        return {"regime": EXPLORATORIO, "eventos_ja_lidos_como_teste": len(ja_lidos),
                "motivo": "evento posterior ao congelamento, mas ja lido como teste por outra avaliacao"}
    sabidos = set(conhecidos["por_fonte"].get(fonte, []))
    tocados = sorted(set(map(str, eventos_de_teste)) & sabidos)
    if tocados:
        return {"regime": EXPLORATORIO, "eventos_conhecidos_no_teste": len(tocados),
                "motivo": ("o teste contem eventos que ja estavam na base em 03/10/2026; "
                           "o vetor foi escolhido com eles e o lacre de 01/10 foi aberto")}
    r = {"regime": CONFIRMATORIO, "hash_conhecidos": conhecidos["hash"]}
    if eq is None:
        r["identidade_entre_fontes"] = "nao_verificada: o teste nao trouxe janelas"
    elif eq["consultados"]:
        # Consultado (imagem ou rotulo lido, nenhuma escolha de modelo feita com ele) nao
        # rebaixa o regime: vira ressalva obrigatoria do resultado.
        r["ressalva_consultados_antes"] = eq["consultados"]
    return r


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    c = congelar()
    r = {"tipo": "reservas_examinadas", "quando_utc": agora_utc(),
         "testes_registrados": testes_registrados(), "lacre": lacre_aberto(),
         "conhecidos": {"arquivo": str(CONHECIDOS.relative_to(REPO)), "hash": c["hash"],
                        "por_fonte": {f: len(v) for f, v in c["por_fonte"].items()}},
         "reserva_intacta_no_ds06": False,
         "por_que": ("sinais_marginais mediu as relacoes com todos os eventos, e o lacre foi "
                     "aberto em 02/10; so evento posterior ao congelamento pode confirmar")}
    r["hash_resumo"] = sha256_json({k: v for k, v in r.items() if k != "quando_utc"})
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "retrato.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: r[k] for k in ("lacre", "conhecidos", "reserva_intacta_no_ds06")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
