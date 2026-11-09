"""Audita eventos GFD e materializa cortes sem evento ou tile compartilhado.

As janelas vem do identificador oficial do asset, como no ds01. O recorte
e dentro do GFD: identidade DFO nao reconcilia eventos de outras fontes.
Nao ajusta modelo e nao declara validade dos preditores para previsao futura.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
from proveniencia import agora_utc, sha256_json, sha256_bytes  # noqa: E402

GFD = REPO / "local_runs" / "gfd-brasil"
SAIDA = REPO / "local_runs" / "generalizacao-por-evento"
LISTA_AMPLIADA = REPO / "local_runs" / "br-02-negativo-nacional" / "eventos_que_tocam_brasil.csv"
FEATURES = ("hand_m", "twi_dinf", "slope_deg", "elevation_m")


def catalogar(catalogo: dict) -> dict:
    eventos = {}
    for registro in catalogo.get("eventos", []):
        asset = registro.get("id_ee", "")
        m = re.search(r"DFO_(\d+)_From_(\d{8})_to_(\d{8})", asset)
        if not m:
            continue
        inicio = datetime.strptime(m[2], "%Y%m%d").date()
        fim = datetime.strptime(m[3], "%Y%m%d").date()
        if fim < inicio:
            raise ValueError(f"janela invertida: {asset}")
        chave = f"GFD:{int(m[1])}"
        janela = {"inicio": inicio.isoformat(), "fim": fim.isoformat(), "asset": asset}
        if chave in eventos and eventos[chave] != janela:
            raise ValueError(f"janelas conflitantes: {chave}")
        eventos[chave] = janela
    return eventos


def reconciliar(catalogo: dict, lista: list[dict]) -> dict:
    """A lista da aquisicao ampliada complementa o catalogo historico.

    Nenhum pais e usado para filtrar: o evento pode atravessar a fronteira.
    Duas janelas diferentes para a mesma identidade exigem revisao.
    """
    eventos = catalogar(catalogo)
    novos = catalogar({"eventos": [{"id_ee": r.get("asset", "")} for r in lista]})
    for chave, janela in novos.items():
        if chave in eventos and eventos[chave] != janela:
            raise ValueError(f"conflito entre catalogo e aquisicao: {chave}")
        eventos[chave] = janela
    return eventos


def auditar(linhas: list[dict], eventos: dict) -> tuple[list[dict], dict]:
    admitidas, recusas, vistos = [], Counter(), set()
    for linha in linhas:
        ponto = linha.get("ponto_id")
        if not ponto or ponto in vistos:
            raise ValueError("ponto_id ausente ou duplicado")
        vistos.add(ponto)
        evento = f"GFD:{linha.get('evento_gfd', '')}"
        motivo = None
        if evento not in eventos:
            motivo = "evento_sem_janela_do_catalogo"
        elif not linha.get("tile"):
            motivo = "sem_tile"
        elif linha.get("classe_obs") not in ("INUNDACAO", "NAO_INUNDADO_OBSERVADO"):
            motivo = "classe_fora_do_binario_observado"
        else:
            try:
                import math
                if not all(math.isfinite(float(linha.get(f, ""))) for f in FEATURES):
                    motivo = "terreno_nao_finito"
            except (TypeError, ValueError):
                motivo = "terreno_incompleto"
        if motivo:
            recusas[motivo] += 1
        else:
            admitidas.append({"ponto_id": ponto, "evento": evento,
                              "grupo": linha["tile"],
                              "classe": int(linha["classe_obs"] == "INUNDACAO"),
                              **eventos[evento]})
    amostrados = {f"GFD:{r.get('evento_gfd', '')}" for r in linhas}
    return admitidas, {"linhas": len(linhas), "admitidas": len(admitidas),
                       "recusas": dict(recusas),
                       "eventos_amostrados": len(amostrados),
                       "eventos_sem_janela": sorted(amostrados - eventos.keys()),
                       "catalogo_cobre_toda_amostragem": not (amostrados - eventos.keys()),
                       "eventos_admitidos": len({r['evento'] for r in admitidas})}


def separar(linhas: list[dict], *, evento: str | None = None,
            corte: str | None = None) -> tuple[list[dict], list[dict], dict]:
    if (evento is None) == (corte is None):
        raise ValueError("escolha evento ou corte temporal")
    if corte:
        marco = date.fromisoformat(corte).isoformat()
        candidatos = [r for r in linhas if r["fim"] < marco]
        teste = [r for r in linhas if r["inicio"] > marco]
    else:
        teste = [r for r in linhas if r["evento"] == evento]
        candidatos = [r for r in linhas if r["evento"] != evento]
    eventos_teste = {r["evento"] for r in teste}
    grupos_teste = {r["grupo"] for r in teste}
    treino = [r for r in candidatos
              if r["evento"] not in eventos_teste and r["grupo"] not in grupos_teste]
    if {r["evento"] for r in treino} & eventos_teste:
        raise ValueError("evento atravessou o corte")
    if {r["grupo"] for r in treino} & grupos_teste:
        raise ValueError("tile atravessou o corte")
    contagens = {}
    for nome, lado in (("treino", treino), ("teste", teste)):
        contagens[nome] = {
            "linhas": len(lado), "eventos": len({r['evento'] for r in lado}),
            "grupos_por_classe": {str(c): len({r['grupo'] for r in lado if r['classe'] == c})
                                  for c in (0, 1)}}
    ambas = all(all(v > 0 for v in lado["grupos_por_classe"].values())
                for lado in contagens.values())
    return treino, teste, {
        "estado": "corte_materializado" if ambas else "not_testable",
        "motivo": None if ambas else "uma classe ausente depois da protecao",
        "contagens": contagens,
        "retiradas_por_sobreposicao": len(candidatos) - len(treino),
        "fora_do_corte": len(linhas) - len(treino) - len(teste),
        "avaliacao": "untested",
        "orcamento_e_disponibilidade_dos_preditores": "pendentes_de_verificacao",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pontos", type=Path, default=GFD / "pontos_gfd_com_terreno.csv")
    ap.add_argument("--catalogo", type=Path, default=GFD / "catalogo.json")
    ap.add_argument("--lista-ampliada", type=Path, default=LISTA_AMPLIADA)
    ap.add_argument("--saida", type=Path, default=SAIDA)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--evento", help="identidade GFD:<numero>")
    g.add_argument("--corte", help="data ISO, fixada antes de avaliar resultados")
    args = ap.parse_args()
    # O manifesto identifica os bytes lidos, mesmo com derivacao em paralelo.
    bruto = args.pontos.read_bytes()
    catalogo = args.catalogo.read_bytes()
    import io
    linhas = list(csv.DictReader(io.StringIO(bruto.decode("utf-8-sig"))))
    ampliada = args.lista_ampliada.read_bytes()
    lista = list(csv.DictReader(io.StringIO(ampliada.decode("utf-8-sig"))))
    eventos = reconciliar(json.loads(catalogo), lista)
    admitidas, auditoria = auditar(linhas, eventos)
    corpo = {"tipo": "auditoria_e_corte_por_evento_gfd", "quando_utc": agora_utc(),
             "entrada": {"pontos": sha256_bytes(bruto), "catalogo": sha256_bytes(catalogo),
                         "lista_ampliada": sha256_bytes(ampliada)},
             "auditoria": auditoria, "eventos": eventos,
             "escopo": "eventos dentro da fonte GFD; nao reconciliados entre fontes",
             "nao_e": "nao e treino, validacao temporal ou dataset admitido pelo ds05"}
    args.saida.mkdir(parents=True, exist_ok=True)
    (args.saida / "catalogo_reconciliado.json").write_text(json.dumps({
        "tipo": "catalogo_gfd_reconciliado_com_amostragem",
        "quando_utc": corpo["quando_utc"], "entrada": corpo["entrada"],
        "eventos": eventos, "cobertura": auditoria,
        "nao_e": "nao e novo evento observado nem reconciliacao entre instrumentos"},
        indent=2, ensure_ascii=False), encoding="utf-8")
    if args.evento or args.corte:
        treino, teste, resultado = separar(admitidas, evento=args.evento, corte=args.corte)
        corpo["corte"] = {"evento": args.evento, "data": args.corte, **resultado}
        for nome, lado in (("treino", treino), ("teste", teste)):
            with (args.saida / f"{nome}.csv").open("w", newline="", encoding="utf-8") as arquivo:
                escritor = csv.DictWriter(arquivo, fieldnames=["ponto_id", "evento", "grupo", "classe", "inicio", "fim", "asset"])
                escritor.writeheader()
                escritor.writerows(lado)
            corpo[f"hash_{nome}"] = sha256_bytes((args.saida / f"{nome}.csv").read_bytes())
    corpo["hash_resumo"] = sha256_json({k: v for k, v in corpo.items() if k != "quando_utc"})
    (args.saida / "manifesto.json").write_text(json.dumps(corpo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(auditoria, ensure_ascii=False))
    if "corte" in corpo:
        print(json.dumps(corpo["corte"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
