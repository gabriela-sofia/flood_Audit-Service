"""Evidências e contexto social locais ligados à resposta, sem atualizar pesos.

Esta memória é um snapshot rastreável. Nenhuma busca externa, interpretação
por LLM ou estatística social muda a decisão física do modelo integrado.
"""
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/servico"))
from proveniencia import sha256_arquivo


def json_valido(v):
    """Cartões históricos usam NaN: publicar ausência como null, nunca zero."""
    if isinstance(v, dict):
        return {k: json_valido(x) for k, x in v.items()}
    if isinstance(v, list):
        return [json_valido(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def construir():
    cartoes = REPO / "local_runs/mecanismos-transferiveis/v3/cartoes_cientificos.json"
    raiz = REPO / "local_runs/social-censo-municipal"
    manifestos = list(raiz.glob("*/manifesto.json"))
    if len(manifestos) != 1:
        raise ValueError("snapshot social ambíguo; escolher versão antes da integração")
    social = manifestos[0]
    meta = json.loads(social.read_text(encoding="utf-8"))
    dados = social.parent / "indicadores.csv"
    if sha256_arquivo(dados) != meta["indicadores_sha256"]:
        raise ValueError("contexto social alterado")
    with dados.open(encoding="utf-8", newline="") as f:
        linhas = list(csv.DictReader(f))
    fontes = [{"arquivo": str(p), "sha256": sha256_arquivo(p)} for p in (cartoes, social, dados)]
    return {"cartoes": json_valido(json.loads(cartoes.read_text(encoding="utf-8"))["cartoes"]),
            "social": {"linhas": linhas, "ano": meta["ano"], "escala": meta["escala"],
                       "limites": meta["limites"], "atribuicao": meta["atribuicao"]},
            "fontes": fontes, "atualiza_pesos": False, "usado_no_escore": False,
            "nao_finitos_historicos": "convertidos para null; não são zeros"}


def recuperar(memoria, territorio=None, municipio=None):
    out = {"fontes": memoria.get("fontes", []), "atualiza_pesos": False, "usado_no_escore": False}
    if territorio:
        cartoes = [c for c in memoria.get("cartoes", []) if c["territorio"] == territorio]
        out["territorio"] = {"id": territorio, "estado": "documentado" if cartoes else "sem_evidencia_local",
                              "cartoes": cartoes,
                              "aviso": "cartões descrevem os experimentos nomeados; não são validação do modelo integrado"}
    if municipio:
        social = memoria.get("social", {})
        linhas = [r for r in social.get("linhas", []) if r["municipio_id"] == municipio]
        out["social"] = {"municipio_id": municipio, "estado": "contexto_municipal" if linhas else "desconhecido",
                         "ano": social.get("ano"), "escala": social.get("escala"), "indicadores": linhas,
                         "limites": social.get("limites", []), "atribuicao": social.get("atribuicao")}
    return out
