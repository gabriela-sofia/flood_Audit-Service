"""
Evidencia visual DINOv2 -- anexada a resposta, lida por nenhum portao.

O QUE ESTE MODULO PODE PROMETER, e o que nao pode:

o corpus de embeddings do projeto tem **12 vetores reais**, quatro por regiao,
com encoder congelado e hash registrado. Nao e escala de produto: e amostra de
analise estrutural. Entao `evidencia_visual` NAO esta disponivel para qualquer
coordenada -- so para ponto que caia num patch que ja tem embedding calculado.

Isso nao e limitacao a contornar, e o estado real do corpus. Quando
`disponivel` e falso, quem le a resposta tem de ver isso escrito, e a camada de
texto tem de dizer que nao ha evidencia -- nunca inventar uma.

-------------------------------------------------------------------------
A REGRA QUE NAO PODE SER QUEBRADA
-------------------------------------------------------------------------

Nenhum portao consulta este modulo. O DINOv2 nao entra em G1-G5, nao entra no
escore, nao desloca IC e nao muda maturidade. E o Limite 1 do artigo, e a razao
de a rota estar congelada: tres tentativas independentes de promove-lo a
preditor fecharam sem sinal.

A similaridade que sai daqui ILUSTRA a resposta para quem revisa. Se algum dia
ela entrar no escore, deixa de ser evidencia anexada e passa a exigir o
orcamento de EPV e a trava de sinal fisico que valem para toda variavel.

-------------------------------------------------------------------------
POR QUE OS ASSETS SAO OPT-IN POR VARIAVEL DE AMBIENTE
-------------------------------------------------------------------------

Os GeoTIFF de origem (de onde sai a caixa envolvente de cada patch) vivem fora
do repositorio -- regra fixa do projeto: nenhum artefato pesado em git. O
prototipo de julho resolvia isso com `REVP_SUSC20D_SENTINEL_DIR`, e o mesmo
padrao vale aqui: sem o diretorio configurado, o modulo responde
`disponivel: false` com o motivo, e nao quebra nada.

Sem a caixa envolvente nao ha como ligar coordenada a patch. Chutar o patch
mais proximo seria pior que nao responder: colaria uma imagem de outro lugar
numa resposta que o usuario leria como sendo daquele ponto.
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
MANIFESTO = (RUNS / "dino_embeddings" / "v1ge"
             / "dino_expanded_embedding_manifest_v1ge.csv")
OUT = RUNS / "evidencia-visual-review-only"

VAR_SENTINEL = "REVP_SENTINEL_DIR"

NUNCA_LIDO_POR_PORTAO = (
    "ilustrativa. Nenhum portao do contrato consulta este campo, ele nao entra "
    "no escore e nao desloca o IC")


def _regiao_chave(valor: str) -> str:
    texto = str(valor or "").strip().lower().replace("�", "o")
    return "".join(c for c in unicodedata.normalize("NFD", texto)
                   if unicodedata.category(c) != "Mn")


def _corpus() -> list[dict]:
    """Os embeddings que existem, direto do manifesto ja versionado."""
    if not MANIFESTO.exists():
        return []
    import pandas as pd

    d = pd.read_csv(MANIFESTO)
    d = d[d.success == "SUCCESS"] if "success" in d.columns else d
    return d.to_dict("records")


def _caixas_dos_patches() -> dict:
    """Caixa envolvente WGS84 de cada patch, se os GeoTIFF estiverem la.

    Sem eles nao ha ligacao coordenada -> patch, e o modulo diz isso em vez de
    escolher o patch mais proximo.
    """
    raiz = os.environ.get(VAR_SENTINEL, "")
    if not raiz or not Path(raiz).is_dir():
        return {}
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError:
        return {}

    caixas = {}
    for tif in sorted(Path(raiz).glob("*.tif")):
        try:
            with rasterio.open(tif) as src:
                caixas[tif.stem] = transform_bounds(src.crs, "EPSG:4326",
                                                    *src.bounds)
        except Exception:
            continue
    return caixas


def consultar(resposta: dict) -> dict:
    """O campo `evidencia_visual` da resposta.

    Nao recebe coordenada: o contrato nunca ecoa a geometria da requisicao, e
    este modulo nao e lugar para reintroduzi-la. O que ele responde e se o
    corpus cobre a REGIAO, e o que impede resolver o ponto quando nao cobre.
    """
    regiao = _regiao_chave(resposta.get("regiao") or "")
    corpus = _corpus()
    caixas = _caixas_dos_patches()

    do_regiao = [r for r in corpus
                 if _regiao_chave(r.get("region", "")) == regiao]

    base = {
        "disponivel": False,
        "similaridade_a_positivos_conhecidos": None,
        "embeddings_no_corpus": len(corpus),
        "embeddings_nesta_regiao": len(do_regiao),
        "backbone": (corpus[0].get("model_backbone") if corpus else None),
        "papel": NUNCA_LIDO_POR_PORTAO,
        "nunca_lida_por_gate": True,
    }

    if not corpus:
        base["motivo"] = (
            f"corpus de embeddings ausente em {MANIFESTO.parent}; nenhuma "
            "evidencia visual pode ser afirmada")
        return base
    if not do_regiao:
        base["motivo"] = (
            f"o corpus tem {len(corpus)} embeddings, nenhum em {regiao!r}")
        return base
    if not caixas:
        base["motivo"] = (
            f"{len(do_regiao)} embeddings existem em {regiao!r}, mas a caixa "
            f"envolvente de cada patch exige os GeoTIFF de origem; configure "
            f"{VAR_SENTINEL} para liga-los a coordenada. Escolher o patch mais "
            "proximo sem isso colaria imagem de outro lugar na resposta")
        return base

    # Com corpus e caixas disponiveis o vinculo por ponto passa a ser possivel.
    # Ele nao e feito aqui de proposito: exigiria a coordenada da requisicao,
    # e ligar patch a ponto e trabalho do resolvedor, nao deste modulo.
    base["motivo"] = (
        "corpus e caixas disponiveis; o vinculo por ponto ainda nao esta "
        "implementado -- ver E8.4")
    base["patches_candidatos"] = len(caixas)
    return base


def auditar() -> dict:
    """Gera um retrato auditavel da frente visual review-only.

    A auditoria nao executa embedding nem baixa imagem. Ela confirma o que o
    produto pode dizer hoje: se existe manifesto local, quantos vetores ha por
    regiao, se as caixas dos TIFs podem ser resolvidas e se a politica
    "nunca lida por gate" continua declarada.
    """
    corpus = _corpus()
    caixas = _caixas_dos_patches()
    por_regiao: dict[str, int] = {}
    for row in corpus:
        regiao = _regiao_chave(row.get("region", "")) or "sem_regiao"
        por_regiao[regiao] = por_regiao.get(regiao, 0) + 1

    consultas = {
        regiao: consultar({"status": "ok", "regiao": regiao})
        for regiao in ("recife", "curitiba", "petropolis")
    }
    checks = [
        {
            "nome": "manifesto_existe",
            "passou": MANIFESTO.exists(),
            "detalhe": str(MANIFESTO),
        },
        {
            "nome": "corpus_review_only",
            "passou": all(
                str(r.get("claim_scope", "")).upper().endswith("NO_PREDICTIVE_CLAIM")
                or "REVIEW" in str(r.get("claim_scope", "")).upper()
                for r in corpus),
            "detalhe": f"n={len(corpus)}",
        },
        {
            "nome": "nunca_lida_por_gate",
            "passou": all(c.get("nunca_lida_por_gate") is True
                          for c in consultas.values()),
            "detalhe": NUNCA_LIDO_POR_PORTAO,
        },
        {
            "nome": "sem_patch_por_chute",
            "passou": not caixas or all(not c.get("disponivel")
                                        for c in consultas.values()),
            "detalhe": (
                "sem caixa de TIF, consulta declara indisponibilidade em vez "
                "de escolher patch proximo"),
        },
    ]
    return {
        "tipo": "auditoria_evidencia_visual_review_only",
        "quando_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifesto": str(MANIFESTO),
        "sentinel_dir_env": VAR_SENTINEL,
        "sentinel_dir_configurado": bool(os.environ.get(VAR_SENTINEL)),
        "embeddings_no_corpus": len(corpus),
        "por_regiao": por_regiao,
        "patches_com_caixa": len(caixas),
        "consultas": consultas,
        "checks": checks,
        "veredito": "APROVADO" if all(c["passou"] for c in checks) else "REPROVADO",
    }


def gravar_auditoria(resultado: dict, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultado.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    linhas = [
        "# Auditoria de evidencia visual review-only",
        "",
        f"Veredito: `{resultado['veredito']}`.",
        f"Embeddings no corpus: `{resultado['embeddings_no_corpus']}`.",
        "",
        "## Checks",
        "",
    ]
    for c in resultado["checks"]:
        linhas.append(f"- `{'OK' if c['passou'] else 'FALHA'}` {c['nome']}: {c['detalhe']}")
    (out / "RELATORIO.md").write_text("\n".join(linhas) + "\n",
                                      encoding="utf-8")


def main() -> int:
    if "--auditar" not in sys.argv:
        print("uso: python scripts/servico/evidencia_visual.py --auditar")
        return 2
    resultado = auditar()
    gravar_auditoria(resultado)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return 0 if resultado["veredito"] == "APROVADO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
