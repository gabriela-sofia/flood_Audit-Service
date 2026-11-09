"""Que afirmacao cada modelo servido sustenta, segundo a regra de uso da admissao.

-------------------------------------------------------------------------
POR QUE ISTO EXISTE
-------------------------------------------------------------------------

A admissao central (`admissao_cientifica`) bloqueia ajuste sobre negativo
de ausencia -- mas os modelos SERVIDOS (`svc01`) foram ajustados antes dela,
e alguns sobre exatamente isso. O de Recife tem 122 dos 124 negativos como
ausencia de registro; os de Nova York e Sao Paulo comparam chamado de agua
com chamado de outro assunto. O escore deles e legitimo, mas mede outra
coisa: a PROPENSAO A REGISTRAR agua, nao "inundou contra nao inundou". E o
`pluvial_urbano_observado` foi ajustado no UFO, que o contrato declara
MISTO_NAO_SEPARAVEL: a etiqueta pluvial nao e sustentada pelo instrumento.

Reajustar ou tirar de servico seria mudar o comportamento do produto, e o
`CONTRATO_VERSIONAMENTO` exige decisao para isso. O que este modulo faz e
impedir que o cartao AFIRME mais do que o ajuste mede: cada modelo recebe o
estimando que ele sustenta, e o `svc02` coloca esse estimando no cartao de
toda resposta.

  suscetibilidade_a_inundacao        so negativo observado; mecanismo do
                                     instrumento igual ao do modelo
  suscetibilidade_com_ressalva       ha negativo inferido por exclusao
  propensao_a_registrar              ha negativo de ausencia de registro
  mecanismo_nao_sustentado           a fonte nao separa o mecanismo rotulado

nao_e: nao reajusta nem tira de servico: diz o que cada escore pode afirmar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELOS = REPO / "local_runs" / "svc-01-modelos"
SAIDA = REPO / "local_runs" / "afirmacao-dos-servidos"
for pasta in ("metodo", "servico", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

FRASE = {
    "suscetibilidade_a_inundacao": "escore de suscetibilidade a inundacao (negativo observado)",
    "suscetibilidade_com_ressalva": ("escore de suscetibilidade com ressalva: parte do negativo e "
                                     "inferida por exclusao, nao vista"),
    "propensao_a_registrar": ("escore de propensao a REGISTRAR agua (chamado/ocorrencia contra ausencia "
                              "de registro); nao e probabilidade de inundar"),
    "mecanismo_nao_sustentado": ("escore de agua observada sem mecanismo separavel; a etiqueta do "
                                 "modelo nao e sustentada pela fonte"),
}


def classificar(modelo: dict) -> dict:
    import contrato_do_fenomeno as cf
    niveis = {k for k, n in (modelo.get("niveis_de_negativo") or {}).items() if n}
    fontes = set((modelo.get("fontes_do_ajuste") or {}).keys())
    mec = modelo.get("mecanismo")
    mec_fontes = {cf.INSTRUMENTOS.get(f, {}).get("mecanismo") for f in fontes}
    # FLUVIAL_AREAL e o nome antigo do mesmo mecanismo de cheia areal.
    mec_norm = "FLUVIAL_ENXURRADA" if mec == "FLUVIAL_AREAL" else mec
    motivos = []
    if fontes and mec_fontes - {mec_norm}:
        estimando = "mecanismo_nao_sustentado"
        motivos.append(f"fontes {sorted(fontes)} observam {sorted(m for m in mec_fontes if m)}, nao {mec}")
    elif "ausencia" in niveis:
        estimando = "propensao_a_registrar"
        motivos.append("negativo de ausencia de registro no ajuste")
    elif "exclusao_qualificada" in niveis:
        estimando = "suscetibilidade_com_ressalva"
        motivos.append("negativo inferido por exclusao no ajuste")
    elif niveis and niveis <= {"observado", "nao_aplicavel"}:
        estimando = "suscetibilidade_a_inundacao"
    else:
        estimando = "propensao_a_registrar"
        motivos.append("nivel de negativo nao declarado no artefato: tratado pelo mais fraco")
    return {"modelo": modelo.get("nome"), "mecanismo_rotulado": mec, "fontes": sorted(fontes),
            "niveis_de_negativo": sorted(niveis), "estimando": estimando, "frase": FRASE[estimando],
            "motivos": motivos}


def estimando_do_modelo(modelo: dict) -> dict:
    """O que o cartao de resposta pode afirmar com este modelo."""
    return classificar(modelo)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    from proveniencia import agora_utc, sha256_json
    import rastreio_de_impacto as ri
    r = {"tipo": "afirmacao_dos_servidos", "quando_utc": agora_utc(), "modelos": [],
         "nao_e": ("nao e validacao dos modelos nem muda quem e servido ou recusado: so declara o que cada "
                   "escore pode afirmar pela regra de uso da admissao")}
    for f in sorted(MODELOS.glob("*.json")):
        if f.name == "indice.json":
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        r["modelos"].append({**classificar(m), "servivel": m.get("servivel")})
    r["nao_e"] = "nao reajusta nem tira de servico: diz o que cada escore pode afirmar"
    r["hash_resumo"] = sha256_json({k: v for k, v in r.items() if k != "quando_utc"})
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "afirmacao.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    ri.carimbar("afirmacao_dos_servidos", "afirmacao_dos_servidos")
    for m in r["modelos"]:
        print(f"  {m['modelo']:26s} servivel={m['servivel']!s:5s} {m['estimando']:30s} {'; '.join(m['motivos'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
