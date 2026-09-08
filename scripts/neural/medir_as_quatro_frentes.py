"""As quatro frentes na mesma regua de predicao seletiva.

Cada frente vira um par `(confianca, erro)` e passa pelo mesmo aparato. O que
muda entre elas e o que significa errar; o que nao muda e a pergunta:

    a confianca deste componente ordena os casos melhor que o acaso, e quanto
    risco ele remove por unidade de cobertura que abre mao?

-------------------------------------------------------------------------
COMO CADA FRENTE VIRA UM SELETOR
-------------------------------------------------------------------------

  A  confianca = margem media onde a rede prediz agua
     erro      = 1 - IoU do chip (sem agua permanente)

  B  confianca = |escore predito - mediana|, a distancia ao centro do ranking
     erro      = erro absoluto de ordenacao (posicao predita vs real)

  C  confianca = similaridade do primeiro resultado
     erro      = 1 se o chunk correto nao esta em k, 0 se esta

  D  confianca = fracao lexica de sobreposicao
     erro      = 1 se a decisao diverge do gabarito

O `erro` de A e continuo e o de C e binario. Isso e legitimo em risco-cobertura
-- o risco e a media do erro, e a media de um erro continuo e uma perda media.
O que NAO seria legitimo e comparar o AURC de A com o de C como se fossem a
mesma escala, e por isso o relatorio compara cada frente ao SEU acaso.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "neural"))

from predicao_seletiva import avaliar_seletor  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "sintese-neural"


def _ler_csv(caminho: Path) -> list[dict]:
    if not caminho.exists():
        return []
    with caminho.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def frente_a() -> dict | None:
    """A rede de SAR: margem onde prediz agua contra 1 - IoU."""
    import torch

    sys.path.insert(0, str(REPO / "scripts" / "sar"))
    from avaliar_rede import (MARGEM_NA_PREDICAO, carregar, confianca,
                              prever)
    from dados_sen1floods11 import ler_chip, nomes_do_split

    try:
        modelo, manifesto = carregar("v1")
    except FileNotFoundError:
        return None

    linhas = _ler_csv(REPO / "local_runs" / "sar-inundacao"
                      / "avaliacao_unet_v1_teste_por_chip.csv")
    if not linhas:
        return None
    iou = {r["nome"]: r["iou_sem_permanente"] for r in linhas}

    confiancas, erros, nomes = [], [], []
    with torch.no_grad():
        for nome in nomes_do_split("teste"):
            valor = iou.get(nome)
            if not valor:                     # IoU indefinido sai da conta
                continue
            chip = ler_chip(nome)
            pred, prob = prever(modelo, nome, manifesto["normalizacao"])
            valido = chip.valido & np.isfinite(chip.s1).all(axis=0)
            c = confianca(prob, valido, pred, MARGEM_NA_PREDICAO)
            if c is None:                     # sem predicao de agua: sem margem
                continue
            confiancas.append(c)
            erros.append(1.0 - float(valor))
            nomes.append(nome)

    if not confiancas:
        return None
    return {**avaliar_seletor("A_rede_sar", confiancas, erros),
            "o_que_e_erro": "1 - IoU do chip, sem agua permanente",
            "o_que_e_confianca": "margem media onde a rede prediz agua",
            "unidade": "chip"}


def frente_b() -> dict | None:
    """O embedding de terreno: distancia ao centro contra erro de ordenacao."""
    caminho = (REPO / "local_runs" / "terreno-encoder"
               / "avaliacao_embedding_v1_combinado.json")
    if not caminho.exists():
        return None
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    por_grupo = dados.get("spearman_por_grupo") or {}

    # Aqui o "caso" e o PAIS, e nao o chip: e a unidade em que a frente B foi
    # avaliada, e mudar de unidade para conseguir mais pontos seria trocar a
    # pergunta para melhorar o numero.
    nomes, confiancas, erros = [], [], []
    for pais, v in por_grupo.items():
        s = v.get("spearman")
        if s is None:
            continue
        nomes.append(pais)
        confiancas.append(float(v.get("n", 0)))     # confianca = tamanho da amostra
        erros.append(1.0 - float(s))                # erro = 1 - Spearman
    if len(nomes) < 3:
        return None
    return {**avaliar_seletor("B_embedding_terreno", confiancas, erros),
            "o_que_e_erro": "1 - Spearman do pais",
            "o_que_e_confianca": "numero de chips do pais",
            "unidade": "pais",
            "aviso": ("com 10 paises a curva tem 10 pontos; o intervalo do "
                      "acaso e largo e a leitura correta e qualitativa")}


def frente_c() -> dict | None:
    """A recuperacao: similaridade do primeiro resultado contra acerto em k."""
    sys.path.insert(0, str(REPO / "scripts" / "recuperacao"))
    try:
        from avaliacao_de_recuperacao import montar_conjunto
        from indice_denso import construir
    except Exception:
        return None

    indice = construir()
    confiancas, erros = [], []
    for caso in montar_conjunto():
        achados = indice.buscar(caso["consulta"],
                                classe=caso["classe_autoridade"], k=5)
        if not achados:
            # abstencao explicita do indice: confianca minima, erro certo
            confiancas.append(0.0)
            erros.append(1.0)
            continue
        confiancas.append(float(achados[0]["similaridade"]))
        erros.append(0.0 if any(a["chunk_id"] == caso["chunk_id"]
                                for a in achados) else 1.0)
    if not confiancas:
        return None
    return {**avaliar_seletor("C_recuperacao_densa", confiancas, erros),
            "o_que_e_erro": "1 se o chunk correto nao esta em k=5",
            "o_que_e_confianca": "similaridade do primeiro resultado",
            "unidade": "consulta"}


def frente_d() -> dict | None:
    """O suporte: sobreposicao lexica contra divergencia do gabarito."""
    caminho = (REPO / "local_runs" / "verificacao-suporte"
               / "medida_de_suporte_com_juiz.json")
    if not caminho.exists():
        return None
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    casos = dados.get("por_caso") or []
    if not casos:
        return None

    confiancas = [float(c["fracao_lexica"]) for c in casos]
    erro_lexico = [0.0 if c["lexico_correto"] else 1.0 for c in casos]
    erro_conj = [0.0 if c["conjuncao_correta"] else 1.0 for c in casos]

    return {
        "lexico": {**avaliar_seletor("D_lexico", confiancas, erro_lexico),
                   "o_que_e_erro": "decisao lexica diverge do gabarito"},
        "conjuncao": {**avaliar_seletor("D_conjuncao", confiancas, erro_conj),
                      "o_que_e_erro": "decisao da conjuncao diverge do gabarito"},
        "o_que_e_confianca": "fracao de sobreposicao lexica",
        "unidade": "par (claim, trecho)",
    }


def executar() -> dict:
    frentes = {}
    for nome, funcao in (("A", frente_a), ("B", frente_b),
                         ("C", frente_c), ("D", frente_d)):
        try:
            resultado = funcao()
        except Exception as e:
            resultado = {"indisponivel": f"{type(e).__name__}: {e}"}
        frentes[nome] = resultado or {"indisponivel": "artefato ausente"}

    resumo = {
        "tipo": "sintese_predicao_seletiva",
        "quando_utc": agora_utc(),
        "enquadramento": (
            "El-Yaniv e Wiener 2010: o classificador pode se abster, trocando "
            "cobertura por risco menor na regiao onde responde"),
        "regua": ("AURC (area sob a curva risco-cobertura), menor e melhor, "
                  "sempre contra o acaso da PROPRIA frente"),
        "por_que_nao_comparar_aurc_entre_frentes": (
            "o erro de A e continuo e o de C e binario; a comparacao valida e "
            "de cada frente com o seu proprio acaso, e nao entre elas"),
        "frentes": frentes,
        "nao_e": ("ranking de qualidade das frentes nem autorizacao para "
                  "encadea-las: AURC diz que a confianca ORDENA, e ordenar nao "
                  "e acertar; o desempenho de cada frente continua no artefato "
                  "dela"),
    }
    resumo["hash_resumo"] = sha256_json(
        {k: v for k, v in resumo.items() if k != "quando_utc"})
    return resumo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gravar", action="store_true")
    args = parser.parse_args()

    resumo = executar()
    if args.gravar:
        SAIDA.mkdir(parents=True, exist_ok=True)
        destino = SAIDA / "predicao_seletiva_das_frentes.json"
        destino.write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        resumo["gravado_em"] = str(destino)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
