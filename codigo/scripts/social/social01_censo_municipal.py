"""Contexto municipal oficial de saneamento, sem alterar o modelo fisico.

A tabela 6805 mede domicilios, nao pessoas expostas a inundacao. As categorias
hierarquicas nao podem ser somadas duas vezes. Dados ausentes ficam ausentes.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/servico"))
from proveniencia import agora_utc, sha256_arquivo, sha256_json

BASE = "https://servicodados.ibge.gov.br/api/v3/agregados/6805"
ANO = "2022"
PADRAO = ("2611606", "3303906", "4106902", "4314902")


def obter_json(url: str) -> object:
    for tentativa in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "REV-P-contexto-social/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resposta:
                bruto = resposta.read(8 * 1024 * 1024 + 1)
            if len(bruto) > 8 * 1024 * 1024:
                raise ValueError("consulta excedeu limite; reduzir recorte")
            if bruto[:2] == b"\x1f\x8b":
                bruto = gzip.decompress(bruto)
            if len(bruto) > 32 * 1024 * 1024:
                raise ValueError("resposta descompactada excedeu limite")
            return json.loads(bruto)
        except (OSError, TimeoutError):
            if tentativa == 2:
                raise
            time.sleep(2 ** tentativa)
    raise RuntimeError("consulta nao concluida")


def normalizar(meta: dict, resposta: list, municipios: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    if str(meta.get("id")) != "6805" or "N6" not in meta.get("nivelTerritorial", {}).get("Administrativo", []):
        raise ValueError("metadados nao correspondem a tabela municipal esperada")
    classificacoes = [c for c in meta["classificacoes"] if str(c["id"]) == "11558"]
    if len(classificacoes) != 1:
        raise ValueError("classificacao de esgotamento ausente ou ambigua")
    categorias = {str(c["id"]): c for c in classificacoes[0]["categorias"]}
    totais = [c for c, info in categorias.items() if info["nivel"] == 0]
    if len(totais) != 1:
        raise ValueError("total da classificacao ambiguo")
    folhas = {c for c, info in categorias.items() if info["nivel"] == 1}
    vistos, linhas = set(), []
    for variavel in resposta:
        if str(variavel["id"]) != "381" or variavel["unidade"] != "Unidades":
            raise ValueError("esperado numero de domicilios, nao percentual ou populacao")
        for resultado in variavel["resultados"]:
            cls = resultado["classificacoes"]
            if len(cls) != 1 or str(cls[0]["id"]) != "11558" or len(cls[0]["categoria"]) != 1:
                raise ValueError("classificacao inesperada na resposta")
            categoria = next(iter(cls[0]["categoria"]))
            if categoria not in categorias:
                raise ValueError("categoria nao consta dos metadados")
            for serie in resultado["series"]:
                local = serie["localidade"]
                codigo = str(local["id"])
                if codigo not in municipios or local["nivel"]["id"] != "N6":
                    raise ValueError("resposta fora do recorte municipal")
                chave = (codigo, categoria)
                if chave in vistos:
                    raise ValueError("serie municipal duplicada")
                vistos.add(chave)
                valor = serie["serie"].get(ANO)
                n = int(valor) if isinstance(valor, str) and re.fullmatch(r"\d+", valor) else None
                linhas.append({"municipio_id": codigo, "municipio": local["nome"], "ano": ANO,
                               "categoria_id": categoria, "categoria": categorias[categoria]["nome"],
                               "nivel_categoria": categorias[categoria]["nivel"], "domicilios": n,
                               "estado": "observado" if n is not None else "nao_disponivel"})
    controle = []
    for municipio in municipios:
        itens = {l["categoria_id"]: l for l in linhas if l["municipio_id"] == municipio}
        total = itens.get(totais[0], {}).get("domicilios")
        valores = [itens.get(c, {}).get("domicilios") for c in folhas]
        completo = total is not None and all(v is not None for v in valores) and set(itens) == set(categorias)
        if completo and sum(valores) != total:
            raise ValueError(f"categorias de primeiro nivel nao fecham o total de {municipio}")
        for item in itens.values():
            n = item["domicilios"]
            if n is not None and total is not None and n > total:
                raise ValueError("categoria maior que o total")
            item["percentual_domicilios"] = round(100 * n / total, 6) if total and n is not None else None
        controle.append({"municipio_id": municipio, "categorias_completas": completo,
                         "soma_primeiro_nivel_confere": completo, "total_domicilios": total})
    if not linhas:
        raise ValueError("resposta sem series municipais")
    return sorted(linhas, key=lambda l: (l["municipio_id"], l["categoria_id"])), controle


def executar(saida: Path, municipios: tuple[str, ...]) -> Path:
    municipios = tuple(sorted(set(municipios)))
    if not municipios or len(municipios) > 20 or any(not re.fullmatch(r"\d{7}", m) for m in municipios):
        raise ValueError("selecionar de 1 a 20 codigos municipais IBGE de sete digitos")
    url = BASE + f"/periodos/{ANO}/variaveis/381?localidades=N6[{','.join(municipios)}]&classificacao=11558[all]"
    meta, dados = obter_json(BASE + "/metadados"), obter_json(url)
    linhas, controle = normalizar(meta, dados, municipios)
    origem = {"metadados": meta, "dados": dados}
    codigo_hash = sha256_arquivo(Path(__file__))
    destino = Path(saida).resolve() / sha256_json({"origem": origem, "codigo": codigo_hash}).split(":")[-1]
    destino.mkdir(parents=True, exist_ok=True)
    r = {"tipo": "social01_censo_municipal", "tabela": "6805", "ano": ANO, "escala": "municipio",
         "urls": [BASE + "/metadados", url], "codigo_sha256": codigo_hash,
         "origem_sha256_json": sha256_json(origem), "municipios": municipios, "controle": controle,
         "linhas": len(linhas), "usado_no_modelo_fisico": False, "promovido": False,
         "estado": "contexto_oficial_adquirido" if all(c["categorias_completas"] for c in controle) else "contexto_parcial",
         "atribuicao": "IBGE, Censo Demografico 2022, SIDRA tabela 6805; preservar fonte e categorias originais",
         "limites": ["nao e indice validado de vulnerabilidade", "nao e estimativa de pessoas inundadas",
                     "nao desagregar municipio para pixel ou setor", "rede geral ou pluvial nao distingue drenagem de esgoto tratado",
                     "referencia 2022 nao descreve automaticamente outro ano", "aquisicao nao altera o contrato social v0"]}
    arq = destino / "manifesto.json"
    if arq.exists():
        antigo = json.loads(arq.read_text(encoding="utf-8"))
        corpo = {k: v for k, v in antigo.items() if k not in ("quando_utc", "hash_resumo", "indicadores_sha256", "origem_arquivo_sha256")}
        if sha256_json(corpo) != sha256_json(r):
            raise RuntimeError("mesma origem com processamento diferente; preservar snapshot")
        for arquivo, chave in (("indicadores.csv", "indicadores_sha256"), ("origem.json", "origem_arquivo_sha256")):
            if sha256_arquivo(destino / arquivo) != antigo.get(chave):
                raise RuntimeError("snapshot existente adulterado")
        if sha256_json({k: v for k, v in antigo.items() if k not in ("hash_resumo", "quando_utc")}) != antigo.get("hash_resumo"):
            raise RuntimeError("manifesto existente divergente")
        return arq
    (destino / "origem.json").write_text(json.dumps(origem, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (destino / "indicadores.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    r["indicadores_sha256"] = sha256_arquivo(destino / "indicadores.csv")
    r["origem_arquivo_sha256"] = sha256_arquivo(destino / "origem.json")
    r["hash_resumo"] = sha256_json(r)
    r["quando_utc"] = agora_utc()
    arq.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return arq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--municipios", nargs="+", default=PADRAO)
    args = ap.parse_args()
    print(f"GRAVADO={executar(args.saida, tuple(args.municipios))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
