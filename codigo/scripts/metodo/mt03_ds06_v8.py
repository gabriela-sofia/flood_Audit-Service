"""MT03, passo 1: ds06 v8 = ds06 v7 com a cobertura e o solo da v2, em base SEPARADA.

POR QUE BASE SEPARADA
`ds06_dataset_de_treino.versao_atual()` devolve a maior versao com manifesto dentro de
`ds-06-dataset-de-treino/`, e ~30 consumidores leem por ela. Gravar o v8 ali migraria todos de uma
vez, inclusive os de outras sessoes e os que reproduzem experimentos congelados. O v8 vai para
`ds-06-dataset-de-treino-cobertura-v2/v8/`, no mesmo formato (pontos, excluidos, manifesto), e quem
quiser le-lo pede `versao_atual(BASE_V8)`. Promover o v8 a versao vigente e decisao de pipeline.

O QUE MUDA DO v7 PARA O v8
So as 9 colunas de cobertura e solo, trocadas pelas da `cob-02-cobertura-v7` (v2), e 11 colunas de
procedencia novas: origem da linha (reaproveitada do cob01 ou recoletada), ano do evento, motivo da
ausencia por atributo e defasagem entre o ano da camada e o do evento. Todas as outras colunas tem de
sair identicas, coluna a coluna, e isso e checado antes de gravar.

nao_e: nao reconstroi a cadeia ds05 -> ds06 (o v8 parte do v7 gravado, para que a unica diferenca seja
a cobertura), nao imputa ausencia, nao altera o v7, o cob01 nem o doc do ds06, e nao identifica producao
de escoamento: corrigir disponibilidade de camada nao diz quanto a superficie gera de agua.

Uso:
    python scripts/metodo/mt03_ds06_v8.py construir
    python scripts/metodo/mt03_ds06_v8.py consumidores
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "geo", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402
import ds06_dataset_de_treino as ds06  # noqa: E402

RUNS = REPO / "local_runs"
V7 = RUNS / "ds-06-dataset-de-treino" / "v7"
COB_V2 = RUNS / "cob-02-cobertura-v7"
BASE_V8 = RUNS / "ds-06-dataset-de-treino-cobertura-v2"
VERSAO = "v8"
COBERTURA = ds06.COLUNAS_DE_COBERTURA
ATRIBUTOS = ("mapbiomas_classe", "modis_lc", "construido_frac", "worldcover_2021", "argila_pct_0_30",
             "areia_pct_0_30")
PROCEDENCIA = ["cobertura_origem", "ano_evento", *[f"motivo_{c}" for c in ATRIBUTOS],
               "defasagem_mapbiomas_anos", "defasagem_modis_anos", "defasagem_ghsl_anos"]
# consumidores que reproduzem resultado congelado: ficam no v7 por contrato
CONGELADOS = {"scripts/metodo/mt01_populacao.py": "MT01 (pre-registro 1e8a6f57)",
              "scripts/metodo/mt02_cobertura_v2.py": "produtor da cobertura v2 (le o v7 por definicao)",
              "scripts/metodo/contexto_de_desenvolvimento.py": "fixa v7 por caminho",
              "scripts/metodo/fechamento_do_framework.py": "confere o hash do ds06 da versao declarada",
              "scripts/geo/acervo_espacial.py": "fixa v7 por caminho"}


def construir_v8() -> tuple[pd.DataFrame, pd.DataFrame]:
    v7 = pd.read_parquet(V7 / "pontos.parquet")
    c = pd.read_parquet(COB_V2 / "cobertura_e_solo_v2.parquet")
    if c.ponto_id.duplicated().any() or set(c.ponto_id) != set(v7.ponto_id):
        raise ValueError("cobertura v2 nao cobre exatamente os pontos do v7")
    c = c.assign(cobertura_origem=c.origem,
                 defasagem_mapbiomas_anos=c.mapbiomas_ano - c.ano_evento,
                 defasagem_modis_anos=c.modis_ano - c.ano_evento,
                 defasagem_ghsl_anos=c.ghsl_ano - c.ano_evento)
    v8 = v7.drop(columns=COBERTURA).merge(c[["ponto_id", *COBERTURA, *PROCEDENCIA]], on="ponto_id",
                                          how="left", validate="one_to_one")
    v8 = v8[[*v7.columns, *PROCEDENCIA]]
    return v7, v8


def auditar(v7: pd.DataFrame, v8: pd.DataFrame) -> dict:
    fixas = [c for c in v7.columns if c not in COBERTURA]
    iguais = {c: bool(v7[c].equals(v8[c])) for c in fixas}
    a = {"colunas_fixas_identicas": iguais,
         "ids": {"ponto_id_iguais_e_na_mesma_ordem": bool(v7.ponto_id.equals(v8.ponto_id)),
                 "unicos": bool(not v8.ponto_id.duplicated().any())},
         "rotulos": {"classe_identica": iguais["classe"], "qualidade_negativo_identica": iguais["qualidade_negativo"],
                     "papel_identico": iguais["papel"]},
         "datas": {"janela_inicio_identica": iguais["janela_inicio"], "janela_fim_identica": iguais["janela_fim"],
                   "precisao_data_identica": iguais["precisao_data"]}}
    mud = {}
    for col in COBERTURA:
        x, y = v7[col], v8[col]
        diferente = ~((x == y) | (x.isna() & y.isna()))
        mud[col] = {"linhas_diferentes": int(diferente.sum()),
                    "preenchido_v7": round(float(x.notna().mean()), 4),
                    "preenchido_v8": round(float(y.notna().mean()), 4),
                    "diferentes_por_fonte": {str(k): int(v) for k, v in
                                             v8.loc[diferente, "fonte"].value_counts().items()}}
    a["cobertura_alterada"] = mud
    so_mudou_onde_recoletou = all(
        not ((~((v7[c] == v8[c]) | (v7[c].isna() & v8[c].isna()))) & (v8.cobertura_origem != "recoleta_v2")).any()
        for c in COBERTURA)
    a["so_mudou_nas_linhas_recoletadas"] = bool(so_mudou_onde_recoletou)
    fxc = v8.fonte + "|" + v8.classe.astype(str)
    a["preenchimento_por_fonte_e_classe"] = {
        k: {c: round(float(g[c].notna().mean()), 4) for c in ATRIBUTOS} for k, g in v8.groupby(fxc)}
    a["mascaras"] = {c: {str(k): int(v) for k, v in v8[f"motivo_{c}"].value_counts().items()} for c in ATRIBUTOS}
    a["mascara_por_classe_solo"] = {
        k: round(float((g.motivo_areia_pct_0_30 == "mascara_da_camada").mean()), 4) for k, g in v8.groupby(fxc)}
    a["camadas_emprestadas"] = {
        nome: {"linhas_com_ano": int(v8[col].notna().sum()),
               "defasagem_zero": int((v8[col] == 0).sum()),
               "defasagem_distribuicao": {str(int(k)): int(v) for k, v in v8[col].value_counts().sort_index().items()}}
        for nome, col in (("mapbiomas", "defasagem_mapbiomas_anos"), ("modis", "defasagem_modis_anos"),
                          ("ghsl", "defasagem_ghsl_anos"))}
    a["camadas_estaticas"] = {"worldcover_2021": "ano fixo 2021 para todo evento",
                              "soilgrids": "estatico; mascara agua e area urbana densa"}
    return a


def construir() -> dict:
    destino = BASE_V8 / VERSAO
    if (destino / "manifesto.json").exists():
        raise FileExistsError(f"{destino} ja existe: v8 nao e sobrescrito")
    v7, v8 = construir_v8()
    aud = auditar(v7, v8)
    checagens = ds06.validar(v8)
    checagens["colunas_fixas_identicas_ao_v7"] = all(aud["colunas_fixas_identicas"].values())
    checagens["so_mudou_nas_linhas_recoletadas"] = aud["so_mudou_nas_linhas_recoletadas"]
    if not all(checagens.values()):
        raise SystemExit(f"v8 reprovado: {[k for k, v in checagens.items() if not v]}")
    m7 = json.loads((V7 / "manifesto.json").read_text(encoding="utf-8"))
    entradas_hoje = {"tabela_unica": ds06.TABELA, "pontos_gfd": ds06.GFD / "pontos_gfd_com_terreno.csv",
                     "catalogo_gfd": ds06.GFD / "catalogo.json", "lista_ampliada": ds06.LISTA_AMPLIADA,
                     "cobertura_e_solo": ds06.COBERTURA}
    deriva = {k: {"no_v7": m7["entradas"].get(k), "hoje": sha256_arquivo(p) if p.exists() else None}
              for k, p in entradas_hoje.items()}
    destino.mkdir(parents=True, exist_ok=False)
    v8.to_parquet(destino / "pontos.parquet", index=False)
    shutil.copyfile(V7 / "excluidos.parquet", destino / "excluidos.parquet")
    m = {"tipo": "dataset_de_treino", "versao": VERSAO, "base": str(BASE_V8.relative_to(REPO)),
         "deriva_de": {"ds06_v7_pontos": m7["saidas"]["pontos"], "ds06_v7_manifesto": m7["hash_resumo"],
                       "cobertura_v2_manifesto": sha256_arquivo(COB_V2 / "manifesto.json"),
                       "cobertura_v2_parquet": sha256_arquivo(COB_V2 / "cobertura_e_solo_v2.parquet")},
         "entradas_do_v7_hoje": deriva,
         "linhas": int(len(v8)), "colunas_novas": PROCEDENCIA,
         "contextos_prontos_para_treino": m7["contextos_prontos_para_treino"],
         "contextos": m7["contextos"], "eventos": m7["eventos"],
         "cobertura_e_solo": {c: round(float(v8[c].notna().mean()), 4) for c in COBERTURA if not c.endswith("_ano")},
         "checagens": checagens, "auditoria": aud,
         "codigo": {str(Path(__file__).relative_to(REPO)): sha256_arquivo(Path(__file__))},
         "nao_e": ("v7 com cobertura e solo corrigidos; nao e versao vigente do ds06, nao imputa ausencia e nao "
                   "identifica producao de escoamento")}
    m["saidas"] = {"pontos": sha256_arquivo(destino / "pontos.parquet"),
                   "excluidos": sha256_arquivo(destino / "excluidos.parquet")}
    m["hash_resumo"] = sha256_json(m)
    m["quando_utc"] = agora_utc()
    (destino / "manifesto.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str),
                                            encoding="utf-8")
    return {k: m[k] for k in ("linhas", "checagens", "saidas", "entradas_do_v7_hoje")} | {
        "cobertura_alterada": aud["cobertura_alterada"]}


def consumidores() -> dict:
    """Quem le o ds06, por qual caminho, e se o v8 muda o que ele calcula."""
    fixas = None
    try:
        v7c = pd.read_parquet(V7 / "pontos.parquet").columns
        fixas = [c for c in v7c if c not in COBERTURA]
    except FileNotFoundError:
        pass
    linhas = []
    for p in sorted((REPO / "scripts").rglob("*.py")):
        rel = p.relative_to(REPO).as_posix()
        if rel.endswith("ds06_dataset_de_treino.py") or rel.startswith("scripts/metodo/mt03_"):
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        if "versao_atual" not in t and "ds-06-dataset-de-treino" not in t:
            continue
        usa_vigente = bool(re.search(r"versao_atual\(\)", t))
        fixa = sorted(set(re.findall(r"ds-06-dataset-de-treino[\"'/ ,]+\"?(v\d+)", t)))
        cobre = sorted(c for c in COBERTURA if re.search(rf"\b{c}\b", t))
        if rel in CONGELADOS:
            classe, efeito = "congelado", CONGELADOS[rel]
        elif not usa_vigente and fixa:
            classe, efeito = "fixa_versao", f"le {','.join(fixa)} por caminho; o v8 nao o alcanca"
        elif usa_vigente and cobre:
            classe, efeito = "sensivel_a_cobertura", "le colunas de cobertura: o resultado mudaria; exige nova rodada declarada"
        elif usa_vigente:
            classe, efeito = "invariante", "so le colunas identicas no v8: migrar nao muda resultado, so a procedencia"
        else:
            classe, efeito = "referencia_indireta", "cita a base sem ler pontos pela versao vigente"
        linhas.append({"script": rel, "classe": classe, "le_pela_versao_vigente": usa_vigente,
                       "versoes_fixadas": fixa, "colunas_de_cobertura_citadas": cobre, "efeito_do_v8": efeito,
                       "alterado_no_worktree": None})
    import subprocess
    sujos = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout
    for l in linhas:
        l["alterado_no_worktree"] = any(l["script"] in s and s.startswith((" M", "M ")) for s in sujos.splitlines())
    resumo = pd.Series([l["classe"] for l in linhas]).value_counts().to_dict()
    out = {"tipo": "mt03_consumidores_do_ds06", "quando_utc": agora_utc(), "resumo": resumo,
           "regra_de_migracao": ("migra so quem e invariante ou novo; congelado e fixo ficam no v7; sensivel a "
                                 "cobertura so com rodada nova declarada; nenhum script de outra sessao e editado"),
           "migrados_neste_ciclo": ["scripts/metodo/mt03_*.py (consumidores novos, leem versao_atual(BASE_V8))"],
           "colunas_identicas_no_v8": fixas, "consumidores": linhas}
    BASE_V8.mkdir(parents=True, exist_ok=True)
    (BASE_V8 / "consumidores.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"resumo": resumo, "n": len(linhas)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=("construir", "consumidores"))
    a = ap.parse_args()
    r = {"construir": construir, "consumidores": consumidores}[a.acao]()
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
