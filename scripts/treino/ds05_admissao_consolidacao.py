"""
DS-05 -- Admissao, duplicidade, contagem e consolidacao da tabela unica.

O QUE ESTE SCRIPT ENTREGA, e por que cada peca existe:

  1. ADMISSAO       aplica os tres criterios declarados no ds03 e grava, para
                    cada linha rejeitada, o motivo NOMEADO. Uma rejeicao sem
                    motivo e um numero que ninguem consegue defender depois.
  2. DUPLICIDADE    nenhum ponto pode existir em duas fontes. Verifica por
                    identificador e por coincidencia geografica, porque duas
                    fontes independentes podem amostrar o mesmo lugar com ids
                    diferentes -- que e justamente o caso perigoso.
  3. CONTAGEM       quantos entram e quantos saem por fonte, com o motivo. E o
                    entregavel que a Secao III promete para a etapa E2: se
                    este relatorio existe, E2 tem prova.
  4. CONSOLIDACAO   um arquivo unico, versionado, com sha256. A "base de
                    conhecimento geoespacial unica" deixa de ser frase e passa
                    a ter caminho e hash.

POR QUE A ADMISSAO NAO E UM BOOLEANO SO:

um unico `admitido` obrigaria a escolher entre duas coisas incompativeis:
ou o criterio de cadeia derruba Recife (que e pluvial e entra a 10 m por
decisao declarada), ou ele deixa passar o CEMS em cadeia global, que e outro
instrumento. Por isso sao dois portoes, com nomes diferentes e motivos proprios:

    admitido               a linha e utilizavel: tem AOI, tem grupo, tem
                           cadeia de terreno declarada e nao-global, e tem
                           rotulo de treino
    elegivel_pool_fluvial  a linha pode entrar no MESMO ajuste que as outras
                           regioes: cadeia wbt30, mecanismo fluvial, as quatro
                           de terreno presentes

Recife e admitido e nao e elegivel ao pool. Isso nao e falha: e o resultado
correto da separacao por mecanismo. Um numero que dissesse o contrario estaria
escondendo a decisao.

O QUE ESTE SCRIPT NAO FAZ: nao treina, nao imputa, nao ajusta criterio para
caber mais ponto, e nao remove linha do arquivo consolidado -- tudo fica, com
a coluna que diz se entra e por que nao.

Uso:
    python scripts/suscetibilidade/ds05_admissao_consolidacao.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds03_esquema_alvo import (  # noqa: E402
    CLASSE, CLASSES_TREINO, COLUNAS, DOMINIOS, OBRIGATORIAS,
    VARIAVEIS_TERRENO, VERSAO,
)

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
RED = RUNS / "ds-04-reducao"
OUT = RUNS / "ds-05-tabela-unica"

# fontes que compoem a tabela unica. As variantes do ds04 ficam de fora por
# construcao: sao a MESMA observacao noutra cadeia, e entrariam como duplicata.
FONTES = ("uk", "cems", "sen1floods11", "ufo", "recife", "curitiba")

# ~11 m no equador. Abaixo de meio pixel de 30 m: dois pontos que caem aqui
# dentro sao a mesma celula, venham de onde vierem.
GRADE_DUP_GRAUS = 1e-4


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def carregar() -> pd.DataFrame:
    partes = []
    for nome in FONTES:
        p = RED / f"{nome}.csv"
        if not p.exists():
            print(f"AVISO: {p.name} ausente -- fonte fica fora da consolidacao")
            continue
        d = pd.read_csv(p, low_memory=False)
        faltando = [c for c in COLUNAS if c not in d.columns]
        if faltando:
            raise SystemExit(f"{p.name} fora do contrato: faltam {faltando}")
        partes.append(d[list(COLUNAS)])
    if not partes:
        raise SystemExit("nenhuma fonte reduzida encontrada. Rode ds04.")
    return pd.concat(partes, ignore_index=True)


def conferir_contrato(d: pd.DataFrame) -> list[dict]:
    """Violacao de contrato e erro de reducao, nao dado. Reporta, nao conserta."""
    achados = []
    for c in OBRIGATORIAS:
        n = int(d[c].isna().sum())
        if n:
            achados.append({"coluna": c, "tipo": "obrigatoria_nula", "n": n})
    for c, dom in DOMINIOS.items():
        fora = d.loc[~d[c].isin(dom) & d[c].notna(), c]
        if len(fora):
            achados.append({"coluna": c, "tipo": "fora_do_dominio",
                            "n": int(len(fora)),
                            "valores": sorted(fora.unique().tolist())[:5]})
    fora_cls = d.loc[~d["classe"].isin(CLASSE), "classe"]
    if len(fora_cls):
        achados.append({"coluna": "classe", "tipo": "fora_do_dominio",
                        "n": int(len(fora_cls)),
                        "valores": sorted(fora_cls.unique().tolist())[:5]})
    return achados


def admitir(d: pd.DataFrame) -> pd.DataFrame:
    """Aplica os tres criterios do ds03 e nomeia cada rejeicao."""
    motivos = pd.Series([""] * len(d), index=d.index, dtype=object)

    def marcar(mascara: pd.Series, motivo: str) -> None:
        motivos.loc[mascara] = (motivos.loc[mascara] + ";" + motivo).str.lstrip(";")

    # A -- dentro da AOI declarada
    marcar(d["aoi"].isna() | (d["aoi"].astype(str).str.strip() == ""),
           "A_sem_aoi_declarada")
    marcar(~np.isfinite(pd.to_numeric(d["lat"], errors="coerce"))
           | ~np.isfinite(pd.to_numeric(d["lon"], errors="coerce")),
           "A_coordenada_nao_finita")

    # B -- mesma cadeia. 'global' e outro instrumento, nao outra resolucao.
    marcar(d["cadeia_terreno"] == "global", "B_cadeia_global_nao_comparavel")
    marcar(d["cadeia_terreno"] == "ausente", "B_sem_cadeia_de_terreno")
    marcar(d[list(VARIAVEIS_TERRENO)].isna().any(axis=1),
           "B_terreno_incompleto")

    # C -- grupo de validacao identificavel
    marcar(d["grupo_cv"].isna() | (d["grupo_cv"].astype(str).str.strip() == ""),
           "C_sem_grupo_de_validacao")

    # rotulo utilizavel. Agua permanente e classe de auditoria, declarada no
    # ds01: sai do treino sem ser erro.
    marcar(~d["classe"].isin(CLASSES_TREINO), "D_classe_fora_de_treino")

    d = d.copy()
    d["motivo_rejeicao"] = motivos
    d["admitido"] = motivos == ""
    d["elegivel_pool_fluvial"] = (
        d["admitido"]
        & (d["cadeia_terreno"] == "wbt30")
        & (d["mecanismo"] == "FLUVIAL_ENXURRADA"))
    return d


def duplicidade(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Nenhum ponto pode existir em duas fontes.

    Duas checagens, porque sao dois riscos diferentes:
      id     colisao de identificador -- erro de construcao do ponto_id
      grade  mesma celula de 30 m em fontes diferentes -- duplicata real,
             que nenhum identificador pega
    """
    rel: dict = {}

    dup_id = d[d.duplicated("ponto_id", keep=False)]
    rel["ponto_id_repetido"] = int(dup_id["ponto_id"].nunique())
    rel["exemplos_ponto_id"] = dup_id["ponto_id"].unique()[:5].tolist()

    g = d.copy()
    g["celula"] = (
        (pd.to_numeric(g["lat"], errors="coerce") / GRADE_DUP_GRAUS).round()
        .astype("Int64").astype(str)
        + "_"
        + (pd.to_numeric(g["lon"], errors="coerce") / GRADE_DUP_GRAUS).round()
        .astype("Int64").astype(str))
    por_celula = g.groupby("celula")["fonte"].nunique()
    celulas_cruzadas = por_celula[por_celula > 1].index
    conflito = g[g["celula"].isin(celulas_cruzadas)].copy()

    rel["celulas_em_mais_de_uma_fonte"] = int(len(celulas_cruzadas))
    rel["pontos_envolvidos"] = int(len(conflito))
    if len(conflito):
        pares = (conflito.groupby("celula")["fonte"]
                 .apply(lambda s: " + ".join(sorted(set(s))))
                 .value_counts().to_dict())
        rel["pares_de_fonte"] = pares

    # duplicidade INTRA-fonte nao e conflito entre fontes; e pseudo-replicacao,
    # ja conhecida em Curitiba (209 linhas repetem unidade de observacao). Nao
    # rejeita: o agrupamento por grupo_cv ja impede vazamento. Fica contada.
    rel["intra_fonte_mesma_celula"] = {
        f: int(s.duplicated("celula", keep=False).sum())
        for f, s in g.groupby("fonte") if s.duplicated("celula").any()}
    rel["grade_graus"] = GRADE_DUP_GRAUS
    return conflito, rel


def contagem(d: pd.DataFrame) -> pd.DataFrame:
    """Quantos entram e quantos saem por fonte, com o motivo. Entregavel de E2."""
    linhas = []
    for fonte, s in d.groupby("fonte"):
        base = {"fonte": fonte, "entrada": len(s),
                "admitido": int(s["admitido"].sum()),
                "elegivel_pool_fluvial": int(s["elegivel_pool_fluvial"].sum()),
                "grupos_admitidos": int(s.loc[s["admitido"], "grupo_cv"].nunique())}
        for motivo, n in (s.loc[~s["admitido"], "motivo_rejeicao"]
                          .str.split(";").explode().value_counts().items()):
            base[f"saiu__{motivo}"] = int(n)
        linhas.append(base)
    t = pd.DataFrame(linhas).fillna(0)
    num = [c for c in t.columns if c != "fonte"]
    t[num] = t[num].astype(int)
    return t.sort_values("entrada", ascending=False)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"===DS05_TABELA_UNICA=== esquema={VERSAO}")

    d = carregar()
    print(f"ENTRADA n={len(d):,} de {d['fonte'].nunique()} fontes")

    viol = conferir_contrato(d)
    if viol:
        print("\n--- VIOLACOES DE CONTRATO (erro de reducao, nao dado) ---")
        for v in viol:
            print(f"   {v}")
    else:
        print("CONTRATO: nenhuma violacao")

    d = admitir(d)
    conflito, dup = duplicidade(d)

    print("\n--- DUPLICIDADE ENTRE FONTES ---")
    print(f"   ponto_id repetido: {dup['ponto_id_repetido']}")
    print(f"   celulas de ~11 m em mais de uma fonte: "
          f"{dup['celulas_em_mais_de_uma_fonte']} "
          f"({dup['pontos_envolvidos']} pontos)")
    if dup.get("pares_de_fonte"):
        for par, n in dup["pares_de_fonte"].items():
            print(f"      {par}: {n} celulas")
    if dup["intra_fonte_mesma_celula"]:
        print(f"   intra-fonte (pseudo-replicacao conhecida, nao rejeita): "
              f"{dup['intra_fonte_mesma_celula']}")

    t = contagem(d)
    print("\n--- CONTAGEM POR FONTE: o que entra, o que sai e por que ---")
    print(t.to_string(index=False))

    adm = d[d["admitido"]]
    pool = d[d["elegivel_pool_fluvial"]]
    print(f"\nADMITIDOS n={len(adm):,} grupos={adm['grupo_cv'].nunique():,} "
          f"fontes={adm['fonte'].nunique()}")
    print(f"POOL_FLUVIAL n={len(pool):,} grupos={pool['grupo_cv'].nunique():,} "
          f"fontes={pool['fonte'].nunique()}")
    if len(pool):
        pos = int((pool["classe"] == 1).sum())
        print(f"   positivos={pos:,} negativos={int((pool['classe'] == 0).sum()):,} "
              f"EPV_4feat={pool['grupo_cv'].nunique() / 4:.1f}")
        print(f"   nivel_negativo: "
              f"{pool.loc[pool['classe'] == 0, 'nivel_negativo'].value_counts().to_dict()}")
        print(f"   classe_relevo: {pool['classe_relevo'].value_counts().to_dict()}")

    fora_do_pool = adm[~adm["elegivel_pool_fluvial"]]
    if len(fora_do_pool):
        print(f"\nADMITIDOS FORA DO POOL n={len(fora_do_pool):,} -- "
              "por mecanismo ou cadeia, nao por falha:")
        print(fora_do_pool.groupby(["fonte", "mecanismo", "cadeia_terreno"])
              .size().to_string())

    # ---- gravacao ----
    consolidado = OUT / f"tabela_unica_{VERSAO}.csv"
    d.to_csv(consolidado, index=False)
    adm.to_csv(OUT / f"tabela_unica_admitidos_{VERSAO}.csv", index=False)
    pool.to_csv(OUT / f"tabela_unica_pool_fluvial_{VERSAO}.csv", index=False)
    t.to_csv(OUT / "relatorio_contagem_por_fonte.csv", index=False)
    d.loc[~d["admitido"], ["ponto_id", "fonte", "aoi", "motivo_rejeicao"]].to_csv(
        OUT / "rejeitados_com_motivo.csv", index=False)
    if len(conflito):
        conflito.to_csv(OUT / "conflitos_de_duplicidade.csv", index=False)

    manifesto = {
        "esquema": VERSAO,
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fontes": list(FONTES),
        "entrada": int(len(d)),
        "admitidos": int(len(adm)),
        "pool_fluvial": int(len(pool)),
        "grupos_admitidos": int(adm["grupo_cv"].nunique()),
        "grupos_pool": int(pool["grupo_cv"].nunique()),
        "violacoes_de_contrato": viol,
        "duplicidade": dup,
        "contagem_por_fonte": json.loads(t.to_json(orient="records")),
        "arquivos": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                     for p in sorted(OUT.glob("*.csv"))},
        "criterio_pool": ("cadeia_terreno=wbt30 E mecanismo=FLUVIAL_ENXURRADA E "
                          "as quatro variaveis de terreno presentes"),
        "nao_e": ("nao e rotulo de referencia, nao e ground truth, e nao "
                  "autoriza claim preditivo: e a tabela de pontos harmonizada"),
    }
    (OUT / f"manifesto_{VERSAO}.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    print(f"\nCONSOLIDADO={consolidado}")
    print(f"SHA256={manifesto['arquivos'][consolidado.name]['sha256'][:16]}...")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
