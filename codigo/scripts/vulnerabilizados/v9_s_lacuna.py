"""v9 secao 13: a lacuna como desfecho mensuravel, por comunidade FCU e por classe social, nas cinco coortes.

Por agrupamento (comunidade FCU ou setor): pontos, fracao observavel (S2 valido), desconhecido, agua permanente,
positivos e negativos observados, particao (adaptacao/teste), atributo de solo ausente (imputado no contrato Q) e,
quando a aquisicao (F) existir, consultas ate o primeiro positivo e positivos obtidos com 400 consultas.
Comparadores fisicamente pareados: decil de HAND x construido, como na v5. Nada disso e rotulo fisico nem mede
atencao institucional; ausencia de chamado nao entra.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_comum import COORTES, V3A, V8, V9, agora, etapa, g9

D = V9 / "s_lacuna"


def main():
    c = pd.read_parquet(V8 / "contrato/contrato_coortes_brasileiras.parquet", columns=["ponto_id", "coorte", "classe_territorial", "CD_FCU", "CD_SETOR", "estado"])
    c = c[c.classe_territorial.isin(["fcu", "baixa_renda", "comparador_urbano"])].copy()
    c["grupo"] = np.where(c.classe_territorial.eq("fcu"), "FCU:" + c.CD_FCU.astype(str), "SET:" + c.CD_SETOR.astype(str))
    part = pd.concat([pd.read_parquet(V3A / f"{k}.parquet", columns=["grupo", "particao"]).drop_duplicates("grupo").assign(coorte=k) for k in ["rmpa", "manaus", "curitiba"]])
    c = c.merge(part, on=["coorte", "grupo"], how="left")
    c["particao"] = c.particao.fillna(pd.Series(np.where(c.coorte.isin(["recife", "rmr"]), "so_avaliacao", "fora_v3"), index=c.index))
    cob = pd.concat([pd.read_parquet(V8 / f"ponte/cobertura_solo_30m/{k}.parquet", columns=["ponto_id", "areia_v2"]) for k in COORTES])
    c = c.merge(cob.assign(com_atributos=True), on="ponto_id", how="left")
    c["com_atributos"] = c.com_atributos.fillna(False).astype(bool)
    e = c.estado.fillna("sem_estado")
    c = c.assign(observavel=e.isin(["positivo", "negativo", "permanente"]), desconhecido=e.isin(["desconhecido", "pre_desconhecido"]),
                 permanente=e.eq("permanente"), pos=e.eq("positivo"), neg=e.eq("negativo"), solo_mascarado=c.areia_v2.isna().where(c.com_atributos))
    g = c.groupby(["coorte", "classe_territorial", "grupo", "particao"]).agg(pontos=("ponto_id", "size"), observavel=("observavel", "mean"),
                                                                            desconhecido=("desconhecido", "mean"), permanente=("permanente", "mean"),
                                                                            positivos=("pos", "sum"), negativos=("neg", "sum"), solo_mascarado=("solo_mascarado", "mean"), com_atributos=("com_atributos", "mean")).reset_index()
    g["duas_classes_5x5"] = (g.positivos >= 5) & (g.negativos >= 5)
    g["sem_positivo"] = g.positivos.eq(0)
    fp = V9 / "f_aquisicao/curvas_semente.parquet"
    if fp.exists():
        f = pd.read_parquet(fp)
        u = f[(f.estrategia == "uniforme") & (f.pool == "todas")]
        custo = u[u.positivos > 0].groupby(["coorte", "semente"]).consultados.min().groupby("coorte").agg(["mean", "max"]).reset_index()
        g9(D / "custo_primeiro_positivo_uniforme.csv", custo)
    resumo = g.groupby(["coorte", "classe_territorial"]).agg(grupos=("grupo", "size"), pontos=("pontos", "sum"), observavel=("observavel", "mean"),
                                                            desconhecido=("desconhecido", "mean"), com_atributos=("com_atributos", "mean"), grupos_sem_positivo=("sem_positivo", "mean"),
                                                            grupos_5x5=("duas_classes_5x5", "sum"), solo_mascarado=("solo_mascarado", "mean"),
                                                            positivos=("positivos", "sum")).reset_index()
    g9(D / "lacuna_por_grupo.csv", g)
    g9(D / "lacuna_resumo.csv", resumo)
    g9(D / "resultado.json", {"quando_utc": agora(), "grupos": len(g), "nota": "medidas de disponibilidade/observabilidade; nao sao rotulo nem atencao institucional"})
    etapa("s_lacuna", grupos=len(g))
    print(resumo.round(3).to_string())


if __name__ == "__main__":
    main()
