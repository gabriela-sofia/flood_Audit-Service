"""v8: analise da campanha (desenvolvimento/reanalise). Contrastes pareados com bootstrap por agrupamento
(comunidade FCU ou setor), 300 reamostragens, IC 95% (reduzido de 1.000 por custo: >1 h por coorte; declarado); sem correcao de multiplicidade (exploratorio declarado).

Sem rotulo local (por coorte, populacoes todas e FCU):
  N1 Q_ext-HAND  N2 Q_ext-(-declive)  N4 Q_ext-Q_perm  N5 Q_congelado-Q_ext  N6 Q_base-Q_terreno (bloco cobertura/solo)
  N7 Q_conexao-Q_base (bloco conexao)  N8 GBM_ext-Q_ext  N9 sem_areia-Q_ext (rota parcial)  N10 Q_extAS-Q_ext  N11 Q_ext-A1_v3
Poucos rotulos (predicoes medias das 6 sementes, E2; politicas urbano_uniforme e fcu_comunidades; tau 1 e 25):
  F1 P_ext-L_escext (prior isolado: mesma escala e tau)  F2 L_escext-L_estrito (escala isolada)  F3 P_ext-P_perm
  F4 P_ext-Q_ext(B=0) (o que os rotulos locais acrescentam)  F5 GBM_local-P_ext  F6 P_ext_escloc-P_ext
Inferencia so com >= 20 positivos e >= 10 agrupamentos com positivo; senao descritivo. Macro FCU por comunidade (>=5/5)
com sementes resumidas antes do bootstrap de comunidades.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum_tv import auc_mann_whitney, bootstrap_auc, intervalo
from v8_comum import V8, agora, etapa, g8

D = V8 / "campanha"
A = V8 / "analise"
N = [("N1", "Q_ext_conexao", "A0_hand"), ("N2", "Q_ext_conexao", "menos_declive"), ("N4", "Q_ext_conexao", "Q_perm_conexao"),
     ("N5", "Q_congelado", "Q_ext_conexao"), ("N6", "Q_ext_base", "Q_ext_terreno"), ("N7", "Q_ext_conexao", "Q_ext_base"),
     ("N8", "GBM_ext", "Q_ext_conexao"), ("N9", "Q_ext_sem_areia", "Q_ext_conexao"), ("N10", "Q_extAS_conexao", "Q_ext_conexao"),
     ("N11", "Q_ext_conexao", "A1_v3")]


def contrastes(q, pares, rotulo):
    cols = sorted({c for _, a, b in pares for c in (a, b)})
    gp = q[q.y.eq(1)].grupo.nunique()
    if q.y.sum() < 20 or gp < 10:
        return [{**rotulo, "contraste": n, "a": a, "b": b, "diferenca": (auc_mann_whitney(q.y, q[a]) or np.nan) - (auc_mann_whitney(q.y, q[b]) or np.nan),
                 "ic95": None, "estado": "descritivo_suporte_insuficiente", "positivos": int(q.y.sum()), "grupos_com_positivo": int(gp)} for n, a, b in pares]
    bs = bootstrap_auc(q, "grupo", cols, 300, 20261022)
    out = []
    for n, a, b in pares:
        iv = intervalo(bs[:, cols.index(a)] - bs[:, cols.index(b)], .95)
        dif = auc_mann_whitney(q.y, q[a]) - auc_mann_whitney(q.y, q[b])
        out.append({**rotulo, "contraste": n, "a": a, "b": b, "auc_a": auc_mann_whitney(q.y, q[a]), "auc_b": auc_mann_whitney(q.y, q[b]),
                    "diferenca": dif, "ic95": iv, "positivos": int(q.y.sum()), "grupos_com_positivo": int(gp),
                    "estado": None if iv is None else ("a_maior" if iv[0] > 0 else "a_menor" if iv[1] < 0 else "inconclusivo")})
    return out


def main():
    import v5_aprendizagem as v5
    A.mkdir(parents=True, exist_ok=True)
    linhas_n, linhas_f, macro = [], [], []
    for c in ["rmpa", "manaus", "curitiba", "recife", "rmr"]:
        p = pd.read_parquet(D / f"predicoes_{c}.parquet")
        pares = [t for t in N if t[1] in p and t[2] in p]
        for pop in ["todas", "fcu"]:
            q = p if pop == "todas" else p[p.classe_territorial.eq(pop)]
            linhas_n += contrastes(q, pares, {"coorte": c, "populacao": pop})
        f = p[p.classe_territorial.eq("fcu")]
        for col in ["A0_hand", "menos_declive", "twi", "Q_congelado", "Q_ext_terreno", "Q_ext_base", "Q_ext_conexao", "GBM_ext", "A1_v3"]:
            if col in f:
                for g_, a_ in v5.auc_por_grupo(f.y, f[col], f.grupo, 5).items():
                    macro.append({"coorte": c, "modelo": col, "grupo": g_, "auc": float(a_)})
        if c in ["rmpa", "manaus", "curitiba"]:
            for pol in ["urbano_uniforme", "fcu_comunidades"]:
                for B in [25, 100, 400]:
                    for tau in ["t25"]:
                        q = p.copy()
                        ren = {f"{k}|{tau}|{pol}|B{B}": k for k in ["P_ext", "L_escext", "L_estrito", "P_perm", "P_ext_escloc"]}
                        ren[f"GBM_local|{pol}|B{B}"] = "GBM_local"
                        q = q.rename(columns=ren)
                        pares = [("F1", "P_ext", "L_escext"), ("F2", "L_escext", "L_estrito"), ("F3", "P_ext", "P_perm"),
                                 ("F4", "P_ext", "Q_ext_conexao"), ("F6", "P_ext_escloc", "P_ext")]
                        if "GBM_local" in q:
                            pares.append(("F5", "GBM_local", "P_ext"))
                        for pop in ["todas", "fcu"]:
                            qq = q if pop == "todas" else q[q.classe_territorial.eq(pop)]
                            linhas_f += contrastes(qq, pares, {"coorte": c, "populacao": pop, "politica": pol, "B": B, "tau": tau})
        print(c, flush=True)
    tn, tf, tm = pd.DataFrame(linhas_n), pd.DataFrame(linhas_f), pd.DataFrame(macro)
    m = pd.read_parquet(D / "metricas_semente.parquet")
    curvas = m.groupby(["coorte", "populacao", "politica", "B", "braco"]).auc.agg(["mean", "std", "count"]).reset_index()
    cs = pd.read_parquet(D / "auc_comunidade_semente.parquet")
    macro_b = cs.groupby(["coorte", "politica", "B", "braco", "grupo"]).auc.mean().groupby(["coorte", "politica", "B", "braco"]).agg(["mean", "count"]).reset_index()
    aj = pd.read_parquet(D / "ajustes.parquet")
    contagem = aj.groupby(["coorte", "politica", "B", "estado_ajuste"]).size().unstack(fill_value=0).reset_index()
    rotulos = aj[aj.braco.eq("P_ext|t25")].groupby(["coorte", "politica", "B"]).agg(consultados=("consultados", "mean"), validos=("validos", "mean"),
                                                                                      positivos_medios=("positivos", "mean"), duas_classes=("duas_classes", "sum")).reset_index()
    for nome, t in [("contrastes_sem_rotulo.csv", tn), ("contrastes_poucos_rotulos.csv", tf), ("macro_fcu_sem_rotulo.csv", tm),
                    ("curvas_aprendizagem.csv", curvas), ("macro_fcu_orcamentos.csv", macro_b), ("contagem_ajustes.csv", contagem),
                    ("custo_rotulos.csv", rotulos)]:
        g8(A / nome, t)
    g8(A / "analise.json", {"quando_utc": agora(), "natureza": "desenvolvimento/reanalise; exploratorio sem multiplicidade",
                            "contrastes_sem_rotulo": len(tn), "contrastes_poucos_rotulos": len(tf)})
    etapa("v8_04b_analise", contrastes=len(tn) + len(tf))


if __name__ == "__main__":
    main()
