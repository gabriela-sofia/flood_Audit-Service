"""
MOD-SERRA-03 -- Coeficientes por classe de relevo (E3/M2) sobre a TABELA UNICA.

O QUE O E3 PEDE: "tabelas de coeficientes e de desempenho por classe de relevo,
com validacao agrupada e aplicacao cruzada entre fontes de negativo".

O QUE EXISTIA: o MOD-SERRA-01 (11/08/2026) fez exatamente isso -- serra contra
planicie, duas variaveis, transferencia nos dois sentidos -- mas na `ds-01`,
base anterior a harmonizacao, e so com as AOIs do Copernicus EMS. A base
congelada do E2/M1 e a tabela unica. Este script refaz o desenho la.

-------------------------------------------------------------------------
O PROBLEMA QUE APARECEU AO REFAZER, e que muda o que se pode afirmar
-------------------------------------------------------------------------

O MOD-SERRA-01 declarou EPV 11,0 dividindo 22 GRUPOS por 2 variaveis. Contar
assim ignora de qual classe sao os grupos. No pool harmonizado o estrato
INGREME tem 24 grupos, mas so **19 tem positivo**. Pela regra como Peduzzi a
enunciou -- eventos da CLASSE MINORITARIA por variavel -- o orcamento do
estrato ingreme e:

    19 grupos positivos / 10 = 1,9  ->  UMA variavel, nao duas.

E a mesma correcao que a trava do MOD-PROSP-02 aplicou ao holdout temporal, e
ela nao pode valer la e nao valer aqui.

Como isso muda uma afirmacao ja publicada (os coeficientes de serra e planicie
aparecem no manuscrito), o script roda as DUAS leituras e reporta as duas, em
vez de escolher uma no escuro:

  LEITURA_PRECEDENTE   grupos totais / n_variaveis >= 10.
                       Ingreme: 24/2 = 12 -> 2 variaveis. E o que o
                       MOD-SERRA-01 usou; existe aqui para que a comparacao
                       com o resultado publicado seja possivel.
  LEITURA_ESTRITA      grupos da classe minoritaria / n_variaveis >= 10.
                       Ingreme: 19/10 -> 1 variavel. E a regra que o projeto
                       aplica desde o MOD-PROSP-02.

A recomendacao esta escrita em `ext_modelo_de_encosta_v2.md`; este script nao
decide, mede as duas.

-------------------------------------------------------------------------
DESENHO, declarado antes de rodar
-------------------------------------------------------------------------

BASE: o pool fluvial da tabela unica na versao declarada pelo contrato do
ds03 (`VERSAO`), classes 0/1, negativo admitido, quatro variaveis de terreno
e duas de chuva presentes.

ESTRATOS: `classe_relevo` INGREME e PLANO_OU_ONDULADO. O estrato
NAO_CLASSIFICADO tem 6 pontos e uma classe so -- fica fora, declarado.

VARIAVEIS: a escolha NAO e por desempenho. `hand_m` e `twi_dinf` sao os dois
termos causais do balanco (quanto precisa subir para ser alcancado, e para
onde a agua converge) e sao os dois com sinal obrigatorio declarado. Quando o
orcamento permite uma so, fica `hand_m`, que e a variavel causal central do
projeto. `elevation_m` e `slope_deg` ficam fora do conjunto comparavel porque
ja foram medidas invertendo de sinal entre fontes -- carregam regiao, nao
processo (`ext_o_que_nao_e_enchente_v2.md` secao 5).

CONJUNTOS RODADOS por estrato: todos os que couberem no orcamento daquele
estrato sob cada leitura. A planicie comporta 42 variaveis pela regra estrita,
entao roda tambem TERRENO (4) e COMPLETO (6) -- mas a comparacao entre
estratos so pode ser feita no conjunto que os DOIS comportam.

VALIDACAO: GroupKFold por `grupo_cv`, 5 folds. IC95 por bootstrap percentil
reamostrando GRUPOS, N=1000 (regra U2). Transferencia nos dois sentidos:
ajusta num estrato, mede no outro, sempre no conjunto comum.

CRITERIOS ja fixados em `ext_criterios_de_acerto_v1.md`: AUC agrupada entre
0,70 e 0,88; acima de 0,95 e suspeita de vazamento; `hand_m` negativo,
`twi_dinf` positivo; IC95 sem cruzar zero nas causais.

NAO FAZ: nao escolhe leitura, nao escolhe conjunto pelo resultado, nao declara
modelo final, nao promove rotulo.

Uso:
    python scripts/suscetibilidade/mod_serra03_relevo_ds05.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susc_firth_adaptador  # noqa: F401,E402
from ds03_esquema_alvo import VERSAO  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
DATASET = RUNS / "ds-05-tabela-unica" / f"tabela_unica_pool_fluvial_{VERSAO}.csv"
OUT = RUNS / "mod-serra-03"

TOPO = ["elevation_m", "slope_deg", "hand_m", "twi_dinf"]
CHUVA = ["rain_max_24h", "rain_decay_index"]
CAUSAIS = ["hand_m", "twi_dinf"]
CONJUNTOS = {
    "CAUSAL_1": ["hand_m"],
    "CAUSAL_2": CAUSAIS,
    "TERRENO": TOPO,
    "COMPLETO": TOPO + CHUVA,
}
SINAL_EXIGIDO = {"hand_m": -1, "twi_dinf": +1}
NEGATIVO_ADMITIDO = ("observado", "exclusao_qualificada", "nao_aplicavel")

EPV_MINIMO = 10
N_BOOT = 1000
SEMENTE = 20260820
AUC_MIN, AUC_MAX = 0.70, 0.88
AUC_SUSPEITA = 0.95


def orcamento(s: pd.DataFrame, leitura: str) -> int:
    """Quantas variaveis o estrato comporta sob cada leitura da regra de EPV."""
    por_classe = s.groupby("classe").grupo_cv.nunique()
    if len(por_classe) < 2:
        return 0
    base = (s.grupo_cv.nunique() if leitura == "LEITURA_PRECEDENTE"
            else int(por_classe.min()))
    return base // EPV_MINIMO


def ajustar(X: np.ndarray, y: np.ndarray):
    from firthlogist import FirthLogisticRegression

    m = FirthLogisticRegression(max_iter=500, skip_pvals=True, skip_ci=True)
    m.fit(X, y)
    return m


def padronizar(tr: np.ndarray):
    mu, sd = tr.mean(0), tr.std(0)
    return mu, np.where(sd == 0, 1, sd)


def cv_agrupada(s: pd.DataFrame, feats: list[str]) -> dict:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    X = s[feats].to_numpy(dtype=float)
    y = s.classe.to_numpy().astype(int)
    g = s.grupo_cv.to_numpy()
    n_folds = min(5, len(np.unique(g)))
    if n_folds < 2:
        return {"auc_cv": None, "motivo": "grupos insuficientes para CV"}
    p = np.full(len(y), np.nan)
    p_tr = []
    for i_tr, i_te in GroupKFold(n_splits=n_folds).split(X, y, g):
        if len(np.unique(y[i_tr])) < 2 or len(np.unique(y[i_te])) < 2:
            continue
        mu, sd = padronizar(X[i_tr])
        m = ajustar((X[i_tr] - mu) / sd, y[i_tr])
        p[i_te] = m.predict_proba((X[i_te] - mu) / sd)[:, 1]
        p_tr.append(roc_auc_score(y[i_tr], m.predict_proba((X[i_tr] - mu) / sd)[:, 1]))
    ok = ~np.isnan(p)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return {"auc_cv": None, "motivo": "nenhum fold com as duas classes"}
    auc = float(roc_auc_score(y[ok], p[ok]))
    return {"auc_cv": round(auc, 4), "folds": n_folds,
            "auc_treino_medio": round(float(np.mean(p_tr)), 4) if p_tr else None,
            "lacuna": (round(float(np.mean(p_tr)) - auc, 4) if p_tr else None),
            "cobertura": round(float(ok.mean()), 3)}


def coef_com_ic(s: pd.DataFrame, feats: list[str],
                rng: np.random.Generator) -> dict:
    """Coeficientes padronizados e IC95 por bootstrap de GRUPOS (regra U2)."""
    X = s[feats].to_numpy(dtype=float)
    y = s.classe.to_numpy().astype(int)
    mu, sd = padronizar(X)
    m = ajustar((X - mu) / sd, y)
    ponto = dict(zip(feats, [float(c) for c in m.coef_.ravel()[:len(feats)]]))

    grupos = s.grupo_cv.to_numpy()
    unicos = np.unique(grupos)
    idx = {g: np.flatnonzero(grupos == g) for g in unicos}
    amostras, validos = {f: [] for f in feats}, 0
    for _ in range(N_BOOT):
        esc = rng.choice(unicos, size=len(unicos), replace=True)
        i = np.concatenate([idx[g] for g in esc])
        yb = y[i]
        if len(np.unique(yb)) < 2:
            continue
        Xb = X[i]
        mub, sdb = padronizar(Xb)
        try:
            mb = ajustar((Xb - mub) / sdb, yb)
        except Exception:
            continue
        validos += 1
        for f, c in zip(feats, mb.coef_.ravel()[:len(feats)]):
            amostras[f].append(float(c))
    ic = {}
    for f in feats:
        if len(amostras[f]) >= 100:
            lo, hi = np.percentile(amostras[f], [2.5, 97.5])
            ic[f] = [round(float(lo), 4), round(float(hi), 4)]
        else:
            ic[f] = None
    return {"coef": {f: round(v, 4) for f, v in ponto.items()},
            "ic95": ic, "boot_validos": validos}


def conferir(coef: dict, ic: dict, auc: float | None) -> list[str]:
    falhas = []
    for f, esperado in SINAL_EXIGIDO.items():
        if f not in coef:
            continue
        if np.sign(coef[f]) != esperado:
            falhas.append(f"sinal de {f} e {coef[f]:+.4f}, esperado "
                          f"{'negativo' if esperado < 0 else 'positivo'}")
        if ic.get(f) and ic[f][0] <= 0 <= ic[f][1]:
            falhas.append(f"IC95 de {f} cruza zero: {ic[f]}")
    if auc is not None:
        if auc > AUC_SUSPEITA:
            falhas.append(f"AUC {auc} acima de {AUC_SUSPEITA}: suspeita de vazamento")
        elif not (AUC_MIN <= auc <= AUC_MAX):
            falhas.append(f"AUC {auc} fora da faixa [{AUC_MIN}; {AUC_MAX}]")
    return falhas


def transferir(origem: pd.DataFrame, destino: pd.DataFrame,
               feats: list[str], rng: np.random.Generator) -> dict:
    """Ajusta num estrato, mede no outro. IC por bootstrap de grupos do destino."""
    from sklearn.metrics import roc_auc_score

    Xo = origem[feats].to_numpy(dtype=float)
    mu, sd = padronizar(Xo)
    m = ajustar((Xo - mu) / sd, origem.classe.to_numpy().astype(int))
    Xd = (destino[feats].to_numpy(dtype=float) - mu) / sd
    y = destino.classe.to_numpy().astype(int)
    p = m.predict_proba(Xd)[:, 1]
    if len(np.unique(y)) < 2:
        return {"auc": None, "motivo": "destino sem as duas classes"}
    auc = float(roc_auc_score(y, p))

    grupos = destino.grupo_cv.to_numpy()
    unicos = np.unique(grupos)
    idx = {g: np.flatnonzero(grupos == g) for g in unicos}
    am = []
    for _ in range(N_BOOT):
        esc = rng.choice(unicos, size=len(unicos), replace=True)
        i = np.concatenate([idx[g] for g in esc])
        if len(np.unique(y[i])) < 2:
            continue
        am.append(roc_auc_score(y[i], p[i]))
    ic = ([round(float(v), 4) for v in np.percentile(am, [2.5, 97.5])]
          if len(am) >= 100 else None)
    return {"auc": round(auc, 4), "ic95": ic, "n_destino": int(len(destino)),
            "grupos_destino": int(destino.grupo_cv.nunique()),
            "boot_validos": len(am)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===MOD-SERRA-03: COEFICIENTES POR CLASSE DE RELEVO (E3/M2)===")

    if not DATASET.exists():
        print(f"ABORTADO: {DATASET} ausente. Gere com: "
              "python scripts/suscetibilidade/ds05_admissao_consolidacao.py")
        return 1

    d = pd.read_csv(DATASET, low_memory=False)
    d = d[d.classe.isin([0, 1]) & d.nivel_negativo.isin(NEGATIVO_ADMITIDO)].copy()
    d = d.dropna(subset=CONJUNTOS["COMPLETO"] + ["grupo_cv", "classe_relevo"])

    estratos = {}
    for nome, s in d.groupby("classe_relevo"):
        por_classe = s.groupby("classe").grupo_cv.nunique()
        estratos[nome] = {
            "n": int(len(s)), "grupos": int(s.grupo_cv.nunique()),
            "grupos_positivos": int(por_classe.get(1, 0)),
            "grupos_negativos": int(por_classe.get(0, 0)),
            "fontes": s.fonte.value_counts().to_dict(),
            "orcamento": {leitura: orcamento(s, leitura) for leitura in
                          ("LEITURA_PRECEDENTE", "LEITURA_ESTRITA")},
            "conjuntos": {},
        }
        e = estratos[nome]
        print(f"\n=== {nome} ===")
        print(f"  n={e['n']:,} grupos={e['grupos']:,} "
              f"(pos={e['grupos_positivos']}, neg={e['grupos_negativos']})")
        print(f"  orcamento de variaveis: precedente={e['orcamento']['LEITURA_PRECEDENTE']}, "
              f"estrita={e['orcamento']['LEITURA_ESTRITA']}")
        print(f"  fontes: {e['fontes']}")

        if e["grupos_positivos"] == 0 or e["grupos_negativos"] == 0:
            e["motivo"] = "estrato sem as duas classes"
            print("  fora do ajuste: estrato sem as duas classes")
            continue

        for conj, feats in CONJUNTOS.items():
            cabe = {leitura: len(feats) <= e["orcamento"][leitura]
                    for leitura in e["orcamento"]}
            if not any(cabe.values()):
                e["conjuntos"][conj] = {"features": feats, "cabe": cabe,
                                        "rodado": False,
                                        "motivo": "acima do orcamento nas duas leituras"}
                print(f"  {conj} ({len(feats)} var): fora do orcamento nas duas leituras")
                continue
            rng = np.random.default_rng(SEMENTE)
            cv = cv_agrupada(s, feats)
            ci = coef_com_ic(s, feats, rng)
            falhas = conferir(ci["coef"], ci["ic95"], cv.get("auc_cv"))
            e["conjuntos"][conj] = {
                "features": feats, "cabe": cabe, "rodado": True,
                **cv, **ci, "falhas": falhas,
                "veredito": "COERENTE_COM_CRITERIOS" if not falhas else "FORA_DOS_CRITERIOS",
            }
            marca = " ".join(k.split("_")[1][:4] for k, v in cabe.items() if v)
            print(f"  {conj} ({len(feats)} var, cabe em: {marca}) "
                  f"AUC_cv={cv.get('auc_cv')} "
                  f"coef={ci['coef']}")
            for f, v in ci["ic95"].items():
                print(f"      {f}: {ci['coef'][f]:+.4f}  IC95 {v}")
            print(f"      veredito={e['conjuntos'][conj]['veredito']}"
                  + (f"  falhas={falhas}" if falhas else ""))

    # transferencia entre estratos, no conjunto que os dois comportam
    transferencias = {}
    ing = d[d.classe_relevo == "INGREME"]
    pla = d[d.classe_relevo == "PLANO_OU_ONDULADO"]
    if len(ing) and len(pla):
        for conj in ("CAUSAL_1", "CAUSAL_2"):
            feats = CONJUNTOS[conj]
            cabe_nos_dois = all(
                len(feats) <= estratos[e]["orcamento"]["LEITURA_PRECEDENTE"]
                for e in ("INGREME", "PLANO_OU_ONDULADO"))
            rng = np.random.default_rng(SEMENTE)
            transferencias[conj] = {
                "features": feats,
                "cabe_nos_dois_estratos": cabe_nos_dois,
                "planicie_para_serra": transferir(pla, ing, feats, rng),
                "serra_para_planicie": transferir(ing, pla, feats, rng),
            }
            t = transferencias[conj]
            print(f"\n--- transferencia ({conj}) ---")
            print(f"  planicie -> serra: AUC={t['planicie_para_serra']['auc']} "
                  f"IC {t['planicie_para_serra']['ic95']}")
            print(f"  serra -> planicie: AUC={t['serra_para_planicie']['auc']} "
                  f"IC {t['serra_para_planicie']['ic95']}")

    linhas = []
    for nome, e in estratos.items():
        for conj, c in e.get("conjuntos", {}).items():
            if not c.get("rodado"):
                continue
            for f in c["features"]:
                linhas.append({
                    "estrato": nome, "conjunto": conj, "n": e["n"],
                    "grupos": e["grupos"], "grupos_positivos": e["grupos_positivos"],
                    "variavel": f, "coef": c["coef"][f],
                    "ic95_lo": (c["ic95"][f] or [None, None])[0],
                    "ic95_hi": (c["ic95"][f] or [None, None])[1],
                    "auc_cv": c.get("auc_cv"), "veredito": c["veredito"],
                    "cabe_leitura_estrita": c["cabe"]["LEITURA_ESTRITA"],
                })
    pd.DataFrame(linhas).to_csv(OUT / "coeficientes_por_relevo.csv", index=False)

    (OUT / "resultado.json").write_text(json.dumps({
        "entrada": str(DATASET.relative_to(REPO)).replace("\\", "/"),
        "epv_minimo": EPV_MINIMO, "n_boot": N_BOOT, "semente": SEMENTE,
        "criterios": {"faixa_auc": [AUC_MIN, AUC_MAX],
                      "auc_suspeita": AUC_SUSPEITA,
                      "sinal_exigido": SINAL_EXIGIDO},
        "leituras_da_regra_de_epv": {
            "LEITURA_PRECEDENTE": "grupos totais / n_variaveis (MOD-SERRA-01)",
            "LEITURA_ESTRITA": "grupos da classe minoritaria / n_variaveis",
        },
        "estratos": estratos,
        "transferencia": transferencias,
        "segundos": round(time.time() - t0, 1),
        "nao_e": ("nao e validacao operacional nem autoriza uso preditivo: e o "
                  "ajuste por classe de relevo sobre a tabela unica"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    (OUT / "commands.txt").write_text(
        "python scripts/suscetibilidade/ds05_admissao_consolidacao.py\n"
        "python scripts/suscetibilidade/mod_serra03_relevo_ds05.py\n"
        "python -m pytest tests/test_mod_serra03_relevo.py -q\n",
        encoding="utf-8")

    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
