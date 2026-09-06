"""
AUD-CHUVA-01 -- A coluna de chuva carrega duas fontes, e uma delas prediz o rotulo.

DE ONDE VEIO ESTA AUDITORIA:

o `ext_resolucao_e_mecanismo_decisao_v1.md`, secao 5, deixou uma pendencia
declarada: "Recife usa CHIRPS e Curitiba usa Open-Meteo ERA5-Land, apesar de as
colunas terem sufixo `_chirps`. Se a chuva e o preditor principal da trilha
pluvial, a fonte de chuva deixa de ser detalhe de proveniencia e vira a
variavel central -- e duas fontes diferentes nao podem ocupar a mesma coluna."

Ao reduzir as fontes ao esquema-alvo (ds04), a pendencia se revelou maior do
que estava escrita: a mistura nao e entre Recife e Curitiba. E DENTRO DE
RECIFE. O campo `rain_data_source` do proprio v12 registra 181 pontos em
CHIRPS v2.0 e 97 em Open-Meteo ERA5-Land, na mesma coluna
`rain_max_24h_chirps`, no mesmo modelo.

POR QUE ISSO NAO E UMA IMPRECISAO DE METADADO:

na trilha pluvial a chuva antecedente e o preditor principal -- coeficiente
+0,9896 com p < 1e-4, contra HAND em -0,0001 com p = 0,978. Se a fonte da
chuva estiver associada ao rotulo, parte do que o coeficiente mede nao e
chuva: e a fonte. Este script nao supoe que esteja; mede.

O QUE ELE MEDE, e por que cada medida:

  1. confundimento   tabela fonte x rotulo. Se as duas fontes tivessem a mesma
                     proporcao de positivos, nao haveria problema a resolver.
  2. distribuicao    as duas fontes medem a mesma grandeza fisica no mesmo
                     lugar? Medianas muito diferentes ja bastam para proibir
                     o pool, independentemente do rotulo.
  3. discriminacao   AUC de tres coisas, comparadas lado a lado:
                       (a) a chuva, como o modelo a usa
                       (b) so o indicador de qual fonte produziu o valor
                       (c) a chuva DENTRO de cada fonte, onde o indicador e
                           constante e nao pode contribuir
                     Se (b) for alto e (c) cair em relacao a (a), a parte da
                     discriminacao que vinha da fonte fica visivel como numero.

  AUC por posto (Mann-Whitney), sem dependencia externa, com IC por bootstrap
  de 2.000 reamostragens e semente fixa.

O QUE ESTE SCRIPT NAO FAZ, deliberadamente:

nao corrige, nao reamostra, nao converte uma fonte na outra e nao remove
pontos. Reamostrar CHIRPS para os 97 pontos de ERA5-Land e aquisicao de dado,
nao auditoria, e a auditoria precisa existir antes para dizer se vale a pena.
Nao decide tambem se o v12 esta invalidado: o v12 e um resultado interno a uma
regiao, e o que este script mede e o quanto da sua discriminacao e atribuivel
a uma variavel que nao e chuva.

Uso:
    python scripts/treino/aud_chuva01_fontes_incompativeis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from chuva.contrato_chuva import METHOD_VERSION, parse_event_dates  # noqa: E402
RUNS = REPO / "local_runs"
RED = RUNS / "ds-04-reducao"
OUT = RUNS / "aud-chuva-01"

VARIAVEIS_CHUVA = ("rain_max_24h", "rain_decay_index")
N_BOOT = 2000
SEMENTE = 20260813


def auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """AUC por posto. Empates recebem posto medio, como manda o Mann-Whitney."""
    ok = np.isfinite(s) & np.isin(y, (0, 1))
    y, s = y[ok], s[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ic(y: np.ndarray, s: np.ndarray) -> dict:
    a = auc(y, s)
    if a is None:
        return {"auc": None, "n": 0, "motivo": "uma das classes esta vazia"}
    ok = np.isfinite(s) & np.isin(y, (0, 1))
    y, s = y[ok], s[ok]
    rng = np.random.default_rng(SEMENTE)
    amostras = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        v = auc(y[i], s[i])
        if v is not None:
            amostras.append(v)
    lo, hi = np.percentile(amostras, [2.5, 97.5]) if amostras else (None, None)
    return {"auc": round(a, 4),
            "ic95": [round(float(lo), 4), round(float(hi), 4)] if amostras else None,
            "n": int(len(y)), "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum())}


def auditar(nome: str, d: pd.DataFrame) -> dict:
    d = d[d["classe"].isin([0, 1])].copy()
    if "fonte_chuva" not in d.columns:
        raise ValueError(f"{nome}: rain provenance column is absent")
    valor = d[list(VARIAVEIS_CHUVA)].notna().any(axis=1)
    sem_procedencia = valor & (d["fonte_chuva"].isna()
                               | (d["fonte_chuva"].astype(str).str.strip() == "")
                               | (d["fonte_chuva"] == "ausente"))
    fontes = d["fonte_chuva"].value_counts().to_dict()
    rel: dict = {"n": int(len(d)), "fontes_de_chuva": fontes,
                 "rain_values_without_provenance": int(sem_procedencia.sum()),
                 "method_version": METHOD_VERSION}

    reais = [f for f in fontes if f != "ausente"]
    if not reais:
        rel["veredito"] = "SEM_CHUVA"
        return rel
    if len(reais) == 1:
        rel["veredito"] = "FONTE_UNICA"

    y = d["classe"].to_numpy()

    # 1. confundimento
    tab = pd.crosstab(d["fonte_chuva"], d["classe"])
    rel["fonte_x_rotulo"] = json.loads(tab.to_json(orient="index"))
    rel["prevalencia_por_fonte"] = {
        f: round(float((s["classe"] == 1).mean()), 4)
        for f, s in d.groupby("fonte_chuva")}

    # 2. distribuicao
    rel["distribuicao"] = {}
    for v in VARIAVEIS_CHUVA:
        rel["distribuicao"][v] = {
            f: {"n": int(s[v].notna().sum()),
                "mediana": round(float(s[v].median()), 3) if s[v].notna().any() else None,
                "media": round(float(s[v].mean()), 3) if s[v].notna().any() else None,
                "p90": round(float(s[v].quantile(0.9)), 3) if s[v].notna().any() else None}
            for f, s in d.groupby("fonte_chuva")}

    # 3. discriminacao
    rel["discriminacao"] = {}
    for v in VARIAVEIS_CHUVA:
        bloco = {"global": auc_ic(y, d[v].to_numpy(dtype="float64"))}
        for f in reais:
            s = d[d["fonte_chuva"] == f]
            bloco[f"dentro__{f}"] = auc_ic(
                s["classe"].to_numpy(), s[v].to_numpy(dtype="float64"))
        rel["discriminacao"][v] = bloco

    if len(reais) > 1:
        # o indicador de fonte como se fosse preditor. Nao e variavel de nenhum
        # modelo -- e exatamente por isso que o numero e informativo.
        ind = (d["fonte_chuva"] == reais[0]).to_numpy(dtype="float64")
        rel["discriminacao"]["INDICADOR_DE_FONTE"] = auc_ic(y, ind)
        rel["veredito"] = "MISTURA_DE_FONTES"
    if sem_procedencia.any():
        rel["veredito"] = "VALOR_SEM_PROCEDENCIA"

    # 4. as duas fontes cobrem o mesmo periodo?
    # coercao explicita: coluna pode chegar como object com string e NaN
    # misturados (ponto sem event_date na origem), e comparar os dois tipos
    # direto quebra o min()/max(). NaT sai do calculo, nao vira 0.
    dt_evento = parse_event_dates(d["data_evento"])
    if dt_evento.notna().any():
        rel["periodo_por_fonte"] = {
            f: {"min": str(dt_evento[s.index].min()), "max": str(dt_evento[s.index].max()),
                "datas_distintas": int(dt_evento[s.index].nunique())}
            for f, s in d.groupby("fonte_chuva") if dt_evento[s.index].notna().any()}
    return rel


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("===AUD_CHUVA01===")
    if not RED.exists():
        print(f"ABORTADO: {RED} ausente. Rode ds04.")
        return 1

    resultado: dict = {"semente": SEMENTE, "n_bootstrap": N_BOOT,
                       "method_version": METHOD_VERSION, "fontes": {}}
    for p in sorted(RED.glob("*.csv")):
        if p.stem.startswith("recife__variante"):
            continue
        d = pd.read_csv(p, low_memory=False)
        r = auditar(p.stem, d)
        resultado["fontes"][p.stem] = r

        print(f"\n[{p.stem}] n={r['n']:,} veredito={r.get('veredito')}")
        print(f"   fontes de chuva: {r['fontes_de_chuva']}")
        if r.get("veredito") in ("SEM_CHUVA",):
            continue
        if "prevalencia_por_fonte" in r:
            print(f"   prevalencia de positivo por fonte: {r['prevalencia_por_fonte']}")
        for v, dist in r.get("distribuicao", {}).items():
            corpo = "  ".join(f"{f.split('_')[0]}: med={x['mediana']} p90={x['p90']}"
                              for f, x in dist.items() if f != "ausente")
            if corpo:
                print(f"   {v:18s} {corpo}")
        for v, bloco in r.get("discriminacao", {}).items():
            if v == "INDICADOR_DE_FONTE":
                continue
            partes = [f"{k}={x['auc']}" for k, x in bloco.items()
                      if isinstance(x, dict) and x.get("auc") is not None]
            print(f"   AUC {v:18s} {'  '.join(partes)}")
        ind = r.get("discriminacao", {}).get("INDICADOR_DE_FONTE")
        if isinstance(ind, dict) and ind.get("auc") is not None:
            print(f"   AUC INDICADOR_DE_FONTE = {ind['auc']} IC95 {ind['ic95']}  "
                  f"<- isto nao e chuva")

    misturadas = [k for k, v in resultado["fontes"].items()
                  if v.get("veredito") == "MISTURA_DE_FONTES"]
    sem_procedencia = [k for k, v in resultado["fontes"].items()
                       if v.get("rain_values_without_provenance", 0)]
    resultado["regra"] = (
        "pool de chuva so e licito dentro de uma unica fonte_chuva. Fonte "
        "misturada na mesma coluna invalida a interpretacao do coeficiente, "
        "nao necessariamente o desempenho")
    resultado["fontes_com_mistura"] = misturadas
    resultado["fontes_com_valor_sem_procedencia"] = sem_procedencia

    print(f"\n--- VEREDITO ---")
    if misturadas:
        print(f"MISTURA_DE_FONTES em: {', '.join(misturadas)}")
        print("A coluna de chuva dessas fontes nao pode ser lida como uma "
              "variavel so ate a fonte ser unificada.")
    else:
        print("nenhuma fonte com mistura de produtos de precipitacao")

    (OUT / "auditoria_fonte_de_chuva.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    linhas = []
    for fonte, r in resultado["fontes"].items():
        for v, bloco in r.get("discriminacao", {}).items():
            if v == "INDICADOR_DE_FONTE":
                linhas.append({"fonte": fonte, "variavel": "INDICADOR_DE_FONTE",
                               "estrato": "-", **bloco})
                continue
            for estrato, x in bloco.items():
                linhas.append({"fonte": fonte, "variavel": v,
                               "estrato": estrato, **x})
    if linhas:
        pd.DataFrame(linhas).to_csv(OUT / "auc_por_estrato_de_fonte.csv", index=False)

    print(f"GRAVADO={OUT}")
    print("===END===")
    return 2 if misturadas or sem_procedencia else 0


if __name__ == "__main__":
    raise SystemExit(main())
