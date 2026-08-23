"""
AUD-CHUVA-02 -- A chuva varia na mesma escala em que o modelo compara?

O QUE A AUD-CHUVA-01 JA RESOLVEU, E POR QUE NAO BASTA:

a AUD-CHUVA-01 perguntou "duas fontes de precipitacao ocupam a mesma coluna?".
A resposta hoje e nao: depois do `chuva02` (Recife) e do `chuva04` (global), as
seis fontes da tabela unica usam Open-Meteo/ERA5-Land, mesma janela de 14 dias
e mesmo fator de decaimento 0,85/dia. Produto unico, formula unica, cobertura
de 99,99%.

Isso fecha a pergunta de PROCEDENCIA e nao toca a pergunta de ESCALA, que e a
que decide se a variavel serve para o que o projeto afirma.

O PROBLEMA, em uma frase: a chuva varia no tempo e em celulas de ~11 km; o
modelo compara pontos DENTRO do mesmo evento, a dezenas ou centenas de metros.
Se a chuva for constante dentro da unidade de comparacao, ela nao pode
discriminar nada ali -- e qualquer poder discriminante que ela mostre vem de
outra escala, tipicamente a data.

Por que isso importa mais do que parece: a pergunta de pesquisa e sobre QUAIS
LUGARES inundam. "Choveu mais no dia em que houve enchente" e uma afirmacao
sobre QUANDO, quase tautologica quando o negativo foi amostrado em outros dias.
Um coeficiente de chuva alto pode estar medindo o desenho da amostra em vez do
fenomeno -- exatamente a especie de confundimento que a AUD-CHUVA-01 achou na
proveniencia, um nivel acima.

O QUE ESTE SCRIPT MEDE, e por que cada medida:

  1. escala espacial    quantas celulas de 0,1 grau e quantos valores distintos
                        de chuva existem por fonte. Poucas celulas significa que
                        a chuva e praticamente uma constante regional.
  2. decomposicao       quanto da variancia da chuva esta ENTRE grupos e quanto
                        DENTRO deles. O grupo e a unidade de validacao do
                        projeto; variancia so entre grupos = nada a discriminar
                        dentro.
  3. AUC dentro do grupo  o teste direto: entre pontos do MESMO evento, a chuva
                        separa o que inundou do que nao inundou?
  4. contraste temporal  positivos e negativos compartilham datas? Se nao
                        compartilham, comparar chuva entre eles e comparar dias,
                        nao lugares. Mede-se tambem a AUC restrita as datas
                        compartilhadas, que e a versao sem esse atalho.

DOIS VEREDITOS, nao um. A primeira versao deste script tinha um veredito so, e
ele escondia o achado principal: Recife saia como "sem variacao intra-grupo" --
verdade, mas trivial, porque la o grupo E o ponto -- e a disjuncao de datas, que
e o problema de verdade, nem aparecia. Espaco e tempo sao perguntas separadas e
sao reportados separados.

VEREDITO DE ESCALA ESPACIAL, com os limiares declarados aqui antes de rodar:

  GRUPO_E_O_PONTO            o grupo tem ~1 ponto (Recife, Curitiba). "Dentro do
                             grupo" nao existe por construcao da unidade de
                             validacao, nao por falta de variacao da chuva.
  GRUPOS_PUROS               os grupos tem varios pontos mas nenhum tem as duas
                             classes (piloto ingles: evento so-positivo, bloco
                             so-negativo). Tambem nao ha contraste interno.
  RUIDO_NA_ESCALA_DO_MODELO  ha grupo com as duas classes e chuva variando, mas
                             a AUC dentro do grupo fica em [0,45; 0,55] -- a
                             faixa que o projeto ja trata como acaso.
  SINAL_NA_ESCALA_DO_MODELO  ha contraste interno e a AUC sai dessa faixa.

VEREDITO DE CONTRASTE TEMPORAL:

  DATAS_DISJUNTAS            menos de 20% dos pontos estao em datas que tem as
                             duas classes. Comparar chuva entre positivo e
                             negativo passa a ser comparar dias, nao lugares. O
                             20% nao e magico: e o ponto abaixo do qual a AUC
                             restrita a datas compartilhadas cai para amostra
                             pequena demais para sustentar leitura.
  DATAS_COMPARTILHADAS       existe massa de "mesmo dia, lugares diferentes", e
                             a AUC restrita a essas datas e interpretavel.

O que este script NAO faz: nao remove variavel, nao reajusta modelo, nao decide
politica. Ele mede e nomeia. A decisao sobre o que fazer com cada veredito e da
Gabriela, e esta escrita em `ext_chuva_estado_do_projeto_v1.md`.

Uso:
    python scripts/suscetibilidade/aud_chuva02_escala_do_contraste.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds03_esquema_alvo import VERSAO  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
DATASET = RUNS / "ds-05-tabela-unica" / f"tabela_unica_{VERSAO}.csv"
OUT = RUNS / "aud-chuva-02"

VARIAVEIS = ("rain_max_24h", "rain_decay_index")
GRADE_GRAUS = 0.1          # celula da aquisicao (chuva04); ~11 km
PCT_DATAS_COMPARTILHADAS = 20.0
FAIXA_ACASO = (0.45, 0.55)
SEMENTE = 20260820


def auc(y: np.ndarray, x: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, x))


def escala_espacial(s: pd.DataFrame, var: str) -> dict:
    cel = (s.lat.round(1).astype(str) + "_" + s.lon.round(1).astype(str))
    return {
        "celulas": int(cel.nunique()),
        "celula_dia": int((cel + "|" + s.dia).nunique()),
        "valores_distintos": int(s[var].nunique()),
        "pontos_por_valor": round(len(s) / max(s[var].nunique(), 1), 1),
        "valores_distintos_por_data_mediana": float(
            s.groupby("dia")[var].nunique().median()),
    }


def decomposicao(s: pd.DataFrame, var: str) -> dict:
    """Quanto da variancia esta entre grupos e quanto dentro deles."""
    total = float(s[var].var())
    if not total or np.isnan(total):
        return {"var_total": total, "pct_entre_grupos": None}
    dentro = float((s[var] - s.groupby("grupo_cv")[var].transform("mean")).var())
    return {"var_total": round(total, 3),
            "pct_entre_grupos": round(100 * (1 - dentro / total), 1)}


def dentro_do_grupo(s: pd.DataFrame, var: str) -> dict:
    """AUC entre pontos do mesmo grupo -- a escala em que o modelo compara."""
    aucs, constantes, usaveis = [], 0, 0
    for _, g in s.groupby("grupo_cv"):
        if g.classe.nunique() < 2:
            continue
        if g[var].nunique() < 2:
            constantes += 1
            continue
        usaveis += 1
        aucs.append(auc(g.classe.to_numpy(), g[var].to_numpy()))
    return {
        "grupos_com_duas_classes": usaveis + constantes,
        "grupos_com_chuva_constante": constantes,
        "grupos_usaveis": usaveis,
        "auc_media_dentro_do_grupo": round(float(np.mean(aucs)), 4) if aucs else None,
    }


def contraste_temporal(s: pd.DataFrame, var: str) -> dict:
    dp = set(s.loc[s.classe == 1, "dia"])
    dn = set(s.loc[s.classe == 0, "dia"])
    comum = dp & dn
    sub = s[s.dia.isin(comum)]
    pct = 100 * len(sub) / len(s)
    d = {
        "datas_de_positivo": len(dp), "datas_de_negativo": len(dn),
        "datas_compartilhadas": len(comum),
        "pontos_em_datas_compartilhadas": int(len(sub)),
        "pct_pontos_em_datas_compartilhadas": round(pct, 1),
        "auc_global": round(auc(s.classe.to_numpy(), s[var].to_numpy()), 4),
        "auc_em_datas_compartilhadas": None,
        "n_em_datas_compartilhadas": int(len(sub)),
    }
    if len(sub) and sub.classe.nunique() == 2:
        d["auc_em_datas_compartilhadas"] = round(
            auc(sub.classe.to_numpy(), sub[var].to_numpy()), 4)
    return d


def veredito_espacial(dentro: dict, pontos_por_grupo: float) -> str:
    """Ha contraste na escala em que o modelo compara, e ele diz alguma coisa?"""
    if dentro["grupos_usaveis"] == 0:
        return "GRUPO_E_O_PONTO" if pontos_por_grupo < 1.5 else "GRUPOS_PUROS"
    a = dentro["auc_media_dentro_do_grupo"]
    if a is None or FAIXA_ACASO[0] <= a <= FAIXA_ACASO[1]:
        return "RUIDO_NA_ESCALA_DO_MODELO"
    return "SINAL_NA_ESCALA_DO_MODELO"


def veredito_temporal(temporal: dict) -> str:
    """Positivo e negativo dividem datas, ou a comparacao e entre dias?"""
    if temporal["pct_pontos_em_datas_compartilhadas"] < PCT_DATAS_COMPARTILHADAS:
        return "DATAS_DISJUNTAS"
    return "DATAS_COMPARTILHADAS"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===AUD-CHUVA-02: ESCALA DA CHUVA x ESCALA DO CONTRASTE===")

    if not DATASET.exists():
        print(f"ABORTADO: {DATASET} ausente. Gere com: "
              "python scripts/suscetibilidade/ds05_admissao_consolidacao.py")
        return 1

    d = pd.read_csv(DATASET, low_memory=False)
    d = d[(d.admitido == True) & d.classe.isin([0, 1])].copy()  # noqa: E712
    d["dia"] = d.data_evento.astype(str).str.slice(0, 10)

    resultados, linhas = {}, []
    for fonte, s0 in d.groupby("fonte"):
        resultados[fonte] = {
            "n": int(len(s0)),
            "fonte_chuva": sorted(s0.fonte_chuva.dropna().unique().tolist()),
            "grupos": int(s0.grupo_cv.nunique()),
            "pontos_por_grupo": round(len(s0) / s0.grupo_cv.nunique(), 1),
            "variaveis": {},
        }
        print(f"\n=== {fonte} (n={len(s0):,}, grupos={s0.grupo_cv.nunique():,}, "
              f"fonte_chuva={resultados[fonte]['fonte_chuva']}) ===")
        for var in VARIAVEIS:
            s = s0.dropna(subset=[var, "dia"])
            if s.empty or s.classe.nunique() < 2:
                resultados[fonte]["variaveis"][var] = {
                    "veredito": "SEM_DADO_UTILIZAVEL", "n": int(len(s))}
                print(f"  {var}: sem dado utilizavel")
                continue
            esp = escala_espacial(s, var)
            dec = decomposicao(s, var)
            den = dentro_do_grupo(s, var)
            tmp = contraste_temporal(s, var)
            v = veredito_espacial(den, resultados[fonte]["pontos_por_grupo"])
            vt = veredito_temporal(tmp)
            resultados[fonte]["variaveis"][var] = {
                "n": int(len(s)), "escala_espacial": esp, "decomposicao": dec,
                "dentro_do_grupo": den, "contraste_temporal": tmp,
                "veredito_espacial": v, "veredito_temporal": vt,
            }
            linhas.append({
                "fonte": fonte, "variavel": var, "n": len(s),
                "celulas_0_1_grau": esp["celulas"],
                "valores_distintos": esp["valores_distintos"],
                "pct_var_entre_grupos": dec["pct_entre_grupos"],
                "grupos_usaveis": den["grupos_usaveis"],
                "auc_dentro_do_grupo": den["auc_media_dentro_do_grupo"],
                "pct_datas_compartilhadas": tmp["pct_pontos_em_datas_compartilhadas"],
                "auc_global": tmp["auc_global"],
                "auc_datas_compartilhadas": tmp["auc_em_datas_compartilhadas"],
                "veredito_espacial": v, "veredito_temporal": vt,
            })
            print(f"  {var}")
            print(f"    escala: {esp['celulas']} celulas de {GRADE_GRAUS} grau, "
                  f"{esp['valores_distintos']} valores distintos "
                  f"({esp['pontos_por_valor']} pontos por valor)")
            print(f"    variancia entre grupos: {dec['pct_entre_grupos']}%")
            print(f"    dentro do grupo: {den['grupos_usaveis']} utilizaveis de "
                  f"{den['grupos_com_duas_classes']} com duas classes "
                  f"({den['grupos_com_chuva_constante']} com chuva constante) "
                  f"-> AUC {den['auc_media_dentro_do_grupo']}")
            print(f"    datas compartilhadas: {tmp['datas_compartilhadas']} "
                  f"({tmp['pct_pontos_em_datas_compartilhadas']}% dos pontos) "
                  f"-> AUC global {tmp['auc_global']}, "
                  f"restrita {tmp['auc_em_datas_compartilhadas']}")
            print(f"    VEREDITO espacial={v}  temporal={vt}")

    tab = pd.DataFrame(linhas)
    tab.to_csv(OUT / "escala_do_contraste_por_fonte.csv", index=False)

    contagem = {"espacial": tab.veredito_espacial.value_counts().to_dict(),
                "temporal": tab.veredito_temporal.value_counts().to_dict()}
    print("\n--- VEREDITO GLOBAL (pares fonte x variavel) ---")
    for eixo, dic in contagem.items():
        print(f"  {eixo}:")
        for k, n in dic.items():
            print(f"    {k}: {n} de {len(tab)}")
    com_sinal = int((tab.veredito_espacial == "SINAL_NA_ESCALA_DO_MODELO").sum())
    print(f"\n  pares com sinal na escala do modelo: {com_sinal} de {len(tab)}")

    (OUT / "auditoria_escala_chuva.json").write_text(json.dumps({
        "entrada": str(DATASET.relative_to(REPO)).replace("\\", "/"),
        "grade_graus": GRADE_GRAUS,
        "limiares": {
            "pct_datas_compartilhadas": PCT_DATAS_COMPARTILHADAS,
            "faixa_acaso": list(FAIXA_ACASO),
        },
        "semente": SEMENTE,
        "fontes": resultados,
        "contagem_de_vereditos": contagem,
        "regra": ("chuva so pode ser interpretada como preditor de LUGAR se "
                  "variar e discriminar na escala em que o modelo compara. "
                  "Discriminacao que so existe entre datas e afirmacao sobre "
                  "quando, nao sobre onde"),
        "nao_e": ("nao e decisao de remover variavel nem reajuste de modelo: "
                  "e medida de escala"),
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    (OUT / "commands.txt").write_text(
        "python scripts/suscetibilidade/ds05_admissao_consolidacao.py\n"
        "python scripts/suscetibilidade/aud_chuva01_fontes_incompativeis.py\n"
        "python scripts/suscetibilidade/aud_chuva02_escala_do_contraste.py\n"
        "python -m pytest tests/test_aud_chuva02_escala.py -q\n",
        encoding="utf-8")

    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
